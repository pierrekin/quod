"""Term-rewriter normal form for predicate-shaped expressions.

Canonicalization exists for **hash stability**, not UX. Two
semantically-equal predicates must produce identical bytes so that
proof-pinning, dedup, and equivalence-edge detection all converge on
the same node identity.

The rules implemented here are the minimum that makes hashing total
over the surface forms we expect:

    1. Comparator direction. `sgt(a, b)` rewrites to `slt(b, a)`;
       `sge(a, b)` to `sle(b, a)`; same for unsigned. Authors who
       write `x >= 0` and lattice analysis emitting `0 <= x` collapse
       to one form.

    2. Commutative-op operand order. For commutative `BinOp` ops
       (`add`, `mul`, `or`, `and`, `xor`, `eq`, `ne`), operands are
       sorted by their canonical JSON bytes. `and(x, y)` and
       `and(y, x)` collapse.

    3. Short-circuit collapse. `ShortCircuitOr` / `ShortCircuitAnd`
       rewrite to `BinOp("or", ...)` / `BinOp("and", ...)`. Predicates
       are required to be side-effect-free (validator gates Calls), so
       the short-circuit semantics are degenerate — collapsing to
       commutative bitwise lets the operand-order rule fire.

    4. Trivial Boolean folds: `not(not(x)) -> x`,
       `not(true) -> false`, `and(x, true) -> x`,
       `and(x, false) -> false`, `or(x, true) -> true`,
       `or(x, false) -> x`, `xor(x, false) -> x`.

What is NOT implemented (deliberately): algebraic identities like
`x + 0 -> x`, range merging like `x >= 0 and x >= 5 -> x >= 5`,
constant evaluation past trivial bool folds. These are outside the
"surface forms we expect to hash equal" set and would expand the rule
table without obvious payoff. Add them if dedup misses appear.

The output is a normal form: `canonicalize(canonicalize(e)) ==
canonicalize(e)` (idempotence is verified by tests).
"""

from __future__ import annotations

from quod.model import (
    BinOp,
    Expr,
    I1Type,
    IfExpr,
    IntLit,
    IntType,
    Not,
    ParamRef,
    ReturnRef,
    ShortCircuitAnd,
    ShortCircuitOr,
)


# Commutative `BinOp` ops. Operands are sorted by JSON bytes after recursion.
_COMMUTATIVE_BINOPS: frozenset[str] = frozenset({
    "add", "mul", "or", "and", "xor", "eq", "ne",
})

# Comparator pairs whose direction is reversed by swapping operands.
# `BinOp(sgt, a, b)` is rewritten to `BinOp(slt, b, a)`; same shape for
# the other entries.
_SWAP_COMPARATOR: dict[str, str] = {
    "sgt": "slt",
    "sge": "sle",
    "ugt": "ult",
    "uge": "ule",
}


def canonicalize(expr: Expr) -> Expr:
    """Return the canonical form of `expr`.

    Safe to call on any `Expr`. Nodes outside the predicate vocabulary
    (Call, FieldRead, etc.) recurse structurally where they have child
    expressions and pass through unchanged otherwise.
    """
    expr = _canonicalize_children(expr)
    expr = _normalize_comparator(expr)
    expr = _collapse_short_circuit(expr)
    expr = _fold_trivial(expr)
    expr = _sort_commutative(expr)
    return expr


def _canonicalize_children(expr: Expr) -> Expr:
    """Recurse through known-shape parents; leave unknown leaves alone."""
    if isinstance(expr, BinOp):
        return expr.model_copy(update={
            "lhs": canonicalize(expr.lhs),
            "rhs": canonicalize(expr.rhs),
        })
    if isinstance(expr, ShortCircuitOr):
        return expr.model_copy(update={
            "lhs": canonicalize(expr.lhs),
            "rhs": canonicalize(expr.rhs),
        })
    if isinstance(expr, ShortCircuitAnd):
        return expr.model_copy(update={
            "lhs": canonicalize(expr.lhs),
            "rhs": canonicalize(expr.rhs),
        })
    if isinstance(expr, Not):
        return expr.model_copy(update={"operand": canonicalize(expr.operand)})
    if isinstance(expr, IfExpr):
        return expr.model_copy(update={
            "cond": canonicalize(expr.cond),
            "then_value": canonicalize(expr.then_value),
            "else_value": canonicalize(expr.else_value),
        })
    return expr


def _normalize_comparator(expr: Expr) -> Expr:
    if isinstance(expr, BinOp) and expr.op in _SWAP_COMPARATOR:
        return BinOp(op=_SWAP_COMPARATOR[expr.op], lhs=expr.rhs, rhs=expr.lhs)
    return expr


def _collapse_short_circuit(expr: Expr) -> Expr:
    # In a predicate, side-effect-free operands make sc-or/sc-and
    # equivalent to bitwise i1 or/and, which fall under the commutative
    # operand-order rule.
    if isinstance(expr, ShortCircuitOr):
        return BinOp(op="or", lhs=expr.lhs, rhs=expr.rhs)
    if isinstance(expr, ShortCircuitAnd):
        return BinOp(op="and", lhs=expr.lhs, rhs=expr.rhs)
    return expr


def _fold_trivial(expr: Expr) -> Expr:
    if isinstance(expr, Not):
        if isinstance(expr.operand, Not):
            return expr.operand.operand
        if _is_bool_lit(expr.operand):
            return IntLit(type=I1Type(), value=1 - expr.operand.value)
        return expr
    if isinstance(expr, BinOp) and expr.op in ("and", "or", "xor"):
        if _is_bool_lit(expr.lhs):
            folded = _fold_bool_binop(expr.op, expr.lhs.value, expr.rhs)
            if folded is not None:
                return folded
        if _is_bool_lit(expr.rhs):
            folded = _fold_bool_binop(expr.op, expr.rhs.value, expr.lhs)
            if folded is not None:
                return folded
    return expr


def _fold_bool_binop(op: str, const_val: int, other: Expr) -> Expr | None:
    """Fold a boolean op with one i1 literal operand. Returns None if no
    fold applies. `const_val` is 0 or 1; `other` is the non-literal side."""
    if op == "and":
        return other if const_val == 1 else IntLit(type=I1Type(), value=0)
    if op == "or":
        return IntLit(type=I1Type(), value=1) if const_val == 1 else other
    if op == "xor":
        # xor(x, false) -> x. xor(x, true) -> not(x), but rewriting that
        # would create churn with the lower-Not-as-xor pattern; leave it.
        return other if const_val == 0 else None
    return None


def _is_bool_lit(expr: Expr) -> bool:
    return (
        isinstance(expr, IntLit)
        and isinstance(expr.type, I1Type)
        and expr.value in (0, 1)
    )


def _sort_commutative(expr: Expr) -> Expr:
    if isinstance(expr, BinOp) and expr.op in _COMMUTATIVE_BINOPS:
        if _sort_key(expr.rhs) < _sort_key(expr.lhs):
            return BinOp(op=expr.op, lhs=expr.rhs, rhs=expr.lhs)
    return expr


def _sort_key(expr: Expr) -> str:
    """Stable canonical-bytes sort key. After children are canonicalized,
    this is deterministic — Pydantic emits fields in declaration order."""
    return expr.model_dump_json()


# ---------- Sugar-shape predicate builders ----------
#
# Build canonical predicates corresponding to the named claim shapes
# (`non_negative`, `int_range`, `return_in_range`). Used by the CLI to
# desugar user input and by the lattice analysis to emit derived
# claims. The render-side recognizer is the inverse.

# CLI vocabulary for the named sugar shapes. Useful for argument parsing
# and autocomplete; the sugar names are only ever a CLI-surface concern,
# never stored in the graph.
SUGAR_KINDS: tuple[str, ...] = ("non_negative", "int_range", "return_in_range")
PARAM_SUGAR_KINDS: tuple[str, ...] = ("non_negative", "int_range")
RETURN_SUGAR_KINDS: tuple[str, ...] = ("return_in_range",)


def predicate_for_param_range(
    param: str, ty: IntType,
    lo: int | None, hi: int | None,
) -> Expr:
    """Canonical predicate for `int_range(param, [lo, hi])` /
    `non_negative(param)` over a function parameter.

    `ty` is the param's declared type — IntLit bounds are pinned to
    the same type so the predicate is well-typed at lower time.
    """
    return _range_predicate(ParamRef(name=param), ty, lo, hi)


def predicate_for_return_range(
    ty: IntType, lo: int | None, hi: int | None,
) -> Expr:
    """Canonical predicate for `return_in_range([lo, hi])`.

    `ty` is the function's return type.
    """
    return _range_predicate(ReturnRef(), ty, lo, hi)


def _range_predicate(
    ref: Expr, ty: IntType, lo: int | None, hi: int | None,
) -> Expr:
    if lo is None and hi is None:
        raise ValueError("range predicate requires at least one of lo/hi")
    parts: list[Expr] = []
    if lo is not None:
        parts.append(BinOp(op="sle", lhs=IntLit(type=ty, value=lo), rhs=ref))
    if hi is not None:
        parts.append(BinOp(op="sle", lhs=ref, rhs=IntLit(type=ty, value=hi)))
    if len(parts) == 1:
        return canonicalize(parts[0])
    return canonicalize(BinOp(op="and", lhs=parts[0], rhs=parts[1]))
