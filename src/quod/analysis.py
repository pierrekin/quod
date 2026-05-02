"""Analyses that derive claims from the graph.

Today: literal-only interprocedural range propagation. For each callee
parameter, if every call site passes an IntLit at that position, derive
an IntRangeClaim(regime=lattice) covering [min, max] of those literals.
Any non-literal arg poisons the parameter (no claim emitted).

The analysis is deliberately limited — no fixpoint, no expression evaluation,
no flow sensitivity. It's the smallest move that makes regime=lattice mean
something concrete and exercises the override flag (--enforce-lattice).
"""

from __future__ import annotations

from collections.abc import Iterator

from quod.hashing import node_hash
from quod.model import (
    Assign,
    BinOp,
    Call,
    Claim,
    DerivedJustification,
    ExprStmt,
    FieldRead,
    FieldSet,
    For,
    If,
    IntLit,
    IntRangeClaim,
    Let,
    Load,
    Program,
    PtrOffset,
    Widen,
    ReturnExpr,
    ShortCircuitAnd,
    ShortCircuitOr,
    StructInit,
    While,
    WithArena,
)


_ANALYSIS_NAME = "literal_range_propagation"


def derive_lattice_claims(program: Program) -> dict[str, tuple[Claim, ...]]:
    """Return derived (regime=lattice) claims keyed by function name."""
    # arg_buckets[fn_name][param_index]: list of (literal_value, source_call_hash),
    # or None once poisoned by a non-literal arg or a type-mismatching literal.
    defined_fns = {fn.name: fn for fn in program.functions}
    arg_buckets: dict[str, list[list[tuple[int, str]] | None]] = {
        fn.name: [[] for _ in fn.params] for fn in program.functions
    }

    for caller in program.functions:
        for stmt in caller.body:
            for call in _walk_calls_in_stmt(stmt):
                callee = defined_fns.get(call.function)
                if callee is None:
                    continue  # extern or dangling; lower-time error
                buckets = arg_buckets[callee.name]
                call_h = node_hash(call)
                for i, arg in enumerate(call.args):
                    if i >= len(buckets) or buckets[i] is None:
                        continue
                    # Only literals of the matching param type contribute. A
                    # type-mismatching call would fail to lower anyway, but
                    # poisoning here keeps the derived bound honest.
                    if isinstance(arg, IntLit) and arg.type == callee.params[i].type:
                        buckets[i].append((arg.value, call_h))
                    else:
                        buckets[i] = None

    result: dict[str, tuple[Claim, ...]] = {}
    for fn in program.functions:
        derived: list[Claim] = []
        buckets = arg_buckets[fn.name]
        for i, p in enumerate(fn.params):
            entries = buckets[i]
            if entries is None or not entries:
                continue
            literals = [v for v, _ in entries]
            input_hashes = tuple(h for _, h in entries)
            lo, hi = min(literals), max(literals)
            derived.append(IntRangeClaim(
                regime="lattice",
                param=p.name,
                min=lo,
                max=hi,
                justification=DerivedJustification(
                    analysis=_ANALYSIS_NAME,
                    inputs=input_hashes,
                    note=f"literal range over {len(literals)} call site(s)",
                ),
            ))
        if derived:
            result[fn.name] = tuple(derived)
    return result


def elaborate(program: Program, derived: dict[str, tuple[Claim, ...]]) -> Program:
    """Return a Program with `derived` claims appended to each function's claims."""
    new_fns = tuple(
        fn.model_copy(update={"claims": fn.claims + derived.get(fn.name, ())})
        for fn in program.functions
    )
    return program.model_copy(update={"functions": new_fns})


# ---------- Call-walking helpers ----------

def _walk_calls_in_stmt(stmt) -> Iterator[Call]:
    match stmt:
        case ReturnExpr(value=expr) | ExprStmt(value=expr):
            yield from _walk_calls_in_expr(expr)
        case If(cond=cond, then_body=t_body, else_body=e_body):
            yield from _walk_calls_in_expr(cond)
            for s in t_body:
                yield from _walk_calls_in_stmt(s)
            for s in e_body:
                yield from _walk_calls_in_stmt(s)
        case Let(init=expr) | Assign(value=expr) | FieldSet(value=expr):
            yield from _walk_calls_in_expr(expr)
        case While(cond=cond, body=body):
            yield from _walk_calls_in_expr(cond)
            for s in body:
                yield from _walk_calls_in_stmt(s)
        case For(lo=lo, hi=hi, body=body):
            yield from _walk_calls_in_expr(lo)
            yield from _walk_calls_in_expr(hi)
            for s in body:
                yield from _walk_calls_in_stmt(s)
        case WithArena(capacity=cap, body=body):
            yield from _walk_calls_in_expr(cap)
            for s in body:
                yield from _walk_calls_in_stmt(s)


def _walk_calls_in_expr(expr) -> Iterator[Call]:
    match expr:
        case Call() as call:
            yield call
            for a in call.args:
                yield from _walk_calls_in_expr(a)
        case BinOp(lhs=l, rhs=r) | ShortCircuitOr(lhs=l, rhs=r) | ShortCircuitAnd(lhs=l, rhs=r):
            yield from _walk_calls_in_expr(l)
            yield from _walk_calls_in_expr(r)
        case FieldRead(value=inner):
            yield from _walk_calls_in_expr(inner)
        case StructInit(fields=field_inits):
            for fi in field_inits:
                yield from _walk_calls_in_expr(fi.value)
        case PtrOffset(base=b, offset=o):
            yield from _walk_calls_in_expr(b)
            yield from _walk_calls_in_expr(o)
        case Widen(value=v):
            yield from _walk_calls_in_expr(v)
        case Load(ptr=p):
            yield from _walk_calls_in_expr(p)
        # IntLit, ParamRef, LocalRef, StringRef carry no nested Calls.
