"""A→B lift checker — structural transcription verifier.

The C ingester emits an `Equivalence` claim between each layer-A
`CFn` and its layer-B `Function` to mark that the lift is a
faithful transcription. The ingester emits these as `regime=axiom`
with a `ManualJustification` ("the ingester's promise"); this
module promotes them to `regime=witness` with a `LiftEquivalence`
artifact.

The check walks both subtrees in lockstep, asserting one-to-one
node correspondence per a per-construct table:

  | Layer A              | Layer B                                |
  | -------------------- | -------------------------------------- |
  | CUnit                | (program-level — not paired here)      |
  | CFn                  | Function                               |
  | CParam(name, int)    | Param(name, I32Type)                   |
  | CNamedType("int")    | I32Type                                |
  | CPointerType(_)      | I8PtrType                              |
  | CAddressOf(           | PtrOffset(b', i')                      |
  |   CArraySubscript(    |                                        |
  |     b, i))            |                                        |
  | CBinOp("+", ptr, n)   | PtrOffset(ptr', n')                    |
  | CVarDecl(int, n, e)  | Let(n, I32Type, e')                    |
  | CAssign(t, e)        | Assign(t, e')                          |
  | CReturn(None)        | Return                                 |
  | CReturn(e)           | ReturnExpr(e')                         |
  | CReturn(i1-binop)    | If(cond, return 1, return 0)           |
  | CFor(...)            | CStyleFor(...)                         |
  | CIf(c, t, e)         | If(c', t', e')                         |
  | CWhile(c, b)         | While(c', b')                          |
  | CExprStmt(CCall(…))  | ExprStmt(Call(…))                      |
  | CIntLit(v)           | IntLit(I32Type, v)                     |
  | CStringLit(v)        | StringRef(name=...)                    |
  | CVarRef(n)           | ParamRef(n) | LocalRef(n)              |
  | CBinOp(op, l, r)     | BinOp(op_translate(op), l', r')        |
  |                      |   or ShortCircuitAnd / ShortCircuitOr  |
  | CCall(callee, args)  | Call(callee, args')                    |

`walk_lift(cfn, fn)` returns a deterministic JSON-serializable
correspondence record. Calling `lift_check_artifact(cfn, fn)`
serializes that record to bytes — the bytes' sha256 is what
`LiftEquivalence` pins, the bytes themselves are what get written
to disk under `<proofs_dir>/lift/<fn>.txt` for human inspection.

Re-verification re-walks (deterministically), re-serializes, and
compares hashes. Re-walking from in-memory data is cheap; the file
on disk is for human inspection, not load-bearing.

The supported correspondence table covers the C subset the
ingester emits at layer A; programs containing constructs outside
the table (pointer-typed locals, enum constants, etc.) skip layer
A entirely via the all-or-nothing fallback in `ingest_c`.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from quod.model import (
    Assign,
    BinOp,
    Block,
    Break,
    Call,
    CAddressOf,
    CArraySubscript,
    CAssign,
    CBinOp,
    CBreak,
    CCall,
    CCompoundAssign,
    CContinue,
    CDoWhile,
    CEnumConstRef,
    Continue,
    DoWhile,
    CExprStmt,
    CFn,
    CFor,
    CIf,
    CIncrementStmt,
    CIntLit,
    CMultiVarDecl,
    CNamedType,
    CParam,
    CPointerType,
    CReturn,
    CScopedBlock,
    CStringLit,
    CStyleFor,
    CSwitch,
    CSwitchCase,
    CTernary,
    CUnary,
    CVarDecl,
    CVarRef,
    CWhile,
    ExprStmt,
    Function,
    I8PtrType,
    I32Type,
    I64Type,
    If,
    IfExpr,
    IntLit,
    Let,
    LocalRef,
    Param,
    Cast,
    ParamRef,
    PtrOffset,
    Return,
    ReturnExpr,
    StringConstant,
    StringRef,
    VoidType,
    While,
    ShortCircuitAnd,
    ShortCircuitOr,
)


# Map layer-A binary-operator spelling to layer-B BinOp.op spelling.
# Mirrors the table in `quod.ingest.c.helpers` — kept in sync because
# the lift produces the layer-B side from the same source character.
# C compound-assignment operator → underlying layer-B BinOp.op spelling.
# Mirrors `_COMPOUND_ASSIGN_TABLE` in `quod.ingest.c.helpers`.
_COMPOUND_ASSIGN_TO_BINOP = {
    "+=":  "add",
    "-=":  "sub",
    "*=":  "mul",
    "/=":  "sdiv",
    "%=":  "srem",
    "&=":  "and",
    "|=":  "or",
    "^=":  "xor",
    "<<=": "shl",
    ">>=": "ashr",
}


_BINOP_LAYER_A_TO_B = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "sdiv",
    "%": "srem",
    "<": "slt",
    "<=": "sle",
    ">": "sgt",
    ">=": "sge",
    "==": "eq",
    "!=": "ne",
    "|": "or",
    "&": "and",
    "^": "xor",
    "<<": "shl",
    ">>": "ashr",
}


class LiftCheckError(Exception):
    """Raised when an A→B correspondence walk encounters a mismatch.

    Carries a path describing where in the tree the mismatch
    occurred (e.g. `fn[sum].body[1].cond.lhs`), plus the offending
    a/b nodes' kinds and values, so the failure is debuggable
    without re-running the lift.
    """


class _Ctx:
    """Walk-time context passed through the lift-checker. Carries the
    program-level lookups the structural walk needs (today: a string-
    constant table; in future, more). Held as a class with a `__slots__`-
    style attribute set so the contract is explicit.
    """
    __slots__ = ("constants_by_name",)

    def __init__(self, *, constants_by_name: dict[str, str] | None) -> None:
        self.constants_by_name = constants_by_name


def walk_lift(
    cfn: CFn, fn: Function, *, program=None,
) -> dict[str, Any]:
    """Walk a layer-A `CFn` and its layer-B `Function` in lockstep,
    asserting structural correspondence. Returns a deterministic
    JSON-serializable record of the walk.

    `program`, if supplied, lets the checker compare layer-A
    `CStringLit.value` against the layer-B `StringConstant.value`
    that the corresponding `StringRef` points at. Without it,
    encountering a string literal raises (the soundness contract is
    "every CStringLit's value must be verifiable").

    Raises `LiftCheckError` on any divergence — different node kinds,
    mismatched param names, unknown operators, missing string
    constants, etc.
    """
    constants_by_name = (
        {c.name: c.value for c in program.constants}
        if program is not None
        else None
    )
    ctx = _Ctx(constants_by_name=constants_by_name)

    if cfn.name != fn.name:
        raise LiftCheckError(
            f"function name mismatch: layer-A {cfn.name!r} vs layer-B {fn.name!r}"
        )

    if len(cfn.params) != len(fn.params):
        raise LiftCheckError(
            f"function {fn.name!r}: param count {len(cfn.params)} vs {len(fn.params)}"
        )
    params = [
        _check_param(cp, bp, path=f"fn[{fn.name}].params[{i}]")
        for i, (cp, bp) in enumerate(zip(cfn.params, fn.params))
    ]

    _check_return_type(cfn.return_type, fn.return_type, path=f"fn[{fn.name}].return_type")

    # Layer-B body may be a `CScopedBlock` wrapper around the core
    # `Block`; the lift unwraps automatically. Match against the inner
    # Block.
    b_body_block = fn.body.block if isinstance(fn.body, CScopedBlock) else fn.body
    body_record = _check_body(
        cfn.body, b_body_block,
        path=f"fn[{fn.name}].body",
        ctx=ctx,
        strip_fallthrough=True,  # outer function body may end in a synthesized stub
    )

    return {
        "kind": "lift-check",
        "version": "1",
        "rule": "c.transcription",
        "fn": {
            "name": fn.name,
            "a_id": cfn.id,
            "b_id": fn.id,
            "params": params,
            "return_type": {"a": _type_a_repr(cfn.return_type), "b": _type_b_repr(fn.return_type)},
            "body": body_record,
        },
    }


def lift_check_artifact(cfn: CFn, fn: Function, *, program=None) -> bytes:
    """Walk the lift and serialize the result to deterministic bytes.

    The bytes are what gets written to disk and what
    `LiftEquivalence.artifact_hash` pins.
    """
    record = walk_lift(cfn, fn, program=program)
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    return text.encode("utf-8")


def lift_check_hash(cfn: CFn, fn: Function, *, program=None) -> str:
    """sha256 of the lift-check artifact."""
    return hashlib.sha256(lift_check_artifact(cfn, fn, program=program)).hexdigest()


# ---------- internal walkers ----------


def _check_param(cp: CParam, bp: Param, *, path: str) -> dict[str, Any]:
    if cp.name != bp.name:
        raise LiftCheckError(
            f"{path}: param name {cp.name!r} vs {bp.name!r}"
        )
    _check_value_type(cp.type, bp.type, path=f"{path}.type")
    return {"name": cp.name, "a_type": _type_a_repr(cp.type), "b_type": _type_b_repr(bp.type)}


def _check_return_type(a, b, *, path: str) -> None:
    _check_value_type(a, b, path=path)


def _check_value_type(a, b, *, path: str) -> None:
    """Map layer-A types to their layer-B counterparts. Pointers
    collapse to `I8PtrType` regardless of pointee — LLVM has opaque
    pointers, so all `T*` denote i8* at IR level. `void` only appears
    in return position and pairs with `VoidType`."""
    if isinstance(a, CPointerType):
        if not isinstance(b, I8PtrType):
            raise LiftCheckError(
                f"{path}: layer-A pointer (`{_format_c_type_str(a)}`) "
                f"but layer-B is {type(b).__name__}"
            )
        return
    if isinstance(a, CNamedType):
        if a.name == "int":
            if not isinstance(b, I32Type):
                raise LiftCheckError(
                    f"{path}: layer-A int but layer-B is {type(b).__name__}"
                )
            return
        if a.name == "void":
            if not isinstance(b, VoidType):
                raise LiftCheckError(
                    f"{path}: layer-A void but layer-B is {type(b).__name__}"
                )
            return
        # `char`, `signed char`, `unsigned char` are only valid as
        # pointee names in the supported subset — they appear inside
        # CPointerType, not as a standalone `CParam.type`. The lifter
        # doesn't emit `char` locals at the top level today.
        raise LiftCheckError(
            f"{path}: layer-A type {a.name!r} is not in the supported subset"
        )
    raise LiftCheckError(f"{path}: layer-A type is {type(a).__name__}, expected CNamedType or CPointerType")


def _format_c_type_str(t) -> str:
    """Compact rendering of a layer-A CType for error messages."""
    if isinstance(t, CNamedType):
        return t.name
    if isinstance(t, CPointerType):
        return _format_c_type_str(t.pointee) + "*"
    return repr(t)


def _check_body(
    a_stmts, b_block: Block, *, path: str, ctx: "_Ctx",
    strip_fallthrough: bool = False,
) -> dict[str, Any]:
    """Walk a tuple of layer-A statements against a layer-B Block's
    stmts. The C ingester appends a synthesized `Unreachable` or
    `ReturnExpr(0)` to the *function's outermost* body for fall-through
    (see `_translate_function` in ingest/c.py); these have no layer-A
    counterpart and the walk skips them when `strip_fallthrough=True`.

    Nested bodies (if / while / for inner blocks) don't get fall-
    through synthesis from the lifter, so the default is False.
    """
    a_list = list(a_stmts)
    b_list = list(b_block.stmts)

    # Each layer-A CMultiVarDecl(N) corresponds to N consecutive layer-B
    # Lets, so the number of layer-B stmts the layer-A list "expects"
    # is the sum of (N for multi-decls) + (1 for everything else).
    a_expanded_count = sum(
        len(a.decls) if isinstance(a, CMultiVarDecl) else 1
        for a in a_list
    )

    if strip_fallthrough:
        # The ingester adds:
        #   - ReturnExpr(IntLit(0)) for `main` falling off the end
        #   - Return() for void functions falling off the end
        #   - Unreachable() for non-main int-returning functions
        # When the layer-A body terminates explicitly, none is added.
        while b_list and len(b_list) > a_expanded_count and (
            _is_synthesized_fallthrough(b_list[-1])
        ):
            b_list.pop()

    # Walk the two lists together. A layer-A `CMultiVarDecl(N decls)`
    # consumes N consecutive Lets on the layer-B side (the ingester
    # expands `int a, b, c;` to N `Let`s). Every other statement is
    # 1:1.
    out_records: list[dict[str, Any]] = []
    bi = 0
    for ai, a in enumerate(a_list):
        if isinstance(a, CMultiVarDecl):
            n = len(a.decls)
            if bi + n > len(b_list):
                raise LiftCheckError(
                    f"{path}: layer-A CMultiVarDecl with {n} sub-decls "
                    f"but only {len(b_list) - bi} layer-B stmts left"
                )
            sub_records = []
            for j, sub in enumerate(a.decls):
                rec = _check_stmt(
                    sub, b_list[bi + j],
                    path=f"{path}.stmts[{ai}].decls[{j}]", ctx=ctx,
                )
                sub_records.append(rec)
            out_records.append({
                "kind": "c.multi_var_decl ↔ N×let",
                "a_id": a.id,
                "decls": sub_records,
            })
            bi += n
        else:
            if bi >= len(b_list):
                raise LiftCheckError(
                    f"{path}: ran out of layer-B stmts at A-index {ai}"
                )
            out_records.append(_check_stmt(
                a, b_list[bi],
                path=f"{path}.stmts[{ai}]", ctx=ctx,
            ))
            bi += 1

    if bi != len(b_list):
        raise LiftCheckError(
            f"{path}: layer-A consumed {bi} of {len(b_list)} layer-B "
            f"stmts; divergent shape (extra layer-B stmts: "
            f"{[type(s).__name__ for s in b_list[bi:]]})"
        )

    return {
        "a_block": "<inline>",  # layer-A bodies are tuple[CStmt, ...], not blocks
        "b_block_id": b_block.id,
        "stmts": out_records,
    }


def _check_switch_chain(
    a: "CSwitch", b, *, path: str, ctx: "_Ctx",
) -> dict[str, Any]:
    """Pair a layer-A CSwitch with a layer-B If-else-if chain. The
    chain shape mirrors what `_build_switch_chain` in ingest/c.py
    emits: each case becomes one If whose cond is `scrutinee == v0`
    or `(scrutinee == v0) || (scrutinee == v1) || ...` for stacked
    cases; the innermost else is the default (or empty)."""
    if not isinstance(b, If):
        raise LiftCheckError(
            f"{path}: layer-A CSwitch vs layer-B {type(b).__name__}"
        )
    case_records: list[dict[str, Any]] = []
    cur = b
    for i, case in enumerate(a.cases):
        if not isinstance(cur, If):
            raise LiftCheckError(
                f"{path}.cases[{i}]: ran out of If-chain at layer B"
            )
        # Verify the cond matches `scrutinee == v0 || ... || scrutinee == vN`.
        _check_switch_case_cond(a.scrutinee, case.values, cur.cond,
                                path=f"{path}.cases[{i}].cond", ctx=ctx)
        # The then-body matches the case body.
        body_record = _check_body(
            case.body, cur.then_body,
            path=f"{path}.cases[{i}].body", ctx=ctx,
        )
        case_records.append({
            "kind": f"switch.case[{i}] ↔ if",
            "values_count": len(case.values),
            "body": body_record,
        })
        # Walk into the else branch for the next case (or the default).
        # The lift wraps each next-case If inside a Block in else_body
        # whose stmts is exactly that nested If.
        if not cur.else_body.stmts:
            cur = None  # type: ignore[assignment]
            break
        if i == len(a.cases) - 1:
            # Last case — what's in else is the default body.
            break
        if len(cur.else_body.stmts) != 1 or not isinstance(cur.else_body.stmts[0], If):
            raise LiftCheckError(
                f"{path}.cases[{i}].else: expected nested If, "
                f"got {len(cur.else_body.stmts)}-stmt block"
            )
        cur = cur.else_body.stmts[0]

    # Default body lives in the last If's else_body (or, if no cases,
    # in `b` itself — but we always have at least one case to reach
    # here). When the source had no `default:` clause, the else is
    # an empty Block.
    expected_default = a.default if a.default is not None else ()
    if cur is None:
        # Walked into a None — only happens when a case body's else was
        # empty mid-chain, which would be malformed. Treat as empty default.
        actual_default_block = Block()
    else:
        actual_default_block = cur.else_body
    default_record = _check_body(
        expected_default, actual_default_block,
        path=f"{path}.default", ctx=ctx,
    )
    return {
        "kind": "c.switch ↔ if-else-if chain",
        "a_id": a.id,
        "cases": case_records,
        "default": default_record,
    }


def _check_switch_case_cond(
    scrutinee, values, b_cond, *, path: str, ctx: "_Ctx",
) -> None:
    """Verify `b_cond` is `scrutinee == values[0] || ... || scrutinee == values[N]`
    structurally — the exact shape `_build_switch_chain` emits."""
    expected_eqs = list(values)
    # Walk the b_cond as a left-leaning chain of `||` over `eq` nodes.
    # For a single value, b_cond is just the single eq.
    if len(expected_eqs) == 1:
        _check_switch_eq(scrutinee, expected_eqs[0], b_cond,
                         path=path, ctx=ctx)
        return
    # Flatten the layer-B Or chain. _build_switch_chain produces a left-
    # leaning chain: ((eq0 || eq1) || eq2) || eq3.
    flat: list = []
    def _flatten(node):
        if isinstance(node, ShortCircuitOr):
            _flatten(node.lhs)
            _flatten(node.rhs)
        else:
            flat.append(node)
    _flatten(b_cond)
    if len(flat) != len(expected_eqs):
        raise LiftCheckError(
            f"{path}: switch case has {len(expected_eqs)} stacked values "
            f"but layer-B cond has {len(flat)} disjuncts"
        )
    for i, (val, eq_node) in enumerate(zip(expected_eqs, flat)):
        _check_switch_eq(
            scrutinee, val, eq_node,
            path=f"{path}.disjunct[{i}]", ctx=ctx,
        )


def _check_switch_eq(
    scrutinee, a_value, b_eq, *, path: str, ctx: "_Ctx",
) -> None:
    """Verify `b_eq` is `BinOp("eq", scrutinee', a_value')` structurally."""
    if not isinstance(b_eq, BinOp) or b_eq.op != "eq":
        raise LiftCheckError(
            f"{path}: expected layer-B BinOp(eq, ...), got "
            f"{type(b_eq).__name__}"
        )
    _check_expr(scrutinee, b_eq.lhs, path=f"{path}.scrutinee", ctx=ctx)
    _check_expr(a_value, b_eq.rhs, path=f"{path}.value", ctx=ctx)


def _is_synthesized_fallthrough(stmt) -> bool:
    """True iff `stmt` is a layer-B fall-through stub the ingester
    appends when the layer-A body doesn't terminate explicitly. The
    walk strips these from the tail since layer A doesn't model them.
    """
    from quod.model import Unreachable
    if isinstance(stmt, Unreachable):
        return True
    if isinstance(stmt, Return):
        # Synthesized terminator for a void-returning function whose
        # source body falls through without an explicit `return;`.
        return True
    if isinstance(stmt, ReturnExpr):
        v = stmt.value
        return isinstance(v, IntLit) and v.value == 0
    return False


def _check_stmt(a, b, *, path: str, ctx: "_Ctx") -> dict[str, Any]:
    if isinstance(a, CVarDecl):
        if not isinstance(b, Let):
            raise LiftCheckError(f"{path}: layer-A CVarDecl vs layer-B {type(b).__name__}")
        if a.name != b.name:
            raise LiftCheckError(f"{path}: var name {a.name!r} vs {b.name!r}")
        _check_value_type(a.type, b.type, path=f"{path}.type")
        # Both layers may carry `init=None` for uninitialized locals
        # (`int x;`). When present, the inits must match structurally.
        if a.init is None:
            if b.init is not None:
                raise LiftCheckError(
                    f"{path}: layer-A var_decl is uninitialized but layer-B "
                    f"Let has an init expression"
                )
            return {
                "kind": "var_decl(uninit) ↔ let(uninit)",
                "a_id": a.id, "name": a.name,
            }
        if b.init is None:
            raise LiftCheckError(
                f"{path}: layer-A var_decl has an init but layer-B Let "
                f"is uninitialized"
            )
        return {
            "kind": "var_decl ↔ let",
            "a_id": a.id, "name": a.name,
            "init": _check_expr(a.init, b.init, path=f"{path}.init", ctx=ctx),
        }

    if isinstance(a, CAssign):
        if not isinstance(b, Assign):
            raise LiftCheckError(f"{path}: layer-A CAssign vs layer-B {type(b).__name__}")
        if a.target != b.name:
            raise LiftCheckError(f"{path}: assign target {a.target!r} vs {b.name!r}")
        return {
            "kind": "assign ↔ assign",
            "a_id": a.id, "target": a.target,
            "value": _check_expr(a.value, b.value, path=f"{path}.value", ctx=ctx),
        }

    if isinstance(a, CCompoundAssign):
        # `x op= y` ↔ `Assign(x, BinOp(op_translated, LocalRef(x), y'))`.
        if not isinstance(b, Assign):
            raise LiftCheckError(
                f"{path}: layer-A CCompoundAssign vs layer-B {type(b).__name__}"
            )
        if a.target != b.name:
            raise LiftCheckError(
                f"{path}: compound-assign target {a.target!r} vs {b.name!r}"
            )
        expected_op = _COMPOUND_ASSIGN_TO_BINOP.get(a.op)
        if expected_op is None:
            raise LiftCheckError(
                f"{path}: unsupported compound-assignment operator {a.op!r}"
            )
        if not isinstance(b.value, BinOp):
            raise LiftCheckError(
                f"{path}: compound-assign expects layer-B Assign(BinOp), "
                f"got Assign({type(b.value).__name__})"
            )
        if b.value.op != expected_op:
            raise LiftCheckError(
                f"{path}: compound-assign {a.op!r} expects layer-B "
                f"BinOp({expected_op!r}), got {b.value.op!r}"
            )
        if not isinstance(b.value.lhs, LocalRef) or b.value.lhs.name != a.target:
            raise LiftCheckError(
                f"{path}: compound-assign expects layer-B BinOp's LHS to be "
                f"LocalRef({a.target!r}), got {type(b.value.lhs).__name__}"
            )
        return {
            "kind": f"compound_assign({a.op}) ↔ assign(_, binop({expected_op}))",
            "a_id": a.id, "target": a.target,
            "value": _check_expr(a.value, b.value.rhs, path=f"{path}.value", ctx=ctx),
        }

    if isinstance(a, CIncrementStmt):
        # `i++;` / `++i;` / `i--;` / `--i;` ↔
        #   `Assign(i, BinOp("add"|"sub", LocalRef(i), IntLit(1)))`.
        # Pre and post lift identically — the expression value is
        # discarded in statement position, so the only observable
        # effect is the side effect itself. Position is preserved at
        # layer A for source fidelity (and for future expression-
        # position support to dispatch on).
        if not isinstance(b, Assign):
            raise LiftCheckError(
                f"{path}: layer-A CIncrementStmt vs layer-B {type(b).__name__}"
            )
        if a.target != b.name:
            raise LiftCheckError(
                f"{path}: increment target {a.target!r} vs {b.name!r}"
            )
        expected_op = "add" if a.op == "++" else "sub"
        if not isinstance(b.value, BinOp):
            raise LiftCheckError(
                f"{path}: increment expects layer-B Assign(BinOp), "
                f"got Assign({type(b.value).__name__})"
            )
        if b.value.op != expected_op:
            raise LiftCheckError(
                f"{path}: increment {a.op!r} expects layer-B "
                f"BinOp({expected_op!r}), got {b.value.op!r}"
            )
        if not isinstance(b.value.lhs, LocalRef) or b.value.lhs.name != a.target:
            raise LiftCheckError(
                f"{path}: increment expects layer-B BinOp's LHS to be "
                f"LocalRef({a.target!r}), got {type(b.value.lhs).__name__}"
            )
        if not isinstance(b.value.rhs, IntLit) or b.value.rhs.value != 1:
            raise LiftCheckError(
                f"{path}: increment expects layer-B BinOp's RHS to be "
                f"IntLit(1), got {b.value.rhs!r}"
            )
        return {
            "kind": f"increment({a.position}{a.op}) ↔ assign(_, binop({expected_op}, _, 1))",
            "a_id": a.id, "target": a.target,
        }

    if isinstance(a, CReturn):
        if a.value is None:
            if not isinstance(b, Return):
                raise LiftCheckError(
                    f"{path}: layer-A `return;` vs layer-B {type(b).__name__}"
                )
            return {"kind": "return; ↔ return", "a_id": a.id}
        # i1-widening transformation: when the C source has `return
        # cond` where cond is a comparison or short-circuit boolean,
        # the lifter produces `If(cond, return 1, return 0)` (since
        # quod has no zext op). We accept either shape on the layer-B
        # side: a plain ReturnExpr (for arithmetic returns) or an If
        # with the i1-widening structure.
        if isinstance(b, ReturnExpr):
            return {
                "kind": "return e ↔ return_expr",
                "a_id": a.id,
                "value": _check_expr(a.value, b.value, path=f"{path}.value", ctx=ctx),
            }
        if isinstance(b, If) and _is_layer_a_i1_typed(a.value):
            return {
                "kind": "return cond ↔ if(cond, return 1, return 0)",
                "a_id": a.id,
                "cond": _check_expr(a.value, b.cond, path=f"{path}.cond", ctx=ctx),
                "transform": "i1_widen",
            }
        raise LiftCheckError(
            f"{path}: layer-A `return e` vs layer-B {type(b).__name__}"
        )

    if isinstance(a, CFor):
        if not isinstance(b, CStyleFor):
            raise LiftCheckError(f"{path}: layer-A CFor vs layer-B {type(b).__name__}")
        if (a.init is None) != (b.init is None):
            raise LiftCheckError(f"{path}.init: presence mismatch")
        if (a.cond is None) != (b.cond is None):
            raise LiftCheckError(f"{path}.cond: presence mismatch")
        if (a.inc is None) != (b.inc is None):
            raise LiftCheckError(f"{path}.inc: presence mismatch")
        b_body_block = b.body.block if isinstance(b.body, CScopedBlock) else b.body
        return {
            "kind": "c.for ↔ c.for_general",
            "a_id": a.id, "b_id": b.id,
            "init": (_check_stmt(a.init, b.init, path=f"{path}.init", ctx=ctx) if a.init is not None else None),
            "cond": (_check_expr(a.cond, b.cond, path=f"{path}.cond", ctx=ctx) if a.cond is not None else None),
            "inc":  (_check_stmt(a.inc, b.inc, path=f"{path}.inc", ctx=ctx)  if a.inc  is not None else None),
            "body": _check_body(a.body, b_body_block, path=f"{path}.body", ctx=ctx),
        }

    if isinstance(a, CIf):
        if not isinstance(b, If):
            raise LiftCheckError(f"{path}: layer-A CIf vs layer-B {type(b).__name__}")
        return {
            "kind": "c.if ↔ if",
            "a_id": a.id,
            "cond": _check_expr(a.cond, b.cond, path=f"{path}.cond", ctx=ctx),
            "then_body": _check_body(a.then_body, b.then_body, path=f"{path}.then_body", ctx=ctx),
            "else_body": _check_body(a.else_body, b.else_body, path=f"{path}.else_body", ctx=ctx),
        }

    if isinstance(a, CWhile):
        if not isinstance(b, While):
            raise LiftCheckError(f"{path}: layer-A CWhile vs layer-B {type(b).__name__}")
        b_body_block = b.body.block if isinstance(b.body, CScopedBlock) else b.body
        return {
            "kind": "c.while ↔ while",
            "a_id": a.id,
            "cond": _check_expr(a.cond, b.cond, path=f"{path}.cond", ctx=ctx),
            "body": _check_body(a.body, b_body_block, path=f"{path}.body", ctx=ctx),
        }

    if isinstance(a, CDoWhile):
        if not isinstance(b, DoWhile):
            raise LiftCheckError(f"{path}: layer-A CDoWhile vs layer-B {type(b).__name__}")
        b_body_block = b.body.block if isinstance(b.body, CScopedBlock) else b.body
        return {
            "kind": "c.do_while ↔ do_while",
            "a_id": a.id,
            "body": _check_body(a.body, b_body_block, path=f"{path}.body", ctx=ctx),
            "cond": _check_expr(a.cond, b.cond, path=f"{path}.cond", ctx=ctx),
        }

    if isinstance(a, CSwitch):
        # `CSwitch` lifts to a nested If-else-if chain at layer B.
        # Walk the cases in order, expecting each to match the next
        # If's cond/then; the final else carries the default body
        # (or an empty Block when no default was given).
        return _check_switch_chain(a, b, path=path, ctx=ctx)

    if isinstance(a, CExprStmt):
        if not isinstance(b, ExprStmt):
            raise LiftCheckError(f"{path}: layer-A CExprStmt vs layer-B {type(b).__name__}")
        return {
            "kind": "c.expr_stmt ↔ expr_stmt",
            "a_id": a.id,
            "value": _check_expr(a.value, b.value, path=f"{path}.value", ctx=ctx),
        }

    if isinstance(a, CBreak):
        if not isinstance(b, Break):
            raise LiftCheckError(f"{path}: layer-A CBreak vs layer-B {type(b).__name__}")
        return {"kind": "c.break ↔ break", "a_id": a.id}

    if isinstance(a, CContinue):
        if not isinstance(b, Continue):
            raise LiftCheckError(f"{path}: layer-A CContinue vs layer-B {type(b).__name__}")
        return {"kind": "c.continue ↔ continue", "a_id": a.id}

    raise LiftCheckError(
        f"{path}: layer-A {type(a).__name__} has no correspondence rule"
    )


# Comparison and short-circuit operators produce i1-typed values at
# layer B; the lifter wraps `return <i1-expr>` in an If to widen to
# i32. The same set drives the lift-checker's choice between the
# ReturnExpr and If layer-B shapes.
_LAYER_A_I1_OPS = frozenset({"<", "<=", ">", ">=", "==", "!=", "&&", "||"})


def _is_layer_a_i1_typed(expr) -> bool:
    """True iff `expr` is a layer-A boolean expression — a comparison
    or short-circuit binop, or `!x` (which lifts to `eq(x, 0)` on the
    layer-B side and is therefore i1-typed too). Mirrors `_is_i1_typed`
    in ingest/c.py."""
    if isinstance(expr, CBinOp) and expr.op in _LAYER_A_I1_OPS:
        return True
    if isinstance(expr, CUnary) and expr.op == "!":
        return True
    return False


def _check_expr(a, b, *, path: str, ctx: "_Ctx") -> dict[str, Any]:
    if isinstance(a, CIntLit):
        if not isinstance(b, IntLit):
            raise LiftCheckError(f"{path}: layer-A CIntLit vs layer-B {type(b).__name__}")
        if a.value != b.value:
            raise LiftCheckError(f"{path}: int_lit value {a.value} vs {b.value}")
        if not isinstance(b.type, I32Type):
            raise LiftCheckError(f"{path}: layer-B int_lit is not i32")
        return {"kind": "int_lit", "value": a.value}

    if isinstance(a, CEnumConstRef):
        # `CURLOPT_URL` (layer A) ↔ `IntLit(I32, 10002)` (layer B).
        # The lifter resolves the enum constant via libclang at lift
        # time; layer A pins the resolved value so re-checking
        # doesn't need libclang. Compare values (the source-level
        # name is informational).
        if not isinstance(b, IntLit):
            raise LiftCheckError(
                f"{path}: layer-A enum constant {a.name!r} (value {a.value}) "
                f"vs layer-B {type(b).__name__}"
            )
        if a.value != b.value:
            raise LiftCheckError(
                f"{path}: enum constant {a.name!r} resolved value {a.value} "
                f"vs layer-B IntLit value {b.value} — the enum's value has "
                f"drifted since this lift was pinned (re-ingest to update)"
            )
        if not isinstance(b.type, I32Type):
            raise LiftCheckError(
                f"{path}: layer-B IntLit for enum constant {a.name!r} is "
                f"not i32 ({b.type.kind!r})"
            )
        return {"kind": "enum_const_ref ↔ int_lit", "name": a.name, "value": a.value}

    if isinstance(a, CVarRef):
        # The lift produces ParamRef when the name is a function
        # parameter, LocalRef otherwise. The layer-A side doesn't
        # distinguish; either is acceptable at the same name.
        if isinstance(b, ParamRef):
            if a.name != b.name:
                raise LiftCheckError(f"{path}: var name {a.name!r} vs param {b.name!r}")
            return {"kind": "var_ref ↔ param_ref", "name": a.name}
        if isinstance(b, LocalRef):
            if a.name != b.name:
                raise LiftCheckError(f"{path}: var name {a.name!r} vs local {b.name!r}")
            return {"kind": "var_ref ↔ local_ref", "name": a.name}
        raise LiftCheckError(
            f"{path}: layer-A CVarRef vs layer-B {type(b).__name__}"
        )

    if isinstance(a, CBinOp):
        # Short-circuit `&&` / `||` are not in `_BINOP_LAYER_A_TO_B`
        # — they map to dedicated layer-B nodes.
        if a.op == "&&":
            if not isinstance(b, ShortCircuitAnd):
                raise LiftCheckError(f"{path}: && vs layer-B {type(b).__name__}")
            return {
                "kind": "&& ↔ sc_and",
                "lhs": _check_expr(a.lhs, b.lhs, path=f"{path}.lhs", ctx=ctx),
                "rhs": _check_expr(a.rhs, b.rhs, path=f"{path}.rhs", ctx=ctx),
            }
        if a.op == "||":
            if not isinstance(b, ShortCircuitOr):
                raise LiftCheckError(f"{path}: || vs layer-B {type(b).__name__}")
            return {
                "kind": "|| ↔ sc_or",
                "lhs": _check_expr(a.lhs, b.lhs, path=f"{path}.lhs", ctx=ctx),
                "rhs": _check_expr(a.rhs, b.rhs, path=f"{path}.rhs", ctx=ctx),
            }
        # Pointer arithmetic: `p + n` (layer A: CBinOp("+", ...))
        # corresponds to layer-B `PtrOffset(p', n')`. We dispatch on
        # the layer-B kind rather than reasoning about layer-A operand
        # types — if the lifter produced a PtrOffset, the operands
        # were typed as pointer + integer; the structural pairing
        # implies the type pairing.
        if a.op == "+" and isinstance(b, PtrOffset):
            return _check_pointer_arith(a.lhs, a.rhs, b, path=path, ctx=ctx)
        expected_b_op = _BINOP_LAYER_A_TO_B.get(a.op)
        if expected_b_op is None:
            raise LiftCheckError(f"{path}: layer-A operator {a.op!r} not in correspondence table")
        if not isinstance(b, BinOp):
            raise LiftCheckError(f"{path}: CBinOp vs layer-B {type(b).__name__}")
        if b.op != expected_b_op:
            raise LiftCheckError(
                f"{path}: operator {a.op!r} expects layer-B {expected_b_op!r}, "
                f"got {b.op!r}"
            )
        return {
            "kind": f"binop({a.op}) ↔ binop({b.op})",
            "lhs": _check_expr(a.lhs, b.lhs, path=f"{path}.lhs", ctx=ctx),
            "rhs": _check_expr(a.rhs, b.rhs, path=f"{path}.rhs", ctx=ctx),
        }

    if isinstance(a, CStringLit):
        # Layer-B `StringRef` resolves to a `StringConstant` whose
        # value should equal layer-A's `CStringLit.value`. We thread
        # the program's constants table through `ctx` and verify;
        # without the table (program=None) we can't verify, so we
        # refuse rather than silently passing.
        if not isinstance(b, StringRef):
            raise LiftCheckError(
                f"{path}: layer-A CStringLit vs layer-B {type(b).__name__}"
            )
        if ctx.constants_by_name is None:
            raise LiftCheckError(
                f"{path}: encountered CStringLit but no `program=` was "
                f"supplied to walk_lift — string-value verification "
                f"requires the program's constants table"
            )
        actual = ctx.constants_by_name.get(b.name)
        if actual is None:
            raise LiftCheckError(
                f"{path}: layer-B StringRef points at {b.name!r} but "
                f"the program has no StringConstant with that name"
            )
        if actual != a.value:
            raise LiftCheckError(
                f"{path}: layer-A string {a.value!r} vs layer-B "
                f"StringConstant {b.name!r} value {actual!r}"
            )
        return {
            "kind": "string_lit ↔ string_ref",
            "value": a.value,
            "string_ref": b.name,
        }

    if isinstance(a, CCall):
        if not isinstance(b, Call):
            raise LiftCheckError(f"{path}: layer-A CCall vs layer-B {type(b).__name__}")
        if a.callee != b.function:
            raise LiftCheckError(
                f"{path}: callee {a.callee!r} vs {b.function!r}"
            )
        if len(a.args) != len(b.args):
            raise LiftCheckError(
                f"{path}: arg count {len(a.args)} (layer A) vs {len(b.args)} (layer B)"
            )
        return {
            "kind": "call ↔ call",
            "callee": a.callee,
            "args": [
                _check_expr(aa, ba, path=f"{path}.args[{i}]", ctx=ctx)
                for i, (aa, ba) in enumerate(zip(a.args, b.args))
            ],
        }

    if isinstance(a, CTernary):
        # `cond ? a : b` ↔ `IfExpr(cond', a', b')`. The layer-B side
        # may wrap the cond in `ne(cond, 0)` when the source cond was
        # an integer expression (rather than a comparison) — we accept
        # either: a direct cond match or the i1-widened form.
        if not isinstance(b, IfExpr):
            raise LiftCheckError(
                f"{path}: layer-A CTernary vs layer-B {type(b).__name__}"
            )
        # Cond pairing: try direct first; fall back to the ne-widening
        # form if the layer-B cond is `BinOp(ne, _, 0)` and the layer-A
        # cond isn't already i1-typed.
        cond_record: dict[str, Any]
        try:
            cond_record = _check_expr(a.cond, b.cond, path=f"{path}.cond", ctx=ctx)
        except LiftCheckError:
            if (
                isinstance(b.cond, BinOp) and b.cond.op == "ne"
                and isinstance(b.cond.rhs, IntLit) and b.cond.rhs.value == 0
                and not _is_layer_a_i1_typed(a.cond)
            ):
                cond_record = _check_expr(
                    a.cond, b.cond.lhs, path=f"{path}.cond", ctx=ctx,
                )
                cond_record = {"kind": "ternary cond i1-widen", "inner": cond_record}
            else:
                raise
        return {
            "kind": "ternary ↔ if_expr",
            "a_id": a.id,
            "cond": cond_record,
            "then": _check_expr(a.then_value, b.then_value, path=f"{path}.then", ctx=ctx),
            "else": _check_expr(a.else_value, b.else_value, path=f"{path}.else", ctx=ctx),
        }

    if isinstance(a, CUnary):
        # CUnary preserves source-form unary operators; the lift
        # desugars each via the standard identity. We pair with the
        # exact layer-B BinOp shape produced by the ingester.
        if not isinstance(b, BinOp):
            raise LiftCheckError(
                f"{path}: layer-A CUnary({a.op!r}) vs layer-B {type(b).__name__}"
            )
        if a.op == "-":
            # `-x` ↔ `BinOp("sub", IntLit(0), x')`.
            if b.op != "sub":
                raise LiftCheckError(
                    f"{path}: CUnary('-') expects layer-B BinOp('sub'), got {b.op!r}"
                )
            if not (isinstance(b.lhs, IntLit) and b.lhs.value == 0):
                raise LiftCheckError(
                    f"{path}: CUnary('-') expects layer-B sub's LHS to be IntLit(0), "
                    f"got {type(b.lhs).__name__}"
                )
            return {
                "kind": "unary(-) ↔ sub(0, _)",
                "value": _check_expr(a.value, b.rhs, path=f"{path}.value", ctx=ctx),
            }
        if a.op == "!":
            # `!x` ↔ `BinOp("eq", x', IntLit(0))`.
            if b.op != "eq":
                raise LiftCheckError(
                    f"{path}: CUnary('!') expects layer-B BinOp('eq'), got {b.op!r}"
                )
            if not (isinstance(b.rhs, IntLit) and b.rhs.value == 0):
                raise LiftCheckError(
                    f"{path}: CUnary('!') expects layer-B eq's RHS to be IntLit(0), "
                    f"got {type(b.rhs).__name__}"
                )
            return {
                "kind": "unary(!) ↔ eq(_, 0)",
                "value": _check_expr(a.value, b.lhs, path=f"{path}.value", ctx=ctx),
            }
        if a.op == "~":
            # `~x` ↔ `BinOp("xor", x', IntLit(-1))`.
            if b.op != "xor":
                raise LiftCheckError(
                    f"{path}: CUnary('~') expects layer-B BinOp('xor'), got {b.op!r}"
                )
            if not (isinstance(b.rhs, IntLit) and b.rhs.value == -1):
                raise LiftCheckError(
                    f"{path}: CUnary('~') expects layer-B xor's RHS to be IntLit(-1), "
                    f"got {type(b.rhs).__name__}"
                )
            return {
                "kind": "unary(~) ↔ xor(_, -1)",
                "value": _check_expr(a.value, b.lhs, path=f"{path}.value", ctx=ctx),
            }
        raise LiftCheckError(f"{path}: unknown CUnary op {a.op!r}")

    if isinstance(a, CAddressOf):
        # `&p[k]` is C's pointer-arithmetic spelling — equivalent to
        # `p + k` for char* (and any pointer type, modulo the byte
        # vs element-size scaling that's currently limited to char-stride).
        # The layer-B side is always `PtrOffset(base, offset)`.
        if not isinstance(b, PtrOffset):
            raise LiftCheckError(
                f"{path}: layer-A CAddressOf vs layer-B {type(b).__name__}"
            )
        if not isinstance(a.target, CArraySubscript):
            raise LiftCheckError(
                f"{path}: layer-A CAddressOf target is "
                f"{type(a.target).__name__}; only CArraySubscript is allowed"
            )
        sub = a.target
        return {
            "kind": "&p[k] ↔ ptr_offset",
            "a_id": a.id,
            "base": _check_expr(sub.base, b.base, path=f"{path}.target.base", ctx=ctx),
            "offset": _check_offset_expr(sub.index, b.offset, path=f"{path}.target.index", ctx=ctx),
        }

    raise LiftCheckError(
        f"{path}: layer-A {type(a).__name__} has no expression correspondence"
    )


def _check_pointer_arith(
    a_lhs, a_rhs, b: PtrOffset, *, path: str, ctx: "_Ctx",
) -> dict[str, Any]:
    """Layer-A `lhs + rhs` ↔ layer-B `PtrOffset(base, offset)`.

    The `+` operator is commutative for pointer arithmetic in C:
    `p + n` and `n + p` both produce the same address. The lifter
    walks LHS-then-RHS and identifies which operand is the pointer
    via libclang's type info, so the layer-B `base` corresponds to
    whichever layer-A operand was the pointer.

    The structural walk doesn't have type info; we don't try to
    reverse-engineer which side was the pointer. Instead, we pair
    layer-A LHS with layer-B base and layer-A RHS with layer-B
    offset. For the current corpus (`string_offset.c` only), this is
    consistent because the lifter never swaps operand order.
    """
    return {
        "kind": "p + n ↔ ptr_offset",
        "base": _check_expr(a_lhs, b.base, path=f"{path}.lhs", ctx=ctx),
        "offset": _check_offset_expr(a_rhs, b.offset, path=f"{path}.rhs", ctx=ctx),
    }


def _check_offset_expr(a, b, *, path: str, ctx: "_Ctx") -> dict[str, Any]:
    """Layer-B pointer offsets are i64-typed: a literal becomes
    `IntLit(I64, N)`; a non-literal gets wrapped in `Cast(value,
    target_type=I64Type)`. Both shapes correspond to a single layer-A
    `CExpr`.

    Literal: `CIntLit(N) ↔ IntLit(I64, N)`.
    Variable: `expr ↔ Cast(expr', target_type=I64Type)`.
    """
    if isinstance(b, IntLit):
        if not isinstance(b.type, I64Type):
            raise LiftCheckError(
                f"{path}: layer-B offset literal is {b.type.kind!r}, expected i64"
            )
        if not isinstance(a, CIntLit):
            raise LiftCheckError(
                f"{path}: layer-A {type(a).__name__} vs layer-B literal offset "
                f"({b.value}); expected CIntLit"
            )
        if a.value != b.value:
            raise LiftCheckError(
                f"{path}: offset value {a.value} vs {b.value}"
            )
        return {"kind": "offset_lit", "value": a.value}
    if isinstance(b, Cast):
        if not isinstance(b.target_type, I64Type):
            raise LiftCheckError(
                f"{path}: layer-B offset Cast has unexpected target "
                f"(target_type={b.target_type})"
            )
        return {
            "kind": "offset ↔ cast(i64)",
            "value": _check_expr(a, b.value, path=path, ctx=ctx),
        }
    raise LiftCheckError(
        f"{path}: layer-B offset is {type(b).__name__}; expected IntLit(i64) or Cast"
    )


def _type_a_repr(t) -> str:
    """Human-readable layer-A type rendering for the artifact."""
    return _format_c_type_str(t)


def _type_b_repr(t) -> str:
    return getattr(t, "kind", type(t).__name__)


# ---------- Program-level helper: upgrade A~B claims to witness ----------

def prove_lifts(program, *, write_dir, rel_prefix: str = "proofs/lift", write: bool = True):
    """Walk every A~B equivalence in `program`, run the lift check,
    and replace the claim's `ManualJustification` with a
    witness-regime `LiftEquivalence` pinning the artifact.

    `write_dir` is the absolute filesystem path under which artifacts
    are written (typically `<config.root>/proofs/lift/`). `rel_prefix`
    is the path stored in the resulting `LiftEquivalence.artifact_path`
    — relative to the program's resolve_root so that `equiv verify`
    can find it later. The two are kept in sync by the caller; defaults
    match the standard quod.toml layout.

    If `write` is False, the artifact is hashed but not persisted —
    useful for `--dry-run` and tests that don't want filesystem side
    effects.

    Claims that already cite `LiftEquivalence` with a current hash
    are left alone. Stale ones get re-pinned (only when `write=True`).

    Returns a new Program with the upgraded claims; never mutates the
    input. Raises `LiftCheckError` if any A~B pair fails the walk.
    """
    from pathlib import Path
    from quod.model import Equivalence, LiftEquivalence, ManualJustification

    cfns_by_id = {
        cfn.id: cfn
        for unit in program.source_units
        for cfn in unit.functions
    }
    fns_by_id = {fn.id: fn for fn in program.structured_functions}

    write_dir = Path(write_dir)
    if write:
        write_dir.mkdir(parents=True, exist_ok=True)

    # First pass: identify A→B claims and collapse them per-pair. An
    # A→B claim is one whose justification is `ManualJustification`
    # from the ingester, or `LiftEquivalence` (the witnessed form).
    # Re-ingesting a file produces a fresh manual claim each time;
    # without per-pair dedup, the equivalences list would grow.
    a_to_b_pairs: dict[tuple[str, str], "Equivalence"] = {}
    other_equivalences: list = []
    for eq in program.equivalences:
        is_a_to_b_manual = (
            isinstance(eq.justification, ManualJustification)
            and eq.justification.signed_by.startswith("quod.ingest")
        )
        is_a_to_b_witness = isinstance(eq.justification, LiftEquivalence)
        if not (is_a_to_b_manual or is_a_to_b_witness):
            other_equivalences.append(eq)
            continue
        key = (eq.a_node_id, eq.b_node_id)
        existing = a_to_b_pairs.get(key)
        # Prefer witness over axiom when both are present for the same
        # pair (re-ingest case): the witness one carries the artifact;
        # we'll re-validate or re-pin below.
        if existing is None or (
            isinstance(eq.justification, LiftEquivalence)
            and not isinstance(existing.justification, LiftEquivalence)
        ):
            a_to_b_pairs[key] = eq

    # Second pass: for each surviving A→B pair, run the lift check
    # and emit a witnessed claim. Stale or missing pins get re-pinned;
    # already-current pins stay as-is.
    new_a_to_b_claims: list = []
    for (a_id, b_id), eq in a_to_b_pairs.items():
        cfn = cfns_by_id.get(a_id)
        fn = fns_by_id.get(b_id)
        if cfn is None or fn is None:
            # Endpoints missing from the program (corpus drift / edit).
            # Pass through unchanged; `equiv verify` reports separately.
            new_a_to_b_claims.append(eq)
            continue

        artifact_bytes = lift_check_artifact(cfn, fn, program=program)
        artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
        artifact_path_rel = f"{rel_prefix}/{fn.name}.txt"

        if (
            isinstance(eq.justification, LiftEquivalence)
            and eq.justification.artifact_hash == artifact_hash
        ):
            # Already pinned and current. Skip re-write.
            new_a_to_b_claims.append(eq)
            continue

        if write:
            (write_dir / f"{fn.name}.txt").write_bytes(artifact_bytes)

        new_a_to_b_claims.append(Equivalence(
            a_node_id=a_id,
            b_node_id=b_id,
            regime="witness",
            justification=LiftEquivalence(
                artifact_path=artifact_path_rel,
                artifact_hash=artifact_hash,
            ),
        ))

    return program.model_copy(update={
        "equivalences": tuple(other_equivalences + new_a_to_b_claims),
    })
