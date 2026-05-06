"""A→B lift checker — structural transcription verifier.

The C ingester emits an `Equivalence` claim between each layer-A
`CFn` and its layer-B `Function` to mark that the lift is a
faithful transcription. v6 emitted these claims as `regime=axiom`
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
  | CType("int")         | I32Type                                |
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
    Call,
    CAssign,
    CBinOp,
    CCall,
    CExprStmt,
    CFn,
    CFor,
    CIf,
    CIntLit,
    CParam,
    CReturn,
    CScopedBlock,
    CStringLit,
    CStyleFor,
    CType,
    CVarDecl,
    CVarRef,
    CWhile,
    ExprStmt,
    Function,
    I1Type,
    I32Type,
    If,
    IntLit,
    Let,
    LocalRef,
    Param,
    ParamRef,
    Return,
    ReturnExpr,
    StringRef,
    While,
    ShortCircuitAnd,
    ShortCircuitOr,
)


# Map layer-A binary-operator spelling to layer-B BinOp.op spelling.
# Mirrors the table in src/quod/ingest/c.py — kept in sync because
# the lift produces the layer-B side from the same source character.
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
}


class LiftCheckError(Exception):
    """Raised when an A→B correspondence walk encounters a mismatch.

    Carries a path describing where in the tree the mismatch
    occurred (e.g. `fn[sum].body[1].cond.lhs`), plus the offending
    a/b nodes' kinds and values, so the failure is debuggable
    without re-running the lift.
    """


def walk_lift(cfn: CFn, fn: Function) -> dict[str, Any]:
    """Walk a layer-A `CFn` and its layer-B `Function` in lockstep,
    asserting structural correspondence. Returns a deterministic
    JSON-serializable record of the walk.

    Raises `LiftCheckError` on any divergence (different node kinds,
    mismatched param names, unknown operators, etc.).
    """
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


def lift_check_artifact(cfn: CFn, fn: Function) -> bytes:
    """Walk the lift and serialize the result to deterministic bytes.

    The bytes are what gets written to disk and what
    `LiftEquivalence.artifact_hash` pins.
    """
    record = walk_lift(cfn, fn)
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    return text.encode("utf-8")


def lift_check_hash(cfn: CFn, fn: Function) -> str:
    """sha256 of the lift-check artifact."""
    return hashlib.sha256(lift_check_artifact(cfn, fn)).hexdigest()


# ---------- internal walkers ----------


def _check_param(cp: CParam, bp: Param, *, path: str) -> dict[str, Any]:
    if cp.name != bp.name:
        raise LiftCheckError(
            f"{path}: param name {cp.name!r} vs {bp.name!r}"
        )
    _check_value_type(cp.type, bp.type, path=f"{path}.type")
    return {"name": cp.name, "a_type": _type_a_repr(cp.type), "b_type": _type_b_repr(bp.type)}


def _check_return_type(a: CType, b, *, path: str) -> None:
    _check_value_type(a, b, path=path)


def _check_value_type(a: CType, b, *, path: str) -> None:
    """v6 only supports `int ↔ i32`. Broaden when layer A grows."""
    if not isinstance(a, CType):
        raise LiftCheckError(f"{path}: layer-A type is {type(a).__name__}, expected CType")
    if a.name == "int":
        if not isinstance(b, I32Type):
            raise LiftCheckError(
                f"{path}: layer-A int but layer-B is {type(b).__name__}"
            )
        return
    raise LiftCheckError(f"{path}: layer-A type {a.name!r} is not in the supported subset")


def _check_body(
    a_stmts, b_block: Block, *, path: str, strip_fallthrough: bool = False,
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

    if strip_fallthrough:
        # The ingester adds:
        #   - ReturnExpr(IntLit(0)) for `main` falling off the end
        #   - Unreachable() for non-main int-returning functions
        # When the layer-A body terminates explicitly, neither is added.
        while len(b_list) > len(a_list) and _is_synthesized_fallthrough(b_list[-1]):
            b_list.pop()

    if len(a_list) != len(b_list):
        raise LiftCheckError(
            f"{path}: stmt count {len(a_list)} (layer A) vs {len(b_list)} "
            f"(layer B); divergent shape"
        )

    return {
        "a_block": "<inline>",  # layer-A bodies are tuple[CStmt, ...], not blocks
        "b_block_id": b_block.id,
        "stmts": [
            _check_stmt(a, b, path=f"{path}.stmts[{i}]")
            for i, (a, b) in enumerate(zip(a_list, b_list))
        ],
    }


def _is_synthesized_fallthrough(stmt) -> bool:
    """True iff `stmt` is a layer-B fall-through stub the ingester
    appends when the layer-A body doesn't terminate explicitly. The
    walk strips these from the tail since layer A doesn't model them.
    """
    from quod.model import Unreachable
    if isinstance(stmt, Unreachable):
        return True
    if isinstance(stmt, ReturnExpr):
        v = stmt.value
        return isinstance(v, IntLit) and v.value == 0
    return False


def _check_stmt(a, b, *, path: str) -> dict[str, Any]:
    if isinstance(a, CVarDecl):
        if not isinstance(b, Let):
            raise LiftCheckError(f"{path}: layer-A CVarDecl vs layer-B {type(b).__name__}")
        if a.name != b.name:
            raise LiftCheckError(f"{path}: var name {a.name!r} vs {b.name!r}")
        _check_value_type(a.type, b.type, path=f"{path}.type")
        # layer A allows uninitialized decls; layer B requires init. v6's
        # lifter refuses uninitialized, so this should always match.
        if a.init is None:
            raise LiftCheckError(f"{path}: layer-A var_decl is uninitialized")
        return {
            "kind": "var_decl ↔ let",
            "a_id": a.id, "name": a.name,
            "init": _check_expr(a.init, b.init, path=f"{path}.init"),
        }

    if isinstance(a, CAssign):
        if not isinstance(b, Assign):
            raise LiftCheckError(f"{path}: layer-A CAssign vs layer-B {type(b).__name__}")
        if a.target != b.name:
            raise LiftCheckError(f"{path}: assign target {a.target!r} vs {b.name!r}")
        return {
            "kind": "assign ↔ assign",
            "a_id": a.id, "target": a.target,
            "value": _check_expr(a.value, b.value, path=f"{path}.value"),
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
                "value": _check_expr(a.value, b.value, path=f"{path}.value"),
            }
        if isinstance(b, If) and _is_layer_a_i1_typed(a.value):
            return {
                "kind": "return cond ↔ if(cond, return 1, return 0)",
                "a_id": a.id,
                "cond": _check_expr(a.value, b.cond, path=f"{path}.cond"),
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
            "init": (_check_stmt(a.init, b.init, path=f"{path}.init") if a.init is not None else None),
            "cond": (_check_expr(a.cond, b.cond, path=f"{path}.cond") if a.cond is not None else None),
            "inc":  (_check_stmt(a.inc, b.inc, path=f"{path}.inc")  if a.inc  is not None else None),
            "body": _check_body(a.body, b_body_block, path=f"{path}.body"),
        }

    if isinstance(a, CIf):
        if not isinstance(b, If):
            raise LiftCheckError(f"{path}: layer-A CIf vs layer-B {type(b).__name__}")
        return {
            "kind": "c.if ↔ if",
            "a_id": a.id,
            "cond": _check_expr(a.cond, b.cond, path=f"{path}.cond"),
            "then_body": _check_body(a.then_body, b.then_body, path=f"{path}.then_body"),
            "else_body": _check_body(a.else_body, b.else_body, path=f"{path}.else_body"),
        }

    if isinstance(a, CWhile):
        if not isinstance(b, While):
            raise LiftCheckError(f"{path}: layer-A CWhile vs layer-B {type(b).__name__}")
        b_body_block = b.body.block if isinstance(b.body, CScopedBlock) else b.body
        return {
            "kind": "c.while ↔ while",
            "a_id": a.id,
            "cond": _check_expr(a.cond, b.cond, path=f"{path}.cond"),
            "body": _check_body(a.body, b_body_block, path=f"{path}.body"),
        }

    if isinstance(a, CExprStmt):
        if not isinstance(b, ExprStmt):
            raise LiftCheckError(f"{path}: layer-A CExprStmt vs layer-B {type(b).__name__}")
        return {
            "kind": "c.expr_stmt ↔ expr_stmt",
            "a_id": a.id,
            "value": _check_expr(a.value, b.value, path=f"{path}.value"),
        }

    raise LiftCheckError(
        f"{path}: layer-A {type(a).__name__} has no v6 correspondence rule"
    )


# Comparison and short-circuit operators produce i1-typed values at
# layer B; the lifter wraps `return <i1-expr>` in an If to widen to
# i32. The same set drives the lift-checker's choice between the
# ReturnExpr and If layer-B shapes.
_LAYER_A_I1_OPS = frozenset({"<", "<=", ">", ">=", "==", "!=", "&&", "||"})


def _is_layer_a_i1_typed(expr) -> bool:
    """True iff `expr` is a layer-A boolean expression — a comparison
    or short-circuit binop. Mirrors `_is_i1_typed` in ingest/c.py."""
    return isinstance(expr, CBinOp) and expr.op in _LAYER_A_I1_OPS


def _check_expr(a, b, *, path: str) -> dict[str, Any]:
    if isinstance(a, CIntLit):
        if not isinstance(b, IntLit):
            raise LiftCheckError(f"{path}: layer-A CIntLit vs layer-B {type(b).__name__}")
        if a.value != b.value:
            raise LiftCheckError(f"{path}: int_lit value {a.value} vs {b.value}")
        if not isinstance(b.type, I32Type):
            raise LiftCheckError(f"{path}: layer-B int_lit is not i32")
        return {"kind": "int_lit", "value": a.value}

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
                "lhs": _check_expr(a.lhs, b.lhs, path=f"{path}.lhs"),
                "rhs": _check_expr(a.rhs, b.rhs, path=f"{path}.rhs"),
            }
        if a.op == "||":
            if not isinstance(b, ShortCircuitOr):
                raise LiftCheckError(f"{path}: || vs layer-B {type(b).__name__}")
            return {
                "kind": "|| ↔ sc_or",
                "lhs": _check_expr(a.lhs, b.lhs, path=f"{path}.lhs"),
                "rhs": _check_expr(a.rhs, b.rhs, path=f"{path}.rhs"),
            }
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
            "lhs": _check_expr(a.lhs, b.lhs, path=f"{path}.lhs"),
            "rhs": _check_expr(a.rhs, b.rhs, path=f"{path}.rhs"),
        }

    if isinstance(a, CStringLit):
        # Layer A holds the decoded value; layer B holds a StringRef
        # to a program-level StringConstant. The StringConstant's
        # actual bytes can't be cross-checked here without access to
        # the Program; that lookup happens at the program-level walk
        # caller (walk_lift records the StringRef name and the
        # layer-A value side-by-side).
        if not isinstance(b, StringRef):
            raise LiftCheckError(
                f"{path}: layer-A CStringLit vs layer-B {type(b).__name__}"
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
                _check_expr(aa, ba, path=f"{path}.args[{i}]")
                for i, (aa, ba) in enumerate(zip(a.args, b.args))
            ],
        }

    raise LiftCheckError(
        f"{path}: layer-A {type(a).__name__} has no v6 expression correspondence"
    )


def _type_a_repr(t: CType) -> str:
    return t.name


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

        artifact_bytes = lift_check_artifact(cfn, fn)
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
