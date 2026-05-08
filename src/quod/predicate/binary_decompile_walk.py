"""Tree-walk equivalence between source CFn and binary-lifted CFn.

After a binary ingest, a function recovered by Ghidra exists in the
graph in two parallel Layer-A forms:

  - source CFn  (id `@cfn_c_<name>`)        — what the user authored
  - lifted CFn  (id `@cfn_lifted_<bin_id>`) — what `decompile_lift`
                                              recovered from the binary

This module walks them in lockstep against a per-construct
correspondence table, succeeding when they're structurally
equivalent and raising `LiftCheckError` (with a path) when they
diverge.

It is the tree-walk sibling of `quod.lift_check.walk_lift` (which
checks Layer-A `CFn` against Layer-B `Function`). Where that
walker bridges two different node families, this one walks two
instances of the same family, so the per-construct rules collapse
to "same kind, same fields, recurse on the children."

**Strict structural correspondence — no canonicalization.** Given
clang's freedom to reorder commutative operands, fold, and apply
algebraic identities, this walker WILL flag `a + b` vs `b + a` as a
mismatch. That's the deliverable: the diagnostic surfaces the
specific structural difference. Soundness comes from "if these
walk pure, they denote the same function"; the converse (different
shape ⇒ different function) is not asserted, and downstream
provers (`z3.bin_relational`) handle the semantic-but-not-
structural cases.

Usage:

    from quod.predicate.binary_decompile_walk import (
        walk_decompile_lift, decompile_lift_check_hash,
    )

    record = walk_decompile_lift(src_cfn, lifted_cfn)  # raises on miss
    h = decompile_lift_check_hash(src_cfn, lifted_cfn)

The CLI entry point is `prove_decompile_lifts(program, ...)`, which
walks the program's equivalence chain to pair (src_cfn, lifted_cfn)
through their shared bin.fn endpoint and emits Equivalence claims
with `LiftEquivalence` justification.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quod.lift_check import LiftCheckError
from quod.model import (
    BinaryProvenance,
    CAddressOf,
    CArraySubscript,
    CAssign,
    CBinOp,
    CBreak,
    CCall,
    CCast,
    CCompoundAssign,
    CContinue,
    CDoWhile,
    CEnumConstRef,
    CExprStmt,
    CFloatLit,
    CFn,
    CFor,
    CForInit,
    CIf,
    CIncrementStmt,
    CIntLit,
    CMultiVarDecl,
    CNamedType,
    CParam,
    CPointerType,
    CReturn,
    CStringLit,
    CSwitch,
    CSwitchCase,
    CTernary,
    CType,
    CUnary,
    CVarDecl,
    CVarRef,
    CWhile,
    DecompileLift,
    Equivalence,
    LiftEquivalence,
    Program,
)


# ---------- Public walker ----------

def walk_decompile_lift(src_cfn: CFn, lifted_cfn: CFn) -> dict[str, Any]:
    """Walk `src_cfn` and `lifted_cfn` in lockstep and return a
    deterministic JSON-serializable record of the correspondence.

    Raises `LiftCheckError` on the first divergence. The error
    message carries a path locating the mismatch within the
    function tree (e.g.
    `fn[affine].body[0].value.lhs: kind mismatch — c.binop vs c.lit_int`).
    """
    if src_cfn.name != lifted_cfn.name:
        raise LiftCheckError(
            f"function name mismatch: source {src_cfn.name!r} "
            f"vs lifted {lifted_cfn.name!r}"
        )

    if len(src_cfn.params) != len(lifted_cfn.params):
        raise LiftCheckError(
            f"function {src_cfn.name!r}: param count "
            f"{len(src_cfn.params)} (source) vs "
            f"{len(lifted_cfn.params)} (lifted)"
        )

    params_record: list[dict[str, Any]] = []
    for i, (sp, lp) in enumerate(zip(src_cfn.params, lifted_cfn.params)):
        params_record.append(
            _check_param(sp, lp, path=f"fn[{src_cfn.name}].params[{i}]")
        )

    _check_type(
        src_cfn.return_type, lifted_cfn.return_type,
        path=f"fn[{src_cfn.name}].return_type",
    )

    body_record = _check_stmt_seq(
        src_cfn.body, lifted_cfn.body,
        path=f"fn[{src_cfn.name}].body",
    )

    return {
        "kind": "decompile-lift-check",
        "version": "1",
        "rule": "c.cfn_correspondence",
        "fn": {
            "name": src_cfn.name,
            "src_id": src_cfn.id,
            "lifted_id": lifted_cfn.id,
            "params": params_record,
            "return_type": _type_repr(src_cfn.return_type),
            "body": body_record,
        },
    }


def decompile_lift_check_artifact(src_cfn: CFn, lifted_cfn: CFn) -> bytes:
    """Walk and serialize to deterministic bytes.

    The bytes are what gets written to disk and what
    `LiftEquivalence.artifact_hash` pins.
    """
    record = walk_decompile_lift(src_cfn, lifted_cfn)
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    return text.encode("utf-8")


def decompile_lift_check_hash(src_cfn: CFn, lifted_cfn: CFn) -> str:
    """sha256 of the decompile-lift-check artifact bytes."""
    return hashlib.sha256(
        decompile_lift_check_artifact(src_cfn, lifted_cfn)
    ).hexdigest()


# ---------- Per-construct walkers ----------

def _check_param(sp: CParam, lp: CParam, *, path: str) -> dict[str, Any]:
    if sp.name != lp.name:
        raise LiftCheckError(
            f"{path}: param name {sp.name!r} (source) "
            f"vs {lp.name!r} (lifted)"
        )
    _check_type(sp.type, lp.type, path=f"{path}.type")
    return {"name": sp.name, "type": _type_repr(sp.type)}


def _check_type(s: CType, l: CType, *, path: str) -> None:
    """Both sides are `c.*` types — the rule is structural identity."""
    if isinstance(s, CNamedType) and isinstance(l, CNamedType):
        if s.name != l.name:
            raise LiftCheckError(
                f"{path}: type name {s.name!r} (source) "
                f"vs {l.name!r} (lifted)"
            )
        return
    if isinstance(s, CPointerType) and isinstance(l, CPointerType):
        _check_type(s.pointee, l.pointee, path=f"{path}.pointee")
        return
    raise LiftCheckError(
        f"{path}: type kind mismatch — "
        f"source {type(s).__name__} vs lifted {type(l).__name__}"
    )


def _type_repr(t: CType) -> str:
    if isinstance(t, CNamedType):
        return t.name
    if isinstance(t, CPointerType):
        return _type_repr(t.pointee) + "*"
    return repr(t)


def _check_stmt_seq(
    src: tuple, lifted: tuple, *, path: str,
) -> list[dict[str, Any]]:
    if len(src) != len(lifted):
        raise LiftCheckError(
            f"{path}: stmt count {len(src)} (source) "
            f"vs {len(lifted)} (lifted)"
        )
    out: list[dict[str, Any]] = []
    for i, (s, l) in enumerate(zip(src, lifted)):
        out.append(_check_stmt(s, l, path=f"{path}[{i}]"))
    return out


def _check_stmt(s, l, *, path: str) -> dict[str, Any]:
    if type(s) is not type(l):
        raise LiftCheckError(
            f"{path}: kind mismatch — "
            f"source {type(s).__name__} vs lifted {type(l).__name__}"
        )

    if isinstance(s, CReturn):
        if (s.value is None) != (l.value is None):
            raise LiftCheckError(
                f"{path}.value: source {'has' if s.value else 'no'} value, "
                f"lifted {'has' if l.value else 'no'} value"
            )
        rec: dict[str, Any] = {"kind": "c.return"}
        if s.value is not None:
            rec["value"] = _check_expr(s.value, l.value, path=f"{path}.value")
        return rec

    if isinstance(s, CIf):
        return {
            "kind": "c.if",
            "cond": _check_expr(s.cond, l.cond, path=f"{path}.cond"),
            "then_body": _check_stmt_seq(
                s.then_body, l.then_body, path=f"{path}.then_body",
            ),
            "else_body": _check_stmt_seq(
                s.else_body, l.else_body, path=f"{path}.else_body",
            ),
        }

    if isinstance(s, CWhile):
        return {
            "kind": "c.while",
            "cond": _check_expr(s.cond, l.cond, path=f"{path}.cond"),
            "body": _check_stmt_seq(
                s.body, l.body, path=f"{path}.body",
            ),
        }

    if isinstance(s, CDoWhile):
        return {
            "kind": "c.do_while",
            "body": _check_stmt_seq(
                s.body, l.body, path=f"{path}.body",
            ),
            "cond": _check_expr(s.cond, l.cond, path=f"{path}.cond"),
        }

    if isinstance(s, CFor):
        return {
            "kind": "c.for",
            "init": _check_for_init(
                s.init, l.init, path=f"{path}.init",
            ),
            "cond": (
                _check_expr(s.cond, l.cond, path=f"{path}.cond")
                if s.cond is not None and l.cond is not None
                else _check_optional(
                    s.cond, l.cond, path=f"{path}.cond",
                )
            ),
            "inc": _check_for_init(
                s.inc, l.inc, path=f"{path}.inc",
            ),
            "body": _check_stmt_seq(
                s.body, l.body, path=f"{path}.body",
            ),
        }

    if isinstance(s, CSwitch):
        if len(s.cases) != len(l.cases):
            raise LiftCheckError(
                f"{path}.cases: case count {len(s.cases)} (source) "
                f"vs {len(l.cases)} (lifted)"
            )
        cases_rec = [
            _check_switch_case(sc, lc, path=f"{path}.cases[{i}]")
            for i, (sc, lc) in enumerate(zip(s.cases, l.cases))
        ]
        return {
            "kind": "c.switch",
            "scrutinee": _check_expr(
                s.scrutinee, l.scrutinee, path=f"{path}.scrutinee",
            ),
            "cases": cases_rec,
            "default": _check_optional_stmt_seq(
                s.default, l.default, path=f"{path}.default",
            ),
        }

    if isinstance(s, CVarDecl):
        if s.name != l.name:
            raise LiftCheckError(
                f"{path}.name: {s.name!r} (source) vs {l.name!r} (lifted)"
            )
        _check_type(s.type, l.type, path=f"{path}.type")
        rec = {"kind": "c.var_decl", "name": s.name,
               "type": _type_repr(s.type)}
        if (s.init is None) != (l.init is None):
            raise LiftCheckError(
                f"{path}.init: source {'has' if s.init else 'no'} init, "
                f"lifted {'has' if l.init else 'no'} init"
            )
        if s.init is not None:
            rec["init"] = _check_expr(s.init, l.init, path=f"{path}.init")
        return rec

    if isinstance(s, CMultiVarDecl):
        if len(s.decls) != len(l.decls):
            raise LiftCheckError(
                f"{path}.decls: count {len(s.decls)} (source) "
                f"vs {len(l.decls)} (lifted)"
            )
        return {
            "kind": "c.multi_var_decl",
            "decls": [
                _check_stmt(sd, ld, path=f"{path}.decls[{i}]")
                for i, (sd, ld) in enumerate(zip(s.decls, l.decls))
            ],
        }

    if isinstance(s, CAssign):
        if s.target != l.target:
            raise LiftCheckError(
                f"{path}.target: {s.target!r} vs {l.target!r}"
            )
        return {
            "kind": "c.assign", "target": s.target,
            "value": _check_expr(s.value, l.value, path=f"{path}.value"),
        }

    if isinstance(s, CCompoundAssign):
        if s.target != l.target or s.op != l.op:
            raise LiftCheckError(
                f"{path}: compound-assign mismatch — source "
                f"{s.target!r}{s.op}=, lifted {l.target!r}{l.op}="
            )
        return {
            "kind": "c.compound_assign", "target": s.target, "op": s.op,
            "value": _check_expr(s.value, l.value, path=f"{path}.value"),
        }

    if isinstance(s, CIncrementStmt):
        if s.target != l.target or s.op != l.op or s.position != l.position:
            raise LiftCheckError(
                f"{path}: increment-stmt mismatch — source "
                f"{s.target!r}{s.op} ({s.position}), lifted "
                f"{l.target!r}{l.op} ({l.position})"
            )
        return {
            "kind": "c.increment_stmt", "target": s.target,
            "op": s.op, "position": s.position,
        }

    if isinstance(s, CExprStmt):
        return {
            "kind": "c.expr_stmt",
            "value": _check_expr(s.value, l.value, path=f"{path}.value"),
        }

    if isinstance(s, CBreak):
        return {"kind": "c.break"}

    if isinstance(s, CContinue):
        return {"kind": "c.continue"}

    raise LiftCheckError(
        f"{path}: unsupported stmt kind {type(s).__name__}"
    )


def _check_for_init(s, l, *, path: str) -> dict[str, Any] | None:
    if s is None and l is None:
        return None
    if (s is None) != (l is None):
        raise LiftCheckError(
            f"{path}: source {'has' if s else 'no'} init, "
            f"lifted {'has' if l else 'no'} init"
        )
    return _check_stmt(s, l, path=path)


def _check_optional(s, l, *, path: str) -> None:
    if (s is None) != (l is None):
        raise LiftCheckError(
            f"{path}: source {'has' if s else 'no'}, "
            f"lifted {'has' if l else 'no'}"
        )
    return None


def _check_optional_stmt_seq(s, l, *, path: str):
    if s is None and l is None:
        return None
    if (s is None) != (l is None):
        raise LiftCheckError(
            f"{path}: source {'has' if s else 'no'} default, "
            f"lifted {'has' if l else 'no'} default"
        )
    return _check_stmt_seq(s, l, path=path)


def _check_switch_case(
    s: CSwitchCase, l: CSwitchCase, *, path: str,
) -> dict[str, Any]:
    if len(s.values) != len(l.values):
        raise LiftCheckError(
            f"{path}.values: count {len(s.values)} vs {len(l.values)}"
        )
    return {
        "kind": "c.switch_case",
        "values": [
            _check_expr(sv, lv, path=f"{path}.values[{i}]")
            for i, (sv, lv) in enumerate(zip(s.values, l.values))
        ],
        "body": _check_stmt_seq(
            s.body, l.body, path=f"{path}.body",
        ),
    }


def _check_expr(s, l, *, path: str) -> dict[str, Any]:
    if type(s) is not type(l):
        raise LiftCheckError(
            f"{path}: kind mismatch — "
            f"source {type(s).__name__} vs lifted {type(l).__name__}"
        )

    if isinstance(s, CIntLit):
        if s.value != l.value:
            raise LiftCheckError(
                f"{path}: literal value {s.value} (source) "
                f"vs {l.value} (lifted)"
            )
        if type(s.type) is not type(l.type) or s.type.kind != l.type.kind:
            raise LiftCheckError(
                f"{path}: literal type {s.type.kind} (source) "
                f"vs {l.type.kind} (lifted)"
            )
        return {"kind": "c.lit_int", "value": s.value, "type": s.type.kind}

    if isinstance(s, CFloatLit):
        if s.bits != l.bits:
            raise LiftCheckError(
                f"{path}: float bits 0x{s.bits:x} (source) "
                f"vs 0x{l.bits:x} (lifted)"
            )
        return {"kind": "c.lit_float", "bits": s.bits,
                "type": s.type.kind}

    if isinstance(s, CStringLit):
        if s.value != l.value:
            raise LiftCheckError(
                f"{path}: string literal mismatch"
            )
        return {"kind": "c.lit_str", "value": s.value}

    if isinstance(s, CVarRef):
        if s.name != l.name:
            raise LiftCheckError(
                f"{path}: var ref {s.name!r} (source) "
                f"vs {l.name!r} (lifted)"
            )
        return {"kind": "c.var_ref", "name": s.name}

    if isinstance(s, CEnumConstRef):
        if s.name != l.name or s.value != l.value:
            raise LiftCheckError(
                f"{path}: enum const ref {s.name!r}={s.value} (source) "
                f"vs {l.name!r}={l.value} (lifted)"
            )
        return {"kind": "c.enum_const_ref", "name": s.name,
                "value": s.value}

    if isinstance(s, CBinOp):
        if s.op != l.op:
            raise LiftCheckError(
                f"{path}: binop {s.op!r} (source) vs {l.op!r} (lifted)"
            )
        return {
            "kind": "c.binop", "op": s.op,
            "lhs": _check_expr(s.lhs, l.lhs, path=f"{path}.lhs"),
            "rhs": _check_expr(s.rhs, l.rhs, path=f"{path}.rhs"),
        }

    if isinstance(s, CUnary):
        if s.op != l.op:
            raise LiftCheckError(
                f"{path}: unary {s.op!r} (source) vs {l.op!r} (lifted)"
            )
        return {
            "kind": "c.unary", "op": s.op,
            "value": _check_expr(s.value, l.value, path=f"{path}.value"),
        }

    if isinstance(s, CTernary):
        return {
            "kind": "c.ternary",
            "cond": _check_expr(s.cond, l.cond, path=f"{path}.cond"),
            "then": _check_expr(
                s.then_value, l.then_value, path=f"{path}.then",
            ),
            "else": _check_expr(
                s.else_value, l.else_value, path=f"{path}.else",
            ),
        }

    if isinstance(s, CCast):
        _check_type(
            s.target_type, l.target_type, path=f"{path}.target_type",
        )
        return {
            "kind": "c.cast", "target_type": _type_repr(s.target_type),
            "value": _check_expr(s.value, l.value, path=f"{path}.value"),
        }

    if isinstance(s, CCall):
        if s.callee != l.callee:
            raise LiftCheckError(
                f"{path}: call target {s.callee!r} (source) "
                f"vs {l.callee!r} (lifted)"
            )
        if len(s.args) != len(l.args):
            raise LiftCheckError(
                f"{path}: arg count {len(s.args)} (source) "
                f"vs {len(l.args)} (lifted)"
            )
        return {
            "kind": "c.call", "callee": s.callee,
            "args": [
                _check_expr(sa, la, path=f"{path}.args[{i}]")
                for i, (sa, la) in enumerate(zip(s.args, l.args))
            ],
        }

    if isinstance(s, CArraySubscript):
        return {
            "kind": "c.array_subscript",
            "base": _check_expr(s.base, l.base, path=f"{path}.base"),
            "index": _check_expr(s.index, l.index, path=f"{path}.index"),
        }

    if isinstance(s, CAddressOf):
        return {
            "kind": "c.addr_of",
            "target": _check_expr(
                s.target, l.target, path=f"{path}.target",
            ),
        }

    raise LiftCheckError(
        f"{path}: unsupported expr kind {type(s).__name__}"
    )


# ---------- Program-level helper ----------

@dataclass(frozen=True)
class _PairResult:
    """Outcome of attempting to walk one (src_cfn, lifted_cfn) pair.

    `status` is one of:
      - `proven`     — walk succeeded; `equivalence` carries the
                       new witnessed Equivalence ready to merge.
      - `refuted`    — walk raised `LiftCheckError`; the binary
                       isn't structurally what the source says.
                       `detail` holds the error message.
      - `current`    — there's already a witnessed Equivalence at
                       the same hash; nothing to do.
    """
    src_fn_name: str
    src_cfn_id: str
    lifted_cfn_id: str
    status: str  # proven / refuted / current
    detail: str
    equivalence: Equivalence | None = None
    artifact_path: Path | None = None


def prove_decompile_lifts(
    program: Program, *, write_dir: Path,
    rel_prefix: str = "proofs/decompile_lift",
    write: bool = True,
) -> tuple[Program, tuple[_PairResult, ...]]:
    """Run the tree-walk over every (source CFn, lifted CFn) pair the
    program's equivalence chain induces.

    Pairing rule: a `BinaryProvenance`-justified Equivalence pins
    `src_cfn ~ bin_fn`; a `DecompileLift`-justified Equivalence pins
    `lifted_cfn ~ bin_fn`. We pair the two CFns through their shared
    `bin_fn` endpoint.

    Successful walks land as `Equivalence(src_cfn, lifted_cfn,
    regime=witness, justification=LiftEquivalence(...))`, deduped
    against any existing witnessed equivalence at the same
    artifact_hash. Failed walks return their `LiftCheckError`
    message in the `_PairResult` for the CLI to report; no
    equivalence is added.

    Returns `(updated_program, per_pair_results)`. The per-pair
    results are returned in addition to the program (rather than
    only mutating the program) so the CLI can report refutations
    without re-deriving them.

    Idempotent: same source + same lifted CFn produces the same
    artifact bytes, the same hash, and the same equivalence; a
    second run sees `status="current"` and skips the write.
    """
    write_dir = Path(write_dir)

    # Index lookups.
    src_cfns_by_id = {
        cfn.id: cfn
        for unit in program.source_units
        for cfn in unit.functions
    }
    lifted_cfns_by_id = {
        cfn.id: cfn
        for unit in program.binary_units
        for cfn in unit.lifted_cfns
    }

    # Collect bin_fn → src_cfn from BinaryProvenance equivalences. The
    # seeder pairs `src_side ~ bin_fn`, where the source side is a
    # source-unit CFn (Layer-A C-frontend output).
    bin_to_src_cfn: dict[str, str] = {}
    for eq in program.equivalences:
        if not isinstance(eq.justification, BinaryProvenance):
            continue
        # bin_fn is whichever endpoint isn't a source CFn.
        if eq.a_node_id in src_cfns_by_id:
            bin_to_src_cfn[eq.b_node_id] = eq.a_node_id
        elif eq.b_node_id in src_cfns_by_id:
            bin_to_src_cfn[eq.a_node_id] = eq.b_node_id

    # Collect bin_fn → lifted_cfn from DecompileLift equivalences.
    bin_to_lifted_cfn: dict[str, str] = {}
    for eq in program.equivalences:
        if not isinstance(eq.justification, DecompileLift):
            continue
        if eq.a_node_id in lifted_cfns_by_id:
            bin_to_lifted_cfn[eq.b_node_id] = eq.a_node_id
        elif eq.b_node_id in lifted_cfns_by_id:
            bin_to_lifted_cfn[eq.a_node_id] = eq.b_node_id

    # Existing witnessed (src_cfn, lifted_cfn) → artifact_hash, used
    # for idempotent re-runs.
    existing_witnesses: dict[tuple[str, str], str] = {}
    for eq in program.equivalences:
        if (
            eq.regime == "witness"
            and isinstance(eq.justification, LiftEquivalence)
            and eq.a_node_id in src_cfns_by_id
            and eq.b_node_id in lifted_cfns_by_id
        ):
            existing_witnesses[(eq.a_node_id, eq.b_node_id)] = (
                eq.justification.artifact_hash
            )

    if write and (bin_to_src_cfn and bin_to_lifted_cfn):
        write_dir.mkdir(parents=True, exist_ok=True)

    new_eqs: list[Equivalence] = []
    results: list[_PairResult] = []

    for bin_fn_id, src_cfn_id in bin_to_src_cfn.items():
        lifted_cfn_id = bin_to_lifted_cfn.get(bin_fn_id)
        if lifted_cfn_id is None:
            continue
        src_cfn = src_cfns_by_id[src_cfn_id]
        lifted_cfn = lifted_cfns_by_id[lifted_cfn_id]

        try:
            artifact_bytes = decompile_lift_check_artifact(
                src_cfn, lifted_cfn,
            )
        except LiftCheckError as e:
            results.append(_PairResult(
                src_fn_name=src_cfn.name,
                src_cfn_id=src_cfn_id,
                lifted_cfn_id=lifted_cfn_id,
                status="refuted",
                detail=str(e),
            ))
            continue

        artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
        artifact_path_rel = f"{rel_prefix}/{src_cfn.name}.json"

        existing_hash = existing_witnesses.get((src_cfn_id, lifted_cfn_id))
        if existing_hash == artifact_hash:
            results.append(_PairResult(
                src_fn_name=src_cfn.name,
                src_cfn_id=src_cfn_id,
                lifted_cfn_id=lifted_cfn_id,
                status="current",
                detail=f"witness still pinned at {artifact_hash[:12]}",
            ))
            continue

        if write:
            (write_dir / f"{src_cfn.name}.json").write_bytes(artifact_bytes)

        eq = Equivalence(
            a_node_id=src_cfn_id,
            b_node_id=lifted_cfn_id,
            regime="witness",
            justification=LiftEquivalence(
                artifact_path=artifact_path_rel,
                artifact_hash=artifact_hash,
                note=(
                    f"decompile_lift tree-walk: "
                    f"{src_cfn.name!r} src_cfn ↔ lifted_cfn"
                ),
            ),
        )
        new_eqs.append(eq)
        results.append(_PairResult(
            src_fn_name=src_cfn.name,
            src_cfn_id=src_cfn_id,
            lifted_cfn_id=lifted_cfn_id,
            status="proven",
            detail=f"hash {artifact_hash[:12]}",
            equivalence=eq,
            artifact_path=write_dir / f"{src_cfn.name}.json" if write else None,
        ))

    if not new_eqs:
        return program, tuple(results)

    # Replace any stale (src_cfn, lifted_cfn) witness with the new
    # one; otherwise append.
    new_keys = {(eq.a_node_id, eq.b_node_id) for eq in new_eqs}
    kept: list[Equivalence] = []
    for eq in program.equivalences:
        is_replaced_witness = (
            (eq.a_node_id, eq.b_node_id) in new_keys
            and eq.regime == "witness"
            and isinstance(eq.justification, LiftEquivalence)
            and eq.a_node_id in src_cfns_by_id
            and eq.b_node_id in lifted_cfns_by_id
        )
        if not is_replaced_witness:
            kept.append(eq)

    return program.model_copy(update={
        "equivalences": tuple(kept) + tuple(new_eqs),
    }), tuple(results)
