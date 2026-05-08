"""Synthetic tests for `walk_decompile_lift` and `prove_decompile_lifts`.

No Ghidra dependency — every test hand-builds CFns + a synthetic
program with the equivalence-chain shape that `prove_decompile_lifts`
walks (`BinaryProvenance` axiom on src↔bin, `DecompileLift` axiom on
lifted↔bin), then asserts on the walker's output.
"""
from __future__ import annotations

import json
import pytest

from quod.lift_check import LiftCheckError
from quod.model import (
    BinaryProvenance,
    BinFunction,
    BinFunctionParam,
    BinUnit,
    CBinOp,
    CFn,
    CIf,
    CIntLit,
    CNamedType,
    CParam,
    CReturn,
    CUnary,
    CVarRef,
    DecompileLift,
    Equivalence,
    CUnit,
    I32Type,
    LiftEquivalence,
    Program,
    load_program,
    save_program,
)
from quod.predicate.binary_decompile_walk import (
    decompile_lift_check_artifact,
    decompile_lift_check_hash,
    prove_decompile_lifts,
    walk_decompile_lift,
)


_INT = CNamedType(name="int")


def _affine_body(rhs_const: int = 5):
    """`return 3 * x + rhs_const;` as a Layer-A body."""
    return (CReturn(value=CBinOp(
        op="+",
        lhs=CBinOp(op="*",
                   lhs=CIntLit(type=I32Type(), value=3),
                   rhs=CVarRef(name="x")),
        rhs=CIntLit(type=I32Type(), value=rhs_const),
    )),)


def _cfn(name: str, body, *, cfn_id: str | None = None) -> CFn:
    return CFn(
        id=cfn_id or f"@cfn_test_{name}",
        name=name, return_type=_INT,
        params=(CParam(name="x", type=_INT),),
        body=body,
    )


# ---------- walk_decompile_lift ----------

def test_identical_bodies_walk_succeeds():
    src = _cfn("affine", _affine_body(), cfn_id="@cfn_c_affine")
    lifted = _cfn("affine", _affine_body(), cfn_id="@cfn_lifted_X")
    record = walk_decompile_lift(src, lifted)
    assert record["kind"] == "decompile-lift-check"
    assert record["rule"] == "c.cfn_correspondence"
    assert record["fn"]["name"] == "affine"
    assert record["fn"]["src_id"] == "@cfn_c_affine"
    assert record["fn"]["lifted_id"] == "@cfn_lifted_X"


def test_artifact_hash_is_stable():
    """Hashing twice on the same input gives the same digest. Idempotence
    is the foundation of the witness-pin contract."""
    src = _cfn("f", _affine_body(), cfn_id="@cfn_c_f")
    lifted = _cfn("f", _affine_body(), cfn_id="@cfn_lifted_f")
    h1 = decompile_lift_check_hash(src, lifted)
    h2 = decompile_lift_check_hash(src, lifted)
    assert h1 == h2 and len(h1) == 64


def test_artifact_bytes_are_deterministic_json():
    """Artifact bytes are JSON; the hash is sha256 over those bytes
    exactly, sorted-keys, indented. Compatibility shape — verifiers
    re-derive the same bytes."""
    src = _cfn("f", _affine_body(), cfn_id="@cfn_c_f")
    lifted = _cfn("f", _affine_body(), cfn_id="@cfn_lifted_f")
    blob = decompile_lift_check_artifact(src, lifted)
    parsed = json.loads(blob.decode("utf-8"))
    assert parsed["kind"] == "decompile-lift-check"
    # Compact-friendly: trailing newline.
    assert blob.endswith(b"\n")


def test_function_name_mismatch_refused():
    src = _cfn("affine", _affine_body())
    lifted = _cfn("totally_different", _affine_body())
    with pytest.raises(LiftCheckError) as e:
        walk_decompile_lift(src, lifted)
    assert "name mismatch" in str(e.value)


def test_param_count_mismatch_refused():
    src = CFn(
        name="f", return_type=_INT,
        params=(CParam(name="x", type=_INT),),
        body=(CReturn(value=CIntLit(type=I32Type(), value=0)),),
    )
    lifted = CFn(
        name="f", return_type=_INT,
        params=(CParam(name="x", type=_INT), CParam(name="y", type=_INT)),
        body=(CReturn(value=CIntLit(type=I32Type(), value=0)),),
    )
    with pytest.raises(LiftCheckError) as e:
        walk_decompile_lift(src, lifted)
    assert "param count" in str(e.value)


def test_intlit_value_mismatch_locates_the_diff():
    """Source has `+5`, lifted has `+6`. The path in the error message
    pins the exact node — that's the deliverable when Ghidra's
    decompile drifts from source."""
    src = _cfn("affine", _affine_body(rhs_const=5))
    lifted = _cfn("affine", _affine_body(rhs_const=6))
    with pytest.raises(LiftCheckError) as e:
        walk_decompile_lift(src, lifted)
    msg = str(e.value)
    assert "literal value 5" in msg and "vs 6" in msg
    assert "fn[affine].body[0].value.rhs" in msg


def test_kind_mismatch_refused():
    """Source returns a value, lifted returns nothing — different
    Layer-A node kinds within the same body position."""
    src = _cfn("f", (CReturn(value=CIntLit(type=I32Type(), value=0)),))
    lifted = _cfn("f", (CReturn(value=None),))
    with pytest.raises(LiftCheckError) as e:
        walk_decompile_lift(src, lifted)
    assert "value" in str(e.value).lower()


def test_commutative_operand_swap_refused_strictly():
    """Strictly structural — `a + b` vs `b + a` is a miss. The
    diagnostic at this layer is "Ghidra recovered something
    different"; the harder semantic question (do they compute the
    same?) is the SMT prover's job, not this layer's."""
    src = CBinOp(op="+", lhs=CVarRef(name="a"), rhs=CVarRef(name="b"))
    swapped = CBinOp(op="+", lhs=CVarRef(name="b"), rhs=CVarRef(name="a"))
    src_fn = _cfn("f", (CReturn(value=src),))
    swapped_fn = _cfn("f", (CReturn(value=swapped),))
    src_fn = src_fn.model_copy(update={
        "params": (CParam(name="a", type=_INT), CParam(name="b", type=_INT)),
    })
    swapped_fn = swapped_fn.model_copy(update={
        "params": (CParam(name="a", type=_INT), CParam(name="b", type=_INT)),
    })
    with pytest.raises(LiftCheckError) as e:
        walk_decompile_lift(src_fn, swapped_fn)
    # The path to the offending op includes either lhs or rhs.
    msg = str(e.value)
    assert "var ref" in msg.lower() or "kind" in msg.lower()


def test_walks_if_statement_in_lockstep():
    """Single-conditional body — both arms walked recursively."""
    cond = CBinOp(op="<",
                  lhs=CVarRef(name="x"),
                  rhs=CIntLit(type=I32Type(), value=0))
    src_body = (CIf(
        cond=cond,
        then_body=(CReturn(value=CUnary(op="-", value=CVarRef(name="x"))),),
        else_body=(CReturn(value=CVarRef(name="x")),),
    ),)
    src = _cfn("absv", src_body)
    lifted = _cfn("absv", src_body)
    record = walk_decompile_lift(src, lifted)
    assert record["fn"]["body"][0]["kind"] == "c.if"


def test_if_branch_body_mismatch_locates_the_diff():
    cond = CBinOp(op="<", lhs=CVarRef(name="x"),
                  rhs=CIntLit(type=I32Type(), value=0))
    src = _cfn("absv", (CIf(
        cond=cond,
        then_body=(CReturn(value=CUnary(op="-", value=CVarRef(name="x"))),),
        else_body=(CReturn(value=CVarRef(name="x")),),
    ),))
    lifted = _cfn("absv", (CIf(
        cond=cond,
        then_body=(CReturn(value=CVarRef(name="x")),),  # WRONG — not negated
        else_body=(CReturn(value=CVarRef(name="x")),),
    ),))
    with pytest.raises(LiftCheckError) as e:
        walk_decompile_lift(src, lifted)
    msg = str(e.value)
    assert "then_body" in msg


# ---------- prove_decompile_lifts ----------

def _program_with_pair(
    src_cfn: CFn, lifted_cfn: CFn, *, bin_fn_id: str = "@binfn_X",
) -> Program:
    """Build a program with the equivalence chain
    `src_cfn ~ bin_fn (BinaryProvenance) ; lifted_cfn ~ bin_fn (DecompileLift)`."""
    bin_fn = BinFunction(
        id=bin_fn_id,
        address=0x401000,
        mangled_name=src_cfn.name,
        demangled_name=src_cfn.name,
        return_type_name="int",
        params=(BinFunctionParam(name="x", type_name="int"),),
        calling_convention="x86_64-sysv",
        decompile_text=f"int {src_cfn.name}(int x) {{ ... }}",
    )
    src_unit = CUnit(source_path=f"{src_cfn.name}.c", functions=(src_cfn,))
    bin_unit = BinUnit(
        path=f"/tmp/{src_cfn.name}.so", sha256="0" * 64,
        arch="x86_64", file_format="elf",
        functions=(bin_fn,),
        lifted_cfns=(lifted_cfn,),
    )
    return Program(
        source_units=(src_unit,),
        binary_units=(bin_unit,),
        equivalences=(
            Equivalence(
                a_node_id=src_cfn.id, b_node_id=bin_fn_id,
                justification=BinaryProvenance(
                    binary_path=f"/tmp/{src_cfn.name}.so",
                    binary_sha256="0" * 64,
                    binary_symbol=src_cfn.name,
                    source_evidence="dwarf",
                ),
            ),
            Equivalence(
                a_node_id=lifted_cfn.id, b_node_id=bin_fn_id,
                justification=DecompileLift(decompile_text_sha256="0" * 64),
            ),
        ),
    )


def test_prove_emits_witness_for_matching_pair(tmp_path):
    src = _cfn("affine", _affine_body(), cfn_id="@cfn_c_affine")
    lifted = _cfn("affine", _affine_body(), cfn_id="@cfn_lifted_X")
    prog = _program_with_pair(src, lifted)

    new_prog, results = prove_decompile_lifts(
        prog, write_dir=tmp_path, rel_prefix="proofs/decompile_lift",
        write=True,
    )
    assert len(results) == 1
    r = results[0]
    assert r.status == "proven"
    assert r.equivalence is not None
    assert r.equivalence.regime == "witness"
    assert r.equivalence.justification.kind == "lift_equivalence"
    assert (tmp_path / "affine.json").exists()

    # The new equivalence is in the program.
    eqs = [
        e for e in new_prog.equivalences
        if e.regime == "witness"
        and isinstance(e.justification, LiftEquivalence)
    ]
    assert len(eqs) == 1


def test_prove_reports_refuted_for_mismatching_pair(tmp_path):
    src = _cfn("affine", _affine_body(rhs_const=5), cfn_id="@cfn_c_affine")
    lifted = _cfn("affine", _affine_body(rhs_const=6), cfn_id="@cfn_lifted_X")
    prog = _program_with_pair(src, lifted)

    new_prog, results = prove_decompile_lifts(
        prog, write_dir=tmp_path, rel_prefix="proofs/decompile_lift",
        write=True,
    )
    assert len(results) == 1
    r = results[0]
    assert r.status == "refuted"
    assert "literal value" in r.detail
    # No new witness landed.
    assert all(
        not (e.regime == "witness"
             and isinstance(e.justification, LiftEquivalence))
        for e in new_prog.equivalences
    )


def test_prove_is_idempotent_on_unchanged_input(tmp_path):
    """A second run sees the existing witness at the same hash and
    reports `current` rather than re-pinning."""
    src = _cfn("f", _affine_body(), cfn_id="@cfn_c_f")
    lifted = _cfn("f", _affine_body(), cfn_id="@cfn_lifted_f")
    prog = _program_with_pair(src, lifted)

    once, _ = prove_decompile_lifts(prog, write_dir=tmp_path, write=True)
    twice, results = prove_decompile_lifts(
        once, write_dir=tmp_path, write=True,
    )
    assert len(results) == 1
    assert results[0].status == "current"
    # Witness count unchanged.
    a = sum(1 for e in once.equivalences
            if e.regime == "witness"
            and isinstance(e.justification, LiftEquivalence))
    b = sum(1 for e in twice.equivalences
            if e.regime == "witness"
            and isinstance(e.justification, LiftEquivalence))
    assert a == b == 1


def test_prove_no_pair_yields_empty_results(tmp_path):
    """Program with neither side of the chain. No-op."""
    prog = Program()
    new_prog, results = prove_decompile_lifts(prog, write_dir=tmp_path)
    assert results == ()
    assert new_prog == prog


def test_witness_round_trips_through_save_load(tmp_path):
    src = _cfn("f", _affine_body(), cfn_id="@cfn_c_f")
    lifted = _cfn("f", _affine_body(), cfn_id="@cfn_lifted_f")
    prog, _ = prove_decompile_lifts(
        _program_with_pair(src, lifted),
        write_dir=tmp_path / "proofs", write=True,
    )
    out = tmp_path / "p.json"
    save_program(prog, out)
    loaded = load_program(out)
    assert loaded == prog
