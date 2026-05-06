"""A→B lift checker — structural correspondence between layer A and B.

Tests cover the two halves of the work:

  - `walk_lift` / `lift_check_artifact` — pure structural walk over
    a (CFn, Function) pair, raising LiftCheckError on divergence.
  - `prove_lifts` — program-level upgrader that promotes
    `ManualJustification` A~B claims to witnessed `LiftEquivalence`,
    deduplicating per-pair so re-ingest doesn't grow the claim list.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from quod.ingest.c import ingest_c
from quod.lift_check import (
    LiftCheckError,
    lift_check_artifact,
    lift_check_hash,
    prove_lifts,
    walk_lift,
)
from quod.model import (
    Assign,
    BinOp,
    Block,
    CAssign,
    CBinOp,
    CFn,
    CIntLit,
    CParam,
    CReturn,
    CType,
    CVarDecl,
    CVarRef,
    Equivalence,
    Function,
    I32Type,
    IntLit,
    Let,
    LiftEquivalence,
    LocalRef,
    ManualJustification,
    Param,
    ParamRef,
    Program,
    ReturnExpr,
)


SUM_C = Path(__file__).resolve().parents[1] / "examples/c_ingest/sum/sum.c"


# ---------- walk_lift on a real ingest ----------


def test_walk_lift_succeeds_on_sum_c():
    p = ingest_c(SUM_C)
    cfn = p.source_units[0].functions[0]
    fn = p.structured_functions[0]
    record = walk_lift(cfn, fn)
    assert record["kind"] == "lift-check"
    assert record["fn"]["name"] == "sum"


def test_lift_check_artifact_is_deterministic():
    p1 = ingest_c(SUM_C)
    p2 = ingest_c(SUM_C)
    cfn1 = p1.source_units[0].functions[0]
    fn1 = p1.structured_functions[0]
    cfn2 = p2.source_units[0].functions[0]
    fn2 = p2.structured_functions[0]
    assert lift_check_artifact(cfn1, fn1) == lift_check_artifact(cfn2, fn2)
    assert lift_check_hash(cfn1, fn1) == lift_check_hash(cfn2, fn2)


# ---------- walk_lift on hand-built mismatches ----------


def _trivial_pair():
    """A minimal (CFn, Function) pair that walks cleanly: `int f(int x)
    { return x; }`."""
    int_t = CType(name="int")
    cfn = CFn(
        id="@cfn_x", name="f", return_type=int_t,
        params=(CParam(name="x", type=int_t),),
        body=(CReturn(value=CVarRef(name="x")),),
    )
    fn = Function(
        id="@fn_x", name="f", return_type=I32Type(),
        params=(Param(name="x", type=I32Type()),),
        body=Block(stmts=(ReturnExpr(value=ParamRef(name="x")),)),
    )
    return cfn, fn


def test_walk_lift_rejects_function_name_mismatch():
    cfn, fn = _trivial_pair()
    fn = fn.model_copy(update={"name": "g"})
    with pytest.raises(LiftCheckError, match="function name mismatch"):
        walk_lift(cfn, fn)


def test_walk_lift_rejects_param_count_mismatch():
    cfn, fn = _trivial_pair()
    fn = fn.model_copy(update={"params": ()})
    with pytest.raises(LiftCheckError, match="param count"):
        walk_lift(cfn, fn)


def test_walk_lift_rejects_param_name_mismatch():
    cfn, fn = _trivial_pair()
    fn = fn.model_copy(update={"params": (Param(name="y", type=I32Type()),)})
    with pytest.raises(LiftCheckError, match="param name"):
        walk_lift(cfn, fn)


def test_walk_lift_rejects_int_lit_value_mismatch():
    int_t = CType(name="int")
    cfn = CFn(
        id="@cfn_z", name="z", return_type=int_t,
        body=(CReturn(value=CIntLit(value=42)),),
    )
    fn = Function(
        id="@fn_z", name="z", return_type=I32Type(),
        body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=99)),)),
    )
    with pytest.raises(LiftCheckError, match="int_lit value 42 vs 99"):
        walk_lift(cfn, fn)


def test_walk_lift_rejects_operator_mismatch():
    """`+` in layer A must map to `add` in layer B; `sub` would fail."""
    int_t = CType(name="int")
    cfn = CFn(
        id="@cfn_q", name="q", return_type=int_t,
        params=(CParam(name="x", type=int_t),),
        body=(CReturn(value=CBinOp(
            op="+", lhs=CVarRef(name="x"), rhs=CIntLit(value=1),
        )),),
    )
    fn = Function(
        id="@fn_q", name="q", return_type=I32Type(),
        params=(Param(name="x", type=I32Type()),),
        body=Block(stmts=(ReturnExpr(value=BinOp(
            op="sub", lhs=ParamRef(name="x"),
            rhs=IntLit(type=I32Type(), value=1),
        )),)),
    )
    with pytest.raises(LiftCheckError, match=r"operator '\+' expects layer-B 'add'"):
        walk_lift(cfn, fn)


def test_walk_lift_rejects_kind_mismatch():
    """A layer-A `return e` must map to layer-B ReturnExpr, not Assign."""
    int_t = CType(name="int")
    cfn = CFn(
        id="@cfn_r", name="r", return_type=int_t,
        body=(CReturn(value=CIntLit(value=0)),),
    )
    fn = Function(
        id="@fn_r", name="r", return_type=I32Type(),
        body=Block(stmts=(Assign(name="x", value=IntLit(type=I32Type(), value=0)),)),
    )
    with pytest.raises(LiftCheckError, match="return e.* vs layer-B Assign"):
        walk_lift(cfn, fn)


def test_walk_lift_strips_synthesized_fall_through():
    """When the layer-A body terminates explicitly but the C ingester
    appends a tail Unreachable / `return 0` (per C99 fall-through
    semantics), the walk should match the layer-A body against the
    layer-B body minus the synthesized stub."""
    from quod.model import Unreachable
    int_t = CType(name="int")
    cfn = CFn(
        id="@cfn_t", name="t", return_type=int_t,
        body=(CReturn(value=CIntLit(value=1)),),
    )
    fn = Function(
        id="@fn_t", name="t", return_type=I32Type(),
        body=Block(stmts=(
            ReturnExpr(value=IntLit(type=I32Type(), value=1)),
            Unreachable(),  # synthesized tail
        )),
    )
    # No exception.
    walk_lift(cfn, fn)


# ---------- prove_lifts ----------


def test_prove_lifts_upgrades_manual_to_witness(tmp_path):
    p = ingest_c(SUM_C)
    # Demote the A~B claim to manual to simulate a hand-authored program.
    new_eqs = []
    for eq in p.equivalences:
        if isinstance(eq.justification, LiftEquivalence):
            new_eqs.append(eq.model_copy(update={
                "regime": "axiom",
                "justification": ManualJustification(
                    signed_by="quod.ingest.c", rationale="...",
                ),
            }))
        else:
            new_eqs.append(eq)
    p_demoted = p.model_copy(update={"equivalences": tuple(new_eqs)})

    proven = prove_lifts(
        p_demoted,
        write_dir=tmp_path / "lift",
        rel_prefix="proofs/lift",
        write=True,
    )

    upgraded = [
        eq for eq in proven.equivalences
        if isinstance(eq.justification, LiftEquivalence)
    ]
    assert len(upgraded) == 1
    assert upgraded[0].regime == "witness"
    assert upgraded[0].justification.artifact_path == "proofs/lift/sum.txt"
    # Artifact written to disk.
    assert (tmp_path / "lift" / "sum.txt").exists()


def test_prove_lifts_dedups_per_pair(tmp_path):
    """Re-ingesting the same source produces a fresh ManualJustification
    claim that overlaps with the existing witnessed one. prove_lifts
    must collapse them, otherwise the equivalences list grows on every
    re-ingest."""
    p = ingest_c(SUM_C)  # already has 1 LiftEquivalence claim
    # Inject a duplicate manual claim (simulating merge-after-re-ingest).
    dup = Equivalence(
        a_node_id="@cfn_c_sum", b_node_id="@fn_c_sum",
        justification=ManualJustification(
            signed_by="quod.ingest.c", rationale="re-ingest",
        ),
    )
    p_with_dup = p.model_copy(update={
        "equivalences": p.equivalences + (dup,),
    })

    out = prove_lifts(
        p_with_dup,
        write_dir=tmp_path / "lift",
        rel_prefix="proofs/lift",
    )

    a_to_b = [
        eq for eq in out.equivalences
        if eq.a_node_id == "@cfn_c_sum" and eq.b_node_id == "@fn_c_sum"
    ]
    assert len(a_to_b) == 1, f"expected exactly one A~B claim, got {len(a_to_b)}"
    assert isinstance(a_to_b[0].justification, LiftEquivalence)


def test_prove_lifts_preserves_current_pin(tmp_path):
    """Running prove_lifts twice with the same write_dir is idempotent
    — the second run sees an already-witnessed claim with a current
    hash and leaves it alone."""
    p = ingest_c(SUM_C)
    once = prove_lifts(
        p, write_dir=tmp_path / "lift", rel_prefix="proofs/lift",
    )
    twice = prove_lifts(
        once, write_dir=tmp_path / "lift", rel_prefix="proofs/lift",
    )
    # Same equivalences, identical model_dump.
    assert once.model_dump_json() == twice.model_dump_json()


def test_prove_lifts_dry_run_does_not_write(tmp_path):
    """`write=False` lets `equiv prove` report what would happen
    without touching disk."""
    p = ingest_c(SUM_C)
    new_eqs = tuple(
        eq.model_copy(update={
            "regime": "axiom",
            "justification": ManualJustification(
                signed_by="quod.ingest.c", rationale="...",
            ),
        }) if isinstance(eq.justification, LiftEquivalence) else eq
        for eq in p.equivalences
    )
    p_demoted = p.model_copy(update={"equivalences": new_eqs})

    out = prove_lifts(
        p_demoted,
        write_dir=tmp_path / "lift",
        rel_prefix="proofs/lift",
        write=False,
    )
    # Claim was upgraded, but no file on disk.
    assert any(isinstance(eq.justification, LiftEquivalence) for eq in out.equivalences)
    assert not (tmp_path / "lift" / "sum.txt").exists()


def test_prove_lifts_skips_orphaned_pairs(tmp_path):
    """If a claim references an a_node_id or b_node_id that's not in
    the program (e.g. corpus drift after a manual edit), prove_lifts
    leaves the claim alone — `equiv verify` reports the broken pair
    separately."""
    int_t = CType(name="int")
    cfn = CFn(id="@cfn_real", name="real", return_type=int_t,
              body=(CReturn(value=CIntLit(value=0)),))
    fn = Function(id="@fn_real", name="real", return_type=I32Type(),
                  body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)))
    from quod.model import CUnit
    program = Program(
        source_units=(CUnit(source_path="real.c", functions=(cfn,)),),
        structured_functions=(fn,),
        equivalences=(
            # Real pair — should be witnessed.
            Equivalence(
                a_node_id="@cfn_real", b_node_id="@fn_real",
                justification=ManualJustification(
                    signed_by="quod.ingest.c", rationale="...",
                ),
            ),
            # Orphan pair — endpoints not in program.
            Equivalence(
                a_node_id="@cfn_ghost", b_node_id="@fn_ghost",
                justification=ManualJustification(
                    signed_by="quod.ingest.c", rationale="...",
                ),
            ),
        ),
    )
    out = prove_lifts(
        program,
        write_dir=tmp_path / "lift",
        rel_prefix="proofs/lift",
        write=True,
    )
    # Two claims survive: one upgraded, one passed through unchanged.
    assert len(out.equivalences) == 2
    real = [eq for eq in out.equivalences if eq.a_node_id == "@cfn_real"][0]
    ghost = [eq for eq in out.equivalences if eq.a_node_id == "@cfn_ghost"][0]
    assert isinstance(real.justification, LiftEquivalence)
    assert isinstance(ghost.justification, ManualJustification)
