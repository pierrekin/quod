"""Step 6 of the C-ingest redesign: pinned proof for c.for_general.

Pins the `c.for_general` rewrite's per-iteration equivalence in an SMT
artifact, sha-checked at lower time, re-checkable via `quod equiv
verify`. The artifact is package-distributed under
`src/quod/lower/c_family_proofs/` and resolved against the installed
package at verify time.

Tests cover:

  - Z3 returns `unsat` on the artifact.
  - The c-family lowering pass emits witness-regime FamilyLowering
    with the artifact's actual sha256.
  - The verify path passes for an unmodified artifact.
  - Hash tampering produces a clear mismatch failure.

Whole-loop equivalence remains the meta-theoretic inductive lift; the
artifact's header comment makes that obligation explicit.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from quod.cli import _verify_equivalence_justification
from quod.ingest.c import ingest_c
from quod.lower.c_family import _PROOFS_DIR, _RULE_PROOFS, lower_c_family
from quod.model import (
    Equivalence,
    FamilyLowering,
    Function,
    I32Type,
    LiftEquivalence,
    ManualJustification,
    Program,
)
from quod.predicate.proof import run_z3_on_file


SUM_C = Path(__file__).resolve().parents[1] / "examples/c_ingest/sum/sum.c"


def _z3_available() -> bool:
    return shutil.which("z3") is not None


def test_c_for_general_proof_artifact_exists():
    rel = _RULE_PROOFS["c.for_general"]
    full = _PROOFS_DIR.parent.parent / rel
    assert full.exists(), f"missing proof artifact: {full}"


@pytest.mark.skipif(not _z3_available(), reason="z3 not installed")
def test_c_for_general_proof_is_unsat():
    """The artifact must be unsat under Z3 — the per-iteration step
    of `for(init; cond; inc) body` is identical to the per-iteration
    step of `init; while(cond) { body; inc }`."""
    rel = _RULE_PROOFS["c.for_general"]
    full = _PROOFS_DIR.parent.parent / rel
    result = run_z3_on_file(full)
    assert result.status == "unsat"


def test_lowering_pins_artifact_with_correct_hash():
    """`c_family.lower` emits witness-regime FamilyLowering with
    the artifact path and a hash that matches the file on disk."""
    p = ingest_c(SUM_C)
    family = [
        e for e in p.equivalences
        if isinstance(e.justification, FamilyLowering)
        and e.justification.rule_name == "c.for_general"
    ]
    assert len(family) == 1
    eq = family[0]
    assert eq.regime == "witness"
    j = eq.justification
    assert j.artifact_path == "lower/c_family_proofs/c_for_general.smt2"
    # Hash must match the actual file content.
    pkg_root = _PROOFS_DIR.parent.parent
    expected = hashlib.sha256((pkg_root / j.artifact_path).read_bytes()).hexdigest()
    assert j.artifact_hash == expected


def test_identity_rule_witnesses_with_pinned_proof():
    """As of step-7's trivial-rule proofs, the `identity` rule pins
    `c_family_proofs/identity.smt2` (a reflexivity proof). Functions
    with no `c.*` extensions still emit a witness-regime equivalence
    so the data model is uniform across C-derived programs."""
    fn = Function(
        id="@fn_b_pure", name="pure", return_type=I32Type(),
        body=__import__("quod.model", fromlist=["Block"]).Block(
            stmts=(__import__("quod.model", fromlist=["ReturnExpr"]).ReturnExpr(
                value=__import__("quod.model", fromlist=["IntLit"]).IntLit(
                    type=I32Type(), value=0,
                ),
            ),),
        ),
    )
    out = lower_c_family(Program(structured_functions=(fn,)))
    eq = [e for e in out.equivalences if isinstance(e.justification, FamilyLowering)][0]
    assert eq.regime == "witness"
    assert eq.justification.rule_name == "identity"
    assert eq.justification.artifact_path == "lower/c_family_proofs/identity.smt2"
    assert eq.justification.artifact_hash is not None


@pytest.mark.skipif(not _z3_available(), reason="z3 not installed")
def test_verify_passes_for_unmodified_artifact():
    """The verify helper accepts a witness-regime FamilyLowering
    whose pinned hash matches the artifact and whose Z3 result is
    `unsat`."""
    p = ingest_c(SUM_C)
    family = [
        e for e in p.equivalences
        if isinstance(e.justification, FamilyLowering)
        and e.justification.rule_name == "c.for_general"
    ]
    ok, msg = _verify_equivalence_justification(family[0])
    assert ok, f"expected verify to pass, got: {msg}"


def test_verify_detects_hash_tampering():
    """Tampering with the `artifact_hash` (e.g. swapping in the proof
    of a different rule) is caught at verify time before Z3 runs."""
    eq = Equivalence(
        a_node_id="@fn_c_sum",
        b_node_id="@fn_c_lowered_sum",
        regime="witness",
        justification=FamilyLowering(
            rule_name="c.for_general",
            artifact_path="lower/c_family_proofs/c_for_general.smt2",
            artifact_hash="deadbeef" * 8,
        ),
    )
    ok, msg = _verify_equivalence_justification(eq)
    assert not ok
    assert "hash mismatch" in msg


def test_verify_handles_axiom_no_artifact():
    """A FamilyLowering claim with no artifact (an unproved rule) is
    not stale — the equivalence is asserted, not witnessed. Verify
    passes; it's the user's job to know which claims are unproven."""
    eq = Equivalence(
        a_node_id="@fn_b", b_node_id="@fn_c",
        justification=FamilyLowering(rule_name="identity"),
    )
    ok, msg = _verify_equivalence_justification(eq)
    assert ok, msg


def test_verify_detects_partial_artifact_pinning():
    """If only one of (artifact_path, artifact_hash) is set, the
    claim is structurally inconsistent and verify rejects it."""
    eq = Equivalence(
        a_node_id="@a", b_node_id="@b",
        justification=FamilyLowering(
            rule_name="custom",
            artifact_path="proofs/custom.smt2",
            # artifact_hash missing
        ),
    )
    ok, msg = _verify_equivalence_justification(eq)
    assert not ok
    assert "artifact_path" in msg or "artifact_hash" in msg


def test_verify_manual_justification():
    """Manual justification passes when signed_by is non-empty,
    fails otherwise — same contract as fn-level claim verify."""
    ok_eq = Equivalence(
        a_node_id="@a", b_node_id="@b",
        justification=ManualJustification(signed_by="alice", rationale="..."),
    )
    ok, _ = _verify_equivalence_justification(ok_eq)
    assert ok

    bad_eq = Equivalence(
        a_node_id="@a", b_node_id="@b",
        justification=ManualJustification(signed_by="   ", rationale="empty"),
    )
    ok, msg = _verify_equivalence_justification(bad_eq)
    assert not ok
    assert "signed_by" in msg


def test_verify_lift_equivalence_artifact_missing(tmp_path, monkeypatch):
    """LiftEquivalence resolves its artifact under the program's
    resolve_root (not the package). When the artifact doesn't exist,
    verify fails with a clear `artifact not found` message."""
    # Build a minimal Config rooted at tmp_path so the verify helper
    # resolves the artifact path there.
    from quod import cli as cli_mod
    from quod.config import Config, ProgramSpec
    cfg = Config(
        programs=(ProgramSpec(name="t", version="0.1", file="program.json"),),
        root=tmp_path,
    )
    monkeypatch.setitem(cli_mod._state, "config", cfg)
    monkeypatch.setitem(cli_mod._state, "config_path", tmp_path / "quod.toml")

    eq = Equivalence(
        a_node_id="@a", b_node_id="@b",
        regime="witness",
        justification=LiftEquivalence(
            artifact_path="proofs/missing.smt2",
            artifact_hash="00" * 32,
        ),
    )
    ok, msg = _verify_equivalence_justification(eq)
    assert not ok
    assert "artifact not found" in msg
