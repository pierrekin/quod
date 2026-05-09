"""End-to-end tests for `quod equiv prove` and `quod equiv verify`.

The CLI flows are exercised against a real tmp project (matching the
demo in `01-structured-quod.md`): write a `quod.toml`, ingest a C
file, and invoke the equiv subcommands. This catches CLI-level
regressions the unit tests miss (config wiring, path resolution,
stdout formatting, exit codes).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def _write_demo_project(root: Path) -> None:
    """Lay down a minimal `quod.toml` + `sum.c` under `root` matching
    the demo from 01-structured-quod.md."""
    (root / "sum.c").write_text(
        "int sum(int n) {\n"
        "    int s = 0;\n"
        "    for (int i = 0; i < n; i = i + 1) { s = s + i; }\n"
        "    return s;\n"
        "}\n"
    )
    (root / "quod.toml").write_text(
        'name = "demo"\n'
        'build_dir  = "build"\n'
        'proofs_dir = "proofs"\n'
        '\n'
        '[[program]]\n'
        'name = "demo"\n'
        'version = "0.1.0"\n'
        'file = "program.json"\n'
        '\n'
        '[[program.bin]]\n'
        'name = "sum_demo"\n'
        'entry = "sum"\n'
        '\n'
        '[[ingest.entry]]\n'
        'kind = "c-file"\n'
        'source = "sum.c"\n'
    )


def _quod(root: Path, *args: str) -> subprocess.CompletedProcess:
    """Invoke the quod CLI inside `root` via the in-process Typer
    runner. We pass through subprocess to mirror the demo flow exactly.
    """
    from typer.testing import CliRunner
    from quod import cli as cli_mod
    cli_mod._state.clear()
    runner = CliRunner()
    return runner.invoke(
        cli_mod.app,
        ["-c", str(root / "quod.toml"), *args],
    )


def test_ingest_emits_witnessed_a_to_b_claim(tmp_path):
    """`quod ingest` runs the lift-check post-merge so the saved
    program.json carries witness-regime LiftEquivalence claims for
    every (CFn, layer-B Function) pair."""
    _write_demo_project(tmp_path)
    result = _quod(tmp_path, "ingest")
    assert result.exit_code == 0, result.output

    program = json.loads((tmp_path / "program.json").read_text())
    eqs = program.get("equivalences", [])
    a_to_b = [
        e for e in eqs
        if e.get("justification", {}).get("kind") == "lift_equivalence"
    ]
    assert len(a_to_b) == 1
    assert a_to_b[0].get("regime") == "witness"
    assert a_to_b[0]["justification"]["artifact_path"] == "proofs/lift/sum.txt"
    # Artifact actually exists on disk.
    assert (tmp_path / "proofs" / "lift" / "sum.txt").exists()


def test_equiv_verify_passes_after_ingest(tmp_path):
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    result = _quod(tmp_path, "equiv", "verify")
    assert result.exit_code == 0, result.output
    # Both equivalences witnessed.
    assert "ok" in result.output


def test_equiv_prove_reports_status_per_claim(tmp_path):
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    result = _quod(tmp_path, "equiv", "prove")
    assert result.exit_code == 0, result.output
    # Both claims are witness-current. Verify the per-claim format
    # surfaces the regime and justification kind.
    assert "[witness/lift_equivalence]" in result.output
    assert "[witness/family_lowering]" in result.output


def test_equiv_prove_filters_by_function_name(tmp_path):
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0

    # Only the sum function exists; filter should keep both its claims.
    sum_only = _quod(tmp_path, "equiv", "prove", "sum")
    assert sum_only.exit_code == 0
    assert "@fn_c_sum" in sum_only.output

    # A name not in any equivalence reports the empty case.
    none = _quod(tmp_path, "equiv", "prove", "nonexistent_fn")
    assert none.exit_code == 0
    assert "no equivalence claims involving 'nonexistent_fn'" in none.output


def test_equiv_prove_marks_axiom_claims(tmp_path):
    """Hand-author a program with an axiom-regime A~B claim (no
    artifact) and confirm `equiv prove` reports it as `axiom` rather
    than failing or pretending it's witnessed."""
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0

    # Demote the LiftEquivalence claim back to ManualJustification.
    program = json.loads((tmp_path / "program.json").read_text())
    for eq in program["equivalences"]:
        if eq.get("justification", {}).get("kind") == "lift_equivalence":
            eq["justification"] = {
                "kind": "manual",
                "signed_by": "quod.ingest.c",
                "rationale": "demoted for test",
            }
            eq.pop("regime", None)  # axiom is default
    (tmp_path / "program.json").write_text(json.dumps(program, indent=2))

    result = _quod(tmp_path, "equiv", "prove")
    assert result.exit_code == 0  # axiom is not a failure on its own
    assert "axiom" in result.output
    assert "[axiom/manual]" in result.output


def test_equiv_prove_bump_upgrades_axiom_to_witness(tmp_path):
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0

    # Demote, then --bump.
    program = json.loads((tmp_path / "program.json").read_text())
    for eq in program["equivalences"]:
        if eq.get("justification", {}).get("kind") == "lift_equivalence":
            eq["justification"] = {
                "kind": "manual",
                "signed_by": "quod.ingest.c",
                "rationale": "demoted",
            }
            eq.pop("regime", None)
    (tmp_path / "program.json").write_text(json.dumps(program, indent=2))

    bumped = _quod(tmp_path, "equiv", "prove", "--bump")
    assert bumped.exit_code == 0
    # Re-load and confirm the claim is witnessed again.
    program2 = json.loads((tmp_path / "program.json").read_text())
    a_to_b = [
        e for e in program2["equivalences"]
        if e.get("justification", {}).get("kind") == "lift_equivalence"
    ]
    assert len(a_to_b) == 1
    assert a_to_b[0]["regime"] == "witness"


def test_equiv_prove_detects_in_memory_drift(tmp_path):
    """Hand-edit a layer-A node so the in-memory walk would hash
    differently than the pinned LiftEquivalence. `equiv prove` reports
    this as stale even when the on-disk artifact is unchanged."""
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0

    program = json.loads((tmp_path / "program.json").read_text())
    # Bump an int literal in the layer-A subtree.
    def walk(n):
        if isinstance(n, dict):
            if n.get("kind") == "c.lit_int":
                n["value"] = n["value"] + 100
                return True
            for v in n.values():
                if walk(v): return True
        elif isinstance(n, list):
            for x in n:
                if walk(x): return True
        return False
    walk(program["source_units"])
    (tmp_path / "program.json").write_text(json.dumps(program, indent=2))

    result = _quod(tmp_path, "equiv", "prove")
    assert result.exit_code == 1  # stale claim is a failure
    assert "FAIL" in result.output


def test_equiv_verify_detects_artifact_tampering(tmp_path):
    """Tampering with the on-disk lift-check artifact (after a
    successful ingest + verify) is caught by hash mismatch."""
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    art = tmp_path / "proofs" / "lift" / "sum.txt"
    art.write_text(art.read_text().replace('"name": "sum"', '"name": "tampered"'))
    result = _quod(tmp_path, "equiv", "verify")
    assert result.exit_code == 1
    assert "hash mismatch" in result.output
