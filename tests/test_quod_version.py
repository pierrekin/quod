"""Program-level `quod_version` stamping and verification.

`Program.quod_version` records "the version of quod that produced
the pinned claims in this Program." Verification compares against
the running build; mismatch (including `None` on either side) means
the pins can't be trusted and verification fails until re-pinned.

Tests cover the matrix:

  - Set: `stamp_quod_version` round-trips through JSON.
  - Match: pinned program at version X verified against current X
    passes.
  - Mismatch: pinned program at X verified against Y fails with a
    clear message.
  - None on stamp side (program has pins but no version): fails.
  - None on current side (running quod can't identify itself): fails.
  - Empty program (no pinned claims): version check is a no-op,
    passes regardless of either side being None.

The conftest's autouse fixture stubs `current_quod_version` to a
stable value for the rest of the suite; this file overrides that
stub per-test to drive each branch.
"""
from __future__ import annotations

import json

import pytest

from quod import version as quod_version
from quod.predicate.predicate_canonical import predicate_for_param_range
from quod.model import (
    Block,
    Equivalence,
    FamilyLowering,
    Function,
    I32Type,
    IntLit,
    ManualJustification,
    Param,
    PredicateClaim,
    Program,
    ReturnExpr,
    Z3Justification,
)
from quod.version import (
    check_program_version,
    current_quod_version,
    program_has_pinned_claims,
    stamp_quod_version,
)


def _set_current(monkeypatch, value):
    """Override the `current_quod_version()` source-of-truth for one
    test. Must be called *after* the autouse fixture has already
    primed the stub, so we replace it cleanly."""
    monkeypatch.setattr(quod_version, "_compute_quod_version", lambda: value)
    quod_version.reset_cache()


# ---------- model field ----------


def test_quod_version_round_trips_through_json():
    p = Program(quod_version="abc123")
    raw = p.model_dump_json()
    assert json.loads(raw)["quod_version"] == "abc123"
    loaded = Program.model_validate_json(raw)
    assert loaded.quod_version == "abc123"


def test_quod_version_drops_when_none():
    """Default `None` doesn't bloat existing JSON corpora."""
    p = Program()
    decoded = json.loads(p.model_dump_json())
    assert "quod_version" not in decoded


# ---------- stamp_quod_version ----------


def test_stamp_writes_current_version(monkeypatch):
    _set_current(monkeypatch, "v-1234")
    p = Program()
    stamped = stamp_quod_version(p)
    assert stamped.quod_version == "v-1234"


def test_stamp_writes_none_when_current_is_none(monkeypatch):
    _set_current(monkeypatch, None)
    p = Program()
    stamped = stamp_quod_version(p)
    assert stamped.quod_version is None


def test_stamp_does_not_mutate_input(monkeypatch):
    _set_current(monkeypatch, "x")
    p = Program(quod_version="orig")
    _ = stamp_quod_version(p)
    assert p.quod_version == "orig"  # input unchanged


# ---------- program_has_pinned_claims ----------


def _fn_with_z3() -> Function:
    return Function(
        name="f",
        params=(Param(name="x", type=I32Type()),),
        return_type=I32Type(),
        body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
        claims=(PredicateClaim(
            regime="witness",
            expr=predicate_for_param_range("x", I32Type(), lo=0, hi=None),
            justification=Z3Justification(
                artifact_path="proofs/x.smt2",
                artifact_hash="deadbeef",
                body_smt_hash="cafef00d",
            ),
        ),),
    )


def test_pin_detection_recognises_function_claims():
    assert program_has_pinned_claims(Program(functions=(_fn_with_z3(),)))


def test_pin_detection_recognises_equivalence_claims():
    eq = Equivalence(
        a_node_id="@a", b_node_id="@b",
        justification=ManualJustification(signed_by="x", rationale="y"),
    )
    assert program_has_pinned_claims(Program(equivalences=(eq,)))


def test_pin_detection_negative_for_empty_program():
    assert not program_has_pinned_claims(Program())


def test_pin_detection_negative_for_axiom_no_justification():
    fn = Function(
        name="f", params=(Param(name="x", type=I32Type()),),
        return_type=I32Type(),
        body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
        claims=(PredicateClaim(
            expr=predicate_for_param_range("x", I32Type(), lo=0, hi=None),
        ),),  # no justification
    )
    assert not program_has_pinned_claims(Program(functions=(fn,)))


# ---------- check_program_version ----------


def test_check_passes_when_versions_match(monkeypatch):
    _set_current(monkeypatch, "v-X")
    p = Program(functions=(_fn_with_z3(),), quod_version="v-X")
    ok, msg = check_program_version(p)
    assert ok and msg == ""


def test_check_fails_when_versions_differ(monkeypatch):
    _set_current(monkeypatch, "v-Y")
    p = Program(functions=(_fn_with_z3(),), quod_version="v-X")
    ok, msg = check_program_version(p)
    assert not ok
    assert "v-X"[:12] in msg or "v-Y"[:12] in msg
    assert "re-pin" in msg


def test_check_fails_when_program_has_no_version(monkeypatch):
    """Pinned program with `quod_version=None` always fails — strict
    rule."""
    _set_current(monkeypatch, "v-X")
    p = Program(functions=(_fn_with_z3(),), quod_version=None)
    ok, msg = check_program_version(p)
    assert not ok
    assert "no quod_version" in msg


def test_check_fails_when_running_version_unknown(monkeypatch):
    """Running quod returning `None` (no version available) fails
    even when the program has a stamped version."""
    _set_current(monkeypatch, None)
    p = Program(functions=(_fn_with_z3(),), quod_version="v-X")
    ok, msg = check_program_version(p)
    assert not ok
    assert "running quod" in msg


def test_check_passes_for_empty_program_regardless_of_version(monkeypatch):
    """Programs with zero pinned claims have nothing to verify;
    version state doesn't matter."""
    _set_current(monkeypatch, None)
    ok, _ = check_program_version(Program())
    assert ok
    ok, _ = check_program_version(Program(quod_version="anything"))
    assert ok
    _set_current(monkeypatch, "v-X")
    ok, _ = check_program_version(Program())
    assert ok


# ---------- end-to-end via CLI ----------


def test_equiv_verify_fails_on_version_mismatch_via_cli(tmp_path, monkeypatch):
    """Pinned program at v-old, running at v-new — `quod equiv verify`
    refuses with a clear stderr message."""
    import subprocess
    from typer.testing import CliRunner
    from quod import cli as cli_mod

    # Stage 1: set version "v-old", ingest a real C program (which
    # produces witness-regime pinned claims and stamps the program).
    _set_current(monkeypatch, "v-old")

    (tmp_path / "sum.c").write_text(
        "int sum(int n) { int s = 0; for (int i = 0; i < n; i = i + 1) "
        "{ s = s + i; } return s; }\n"
    )
    (tmp_path / "quod.toml").write_text(
        'build_dir = "build"\n'
        'proofs_dir = "proofs"\n'
        '\n'
        '[[program]]\n'
        'name = "demo"\n'
        'version = "0.1"\n'
        'file = "program.json"\n'
        '\n'
        '[[ingest.entry]]\n'
        'kind = "c-file"\n'
        'source = "sum.c"\n'
    )

    def _run(*args):
        cli_mod._state.clear()
        runner = CliRunner()
        return runner.invoke(
            cli_mod.app, ["-c", str(tmp_path / "quod.toml"), *args],
        )

    assert _run("ingest").exit_code == 0
    program_json = json.loads((tmp_path / "program.json").read_text())
    assert program_json.get("quod_version") == "v-old"

    # Stage 2: bump running version to "v-new", verify should fail.
    _set_current(monkeypatch, "v-new")
    result = _run("equiv", "verify")
    assert result.exit_code == 1
    assert "v-old"[:12] in result.output
    assert "v-new"[:12] in result.output

    # Stage 3: --bump should refresh and stamp current version.
    bump = _run("equiv", "prove", "--bump")
    assert bump.exit_code == 0
    program_json2 = json.loads((tmp_path / "program.json").read_text())
    assert program_json2.get("quod_version") == "v-new"

    # Verify should now pass.
    assert _run("equiv", "verify").exit_code == 0
