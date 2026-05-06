"""Tests for the body-drift integrity check on Z3Justification.

The justification carries two hashes that are equal at prove time but
answer different questions at verify time:

  artifact_hash  — sha256 of the .smt2 file on disk; detects file tamper.
  body_smt_hash  — sha256 of the SMT text the current body+claim *would*
                   produce; detects body drift.

Each test exercises one of those modes (happy / file-tamper / body-drift /
SMT-stable cosmetic edit).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from quod.cli import _verify_justification
from quod.model import (
    I32Type,
    IntLit,
    Function,
    Param,
    Program,
    ReturnExpr,
    ReturnInRangeClaim,
    add_claim,
    replace_function,

    Block,
)
from quod.providers import ClaimRequest, _z3_qf_lia_prove


_I32 = I32Type()


pytestmark = pytest.mark.skipif(
    shutil.which("z3") is None, reason="z3 not installed"
)


def _make_program() -> Program:
    """A function f(x: i32) -> i32 that returns the literal 5."""
    f = Function(
        name="f",
        params=(Param(name="x", type=_I32),),
        return_type=_I32,
        body=Block(stmts=(ReturnExpr(value=IntLit(type=_I32, value=5)),)),
    )
    return Program(functions=(f,))


def _prove_return_ge_zero(program: Program, proofs_dir: Path) -> Program:
    req = ClaimRequest(function="f", kind="return_in_range", target=None, min=0, max=None)
    result = _z3_qf_lia_prove(program, req, proofs_dir)
    assert result.status == "proven", f"expected proven, got {result.status}: {result.detail}"
    assert result.claim is not None
    return add_claim(program, "f", result.claim)


def test_happy_path(tmp_path: Path) -> None:
    """Prove + immediately verify: both hashes match, verify passes."""
    program = _prove_return_ge_zero(_make_program(), tmp_path / "proofs")
    fn = program.functions[0]
    claim = fn.claims[0]
    assert claim.justification is not None
    ok, msg = _verify_justification(claim.justification, tmp_path, fn, claim, program)
    assert ok, f"verify failed unexpectedly: {msg}"


def test_file_tamper_detected(tmp_path: Path) -> None:
    """Mutating the .smt2 bytes on disk → 'hash mismatch'."""
    program = _prove_return_ge_zero(_make_program(), tmp_path / "proofs")
    fn = program.functions[0]
    claim = fn.claims[0]
    assert claim.justification is not None
    smt_path = tmp_path / claim.justification.artifact_path
    smt_path.write_text(smt_path.read_text() + "\n; tampered\n")

    ok, msg = _verify_justification(claim.justification, tmp_path, fn, claim, program)
    assert not ok
    assert "hash mismatch" in msg


def test_body_drift_detected(tmp_path: Path) -> None:
    """Edit the body so the SMT meaning changes; verify must reject the
    proof BEFORE running Z3 (the .smt2 file is still on-disk-valid; the
    drift is in what the body now means)."""
    program = _prove_return_ge_zero(_make_program(), tmp_path / "proofs")
    fn = program.functions[0]
    claim = fn.claims[0]
    assert claim.justification is not None

    # Replace the function body: was `return 5`, now `return -5`. The
    # claim (return >= 0) no longer holds; the on-disk .smt2 is unchanged
    # because the file still corresponds to the *old* body, but the SMT
    # we'd generate from the *current* body is different.
    drifted_fn = fn.model_copy(update={
        "body": Block(stmts=(ReturnExpr(value=IntLit(type=_I32, value=-5)),)),
    })
    drifted_program = replace_function(program, drifted_fn)

    ok, msg = _verify_justification(
        claim.justification, tmp_path, drifted_fn, claim, drifted_program,
    )
    assert not ok
    assert "body changed since proof" in msg


def test_cosmetic_edit_does_not_drift(tmp_path: Path) -> None:
    """Mutate `notes` (a developer-facing field that doesn't enter the
    SMT lowering). body_smt_hash should be stable; verify still passes."""
    program = _prove_return_ge_zero(_make_program(), tmp_path / "proofs")
    fn = program.functions[0]
    claim = fn.claims[0]
    assert claim.justification is not None

    # `notes` doesn't affect anything `goal_smt_lib` reads.
    cosmetic_fn = fn.model_copy(update={"notes": ("a note that didn't exist before",)})
    cosmetic_program = replace_function(program, cosmetic_fn)
    cosmetic_claim = cosmetic_fn.claims[0]

    ok, msg = _verify_justification(
        cosmetic_claim.justification, tmp_path, cosmetic_fn, cosmetic_claim,
        cosmetic_program,
    )
    assert ok, f"cosmetic edit should not invalidate: {msg}"
