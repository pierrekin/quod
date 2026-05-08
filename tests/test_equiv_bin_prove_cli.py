"""End-to-end tests for `quod equiv prove --bump` with bin-relational.

The CLI flow is exercised against a real tmp project: write a
`quod.toml`, author a tiny C function, compile to .so, ingest both,
and invoke `equiv prove --bump`. Confirms that the binary ingester's
auto-lifts (signature_binding + decompile_lift) plus the prove
command's z3.bin_relational pass jointly produce a Z3-witnessed
bin~src equivalence on the saved program.json.

Skipped when Ghidra (or PyGhidra, or clang, or z3) isn't available.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


pytest.importorskip("pyghidra", reason="pyghidra not installed; skip real-Ghidra CLI test")


def _ghidra_dir() -> str | None:
    if env := os.environ.get("GHIDRA_INSTALL_DIR"):
        return env if Path(env).is_dir() else None
    for candidate in ("/opt/ghidra", "/usr/share/ghidra", "/usr/local/ghidra"):
        if Path(candidate).is_dir():
            return candidate
    return None


_GHIDRA_DIR = _ghidra_dir()
if _GHIDRA_DIR is None:
    pytest.skip("no Ghidra install found", allow_module_level=True)
os.environ.setdefault("GHIDRA_INSTALL_DIR", _GHIDRA_DIR)


_NEED_TOOLS = ("clang", "z3")
for _tool in _NEED_TOOLS:
    if shutil.which(_tool) is None:
        pytest.skip(f"{_tool} not on PATH", allow_module_level=True)


# Source kept simple — the v0 prover only handles straight-line int
# arithmetic, and `-O1` keeps params in registers (no stack spills
# the encoder doesn't model).
#
# Two functions: `add` (Ghidra preserves operand order, so the
# strict tree-walk passes cleanly — useful for the witness-emission
# tests) and `affine` (Ghidra reorders `3 * x` to `x * 3` during
# decompile, so the strict walk refuses — useful for refutation
# tests AND to demonstrate that "Ghidra recovered something
# structurally different" surfaces clearly).
_C_SOURCE = """\
int add(int a, int b) { return a + b; }
int affine(int x) { return 3 * x + 5; }
"""

_C_SOURCE_MUTATED = """\
int add(int a, int b) { return a + b; }
int affine(int x) { return 3 * x + 6; }
"""


def _write_project(root: Path, c_source: str = _C_SOURCE) -> None:
    (root / "poc.c").write_text(c_source)
    (root / "quod.toml").write_text(
        'build_dir  = "build"\n'
        'proofs_dir = "proofs"\n'
        '\n'
        '[[program]]\n'
        'name = "demo"\n'
        'version = "0.1.0"\n'
        'file = "program.json"\n'
        '\n'
        '[[program.bin]]\n'
        'name = "affine"\n'
        'entry = "affine"\n'
    )


def _build_so(root: Path) -> Path:
    so = root / "libpoc.so"
    subprocess.run(
        ["clang", "-O1", "-g", "-shared", "-fPIC",
         "-o", str(so), str(root / "poc.c")],
        check=True, capture_output=True, text=True,
    )
    return so


def _quod(root: Path, *args: str) -> "subprocess.CompletedProcess":
    """Invoke the CLI in-process via Typer's runner."""
    from typer.testing import CliRunner
    from quod import cli as cli_mod
    cli_mod._state.clear()
    runner = CliRunner()
    return runner.invoke(
        cli_mod.app,
        ["-c", str(root / "quod.toml"), *args],
    )


def _ingest_into(work: Path) -> None:
    """Helper used by every test: write the project, build the .so,
    ingest both. Function-scoped because conftest's `_stub_quod_version`
    is also function-scoped — a module-scoped fixture would run *before*
    the version stub and the C ingest would stamp None, tripping the
    pinned-claims-without-version check on later commands."""
    _write_project(work)
    so = _build_so(work)
    r = _quod(work, "ingest", "c", str(work / "poc.c"))
    assert r.exit_code == 0, r.output
    r = _quod(work, "ingest", "binary", str(so))
    assert r.exit_code == 0, r.output


def test_ingest_binary_emits_signature_bindings_and_decompile_lifts(tmp_path):
    """The binary ingester runs both structural lifts inline. After
    `quod ingest binary`, the saved program.json carries
    `signature_bindings` and a `decompile_lift`-justified equivalence
    for the affine function."""
    _ingest_into(tmp_path)
    program = json.loads((tmp_path / "program.json").read_text())

    # signature_bindings populated.
    sigs = program.get("signature_bindings", [])
    affine_sig = next(
        (sb for sb in sigs if "affine" in sb["src_fn_id"]), None,
    )
    assert affine_sig is not None, (
        f"no signature_binding for affine; got {[sb['src_fn_id'] for sb in sigs]}"
    )
    assert affine_sig["abi"] == "x86_64-sysv"

    # DecompileLift equivalence with hash present. The bin.fn's id is
    # an opaque hash, but we can find it by demangled_name in the
    # binary_units subtree.
    eqs = program.get("equivalences", [])
    bin_units = program.get("binary_units", [])
    affine_bin_id = next(
        f["id"] for u in bin_units for f in u.get("functions", [])
        if f.get("demangled_name") == "affine"
    )
    decompile_eqs = [
        e for e in eqs
        if e.get("justification", {}).get("kind") == "decompile_lift"
        and e["b_node_id"] == affine_bin_id
    ]
    assert len(decompile_eqs) == 1
    j = decompile_eqs[0]["justification"]
    assert len(j["decompile_text_sha256"]) == 64


def test_equiv_prove_bump_emits_decompile_lift_witness(tmp_path):
    """`equiv prove --bump` walks (src_cfn, lifted_cfn) pairs and
    emits a `LiftEquivalence`-witnessed equivalence on a match.

    Tested on `add` because Ghidra preserves `a + b` operand order
    when decompiling — the strict tree-walk succeeds. (`affine` is
    `3 * x + 5` in source but `x * 3 + 5` in Ghidra's decompile,
    so the strict walk for that function REFUSES — see the next
    test.)"""
    _ingest_into(tmp_path)

    bumped = _quod(tmp_path, "equiv", "prove", "--bump")
    assert bumped.exit_code == 0, bumped.output
    assert "[decompile-lift]" in bumped.output

    program = json.loads((tmp_path / "program.json").read_text())
    bin_units = program["binary_units"]
    add_lifted_id = next(
        c["id"] for u in bin_units for c in u.get("lifted_cfns", [])
        if c["name"] == "add"
    )
    add_src_id = next(
        c["id"] for u in program["source_units"]
        for c in u["functions"]
        if c["name"] == "add"
    )
    matches = [
        e for e in program["equivalences"]
        if e.get("justification", {}).get("kind") == "lift_equivalence"
        and e["a_node_id"] == add_src_id
        and e["b_node_id"] == add_lifted_id
    ]
    assert len(matches) == 1
    eq = matches[0]
    assert eq["regime"] == "witness"
    artifact_path = tmp_path / eq["justification"]["artifact_path"]
    assert artifact_path.exists()
    blob = json.loads(artifact_path.read_text())
    assert blob["rule"] == "c.cfn_correspondence"


def test_equiv_prove_bump_decompile_lift_refutes_on_structural_drift(tmp_path):
    """`affine` source is `3 * x + 5` but Ghidra decompiles as
    `x * 3 + 5` (operand-order swap). The strict tree-walk refuses
    on this structural difference — REFUTED with a path locating
    where the kinds first diverge. This is the deliverable: "Ghidra
    recovered something structurally different from the source."
    """
    _ingest_into(tmp_path)

    bumped = _quod(tmp_path, "equiv", "prove", "--bump")
    assert bumped.exit_code == 0, bumped.output
    assert "REFUTED" in bumped.output
    assert "[decompile-lift]" in bumped.output
    # The path fingerprints the body location AND names the kind
    # of divergence (kind / literal / var-ref / etc.).
    assert "fn[affine].body" in bumped.output


def test_equiv_prove_reports_axiom_for_unproven_bin_eqs(tmp_path):
    """Before --bump, the bin~src equivalences are still axioms
    (BinaryProvenance, DecompileLift). `equiv prove` reports them
    as `axiom` rather than failing."""
    _ingest_into(tmp_path)
    result = _quod(tmp_path, "equiv", "prove", "affine")
    assert result.exit_code == 0, result.output
    assert "[axiom/binary_provenance]" in result.output
    assert "[axiom/decompile_lift]" in result.output


def test_equiv_prove_bump_upgrades_bin_src_to_z3_witness(tmp_path):
    """Run `equiv prove --bump` and confirm a fresh Z3-witnessed
    equivalence between the source `Function` and the `BinFunction`
    is added, with an .smt2 artifact persisted under proofs_dir."""
    _ingest_into(tmp_path)

    bumped = _quod(tmp_path, "equiv", "prove", "--bump")
    assert bumped.exit_code == 0, bumped.output
    assert "ok" in bumped.output  # the bin_relational badge
    assert "[bin_relational]" in bumped.output

    program = json.loads((tmp_path / "program.json").read_text())
    z3_eqs = [
        e for e in program["equivalences"]
        if e.get("justification", {}).get("kind") == "z3"
        and "affine" in e["a_node_id"]
    ]
    assert len(z3_eqs) == 1, (
        f"expected one Z3 witness for affine; got "
        f"{[e['justification'] for e in z3_eqs]}"
    )
    eq = z3_eqs[0]
    assert eq["regime"] == "witness"
    artifact_path = tmp_path / eq["justification"]["artifact_path"]
    assert artifact_path.exists(), f"smt2 artifact missing: {artifact_path}"
    assert artifact_path.read_text().startswith(";")  # SMT-LIB comment


def test_equiv_prove_bump_is_idempotent(tmp_path):
    """Running --bump twice should not duplicate the Z3 witness."""
    _ingest_into(tmp_path)
    assert _quod(tmp_path, "equiv", "prove", "--bump").exit_code == 0
    assert _quod(tmp_path, "equiv", "prove", "--bump").exit_code == 0

    program = json.loads((tmp_path / "program.json").read_text())
    z3_eqs = [
        e for e in program["equivalences"]
        if e.get("justification", {}).get("kind") == "z3"
        and "affine" in e["a_node_id"]
    ]
    assert len(z3_eqs) == 1


def test_equiv_prove_bump_refutes_when_source_disagrees(tmp_path):
    """Mutate the source so it disagrees with the binary, re-ingest C
    only (keep the original .so), then run --bump. The bin_relational
    prover must report REFUTED and not add a Z3 witness."""
    # Build .so from the ORIGINAL source (3*x + 5).
    _write_project(tmp_path, _C_SOURCE)
    so = _build_so(tmp_path)
    # Now write a DIFFERENT source (3*x + 6) and ingest THAT.
    (tmp_path / "poc.c").write_text(_C_SOURCE_MUTATED)
    assert _quod(tmp_path, "ingest", "c", str(tmp_path / "poc.c")).exit_code == 0
    assert _quod(tmp_path, "ingest", "binary", str(so)).exit_code == 0

    bumped = _quod(tmp_path, "equiv", "prove", "--bump")
    assert bumped.exit_code == 0
    # Refutation reported on the bin_relational line. The badge text
    # is "REFUTED" (see _bump_bin_relational in equiv.py).
    assert "REFUTED" in bumped.output, bumped.output
    # Counterexample was returned by z3 — at least the word "sat"
    # appears in the detail line.
    assert "sat" in bumped.output

    program = json.loads((tmp_path / "program.json").read_text())
    z3_eqs = [
        e for e in program["equivalences"]
        if e.get("justification", {}).get("kind") == "z3"
        and "affine" in e["a_node_id"]
    ]
    assert z3_eqs == []


def test_equiv_prove_bump_invalidates_stale_witness_on_source_drift(tmp_path):
    """A common workflow: build the binary, prove bin~src, then edit
    the source so it no longer matches. The next `--bump` must
    re-run z3 (don't blindly skip on existing witness), detect the
    refutation, and remove the stale Z3 witness from the program.
    """
    # Build .so from the original source (3*x + 5) and prove bin~src.
    _write_project(tmp_path, _C_SOURCE)
    so = _build_so(tmp_path)
    assert _quod(tmp_path, "ingest", "c", str(tmp_path / "poc.c")).exit_code == 0
    assert _quod(tmp_path, "ingest", "binary", str(so)).exit_code == 0
    assert _quod(tmp_path, "equiv", "prove", "--bump").exit_code == 0
    program = json.loads((tmp_path / "program.json").read_text())
    assert any(
        e.get("justification", {}).get("kind") == "z3"
        for e in program["equivalences"]
    ), "first --bump should land a Z3 witness"

    # Mutate the source so it disagrees with the binary; re-ingest C.
    (tmp_path / "poc.c").write_text(_C_SOURCE_MUTATED)
    assert _quod(tmp_path, "ingest", "c", str(tmp_path / "poc.c")).exit_code == 0

    bumped = _quod(tmp_path, "equiv", "prove", "--bump")
    assert bumped.exit_code == 0
    assert "REFUTED" in bumped.output, bumped.output

    # The stale Z3 witness must be gone.
    program = json.loads((tmp_path / "program.json").read_text())
    z3_eqs = [
        e for e in program["equivalences"]
        if e.get("justification", {}).get("kind") == "z3"
        and "affine" in e["a_node_id"]
    ]
    assert z3_eqs == [], (
        f"stale Z3 witness was not removed; saw {z3_eqs}"
    )


def test_equiv_verify_passes_on_z3_witnessed_bin_eq(tmp_path):
    """`equiv verify` re-runs z3 against the persisted .smt2; should
    pass on the freshly-bumped equivalence."""
    _ingest_into(tmp_path)
    assert _quod(tmp_path, "equiv", "prove", "--bump").exit_code == 0

    result = _quod(tmp_path, "equiv", "verify")
    assert result.exit_code == 0, result.output


def test_equiv_verify_detects_smt_artifact_tampering(tmp_path):
    """Tampering with the on-disk .smt2 artifact is caught by hash
    mismatch — same shape as the LiftEquivalence tampering case."""
    _ingest_into(tmp_path)
    assert _quod(tmp_path, "equiv", "prove", "--bump").exit_code == 0

    program = json.loads((tmp_path / "program.json").read_text())
    z3_eq = next(
        e for e in program["equivalences"]
        if e.get("justification", {}).get("kind") == "z3"
    )
    art = tmp_path / z3_eq["justification"]["artifact_path"]
    art.write_text(art.read_text() + "\n; tampered\n")

    result = _quod(tmp_path, "equiv", "verify")
    assert result.exit_code == 1
    assert "hash mismatch" in result.output
