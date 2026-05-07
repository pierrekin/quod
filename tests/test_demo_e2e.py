"""End-to-end test of the staged-lift demo flow.

Walks the demo from `.scratch/c-ingest/01-structured-quod.md`:
  - write a quod.toml + sum.c
  - `quod ingest` → program.json with all three layers + edges +
    witness-regime equivalences
  - `quod equiv verify` → both A~B and B~C claims pass
  - `quod equiv prove` → all witnessed-current
  - `quod build` → emit a binary
  - run the binary, assert sum(0..N-1) as exit code for several N

Most of the individual commands have unit tests already; this test
exists to catch regressions in the *full path* from C source to
running binary, including the build leg that the unit tests stub
out via `compile_program(..., link=True)` directly.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from typer.testing import CliRunner

from quod import cli as cli_mod


def _quod(root: Path, *args: str):
    """Invoke the quod CLI in-process via Typer's CliRunner."""
    cli_mod._state.clear()
    runner = CliRunner()
    return runner.invoke(cli_mod.app, ["-c", str(root / "quod.toml"), *args])


def _write_demo_project(root: Path) -> None:
    (root / "sum.c").write_text(
        "int sum(int n) {\n"
        "    int s = 0;\n"
        "    for (int i = 0; i < n; i = i + 1) { s = s + i; }\n"
        "    return s;\n"
        "}\n"
    )
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
        'name = "sum_demo"\n'
        'entry = "sum"\n'
        '\n'
        '[[ingest.entry]]\n'
        'kind = "c-file"\n'
        'source = "sum.c"\n'
    )


def test_demo_flow_end_to_end(tmp_path):
    _write_demo_project(tmp_path)

    # ---------- ingest ----------
    ingest = _quod(tmp_path, "ingest")
    assert ingest.exit_code == 0, ingest.output
    assert (tmp_path / "program.json").exists()

    # ---------- equiv verify ----------
    # Two equivalences for sum: A~B (lift_equivalence) and B~C
    # (family_lowering). Both should pass.
    verify = _quod(tmp_path, "equiv", "verify")
    assert verify.exit_code == 0, verify.output
    ok_lines = [ln for ln in verify.output.splitlines() if ln.startswith("ok")]
    assert len(ok_lines) == 2, f"expected 2 ok lines, got: {verify.output}"

    # ---------- equiv prove ----------
    # Same claims, walked as a chain. All should report
    # witness-current ([witness/...]); none stale, none axiom.
    prove = _quod(tmp_path, "equiv", "prove")
    assert prove.exit_code == 0, prove.output
    assert "[witness/lift_equivalence]" in prove.output
    assert "[witness/family_lowering]" in prove.output
    assert "axiom" not in prove.output  # no unproven claims

    # ---------- build ----------
    # Compiles layer-C through quod.lower → LLVM IR → object → linked
    # binary at build/demo/sum_demo per quod.toml's [[program.bin]].
    build = _quod(tmp_path, "build")
    assert build.exit_code == 0, build.output
    binary = tmp_path / "build" / "demo" / "sum_demo"
    assert binary.exists(), (
        f"expected binary at {binary}, build output:\n{build.output}"
    )
    assert binary.stat().st_mode & 0o111, "binary not executable"

    # ---------- run the binary ----------
    # sum(N) = N*(N-1)/2 returned as the exit code. Process exit
    # codes are 8-bit unsigned so we keep N small.
    for n, expected in [(0, 0), (1, 0), (5, 10), (10, 45), (15, 105)]:
        completed = subprocess.run(
            [str(binary), str(n)],
            capture_output=True, text=True, timeout=10,
        )
        assert completed.returncode == expected, (
            f"sum({n}): got exit {completed.returncode}, expected "
            f"{expected}; stderr: {completed.stderr!r}"
        )
