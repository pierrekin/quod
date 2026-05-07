"""End-to-end binary ingest test against a real Ghidra install.

This is the v1 acceptance criterion from `.scratch/ghidra/00-overview.md`:

    > One end-to-end test: ingest lib.c, build lib.so, ingest .so,
    > assert at least one auto-seeded Equivalence and that quod show
    > round-trips.

Skipped when PyGhidra isn't installed or when no Ghidra install is
discoverable. On a developer box with Ghidra (Arch's `ghidra` package
or `/opt/ghidra`), it runs. Pays a one-time JVM startup cost (~5s);
subsequent ingests in the same Python process reuse the JVM.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from quod.ingest import ingest_c
from quod.merge import merge_program
from quod.model import (
    BinaryProvenance,
    Program,
    load_program,
    save_program,
)


pytest.importorskip("pyghidra", reason="pyghidra not installed; skip real-Ghidra e2e")


def _ghidra_dir() -> str | None:
    """Discover a Ghidra install. Honor `GHIDRA_INSTALL_DIR` first;
    fall back to a few common system locations. Returns None if none
    found, in which case the test skips."""
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


def _need_clang() -> str:
    if shutil_which := __import__("shutil").which("clang"):
        return shutil_which
    pytest.skip("clang not on PATH; needed to build the test .so")


_C_SOURCE = """\
#include <string.h>

int greet(const char *name) {
    return (int)strlen(name);
}

int compare_const(int v) {
    if (v < 100) return v;
    return 100;
}
"""


@pytest.fixture(scope="module")
def ingested_program(tmp_path_factory):
    """Build `libgreet.so`, ingest its C source AND the binary, return
    the merged `Program`. Done **once per module** because PyGhidra
    drives a JVM with a Ghidra project lifecycle that doesn't recover
    cleanly from multiple loads in the same process — sharing the
    ingested program across assertion tests avoids the JVM crash and
    keeps the suite fast (~5–10s of Ghidra time, paid once)."""
    from quod.ingest.binary import ingest_binary
    clang = _need_clang()
    work = tmp_path_factory.mktemp("libgreet")
    src = work / "greet.c"
    src.write_text(_C_SOURCE)
    so = work / "libgreet.so"
    subprocess.run(
        [clang, "-O0", "-g", "-shared", "-fPIC", "-o", str(so), str(src)],
        check=True, capture_output=True, text=True,
    )
    c_program = ingest_c(src)
    program, _ = merge_program(Program(), c_program)
    program = ingest_binary(so, program=program)
    return program, src, so


def test_real_ghidra_ingest_produces_bin_unit(ingested_program):
    """The exporter loads the .so, runs Ghidra auto-analysis, and
    produces at least one BinFunction. Specifically `greet` (the
    non-thunk function we authored) appears with parameters and a
    decompile body."""
    program, _, _ = ingested_program
    assert len(program.binary_units) == 1
    unit = program.binary_units[0]
    assert unit.file_format == "elf"
    assert "x86" in unit.arch
    assert unit.build_id is not None  # clang -g exposes a GNU build-id

    greet_fns = [f for f in unit.functions if f.demangled_name == "greet"]
    assert len(greet_fns) == 1
    fn = greet_fns[0]
    assert fn.basic_blocks
    assert any(op.opcode for bb in fn.basic_blocks for op in bb.pcode_ops)
    assert "greet" in fn.decompile_text


def test_real_ghidra_seeder_pairs_to_source(ingested_program):
    """v1 acceptance: ingest lib.c → ingest lib.so → at least one
    auto-seeded Equivalence with `BinaryProvenance` justification
    pairing source `greet` to binary `greet`."""
    program, _, _ = ingested_program
    seeded = [
        eq for eq in program.equivalences
        if isinstance(eq.justification, BinaryProvenance)
    ]
    assert len(seeded) >= 1, "seeder failed to pair any source→binary function"

    eq = next(e for e in seeded if e.justification.binary_symbol == "greet")
    assert eq.regime == "axiom"
    assert eq.justification.source_evidence == "symtab"
    cfns = {cfn.id: cfn for u in program.source_units for cfn in u.functions}
    bins = {bf.id: bf for u in program.binary_units for bf in u.functions}
    assert eq.a_node_id in cfns
    assert eq.b_node_id in bins
    assert cfns[eq.a_node_id].name == "greet"
    assert bins[eq.b_node_id].demangled_name == "greet"


def test_real_ghidra_program_round_trips_through_save_load(ingested_program, tmp_path):
    """The full A↔B↔C↔Binary program — with seeded BinaryProvenance —
    round-trips identically through the JSON I/O boundary."""
    program, _, _ = ingested_program
    out = tmp_path / "program.json"
    save_program(program, out)
    loaded = load_program(out)
    assert loaded == program


def test_real_ghidra_range_hints_provider_fires_on_compare_const(ingested_program):
    """First cross-layer claim flow (file 04, first half): the
    `ghidra.range_hints` provider walks the BinaryProvenance pairing
    for `compare_const`, finds INT_SLESS constants in the bin.fn's
    p-code, and emits at least one candidate `int_range` claim against
    the source `Function`. The provider doesn't have to be sound — the
    point is that the cross-layer flow runs end-to-end against real
    Ghidra output."""
    from quod.predicate.binary_hints import derive_binary_range_hints
    program, _, _ = ingested_program

    derived = derive_binary_range_hints(program)
    assert "compare_const" in derived, (
        f"expected at least one candidate against compare_const; got {list(derived)}"
    )
    claims = derived["compare_const"]
    assert claims
    for c in claims:
        assert c.regime == "lattice"
        assert c.justification.analysis == "ghidra.range_hints"
        assert c.justification.inputs  # bin.fn id pinned


def test_real_ghidra_chain_walker_recovers_compare_const_threshold(ingested_program):
    """Source: `if (v < 100) return v; return 100;`. clang at `-O0 -g`
    lowers `v < 100` to `INT_SUB v, 100; INT_SLESS tmp, 0` — the
    threshold lives on the INT_SUB. The chain walker must recover
    K=100 from a real Ghidra dump, not just the K=0 sign-test from the
    direct INT_SLESS."""
    from quod.predicate.binary_hints import derive_binary_range_hints
    program, _, _ = ingested_program

    claims = derive_binary_range_hints(program).get("compare_const", ())
    rendered = "\n".join(c.expr.model_dump_json() for c in claims)
    # Either 100 directly (the int_range upper bound) or 99 (when the
    # candidate is `int_range(v, [-inf, K-1])`) confirms recovery.
    assert "100" in rendered or "99" in rendered, (
        f"chain walker missed clang's INT_SUB→INT_SLESS lowering of "
        f"`v < 100`; recovered constants only: {rendered}"
    )
