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
    the merged `Program`. Done **once per module** as a perf
    optimization — Ghidra analysis costs ~3–5s per binary even on a
    tiny `.so`, and several independent assertions can share the same
    output. Per-test ingest also works (function-scope is fine for the
    JVM lifecycle); the module scope just keeps the suite fast.

    Companion regression: `test_real_ghidra_multi_ingest_in_one_process`
    explicitly exercises the multi-call pattern (an earlier version of
    this code segfaulted on the second `pyghidra.analyze` in one
    process). If the workaround starts being needed again, that test
    fails first."""
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
    pairing source `greet` to binary `greet`. The fixture builds with
    `clang -g`, so `source_evidence` should be `dwarf` (Ghidra
    populates `decl_file` from DW_AT_decl_file)."""
    program, _, _ = ingested_program
    seeded = [
        eq for eq in program.equivalences
        if isinstance(eq.justification, BinaryProvenance)
    ]
    assert len(seeded) >= 1, "seeder failed to pair any source→binary function"

    eq = next(e for e in seeded if e.justification.binary_symbol == "greet")
    assert eq.regime == "axiom"
    assert eq.justification.source_evidence == "dwarf"
    cfns = {cfn.id: cfn for u in program.source_units for cfn in u.functions}
    bins = {bf.id: bf for u in program.binary_units for bf in u.functions}
    assert eq.a_node_id in cfns
    assert eq.b_node_id in bins
    assert cfns[eq.a_node_id].name == "greet"
    assert bins[eq.b_node_id].demangled_name == "greet"


def test_real_ghidra_dwarf_decl_fields_populated(ingested_program):
    """`clang -g` emits DWARF; Ghidra's source-file manager surfaces it
    via `getSourceMapEntries(entry_point)`. The exporter plumbs the
    first entry's filename + line into `BinFunction.decl_file` /
    `decl_line`. Confirm against the real `greet` function."""
    from pathlib import PurePosixPath
    program, src, _ = ingested_program
    bin_fn = next(
        f for u in program.binary_units for f in u.functions
        if f.demangled_name == "greet"
    )
    assert bin_fn.decl_file is not None, (
        "expected DWARF source-map entry on `greet`; clang -g did not emit, "
        "or the exporter dropped the field"
    )
    assert PurePosixPath(bin_fn.decl_file).name == src.name
    assert bin_fn.decl_line is not None and bin_fn.decl_line >= 1


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


def test_real_ghidra_data_dump_emits_strings_and_globals(ingested_program):
    """The exporter surfaces both string literals (`data_kind="string"`)
    and user-data globals from `.data` / `.bss` / `.rodata`
    (`data_kind="global"`, base64-encoded). Anything in structural
    sections (`.dynamic`, `.got`, `.plt`, …) is filtered out — those
    are linker bookkeeping with no Layer-A consumer.

    For the demo binaries (built with `clang -O0 -g`), at least one
    global should land (clang emits `.data.rel.ro` / `.eh_frame_hdr`
    aliases that Ghidra labels as user-data). The bytes round-trip as
    valid base64."""
    import base64
    program, _, _ = ingested_program
    unit = program.binary_units[0]
    strings = [d for d in unit.data_items if d.data_kind == "string"]
    globals_ = [d for d in unit.data_items if d.data_kind == "global"]
    assert strings, "expected at least one string in a `clang -g` .so"
    assert globals_, (
        "expected at least one user-data global; if none surface, the "
        "section-name filter is rejecting everything (regression)"
    )
    for g in globals_:
        # Round-trip through base64 to confirm the value is valid.
        decoded = base64.b64decode(g.value)
        assert len(decoded) > 0


def test_real_ghidra_type_refs_bounded_by_signatures(ingested_program):
    """The exporter filters Ghidra's DataType universe down to types
    that actually appear on a function signature in this dump. Without
    the filter, every binary dumps ~150 types from
    `generic_clib_64.gdt` regardless of whether the binary uses them.

    This is a structural cap (each emitted type's name must appear on
    some signature), not a count assertion — the real number depends
    on which signatures Ghidra recovered."""
    program, _, _ = ingested_program
    unit = program.binary_units[0]
    sig_names: set[str] = set()
    for fn in unit.functions:
        sig_names.add(fn.return_type_name)
        for p in fn.params:
            sig_names.add(p.type_name)
    type_ref_names = {t.name for t in unit.type_refs}
    extras = type_ref_names - sig_names
    assert not extras, (
        f"type_refs leaked names not on any signature: {sorted(extras)}; "
        f"the exporter should filter to signature-referenced types only"
    )
    # And we should be well under "Ghidra's full archive" — ~150 was
    # the pre-filter count for a toy .so.
    assert len(unit.type_refs) < 50


# --------------------------------------------------------------------
# Relational POC: lift_v2.signature_binding + z3.bin_relational
#
# The very first end-to-end relational proof of bin↔src equivalence —
# small int-arithmetic functions whose binary form (built with
# `clang -O1`) stays in registers, so the v0 prover (no memory model,
# no branches) can encode them and z3 can close the proof. See the
# module docstring of `quod.predicate.binary_relational` for the v0
# universe.
# --------------------------------------------------------------------

_RELATIONAL_POC_SOURCE = """\
int ident(int x) { return x; }
int add(int a, int b) { return a + b; }
int affine(int x) { return 3 * x + 5; }
"""


@pytest.fixture(scope="module")
def relational_poc_program(tmp_path_factory):
    """Build the POC fixture at `-O1` (registers stay in registers,
    no stack spills) and ingest both the C source and the resulting
    `.so`. Run lift_v2.signature_binding so the program carries
    explicit varnode↔param bindings.

    `-O1` is intentional: at `-O0` clang emits a store-to-stack /
    load-from-stack round-trip even for `return x;`, which the v0
    prover doesn't model (memory is out of universe). `-O1` keeps
    everything in registers and stays under the prover's whitelist.
    """
    from quod.ingest.binary import ingest_binary
    from quod.predicate.binary_lift import derive_signature_bindings
    clang = _need_clang()
    work = tmp_path_factory.mktemp("rel_poc")
    src = work / "poc.c"
    src.write_text(_RELATIONAL_POC_SOURCE)
    so = work / "libpoc.so"
    subprocess.run(
        [clang, "-O1", "-g", "-shared", "-fPIC", "-o", str(so), str(src)],
        check=True, capture_output=True, text=True,
    )
    c_program = ingest_c(src)
    program, _ = merge_program(Program(), c_program)
    program = ingest_binary(so, program=program)

    bindings = derive_signature_bindings(program)
    program = program.model_copy(update={"signature_bindings": bindings})
    return program


def test_relational_poc_signature_bindings_emitted(relational_poc_program):
    """lift_v2.signature_binding produces one binding per int-only
    function in the POC. The 3 demo functions all qualify."""
    program = relational_poc_program
    names = {
        f.name for f in program.functions if f.name in ("ident", "add", "affine")
    }
    assert names == {"ident", "add", "affine"}

    sig_by_src_name: dict[str, object] = {}
    for sb in program.signature_bindings:
        src_fn = next(f for f in program.functions if f.id == sb.src_fn_id)
        sig_by_src_name[src_fn.name] = sb
    for name in ("ident", "add", "affine"):
        assert name in sig_by_src_name, (
            f"lift_v2 missed {name!r}; bindings: {sorted(sig_by_src_name)}"
        )

    # ABI-derived varnodes should mirror SysV: first int param at RDI
    # (register-space offset 0x38), second at RSI (0x30), return at
    # RAX (0x00).
    add_sb = sig_by_src_name["add"]
    assert add_sb.abi == "x86_64-sysv"
    assert add_sb.param_bindings[0].varnode.offset == 0x38
    assert add_sb.param_bindings[1].varnode.offset == 0x30
    assert add_sb.return_binding.offset == 0x00


def test_relational_poc_decompile_lift_emits_cfn_for_each(relational_poc_program):
    """Real Ghidra → real decompile_text → libclang → Layer-A CFn.

    For our v0 universe (clang -O1 int arithmetic), Ghidra's decompile
    output is clean enough C that libclang parses it. The lift then
    walks the AST through the same translator the source ingester uses,
    so the lifted CFn has the same `c.*` shape as a hand-authored one.

    This is the second axis of "lift v2" (sibling to
    lift_v2.signature_binding) — structural recovery, not just
    register-mapping. With both, downstream provers can pair source
    and binary by *structure*, not just final return value.
    """
    from quod.predicate.binary_decompile_lift import derive_decompile_lifts

    program = relational_poc_program
    new_prog, lifts = derive_decompile_lifts(program)
    by_name = {lift.cfn.name: lift for lift in lifts}
    for name in ("ident", "add", "affine"):
        assert name in by_name, (
            f"decompile_lift didn't produce a CFn for {name!r}; "
            f"got {sorted(by_name)}"
        )

    # Each lifted CFn nests under the BinUnit it came from.
    [u] = new_prog.binary_units
    lifted_names = {c.name for c in u.lifted_cfns}
    assert {"ident", "add", "affine"}.issubset(lifted_names)

    # Each lift gets an Equivalence with DecompileLift justification.
    bin_fns_by_name = {f.demangled_name: f.id for f in u.functions}
    decompile_eqs = [
        e for e in new_prog.equivalences
        if e.justification is not None
        and e.justification.kind == "decompile_lift"
    ]
    assert len(decompile_eqs) >= 3
    for name in ("ident", "add", "affine"):
        eq = next(
            (e for e in decompile_eqs if e.b_node_id == bin_fns_by_name[name]),
            None,
        )
        assert eq is not None, f"no decompile_lift equivalence for {name!r}"
        assert eq.justification.decompile_text_sha256, "missing decompile hash"


def test_relational_poc_prover_proves_all_three(relational_poc_program, tmp_path):
    """The end-to-end relational result: each (bin.fn, src.fn) pair
    in the POC fixture is proven equivalent by z3.

    If this test starts failing because `clang -O1` decided to use a
    pcode op the v0 encoder doesn't handle (e.g. `lea` lowering
    surfaces an `INT_LEFT` for the `*3` step that's already encodable,
    or perhaps a SUBPIECE pattern that's beyond v0), the right fix is
    to teach the encoder another opcode — not to dilute this test."""
    if shutil_which := __import__("shutil").which("z3"):
        pass
    else:
        pytest.skip("z3 binary not on PATH; needed to close the proof")

    from quod.predicate.binary_relational import prove_all_bin_relational

    program = relational_poc_program
    results = prove_all_bin_relational(program, proofs_dir=tmp_path)
    by_src_name: dict[str, object] = {}
    for r in results:
        src_fn = next(f for f in program.functions if f.id == r.src_fn_id)
        by_src_name[src_fn.name] = r

    failures = []
    for name in ("ident", "add", "affine"):
        r = by_src_name.get(name)
        if r is None:
            failures.append(f"{name}: no result")
            continue
        if r.status != "proven":
            failures.append(f"{name}: status={r.status}, detail={r.detail}")
    assert not failures, "\n".join(failures)


def test_real_ghidra_multi_ingest_in_one_process(tmp_path_factory):
    """Regression for the JVM-multi-load issue (P0 in 06-polish.md).

    An earlier draft of the binary frontend segfaulted on the second
    `pyghidra.analyze` call within a single Python process — the
    workaround was the `scope="module"` shared-fixture above. The
    underlying cause was likely tied to the deprecated `open_program`
    API (now replaced with `program_loader`) or to PyGhidra/JPype
    versions we've since picked up; the symptom is no longer
    reproducible. This test locks in the working behavior so a
    regression surfaces immediately rather than at the next person to
    write multi-binary tests.

    Three independent ingest cycles in one process (each with its own
    .so, each going through `ingest_c` first because the original
    failure surfaced specifically *after* libclang was used). Asserts
    each cycle's `binary_units` is intact and contains the function
    we authored."""
    from quod.ingest import ingest_c
    from quod.ingest.binary import ingest_binary
    from quod.merge import merge_program
    from quod.model import Program

    clang = _need_clang()
    for i in range(3):
        work = tmp_path_factory.mktemp(f"multi_ingest_{i}")
        src = work / "src.c"
        src.write_text(f"int f{i}(int n) {{ return n + {i}; }}\n")
        so = work / "lib.so"
        subprocess.run(
            [clang, "-O0", "-g", "-shared", "-fPIC", "-o", str(so), str(src)],
            check=True, capture_output=True, text=True,
        )
        c_program = ingest_c(src)
        program, _ = merge_program(Program(), c_program)
        program = ingest_binary(so, program=program)

        assert len(program.binary_units) == 1, (
            f"iteration {i}: expected 1 binary_unit, got {len(program.binary_units)}"
        )
        names = {f.demangled_name for u in program.binary_units for f in u.functions}
        assert f"f{i}" in names, (
            f"iteration {i}: bin.fn `f{i}` missing from {sorted(names)}"
        )
