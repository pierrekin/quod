"""End-to-end test for `quod.ingest.binary` against a hand-crafted
schema_version=1 JSON fixture.

CI can't drive Ghidra (it requires a JDK + `ghidra-analyzeHeadless` on
PATH), so we drive the parser directly via `ingest_binary_dump`. The
upstream subprocess path (`ingest_binary`) is the same code minus the
subprocess shell-out; whatever Ghidra produces, the parser is what
makes it nodes.

What this pins:
- A schema_version=1 dump turns into a `BinUnit` under
  `Program.binary_units` with the right field shapes.
- The seeder pairs `bin.fn ↔ CFn` by symtab name match and emits both
  the `Equivalence(BinaryProvenance)` and the `ProvenanceEdge`.
- The whole `Program` round-trips through `save_program` /
  `load_program` after ingest.
"""
from __future__ import annotations

import json

import pytest

from quod.ingest.binary import (
    BinaryIngestError,
    ingest_binary_dump,
    seed_binary_equivalences,
)
from quod.model import (
    BinaryProvenance,
    CFn,
    CIntLit,
    CNamedType,
    CReturn,
    CUnit,
    Function,
    I32Type,
    IntLit,
    Param,
    Program,
    ReturnExpr,
    load_program,
    save_program,
)
from quod.model.statements import Block


def _libdemo_dump() -> dict:
    """A 1-function ELF dump shaped per .scratch/ghidra/02-ghidra-export.md.

    Models a hypothetical `void greet(void) { puts("hi"); }` compiled to
    `libdemo.so`. One function, one basic block, one CALL into `puts`,
    one string in `.rodata`, one extern.
    """
    return {
        "schema_version": 1,
        "binary": {
            "path": "/tmp/build/libdemo.so",
            "sha256": "f" * 64,
            "arch": "x86_64",
            "format": "elf",
            "build_id": "feedbabe",
        },
        "functions": [
            {
                "address": "0x401120",
                "name_mangled": "greet",
                "name_demangled": "greet",
                "signature": {
                    "return_type": "void",
                    "params": [],
                },
                "calling_convention": "__cdecl",
                "decompile": "void greet(void) {\n  puts(\"hi\");\n}\n",
                "basic_blocks": [
                    {
                        "address": "0x401120",
                        "end": "0x401130",
                        "successors": [],
                        "pcode": [
                            {
                                "opcode": "CALL",
                                "inputs": [
                                    {"space": "ram", "offset": "0x401030", "size": 8}
                                ],
                                "output": None,
                                "instr_address": "0x401128",
                            },
                            {
                                "opcode": "RETURN",
                                "inputs": [
                                    {"space": "register", "offset": "0x20", "size": 8}
                                ],
                                "output": None,
                                "instr_address": "0x40112e",
                            },
                        ],
                    },
                ],
                "calls": [
                    {
                        "from_block": "0x401120",
                        "instr_address": "0x401128",
                        "to": {"kind": "external", "name": "puts", "address": "0x401030"},
                        "call_kind": "direct",
                    }
                ],
            },
        ],
        "data": [
            {"address": "0x402000", "kind": "string", "value": "hi"},
        ],
        "externs": [
            {"name": "puts", "address": "0x401030"},
        ],
        "type_refs": [],
    }


def _greet_c_unit() -> CUnit:
    int_t = CNamedType(name="int")
    return CUnit(
        source_path="greet.c",
        functions=(
            CFn(
                name="greet",
                return_type=int_t,
                body=(CReturn(value=CIntLit(type=I32Type(), value=0)),),
            ),
        ),
    )


def _greet_function() -> Function:
    return Function(
        name="greet",
        params=(),
        return_type=I32Type(),
        body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
    )


def _write_dump(tmp_path, dump: dict):
    p = tmp_path / "libdemo.json"
    p.write_text(json.dumps(dump))
    return p


# ----- Parser shape -----


def test_ingest_binary_dump_builds_a_bin_unit(tmp_path):
    dump_path = _write_dump(tmp_path, _libdemo_dump())
    program = ingest_binary_dump(dump_path)

    assert len(program.binary_units) == 1
    unit = program.binary_units[0]
    assert unit.path == "/tmp/build/libdemo.so"
    assert unit.arch == "x86_64"
    assert unit.file_format == "elf"
    assert unit.build_id == "feedbabe"

    assert len(unit.functions) == 1
    fn = unit.functions[0]
    assert fn.address == 0x401120
    assert fn.demangled_name == "greet"
    assert "puts" in fn.decompile_text
    assert len(fn.basic_blocks) == 1

    bb = fn.basic_blocks[0]
    assert bb.start_address == 0x401120
    assert len(bb.pcode_ops) == 2
    assert bb.pcode_ops[0].opcode == "CALL"
    assert bb.pcode_ops[1].opcode == "RETURN"
    # RETURN has no output varnode.
    assert bb.pcode_ops[1].output is None

    # The call edge resolved to the BinExternRef for puts.
    assert len(fn.call_edges) == 1
    call = fn.call_edges[0]
    assert call.call_kind == "direct"
    assert len(unit.extern_refs) == 1
    assert call.callee_id == unit.extern_refs[0].id
    assert unit.extern_refs[0].symbol == "puts"

    # Caller block id resolves to the only block.
    assert call.caller_block_id == bb.id


def test_ingest_extends_existing_program(tmp_path):
    """`ingest_binary_dump` is non-mutating — it returns a new Program
    with the BinUnit appended; the input is unchanged."""
    dump_path = _write_dump(tmp_path, _libdemo_dump())
    base = Program(source_units=(_greet_c_unit(),))
    extended = ingest_binary_dump(dump_path, program=base)

    assert len(base.binary_units) == 0
    assert len(extended.binary_units) == 1
    assert extended.source_units == base.source_units


# ----- Seeder -----


def test_seeder_pairs_bin_fn_to_cfn_by_name(tmp_path):
    """The fixture's `bin.fn` named `greet` matches the source `CFn`
    named `greet` — emit one axiom-Equivalence with BinaryProvenance and
    one ProvenanceEdge."""
    dump_path = _write_dump(tmp_path, _libdemo_dump())
    program = ingest_binary_dump(dump_path, program=Program(source_units=(_greet_c_unit(),)))

    cfn = program.source_units[0].functions[0]
    bin_fn = program.binary_units[0].functions[0]

    assert len(program.equivalences) == 1
    eq = program.equivalences[0]
    assert eq.a_node_id == cfn.id
    assert eq.b_node_id == bin_fn.id
    assert eq.regime == "axiom"
    assert isinstance(eq.justification, BinaryProvenance)
    assert eq.justification.binary_symbol == "greet"
    assert eq.justification.binary_sha256 == "f" * 64
    assert eq.justification.source_evidence == "symtab"

    assert len(program.edges) == 1
    edge = program.edges[0]
    assert edge.source == cfn.id
    assert edge.target == bin_fn.id


def test_seeder_prefers_cfn_over_function_when_both_exist(tmp_path):
    """The c-ingester emits parallel CFn (Layer A) and Function (Layer C)
    nodes. That's the normal post-ingest state, not a collision —
    pair Layer A first, since the binary was built from the C source."""
    dump_path = _write_dump(tmp_path, _libdemo_dump())
    base = Program(
        source_units=(_greet_c_unit(),),
        functions=(_greet_function(),),
    )
    program = ingest_binary_dump(dump_path, program=base)

    cfn = program.source_units[0].functions[0]
    fn = program.functions[0]
    bin_fn = program.binary_units[0].functions[0]

    # One equivalence — to the CFn, not the Function.
    assert len(program.equivalences) == 1
    eq = program.equivalences[0]
    assert eq.a_node_id == cfn.id
    assert eq.a_node_id != fn.id
    assert eq.b_node_id == bin_fn.id


def test_seeder_skips_unpaired_bin_fns(tmp_path):
    """No source `greet` → no seeded equivalence."""
    dump_path = _write_dump(tmp_path, _libdemo_dump())
    program = ingest_binary_dump(dump_path)  # no source_units / functions

    assert len(program.binary_units) == 1
    assert len(program.equivalences) == 0
    assert len(program.edges) == 0


def test_seeder_is_idempotent(tmp_path):
    """Re-running the seeder on a program that already carries the
    equivalence does not duplicate it."""
    dump_path = _write_dump(tmp_path, _libdemo_dump())
    program = ingest_binary_dump(dump_path, program=Program(source_units=(_greet_c_unit(),)))
    again = seed_binary_equivalences(program)

    assert len(again.equivalences) == 1
    assert len(again.edges) == 1
    # Same content → same model_dump.
    assert program.model_dump_json() == again.model_dump_json()


def test_seeder_refuses_same_layer_collision(tmp_path):
    """Two CFns named `greet` in different CUnits → identity collision;
    leave the bin.fn unpaired."""
    dump_path = _write_dump(tmp_path, _libdemo_dump())
    other = CUnit(
        source_path="other.c",
        functions=(
            CFn(
                name="greet",
                return_type=CNamedType(name="int"),
                body=(CReturn(value=CIntLit(type=I32Type(), value=1)),),
            ),
        ),
    )
    program = ingest_binary_dump(
        dump_path,
        program=Program(source_units=(_greet_c_unit(), other)),
    )

    assert len(program.binary_units) == 1
    assert len(program.equivalences) == 0


# ----- Round-trip -----


def test_program_with_bin_unit_and_seeded_equivalence_round_trips(tmp_path):
    dump_path = _write_dump(tmp_path, _libdemo_dump())
    program = ingest_binary_dump(dump_path, program=Program(source_units=(_greet_c_unit(),)))
    out_path = tmp_path / "program.json"
    save_program(program, out_path)
    loaded = load_program(out_path)
    assert loaded == program


# ----- Refusals -----


def test_unsupported_schema_version_raises(tmp_path):
    bad = _libdemo_dump()
    bad["schema_version"] = 99
    dump_path = _write_dump(tmp_path, bad)
    with pytest.raises(BinaryIngestError, match="unsupported schema_version"):
        ingest_binary_dump(dump_path)


def test_unknown_file_format_raises(tmp_path):
    bad = _libdemo_dump()
    bad["binary"]["format"] = "coff"
    dump_path = _write_dump(tmp_path, bad)
    with pytest.raises(BinaryIngestError, match="unsupported file_format"):
        ingest_binary_dump(dump_path)


def test_invalid_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(BinaryIngestError, match="invalid JSON"):
        ingest_binary_dump(p)


def test_merge_preserves_binary_units(tmp_path):
    """`merge_program` must not drop `binary_units` when folding two
    programs together. Regression: merging a c-ingest result into a
    program that already had a `BinUnit` from a prior `ingest binary`
    call would erase the binary unit, because `merge_program` built
    its result via the `Program(...)` constructor without copying
    `binary_units` over.

    Concretely: the bare `quod ingest` callback runs entries in
    declaration order. After `[bin a, c b, bin c]`, the second c-file
    ingest's merge must still preserve `bin a`'s unit; the third
    binary ingest then appends `bin c` alongside it.
    """
    from quod.merge import merge_program
    dump_path = _write_dump(tmp_path, _libdemo_dump())
    a = ingest_binary_dump(dump_path)  # has 1 binary_unit
    assert len(a.binary_units) == 1

    # A fresh c-ingest-style Program (no binary_units of its own).
    b = Program(source_units=(_greet_c_unit(),))
    assert len(b.binary_units) == 0

    merged, _ = merge_program(a, b)
    assert len(merged.binary_units) == 1, (
        "merge_program dropped a.binary_units when folding in b"
    )
    assert merged.binary_units[0].path == a.binary_units[0].path
    assert merged.source_units == b.source_units


def test_unknown_call_kind_raises(tmp_path):
    bad = _libdemo_dump()
    bad["functions"][0]["calls"][0]["call_kind"] = "virtual"
    dump_path = _write_dump(tmp_path, bad)
    with pytest.raises(BinaryIngestError, match="unknown call_kind"):
        ingest_binary_dump(dump_path)
