"""Layer-A `bin.*` family — Ghidra-ingest source-faithful nodes, and the
`Program.binary_units` collection.

Pins JSON shape, ID-prefix conventions, and round-trip behavior. No
codegen, no validation, no semantic checks — Layer-A is inert by
contract (see `.scratch/ghidra/01-layer-a-nodes.md`).

The fixture is a small recovered binary with one function that calls
`malloc` and reads a `.rodata` string — minimal enough to pin the
schema, big enough to exercise nested ownership (BinUnit → BinFunction
→ BinBasicBlock → BinPCodeOp), in-function-by-id edges (BinBlockEdge,
BinCallEdge), and cross-unit references (BinExternRef, BinDataItem).
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from quod.model import (
    BinBasicBlock,
    BinBlockEdge,
    BinCallEdge,
    BinDataItem,
    BinExternRef,
    BinFunction,
    BinFunctionParam,
    BinPCodeOp,
    BinTypeRef,
    BinUnit,
    BinVarnode,
    Program,
    load_program,
    save_program,
)


def _bin_unit() -> BinUnit:
    bb_entry = BinBasicBlock(
        start_address=0x401120,
        end_address=0x401140,
        pcode_ops=(
            BinPCodeOp(
                opcode="COPY",
                inputs=(BinVarnode(space="const", offset=64, size=8),),
                output=BinVarnode(space="register", offset=0x38, size=8),
                source_address=0x401120,
            ),
            BinPCodeOp(
                opcode="CALL",
                inputs=(BinVarnode(space="ram", offset=0x401030, size=8),),
                output=None,
                source_address=0x401128,
            ),
        ),
    )
    bb_exit = BinBasicBlock(
        start_address=0x401140,
        end_address=0x40114a,
        pcode_ops=(
            BinPCodeOp(
                opcode="RETURN",
                inputs=(BinVarnode(space="register", offset=0x20, size=8),),
                output=None,
                source_address=0x401140,
            ),
        ),
    )
    bb_entry = bb_entry.model_copy(update={
        "successors": (BinBlockEdge(successor_id=bb_exit.id, edge_kind="call_return"),),
    })

    extern_malloc = BinExternRef(
        symbol="malloc",
        abi_hint="System V AMD64",
        linked_extern_name="malloc",
    )
    call_malloc = BinCallEdge(
        caller_block_id=bb_entry.id,
        instruction_address=0x401128,
        callee_id=extern_malloc.id,
        call_kind="direct",
    )

    fn = BinFunction(
        address=0x401120,
        mangled_name="alloc_buffer",
        demangled_name="alloc_buffer",
        return_type_name="void *",
        params=(BinFunctionParam(name="size", type_name="ulong"),),
        calling_convention="__cdecl",
        basic_blocks=(bb_entry, bb_exit),
        call_edges=(call_malloc,),
        decompile_text="void * alloc_buffer(ulong size) {\n  return malloc(size);\n}\n",
    )

    greeting = BinDataItem(
        address=0x402000,
        data_kind="string",
        value="hello, world",
        referenced_by=(fn.id,),
    )
    long_t = BinTypeRef(name="ulong", size=8, structural_hash="ghidra:builtin:ulong")

    return BinUnit(
        path="build/libdemo.so",
        sha256="0" * 64,
        arch="x86_64",
        file_format="elf",
        build_id="abcdef0123456789",
        functions=(fn,),
        data_items=(greeting,),
        extern_refs=(extern_malloc,),
        type_refs=(long_t,),
    )


def test_binary_unit_round_trips_through_json(tmp_path):
    p = Program(binary_units=(_bin_unit(),))
    path = tmp_path / "program.json"
    save_program(p, path)
    loaded = load_program(path)
    assert loaded == p


def test_binary_layer_a_node_ids_persist_across_save_load(tmp_path):
    p = Program(binary_units=(_bin_unit(),))
    path = tmp_path / "program.json"
    save_program(p, path)

    first = load_program(path)
    unit = first.binary_units[0]
    assert unit.id.startswith("@binunit_")

    fn = unit.functions[0]
    assert fn.id.startswith("@binfn_")
    assert fn.basic_blocks[0].id.startswith("@binbb_")
    assert fn.basic_blocks[0].pcode_ops[0].id.startswith("@binpcode_")
    assert fn.call_edges[0].id.startswith("@bincall_")
    assert unit.data_items[0].id.startswith("@bindata_")
    assert unit.extern_refs[0].id.startswith("@binextern_")
    assert unit.type_refs[0].id.startswith("@bintype_")

    # The CFG edge inside bb_entry references bb_exit by ID — that ID
    # must survive round-trip and still match the bb_exit object so the
    # graph is reconstructable.
    bb_entry, bb_exit = fn.basic_blocks
    assert bb_entry.successors[0].successor_id == bb_exit.id

    # And the call edge references the extern ref by ID.
    assert fn.call_edges[0].callee_id == unit.extern_refs[0].id

    # Save what we just loaded; IDs round-trip identically.
    save_program(first, path)
    second = load_program(path)
    assert first.model_dump_json() == second.model_dump_json()


def test_program_drops_empty_binary_units_from_json():
    p = Program()
    decoded = json.loads(p.model_dump_json())
    assert "binary_units" not in decoded


def test_bin_unit_drops_empty_collections_and_optional_build_id():
    unit = BinUnit(
        path="raw.bin",
        sha256="0" * 64,
        arch="x86_64",
        file_format="raw",
    )
    decoded = json.loads(unit.model_dump_json())
    assert "build_id" not in decoded
    assert "functions" not in decoded
    assert "data_items" not in decoded
    assert "extern_refs" not in decoded
    assert "type_refs" not in decoded


def test_bin_extern_ref_drops_none_optionals():
    ref = BinExternRef(symbol="malloc")
    decoded = json.loads(ref.model_dump_json())
    assert "abi_hint" not in decoded
    assert "linked_extern_name" not in decoded


def test_bin_function_drops_empty_decompile_text():
    bb = BinBasicBlock(start_address=0, end_address=1)
    fn = BinFunction(
        address=0x1000,
        mangled_name="f",
        demangled_name="f",
        return_type_name="void",
        calling_convention="__cdecl",
        basic_blocks=(bb,),
    )
    decoded = json.loads(fn.model_dump_json())
    assert "decompile_text" not in decoded


def test_bin_pcode_op_output_can_be_null():
    """RETURN/CALL/STORE p-code ops have no output varnode; the schema
    must round-trip None faithfully."""
    op = BinPCodeOp(
        opcode="RETURN",
        inputs=(BinVarnode(space="const", offset=0, size=8),),
        output=None,
        source_address=0x1000,
    )
    decoded = json.loads(op.model_dump_json())
    assert decoded["output"] is None


def test_bin_block_edge_kind_is_constrained():
    """The successor `edge_kind` is a finite enum — a wrong-cased or
    invented label fails fast at construction."""
    with pytest.raises(ValidationError):
        BinBlockEdge(successor_id="@binbb_x", edge_kind="taken")


def test_bin_unit_file_format_is_constrained():
    with pytest.raises(ValidationError):
        BinUnit(
            path="x", sha256="0" * 64, arch="x86_64",
            file_format="exe",  # noqa: not in {elf, pe, mach-o, raw}
        )


def test_bin_data_item_kind_is_constrained():
    with pytest.raises(ValidationError):
        BinDataItem(address=0, data_kind="rodata", value="x")


def test_bin_call_edge_kind_is_constrained():
    with pytest.raises(ValidationError):
        BinCallEdge(
            caller_block_id="@binbb_x",
            instruction_address=0,
            callee_id="@binfn_x",
            call_kind="virtual",
        )


def test_format_bin_fn_clean_decompile_strips_block_comments():
    """Display-only filter: `_clean_decompile_text` strips `/* */`
    blocks and trims surrounding blank lines. The underlying node's
    `decompile_text` is unaltered."""
    from quod.model.pretty import _clean_decompile_text
    text = (
        "\n"
        "/* WARNING: Removing unreachable block (ram,0x123) */\n"
        "void f(int x)\n"
        "{\n"
        "  /* Subroutine type: stdcall */\n"
        "  return x + 1;\n"
        "}\n"
        "\n"
    )
    cleaned = _clean_decompile_text(text)
    assert "WARNING" not in cleaned
    assert "Subroutine type" not in cleaned
    assert "void f(int x)" in cleaned
    assert "return x + 1;" in cleaned
    # Trimmed top/bottom blank lines but preserved internal structure.
    assert not cleaned.startswith("\n")
    assert not cleaned.endswith("\n\n")


def test_format_bin_fn_clean_decompile_handles_multiline_blocks():
    """A `/* ... */` that spans multiple lines is stripped wholesale."""
    from quod.model.pretty import _clean_decompile_text
    text = (
        "/* multi\n"
        " * line\n"
        " * comment */\n"
        "int f(void) { return 0; }\n"
    )
    cleaned = _clean_decompile_text(text)
    assert "multi" not in cleaned
    assert "int f(void)" in cleaned


def test_format_bin_fn_raw_decompile_preserves_text():
    """`format_bin_fn(raw_decompile=True)` keeps the full text."""
    from quod.model import format_bin_fn
    fn = BinFunction(
        address=0x1000,
        mangled_name="f",
        demangled_name="f",
        return_type_name="void",
        calling_convention="__cdecl",
        decompile_text="/* WARNING ... */\nvoid f(void) {}\n",
    )
    rendered = format_bin_fn(fn, raw_decompile=True)
    assert "WARNING" in rendered
    rendered_clean = format_bin_fn(fn)
    assert "WARNING" not in rendered_clean
    assert "void f(void)" in rendered_clean


def test_bin_unit_round_trips_alongside_source_units(tmp_path):
    """`source_units` (C frontend) and `binary_units` (binary frontend)
    are independent collections at program level — neither should
    interfere with the other's serialization."""
    from quod.model import CFn, CIntLit, CNamedType, CReturn, CUnit, I32Type

    int_t = CNamedType(name="int")
    c_unit = CUnit(
        source_path="zero.c",
        functions=(
            CFn(
                name="zero",
                return_type=int_t,
                body=(CReturn(value=CIntLit(type=I32Type(), value=0)),),
            ),
        ),
    )

    p = Program(source_units=(c_unit,), binary_units=(_bin_unit(),))
    path = tmp_path / "program.json"
    save_program(p, path)
    loaded = load_program(path)
    assert loaded == p
    assert loaded.source_units == p.source_units
    assert loaded.binary_units == p.binary_units
