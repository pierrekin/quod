"""End-to-end CLI tests for the binary frontend.

Pins the CLI surface added in the `quod ingest binary` v1 work:

- `quod show --binary` filters the program rendering to layer-A
  binary units (mutually exclusive with `--source` / `--structured`).
- `quod fn ls --binary` lists `bin.fn`s across every `BinUnit`.
- `quod fn show <ref> --binary` renders one `BinFunction`, accepting
  demangled name, mangled name, or `@binfn_…` id prefix.
- The `[[ingest.entry]]` / bare `quod ingest` callback dispatches
  `kind = "binary"` and `kind = "c-file"` correctly.

The full `quod ingest binary <path>` subprocess path requires
`ghidra-analyzeHeadless` on PATH and isn't exercised here — driven
in `test_binary_ingest.py` against a hand-crafted JSON fixture
through `ingest_binary_dump`.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from quod.ingest.binary import ingest_binary_dump
from quod.model import (
    CFn,
    CIntLit,
    CNamedType,
    CReturn,
    CUnit,
    I32Type,
    Program,
    save_program,
)


def _libdemo_dump() -> dict:
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
        "data": [],
        "externs": [
            {"name": "puts", "address": "0x401030"},
        ],
        "type_refs": [],
    }


def _write_project(root: Path, *, with_source: bool = True) -> None:
    """Build a project with one binary unit and (optionally) a paired
    source CUnit. We populate program.json directly via
    `ingest_binary_dump` rather than the subprocess `quod ingest binary`
    path because CI doesn't have Ghidra."""
    dump_path = root / "libdemo.json"
    dump_path.write_text(json.dumps(_libdemo_dump()))

    if with_source:
        int_t = CNamedType(name="int")
        cunit = CUnit(
            source_path="greet.c",
            functions=(
                CFn(
                    name="greet",
                    return_type=int_t,
                    body=(CReturn(value=CIntLit(type=I32Type(), value=0)),),
                ),
            ),
        )
        base = Program(source_units=(cunit,))
    else:
        base = Program()

    program = ingest_binary_dump(dump_path, program=base)
    save_program(program, root / "program.json")

    (root / "quod.toml").write_text(
        'build_dir  = "build"\n'
        'proofs_dir = "proofs"\n'
        '\n'
        '[[program]]\n'
        'name = "demo"\n'
        'version = "0.1.0"\n'
        'file = "program.json"\n'
    )


def _quod(root: Path, *args: str):
    from quod import cli as cli_mod
    cli_mod._state.clear()
    runner = CliRunner()
    return runner.invoke(
        cli_mod.app,
        ["-c", str(root / "quod.toml"), *args],
    )


# ----- quod show --binary -----


def test_show_default_includes_binary_units(tmp_path):
    """No flag → every populated section, including binary_units."""
    _write_project(tmp_path)
    result = _quod(tmp_path, "show")
    assert result.exit_code == 0, result.output
    assert "binary_units:" in result.output
    assert "libdemo.so" in result.output


def test_show_binary_force_renders_section(tmp_path):
    """`--binary` forces the section to render — non-empty case prints
    contents (libdemo.so name shows up); empty case would print
    `(none)` placeholder. Other sections still render per their
    default rules."""
    _write_project(tmp_path)
    result = _quod(tmp_path, "show", "--binary")
    assert result.exit_code == 0, result.output
    assert "binary_units:" in result.output
    assert "libdemo.so" in result.output


def test_show_no_binary_suppresses_binary_units(tmp_path):
    """--no-binary explicitly hides the section even when populated."""
    _write_project(tmp_path)
    result = _quod(tmp_path, "show", "--no-binary")
    assert result.exit_code == 0, result.output
    assert "binary_units:" not in result.output
    # other sections still render.
    assert "source_units:" in result.output


def test_show_source_and_binary_compose(tmp_path):
    """Pre-redesign mutex; now both render side by side."""
    _write_project(tmp_path)
    result = _quod(tmp_path, "show", "--source", "--binary")
    assert result.exit_code == 0, result.output
    assert "source_units:" in result.output
    assert "binary_units:" in result.output


def test_show_binary_default_omits_pcode_opcode_list(tmp_path):
    """Default rendering of a bin.fn block summary is `[N ops]` without
    the per-opcode comma list — that part is noise at the program-level
    view. The fixture's block contains a CALL + RETURN sequence; in
    summary mode the block header shows the count `[2 ops]` only."""
    _write_project(tmp_path)
    result = _quod(tmp_path, "show", "--binary")
    assert result.exit_code == 0, result.output
    # Block summary header is there with op count.
    assert "[2 ops]" in result.output
    # But the opcodes themselves aren't named in the summary line.
    assert "[2 ops: CALL" not in result.output


def test_show_binary_detail_includes_pcode_opcode_list(tmp_path):
    """--binary-detail brings the comma-separated opcode list back."""
    _write_project(tmp_path)
    result = _quod(tmp_path, "show", "--binary", "--binary-detail")
    assert result.exit_code == 0, result.output
    # With detail on, the opcodes are named in the block header.
    assert "[2 ops: CALL, RETURN]" in result.output


def test_show_binary_detail_off_explicit_matches_default(tmp_path):
    """`--no-binary-detail` is the explicit form of the default; same
    output as omitting the flag."""
    _write_project(tmp_path)
    a = _quod(tmp_path, "show", "--binary")
    b = _quod(tmp_path, "show", "--binary", "--no-binary-detail")
    assert a.exit_code == 0 and b.exit_code == 0
    assert a.output == b.output


# ----- quod fn ls --binary -----


def test_fn_ls_binary_lists_bin_fns(tmp_path):
    _write_project(tmp_path)
    result = _quod(tmp_path, "fn", "ls", "--binary")
    assert result.exit_code == 0, result.output
    assert "greet" in result.output
    assert "libdemo.so" in result.output


def test_fn_ls_binary_json(tmp_path):
    _write_project(tmp_path)
    result = _quod(tmp_path, "fn", "ls", "--binary", "--json")
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert len(rows) == 1
    assert rows[0]["demangled_name"] == "greet"
    assert rows[0]["unit_path"] == "/tmp/build/libdemo.so"
    assert rows[0]["address"] == 0x401120
    assert rows[0]["block_count"] == 1


def test_fn_ls_binary_with_no_binary_units(tmp_path):
    """A program without binary_units — `--binary` just prints the
    empty marker."""
    (tmp_path / "quod.toml").write_text(
        'build_dir  = "build"\n'
        'proofs_dir = "proofs"\n'
        '\n'
        '[[program]]\n'
        'name = "demo"\n'
        'version = "0.1.0"\n'
        'file = "program.json"\n'
    )
    save_program(Program(), tmp_path / "program.json")
    result = _quod(tmp_path, "fn", "ls", "--binary")
    assert result.exit_code == 0, result.output
    assert "(no binary functions)" in result.output


# ----- quod fn show --binary -----


def test_fn_show_binary_renders_bin_fn(tmp_path):
    _write_project(tmp_path)
    result = _quod(tmp_path, "fn", "show", "greet", "--binary")
    assert result.exit_code == 0, result.output
    # format_bin_fn surface form: "bin.fn 0x401120 void greet() [__cdecl] {"
    assert "bin.fn" in result.output
    assert "0x401120" in result.output
    assert "greet" in result.output
    # Decompile text is rendered verbatim under "decompile:".
    assert "decompile:" in result.output
    assert "puts" in result.output


def test_fn_show_binary_json(tmp_path):
    _write_project(tmp_path)
    result = _quod(tmp_path, "fn", "show", "greet", "--binary", "--json")
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed["kind"] == "bin.fn"
    assert parsed["demangled_name"] == "greet"


def test_fn_show_binary_unknown_ref_fails_clearly(tmp_path):
    _write_project(tmp_path)
    result = _quod(tmp_path, "fn", "show", "nonexistent", "--binary")
    assert result.exit_code == 1
    assert "no binary function matches" in result.output


def test_fn_show_three_layer_flags_mutually_exclusive(tmp_path):
    _write_project(tmp_path)
    result = _quod(tmp_path, "fn", "show", "greet", "--source", "--binary")
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_fn_show_binary_elides_block_comments_by_default(tmp_path):
    """Ghidra emits `/* WARNING ... */` and similar block comments in
    its decompile output. The default renderer strips them; the raw
    text in the node is preserved for callers that need it."""
    dump = _libdemo_dump()
    dump["functions"][0]["decompile"] = (
        "/* WARNING: Removing unreachable block (ram,0x12345) */\n"
        "\n"
        "void greet(void)\n"
        "{\n"
        "  puts(\"hi\");\n"
        "  return;\n"
        "}\n"
    )
    dump_path = tmp_path / "libdemo.json"
    dump_path.write_text(json.dumps(dump))
    program = ingest_binary_dump(dump_path)
    save_program(program, tmp_path / "program.json")
    (tmp_path / "quod.toml").write_text(
        'build_dir  = "build"\n'
        'proofs_dir = "proofs"\n'
        '\n'
        '[[program]]\n'
        'name = "demo"\n'
        'version = "0.1.0"\n'
        'file = "program.json"\n'
    )

    default = _quod(tmp_path, "fn", "show", "greet", "--binary")
    assert default.exit_code == 0
    assert "WARNING" not in default.output
    assert "Removing unreachable block" not in default.output
    # The actual decompiled body must still render.
    assert "void greet(void)" in default.output


def test_fn_show_binary_raw_decompile_keeps_block_comments(tmp_path):
    """`--raw-decompile` opt-out — the full decompile text including
    `/* */` blocks renders verbatim. Useful when diagnosing what
    Ghidra actually emitted (e.g., the warnings name addresses that
    point at unanalyzable code)."""
    dump = _libdemo_dump()
    dump["functions"][0]["decompile"] = (
        "/* WARNING: Removing unreachable block (ram,0x12345) */\n"
        "void greet(void) { puts(\"hi\"); }\n"
    )
    dump_path = tmp_path / "libdemo.json"
    dump_path.write_text(json.dumps(dump))
    program = ingest_binary_dump(dump_path)
    save_program(program, tmp_path / "program.json")
    (tmp_path / "quod.toml").write_text(
        'build_dir  = "build"\n'
        'proofs_dir = "proofs"\n'
        '\n'
        '[[program]]\n'
        'name = "demo"\n'
        'version = "0.1.0"\n'
        'file = "program.json"\n'
    )

    raw = _quod(tmp_path, "fn", "show", "greet", "--binary", "--raw-decompile")
    assert raw.exit_code == 0
    assert "WARNING" in raw.output
    assert "Removing unreachable block" in raw.output


def test_fn_show_raw_decompile_requires_binary(tmp_path):
    """`--raw-decompile` is meaningless without `--binary` — refuse
    rather than silently ignoring the flag."""
    _write_project(tmp_path)
    result = _quod(tmp_path, "fn", "show", "greet", "--raw-decompile")
    assert result.exit_code == 2
    assert "only applies to --binary" in result.output


def test_fn_show_binary_json_keeps_full_decompile_text(tmp_path):
    """Display-only filtering: the underlying `BinFunction.decompile_text`
    in the node is preserved verbatim — `--json` must still emit the
    `/* */` block comments because layer-A's preserve-verbatim rule
    holds for the data, not just the renderer."""
    dump = _libdemo_dump()
    dump["functions"][0]["decompile"] = (
        "/* WARNING: text in node */\nvoid greet(void) {}\n"
    )
    dump_path = tmp_path / "libdemo.json"
    dump_path.write_text(json.dumps(dump))
    program = ingest_binary_dump(dump_path)
    save_program(program, tmp_path / "program.json")
    (tmp_path / "quod.toml").write_text(
        'build_dir  = "build"\n'
        'proofs_dir = "proofs"\n'
        '\n'
        '[[program]]\n'
        'name = "demo"\n'
        'version = "0.1.0"\n'
        'file = "program.json"\n'
    )
    result = _quod(tmp_path, "fn", "show", "greet", "--binary", "--json")
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert "WARNING" in parsed["decompile_text"]


# ----- quod ingest dispatch on kind -----


def test_ingest_callback_refuses_unknown_kind(tmp_path):
    """The bare `quod ingest` callback dispatches on `entry.kind`. An
    unknown kind fails with a message naming the supported set."""
    (tmp_path / "x.txt").write_text("noop")
    (tmp_path / "quod.toml").write_text(
        'build_dir  = "build"\n'
        'proofs_dir = "proofs"\n'
        '\n'
        '[[program]]\n'
        'name = "demo"\n'
        'version = "0.1.0"\n'
        'file = "program.json"\n'
        '\n'
        '[[ingest.entry]]\n'
        'kind = "rust-crate"\n'
        'source = "x.txt"\n'
    )
    result = _quod(tmp_path, "ingest")
    assert result.exit_code == 1
    assert "rust-crate" in result.output
    assert "binary" in result.output  # supported set named in the error


def test_ingest_binary_subcommand_help_lists_options(tmp_path):
    """The `quod ingest binary` subcommand exists and surfaces the
    documented options. We don't actually invoke it here (Ghidra
    analysis takes seconds; the real e2e lives in test_binary_e2e)."""
    result = _quod(tmp_path, "ingest", "binary", "--help")
    assert result.exit_code == 0
    assert "--keep-dump" in result.output
    assert "PyGhidra" in result.output or "Ghidra" in result.output
