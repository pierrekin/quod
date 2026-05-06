"""End-to-end tests for the program-level `quod show --source /
--structured` flags.

Pins the same per-layer view selection as `quod fn show`, but at
the whole-program level: by default `quod show` prints every
populated section; `--source` / `--structured` filter the
function-ish sections to the requested layer. Program scaffolding
(constants, externs, structs, enums, imports, edges, equivalences)
always renders — those are layer-independent or cross-layer.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


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
        '[[ingest.entry]]\n'
        'kind = "c-file"\n'
        'source = "sum.c"\n'
    )


def _quod(root: Path, *args: str) -> subprocess.CompletedProcess:
    from typer.testing import CliRunner
    from quod import cli as cli_mod
    cli_mod._state.clear()
    runner = CliRunner()
    return runner.invoke(
        cli_mod.app,
        ["-c", str(root / "quod.toml"), *args],
    )


def test_show_default_renders_all_three_sections(tmp_path):
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    result = _quod(tmp_path, "show")
    assert result.exit_code == 0, result.output
    assert "source_units:" in result.output
    assert "structured_functions:" in result.output
    assert "functions:" in result.output
    # Cross-layer sections are present too.
    assert "edges:" in result.output
    assert "equivalences:" in result.output


def test_show_source_filters_to_layer_a(tmp_path):
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    result = _quod(tmp_path, "show", "--source")
    assert result.exit_code == 0, result.output
    # Layer A present, others hidden.
    assert "source_units:" in result.output
    assert "structured_functions:" not in result.output
    # Match `functions:` only with the leading indent of the
    # section header — substring match would catch `structured_functions:`
    # too.
    assert "  functions:" not in result.output
    # Cross-layer sections still render.
    assert "edges:" in result.output
    assert "equivalences:" in result.output
    # The actual C source body shows up under c_unit.
    assert "int sum(int n)" in result.output


def test_show_structured_filters_to_layer_b(tmp_path):
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    result = _quod(tmp_path, "show", "--structured")
    assert result.exit_code == 0, result.output
    assert "source_units:" not in result.output
    assert "structured_functions:" in result.output
    # Use a leading-space match to distinguish `functions:` from the
    # `structured_functions:` section header.
    assert "  functions:" not in result.output
    # Layer B keeps `c.for_general`; layer C has it lowered to while.
    assert "c.for_general" in result.output


def test_show_source_and_structured_mutually_exclusive(tmp_path):
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    result = _quod(tmp_path, "show", "--source", "--structured")
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_show_layer_filter_with_hashes_refused(tmp_path):
    """`--hashes` dumps every node and is layer-agnostic. Combining
    with `--source` or `--structured` is incoherent — refuse with a
    message naming the conflict."""
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    result = _quod(tmp_path, "show", "--source", "--hashes")
    assert result.exit_code == 2
    assert "--hashes is layer-independent" in result.output


def test_show_source_with_json_emits_filtered_program(tmp_path):
    """`--source --json` emits a Program JSON with only `source_units`
    populated under the function-ish keys."""
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    result = _quod(tmp_path, "show", "--source", "--json")
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert "source_units" in parsed and parsed["source_units"]
    # The other function-ish keys are absent (drop-when-empty
    # serializer).
    assert "structured_functions" not in parsed
    assert "functions" in parsed and parsed["functions"] == []


def test_show_layer_filter_on_pure_core_program(tmp_path):
    """A hand-authored core program (no source_units / no
    structured_functions) renders empty under `--source` /
    `--structured` but doesn't error — it's a valid query, just
    nothing in that layer."""
    program_path = tmp_path / "program.json"
    program_path.write_text(json.dumps({
        "functions": [{
            "name": "f",
            "params": [],
            "return_type": {"kind": "llvm.i32"},
            "body": {"stmts": [{
                "kind": "quod.return_expr",
                "value": {"kind": "llvm.const_int",
                          "type": {"kind": "llvm.i32"}, "value": 0},
            }]},
            "claims": [],
        }],
    }))
    (tmp_path / "quod.toml").write_text(
        'build_dir  = "build"\n'
        'proofs_dir = "proofs"\n'
        '\n'
        '[[program]]\n'
        'name = "demo"\n'
        'version = "0.1.0"\n'
        'file = "program.json"\n'
    )
    result = _quod(tmp_path, "show", "--source")
    assert result.exit_code == 0, result.output
    # Nothing layer-A to render; output is just the empty-program
    # marker.
    assert "(empty)" in result.output
