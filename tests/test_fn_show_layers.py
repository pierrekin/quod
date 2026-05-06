"""End-to-end tests for `quod fn show --source / --structured`.

Pins the per-layer view selection: by default `quod fn show <fn>`
prints the canonical core form (`Program.functions`); `--source`
prints the layer-A C subtree (`Program.source_units`); `--structured`
prints the layer-B extension-bearing form (`Program.structured_functions`).
The flags are mutually exclusive and `--json` works with any.
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


def test_fn_show_default_renders_canonical(tmp_path):
    """No flag → canonical core form. Layer-C `Let + While` is what
    sum.c lowers to (the for-loop is gone)."""
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0

    result = _quod(tmp_path, "fn", "show", "sum")
    assert result.exit_code == 0, result.output
    # Canonical body: `let i = 0; while ((i < n)) { ...; i = (i+1); }`
    # — no `c.for_general` since this is layer C.
    assert "while" in result.output
    assert "c.for_general" not in result.output


def test_fn_show_structured_renders_layer_b(tmp_path):
    """--structured → the layer-B form with `c.for_general` preserved."""
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0

    result = _quod(tmp_path, "fn", "show", "sum", "--structured")
    assert result.exit_code == 0, result.output
    # Layer B keeps the C-style for as a CStyleFor extension.
    assert "c.for_general" in result.output


def test_fn_show_source_renders_layer_a_c(tmp_path):
    """--source → the layer-A C subtree (CFn rendered as C-flavored
    text)."""
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0

    result = _quod(tmp_path, "fn", "show", "sum", "--source")
    assert result.exit_code == 0, result.output
    # Layer A renders as C source: `int sum(int n) { for (...) ... }`.
    assert "int sum(int n)" in result.output
    assert "for (int i = 0;" in result.output
    # The structured/canonical-form vocabulary doesn't appear at
    # layer A.
    assert "c.for_general" not in result.output
    assert "quod.let" not in result.output


def test_fn_show_source_and_structured_mutually_exclusive(tmp_path):
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    result = _quod(tmp_path, "fn", "show", "sum", "--source", "--structured")
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_fn_show_json_works_with_each_view(tmp_path):
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0

    canonical = _quod(tmp_path, "fn", "show", "sum", "--json")
    assert canonical.exit_code == 0
    parsed = json.loads(canonical.output)
    # Canonical Function — has `body.stmts`, no CStyleFor.
    assert parsed["name"] == "sum"
    body = parsed["body"]["stmts"]
    assert any(s.get("kind") == "quod.while" for s in body)
    assert not any(s.get("kind") == "c.for_general" for s in body)

    structured = _quod(tmp_path, "fn", "show", "sum", "--structured", "--json")
    assert structured.exit_code == 0
    parsed_b = json.loads(structured.output)
    body_b = parsed_b["body"]["stmts"]
    assert any(s.get("kind") == "c.for_general" for s in body_b)

    source = _quod(tmp_path, "fn", "show", "sum", "--source", "--json")
    assert source.exit_code == 0
    parsed_a = json.loads(source.output)
    # Layer A: a CFn. Body holds layer-A statements.
    assert parsed_a["kind"] == "c.fn"
    assert parsed_a["name"] == "sum"
    body_a = parsed_a["body"]
    assert any(s.get("kind") == "c.for" for s in body_a)


def test_fn_show_source_for_unknown_function_fails_clearly(tmp_path):
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    result = _quod(tmp_path, "fn", "show", "nonexistent", "--source")
    assert result.exit_code == 1
    assert "no layer-A function matches" in result.output


def test_fn_show_structured_for_unknown_function_fails_clearly(tmp_path):
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    result = _quod(tmp_path, "fn", "show", "nonexistent", "--structured")
    assert result.exit_code == 1
    assert "no layer-B function matches" in result.output


def test_fn_show_source_on_program_with_no_source_units_fails(tmp_path):
    """Hand-authored core programs (no C ingest) have no
    `source_units`; `--source` should refuse with a message naming
    the situation."""
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
    result = _quod(tmp_path, "fn", "show", "f", "--source")
    assert result.exit_code == 1
    assert "no layer-A function matches" in result.output
