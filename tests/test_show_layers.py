"""End-to-end tests for `quod show` tri-state section flags.

`quod show` exposes one tri-state flag per Program collection:

  --<name>      force-render the section (renders `(none)` if empty)
  --no-<name>   suppress the section
  (omitted)     render iff non-empty (default)

Twelve roots (source / structured / functions / binary / externs /
constants / structs / enums / traits / impls / imports / wirables)
plus three widgets (equivalences / edges / bindings) that are filtered
to relationships involving at least one visible root.

The pre-redesign behavior of `--source` / `--structured` / `--binary`
as mutually-exclusive layer filters is gone — combine them freely,
or reach for the same effect via explicit `--no-X` on the layers
you don't want.
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


def test_show_default_renders_every_populated_section(tmp_path):
    """No flags: every section that has content renders. Empty sections
    are skipped silently."""
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    result = _quod(tmp_path, "show")
    assert result.exit_code == 0, result.output
    assert "source_units:" in result.output
    assert "structured_functions:" in result.output
    assert "functions:" in result.output
    assert "equivalences:" in result.output
    # No binary side ingested, so the section is silently absent.
    assert "binary_units:" not in result.output
    # `--edges` defaults to show-iff-nonempty too — there are A→B edges
    # from the c-ingester, so they show.
    assert "edges:" in result.output


def test_show_no_source_suppresses_source_units(tmp_path):
    """--no-source explicitly hides the section even though it has content."""
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    result = _quod(tmp_path, "show", "--no-source")
    assert result.exit_code == 0, result.output
    assert "source_units:" not in result.output
    # Other sections still render.
    assert "structured_functions:" in result.output
    assert "functions:" in result.output


def test_show_combined_no_flags_zero_in_to_a_layer(tmp_path):
    """The pre-redesign 'filter to source only' is now spelled as
    explicit suppression of the other layers."""
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    result = _quod(
        tmp_path, "show",
        "--no-structured", "--no-functions", "--no-binary",
        "--no-externs", "--no-edges", "--no-equivalences",
    )
    assert result.exit_code == 0, result.output
    assert "source_units:" in result.output
    assert "structured_functions:" not in result.output
    # The leading-space distinguishes section header from
    # `structured_functions:` substring.
    assert "  functions:" not in result.output
    assert "edges:" not in result.output
    assert "equivalences:" not in result.output


def test_show_source_and_structured_compose_now(tmp_path):
    """Pre-redesign these were mutually exclusive; now they compose
    additively (both forced-on, no error)."""
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    result = _quod(tmp_path, "show", "--source", "--structured")
    assert result.exit_code == 0, result.output
    assert "source_units:" in result.output
    assert "structured_functions:" in result.output


def test_show_force_show_renders_none_placeholder_for_empty_section(tmp_path):
    """`--binary` on a program with no binary_units forces the section
    header to render with a `(none)` placeholder."""
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    result = _quod(tmp_path, "show", "--binary")
    assert result.exit_code == 0, result.output
    assert "binary_units:" in result.output
    assert "(none)" in result.output


def test_show_default_skips_empty_sections_silently(tmp_path):
    """Without `--binary`, an empty binary_units section is just absent
    — no header, no placeholder, no noise."""
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    result = _quod(tmp_path, "show")
    assert "binary_units:" not in result.output
    # Same for traits / impls / wirables / imports — none of those
    # are populated in this fixture.
    assert "traits:" not in result.output
    assert "impls:" not in result.output
    assert "wirables:" not in result.output


def test_show_hashes_refuses_section_flags(tmp_path):
    """`--hashes` is a different mode (dumps every node hash) and is
    incompatible with section toggles."""
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    result = _quod(tmp_path, "show", "--source", "--hashes")
    assert result.exit_code == 2
    assert "incompatible" in result.output


def test_show_json_with_no_source_omits_source_units(tmp_path):
    """JSON output mirrors the same hide rule: --no-source clears the
    collection, the model serializer drops the empty key."""
    _write_demo_project(tmp_path)
    assert _quod(tmp_path, "ingest").exit_code == 0
    result = _quod(tmp_path, "show", "--no-source", "--json")
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert "source_units" not in parsed
    # functions still present (default-on, has content).
    assert "functions" in parsed and parsed["functions"]


def test_show_layer_filter_on_pure_core_program(tmp_path):
    """A hand-authored core program with only `functions` populated:
    `--source` forces the empty section to render with `(none)`."""
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
    assert "source_units:" in result.output
    assert "(none)" in result.output
    # The `functions` section is non-empty, so it's there too.
    assert "functions:" in result.output
