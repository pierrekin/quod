"""End-to-end tests for `quod ingest`.

Lives outside the JSON case framework because each case needs a custom
quod.toml and a vendored source file in the sandbox — the JSON cases'
hardcoded `_SANDBOX_TOML` doesn't cover that.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quod import cli as cli_mod


def _run(toml_path: Path, *args: str):
    cli_mod._state.clear()
    runner = CliRunner()
    return runner.invoke(cli_mod.app, ["-c", str(toml_path), *args])


def _toml_basic_with_profile() -> str:
    return """\
[build]
profile = 2

[[program]]
name = "p"
version = "0.1.0"
file = "program.json"

[ingest.profile.knr]
clang_args = ["-std=c89", "-Wno-implicit-int"]
"""


def _toml_with_entry(source: str, profile: str = "knr") -> str:
    return _toml_basic_with_profile() + f"""
[[ingest.entry]]
kind = "c-file"
source = "{source}"
profile = "{profile}"
"""


def test_ingest_declarative_creates_program_json():
    with tempfile.TemporaryDirectory(prefix="quod-ingest-test-") as td:
        sandbox = Path(td)
        (sandbox / "vendor").mkdir()
        (sandbox / "vendor" / "hello.c").write_text(
            'int add(int a, int b) { return a + b; }\n'
        )
        toml_path = sandbox / "quod.toml"
        toml_path.write_text(_toml_with_entry("vendor/hello.c"))

        result = _run(toml_path, "ingest")
        assert result.exit_code == 0, result.output
        program_path = sandbox / "program.json"
        assert program_path.exists()

        data = json.loads(program_path.read_text())
        names = {f["name"] for f in data.get("functions", [])}
        assert "add" in names


def test_ingest_declarative_is_deterministic():
    """Re-running on unchanged sources produces an identical program.json."""
    with tempfile.TemporaryDirectory(prefix="quod-ingest-test-") as td:
        sandbox = Path(td)
        (sandbox / "vendor").mkdir()
        (sandbox / "vendor" / "f.c").write_text(
            'int sq(int x) { return x * x; }\n'
        )
        toml_path = sandbox / "quod.toml"
        toml_path.write_text(_toml_with_entry("vendor/f.c"))

        assert _run(toml_path, "ingest").exit_code == 0
        first = (sandbox / "program.json").read_text()
        assert _run(toml_path, "ingest").exit_code == 0
        second = (sandbox / "program.json").read_text()
        assert first == second


def test_ingest_adhoc_merges_into_existing_program():
    """Ad-hoc ingest does not create a project; it merges into an existing one
    and never touches quod.toml."""
    with tempfile.TemporaryDirectory(prefix="quod-ingest-test-") as td:
        sandbox = Path(td)
        (sandbox / "vendor").mkdir()
        (sandbox / "vendor" / "a.c").write_text(
            'int a(int x) { return x + 1; }\n'
        )
        (sandbox / "vendor" / "b.c").write_text(
            'int b(int x) { return x * 2; }\n'
        )
        toml_path = sandbox / "quod.toml"
        toml_path.write_text(_toml_with_entry("vendor/a.c"))
        toml_before = toml_path.read_text()

        assert _run(toml_path, "ingest").exit_code == 0
        assert _run(toml_path, "ingest", "c", str(sandbox / "vendor" / "b.c")).exit_code == 0
        # Both a and b should be in the merged program.
        data = json.loads((sandbox / "program.json").read_text())
        names = {f["name"] for f in data.get("functions", [])}
        assert names == {"a", "b"}
        # quod.toml unchanged by the ad-hoc invocation.
        assert toml_path.read_text() == toml_before


def test_ingest_adhoc_profile_lookup():
    """`--profile` resolves a [ingest.profile.<name>] from quod.toml. Without
    the profile, K&R-style implicit-int would refuse via clang's diagnostics."""
    with tempfile.TemporaryDirectory(prefix="quod-ingest-test-") as td:
        sandbox = Path(td)
        (sandbox / "vendor").mkdir()
        (sandbox / "vendor" / "knr.c").write_text(
            'kr(a, b) { return a + b; }\n'
        )
        toml_path = sandbox / "quod.toml"
        toml_path.write_text(_toml_basic_with_profile())

        # Without profile → fails on implicit-int.
        result = _run(toml_path, "ingest", "c", str(sandbox / "vendor" / "knr.c"))
        assert result.exit_code != 0

        # With profile → succeeds.
        result = _run(
            toml_path, "ingest", "c",
            str(sandbox / "vendor" / "knr.c"), "--profile", "knr",
        )
        assert result.exit_code == 0, result.output


def test_ingest_adhoc_clang_passthrough():
    with tempfile.TemporaryDirectory(prefix="quod-ingest-test-") as td:
        sandbox = Path(td)
        (sandbox / "vendor").mkdir()
        (sandbox / "vendor" / "p.c").write_text('p(a) { return a; }\n')
        toml_path = sandbox / "quod.toml"
        toml_path.write_text(_toml_basic_with_profile())

        result = _run(
            toml_path, "ingest", "c", str(sandbox / "vendor" / "p.c"),
            "--", "-std=c89", "-Wno-implicit-int",
        )
        assert result.exit_code == 0, result.output


def test_ingest_string_constants_namespaced():
    """Two ingests producing string literals don't collide on `.str.0`."""
    with tempfile.TemporaryDirectory(prefix="quod-ingest-test-") as td:
        sandbox = Path(td)
        (sandbox / "vendor").mkdir()
        (sandbox / "vendor" / "s1.c").write_text(
            '#include <stdio.h>\nint f1(void) { printf("one\\n"); return 0; }\n'
        )
        (sandbox / "vendor" / "s2.c").write_text(
            '#include <stdio.h>\nint f2(void) { printf("two\\n"); return 0; }\n'
        )
        toml_path = sandbox / "quod.toml"
        toml_path.write_text(_toml_basic_with_profile())

        assert _run(toml_path, "ingest", "c", str(sandbox / "vendor" / "s1.c")).exit_code == 0
        assert _run(toml_path, "ingest", "c", str(sandbox / "vendor" / "s2.c")).exit_code == 0

        data = json.loads((sandbox / "program.json").read_text())
        constant_names = {c["name"] for c in data.get("constants", [])}
        constant_values = {c["value"] for c in data.get("constants", [])}
        # Both string literals survived as distinct constants.
        assert constant_values == {"one\n", "two\n"}
        assert len(constant_names) == 2


def test_ingest_bare_with_no_entries_errors():
    with tempfile.TemporaryDirectory(prefix="quod-ingest-test-") as td:
        sandbox = Path(td)
        toml_path = sandbox / "quod.toml"
        toml_path.write_text(_toml_basic_with_profile())  # no [[ingest.entry]]

        result = _run(toml_path, "ingest")
        assert result.exit_code != 0
        assert "no [[ingest.entry]]" in result.output


def test_ingest_unknown_profile_in_toml_rejected_at_load():
    with tempfile.TemporaryDirectory(prefix="quod-ingest-test-") as td:
        sandbox = Path(td)
        (sandbox / "vendor").mkdir()
        (sandbox / "vendor" / "x.c").write_text('int x(void) { return 0; }\n')
        toml_path = sandbox / "quod.toml"
        toml_path.write_text(_toml_with_entry("vendor/x.c", profile="undefined_profile"))

        result = _run(toml_path, "ingest")
        assert result.exit_code != 0
        assert "undefined_profile" in result.output
