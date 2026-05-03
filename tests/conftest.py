"""End-to-end test collector for quod.

A "case" is a JSON file under `tests/cases/` describing a test. Pytest
auto-collects every `*.json` under `cases/` via the hook below — drop a
file in the right shape and it becomes a test, no Python required.

Two case shapes share one collector:

1. **Behavior** — compile a program to a binary and run it:

    {
      "program_file": "../../examples/basics/helloworld.json",  # or:
      "program":      { ...inline Program JSON... },
      "entry":   "main",          # default "main"
      "args":    ["foo", "bar"],  # argv to forward; default []
      "stdin":   "",              # blob piped to stdin; default ""
      "expect":  {
        "stdout": "hello, world\n",   # exact match; omit to skip
        "exit":   0                    # default 0
      }
    }

2. **CLI** — exercise the quod CLI in-process via Typer's CliRunner.
   Single-step (sugar):

    {
      "cli":        ["fn", "ls", "--json"],
      "in_program": "before.json",  # copied into a sandbox; relative to case
      "expect": {
        "exit": 0,
        "stdout":      "...",       # exact, OR:
        "stdout_json": [...]        # structural JSON compare
      }
    }

   Multi-step (mutation flows):

    {
      "in_program": "before.json",
      "steps": [
        { "cli": ["claim", "add", "..."], "expect": { "exit": 0 } },
        { "cli": ["fn", "ls", "--json"], "expect": { "stdout_json": [...] } }
      ],
      "expect": { "program_json": "after.json" }   # final file state
    }

A case file may also hold a JSON list — each element becomes its own
test item, named by `name` or positionally.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

from quod.lower import compile_program
from quod.model import Program


_CASES_ROOT = Path(__file__).parent / "cases"


def pytest_collect_file(parent, file_path: Path):
    if file_path.suffix != ".json":
        return None
    try:
        file_path.resolve().relative_to(_CASES_ROOT.resolve())
    except ValueError:
        return None
    return CaseFile.from_parent(parent, path=file_path)


_CASE_KEYS = frozenset({"cli", "steps", "program", "program_file", "c_file"})


def _looks_like_case(obj: Any) -> bool:
    return isinstance(obj, dict) and bool(_CASE_KEYS & obj.keys())


class CaseFile(pytest.File):
    def collect(self):
        data = json.loads(self.path.read_text())
        cases = data if isinstance(data, list) else [data]
        # Skip support files (e.g. before.json / after.json sitting next to a
        # case.json) — they're plain Program JSON, not case definitions.
        if not all(_looks_like_case(c) for c in cases):
            return
        for i, case in enumerate(cases):
            name = case.get("name") or (self.path.stem if len(cases) == 1 else f"case{i}")
            yield CaseItem.from_parent(self, name=name, case=case)


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Patch the terminal reporter so its compact-mode progress display
    groups items by *directory*, not by individual case JSON file. The
    actual item nodeids are unchanged — `pytest tests/cases/lang/arena/
    scratch.json::scratch` still selects correctly — but the per-line
    fspath shown during a run collapses every case in a folder onto one
    line (e.g. `tests/cases/lang/arena .....`) so the output stays scannable
    as the corpus grows. trylast= so terminal's own pytest_configure (which
    registers the reporter under that name) runs first."""
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None or not hasattr(reporter, "write_fspath_result"):
        return
    original = reporter.write_fspath_result

    def grouped(nodeid: str, res, **markup):
        parts = nodeid.split("::", 1)
        file_part = parts[0]
        if file_part.endswith(".json"):
            file_part = str(Path(file_part).parent)
        regrouped = file_part + (("::" + parts[1]) if len(parts) > 1 else "")
        return original(regrouped, res, **markup)

    reporter.write_fspath_result = grouped


class CaseItem(pytest.Item):
    def __init__(self, *, case: dict[str, Any], **kw):
        super().__init__(**kw)
        self.case = case
        self._failure_blob: list[str] = []

    def runtest(self) -> None:
        if "cli" in self.case or "steps" in self.case:
            _run_cli_case(self)
        elif (
            "program_file" in self.case or "program" in self.case
            or "c_file" in self.case
        ):
            _run_behavior_case(self)
        else:
            raise ValueError(
                "case has none of: 'cli', 'steps', 'program', 'program_file', 'c_file'"
            )

    def repr_failure(self, excinfo):
        if not self._failure_blob:
            return super().repr_failure(excinfo)
        return "\n".join([f"case {self.name!r} failed", *self._failure_blob])

    def reportinfo(self):
        return self.path, 0, f"case: {self.name}"


# ---------- Behavior cases ----------

def _run_behavior_case(item: CaseItem) -> None:
    case = item.case
    expect = case.get("expect", {})
    expected_stdout = expect.get("stdout")
    expected_exit = expect.get("exit", 0)
    expected_program_json = expect.get("program_json")
    expected_ingest_error = expect.get("ingest_error")

    # Negative-path: case asserts the loader (typically ingest_c) raises with
    # a message containing `expected_ingest_error`. Skip everything else.
    if expected_ingest_error is not None:
        try:
            _load_program(case, item.path.parent)
        except Exception as exc:
            if expected_ingest_error not in str(exc):
                pytest.fail(
                    f"case {item.name!r}: ingest error\n"
                    f"  expected substring: {expected_ingest_error!r}\n"
                    f"  got message:        {str(exc)!r}",
                    pytrace=False,
                )
            return
        pytest.fail(
            f"case {item.name!r}: expected ingest to fail with "
            f"{expected_ingest_error!r} but it succeeded",
            pytrace=False,
        )

    program = _load_program(case, item.path.parent)
    entry = case.get("entry", "main")
    args = [str(a) for a in case.get("args", [])]
    stdin_blob = case.get("stdin", "")

    # Pre-build assertion: compare the loaded Program against an expected
    # JSON snapshot. Lets c_file cases verify the ingester's output without
    # also running the binary, and cross-checks behavior cases too.
    if expected_program_json is not None:
        actual = json.loads(program.model_dump_json())
        expected = _resolve_program_ref(expected_program_json, item.path.parent)
        if actual != expected:
            item._failure_blob.append(
                "  program_json mismatch (loaded program ≠ expected):\n"
                + _json_diff(actual, expected, indent="    ")
            )
            pytest.fail("\n".join([f"case {item.name!r} failed", *item._failure_blob]), pytrace=False)

    # Skip the build/run leg when the case is purely a snapshot check (no
    # stdout/exit assertion either). Useful for ingester-only fixtures.
    if (
        expected_stdout is None
        and "exit" not in expect
        and expected_program_json is not None
    ):
        return

    with tempfile.TemporaryDirectory(prefix=f"quod-test-{item.name}-") as td:
        result = compile_program(
            program,
            build_dir=Path(td),
            bins=((item.name, entry),),
            profile=2,
            link=True,
        )
        binary = result.bins[0].binary
        assert binary is not None, f"compile_program produced no binary for {item.name!r}"
        completed = subprocess.run(
            [str(binary), *args],
            input=stdin_blob,
            capture_output=True,
            text=True,
            timeout=30,
        )

    item._failure_blob = [
        f"  exit:   got {completed.returncode}, expected {expected_exit}",
    ]
    if expected_stdout is not None:
        item._failure_blob.append(f"  stdout: got {completed.stdout!r}")
        item._failure_blob.append(f"          exp {expected_stdout!r}")
    if completed.stderr:
        item._failure_blob.append(f"  stderr: {completed.stderr!r}")

    if expected_stdout is not None:
        assert completed.stdout == expected_stdout
    assert completed.returncode == expected_exit


def _load_program(case: dict[str, Any], case_dir: Path) -> Program:
    sources = [k for k in ("program", "program_file", "c_file") if k in case]
    if len(sources) != 1:
        raise ValueError(
            f"case must set exactly one of 'program' / 'program_file' / 'c_file', got {sources}"
        )
    if "program" in case:
        return Program.model_validate_json(json.dumps(case["program"]))
    if "program_file" in case:
        path = (case_dir / case["program_file"]).resolve()
        return Program.model_validate_json(path.read_text())
    if "c_file" in case:
        from quod.ingest.c import ingest_c
        path = (case_dir / case["c_file"]).resolve()
        clang_args = tuple(case.get("clang_args", ()))
        program = ingest_c(path, clang_args=clang_args)
        # Post-ingest hook: a c_file case may declare extra `imports` so the
        # ingested program can call into stdlib (core.str etc.) — the C source
        # itself can't express that, but the resulting Program can.
        extra_imports = case.get("imports")
        if extra_imports:
            program = program.model_copy(update={"imports": tuple(extra_imports)})
        return program
    raise AssertionError("unreachable")


# ---------- CLI cases ----------

_SANDBOX_TOML = """\
build_dir  = "build"
proofs_dir = "proofs"

[[program]]
name = "test"
version = "0.1.0"
file = "program.json"
"""


def _run_cli_case(item: CaseItem) -> None:
    case = item.case
    steps = _normalize_steps(case)
    in_program = case.get("in_program")

    with tempfile.TemporaryDirectory(prefix=f"quod-cli-{item.name}-") as td:
        sandbox = Path(td)
        toml_path = sandbox / "quod.toml"
        toml_path.write_text(_SANDBOX_TOML)
        program_path = sandbox / "program.json"

        if in_program is not None:
            src = (item.path.parent / in_program).resolve()
            shutil.copy(src, program_path)
        elif "program" in case:
            program_path.write_text(json.dumps(case["program"]))

        for i, step in enumerate(steps):
            _run_one_cli_step(item, step, toml_path, i)

        # Final whole-file assertion (multi-step, after a mutation).
        outer_expect = case.get("expect", {}) if "steps" in case else {}
        ref = outer_expect.get("program_json")
        if ref is not None:
            actual = json.loads(program_path.read_text())
            expected = _resolve_program_ref(ref, item.path.parent)
            if actual != expected:
                item._failure_blob.append(
                    "  program_json mismatch (final file ≠ expected):\n"
                    + _json_diff(actual, expected, indent="    ")
                )
                pytest.fail("\n".join(item._failure_blob), pytrace=False)


def _normalize_steps(case: dict[str, Any]) -> list[dict[str, Any]]:
    if "steps" in case and "cli" in case:
        raise ValueError("case must set exactly one of 'cli' / 'steps'")
    if "steps" in case:
        return list(case["steps"])
    return [{"cli": case["cli"], "expect": case.get("expect", {})}]


def _run_one_cli_step(item: CaseItem, step: dict[str, Any], toml_path: Path, index: int) -> None:
    from typer.testing import CliRunner
    from quod import cli as cli_mod

    cli_mod._state.clear()  # cli.py caches Config across invocations; force a fresh load
    args = ["-c", str(toml_path), *[str(a) for a in step["cli"]]]
    runner = CliRunner()
    result = runner.invoke(cli_mod.app, args, input=step.get("stdin", ""))

    expect = step.get("expect", {})
    expected_exit = expect.get("exit", 0)
    expected_stdout = expect.get("stdout")
    expected_stdout_json = expect.get("stdout_json", _MISSING)

    label = f"step[{index}] {' '.join(args)!r}"
    item._failure_blob.append(f"  {label}")
    item._failure_blob.append(f"    exit:   got {result.exit_code}, expected {expected_exit}")
    if expected_stdout is not None:
        item._failure_blob.append(f"    stdout: got {result.stdout!r}")
        item._failure_blob.append(f"            exp {expected_stdout!r}")
    if result.stderr:
        item._failure_blob.append(f"    stderr: {result.stderr!r}")
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        item._failure_blob.append(f"    exception: {result.exception!r}")

    if result.exit_code != expected_exit:
        pytest.fail("\n".join([f"case {item.name!r} failed", *item._failure_blob]), pytrace=False)
    if expected_stdout is not None and result.stdout != expected_stdout:
        pytest.fail("\n".join([f"case {item.name!r} failed", *item._failure_blob]), pytrace=False)
    if expected_stdout_json is not _MISSING:
        try:
            actual = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            item._failure_blob.append(f"    stdout_json: not valid JSON ({e})")
            pytest.fail("\n".join([f"case {item.name!r} failed", *item._failure_blob]), pytrace=False)
        if actual != expected_stdout_json:
            item._failure_blob.append(
                "    stdout_json mismatch:\n"
                + _json_diff(actual, expected_stdout_json, indent="      ")
            )
            pytest.fail("\n".join([f"case {item.name!r} failed", *item._failure_blob]), pytrace=False)

    # Step passed; pop its noise from the failure blob so later steps own the trail.
    while item._failure_blob and not item._failure_blob[-1].startswith("  step["):
        item._failure_blob.pop()
    item._failure_blob.pop()  # the label itself


def _resolve_program_ref(ref: Any, case_dir: Path) -> Any:
    """`program_json` accepts either a path string (relative to case dir) or an inline object."""
    if isinstance(ref, str):
        return json.loads((case_dir / ref).resolve().read_text())
    return ref


def _json_diff(actual: Any, expected: Any, *, indent: str = "") -> str:
    a = json.dumps(actual, indent=2, sort_keys=True).splitlines()
    e = json.dumps(expected, indent=2, sort_keys=True).splitlines()
    import difflib
    diff = difflib.unified_diff(e, a, fromfile="expected", tofile="actual", lineterm="")
    return "\n".join(indent + line for line in diff)


_MISSING = object()
