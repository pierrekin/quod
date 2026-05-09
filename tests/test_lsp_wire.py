"""Unit tests for the LSP server's custom `quod/*` methods.

Drives the request handlers directly (not through stdio framing) — pygls
exposes them as registered features we can call with synthetic params.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quod.lsp import (
    ActiveProgram,
    QuodServer,
    _program_payload,
    build_server,
)


# ---------- helpers ----------


def _write_project(root: Path, name: str, programs: list[dict]) -> Path:
    """Write a minimal quod.toml + program.json files and return the toml path."""
    parts = [f'name = "{name}"\n']
    for p in programs:
        parts.append(
            "\n[[program]]\n"
            f'name = "{p["name"]}"\n'
            f'version = "{p.get("version", "0.1")}"\n'
            f'file = "{p["file"]}"\n'
        )
    toml = root / "quod.toml"
    toml.write_text("".join(parts))
    for p in programs:
        (root / p["file"]).parent.mkdir(parents=True, exist_ok=True)
        (root / p["file"]).write_text(json.dumps({}))
    return toml


def _server_with_projects(roots: list[Path]) -> QuodServer:
    server = build_server()
    for root in roots:
        added = server.add_project(root)
        assert added, f"add_project failed for {root}"
    return server


def _call(server: QuodServer, method: str, params: dict) -> dict:
    """Call a registered feature directly. pygls stores handlers on the
    protocol's feature manager (`fm.features`); each handler is closed over
    the server, so it just takes `params`."""
    handler = server.protocol.fm.features[method]
    return handler(params)


# ---------- programs payload uses Config.name ----------


def test_programs_payload_uses_config_name(tmp_path):
    _write_project(tmp_path, "My Project", [{"name": "hello", "file": "program.json"}])
    server = _server_with_projects([tmp_path])
    payload = server.programs_payload()
    assert len(payload) == 1
    entry = payload[0]
    assert entry["projectName"] == "My Project"
    assert entry["projectPath"] == str(tmp_path)
    assert entry["name"] == "hello"


def test_programs_payload_distinct_paths_with_same_basename(tmp_path):
    """Two roots whose basename collides must remain independently
    addressable — that's the whole point of identity-by-path."""
    a = tmp_path / "work" / "foo"
    b = tmp_path / "play" / "foo"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    _write_project(a, "Work Foo", [{"name": "p", "file": "program.json"}])
    _write_project(b, "Play Foo", [{"name": "p", "file": "program.json"}])
    server = _server_with_projects([a, b])

    paths = {entry["projectPath"] for entry in server.programs_payload()}
    assert paths == {str(a), str(b)}


# ---------- setActiveProgram ----------


def test_set_active_resolves_by_path(tmp_path):
    a = tmp_path / "work" / "foo"
    b = tmp_path / "play" / "foo"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    _write_project(a, "Work Foo", [{"name": "p", "file": "program.json"}])
    _write_project(b, "Play Foo", [{"name": "p", "file": "program.json"}])
    server = _server_with_projects([a, b])

    resp = _call(server, "quod/setActiveProgram", {
        "projectPath": str(b), "name": "p",
    })
    assert resp["projectName"] == "Play Foo"
    assert resp["projectPath"] == str(b)
    assert resp["via"] == "name"
    assert server.active is not None
    assert server.active.project_path == b


def test_set_active_rejects_unknown_project_path(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    _write_project(a, "A", [{"name": "p", "file": "program.json"}])
    server = _server_with_projects([a])
    with pytest.raises(ValueError, match="no open project at path"):
        _call(server, "quod/setActiveProgram", {
            "projectPath": str(tmp_path / "b"), "name": "p",
        })


def test_set_active_single_project_omits_path(tmp_path):
    """When exactly one project is open, the client can omit projectPath."""
    a = tmp_path / "a"
    a.mkdir()
    _write_project(a, "A", [{"name": "p", "file": "program.json"}])
    server = _server_with_projects([a])
    resp = _call(server, "quod/setActiveProgram", {"name": "p"})
    assert resp["projectName"] == "A"


def test_set_active_multiple_projects_requires_path(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_project(a, "A", [{"name": "p", "file": "program.json"}])
    _write_project(b, "B", [{"name": "p", "file": "program.json"}])
    server = _server_with_projects([a, b])
    with pytest.raises(ValueError, match="multiple projects open"):
        _call(server, "quod/setActiveProgram", {"name": "p"})


def test_set_active_file_mode(tmp_path):
    """`{file}` opens a standalone program — no project context."""
    p = tmp_path / "loose.json"
    p.write_text(json.dumps({}))
    server = build_server()
    resp = _call(server, "quod/setActiveProgram", {"file": str(p)})
    assert resp["projectName"] is None
    assert resp["projectPath"] is None
    assert resp["via"] == "file"


def test_set_active_file_mode_rejects_mixed_args(tmp_path):
    p = tmp_path / "loose.json"
    p.write_text(json.dumps({}))
    server = build_server()
    with pytest.raises(ValueError, match="not both"):
        _call(server, "quod/setActiveProgram", {
            "file": str(p), "projectPath": "/x", "name": "y",
        })


# ---------- getActiveProgramShape ----------


def test_get_active_program_shape_no_active():
    server = build_server()
    shape = _call(server, "quod/getActiveProgramShape", {})
    assert shape == {
        "hasSourceUnits": False,
        "hasBinaryUnits": False,
        "hasEquivalences": False,
    }


def test_get_active_program_shape_after_load(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    _write_project(a, "A", [{"name": "p", "file": "program.json"}])
    server = _server_with_projects([a])
    _call(server, "quod/setActiveProgram", {
        "projectPath": str(a), "name": "p",
    })
    shape = _call(server, "quod/getActiveProgramShape", {})
    # Empty program → all flags False.
    assert shape == {
        "hasSourceUnits": False,
        "hasBinaryUnits": False,
        "hasEquivalences": False,
    }
