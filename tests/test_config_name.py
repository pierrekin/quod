"""Tests for the required top-level `name` field on Config."""

from __future__ import annotations

import pytest

from quod.config import load_config


def _write(tmp_path, body):
    p = tmp_path / "quod.toml"
    p.write_text(body)
    return p


def test_load_requires_name(tmp_path):
    p = _write(tmp_path, '[build]\nprofile = 2\n')
    with pytest.raises(ValueError, match="missing required top-level key `name`"):
        load_config(p)


def test_load_rejects_empty_name(tmp_path):
    p = _write(tmp_path, 'name = ""\n')
    with pytest.raises(ValueError, match="must be a non-empty string"):
        load_config(p)


def test_load_accepts_name(tmp_path):
    p = _write(tmp_path, 'name = "My Project"\n')
    cfg = load_config(p)
    assert cfg.name == "My Project"


def test_starter_tomls_are_valid(tmp_path):
    """Each starter template should round-trip through load_config."""
    from quod.config import _STARTER_TOMLS
    for tmpl, body in _STARTER_TOMLS.items():
        # The starters reference program.json; touch it so file= validates
        # at load time even though we don't load_program here.
        p = tmp_path / f"{tmpl}.toml"
        p.write_text(body)
        cfg = load_config(p)
        assert cfg.name, f"starter {tmpl!r} produced empty name"
