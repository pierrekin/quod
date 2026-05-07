"""Output helpers — color/theme, JSON emission, error printing, hashing."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import typer

from quod.cli.cli_state import _state
from quod.render import Theme, ansi_theme, plain_theme


REGIMES = ("axiom", "witness", "lattice")
STORED_REGIMES = ("axiom", "witness")  # lattice is derived, never stored
ENFORCEMENTS = ("trust", "verify")

_JSON_HELP = "Emit machine-readable JSON instead of human-readable output."


def _color_on() -> bool:
    """Color on iff stdout is a TTY, NO_COLOR is unset, and --no-color wasn't passed."""
    if _state.get("no_color"):
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _theme() -> Theme:
    return ansi_theme if _color_on() else plain_theme


def _json_default(o):
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


def _emit_json(payload) -> None:
    """Print a JSON payload. Pydantic models are serialized via model_dump."""
    typer.echo(json.dumps(payload, default=_json_default, indent=2))


def _echo_err(msg: str) -> int:
    """Helper: print to stderr and return exit code 1. Lets call sites
    write `raise typer.Exit(_echo_err(...))` as one expression."""
    typer.echo(f"error: {msg}", err=True)
    return 1


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
