"""State accessors and the program-mutation lock.

Holds the global `_state` dict that the root callback populates from CLI
options, plus the lazy-load wrappers around quod.toml and the program
JSON file. `_exclusive_lock` guards mutation paths against concurrent
writers.
"""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path

import typer

from quod.config import Config, load_config
from quod.model import Program, load_program, save_program


_state: dict[str, object] = {}


def _cfg_path() -> Path:
    return _state["config_path"]  # type: ignore[return-value]


def _cfg() -> Config:
    """Lazy-load quod.toml. Init writes the file; other commands read it."""
    if "config" not in _state:
        try:
            _state["config"] = load_config(_cfg_path())
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
    return _state["config"]  # type: ignore[return-value]


def _selected_program_name() -> str | None:
    return _state.get("program_name")  # type: ignore[return-value]


def _selected_program():
    cfg = _cfg()
    try:
        return cfg.select(_selected_program_name())
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)


def _file_override() -> Path | None:
    """The --file / -f override, if any. When set, all CLI commands
    that load/save a program operate on this file directly — no
    quod.toml or [[program]] entry required. Useful for inspecting
    standalone .json files (e.g. stdlib modules in src/quod/stdlib/)."""
    return _state.get("file_path")  # type: ignore[return-value]


def _path() -> Path:
    f = _file_override()
    if f is not None:
        return f
    cfg = _cfg()
    return cfg.resolve(_selected_program().file)


def _load() -> Program:
    p = _path()
    if not p.exists():
        if _file_override() is not None:
            typer.echo(f"error: {p} does not exist", err=True)
        else:
            typer.echo(f"error: {p} does not exist (run `quod init` first)", err=True)
        raise typer.Exit(1)
    return load_program(p)


def _save(program: Program) -> None:
    save_program(program, _path())


@contextmanager
def _exclusive_lock():
    """Hold an exclusive advisory lock for the duration of a mutation.

    Cooperating quod invocations serialize on this lock to avoid the
    load → mutate → save race where parallel writers clobber each other's
    in-memory state at the save step. The lock lives on a sidecar file
    (`<program>.lock`) so that save_program's atomic rename doesn't break
    the lock by replacing the locked inode.

    Read-only commands don't need the lock — save_program writes atomically
    via tmp + rename, so readers see either the old or new file, never a
    half-written one.
    """
    lock_path = _path().with_suffix(_path().suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)
    with open(lock_path, "rb") as lock_fd:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
