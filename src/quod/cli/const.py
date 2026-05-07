"""const sub-app — operations on string constants."""

from __future__ import annotations

import typer

from quod import completion as _comp
from quod.cli.app import const_app
from quod.cli.output import _emit_json, _JSON_HELP, _theme
from quod.cli.state import _exclusive_lock, _load, _save
from quod.editor import (
    add_constant_to_program,
    remove_constant_from_program,
)
from quod.model import StringConstant
from quod.render import Span, constant_spans, hash_brackets, paint


@const_app.command("ls")
def const_ls(
    json_output: bool = typer.Option(False, "--json", help=_JSON_HELP),
) -> None:
    """List declared string constants."""
    program = _load()
    if json_output:
        _emit_json(list(program.constants))
        return
    if not program.constants:
        typer.echo("(no constants)")
        return
    theme = _theme()
    for c in program.constants:
        typer.echo(paint((
            *hash_brackets(c), Span(" ", "ws"), *constant_spans(c),
        ), theme))


@const_app.command("add")
def const_add(
    name: str = typer.Argument(..., help="Constant name (e.g. '.str.fmt')."),
    value: str = typer.Argument(..., help="Constant value (raw string; not C-escaped)."),
) -> None:
    """Declare a string constant. Reference it from code with quod.string_ref.

    The value is the raw string as you want it in the program. To embed a
    newline, pass an actual newline (the shell will likely need $'...\\n' or
    a heredoc). Quod adds a trailing NUL byte automatically when lowering.
    """
    with _exclusive_lock():
        program = _load()
        try:
            program = add_constant_to_program(program, StringConstant(name=name, value=value))
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"declared constant {name} = {value!r}")


@const_app.command("rm")
def const_rm(
    name: str = typer.Argument(..., help="Constant name to remove.",
                               autocompletion=_comp.constant_names),
) -> None:
    """Remove a string constant from the program.

    Permissive: doesn't refuse if a quod.string_ref still points at it. The
    dangling reference surfaces at `quod build` time.
    """
    with _exclusive_lock():
        program = _load()
        try:
            program = remove_constant_from_program(program, name)
        except KeyError as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"removed constant {name}")


@const_app.command("rename")
def const_rename(
    old: str = typer.Argument(..., help="Existing constant name.",
                              autocompletion=_comp.constant_names),
    new: str = typer.Argument(..., help="New constant name."),
) -> None:
    """Rename a string constant and update every quod.string_ref to it."""
    from quod.editor import rename_constant
    with _exclusive_lock():
        program = _load()
        try:
            program = rename_constant(program, old, new)
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"renamed constant {old} -> {new}")
