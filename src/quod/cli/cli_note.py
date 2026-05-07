"""note sub-app — attach free-form notes to functions."""

from __future__ import annotations

import typer

from quod import completion as _comp
from quod.cli.cli_app import note_app
from quod.cli.cli_state import _exclusive_lock, _load, _save
from quod.editor import find_function_ref
from quod.model import replace_function


@note_app.command("add")
def note_add(
    function: str = typer.Argument(..., help="Function name or hash prefix.",
                                    autocompletion=_comp.function_or_hash),
    text: str = typer.Argument(..., help="Note content (free-form intent / TODO / rationale)."),
) -> None:
    """Attach a free-form note to a function."""
    with _exclusive_lock():
        program = _load()
        try:
            fn = find_function_ref(program, function)
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        new_fn = fn.model_copy(update={"notes": fn.notes + (text,)})
        program = replace_function(program, new_fn)
        _save(program)
    typer.echo(f"noted on {fn.name}: {text}")


@note_app.command("rm")
def note_rm(
    function: str = typer.Argument(..., help="Function name or hash prefix.",
                                    autocompletion=_comp.function_or_hash),
    index: int = typer.Argument(..., help="0-based index of the note to remove."),
) -> None:
    """Remove a note by index from a function."""
    with _exclusive_lock():
        program = _load()
        try:
            fn = find_function_ref(program, function)
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        if not 0 <= index < len(fn.notes):
            typer.echo(f"error: index {index} out of range (function has {len(fn.notes)} note(s))", err=True)
            raise typer.Exit(1)
        new_notes = fn.notes[:index] + fn.notes[index + 1:]
        new_fn = fn.model_copy(update={"notes": new_notes})
        program = replace_function(program, new_fn)
        _save(program)
    typer.echo(f"removed note {index} from {fn.name}")
