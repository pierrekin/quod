"""stmt sub-app — operations on statements within a function."""

from __future__ import annotations

import typer

from quod import completion as _comp
from quod.cli.cli_app import stmt_app
from quod.cli.cli_state import _exclusive_lock, _load, _save
from quod.editor import (
    add_statement_in_function,
    find_function_ref,
    parse_statement_spec,
    read_json_arg,
    remove_statement_in_function,
)


@stmt_app.command("add")
def stmt_add(
    function: str = typer.Argument(..., help="Function name or hash prefix.",
                                    autocompletion=_comp.function_or_hash),
    spec: str = typer.Argument("-", help="Path to JSON spec, or '-' for stdin."),
    at_end: bool = typer.Option(False, "--at-end"),
    at_start: bool = typer.Option(False, "--at-start"),
    before: str | None = typer.Option(None, "--before", help="Hash prefix of an existing statement."),
    after: str | None = typer.Option(None, "--after", help="Hash prefix of an existing statement."),
) -> None:
    """Insert a statement into a function. Exactly one anchor is required."""
    anchors = [at_end, at_start, before is not None, after is not None]
    if sum(map(bool, anchors)) != 1:
        typer.echo("error: pass exactly one of --at-end, --at-start, --before, --after", err=True)
        raise typer.Exit(2)
    with _exclusive_lock():
        program = _load()
        try:
            fn = find_function_ref(program, function)
            stmt = parse_statement_spec(read_json_arg(spec))
            if at_end:
                program = add_statement_in_function(program, fn, stmt, where="end")
            elif at_start:
                program = add_statement_in_function(program, fn, stmt, where="start")
            elif before is not None:
                program = add_statement_in_function(program, fn, stmt, where="before", anchor_ref=before)
            else:
                program = add_statement_in_function(program, fn, stmt, where="after", anchor_ref=after)
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"added statement to {fn.name}")


@stmt_app.command("rm")
def stmt_rm(
    function: str = typer.Argument(..., help="Function name or hash prefix.",
                                    autocompletion=_comp.function_or_hash),
    hash_prefix: str = typer.Argument(
        ..., help="Content-hash prefix of the statement to remove.",
        autocompletion=_comp.hash_prefixes,
    ),
) -> None:
    """Remove a statement from a function by content-hash prefix.

    Find the hash via `quod fn show FN` (each statement is shown with its
    short hash) or `quod show --hashes`.
    """
    with _exclusive_lock():
        program = _load()
        try:
            fn = find_function_ref(program, function)
            program = remove_statement_in_function(program, fn, hash_prefix)
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"removed statement {hash_prefix} from {fn.name}")
