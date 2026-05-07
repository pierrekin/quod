"""enum sub-app — operations on enum (sum-type) definitions."""

from __future__ import annotations

import typer

from quod.cli.app import enum_app
from quod.cli.output import _emit_json, _JSON_HELP
from quod.cli.state import _exclusive_lock, _load, _save
from quod.editor import (
    add_enum_to_program,
    parse_enum_spec,
    read_json_arg,
    remove_enum_from_program,
)
from quod.hashing import short_hash


@enum_app.command("ls")
def enum_ls(
    json_output: bool = typer.Option(False, "--json", help=_JSON_HELP),
) -> None:
    """List declared enums with their variants."""
    from quod.model import format_enum_def
    program = _load()
    if json_output:
        _emit_json(list(program.enums))
        return
    if not program.enums:
        typer.echo("(no enums)")
        return
    for ed in program.enums:
        typer.echo(format_enum_def(ed))


@enum_app.command("show")
def enum_show(
    name: str = typer.Argument(..., help="Enum name."),
    json_output: bool = typer.Option(False, "--json", help=_JSON_HELP),
) -> None:
    """Print one enum definition."""
    from quod.model import format_enum_def
    program = _load()
    ed = next((e for e in program.enums if e.name == name), None)
    if ed is None:
        typer.echo(f"error: no enum named {name!r}", err=True)
        raise typer.Exit(1)
    if json_output:
        _emit_json(ed)
        return
    typer.echo(format_enum_def(ed))


@enum_app.command("add")
def enum_add(
    spec: str = typer.Argument("-", help="Path to JSON EnumDef spec, or '-' for stdin."),
) -> None:
    """Append a new enum.

    The CLI surface for enums is JSON-only — variant payloads have
    enough structure that the shorthand `name:type` form for structs
    doesn't generalize cleanly. Author the EnumDef as a JSON object
    and pipe it in: `cat enum.json | quod enum add -`.

    See `quod schema EnumDef` for the canonical shape.
    """
    with _exclusive_lock():
        program = _load()
        try:
            ed = parse_enum_spec(read_json_arg(spec))
            program = add_enum_to_program(program, ed)
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    var_summary = ", ".join(v.name for v in ed.variants)
    typer.echo(f"declared enum {ed.name} {{ {var_summary} }} (hash={short_hash(ed)})")


@enum_app.command("rm")
def enum_rm(
    name: str = typer.Argument(..., help="Enum name to remove."),
) -> None:
    """Remove an enum definition. Strict: refuses if anything references it."""
    with _exclusive_lock():
        program = _load()
        try:
            program = remove_enum_from_program(program, name)
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"removed enum {name}")


@enum_app.command("rename")
def enum_rename(
    old: str = typer.Argument(..., help="Existing enum name."),
    new: str = typer.Argument(..., help="New enum name."),
) -> None:
    """Rename an enum and update every reference (EnumType, EnumInit)."""
    from quod.editor import rename_enum
    with _exclusive_lock():
        program = _load()
        try:
            program = rename_enum(program, old, new)
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"renamed enum {old} -> {new}")


@enum_app.command("rename-variant")
def enum_rename_variant(
    enum_name: str = typer.Argument(..., help="Enum the variant belongs to."),
    old: str = typer.Argument(..., help="Existing variant name."),
    new: str = typer.Argument(..., help="New variant name."),
) -> None:
    """Rename a variant within an enum.

    Updates the variant in the EnumDef, every EnumInit that names it
    against this enum, and every Match arm that targets this enum.

    Match-arm rewriting uses a structural heuristic: an arm is rewritten
    only if all of that match's arm names (excluding `_`) are valid
    variant names of the renamed enum. Matches against a different enum
    that happen to share the variant name `old` won't be touched.
    """
    from quod.editor import rename_variant
    with _exclusive_lock():
        program = _load()
        try:
            program = rename_variant(program, enum_name, old, new)
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"renamed variant {enum_name}::{old} -> {enum_name}::{new}")
