"""struct sub-app — operations on struct definitions."""

from __future__ import annotations

import typer
from pydantic import ValidationError

from quod import completion as _comp
from quod.cli.cli_app import struct_app
from quod.cli.cli_output import _emit_json, _JSON_HELP, _theme
from quod.cli.cli_state import _exclusive_lock, _load, _save
from quod.cli.cli_types import _TYPE_NAMES, _parse_type_name
from quod.editor import (
    add_struct_to_program,
    remove_struct_from_program,
)
from quod.model import StructDef, StructField, StructType
from quod.render import paint, struct_def_spans


def _parse_struct_field_spec(spec: str, *, struct_names: tuple[str, ...]) -> StructField:
    """Parse a `name:type` token into a StructField. Type is resolved
    against the built-in widths plus any struct names already in the
    program (a struct can reference other structs defined earlier)."""
    if ":" not in spec:
        raise typer.BadParameter(
            f"field spec must be NAME:TYPE, got {spec!r}"
        )
    name, _, ty_token = spec.partition(":")
    if not name:
        raise typer.BadParameter(f"missing field name in {spec!r}")
    return StructField(name=name, type=_parse_type_name(ty_token, struct_names=struct_names))


@struct_app.command("ls")
def struct_ls(
    json_output: bool = typer.Option(False, "--json", help=_JSON_HELP),
) -> None:
    """List declared structs with their field signatures."""
    program = _load()
    if json_output:
        _emit_json(list(program.structs))
        return
    if not program.structs:
        typer.echo("(no structs)")
        return
    theme = _theme()
    for sd in program.structs:
        typer.echo(paint(struct_def_spans(sd), theme))


@struct_app.command("show")
def struct_show(
    name: str = typer.Argument(..., help="Struct name.",
                               autocompletion=_comp.struct_names),
    json_output: bool = typer.Option(False, "--json", help=_JSON_HELP),
) -> None:
    """Print one struct definition."""
    program = _load()
    sd = next((s for s in program.structs if s.name == name), None)
    if sd is None:
        typer.echo(f"error: no struct named {name!r}", err=True)
        raise typer.Exit(1)
    if json_output:
        _emit_json(sd)
        return
    theme = _theme()
    typer.echo(paint(struct_def_spans(sd), theme))


@struct_app.command("add")
def struct_add(
    name: str = typer.Argument(..., help="Struct name (e.g. 'Arena')."),
    fields: list[str] = typer.Argument(
        ..., help="Fields as NAME:TYPE tokens, e.g. base:i8_ptr cur:i8_ptr.",
    ),
) -> None:
    """Define a new struct.

    Field types are int widths (i1/i8/i16/i32/i64), `i8_ptr`, or any struct
    already defined in the program. The named struct is appended to the
    program; the model validator catches dangling refs and cycles before
    the file is written.
    """
    with _exclusive_lock():
        program = _load()
        if any(sd.name == name for sd in program.structs):
            typer.echo(f"error: struct {name!r} already declared", err=True)
            raise typer.Exit(1)
        struct_names = tuple(sd.name for sd in program.structs)
        try:
            field_nodes = tuple(
                _parse_struct_field_spec(s, struct_names=struct_names) for s in fields
            )
        except typer.BadParameter as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        try:
            sd = StructDef(name=name, fields=field_nodes)
            program = add_struct_to_program(program, sd)
        except (ValueError, ValidationError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    field_summary = ", ".join(f"{f.name}:{_format_field_type(f.type)}" for f in field_nodes)
    typer.echo(f"declared struct {name} {{ {field_summary} }}")


@struct_app.command("rm")
def struct_rm(
    name: str = typer.Argument(..., help="Struct name to remove.",
                               autocompletion=_comp.struct_names),
) -> None:
    """Remove a struct definition. Strict: refuses if anything references it."""
    with _exclusive_lock():
        program = _load()
        try:
            program = remove_struct_from_program(program, name)
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"removed struct {name}")


@struct_app.command("rename")
def struct_rename(
    old: str = typer.Argument(..., help="Existing struct name.",
                              autocompletion=_comp.struct_names),
    new: str = typer.Argument(..., help="New struct name."),
) -> None:
    """Rename a struct and update every reference (StructType, StructInit)."""
    from quod.editor import rename_struct
    with _exclusive_lock():
        program = _load()
        try:
            program = rename_struct(program, old, new)
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"renamed struct {old} -> {new}")


def _format_field_type(t) -> str:
    """Short rendering of a struct field's type for the `quod struct add` ack."""
    for tok, cls in _TYPE_NAMES.items():
        if isinstance(t, cls):
            return tok
    if isinstance(t, StructType):
        return t.name
    from quod.model import EnumType
    if isinstance(t, EnumType):
        return t.name
    return repr(t)
