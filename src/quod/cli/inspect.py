"""Whole-program inspection: show, schema, find."""

from __future__ import annotations

import typer

from quod import completion as _comp
from quod.cli.app import app
from quod.cli.output import _emit_json, _JSON_HELP
from quod.cli.state import _load
from quod.hashing import HASH_DISPLAY_LEN, find_by_prefix, node_hash, short_hash, walk
from quod.model import Program
from quod.render import (
    Span,
    format_function_lines,
    format_program_lines,
    paint,
    render,
)
from quod.schema import render_categories, render_category, render_kind


@app.command()
def show(
    hashes: bool = typer.Option(
        False, "--hashes",
        help="Dump every node and its short hash, instead of the program form.",
    ),
    json_output: bool = typer.Option(False, "--json", help=_JSON_HELP),
    source: bool = typer.Option(
        False, "--source",
        help="Filter the rendering to layer A — `Program.source_units`, "
             "the original source-language subtree. Other function-ish "
             "sections (`structured_functions`, `functions`) are hidden; "
             "program-scaffolding (constants / externs / structs / enums "
             "/ imports / edges / equivalences) and the layer-A section "
             "still render.",
    ),
    structured: bool = typer.Option(
        False, "--structured",
        help="Filter the rendering to layer B — `Program.structured_functions`, "
             "the extension-bearing transcription. Other function-ish "
             "sections are hidden; program-scaffolding still renders.",
    ),
) -> None:
    """Print the program. Color follows TTY (disable with `quod --no-color`).

    By default prints every populated section (the canonical layer-C
    `functions` plus, for C-derived programs, `source_units` and
    `structured_functions`). `--source` and `--structured` filter to a
    single function-section view; both are mutually exclusive with
    each other and with `--hashes` (which dumps node hashes regardless
    of layer)."""
    from quod.cli.output import _theme
    if source and structured:
        typer.echo("error: --source and --structured are mutually exclusive", err=True)
        raise typer.Exit(2)
    if (source or structured) and hashes:
        typer.echo(
            "error: --hashes is layer-independent (dumps every node); "
            "drop --source / --structured to combine, or use them without --hashes",
            err=True,
        )
        raise typer.Exit(2)

    program = _load()
    if source or structured:
        program = _project_to_layer(
            program,
            source=source, structured=structured,
        )

    if json_output:
        if hashes:
            seen: set[str] = set()
            rows: list[dict] = []
            for hn in walk(program):
                if hn.hash in seen:
                    continue
                seen.add(hn.hash)
                rows.append({"hash": hn.hash, "type": type(hn.node).__name__})
            _emit_json(rows)
        else:
            _emit_json(program)
        return
    theme = _theme()
    if hashes:
        seen: set[str] = set()
        for hn in walk(program):
            if hn.hash in seen:
                continue
            seen.add(hn.hash)
            typer.echo(paint((
                Span(hn.hash[:HASH_DISPLAY_LEN], "hash"),
                Span("  ", "ws"),
                Span(type(hn.node).__name__, "type"),
            ), theme))
        return
    typer.echo(render(format_program_lines(program), theme=theme, mode="columnar"))


def _project_to_layer(program: Program, *, source: bool, structured: bool) -> Program:
    """Return a copy of `program` with the function-ish sections
    other than the requested layer's hidden. Program-scaffolding
    (constants, externs, structs, enums, imports, edges,
    equivalences) is preserved — those are layer-independent or
    cross-layer, and hiding them would obscure the context the
    layer's rendering depends on."""
    if source:
        return program.model_copy(update={
            "structured_functions": (),
            "functions": (),
        })
    if structured:
        return program.model_copy(update={
            "source_units": (),
            "functions": (),
        })
    return program


@app.command()
def schema(
    kind: str | None = typer.Argument(
        None,
        help="A node kind (e.g. 'quod.let', 'llvm.binop', 'int_range') for full schema.",
    ),
    category: str | None = typer.Option(
        None, "--category",
        help="A category (statement, expression, type, claim, justification, program) to list its kinds.",
    ),
) -> None:
    """Show the schema for a node kind, a category, or list all categories.

    With no arguments, lists all categories. With --category, lists kinds in
    that category. With a kind argument, shows that kind's required/optional
    fields, types, and a minimal example.
    """
    try:
        if kind is not None:
            typer.echo(render_kind(kind))
        elif category is not None:
            typer.echo(render_category(category))
        else:
            typer.echo(render_categories())
    except KeyError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def find(
    prefix: str = typer.Argument(..., autocompletion=_comp.hash_prefixes),
    json_output: bool = typer.Option(False, "--json", help=_JSON_HELP),
) -> None:
    """Resolve a hash prefix to a node and print it."""
    from quod.cli.output import _theme
    program = _load()
    try:
        node = find_by_prefix(program, prefix)
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)

    if json_output:
        _emit_json({
            "hash": node_hash(node),
            "short": short_hash(node),
            "type": type(node).__name__,
            "node": node,
        })
        return

    theme = _theme()

    def row(label: str, value: str, value_style: str) -> str:
        return paint((
            Span(f"{label}:  ", "meta_label"),
            Span(value, value_style),  # type: ignore[arg-type]
        ), theme)

    typer.echo(row("hash", node_hash(node), "hash"))
    typer.echo(row("short", short_hash(node), "hash"))
    typer.echo(row("type", type(node).__name__, "type"))
    typer.echo(row("json", node.model_dump_json(), "literal_str"))
