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


_SHOW_HELP = (
    "tri-state: --{name} forces the section to render (renders "
    "`(none)` if empty); --no-{name} suppresses; the default is "
    "show-iff-non-empty"
)


# Roots — each maps to a single Program collection.
_ROOT_FLAGS: tuple[tuple[str, str], ...] = (
    ("source", "source_units"),
    ("structured", "structured_functions"),
    ("functions", "functions"),
    ("binary", "binary_units"),
    ("externs", "externs"),
    ("constants", "constants"),
    ("structs", "structs"),
    ("enums", "enums"),
    ("traits", "traits"),
    ("impls", "impls"),
    ("imports", "imports"),
    ("wirables", "wirables"),
)

# Widgets — render only when ≥1 endpoint is in a visible root.
_WIDGET_FLAGS: tuple[tuple[str, str], ...] = (
    ("equivalences", "equivalences"),
    ("edges", "edges"),
    ("bindings", "signature_bindings"),
)


@app.command()
def show(
    hashes: bool = typer.Option(
        False, "--hashes",
        help="Dump every node and its short hash, instead of the program form. "
             "Incompatible with the section flags below.",
    ),
    json_output: bool = typer.Option(False, "--json", help=_JSON_HELP),
    # Roots — tri-state via Optional[bool].
    source: bool | None = typer.Option(None, "--source/--no-source", help=_SHOW_HELP.format(name="source")),
    structured: bool | None = typer.Option(None, "--structured/--no-structured", help=_SHOW_HELP.format(name="structured")),
    functions: bool | None = typer.Option(None, "--functions/--no-functions", help=_SHOW_HELP.format(name="functions")),
    binary: bool | None = typer.Option(None, "--binary/--no-binary", help=_SHOW_HELP.format(name="binary")),
    externs: bool | None = typer.Option(None, "--externs/--no-externs", help=_SHOW_HELP.format(name="externs")),
    constants: bool | None = typer.Option(None, "--constants/--no-constants", help=_SHOW_HELP.format(name="constants")),
    structs: bool | None = typer.Option(None, "--structs/--no-structs", help=_SHOW_HELP.format(name="structs")),
    enums: bool | None = typer.Option(None, "--enums/--no-enums", help=_SHOW_HELP.format(name="enums")),
    traits: bool | None = typer.Option(None, "--traits/--no-traits", help=_SHOW_HELP.format(name="traits")),
    impls: bool | None = typer.Option(None, "--impls/--no-impls", help=_SHOW_HELP.format(name="impls")),
    imports: bool | None = typer.Option(None, "--imports/--no-imports", help=_SHOW_HELP.format(name="imports")),
    wirables: bool | None = typer.Option(None, "--wirables/--no-wirables", help=_SHOW_HELP.format(name="wirables")),
    # Widgets — tri-state via Optional[bool], scoped to relationships
    # touching visible roots.
    equivalences: bool | None = typer.Option(None, "--equivalences/--no-equivalences", help=_SHOW_HELP.format(name="equivalences")),
    edges: bool | None = typer.Option(None, "--edges/--no-edges", help=_SHOW_HELP.format(name="edges")),
    bindings: bool | None = typer.Option(None, "--bindings/--no-bindings", help=_SHOW_HELP.format(name="bindings")),
) -> None:
    """Print the program. Color follows TTY (disable with `quod --no-color`).

    Each section has a tri-state flag with uniform semantics:

      --<name>      force-render the section (renders `(none)` if empty)
      --no-<name>   suppress the section regardless of contents
      (omitted)     render iff non-empty (default)

    Roots (12) are top-level Program collections. Widgets (3) are
    relationships — `equivalences`, `edges`, `signature_bindings` —
    that the renderer filters to entries with at least one endpoint
    in a visible root.

    `--hashes` is a different mode (dumps every node hash) and is
    incompatible with the section flags."""
    from quod.cli.output import _theme

    # Local map of CLI flag → tri-state value.
    flag_values: dict[str, bool | None] = {
        "source": source, "structured": structured, "functions": functions,
        "binary": binary, "externs": externs, "constants": constants,
        "structs": structs, "enums": enums, "traits": traits,
        "impls": impls, "imports": imports, "wirables": wirables,
        "equivalences": equivalences, "edges": edges, "bindings": bindings,
    }
    if hashes and any(v is not None for v in flag_values.values()):
        typer.echo(
            "error: --hashes is incompatible with section flags; --hashes "
            "dumps every node hash regardless of program structure",
            err=True,
        )
        raise typer.Exit(2)

    program = _load()

    if hashes:
        if json_output:
            seen: set[str] = set()
            rows: list[dict] = []
            for hn in walk(program):
                if hn.hash in seen:
                    continue
                seen.add(hn.hash)
                rows.append({"hash": hn.hash, "type": type(hn.node).__name__})
            _emit_json(rows)
            return
        theme = _theme()
        seen2: set[str] = set()
        for hn in walk(program):
            if hn.hash in seen2:
                continue
            seen2.add(hn.hash)
            typer.echo(paint((
                Span(hn.hash[:HASH_DISPLAY_LEN], "hash"),
                Span("  ", "ws"),
                Span(type(hn.node).__name__, "type"),
            ), theme))
        return

    force_show, hide = _resolve_section_flags(flag_values, program)
    program = _filter_program(program, hide)

    if json_output:
        _emit_json(program)
        return

    theme = _theme()
    typer.echo(render(
        format_program_lines(program, force_show=force_show, hide=hide),
        theme=theme, mode="columnar",
    ))


def _resolve_section_flags(
    flag_values: dict[str, bool | None], program: Program,
) -> tuple[frozenset[str], frozenset[str]]:
    """Translate CLI tri-states into (force_show, hide) sets keyed by
    Program-collection name.

    Tri-state rules:
      - True  → add to force_show
      - False → add to hide
      - None  → no-op (default applies in the renderer)

    Widget filtering (edges / equivalences / signature_bindings) is
    handled separately by `_filter_program`: it strips entries whose
    endpoints all land in hidden roots, so the renderer's "show iff
    non-empty" rule does the right thing.
    """
    force_show: set[str] = set()
    hide: set[str] = set()
    name_map: dict[str, str] = {
        cli: collection
        for cli, collection in (*_ROOT_FLAGS, *_WIDGET_FLAGS)
    }
    for cli_name, value in flag_values.items():
        if value is None:
            continue
        collection = name_map[cli_name]
        if value:
            force_show.add(collection)
        else:
            hide.add(collection)
    return frozenset(force_show), frozenset(hide)


def _filter_program(program: Program, hide: frozenset[str]) -> Program:
    """Return a copy of `program` with hidden root collections cleared
    and widget collections (edges / equivalences / signature_bindings)
    filtered to entries with ≥1 endpoint in a visible root.

    The renderer also honors `hide` for sections it walks, so an
    already-cleared collection still gets the right "(none) if force-show"
    behavior — `_filter_program`'s job is just to make the JSON
    output match the rendered view, and to give the widget filter
    a stable program shape to operate on.
    """
    updates: dict[str, object] = {}
    for cli_name, collection in _ROOT_FLAGS:
        if collection in hide:
            updates[collection] = ()
    if updates:
        program = program.model_copy(update=updates)

    visible_ids = _collect_visible_ids(program)

    widget_updates: dict[str, object] = {}
    if "edges" not in hide:
        filtered_edges = tuple(
            e for e in program.edges
            if e.source in visible_ids or e.target in visible_ids
        )
        if filtered_edges != program.edges:
            widget_updates["edges"] = filtered_edges
    if "equivalences" not in hide:
        filtered_eqs = tuple(
            eq for eq in program.equivalences
            if eq.a_node_id in visible_ids or eq.b_node_id in visible_ids
        )
        if filtered_eqs != program.equivalences:
            widget_updates["equivalences"] = filtered_eqs
    if "signature_bindings" not in hide:
        filtered_sbs = tuple(
            sb for sb in program.signature_bindings
            if sb.bin_fn_id in visible_ids or sb.src_fn_id in visible_ids
        )
        if filtered_sbs != program.signature_bindings:
            widget_updates["signature_bindings"] = filtered_sbs

    if widget_updates:
        program = program.model_copy(update=widget_updates)
    return program


def _collect_visible_ids(program: Program) -> frozenset[str]:
    """Walk every still-populated Program collection and return the set
    of node IDs that any edge / equivalence / signature_binding
    endpoint can refer to.

    Hidden roots have already been cleared to `()` by the caller
    (`_filter_program`), so a walk over `program` covers exactly the
    visible IDs. Includes nested IDs (block IDs, lifted CFn IDs,
    etc.) since those also appear as edge/equivalence endpoints."""
    ids: set[str] = set()
    for hn in walk(program):
        node_id = getattr(hn.node, "id", None)
        if isinstance(node_id, str):
            ids.add(node_id)
    return frozenset(ids)


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
