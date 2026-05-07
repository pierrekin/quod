"""fn sub-app — operations on functions."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from quod import completion as _comp
from quod.cli.app import fn_app
from quod.cli.output import _emit_json, _JSON_HELP, _echo_err, _theme
from quod.cli.state import _exclusive_lock, _load, _save
from quod.editor import (
    add_function_to_program,
    add_statement_in_function,
    find_function_ref,
    parse_function_spec,
    read_json_arg,
)
from quod.hashing import HASH_DISPLAY_LEN, node_hash, short_hash
from quod.model import (
    CFn,
    Function,
    Program,
    claim_param,
    format_c_fn,
    function_callees,
    remove_function,
)
from quod.render import (
    Span,
    format_function_lines,
    function_signature_spans,
    hash_brackets,
    paint,
    render,
)


@fn_app.command("ls")
def fn_ls(
    json_output: bool = typer.Option(False, "--json", help=_JSON_HELP),
) -> None:
    """List all functions with signatures and hashes."""
    program = _load()
    if json_output:
        _emit_json([
            {
                "name": fn.name,
                "hash": node_hash(fn),
                "params": [{"name": p.name, "type": p.type.model_dump(mode="json")} for p in fn.params],
                "return_type": fn.return_type.model_dump(mode="json"),
                "claim_count": len(fn.claims),
            }
            for fn in program.functions
        ])
        return
    if not program.functions:
        typer.echo("(no functions)")
        return
    theme = _theme()
    for fn in program.functions:
        spans = [*hash_brackets(fn), Span(" ", "ws"), *function_signature_spans(fn)]
        if fn.claims:
            spans.append(Span(f"  [{len(fn.claims)} claim(s)]", "meta_label"))
        typer.echo(paint(spans, theme))


@fn_app.command("show")
def fn_show(
    ref: str = typer.Argument(..., autocompletion=_comp.function_or_hash),
    json_output: bool = typer.Option(False, "--json", help=_JSON_HELP),
    source: bool = typer.Option(
        False, "--source",
        help="Render the layer-A C-source subtree from "
             "Program.source_units (the original C, preserved as quod "
             "nodes). Only available for C-derived programs.",
    ),
    structured: bool = typer.Option(
        False, "--structured",
        help="Render the layer-B structured form from "
             "Program.structured_functions (extension-bearing, with "
             "constructs like c.for_general). Only available for "
             "C-derived programs.",
    ),
) -> None:
    """Print a single function. Accepts a name or a content-hash prefix.

    By default prints the canonical core form from `Program.functions`
    — the lowered, `quod.lower`-bound version. `--source` and
    `--structured` select alternate views for C-derived programs
    (mutually exclusive with each other; `--json` works with any of the
    three)."""
    if source and structured:
        typer.echo("error: --source and --structured are mutually exclusive", err=True)
        raise typer.Exit(2)

    program = _load()

    if source:
        cfn = _find_csource_fn_ref(program, ref)
        if json_output:
            _emit_json(cfn)
            return
        typer.echo(format_c_fn(cfn))
        return

    if structured:
        fn = _find_structured_fn_ref(program, ref)
        if json_output:
            _emit_json(fn)
            return
        typer.echo(render(format_function_lines(fn), theme=_theme(), mode="columnar"))
        return

    try:
        fn = find_function_ref(program, ref)
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    if json_output:
        _emit_json(fn)
        return
    typer.echo(render(format_function_lines(fn), theme=_theme(), mode="columnar"))


def _find_csource_fn_ref(program: Program, ref: str) -> CFn:
    """Resolve a function ref against `Program.source_units` (layer A).
    Refs are matched first by name, then by `id` prefix — paralleling
    the canonical-form `find_function_ref` but for layer-A `CFn` nodes
    which don't have content-hash refs (yet)."""
    cfns = [cfn for unit in program.source_units for cfn in unit.functions]
    by_name = [cfn for cfn in cfns if cfn.name == ref]
    if by_name:
        if len(by_name) > 1:
            raise typer.Exit(
                _echo_err(f"ref {ref!r} is ambiguous across source_units: "
                          f"{[cfn.id for cfn in by_name]}")
            )
        return by_name[0]
    by_id = [cfn for cfn in cfns if cfn.id.startswith(ref)]
    if not by_id:
        raise typer.Exit(_echo_err(
            f"no layer-A function matches {ref!r} — either the program "
            f"isn't C-derived (no source_units), or the function uses "
            f"constructs outside the supported layer-A subset (and hit "
            f"the all-or-nothing fallback)"
        ))
    if len({cfn.id for cfn in by_id}) > 1:
        raise typer.Exit(_echo_err(
            f"ref {ref!r} is an ambiguous id prefix: "
            f"{[cfn.id for cfn in by_id]}"
        ))
    return by_id[0]


def _find_structured_fn_ref(program: Program, ref: str) -> Function:
    """Resolve a function ref against `Program.structured_functions`
    (layer B). Same name-or-id-prefix matching as the layer-A and
    canonical helpers."""
    by_name = [fn for fn in program.structured_functions if fn.name == ref]
    if by_name:
        if len(by_name) > 1:
            raise typer.Exit(_echo_err(
                f"ref {ref!r} is ambiguous in structured_functions: "
                f"{[fn.id for fn in by_name]}"
            ))
        return by_name[0]
    by_id = [fn for fn in program.structured_functions if fn.id.startswith(ref)]
    if not by_id:
        raise typer.Exit(_echo_err(
            f"no layer-B function matches {ref!r} — either the program "
            f"isn't C-derived (no structured_functions) or the ref doesn't "
            f"match any function name or id prefix"
        ))
    if len({fn.id for fn in by_id}) > 1:
        raise typer.Exit(_echo_err(
            f"ref {ref!r} is an ambiguous id prefix: "
            f"{[fn.id for fn in by_id]}"
        ))
    return by_id[0]


@fn_app.command("add")
def fn_add(
    spec: str = typer.Argument("-", help="Path to JSON spec, or '-' for stdin."),
    script: str = typer.Option(
        None, "--script",
        help="Inline quod-script source instead of a JSON spec. Use '-' to "
             "read script from stdin. See `quod schema --category script`.",
    ),
    script_file: str = typer.Option(
        None, "--script-file",
        help="Path to a quod-script file instead of inline --script.",
    ),
) -> None:
    """Append a new function. Spec is a JSON Function object, OR a
    quod-script source via --script / --script-file.

    JSON example: {"name": "g", "params": [...], "body": [...]}

    Script example: --script "fn g(x: i32) -> i32 { return x + 1 }"
    """
    if sum(s is not None for s in (script, script_file)) > 1:
        typer.echo("error: --script and --script-file are mutually exclusive", err=True)
        raise typer.Exit(1)

    with _exclusive_lock():
        program = _load()
        try:
            if script is not None or script_file is not None:
                from quod.script import parse_function as _parse_script
                from quod.resolve import resolve_imports as _resolve_imports
                if script_file is not None:
                    text = (sys.stdin.read() if script_file == "-"
                            else Path(script_file).read_text())
                else:
                    text = sys.stdin.read() if script == "-" else script
                # Resolve imports transiently so the script parser knows
                # which dotted type names are enums vs structs. The
                # resolved program is discarded — we only save the
                # user's view (`program`), not the inlined stdlib.
                enum_names = frozenset(
                    ed.name for ed in _resolve_imports(program).enums
                )
                fn = _parse_script(text, enum_names=enum_names)
            else:
                fn = parse_function_spec(read_json_arg(spec))
            program = add_function_to_program(program, fn)
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"added function {fn.name} (hash={short_hash(fn)})")


@fn_app.command("rename")
def fn_rename_cmd(
    old: str = typer.Argument(..., help="Existing function name (or hash prefix).",
                              autocompletion=_comp.function_or_hash),
    new: str = typer.Argument(..., help="New function name."),
) -> None:
    """Rename a function and update every call site that names it.

    Renames the function definition AND rewrites Call expressions
    across every other function so dangling-call errors don't fire
    at build time. If `old` is a hash prefix, it's resolved to the
    function name before the rewrite.
    """
    from quod.editor import rename_function
    with _exclusive_lock():
        program = _load()
        try:
            fn = find_function_ref(program, old)
            program = rename_function(program, fn.name, new)
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"renamed {fn.name} -> {new}")


@fn_app.command("rm")
def fn_rm(
    function: str = typer.Argument(..., help="Function name or hash prefix.",
                                    autocompletion=_comp.function_or_hash),
) -> None:
    """Remove a function from the program.

    Permissive: doesn't refuse if other functions still call this one. Run
    `quod fn callers FN` first if you want to know who'd be affected; the
    dangling call surfaces as an error at `quod build`.
    """
    with _exclusive_lock():
        program = _load()
        try:
            fn = find_function_ref(program, function)
            program = remove_function(program, fn.name)
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"removed function {fn.name}")


@fn_app.command("callers")
def fn_callers(
    target: str = typer.Argument(..., help="Function whose callers we want.",
                                 autocompletion=_comp.function_names),
) -> None:
    """List every call site to `target` across the program."""
    from quod.analysis import _walk_calls_in_stmt
    program = _load()
    found = False
    for caller in program.functions:
        for i, stmt in enumerate(caller.body.stmts):
            seen: set[str] = set()
            for call in _walk_calls_in_stmt(stmt):
                if call.function != target:
                    continue
                h = node_hash(call)
                if h in seen:
                    continue
                seen.add(h)
                found = True
                typer.echo(
                    f"{caller.name}.body[{i}] [{short_hash(stmt)}] → "
                    f"{target}/{len(call.args)} [{h[:HASH_DISPLAY_LEN]}]"
                )
    if not found:
        defined = {fn.name for fn in program.functions}
        extern = {ext.name for ext in program.externs}
        if target not in defined and target not in extern:
            typer.echo(f"warning: {target!r} is not declared in this program", err=True)
        typer.echo(f"(no callers of {target!r})")


@fn_app.command("data-flow")
def fn_data_flow(
    function: str = typer.Argument(..., help="Function name or hash prefix.",
                                    autocompletion=_comp.function_or_hash),
    param: str = typer.Argument(..., help="Parameter name.",
                                autocompletion=_comp.param_names_for_function),
) -> None:
    """Show every statement in `function` that reads `param`."""
    program = _load()
    try:
        fn = find_function_ref(program, function)
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    if fn.param(param) is None:
        typer.echo(f"error: {fn.name!r} has no parameter {param!r}", err=True)
        raise typer.Exit(1)
    any_read = False
    for i, stmt in enumerate(fn.body.stmts):
        n = _count_paramrefs(stmt, param)
        if n:
            any_read = True
            typer.echo(f"  body[{i}] [{short_hash(stmt)}]: {n} read(s)")
    if not any_read:
        typer.echo(f"({param!r} is unused in {fn.name})")


def _count_paramrefs(node, name: str) -> int:
    from quod.model import ParamRef, _Node
    total = 0
    if isinstance(node, ParamRef) and node.name == name:
        total += 1
    for _, value in node:
        if isinstance(value, _Node):
            total += _count_paramrefs(value, name)
        elif isinstance(value, tuple):
            for v in value:
                if isinstance(v, _Node):
                    total += _count_paramrefs(v, name)
    return total


@fn_app.command("call-graph")
def fn_call_graph(
    json_output: bool = typer.Option(False, "--json", help=_JSON_HELP),
) -> None:
    """Print the static call graph."""
    program = _load()
    if json_output:
        defined = {fn.name for fn in program.functions}
        extern_names = {ext.name for ext in program.externs}
        edges = {fn.name: list(function_callees(fn)) for fn in program.functions}
        called: set[str] = set()
        for callees in edges.values():
            called.update(callees)
        roots = [name for name in edges if name not in called]
        leaves = [name for name, cs in edges.items() if not cs]
        dangling = sorted({c for cs in edges.values() for c in cs if c not in defined and c not in extern_names})
        externs = sorted({c for cs in edges.values() for c in cs if c in extern_names})
        _emit_json({
            "edges": edges,
            "roots": roots,
            "leaves": leaves,
            "dangling": dangling,
            "externs": externs,
        })
        return
    if not program.functions:
        typer.echo("(no functions)")
        return

    defined = {fn.name for fn in program.functions}
    extern_names = {ext.name for ext in program.externs}
    edges: dict[str, tuple[str, ...]] = {fn.name: function_callees(fn) for fn in program.functions}

    called: set[str] = set()
    for callees in edges.values():
        called.update(callees)
    roots = [name for name in edges if name not in called]
    leaves = [name for name, cs in edges.items() if not cs]

    def _decorate(c: str) -> str:
        if c in defined:
            return c
        if c in extern_names:
            return f"{c}@extern"
        return f"{c}!"

    for fn in program.functions:
        callees = edges[fn.name]
        if not callees:
            typer.echo(f"{fn.name} -> (leaf)")
            continue
        rendered = ", ".join(_decorate(c) for c in callees)
        typer.echo(f"{fn.name} -> {rendered}")

    if roots or leaves:
        typer.echo("")
        typer.echo(f"roots:  {', '.join(roots) if roots else '(none)'}")
        typer.echo(f"leaves: {', '.join(leaves) if leaves else '(none)'}")
    has_dangling = any(c not in defined and c not in extern_names for cs in edges.values() for c in cs)
    has_extern = any(c in extern_names for cs in edges.values() for c in cs)
    if has_dangling:
        typer.echo("(! marks a callee not defined in this Program)")
    if has_extern:
        typer.echo("(@extern marks a callee declared as an extern, e.g. libc)")


@fn_app.command("unconstrained")
def fn_unconstrained(
    json_output: bool = typer.Option(False, "--json", help=_JSON_HELP),
) -> None:
    """List parameters that have no claim attached. A scout for the agent."""
    program = _load()
    if json_output:
        rows: list[dict[str, str]] = []
        for fn in program.functions:
            constrained = {claim_param(c) for c in fn.claims if claim_param(c) is not None}
            for p in fn.params:
                if p.name not in constrained:
                    rows.append({"function": fn.name, "param": p.name})
        _emit_json(rows)
        return
    found = False
    for fn in program.functions:
        constrained = {claim_param(c) for c in fn.claims if claim_param(c) is not None}
        for p in fn.params:
            if p.name not in constrained:
                found = True
                typer.echo(f"{fn.name}.{p.name}")
    if not found:
        typer.echo("(none)")
