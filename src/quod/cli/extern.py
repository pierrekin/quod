"""extern sub-app and extern-claim sub-app — operations on externs."""

from __future__ import annotations

from pathlib import Path

import typer

from quod import completion as _comp
from quod.cli.app import extern_app, extern_claim_app
from quod.cli.claim import _build_claim, _parse_justification_spec, _predicate_for_sugar
from quod.cli.output import (
    ENFORCEMENTS,
    STORED_REGIMES,
    _emit_json,
    _JSON_HELP,
    _theme,
)
from quod.cli.state import _exclusive_lock, _load, _save
from quod.cli.types import _LINKAGE_NAMES, _TYPE_NAMES, _parse_linkage, _parse_type_name
from quod.predicate.canonical import RETURN_SUGAR_KINDS
from quod.editor import (
    remove_extern_from_program,
    set_extern_linkage_in_program,
)
from quod.ingest import IngestError, ingest_header
from quod.model import (
    ExternFunction,
    Linkage,
    add_extern_claim,
    relax_extern_claim,
)
from quod.render import (
    Span,
    claim_full_spans,
    extern_signature_spans,
    paint,
)


@extern_app.command("ls")
def extern_ls(
    json_output: bool = typer.Option(False, "--json", help=_JSON_HELP),
) -> None:
    """List declared externs with their signatures."""
    program = _load()
    if json_output:
        _emit_json(list(program.externs))
        return
    if not program.externs:
        typer.echo("(no externs)")
        return
    theme = _theme()
    for ext in program.externs:
        typer.echo(paint(extern_signature_spans(ext), theme))


@extern_app.command("add")
def extern_add(
    name: str = typer.Argument(..., help="Extern function name (must match the libc/library symbol)."),
    arity: int = typer.Option(0, "--arity", min=0, help="Number of i32 parameters (shorthand)."),
    param_type: list[str] = typer.Option(
        [], "--param-type",
        help=f"Typed parameter (repeatable). One of: {', '.join(_TYPE_NAMES)}.",
    ),
    return_type: str = typer.Option("i32", "--return-type"),
    varargs: bool = typer.Option(False, "--varargs"),
    linkage: str = typer.Option(
        "libc", "--linkage",
        help="Symbol provenance: 'libc' (clang links it) or 'runtime' (quod's libquodrt).",
        autocompletion=_comp.linkage_names,
    ),
) -> None:
    """Declare an extern function and record its provenance.

    `--linkage libc` (default) for symbols clang already links from libc;
    `--linkage runtime` for symbols defined in quod's runtime archive
    (libquodrt — every src/quod/runtime/*.c is compiled in)."""
    with _exclusive_lock():
        program = _load()
        if any(ext.name == name for ext in program.externs):
            typer.echo(f"error: extern {name!r} already declared", err=True)
            raise typer.Exit(1)
        if any(fn.name == name for fn in program.functions):
            typer.echo(f"error: {name!r} already exists as a user function", err=True)
            raise typer.Exit(1)
        if param_type and arity:
            raise typer.BadParameter("pass either --arity or --param-type, not both")
        struct_names = tuple(sd.name for sd in program.structs)
        param_types = tuple(_parse_type_name(t, struct_names=struct_names) for t in param_type)
        ret_ty = _parse_type_name(return_type, struct_names=struct_names)
        link_obj: Linkage = _parse_linkage(linkage)
        ext = ExternFunction(
            name=name,
            arity=arity if not param_types else 0,
            param_types=param_types,
            return_type=ret_ty,
            varargs=varargs,
            linkage=link_obj,
        )
        new_externs = program.externs + (ext,)
        program = program.model_copy(update={"externs": new_externs})
        _save(program)
    sig_parts = list(param_type or ["i32"] * arity)
    if varargs:
        sig_parts.append("...")
    typer.echo(f"declared extern {name}({', '.join(sig_parts)}) -> {return_type}")


@extern_app.command("rm")
def extern_rm(
    name: str = typer.Argument(..., help="Extern name to remove.",
                               autocompletion=_comp.extern_names),
) -> None:
    """Remove an extern declaration.

    Permissive: doesn't refuse if a llvm.call still targets it. The dangling
    call surfaces at `quod build` time as 'call to undeclared function'.
    """
    with _exclusive_lock():
        program = _load()
        try:
            program = remove_extern_from_program(program, name)
        except KeyError as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"removed extern {name}")


@extern_app.command("set-linkage")
def extern_set_linkage(
    name: str = typer.Argument(..., help="Extern name.",
                               autocompletion=_comp.extern_names),
    linkage: str = typer.Argument(..., help="New linkage: 'libc' or 'runtime'.",
                                  autocompletion=_comp.linkage_names),
) -> None:
    """Change an extern's linkage in place. Useful when a symbol moved
    between libc and the quod runtime, or to fix a mistaken `extern add`."""
    link_obj = _parse_linkage(linkage)
    with _exclusive_lock():
        program = _load()
        try:
            program = set_extern_linkage_in_program(program, name, link_obj)
        except KeyError as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"set linkage of extern {name} to {linkage}")


@extern_claim_app.command("ls")
def extern_claim_ls(
    extern: str | None = typer.Argument(None, help="Restrict to one extern (omit for all).",
                                        autocompletion=_comp.extern_names),
    json_output: bool = typer.Option(False, "--json", help=_JSON_HELP),
) -> None:
    """List claims attached to externs (axiom + witness regimes)."""
    program = _load()
    if extern is not None and not any(e.name == extern for e in program.externs):
        typer.echo(f"error: no extern named {extern!r}", err=True)
        raise typer.Exit(1)
    exts = [e for e in program.externs if extern is None or e.name == extern]
    if json_output:
        _emit_json([
            {"extern": e.name, "claim": c}
            for e in exts for c in e.claims
        ])
        return
    theme = _theme()
    found = False
    for e in exts:
        for c in e.claims:
            found = True
            typer.echo(paint((
                Span(e.name, "fn_name"), Span(": ", "punct"),
                *claim_full_spans(c),
            ), theme))
    if not found:
        typer.echo("(no claims)")


@extern_claim_app.command("add")
def extern_claim_add(
    extern: str = typer.Argument(..., help="Extern name.",
                                 autocompletion=_comp.extern_names),
    kind: str = typer.Argument(..., help=f"Claim kind. Externs currently support: {', '.join(RETURN_SUGAR_KINDS)}.",
                               autocompletion=_comp.claim_kinds),
    lo: int | None = typer.Option(None, "--min"),
    hi: int | None = typer.Option(None, "--max"),
    regime: str = typer.Option(
        "axiom", "--regime",
        help=f"Epistemic source. One of: {', '.join(STORED_REGIMES)}.",
        autocompletion=_comp.stored_regimes,
    ),
    enforcement: str = typer.Option(
        "trust", "--enforcement",
        help=f"trust = llvm.assume (UB if false); verify = runtime branch + abort. "
             f"One of: {', '.join(ENFORCEMENTS)}.",
        autocompletion=_comp.enforcements,
    ),
    justification: str | None = typer.Option(
        None, "--justification",
        help='JSON Justification spec, e.g. \'{"kind":"manual","signed_by":"alice","rationale":"libc(2)"}\'.',
    ),
) -> None:
    """Attach a claim to an extern. Lowered as `llvm.assume` after every
    call site so the optimizer can exploit the bound at each caller.

    Externs currently only support return-scoped claims (e.g. return_in_range).
    Param-scoped claims need named extern params, which is a follow-up.
    """
    if kind not in RETURN_SUGAR_KINDS:
        typer.echo(
            f"error: extern claims currently only support: {', '.join(RETURN_SUGAR_KINDS)}. "
            f"Param-scoped kinds need named extern params (not yet on the model).",
            err=True,
        )
        raise typer.Exit(1)
    with _exclusive_lock():
        program = _load()
        ext = next((e for e in program.externs if e.name == extern), None)
        if ext is None:
            typer.echo(f"error: no extern named {extern!r}", err=True)
            raise typer.Exit(1)
        try:
            just_obj = _parse_justification_spec(justification) if justification else None
            claim = _build_claim(
                ext, kind, target=None, lo=lo, hi=hi,
                regime=regime, enforcement=enforcement, justification=just_obj,
            )
            program = add_extern_claim(program, extern, claim)
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"added {kind} on extern {extern} [regime={regime}, enforcement={enforcement}]")


@extern_claim_app.command("relax")
def extern_claim_relax(
    extern: str = typer.Argument(..., help="Extern name.",
                                 autocompletion=_comp.extern_names),
    kind: str = typer.Argument(..., help=f"Claim kind to remove. One of: {', '.join(RETURN_SUGAR_KINDS)}.",
                               autocompletion=_comp.claim_kinds),
    lo: int | None = typer.Option(None, "--min"),
    hi: int | None = typer.Option(None, "--max"),
) -> None:
    """Remove a claim from an extern. Symmetric to `extern claim add`:
    pass the same sugar arguments that were used to add the claim."""
    with _exclusive_lock():
        program = _load()
        ext = next((e for e in program.externs if e.name == extern), None)
        if ext is None:
            typer.echo(f"error: no extern named {extern!r}", err=True)
            raise typer.Exit(1)
        try:
            expr = _predicate_for_sugar(ext, kind, None, lo, hi)
            program = relax_extern_claim(program, extern, expr)
        except (KeyError, typer.BadParameter) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"relaxed {kind} on extern {extern}")


@extern_app.command("ingest")
def extern_ingest(
    header: Path = typer.Argument(..., help="C header file (e.g. /usr/include/stdio.h)."),
) -> None:
    """Append externs from every supported FUNCTION_DECL in HEADER.

    Skips declarations whose signatures use unsupported types (structs,
    floats, wider ints, function pointers) and skips names that already
    have an extern in the current program. Prints a summary tally.
    """
    if not header.exists():
        typer.echo(f"error: {header} does not exist", err=True)
        raise typer.Exit(1)

    try:
        new_externs, skipped_unsupported = ingest_header(header)
    except IngestError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)

    with _exclusive_lock():
        program = _load()
        existing = {ext.name for ext in program.externs}
        existing |= {fn.name for fn in program.functions}
        to_add = tuple(ext for ext in new_externs if ext.name not in existing)
        skipped_duplicate = tuple(ext.name for ext in new_externs if ext.name in existing)
        if to_add:
            program = program.model_copy(update={"externs": program.externs + to_add})
            _save(program)

    typer.echo(f"added {len(to_add)} extern(s) from {header}")
    if skipped_unsupported:
        typer.echo(f"  skipped {len(skipped_unsupported)} (unsupported signatures)")
    if skipped_duplicate:
        typer.echo(f"  skipped {len(skipped_duplicate)} (already declared)")
