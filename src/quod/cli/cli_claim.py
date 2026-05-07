"""claim sub-app — operations on function claims.

Hosts the shared `_build_claim` / `_predicate_for_sugar` /
`_parse_justification_spec` / `_build_predicate_claim` helpers used by
both the function-claim sub-app and the extern-claim sub-app
(`cli_extern.py`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer
from pydantic import TypeAdapter, ValidationError

from quod import completion as _comp
from quod import lower as lower_mod
from quod.analysis import derive_lattice_claims
from quod.canonicalize import (
    PARAM_SUGAR_KINDS,
    RETURN_SUGAR_KINDS,
    SUGAR_KINDS,
    predicate_for_param_range,
    predicate_for_return_range,
)
from quod.cli.cli_app import claim_app
from quod.cli.cli_output import (
    ENFORCEMENTS,
    STORED_REGIMES,
    _emit_json,
    _JSON_HELP,
    _sha256_of_file,
    _theme,
)
from quod.cli.cli_state import _cfg, _exclusive_lock, _load, _save, _selected_program
from quod.editor import find_function_ref
from quod.model import (
    Claim,
    DerivedJustification,
    ExternFunction,
    Function,
    I1Type,
    I8Type,
    I16Type,
    I32Type,
    I64Type,
    IsizeType,
    Justification,
    ManualJustification,
    PredicateClaim,
    Program,
    U8Type,
    U16Type,
    U32Type,
    U64Type,
    UsizeType,
    Z3Justification,
    add_claim,
    claim_param,
)
from quod.proof import Z3NotInstalled, goal_smt_lib, run_z3_on_file
from quod.providers import (
    ClaimRequest,
    default_for,
    get_provider,
)
from quod.render import (
    Span,
    claim_full_spans,
    paint,
)


@claim_app.command("ls")
def claim_ls(
    function: str | None = typer.Argument(None, help="Restrict to one function (omit for all).",
                                          autocompletion=_comp.function_or_hash),
    json_output: bool = typer.Option(False, "--json", help=_JSON_HELP),
) -> None:
    """List stored claims (axiom + witness regimes) across the program."""
    program = _load()
    try:
        fns = [find_function_ref(program, function)] if function else list(program.functions)
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    if json_output:
        _emit_json([
            {"function": fn.name, "claim": c}
            for fn in fns for c in fn.claims
        ])
        return
    theme = _theme()
    found = False
    for fn in fns:
        for c in fn.claims:
            found = True
            typer.echo(paint((
                Span(fn.name, "fn_name"), Span(": ", "punct"),
                *claim_full_spans(c),
            ), theme))
    if not found:
        typer.echo("(no claims)")


_JustificationAdapter: TypeAdapter[Justification] = TypeAdapter(Justification)


def _parse_justification_spec(raw: str) -> Justification:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise typer.BadParameter(f"--justification is not valid JSON: {e}")
    if isinstance(data, dict) and data.get("kind") == "z3":
        if "artifact_path" in data and not data.get("artifact_hash"):
            p = Path(data["artifact_path"])
            if not p.exists():
                raise typer.BadParameter(
                    f"--justification artifact not found: {p} "
                    f"(create the proof file before attaching)"
                )
            data["artifact_hash"] = _sha256_of_file(p)
    try:
        return _JustificationAdapter.validate_python(data)
    except ValidationError as e:
        raise typer.BadParameter(f"invalid --justification:\n{e}")


def _build_claim(
    target_fn: Function | ExternFunction,
    kind: str, target: str | None, *,
    lo: int | None, hi: int | None,
    regime: str, enforcement: str, justification: Justification | None,
) -> PredicateClaim:
    """Desugar a sugar-form claim invocation against `target_fn` into a
    `PredicateClaim` carrying the canonical predicate.

    `target_fn` is the function or extern the claim attaches to — needed
    to look up the param's type (for `non_negative`/`int_range`) or the
    return type (for `return_in_range`) so the IntLit bounds in the
    predicate are well-typed.
    """
    if regime not in STORED_REGIMES:
        raise typer.BadParameter(
            f"can't add claim with regime={regime!r}: stored claims must be one of "
            f"{', '.join(STORED_REGIMES)}. Lattice claims are derived; see `quod claim derive`."
        )
    if enforcement not in ENFORCEMENTS:
        raise typer.BadParameter(f"unknown enforcement {enforcement!r}; choices: {', '.join(ENFORCEMENTS)}")
    if kind in PARAM_SUGAR_KINDS and target is None:
        raise typer.BadParameter(f"{kind!r} requires --target / -t (the parameter name)")
    if kind in RETURN_SUGAR_KINDS and target is not None:
        raise typer.BadParameter(f"{kind!r} is function-scoped; --target / -t must not be set")
    expr = _predicate_for_sugar(target_fn, kind, target, lo, hi)
    return PredicateClaim(
        regime=regime, enforcement=enforcement, justification=justification,
        expr=expr,
    )


def _predicate_for_sugar(
    target_fn: Function | ExternFunction,
    kind: str, target: str | None,
    lo: int | None, hi: int | None,
):
    """Build the canonical predicate for one of the named sugar shapes."""
    if kind == "non_negative":
        if lo is not None or hi is not None:
            raise typer.BadParameter("non_negative does not take --min / --max")
        param_ty = _param_type(target_fn, target)
        return predicate_for_param_range(target, param_ty, lo=0, hi=None)
    if kind == "int_range":
        if lo is None and hi is None:
            raise typer.BadParameter("int_range requires --min and/or --max")
        param_ty = _param_type(target_fn, target)
        return predicate_for_param_range(target, param_ty, lo, hi)
    if kind == "return_in_range":
        if lo is None and hi is None:
            raise typer.BadParameter("return_in_range requires --min and/or --max")
        return predicate_for_return_range(target_fn.return_type, lo, hi)
    raise typer.BadParameter(f"unknown claim kind {kind!r}; choices: {', '.join(SUGAR_KINDS)}")


def _param_type(target_fn: Function | ExternFunction, name: str):
    """Resolve a param's type. Externs don't carry param names, so
    param-scoped claims aren't supported on them — this helper raises a
    user-facing error in that case."""
    if isinstance(target_fn, ExternFunction):
        raise typer.BadParameter(
            "extern claims are return-scoped only — externs don't carry "
            "named parameters in the model"
        )
    p = target_fn.param(name)
    if p is None:
        raise KeyError(
            f"function {target_fn.name!r} has no parameter {name!r}"
        )
    return p.type


def _parse_predicate_arg(target_fn: Function | ExternFunction, src: str):
    """Parse a `--predicate` source string against `target_fn`'s param
    scope, validate it's a side-effect-free predicate, and canonicalize.
    Returns the canonical `Expr`.

    Resolves bare integer literals from the function's signature (param
    types + return type), so `x >= 0` against an i32 param produces an
    i32 IntLit. Externs don't supply named params, so the resolver only
    has return-type context for them — predicates against externs that
    need param shape are filtered out earlier in the claim-add path.
    """
    from quod.canonicalize import canonicalize
    from quod.script import ScriptError, parse_predicate
    from quod.validate import PredicateError, assert_is_predicate

    if isinstance(target_fn, Function):
        param_types = {p.name: p.type for p in target_fn.params}
    else:
        param_types = {}
    try:
        expr = parse_predicate(
            src, param_types=param_types, return_type=target_fn.return_type,
        )
    except ScriptError as e:
        raise typer.BadParameter(f"--predicate: {e}")
    try:
        assert_is_predicate(expr)
    except PredicateError as e:
        raise typer.BadParameter(f"--predicate: {e}")
    return canonicalize(expr)


def _build_predicate_claim(
    target_fn: Function | ExternFunction, src: str, *,
    regime: str, enforcement: str, justification,
) -> PredicateClaim:
    """Parse a quod-script predicate body and wrap it in a `PredicateClaim`."""
    if regime not in STORED_REGIMES:
        raise typer.BadParameter(
            f"can't add claim with regime={regime!r}: stored claims must be one of "
            f"{', '.join(STORED_REGIMES)}. Lattice claims are derived; see `quod claim derive`."
        )
    if enforcement not in ENFORCEMENTS:
        raise typer.BadParameter(f"unknown enforcement {enforcement!r}; choices: {', '.join(ENFORCEMENTS)}")
    expr = _parse_predicate_arg(target_fn, src)
    return PredicateClaim(
        regime=regime, enforcement=enforcement, justification=justification,
        expr=expr,
    )


@claim_app.command("add")
def claim_add(
    function: str = typer.Argument(..., help="Function name or hash prefix.",
                                    autocompletion=_comp.function_or_hash),
    kind: str | None = typer.Argument(
        None,
        help=f"Sugar claim kind. One of: {', '.join(SUGAR_KINDS)}. "
             f"Omit when using --predicate.",
        autocompletion=_comp.claim_kinds,
    ),
    target: str | None = typer.Argument(
        None,
        help=f"Parameter name. Required for: {', '.join(PARAM_SUGAR_KINDS)}. "
             f"Must be omitted for: {', '.join(RETURN_SUGAR_KINDS)}.",
        autocompletion=_comp.param_names_for_function,
    ),
    lo: int | None = typer.Option(None, "--min"),
    hi: int | None = typer.Option(None, "--max"),
    predicate: str | None = typer.Option(
        None, "--predicate",
        help='Quod-script predicate body, e.g. "x >= 0 && x <= y" or '
             '"return > 0". Mutually exclusive with the sugar arguments. '
             'Use bare param names for ParamRef and `return` for ReturnRef.',
    ),
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
        help='JSON Justification spec, e.g. \'{"kind":"z3","artifact_path":"proofs/x.smt2"}\'.',
    ),
) -> None:
    """Attach a claim to a function. The optimizer will trust this assertion."""
    if predicate is not None and (kind is not None or target is not None or lo is not None or hi is not None):
        typer.echo(
            "error: --predicate is mutually exclusive with the sugar arguments "
            "(kind / target / --min / --max)",
            err=True,
        )
        raise typer.Exit(2)
    if predicate is None and kind is None:
        typer.echo(
            f"error: provide a sugar kind ({', '.join(SUGAR_KINDS)}) or --predicate",
            err=True,
        )
        raise typer.Exit(2)
    with _exclusive_lock():
        program = _load()
        try:
            fn = find_function_ref(program, function)
            just_obj = _parse_justification_spec(justification) if justification else None
            if predicate is not None:
                claim = _build_predicate_claim(
                    fn, predicate,
                    regime=regime, enforcement=enforcement, justification=just_obj,
                )
                summary = "predicate"
            else:
                claim = _build_claim(
                    fn, kind, target, lo=lo, hi=hi,
                    regime=regime, enforcement=enforcement, justification=just_obj,
                )
                summary = f"{kind}({target})" if target is not None else kind
            program = add_claim(program, fn.name, claim)
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"added {summary} on {fn.name} [regime={regime}, enforcement={enforcement}]")


@claim_app.command("relax")
def claim_relax(
    function: str = typer.Argument(..., help="Function name or hash prefix.",
                                    autocompletion=_comp.function_or_hash),
    kind: str | None = typer.Argument(
        None,
        help=f"Sugar claim kind. One of: {', '.join(SUGAR_KINDS)}. "
             f"Omit when using --predicate.",
        autocompletion=_comp.claim_kinds,
    ),
    target: str | None = typer.Argument(None, help="Parameter name (omit for return-scoped claims).",
                                        autocompletion=_comp.param_names_for_function),
    lo: int | None = typer.Option(None, "--min"),
    hi: int | None = typer.Option(None, "--max"),
    predicate: str | None = typer.Option(
        None, "--predicate",
        help="Quod-script predicate body. Mutually exclusive with the sugar arguments.",
    ),
) -> None:
    """Remove a claim (always safe — drops an assertion).

    Symmetric to `claim add`: pass the same sugar arguments — or the
    same `--predicate` — that were used to add the claim. Multiple
    predicates over the same param are distinct; bounds are part of
    the match.
    """
    from quod.model import relax_claim
    if predicate is not None and (kind is not None or target is not None or lo is not None or hi is not None):
        typer.echo(
            "error: --predicate is mutually exclusive with the sugar arguments",
            err=True,
        )
        raise typer.Exit(2)
    if predicate is None and kind is None:
        typer.echo(
            f"error: provide a sugar kind ({', '.join(SUGAR_KINDS)}) or --predicate",
            err=True,
        )
        raise typer.Exit(2)
    with _exclusive_lock():
        program = _load()
        try:
            fn = find_function_ref(program, function)
            if predicate is not None:
                expr = _parse_predicate_arg(fn, predicate)
                summary = "predicate"
            else:
                expr = _predicate_for_sugar(fn, kind, target, lo, hi)
                scope = f"({target})" if target is not None else "(return)"
                summary = f"{kind}{scope}"
            program = relax_claim(program, fn.name, expr)
        except (KeyError, typer.BadParameter) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"relaxed {summary} on {fn.name}")


@claim_app.command("verify")
def claim_verify(
    root: Path | None = typer.Option(
        None, "--root",
        help="Root for resolving justification artifact_path. "
             "Defaults to the quod.toml directory.",
    ),
) -> None:
    """Re-check evidence attached to stored claims."""
    from quod.version import check_program_version

    cfg = _cfg()
    program = _load()

    version_ok, version_msg = check_program_version(program)
    if not version_ok:
        typer.echo(f"error: {version_msg}", err=True)
        raise typer.Exit(1)

    resolve_root = root if root is not None else cfg.root
    theme = _theme()
    failures = 0
    checked = 0
    for fn in program.functions:
        for c in fn.claims:
            if c.justification is None:
                continue
            checked += 1
            ok, msg = _verify_justification(c.justification, resolve_root, fn, c, program)
            status_span = Span("ok  ", "ok") if ok else Span("FAIL", "warn")
            typer.echo(paint((
                status_span, Span(" ", "ws"),
                Span(fn.name, "fn_name"), Span(": ", "punct"),
                *claim_full_spans(c),
            ), theme))
            if not ok:
                typer.echo(f"     {msg}")
                failures += 1
    if checked == 0:
        typer.echo("(no claims with justifications)")
    if failures:
        raise typer.Exit(1)


def _verify_justification(
    j: Justification, root: Path, fn: Function, claim: Claim, program: Program,
) -> tuple[bool, str]:
    match j:
        case Z3Justification(artifact_path=p, artifact_hash=stored, body_smt_hash=body_stored):
            # Body-drift check (runs first — short-circuits before file I/O).
            # Re-derive SMT from the current body+claim. The goal claim must
            # have justification=None (the prove-time shape; otherwise the
            # `; goal: ...` repr in the SMT comment differs). Hypotheses are
            # the other claims on this fn — exclude the claim being verified
            # (at prove time it wasn't yet in fn.claims).
            goal = claim.model_copy(update={"justification": None})
            hypotheses = tuple(c for c in fn.claims if c is not claim)
            try:
                current_smt = goal_smt_lib(fn, goal, hypotheses=hypotheses, program=program)
            except NotImplementedError as e:
                return False, f"could not re-derive SMT for body-hash check: {e}"
            current_hash = hashlib.sha256(current_smt.encode("utf-8")).hexdigest()
            if current_hash != body_stored:
                return False, (
                    f"body changed since proof — re-run `claim prove` "
                    f"(stored body_smt_hash={body_stored[:12]}, current={current_hash[:12]})"
                )

            full = root / p
            if not full.exists():
                return False, f"artifact not found: {full}"
            actual = _sha256_of_file(full)
            if actual != stored:
                return False, f"hash mismatch: stored={stored[:12]}, file={actual[:12]}"
            try:
                result = run_z3_on_file(full)
            except Z3NotInstalled as e:
                return False, str(e)
            except Exception as e:
                return False, f"z3 invocation failed: {e}"
            if result.status != "unsat":
                return False, f"z3 returned {result.status!r} (expected 'unsat')"
            return True, ""
        case ManualJustification(signed_by=s):
            if not s.strip():
                return False, "manual signed_by is empty"
            return True, ""
        case DerivedJustification():
            return True, ""
    return False, f"unknown justification kind: {j!r}"


@claim_app.command("suggest")
def claim_suggest(
    top_n: int = typer.Option(10, "--top-n", help="Show this many top suggestions."),
) -> None:
    """Speculatively compile candidate claims; surface those that shrink IR."""
    program = _load()
    try:
        baseline = _ir_line_count(program)
    except Exception as e:
        typer.echo(f"error: baseline compile failed: {e}", err=True)
        raise typer.Exit(1)

    candidates = _generate_candidates(program)
    typer.echo(f"baseline: {baseline} IR line(s); evaluating {len(candidates)} candidate claim(s)...")

    results: list[tuple[int, str, object]] = []
    for fn_name, candidate in candidates:
        try:
            modified = add_claim(program, fn_name, candidate)
        except (KeyError, ValueError):
            continue
        try:
            size = _ir_line_count(modified)
        except Exception:
            continue
        delta = baseline - size
        if delta > 0:
            results.append((delta, fn_name, candidate))

    results.sort(key=lambda t: -t[0])
    if not results:
        typer.echo("no candidates shrink IR — current codegen is already tight, "
                   "or candidates were trivially redundant.")
        return
    typer.echo("\ntop suggestions (lines saved):")
    theme = _theme()
    for delta, fn_name, claim in results[:top_n]:
        typer.echo(paint((
            Span(f"  -{delta:>3} lines  ", "literal_int"),
            Span("on ", "punct"),
            Span(fn_name, "fn_name"), Span(": ", "punct"),
            *claim_full_spans(claim),
        ), theme))
    typer.echo("\nNext: try `quod claim prove KIND -f FN [...]` for the candidates that "
               "should actually be true.")


def _ir_line_count(program: Program) -> int:
    from quod.analysis import elaborate
    derived = derive_lattice_claims(program)
    program = elaborate(program, derived)
    module = lower_mod.lower(program)
    target_machine = lower_mod.make_target_machine()
    parsed = lower_mod.parse_and_verify(module)
    lower_mod.optimize_module(parsed, target_machine, speed_level=2)
    return len(str(parsed).splitlines())


_INT_TYPE_CLASSES = (
    I1Type, I8Type, I16Type, I32Type, I64Type,
    U8Type, U16Type, U32Type, U64Type, IsizeType, UsizeType,
)


def _generate_candidates(program: Program) -> list[tuple[str, PredicateClaim]]:
    """Enumerate candidate sugar-shape claims for `claim suggest`. One
    `non_negative(p)` per param, plus `return_in_range([-1, +inf])` and
    `return_in_range([0, +inf])` for int-returning functions. Skips any
    candidate already present (by canonical predicate equality)."""
    out: list[tuple[str, PredicateClaim]] = []
    for fn in program.functions:
        existing_exprs = {c.expr for c in fn.claims}
        for p in fn.params:
            if not isinstance(p.type, _INT_TYPE_CLASSES):
                continue
            cand = predicate_for_param_range(p.name, p.type, lo=0, hi=None)
            if cand not in existing_exprs:
                out.append((fn.name, PredicateClaim(regime="axiom", expr=cand)))
        if isinstance(fn.return_type, _INT_TYPE_CLASSES):
            for lo in (-1, 0):
                cand = predicate_for_return_range(fn.return_type, lo=lo, hi=None)
                if cand not in existing_exprs:
                    out.append((fn.name, PredicateClaim(regime="axiom", expr=cand)))
    return out


@claim_app.command("derive")
def claim_derive(
    provider: str | None = typer.Option(
        None, "--provider", help="Provider name (defaults to the first lattice/derive provider).",
        autocompletion=_comp.provider_names_for("lattice"),
    ),
) -> None:
    """Run a lattice provider and print derived (regime=lattice) claims."""
    program = _load()
    try:
        prov = get_provider(provider) if provider else default_for(regime="lattice", mode="derive")
    except KeyError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    if prov.derive is None:
        typer.echo(f"error: provider {prov.name!r} does not support derive mode", err=True)
        raise typer.Exit(1)
    derived = prov.derive(program)
    if not derived:
        typer.echo(f"(no derived claims from {prov.name})")
        return
    theme = _theme()
    for fn in program.functions:
        for c in derived.get(fn.name, ()):
            typer.echo(paint((
                Span(fn.name, "fn_name"), Span(": ", "punct"),
                *claim_full_spans(c),
            ), theme))


@claim_app.command("prove")
def claim_prove(
    function: str = typer.Argument(..., help="Function name or hash prefix.",
                                    autocompletion=_comp.function_or_hash),
    kind: str | None = typer.Argument(
        None,
        help=f"Sugar claim kind. One of: {', '.join(SUGAR_KINDS)}. "
             f"Omit when using --predicate.",
        autocompletion=_comp.claim_kinds,
    ),
    target: str | None = typer.Argument(None, help="Parameter name (omit for return-scoped claims).",
                                        autocompletion=_comp.param_names_for_function),
    lo: int | None = typer.Option(None, "--min"),
    hi: int | None = typer.Option(None, "--max"),
    predicate: str | None = typer.Option(
        None, "--predicate",
        help="Quod-script predicate body. Mutually exclusive with the sugar arguments.",
    ),
    enforcement: str = typer.Option("trust", "--enforcement",
                                    autocompletion=_comp.enforcements),
    provider: str | None = typer.Option(
        None, "--provider", help="Provider name (defaults to the first witness/prove provider).",
        autocompletion=_comp.provider_names_for("witness"),
    ),
) -> None:
    """Synthesize a proof of a claim via a provider, attach as a witness."""
    cfg = _cfg()
    prog_spec = _selected_program()
    proofs_dir = cfg.resolve(cfg.proofs_dir) / prog_spec.name
    try:
        prov = get_provider(provider) if provider else default_for(regime="witness", mode="prove")
    except KeyError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    if prov.prove is None:
        typer.echo(f"error: provider {prov.name!r} does not support prove mode", err=True)
        raise typer.Exit(1)
    if predicate is not None and (kind is not None or target is not None or lo is not None or hi is not None):
        typer.echo(
            "error: --predicate is mutually exclusive with the sugar arguments",
            err=True,
        )
        raise typer.Exit(2)
    if predicate is None:
        if kind is None:
            typer.echo(
                f"error: provide a sugar kind ({', '.join(SUGAR_KINDS)}) or --predicate",
                err=True,
            )
            raise typer.Exit(2)
        if kind not in SUGAR_KINDS:
            typer.echo(f"error: unknown claim kind {kind!r}; one of: {', '.join(SUGAR_KINDS)}", err=True)
            raise typer.Exit(2)
    if enforcement not in ENFORCEMENTS:
        typer.echo(f"error: --enforcement must be one of {ENFORCEMENTS}", err=True)
        raise typer.Exit(2)

    # Hold the lock end-to-end: the proof's correctness depends on fn.body
    # not changing between load and save.
    with _exclusive_lock():
        program = _load()
        try:
            fn = find_function_ref(program, function)
            if predicate is not None:
                expr = _parse_predicate_arg(fn, predicate)
                summary = "predicate"
            else:
                expr = _predicate_for_sugar(fn, kind, target, lo, hi)
                summary = kind
        except (KeyError, ValueError, typer.BadParameter) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)

        request = ClaimRequest(function=fn.name, expr=expr, enforcement=enforcement)
        result = prov.prove(program, request, proofs_dir)
        if result.status != "proven":
            tag = result.status
            typer.echo(f"could not prove {summary}: {prov.name} reported {tag} ({result.detail})", err=True)
            if tag == "refuted":
                typer.echo("(provider found a counterexample; the claim does not hold)", err=True)
            raise typer.Exit(1)

        assert result.claim is not None
        try:
            program = add_claim(program, fn.name, result.claim)
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        from quod.version import stamp_quod_version
        program = stamp_quod_version(program)
        _save(program)

    theme = _theme()
    typer.echo(paint((
        Span("proved ", "ok"),
        *claim_full_spans(result.claim),
        Span(" via ", "punct"),
        Span(prov.name, "fn_name"),
    ), theme))
    if result.artifact_path is not None and result.artifact_hash is not None:
        typer.echo(f"  artifact: {result.artifact_path} (sha256={result.artifact_hash[:12]})")
