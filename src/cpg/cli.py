"""Typer CLI. Each command maps 1:1 to a future agent tool call.

State lives in ./program.json by default; pass --program PATH to override.
Inspection commands print to stdout; mutations write the file back atomically.

Function and statement references accept either a name (functions only) or a
content-hash prefix (any node). The CLI shows short hashes inline in `show`
output so they can be copy-pasted as refs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import typer
from pydantic import TypeAdapter, ValidationError

from cpg import lower as lower_mod
from cpg.editor import (
    add_function_to_program,
    add_statement_in_function,
    find_function_ref,
    parse_function_spec,
    parse_statement_spec,
    read_json_arg,
)
from cpg.hashing import HASH_DISPLAY_LEN, find_by_prefix, node_hash, short_hash, walk
from cpg.analysis import derive_lattice_claims
from cpg.model import (
    CLAIM_KINDS,
    PARAM_CLAIM_KINDS,
    RETURN_CLAIM_KINDS,
    DerivedJustification,
    IntRangeClaim,
    Justification,
    ManualJustification,
    NonNegativeClaim,
    Program,
    ReturnInRangeClaim,
    Z3Justification,
    add_claim,
    claim_param,
    format_claim,
    format_function,
    format_program,
    function_callees,
    load_program,
    relax_claim,
    save_program,
)
from cpg.proof import Z3NotInstalled, goal_smt_lib, run_z3_on_file, run_z3_on_smt


REGIMES = ("axiom", "witness", "lattice")
STORED_REGIMES = ("axiom", "witness")  # lattice is derived, never stored
ENFORCEMENTS = ("trust", "verify")
from cpg.templates import TEMPLATES


app = typer.Typer(
    no_args_is_help=True,
    help="cpg: edit a code-property graph and compile it through LLVM.",
    pretty_exceptions_show_locals=False,
)


# ---------- Shared state ----------

_state: dict[str, Path] = {"program_path": Path("program.json")}


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    program: Path = typer.Option(
        Path("program.json"),
        "--program", "-p",
        help="Path to the program JSON file.",
    ),
) -> None:
    _state["program_path"] = program
    # `no_args_is_help=True` only fires with literally zero args, so
    # `cpg -p foo.json` (options but no subcommand) still falls through.
    # Catch that here and print full help.
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def _path() -> Path:
    return _state["program_path"]


def _load() -> Program:
    p = _path()
    if not p.exists():
        typer.echo(f"error: {p} does not exist (run `cpg init` first)", err=True)
        raise typer.Exit(1)
    return load_program(p)


def _save(program: Program) -> None:
    save_program(program, _path())


def _hash_label(node) -> str:
    return f"[{short_hash(node)}] "


# ---------- Lifecycle ----------

@app.command()
def init(
    template: str = typer.Option(
        "hello", "--template", "-t",
        help=f"Starter template. One of: {', '.join(TEMPLATES)}.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing program file."),
) -> None:
    """Write a starter program file."""
    if template not in TEMPLATES:
        typer.echo(f"error: unknown template {template!r}; choices: {', '.join(TEMPLATES)}", err=True)
        raise typer.Exit(2)
    if _path().exists() and not force:
        typer.echo(f"error: {_path()} already exists (use --force to overwrite)", err=True)
        raise typer.Exit(1)
    _save(TEMPLATES[template])
    typer.echo(f"wrote {_path()} ({template} starter)")


@app.command()
def validate() -> None:
    """Parse, lower, and LLVM-verify the program. No artifacts emitted."""
    program = _load()
    try:
        module = lower_mod.lower(program)
        parsed = lower_mod.parse_and_verify(module)
    except (ValueError, KeyError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    del parsed
    typer.echo("ok")


# ---------- Inspection ----------

@app.command()
def show() -> None:
    """Print the program in canonical form, with content-hash prefixes."""
    typer.echo(format_program(_load(), label=_hash_label))


@app.command("show-function")
def show_function_cmd(ref: str) -> None:
    """Print a single function. Accepts a name or a content-hash prefix."""
    try:
        fn = find_function_ref(_load(), ref)
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(format_function(fn, label=_hash_label))


@app.command("list-functions")
def list_functions() -> None:
    """List all functions with their signatures and hashes."""
    program = _load()
    if not program.functions:
        typer.echo("(no functions)")
        return
    for fn in program.functions:
        sig = ", ".join(f"{p}: i32" for p in fn.params)
        suffix = f"  [{len(fn.claims)} claim(s)]" if fn.claims else ""
        typer.echo(f"[{short_hash(fn)}] {fn.name}({sig}) -> i32{suffix}")


@app.command("list-claims")
def list_claims(
    function: str | None = typer.Option(None, "--function", "-f", help="Restrict to one function (name or hash)."),
) -> None:
    """List stored claims across the program (or one function).

    Stored = axiom + witness regimes. Lattice claims are derived; see
    `cpg derive-claims` for those.
    """
    program = _load()
    try:
        fns = [find_function_ref(program, function)] if function else list(program.functions)
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    found = False
    for fn in fns:
        for c in fn.claims:
            found = True
            typer.echo(f"{fn.name}: {format_claim(c)}")
    if not found:
        typer.echo("(no claims)")


@app.command("verify-claims")
def verify_claims_cmd(
    root: Path = typer.Option(
        Path("."), "--root",
        help="Project root for resolving justification artifact_path.",
    ),
) -> None:
    """Re-check evidence attached to stored claims.

    z3:       re-sha256 artifact_path, compare to artifact_hash
    manual:   check signed_by is non-empty
    derived:  skip (re-derived from the graph each compile)
    None:     skip (no evidence claimed)

    Exit nonzero if any check fails.
    """
    program = _load()
    failures = 0
    checked = 0
    for fn in program.functions:
        for c in fn.claims:
            if c.justification is None:
                continue
            checked += 1
            ok, msg = _verify_justification(c.justification, root)
            status = "ok  " if ok else "FAIL"
            typer.echo(f"{status} {fn.name}: {format_claim(c)}")
            if not ok:
                typer.echo(f"     {msg}")
                failures += 1
    if checked == 0:
        typer.echo("(no claims with justifications)")
    if failures:
        raise typer.Exit(1)


def _verify_justification(j: Justification, root: Path) -> tuple[bool, str]:
    match j:
        case Z3Justification(artifact_path=p, artifact_hash=stored):
            full = root / p
            if not full.exists():
                return False, f"artifact not found: {full}"
            actual = _sha256_of_file(full)
            if actual != stored:
                return False, f"hash mismatch: stored={stored[:12]}, file={actual[:12]}"
            # MVP3: actually run Z3 on the artifact and confirm `unsat`.
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
            return True, ""  # re-derivable from program; skip
    return False, f"unknown justification kind: {j!r}"


@app.command("derive-claims")
def derive_claims_cmd() -> None:
    """Run the lattice analysis and print derived (regime=lattice) claims.

    Read-only: doesn't mutate program.json. Each compile re-derives from
    scratch, so the output reflects the current graph.
    """
    program = _load()
    derived = derive_lattice_claims(program)
    if not derived:
        typer.echo("(no derived claims)")
        return
    for fn in program.functions:
        for c in derived.get(fn.name, ()):
            typer.echo(f"{fn.name}: {format_claim(c)}")


@app.command("show-call-graph")
def show_call_graph() -> None:
    """Print the static call graph: caller -> callees, plus orphan callers and roots.

    Edges are deduped per caller. A `!` suffix flags a callee that isn't
    defined in this Program (a dangling reference; lower-time error).
    """
    program = _load()
    if not program.functions:
        typer.echo("(no functions)")
        return

    defined = {fn.name for fn in program.functions}
    edges: dict[str, tuple[str, ...]] = {fn.name: function_callees(fn) for fn in program.functions}

    called: set[str] = set()
    for callees in edges.values():
        called.update(callees)
    roots = [name for name in edges if name not in called]
    leaves = [name for name, cs in edges.items() if not cs]

    for fn in program.functions:
        callees = edges[fn.name]
        if not callees:
            typer.echo(f"{fn.name} -> (leaf)")
            continue
        rendered = ", ".join(c if c in defined else f"{c}!" for c in callees)
        typer.echo(f"{fn.name} -> {rendered}")

    if roots or leaves:
        typer.echo("")
        typer.echo(f"roots:  {', '.join(roots) if roots else '(none)'}")
        typer.echo(f"leaves: {', '.join(leaves) if leaves else '(none)'}")
    if any(c not in defined for cs in edges.values() for c in cs):
        typer.echo("(! marks a callee not defined in this Program)")


@app.command("find-unconstrained-params")
def find_unconstrained_params() -> None:
    """List parameters that have no claim attached. A scout for the agent."""
    program = _load()
    found = False
    for fn in program.functions:
        constrained = {claim_param(c) for c in fn.claims if claim_param(c) is not None}
        for p in fn.params:
            if p not in constrained:
                found = True
                typer.echo(f"{fn.name}.{p}")
    if not found:
        typer.echo("(none)")


@app.command()
def find(prefix: str) -> None:
    """Resolve a hash prefix to a node and print it.

    Useful for an agent (or human) that has a hash from `show` and wants to
    confirm what it points to before editing.
    """
    program = _load()
    try:
        node = find_by_prefix(program, prefix)
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"hash:  {node_hash(node)}")
    typer.echo(f"short: {short_hash(node)}")
    typer.echo(f"type:  {type(node).__name__}")
    typer.echo(f"json:  {node.model_dump_json()}")


@app.command("list-hashes")
def list_hashes() -> None:
    """Dump every node and its short hash. Useful for finding addressable nodes."""
    program = _load()
    seen: set[str] = set()
    for hn in walk(program):
        if hn.hash in seen:
            continue
        seen.add(hn.hash)
        typer.echo(f"{hn.hash[:HASH_DISPLAY_LEN]}  {type(hn.node).__name__}")


# ---------- Mutation: claims ----------

_JustificationAdapter: TypeAdapter[Justification] = TypeAdapter(Justification)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_justification_spec(raw: str) -> Justification:
    """Parse a JSON Justification spec; auto-fill artifact_hash for kinds with
    artifact_path when the user omitted it."""
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
    kind: str, target: str | None, *,
    lo: int | None, hi: int | None,
    regime: str, enforcement: str, justification: Justification | None,
):
    if regime not in STORED_REGIMES:
        raise typer.BadParameter(
            f"can't add claim with regime={regime!r}: stored claims must be one of "
            f"{', '.join(STORED_REGIMES)}. Lattice claims are derived; see `cpg derive-claims`."
        )
    if enforcement not in ENFORCEMENTS:
        raise typer.BadParameter(f"unknown enforcement {enforcement!r}; choices: {', '.join(ENFORCEMENTS)}")
    if kind in PARAM_CLAIM_KINDS and target is None:
        raise typer.BadParameter(f"{kind!r} requires --target / -t (the parameter name)")
    if kind in RETURN_CLAIM_KINDS and target is not None:
        raise typer.BadParameter(f"{kind!r} is function-scoped; --target / -t must not be set")
    common = {"regime": regime, "enforcement": enforcement, "justification": justification}
    if kind == "non_negative":
        if lo is not None or hi is not None:
            raise typer.BadParameter("non_negative does not take --min / --max")
        return NonNegativeClaim(param=target, **common)
    if kind == "int_range":
        if lo is None and hi is None:
            raise typer.BadParameter("int_range requires --min and/or --max")
        return IntRangeClaim(param=target, min=lo, max=hi, **common)
    if kind == "return_in_range":
        if lo is None and hi is None:
            raise typer.BadParameter("return_in_range requires --min and/or --max")
        return ReturnInRangeClaim(min=lo, max=hi, **common)
    raise typer.BadParameter(f"unknown claim kind {kind!r}; choices: {', '.join(CLAIM_KINDS)}")


@app.command("add-claim")
def add_claim_cmd(
    kind: str = typer.Argument(..., help=f"Claim kind. One of: {', '.join(CLAIM_KINDS)}."),
    function: str = typer.Option(..., "--function", "-f", help="Function name or hash prefix."),
    target: str | None = typer.Option(
        None, "--target", "-t",
        help=f"Parameter name. Required for: {', '.join(PARAM_CLAIM_KINDS)}. "
             f"Must be omitted for: {', '.join(RETURN_CLAIM_KINDS)}.",
    ),
    lo: int | None = typer.Option(None, "--min", help="Lower bound."),
    hi: int | None = typer.Option(None, "--max", help="Upper bound."),
    regime: str = typer.Option(
        "axiom", "--regime",
        help=f"Epistemic source. One of: {', '.join(STORED_REGIMES)}. "
             f"(Lattice is derived, not stored — see `cpg derive-claims`.)",
    ),
    enforcement: str = typer.Option(
        "trust", "--enforcement",
        help=f"trust = llvm.assume (UB if false); "
             f"verify = runtime branch + abort if false. One of: {', '.join(ENFORCEMENTS)}.",
    ),
    justification: str | None = typer.Option(
        None, "--justification",
        help='JSON Justification spec, e.g. \'{"kind":"z3","artifact_path":"proofs/x.smt2"}\' '
             "(artifact_hash auto-filled from file if omitted). "
             'Or \'{"kind":"manual","signed_by":"alice","rationale":"..."}\'.',
    ),
) -> None:
    """Attach a claim to a function. The optimizer will trust this assertion."""
    program = _load()
    try:
        fn = find_function_ref(program, function)
        just_obj = _parse_justification_spec(justification) if justification else None
        claim = _build_claim(
            kind, target, lo=lo, hi=hi,
            regime=regime, enforcement=enforcement, justification=just_obj,
        )
        program = add_claim(program, fn.name, claim)
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    _save(program)
    typer.echo(f"added {kind}({target}) on {fn.name} [regime={regime}, enforcement={enforcement}]")


@app.command("relax-claim")
def relax_claim_cmd(
    kind: str = typer.Argument(..., help=f"Claim kind. One of: {', '.join(CLAIM_KINDS)}."),
    function: str = typer.Option(..., "--function", "-f", help="Function name or hash prefix."),
    target: str | None = typer.Option(
        None, "--target", "-t",
        help="Parameter name (omit for return-value claims).",
    ),
) -> None:
    """Remove a claim (always safe — drops an assertion)."""
    program = _load()
    try:
        fn = find_function_ref(program, function)
        program = relax_claim(program, fn.name, kind, target)
    except KeyError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    _save(program)
    scope = f"({target})" if target is not None else "(return)"
    typer.echo(f"relaxed {kind}{scope} on {fn.name}")


# ---------- Mutation: prove (MVP4) ----------

@app.command("prove")
def prove_cmd(
    kind: str = typer.Argument(..., help=f"Claim kind to prove. One of: {', '.join(CLAIM_KINDS)}."),
    function: str = typer.Option(..., "--function", "-f", help="Function name or hash prefix."),
    target: str | None = typer.Option(
        None, "--target", "-t",
        help=f"Parameter name. Required for: {', '.join(PARAM_CLAIM_KINDS)}.",
    ),
    lo: int | None = typer.Option(None, "--min", help="Lower bound."),
    hi: int | None = typer.Option(None, "--max", help="Upper bound."),
    proofs_dir: Path = typer.Option(
        Path("proofs"), "--proofs-dir",
        help="Directory to write the .smt2 artifact.",
    ),
    enforcement: str = typer.Option(
        "trust", "--enforcement",
        help=f"Enforcement for the resulting witness claim. One of: {', '.join(ENFORCEMENTS)}.",
    ),
) -> None:
    """Synthesize a proof of a claim, attach it as a witness.

    Pipeline: build goal claim from CLI args -> lower goal + function body
    + existing claims as hypotheses to SMT-LIB -> run Z3 -> if unsat, write
    artifact and add the claim with regime=witness + Z3Justification.
    """
    program = _load()
    try:
        fn = find_function_ref(program, function)
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)

    # Build the goal claim (no justification yet — that's what we're proving).
    try:
        goal = _build_claim(
            kind, target, lo=lo, hi=hi,
            regime="witness", enforcement=enforcement, justification=None,
        )
    except typer.BadParameter as e:
        typer.echo(f"error: {e.message}", err=True)
        raise typer.Exit(2)

    # Lower function + existing claims (as hypotheses) + negated goal to SMT.
    # Pass the full program so callees' return claims become per-call hypotheses.
    try:
        smt = goal_smt_lib(fn, goal, hypotheses=fn.claims, program=program)
    except NotImplementedError as e:
        typer.echo(f"error: cannot synthesize proof: {e}", err=True)
        raise typer.Exit(1)

    # Run Z3.
    try:
        result = run_z3_on_smt(smt)
    except Z3NotInstalled as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    if result.status != "unsat":
        typer.echo(f"could not prove {kind}: z3 returned {result.status!r}", err=True)
        if result.status == "sat":
            typer.echo("(z3 found a counterexample; the claim does not hold)", err=True)
        raise typer.Exit(1)

    # Write artifact + attach to a witness claim. Filename includes a content
    # hash prefix so different proofs (same fn/kind, different bounds) don't
    # collide — and same proof regenerates to the same path (idempotent).
    proofs_dir.mkdir(parents=True, exist_ok=True)
    target_part = target or "return"
    artifact_hash = hashlib.sha256(smt.encode("utf-8")).hexdigest()
    artifact_path = proofs_dir / f"{fn.name}_{kind}_{target_part}_{artifact_hash[:12]}.smt2"
    artifact_path.write_text(smt)

    proven = goal.model_copy(update={
        "justification": Z3Justification(
            artifact_path=str(artifact_path),
            artifact_hash=artifact_hash,
        ),
    })
    try:
        program = add_claim(program, fn.name, proven)
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    _save(program)
    typer.echo(
        f"proved {format_claim(proven)}\n"
        f"  artifact: {artifact_path} (sha256={artifact_hash[:12]})"
    )


# ---------- Mutation: construction ----------

@app.command("add-function")
def add_function_cmd(
    spec: str = typer.Argument("-", help="Path to JSON spec, or '-' for stdin (default)."),
) -> None:
    """Append a new function to the program. Spec is a JSON Function object.

    Example spec:
        {"name": "g", "params": ["x"], "body": [{"kind": "return_int", "value": 0}]}
    """
    program = _load()
    try:
        fn = parse_function_spec(read_json_arg(spec))
        program = add_function_to_program(program, fn)
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    _save(program)
    typer.echo(f"added function {fn.name} (hash={short_hash(fn)})")


@app.command("add-statement")
def add_statement_cmd(
    spec: str = typer.Argument("-", help="Path to JSON spec, or '-' for stdin (default)."),
    in_function: str = typer.Option(..., "--in-function", help="Function name or hash prefix."),
    at_end: bool = typer.Option(False, "--at-end"),
    at_start: bool = typer.Option(False, "--at-start"),
    before: str | None = typer.Option(None, "--before", help="Hash prefix of an existing statement."),
    after: str | None = typer.Option(None, "--after", help="Hash prefix of an existing statement."),
) -> None:
    """Insert a statement into a function. Exactly one anchor is required.

    Anchors: --at-end, --at-start, --before HASH, --after HASH.
    Spec is a JSON Statement object (a discriminated union; needs a `kind` field).
    """
    anchors = [at_end, at_start, before is not None, after is not None]
    if sum(map(bool, anchors)) != 1:
        typer.echo("error: pass exactly one of --at-end, --at-start, --before, --after", err=True)
        raise typer.Exit(2)

    program = _load()
    try:
        fn = find_function_ref(program, in_function)
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


# ---------- Compile ----------

_ENFORCE_HELP = (
    f"Override enforcement for all claims of this regime, regardless of stored "
    f"value. One of: {', '.join(ENFORCEMENTS)}."
)


@app.command()
def compile(
    profile: int = typer.Option(
        2, "--profile",
        min=0, max=3,
        help="LLVM optimization level. 0 skips the optimize pass entirely.",
    ),
    target: str | None = typer.Option(
        None, "--target",
        help="LLVM target triple (e.g. aarch64-unknown-linux-gnu). Defaults to host.",
    ),
    enforce_axiom: str | None = typer.Option(None, "--enforce-axiom", help=_ENFORCE_HELP),
    enforce_witness: str | None = typer.Option(None, "--enforce-witness", help=_ENFORCE_HELP),
    enforce_lattice: str | None = typer.Option(None, "--enforce-lattice", help=_ENFORCE_HELP),
    link: bool = typer.Option(True, "--link/--no-link"),
    run: bool = typer.Option(False, "--run", help="Execute the linked binary (requires `main`)."),
    show_ir: bool = typer.Option(False, "--show-ir", help="Print the optimized IR to stdout."),
    build_dir: Path = typer.Option(Path("build"), "--build-dir"),
) -> None:
    """Lower -> optimize -> object -> link -> (optional) run."""
    program = _load()
    overrides: dict[str, str] = {}
    for flag, regime, val in [
        ("--enforce-axiom", "axiom", enforce_axiom),
        ("--enforce-witness", "witness", enforce_witness),
        ("--enforce-lattice", "lattice", enforce_lattice),
    ]:
        if val is None:
            continue
        if val not in ENFORCEMENTS:
            raise typer.BadParameter(f"{flag}={val!r}; expected one of: {', '.join(ENFORCEMENTS)}")
        overrides[regime] = val
    try:
        result = lower_mod.compile_program(
            program, build_dir=build_dir, profile=profile, link=link, target=target,
            overrides=overrides,
        )
    except subprocess.CalledProcessError as e:
        # clang already printed its diagnostics to stderr; don't add a Python
        # traceback on top.
        typer.echo(f"error: link step failed (exit {e.returncode})", err=True)
        raise typer.Exit(e.returncode)

    typer.echo(f"emitted unoptimized IR -> {result.ir_unopt}")
    if result.ir_opt is not None:
        typer.echo(f"emitted optimized IR  -> {result.ir_opt}")
    typer.echo(f"emitted object         -> {result.object_path}")
    if result.binary is not None:
        typer.echo(f"linked binary          -> {result.binary}")

    if show_ir and result.ir_opt is not None:
        typer.echo("\n--- optimized IR ---")
        typer.echo(result.ir_opt.read_text())

    if run:
        if result.binary is None:
            typer.echo("error: --run requested but no `main` to link", err=True)
            raise typer.Exit(1)
        typer.echo("\n--- run ---")
        # Don't pass check=True: the binary's exit code is meaningful output
        # (a `main` that returns a computed i32 produces a nonzero exit by design).
        completed = subprocess.run([str(result.binary)], capture_output=True, text=True)
        typer.echo(f"stdout: {completed.stdout!r}")
        typer.echo(f"exit:   {completed.returncode}")


if __name__ == "__main__":
    app()
