"""Typer CLI. Noun-first sub-apps; each leaf command maps 1:1 to a tool call.

Layout:
    quod init / check / build / run     # lifecycle
    quod show [--hashes] / find PREFIX  # whole-program inspection
    quod fn ...                         # functions
    quod claim ...                      # claims
    quod stmt ...                       # statements
    quod extern ...                     # externs
    quod note ...                       # notes

Every command except `init` requires a quod.toml. `--config PATH` (default
./quod.toml) selects which one. Paths inside quod.toml resolve relative to
its parent dir, so `quod run -c /elsewhere/quod.toml` works regardless of
CWD; the launched binary inherits CWD from the invocation.

Function and statement references accept either a name (functions only) or
a content-hash prefix (any node).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import typer
from pydantic import TypeAdapter, ValidationError

from quod import lower as lower_mod
from quod.analysis import derive_lattice_claims
from quod.config import (
    Config,
    load_config,
    starter_toml,
    with_overrides,
)
from quod.editor import (
    add_constant_to_program,
    add_function_to_program,
    add_statement_in_function,
    find_function_ref,
    parse_function_spec,
    parse_statement_spec,
    read_json_arg,
    remove_statement_in_function,
)
from quod.hashing import HASH_DISPLAY_LEN, find_by_prefix, node_hash, short_hash, walk
from quod.model import (
    CLAIM_KINDS,
    PARAM_CLAIM_KINDS,
    RETURN_CLAIM_KINDS,
    DerivedJustification,
    ExternFunction,
    I32Type,
    I8PtrType,
    IntRangeClaim,
    Justification,
    ManualJustification,
    NonNegativeClaim,
    Program,
    ReturnInRangeClaim,
    StringConstant,
    Z3Justification,
    add_claim,
    claim_param,
    format_claim,
    format_function,
    format_program,
    function_callees,
    load_program,
    relax_claim,
    remove_function,
    replace_function,
    save_program,
)
from quod.proof import Z3NotInstalled, goal_smt_lib, run_z3_on_file, run_z3_on_smt
from quod.templates import TEMPLATES


REGIMES = ("axiom", "witness", "lattice")
STORED_REGIMES = ("axiom", "witness")  # lattice is derived, never stored
ENFORCEMENTS = ("trust", "verify")


# ---------- App tree ----------

app = typer.Typer(
    no_args_is_help=True,
    help="quod: edit a code-property graph and compile it through LLVM.",
    pretty_exceptions_show_locals=False,
)
fn_app = typer.Typer(no_args_is_help=True, help="Operations on functions.")
claim_app = typer.Typer(no_args_is_help=True, help="Operations on claims.")
stmt_app = typer.Typer(no_args_is_help=True, help="Operations on statements.")
extern_app = typer.Typer(no_args_is_help=True, help="Operations on externs.")
note_app = typer.Typer(no_args_is_help=True, help="Operations on notes.")
const_app = typer.Typer(no_args_is_help=True, help="Operations on string constants.")

app.add_typer(fn_app, name="fn")
app.add_typer(claim_app, name="claim")
app.add_typer(stmt_app, name="stmt")
app.add_typer(extern_app, name="extern")
app.add_typer(note_app, name="note")
app.add_typer(const_app, name="const")


# ---------- Shared state ----------

_state: dict[str, object] = {}


def _cfg_path() -> Path:
    return _state["config_path"]  # type: ignore[return-value]


def _cfg() -> Config:
    """Lazy-load quod.toml. Init writes the file; other commands read it."""
    if "config" not in _state:
        try:
            _state["config"] = load_config(_cfg_path())
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
    return _state["config"]  # type: ignore[return-value]


def _path() -> Path:
    cfg = _cfg()
    return cfg.resolve(cfg.program)


def _load() -> Program:
    p = _path()
    if not p.exists():
        typer.echo(f"error: {p} does not exist (run `quod init` first)", err=True)
        raise typer.Exit(1)
    return load_program(p)


def _save(program: Program) -> None:
    save_program(program, _path())


def _hash_label(node) -> str:
    return f"[{short_hash(node)}] "


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    config: Path = typer.Option(
        Path("quod.toml"), "--config", "-c",
        help="Path to quod.toml (default: ./quod.toml).",
    ),
) -> None:
    _state["config_path"] = config
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


# ---------- Lifecycle ----------

@app.command()
def init(
    template: str = typer.Option(
        "hello", "--template", "-t",
        help=f"Starter template. One of: {', '.join(TEMPLATES)}.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files."),
) -> None:
    """Initialize a project: writes quod.toml and program.json side-by-side."""
    if template not in TEMPLATES:
        typer.echo(f"error: unknown template {template!r}; choices: {', '.join(TEMPLATES)}", err=True)
        raise typer.Exit(2)

    cfg_path = _cfg_path().resolve()
    program_path = cfg_path.parent / "program.json"

    if cfg_path.exists() and not force:
        typer.echo(f"error: {cfg_path} already exists (use --force to overwrite)", err=True)
        raise typer.Exit(1)
    if program_path.exists() and not force:
        typer.echo(f"error: {program_path} already exists (use --force to overwrite)", err=True)
        raise typer.Exit(1)

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(starter_toml(template))
    save_program(TEMPLATES[template], program_path)

    typer.echo(f"wrote {cfg_path}")
    typer.echo(f"wrote {program_path} ({template} starter)")


@app.command()
def check() -> None:
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


def _build_impl(
    profile: int | None,
    target: str | None,
    link: bool | None,
    show_ir: bool,
    enforce_axiom: str | None,
    enforce_witness: str | None,
    enforce_lattice: str | None,
) -> tuple[Config, lower_mod.CompileResult]:
    cfg = _cfg()
    cfg = with_overrides(
        cfg,
        profile=profile, target=target, link=link,
        enforce_axiom=enforce_axiom,
        enforce_witness=enforce_witness,
        enforce_lattice=enforce_lattice,
    )
    if not cfg.bins:
        typer.echo(
            f"error: no [[bin]] entries in {_cfg_path()}; "
            f"declare at least one to build", err=True,
        )
        raise typer.Exit(1)
    program = _load()
    overrides = cfg.enforce.overrides()
    for regime, val in overrides.items():
        if val not in ENFORCEMENTS:
            raise typer.BadParameter(
                f"enforce.{regime}={val!r}; expected one of: {', '.join(ENFORCEMENTS)}"
            )
    bins = tuple((b.name, b.entry) for b in cfg.bins)
    target_or_none = cfg.build.target or None
    try:
        result = lower_mod.compile_program(
            program,
            build_dir=cfg.resolve(cfg.build_dir),
            bins=bins,
            profile=cfg.build.profile,
            link=cfg.build.link,
            target=target_or_none,
            overrides=overrides,
        )
    except subprocess.CalledProcessError as e:
        typer.echo(f"error: link step failed (exit {e.returncode})", err=True)
        raise typer.Exit(e.returncode)
    except (ValueError, KeyError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)

    for br in result.bins:
        typer.echo(f"[{br.name}] entry={br.entry}")
        typer.echo(f"  unopt IR -> {br.ir_unopt}")
        if br.ir_opt is not None:
            typer.echo(f"  opt IR   -> {br.ir_opt}")
        typer.echo(f"  object   -> {br.object_path}")
        if br.binary is not None:
            typer.echo(f"  binary   -> {br.binary}")
        if show_ir and br.ir_opt is not None:
            typer.echo(f"\n--- {br.name} optimized IR ---")
            typer.echo(br.ir_opt.read_text())
    return cfg, result


_ENFORCE_HELP = "Override enforcement for claims of this regime. trust|verify."


@app.command()
def build(
    profile: int | None = typer.Option(
        None, "--profile",
        help="LLVM optimization level (0..3). 0 skips the optimize pass entirely.",
    ),
    target: str | None = typer.Option(
        None, "--target",
        help="LLVM target triple. Defaults to host (or quod.toml [build].target).",
    ),
    link: bool | None = typer.Option(
        None, "--link/--no-link",
        help="Link object files into a binary (defaults to quod.toml [build].link).",
    ),
    show_ir: bool = typer.Option(False, "--show-ir", help="Print optimized IR to stdout."),
    enforce_axiom: str | None = typer.Option(None, "--enforce-axiom", help=_ENFORCE_HELP),
    enforce_witness: str | None = typer.Option(None, "--enforce-witness", help=_ENFORCE_HELP),
    enforce_lattice: str | None = typer.Option(None, "--enforce-lattice", help=_ENFORCE_HELP),
) -> None:
    """Lower -> optimize -> object -> link, for every [[bin]] in quod.toml."""
    if profile is not None and not 0 <= profile <= 3:
        raise typer.BadParameter("--profile must be in 0..3")
    _build_impl(profile, target, link, show_ir, enforce_axiom, enforce_witness, enforce_lattice)


@app.command()
def run(
    bin_name: str | None = typer.Argument(
        None, help="Which [[bin]] to run. Required if more than one is configured."
    ),
    profile: int | None = typer.Option(None, "--profile"),
    target: str | None = typer.Option(None, "--target"),
    enforce_axiom: str | None = typer.Option(None, "--enforce-axiom", help=_ENFORCE_HELP),
    enforce_witness: str | None = typer.Option(None, "--enforce-witness", help=_ENFORCE_HELP),
    enforce_lattice: str | None = typer.Option(None, "--enforce-lattice", help=_ENFORCE_HELP),
) -> None:
    """Build and execute a binary. Like `cargo run`."""
    cfg, result = _build_impl(
        profile, target, link=True, show_ir=False,
        enforce_axiom=enforce_axiom, enforce_witness=enforce_witness, enforce_lattice=enforce_lattice,
    )
    if bin_name is None:
        if len(result.bins) != 1:
            names = ", ".join(b.name for b in result.bins)
            typer.echo(f"error: multiple bins ({names}); pass one as the argument", err=True)
            raise typer.Exit(2)
        chosen = result.bins[0]
    else:
        chosen = next((b for b in result.bins if b.name == bin_name), None)
        if chosen is None:
            names = ", ".join(b.name for b in result.bins)
            typer.echo(f"error: no bin named {bin_name!r}; choices: {names}", err=True)
            raise typer.Exit(2)
    if chosen.binary is None:
        typer.echo(f"error: bin {chosen.name!r} was not linked", err=True)
        raise typer.Exit(1)
    typer.echo(f"\n--- {chosen.name} ---")
    completed = subprocess.run([str(chosen.binary)], capture_output=True, text=True)
    typer.echo(f"stdout: {completed.stdout!r}")
    typer.echo(f"exit:   {completed.returncode}")


# ---------- Whole-program inspection ----------

@app.command()
def show(
    hashes: bool = typer.Option(
        False, "--hashes",
        help="Dump every node and its short hash, instead of the program form.",
    ),
) -> None:
    """Print the program in canonical form, with content-hash prefixes."""
    program = _load()
    if hashes:
        seen: set[str] = set()
        for hn in walk(program):
            if hn.hash in seen:
                continue
            seen.add(hn.hash)
            typer.echo(f"{hn.hash[:HASH_DISPLAY_LEN]}  {type(hn.node).__name__}")
        return
    typer.echo(format_program(program, label=_hash_label))


@app.command()
def find(prefix: str) -> None:
    """Resolve a hash prefix to a node and print it."""
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


# ---------- fn sub-app ----------

@fn_app.command("ls")
def fn_ls() -> None:
    """List all functions with signatures and hashes."""
    program = _load()
    if not program.functions:
        typer.echo("(no functions)")
        return
    for fn in program.functions:
        sig = ", ".join(f"{p}: i32" for p in fn.params)
        suffix = f"  [{len(fn.claims)} claim(s)]" if fn.claims else ""
        typer.echo(f"[{short_hash(fn)}] {fn.name}({sig}) -> i32{suffix}")


@fn_app.command("show")
def fn_show(ref: str) -> None:
    """Print a single function. Accepts a name or a content-hash prefix."""
    try:
        fn = find_function_ref(_load(), ref)
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(format_function(fn, label=_hash_label))


@fn_app.command("add")
def fn_add(
    spec: str = typer.Argument("-", help="Path to JSON spec, or '-' for stdin."),
) -> None:
    """Append a new function. Spec is a JSON Function object.

    Example: {"name": "g", "params": ["x"], "body": [{"kind": "quod.return_int", "value": 0}]}
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


@fn_app.command("rm")
def fn_rm(
    function: str = typer.Argument(..., help="Function name or hash prefix."),
) -> None:
    """Remove a function from the program.

    Permissive: doesn't refuse if other functions still call this one. Run
    `quod fn callers FN` first if you want to know who'd be affected; the
    dangling call surfaces as an error at `quod build`.
    """
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
    target: str = typer.Argument(..., help="Function whose callers we want."),
) -> None:
    """List every call site to `target` across the program."""
    from quod.analysis import _walk_calls_in_stmt
    program = _load()
    found = False
    for caller in program.functions:
        for i, stmt in enumerate(caller.body):
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
    function: str = typer.Argument(..., help="Function name or hash prefix."),
    param: str = typer.Argument(..., help="Parameter name."),
) -> None:
    """Show every statement in `function` that reads `param`."""
    program = _load()
    try:
        fn = find_function_ref(program, function)
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    if param not in fn.params:
        typer.echo(f"error: {fn.name!r} has no parameter {param!r}", err=True)
        raise typer.Exit(1)
    any_read = False
    for i, stmt in enumerate(fn.body):
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
def fn_call_graph() -> None:
    """Print the static call graph."""
    program = _load()
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
def fn_unconstrained() -> None:
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


# ---------- claim sub-app ----------

@claim_app.command("ls")
def claim_ls(
    function: str | None = typer.Argument(None, help="Restrict to one function (omit for all)."),
) -> None:
    """List stored claims (axiom + witness regimes) across the program."""
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


_JustificationAdapter: TypeAdapter[Justification] = TypeAdapter(Justification)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


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
    kind: str, target: str | None, *,
    lo: int | None, hi: int | None,
    regime: str, enforcement: str, justification: Justification | None,
):
    if regime not in STORED_REGIMES:
        raise typer.BadParameter(
            f"can't add claim with regime={regime!r}: stored claims must be one of "
            f"{', '.join(STORED_REGIMES)}. Lattice claims are derived; see `quod claim derive`."
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


@claim_app.command("add")
def claim_add(
    function: str = typer.Argument(..., help="Function name or hash prefix."),
    kind: str = typer.Argument(..., help=f"Claim kind. One of: {', '.join(CLAIM_KINDS)}."),
    target: str | None = typer.Argument(
        None,
        help=f"Parameter name. Required for: {', '.join(PARAM_CLAIM_KINDS)}. "
             f"Must be omitted for: {', '.join(RETURN_CLAIM_KINDS)}.",
    ),
    lo: int | None = typer.Option(None, "--min"),
    hi: int | None = typer.Option(None, "--max"),
    regime: str = typer.Option(
        "axiom", "--regime",
        help=f"Epistemic source. One of: {', '.join(STORED_REGIMES)}.",
    ),
    enforcement: str = typer.Option(
        "trust", "--enforcement",
        help=f"trust = llvm.assume (UB if false); verify = runtime branch + abort. "
             f"One of: {', '.join(ENFORCEMENTS)}.",
    ),
    justification: str | None = typer.Option(
        None, "--justification",
        help='JSON Justification spec, e.g. \'{"kind":"z3","artifact_path":"proofs/x.smt2"}\'.',
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


@claim_app.command("relax")
def claim_relax(
    function: str = typer.Argument(..., help="Function name or hash prefix."),
    kind: str = typer.Argument(..., help=f"Claim kind. One of: {', '.join(CLAIM_KINDS)}."),
    target: str | None = typer.Argument(None, help="Parameter name (omit for return-scoped claims)."),
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


@claim_app.command("verify")
def claim_verify(
    root: Path = typer.Option(
        Path("."), "--root",
        help="Project root for resolving justification artifact_path.",
    ),
) -> None:
    """Re-check evidence attached to stored claims."""
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
    for delta, fn_name, claim in results[:top_n]:
        typer.echo(f"  -{delta:>3} lines  on {fn_name}: {format_claim(claim)}")
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


def _generate_candidates(program: Program) -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for fn in program.functions:
        existing = {(claim_param(c), c.kind) for c in fn.claims}
        for p in fn.params:
            if (p, "non_negative") not in existing:
                out.append((fn.name, NonNegativeClaim(param=p, regime="axiom")))
        has_return_claim = any(c.kind == "return_in_range" for c in fn.claims)
        if not has_return_claim:
            for lo in (-1, 0):
                out.append((fn.name, ReturnInRangeClaim(min=lo, regime="axiom")))
    return out


@claim_app.command("derive")
def claim_derive() -> None:
    """Run the lattice analysis and print derived (regime=lattice) claims."""
    program = _load()
    derived = derive_lattice_claims(program)
    if not derived:
        typer.echo("(no derived claims)")
        return
    for fn in program.functions:
        for c in derived.get(fn.name, ()):
            typer.echo(f"{fn.name}: {format_claim(c)}")


@claim_app.command("prove")
def claim_prove(
    function: str = typer.Argument(..., help="Function name or hash prefix."),
    kind: str = typer.Argument(..., help=f"Claim kind to prove. One of: {', '.join(CLAIM_KINDS)}."),
    target: str | None = typer.Argument(None, help="Parameter name (omit for return-scoped claims)."),
    lo: int | None = typer.Option(None, "--min"),
    hi: int | None = typer.Option(None, "--max"),
    enforcement: str = typer.Option("trust", "--enforcement"),
) -> None:
    """Synthesize a proof of a claim, attach it as a witness."""
    cfg = _cfg()
    proofs_dir = cfg.resolve(cfg.proofs_dir)
    program = _load()
    try:
        fn = find_function_ref(program, function)
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)

    try:
        goal = _build_claim(
            kind, target, lo=lo, hi=hi,
            regime="witness", enforcement=enforcement, justification=None,
        )
    except typer.BadParameter as e:
        typer.echo(f"error: {e.message}", err=True)
        raise typer.Exit(2)

    try:
        smt = goal_smt_lib(fn, goal, hypotheses=fn.claims, program=program)
    except NotImplementedError as e:
        typer.echo(f"error: cannot synthesize proof: {e}", err=True)
        raise typer.Exit(1)

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


# ---------- stmt sub-app ----------

@stmt_app.command("add")
def stmt_add(
    function: str = typer.Argument(..., help="Function name or hash prefix."),
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
    function: str = typer.Argument(..., help="Function name or hash prefix."),
    hash_prefix: str = typer.Argument(
        ..., help="Content-hash prefix of the statement to remove."
    ),
) -> None:
    """Remove a statement from a function by content-hash prefix.

    Find the hash via `quod fn show FN` (each statement is shown with its
    short hash) or `quod show --hashes`.
    """
    program = _load()
    try:
        fn = find_function_ref(program, function)
        program = remove_statement_in_function(program, fn, hash_prefix)
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    _save(program)
    typer.echo(f"removed statement {hash_prefix} from {fn.name}")


# ---------- const sub-app ----------

@const_app.command("ls")
def const_ls() -> None:
    """List declared string constants."""
    program = _load()
    if not program.constants:
        typer.echo("(no constants)")
        return
    for c in program.constants:
        typer.echo(f"[{short_hash(c)}] {c.name} = {c.value!r}")


@const_app.command("add")
def const_add(
    name: str = typer.Argument(..., help="Constant name (e.g. '.str.fmt')."),
    value: str = typer.Argument(..., help="Constant value (raw string; not C-escaped)."),
) -> None:
    """Declare a string constant. Reference it from code with quod.string_ref.

    The value is the raw string as you want it in the program. To embed a
    newline, pass an actual newline (the shell will likely need $'...\\n' or
    a heredoc). Quod adds a trailing NUL byte automatically when lowering.
    """
    program = _load()
    try:
        program = add_constant_to_program(program, StringConstant(name=name, value=value))
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    _save(program)
    typer.echo(f"declared constant {name} = {value!r}")


# ---------- extern sub-app ----------

_TYPE_NAMES = {"i32": I32Type, "i8_ptr": I8PtrType}


def _parse_type_name(s: str):
    cls = _TYPE_NAMES.get(s)
    if cls is None:
        raise typer.BadParameter(f"unknown type {s!r}; choices: {', '.join(_TYPE_NAMES)}")
    return cls()


def _format_type(t) -> str:
    return {I32Type: "i32", I8PtrType: "i8_ptr"}.get(type(t), type(t).__name__)


@extern_app.command("ls")
def extern_ls() -> None:
    """List declared externs with their signatures."""
    program = _load()
    if not program.externs:
        typer.echo("(no externs)")
        return
    for ext in program.externs:
        if ext.param_types:
            params = [_format_type(t) for t in ext.param_types]
        else:
            params = ["i32"] * ext.arity
        if ext.varargs:
            params.append("...")
        ret = _format_type(ext.return_type)
        typer.echo(f"{ext.name}({', '.join(params)}) -> {ret}")


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
) -> None:
    """Declare an extern (libc-or-similar) function."""
    program = _load()
    if any(ext.name == name for ext in program.externs):
        typer.echo(f"error: extern {name!r} already declared", err=True)
        raise typer.Exit(1)
    if any(fn.name == name for fn in program.functions):
        typer.echo(f"error: {name!r} already exists as a user function", err=True)
        raise typer.Exit(1)
    if param_type and arity:
        raise typer.BadParameter("pass either --arity or --param-type, not both")
    param_types = tuple(_parse_type_name(t) for t in param_type)
    ret_ty = _parse_type_name(return_type)
    ext = ExternFunction(
        name=name,
        arity=arity if not param_types else 0,
        param_types=param_types,
        return_type=ret_ty,
        varargs=varargs,
    )
    new_externs = program.externs + (ext,)
    program = program.model_copy(update={"externs": new_externs})
    _save(program)
    sig_parts = list(param_type or ["i32"] * arity)
    if varargs:
        sig_parts.append("...")
    typer.echo(f"declared extern {name}({', '.join(sig_parts)}) -> {return_type}")


# ---------- note sub-app ----------

@note_app.command("add")
def note_add(
    function: str = typer.Argument(..., help="Function name or hash prefix."),
    text: str = typer.Argument(..., help="Note content (free-form intent / TODO / rationale)."),
) -> None:
    """Attach a free-form note to a function."""
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
    function: str = typer.Argument(..., help="Function name or hash prefix."),
    index: int = typer.Argument(..., help="0-based index of the note to remove."),
) -> None:
    """Remove a note by index from a function."""
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


if __name__ == "__main__":
    app()
