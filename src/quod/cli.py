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

import fcntl
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
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
    add_enum_to_program,
    add_struct_to_program,
    find_function_ref,
    parse_enum_spec,
    parse_function_spec,
    parse_statement_spec,
    read_json_arg,
    remove_constant_from_program,
    remove_extern_from_program,
    remove_statement_in_function,
    remove_enum_from_program,
    remove_struct_from_program,
)
from quod.hashing import HASH_DISPLAY_LEN, find_by_prefix, node_hash, short_hash, walk
from quod.ingest import IngestError, ingest_c, ingest_header
from quod.model import (
    CLAIM_KINDS,
    PARAM_CLAIM_KINDS,
    RETURN_CLAIM_KINDS,
    DerivedJustification,
    ExternFunction,
    I1Type,
    I8PtrType,
    I8Type,
    I16Type,
    I32Type,
    I64Type,
    IntRangeClaim,
    Justification,
    ManualJustification,
    NonNegativeClaim,
    Program,
    ReturnInRangeClaim,
    StringConstant,
    StructDef,
    StructField,
    StructType,
    Z3Justification,
    add_claim,
    claim_param,
    function_callees,
    load_program,
    relax_claim,
    remove_function,
    replace_function,
    save_program,
)
from quod.proof import Z3NotInstalled, run_z3_on_file
from quod.providers import (
    ClaimRequest,
    all_providers,
    default_for,
    get_provider,
)
from quod import completion as _comp
from quod.render import (
    Span,
    Theme,
    ansi_theme,
    claim_full_spans,
    claim_spans,
    constant_spans,
    extern_signature_spans,
    format_function_lines,
    format_program_lines,
    function_signature_spans,
    hash_brackets,
    paint,
    plain_theme,
    render,
    struct_def_spans,
)
from quod.schema import render_categories, render_category, render_kind
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
struct_app = typer.Typer(no_args_is_help=True, help="Operations on struct definitions.")
enum_app = typer.Typer(no_args_is_help=True, help="Operations on enum (sum-type) definitions.")
provider_app = typer.Typer(no_args_is_help=True, help="Inspect registered claim providers.")

app.add_typer(fn_app, name="fn")
app.add_typer(claim_app, name="claim")
app.add_typer(stmt_app, name="stmt")
app.add_typer(extern_app, name="extern")
app.add_typer(note_app, name="note")
app.add_typer(const_app, name="const")
app.add_typer(struct_app, name="struct")
app.add_typer(enum_app, name="enum")
app.add_typer(provider_app, name="provider")


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


def _selected_program_name() -> str | None:
    return _state.get("program_name")  # type: ignore[return-value]


def _selected_program():
    cfg = _cfg()
    try:
        return cfg.select(_selected_program_name())
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)


def _path() -> Path:
    cfg = _cfg()
    return cfg.resolve(_selected_program().file)


def _load() -> Program:
    p = _path()
    if not p.exists():
        typer.echo(f"error: {p} does not exist (run `quod init` first)", err=True)
        raise typer.Exit(1)
    return load_program(p)


def _save(program: Program) -> None:
    save_program(program, _path())


@contextmanager
def _exclusive_lock():
    """Hold an exclusive advisory lock for the duration of a mutation.

    Cooperating quod invocations serialize on this lock to avoid the
    load → mutate → save race where parallel writers clobber each other's
    in-memory state at the save step. The lock lives on a sidecar file
    (`<program>.lock`) so that save_program's atomic rename doesn't break
    the lock by replacing the locked inode.

    Read-only commands don't need the lock — save_program writes atomically
    via tmp + rename, so readers see either the old or new file, never a
    half-written one.
    """
    lock_path = _path().with_suffix(_path().suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)
    with open(lock_path, "rb") as lock_fd:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)


def _color_on() -> bool:
    """Color on iff stdout is a TTY, NO_COLOR is unset, and --no-color wasn't passed."""
    if _state.get("no_color"):
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _theme() -> Theme:
    return ansi_theme if _color_on() else plain_theme


def _json_default(o):
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


def _emit_json(payload) -> None:
    """Print a JSON payload. Pydantic models are serialized via model_dump."""
    typer.echo(json.dumps(payload, default=_json_default, indent=2))


_JSON_HELP = "Emit machine-readable JSON instead of human-readable output."


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    config: Path = typer.Option(
        Path("quod.toml"), "--config", "-c",
        help="Path to quod.toml (default: ./quod.toml).",
    ),
    program: str | None = typer.Option(
        None, "--program", "-p",
        help="Which [[program]] to operate on (omit if quod.toml has only one).",
        autocompletion=_comp.program_names,
    ),
    no_color: bool = typer.Option(
        False, "--no-color",
        help="Disable ANSI color even on a TTY. NO_COLOR env var also works.",
    ),
) -> None:
    _state["config_path"] = config
    _state["program_name"] = program
    _state["no_color"] = no_color
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


# ---------- Lifecycle ----------

@app.command()
def init(
    template: str = typer.Option(
        "hello", "--template", "-t",
        help=f"Starter template. One of: {', '.join(TEMPLATES)}.",
        autocompletion=_comp.template_names,
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

    next_steps = {
        "hello": "next: `quod show` to inspect, `quod run` to compile and execute.",
        "guarded": "next: `quod fn show f` to see the guarded function, "
                   "`quod claim suggest` to find provable optimizations.",
        "empty": "next: `quod fn add` to start writing functions, "
                 "or `quod schema` to discover node shapes.",
    }
    typer.echo(f"\n{next_steps[template]}")


@app.command()
def ingest(
    source: Path = typer.Argument(..., help="Source file to ingest (e.g. hello.c)."),
    name: str = typer.Option(
        None, "--name", "-n",
        help="Program name in quod.toml. Defaults to the source file's stem.",
    ),
    imports: list[str] = typer.Option(
        [], "--import",
        help=(
            "Stdlib module to add to the resulting program's `imports` list. "
            "Repeatable. The module must exist under quod's stdlib directory; "
            "see `quod schema --category program` for what's available."
        ),
    ),
) -> None:
    """Ingest a source file into a fresh quod project.

    Sibling to `init`: writes `quod.toml` and `program.json` in the cwd, and
    refuses if either already exists. Currently supports C only.
    """
    if source.suffix != ".c":
        typer.echo(f"error: only `.c` files are supported (got {source.suffix!r})", err=True)
        raise typer.Exit(2)
    if not source.exists():
        typer.echo(f"error: {source} does not exist", err=True)
        raise typer.Exit(1)

    cfg_path = _cfg_path().resolve()
    program_path = cfg_path.parent / "program.json"
    if cfg_path.exists():
        typer.echo(f"error: {cfg_path} already exists", err=True)
        raise typer.Exit(1)
    if program_path.exists():
        typer.echo(f"error: {program_path} already exists", err=True)
        raise typer.Exit(1)

    try:
        program = ingest_c(source)
    except IngestError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)

    if imports:
        program = program.model_copy(update={"imports": tuple(imports)})

    program_name = name or source.stem
    main_fn = next((f for f in program.functions if f.name == "main" and not f.params), None)
    bin_block = (
        f"\n  [[program.bin]]\n  name  = \"{program_name}\"\n  entry = \"main\"\n"
        if main_fn is not None
        else ""
    )
    toml = (
        "[build]\nprofile = 2\n\n"
        f"[[program]]\nname    = \"{program_name}\"\nversion = \"0.1.0\"\nfile    = \"program.json\"\n"
        f"{bin_block}"
    )

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(toml)
    save_program(program, program_path)

    typer.echo(f"wrote {cfg_path}")
    typer.echo(f"wrote {program_path} ({len(program.functions)} function(s) ingested from {source})")
    if main_fn is None:
        typer.echo("\nnote: no `int main()` found — add a [[program.bin]] entry to build a binary.")
    typer.echo("\nnext: `quod show` to inspect, `quod check` to verify it lowers cleanly.")


@app.command()
def check() -> None:
    """Parse, lower, and LLVM-verify each program. No artifacts emitted.

    With multiple `[[program]]` entries, checks all of them by default; pass
    `--program / -p NAME` (at the root level) to check just one.
    """
    cfg = _cfg()
    selector = _selected_program_name()
    if selector is None:
        targets = cfg.programs
    else:
        try:
            targets = (cfg.select(selector),)
        except ValueError as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
    if not targets:
        typer.echo(f"error: no [[program]] entries in {_cfg_path()}", err=True)
        raise typer.Exit(1)
    failures = 0
    for prog in targets:
        program_path = cfg.resolve(prog.file)
        if not program_path.exists():
            typer.echo(f"[{prog.name}] FAIL: {program_path} does not exist")
            failures += 1
            continue
        try:
            program_obj = load_program(program_path)
            module = lower_mod.lower(program_obj)
            parsed = lower_mod.parse_and_verify(module)
        except (ValueError, KeyError) as e:
            typer.echo(f"[{prog.name}] FAIL: {e}")
            failures += 1
            continue
        del parsed
        typer.echo(f"[{prog.name}] ok")
    if failures:
        raise typer.Exit(1)


def _build_impl(
    profile: int | None,
    target: str | None,
    link: bool | None,
    show_ir: bool,
    enforce_axiom: str | None,
    enforce_witness: str | None,
    enforce_lattice: str | None,
    *,
    no_std: bool = False,
    no_alloc: bool = False,
) -> tuple[Config, tuple[lower_mod.BinResult, ...]]:
    cfg = _cfg()
    cfg = with_overrides(
        cfg,
        profile=profile, target=target, link=link,
        enforce_axiom=enforce_axiom,
        enforce_witness=enforce_witness,
        enforce_lattice=enforce_lattice,
    )
    overrides = cfg.enforce.overrides()
    for regime, val in overrides.items():
        if val not in ENFORCEMENTS:
            raise typer.BadParameter(
                f"enforce.{regime}={val!r}; expected one of: {', '.join(ENFORCEMENTS)}"
            )

    selector = _selected_program_name()
    if selector is None:
        if not cfg.programs:
            typer.echo(
                f"error: no [[program]] entries in {_cfg_path()}; "
                f"declare at least one to build", err=True,
            )
            raise typer.Exit(1)
        targets = cfg.programs
    else:
        try:
            targets = (cfg.select(selector),)
        except ValueError as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)

    if not any(prog.bins for prog in targets):
        typer.echo(
            f"error: no [[program.bin]] entries in {_cfg_path()}; "
            f"declare at least one to build", err=True,
        )
        raise typer.Exit(1)

    target_or_none = cfg.build.target or None
    # --no-alloc subsumes --no-std (alloc < std in the dependency stack).
    disabled_tiers: set[str] = set()
    if no_std:
        disabled_tiers.add("std")
    if no_alloc:
        disabled_tiers.add("alloc")
        disabled_tiers.add("std")
    disabled_tiers_fz = frozenset(disabled_tiers)
    all_results: list[lower_mod.BinResult] = []
    for prog in targets:
        if not prog.bins:
            continue
        program_path = cfg.resolve(prog.file)
        if not program_path.exists():
            typer.echo(f"error: {program_path} does not exist", err=True)
            raise typer.Exit(1)
        program_obj = load_program(program_path)
        bins = tuple((b.name, b.entry) for b in prog.bins)
        try:
            result = lower_mod.compile_program(
                program_obj,
                build_dir=cfg.resolve(cfg.build_dir) / prog.name,
                bins=bins,
                profile=cfg.build.profile,
                link=cfg.build.link,
                libraries=cfg.link.libraries,
                target=target_or_none,
                overrides=overrides,
                disabled_tiers=disabled_tiers_fz,
            )
        except subprocess.CalledProcessError as e:
            typer.echo(f"error: link step failed (exit {e.returncode})", err=True)
            raise typer.Exit(e.returncode)
        except (ValueError, KeyError) as e:
            typer.echo(f"error: [{prog.name}] {e}", err=True)
            raise typer.Exit(1)

        for br in result.bins:
            typer.echo(f"[{prog.name}/{br.name}] entry={br.entry}")
            typer.echo(f"  unopt IR -> {br.ir_unopt}")
            if br.ir_opt is not None:
                typer.echo(f"  opt IR   -> {br.ir_opt}")
            typer.echo(f"  object   -> {br.object_path}")
            if br.binary is not None:
                typer.echo(f"  binary   -> {br.binary}")
            if show_ir and br.ir_opt is not None:
                typer.echo(f"\n--- {prog.name}/{br.name} optimized IR ---")
                typer.echo(br.ir_opt.read_text())
        all_results.extend(result.bins)
    return cfg, tuple(all_results)


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
    enforce_axiom: str | None = typer.Option(None, "--enforce-axiom", help=_ENFORCE_HELP,
                                              autocompletion=_comp.enforcements),
    enforce_witness: str | None = typer.Option(None, "--enforce-witness", help=_ENFORCE_HELP,
                                                autocompletion=_comp.enforcements),
    enforce_lattice: str | None = typer.Option(None, "--enforce-lattice", help=_ENFORCE_HELP,
                                                autocompletion=_comp.enforcements),
    no_std: bool = typer.Option(
        False, "--no-std",
        help="Refuse to resolve imports from the std.* tier (OS-dependent). "
             "core.* and alloc.* still available.",
    ),
    no_alloc: bool = typer.Option(
        False, "--no-alloc",
        help="Refuse to resolve imports from alloc.* and std.*; refuse "
             "with_arena. Bare-metal mode — only core.* available.",
    ),
) -> None:
    """Lower -> optimize -> object -> link, every [[program.bin]] in quod.toml.

    With multiple `[[program]]` entries, builds all of them by default; pass
    `--program / -p NAME` (at the root level) to build just one.
    """
    if profile is not None and not 0 <= profile <= 3:
        raise typer.BadParameter("--profile must be in 0..3")
    _build_impl(profile, target, link, show_ir, enforce_axiom, enforce_witness,
                enforce_lattice, no_std=no_std, no_alloc=no_alloc)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run(
    bin_name: str | None = typer.Option(
        None, "--bin", help="Which [[program.bin]] to run. Required if multiple bins are configured.",
        autocompletion=_comp.bin_names,
    ),
    profile: int | None = typer.Option(None, "--profile"),
    target: str | None = typer.Option(None, "--target"),
    enforce_axiom: str | None = typer.Option(None, "--enforce-axiom", help=_ENFORCE_HELP,
                                              autocompletion=_comp.enforcements),
    enforce_witness: str | None = typer.Option(None, "--enforce-witness", help=_ENFORCE_HELP,
                                                autocompletion=_comp.enforcements),
    enforce_lattice: str | None = typer.Option(None, "--enforce-lattice", help=_ENFORCE_HELP,
                                                autocompletion=_comp.enforcements),
    no_std: bool = typer.Option(False, "--no-std"),
    no_alloc: bool = typer.Option(False, "--no-alloc"),
) -> None:
    """Build and execute a binary. Like `cargo run`.

    Usage:
        quod run                            # single bin, no program args
        quod run --bin NAME                 # pick a bin, no program args
        quod run -- ARG ...                 # forward ARGs as argv to the binary
        quod run --bin NAME -- ARG ...      # both

    If the entry function declares int params, the synthesized main wrapper
    parses each argv slot via atoll, then trunc/sext's to the param's width.
    """
    # Click eats `--` and folds args into typer's parameter parsing, so we
    # read sys.argv directly to recover whatever was passed after `--`.
    import sys
    program_args: list[str] = []
    if "--" in sys.argv:
        program_args = sys.argv[sys.argv.index("--") + 1:]
    cfg, bin_results = _build_impl(
        profile, target, link=True, show_ir=False,
        enforce_axiom=enforce_axiom, enforce_witness=enforce_witness, enforce_lattice=enforce_lattice,
        no_std=no_std, no_alloc=no_alloc,
    )
    if bin_name is None:
        if len(bin_results) != 1:
            names = ", ".join(b.name for b in bin_results)
            typer.echo(f"error: multiple bins ({names}); pass --bin NAME", err=True)
            raise typer.Exit(2)
        chosen = bin_results[0]
    else:
        matches = [b for b in bin_results if b.name == bin_name]
        if not matches:
            names = ", ".join(b.name for b in bin_results)
            typer.echo(f"error: no bin named {bin_name!r}; choices: {names}", err=True)
            raise typer.Exit(2)
        if len(matches) > 1:
            typer.echo(
                f"error: bin name {bin_name!r} appears in multiple programs; "
                f"pass --program / -p NAME at the root to disambiguate", err=True,
            )
            raise typer.Exit(2)
        chosen = matches[0]
    if chosen.binary is None:
        typer.echo(f"error: bin {chosen.name!r} was not linked", err=True)
        raise typer.Exit(1)
    typer.echo(f"\n--- {chosen.name} ---")
    cmd = [str(chosen.binary), *program_args]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if program_args:
        typer.echo(f"argv:   {program_args}")
    typer.echo(f"stdout: {completed.stdout!r}")
    typer.echo(f"exit:   {completed.returncode}")


# ---------- Whole-program inspection ----------

@app.command()
def show(
    hashes: bool = typer.Option(
        False, "--hashes",
        help="Dump every node and its short hash, instead of the program form.",
    ),
    json_output: bool = typer.Option(False, "--json", help=_JSON_HELP),
) -> None:
    """Print the program. Color follows TTY (disable with `quod --no-color`)."""
    program = _load()
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


# ---------- fn sub-app ----------

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
) -> None:
    """Print a single function. Accepts a name or a content-hash prefix."""
    try:
        fn = find_function_ref(_load(), ref)
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    if json_output:
        _emit_json(fn)
        return
    typer.echo(render(format_function_lines(fn), theme=_theme(), mode="columnar"))


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
                from quod.stdlib import resolve_imports as _resolve_imports
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


# ---------- claim sub-app ----------

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
    function: str = typer.Argument(..., help="Function name or hash prefix.",
                                    autocompletion=_comp.function_or_hash),
    kind: str = typer.Argument(..., help=f"Claim kind. One of: {', '.join(CLAIM_KINDS)}.",
                               autocompletion=_comp.claim_kinds),
    target: str | None = typer.Argument(
        None,
        help=f"Parameter name. Required for: {', '.join(PARAM_CLAIM_KINDS)}. "
             f"Must be omitted for: {', '.join(RETURN_CLAIM_KINDS)}.",
        autocompletion=_comp.param_names_for_function,
    ),
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
        help='JSON Justification spec, e.g. \'{"kind":"z3","artifact_path":"proofs/x.smt2"}\'.',
    ),
) -> None:
    """Attach a claim to a function. The optimizer will trust this assertion."""
    with _exclusive_lock():
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
    function: str = typer.Argument(..., help="Function name or hash prefix.",
                                    autocompletion=_comp.function_or_hash),
    kind: str = typer.Argument(..., help=f"Claim kind. One of: {', '.join(CLAIM_KINDS)}.",
                               autocompletion=_comp.claim_kinds),
    target: str | None = typer.Argument(None, help="Parameter name (omit for return-scoped claims).",
                                        autocompletion=_comp.param_names_for_function),
) -> None:
    """Remove a claim (always safe — drops an assertion)."""
    with _exclusive_lock():
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
    root: Path | None = typer.Option(
        None, "--root",
        help="Root for resolving justification artifact_path. "
             "Defaults to the quod.toml directory.",
    ),
) -> None:
    """Re-check evidence attached to stored claims."""
    cfg = _cfg()
    program = _load()
    resolve_root = root if root is not None else cfg.root
    theme = _theme()
    failures = 0
    checked = 0
    for fn in program.functions:
        for c in fn.claims:
            if c.justification is None:
                continue
            checked += 1
            ok, msg = _verify_justification(c.justification, resolve_root)
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


def _generate_candidates(program: Program) -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for fn in program.functions:
        existing = {(claim_param(c), c.kind) for c in fn.claims}
        for p in fn.params:
            if (p.name, "non_negative") not in existing:
                out.append((fn.name, NonNegativeClaim(param=p.name, regime="axiom")))
        has_return_claim = any(c.kind == "return_in_range" for c in fn.claims)
        if not has_return_claim:
            for lo in (-1, 0):
                out.append((fn.name, ReturnInRangeClaim(min=lo, regime="axiom")))
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
    kind: str = typer.Argument(..., help=f"Claim kind to prove. One of: {', '.join(CLAIM_KINDS)}.",
                               autocompletion=_comp.claim_kinds),
    target: str | None = typer.Argument(None, help="Parameter name (omit for return-scoped claims).",
                                        autocompletion=_comp.param_names_for_function),
    lo: int | None = typer.Option(None, "--min"),
    hi: int | None = typer.Option(None, "--max"),
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
    if kind not in CLAIM_KINDS:
        typer.echo(f"error: unknown claim kind {kind!r}; one of: {', '.join(CLAIM_KINDS)}", err=True)
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
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)

        request = ClaimRequest(
            function=fn.name, kind=kind, target=target,
            min=lo, max=hi, enforcement=enforcement,
        )
        result = prov.prove(program, request, proofs_dir)
        if result.status != "proven":
            tag = result.status
            typer.echo(f"could not prove {kind}: {prov.name} reported {tag} ({result.detail})", err=True)
            if tag == "refuted":
                typer.echo("(provider found a counterexample; the claim does not hold)", err=True)
            raise typer.Exit(1)

        assert result.claim is not None
        try:
            program = add_claim(program, fn.name, result.claim)
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
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


# ---------- stmt sub-app ----------

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


# ---------- const sub-app ----------

@const_app.command("ls")
def const_ls(
    json_output: bool = typer.Option(False, "--json", help=_JSON_HELP),
) -> None:
    """List declared string constants."""
    program = _load()
    if json_output:
        _emit_json(list(program.constants))
        return
    if not program.constants:
        typer.echo("(no constants)")
        return
    theme = _theme()
    for c in program.constants:
        typer.echo(paint((
            *hash_brackets(c), Span(" ", "ws"), *constant_spans(c),
        ), theme))


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
    with _exclusive_lock():
        program = _load()
        try:
            program = add_constant_to_program(program, StringConstant(name=name, value=value))
        except (KeyError, ValueError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"declared constant {name} = {value!r}")


@const_app.command("rm")
def const_rm(
    name: str = typer.Argument(..., help="Constant name to remove.",
                               autocompletion=_comp.constant_names),
) -> None:
    """Remove a string constant from the program.

    Permissive: doesn't refuse if a quod.string_ref still points at it. The
    dangling reference surfaces at `quod build` time.
    """
    with _exclusive_lock():
        program = _load()
        try:
            program = remove_constant_from_program(program, name)
        except KeyError as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        _save(program)
    typer.echo(f"removed constant {name}")


# ---------- extern sub-app ----------

_TYPE_NAMES = {
    "i1": I1Type, "i8": I8Type, "i16": I16Type, "i32": I32Type, "i64": I64Type,
    "i8_ptr": I8PtrType,
}


def _parse_type_name(
    s: str,
    *,
    struct_names: tuple[str, ...] = (),
    enum_names: tuple[str, ...] = (),
):
    """Parse a CLI type token. Accepts the built-in width names plus any
    `struct_names` (-> StructType) and `enum_names` (-> EnumType) passed
    in. Pass the program's current names to allow struct/enum types in
    extern or struct-field declarations; pass nothing for legacy
    (int-only) callsites."""
    from quod.model import EnumType
    cls = _TYPE_NAMES.get(s)
    if cls is not None:
        return cls()
    if s in struct_names:
        return StructType(name=s)
    if s in enum_names:
        return EnumType(name=s)
    choices = list(_TYPE_NAMES) + list(struct_names) + list(enum_names)
    raise typer.BadParameter(
        f"unknown type {s!r}; choices: {', '.join(choices)}"
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
) -> None:
    """Declare an extern (libc-or-similar) function."""
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


# ---------- struct sub-app ----------

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


# ---------- enum sub-app ----------

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

    The CLI surface for enums is JSON-only (for now) — variant payloads
    have enough structure that the shorthand `name:type` form for structs
    doesn't generalize cleanly. Author the EnumDef as a JSON object and
    pipe it in: `cat enum.json | quod enum add -`.

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


# ---------- note sub-app ----------

@note_app.command("add")
def note_add(
    function: str = typer.Argument(..., help="Function name or hash prefix.",
                                    autocompletion=_comp.function_or_hash),
    text: str = typer.Argument(..., help="Note content (free-form intent / TODO / rationale)."),
) -> None:
    """Attach a free-form note to a function."""
    with _exclusive_lock():
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
    function: str = typer.Argument(..., help="Function name or hash prefix.",
                                    autocompletion=_comp.function_or_hash),
    index: int = typer.Argument(..., help="0-based index of the note to remove."),
) -> None:
    """Remove a note by index from a function."""
    with _exclusive_lock():
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


# ---------- provider sub-app ----------

@provider_app.command("ls")
def provider_ls() -> None:
    """List registered claim providers (regimes + supported modes)."""
    providers = all_providers()
    if not providers:
        typer.echo("(no providers registered)")
        return
    theme = _theme()
    name_w = max(len(p.name) for p in providers.values())
    regime_w = max(len(p.regime) for p in providers.values())
    for p in providers.values():
        modes = "+".join(p.modes) if p.modes else "(none)"
        name_pad = " " * (name_w - len(p.name))
        regime_pad = " " * (regime_w - len(p.regime))
        typer.echo(paint((
            Span(p.name, "fn_name"), Span(name_pad, "ws"), Span("  ", "ws"),
            Span("regime=", "meta_label"), Span(p.regime, "meta_value"),
            Span(regime_pad, "ws"), Span("  ", "ws"),
            Span("modes=", "meta_label"), Span(modes, "meta_value"),
        ), theme))
        typer.echo(paint((Span(f"  {p.description}", "comment"),), theme))


if __name__ == "__main__":
    app()
