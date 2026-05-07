"""Root Typer app, sub-app declarations, root callback, lifecycle commands.

Owns the `app` instance imported by the `quod` console_scripts entry-point
plus the lifecycle commands (`init`, `check`, `build`, `run`) that hang
directly off the root.

Each sub-app (`fn_app`, `claim_app`, etc.) is declared here; the
sub-app's commands are registered in their respective `cli_*.py` modules.
The bottom of `cli/__init__.py` imports those modules so registration
fires at package-import time.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

from quod import completion as _comp
from quod import lower as lower_mod
from quod.cli.output import ENFORCEMENTS, _emit_json, _JSON_HELP
from quod.cli.state import _cfg, _cfg_path, _selected_program_name, _state
from quod.config import Config, with_overrides
from quod.editor import parse_function_spec, read_json_arg
from quod.model import Program, load_program, save_program
from quod.templates import TEMPLATES
from quod.validate import ValidationError as ValidationError_


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
extern_claim_app = typer.Typer(no_args_is_help=True, help="Operations on extern claims.")
note_app = typer.Typer(no_args_is_help=True, help="Operations on notes.")
const_app = typer.Typer(no_args_is_help=True, help="Operations on string constants.")
struct_app = typer.Typer(no_args_is_help=True, help="Operations on struct definitions.")
enum_app = typer.Typer(no_args_is_help=True, help="Operations on enum (sum-type) definitions.")
provider_app = typer.Typer(no_args_is_help=True, help="Inspect registered claim providers.")
equiv_app = typer.Typer(
    no_args_is_help=True,
    help="Operations on Equivalence claims (program-level, between layers).",
)
# Bare `quod ingest` runs the [[ingest.entry]] array; subcommands like
# `quod ingest c <path>` are ad-hoc, single-source forms.
ingest_app = typer.Typer(invoke_without_command=True, help="Ingest source code into the project's program.")

app.add_typer(fn_app, name="fn")
app.add_typer(claim_app, name="claim")
app.add_typer(stmt_app, name="stmt")
app.add_typer(extern_app, name="extern")
extern_app.add_typer(extern_claim_app, name="claim")
app.add_typer(note_app, name="note")
app.add_typer(const_app, name="const")
app.add_typer(struct_app, name="struct")
app.add_typer(enum_app, name="enum")
app.add_typer(provider_app, name="provider")
app.add_typer(ingest_app, name="ingest")
app.add_typer(equiv_app, name="equiv")


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
    file: Path | None = typer.Option(
        None, "--file", "-f",
        help=(
            "Operate on this program.json file directly, bypassing "
            "quod.toml. Useful for inspecting standalone modules "
            "(e.g. stdlib files in src/quod/stdlib/). Mutually "
            "exclusive with --config / --program; build / run still "
            "require a quod.toml since they need bin entries."
        ),
    ),
    no_color: bool = typer.Option(
        False, "--no-color",
        help="Disable ANSI color even on a TTY. NO_COLOR env var also works.",
    ),
) -> None:
    if file is not None and program is not None:
        typer.echo("error: --file and --program are mutually exclusive", err=True)
        raise typer.Exit(2)
    _state["config_path"] = config
    _state["program_name"] = program
    _state["file_path"] = file
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
    from quod.config import starter_toml
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
            prepared = lower_mod.prepare_program(program_obj)
            module = lower_mod.lower(prepared)
            parsed = lower_mod.parse_and_verify(module)
        except ValidationError_ as e:
            typer.echo(f"[{prog.name}] FAIL ({len(e.diagnostics)} errors):")
            for d in e.diagnostics:
                typer.echo(f"  {d.format()}")
            failures += 1
            continue
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
