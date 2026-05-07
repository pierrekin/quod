"""Ingest sub-app — bare `quod ingest` and the `quod ingest c <path>` form.

Includes the lift-stamping helper that runs the A→B lift checker after
each ingest and re-pins the equivalence claims as witnesses.
"""

from __future__ import annotations

from pathlib import Path

import typer

from quod.cli.app import ingest_app
from quod.cli.state import _cfg, _cfg_path, _exclusive_lock, _path
from quod.ingest import IngestError, ingest_c
from quod.ingest.binary import BinaryIngestError, ingest_binary
from quod.merge import merge_program
from quod.model import Program, load_program, save_program


def _empty_program() -> Program:
    return Program()


def _load_or_init_program(path: Path) -> Program:
    """Read program.json if present; otherwise return an empty Program. Used
    by the ingest path — the project must exist (quod.toml must be there),
    but program.json may not yet exist if the user has only just `quod init`'d
    and never written to it.
    """
    if path.exists():
        return load_program(path)
    return _empty_program()


def _resolve_ingest_args(
    cfg, entry, *, source_path: Path,
) -> tuple[str, ...]:
    """Resolve clang_args for an [[ingest.entry]] by following its profile
    reference (or returning the entry's inline args). The CLI ad-hoc form
    bypasses this and passes args directly."""
    if entry.profile is not None:
        prof = cfg.ingest_profiles.get(entry.profile)
        if prof is None:
            typer.echo(
                f"error: [[ingest.entry]] for {source_path} references unknown "
                f"profile {entry.profile!r}", err=True,
            )
            raise typer.Exit(1)
        return prof.clang_args
    return entry.clang_args


def _run_one_c_ingest(
    source: Path, *, clang_args: tuple[str, ...], string_prefix: str | None = None,
) -> Program:
    try:
        return ingest_c(source, clang_args=clang_args, string_prefix=string_prefix)
    except IngestError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)


def _run_one_binary_ingest(
    source: Path, *, base_program: Program,
) -> Program:
    """Drive `ghidra-analyzeHeadless` on `source` and return the program
    with the new `BinUnit` appended (and any seeded equivalences merged).
    Unlike `ingest_c`, the binary ingester *extends* an existing program
    rather than producing a fresh one to merge — equivalence seeding
    needs the source-side functions to be visible at ingest time."""
    try:
        return ingest_binary(source, program=base_program)
    except BinaryIngestError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)


def _prove_and_stamp_lifts(program: Program, cfg) -> Program:
    """Run the A→B lift checker for every (CFn, layer-B Function)
    pair the ingester produced, write the trace artifacts under
    `<config.root>/proofs/lift/`, and replace each manual A~B
    `Equivalence` with a witness-regime `LiftEquivalence` claim.
    Stamps the program with the current quod version after pinning —
    see `quod.version.stamp_quod_version`."""
    from quod.lift_check import LiftCheckError, prove_lifts
    from quod.version import stamp_quod_version
    write_dir = cfg.resolve(cfg.proofs_dir) / "lift"
    # rel_prefix mirrors the cfg.proofs_dir name so equiv verify can
    # resolve the artifact relative to cfg.root.
    rel_prefix = f"{cfg.proofs_dir}/lift" if not Path(cfg.proofs_dir).is_absolute() else "proofs/lift"
    try:
        program = prove_lifts(
            program,
            write_dir=write_dir,
            rel_prefix=rel_prefix,
            write=True,
        )
    except LiftCheckError as e:
        typer.echo(f"error: lift check failed: {e}", err=True)
        raise typer.Exit(1)
    return stamp_quod_version(program)


@ingest_app.callback(invoke_without_command=True)
def ingest_callback(ctx: typer.Context) -> None:
    """Replay every [[ingest.entry]] declared in quod.toml, merging each
    result into the project's program.json. No-op (deterministic) if every
    source is unchanged. Re-running after a source edit overwrites by name;
    nodes from a previous ingest that the new run no longer produces stay
    behind as orphans (cleanup is out of scope for ingest)."""
    if ctx.invoked_subcommand is not None:
        return

    cfg = _cfg()
    program_path = _path()
    if not cfg.ingests:
        typer.echo(
            f"error: no [[ingest.entry]] declared in {_cfg_path()} — either "
            f"add entries or use `quod ingest c <source>` for an ad-hoc ingest",
            err=True,
        )
        raise typer.Exit(1)

    with _exclusive_lock():
        program = _load_or_init_program(program_path)

        for entry in cfg.ingests:
            source = cfg.resolve(entry.source)
            if not source.exists():
                typer.echo(f"error: ingest source {source} does not exist", err=True)
                raise typer.Exit(1)
            if entry.kind == "c-file":
                clang_args = _resolve_ingest_args(cfg, entry, source_path=source)
                ingested = _run_one_c_ingest(source, clang_args=clang_args)
                program, warnings = merge_program(program, ingested)
                for w in warnings:
                    typer.echo(f"warning: {w}", err=True)
                typer.echo(
                    f"ingested {source} ({len(ingested.functions)} function(s))"
                )
            elif entry.kind == "binary":
                program = _run_one_binary_ingest(source, base_program=program)
                new_unit = program.binary_units[-1]
                typer.echo(
                    f"ingested {source} ({len(new_unit.functions)} bin.fn(s), "
                    f"sha256 {new_unit.sha256[:12]}…)"
                )
            else:
                typer.echo(
                    f"error: [[ingest.entry]] kind {entry.kind!r} not supported "
                    f"(supported kinds: 'c-file', 'binary')", err=True,
                )
                raise typer.Exit(1)

        program = _prove_and_stamp_lifts(program, cfg)
        save_program(program, program_path)
    typer.echo(f"wrote {program_path}")


@ingest_app.command(
    "c",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def ingest_c_cmd(
    ctx: typer.Context,
    source: Path = typer.Argument(..., help="C source file to ingest."),
    profile: str | None = typer.Option(
        None, "--profile",
        help="Named [ingest.profile.<name>] to apply (mutually exclusive with -- args).",
    ),
) -> None:
    """Ad-hoc one-off ingest of a single C file into the project's program.json.

    Does not modify quod.toml; for repeatable ingests, declare an
    [[ingest.entry]] there. Extra args after `--` are forwarded to clang
    (e.g. `quod ingest c foo.c -- -std=c89`).
    """
    if not source.exists():
        typer.echo(f"error: {source} does not exist", err=True)
        raise typer.Exit(1)
    if source.suffix != ".c":
        typer.echo(
            f"error: expected a .c file (got {source.suffix!r}); "
            f"only kind 'c-file' is supported", err=True,
        )
        raise typer.Exit(2)

    extra_clang = tuple(ctx.args)
    if profile is not None and extra_clang:
        typer.echo(
            "error: --profile and extra clang args (after `--`) are mutually exclusive",
            err=True,
        )
        raise typer.Exit(2)

    cfg = _cfg()
    if profile is not None:
        prof = cfg.ingest_profiles.get(profile)
        if prof is None:
            typer.echo(
                f"error: unknown profile {profile!r}; defined: "
                f"{sorted(cfg.ingest_profiles)}", err=True,
            )
            raise typer.Exit(1)
        clang_args = prof.clang_args
    else:
        clang_args = extra_clang

    program_path = _path()
    with _exclusive_lock():
        program = _load_or_init_program(program_path)
        ingested = _run_one_c_ingest(source, clang_args=clang_args)
        program, warnings = merge_program(program, ingested)
        for w in warnings:
            typer.echo(f"warning: {w}", err=True)
        program = _prove_and_stamp_lifts(program, cfg)
        save_program(program, program_path)

    typer.echo(
        f"ingested {source} ({len(ingested.functions)} function(s)) into {program_path}"
    )


@ingest_app.command("binary")
def ingest_binary_cmd(
    source: Path = typer.Argument(
        ..., help="Binary artifact to ingest (`.so` / `.exe` / `.o`).",
    ),
    keep_dump: Path | None = typer.Option(
        None, "--keep-dump",
        help="Write the raw Ghidra JSON dump to this path (in addition "
             "to parsing it). Useful for diagnosing parser issues.",
    ),
) -> None:
    """Ad-hoc one-off binary ingest. Drives Ghidra in-process via
    PyGhidra, parses the JSON dump the exporter produces, builds
    layer-A `bin.*` nodes, and seeds equivalences against any matching
    source functions already in the program.

    Does not modify quod.toml; for repeatable ingests, declare an
    `[[ingest.entry]] kind = "binary"` there. Ghidra analysis can take
    minutes on a real library; the JVM stays warm across ingests in
    the same process so subsequent calls skip the ~5s startup."""
    if not source.exists():
        typer.echo(f"error: {source} does not exist", err=True)
        raise typer.Exit(1)

    program_path = _path()
    with _exclusive_lock():
        program = _load_or_init_program(program_path)
        try:
            program = ingest_binary(
                source,
                program=program,
                keep_dump=keep_dump,
            )
        except BinaryIngestError as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        save_program(program, program_path)

    new_unit = program.binary_units[-1]
    typer.echo(
        f"ingested {source} ({len(new_unit.functions)} bin.fn(s), "
        f"sha256 {new_unit.sha256[:12]}…) into {program_path}"
    )
