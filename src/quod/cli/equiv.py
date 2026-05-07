"""equiv sub-app — operations on program-level Equivalence claims."""

from __future__ import annotations

from pathlib import Path

import typer

from quod.cli.app import equiv_app
from quod.cli.output import _sha256_of_file, _theme
from quod.cli.state import _cfg, _load
from quod.model import (
    DerivedJustification,
    Equivalence,
    FamilyLowering,
    LiftEquivalence,
    ManualJustification,
    Program,
    save_program,
)
from quod.predicate.proof import Z3NotInstalled, run_z3_on_file
from quod.render import Span, paint


@equiv_app.command("prove")
def equiv_prove(
    function: str | None = typer.Argument(
        None,
        help="Optional function name to filter chains. Without it, every "
             "Equivalence claim in the program is reported.",
    ),
    bump: bool = typer.Option(
        False, "--bump",
        help="Upgrade unproven A~B claims by running the lift-check, and "
             "re-pin stale lift artifacts. Saves program.json. Has no "
             "effect on FamilyLowering claims (those pin package-shipped "
             "proof artifacts that don't drift).",
    ),
    write_proofs: bool = typer.Option(
        True, "--write-proofs/--no-write-proofs",
        help="When --bump runs, write the lift-check artifact to disk "
             "under proofs_dir/lift/. Disable for dry-run inspection.",
    ),
) -> None:
    """Walk Equivalence chains and report per-claim status.

    Three categories per claim:

      - **witnessed-current**: regime=witness, artifact resolves, hash
        matches (and Z3 returns unsat for FamilyLowering).
      - **witnessed-stale**: regime=witness, but verification fails
        (hash mismatch, missing artifact, or — only for LiftEquivalence
        — the in-memory walk would produce different bytes).
      - **unproven**: regime=axiom, with or without a justification.

    With --bump, unproven A~B claims (ManualJustification from the
    ingester) are upgraded by running the lift-check, and stale
    LiftEquivalence claims are re-pinned. FamilyLowering claims are
    not touched — those pin package-shipped proof artifacts, and
    "staleness" there means the rule's source changed, which is a
    different problem (drift detection — see `02-next.md`).
    """
    from quod.cli.state import _path
    from quod.lift_check import LiftCheckError, prove_lifts
    from quod.version import check_program_version, stamp_quod_version

    cfg = _cfg()
    program = _load()
    theme = _theme()

    if bump:
        write_dir = cfg.resolve(cfg.proofs_dir) / "lift"
        rel_prefix = (
            f"{cfg.proofs_dir}/lift"
            if not Path(str(cfg.proofs_dir)).is_absolute()
            else "proofs/lift"
        )
        try:
            program = prove_lifts(
                program,
                write_dir=write_dir,
                rel_prefix=rel_prefix,
                write=write_proofs,
            )
        except LiftCheckError as e:
            typer.echo(f"error: lift check failed: {e}", err=True)
            raise typer.Exit(1)
        program = stamp_quod_version(program)
        save_program(program, _path())

    version_ok, version_msg = check_program_version(program)
    if not version_ok:
        typer.echo(f"error: {version_msg}", err=True)
        raise typer.Exit(1)

    failures = 0
    checked = 0
    for eq in program.equivalences:
        if function is not None and not _equivalence_involves(eq, function):
            continue
        checked += 1
        status, msg = _classify_equivalence(eq, program)
        if status == "witnessed-current":
            badge = Span("ok  ", "ok")
        elif status == "unproven":
            badge = Span("axiom", "warn")
        else:  # witnessed-stale or other failure
            badge = Span("FAIL", "warn")
            failures += 1

        head = f"{eq.a_node_id} ~ {eq.b_node_id}"
        j = eq.justification
        kind = j.kind if j is not None else "(none)"
        typer.echo(paint((
            badge, Span("  ", "ws"),
            Span(head, "fn_name"),
            Span(f"  [{eq.regime}/{kind}]", "punct"),
        ), theme))
        if msg:
            typer.echo(f"      {msg}")

    if checked == 0:
        if function is not None:
            typer.echo(f"(no equivalence claims involving {function!r})")
        else:
            typer.echo("(no equivalence claims)")

    if failures:
        raise typer.Exit(1)


def _equivalence_involves(eq: "Equivalence", function: str) -> bool:
    """True if the function name appears as a substring of either
    endpoint ID. Cheap heuristic — works for the `@fn_c_sum`-style
    IDs the ingester produces; refine if functions get IDs that
    don't embed the name."""
    return function in eq.a_node_id or function in eq.b_node_id


def _classify_equivalence(
    eq: "Equivalence", program: Program,
) -> tuple[str, str]:
    """Return (status, message) for one Equivalence claim. Status
    is one of `witnessed-current`, `witnessed-stale`, `unproven`.

    For LiftEquivalence, also detects the case where the in-memory
    walk would produce different bytes than the pinned artifact —
    that's witnessed-stale (a hash-match-but-out-of-date isn't
    possible since hash IS the match criterion, but the in-memory
    walk gives us a definitive "what should the hash be now").
    """
    if eq.regime == "axiom":
        return "unproven", ""
    ok, msg = _verify_equivalence_justification(eq)
    if not ok:
        return "witnessed-stale", msg

    # For LiftEquivalence specifically, also re-walk the in-memory
    # tree and confirm the bytes still hash to the pinned value.
    # Catches "someone hand-edited the program but forgot to re-run
    # `equiv prove --bump`" — verify alone misses this when the
    # artifact file on disk is unchanged.
    if isinstance(eq.justification, LiftEquivalence):
        from quod.lift_check import lift_check_hash
        cfn = next(
            (c for u in program.source_units for c in u.functions if c.id == eq.a_node_id),
            None,
        )
        fn = next((f for f in program.structured_functions if f.id == eq.b_node_id), None)
        if cfn is not None and fn is not None:
            try:
                live_hash = lift_check_hash(cfn, fn)
            except Exception as e:
                return "witnessed-stale", f"in-memory walk failed: {e}"
            if live_hash != eq.justification.artifact_hash:
                return "witnessed-stale", (
                    f"in-memory walk would hash to {live_hash[:12]}, "
                    f"pinned is {eq.justification.artifact_hash[:12]} — "
                    f"run `quod equiv prove --bump` to re-pin"
                )

    return "witnessed-current", ""


@equiv_app.command("verify")
def equiv_verify() -> None:
    """Re-check evidence on Equivalence claims.

    Walks `Program.equivalences` and validates each claim's
    justification:

      - `FamilyLowering(rule_name, artifact_path, artifact_hash)` —
        for rules with a pinned proof, the artifact's bytes are
        sha256-checked against the stored hash and Z3 is run on the
        artifact (expected `unsat`). Rules without an artifact
        (e.g. `identity`) verify trivially.
      - `LiftEquivalence(artifact_path, artifact_hash)` — same shape;
        proof artifact lives under the program's resolve_root rather
        than the package's source tree.
      - `ManualJustification` — passes if `signed_by` is non-empty.

    Per-rule artifacts are package-distributed under
    `src/quod/lower/c_family_proofs/`; the `artifact_path` is
    package-relative (rooted at `quod/`) and resolved against the
    installed package's location, so verification works regardless
    of the user's working directory.
    """
    from quod.version import check_program_version

    program = _load()

    version_ok, version_msg = check_program_version(program)
    if not version_ok:
        typer.echo(f"error: {version_msg}", err=True)
        raise typer.Exit(1)

    theme = _theme()
    failures = 0
    checked = 0
    for eq in program.equivalences:
        if eq.justification is None:
            continue
        checked += 1
        ok, msg = _verify_equivalence_justification(eq)
        status_span = Span("ok  ", "ok") if ok else Span("FAIL", "warn")
        head = f"{eq.a_node_id} ~ {eq.b_node_id}"
        typer.echo(paint((
            status_span, Span(" ", "ws"),
            Span(head, "fn_name"),
        ), theme))
        if not ok:
            typer.echo(f"     {msg}")
            failures += 1
    if checked == 0:
        typer.echo("(no equivalence claims with justifications)")
    if failures:
        raise typer.Exit(1)


def _verify_equivalence_justification(eq: Equivalence) -> tuple[bool, str]:
    """Verify the justification on a program-level Equivalence claim.
    Mirrors `_verify_justification` for fn-level claims, but resolves
    package-distributed artifacts (FamilyLowering) against the
    `quod/` package directory rather than the user's resolve_root."""
    j = eq.justification
    if isinstance(j, FamilyLowering):
        # No artifact on this rule yet — claim is asserted (axiom),
        # not witnessed. We pass; it's not stale, just unproven.
        if j.artifact_path is None or j.artifact_hash is None:
            if j.artifact_path is None and j.artifact_hash is None:
                return True, ""
            return False, (
                "FamilyLowering has only one of (artifact_path, "
                "artifact_hash) — both must be set or neither"
            )
        # Resolve package-relative path against the installed package.
        from quod import lower as _lower_pkg
        pkg_root = Path(_lower_pkg.__file__).parent.parent
        full = pkg_root / j.artifact_path
        if not full.exists():
            return False, f"artifact not found: {full}"
        actual = _sha256_of_file(full)
        if actual != j.artifact_hash:
            return False, (
                f"hash mismatch: stored={j.artifact_hash[:12]}, "
                f"file={actual[:12]}"
            )
        try:
            result = run_z3_on_file(full)
        except Z3NotInstalled as e:
            return False, str(e)
        except Exception as e:
            return False, f"z3 invocation failed: {e}"
        if result.status != "unsat":
            return False, f"z3 returned {result.status!r} (expected 'unsat')"
        return True, ""
    if isinstance(j, LiftEquivalence):
        # LiftEquivalence pins a *structural* transcription record
        # (produced by quod.lift_check.prove_lifts). Verification is
        # hash-only: re-walking the layer-A and layer-B subtrees in
        # memory would reproduce the artifact bytes deterministically;
        # the file on disk is for human inspection. We don't invoke
        # Z3 here — the artifact isn't SMT.
        cfg = _cfg()
        full = cfg.root / j.artifact_path
        if not full.exists():
            return False, f"artifact not found: {full}"
        actual = _sha256_of_file(full)
        if actual != j.artifact_hash:
            return False, (
                f"hash mismatch: stored={j.artifact_hash[:12]}, "
                f"file={actual[:12]}"
            )
        return True, ""
    if isinstance(j, ManualJustification):
        if not j.signed_by.strip():
            return False, "manual signed_by is empty"
        return True, ""
    if isinstance(j, DerivedJustification):
        return True, ""
    return False, f"unknown justification kind: {j!r}"
