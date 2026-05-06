"""Quod version stamping for pinned-claim verification.

`Program.quod_version` records "which build of quod produced the
pinned claims in this Program." Verification compares against the
current build; mismatch means the pins may not reflect what the
running quod would produce, and they need re-checking.

`None` means "no sane version available" and always counts as a
mismatch. The friction is the point: pinned claims should only
verify against a build whose identity we can attest to.

`current_quod_version()` is the source of truth for "what version
is running now." During R&D it returns a git commit hash when one
can be determined cleanly, `None` otherwise. Later it can be
release-based; the rest of the pipeline doesn't care.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import quod
from quod.model import Program


_CACHED: tuple[bool, str | None] = (False, None)


def current_quod_version() -> str | None:
    """Return the running quod's version identifier, or `None` when
    one can't be determined. Cached per-process — quod's identity
    doesn't change underneath a running CLI invocation, and the
    subprocess overhead would otherwise add up across stamping +
    verifying multiple claims.

    Tests can monkey-patch this; see `tests/conftest.py`.
    """
    global _CACHED
    if _CACHED[0]:
        return _CACHED[1]
    value = _compute_quod_version()
    _CACHED = (True, value)
    return value


def _compute_quod_version() -> str | None:
    """Uncached implementation. Runs git in the quod source dir; any
    failure mode collapses to `None`."""
    quod_dir = Path(quod.__file__).resolve().parent  # …/src/quod
    # `git rev-parse HEAD` and `git diff-index --quiet HEAD` both work
    # from any directory inside the repo via cwd=. If the quod source
    # isn't in a git checkout (installed package, shallow clone with no
    # working tree, …) the first call returns nonzero and we bail.
    try:
        # Clean check: nonzero exit means dirty.
        subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD"],
            cwd=quod_dir, check=True, capture_output=True, timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=quod_dir, check=True, capture_output=True, text=True, timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    out = result.stdout.strip()
    return out or None


def reset_cache() -> None:
    """Clear the cached value. Tests use this when monkey-patching
    `_compute_quod_version` so the next `current_quod_version()` call
    reads the new fixture."""
    global _CACHED
    _CACHED = (False, None)


def stamp_quod_version(program: Program) -> Program:
    """Return a copy of `program` with `quod_version` set to whatever
    `current_quod_version()` reports right now. Called by any CLI
    operation that produces or refreshes pinned claims (`prove_lifts`
    post-ingest, `equiv prove --bump` after re-pinning, etc.).
    """
    return program.model_copy(update={"quod_version": current_quod_version()})


def program_has_pinned_claims(program: Program) -> bool:
    """True if `program` has at least one claim with a justification
    — equivalence-level (`Program.equivalences`), function-level
    (`Function.claims`), or extern-level (`ExternFunction.claims`).
    Verification's version check fires only when there's at least one
    pinned claim to gate on."""
    if any(eq.justification is not None for eq in program.equivalences):
        return True
    for fn in program.functions:
        if any(c.justification is not None for c in fn.claims):
            return True
    for fn in program.structured_functions:
        if any(c.justification is not None for c in fn.claims):
            return True
    for ext in program.externs:
        if any(c.justification is not None for c in ext.claims):
            return True
    return False


def check_program_version(program: Program) -> tuple[bool, str]:
    """Return (ok, message). Programs with no pinned claims pass
    trivially. Otherwise the program's `quod_version` must equal the
    current quod version, and neither may be `None`."""
    if not program_has_pinned_claims(program):
        return True, ""
    expected = program.quod_version
    actual = current_quod_version()
    if expected is None:
        return False, (
            "program has pinned claims but no quod_version on record — "
            "re-pin (`quod equiv prove --bump`)"
        )
    if actual is None:
        return False, (
            "running quod has no version available; cannot verify "
            "pinned claims against an unidentifiable build"
        )
    if expected != actual:
        return False, (
            f"program was pinned by quod {expected[:12]}, "
            f"running quod is {actual[:12]} — re-pin "
            f"(`quod equiv prove --bump`)"
        )
    return True, ""
