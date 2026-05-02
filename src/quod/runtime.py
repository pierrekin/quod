"""quod's tiny C runtime — compiled on demand into a static archive.
Currently ships the arena allocator (`runtime/quod_arena.c`); future runtime
helpers (string copies, panic abort, etc.) drop in alongside as new .c files.

Why a static `.a` and not a plain `.o`?
  Archive members are pulled in by reference: a binary that never calls
  `quod_arena_alloc` doesn't drag the arena code into its image. A bare .o
  would always be linked in.

Why build per-program (in build_dir) instead of once at install time?
  Cross-compilation. The runtime has to match the user's `--target`; the
  install-time host triple isn't enough.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


_RUNTIME_SOURCE_DIR = Path(__file__).parent / "runtime"

# Bumped whenever the runtime ABI changes — invalidates cached archives in old
# build dirs without forcing a manual `rm -rf build`.
_ARCHIVE_TAG = "v1"


def runtime_sources() -> tuple[Path, ...]:
    """Every .c file shipped with the package — discovered, not hard-coded,
    so adding a new runtime source is a one-step change."""
    return tuple(sorted(_RUNTIME_SOURCE_DIR.glob("*.c")))


def runtime_archive_path(build_dir: Path) -> Path:
    """Where the compiled archive lives. Stable so the linker invocation can
    reference it by `-L<dir> -lquodrt`."""
    return build_dir / "rt" / f"libquodrt-{_ARCHIVE_TAG}.a"


def build_runtime_archive(build_dir: Path, *, target: str | None = None) -> Path:
    """Compile every runtime/*.c into one static archive. Idempotent: returns
    the cached archive when every source's mtime is older than the archive's.

    `clang` is used as the C compiler (matches the linker driver). `ar` is
    used to bundle the resulting objects into the archive.
    """
    sources = runtime_sources()
    if not sources:
        raise RuntimeError(f"no runtime sources found under {_RUNTIME_SOURCE_DIR}")

    archive = runtime_archive_path(build_dir)
    if _archive_is_fresh(archive, sources):
        return archive

    out_dir = archive.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    objects: list[Path] = []
    for src in sources:
        obj = out_dir / f"{src.stem}.o"
        cmd = ["clang", "-c", "-O2", "-fPIC", "-Wall", "-Wextra"]
        if target:
            cmd += ["-target", target]
        cmd += [str(src), "-o", str(obj)]
        subprocess.run(cmd, check=True)
        objects.append(obj)

    if archive.exists():
        archive.unlink()
    subprocess.run(
        ["ar", "rcs", str(archive), *(str(o) for o in objects)],
        check=True,
    )
    return archive


def _archive_is_fresh(archive: Path, sources: tuple[Path, ...]) -> bool:
    if not archive.exists():
        return False
    archive_mtime = archive.stat().st_mtime
    return all(s.stat().st_mtime <= archive_mtime for s in sources)


def link_flags_for_archive(archive: Path) -> list[str]:
    """Linker flags that pull in the runtime archive. We pass it as a path so
    we don't have to fight with `-L` search ordering, then guard it with
    `--whole-archive` only when the user explicitly asks (default: by-reference,
    so unused symbols stay stripped)."""
    return [str(archive)]


def runtime_available() -> bool:
    """True if every external tool we need is on PATH. CLI surfaces this in
    error messages so the user knows whether `clang` or `ar` is the missing
    piece."""
    return shutil.which("clang") is not None and shutil.which("ar") is not None
