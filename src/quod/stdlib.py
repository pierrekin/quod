"""Stdlib module loading + import resolution.

Programs declare what they want to use via `imports`; this module knows
where to find those modules on disk and how to merge their contents into
a Program before it heads to the lowering pipeline.

The merge is first-wins by name: if a user-declared struct/extern/function
shadows an imported one (same `name`), the user's wins. Signature mismatch
is left for the LLVM verifier — we don't try to type-check across the
boundary at the model layer.

After resolution `program.imports` is cleared, so a resolved program is
indistinguishable from one the user wrote flat. Don't `save_program` on
a resolved program — it would inline the stdlib into the user's source.
"""

from __future__ import annotations

from pathlib import Path

from quod.model import InputProgram, Program


_STDLIB_DIR = Path(__file__).parent / "stdlib"


def stdlib_dir() -> Path:
    """Public for tests / introspection — modules live here as `<name>.json`."""
    return _STDLIB_DIR


class ImportError_(Exception):
    """Module name doesn't resolve, or its file is malformed.

    Trailing underscore avoids shadowing the builtin `ImportError`; raising
    that one would lose the quod-specific error context."""


def resolve_imports(program: Program) -> Program:
    """Walk `program.imports` (and any nested imports declared by those
    modules), fold their structs/externs/functions into `program`, and
    clear `program.imports`. First-wins dedupe by name — user-declared
    items always shadow imports."""
    if not program.imports:
        return program

    structs = list(program.structs)
    externs = list(program.externs)
    functions = list(program.functions)
    seen_struct = {s.name for s in structs}
    seen_extern = {e.name for e in externs}
    seen_fn = {f.name for f in functions}

    queue: list[str] = list(program.imports)
    visited: set[str] = set()
    while queue:
        name = queue.pop(0)
        if name in visited:
            continue
        visited.add(name)
        mod = _load_module(name)
        for nested in mod.imports:
            if nested not in visited:
                queue.append(nested)
        for s in mod.structs:
            if s.name not in seen_struct:
                structs.append(s)
                seen_struct.add(s.name)
        for e in mod.externs:
            if e.name not in seen_extern:
                externs.append(e)
                seen_extern.add(e.name)
        for f in mod.functions:
            if f.name not in seen_fn:
                functions.append(f)
                seen_fn.add(f.name)

    # Construct rather than model_copy so the Program validator runs on the
    # merged result — catches dangling struct refs that were deferred while
    # imports were unresolved.
    return Program(
        constants=program.constants,
        structs=tuple(structs),
        externs=tuple(externs),
        functions=tuple(functions),
        imports=(),
    )


def _load_module(name: str) -> InputProgram:
    """Load a stdlib module by name (e.g. `"std.str"` -> `stdlib/std.str.json`).

    Modules are validated through `InputProgram` — the same gate user files
    pass. They may declare their own `imports`, which the caller queues
    for the recursive resolution."""
    path = _STDLIB_DIR / f"{name}.json"
    if not path.is_file():
        available = sorted(p.stem for p in _STDLIB_DIR.glob("*.json"))
        raise ImportError_(
            f"unknown stdlib module {name!r}; available: "
            f"{', '.join(available) or '(none)'}"
        )
    try:
        return InputProgram.model_validate_json(path.read_text())
    except Exception as exc:
        raise ImportError_(f"failed to load {name!r} from {path}: {exc}") from exc
