"""Merge an ingested Program into an existing one.

Ingest is purely additive — it never removes nodes. Each ingest produces
some functions, externs, structs, enums, string constants, and possibly
imports; those get folded into the existing program.json.

Merge rule, per collection: name-keyed, new-wins. Anything in `existing`
whose name doesn't appear in `new` is preserved untouched. Anything in
`new` overwrites the entry of the same name in `existing`.

Imports are unioned (order-preserving: existing first, then new entries
not already present).

Determinism: re-running ingest on an unchanged source produces an
identical Program, so merging is a no-op. Renaming a function in the
source leaves the old name as an orphan in the merged program — cleanup
is the user's responsibility (or a future `quod prune` command).
"""

from __future__ import annotations

from quod.model import Program


def _by_name(items, key: str = "name"):
    """Build an order-preserving dict keyed by `name`. Later entries
    overwrite earlier ones — caller controls precedence by passing
    existing first, then new."""
    out: dict[str, object] = {}
    for it in items:
        out[getattr(it, key)] = it
    return out


def merge_program(existing: Program, new: Program) -> Program:
    """Fold `new` into `existing`. New entries win on name collision.

    See module docstring for the full rule. Returns a fresh Program;
    neither input is mutated.
    """
    functions = _by_name(existing.functions) | _by_name(new.functions)
    externs = _by_name(existing.externs) | _by_name(new.externs)
    constants = _by_name(existing.constants) | _by_name(new.constants)
    structs = _by_name(existing.structs) | _by_name(new.structs)
    enums = _by_name(existing.enums) | _by_name(new.enums)

    seen = set(existing.imports)
    imports = list(existing.imports)
    for imp in new.imports:
        if imp not in seen:
            imports.append(imp)
            seen.add(imp)

    return Program(
        functions=tuple(functions.values()),
        externs=tuple(externs.values()),
        constants=tuple(constants.values()),
        structs=tuple(structs.values()),
        enums=tuple(enums.values()),
        imports=tuple(imports),
    )
