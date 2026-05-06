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

    Staged-lift fields (source_units, structured_functions, edges,
    equivalences) merge additively:
      - source_units: keyed by source_path (one CUnit per ingested file).
      - structured_functions: name-keyed in parallel with `functions`.
      - edges and equivalences: concatenated, deduplicated by full
        structural equality (same source/target ⇒ same edge).
    """
    functions = _by_name(existing.functions) | _by_name(new.functions)
    structured = (
        _by_name(existing.structured_functions)
        | _by_name(new.structured_functions)
    )
    externs = _by_name(existing.externs) | _by_name(new.externs)
    constants = _by_name(existing.constants) | _by_name(new.constants)
    structs = _by_name(existing.structs) | _by_name(new.structs)
    enums = _by_name(existing.enums) | _by_name(new.enums)
    source_units = (
        _by_name(existing.source_units, key="source_path")
        | _by_name(new.source_units, key="source_path")
    )

    seen = {imp.module for imp in existing.imports}
    imports = list(existing.imports)
    for imp in new.imports:
        if imp.module not in seen:
            imports.append(imp)
            seen.add(imp.module)

    # Edges and equivalences merge by structural identity. Re-ingesting
    # the same source produces identical edges/equivalences, so dedup
    # keeps the merged program byte-stable across re-ingests.
    edges_seen: set = set()
    edges = []
    for e in (*existing.edges, *new.edges):
        key = (e.source, e.target)
        if key not in edges_seen:
            edges.append(e)
            edges_seen.add(key)

    eq_seen: set = set()
    equivalences = []
    for eq in (*existing.equivalences, *new.equivalences):
        # Key on (a, b, justification kind, rule/path/hash) so two
        # claims with different justifications don't collapse, but
        # re-ingest of the same source does dedup.
        j = eq.justification
        if j is None:
            jkey = ("none",)
        elif j.kind == "family_lowering":
            jkey = ("family_lowering", j.rule_name, j.artifact_hash)
        elif j.kind == "lift_equivalence":
            jkey = ("lift_equivalence", j.artifact_path, j.artifact_hash)
        elif j.kind == "manual":
            jkey = ("manual", j.signed_by)
        elif j.kind == "z3":
            jkey = ("z3", j.artifact_hash, j.body_smt_hash)
        elif j.kind == "derived":
            jkey = ("derived", j.analysis)
        else:
            jkey = (j.kind,)
        key = (eq.a_node_id, eq.b_node_id, eq.regime, jkey)
        if key not in eq_seen:
            equivalences.append(eq)
            eq_seen.add(key)

    return Program(
        functions=tuple(functions.values()),
        structured_functions=tuple(structured.values()),
        externs=tuple(externs.values()),
        constants=tuple(constants.values()),
        structs=tuple(structs.values()),
        enums=tuple(enums.values()),
        imports=tuple(imports),
        source_units=tuple(source_units.values()),
        edges=tuple(edges),
        equivalences=tuple(equivalences),
    )
