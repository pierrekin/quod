"""Merge an ingested Program into an existing one.

Ingest is purely additive — it never removes nodes. Each ingest produces
some functions, externs, structs, enums, string constants, and possibly
imports; those get folded into the existing program.json.

Merge rule, per collection: name-keyed, new-wins. Anything in `existing`
whose name doesn't appear in `new` is preserved untouched. Anything in
`new` overwrites the entry of the same name in `existing`.

**Annotation survival.** When a function (or extern) appears in both
sides, the new structure (body / params / return_type) wins, but
existing claims are reconciled into the merged result rather than
silently dropped. A claim survives if its target still exists in the
new shape — for param-scoped claims, "param exists by name and is
still an int type"; for return-scoped, "return is still an int type".
Claims whose target is gone are dropped with a warning. This honors
the design rule from `.scratch/c-ingest/00-overview.md` ("Survival
across re-lifts") that user-authored work shouldn't disappear when
the source is touched in a way that doesn't actually invalidate the
claim.

`merge_program` returns `(merged: Program, warnings: tuple[str, ...])`.
Callers print the warnings; `merge.py` stays UI-agnostic.

Imports are unioned (order-preserving: existing first, then new entries
not already present).

Determinism: re-running ingest on an unchanged source produces an
identical Program, so merging is a no-op. Renaming a function in the
source leaves the old name as an orphan in the merged program — cleanup
is the user's responsibility (or a future `quod prune` command).
"""

from __future__ import annotations

from quod.model import (
    Claim,
    ExternFunction,
    Function,
    I1Type,
    I8Type,
    I16Type,
    I32Type,
    I64Type,
    IsizeType,
    Program,
    U8Type,
    U16Type,
    U32Type,
    U64Type,
    UsizeType,
    claim_param,
)


# Int types valid for claims (param- or return-scoped). Mirrors the
# tuple in ExternFunction's `_check_claims_supported` validator.
_INT_TYPES_FOR_CLAIMS = (
    I1Type, I8Type, I16Type, I32Type, I64Type,
    U8Type, U16Type, U32Type, U64Type,
    IsizeType, UsizeType,
)


def _by_name(items, key: str = "name"):
    """Build an order-preserving dict keyed by `name`. Later entries
    overwrite earlier ones — caller controls precedence by passing
    existing first, then new."""
    out: dict[str, object] = {}
    for it in items:
        out[getattr(it, key)] = it
    return out


def _reconcile_function_claims(
    existing: Function, new: Function, *, qualifier: str = "",
) -> tuple[tuple[Claim, ...], list[str]]:
    """Decide which of `existing`'s claims survive in `new`'s shape.

    Returns `(combined_claims, warnings)`. New's claims are kept
    as-is; existing's claims are added if their target survives and
    they aren't already present (same kind + same target). Surviving
    rules:

      - Param-scoped claim (`non_negative`, `int_range`): the new
        function still has a param of that name, and the param's
        type is still an int.
      - Return-scoped claim (`return_in_range`): the new function's
        return_type is still an int.

    `qualifier` prefixes the warning text — `"function "` or
    `"structured function "` — so the source-of-truth is clear when a
    function appears in both `Program.functions` and
    `Program.structured_functions`.
    """
    new_params_by_name = {p.name: p for p in new.params}
    new_keys = {(c.kind, claim_param(c)) for c in new.claims}

    surviving: list[Claim] = list(new.claims)
    warnings: list[str] = []
    for c in existing.claims:
        target = claim_param(c)
        key = (c.kind, target)
        if key in new_keys:
            # New already has a same-kind/same-target claim. New wins
            # on collision (the bounds/regime may differ but that's
            # the user's intent on the new side).
            continue
        if target is not None:
            # param-scoped
            new_p = new_params_by_name.get(target)
            if new_p is None:
                warnings.append(
                    f"merge: claim {c.kind} on {qualifier}{existing.name}.{target} "
                    f"dropped — param removed in new"
                )
                continue
            if not isinstance(new_p.type, _INT_TYPES_FOR_CLAIMS):
                warnings.append(
                    f"merge: claim {c.kind} on {qualifier}{existing.name}.{target} "
                    f"dropped — param retyped to {new_p.type.kind!r} "
                    f"(claim requires an int type)"
                )
                continue
        else:
            # return-scoped
            if not isinstance(new.return_type, _INT_TYPES_FOR_CLAIMS):
                warnings.append(
                    f"merge: claim {c.kind} on {qualifier}{existing.name} return "
                    f"dropped — return retyped to {new.return_type.kind!r} "
                    f"(claim requires an int return type)"
                )
                continue
        surviving.append(c)
    return tuple(surviving), warnings


def _reconcile_extern_claims(
    existing: ExternFunction, new: ExternFunction,
) -> tuple[tuple[Claim, ...], list[str]]:
    """Same shape as `_reconcile_function_claims` but for externs.
    ExternFunctions today only carry return-scoped claims (their
    params are positional, not named) so the param-scoped branch is
    unreachable; the model's own validator enforces that."""
    new_keys = {c.kind for c in new.claims}
    surviving: list[Claim] = list(new.claims)
    warnings: list[str] = []
    for c in existing.claims:
        if c.kind in new_keys:
            continue
        if not isinstance(new.return_type, _INT_TYPES_FOR_CLAIMS):
            warnings.append(
                f"merge: claim {c.kind} on extern {existing.name} return "
                f"dropped — return retyped to {new.return_type.kind!r} "
                f"(claim requires an int return type)"
            )
            continue
        surviving.append(c)
    return tuple(surviving), warnings


def _merge_named_collection(
    existing_items, new_items,
    reconcile,  # callable(existing_item, new_item) -> (merged_item, warnings)
):
    """Generic name-keyed merge: existing wins where new is absent;
    where both sides have the same name, `reconcile` decides what the
    merged entry looks like and surfaces any warnings."""
    existing_by_name = _by_name(existing_items)
    new_by_name = _by_name(new_items)

    merged: dict[str, object] = {}
    warnings: list[str] = []

    # Walk existing first so the merged dict's iteration order
    # preserves existing's order for unchanged entries.
    for name, e_item in existing_by_name.items():
        if name not in new_by_name:
            merged[name] = e_item
        else:
            merged_item, w = reconcile(e_item, new_by_name[name])
            merged[name] = merged_item
            warnings.extend(w)

    # New-only entries land at the end.
    for name, n_item in new_by_name.items():
        if name not in existing_by_name:
            merged[name] = n_item

    return merged, warnings


def merge_program(
    existing: Program, new: Program,
) -> tuple[Program, tuple[str, ...]]:
    """Fold `new` into `existing`. New entries win on name collision,
    but existing claims survive into the merged result when their
    target node is still present and compatible.

    Returns `(merged_program, warnings)`. Warnings are one-line
    human-readable strings describing claims that were dropped during
    reconciliation; callers print them as they see fit.

    Staged-lift fields (source_units, structured_functions, edges,
    equivalences) merge additively:
      - source_units: keyed by source_path (one CUnit per ingested file).
      - structured_functions: name-keyed in parallel with `functions`,
        with the same claim-reconciliation as `functions`.
      - edges and equivalences: concatenated, deduplicated by full
        structural equality (same source/target ⇒ same edge).
    """
    warnings: list[str] = []

    functions, w = _merge_named_collection(
        existing.functions, new.functions,
        reconcile=lambda e, n: _reconcile_function_pair(e, n, qualifier=""),
    )
    warnings.extend(w)

    structured, w = _merge_named_collection(
        existing.structured_functions, new.structured_functions,
        reconcile=lambda e, n: _reconcile_function_pair(e, n, qualifier="structured "),
    )
    warnings.extend(w)

    externs, w = _merge_named_collection(
        existing.externs, new.externs,
        reconcile=_reconcile_extern_pair,
    )
    warnings.extend(w)

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

    merged = Program(
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
    return merged, tuple(warnings)


def _reconcile_function_pair(
    existing: Function, new: Function, *, qualifier: str,
) -> tuple[Function, list[str]]:
    """Take new's structure, reconcile claims from existing into it."""
    claims, warnings = _reconcile_function_claims(existing, new, qualifier=qualifier)
    if claims == new.claims:
        # No claims survived from existing — and new's claims are
        # what they were. Avoid the model_copy churn.
        return new, warnings
    return new.model_copy(update={"claims": claims}), warnings


def _reconcile_extern_pair(
    existing: ExternFunction, new: ExternFunction,
) -> tuple[ExternFunction, list[str]]:
    claims, warnings = _reconcile_extern_claims(existing, new)
    if claims == new.claims:
        return new, warnings
    return new.model_copy(update={"claims": claims}), warnings
