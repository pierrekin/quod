"""decompile_lift v2 — parse Ghidra's decompile_text into Layer-A `c.*` nodes.

The complement to `lift_v2.signature_binding`: where signature_binding
gives downstream provers a register↔param map, decompile_lift gives
them a *structured AST* — the same `c.*` shape the C frontend produces
from authored source. With both in hand, a future relational prover
can pair "the source CFn the user wrote" with "the CFn Ghidra
recovered from the binary" by walking matching subtrees, not just by
matching final return values.

Strategy:

  1. Take a `BinFunction.decompile_text` (a C-like string that Ghidra's
     decompiler emitted).
  2. Parse it with libclang via the same in-memory translation-unit API
     the C frontend uses.
  3. Walk the resulting AST with the existing `_translate_function_layer_a`
     translator, which converts libclang cursors into the
     `c.*` Layer-A node family.
  4. Mint a fresh CFn id (`@cfn_lifted_<hash>`) so the lifted CFn
     doesn't collide with a same-named source CFn.
  5. Emit an `Equivalence(bin.fn, lifted_cfn)` justified with
     `DecompileLift(decompile_text_sha256=...)` — pure provenance
     evidence, not a semantic proof.

What the lift refuses to do:

  - No semantic comparison with the source CFn. That belongs in a
    relational provider; here we only land the structure.
  - No filling-in of missing details. If `decompile_text` is empty or
    libclang chokes on Ghidra's output (`undefined4`, `pcVar1`,
    function-pointer noise, etc.), the lift returns None for that
    function. Silent partial lifts are not on offer.
  - No type unification. The lifted CFn carries whatever types the
    decompile_text named, even if Ghidra inferred them wrong. The
    relational prover's job is to spot mismatches.

The lift is structurally weaker than `LiftEquivalence` (which pins a
walk_lift artifact) and weaker than a Z3 witness; the
`DecompileLift` justification carries that weakness explicitly via
its discriminator.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import clang.cindex as cx

from quod.ingest.c.driver import _detect_resource_dir
from quod.ingest.c.helpers import IngestError
from quod.ingest.c.layer_a import _translate_function_layer_a
from quod.model import (
    BinFunction,
    BinUnit,
    CFn,
    DecompileLift,
    Equivalence,
    Program,
    ProvenanceEdge,
)


@dataclass(frozen=True)
class LiftedCFn:
    """Pair of a lifted CFn with its bin-fn endpoint, plus the
    pre-computed `DecompileLift` for the equivalence the caller will
    emit. Returned by `lift_decompile` so the caller can install both
    the CFn (under `BinUnit.lifted_cfns`) and the equivalence
    (under `Program.equivalences`) atomically."""
    bin_fn_id: str
    cfn: CFn
    decompile_text_sha256: str


def lift_decompile(bin_fn: BinFunction) -> LiftedCFn | None:
    """Parse one `BinFunction.decompile_text` and emit a Layer-A CFn.

    Returns None when:
      - `decompile_text` is empty (Ghidra didn't produce a body, or the
        ingester suppressed it for a thunk),
      - libclang refuses the text (function-pointer noise, undefined
        types, syntactic invalid C — common on real-world binaries
        outside the v0 universe),
      - the parsed text doesn't contain a single function definition
        with the binary's demangled name.

    The returned CFn has a stable id derived from the bin.fn's id —
    `@cfn_lifted_{bin_fn.id}` — so it never collides with a source
    CFn of the same `name` (which would be `@cfn_c_{name}`).
    """
    text = bin_fn.decompile_text
    if not text or not text.strip():
        return None

    cursor = _parse_decompile_function(text, bin_fn.demangled_name)
    if cursor is None:
        return None

    try:
        cfn = _translate_function_layer_a(cursor, Path(bin_fn.demangled_name + ".c"))
    except IngestError:
        # The translator refuses constructs outside its subset; that's
        # a normal "lift declined" outcome for binary code, not a bug.
        return None

    cfn = cfn.model_copy(update={"id": f"@cfn_lifted_{bin_fn.id.removeprefix('@')}"})

    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return LiftedCFn(
        bin_fn_id=bin_fn.id,
        cfn=cfn,
        decompile_text_sha256=text_sha,
    )


def _parse_decompile_function(text: str, want_name: str) -> cx.Cursor | None:
    """Parse `text` as C via libclang and return the cursor of the
    function definition named `want_name`, or None if no such
    function definition is present (or the parse failed).

    Uses libclang's `unsaved_files` API to feed the source from memory
    — no temp files, no cleanup, hermetic to the caller. The
    synthetic filename ends in `.c` so libclang picks the C frontend
    rather than C++.
    """
    fake_path = f"/__quod_decompile_lift__/{want_name}.c"
    index = cx.Index.create()
    args: tuple[str, ...] = ("-x", "c")
    resource_dir = _detect_resource_dir()
    if resource_dir is not None:
        args = (*args, f"-resource-dir={resource_dir}")
    try:
        tu = index.parse(
            fake_path,
            args=args,
            unsaved_files=[(fake_path, text)],
        )
    except cx.TranslationUnitLoadError:
        return None
    if not tu:
        return None
    # Hard parse errors abort. Warnings are tolerated — Ghidra's
    # decompile output is C-shaped but not always C-pedantic-clean.
    diags = [d for d in tu.diagnostics if d.severity >= cx.Diagnostic.Error]
    if diags:
        return None

    for cursor in tu.cursor.get_children():
        if cursor.kind != cx.CursorKind.FUNCTION_DECL:
            continue
        if not cursor.is_definition():
            continue
        if cursor.spelling != want_name:
            continue
        return cursor
    return None


def derive_decompile_lifts(
    program: Program,
) -> tuple[Program, tuple[LiftedCFn, ...]]:
    """Run the lift over every `BinFunction` in `program`. Returns a new
    program with the lifted CFns nested under their `BinUnit` plus
    new `Equivalence` claims (justified with `DecompileLift`) and
    `ProvenanceEdge`s from each bin.fn to its lifted CFn.

    The second tuple is the per-function `LiftedCFn` records, useful
    for callers that want to introspect (e.g. tests asserting on
    individual lifts).

    Idempotent: re-running on a program that already carries a
    `decompile_lift`-justified equivalence for a (bin.fn, hash) pair
    skips that pair. A binary re-ingest that produces a different
    decompile_text (different `text_sha256`) is *not* skipped — it
    lands as a fresh equivalence with the new hash, and the merge
    layer's per-justification dedup (already keyed on
    `decompile_text_sha256`) keeps both around for diff visibility.
    """
    if not program.binary_units:
        return program, ()

    existing_lifts: set[tuple[str, str]] = {
        (eq.b_node_id, eq.justification.decompile_text_sha256)
        for eq in program.equivalences
        if eq.justification is not None
        and eq.justification.kind == "decompile_lift"
    }

    new_units: list[BinUnit] = []
    new_eqs: list[Equivalence] = []
    new_edges: list[ProvenanceEdge] = []
    lifted_records: list[LiftedCFn] = []

    for unit in program.binary_units:
        existing_lifted_ids: set[str] = {c.id for c in unit.lifted_cfns}
        unit_new_cfns: list[CFn] = []
        for fn in unit.functions:
            lifted = lift_decompile(fn)
            if lifted is None:
                continue
            lifted_records.append(lifted)
            key = (fn.id, lifted.decompile_text_sha256)
            if key in existing_lifts:
                continue
            existing_lifts.add(key)
            if lifted.cfn.id not in existing_lifted_ids:
                unit_new_cfns.append(lifted.cfn)
                existing_lifted_ids.add(lifted.cfn.id)
            new_eqs.append(Equivalence(
                a_node_id=lifted.cfn.id,
                b_node_id=fn.id,
                regime="axiom",
                justification=DecompileLift(
                    decompile_text_sha256=lifted.decompile_text_sha256,
                ),
            ))
            new_edges.append(ProvenanceEdge(
                source=fn.id, target=lifted.cfn.id,
            ))
        if unit_new_cfns:
            new_units.append(unit.model_copy(update={
                "lifted_cfns": unit.lifted_cfns + tuple(unit_new_cfns),
            }))
        else:
            new_units.append(unit)

    if not new_eqs and not new_edges:
        return program, tuple(lifted_records)

    existing_edge_keys = {(e.source, e.target) for e in program.edges}
    fresh_edges = tuple(
        e for e in new_edges if (e.source, e.target) not in existing_edge_keys
    )

    updated = program.model_copy(update={
        "binary_units": tuple(new_units),
        "equivalences": program.equivalences + tuple(new_eqs),
        "edges": program.edges + fresh_edges,
    })
    return updated, tuple(lifted_records)
