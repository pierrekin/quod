"""Top-level driver for C ingestion.

Walks a libclang AST once and emits **three parallel subtrees**:

  - Layer A (`Program.source_units`): the original C preserved as quod
    nodes (`CUnit`, `CFn`, `CFor`, `CVarDecl`, …). Inert — no codegen,
    no validation — but addressable by stable IDs so downstream
    analyses can reach the source-form.
  - Layer B (`Program.structured_functions`): the c-like-quod
    transcription. Mostly core nodes, with `c.*` family extensions
    where layer-B carries information the lift hasn't finished
    collapsing (e.g. C `for` becomes `CStyleFor`).
  - Layer C (`Program.functions`): pure core, what `quod.lower`
    consumes. Produced by the c-family lowering pass
    (`lower/c_family.py`) at the end of `ingest_c`.

The subtrees are paired by `ProvenanceEdge`s and `Equivalence` claims
(see `Program.edges`, `Program.equivalences`). The ingester emits A↔B
equivalences with a `ManualJustification` ("the ingester promises a
structural lift"); `quod equiv prove` (or `prove_lifts` directly)
upgrades them to `regime=witness` with a `LiftEquivalence` artifact
under `<proofs_dir>/lift/<fn>.txt`. The B↔C `FamilyLowering`
equivalences carry per-rule SMT proofs from
`src/quod/lower/c_family_proofs/*.smt2`.

The supported C subset covers int- and char-pointer-only programs
with the standard control-flow vocabulary: if / while / do-while /
for (any of init/cond/inc may be absent) / return / break /
continue / switch (no implicit fallthrough). Expressions cover
arithmetic, comparison, short-circuit boolean, bitwise (`& | ^ ~ <<
>>`), unary (`- ! ~`), ternary `? :`, compound assignments
(`+= -= ...`), calls. Locals can be declared without an initializer
(`int x;`); the validator's definite-init analysis refuses any read
that isn't dominated by a write. Constructs not yet in the supported
subset (loads, casts, wider integer types, structs, floats, …) raise
`IngestError` rather than silently lowering with surprising semantics.

Constructs outside the supported subset raise `IngestError` with the
offending source location.

Macros / #include / #ifdef are handled by clang's preprocessor before
we see the AST — we ingest one build configuration of the source. We
filter cursors by source file so headers don't pollute the program.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import clang.cindex as cx

from quod.ingest.c.c_helpers import IngestError
from quod.ingest.c.c_layer_a import _translate_function_layer_a
from quod.ingest.c.c_layer_b import (
    _ProgramState,
    _build_extern_from_decl,
    _refuse,
    _translate_function,
)
from quod.model import (
    CFn,
    CUnit,
    Equivalence,
    ExternFunction,
    Function,
    ManualJustification,
    Program,
    ProvenanceEdge,
)


def _detect_resource_dir() -> str | None:
    """Ask `clang` where its resource directory is.

    The PyPI `libclang` package ships a `libclang.so` but not clang's
    resource headers (stddef.h, stdarg.h, …). Without `-resource-dir`, even
    `#include <stdio.h>` fails because stdio internally includes stddef.h.
    Run once per ingest; ignore failures (caller can override via
    clang_args).
    """
    try:
        out = subprocess.run(
            ["clang", "-print-resource-dir"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


_PREFIX_SAFE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def _default_string_prefix(path: Path) -> str:
    """Sanitize a source path's stem for use as a string-constant prefix.

    Two ingests need different prefixes so their string constants don't
    collide on merge. Stem is short and recognizable; collision-resistant
    enough for practical corpora — if the same basename appears under two
    paths, callers should pass an explicit prefix.
    """
    return "".join(c if c in _PREFIX_SAFE else "_" for c in path.stem)


def ingest_c(
    path: Path, *,
    clang_args: tuple[str, ...] = (),
    string_prefix: str | None = None,
) -> Program:
    """Parse a C file and return a quod Program.

    Anything outside the supported subset raises IngestError with the
    offending source location. Only declarations whose primary location is
    in `path` itself are translated — header-included declarations are
    skipped, but their types/symbols are visible to the parser, so calls
    into stdlib resolve to externs with proper signatures.

    `string_prefix` namespaces auto-generated string-constant names so two
    ingests can be merged into one program.json without colliding on
    `.str.0`. Defaults to a sanitized stem of the source path.
    """
    path = path.resolve()
    if not path.exists():
        raise IngestError(f"{path}: no such file")

    index = cx.Index.create()
    args: tuple[str, ...] = ("-x", "c")
    resource_dir = _detect_resource_dir()
    if resource_dir is not None:
        args = (*args, f"-resource-dir={resource_dir}")
    args = (*args, *clang_args)
    tu = index.parse(str(path), args=args)
    if not tu:
        raise IngestError(f"{path}: clang failed to parse file")

    diags = [d for d in tu.diagnostics if d.severity >= cx.Diagnostic.Error]
    if diags:
        msg = "; ".join(f"{d.location.file}:{d.location.line}: {d.spelling}" for d in diags)
        raise IngestError(f"{path}: parse errors: {msg}")

    if string_prefix is None:
        string_prefix = _default_string_prefix(path)
    state = _ProgramState(string_prefix=string_prefix)
    functions: list[Function] = []
    fn_cursors: list[cx.Cursor] = []
    defined_names: set[str] = set()

    for cursor in tu.cursor.get_children():
        loc_file = cursor.location.file
        if loc_file is None or Path(loc_file.name).resolve() != path:
            continue
        if cursor.kind != cx.CursorKind.FUNCTION_DECL:
            raise _refuse(cursor, f"top-level {cursor.kind.name} not supported (only functions)")
        if not cursor.is_definition():
            continue
        functions.append(_translate_function(cursor, path, state))
        fn_cursors.append(cursor)
        defined_names.add(cursor.spelling)

    # Drop externs that turned out to be locally defined functions (e.g. a
    # call to one ingested function from another doesn't need an extern).
    externs = tuple(e for name, e in state.externs.items() if name not in defined_names)

    # Layer-A pass — best-effort. The layer-A translator covers a narrower
    # C subset than layer B; if translation fails for any function in the
    # file, we emit no layer-A subtree for the whole file — all-or-nothing
    # keeps the corpus less surprising than partial source_units. Layer B
    # is unaffected; the file still compiles via the existing path.
    cfns: list[CFn] = []
    layer_a_failed = False
    for cursor in fn_cursors:
        try:
            cfns.append(_translate_function_layer_a(cursor, path))
        except IngestError:
            layer_a_failed = True
            break

    # Layer-B Functions go into `structured_functions` (the
    # extension-bearing transcription); the canonical core form lives
    # in `Program.functions`, populated next by the c-family lowering
    # pass. `quod.lower`'s contract is unchanged — it consumes the
    # canonical form only.
    if layer_a_failed or not cfns:
        program = Program(
            constants=tuple(state.constants),
            structured_functions=tuple(functions),
            externs=externs,
        )
    else:
        a_to_b_edges = tuple(
            ProvenanceEdge(source=cfn.id, target=fn.id)
            for cfn, fn in zip(cfns, functions)
        )
        # Function-level A→B Equivalence claims mark the transcription.
        # The ingester emits `regime=axiom` with a manual justification
        # (it promises a structural lift); `quod equiv prove` runs the
        # lift-checker and promotes these to `regime=witness` with a
        # `LiftEquivalence` artifact.
        a_to_b_equivalences = tuple(
            Equivalence(
                a_node_id=cfn.id,
                b_node_id=fn.id,
                justification=ManualJustification(
                    signed_by="quod.ingest.c",
                    rationale=(
                        "structural transcription from C source to layer-B "
                        "c-like-quod; no semantic decisions in the lift"
                    ),
                ),
            )
            for cfn, fn in zip(cfns, functions)
        )
        source_units = (CUnit(
            id=f"@cunit_c_{_default_string_prefix(path)}",
            source_path=path.name,
            functions=tuple(cfns),
        ),)
        program = Program(
            constants=tuple(state.constants),
            structured_functions=tuple(functions),
            externs=externs,
            source_units=source_units,
            edges=a_to_b_edges,
            equivalences=a_to_b_equivalences,
        )

    # Run the c-family lowering pass to populate `Program.functions`
    # (layer C, lowerable). This adds B→C ProvenanceEdges and
    # FamilyLowering Equivalence claims to the program. Imported here
    # to avoid the import cycle between quod.lower (which depends on
    # quod.model at module import time) and quod.ingest.c.
    from quod.lower.c_family import lower_c_family
    return lower_c_family(program)


def _parse_translation_unit(path: Path, *, language: str, clang_args: tuple[str, ...]) -> cx.TranslationUnit:
    """Shared libclang entry point. `language` is `c` or `c-header`."""
    if not path.exists():
        raise IngestError(f"{path}: no such file")
    index = cx.Index.create()
    args: tuple[str, ...] = ("-x", language)
    resource_dir = _detect_resource_dir()
    if resource_dir is not None:
        args = (*args, f"-resource-dir={resource_dir}")
    args = (*args, *clang_args)
    tu = index.parse(str(path), args=args)
    if not tu:
        raise IngestError(f"{path}: clang failed to parse file")
    diags = [d for d in tu.diagnostics if d.severity >= cx.Diagnostic.Error]
    if diags:
        msg = "; ".join(f"{d.location.file}:{d.location.line}: {d.spelling}" for d in diags)
        raise IngestError(f"{path}: parse errors: {msg}")
    return tu


def ingest_header(
    path: Path, *, clang_args: tuple[str, ...] = (),
) -> tuple[tuple[ExternFunction, ...], tuple[str, ...]]:
    """Parse a C header and emit ExternFunction declarations.

    Walks every FUNCTION_DECL reachable from the translation unit and
    builds an `ExternFunction` for each whose signature fits the supported
    type system (`int`, `char*`, varargs). Returns:

        (externs_built, names_skipped)

    Names that appeared as function declarations but had unsupported
    signatures (struct params, floats, wider ints, etc.) are returned in
    `names_skipped` so the caller can show a count or list. Symbols
    declared multiple times (e.g. via redeclaration) are deduplicated by
    name; first sighting wins.
    """
    path = path.resolve()
    tu = _parse_translation_unit(path, language="c-header", clang_args=clang_args)

    externs: list[ExternFunction] = []
    skipped: list[str] = []
    seen: set[str] = set()

    for cursor in tu.cursor.walk_preorder():
        if cursor.kind != cx.CursorKind.FUNCTION_DECL:
            continue
        # Headers may contain `static inline` definitions (e.g. from
        # libc's transitive includes). Those have bodies — skip them, we
        # only want pure declarations to expose as externs.
        if cursor.is_definition():
            continue
        name = cursor.spelling
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            ext = _build_extern_from_decl(cursor, cursor)
        except IngestError:
            # Bulk-import path is tolerant: a header full of unsupported
            # signatures shouldn't refuse the whole ingest. Caller gets
            # a tally of what was skipped.
            skipped.append(name)
            continue
        externs.append(ext)

    return tuple(externs), tuple(skipped)
