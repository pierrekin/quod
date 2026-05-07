"""Program — the root container.

Holds every collection (functions, externs, structs, enums, traits,
impls, imports, edges, equivalences, source units, structured
functions) plus version metadata. `Program` is the permissive form
used for in-memory editing; `InputProgram` is the load-time gate that
rejects lattice claims in stored data.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator, model_serializer, model_validator

from quod.model.base import _Node
from quod.model.expressions import StringConstant
from quod.model.layer_a import CUnit
from quod.model.layer_a_bin import BinUnit
from quod.model.relations import Equivalence, Import, ProvenanceEdge
from quod.model.top_level import (
    EnumDef,
    ExternFunction,
    Function,
    StructDef,
    TypeParam,
)
from quod.model.traits import ImplDef, TraitDef
from quod.model.types import StructType


class _ProgramBase(_Node):
    """Shared shape for Program and InputProgram."""
    constants: tuple[StringConstant, ...] = ()
    functions: tuple[Function, ...] = ()
    externs: tuple[ExternFunction, ...] = ()
    structs: tuple[StructDef, ...] = ()
    enums: tuple[EnumDef, ...] = ()
    traits: tuple[TraitDef, ...] = ()
    impls: tuple[ImplDef, ...] = ()
    wirables: tuple[TypeParam, ...] = ()
    imports: tuple[Import, ...] = ()
    # Provenance edges and equivalence claims live at program level
    # (rather than on individual functions) because they're relational
    # — they connect two nodes that can be in different functions or in
    # different language-family layers. Both default to empty; existing
    # programs that don't yet carry provenance round-trip unchanged.
    edges: tuple[ProvenanceEdge, ...] = ()
    equivalences: tuple[Equivalence, ...] = ()
    # Layer-A subtree: original source-language programs preserved as
    # quod nodes (one CUnit per ingested file). Inert — no validation,
    # no codegen — but addressable by stable IDs so cross-layer
    # provenance edges can anchor here.
    source_units: tuple[CUnit, ...] = ()
    # Layer-A subtree (binary side): one BinUnit per ingested .so/.exe/.o.
    # Mirrors `source_units` for the C frontend — same inert-preservation
    # rule (no type recovery into StructDef, no re-parse of decompile
    # text into c.* nodes), but addressable by stable IDs so cross-layer
    # Equivalence claims have a binary-side endpoint to anchor on.
    # Empty for programs that never went through `quod ingest binary`.
    binary_units: tuple[BinUnit, ...] = ()
    # Structured-form functions: per-language extension-bearing
    # transcriptions of the source. For C, these contain `CStyleFor`
    # and other `c.*` family extensions; the c-family lowering pass
    # (lower/c_family.py) reads these and produces the canonical core
    # functions in `Program.functions`. Both forms persist on disk so
    # cross-layer analysis and drift detection (lowering-rule changes)
    # work without re-ingesting from source. Empty for hand-authored
    # core programs that didn't go through a source-language ingest.
    #
    # The name is deliberately layer-neutral — the data model will
    # evolve as more language families land, and "structured" captures
    # the durable property (preserves source-language structure via
    # extension nodes) without committing to "layer B" terminology.
    structured_functions: tuple[Function, ...] = ()
    # Version stamp: which build of quod produced the pinned claims in
    # this Program. During R&D this is the commit hash of the quod
    # source tree at pin time; later it can be a release tag. `None`
    # means "no version on record" — under the strict policy this is
    # always treated as a mismatch by `quod equiv verify` etc., so a
    # `None` Program with pinned claims fails verification until
    # re-pinned from a clean checkout.
    #
    # Set by `quod.version.stamp_quod_version` at any operation that
    # produces or refreshes pinned claims (ingest's `prove_lifts`,
    # `equiv prove --bump`, etc.). Pinning from a dirty quod tree
    # captures `None` deliberately — only clean checkouts produce
    # verifiable pins.
    quod_version: str | None = None

    @model_serializer(mode="wrap")
    def _drop_empty_collections(self, handler, info):
        data = handler(self)
        if not self.structs:
            data.pop("structs", None)
        if not self.enums:
            data.pop("enums", None)
        if not self.traits:
            data.pop("traits", None)
        if not self.impls:
            data.pop("impls", None)
        if not self.wirables:
            data.pop("wirables", None)
        if not self.imports:
            data.pop("imports", None)
        if not self.edges:
            data.pop("edges", None)
        if not self.equivalences:
            data.pop("equivalences", None)
        if not self.source_units:
            data.pop("source_units", None)
        if not self.binary_units:
            data.pop("binary_units", None)
        if not self.structured_functions:
            data.pop("structured_functions", None)
        if self.quod_version is None:
            data.pop("quod_version", None)
        return data

    @field_validator("imports", mode="before")
    @classmethod
    def _coerce_imports(cls, raw):
        """Allow `["alloc.list", {"module": "alloc.list", "wire": [...]}]`
        — bare strings become `{module: <s>}`. Returns a tuple to keep
        strict-mode validation happy (the field is `tuple[Import, ...]`
        and `strict=True` doesn't auto-coerce list→tuple).
        """
        out = []
        for item in raw:
            if isinstance(item, str):
                out.append({"module": item})
            else:
                out.append(item)
        return tuple(out)

    @field_validator("imports")
    @classmethod
    def _validate_import_names(cls, imports: tuple["Import", ...]) -> tuple["Import", ...]:
        # Sanitize: only allow [A-Za-z0-9_.] so module names can't
        # path-traverse to disk locations outside the stdlib directory.
        # Names map to file paths via `stdlib/<name>.json` — no slashes,
        # no leading/trailing dots, no empty segments.
        seen: set[str] = set()
        for imp in imports:
            n = imp.module
            if not n or not all(c.isalnum() or c in "._" for c in n):
                raise ValueError(
                    f"invalid import name {n!r}: must match [A-Za-z0-9_.] only"
                )
            if n.startswith(".") or n.endswith(".") or ".." in n:
                raise ValueError(
                    f"invalid import name {n!r}: no leading/trailing/empty segments"
                )
            if n in seen:
                raise ValueError(f"duplicate import {n!r}")
            seen.add(n)
        return imports


def _validate_structs(program: "_ProgramBase") -> None:
    """Program-wide struct + enum *structural* sanity. Runs on both
    Program and InputProgram at construction time (Pydantic
    model_validator).

    Owns only the cheap, definition-local checks that don't need
    whole-program context:
    - Struct names are unique.
    - No duplicate field names within a struct.
    - Enum names are unique and don't collide with structs.
    - No duplicate variants per enum, no duplicate fields per variant.
    - Every enum has at least one variant.
    - No struct contains itself by value (direct or transitive).

    Use-site checks (refs resolve, exhaustive matches, struct/enum_init
    field correctness) live in `quod.validate` — they need the resolved
    program (post-import-resolution) and benefit from collecting all
    diagnostics rather than failing fast.
    """
    seen_names: set[str] = set()
    for sd in program.structs:
        if sd.name in seen_names:
            raise ValueError(f"duplicate struct definition {sd.name!r}")
        seen_names.add(sd.name)
        field_names: set[str] = set()
        for f in sd.fields:
            if f.name in field_names:
                raise ValueError(
                    f"struct {sd.name!r} has duplicate field {f.name!r}"
                )
            field_names.add(f.name)

    by_name: dict[str, StructDef] = {sd.name: sd for sd in program.structs}
    enums_by_name: dict[str, EnumDef] = {}
    for ed in program.enums:
        if ed.name in enums_by_name:
            raise ValueError(f"duplicate enum definition {ed.name!r}")
        if ed.name in by_name:
            raise ValueError(
                f"enum name {ed.name!r} collides with a struct of the same name"
            )
        enums_by_name[ed.name] = ed
        seen_variants: set[str] = set()
        for v in ed.variants:
            if v.name in seen_variants:
                raise ValueError(
                    f"enum {ed.name!r} has duplicate variant {v.name!r}"
                )
            seen_variants.add(v.name)
            seen_fields: set[str] = set()
            for f in v.fields:
                if f.name in seen_fields:
                    raise ValueError(
                        f"variant {ed.name!r}::{v.name} has duplicate field {f.name!r}"
                    )
                seen_fields.add(f.name)
        if not ed.variants:
            raise ValueError(f"enum {ed.name!r} has no variants")

    # Reject by-value cycles. Walk each struct's transitive struct-typed
    # fields; a path that revisits the start is a cycle.
    for sd in program.structs:
        _check_no_struct_cycle(sd.name, by_name)


def _check_no_struct_cycle(start: str, by_name: dict[str, "StructDef"]) -> None:
    """DFS: refuse if `start` reaches itself through StructType fields.

    Skips StructType references with non-empty `type_args` — those are
    generic instantiations that resolve to a *different* nominal type
    post-monomorphization, so they don't form a cycle with the
    template. The post-mono cycle check (which validates the
    monomorphized program) catches real cycles between concrete types.
    """
    visiting: set[str] = set()

    def go(name: str, path: tuple[str, ...]) -> None:
        if name == start and path:
            chain = " -> ".join(path + (name,))
            raise ValueError(
                f"struct {start!r} contains itself by value (cycle: {chain}); "
                f"quod has no pointer-to-struct, so recursive structs are unrepresentable"
            )
        if name in visiting:
            return
        visiting.add(name)
        sd = by_name.get(name)
        if sd is None:
            return
        for f in sd.fields:
            if isinstance(f.type, StructType) and not f.type.type_args:
                go(f.type.name, path + (name,))
        visiting.discard(name)

    sd = by_name.get(start)
    if sd is None:
        return
    for f in sd.fields:
        if isinstance(f.type, StructType) and not f.type.type_args:
            go(f.type.name, (start,))


class Program(_ProgramBase):
    """The fully-elaborated graph: stored claims + any derived (lattice) claims.

    Permissive: any regime is allowed in fn.claims. This is what `lower()`
    consumes and what editor mutators return.
    """

    @model_validator(mode="after")
    def _check_structs(self) -> "Program":
        _validate_structs(self)
        return self


class InputProgram(_ProgramBase):
    """The graph as authored. Only stored claims (axiom, witness) allowed.

    Used as the validation gate at the JSON I/O boundary: load parses through
    InputProgram (rejects lattice in stored), save round-trips through it
    before writing. Lattice claims live in memory only — they're derived by
    the analysis pass and lowered each build.
    """

    @field_validator("functions")
    @classmethod
    def _no_lattice_in_stored(cls, fns: tuple[Function, ...]) -> tuple[Function, ...]:
        for fn in fns:
            for c in fn.claims:
                if c.regime == "lattice":
                    raise ValueError(
                        f"lattice claims are derived, not stored; "
                        f"function {fn.name!r} has stored claim {c!r}"
                    )
        return fns

    @model_validator(mode="after")
    def _check_structs(self) -> "InputProgram":
        _validate_structs(self)
        return self


def load_program(path: Path) -> Program:
    """Parse program.json. Validates as InputProgram (no lattice in stored)
    then returns the permissive Program type for in-memory editing."""
    raw = path.read_text()
    InputProgram.model_validate_json(raw)
    return Program.model_validate_json(raw)


def save_program(program: Program, path: Path) -> None:
    """Validate as InputProgram (raises if any lattice claims slipped into
    stored), then write JSON atomically.

    Atomic via write-tmp-then-rename: a concurrent reader sees either the old
    file or the new file, never a partially-written one. Mutations also need
    an external lock to prevent two writers from racing on load→save (last
    writer wins); see `_exclusive_lock` in `cli/cli_state.py`.
    """
    InputProgram.model_validate(program.model_dump())
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(program.model_dump_json(indent=2))
    tmp.replace(path)
