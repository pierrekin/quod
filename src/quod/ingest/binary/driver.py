"""Top-level driver for binary ingestion.

Two entry points:

- `ingest_binary(path, program=...)` drives Ghidra over `path` via
  PyGhidra (`quod.ingest.binary.ghidra_export`), parses the resulting
  JSON dump, builds layer-A `bin.*` nodes, runs the equivalence
  seeder, and returns the updated `Program`. Ghidra runs **in
  process** through PyGhidra's JPype bridge — modern Ghidra (12+)
  dropped the Jython 2.7 path, so the design memo's earlier
  subprocess-Jython sketch no longer applies. The JSON contract
  itself is preserved (see `.scratch/ghidra/02-ghidra-export.md`)
  because that's the swap-Ghidra-for-radare2/BAP/angr boundary.

- `ingest_binary_dump(json_path, program=...)` skips Ghidra entirely
  and parses an existing JSON dump. Useful (a) for tests that can't
  pay JVM startup cost, and (b) for incremental workflows where a
  dump already exists. The two paths share the same parser and seeder.

Layer-A is **inert** — see `quod.model.layer_a_bin` for the
non-negotiables (no type recovery into `StructDef`, no re-parse of
the decompile text into `c.*` nodes). The driver's job is to translate
the JSON contract into nodes, not to interpret the binary.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quod.model import (
    BinaryProvenance,
    BinBasicBlock,
    BinBlockEdge,
    BinCallEdge,
    BinDataItem,
    BinExternRef,
    BinFunction,
    BinFunctionParam,
    BinPCodeOp,
    BinTypeRef,
    BinUnit,
    BinVarnode,
    Equivalence,
    Program,
    ProvenanceEdge,
)


SUPPORTED_SCHEMA_VERSION = 1
"""The JSON contract version this driver speaks. The exporter emits
`schema_version: 1`; future Ghidra-side changes that break the contract
bump this number on both sides."""


class BinaryIngestError(Exception):
    """Raised when a binary dump can't be parsed or when Ghidra itself
    failed. Mirrors `IngestError` in the C frontend — caller-visible
    failures all funnel through one exception type."""


# ---------- Public API ----------


def ingest_binary(
    binary_path: Path | str,
    *,
    program: Program | None = None,
    keep_dump: Path | str | None = None,
) -> Program:
    """Ingest a binary by driving Ghidra in-process via PyGhidra.

    `binary_path` is the artifact to ingest. `program` is the existing
    program to extend (a fresh empty `Program()` is used if None).
    `keep_dump`, when given, writes the raw JSON dump to that path
    (in addition to parsing it) for diagnosing parser issues.

    The Ghidra project is ephemeral: each call creates a fresh project
    in a temp directory and tears it down afterwards. The JVM is
    started lazily on first call (~5s) and persists for the rest of
    the process — subsequent ingests in the same Python process don't
    pay the startup cost again.

    Raises `BinaryIngestError` on missing binary, JSON parse failure,
    unsupported schema version, or any error PyGhidra raises during
    analysis.
    """
    binary_path = Path(binary_path).resolve()
    if not binary_path.is_file():
        raise BinaryIngestError(f"binary not found: {binary_path}")

    from quod.ingest.binary.ghidra_export import export_to_json

    with tempfile.TemporaryDirectory(prefix="quod-bin-ingest-") as tmp:
        dump_path = Path(tmp) / "dump.json"
        try:
            export_to_json(binary_path, dump_path)
        except RuntimeError as e:
            raise BinaryIngestError(str(e)) from e
        except Exception as e:
            raise BinaryIngestError(
                f"Ghidra analysis of {binary_path} failed: {e}"
            ) from e

        if not dump_path.is_file():
            raise BinaryIngestError(
                f"exporter did not produce {dump_path}"
            )

        if keep_dump is not None:
            Path(keep_dump).write_bytes(dump_path.read_bytes())

        return ingest_binary_dump(dump_path, program=program)


def ingest_binary_dump(
    dump_path: Path | str,
    *,
    program: Program | None = None,
) -> Program:
    """Parse an existing JSON dump (produced by `binary_ghidra_export.py`
    or a compatible tool) and add a `BinUnit` to `program`. Subprocess-
    free; the test path.

    Returns a new `Program` with the binary unit appended to
    `binary_units` and seeded equivalences/provenance edges merged into
    `equivalences`/`edges`. Existing `program` is not mutated.
    """
    dump_path = Path(dump_path)
    try:
        raw = json.loads(dump_path.read_text())
    except json.JSONDecodeError as e:
        raise BinaryIngestError(f"{dump_path}: invalid JSON: {e}") from e
    return _ingest_parsed(raw, program=program, source_label=str(dump_path))


# ---------- Parser ----------


def _ingest_parsed(
    raw: dict[str, Any],
    *,
    program: Program | None,
    source_label: str,
) -> Program:
    schema = raw.get("schema_version")
    if schema != SUPPORTED_SCHEMA_VERSION:
        raise BinaryIngestError(
            f"{source_label}: unsupported schema_version {schema!r} "
            f"(expected {SUPPORTED_SCHEMA_VERSION})"
        )

    binary = raw.get("binary") or {}
    unit = _build_unit(binary, raw, source_label)

    base = program if program is not None else Program()
    new_units = base.binary_units + (unit,)
    extended = base.model_copy(update={"binary_units": new_units})

    return seed_binary_equivalences(extended, unit=unit)


def _parse_int_addr(s: str | int | None) -> int:
    """Accept hex strings (`"0x401120"`) or already-decoded ints. The
    export script emits hex strings; tests sometimes hand-craft ints."""
    if s is None:
        return 0
    if isinstance(s, int):
        return s
    text = s.strip()
    if text.startswith("0x") or text.startswith("0X"):
        return int(text, 16)
    return int(text)


def _build_varnode(raw: dict[str, Any]) -> BinVarnode:
    return BinVarnode(
        space=str(raw["space"]),
        offset=_parse_int_addr(raw.get("offset", 0)),
        size=int(raw["size"]),
    )


def _build_pcode(raw: dict[str, Any]) -> BinPCodeOp:
    inputs = tuple(_build_varnode(v) for v in raw.get("inputs") or ())
    output = raw.get("output")
    return BinPCodeOp(
        opcode=str(raw["opcode"]),
        inputs=inputs,
        output=_build_varnode(output) if output is not None else None,
        source_address=_parse_int_addr(raw.get("instr_address")),
    )


_VALID_BLOCK_EDGE_KINDS = {
    "fallthrough", "true", "false", "unconditional", "indirect", "call_return",
}


def _coerce_edge_kind(kind: str) -> str:
    """Map exporter-emitted edge kinds onto the model's enum, refusing
    anything outside the contract. The exporter is supposed to emit
    one of the six labels; this is a belt-and-braces check."""
    if kind in _VALID_BLOCK_EDGE_KINDS:
        return kind
    raise BinaryIngestError(
        f"unknown CFG edge kind {kind!r}; expected one of "
        f"{sorted(_VALID_BLOCK_EDGE_KINDS)}"
    )


def _build_block(
    raw: dict[str, Any],
    addr_to_block_id: dict[int, str],
) -> BinBasicBlock:
    """First pass: build the block without successors (we don't yet
    have IDs for the successor blocks). Caller fills successors in a
    second pass."""
    pcode = tuple(_build_pcode(p) for p in raw.get("pcode") or ())
    block = BinBasicBlock(
        start_address=_parse_int_addr(raw.get("address")),
        end_address=_parse_int_addr(raw.get("end")),
        pcode_ops=pcode,
    )
    addr_to_block_id[block.start_address] = block.id
    return block


def _attach_successors(
    block: BinBasicBlock,
    raw_succs: list[dict[str, Any]],
    addr_to_block_id: dict[int, str],
) -> BinBasicBlock:
    edges: list[BinBlockEdge] = []
    for s in raw_succs:
        addr = _parse_int_addr(s.get("address"))
        kind = _coerce_edge_kind(str(s.get("kind", "unconditional")))
        target_id = addr_to_block_id.get(addr)
        if target_id is None:
            # Edge points outside the function (e.g. tail-call or
            # noreturn dispatch). Skip — the call_edges list captures
            # cross-function transfers; intra-function CFG is what we
            # store on `successors`.
            continue
        edges.append(BinBlockEdge(successor_id=target_id, edge_kind=kind))
    return block.model_copy(update={"successors": tuple(edges)})


def _build_param(raw: dict[str, Any]) -> BinFunctionParam:
    return BinFunctionParam(
        name=str(raw.get("name") or ""),
        type_name=str(raw.get("type") or ""),
    )


def _build_function(
    raw: dict[str, Any],
    extern_id_by_address: dict[int, str],
    extern_id_by_name: dict[str, str],
    fn_id_by_address: dict[int, str],
) -> BinFunction:
    addr_to_block_id: dict[int, str] = {}
    blocks = [_build_block(b, addr_to_block_id) for b in raw.get("basic_blocks") or ()]
    blocks = [
        _attach_successors(b, raw_block.get("successors") or (), addr_to_block_id)
        for b, raw_block in zip(blocks, raw.get("basic_blocks") or ())
    ]

    sig = raw.get("signature") or {}
    params = tuple(_build_param(p) for p in sig.get("params") or ())

    fn = BinFunction(
        address=_parse_int_addr(raw.get("address")),
        mangled_name=str(raw.get("name_mangled") or ""),
        demangled_name=str(raw.get("name_demangled") or ""),
        return_type_name=str(sig.get("return_type") or ""),
        params=params,
        calling_convention=str(raw.get("calling_convention") or ""),
        basic_blocks=tuple(blocks),
        decompile_text=str(raw.get("decompile") or ""),
    )

    call_edges: list[BinCallEdge] = []
    for c in raw.get("calls") or ():
        from_block_addr = _parse_int_addr(c.get("from_block"))
        block_id = addr_to_block_id.get(from_block_addr)
        if block_id is None:
            # Call recorded against an address Ghidra didn't promote
            # to a basic block start — happens for compiler-emitted
            # epilogue thunks. Fall back to the function's first block;
            # the call edge is still useful even if the caller-block
            # link is approximate.
            block_id = blocks[0].id if blocks else fn.id

        callee_id = _resolve_callee(c.get("to") or {}, extern_id_by_address, extern_id_by_name, fn_id_by_address)
        call_kind = str(c.get("call_kind") or "direct")
        if call_kind not in {"direct", "indirect", "tail"}:
            raise BinaryIngestError(
                f"unknown call_kind {call_kind!r}; expected direct/indirect/tail"
            )
        if callee_id is None:
            # Indirect call site Ghidra couldn't resolve; emit a synthetic
            # extern ref so the callee_id is still a valid stable ID.
            # The caller is responsible for adding the synthetic extern
            # to the unit's extern_refs — for now we just skip.
            continue
        call_edges.append(BinCallEdge(
            caller_block_id=block_id,
            instruction_address=_parse_int_addr(c.get("instr_address")),
            callee_id=callee_id,
            call_kind=call_kind,  # type: ignore[arg-type]
        ))

    return fn.model_copy(update={"call_edges": tuple(call_edges)})


def _resolve_callee(
    to: dict[str, Any],
    extern_id_by_address: dict[int, str],
    extern_id_by_name: dict[str, str],
    fn_id_by_address: dict[int, str],
) -> str | None:
    kind = to.get("kind")
    if kind == "internal":
        addr = _parse_int_addr(to.get("address"))
        return fn_id_by_address.get(addr)
    if kind == "external":
        name = to.get("name")
        if name and name in extern_id_by_name:
            return extern_id_by_name[name]
        addr = _parse_int_addr(to.get("address"))
        if addr and addr in extern_id_by_address:
            return extern_id_by_address[addr]
    return None


def _build_data(raw: dict[str, Any]) -> BinDataItem:
    kind = str(raw.get("kind") or "")
    if kind not in ("string", "global"):
        raise BinaryIngestError(
            f"unknown data kind {kind!r}; expected 'string' or 'global'"
        )
    return BinDataItem(
        address=_parse_int_addr(raw.get("address")),
        data_kind=kind,  # type: ignore[arg-type]
        value=str(raw.get("value") or ""),
    )


def _build_extern(raw: dict[str, Any]) -> BinExternRef:
    return BinExternRef(
        symbol=str(raw.get("name") or ""),
        abi_hint=raw.get("abi_hint"),
    )


def _build_type_ref(raw: dict[str, Any]) -> BinTypeRef:
    return BinTypeRef(
        name=str(raw.get("name") or ""),
        size=int(raw.get("size") or 0),
        structural_hash=str(raw.get("structural_hash") or ""),
    )


def _build_unit(
    binary: dict[str, Any],
    raw: dict[str, Any],
    source_label: str,
) -> BinUnit:
    file_format = binary.get("format", "raw")
    if file_format not in ("elf", "pe", "mach-o", "raw"):
        raise BinaryIngestError(
            f"{source_label}: unsupported file_format {file_format!r}"
        )

    extern_refs = tuple(_build_extern(e) for e in raw.get("externs") or ())
    extern_id_by_name = {e.symbol: e.id for e in extern_refs}
    extern_id_by_address: dict[int, str] = {}
    for raw_extern, ref in zip(raw.get("externs") or (), extern_refs):
        addr = raw_extern.get("address")
        if addr:
            extern_id_by_address[_parse_int_addr(addr)] = ref.id

    # First pass over functions: just to populate the address→id map for
    # internal-call resolution. We build the real BinFunctions afterward
    # because resolution needs all addresses available up front.
    raw_functions = list(raw.get("functions") or ())
    fn_id_by_address: dict[int, str] = {}
    placeholder_ids: list[str] = []
    for fn_raw in raw_functions:
        # Mint an ID up front so callers in the same dump can resolve
        # internal calls. The actual BinFunction is built below; we
        # pass the pre-minted ID through `model_copy(update={"id": ...})`.
        from quod.model.base import _mint_node_id
        placeholder = _mint_node_id("binfn")
        placeholder_ids.append(placeholder)
        fn_id_by_address[_parse_int_addr(fn_raw.get("address"))] = placeholder

    functions: list[BinFunction] = []
    for fn_raw, fn_id in zip(raw_functions, placeholder_ids):
        fn = _build_function(fn_raw, extern_id_by_address, extern_id_by_name, fn_id_by_address)
        functions.append(fn.model_copy(update={"id": fn_id}))

    # Stitch data → fn xrefs by address (which functions reference each
    # data item). Ghidra has this info but the v1 export doesn't carry it
    # explicitly; we leave referenced_by empty for now and let a future
    # exporter version populate it.
    data_items = tuple(_build_data(d) for d in raw.get("data") or ())
    type_refs = tuple(_build_type_ref(t) for t in raw.get("type_refs") or ())

    return BinUnit(
        path=str(binary.get("path") or ""),
        sha256=str(binary.get("sha256") or ""),
        arch=str(binary.get("arch") or ""),
        file_format=file_format,  # type: ignore[arg-type]
        build_id=binary.get("build_id"),
        functions=tuple(functions),
        data_items=data_items,
        extern_refs=extern_refs,
        type_refs=type_refs,
    )


# ---------- Equivalence seeder ----------


@dataclass(frozen=True)
class _SourceTarget:
    """One candidate source-side endpoint for a `bin.fn` to pair with."""
    fn_id: str
    name: str
    evidence: str  # "dwarf" | "symtab"


def seed_binary_equivalences(program: Program, *, unit: BinUnit | None = None) -> Program:
    """Walk `program.binary_units` and emit `Equivalence` + `ProvenanceEdge`
    pairs for every `bin.fn` whose name matches a known source function.

    Strategy (priority order; v1 implements DWARF-omitted symtab match,
    DWARF preferred when present):

    1. DWARF line info match — not yet implemented. The export script
       doesn't yet surface `DW_AT_decl_file`/`DW_AT_decl_line`; when it
       does, this seeder upgrades to prefer DWARF over plain symtab.
    2. Symtab name match — demangle `bin.fn.demangled_name`, match
       against `Function.name` in `program.functions` and `CFn.name` in
       any `CUnit`. Source unit found → seed.
    3. No match — leave the bin.fn unpaired. Stripped binaries land
       here by design (see `.scratch/ghidra/05-open-questions.md`).

    The seeder is idempotent: re-running it on a program that already
    carries a seeded equivalence (same `(a_node_id, b_node_id)`) does
    not duplicate. Pass `unit=` to seed only that unit; otherwise
    every unit in the program is considered.
    """
    candidates = _collect_source_candidates(program)
    units = (unit,) if unit is not None else program.binary_units

    new_eqs: list[Equivalence] = []
    new_edges: list[ProvenanceEdge] = []
    existing_pairs = {(eq.a_node_id, eq.b_node_id) for eq in program.equivalences}
    existing_edges = {(e.source, e.target) for e in program.edges}

    for u in units:
        for fn in u.functions:
            target = _match_source(fn, candidates)
            if target is None:
                continue
            pair = (target.fn_id, fn.id)
            if pair not in existing_pairs:
                new_eqs.append(Equivalence(
                    a_node_id=target.fn_id,
                    b_node_id=fn.id,
                    regime="axiom",
                    enforcement="trust",
                    justification=BinaryProvenance(
                        binary_path=u.path,
                        binary_sha256=u.sha256,
                        binary_symbol=fn.mangled_name or fn.demangled_name,
                        source_evidence=target.evidence,  # type: ignore[arg-type]
                    ),
                ))
                existing_pairs.add(pair)
            if pair not in existing_edges:
                new_edges.append(ProvenanceEdge(source=target.fn_id, target=fn.id))
                existing_edges.add(pair)

    if not new_eqs and not new_edges:
        return program

    return program.model_copy(update={
        "equivalences": program.equivalences + tuple(new_eqs),
        "edges": program.edges + tuple(new_edges),
    })


@dataclass(frozen=True)
class _SourceCandidates:
    """Per-name buckets of source candidates, split by layer.

    The c-ingester emits parallel CFn (Layer A) and Function (Layer C)
    nodes for every C function, so the *normal* post-ingest state is
    "two source candidates per name" — that's not a collision, it's the
    A↔B↔C decomposition. The seeder prefers Layer A: pairing
    `bin.fn ↔ CFn` is more authoritative because the binary was built
    from the C source, and the existing A↔B↔C equivalence chain then
    propagates the relationship to Layer C transitively.

    True collisions are when the *same layer* has two functions with
    the same name (e.g. `static int helper()` in two CUnits). Those
    are refused — the user disambiguates via `quod bin equiv add`.
    """
    cfns: dict[str, list[_SourceTarget]]
    fns: dict[str, list[_SourceTarget]]


def _collect_source_candidates(program: Program) -> _SourceCandidates:
    cfns: dict[str, list[_SourceTarget]] = {}
    for cunit in program.source_units:
        for cfn in cunit.functions:
            cfns.setdefault(cfn.name, []).append(
                _SourceTarget(fn_id=cfn.id, name=cfn.name, evidence="symtab")
            )
    fns: dict[str, list[_SourceTarget]] = {}
    for fn in program.functions:
        fns.setdefault(fn.name, []).append(
            _SourceTarget(fn_id=fn.id, name=fn.name, evidence="symtab")
        )
    return _SourceCandidates(cfns=cfns, fns=fns)


def _match_source(
    bin_fn: BinFunction,
    candidates: _SourceCandidates,
) -> _SourceTarget | None:
    """Pair a `bin.fn` to a unique source endpoint by name. Layer A
    (CFn) wins over Layer C (Function) when both have a single match;
    a same-layer collision (multiple CFns or multiple Functions with
    the same name) leaves the function unpaired."""
    for name in (bin_fn.demangled_name, bin_fn.mangled_name):
        if not name:
            continue
        cfn_targets = candidates.cfns.get(name) or []
        if len(cfn_targets) == 1:
            return cfn_targets[0]
        if len(cfn_targets) > 1:
            return None  # same-layer collision in CFns
        fn_targets = candidates.fns.get(name) or []
        if len(fn_targets) == 1:
            return fn_targets[0]
        if len(fn_targets) > 1:
            return None  # same-layer collision in Functions
    return None
