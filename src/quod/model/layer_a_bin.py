"""Layer A — binary-artifact source-faithful nodes.

Inert: no validation, no codegen, no semantic checks. Their job is to
preserve a Ghidra dump (`foo.so`, `foo.exe`, `foo.o`) as a subtree of
the program graph so cross-language Equivalence and ProvenanceEdge
claims have a binary-side endpoint to anchor on.

Mirrors `quod.model.layer_a` (the C source-language family) one for
one — same `_Node` base, same content-hashing participation, same ID-
minting convention. The cross-cutting design rule is **don't desugar
at lift**: the c-ingester preserves what clang emitted, this module
preserves what Ghidra emitted. Type recovery, control-flow
restructuring, and re-parsing the decompile text into `c.*` nodes are
all things a binary lifter is normally expected to do; v1 explicitly
refuses them. The decompile is a hint for humans and for the relational
SMT prover, not authoritative C.

See `.scratch/ghidra/01-layer-a-nodes.md` for the design memo.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_serializer

from quod.model.base import _Node, _mint_node_id
from quod.model.layer_a import CFn


# ---------- Embedded value types (no kind, addressed by composition) ----------


class BinVarnode(_Node):
    """A Ghidra p-code varnode — the (address-space, offset, size) triple
    that names a value location. `space` is one of Ghidra's storage
    spaces: `register`, `ram`, `stack`, `unique` (SSA temporaries),
    `const` (immediate operands), `iop` (indirect-op tags). The exact
    set is arch-dependent; we store it as an opaque string rather than
    a Literal because cross-arch consumers will see space names we
    haven't enumerated.

    Carries only the *physical* triple — Ghidra also exposes high-
    variable references and dataflow links, but those are heuristics
    we deliberately don't promote to graph facts in v1.
    """
    space: str
    offset: int
    size: int


class BinFunctionParam(_Node):
    """One parameter on a `BinFunction`'s Ghidra-inferred signature.

    `name` is whatever Ghidra chose (often `param_1`, `param_2`, …
    when no symbol info is available; the symbol-supplied name when
    DWARF or a PDB was present). `type_name` is an opaque Ghidra
    `DataType` name — **not** unified with quod's `Type`. Two
    Ghidra-inferred parameters with the same `type_name` aren't
    guaranteed to be the same quod type; the relational SMT prover
    is what eventually decides whether the binary's parameter shape
    matches the source's.
    """
    name: str
    type_name: str


class BinBlockEdge(_Node):
    """One outgoing CFG edge from a `BinBasicBlock` to a successor block.

    `successor_id` is the target `BinBasicBlock.id` (stable across
    serialization). `edge_kind` distinguishes fall-through from taken
    branches and from non-trivial cases:

      - `fallthrough`: physical next instruction follows in memory;
        emitted on blocks ending in something other than a branch.
      - `true` / `false`: the two arms of a conditional branch.
      - `unconditional`: an unconditional jump.
      - `indirect`: indirect branch (jump table, function pointer); the
        successor here is one resolved target out of possibly many.
      - `call_return`: edge from a call-site block to its
        fall-through successor (the post-call resumption point).
    """
    successor_id: str
    edge_kind: Literal[
        "fallthrough", "true", "false", "unconditional", "indirect", "call_return"
    ]


# ---------- Addressable nodes (own a kind, own an id) ----------


class BinPCodeOp(_Node):
    """One Ghidra p-code operation — the binary frontend's substrate IR.

    P-code is Ghidra's micro-instruction set: each native instruction
    lowers to a small bag of p-code ops with a fixed shape (opcode +
    typed input varnodes + optional output varnode). Preserving p-code
    verbatim gives the graph a uniform binary-substrate the way LLVM IR
    is for source — and a future relational SMT prover for
    `bin.fn ~ src.fn` will consume p-code, not the decompile string.

    `opcode` is a string from Ghidra's pcode opcode set (e.g.
    `INT_ADD`, `LOAD`, `BRANCH`, `CALL`); the set is finite and stable
    so we don't enumerate it in a Literal. `inputs` is the op's input
    varnodes; `output` is None for ops that don't produce a value
    (stores, branches). `source_address` is the native instruction
    address this op was lowered from.
    """
    kind: Literal["bin.pcode"] = "bin.pcode"
    id: str = Field(default_factory=lambda: _mint_node_id("binpcode"))
    opcode: str
    inputs: tuple[BinVarnode, ...] = ()
    output: BinVarnode | None = None
    source_address: int


class BinBasicBlock(_Node):
    """One basic block in a `BinFunction`'s CFG.

    A basic block is a maximal straight-line p-code sequence with one
    entry (`start_address`) and one exit; control flow into the
    block's middle is impossible. `pcode_ops` holds the block's ops in
    execution order (nested by ownership); `successors` are
    `BinBlockEdge`s referencing other blocks **by id** because the CFG
    is a graph, not a tree — nesting block objects by containment
    would force duplication on cycles.

    Addresses are carried for human readability and Ghidra-side cross-
    reference but are not stable identifiers — the same function in
    two builds of the same source can land at different addresses, so
    edges anchor on `id` instead.
    """
    kind: Literal["bin.bb"] = "bin.bb"
    id: str = Field(default_factory=lambda: _mint_node_id("binbb"))
    start_address: int
    end_address: int
    pcode_ops: tuple[BinPCodeOp, ...] = ()
    successors: tuple[BinBlockEdge, ...] = ()


class BinCallEdge(_Node):
    """One outgoing call from a `BinFunction` — caller-side record of a
    `CALL`/`CALLIND`/tail-call site.

    `caller_block_id` is the `BinBasicBlock.id` containing the call
    site; `instruction_address` is the native address of the call
    instruction. `callee_id` is the resolved target's stable ID — a
    `BinFunction.id` for an in-binary call or a `BinExternRef.id` for
    a call into a dynamic dependency. Indirect-call sites where Ghidra
    couldn't resolve a target are still emitted (with `kind="indirect"`)
    but `callee_id` may then point to a synthesized `BinExternRef` for
    the unresolved symbol, or be empty by convention; the ingester
    decides per its policy.

    `kind` distinguishes ordinary direct calls, indirect calls (through
    a function pointer or vtable slot), and tail calls (`JMP` to a
    function). C++ virtual dispatch surfaces here as `kind="indirect"`
    in v1 — vtable/RTTI modeling is deferred.

    `callee_id` is the resolved target's stable ID; `None` means the
    call exists in the binary but the callee couldn't be resolved
    (typically `kind="indirect"` calls where Ghidra didn't propagate
    a constant target — virtual dispatch, function pointers from
    arrays, jump tables). The strategy memo's framing for these:
    silent omission is worse than unresolved-but-marked, so we keep
    the edge in the graph even when the target is unknown — claim
    providers that need full call-graph reachability can see the
    edge exists and decide how to model the unresolved target
    (uninterpreted function, conservative over-approximation, etc.).
    """
    kind: Literal["bin.call"] = "bin.call"
    id: str = Field(default_factory=lambda: _mint_node_id("bincall"))
    caller_block_id: str
    instruction_address: int
    callee_id: str | None = None
    call_kind: Literal["direct", "indirect", "tail"]

    @model_serializer(mode="wrap")
    def _drop_none_callee_id(self, handler, info):
        data = handler(self)
        if self.callee_id is None:
            data.pop("callee_id", None)
        return data


class BinFunction(_Node):
    """One function recovered from a binary by Ghidra.

    Carries Ghidra's recovered signature (`return_type_name`, `params`)
    as **opaque strings** — types are hypotheses the verifier confirms
    or rejects, so we don't fold them into quod's type lattice on
    ingest. `mangled_name` and `demangled_name` are both kept;
    demangling is Ghidra-supplied and informational. `calling_convention`
    is opaque (`__cdecl`, `__stdcall`, `__fastcall`, target-specific
    names); we don't validate it against the ABI.

    `decompile_text` is Ghidra's C-like decompile output stored as
    a single opaque string. The temptation is to re-parse it through
    libclang and populate `c.*` Layer-A nodes; **don't, in v1**. Two
    reasons: (1) the decompile is a heuristic — type recovery,
    variable splitting, loop reconstruction — and treating it as
    typed C forces false confidence into the graph; (2) types are
    hypotheses the verifier confirms, so re-parsing would short-
    circuit that. The string lives here because it's useful to humans
    and to agents-as-readers, not because it's authoritative. v2 may
    add a `bin.decompile_lift` pass that emits a Layer-B `c.*` subtree
    alongside with a pinned `LiftEquivalence`; that's a separate proof
    obligation, not a free byproduct.

    Addresses are carried but are not stable identifiers — see
    `BinBasicBlock` for the rationale.
    """
    kind: Literal["bin.fn"] = "bin.fn"
    id: str = Field(default_factory=lambda: _mint_node_id("binfn"))
    address: int
    mangled_name: str
    demangled_name: str
    return_type_name: str
    params: tuple[BinFunctionParam, ...] = ()
    calling_convention: str
    basic_blocks: tuple[BinBasicBlock, ...] = ()
    call_edges: tuple[BinCallEdge, ...] = ()
    decompile_text: str = ""
    # DWARF source-line attribution (DW_AT_decl_file / DW_AT_decl_line),
    # populated by Ghidra's source-file manager when the binary was
    # built with `-g`. The seeder prefers DWARF over symtab name match
    # — DWARF disambiguates `static int helper()` collisions across
    # translation units, where symtab alone would refuse to seed.
    # Both fields are None on stripped binaries and on compiler-emitted
    # bookkeeping functions (`_init`, `frame_dummy`, etc.). `decl_file`
    # carries Ghidra's recorded path verbatim (typically the compile-
    # time absolute path); the seeder compares basenames so different
    # working directories don't break attribution.
    decl_file: str | None = None
    decl_line: int | None = None

    @model_serializer(mode="wrap")
    def _drop_empty_optionals(self, handler, info):
        data = handler(self)
        if not self.decompile_text:
            data.pop("decompile_text", None)
        if self.decl_file is None:
            data.pop("decl_file", None)
        if self.decl_line is None:
            data.pop("decl_line", None)
        return data


class BinDataItem(_Node):
    """A data symbol Ghidra recovered — a string literal in `.rodata`,
    a global in `.data`, or similar.

    `data_kind` distinguishes the two cases; `value` carries the
    payload. For `string` items, `value` is the decoded utf8 (escapes
    resolved). For `global` items, `value` is base64-encoded raw
    bytes — the schema doesn't impose a type, since type recovery on
    globals is heuristic and we keep it opaque per the
    `BinTypeRef`-opacity rule.

    `referenced_by` is a tuple of `BinFunction.id`s whose code reads
    or writes this data item — Ghidra's xref information surfaced as
    a graph fact. Empty when no references were found (or when the
    ingester didn't compute xrefs).

    The split between `.rodata` strings and `.data` globals is
    section-derived, not type-derived; v2 may want to record the
    section explicitly, but for v1 the `data_kind` enum is enough.
    """
    kind: Literal["bin.data"] = "bin.data"
    id: str = Field(default_factory=lambda: _mint_node_id("bindata"))
    address: int
    data_kind: Literal["string", "global"]
    value: str
    referenced_by: tuple[str, ...] = ()


class BinExternRef(_Node):
    """A reference into a binary the analyzed artifact depends on —
    typically a `PLT`/`IAT` entry for an imported libc symbol.

    `symbol` is the linker-level name (`malloc`, `printf`, …);
    `abi_hint` is an opaque string Ghidra may have inferred about the
    calling convention or stack discipline (often `None`).
    `linked_extern_name` cross-references an existing
    `ExternFunction.name` in the same `Program` when name-match
    succeeds — the ingester populates it as a candidate for the
    `Equivalence` seeder; the cross-link is a hint, not an assertion.
    Stripped binaries break the name-match path; that's a known v1
    limitation captured in `.scratch/ghidra/05-open-questions.md`.
    """
    kind: Literal["bin.extern"] = "bin.extern"
    id: str = Field(default_factory=lambda: _mint_node_id("binextern"))
    symbol: str
    abi_hint: str | None = None
    linked_extern_name: str | None = None

    @model_serializer(mode="wrap")
    def _drop_none_optionals(self, handler, info):
        data = handler(self)
        if self.abi_hint is None:
            data.pop("abi_hint", None)
        if self.linked_extern_name is None:
            data.pop("linked_extern_name", None)
        return data


class BinTypeRef(_Node):
    """A Ghidra `DataType` Ghidra inferred (either from debug info or
    from field-access patterns) — preserved as opaque metadata.

    Deliberately **not** unified with quod's `Type`. Folding it in
    would force every consumer of `Program.functions` to handle types
    that weren't authored, and would assume Ghidra's struct splitting
    is correct. Claim providers that want to reason about
    "the field at offset 8 is an i32 because Ghidra said so" can read
    `BinTypeRef`s directly, but the core graph never trusts them.

    `name` is the type's display name (`int`, `MyStruct`, `char *`,
    `undefined4`, …); `size` is its in-memory size in bytes;
    `structural_hash` is whatever stable hash the export script
    computed over Ghidra's structural representation, used for
    deduplication and drift detection across re-runs of the export.
    """
    kind: Literal["bin.type"] = "bin.type"
    id: str = Field(default_factory=lambda: _mint_node_id("bintype"))
    name: str
    size: int
    structural_hash: str


class BinUnit(_Node):
    """One ingested binary artifact — the binary-side analogue of `CUnit`.

    `path` is the on-disk path the artifact was ingested from (recorded
    so the graph can be paired back with the file and re-ingested if
    the binary changes). `sha256` content-addresses the file at ingest
    time; `build_id` (when present) is the linker-supplied `NT_GNU_BUILD_ID`
    or equivalent — load-bearing for matching against debug info /
    DWARF lookups by build-id.

    `arch` and `file_format` are stored as strings rather than
    enumerations so cross-arch and exotic-format consumers don't need
    schema bumps; common values are listed in the field types.

    Owns `functions`, `data_items`, `extern_refs`, and `type_refs` by
    composition — they're addressed by their `id` from edges and
    cross-references but live nested in the unit because that's the
    natural ingest scope. Mirrors how `CUnit.functions` owns `CFn`s.
    """
    kind: Literal["bin.unit"] = "bin.unit"
    id: str = Field(default_factory=lambda: _mint_node_id("binunit"))
    path: str
    sha256: str
    arch: str
    file_format: Literal["elf", "pe", "mach-o", "raw"]
    build_id: str | None = None
    functions: tuple[BinFunction, ...] = ()
    data_items: tuple[BinDataItem, ...] = ()
    extern_refs: tuple[BinExternRef, ...] = ()
    type_refs: tuple[BinTypeRef, ...] = ()
    # Layer-A `c.*` CFns reconstructed from each `BinFunction`'s
    # `decompile_text` by `binary_decompile_lift`. Empty until the
    # lift pass runs. Kept nested under the binary they came from
    # (rather than mixed into `Program.source_units`) so authored C
    # and decompiler-recovered C stay clearly separated — the lifted
    # CFns are *evidence* of what Ghidra recovered, not source the
    # user wrote. Pairings to the originating `BinFunction` are
    # carried by `Equivalence` claims at program level, justified
    # with `DecompileLift`.
    lifted_cfns: tuple[CFn, ...] = ()

    @model_serializer(mode="wrap")
    def _drop_empty_collections(self, handler, info):
        data = handler(self)
        if self.build_id is None:
            data.pop("build_id", None)
        if not self.functions:
            data.pop("functions", None)
        if not self.data_items:
            data.pop("data_items", None)
        if not self.extern_refs:
            data.pop("extern_refs", None)
        if not self.type_refs:
            data.pop("type_refs", None)
        if not self.lifted_cfns:
            data.pop("lifted_cfns", None)
        return data
