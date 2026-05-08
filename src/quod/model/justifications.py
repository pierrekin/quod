"""Justification nodes — polymorphic evidence channel for claims.

The kind discriminator says what flavor of evidence is on offer; the
regime field on the claim is a coarse epistemic label (loosely
correlated, not enforced).

  z3        — external proof in SMT-LIB; verifiable by re-running Z3
              or, in MVP2, just by re-hashing the artifact
  manual    — a human signed off; no machine-checkable evidence
  derived   — produced by an analysis pass; reproducible from `inputs`
              (content-hash refs to the graph nodes the analysis read)
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, model_serializer

from quod.model.base import _Node


class Z3Justification(_Node):
    # `artifact_hash` pins the .smt2 file's bytes ("did the file get
    # tampered with"). `body_smt_hash` pins the SMT text the current
    # body+claim *would* produce ("did the body drift"). Both are
    # sha256 of the same bytes at prove time; they answer different
    # questions at verify time.
    kind: Literal["z3"] = "z3"
    artifact_path: str
    artifact_hash: str
    body_smt_hash: str
    note: str | None = None


class ManualJustification(_Node):
    kind: Literal["manual"] = "manual"
    signed_by: str
    rationale: str


class DerivedJustification(_Node):
    kind: Literal["derived"] = "derived"
    analysis: str                       # name of the analysis pass
    inputs: tuple[str, ...] = ()        # content hashes of nodes the pass read
    note: str | None = None


class LiftEquivalence(_Node):
    """Justifies an A→B (source-language → c-like-quod) equivalence
    via a pinned *structural transcription record* — produced by
    `quod.lift_check.walk_lift`, which walks both subtrees in
    lockstep and confirms one-to-one node correspondence per a
    per-construct table.

    `artifact_path` is relative to the program's resolve_root; the
    file's bytes are hashed at prove time and re-checked at verify
    time. Verification is hash-only: re-walking the layer-A and
    layer-B subtrees in memory reproduces the artifact bytes
    deterministically, so a hash match is sufficient evidence that
    the lift is still faithful. No Z3 invocation — the artifact
    is JSON, not SMT.

    SMT-based A→B proofs would land as a distinct justification
    kind; this one is reserved for the structural-correspondence-
    record shape.
    """
    kind: Literal["lift_equivalence"] = "lift_equivalence"
    artifact_path: str
    artifact_hash: str
    note: str | None = None

    @model_serializer(mode="wrap")
    def _drop_default_metadata(self, handler, info):
        data = handler(self)
        if self.note is None:
            data.pop("note", None)
        return data


class BinaryProvenance(_Node):
    """Justifies an `Equivalence` between a source-language function and
    a binary function via build-time symbol provenance.

    The claim: the binary at `binary_path` (pinned by `binary_sha256`)
    defines a symbol `binary_symbol` whose linker-resolved source is the
    function the equivalence's other endpoint names. Verification re-
    reads the binary and confirms (a) the sha256 still matches, and (b)
    the symbol table still resolves the symbol at the recorded address.

    Strictly weaker evidence than Z3 — a name match doesn't prove
    semantic equivalence, only that the build pipeline thinks these are
    the same function. But it's the right starting axiom for a future
    relational prover (the `z3.bin_relational` provider sketched in
    `.scratch/ghidra/04-providers-and-the-bet.md`) to upgrade
    axiom→witness.

    `source_evidence` is a coarse epistemic label:
      - `dwarf`   — DW_AT_decl_file/line points back at the source file.
      - `symtab`  — only the demangled name matched; no source-line link.
      - `explicit` — user-asserted (`quod bin equiv add`); no automatic
                     evidence.

    The cross-language regime here is genuinely an axiom in the existing
    sense: if it's false (the build was broken, the user replaced the
    binary, etc.) then claims the optimizer derives from one side won't
    hold on the other — the same UB-on-falsity contract `axiom` already
    carries, just at the cross-language layer.
    """
    kind: Literal["binary_provenance"] = "binary_provenance"
    binary_path: str
    binary_sha256: str
    binary_symbol: str
    source_evidence: Literal["symtab", "dwarf", "explicit"]
    note: str | None = None

    @model_serializer(mode="wrap")
    def _drop_default_metadata(self, handler, info):
        data = handler(self)
        if self.note is None:
            data.pop("note", None)
        return data


class DecompileLift(_Node):
    """Justifies an Equivalence between a `BinFunction` and a Layer-A
    `CFn` lifted from Ghidra's decompile output.

    The CFn is produced by parsing `BinFunction.decompile_text` via
    libclang and walking the resulting AST with the same translator
    the C frontend uses for authored source. `decompile_text_sha256`
    pins the input bytes — a re-run of the lift on the same bin.fn
    must produce decompile_text whose hash matches, otherwise the
    equivalence is invalidated (typically by a Ghidra version bump
    that changed the decompile output).

    Strictly weaker than the relational SMT proof
    (`z3.bin_relational`): DecompileLift only attests "this CFn is
    what Ghidra's decompiler said the binary was at this point in
    time", not that Ghidra got the recovery right. The structural
    transcription from the libclang AST to `c.*` nodes is the *same*
    pass the source ingester runs, so that half is faithful by
    construction (and shares its hazards). The weak link is the
    decompiler's output itself.

    A future provider can promote (axiom → witness) by proving
    `bin.fn ~ lifted_cfn` relationally — a separate obligation that
    consumes this equivalence as a starting point.
    """
    kind: Literal["decompile_lift"] = "decompile_lift"
    decompile_text_sha256: str
    note: str | None = None

    @model_serializer(mode="wrap")
    def _drop_default_metadata(self, handler, info):
        data = handler(self)
        if self.note is None:
            data.pop("note", None)
        return data


class FamilyLowering(_Node):
    """Justifies a B→C equivalence by citing a named lowering rule
    (e.g. `c.for_general`) whose equivalence theorem was proved once,
    out of band, against the rule itself rather than per program.

    `rule_name` identifies the rule in the family's lowering pass.
    `artifact_path`/`artifact_hash` optionally pin the rule's proof
    artifact; when None, the citation is a manual claim that the rule
    has been proved elsewhere.
    """
    kind: Literal["family_lowering"] = "family_lowering"
    rule_name: str
    artifact_path: str | None = None
    artifact_hash: str | None = None
    note: str | None = None

    @model_serializer(mode="wrap")
    def _drop_default_metadata(self, handler, info):
        data = handler(self)
        if self.artifact_path is None:
            data.pop("artifact_path", None)
        if self.artifact_hash is None:
            data.pop("artifact_hash", None)
        if self.note is None:
            data.pop("note", None)
        return data


Justification = Annotated[
    Union[
        Z3Justification,
        ManualJustification,
        DerivedJustification,
        LiftEquivalence,
        FamilyLowering,
        BinaryProvenance,
        DecompileLift,
    ],
    Field(discriminator="kind"),
]
