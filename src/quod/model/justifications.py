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
    ],
    Field(discriminator="kind"),
]
