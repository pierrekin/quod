"""Composition + cross-layer relation forms.

`WireBinding` and `Import` describe module composition (how programs
glue together imported modules); `ProvenanceEdge` and `Equivalence`
describe cross-layer relations (how nodes from layer A, B, C relate to
each other). Both are program-level metadata and stay together because
they share the same "asserts something between two parties" shape.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_serializer

from quod.model.base import _Node
from quod.model.claims import Enforcement, Regime
from quod.model.justifications import Justification
from quod.model.types import Type


class WireBinding(_Node):
    """One `wire X=Y` clause inside an `Import`. Binds the wirable named
    `name` (declared at the imported module's scope) to `type`."""
    name: str
    type: Type


class Import(_Node):
    """A structured import. JSON form accepts either a bare string
    (`"alloc.list"`) or an object (`{"module": "alloc.list", "wire":
    [{"name": "A", "type": ...}]}`); a `field_validator` coerces
    strings to `{module: str}` so existing JSONs keep working.

    `wire` binds the imported module's wirables. The resolver
    substitutes those bindings throughout the module's body before
    merging it into the consumer.
    """
    module: str
    wire: tuple[WireBinding, ...] = ()

    @field_validator("wire", mode="before")
    @classmethod
    def _coerce_wire(cls, raw):
        # strict-mode Pydantic doesn't auto-coerce list→tuple; do it here
        # so JSON arrays parse cleanly.
        if isinstance(raw, list):
            return tuple(raw)
        return raw

    @model_serializer(mode="wrap")
    def _bare_string_when_no_wire(self, handler, info):
        # Round-trip optimization: if there's no wire, serialize as a
        # bare string. Existing programs without wirables stay byte-for-byte
        # identical on save.
        if not self.wire:
            return self.module
        data = handler(self)
        return data


class ProvenanceEdge(_Node):
    """An unkinded provenance edge: "this came from that," nothing more.

    `source` and `target` are stable node IDs (e.g. Function.id or
    Block.id). All semantic content for what the edge *means* lives in
    the `Equivalence` claims that anchor on the same IDs — the edge
    itself only records connectivity. N:M lowerings emit one
    ProvenanceEdge per (source, target) pair so the graph stays
    normalized.
    """
    kind: Literal["edge.provenance"] = "edge.provenance"
    source: str
    target: str


class Equivalence(_Node):
    """Program-level equivalence between two nodes by ID.

    Where ordinary claims live in `fn.claims` and constrain a single
    function's parameters or return value, an Equivalence is *relational*
    — it asserts that two nodes (typically across language-family layers,
    e.g. a layer-A `c.fn` and a layer-B `Function`, or a layer-B Block
    and a layer-C Block) compute the same value over a domain of inputs.

    The metadata fields (regime/enforcement/justification) mirror the
    `_Claim` shape so the existing claim plumbing — provers, the verify
    command, the stored-vs-derived discipline — extends uniformly.
    `domain` is the predicate over which the equivalence holds;
    currently always `None` (always-true). A real predicate domain
    (a `PredicateClaim`) is not yet supported.

    The two endpoints are symmetric — `~` is symmetric — but stored as
    `(a_node_id, b_node_id)` for stable JSON ordering. The `kind`
    discriminator stays `"equivalent_to"` to match the design doc's
    naming even though the program-level form is symmetric; the
    asymmetric `EquivalentTo(other_node_id)` form (a claim attached to
    a node that names its counterpart) is sugar over this for future
    authoring tools.
    """
    kind: Literal["equivalent_to"] = "equivalent_to"
    a_node_id: str
    b_node_id: str
    regime: Regime = "axiom"
    enforcement: Enforcement = "trust"
    justification: Justification | None = None
    # Currently always None (every equivalence is "true everywhere");
    # storing it keeps the JSON shape forward-compatible for when a real
    # predicate domain is introduced.
    domain: None = None

    @model_serializer(mode="wrap")
    def _drop_default_metadata(self, handler, info):
        data = handler(self)
        if self.regime == "axiom":
            data.pop("regime", None)
        if self.enforcement == "trust":
            data.pop("enforcement", None)
        if self.justification is None:
            data.pop("justification", None)
        if self.domain is None:
            data.pop("domain", None)
        return data
