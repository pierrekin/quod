"""Layer B — c.* extension nodes.

Constructs core quod can't represent on its own. Lowered to core by
`lower/c_family.py`. `lower.py` refuses to consume them directly —
the c-family pass must run first.
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import Field, model_serializer

from quod.model.base import _Node, _mint_node_id
from quod.model.expressions import Expr
from quod.model.statements import Block, Statement


class CScopedBlock(_Node):
    """C-style block wrapper. `block` is the inner `core.Block` that
    edges anchor on; the wrapper carries family-specific scope semantics
    (which decls die at the closing brace). Lowered by c-family lowering
    to its inner block — the wrapper is discarded by the time `lower.py`
    sees the program.

    `scope_locals` records the names of locals whose scope ends with
    this block. Currently a tuple of names; richer scope metadata
    (types, kill points within the block) can accumulate as lowering
    rules grow.
    """
    kind: Literal["c.scoped_block"] = "c.scoped_block"
    id: str = Field(default_factory=lambda: _mint_node_id("cscope"))
    block: Block
    scope_locals: tuple[str, ...] = ()

    @model_serializer(mode="wrap")
    def _drop_default_metadata(self, handler, info):
        data = handler(self)
        if not self.scope_locals:
            data.pop("scope_locals", None)
        return data


# Body slots that may host either a plain `Block` (after c-family
# lowering, or for non-c-derived programs) or a `CScopedBlock` wrapper
# (in layer-B programs). Pydantic's smart union picks by structure:
# Block has no `kind` field; CScopedBlock's `kind` discriminates.
BlockOrScoped = Union[Block, CScopedBlock]


class CStyleFor(_Node):
    """`for (init; cond; inc) { body }` — the C for loop transcribed
    into layer B. Distinct from core's `For` (which is bounded
    iteration over a half-open integer range). The c-family lowering
    rule transforms `c.for_general` into `Let + While + Assign`; the
    rule's equivalence theorem is proved once and cited via
    `FamilyLowering("c.for_general")` justifications.
    """
    kind: Literal["c.for_general"] = "c.for_general"
    id: str = Field(default_factory=lambda: _mint_node_id("cfor_general"))
    init: "Statement | None" = None
    cond: "Expr | None" = None  # i1 when present
    inc: "Statement | None" = None
    body: BlockOrScoped
