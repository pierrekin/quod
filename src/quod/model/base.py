"""Base node + ID minters shared across the model package."""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, ConfigDict


class _Node(BaseModel):
    # strict=True: no silent coercion. frozen=True: graph is read-only;
    # mutators must build new instances via model_copy.
    model_config = ConfigDict(strict=True, frozen=True)


def _mint_block_id() -> str:
    return f"@blk_{uuid4().hex[:12]}"


def _mint_function_id() -> str:
    return f"@fn_{uuid4().hex[:12]}"


def _mint_node_id(prefix: str) -> str:
    """Mint an opaque node ID. Used as a `default_factory` so every new
    layer-A or c-extension node gets a stable ID at construction. The
    `prefix` is a short tag indicating the node kind (e.g. "cunit",
    "cfn", "cfor") — no semantic load, only useful for hand-debugging."""
    return f"@{prefix}_{uuid4().hex[:12]}"
