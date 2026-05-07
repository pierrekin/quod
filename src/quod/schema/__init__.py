"""Schema introspection for the `quod schema` CLI / `quod_schema` tool.

Renders a condensed text description of any node kind: required/optional
fields with types, plus a hand-curated minimal example. Field info is read
from the pydantic model so types stay in sync; summaries and examples are
hand-curated next to the model.

Three query modes:
    schema()                 -> list categories + one-line summary each
    schema(category="...")   -> list kinds in the category, one-liner each
    schema(kind="quod.let")  -> full per-kind schema with example

Categories: expression, statement, type, claim, justification, program.
"""

from __future__ import annotations

from quod.schema.render import render_categories, render_category, render_kind


__all__ = ["render_categories", "render_category", "render_kind"]
