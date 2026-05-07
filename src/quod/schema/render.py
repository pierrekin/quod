"""Schema rendering — turn a kind name (or category) into condensed text.

Composes the per-stratum catalogs (core / layer-A / layer-B) into a single
`_KIND_INFO` map, and walks Pydantic field metadata to render schemas with
hand-curated examples.
"""

from __future__ import annotations

import json
import types
import typing
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from quod import model

from quod.schema.aliases import _matches_alias
from quod.schema.core import _CORE_CATALOG
from quod.schema.layer_a import _LAYER_A_CATALOG
from quod.schema.layer_b import _LAYER_B_CATALOG
from quod.schema.categories import _CATEGORIES, _category_of


_KIND_INFO: dict[str, dict[str, Any]] = {
    **_CORE_CATALOG,
    **_LAYER_A_CATALOG,
    **_LAYER_B_CATALOG,
}


def _render_type(annotation: Any) -> str:
    """Render a Python type annotation as a short human label."""
    # Recognized discriminated-union aliases — render as alias name and stop.
    alias = _matches_alias(annotation)
    if alias is not None:
        return alias

    # Strip Annotated[...] metadata (e.g. `Annotated[Union[...], Field(discriminator=...)]`).
    if hasattr(annotation, "__metadata__"):
        annotation = annotation.__origin__

    # Forward refs ("Expr" string annotations not yet resolved): use the name.
    if isinstance(annotation, typing.ForwardRef):
        # Pydantic sometimes stores a doubly-quoted name — strip stray quotes.
        return annotation.__forward_arg__.strip("'\"")

    if annotation is type(None):
        return "null"

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Union / Optional. typing.Union and `X | Y` (PEP 604 / types.UnionType) both apply.
    if origin is typing.Union or origin is types.UnionType:
        has_none = type(None) in args
        non_none = [a for a in args if a is not type(None)]
        rendered = [_render_type(a) for a in non_none]
        joined = " | ".join(rendered)
        return f"{joined}?" if has_none else joined

    # Literal["a", "b", ...] — render as `'a' | 'b' | ...` for closed-set fields.
    if origin is typing.Literal:
        return " | ".join(repr(a) for a in args)

    # tuple[X, ...] is the canonical container shape we use throughout.
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return f"list[{_render_type(args[0])}]"
        return f"tuple[{', '.join(_render_type(a) for a in args)}]"
    if origin is list:
        return f"list[{_render_type(args[0])}]" if args else "list"

    # Plain types.
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _render_default(default: Any) -> str:
    if default is None:
        return "null"
    if default == ():
        return "[]"
    return repr(default)


def _resolve_name(name: str) -> str:
    """Accept canonical kinds, also aliases like 'function' → 'Function'."""
    if name in _KIND_INFO:
        return name
    # Try case-insensitive match for top-level types.
    for k in _KIND_INFO:
        if k.lower() == name.lower():
            return k
    return name  # caller decides how to error


def _resolved_hints(cls: type[BaseModel]) -> dict[str, Any]:
    """Resolve forward refs in `cls`'s annotations against `quod.model`'s globals.

    Pydantic stores raw annotations (often ForwardRefs containing strings like
    `tuple['Statement', ...]`); typing.get_type_hints walks the type and resolves
    those refs by looking up names in the provided globals.
    """
    return typing.get_type_hints(
        cls, globalns=vars(model), include_extras=True,
    )


def render_kind(name: str) -> str:
    """Render a single kind's schema as a condensed text block."""
    name = _resolve_name(name)
    if name not in _KIND_INFO:
        known = ", ".join(sorted(_KIND_INFO.keys()))
        raise KeyError(f"unknown kind {name!r}. Known kinds:\n  {known}")
    info = _KIND_INFO[name]
    cls: type[BaseModel] = info["class"]
    cat = _category_of(name) or "?"
    lines = [f"{name} ({cat}) — {info['summary']}"]
    field_descriptions = info.get("field_descriptions", {})
    hints = _resolved_hints(cls)
    for fname, finfo in cls.model_fields.items():
        if fname == "kind":
            continue
        annotation = hints.get(fname, finfo.annotation)
        ty = _render_type(annotation)
        if finfo.is_required():
            req = "required"
        else:
            req = f"optional, default={_render_default(finfo.default)}"
        desc = field_descriptions.get(fname, "")
        suffix = f" — {desc}" if desc else ""
        lines.append(f"  {fname} ({ty}, {req}){suffix}")
    lines.append("example:")
    lines.append(f"  {json.dumps(info['example'], separators=(', ', ': '))}")
    if info.get("see_also"):
        lines.append(f"see also: {', '.join(info['see_also'])}")
    return "\n".join(lines)


def render_category(cat: str) -> str:
    """Render all kinds in a category as one-liners."""
    if cat not in _CATEGORIES:
        known = ", ".join(_CATEGORIES.keys())
        raise KeyError(f"unknown category {cat!r}. Known categories: {known}")
    lines = [f"category: {cat}"]
    for name in _CATEGORIES[cat]:
        info = _KIND_INFO[name]
        lines.append(f"  {name} — {info['summary']}")
    lines.append(f"\nFor full schema of one kind: quod schema KIND")
    return "\n".join(lines)


def render_categories() -> str:
    """Render all categories with kind counts."""
    lines = ["categories:"]
    for cat, kinds in _CATEGORIES.items():
        lines.append(f"  {cat} ({len(kinds)} kinds): {', '.join(kinds)}")
    lines.append("\nFor a category overview: quod schema --category CAT")
    lines.append("For a kind's full schema: quod schema KIND")
    return "\n".join(lines)
