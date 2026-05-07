"""Discriminated-union alias detection for schema rendering.

When a field is typed as one of these unions, render its alias name instead
of expanding the full member list — keeps output terse and points the reader
at the right category for further lookup.
"""

from __future__ import annotations

import types
import typing
from typing import Any, get_args, get_origin

from quod import model


_ALIASES = [
    (lambda: model.Expr, "Expr"),
    (lambda: model.Statement, "Statement"),
    (lambda: model.Type, "Type"),
    (lambda: model.IntType, "IntType"),
    (lambda: model.Justification, "Justification"),
    (lambda: model.Claim, "Claim"),
]


def _union_args(t: Any) -> tuple | None:
    """Return Union args if `t` is a Union/Annotated[Union]; else None."""
    if hasattr(t, "__metadata__"):
        t = t.__origin__
    origin = get_origin(t)
    if origin is typing.Union or origin is types.UnionType:
        return tuple(get_args(t))
    return None


def _matches_alias(annotation: Any) -> str | None:
    """Match `annotation` against a registered alias by union-arg set."""
    a_args = _union_args(annotation)
    if a_args is None:
        return None
    a_set = frozenset(a_args)
    for alias_fn, name in _ALIASES:
        b_args = _union_args(alias_fn())
        if b_args is not None and frozenset(b_args) == a_set:
            return name
    return None
