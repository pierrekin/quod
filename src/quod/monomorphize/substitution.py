"""Type-parameter substitution.

Replaces TypeParamRefs in a Type or AST with concrete Types. The
substitute-only walker thunks bind to `quod.traversal._substitute_in_*`
shared with model.py's ImplDef Self substitution.
"""

from __future__ import annotations

from ..model import EnumType, StructType, TypeParamRef
from ..traversal import substitute_in_expr as _substitute_in_expr_walker
from ..traversal import substitute_in_stmt as _substitute_in_stmt_walker


def _substitute_type(t, sub: dict[str, object]):
    """Replace any TypeParamRef in `t` via `sub`. Recursively descends
    into StructType/EnumType `type_args`."""
    if isinstance(t, TypeParamRef):
        if t.name not in sub:
            raise ValueError(
                f"unbound type parameter {t.name!r} (available: {sorted(sub)})"
            )
        return sub[t.name]
    if isinstance(t, StructType):
        if not t.type_args:
            return t
        new_args = tuple(_substitute_type(a, sub) for a in t.type_args)
        return t.model_copy(update={"type_args": new_args})
    if isinstance(t, EnumType):
        if not t.type_args:
            return t
        new_args = tuple(_substitute_type(a, sub) for a in t.type_args)
        return t.model_copy(update={"type_args": new_args})
    # Scalar / void: no params possible.
    return t


def _substitute_in_expr(expr, sub):
    return _substitute_in_expr_walker(expr, lambda t: _substitute_type(t, sub))


def _substitute_in_stmt(stmt, sub):
    return _substitute_in_stmt_walker(stmt, lambda t: _substitute_type(t, sub))
