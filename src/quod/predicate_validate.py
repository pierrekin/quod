"""Predicate-vocabulary validation.

A `PredicateClaim.expr` is a structurally-restricted `Expr`: i1-typed
at the top level, side-effect-free, references only params or
`ReturnRef`. The model-level type is `Expr` for graph-edge uniformity
(one expression vocabulary, not two), so the structural constraint is
enforced by this helper rather than the type system.

Used at PredicateClaim construction time and during the validate pass
when a Function's claims include predicate forms.
"""
from __future__ import annotations

from quod.model import (
    BinOp,
    IntLit,
    Not,
    ParamRef,
    ReturnRef,
    ShortCircuitAnd,
    ShortCircuitOr,
)


class PredicateError(ValueError):
    """Raised when an expression cannot be a predicate (impure op,
    aggregate access, dangling reference, etc.). The message names the
    offending node kind."""


def assert_is_predicate(expr) -> None:
    """Raise `PredicateError` if `expr` is not a valid predicate body.

    Allowed: IntLit, ParamRef, ReturnRef, BinOp, ShortCircuitOr/And,
    Not. Forbidden: Calls, aggregate access, memory ops, locals, or
    anything else with side effects or non-predicate semantics.
    """
    _walk_predicate(expr)


def _walk_predicate(expr) -> None:
    if isinstance(expr, IntLit):
        return
    if isinstance(expr, ParamRef):
        return
    if isinstance(expr, ReturnRef):
        return
    if isinstance(expr, Not):
        _walk_predicate(expr.operand)
        return
    if isinstance(expr, BinOp):
        _walk_predicate(expr.lhs)
        _walk_predicate(expr.rhs)
        return
    if isinstance(expr, (ShortCircuitOr, ShortCircuitAnd)):
        _walk_predicate(expr.lhs)
        _walk_predicate(expr.rhs)
        return
    raise PredicateError(
        f"predicate cannot contain {type(expr).__name__} — predicates "
        f"must be side-effect-free and reference only params or ReturnRef"
    )
