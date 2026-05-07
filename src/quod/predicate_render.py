"""Predicate-sugar recognizer for the human-readable program view.

Inverse of `quod.canonicalize`'s sugar predicate builders. Operates on
canonicalized predicate expressions: identifies the structural shapes
emitted by the named claim shapes (non_negative, int_range,
return_in_range) and renders them with their friendly form. Predicates
outside the sugar table fall back to raw expression rendering via
`quod.render._expr_spans`.

Lives next to the predicate domain rather than inside `render.py` so
that the recognizer pairs visibly with the builder side in
`canonicalize.py`. Render-pipeline machinery (Span/Line/themes,
expression spans, statement lines, row builders) stays in `render.py`.
"""

from __future__ import annotations

from quod.model import BinOp, IntLit, ParamRef, ReturnRef
from quod.render import Span, _expr_spans


# Marker used in the bound-tuple to indicate ReturnRef rather than a
# named parameter. Not a valid quod identifier (angle brackets) so it
# cannot collide with a real param.
_RETURN_MARKER = "<return>"


def recognize_predicate(expr) -> str | None:
    """Return the friendly sugar name for a canonical predicate, or None.

    Recognized shapes (after `quod.canonicalize.canonicalize`):
        BinOp(sle, IntLit(0), ParamRef(p))         -> 'non_negative(p)'
        BinOp(sle, IntLit(lo), <Ref>)              -> 'int_range(p, [lo, +inf])'
        BinOp(sle, <Ref>, IntLit(hi))              -> 'int_range(p, [-inf, hi])'
        BinOp(and, sle(lo, x), sle(x, hi))         -> 'int_range(p, [lo, hi])'
        Same shapes over ReturnRef                 -> 'return_in_range(...)'
    """
    spans = _recognize_predicate_spans(expr)
    if spans is None:
        return None
    return "".join(s.text for s in spans)


def predicate_spans(expr) -> tuple[Span, ...]:
    """Span rendering of a `PredicateClaim.expr`.

    Recognizes canonical sugar shapes and renders them as the friendly
    form (`non_negative`, `int_range`, `return_in_range`). Predicates
    outside the sugar table render as the raw expression.
    """
    sugar = _recognize_predicate_spans(expr)
    if sugar is not None:
        return sugar
    return _expr_spans(expr)


def _recognize_predicate_spans(expr) -> tuple[Span, ...] | None:
    bound = _match_single_bound(expr)
    if bound is not None:
        ref_name, lo, hi = bound
        return _sugar_spans(ref_name, lo, hi)
    if isinstance(expr, BinOp) and expr.op == "and":
        left = _match_single_bound(expr.lhs)
        right = _match_single_bound(expr.rhs)
        if left is not None and right is not None and left[0] == right[0]:
            l_ref, l_lo, l_hi = left
            r_lo, r_hi = right[1], right[2]
            # Combine if one side is lo-only and the other hi-only.
            if l_lo is not None and l_hi is None and r_lo is None and r_hi is not None:
                return _sugar_spans(l_ref, l_lo, r_hi)
            if l_lo is None and l_hi is not None and r_lo is not None and r_hi is None:
                return _sugar_spans(l_ref, r_lo, l_hi)
    return None


def _match_single_bound(expr) -> tuple[str, int | None, int | None] | None:
    """Match a single-bound canonical comparison.

    Returns `(ref_name, lo, hi)` where exactly one of `lo`/`hi` is non-None,
    `ref_name` is the param name (or `_RETURN_MARKER` for ReturnRef), or
    None if no match.
    """
    if not isinstance(expr, BinOp) or expr.op != "sle":
        return None
    lhs, rhs = expr.lhs, expr.rhs
    # IntLit(lo) <= ref  -- lower-bound form
    if isinstance(lhs, IntLit) and (rname := _ref_name(rhs)) is not None:
        return (rname, lhs.value, None)
    # ref <= IntLit(hi)  -- upper-bound form
    if isinstance(rhs, IntLit) and (lname := _ref_name(lhs)) is not None:
        return (lname, None, rhs.value)
    return None


def _ref_name(expr) -> str | None:
    if isinstance(expr, ParamRef):
        return expr.name
    if isinstance(expr, ReturnRef):
        return _RETURN_MARKER
    return None


def _sugar_spans(ref_name: str, lo: int | None, hi: int | None) -> tuple[Span, ...]:
    if ref_name == _RETURN_MARKER:
        return (
            Span("return_in_range", "meta_label"), Span("([", "punct"),
            Span("-inf" if lo is None else str(lo), "literal_int"),
            Span(", ", "punct"),
            Span("+inf" if hi is None else str(hi), "literal_int"),
            Span("])", "punct"),
        )
    if lo == 0 and hi is None:
        return (
            Span("non_negative", "meta_label"), Span("(", "punct"),
            Span(ref_name, "param"), Span(")", "punct"),
        )
    return (
        Span("int_range", "meta_label"), Span("(", "punct"),
        Span(ref_name, "param"), Span(", [", "punct"),
        Span("-inf" if lo is None else str(lo), "literal_int"),
        Span(", ", "punct"),
        Span("+inf" if hi is None else str(hi), "literal_int"),
        Span("])", "punct"),
    )
