"""Tests for the render-side predicate-sugar recognizer.

The recognizer is the inverse of the CLI desugaring: it pattern-matches
canonicalized predicate expressions against known sugar shapes
(non_negative, int_range, return_in_range) and produces the friendly
form. Predicates outside the table fall back to raw expression
rendering.

The contract is: parse(sugar) -> canonicalize -> recognize -> sugar
(round-trip identity). These tests build the canonical predicate that
each sugar form would produce and verify the recognizer names it back.
"""

from __future__ import annotations

from quod.canonicalize import canonicalize
from quod.model import (
    BinOp,
    I32Type,
    IntLit,
    Not,
    ParamRef,
    ReturnRef,
)
from quod.predicate_render import predicate_spans, recognize_predicate


def _x():
    return ParamRef(name="x")


def _y():
    return ParamRef(name="y")


def _i32(v: int):
    return IntLit(type=I32Type(), value=v)


def _spans_text(spans) -> str:
    return "".join(s.text for s in spans)


# ---------- non_negative ----------

def test_non_negative_recognized_from_sge_form():
    e = canonicalize(BinOp(op="sge", lhs=_x(), rhs=_i32(0)))
    assert recognize_predicate(e) == "non_negative(x)"


def test_non_negative_recognized_from_sle_form():
    e = canonicalize(BinOp(op="sle", lhs=_i32(0), rhs=_x()))
    assert recognize_predicate(e) == "non_negative(x)"


# ---------- int_range ----------

def test_int_range_two_bounds():
    e = canonicalize(BinOp(
        op="and",
        lhs=BinOp(op="sge", lhs=_x(), rhs=_i32(0)),
        rhs=BinOp(op="sle", lhs=_x(), rhs=_i32(100)),
    ))
    assert recognize_predicate(e) == "int_range(x, [0, 100])"


def test_int_range_negative_bounds():
    e = canonicalize(BinOp(
        op="and",
        lhs=BinOp(op="sge", lhs=_x(), rhs=_i32(-5)),
        rhs=BinOp(op="sle", lhs=_x(), rhs=_i32(5)),
    ))
    assert recognize_predicate(e) == "int_range(x, [-5, 5])"


def test_int_range_lo_only_with_nonzero():
    # Lower bound only, lo != 0 — should render as int_range, not non_negative.
    e = canonicalize(BinOp(op="sge", lhs=_x(), rhs=_i32(5)))
    assert recognize_predicate(e) == "int_range(x, [5, +inf])"


def test_int_range_hi_only():
    e = canonicalize(BinOp(op="sle", lhs=_x(), rhs=_i32(100)))
    assert recognize_predicate(e) == "int_range(x, [-inf, 100])"


# ---------- return_in_range ----------

def test_return_in_range_two_bounds():
    e = canonicalize(BinOp(
        op="and",
        lhs=BinOp(op="sge", lhs=ReturnRef(), rhs=_i32(0)),
        rhs=BinOp(op="sle", lhs=ReturnRef(), rhs=_i32(100)),
    ))
    assert recognize_predicate(e) == "return_in_range([0, 100])"


def test_return_in_range_lo_only():
    e = canonicalize(BinOp(op="sge", lhs=ReturnRef(), rhs=_i32(0)))
    assert recognize_predicate(e) == "return_in_range([0, +inf])"


def test_return_in_range_hi_only():
    e = canonicalize(BinOp(op="sle", lhs=ReturnRef(), rhs=_i32(100)))
    assert recognize_predicate(e) == "return_in_range([-inf, 100])"


# ---------- Round-trip: hash-equal canonical forms recognize identically ----------

def test_round_trip_non_negative_across_sge_and_sle():
    # Two different surface shapes for the same predicate must canonicalize
    # to the same expr and recognize as the same friendly name.
    via_sge = canonicalize(BinOp(op="sge", lhs=_x(), rhs=_i32(0)))
    via_sle = canonicalize(BinOp(op="sle", lhs=_i32(0), rhs=_x()))
    assert via_sge == via_sle
    assert recognize_predicate(via_sge) == recognize_predicate(via_sle)


def test_round_trip_int_range_with_swapped_conjuncts():
    a = BinOp(op="sge", lhs=_x(), rhs=_i32(0))
    b = BinOp(op="sle", lhs=_x(), rhs=_i32(100))
    via_ab = canonicalize(BinOp(op="and", lhs=a, rhs=b))
    via_ba = canonicalize(BinOp(op="and", lhs=b, rhs=a))
    assert via_ab == via_ba
    assert recognize_predicate(via_ab) == "int_range(x, [0, 100])"


# ---------- Fall-through cases (raw expression rendering) ----------

def test_unrecognized_multi_param_predicate_falls_through():
    e = canonicalize(BinOp(op="slt", lhs=_x(), rhs=_y()))
    assert recognize_predicate(e) is None
    # Raw expression rendering is non-empty and shows the operator.
    text = _spans_text(predicate_spans(e))
    assert "<" in text


def test_unrecognized_negation_falls_through():
    e = canonicalize(Not(operand=BinOp(op="eq", lhs=_x(), rhs=_i32(0))))
    assert recognize_predicate(e) is None
    text = _spans_text(predicate_spans(e))
    assert "!" in text


def test_unrecognized_bound_against_other_param():
    # `x <= y` (variable upper bound) is not a sugar shape.
    e = canonicalize(BinOp(op="sle", lhs=_x(), rhs=_y()))
    assert recognize_predicate(e) is None


def test_predicate_spans_renders_friendly_form_when_matched():
    e = canonicalize(BinOp(op="sge", lhs=_x(), rhs=_i32(0)))
    text = _spans_text(predicate_spans(e))
    assert text == "non_negative(x)"


def test_predicate_spans_falls_back_to_raw_when_unmatched():
    e = canonicalize(BinOp(op="slt", lhs=_x(), rhs=_y()))
    text = _spans_text(predicate_spans(e))
    assert text == "(x < y)"


# ---------- ReturnRef does NOT collapse to non_negative ----------

def test_return_lo_zero_renders_as_return_in_range():
    # Even though lo == 0, the friendly name for ReturnRef bounds is
    # always return_in_range — the sugar table does not have a
    # `return_non_negative` form.
    e = canonicalize(BinOp(op="sge", lhs=ReturnRef(), rhs=_i32(0)))
    assert recognize_predicate(e) == "return_in_range([0, +inf])"
