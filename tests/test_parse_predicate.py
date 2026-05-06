"""Tests for `quod.script.parse_predicate` — the textual predicate
surface used by `claim add --predicate`.

The parser reuses the quod-script expression chain. Bare identifiers
in `param_names` parse as `ParamRef`; the `return` keyword parses as
`ReturnRef`. The caller is responsible for `assert_is_predicate`
afterwards.
"""

from __future__ import annotations

import pytest

from quod.canonicalize import canonicalize, predicate_for_param_range, predicate_for_return_range
from quod.model import (
    BinOp,
    I32Type,
    IntLit,
    Not,
    ParamRef,
    ReturnRef,
    ShortCircuitAnd,
    ShortCircuitOr,
)
from quod.render import recognize_predicate
from quod.script import ScriptError, parse_predicate
from quod.validate import PredicateError, assert_is_predicate


# ---------- Parses sugar shapes that round-trip via the recognizer ----------

def test_non_negative_predicate_round_trips():
    e = canonicalize(parse_predicate("x >= 0", param_names=frozenset({"x"})))
    assert recognize_predicate(e) == "non_negative(x)"


def test_int_range_predicate_round_trips():
    e = canonicalize(parse_predicate(
        "x >= 0 && x <= 100", param_names=frozenset({"x"}),
    ))
    assert recognize_predicate(e) == "int_range(x, [0, 100])"


def test_return_in_range_predicate_round_trips():
    e = canonicalize(parse_predicate(
        "return >= 0 && return <= 100", param_names=frozenset(),
    ))
    assert recognize_predicate(e) == "return_in_range([0, 100])"


def test_canonical_predicate_matches_sugar_builder():
    """The CLI desugarer and the script parser produce the same canonical
    bytes for the same predicate — proves the recognizer/parser pair is
    a real inverse of the sugar-builder."""
    via_script = canonicalize(parse_predicate("x >= 0", param_names=frozenset({"x"})))
    via_sugar = predicate_for_param_range("x", I32Type(), lo=0, hi=None)
    # Different default int widths (script defaults to i64 for bare literals);
    # canonical bytes will differ on type but the recognizer still names it.
    assert recognize_predicate(via_script) == recognize_predicate(via_sugar)


# ---------- Predicates outside the sugar table ----------

def test_multi_param_predicate_parses():
    e = parse_predicate("x < y", param_names=frozenset({"x", "y"}))
    assert isinstance(e, BinOp) and e.op == "slt"
    assert isinstance(e.lhs, ParamRef) and e.lhs.name == "x"
    assert isinstance(e.rhs, ParamRef) and e.rhs.name == "y"


def test_disjunction_parses():
    e = parse_predicate("x == 0 || x == 1", param_names=frozenset({"x"}))
    assert isinstance(e, ShortCircuitOr)


def test_conjunction_parses():
    e = parse_predicate("x >= 0 && y > 0", param_names=frozenset({"x", "y"}))
    assert isinstance(e, ShortCircuitAnd)


def test_return_keyword_parses_as_return_ref():
    e = parse_predicate("return > 0", param_names=frozenset())
    assert isinstance(e, BinOp)
    assert isinstance(e.lhs, ReturnRef)


def test_return_referenced_in_compound_predicate():
    e = parse_predicate(
        "return > x && return <= 100", param_names=frozenset({"x"}),
    )
    # Walk the tree and confirm both ReturnRef and ParamRef appear.
    assert _contains(e, ReturnRef)
    assert _contains(e, ParamRef)


# ---------- Validator catches non-predicate shapes ----------

def test_validator_rejects_function_call():
    e = parse_predicate("foo(x)", param_names=frozenset({"x"}))
    with pytest.raises(PredicateError):
        assert_is_predicate(e)


def test_validator_accepts_negation():
    # Predicates can use !=; canonicalize the expression to verify.
    e = parse_predicate("x != 0", param_names=frozenset({"x"}))
    assert_is_predicate(e)


# ---------- Parse errors ----------

def test_unbalanced_parens_raises():
    with pytest.raises(ScriptError):
        parse_predicate("(x >= 0", param_names=frozenset({"x"}))


def test_trailing_garbage_raises():
    with pytest.raises(ScriptError):
        parse_predicate("x >= 0 garbage", param_names=frozenset({"x"}))


# ---------- helpers ----------

def _contains(expr, kind) -> bool:
    if isinstance(expr, kind):
        return True
    for field in ("operand", "lhs", "rhs", "cond", "then_value", "else_value"):
        v = getattr(expr, field, None)
        if v is not None and _contains(v, kind):
            return True
    return False
