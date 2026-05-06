"""Tests for `quod.canonicalize` — the predicate-expression normal form.

The canonicalizer's only contract is hash stability: two semantically-
equal predicates must produce identical bytes. Tests below verify
each rewrite rule individually, then verify idempotence (a rewritten
expression is unchanged by a second pass) and hash stability across
surface variants.
"""

from __future__ import annotations

from quod.canonicalize import canonicalize
from quod.model import (
    BinOp,
    I1Type,
    I32Type,
    IntLit,
    Not,
    ParamRef,
    ReturnRef,
    ShortCircuitAnd,
    ShortCircuitOr,
)


# ---------- Helpers ----------

def _x():
    return ParamRef(name="x")


def _y():
    return ParamRef(name="y")


def _i32(v: int):
    return IntLit(type=I32Type(), value=v)


def _bool(v: int):
    return IntLit(type=I1Type(), value=v)


def _hash_bytes(expr) -> str:
    return expr.model_dump_json()


# ---------- Comparator-direction normalization ----------

def test_sgt_rewrites_to_slt():
    a, b = _x(), _i32(0)
    sgt = BinOp(op="sgt", lhs=a, rhs=b)
    slt = BinOp(op="slt", lhs=b, rhs=a)
    assert canonicalize(sgt) == canonicalize(slt)


def test_sge_rewrites_to_sle():
    a, b = _x(), _i32(0)
    sge = BinOp(op="sge", lhs=a, rhs=b)
    sle = BinOp(op="sle", lhs=b, rhs=a)
    assert canonicalize(sge) == canonicalize(sle)


def test_ugt_rewrites_to_ult():
    a, b = _x(), _i32(5)
    ugt = BinOp(op="ugt", lhs=a, rhs=b)
    ult = BinOp(op="ult", lhs=b, rhs=a)
    assert canonicalize(ugt) == canonicalize(ult)


def test_uge_rewrites_to_ule():
    a, b = _x(), _i32(5)
    uge = BinOp(op="uge", lhs=a, rhs=b)
    ule = BinOp(op="ule", lhs=b, rhs=a)
    assert canonicalize(uge) == canonicalize(ule)


# ---------- Commutative-op operand sort ----------

def test_and_operands_sort_deterministically():
    a = BinOp(op="sle", lhs=_i32(0), rhs=_x())
    b = BinOp(op="sle", lhs=_x(), rhs=_i32(100))
    e1 = canonicalize(BinOp(op="and", lhs=a, rhs=b))
    e2 = canonicalize(BinOp(op="and", lhs=b, rhs=a))
    assert e1 == e2


def test_or_operands_sort_deterministically():
    a = BinOp(op="eq", lhs=_x(), rhs=_i32(0))
    b = BinOp(op="eq", lhs=_x(), rhs=_i32(1))
    e1 = canonicalize(BinOp(op="or", lhs=a, rhs=b))
    e2 = canonicalize(BinOp(op="or", lhs=b, rhs=a))
    assert e1 == e2


def test_eq_operands_sort_commutatively():
    e1 = canonicalize(BinOp(op="eq", lhs=_x(), rhs=_i32(0)))
    e2 = canonicalize(BinOp(op="eq", lhs=_i32(0), rhs=_x()))
    assert e1 == e2


def test_non_commutative_ops_preserve_order():
    # `slt` is not commutative — sort must NOT swap operands.
    e = canonicalize(BinOp(op="slt", lhs=_x(), rhs=_y()))
    assert e.lhs == _x()
    assert e.rhs == _y()


# ---------- Short-circuit collapse ----------

def test_short_circuit_or_becomes_bitwise_or():
    a = BinOp(op="eq", lhs=_x(), rhs=_i32(0))
    b = BinOp(op="eq", lhs=_y(), rhs=_i32(0))
    sc = canonicalize(ShortCircuitOr(lhs=a, rhs=b))
    plain = canonicalize(BinOp(op="or", lhs=a, rhs=b))
    assert sc == plain


def test_short_circuit_and_becomes_bitwise_and():
    a = BinOp(op="sge", lhs=_x(), rhs=_i32(0))
    b = BinOp(op="sle", lhs=_x(), rhs=_i32(100))
    sc = canonicalize(ShortCircuitAnd(lhs=a, rhs=b))
    plain = canonicalize(BinOp(op="and", lhs=a, rhs=b))
    assert sc == plain


# ---------- Trivial Boolean folds ----------

def test_double_negation_folds():
    inner = BinOp(op="eq", lhs=_x(), rhs=_i32(0))
    e = Not(operand=Not(operand=inner))
    assert canonicalize(e) == canonicalize(inner)


def test_not_true_folds_to_false():
    assert canonicalize(Not(operand=_bool(1))) == _bool(0)


def test_not_false_folds_to_true():
    assert canonicalize(Not(operand=_bool(0))) == _bool(1)


def test_and_x_true_folds_to_x():
    inner = BinOp(op="eq", lhs=_x(), rhs=_i32(0))
    canon_inner = canonicalize(inner)
    e = BinOp(op="and", lhs=inner, rhs=_bool(1))
    assert canonicalize(e) == canon_inner


def test_and_x_false_folds_to_false():
    inner = BinOp(op="eq", lhs=_x(), rhs=_i32(0))
    e = BinOp(op="and", lhs=inner, rhs=_bool(0))
    assert canonicalize(e) == _bool(0)


def test_or_x_true_folds_to_true():
    inner = BinOp(op="eq", lhs=_x(), rhs=_i32(0))
    e = BinOp(op="or", lhs=inner, rhs=_bool(1))
    assert canonicalize(e) == _bool(1)


def test_or_x_false_folds_to_x():
    inner = BinOp(op="eq", lhs=_x(), rhs=_i32(0))
    canon_inner = canonicalize(inner)
    e = BinOp(op="or", lhs=inner, rhs=_bool(0))
    assert canonicalize(e) == canon_inner


def test_xor_x_false_folds_to_x():
    inner = BinOp(op="eq", lhs=_x(), rhs=_i32(0))
    canon_inner = canonicalize(inner)
    e = BinOp(op="xor", lhs=inner, rhs=_bool(0))
    assert canonicalize(e) == canon_inner


# ---------- Idempotence ----------

def test_idempotent_on_simple_predicate():
    e = BinOp(op="sge", lhs=_x(), rhs=_i32(0))
    once = canonicalize(e)
    twice = canonicalize(once)
    assert once == twice


def test_idempotent_on_int_range_shape():
    e = BinOp(
        op="and",
        lhs=BinOp(op="sge", lhs=_x(), rhs=_i32(0)),
        rhs=BinOp(op="sle", lhs=_x(), rhs=_i32(100)),
    )
    once = canonicalize(e)
    twice = canonicalize(once)
    assert once == twice


def test_idempotent_on_negation():
    e = Not(operand=BinOp(op="eq", lhs=_x(), rhs=_i32(0)))
    once = canonicalize(e)
    twice = canonicalize(once)
    assert once == twice


def test_idempotent_on_short_circuit():
    e = ShortCircuitAnd(
        lhs=BinOp(op="sge", lhs=_x(), rhs=_i32(0)),
        rhs=BinOp(op="sle", lhs=_x(), rhs=_i32(100)),
    )
    once = canonicalize(e)
    twice = canonicalize(once)
    assert once == twice


# ---------- Hash stability across surface variants ----------

def test_hash_stable_across_comparator_direction():
    h1 = _hash_bytes(canonicalize(BinOp(op="sgt", lhs=_x(), rhs=_i32(0))))
    h2 = _hash_bytes(canonicalize(BinOp(op="slt", lhs=_i32(0), rhs=_x())))
    assert h1 == h2


def test_hash_stable_across_int_range_surface_variants():
    # Sugar might compose either as (x >= lo) AND (x <= hi) or as
    # (lo <= x) AND (x <= hi); both must hash the same canonically.
    forms = [
        BinOp(op="and",
              lhs=BinOp(op="sge", lhs=_x(), rhs=_i32(0)),
              rhs=BinOp(op="sle", lhs=_x(), rhs=_i32(100))),
        BinOp(op="and",
              lhs=BinOp(op="sle", lhs=_i32(0), rhs=_x()),
              rhs=BinOp(op="sle", lhs=_x(), rhs=_i32(100))),
        BinOp(op="and",
              lhs=BinOp(op="sle", lhs=_x(), rhs=_i32(100)),
              rhs=BinOp(op="sle", lhs=_i32(0), rhs=_x())),
    ]
    hashes = {_hash_bytes(canonicalize(f)) for f in forms}
    assert len(hashes) == 1, f"expected one canonical form, got {hashes}"


def test_hash_stable_across_short_circuit_vs_bitwise():
    a = BinOp(op="sge", lhs=_x(), rhs=_i32(0))
    b = BinOp(op="sle", lhs=_x(), rhs=_i32(100))
    h1 = _hash_bytes(canonicalize(ShortCircuitAnd(lhs=a, rhs=b)))
    h2 = _hash_bytes(canonicalize(BinOp(op="and", lhs=a, rhs=b)))
    assert h1 == h2


# ---------- Recursion through structure ----------

def test_recurses_into_binop_children():
    # The outer BinOp is non-commutative (slt) so won't sort, but the
    # inner sgt should rewrite to slt during recursion.
    e = BinOp(
        op="slt",
        lhs=BinOp(op="sgt", lhs=_x(), rhs=_i32(0)),
        rhs=_bool(1),
    )
    canon = canonicalize(e)
    inner = canon.lhs
    # Inner sgt(x, 0) becomes slt(0, x) after normalization.
    assert isinstance(inner, BinOp) and inner.op == "slt"


def test_recurses_into_not_operand():
    e = Not(operand=BinOp(op="sgt", lhs=_x(), rhs=_i32(0)))
    canon = canonicalize(e)
    assert isinstance(canon, Not)
    assert canon.operand.op == "slt"


def test_return_ref_treated_as_a_term():
    ret = ReturnRef()
    e = BinOp(op="sge", lhs=ret, rhs=_i32(0))
    canon = canonicalize(e)
    # Becomes sle(0, return_ref).
    assert canon.op == "sle"
    assert isinstance(canon.lhs, IntLit) and canon.lhs.value == 0
    assert isinstance(canon.rhs, ReturnRef)
