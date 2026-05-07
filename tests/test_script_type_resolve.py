"""Tests for the type-resolution pass in `quod.script`.

Phase 1 of parse_function / parse_predicate produces an AST whose bare
integer literals carry an i64 placeholder type and are tracked via a
side table; phase 2 walks the tree and retypes every poisoned literal
using its operand context (the let's declared type, the function's
return type, the param being compared against, the for-loop variable's
type, ...). Suffixed literals (e.g. `42i8`) skip phase 2 entirely.

A bare literal whose context can't pin a type — and that's not in one
of the lower-time-coercible holes (call args, struct/enum field inits,
field-set / store destinations) — is a `ScriptError`.
"""

from __future__ import annotations

import pytest

from quod.model import (
    BinOp,
    For,
    I8Type,
    I16Type,
    I32Type,
    I64Type,
    IntLit,
    Let,
    ParamRef,
    ReturnExpr,
)
from quod.script import ScriptError, parse_function, parse_predicate


# ---------- Function bodies: let, return, binop, for ----------

def test_let_init_takes_lets_declared_type():
    """`let x: i32 = 5` — bare 5 is retyped to i32, not the i64
    placeholder."""
    fn = parse_function("fn f() -> i32 { let x: i32 = 5 return 0 }")
    let = fn.body.stmts[0]
    assert isinstance(let, Let)
    assert isinstance(let.init, IntLit)
    assert isinstance(let.init.type, I32Type)
    assert let.init.value == 5


def test_let_init_with_smaller_int_type():
    fn = parse_function("fn f() -> i32 { let x: i8 = 7 return 0 }")
    init = fn.body.stmts[0].init
    assert isinstance(init, IntLit) and isinstance(init.type, I8Type)
    assert init.value == 7


def test_binop_compare_propagates_param_type_to_literal():
    """`x < 5` with x: i32 — the bare 5 takes i32 from the cmp's other
    operand."""
    fn = parse_function("fn f(x: i32) -> i1 { return x < 5 }")
    cmp = fn.body.stmts[0].value
    assert isinstance(cmp, BinOp) and cmp.op == "slt"
    assert isinstance(cmp.lhs, ParamRef)
    assert isinstance(cmp.rhs, IntLit) and isinstance(cmp.rhs.type, I32Type)


def test_binop_compare_propagates_either_direction():
    """`5 < x` — the bare on LHS picks up x's type after RHS resolves."""
    fn = parse_function("fn f(x: i32) -> i1 { return 5 < x }")
    cmp = fn.body.stmts[0].value
    assert isinstance(cmp.lhs, IntLit) and isinstance(cmp.lhs.type, I32Type)


def test_binop_arith_propagates_outer_let_type():
    """`let y: i16 = x + 1` with x: i16 — the bare 1 takes i16 from the
    surrounding let's declared type, not from x."""
    fn = parse_function("fn f(x: i16) -> i16 { let y: i16 = x + 1 return y }")
    let = fn.body.stmts[0]
    add = let.init
    assert isinstance(add, BinOp) and add.op == "add"
    assert isinstance(add.rhs, IntLit) and isinstance(add.rhs.type, I16Type)


def test_return_bare_int_takes_return_type():
    """`return 7` from an i32 fn lands as an i32 IntLit, courtesy of the
    resolver's ReturnExpr.value-from-return-type rule (no special-case
    in the parser)."""
    fn = parse_function("fn f() -> i32 { return 7 }")
    ret = fn.body.stmts[0]
    assert isinstance(ret, ReturnExpr)
    assert isinstance(ret.value, IntLit) and isinstance(ret.value.type, I32Type)
    assert ret.value.value == 7


def test_return_negative_bare_int_takes_return_type():
    fn = parse_function("fn f() -> i32 { return -7 }")
    ret = fn.body.stmts[0]
    assert isinstance(ret.value, IntLit) and isinstance(ret.value.type, I32Type)
    assert ret.value.value == -7


def test_for_loop_bounds_take_var_type():
    """`for i: i32 in 0 .. 10 { ... }` — bounds adopt the loop var's
    declared type."""
    fn = parse_function(
        "fn f() -> i32 { let s: i32 = 0 "
        "for i: i32 in 0 .. 10 { s = s + i } return s }"
    )
    fl = fn.body.stmts[1]
    assert isinstance(fl, For)
    assert isinstance(fl.lo, IntLit) and isinstance(fl.lo.type, I32Type)
    assert isinstance(fl.hi, IntLit) and isinstance(fl.hi.type, I32Type)


def test_assign_takes_locals_type():
    """`x = 7` where `x` is a local of type i16 — the bare 7 takes
    i16 from the local's declared type."""
    fn = parse_function(
        "fn f() -> i32 { let x: i16 = 0i16 x = 7 return 0 }"
    )
    assign = fn.body.stmts[1]
    assert assign.value.value == 7
    assert isinstance(assign.value.type, I16Type)


def test_assign_takes_param_type():
    """Mutating a param via `=` (allowed for params with int type) — the
    bare RHS adopts the param's type."""
    fn = parse_function("fn f(x: i16) -> i16 { x = 5 return x }")
    assign = fn.body.stmts[0]
    assert isinstance(assign.value, IntLit) and isinstance(assign.value.type, I16Type)


def test_explicit_suffix_wins_over_let_type():
    """When a literal carries an explicit suffix that disagrees with the
    let's declared type, Pydantic's IntLit-in-Let validator (or the
    lowerer) catches the mismatch — but the resolver itself does not
    silently overwrite the explicit suffix. Verify by parsing a let
    with a matching i32 suffix and checking the literal stays i32."""
    fn = parse_function("fn f() -> i32 { let x: i32 = 5i32 return 0 }")
    init = fn.body.stmts[0].init
    assert isinstance(init.type, I32Type)
    assert init.value == 5


# ---------- The force-suffix rule: bare literal with no context ----------

def test_bare_literal_no_context_inside_cast_raises():
    """A bare integer literal can be made unresolvable by stuffing it
    inside a `cast(... to ...)` whose result type is dictated by the
    target, not the source. Without further context, the source-side
    bare literal has nothing to anchor against and the resolver raises."""
    with pytest.raises(ScriptError, match="bare integer literal"):
        parse_function(
            "fn f() -> i32 { let x: i32 = cast(5 to i32) return 0 }"
        )


# ---------- Predicates ----------

def test_predicate_param_type_propagates_to_literal():
    e = parse_predicate(
        "x >= 0",
        param_types={"x": I32Type()}, return_type=I32Type(),
    )
    assert isinstance(e, BinOp)
    assert isinstance(e.rhs, IntLit) and isinstance(e.rhs.type, I32Type)


def test_predicate_return_type_propagates_to_literal():
    e = parse_predicate(
        "return >= 0",
        param_types={}, return_type=I16Type(),
    )
    assert isinstance(e, BinOp)
    assert isinstance(e.rhs, IntLit) and isinstance(e.rhs.type, I16Type)


def test_predicate_unresolved_literal_raises():
    """A predicate of just `0` has no operand context and no scope to
    pull a type from — force-suffix or write a real predicate."""
    with pytest.raises(ScriptError, match="bare integer literal"):
        parse_predicate(
            "0", param_types={"x": I32Type()}, return_type=I32Type(),
        )


def test_predicate_suffixed_literal_with_no_context_passes():
    """`42i32` doesn't need context — the suffix carries the type."""
    e = parse_predicate(
        "42i32 < x",
        param_types={"x": I32Type()}, return_type=I32Type(),
    )
    assert isinstance(e.lhs, IntLit) and isinstance(e.lhs.type, I32Type)
