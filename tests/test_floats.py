"""End-to-end tests for FloatLit, FNeg, and float-typed `BinOp`.

Commit 1 of the floats roadmap. Cast's float arms (sitofp / uitofp /
fpext / fptrunc / fptosi.sat / fptoui.sat) ride on this — they were
declared in Commit 0 but unreachable until FloatLit existed.

Float ops follow strict IEEE 754. NaN comparisons go via LLVM
ordered preds + `une` for `!=` so `NaN != NaN` returns true. FNeg
flips the sign bit unconditionally (distinct from `0.0 - x`, which
returns +0.0 for -0.0 input).
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from quod.lower import compile_program
from quod.model import (
    BinOp,
    Block,
    Call,
    Cast,
    ExprStmt,
    ExternFunction,
    F32Type,
    F64Type,
    FloatLit,
    FNeg,
    Function,
    I8PtrType,
    I32Type,
    I64Type,
    IntLit,
    LibcLinkage,
    Param,
    ParamRef,
    Program,
    ReturnExpr,
    StringConstant,
    StringRef,
    U32Type,
    U64Type,
)
from quod.predicate.validate import (
    PredicateError,
    assert_is_predicate,
)
from quod.validate import (
    BINOP_FLOAT_NON_FLOAT,
    BINOP_FLOAT_OPERAND_MISMATCH,
    FNEG_NON_FLOAT,
    validate,
)


_PRINTF = ExternFunction(
    name="printf",
    param_types=(I8PtrType(),),
    return_type=I32Type(),
    varargs=True,
    linkage=LibcLinkage(),
)
_FMT_LLD = StringConstant(name=".fmt_lld", value="%lld\n")
_FMT_LLU = StringConstant(name=".fmt_llu", value="%llu\n")
_FMT_G = StringConstant(name=".fmt_g", value="%g\n")
_FMT_17G = StringConstant(name=".fmt_17g", value="%.17g\n")


def _build_and_run(program: Program) -> str:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        result = compile_program(
            program,
            build_dir=td_path,
            bins=(("test", "main"),),
        )
        binary = result.bins[0].binary
        assert binary is not None
        out = subprocess.run(
            [str(binary)], capture_output=True, text=True, check=False, timeout=10,
        )
        return out.stdout


def _print_lld(value):
    return ExprStmt(value=Call(
        function="printf",
        args=(StringRef(name=".fmt_lld"), value),
    ))


def _print_g(value):
    return ExprStmt(value=Call(
        function="printf",
        args=(StringRef(name=".fmt_g"), value),
    ))


def _print_17g(value):
    return ExprStmt(value=Call(
        function="printf",
        args=(StringRef(name=".fmt_17g"), value),
    ))


def _main(stmts):
    return Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=tuple(stmts) + (
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )


# ---------- FloatLit construction & round-trip ----------

def test_float_lit_finite_round_trips_through_json():
    """Construct a FloatLit, serialize to JSON, validate back: same node."""
    fl = FloatLit(type=F64Type(), value=0.1)
    j = fl.model_dump_json()
    parsed = FloatLit.model_validate_json(j)
    assert parsed == fl
    assert parsed.value == 0.1


def test_float_lit_rejects_nan_at_construction():
    """Pydantic JSON serialization silently coerces NaN to null, which
    would round-trip as a parse error. Reject at construction so the
    failure is loud."""
    with pytest.raises(ValidationError):
        FloatLit(type=F64Type(), value=float("nan"))


def test_float_lit_rejects_inf_at_construction():
    with pytest.raises(ValidationError):
        FloatLit(type=F64Type(), value=float("inf"))
    with pytest.raises(ValidationError):
        FloatLit(type=F64Type(), value=float("-inf"))


# ---------- Float arithmetic ----------

def test_fadd_f64_lowers_and_runs():
    """`(1.5 + 2.25)` lowers to `fadd`; printf %g prints `3.75`."""
    main = _main((
        _print_g(BinOp(
            op="fadd",
            lhs=FloatLit(type=F64Type(), value=1.5),
            rhs=FloatLit(type=F64Type(), value=2.25),
        )),
    ))
    prog = Program(constants=(_FMT_G,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "3.75\n"


def test_fsub_fmul_fdiv_chain():
    """Compound expression exercising fsub, fmul, fdiv."""
    # ((10.0 - 4.0) * 2.0) / 3.0 = 4.0
    expr = BinOp(
        op="fdiv",
        lhs=BinOp(
            op="fmul",
            lhs=BinOp(
                op="fsub",
                lhs=FloatLit(type=F64Type(), value=10.0),
                rhs=FloatLit(type=F64Type(), value=4.0),
            ),
            rhs=FloatLit(type=F64Type(), value=2.0),
        ),
        rhs=FloatLit(type=F64Type(), value=3.0),
    )
    main = _main((_print_g(expr),))
    prog = Program(constants=(_FMT_G,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "4\n"


def test_arithmetic_round_trip_exactly_representable():
    """`(a + b) - b == a` for powers-of-2 (exactly representable in
    f64). printf %.17g prints the exact value."""
    # 2^10 + 1.0 = 1025.0; subtract 1.0 = 1024.0
    expr = BinOp(
        op="fsub",
        lhs=BinOp(
            op="fadd",
            lhs=FloatLit(type=F64Type(), value=1024.0),
            rhs=FloatLit(type=F64Type(), value=1.0),
        ),
        rhs=FloatLit(type=F64Type(), value=1.0),
    )
    main = _main((_print_17g(expr),))
    prog = Program(constants=(_FMT_17G,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "1024\n"


def test_frem_is_fmod_like():
    """LLVM `frem` is fmod-like: `5.5 % 2.0 == 1.5`."""
    expr = BinOp(
        op="frem",
        lhs=FloatLit(type=F64Type(), value=5.5),
        rhs=FloatLit(type=F64Type(), value=2.0),
    )
    main = _main((_print_g(expr),))
    prog = Program(constants=(_FMT_G,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "1.5\n"


# ---------- Float comparison ----------

def test_feq_lowers_to_oeq_and_returns_true_for_equal():
    """`feq` lowers to LLVM `fcmp oeq`; equal values return true (1)."""
    cmp = BinOp(
        op="feq",
        lhs=FloatLit(type=F64Type(), value=3.14),
        rhs=FloatLit(type=F64Type(), value=3.14),
    )
    # Cast i1 → i64 for printing.
    main = _main((
        _print_lld(Cast(value=cmp, target_type=I64Type())),
    ))
    prog = Program(constants=(_FMT_LLD,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "1\n"


def test_fne_lowers_to_une_so_nan_ne_nan_is_true():
    """`fne` lowers to `une` so `NaN != NaN` returns true. We can't
    construct a NaN literal directly (Pydantic rejects), so build NaN
    via `0.0 / 0.0`."""
    nan = BinOp(
        op="fdiv",
        lhs=FloatLit(type=F64Type(), value=0.0),
        rhs=FloatLit(type=F64Type(), value=0.0),
    )
    cmp = BinOp(op="fne", lhs=nan, rhs=nan)
    main = _main((
        _print_lld(Cast(value=cmp, target_type=I64Type())),
    ))
    prog = Program(constants=(_FMT_LLD,), externs=(_PRINTF,), functions=(main,))
    # NaN != NaN → true (1)
    assert _build_and_run(prog) == "1\n"


def test_feq_with_nan_is_false():
    """`feq` lowers to `oeq` (ordered) — false if either operand is NaN."""
    nan = BinOp(
        op="fdiv",
        lhs=FloatLit(type=F64Type(), value=0.0),
        rhs=FloatLit(type=F64Type(), value=0.0),
    )
    cmp = BinOp(op="feq", lhs=nan, rhs=nan)
    main = _main((
        _print_lld(Cast(value=cmp, target_type=I64Type())),
    ))
    prog = Program(constants=(_FMT_LLD,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "0\n"


def test_flt_with_nan_is_false():
    """All ordered magnitude predicates (flt/fle/fgt/fge) return false
    if either operand is NaN."""
    nan = BinOp(
        op="fdiv",
        lhs=FloatLit(type=F64Type(), value=0.0),
        rhs=FloatLit(type=F64Type(), value=0.0),
    )
    cmp = BinOp(op="flt", lhs=nan, rhs=FloatLit(type=F64Type(), value=1.0))
    main = _main((
        _print_lld(Cast(value=cmp, target_type=I64Type())),
    ))
    prog = Program(constants=(_FMT_LLD,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "0\n"


def test_flt_finite_is_true():
    cmp = BinOp(
        op="flt",
        lhs=FloatLit(type=F64Type(), value=1.0),
        rhs=FloatLit(type=F64Type(), value=2.0),
    )
    main = _main((
        _print_lld(Cast(value=cmp, target_type=I64Type())),
    ))
    prog = Program(constants=(_FMT_LLD,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "1\n"


# ---------- FNeg semantics ----------

def test_fneg_of_positive_zero_yields_negative_zero():
    """FNeg flips the sign bit unconditionally — `-0.0` is observable
    via `%g` as `-0`. This is the IEEE corner that justifies a separate
    FNeg node (vs `0.0 - x`, which returns +0.0)."""
    main = _main((
        _print_g(FNeg(operand=FloatLit(type=F64Type(), value=0.0))),
    ))
    prog = Program(constants=(_FMT_G,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "-0\n"


def test_fneg_of_finite_flips_sign():
    main = _main((
        _print_g(FNeg(operand=FloatLit(type=F64Type(), value=3.5))),
    ))
    prog = Program(constants=(_FMT_G,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "-3.5\n"


def test_fsub_zero_minus_zero_is_positive_zero_not_negative():
    """Confirms the rationale for FNeg: `0.0 - 0.0` yields +0.0 even
    though IEEE negation of 0.0 yields -0.0. Round-tripping printf %g
    distinguishes them: +0.0 prints as `0`, -0.0 prints as `-0`."""
    main = _main((
        _print_g(BinOp(
            op="fsub",
            lhs=FloatLit(type=F64Type(), value=0.0),
            rhs=FloatLit(type=F64Type(), value=0.0),
        )),
    ))
    prog = Program(constants=(_FMT_G,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "0\n"


# ---------- Cast float arms (now exercised) ----------

def test_cast_int_to_float_signed_is_sitofp():
    """Cast(IntLit(I32, -1), F64Type()) → -1.0 via `sitofp`."""
    main = _main((
        _print_g(Cast(
            value=IntLit(type=I32Type(), value=-1),
            target_type=F64Type(),
        )),
    ))
    prog = Program(constants=(_FMT_G,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "-1\n"


def test_cast_int_to_float_unsigned_is_uitofp():
    """Cast(IntLit(U32, 0xFFFFFFFF), F64Type()) → 4294967295.0 via
    `uitofp`. (`sitofp` would treat the bit pattern as signed and give
    -1.0.)"""
    main = _main((
        _print_g(Cast(
            value=IntLit(type=U32Type(), value=0xFFFFFFFF),
            target_type=F64Type(),
        )),
    ))
    prog = Program(constants=(_FMT_G,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "4.29497e+09\n"


def test_cast_float_to_signed_int_saturates_at_max():
    """Cast(FloatLit(F64, 1e20), I32Type()) saturates to INT32_MAX
    (2147483647) via `llvm.fptosi.sat.i32.f64`."""
    main = _main((
        _print_lld(Cast(
            value=Cast(
                value=FloatLit(type=F64Type(), value=1e20),
                target_type=I32Type(),
            ),
            target_type=I64Type(),
        )),
    ))
    prog = Program(constants=(_FMT_LLD,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "2147483647\n"


def test_cast_nan_to_signed_int_yields_zero():
    """`fptosi.sat` maps NaN to 0. Build NaN via 0.0/0.0."""
    nan = BinOp(
        op="fdiv",
        lhs=FloatLit(type=F64Type(), value=0.0),
        rhs=FloatLit(type=F64Type(), value=0.0),
    )
    main = _main((
        _print_lld(Cast(
            value=Cast(value=nan, target_type=I32Type()),
            target_type=I64Type(),
        )),
    ))
    prog = Program(constants=(_FMT_LLD,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "0\n"


def test_cast_f32_to_f64_is_exact_fpext():
    """f32 → f64 widening is exact (`fpext`). 1.5 has the same bit
    pattern in f32 and f64 mantissas for short mantissas, so 1.5 ⊆ both."""
    main = _main((
        _print_g(Cast(
            value=FloatLit(type=F32Type(), value=1.5),
            target_type=F64Type(),
        )),
    ))
    prog = Program(constants=(_FMT_G,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "1.5\n"


def test_cast_f64_to_f32_rounds_to_nearest_even():
    """0.1 in f64 rounds-to-nearest-even to f32. f32(0.1) ≈ 0.10000000149…;
    widen back to f64 (fpext) before passing to printf — C's varargs
    promote float→double on the call boundary, but quod doesn't, so
    the test does it explicitly. Print via %.17g to observe the f32
    rounding (it differs from the original f64 0.1 in low bits)."""
    f32_val = Cast(
        value=FloatLit(type=F64Type(), value=0.1),
        target_type=F32Type(),
    )
    f64_again = Cast(value=f32_val, target_type=F64Type())
    main = _main((_print_17g(f64_again),))
    prog = Program(constants=(_FMT_17G,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "0.10000000149011612\n"


# ---------- Validator rejection ----------

def test_validator_rejects_fadd_on_int_operands():
    fn = Function(
        name="bad",
        params=(Param(name="x", type=I32Type()),),
        return_type=I32Type(),
        body=Block(stmts=(ReturnExpr(value=BinOp(
            op="fadd", lhs=ParamRef(name="x"), rhs=ParamRef(name="x"),
        )),)),
    )
    diags = validate(Program(functions=(fn,)))
    assert BINOP_FLOAT_NON_FLOAT in [d.code for d in diags]


def test_validator_rejects_fadd_with_mixed_float_widths():
    fn = Function(
        name="bad",
        params=(Param(name="a", type=F32Type()), Param(name="b", type=F64Type())),
        return_type=F64Type(),
        body=Block(stmts=(ReturnExpr(value=BinOp(
            op="fadd", lhs=ParamRef(name="a"), rhs=ParamRef(name="b"),
        )),)),
    )
    diags = validate(Program(functions=(fn,)))
    assert BINOP_FLOAT_OPERAND_MISMATCH in [d.code for d in diags]


def test_validator_rejects_fneg_of_int():
    fn = Function(
        name="bad",
        params=(Param(name="x", type=I32Type()),),
        return_type=I32Type(),
        body=Block(stmts=(ReturnExpr(value=FNeg(operand=ParamRef(name="x"))),)),
    )
    diags = validate(Program(functions=(fn,)))
    assert FNEG_NON_FLOAT in [d.code for d in diags]


# ---------- Predicate-validator rejection ----------

def test_predicate_rejects_float_lit():
    with pytest.raises(PredicateError, match="FloatLit"):
        assert_is_predicate(FloatLit(type=F64Type(), value=1.0))


def test_predicate_rejects_fneg():
    with pytest.raises(PredicateError, match="FNeg"):
        assert_is_predicate(FNeg(operand=ParamRef(name="x")))


def test_predicate_rejects_float_binop_op():
    with pytest.raises(PredicateError, match="float ops"):
        assert_is_predicate(BinOp(
            op="fadd",
            lhs=ParamRef(name="x"),
            rhs=ParamRef(name="y"),
        ))


def test_predicate_rejects_float_comparison_op():
    with pytest.raises(PredicateError, match="float ops"):
        assert_is_predicate(BinOp(
            op="feq",
            lhs=ParamRef(name="x"),
            rhs=ParamRef(name="y"),
        ))


def test_predicate_still_accepts_int_binop():
    """Sanity check: int ops continue to pass the predicate validator."""
    assert_is_predicate(BinOp(
        op="slt",
        lhs=ParamRef(name="x"),
        rhs=IntLit(type=I32Type(), value=10),
    ))
