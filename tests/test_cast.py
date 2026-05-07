"""End-to-end tests for the new `Cast` numeric-conversion node.

Cast covers int↔int, int↔float, and float↔float arms; this commit
exercises only the int arms because no FloatLit constructor exists yet
(F32Type / F64Type are defined but unconstructable until the next
commit). Float arms are tested in the float commit.

Each test builds a small program that returns or prints a Cast result,
then checks the observable output. Validator-rejection tests run the
in-process validator, not the binary.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from quod.lower import compile_program
from quod.model import (
    BinOp,
    Block,
    Call,
    Cast,
    ExprStmt,
    ExternFunction,
    Function,
    I8PtrType,
    I16Type,
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
    StructDef,
    StructField,
    StructInit,
    StructType,
    U8Type,
    U16Type,
    U32Type,
    U64Type,
    FieldInit,
)
from quod.validate import CAST_NON_NUMERIC, validate


_PRINTF = ExternFunction(
    name="printf",
    param_types=(I8PtrType(),),
    return_type=I32Type(),
    varargs=True,
    linkage=LibcLinkage(),
)
_FMT_LLD = StringConstant(name=".fmt_lld", value="%lld\n")
_FMT_LLU = StringConstant(name=".fmt_llu", value="%llu\n")


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


def _print_llu(value):
    return ExprStmt(value=Call(
        function="printf",
        args=(StringRef(name=".fmt_llu"), value),
    ))


def _main(stmts):
    return Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=tuple(stmts) + (
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )


# ---------- int → int wider: sext (signed source) ----------

def test_cast_i32_to_i64_sign_extends():
    """Cast(i32(-1), I64) → -1 (sext). LLVM `sext` chosen because
    i32 is signed."""
    main = _main((
        _print_lld(Cast(
            value=IntLit(type=I32Type(), value=-1),
            target_type=I64Type(),
        )),
    ))
    prog = Program(constants=(_FMT_LLD,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "-1\n"


# ---------- int → int wider: zext (unsigned source) ----------

def test_cast_u32_to_u64_zero_extends():
    """Cast(u32(0xFFFFFFFF), U64) → 4294967295 (zext). LLVM `zext`
    chosen because u32 is unsigned. Result printed as %llu."""
    main = _main((
        _print_llu(Cast(
            value=IntLit(type=U32Type(), value=0xFFFFFFFF),
            target_type=U64Type(),
        )),
    ))
    prog = Program(constants=(_FMT_LLU,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "4294967295\n"


# ---------- int → int narrower: trunc ----------

def test_cast_i64_to_i32_truncates():
    """Cast(i64(0x100000001), I32) → 1. LLVM `trunc` keeps the low 32
    bits; print as i64 (zero-extended via a same-width identity to u32
    then to i64) to confirm the truncated value."""
    main = _main((
        _print_lld(Cast(
            value=Cast(
                # 0x100000001 = 2^32 + 1; trunc to i32 yields 1
                value=IntLit(type=I64Type(), value=0x100000001),
                target_type=I32Type(),
            ),
            target_type=I64Type(),
        )),
    ))
    prog = Program(constants=(_FMT_LLD,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "1\n"


# ---------- int → int same width, signedness reinterpret: identity ----------

def test_cast_i32_to_u32_reinterprets_bit_pattern():
    """Cast(i32(-1), U32) → 0xFFFFFFFF. Same-width signed↔unsigned is
    an LLVM-level identity (no instruction emitted); printing as %llu
    after a u32→u64 widen confirms the bit pattern."""
    main = _main((
        _print_llu(Cast(
            value=Cast(
                value=IntLit(type=I32Type(), value=-1),
                target_type=U32Type(),
            ),
            target_type=U64Type(),
        )),
    ))
    prog = Program(constants=(_FMT_LLU,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "4294967295\n"


def test_cast_u32_to_i32_reinterprets_bit_pattern():
    """Cast(u32(0xFFFFFFFF), I32) → -1. Same-width identity; printing
    via i32→i64 sext as %lld confirms the signed interpretation."""
    main = _main((
        _print_lld(Cast(
            value=Cast(
                value=IntLit(type=U32Type(), value=0xFFFFFFFF),
                target_type=I32Type(),
            ),
            target_type=I64Type(),
        )),
    ))
    prog = Program(constants=(_FMT_LLD,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "-1\n"


# ---------- intermediate-width chains ----------

def test_cast_i16_to_i64_then_back_round_trips():
    """Cast(i16(-7), I64) sign-extends to -7; Cast(i64(-7), I16) trunc
    keeps the low 16 bits = -7. Round-trip via two Casts."""
    main = _main((
        _print_lld(Cast(
            value=Cast(
                value=IntLit(type=I16Type(), value=-7),
                target_type=I64Type(),
            ),
            target_type=I64Type(),
        )),
    ))
    prog = Program(constants=(_FMT_LLD,), externs=(_PRINTF,), functions=(main,))
    assert _build_and_run(prog) == "-7\n"


# ---------- validator rejects non-numeric source ----------

def test_validator_rejects_struct_source():
    """Cast(StructInit(Point), I32) is rejected by the validator —
    structs are not numeric."""
    sd = StructDef(
        name="Point",
        fields=(StructField(name="x", type=I32Type()),),
    )
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            ReturnExpr(value=Cast(
                value=StructInit(
                    type="Point",
                    fields=(FieldInit(name="x", value=IntLit(type=I32Type(), value=1)),),
                ),
                target_type=I32Type(),
            )),
        )),
    )
    prog = Program(structs=(sd,), functions=(main,))
    diags = validate(prog)
    codes = [d.code for d in diags]
    assert CAST_NON_NUMERIC in codes


# ---------- validator rejects non-numeric target ----------

def test_validator_rejects_struct_target():
    """Cast(IntLit(1), StructType('Point')) is rejected — target must
    be numeric."""
    sd = StructDef(
        name="Point",
        fields=(StructField(name="x", type=I32Type()),),
    )
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            ReturnExpr(value=Cast(
                value=IntLit(type=I32Type(), value=1),
                target_type=StructType(name="Point"),
            )),
        )),
    )
    prog = Program(structs=(sd,), functions=(main,))
    diags = validate(prog)
    codes = [d.code for d in diags]
    assert CAST_NON_NUMERIC in codes
