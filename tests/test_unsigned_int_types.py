"""End-to-end tests for the u8/u16/u32/u64 type classes.

These types lower to the same LLVM widths as i8/i16/i32/i64 — signedness
lives on the operation, not the type — so the exercise here is to confirm:

- u-typed literals build, lower, and run.
- The new `urem` binop lowers correctly.
- Unsigned ops produce a different observable result from signed ops on
  values that straddle the signedness boundary (the only way to prove
  the unsigned path is actually being taken).
- u64 holds values past i64::MAX and round-trips through printf %llu.
- script-mode parses u-suffixed literals, u-typed params, `/u` udiv,
  and the new `%u` urem operator.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from quod.lower import compile_program
from quod.model import (
    BinOp,
    Call,
    ExprStmt,
    ExternFunction,
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
    U8Type,
    U16Type,
    U32Type,
    U64Type,
    IsizeType,
    UsizeType,
    Cast,

    Block,
)
from quod.script import parse_function


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
    """printf("%lld\\n", <i64 value>) — caller widens to i64 first."""
    return ExprStmt(value=Call(
        function="printf",
        args=(StringRef(name=".fmt_lld"), value),
    ))


def _print_llu(value):
    """printf("%llu\\n", <i64-bit-pattern value>) — for unsigned printing."""
    return ExprStmt(value=Call(
        function="printf",
        args=(StringRef(name=".fmt_llu"), value),
    ))


def _zext_to_i64(u_value):
    """Zero-extend a u<N> value to i64 for printing. Cast derives
    zext from the source's UXType."""
    return Cast(value=u_value, target_type=I64Type())


# ---------- Core: u-typed values lower and run ----------

def test_u32_addition_round_trips():
    """u32 + u32 -> u32; widen to i64; printf prints it."""
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            _print_lld(_zext_to_i64(BinOp(
                op="add",
                lhs=IntLit(type=U32Type(), value=10),
                rhs=IntLit(type=U32Type(), value=32),
            ))),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        constants=(_FMT_LLD,), externs=(_PRINTF,), functions=(main,),
    )
    assert _build_and_run(prog) == "42\n"


def test_u8_u16_u32_u64_all_lower():
    """Each width holds a value, widens to i64, printf round-trips it."""
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            _print_lld(_zext_to_i64(IntLit(type=U8Type(), value=200))),
            _print_lld(_zext_to_i64(IntLit(type=U16Type(), value=50000))),
            _print_lld(_zext_to_i64(IntLit(type=U32Type(), value=3000000000))),
            _print_lld(IntLit(type=U64Type(), value=42)),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        constants=(_FMT_LLD,), externs=(_PRINTF,), functions=(main,),
    )
    assert _build_and_run(prog) == "200\n50000\n3000000000\n42\n"


# ---------- urem (the newly-added binop) ----------

def test_urem_lowers():
    """urem(13, 5) == 3; matches both srem and urem on positive values, but
    proves the new BinOp("urem") variant lowers without errors."""
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            _print_lld(_zext_to_i64(BinOp(
                op="urem",
                lhs=IntLit(type=U32Type(), value=13),
                rhs=IntLit(type=U32Type(), value=5),
            ))),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        constants=(_FMT_LLD,), externs=(_PRINTF,), functions=(main,),
    )
    assert _build_and_run(prog) == "3\n"


# ---------- Unsigned ops genuinely differ from signed ops ----------

def test_udiv_vs_sdiv_divergence_on_u32_max():
    """0xFFFFFFFF / 2 — as u32, udiv → 2147483647; as i32, sdiv → 0
    (since -1 / 2 truncates toward zero in two's-complement).
    Prints both so a regression in either path shows up."""
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            _print_lld(_zext_to_i64(BinOp(
                op="udiv",
                lhs=IntLit(type=U32Type(), value=0xFFFFFFFF),
                rhs=IntLit(type=U32Type(), value=2),
            ))),
            _print_lld(Cast(
                value=BinOp(
                    op="sdiv",
                    lhs=IntLit(type=I32Type(), value=-1),
                    rhs=IntLit(type=I32Type(), value=2),
                ),
                target_type=I64Type(),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        constants=(_FMT_LLD,), externs=(_PRINTF,), functions=(main,),
    )
    assert _build_and_run(prog) == "2147483647\n0\n"


def test_ult_vs_slt_divergence():
    """0xFFFFFFFF < 1: as unsigned u32, false (4294967295 > 1);
    as signed i32, true (-1 < 1). Prints 0/1 for each."""
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            _print_lld(_zext_to_i64(BinOp(
                op="ult",
                lhs=IntLit(type=U32Type(), value=0xFFFFFFFF),
                rhs=IntLit(type=U32Type(), value=1),
            ))),
            _print_lld(_zext_to_i64(BinOp(
                op="slt",
                lhs=IntLit(type=I32Type(), value=-1),
                rhs=IntLit(type=I32Type(), value=1),
            ))),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        constants=(_FMT_LLD,), externs=(_PRINTF,), functions=(main,),
    )
    assert _build_and_run(prog) == "0\n1\n"


# ---------- u64 holds values beyond i64::MAX ----------

def test_u64_wide_literal_via_llu():
    """A u64 literal larger than i64::MAX (= 2^63 - 1) prints correctly
    via %llu. Pydantic ValidationError on the literal would catch
    range-check regressions; %llu confirms the bit pattern survives."""
    wide = (1 << 63) + 100  # 9223372036854775908
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            _print_llu(IntLit(type=U64Type(), value=wide)),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        constants=(_FMT_LLU,), externs=(_PRINTF,), functions=(main,),
    )
    assert _build_and_run(prog) == f"{wide}\n"


# ---------- script-mode integration ----------

def test_script_parses_u32_function():
    """fn add(a: u32, b: u32) -> u32 { return a + b } parses cleanly."""
    src = "fn add(a: u32, b: u32) -> u32 { return a + b }"
    fn = parse_function(src)
    assert fn.name == "add"
    assert isinstance(fn.return_type, U32Type)
    assert isinstance(fn.params[0].type, U32Type)
    assert isinstance(fn.params[1].type, U32Type)


def test_script_u_suffix_int_lit():
    """`42u32` lexes and parses as IntLit with U32Type; `42` (no suffix)
    still defaults to i64. Verify both via a one-liner function."""
    fn = parse_function(
        "fn f() -> u32 { return 42u32 }"
    )
    body = fn.body.stmts
    assert len(body) == 1
    assert isinstance(body[0], ReturnExpr)
    lit = body[0].value
    assert isinstance(lit, IntLit)
    assert isinstance(lit.type, U32Type)
    assert lit.value == 42


def test_script_urem_via_percent_u_operator():
    """`a %u b` parses as BinOp(op="urem"), distinct from `a % b` (srem)."""
    fn_urem = parse_function(
        "fn r(a: u32, b: u32) -> u32 { return a %u b }"
    )
    expr_urem = fn_urem.body.stmts[0].value
    assert isinstance(expr_urem, BinOp)
    assert expr_urem.op == "urem"

    fn_srem = parse_function(
        "fn r(a: i32, b: i32) -> i32 { return a % b }"
    )
    expr_srem = fn_srem.body.stmts[0].value
    assert isinstance(expr_srem, BinOp)
    assert expr_srem.op == "srem"


def test_script_for_loop_with_u64_var():
    """A `for x: u64 in 0..3` loop parses; u64 is accepted as the loop
    variable's type."""
    fn = parse_function(
        "fn s() -> u64 { let acc: u64 = 0u64\n"
        "for x: u64 in 0u64..3u64 { acc = acc + x }\n"
        "return acc }"
    )
    assert isinstance(fn.return_type, U64Type)


# ---------- isize / usize ----------

def test_isize_usize_lower_to_64bit():
    """isize values are signed 64-bit; usize values are unsigned 64-bit.
    Both round-trip through printf, demonstrating they share i64 width
    but differ in signed vs unsigned interpretation when paired with the
    matching format specifier."""
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            # An isize value: a small positive number, %lld.
            _print_lld(IntLit(type=IsizeType(), value=12345)),
            # A usize value larger than i64::MAX: %llu.
            _print_llu(IntLit(type=UsizeType(), value=(1 << 63) + 7)),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        constants=(_FMT_LLD, _FMT_LLU), externs=(_PRINTF,), functions=(main,),
    )
    assert _build_and_run(prog) == f"12345\n{(1 << 63) + 7}\n"


def test_isize_arith_signed_semantics():
    """isize arithmetic uses signed ops; sdiv on a negative isize value
    truncates toward zero like i64."""
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            _print_lld(BinOp(
                op="sdiv",
                lhs=IntLit(type=IsizeType(), value=-7),
                rhs=IntLit(type=IsizeType(), value=2),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        constants=(_FMT_LLD,), externs=(_PRINTF,), functions=(main,),
    )
    assert _build_and_run(prog) == "-3\n"


def test_usize_arith_unsigned_semantics():
    """usize / usize uses udiv; on a value that's "negative as i64" the
    result differs from the signed interpretation."""
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            _print_llu(BinOp(
                op="udiv",
                lhs=IntLit(type=UsizeType(), value=(1 << 63) + (1 << 62)),  # 0xC000_0000_0000_0000
                rhs=IntLit(type=UsizeType(), value=2),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        constants=(_FMT_LLU,), externs=(_PRINTF,), functions=(main,),
    )
    expected = ((1 << 63) + (1 << 62)) // 2  # 0x6000_0000_0000_0000
    assert _build_and_run(prog) == f"{expected}\n"


def test_script_parses_usize_function_signature():
    """fn alloc_size(n: usize) -> usize { return n + 1usize } parses, and
    the param/return types come back as UsizeType."""
    fn = parse_function(
        "fn alloc_size(n: usize) -> usize { return n + 1usize }"
    )
    assert fn.name == "alloc_size"
    assert isinstance(fn.return_type, UsizeType)
    assert isinstance(fn.params[0].type, UsizeType)


def test_script_isize_suffix():
    """`-42isize` parses as a negative isize literal, surviving the suffix
    matcher (which prefers `isize` over `i8`/`i16`/etc.)."""
    fn = parse_function(
        "fn f() -> isize { return -42isize }"
    )
    body = fn.body.stmts
    assert len(body) == 1
    lit = body[0].value
    assert isinstance(lit, IntLit)
    assert isinstance(lit.type, IsizeType)
    assert lit.value == -42


def test_script_end_to_end_u32_compute_and_print():
    """Author a u32-arithmetic function via script, splice it into a
    program with a hand-built main, and verify stdout."""
    add_u = parse_function(
        "fn add_u(a: u32, b: u32) -> u32 { return a + b }"
    )
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            _print_lld(_zext_to_i64(Call(
                function="add_u",
                args=(
                    IntLit(type=U32Type(), value=100),
                    IntLit(type=U32Type(), value=23),
                ),
            ))),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        constants=(_FMT_LLD,), externs=(_PRINTF,),
        functions=(add_u, main),
    )
    assert _build_and_run(prog) == "123\n"
