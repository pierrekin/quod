"""End-to-end tests for the new `core.num` stdlib module.

Covers:

- `bswap_u{16,32,64}` — extern wrappers around `__builtin_bswap*`.
  Pinned via known bit patterns.
- `to_be` / `from_be` — call bswap; round-trip on non-symmetric values
  proves the chain is correct.
- `to_le` / `from_le` — identity on the only target (x86_64 LE).
- The four claim-bearing conversions (`narrow_i64_to_i32`,
  `narrow_u64_to_u32`, `i64_to_u64`, `u64_to_i64`).
- Runtime-archive activation: this module is the first to require a
  runtime `.c` file post-arena-lift, so a smoke check that
  `libquodrt-v1.a` actually got built pins that infrastructure.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from quod.lower import compile_program
from quod.model import (
    Call,
    ExprStmt,
    ExternFunction,
    Function,
    I8PtrType,
    I32Type,
    I64Type,
    Import,
    IntLit,
    LibcLinkage,
    Param,
    Program,
    ReturnExpr,
    StringConstant,
    StringRef,
    U16Type,
    U32Type,
    U64Type,
    Widen,

    Block,
)
from quod.runtime import runtime_archive_path


_PRINTF = ExternFunction(
    name="printf",
    param_types=(I8PtrType(),),
    return_type=I32Type(),
    varargs=True,
    linkage=LibcLinkage(),
)
_FMT_LLD = StringConstant(name=".fmt_lld", value="%lld\n")
_FMT_LLU = StringConstant(name=".fmt_llu", value="%llu\n")


def _build_and_run(program: Program, *, build_dir: Path | None = None) -> tuple[str, Path]:
    """Compile + run; return (stdout, build_dir). Caller may pass an
    explicit build_dir so the runtime archive can be inspected afterwards."""
    if build_dir is None:
        td = tempfile.TemporaryDirectory()
        build_dir = Path(td.name)
        owner = td
    else:
        owner = None
    try:
        result = compile_program(
            program,
            build_dir=build_dir,
            bins=(("test", "main"),),
        )
        binary = result.bins[0].binary
        assert binary is not None
        out = subprocess.run(
            [str(binary)], capture_output=True, text=True, check=False, timeout=10,
        )
        return out.stdout, build_dir
    finally:
        if owner is not None:
            owner.cleanup()


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


def _zext(u_value):
    return Widen(value=u_value, target=I64Type(), signed=False)


def _print_u(call_expr):
    """For unsigned-typed values: zero-extend to i64, print via %llu."""
    return _print_llu(_zext(call_expr))


def _make_bswap_program(width: int, value: int) -> Program:
    u_type = {16: U16Type, 32: U32Type, 64: U64Type}[width]()
    fn_name = f"core.num.bswap_u{width}"
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            _print_u(Call(
                function=fn_name,
                args=(IntLit(type=u_type, value=value),),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    return Program(
        imports=(Import(module="core.num"),),
        constants=(_FMT_LLU,),
        externs=(_PRINTF,),
        functions=(main,),
    )


# ---------- bswap ----------

def test_bswap_u16_known_value():
    out, _ = _build_and_run(_make_bswap_program(16, 0x1234))
    assert out == f"{0x3412}\n"


def test_bswap_u32_known_value():
    out, _ = _build_and_run(_make_bswap_program(32, 0xDEADBEEF))
    assert out == f"{0xEFBEADDE}\n"


def test_bswap_u64_known_value():
    out, _ = _build_and_run(_make_bswap_program(64, 0x0123456789ABCDEF))
    assert out == f"{0xEFCDAB8967452301}\n"


# ---------- to_be / from_be round-trip ----------

def _roundtrip_be_program(width: int, value: int) -> Program:
    """Print `from_be(to_be(value))` — must equal `value`."""
    u_type = {16: U16Type, 32: U32Type, 64: U64Type}[width]()
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            _print_u(Call(
                function=f"core.num.from_be_u{width}",
                args=(Call(
                    function=f"core.num.to_be_u{width}",
                    args=(IntLit(type=u_type, value=value),),
                ),),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    return Program(
        imports=(Import(module="core.num"),),
        constants=(_FMT_LLU,),
        externs=(_PRINTF,),
        functions=(main,),
    )


def test_to_be_from_be_roundtrip_u16():
    out, _ = _build_and_run(_roundtrip_be_program(16, 0xBEEF))
    assert out == f"{0xBEEF}\n"


def test_to_be_from_be_roundtrip_u32():
    out, _ = _build_and_run(_roundtrip_be_program(32, 0xCAFEBABE))
    assert out == f"{0xCAFEBABE}\n"


def test_to_be_from_be_roundtrip_u64():
    val = 0x0102030405060708
    out, _ = _build_and_run(_roundtrip_be_program(64, val))
    assert out == f"{val}\n"


# ---------- to_le / from_le identity (x86_64) ----------

def test_to_le_u32_identity_on_x86_64():
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            _print_u(Call(
                function="core.num.to_le_u32",
                args=(IntLit(type=U32Type(), value=0x11223344),),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        imports=(Import(module="core.num"),),
        constants=(_FMT_LLU,),
        externs=(_PRINTF,),
        functions=(main,),
    )
    out, _ = _build_and_run(prog)
    assert out == f"{0x11223344}\n"


def test_from_le_u64_identity_on_x86_64():
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            _print_u(Call(
                function="core.num.from_le_u64",
                args=(IntLit(type=U64Type(), value=0xAABBCCDDEEFF0011),),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        imports=(Import(module="core.num"),),
        constants=(_FMT_LLU,),
        externs=(_PRINTF,),
        functions=(main,),
    )
    out, _ = _build_and_run(prog)
    assert out == f"{0xAABBCCDDEEFF0011}\n"


# ---------- conversions ----------

def test_narrow_i64_to_i32_in_range():
    """narrow_i64_to_i32 with values that fit i32 — both positive and
    negative — succeeds and preserves the value."""
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            _print_lld(Widen(
                value=Call(
                    function="core.num.narrow_i64_to_i32",
                    args=(IntLit(type=I64Type(), value=42),),
                ),
                target=I64Type(), signed=True,
            )),
            _print_lld(Widen(
                value=Call(
                    function="core.num.narrow_i64_to_i32",
                    args=(IntLit(type=I64Type(), value=-1),),
                ),
                target=I64Type(), signed=True,
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        imports=(Import(module="core.num"),),
        constants=(_FMT_LLD,),
        externs=(_PRINTF,),
        functions=(main,),
    )
    out, _ = _build_and_run(prog)
    assert out == "42\n-1\n"


def test_narrow_u64_to_u32_in_range():
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            _print_u(Call(
                function="core.num.narrow_u64_to_u32",
                args=(IntLit(type=U64Type(), value=0xDEADBEEF),),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        imports=(Import(module="core.num"),),
        constants=(_FMT_LLU,),
        externs=(_PRINTF,),
        functions=(main,),
    )
    out, _ = _build_and_run(prog)
    assert out == f"{0xDEADBEEF}\n"


def test_i64_to_u64_nonnegative_bit_preserved():
    """Largest non-negative i64 — i64::MAX — should bit-cast cleanly to
    u64 = 9223372036854775807 (no sign-bit set, so signed and unsigned
    interpretations agree)."""
    val = (1 << 63) - 1
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            _print_llu(Call(
                function="core.num.i64_to_u64",
                args=(IntLit(type=I64Type(), value=val),),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        imports=(Import(module="core.num"),),
        constants=(_FMT_LLU,),
        externs=(_PRINTF,),
        functions=(main,),
    )
    out, _ = _build_and_run(prog)
    assert out == f"{val}\n"


def test_u64_to_i64_max_in_range():
    """A u64 value at i64::MAX bit-casts to the same numeric value as
    i64."""
    val = (1 << 63) - 1
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            _print_lld(Call(
                function="core.num.u64_to_i64",
                args=(IntLit(type=U64Type(), value=val),),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        imports=(Import(module="core.num"),),
        constants=(_FMT_LLD,),
        externs=(_PRINTF,),
        functions=(main,),
    )
    out, _ = _build_and_run(prog)
    assert out == f"{val}\n"


# ---------- runtime archive activation ----------

def test_runtime_archive_built_when_core_num_imported():
    """First runtime/.c file post-arena-lift; pin that the archive
    actually gets built and linked when something imports core.num."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        out, build_dir = _build_and_run(
            _make_bswap_program(32, 0x12345678),
            build_dir=td_path,
        )
        assert out == f"{0x78563412}\n"
        archive = runtime_archive_path(build_dir)
        assert archive.exists(), (
            f"runtime archive not built at {archive!s} after compiling a "
            f"program that imports core.num and calls bswap_u32"
        )
