"""End-to-end tests for the new `core.mem` and `core.cmp` stdlib modules.

Builds tiny programs that import `core.mem` (which transitively pulls in
`core.cmp` for `Ordering`), compiles them, runs them, and asserts on
stdout.

Pinned behaviours:
- `align_up(x, a)` rounds up to the nearest multiple of `a`.
- `align_down(x, a)` rounds down.
- `set(dst, value, n)` writes `value` to `n` bytes at `dst` (memset).
- `copy_overlapping(dst, src, n)` memmove — handles forward overlap
  correctly (a memcpy bug here would corrupt the result).
- `compare(a, alen, b, blen)` returns the three-way Ordering, with
  shorter-as-prefix-of-longer as Less.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from quod.lower import compile_program
from quod.model import (
    Assign,
    BinOp,
    Call,
    EnumType,
    ExprStmt,
    ExternFunction,
    I32Type,
    I64Type,
    I8PtrType,
    I8Type,
    If,
    Import,
    IntLit,
    LibcLinkage,
    Let,
    Load,
    LocalRef,
    Match,
    MatchArm,
    Param,
    ParamRef,
    Program,
    ReturnExpr,
    StringConstant,
    StringRef,
    Function,
    Widen,
    WithArena,
)


_PRINTF = ExternFunction(
    name="printf",
    param_types=(I8PtrType(),),
    return_type=I32Type(),
    varargs=True,
    linkage=LibcLinkage(),
)
_FMT_INT = StringConstant(name=".fmt_int", value="%lld\n")


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


def _print_int_call(value):
    return ExprStmt(value=Call(
        function="printf",
        args=(StringRef(name=".fmt_int"), value),
    ))


def _print_align_up_main(x: int, align: int) -> Program:
    """Build a program that prints `align_up(x, align)`."""
    main = Function(
        name="main",
        return_type=I32Type(),
        body=(
            _print_int_call(Call(
                function="core.mem.align_up",
                args=(
                    IntLit(type=I64Type(), value=x),
                    IntLit(type=I64Type(), value=align),
                ),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    return Program(
        imports=(Import(module="core.mem"),),
        constants=(_FMT_INT,),
        externs=(_PRINTF,),
        functions=(main,),
    )


def _print_align_down_main(x: int, align: int) -> Program:
    main = Function(
        name="main",
        return_type=I32Type(),
        body=(
            _print_int_call(Call(
                function="core.mem.align_down",
                args=(
                    IntLit(type=I64Type(), value=x),
                    IntLit(type=I64Type(), value=align),
                ),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    return Program(
        imports=(Import(module="core.mem"),),
        constants=(_FMT_INT,),
        externs=(_PRINTF,),
        functions=(main,),
    )


def test_align_up_zero():
    assert _build_and_run(_print_align_up_main(0, 8)) == "0\n"


def test_align_up_partial():
    assert _build_and_run(_print_align_up_main(13, 8)) == "16\n"


def test_align_up_already_aligned():
    assert _build_and_run(_print_align_up_main(16, 8)) == "16\n"


def test_align_up_just_over():
    assert _build_and_run(_print_align_up_main(17, 8)) == "24\n"


def test_align_down_zero():
    assert _build_and_run(_print_align_down_main(0, 8)) == "0\n"


def test_align_down_partial():
    assert _build_and_run(_print_align_down_main(13, 8)) == "8\n"


def test_align_down_already_aligned():
    assert _build_and_run(_print_align_down_main(16, 8)) == "16\n"


def _byte_offset(base_name: str, offset: int):
    """`base_name + offset` as an i8* expression."""
    return BinOp(
        op="add",
        lhs=LocalRef(name=base_name),
        rhs=IntLit(type=I64Type(), value=offset),
    )


def _load_byte_print(base_name: str, offset: int):
    """Load byte at base+offset, widen to i64, printf as %lld."""
    return _print_int_call(Widen(
        value=Load(ptr=_byte_offset(base_name, offset), type=I8Type()),
        target=I64Type(),
    ))


def test_set_writes_byte_pattern():
    """core.mem.set(buf, 0x41, 4) → buf is "AAAA" (65 four times)."""
    main = Function(
        name="main",
        return_type=I32Type(),
        body=(
            WithArena(
                name="a",
                capacity=IntLit(type=I64Type(), value=64),
                body=(
                    Let(
                        name="buf",
                        type=I8PtrType(),
                        init=Call(
                            function="mem.arena.alloc",
                            args=(LocalRef(name="a"), IntLit(type=I64Type(), value=4)),
                        ),
                    ),
                    ExprStmt(value=Call(
                        function="core.mem.set",
                        args=(
                            LocalRef(name="buf"),
                            IntLit(type=I32Type(), value=0x41),
                            IntLit(type=I64Type(), value=4),
                        ),
                    )),
                    _load_byte_print("buf", 0),
                    _load_byte_print("buf", 1),
                    _load_byte_print("buf", 2),
                    _load_byte_print("buf", 3),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(
        imports=(Import(module="core.mem"), Import(module="mem.arena")),
        constants=(_FMT_INT,),
        externs=(_PRINTF,),
        functions=(main,),
    )
    assert _build_and_run(prog) == "65\n65\n65\n65\n"


def test_copy_overlapping_handles_forward_overlap():
    """Fill 8 bytes [1..=8], then copy_overlapping(buf+2, buf, 4).

    A memcpy bug would yield 1 2 1 2 1 2 7 8 (source bytes overwritten
    before being read). memmove yields 1 2 1 2 3 4 7 8.
    """
    fills = []
    for i in range(8):
        fills.append(ExprStmt(value=Call(
            function="core.mem.set",
            args=(
                _byte_offset("buf", i),
                IntLit(type=I32Type(), value=i + 1),
                IntLit(type=I64Type(), value=1),
            ),
        )))
    prints = [_load_byte_print("buf", i) for i in range(8)]
    main = Function(
        name="main",
        return_type=I32Type(),
        body=(
            WithArena(
                name="a",
                capacity=IntLit(type=I64Type(), value=64),
                body=(
                    Let(
                        name="buf",
                        type=I8PtrType(),
                        init=Call(
                            function="mem.arena.alloc",
                            args=(LocalRef(name="a"), IntLit(type=I64Type(), value=8)),
                        ),
                    ),
                    *fills,
                    ExprStmt(value=Call(
                        function="core.mem.copy_overlapping",
                        args=(
                            _byte_offset("buf", 2),
                            LocalRef(name="buf"),
                            IntLit(type=I64Type(), value=4),
                        ),
                    )),
                    *prints,
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(
        imports=(Import(module="core.mem"), Import(module="mem.arena")),
        constants=(_FMT_INT,),
        externs=(_PRINTF,),
        functions=(main,),
    )
    assert _build_and_run(prog) == "1\n2\n1\n2\n3\n4\n7\n8\n"


def _compare_main(a: str, alen: int, b: str, blen: int) -> Program:
    """Build a program that calls core.mem.compare and prints 0/1/2 per arm."""
    a_const = StringConstant(name=".sa", value=a)
    b_const = StringConstant(name=".sb", value=b)
    main = Function(
        name="main",
        return_type=I32Type(),
        body=(
            Let(
                name="ord",
                type=EnumType(name="core.cmp.Ordering"),
                init=Call(
                    function="core.mem.compare",
                    args=(
                        StringRef(name=".sa"),
                        IntLit(type=I64Type(), value=alen),
                        StringRef(name=".sb"),
                        IntLit(type=I64Type(), value=blen),
                    ),
                ),
            ),
            Match(
                scrutinee=LocalRef(name="ord"),
                arms=(
                    MatchArm(
                        variant="Less",
                        body=(_print_int_call(IntLit(type=I64Type(), value=0)),),
                    ),
                    MatchArm(
                        variant="Equal",
                        body=(_print_int_call(IntLit(type=I64Type(), value=1)),),
                    ),
                    MatchArm(
                        variant="Greater",
                        body=(_print_int_call(IntLit(type=I64Type(), value=2)),),
                    ),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    return Program(
        imports=(Import(module="core.mem"),),
        constants=(_FMT_INT, a_const, b_const),
        externs=(_PRINTF,),
        functions=(main,),
    )


def test_compare_less():
    assert _build_and_run(_compare_main("abc", 3, "abd", 3)) == "0\n"


def test_compare_equal():
    assert _build_and_run(_compare_main("abc", 3, "abc", 3)) == "1\n"


def test_compare_greater():
    assert _build_and_run(_compare_main("abd", 3, "abc", 3)) == "2\n"


def test_compare_length_tiebreak_less():
    """Common prefix, shorter is Less."""
    assert _build_and_run(_compare_main("abc", 3, "abcd", 4)) == "0\n"


def test_compare_length_tiebreak_greater():
    """Common prefix, longer is Greater."""
    assert _build_and_run(_compare_main("abcd", 4, "abc", 3)) == "2\n"
