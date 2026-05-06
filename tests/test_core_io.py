"""End-to-end tests for the new `core.io` stdlib module.

Covers:

- `IoError` enum construct + match (smoke for the variant set).
- `BytesReader` reads bytes from an in-memory buffer; trait dispatch via
  `quod.trait_call` resolves to the Reader-for-BytesReader impl.
- EOF behaviour (read at end returns `Ok(0)`).
- `BytesWriter` writes bytes; short write at capacity behaves as
  expected; subsequent `flush` returns `None`.

This is the first stdlib usage of `Result<T, E>` with a non-trivial
`E` (an enum payload), validating that mono handles enum-payload
substitution end-to-end.
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
    EnumInit,
    EnumType,
    ExprStmt,
    ExternFunction,
    FieldInit,
    Function,
    I8PtrType,
    I8Type,
    I32Type,
    I64Type,
    If,
    Import,
    IntLit,
    LibcLinkage,
    Let,
    Load,
    LoadField,
    LocalRef,
    Match,
    MatchArm,
    Param,
    Program,
    PtrOffset,
    ReturnExpr,
    SizeOf,
    StoreField,
    StringConstant,
    StringRef,
    StructType,
    TraitCall,
    UsizeType,
    Widen,
    WithArena,

    Block,
)


_PRINTF = ExternFunction(
    name="printf",
    param_types=(I8PtrType(),),
    return_type=I32Type(),
    varargs=True,
    linkage=LibcLinkage(),
)
_FMT_LLD = StringConstant(name=".fmt_lld", value="%lld\n")


def _build_and_run(program: Program) -> str:
    with tempfile.TemporaryDirectory() as td:
        result = compile_program(
            program,
            build_dir=Path(td),
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


def _zext64(uvalue):
    return Widen(value=uvalue, target=I64Type(), signed=False)


def _print_byte_at(buf_name: str, offset: int):
    """Load the byte at buf+offset, widen to i64, printf as int."""
    return _print_lld(_zext64(Load(
        ptr=BinOp(
            op="add",
            lhs=LocalRef(name=buf_name),
            rhs=IntLit(type=I64Type(), value=offset),
        ),
        type=I8Type(),
    )))


# ---------- IoError variant smoke ----------

def test_io_error_construct_and_match_unit_and_payload():
    """Construct IoError::NotFound (unit) and IoError::Other(42), match
    each, print a discriminator. Exercises both an empty-fields variant
    and the i32-payload variant."""
    err_ty = EnumType(name="core.io.IoError")

    def match_print(scrutinee, label_for_other: int):
        return Match(
            scrutinee=scrutinee,
            arms=(
                MatchArm(
                    variant="NotFound",
                    body=Block(stmts=(_print_lld(IntLit(type=I64Type(), value=1)),)),
                ),
                MatchArm(
                    variant="PermissionDenied",
                    body=Block(stmts=(_print_lld(IntLit(type=I64Type(), value=2)),)),
                ),
                MatchArm(
                    variant="Interrupted",
                    body=Block(stmts=(_print_lld(IntLit(type=I64Type(), value=3)),)),
                ),
                MatchArm(
                    variant="AlreadyExists",
                    body=Block(stmts=(_print_lld(IntLit(type=I64Type(), value=4)),)),
                ),
                MatchArm(
                    variant="WouldBlock",
                    body=Block(stmts=(_print_lld(IntLit(type=I64Type(), value=5)),)),
                ),
                MatchArm(
                    variant="Other",
                    bindings=("e",),
                    body=Block(stmts=(_print_lld(_zext64(LocalRef(name="e"))),)),
                ),
            ),
        )

    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            Let(
                name="e1", type=err_ty,
                init=EnumInit(enum="core.io.IoError", variant="NotFound"),
            ),
            match_print(LocalRef(name="e1"), 1),
            Let(
                name="e2", type=err_ty,
                init=EnumInit(
                    enum="core.io.IoError", variant="Other",
                    fields=(FieldInit(name="errno",
                                      value=IntLit(type=I32Type(), value=42)),),
                ),
            ),
            match_print(LocalRef(name="e2"), 42),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        imports=(Import(module="core.io"),),
        constants=(_FMT_LLD,),
        externs=(_PRINTF,),
        functions=(main,),
    )
    assert _build_and_run(prog) == "1\n42\n"


# ---------- BytesReader ----------

_BYTES_READER = StructType(name="core.io.BytesReader")
_BYTES_WRITER = StructType(name="core.io.BytesWriter")
_RESULT_USIZE_IOERROR = EnumType(
    name="core.result.Result",
    type_args=(UsizeType(), EnumType(name="core.io.IoError")),
)
_OPTION_IOERROR = EnumType(
    name="core.option.Option",
    type_args=(EnumType(name="core.io.IoError"),),
)


def _arena_alloc_bytes(arena_name: str, n: int, *, dest: str):
    """Let dest = mem.arena.alloc(arena, n)."""
    return Let(
        name=dest, type=I8PtrType(),
        init=Call(
            function="mem.arena.alloc",
            args=(LocalRef(name=arena_name), IntLit(type=I64Type(), value=n)),
        ),
    )


def _arena_alloc_struct(arena_name: str, struct_name: str, *, dest: str):
    """Let dest = mem.arena.alloc(arena, sizeof(StructName))."""
    return Let(
        name=dest, type=I8PtrType(),
        init=Call(
            function="mem.arena.alloc",
            args=(
                LocalRef(name=arena_name),
                SizeOf(type=StructType(name=struct_name)),
            ),
        ),
    )


def _init_bytes_reader(handle: str, ptr: str, length_bytes: int):
    """Set BytesReader.ptr, .len, .pos via store_field."""
    return (
        StoreField(
            ptr=LocalRef(name=handle),
            struct_type="core.io.BytesReader",
            name="ptr", value=LocalRef(name=ptr),
        ),
        StoreField(
            ptr=LocalRef(name=handle),
            struct_type="core.io.BytesReader",
            name="len", value=IntLit(type=UsizeType(), value=length_bytes),
        ),
        StoreField(
            ptr=LocalRef(name=handle),
            struct_type="core.io.BytesReader",
            name="pos", value=IntLit(type=UsizeType(), value=0),
        ),
    )


def _init_bytes_writer(handle: str, ptr: str, cap_bytes: int):
    return (
        StoreField(
            ptr=LocalRef(name=handle),
            struct_type="core.io.BytesWriter",
            name="ptr", value=LocalRef(name=ptr),
        ),
        StoreField(
            ptr=LocalRef(name=handle),
            struct_type="core.io.BytesWriter",
            name="cap", value=IntLit(type=UsizeType(), value=cap_bytes),
        ),
        StoreField(
            ptr=LocalRef(name=handle),
            struct_type="core.io.BytesWriter",
            name="pos", value=IntLit(type=UsizeType(), value=0),
        ),
    )


def _reader_read(handle_name: str, buf_name: str, n: int):
    """trait_call Reader::read on BytesReader handle."""
    return TraitCall(
        trait="core.io.Reader",
        method="read",
        dispatch_type=_BYTES_READER,
        args=(
            LocalRef(name=handle_name),
            LocalRef(name=buf_name),
            IntLit(type=UsizeType(), value=n),
        ),
    )


def _writer_write(handle_name: str, buf_name: str, n: int):
    return TraitCall(
        trait="core.io.Writer",
        method="write",
        dispatch_type=_BYTES_WRITER,
        args=(
            LocalRef(name=handle_name),
            LocalRef(name=buf_name),
            IntLit(type=UsizeType(), value=n),
        ),
    )


def _writer_write_strref(handle_name: str, str_const: str, n: int):
    return TraitCall(
        trait="core.io.Writer",
        method="write",
        dispatch_type=_BYTES_WRITER,
        args=(
            LocalRef(name=handle_name),
            StringRef(name=str_const),
            IntLit(type=UsizeType(), value=n),
        ),
    )


def _writer_flush(handle_name: str):
    return TraitCall(
        trait="core.io.Writer",
        method="flush",
        dispatch_type=_BYTES_WRITER,
        args=(LocalRef(name=handle_name),),
    )


def _print_result_or_err(result_local: str):
    """Match Result<usize, IoError>; print Ok(n) as `n`, Err(_) as `-1`."""
    return Match(
        scrutinee=LocalRef(name=result_local),
        arms=(
            MatchArm(
                variant="Ok", bindings=("n",),
                body=Block(stmts=(_print_lld(_zext64(LocalRef(name="n"))),)),
            ),
            MatchArm(
                variant="Err", bindings=("_e",),
                body=Block(stmts=(_print_lld(IntLit(type=I64Type(), value=-1)),)),
            ),
        ),
    )


def test_bytes_reader_reads_full_buffer():
    """Allocate a 4-byte source 'ABCD', wrap in a BytesReader, read 4
    bytes into a dest buffer, print bytes-read + each byte."""
    src = StringConstant(name=".src", value="ABCD")
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            WithArena(
                name="a",
                capacity=IntLit(type=I64Type(), value=128),
                body=Block(stmts=(
                    _arena_alloc_struct("a", "core.io.BytesReader", dest="r"),
                    Let(name="src_ptr", type=I8PtrType(),
                        init=StringRef(name=".src")),
                    *_init_bytes_reader("r", "src_ptr", 4),
                    _arena_alloc_bytes("a", 4, dest="dest"),
                    Let(name="result", type=_RESULT_USIZE_IOERROR,
                        init=_reader_read("r", "dest", 4)),
                    _print_result_or_err("result"),
                    _print_byte_at("dest", 0),
                    _print_byte_at("dest", 1),
                    _print_byte_at("dest", 2),
                    _print_byte_at("dest", 3),
                )),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        imports=(Import(module="core.io"), Import(module="mem.arena")),
        constants=(_FMT_LLD, src),
        externs=(_PRINTF,),
        functions=(main,),
    )
    assert _build_and_run(prog) == "4\n65\n66\n67\n68\n"


def test_bytes_reader_short_read_when_n_exceeds_remaining():
    """Source has 3 bytes; ask for 10. Reader should return Ok(3)."""
    src = StringConstant(name=".src", value="XYZ")
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            WithArena(
                name="a", capacity=IntLit(type=I64Type(), value=128),
                body=Block(stmts=(
                    _arena_alloc_struct("a", "core.io.BytesReader", dest="r"),
                    Let(name="src_ptr", type=I8PtrType(),
                        init=StringRef(name=".src")),
                    *_init_bytes_reader("r", "src_ptr", 3),
                    _arena_alloc_bytes("a", 16, dest="dest"),
                    Let(name="result", type=_RESULT_USIZE_IOERROR,
                        init=_reader_read("r", "dest", 10)),
                    _print_result_or_err("result"),
                )),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        imports=(Import(module="core.io"), Import(module="mem.arena")),
        constants=(_FMT_LLD, src),
        externs=(_PRINTF,),
        functions=(main,),
    )
    assert _build_and_run(prog) == "3\n"


def test_bytes_reader_eof_returns_ok_zero():
    """After reading the whole buffer, a subsequent read returns Ok(0)
    (EOF semantics per the trait spec)."""
    src = StringConstant(name=".src", value="AB")
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            WithArena(
                name="a", capacity=IntLit(type=I64Type(), value=128),
                body=Block(stmts=(
                    _arena_alloc_struct("a", "core.io.BytesReader", dest="r"),
                    Let(name="src_ptr", type=I8PtrType(),
                        init=StringRef(name=".src")),
                    *_init_bytes_reader("r", "src_ptr", 2),
                    _arena_alloc_bytes("a", 8, dest="dest"),
                    # First read drains the buffer
                    Let(name="r1", type=_RESULT_USIZE_IOERROR,
                        init=_reader_read("r", "dest", 2)),
                    _print_result_or_err("r1"),
                    # Second read should return Ok(0)
                    Let(name="r2", type=_RESULT_USIZE_IOERROR,
                        init=_reader_read("r", "dest", 4)),
                    _print_result_or_err("r2"),
                )),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        imports=(Import(module="core.io"), Import(module="mem.arena")),
        constants=(_FMT_LLD, src),
        externs=(_PRINTF,),
        functions=(main,),
    )
    assert _build_and_run(prog) == "2\n0\n"


# ---------- BytesWriter ----------

def test_bytes_writer_writes_full_buffer():
    """Write 'HELLO' into a 16-byte dest; print bytes-written and each
    written byte."""
    src = StringConstant(name=".src", value="HELLO")
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            WithArena(
                name="a", capacity=IntLit(type=I64Type(), value=128),
                body=Block(stmts=(
                    _arena_alloc_struct("a", "core.io.BytesWriter", dest="w"),
                    _arena_alloc_bytes("a", 16, dest="dest"),
                    *_init_bytes_writer("w", "dest", 16),
                    Let(name="result", type=_RESULT_USIZE_IOERROR,
                        init=_writer_write_strref("w", ".src", 5)),
                    _print_result_or_err("result"),
                    _print_byte_at("dest", 0),
                    _print_byte_at("dest", 1),
                    _print_byte_at("dest", 2),
                    _print_byte_at("dest", 3),
                    _print_byte_at("dest", 4),
                )),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        imports=(Import(module="core.io"), Import(module="mem.arena")),
        constants=(_FMT_LLD, src),
        externs=(_PRINTF,),
        functions=(main,),
    )
    # H=72, E=69, L=76, L=76, O=79
    assert _build_and_run(prog) == "5\n72\n69\n76\n76\n79\n"


def test_bytes_writer_short_write_at_capacity():
    """Cap=4; write 8 bytes. Writer takes only what fits and returns Ok(4)."""
    src = StringConstant(name=".src", value="OVERFLOW")
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            WithArena(
                name="a", capacity=IntLit(type=I64Type(), value=128),
                body=Block(stmts=(
                    _arena_alloc_struct("a", "core.io.BytesWriter", dest="w"),
                    _arena_alloc_bytes("a", 4, dest="dest"),
                    *_init_bytes_writer("w", "dest", 4),
                    Let(name="result", type=_RESULT_USIZE_IOERROR,
                        init=_writer_write_strref("w", ".src", 8)),
                    _print_result_or_err("result"),
                )),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        imports=(Import(module="core.io"), Import(module="mem.arena")),
        constants=(_FMT_LLD, src),
        externs=(_PRINTF,),
        functions=(main,),
    )
    assert _build_and_run(prog) == "4\n"


def test_bytes_writer_flush_returns_none():
    """flush is a no-op for BytesWriter — returns None (no error).
    Encoded as Option<IoError> per the trait spec deviation note."""
    main = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            WithArena(
                name="a", capacity=IntLit(type=I64Type(), value=128),
                body=Block(stmts=(
                    _arena_alloc_struct("a", "core.io.BytesWriter", dest="w"),
                    _arena_alloc_bytes("a", 4, dest="dest"),
                    *_init_bytes_writer("w", "dest", 4),
                    Let(name="result", type=_OPTION_IOERROR,
                        init=_writer_flush("w")),
                    Match(
                        scrutinee=LocalRef(name="result"),
                        arms=(
                            MatchArm(
                                variant="None",
                                body=Block(stmts=(_print_lld(IntLit(type=I64Type(), value=0)),)),
                            ),
                            MatchArm(
                                variant="Some", bindings=("_e",),
                                body=Block(stmts=(_print_lld(IntLit(type=I64Type(), value=-1)),)),
                            ),
                        ),
                    ),
                )),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(
        imports=(Import(module="core.io"), Import(module="mem.arena")),
        constants=(_FMT_LLD,),
        externs=(_PRINTF,),
        functions=(main,),
    )
    assert _build_and_run(prog) == "0\n"
