"""Codegen-level tests for enums.

These are constructed-by-hand pytest cases that exercise specific
enum/match patterns at the LLVM lowering level — building tiny programs
in memory, compiling them to binaries, running them, and asserting the
observable output (typically the tag of an extracted value, printed via
printf).

Why this file exists: when we lifted the enum payload restriction
(variants can carry any type, not just scalars), we changed the lowering
from a pure-SSA insertvalue/extractvalue path to an alloca + bitcast
path. The end-to-end test suite kept passing, but a real consumer
(JSON parser refactored to ParseResult { Ok(JsonValue), Err }) read
back garbage. This file pins the codegen behavior so future changes
break loud and small instead of silent and big.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from quod.lower import compile_program
from quod.model import (
    Assign,
    BinOp,
    Call,
    EnumDef,
    EnumInit,
    EnumPayloadField,
    EnumType,
    EnumVariant,
    ExprStmt,
    ExternFunction,
    FieldInit,
    FieldRead,
    Function,
    I1Type,
    I8PtrType,
    I8Type,
    I32Type,
    I64Type,
    If,
    IntLit,
    LibcLinkage,
    Let,
    LocalRef,
    Match,
    MatchArm,
    NullPtr,
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
    TryExpr,
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
    """Compile + run a program; return its stdout as text."""
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
    """Build an ExprStmt that calls printf("%lld\\n", value)."""
    return ExprStmt(value=Call(
        function="printf",
        args=(StringRef(name=".fmt_int"), value),
    ))


# ---------- 1. scalar payload, single function ----------

def test_scalar_payload_match_in_same_fn():
    """Minimum viable enum: Maybe { None, Some(value: i64) }, constructed
    and matched in the same function."""
    maybe = EnumDef(
        name="Maybe",
        variants=(
            EnumVariant(name="None"),
            EnumVariant(name="Some", fields=(EnumPayloadField(name="value", type=I64Type()),)),
        ),
    )
    main = Function(
        name="main",
        return_type=I32Type(),
        body=(
            Let(
                name="m",
                type=EnumType(name="Maybe"),
                init=EnumInit(
                    enum="Maybe", variant="Some",
                    fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=42)),),
                ),
            ),
            Match(
                scrutinee=LocalRef(name="m"),
                arms=(
                    MatchArm(
                        variant="Some", bindings=("v",),
                        body=(_print_int_call(LocalRef(name="v")),),
                    ),
                    MatchArm(
                        variant="None",
                        body=(_print_int_call(IntLit(type=I64Type(), value=0)),),
                    ),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(
        constants=(_FMT_INT,),
        externs=(_PRINTF,),
        enums=(maybe,),
        functions=(main,),
    )
    assert _build_and_run(prog) == "42\n"


# ---------- 2. scalar payload, return + match across fns ----------

def test_scalar_payload_round_trip_via_return():
    """fn make() returns Some(42); main matches the returned value."""
    maybe = EnumDef(
        name="Maybe",
        variants=(
            EnumVariant(name="None"),
            EnumVariant(name="Some", fields=(EnumPayloadField(name="value", type=I64Type()),)),
        ),
    )
    make = Function(
        name="make",
        return_type=EnumType(name="Maybe"),
        body=(ReturnExpr(value=EnumInit(
            enum="Maybe", variant="Some",
            fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=42)),),
        )),),
    )
    main = Function(
        name="main",
        return_type=I32Type(),
        body=(
            Let(
                name="m", type=EnumType(name="Maybe"),
                init=Call(function="make"),
            ),
            Match(
                scrutinee=LocalRef(name="m"),
                arms=(
                    MatchArm(
                        variant="Some", bindings=("v",),
                        body=(_print_int_call(LocalRef(name="v")),),
                    ),
                    MatchArm(
                        variant="None",
                        body=(_print_int_call(IntLit(type=I64Type(), value=0)),),
                    ),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(
        constants=(_FMT_INT,),
        externs=(_PRINTF,),
        enums=(maybe,),
        functions=(make, main),
    )
    assert _build_and_run(prog) == "42\n"


# ---------- 3. multi-field scalar payload, return + match ----------

def test_multi_scalar_payload_via_return():
    """Pair { x: i64, y: i64 } — multiple scalar fields in one variant."""
    enum_def = EnumDef(
        name="E",
        variants=(EnumVariant(name="V", fields=(
            EnumPayloadField(name="x", type=I64Type()),
            EnumPayloadField(name="y", type=I64Type()),
        )),),
    )
    make = Function(
        name="make", return_type=EnumType(name="E"),
        body=(ReturnExpr(value=EnumInit(
            enum="E", variant="V",
            fields=(
                FieldInit(name="x", value=IntLit(type=I64Type(), value=7)),
                FieldInit(name="y", value=IntLit(type=I64Type(), value=11)),
            ),
        )),),
    )
    main = Function(
        name="main", return_type=I32Type(),
        body=(
            Let(name="e", type=EnumType(name="E"), init=Call(function="make")),
            Match(scrutinee=LocalRef(name="e"), arms=(
                MatchArm(
                    variant="V", bindings=("a", "b"),
                    body=(_print_int_call(BinOp(
                        op="add", lhs=LocalRef(name="a"), rhs=LocalRef(name="b"),
                    )),),
                ),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(
        constants=(_FMT_INT,), externs=(_PRINTF,), enums=(enum_def,),
        functions=(make, main),
    )
    assert _build_and_run(prog) == "18\n"


# ---------- 4. struct payload, return + match ----------

def test_struct_payload_via_return():
    """Variant carries a named StructType."""
    pair = StructDef(name="Pair", fields=(
        StructField(name="x", type=I64Type()),
        StructField(name="y", type=I64Type()),
    ))
    enum_def = EnumDef(name="E", variants=(
        EnumVariant(name="Wrap", fields=(EnumPayloadField(name="p", type=StructType(name="Pair")),)),
    ))
    make = Function(
        name="make", return_type=EnumType(name="E"),
        body=(ReturnExpr(value=EnumInit(
            enum="E", variant="Wrap",
            fields=(FieldInit(name="p", value=StructInit(
                type="Pair", fields=(
                    FieldInit(name="x", value=IntLit(type=I64Type(), value=100)),
                    FieldInit(name="y", value=IntLit(type=I64Type(), value=23)),
                ),
            )),),
        )),),
    )
    main = Function(
        name="main", return_type=I32Type(),
        body=(
            Let(name="e", type=EnumType(name="E"), init=Call(function="make")),
            Match(scrutinee=LocalRef(name="e"), arms=(
                MatchArm(
                    variant="Wrap", bindings=("p",),
                    body=(_print_int_call(BinOp(
                        op="add",
                        lhs=FieldRead(value=LocalRef(name="p"), name="x"),
                        rhs=FieldRead(value=LocalRef(name="p"), name="y"),
                    )),),
                ),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(
        constants=(_FMT_INT,), externs=(_PRINTF,), structs=(pair,), enums=(enum_def,),
        functions=(make, main),
    )
    assert _build_and_run(prog) == "123\n"


# ---------- 5. ENUM payload — the case that's failing in the parser ----------

def test_enum_payload_via_return():
    """Outer { Wrap(inner: Inner) } where Inner is itself an enum.
    Mirrors `ParseResult { Ok(value: JsonValue), Err }` shape."""
    inner = EnumDef(name="Inner", variants=(
        EnumVariant(name="Tag0"),
        EnumVariant(name="Tag1", fields=(EnumPayloadField(name="n", type=I64Type()),)),
    ))
    outer = EnumDef(name="Outer", variants=(
        EnumVariant(name="Wrap", fields=(EnumPayloadField(name="inner", type=EnumType(name="Inner")),)),
        EnumVariant(name="Empty"),
    ))
    make = Function(
        name="make", return_type=EnumType(name="Outer"),
        body=(ReturnExpr(value=EnumInit(
            enum="Outer", variant="Wrap",
            fields=(FieldInit(name="inner", value=EnumInit(
                enum="Inner", variant="Tag1",
                fields=(FieldInit(name="n", value=IntLit(type=I64Type(), value=99)),),
            )),),
        )),),
    )
    main = Function(
        name="main", return_type=I32Type(),
        body=(
            Let(name="o", type=EnumType(name="Outer"), init=Call(function="make")),
            Match(scrutinee=LocalRef(name="o"), arms=(
                MatchArm(
                    variant="Wrap", bindings=("inner",),
                    body=(Match(scrutinee=LocalRef(name="inner"), arms=(
                        MatchArm(
                            variant="Tag1", bindings=("n",),
                            body=(_print_int_call(LocalRef(name="n")),),
                        ),
                        MatchArm(
                            variant="Tag0",
                            body=(_print_int_call(IntLit(type=I64Type(), value=-1)),),
                        ),
                    )),),
                ),
                MatchArm(
                    variant="Empty",
                    body=(_print_int_call(IntLit(type=I64Type(), value=-2)),),
                ),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(
        constants=(_FMT_INT,), externs=(_PRINTF,), enums=(inner, outer),
        functions=(make, main),
    )
    assert _build_and_run(prog) == "99\n"


# ---------- 6. enum payload, multiple round-trips through fns ----------

def test_enum_payload_through_chain():
    """make() -> forward() -> main. Mimics parse() -> parse_value() ->
    parse_object() chain in the JSON parser."""
    inner = EnumDef(name="Inner", variants=(
        EnumVariant(name="N", fields=(EnumPayloadField(name="n", type=I64Type()),)),
    ))
    outer = EnumDef(name="Outer", variants=(
        EnumVariant(name="W", fields=(EnumPayloadField(name="i", type=EnumType(name="Inner")),)),
    ))
    make = Function(
        name="make", return_type=EnumType(name="Outer"),
        body=(ReturnExpr(value=EnumInit(
            enum="Outer", variant="W",
            fields=(FieldInit(name="i", value=EnumInit(
                enum="Inner", variant="N",
                fields=(FieldInit(name="n", value=IntLit(type=I64Type(), value=77)),),
            )),),
        )),),
    )
    forward = Function(
        name="forward", return_type=EnumType(name="Outer"),
        body=(ReturnExpr(value=Call(function="make")),),
    )
    main = Function(
        name="main", return_type=I32Type(),
        body=(
            Let(name="o", type=EnumType(name="Outer"), init=Call(function="forward")),
            Match(scrutinee=LocalRef(name="o"), arms=(
                MatchArm(
                    variant="W", bindings=("i",),
                    body=(Match(scrutinee=LocalRef(name="i"), arms=(
                        MatchArm(
                            variant="N", bindings=("n",),
                            body=(_print_int_call(LocalRef(name="n")),),
                        ),
                    )),),
                ),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(
        constants=(_FMT_INT,), externs=(_PRINTF,), enums=(inner, outer),
        functions=(make, forward, main),
    )
    assert _build_and_run(prog) == "77\n"


# ---------- 7. enum payload, store + load through memory ----------

def test_enum_payload_through_arena_memory():
    """Construct an Outer, store it via i8* into arena-style memory,
    load it back, match. This is what the JSON parser's array/object
    storage does — write JsonValue values into a contiguous buffer,
    later iterate and load each."""
    # We don't have a full arena setup here, so use a stack alloca
    # (via Let-of-pointer + store). Skipping — alloca-on-stack via
    # quod doesn't exist as a builtin. The best proxy is just to use
    # the same value across two sequential Match operations, which
    # exercises the load+store path inside lower_stmt.
    inner = EnumDef(name="Inner", variants=(
        EnumVariant(name="N", fields=(EnumPayloadField(name="n", type=I64Type()),)),
    ))
    outer = EnumDef(name="Outer", variants=(
        EnumVariant(name="W", fields=(EnumPayloadField(name="i", type=EnumType(name="Inner")),)),
    ))
    main = Function(
        name="main", return_type=I32Type(),
        body=(
            Let(name="o", type=EnumType(name="Outer"), init=EnumInit(
                enum="Outer", variant="W",
                fields=(FieldInit(name="i", value=EnumInit(
                    enum="Inner", variant="N",
                    fields=(FieldInit(name="n", value=IntLit(type=I64Type(), value=55)),),
                )),),
            )),
            # Match the same value twice — second match must see the same
            # bytes. If allocas-via-EnumInit reused stack slots improperly,
            # this would observe corruption.
            Match(scrutinee=LocalRef(name="o"), arms=(
                MatchArm(
                    variant="W", bindings=("i1",),
                    body=(Match(scrutinee=LocalRef(name="i1"), arms=(
                        MatchArm(
                            variant="N", bindings=("n1",),
                            body=(_print_int_call(LocalRef(name="n1")),),
                        ),
                    )),),
                ),
            )),
            Match(scrutinee=LocalRef(name="o"), arms=(
                MatchArm(
                    variant="W", bindings=("i2",),
                    body=(Match(scrutinee=LocalRef(name="i2"), arms=(
                        MatchArm(
                            variant="N", bindings=("n2",),
                            body=(_print_int_call(LocalRef(name="n2")),),
                        ),
                    )),),
                ),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(
        constants=(_FMT_INT,), externs=(_PRINTF,), enums=(inner, outer),
        functions=(main,),
    )
    assert _build_and_run(prog) == "55\n55\n"


# ---------- 8. JSON-parser-shape: 6-variant enum with widest variant 24 bytes ----------

def test_json_parser_shape_round_trip():
    """Replicates the failing case: a 6-variant enum (like JsonValue)
    where the largest variant has 3 i64-sized fields = 24 bytes,
    wrapped in a 2-variant Result enum (like ParseResult). Construct
    the largest variant and pull it back out across a function boundary."""
    big = EnumDef(name="Big", variants=(
        EnumVariant(name="V0"),
        EnumVariant(name="V1", fields=(EnumPayloadField(name="b", type=I1Type()),)),
        EnumVariant(name="V2", fields=(EnumPayloadField(name="n", type=I64Type()),)),
        EnumVariant(name="V3", fields=(
            EnumPayloadField(name="ptr", type=I8PtrType()),
            EnumPayloadField(name="len", type=I64Type()),
        )),
        EnumVariant(name="V4", fields=(
            EnumPayloadField(name="items", type=I8PtrType()),
            EnumPayloadField(name="count", type=I64Type()),
        )),
        # The widest variant — 3 fields, 24 bytes total
        EnumVariant(name="V5", fields=(
            EnumPayloadField(name="a", type=I8PtrType()),
            EnumPayloadField(name="b", type=I8PtrType()),
            EnumPayloadField(name="c", type=I64Type()),
        )),
    ))
    result = EnumDef(name="Result", variants=(
        EnumVariant(name="Ok", fields=(EnumPayloadField(name="value", type=EnumType(name="Big")),)),
        EnumVariant(name="Err"),
    ))
    # Construct Big::V5 (the widest variant) and wrap in Result::Ok
    make = Function(
        name="make", return_type=EnumType(name="Result"),
        body=(ReturnExpr(value=EnumInit(
            enum="Result", variant="Ok",
            fields=(FieldInit(name="value", value=EnumInit(
                enum="Big", variant="V5",
                fields=(
                    FieldInit(name="a", value=NullPtr()),
                    FieldInit(name="b", value=NullPtr()),
                    FieldInit(name="c", value=IntLit(type=I64Type(), value=42)),
                ),
            )),),
        )),),
    )
    main = Function(
        name="main", return_type=I32Type(),
        body=(
            Let(name="r", type=EnumType(name="Result"), init=Call(function="make")),
            Match(scrutinee=LocalRef(name="r"), arms=(
                MatchArm(
                    variant="Ok", bindings=("v",),
                    body=(Match(scrutinee=LocalRef(name="v"), arms=(
                        MatchArm(
                            variant="V5", bindings=("aa", "bb", "cc"),
                            body=(_print_int_call(LocalRef(name="cc")),),
                        ),
                        MatchArm(
                            variant="_", body=(_print_int_call(IntLit(type=I64Type(), value=-1)),),
                        ),
                    )),),
                ),
                MatchArm(
                    variant="Err",
                    body=(_print_int_call(IntLit(type=I64Type(), value=-2)),),
                ),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(
        constants=(_FMT_INT,), externs=(_PRINTF,), enums=(big, result),
        functions=(make, main),
    )
    assert _build_and_run(prog) == "42\n"


# ---------- 9. Enum-of-enum AROUND with_arena boundary ----------

def test_enum_of_enum_inside_with_arena():
    """Caller wraps everything in `with_arena`. The arena_drop is
    inserted before main's return; if we're getting alloca lifetime
    interactions wrong, this would surface."""
    inner = EnumDef(name="Inner", variants=(
        EnumVariant(name="N", fields=(EnumPayloadField(name="n", type=I64Type()),)),
    ))
    outer = EnumDef(name="Outer", variants=(
        EnumVariant(name="W", fields=(EnumPayloadField(name="i", type=EnumType(name="Inner")),)),
        EnumVariant(name="E"),
    ))
    make = Function(
        name="make", return_type=EnumType(name="Outer"),
        body=(ReturnExpr(value=EnumInit(
            enum="Outer", variant="W",
            fields=(FieldInit(name="i", value=EnumInit(
                enum="Inner", variant="N",
                fields=(FieldInit(name="n", value=IntLit(type=I64Type(), value=88)),),
            )),),
        )),),
    )
    main = Function(
        name="main", return_type=I32Type(),
        body=(
            WithArena(
                name="a",
                capacity=IntLit(type=I64Type(), value=4096),
                body=(
                    Let(name="o", type=EnumType(name="Outer"), init=Call(function="make")),
                    Match(scrutinee=LocalRef(name="o"), arms=(
                        MatchArm(
                            variant="W", bindings=("i",),
                            body=(Match(scrutinee=LocalRef(name="i"), arms=(
                                MatchArm(
                                    variant="N", bindings=("n",),
                                    body=(_print_int_call(LocalRef(name="n")),),
                                ),
                            )),),
                        ),
                        MatchArm(
                            variant="E",
                            body=(_print_int_call(IntLit(type=I64Type(), value=-1)),),
                        ),
                    )),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(
        constants=(_FMT_INT,), externs=(_PRINTF,), enums=(inner, outer),
        functions=(make, main),
    )
    assert _build_and_run(prog) == "88\n"


# ---------- 10. Caller-side: chain Inner construct THROUGH a function param ----------

def test_enum_of_enum_via_param():
    """Pass an Outer enum AS a parameter. Receiver matches on it.
    parse_value passes parser results around through param/return chains."""
    inner = EnumDef(name="Inner", variants=(
        EnumVariant(name="N", fields=(EnumPayloadField(name="n", type=I64Type()),)),
    ))
    outer = EnumDef(name="Outer", variants=(
        EnumVariant(name="W", fields=(EnumPayloadField(name="i", type=EnumType(name="Inner")),)),
    ))
    inspect = Function(
        name="inspect",
        params=(Param(name="o", type=EnumType(name="Outer")),),
        return_type=I64Type(),
        body=(Match(scrutinee=ParamRef(name="o"), arms=(
            MatchArm(
                variant="W", bindings=("i",),
                body=(Match(scrutinee=LocalRef(name="i"), arms=(
                    MatchArm(
                        variant="N", bindings=("n",),
                        body=(ReturnExpr(value=LocalRef(name="n")),),
                    ),
                )),),
            ),
        )),),
    )
    main = Function(
        name="main", return_type=I32Type(),
        body=(
            Let(name="o", type=EnumType(name="Outer"), init=EnumInit(
                enum="Outer", variant="W",
                fields=(FieldInit(name="i", value=EnumInit(
                    enum="Inner", variant="N",
                    fields=(FieldInit(name="n", value=IntLit(type=I64Type(), value=33)),),
                )),),
            )),
            _print_int_call(Call(function="inspect", args=(LocalRef(name="o"),))),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(
        constants=(_FMT_INT,), externs=(_PRINTF,), enums=(inner, outer),
        functions=(inspect, main),
    )
    assert _build_and_run(prog) == "33\n"


# ---------- 11. Mid-flight store + load through an i8* (arena-style) ----------

def test_enum_round_trip_through_i8_ptr():
    """The JSON parser stores parsed values into an arena buffer via
    `store(ptr + offset, value)` and reads them back via `load[T](ptr +
    offset)`. This tests that round trip directly, since the bug
    surfaced in object_get / array_get."""
    from quod.model import Load, PtrOffset, Store
    inner = EnumDef(name="Inner", variants=(
        EnumVariant(name="N", fields=(EnumPayloadField(name="n", type=I64Type()),)),
    ))
    outer = EnumDef(name="Outer", variants=(
        EnumVariant(name="W", fields=(EnumPayloadField(name="i", type=EnumType(name="Inner")),)),
    ))
    main = Function(
        name="main", return_type=I32Type(),
        body=(
            WithArena(
                name="a", capacity=IntLit(type=I64Type(), value=4096),
                body=(
                    # Allocate a buffer big enough for one Outer
                    Let(name="buf", type=I8PtrType(), init=Call(
                        function="alloc.arena.alloc",
                        args=(LocalRef(name="a"), IntLit(type=I64Type(), value=128)),
                    )),
                    # Construct an Outer and store it through the buffer ptr
                    Store(ptr=LocalRef(name="buf"), value=EnumInit(
                        enum="Outer", variant="W",
                        fields=(FieldInit(name="i", value=EnumInit(
                            enum="Inner", variant="N",
                            fields=(FieldInit(name="n", value=IntLit(type=I64Type(), value=66)),),
                        )),),
                    )),
                    # Load it back as Outer
                    Let(name="loaded", type=EnumType(name="Outer"), init=Load(
                        ptr=LocalRef(name="buf"),
                        type=EnumType(name="Outer"),
                    )),
                    Match(scrutinee=LocalRef(name="loaded"), arms=(
                        MatchArm(
                            variant="W", bindings=("i",),
                            body=(Match(scrutinee=LocalRef(name="i"), arms=(
                                MatchArm(
                                    variant="N", bindings=("n",),
                                    body=(_print_int_call(LocalRef(name="n")),),
                                ),
                            )),),
                        ),
                    )),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(
        constants=(_FMT_INT,), externs=(_PRINTF,),
        imports=("alloc.arena",),
        enums=(inner, outer), functions=(main,),
    )
    assert _build_and_run(prog) == "66\n"


# ---------- 12. ? happy path: returns the payload field ----------

def _maybe_enum() -> EnumDef:
    return EnumDef(
        name="Maybe",
        variants=(
            EnumVariant(name="Some", fields=(EnumPayloadField(name="value", type=I64Type()),)),
            EnumVariant(name="None"),
        ),
    )


def test_try_happy_path_extracts_payload():
    """`Some(42)? + 1` evaluates to 43 (payload + 1)."""
    maybe = _maybe_enum()
    inner_call = Function(
        name="get_some", return_type=EnumType(name="Maybe"),
        body=(ReturnExpr(value=EnumInit(
            enum="Maybe", variant="Some",
            fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=42)),),
        )),),
    )
    use = Function(
        name="use_it", return_type=EnumType(name="Maybe"),
        body=(
            Let(name="v", type=I64Type(), init=TryExpr(value=Call(function="get_some"))),
            ReturnExpr(value=EnumInit(
                enum="Maybe", variant="Some",
                fields=(FieldInit(name="value", value=BinOp(
                    op="add",
                    lhs=LocalRef(name="v"),
                    rhs=IntLit(type=I64Type(), value=1),
                )),),
            )),
        ),
    )
    main = Function(
        name="main", return_type=I32Type(),
        body=(
            Let(name="r", type=EnumType(name="Maybe"), init=Call(function="use_it")),
            Match(scrutinee=LocalRef(name="r"), arms=(
                MatchArm(variant="Some", bindings=("v",),
                         body=(_print_int_call(LocalRef(name="v")),)),
                MatchArm(variant="None",
                         body=(_print_int_call(IntLit(type=I64Type(), value=-1)),)),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(
        constants=(_FMT_INT,), externs=(_PRINTF,), enums=(maybe,),
        functions=(inner_call, use, main),
    )
    assert _build_and_run(prog) == "43\n"


# ---------- 13. ? sad path: propagates the sad variant ----------

def test_try_sad_path_propagates():
    """`get_none()?` makes the calling function return None."""
    maybe = _maybe_enum()
    get_none = Function(
        name="get_none", return_type=EnumType(name="Maybe"),
        body=(ReturnExpr(value=EnumInit(enum="Maybe", variant="None")),),
    )
    use = Function(
        name="use_it", return_type=EnumType(name="Maybe"),
        body=(
            Let(name="v", type=I64Type(), init=TryExpr(value=Call(function="get_none"))),
            # Unreachable — the ? above propagates None back up.
            ReturnExpr(value=EnumInit(
                enum="Maybe", variant="Some",
                fields=(FieldInit(name="value", value=LocalRef(name="v")),),
            )),
        ),
    )
    main = Function(
        name="main", return_type=I32Type(),
        body=(
            Let(name="r", type=EnumType(name="Maybe"), init=Call(function="use_it")),
            Match(scrutinee=LocalRef(name="r"), arms=(
                MatchArm(variant="Some", bindings=("v",),
                         body=(_print_int_call(LocalRef(name="v")),)),
                MatchArm(variant="None",
                         body=(_print_int_call(IntLit(type=I64Type(), value=-99)),)),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(
        constants=(_FMT_INT,), externs=(_PRINTF,), enums=(maybe,),
        functions=(get_none, use, main),
    )
    assert _build_and_run(prog) == "-99\n"


# ---------- 14. ? on enum-payload variant (the JSON parser shape) ----------

def test_try_on_result_carrying_enum_payload():
    """ParseResult { Ok(value: JsonValue), Err } — `?` should extract
    the JsonValue payload on happy path, propagate Err otherwise."""
    inner = EnumDef(name="Inner", variants=(
        EnumVariant(name="N", fields=(EnumPayloadField(name="n", type=I64Type()),)),
    ))
    result = EnumDef(name="R", variants=(
        EnumVariant(name="Ok", fields=(EnumPayloadField(name="value", type=EnumType(name="Inner")),)),
        EnumVariant(name="Err"),
    ))
    make_ok = Function(
        name="make_ok", return_type=EnumType(name="R"),
        body=(ReturnExpr(value=EnumInit(
            enum="R", variant="Ok",
            fields=(FieldInit(name="value", value=EnumInit(
                enum="Inner", variant="N",
                fields=(FieldInit(name="n", value=IntLit(type=I64Type(), value=77)),),
            )),),
        )),),
    )
    use = Function(
        name="use_it", return_type=EnumType(name="R"),
        body=(
            Let(name="i", type=EnumType(name="Inner"),
                init=TryExpr(value=Call(function="make_ok"))),
            ReturnExpr(value=EnumInit(
                enum="R", variant="Ok",
                fields=(FieldInit(name="value", value=LocalRef(name="i")),),
            )),
        ),
    )
    main = Function(
        name="main", return_type=I32Type(),
        body=(
            Let(name="r", type=EnumType(name="R"), init=Call(function="use_it")),
            Match(scrutinee=LocalRef(name="r"), arms=(
                MatchArm(variant="Ok", bindings=("v",), body=(
                    Match(scrutinee=LocalRef(name="v"), arms=(
                        MatchArm(variant="N", bindings=("n",),
                                 body=(_print_int_call(LocalRef(name="n")),)),
                    )),
                )),
                MatchArm(variant="Err",
                         body=(_print_int_call(IntLit(type=I64Type(), value=-1)),)),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(
        constants=(_FMT_INT,), externs=(_PRINTF,), enums=(inner, result),
        functions=(make_ok, use, main),
    )
    assert _build_and_run(prog) == "77\n"


# ---------- 15. validator rejects ?-ineligible enums ----------

def test_try_rejects_ineligible_enum():
    """Three-variant enum is not ?-eligible; lower must error."""
    bad = EnumDef(name="Bad", variants=(
        EnumVariant(name="A"),
        EnumVariant(name="B"),
        EnumVariant(name="C"),
    ))
    make = Function(
        name="make", return_type=EnumType(name="Bad"),
        body=(ReturnExpr(value=EnumInit(enum="Bad", variant="A")),),
    )
    use = Function(
        name="use_it", return_type=EnumType(name="Bad"),
        body=(
            ExprStmt(value=TryExpr(value=Call(function="make"))),
            ReturnExpr(value=EnumInit(enum="Bad", variant="A")),
        ),
    )
    main = Function(
        name="main", return_type=I32Type(),
        body=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),),
    )
    prog = Program(
        constants=(_FMT_INT,), externs=(_PRINTF,), enums=(bad,),
        functions=(make, use, main),
    )
    with pytest.raises(Exception, match="not \\?-eligible"):
        _build_and_run(prog)


def test_try_rejects_mismatched_return_type():
    """Function returns Inner but ? is on R — should error."""
    inner = _maybe_enum()
    other = EnumDef(name="R", variants=(
        EnumVariant(name="Ok", fields=(EnumPayloadField(name="value", type=I64Type()),)),
        EnumVariant(name="Err"),
    ))
    make = Function(
        name="make", return_type=EnumType(name="R"),
        body=(ReturnExpr(value=EnumInit(
            enum="R", variant="Ok",
            fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=1)),),
        )),),
    )
    bad_use = Function(
        name="bad_use", return_type=EnumType(name="Maybe"),
        body=(
            Let(name="v", type=I64Type(), init=TryExpr(value=Call(function="make"))),
            ReturnExpr(value=EnumInit(enum="Maybe", variant="None")),
        ),
    )
    main = Function(
        name="main", return_type=I32Type(),
        body=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),),
    )
    prog = Program(
        constants=(_FMT_INT,), externs=(_PRINTF,), enums=(inner, other),
        functions=(make, bad_use, main),
    )
    with pytest.raises(Exception, match="requires the enclosing function"):
        _build_and_run(prog)
