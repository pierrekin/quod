"""Unit tests for the monomorphization pass.

Each test builds a small Program with a generic struct or enum, runs
`monomorphize`, and asserts on the post-pass shape — independently of
the lowerer. End-to-end compile-and-run lives in tests/cases/lang/generics/.
"""

from __future__ import annotations

import pytest

from quod.model import (
    EnumDef,
    EnumPayloadField,
    EnumType,
    EnumVariant,
    FieldInit,
    Function,
    I32Type,
    I64Type,
    IntLit,
    Let,
    Param,
    Program,
    ReturnExpr,
    StructDef,
    StructField,
    StructInit,
    StructType,
    TypeParamRef,
)
from quod.monomorphize import monomorphize


def _box_program() -> Program:
    """`struct Box<T> { value: T }` plus a function that uses Box<i64>."""
    box_def = StructDef(
        name="Box",
        type_params=("T",),
        fields=(StructField(name="value", type=TypeParamRef(name="T")),),
    )
    fn = Function(
        name="make_box_i64",
        params=(),
        return_type=StructType(name="Box", type_args=(I64Type(),)),
        body=(
            Let(
                name="b",
                type=StructType(name="Box", type_args=(I64Type(),)),
                init=StructInit(
                    type="Box",
                    type_args=(I64Type(),),
                    fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=42)),),
                ),
            ),
            ReturnExpr(value=StructInit(
                type="Box",
                type_args=(I64Type(),),
                fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=42)),),
            )),
        ),
    )
    return Program(structs=(box_def,), functions=(fn,))


def test_monomorphize_box_i64_drops_template_and_emits_concrete():
    prog = _box_program()
    out = monomorphize(prog)

    # Box<i64> emitted, Box (template) gone.
    names = {sd.name for sd in out.structs}
    assert "Box" not in names, f"generic template Box should be dropped; got {names}"
    assert "Box<i64>" in names, f"expected Box<i64> in {names}"

    # The concrete struct has its `value` field substituted to i64.
    box_i64 = next(sd for sd in out.structs if sd.name == "Box<i64>")
    assert box_i64.type_params == ()
    assert len(box_i64.fields) == 1
    assert box_i64.fields[0].name == "value"
    assert isinstance(box_i64.fields[0].type, I64Type)


def test_monomorphize_rewrites_struct_init_and_type_refs():
    prog = _box_program()
    out = monomorphize(prog)

    fn = out.functions[0]

    # Function return_type rewrites to mangled name with empty type_args.
    assert isinstance(fn.return_type, StructType)
    assert fn.return_type.name == "Box<i64>"
    assert fn.return_type.type_args == ()

    # Let.type rewrites the same way.
    let = fn.body[0]
    assert isinstance(let, Let)
    assert isinstance(let.type, StructType)
    assert let.type.name == "Box<i64>"
    assert let.type.type_args == ()

    # StructInit.type carries the mangled name; type_args is emptied.
    assert isinstance(let.init, StructInit)
    assert let.init.type == "Box<i64>"
    assert let.init.type_args == ()


def test_monomorphize_two_distinct_instantiations_yield_two_structs():
    box_def = StructDef(
        name="Box",
        type_params=("T",),
        fields=(StructField(name="value", type=TypeParamRef(name="T")),),
    )
    fn = Function(
        name="two",
        params=(),
        return_type=I64Type(),
        body=(
            Let(
                name="a",
                type=StructType(name="Box", type_args=(I64Type(),)),
                init=StructInit(
                    type="Box",
                    type_args=(I64Type(),),
                    fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=1)),),
                ),
            ),
            Let(
                name="b",
                type=StructType(name="Box", type_args=(I32Type(),)),
                init=StructInit(
                    type="Box",
                    type_args=(I32Type(),),
                    fields=(FieldInit(name="value", value=IntLit(type=I32Type(), value=2)),),
                ),
            ),
            ReturnExpr(value=IntLit(type=I64Type(), value=0)),
        ),
    )
    prog = Program(structs=(box_def,), functions=(fn,))
    out = monomorphize(prog)

    names = {sd.name for sd in out.structs}
    assert names == {"Box<i64>", "Box<i32>"}, names


def test_monomorphize_generic_enum_and_payload_substitution():
    """`enum Option<T> { Some(value: T), None }` with one Option<i64> use."""
    opt_def = EnumDef(
        name="Option",
        type_params=("T",),
        variants=(
            EnumVariant(name="Some", fields=(EnumPayloadField(name="value", type=TypeParamRef(name="T")),)),
            EnumVariant(name="None"),
        ),
    )
    fn = Function(
        name="dummy",
        params=(Param(name="x", type=EnumType(name="Option", type_args=(I64Type(),))),),
        return_type=I64Type(),
        body=(ReturnExpr(value=IntLit(type=I64Type(), value=0)),),
    )
    prog = Program(enums=(opt_def,), functions=(fn,))
    out = monomorphize(prog)

    enum_names = {ed.name for ed in out.enums}
    assert "Option" not in enum_names
    assert "Option<i64>" in enum_names

    # Some variant payload substituted to i64.
    opt_i64 = next(ed for ed in out.enums if ed.name == "Option<i64>")
    some = opt_i64.variant("Some")
    assert some is not None
    assert isinstance(some.fields[0].type, I64Type)

    # Function param type rewritten.
    assert fn.params == prog.functions[0].params  # original unchanged
    new_fn = out.functions[0]
    assert isinstance(new_fn.params[0].type, EnumType)
    assert new_fn.params[0].type.name == "Option<i64>"


def test_monomorphize_nested_generics():
    """`struct Pair<A, B> { fst: A, snd: B }` + use of Pair<i64, Pair<i32, i64>>."""
    pair_def = StructDef(
        name="Pair",
        type_params=("A", "B"),
        fields=(
            StructField(name="fst", type=TypeParamRef(name="A")),
            StructField(name="snd", type=TypeParamRef(name="B")),
        ),
    )
    inner = StructType(name="Pair", type_args=(I32Type(), I64Type()))
    outer = StructType(name="Pair", type_args=(I64Type(), inner))
    fn = Function(
        name="dummy",
        params=(Param(name="p", type=outer),),
        return_type=I64Type(),
        body=(ReturnExpr(value=IntLit(type=I64Type(), value=0)),),
    )
    prog = Program(structs=(pair_def,), functions=(fn,))
    out = monomorphize(prog)

    names = {sd.name for sd in out.structs}
    # Both the inner and the outer instantiations should be present.
    assert "Pair<i32,i64>" in names, names
    assert "Pair<i64,Pair<i32,i64>>" in names, names
    assert "Pair" not in names, names


def test_monomorphize_noop_for_program_with_no_generics():
    fn = Function(
        name="trivial",
        params=(),
        return_type=I64Type(),
        body=(ReturnExpr(value=IntLit(type=I64Type(), value=42)),),
    )
    prog = Program(functions=(fn,))
    out = monomorphize(prog)
    # Programs are immutable; either the same object back, or a deep-equal copy.
    assert out.functions[0].name == "trivial"
    assert out.structs == ()
    assert out.enums == ()
