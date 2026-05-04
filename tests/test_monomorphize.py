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
    TypeParam,
    TypeParamRef,
)
from quod.monomorphize import monomorphize


def _box_program() -> Program:
    """`struct Box<T> { value: T }` plus a function that uses Box<i64>."""
    box_def = StructDef(
        name="Box",
        type_params=(TypeParam(name="T"),),
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
        type_params=(TypeParam(name="T"),),
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
        type_params=(TypeParam(name="T"),),
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
        type_params=(TypeParam(name="A"), TypeParam(name="B"),),
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


# ---------- Generic functions ----------

from quod.model import (
    Call,
    FieldRead,
    ImplDef,
    ParamRef,
    SelfType,
    TraitCall,
    TraitDef,
    TraitMethodSig,
)


def test_monomorphize_generic_function_drops_template_emits_concrete():
    """`fn id<T>(x: T) -> T { return x }` plus a `Call(id, type_args=[i64])`."""
    id_fn = Function(
        name="id",
        type_params=(TypeParam(name="T"),),
        params=(Param(name="x", type=TypeParamRef(name="T")),),
        return_type=TypeParamRef(name="T"),
        body=(ReturnExpr(value=ParamRef(name="x")),),
    )
    main_fn = Function(
        name="main",
        return_type=I64Type(),
        body=(
            ReturnExpr(value=Call(
                function="id",
                type_args=(I64Type(),),
                args=(IntLit(type=I64Type(), value=42),),
            )),
        ),
    )
    prog = Program(functions=(id_fn, main_fn))
    out = monomorphize(prog)

    fn_names = {fn.name for fn in out.functions}
    assert "id" not in fn_names, f"generic template id should be dropped; got {fn_names}"
    assert "id<i64>" in fn_names, f"expected id<i64>; got {fn_names}"

    # The concrete instance has T → i64 throughout.
    id_i64 = next(fn for fn in out.functions if fn.name == "id<i64>")
    assert id_i64.type_params == ()
    assert isinstance(id_i64.return_type, I64Type)
    assert isinstance(id_i64.params[0].type, I64Type)

    # main's Call gets rewritten to mangled name with empty type_args.
    main_after = next(fn for fn in out.functions if fn.name == "main")
    body0 = main_after.body[0]
    assert isinstance(body0, ReturnExpr)
    assert isinstance(body0.value, Call)
    assert body0.value.function == "id<i64>"
    assert body0.value.type_args == ()


def test_monomorphize_generic_function_two_instantiations():
    id_fn = Function(
        name="id",
        type_params=(TypeParam(name="T"),),
        params=(Param(name="x", type=TypeParamRef(name="T")),),
        return_type=TypeParamRef(name="T"),
        body=(ReturnExpr(value=ParamRef(name="x")),),
    )
    main_fn = Function(
        name="main",
        return_type=I64Type(),
        body=(
            Let(
                name="a", type=I64Type(),
                init=Call(function="id", type_args=(I64Type(),),
                          args=(IntLit(type=I64Type(), value=1),)),
            ),
            Let(
                name="b", type=I32Type(),
                init=Call(function="id", type_args=(I32Type(),),
                          args=(IntLit(type=I32Type(), value=2),)),
            ),
            ReturnExpr(value=IntLit(type=I64Type(), value=0)),
        ),
    )
    prog = Program(functions=(id_fn, main_fn))
    out = monomorphize(prog)
    fn_names = {fn.name for fn in out.functions}
    assert {"id<i64>", "id<i32>"} <= fn_names
    assert "id" not in fn_names


def test_monomorphize_generic_function_referencing_generic_struct():
    """`struct Box<T> { value: T }` + `fn make<T>(v: T) -> Box<T> { ... }`.
    Calling make<i64>(...) should produce both `make<i64>` AND `Box<i64>`."""
    box_def = StructDef(
        name="Box",
        type_params=(TypeParam(name="T"),),
        fields=(StructField(name="value", type=TypeParamRef(name="T")),),
    )
    make_fn = Function(
        name="make",
        type_params=(TypeParam(name="T"),),
        params=(Param(name="v", type=TypeParamRef(name="T")),),
        return_type=StructType(name="Box", type_args=(TypeParamRef(name="T"),)),
        body=(ReturnExpr(value=StructInit(
            type="Box",
            type_args=(TypeParamRef(name="T"),),
            fields=(FieldInit(name="value", value=ParamRef(name="v")),),
        )),),
    )
    main_fn = Function(
        name="main",
        return_type=I64Type(),
        body=(
            ReturnExpr(value=Call(
                function="make",
                type_args=(I64Type(),),
                args=(IntLit(type=I64Type(), value=42),),
            )),
        ),
    )
    prog = Program(structs=(box_def,), functions=(make_fn, main_fn))
    out = monomorphize(prog)

    struct_names = {sd.name for sd in out.structs}
    fn_names = {fn.name for fn in out.functions}
    # Generic templates dropped, concretes present.
    assert "Box" not in struct_names
    assert "Box<i64>" in struct_names
    assert "make" not in fn_names
    assert "make<i64>" in fn_names

    # The make<i64> body's StructInit should be rewritten to "Box<i64>".
    make_i64 = next(fn for fn in out.functions if fn.name == "make<i64>")
    body = make_i64.body[0]
    assert isinstance(body, ReturnExpr)
    assert isinstance(body.value, StructInit)
    assert body.value.type == "Box<i64>"
    assert body.value.type_args == ()


# ---------- Traits + impls ----------

def _counter_program():
    counter = StructDef(name="Counter", fields=(StructField(name="count", type=I32Type()),))
    add_trait = TraitDef(name="Add", methods=(
        TraitMethodSig(
            name="add",
            params=(Param(name="self", type=SelfType()), Param(name="n", type=I32Type())),
            return_type=I32Type(),
        ),
    ))
    add_impl = ImplDef(
        trait="Add",
        for_type=StructType(name="Counter"),
        methods=(
            Function(
                name="add",
                params=(Param(name="self", type=SelfType()), Param(name="n", type=I32Type())),
                return_type=I32Type(),
                body=(ReturnExpr(value=ParamRef(name="n")),),  # body doesn't matter for these checks
            ),
        ),
    )
    return counter, add_trait, add_impl


def test_impldef_substitutes_self_in_signatures():
    """ImplDef's validator must substitute Self → for_type in params and return_type."""
    counter, add_trait, add_impl = _counter_program()
    method = add_impl.methods[0]
    # self's type is no longer SelfType — it's been rewritten to the for_type.
    assert isinstance(method.params[0].type, StructType)
    assert method.params[0].type.name == "Counter"
    # n stays i32.
    assert isinstance(method.params[1].type, I32Type)
    # return_type is i32 (no Self there to substitute).
    assert isinstance(method.return_type, I32Type)


def test_impldef_substitutes_self_in_body():
    """Self appearing in method-body Type positions (e.g. Let.type) gets
    substituted to for_type by ImplDef's validator. Without this, the
    lowerer would crash on an unhandled SelfType."""
    counter = StructDef(name="Counter", fields=(StructField(name="count", type=I32Type()),))
    add_trait = TraitDef(name="Add", methods=(
        TraitMethodSig(
            name="add",
            params=(Param(name="self", type=SelfType()), Param(name="n", type=I32Type())),
            return_type=I32Type(),
        ),
    ))
    # Body uses `let s: Self = self` — Self in Let.type, must be substituted.
    impl = ImplDef(
        trait="Add",
        for_type=StructType(name="Counter"),
        methods=(
            Function(
                name="add",
                params=(Param(name="self", type=SelfType()), Param(name="n", type=I32Type())),
                return_type=I32Type(),
                body=(
                    Let(name="s", type=SelfType(), init=ParamRef(name="self")),
                    ReturnExpr(value=BinOp(
                        op="add",
                        lhs=FieldRead(value=LocalRef(name="s"), name="count"),
                        rhs=ParamRef(name="n"),
                    )),
                ),
            ),
        ),
    )
    # Walk the body and assert no SelfType remains anywhere.
    let = impl.methods[0].body[0]
    assert isinstance(let, Let)
    assert isinstance(let.type, StructType)
    assert let.type.name == "Counter"


from quod.model import BinOp  # noqa: E402  (used by the body-Self test)


def test_monomorphize_promotes_impl_methods_to_top_level_fns():
    counter, add_trait, add_impl = _counter_program()
    main_fn = Function(
        name="main",
        return_type=I32Type(),
        body=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),),
    )
    prog = Program(
        structs=(counter,), traits=(add_trait,), impls=(add_impl,),
        functions=(main_fn,),
    )
    out = monomorphize(prog)

    fn_names = {fn.name for fn in out.functions}
    # The impl's method gets promoted with a mangled symbol.
    assert "Counter::add" in fn_names, f"expected Counter::add; got {fn_names}"
    # impls is consumed.
    assert out.impls == ()


def test_monomorphize_resolves_trait_call_to_impl_method():
    counter, add_trait, add_impl = _counter_program()
    main_fn = Function(
        name="main",
        return_type=I32Type(),
        body=(
            Let(
                name="c", type=StructType(name="Counter"),
                init=StructInit(type="Counter", fields=(
                    FieldInit(name="count", value=IntLit(type=I32Type(), value=10)),
                )),
            ),
            ReturnExpr(value=TraitCall(
                trait="Add",
                method="add",
                dispatch_type=StructType(name="Counter"),
                args=(LocalRef(name="c"), IntLit(type=I32Type(), value=32)),
            )),
        ),
    )
    prog = Program(
        structs=(counter,), traits=(add_trait,), impls=(add_impl,),
        functions=(main_fn,),
    )
    out = monomorphize(prog)

    new_main = next(fn for fn in out.functions if fn.name == "main")
    return_stmt = new_main.body[1]
    assert isinstance(return_stmt, ReturnExpr)
    # TraitCall is gone; replaced by a regular Call to the mangled impl method.
    assert isinstance(return_stmt.value, Call)
    assert return_stmt.value.function == "Counter::add"
    assert return_stmt.value.type_args == ()
    # Args pass through unchanged.
    assert len(return_stmt.value.args) == 2


def test_monomorphize_bound_violation_at_instantiation_no_trait_call_in_body():
    """A generic fn with a bound but no trait calls in its body. Without
    bound checking at the instantiation site, mono would happily produce
    the concrete fn (the body doesn't dispatch). With bound checking,
    instantiating with a type that lacks the impl errors clearly."""
    counter, add_trait, add_impl = _counter_program()
    # `fn passthrough<T: Add>(x: T) -> i32 { return 0 }` — bound but body
    # never uses Add::add. The bound is the only constraint that says
    # T must impl Add.
    passthrough = Function(
        name="passthrough",
        type_params=(TypeParam(name="T", bound="Add"),),
        params=(Param(name="x", type=TypeParamRef(name="T")),),
        return_type=I32Type(),
        body=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),),
    )
    # Instantiate with i64, which has no impl Add for i64.
    main_fn = Function(
        name="main",
        return_type=I32Type(),
        body=(
            ReturnExpr(value=Call(
                function="passthrough",
                type_args=(I64Type(),),
                args=(IntLit(type=I64Type(), value=42),),
            )),
        ),
    )
    prog = Program(
        structs=(counter,), traits=(add_trait,), impls=(add_impl,),
        functions=(passthrough, main_fn),
    )
    with pytest.raises(ValueError, match="bound by 'Add'"):
        monomorphize(prog)


def test_monomorphize_bound_satisfied_succeeds():
    """Same shape, but instantiate with a type that DOES impl the trait."""
    counter, add_trait, add_impl = _counter_program()
    passthrough = Function(
        name="passthrough",
        type_params=(TypeParam(name="T", bound="Add"),),
        params=(Param(name="x", type=TypeParamRef(name="T")),),
        return_type=I32Type(),
        body=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),),
    )
    main_fn = Function(
        name="main",
        return_type=I32Type(),
        body=(
            Let(
                name="c", type=StructType(name="Counter"),
                init=StructInit(type="Counter", fields=(
                    FieldInit(name="count", value=IntLit(type=I32Type(), value=10)),
                )),
            ),
            ReturnExpr(value=Call(
                function="passthrough",
                type_args=(StructType(name="Counter"),),
                args=(LocalRef(name="c"),),
            )),
        ),
    )
    prog = Program(
        structs=(counter,), traits=(add_trait,), impls=(add_impl,),
        functions=(passthrough, main_fn),
    )
    out = monomorphize(prog)
    fn_names = {fn.name for fn in out.functions}
    assert "passthrough<Counter>" in fn_names


def test_monomorphize_missing_impl_raises_clear_error():
    counter, add_trait, _add_impl = _counter_program()
    # Note: NO impls registered.
    main_fn = Function(
        name="main",
        return_type=I32Type(),
        body=(
            Let(
                name="c", type=StructType(name="Counter"),
                init=StructInit(type="Counter", fields=(
                    FieldInit(name="count", value=IntLit(type=I32Type(), value=10)),
                )),
            ),
            ReturnExpr(value=TraitCall(
                trait="Add",
                method="add",
                dispatch_type=StructType(name="Counter"),
                args=(LocalRef(name="c"), IntLit(type=I32Type(), value=32)),
            )),
        ),
    )
    prog = Program(structs=(counter,), traits=(add_trait,), functions=(main_fn,))
    with pytest.raises(ValueError, match="no impl of trait"):
        monomorphize(prog)


from quod.model import LocalRef  # noqa: E402  (imported here for the trait tests above)
