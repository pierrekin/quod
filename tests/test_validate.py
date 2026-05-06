"""Tests for `quod.validate` — the semantic validation pass.

These pin the diagnostic-emitting behavior: each test constructs a
program that's *structurally* well-formed (Pydantic accepts it) but
violates a semantic rule, and asserts the validator surfaces the
expected error code(s).

The `_program(...)` helper hides the boilerplate of building a Program
with sensible defaults. Every test goes through real `validate()` (not a
mock), so changes to the rule set show up here.
"""
from __future__ import annotations

import pytest

from quod.model import (
    Assign,
    Call,
    EnumDef,
    EnumInit,
    EnumPayloadField,
    EnumType,
    EnumVariant,
    ExprStmt,
    FieldInit,
    FieldRead,
    FieldSet,
    For,
    Function,
    I32Type,
    I64Type,
    IntLit,
    Let,
    LocalRef,
    Match,
    MatchArm,
    Param,
    ParamRef,
    Program,
    Return,
    ReturnExpr,
    StructDef,
    StructField,
    StructInit,
    StructType,
    TryExpr,
    VoidType,

    Block,
)
from quod.validate import (
    ASSIGN_UNDECLARED_LOCAL,
    BARE_RETURN_NON_VOID,
    DUPLICATE_BINDING,
    DUPLICATE_FIELD_INIT,
    DUPLICATE_MATCH_ARM,
    EXTRA_FIELD_INIT,
    EXTRA_MATCH_ARM,
    FIELDSET_NON_STRUCT_LOCAL,
    FIELDSET_UNDECLARED_LOCAL,
    FIELD_READ_NON_STRUCT,
    FOR_VAR_CONFLICT,
    LOCAL_DECLARED_TWICE,
    LOCAL_SHADOWS_PARAM,
    MATCH_ARITY,
    MISSING_FIELD_INIT,
    MULTIPLE_WILDCARDS,
    NON_EXHAUSTIVE_MATCH,
    RETURN_EXPR_VOID,
    TRY_INELIGIBLE_ENUM,
    TRY_RETURN_TYPE_MISMATCH,
    UNDECLARED_LOCAL,
    UNDEFINED_FUNCTION,
    UNKNOWN_FIELD,
    UNKNOWN_VARIANT,
    UNRESOLVED_ENUM,
    UNRESOLVED_STRUCT,
    WILDCARD_BINDS,
    ValidationError,
    validate,
    validate_or_raise,
)


# ---------- Helpers ----------


def _maybe_enum() -> EnumDef:
    return EnumDef(
        name="Maybe",
        variants=(
            EnumVariant(name="Some", fields=(EnumPayloadField(name="value", type=I64Type()),)),
            EnumVariant(name="None"),
        ),
    )


def _point_struct() -> StructDef:
    return StructDef(name="Point", fields=(
        StructField(name="x", type=I64Type()),
        StructField(name="y", type=I64Type()),
    ))


def _codes(diags) -> set[str]:
    return {d.code for d in diags}


# ---------- Type reference resolution ----------


def test_unresolved_struct_in_param():
    fn = Function(
        name="f", return_type=I32Type(),
        params=(),
        body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
    )
    # Function returning a non-existent struct via let init won't even
    # compile expression-wise, so we exercise via a let with a missing
    # struct type.
    fn = Function(
        name="g", return_type=I32Type(),
        body=Block(stmts=(
            Let(name="p", type=StructType(name="Nope"),
                init=StructInit(type="Nope", fields=())),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(functions=(fn,))
    diags = validate(prog)
    # "Nope" is referenced from let type AND struct_init.
    assert UNRESOLVED_STRUCT in _codes(diags)


def test_unresolved_enum_in_let_type():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Let(name="m", type=EnumType(name="MissingEnum"),
                init=IntLit(type=I32Type(), value=0)),  # init is wrong-type but typing isn't checked yet
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(functions=(fn,))
    diags = validate(prog)
    assert UNRESOLVED_ENUM in _codes(diags)


# ---------- StructInit shape ----------


def test_struct_init_unknown_struct():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Let(name="p", type=StructType(name="Point"),
                init=StructInit(type="Point", fields=(
                    FieldInit(name="x", value=IntLit(type=I64Type(), value=1)),
                    FieldInit(name="y", value=IntLit(type=I64Type(), value=2)),
                ))),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    # No structs declared; Point is unresolved.
    prog = Program(functions=(fn,))
    diags = validate(prog)
    assert UNRESOLVED_STRUCT in _codes(diags)


def test_struct_init_missing_field():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Let(name="p", type=StructType(name="Point"),
                init=StructInit(type="Point", fields=(
                    FieldInit(name="x", value=IntLit(type=I64Type(), value=1)),
                    # missing 'y'
                ))),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(structs=(_point_struct(),), functions=(fn,))
    diags = validate(prog)
    assert MISSING_FIELD_INIT in _codes(diags)


def test_struct_init_extra_field():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Let(name="p", type=StructType(name="Point"),
                init=StructInit(type="Point", fields=(
                    FieldInit(name="x", value=IntLit(type=I64Type(), value=1)),
                    FieldInit(name="y", value=IntLit(type=I64Type(), value=2)),
                    FieldInit(name="z", value=IntLit(type=I64Type(), value=3)),  # not a field
                ))),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(structs=(_point_struct(),), functions=(fn,))
    diags = validate(prog)
    assert EXTRA_FIELD_INIT in _codes(diags)


def test_struct_init_duplicate_field():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Let(name="p", type=StructType(name="Point"),
                init=StructInit(type="Point", fields=(
                    FieldInit(name="x", value=IntLit(type=I64Type(), value=1)),
                    FieldInit(name="x", value=IntLit(type=I64Type(), value=99)),
                    FieldInit(name="y", value=IntLit(type=I64Type(), value=2)),
                ))),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(structs=(_point_struct(),), functions=(fn,))
    diags = validate(prog)
    assert DUPLICATE_FIELD_INIT in _codes(diags)


# ---------- EnumInit shape ----------


def test_enum_init_unknown_variant():
    fn = Function(
        name="f", return_type=EnumType(name="Maybe"),
        body=Block(stmts=(ReturnExpr(value=EnumInit(
            enum="Maybe", variant="Nope",  # variant doesn't exist
            fields=(),
        )),)),
    )
    prog = Program(enums=(_maybe_enum(),), functions=(fn,))
    diags = validate(prog)
    assert UNKNOWN_VARIANT in _codes(diags)


def test_enum_init_unresolved_enum():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Let(name="m", type=I32Type(),
                init=IntLit(type=I32Type(), value=0)),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    # Override to use an unresolved enum init.
    fn = fn.model_copy(update={
        "body": Block(stmts=(
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
            # ExprStmt-equivalent path doesn't exist; use the return value.
        )),
        "return_type": EnumType(name="Phantom"),
    })
    fn = fn.model_copy(update={
        "body": Block(stmts=(ReturnExpr(value=EnumInit(
            enum="Phantom", variant="Whatever", fields=(),
        )),)),
    })
    prog = Program(enums=(), functions=(fn,))
    diags = validate(prog)
    assert UNRESOLVED_ENUM in _codes(diags)


# ---------- Match exhaustiveness + arms ----------


def test_match_non_exhaustive():
    # Match on Maybe with only Some arm (missing None).
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Match(
                scrutinee=EnumInit(
                    enum="Maybe", variant="Some",
                    fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=1)),),
                ),
                arms=(
                    MatchArm(variant="Some", bindings=("v",),
                             body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=1)),))),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(enums=(_maybe_enum(),), functions=(fn,))
    diags = validate(prog)
    assert NON_EXHAUSTIVE_MATCH in _codes(diags)


def test_match_extra_arm():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Match(
                scrutinee=EnumInit(
                    enum="Maybe", variant="Some",
                    fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=1)),),
                ),
                arms=(
                    MatchArm(variant="Some", bindings=("v",),
                             body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=1)),))),
                    MatchArm(variant="None",
                             body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),))),
                    MatchArm(variant="Nope",  # not a real variant
                             body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=2)),))),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(enums=(_maybe_enum(),), functions=(fn,))
    diags = validate(prog)
    assert EXTRA_MATCH_ARM in _codes(diags)


def test_match_duplicate_arm():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Match(
                scrutinee=EnumInit(
                    enum="Maybe", variant="Some",
                    fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=1)),),
                ),
                arms=(
                    MatchArm(variant="Some", bindings=("v",),
                             body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=1)),))),
                    MatchArm(variant="Some", bindings=("v",),
                             body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=2)),))),
                    MatchArm(variant="None",
                             body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),))),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(enums=(_maybe_enum(),), functions=(fn,))
    diags = validate(prog)
    assert DUPLICATE_MATCH_ARM in _codes(diags)


def test_match_arity_mismatch():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Match(
                scrutinee=EnumInit(
                    enum="Maybe", variant="Some",
                    fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=1)),),
                ),
                arms=(
                    MatchArm(variant="Some", bindings=("v", "extra"),  # Some has 1 field
                             body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=1)),))),
                    MatchArm(variant="None",
                             body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),))),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(enums=(_maybe_enum(),), functions=(fn,))
    diags = validate(prog)
    assert MATCH_ARITY in _codes(diags)


def test_match_wildcard_with_bindings_rejected():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Match(
                scrutinee=EnumInit(
                    enum="Maybe", variant="Some",
                    fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=1)),),
                ),
                arms=(
                    MatchArm(variant="_", bindings=("v",),  # wildcard can't bind
                             body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),))),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(enums=(_maybe_enum(),), functions=(fn,))
    diags = validate(prog)
    assert WILDCARD_BINDS in _codes(diags)


def test_match_multiple_wildcards():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Match(
                scrutinee=EnumInit(
                    enum="Maybe", variant="Some",
                    fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=1)),),
                ),
                arms=(
                    MatchArm(variant="_",
                             body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),))),
                    MatchArm(variant="_",
                             body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=1)),))),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(enums=(_maybe_enum(),), functions=(fn,))
    diags = validate(prog)
    assert MULTIPLE_WILDCARDS in _codes(diags)


def test_match_duplicate_binding():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Match(
                scrutinee=EnumInit(
                    enum="Pair", variant="Both",
                    fields=(
                        FieldInit(name="a", value=IntLit(type=I64Type(), value=1)),
                        FieldInit(name="b", value=IntLit(type=I64Type(), value=2)),
                    ),
                ),
                arms=(
                    MatchArm(variant="Both", bindings=("v", "v"),  # dup name
                             body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),))),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    pair = EnumDef(name="Pair", variants=(
        EnumVariant(name="Both", fields=(
            EnumPayloadField(name="a", type=I64Type()),
            EnumPayloadField(name="b", type=I64Type()),
        )),
    ))
    prog = Program(enums=(pair,), functions=(fn,))
    diags = validate(prog)
    assert DUPLICATE_BINDING in _codes(diags)


# ---------- Validator collects ALL errors, not just the first ----------


def test_collects_multiple_diagnostics():
    """Two unrelated errors in the same function should both surface.
    The whole point of the diagnostic-collecting validator is that you
    don't have to fix-and-rerun for every error."""
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Let(name="p", type=StructType(name="MissingA"),
                init=StructInit(type="MissingB", fields=())),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(functions=(fn,))
    diags = validate(prog)
    assert len(diags) >= 2  # let-type ref + struct_init ref
    assert UNRESOLVED_STRUCT in _codes(diags)


# ---------- validate_or_raise wrapper ----------


def test_validate_or_raise_clean_program():
    prog = Program(functions=(Function(
        name="main", return_type=I32Type(),
        body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
    ),))
    validate_or_raise(prog)  # no exception


def test_validate_or_raise_raises_validation_error():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Let(name="p", type=StructType(name="Nope"),
                init=StructInit(type="Nope", fields=())),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    prog = Program(functions=(fn,))
    with pytest.raises(ValidationError) as exc_info:
        validate_or_raise(prog)
    assert any(d.code == UNRESOLVED_STRUCT for d in exc_info.value.diagnostics)


# ---------- Phase 2: scope checks ----------


def test_undeclared_local():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(ReturnExpr(value=LocalRef(name="ghost")),)),
    )
    diags = validate(Program(functions=(fn,)))
    assert UNDECLARED_LOCAL in _codes(diags)


def test_local_declared_twice():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Let(name="x", type=I64Type(), init=IntLit(type=I64Type(), value=1)),
            Let(name="x", type=I64Type(), init=IntLit(type=I64Type(), value=2)),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    diags = validate(Program(functions=(fn,)))
    assert LOCAL_DECLARED_TWICE in _codes(diags)


def test_local_shadows_param():
    fn = Function(
        name="f", return_type=I32Type(),
        params=(Param(name="x", type=I32Type()),),
        body=Block(stmts=(
            Let(name="x", type=I64Type(), init=IntLit(type=I64Type(), value=1)),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    diags = validate(Program(functions=(fn,)))
    assert LOCAL_SHADOWS_PARAM in _codes(diags)


def test_for_var_conflict():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Let(name="i", type=I64Type(), init=IntLit(type=I64Type(), value=0)),
            For(var="i", type=I64Type(),
                lo=IntLit(type=I64Type(), value=0),
                hi=IntLit(type=I64Type(), value=10),
                body=Block(stmts=())),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    diags = validate(Program(functions=(fn,)))
    assert FOR_VAR_CONFLICT in _codes(diags)


def test_undefined_function_call():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            ExprStmt(value=Call(function="ghost_fn", args=())),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    diags = validate(Program(functions=(fn,)))
    assert UNDEFINED_FUNCTION in _codes(diags)


def test_assign_undeclared_local():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Assign(name="ghost", value=IntLit(type=I64Type(), value=1)),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    diags = validate(Program(functions=(fn,)))
    assert ASSIGN_UNDECLARED_LOCAL in _codes(diags)


def test_fieldset_undeclared_local():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            FieldSet(local="ghost", name="x",
                     value=IntLit(type=I64Type(), value=1)),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    diags = validate(Program(functions=(fn,)))
    assert FIELDSET_UNDECLARED_LOCAL in _codes(diags)


def test_bare_return_non_void():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(Return(),)),
    )
    diags = validate(Program(functions=(fn,)))
    assert BARE_RETURN_NON_VOID in _codes(diags)


def test_return_expr_void_function():
    fn = Function(
        name="f", return_type=VoidType(),
        body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
    )
    diags = validate(Program(functions=(fn,)))
    assert RETURN_EXPR_VOID in _codes(diags)


def test_match_arm_bindings_introduce_local_in_arm_scope():
    """Inside a Some arm, `v` must be referenceable as a local — the
    validator must push the binding before walking the arm body."""
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Match(
                scrutinee=EnumInit(
                    enum="Maybe", variant="Some",
                    fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=1)),),
                ),
                arms=(
                    MatchArm(variant="Some", bindings=("v",),
                             body=Block(stmts=(ReturnExpr(value=LocalRef(name="v")),))),
                    MatchArm(variant="None",
                             body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),))),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    diags = validate(Program(enums=(_maybe_enum(),), functions=(fn,)))
    assert UNDECLARED_LOCAL not in _codes(diags)


# ---------- Phase 2: try-operator checks ----------


def test_try_ineligible_enum():
    """A 3-variant enum is not ?-eligible."""
    bad = EnumDef(name="Bad", variants=(
        EnumVariant(name="A"),
        EnumVariant(name="B"),
        EnumVariant(name="C"),
    ))
    fn = Function(
        name="f", return_type=EnumType(name="Bad"),
        body=Block(stmts=(
            ReturnExpr(value=TryExpr(value=EnumInit(
                enum="Bad", variant="A", fields=(),
            ))),
        )),
    )
    diags = validate(Program(enums=(bad,), functions=(fn,)))
    assert TRY_INELIGIBLE_ENUM in _codes(diags)


def test_try_return_type_mismatch():
    """Function returning Maybe but ? is on Result — should error."""
    result = EnumDef(name="R", variants=(
        EnumVariant(name="Ok", fields=(EnumPayloadField(name="value", type=I64Type()),)),
        EnumVariant(name="Err"),
    ))
    fn = Function(
        name="f", return_type=EnumType(name="Maybe"),
        body=Block(stmts=(
            Let(name="v", type=I64Type(), init=TryExpr(value=EnumInit(
                enum="R", variant="Ok",
                fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=1)),),
            ))),
            ReturnExpr(value=EnumInit(enum="Maybe", variant="None", fields=())),
        )),
    )
    diags = validate(Program(enums=(_maybe_enum(), result), functions=(fn,)))
    assert TRY_RETURN_TYPE_MISMATCH in _codes(diags)


# ---------- Phase 3: type-aware checks ----------


def test_field_read_on_non_struct():
    """Reading .x off an i64 local should be flagged."""
    fn = Function(
        name="f", return_type=I64Type(),
        body=Block(stmts=(
            Let(name="n", type=I64Type(), init=IntLit(type=I64Type(), value=42)),
            ReturnExpr(value=FieldRead(value=LocalRef(name="n"), name="x")),
        )),
    )
    diags = validate(Program(functions=(fn,)))
    assert FIELD_READ_NON_STRUCT in _codes(diags)


def test_field_read_unknown_field_via_inferred_type():
    """LocalRef → struct type inference → field lookup → unknown field error."""
    point = StructDef(name="Point", fields=(
        StructField(name="x", type=I64Type()),
        StructField(name="y", type=I64Type()),
    ))
    fn = Function(
        name="f", return_type=I64Type(),
        body=Block(stmts=(
            Let(name="p", type=StructType(name="Point"),
                init=StructInit(type="Point", fields=(
                    FieldInit(name="x", value=IntLit(type=I64Type(), value=1)),
                    FieldInit(name="y", value=IntLit(type=I64Type(), value=2)),
                ))),
            ReturnExpr(value=FieldRead(value=LocalRef(name="p"), name="z")),
        )),
    )
    diags = validate(Program(structs=(point,), functions=(fn,)))
    assert UNKNOWN_FIELD in _codes(diags)


def test_fieldset_non_struct_local():
    fn = Function(
        name="f", return_type=I32Type(),
        body=Block(stmts=(
            Let(name="n", type=I64Type(), init=IntLit(type=I64Type(), value=0)),
            FieldSet(local="n", name="x", value=IntLit(type=I64Type(), value=1)),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    diags = validate(Program(functions=(fn,)))
    assert FIELDSET_NON_STRUCT_LOCAL in _codes(diags)


def test_param_ref_resolves():
    """ParamRef to a declared param should NOT error."""
    fn = Function(
        name="f", return_type=I32Type(),
        params=(Param(name="x", type=I32Type()),),
        body=Block(stmts=(ReturnExpr(value=ParamRef(name="x")),)),
    )
    diags = validate(Program(functions=(fn,)))
    # No undeclared-anything diagnostics.
    assert not any(d.code in (UNDECLARED_LOCAL,) for d in diags)
