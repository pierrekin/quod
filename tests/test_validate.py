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
    EnumDef,
    EnumInit,
    EnumPayloadField,
    EnumType,
    EnumVariant,
    FieldInit,
    Function,
    I32Type,
    I64Type,
    IntLit,
    Let,
    LocalRef,
    Match,
    MatchArm,
    Program,
    ReturnExpr,
    StructDef,
    StructField,
    StructInit,
    StructType,
)
from quod.validate import (
    DUPLICATE_BINDING,
    DUPLICATE_FIELD_INIT,
    DUPLICATE_MATCH_ARM,
    EXTRA_FIELD_INIT,
    EXTRA_MATCH_ARM,
    MATCH_ARITY,
    MISSING_FIELD_INIT,
    MULTIPLE_WILDCARDS,
    NON_EXHAUSTIVE_MATCH,
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
        body=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),),
    )
    # Function returning a non-existent struct via let init won't even
    # compile expression-wise, so we exercise via a let with a missing
    # struct type.
    fn = Function(
        name="g", return_type=I32Type(),
        body=(
            Let(name="p", type=StructType(name="Nope"),
                init=StructInit(type="Nope", fields=())),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(functions=(fn,))
    diags = validate(prog)
    # "Nope" is referenced from let type AND struct_init.
    assert UNRESOLVED_STRUCT in _codes(diags)


def test_unresolved_enum_in_let_type():
    fn = Function(
        name="f", return_type=I32Type(),
        body=(
            Let(name="m", type=EnumType(name="MissingEnum"),
                init=IntLit(type=I32Type(), value=0)),  # init is wrong-type but typing isn't checked yet
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(functions=(fn,))
    diags = validate(prog)
    assert UNRESOLVED_ENUM in _codes(diags)


# ---------- StructInit shape ----------


def test_struct_init_unknown_struct():
    fn = Function(
        name="f", return_type=I32Type(),
        body=(
            Let(name="p", type=StructType(name="Point"),
                init=StructInit(type="Point", fields=(
                    FieldInit(name="x", value=IntLit(type=I64Type(), value=1)),
                    FieldInit(name="y", value=IntLit(type=I64Type(), value=2)),
                ))),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    # No structs declared; Point is unresolved.
    prog = Program(functions=(fn,))
    diags = validate(prog)
    assert UNRESOLVED_STRUCT in _codes(diags)


def test_struct_init_missing_field():
    fn = Function(
        name="f", return_type=I32Type(),
        body=(
            Let(name="p", type=StructType(name="Point"),
                init=StructInit(type="Point", fields=(
                    FieldInit(name="x", value=IntLit(type=I64Type(), value=1)),
                    # missing 'y'
                ))),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(structs=(_point_struct(),), functions=(fn,))
    diags = validate(prog)
    assert MISSING_FIELD_INIT in _codes(diags)


def test_struct_init_extra_field():
    fn = Function(
        name="f", return_type=I32Type(),
        body=(
            Let(name="p", type=StructType(name="Point"),
                init=StructInit(type="Point", fields=(
                    FieldInit(name="x", value=IntLit(type=I64Type(), value=1)),
                    FieldInit(name="y", value=IntLit(type=I64Type(), value=2)),
                    FieldInit(name="z", value=IntLit(type=I64Type(), value=3)),  # not a field
                ))),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(structs=(_point_struct(),), functions=(fn,))
    diags = validate(prog)
    assert EXTRA_FIELD_INIT in _codes(diags)


def test_struct_init_duplicate_field():
    fn = Function(
        name="f", return_type=I32Type(),
        body=(
            Let(name="p", type=StructType(name="Point"),
                init=StructInit(type="Point", fields=(
                    FieldInit(name="x", value=IntLit(type=I64Type(), value=1)),
                    FieldInit(name="x", value=IntLit(type=I64Type(), value=99)),
                    FieldInit(name="y", value=IntLit(type=I64Type(), value=2)),
                ))),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(structs=(_point_struct(),), functions=(fn,))
    diags = validate(prog)
    assert DUPLICATE_FIELD_INIT in _codes(diags)


# ---------- EnumInit shape ----------


def test_enum_init_unknown_variant():
    fn = Function(
        name="f", return_type=EnumType(name="Maybe"),
        body=(ReturnExpr(value=EnumInit(
            enum="Maybe", variant="Nope",  # variant doesn't exist
            fields=(),
        )),),
    )
    prog = Program(enums=(_maybe_enum(),), functions=(fn,))
    diags = validate(prog)
    assert UNKNOWN_VARIANT in _codes(diags)


def test_enum_init_unresolved_enum():
    fn = Function(
        name="f", return_type=I32Type(),
        body=(
            Let(name="m", type=I32Type(),
                init=IntLit(type=I32Type(), value=0)),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    # Override to use an unresolved enum init.
    fn = fn.model_copy(update={
        "body": (
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
            # ExprStmt-equivalent path doesn't exist; use the return value.
        ),
        "return_type": EnumType(name="Phantom"),
    })
    fn = fn.model_copy(update={
        "body": (ReturnExpr(value=EnumInit(
            enum="Phantom", variant="Whatever", fields=(),
        )),),
    })
    prog = Program(enums=(), functions=(fn,))
    diags = validate(prog)
    assert UNRESOLVED_ENUM in _codes(diags)


# ---------- Match exhaustiveness + arms ----------


def test_match_non_exhaustive():
    # Match on Maybe with only Some arm (missing None).
    fn = Function(
        name="f", return_type=I32Type(),
        body=(
            Match(
                scrutinee=EnumInit(
                    enum="Maybe", variant="Some",
                    fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=1)),),
                ),
                arms=(
                    MatchArm(variant="Some", bindings=("v",),
                             body=(ReturnExpr(value=IntLit(type=I32Type(), value=1)),)),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(enums=(_maybe_enum(),), functions=(fn,))
    diags = validate(prog)
    assert NON_EXHAUSTIVE_MATCH in _codes(diags)


def test_match_extra_arm():
    fn = Function(
        name="f", return_type=I32Type(),
        body=(
            Match(
                scrutinee=EnumInit(
                    enum="Maybe", variant="Some",
                    fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=1)),),
                ),
                arms=(
                    MatchArm(variant="Some", bindings=("v",),
                             body=(ReturnExpr(value=IntLit(type=I32Type(), value=1)),)),
                    MatchArm(variant="None",
                             body=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
                    MatchArm(variant="Nope",  # not a real variant
                             body=(ReturnExpr(value=IntLit(type=I32Type(), value=2)),)),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(enums=(_maybe_enum(),), functions=(fn,))
    diags = validate(prog)
    assert EXTRA_MATCH_ARM in _codes(diags)


def test_match_duplicate_arm():
    fn = Function(
        name="f", return_type=I32Type(),
        body=(
            Match(
                scrutinee=EnumInit(
                    enum="Maybe", variant="Some",
                    fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=1)),),
                ),
                arms=(
                    MatchArm(variant="Some", bindings=("v",),
                             body=(ReturnExpr(value=IntLit(type=I32Type(), value=1)),)),
                    MatchArm(variant="Some", bindings=("v",),
                             body=(ReturnExpr(value=IntLit(type=I32Type(), value=2)),)),
                    MatchArm(variant="None",
                             body=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(enums=(_maybe_enum(),), functions=(fn,))
    diags = validate(prog)
    assert DUPLICATE_MATCH_ARM in _codes(diags)


def test_match_arity_mismatch():
    fn = Function(
        name="f", return_type=I32Type(),
        body=(
            Match(
                scrutinee=EnumInit(
                    enum="Maybe", variant="Some",
                    fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=1)),),
                ),
                arms=(
                    MatchArm(variant="Some", bindings=("v", "extra"),  # Some has 1 field
                             body=(ReturnExpr(value=IntLit(type=I32Type(), value=1)),)),
                    MatchArm(variant="None",
                             body=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(enums=(_maybe_enum(),), functions=(fn,))
    diags = validate(prog)
    assert MATCH_ARITY in _codes(diags)


def test_match_wildcard_with_bindings_rejected():
    fn = Function(
        name="f", return_type=I32Type(),
        body=(
            Match(
                scrutinee=EnumInit(
                    enum="Maybe", variant="Some",
                    fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=1)),),
                ),
                arms=(
                    MatchArm(variant="_", bindings=("v",),  # wildcard can't bind
                             body=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(enums=(_maybe_enum(),), functions=(fn,))
    diags = validate(prog)
    assert WILDCARD_BINDS in _codes(diags)


def test_match_multiple_wildcards():
    fn = Function(
        name="f", return_type=I32Type(),
        body=(
            Match(
                scrutinee=EnumInit(
                    enum="Maybe", variant="Some",
                    fields=(FieldInit(name="value", value=IntLit(type=I64Type(), value=1)),),
                ),
                arms=(
                    MatchArm(variant="_",
                             body=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
                    MatchArm(variant="_",
                             body=(ReturnExpr(value=IntLit(type=I32Type(), value=1)),)),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(enums=(_maybe_enum(),), functions=(fn,))
    diags = validate(prog)
    assert MULTIPLE_WILDCARDS in _codes(diags)


def test_match_duplicate_binding():
    fn = Function(
        name="f", return_type=I32Type(),
        body=(
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
                             body=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
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
        body=(
            Let(name="p", type=StructType(name="MissingA"),
                init=StructInit(type="MissingB", fields=())),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(functions=(fn,))
    diags = validate(prog)
    assert len(diags) >= 2  # let-type ref + struct_init ref
    assert UNRESOLVED_STRUCT in _codes(diags)


# ---------- validate_or_raise wrapper ----------


def test_validate_or_raise_clean_program():
    prog = Program(functions=(Function(
        name="main", return_type=I32Type(),
        body=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),),
    ),))
    validate_or_raise(prog)  # no exception


def test_validate_or_raise_raises_validation_error():
    fn = Function(
        name="f", return_type=I32Type(),
        body=(
            Let(name="p", type=StructType(name="Nope"),
                init=StructInit(type="Nope", fields=())),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        ),
    )
    prog = Program(functions=(fn,))
    with pytest.raises(ValidationError) as exc_info:
        validate_or_raise(prog)
    assert any(d.code == UNRESOLVED_STRUCT for d in exc_info.value.diagnostics)
