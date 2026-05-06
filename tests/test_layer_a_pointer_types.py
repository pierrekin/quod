"""Layer-A pointer types + pointer arithmetic.

Pins the v6 widening that brings `string_offset.c` into layer A:

  - `CNamedType("int")` ↔ `I32Type` (existing) and `CPointerType(p)`
    ↔ `I8PtrType` (new) under the union `CType`.
  - `CAddressOf(CArraySubscript(b, i))` ↔ `PtrOffset(b', i')` (the
    `&buf[k]` shape).
  - `CBinOp("+", ptr, n)` ↔ `PtrOffset(ptr', n')` (the `p + n`
    shape).
  - String-value tightening: `CStringLit ↔ StringRef` now compares
    the layer-A decoded value against the layer-B `StringConstant`'s
    actual bytes via the program's constants table.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from quod.ingest.c import ingest_c
from quod.lift_check import LiftCheckError, walk_lift
from quod.model import (
    Block,
    CAddressOf,
    CArraySubscript,
    CBinOp,
    CCall,
    CExprStmt,
    CFn,
    CIntLit,
    CNamedType,
    CParam,
    CPointerType,
    CReturn,
    CStringLit,
    CType,
    CUnit,
    CVarDecl,
    CVarRef,
    Function,
    I32Type,
    I8PtrType,
    IntLit,
    Param,
    Program,
    ReturnExpr,
    StringConstant,
)


EXAMPLES = Path(__file__).resolve().parents[1] / "examples/c_ingest"


# ---------- model ----------


def test_cpointer_type_round_trips_through_json():
    t = CPointerType(pointee=CNamedType(name="char"))
    raw = t.model_dump_json()
    loaded = CPointerType.model_validate_json(raw)
    assert isinstance(loaded.pointee, CNamedType)
    assert loaded.pointee.name == "char"


def test_cpointer_type_nests_for_pointer_to_pointer():
    t = CPointerType(pointee=CPointerType(pointee=CNamedType(name="int")))
    assert t.pointee.kind == "c.type.ptr"
    assert t.pointee.pointee.name == "int"


# ---------- ingester ----------


def test_string_offset_emits_layer_a_with_char_pointer_local():
    p = ingest_c(EXAMPLES / "string_offset/string_offset.c")
    assert len(p.source_units) == 1
    fn = p.source_units[0].functions[0]
    decl = fn.body[0]
    assert isinstance(decl, CVarDecl)
    assert decl.name == "greeting"
    # Local type: char*
    assert isinstance(decl.type, CPointerType)
    assert isinstance(decl.type.pointee, CNamedType)
    assert decl.type.pointee.name == "char"
    # Initializer: a string literal
    assert isinstance(decl.init, CStringLit)
    assert decl.init.value == "hello, world!"


def test_string_offset_emits_p_plus_n_via_cbinop():
    p = ingest_c(EXAMPLES / "string_offset/string_offset.c")
    fn = p.source_units[0].functions[0]
    # body: decl, printf full, printf p+7, printf &p[7], printf p+12, return 0
    p_plus_7_stmt = fn.body[2]
    call = p_plus_7_stmt.value
    assert isinstance(call, CCall) and call.callee == "printf"
    second = call.args[1]
    assert isinstance(second, CBinOp) and second.op == "+"
    assert isinstance(second.lhs, CVarRef) and second.lhs.name == "greeting"
    assert isinstance(second.rhs, CIntLit) and second.rhs.value == 7


def test_string_offset_emits_address_of_array_subscript():
    p = ingest_c(EXAMPLES / "string_offset/string_offset.c")
    fn = p.source_units[0].functions[0]
    addr_of_stmt = fn.body[3]  # printf("&p[7]  = %s\n", &greeting[7]);
    call = addr_of_stmt.value
    second = call.args[1]
    assert isinstance(second, CAddressOf)
    sub = second.target
    assert isinstance(sub, CArraySubscript)
    assert isinstance(sub.base, CVarRef) and sub.base.name == "greeting"
    assert isinstance(sub.index, CIntLit) and sub.index.value == 7


# ---------- lift-check ----------


def test_lift_check_passes_on_string_offset():
    p = ingest_c(EXAMPLES / "string_offset/string_offset.c")
    cfn = p.source_units[0].functions[0]
    fn = p.structured_functions[0]
    walk_lift(cfn, fn, program=p)


def test_lift_check_pointer_arith_via_binop():
    """`p + 7` (layer A: CBinOp) ↔ PtrOffset(layer-B local, IntLit(i64, 7))."""
    from quod.model import LocalRef, PtrOffset
    char_p = CPointerType(pointee=CNamedType(name="char"))
    cfn = CFn(
        id="@cfn_p", name="p", return_type=CNamedType(name="int"),
        params=(CParam(name="p", type=char_p),),
        body=(CReturn(value=CIntLit(value=0)),),  # body irrelevant; we hand-build the expr below
    )
    fn = Function(
        id="@fn_p", name="p", return_type=I32Type(),
        params=(Param(name="p", type=I8PtrType()),),
        body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
    )
    walk_lift(cfn, fn, program=Program())  # baseline: no strings

    # Now exercise the pointer-arith correspondence directly via _check_expr.
    from quod.lift_check import _check_expr, _Ctx
    from quod.model import I64Type
    a = CBinOp(op="+", lhs=CVarRef(name="p"), rhs=CIntLit(value=7))
    b = PtrOffset(
        base=LocalRef(name="p"),
        offset=IntLit(type=I64Type(), value=7),
    )
    ctx = _Ctx(constants_by_name={})
    record = _check_expr(a, b, path="<test>", ctx=ctx)
    assert record["kind"] == "p + n ↔ ptr_offset"


def test_lift_check_address_of_array_subscript_via_helper():
    """`&p[k]` (layer A) ↔ PtrOffset (layer B). Same equivalence as
    `p + k` but a different layer-A spelling."""
    from quod.lift_check import _check_expr, _Ctx
    from quod.model import I64Type, LocalRef, PtrOffset
    a = CAddressOf(
        target=CArraySubscript(
            base=CVarRef(name="p"),
            index=CIntLit(value=3),
        ),
    )
    b = PtrOffset(
        base=LocalRef(name="p"),
        offset=IntLit(type=I64Type(), value=3),
    )
    ctx = _Ctx(constants_by_name={})
    record = _check_expr(a, b, path="<test>", ctx=ctx)
    assert record["kind"] == "&p[k] ↔ ptr_offset"


def test_lift_check_pointer_type_against_non_pointer_layer_b_fails():
    char_p = CPointerType(pointee=CNamedType(name="char"))
    int_t = CNamedType(name="int")
    cfn = CFn(
        id="@cfn_b", name="b", return_type=int_t,
        params=(CParam(name="x", type=char_p),),
        body=(CReturn(value=CIntLit(value=0)),),
    )
    fn = Function(
        id="@fn_b", name="b", return_type=I32Type(),
        params=(Param(name="x", type=I32Type()),),  # ← wrong; expected i8*
        body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
    )
    with pytest.raises(LiftCheckError, match="layer-A pointer .* but layer-B is I32Type"):
        walk_lift(cfn, fn)


# ---------- string-value tightening ----------


def test_string_value_check_catches_constant_table_drift():
    """Hand-build a (CFn, Function) pair where layer-A `CStringLit`
    says one thing and the program's `StringConstant` (referenced by
    layer-B `StringRef`) says another. The lift-check must surface
    the divergence."""
    from quod.model import StringRef
    cfn = CFn(
        id="@cfn_s", name="s", return_type=CNamedType(name="int"),
        body=(CExprStmt(value=CCall(
            callee="puts",
            args=(CStringLit(value="hello"),),
        )), CReturn(value=CIntLit(value=0))),
    )
    fn = Function(
        id="@fn_s", name="s", return_type=I32Type(),
        body=Block(stmts=(
            __import__("quod.model", fromlist=["ExprStmt"]).ExprStmt(
                value=__import__("quod.model", fromlist=["Call"]).Call(
                    function="puts",
                    args=(StringRef(name=".str.0"),),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    # Constant table says ".str.0" is "WRONG" — diverges from layer A.
    program = Program(
        functions=(fn,),
        constants=(StringConstant(name=".str.0", value="WRONG"),),
    )
    with pytest.raises(LiftCheckError, match="layer-A string 'hello' vs layer-B"):
        walk_lift(cfn, fn, program=program)


def test_walk_lift_refuses_string_lit_without_program():
    """If a `CStringLit` appears in the walk and `program=None`, the
    string value can't be verified against the layer-B
    `StringConstant`. The checker refuses rather than silently
    passing a kind-only check."""
    from quod.model import StringRef
    cfn = CFn(
        id="@cfn_s", name="s", return_type=CNamedType(name="int"),
        body=(CExprStmt(value=CCall(
            callee="puts",
            args=(CStringLit(value="hello"),),
        )), CReturn(value=CIntLit(value=0))),
    )
    fn = Function(
        id="@fn_s", name="s", return_type=I32Type(),
        body=Block(stmts=(
            __import__("quod.model", fromlist=["ExprStmt"]).ExprStmt(
                value=__import__("quod.model", fromlist=["Call"]).Call(
                    function="puts",
                    args=(StringRef(name=".str.0"),),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    with pytest.raises(LiftCheckError, match="no `program=` was supplied"):
        walk_lift(cfn, fn)  # no program=


# ---------- enum-constant refusal (preserves curl_fetch fallback) ----------


def test_curl_fetch_layer_a_after_enum_const_ref_landed():
    """Once `CEnumConstRef` lands (next commit on top of pointer
    types), curl_fetch.c gets full layer-A coverage too — `CURL*`
    is a pointer type, `CURLOPT_URL` is an enum constant resolved
    to its integer value."""
    p = ingest_c(EXAMPLES / "curl_fetch/curl_fetch.c")
    assert len(p.source_units) == 1
    fn = p.source_units[0].functions[0]
    # The setopt call's second arg is CURLOPT_URL.
    setopt_stmt = fn.body[1]
    call = setopt_stmt.value
    assert call.callee == "curl_easy_setopt"
    second = call.args[1]
    from quod.model import CEnumConstRef
    assert isinstance(second, CEnumConstRef)
    assert second.name == "CURLOPT_URL"
    assert second.value == 10002
