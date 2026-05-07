"""Layer-A C source-language nodes, `c.*` family-extension nodes
(CStyleFor, CScopedBlock), and the `Program.source_units` collection.

These tests pin JSON shape, format_program rendering, the smart-union
behavior of `Function.body`, and the `quod.lower` refusal — not codegen.

The fixture program is `sum.c`-shaped:

    int sum(int n) {
      int s = 0;
      for (int i = 0; i < n; i = i + 1) {
        s = s + i;
      }
      return s;
    }
"""
from __future__ import annotations

import json

import pytest

from quod.lower import compile_program
from quod.model import (
    Block,
    CAssign,
    CBinOp,
    CFn,
    CFor,
    CIntLit,
    CParam,
    CReturn,
    CScopedBlock,
    CStyleFor,
    CNamedType,
    CUnit,
    CVarDecl,
    CVarRef,
    Function,
    I32Type,
    IntLit,
    Program,
    ReturnExpr,
    format_program,
    load_program,
    save_program,
)


def _sum_c_unit() -> CUnit:
    int_t = CNamedType(name="int")
    return CUnit(
        source_path="sum.c",
        functions=(
            CFn(
                name="sum",
                return_type=int_t,
                params=(CParam(name="n", type=int_t),),
                body=(
                    CVarDecl(type=int_t, name="s", init=CIntLit(type=I32Type(), value=0)),
                    CFor(
                        init=CVarDecl(type=int_t, name="i", init=CIntLit(type=I32Type(), value=0)),
                        cond=CBinOp(op="<",
                                    lhs=CVarRef(name="i"),
                                    rhs=CVarRef(name="n")),
                        inc=CAssign(target="i",
                                    value=CBinOp(op="+",
                                                 lhs=CVarRef(name="i"),
                                                 rhs=CIntLit(type=I32Type(), value=1))),
                        body=(
                            CAssign(target="s",
                                    value=CBinOp(op="+",
                                                 lhs=CVarRef(name="s"),
                                                 rhs=CVarRef(name="i"))),
                        ),
                    ),
                    CReturn(value=CVarRef(name="s")),
                ),
            ),
        ),
    )


# ----- Layer A: round-trip and format -----


def test_c_unit_round_trips_through_json(tmp_path):
    p = Program(source_units=(_sum_c_unit(),))
    path = tmp_path / "program.json"
    save_program(p, path)
    loaded = load_program(path)
    assert loaded == p


def test_layer_a_node_ids_persist_across_save_load(tmp_path):
    p = Program(source_units=(_sum_c_unit(),))
    path = tmp_path / "program.json"
    save_program(p, path)

    first = load_program(path)
    fn = first.source_units[0].functions[0]
    assert fn.id.startswith("@cfn_")
    decl = fn.body[0]
    assert decl.id.startswith("@cvardecl_")
    for_loop = fn.body[1]
    assert for_loop.id.startswith("@cfor_")
    ret = fn.body[2]
    assert ret.id.startswith("@creturn_")

    # Save what we just loaded; IDs should round-trip identically.
    save_program(first, path)
    second = load_program(path)
    assert first.model_dump_json() == second.model_dump_json()


def test_format_program_renders_source_units():
    p = Program(source_units=(_sum_c_unit(),))
    out = format_program(p)
    assert 'c_unit "sum.c"' in out
    assert "int sum(int n)" in out
    assert "int s = 0;" in out
    assert "for (int i = 0; (i < n); i = (i + 1))" in out
    assert "return s;" in out


def test_program_drops_empty_source_units_from_json():
    p = Program()
    decoded = json.loads(p.model_dump_json())
    assert "source_units" not in decoded


# ----- Layer B: c.* extension nodes -----


def test_c_style_for_round_trips_with_block_body():
    body = Block(stmts=())
    csf = CStyleFor(body=body)
    raw = csf.model_dump_json()
    loaded = CStyleFor.model_validate_json(raw)
    assert loaded.kind == "c.for_general"
    assert loaded.body.id == body.id


def test_c_scoped_block_wraps_block_with_scope_locals():
    inner = Block(id="@blk_for_body", stmts=())
    sb = CScopedBlock(block=inner, scope_locals=("i",))
    raw = sb.model_dump_json()
    loaded = CScopedBlock.model_validate_json(raw)
    assert loaded.block.id == "@blk_for_body"
    assert loaded.scope_locals == ("i",)


def test_c_scoped_block_drops_empty_scope_locals():
    sb = CScopedBlock(block=Block())
    decoded = json.loads(sb.model_dump_json())
    assert "scope_locals" not in decoded


def test_function_body_accepts_block_or_scoped_block():
    """Smart union: Function.body picks `Block` when there's no `kind`,
    `CScopedBlock` when `kind == "c.scoped_block"`. Existing all-core
    programs still validate as Block (the first branch of the union)."""
    plain = Function(name="f", return_type=I32Type(),
                     body=Block(stmts=()))
    assert isinstance(plain.body, Block)

    inner = Block(id="@blk_inner", stmts=())
    wrapped = Function(name="g", return_type=I32Type(),
                       body=CScopedBlock(block=inner, scope_locals=("i",)))
    assert isinstance(wrapped.body, CScopedBlock)
    assert wrapped.body.block.id == "@blk_inner"

    # Round-trip preserves the wrapper.
    raw = wrapped.model_dump_json()
    loaded = Function.model_validate_json(raw)
    assert isinstance(loaded.body, CScopedBlock)


# ----- quod.lower refusal -----


def test_lower_refuses_function_with_scoped_block_body(tmp_path):
    """`quod.lower` operates on layer C only. Surface a clear error pointing
    at the c-family lowering pass when a wrapper appears."""
    fn = Function(
        name="main",
        return_type=I32Type(),
        body=CScopedBlock(
            block=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
        ),
    )
    p = Program(functions=(fn,))
    with pytest.raises(ValueError, match="layer C must be pure core"):
        compile_program(
            p, build_dir=tmp_path, bins=(("main", "main"),),
            profile=0, link=False,
        )


def test_lower_refuses_c_style_for_in_function_body(tmp_path):
    """A `CStyleFor` reaching `quod.lower` means the c-family lowering pass
    didn't run. The error names the missing pass so the fix is obvious."""
    inner = Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),))
    fn = Function(
        name="main",
        return_type=I32Type(),
        body=Block(stmts=(
            CStyleFor(body=inner),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    p = Program(functions=(fn,))
    with pytest.raises(ValueError, match="c.for_general"):
        compile_program(
            p, build_dir=tmp_path, bins=(("main", "main"),),
            profile=0, link=False,
        )
