"""Step 5 of the C-ingest redesign: the c-family lowering pass.

Pins the rules independently of `ingest_c` so the lowering surface has
its own coverage:

  - `c.for_general` rewrites `for (init; cond; inc) body` to
    `init; while (cond) { body; inc }` with init hoisted into the
    enclosing block.
  - `c.scoped_block` strips wrappers and surfaces the inner core
    `Block`; the rule fires (recorded in the equivalence claim) but at
    v5 it's structurally a no-op.
  - Functions that already contain only core constructs go through
    unchanged structurally; the equivalence claim cites `identity` so
    the data model is uniform.
  - Block-level `ProvenanceEdge`s pair every layer-B block with its
    layer-C counterpart.
  - The pass is idempotent on layer-C input (empty
    `structured_functions` ⇒ Program returned unchanged).

Each test constructs its inputs directly to avoid coupling to the C
ingester — `c_family.lower` operates on Programs and is testable
without libclang.
"""
from __future__ import annotations

from quod.lower.c_family import lower_c_family
from quod.model import (
    Assign,
    BinOp,
    Block,
    CScopedBlock,
    CStyleFor,
    Equivalence,
    FamilyLowering,
    Function,
    I32Type,
    IntLit,
    Let,
    LocalRef,
    ParamRef,
    Param,
    Program,
    ProvenanceEdge,
    ReturnExpr,
    While,
)


def _sum_b_function() -> Function:
    """sum.c at layer B (matches what the C ingester emits for sum.c)."""
    body = Block(
        id="@blk_sum_body",
        stmts=(
            Let(name="s", type=I32Type(), init=IntLit(type=I32Type(), value=0)),
            CStyleFor(
                init=Let(name="i", type=I32Type(), init=IntLit(type=I32Type(), value=0)),
                cond=BinOp(op="slt", lhs=LocalRef(name="i"), rhs=ParamRef(name="n")),
                inc=Assign(
                    name="i",
                    value=BinOp(op="add", lhs=LocalRef(name="i"), rhs=IntLit(type=I32Type(), value=1)),
                ),
                body=Block(
                    id="@blk_sum_for_body",
                    stmts=(Assign(
                        name="s",
                        value=BinOp(op="add", lhs=LocalRef(name="s"), rhs=LocalRef(name="i")),
                    ),),
                ),
            ),
            ReturnExpr(value=LocalRef(name="s")),
        ),
    )
    return Function(
        id="@fn_b_sum", name="sum",
        params=(Param(name="n", type=I32Type()),),
        return_type=I32Type(),
        body=body,
    )


def test_lower_c_family_no_op_when_structured_empty():
    p = Program()
    assert lower_c_family(p) == p


def test_for_general_rule_rewrites_to_while():
    p = Program(structured_functions=(_sum_b_function(),))
    out = lower_c_family(p)

    assert len(out.functions) == 1
    fn_c = out.functions[0]
    # init hoisted; CStyleFor replaced by While.
    let_s, let_i, while_loop, ret = fn_c.body.stmts
    assert isinstance(let_s, Let) and let_s.name == "s"
    assert isinstance(let_i, Let) and let_i.name == "i"
    assert isinstance(while_loop, While)
    # Cond preserved.
    assert isinstance(while_loop.cond, BinOp) and while_loop.cond.op == "slt"
    # Body has the original assignment plus the increment appended.
    body_stmts = while_loop.body.stmts
    assert len(body_stmts) == 2
    assert isinstance(body_stmts[0], Assign) and body_stmts[0].name == "s"
    assert isinstance(body_stmts[1], Assign) and body_stmts[1].name == "i"
    assert isinstance(ret, ReturnExpr)


def test_for_general_rule_emits_block_edges_and_equivalence():
    p = Program(structured_functions=(_sum_b_function(),))
    out = lower_c_family(p)

    fn_c = out.functions[0]
    # Function-level edge.
    assert ProvenanceEdge(source="@fn_b_sum", target=fn_c.id) in out.edges
    # Block-level edge from the layer-B function-body block to the
    # layer-C function-body block.
    targets_for_body = {
        e.target for e in out.edges if e.source == "@blk_sum_body"
    }
    assert fn_c.body.id in targets_for_body
    # Block-level edge from the layer-B for-body block to the layer-C
    # equivalent (which is the inner block of the produced While).
    while_loop = fn_c.body.stmts[2]
    targets_for_loop = {
        e.target for e in out.edges if e.source == "@blk_sum_for_body"
    }
    # The for-body's lowered output is then wrapped with `inc` appended,
    # so we get an additional pairing in the chain.
    assert len(targets_for_loop) == 1

    # Function-level equivalence cites the rule that fired.
    family = [
        e for e in out.equivalences
        if e.a_node_id == "@fn_b_sum" and e.justification.kind == "family_lowering"
    ]
    assert len(family) == 1
    assert family[0].justification.rule_name == "c.for_general"


def test_pure_core_function_lowers_to_identity():
    """A function with no `c.*` extensions still lowers (block IDs
    refresh, edges recorded), with an `identity` rule cited so the
    equivalence claim is uniform across C-derived programs."""
    body = Block(
        id="@blk_pure",
        stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=42)),),
    )
    fn = Function(id="@fn_b_pure", name="pure", return_type=I32Type(), body=body)
    out = lower_c_family(Program(structured_functions=(fn,)))

    fn_c = out.functions[0]
    # Same shape, fresh IDs.
    assert fn_c.id != fn.id
    assert fn_c.body.id != fn.body.id
    assert fn_c.body.stmts == body.stmts

    family = [
        e for e in out.equivalences
        if e.justification.kind == "family_lowering"
    ]
    assert len(family) == 1
    assert family[0].justification.rule_name == "identity"


def test_scoped_block_wrapper_is_stripped():
    """`CScopedBlock` wraps a core Block; the lowering pass surfaces
    the inner block and records `c.scoped_block` in the rules-used
    set so the equivalence claim cites it."""
    inner = Block(
        id="@blk_inner",
        stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),),
    )
    fn = Function(
        id="@fn_b_wrapped", name="wrapped", return_type=I32Type(),
        body=CScopedBlock(block=inner, scope_locals=("x",)),
    )
    out = lower_c_family(Program(structured_functions=(fn,)))

    fn_c = out.functions[0]
    # The wrapper is gone; the inner stmts surface directly.
    assert isinstance(fn_c.body, Block)
    assert fn_c.body.stmts == inner.stmts

    family = [
        e for e in out.equivalences
        if e.justification.kind == "family_lowering"
    ]
    rules = {e.justification.rule_name for e in family}
    assert "c.scoped_block" in rules


def test_lower_is_deterministic_and_idempotent_on_layer_c():
    """Running on identical input twice produces identical output —
    block-ID minting uses a per-function counter, not uuid."""
    p = Program(structured_functions=(_sum_b_function(),))
    out1 = lower_c_family(p)
    out2 = lower_c_family(p)
    assert out1.model_dump_json() == out2.model_dump_json()

    # Pass-through on a layer-C-only program.
    fn = Function(
        id="@fn_pure", name="pure", return_type=I32Type(),
        body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
    )
    pure_program = Program(functions=(fn,))
    assert lower_c_family(pure_program) == pure_program


def test_for_general_with_no_cond_lowers_to_while_true():
    """Sparse for-loops with `cond` absent (`for (init;;inc) body`)
    lower to `init; while (true) { body; inc; }`. The constant-true
    cond is an i1-typed `IntLit(1)` — directly lowerable to LLVM."""
    from quod.model import I1Type

    body = Block(
        id="@blk_x",
        stmts=(CStyleFor(
            init=Let(name="i", type=I32Type(), init=IntLit(type=I32Type(), value=0)),
            cond=None,
            inc=Assign(name="i", value=IntLit(type=I32Type(), value=1)),
            body=Block(stmts=()),
        ),),
    )
    fn = Function(id="@fn_b_x", name="x", return_type=I32Type(), body=body)
    out = lower_c_family(Program(structured_functions=(fn,)))

    fn_c = out.functions[0]
    let_i, while_loop = fn_c.body.stmts
    assert isinstance(let_i, Let) and let_i.name == "i"
    assert isinstance(while_loop, While)
    # Cond is `IntLit(I1Type, 1)` — i.e., `true`.
    assert isinstance(while_loop.cond, IntLit)
    assert isinstance(while_loop.cond.type, I1Type)
    assert while_loop.cond.value == 1
