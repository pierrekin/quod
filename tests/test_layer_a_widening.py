"""Layer-A widening: CStringLit, CCall, CIf, CWhile, CExprStmt.

Pins the new correspondence rules end-to-end against the existing
C corpus. Each example is ingested, the lift-checker is run on
every (CFn, Function) pair, and key shape assertions are made about
the layer-A subtree.
"""
from __future__ import annotations

from pathlib import Path

from quod.ingest.c import ingest_c
from quod.lift_check import (
    LiftCheckError,
    walk_lift,
)
from quod.model import (
    Assign,
    Block,
    CAssign,
    CBinOp,
    CCall,
    CExprStmt,
    CFn,
    CIf,
    CIntLit,
    CParam,
    CReturn,
    CStringLit,
    CNamedType,
    CVarDecl,
    CVarRef,
    CWhile,
    Function,
    I32Type,
    If,
    IntLit,
    Param,
    ParamRef,
    ReturnExpr,
)


EXAMPLES = Path(__file__).resolve().parents[1] / "examples/c_ingest"


# ---------- corpus coverage ----------


def test_hello_emits_layer_a_call_with_string_literal():
    p = ingest_c(EXAMPLES / "hello/hello.c")
    assert len(p.source_units) == 1
    fn = p.source_units[0].functions[0]
    # body: printf("hello, world\n"); return 0;
    expr_stmt, ret = fn.body
    assert isinstance(expr_stmt, CExprStmt)
    call = expr_stmt.value
    assert isinstance(call, CCall)
    assert call.callee == "printf"
    assert len(call.args) == 1
    assert isinstance(call.args[0], CStringLit)
    assert call.args[0].value == "hello, world\n"


def test_arithmetic_emits_user_calls_with_int_args():
    p = ingest_c(EXAMPLES / "arithmetic/arithmetic.c")
    sq, sum_sq, _main = p.source_units[0].functions
    # sum_squares: return square(a) + square(b);
    ret = sum_sq.body[0]
    assert isinstance(ret, CReturn)
    add = ret.value
    assert isinstance(add, CBinOp) and add.op == "+"
    assert isinstance(add.lhs, CCall) and add.lhs.callee == "square"
    assert isinstance(add.rhs, CCall) and add.rhs.callee == "square"


def test_control_flow_emits_if_else_chains_and_i1_widening():
    """`classify` has nested if/else; `in_unit_range` returns an i1
    expression which the lifter widens to `if (cond) return 1 else
    return 0`. The lift-checker must recognize both shapes."""
    p = ingest_c(EXAMPLES / "control_flow/control_flow.c")
    classify, in_unit_range, _main = p.source_units[0].functions

    # classify: outer if, inner if, no top-level CExprStmt.
    outer_if = classify.body[0]
    assert isinstance(outer_if, CIf)
    assert outer_if.then_body[0].kind == "c.return"
    inner_if = outer_if.else_body[0]
    assert isinstance(inner_if, CIf)

    # in_unit_range: single return with an && comparison expression.
    ret = in_unit_range.body[0]
    assert isinstance(ret, CReturn)
    assert isinstance(ret.value, CBinOp) and ret.value.op == "&&"

    # Lift-check both — the i1-widening case is tested implicitly
    # because in_unit_range's return is i1-typed at layer B.
    fns_by_name = {fn.name: fn for fn in p.structured_functions}
    walk_lift(classify, fns_by_name["classify"])
    walk_lift(in_unit_range, fns_by_name["in_unit_range"])


def test_loops_emits_while_with_assignment_body():
    p = ingest_c(EXAMPLES / "loops/loops.c")
    sum_to = p.source_units[0].functions[0]
    # body: int total=0; int i=1; while (...) { ... }; return total;
    decl_total, decl_i, while_loop, ret = sum_to.body
    assert isinstance(decl_total, CVarDecl) and decl_total.name == "total"
    assert isinstance(decl_i, CVarDecl) and decl_i.name == "i"
    assert isinstance(while_loop, CWhile)
    # while body: total = total + i; i = i + 1;
    body = while_loop.body
    assert len(body) == 2
    assert isinstance(body[0], CAssign) and body[0].target == "total"
    assert isinstance(body[1], CAssign) and body[1].target == "i"


def test_fizzbuzz_combines_while_nested_if_calls_modulo():
    """fizzbuzz exercises every new construct at once: while, nested
    if/else, call-as-expr-stmt, %, string literals."""
    p = ingest_c(EXAMPLES / "fizzbuzz/fizzbuzz.c")
    main = p.source_units[0].functions[0]
    # int i = 1; while (...) { if/else chain; i = i + 1; } return 0;
    decl_i, while_loop, ret = main.body
    assert isinstance(while_loop, CWhile)
    inner = while_loop.body
    # if (...) { printf(...); } else { ...nested... } i = i + 1;
    outer_if, inc = inner
    assert isinstance(outer_if, CIf) and isinstance(inc, CAssign)
    # Each leaf branch ends in a CExprStmt(CCall("printf", ...)).
    leaf = outer_if.then_body[0]
    assert isinstance(leaf, CExprStmt)
    assert isinstance(leaf.value, CCall) and leaf.value.callee == "printf"


# ---------- lift-check covers all corpus examples ----------


def test_lift_check_passes_on_every_layer_a_corpus_example():
    """Walk every (CFn, Function) pair the corpus produces; any
    LiftCheckError fails the test with the offending function name."""
    examples = [
        "hello/hello.c", "arithmetic/arithmetic.c",
        "control_flow/control_flow.c", "fizzbuzz/fizzbuzz.c",
        "loops/loops.c", "sum/sum.c",
    ]
    for src in examples:
        p = ingest_c(EXAMPLES / src)
        assert p.source_units, f"{src}: expected layer-A subtree"
        cfns_by_name = {f.name: f for f in p.source_units[0].functions}
        fns_by_name = {f.name: f for f in p.structured_functions}
        assert cfns_by_name.keys() == fns_by_name.keys(), (
            f"{src}: layer-A and layer-B function name sets differ"
        )
        for name, cfn in cfns_by_name.items():
            try:
                walk_lift(cfn, fns_by_name[name], program=p)
            except LiftCheckError as e:
                raise AssertionError(f"{src}::{name}: {e}") from e


# ---------- hand-built mismatches (i1 widening) ----------


def test_i1_widened_return_correspondence():
    """`return x < 0;` lifts to `if (x < 0) return 1 else return 0`
    at layer B. The lift-checker accepts this shape (a CReturn with
    an i1-typed CBinOp ↔ a layer-B If with the standard widening
    structure)."""
    int_t = CNamedType(name="int")
    cfn = CFn(
        id="@cfn_neg", name="neg", return_type=int_t,
        params=(CParam(name="x", type=int_t),),
        body=(CReturn(value=CBinOp(
            op="<", lhs=CVarRef(name="x"), rhs=CIntLit(type=I32Type(), value=0),
        )),),
    )
    fn = Function(
        id="@fn_neg", name="neg", return_type=I32Type(),
        params=(Param(name="x", type=I32Type()),),
        body=Block(stmts=(If(
            cond=__import__("quod.model", fromlist=["BinOp"]).BinOp(
                op="slt", lhs=ParamRef(name="x"),
                rhs=IntLit(type=I32Type(), value=0),
            ),
            then_body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=1)),)),
            else_body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
        ),)),
    )
    record = walk_lift(cfn, fn)
    # The body record's first stmt cites the i1_widen transform.
    body_stmt = record["fn"]["body"]["stmts"][0]
    assert "if(cond, return 1, return 0)" in body_stmt["kind"]
    assert body_stmt["transform"] == "i1_widen"


# ---------- format_program renders new constructs ----------


def test_format_program_renders_call_and_string_literal():
    from quod.model import format_program
    p = ingest_c(EXAMPLES / "hello/hello.c")
    out = format_program(p)
    assert "c_unit \"hello.c\"" in out
    assert "printf(" in out
    assert "'hello, world\\n'" in out


def test_format_program_renders_if_else_and_while():
    from quod.model import format_program
    p = ingest_c(EXAMPLES / "fizzbuzz/fizzbuzz.c")
    out = format_program(p)
    assert "while ((i <= 15))" in out
    assert "if (((i % 15) == 0))" in out
    assert "} else {" in out
