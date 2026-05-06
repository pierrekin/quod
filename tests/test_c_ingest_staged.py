"""Step 4 of the C-ingest redesign: staged-lift ingest_c.

`ingest_c` now emits both layers when the layer-A subset covers the
file:

  - Layer A under `Program.source_units` (one CUnit per file).
  - Layer B under `Program.functions` (with `CStyleFor` where C used
    a fully-populated `for`).
  - Function-level `ProvenanceEdge` and `Equivalence` claims pairing
    the two.

For files that use C constructs the layer-A translator doesn't yet
cover (string literals, calls, if/while, …), ingest falls back to the
pre-step-4 layer-B-only output. That all-or-nothing fallback keeps the
existing C corpus working while sum.c exercises the full staged-lift
path.

`lower.py` refuses any program containing `CStyleFor` until step 5's
c-family lowering pass strips the family extensions; this test pins
that behavior.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from quod.ingest.c import IngestError, ingest_c
from quod.lower import compile_program
from quod.model import (
    CAssign,
    CBinOp,
    CFor,
    CIntLit,
    CReturn,
    CStyleFor,
    CVarDecl,
    CVarRef,
    Equivalence,
    ProvenanceEdge,
)


SUM_C = Path(__file__).resolve().parents[1] / "examples/c_ingest/sum/sum.c"


def test_ingest_sum_emits_layer_a_subtree():
    p = ingest_c(SUM_C)

    assert len(p.source_units) == 1
    unit = p.source_units[0]
    assert unit.source_path == "sum.c"
    assert len(unit.functions) == 1
    cfn = unit.functions[0]
    assert cfn.id == "@cfn_c_sum"
    assert cfn.name == "sum"

    # Body shape: int s = 0; for (...) { ... } return s;
    decl_s, for_loop, ret = cfn.body
    assert isinstance(decl_s, CVarDecl)
    assert decl_s.name == "s"
    assert isinstance(decl_s.init, CIntLit) and decl_s.init.value == 0

    assert isinstance(for_loop, CFor)
    assert isinstance(for_loop.init, CVarDecl)
    assert for_loop.init.name == "i"
    assert isinstance(for_loop.cond, CBinOp) and for_loop.cond.op == "<"
    assert isinstance(for_loop.inc, CAssign) and for_loop.inc.target == "i"

    assert isinstance(ret, CReturn)
    assert isinstance(ret.value, CVarRef) and ret.value.name == "s"


def test_ingest_sum_emits_structured_layer_b_with_c_style_for():
    """The `for` survives transcription as `CStyleFor` in
    `structured_functions`. The c-family lowering pass (run at the
    end of ingest_c) consumes this to produce the canonical core form
    in `Program.functions`."""
    p = ingest_c(SUM_C)
    assert len(p.structured_functions) == 1
    fn_b = p.structured_functions[0]
    assert fn_b.id == "@fn_c_sum"
    cfor = fn_b.body.stmts[1]
    assert isinstance(cfor, CStyleFor)
    assert cfor.kind == "c.for_general"


def test_ingest_sum_emits_layer_c_function_in_functions():
    """After c-family lowering, `Program.functions` carries the pure-
    core layer-C version. `CStyleFor` becomes `Let + While + Assign`."""
    from quod.model import Let, While
    p = ingest_c(SUM_C)
    assert len(p.functions) == 1
    fn_c = p.functions[0]
    assert fn_c.id == "@fn_c_lowered_sum"
    # let s; let i (hoisted from for-init); while ((i < n)) { ... }; return s
    let_s, let_i, while_loop, ret = fn_c.body.stmts
    assert isinstance(let_s, Let) and let_s.name == "s"
    assert isinstance(let_i, Let) and let_i.name == "i"
    assert isinstance(while_loop, While)
    # The body picked up the increment as the last statement.
    assert while_loop.body.stmts[-1].kind == "quod.assign"
    assert while_loop.body.stmts[-1].name == "i"


def test_ingest_sum_pairs_layers_via_edges_and_equivalences():
    p = ingest_c(SUM_C)
    # A→B function-level edge + B→C function-level edge + per-block
    # B→C edges. Function-level pair plus body-block + lowered for-body
    # block plus the with-inc-appended block (which is a fresh block
    # produced by the rule).
    sources = {e.source for e in p.edges}
    targets = {e.target for e in p.edges}
    assert "@cfn_c_sum" in sources
    assert "@fn_c_sum" in sources         # B→C function-level edge
    assert "@fn_c_lowered_sum" in targets

    # Equivalences: A~B (manual transcription) and B~C (FamilyLowering).
    eq_kinds = {(e.a_node_id, e.b_node_id, e.justification.kind) for e in p.equivalences}
    assert ("@cfn_c_sum", "@fn_c_sum", "manual") in eq_kinds
    assert ("@fn_c_sum", "@fn_c_lowered_sum", "family_lowering") in eq_kinds
    # The FamilyLowering claim cites c.for_general (the rule used).
    family = [e for e in p.equivalences if e.justification.kind == "family_lowering"][0]
    assert family.justification.rule_name == "c.for_general"


def test_ingest_sum_round_trips_through_json(tmp_path):
    from quod.model import load_program, save_program
    p = ingest_c(SUM_C)
    out = tmp_path / "program.json"
    save_program(p, out)
    loaded = load_program(out)
    assert loaded == p


def test_sum_c_compiles_and_runs(tmp_path):
    """Sum.c reaches a binary via the full staged pipeline:
    ingest → A+B+C → lower.py → LLVM IR → object → linked binary.
    The "smallest end-to-end slice" from .scratch/c-ingest is now
    end-to-end — sum(N) returns the sum 0..N-1 as the exit code."""
    import subprocess
    p = ingest_c(SUM_C)
    result = compile_program(
        p, build_dir=tmp_path, bins=(("sum", "sum"),),
        profile=2, link=True,
    )
    binary = result.bins[0].binary
    assert binary is not None

    for n, expected in [(0, 0), (1, 0), (5, 10), (10, 45), (20, 190)]:
        out = subprocess.run([str(binary), str(n)], capture_output=True, text=True, timeout=10)
        assert out.returncode == expected, f"sum({n}): got {out.returncode}, expected {expected}"


def test_existing_c_examples_get_full_three_layer_lift():
    """All examples in the corpus now produce all three layers — layer A
    under `source_units`, layer B under `structured_functions`,
    layer C under `functions`. The B→C rule cited is `identity`
    when no `c.*` extensions are present (i.e. no for-loops)."""
    examples = Path(__file__).resolve().parents[1] / "examples/c_ingest"
    p = ingest_c(examples / "loops/loops.c")

    # Layer A populated.
    assert len(p.source_units) == 1
    assert p.source_units[0].source_path == "loops.c"
    assert {fn.name for fn in p.source_units[0].functions} == {
        "sum_to", "factorial", "main",
    }

    # Layer B / C populated as before.
    assert len(p.structured_functions) == 3
    assert len(p.functions) == 3
    assert {fn.name for fn in p.functions} == {"sum_to", "factorial", "main"}

    # B→C `identity` (no extensions); A→B manual (CLI ingest path
    # would upgrade to LiftEquivalence; pure-Python ingest_c stops at
    # ManualJustification).
    family_claims = [
        e for e in p.equivalences if e.justification.kind == "family_lowering"
    ]
    assert {c.justification.rule_name for c in family_claims} == {"identity"}
    a_to_b = [e for e in p.equivalences if e.justification.kind == "manual"]
    assert len(a_to_b) == 3


BITWISE_C = Path(__file__).resolve().parents[1] / "examples/c_ingest/bitwise/bitwise.c"


def test_bitwise_example_lifts_three_layers():
    """`<< >> ^ ~ !` round-trip through the staged lift. `~` and `!`
    are layer-A `CUnary` nodes (preserving source) that pair with the
    standard layer-B BinOp identities (`xor x, -1` and `eq x, 0`)."""
    from quod.model import CUnary
    p = ingest_c(BITWISE_C)

    names = {fn.name for fn in p.functions}
    assert names == {"low_bits", "swap_nibbles", "xor_round_trip", "complement", "is_zero", "main"}

    # Every example function got a layer-A subtree.
    assert len(p.source_units) == 1
    cfn_names = {cfn.name for cfn in p.source_units[0].functions}
    assert cfn_names == names

    # `~x` at layer A is preserved as CUnary("~"); the lift's layer-B
    # form is BinOp("xor", x, IntLit(-1)).
    [cfn_complement] = [cfn for cfn in p.source_units[0].functions if cfn.name == "complement"]
    cret = cfn_complement.body[0]
    assert isinstance(cret.value, CUnary) and cret.value.op == "~"

    # `!x` at layer A is CUnary("!"); paired with BinOp("eq", x, 0)
    # under the i1-widening shape.
    [cfn_is_zero] = [cfn for cfn in p.source_units[0].functions if cfn.name == "is_zero"]
    cret = cfn_is_zero.body[0]
    assert isinstance(cret.value, CUnary) and cret.value.op == "!"


def test_bitwise_example_compiles_and_runs(tmp_path):
    import subprocess
    from quod.lower import compile_program
    p = ingest_c(BITWISE_C)
    res = compile_program(
        p, build_dir=tmp_path, bins=(("bitwise", "main"),),
        profile=2, link=True,
    )
    binary = res.bins[0].binary
    assert binary is not None
    out = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert out.returncode == 0
    expected = (
        "low_bits(0xFF, 4)     = 15\n"
        "swap_nibbles(0x12)    = 33\n"
        "xor_round_trip(7, 13) = 7\n"
        "complement(5)         = -6\n"
        "is_zero(0)            = 1\n"
        "is_zero(42)           = 0\n"
    )
    assert out.stdout == expected


def test_bitwise_example_passes_lift_check():
    """The CUnary↔BinOp pairings round-trip through walk_lift cleanly
    for every function in the bitwise example."""
    from quod.lift_check import walk_lift
    p = ingest_c(BITWISE_C)
    cfns_by_name = {cfn.name: cfn for cfn in p.source_units[0].functions}
    fns_by_name = {fn.name: fn for fn in p.structured_functions}
    for name, cfn in cfns_by_name.items():
        walk_lift(cfn, fns_by_name[name], program=p)


VOID_FNS_C = Path(__file__).resolve().parents[1] / "examples/c_ingest/void_fns/void_fns.c"


def test_ingest_void_returning_function():
    """`void f(...)` ingests as a Function with VoidType return; bare
    `return;` emits a Return() statement; void externs now carry a
    real VoidType (not the previous I32 stand-in)."""
    from quod.model import Return as ReturnNode, VoidType
    p = ingest_c(VOID_FNS_C)

    fns_by_name = {fn.name: fn for fn in p.functions}
    assert isinstance(fns_by_name["greet"].return_type, VoidType)
    assert isinstance(fns_by_name["no_explicit_return"].return_type, VoidType)
    # Bare `return;` lifted to Return().
    greet = fns_by_name["greet"]
    iff = greet.body.stmts[0]  # if (n <= 0) { return; }
    assert isinstance(iff.then_body.stmts[0], ReturnNode)
    # Falling off the end of a void body synthesizes Return() at the tail.
    no_ret = fns_by_name["no_explicit_return"]
    assert isinstance(no_ret.body.stmts[-1], ReturnNode)

    # Void externs (e.g. printf is i32-returning, but if exit() were
    # called we'd see VoidType). Here printf is the only extern.
    [printf] = [e for e in p.externs if e.name == "printf"]
    assert printf.return_type.kind == "llvm.i32"


def test_void_fns_example_compiles_and_runs(tmp_path):
    import subprocess
    from quod.lower import compile_program
    p = ingest_c(VOID_FNS_C)
    res = compile_program(
        p, build_dir=tmp_path, bins=(("void_fns", "main"),),
        profile=2, link=True,
    )
    out = subprocess.run([str(res.bins[0].binary)], capture_output=True, text=True, timeout=10)
    assert out.returncode == 0
    assert out.stdout == "hello 1\nhello 2\ncount = 42\n"


def test_bare_return_in_non_void_function_refuses(tmp_path):
    """Bare `return;` from a non-void function is rejected — clang's
    parse step catches it before the ingester walks the AST, but the
    ingester also has its own defensive check for the same case (so
    AST shapes that bypass the parse error still refuse cleanly)."""
    src = tmp_path / "bad.c"
    src.write_text("int f(int x) { if (x < 0) { return; } return x; }\n")
    with pytest.raises(IngestError, match="non-void function|bare `return;`"):
        ingest_c(src)


def test_void_extern_signature_uses_voidtype(tmp_path):
    """Calls to void-returning externs (e.g. libc `exit`) flow through
    the ingester with a real VoidType in the extern signature, and the
    layer-B program lowers and runs cleanly."""
    import subprocess
    from quod.lower import compile_program
    src = tmp_path / "void_extern.c"
    src.write_text(
        "#include <stdlib.h>\n"
        "int main(void) { exit(0); return 1; }\n"
    )
    p = ingest_c(src)
    [exit_ext] = [e for e in p.externs if e.name == "exit"]
    assert exit_ext.return_type.kind == "llvm.void"
    res = compile_program(
        p, build_dir=tmp_path, bins=(("vox", "main"),),
        profile=2, link=True,
    )
    out = subprocess.run([str(res.bins[0].binary)], capture_output=True, text=True, timeout=10)
    assert out.returncode == 0


MULTI_DECL_C = Path(__file__).resolve().parents[1] / "examples/c_ingest/multi_decl/multi_decl.c"


def test_multi_decl_layer_a_preserves_grouping_and_layer_b_expands():
    """`int a = x, b = y, c = z;` lifts to a single layer-A
    CMultiVarDecl wrapping three CVarDecls, and to three consecutive
    Lets at layer B. The lift-checker pairs them 1:3."""
    from quod.model import CMultiVarDecl, Let
    p = ingest_c(MULTI_DECL_C)

    sum3_a = next(cf for cf in p.source_units[0].functions if cf.name == "sum3")
    first = sum3_a.body[0]
    assert isinstance(first, CMultiVarDecl)
    assert tuple(d.name for d in first.decls) == ("a", "b", "c")

    sum3_b = next(fn for fn in p.structured_functions if fn.name == "sum3")
    body = sum3_b.body.stmts
    assert isinstance(body[0], Let) and body[0].name == "a"
    assert isinstance(body[1], Let) and body[1].name == "b"
    assert isinstance(body[2], Let) and body[2].name == "c"


def test_multi_decl_lift_check_pairs_n_lets():
    """walk_lift records the multi-decl 1:N pairing in its output."""
    from quod.lift_check import walk_lift
    p = ingest_c(MULTI_DECL_C)
    sum3_a = next(cf for cf in p.source_units[0].functions if cf.name == "sum3")
    sum3_b = next(fn for fn in p.structured_functions if fn.name == "sum3")
    rec = walk_lift(sum3_a, sum3_b, program=p)
    body_stmts = rec["fn"]["body"]["stmts"]
    multi = body_stmts[0]
    assert multi["kind"] == "c.multi_var_decl ↔ N×let"
    assert len(multi["decls"]) == 3


def test_multi_decl_example_compiles_and_runs(tmp_path):
    import subprocess
    from quod.lower import compile_program
    p = ingest_c(MULTI_DECL_C)
    res = compile_program(
        p, build_dir=tmp_path, bins=(("multi_decl", "main"),),
        profile=2, link=True,
    )
    out = subprocess.run([str(res.bins[0].binary)], capture_output=True, text=True, timeout=10)
    assert out.returncode == 0
    assert "sum3(1, 2, 3)        = 6" in out.stdout
    assert "linear_combo(10)     = 21" in out.stdout


def test_multi_decl_in_for_init_refuses(tmp_path):
    """Multi-declarator inside a for-loop init slot (`for (int a, b; ...)`)
    isn't supported — the init slot wants a single statement."""
    src = tmp_path / "bad_for.c"
    src.write_text(
        "int f(int n) { int s = 0; for (int i = 0, j = 0; i < n; i = i + 1) { s = s + i + j; } return s; }\n"
    )
    with pytest.raises(IngestError, match="single statement"):
        ingest_c(src)


COMPOUND_ASSIGN_C = Path(__file__).resolve().parents[1] / "examples/c_ingest/compound_assign/compound_assign.c"


def test_compound_assign_layer_a_preserves_op():
    """`x += y` lifts to a layer-A CCompoundAssign('+=') preserving
    source form, and desugars to Assign(x, BinOp("add", LocalRef(x),
    y')) on the layer-B side."""
    from quod.model import CCompoundAssign, Assign, BinOp, LocalRef
    p = ingest_c(COMPOUND_ASSIGN_C)

    # Layer A: while-body has two compound assigns (total += i; i += 1;)
    sum_to_a = next(cf for cf in p.source_units[0].functions if cf.name == "sum_to")
    while_stmt = sum_to_a.body[2]
    body_stmts = while_stmt.body
    assert isinstance(body_stmts[0], CCompoundAssign) and body_stmts[0].op == "+="
    assert body_stmts[0].target == "total"
    assert isinstance(body_stmts[1], CCompoundAssign) and body_stmts[1].op == "+="

    # Layer B: same positions are Assign(name, BinOp("add", LocalRef(name), ...))
    sum_to_b = next(fn for fn in p.structured_functions if fn.name == "sum_to")
    while_b = sum_to_b.body.stmts[2]
    body_b = while_b.body.stmts
    assign = body_b[0]
    assert isinstance(assign, Assign) and assign.name == "total"
    assert isinstance(assign.value, BinOp) and assign.value.op == "add"
    assert isinstance(assign.value.lhs, LocalRef) and assign.value.lhs.name == "total"


def test_compound_assign_lift_check_pairs_op():
    """walk_lift records the compound-assign correspondence with the
    underlying BinOp.op."""
    from quod.lift_check import walk_lift
    p = ingest_c(COMPOUND_ASSIGN_C)
    cfn = next(cf for cf in p.source_units[0].functions if cf.name == "reduce")
    fn = next(f for f in p.structured_functions if f.name == "reduce")
    rec = walk_lift(cfn, fn, program=p)
    body_stmts = rec["fn"]["body"]["stmts"]
    # `int x = n;` then x *=, -=, /=, %= in order.
    ops = [
        s["kind"] for s in body_stmts
        if s["kind"].startswith("compound_assign")
    ]
    assert ops == [
        "compound_assign(*=) ↔ assign(_, binop(mul))",
        "compound_assign(-=) ↔ assign(_, binop(sub))",
        "compound_assign(/=) ↔ assign(_, binop(sdiv))",
        "compound_assign(%=) ↔ assign(_, binop(srem))",
    ]


def test_compound_assign_example_compiles_and_runs(tmp_path):
    import subprocess
    from quod.lower import compile_program
    p = ingest_c(COMPOUND_ASSIGN_C)
    res = compile_program(
        p, build_dir=tmp_path, bins=(("compound", "main"),),
        profile=2, link=True,
    )
    out = subprocess.run([str(res.bins[0].binary)], capture_output=True, text=True, timeout=10)
    assert out.returncode == 0
    assert "sum_to(10)         = 55" in out.stdout
    assert "reduce(50)         = 73" in out.stdout
    assert "bit_ops(0x1234, 2) = 98" in out.stdout


def test_compound_assign_to_parameter_refuses(tmp_path):
    """Compound assignment to a parameter is refused, matching the
    existing rule for plain Assign."""
    src = tmp_path / "bad_param.c"
    src.write_text("int f(int x) { x += 1; return x; }\n")
    with pytest.raises(IngestError, match="cannot assign to 'x'"):
        ingest_c(src)


TERNARY_C = Path(__file__).resolve().parents[1] / "examples/c_ingest/ternary/ternary.c"


def test_ternary_lifts_to_ifexpr_at_layer_b():
    """`cond ? a : b` lifts to layer-A CTernary (preserving source) and
    layer-B IfExpr (a core node). The lift-checker pairs them 1:1."""
    from quod.model import CTernary, IfExpr
    p = ingest_c(TERNARY_C)

    abs_a = next(cf for cf in p.source_units[0].functions if cf.name == "abs_val")
    ret_a = abs_a.body[0]
    assert isinstance(ret_a.value, CTernary)

    abs_b = next(fn for fn in p.structured_functions if fn.name == "abs_val")
    ret_b = abs_b.body.stmts[0]
    assert isinstance(ret_b.value, IfExpr)


def test_ternary_with_integer_cond_widens_to_ne_zero():
    """A ternary whose `cond` is an integer expression (not a comparison)
    gets the C "nonzero ⇒ true" widening — the lift wraps it as
    `BinOp("ne", cond, IntLit(0))` on the layer-B side."""
    from quod.model import IfExpr, BinOp
    p = ingest_c(TERNARY_C)
    sign_b = next(fn for fn in p.structured_functions if fn.name == "sign_or_zero")
    ret_b = sign_b.body.stmts[0]
    outer = ret_b.value
    assert isinstance(outer, IfExpr)
    # Outer cond was just `x` in source — layer B wraps it as `x != 0`.
    assert isinstance(outer.cond, BinOp) and outer.cond.op == "ne"


def test_ternary_example_compiles_and_runs(tmp_path):
    import subprocess
    from quod.lower import compile_program
    p = ingest_c(TERNARY_C)
    res = compile_program(
        p, build_dir=tmp_path, bins=(("ternary", "main"),),
        profile=2, link=True,
    )
    out = subprocess.run([str(res.bins[0].binary)], capture_output=True, text=True, timeout=10)
    assert out.returncode == 0
    assert "abs_val(-7)         = 7" in out.stdout
    assert "max3(2, 9, 4)       = 9" in out.stdout
    assert "sign_or_zero(-3)    = -1" in out.stdout
    assert "sign_or_zero(42)    = 1" in out.stdout


def test_every_c_corpus_example_emits_layer_a():
    """Coverage sweep: every example now produces a `source_units`
    entry. The layer-A widening landed in three steps (calls /
    strings / if / while; pointer types; enum constants); this test
    pins the now-uniform invariant."""
    examples_dir = Path(__file__).resolve().parents[1] / "examples/c_ingest"
    for example_dir in sorted(examples_dir.iterdir()):
        c_files = list(example_dir.glob("*.c"))
        if not c_files:
            continue
        p = ingest_c(c_files[0])
        assert p.source_units, f"{example_dir.name}: expected layer-A subtree"
