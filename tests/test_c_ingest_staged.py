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

from quod.ingest.c import ingest_c
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


def test_existing_c_examples_get_lowered_to_pure_core():
    """The existing C ingest corpus uses constructs (string literals,
    printf calls, if/while) that v5's layer-A translator doesn't
    cover yet. The all-or-nothing layer-A fallback skips
    `source_units` for those files, but the c-family lowering pass
    still runs against `structured_functions` and produces a layer-C
    `Program.functions`. For programs that don't use any `c.*`
    extensions, the lowering rule that fires is `identity`."""
    examples = Path(__file__).resolve().parents[1] / "examples/c_ingest"
    p = ingest_c(examples / "loops/loops.c")
    assert p.source_units == ()  # layer-A translator doesn't cover printf/if/while yet
    # Layer-B and layer-C populated:
    assert len(p.structured_functions) == 3
    assert len(p.functions) == 3
    assert {fn.name for fn in p.functions} == {"sum_to", "factorial", "main"}
    # B→C function-level edges + an `identity` FamilyLowering claim
    # per function (no extensions present).
    family_claims = [
        e for e in p.equivalences if e.justification.kind == "family_lowering"
    ]
    assert len(family_claims) == 3
    assert {c.justification.rule_name for c in family_claims} == {"identity"}
