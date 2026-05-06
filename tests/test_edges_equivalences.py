"""Step 2 of the C-ingest redesign: stable IDs + Program.edges +
Equivalence claims (with LiftEquivalence/FamilyLowering justifications).

These tests pin the JSON shape, the format_program rendering, and a few
load-bearing invariants the rest of the staged-lift work will depend on:

- Function.id is auto-minted, persists through model_copy, and round-trips
  through JSON.
- ProvenanceEdge and Equivalence round-trip and are dropped from JSON
  when empty so existing programs stay unchanged byte-for-byte.
- format_program prints `edges:` and `equivalences:` sections matching
  the worked-example shape from `.scratch/c-ingest/00-overview.md`.

See the design doc for the broader scope; tests for c-family lowering
rules and predicate domains land in subsequent steps.
"""
from __future__ import annotations

import json

from quod.model import (
    Block,
    Equivalence,
    FamilyLowering,
    Function,
    I32Type,
    IntLit,
    LiftEquivalence,
    Program,
    ProvenanceEdge,
    ReturnExpr,
    format_program,
    load_program,
    save_program,
)


def _trivial_function(*, fn_id: str | None = None, blk_id: str | None = None) -> Function:
    block = Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),))
    if blk_id is not None:
        block = block.model_copy(update={"id": blk_id})
    fn = Function(name="f", return_type=I32Type(), body=block)
    if fn_id is not None:
        fn = fn.model_copy(update={"id": fn_id})
    return fn


def test_function_id_auto_minted():
    fn = Function(name="f", return_type=I32Type(), body=Block())
    assert fn.id.startswith("@fn_")
    assert len(fn.id) > len("@fn_")


def test_function_id_round_trips_through_json():
    fn = _trivial_function(fn_id="@fn_explicit", blk_id="@blk_explicit")
    p = Program(functions=(fn,))
    raw = p.model_dump_json()
    loaded = Program.model_validate_json(raw)
    assert loaded.functions[0].id == "@fn_explicit"
    assert loaded.functions[0].body.id == "@blk_explicit"


def test_function_id_preserved_across_model_copy():
    """Editor mutations go through Function.model_copy(update=...). The
    `id` field is the anchor for edges and equivalence claims; if it
    were re-minted on copy, every editor pass would invalidate every
    pinned edge silently. See the 'Stable IDs are infectious' section
    in .scratch/c-ingest/00-overview.md."""
    fn = _trivial_function(fn_id="@fn_pinned")
    new_block = Block(stmts=())
    new_fn = fn.model_copy(update={"body": new_block})
    assert new_fn.id == "@fn_pinned"


def test_provenance_edge_round_trips():
    edge = ProvenanceEdge(source="@a.sd", target="@b.lets")
    raw = edge.model_dump_json()
    loaded = ProvenanceEdge.model_validate_json(raw)
    assert loaded == edge
    assert json.loads(raw) == {
        "kind": "edge.provenance",
        "source": "@a.sd",
        "target": "@b.lets",
    }


def test_equivalence_round_trips_with_family_lowering():
    eq = Equivalence(
        a_node_id="@blk_b",
        b_node_id="@blk_c",
        regime="witness",
        justification=FamilyLowering(rule_name="c.for_general"),
    )
    raw = eq.model_dump_json()
    loaded = Equivalence.model_validate_json(raw)
    assert loaded == eq
    decoded = json.loads(raw)
    # Default metadata (enforcement=trust, domain=None) is dropped from JSON
    # so equivalence-heavy programs stay compact. `regime=witness` is non-
    # default so it persists.
    assert "enforcement" not in decoded
    assert "domain" not in decoded
    assert decoded["regime"] == "witness"


def test_equivalence_round_trips_with_lift_equivalence():
    eq = Equivalence(
        a_node_id="@a.fn",
        b_node_id="@b",
        regime="witness",
        justification=LiftEquivalence(
            artifact_path="proofs/sum_lift.smt2",
            artifact_hash="deadbeef" * 4,
        ),
    )
    raw = eq.model_dump_json()
    loaded = Equivalence.model_validate_json(raw)
    assert loaded == eq
    j = loaded.justification
    assert isinstance(j, LiftEquivalence)
    assert j.artifact_path == "proofs/sum_lift.smt2"


def test_program_drops_empty_edges_and_equivalences_from_json():
    p = Program(functions=(_trivial_function(),))
    decoded = json.loads(p.model_dump_json())
    assert "edges" not in decoded
    assert "equivalences" not in decoded


def test_program_with_edges_and_equivalences_round_trips(tmp_path):
    fn = _trivial_function(fn_id="@fn_x", blk_id="@blk_x")
    edge = ProvenanceEdge(source="@blk_x", target="@blk_y")
    eq = Equivalence(
        a_node_id="@blk_x",
        b_node_id="@blk_y",
        regime="axiom",
        justification=FamilyLowering(rule_name="c.for_general"),
    )
    p = Program(functions=(fn,), edges=(edge,), equivalences=(eq,))
    path = tmp_path / "program.json"
    save_program(p, path)
    loaded = load_program(path)
    assert loaded.edges == (edge,)
    assert loaded.equivalences == (eq,)


def test_format_program_prints_edges_and_equivalences():
    fn = _trivial_function(fn_id="@fn_x", blk_id="@blk_x")
    p = Program(
        functions=(fn,),
        edges=(ProvenanceEdge(source="@blk_a", target="@blk_b"),),
        equivalences=(
            Equivalence(
                a_node_id="@blk_a",
                b_node_id="@blk_b",
                regime="witness",
                justification=FamilyLowering(rule_name="c.for_general"),
            ),
        ),
    )
    out = format_program(p)
    assert "edges:" in out
    assert "@blk_a -> @blk_b" in out
    assert "equivalences:" in out
    assert "@blk_a ~ @blk_b" in out
    assert "family_lowering(c.for_general)" in out


def test_format_program_omits_sections_when_empty():
    fn = _trivial_function()
    out = format_program(Program(functions=(fn,)))
    assert "edges:" not in out
    assert "equivalences:" not in out


def test_family_lowering_drops_optional_fields_from_json():
    j = FamilyLowering(rule_name="c.for_general")
    decoded = json.loads(j.model_dump_json())
    assert decoded == {"kind": "family_lowering", "rule_name": "c.for_general"}


def test_family_lowering_keeps_pinned_artifact_when_present():
    j = FamilyLowering(
        rule_name="c.for_general",
        artifact_path="proofs/cfor.smt2",
        artifact_hash="abc",
    )
    decoded = json.loads(j.model_dump_json())
    assert decoded["artifact_path"] == "proofs/cfor.smt2"
    assert decoded["artifact_hash"] == "abc"
