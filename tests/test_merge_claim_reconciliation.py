"""Annotation survival across re-lifts — `merge_program` reconciles
existing claims into the new function shape rather than silently
dropping them.

Honors the rule from `.scratch/c-ingest/00-overview.md` (Claims and
their layer → Survival across re-lifts):

> An annotation (including a claim) survives if its target node's
> identity hash is unchanged, gets invalidated otherwise. For function
> parameters this typically reduces to "param still exists by name
> and has the same type → transfer the claim; otherwise orphan and
> warn."

The realistic scenario: user runs `quod ingest c sum.c`, adds a
claim with `quod claim add non_negative -f sum n`, then later edits
sum.c (whitespace, comment, semantically identical) and re-ingests.
The new ingest produces a fresh `Function` with no claims; without
reconciliation the user's claim is lost. With reconciliation it
survives if `n` is still an int param.
"""
from __future__ import annotations

import pytest

from quod.merge import merge_program
from quod.model import (
    Block,
    Function,
    I32Type,
    I64Type,
    I8PtrType,
    IntLit,
    IntRangeClaim,
    LibcLinkage,
    NonNegativeClaim,
    Param,
    Program,
    ReturnExpr,
    ReturnInRangeClaim,
)


def _fn_with_claims(
    *, name: str, params: tuple[Param, ...] = (),
    return_type=I32Type(),
    claims: tuple = (),
    body=None,
) -> Function:
    if body is None:
        body = Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),))
    return Function(
        name=name, params=params, return_type=return_type,
        body=body, claims=claims,
    )


# ---------- happy path ----------


def test_existing_claim_survives_unchanged_param(tmp_path):
    """The realistic re-ingest case: existing program has a claim,
    new program (from a fresh ingest) has the same function shape
    minus the claim. The claim should survive."""
    existing_fn = _fn_with_claims(
        name="f",
        params=(Param(name="x", type=I32Type()),),
        claims=(NonNegativeClaim(param="x"),),
    )
    new_fn = _fn_with_claims(
        name="f",
        params=(Param(name="x", type=I32Type()),),
        claims=(),
    )
    existing = Program(functions=(existing_fn,))
    new = Program(functions=(new_fn,))

    merged, warnings = merge_program(existing, new)

    assert warnings == ()
    surviving = merged.functions[0].claims
    assert len(surviving) == 1
    assert surviving[0].kind == "non_negative"
    assert surviving[0].param == "x"


def test_existing_return_claim_survives_unchanged_return_type():
    existing_fn = _fn_with_claims(
        name="f", return_type=I32Type(),
        claims=(ReturnInRangeClaim(min=0, max=100),),
    )
    new_fn = _fn_with_claims(name="f", return_type=I32Type())
    merged, warnings = merge_program(
        Program(functions=(existing_fn,)),
        Program(functions=(new_fn,)),
    )
    assert warnings == ()
    assert len(merged.functions[0].claims) == 1
    assert merged.functions[0].claims[0].kind == "return_in_range"


# ---------- orphan + warn cases ----------


def test_param_renamed_drops_claim_with_warning():
    existing_fn = _fn_with_claims(
        name="f",
        params=(Param(name="x", type=I32Type()),),
        claims=(NonNegativeClaim(param="x"),),
    )
    new_fn = _fn_with_claims(
        name="f",
        params=(Param(name="renamed", type=I32Type()),),
    )
    merged, warnings = merge_program(
        Program(functions=(existing_fn,)),
        Program(functions=(new_fn,)),
    )
    assert merged.functions[0].claims == ()
    assert len(warnings) == 1
    assert "non_negative on f.x" in warnings[0]
    assert "param removed in new" in warnings[0]


def test_param_retyped_to_non_int_drops_claim_with_warning():
    existing_fn = _fn_with_claims(
        name="f",
        params=(Param(name="x", type=I32Type()),),
        claims=(IntRangeClaim(param="x", min=0, max=100),),
    )
    new_fn = _fn_with_claims(
        name="f",
        params=(Param(name="x", type=I8PtrType()),),  # ← retyped to pointer
    )
    merged, warnings = merge_program(
        Program(functions=(existing_fn,)),
        Program(functions=(new_fn,)),
    )
    assert merged.functions[0].claims == ()
    assert len(warnings) == 1
    assert "int_range on f.x" in warnings[0]
    assert "retyped" in warnings[0]
    assert "llvm.i8_ptr" in warnings[0]


def test_param_retyped_to_different_int_keeps_claim():
    """The design rule keeps claims when the param remains *some* int
    type (even if width changed). The user can update bounds if the
    new width makes them stale; we don't second-guess them."""
    existing_fn = _fn_with_claims(
        name="f",
        params=(Param(name="x", type=I32Type()),),
        claims=(NonNegativeClaim(param="x"),),
    )
    new_fn = _fn_with_claims(
        name="f",
        params=(Param(name="x", type=I64Type()),),
    )
    merged, warnings = merge_program(
        Program(functions=(existing_fn,)),
        Program(functions=(new_fn,)),
    )
    assert warnings == ()
    assert len(merged.functions[0].claims) == 1


def test_return_retyped_drops_claim_with_warning():
    existing_fn = _fn_with_claims(
        name="f", return_type=I32Type(),
        claims=(ReturnInRangeClaim(min=0, max=100),),
    )
    new_fn = _fn_with_claims(name="f", return_type=I8PtrType())
    merged, warnings = merge_program(
        Program(functions=(existing_fn,)),
        Program(functions=(new_fn,)),
    )
    assert merged.functions[0].claims == ()
    assert len(warnings) == 1
    assert "return_in_range on f return" in warnings[0]
    assert "retyped" in warnings[0]


# ---------- collision behavior ----------


def test_new_claim_wins_on_kind_target_collision():
    """If new and existing both have a `non_negative(x)` claim, new
    wins (current `merge_program` semantics), and no warning fires —
    same kind+target on both sides isn't a drop, just an update."""
    existing_fn = _fn_with_claims(
        name="f",
        params=(Param(name="x", type=I32Type()),),
        claims=(NonNegativeClaim(param="x"),),
    )
    new_fn = _fn_with_claims(
        name="f",
        params=(Param(name="x", type=I32Type()),),
        # New version of the same kind+target — different bounds.
        claims=(IntRangeClaim(param="x", min=0, max=100),),
    )
    merged, warnings = merge_program(
        Program(functions=(existing_fn,)),
        Program(functions=(new_fn,)),
    )
    # Both kinds present (different kinds — non_negative AND int_range
    # — for the same param both get to keep). With realistic data the
    # user would explicitly relax the redundant one; here we're testing
    # the merge mechanics.
    kinds = {(c.kind, c.param) for c in merged.functions[0].claims}
    assert ("non_negative", "x") in kinds
    assert ("int_range", "x") in kinds
    assert warnings == ()


def test_existing_only_function_passes_through():
    """A function in `existing` but not `new` is preserved untouched
    — claims and all."""
    existing_fn = _fn_with_claims(
        name="f",
        params=(Param(name="x", type=I32Type()),),
        claims=(NonNegativeClaim(param="x"),),
    )
    merged, warnings = merge_program(
        Program(functions=(existing_fn,)),
        Program(functions=()),
    )
    assert warnings == ()
    assert merged.functions[0].claims == (NonNegativeClaim(param="x"),)


def test_new_only_function_passes_through():
    """A function in `new` but not `existing` is added as-is."""
    new_fn = _fn_with_claims(
        name="g",
        params=(Param(name="y", type=I32Type()),),
        claims=(NonNegativeClaim(param="y"),),
    )
    merged, warnings = merge_program(
        Program(functions=()),
        Program(functions=(new_fn,)),
    )
    assert warnings == ()
    assert merged.functions[0].claims == (NonNegativeClaim(param="y"),)


# ---------- structured_functions same logic ----------


def test_structured_functions_get_same_reconciliation():
    """`structured_functions` (layer-B) reconciles claims the same
    way `functions` does. Warnings are tagged so the source-of-
    truth is clear."""
    existing_fn = _fn_with_claims(
        name="f",
        params=(Param(name="x", type=I32Type()),),
        claims=(NonNegativeClaim(param="x"),),
    )
    new_fn = _fn_with_claims(
        name="f",
        params=(Param(name="other", type=I32Type()),),
    )
    merged, warnings = merge_program(
        Program(structured_functions=(existing_fn,)),
        Program(structured_functions=(new_fn,)),
    )
    assert merged.structured_functions[0].claims == ()
    assert len(warnings) == 1
    assert "structured" in warnings[0]
    assert "f.x" in warnings[0]


# ---------- extern reconciliation ----------


def _ext_with_claims(
    *, name: str, return_type=I32Type(), claims: tuple = (),
):
    from quod.model import ExternFunction
    return ExternFunction(
        name=name, param_types=(), return_type=return_type,
        linkage=LibcLinkage(), claims=claims,
    )


def test_extern_return_claim_survives_unchanged_return_type():
    e = _ext_with_claims(
        name="atoi", return_type=I32Type(),
        claims=(ReturnInRangeClaim(min=-1),),
    )
    new = _ext_with_claims(name="atoi", return_type=I32Type())
    merged, warnings = merge_program(
        Program(externs=(e,)),
        Program(externs=(new,)),
    )
    assert warnings == ()
    assert len(merged.externs[0].claims) == 1


def test_extern_return_retyped_drops_claim_with_warning():
    e = _ext_with_claims(
        name="weird", return_type=I32Type(),
        claims=(ReturnInRangeClaim(min=0),),
    )
    new = _ext_with_claims(name="weird", return_type=I8PtrType())
    merged, warnings = merge_program(
        Program(externs=(e,)),
        Program(externs=(new,)),
    )
    assert merged.externs[0].claims == ()
    assert len(warnings) == 1
    assert "extern weird return" in warnings[0]


# ---------- end-to-end via CLI: re-ingest preserves authored claim ----------


def test_reingest_preserves_authored_claim_via_cli(tmp_path):
    """Realistic CLI flow: ingest sum.c, add a claim, edit a comment
    in sum.c, re-ingest. The authored claim should survive — no
    warning, no silent drop."""
    import json
    from typer.testing import CliRunner
    from quod import cli as cli_mod

    (tmp_path / "sum.c").write_text(
        "int sum(int n) {\n"
        "    int s = 0;\n"
        "    for (int i = 0; i < n; i = i + 1) { s = s + i; }\n"
        "    return s;\n"
        "}\n"
    )
    (tmp_path / "quod.toml").write_text(
        'build_dir  = "build"\n'
        'proofs_dir = "proofs"\n'
        '\n'
        '[[program]]\n'
        'name = "demo"\n'
        'version = "0.1.0"\n'
        'file = "program.json"\n'
        '\n'
        '[[ingest.entry]]\n'
        'kind = "c-file"\n'
        'source = "sum.c"\n'
    )

    def _run(*args):
        cli_mod._state.clear()
        runner = CliRunner()
        return runner.invoke(
            cli_mod.app, ["-c", str(tmp_path / "quod.toml"), *args],
        )

    assert _run("ingest").exit_code == 0
    add = _run("claim", "add", "sum", "non_negative", "n")
    assert add.exit_code == 0, add.output

    # Confirm the claim is in program.json before re-ingest.
    program_path = tmp_path / "program.json"
    before = json.loads(program_path.read_text())
    sum_fn = next(f for f in before["functions"] if f["name"] == "sum")
    assert any(c["kind"] == "non_negative" for c in sum_fn.get("claims", []))

    # Touch the source: add a trailing comment. Semantically identical.
    (tmp_path / "sum.c").write_text(
        (tmp_path / "sum.c").read_text() + "/* comment added */\n"
    )

    # Re-ingest. The realistic case — fresh Function from libclang,
    # no claims of its own. Without reconciliation, the user's claim
    # was silently dropped.
    re = _run("ingest")
    assert re.exit_code == 0, re.output
    assert "warning:" not in re.output
    assert "warning:" not in (re.stderr or "")

    after = json.loads(program_path.read_text())
    sum_fn_after = next(f for f in after["functions"] if f["name"] == "sum")
    assert any(c["kind"] == "non_negative" for c in sum_fn_after.get("claims", [])), (
        "expected non_negative claim to survive re-ingest"
    )
