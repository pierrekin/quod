"""Layer-A enum constant references.

`CURLOPT_URL` and friends — header-defined enum constants resolved
to integer values at layer B. v6's `CEnumConstRef(name, value)`
preserves both the source-level identifier and the resolved value
so the lift-check can verify equivalence without re-running
libclang. The pinned `value` is what the lift-check compares
against the layer-B `IntLit`; if the enum's resolved value drifts
(library version bump, header re-numbering), `equiv verify` flags
the mismatch.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from quod.ingest.c import ingest_c
from quod.lift_check import LiftCheckError, walk_lift
from quod.model import (
    Block,
    CCall,
    CEnumConstRef,
    CExprStmt,
    CFn,
    CIntLit,
    CNamedType,
    CReturn,
    Function,
    I32Type,
    IntLit,
    Program,
    ReturnExpr,
)


EXAMPLES = Path(__file__).resolve().parents[1] / "examples/c_ingest"


# ---------- model ----------


def test_cenum_const_ref_round_trips():
    e = CEnumConstRef(name="CURLOPT_URL", value=10002)
    raw = e.model_dump_json()
    loaded = CEnumConstRef.model_validate_json(raw)
    assert loaded.name == "CURLOPT_URL"
    assert loaded.value == 10002
    assert loaded.id == e.id


# ---------- ingester ----------


def test_curl_fetch_emits_enum_const_ref_at_layer_a():
    p = ingest_c(EXAMPLES / "curl_fetch/curl_fetch.c")
    fn = p.source_units[0].functions[0]
    setopt = fn.body[1]
    assert setopt.value.callee == "curl_easy_setopt"
    second_arg = setopt.value.args[1]
    assert isinstance(second_arg, CEnumConstRef)
    assert second_arg.name == "CURLOPT_URL"
    # CURLOPT_URL resolves to 10002 in libcurl.
    assert second_arg.value == 10002


def test_curl_fetch_lift_check_passes():
    """Now that `CEnumConstRef` lands, curl_fetch.c gets the full
    three-layer lift and the lift-check verifies the enum value."""
    p = ingest_c(EXAMPLES / "curl_fetch/curl_fetch.c")
    cfn = p.source_units[0].functions[0]
    fn = p.structured_functions[0]
    walk_lift(cfn, fn, program=p)


# ---------- lift-check ----------


def _trivial_enum_const_pair(*, layer_a_value: int, layer_b_value: int):
    cfn = CFn(
        id="@cfn_e", name="e", return_type=CNamedType(name="int"),
        body=(CExprStmt(value=CCall(
            callee="setopt",
            args=(CEnumConstRef(name="MY_OPT", value=layer_a_value),),
        )), CReturn(value=CIntLit(type=I32Type(), value=0))),
    )
    fn = Function(
        id="@fn_e", name="e", return_type=I32Type(),
        body=Block(stmts=(
            __import__("quod.model", fromlist=["ExprStmt"]).ExprStmt(
                value=__import__("quod.model", fromlist=["Call"]).Call(
                    function="setopt",
                    args=(IntLit(type=I32Type(), value=layer_b_value),),
                ),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    return cfn, fn


def test_enum_const_value_match_passes():
    cfn, fn = _trivial_enum_const_pair(layer_a_value=10002, layer_b_value=10002)
    walk_lift(cfn, fn, program=Program(functions=(fn,)))


def test_enum_const_value_drift_is_detected():
    """The pinned layer-A value (10002) disagrees with the layer-B
    IntLit (99999) — simulating an enum re-numbering between when
    the lift was pinned and when it's verified. The lift-check
    surfaces the drift with a clear message."""
    cfn, fn = _trivial_enum_const_pair(layer_a_value=10002, layer_b_value=99999)
    with pytest.raises(LiftCheckError, match="enum constant 'MY_OPT' resolved value 10002"):
        walk_lift(cfn, fn, program=Program(functions=(fn,)))


def test_enum_const_against_non_int_lit_fails():
    """Layer A says enum constant; layer B is something other than
    an IntLit (e.g. a Call). Should refuse."""
    from quod.model import Call, ExprStmt
    cfn = CFn(
        id="@cfn_w", name="w", return_type=CNamedType(name="int"),
        body=(CExprStmt(value=CCall(
            callee="f",
            args=(CEnumConstRef(name="OPT", value=1),),
        )), CReturn(value=CIntLit(type=I32Type(), value=0))),
    )
    fn = Function(
        id="@fn_w", name="w", return_type=I32Type(),
        body=Block(stmts=(
            ExprStmt(value=Call(
                function="f",
                args=(Call(function="resolved_at_runtime", args=()),),
            )),
            ReturnExpr(value=IntLit(type=I32Type(), value=0)),
        )),
    )
    with pytest.raises(LiftCheckError, match="layer-A enum constant 'OPT'"):
        walk_lift(cfn, fn, program=Program(functions=(fn,)))
