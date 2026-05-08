"""decompile_lift v2 — parse Ghidra's decompile_text into Layer-A CFns.

Synthetic tests run libclang against hand-built decompile_text strings;
no Ghidra needed (the lift's input is just bytes — Ghidra is the
typical *source* of those bytes, not a runtime dependency for the
lift itself).
"""
from __future__ import annotations

import hashlib

import pytest

from quod.model import (
    BinFunction,
    BinFunctionParam,
    BinUnit,
    CFn,
    CIntLit,
    CBinOp,
    CReturn,
    CVarRef,
    DecompileLift,
    Equivalence,
    Program,
    load_program,
    save_program,
)
from quod.predicate.binary_decompile_lift import (
    derive_decompile_lifts,
    lift_decompile,
)


def _bin_fn(
    name: str, *, decompile_text: str, n_params: int = 0, fn_id: str | None = None,
) -> BinFunction:
    return BinFunction(
        id=fn_id or f"@binfn_test_{name}",
        address=0x1000,
        mangled_name=name,
        demangled_name=name,
        return_type_name="int",
        params=tuple(
            BinFunctionParam(name=f"p{i}", type_name="int")
            for i in range(n_params)
        ),
        calling_convention="x86_64-sysv",
        decompile_text=decompile_text,
    )


# ---------- lift_decompile ----------

def test_ident_lifts_to_return_varref():
    """`int ident(int x) { return x; }` → CFn with `CReturn(CVarRef("x"))`."""
    bf = _bin_fn(
        "ident", n_params=1,
        decompile_text="int ident(int x) { return x; }\n",
    )
    result = lift_decompile(bf)

    assert result is not None
    cfn = result.cfn
    assert cfn.name == "ident"
    assert cfn.id == f"@cfn_lifted_{bf.id.removeprefix('@')}"
    assert len(cfn.params) == 1 and cfn.params[0].name == "x"
    assert len(cfn.body) == 1
    ret = cfn.body[0]
    assert isinstance(ret, CReturn)
    assert isinstance(ret.value, CVarRef) and ret.value.name == "x"
    # The hash pins the bytes that produced this lift.
    assert result.decompile_text_sha256 == hashlib.sha256(
        bf.decompile_text.encode("utf-8")
    ).hexdigest()


def test_add_lifts_to_binop():
    """`int add(int a, int b) { return a + b; }` → CReturn(CBinOp(add, …))."""
    bf = _bin_fn(
        "add", n_params=2,
        decompile_text="int add(int a, int b) { return a + b; }\n",
    )
    result = lift_decompile(bf)

    assert result is not None
    body = result.cfn.body
    assert len(body) == 1
    ret = body[0]
    assert isinstance(ret, CReturn)
    assert isinstance(ret.value, CBinOp)
    assert ret.value.op == "+"
    # Operands are the param refs.
    assert isinstance(ret.value.lhs, CVarRef) and ret.value.lhs.name == "a"
    assert isinstance(ret.value.rhs, CVarRef) and ret.value.rhs.name == "b"


def test_affine_lifts_with_nested_binops():
    """`int affine(int x) { return x * 3 + 5; }` — Ghidra's exact
    output for our POC fixture, including the operator precedence."""
    bf = _bin_fn(
        "affine", n_params=1,
        decompile_text="int affine(int x) { return x * 3 + 5; }\n",
    )
    result = lift_decompile(bf)

    assert result is not None
    body = result.cfn.body
    assert isinstance(body[0], CReturn)
    outer = body[0].value
    # `x * 3 + 5` is `(x * 3) + 5` by C operator precedence.
    assert isinstance(outer, CBinOp) and outer.op == "+"
    assert isinstance(outer.lhs, CBinOp) and outer.lhs.op == "*"
    assert isinstance(outer.rhs, CIntLit) and outer.rhs.value == 5


def test_empty_decompile_text_returns_none():
    bf = _bin_fn("noop", decompile_text="")
    assert lift_decompile(bf) is None


def test_unparseable_decompile_text_returns_none():
    """Real Ghidra emits things like `undefined4 *pcVar1` for
    type-recovery failures. libclang chokes on `undefined4` as an
    unknown type — the lift must return None rather than raise."""
    bf = _bin_fn(
        "broken", n_params=0,
        decompile_text="undefined4 broken(undefined4 x) { return x; }\n",
    )
    assert lift_decompile(bf) is None


def test_lift_id_is_stable_across_calls():
    """Two lifts of the same bin.fn produce the same lifted CFn id —
    re-running the lift is idempotent at the id level."""
    bf = _bin_fn("f", decompile_text="int f(void) { return 7; }\n")
    a = lift_decompile(bf)
    b = lift_decompile(bf)
    assert a is not None and b is not None
    assert a.cfn.id == b.cfn.id
    assert a.decompile_text_sha256 == b.decompile_text_sha256


def test_lift_id_does_not_collide_with_source_cfn_id():
    """Source CFns are minted as `@cfn_c_<name>`; lifted CFns must use
    a distinct prefix so the same name in source and binary doesn't
    yield a duplicate id."""
    bf = _bin_fn("ident", decompile_text="int ident(int x) { return x; }\n")
    result = lift_decompile(bf)
    assert result is not None
    assert result.cfn.id != "@cfn_c_ident"
    assert result.cfn.id.startswith("@cfn_lifted_")


# ---------- derive_decompile_lifts ----------

def _program_with_one_bin_fn(bf: BinFunction) -> Program:
    bu = BinUnit(
        path="/tmp/x.so", sha256="0" * 64,
        arch="x86_64", file_format="elf",
        functions=(bf,),
    )
    return Program(binary_units=(bu,))


def test_derive_attaches_cfns_and_emits_equivalences():
    bf = _bin_fn("f", decompile_text="int f(int x) { return x; }\n", n_params=1)
    prog = _program_with_one_bin_fn(bf)

    new_prog, lifts = derive_decompile_lifts(prog)

    assert len(lifts) == 1
    [u] = new_prog.binary_units
    assert len(u.lifted_cfns) == 1
    assert u.lifted_cfns[0].name == "f"

    # Exactly one equivalence with DecompileLift justification.
    eqs = [e for e in new_prog.equivalences
           if e.justification is not None
           and e.justification.kind == "decompile_lift"]
    assert len(eqs) == 1
    eq = eqs[0]
    assert eq.b_node_id == bf.id
    assert eq.a_node_id == u.lifted_cfns[0].id
    assert eq.regime == "axiom"
    assert eq.justification.decompile_text_sha256 == lifts[0].decompile_text_sha256


def test_derive_is_idempotent_on_unchanged_text():
    bf = _bin_fn("f", decompile_text="int f(void) { return 1; }\n")
    prog = _program_with_one_bin_fn(bf)
    once, _ = derive_decompile_lifts(prog)
    twice, _ = derive_decompile_lifts(once)

    [u] = twice.binary_units
    assert len(u.lifted_cfns) == 1, (
        "second run should not add a duplicate CFn"
    )
    eqs = [e for e in twice.equivalences
           if e.justification is not None
           and e.justification.kind == "decompile_lift"]
    assert len(eqs) == 1


def test_derive_skips_unparseable_function_silently():
    """Mixed input: one parseable function, one with `undefined4` — the
    lift produces a CFn for the first and silently drops the second."""
    bf_ok = _bin_fn(
        "ok", n_params=0,
        decompile_text="int ok(void) { return 0; }\n",
        fn_id="@binfn_ok",
    )
    bf_bad = _bin_fn(
        "bad", n_params=0,
        decompile_text="undefined4 bad(undefined4 x) { return x; }\n",
        fn_id="@binfn_bad",
    )
    bu = BinUnit(
        path="/tmp/x.so", sha256="0" * 64,
        arch="x86_64", file_format="elf",
        functions=(bf_ok, bf_bad),
    )
    prog = Program(binary_units=(bu,))

    new_prog, lifts = derive_decompile_lifts(prog)

    assert len(lifts) == 1
    assert lifts[0].bin_fn_id == bf_ok.id
    [u] = new_prog.binary_units
    assert len(u.lifted_cfns) == 1
    assert u.lifted_cfns[0].name == "ok"


def test_derive_round_trips_through_save_load(tmp_path):
    """The full program — with lifted CFns and DecompileLift-justified
    equivalences — round-trips byte-for-byte through JSON I/O."""
    bf = _bin_fn(
        "f", n_params=1,
        decompile_text="int f(int x) { return x + 1; }\n",
    )
    prog, _ = derive_decompile_lifts(_program_with_one_bin_fn(bf))
    out = tmp_path / "p.json"
    save_program(prog, out)
    loaded = load_program(out)
    assert loaded == prog


def test_derive_with_no_binary_units_is_noop():
    prog = Program()
    new_prog, lifts = derive_decompile_lifts(prog)
    assert new_prog == prog
    assert lifts == ()
