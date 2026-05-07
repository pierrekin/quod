"""Cross-layer range-hint provider — `ghidra.range_hints`.

Exercises the first cross-layer claim flow per
`.scratch/ghidra/04-providers-and-the-bet.md`: walk
`Program.binary_units`, find signed-compare constants in p-code, emit
candidate `PredicateClaim`s on the paired source `Function`.

Uses hand-crafted JSON fixtures via `ingest_binary_dump` so the tests
don't need Ghidra; the matching real-Ghidra exercise lives in
`test_binary_e2e.py`.
"""
from __future__ import annotations

import json

import pytest

from quod.ingest.binary import ingest_binary_dump
from quod.merge import merge_program
from quod.model import (
    BinaryProvenance,
    DerivedJustification,
    Function,
    I32Type,
    IntLit,
    Param,
    PredicateClaim,
    Program,
    ReturnExpr,
)
from quod.model.statements import Block
from quod.predicate.binary_hints import derive_binary_range_hints
from quod.predicate.providers import all_providers, get_provider


# ---------- Fixtures ----------


def _libsign_dump() -> dict:
    """Mimics Ghidra's output for a tiny .so containing one function:

        int compare_const(int v) {
            if (v < 100) return v;
            return 100;
        }

    Two basic blocks; the entry block contains an INT_SLESS against
    100 (the comparison threshold). One bin.fn, no externs, no calls.
    """
    return {
        "schema_version": 1,
        "binary": {
            "path": "/tmp/build/libsign.so",
            "sha256": "a" * 64,
            "arch": "x86:LE:64:default",
            "format": "elf",
            "build_id": None,
        },
        "functions": [
            {
                "address": "0x401120",
                "name_mangled": "compare_const",
                "name_demangled": "compare_const",
                "signature": {
                    "return_type": "int",
                    "params": [{"name": "v", "type": "int"}],
                },
                "calling_convention": "__stdcall",
                "decompile": "int compare_const(int v) {\n  if (v < 100) return v;\n  return 100;\n}\n",
                "basic_blocks": [
                    {
                        "address": "0x401120",
                        "end": "0x401140",
                        "successors": [],
                        "pcode": [
                            {
                                "opcode": "INT_SLESS",
                                "inputs": [
                                    {"space": "register", "offset": 0x38, "size": 4},
                                    {"space": "const", "offset": 100, "size": 4},
                                ],
                                "output": {"space": "register", "offset": 0x200, "size": 1},
                                "instr_address": "0x401128",
                            },
                            {
                                "opcode": "RETURN",
                                "inputs": [
                                    {"space": "register", "offset": 0x20, "size": 8}
                                ],
                                "output": None,
                                "instr_address": "0x40113e",
                            },
                        ],
                    },
                ],
                "calls": [],
            },
        ],
        "data": [],
        "externs": [],
        "type_refs": [],
    }


def _compare_const_function() -> Function:
    """Layer-C `Function` matching the `compare_const` from
    `_libsign_dump` by name. The seeder pairs by demangled name."""
    return Function(
        name="compare_const",
        params=(Param(name="v", type=I32Type()),),
        return_type=I32Type(),
        body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
    )


def _seed_program(tmp_path) -> Program:
    """Drive the binary-side ingest against `_libsign_dump`, and pair
    it to a Layer-C source `Function` so the seeder writes a
    `BinaryProvenance` equivalence — that's the cross-layer link the
    range-hint provider needs to attribute claims back."""
    dump = tmp_path / "libsign.json"
    dump.write_text(json.dumps(_libsign_dump()))
    base = Program(functions=(_compare_const_function(),))
    return ingest_binary_dump(dump, program=base)


# ---------- Provider behavior ----------


def test_range_hints_emits_candidates_against_paired_function(tmp_path):
    """The provider walks `BinaryProvenance` equivalences, finds the
    INT_SLESS constant K=100 in the bin.fn's p-code, and emits
    candidate `int_range` claims keyed on the paired source
    `Function.name`."""
    program = _seed_program(tmp_path)
    derived = derive_binary_range_hints(program)
    assert "compare_const" in derived
    claims = derived["compare_const"]
    assert claims, "expected at least one candidate"
    for c in claims:
        assert isinstance(c, PredicateClaim)
        assert c.regime == "lattice"
        assert isinstance(c.justification, DerivedJustification)
        assert c.justification.analysis == "ghidra.range_hints"
        # `inputs` pins the bin.fn id the hint came from.
        assert c.justification.inputs
        assert any(i.startswith("@binfn_") for i in c.justification.inputs)


def test_range_hints_constant_appears_in_emitted_predicate(tmp_path):
    """K=100 from the INT_SLESS should appear as a bound on at least
    one emitted predicate. We don't pin a specific shape — the canonical
    form has the literal somewhere in the AST — but it should be in
    the rendered text."""
    program = _seed_program(tmp_path)
    derived = derive_binary_range_hints(program)
    claims = derived["compare_const"]
    rendered = "\n".join(c.expr.model_dump_json() for c in claims)
    assert "100" in rendered or "99" in rendered, rendered


def test_range_hints_produces_no_claims_without_pairing(tmp_path):
    """If the program has no source-side `Function` matching the
    bin.fn's demangled_name, the seeder doesn't pair and the
    range-hint provider has nothing to attribute claims to."""
    dump = tmp_path / "libsign.json"
    dump.write_text(json.dumps(_libsign_dump()))
    program = ingest_binary_dump(dump)  # no source-side Function
    derived = derive_binary_range_hints(program)
    assert derived == {}


def test_range_hints_skips_multi_int_param_functions(tmp_path):
    """V1 only handles single-int-parameter functions — multi-param
    needs ABI-aware varnode→param mapping (see module docstring)."""
    base = Program(functions=(
        Function(
            name="compare_const",
            params=(Param(name="v", type=I32Type()), Param(name="w", type=I32Type())),
            return_type=I32Type(),
            body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
        ),
    ))
    dump = tmp_path / "libsign.json"
    dump.write_text(json.dumps(_libsign_dump()))
    program = ingest_binary_dump(dump, program=base)
    derived = derive_binary_range_hints(program)
    assert derived == {}, "v1 should skip multi-param functions"


def test_range_hints_filters_artifact_constants(tmp_path):
    """Constants like 0xFF and 0x1 are compiler-emitted artifacts
    (byte masks, popcount lanes) — they should not produce hints
    even though they appear in p-code."""
    dump_data = _libsign_dump()
    # Replace the K=100 comparison with K=0xFF (artifact).
    dump_data["functions"][0]["basic_blocks"][0]["pcode"][0]["inputs"][1] = {
        "space": "const", "offset": 0xFF, "size": 4,
    }
    dump = tmp_path / "libsign.json"
    dump.write_text(json.dumps(dump_data))
    base = Program(functions=(_compare_const_function(),))
    program = ingest_binary_dump(dump, program=base)
    derived = derive_binary_range_hints(program)
    assert derived == {}, "0xFF is a byte mask artifact and must not surface"


def test_range_hints_caps_per_function(tmp_path):
    """A function with many distinct compare constants should emit at
    most _MAX_HINTS_PER_FUNCTION candidates (default 6)."""
    dump_data = _libsign_dump()
    pcode = dump_data["functions"][0]["basic_blocks"][0]["pcode"]
    # Replace the single INT_SLESS with 10 distinct comparisons.
    cmps = []
    for k in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
        cmps.append({
            "opcode": "INT_SLESS",
            "inputs": [
                {"space": "register", "offset": 0x38, "size": 4},
                {"space": "const", "offset": k, "size": 4},
            ],
            "output": {"space": "register", "offset": 0x200, "size": 1},
            "instr_address": "0x401128",
        })
    dump_data["functions"][0]["basic_blocks"][0]["pcode"] = cmps + pcode[1:]
    dump = tmp_path / "libsign.json"
    dump.write_text(json.dumps(dump_data))
    base = Program(functions=(_compare_const_function(),))
    program = ingest_binary_dump(dump, program=base)
    derived = derive_binary_range_hints(program)
    assert "compare_const" in derived
    assert len(derived["compare_const"]) <= 6


def test_range_hints_signed_constants_reinterpret_correctly(tmp_path):
    """Ghidra encodes `-1` as the unsigned 0xFFFFFFFF in a 4-byte
    varnode. The provider must reinterpret to signed so the candidate
    bound is human-meaningful (`v <= -2`, not `v <= 4294967294`)."""
    dump_data = _libsign_dump()
    dump_data["functions"][0]["basic_blocks"][0]["pcode"][0]["inputs"][1] = {
        "space": "const", "offset": 0xFFFFFFFF, "size": 4,  # encoding of -1
    }
    dump = tmp_path / "libsign.json"
    dump.write_text(json.dumps(dump_data))
    base = Program(functions=(_compare_const_function(),))
    program = ingest_binary_dump(dump, program=base)
    derived = derive_binary_range_hints(program)
    # 0xFFFFFFFF is filtered as an artifact (whole-bits mask), so this
    # particular encoding produces no claim — sanity-check the artifact
    # filter; the resign is exercised below for non-filtered values.
    assert derived == {}

    # Now encode -2 (which is not artifact-filtered) and confirm the
    # bound flowing into predicate sugar is signed.
    dump_data["functions"][0]["basic_blocks"][0]["pcode"][0]["inputs"][1] = {
        "space": "const", "offset": 0xFFFFFFFE, "size": 4,  # encoding of -2
    }
    dump.write_text(json.dumps(dump_data))
    program2 = ingest_binary_dump(dump, program=base)
    derived2 = derive_binary_range_hints(program2)
    rendered = "\n".join(c.expr.model_dump_json() for c in derived2["compare_const"])
    # The literal -2 (or -3 from K-1) should appear as a signed value,
    # not as 4294967294 / 4294967293.
    assert '"value":-' in rendered or '"value": -' in rendered
    assert "4294967294" not in rendered
    assert "4294967293" not in rendered


# ---------- Compiler-lowered comparison chains ----------


def _libsign_dump_with_sub_chain() -> dict:
    """Same shape as `_libsign_dump`, but the INT_SLESS pattern follows
    clang's `-O0 -g` lowering of `v < 100`:

        INT_SUB v_loaded, 100  -> tmp
        INT_SLESS tmp, 0

    The constant 100 lives on the INT_SUB, not the INT_SLESS. The
    chain walker should recover it by walking back through the
    per-block varnode-definition map.
    """
    base = _libsign_dump()
    base["functions"][0]["basic_blocks"][0]["pcode"] = [
        # Pretend the parameter is loaded into a temp varnode (in real
        # output this would be a stack roundtrip; the precise shape
        # doesn't matter for the chain walker — only the producer
        # relationship between INT_SUB and INT_SLESS does).
        {
            "opcode": "COPY",
            "inputs": [{"space": "register", "offset": 0x38, "size": 4}],
            "output": {"space": "unique", "offset": 0x100, "size": 4},
            "instr_address": "0x401120",
        },
        # The lowered comparison: subtract the threshold first…
        {
            "opcode": "INT_SUB",
            "inputs": [
                {"space": "unique", "offset": 0x100, "size": 4},
                {"space": "const", "offset": 100, "size": 4},
            ],
            "output": {"space": "unique", "offset": 0x200, "size": 4},
            "instr_address": "0x401124",
        },
        # …then sign-test the result.
        {
            "opcode": "INT_SLESS",
            "inputs": [
                {"space": "unique", "offset": 0x200, "size": 4},
                {"space": "const", "offset": 0, "size": 4},
            ],
            "output": {"space": "register", "offset": 0x207, "size": 1},
            "instr_address": "0x401128",
        },
        {
            "opcode": "RETURN",
            "inputs": [{"space": "register", "offset": 0x20, "size": 8}],
            "output": None,
            "instr_address": "0x40113e",
        },
    ]
    return base


def test_chain_walker_recovers_int_sub_threshold(tmp_path):
    """clang's `-O0 -g` lowers `v < 100` to `INT_SUB v, 100 -> tmp;
    INT_SLESS tmp, 0`. The chain walker must recover K=100 from the
    INT_SUB and emit candidates against it (not just K=0 from the
    INT_SLESS sign test)."""
    dump = tmp_path / "libsign.json"
    dump.write_text(json.dumps(_libsign_dump_with_sub_chain()))
    base = Program(functions=(_compare_const_function(),))
    program = ingest_binary_dump(dump, program=base)
    derived = derive_binary_range_hints(program)
    assert "compare_const" in derived
    # K=100 should appear in the rendered predicates. The K=0 sign-test
    # also appears (we don't suppress it; the verifier filters), but
    # the load-bearing assertion is K=100 isn't lost.
    rendered = "\n".join(c.expr.model_dump_json() for c in derived["compare_const"])
    assert "100" in rendered or "99" in rendered, rendered


def test_chain_walker_recovers_int_add_negative_threshold(tmp_path):
    """The symmetric form: `v + (-100) < 0` ≡ `v < 100`. clang
    sometimes emits this when the immediate fits the ADD encoding
    better than SUB."""
    dump_data = _libsign_dump_with_sub_chain()
    pcode = dump_data["functions"][0]["basic_blocks"][0]["pcode"]
    # Replace the INT_SUB with an INT_ADD against the *negative* of K.
    pcode[1] = {
        "opcode": "INT_ADD",
        "inputs": [
            {"space": "unique", "offset": 0x100, "size": 4},
            {"space": "const", "offset": (-100) & 0xFFFFFFFF, "size": 4},
        ],
        "output": {"space": "unique", "offset": 0x200, "size": 4},
        "instr_address": "0x401124",
    }
    dump = tmp_path / "libsign.json"
    dump.write_text(json.dumps(dump_data))
    base = Program(functions=(_compare_const_function(),))
    program = ingest_binary_dump(dump, program=base)
    derived = derive_binary_range_hints(program)
    rendered = "\n".join(c.expr.model_dump_json() for c in derived["compare_const"])
    assert "100" in rendered or "99" in rendered, rendered


def test_chain_walker_does_not_cross_basic_blocks(tmp_path):
    """Per-block scope only: an INT_SUB in block A producing varnode X
    must not be picked up by an INT_SLESS X, 0 in block B (because at
    block B's entry, X could come from any predecessor — the chain is
    not sound across CFG edges)."""
    dump_data = _libsign_dump()
    # Replace the single block with two blocks; the SUB lives in the
    # first, the SLESS in the second.
    dump_data["functions"][0]["basic_blocks"] = [
        {
            "address": "0x401120",
            "end": "0x401128",
            "successors": [],  # successors are stitched by addr; empty is fine for v1
            "pcode": [
                {
                    "opcode": "INT_SUB",
                    "inputs": [
                        {"space": "register", "offset": 0x38, "size": 4},
                        {"space": "const", "offset": 100, "size": 4},
                    ],
                    "output": {"space": "unique", "offset": 0x200, "size": 4},
                    "instr_address": "0x401120",
                },
            ],
        },
        {
            "address": "0x401128",
            "end": "0x401130",
            "successors": [],
            "pcode": [
                # The INT_SLESS in a *different* block; the per-block
                # defs map is fresh here, so the chain walker cannot
                # reach the INT_SUB in the previous block.
                {
                    "opcode": "INT_SLESS",
                    "inputs": [
                        {"space": "unique", "offset": 0x200, "size": 4},
                        {"space": "const", "offset": 0, "size": 4},
                    ],
                    "output": {"space": "register", "offset": 0x207, "size": 1},
                    "instr_address": "0x401128",
                },
            ],
        },
    ]
    dump = tmp_path / "libsign.json"
    dump.write_text(json.dumps(dump_data))
    base = Program(functions=(_compare_const_function(),))
    program = ingest_binary_dump(dump, program=base)
    derived = derive_binary_range_hints(program)
    # Only the K=0 from the cross-block INT_SLESS should surface, NOT
    # K=100 (the cross-block chain isn't a valid recovery path).
    if "compare_const" in derived:
        rendered = "\n".join(c.expr.model_dump_json() for c in derived["compare_const"])
        assert "100" not in rendered, (
            f"chain walker reached across a basic block boundary; "
            f"recovered K=100 from cross-block INT_SUB: {rendered}"
        )


# ---------- Provider registry integration ----------


def test_provider_is_registered():
    """`ghidra.range_hints` is a built-in provider, discoverable via
    the registry and supports `derive` mode."""
    p = get_provider("ghidra.range_hints")
    assert p.regime == "lattice"
    assert "derive" in p.modes
    assert p in all_providers().values()


def test_provider_default_for_lattice_unchanged():
    """Adding ghidra.range_hints must not displace lattice.literal_range
    as the default lattice/derive provider — the existing
    `quod claim derive` behavior should still pick literal_range."""
    from quod.predicate.providers import default_for
    p = default_for(regime="lattice", mode="derive")
    assert p.name == "lattice.literal_range"
