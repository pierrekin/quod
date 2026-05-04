"""Round-trip tests for the alloc.json.write serializer.

For each input JSON literal: parse with alloc.json -> JsonValue, serialize
back via alloc.json.write.write_value, and assert the bytes equal the
input. Covers each JsonValue variant plus the escape/encoding edge cases
the serializer is responsible for.

Inputs are kept compact (no whitespace) so byte equality is the right
oracle. Pretty mode is exercised separately by re-parsing its output and
comparing the value tree (via Python's json module).

Escape coverage is limited to what the existing parser (alloc.json)
decodes: \\n, \\t, \\r, \\\\, \\". The parser doesn't yet handle \\b,
\\f, or \\uXXXX, so those round-trip cases are deferred until the
parser catches up. The serializer can still emit \\b / \\f / \\uXXXX
correctly when fed raw control bytes.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from quod.lower import compile_program
from quod.model import Program
from quod.stdlib import resolve_imports


ROUND_TRIP_INPUTS = [
    "null",
    "true",
    "false",
    "0",
    "42",
    "-7",
    "1234567890",
    "-1234567890",
    '""',
    '"hi"',
    '"a b c"',
    r'"\n"',
    r'"\t"',
    r'"\r"',
    r'"\\"',
    r'"\""',
    "[]",
    "[1]",
    "[1,2,3]",
    "[true,false,null]",
    '["a","b","c"]',
    "[[1,2],[3,4]]",
    "{}",
    '{"a":1}',
    '{"a":1,"b":2}',
    '{"a":[1,2],"b":{"c":-3}}',
    '{"k":"v\\nw"}',
]


PRETTY_INPUTS = [
    "null",
    "[1,2,3]",
    '{"a":1,"b":[true,null]}',
    '{"nested":{"deep":[1,[2,[3]]]}}',
]


def _make_program(input_text: str, *, pretty: bool) -> Program:
    return Program.model_validate_json(json.dumps({
        "imports": ["alloc.json", "alloc.json.write", "alloc.str"],
        "constants": [
            {"name": ".input", "value": input_text},
            {"name": ".err",   "value": "ERR"},
        ],
        "externs": [
            {
                "name": "puts",
                "param_types": [{"kind": "llvm.i8_ptr"}],
                "linkage": {"kind": "linkage.libc"},
            },
        ],
        "functions": [
            {
                "name": "main",
                "params": [],
                "return_type": {"kind": "llvm.i32"},
                "body": [
                    {
                        "kind": "quod.with_arena",
                        "name": "a",
                        "capacity": {"kind": "llvm.const_int", "type": {"kind": "llvm.i64"}, "value": 16384},
                        "body": _round_trip_body(pretty=pretty),
                    },
                    {
                        "kind": "quod.return_expr",
                        "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 0},
                    },
                ],
                "claims": [],
            },
        ],
    }))


def _round_trip_body(*, pretty: bool) -> list[dict]:
    pretty_v = 1 if pretty else 0
    return [
        {
            "kind": "quod.let", "name": "input",
            "type": {"kind": "llvm.struct", "name": "core.str.String"},
            "init": {
                "kind": "llvm.call", "function": "core.str.from_cstr",
                "args": [{"kind": "quod.string_ref", "name": ".input"}],
            },
        },
        {
            "kind": "quod.let", "name": "result",
            "type": {"kind": "llvm.enum", "name": "alloc.json.ParseResult"},
            "init": {
                "kind": "llvm.call", "function": "alloc.json.parse",
                "args": [
                    {"kind": "quod.field", "value": {"kind": "quod.local_ref", "name": "input"}, "name": "ptr"},
                    {"kind": "quod.field", "value": {"kind": "quod.local_ref", "name": "input"}, "name": "len"},
                    {"kind": "quod.local_ref", "name": "a"},
                ],
            },
        },
        {
            "kind": "quod.match",
            "scrutinee": {"kind": "quod.local_ref", "name": "result"},
            "arms": [
                {
                    "variant": "Ok", "bindings": ["v"],
                    "body": [
                        {
                            "kind": "quod.let", "name": "out",
                            "type": {"kind": "llvm.struct", "name": "core.str.String"},
                            "init": {
                                "kind": "llvm.call", "function": "alloc.json.write.write_value",
                                "args": [
                                    {"kind": "quod.local_ref", "name": "a"},
                                    {"kind": "quod.local_ref", "name": "v"},
                                    {"kind": "llvm.const_int", "type": {"kind": "llvm.i1"}, "value": pretty_v},
                                ],
                            },
                        },
                        {
                            "kind": "quod.let", "name": "cs",
                            "type": {"kind": "llvm.i8_ptr"},
                            "init": {
                                "kind": "llvm.call", "function": "alloc.str.to_cstr_in",
                                "args": [
                                    {"kind": "quod.local_ref", "name": "out"},
                                    {"kind": "quod.local_ref", "name": "a"},
                                ],
                            },
                        },
                        {
                            "kind": "quod.expr_stmt",
                            "value": {
                                "kind": "llvm.call", "function": "puts",
                                "args": [{"kind": "quod.local_ref", "name": "cs"}],
                            },
                        },
                    ],
                },
                {
                    "variant": "Err", "bindings": [],
                    "body": [
                        {
                            "kind": "quod.expr_stmt",
                            "value": {
                                "kind": "llvm.call", "function": "puts",
                                "args": [{"kind": "quod.string_ref", "name": ".err"}],
                            },
                        },
                    ],
                },
            ],
        },
    ]


def _build_and_run(prog: Program) -> str:
    prog = resolve_imports(prog)
    with tempfile.TemporaryDirectory() as td:
        res = compile_program(prog, build_dir=Path(td), bins=(("rt", "main"),), profile=2, link=True)
        binary = res.bins[0].binary
        out = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15)
    assert out.returncode == 0, f"binary exited {out.returncode}; stderr={out.stderr!r}"
    return out.stdout


@pytest.mark.parametrize("inp", ROUND_TRIP_INPUTS, ids=lambda s: s)
def test_roundtrip_compact(inp):
    prog = _make_program(inp, pretty=False)
    actual = _build_and_run(prog)
    assert actual == inp + "\n", f"input={inp!r}, output={actual!r}"


@pytest.mark.parametrize("inp", PRETTY_INPUTS, ids=lambda s: s)
def test_roundtrip_pretty(inp):
    prog = _make_program(inp, pretty=True)
    actual = _build_and_run(prog)
    pretty_text = actual.rstrip("\n")
    reparsed = json.loads(pretty_text)
    expected = json.loads(inp)
    assert reparsed == expected, (
        f"value tree differs after pretty round-trip\n"
        f"  input:    {inp}\n  pretty:   {pretty_text}\n  reparsed: {reparsed}\n  expected: {expected}"
    )
