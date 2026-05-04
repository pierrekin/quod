"""Corpus round-trip tests for the alloc.json.write serializer.

For each file in CORPUS: read its bytes, embed them as a string constant,
parse with alloc.json -> JsonValue, serialize back via
alloc.json.write.write_value (compact), and assert the re-parsed value
tree equals the value tree of the original input.

Byte-equality is the wrong oracle here because the inputs are
hand-formatted (whitespace, key order, indentation) and the serializer
emits compact. So we go through Python's json on both sides and compare
structurally — the same approach the pretty-mode tests in
test_json_writer_roundtrip.py use.

Corpus is the program JSONs that exercise alloc.json end-to-end (the
substrate eating its own dog food) plus the stdlib JSONs themselves.
The 8 examples/json/phase* folders are excluded — those are the
hand-built-parser walkthrough, not value corpus.
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


_REPO_ROOT = Path(__file__).resolve().parents[1]

CORPUS = [
    "examples/json_demo/program.json",
    "examples/json_v2/program.json",
    "examples/json_v3/program.json",
    "examples/json_writer_smoke/program.json",
    "src/quod/stdlib/core.bytes.json",
    "src/quod/stdlib/core.str.json",
    "src/quod/stdlib/mem.arena.json",
    "src/quod/stdlib/alloc.str.json",
    "src/quod/stdlib/alloc.json.json",
    "src/quod/stdlib/alloc.json.write.json",
    "src/quod/stdlib/std.io.json",
]


def _make_program(input_text: str, *, capacity: int) -> Program:
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
                        "capacity": {"kind": "llvm.const_int", "type": {"kind": "llvm.i64"}, "value": capacity},
                        "body": _round_trip_body(),
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


def _round_trip_body() -> list[dict]:
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
                                    {"kind": "llvm.const_int", "type": {"kind": "llvm.i1"}, "value": 0},
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
        out = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, f"binary exited {out.returncode}; stderr={out.stderr!r}"
    return out.stdout


@pytest.mark.parametrize("rel_path", CORPUS, ids=lambda p: p)
def test_roundtrip_corpus(rel_path):
    src = _REPO_ROOT / rel_path
    input_text = src.read_text()

    # Arena holds: parsed JsonValue tree + Builder bytes for the compact
    # output + small intermediates. 32x input is comfortable headroom for
    # programs with deeply nested arrays/objects (each node is a tagged
    # union > one byte of input).
    capacity = max(64 * 1024, len(input_text) * 32)

    prog = _make_program(input_text, capacity=capacity)
    actual = _build_and_run(prog).rstrip("\n")

    actual_tree = json.loads(actual)
    expected_tree = json.loads(input_text)
    assert actual_tree == expected_tree, (
        f"value tree differs after compact round-trip of {rel_path}\n"
        f"  expected keys at top: {sorted(expected_tree) if isinstance(expected_tree, dict) else type(expected_tree).__name__}\n"
        f"  actual   keys at top: {sorted(actual_tree)   if isinstance(actual_tree,   dict) else type(actual_tree).__name__}"
    )
