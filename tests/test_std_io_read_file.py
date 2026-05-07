"""Smoke tests for `std.io.read_file<A>` returning a Reader.

Each test writes a JSON document to a tempfile, builds a quod program
that reads it with `read_file<Arena>` and parses the stream with
`alloc.json.io.read_value<BufReader<File>>`, and verifies the program's
exit code reflects the parsed value.

Distinct exit codes for each failure mode let one assertion pin down
exactly where a regression broke:

  10 — read_file returned Err (open failed, permission, etc.)
  11 — read_value returned None (parse failed mid-stream)
  12 — parsed JsonValue was not a Number variant
  N  — parsed JsonValue::Number(N) — the success path
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from quod.lower import compile_program
from quod.model import Program
from quod.resolve import resolve_imports


ARENA = {"kind": "llvm.struct", "name": "mem.arena.Arena"}


def _wired_arena_import(module: str) -> dict:
    return {"module": module, "wire": [{"name": "A", "type": ARENA}]}


def _make_program(path_const: str) -> Program:
    return Program.model_validate_json(json.dumps({
        "imports": [
            _wired_arena_import("alloc.json"),
            _wired_arena_import("alloc.json.io"),
            _wired_arena_import("alloc.io"),
            _wired_arena_import("std.io"),
        ],
        "constants": [
            {"name": ".path", "value": path_const},
        ],
        "functions": [
            {
                "name": "main",
                "params": [],
                "return_type": {"kind": "llvm.i32"},
                "body": {"stmts": [
                    {
                        "kind": "quod.with_arena",
                        "name": "a",
                        "capacity": {"kind": "llvm.const_int", "type": {"kind": "llvm.i64"}, "value": 65536},
                        "body": {"stmts": _body()},
                    },
                ]},
                "claims": [],
            },
        ],
    }))


def _ret(n: int) -> dict:
    return {
        "kind": "quod.return_expr",
        "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": n},
    }


def _jsonvalue_arms() -> list[dict]:
    arms: list[dict] = [
        {
            "variant": "Number", "bindings": ["n"],
            "body": {"stmts": [
                {
                    "kind": "quod.return_expr",
                    "value": {
                        "kind": "quod.cast",
                        "value": {"kind": "quod.local_ref", "name": "n"},
                        "target_type": {"kind": "llvm.i32"},
                    },
                },
            ]},
        },
        {"variant": "Null",   "bindings": [],                "body": {"stmts": [_ret(12)]}},
        {"variant": "Bool",   "bindings": ["_b"],            "body": {"stmts": [_ret(12)]}},
        {"variant": "String", "bindings": ["_p", "_l"],      "body": {"stmts": [_ret(12)]}},
        {"variant": "Array",  "bindings": ["_items"],        "body": {"stmts": [_ret(12)]}},
        {"variant": "Object", "bindings": ["_keys", "_vals"], "body": {"stmts": [_ret(12)]}},
    ]
    return arms


def _body() -> list[dict]:
    result_t = {
        "kind": "llvm.enum", "name": "core.result.Result",
        "type_args": [{"kind": "llvm.i8_ptr"}, {"kind": "llvm.enum", "name": "core.io.IoError"}],
    }
    option_jv_t = {
        "kind": "llvm.enum", "name": "core.option.Option",
        "type_args": [{"kind": "llvm.enum", "name": "alloc.json.JsonValue"}],
    }
    bufreader_file_t = {
        "kind": "llvm.struct", "name": "alloc.io.BufReader",
        "type_args": [{"kind": "llvm.struct", "name": "std.io.File"}],
    }

    return [
        {
            "kind": "quod.let", "name": "rd",
            "type": result_t,
            "init": {
                "kind": "llvm.call", "function": "std.io.read_file",
                "args": [
                    {"kind": "quod.string_ref", "name": ".path"},
                    {"kind": "quod.local_ref", "name": "a"},
                ],
            },
        },
        {
            "kind": "quod.match",
            "scrutinee": {"kind": "quod.local_ref", "name": "rd"},
            "arms": [
                {
                    "variant": "Err", "bindings": ["_e"],
                    "body": {"stmts": [
                        {
                            "kind": "quod.return_expr",
                            "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 10},
                        },
                    ]},
                },
                {
                    "variant": "Ok", "bindings": ["reader_ptr"],
                    "body": {"stmts": [
                        {
                            "kind": "quod.let", "name": "pr",
                            "type": option_jv_t,
                            "init": {
                                "kind": "llvm.call", "function": "alloc.json.io.read_value",
                                "type_args": [bufreader_file_t],
                                "args": [
                                    {"kind": "quod.local_ref", "name": "reader_ptr"},
                                    {"kind": "quod.local_ref", "name": "a"},
                                ],
                            },
                        },
                        {
                            "kind": "quod.match",
                            "scrutinee": {"kind": "quod.local_ref", "name": "pr"},
                            "arms": [
                                {
                                    "variant": "None", "bindings": [],
                                    "body": {"stmts": [
                                        {
                                            "kind": "quod.return_expr",
                                            "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 11},
                                        },
                                    ]},
                                },
                                {
                                    "variant": "Some", "bindings": ["v"],
                                    "body": {"stmts": [
                                        {
                                            "kind": "quod.match",
                                            "scrutinee": {"kind": "quod.local_ref", "name": "v"},
                                            "arms": _jsonvalue_arms(),
                                        },
                                    ]},
                                },
                            ],
                        },
                    ]},
                },
            ],
        },
    ]


def _build_and_exit(path: Path) -> int:
    prog = _make_program(str(path))
    prog = resolve_imports(prog)
    with tempfile.TemporaryDirectory() as td:
        res = compile_program(prog, build_dir=Path(td), bins=(("rt", "main"),), profile=2, link=True)
        binary = res.bins[0].binary
        out = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15)
    return out.returncode


def test_read_file_parses_number(tmp_path):
    f = tmp_path / "n.json"
    f.write_text("42")
    assert _build_and_exit(f) == 42


def test_read_file_missing_path_errs(tmp_path):
    missing = tmp_path / "does-not-exist.json"
    assert _build_and_exit(missing) == 10
