"""Smoke test for `std.io.open_write`.

Builds a quod program that opens a tempfile via open_write (which calls
the variadic libc `open` extern with O_WRONLY|O_CREAT|O_TRUNC + mode
0644), writes a fixed payload through the File Reader/Writer trait, and
closes the fd. Pytest asserts the file content matches.

Exit codes pin which step broke:
  10 — open_write returned Err
  11 — Writer.write returned Err
   0 — wrote payload, closed fd
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


PAYLOAD = "v1\n"
ARENA = {"kind": "llvm.struct", "name": "mem.arena.Arena"}


def _wired(module: str) -> dict:
    return {"module": module, "wire": [{"name": "A", "type": ARENA}]}


def _make_program(path: str) -> Program:
    return Program.model_validate_json(json.dumps({
        "imports": [_wired("std.io")],
        "constants": [
            {"name": ".path", "value": path},
            {"name": ".payload", "value": PAYLOAD},
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
                        "capacity": {"kind": "llvm.const_int", "type": {"kind": "llvm.i64"}, "value": 4096},
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


def _body() -> list[dict]:
    result_file_t = {
        "kind": "llvm.enum", "name": "core.result.Result",
        "type_args": [
            {"kind": "llvm.struct", "name": "std.io.File"},
            {"kind": "llvm.enum", "name": "core.io.IoError"},
        ],
    }
    result_usize_t = {
        "kind": "llvm.enum", "name": "core.result.Result",
        "type_args": [
            {"kind": "llvm.usize"},
            {"kind": "llvm.enum", "name": "core.io.IoError"},
        ],
    }

    return [
        {
            "kind": "quod.let", "name": "rd",
            "type": result_file_t,
            "init": {
                "kind": "llvm.call", "function": "std.io.open_write",
                "args": [{"kind": "quod.string_ref", "name": ".path"}],
            },
        },
        {
            "kind": "quod.match",
            "scrutinee": {"kind": "quod.local_ref", "name": "rd"},
            "arms": [
                {"variant": "Err", "bindings": ["_e"], "body": {"stmts": [_ret(10)]}},
                {
                    "variant": "Ok", "bindings": ["file"],
                    "body": {"stmts": [
                        {
                            "kind": "quod.let", "name": "file_box",
                            "type": {"kind": "llvm.i8_ptr"},
                            "init": {
                                "kind": "llvm.call", "function": "mem.arena.alloc",
                                "args": [
                                    {"kind": "quod.local_ref", "name": "a"},
                                    {"kind": "quod.sizeof", "type": {"kind": "llvm.struct", "name": "std.io.File"}},
                                ],
                            },
                        },
                        {
                            "kind": "quod.store",
                            "ptr": {"kind": "quod.local_ref", "name": "file_box"},
                            "value": {"kind": "quod.local_ref", "name": "file"},
                        },
                        {
                            "kind": "quod.let", "name": "wr",
                            "type": result_usize_t,
                            "init": {
                                "kind": "quod.trait_call",
                                "trait": "core.io.Writer", "method": "write",
                                "dispatch_type": {"kind": "llvm.struct", "name": "std.io.File"},
                                "args": [
                                    {"kind": "quod.local_ref", "name": "file_box"},
                                    {"kind": "quod.string_ref", "name": ".payload"},
                                    {"kind": "llvm.const_int", "type": {"kind": "llvm.usize"}, "value": len(PAYLOAD)},
                                ],
                            },
                        },
                        {
                            "kind": "quod.match",
                            "scrutinee": {"kind": "quod.local_ref", "name": "wr"},
                            "arms": [
                                {"variant": "Err", "bindings": ["_we"], "body": {"stmts": [_ret(11)]}},
                                {
                                    "variant": "Ok", "bindings": ["_n"],
                                    "body": {"stmts": [
                                        {
                                            "kind": "quod.expr_stmt",
                                            "value": {
                                                "kind": "llvm.call", "function": "std.io.file_close",
                                                "args": [{"kind": "quod.local_ref", "name": "file"}],
                                            },
                                        },
                                        _ret(0),
                                    ]},
                                },
                            ],
                        },
                    ]},
                },
            ],
        },
    ]


def _build_and_run(prog: Program) -> int:
    prog = resolve_imports(prog)
    with tempfile.TemporaryDirectory() as td:
        res = compile_program(prog, build_dir=Path(td), bins=(("rt", "main"),), profile=2, link=True)
        binary = res.bins[0].binary
        out = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15)
    return out.returncode


def test_open_write_roundtrip(tmp_path):
    f = tmp_path / "out.txt"
    rc = _build_and_run(_make_program(str(f)))
    assert rc == 0, f"binary exited {rc}"
    assert f.read_bytes() == PAYLOAD.encode(), f"file content mismatch: {f.read_bytes()!r}"
