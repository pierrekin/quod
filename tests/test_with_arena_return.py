"""Regression test for the with_arena drop-vs-return ordering fix.

Returning a value from inside `with_arena { ... }` where the value
reads memory backed by the arena used to read dangling memory at
profile=0 (and was hidden by the optimizer at profile=2). The fix
hoists the return-expr value into a Let *before* the drop, so the
read happens while the arena is still live.

This test compiles at profile=0 explicitly so a regression would
manifest as a wrong return value — not a test that happens to pass
because the optimizer hoisted the load above the drop.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from quod.lower import compile_program
from quod.model import Program


def _build_and_run(prog: Program, *, profile: int) -> int:
    with tempfile.TemporaryDirectory() as td:
        res = compile_program(
            prog, build_dir=Path(td),
            bins=(("rt", "main"),), profile=profile, link=True,
        )
        out = subprocess.run([str(res.bins[0].binary)], capture_output=True, text=True, timeout=15)
    return out.returncode


def _arena_return_program() -> Program:
    """main() opens an arena, allocs 4 bytes, stores 42 into them,
    returns load(buf). The naïve desugaring would `drop; ret load(buf)`
    which reads from freed memory; the correct one hoists the load
    into a Let before drop."""
    return Program.model_validate_json(json.dumps({
        "imports": ["mem.arena"],
        "functions": [{
            "name": "main", "params": [],
            "return_type": {"kind": "llvm.i32"},
            "body": {"stmts": [
                {"kind": "quod.with_arena", "name": "a",
                 "capacity": {"kind": "llvm.const_int", "type": {"kind": "llvm.i64"}, "value": 4096},
                 "body": {"stmts": [
                    {"kind": "quod.let", "name": "buf", "type": {"kind": "llvm.i8_ptr"},
                     "init": {"kind": "llvm.call", "function": "mem.arena.alloc",
                              "args": [
                                  {"kind": "quod.local_ref", "name": "a"},
                                  {"kind": "llvm.const_int", "type": {"kind": "llvm.i64"}, "value": 4},
                              ]}},
                    {"kind": "quod.store",
                     "ptr": {"kind": "quod.local_ref", "name": "buf"},
                     "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 42}},
                    {"kind": "quod.return_expr",
                     "value": {"kind": "quod.load",
                               "ptr": {"kind": "quod.local_ref", "name": "buf"},
                               "type": {"kind": "llvm.i32"}}},
                 ]}},
                {"kind": "quod.return_expr",
                 "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 99}},
            ]
        }}]
    }))


def test_return_from_with_arena_unoptimized_reads_correct_value():
    prog = _arena_return_program()
    assert _build_and_run(prog, profile=0) == 42, (
        "return-from-inside-with_arena read freed memory at profile=0 — "
        "the drop is happening before the return-expr value is evaluated"
    )


def test_return_from_with_arena_optimized_still_correct():
    prog = _arena_return_program()
    assert _build_and_run(prog, profile=2) == 42
