"""Layer-B kind catalog: c.* family-extension nodes.
"""

from __future__ import annotations

from typing import Any

from quod.model import (
    CScopedBlock,
    CStyleFor,
)


_LAYER_B_CATALOG: dict[str, dict[str, Any]] = {
    "c.scoped_block": {
        "class": CScopedBlock,
        "summary": (
            "C-style block wrapper. `block` is the inner core.Block that "
            "edges anchor on; the wrapper carries family-specific scope "
            "semantics (which decls die at the closing brace). Lowered by "
            "c-family lowering to its inner block — `quod.lower` refuses "
            "to see this wrapper."
        ),
        "example": {
            "kind": "c.scoped_block",
            "block": {"id": "@blk_for_body", "stmts": []},
            "scope_locals": ["i"],
        },
    },

    "c.for_general": {
        "class": CStyleFor,
        "summary": (
            "C-style for loop with arbitrary init/cond/inc — the layer-B "
            "transcription of `c.for` from layer A. Lowered to "
            "`Let + While + Assign` by c-family lowering, with the rule "
            "cited as `FamilyLowering(\"c.for_general\")` in the resulting "
            "equivalence claim. `quod.lower` refuses to consume this — the "
            "c-family lowering pass must run first."
        ),
        "example": {
            "kind": "c.for_general",
            "init": {"kind": "quod.let", "name": "i",
                     "type": {"kind": "llvm.i32"},
                     "init": {"kind": "llvm.const_int",
                              "type": {"kind": "llvm.i32"}, "value": 0}},
            "cond": {"kind": "llvm.binop", "op": "lt",
                     "lhs": {"kind": "quod.local_ref", "name": "i"},
                     "rhs": {"kind": "llvm.param_ref", "name": "n"}},
            "inc": {"kind": "quod.assign", "name": "i",
                    "value": {"kind": "llvm.binop", "op": "add",
                              "lhs": {"kind": "quod.local_ref", "name": "i"},
                              "rhs": {"kind": "llvm.const_int",
                                      "type": {"kind": "llvm.i32"}, "value": 1}}},
            "body": {"stmts": []},
        },
    },

}