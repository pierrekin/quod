"""Schema introspection for the `quod schema` CLI / `quod_schema` tool.

Renders a condensed text description of any node kind: required/optional
fields with types, plus a hand-curated minimal example. Field info is read
from the pydantic model so types stay in sync; summaries and examples are
hand-curated next to the model.

Three query modes:
    schema()                 -> list categories + one-line summary each
    schema(category="...")   -> list kinds in the category, one-liner each
    schema(kind="quod.let")  -> full per-kind schema with example

Categories: expression, statement, type, claim, justification, program.
"""

from __future__ import annotations

import json
import types
import typing
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from quod import model
from quod.model import (
    Assign,
    BinOp,
    Call,
    CharLit,
    DerivedJustification,
    EnumDef,
    EnumInit,
    EnumPayloadField,
    EnumType,
    EnumVariant,
    ExprStmt,
    ExternFunction,
    FieldInit,
    FieldRead,
    FieldSet,
    For,
    Function,
    I1Type,
    I8PtrType,
    I8Type,
    I16Type,
    I32Type,
    I64Type,
    If,
    IntLit,
    IntRangeClaim,
    Let,
    Load,
    LocalRef,
    ManualJustification,
    Match,
    MatchArm,
    NonNegativeClaim,
    NullPtr,
    Param,
    ParamRef,
    PtrOffset,
    Return,
    ReturnExpr,
    ReturnInRangeClaim,
    ShortCircuitAnd,
    ShortCircuitOr,
    StringConstant,
    StringRef,
    StructDef,
    StructField,
    StructInit,
    StructType,
    Store,
    VoidType,
    While,
    Widen,
    WithArena,
    Z3Justification,
)


# Discriminated-union aliases. When a field is typed as one of these, render
# its alias name instead of expanding the full member list — keeps output
# terse and points the reader at the right category for further lookup.
_ALIASES = [
    (lambda: model.Expr, "Expr"),
    (lambda: model.Statement, "Statement"),
    (lambda: model.Type, "Type"),
    (lambda: model.IntType, "IntType"),
    (lambda: model.Justification, "Justification"),
    (lambda: model.Claim, "Claim"),
]


def _union_args(t: Any) -> tuple | None:
    """Return Union args if `t` is a Union/Annotated[Union]; else None."""
    if hasattr(t, "__metadata__"):
        t = t.__origin__
    origin = get_origin(t)
    if origin is typing.Union or origin is types.UnionType:
        return tuple(get_args(t))
    return None


def _matches_alias(annotation: Any) -> str | None:
    """Match `annotation` against a registered alias by union-arg set."""
    a_args = _union_args(annotation)
    if a_args is None:
        return None
    a_set = frozenset(a_args)
    for alias_fn, name in _ALIASES:
        b_args = _union_args(alias_fn())
        if b_args is not None and frozenset(b_args) == a_set:
            return name
    return None


# ---------- Catalog ----------
#
# Each entry maps a canonical name (the discriminator string for nodes that
# have one; a stable label for top-level types that don't) to:
#   class:    the Pydantic model class (used for field introspection)
#   summary:  one-line description shown in listings and at the top of
#             per-kind output
#   example:  a hand-curated minimal valid instance, rendered as JSON
#   field_descriptions: optional per-field human notes (overrides default)
#   see_also: optional cross-references to related kinds


_KIND_INFO: dict[str, dict[str, Any]] = {
    # ---------- expression ----------
    "llvm.const_int": {
        "class": IntLit,
        "summary": "Literal integer of an explicit width. The `type` field decides which iN constant is emitted.",
        "example": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 42},
    },
    "llvm.param_ref": {
        "class": ParamRef,
        "summary": "Read a function parameter.",
        "example": {"kind": "llvm.param_ref", "name": "x"},
    },
    "quod.local_ref": {
        "class": LocalRef,
        "summary": "Read a local variable previously introduced by quod.let or a quod.for loop var.",
        "example": {"kind": "quod.local_ref", "name": "i"},
        "see_also": ["quod.let", "quod.assign", "quod.for"],
    },
    "llvm.binop": {
        "class": BinOp,
        "summary": "Binary operation. Operands must agree in type; the op determines the result type.",
        "field_descriptions": {
            "op": (
                "one of: add, sub, mul, sdiv, srem (signed iN→iN); "
                "udiv (unsigned iN→iN); "
                "slt, sle, sgt, sge, eq, ne (signed cmp, iN→i1); "
                "ult, ule, ugt, uge (unsigned cmp, iN→i1); "
                "or, and (iN→iN, bitwise). "
                "Division by zero is UB. Use quod.sc_or/sc_and for short-circuit booleans."
            ),
        },
        "example": {
            "kind": "llvm.binop", "op": "add",
            "lhs": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 1},
            "rhs": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 2},
        },
        "see_also": ["quod.sc_or", "quod.sc_and"],
    },
    "quod.sc_or": {
        "class": ShortCircuitOr,
        "summary": "Short-circuit boolean OR. lhs and rhs must lower to i1.",
        "example": {
            "kind": "quod.sc_or",
            "lhs": {"kind": "llvm.binop", "op": "slt",
                    "lhs": {"kind": "llvm.param_ref", "name": "x"},
                    "rhs": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 0}},
            "rhs": {"kind": "llvm.binop", "op": "sgt",
                    "lhs": {"kind": "llvm.param_ref", "name": "x"},
                    "rhs": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 100}},
        },
        "see_also": ["llvm.binop", "quod.sc_and"],
    },
    "quod.sc_and": {
        "class": ShortCircuitAnd,
        "summary": "Short-circuit boolean AND. lhs and rhs must lower to i1.",
        "example": {
            "kind": "quod.sc_and",
            "lhs": {"kind": "llvm.binop", "op": "sge",
                    "lhs": {"kind": "llvm.param_ref", "name": "x"},
                    "rhs": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 0}},
            "rhs": {"kind": "llvm.binop", "op": "slt",
                    "lhs": {"kind": "llvm.param_ref", "name": "x"},
                    "rhs": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 100}},
        },
        "see_also": ["llvm.binop", "quod.sc_or"],
    },
    "llvm.call": {
        "class": Call,
        "summary": "Call a user function or extern by name. Result type matches the callee's return type.",
        "example": {
            "kind": "llvm.call", "function": "puts",
            "args": [{"kind": "quod.string_ref", "name": ".str.greeting"}],
        },
        "see_also": ["quod.expr_stmt", "quod.return_expr"],
    },
    "quod.string_ref": {
        "class": StringRef,
        "summary": "i8* pointer to a StringConstant declared at the program top level.",
        "example": {"kind": "quod.string_ref", "name": ".str.greeting"},
        "see_also": ["StringConstant"],
    },
    "quod.struct_init": {
        "class": StructInit,
        "summary": "Construct a struct value. Every field of the named def must be initialized exactly once.",
        "example": {
            "kind": "quod.struct_init", "type": "Point",
            "fields": [
                {"name": "x", "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 3}},
                {"name": "y", "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 4}},
            ],
        },
        "see_also": ["StructDef", "quod.field"],
    },
    "quod.field": {
        "class": FieldRead,
        "summary": "Read a named field from a struct-typed expression. Inner value must be of some StructType.",
        "example": {
            "kind": "quod.field",
            "value": {"kind": "quod.local_ref", "name": "p"},
            "name": "x",
        },
        "see_also": ["quod.struct_init", "quod.field_set"],
    },
    "quod.ptr_offset": {
        "class": PtrOffset,
        "summary": (
            "Pointer arithmetic on an i8* base: returns base + offset as i8*. "
            "Lowered to a single byte-stride GEP. Out-of-bounds offsets are UB; "
            "if you need a check, attach an int_range claim to the offset."
        ),
        "example": {
            "kind": "quod.ptr_offset",
            "base": {"kind": "quod.string_ref", "name": ".str.greeting"},
            "offset": {"kind": "llvm.const_int", "type": {"kind": "llvm.i64"}, "value": 7},
        },
        "see_also": ["quod.string_ref", "llvm.i8_ptr"],
    },
    "quod.widen": {
        "class": Widen,
        "summary": (
            "Cast an integer between widths. Narrower→wider sign-extends "
            "(or zero-extends when signed=false); wider→narrower truncates. "
            "Lowered to LLVM `sext` / `zext` / `trunc`."
        ),
        "example": {
            "kind": "quod.widen",
            "value": {"kind": "llvm.param_ref", "name": "k"},
            "target": {"kind": "llvm.i64"},
        },
        "see_also": ["quod.ptr_offset"],
    },
    "quod.load": {
        "class": Load,
        "summary": (
            "Read a value of `type` from an i8* base pointer. Lowered to a "
            "bitcast to the target pointer type plus an LLVM `load`. Compose "
            "with `quod.ptr_offset` to read at a non-zero offset."
        ),
        "example": {
            "kind": "quod.load",
            "ptr": {"kind": "quod.string_ref", "name": ".str.greeting"},
            "type": {"kind": "llvm.i8"},
        },
        "see_also": ["quod.ptr_offset", "quod.widen"],
    },
    "quod.null_ptr": {
        "class": NullPtr,
        "summary": (
            "The null i8* literal. Lowers to `i8* null`. Useful for "
            "placeholder values in `quod.struct_init` for pointer-typed "
            "fields that aren't meaningful for the active variant."
        ),
        "example": {"kind": "quod.null_ptr"},
        "see_also": ["quod.struct_init", "llvm.i8_ptr"],
    },
    "quod.char_lit": {
        "class": CharLit,
        "summary": (
            "A byte literal written as a single-character string, lowered "
            "to `const_int i8 ord(value)`. Use instead of a numeric "
            "const_int when you mean a character: `'l'` reads better than "
            "`108`. JSON's native escapes work — `\"\\n\"` is one byte."
        ),
        "example": {"kind": "quod.char_lit", "value": "n"},
        "see_also": ["llvm.const_int"],
    },
    "quod.enum_init": {
        "class": EnumInit,
        "summary": (
            "Construct an enum value by selecting a variant and "
            "initializing its payload fields. `enum` names an EnumDef, "
            "`variant` names one of its variants, and `fields` covers "
            "exactly the variant's payload fields by name."
        ),
        "example": {
            "kind": "quod.enum_init",
            "enum": "Maybe",
            "variant": "Some",
            "fields": [
                {"name": "value", "value": {"kind": "llvm.const_int",
                                            "type": {"kind": "llvm.i64"}, "value": 42}},
            ],
        },
        "see_also": ["EnumDef", "quod.match"],
    },

    # ---------- statement ----------
    "quod.return_expr": {
        "class": ReturnExpr,
        "summary": "Return the value of an expression. The expression's type must match the function's return_type.",
        "example": {
            "kind": "quod.return_expr",
            "value": {"kind": "llvm.binop", "op": "add",
                      "lhs": {"kind": "llvm.param_ref", "name": "x"},
                      "rhs": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 1}},
        },
    },
    "quod.return": {
        "class": Return,
        "summary": (
            "Bare return for void functions. The enclosing function's "
            "return_type must be llvm.void; non-void functions must use "
            "return_expr. Void functions also get an implicit ret void at "
            "the end if the body falls through."
        ),
        "example": {"kind": "quod.return"},
        "see_also": ["llvm.void", "quod.return_expr"],
    },
    "quod.if": {
        "class": If,
        "summary": "Two-branch conditional. cond must lower to i1. Both branches are required (use [] for an empty branch).",
        "example": {
            "kind": "quod.if",
            "cond": {"kind": "llvm.binop", "op": "slt",
                     "lhs": {"kind": "llvm.param_ref", "name": "x"},
                     "rhs": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 0}},
            "then_body": [{"kind": "quod.return_expr",
                           "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": -1}}],
            "else_body": [{"kind": "quod.return_expr",
                           "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 1}}],
        },
    },
    "quod.let": {
        "class": Let,
        "summary": "Introduce a mutable local. Lowered to alloca-at-entry + store. Field is `init`, NOT `value`. The init's type must match `type`.",
        "example": {
            "kind": "quod.let", "name": "sum",
            "type": {"kind": "llvm.i32"},
            "init": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 0},
        },
        "see_also": ["quod.assign", "quod.local_ref"],
    },
    "quod.assign": {
        "class": Assign,
        "summary": "Mutate an existing local. The local must have been declared by quod.let or be a quod.for loop var. The value's type must match the local's declared type.",
        "example": {
            "kind": "quod.assign", "name": "sum",
            "value": {"kind": "llvm.binop", "op": "add",
                      "lhs": {"kind": "quod.local_ref", "name": "sum"},
                      "rhs": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 1}},
        },
        "see_also": ["quod.let", "quod.local_ref"],
    },
    "quod.while": {
        "class": While,
        "summary": "Pre-test loop. cond evaluated each iteration; body runs while true.",
        "example": {
            "kind": "quod.while",
            "cond": {"kind": "llvm.binop", "op": "slt",
                     "lhs": {"kind": "quod.local_ref", "name": "i"},
                     "rhs": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 10}},
            "body": [{"kind": "quod.assign", "name": "i",
                      "value": {"kind": "llvm.binop", "op": "add",
                                "lhs": {"kind": "quod.local_ref", "name": "i"},
                                "rhs": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 1}}}],
        },
        "see_also": ["quod.for"],
    },
    "quod.for": {
        "class": For,
        "summary": "Bounded iteration: var (of type `type`) runs from lo (inclusive) to hi (exclusive). lo/hi must match `type`. Bounds evaluated once (snapshot).",
        "example": {
            "kind": "quod.for", "var": "i", "type": {"kind": "llvm.i32"},
            "lo": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 0},
            "hi": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 10},
            "body": [{"kind": "quod.expr_stmt",
                      "value": {"kind": "llvm.call", "function": "putchar",
                                "args": [{"kind": "quod.local_ref", "name": "i"}]}}],
        },
        "see_also": ["quod.while"],
    },
    "quod.expr_stmt": {
        "class": ExprStmt,
        "summary": "Evaluate an expression for side effects, discard the result. Natural shape for printf and other void calls.",
        "example": {
            "kind": "quod.expr_stmt",
            "value": {"kind": "llvm.call", "function": "puts",
                      "args": [{"kind": "quod.string_ref", "name": ".str.greeting"}]},
        },
    },
    "quod.field_set": {
        "class": FieldSet,
        "summary": "Mutate one field of a struct-typed local. Same scoping as quod.assign — `local` must be a Let-introduced struct local.",
        "example": {
            "kind": "quod.field_set", "local": "p", "name": "y",
            "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 7},
        },
        "see_also": ["quod.struct_init", "quod.field"],
    },
    "quod.store": {
        "class": Store,
        "summary": (
            "Write `value` to memory at the i8* pointer `ptr`. Lowered to a "
            "bitcast + LLVM `store`. Pair with `quod.ptr_offset` for non-zero "
            "offsets and with `quod.load` for round-trips through arena memory."
        ),
        "example": {
            "kind": "quod.store",
            "ptr": {"kind": "quod.local_ref", "name": "buf"},
            "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i8"}, "value": 65},
        },
        "see_also": ["quod.load", "quod.ptr_offset"],
    },
    "quod.with_arena": {
        "class": WithArena,
        "summary": (
            "Open a bump-allocated arena for the duration of `body`; the runtime's "
            "`quod_arena_drop` is called automatically on every exit edge "
            "(fall-through and every `return` reachable from the body). The arena "
            "handle is bound to a local named `name` of type i8*. `capacity` must "
            "lower to i64. Auto-declares `quod_arena_new` / `quod_arena_drop` "
            "externs if absent — declare `quod_arena_alloc` yourself when you call it."
        ),
        "example": {
            "kind": "quod.with_arena", "name": "a",
            "capacity": {"kind": "llvm.const_int", "type": {"kind": "llvm.i64"}, "value": 4096},
            "body": [
                {"kind": "quod.expr_stmt",
                 "value": {"kind": "llvm.call", "function": "quod_arena_alloc",
                           "args": [{"kind": "quod.local_ref", "name": "a"},
                                    {"kind": "llvm.const_int", "type": {"kind": "llvm.i64"}, "value": 64}]}},
            ],
        },
        "see_also": ["quod.let", "quod.expr_stmt"],
    },

    # ---------- type ----------
    "llvm.i1": {
        "class": I1Type,
        "summary": "1-bit integer. Boolean values: cmp results, short-circuit results, explicit booleans.",
        "example": {"kind": "llvm.i1"},
    },
    "llvm.i8": {
        "class": I8Type,
        "summary": "8-bit integer. Byte-sized values; commonly used with truncation from wider widths.",
        "example": {"kind": "llvm.i8"},
    },
    "llvm.i16": {
        "class": I16Type,
        "summary": "16-bit integer.",
        "example": {"kind": "llvm.i16"},
    },
    "llvm.i32": {
        "class": I32Type,
        "summary": "32-bit integer. The conventional 'int' for most quod programs.",
        "example": {"kind": "llvm.i32"},
    },
    "llvm.i64": {
        "class": I64Type,
        "summary": "64-bit integer. Wide values; the type the argv wrapper parses argv slots into.",
        "example": {"kind": "llvm.i64"},
    },
    "llvm.i8_ptr": {
        "class": I8PtrType,
        "summary": "Pointer to i8. Used for C-style strings (via quod.string_ref) and opaque buffers.",
        "example": {"kind": "llvm.i8_ptr"},
    },
    "llvm.struct": {
        "class": StructType,
        "summary": "Reference to a named StructDef by name. Pass-by-value at the LLVM level.",
        "example": {"kind": "llvm.struct", "name": "Point"},
        "see_also": ["StructDef"],
    },
    "llvm.enum": {
        "class": EnumType,
        "summary": (
            "Reference to a named EnumDef by name. Lowered as an LLVM "
            "identified struct `{i8 tag, [N x i64] payload}` where each "
            "variant's payload fields share the same i64-slot array."
        ),
        "example": {"kind": "llvm.enum", "name": "Maybe"},
        "see_also": ["EnumDef", "quod.enum_init", "quod.match"],
    },
    "llvm.void": {
        "class": VoidType,
        "summary": (
            "The LLVM void type. Only valid as a function's return_type. "
            "Functions returning void use bare quod.return statements (no "
            "value) and may not appear in any value position."
        ),
        "example": {"kind": "llvm.void"},
        "see_also": ["quod.return"],
    },

    # ---------- claim ----------
    "non_negative": {
        "class": NonNegativeClaim,
        "summary": "Asserts param >= 0. Subsumed by int_range(min=0); kept as a convenience.",
        "example": {"kind": "non_negative", "param": "x"},
    },
    "int_range": {
        "class": IntRangeClaim,
        "summary": "Asserts min <= param <= max. Either bound optional (omit for unbounded on that side).",
        "example": {"kind": "int_range", "param": "x", "min": 0, "max": 100},
    },
    "return_in_range": {
        "class": ReturnInRangeClaim,
        "summary": "Asserts the function's return value is in [min, max]. Function-scoped — no `param` field.",
        "example": {"kind": "return_in_range", "min": -1},
    },

    # ---------- justification (evidence on a claim) ----------
    "z3": {
        "class": Z3Justification,
        "summary": "External proof in SMT-LIB; verifiable by re-running Z3. Auto-attached by `quod claim prove`.",
        "example": {"kind": "z3", "artifact_path": "proofs/f_x.smt2",
                    "artifact_hash": "abc123..."},
    },
    "manual": {
        "class": ManualJustification,
        "summary": "Human signoff. signed_by must be non-empty.",
        "example": {"kind": "manual", "signed_by": "alice", "rationale": "by induction on n"},
    },
    "derived": {
        "class": DerivedJustification,
        "summary": "Re-derived from the program graph each compile (lattice analysis). Skipped by `quod claim verify`.",
        "example": {"kind": "derived"},
    },

    # ---------- program-level (no kind discriminator) ----------
    "StringConstant": {
        "class": StringConstant,
        "summary": "A null-terminated byte string declared at program top level. Reference via quod.string_ref.",
        "example": {"name": ".str.greeting", "value": "hello, world"},
    },
    "ExternFunction": {
        "class": ExternFunction,
        "summary": "A libc-or-similar function declared but not defined here. Use `arity` for all-i32 sigs, or `param_types` for typed.",
        "example": {
            "name": "printf",
            "param_types": [{"kind": "llvm.i8_ptr"}],
            "varargs": True,
        },
    },
    "Function": {
        "class": Function,
        "summary": "A user function. params is a list of typed Params; return_type is required; body is a list of statements; claims optional. Entry-point functions may declare params; the synthesized main wrapper parses each argv slot via atoll then trunc/sext's to the param's type (so `quod run -- 42 7` calls entry(42, 7)). An entry called 'main' must be nullary — rename it if you want params.",
        "example": {
            "name": "main", "params": [],
            "return_type": {"kind": "llvm.i32"},
            "body": [{"kind": "quod.return_expr",
                      "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 0}}],
        },
    },
    "Param": {
        "class": Param,
        "summary": "A typed function parameter. `type` is any Type (int width, i8_ptr, or named StructType). The argv main wrapper still requires int-only params on the entry function.",
        "example": {"name": "x", "type": {"kind": "llvm.i32"}},
    },
    "StructDef": {
        "class": StructDef,
        "summary": "A named record type with ordered, uniquely-named fields. Lowered to an LLVM identified struct type, passed and returned by value.",
        "example": {
            "name": "Point",
            "fields": [
                {"name": "x", "type": {"kind": "llvm.i32"}},
                {"name": "y", "type": {"kind": "llvm.i32"}},
            ],
        },
        "see_also": ["llvm.struct", "quod.struct_init", "quod.field", "quod.field_set"],
    },
    "StructField": {
        "class": StructField,
        "summary": "One field in a StructDef. Field types may be any Type, including other named structs (no recursion).",
        "example": {"name": "x", "type": {"kind": "llvm.i32"}},
    },
    "FieldInit": {
        "class": FieldInit,
        "summary": "One field's value in a quod.struct_init or quod.enum_init.",
        "example": {"name": "x", "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 3}},
    },
    "EnumDef": {
        "class": EnumDef,
        "summary": (
            "A named tagged-union type. Variants are ordered (first variant "
            "gets discriminant 0). Lowered to `{i8 tag, [N x i64] payload}` "
            "where N = max variant payload field count."
        ),
        "example": {
            "name": "Maybe",
            "variants": [
                {"name": "None", "fields": []},
                {"name": "Some", "fields": [{"name": "value", "type": {"kind": "llvm.i64"}}]},
            ],
        },
        "see_also": ["llvm.enum", "quod.enum_init", "quod.match"],
    },
    "EnumVariant": {
        "class": EnumVariant,
        "summary": "One variant of an EnumDef. Empty fields means a unit variant.",
        "example": {"name": "Some", "fields": [{"name": "value", "type": {"kind": "llvm.i64"}}]},
    },
    "EnumPayloadField": {
        "class": EnumPayloadField,
        "summary": (
            "One payload field of an EnumVariant. Restricted to scalar "
            "types (int widths up to i64, plus i8*) so each field fits "
            "in a single i64 slot."
        ),
        "example": {"name": "value", "type": {"kind": "llvm.i64"}},
    },
    "MatchArm": {
        "class": MatchArm,
        "summary": (
            "One arm of a quod.match. Names a variant, binds its payload "
            "fields to locals (one binding name per field, in declaration "
            "order), and runs `body`."
        ),
        "example": {
            "variant": "Some",
            "bindings": ["v"],
            "body": [{"kind": "quod.return_expr",
                      "value": {"kind": "quod.local_ref", "name": "v"}}],
        },
        "see_also": ["quod.match"],
    },
}


_KIND_INFO["quod.match"] = {
    "class": Match,
    "summary": (
        "Pattern-match on an enum value. One arm per variant, exhaustive. "
        "scrutinee must lower to a value of an EnumType. Lowered to a "
        "switch on the discriminant byte."
    ),
    "example": {
        "kind": "quod.match",
        "scrutinee": {"kind": "quod.local_ref", "name": "m"},
        "arms": [
            {"variant": "None", "bindings": [], "body": [
                {"kind": "quod.return_expr",
                 "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i64"}, "value": 0}}]},
            {"variant": "Some", "bindings": ["v"], "body": [
                {"kind": "quod.return_expr",
                 "value": {"kind": "quod.local_ref", "name": "v"}}]},
        ],
    },
    "see_also": ["EnumDef", "quod.enum_init"],
}


_CATEGORIES: dict[str, list[str]] = {
    "expression": [
        "llvm.const_int", "llvm.param_ref", "quod.local_ref", "llvm.binop",
        "quod.sc_or", "quod.sc_and", "llvm.call", "quod.string_ref",
        "quod.struct_init", "quod.field", "quod.ptr_offset", "quod.widen",
        "quod.load", "quod.null_ptr", "quod.char_lit", "quod.enum_init",
    ],
    "statement": [
        "quod.return_expr", "quod.return", "quod.if",
        "quod.let", "quod.assign", "quod.while", "quod.for", "quod.expr_stmt",
        "quod.field_set", "quod.store", "quod.with_arena", "quod.match",
    ],
    "type": [
        "llvm.i1", "llvm.i8", "llvm.i16", "llvm.i32", "llvm.i64",
        "llvm.i8_ptr", "llvm.struct", "llvm.enum", "llvm.void",
    ],
    "claim": ["non_negative", "int_range", "return_in_range"],
    "justification": ["z3", "manual", "derived"],
    "program": [
        "StringConstant", "ExternFunction", "Function", "Param",
        "StructDef", "StructField", "FieldInit",
        "EnumDef", "EnumVariant", "EnumPayloadField", "MatchArm",
    ],
}


# ---------- Type rendering ----------

def _render_type(annotation: Any) -> str:
    """Render a Python type annotation as a short human label."""
    # Recognized discriminated-union aliases — render as alias name and stop.
    alias = _matches_alias(annotation)
    if alias is not None:
        return alias

    # Strip Annotated[...] metadata (e.g. `Annotated[Union[...], Field(discriminator=...)]`).
    if hasattr(annotation, "__metadata__"):
        annotation = annotation.__origin__

    # Forward refs ("Expr" string annotations not yet resolved): use the name.
    if isinstance(annotation, typing.ForwardRef):
        # Pydantic sometimes stores a doubly-quoted name — strip stray quotes.
        return annotation.__forward_arg__.strip("'\"")

    if annotation is type(None):
        return "null"

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Union / Optional. typing.Union and `X | Y` (PEP 604 / types.UnionType) both apply.
    if origin is typing.Union or origin is types.UnionType:
        has_none = type(None) in args
        non_none = [a for a in args if a is not type(None)]
        rendered = [_render_type(a) for a in non_none]
        joined = " | ".join(rendered)
        return f"{joined}?" if has_none else joined

    # Literal["a", "b", ...] — render as `'a' | 'b' | ...` for closed-set fields.
    if origin is typing.Literal:
        return " | ".join(repr(a) for a in args)

    # tuple[X, ...] is the canonical container shape we use throughout.
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return f"list[{_render_type(args[0])}]"
        return f"tuple[{', '.join(_render_type(a) for a in args)}]"
    if origin is list:
        return f"list[{_render_type(args[0])}]" if args else "list"

    # Plain types.
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _render_default(default: Any) -> str:
    if default is None:
        return "null"
    if default == ():
        return "[]"
    return repr(default)


# ---------- Render functions ----------

def _resolve_name(name: str) -> str:
    """Accept canonical kinds, also aliases like 'function' → 'Function'."""
    if name in _KIND_INFO:
        return name
    # Try case-insensitive match for top-level types.
    for k in _KIND_INFO:
        if k.lower() == name.lower():
            return k
    return name  # caller decides how to error


def _resolved_hints(cls: type[BaseModel]) -> dict[str, Any]:
    """Resolve forward refs in `cls`'s annotations against `quod.model`'s globals.

    Pydantic stores raw annotations (often ForwardRefs containing strings like
    `tuple['Statement', ...]`); typing.get_type_hints walks the type and resolves
    those refs by looking up names in the provided globals.
    """
    return typing.get_type_hints(
        cls, globalns=vars(model), include_extras=True,
    )


def render_kind(name: str) -> str:
    """Render a single kind's schema as a condensed text block."""
    name = _resolve_name(name)
    if name not in _KIND_INFO:
        known = ", ".join(sorted(_KIND_INFO.keys()))
        raise KeyError(f"unknown kind {name!r}. Known kinds:\n  {known}")
    info = _KIND_INFO[name]
    cls: type[BaseModel] = info["class"]
    cat = _category_of(name) or "?"
    lines = [f"{name} ({cat}) — {info['summary']}"]
    field_descriptions = info.get("field_descriptions", {})
    hints = _resolved_hints(cls)
    for fname, finfo in cls.model_fields.items():
        if fname == "kind":
            continue
        annotation = hints.get(fname, finfo.annotation)
        ty = _render_type(annotation)
        if finfo.is_required():
            req = "required"
        else:
            req = f"optional, default={_render_default(finfo.default)}"
        desc = field_descriptions.get(fname, "")
        suffix = f" — {desc}" if desc else ""
        lines.append(f"  {fname} ({ty}, {req}){suffix}")
    lines.append("example:")
    lines.append(f"  {json.dumps(info['example'], separators=(', ', ': '))}")
    if info.get("see_also"):
        lines.append(f"see also: {', '.join(info['see_also'])}")
    return "\n".join(lines)


def render_category(cat: str) -> str:
    """Render all kinds in a category as one-liners."""
    if cat not in _CATEGORIES:
        known = ", ".join(_CATEGORIES.keys())
        raise KeyError(f"unknown category {cat!r}. Known categories: {known}")
    lines = [f"category: {cat}"]
    for name in _CATEGORIES[cat]:
        info = _KIND_INFO[name]
        lines.append(f"  {name} — {info['summary']}")
    lines.append(f"\nFor full schema of one kind: quod schema KIND")
    return "\n".join(lines)


def render_categories() -> str:
    """Render all categories with kind counts."""
    lines = ["categories:"]
    for cat, kinds in _CATEGORIES.items():
        lines.append(f"  {cat} ({len(kinds)} kinds): {', '.join(kinds)}")
    lines.append("\nFor a category overview: quod schema --category CAT")
    lines.append("For a kind's full schema: quod schema KIND")
    return "\n".join(lines)


def _category_of(name: str) -> str | None:
    for cat, kinds in _CATEGORIES.items():
        if name in kinds:
            return cat
    return None
