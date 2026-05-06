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
    CAddressOf,
    CArraySubscript,
    CAssign,
    CBinOp,
    CCall,
    CCompoundAssign,
    CEnumConstRef,
    CExprStmt,
    CFn,
    CFor,
    CIf,
    CIntLit,
    CMultiVarDecl,
    CNamedType,
    CParam,
    CPointerType,
    CReturn,
    CScopedBlock,
    CStringLit,
    CStyleFor,
    CUnary,
    CUnit,
    CVarDecl,
    CVarRef,
    CWhile,
    DerivedJustification,
    Equivalence,
    EnumDef,
    EnumInit,
    EnumPayloadField,
    EnumType,
    EnumVariant,
    ExprStmt,
    ExternFunction,
    LibcLinkage,
    RuntimeLinkage,
    FieldInit,
    FieldRead,
    FieldSet,
    Block,
    For,
    Function,
    I1Type,
    I8PtrType,
    I8Type,
    I16Type,
    I32Type,
    I64Type,
    U8Type,
    U16Type,
    U32Type,
    U64Type,
    IsizeType,
    UsizeType,
    If,
    IntLit,
    IntRangeClaim,
    Let,
    Load,
    LoadField,
    LocalRef,
    FamilyLowering,
    LiftEquivalence,
    ManualJustification,
    Match,
    MatchArm,
    SizeOf,
    TryExpr,
    NonNegativeClaim,
    NullPtr,
    Param,
    ParamRef,
    ProvenanceEdge,
    PtrOffset,
    Return,
    ReturnExpr,
    ReturnInRangeClaim,
    Unreachable,
    ShortCircuitAnd,
    ShortCircuitOr,
    StringConstant,
    StringRef,
    StructDef,
    StructField,
    StructInit,
    StructType,
    Store,
    StoreField,
    SelfType,
    TraitCall,
    TraitDef,
    TraitMethodSig,
    ImplDef,
    TypeParam,
    TypeParamRef,
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
    "quod.load_field": {
        "class": LoadField,
        "summary": (
            "Read one named field of a struct stored at an i8* pointer. "
            "Lowered to bitcast(ptr, T*) + GEP(field index) + load. "
            "Targeted access — no whole-struct register copy. Use this "
            "for struct-on-heap field reads. `type_args` populates type "
            "parameters when struct_type names a generic StructDef; the "
            "monomorphizer mangles struct_type using the args."
        ),
        "example": {
            "kind": "quod.load_field",
            "ptr": {"kind": "quod.local_ref", "name": "arena"},
            "struct_type": "Arena",
            "name": "head",
        },
        "see_also": ["quod.store_field", "quod.field", "quod.load"],
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
    "quod.try": {
        "class": TryExpr,
        "summary": (
            "Postfix `?` propagation. `value` must produce a value of "
            "a 2-variant enum where one variant has a single payload "
            "field (the happy variant) and the other has no payload "
            "(the sad variant). On the sad value, the enclosing "
            "function returns the same sad variant immediately. On the "
            "happy value, evaluates to the payload field. Variant "
            "names don't matter — Ok/Err, Some/None, Found/Missing all "
            "qualify by shape. Function return type must be the same "
            "enum (no cross-enum conversion)."
        ),
        "example": {"kind": "quod.try", "value": {"kind": "llvm.call", "function": "alloc.json.parse", "args": []}},
        "see_also": ["EnumDef", "quod.match"],
    },
    "quod.sizeof": {
        "class": SizeOf,
        "summary": (
            "Size in bytes of a quod type, computed at lower time. Returns "
            "i64. Use for stride-correct pointer arithmetic over arena-"
            "allocated arrays of structs or enums."
        ),
        "example": {"kind": "quod.sizeof", "type": {"kind": "llvm.enum", "name": "JsonValue"}},
        "see_also": ["quod.ptr_offset"],
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
    "quod.unreachable": {
        "class": Unreachable,
        "summary": (
            "A statement that must not be executed at runtime. Lowers to "
            "LLVM `unreachable`. Used to terminate a block when the source "
            "language's semantics for reaching this point are undefined — "
            "e.g. the C ingest emits this for fall-through off a non-`main` "
            "int-returning function (UB per C99 §6.9.1/12), so analysis can "
            "flag the path rather than fabricating a return value."
        ),
        "example": {"kind": "quod.unreachable"},
        "see_also": ["quod.return_expr"],
    },
    "quod.if": {
        "class": If,
        "summary": "Two-branch conditional. cond must lower to i1. Both branches are required (empty branch is `{\"stmts\": []}`).",
        "example": {
            "kind": "quod.if",
            "cond": {"kind": "llvm.binop", "op": "slt",
                     "lhs": {"kind": "llvm.param_ref", "name": "x"},
                     "rhs": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 0}},
            "then_body": {"stmts": [{"kind": "quod.return_expr",
                                     "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": -1}}]},
            "else_body": {"stmts": [{"kind": "quod.return_expr",
                                     "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 1}}]},
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
            "body": {"stmts": [{"kind": "quod.assign", "name": "i",
                                "value": {"kind": "llvm.binop", "op": "add",
                                          "lhs": {"kind": "quod.local_ref", "name": "i"},
                                          "rhs": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 1}}}]},
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
            "body": {"stmts": [{"kind": "quod.expr_stmt",
                                "value": {"kind": "llvm.call", "function": "putchar",
                                          "args": [{"kind": "quod.local_ref", "name": "i"}]}}]},
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
    "quod.store_field": {
        "class": StoreField,
        "summary": (
            "Write a value into one named field of a struct stored at an "
            "i8* pointer. Mutating counterpart of quod.load_field; lowered "
            "to bitcast(ptr, T*) + GEP(field index) + store. Use for "
            "struct-on-heap field writes. `type_args` populates type "
            "parameters when struct_type names a generic StructDef; the "
            "monomorphizer mangles struct_type using the args."
        ),
        "example": {
            "kind": "quod.store_field",
            "ptr": {"kind": "quod.local_ref", "name": "arena"},
            "struct_type": "Arena",
            "name": "head",
            "value": {"kind": "quod.local_ref", "name": "new_chunk"},
        },
        "see_also": ["quod.load_field", "quod.field_set", "quod.store"],
    },
    "quod.with_arena": {
        "class": WithArena,
        "summary": (
            "Open a bump-allocated arena for the duration of `body`; "
            "`mem.arena.drop` is called automatically on every exit edge "
            "(fall-through and every `return` reachable from the body). The arena "
            "handle is bound to a local named `name` of type i8*. `capacity` must "
            "lower to i64. Auto-injects `imports: [\"mem.arena\"]` if absent, "
            "so `mem.arena.alloc` is visible to body code without a manual import."
        ),
        "example": {
            "kind": "quod.with_arena", "name": "a",
            "capacity": {"kind": "llvm.const_int", "type": {"kind": "llvm.i64"}, "value": 4096},
            "body": {"stmts": [
                {"kind": "quod.expr_stmt",
                 "value": {"kind": "llvm.call", "function": "mem.arena.alloc",
                           "args": [{"kind": "quod.local_ref", "name": "a"},
                                    {"kind": "llvm.const_int", "type": {"kind": "llvm.i64"}, "value": 64}]}},
            ]},
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
    "llvm.u8": {
        "class": U8Type,
        "summary": "8-bit unsigned integer. Same width as i8; signedness lives on ops (udiv/urem/ult/...).",
        "example": {"kind": "llvm.u8"},
    },
    "llvm.u16": {
        "class": U16Type,
        "summary": "16-bit unsigned integer.",
        "example": {"kind": "llvm.u16"},
    },
    "llvm.u32": {
        "class": U32Type,
        "summary": "32-bit unsigned integer.",
        "example": {"kind": "llvm.u32"},
    },
    "llvm.u64": {
        "class": U64Type,
        "summary": "64-bit unsigned integer. Holds values beyond i64's positive range up to 2^64-1.",
        "example": {"kind": "llvm.u64"},
    },
    "llvm.isize": {
        "class": IsizeType,
        "summary": "Pointer-sized signed integer. Lowers to i64 on 64-bit targets; nominal alias for size/offset values.",
        "example": {"kind": "llvm.isize"},
    },
    "llvm.usize": {
        "class": UsizeType,
        "summary": "Pointer-sized unsigned integer. Lowers to i64 width on 64-bit targets; the conventional type for sizes, lengths, and indices.",
        "example": {"kind": "llvm.usize"},
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
    "quod.type_param": {
        "class": TypeParamRef,
        "summary": (
            "Reference to an in-scope type parameter (e.g. T inside a generic "
            "struct/enum). Substituted for a concrete Type by the "
            "monomorphization pass before lowering — never reaches the "
            "lowerer."
        ),
        "example": {"kind": "quod.type_param", "name": "T"},
        "see_also": ["StructDef", "EnumDef"],
    },
    "quod.self_type": {
        "class": SelfType,
        "summary": (
            "Reference to the receiver type inside a TraitDef method signature "
            "or an ImplDef method. ImplDef's validator eagerly substitutes "
            "Self → for_type at construction, so the lowerer never sees Self."
        ),
        "example": {"kind": "quod.self_type"},
        "see_also": ["TraitDef", "ImplDef"],
    },
    "quod.trait_call": {
        "class": TraitCall,
        "summary": (
            "Call a trait method, dispatched on dispatch_type. After mono, "
            "rewritten to a direct Call to the resolved impl's mangled method "
            "(e.g. Counter::add). dispatch_type may be a TypeParamRef pre-mono."
        ),
        "example": {
            "kind": "quod.trait_call", "trait": "Add", "method": "add",
            "dispatch_type": {"kind": "llvm.struct", "name": "Counter"},
            "args": [],
        },
        "see_also": ["TraitDef", "ImplDef"],
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
    "lift_equivalence": {
        "class": LiftEquivalence,
        "summary": (
            "Justifies a layer-A → layer-B (transcription) equivalence "
            "via a pinned proof artifact. Same shape as `z3` — the file's "
            "bytes hashed at prove time, re-checked at verify time."
        ),
        "example": {
            "kind": "lift_equivalence",
            "artifact_path": "proofs/sum_lift.smt2",
            "artifact_hash": "abc123...",
        },
    },
    "family_lowering": {
        "class": FamilyLowering,
        "summary": (
            "Justifies a layer-B → layer-C equivalence by citing a named "
            "lowering rule (e.g. `c.for_general`) whose theorem was "
            "proved once, out of band, against the rule itself rather "
            "than per program. `artifact_path`/`artifact_hash` "
            "optionally pin the rule's proof."
        ),
        "example": {"kind": "family_lowering", "rule_name": "c.for_general"},
    },

    # ---------- edges + equivalences (program-level relational claims) ----------
    "edge.provenance": {
        "class": ProvenanceEdge,
        "summary": (
            "An unkinded provenance edge: 'this came from that,' nothing "
            "more. `source` and `target` are stable node IDs (Function.id, "
            "Block.id, ...). All semantic content for what the edge means "
            "lives in the `equivalent_to` claims that anchor on the same "
            "IDs."
        ),
        "example": {"kind": "edge.provenance", "source": "@a.sd", "target": "@b.lets"},
    },
    "equivalent_to": {
        "class": Equivalence,
        "summary": (
            "Program-level equivalence between two nodes by ID. "
            "Relational, where ordinary claims are unary; lives in "
            "`Program.equivalences`, not in `fn.claims`. Carries the "
            "regime / enforcement / justification metadata of any other "
            "claim. `domain` (predicate over which the equivalence "
            "holds) is reserved for the predicates spike — v2 lands "
            "with `domain=None` (always-true)."
        ),
        "example": {
            "kind": "equivalent_to",
            "a_node_id": "@blk_a",
            "b_node_id": "@blk_b",
            "regime": "witness",
            "justification": {
                "kind": "family_lowering",
                "rule_name": "c.for_general",
            },
        },
    },

    # ---------- program-level (no kind discriminator) ----------
    "StringConstant": {
        "class": StringConstant,
        "summary": "A null-terminated byte string declared at program top level. Reference via quod.string_ref.",
        "example": {"name": ".str.greeting", "value": "hello, world"},
    },
    "ExternFunction": {
        "class": ExternFunction,
        "summary": "A libc-or-similar function declared but not defined here. Use `arity` for all-i32 sigs, or `param_types` for typed. `linkage` is required and records the symbol's home: `linkage.libc` for clang-linked libc symbols, `linkage.runtime` for symbols in quod's runtime archive (libquodrt). `claims` carry contracts the optimizer exploits at every call site (return-scoped only today; lowered as `llvm.assume` after each call).",
        "example": {
            "name": "read",
            "param_types": [
                {"kind": "llvm.i32"}, {"kind": "llvm.i8_ptr"}, {"kind": "llvm.i64"}
            ],
            "return_type": {"kind": "llvm.i64"},
            "linkage": {"kind": "linkage.libc"},
            "claims": [{"kind": "return_in_range", "min": -1}],
        },
    },
    "LibcLinkage": {
        "class": LibcLinkage,
        "summary": "Extern provenance: the symbol comes from libc; clang's default link line provides it.",
        "example": {"kind": "linkage.libc"},
    },
    "RuntimeLinkage": {
        "class": RuntimeLinkage,
        "summary": "Extern provenance: the symbol comes from quod's runtime archive (every .c file under src/quod/runtime/ is compiled into libquodrt).",
        "example": {"kind": "linkage.runtime"},
    },
    "Function": {
        "class": Function,
        "summary": "A user function. params is a list of typed Params; return_type is required; body is a `Block` wrapping a list of statements; claims optional. Entry-point functions may declare params; the synthesized main wrapper parses each argv slot via atoll then trunc/sext's to the param's type (so `quod run -- 42 7` calls entry(42, 7)). An entry called 'main' must be nullary — rename it if you want params.",
        "example": {
            "name": "main", "params": [],
            "return_type": {"kind": "llvm.i32"},
            "body": {"stmts": [{"kind": "quod.return_expr",
                                "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 0}}]},
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
            "One payload field of an EnumVariant. Any value Type is "
            "allowed — int widths, i8*, named structs, even other enums. "
            "The enum's payload area is sized as `[N x i64]` where N "
            "covers the largest variant; per-variant layout uses a "
            "literal LLVM struct of the field types via bitcast."
        ),
        "example": {"name": "value", "type": {"kind": "llvm.i64"}},
    },
    "MatchArm": {
        "class": MatchArm,
        "summary": (
            "One arm of a quod.match. Names a variant, binds its payload "
            "fields to locals (one binding name per field, in declaration "
            "order), and runs `body` (a `Block`)."
        ),
        "example": {
            "variant": "Some",
            "bindings": ["v"],
            "body": {"stmts": [{"kind": "quod.return_expr",
                                "value": {"kind": "quod.local_ref", "name": "v"}}]},
        },
        "see_also": ["quod.match"],
    },
    "Block": {
        "class": Block,
        "summary": (
            "Identified container for a sequence of statements. Used wherever "
            "a body appears (Function.body, If.then_body / else_body, While.body, "
            "For.body, WithArena.body, MatchArm.body). The `id` is opaque, "
            "auto-minted at construction, and persists in JSON so reloads stay "
            "deterministic. Edges and equivalence claims anchor on Block IDs."
        ),
        "example": {
            "id": "@blk_example",
            "stmts": [
                {"kind": "quod.return_expr",
                 "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 0}},
            ],
        },
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
            {"variant": "None", "bindings": [], "body": {"stmts": [
                {"kind": "quod.return_expr",
                 "value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i64"}, "value": 0}}]}},
            {"variant": "Some", "bindings": ["v"], "body": {"stmts": [
                {"kind": "quod.return_expr",
                 "value": {"kind": "quod.local_ref", "name": "v"}}]}},
        ],
    },
    "see_also": ["EnumDef", "quod.enum_init"],
}


# ---------- Layer-A: C source-language nodes ----------
#
# Inert structural nodes that preserve the original C as a subtree of
# the program graph. See `.scratch/c-ingest/00-overview.md`. v3 supports
# the same int-only, no-structs/floats/switch C subset as the existing
# ingester, narrowed further to what `sum.c` exercises (the smallest
# end-to-end slice).

_KIND_INFO["c.type"] = {
    "class": CNamedType,
    "summary": (
        "A named scalar C type. v6 supports `int` and `char` "
        "(`char*` is `c.type.ptr` wrapping `c.type` with name='char')."
    ),
    "example": {"kind": "c.type", "name": "int"},
}

_KIND_INFO["c.type.ptr"] = {
    "class": CPointerType,
    "summary": (
        "A pointer C type — `T*`. The pointee is any `c.type` or "
        "another `c.type.ptr` (for `int**` etc.). All pointers "
        "collapse to `i8_ptr` at layer B; the pointee name is "
        "informational at the layer-A surface."
    ),
    "example": {
        "kind": "c.type.ptr",
        "pointee": {"kind": "c.type", "name": "char"},
    },
}

_KIND_INFO["c.lit_int"] = {
    "class": CIntLit,
    "summary": "A C integer literal.",
    "example": {"kind": "c.lit_int", "value": 0},
}

_KIND_INFO["c.var_ref"] = {
    "class": CVarRef,
    "summary": "A C identifier reference (parameter, local, …).",
    "example": {"kind": "c.var_ref", "name": "i"},
}

_KIND_INFO["c.enum_const_ref"] = {
    "class": CEnumConstRef,
    "summary": (
        "A C enum-constant reference. The layer-B lifter resolves "
        "these to integer values; layer A preserves both the "
        "source-level identifier (`name`) and the resolved integer "
        "(`value`) so the lift-check can verify equivalence without "
        "re-running libclang. If the enum's resolved value drifts, "
        "the pinned `value` disagrees with layer-B's IntLit and "
        "`equiv verify` catches it."
    ),
    "example": {"kind": "c.enum_const_ref", "name": "CURLOPT_URL", "value": 10002},
}

_KIND_INFO["c.binop"] = {
    "class": CBinOp,
    "summary": (
        "A C binary operator. `op` is the source-form spelling "
        "(`+`, `<`, `&&`, …). The lifter is responsible for refusing "
        "operators outside the supported subset."
    ),
    "example": {
        "kind": "c.binop", "op": "<",
        "lhs": {"kind": "c.var_ref", "name": "i"},
        "rhs": {"kind": "c.var_ref", "name": "n"},
    },
}

_KIND_INFO["c.lit_str"] = {
    "class": CStringLit,
    "summary": (
        "A C string literal. Layer A preserves the decoded payload "
        "(escapes resolved); the layer-B lifter interns these into "
        "a `StringConstant` and references via `StringRef`."
    ),
    "example": {"kind": "c.lit_str", "value": "hello, world\n"},
}

_KIND_INFO["c.call"] = {
    "class": CCall,
    "summary": (
        "A C function call expression. v6 supports only direct calls "
        "(non-indirect, non-function-pointer)."
    ),
    "example": {
        "kind": "c.call", "callee": "printf",
        "args": [
            {"kind": "c.lit_str", "value": "%d\n"},
            {"kind": "c.var_ref", "name": "x"},
        ],
    },
}

_KIND_INFO["c.array_subscript"] = {
    "class": CArraySubscript,
    "summary": (
        "`base[index]` — array subscript. v6 only emits this inside "
        "a `c.addr_of`; bare reads aren't yet supported."
    ),
    "example": {
        "kind": "c.array_subscript",
        "base": {"kind": "c.var_ref", "name": "buf"},
        "index": {"kind": "c.lit_int", "value": 7},
    },
}

_KIND_INFO["c.unary"] = {
    "class": CUnary,
    "summary": (
        "Unary prefix operator on an expression: `-x`, `!x`, `~x`. "
        "Layer A preserves the source operator; the lift pairs each "
        "with a layer-B BinOp identity (`-x ↔ sub(0, _)`, `!x ↔ "
        "eq(_, 0)`, `~x ↔ xor(_, -1)`)."
    ),
    "example": {
        "kind": "c.unary",
        "op": "~",
        "value": {"kind": "c.var_ref", "name": "x"},
    },
}

_KIND_INFO["c.addr_of"] = {
    "class": CAddressOf,
    "summary": (
        "`&expr` — address-of. v6 only emits this with a "
        "`c.array_subscript` target (`&p[k]` is C's pointer-arithmetic "
        "spelling, equivalent to `p + k` for char*)."
    ),
    "example": {
        "kind": "c.addr_of",
        "target": {
            "kind": "c.array_subscript",
            "base": {"kind": "c.var_ref", "name": "buf"},
            "index": {"kind": "c.lit_int", "value": 7},
        },
    },
}

_KIND_INFO["c.param"] = {
    "class": CParam,
    "summary": "A function parameter in C source.",
    "example": {"kind": "c.param", "name": "n", "type": {"kind": "c.type", "name": "int"}},
}

_KIND_INFO["c.var_decl"] = {
    "class": CVarDecl,
    "summary": "`int s = 0;` — a local variable declaration.",
    "example": {
        "kind": "c.var_decl",
        "type": {"kind": "c.type", "name": "int"},
        "name": "s",
        "init": {"kind": "c.lit_int", "value": 0},
    },
}

_KIND_INFO["c.multi_var_decl"] = {
    "class": CMultiVarDecl,
    "summary": (
        "`int a, b, c;` — a single declaration statement introducing "
        "multiple locals. Layer-A only: the lift expands each sub-decl "
        "to a separate layer-B `Let`, and the lift-checker pairs the "
        "multi-decl with the resulting N consecutive Lets."
    ),
    "example": {
        "kind": "c.multi_var_decl",
        "decls": [
            {"kind": "c.var_decl", "type": {"kind": "c.type", "name": "int"},
             "name": "a", "init": {"kind": "c.lit_int", "value": 0}},
            {"kind": "c.var_decl", "type": {"kind": "c.type", "name": "int"},
             "name": "b", "init": {"kind": "c.lit_int", "value": 1}},
        ],
    },
}

_KIND_INFO["c.compound_assign"] = {
    "class": CCompoundAssign,
    "summary": (
        "`x += y`, `x -= y`, etc. — compound assignment combining a "
        "binary operator with assignment. Layer-A only: the lift "
        "desugars to `Assign(x, BinOp(op_translated, LocalRef(x), y'))` "
        "on the layer-B side, paired by the lift-checker."
    ),
    "example": {
        "kind": "c.compound_assign",
        "target": "s",
        "op": "+=",
        "value": {"kind": "c.var_ref", "name": "i"},
    },
}

_KIND_INFO["c.assign"] = {
    "class": CAssign,
    "summary": (
        "`s = s + i;` — assignment to an in-scope variable. v3 doesn't "
        "model field/indexed/dereferenced targets."
    ),
    "example": {
        "kind": "c.assign", "target": "i",
        "value": {"kind": "c.binop", "op": "+",
                  "lhs": {"kind": "c.var_ref", "name": "i"},
                  "rhs": {"kind": "c.lit_int", "value": 1}},
    },
}

_KIND_INFO["c.return"] = {
    "class": CReturn,
    "summary": "`return s;` or `return;`.",
    "example": {"kind": "c.return", "value": {"kind": "c.var_ref", "name": "s"}},
}

_KIND_INFO["c.for"] = {
    "class": CFor,
    "summary": (
        "`for (init; cond; inc) { body }` — the C for loop verbatim. "
        "Each of init/cond/inc is independently optional."
    ),
    "example": {
        "kind": "c.for",
        "init": {"kind": "c.var_decl",
                 "type": {"kind": "c.type", "name": "int"},
                 "name": "i", "init": {"kind": "c.lit_int", "value": 0}},
        "cond": {"kind": "c.binop", "op": "<",
                 "lhs": {"kind": "c.var_ref", "name": "i"},
                 "rhs": {"kind": "c.var_ref", "name": "n"}},
        "inc":  {"kind": "c.assign", "target": "i",
                 "value": {"kind": "c.binop", "op": "+",
                           "lhs": {"kind": "c.var_ref", "name": "i"},
                           "rhs": {"kind": "c.lit_int", "value": 1}}},
        "body": [],
    },
}

_KIND_INFO["c.if"] = {
    "class": CIf,
    "summary": (
        "`if (cond) { then_body } else { else_body }` — empty "
        "`else_body` represents an if without else."
    ),
    "example": {
        "kind": "c.if",
        "cond": {"kind": "c.binop", "op": "<",
                 "lhs": {"kind": "c.var_ref", "name": "x"},
                 "rhs": {"kind": "c.lit_int", "value": 0}},
        "then_body": [
            {"kind": "c.return", "value": {"kind": "c.lit_int", "value": -1}},
        ],
        "else_body": [],
    },
}

_KIND_INFO["c.while"] = {
    "class": CWhile,
    "summary": "`while (cond) { body }` — pre-test loop.",
    "example": {
        "kind": "c.while",
        "cond": {"kind": "c.binop", "op": "<",
                 "lhs": {"kind": "c.var_ref", "name": "i"},
                 "rhs": {"kind": "c.var_ref", "name": "n"}},
        "body": [],
    },
}

_KIND_INFO["c.expr_stmt"] = {
    "class": CExprStmt,
    "summary": (
        "A C expression evaluated for its side effect — typically a "
        "call like `printf(...)`. v6 only emits this for calls; bare "
        "expression statements (e.g. `x;`) are refused at ingest time."
    ),
    "example": {
        "kind": "c.expr_stmt",
        "value": {"kind": "c.call", "callee": "printf",
                  "args": [{"kind": "c.lit_str", "value": "hi\n"}]},
    },
}

_KIND_INFO["c.fn"] = {
    "class": CFn,
    "summary": "A C function definition: `int sum(int n) { ... }`.",
    "example": {
        "kind": "c.fn", "name": "sum",
        "return_type": {"kind": "c.type", "name": "int"},
        "params": [{"kind": "c.param", "name": "n",
                    "type": {"kind": "c.type", "name": "int"}}],
        "body": [],
    },
}

_KIND_INFO["c_unit"] = {
    "class": CUnit,
    "summary": (
        "A C translation unit — one source file's contents preserved as "
        "layer-A nodes. Lives in `Program.source_units`."
    ),
    "example": {"kind": "c_unit", "source_path": "sum.c", "functions": []},
}


# ---------- Layer-B: c.* extension nodes ----------

_KIND_INFO["c.scoped_block"] = {
    "class": CScopedBlock,
    "summary": (
        "C-style block wrapper. `block` is the inner core.Block that "
        "edges anchor on; the wrapper carries family-specific scope "
        "semantics (which decls die at the closing brace). Lowered by "
        "c-family lowering to its inner block — `lower.py` refuses to "
        "see this wrapper."
    ),
    "example": {
        "kind": "c.scoped_block",
        "block": {"id": "@blk_for_body", "stmts": []},
        "scope_locals": ["i"],
    },
}

_KIND_INFO["c.for_general"] = {
    "class": CStyleFor,
    "summary": (
        "C-style for loop with arbitrary init/cond/inc — the layer-B "
        "transcription of `c.for` from layer A. Lowered to "
        "`Let + While + Assign` by c-family lowering, with the rule "
        "cited as `FamilyLowering(\"c.for_general\")` in the resulting "
        "equivalence claim. `lower.py` refuses to consume this — the "
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
}


_CATEGORIES: dict[str, list[str]] = {
    "expression": [
        "llvm.const_int", "llvm.param_ref", "quod.local_ref", "llvm.binop",
        "quod.sc_or", "quod.sc_and", "llvm.call", "quod.string_ref",
        "quod.struct_init", "quod.field", "quod.load_field",
        "quod.ptr_offset", "quod.widen", "quod.load", "quod.null_ptr",
        "quod.char_lit", "quod.enum_init", "quod.sizeof", "quod.try",
        "quod.trait_call",
    ],
    "statement": [
        "quod.return_expr", "quod.return", "quod.unreachable", "quod.if",
        "quod.let", "quod.assign", "quod.while", "quod.for", "quod.expr_stmt",
        "quod.field_set", "quod.store", "quod.store_field",
        "quod.with_arena", "quod.match",
    ],
    "type": [
        "llvm.i1", "llvm.i8", "llvm.i16", "llvm.i32", "llvm.i64",
        "llvm.i8_ptr", "llvm.struct", "llvm.enum", "llvm.void",
        "quod.type_param", "quod.self_type",
    ],
    "claim": ["non_negative", "int_range", "return_in_range", "equivalent_to"],
    "justification": [
        "z3", "manual", "derived", "lift_equivalence", "family_lowering",
    ],
    "program": [
        "StringConstant", "ExternFunction", "Function", "Param",
        "StructDef", "StructField", "FieldInit",
        "EnumDef", "EnumVariant", "EnumPayloadField", "MatchArm", "Block",
    ],
    "edge": ["edge.provenance"],
    "linkage": ["LibcLinkage", "RuntimeLinkage"],
    # Layer-A nodes — original C source preserved as quod nodes. See
    # .scratch/c-ingest/00-overview.md.
    "source.c": [
        "c_unit", "c.fn", "c.param", "c.type", "c.type.ptr",
        "c.var_decl", "c.multi_var_decl", "c.assign", "c.compound_assign",
        "c.return", "c.for", "c.if", "c.while", "c.expr_stmt",
        "c.binop", "c.lit_int", "c.lit_str", "c.var_ref",
        "c.enum_const_ref", "c.call",
        "c.array_subscript", "c.addr_of", "c.unary",
    ],
    # Layer-B `c.*` extensions — constructs core quod can't represent;
    # lowered to core by lower/c_family.py (step 5).
    "c": ["c.scoped_block", "c.for_general"],
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
