"""Core kind catalog: llvm.* and quod.* node kinds.

Each entry maps a canonical name (the discriminator string for nodes that
have one; a stable label for top-level types that don't) to:
    class:    the Pydantic model class (used for field introspection)
    summary:  one-line description shown in listings and at the top of
              per-kind output
    example:  a hand-curated minimal valid instance, rendered as JSON
    field_descriptions: optional per-field human notes (overrides default)
    see_also: optional cross-references to related kinds
"""

from __future__ import annotations

from typing import Any

from quod.model import (
    Assign,
    BinOp,
    Block,
    Break,
    Call,
    CharLit,
    Continue,
    DerivedJustification,
    DoWhile,
    EnumDef,
    EnumInit,
    EnumPayloadField,
    EnumType,
    EnumVariant,
    Equivalence,
    ExprStmt,
    ExternFunction,
    FamilyLowering,
    FieldInit,
    FieldRead,
    FieldSet,
    For,
    Function,
    I16Type,
    I1Type,
    I32Type,
    I64Type,
    I8PtrType,
    I8Type,
    If,
    IfExpr,
    IntLit,
    IsizeType,
    Let,
    LibcLinkage,
    LiftEquivalence,
    Load,
    LoadField,
    LocalRef,
    ManualJustification,
    Match,
    MatchArm,
    Not,
    NullPtr,
    Param,
    ParamRef,
    PredicateClaim,
    ProvenanceEdge,
    PtrOffset,
    Return,
    ReturnExpr,
    ReturnRef,
    RuntimeLinkage,
    SelfType,
    ShortCircuitAnd,
    ShortCircuitOr,
    SizeOf,
    Store,
    StoreField,
    StringConstant,
    StringRef,
    StructDef,
    StructField,
    StructInit,
    StructType,
    TraitCall,
    TryExpr,
    TypeParamRef,
    U16Type,
    U32Type,
    U64Type,
    U8Type,
    Unreachable,
    UsizeType,
    VoidType,
    While,
    Widen,
    WithArena,
    Z3Justification,
)


_CORE_CATALOG: dict[str, dict[str, Any]] = {
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
    "quod.break": {
        "class": Break,
        "summary": (
            "Exit the innermost enclosing loop. Validator refuses any "
            "Break outside a loop body."
        ),
        "example": {"kind": "quod.break"},
        "see_also": ["quod.continue", "quod.while", "quod.for"],
    },
    "quod.continue": {
        "class": Continue,
        "summary": (
            "Skip to the next iteration of the innermost enclosing loop. "
            "Validator refuses any Continue outside a loop body. Inside "
            "a c.for_general body, the c-family lowering pre-rewrites "
            "Continue to `inc; continue` so C semantics are preserved."
        ),
        "example": {"kind": "quod.continue"},
        "see_also": ["quod.break", "quod.while", "quod.for"],
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
    "quod.if_expr": {
        "class": IfExpr,
        "summary": (
            "Expression-level conditional (`cond ? a : b`). cond must "
            "lower to i1; then_value and else_value must have the same "
            "type. Lowered to branch + phi — only the selected branch "
            "is evaluated, so side effects in the unselected branch are "
            "skipped."
        ),
        "example": {
            "kind": "quod.if_expr",
            "cond": {"kind": "llvm.binop", "op": "slt",
                     "lhs": {"kind": "llvm.param_ref", "name": "x"},
                     "rhs": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 0}},
            "then_value": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": -1},
            "else_value": {"kind": "llvm.param_ref", "name": "x"},
        },
        "see_also": ["quod.if", "quod.sc_or", "quod.sc_and"],
    },
    "quod.not": {
        "class": Not,
        "summary": (
            "Boolean negation. Operand must lower to i1; result is i1. "
            "Lowered to `xor operand, 1`. Used in predicate expressions; "
            "prefer over `eq(x, 0)` for booleans."
        ),
        "example": {
            "kind": "quod.not",
            "operand": {"kind": "llvm.binop", "op": "eq",
                        "lhs": {"kind": "llvm.param_ref", "name": "x"},
                        "rhs": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 0}},
        },
        "see_also": ["llvm.binop", "quod.sc_or", "quod.sc_and"],
    },
    "quod.return_ref": {
        "class": ReturnRef,
        "summary": (
            "Symbolic reference to the enclosing function's return "
            "value. Valid only inside `PredicateClaim.expr`. Type is "
            "the function's return type — there's no type field, "
            "mirroring `llvm.param_ref`."
        ),
        "example": {"kind": "quod.return_ref"},
        "see_also": ["predicate", "llvm.param_ref"],
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
        "see_also": ["quod.do_while", "quod.for"],
    },
    "quod.do_while": {
        "class": DoWhile,
        "summary": (
            "Post-test loop. Body runs unconditionally first, then cond "
            "is evaluated; loops back if true. The body always executes "
            "at least once. Inside the body, `continue` jumps to the "
            "cond check (matches C `do { ... } while (...);`)."
        ),
        "example": {
            "kind": "quod.do_while",
            "body": {"stmts": [{"kind": "quod.assign", "name": "i",
                                "value": {"kind": "llvm.binop", "op": "add",
                                          "lhs": {"kind": "quod.local_ref", "name": "i"},
                                          "rhs": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 1}}}]},
            "cond": {"kind": "llvm.binop", "op": "slt",
                     "lhs": {"kind": "quod.local_ref", "name": "i"},
                     "rhs": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 10}},
        },
        "see_also": ["quod.while", "quod.for"],
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
    "predicate": {
        "class": PredicateClaim,
        "summary": (
            "Single canonical claim: a predicate over the function's "
            "params and (optionally) `quod.return_ref`. Pre/post is "
            "implicit — a predicate referencing return_ref is a "
            "postcondition, otherwise a precondition. Stored in "
            "canonical form so identical predicates hash identically. "
            "The named claim shapes (non_negative, int_range, "
            "return_in_range) are CLI sugar that desugar to this kind."
        ),
        "example": {
            "kind": "predicate",
            "expr": {
                "kind": "llvm.binop", "op": "sge",
                "lhs": {"kind": "llvm.param_ref", "name": "x"},
                "rhs": {"kind": "llvm.const_int", "type": {"kind": "llvm.i32"}, "value": 0},
            },
        },
        "see_also": ["quod.return_ref", "quod.not"],
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
            "holds) is currently always `None` (always-true); a real "
            "predicate domain is not yet supported."
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
            "claims": [{
                "kind": "predicate",
                "expr": {
                    "kind": "llvm.binop", "op": "sle",
                    "lhs": {"kind": "llvm.const_int", "type": {"kind": "llvm.i64"}, "value": -1},
                    "rhs": {"kind": "quod.return_ref"},
                },
            }],
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


_CORE_CATALOG["quod.match"] = {
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
