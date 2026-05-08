"""Layer-A kind catalog: C source-language nodes.

Inert structural nodes that preserve the original C as a subtree of the
program graph. The supported subset is int-only, no structs/floats/switch
— matching what the C ingester emits.
"""

from __future__ import annotations

from typing import Any

from quod.model import (
    CAddressOf,
    CArraySubscript,
    CAssign,
    CBinOp,
    CBreak,
    CCall,
    CCast,
    CCompoundAssign,
    CContinue,
    CDeref,
    CDerefStore,
    CDoWhile,
    CEnumConstRef,
    CExprStmt,
    CField,
    CFieldArrow,
    CFieldArrowStore,
    CFieldInit,
    CFieldRead,
    CFloatLit,
    CFn,
    CFor,
    CIf,
    CIntLit,
    CMultiVarDecl,
    CNamedType,
    CParam,
    CPointerType,
    CReturn,
    CStringLit,
    CStructDef,
    CStructInit,
    CStructType,
    CSubscriptStore,
    CSwitch,
    CSwitchCase,
    CTernary,
    CUnary,
    CUnit,
    CVarDecl,
    CVarRef,
    CWhile,
)


_LAYER_A_CATALOG: dict[str, dict[str, Any]] = {
    "c.type": {
        "class": CNamedType,
        "summary": (
            "A named scalar C type. Currently supports `int` and `char` "
            "(`char*` is `c.type.ptr` wrapping `c.type` with name='char')."
        ),
        "example": {"kind": "c.type", "name": "int"},
    },

    "c.type.ptr": {
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
    },

    "c.lit_int": {
        "class": CIntLit,
        "summary": "A C integer literal.",
        "example": {"kind": "c.lit_int", "value": 0},
    },

    "c.lit_float": {
        "class": CFloatLit,
        "summary": (
            "A C floating-point literal stored as its IEEE 754 bit "
            "pattern. `type` selects f32 vs f64 based on the source "
            "suffix (`1.5f` → f32; plain `1.5` → f64). Decimal forms "
            "parse via libc `strtof`/`strtod` (single-rounding for the "
            "target precision). Hex floats (`0x1.8p+1`) parse via "
            "`float.fromhex` and round to the target if needed. "
            "Special values (`+inf`, `-inf`, NaN) are ordinary bit "
            "patterns."
        ),
        "example": {"kind": "c.lit_float", "type": {"kind": "llvm.f64"}, "bits": 4609434218613702656},
    },

    "c.cast": {
        "class": CCast,
        "summary": (
            "An explicit C cast: `(T)expr` (CSTYLE_CAST_EXPR) or T(expr) "
            "(CXX_FUNCTIONAL_CAST_EXPR). Implicit promotions inserted by "
            "clang aren't source syntax — they have no layer-A "
            "representation; they synthesize as layer-B `Cast` nodes only."
        ),
        "example": {
            "kind": "c.cast",
            "target_type": {"kind": "c.type", "name": "double"},
            "value": {"kind": "c.lit_int", "value": 1},
        },
    },

    "c.var_ref": {
        "class": CVarRef,
        "summary": "A C identifier reference (parameter, local, …).",
        "example": {"kind": "c.var_ref", "name": "i"},
    },

    "c.enum_const_ref": {
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
    },

    "c.binop": {
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
    },

    "c.lit_str": {
        "class": CStringLit,
        "summary": (
            "A C string literal. Layer A preserves the decoded payload "
            "(escapes resolved); the layer-B lifter interns these into "
            "a `StringConstant` and references via `StringRef`."
        ),
        "example": {"kind": "c.lit_str", "value": "hello, world\n"},
    },

    "c.call": {
        "class": CCall,
        "summary": (
            "A C function call expression. Only direct calls are supported "
            "(no indirect / function-pointer calls)."
        ),
        "example": {
            "kind": "c.call", "callee": "printf",
            "args": [
                {"kind": "c.lit_str", "value": "%d\n"},
                {"kind": "c.var_ref", "name": "x"},
            ],
        },
    },

    "c.array_subscript": {
        "class": CArraySubscript,
        "summary": (
            "`base[index]` — array subscript. Two roles: inside a "
            "`c.addr_of` (pointer arithmetic, `elem_type=null`); "
            "standalone in expression position (typed load, "
            "`elem_type` set to the element's quod type)."
        ),
        "example": {
            "kind": "c.array_subscript",
            "base": {"kind": "c.var_ref", "name": "p"},
            "index": {"kind": "c.var_ref", "name": "k"},
            "elem_type": {"kind": "c.type", "name": "int"},
        },
    },

    "c.deref": {
        "class": CDeref,
        "summary": (
            "`*p` — pointer dereference (rvalue load). The pointee "
            "value's quod type is carried on the node and pairs with "
            "a layer-B `Load(p', load_type')`."
        ),
        "example": {
            "kind": "c.deref",
            "operand": {"kind": "c.var_ref", "name": "p"},
            "load_type": {"kind": "c.type", "name": "int"},
        },
    },

    "c.deref_store": {
        "class": CDerefStore,
        "summary": (
            "`*p = v;` — store via pointer dereference. Pairs with "
            "a layer-B `Store(p', v')`."
        ),
        "example": {
            "kind": "c.deref_store",
            "operand": {"kind": "c.var_ref", "name": "p"},
            "value": {"kind": "c.lit_int", "value": 0},
            "store_type": {"kind": "c.type", "name": "int"},
        },
    },

    "c.subscript_store": {
        "class": CSubscriptStore,
        "summary": (
            "`arr[k] = v;` — store at a subscripted location. Pairs "
            "with a layer-B `Store(PtrOffset(base, mul(widen64(k), "
            "sizeof(T))), v)`."
        ),
        "example": {
            "kind": "c.subscript_store",
            "base": {"kind": "c.var_ref", "name": "p"},
            "index": {"kind": "c.var_ref", "name": "k"},
            "value": {"kind": "c.lit_int", "value": 0},
            "elem_type": {"kind": "c.type", "name": "int"},
        },
    },

    "c.type.struct": {
        "class": CStructType,
        "summary": (
            "A named struct type reference: `struct Foo`. Pairs with "
            "layer-B `StructType(name='Foo')`."
        ),
        "example": {"kind": "c.type.struct", "name": "Point"},
    },

    "c.struct_def": {
        "class": CStructDef,
        "summary": (
            "`struct Foo { int x; int y; };` — file-scope struct "
            "definition. Pairs with layer-B `StructDef`."
        ),
        "example": {
            "kind": "c.struct_def", "name": "Point",
            "fields": [
                {"kind": "c.field_decl", "name": "x", "type": {"kind": "c.type", "name": "int"}},
                {"kind": "c.field_decl", "name": "y", "type": {"kind": "c.type", "name": "int"}},
            ],
        },
    },

    "c.field_decl": {
        "class": CField,
        "summary": "One field of a `c.struct_def`.",
        "example": {
            "kind": "c.field_decl", "name": "x",
            "type": {"kind": "c.type", "name": "int"},
        },
    },

    "c.struct_init": {
        "class": CStructInit,
        "summary": (
            "`= {a, b}` aggregate initialiser for a struct value. "
            "Pairs with layer-B `StructInit(type=type_name, "
            "fields=...)`."
        ),
        "example": {
            "kind": "c.struct_init", "type_name": "Point",
            "fields": [
                {"kind": "c.field_init", "name": "x",
                 "value": {"kind": "c.var_ref", "name": "a"}},
                {"kind": "c.field_init", "name": "y",
                 "value": {"kind": "c.var_ref", "name": "b"}},
            ],
        },
    },

    "c.field_init": {
        "class": CFieldInit,
        "summary": (
            "One field's value in a `c.struct_init`. `name` is None "
            "for positional initialisers."
        ),
        "example": {
            "kind": "c.field_init", "name": "x",
            "value": {"kind": "c.var_ref", "name": "a"},
        },
    },

    "c.field": {
        "class": CFieldRead,
        "summary": (
            "`p.x` — by-value struct field access. Pairs with "
            "layer-B `FieldRead(value=value', name='x')`."
        ),
        "example": {
            "kind": "c.field",
            "value": {"kind": "c.var_ref", "name": "p"},
            "name": "x",
            "field_type": {"kind": "c.type", "name": "int"},
        },
    },

    "c.field_arrow": {
        "class": CFieldArrow,
        "summary": (
            "`p->x` — struct-pointer field access. Pairs with "
            "layer-B `LoadField(ptr=p', struct_type=..., name='x')`."
        ),
        "example": {
            "kind": "c.field_arrow",
            "ptr": {"kind": "c.var_ref", "name": "p"},
            "struct_type": "Point",
            "name": "x",
            "field_type": {"kind": "c.type", "name": "int"},
        },
    },

    "c.field_arrow_store": {
        "class": CFieldArrowStore,
        "summary": (
            "`p->x = v;` — store via struct pointer. Pairs with "
            "layer-B `StoreField(ptr=p', struct_type=..., "
            "name='x', value=v')`. By-value field assignment "
            "(`p.x = v`) is refused at ingest."
        ),
        "example": {
            "kind": "c.field_arrow_store",
            "ptr": {"kind": "c.var_ref", "name": "p"},
            "struct_type": "Point",
            "name": "x",
            "value": {"kind": "c.lit_int", "value": 5},
        },
    },

    "c.break": {
        "class": CBreak,
        "summary": "`break;` — exit the innermost enclosing loop. Lifts to layer-B `Break`.",
        "example": {"kind": "c.break"},
    },

    "c.continue": {
        "class": CContinue,
        "summary": (
            "`continue;` — skip to the next iteration. Lifts to layer-B "
            "`Continue`. Inside a c.for_general body, the c-family lowering "
            "pre-rewrites this to `inc; continue` so C semantics are preserved."
        ),
        "example": {"kind": "c.continue"},
    },

    "c.ternary": {
        "class": CTernary,
        "summary": (
            "`cond ? then_value : else_value` — the C ternary operator. "
            "Layer A preserves source form; the lift maps each CTernary "
            "to a layer-B `IfExpr` with the same three sub-expressions."
        ),
        "example": {
            "kind": "c.ternary",
            "cond": {"kind": "c.binop", "op": "<",
                     "lhs": {"kind": "c.var_ref", "name": "x"},
                     "rhs": {"kind": "c.lit_int", "value": 0}},
            "then_value": {"kind": "c.unary", "op": "-",
                           "value": {"kind": "c.var_ref", "name": "x"}},
            "else_value": {"kind": "c.var_ref", "name": "x"},
        },
    },

    "c.unary": {
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
    },

    "c.addr_of": {
        "class": CAddressOf,
        "summary": (
            "`&expr` — address-of. Only emitted with a "
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
    },

    "c.param": {
        "class": CParam,
        "summary": "A function parameter in C source.",
        "example": {"kind": "c.param", "name": "n", "type": {"kind": "c.type", "name": "int"}},
    },

    "c.var_decl": {
        "class": CVarDecl,
        "summary": "`int s = 0;` — a local variable declaration.",
        "example": {
            "kind": "c.var_decl",
            "type": {"kind": "c.type", "name": "int"},
            "name": "s",
            "init": {"kind": "c.lit_int", "value": 0},
        },
    },

    "c.multi_var_decl": {
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
    },

    "c.compound_assign": {
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
    },

    "c.assign": {
        "class": CAssign,
        "summary": (
            "`s = s + i;` — assignment to an in-scope variable. Layer A doesn't "
            "model field/indexed/dereferenced targets."
        ),
        "example": {
            "kind": "c.assign", "target": "i",
            "value": {"kind": "c.binop", "op": "+",
                      "lhs": {"kind": "c.var_ref", "name": "i"},
                      "rhs": {"kind": "c.lit_int", "value": 1}},
        },
    },

    "c.return": {
        "class": CReturn,
        "summary": "`return s;` or `return;`.",
        "example": {"kind": "c.return", "value": {"kind": "c.var_ref", "name": "s"}},
    },

    "c.for": {
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
    },

    "c.if": {
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
    },

    "c.switch": {
        "class": CSwitch,
        "summary": (
            "`switch (scrutinee) { case ...: ...; default: ...; }` — "
            "multiway dispatch on an integer value. The lift produces an "
            "if-else-if chain at layer B (each case becomes one If with "
            "`scrutinee == value` cond, possibly OR'd for stacked cases). "
            "Each case body must end with `break;` or `return ...;`; "
            "implicit fallthrough refuses at ingest."
        ),
        "example": {
            "kind": "c.switch",
            "scrutinee": {"kind": "c.var_ref", "name": "x"},
            "cases": [
                {"kind": "c.switch_case",
                 "values": [{"kind": "c.lit_int", "value": 1}],
                 "body": [{"kind": "c.return", "value": {"kind": "c.lit_int", "value": 100}}]},
            ],
            "default": [{"kind": "c.return", "value": {"kind": "c.lit_int", "value": 0}}],
        },
    },

    "c.switch_case": {
        "class": CSwitchCase,
        "summary": (
            "One arm of a `c.switch`. Stacked-empty-case labels (`case 1: "
            "case 2: shared;`) share one body, encoded as a single "
            "CSwitchCase with multiple values."
        ),
        "example": {
            "kind": "c.switch_case",
            "values": [
                {"kind": "c.lit_int", "value": 1},
                {"kind": "c.lit_int", "value": 2},
            ],
            "body": [{"kind": "c.return", "value": {"kind": "c.lit_int", "value": 12}}],
        },
    },

    "c.do_while": {
        "class": CDoWhile,
        "summary": (
            "`do { body } while (cond);` — post-test loop. Layer A "
            "preserves the source statement; lifts to layer-B `DoWhile` "
            "(core)."
        ),
        "example": {
            "kind": "c.do_while",
            "body": [],
            "cond": {"kind": "c.lit_int", "value": 1},
        },
    },

    "c.while": {
        "class": CWhile,
        "summary": "`while (cond) { body }` — pre-test loop.",
        "example": {
            "kind": "c.while",
            "cond": {"kind": "c.binop", "op": "<",
                     "lhs": {"kind": "c.var_ref", "name": "i"},
                     "rhs": {"kind": "c.var_ref", "name": "n"}},
            "body": [],
        },
    },

    "c.expr_stmt": {
        "class": CExprStmt,
        "summary": (
            "A C expression evaluated for its side effect — typically a "
            "call like `printf(...)`. Only emitted for calls; bare "
            "expression statements (e.g. `x;`) are refused at ingest time."
        ),
        "example": {
            "kind": "c.expr_stmt",
            "value": {"kind": "c.call", "callee": "printf",
                      "args": [{"kind": "c.lit_str", "value": "hi\n"}]},
        },
    },

    "c.fn": {
        "class": CFn,
        "summary": "A C function definition: `int sum(int n) { ... }`.",
        "example": {
            "kind": "c.fn", "name": "sum",
            "return_type": {"kind": "c.type", "name": "int"},
            "params": [{"kind": "c.param", "name": "n",
                        "type": {"kind": "c.type", "name": "int"}}],
            "body": [],
        },
    },

    "c_unit": {
        "class": CUnit,
        "summary": (
            "A C translation unit — one source file's contents preserved as "
            "layer-A nodes. Lives in `Program.source_units`."
        ),
        "example": {"kind": "c_unit", "source_path": "sum.c", "functions": []},
    },

}