"""Category groupings for schema rendering.

Maps a category name (expression / statement / type / claim / justification /
program / edge / linkage / source.c / c) to the list of kinds it contains.
The render layer uses this to drive `quod schema` overview output.
"""

from __future__ import annotations


_CATEGORIES: dict[str, list[str]] = {
    "expression": [
        "llvm.const_int", "quod.float_lit",
        "llvm.param_ref", "quod.local_ref", "llvm.binop",
        "quod.sc_or", "quod.sc_and", "quod.if_expr", "quod.not", "quod.fneg",
        "quod.return_ref",
        "llvm.call", "quod.string_ref",
        "quod.struct_init", "quod.field", "quod.load_field",
        "quod.ptr_offset", "quod.cast", "quod.load", "quod.null_ptr",
        "quod.char_lit", "quod.enum_init", "quod.sizeof", "quod.try",
        "quod.trait_call",
    ],
    "statement": [
        "quod.return_expr", "quod.return", "quod.unreachable",
        "quod.break", "quod.continue", "quod.if",
        "quod.let", "quod.assign", "quod.while", "quod.do_while",
        "quod.for", "quod.expr_stmt",
        "quod.field_set", "quod.store", "quod.store_field",
        "quod.with_arena", "quod.match",
    ],
    "type": [
        "llvm.i1", "llvm.i8", "llvm.i16", "llvm.i32", "llvm.i64",
        "llvm.f32", "llvm.f64",
        "llvm.i8_ptr", "llvm.struct", "llvm.enum", "llvm.void",
        "quod.type_param", "quod.self_type",
    ],
    "claim": ["predicate", "equivalent_to"],
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
    # Layer-A nodes — original C source preserved as quod nodes.
    "source.c": [
        "c_unit", "c.fn", "c.param", "c.type", "c.type.ptr",
        "c.var_decl", "c.multi_var_decl", "c.assign", "c.compound_assign",
        "c.return", "c.for", "c.if", "c.while", "c.do_while", "c.expr_stmt",
        "c.break", "c.continue", "c.switch", "c.switch_case",
        "c.binop", "c.lit_int", "c.lit_float", "c.lit_str", "c.var_ref",
        "c.enum_const_ref", "c.call",
        "c.array_subscript", "c.addr_of", "c.unary", "c.ternary", "c.cast",
    ],
    # Layer-B `c.*` extensions — constructs core quod can't represent;
    # lowered to core by lower/c_family.py.
    "c": ["c.scoped_block", "c.for_general"],
}


def _category_of(name: str) -> str | None:
    for cat, kinds in _CATEGORIES.items():
        if name in kinds:
            return cat
    return None
