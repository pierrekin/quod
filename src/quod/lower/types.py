"""LLVM type translation and aggregate-layout helpers.

`_type_to_llvm` is the substrate every other lowering module uses to
produce LLVM types from quod model types. Layout helpers
(`_size_of_quod_type`, `_variant_struct_layout`, `_enum_layout`,
`_align_to`) compute byte offsets and alignments that mirror the IR
shapes emitted by `lower()`. `_coerce_int_lit` retypes bare integer
literals at lowering boundaries the script-time resolver can't reach
(call args, struct/enum init values, store destinations).
"""

from __future__ import annotations

from llvmlite import ir

from quod.model import (
    EnumDef,
    EnumType,
    EnumVariant,
    I1Type,
    I8PtrType,
    I8Type,
    I16Type,
    I32Type,
    I64Type,
    IntLit,
    IsizeType,
    StructDef,
    StructType,
    U8Type,
    U16Type,
    U32Type,
    U64Type,
    UsizeType,
    VoidType,
    int_type_width,
)


I1 = ir.IntType(1)
I8 = ir.IntType(8)
I16 = ir.IntType(16)
I32 = ir.IntType(32)
I64 = ir.IntType(64)


# Map quod.BinOp.op -> the icmp predicate (cmp ops only).
_ICMP_SIGNED = {
    "slt": "<", "sle": "<=", "sgt": ">", "sge": ">=",
    "eq": "==", "ne": "!=",
}
_ICMP_UNSIGNED = {
    "ult": "<", "ule": "<=", "ugt": ">", "uge": ">=",
}


def _type_to_llvm(
    t,
    struct_tys: dict[str, "ir.IdentifiedStructType"] | None = None,
    enum_tys: dict[str, "ir.IdentifiedStructType"] | None = None,
):
    """Lower a quod type to its LLVM equivalent.

    `struct_tys` is the per-module registry of identified struct types,
    threaded through every site that lowers a Type. None is allowed for
    legacy callers that operate on int-only contexts; passing None when
    the type IS a struct raises.

    `enum_tys` is the parallel registry for enums. Same None semantics.
    """
    match t:
        case I1Type():
            return I1
        case I8Type() | U8Type():
            return I8
        case I16Type() | U16Type():
            return I16
        case I32Type() | U32Type():
            return I32
        case I64Type() | U64Type() | IsizeType() | UsizeType():
            return I64
        case I8PtrType():
            return I8.as_pointer()
        case StructType(name=name):
            if struct_tys is None or name not in struct_tys:
                raise ValueError(f"struct type {name!r} not registered with the module")
            return struct_tys[name]
        case EnumType(name=name):
            if enum_tys is None or name not in enum_tys:
                raise ValueError(f"enum type {name!r} not registered with the module")
            return enum_tys[name]
        case VoidType():
            return ir.VoidType()
    raise ValueError(f"unhandled quod.Type: {t!r}")


def _variant_struct_ty(
    var: "EnumVariant",
    struct_tys: dict[str, "ir.IdentifiedStructType"],
    enum_tys: dict[str, "ir.IdentifiedStructType"],
) -> ir.LiteralStructType:
    """Build the anonymous LLVM struct type that holds one variant's
    payload fields, in declaration order. Empty for unit variants."""
    field_tys = [_type_to_llvm(f.type, struct_tys, enum_tys) for f in var.fields]
    return ir.LiteralStructType(field_tys)


_LLVM_INT_TO_QUOD: dict[int, type] = {
    1: I1Type, 8: I8Type, 16: I16Type, 32: I32Type, 64: I64Type,
}


def _coerce_int_lit(expr, dest_ty):
    """If `expr` is a bare IntLit and `dest_ty` is a different-width
    integer type, return a new IntLit retyped to `dest_ty`. Otherwise
    return `expr` unchanged.

    The rule is bare-literals-only: `parser.had_error = 1` retypes 1 to
    i8 because the field is i8, but `parser.had_error = (x + 1)` does
    NOT — composite expressions don't auto-narrow (silent truncation
    would hide bugs).

    The script-time type-resolution pass handles the cases it can see
    (`let x: i32 = 5`, `return 5`, `x < 5` where x is a typed param).
    What's left for this lower-time coercion is exactly the contexts
    the script can't reach without dragging the whole Program into the
    parser:
      - Call args (callee param_types live on a Function/ExternFunction
        in the same Program)
      - StructInit / EnumInit field values (field types live on the
        StructDef / EnumVariant)
      - FieldSet / StoreField (struct field destinations)
      - Store (the destination type comes from the value being written;
        for bare literals, it's the post-coercion type)
    Programs authored as raw JSON or lifted from C also bypass the
    resolver, so this fallback covers them too.
    """
    if not isinstance(expr, IntLit):
        return expr
    if not isinstance(dest_ty, ir.IntType):
        return expr
    src_w = int_type_width(expr.type)
    if src_w == dest_ty.width:
        return expr
    cls = _LLVM_INT_TO_QUOD.get(dest_ty.width)
    if cls is None:
        return expr
    return expr.model_copy(update={"type": cls()})


def _size_of_quod_type(
    t,
    struct_defs: dict[str, StructDef],
    enum_defs: dict[str, EnumDef],
) -> tuple[int, int]:
    """Return (abi_size, abi_alignment) in bytes for a quod type.

    Assumes a 64-bit data model (i8* is 8 bytes, 8-byte aligned). Mirrors
    the LLVM data-layout rules for the quod type system: integers are
    naturally aligned to their width; structs accumulate fields with
    per-field alignment + tail padding to the struct's max-alignment;
    enums layout as `{i8 tag, [N x i8] payload}` where N is the largest
    variant's struct size — and the enum's alignment is the max alignment
    across all variant fields (so the payload bytes are correctly
    aligned for the most-aligned field).
    """
    match t:
        case I1Type() | I8Type() | U8Type():
            return (1, 1)
        case I16Type() | U16Type():
            return (2, 2)
        case I32Type() | U32Type():
            return (4, 4)
        case I64Type() | U64Type() | IsizeType() | UsizeType() | I8PtrType():
            return (8, 8)
        case StructType(name=name):
            sd = struct_defs.get(name)
            if sd is None:
                raise ValueError(f"sizeof: undefined struct {name!r}")
            offset = 0
            max_align = 1
            for f in sd.fields:
                fsize, falign = _size_of_quod_type(f.type, struct_defs, enum_defs)
                offset = _align_to(offset, falign)
                offset += fsize
                if falign > max_align:
                    max_align = falign
            offset = _align_to(offset, max_align)
            return (offset, max_align)
        case EnumType(name=name):
            ed = enum_defs.get(name)
            if ed is None:
                raise ValueError(f"sizeof: undefined enum {name!r}")
            return _enum_layout(ed, struct_defs, enum_defs)
    raise ValueError(f"sizeof: unhandled type {t!r}")


def _variant_struct_layout(
    var: "EnumVariant",
    struct_defs: dict[str, StructDef],
    enum_defs: dict[str, EnumDef],
) -> tuple[int, int]:
    """(size, alignment) of a single variant's payload struct."""
    offset = 0
    max_align = 1
    for f in var.fields:
        fsize, falign = _size_of_quod_type(f.type, struct_defs, enum_defs)
        offset = _align_to(offset, falign)
        offset += fsize
        if falign > max_align:
            max_align = falign
    offset = _align_to(offset, max_align)
    return (offset, max_align)


def _enum_layout(
    ed: EnumDef,
    struct_defs: dict[str, StructDef],
    enum_defs: dict[str, EnumDef],
) -> tuple[int, int]:
    """(size, alignment) of an enum value. Must mirror the actual LLVM
    layout `{i8 tag, [N x i64] payload}` emitted by lower_program — N
    is ceil(largest-variant-payload-bytes / 8), min 1. The `[N x i64]`
    array is 8-aligned, so the whole enum is 8-aligned regardless of
    what individual variant fields would naturally need; an enum with
    only an `i32` payload still occupies 16 bytes (1 tag + 7 padding +
    8 slot). Computing alignment from the variant's natural alignment
    here would undercount when this enum is itself a payload field of
    a larger enum."""
    payload_size = 0
    for v in ed.variants:
        vsize, _valign = _variant_struct_layout(v, struct_defs, enum_defs)
        if vsize > payload_size:
            payload_size = vsize
    # n_slots mirrors lower_program: at least 1, ceil-div by 8.
    n_slots = (payload_size + 7) // 8 or 1
    payload_align = 8
    payload_offset = _align_to(1, payload_align)
    total = _align_to(payload_offset + 8 * n_slots, payload_align)
    return (total, payload_align)


def _align_to(offset: int, alignment: int) -> int:
    """Round `offset` up to the next multiple of `alignment`."""
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)
