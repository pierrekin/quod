"""Type nodes — int widths, pointer, struct, enum, type-param refs, void.

Width-per-class follows LLVM's "type carries no signedness" convention —
signedness lives on the operation (e.g. BinOp.slt vs ult). i1 is a
first-class type used for boolean values (cmp results, short-circuits,
explicit booleans).
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, model_serializer

from quod.model.base import _Node


class I1Type(_Node):
    kind: Literal["llvm.i1"] = "llvm.i1"


class I8Type(_Node):
    kind: Literal["llvm.i8"] = "llvm.i8"


class I16Type(_Node):
    kind: Literal["llvm.i16"] = "llvm.i16"


class I32Type(_Node):
    kind: Literal["llvm.i32"] = "llvm.i32"


class I64Type(_Node):
    kind: Literal["llvm.i64"] = "llvm.i64"


class U8Type(_Node):
    kind: Literal["llvm.u8"] = "llvm.u8"


class U16Type(_Node):
    kind: Literal["llvm.u16"] = "llvm.u16"


class U32Type(_Node):
    kind: Literal["llvm.u32"] = "llvm.u32"


class U64Type(_Node):
    kind: Literal["llvm.u64"] = "llvm.u64"


class IsizeType(_Node):
    """Pointer-sized signed integer. Lowers to i64 on the only target
    (64-bit Linux). Distinct from i64 at the type level so APIs that
    talk about "a count or offset that fits a pointer" can say so."""
    kind: Literal["llvm.isize"] = "llvm.isize"


class UsizeType(_Node):
    """Pointer-sized unsigned integer. Lowers to i64 width on the only
    target (64-bit Linux). Used for sizes, lengths, and indices."""
    kind: Literal["llvm.usize"] = "llvm.usize"


class F32Type(_Node):
    """IEEE 754 binary32 floating-point type.

    Strict-IEEE semantics in quod:

    - NaN-payload-undefined (specific bit patterns of NaN payloads are
      not preserved across operations; no sNaN/qNaN distinction).
    - Subnormals preserved (no FTZ / DAZ).
    - No FMA contraction in codegen.
    - Round-to-nearest-even (IEEE default); no fenv manipulation.
    - Float-to-int out-of-range saturates; NaN-to-int is 0.
    - Comparisons use LLVM ordered predicates plus `une` for `!=` so
      `NaN != NaN` is true.

    No constructor for f32 values exists in this commit (Cast preview
    only); FloatLit / FNeg / float BinOp ops land in the next commit.
    """
    kind: Literal["llvm.f32"] = "llvm.f32"


class F64Type(_Node):
    """IEEE 754 binary64 floating-point type. Same strict-IEEE
    semantics as F32Type — see that docstring for the full list."""
    kind: Literal["llvm.f64"] = "llvm.f64"


class I8PtrType(_Node):
    kind: Literal["llvm.i8_ptr"] = "llvm.i8_ptr"


class StructType(_Node):
    """Reference to a named StructDef. Pass-by-value at the LLVM level.

    The `name` must match a `StructDef.name` in the same Program; the
    Program-level validator catches dangling refs at load time.

    `type_args` populates the type parameters of a generic StructDef. An
    empty tuple is the non-generic case (matches a StructDef with empty
    `type_params`). The monomorphization pass walks every StructType
    with non-empty `type_args`, generates a fresh nominal struct, and
    rewrites this reference to the mangled name with empty type_args.
    Post-mono, every StructType has empty `type_args`.
    """
    kind: Literal["llvm.struct"] = "llvm.struct"
    name: str
    type_args: tuple["Type", ...] = ()

    @model_serializer(mode="wrap")
    def _drop_empty_type_args(self, handler, info):
        data = handler(self)
        if not self.type_args:
            data.pop("type_args", None)
        return data


class EnumType(_Node):
    """Reference to a named EnumDef. Pass-by-value at the LLVM level.

    Lowered as a tagged union: i8 discriminant + [N x i64] payload, where
    N is `max(1, max(len(variant.fields)))` and each payload field occupies
    one i64-sized slot (so payload field types are restricted to scalar
    types — int widths up to i64, plus i8*; struct and enum payload
    fields are not yet supported).

    `type_args`: same story as StructType — the monomorphization pass
    rewrites generic instantiations to mangled-name references with
    empty `type_args`.
    """
    kind: Literal["llvm.enum"] = "llvm.enum"
    name: str
    type_args: tuple["Type", ...] = ()

    @model_serializer(mode="wrap")
    def _drop_empty_type_args(self, handler, info):
        data = handler(self)
        if not self.type_args:
            data.pop("type_args", None)
        return data


class TypeParamRef(_Node):
    """A reference to an in-scope type parameter, e.g. `T` inside a generic
    `struct List<T> { ptr: *T, ... }`.

    Only valid inside the body of a generic StructDef / EnumDef / Function
    whose `type_params` includes `name`. The monomorphization pass
    substitutes every TypeParamRef for the corresponding concrete `Type`
    when emitting a monomorphized def. Post-mono, no TypeParamRef
    survives — encountering one at lower time is a bug.
    """
    kind: Literal["quod.type_param"] = "quod.type_param"
    name: str


class SelfType(_Node):
    """Reference to the receiver type inside a `TraitDef` method
    signature or an `ImplDef` method. `ImplDef`'s post-construction
    validator eagerly rewrites every `SelfType` inside its methods to
    the impl's `for_type`, so by the time the lowerer runs no `SelfType`
    survives — a `SelfType` reaching mono is a bug in TraitDef plumbing.
    """
    kind: Literal["quod.self_type"] = "quod.self_type"


class VoidType(_Node):
    """The LLVM `void` type. Only valid as a function return type.

    Functions returning void must use the bare `quod.return` statement
    (no value) and may not appear in any value position.
    """
    kind: Literal["llvm.void"] = "llvm.void"


# Integer-only sub-union: usable wherever a pointer would be nonsense
# (IntLit, function params/return, For loop var, claim-bearing locals).
# Signedness lives on the type for u8..u64 only as a typing aid — the
# LLVM lowering is the same width as the corresponding iN, and ops
# carry their own signedness (sdiv vs udiv, slt vs ult).
IntType = Annotated[
    Union[I1Type, I8Type, I16Type, I32Type, I64Type,
          U8Type, U16Type, U32Type, U64Type,
          IsizeType, UsizeType],
    Field(discriminator="kind"),
]

# Float-only sub-union. F32 / F64 are deliberately NOT in IntType —
# floats must not pass `isinstance(_, IntType)` anywhere.
FloatType = Annotated[
    Union[F32Type, F64Type],
    Field(discriminator="kind"),
]

# Full type union, including pointer, struct, and enum types — used for
# Let bindings, struct fields, and other value-bearing contexts. Void is
# deliberately excluded; see ReturnType for return positions.
# TypeParamRef is included for pre-monomorphization use; the mono pass
# substitutes them before lowering. SelfType is for trait/impl
# declarations; an ImplDef validator eagerly rewrites Self → for_type at
# construction time, so SelfType only appears in TraitDef method
# signatures (and the rewriter never sees it).
Type = Annotated[
    Union[I1Type, I8Type, I16Type, I32Type, I64Type,
          U8Type, U16Type, U32Type, U64Type,
          IsizeType, UsizeType,
          F32Type, F64Type,
          I8PtrType,
          StructType, EnumType, TypeParamRef, SelfType],
    Field(discriminator="kind"),
]

# Type that can appear at a function return position, including void.
ReturnType = Annotated[
    Union[I1Type, I8Type, I16Type, I32Type, I64Type,
          U8Type, U16Type, U32Type, U64Type,
          IsizeType, UsizeType,
          F32Type, F64Type,
          I8PtrType,
          StructType, EnumType, TypeParamRef, SelfType, VoidType],
    Field(discriminator="kind"),
]



def int_type_width(t: "IntType") -> int:
    """Bit width of an int type. isize/usize are pointer-sized = 64 on
    the only target."""
    match t:
        case I1Type():
            return 1
        case I8Type() | U8Type():
            return 8
        case I16Type() | U16Type():
            return 16
        case I32Type() | U32Type():
            return 32
        case I64Type() | U64Type() | IsizeType() | UsizeType():
            return 64
    raise ValueError(f"not an int type: {t!r}")


def int_type_signed(t: "IntType") -> bool:
    """Whether an int type is signed. i1 is treated as unsigned (boolean)."""
    match t:
        case I8Type() | I16Type() | I32Type() | I64Type() | IsizeType():
            return True
        case I1Type() | U8Type() | U16Type() | U32Type() | U64Type() | UsizeType():
            return False
    raise ValueError(f"not an int type: {t!r}")
