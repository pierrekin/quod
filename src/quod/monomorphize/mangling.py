"""Name-mangling helpers shared by every stage of monomorphization."""

from __future__ import annotations

from ..model import (
    EnumType,
    I1Type,
    I8PtrType,
    I8Type,
    I16Type,
    I32Type,
    I64Type,
    IsizeType,
    StructType,
    TypeParamRef,
    U8Type,
    U16Type,
    U32Type,
    U64Type,
    UsizeType,
    VoidType,
)


def _type_to_name(t) -> str:
    """Stable, human-readable string for a fully-concrete type. Used as
    a component in mangled struct/enum names. Must be deterministic and
    distinct for distinct types — `i64` ≠ `i32` ≠ `i8*` ≠ `core.str.String`.
    """
    if isinstance(t, I1Type):
        return "i1"
    if isinstance(t, I8Type):
        return "i8"
    if isinstance(t, I16Type):
        return "i16"
    if isinstance(t, I32Type):
        return "i32"
    if isinstance(t, I64Type):
        return "i64"
    if isinstance(t, U8Type):
        return "u8"
    if isinstance(t, U16Type):
        return "u16"
    if isinstance(t, U32Type):
        return "u32"
    if isinstance(t, U64Type):
        return "u64"
    if isinstance(t, IsizeType):
        return "isize"
    if isinstance(t, UsizeType):
        return "usize"
    if isinstance(t, I8PtrType):
        return "i8ptr"
    if isinstance(t, StructType):
        # Post-mono StructType has empty type_args, so the name is
        # already mangled. Pre-mono uses with type_args shouldn't appear
        # here — substitute resolves them first.
        if t.type_args:
            inner = ",".join(_type_to_name(a) for a in t.type_args)
            return f"{t.name}<{inner}>"
        return t.name
    if isinstance(t, EnumType):
        if t.type_args:
            inner = ",".join(_type_to_name(a) for a in t.type_args)
            return f"{t.name}<{inner}>"
        return t.name
    if isinstance(t, VoidType):
        return "void"
    if isinstance(t, TypeParamRef):
        # Substitute should have resolved this before mangling.
        raise AssertionError(f"unsubstituted TypeParamRef {t.name!r} in mangling")
    raise AssertionError(f"unhandled type for mangling: {t!r}")


def _mangle(template: str, args: tuple) -> str:
    """`Box<i64>`, `List<core.str.String>`, `Result<i64,ParseError>`. The
    raw form is kept literal — angle brackets and commas survive into
    LLVM identified-type names (llvmlite quotes as needed)."""
    inner = ",".join(_type_to_name(a) for a in args)
    return f"{template}<{inner}>"
