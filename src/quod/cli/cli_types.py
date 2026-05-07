"""Type-name parsing shared between cli_extern and cli_struct.

Maps the CLI tokens (i32, i8_ptr, libc, etc.) to model classes, including
struct/enum names looked up against the current program.
"""

from __future__ import annotations

import typer

from quod.model import (
    I1Type,
    I8PtrType,
    I8Type,
    I16Type,
    I32Type,
    I64Type,
    IsizeType,
    LibcLinkage,
    Linkage,
    RuntimeLinkage,
    StructType,
    U8Type,
    U16Type,
    U32Type,
    U64Type,
    UsizeType,
)


_TYPE_NAMES = {
    "i1": I1Type, "i8": I8Type, "i16": I16Type, "i32": I32Type, "i64": I64Type,
    "u8": U8Type, "u16": U16Type, "u32": U32Type, "u64": U64Type,
    "isize": IsizeType, "usize": UsizeType,
    "i8_ptr": I8PtrType,
}


_LINKAGE_NAMES = {
    "libc": LibcLinkage,
    "runtime": RuntimeLinkage,
}


def _parse_linkage(s: str) -> Linkage:
    cls = _LINKAGE_NAMES.get(s)
    if cls is None:
        raise typer.BadParameter(
            f"unknown linkage {s!r}; choices: {', '.join(_LINKAGE_NAMES)}"
        )
    return cls()


def _parse_type_name(
    s: str,
    *,
    struct_names: tuple[str, ...] = (),
    enum_names: tuple[str, ...] = (),
):
    """Parse a CLI type token. Accepts the built-in width names plus any
    `struct_names` (-> StructType) and `enum_names` (-> EnumType) passed
    in. Pass the program's current names to allow struct/enum types in
    extern or struct-field declarations; pass nothing for legacy
    (int-only) callsites."""
    from quod.model import EnumType
    cls = _TYPE_NAMES.get(s)
    if cls is not None:
        return cls()
    if s in struct_names:
        return StructType(name=s)
    if s in enum_names:
        return EnumType(name=s)
    choices = list(_TYPE_NAMES) + list(struct_names) + list(enum_names)
    raise typer.BadParameter(
        f"unknown type {s!r}; choices: {', '.join(choices)}"
    )
