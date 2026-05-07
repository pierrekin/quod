"""Shared utilities for C ingestion (Layer A and Layer B both consume).

Op tables, type predicates, error helpers, AST navigation, and type
translation. Leaf module: imports only from `clang.cindex` and
`quod.model`; the layer-A / layer-B / driver modules import from here.
"""

from __future__ import annotations

import clang.cindex as cx

from quod.model import (
    BinOp,
    Expr,
    F32Type,
    F64Type,
    I8PtrType,
    I32Type,
    I64Type,
    ShortCircuitAnd,
    ShortCircuitOr,
    Type,
    VoidType,
)


_I32 = I32Type()
_I64 = I64Type()
_F32 = F32Type()
_F64 = F64Type()
_I8PTR = I8PtrType()
_VOID = VoidType()


# Float type kinds clang exposes. Used to route BinOp dispatch and
# identify implicit casts that change the quod-level type.
_FLOAT_TYPE_KINDS = frozenset({cx.TypeKind.FLOAT, cx.TypeKind.DOUBLE})


def _is_clang_float(t: cx.Type) -> bool:
    return t.get_canonical().kind in _FLOAT_TYPE_KINDS


# Char-typed pointee kinds. Pointer arithmetic on these has byte stride,
# which matches quod.ptr_offset. Wider pointee types (int*, struct*) need
# scaling by sizeof — we refuse rather than silently miscompile.
_CHAR_POINTEE_KINDS = frozenset({
    cx.TypeKind.CHAR_S,
    cx.TypeKind.CHAR_U,
    cx.TypeKind.SCHAR,
    cx.TypeKind.UCHAR,
})


def _is_pointer(c: cx.Cursor) -> bool:
    return c.type.get_canonical().kind == cx.TypeKind.POINTER


def _is_char_pointer(c: cx.Cursor) -> bool:
    canon = c.type.get_canonical()
    if canon.kind != cx.TypeKind.POINTER:
        return False
    return canon.get_pointee().kind in _CHAR_POINTEE_KINDS


def _is_char_array(c: cx.Cursor) -> bool:
    """C arrays in expression context decay to pointers; we treat
    `char buf[N]` the same as `char *buf` for arithmetic purposes."""
    canon = c.type.get_canonical()
    if canon.kind not in (cx.TypeKind.CONSTANTARRAY, cx.TypeKind.INCOMPLETEARRAY):
        return False
    return canon.element_type.kind in _CHAR_POINTEE_KINDS


# Quod BinOp ops that yield i1 (comparisons). Returning one in an `int`-typed
# function context requires a synthetic i1→i32 widening.
_I1_BINOPS = frozenset({
    "slt", "sle", "sgt", "sge", "eq", "ne",
    "ult", "ule", "ugt", "uge",
    "feq", "fne", "flt", "fle", "fgt", "fge",
})


def _is_i1_typed(expr: Expr) -> bool:
    if isinstance(expr, BinOp):
        return expr.op in _I1_BINOPS
    if isinstance(expr, (ShortCircuitAnd, ShortCircuitOr)):
        return True
    return False


# Mapping from C compound-assignment operators to the underlying quod
# BinOp.op (used to desugar `x op= y` to `x = x op y` at lift time).
_COMPOUND_ASSIGN_TABLE: dict[str, str] = {
    "+=":  "add",
    "-=":  "sub",
    "*=":  "mul",
    "/=":  "sdiv",
    "%=":  "srem",
    "&=":  "and",
    "|=":  "or",
    "^=":  "xor",
    "<<=": "shl",
    ">>=": "ashr",
}


# Mapping from C operator spellings (read from tokens) to quod BinOp.op.
# Signedness defaults to signed because the supported C subset is int-only.
# Right-shift defaults to `ashr` (arithmetic) for the same reason — C's `>>`
# on a signed operand is implementation-defined; mainstream targets (LLVM
# included) sign-extend, which `ashr` matches. The `lshr` op is reachable
# only once `unsigned int` operands are supported.
_INT_BIN_OP_TABLE: dict[str, str] = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "sdiv",
    "%": "srem",
    "<": "slt",
    "<=": "sle",
    ">": "sgt",
    ">=": "sge",
    "==": "eq",
    "!=": "ne",
    "|": "or",
    "&": "and",
    "^": "xor",
    "<<": "shl",
    ">>": "ashr",
}

# Float BinOps. No bitwise / shift ops — IEEE 754 floats don't support
# them and C forbids them on float operands. `fne` lowers to LLVM `une`
# so `NaN != NaN` returns true; `feq` and the magnitude comparisons use
# ordered preds (false if either operand is NaN).
_FLOAT_BIN_OP_TABLE: dict[str, str] = {
    "+": "fadd",
    "-": "fsub",
    "*": "fmul",
    "/": "fdiv",
    "%": "frem",
    "<": "flt",
    "<=": "fle",
    ">": "fgt",
    ">=": "fge",
    "==": "feq",
    "!=": "fne",
}



class IngestError(Exception):
    """Raised when a C construct falls outside the supported subset."""


def _loc(cursor: cx.Cursor) -> str:
    f = cursor.location.file
    fname = f.name if f else "<unknown>"
    return f"{fname}:{cursor.location.line}:{cursor.location.column}"


def _refuse(cursor: cx.Cursor, why: str) -> "IngestError":
    return IngestError(f"{_loc(cursor)}: {why}")


def _is_int_type(t: cx.Type) -> bool:
    return t.get_canonical().kind == cx.TypeKind.INT


def _quod_type(cursor: cx.Cursor, t: cx.Type) -> I32Type:
    if not _is_int_type(t):
        raise _refuse(cursor, f"only `int` types are supported, got {t.spelling!r}")
    return _I32


_LONG_DOUBLE_REFUSAL = (
    "long double not supported (extended-precision is implementation-"
    "defined and x87-only on the Linux target)"
)


def _local_type(cursor: cx.Cursor, t: cx.Type) -> Type:
    """Map a clang local-var type to a quod Type. Wider than `_quod_type`:
    accepts `int`, `enum`, any pointer (modeled as i8_ptr), and `float` /
    `double`. `long double` refused."""
    canon = t.get_canonical()
    if canon.kind in (cx.TypeKind.INT, cx.TypeKind.ENUM):
        return _I32
    if canon.kind == cx.TypeKind.POINTER:
        return _I8PTR
    if canon.kind == cx.TypeKind.FLOAT:
        return _F32
    if canon.kind == cx.TypeKind.DOUBLE:
        return _F64
    if canon.kind == cx.TypeKind.LONGDOUBLE:
        raise _refuse(cursor, _LONG_DOUBLE_REFUSAL)
    raise _refuse(cursor, f"unsupported local-var type {t.spelling!r} (only `int`, `enum`, pointers, `float`, and `double` are supported)")


def _unwrap(cursor: cx.Cursor) -> cx.Cursor:
    """Skip implicit casts / parens that libclang exposes as UNEXPOSED_EXPR."""
    while cursor.kind in (cx.CursorKind.UNEXPOSED_EXPR, cx.CursorKind.PAREN_EXPR):
        children = list(cursor.get_children())
        if len(children) != 1:
            return cursor
        cursor = children[0]
    return cursor


def _parse_c_float_literal_text(text: str, cursor: cx.Cursor) -> tuple[float, "F32Type | F64Type"]:
    """Parse a C float-literal token into `(value, quod_type)`.

    Suffix `f`/`F` → f32; default and `l`/`L` → f64 (we refuse `l`/`L`).
    Decimal forms parse via `float()`; hex forms (`0x1.8p+1`) via
    `float.fromhex`. Returns the type as the quod F32Type/F64Type
    instance so the caller can build CFloatLit / FloatLit identically.
    """
    if not text:
        raise _refuse(cursor, "float literal with empty token")
    suffix = text[-1] if text[-1] in "fFlL" else ""
    if suffix in ("l", "L"):
        raise _refuse(cursor, "long double literal not supported")
    body = text[:-1] if suffix else text
    lower = body.lower()
    if lower.startswith(("0x", "-0x", "+0x")):
        v = float.fromhex(body)
    else:
        v = float(body)
    ftype = _F32 if suffix in ("f", "F") else _F64
    return v, ftype


def _parse_switch_groups(
    body_cursor: cx.Cursor,
) -> list[tuple[list[cx.Cursor], list[cx.Cursor], str]]:
    """Walk a SWITCH_STMT's body (a COMPOUND_STMT) and group its
    children into (case_values, body_cursors, kind) tuples where kind
    is `"case"` or `"default"`.

    libclang's representation: each CASE_STMT has children
    (value_expr, first_body_stmt). Subsequent statements that belong
    to the same case appear as siblings of the CASE_STMT in the
    enclosing COMPOUND_STMT. Stacked-empty-case labels (`case 2: case
    3: stmt;`) are represented as nested CASE_STMTs.
    """
    if body_cursor.kind != cx.CursorKind.COMPOUND_STMT:
        raise _refuse(body_cursor, "switch body must be a compound statement")
    children = list(body_cursor.get_children())
    groups: list[tuple[list[cx.Cursor], list[cx.Cursor], str]] = []
    cur_values: list[cx.Cursor] = []
    cur_body: list[cx.Cursor] = []
    cur_kind = ""

    def flush():
        if cur_kind:
            groups.append((list(cur_values), list(cur_body), cur_kind))

    for child in children:
        if child.kind == cx.CursorKind.CASE_STMT:
            flush()
            cur_values = []
            cur_body = []
            cur_kind = "case"
            # Walk into nested CASE_STMTs to gather stacked labels.
            inner = child
            while inner.kind == cx.CursorKind.CASE_STMT:
                inner_children = list(inner.get_children())
                if not inner_children:
                    raise _refuse(inner, "case-stmt with no value")
                cur_values.append(inner_children[0])
                if len(inner_children) > 1:
                    nxt = inner_children[1]
                    if nxt.kind == cx.CursorKind.CASE_STMT:
                        inner = nxt
                        continue
                    if nxt.kind == cx.CursorKind.DEFAULT_STMT:
                        # `case 1: default: ...` — non-trivial fallthrough
                        raise _refuse(
                            inner,
                            "fallthrough from `case` into `default` is not "
                            "supported (the supported subset requires each "
                            "case body to end with break/return)"
                        )
                    cur_body.append(nxt)
                break
        elif child.kind == cx.CursorKind.DEFAULT_STMT:
            flush()
            cur_values = []
            cur_body = []
            cur_kind = "default"
            inner_children = list(child.get_children())
            if inner_children:
                first = inner_children[0]
                if first.kind == cx.CursorKind.CASE_STMT:
                    raise _refuse(
                        child,
                        "fallthrough from `default` into `case` is not supported"
                    )
                cur_body.append(first)
        else:
            if not cur_kind:
                raise _refuse(child, "statement before any case label in switch")
            cur_body.append(child)

    flush()
    return groups


def _split_for_children(c: cx.Cursor) -> tuple[
    cx.Cursor | None, cx.Cursor | None, cx.Cursor | None, cx.Cursor,
]:
    """Bucket a FOR_STMT cursor's children into (init, cond, inc, body).

    libclang omits absent slots from the child list, so a 3-child
    FOR_STMT could be missing init, cond, OR inc. We recover the slot
    layout by scanning tokens for the two `;` separators in the for-
    header and bucketing each child by its source-offset position
    relative to those separators.
    """
    children = list(c.get_children())
    if not children:
        raise _refuse(c, "for-stmt with no children")
    body_cursor = children[-1]
    header_children = children[:-1]

    # Find the two `;` token offsets inside the for-header. Tokens
    # before the first `(` and after the matching `)` are out of scope.
    paren_depth = 0
    semicolon_offsets: list[int] = []
    header_start: int | None = None
    header_end: int | None = None
    for tok in c.get_tokens():
        s = tok.spelling
        if s == "(":
            if header_start is None:
                header_start = tok.extent.start.offset
            paren_depth += 1
            continue
        if s == ")":
            paren_depth -= 1
            if paren_depth == 0:
                header_end = tok.extent.start.offset
                break
            continue
        if paren_depth >= 1 and s == ";" and paren_depth == 1:
            semicolon_offsets.append(tok.extent.start.offset)

    if len(semicolon_offsets) != 2 or header_start is None or header_end is None:
        raise _refuse(
            c,
            f"for-stmt: expected exactly two `;` separators in header, "
            f"found {len(semicolon_offsets)} — supported shapes are "
            f"`for (init?; cond?; inc?) body`."
        )

    sem1, sem2 = semicolon_offsets
    init_cursor: cx.Cursor | None = None
    cond_cursor: cx.Cursor | None = None
    inc_cursor: cx.Cursor | None = None
    for child in header_children:
        off = child.extent.start.offset
        if off < sem1:
            init_cursor = child
        elif off < sem2:
            cond_cursor = child
        else:
            inc_cursor = child

    return init_cursor, cond_cursor, inc_cursor, body_cursor


def _binop_token(cursor: cx.Cursor) -> str:
    """Pull the operator token out of a BINARY_OPERATOR cursor's extent.

    libclang's Python bindings don't expose `binary_operator`, so we read
    tokens between the LHS and RHS child extents.
    """
    children = list(cursor.get_children())
    if len(children) != 2:
        raise _refuse(cursor, f"binary operator with {len(children)} children")
    lhs_end = children[0].extent.end.offset
    rhs_start = children[1].extent.start.offset
    # Float ops use the same operator spellings as int ops (`+`, `<`, …),
    # so the int table covers the candidates set without needing the
    # float table.
    candidates = set(_INT_BIN_OP_TABLE.keys()) | {"&&", "||"}
    for tok in cursor.get_tokens():
        off = tok.extent.start.offset
        if lhs_end <= off < rhs_start and tok.spelling in candidates:
            return tok.spelling
    raise _refuse(cursor, "could not identify binary operator from tokens")
