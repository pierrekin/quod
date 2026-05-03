"""C source → quod Program (v1: int-typed subset).

Walks a libclang AST and emits quod model nodes. The supported subset is
deliberately narrow: int-only types, arithmetic / comparison / boolean
binops, if / while / return, calls between ingested functions, locals via
plain int declarations. Anything outside the subset raises IngestError
with the offending source location.

Macros / #include / #ifdef are handled by clang's preprocessor before we
see the AST — we ingest one build configuration of the source. We filter
cursors by source file so headers don't pollute the program.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import cast

import clang.cindex as cx

from quod.model import (
    Assign,
    BinOp,
    Call,
    ExprStmt,
    Expr,
    ExternFunction,
    Function,
    I8PtrType,
    I32Type,
    I64Type,
    If,
    IntLit,
    LibcLinkage,
    Let,
    LocalRef,
    Param,
    ParamRef,
    Program,
    PtrOffset,
    ReturnExpr,
    ShortCircuitAnd,
    ShortCircuitOr,
    StringConstant,
    StringRef,
    Statement,
    Type,
    While,
    Widen,
)


_I32 = I32Type()
_I64 = I64Type()
_I8PTR = I8PtrType()


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
_I1_BINOPS = frozenset({"slt", "sle", "sgt", "sge", "eq", "ne", "ult", "ule", "ugt", "uge"})


def _is_i1_typed(expr: Expr) -> bool:
    if isinstance(expr, BinOp):
        return expr.op in _I1_BINOPS
    if isinstance(expr, (ShortCircuitAnd, ShortCircuitOr)):
        return True
    return False


# Mapping from C operator spellings (read from tokens) to quod BinOp.op.
# Signedness defaults to signed since v1 only supports `int`.
_BIN_OP_TABLE: dict[str, str] = {
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
}


class IngestError(Exception):
    """Raised when a C construct falls outside the v1 subset."""


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
        raise _refuse(cursor, f"only `int` types are supported in v1, got {t.spelling!r}")
    return _I32


def _local_type(cursor: cx.Cursor, t: cx.Type) -> Type:
    """Map a clang local-var type to a quod Type. Wider than `_quod_type`:
    accepts `int`, `enum`, and any pointer (modeled as i8_ptr)."""
    canon = t.get_canonical()
    if canon.kind in (cx.TypeKind.INT, cx.TypeKind.ENUM):
        return _I32
    if canon.kind == cx.TypeKind.POINTER:
        return _I8PTR
    raise _refuse(cursor, f"unsupported local-var type {t.spelling!r} (only `int`, `enum`, and pointers in v1)")


def _unwrap(cursor: cx.Cursor) -> cx.Cursor:
    """Skip implicit casts / parens that libclang exposes as UNEXPOSED_EXPR."""
    while cursor.kind in (cx.CursorKind.UNEXPOSED_EXPR, cx.CursorKind.PAREN_EXPR):
        children = list(cursor.get_children())
        if len(children) != 1:
            return cursor
        cursor = children[0]
    return cursor


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
    candidates = set(_BIN_OP_TABLE.keys()) | {"&&", "||"}
    for tok in cursor.get_tokens():
        off = tok.extent.start.offset
        if lhs_end <= off < rhs_start and tok.spelling in candidates:
            return tok.spelling
    raise _refuse(cursor, "could not identify binary operator from tokens")


class _ProgramState:
    """Program-level state shared across function translators.

    Tracks string-literal deduplication and external-call signatures we've
    inferred from libclang's resolved declarations. Built up as functions
    are translated; consumed by `ingest_c` to populate the Program.
    """

    def __init__(self) -> None:
        # Dedupe by literal value so identical strings collapse to one constant.
        self._string_by_value: dict[str, str] = {}
        self.constants: list[StringConstant] = []
        # Map external-symbol name → its inferred ExternFunction. First sighting
        # wins; later calls just look up by name.
        self.externs: dict[str, ExternFunction] = {}

    def intern_string(self, value: str) -> StringRef:
        if value in self._string_by_value:
            return StringRef(name=self._string_by_value[value])
        name = f".str.{len(self.constants)}"
        self._string_by_value[value] = name
        self.constants.append(StringConstant(name=name, value=value))
        return StringRef(name=name)

    def record_extern(self, cursor: cx.Cursor, name: str, decl: cx.Cursor | None) -> None:
        """Record an extern at a call site.

        Refuses if `decl` is provided but the signature can't be represented
        — silently producing a stub extern would have us emit IR that calls
        `@foo()` while passing args, which is a quiet miscompilation. If the
        callee genuinely can't be resolved (rare; usually means a missing
        `#include`), we keep the all-i32 default — the build step will fail
        clearly when the symbol can't be linked.
        """
        if name in self.externs:
            return
        if decl is None:
            # Symbol couldn't be resolved (rare — usually a missing #include).
            # We still emit the extern so the call type-checks; the linker
            # surfaces the missing symbol later. Provenance is libc-class:
            # we ingested from C source / a C header.
            self.externs[name] = ExternFunction(name=name, linkage=LibcLinkage())
            return
        # IngestError propagates up — the caller fails the whole ingest.
        self.externs[name] = _build_extern_from_decl(cursor, decl)


def _build_extern_from_decl(call_cursor: cx.Cursor, decl: cx.Cursor) -> ExternFunction:
    """Build an ExternFunction from a libclang FUNCTION_DECL cursor.

    Raises IngestError with the call-site location if any element of the
    signature can't be represented in quod's type system (struct, float,
    wider int, etc.). `call_cursor` is used purely for error attribution
    so users see *where* the call was that triggered the failure.
    """
    fn_type = decl.type
    if fn_type.kind != cx.TypeKind.FUNCTIONPROTO:
        raise _refuse(call_cursor, f"call to {decl.spelling!r}: declaration has no function prototype (KR-style or otherwise unsupported)")
    param_types = tuple(_extern_type(call_cursor, t) for t in fn_type.argument_types())
    return_type = _extern_type(call_cursor, fn_type.get_result(), is_return=True)
    return ExternFunction(
        name=decl.spelling,
        param_types=param_types,
        return_type=return_type,
        varargs=fn_type.is_function_variadic(),
        linkage=LibcLinkage(),
    )


def _extern_type(cursor: cx.Cursor, t: cx.Type, *, is_return: bool = False) -> Type:
    """Map a clang Type to a quod Type, for use in extern signatures.

    Wider than `_quod_type` (which only allows int):

      - `int` and `enum` → I32. Enums are int-typed at IR level.
      - any pointer → I8Ptr. LLVM has opaque pointers, so `char*`, `void*`,
        and `CURL*` are all the same type at IR level — modeling them as
        i8_ptr is honest, not a hack.
      - `void` (return only) → I32. Stand-in until VoidType lands; callers
        must discard the return value, since the runtime ABI doesn't put
        anything in the return register and we'd be reading garbage.

    Floats, wider ints, structs, function pointers (other than as opaque
    i8_ptr) all refuse — quod can't represent them yet.
    """
    canon = t.get_canonical()
    if canon.kind in (cx.TypeKind.INT, cx.TypeKind.ENUM):
        return _I32
    if canon.kind == cx.TypeKind.POINTER:
        return _I8PTR
    if is_return and canon.kind == cx.TypeKind.VOID:
        # TODO: replace with VoidType once quod gains one — i32 stand-in
        # works only because expr-stmt calls discard the return value.
        return _I32
    raise _refuse(cursor, f"unsupported extern signature type {t.spelling!r}")


class _FunctionTranslator:
    """Per-function state: tracks param/local names so we can disambiguate
    ParamRef vs LocalRef. Locals introduced by Let are added as we go.

    Holds a reference to shared _ProgramState for string-literal interning
    and extern-signature recording.
    """

    def __init__(self, params: tuple[str, ...], program_state: _ProgramState) -> None:
        self._params = set(params)
        self._locals: set[str] = set()
        self._state = program_state

    def _ref(self, cursor: cx.Cursor, name: str) -> Expr:
        if name in self._params:
            return ParamRef(name=name)
        if name in self._locals:
            return LocalRef(name=name)
        raise _refuse(cursor, f"unknown identifier {name!r} (only params/locals are supported)")

    def expr(self, cursor: cx.Cursor) -> Expr:
        c = _unwrap(cursor)
        k = c.kind

        if k == cx.CursorKind.INTEGER_LITERAL:
            tokens = [t.spelling for t in c.get_tokens()]
            if not tokens:
                raise _refuse(c, "integer literal with no tokens")
            return IntLit(type=_I32, value=int(tokens[0], 0))

        if k == cx.CursorKind.STRING_LITERAL:
            tokens = [t.spelling for t in c.get_tokens()]
            if not tokens:
                raise _refuse(c, "string literal with no tokens")
            try:
                # C string-literal syntax is a subset of Python's, so
                # literal_eval handles \n, \t, \\, \", \xNN, etc.
                value = ast.literal_eval(tokens[0])
            except (ValueError, SyntaxError) as e:
                raise _refuse(c, f"could not decode string literal: {e}")
            if not isinstance(value, str):
                raise _refuse(c, f"string literal decoded to non-str ({type(value).__name__})")
            return self._state.intern_string(value)

        if k == cx.CursorKind.DECL_REF_EXPR:
            referenced = c.referenced
            if referenced is not None and referenced.kind == cx.CursorKind.ENUM_CONSTANT_DECL:
                # Header-defined enum constant (e.g. CURLOPT_URL = 10002).
                # libclang resolves the value for us; emit it as a plain
                # int literal since the source-level name doesn't survive.
                return IntLit(type=_I32, value=referenced.enum_value)
            return self._ref(c, c.spelling)

        if k == cx.CursorKind.UNARY_OPERATOR:
            children = list(c.get_children())
            if len(children) != 1:
                raise _refuse(c, "unary operator with non-1 children")
            tokens = [t.spelling for t in c.get_tokens()]
            if not tokens:
                raise _refuse(c, "unary operator with no tokens")
            op = tokens[0]
            # `&buf[k]` and `&buf[k+m]` are pointer arithmetic — handle before
            # we recurse into the array-subscript child (which we'd otherwise
            # have to lower as a load).
            if op == "&":
                inner = _unwrap(children[0])
                if inner.kind == cx.CursorKind.ARRAY_SUBSCRIPT_EXPR:
                    return self._array_address_of(c, inner)
                raise _refuse(c, "address-of only supported for array subscripts (e.g. `&buf[k]`)")
            inner_expr = self.expr(children[0])
            if op == "-":
                if isinstance(inner_expr, IntLit):
                    return IntLit(type=_I32, value=-inner_expr.value)
                return BinOp(op="sub", lhs=IntLit(type=_I32, value=0), rhs=inner_expr)
            if op == "+":
                return inner_expr
            raise _refuse(c, f"unsupported unary operator {op!r}")

        if k == cx.CursorKind.BINARY_OPERATOR:
            tok = _binop_token(c)
            children = list(c.get_children())
            # Pointer arithmetic must be detected before recursing, since the
            # quod Expr nodes don't carry the C type info we need to tell
            # `p + 1` (ptr_offset) from `n + 1` (regular add).
            if tok == "+":
                ptr_arith = self._maybe_pointer_add(c, children)
                if ptr_arith is not None:
                    return ptr_arith
            lhs = self.expr(children[0])
            rhs = self.expr(children[1])
            if tok == "&&":
                return ShortCircuitAnd(lhs=lhs, rhs=rhs)
            if tok == "||":
                return ShortCircuitOr(lhs=lhs, rhs=rhs)
            if tok in _BIN_OP_TABLE:
                return BinOp(op=cast(any, _BIN_OP_TABLE[tok]), lhs=lhs, rhs=rhs)
            raise _refuse(c, f"unsupported binary operator {tok!r}")


        if k == cx.CursorKind.CALL_EXPR:
            children = list(c.get_children())
            # First child is the callee (a DECL_REF_EXPR after unwrapping); rest are args.
            if not children:
                raise _refuse(c, "call expr with no children")
            callee = _unwrap(children[0])
            if callee.kind != cx.CursorKind.DECL_REF_EXPR:
                raise _refuse(c, "indirect / function-pointer calls not supported")
            # Record signature for later extern construction (no-op if the
            # callee is one of our own functions).
            self._state.record_extern(c, callee.spelling, callee.referenced)
            args = tuple(self.expr(a) for a in children[1:])
            return Call(function=callee.spelling, args=args)

        raise _refuse(c, f"unsupported expression kind: {k.name}")

    def stmt(self, cursor: cx.Cursor) -> Statement:
        c = cursor
        k = c.kind

        if k == cx.CursorKind.RETURN_STMT:
            children = list(c.get_children())
            if not children:
                raise _refuse(c, "bare `return;` not supported (function must return int)")
            inner = _unwrap(children[0])
            if inner.kind == cx.CursorKind.INTEGER_LITERAL:
                tokens = [t.spelling for t in inner.get_tokens()]
                return ReturnExpr(value=IntLit(type=_I32, value=int(tokens[0], 0)))
            value = self.expr(children[0])
            if _is_i1_typed(value):
                # C's `return cond;` implicitly widens i1→int. quod has no
                # zext node, so synthesize the equivalent branch: if cond
                # then return 1 else return 0.
                return If(
                    cond=value,
                    then_body=(ReturnExpr(value=IntLit(type=_I32, value=1)),),
                    else_body=(ReturnExpr(value=IntLit(type=_I32, value=0)),),
                )
            return ReturnExpr(value=value)

        if k == cx.CursorKind.IF_STMT:
            children = list(c.get_children())
            if len(children) not in (2, 3):
                raise _refuse(c, f"if-stmt with {len(children)} children")
            cond = self.expr(children[0])
            then_body = self._block(children[1])
            else_body = self._block(children[2]) if len(children) == 3 else ()
            return If(cond=cond, then_body=then_body, else_body=else_body)

        if k == cx.CursorKind.WHILE_STMT:
            children = list(c.get_children())
            if len(children) != 2:
                raise _refuse(c, f"while-stmt with {len(children)} children")
            cond = self.expr(children[0])
            body = self._block(children[1])
            return While(cond=cond, body=body)

        if k == cx.CursorKind.DECL_STMT:
            children = list(c.get_children())
            if len(children) != 1:
                raise _refuse(c, "multi-declarator declarations not supported")
            decl = children[0]
            if decl.kind != cx.CursorKind.VAR_DECL:
                raise _refuse(decl, f"only var declarations supported, got {decl.kind.name}")
            local_ty = _local_type(decl, decl.type)
            init_children = list(decl.get_children())
            if not init_children:
                raise _refuse(decl, "uninitialized locals not supported (require `T x = …;`)")
            # The last child is the initializer; earlier children are type refs we ignore.
            init_expr = self.expr(init_children[-1])
            self._locals.add(decl.spelling)
            return Let(name=decl.spelling, type=local_ty, init=init_expr)

        if k == cx.CursorKind.BINARY_OPERATOR:
            # Bare assignment as a statement: `x = expr;`
            tokens = [t.spelling for t in c.get_tokens()]
            if "=" in tokens and "==" not in tokens:
                children = list(c.get_children())
                lhs = _unwrap(children[0])
                if lhs.kind != cx.CursorKind.DECL_REF_EXPR:
                    raise _refuse(lhs, "only simple `name = expr` assignment supported")
                if lhs.spelling not in self._locals:
                    raise _refuse(lhs, f"cannot assign to {lhs.spelling!r} (must be a local declared with `int`)")
                value = self.expr(children[1])
                return Assign(name=lhs.spelling, value=value)
            return ExprStmt(value=self.expr(c))

        if k == cx.CursorKind.CALL_EXPR:
            return ExprStmt(value=self.expr(c))

        raise _refuse(c, f"unsupported statement kind: {k.name}")

    def _block(self, cursor: cx.Cursor) -> tuple[Statement, ...]:
        if cursor.kind == cx.CursorKind.COMPOUND_STMT:
            return tuple(self.stmt(s) for s in cursor.get_children())
        # Single-statement bodies (e.g. `if (c) return 0;`) are valid C and
        # libclang exposes them as the statement directly.
        return (self.stmt(cursor),)

    def _maybe_pointer_add(
        self, c: cx.Cursor, children: list[cx.Cursor],
    ) -> Expr | None:
        """Recognize `p + n` as pointer arithmetic when `p` is char-pointer-typed.

        Returns a `quod.ptr_offset` Expr when one operand is a char* (or char
        array, which decays to char*) and the other is an integer offset; None
        otherwise (caller falls back to the integer-arithmetic path).

        Refuses unsupported pointer arithmetic outright (non-char pointee,
        pointer minus pointer) so we never silently miscompile the byte stride.
        """
        lhs_c, rhs_c = _unwrap(children[0]), _unwrap(children[1])
        lhs_is_ptr = _is_pointer(lhs_c) or _is_char_array(lhs_c)
        rhs_is_ptr = _is_pointer(rhs_c) or _is_char_array(rhs_c)

        if not (lhs_is_ptr or rhs_is_ptr):
            return None
        if lhs_is_ptr and rhs_is_ptr:
            raise _refuse(c, "pointer-plus-pointer is not a valid C expression")

        if lhs_is_ptr:
            ptr_c, off_c = lhs_c, rhs_c
        else:
            ptr_c, off_c = rhs_c, lhs_c

        if not (_is_char_pointer(ptr_c) or _is_char_array(ptr_c)):
            raise _refuse(
                c,
                f"pointer arithmetic on {ptr_c.type.spelling!r}: only char* "
                f"(byte stride) is supported. Cast to (char*) or compute "
                f"the byte offset explicitly."
            )

        return PtrOffset(
            base=self.expr(ptr_c),
            offset=self._i64_offset(off_c),
        )

    def _array_address_of(self, outer: cx.Cursor, sub: cx.Cursor) -> Expr:
        """Translate `&arr[k]` (UNARY `&` of ARRAY_SUBSCRIPT_EXPR) into
        `quod.ptr_offset(arr, k)`. Same pointee restriction as `_maybe_pointer_add`."""
        children = list(sub.get_children())
        if len(children) != 2:
            raise _refuse(sub, f"array subscript with {len(children)} children")
        arr_c, idx_c = _unwrap(children[0]), _unwrap(children[1])
        if not (_is_char_pointer(arr_c) or _is_char_array(arr_c)):
            raise _refuse(
                outer,
                f"&{arr_c.spelling}[…]: only char arrays / char* bases are "
                f"supported (got {arr_c.type.spelling!r})"
            )
        return PtrOffset(
            base=self.expr(arr_c),
            offset=self._i64_offset(idx_c),
        )

    def _i64_offset(self, cursor: cx.Cursor) -> Expr:
        """Translate an offset expression into an i64-typed Expr suitable
        for `quod.ptr_offset`.

        Literal `int`s become i64 IntLits directly (no IR cost). Variable
        offsets — e.g. a loop counter — get wrapped in `quod.widen(…, i64)`,
        which lowers to a single `sext` instruction. The C `int` type is
        signed, so sign-extension matches C's promotion rules.
        """
        c = _unwrap(cursor)
        if c.kind == cx.CursorKind.INTEGER_LITERAL:
            tokens = [t.spelling for t in c.get_tokens()]
            if not tokens:
                raise _refuse(c, "integer literal with no tokens")
            return IntLit(type=_I64, value=int(tokens[0], 0))
        # Negation: `&buf[-1]` or `p + (-1)` — accept literal-after-unary-minus.
        if c.kind == cx.CursorKind.UNARY_OPERATOR:
            tokens = [t.spelling for t in c.get_tokens()]
            inner = list(c.get_children())
            if (
                tokens and tokens[0] == "-"
                and len(inner) == 1
                and _unwrap(inner[0]).kind == cx.CursorKind.INTEGER_LITERAL
            ):
                lit_tokens = [t.spelling for t in _unwrap(inner[0]).get_tokens()]
                return IntLit(type=_I64, value=-int(lit_tokens[0], 0))
        # Variable offset: translate as an int expression and widen to i64.
        return Widen(value=self.expr(cursor), target=_I64, signed=True)


def _translate_function(
    cursor: cx.Cursor, source_path: Path, state: _ProgramState,
) -> Function:
    if not _is_int_type(cursor.result_type):
        raise _refuse(cursor, f"only `int`-returning functions are supported, got {cursor.result_type.spelling!r}")

    params: list[Param] = []
    body_cursor: cx.Cursor | None = None
    for child in cursor.get_children():
        if child.kind == cx.CursorKind.PARM_DECL:
            _quod_type(child, child.type)
            params.append(Param(name=child.spelling, type=_I32))
        elif child.kind == cx.CursorKind.COMPOUND_STMT:
            body_cursor = child

    if body_cursor is None:
        raise _refuse(cursor, "function has no body (forward declarations are skipped, not ingested)")

    translator = _FunctionTranslator(tuple(p.name for p in params), state)
    body = tuple(translator.stmt(s) for s in body_cursor.get_children())

    note = f"ingested from {source_path.name}:{cursor.location.line}"
    return Function(
        name=cursor.spelling,
        params=tuple(params),
        return_type=_I32,
        body=body,
        notes=(note,),
    )


def _detect_resource_dir() -> str | None:
    """Ask `clang` where its resource directory is.

    The PyPI `libclang` package ships a `libclang.so` but not clang's
    resource headers (stddef.h, stdarg.h, …). Without `-resource-dir`, even
    `#include <stdio.h>` fails because stdio internally includes stddef.h.
    Run once per ingest; ignore failures (caller can override via
    clang_args).
    """
    try:
        out = subprocess.run(
            ["clang", "-print-resource-dir"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def ingest_c(path: Path, *, clang_args: tuple[str, ...] = ()) -> Program:
    """Parse a C file and return a quod Program.

    Anything outside the supported v1 subset raises IngestError with the
    offending source location. Only declarations whose primary location is
    in `path` itself are translated — header-included declarations are
    skipped, but their types/symbols are visible to the parser, so calls
    into stdlib resolve to externs with proper signatures.
    """
    path = path.resolve()
    if not path.exists():
        raise IngestError(f"{path}: no such file")

    index = cx.Index.create()
    args: tuple[str, ...] = ("-x", "c")
    resource_dir = _detect_resource_dir()
    if resource_dir is not None:
        args = (*args, f"-resource-dir={resource_dir}")
    args = (*args, *clang_args)
    tu = index.parse(str(path), args=args)
    if not tu:
        raise IngestError(f"{path}: clang failed to parse file")

    diags = [d for d in tu.diagnostics if d.severity >= cx.Diagnostic.Error]
    if diags:
        msg = "; ".join(f"{d.location.file}:{d.location.line}: {d.spelling}" for d in diags)
        raise IngestError(f"{path}: parse errors: {msg}")

    state = _ProgramState()
    functions: list[Function] = []
    defined_names: set[str] = set()

    for cursor in tu.cursor.get_children():
        loc_file = cursor.location.file
        if loc_file is None or Path(loc_file.name).resolve() != path:
            continue
        if cursor.kind != cx.CursorKind.FUNCTION_DECL:
            raise _refuse(cursor, f"top-level {cursor.kind.name} not supported (only functions in v1)")
        if not cursor.is_definition():
            continue
        functions.append(_translate_function(cursor, path, state))
        defined_names.add(cursor.spelling)

    # Drop externs that turned out to be locally defined functions (e.g. a
    # call to one ingested function from another doesn't need an extern).
    externs = tuple(e for name, e in state.externs.items() if name not in defined_names)

    return Program(
        constants=tuple(state.constants),
        functions=tuple(functions),
        externs=externs,
    )


def _parse_translation_unit(path: Path, *, language: str, clang_args: tuple[str, ...]) -> cx.TranslationUnit:
    """Shared libclang entry point. `language` is `c` or `c-header`."""
    if not path.exists():
        raise IngestError(f"{path}: no such file")
    index = cx.Index.create()
    args: tuple[str, ...] = ("-x", language)
    resource_dir = _detect_resource_dir()
    if resource_dir is not None:
        args = (*args, f"-resource-dir={resource_dir}")
    args = (*args, *clang_args)
    tu = index.parse(str(path), args=args)
    if not tu:
        raise IngestError(f"{path}: clang failed to parse file")
    diags = [d for d in tu.diagnostics if d.severity >= cx.Diagnostic.Error]
    if diags:
        msg = "; ".join(f"{d.location.file}:{d.location.line}: {d.spelling}" for d in diags)
        raise IngestError(f"{path}: parse errors: {msg}")
    return tu


def ingest_header(
    path: Path, *, clang_args: tuple[str, ...] = (),
) -> tuple[tuple[ExternFunction, ...], tuple[str, ...]]:
    """Parse a C header and emit ExternFunction declarations.

    Walks every FUNCTION_DECL reachable from the translation unit and
    builds an `ExternFunction` for each whose signature fits the supported
    type system (`int`, `char*`, varargs). Returns:

        (externs_built, names_skipped)

    Names that appeared as function declarations but had unsupported
    signatures (struct params, floats, wider ints, etc.) are returned in
    `names_skipped` so the caller can show a count or list. Symbols
    declared multiple times (e.g. via redeclaration) are deduplicated by
    name; first sighting wins.
    """
    path = path.resolve()
    tu = _parse_translation_unit(path, language="c-header", clang_args=clang_args)

    externs: list[ExternFunction] = []
    skipped: list[str] = []
    seen: set[str] = set()

    for cursor in tu.cursor.walk_preorder():
        if cursor.kind != cx.CursorKind.FUNCTION_DECL:
            continue
        # Headers may contain `static inline` definitions (e.g. from
        # libc's transitive includes). Those have bodies — skip them, we
        # only want pure declarations to expose as externs.
        if cursor.is_definition():
            continue
        name = cursor.spelling
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            ext = _build_extern_from_decl(cursor, cursor)
        except IngestError:
            # Bulk-import path is tolerant: a header full of unsupported
            # signatures shouldn't refuse the whole ingest. Caller gets
            # a tally of what was skipped.
            skipped.append(name)
            continue
        externs.append(ext)

    return tuple(externs), tuple(skipped)
