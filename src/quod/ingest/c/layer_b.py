"""Layer-B C ingestion: produces the c-like-quod transcription.

The translator emits mostly core nodes, with `c.*` family extensions
where Layer-B carries information the lift hasn't finished collapsing
(e.g. C `for` becomes `CStyleFor`).
"""

from __future__ import annotations

import ast
from typing import cast

import clang.cindex as cx

from quod.ingest.c.helpers import (
    IngestError,
    _F32,
    _F64,
    _FENV_API_NAMES,
    _FLOAT_BIN_OP_TABLE,
    _FLOAT_TYPE_KINDS,
    _I8PTR,
    _I32,
    _I64,
    _lookup_compound_assign,
    _lookup_int_binop,
    _LONG_DOUBLE_REFUSAL,
    _StructInfo,
    _StructRegistry,
    _VOID,
    _binop_token,
    _fenv_call_refusal,
    _is_char_array,
    _is_char_pointer,
    _is_clang_float,
    _is_i1_typed,
    _is_pointer,
    _int_type_for,
    _local_type,
    _parse_c_float_literal_text,
    _parse_c_int_literal_text,
    _parse_switch_groups,
    _refuse,
    _sizeof_quod_type,
    _split_for_children,
    _struct_name_from_record_type,
    _unwrap,
)
from quod.model import (
    Assign,
    BinOp,
    Block,
    Break,
    Call,
    Cast,
    Continue,
    DoWhile,
    Expr,
    ExprStmt,
    ExternFunction,
    FNeg,
    FloatLit,
    Function,
    If,
    IfExpr,
    IntLit,
    LibcLinkage,
    Let,
    Load,
    LoadField,
    LocalRef,
    Param,
    ParamRef,
    PtrOffset,
    Return,
    ReturnExpr,
    ShortCircuitAnd,
    ShortCircuitOr,
    Statement,
    Store,
    StoreField,
    StringConstant,
    StringRef,
    StructDef,
    StructField,
    StructInit,
    StructType,
    FieldInit,
    FieldRead,
    Type,
    Unreachable,
    While,
    body_always_terminates,
    int_type_signed,
)
from quod.model import CStyleFor


def _walk_cursors(cursor: cx.Cursor):
    """Yield `cursor` and every descendant cursor."""
    yield cursor
    for child in cursor.get_children():
        yield from _walk_cursors(child)


def _check_switch_body(
    switch_cursor: cx.Cursor, body: list[cx.Cursor], *, label: str,
) -> None:
    """Refuse switch-case bodies that don't end with a clear terminator
    (break / return / unreachable). The supported subset is "every case
    ends in break or return" — implicit fallthrough between cases (other
    than shared-empty-case stacking) is refused per the deferred
    design question on UB-handling. Also refuse non-trailing `break;`
    inside a case body (it'd be a switch-break that the if-else-if
    rewrite can't represent without dedicated control flow)."""
    if not body:
        # Empty case body — only valid if this group is shared-empty
        # with the next group (handled by the parser nesting). A flushed
        # empty body here means the case has no statements at all,
        # which would fall through to the next case implicitly. Refuse.
        raise _refuse(
            switch_cursor,
            f"`{label}` body is empty — implicit fallthrough to the next "
            f"case is not supported (use shared-empty-case stacking like "
            f"`case 1: case 2: body break;` for shared bodies)."
        )
    last = body[-1]
    if last.kind not in (
        cx.CursorKind.BREAK_STMT, cx.CursorKind.RETURN_STMT,
    ):
        raise _refuse(
            last,
            f"`{label}` body's last statement is {last.kind.name} — every "
            f"case body must end with `break;` or `return ...;` "
            f"(implicit fallthrough is not supported)."
        )
    # Refuse any non-trailing `break` inside the body — would be a
    # switch-break the if-else-if rewrite can't encode.
    for stmt in body[:-1]:
        for descendant in _walk_cursors(stmt):
            if descendant.kind == cx.CursorKind.BREAK_STMT:
                raise _refuse(
                    descendant,
                    f"`break;` inside a switch case body (other than as "
                    f"the trailing terminator) is not supported."
                )


def _build_switch_chain(
    scrutinee: Expr,
    groups: list[tuple[list[cx.Cursor], list[cx.Cursor], str]],
    translator: "_FunctionTranslator",
) -> Statement:
    """Build the layer-B if-else-if chain for a parsed switch. Drops
    trailing `break;` from each case body (it's a switch-break which
    becomes implicit at the if-else-if level)."""
    # Translate each group's body, dropping the trailing `break;`.
    translated_groups: list[tuple[list[Expr], tuple[Statement, ...], str]] = []
    for value_cursors, body_cursors, kind in groups:
        # Drop trailing break (return stays).
        trimmed_body = body_cursors
        if trimmed_body and trimmed_body[-1].kind == cx.CursorKind.BREAK_STMT:
            trimmed_body = trimmed_body[:-1]
        body_stmts: list[Statement] = []
        for s in trimmed_body:
            body_stmts.extend(translator.stmts(s))
        # Translate the values.
        values = [translator.expr(v) for v in value_cursors]
        translated_groups.append((values, tuple(body_stmts), kind))

    # Build the chain right-to-left.
    default_body: tuple[Statement, ...] = ()
    case_groups: list[tuple[list[Expr], tuple[Statement, ...]]] = []
    for values, body, kind in translated_groups:
        if kind == "default":
            default_body = body
        else:
            case_groups.append((values, body))

    # Innermost else is the default body (or empty Block).
    else_block = Block(
        id=translator._state.mint_block_id(),
        stmts=default_body,
    )
    for values, body in reversed(case_groups):
        # Build the cond: scrutinee == v[0] || scrutinee == v[1] || ...
        cond = _eq_or_chain(scrutinee, values)
        then_block = Block(
            id=translator._state.mint_block_id(),
            stmts=body,
        )
        else_block = Block(
            id=translator._state.mint_block_id(),
            stmts=(If(cond=cond, then_body=then_block, else_body=else_block),),
        )
    # The outermost If is the first wrapper of `else_block`'s only stmt.
    if not else_block.stmts:
        # No cases — degenerate switch with only a default. Just emit
        # the default body inline (wrapped in an If with `true` cond
        # would change observability of the scrutinee).
        return ExprStmt(value=scrutinee)  # evaluate scrutinee for side effects
    return else_block.stmts[0]


def _eq_or_chain(scrutinee: Expr, values: list[Expr]) -> Expr:
    """Build `scrutinee == v[0] || scrutinee == v[1] || ...` from a
    list of case values. Single-value case is just `scrutinee == v[0]`."""
    eqs = [BinOp(op="eq", lhs=scrutinee, rhs=v) for v in values]
    cond: Expr = eqs[0]
    for e in eqs[1:]:
        cond = ShortCircuitOr(lhs=cond, rhs=e)
    return cond


class _ProgramState:
    """Program-level state shared across function translators.

    Tracks string-literal deduplication and external-call signatures we've
    inferred from libclang's resolved declarations. Built up as functions
    are translated; consumed by `ingest_c` to populate the Program.
    """

    def __init__(
        self, string_prefix: str = "", *,
        structs: "_StructRegistry | None" = None,
    ) -> None:
        from quod.ingest.c.helpers import _EMPTY_STRUCT_REGISTRY
        # Dedupe by literal value so identical strings collapse to one constant.
        self._string_by_value: dict[str, str] = {}
        self.constants: list[StringConstant] = []
        # Map external-symbol name → its inferred ExternFunction. First sighting
        # wins; later calls just look up by name.
        self.externs: dict[str, ExternFunction] = {}
        # Per-TU struct registry — collected by the driver in pass 1
        # before any function bodies are visited. Threaded through
        # `_FunctionTranslator` so MEMBER_REF and INIT_LIST_EXPR
        # dispatch can look up struct field metadata.
        self.structs = structs if structs is not None else _EMPTY_STRUCT_REGISTRY
        # Per-source prefix for generated string-constant names. Without it,
        # two ingests both produce `.str.0` referring to different content,
        # which collides on merge into a single program.json. Convention:
        # CLI threads in the source path's sanitized stem.
        self._string_prefix = string_prefix
        # Sequential counter for Block IDs so two ingests of the same C
        # source produce identical output. Block IDs are inert today
        # (no edges yet) but determinism makes test fixtures pinnable.
        self._block_counter = 0
        # Counter for layer-B CStyleFor IDs — deterministic so two
        # ingests of the same source produce byte-identical output.
        self._cstyle_for_counter = 0

    def mint_block_id(self) -> str:
        self._block_counter += 1
        return f"@blk_c_{self._block_counter}"

    def mint_function_id(self, name: str) -> str:
        # Function IDs are stable across re-ingest of the same source by
        # using the C function's spelling. C linkers reject duplicate
        # symbols across the same translation unit, so the spelling is
        # already unique per ingest.
        return f"@fn_c_{name}"

    def mint_cstyle_for_id(self) -> str:
        # Layer-B CStyleFor IDs need a deterministic counter so two
        # ingests of the same source produce byte-identical output.
        # The model's Field(default_factory) uses uuid, which is fine
        # for hand-authored programs but breaks ingest determinism.
        self._cstyle_for_counter += 1
        return f"@cfor_general_c_{self._cstyle_for_counter}"

    def intern_string(self, value: str) -> StringRef:
        if value in self._string_by_value:
            return StringRef(name=self._string_by_value[value])
        if self._string_prefix:
            name = f".str.{self._string_prefix}.{len(self.constants)}"
        else:
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

    Same coverage as `_local_type` (every signed/unsigned integer width,
    enum, pointer, float, double) plus `void` in return position.
    Structs, function pointers (other than as opaque i8_ptr), and
    `long double` all refuse.
    """
    canon = t.get_canonical()
    if is_return and canon.kind == cx.TypeKind.VOID:
        return _VOID
    return _local_type(cursor, t)


class _FunctionTranslator:
    """Per-function state: tracks param/local names so we can disambiguate
    ParamRef vs LocalRef. Locals introduced by Let are added as we go.

    Holds a reference to shared _ProgramState for string-literal interning
    and extern-signature recording. `is_void` controls whether a bare
    `return;` statement is accepted (only valid in void-returning functions).
    """

    def __init__(
        self, params: tuple[str, ...], program_state: _ProgramState,
        *, is_void: bool = False,
    ) -> None:
        self._params = set(params)
        self._locals: set[str] = set()
        self._state = program_state
        self._is_void = is_void

    def _ref(self, cursor: cx.Cursor, name: str) -> Expr:
        if name in self._params:
            return ParamRef(name=name)
        if name in self._locals:
            return LocalRef(name=name)
        raise _refuse(cursor, f"unknown identifier {name!r} (only params/locals are supported)")

    def _qty(self, cursor: cx.Cursor, t: cx.Type) -> Type:
        """Wrapper around `_local_type` that threads the per-TU
        struct registry so `struct Foo` references resolve to
        `StructType('Foo')` instead of refusing."""
        return _local_type(cursor, t, structs=self._state.structs)

    def _is_arrow_member_ref(
        self, member_cursor: cx.Cursor, base_cursor: cx.Cursor,
    ) -> bool:
        """For a MEMBER_REF_EXPR cursor, determine whether the access
        is `p->x` (arrow) or `p.x` (dot). libclang's Python bindings
        expose neither directly, so we check the base's type: a
        pointer-typed base is `->`, otherwise `.`."""
        canon = base_cursor.type.get_canonical()
        return canon.kind == cx.TypeKind.POINTER

    def _struct_name_for_pointer_cursor(
        self, cursor: cx.Cursor,
    ) -> str | None:
        """For an expression cursor whose type is `struct Foo *`,
        return `'Foo'` if the struct is in the per-TU registry.
        Returns None for non-struct-pointer types."""
        canon = cursor.type.get_canonical()
        if canon.kind != cx.TypeKind.POINTER:
            return None
        pointee = canon.get_pointee().get_canonical()
        if pointee.kind != cx.TypeKind.RECORD:
            return None
        name = _struct_name_from_record_type(pointee)
        if name is None or name not in self._state.structs:
            return None
        return name

    def _maybe_implicit_cast(self, cursor: cx.Cursor) -> Expr | None:
        """If `cursor` is an UNEXPOSED_EXPR / PAREN_EXPR whose `cursor.type`
        differs at the quod level from the inner cursor's `type`, return
        a `Cast(inner, target_quod_type)`. Otherwise return None and let
        the normal `_unwrap` path proceed.

        Fires on:

        - **Float-involving casts** (`int → double`, `double → int`,
          `float → double`, `double → float`).
        - **Width-changing int casts** — C's "usual arithmetic
          conversions" promote `char`/`short` to `int` before
          arithmetic, and assignments to wider/narrower integer types
          insert implicit conversions. clang materializes these as
          `ImplicitCastExpr` cursors (exposed as `UNEXPOSED_EXPR`); we
          surface them as `Cast` so the IR is explicit about widening
          (sext/zext) and narrowing (trunc).
        - **Sign-only casts** (signed↔unsigned same-width) — also
          surfaced because the C source distinguishes signedness, even
          if LLVM's i32 doesn't; `Cast` lowers same-width cases to a
          no-op but the lift-check pairs the explicit cast.

        Same-type wrappers (clang inserts these for lvalue→rvalue
        conversions on ints) fall through to `_unwrap`.
        """
        if cursor.kind not in (cx.CursorKind.UNEXPOSED_EXPR, cx.CursorKind.PAREN_EXPR):
            return None
        children = list(cursor.get_children())
        if len(children) != 1:
            return None
        inner = children[0]
        # Only pre-empt `_unwrap` when both sides have a quod type and
        # they differ. Anything else (struct, unsupported, …) flows
        # through to the standard path where it'll refuse with the
        # right message.
        try:
            outer_qty = self._qty(cursor, cursor.type)
            inner_qty = self._qty(inner, inner.type)
        except IngestError:
            return None
        if type(outer_qty) is type(inner_qty):
            # No quod-level type change — `_unwrap` is correct.
            return None
        return Cast(value=self.expr(inner), target_type=outer_qty)

    def expr(self, cursor: cx.Cursor) -> Expr:
        # Detect type-changing implicit casts BEFORE unwrapping. clang
        # surfaces them as `UNEXPOSED_EXPR(target_type) → inner(source_type)`
        # — for ints (where source and target lower to the same LLVM type)
        # the existing `_unwrap` peels through transparently, which is
        # correct. For float-involving casts the LLVM types differ and we
        # must materialize a `Cast(value, target_type)`.
        cast_node = self._maybe_implicit_cast(cursor)
        if cast_node is not None:
            return cast_node

        c = _unwrap(cursor)
        k = c.kind

        if k == cx.CursorKind.INTEGER_LITERAL:
            tokens = [t.spelling for t in c.get_tokens()]
            if not tokens:
                raise _refuse(c, "integer literal with no tokens")
            qty = _int_type_for(c.type)
            if qty is None:
                raise _refuse(c, f"integer literal of unsupported type {c.type.spelling!r}")
            return IntLit(type=qty, value=_parse_c_int_literal_text(tokens[0], c))

        if k == cx.CursorKind.FLOATING_LITERAL:
            tokens = [t.spelling for t in c.get_tokens()]
            if not tokens:
                raise _refuse(c, "float literal with no tokens")
            bits, ftype = _parse_c_float_literal_text(tokens[0], c)
            return FloatLit(type=ftype, bits=bits)

        if k in (cx.CursorKind.CSTYLE_CAST_EXPR, cx.CursorKind.CXX_FUNCTIONAL_CAST_EXPR):
            target_qty = self._qty(c, c.type)
            inner = None
            for child in c.get_children():
                if child.kind != cx.CursorKind.TYPE_REF:
                    inner = child
            if inner is None:
                raise _refuse(c, "explicit cast with no expression operand")
            return Cast(value=self.expr(inner), target_type=target_qty)

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
            # Postfix `i++` / `i--` lay the operator in the last token;
            # prefix forms put it first. Sniff both ends so the
            # expression-position refusal below has the right operator.
            if tokens[-1] in ("++", "--") and tokens[0] not in ("++", "--"):
                op = tokens[-1]
            else:
                op = tokens[0]
            # `&buf[k]` and `&buf[k+m]` are pointer arithmetic — handle before
            # we recurse into the array-subscript child (which we'd otherwise
            # have to lower as a load).
            if op == "&":
                inner = _unwrap(children[0])
                if inner.kind == cx.CursorKind.ARRAY_SUBSCRIPT_EXPR:
                    return self._array_address_of(c, inner)
                raise _refuse(c, "address-of only supported for array subscripts (e.g. `&buf[k]`)")
            # Determine operand type *before* recursion so we can route
            # `-x` on float operands to FNeg (IEEE sign-bit flip) rather
            # than the int `0 - x` desugaring.
            operand_is_float = _is_clang_float(children[0].type)
            operand_int_qty = _int_type_for(children[0].type)
            inner_expr = self.expr(children[0])
            if op == "-":
                if operand_is_float:
                    if isinstance(inner_expr, FloatLit):
                        # Constant-fold: -1.5 → FloatLit with sign bit
                        # flipped. FNeg of a literal is just the literal
                        # with the sign bit toggled — this preserves
                        # NaN payloads and avoids the IEEE corner where
                        # `0.0 - 0.0 = +0.0` rather than -0.0.
                        from quod.model import F32Type as _F32T
                        sign_bit = 1 << (31 if isinstance(inner_expr.type, _F32T) else 63)
                        return FloatLit(
                            type=inner_expr.type,
                            bits=inner_expr.bits ^ sign_bit,
                        )
                    return FNeg(operand=inner_expr)
                if isinstance(inner_expr, IntLit):
                    return IntLit(type=inner_expr.type, value=-inner_expr.value)
                zero_qty = operand_int_qty if operand_int_qty is not None else _I32
                return BinOp(op="sub", lhs=IntLit(type=zero_qty, value=0), rhs=inner_expr)
            if op == "+":
                return inner_expr
            if op == "*":
                # `*p` rvalue — typed pointer dereference. The
                # cursor's type is the pointee (already the loaded
                # value's type). Refuse `void *` (no width to load)
                # and pointers whose pointee can't be expressed as a
                # scalar quod type.
                pointee_canon = children[0].type.get_canonical()
                if pointee_canon.kind != cx.TypeKind.POINTER:
                    raise _refuse(
                        c,
                        f"`*` operator on non-pointer type "
                        f"{children[0].type.spelling!r}",
                    )
                pointee = pointee_canon.get_pointee()
                if pointee.kind == cx.TypeKind.VOID:
                    raise _refuse(
                        c,
                        "`*p` on void* has no value type to load — "
                        "cast to a typed pointer first",
                    )
                load_qty = self._qty(c, c.type)
                return Load(ptr=inner_expr, type=load_qty)
            if op == "!":
                # C's logical-not is i1-typed (`!x` == 1 iff x == 0). Lower
                # to `eq(x, 0)` so the result type matches its uses
                # (cond positions, etc.). The lift-checker pairs CUnary("!")
                # with this BinOp shape.
                zero_qty = operand_int_qty if operand_int_qty is not None else _I32
                return BinOp(op="eq", lhs=inner_expr, rhs=IntLit(type=zero_qty, value=0))
            if op == "~":
                # One's complement is `x ^ -1` over two's-complement
                # ints. Lift-checker pairs CUnary("~") with this shape.
                neg1_qty = operand_int_qty if operand_int_qty is not None else _I32
                return BinOp(op="xor", lhs=inner_expr, rhs=IntLit(type=neg1_qty, value=-1))
            if op in ("++", "--"):
                raise _refuse(
                    c,
                    f"{op!r} in expression position is not yet supported "
                    f"(only bare-statement and for-loop inc positions are supported)"
                )
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
            # Route int vs float ops by inspecting clang's operand types
            # *before* recursion. If both operands are float-typed, use
            # the float op table (fadd/.../feq/...). Mixed types should
            # already be normalized by clang's implicit-cast insertion;
            # if we see them here, that's a bug in implicit-cast handling.
            both_float = (
                _is_clang_float(children[0].type)
                and _is_clang_float(children[1].type)
            )
            # Operand-typed dispatch: clang's "usual arithmetic conversions"
            # have already promoted both operands to a common type by the
            # time we see them, so the LHS operand's quod type tells us
            # whether to emit signed or unsigned variants of /, %, <, <=,
            # >, >=, >>.
            lhs_qty = _int_type_for(children[0].type) if not both_float else None
            signed = True if lhs_qty is None else int_type_signed(lhs_qty)
            lhs = self.expr(children[0])
            rhs = self.expr(children[1])
            if tok == "&&":
                return ShortCircuitAnd(lhs=lhs, rhs=rhs)
            if tok == "||":
                return ShortCircuitOr(lhs=lhs, rhs=rhs)
            if both_float:
                op = _FLOAT_BIN_OP_TABLE.get(tok)
                if op is not None:
                    return BinOp(op=cast(any, op), lhs=lhs, rhs=rhs)
                raise _refuse(c, f"unsupported float binary operator {tok!r}")
            op = _lookup_int_binop(tok, signed=signed)
            if op is not None:
                return BinOp(op=cast(any, op), lhs=lhs, rhs=rhs)
            raise _refuse(c, f"unsupported int binary operator {tok!r}")


        if k == cx.CursorKind.CALL_EXPR:
            children = list(c.get_children())
            # First child is the callee (a DECL_REF_EXPR after unwrapping); rest are args.
            if not children:
                raise _refuse(c, "call expr with no children")
            callee = _unwrap(children[0])
            if callee.kind != cx.CursorKind.DECL_REF_EXPR:
                raise _refuse(c, "indirect / function-pointer calls not supported")
            # Belt-and-braces against `extern int fesetround(int);` declared
            # without `<fenv.h>`. The TU-level include refusal in driver.py
            # covers the standard case.
            if callee.spelling in _FENV_API_NAMES:
                raise _refuse(c, _fenv_call_refusal(callee.spelling))
            # Record signature for later extern construction (no-op if the
            # callee is one of our own functions).
            self._state.record_extern(c, callee.spelling, callee.referenced)
            args = tuple(self.expr(a) for a in children[1:])
            return Call(function=callee.spelling, args=args)

        if k == cx.CursorKind.MEMBER_REF_EXPR:
            # `p.x` or `p->x` — base is the first child; arrow vs dot
            # is read from the source-range tokens between base-end
            # and the field name.
            children = list(c.get_children())
            if len(children) != 1:
                raise _refuse(c, f"member-ref with {len(children)} children")
            base_cursor = _unwrap(children[0])
            field_name = c.spelling
            if not field_name:
                raise _refuse(c, "member-ref with no field name")
            arrow = self._is_arrow_member_ref(c, base_cursor)
            field_qty = self._qty(c, c.type)
            if arrow:
                struct_name = self._struct_name_for_pointer_cursor(base_cursor)
                if struct_name is None:
                    raise _refuse(
                        c,
                        f"`->` on non-struct-pointer base "
                        f"({base_cursor.type.spelling!r})",
                    )
                return LoadField(
                    ptr=self.expr(base_cursor),
                    struct_type=struct_name,
                    name=field_name,
                )
            return FieldRead(value=self.expr(base_cursor), name=field_name)

        if k == cx.CursorKind.INIT_LIST_EXPR:
            # `(struct Foo){...}` or the RHS of a `struct Foo p = {...};`
            # decl. The cursor's type tells us which struct.
            canon = c.type.get_canonical()
            if canon.kind != cx.TypeKind.RECORD:
                raise _refuse(
                    c,
                    f"non-struct init-list ({c.type.spelling!r}) is not "
                    f"yet supported (only struct init lists are)",
                )
            struct_name = _struct_name_from_record_type(canon)
            if struct_name is None or struct_name not in self._state.structs:
                raise _refuse(
                    c,
                    f"unknown struct type for init-list "
                    f"({c.type.spelling!r})",
                )
            info = self._state.structs.get(struct_name)
            assert info is not None  # by the check above
            child_cursors = list(c.get_children())
            if len(child_cursors) > len(info.fields):
                raise _refuse(
                    c,
                    f"too many initialisers for struct {struct_name!r} "
                    f"({len(child_cursors)} for {len(info.fields)} fields)",
                )
            field_inits: list[FieldInit] = []
            # Positional initialisers: pair in declaration order.
            # Designated initialisers (`{.x = 1}`) require a different
            # cursor shape (DESIGNATED_INIT_EXPR) and are refused for
            # now if encountered.
            for i, child in enumerate(child_cursors):
                if child.kind == cx.CursorKind.UNEXPOSED_EXPR:
                    pass  # let _unwrap handle it inside expr
                if child.kind == cx.CursorKind.MEMBER_REF:
                    raise _refuse(
                        child,
                        "designated struct initialisers (`{.field = "
                        "value}`) are not yet supported",
                    )
                field_name = info.field_names[i]
                field_inits.append(FieldInit(
                    name=field_name, value=self.expr(child),
                ))
            return StructInit(
                type=struct_name, fields=tuple(field_inits),
            )

        if k == cx.CursorKind.ARRAY_SUBSCRIPT_EXPR:
            # `arr[k]` rvalue — load at a byte-strided offset.
            # `&arr[k]` is handled separately in the UNARY `&` arm
            # before reaching here. The cursor's type is the element
            # type (already the loaded value's type).
            children = list(c.get_children())
            if len(children) != 2:
                raise _refuse(c, f"array subscript with {len(children)} children")
            arr_c, idx_c = _unwrap(children[0]), _unwrap(children[1])
            elem_size = self._pointer_element_size(c, arr_c)
            elem_qty = self._qty(c, c.type)
            return Load(
                ptr=PtrOffset(
                    base=self.expr(arr_c),
                    offset=self._byte_stride_offset(idx_c, elem_size),
                ),
                type=elem_qty,
            )

        if k == cx.CursorKind.CONDITIONAL_OPERATOR:
            # `cond ? a : b` lifts to layer-B IfExpr. The condition's
            # type must lower to i1 — comparisons / short-circuits do
            # so naturally; an integer cond gets the C "nonzero ⇒ true"
            # widening, which we represent as an explicit `cond != 0`
            # binop so the layer-B i1-type is visible to the validator.
            children = list(c.get_children())
            if len(children) != 3:
                raise _refuse(c, f"ternary with {len(children)} children")
            cond_expr = self.expr(children[0])
            if not _is_i1_typed(cond_expr):
                cond_expr = BinOp(
                    op="ne", lhs=cond_expr,
                    rhs=IntLit(type=_I32, value=0),
                )
            then_expr = self.expr(children[1])
            else_expr = self.expr(children[2])
            return IfExpr(
                cond=cond_expr,
                then_value=then_expr,
                else_value=else_expr,
            )

        raise _refuse(c, f"unsupported expression kind: {k.name}")

    def stmt(self, cursor: cx.Cursor) -> Statement:
        """Translate a cursor that's expected to produce exactly one
        statement. Refuses if the cursor expands to multiple (e.g. a
        multi-declarator `int a, b;` in a for-loop init slot)."""
        out = self.stmts(cursor)
        if len(out) != 1:
            raise _refuse(
                cursor,
                f"this position expects a single statement, got "
                f"{len(out)} (multi-declarator declarations are only "
                f"supported as top-level statements)"
            )
        return out[0]

    def stmts(self, cursor: cx.Cursor) -> tuple[Statement, ...]:
        c = cursor
        k = c.kind

        if k == cx.CursorKind.RETURN_STMT:
            children = list(c.get_children())
            if not children:
                # Bare `return;` is valid only in void-returning functions.
                # In a non-void function, falling off the end (or bare-
                # returning) is C99 §6.9.1/12 UB. Refusing is the
                # conservative response — see project memory on
                # how-to-handle-invalid-C.
                if not self._is_void:
                    raise _refuse(
                        c,
                        "bare `return;` is only valid in void-returning "
                        "functions; this function returns a non-void type."
                    )
                return (Return(),)
            value = self.expr(children[0])
            if _is_i1_typed(value):
                # C's `return cond;` implicitly widens i1→int. quod has no
                # zext node, so synthesize the equivalent branch: if cond
                # then return 1 else return 0.
                return (If(
                    cond=value,
                    then_body=Block(
                        id=self._state.mint_block_id(),
                        stmts=(ReturnExpr(value=IntLit(type=_I32, value=1)),),
                    ),
                    else_body=Block(
                        id=self._state.mint_block_id(),
                        stmts=(ReturnExpr(value=IntLit(type=_I32, value=0)),),
                    ),
                ),)
            return (ReturnExpr(value=value),)

        if k == cx.CursorKind.IF_STMT:
            children = list(c.get_children())
            if len(children) not in (2, 3):
                raise _refuse(c, f"if-stmt with {len(children)} children")
            cond = self.expr(children[0])
            then_body = self._block(children[1])
            else_body = (
                self._block(children[2]) if len(children) == 3
                else Block(id=self._state.mint_block_id())
            )
            return (If(cond=cond, then_body=then_body, else_body=else_body),)

        if k == cx.CursorKind.WHILE_STMT:
            children = list(c.get_children())
            if len(children) != 2:
                raise _refuse(c, f"while-stmt with {len(children)} children")
            cond = self.expr(children[0])
            body = self._block(children[1])
            return (While(cond=cond, body=body),)

        if k == cx.CursorKind.DO_STMT:
            # DO_STMT children are (body, cond) in source order.
            children = list(c.get_children())
            if len(children) != 2:
                raise _refuse(c, f"do-while with {len(children)} children")
            body = self._block(children[0])
            cond = self.expr(children[1])
            return (DoWhile(body=body, cond=cond),)

        if k == cx.CursorKind.FOR_STMT:
            # Layer B: emit `c.for_general` (CStyleFor). The c-family
            # lowering pass rewrites this to `Let + While + Assign`.
            #
            # libclang exposes FOR_STMT children in source order with
            # missing slots simply omitted from the children list, so
            # `for (;;) body` produces just one child (the body) and
            # `for (init;;inc) body` produces three. We bucket each
            # child by its source position relative to the two `;`
            # tokens to recover which slot it fills.
            init_cursor, cond_cursor, inc_cursor, body_cursor = _split_for_children(c)
            init_stmt = self.stmt(init_cursor) if init_cursor is not None else None
            cond_expr = self.expr(cond_cursor) if cond_cursor is not None else None
            inc_stmt = self.stmt(inc_cursor) if inc_cursor is not None else None
            body = self._block(body_cursor)
            return (CStyleFor(
                id=self._state.mint_cstyle_for_id(),
                init=init_stmt, cond=cond_expr, inc=inc_stmt, body=body,
            ),)

        if k == cx.CursorKind.DECL_STMT:
            children = list(c.get_children())
            if not children:
                raise _refuse(c, "decl-stmt with no children")
            # Multi-declarator: `int a, b, c;` becomes N consecutive Lets.
            # libclang exposes each declarator as its own VAR_DECL child.
            # An uninitialized declarator (`int x;`) lifts to
            # `Let(name, type, init=None)` — the validator's definite-init
            # analysis refuses any program where such a local is read
            # before being written.
            lets: list[Statement] = []
            for decl in children:
                if decl.kind != cx.CursorKind.VAR_DECL:
                    raise _refuse(decl, f"only var declarations supported, got {decl.kind.name}")
                local_ty = self._qty(decl, decl.type)
                # libclang's VAR_DECL children include both type refs
                # (TYPE_REF, NAMESPACE_REF, …) and the optional
                # initializer expression. Filter to only expression-
                # kind cursors; the last one (if any) is the init.
                init_cursors = [
                    ic for ic in decl.get_children() if ic.kind.is_expression()
                ]
                init_expr = self.expr(init_cursors[-1]) if init_cursors else None
                self._locals.add(decl.spelling)
                lets.append(Let(name=decl.spelling, type=local_ty, init=init_expr))
            return tuple(lets)

        if k == cx.CursorKind.BINARY_OPERATOR:
            # Bare assignment as a statement: `x = expr;`,
            # `*p = expr;`, or `arr[k] = expr;`.
            tokens = [t.spelling for t in c.get_tokens()]
            if "=" in tokens and "==" not in tokens:
                children = list(c.get_children())
                lhs = _unwrap(children[0])
                # Pointer-deref store: `*p = v;`
                if lhs.kind == cx.CursorKind.UNARY_OPERATOR:
                    lhs_tokens = [t.spelling for t in lhs.get_tokens()]
                    if lhs_tokens and lhs_tokens[0] == "*":
                        deref_children = list(lhs.get_children())
                        if len(deref_children) != 1:
                            raise _refuse(lhs, "deref-store with non-1 deref children")
                        # Validate the pointee type up front; reuse the
                        # rvalue path's refusal messages by routing
                        # through Load and discarding (cheap, and keeps
                        # error reporting uniform).
                        pointee_canon = deref_children[0].type.get_canonical()
                        if pointee_canon.kind != cx.TypeKind.POINTER:
                            raise _refuse(
                                lhs,
                                f"`*` operator on non-pointer type "
                                f"{deref_children[0].type.spelling!r}",
                            )
                        if pointee_canon.get_pointee().kind == cx.TypeKind.VOID:
                            raise _refuse(
                                lhs,
                                "`*p = …` on void* has no value type to "
                                "store — cast to a typed pointer first",
                            )
                        ptr_expr = self.expr(deref_children[0])
                        value = self.expr(children[1])
                        return (Store(ptr=ptr_expr, value=value),)
                # Field-arrow store: `p->x = v;`. Field-dot store
                # (`p.x = v;` on a by-value local struct) is refused
                # — quod's model has StoreField for heap structs but
                # no by-value field-rebind primitive.
                if lhs.kind == cx.CursorKind.MEMBER_REF_EXPR:
                    member_children = list(lhs.get_children())
                    if len(member_children) != 1:
                        raise _refuse(lhs, "member-ref store with non-1 base children")
                    base_cursor = _unwrap(member_children[0])
                    field_name = lhs.spelling
                    arrow = self._is_arrow_member_ref(lhs, base_cursor)
                    if not arrow:
                        raise _refuse(
                            lhs,
                            "by-value struct field assignment "
                            f"({base_cursor.spelling}.{field_name} = ...) is "
                            f"not supported; use a pointer to the struct "
                            f"and `->` (e.g. `p->{field_name} = ...`)",
                        )
                    struct_name = self._struct_name_for_pointer_cursor(base_cursor)
                    if struct_name is None:
                        raise _refuse(
                            lhs,
                            f"`->` store on non-struct-pointer base "
                            f"({base_cursor.type.spelling!r})",
                        )
                    value = self.expr(children[1])
                    return (StoreField(
                        ptr=self.expr(base_cursor),
                        struct_type=struct_name,
                        name=field_name,
                        value=value,
                    ),)
                # Subscript store: `arr[k] = v;`
                if lhs.kind == cx.CursorKind.ARRAY_SUBSCRIPT_EXPR:
                    sub_children = list(lhs.get_children())
                    if len(sub_children) != 2:
                        raise _refuse(
                            lhs,
                            f"subscript-store with {len(sub_children)} children",
                        )
                    arr_c, idx_c = _unwrap(sub_children[0]), _unwrap(sub_children[1])
                    elem_size = self._pointer_element_size(lhs, arr_c)
                    value = self.expr(children[1])
                    return (Store(
                        ptr=PtrOffset(
                            base=self.expr(arr_c),
                            offset=self._byte_stride_offset(idx_c, elem_size),
                        ),
                        value=value,
                    ),)
                if lhs.kind != cx.CursorKind.DECL_REF_EXPR:
                    raise _refuse(lhs, "only simple `name = expr` assignment supported")
                if lhs.spelling not in self._locals:
                    raise _refuse(lhs, f"cannot assign to {lhs.spelling!r} (must be a local declared with `int`)")
                value = self.expr(children[1])
                return (Assign(name=lhs.spelling, value=value),)
            return (ExprStmt(value=self.expr(c)),)

        if k == cx.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
            # `x op= y;` desugars to `x = x op y;`. The lift-checker
            # pairs the layer-A CCompoundAssign with this Assign+BinOp
            # shape via _COMPOUND_ASSIGN_TABLE.
            children = list(c.get_children())
            if len(children) != 2:
                raise _refuse(c, f"compound assignment with {len(children)} children")
            lhs = _unwrap(children[0])
            if lhs.kind != cx.CursorKind.DECL_REF_EXPR:
                raise _refuse(lhs, "only simple `name op= expr` assignment supported")
            if lhs.spelling not in self._locals:
                raise _refuse(
                    lhs,
                    f"cannot assign to {lhs.spelling!r} (compound assignment "
                    f"requires a local declared with `int`; assignment to "
                    f"parameters is not supported)"
                )
            op = c.spelling
            # Operand-typed dispatch — pick signed or unsigned variant
            # based on the target local's quod type. (Compound assignment
            # in C requires the target's type to drive the operation;
            # the RHS is converted to that type by clang's implicit
            # cast.)
            target_qty = _int_type_for(lhs.type)
            signed = True if target_qty is None else int_type_signed(target_qty)
            translated = _lookup_compound_assign(op, signed=signed)
            if translated is None:
                raise _refuse(c, f"unsupported compound-assignment operator {op!r}")
            value = self.expr(children[1])
            return (Assign(
                name=lhs.spelling,
                value=BinOp(
                    op=cast(any, translated),
                    lhs=LocalRef(name=lhs.spelling),
                    rhs=value,
                ),
            ),)

        if k == cx.CursorKind.UNARY_OPERATOR:
            # Statement-position `i++;`, `++i;`, `i--;`, `--i;` — desugar
            # to `i = i + 1;` / `i = i - 1;`. Pre and post lift identically
            # because the expression value is discarded (statement
            # position); the source-form distinction is preserved at
            # layer A only. The lift-checker pairs CIncrementStmt against
            # this Assign+BinOp shape.
            #
            # Other unary operators don't reach this dispatcher because
            # their statement form is refused upstream (a bare `-x;` etc.
            # falls into the BINARY_OPERATOR branch's "expression-as-
            # statement" refusal); only `++`/`--` are statement-shaped.
            children = list(c.get_children())
            if len(children) != 1:
                raise _refuse(c, "unary statement with non-1 children")
            tokens = [t.spelling for t in c.get_tokens()]
            if not tokens:
                raise _refuse(c, "unary statement with no tokens")
            op = tokens[0] if tokens[0] in ("++", "--") else tokens[-1]
            if op not in ("++", "--"):
                raise _refuse(c, f"unsupported unary statement operator {op!r}")
            target = _unwrap(children[0])
            if target.kind != cx.CursorKind.DECL_REF_EXPR:
                raise _refuse(target, f"only simple `name{op}` / `{op}name` increment supported")
            if target.spelling not in self._locals:
                raise _refuse(
                    target,
                    f"cannot assign to {target.spelling!r} (increment requires "
                    f"a local declared with `int`; assignment to parameters "
                    f"is not supported)"
                )
            translated = "add" if op == "++" else "sub"
            target_qty = self._qty(target, target.type)
            return (Assign(
                name=target.spelling,
                value=BinOp(
                    op=cast(any, translated),
                    lhs=LocalRef(name=target.spelling),
                    rhs=IntLit(type=target_qty, value=1),
                ),
            ),)

        if k == cx.CursorKind.CALL_EXPR:
            return (ExprStmt(value=self.expr(c)),)

        if k == cx.CursorKind.BREAK_STMT:
            return (Break(),)

        if k == cx.CursorKind.CONTINUE_STMT:
            return (Continue(),)

        if k == cx.CursorKind.SWITCH_STMT:
            # Lift to a layer-B if-else-if chain. Each case becomes
            # `if (scrutinee == val[0] || scrutinee == val[1] || ...) body`,
            # nested in the previous case's else-branch. Default (if
            # present) becomes the innermost else.
            children = list(c.get_children())
            if len(children) != 2:
                raise _refuse(c, f"switch with {len(children)} children")
            scrutinee_b = self.expr(children[0])
            groups = _parse_switch_groups(children[1])
            # Validate each group's body ends with break/return/unreachable
            # and contains no stray `break` (which would be a switch-break
            # we can't trivially handle in the if-else-if rewrite).
            for values, body, kind in groups:
                _check_switch_body(c, body, label=kind)
            return (_build_switch_chain(scrutinee_b, groups, self),)

        raise _refuse(c, f"unsupported statement kind: {k.name}")

    def _block(self, cursor: cx.Cursor) -> Block:
        block_id = self._state.mint_block_id()
        if cursor.kind == cx.CursorKind.COMPOUND_STMT:
            stmts: list[Statement] = []
            for s in cursor.get_children():
                stmts.extend(self.stmts(s))
            return Block(id=block_id, stmts=tuple(stmts))
        # Single-statement bodies (e.g. `if (c) return 0;`) are valid C and
        # libclang exposes them as the statement directly.
        return Block(id=block_id, stmts=self.stmts(cursor))

    def _maybe_pointer_add(
        self, c: cx.Cursor, children: list[cx.Cursor],
    ) -> Expr | None:
        """Recognize `p + n` as pointer arithmetic.

        Returns a `quod.ptr_offset` Expr when one operand is pointer-
        typed (or a char array, which decays to char*) and the other
        is an integer offset; None otherwise (caller falls back to
        the integer-arithmetic path).

        Element stride: `char *p + n` uses byte stride (multiplier=1,
        the existing fast path). `int *p + n` uses element stride
        (multiplier = sizeof(int) = 4): the layer-B side becomes
        `PtrOffset(p, mul(widen64(n), 4))`. Refuses pointer-minus-
        pointer (not a valid C expression) and pointer pointee types
        we can't compute a quod sizeof for (function pointers,
        struct pointers in Stage A).
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

        elem_size = self._pointer_element_size(c, ptr_c)
        return PtrOffset(
            base=self.expr(ptr_c),
            offset=self._byte_stride_offset(off_c, elem_size),
        )

    def _array_address_of(self, outer: cx.Cursor, sub: cx.Cursor) -> Expr:
        """Translate `&arr[k]` (UNARY `&` of ARRAY_SUBSCRIPT_EXPR) into
        `quod.ptr_offset(arr, k * sizeof(elem))`. Char-stride bases
        keep the byte-offset shortcut; wider element types route
        through the byte-stride multiplication."""
        children = list(sub.get_children())
        if len(children) != 2:
            raise _refuse(sub, f"array subscript with {len(children)} children")
        arr_c, idx_c = _unwrap(children[0]), _unwrap(children[1])
        elem_size = self._pointer_element_size(outer, arr_c)
        return PtrOffset(
            base=self.expr(arr_c),
            offset=self._byte_stride_offset(idx_c, elem_size),
        )

    def _pointer_element_size(
        self, refusal_cursor: cx.Cursor, ptr_cursor: cx.Cursor,
    ) -> int:
        """Byte size of the element pointed to by `ptr_cursor`'s C
        pointer/array type. Drives byte-stride pointer arithmetic.
        Refuses pointer/array types whose pointee/element isn't a
        scalar quod type (struct, function, void)."""
        canon = ptr_cursor.type.get_canonical()
        if canon.kind == cx.TypeKind.POINTER:
            pointee = canon.get_pointee()
        elif canon.kind in (cx.TypeKind.CONSTANTARRAY, cx.TypeKind.INCOMPLETEARRAY):
            pointee = canon.element_type
        else:
            raise _refuse(
                refusal_cursor,
                f"pointer arithmetic on non-pointer type {ptr_cursor.type.spelling!r}",
            )
        try:
            elem_qty = _local_type(refusal_cursor, pointee)
        except IngestError:
            raise _refuse(
                refusal_cursor,
                f"pointer arithmetic on {ptr_cursor.type.spelling!r}: "
                f"unsupported element type {pointee.spelling!r}",
            )
        try:
            return _sizeof_quod_type(elem_qty)
        except ValueError:
            raise _refuse(
                refusal_cursor,
                f"pointer arithmetic on {ptr_cursor.type.spelling!r}: "
                f"no byte size for element type {pointee.spelling!r}",
            )

    def _byte_stride_offset(
        self, off_cursor: cx.Cursor, elem_size: int,
    ) -> Expr:
        """Compute the byte offset for a typed pointer add:
        `index * sizeof(element)`. For `elem_size == 1` (char), this
        folds to the plain `_i64_offset` (existing char-stride
        behavior). For larger sizes, multiplies the i64-typed offset
        by the element size as an i64 literal."""
        if elem_size == 1:
            return self._i64_offset(off_cursor)
        idx_i64 = self._i64_offset(off_cursor)
        if isinstance(idx_i64, IntLit):
            # Constant-fold: `&buf[3]` on int* becomes IntLit(I64, 12).
            return IntLit(type=_I64, value=idx_i64.value * elem_size)
        return BinOp(
            op="mul",
            lhs=idx_i64,
            rhs=IntLit(type=_I64, value=elem_size),
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
            return IntLit(type=_I64, value=_parse_c_int_literal_text(tokens[0], c))
        # Negation: `&buf[-1]` or `p + (-1)` — accept literal-after-unary-minus.
        if c.kind == cx.CursorKind.UNARY_OPERATOR:
            tokens = [t.spelling for t in c.get_tokens()]
            inner = list(c.get_children())
            if (
                tokens and tokens[0] == "-"
                and len(inner) == 1
                and _unwrap(inner[0]).kind == cx.CursorKind.INTEGER_LITERAL
            ):
                lit_cursor = _unwrap(inner[0])
                lit_tokens = [t.spelling for t in lit_cursor.get_tokens()]
                return IntLit(type=_I64, value=-_parse_c_int_literal_text(lit_tokens[0], lit_cursor))
        # Variable offset: translate as an int expression and cast to i64.
        # Source signedness is implicit in the value's quod type (typical
        # C ints lift to i32 / i64 — signed — and the resolver propagates
        # types through arithmetic).
        return Cast(value=self.expr(cursor), target_type=_I64)


def _translate_function(
    cursor: cx.Cursor, source_path, state: _ProgramState,
) -> Function:
    # Reuse the extern type-mapping path so int / void / float / double
    # all resolve through the same dispatch (and `long double` refuses
    # uniformly). The is_void flag drives later void-return handling.
    return_type: Type = _extern_type(cursor, cursor.result_type, is_return=True)
    is_void = isinstance(return_type, type(_VOID))

    params: list[Param] = []
    body_cursor: cx.Cursor | None = None
    for child in cursor.get_children():
        if child.kind == cx.CursorKind.PARM_DECL:
            params.append(Param(
                name=child.spelling,
                type=_local_type(child, child.type, structs=state.structs),
            ))
        elif child.kind == cx.CursorKind.COMPOUND_STMT:
            body_cursor = child

    if body_cursor is None:
        raise _refuse(cursor, "function has no body (forward declarations are skipped, not ingested)")

    translator = _FunctionTranslator(
        tuple(p.name for p in params), state, is_void=is_void,
    )
    body_list: list[Statement] = []
    for s in body_cursor.get_children():
        body_list.extend(translator.stmts(s))
    body = tuple(body_list)

    # Faithful translation of C fall-through. C99 §5.1.2.2.3 defines falling
    # off `main` as `return 0;` — synthesize that for int-returning `main`.
    # Void-returning functions may fall through (the standard treats this
    # as an implicit `return;`) — append an explicit `Return()` so layer C
    # carries a terminator. Any other int-returning function falling off
    # the end is UB (§6.9.1/12) — represent it explicitly with
    # `Unreachable` so analysis can flag the path.
    if not body_always_terminates(body):
        if is_void:
            body = body + (Return(),)
        elif cursor.spelling == "main":
            body = body + (ReturnExpr(value=IntLit(type=_I32, value=0)),)
        else:
            body = body + (Unreachable(),)

    note = f"ingested from {source_path.name}:{cursor.location.line}"
    return Function(
        id=state.mint_function_id(cursor.spelling),
        name=cursor.spelling,
        params=tuple(params),
        return_type=return_type,
        body=Block(id=state.mint_block_id(), stmts=body),
        notes=(note,),
    )



def _struct_def_layer_b(info: _StructInfo) -> StructDef:
    """Convert a per-TU `_StructInfo` into a layer-B `StructDef`.
    Pairs with `_struct_def_layer_a` so source_units and the layer-B
    program share the same struct definitions, just at different
    layers. Field order is preserved."""
    return StructDef(
        name=info.name,
        fields=tuple(StructField(name=n, type=t) for n, t in info.fields),
    )
