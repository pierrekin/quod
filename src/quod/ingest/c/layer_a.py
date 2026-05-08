"""Layer-A C ingestion: produces the c-source subtree.

A second walk over the same libclang AST that produces the c-source
subtree (`Program.source_units`). Mostly mirrors the layer-B walker;
splits out into its own class to keep concerns clean — layer A is
inert structural data, layer B is the c-like-quod transcription that
eventually lowers to LLVM.

The two walkers share the libclang AST as the source of truth; every
layer-A node is paired with a layer-B node by being produced from the
same FUNCTION_DECL cursor (function-level pairing — finer-grained
pairing can be added when edges need to grow per-statement).
"""

from __future__ import annotations

import ast
from pathlib import Path

import clang.cindex as cx

from quod.ingest.c.helpers import (
    _COMPOUND_ASSIGN_TABLE,
    _FENV_API_NAMES,
    _LONG_DOUBLE_REFUSAL,
    _StructInfo,
    _StructRegistry,
    _binop_token,
    _fenv_call_refusal,
    _int_type_for,
    _parse_c_float_literal_text,
    _parse_c_int_literal_text,
    _parse_switch_groups,
    _refuse,
    _split_for_children,
    _struct_name_from_record_type,
    _unwrap,
)
from quod.model import (
    CAddressOf,
    CArraySubscript,
    CAssign,
    CBinOp,
    CBreak,
    CCall,
    CCompoundAssign,
    CContinue,
    CCast,
    CDeref,
    CDerefStore,
    CDoWhile,
    CEnumConstRef,
    CExpr,
    CExprStmt,
    CField,
    CFieldArrow,
    CFieldArrowStore,
    CFieldInit,
    CFieldRead,
    CFloatLit,
    CFn,
    CFor,
    CForInit,
    CIf,
    CIncrementStmt,
    CIntLit,
    CMultiVarDecl,
    CNamedType,
    CParam,
    CPointerType,
    CReturn,
    CStmt,
    CStringLit,
    CStructDef,
    CStructInit,
    CStructType,
    CSubscriptStore,
    CSwitch,
    CSwitchCase,
    CTernary,
    CType,
    CUnary,
    CVarDecl,
    CVarRef,
    CWhile,
)


class _LayerATranslator:
    """Walks a function body and returns the layer-A subtree.

    Pure mechanical transcription: no semantic decisions, no widening,
    no synthesized fall-through. The supported C subset matches the
    layer-B translator's; unsupported constructs raise `IngestError`
    with the original source location.

    IDs are minted with a per-translator counter (one translator per
    function) so re-ingest of the same source produces byte-identical
    output. The default `Field(default_factory=...)` on the model
    nodes uses uuid4, which is fine for hand-authored programs but
    breaks ingest determinism.
    """

    def __init__(
        self, fn_name: str, *,
        structs: _StructRegistry | None = None,
    ) -> None:
        from quod.ingest.c.helpers import _EMPTY_STRUCT_REGISTRY
        self._fn_name = fn_name
        self._counters: dict[str, int] = {}
        self._structs = structs if structs is not None else _EMPTY_STRUCT_REGISTRY

    def _mint(self, prefix: str) -> str:
        n = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = n
        return f"@{prefix}_c_{self._fn_name}_{n}"

    def expr(self, cursor: cx.Cursor) -> CExpr:
        c = _unwrap(cursor)
        k = c.kind
        if k == cx.CursorKind.INTEGER_LITERAL:
            tokens = [t.spelling for t in c.get_tokens()]
            if not tokens:
                raise _refuse(c, "integer literal with no tokens")
            qty = _int_type_for(c.type)
            if qty is None:
                raise _refuse(c, f"integer literal of unsupported type {c.type.spelling!r}")
            return CIntLit(type=qty, value=_parse_c_int_literal_text(tokens[0], c))
        if k == cx.CursorKind.FLOATING_LITERAL:
            tokens = [t.spelling for t in c.get_tokens()]
            if not tokens:
                raise _refuse(c, "float literal with no tokens")
            bits, ftype = _parse_c_float_literal_text(tokens[0], c)
            return CFloatLit(type=ftype, bits=bits)
        if k in (cx.CursorKind.CSTYLE_CAST_EXPR, cx.CursorKind.CXX_FUNCTIONAL_CAST_EXPR):
            target_c_type = _c_source_type(c, c.type)
            # The cast cursor has children (TYPE_REF, the inner expr).
            # The inner expression is the last non-TYPE_REF child.
            inner = None
            for child in c.get_children():
                if child.kind != cx.CursorKind.TYPE_REF:
                    inner = child
            if inner is None:
                raise _refuse(c, "explicit cast with no expression operand")
            return CCast(
                id=self._mint("ccast"),
                target_type=target_c_type,
                value=self.expr(inner),
            )
        if k == cx.CursorKind.STRING_LITERAL:
            # Decode via Python's literal_eval (a strict superset of C
            # string-literal escape syntax — same path the layer-B
            # ingester uses to intern strings).
            tokens = [t.spelling for t in c.get_tokens()]
            if not tokens:
                raise _refuse(c, "string literal with no tokens")
            try:
                value = ast.literal_eval(tokens[0])
            except (ValueError, SyntaxError) as e:
                raise _refuse(c, f"could not decode string literal: {e}")
            if not isinstance(value, str):
                raise _refuse(c, f"string literal decoded to non-str ({type(value).__name__})")
            return CStringLit(id=self._mint("clitstr"), value=value)
        if k == cx.CursorKind.DECL_REF_EXPR:
            # Enum constants resolve to integer values at layer B
            # (`CURLOPT_URL` → `IntLit(10002)`); layer A preserves
            # both the source-level identifier and the resolved
            # value via `CEnumConstRef`. The pinned value is what the
            # lift-check compares against the layer-B `IntLit`.
            referenced = c.referenced
            if referenced is not None and referenced.kind == cx.CursorKind.ENUM_CONSTANT_DECL:
                return CEnumConstRef(
                    id=self._mint("cenumconst"),
                    name=c.spelling,
                    value=referenced.enum_value,
                )
            return CVarRef(name=c.spelling)
        if k == cx.CursorKind.CALL_EXPR:
            children = list(c.get_children())
            if not children:
                raise _refuse(c, "call expr with no children")
            callee = _unwrap(children[0])
            if callee.kind != cx.CursorKind.DECL_REF_EXPR:
                raise _refuse(c, "layer A: indirect / function-pointer calls not supported")
            if callee.spelling in _FENV_API_NAMES:
                raise _refuse(c, _fenv_call_refusal(callee.spelling))
            args = tuple(self.expr(a) for a in children[1:])
            return CCall(
                id=self._mint("ccall"),
                callee=callee.spelling,
                args=args,
            )
        if k == cx.CursorKind.BINARY_OPERATOR:
            tok = _binop_token(c)
            children = list(c.get_children())
            return CBinOp(
                id=self._mint("cbinop"),
                op=tok,
                lhs=self.expr(children[0]),
                rhs=self.expr(children[1]),
            )
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
            # `&buf[k]` — array-subscript address-of. Layer A preserves
            # both operators (CAddressOf wrapping CArraySubscript); the
            # lift-checker pairs the composed shape with the layer-B
            # `PtrOffset(buf, k)`.
            if op == "&":
                inner_cur = _unwrap(children[0])
                if inner_cur.kind == cx.CursorKind.ARRAY_SUBSCRIPT_EXPR:
                    sub_children = list(inner_cur.get_children())
                    if len(sub_children) != 2:
                        raise _refuse(inner_cur, "layer A: array subscript with non-2 children")
                    base, index = sub_children
                    return CAddressOf(
                        id=self._mint("caddrof"),
                        target=CArraySubscript(
                            id=self._mint("carrsub"),
                            base=self.expr(base),
                            index=self.expr(index),
                        ),
                    )
                raise _refuse(c, "layer A: address-of only supported for array subscripts (e.g. `&buf[k]`)")
            inner = self.expr(children[0])
            # Layer A preserves the source operator faithfully via CUnary.
            # Constant-folding of `-N` to a single CIntLit is OK because
            # the layer-B side does the same fold (lift-checker compares
            # values, not tree shapes, for IntLits).
            if op == "-":
                if isinstance(inner, CIntLit):
                    return CIntLit(type=inner.type, value=-inner.value)
                if isinstance(inner, CFloatLit):
                    # Same constant-fold for floats — layer-B flips the
                    # sign bit, so the lift-check sees identical
                    # CFloatLit / FloatLit bit patterns on both sides.
                    from quod.model import F32Type as _F32T
                    sign_bit = 1 << (31 if isinstance(inner.type, _F32T) else 63)
                    return CFloatLit(type=inner.type, bits=inner.bits ^ sign_bit)
                return CUnary(id=self._mint("cunary"), op="-", value=inner)
            if op == "!":
                return CUnary(id=self._mint("cunary"), op="!", value=inner)
            if op == "~":
                return CUnary(id=self._mint("cunary"), op="~", value=inner)
            if op == "+":
                # Unary plus is a no-op in C; layer A drops it (the
                # source-form lossiness here matches the BinOp side and
                # is a known minor infraction; preserving it would need
                # CUnary("+") which has no observable effect).
                return inner
            if op == "*":
                # `*p` rvalue — layer A preserves the deref with the
                # pointee type carried for the lift-check.
                load_ctype = _c_source_type(c, c.type)
                return CDeref(
                    id=self._mint("cderef"),
                    operand=inner,
                    load_type=load_ctype,
                )
            if op in ("++", "--"):
                raise _refuse(
                    c,
                    f"layer A: {op!r} in expression position is not yet supported "
                    f"(only bare-statement and for-loop inc positions are supported)"
                )
            raise _refuse(c, f"layer A: unsupported unary operator {op!r}")
        if k == cx.CursorKind.ARRAY_SUBSCRIPT_EXPR:
            # `arr[k]` standalone rvalue — layer A preserves the
            # subscript with `elem_type` set (the lift-check pairs
            # this against layer-B `Load(PtrOffset(...), T)`).
            children = list(c.get_children())
            if len(children) != 2:
                raise _refuse(c, f"layer A: array subscript with {len(children)} children")
            base, index = children
            elem_ctype = _c_source_type(c, c.type)
            return CArraySubscript(
                id=self._mint("carrsub"),
                base=self.expr(base),
                index=self.expr(index),
                elem_type=elem_ctype,
            )
        if k == cx.CursorKind.MEMBER_REF_EXPR:
            children = list(c.get_children())
            if len(children) != 1:
                raise _refuse(c, f"layer A: member-ref with {len(children)} children")
            base_cursor = _unwrap(children[0])
            field_name = c.spelling
            if not field_name:
                raise _refuse(c, "layer A: member-ref with no field name")
            field_ctype = _c_source_type(c, c.type)
            base_canon = base_cursor.type.get_canonical()
            arrow = base_canon.kind == cx.TypeKind.POINTER
            if arrow:
                pointee = base_canon.get_pointee().get_canonical()
                if pointee.kind != cx.TypeKind.RECORD:
                    raise _refuse(
                        c,
                        f"layer A: `->` on non-struct-pointer base "
                        f"({base_cursor.type.spelling!r})",
                    )
                struct_name = _struct_name_from_record_type(pointee)
                if struct_name is None or struct_name not in self._structs:
                    raise _refuse(
                        c,
                        f"layer A: unknown struct type for `->` "
                        f"({base_cursor.type.spelling!r})",
                    )
                return CFieldArrow(
                    id=self._mint("cfieldarrow"),
                    ptr=self.expr(base_cursor),
                    struct_type=struct_name,
                    name=field_name,
                    field_type=field_ctype,
                )
            return CFieldRead(
                id=self._mint("cfield"),
                value=self.expr(base_cursor),
                name=field_name,
                field_type=field_ctype,
            )
        if k == cx.CursorKind.INIT_LIST_EXPR:
            canon = c.type.get_canonical()
            if canon.kind != cx.TypeKind.RECORD:
                raise _refuse(
                    c,
                    f"layer A: non-struct init-list ({c.type.spelling!r}) "
                    f"is not yet supported",
                )
            struct_name = _struct_name_from_record_type(canon)
            if struct_name is None or struct_name not in self._structs:
                raise _refuse(
                    c,
                    f"layer A: unknown struct type for init-list "
                    f"({c.type.spelling!r})",
                )
            info = self._structs.get(struct_name)
            assert info is not None
            child_cursors = list(c.get_children())
            field_inits: list[CFieldInit] = []
            for i, child in enumerate(child_cursors):
                if child.kind == cx.CursorKind.MEMBER_REF:
                    raise _refuse(
                        child,
                        "layer A: designated struct initialisers are not "
                        "yet supported",
                    )
                if i >= len(info.fields):
                    raise _refuse(
                        child,
                        f"layer A: too many initialisers for struct "
                        f"{struct_name!r}",
                    )
                field_inits.append(CFieldInit(
                    name=info.field_names[i],
                    value=self.expr(child),
                ))
            return CStructInit(
                id=self._mint("cstructinit"),
                type_name=struct_name,
                fields=tuple(field_inits),
            )
        if k == cx.CursorKind.CONDITIONAL_OPERATOR:
            children = list(c.get_children())
            if len(children) != 3:
                raise _refuse(c, f"layer A: ternary with {len(children)} children")
            return CTernary(
                id=self._mint("cternary"),
                cond=self.expr(children[0]),
                then_value=self.expr(children[1]),
                else_value=self.expr(children[2]),
            )
        raise _refuse(c, f"layer A: unsupported expression kind: {k.name}")

    def stmt(self, cursor: cx.Cursor) -> CStmt:
        c = cursor
        k = c.kind

        if k == cx.CursorKind.RETURN_STMT:
            children = list(c.get_children())
            if not children:
                # Layer A preserves `return;` faithfully.
                return CReturn(id=self._mint("creturn"), value=None)
            return CReturn(
                id=self._mint("creturn"),
                value=self.expr(children[0]),
            )

        if k == cx.CursorKind.DECL_STMT:
            children = list(c.get_children())
            if not children:
                raise _refuse(c, "layer A: decl-stmt with no children")
            sub_decls: list[CVarDecl] = []
            for decl in children:
                if decl.kind != cx.CursorKind.VAR_DECL:
                    raise _refuse(decl, f"layer A: only var declarations supported, got {decl.kind.name}")
                init_cursors = [
                    ic for ic in decl.get_children() if ic.kind.is_expression()
                ]
                init = self.expr(init_cursors[-1]) if init_cursors else None
                sub_decls.append(CVarDecl(
                    id=self._mint("cvardecl"),
                    type=_c_source_type(decl, decl.type),
                    name=decl.spelling,
                    init=init,
                ))
            if len(sub_decls) == 1:
                return sub_decls[0]
            # Multi-declarator: `int a, b, c;` becomes a single layer-A
            # CMultiVarDecl wrapping the sub-decls. The lift-checker
            # pairs this 1:N against the layer-B sequence of Lets.
            return CMultiVarDecl(id=self._mint("cmultivardecl"), decls=tuple(sub_decls))

        if k == cx.CursorKind.BINARY_OPERATOR:
            # Bare assignment as a statement: `x = expr;`,
            # `*p = expr;`, or `arr[k] = expr;`.
            tokens = [t.spelling for t in c.get_tokens()]
            if "=" in tokens and "==" not in tokens:
                children = list(c.get_children())
                lhs = _unwrap(children[0])
                # Pointer-deref store
                if lhs.kind == cx.CursorKind.UNARY_OPERATOR:
                    lhs_tokens = [t.spelling for t in lhs.get_tokens()]
                    if lhs_tokens and lhs_tokens[0] == "*":
                        deref_children = list(lhs.get_children())
                        if len(deref_children) != 1:
                            raise _refuse(lhs, "layer A: deref-store with non-1 deref children")
                        store_ctype = _c_source_type(lhs, lhs.type)
                        return CDerefStore(
                            id=self._mint("cderefstore"),
                            operand=self.expr(deref_children[0]),
                            value=self.expr(children[1]),
                            store_type=store_ctype,
                        )
                # Field-arrow store: `p->x = v;`
                if lhs.kind == cx.CursorKind.MEMBER_REF_EXPR:
                    member_children = list(lhs.get_children())
                    if len(member_children) != 1:
                        raise _refuse(
                            lhs, "layer A: member-ref store with non-1 base children",
                        )
                    base_cursor = _unwrap(member_children[0])
                    field_name = lhs.spelling
                    base_canon = base_cursor.type.get_canonical()
                    arrow = base_canon.kind == cx.TypeKind.POINTER
                    if not arrow:
                        raise _refuse(
                            lhs,
                            "layer A: by-value struct field assignment is "
                            "not supported; use a pointer to the struct "
                            "and `->`",
                        )
                    pointee = base_canon.get_pointee().get_canonical()
                    struct_name = _struct_name_from_record_type(pointee)
                    if struct_name is None or struct_name not in self._structs:
                        raise _refuse(
                            lhs,
                            f"layer A: unknown struct type for `->` store "
                            f"({base_cursor.type.spelling!r})",
                        )
                    return CFieldArrowStore(
                        id=self._mint("cfieldarrowstore"),
                        ptr=self.expr(base_cursor),
                        struct_type=struct_name,
                        name=field_name,
                        value=self.expr(children[1]),
                    )
                # Subscript store
                if lhs.kind == cx.CursorKind.ARRAY_SUBSCRIPT_EXPR:
                    sub_children = list(lhs.get_children())
                    if len(sub_children) != 2:
                        raise _refuse(
                            lhs,
                            f"layer A: subscript-store with {len(sub_children)} children",
                        )
                    base, index = sub_children
                    elem_ctype = _c_source_type(lhs, lhs.type)
                    return CSubscriptStore(
                        id=self._mint("csubscriptstore"),
                        base=self.expr(base),
                        index=self.expr(index),
                        value=self.expr(children[1]),
                        elem_type=elem_ctype,
                    )
                if lhs.kind != cx.CursorKind.DECL_REF_EXPR:
                    raise _refuse(lhs, "layer A: only simple `name = expr` assignment supported")
                return CAssign(
                    id=self._mint("cassign"),
                    target=lhs.spelling,
                    value=self.expr(children[1]),
                )
            raise _refuse(c, "layer A: bare expression-as-statement only supported for assignments")

        if k == cx.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
            # Layer A preserves the source operator faithfully.
            children = list(c.get_children())
            if len(children) != 2:
                raise _refuse(c, f"layer A: compound assignment with {len(children)} children")
            lhs = _unwrap(children[0])
            if lhs.kind != cx.CursorKind.DECL_REF_EXPR:
                raise _refuse(lhs, "layer A: only simple `name op= expr` assignment supported")
            op = c.spelling
            if op not in _COMPOUND_ASSIGN_TABLE:
                raise _refuse(c, f"layer A: unsupported compound-assignment operator {op!r}")
            return CCompoundAssign(
                id=self._mint("ccompound"),
                target=lhs.spelling,
                op=op,
                value=self.expr(children[1]),
            )

        if k == cx.CursorKind.UNARY_OPERATOR:
            # Statement-position `i++;`, `++i;`, `i--;`, `--i;`. libclang
            # exposes both pre- and post-forms as a single UNARY_OPERATOR
            # cursor; we recover position from token order — operator
            # token first ⇒ pre, last ⇒ post. Other unary operators in
            # statement position aren't supported (a bare `-x;` etc. is
            # already refused upstream).
            children = list(c.get_children())
            if len(children) != 1:
                raise _refuse(c, "layer A: unary statement with non-1 children")
            tokens = [t.spelling for t in c.get_tokens()]
            if not tokens:
                raise _refuse(c, "layer A: unary statement with no tokens")
            op = tokens[0] if tokens[0] in ("++", "--") else tokens[-1]
            if op not in ("++", "--"):
                raise _refuse(c, f"layer A: unsupported unary statement operator {op!r}")
            position = "pre" if tokens[0] == op else "post"
            target = _unwrap(children[0])
            if target.kind != cx.CursorKind.DECL_REF_EXPR:
                raise _refuse(target, f"layer A: only simple `name{op}` / `{op}name` increment supported")
            return CIncrementStmt(
                id=self._mint("cincstmt"),
                target=target.spelling,
                op=op,
                position=position,
            )

        if k == cx.CursorKind.FOR_STMT:
            init_cursor, cond_cursor, inc_cursor, body_cursor = _split_for_children(c)
            return CFor(
                id=self._mint("cfor"),
                init=(self._for_init(init_cursor) if init_cursor is not None else None),
                cond=(self.expr(cond_cursor) if cond_cursor is not None else None),
                inc=(self._for_init(inc_cursor) if inc_cursor is not None else None),
                body=tuple(self.stmt(s) for s in self._compound_children(body_cursor)),
            )

        if k == cx.CursorKind.IF_STMT:
            children = list(c.get_children())
            if len(children) not in (2, 3):
                raise _refuse(c, f"layer A: if-stmt with {len(children)} children")
            cond = self.expr(children[0])
            then_body = tuple(self.stmt(s) for s in self._compound_children(children[1]))
            else_body = (
                tuple(self.stmt(s) for s in self._compound_children(children[2]))
                if len(children) == 3
                else ()
            )
            return CIf(
                id=self._mint("cif"),
                cond=cond, then_body=then_body, else_body=else_body,
            )

        if k == cx.CursorKind.WHILE_STMT:
            children = list(c.get_children())
            if len(children) != 2:
                raise _refuse(c, f"layer A: while-stmt with {len(children)} children")
            cond = self.expr(children[0])
            body = tuple(self.stmt(s) for s in self._compound_children(children[1]))
            return CWhile(
                id=self._mint("cwhile"),
                cond=cond, body=body,
            )

        if k == cx.CursorKind.DO_STMT:
            children = list(c.get_children())
            if len(children) != 2:
                raise _refuse(c, f"layer A: do-while with {len(children)} children")
            body = tuple(self.stmt(s) for s in self._compound_children(children[0]))
            cond = self.expr(children[1])
            return CDoWhile(
                id=self._mint("cdowhile"),
                body=body, cond=cond,
            )

        if k == cx.CursorKind.CALL_EXPR:
            # `printf(...);` and similar — call as a statement-effect.
            return CExprStmt(
                id=self._mint("cexprstmt"),
                value=self.expr(c),
            )

        if k == cx.CursorKind.BREAK_STMT:
            return CBreak(id=self._mint("cbreak"))

        if k == cx.CursorKind.CONTINUE_STMT:
            return CContinue(id=self._mint("ccontinue"))

        if k == cx.CursorKind.SWITCH_STMT:
            children = list(c.get_children())
            if len(children) != 2:
                raise _refuse(c, f"layer A: switch with {len(children)} children")
            scrutinee = self.expr(children[0])
            groups = _parse_switch_groups(children[1])
            cases: list[CSwitchCase] = []
            default_body: tuple[CStmt, ...] | None = None
            for value_cursors, body_cursors, kind in groups:
                # Drop trailing break (matches the layer-B trim).
                trimmed = body_cursors
                if trimmed and trimmed[-1].kind == cx.CursorKind.BREAK_STMT:
                    trimmed = trimmed[:-1]
                body_a = tuple(self.stmt(s) for s in trimmed)
                if kind == "default":
                    default_body = body_a
                else:
                    cases.append(CSwitchCase(
                        id=self._mint("cswitchcase"),
                        values=tuple(self.expr(v) for v in value_cursors),
                        body=body_a,
                    ))
            return CSwitch(
                id=self._mint("cswitch"),
                scrutinee=scrutinee,
                cases=tuple(cases),
                default=default_body,
            )

        raise _refuse(c, f"layer A: unsupported statement kind: {k.name}")

    def _for_init(self, cursor: cx.Cursor) -> CForInit:
        """Translate a for-loop init or inc slot into a CForInit
        (CVarDecl, CAssign, or CIncrementStmt). Mirrors the layer-B
        path; the validation is shared because the C grammar is the
        same."""
        s = self.stmt(cursor)
        if isinstance(s, (CVarDecl, CAssign, CIncrementStmt)):
            return s
        raise _refuse(cursor, f"layer A: for init/inc must be a decl, assignment, or increment, got {type(s).__name__}")

    def _compound_children(self, cursor: cx.Cursor) -> list[cx.Cursor]:
        """A for-loop body may be a `{ ... }` block or a single statement;
        normalize to a list of child statements."""
        if cursor.kind == cx.CursorKind.COMPOUND_STMT:
            return list(cursor.get_children())
        return [cursor]


def _c_source_type(cursor: cx.Cursor, t: cx.Type) -> CType:
    """Map a clang Type to a layer-A `CType` — a `CNamedType` for
    scalar types (int, char) or a `CPointerType` wrapping the
    pointee. The pointee can recurse for `int**`, `char**`, etc.

    For typedef'd pointer aliases (`CURL*` is `struct Curl_easy *`
    canonically), we use the *declaration's* spelling so the
    source-level typedef name survives at layer A. The lift-check
    treats any `CPointerType` as corresponding to layer-B's
    `I8PtrType` regardless of pointee — the name is informational.
    """
    if t.kind == cx.TypeKind.POINTER:
        # Use the source-level pointee spelling when possible (a
        # typedef'd `CURL*`'s pointee spells `CURL`, not `struct
        # Curl_easy`). When we can't recover that, fall back to the
        # canonical pointee.
        pointee_t = t.get_pointee()
        return CPointerType(pointee=_c_source_type(cursor, pointee_t))
    canon = t.get_canonical()
    # Integer types (every supported width and signedness, plus typedef'd
    # standards like `size_t`, `int64_t`, `uint8_t`). Layer A preserves
    # the source spelling — `t.spelling` keeps typedefs and explicit
    # multi-word names (`"unsigned long long"`, `"size_t"`) intact. The
    # lift-check canonicalizes spellings against quod's IntType classes.
    if _int_type_for(t) is not None:
        return CNamedType(name=t.spelling)
    if canon.kind == cx.TypeKind.FLOAT:
        return CNamedType(name="float")
    if canon.kind == cx.TypeKind.DOUBLE:
        return CNamedType(name="double")
    if canon.kind == cx.TypeKind.LONGDOUBLE:
        raise _refuse(cursor, _LONG_DOUBLE_REFUSAL)
    if canon.kind == cx.TypeKind.RECORD:
        # Named struct: prefer the canonical struct name when libclang
        # exposes it (e.g. `struct Foo` → `CStructType("Foo")`); fall
        # back to the source spelling for opaque records (typedef'd
        # aliases like `CURL` for `struct Curl_easy`).
        struct_name = _struct_name_from_record_type(canon)
        if struct_name is not None:
            return CStructType(name=struct_name)
        return CNamedType(name=t.spelling)
    if canon.kind == cx.TypeKind.VOID:
        return CNamedType(name="void")
    raise _refuse(cursor, f"layer A: unsupported type {t.spelling!r}")


def _translate_function_layer_a(
    cursor: cx.Cursor, source_path: Path,
    structs: _StructRegistry | None = None,
) -> CFn:
    """Build the layer-A `CFn` for one C function definition. ID is
    derived from the spelling so it's stable across re-ingest of the
    same source — the same convention the layer-B Function uses, with a
    distinct `@cfn_c_*` prefix so the two are addressable separately."""
    # Reuse the source-type mapping so int/void/float/double/char/etc.
    # all resolve through the same dispatch (and `long double` refuses
    # uniformly).
    return_type = _c_source_type(cursor, cursor.result_type)
    params: list[CParam] = []
    body_cursor: cx.Cursor | None = None
    for child in cursor.get_children():
        if child.kind == cx.CursorKind.PARM_DECL:
            params.append(CParam(name=child.spelling, type=_c_source_type(child, child.type)))
        elif child.kind == cx.CursorKind.COMPOUND_STMT:
            body_cursor = child
    if body_cursor is None:
        raise _refuse(cursor, "layer A: function has no body")
    translator = _LayerATranslator(cursor.spelling, structs=structs)
    body = tuple(translator.stmt(s) for s in body_cursor.get_children())
    return CFn(
        id=f"@cfn_c_{cursor.spelling}",
        name=cursor.spelling,
        return_type=return_type,
        params=tuple(params),
        body=body,
    )


def _struct_def_layer_a(info: _StructInfo) -> CStructDef:
    """Convert a per-TU `_StructInfo` into a layer-A `CStructDef`.
    Used to populate `CUnit.struct_defs` so source units carry the
    struct decls they reference. Field types map to layer-A `CType`s
    via the cursor's source-form spelling."""
    fields: list[CField] = []
    for child in info.cursor.get_children():
        if child.kind != cx.CursorKind.FIELD_DECL:
            continue
        if child.is_bitfield():
            # Already refused at registry build, but defensively
            # re-check.
            raise _refuse(child, "bit-fields not yet supported")
        fields.append(CField(
            name=child.spelling,
            type=_c_source_type(child, child.type),
        ))
    return CStructDef(
        id=f"@cstructdef_c_{info.name}",
        name=info.name,
        fields=tuple(fields),
    )
