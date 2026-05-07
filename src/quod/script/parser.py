"""quod-script parser: tokens -> AST.

Phase 1 of parsing. Produces frozen-Pydantic model nodes plus a
`poison_locs` side table marking every bare (suffix-less) IntLit so
the resolver pass can retype them from operand context."""

from __future__ import annotations

from typing import Optional

from quod.model import (
    Assign,
    BinOp,
    Block,
    CharLit,
    Call,
    EnumInit,
    EnumType,
    SizeOf,
    TryExpr,
    ExprStmt,
    FieldInit,
    FieldRead,
    FieldSet,
    For,
    Function,
    I1Type,
    I8PtrType,
    I8Type,
    I16Type,
    I32Type,
    I64Type,
    IfExpr,
    U8Type,
    U16Type,
    U32Type,
    U64Type,
    IsizeType,
    UsizeType,
    If,
    IntLit,
    Let,
    Load,
    LocalRef,
    Match,
    MatchArm,
    Not,
    NullPtr,
    Param,
    ParamRef,
    PtrOffset,
    Return,
    ReturnExpr,
    ReturnRef,
    ShortCircuitAnd,
    ShortCircuitOr,
    Store,
    StringRef,
    StructInit,
    StructType,
    VoidType,
    While,
    Widen,
    WithArena,
)

from quod.script.tokens import ScriptError, Token


_INT_TYPE_BY_SUFFIX = {
    "i1": I1Type, "i8": I8Type, "i16": I16Type, "i32": I32Type, "i64": I64Type,
    "u8": U8Type, "u16": U16Type, "u32": U32Type, "u64": U64Type,
    "isize": IsizeType, "usize": UsizeType,
}


def _split_int_suffix(text: str) -> tuple[str, str | None]:
    """Split '42i8' into ('42', 'i8'); '42' into ('42', None).
    isize/usize matched first so they win over the i/u prefix."""
    for suf in ("isize", "usize",
                "i64", "i32", "i16", "i8", "i1",
                "u64", "u32", "u16", "u8"):
        if text.endswith(suf):
            return text[:-len(suf)], suf
    return text, None


# Sentinel concrete int-type used as a placeholder on bare integer
# literals during phase-1 parsing. The resolver pass replaces every
# poison-marked IntLit with one carrying its real type from operand
# context; any literal whose marker survives both walks is a parse
# error ("bare integer literal needs a width suffix here"). The
# placeholder type is i64 — never observed by callers, since either
# the resolver retypes the literal or raises.
_PLACEHOLDER_INT_TYPE = I64Type


# Comparison ops -> BinOp.op
_CMP_OPS = {
    "==": "eq", "!=": "ne",
    "<": "slt", "<=": "sle", ">": "sgt", ">=": "sge",
    "<u": "ult", "<=u": "ule", ">u": "ugt", ">=u": "uge",
}


class Parser:
    def __init__(self, tokens: list[Token], *, enum_names: frozenset[str] = frozenset()):
        self.toks = tokens
        self.pos = 0
        self.param_names: frozenset[str] = frozenset()
        # Names known to be enums (from the surrounding program). When a type
        # token is a bare IDENT, we emit EnumType for names in this set and
        # StructType for everything else. Empty by default — if the caller
        # didn't tell us, every custom type becomes a struct.
        self.enum_names = enum_names
        # Disabled in the condition position of if/while and the bounds of
        # for, where `{` always begins the body block. Mirrors Rust's
        # struct-literal-in-cond restriction; parens force the issue when
        # you really do want a literal there: `if (Foo({a: 1}).b == 2) {..}`
        self._struct_init_allowed = True
        # Side tables for the type-resolution pass. `poison_locs` keys are
        # the `id()` of every bare (suffix-less) IntLit produced by phase-1
        # parsing — its presence in the dict marks the literal as needing
        # context-driven retyping; the value is `(line, col)` for error
        # reporting. The resolver removes entries as it retypes literals;
        # any entry that survives the walk becomes a ScriptError. Pydantic
        # nodes are frozen and carry no source location, so the side table
        # is the only place this metadata lives.
        self.poison_locs: dict[int, tuple[int, int]] = {}

    def _make_int_lit(self, tok: Token, *, negate: bool = False) -> "IntLit":
        """Build an IntLit from an INT token. Suffix-less literals are
        marked poison via the side table for the resolver to retype later;
        suffixed literals carry their declared width and are not poisoned."""
        digits, suf = _split_int_suffix(tok.value)
        value = int(digits)
        if negate:
            value = -value
        if suf is None:
            lit = IntLit(type=_PLACEHOLDER_INT_TYPE(), value=value)
            self.poison_locs[id(lit)] = (tok.line, tok.col)
            return lit
        return IntLit(type=_INT_TYPE_BY_SUFFIX[suf](), value=value)

    # -- cursor helpers --

    def peek(self, offset: int = 0) -> Token:
        return self.toks[self.pos + offset]

    def eat(self) -> Token:
        t = self.toks[self.pos]
        self.pos += 1
        return t

    def at(self, kind: str, value: Optional[str] = None) -> bool:
        t = self.peek()
        if t.kind != kind:
            return False
        return value is None or t.value == value

    def expect(self, kind: str, value: Optional[str] = None) -> Token:
        t = self.peek()
        if t.kind != kind or (value is not None and t.value != value):
            want = value if value is not None else kind
            raise ScriptError(
                f"expected {want!r}, got {t.kind} {t.value!r}", t.line, t.col
            )
        return self.eat()

    def consume_terminator(self) -> None:
        """Optional ';' between statements; whitespace handled by lexer."""
        if self.at("OP", ";"):
            self.eat()

    def _dotted_after(self, first: str) -> str:
        """Consume a `.IDENT` chain following an already-eaten initial
        IDENT and return the joined dotted name (`a.b.c`). Returns
        `first` unchanged if no dotted continuation."""
        parts = [first]
        while self.at("OP", ".") and self.peek(1).kind == "IDENT":
            self.eat()
            parts.append(self.eat().value)
        return ".".join(parts)

    # -- top-level --

    def parse_function(self) -> Function:
        self.expect("KW", "fn")
        # Dotted name supported so stdlib modules can author
        # `fn alloc.json.parse_array(...)` directly.
        name = self._dotted_after(self.expect("IDENT").value)
        self.expect("OP", "(")
        params: list[Param] = []
        if not self.at("OP", ")"):
            params.append(self._param())
            while self.at("OP", ","):
                self.eat()
                params.append(self._param())
        self.expect("OP", ")")
        self.expect("OP", "->")
        ret_ty = self._type(allow_void=True)
        self.param_names = frozenset(p.name for p in params)
        body = self._block()
        # The model's Function has `claims: tuple[Claim, ...] = ()` and
        # `notes: tuple[str, ...] = ()`; both default. We don't author either
        # here — claims have their own surface, notes too.
        return Function(
            name=name,
            params=tuple(params),
            return_type=ret_ty,
            body=Block(stmts=tuple(body)),
        )

    def _param(self) -> Param:
        name = self.expect("IDENT").value
        self.expect("OP", ":")
        ty = self._type(allow_void=False)
        return Param(name=name, type=ty)

    # -- types --

    _PRIM_TYPE_MAP = {
        "i1": I1Type, "i8": I8Type, "i16": I16Type,
        "i32": I32Type, "i64": I64Type,
        "u8": U8Type, "u16": U16Type, "u32": U32Type, "u64": U64Type,
        "isize": IsizeType, "usize": UsizeType,
    }

    def _type(self, *, allow_void: bool):
        t = self.peek()
        if t.kind == "KW" and t.value == "void":
            if not allow_void:
                raise ScriptError(
                    "void only valid as a function return type", t.line, t.col
                )
            self.eat()
            return VoidType()
        if t.kind == "KW" and t.value in self._PRIM_TYPE_MAP:
            self.eat()
            if t.value == "i8" and self.at("OP", "*"):
                self.eat()
                return I8PtrType()
            return self._PRIM_TYPE_MAP[t.value]()
        if t.kind == "IDENT":
            self.eat()
            # Allow dotted type names like `core.str.String` or
            # `alloc.json.JsonValue`.
            full = self._dotted_after(t.value)
            if full in self.enum_names:
                return EnumType(name=full)
            return StructType(name=full)
        raise ScriptError(f"expected a type, got {t.kind} {t.value!r}", t.line, t.col)

    # -- statements --

    def _block(self) -> list:
        self.expect("OP", "{")
        out = []
        while not self.at("OP", "}"):
            out.append(self._stmt())
            self.consume_terminator()
        self.expect("OP", "}")
        return out

    def _stmt(self):
        t = self.peek()
        if t.kind == "KW":
            match t.value:
                case "let": return self._let()
                case "if": return self._if()
                case "while": return self._while()
                case "for": return self._for()
                case "return": return self._return()
                case "store": return self._store_stmt()
                case "with_arena": return self._with_arena()
                case "match": return self._match()
        # IDENT — could be assign / field_set / expr_stmt (call)
        if t.kind == "IDENT":
            # Look ahead: IDENT '=' is assign; IDENT '.' IDENT '=' is field_set;
            # else expression statement.
            if self.peek(1).kind == "OP" and self.peek(1).value == "=":
                return self._assign()
            if (
                self.peek(1).kind == "OP" and self.peek(1).value == "."
                and self.peek(2).kind == "IDENT"
                and self.peek(3).kind == "OP" and self.peek(3).value == "="
            ):
                return self._field_set()
        # Fall-through: an expression statement (typically a call).
        expr = self._expr()
        return ExprStmt(value=expr)

    def _let(self) -> Let:
        self.expect("KW", "let")
        name = self.expect("IDENT").value
        self.expect("OP", ":")
        ty = self._type(allow_void=False)
        self.expect("OP", "=")
        init = self._expr()
        return Let(name=name, type=ty, init=init)

    def _assign(self) -> Assign:
        name = self.expect("IDENT").value
        self.expect("OP", "=")
        value = self._expr()
        return Assign(name=name, value=value)

    def _field_set(self) -> FieldSet:
        local = self.expect("IDENT").value
        self.expect("OP", ".")
        field = self.expect("IDENT").value
        self.expect("OP", "=")
        value = self._expr()
        return FieldSet(local=local, name=field, value=value)

    def _if(self) -> If:
        self.expect("KW", "if")
        self.expect("OP", "(")
        cond = self._cond_expr()
        self.expect("OP", ")")
        then_body = self._block()
        else_body: list = []
        if self.at("KW", "else"):
            self.eat()
            else_body = self._block()
        return If(
            cond=cond,
            then_body=Block(stmts=tuple(then_body)),
            else_body=Block(stmts=tuple(else_body)),
        )

    def _while(self) -> While:
        self.expect("KW", "while")
        self.expect("OP", "(")
        cond = self._cond_expr()
        self.expect("OP", ")")
        body = self._block()
        return While(cond=cond, body=Block(stmts=tuple(body)))

    def _for(self) -> For:
        self.expect("KW", "for")
        var = self.expect("IDENT").value
        self.expect("OP", ":")
        ty = self._type(allow_void=False)
        if not isinstance(ty, (I1Type, I8Type, I16Type, I32Type, I64Type,
                                U8Type, U16Type, U32Type, U64Type,
                                IsizeType, UsizeType)):
            t = self.peek(-1)
            raise ScriptError("for-loop variable must be an integer type", t.line, t.col)
        self.expect("KW", "in")
        lo = self._cond_expr()
        self.expect("OP", "..")
        hi = self._cond_expr()
        body = self._block()
        return For(var=var, type=ty, lo=lo, hi=hi, body=Block(stmts=tuple(body)))

    def _cond_expr(self):
        """Expression with struct-init disabled, so a trailing `{` always
        belongs to the enclosing if/while/for block, not to a struct literal.
        Use `(Foo { ... })` explicitly when you really want one there."""
        prev = self._struct_init_allowed
        self._struct_init_allowed = False
        try:
            return self._expr()
        finally:
            self._struct_init_allowed = prev

    def _return(self):
        self.expect("KW", "return")
        if not self._is_expr_start():
            return Return()
        return ReturnExpr(value=self._expr())

    def _store_stmt(self) -> Store:
        self.expect("KW", "store")
        self.expect("OP", "(")
        ptr = self._expr()
        self.expect("OP", ",")
        value = self._expr()
        self.expect("OP", ")")
        return Store(ptr=ptr, value=value)

    def _match(self) -> Match:
        self.expect("KW", "match")
        scrut = self._cond_expr()  # struct_init disabled while reading scrutinee
        self.expect("OP", "{")
        arms: list[MatchArm] = []
        while not self.at("OP", "}"):
            # `_` is a wildcard arm — no bindings, matches anything not
            # handled by another arm. The lexer treats it as an IDENT.
            head_tok = self.peek()
            if head_tok.kind == "IDENT" and head_tok.value == "_":
                self.eat()
                self.expect("OP", "=>")
                body = self._block()
                self.consume_terminator()
                arms.append(MatchArm(
                    variant="_", bindings=(), body=Block(stmts=tuple(body)),
                ))
                continue
            variant = self.expect("IDENT").value
            bindings: list[str] = []
            if self.at("OP", "("):
                self.eat()
                if not self.at("OP", ")"):
                    bindings.append(self.expect("IDENT").value)
                    while self.at("OP", ","):
                        self.eat()
                        bindings.append(self.expect("IDENT").value)
                self.expect("OP", ")")
            self.expect("OP", "=>")
            body = self._block()
            self.consume_terminator()
            arms.append(MatchArm(
                variant=variant, bindings=tuple(bindings), body=Block(stmts=tuple(body)),
            ))
        self.expect("OP", "}")
        return Match(scrutinee=scrut, arms=tuple(arms))

    def _with_arena(self) -> WithArena:
        self.expect("KW", "with_arena")
        name = self.expect("IDENT").value
        self.expect("OP", "(")
        self.expect("KW", "capacity")
        self.expect("OP", "=")
        cap = self._expr()
        self.expect("OP", ")")
        body = self._block()
        return WithArena(name=name, capacity=cap, body=Block(stmts=tuple(body)))

    # -- expressions (Pratt-ish via precedence climbing) --

    def _is_expr_start(self) -> bool:
        t = self.peek()
        if t.kind in ("INT", "CHAR", "IDENT"):
            return True
        if t.kind == "KW" and t.value in (
            "null", "true", "false", "load", "widen", "uwiden", "ptr_offset",
            "sizeof",
        ):
            return True
        if t.kind == "OP" and t.value in ("(", "&", "-"):
            return True
        return False

    def _expr(self):
        return self._or()

    def _or(self):
        lhs = self._and()
        while self.at("OP", "||"):
            self.eat()
            rhs = self._and()
            lhs = ShortCircuitOr(lhs=lhs, rhs=rhs)
        return lhs

    def _and(self):
        lhs = self._cmp()
        while self.at("OP", "&&"):
            self.eat()
            rhs = self._cmp()
            lhs = ShortCircuitAnd(lhs=lhs, rhs=rhs)
        return lhs

    def _cmp(self):
        lhs = self._add()
        if self.at("OP") and self.peek().value in _CMP_OPS:
            op_tok = self.eat()
            rhs = self._add()
            return BinOp(op=_CMP_OPS[op_tok.value], lhs=lhs, rhs=rhs)
        return lhs

    def _add(self):
        lhs = self._mul()
        while self.at("OP") and self.peek().value in ("+", "-"):
            op_tok = self.eat()
            rhs = self._mul()
            lhs = BinOp(op="add" if op_tok.value == "+" else "sub", lhs=lhs, rhs=rhs)
        return lhs

    def _mul(self):
        lhs = self._unary()
        while self.at("OP") and self.peek().value in ("*", "/", "%", "/u", "%u"):
            op_tok = self.eat()
            op = {"*": "mul", "/": "sdiv", "%": "srem",
                  "/u": "udiv", "%u": "urem"}[op_tok.value]
            rhs = self._unary()
            lhs = BinOp(op=op, lhs=lhs, rhs=rhs)
        return lhs

    def _unary(self):
        # Negative integer sugar: -INT becomes IntLit(value=-N). Otherwise
        # parse a postfix.
        if self.at("OP", "-") and self.peek(1).kind == "INT":
            self.eat()
            tok = self.eat()
            return self._make_int_lit(tok, negate=True)
        return self._postfix()

    def _postfix(self):
        e = self._primary()
        while True:
            if self.at("OP", ".") and self.peek(1).kind == "IDENT":
                self.eat()
                field = self.expect("IDENT").value
                e = FieldRead(value=e, name=field)
            elif self.at("OP", "?"):
                self.eat()
                e = TryExpr(value=e)
            else:
                break
        return e

    def _primary(self):
        t = self.peek()
        # Parens — re-enable struct_init for the bracketed sub-expression so
        # `(Foo { ... })` works inside a cond context.
        if t.kind == "OP" and t.value == "(":
            self.eat()
            prev = self._struct_init_allowed
            self._struct_init_allowed = True
            try:
                e = self._expr()
            finally:
                self._struct_init_allowed = prev
            self.expect("OP", ")")
            return e
        # &.const_name (string ref)
        if t.kind == "OP" and t.value == "&":
            self.eat()
            self.expect("OP", ".")
            head = self.expect("IDENT").value
            # Allow dotted constant names like .str.greeting
            full = "." + head
            while self.at("OP", ".") and self.peek(1).kind == "IDENT":
                self.eat()
                full += "." + self.expect("IDENT").value
            return StringRef(name=full)
        # Integer — optional type suffix (e.g. 42i8) carries the int width.
        # Suffix-less literals are marked poison; the resolver pass retypes
        # them from operand context after parsing finishes.
        if t.kind == "INT":
            tok = self.eat()
            return self._make_int_lit(tok)
        # Char literal
        if t.kind == "CHAR":
            self.eat()
            return CharLit(value=t.value)
        # Keywords
        if t.kind == "KW":
            match t.value:
                case "null":
                    self.eat()
                    return NullPtr()
                case "true":
                    self.eat()
                    return IntLit(type=I1Type(), value=1)
                case "false":
                    self.eat()
                    return IntLit(type=I1Type(), value=0)
                case "return":
                    # `return` in expression position yields the symbolic
                    # return value. Only meaningful inside a predicate; the
                    # statement-level `return EXPR` consumes the keyword
                    # before the expression parser ever sees it.
                    self.eat()
                    return ReturnRef()
                case "load":
                    return self._load()
                case "widen":
                    return self._widen(signed=True)
                case "uwiden":
                    return self._widen(signed=False)
                case "ptr_offset":
                    return self._ptr_offset()
                case "sizeof":
                    return self._sizeof()
        # Identifier — could be call (incl. dotted `core.bytes.eq(...)`),
        # struct_init, enum_init, or local/param ref.
        if t.kind == "IDENT":
            self.eat()
            if self.at("OP", "::"):
                return self._enum_init(t.value)
            # Lookahead for a dotted name followed by call / struct-init /
            # variant constructor: `a.b.c(...)`, `a.b.c { ... }`,
            # `a.b.c::Variant(...)`. Distinguish from a field-read chain
            # like `parser.input_ptr` by what terminates the dotted chain.
            if self.at("OP", ".") and self.peek(1).kind == "IDENT":
                save = self.pos
                full = self._dotted_after(t.value)
                if self.at("OP", "("):
                    return self._call_args(full)
                if self.at("OP", "::"):
                    return self._enum_init(full)
                if self.at("OP", "{") and self._struct_init_allowed:
                    return self._struct_init(full)
                self.pos = save
            if self.at("OP", "("):
                return self._call_args(t.value)
            if self.at("OP", "{") and self._struct_init_allowed:
                return self._struct_init(t.value)
            # Bare identifier — disambiguate via param_names captured at the
            # start of the body. Anything declared by `let` or a `for` loop
            # var is a local; everything else falls back to local too (with
            # the model validator catching truly undefined refs).
            if t.value in self.param_names:
                return ParamRef(name=t.value)
            return LocalRef(name=t.value)
        raise ScriptError(f"unexpected token {t.kind} {t.value!r}", t.line, t.col)

    def _call_args(self, fn_name: str) -> Call:
        self.expect("OP", "(")
        args = []
        if not self.at("OP", ")"):
            args.append(self._expr())
            while self.at("OP", ","):
                self.eat()
                args.append(self._expr())
        self.expect("OP", ")")
        return Call(function=fn_name, args=tuple(args))

    def _enum_init(self, enum_name: str) -> EnumInit:
        self.expect("OP", "::")
        variant = self.expect("IDENT").value
        fields: list[FieldInit] = []
        if self.at("OP", "("):
            self.eat()
            if not self.at("OP", ")"):
                fields.append(self._field_init())
                while self.at("OP", ","):
                    self.eat()
                    if self.at("OP", ")"):
                        break  # trailing comma
                    fields.append(self._field_init())
            self.expect("OP", ")")
        return EnumInit(enum=enum_name, variant=variant, fields=tuple(fields))

    def _struct_init(self, name: str) -> StructInit:
        self.expect("OP", "{")
        fields = []
        if not self.at("OP", "}"):
            fields.append(self._field_init())
            while self.at("OP", ","):
                self.eat()
                if self.at("OP", "}"):
                    break  # trailing comma allowed
                fields.append(self._field_init())
        self.expect("OP", "}")
        return StructInit(type=name, fields=tuple(fields))

    def _field_init(self) -> FieldInit:
        name = self.expect("IDENT").value
        self.expect("OP", ":")
        value = self._expr()
        return FieldInit(name=name, value=value)

    def _load(self) -> Load:
        self.expect("KW", "load")
        self.expect("OP", "[")
        ty = self._type(allow_void=False)
        self.expect("OP", "]")
        self.expect("OP", "(")
        ptr = self._expr()
        self.expect("OP", ")")
        return Load(ptr=ptr, type=ty)

    def _widen(self, *, signed: bool) -> Widen:
        self.expect("KW", "uwiden" if not signed else "widen")
        self.expect("OP", "(")
        v = self._expr()
        self.expect("KW", "to")
        target = self._type(allow_void=False)
        self.expect("OP", ")")
        return Widen(value=v, target=target, signed=signed)

    def _sizeof(self) -> SizeOf:
        self.expect("KW", "sizeof")
        self.expect("OP", "[")
        ty = self._type(allow_void=False)
        self.expect("OP", "]")
        return SizeOf(type=ty)

    def _ptr_offset(self) -> PtrOffset:
        self.expect("KW", "ptr_offset")
        self.expect("OP", "(")
        base = self._expr()
        self.expect("OP", ",")
        offset = self._expr()
        self.expect("OP", ")")
        return PtrOffset(base=base, offset=offset)
