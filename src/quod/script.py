"""quod-script: a compact textual surface for authoring function bodies.

Designed for the LLM-to-CLI handoff: instead of writing a full JSON
spec to a file and pointing `quod fn add` at it, you pass a short
script string. The script is one-way (script -> JSON nodes) and
covers the authoring-code subset of the model: function signatures,
statements, and expressions. Things outside that subset (claims,
struct definitions, externs, string constants, imports) stay on the
existing `quod struct add` / `quod claim add` / etc. surfaces.

The grammar:

    function   := 'fn' IDENT '(' params? ')' '->' type body
    params     := param (',' param)*
    param      := IDENT ':' type
    body       := '{' stmt* '}'

    type       := 'i1' | 'i8' '*'? | 'i16' | 'i32' | 'i64' | 'void' | IDENT

    stmt       := let_stmt | if_stmt | while_stmt | for_stmt | return_stmt
                | with_arena | store_stmt | assign_or_field_set_or_expr
    let_stmt   := 'let' IDENT ':' type '=' expr
    if_stmt    := 'if' '(' expr ')' block ('else' block)?
    while_stmt := 'while' '(' expr ')' block
    for_stmt   := 'for' IDENT ':' type 'in' expr '..' expr block
    return_stmt:= 'return' expr?
    store_stmt := 'store' '(' expr ',' expr ')'
    with_arena := 'with_arena' IDENT '(' 'capacity' '=' expr ')' block
    assign_or_field_set_or_expr
               := IDENT '=' expr                       # assign
                | IDENT '.' IDENT '=' expr             # field_set
                | expr                                  # expr stmt

    block      := '{' stmt* '}'

    expr       := or_expr
    or_expr    := and_expr ('||' and_expr)*
    and_expr   := cmp_expr ('&&' cmp_expr)*
    cmp_expr   := add_expr (CMPOP add_expr)?
    add_expr   := mul_expr (('+' | '-') mul_expr)*
    mul_expr   := unary_expr (('*' | '/' | '%' | '/u') unary_expr)*
    unary_expr := postfix
    postfix    := primary ('.' IDENT)*

    primary    := INT | CHAR | 'null' | 'true' | 'false'
                | '&' DOT_IDENT
                | 'load' '[' type ']' '(' expr ')'
                | 'widen' '(' expr 'to' type ')'
                | 'uwiden' '(' expr 'to' type ')'
                | 'ptr_offset' '(' expr ',' expr ')'
                | IDENT '(' args? ')'      # call
                | IDENT '{' field_inits '}' # struct_init
                | IDENT                    # local/param ref
                | '(' expr ')'

    field_inits := field_init (',' field_init)* ','?
    field_init  := IDENT ':' expr
    args        := expr (',' expr)*

CMPOP is one of: == != < <= > >= <u <=u >u >=u

Statements may be terminated by newlines or ';' (both work; either is
optional at end of block). Integer literals default to i64; use a
width suffix (`0i8`, `42i32`, `-3i8`) to opt into a narrower type.
A bare integer literal at return position adopts the function's
declared return_type, so `return 0` works in any int-returning fn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from quod.model import (
    Assign,
    BinOp,
    CharLit,
    Call,
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
    If,
    IntLit,
    Let,
    Load,
    LocalRef,
    NullPtr,
    Param,
    ParamRef,
    PtrOffset,
    Return,
    ReturnExpr,
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


# ---------- Tokens ----------

@dataclass
class Token:
    kind: str          # 'IDENT', 'INT', 'CHAR', 'DOT_IDENT', 'OP', 'KW', 'EOF'
    value: str
    line: int
    col: int


_KEYWORDS = frozenset({
    "fn", "let", "if", "else", "while", "for", "in", "return",
    "store", "with_arena", "capacity", "load", "widen", "uwiden",
    "ptr_offset", "to", "null", "true", "false",
    # type keywords
    "i1", "i8", "i16", "i32", "i64", "void",
})

# Multi-char operators must be matched before single-char ones.
_MULTI_OPS = (
    "->", "==", "!=", "<=", ">=", "<u", ">u", "<=u", ">=u", "/u",
    "||", "&&", "..",
)
_SINGLE_OPS = "(){}[],:;=+-*/%<>.&|"


class ScriptError(ValueError):
    """Parse error with line/column."""

    def __init__(self, msg: str, line: int, col: int):
        super().__init__(f"line {line}, col {col}: {msg}")
        self.line = line
        self.col = col


def tokenize(src: str) -> list[Token]:
    """Lex `src` into tokens. Whitespace and # comments are skipped."""
    tokens: list[Token] = []
    i = 0
    line, col = 1, 1
    n = len(src)

    def at(k: int) -> str:
        return src[k] if k < n else ""

    while i < n:
        c = src[i]

        # Newline
        if c == "\n":
            i += 1
            line += 1
            col = 1
            continue

        # Whitespace
        if c in " \t\r":
            i += 1
            col += 1
            continue

        # Line comment
        if c == "#":
            while i < n and src[i] != "\n":
                i += 1
                col += 1
            continue

        # Multi-char operators
        matched = False
        for op in _MULTI_OPS:
            if src[i:i + len(op)] == op:
                tokens.append(Token("OP", op, line, col))
                i += len(op)
                col += len(op)
                matched = True
                break
        if matched:
            continue

        # Single-char operators
        if c in _SINGLE_OPS:
            tokens.append(Token("OP", c, line, col))
            i += 1
            col += 1
            continue

        # Char literal: '\n', 'l', etc. Supports the basic JSON-style escapes.
        if c == "'":
            start_col = col
            i += 1
            col += 1
            if i >= n:
                raise ScriptError("unterminated char literal", line, start_col)
            if src[i] == "\\":
                i += 1
                col += 1
                if i >= n:
                    raise ScriptError("unterminated escape in char literal", line, start_col)
                esc = src[i]
                ch = {"n": "\n", "t": "\t", "r": "\r", "0": "\0",
                      "\\": "\\", "'": "'", '"': '"'}.get(esc)
                if ch is None:
                    raise ScriptError(f"unknown char escape \\{esc}", line, col)
                i += 1
                col += 1
            else:
                ch = src[i]
                i += 1
                col += 1
            if at(i) != "'":
                raise ScriptError("expected closing ' in char literal", line, col)
            i += 1
            col += 1
            tokens.append(Token("CHAR", ch, line, start_col))
            continue

        # Integer literal: optional leading -, then digits. The '-' belongs to
        # the literal only if it's immediately followed by a digit AND not
        # parseable as a binary minus. We disambiguate at parse time by always
        # tokenising '-' as OP and letting the parser handle unary negation.
        if c.isdigit():
            start_col = col
            j = i
            while j < n and src[j].isdigit():
                j += 1
            # Optional type suffix on integer literals: 0i8, 42i32, etc.
            # Longest first so 'i16' beats 'i1'. The suffix only counts when
            # the next character isn't an identifier char — '42i8x' stays a
            # single literal that will fail to parse cleanly downstream.
            for suf in ("i64", "i32", "i16", "i8", "i1"):
                end = j + len(suf)
                if (src[j:end] == suf
                        and (end >= n or not (src[end].isalnum() or src[end] == "_"))):
                    j = end
                    break
            tokens.append(Token("INT", src[i:j], line, start_col))
            col += j - i
            i = j
            continue

        # Dotted identifier: .name (used by &.const_name)
        if c == ".":
            # Already handled as OP if not followed by an identifier char;
            # the OP path covers `..` (range) too. Here we only fall in if we
            # need a special case. Actually: keep '.' as OP; the parser
            # composes &name. — see below.
            # (No-op; the OP block above already consumed it.)
            pass

        # Identifier or keyword
        if c.isalpha() or c == "_":
            start_col = col
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            text = src[i:j]
            kind = "KW" if text in _KEYWORDS else "IDENT"
            tokens.append(Token(kind, text, line, start_col))
            col += j - i
            i = j
            continue

        raise ScriptError(f"unexpected character {c!r}", line, col)

    tokens.append(Token("EOF", "", line, col))
    return tokens


# ---------- Parser ----------

_INT_TYPE_BY_SUFFIX = {
    "i1": I1Type, "i8": I8Type, "i16": I16Type, "i32": I32Type, "i64": I64Type,
}


def _split_int_suffix(text: str) -> tuple[str, str | None]:
    """Split '42i8' into ('42', 'i8'); '42' into ('42', None)."""
    for suf in ("i64", "i32", "i16", "i8", "i1"):
        if text.endswith(suf):
            return text[:-len(suf)], suf
    return text, None


def _int_lit_from_token(text: str, *, negate: bool = False) -> "IntLit":
    """Build an IntLit from a lexed INT token. Suffix-less defaults to i64;
    typed suffixes (`42i8`) carry their explicit width."""
    digits, suf = _split_int_suffix(text)
    ty = _INT_TYPE_BY_SUFFIX[suf]() if suf else I64Type()
    value = int(digits)
    if negate:
        value = -value
    return IntLit(type=ty, value=value)


# Comparison ops -> BinOp.op
_CMP_OPS = {
    "==": "eq", "!=": "ne",
    "<": "slt", "<=": "sle", ">": "sgt", ">=": "sge",
    "<u": "ult", "<=u": "ule", ">u": "ugt", ">=u": "uge",
}


class Parser:
    def __init__(self, tokens: list[Token]):
        self.toks = tokens
        self.pos = 0
        self.param_names: frozenset[str] = frozenset()
        # Disabled in the condition position of if/while and the bounds of
        # for, where `{` always begins the body block. Mirrors Rust's
        # struct-literal-in-cond restriction; parens force the issue when
        # you really do want a literal there: `if (Foo({a: 1}).b == 2) {..}`
        self._struct_init_allowed = True
        # Captured at function entry; used by _return to retype a bare
        # integer literal to whatever the function actually returns.
        self._return_type = None

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

    # -- top-level --

    def parse_function(self) -> Function:
        self.expect("KW", "fn")
        name = self.expect("IDENT").value
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
        self._return_type = ret_ty
        body = self._block()
        # The model's Function has `claims: tuple[Claim, ...] = ()` and
        # `notes: tuple[str, ...] = ()`; both default. We don't author either
        # here — claims have their own surface, notes too.
        return Function(
            name=name,
            params=tuple(params),
            return_type=ret_ty,
            body=tuple(body),
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
            return StructType(name=t.value)
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
        return If(cond=cond, then_body=tuple(then_body), else_body=tuple(else_body))

    def _while(self) -> While:
        self.expect("KW", "while")
        self.expect("OP", "(")
        cond = self._cond_expr()
        self.expect("OP", ")")
        body = self._block()
        return While(cond=cond, body=tuple(body))

    def _for(self) -> For:
        self.expect("KW", "for")
        var = self.expect("IDENT").value
        self.expect("OP", ":")
        ty = self._type(allow_void=False)
        if not isinstance(ty, (I1Type, I8Type, I16Type, I32Type, I64Type)):
            t = self.peek(-1)
            raise ScriptError("for-loop variable must be an integer type", t.line, t.col)
        self.expect("KW", "in")
        lo = self._cond_expr()
        self.expect("OP", "..")
        hi = self._cond_expr()
        body = self._block()
        return For(var=var, type=ty, lo=lo, hi=hi, body=tuple(body))

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
        # Special path: when the entire return expression is a single
        # integer literal (with optional unary minus), retype it to the
        # function's declared return type so `return 0` works for any
        # int-returning function. Composite expressions (binops, calls,
        # etc.) follow normal type rules.
        save = self.pos
        sign = 1
        if self.at("OP", "-") and self.peek(1).kind == "INT":
            self.eat()
            sign = -1
        if (
            self.peek().kind == "INT"
            and self.peek(1).kind == "OP" and self.peek(1).value in ("}", ";")
            and isinstance(self._return_type,
                           (I1Type, I8Type, I16Type, I32Type, I64Type))
        ):
            tok = self.eat()
            digits, _ = _split_int_suffix(tok.value)
            return ReturnExpr(value=IntLit(type=self._return_type,
                                           value=sign * int(digits)))
        self.pos = save
        return ReturnExpr(value=self._expr())

    def _store_stmt(self) -> Store:
        self.expect("KW", "store")
        self.expect("OP", "(")
        ptr = self._expr()
        self.expect("OP", ",")
        value = self._expr()
        self.expect("OP", ")")
        return Store(ptr=ptr, value=value)

    def _with_arena(self) -> WithArena:
        self.expect("KW", "with_arena")
        name = self.expect("IDENT").value
        self.expect("OP", "(")
        self.expect("KW", "capacity")
        self.expect("OP", "=")
        cap = self._expr()
        self.expect("OP", ")")
        body = self._block()
        return WithArena(name=name, capacity=cap, body=tuple(body))

    # -- expressions (Pratt-ish via precedence climbing) --

    def _is_expr_start(self) -> bool:
        t = self.peek()
        if t.kind in ("INT", "CHAR", "IDENT"):
            return True
        if t.kind == "KW" and t.value in (
            "null", "true", "false", "load", "widen", "uwiden", "ptr_offset"
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
        while self.at("OP") and self.peek().value in ("*", "/", "%", "/u"):
            op_tok = self.eat()
            op = {"*": "mul", "/": "sdiv", "%": "srem", "/u": "udiv"}[op_tok.value]
            rhs = self._unary()
            lhs = BinOp(op=op, lhs=lhs, rhs=rhs)
        return lhs

    def _unary(self):
        # Negative integer sugar: -INT becomes IntLit(value=-N). Otherwise
        # parse a postfix.
        if self.at("OP", "-") and self.peek(1).kind == "INT":
            self.eat()
            tok = self.eat()
            return _int_lit_from_token(tok.value, negate=True)
        return self._postfix()

    def _postfix(self):
        e = self._primary()
        while self.at("OP", ".") and self.peek(1).kind == "IDENT":
            self.eat()
            field = self.expect("IDENT").value
            e = FieldRead(value=e, name=field)
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
        # Integer — optional type suffix (e.g. 42i8) carries the int width;
        # otherwise the literal defaults to i64. Suffix-less literals at
        # return position get retyped to the function's return_type later.
        if t.kind == "INT":
            self.eat()
            return _int_lit_from_token(t.value)
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
                case "load":
                    return self._load()
                case "widen":
                    return self._widen(signed=True)
                case "uwiden":
                    return self._widen(signed=False)
                case "ptr_offset":
                    return self._ptr_offset()
        # Identifier — could be call, struct_init, or local/param ref.
        if t.kind == "IDENT":
            self.eat()
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

    def _ptr_offset(self) -> PtrOffset:
        self.expect("KW", "ptr_offset")
        self.expect("OP", "(")
        base = self._expr()
        self.expect("OP", ",")
        offset = self._expr()
        self.expect("OP", ")")
        return PtrOffset(base=base, offset=offset)


# ---------- Public API ----------

def parse_function(src: str) -> Function:
    """Parse a quod-script function definition into a `Function` model.

    Raises `ScriptError` for syntax problems (with line/col); raises
    `pydantic.ValidationError` if the parsed structure violates model
    invariants.
    """
    tokens = tokenize(src)
    parser = Parser(tokens)
    fn = parser.parse_function()
    if not parser.at("EOF"):
        t = parser.peek()
        raise ScriptError(
            f"trailing tokens after function: {t.kind} {t.value!r}",
            t.line, t.col,
        )
    return fn
