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

    type       := 'i1' | 'i8' '*'? | 'i16' | 'i32' | 'i64'
                | 'u8' | 'u16' | 'u32' | 'u64'
                | 'isize' | 'usize'
                | 'void' | IDENT

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
    mul_expr   := unary_expr (('*' | '/' | '%' | '/u' | '%u') unary_expr)*
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
optional at end of block).

Integer-literal typing: a width-suffixed literal (`0i8`, `42i32`,
`-3i8`) carries its declared type. A bare literal (`0`, `42`) is
poisoned at parse time and resolved by the type-resolution pass that
runs immediately after parsing — its type comes from the operand
context (the param being compared against, the let's declared type,
the function's return type, etc.). A bare literal that the resolver
can't pin to a context is a parse error: write the suffix.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    "ptr_offset", "sizeof", "to", "null", "true", "false", "match",
    # type keywords
    "i1", "i8", "i16", "i32", "i64",
    "u8", "u16", "u32", "u64",
    "isize", "usize",
    "void",
})

# Multi-char operators must be matched before single-char ones.
_MULTI_OPS = (
    "->", "==", "!=", "<=", ">=", "<u", ">u", "<=u", ">=u", "/u", "%u",
    "||", "&&", "..", "::", "=>",
)
_SINGLE_OPS = "(){}[],:;=+-*/%<>.&|?"


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
            for suf in ("isize", "usize",
                        "i64", "i32", "i16", "i8", "i1",
                        "u64", "u32", "u16", "u8"):
                end = j + len(suf)
                if (src[j:end] == suf
                        and (end >= n or not (src[end].isalnum() or src[end] == "_"))):
                    j = end
                    break
            tokens.append(Token("INT", src[i:j], line, start_col))
            col += j - i
            i = j
            continue

        # `.` is tokenised as OP above; the parser composes dotted forms
        # like `&.name` from the OP token.
        if c == ".":
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


# ---------- Type resolution ----------
#
# Phase 2 of parsing. Walks the parsed AST, retyping every poison-marked
# IntLit using the type of its operand context (the enclosing let's
# declared type, the function's return type, the param being compared
# against, etc.). Operates by reflection at the leaves (ParamRef,
# LocalRef, ReturnRef pull their types from scope) plus a same-type
# constraint at every BinOp / shift / IfExpr / boolean combinator.
#
# Limits, deliberate: type inference doesn't reach into struct fields,
# call args, store/load destinations, or widen targets — those still
# resolve at lower time via `_coerce_int_lit`. The script doesn't have
# the program in scope, so it can't see struct/extern signatures; pushing
# a full program-aware resolver would be a different feature.

_INT_TYPE_CLASSES = (
    I1Type, I8Type, I16Type, I32Type, I64Type,
    U8Type, U16Type, U32Type, U64Type,
    IsizeType, UsizeType,
)

_BINOP_CMP = frozenset({"slt", "sle", "sgt", "sge", "eq", "ne",
                        "ult", "ule", "ugt", "uge"})


class _Scope:
    """Per-function (or per-predicate) lookup table for ParamRef /
    LocalRef / ReturnRef types. Locals extend as the resolver walks
    into Let / For; the resolver discards them on the way out."""

    __slots__ = ("params", "locals_", "return_type")

    def __init__(self, params: dict, return_type, locals_=None):
        self.params: dict[str, object] = dict(params)
        self.return_type = return_type
        self.locals_: dict[str, object] = dict(locals_) if locals_ else {}

    def lookup_local(self, name: str):
        return self.locals_.get(name)

    def lookup_param(self, name: str):
        return self.params.get(name)


class _Resolver:
    """Walks the parsed tree, retyping bare IntLits in place via
    model_copy. Carries the parser's `poison_locs` side table so it can
    drop a literal's poison marker the moment it's typed and so the
    final-scan can report the source location of any unresolved
    literal."""

    def __init__(self, poison_locs: dict[int, tuple[int, int]]):
        self.poison_locs = poison_locs

    # -- expression walk --

    def expr(self, e, expected, scope: _Scope):
        """Resolve `e` under `scope`, with `expected` providing the
        outer context's int type (or None). Returns (new_e, type_or_None)
        where the type is the resolved type if known."""
        match e:
            case IntLit():
                return self._int_lit(e, expected)
            case ParamRef(name=name):
                return e, scope.lookup_param(name)
            case LocalRef(name=name):
                return e, scope.lookup_local(name)
            case ReturnRef():
                rt = scope.return_type
                return e, rt if isinstance(rt, _INT_TYPE_CLASSES) else None
            case BinOp(op=op, lhs=lhs, rhs=rhs):
                return self._binop(e, op, lhs, rhs, expected, scope)
            case ShortCircuitOr(lhs=lhs, rhs=rhs):
                new_lhs, _ = self.expr(lhs, I1Type(), scope)
                new_rhs, _ = self.expr(rhs, I1Type(), scope)
                return e.model_copy(update={"lhs": new_lhs, "rhs": new_rhs}), I1Type()
            case ShortCircuitAnd(lhs=lhs, rhs=rhs):
                new_lhs, _ = self.expr(lhs, I1Type(), scope)
                new_rhs, _ = self.expr(rhs, I1Type(), scope)
                return e.model_copy(update={"lhs": new_lhs, "rhs": new_rhs}), I1Type()
            case Not(operand=op):
                new_op, _ = self.expr(op, I1Type(), scope)
                return e.model_copy(update={"operand": new_op}), I1Type()
            case IfExpr(cond=cond, then_value=tv, else_value=ev):
                new_cond, _ = self.expr(cond, I1Type(), scope)
                new_tv, t_ty = self.expr(tv, expected, scope)
                new_ev, e_ty = self.expr(ev, expected or t_ty, scope)
                # Cross-propagate: if one branch resolved and the other is
                # still bare, retype the bare branch to match.
                if t_ty is None and e_ty is not None:
                    new_tv, t_ty = self.expr(tv, e_ty, scope)
                inferred = expected or t_ty or e_ty
                return e.model_copy(
                    update={"cond": new_cond, "then_value": new_tv,
                            "else_value": new_ev}
                ), inferred
            case Call(args=args):
                # The script doesn't carry callee signatures, so we
                # can't type call args from context. Resolve their
                # subtrees (so e.g. nested let-init works) and then
                # release any leftover poison to the lower pass's
                # `_coerce_int_lit`.
                new_args = []
                changed = False
                for a in args:
                    new_a, _ = self.expr(a, None, scope)
                    if new_a is not a:
                        changed = True
                    new_args.append(new_a)
                    self._drop_poison_in(new_a)
                if changed:
                    return e.model_copy(update={"args": tuple(new_args)}), None
                return e, None
            case StructInit(fields=fields):
                # Struct field types live on the StructDef, which the
                # script doesn't see; fall through to lower-time coercion.
                new_fields, changed = self._field_inits(fields, scope, drop_poison=True)
                if changed:
                    return e.model_copy(update={"fields": new_fields}), None
                return e, None
            case EnumInit(fields=fields):
                new_fields, changed = self._field_inits(fields, scope, drop_poison=True)
                if changed:
                    return e.model_copy(update={"fields": new_fields}), None
                return e, None
            case FieldRead(value=inner):
                new_inner, _ = self.expr(inner, None, scope)
                if new_inner is not inner:
                    return e.model_copy(update={"value": new_inner}), None
                return e, None
            case PtrOffset(base=b, offset=o):
                new_b, _ = self.expr(b, None, scope)
                # Offset is i64 by lowering convention.
                new_o, _ = self.expr(o, I64Type(), scope)
                if new_b is not b or new_o is not o:
                    return e.model_copy(update={"base": new_b, "offset": new_o}), None
                return e, None
            case Widen(value=v, target=target):
                # The widen's source type is the operand's natural type;
                # let inner resolution figure it out (no expected
                # propagation across a width change). Result is `target`.
                new_v, _ = self.expr(v, None, scope)
                if new_v is not v:
                    return e.model_copy(update={"value": new_v}), target if isinstance(target, _INT_TYPE_CLASSES) else None
                return e, target if isinstance(target, _INT_TYPE_CLASSES) else None
            case Load(ptr=p):
                new_p, _ = self.expr(p, None, scope)
                if new_p is not p:
                    return e.model_copy(update={"ptr": new_p}), None
                return e, None
            case TryExpr(value=v):
                new_v, _ = self.expr(v, None, scope)
                if new_v is not v:
                    return e.model_copy(update={"value": new_v}), None
                return e, None
            case _:
                # Leaf nodes with no children to walk (CharLit, NullPtr,
                # SizeOf, StringRef) and any other Expr we don't
                # specifically handle. CharLit returns i8.
                if isinstance(e, CharLit):
                    return e, I8Type()
                return e, None

    def _int_lit(self, lit: IntLit, expected):
        if id(lit) not in self.poison_locs:
            return lit, lit.type
        if expected is not None and isinstance(expected, _INT_TYPE_CLASSES):
            new = lit.model_copy(update={"type": expected})
            del self.poison_locs[id(lit)]
            return new, expected
        return lit, None

    def _drop_poison_in(self, e):
        """Mark every IntLit reachable from `e` as 'lower-time will fix
        this'. Used for contexts the script-time resolver can't reach
        (Call args without a callee signature, struct/enum field
        initializers without a struct def, store/field-set
        destinations). The literal keeps its i64 placeholder; the
        lower pass's `_coerce_int_lit` retypes it once the destination
        type is in scope."""
        for lit in _iter_int_lits(e):
            self.poison_locs.pop(id(lit), None)

    def _binop(self, e, op, lhs, rhs, expected, scope):
        if op in _BINOP_CMP:
            # cmp: result is i1; operand width is shared. Resolve LHS
            # first with no outer expectation (since `expected` here is
            # the *cmp result's* type, not the operands'); then use LHS's
            # inferred type as the expectation for RHS. Re-resolve LHS if
            # RHS pinned a type that LHS missed.
            new_lhs, l_ty = self.expr(lhs, None, scope)
            new_rhs, r_ty = self.expr(rhs, l_ty, scope)
            if l_ty is None and r_ty is not None:
                new_lhs, l_ty = self.expr(lhs, r_ty, scope)
            return e.model_copy(update={"lhs": new_lhs, "rhs": new_rhs}), I1Type()
        # arith / bitwise / shift: operands and result share a type.
        # `expected` may flow from the surrounding context (the let's
        # declared type, the return type, etc.).
        new_lhs, l_ty = self.expr(lhs, expected, scope)
        new_rhs, r_ty = self.expr(rhs, expected or l_ty, scope)
        if expected is None and l_ty is None and r_ty is not None:
            new_lhs, l_ty = self.expr(lhs, r_ty, scope)
        result_ty = expected or l_ty or r_ty
        return e.model_copy(update={"lhs": new_lhs, "rhs": new_rhs}), result_ty

    def _field_inits(self, fields, scope, *, drop_poison: bool = False):
        """Walk a tuple of FieldInit; struct/enum field types aren't
        visible at script time so we resolve children with no
        expectation. With `drop_poison=True`, leftover bare literals are
        released to lower-time coercion (StructInit / EnumInit reach
        here). Returns (new_fields_tuple, any_changed)."""
        out = []
        changed = False
        for fi in fields:
            new_v, _ = self.expr(fi.value, None, scope)
            if drop_poison:
                self._drop_poison_in(new_v)
            if new_v is not fi.value:
                out.append(fi.model_copy(update={"value": new_v}))
                changed = True
            else:
                out.append(fi)
        return tuple(out), changed

    # -- statement walk --

    def block(self, b: Block, scope: _Scope) -> Block:
        new_stmts = []
        for s in b.stmts:
            new_stmts.append(self.stmt(s, scope))
        return b.model_copy(update={"stmts": tuple(new_stmts)})

    def stmt(self, s, scope: _Scope):
        match s:
            case Let(name=name, type=ty, init=init):
                new_init = init
                if init is not None:
                    expected = ty if isinstance(ty, _INT_TYPE_CLASSES) else None
                    new_init, _ = self.expr(init, expected, scope)
                # Bind the local in scope for subsequent statements.
                scope.locals_[name] = ty
                if new_init is not init:
                    return s.model_copy(update={"init": new_init})
                return s
            case Assign(name=name, value=v):
                ty = scope.lookup_local(name) or scope.lookup_param(name)
                expected = ty if isinstance(ty, _INT_TYPE_CLASSES) else None
                new_v, _ = self.expr(v, expected, scope)
                if new_v is not v:
                    return s.model_copy(update={"value": new_v})
                return s
            case ReturnExpr(value=v):
                rt = scope.return_type
                expected = rt if isinstance(rt, _INT_TYPE_CLASSES) else None
                new_v, _ = self.expr(v, expected, scope)
                if new_v is not v:
                    return s.model_copy(update={"value": new_v})
                return s
            case ExprStmt(value=v):
                new_v, _ = self.expr(v, None, scope)
                if new_v is not v:
                    return s.model_copy(update={"value": new_v})
                return s
            case If(cond=cond, then_body=tb, else_body=eb):
                new_cond, _ = self.expr(cond, I1Type(), scope)
                new_tb = self.block(tb, _Scope(scope.params, scope.return_type, scope.locals_))
                new_eb = self.block(eb, _Scope(scope.params, scope.return_type, scope.locals_))
                return s.model_copy(update={
                    "cond": new_cond, "then_body": new_tb, "else_body": new_eb,
                })
            case While(cond=cond, body=b):
                new_cond, _ = self.expr(cond, I1Type(), scope)
                new_body = self.block(b, _Scope(scope.params, scope.return_type, scope.locals_))
                return s.model_copy(update={"cond": new_cond, "body": new_body})
            case For(var=var, type=ty, lo=lo, hi=hi, body=b):
                expected = ty if isinstance(ty, _INT_TYPE_CLASSES) else None
                new_lo, _ = self.expr(lo, expected, scope)
                new_hi, _ = self.expr(hi, expected, scope)
                inner = _Scope(scope.params, scope.return_type, scope.locals_)
                inner.locals_[var] = ty
                new_body = self.block(b, inner)
                return s.model_copy(update={
                    "lo": new_lo, "hi": new_hi, "body": new_body,
                })
            case FieldSet(value=v):
                # Struct field types aren't visible to the script;
                # rely on lower-time coercion.
                new_v, _ = self.expr(v, None, scope)
                self._drop_poison_in(new_v)
                if new_v is not v:
                    return s.model_copy(update={"value": new_v})
                return s
            case Store(ptr=p, value=v):
                new_p, _ = self.expr(p, None, scope)
                new_v, _ = self.expr(v, None, scope)
                self._drop_poison_in(new_v)
                if new_p is not p or new_v is not v:
                    return s.model_copy(update={"ptr": new_p, "value": new_v})
                return s
            case WithArena(capacity=cap, body=b):
                # capacity is i64 by convention.
                new_cap, _ = self.expr(cap, I64Type(), scope)
                new_body = self.block(b, _Scope(scope.params, scope.return_type, scope.locals_))
                return s.model_copy(update={"capacity": new_cap, "body": new_body})
            case Match(scrutinee=sc, arms=arms):
                new_sc, _ = self.expr(sc, None, scope)
                new_arms = []
                for arm in arms:
                    new_body = self.block(
                        arm.body,
                        _Scope(scope.params, scope.return_type, scope.locals_),
                    )
                    new_arms.append(arm.model_copy(update={"body": new_body}))
                return s.model_copy(update={
                    "scrutinee": new_sc, "arms": tuple(new_arms),
                })
            case _:
                return s

    # -- final scan: any leftover poison is an error --

    def assert_no_poison(self, root):
        """Walk `root` (Function | Block | Expr) and raise ScriptError if
        any IntLit's id is still in the poison set."""
        for lit_id, (line, col) in list(self.poison_locs.items()):
            # Only literals reachable from `root` matter; entries for
            # original-tree IntLits that the resolver has copied away
            # have already been removed via `del`. The rest are real
            # unresolved bare literals.
            if _find_intlit_with_id(root, lit_id):
                raise ScriptError(
                    "bare integer literal needs a width suffix here — "
                    "no operand context to infer the type from",
                    line, col,
                )


def _find_intlit_with_id(node, target_id: int) -> bool:
    """True if some IntLit reachable from `node` has the given id()."""
    for lit in _iter_int_lits(node):
        if id(lit) == target_id:
            return True
    return False


def _iter_int_lits(node):
    """Yield every IntLit reachable from `node`. Walks frozen-Pydantic
    nodes via `__pydantic_fields__` so it stays generic over the model."""
    if isinstance(node, IntLit):
        yield node
        return
    fields = getattr(node, "__pydantic_fields__", None)
    if fields is None:
        return
    for fname in fields:
        v = getattr(node, fname, None)
        if v is None:
            continue
        if isinstance(v, tuple):
            for item in v:
                yield from _iter_int_lits(item)
        elif hasattr(v, "__pydantic_fields__"):
            yield from _iter_int_lits(v)


# ---------- Public API ----------

def parse_function(src: str, *, enum_names: frozenset[str] = frozenset()) -> Function:
    """Parse a quod-script function definition into a `Function` model.

    Two phases: tokenise+parse to produce the AST, then walk it to resolve
    every bare integer literal's type from operand context (the let's
    declared type, the function's return type, the param being compared
    against, ...). A bare literal whose context can't pin a type is a
    `ScriptError` — write the suffix.

    Raises `ScriptError` for syntax problems and unresolved bare literals
    (both with line/col); raises `pydantic.ValidationError` if the parsed
    structure violates model invariants.

    `enum_names` lets the caller specify which custom type names refer to
    enums (so a bare `Maybe` in type position becomes EnumType("Maybe")
    rather than StructType("Maybe")). The CLI passes the program's
    current enum names; standalone use can leave it empty.
    """
    tokens = tokenize(src)
    parser = Parser(tokens, enum_names=enum_names)
    fn = parser.parse_function()
    if not parser.at("EOF"):
        t = parser.peek()
        raise ScriptError(
            f"trailing tokens after function: {t.kind} {t.value!r}",
            t.line, t.col,
        )
    resolver = _Resolver(parser.poison_locs)
    scope = _Scope(
        params={p.name: p.type for p in fn.params},
        return_type=fn.return_type,
    )
    new_body = resolver.block(fn.body, scope)
    fn = fn.model_copy(update={"body": new_body})
    resolver.assert_no_poison(fn)
    return fn


def parse_predicate(
    src: str, *,
    param_types: dict[str, "IntType"],
    return_type=None,
):
    """Parse a quod-script expression as a predicate body and resolve
    every bare integer literal against the function's signature.

    `param_types` maps each in-scope param name to its int type — bare
    identifiers in `param_types.keys()` parse as `ParamRef`, and a
    comparison like `x >= 0` retypes the literal to match `x`'s type.
    `return_type` is the enclosing function's return type; the keyword
    `return` parses as `ReturnRef`, and `return >= 0` retypes the
    literal accordingly. Pass `None` only when the call site knows no
    `ReturnRef` can appear (extern targets, etc.) — a `ReturnRef` with
    no `return_type` raises.

    Anything else (locals, calls, aggregate access) is rejected by the
    predicate validator at the call site — `parse_predicate` only
    handles the syntactic shape and integer-literal typing.

    Raises `ScriptError` on syntax errors and unresolved bare literals.
    """
    tokens = tokenize(src)
    parser = Parser(tokens)
    parser.param_names = frozenset(param_types.keys())
    expr = parser._expr()
    if not parser.at("EOF"):
        t = parser.peek()
        raise ScriptError(
            f"trailing tokens after predicate: {t.kind} {t.value!r}",
            t.line, t.col,
        )
    resolver = _Resolver(parser.poison_locs)
    scope = _Scope(params=dict(param_types), return_type=return_type)
    new_expr, _ = resolver.expr(expr, None, scope)
    resolver.assert_no_poison(new_expr)
    return new_expr
