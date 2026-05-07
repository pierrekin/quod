"""quod-script tokens and lexer."""

from __future__ import annotations

from dataclasses import dataclass


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
