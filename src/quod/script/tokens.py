"""quod-script tokens and lexer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Token:
    kind: str          # 'IDENT', 'INT', 'FLOAT', 'CHAR', 'DOT_IDENT', 'OP', 'KW', 'EOF'
    value: str
    line: int
    col: int


_KEYWORDS = frozenset({
    "fn", "let", "if", "else", "while", "for", "in", "return",
    "store", "with_arena", "capacity", "load", "cast",
    "ptr_offset", "sizeof", "to", "null", "true", "false", "match",
    # type keywords
    "i1", "i8", "i16", "i32", "i64",
    "u8", "u16", "u32", "u64",
    "isize", "usize",
    "f32", "f64",
    "void",
    # float-literal special-value keywords. Suffix names the width;
    # the parser consumes one of these and produces a `FloatLit` with
    # the canonical bit pattern. Negative inf/nan: `-inf_f64`,
    # `-nan_f32` — the `-` is a unary-minus op tokenised separately,
    # which the parser folds via sign-bit flip.
    "inf_f32", "inf_f64", "nan_f32", "nan_f64",
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

        # Numeric literal: integer or float. We always tokenise '-' as
        # OP and let the parser handle unary negation, so the literal
        # itself never has a leading sign.
        #
        # Integer:  `42`, `42i32`, `0xff`, `0xffu64`
        # Float:    `1.5`, `1.5f32`, `1e10`, `1e10f64`,
        #           `0x1.8p+1`, `0x1.8p+1f32`
        # Hex requires a `p`-style exponent to be a hex float
        # (matches C99); without `p` it's a hex int.
        if c.isdigit():
            start_col = col
            j = i
            is_hex = (src[j:j + 2] in ("0x", "0X"))
            if is_hex:
                j += 2
                while j < n and (src[j].isalnum() or src[j] == "."):
                    # hex digits, '.', and 'p'/'P' all alnum or '.'
                    j += 1
                # Hex floats have a `p`/`P` exponent that may be
                # followed by a sign + digits (`0x1.8p+1`). The
                # alnum-loop stopped at the `+`/`-`; continue if
                # this is the case.
                if j < n and src[j] in "+-" and j > i + 2 and src[j - 1] in "pP":
                    j += 1
                    while j < n and src[j].isdigit():
                        j += 1
            else:
                while j < n and src[j].isdigit():
                    j += 1
                # Decimal '.' starts a float; lookahead requires a
                # following digit to avoid consuming the field-access
                # `.` in `42.foo` (not currently valid script, but
                # keeps the lexer consistent).
                if j < n and src[j] == "." and (j + 1 < n and src[j + 1].isdigit()):
                    j += 1
                    while j < n and src[j].isdigit():
                        j += 1
                # Scientific exponent: `e[+-]?digits`.
                if j < n and src[j] in "eE":
                    k = j + 1
                    if k < n and src[k] in "+-":
                        k += 1
                    if k < n and src[k].isdigit():
                        j = k
                        while j < n and src[j].isdigit():
                            j += 1
            body_text = src[i:j]
            # Float literals are recognized by `.`, `p`/`P` (hex
            # exponent), or `e`/`E` (decimal exponent) anywhere in the
            # body. Hex int literals (`0xff`) have none of these.
            is_float = (
                "." in body_text
                or (is_hex and ("p" in body_text or "P" in body_text))
                or (not is_hex and ("e" in body_text or "E" in body_text))
            )
            if is_float:
                # Optional float suffix. `f32` / `f64` only count when
                # the following character isn't an identifier char.
                for suf in ("f32", "f64"):
                    end = j + len(suf)
                    if (src[j:end] == suf
                            and (end >= n or not (src[end].isalnum() or src[end] == "_"))):
                        j = end
                        break
                tokens.append(Token("FLOAT", src[i:j], line, start_col))
            else:
                # Integer suffix. Longest first so 'i16' beats 'i1'.
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
