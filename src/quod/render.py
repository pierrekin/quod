"""Span/line model + theme-driven renderer for the human-readable program view.

The pretty-printer in model.py emits canonical strings with optional inline
hash labels — that is the machine view. This module produces a structured
form (lines of styled spans) that can be rendered with a hash gutter, syntax
highlighting, and a metadata column. Same source program, different surface.

Pipeline:

    format_*_lines(node) -> Iterator[Line]    # statement-level: one node may
                                              # span several Lines (compound stmts)
    _expr_spans(expr)    -> tuple[Span, ...]  # expression-level: flat
    render(lines, ...)   -> str               # paint + lay out columns

Adding a new theme (ANSI-256 variant, light-on-dark vs dark-on-light, a
future Rich/ncurses backend) touches the `Theme` callable only — every emit
site already knows what kind of token it is producing (see `SpanStyle`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

from quod.hashing import HASH_DISPLAY_LEN, short_hash
from quod.model import (
    Assign,
    BinOp,
    Call,
    ExprStmt,
    ExternFunction,
    For,
    Function,
    I1Type,
    I8Type,
    I16Type,
    I32Type,
    I64Type,
    I8PtrType,
    If,
    IntLit,
    IntRangeClaim,
    Let,
    LocalRef,
    NonNegativeClaim,
    ParamRef,
    Program,
    ReturnExpr,
    ReturnInRangeClaim,
    ReturnInt,
    ShortCircuitAnd,
    ShortCircuitOr,
    StringConstant,
    StringRef,
    While,
    _Node,
    format_claim,
)


SpanStyle = Literal[
    "keyword",       # if / else / let / return / while / for / in / extern / program
    "type",          # i32 / i8* / ...
    "fn_name",       # function being defined or referenced (call / extern / decl)
    "param",         # parameter name (declarations and ParamRef)
    "local",         # let-introduced local (declarations and LocalRef)
    "const_name",    # name of a string constant (decl or &-reference)
    "literal_int",   # integer literals
    "literal_str",   # string-constant value (the quoted literal)
    "op",            # +, -, ==, &&, ||, =, ->, ..
    "punct",         # parens, braces, commas, colons
    "comment",       # // notes
    "section",       # constants: / externs: / functions: headers
    "ws",            # whitespace (themes leave this untouched)
    "meta_label",    # "axiom:" / "witness:" / claim-kind in meta column
    "meta_value",    # value text of a meta entry
    "hash",          # gutter hash text
    "gutter_punct",  # the [] brackets around a gutter hash
]


@dataclass(frozen=True)
class Span:
    text: str
    style: SpanStyle


@dataclass(frozen=True)
class Line:
    """One rendered line of program output.

    owner: the AST node this line "is about" — drives the hash column. Compound
        statement closers (lone `}`) and section labels carry None.
    indent: leading-space count. Not part of `spans`.
    spans: the code column, left-to-right.
    meta: the right column. Empty for most lines; populated for function
        headers (claims summary) and similar.
    """
    owner: _Node | None
    indent: int
    spans: tuple[Span, ...]
    meta: tuple[Span, ...] = ()


Theme = Callable[[Span], str]
plain_theme: Theme = lambda s: s.text


# SGR codes per SpanStyle. A theme is a pure mapping from style to escape
# sequence; new themes (light-bg, high-contrast, 16-color) add a new dict.
_ANSI_DEFAULT: dict[SpanStyle, str] = {
    "keyword":      "38;5;141",   # purple
    "type":         "38;5;110",   # cyan-blue
    "fn_name":      "38;5;179",   # gold
    "param":        "38;5;180",   # tan
    "local":        "38;5;180",
    "const_name":   "38;5;179",
    "literal_int":  "38;5;180",
    "literal_str":  "38;5;108",   # sage
    "op":           "38;5;245",   # mid grey
    "punct":        "38;5;245",
    "comment":      "38;5;102;3", # dim italic
    "section":      "38;5;102;3",
    "ws":           "",
    "meta_label":   "38;5;102",
    "meta_value":   "38;5;108",
    "hash":         "38;5;245",
    "gutter_punct": "38;5;238",   # very dim
}


def make_ansi_theme(palette: dict[SpanStyle, str] = _ANSI_DEFAULT) -> Theme:
    def theme(span: Span) -> str:
        code = palette.get(span.style, "")
        if not code or not span.text:
            return span.text
        return f"\x1b[{code}m{span.text}\x1b[0m"
    return theme


ansi_theme: Theme = make_ansi_theme()


# ---------- Gutter + width helpers ----------

_GUTTER_BLANK = " " * (HASH_DISPLAY_LEN + 4)  # matches "[xxxxxxxxxxxx]  "


def _gutter_spans(owner: _Node | None) -> tuple[Span, ...]:
    if owner is None:
        return (Span(_GUTTER_BLANK, "ws"),)
    return (
        Span("[", "gutter_punct"),
        Span(short_hash(owner), "hash"),
        Span("]  ", "gutter_punct"),
    )


def _visual_width(spans: Iterable[Span]) -> int:
    return sum(len(s.text) for s in spans)


# ---------- Type spans ----------

_TYPE_NAMES: dict[type, str] = {
    I1Type: "i1", I8Type: "i8", I16Type: "i16",
    I32Type: "i32", I64Type: "i64", I8PtrType: "i8*",
}


def _type_span(t) -> Span:
    name = _TYPE_NAMES.get(type(t))
    if name is None:
        raise ValueError(f"unhandled type: {t!r}")
    return Span(name, "type")


# ---------- Expression spans (flat — never wraps) ----------

_BINOP_SYMBOL = {
    "add": "+", "sub": "-", "mul": "*", "sdiv": "/", "udiv": "/u", "srem": "%",
    "slt": "<", "sle": "<=", "sgt": ">", "sge": ">=", "eq": "==", "ne": "!=",
    "ult": "<u", "ule": "<=u", "ugt": ">u", "uge": ">=u",
    "or": "|", "and": "&",
}


def _expr_spans(expr) -> tuple[Span, ...]:
    match expr:
        case IntLit(value=v):
            return (Span(str(v), "literal_int"),)
        case ParamRef(name=n):
            return (Span(n, "param"),)
        case LocalRef(name=n):
            return (Span(n, "local"),)
        case BinOp(op=op, lhs=l, rhs=r):
            return (
                Span("(", "punct"),
                *_expr_spans(l),
                Span(" ", "ws"), Span(_BINOP_SYMBOL[op], "op"), Span(" ", "ws"),
                *_expr_spans(r),
                Span(")", "punct"),
            )
        case ShortCircuitOr(lhs=l, rhs=r):
            return (
                Span("(", "punct"),
                *_expr_spans(l),
                Span(" ", "ws"), Span("||", "op"), Span(" ", "ws"),
                *_expr_spans(r),
                Span(")", "punct"),
            )
        case ShortCircuitAnd(lhs=l, rhs=r):
            return (
                Span("(", "punct"),
                *_expr_spans(l),
                Span(" ", "ws"), Span("&&", "op"), Span(" ", "ws"),
                *_expr_spans(r),
                Span(")", "punct"),
            )
        case Call(function=fn_name, args=args):
            out: list[Span] = [Span(fn_name, "fn_name"), Span("(", "punct")]
            for i, a in enumerate(args):
                if i > 0:
                    out.append(Span(", ", "punct"))
                out.extend(_expr_spans(a))
            out.append(Span(")", "punct"))
            return tuple(out)
        case StringRef(name=n):
            return (Span("&", "op"), Span(n, "const_name"))
    raise ValueError(f"unhandled expr: {expr!r}")


# ---------- Statement lines ----------

def _stmt_lines(stmt, indent: int) -> Iterator[Line]:
    match stmt:
        case ReturnInt(value=v):
            yield Line(stmt, indent, (
                Span("return", "keyword"), Span(" ", "ws"),
                Span(str(v), "literal_int"),
            ))
        case ReturnExpr(value=expr):
            yield Line(stmt, indent, (
                Span("return", "keyword"), Span(" ", "ws"),
                *_expr_spans(expr),
            ))
        case If(cond=c, then_body=tb, else_body=eb):
            yield Line(stmt, indent, (
                Span("if", "keyword"), Span(" (", "punct"),
                *_expr_spans(c),
                Span(") {", "punct"),
            ))
            for s in tb:
                yield from _stmt_lines(s, indent + 2)
            if eb:
                yield Line(None, indent, (Span("} else {", "punct"),))
                for s in eb:
                    yield from _stmt_lines(s, indent + 2)
            yield Line(None, indent, (Span("}", "punct"),))
        case Let(name=n, type=ty, init=init):
            yield Line(stmt, indent, (
                Span("let", "keyword"), Span(" ", "ws"),
                Span(n, "local"), Span(": ", "punct"),
                _type_span(ty), Span(" ", "ws"),
                Span("=", "op"), Span(" ", "ws"),
                *_expr_spans(init),
            ))
        case Assign(name=n, value=v):
            yield Line(stmt, indent, (
                Span(n, "local"), Span(" ", "ws"),
                Span("=", "op"), Span(" ", "ws"),
                *_expr_spans(v),
            ))
        case While(cond=c, body=body):
            yield Line(stmt, indent, (
                Span("while", "keyword"), Span(" (", "punct"),
                *_expr_spans(c),
                Span(") {", "punct"),
            ))
            for s in body:
                yield from _stmt_lines(s, indent + 2)
            yield Line(None, indent, (Span("}", "punct"),))
        case For(var=v, type=ty, lo=lo, hi=hi, body=body):
            yield Line(stmt, indent, (
                Span("for", "keyword"), Span(" ", "ws"),
                Span(v, "local"), Span(": ", "punct"),
                _type_span(ty), Span(" ", "ws"),
                Span("in", "keyword"), Span(" ", "ws"),
                *_expr_spans(lo),
                Span("..", "op"),
                *_expr_spans(hi),
                Span(" {", "punct"),
            ))
            for s in body:
                yield from _stmt_lines(s, indent + 2)
            yield Line(None, indent, (Span("}", "punct"),))
        case ExprStmt(value=v):
            yield Line(stmt, indent, _expr_spans(v))
        case _:
            raise ValueError(f"unhandled stmt: {stmt!r}")


# ---------- Claim / meta-column spans ----------

def _claim_spans(claim) -> tuple[Span, ...]:
    """Compact rendering of one claim for the meta column."""
    match claim:
        case NonNegativeClaim(param=p):
            return (
                Span("non_negative", "meta_label"), Span("(", "punct"),
                Span(p, "param"), Span(")", "punct"),
            )
        case IntRangeClaim(param=p, min=lo, max=hi):
            return (
                Span("int_range", "meta_label"), Span("(", "punct"),
                Span(p, "param"), Span(", [", "punct"),
                Span("-inf" if lo is None else str(lo), "literal_int"),
                Span(", ", "punct"),
                Span("+inf" if hi is None else str(hi), "literal_int"),
                Span("])", "punct"),
            )
        case ReturnInRangeClaim(min=lo, max=hi):
            return (
                Span("return_in_range", "meta_label"), Span("([", "punct"),
                Span("-inf" if lo is None else str(lo), "literal_int"),
                Span(", ", "punct"),
                Span("+inf" if hi is None else str(hi), "literal_int"),
                Span("])", "punct"),
            )
    return (Span(format_claim(claim), "meta_value"),)


_REGIME_ORDER = ("axiom", "witness", "lattice")


def _function_meta(fn: Function) -> tuple[Span, ...]:
    """Meta column for a function header: regime-grouped claim summary.

    Empty if the function has no claims. Enforcement (verify) and
    justification details are left out at this density — see `quod claim ls`
    for the full picture.
    """
    if not fn.claims:
        return ()
    out: list[Span] = [Span("[", "punct")]
    first = True
    for regime in _REGIME_ORDER:
        cs = [c for c in fn.claims if c.regime == regime]
        if not cs:
            continue
        if not first:
            out.append(Span("; ", "punct"))
        first = False
        out.append(Span(f"{regime}: ", "meta_label"))
        for i, c in enumerate(cs):
            if i > 0:
                out.append(Span(", ", "punct"))
            out.extend(_claim_spans(c))
    out.append(Span("]", "punct"))
    return tuple(out)


# ---------- Function / program lines ----------

def format_function_lines(fn: Function, indent: int = 0) -> Iterator[Line]:
    for note in fn.notes:
        yield Line(None, indent, (Span(f"// {note}", "comment"),))
    header: list[Span] = [Span(fn.name, "fn_name"), Span("(", "punct")]
    for i, p in enumerate(fn.params):
        if i > 0:
            header.append(Span(", ", "punct"))
        header.extend((
            Span(p.name, "param"), Span(": ", "punct"), _type_span(p.type),
        ))
    header.extend((
        Span(") ", "punct"), Span("->", "op"), Span(" ", "ws"),
        _type_span(fn.return_type), Span(" {", "punct"),
    ))
    yield Line(fn, indent, tuple(header), meta=_function_meta(fn))
    for s in fn.body:
        yield from _stmt_lines(s, indent + 2)
    yield Line(None, indent, (Span("}", "punct"),))


def _extern_line(ext: ExternFunction, indent: int) -> Line:
    spans: list[Span] = [
        Span("extern", "keyword"), Span(" ", "ws"),
        Span(ext.name, "fn_name"), Span("(", "punct"),
    ]
    types = list(ext.effective_param_types())
    for i, t in enumerate(types):
        if i > 0:
            spans.append(Span(", ", "punct"))
        spans.append(_type_span(t))
    if ext.varargs:
        if types:
            spans.append(Span(", ", "punct"))
        spans.append(Span("...", "op"))
    spans.extend((
        Span(") ", "punct"), Span("->", "op"), Span(" ", "ws"),
        _type_span(ext.return_type),
    ))
    return Line(ext, indent, tuple(spans))


def _constant_line(c: StringConstant, indent: int) -> Line:
    return Line(c, indent, (
        Span(c.name, "const_name"), Span(" ", "ws"),
        Span("=", "op"), Span(" ", "ws"),
        Span(repr(c.value), "literal_str"),
    ))


def format_program_lines(program: Program) -> Iterator[Line]:
    yield Line(program, 0, (Span("program", "keyword"), Span(" {", "punct")))
    if program.constants:
        yield Line(None, 2, (Span("constants:", "section"),))
        for c in program.constants:
            yield _constant_line(c, 4)
    if program.externs:
        yield Line(None, 2, (Span("externs:", "section"),))
        for ext in program.externs:
            yield _extern_line(ext, 4)
    if program.functions:
        yield Line(None, 2, (Span("functions:", "section"),))
        for fn in program.functions:
            yield from format_function_lines(fn, indent=4)
    if not program.constants and not program.functions and not program.externs:
        yield Line(None, 2, (Span("(empty)", "comment"),))
    yield Line(None, 0, (Span("}", "punct"),))


# ---------- Renderer ----------

def render(
    lines: Iterable[Line],
    *,
    theme: Theme = plain_theme,
    hash_col: bool = True,
    meta_col: bool = True,
) -> str:
    """Lay out a stream of Lines into a single string.

    hash_col: prepend a fixed-width gutter `[hash]  ` on each line (blank for
        ownerless lines). False suppresses the column entirely.
    meta_col: pad the code column to the widest line and append per-line meta
        spans. Lines with no meta render flush-left within the code column.
    """
    materialized = list(lines)
    code_widths = [line.indent + _visual_width(line.spans) for line in materialized]
    max_code_width = max(code_widths, default=0) if meta_col else 0

    out: list[str] = []
    for line, code_w in zip(materialized, code_widths):
        gutter = "".join(theme(s) for s in _gutter_spans(line.owner)) if hash_col else ""
        code = (" " * line.indent) + "".join(theme(s) for s in line.spans)
        if meta_col and line.meta:
            pad = " " * max(2, max_code_width - code_w + 2)
            meta = pad + "".join(theme(s) for s in line.meta)
            out.append(f"{gutter}{code}{meta}")
        else:
            out.append(f"{gutter}{code}")
    return "\n".join(out)
