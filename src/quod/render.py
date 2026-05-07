"""Span/line model + theme-driven renderer for the human-readable program view.

The pretty-printer in `quod.model.pretty` emits canonical strings with
optional inline hash labels — that is the machine view. This module
produces a structured form (lines of styled spans) that can be rendered
with a hash gutter, syntax highlighting, and a metadata column. Same
source program, different surface.

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
    Block,
    Call,
    CFn,
    CStyleFor,
    CUnit,
    Equivalence,
    ExprStmt,
    format_c_fn,
    ExternFunction,
    FieldRead,
    FieldSet,
    LoadField,
    StoreField,
    For,
    Function,
    I1Type,
    I8Type,
    I16Type,
    I32Type,
    I64Type,
    U8Type,
    U16Type,
    U32Type,
    U64Type,
    IsizeType,
    UsizeType,
    I8PtrType,
    If,
    IfExpr,
    IntLit,
    Let,
    CharLit,
    Load,
    LocalRef,
    Not,
    NullPtr,
    ParamRef,
    PredicateClaim,
    Return,
    VoidType,
    Program,
    PtrOffset,
    ReturnExpr,
    ReturnRef,
    ShortCircuitAnd,
    ShortCircuitOr,
    StringConstant,
    StringRef,
    Store,
    EnumDef,
    EnumInit,
    EnumType,
    Match,
    SizeOf,
    TryExpr,
    StructDef,
    StructInit,
    StructType,
    SelfType,
    TraitCall,
    TypeParamRef,
    Unreachable,
    While,
    Widen,
    WithArena,
    _Node,
    format_claim_metadata,
    format_equivalence_metadata,
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
    "ok",            # success status (e.g. `ok` in `quod claim verify`)
    "warn",          # failure / warning status (e.g. `FAIL` in `quod claim verify`)
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
    "ok":           "38;5;108",   # sage (matches literal_str — "things are good")
    "warn":         "38;5;167;1", # bold red-orange
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
    I32Type: "i32", I64Type: "i64",
    U8Type: "u8", U16Type: "u16", U32Type: "u32", U64Type: "u64",
    IsizeType: "isize", UsizeType: "usize",
    I8PtrType: "i8*",
    VoidType: "void",
}


def type_span(t) -> Span:
    if isinstance(t, StructType):
        if t.type_args:
            args = ", ".join(_format_type_short(a) for a in t.type_args)
            return Span(f"{t.name}<{args}>", "type")
        return Span(t.name, "type")
    if isinstance(t, EnumType):
        if t.type_args:
            args = ", ".join(_format_type_short(a) for a in t.type_args)
            return Span(f"{t.name}<{args}>", "type")
        return Span(t.name, "type")
    if isinstance(t, TypeParamRef):
        return Span(t.name, "type")
    if isinstance(t, SelfType):
        return Span("Self", "type")
    name = _TYPE_NAMES.get(type(t))
    if name is None:
        raise ValueError(f"unhandled type: {t!r}")
    return Span(name, "type")


def _format_type_short(t) -> str:
    """Compact type-to-string for use inside angle brackets."""
    from quod.model import _format_type
    return _format_type(t)


# ---------- Expression spans (flat — never wraps) ----------

_BINOP_SYMBOL = {
    "add": "+", "sub": "-", "mul": "*", "sdiv": "/", "udiv": "/u", "srem": "%",
    "slt": "<", "sle": "<=", "sgt": ">", "sge": ">=", "eq": "==", "ne": "!=",
    "ult": "<u", "ule": "<=u", "ugt": ">u", "uge": ">=u",
    "or": "|", "and": "&", "xor": "^",
    "shl": "<<", "ashr": ">>", "lshr": ">>l",
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
        case IfExpr(cond=cond, then_value=t, else_value=e):
            return (
                Span("(", "punct"),
                *_expr_spans(cond),
                Span(" ", "ws"), Span("?", "op"), Span(" ", "ws"),
                *_expr_spans(t),
                Span(" ", "ws"), Span(":", "op"), Span(" ", "ws"),
                *_expr_spans(e),
                Span(")", "punct"),
            )
        case Not(operand=op):
            return (Span("!", "op"), *_expr_spans(op))
        case ReturnRef():
            return (Span("<return>", "param"),)
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
        case FieldRead(value=inner, name=fname):
            return (
                *_expr_spans(inner),
                Span(".", "op"),
                Span(fname, "param"),
            )
        case LoadField(ptr=p, struct_type=tname, name=fname):
            return (
                *_expr_spans(p),
                Span("->", "op"),
                Span(tname, "type"),
                Span(".", "op"),
                Span(fname, "param"),
            )
        case StructInit(type=tname, fields=field_inits):
            out: list[Span] = [
                Span(tname, "type"), Span(" {", "punct"), Span(" ", "ws"),
            ]
            for i, fi in enumerate(field_inits):
                if i > 0:
                    out.append(Span(", ", "punct"))
                out.extend((
                    Span(fi.name, "param"), Span(": ", "punct"),
                    *_expr_spans(fi.value),
                ))
            out.extend((Span(" ", "ws"), Span("}", "punct")))
            return tuple(out)
        case PtrOffset(base=b, offset=o):
            return (
                Span("(", "punct"),
                *_expr_spans(b),
                Span(" + ", "op"),
                *_expr_spans(o),
                Span(")", "punct"),
            )
        case Widen(value=v, target=t, signed=signed):
            kind = "widen" if signed else "uwiden"
            return (
                Span(kind, "fn_name"), Span("(", "punct"),
                *_expr_spans(v),
                Span(" to ", "keyword"),
                type_span(t),
                Span(")", "punct"),
            )
        case Load(ptr=p, type=t):
            return (
                Span("load", "fn_name"), Span("[", "punct"),
                type_span(t),
                Span("](", "punct"),
                *_expr_spans(p),
                Span(")", "punct"),
            )
        case NullPtr():
            return (Span("null", "keyword"),)
        case CharLit(value=v):
            return (Span(repr(v), "literal_int"),)
        case SizeOf(type=t):
            return (
                Span("sizeof", "fn_name"), Span("[", "punct"),
                type_span(t), Span("]", "punct"),
            )
        case TryExpr(value=v):
            return (*_expr_spans(v), Span("?", "op"))
        case EnumInit(enum=ename, variant=vname, fields=field_inits):
            head = (
                Span(ename, "type"), Span("::", "op"), Span(vname, "fn_name"),
            )
            if not field_inits:
                return head
            out: list[Span] = list(head)
            out.append(Span("(", "punct"))
            for i, fi in enumerate(field_inits):
                if i > 0:
                    out.append(Span(", ", "punct"))
                out.extend((
                    Span(fi.name, "param"), Span(": ", "punct"),
                    *_expr_spans(fi.value),
                ))
            out.append(Span(")", "punct"))
            return tuple(out)
        case TraitCall(trait=tname, method=mname, dispatch_type=dt, args=args):
            out: list[Span] = [
                Span(tname, "type"), Span("::", "op"),
                Span(mname, "fn_name"),
                Span("[", "punct"), type_span(dt), Span("]", "punct"),
                Span("(", "punct"),
            ]
            for i, a in enumerate(args):
                if i > 0:
                    out.append(Span(", ", "punct"))
                out.extend(_expr_spans(a))
            out.append(Span(")", "punct"))
            return tuple(out)
    raise ValueError(f"unhandled expr: {expr!r}")


# ---------- Statement lines ----------

def _stmt_lines(stmt, indent: int) -> Iterator[Line]:
    match stmt:
        case Return():
            yield Line(stmt, indent, (Span("return", "keyword"),))
        case Unreachable():
            yield Line(stmt, indent, (Span("unreachable", "keyword"),))
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
            for s in tb.stmts:
                yield from _stmt_lines(s, indent + 2)
            if eb.stmts:
                yield Line(None, indent, (Span("} else {", "punct"),))
                for s in eb.stmts:
                    yield from _stmt_lines(s, indent + 2)
            yield Line(None, indent, (Span("}", "punct"),))
        case Let(name=n, type=ty, init=init):
            spans: list[Span] = [
                Span("let", "keyword"), Span(" ", "ws"),
                Span(n, "local"), Span(": ", "punct"),
                type_span(ty),
            ]
            if init is not None:
                spans.extend((
                    Span(" ", "ws"), Span("=", "op"), Span(" ", "ws"),
                    *_expr_spans(init),
                ))
            yield Line(stmt, indent, tuple(spans))
        case Assign(name=n, value=v):
            yield Line(stmt, indent, (
                Span(n, "local"), Span(" ", "ws"),
                Span("=", "op"), Span(" ", "ws"),
                *_expr_spans(v),
            ))
        case FieldSet(local=loc, name=fname, value=v):
            yield Line(stmt, indent, (
                Span(loc, "local"), Span(".", "op"), Span(fname, "param"),
                Span(" ", "ws"), Span("=", "op"), Span(" ", "ws"),
                *_expr_spans(v),
            ))
        case Store(ptr=p, value=v):
            yield Line(stmt, indent, (
                Span("store", "fn_name"), Span("(", "punct"),
                *_expr_spans(p),
                Span(", ", "punct"),
                *_expr_spans(v),
                Span(")", "punct"),
            ))
        case StoreField(ptr=p, struct_type=tname, name=fname, value=v):
            yield Line(stmt, indent, (
                *_expr_spans(p),
                Span("->", "op"), Span(tname, "type"), Span(".", "op"),
                Span(fname, "param"),
                Span(" ", "ws"), Span("=", "op"), Span(" ", "ws"),
                *_expr_spans(v),
            ))
        case While(cond=c, body=body):
            yield Line(stmt, indent, (
                Span("while", "keyword"), Span(" (", "punct"),
                *_expr_spans(c),
                Span(") {", "punct"),
            ))
            for s in body.stmts:
                yield from _stmt_lines(s, indent + 2)
            yield Line(None, indent, (Span("}", "punct"),))
        case For(var=v, type=ty, lo=lo, hi=hi, body=body):
            yield Line(stmt, indent, (
                Span("for", "keyword"), Span(" ", "ws"),
                Span(v, "local"), Span(": ", "punct"),
                type_span(ty), Span(" ", "ws"),
                Span("in", "keyword"), Span(" ", "ws"),
                *_expr_spans(lo),
                Span("..", "op"),
                *_expr_spans(hi),
                Span(" {", "punct"),
            ))
            for s in body.stmts:
                yield from _stmt_lines(s, indent + 2)
            yield Line(None, indent, (Span("}", "punct"),))
        case ExprStmt(value=v):
            yield Line(stmt, indent, _expr_spans(v))
        case WithArena(name=n, capacity=cap, body=body):
            yield Line(stmt, indent, (
                Span("with_arena", "keyword"), Span(" ", "ws"),
                Span(n, "local"), Span(" ", "ws"),
                Span("=", "op"), Span(" ", "ws"),
                Span("arena_new", "fn_name"), Span("(", "punct"),
                *_expr_spans(cap),
                Span(") {", "punct"),
            ))
            for s in body.stmts:
                yield from _stmt_lines(s, indent + 2)
            yield Line(None, indent, (Span("}", "punct"),))
        case Match(scrutinee=scrut, arms=arms):
            yield Line(stmt, indent, (
                Span("match", "keyword"), Span(" ", "ws"),
                *_expr_spans(scrut),
                Span(" {", "punct"),
            ))
            for arm in arms:
                head: list[Span] = [Span(arm.variant, "fn_name")]
                if arm.bindings:
                    head.append(Span("(", "punct"))
                    for i, b in enumerate(arm.bindings):
                        if i > 0:
                            head.append(Span(", ", "punct"))
                        head.append(Span(b, "local"))
                    head.append(Span(")", "punct"))
                head.extend((Span(" ", "ws"), Span("=>", "op"),
                             Span(" {", "punct")))
                yield Line(None, indent + 2, tuple(head))
                for s in arm.body.stmts:
                    yield from _stmt_lines(s, indent + 4)
                yield Line(None, indent + 2, (Span("}", "punct"),))
            yield Line(None, indent, (Span("}", "punct"),))
        case CStyleFor(init=init, cond=cond, inc=inc, body=body):
            # Layer-B `c.for_general` — a structured-form C-style for
            # loop. Renders header on one line, body indented. The
            # init/inc statement spans come from a recursive single-
            # line call into _stmt_lines stripped of indent.
            init_spans = _stmt_inline_spans(init) if init is not None else ()
            inc_spans = _stmt_inline_spans(inc) if inc is not None else ()
            cond_spans = _expr_spans(cond) if cond is not None else ()
            inner = body if isinstance(body, Block) else body.block
            yield Line(stmt, indent, (
                Span("c.for_general", "keyword"), Span(" (", "punct"),
                *init_spans, Span("; ", "punct"),
                *cond_spans, Span("; ", "punct"),
                *inc_spans, Span(") {", "punct"),
            ))
            for s in inner.stmts:
                yield from _stmt_lines(s, indent + 2)
            yield Line(None, indent, (Span("}", "punct"),))
        case _:
            raise ValueError(f"unhandled stmt: {stmt!r}")


def _stmt_inline_spans(stmt) -> tuple:
    """Collect a statement's spans for inline rendering inside a
    for-header. We strip the leading indent and trailing newline by
    grabbing the first line's spans only — works for the simple
    Let/Assign cases that occupy a for-loop's init / inc slots."""
    lines = list(_stmt_lines(stmt, 0))
    if not lines:
        return ()
    return lines[0].spans


# ---------- Claim / meta-column spans ----------

def claim_spans(claim) -> tuple[Span, ...]:
    """Compact rendering of one claim for the meta column.

    Routes through the predicate-sugar recognizer: matches the
    canonicalized predicate against the sugar table and renders the
    friendly form (`non_negative(p)` / `int_range(...)` /
    `return_in_range(...)`) when matched, otherwise the raw expression.
    """
    return predicate_spans(claim.expr)


_REGIME_ORDER = ("axiom", "witness", "lattice")


def _function_meta(fn: Function) -> tuple[Span, ...]:
    """Meta column for a function header: regime-grouped claim summary.

    Empty if the function has no claims. Enforcement (verify) and
    justification details are left out at this density — see `quod claim ls`
    for the full picture.
    """
    return _claim_meta(fn.claims)


def _claim_meta(claims: tuple) -> tuple[Span, ...]:
    """Regime-grouped claim summary."""
    if not claims:
        return ()
    out: list[Span] = [Span("[", "punct")]
    first = True
    for regime in _REGIME_ORDER:
        cs = [c for c in claims if c.regime == regime]
        if not cs:
            continue
        if not first:
            out.append(Span("; ", "punct"))
        first = False
        out.append(Span(f"{regime}: ", "meta_label"))
        for i, c in enumerate(cs):
            if i > 0:
                out.append(Span(", ", "punct"))
            out.extend(claim_spans(c))
    out.append(Span("]", "punct"))
    return tuple(out)


def claim_full_spans(claim) -> tuple[Span, ...]:
    """`claim_spans` + the trailing `{regime=..., enforcement=..., justification=...}`
    block for non-default fields. The full per-claim rendering."""
    head = claim_spans(claim)
    suffix = format_claim_metadata(claim)
    if not suffix:
        return head
    return (*head, Span(suffix, "meta_value"))


# Predicate sugar recognizer lives in quod.predicate.render — see comment
# at top of render.py for the rationale. Imported here (after Span and
# _expr_spans are defined) so claim_spans above can resolve the call.
from quod.predicate.render import predicate_spans, recognize_predicate  # noqa: E402, F401


# ---------- Row builders (reusable across listing commands) ----------

def hash_brackets(node: _Node) -> tuple[Span, ...]:
    """`[hash]` spans for inline use (e.g. `quod fn ls`, `quod show` default)."""
    return (
        Span("[", "gutter_punct"),
        Span(short_hash(node), "hash"),
        Span("]", "gutter_punct"),
    )


def function_signature_spans(fn: Function) -> tuple[Span, ...]:
    """`name(p: type, ...) -> ret` — no trailing brace, no claims summary."""
    out: list[Span] = [Span(fn.name, "fn_name"), Span("(", "punct")]
    for i, p in enumerate(fn.params):
        if i > 0:
            out.append(Span(", ", "punct"))
        out.extend((Span(p.name, "param"), Span(": ", "punct"), type_span(p.type)))
    out.extend((
        Span(") ", "punct"), Span("->", "op"), Span(" ", "ws"),
        type_span(fn.return_type),
    ))
    return tuple(out)


def extern_signature_spans(ext: ExternFunction) -> tuple[Span, ...]:
    """`name(types..., ...) -> ret  [linkage]` — leading `extern ` keyword is the caller's choice."""
    out: list[Span] = [Span(ext.name, "fn_name"), Span("(", "punct")]
    types = list(ext.effective_param_types())
    for i, t in enumerate(types):
        if i > 0:
            out.append(Span(", ", "punct"))
        out.append(type_span(t))
    if ext.varargs:
        if types:
            out.append(Span(", ", "punct"))
        out.append(Span("...", "op"))
    out.extend((
        Span(") ", "punct"), Span("->", "op"), Span(" ", "ws"),
        type_span(ext.return_type),
    ))
    out.extend((Span("  ", "ws"), Span(_linkage_tag(ext.linkage), "comment")))
    return tuple(out)


def _linkage_tag(linkage) -> str:
    """Bracketed shorthand for an extern's linkage. Drops the `linkage.`
    prefix so the tag stays compact (`[libc]`, `[runtime]`)."""
    kind = linkage.kind  # e.g. "linkage.libc"
    short = kind.split(".", 1)[1] if "." in kind else kind
    return f"[{short}]"


def constant_spans(c: StringConstant) -> tuple[Span, ...]:
    return (
        Span(c.name, "const_name"), Span(" ", "ws"),
        Span("=", "op"), Span(" ", "ws"),
        Span(repr(c.value), "literal_str"),
    )


def struct_def_spans(sd: StructDef) -> tuple[Span, ...]:
    """`struct Name { f1: t1, f2: t2 }` rendered as a single line."""
    out: list[Span] = [
        Span("struct", "keyword"), Span(" ", "ws"),
        Span(sd.name, "type"), Span(" {", "punct"), Span(" ", "ws"),
    ]
    for i, f in enumerate(sd.fields):
        if i > 0:
            out.append(Span(", ", "punct"))
        out.extend((
            Span(f.name, "param"), Span(": ", "punct"), type_span(f.type),
        ))
    out.extend((Span(" ", "ws"), Span("}", "punct")))
    return tuple(out)


def paint(spans: Iterable[Span], theme: Theme = plain_theme) -> str:
    """Render a span sequence to a single string. The one-liner cousin of `render`."""
    return "".join(theme(s) for s in spans)


# ---------- Function / program lines ----------

def format_function_lines(fn: Function, indent: int = 0) -> Iterator[Line]:
    for note in fn.notes:
        yield Line(None, indent, (Span(f"// {note}", "comment"),))
    header = (*function_signature_spans(fn), Span(" {", "punct"))
    yield Line(fn, indent, header, meta=_function_meta(fn))
    for s in fn.body.stmts:
        yield from _stmt_lines(s, indent + 2)
    yield Line(None, indent, (Span("}", "punct"),))


def _extern_line(ext: ExternFunction, indent: int) -> Line:
    spans = (Span("extern", "keyword"), Span(" ", "ws"), *extern_signature_spans(ext))
    return Line(ext, indent, spans, meta=_claim_meta(ext.claims))


def _constant_line(c: StringConstant, indent: int) -> Line:
    return Line(c, indent, constant_spans(c))


def _struct_def_line(sd: StructDef, indent: int) -> Line:
    return Line(sd, indent, struct_def_spans(sd))


def _import_spans(imp) -> tuple:
    spans = [Span(imp.module, "const_name")]
    if imp.wire:
        from quod.model import _format_type
        spans.append(Span(" wire ", "keyword"))
        bindings = ", ".join(f"{w.name}={_format_type(w.type)}" for w in imp.wire)
        spans.append(Span(bindings, "type"))
    return tuple(spans)


def format_program_lines(program: Program) -> Iterator[Line]:
    yield Line(program, 0, (Span("program", "keyword"), Span(" {", "punct")))
    if program.wirables:
        yield Line(None, 2, (Span("wirables:", "section"),))
        for w in program.wirables:
            label = f"{w.name}: {w.bound}" if w.bound else w.name
            yield Line(None, 4, (Span(label, "type"),))
    if program.imports:
        yield Line(None, 2, (Span("imports:", "section"),))
        for imp in program.imports:
            yield Line(None, 4, _import_spans(imp))
    if program.constants:
        yield Line(None, 2, (Span("constants:", "section"),))
        for c in program.constants:
            yield _constant_line(c, 4)
    if program.structs:
        yield Line(None, 2, (Span("structs:", "section"),))
        for sd in program.structs:
            yield _struct_def_line(sd, 4)
    if program.enums:
        yield Line(None, 2, (Span("enums:", "section"),))
        for ed in program.enums:
            yield from _enum_def_lines(ed, 4)
    if program.externs:
        yield Line(None, 2, (Span("externs:", "section"),))
        for ext in program.externs:
            yield _extern_line(ext, 4)
    if program.source_units:
        yield Line(None, 2, (Span("source_units:", "section"),))
        for unit in program.source_units:
            yield from _c_unit_lines(unit, 4)
    if program.structured_functions:
        yield Line(None, 2, (Span("structured_functions:", "section"),))
        for fn in program.structured_functions:
            yield from format_function_lines(fn, indent=4)
    if program.functions:
        yield Line(None, 2, (Span("functions:", "section"),))
        for fn in program.functions:
            yield from format_function_lines(fn, indent=4)
    if program.edges:
        yield Line(None, 2, (Span("edges:", "section"),))
        for e in program.edges:
            yield Line(e, 4, (
                Span(e.source, "local"),
                Span(" -> ", "op"),
                Span(e.target, "local"),
            ))
    if program.equivalences:
        yield Line(None, 2, (Span("equivalences:", "section"),))
        for eq in program.equivalences:
            spans: list[Span] = [
                Span(eq.a_node_id, "local"),
                Span(" ~ ", "op"),
                Span(eq.b_node_id, "local"),
            ]
            meta = format_equivalence_metadata(eq)
            if meta:
                spans.append(Span(meta, "comment"))
            yield Line(eq, 4, tuple(spans))
    if (
        not program.constants and not program.functions
        and not program.externs and not program.structs
        and not program.enums and not program.imports
        and not program.source_units and not program.structured_functions
        and not program.edges and not program.equivalences
    ):
        yield Line(None, 2, (Span("(empty)", "comment"),))
    yield Line(None, 0, (Span("}", "punct"),))


def _c_unit_lines(unit: CUnit, indent: int) -> Iterator[Line]:
    """Render a layer-A `CUnit` as a section with the C source text
    inside. Layer A is preserved C — coloring it with quod's vocabulary
    would lie about what it is — so we drop into plain-text spans."""
    yield Line(unit, indent, (
        Span("c_unit ", "keyword"),
        Span(repr(unit.source_path), "string"),
        Span(" {", "punct"),
    ))
    for fn in unit.functions:
        for raw_line in format_c_fn(fn).splitlines():
            yield Line(None, indent + 2, (Span(raw_line, "comment"),))
    yield Line(None, indent, (Span("}", "punct"),))


def _enum_def_lines(ed: EnumDef, indent: int) -> Iterator[Line]:
    yield Line(ed, indent, (
        Span("enum", "keyword"), Span(" ", "ws"),
        Span(ed.name, "type"), Span(" {", "punct"),
    ))
    for v in ed.variants:
        spans: list[Span] = [Span(v.name, "fn_name")]
        if v.fields:
            spans.append(Span("(", "punct"))
            for i, f in enumerate(v.fields):
                if i > 0:
                    spans.append(Span(", ", "punct"))
                spans.extend((
                    Span(f.name, "param"), Span(": ", "punct"),
                    type_span(f.type),
                ))
            spans.append(Span(")", "punct"))
        yield Line(v, indent + 2, tuple(spans))
    yield Line(None, indent, (Span("}", "punct"),))


# ---------- Renderer ----------

def render(
    lines: Iterable[Line],
    *,
    theme: Theme = plain_theme,
    mode: Literal["columnar", "inline"] = "columnar",
) -> str:
    """Lay out a stream of Lines into a single string.

    columnar: hash gutter `[hash]  ` on the left, code column padded to the
        widest line, meta spans appended in a right column.
    inline:   hash prefix `[hash] ` inserted before each owner line's spans
        (after the indent); meta spans appended after the code on the same
        line. Reproduces the canonical `quod show` layout.
    """
    materialized = list(lines)
    if mode == "inline":
        return _render_inline(materialized, theme)
    return _render_columnar(materialized, theme)


def _render_columnar(lines: list[Line], theme: Theme) -> str:
    code_widths = [line.indent + _visual_width(line.spans) for line in lines]
    max_code_width = max(code_widths, default=0)
    out: list[str] = []
    for line, code_w in zip(lines, code_widths):
        gutter = paint(_gutter_spans(line.owner), theme)
        code = (" " * line.indent) + paint(line.spans, theme)
        if line.meta:
            pad = " " * max(2, max_code_width - code_w + 2)
            meta = pad + paint(line.meta, theme)
            out.append(f"{gutter}{code}{meta}")
        else:
            out.append(f"{gutter}{code}")
    return "\n".join(out)


def _render_inline(lines: list[Line], theme: Theme) -> str:
    out: list[str] = []
    for line in lines:
        prefix = ""
        if line.owner is not None:
            prefix = paint(hash_brackets(line.owner), theme) + " "
        code = (" " * line.indent) + prefix + paint(line.spans, theme)
        if line.meta:
            code += "  " + paint(line.meta, theme)
        out.append(code)
    return "\n".join(out)
