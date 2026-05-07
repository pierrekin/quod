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

from quod.model import Function

from quod.script.parser import Parser
from quod.script.resolver import _Resolver, _Scope
from quod.script.tokens import ScriptError, tokenize


__all__ = ["tokenize", "parse_function", "parse_predicate", "ScriptError"]


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
