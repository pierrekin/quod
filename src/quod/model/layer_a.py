"""Layer A — C source-language nodes.

Inert: no validation, no codegen, no semantic checks. Their job is to
preserve the original C as a subtree of the program graph so
provenance edges to lifted quod nodes have something to point at.
The supported subset matches the existing C ingester (int-only, no
structs/floats), narrowed further to what the corpus exercises.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field

from quod.model.base import _Node, _mint_node_id


class CNamedType(_Node):
    """A named scalar C type (`int`, `char`, etc.) — anything not a
    composite (pointer, array, struct). Currently only `int` and
    `char` are supported; the lift-checker decides which `CNamedType`
    names map cleanly onto layer-B types and refuses the rest.

    JSON kind stays `c.type` for backward compatibility with the
    existing layer-A corpus; the Python class was renamed when
    `CPointerType` joined to make `CType` a real union.
    """
    kind: Literal["c.type"] = "c.type"
    name: str


class CPointerType(_Node):
    """A pointer-to-T C type: `int*`, `char*`, `CURL*`, …. The pointee
    can be any `CType` (named scalar or another pointer for `int**`),
    so `int **p` round-trips as
    `CPointerType(CPointerType(CNamedType("int")))`.

    At layer B all pointers collapse to `I8PtrType` (LLVM's opaque-
    pointer convention); the lift-checker treats any `CPointerType`
    as corresponding to `I8PtrType`. The pointee name is informational
    — useful for human-readable rendering and provenance, no semantic
    weight in the equivalence claim.
    """
    kind: Literal["c.type.ptr"] = "c.type.ptr"
    pointee: "CType"


# Layer-A C type — a named scalar or a pointer-to-CType. Used wherever
# a type annotation appears at layer A (CParam.type, CVarDecl.type,
# CFn.return_type, CPointerType.pointee).
CType = Annotated[
    Union[CNamedType, CPointerType],
    Field(discriminator="kind"),
]


class CIntLit(_Node):
    kind: Literal["c.lit_int"] = "c.lit_int"
    value: int


class CVarRef(_Node):
    """A C identifier reference — to a parameter, local, or any in-scope
    variable. Layer A doesn't distinguish these; the lifter does."""
    kind: Literal["c.var_ref"] = "c.var_ref"
    name: str


class CEnumConstRef(_Node):
    """A C enum-constant reference — `CURLOPT_URL`, `EAGAIN`, `O_RDONLY`,
    etc. The layer-B lifter resolves these via libclang to integer
    values (`CURLOPT_URL` → `IntLit(10002)`); layer A preserves the
    source-level identifier *and* records the resolved value so the
    lift-check can verify equivalence without re-running libclang.

    Both fields are load-bearing in different ways:
      - `name` is the source-level spelling (provenance + readability).
      - `value` is what the lift-check actually compares against the
        layer-B `IntLit.value`.

    If the enum's resolved value drifts (e.g. you ingested against
    libcurl 7.x and rebuild against 8.x with a re-numbered enum), the
    pinned `value` here disagrees with the new layer B's `IntLit` and
    `equiv verify` flags it. Catching that drift is half the point.
    """
    kind: Literal["c.enum_const_ref"] = "c.enum_const_ref"
    id: str = Field(default_factory=lambda: _mint_node_id("cenumconst"))
    name: str
    value: int


class CBinOp(_Node):
    """A binary operator in C source — arithmetic, comparison, bitwise,
    or logical. `op` is the operator's source-form spelling (`+`, `<`,
    `&&`, etc.). Layer A doesn't enumerate; the lifter is responsible
    for refusing operators outside the supported subset.

    Has its own ID — for-loop conditions and other named expression
    positions are edge endpoints.
    """
    kind: Literal["c.binop"] = "c.binop"
    id: str = Field(default_factory=lambda: _mint_node_id("cbinop"))
    op: str
    lhs: "CExpr"
    rhs: "CExpr"


class CStringLit(_Node):
    """A C string literal — `"hello, world"` etc. The value is the
    decoded payload (escapes resolved). The layer-B lifter interns
    these into `StringConstant`s and references them via `StringRef`;
    layer A preserves the literal value before interning so the
    original source spelling is recoverable."""
    kind: Literal["c.lit_str"] = "c.lit_str"
    id: str = Field(default_factory=lambda: _mint_node_id("clitstr"))
    value: str


class CCall(_Node):
    """A C function call expression — `printf("...", x)`,
    `square(a)`, etc. `callee` is the called function's spelling; only
    direct (non-indirect) calls are supported."""
    kind: Literal["c.call"] = "c.call"
    id: str = Field(default_factory=lambda: _mint_node_id("ccall"))
    callee: str
    args: tuple["CExpr", ...] = ()


class CArraySubscript(_Node):
    """`base[index]` — array subscript. Only emitted inside a
    `CAddressOf` (the lifter recognizes `&p[k]` as pointer arithmetic
    and produces a `PtrOffset` at layer B). Bare `arr[k]` reads —
    e.g. for an `int arr[]` value — aren't yet supported by the
    layer-A or layer-B translators.
    """
    kind: Literal["c.array_subscript"] = "c.array_subscript"
    id: str = Field(default_factory=lambda: _mint_node_id("carrsub"))
    base: "CExpr"
    index: "CExpr"


class CAddressOf(_Node):
    """`&expr` — address-of. Only emitted with a
    `CArraySubscript` target (`&p[k]` ≡ `p + k` for char-pointer
    arithmetic). Other `&` forms (`&local`, `&struct.field`, …) are
    refused at ingest time."""
    kind: Literal["c.addr_of"] = "c.addr_of"
    id: str = Field(default_factory=lambda: _mint_node_id("caddrof"))
    target: "CExpr"


class CBreak(_Node):
    """`break;` — exit the innermost enclosing loop. Layer A preserves
    the source statement; the lift produces a layer-B `Break` (core)."""
    kind: Literal["c.break"] = "c.break"
    id: str = Field(default_factory=lambda: _mint_node_id("cbreak"))


class CContinue(_Node):
    """`continue;` — skip to the next iteration of the innermost
    enclosing loop. Layer A preserves the source statement; the lift
    produces a layer-B `Continue` (core). Inside a c.for_general,
    the c-family lowering pre-rewrites `continue` to `inc; continue`
    so the inc step is preserved per C semantics."""
    kind: Literal["c.continue"] = "c.continue"
    id: str = Field(default_factory=lambda: _mint_node_id("ccontinue"))


class CTernary(_Node):
    """`cond ? then_value : else_value` — the C ternary operator.

    Layer A preserves the source-form ternary; the lift maps each
    layer-A CTernary to a layer-B `IfExpr` with the same three sub-
    expressions. The lift-checker pairs the two 1:1.
    """
    kind: Literal["c.ternary"] = "c.ternary"
    id: str = Field(default_factory=lambda: _mint_node_id("cternary"))
    cond: "CExpr"
    then_value: "CExpr"
    else_value: "CExpr"


class CUnary(_Node):
    """Unary prefix operator on an expression: `-x`, `!x`, `~x`.

    Layer A preserves the source operator faithfully; the lift to
    layer B desugars each via the standard identity:

      -x  ↔  BinOp("sub", IntLit(0), x')      (zero-minus form)
      !x  ↔  BinOp("eq",  x',         IntLit(0))   (i1-typed)
      ~x  ↔  BinOp("xor", x',         IntLit(-1))  (one's-complement)

    The lift-checker pairs CUnary with the corresponding layer-B
    BinOp shape.
    """
    kind: Literal["c.unary"] = "c.unary"
    id: str = Field(default_factory=lambda: _mint_node_id("cunary"))
    op: Literal["-", "!", "~"]
    value: "CExpr"


CExpr = Annotated[
    Union[CIntLit, CVarRef, CEnumConstRef, CBinOp, CStringLit, CCall,
          CArraySubscript, CAddressOf, CUnary, CTernary],
    Field(discriminator="kind"),
]


class CParam(_Node):
    kind: Literal["c.param"] = "c.param"
    name: str
    type: CType


class CVarDecl(_Node):
    """`int s = 0;` or `int i;` — a local variable declaration."""
    kind: Literal["c.var_decl"] = "c.var_decl"
    id: str = Field(default_factory=lambda: _mint_node_id("cvardecl"))
    type: CType
    name: str
    init: CExpr | None = None


class CAssign(_Node):
    """`s = s + i;` — assignment to an in-scope variable. The target is
    a name; assignments to fields, indexed locations, or pointer
    dereferences aren't yet modeled."""
    kind: Literal["c.assign"] = "c.assign"
    id: str = Field(default_factory=lambda: _mint_node_id("cassign"))
    target: str
    value: CExpr


class CReturn(_Node):
    """`return s;` or `return;`."""
    kind: Literal["c.return"] = "c.return"
    id: str = Field(default_factory=lambda: _mint_node_id("creturn"))
    value: CExpr | None = None


class CFor(_Node):
    """`for (init; cond; inc) { body }` — the C for loop verbatim. Each
    of init/cond/inc is independently optional (matching C's three-empty-
    parts shape); body is a list of layer-A statements.
    """
    kind: Literal["c.for"] = "c.for"
    id: str = Field(default_factory=lambda: _mint_node_id("cfor"))
    init: "CForInit | None" = None
    cond: CExpr | None = None
    inc: "CForInit | None" = None
    body: tuple["CStmt", ...] = ()


class CIf(_Node):
    """`if (cond) { then } else { else }` — both bodies as flat lists
    of layer-A statements. Empty `else_body` means the if had no else
    clause."""
    kind: Literal["c.if"] = "c.if"
    id: str = Field(default_factory=lambda: _mint_node_id("cif"))
    cond: CExpr
    then_body: tuple["CStmt", ...] = ()
    else_body: tuple["CStmt", ...] = ()


class CWhile(_Node):
    """`while (cond) { body }` — pre-test loop."""
    kind: Literal["c.while"] = "c.while"
    id: str = Field(default_factory=lambda: _mint_node_id("cwhile"))
    cond: CExpr
    body: tuple["CStmt", ...] = ()


class CSwitchCase(_Node):
    """One arm of a `CSwitch`. Stacked-empty-case labels share one
    body: `case 1: case 2: stmt; break;` is one CSwitchCase with
    `values=(1, 2)`. Each case's body must end with `break`, `return`,
    or `unreachable` — fall-through to the next case (other than via
    shared-empty-case stacking) is refused at ingest time per the
    deferred design question on UB-handling.
    """
    kind: Literal["c.switch_case"] = "c.switch_case"
    id: str = Field(default_factory=lambda: _mint_node_id("cswitchcase"))
    values: tuple["CExpr", ...]
    body: tuple["CStmt", ...] = ()


class CSwitch(_Node):
    """`switch (scrutinee) { case ...: ...; default: ...; }` —
    multiway dispatch on an integer value. Layer A preserves the
    source structure; the lift produces an if-else-if chain at
    layer B (no Switch in core; a tag-on-int dispatch can always
    be re-expressed as comparisons).
    """
    kind: Literal["c.switch"] = "c.switch"
    id: str = Field(default_factory=lambda: _mint_node_id("cswitch"))
    scrutinee: "CExpr"
    cases: tuple[CSwitchCase, ...] = ()
    default: tuple["CStmt", ...] | None = None  # None = no `default:` clause


class CDoWhile(_Node):
    """`do { body } while (cond);` — post-test loop. Layer A preserves
    the source statement; the lift produces a layer-B `DoWhile` (core).
    The body always executes at least once."""
    kind: Literal["c.do_while"] = "c.do_while"
    id: str = Field(default_factory=lambda: _mint_node_id("cdowhile"))
    body: tuple["CStmt", ...] = ()
    cond: CExpr


class CExprStmt(_Node):
    """An expression evaluated for its side effect — typically a call
    like `printf(...)`. Only `CExprStmt(CCall(...))` is emitted; bare
    expression statements (e.g. `x;`) are refused at ingest time."""
    kind: Literal["c.expr_stmt"] = "c.expr_stmt"
    id: str = Field(default_factory=lambda: _mint_node_id("cexprstmt"))
    value: CExpr


# CForInit is the union of statements that may appear in a C for-loop's
# init or inc slot — a declaration or an assignment. Distinct from CStmt
# because for-init permits a declaration even outside a block scope; the
# lifter folds the loop's scope into the layer-B `CStyleFor` envelope.
CForInit = Annotated[
    Union[CVarDecl, CAssign, "CIncrementStmt"],
    Field(discriminator="kind"),
]


class CCompoundAssign(_Node):
    """`x += y`, `x -= y`, `x &= y`, etc. — assignment combined with a
    binary operator. Layer A preserves the source operator; the lift
    desugars to `Assign(x, BinOp(op_translated, LocalRef(x), y'))` on
    the layer-B side. The lift-checker pairs the source-form operator
    with the corresponding layer-B BinOp.

    Only locals declared with `Let` can be the target — assignment to
    parameters is refused at ingest, matching plain `Assign`.
    """
    kind: Literal["c.compound_assign"] = "c.compound_assign"
    id: str = Field(default_factory=lambda: _mint_node_id("ccompound"))
    target: str
    op: Literal["+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="]
    value: "CExpr"


class CIncrementStmt(_Node):
    """`i++;`, `++i;`, `i--;`, `--i;` — statement-position increment or
    decrement of a local. Layer A preserves both the operator and the
    pre/post position for source fidelity; for statement-position the
    expression value is discarded, so pre and post lift identically to
    `Assign(target, BinOp("add"|"sub", LocalRef(target), IntLit(1)))`
    on the layer-B side.

    Expression-position `++/--` (e.g. `int y = i++;`, `arr[i++]`) is
    refused at ingest — the sequencing semantics need a separate
    design pass and aren't yet supported.

    Only locals declared with `Let` can be the target — increment of
    a parameter is refused at ingest, matching plain `Assign` and
    `CCompoundAssign`.
    """
    kind: Literal["c.increment_stmt"] = "c.increment_stmt"
    id: str = Field(default_factory=lambda: _mint_node_id("cincstmt"))
    target: str
    op: Literal["++", "--"]
    position: Literal["pre", "post"]


class CMultiVarDecl(_Node):
    """`int a, b, c;` or `int a = 1, b = 2;` — a single declaration
    statement that introduces multiple locals.

    Layer A preserves the source-form grouping; the lift expands this
    to N consecutive `Let` statements on the layer-B side. The lift-
    checker recognizes the 1:N pairing and walks each sub-decl against
    its corresponding Let. All `decls` share the same C type (the type
    appears once in source, before the comma-separated declarator
    list).
    """
    kind: Literal["c.multi_var_decl"] = "c.multi_var_decl"
    id: str = Field(default_factory=lambda: _mint_node_id("cmultivardecl"))
    decls: tuple[CVarDecl, ...]


CStmt = Annotated[
    Union[CVarDecl, CMultiVarDecl, CAssign, CCompoundAssign, CIncrementStmt,
          CReturn, CFor, CIf, CWhile, CDoWhile, CExprStmt, CBreak, CContinue,
          CSwitch],
    Field(discriminator="kind"),
]


class CFn(_Node):
    """A C function definition: `int sum(int n) { ... }`."""
    kind: Literal["c.fn"] = "c.fn"
    id: str = Field(default_factory=lambda: _mint_node_id("cfn"))
    name: str
    return_type: CType
    params: tuple[CParam, ...] = ()
    body: tuple[CStmt, ...] = ()


class CUnit(_Node):
    """A C translation unit — one source file's contents preserved as
    layer-A nodes. `source_path` is recorded so the graph can be paired
    back with the original file (and re-ingested if the source changes).
    """
    kind: Literal["c_unit"] = "c_unit"
    id: str = Field(default_factory=lambda: _mint_node_id("cunit"))
    source_path: str
    functions: tuple[CFn, ...] = ()
