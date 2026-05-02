"""CPG data model + pretty-printer + immutable update helpers.

The graph is the asset. Nodes are frozen Pydantic models; mutators return
new Programs via model_copy. Addressing is by name *or* content-hash prefix
(the latter implemented in quod.hashing / quod.editor).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator


# ---------- Base ----------

class _Node(BaseModel):
    # strict=True: no silent coercion. frozen=True: graph is read-only;
    # mutators must build new instances via model_copy.
    model_config = ConfigDict(strict=True, frozen=True)


# ---------- Constants ----------

class StringConstant(_Node):
    name: str
    value: str


# ---------- Expressions ----------

class IntLit(_Node):
    kind: Literal["llvm.const_int"] = "llvm.const_int"
    type: "IntType"
    value: int


class ParamRef(_Node):
    kind: Literal["llvm.param_ref"] = "llvm.param_ref"
    name: str


class BinOp(_Node):
    """Binary operation. The operator determines the result type:

      arith (s) — add, sub, mul, sdiv, srem        : iN in / iN out
      arith (u) — udiv                              : iN in / iN out
      cmp (s)   — slt, sle, sgt, sge, eq, ne       : iN in / i1 out
      cmp (u)   — ult, ule, ugt, uge               : iN in / i1 out
      bitwise   — or, and                           : iN in / iN out

    Operands of arith/bitwise/cmp must have the same type; LLVM's verifier
    enforces this at lower time. The signed/unsigned distinction matches
    LLVM IR predicates — signedness lives on the op, not the type. Division
    by zero is undefined behaviour (matches LLVM); guard with an int_range
    or runtime branch if the divisor isn't statically nonzero.

    For short-circuit boolean combinators (correct in the presence of
    side-effecting operands), use `ShortCircuitOr` / `ShortCircuitAnd` —
    those are i1-only and synthesise branches.
    """
    kind: Literal["llvm.binop"] = "llvm.binop"
    op: Literal[
        "add", "sub", "mul", "sdiv", "udiv", "srem",
        "slt", "sle", "sgt", "sge", "eq", "ne",
        "ult", "ule", "ugt", "uge",
        "or", "and",
    ]
    lhs: "Expr"
    rhs: "Expr"


class ShortCircuitOr(_Node):
    """`lhs || rhs` with C-style short-circuit. If `lhs` is true, `rhs` is
    not evaluated. Lowered to branch + phi."""
    kind: Literal["quod.sc_or"] = "quod.sc_or"
    lhs: "Expr"
    rhs: "Expr"


class ShortCircuitAnd(_Node):
    """`lhs && rhs` with C-style short-circuit. If `lhs` is false, `rhs` is
    not evaluated. Lowered to branch + phi."""
    kind: Literal["quod.sc_and"] = "quod.sc_and"
    lhs: "Expr"
    rhs: "Expr"


class Call(_Node):
    """Call a user function or an extern in the same Program.

    User function calls are i32-in/i32-out. Extern calls follow the extern's
    declared `param_types` / `return_type` — pass `StringRef` for i8*-typed
    args, IntLit/ParamRef/etc. for i32 args. For varargs externs (printf etc.),
    pass any number of args beyond the fixed prefix.
    """
    kind: Literal["llvm.call"] = "llvm.call"
    function: str
    args: tuple["Expr", ...] = ()


class StringRef(_Node):
    """An i8* value: pointer to a `StringConstant`'s underlying bytes. Used
    as an arg to externs that take `const char *` (e.g. system, getenv,
    puts, printf's format)."""
    kind: Literal["quod.string_ref"] = "quod.string_ref"
    name: str  # name of a StringConstant in the Program


class LocalRef(_Node):
    """Read the current value of a local introduced by `Let` (or a `For`
    loop variable)."""
    kind: Literal["quod.local_ref"] = "quod.local_ref"
    name: str


class FieldRead(_Node):
    """Read a named field from a struct-typed expression. The inner `value`
    must produce a value of some `StructType("X")`, and `name` must be a
    field of that struct's def."""
    kind: Literal["quod.field"] = "quod.field"
    value: "Expr"
    name: str


class FieldInit(_Node):
    """One field's value in a `StructInit`."""
    name: str
    value: "Expr"


class StructInit(_Node):
    """Construct a struct value. v1: every field of the named def must be
    initialized exactly once, in any order. Lowered to an `insertvalue`
    chain on `undef`."""
    kind: Literal["quod.struct_init"] = "quod.struct_init"
    type: str
    fields: tuple[FieldInit, ...]


class PtrOffset(_Node):
    """Pointer arithmetic on an i8* base. The result is `base + offset` as
    another i8*, computed via a byte-stride GEP — no scaling for element
    size, since the underlying type is always i8.

    The result aliases the same allocation as `base`. Lifetime is the
    caller's responsibility (a slice into an arena is valid until the arena
    drops, and not after). Out-of-bounds offsets are undefined behaviour,
    matching the LLVM `getelementptr inbounds` story; if you need a check,
    add an int_range claim on the offset.
    """
    kind: Literal["quod.ptr_offset"] = "quod.ptr_offset"
    base: "Expr"     # must lower to i8*
    offset: "Expr"   # must lower to i64


class Widen(_Node):
    """Cast an integer value between widths.

    Lowers to `sext` (default) / `zext` / `trunc` depending on the
    relationship between the source type's width and `target`'s. When
    `signed=True` (default) and the target is wider, the high bits get
    sign-extended; when `signed=False`, zero-extended. Truncation is
    width-only and ignores `signed`. A no-op cast (same width) returns
    the value unchanged.

    Quod's convention is signed integers, so `signed=True` matches the
    common case (`int x; (int64_t)x;` → sign-extend). Reach for
    `signed=False` only when the source value is genuinely unsigned in
    intent (e.g., a byte read from a buffer being widened to i64 for
    arithmetic).
    """
    kind: Literal["quod.widen"] = "quod.widen"
    value: "Expr"
    target: "IntType"
    signed: bool = True


class Load(_Node):
    """Read a value of `type` from an i8* base pointer.

    Lowered as `bitcast(ptr, type*)` + `load type, type* %`. The base must
    lower to i8*; the value type can be any int width or a named struct.
    Alignment isn't expressed at the model level — LLVM picks the natural
    alignment for the type. If the underlying memory isn't actually so
    aligned, that's the caller's problem (and undefined behaviour).

    Pair with `quod.ptr_offset` to read from a non-zero offset of a buffer:
    `load(ptr_offset(buf, k), i8)`.
    """
    kind: Literal["quod.load"] = "quod.load"
    ptr: "Expr"   # must lower to i8*
    type: "Type"  # the value type to return


Expr = Annotated[
    Union[
        IntLit, ParamRef, LocalRef, BinOp, ShortCircuitOr, ShortCircuitAnd,
        Call, StringRef, FieldRead, StructInit, PtrOffset, Widen, Load,
    ],
    Field(discriminator="kind"),
]


# ---------- Types ----------
#
# Width-per-class follows LLVM's "type carries no signedness" convention —
# signedness lives on the operation (e.g. BinOp.slt vs ult). i1 is a
# first-class type used for boolean values (cmp results, short-circuits,
# explicit booleans).

class I1Type(_Node):
    kind: Literal["llvm.i1"] = "llvm.i1"


class I8Type(_Node):
    kind: Literal["llvm.i8"] = "llvm.i8"


class I16Type(_Node):
    kind: Literal["llvm.i16"] = "llvm.i16"


class I32Type(_Node):
    kind: Literal["llvm.i32"] = "llvm.i32"


class I64Type(_Node):
    kind: Literal["llvm.i64"] = "llvm.i64"


class I8PtrType(_Node):
    kind: Literal["llvm.i8_ptr"] = "llvm.i8_ptr"


class StructType(_Node):
    """Reference to a named StructDef. Pass-by-value at the LLVM level.

    The `name` must match a `StructDef.name` in the same Program; the
    Program-level validator catches dangling refs at load time.
    """
    kind: Literal["llvm.struct"] = "llvm.struct"
    name: str


# Integer-only sub-union: usable wherever a pointer would be nonsense
# (IntLit, function params/return, For loop var, claim-bearing locals).
IntType = Annotated[
    Union[I1Type, I8Type, I16Type, I32Type, I64Type],
    Field(discriminator="kind"),
]

# Full type union, including pointer and struct types — used for ExternFunction
# signatures, function params/return, Let bindings, and struct fields.
Type = Annotated[
    Union[I1Type, I8Type, I16Type, I32Type, I64Type, I8PtrType, StructType],
    Field(discriminator="kind"),
]


def int_type_width(t: "IntType") -> int:
    """Bit width of an int type."""
    match t:
        case I1Type():
            return 1
        case I8Type():
            return 8
        case I16Type():
            return 16
        case I32Type():
            return 32
        case I64Type():
            return 64
    raise ValueError(f"not an int type: {t!r}")


# ---------- Statements ----------

class ReturnInt(_Node):
    """Return a constant integer. Shorthand kept for hello-world brevity."""
    kind: Literal["quod.return_int"] = "quod.return_int"
    value: int


class ReturnExpr(_Node):
    kind: Literal["quod.return_expr"] = "quod.return_expr"
    value: Expr


class If(_Node):
    """Two-branch conditional. Branches may both terminate (return), or both
    fall through to the next statement, or mix — a merge block is created
    on demand by the lowering pass."""
    kind: Literal["quod.if"] = "quod.if"
    cond: Expr  # must lower to i1
    then_body: tuple["Statement", ...]
    else_body: tuple["Statement", ...]


class Let(_Node):
    """Introduce a mutable local variable. `name` must not shadow a parameter
    or another local in the same function. Lowered to alloca-at-entry + store."""
    kind: Literal["quod.let"] = "quod.let"
    name: str
    type: Type
    init: Expr


class Assign(_Node):
    """Mutate an existing local. `name` must reference a local previously
    introduced by `Let` (or a `For` loop variable in scope)."""
    kind: Literal["quod.assign"] = "quod.assign"
    name: str
    value: Expr


class While(_Node):
    """Pre-test loop. Evaluates `cond` each iteration; runs `body` if true."""
    kind: Literal["quod.while"] = "quod.while"
    cond: Expr  # must lower to i1
    body: tuple["Statement", ...]


class For(_Node):
    """Bounded iteration: `var` runs from `lo` (inclusive) to `hi` (exclusive),
    incrementing by 1 each iteration. `lo` and `hi` are evaluated once before
    the loop (snapshot semantics, not C-style re-evaluation). `var` is a local
    of type `type`, scoped to `body` only. `lo` and `hi` must produce values
    of the same type as `var`."""
    kind: Literal["quod.for"] = "quod.for"
    var: str
    type: IntType
    lo: Expr
    hi: Expr
    body: tuple["Statement", ...]


class ExprStmt(_Node):
    """Evaluate an expression for its side effects, discard the result.
    The natural shape for `printf(...)` and other void-effect calls."""
    kind: Literal["quod.expr_stmt"] = "quod.expr_stmt"
    value: Expr


class FieldSet(_Node):
    """Mutate one field of a struct-typed local. Same scoping as `Assign`:
    `local` must reference a Let-introduced local of some `StructType("X")`,
    and `name` must be a field of that struct's def."""
    kind: Literal["quod.field_set"] = "quod.field_set"
    local: str
    name: str
    value: Expr


class WithArena(_Node):
    """Bracket a body with an arena that's freed automatically.

    Lowering is a desugar: at block entry the runtime's `quod_arena_new` is
    called with `capacity` (i64) and the result (i8*) bound to a local named
    `name` for the duration of `body`. On every exit edge — fall-through and
    every `return` reachable from `body` — `quod_arena_drop` is called on
    that handle.

    The desugaring auto-declares the `quod_arena_new` / `quod_arena_drop`
    externs if the program doesn't already have them, so a `with_arena`
    block is one-stop sugar; users only need to declare `quod_arena_alloc`
    (and friends) explicitly when they call them inside the body.
    """
    kind: Literal["quod.with_arena"] = "quod.with_arena"
    name: str
    capacity: Expr   # must lower to i64
    body: tuple["Statement", ...]


Statement = Annotated[
    Union[ReturnInt, ReturnExpr, If, Let, Assign, While, For, ExprStmt, FieldSet, WithArena],
    Field(discriminator="kind"),
]


# ---------- Justifications ----------

# Polymorphic evidence channel attached to a claim. The kind discriminator
# tells you what flavor of evidence is on offer; the regime field on the
# claim is a coarse epistemic label (loosely correlated, not enforced).
#
#   z3        — external proof in SMT-LIB; verifiable by re-running Z3
#               or, in MVP2, just by re-hashing the artifact
#   manual    — a human signed off; no machine-checkable evidence
#   derived   — produced by an analysis pass; reproducible from `inputs`
#               (content-hash refs to the graph nodes the analysis read)


class Z3Justification(_Node):
    kind: Literal["z3"] = "z3"
    artifact_path: str
    artifact_hash: str
    note: str | None = None


class ManualJustification(_Node):
    kind: Literal["manual"] = "manual"
    signed_by: str
    rationale: str


class DerivedJustification(_Node):
    kind: Literal["derived"] = "derived"
    analysis: str                       # name of the analysis pass
    inputs: tuple[str, ...] = ()        # content hashes of nodes the pass read
    note: str | None = None


Justification = Annotated[
    Union[Z3Justification, ManualJustification, DerivedJustification],
    Field(discriminator="kind"),
]


# ---------- Claims ----------

# Epistemic source of a claim (who/what is making the assertion):
#   axiom   = the programmer asserts it (typically: no justification, or manual)
#   witness = a proof was produced out-of-band (typically: z3/coq/lean/...)
#   lattice = derived by an analysis pass (typically: derived)
Regime = Literal["axiom", "witness", "lattice"]

# Enforcement: do we trust the source named by `regime`, or verify at runtime?
#   trust  = lowered to llvm.assume; falsity is undefined behaviour
#   verify = lowered to a runtime branch + abort; falsity aborts the program
Enforcement = Literal["trust", "verify"]


class _Claim(_Node):
    """Common metadata carried by every claim.

    Defaults: a programmer assertion (regime=axiom), trusted unconditionally
    (enforcement=trust), without a justification (justification=None).
    """
    regime: Regime = "axiom"
    enforcement: Enforcement = "trust"
    justification: Justification | None = None

    # Drop metadata fields from serialized JSON when they're at default. This
    # keeps program.json compact for the common case while preserving the
    # discriminator `kind` (which is also default-valued but must round-trip).
    @model_serializer(mode="wrap")
    def _drop_default_metadata(self, handler, info):
        data = handler(self)
        if self.regime == "axiom":
            data.pop("regime", None)
        if self.enforcement == "trust":
            data.pop("enforcement", None)
        if self.justification is None:
            data.pop("justification", None)
        return data


class NonNegativeClaim(_Claim):
    """Asserts param >= 0. Subsumed by IntRangeClaim(min=0); kept as a convenience."""
    kind: Literal["non_negative"] = "non_negative"
    param: str


class IntRangeClaim(_Claim):
    """Asserts `min <= param <= max` (either bound optional).

    Lowered to one or two predicates (llvm.assume or runtime branch, per enforcement).
    min=None / max=None means unbounded on that side.
    """
    kind: Literal["int_range"] = "int_range"
    param: str
    min: int | None = None
    max: int | None = None


class ReturnInRangeClaim(_Claim):
    """Asserts the function's return value is in [min, max] (either bound optional).

    Function-scoped, not param-scoped — there's no `param` field. Today this
    is metadata only: the claim is provable via Z3 (quod claim prove) and
    verifiable (quod claim verify) but not yet exploited by the LLVM lowering
    pass.
    """
    kind: Literal["return_in_range"] = "return_in_range"
    min: int | None = None
    max: int | None = None


Claim = Annotated[
    Union[NonNegativeClaim, IntRangeClaim, ReturnInRangeClaim],
    Field(discriminator="kind"),
]


CLAIM_KINDS: tuple[str, ...] = ("non_negative", "int_range", "return_in_range")
PARAM_CLAIM_KINDS: tuple[str, ...] = ("non_negative", "int_range")
RETURN_CLAIM_KINDS: tuple[str, ...] = ("return_in_range",)


def claim_param(claim: Claim) -> str | None:
    """The parameter a claim targets, or None for function-scoped (return-value) claims."""
    match claim:
        case NonNegativeClaim(param=p) | IntRangeClaim(param=p):
            return p
        case ReturnInRangeClaim():
            return None
    raise ValueError(f"unhandled claim: {claim!r}")


def function_callees(fn: "Function") -> tuple[str, ...]:
    """Names of functions (user or extern) called from fn's body, deduplicated,
    first-seen order."""
    seen: dict[str, None] = {}

    def visit_expr(e) -> None:
        match e:
            case Call(function=name, args=args):
                seen.setdefault(name, None)
                for a in args:
                    visit_expr(a)
            case BinOp(lhs=l, rhs=r) | ShortCircuitOr(lhs=l, rhs=r) | ShortCircuitAnd(lhs=l, rhs=r):
                visit_expr(l)
                visit_expr(r)
            case FieldRead(value=inner):
                visit_expr(inner)
            case StructInit(fields=field_inits):
                for fi in field_inits:
                    visit_expr(fi.value)
            case PtrOffset(base=b, offset=o):
                visit_expr(b)
                visit_expr(o)
            case Widen(value=v):
                visit_expr(v)
            case Load(ptr=p):
                visit_expr(p)
            case _:
                pass

    def visit_stmt(s) -> None:
        match s:
            case ReturnExpr(value=expr) | ExprStmt(value=expr):
                visit_expr(expr)
            case If(cond=cond, then_body=t_body, else_body=e_body):
                visit_expr(cond)
                for x in t_body:
                    visit_stmt(x)
                for x in e_body:
                    visit_stmt(x)
            case Let(init=expr) | Assign(value=expr) | FieldSet(value=expr):
                visit_expr(expr)
            case While(cond=cond, body=body):
                visit_expr(cond)
                for x in body:
                    visit_stmt(x)
            case For(lo=lo, hi=hi, body=body):
                visit_expr(lo)
                visit_expr(hi)
                for x in body:
                    visit_stmt(x)
            case WithArena(capacity=cap, body=body):
                visit_expr(cap)
                for x in body:
                    visit_stmt(x)
            case _:
                pass

    for stmt in fn.body:
        visit_stmt(stmt)
    return tuple(seen)


# ---------- Top-level ----------

class StructField(_Node):
    """One field in a StructDef. Field types may be any `Type`, including
    other structs (no recursion: a struct can't directly contain itself
    by value)."""
    name: str
    type: Type


class StructDef(_Node):
    """A named record type. Fields are ordered and uniquely named.

    By-value semantics: lowered to an LLVM identified struct type, passed
    and returned as values, no implicit pointer indirection. Pointers to
    structs are out of v1 scope — opaque `i8*` if you need to hand one
    to an extern.
    """
    name: str
    fields: tuple[StructField, ...]

    def field(self, name: str) -> StructField | None:
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def field_index(self, name: str) -> int:
        for i, f in enumerate(self.fields):
            if f.name == name:
                return i
        raise KeyError(f"struct {self.name!r} has no field {name!r}")


class Param(_Node):
    """A typed function parameter. Any `Type` is accepted (int widths,
    `i8*`, named structs)."""
    name: str
    type: Type


class Function(_Node):
    name: str
    params: tuple[Param, ...] = ()
    return_type: Type
    body: tuple[Statement, ...]
    claims: tuple[Claim, ...] = ()
    notes: tuple[str, ...] = ()       # free-form developer/agent intent

    @model_serializer(mode="wrap")
    def _drop_empty_notes(self, handler, info):
        data = handler(self)
        if not self.notes:
            data.pop("notes", None)
        return data

    def param(self, name: str) -> Param | None:
        for p in self.params:
            if p.name == name:
                return p
        return None


class ExternFunction(_Node):
    """A libc-or-similar function declared but not defined by us.

    `arity` is a convenience for all-i32 signatures: when set, it expands
    to `param_types = (I32Type,) * arity` and `return_type = I32Type` at
    use time. For non-i32 sigs, set `param_types` and `return_type` directly
    and leave `arity` at 0. Set `varargs=True` for variadic libc functions
    like printf — callers may pass any number of args after the fixed prefix.
    """
    name: str
    arity: int = 0
    param_types: tuple[Type, ...] = ()
    return_type: Type = I32Type()
    varargs: bool = False

    @model_serializer(mode="wrap")
    def _drop_extern_defaults(self, handler, info):
        data = handler(self)
        # Drop arity when zero AND when unused (i.e., param_types non-empty).
        if self.arity == 0:
            data.pop("arity", None)
        if not self.param_types:
            data.pop("param_types", None)
        # Drop return_type when default i32.
        if isinstance(self.return_type, I32Type):
            data.pop("return_type", None)
        if not self.varargs:
            data.pop("varargs", None)
        return data

    def effective_param_types(self) -> tuple["Type", ...]:
        """Resolved param types: explicit `param_types` if given, otherwise
        `arity` copies of I32Type."""
        if self.param_types:
            return self.param_types
        return tuple(I32Type() for _ in range(self.arity))


class _ProgramBase(_Node):
    """Shared shape for Program and InputProgram."""
    constants: tuple[StringConstant, ...] = ()
    functions: tuple[Function, ...] = ()
    externs: tuple[ExternFunction, ...] = ()
    structs: tuple[StructDef, ...] = ()
    imports: tuple[str, ...] = ()

    @model_serializer(mode="wrap")
    def _drop_empty_collections(self, handler, info):
        data = handler(self)
        if not self.structs:
            data.pop("structs", None)
        if not self.imports:
            data.pop("imports", None)
        return data

    @field_validator("imports")
    @classmethod
    def _validate_import_names(cls, names: tuple[str, ...]) -> tuple[str, ...]:
        # Sanitize: only allow [A-Za-z0-9_.] so imports can't path-traverse
        # to disk locations outside the stdlib directory. Names map to file
        # paths via `stdlib/<name>.json` — no slashes, no leading/trailing
        # dots, no empty segments.
        seen: set[str] = set()
        for n in names:
            if not n or not all(c.isalnum() or c in "._" for c in n):
                raise ValueError(
                    f"invalid import name {n!r}: must match [A-Za-z0-9_.] only"
                )
            if n.startswith(".") or n.endswith(".") or ".." in n:
                raise ValueError(
                    f"invalid import name {n!r}: no leading/trailing/empty segments"
                )
            if n in seen:
                raise ValueError(f"duplicate import {n!r}")
            seen.add(n)
        return names


def _validate_structs(program: "_ProgramBase") -> None:
    """Program-wide struct sanity. Runs on both Program and InputProgram.

    - Struct names are unique.
    - No struct field references an undefined struct.
    - No struct contains itself by value (direct or transitive).
    - Every `StructType` mentioned in a Param/return/Let/Field/StructInit
      resolves to a defined struct.
    - Every `StructInit` covers exactly the fields of the named def, with
      no missing or extra names and no duplicates.

    When `program.imports` is non-empty, struct refs reachable from
    function bodies / params / externs are deferred — the imported module
    may bring the struct in, and we can't tell from this side. The fully
    resolved Program (with imports cleared) gets the complete check.
    """
    seen_names: set[str] = set()
    for sd in program.structs:
        if sd.name in seen_names:
            raise ValueError(f"duplicate struct definition {sd.name!r}")
        seen_names.add(sd.name)
        field_names: set[str] = set()
        for f in sd.fields:
            if f.name in field_names:
                raise ValueError(
                    f"struct {sd.name!r} has duplicate field {f.name!r}"
                )
            field_names.add(f.name)

    by_name: dict[str, StructDef] = {sd.name: sd for sd in program.structs}

    # Reject by-value cycles. Walk each struct's transitive struct-typed
    # fields; a path that revisits the start is a cycle.
    for sd in program.structs:
        _check_no_struct_cycle(sd.name, by_name)

    if program.imports:
        return

    # Validate every struct ref reachable from the program is defined.
    for sd in program.structs:
        for f in sd.fields:
            _check_type_refs(f.type, by_name, where=f"struct {sd.name!r} field {f.name!r}")
    for fn in program.functions:
        _check_type_refs(fn.return_type, by_name, where=f"function {fn.name!r} return type")
        for p in fn.params:
            _check_type_refs(p.type, by_name, where=f"function {fn.name!r} param {p.name!r}")
        for stmt in fn.body:
            _check_struct_uses_in_stmt(stmt, by_name, where=f"function {fn.name!r}")
    for ext in program.externs:
        _check_type_refs(ext.return_type, by_name, where=f"extern {ext.name!r} return type")
        for t in ext.param_types:
            _check_type_refs(t, by_name, where=f"extern {ext.name!r} param")


def _check_no_struct_cycle(start: str, by_name: dict[str, "StructDef"]) -> None:
    """DFS: refuse if `start` reaches itself through StructType fields."""
    visiting: set[str] = set()

    def go(name: str, path: tuple[str, ...]) -> None:
        if name == start and path:
            chain = " -> ".join(path + (name,))
            raise ValueError(
                f"struct {start!r} contains itself by value (cycle: {chain}); "
                f"v1 has no pointer-to-struct, so recursive structs are unrepresentable"
            )
        if name in visiting:
            return
        visiting.add(name)
        sd = by_name.get(name)
        if sd is None:
            return
        for f in sd.fields:
            if isinstance(f.type, StructType):
                go(f.type.name, path + (name,))
        visiting.discard(name)

    sd = by_name.get(start)
    if sd is None:
        return
    for f in sd.fields:
        if isinstance(f.type, StructType):
            go(f.type.name, (start,))


def _check_type_refs(t: "Type", by_name: dict[str, "StructDef"], *, where: str) -> None:
    if isinstance(t, StructType) and t.name not in by_name:
        raise ValueError(f"{where}: references undefined struct {t.name!r}")


def _check_struct_uses_in_stmt(stmt, by_name: dict[str, "StructDef"], *, where: str) -> None:
    match stmt:
        case ReturnExpr(value=expr) | ExprStmt(value=expr):
            _check_struct_uses_in_expr(expr, by_name, where=where)
        case If(cond=cond, then_body=t_body, else_body=e_body):
            _check_struct_uses_in_expr(cond, by_name, where=where)
            for s in t_body:
                _check_struct_uses_in_stmt(s, by_name, where=where)
            for s in e_body:
                _check_struct_uses_in_stmt(s, by_name, where=where)
        case Let(type=ty, init=expr):
            _check_type_refs(ty, by_name, where=f"{where}: let")
            _check_struct_uses_in_expr(expr, by_name, where=where)
        case Assign(value=expr) | FieldSet(value=expr):
            _check_struct_uses_in_expr(expr, by_name, where=where)
        case While(cond=cond, body=body):
            _check_struct_uses_in_expr(cond, by_name, where=where)
            for s in body:
                _check_struct_uses_in_stmt(s, by_name, where=where)
        case For(lo=lo, hi=hi, body=body):
            _check_struct_uses_in_expr(lo, by_name, where=where)
            _check_struct_uses_in_expr(hi, by_name, where=where)
            for s in body:
                _check_struct_uses_in_stmt(s, by_name, where=where)
        case WithArena(capacity=cap, body=body):
            _check_struct_uses_in_expr(cap, by_name, where=where)
            for s in body:
                _check_struct_uses_in_stmt(s, by_name, where=where)


def _check_struct_uses_in_expr(expr, by_name: dict[str, "StructDef"], *, where: str) -> None:
    match expr:
        case StructInit(type=name, fields=field_inits):
            sd = by_name.get(name)
            if sd is None:
                raise ValueError(f"{where}: struct_init references undefined struct {name!r}")
            init_names = [fi.name for fi in field_inits]
            seen: set[str] = set()
            for n in init_names:
                if n in seen:
                    raise ValueError(
                        f"{where}: struct_init for {name!r} sets field {n!r} twice"
                    )
                seen.add(n)
            def_names = {f.name for f in sd.fields}
            extra = seen - def_names
            if extra:
                raise ValueError(
                    f"{where}: struct_init for {name!r} sets unknown field(s): "
                    f"{sorted(extra)}"
                )
            missing = def_names - seen
            if missing:
                raise ValueError(
                    f"{where}: struct_init for {name!r} missing field(s): "
                    f"{sorted(missing)}"
                )
            for fi in field_inits:
                _check_struct_uses_in_expr(fi.value, by_name, where=where)
        case FieldRead(value=inner):
            _check_struct_uses_in_expr(inner, by_name, where=where)
        case BinOp(lhs=l, rhs=r) | ShortCircuitOr(lhs=l, rhs=r) | ShortCircuitAnd(lhs=l, rhs=r):
            _check_struct_uses_in_expr(l, by_name, where=where)
            _check_struct_uses_in_expr(r, by_name, where=where)
        case Call(args=args):
            for a in args:
                _check_struct_uses_in_expr(a, by_name, where=where)
        case PtrOffset(base=b, offset=o):
            _check_struct_uses_in_expr(b, by_name, where=where)
            _check_struct_uses_in_expr(o, by_name, where=where)
        case Widen(value=v):
            _check_struct_uses_in_expr(v, by_name, where=where)
        case Load(ptr=p, type=t):
            _check_struct_uses_in_expr(p, by_name, where=where)
            _check_type_refs(t, by_name, where=where)


class Program(_ProgramBase):
    """The fully-elaborated graph: stored claims + any derived (lattice) claims.

    Permissive: any regime is allowed in fn.claims. This is what `lower()`
    consumes and what editor mutators return.
    """

    @model_validator(mode="after")
    def _check_structs(self) -> "Program":
        _validate_structs(self)
        return self


class InputProgram(_ProgramBase):
    """The graph as authored. Only stored claims (axiom, witness) allowed.

    Used as the validation gate at the JSON I/O boundary: load parses through
    InputProgram (rejects lattice in stored), save round-trips through it
    before writing. Lattice claims live in memory only — they're derived by
    the analysis pass and lowered each build.
    """

    @field_validator("functions")
    @classmethod
    def _no_lattice_in_stored(cls, fns: tuple[Function, ...]) -> tuple[Function, ...]:
        for fn in fns:
            for c in fn.claims:
                if c.regime == "lattice":
                    raise ValueError(
                        f"lattice claims are derived, not stored; "
                        f"function {fn.name!r} has stored claim {c!r}"
                    )
        return fns

    @model_validator(mode="after")
    def _check_structs(self) -> "InputProgram":
        _validate_structs(self)
        return self


# ---------- File I/O ----------

def load_program(path: Path) -> Program:
    """Parse program.json. Validates as InputProgram (no lattice in stored)
    then returns the permissive Program type for in-memory editing."""
    raw = path.read_text()
    InputProgram.model_validate_json(raw)
    return Program.model_validate_json(raw)


def save_program(program: Program, path: Path) -> None:
    """Validate as InputProgram (raises if any lattice claims slipped into
    stored), then write JSON atomically.

    Atomic via write-tmp-then-rename: a concurrent reader sees either the old
    file or the new file, never a partially-written one. Mutations also need
    an external lock to prevent two writers from racing on load→save (last
    writer wins); see `_exclusive_lock` in cli.py.
    """
    InputProgram(constants=program.constants, functions=program.functions)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(program.model_dump_json(indent=2))
    tmp.replace(path)


# ---------- Lookups + immutable updates (by name) ----------

def find_function(program: Program, name: str) -> Function | None:
    for fn in program.functions:
        if fn.name == name:
            return fn
    return None


def require_function(program: Program, name: str) -> Function:
    fn = find_function(program, name)
    if fn is None:
        raise KeyError(f"no function named {name!r}")
    return fn


def replace_function(program: Program, new_fn: Function) -> Program:
    """Return a new Program with the same-named function replaced."""
    updated = tuple(new_fn if fn.name == new_fn.name else fn for fn in program.functions)
    if updated == program.functions:
        raise KeyError(f"no function named {new_fn.name!r}")
    return program.model_copy(update={"functions": updated})


def remove_function(program: Program, function_name: str) -> Program:
    """Return a new Program with the named function removed.

    Permissive about dangling calls — if other functions reference this one,
    the dangling-callee error surfaces at lower time (matches the
    callgraph.json example with `ghost`). Use `quod fn callers` first if you
    want to know who'd be affected.
    """
    kept = tuple(fn for fn in program.functions if fn.name != function_name)
    if kept == program.functions:
        raise KeyError(f"no function named {function_name!r}")
    return program.model_copy(update={"functions": kept})


def add_claim(program: Program, function: str, claim: Claim) -> Program:
    fn = require_function(program, function)
    target = claim_param(claim)
    if target is not None and fn.param(target) is None:
        raise KeyError(f"function {function!r} has no parameter {target!r}")
    for existing in fn.claims:
        if existing.kind == claim.kind and claim_param(existing) == target:
            scope = f"on {target!r}" if target is not None else "on return value"
            raise ValueError(
                f"{claim.kind} claim {scope} already present on {function}; "
                f"relax it first if you need to change bounds"
            )
    new_fn = fn.model_copy(update={"claims": fn.claims + (claim,)})
    return replace_function(program, new_fn)


def relax_claim(program: Program, function: str, kind: str, target: str | None) -> Program:
    """Remove the matching claim (no-op disallowed: must exist).

    target=None matches return-value claims (which have no parameter scope)."""
    fn = require_function(program, function)
    kept = tuple(
        c for c in fn.claims
        if not (c.kind == kind and claim_param(c) == target)
    )
    if len(kept) == len(fn.claims):
        scope = f"targeting {target!r}" if target is not None else "(return-value)"
        raise KeyError(f"no {kind} claim on {function} {scope}")
    new_fn = fn.model_copy(update={"claims": kept})
    return replace_function(program, new_fn)


# ---------- Pretty-printer ----------

# A "label" is an optional prefix the formatter inserts before each addressable
# node. The default is empty; the CLI passes a function returning [hashprefix]
# so each node prints with its addressable identity inline.
NodeLabel = Callable[[_Node], str]
_NO_LABEL: NodeLabel = lambda _node: ""


def format_program(program: Program, *, label: NodeLabel = _NO_LABEL) -> str:
    lines: list[str] = ["program {"]
    if program.imports:
        lines.append("  imports:")
        for name in program.imports:
            lines.append(f"    {name}")
    if program.constants:
        lines.append("  constants:")
        for c in program.constants:
            lines.append(f"    {label(c)}{c.name} = {c.value!r}")
    if program.structs:
        lines.append("  structs:")
        for sd in program.structs:
            lines.append(f"    {label(sd)}{format_struct_def(sd)}")
    if program.externs:
        lines.append("  externs:")
        for ext in program.externs:
            sig_parts = [_format_type(t) for t in ext.effective_param_types()]
            if ext.varargs:
                sig_parts.append("...")
            sig = ", ".join(sig_parts)
            ret = _format_type(ext.return_type)
            lines.append(f"    {label(ext)}extern {ext.name}({sig}) -> {ret}")
    if program.functions:
        lines.append("  functions:")
        for fn in program.functions:
            lines.extend("    " + line for line in format_function(fn, label=label).splitlines())
    if (
        not program.constants and not program.functions
        and not program.externs and not program.structs and not program.imports
    ):
        lines.append("  (empty)")
    lines.append("}")
    return "\n".join(lines)


def format_struct_def(sd: StructDef) -> str:
    body = ", ".join(f"{f.name}: {_format_type(f.type)}" for f in sd.fields)
    return f"struct {sd.name} {{ {body} }}"


def format_function(fn: Function, *, label: NodeLabel = _NO_LABEL) -> str:
    sig_params = ", ".join(f"{p.name}: {_format_type(p.type)}" for p in fn.params)
    header = f"{label(fn)}{fn.name}({sig_params}) -> {_format_type(fn.return_type)}"
    if fn.claims:
        header += "  [claims: " + ", ".join(format_claim(c) for c in fn.claims) + "]"
    lines: list[str] = []
    for note in fn.notes:
        lines.append(f"// {note}")
    lines.append(header + " {")
    for s in fn.body:
        lines.append(_format_stmt(s, indent=2, label=label))
    lines.append("}")
    return "\n".join(lines)


def _format_type(t) -> str:
    match t:
        case I1Type():
            return "i1"
        case I8Type():
            return "i8"
        case I16Type():
            return "i16"
        case I32Type():
            return "i32"
        case I64Type():
            return "i64"
        case I8PtrType():
            return "i8*"
        case StructType(name=n):
            return n
    raise ValueError(f"unhandled type: {t!r}")


def format_claim(c: Claim) -> str:
    match c:
        case NonNegativeClaim(param=p):
            head = f"non_negative({p})"
        case IntRangeClaim(param=p, min=lo, max=hi):
            lo_s = "-inf" if lo is None else str(lo)
            hi_s = "+inf" if hi is None else str(hi)
            head = f"int_range({p}, [{lo_s}, {hi_s}])"
        case ReturnInRangeClaim(min=lo, max=hi):
            lo_s = "-inf" if lo is None else str(lo)
            hi_s = "+inf" if hi is None else str(hi)
            head = f"return_in_range([{lo_s}, {hi_s}])"
        case _:
            raise ValueError(f"unhandled claim: {c!r}")
    return head + format_claim_metadata(c)


def format_claim_metadata(c: Claim) -> str:
    """Return ` {regime,enforcement,justification}` if any field is non-default, else ''."""
    bits: list[str] = []
    if c.regime != "axiom":
        bits.append(f"regime={c.regime}")
    if c.enforcement != "trust":
        bits.append(f"enforcement={c.enforcement}")
    if c.justification is not None:
        bits.append(f"justification={_format_justification(c.justification)}")
    return " {" + ", ".join(bits) + "}" if bits else ""


def _format_justification(j: Justification) -> str:
    match j:
        case Z3Justification(artifact_path=p, artifact_hash=h):
            return f"z3({p}@{h[:12]})"
        case ManualJustification(signed_by=s):
            return f"manual(signed_by={s!r})"
        case DerivedJustification(analysis=a, inputs=i):
            return f"derived({a}, {len(i)} input(s))"
    raise ValueError(f"unhandled justification: {j!r}")


def _format_stmt(stmt, indent: int, *, label: NodeLabel) -> str:
    pad = " " * indent
    prefix = label(stmt)
    match stmt:
        case ReturnInt(value=v):
            return f"{pad}{prefix}return {v}"
        case ReturnExpr(value=expr):
            return f"{pad}{prefix}return {_format_expr(expr)}"
        case If(cond=cond, then_body=t_body, else_body=e_body):
            then_lines = "\n".join(_format_stmt(s, indent + 2, label=label) for s in t_body)
            head = f"{pad}{prefix}if ({_format_expr(cond)}) {{"
            if not e_body:
                return f"{head}\n{then_lines}\n{pad}}}"
            else_lines = "\n".join(_format_stmt(s, indent + 2, label=label) for s in e_body)
            return f"{head}\n{then_lines}\n{pad}}} else {{\n{else_lines}\n{pad}}}"
        case Let(name=n, type=ty, init=init):
            return f"{pad}{prefix}let {n}: {_format_type(ty)} = {_format_expr(init)}"
        case Assign(name=n, value=v):
            return f"{pad}{prefix}{n} = {_format_expr(v)}"
        case FieldSet(local=loc, name=fname, value=v):
            return f"{pad}{prefix}{loc}.{fname} = {_format_expr(v)}"
        case While(cond=cond, body=body):
            body_lines = "\n".join(_format_stmt(s, indent + 2, label=label) for s in body)
            return f"{pad}{prefix}while ({_format_expr(cond)}) {{\n{body_lines}\n{pad}}}"
        case For(var=var, type=ty, lo=lo, hi=hi, body=body):
            body_lines = "\n".join(_format_stmt(s, indent + 2, label=label) for s in body)
            return (
                f"{pad}{prefix}for {var}: {_format_type(ty)} in "
                f"{_format_expr(lo)}..{_format_expr(hi)} {{\n"
                f"{body_lines}\n{pad}}}"
            )
        case ExprStmt(value=v):
            return f"{pad}{prefix}{_format_expr(v)}"
        case WithArena(name=n, capacity=cap, body=body):
            body_lines = "\n".join(_format_stmt(s, indent + 2, label=label) for s in body)
            return (
                f"{pad}{prefix}with_arena {n} = arena_new({_format_expr(cap)}) {{\n"
                f"{body_lines}\n{pad}}}"
            )
    raise ValueError(f"unhandled stmt: {stmt!r}")


_BINOP_SYMBOL = {
    "add": "+", "sub": "-", "mul": "*", "sdiv": "/", "udiv": "/u", "srem": "%",
    "slt": "<", "sle": "<=", "sgt": ">", "sge": ">=", "eq": "==", "ne": "!=",
    "ult": "<u", "ule": "<=u", "ugt": ">u", "uge": ">=u",
    "or": "|", "and": "&",
}


def _format_expr(expr) -> str:
    match expr:
        case IntLit(value=v):
            return str(v)
        case ParamRef(name=n):
            return n
        case LocalRef(name=n):
            return n
        case BinOp(op=op, lhs=l, rhs=r):
            return f"({_format_expr(l)} {_BINOP_SYMBOL[op]} {_format_expr(r)})"
        case ShortCircuitOr(lhs=l, rhs=r):
            return f"({_format_expr(l)} || {_format_expr(r)})"
        case ShortCircuitAnd(lhs=l, rhs=r):
            return f"({_format_expr(l)} && {_format_expr(r)})"
        case Call(function=fn_name, args=args):
            return f"{fn_name}({', '.join(_format_expr(a) for a in args)})"
        case StringRef(name=n):
            return f"&{n}"
        case FieldRead(value=inner, name=fname):
            return f"{_format_expr(inner)}.{fname}"
        case StructInit(type=tname, fields=field_inits):
            inner = ", ".join(f"{fi.name}: {_format_expr(fi.value)}" for fi in field_inits)
            return f"{tname} {{ {inner} }}"
        case PtrOffset(base=b, offset=o):
            return f"({_format_expr(b)} + {_format_expr(o)})"
        case Widen(value=v, target=t, signed=signed):
            kind = "" if signed else "u"
            return f"{kind}widen({_format_expr(v)} to {_format_type(t)})"
        case Load(ptr=p, type=t):
            return f"load[{_format_type(t)}]({_format_expr(p)})"
    raise ValueError(f"unhandled expr: {expr!r}")
