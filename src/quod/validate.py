"""Semantic validation pass.

Sits between construction and lowering: takes a Program (already
structurally well-formed via the Pydantic model), checks the semantic
rules that depend on whole-program context, and returns a list of
diagnostics. Lowering assumes a validated program — its remaining
`raise ValueError` lines are bug surfaces, not user errors.

Pipeline order:
    parse → resolve_imports → validate(generic) → monomorphize
                                                 → validate(concrete) → lower

The pre-mono pass catches errors that produce better messages on the
generic program (unbound type params, generic-arity mismatches). The
post-mono pass is the real validator — by then every type is concrete
and every reference can be checked for resolution.

Today the structural checks (uniqueness, no-cycles) still live in the
Pydantic `model_validator`; this module owns everything that requires
whole-program context (resolve refs, exhaustiveness, scoping, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from quod.model import (
    Assign,
    BinOp,
    Call,
    CharLit,
    EnumDef,
    EnumInit,
    EnumType,
    ExprStmt,
    ExternFunction,
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
    U8Type,
    U16Type,
    U32Type,
    U64Type,
    IsizeType,
    UsizeType,
    If,
    IfExpr,
    IntLit,
    Let,
    Load,
    LoadField,
    LocalRef,
    Match,
    Not,
    NullPtr,
    Param,
    ParamRef,
    PredicateClaim,
    Program,
    PtrOffset,
    TypeParamRef,
    Return,
    ReturnExpr,
    ReturnRef,
    ShortCircuitAnd,
    ShortCircuitOr,
    SizeOf,
    Store,
    StoreField,
    StringRef,
    StructDef,
    StructInit,
    StructType,
    Break,
    Continue,
    DoWhile,
    TraitCall,
    TryExpr,
    Unreachable,
    VoidType,
    While,
    Widen,
    WithArena,
)


# ---------- Error codes ----------
#
# String constants — tooling-friendly, no numeric churn. Keep names
# stable across versions; new codes get appended.

UNRESOLVED_STRUCT = "unresolved_struct"
UNRESOLVED_ENUM = "unresolved_enum"
UNKNOWN_VARIANT = "unknown_variant"
UNKNOWN_FIELD = "unknown_field"
DUPLICATE_FIELD_INIT = "duplicate_field_init"
MISSING_FIELD_INIT = "missing_field_init"
EXTRA_FIELD_INIT = "extra_field_init"
NON_EXHAUSTIVE_MATCH = "non_exhaustive_match"
EXTRA_MATCH_ARM = "extra_match_arm"
DUPLICATE_MATCH_ARM = "duplicate_match_arm"
WILDCARD_BINDS = "wildcard_binds"
MULTIPLE_WILDCARDS = "multiple_wildcards"
DUPLICATE_BINDING = "duplicate_binding"
MATCH_ARITY = "match_arity"

# Phase 2: scope and reference checks.
UNDECLARED_LOCAL = "undeclared_local"
UNDEFINED_FUNCTION = "undefined_function"
LOCAL_DECLARED_TWICE = "local_declared_twice"
LOCAL_SHADOWS_PARAM = "local_shadows_param"
FOR_VAR_CONFLICT = "for_var_conflict"
TRY_INELIGIBLE_ENUM = "try_ineligible_enum"
TRY_RETURN_TYPE_MISMATCH = "try_return_type_mismatch"
TRY_NON_ENUM = "try_non_enum"
BARE_RETURN_NON_VOID = "bare_return_non_void"
RETURN_EXPR_VOID = "return_expr_void"
ASSIGN_UNDECLARED_LOCAL = "assign_undeclared_local"
FIELDSET_UNDECLARED_LOCAL = "fieldset_undeclared_local"
UNDECLARED_PARAM = "undeclared_param"
READ_OF_UNINIT_LOCAL = "read_of_uninit_local"
BREAK_OUTSIDE_LOOP = "break_outside_loop"
CONTINUE_OUTSIDE_LOOP = "continue_outside_loop"

# Phase 3: type-aware checks (downstream of inference).
FIELD_READ_NON_STRUCT = "field_read_non_struct"
FIELDSET_NON_STRUCT_LOCAL = "fieldset_non_struct_local"


# ---------- Diagnostic ----------


@dataclass(frozen=True)
class Location:
    """Logical position in the program graph. Points at a definition
    or a path within it; not a source span (we don't have those from
    JSON-loaded programs)."""
    function: str | None = None
    extern: str | None = None
    struct: str | None = None
    enum: str | None = None
    detail: str | None = None  # free-form, e.g. "let init", "match scrutinee"

    def __str__(self) -> str:
        parts: list[str] = []
        if self.function:
            parts.append(f"function {self.function!r}")
        if self.extern:
            parts.append(f"extern {self.extern!r}")
        if self.struct:
            parts.append(f"struct {self.struct!r}")
        if self.enum:
            parts.append(f"enum {self.enum!r}")
        if self.detail:
            parts.append(self.detail)
        return ": ".join(parts) if parts else "<program>"


@dataclass(frozen=True)
class Diagnostic:
    severity: Literal["error", "warning"]
    code: str
    message: str
    location: Location

    def format(self) -> str:
        return f"[{self.code}] {self.location}: {self.message}"


class ValidationError(Exception):
    """Raised when validate() finds errors and the caller wants a hard fail.
    Holds the full list so tooling can render all of them."""
    def __init__(self, diagnostics: tuple[Diagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        msg = "\n".join(d.format() for d in diagnostics)
        super().__init__(msg)


# ---------- Internal walker state ----------


@dataclass
class _CallableSig:
    """Just enough about a function/extern to validate calls and
    type-infer call expressions."""
    return_type: object  # a Type node from model.py
    is_void: bool


@dataclass
class _Ctx:
    """Per-validation state. `where` is the current Location; mutated as
    we descend so each diagnostic has a meaningful position.

    The function-scoped state (`fn`, `params`, `locals`, `arm_bindings`)
    is only populated during a function-body walk; outside of that it's
    None / empty.
    """
    structs: dict[str, StructDef]
    enums: dict[str, EnumDef]
    callables: dict[str, _CallableSig]
    diagnostics: list[Diagnostic] = field(default_factory=list)
    where: Location = field(default_factory=Location)
    # Per-function fields (set up by _check_function before walking):
    fn: Function | None = None
    params: dict[str, object] = field(default_factory=dict)   # name -> Type
    locals: dict[str, object] = field(default_factory=dict)   # name -> Type
    # Stack of arm-scoped binding maps (innermost last).
    arm_bindings: list[dict[str, object]] = field(default_factory=list)
    # Number of enclosing loops (`While` / `For`). Incremented when
    # entering a loop body, decremented on exit; `Break` / `Continue`
    # are valid only when this is positive.
    loop_depth: int = 0

    def emit(self, code: str, message: str, *, detail: str | None = None) -> None:
        loc = self.where if detail is None else Location(
            function=self.where.function,
            extern=self.where.extern,
            struct=self.where.struct,
            enum=self.where.enum,
            detail=detail,
        )
        self.diagnostics.append(Diagnostic(
            severity="error", code=code, message=message, location=loc,
        ))

    def name_in_scope(self, name: str) -> bool:
        """Is this name a declared local, parameter, or active match-arm
        binding? Used for the existence check (UNDECLARED_LOCAL); type
        information is not required."""
        for scope in reversed(self.arm_bindings):
            if name in scope:
                return True
        return name in self.locals or name in self.params

    def lookup_local_type(self, name: str) -> object | None:
        """Return the declared Type for a name in scope, or None if the
        name isn't declared *or* the type couldn't be inferred (e.g.,
        an arm binding whose variant wasn't statically known). Callers
        that need the existence check should use name_in_scope."""
        for scope in reversed(self.arm_bindings):
            if name in scope:
                return scope[name]
        if name in self.locals:
            return self.locals[name]
        if name in self.params:
            return self.params[name]
        return None


# ---------- Entry point ----------


def validate(program: Program) -> tuple[Diagnostic, ...]:
    """Run all semantic checks; return all diagnostics found. Empty
    tuple means the program passed validation."""
    structs = {sd.name: sd for sd in program.structs}
    enums = {ed.name: ed for ed in program.enums}
    callables: dict[str, _CallableSig] = {}
    for ext in program.externs:
        callables[ext.name] = _CallableSig(
            return_type=ext.return_type,
            is_void=isinstance(ext.return_type, VoidType),
        )
    for fn in program.functions:
        callables[fn.name] = _CallableSig(
            return_type=fn.return_type,
            is_void=isinstance(fn.return_type, VoidType),
        )
    ctx = _Ctx(structs=structs, enums=enums, callables=callables)

    # Type refs in struct field types.
    for sd in program.structs:
        ctx.where = Location(struct=sd.name)
        for f in sd.fields:
            _check_type(ctx, f.type, detail=f"field {f.name!r}")

    # Type refs in extern signatures.
    for ext in program.externs:
        ctx.where = Location(extern=ext.name)
        _check_type(ctx, ext.return_type, detail="return type")
        for i, t in enumerate(ext.param_types):
            _check_type(ctx, t, detail=f"param {i}")

    # Function signatures + bodies.
    for fn in program.functions:
        _check_function(ctx, fn)

    return tuple(ctx.diagnostics)


def _check_function(ctx: _Ctx, fn: Function) -> None:
    """Set up per-function scope, walk body, then tear down."""
    ctx.where = Location(function=fn.name)
    ctx.fn = fn
    ctx.params = {p.name: p.type for p in fn.params}
    ctx.locals = {}
    ctx.arm_bindings = []

    _check_type(ctx, fn.return_type, detail="return type")
    for p in fn.params:
        _check_type(ctx, p.type, detail=f"param {p.name!r}")

    # Pre-pass: collect all let/for-introduced locals, checking dup &
    # shadowing. Match-arm bindings are arm-scoped, so they're handled
    # inline during the main walk, not collected here.
    _collect_locals(ctx, fn.body)

    # Main walk: check each statement against the populated scope.
    for stmt in fn.body.stmts:
        _check_stmt(ctx, stmt)

    # Definite-init analysis: locals introduced by `Let(init=None)` are
    # uninitialized at the binding point. Reading them is undefined
    # behaviour (matches C `int x;`); we refuse if any path can reach
    # a read before a definite write.
    _check_definite_init(ctx, fn.body, initially_defined=set(ctx.params.keys()))

    # Reset (paranoid hygiene; callers should always set these afresh).
    ctx.fn = None
    ctx.params = {}
    ctx.locals = {}
    ctx.arm_bindings = []


def _collect_locals(ctx: _Ctx, body) -> None:
    """Walk every Let/For in the body, populating `ctx.locals`.
    Emits LOCAL_DECLARED_TWICE / LOCAL_SHADOWS_PARAM / FOR_VAR_CONFLICT
    when a name conflicts. Match-arm bindings are NOT collected here —
    they're scoped to their arm body. `body` is a `Block`."""
    for s in body.stmts:
        match s:
            case Let(name=name, type=ty):
                _declare_local(ctx, name, ty, kind="let")
            case For(var=var, type=ty, body=for_body):
                if var in ctx.locals or var in ctx.params:
                    ctx.emit(FOR_VAR_CONFLICT,
                             f"for-loop var {var!r} conflicts with another local or parameter")
                else:
                    ctx.locals[var] = ty
                _collect_locals(ctx, for_body)
            case If(then_body=t, else_body=e):
                _collect_locals(ctx, t)
                _collect_locals(ctx, e)
            case While(body=w_body):
                _collect_locals(ctx, w_body)
            case Match(arms=arms):
                for arm in arms:
                    _collect_locals(ctx, arm.body)
            case WithArena(name=name, body=wa_body):
                # Arena handle is bound for the duration of the body.
                # Lowering desugars to a Let of i8*, so we mirror that here.
                _declare_local(ctx, name, I8PtrType(), kind="with_arena")
                _collect_locals(ctx, wa_body)
            case _:
                pass


def _declare_local(ctx: _Ctx, name: str, ty, *, kind: str) -> None:
    if name in ctx.params:
        ctx.emit(LOCAL_SHADOWS_PARAM,
                 f"local {name!r} shadows parameter of the enclosing function")
        return
    if name in ctx.locals:
        ctx.emit(LOCAL_DECLARED_TWICE,
                 f"local {name!r} declared twice in the same function")
        return
    ctx.locals[name] = ty


def validate_or_raise(program: Program) -> None:
    """Validate; raise ValidationError if any diagnostics surfaced."""
    diags = validate(program)
    if diags:
        raise ValidationError(diags)


# ---------- Checks ----------


def _check_type(ctx: _Ctx, t, *, detail: str | None = None) -> None:
    """Confirm a type reference resolves (StructType/EnumType point at
    something defined). Other types are leaves with nothing to check."""
    if isinstance(t, StructType) and t.name not in ctx.structs:
        ctx.emit(UNRESOLVED_STRUCT,
                 f"references undefined struct {t.name!r}", detail=detail)
    if isinstance(t, EnumType) and t.name not in ctx.enums:
        ctx.emit(UNRESOLVED_ENUM,
                 f"references undefined enum {t.name!r}", detail=detail)


def _check_stmt(ctx: _Ctx, stmt) -> None:
    match stmt:
        case ReturnExpr(value=expr):
            _check_expr(ctx, expr)
            if ctx.fn is not None and isinstance(ctx.fn.return_type, VoidType):
                ctx.emit(RETURN_EXPR_VOID,
                         f"function returns void; use bare `return`, not return_expr")
        case Return():
            if ctx.fn is not None and not isinstance(ctx.fn.return_type, VoidType):
                ctx.emit(BARE_RETURN_NON_VOID,
                         f"function returns {_type_name(ctx.fn.return_type)}, "
                         f"not void; use return_expr")
        case ExprStmt(value=expr):
            _check_expr(ctx, expr)
        case Let(name=name, type=ty, init=expr):
            _check_type(ctx, ty, detail="let")
            if expr is not None:
                _check_expr(ctx, expr)
        case Assign(name=name, value=v):
            if not ctx.name_in_scope(name):
                ctx.emit(ASSIGN_UNDECLARED_LOCAL,
                         f"assign to undeclared local {name!r}")
            _check_expr(ctx, v)
        case FieldSet(local=lname, name=fname, value=v):
            if not ctx.name_in_scope(lname):
                ctx.emit(FIELDSET_UNDECLARED_LOCAL,
                         f"field-set on undeclared local {lname!r}")
            else:
                local_ty = ctx.lookup_local_type(lname)
                # Skip if type unknown or still generic — both resolve
                # post-mono, where this same validator pass re-checks.
                if local_ty is None or isinstance(local_ty, TypeParamRef):
                    pass
                elif not isinstance(local_ty, StructType):
                    ctx.emit(FIELDSET_NON_STRUCT_LOCAL,
                             f"field-set {fname!r} on non-struct local "
                             f"{lname!r} (local type {_type_name(local_ty)})")
                else:
                    sd = ctx.structs.get(local_ty.name)
                    if sd is None:
                        ctx.emit(UNRESOLVED_STRUCT,
                                 f"field-set on local {lname!r} of unknown "
                                 f"struct {local_ty.name!r}")
                    elif sd.field(fname) is None:
                        ctx.emit(UNKNOWN_FIELD,
                                 f"field-set references unknown field "
                                 f"{fname!r} of struct {local_ty.name!r}")
            _check_expr(ctx, v)
        case Store(ptr=p, value=v):
            _check_expr(ctx, p)
            _check_expr(ctx, v)
        case StoreField(ptr=p, struct_type=tname, name=fname, value=v):
            _check_expr(ctx, p)
            _check_expr(ctx, v)
            sd = ctx.structs.get(tname)
            if sd is None:
                ctx.emit(UNRESOLVED_STRUCT,
                         f"store_field references undefined struct {tname!r}")
            elif sd.field(fname) is None:
                ctx.emit(UNKNOWN_FIELD,
                         f"store_field references unknown field {fname!r} "
                         f"of struct {tname!r}")
        case If(cond=cond, then_body=t_body, else_body=e_body):
            _check_expr(ctx, cond)
            for s in t_body.stmts:
                _check_stmt(ctx, s)
            for s in e_body.stmts:
                _check_stmt(ctx, s)
        case While(cond=cond, body=body):
            _check_expr(ctx, cond)
            ctx.loop_depth += 1
            try:
                for s in body.stmts:
                    _check_stmt(ctx, s)
            finally:
                ctx.loop_depth -= 1
        case DoWhile(cond=cond, body=body):
            ctx.loop_depth += 1
            try:
                for s in body.stmts:
                    _check_stmt(ctx, s)
            finally:
                ctx.loop_depth -= 1
            _check_expr(ctx, cond)
        case For(lo=lo, hi=hi, body=body):
            _check_expr(ctx, lo)
            _check_expr(ctx, hi)
            ctx.loop_depth += 1
            try:
                for s in body.stmts:
                    _check_stmt(ctx, s)
            finally:
                ctx.loop_depth -= 1
        case Break():
            if ctx.loop_depth == 0:
                ctx.emit(BREAK_OUTSIDE_LOOP, "`break` outside any enclosing loop")
        case Continue():
            if ctx.loop_depth == 0:
                ctx.emit(CONTINUE_OUTSIDE_LOOP, "`continue` outside any enclosing loop")
        case WithArena(capacity=cap, body=body):
            _check_expr(ctx, cap)
            for s in body.stmts:
                _check_stmt(ctx, s)
        case Match():
            _check_match(ctx, stmt)
        case _:
            # Unknown stmt kinds are out of scope here — lowering will
            # blow up if a stmt is genuinely unhandled. Validation only
            # owns the rules listed above.
            pass


def _check_definite_init(ctx: _Ctx, body, *, initially_defined: set[str]) -> None:
    """Forward must-init analysis. Emits READ_OF_UNINIT_LOCAL when any
    `LocalRef(x)` is reachable on a path where `x` was introduced by
    `Let(init=None)` and not yet definitely written.

    `initially_defined` seeds the analysis with names that are
    pre-defined at the function entry (parameters). The walk returns
    the set of names definitely-defined when control falls through the
    body; callers use this to merge across branches (intersection).
    """
    _walk_definite_init(ctx, body, set(initially_defined))


def _walk_definite_init(ctx: _Ctx, body, defined: set[str]) -> set[str] | None:
    """Walk `body.stmts` updating `defined`. Returns the set of names
    definitely-defined when control falls through, or None if every
    path through `body` terminates (return / unreachable) — a sentinel
    that lets callers merge cleanly.
    """
    for stmt in body.stmts:
        new_defined = _check_stmt_init(ctx, stmt, defined)
        if new_defined is None:
            return None  # this stmt terminates control flow
        defined = new_defined
    return defined


def _check_stmt_init(ctx: _Ctx, stmt, defined: set[str]) -> set[str] | None:
    """Process one statement: (1) check that any LocalRef it contains
    references a name in `defined`; (2) update and return the new
    `defined` set, or None if control flow doesn't fall through."""
    match stmt:
        case Let(name=name, init=init):
            if init is not None:
                _check_expr_reads(ctx, init, defined)
                defined = defined | {name}
            # init=None: name remains undefined until a future Assign.
            return defined
        case Assign(name=name, value=v):
            _check_expr_reads(ctx, v, defined)
            return defined | {name}
        case ExprStmt(value=expr) | ReturnExpr(value=expr):
            _check_expr_reads(ctx, expr, defined)
            return None if isinstance(stmt, ReturnExpr) else defined
        case Return() | Unreachable() | Break() | Continue():
            return None
        case If(cond=cond, then_body=t, else_body=e):
            _check_expr_reads(ctx, cond, defined)
            then_def = _walk_definite_init(ctx, t, set(defined))
            else_def = _walk_definite_init(ctx, e, set(defined))
            # Intersection — only locals defined on BOTH paths are
            # definitely-defined after the if. If one branch
            # terminates (None), inherit the other's set; if both
            # terminate, the if doesn't fall through.
            if then_def is None and else_def is None:
                return None
            if then_def is None:
                return else_def
            if else_def is None:
                return then_def
            return then_def & else_def
        case While(cond=cond, body=b):
            _check_expr_reads(ctx, cond, defined)
            # Body might not execute (zero-iteration case), so any
            # writes inside don't count as definite. Still need to
            # check reads inside the body against `defined`.
            _walk_definite_init(ctx, b, set(defined))
            return defined
        case DoWhile(cond=cond, body=b):
            # Body always runs at least once, so writes inside *can*
            # count toward definite-init for reads of cond and after.
            # But subsequent iterations re-execute body in arbitrary
            # order, so we conservatively merge: defined-after-body
            # if it falls through, else just `defined`.
            after_body = _walk_definite_init(ctx, b, set(defined))
            if after_body is None:
                # body never falls through (every path returns/breaks)
                return defined
            _check_expr_reads(ctx, cond, after_body)
            return after_body
        case For(var=var, lo=lo, hi=hi, body=b):
            _check_expr_reads(ctx, lo, defined)
            _check_expr_reads(ctx, hi, defined)
            # Loop body sees `var` as defined; same zero-iteration
            # caveat as While for any other writes.
            _walk_definite_init(ctx, b, set(defined) | {var})
            return defined
        case FieldSet(local=lname, value=v):
            _check_expr_reads(ctx, v, defined)
            return defined  # field-set doesn't define the local itself
        case Store(ptr=p, value=v):
            _check_expr_reads(ctx, p, defined)
            _check_expr_reads(ctx, v, defined)
            return defined
        case StoreField(ptr=p, value=v):
            _check_expr_reads(ctx, p, defined)
            _check_expr_reads(ctx, v, defined)
            return defined
        case Match(scrutinee=scrut, arms=arms):
            _check_expr_reads(ctx, scrut, defined)
            arm_defs: list[set[str] | None] = []
            for arm in arms:
                arm_defs.append(_walk_definite_init(
                    ctx, arm.body, set(defined) | set(arm.bindings),
                ))
            non_none = [d for d in arm_defs if d is not None]
            if not non_none:
                return None  # every arm terminates
            result = non_none[0]
            for d in non_none[1:]:
                result = result & d
            # Locals introduced inside arms are scoped to the arm; only
            # names that were already in `defined` (or universally
            # defined across arms) survive.
            return result & defined if not arms else result if non_none else defined
        case WithArena(name=name, capacity=cap, body=b):
            _check_expr_reads(ctx, cap, defined)
            _walk_definite_init(ctx, b, set(defined) | {name})
            return defined
        case _:
            return defined


def _check_expr_reads(ctx: _Ctx, expr, defined: set[str]) -> None:
    """Walk `expr` and emit READ_OF_UNINIT_LOCAL at every LocalRef whose
    name is declared but not in `defined`."""
    match expr:
        case LocalRef(name=name):
            if name in ctx.locals and name not in defined:
                ctx.emit(
                    READ_OF_UNINIT_LOCAL,
                    f"local {name!r} is read before any definite write — "
                    f"declared without an initializer, no path through "
                    f"the program writes it before this read"
                )
        case BinOp(lhs=l, rhs=r):
            _check_expr_reads(ctx, l, defined)
            _check_expr_reads(ctx, r, defined)
        case ShortCircuitAnd(lhs=l, rhs=r) | ShortCircuitOr(lhs=l, rhs=r):
            _check_expr_reads(ctx, l, defined)
            _check_expr_reads(ctx, r, defined)
        case IfExpr(cond=cond, then_value=t, else_value=e):
            _check_expr_reads(ctx, cond, defined)
            _check_expr_reads(ctx, t, defined)
            _check_expr_reads(ctx, e, defined)
        case Call(args=args):
            for a in args:
                _check_expr_reads(ctx, a, defined)
        case PtrOffset(base=b, offset=o):
            _check_expr_reads(ctx, b, defined)
            _check_expr_reads(ctx, o, defined)
        case Widen(value=v):
            _check_expr_reads(ctx, v, defined)
        case Load(ptr=p):
            _check_expr_reads(ctx, p, defined)
        case LoadField(ptr=p):
            _check_expr_reads(ctx, p, defined)
        case FieldRead(value=v):
            _check_expr_reads(ctx, v, defined)
        case StructInit(fields=fis):
            for fi in fis:
                _check_expr_reads(ctx, fi.value, defined)
        case EnumInit(fields=fis):
            for fi in fis:
                _check_expr_reads(ctx, fi.value, defined)
        case TryExpr(value=v):
            _check_expr_reads(ctx, v, defined)
        case TraitCall(args=args):
            for a in args:
                _check_expr_reads(ctx, a, defined)
        case _:
            # Leaves: IntLit, ParamRef, StringRef, NullPtr, CharLit,
            # SizeOf — no LocalRef inside.
            pass


def _check_match(ctx: _Ctx, m: Match) -> None:
    _check_expr(ctx, m.scrutinee)

    # Try to resolve the scrutinee's enum type — used both for
    # exhaustiveness AND for binding the right payload types in arm
    # scopes.
    scrut_ty = _infer_type(ctx, m.scrutinee)
    scrut_enum = scrut_ty.name if isinstance(scrut_ty, EnumType) else None
    ed = ctx.enums.get(scrut_enum) if scrut_enum else None

    seen_arms: set[str] = set()
    wildcard_count = 0
    for arm in m.arms:
        if arm.variant == "_":
            wildcard_count += 1
            if arm.bindings:
                ctx.emit(WILDCARD_BINDS,
                         "match wildcard arm `_` cannot take bindings "
                         "(use a named variant arm if you need the payload)")
        seen_binding: set[str] = set()
        for b in arm.bindings:
            if b in seen_binding:
                ctx.emit(DUPLICATE_BINDING,
                         f"match arm for {arm.variant!r} binds {b!r} more than once")
            seen_binding.add(b)

        # Build per-arm scope: bindings → declared payload field types
        # if we know the variant, else just register the names so
        # LocalRef inside the arm doesn't false-positive.
        arm_scope: dict[str, object] = {}
        if ed is not None and arm.variant != "_":
            var = ed.variant(arm.variant)
            if var is not None:
                for i, b in enumerate(arm.bindings):
                    if i < len(var.fields):
                        arm_scope[b] = var.fields[i].type
        # Names not paired with a known field type still get a sentinel
        # so undefined-local checks accept them (we just can't infer
        # their type — lowering would catch a true mismatch).
        for b in arm.bindings:
            arm_scope.setdefault(b, None)

        ctx.arm_bindings.append(arm_scope)
        try:
            for s in arm.body.stmts:
                _check_stmt(ctx, s)
        finally:
            ctx.arm_bindings.pop()

        if arm.variant in seen_arms:
            ctx.emit(DUPLICATE_MATCH_ARM,
                     f"match has duplicate arm for variant {arm.variant!r}")
        seen_arms.add(arm.variant)

    if wildcard_count > 1:
        ctx.emit(MULTIPLE_WILDCARDS, "match has more than one wildcard arm `_`")

    if ed is None:
        return  # Can't statically tell the enum; lower-time runtime check.

    has_wildcard = "_" in seen_arms
    declared = {v.name for v in ed.variants}
    named_arms = seen_arms - {"_"}
    missing = declared - named_arms
    extra = named_arms - declared
    if missing and not has_wildcard:
        ctx.emit(NON_EXHAUSTIVE_MATCH,
                 f"match on {scrut_enum!r} non-exhaustive — "
                 f"missing {sorted(missing)} (use `_` for a default arm)")
    if extra:
        ctx.emit(EXTRA_MATCH_ARM,
                 f"match on {scrut_enum!r} has unknown variant arms {sorted(extra)}")
    for arm in m.arms:
        if arm.variant == "_":
            continue
        var = ed.variant(arm.variant)
        if var is not None and len(arm.bindings) != len(var.fields):
            ctx.emit(MATCH_ARITY,
                     f"match arm {scrut_enum}::{arm.variant} binds "
                     f"{len(arm.bindings)} field(s), expected {len(var.fields)}")


def _check_expr(ctx: _Ctx, expr) -> None:
    match expr:
        case StructInit(type=name, fields=field_inits):
            sd = ctx.structs.get(name)
            if sd is None:
                ctx.emit(UNRESOLVED_STRUCT,
                         f"struct_init references undefined struct {name!r}")
            else:
                _check_field_inits(ctx, name, field_inits, {f.name for f in sd.fields},
                                   what="struct_init")
            for fi in field_inits:
                _check_expr(ctx, fi.value)
        case EnumInit(enum=ename, variant=vname, fields=field_inits):
            ed = ctx.enums.get(ename)
            if ed is None:
                ctx.emit(UNRESOLVED_ENUM,
                         f"enum_init references undefined enum {ename!r}")
            else:
                var = ed.variant(vname)
                if var is None:
                    ctx.emit(UNKNOWN_VARIANT,
                             f"enum_init references unknown variant "
                             f"{ename}::{vname}")
                else:
                    _check_field_inits(ctx, f"{ename}::{vname}", field_inits,
                                       {f.name for f in var.fields},
                                       what="enum_init")
            for fi in field_inits:
                _check_expr(ctx, fi.value)
        case FieldRead(value=inner, name=fname):
            _check_expr(ctx, inner)
            inner_ty = _infer_type(ctx, inner)
            # Skip if unknown or generic (TypeParamRef gets resolved at
            # mono time; the post-mono validation pass re-checks).
            if inner_ty is None or isinstance(inner_ty, TypeParamRef):
                pass
            elif not isinstance(inner_ty, StructType):
                ctx.emit(FIELD_READ_NON_STRUCT,
                         f"field read {fname!r} on non-struct value of "
                         f"type {_type_name(inner_ty)}")
            else:
                sd = ctx.structs.get(inner_ty.name)
                if sd is not None and sd.field(fname) is None:
                    ctx.emit(UNKNOWN_FIELD,
                             f"field read references unknown field "
                             f"{fname!r} of struct {inner_ty.name!r}")
        case LoadField(ptr=p, struct_type=tname, name=fname):
            _check_expr(ctx, p)
            sd = ctx.structs.get(tname)
            if sd is None:
                ctx.emit(UNRESOLVED_STRUCT,
                         f"load_field references undefined struct {tname!r}")
            elif sd.field(fname) is None:
                ctx.emit(UNKNOWN_FIELD,
                         f"load_field references unknown field {fname!r} "
                         f"of struct {tname!r}")
        case BinOp(lhs=l, rhs=r):
            _check_expr(ctx, l)
            _check_expr(ctx, r)
        case Call(function=fname, args=args):
            if fname not in ctx.callables:
                ctx.emit(UNDEFINED_FUNCTION,
                         f"call to undefined function {fname!r}")
            for a in args:
                _check_expr(ctx, a)
        case ShortCircuitAnd(lhs=a, rhs=b) | ShortCircuitOr(lhs=a, rhs=b):
            _check_expr(ctx, a)
            _check_expr(ctx, b)
        case IfExpr(cond=cond, then_value=t, else_value=e):
            _check_expr(ctx, cond)
            _check_expr(ctx, t)
            _check_expr(ctx, e)
        case TryExpr(value=v):
            _check_expr(ctx, v)
            _check_try(ctx, v)
        case PtrOffset(base=p, offset=o):
            _check_expr(ctx, p)
            _check_expr(ctx, o)
        case Widen(value=v):
            _check_expr(ctx, v)
        case Load(ptr=p, type=t):
            _check_expr(ctx, p)
            _check_type(ctx, t)
        case SizeOf(type=t):
            _check_type(ctx, t)
        case LocalRef(name=name):
            if not ctx.name_in_scope(name):
                ctx.emit(UNDECLARED_LOCAL,
                         f"reference to undeclared local {name!r}")
        case ParamRef(name=name):
            if ctx.fn is not None and name not in ctx.params:
                ctx.emit(UNDECLARED_PARAM,
                         f"reference to undeclared param {name!r}")
        case _:
            # Leaves (IntLit, StringRef, NullPtr, CharLit, etc.) —
            # nothing to recurse into.
            pass


def _check_try(ctx: _Ctx, inner) -> None:
    """Validate a TryExpr's enum eligibility and return-type match.
    Type inference tells us the inner enum statically; if it's not
    inferrable, we skip and let lower catch it (rare in practice)."""
    if ctx.fn is None:
        return
    inner_ty = _infer_type(ctx, inner)
    if inner_ty is None or isinstance(inner_ty, TypeParamRef):
        return  # Unknown / generic; post-mono pass re-checks.
    if not isinstance(inner_ty, EnumType):
        ctx.emit(TRY_NON_ENUM,
                 f"? requires an enum value, got {_type_name(inner_ty)}")
        return
    ed = ctx.enums.get(inner_ty.name)
    if ed is None:
        # Already emitted UNRESOLVED_ENUM elsewhere; don't double-report.
        return
    happy, sad = ed.try_variants()
    if happy is None:
        ctx.emit(TRY_INELIGIBLE_ENUM,
                 f"? on enum {ed.name!r}: not ?-eligible "
                 f"(needs exactly two variants — one with a single "
                 f"payload field, one with no payload)")
        return
    ret_ty = ctx.fn.return_type
    if not (isinstance(ret_ty, EnumType) and ret_ty.name == ed.name):
        ctx.emit(TRY_RETURN_TYPE_MISMATCH,
                 f"? on {ed.name!r} requires the enclosing function to "
                 f"return {ed.name!r}, got {_type_name(ret_ty)}")


def _check_field_inits(
    ctx: _Ctx,
    target: str,
    field_inits: tuple,
    declared: set[str],
    *,
    what: str,
) -> None:
    seen: set[str] = set()
    for fi in field_inits:
        if fi.name in seen:
            ctx.emit(DUPLICATE_FIELD_INIT,
                     f"{what} for {target} sets field {fi.name!r} twice")
        seen.add(fi.name)
    extra = seen - declared
    if extra:
        ctx.emit(EXTRA_FIELD_INIT,
                 f"{what} for {target} sets unknown field(s): {sorted(extra)}")
    missing = declared - seen
    if missing:
        ctx.emit(MISSING_FIELD_INIT,
                 f"{what} for {target} missing field(s): {sorted(missing)}")


# ---------- Type inference ----------
#
# Best-effort: returns the static type of an expression when we can tell
# (literals, scope lookups, calls to known functions, struct/enum_init,
# field reads on inferrable structs, ?-extracted payloads). Returns None
# when we can't — the caller treats `None` as "skip this check, let
# lower report at codegen time."


def _infer_type(ctx: _Ctx, expr) -> object | None:
    match expr:
        case IntLit(type=t):
            return t
        case CharLit():
            return I8Type()
        case NullPtr():
            return I8PtrType()
        case StringRef():
            return I8PtrType()
        case LocalRef(name=name):
            return ctx.lookup_local_type(name)
        case ParamRef(name=name):
            return ctx.params.get(name)
        case Call(function=fname):
            sig = ctx.callables.get(fname)
            return sig.return_type if sig is not None else None
        case StructInit(type=name):
            return StructType(name=name)
        case EnumInit(enum=ename):
            return EnumType(name=ename)
        case Load(type=t):
            return t
        case SizeOf():
            return I64Type()
        case Widen(target=t):
            return t
        case ShortCircuitAnd() | ShortCircuitOr():
            return I1Type()
        case IfExpr(then_value=t, else_value=e):
            # Both branches must agree by validation; pick the first
            # one that infers a type.
            return _infer_type(ctx, t) or _infer_type(ctx, e)
        case BinOp(op=op, lhs=l, rhs=r):
            # Comparisons return i1; arithmetic propagates the LHS type.
            if op in ("eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"):
                return I1Type()
            return _infer_type(ctx, l) or _infer_type(ctx, r)
        case PtrOffset():
            return I8PtrType()
        case LoadField(struct_type=tname, name=fname):
            sd = ctx.structs.get(tname)
            if sd is None:
                return None
            f = sd.field(fname)
            return f.type if f is not None else None
        case FieldRead(value=inner, name=fname):
            inner_ty = _infer_type(ctx, inner)
            if not isinstance(inner_ty, StructType):
                return None
            sd = ctx.structs.get(inner_ty.name)
            if sd is None:
                return None
            f = sd.field(fname)
            return f.type if f is not None else None
        case TryExpr(value=inner):
            inner_ty = _infer_type(ctx, inner)
            if not isinstance(inner_ty, EnumType):
                return None
            ed = ctx.enums.get(inner_ty.name)
            if ed is None:
                return None
            happy, _ = ed.try_variants()
            if happy is None or len(happy.fields) != 1:
                return None
            return happy.fields[0].type
    return None


def _type_name(t) -> str:
    """Render a Type for diagnostic messages. Cheap, no formatter dep."""
    match t:
        case None:
            return "?"
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
        case U8Type():
            return "u8"
        case U16Type():
            return "u16"
        case U32Type():
            return "u32"
        case U64Type():
            return "u64"
        case IsizeType():
            return "isize"
        case UsizeType():
            return "usize"
        case I8PtrType():
            return "i8*"
        case VoidType():
            return "void"
        case StructType(name=n):
            return n
        case EnumType(name=n):
            return n
        case _:
            return repr(t)


# ---------- Predicate validation ----------
#
# A `PredicateClaim.expr` is a structurally-restricted `Expr`:
# i1-typed at the top level, side-effect-free, references only params
# or `ReturnRef`. The model-level type is `Expr` for graph-edge
# uniformity (one expression vocabulary, not two), so the structural
# constraint is enforced by this helper rather than the type system.
#
# Used at PredicateClaim construction time and during the validate
# pass when a Function's claims include predicate forms.

class PredicateError(ValueError):
    """Raised when an expression cannot be a predicate (impure op,
    aggregate access, dangling reference, etc.). The message names the
    offending node kind."""


def assert_is_predicate(expr) -> None:
    """Raise `PredicateError` if `expr` is not a valid predicate body.

    Allowed: IntLit, ParamRef, ReturnRef, BinOp, ShortCircuitOr/And,
    Not. Forbidden: Calls, aggregate access, memory ops, locals, or
    anything else with side effects or non-predicate semantics.
    """
    _walk_predicate(expr)


def _walk_predicate(expr) -> None:
    if isinstance(expr, IntLit):
        return
    if isinstance(expr, ParamRef):
        return
    if isinstance(expr, ReturnRef):
        return
    if isinstance(expr, Not):
        _walk_predicate(expr.operand)
        return
    if isinstance(expr, BinOp):
        _walk_predicate(expr.lhs)
        _walk_predicate(expr.rhs)
        return
    if isinstance(expr, (ShortCircuitOr, ShortCircuitAnd)):
        _walk_predicate(expr.lhs)
        _walk_predicate(expr.rhs)
        return
    raise PredicateError(
        f"predicate cannot contain {type(expr).__name__} — predicates "
        f"must be side-effect-free and reference only params or ReturnRef"
    )
