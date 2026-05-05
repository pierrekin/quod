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
    EnumDef,
    EnumInit,
    EnumType,
    ExprStmt,
    FieldRead,
    FieldSet,
    For,
    If,
    Let,
    Load,
    LoadField,
    Match,
    Program,
    PtrOffset,
    Return,
    ReturnExpr,
    ShortCircuitAnd,
    ShortCircuitOr,
    SizeOf,
    Store,
    StoreField,
    StructDef,
    StructInit,
    StructType,
    TryExpr,
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
class _Ctx:
    """Per-validation state. `where` is the current Location; we mutate
    its `detail` as we descend into expressions / statements so any
    diagnostic emitted gets a meaningful position."""
    structs: dict[str, StructDef]
    enums: dict[str, EnumDef]
    diagnostics: list[Diagnostic] = field(default_factory=list)
    where: Location = field(default_factory=Location)

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


# ---------- Entry point ----------


def validate(program: Program) -> tuple[Diagnostic, ...]:
    """Run all semantic checks; return all diagnostics found. Empty
    tuple means the program passed validation."""
    structs = {sd.name: sd for sd in program.structs}
    enums = {ed.name: ed for ed in program.enums}
    ctx = _Ctx(structs=structs, enums=enums)

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
        ctx.where = Location(function=fn.name)
        _check_type(ctx, fn.return_type, detail="return type")
        for p in fn.params:
            _check_type(ctx, p.type, detail=f"param {p.name!r}")
        for stmt in fn.body:
            _check_stmt(ctx, stmt)

    return tuple(ctx.diagnostics)


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
        case ReturnExpr(value=expr) | ExprStmt(value=expr):
            _check_expr(ctx, expr)
        case Return():
            pass
        case Let(type=ty, init=expr):
            _check_type(ctx, ty, detail="let")
            _check_expr(ctx, expr)
        case Assign(value=expr):
            _check_expr(ctx, expr)
        case FieldSet(value=expr):
            _check_expr(ctx, expr)
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
            for s in t_body:
                _check_stmt(ctx, s)
            for s in e_body:
                _check_stmt(ctx, s)
        case While(cond=cond, body=body):
            _check_expr(ctx, cond)
            for s in body:
                _check_stmt(ctx, s)
        case For(lo=lo, hi=hi, body=body):
            _check_expr(ctx, lo)
            _check_expr(ctx, hi)
            for s in body:
                _check_stmt(ctx, s)
        case WithArena(capacity=cap, body=body):
            _check_expr(ctx, cap)
            for s in body:
                _check_stmt(ctx, s)
        case Match():
            _check_match(ctx, stmt)
        case _:
            # Unknown stmt kinds are out of scope here — lowering will
            # blow up if a stmt is genuinely unhandled. Validation only
            # owns the rules listed above.
            pass


def _check_match(ctx: _Ctx, m: Match) -> None:
    _check_expr(ctx, m.scrutinee)

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
        for s in arm.body:
            _check_stmt(ctx, s)
        if arm.variant in seen_arms:
            ctx.emit(DUPLICATE_MATCH_ARM,
                     f"match has duplicate arm for variant {arm.variant!r}")
        seen_arms.add(arm.variant)

    if wildcard_count > 1:
        ctx.emit(MULTIPLE_WILDCARDS, "match has more than one wildcard arm `_`")

    has_wildcard = "_" in seen_arms
    scrut_enum = _scrutinee_enum_name(m.scrutinee, ctx.enums)
    if scrut_enum is None:
        return  # Can't statically tell the enum; lower-time runtime check.

    ed = ctx.enums[scrut_enum]
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
        case FieldRead(value=inner):
            _check_expr(ctx, inner)
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
        case Call(args=args):
            for a in args:
                _check_expr(ctx, a)
        case ShortCircuitAnd(lhs=a, rhs=b) | ShortCircuitOr(lhs=a, rhs=b):
            _check_expr(ctx, a)
            _check_expr(ctx, b)
        case TryExpr(value=v):
            _check_expr(ctx, v)
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
        case _:
            # Leaves (IntLit, LocalRef, ParamRef, StringRef, NullPtr,
            # CharLit, EnumPayloadRead, etc.) — nothing to recurse into.
            pass


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


def _scrutinee_enum_name(expr, enums: dict[str, EnumDef]) -> str | None:
    """Best-effort static enum-name extraction. Mirrors the old helper
    in model.py; in v1 we only check exhaustiveness when the scrutinee
    is a literal EnumInit. Other shapes get a runtime check at lower."""
    match expr:
        case EnumInit(enum=name) if name in enums:
            return name
    return None
