"""Top-level definitions — Function, struct/enum defs, Param, extern + linkage.

`function_callees` lives here too: it's a Function helper that walks a
function body collecting called function names.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, model_serializer, model_validator

from quod.model.base import _Node, _mint_function_id
from quod.model.claims import Claim, claim_param
from quod.model.expressions import (
    BinOp,
    Call,
    EnumInit,
    FieldRead,
    Load,
    PtrOffset,
    ShortCircuitAnd,
    ShortCircuitOr,
    StructInit,
    TryExpr,
    Widen,
)
from quod.model.layer_b import BlockOrScoped
from quod.model.statements import (
    Assign,
    ExprStmt,
    FieldSet,
    For,
    If,
    Let,
    Match,
    ReturnExpr,
    Store,
    While,
    WithArena,
)
from quod.model.types import (
    I1Type,
    I8Type,
    I16Type,
    I32Type,
    I64Type,
    IsizeType,
    ReturnType,
    Type,
    U8Type,
    U16Type,
    U32Type,
    U64Type,
    UsizeType,
)


class StructField(_Node):
    """One field in a StructDef. Field types may be any `Type`, including
    other structs (no recursion: a struct can't directly contain itself
    by value)."""
    name: str
    type: Type


class TypeParam(_Node):
    """One type parameter on a generic StructDef / EnumDef / Function.

    `name` is what TypeParamRef binds against (`T`, `A`, `K`, …).
    `bound`, when set, names a TraitDef; the monomorphizer rejects an
    instantiation whose concrete type lacks an `impl <bound> for <type>`.
    Bounds are only meaningful once trait dispatch lands; until then,
    `bound=None` is the only valid form.
    """
    name: str
    bound: str | None = None

    @model_serializer(mode="wrap")
    def _drop_none_bound(self, handler, info):
        data = handler(self)
        if self.bound is None:
            data.pop("bound", None)
        return data


class StructDef(_Node):
    """A named record type. Fields are ordered and uniquely named.

    By-value semantics: lowered to an LLVM identified struct type, passed
    and returned as values, no implicit pointer indirection. Pointers to
    structs aren't yet modeled — use opaque `i8*` if you need to hand one
    to an extern.

    `type_params` lists this struct's type parameters, e.g.
    `(TypeParam(name="T"),)` for `struct Box<T> { value: T }`. Field
    types may reference these via `TypeParamRef`. A struct with
    non-empty `type_params` is generic and gets monomorphized into one
    fresh nominal struct per concrete `type_args` tuple before lowering.
    """
    name: str
    type_params: tuple[TypeParam, ...] = ()
    fields: tuple[StructField, ...]

    @model_serializer(mode="wrap")
    def _drop_empty_type_params(self, handler, info):
        data = handler(self)
        if not self.type_params:
            data.pop("type_params", None)
        return data

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


class EnumPayloadField(_Node):
    """One payload field of an EnumVariant. Any value Type is allowed —
    int widths, i8*, named structs, even other enums. Variants lower
    to per-variant LLVM struct types stored in the enum's payload byte
    array via bitcast."""
    name: str
    type: Type


class EnumVariant(_Node):
    """One variant of an EnumDef. The empty `fields` tuple means a unit
    variant (no payload, like `JsonValue::Null`)."""
    name: str
    fields: tuple[EnumPayloadField, ...] = ()

    def field(self, name: str) -> EnumPayloadField | None:
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def field_index(self, name: str) -> int:
        for i, f in enumerate(self.fields):
            if f.name == name:
                return i
        raise KeyError(f"variant {self.name!r} has no field {name!r}")


class EnumDef(_Node):
    """A named tagged-union type. Variants are ordered (first variant gets
    discriminant 0) and uniquely named within the enum.

    Lowered to an LLVM identified struct `{i8 tag, [N x i64] payload}`
    where N = max(1, max(len(v.fields) for v in variants)). EnumInit
    bitcasts payload to a per-variant LLVM struct type to set fields;
    Match likewise bitcasts to extract bindings.

    `type_params` is the same generic-parameter list as on StructDef.
    Generic enums are monomorphized away before lowering.
    """
    name: str
    type_params: tuple[TypeParam, ...] = ()
    variants: tuple[EnumVariant, ...]

    @model_serializer(mode="wrap")
    def _drop_empty_type_params(self, handler, info):
        data = handler(self)
        if not self.type_params:
            data.pop("type_params", None)
        return data

    def variant(self, name: str) -> EnumVariant | None:
        for v in self.variants:
            if v.name == name:
                return v
        return None

    def variant_index(self, name: str) -> int:
        for i, v in enumerate(self.variants):
            if v.name == name:
                return i
        raise KeyError(f"enum {self.name!r} has no variant {name!r}")

    def max_payload_slots(self) -> int:
        """Number of i64 slots needed to hold the largest variant's
        payload. At least 1 to avoid [0 x i64] arrays at lower time."""
        return max((len(v.fields) for v in self.variants), default=0) or 1

    def try_variants(self) -> tuple["EnumVariant | None", "EnumVariant | None"]:
        """If this enum is `?`-eligible, return (happy_variant, sad_variant);
        otherwise (None, None). Eligible iff: exactly two variants, one with
        exactly one payload field (happy), one with zero (sad). Variant
        names are irrelevant — Ok/Err, Some/None, Found/Missing etc. all
        qualify by shape."""
        if len(self.variants) != 2:
            return (None, None)
        a, b = self.variants
        if len(a.fields) == 1 and len(b.fields) == 0:
            return (a, b)
        if len(b.fields) == 1 and len(a.fields) == 0:
            return (b, a)
        return (None, None)


class Param(_Node):
    """A typed function parameter. Any `Type` is accepted (int widths,
    `i8*`, named structs)."""
    name: str
    type: Type


class Function(_Node):
    # Stable opaque ID — endpoint of provenance and equivalence edges
    # across language-family layers. Auto-minted at construction;
    # persists in JSON so reloads stay deterministic. Hand-supplied IDs
    # (in JSON) override the default.
    id: str = Field(default_factory=_mint_function_id)
    name: str
    type_params: tuple[TypeParam, ...] = ()
    params: tuple[Param, ...] = ()
    return_type: ReturnType
    # `body` widens to `Block | CScopedBlock` (smart union) to host the
    # c-family scope wrapper at layer B. Layer C is pure core, so
    # `lower.py` refuses the wrapper at codegen time; the c-family
    # lowering pass strips it before producing layer C. Existing
    # all-core programs are unaffected — Pydantic's smart union picks
    # `Block` for any body without the wrapper's `kind` field.
    body: BlockOrScoped
    claims: tuple[Claim, ...] = ()
    notes: tuple[str, ...] = ()       # free-form developer/agent intent

    @model_serializer(mode="wrap")
    def _drop_defaults(self, handler, info):
        data = handler(self)
        if not self.notes:
            data.pop("notes", None)
        if not self.type_params:
            data.pop("type_params", None)
        return data

    def param(self, name: str) -> Param | None:
        for p in self.params:
            if p.name == name:
                return p
        return None


# ---------- Extern linkage ----------
#
# Records where an extern's symbol is expected to come from. Today this is
# purely declarative — the build pipeline resolves libc via clang and the
# quod runtime via libquodrt regardless of what's declared. The annotation
# exists so future extern-level claims can carry provenance ("axiom: trust
# the libc manpage" vs "witness: we proved our runtime against this") and
# so future tools can validate that a `linkage.runtime` symbol actually
# exists in the shipped runtime.

class LibcLinkage(_Node):
    kind: Literal["linkage.libc"] = "linkage.libc"


class RuntimeLinkage(_Node):
    kind: Literal["linkage.runtime"] = "linkage.runtime"


Linkage = Annotated[
    Union[LibcLinkage, RuntimeLinkage],
    Field(discriminator="kind"),
]


class ExternFunction(_Node):
    """A libc-or-similar function declared but not defined by us.

    `arity` is a convenience for all-i32 signatures: when set, it expands
    to `param_types = (I32Type,) * arity` and `return_type = I32Type` at
    use time. For non-i32 sigs, set `param_types` and `return_type` directly
    and leave `arity` at 0. Set `varargs=True` for variadic libc functions
    like printf — callers may pass any number of args after the fixed prefix.

    `linkage` records the symbol's provenance (libc vs quod runtime) and
    is required — every extern has a home, and "unspecified" is not a
    real semantic state. The annotation is what lets future extern-level
    claims carry provenance ("axiom: trust the libc manpage" vs "witness:
    we proved our runtime against this") and what lets future tools
    validate that a `linkage.runtime` symbol actually exists in the
    shipped runtime archive.

    `claims` is a tuple of contracts the caller may exploit at every call
    site. Currently restricted to return-scoped kinds (e.g.
    `return_in_range`); param-scoped kinds (`non_negative`, `int_range`)
    need named extern params, which is a follow-up model migration.
    """
    name: str
    arity: int = 0
    param_types: tuple[Type, ...] = ()
    return_type: ReturnType = I32Type()
    varargs: bool = False
    linkage: Linkage
    claims: tuple[Claim, ...] = ()

    @model_validator(mode="after")
    def _check_claims_supported(self):
        ret_is_int = isinstance(
            self.return_type,
            (I1Type, I8Type, I16Type, I32Type, I64Type,
             U8Type, U16Type, U32Type, U64Type,
             IsizeType, UsizeType),
        )
        for c in self.claims:
            if claim_param(c) is not None:
                raise ValueError(
                    f"extern {self.name!r}: predicate references parameter "
                    f"{claim_param(c)!r}, but externs don't yet carry named "
                    f"params. Only return-scoped predicates are supported on "
                    f"externs."
                )
            if not ret_is_int:
                raise ValueError(
                    f"extern {self.name!r}: predicate over return value "
                    f"requires an integer return type, got "
                    f"{self.return_type.kind!r}"
                )
        return self

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
        if not self.claims:
            data.pop("claims", None)
        return data

    def effective_param_types(self) -> tuple["Type", ...]:
        """Resolved param types: explicit `param_types` if given, otherwise
        `arity` copies of I32Type."""
        if self.param_types:
            return self.param_types
        return tuple(I32Type() for _ in range(self.arity))


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
            case EnumInit(fields=field_inits):
                for fi in field_inits:
                    visit_expr(fi.value)
            case PtrOffset(base=b, offset=o):
                visit_expr(b)
                visit_expr(o)
            case Widen(value=v):
                visit_expr(v)
            case Load(ptr=p):
                visit_expr(p)
            case TryExpr(value=v):
                visit_expr(v)
            case _:
                pass

    def visit_stmt(s) -> None:
        match s:
            case ReturnExpr(value=expr) | ExprStmt(value=expr):
                visit_expr(expr)
            case If(cond=cond, then_body=t_body, else_body=e_body):
                visit_expr(cond)
                for x in t_body.stmts:
                    visit_stmt(x)
                for x in e_body.stmts:
                    visit_stmt(x)
            case Let(init=expr) | Assign(value=expr) | FieldSet(value=expr):
                visit_expr(expr)
            case Store(ptr=p, value=v):
                visit_expr(p)
                visit_expr(v)
            case While(cond=cond, body=body):
                visit_expr(cond)
                for x in body.stmts:
                    visit_stmt(x)
            case For(lo=lo, hi=hi, body=body):
                visit_expr(lo)
                visit_expr(hi)
                for x in body.stmts:
                    visit_stmt(x)
            case WithArena(capacity=cap, body=body):
                visit_expr(cap)
                for x in body.stmts:
                    visit_stmt(x)
            case Match(scrutinee=scrut, arms=arms):
                visit_expr(scrut)
                for arm in arms:
                    for x in arm.body.stmts:
                        visit_stmt(x)
            case _:
                pass

    for stmt in fn.body.stmts:
        visit_stmt(stmt)
    return tuple(seen)
