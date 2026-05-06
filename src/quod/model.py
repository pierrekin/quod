"""CPG data model + pretty-printer + immutable update helpers.

The graph is the asset. Nodes are frozen Pydantic models; mutators return
new Programs via model_copy. Addressing is by name *or* content-hash prefix
(the latter implemented in quod.hashing / quod.editor).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal, Union
from uuid import uuid4

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
      arith (u) — udiv, urem                        : iN in / iN out
      cmp (s)   — slt, sle, sgt, sge, eq, ne       : iN in / i1 out
      cmp (u)   — ult, ule, ugt, uge               : iN in / i1 out
      bitwise   — or, and, xor                      : iN in / iN out
      shifts    — shl, ashr, lshr                   : iN in / iN out

    Operands of arith/bitwise/cmp must have the same type; LLVM's verifier
    enforces this at lower time. The signed/unsigned distinction matches
    LLVM IR predicates — signedness lives on the op, not the type. Division
    by zero is undefined behaviour (matches LLVM); guard with an int_range
    or runtime branch if the divisor isn't statically nonzero.

    Shifts: `shl` is logical left, `ashr` is arithmetic (sign-extending)
    right, `lshr` is logical (zero-extending) right. Shift count must
    have the same iN type as the value; shift count >= bitwidth is
    undefined behaviour (matches LLVM).

    For short-circuit boolean combinators (correct in the presence of
    side-effecting operands), use `ShortCircuitOr` / `ShortCircuitAnd` —
    those are i1-only and synthesise branches.
    """
    kind: Literal["llvm.binop"] = "llvm.binop"
    op: Literal[
        "add", "sub", "mul", "sdiv", "udiv", "srem", "urem",
        "slt", "sle", "sgt", "sge", "eq", "ne",
        "ult", "ule", "ugt", "uge",
        "or", "and", "xor",
        "shl", "ashr", "lshr",
    ]
    lhs: "Expr"
    rhs: "Expr"


class ShortCircuitOr(_Node):
    """`lhs || rhs` with C-style short-circuit. If `lhs` is true, `rhs` is
    not evaluated. Lowered to branch + phi."""
    kind: Literal["quod.sc_or"] = "quod.sc_or"
    lhs: "Expr"
    rhs: "Expr"


class IfExpr(_Node):
    """Expression-level `if`: evaluate `cond` (i1), then evaluate exactly
    one of `then_value` / `else_value` and yield it as the expression's
    value. Both branches must produce values of the same type.

    Used by the C ingester to lift the ternary operator (`cond ? a : b`)
    and by any language whose front-end produces expression-level
    conditionals. Lowered to branch + phi (the same shape as
    `ShortCircuitAnd` / `ShortCircuitOr`); only the branch whose
    condition matches is evaluated, so side-effecting branches are
    correctly sequenced.
    """
    kind: Literal["quod.if_expr"] = "quod.if_expr"
    cond: "Expr"        # must lower to i1
    then_value: "Expr"
    else_value: "Expr"  # same type as then_value


class ShortCircuitAnd(_Node):
    """`lhs && rhs` with C-style short-circuit. If `lhs` is false, `rhs` is
    not evaluated. Lowered to branch + phi."""
    kind: Literal["quod.sc_and"] = "quod.sc_and"
    lhs: "Expr"
    rhs: "Expr"


class Not(_Node):
    """Boolean negation. Operand must lower to i1; result is i1.

    Lowered to `xor operand, 1` (cheap on every backend). Cleaner than
    encoding negation as `eq(x, 0)` — the latter conflates "is this
    value zero" (defined for any iN) with "is this boolean false."
    """
    kind: Literal["quod.not"] = "quod.not"
    operand: "Expr"


class ReturnRef(_Node):
    """Symbolic reference to the enclosing function's return value.

    Valid only inside `PredicateClaim.expr`. The validator rejects
    ReturnRef outside a predicate context. Type is determined by the
    enclosing function's return type — there's no type field, mirroring
    `ParamRef`.
    """
    kind: Literal["quod.return_ref"] = "quod.return_ref"


class Call(_Node):
    """Call a user function or an extern in the same Program.

    User function calls are i32-in/i32-out. Extern calls follow the extern's
    declared `param_types` / `return_type` — pass `StringRef` for i8*-typed
    args, IntLit/ParamRef/etc. for i32 args. For varargs externs (printf etc.),
    pass any number of args beyond the fixed prefix.

    For a generic function, `type_args` provides the concrete instantiation.
    Post-monomorphization, `function` carries the mangled name and
    `type_args` is empty.
    """
    kind: Literal["llvm.call"] = "llvm.call"
    function: str
    type_args: tuple["Type", ...] = ()
    args: tuple["Expr", ...] = ()

    @model_serializer(mode="wrap")
    def _drop_empty_type_args(self, handler, info):
        data = handler(self)
        if not self.type_args:
            data.pop("type_args", None)
        return data


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


class LoadField(_Node):
    """Read a named field from a struct stored at an i8* pointer.

    `ptr` must lower to i8*; `struct_type` names a StructDef; `name` is
    a field of that struct's def. Lowered as `bitcast(ptr, T*)` +
    `getelementptr` (field index) + `load` — a direct field access, no
    whole-struct copy.

    Equivalent in effect to `FieldRead(Load(ptr, StructType(struct_type)),
    name)` but emits the targeted GEP+load instead of materializing the
    whole struct first. Use this for struct-on-heap field reads where the
    register-pressure of a whole-struct load would be wasteful (the arena
    allocator's hot path is the canonical example).

    `type_args` populates the type parameters when `struct_type` names a
    generic StructDef. The monomorphizer mangles `struct_type` using the
    args (same pattern as StructInit/EnumInit/Call) and clears `type_args`.
    Empty tuple is the non-generic case."""
    kind: Literal["quod.load_field"] = "quod.load_field"
    ptr: "Expr"
    struct_type: str
    type_args: tuple["Type", ...] = ()
    name: str


class FieldInit(_Node):
    """One field's value in a `StructInit`."""
    name: str
    value: "Expr"


class StructInit(_Node):
    """Construct a struct value. Every field of the named def must be
    initialized exactly once, in any order. Lowered to an `insertvalue`
    chain on `undef`.

    For a generic struct, `type_args` provides the concrete instantiation.
    Post-monomorphization, `type` carries the mangled name and `type_args`
    is empty.
    """
    kind: Literal["quod.struct_init"] = "quod.struct_init"
    type: str
    type_args: tuple["Type", ...] = ()
    fields: tuple[FieldInit, ...]

    @model_serializer(mode="wrap")
    def _drop_empty_type_args(self, handler, info):
        data = handler(self)
        if not self.type_args:
            data.pop("type_args", None)
        return data


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


class NullPtr(_Node):
    """The null i8* literal. Lowers to `i8* null`.

    Useful as the placeholder for unused pointer-typed struct fields
    (e.g. JsonValue.str_ptr when tag != string), since `quod.struct_init`
    requires every field to be initialised.
    """
    kind: Literal["quod.null_ptr"] = "quod.null_ptr"


class TryExpr(_Node):
    """Postfix `?` propagation. `value` must produce a value of a
    2-variant enum where exactly one variant has a single payload field
    (the "happy" variant) and the other has no payload (the "sad"
    variant). Variant names don't matter — Ok/Err, Some/None, Found/
    Missing all qualify.

    On evaluation: if value is the sad variant, the enclosing function
    immediately returns the sad variant (function return type must be
    the same enum). If happy, the expression evaluates to the payload
    field value.

    Lowered as: spill value to alloca, switch on tag; sad-arm
    constructs the return type's sad variant and rets; happy-arm
    bitcasts the payload to the variant struct, GEP+loads the single
    field. The TryExpr's value is that loaded field.
    """
    kind: Literal["quod.try"] = "quod.try"
    value: "Expr"


class SizeOf(_Node):
    """Size in bytes of a quod type, computed at lower time from LLVM's
    target data layout. Returns i64. Useful for stride-correct pointer
    arithmetic over arena-allocated arrays of structs or enums.

    Lowered to a constant — `getelementptr null, i32 1` then `ptrtoint`
    is the classic LLVM trick — but practically we just ask llvmlite's
    target_data for the ABI size.
    """
    kind: Literal["quod.sizeof"] = "quod.sizeof"
    type: "Type"


class EnumInit(_Node):
    """Construct a value of an enum type by selecting a variant and
    initializing its payload fields. Lowered to: store tag byte + bitcast
    payload + insertvalue chain into the variant's struct shape.

    Validation: `enum` must name an `EnumDef`, `variant` must name one of
    its variants, and `fields` must cover exactly the variant's payload
    fields by name with matching types.

    For a generic enum, `type_args` provides the concrete instantiation.
    Post-monomorphization, `enum` carries the mangled name and
    `type_args` is empty.
    """
    kind: Literal["quod.enum_init"] = "quod.enum_init"
    enum: str
    type_args: tuple["Type", ...] = ()
    variant: str
    fields: tuple[FieldInit, ...] = ()

    @model_serializer(mode="wrap")
    def _drop_empty_type_args(self, handler, info):
        data = handler(self)
        if not self.type_args:
            data.pop("type_args", None)
        return data


class TraitCall(_Node):
    """Call a method declared in a `TraitDef`, dispatched on
    `dispatch_type`. Pre-mono, `dispatch_type` may be a `TypeParamRef`
    (the dispatch resolves once the type-param is bound). At mono time,
    `dispatch_type` is substituted to a concrete type, the impl
    `impl <trait> for <concrete>` is looked up, and this node is
    rewritten to a regular `Call` to that impl's mangled method name.

    Post-monomorphization, no `TraitCall` remains.

    `args[0]` is the receiver value (the impl's first parameter, by
    convention `self`); subsequent args follow the method signature.
    Keeping the receiver in `args` rather than a separate field matches
    how the lowered Call ends up: a flat positional argument list.
    """
    kind: Literal["quod.trait_call"] = "quod.trait_call"
    trait: str
    method: str
    dispatch_type: "Type"
    args: tuple["Expr", ...] = ()


class CharLit(_Node):
    """A byte literal written as a single-character string. Lowers to
    `const_int i8 ord(value)`.

    `value` must be a single character with codepoint < 256 (Latin-1
    range). JSON's native string escapes work — `"\\n"` is one byte 10.
    Use this instead of `{"kind": "llvm.const_int", "type": "i8",
    "value": 110}` when you mean `'n'`.
    """
    kind: Literal["quod.char_lit"] = "quod.char_lit"
    value: str

    @field_validator("value")
    @classmethod
    def _one_byte(cls, v: str) -> str:
        if len(v) != 1:
            raise ValueError(
                f"char_lit value must be exactly 1 character, got {v!r} "
                f"(length {len(v)})"
            )
        if ord(v) > 255:
            raise ValueError(
                f"char_lit value must fit in a byte (codepoint < 256), "
                f"got {v!r} (codepoint {ord(v)})"
            )
        return v


Expr = Annotated[
    Union[
        IntLit, ParamRef, LocalRef, BinOp, ShortCircuitOr, ShortCircuitAnd,
        IfExpr, Not, ReturnRef,
        Call, StringRef, FieldRead, LoadField, StructInit, PtrOffset, Widen,
        Load, NullPtr, CharLit, EnumInit, SizeOf, TryExpr, TraitCall,
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


class U8Type(_Node):
    kind: Literal["llvm.u8"] = "llvm.u8"


class U16Type(_Node):
    kind: Literal["llvm.u16"] = "llvm.u16"


class U32Type(_Node):
    kind: Literal["llvm.u32"] = "llvm.u32"


class U64Type(_Node):
    kind: Literal["llvm.u64"] = "llvm.u64"


class IsizeType(_Node):
    """Pointer-sized signed integer. Lowers to i64 on the only target
    (64-bit Linux). Distinct from i64 at the type level so APIs that
    talk about "a count or offset that fits a pointer" can say so."""
    kind: Literal["llvm.isize"] = "llvm.isize"


class UsizeType(_Node):
    """Pointer-sized unsigned integer. Lowers to i64 width on the only
    target (64-bit Linux). Used for sizes, lengths, and indices."""
    kind: Literal["llvm.usize"] = "llvm.usize"


class I8PtrType(_Node):
    kind: Literal["llvm.i8_ptr"] = "llvm.i8_ptr"


class StructType(_Node):
    """Reference to a named StructDef. Pass-by-value at the LLVM level.

    The `name` must match a `StructDef.name` in the same Program; the
    Program-level validator catches dangling refs at load time.

    `type_args` populates the type parameters of a generic StructDef. An
    empty tuple is the non-generic case (matches a StructDef with empty
    `type_params`). The monomorphization pass walks every StructType
    with non-empty `type_args`, generates a fresh nominal struct, and
    rewrites this reference to the mangled name with empty type_args.
    Post-mono, every StructType has empty `type_args`.
    """
    kind: Literal["llvm.struct"] = "llvm.struct"
    name: str
    type_args: tuple["Type", ...] = ()

    @model_serializer(mode="wrap")
    def _drop_empty_type_args(self, handler, info):
        data = handler(self)
        if not self.type_args:
            data.pop("type_args", None)
        return data


class EnumType(_Node):
    """Reference to a named EnumDef. Pass-by-value at the LLVM level.

    Lowered as a tagged union: i8 discriminant + [N x i64] payload, where
    N is `max(1, max(len(variant.fields)))` and each payload field occupies
    one i64-sized slot (so payload field types are restricted to scalar
    types — int widths up to i64, plus i8*; struct and enum payload
    fields are not yet supported).

    `type_args`: same story as StructType — the monomorphization pass
    rewrites generic instantiations to mangled-name references with
    empty `type_args`.
    """
    kind: Literal["llvm.enum"] = "llvm.enum"
    name: str
    type_args: tuple["Type", ...] = ()

    @model_serializer(mode="wrap")
    def _drop_empty_type_args(self, handler, info):
        data = handler(self)
        if not self.type_args:
            data.pop("type_args", None)
        return data


class TypeParamRef(_Node):
    """A reference to an in-scope type parameter, e.g. `T` inside a generic
    `struct List<T> { ptr: *T, ... }`.

    Only valid inside the body of a generic StructDef / EnumDef / Function
    whose `type_params` includes `name`. The monomorphization pass
    substitutes every TypeParamRef for the corresponding concrete `Type`
    when emitting a monomorphized def. Post-mono, no TypeParamRef
    survives — encountering one at lower time is a bug.
    """
    kind: Literal["quod.type_param"] = "quod.type_param"
    name: str


class SelfType(_Node):
    """Reference to the receiver type inside a `TraitDef` method
    signature or an `ImplDef` method. `ImplDef`'s post-construction
    validator eagerly rewrites every `SelfType` inside its methods to
    the impl's `for_type`, so by the time the lowerer runs no `SelfType`
    survives — a `SelfType` reaching mono is a bug in TraitDef plumbing.
    """
    kind: Literal["quod.self_type"] = "quod.self_type"


class VoidType(_Node):
    """The LLVM `void` type. Only valid as a function return type.

    Functions returning void must use the bare `quod.return` statement
    (no value) and may not appear in any value position.
    """
    kind: Literal["llvm.void"] = "llvm.void"


# Integer-only sub-union: usable wherever a pointer would be nonsense
# (IntLit, function params/return, For loop var, claim-bearing locals).
# Signedness lives on the type for u8..u64 only as a typing aid — the
# LLVM lowering is the same width as the corresponding iN, and ops
# carry their own signedness (sdiv vs udiv, slt vs ult).
IntType = Annotated[
    Union[I1Type, I8Type, I16Type, I32Type, I64Type,
          U8Type, U16Type, U32Type, U64Type,
          IsizeType, UsizeType],
    Field(discriminator="kind"),
]

# Full type union, including pointer, struct, and enum types — used for
# Let bindings, struct fields, and other value-bearing contexts. Void is
# deliberately excluded; see ReturnType for return positions.
# TypeParamRef is included for pre-monomorphization use; the mono pass
# substitutes them before lowering. SelfType is for trait/impl
# declarations; an ImplDef validator eagerly rewrites Self → for_type at
# construction time, so SelfType only appears in TraitDef method
# signatures (and the rewriter never sees it).
Type = Annotated[
    Union[I1Type, I8Type, I16Type, I32Type, I64Type,
          U8Type, U16Type, U32Type, U64Type,
          IsizeType, UsizeType, I8PtrType,
          StructType, EnumType, TypeParamRef, SelfType],
    Field(discriminator="kind"),
]

# Type that can appear at a function return position, including void.
ReturnType = Annotated[
    Union[I1Type, I8Type, I16Type, I32Type, I64Type,
          U8Type, U16Type, U32Type, U64Type,
          IsizeType, UsizeType, I8PtrType,
          StructType, EnumType, TypeParamRef, SelfType, VoidType],
    Field(discriminator="kind"),
]



def int_type_width(t: "IntType") -> int:
    """Bit width of an int type. isize/usize are pointer-sized = 64 on
    the only target."""
    match t:
        case I1Type():
            return 1
        case I8Type() | U8Type():
            return 8
        case I16Type() | U16Type():
            return 16
        case I32Type() | U32Type():
            return 32
        case I64Type() | U64Type() | IsizeType() | UsizeType():
            return 64
    raise ValueError(f"not an int type: {t!r}")


def int_type_signed(t: "IntType") -> bool:
    """Whether an int type is signed. i1 is treated as unsigned (boolean)."""
    match t:
        case I8Type() | I16Type() | I32Type() | I64Type() | IsizeType():
            return True
        case I1Type() | U8Type() | U16Type() | U32Type() | U64Type() | UsizeType():
            return False
    raise ValueError(f"not an int type: {t!r}")


# ---------- Statements ----------

class ReturnExpr(_Node):
    kind: Literal["quod.return_expr"] = "quod.return_expr"
    value: Expr


class Return(_Node):
    """Bare return for void functions. The enclosing function's return_type
    must be llvm.void; non-void functions must use return_expr."""
    kind: Literal["quod.return"] = "quod.return"


class Unreachable(_Node):
    """A statement that must not be executed at runtime. Lowers to LLVM
    `unreachable`. Used to terminate a basic block when the source
    language's semantics for reaching this point are undefined — e.g. C's
    fall-through off a non-`main` int-returning function (UB per C99
    §6.9.1/12). Lets downstream analysis flag the path as a finding rather
    than silently fabricating a return value."""
    kind: Literal["quod.unreachable"] = "quod.unreachable"


class Break(_Node):
    """Exit the innermost enclosing loop (`while` or `for`). The
    validator refuses any `Break` outside a loop body. Lowers to
    `br loop-exit`."""
    kind: Literal["quod.break"] = "quod.break"


class Continue(_Node):
    """Skip to the next iteration of the innermost enclosing loop.
    The validator refuses any `Continue` outside a loop body. Lowers
    to `br loop-cond` (or, when the source is a c-family
    `for (init; cond; inc) body`, the c-family lowering pre-rewrites
    `continue` to `inc; continue` so the inc step is preserved per
    C semantics)."""
    kind: Literal["quod.continue"] = "quod.continue"


def _mint_block_id() -> str:
    return f"@blk_{uuid4().hex[:12]}"


def _mint_function_id() -> str:
    return f"@fn_{uuid4().hex[:12]}"


class Block(_Node):
    """Identified container for a sequence of statements.

    Endpoint of provenance and equivalence edges across language-family
    layers. The `id` is opaque, minted at construction time, and
    persisted in JSON so reloads are deterministic. `Program.edges` and
    `Equivalence` claims anchor on these IDs.
    """
    id: str = Field(default_factory=_mint_block_id)
    stmts: tuple["Statement", ...] = ()


class If(_Node):
    """Two-branch conditional. Branches may both terminate (return), or both
    fall through to the next statement, or mix — a merge block is created
    on demand by the lowering pass."""
    kind: Literal["quod.if"] = "quod.if"
    cond: Expr  # must lower to i1
    then_body: Block
    else_body: Block


class Let(_Node):
    """Introduce a mutable local variable. `name` must not shadow a parameter
    or another local in the same function. Lowered to alloca-at-entry plus
    a store of `init` at the binding point. When `init` is None, the
    local starts out **uninitialized** — alloca only, no store; reading
    such a local is undefined behaviour, and the validator's forward
    must-init analysis refuses any program that lets such a read be
    reachable. The C ingester uses this for `int x;` (no initializer);
    other source languages can use it too."""
    kind: Literal["quod.let"] = "quod.let"
    name: str
    type: Type
    init: Expr | None = None


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
    body: Block


class DoWhile(_Node):
    """Post-test loop. Runs `body` unconditionally, then evaluates
    `cond` and loops back if true. The body always executes at least
    once. Inside `body`, `continue` jumps to the cond check (matching
    C's `do { ... } while (...);` semantics)."""
    kind: Literal["quod.do_while"] = "quod.do_while"
    body: Block
    cond: Expr  # must lower to i1


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
    body: Block


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


class Store(_Node):
    """Write `value` to the memory at the i8* pointer `ptr`.

    Lowered as `bitcast(ptr, T*)` + an LLVM `store` of `value` (which has
    type T). The store happens at the natural alignment of T; mismatched
    underlying memory is undefined behaviour, same caveat as `quod.load`.

    Pair with `quod.ptr_offset` to write at a non-zero offset of a buffer:
    `store(ptr_offset(buf, k), 0x42 as i8)`.
    """
    kind: Literal["quod.store"] = "quod.store"
    ptr: Expr     # must lower to i8*
    value: Expr   # iN or named struct value


class StoreField(_Node):
    """Write a value into a named field of a struct stored at an i8*
    pointer. Same shape and lowering rationale as `LoadField` but
    mutating: lowered as `bitcast(ptr, T*)` + `getelementptr` + `store`.

    Equivalent in effect to loading the whole struct, mutating one
    field, and storing it back, but emits a targeted GEP+store instead.
    The point is the same as `LoadField` — make struct-on-heap mutation
    expressible without forcing the whole struct through a register.

    `type_args` populates the type parameters when `struct_type` names a
    generic StructDef. The monomorphizer mangles `struct_type` using the
    args (same pattern as StructInit/EnumInit/Call) and clears `type_args`.
    Empty tuple is the non-generic case."""
    kind: Literal["quod.store_field"] = "quod.store_field"
    ptr: Expr                                  # must lower to i8*
    struct_type: str                           # name of a StructDef
    type_args: tuple["Type", ...] = ()         # generic instantiation, if any
    name: str                                  # field name
    value: Expr                                # the field's declared type


class WithArena(_Node):
    """Bracket a body with an arena that's freed automatically.

    Lowering is a desugar: at block entry `mem.arena.new` is called with
    `capacity` (i64) and the result (i8*) bound to a local named `name`
    for the duration of `body`. On every exit edge — fall-through and
    every `return` reachable from `body` — `mem.arena.drop` is called
    on that handle.

    The desugaring auto-injects `imports: ["mem.arena"]` if the program
    doesn't already declare it, so a `with_arena` block is one-stop sugar;
    `mem.arena.alloc` (and the other arena functions) are visible to
    code inside the body via the same import.
    """
    kind: Literal["quod.with_arena"] = "quod.with_arena"
    name: str
    capacity: Expr   # must lower to i64
    body: Block


class MatchArm(_Node):
    """One arm of a `Match`. Names a variant, binds its payload fields to
    locals (one name per field, in declaration order), and runs `body`
    with those locals in scope.

    The literal variant name `_` is a wildcard — it matches every variant
    not handled by another arm of the same match. Wildcards take no
    bindings (use a normal variant arm if you need the payload). At most
    one wildcard per match. With a wildcard present, the named-variant
    arms don't have to be exhaustive.

    Bindings are scoped to `body` only — they don't leak into sibling
    arms or out of the match.
    """
    variant: str
    bindings: tuple[str, ...] = ()
    body: Block


class Match(_Node):
    """Pattern-match on an enum value. One arm per variant, exhaustive.

    `scrutinee` must lower to a value of some `EnumType("E")`. Arms must
    cover every variant of `E` exactly once, in any order. No wildcards,
    no guards, no nested patterns yet.

    Lowered to a `switch` on the discriminant byte; each arm's body
    runs in its own basic block with the variant's payload fields
    bound as locals.
    """
    kind: Literal["quod.match"] = "quod.match"
    scrutinee: Expr
    arms: tuple[MatchArm, ...]


Statement = Annotated[
    Union[
        ReturnExpr, Return, Unreachable, Break, Continue, If, Let, Assign,
        While, DoWhile, For, ExprStmt, FieldSet, Store, StoreField, WithArena,
        Match,
        # Forward-declared `c.*` family extension. CStyleFor is defined
        # below near the staged-lift section; the union uses a string
        # forward-ref to keep its definition close to the other family
        # extensions while still allowing it as a Statement.
        "CStyleFor",
    ],
    Field(discriminator="kind"),
]


def body_always_terminates(stmts) -> bool:
    """Conservative: True only when the last reachable statement is provably
    a terminator — a `return`, an `unreachable`, a `break` or `continue`
    (which exit the enclosing loop without falling through), an `if`
    whose branches both terminate, or a `match` whose arms (and wildcard,
    if present) all terminate. Used by the C ingest to decide whether a
    fall-through needs synthesizing, and by the lowering pass to suppress
    dead trailing instructions (e.g. arena drops after a body that never
    falls through)."""
    if not stmts:
        return False
    last = stmts[-1]
    match last:
        case ReturnExpr() | Return() | Unreachable() | Break() | Continue():
            return True
        case If(then_body=t, else_body=e):
            return body_always_terminates(t.stmts) and body_always_terminates(e.stmts)
        case Match(arms=arms):
            return all(body_always_terminates(arm.body.stmts) for arm in arms)
    return False


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
    # `artifact_hash` pins the .smt2 file's bytes ("did the file get
    # tampered with"). `body_smt_hash` pins the SMT text the current
    # body+claim *would* produce ("did the body drift"). Both are
    # sha256 of the same bytes at prove time; they answer different
    # questions at verify time. See cli.py:_verify_justification.
    kind: Literal["z3"] = "z3"
    artifact_path: str
    artifact_hash: str
    body_smt_hash: str
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


class LiftEquivalence(_Node):
    """Justifies an A→B (source-language → c-like-quod) equivalence
    via a pinned *structural transcription record* — produced by
    `quod.lift_check.walk_lift`, which walks both subtrees in
    lockstep and confirms one-to-one node correspondence per a
    per-construct table.

    `artifact_path` is relative to the program's resolve_root; the
    file's bytes are hashed at prove time and re-checked at verify
    time. Verification is hash-only: re-walking the layer-A and
    layer-B subtrees in memory reproduces the artifact bytes
    deterministically, so a hash match is sufficient evidence that
    the lift is still faithful. No Z3 invocation — the artifact
    is JSON, not SMT.

    Future SMT-based A→B proofs (once we have them) would land as
    a distinct justification kind; this one is reserved for the
    structural-correspondence-record shape.
    """
    kind: Literal["lift_equivalence"] = "lift_equivalence"
    artifact_path: str
    artifact_hash: str
    note: str | None = None

    @model_serializer(mode="wrap")
    def _drop_default_metadata(self, handler, info):
        data = handler(self)
        if self.note is None:
            data.pop("note", None)
        return data


class FamilyLowering(_Node):
    """Justifies a B→C equivalence by citing a named lowering rule
    (e.g. `c.for_general`) whose equivalence theorem was proved once,
    out of band, against the rule itself rather than per program.

    `rule_name` identifies the rule in the family's lowering pass.
    `artifact_path`/`artifact_hash` optionally pin the rule's proof
    artifact; when None, the citation is a manual claim that the rule
    has been proved elsewhere.
    """
    kind: Literal["family_lowering"] = "family_lowering"
    rule_name: str
    artifact_path: str | None = None
    artifact_hash: str | None = None
    note: str | None = None

    @model_serializer(mode="wrap")
    def _drop_default_metadata(self, handler, info):
        data = handler(self)
        if self.artifact_path is None:
            data.pop("artifact_path", None)
        if self.artifact_hash is None:
            data.pop("artifact_hash", None)
        if self.note is None:
            data.pop("note", None)
        return data


Justification = Annotated[
    Union[
        Z3Justification,
        ManualJustification,
        DerivedJustification,
        LiftEquivalence,
        FamilyLowering,
    ],
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


class PredicateClaim(_Claim):
    """Single canonical claim form: a predicate over the function's
    parameters and (optionally) `ReturnRef`.

    Pre/post is implicit: a predicate that mentions `ReturnRef` is a
    postcondition, otherwise a precondition. The body says everything
    about scope — there is no separate scope field.

    `expr` must satisfy `assert_is_predicate`: i1-typed, side-effect-free,
    references resolve against the enclosing function. Stored in
    canonical form (see `quod.canonicalize`) so identical predicates
    produce identical hashes for proof pinning, dedup, and equivalence
    edges.

    The named claim kinds (`non_negative`, `int_range`, `return_in_range`)
    are CLI sugar that desugar to canonicalized PredicateClaim at parse
    time and resugar via a recognizer at render time. Sugar shapes are
    never stored.
    """
    kind: Literal["predicate"] = "predicate"
    expr: "Expr"


Claim = PredicateClaim


def claim_param(claim: PredicateClaim) -> str | None:
    """The sole parameter referenced by a claim's predicate, or None.

    Returns None if the predicate references zero params, multiple
    distinct params, or any `ReturnRef`. Used by introspection that
    only makes sense for single-param preconditions (e.g.
    `quod fn unconstrained`); callers that want all referenced params
    should walk `claim.expr` directly.
    """
    refs: set[str] = set()
    has_return = _collect_predicate_refs(claim.expr, refs)
    if has_return or len(refs) != 1:
        return None
    return next(iter(refs))


def _collect_predicate_refs(expr, refs: set[str]) -> bool:
    """Walk a predicate expression. Add each ParamRef name to `refs`;
    return True if any ReturnRef was encountered."""
    if isinstance(expr, ParamRef):
        refs.add(expr.name)
        return False
    if isinstance(expr, ReturnRef):
        return True
    if isinstance(expr, IntLit):
        return False
    if isinstance(expr, Not):
        return _collect_predicate_refs(expr.operand, refs)
    if isinstance(expr, BinOp):
        l_ret = _collect_predicate_refs(expr.lhs, refs)
        r_ret = _collect_predicate_refs(expr.rhs, refs)
        return l_ret or r_ret
    if isinstance(expr, (ShortCircuitOr, ShortCircuitAnd)):
        l_ret = _collect_predicate_refs(expr.lhs, refs)
        r_ret = _collect_predicate_refs(expr.rhs, refs)
        return l_ret or r_ret
    return False


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


# ---------- Top-level ----------

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
    body: "BlockOrScoped"
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


# ---------- Traits + impls ----------

class TraitMethodSig(_Node):
    """One method signature in a `TraitDef`. No body; impls supply the
    body. Param/return types may reference `SelfType` (the implementing
    type) and any `TypeParamRef`s declared by the trait itself
    (currently always empty — generic traits aren't yet supported)."""
    name: str
    params: tuple[Param, ...] = ()
    return_type: ReturnType


class TraitDef(_Node):
    """A trait: a name plus a set of method signatures any conforming
    `ImplDef` must provide. Pure declaration; no runtime cost.

    A `<T: TraitName>` bound on a generic type parameter constrains
    instantiations to types that have an `impl TraitName for ...`
    visible at mono time. The mono pass rewrites `TraitCall` nodes to
    direct `Call`s of the impl's mangled method symbols.
    """
    name: str
    methods: tuple[TraitMethodSig, ...]


def _substitute_self_in_type(t, for_type):
    """Eagerly substitute `SelfType` → `for_type` inside a Type tree.
    Used by `ImplDef`'s post-construction validator so impls store
    Self-free methods.
    """
    if isinstance(t, SelfType):
        return for_type
    if isinstance(t, StructType) and t.type_args:
        return t.model_copy(update={
            "type_args": tuple(_substitute_self_in_type(a, for_type) for a in t.type_args),
        })
    if isinstance(t, EnumType) and t.type_args:
        return t.model_copy(update={
            "type_args": tuple(_substitute_self_in_type(a, for_type) for a in t.type_args),
        })
    return t


class ImplDef(_Node):
    """`impl<...> <trait> for <for_type> { <methods> }`.

    Provides concrete bodies for the named trait's methods on the named
    type. Multiple impls of distinct traits for the same type are
    allowed; two impls of the same trait for the same type are not
    (coherence — checked at Program-level validation).

    On construction, every `SelfType` inside `methods` is rewritten to
    `for_type` — the lowerer and the mono pass never see Self.

    `type_params`: a generic impl introduces type variables that appear
    in `for_type` (e.g. `impl<T> Drop for Box<T>`). When the
    corresponding template (`Box`) is instantiated, the mono pass
    generates one concrete impl per instantiation by binding the
    impl's type-params from positions in `for_type.type_args`. Current
    restriction: each `for_type.type_args[i]` must be either a
    `TypeParamRef` naming one of the impl's `type_params`, or a
    concrete type — no nested patterns like `Box<List<T>>`.

    Methods are stored as full `Function`s so they can be promoted to
    top-level by the monomorphization pass with mangled names like
    `<for_type>::<method>` (e.g., `Arena::alloc`, `Box<i64>::drop`).
    """
    trait: str
    type_params: tuple[TypeParam, ...] = ()
    for_type: Type
    methods: tuple[Function, ...]

    @model_validator(mode="after")
    def _resolve_self(self) -> "ImplDef":
        # Eagerly substitute Self → for_type in every method's params,
        # return_type, AND body. After this validator runs, no SelfType
        # survives in the impl methods; the lowerer and the mono pass
        # never see Self.
        from .traversal import substitute_in_stmt
        type_fn = lambda t: _substitute_self_in_type(t, self.for_type)
        new_methods = []
        for fn in self.methods:
            new_params = tuple(
                Param(name=p.name, type=type_fn(p.type)) for p in fn.params
            )
            new_return = type_fn(fn.return_type)
            new_stmts = tuple(substitute_in_stmt(s, type_fn) for s in fn.body.stmts)
            new_body = fn.body.model_copy(update={"stmts": new_stmts})
            new_methods.append(fn.model_copy(update={
                "params":      new_params,
                "return_type": new_return,
                "body":        new_body,
            }))
        object.__setattr__(self, "methods", tuple(new_methods))
        return self


class WireBinding(_Node):
    """One `wire X=Y` clause inside an `Import`. Binds the wirable named
    `name` (declared at the imported module's scope) to `type`."""
    name: str
    type: Type


class Import(_Node):
    """A structured import. JSON form accepts either a bare string
    (`"alloc.list"`) or an object (`{"module": "alloc.list", "wire":
    [{"name": "A", "type": ...}]}`); a `field_validator` coerces
    strings to `{module: str}` so existing JSONs keep working.

    `wire` binds the imported module's wirables. The resolver
    substitutes those bindings throughout the module's body before
    merging it into the consumer.
    """
    module: str
    wire: tuple[WireBinding, ...] = ()

    @field_validator("wire", mode="before")
    @classmethod
    def _coerce_wire(cls, raw):
        # strict-mode Pydantic doesn't auto-coerce list→tuple; do it here
        # so JSON arrays parse cleanly.
        if isinstance(raw, list):
            return tuple(raw)
        return raw

    @model_serializer(mode="wrap")
    def _bare_string_when_no_wire(self, handler, info):
        # Round-trip optimization: if there's no wire, serialize as a
        # bare string. Existing programs without wirables stay byte-for-byte
        # identical on save.
        if not self.wire:
            return self.module
        data = handler(self)
        return data


class ProvenanceEdge(_Node):
    """An unkinded provenance edge: "this came from that," nothing more.

    `source` and `target` are stable node IDs (e.g. Function.id or
    Block.id). All semantic content for what the edge *means* lives in
    the `Equivalence` claims that anchor on the same IDs — the edge
    itself only records connectivity. N:M lowerings emit one
    ProvenanceEdge per (source, target) pair so the graph stays
    normalized.
    """
    kind: Literal["edge.provenance"] = "edge.provenance"
    source: str
    target: str


class Equivalence(_Node):
    """Program-level equivalence between two nodes by ID.

    Where ordinary claims live in `fn.claims` and constrain a single
    function's parameters or return value, an Equivalence is *relational*
    — it asserts that two nodes (typically across language-family layers,
    e.g. a layer-A `c.fn` and a layer-B `Function`, or a layer-B Block
    and a layer-C Block) compute the same value over a domain of inputs.

    The metadata fields (regime/enforcement/justification) mirror the
    `_Claim` shape so the existing claim plumbing — provers, the verify
    command, the stored-vs-derived discipline — extends uniformly.
    `domain` is the predicate over which the equivalence holds;
    currently always `None` (always-true). A real predicate domain
    (a `PredicateClaim`) is not yet supported.

    The two endpoints are symmetric — `~` is symmetric — but stored as
    `(a_node_id, b_node_id)` for stable JSON ordering. The `kind`
    discriminator stays `"equivalent_to"` to match the design doc's
    naming even though the program-level form is symmetric; the
    asymmetric `EquivalentTo(other_node_id)` form (a claim attached to
    a node that names its counterpart) is sugar over this for future
    authoring tools.
    """
    kind: Literal["equivalent_to"] = "equivalent_to"
    a_node_id: str
    b_node_id: str
    regime: Regime = "axiom"
    enforcement: Enforcement = "trust"
    justification: Justification | None = None
    # Currently always None (every equivalence is "true everywhere");
    # storing it keeps the JSON shape forward-compatible for when a real
    # predicate domain is introduced.
    domain: None = None

    @model_serializer(mode="wrap")
    def _drop_default_metadata(self, handler, info):
        data = handler(self)
        if self.regime == "axiom":
            data.pop("regime", None)
        if self.enforcement == "trust":
            data.pop("enforcement", None)
        if self.justification is None:
            data.pop("justification", None)
        if self.domain is None:
            data.pop("domain", None)
        return data


# ---------- Staged-lift: source-language and family-extension nodes ----------
#
# The graph has three layers:
#
#   Layer A — source language as authored (here: C). Lives under
#             `Program.source_units`. Inert (no validation, no codegen)
#             — exists so the original program is preserved as a
#             first-class subtree of the graph.
#   Layer B — core quod ∪ family extensions (here: `c.*`). The c-family
#             lowering pass produces this from layer A; `lower.py`
#             refuses to consume it.
#   Layer C — pure core quod. What `lower.py` and the proof tooling
#             operate on.


def _mint_node_id(prefix: str) -> str:
    """Mint an opaque node ID. Used as a `default_factory` so every new
    layer-A or c-extension node gets a stable ID at construction. The
    `prefix` is a short tag indicating the node kind (e.g. "cunit",
    "cfn", "cfor") — no semantic load, only useful for hand-debugging."""
    return f"@{prefix}_{uuid4().hex[:12]}"


# ----- Layer A: C source-language nodes -----
#
# Inert: no validation, no codegen, no semantic checks. Their job is to
# preserve the original C as a subtree of the program graph so
# provenance edges to lifted quod nodes have something to point at.
# The supported subset matches the existing C ingester (int-only, no
# structs/floats/switch), narrowed further to what `sum.c` exercises.


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
    Union[CVarDecl, CAssign],
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
    Union[CVarDecl, CMultiVarDecl, CAssign, CCompoundAssign, CReturn,
          CFor, CIf, CWhile, CDoWhile, CExprStmt, CBreak, CContinue,
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


# ----- Layer B: c.* extension nodes -----
#
# Constructs core quod can't represent on its own. Lowered to core by
# `lower/c_family.py`. `lower.py` refuses to consume them directly —
# the c-family pass must run first.


class CScopedBlock(_Node):
    """C-style block wrapper. `block` is the inner `core.Block` that
    edges anchor on; the wrapper carries family-specific scope semantics
    (which decls die at the closing brace). Lowered by c-family lowering
    to its inner block — the wrapper is discarded by the time `lower.py`
    sees the program.

    `scope_locals` records the names of locals whose scope ends with
    this block. Currently a tuple of names; richer scope metadata
    (types, kill points within the block) can accumulate as lowering
    rules grow.
    """
    kind: Literal["c.scoped_block"] = "c.scoped_block"
    id: str = Field(default_factory=lambda: _mint_node_id("cscope"))
    block: Block
    scope_locals: tuple[str, ...] = ()

    @model_serializer(mode="wrap")
    def _drop_default_metadata(self, handler, info):
        data = handler(self)
        if not self.scope_locals:
            data.pop("scope_locals", None)
        return data


# Body slots that may host either a plain `Block` (after c-family
# lowering, or for non-c-derived programs) or a `CScopedBlock` wrapper
# (in layer-B programs). Pydantic's smart union picks by structure:
# Block has no `kind` field; CScopedBlock's `kind` discriminates.
BlockOrScoped = Union[Block, CScopedBlock]


class CStyleFor(_Node):
    """`for (init; cond; inc) { body }` — the C for loop transcribed
    into layer B. Distinct from core's `For` (which is bounded
    iteration over a half-open integer range). The c-family lowering
    rule transforms `c.for_general` into `Let + While + Assign`; the
    rule's equivalence theorem is proved once and cited via
    `FamilyLowering("c.for_general")` justifications.
    """
    kind: Literal["c.for_general"] = "c.for_general"
    id: str = Field(default_factory=lambda: _mint_node_id("cfor_general"))
    init: "Statement | None" = None
    cond: "Expr | None" = None  # i1 when present
    inc: "Statement | None" = None
    body: BlockOrScoped


class _ProgramBase(_Node):
    """Shared shape for Program and InputProgram."""
    constants: tuple[StringConstant, ...] = ()
    functions: tuple[Function, ...] = ()
    externs: tuple[ExternFunction, ...] = ()
    structs: tuple[StructDef, ...] = ()
    enums: tuple[EnumDef, ...] = ()
    traits: tuple[TraitDef, ...] = ()
    impls: tuple[ImplDef, ...] = ()
    wirables: tuple[TypeParam, ...] = ()
    imports: tuple[Import, ...] = ()
    # Provenance edges and equivalence claims live at program level
    # (rather than on individual functions) because they're relational
    # — they connect two nodes that can be in different functions or in
    # different language-family layers. Both default to empty; existing
    # programs that don't yet carry provenance round-trip unchanged.
    edges: tuple[ProvenanceEdge, ...] = ()
    equivalences: tuple[Equivalence, ...] = ()
    # Layer-A subtree: original source-language programs preserved as
    # quod nodes (one CUnit per ingested file). Inert — no validation,
    # no codegen — but addressable by stable IDs so cross-layer
    # provenance edges can anchor here.
    source_units: tuple[CUnit, ...] = ()
    # Structured-form functions: per-language extension-bearing
    # transcriptions of the source. For C, these contain `CStyleFor`
    # and other `c.*` family extensions; the c-family lowering pass
    # (lower/c_family.py) reads these and produces the canonical core
    # functions in `Program.functions`. Both forms persist on disk so
    # cross-layer analysis and drift detection (lowering-rule changes)
    # work without re-ingesting from source. Empty for hand-authored
    # core programs that didn't go through a source-language ingest.
    #
    # The name is deliberately layer-neutral — the data model will
    # evolve as more language families land, and "structured" captures
    # the durable property (preserves source-language structure via
    # extension nodes) without committing to "layer B" terminology.
    structured_functions: tuple[Function, ...] = ()
    # Version stamp: which build of quod produced the pinned claims in
    # this Program. During R&D this is the commit hash of the quod
    # source tree at pin time; later it can be a release tag. `None`
    # means "no version on record" — under the strict policy this is
    # always treated as a mismatch by `quod equiv verify` etc., so a
    # `None` Program with pinned claims fails verification until
    # re-pinned from a clean checkout.
    #
    # Set by `quod.version.stamp_quod_version` at any operation that
    # produces or refreshes pinned claims (ingest's `prove_lifts`,
    # `equiv prove --bump`, etc.). Pinning from a dirty quod tree
    # captures `None` deliberately — only clean checkouts produce
    # verifiable pins.
    quod_version: str | None = None

    @model_serializer(mode="wrap")
    def _drop_empty_collections(self, handler, info):
        data = handler(self)
        if not self.structs:
            data.pop("structs", None)
        if not self.enums:
            data.pop("enums", None)
        if not self.traits:
            data.pop("traits", None)
        if not self.impls:
            data.pop("impls", None)
        if not self.wirables:
            data.pop("wirables", None)
        if not self.imports:
            data.pop("imports", None)
        if not self.edges:
            data.pop("edges", None)
        if not self.equivalences:
            data.pop("equivalences", None)
        if not self.source_units:
            data.pop("source_units", None)
        if not self.structured_functions:
            data.pop("structured_functions", None)
        if self.quod_version is None:
            data.pop("quod_version", None)
        return data

    @field_validator("imports", mode="before")
    @classmethod
    def _coerce_imports(cls, raw):
        """Allow `["alloc.list", {"module": "alloc.list", "wire": [...]}]`
        — bare strings become `{module: <s>}`. Returns a tuple to keep
        strict-mode validation happy (the field is `tuple[Import, ...]`
        and `strict=True` doesn't auto-coerce list→tuple).
        """
        out = []
        for item in raw:
            if isinstance(item, str):
                out.append({"module": item})
            else:
                out.append(item)
        return tuple(out)

    @field_validator("imports")
    @classmethod
    def _validate_import_names(cls, imports: tuple["Import", ...]) -> tuple["Import", ...]:
        # Sanitize: only allow [A-Za-z0-9_.] so module names can't
        # path-traverse to disk locations outside the stdlib directory.
        # Names map to file paths via `stdlib/<name>.json` — no slashes,
        # no leading/trailing dots, no empty segments.
        seen: set[str] = set()
        for imp in imports:
            n = imp.module
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
        return imports


def _validate_structs(program: "_ProgramBase") -> None:
    """Program-wide struct + enum *structural* sanity. Runs on both
    Program and InputProgram at construction time (Pydantic
    model_validator).

    Owns only the cheap, definition-local checks that don't need
    whole-program context:
    - Struct names are unique.
    - No duplicate field names within a struct.
    - Enum names are unique and don't collide with structs.
    - No duplicate variants per enum, no duplicate fields per variant.
    - Every enum has at least one variant.
    - No struct contains itself by value (direct or transitive).

    Use-site checks (refs resolve, exhaustive matches, struct/enum_init
    field correctness) live in `quod.validate` — they need the resolved
    program (post-import-resolution) and benefit from collecting all
    diagnostics rather than failing fast.
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
    enums_by_name: dict[str, EnumDef] = {}
    for ed in program.enums:
        if ed.name in enums_by_name:
            raise ValueError(f"duplicate enum definition {ed.name!r}")
        if ed.name in by_name:
            raise ValueError(
                f"enum name {ed.name!r} collides with a struct of the same name"
            )
        enums_by_name[ed.name] = ed
        seen_variants: set[str] = set()
        for v in ed.variants:
            if v.name in seen_variants:
                raise ValueError(
                    f"enum {ed.name!r} has duplicate variant {v.name!r}"
                )
            seen_variants.add(v.name)
            seen_fields: set[str] = set()
            for f in v.fields:
                if f.name in seen_fields:
                    raise ValueError(
                        f"variant {ed.name!r}::{v.name} has duplicate field {f.name!r}"
                    )
                seen_fields.add(f.name)
        if not ed.variants:
            raise ValueError(f"enum {ed.name!r} has no variants")

    # Reject by-value cycles. Walk each struct's transitive struct-typed
    # fields; a path that revisits the start is a cycle.
    for sd in program.structs:
        _check_no_struct_cycle(sd.name, by_name)


def _check_no_struct_cycle(start: str, by_name: dict[str, "StructDef"]) -> None:
    """DFS: refuse if `start` reaches itself through StructType fields.

    Skips StructType references with non-empty `type_args` — those are
    generic instantiations that resolve to a *different* nominal type
    post-monomorphization, so they don't form a cycle with the
    template. The post-mono cycle check (which validates the
    monomorphized program) catches real cycles between concrete types.
    """
    visiting: set[str] = set()

    def go(name: str, path: tuple[str, ...]) -> None:
        if name == start and path:
            chain = " -> ".join(path + (name,))
            raise ValueError(
                f"struct {start!r} contains itself by value (cycle: {chain}); "
                f"quod has no pointer-to-struct, so recursive structs are unrepresentable"
            )
        if name in visiting:
            return
        visiting.add(name)
        sd = by_name.get(name)
        if sd is None:
            return
        for f in sd.fields:
            if isinstance(f.type, StructType) and not f.type.type_args:
                go(f.type.name, path + (name,))
        visiting.discard(name)

    sd = by_name.get(start)
    if sd is None:
        return
    for f in sd.fields:
        if isinstance(f.type, StructType) and not f.type.type_args:
            go(f.type.name, (start,))


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
    InputProgram.model_validate(program.model_dump())
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


def add_claim(program: Program, function: str, claim: PredicateClaim) -> Program:
    """Append a claim to a function. Refuses duplicate predicates
    (canonical-equal) — relax first to overwrite."""
    fn = require_function(program, function)
    target = claim_param(claim)
    if target is not None and fn.param(target) is None:
        raise KeyError(f"function {function!r} has no parameter {target!r}")
    for existing in fn.claims:
        if existing.expr == claim.expr:
            raise ValueError(
                f"predicate already present on {function}; "
                f"relax it first if you need to change bounds"
            )
    new_fn = fn.model_copy(update={"claims": fn.claims + (claim,)})
    return replace_function(program, new_fn)


def add_extern_claim(program: Program, extern: str, claim: PredicateClaim) -> Program:
    """Append a claim to the named extern. Mirrors `add_claim`'s
    duplicate-check contract — re-adding the same predicate requires
    `relax` first."""
    new_externs = []
    found = False
    for ext in program.externs:
        if ext.name != extern:
            new_externs.append(ext)
            continue
        found = True
        for existing in ext.claims:
            if existing.expr == claim.expr:
                raise ValueError(
                    f"predicate already present on extern {extern}; "
                    f"relax it first if you need to change bounds"
                )
        new_externs.append(ext.model_copy(update={"claims": ext.claims + (claim,)}))
    if not found:
        raise KeyError(f"no extern named {extern!r}")
    return program.model_copy(update={"externs": tuple(new_externs)})


def relax_extern_claim(program: Program, extern: str, expr) -> Program:
    """Remove an extern claim by exact predicate match."""
    new_externs = []
    found_extern = False
    removed = False
    for ext in program.externs:
        if ext.name != extern:
            new_externs.append(ext)
            continue
        found_extern = True
        kept = tuple(c for c in ext.claims if c.expr != expr)
        if len(kept) != len(ext.claims):
            removed = True
        new_externs.append(ext.model_copy(update={"claims": kept}))
    if not found_extern:
        raise KeyError(f"no extern named {extern!r}")
    if not removed:
        raise KeyError(f"no matching predicate on extern {extern!r}")
    return program.model_copy(update={"externs": tuple(new_externs)})


def relax_claim(program: Program, function: str, expr) -> Program:
    """Remove a claim by exact predicate match (no-op disallowed:
    the predicate must exist)."""
    fn = require_function(program, function)
    kept = tuple(c for c in fn.claims if c.expr != expr)
    if len(kept) == len(fn.claims):
        raise KeyError(f"no matching predicate on {function!r}")
    new_fn = fn.model_copy(update={"claims": kept})
    return replace_function(program, new_fn)


# ---------- Pretty-printer ----------

# A "label" is an optional prefix the formatter inserts before each addressable
# node. The default is empty; the CLI passes a function returning [hashprefix]
# so each node prints with its addressable identity inline.
NodeLabel = Callable[[_Node], str]
_NO_LABEL: NodeLabel = lambda _node: ""


def _format_import(imp: Import) -> str:
    s = imp.module
    if imp.wire:
        bindings = ", ".join(f"{w.name}={_format_type(w.type)}" for w in imp.wire)
        s += f" wire {bindings}"
    return s


def _format_type_param(tp: TypeParam) -> str:
    s = tp.name
    if tp.bound:
        s += f": {tp.bound}"
    return s


def _format_type_params(tps: tuple) -> str:
    if not tps:
        return ""
    return "<" + ", ".join(_format_type_param(tp) for tp in tps) + ">"


def format_program(program: Program, *, label: NodeLabel = _NO_LABEL) -> str:
    lines: list[str] = ["program {"]
    if program.source_units:
        lines.append("  source_units:")
        for unit in program.source_units:
            lines.extend("    " + line for line in _format_c_unit(unit, label=label).splitlines())
    if program.wirables:
        lines.append("  wirables:")
        for w in program.wirables:
            lines.append(f"    {_format_type_param(w)}")
    if program.imports:
        lines.append("  imports:")
        for imp in program.imports:
            lines.append(f"    {_format_import(imp)}")
    if program.constants:
        lines.append("  constants:")
        for c in program.constants:
            lines.append(f"    {label(c)}{c.name} = {c.value!r}")
    if program.structs:
        lines.append("  structs:")
        for sd in program.structs:
            lines.append(f"    {label(sd)}{format_struct_def(sd)}")
    if program.enums:
        lines.append("  enums:")
        for ed in program.enums:
            lines.extend("    " + line for line in format_enum_def(ed, label=label).splitlines())
    if program.externs:
        lines.append("  externs:")
        for ext in program.externs:
            sig_parts = [_format_type(t) for t in ext.effective_param_types()]
            if ext.varargs:
                sig_parts.append("...")
            sig = ", ".join(sig_parts)
            ret = _format_type(ext.return_type)
            lines.append(f"    {label(ext)}extern {ext.name}({sig}) -> {ret}")
    if program.structured_functions:
        lines.append("  structured_functions:")
        for fn in program.structured_functions:
            lines.extend("    " + line for line in format_function(fn, label=label).splitlines())
    if program.functions:
        lines.append("  functions:")
        for fn in program.functions:
            lines.extend("    " + line for line in format_function(fn, label=label).splitlines())
    if program.edges:
        lines.append("  edges:")
        for e in program.edges:
            lines.append(f"    {e.source} -> {e.target}")
    if program.equivalences:
        lines.append("  equivalences:")
        for eq in program.equivalences:
            lines.append(f"    {eq.a_node_id} ~ {eq.b_node_id}{format_equivalence_metadata(eq)}")
    if (
        not program.constants and not program.functions
        and not program.externs and not program.structs and not program.enums
        and not program.imports and not program.edges
        and not program.equivalences and not program.source_units
        and not program.structured_functions
    ):
        lines.append("  (empty)")
    lines.append("}")
    return "\n".join(lines)


def _format_c_unit(unit: "CUnit", *, label: NodeLabel) -> str:
    """Render a layer-A C translation unit. Surface form approximates C
    source so the unit reads naturally in `quod show` output; no
    semantic processing — just structural pretty-printing."""
    head = f'{label(unit)}c_unit "{unit.source_path}" {{'
    lines = [head]
    for fn in unit.functions:
        lines.extend("  " + line for line in format_c_fn(fn, label=label).splitlines())
    lines.append("}")
    return "\n".join(lines)


def format_c_fn(fn: "CFn", *, label: NodeLabel = _NO_LABEL) -> str:
    """Render a layer-A `CFn` as C-flavored text. Used by
    `format_program` for the `source_units` section and by the CLI's
    `quod fn show --source` for single-function rendering."""
    params = ", ".join(f"{_format_c_type(p.type)} {p.name}" for p in fn.params)
    head = f"{label(fn)}{_format_c_type(fn.return_type)} {fn.name}({params}) {{"
    lines = [head]
    for s in fn.body:
        lines.append(_format_c_stmt(s, indent=2, label=label))
    lines.append("}")
    return "\n".join(lines)


def _format_c_type(t) -> str:
    match t:
        case CNamedType(name=n):
            return n
        case CPointerType(pointee=p):
            return f"{_format_c_type(p)}*"
    raise ValueError(f"unhandled c.* type: {t!r}")


def _format_c_stmt(stmt, indent: int, *, label: NodeLabel) -> str:
    pad = " " * indent
    prefix = label(stmt)
    match stmt:
        case CVarDecl(type=ty, name=n, init=None):
            return f"{pad}{prefix}{_format_c_type(ty)} {n};"
        case CVarDecl(type=ty, name=n, init=init):
            return f"{pad}{prefix}{_format_c_type(ty)} {n} = {_format_c_expr(init)};"
        case CAssign(target=t, value=v):
            return f"{pad}{prefix}{t} = {_format_c_expr(v)};"
        case CReturn(value=None):
            return f"{pad}{prefix}return;"
        case CReturn(value=v):
            return f"{pad}{prefix}return {_format_c_expr(v)};"
        case CFor(init=init, cond=cond, inc=inc, body=body):
            init_s = _format_c_for_part(init) if init is not None else ""
            cond_s = _format_c_expr(cond) if cond is not None else ""
            inc_s = _format_c_for_part(inc) if inc is not None else ""
            head = f"{pad}{prefix}for ({init_s}; {cond_s}; {inc_s}) {{"
            body_lines = "\n".join(_format_c_stmt(s, indent + 2, label=label) for s in body)
            return f"{head}\n{body_lines}\n{pad}}}"
        case CIf(cond=cond, then_body=tb, else_body=eb):
            head = f"{pad}{prefix}if ({_format_c_expr(cond)}) {{"
            then_lines = "\n".join(_format_c_stmt(s, indent + 2, label=label) for s in tb)
            if not eb:
                return f"{head}\n{then_lines}\n{pad}}}"
            else_lines = "\n".join(_format_c_stmt(s, indent + 2, label=label) for s in eb)
            return f"{head}\n{then_lines}\n{pad}}} else {{\n{else_lines}\n{pad}}}"
        case CWhile(cond=cond, body=body):
            head = f"{pad}{prefix}while ({_format_c_expr(cond)}) {{"
            body_lines = "\n".join(_format_c_stmt(s, indent + 2, label=label) for s in body)
            return f"{head}\n{body_lines}\n{pad}}}"
        case CExprStmt(value=v):
            return f"{pad}{prefix}{_format_c_expr(v)};"
    raise ValueError(f"unhandled c.* statement: {stmt!r}")


def _format_c_for_part(s) -> str:
    """Render a CForInit (CVarDecl or CAssign) inline — no trailing
    semicolon, since the for-header supplies its own."""
    match s:
        case CVarDecl(type=ty, name=n, init=None):
            return f"{_format_c_type(ty)} {n}"
        case CVarDecl(type=ty, name=n, init=init):
            return f"{_format_c_type(ty)} {n} = {_format_c_expr(init)}"
        case CAssign(target=t, value=v):
            return f"{t} = {_format_c_expr(v)}"
    raise ValueError(f"unhandled c.* for-part: {s!r}")


def _format_c_expr(e) -> str:
    match e:
        case CIntLit(value=v):
            return str(v)
        case CStringLit(value=v):
            return repr(v)
        case CVarRef(name=n):
            return n
        case CEnumConstRef(name=n, value=v):
            return f"{n}={v}"
        case CBinOp(op=op, lhs=l, rhs=r):
            return f"({_format_c_expr(l)} {op} {_format_c_expr(r)})"
        case CCall(callee=callee, args=args):
            args_s = ", ".join(_format_c_expr(a) for a in args)
            return f"{callee}({args_s})"
        case CArraySubscript(base=b, index=i):
            return f"{_format_c_expr(b)}[{_format_c_expr(i)}]"
        case CAddressOf(target=t):
            return f"&{_format_c_expr(t)}"
    raise ValueError(f"unhandled c.* expression: {e!r}")


def format_equivalence_metadata(eq: "Equivalence") -> str:
    """Render the regime/enforcement/justification metadata trailing an
    equivalence, mirroring `format_claim_metadata` for plain claims.
    Returns " {…}" with a leading space when any non-default field is
    present, or "" when everything is at default."""
    bits: list[str] = []
    if eq.regime != "axiom":
        bits.append(f"regime={eq.regime}")
    if eq.enforcement != "trust":
        bits.append(f"enforcement={eq.enforcement}")
    if eq.justification is not None:
        bits.append(f"justification={_format_justification(eq.justification)}")
    return " {" + ", ".join(bits) + "}" if bits else ""


def format_struct_def(sd: StructDef) -> str:
    tp = _format_type_params(sd.type_params)
    body = ", ".join(f"{f.name}: {_format_type(f.type)}" for f in sd.fields)
    return f"struct {sd.name}{tp} {{ {body} }}"


def format_enum_def(ed: EnumDef, *, label: NodeLabel = _NO_LABEL) -> str:
    lines = [f"{label(ed)}enum {ed.name} {{"]
    for v in ed.variants:
        if v.fields:
            args = ", ".join(f"{f.name}: {_format_type(f.type)}" for f in v.fields)
            lines.append(f"  {label(v)}{v.name}({args})")
        else:
            lines.append(f"  {label(v)}{v.name}")
    lines.append("}")
    return "\n".join(lines)


def format_function(fn: Function, *, label: NodeLabel = _NO_LABEL) -> str:
    tp = _format_type_params(fn.type_params)
    sig_params = ", ".join(f"{p.name}: {_format_type(p.type)}" for p in fn.params)
    header = f"{label(fn)}{fn.name}{tp}({sig_params}) -> {_format_type(fn.return_type)}"
    if fn.claims:
        header += "  [claims: " + ", ".join(format_claim(c) for c in fn.claims) + "]"
    lines: list[str] = []
    for note in fn.notes:
        lines.append(f"// {note}")
    lines.append(header + " {")
    for s in fn.body.stmts:
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
        case StructType(name=n, type_args=ta) if ta:
            args = ", ".join(_format_type(a) for a in ta)
            return f"{n}<{args}>"
        case StructType(name=n):
            return n
        case EnumType(name=n, type_args=ta) if ta:
            args = ", ".join(_format_type(a) for a in ta)
            return f"{n}<{args}>"
        case EnumType(name=n):
            return n
        case TypeParamRef(name=n):
            return n
        case SelfType():
            return "Self"
        case VoidType():
            return "void"
    raise ValueError(f"unhandled type: {t!r}")


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
        case LiftEquivalence(artifact_path=p, artifact_hash=h):
            return f"lift_equivalence({p}@{h[:12]})"
        case FamilyLowering(rule_name=r, artifact_hash=h):
            tail = f"@{h[:12]}" if h else ""
            return f"family_lowering({r}{tail})"
    raise ValueError(f"unhandled justification: {j!r}")


def _format_stmt(stmt, indent: int, *, label: NodeLabel) -> str:
    pad = " " * indent
    prefix = label(stmt)
    match stmt:
        case ReturnExpr(value=expr):
            return f"{pad}{prefix}return {_format_expr(expr)}"
        case Return():
            return f"{pad}{prefix}return"
        case Unreachable():
            return f"{pad}{prefix}unreachable"
        case If(cond=cond, then_body=t_body, else_body=e_body):
            then_lines = "\n".join(_format_stmt(s, indent + 2, label=label) for s in t_body.stmts)
            head = f"{pad}{prefix}if ({_format_expr(cond)}) {{"
            if not e_body.stmts:
                return f"{head}\n{then_lines}\n{pad}}}"
            else_lines = "\n".join(_format_stmt(s, indent + 2, label=label) for s in e_body.stmts)
            return f"{head}\n{then_lines}\n{pad}}} else {{\n{else_lines}\n{pad}}}"
        case Let(name=n, type=ty, init=init):
            return f"{pad}{prefix}let {n}: {_format_type(ty)} = {_format_expr(init)}"
        case Assign(name=n, value=v):
            return f"{pad}{prefix}{n} = {_format_expr(v)}"
        case FieldSet(local=loc, name=fname, value=v):
            return f"{pad}{prefix}{loc}.{fname} = {_format_expr(v)}"
        case Store(ptr=p, value=v):
            return f"{pad}{prefix}store({_format_expr(p)}, {_format_expr(v)})"
        case StoreField(ptr=p, struct_type=tname, name=fname, value=v):
            return f"{pad}{prefix}{_format_expr(p)}->{tname}.{fname} = {_format_expr(v)}"
        case While(cond=cond, body=body):
            body_lines = "\n".join(_format_stmt(s, indent + 2, label=label) for s in body.stmts)
            return f"{pad}{prefix}while ({_format_expr(cond)}) {{\n{body_lines}\n{pad}}}"
        case For(var=var, type=ty, lo=lo, hi=hi, body=body):
            body_lines = "\n".join(_format_stmt(s, indent + 2, label=label) for s in body.stmts)
            return (
                f"{pad}{prefix}for {var}: {_format_type(ty)} in "
                f"{_format_expr(lo)}..{_format_expr(hi)} {{\n"
                f"{body_lines}\n{pad}}}"
            )
        case ExprStmt(value=v):
            return f"{pad}{prefix}{_format_expr(v)}"
        case WithArena(name=n, capacity=cap, body=body):
            body_lines = "\n".join(_format_stmt(s, indent + 2, label=label) for s in body.stmts)
            return (
                f"{pad}{prefix}with_arena {n} = arena_new({_format_expr(cap)}) {{\n"
                f"{body_lines}\n{pad}}}"
            )
        case Match(scrutinee=scrut, arms=arms):
            arm_lines = []
            for arm in arms:
                arm_pad = " " * (indent + 2)
                inner_pad = " " * (indent + 4)
                if arm.bindings:
                    head = f"{arm_pad}{arm.variant}({', '.join(arm.bindings)}) => {{"
                else:
                    head = f"{arm_pad}{arm.variant} => {{"
                arm_lines.append(head)
                for s in arm.body.stmts:
                    arm_lines.append(_format_stmt(s, indent + 4, label=label))
                arm_lines.append(f"{arm_pad}}}")
            arms_text = "\n".join(arm_lines)
            return f"{pad}{prefix}match {_format_expr(scrut)} {{\n{arms_text}\n{pad}}}"
        case CStyleFor(init=init, cond=cond, inc=inc, body=body):
            init_s = _format_stmt(init, indent=0, label=label).strip().rstrip(";") if init is not None else ""
            cond_s = _format_expr(cond) if cond is not None else ""
            inc_s = _format_stmt(inc, indent=0, label=label).strip().rstrip(";") if inc is not None else ""
            inner = body if isinstance(body, Block) else body.block
            body_lines = "\n".join(_format_stmt(s, indent + 2, label=label) for s in inner.stmts)
            return (
                f"{pad}{prefix}c.for_general ({init_s}; {cond_s}; {inc_s}) {{\n"
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
        case LoadField(ptr=p, struct_type=tname, name=fname):
            return f"{_format_expr(p)}->{tname}.{fname}"
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
        case NullPtr():
            return "null"
        case CharLit(value=v):
            return repr(v)
        case SizeOf(type=t):
            return f"sizeof[{_format_type(t)}]"
        case TryExpr(value=v):
            return f"{_format_expr(v)}?"
        case EnumInit(enum=ename, variant=vname, fields=field_inits):
            if field_inits:
                inner = ", ".join(f"{fi.name}: {_format_expr(fi.value)}" for fi in field_inits)
                return f"{ename}::{vname}({inner})"
            return f"{ename}::{vname}"
    raise ValueError(f"unhandled expr: {expr!r}")
