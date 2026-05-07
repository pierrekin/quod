"""Expression nodes — leaves of the value-bearing graph.

Includes the StringConstant since it sits beside the string-related
expression nodes (StringRef, EnumInit fields, etc.) and is referenced
from program-level only as an interned constant.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, field_validator, model_serializer

from quod.model.base import _Node
from quod.model.types import FloatType, IntType, Type


class StringConstant(_Node):
    name: str
    value: str


class IntLit(_Node):
    kind: Literal["llvm.const_int"] = "llvm.const_int"
    type: "IntType"
    value: int


class FloatLit(_Node):
    """IEEE 754 floating-point literal stored as its bit pattern.

    `bits` is the exact IEEE 754 encoding for `type` — uint32 for f32,
    uint64 for f64. Storing the bit pattern (rather than a Python float
    which is always f64) eliminates the silent double-rounding hazard
    that arises when an f32 value is round-tripped through f64: the
    decimal `0.1f` parses to one f32 bit pattern via direct decimal→f32
    rounding (what a C compiler does), and a *different* f32 bit
    pattern via decimal→f64→f32 (double rounding). Bit storage means
    the literal carries exactly what we'll emit.

    Special values (`+inf`, `-inf`, NaN, and NaN-with-payload) are
    representable as ordinary bit patterns — `0x7F800000` for `+inf`
    f32, `0x7FC00000` for canonical-NaN f32, etc. The
    `quod.model.float_bits_for_special` helper returns the canonical
    bit patterns by name.

    Use `python_float_to_bits` and `bits_to_python_float` (also in
    `quod.model`) to convert at boundaries.
    """
    kind: Literal["quod.float_lit"] = "quod.float_lit"
    type: "FloatType"
    bits: int

    def model_post_init(self, __context) -> None:
        from quod.model.types import F32Type, F64Type
        if isinstance(self.type, F32Type):
            width = 32
        elif isinstance(self.type, F64Type):
            width = 64
        else:
            raise ValueError(f"FloatLit type must be F32Type or F64Type; got {self.type!r}")
        if self.bits < 0:
            raise ValueError(
                f"FloatLit bits must be a non-negative IEEE 754 encoding; "
                f"got {self.bits}"
            )
        if self.bits >= (1 << width):
            raise ValueError(
                f"FloatLit bits 0x{self.bits:x} does not fit in {width} bits "
                f"(type {self.type.kind})"
            )


class FNeg(_Node):
    """IEEE 754 unary negation: flip the sign bit unconditionally,
    including for NaN and `-0.0`.

    Lowers to LLVM `fneg`. Cannot be folded to
    `BinOp(fsub, FloatLit(0.0), x)` because that returns `+0.0` for
    `-0.0` input — fsub follows the IEEE rule that
    `0.0 - (-0.0) = +0.0`. `fneg` flips the sign bit directly.
    """
    kind: Literal["quod.fneg"] = "quod.fneg"
    operand: "Expr"


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
      f-arith   — fadd, fsub, fmul, fdiv, frem     : fN in / fN out
      f-cmp     — feq, fne, flt, fle, fgt, fge     : fN in / i1 out

    Operands of arith/bitwise/cmp must have the same type; LLVM's verifier
    enforces this at lower time. The signed/unsigned distinction matches
    LLVM IR predicates — signedness lives on the op, not the type. Division
    by zero is undefined behaviour (matches LLVM); guard with an int_range
    or runtime branch if the divisor isn't statically nonzero.

    Shifts: `shl` is logical left, `ashr` is arithmetic (sign-extending)
    right, `lshr` is logical (zero-extending) right. Shift count must
    have the same iN type as the value; shift count >= bitwidth is
    undefined behaviour (matches LLVM).

    Float ops follow strict IEEE 754. `fne` lowers to LLVM `une` so
    `NaN != NaN` returns true; the magnitude comparisons (`flt`/`fle`/
    `fgt`/`fge`) and `feq` lower to ordered preds, which return false
    when either operand is NaN. `frem` is `fmod`-like (LLVM `frem`).
    No FMA contraction; no fenv access.

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
        # Float arithmetic — both operands same float type; result is operand type.
        "fadd", "fsub", "fmul", "fdiv", "frem",
        # Float comparison — both operands same float type; result is i1.
        # Lower to LLVM ordered preds (false if either is NaN) plus `une`
        # for fne so `NaN != NaN` returns true. Rust + C agree on this
        # mapping.
        "feq", "fne", "flt", "fle", "fgt", "fge",
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


class Cast(_Node):
    """Cast a numeric value to a different numeric type.

    The (source-type, target-type) pair determines the LLVM operation
    uniquely — quod's IXType / UXType partition encodes signedness on
    the type, so no separate `signed` flag is needed:

    - int → int wider, signed source: `sext`
    - int → int wider, unsigned source: `zext`
    - int → int narrower (signed or unsigned): `trunc`
    - int → int same width, signedness reinterpret: identity (LLVM no-op)
    - int → float, signed source: `sitofp`
    - int → float, unsigned source: `uitofp`
    - float → int, signed target: `llvm.fptosi.sat.iN.fM`
    - float → int, unsigned target: `llvm.fptoui.sat.iN.fM`
    - float → float wider: `fpext`
    - float → float narrower: `fptrunc`
    - same source and target: identity

    Float-to-int saturates out-of-range values to the target's MAX/MIN;
    NaN-to-int is 0. To zero-extend a *signed* int (e.g. an i32 read
    that should not be sign-extended), reinterpret-cast first to the
    same-width unsigned type, then widen — two casts, but explicit
    intent.

    Non-numeric source/target types (struct, enum, pointer, void) are
    rejected by the validator.
    """
    kind: Literal["quod.cast"] = "quod.cast"
    value: "Expr"
    target_type: "Type"


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
        IntLit, FloatLit, FNeg,
        ParamRef, LocalRef, BinOp, ShortCircuitOr, ShortCircuitAnd,
        IfExpr, Not, ReturnRef,
        Call, StringRef, FieldRead, LoadField, StructInit, PtrOffset, Cast,
        Load, NullPtr, CharLit, EnumInit, SizeOf, TryExpr, TraitCall,
    ],
    Field(discriminator="kind"),
]


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
