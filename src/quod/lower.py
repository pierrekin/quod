"""Lowering: Program -> LLVM IR -> object/binary, plus the optimization pass.

Two-pass over functions: first declare every user function (so user-level
calls in expressions resolve regardless of definition order), then lower
each function body.

Locals are alloca'd at the function's entry block (the canonical mem2reg
shape). Loops, ExprStmt, short-circuit booleans, and fall-through Ifs
each get their own basic-block layout.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from llvmlite import binding as llvm
from llvmlite import ir

from quod.analysis import derive_lattice_claims, elaborate
from quod.runtime import build_runtime_archive
from quod.stdlib import resolve_imports
from quod.model import (
    Assign,
    BinOp,
    Call,
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
    If,
    IntLit,
    IntRangeClaim,
    int_type_width,
    Let,
    CharLit,
    Load,
    LocalRef,
    Match,
    NonNegativeClaim,
    NullPtr,
    Return,
    ParamRef,
    Program,
    PtrOffset,
    ReturnExpr,
    SizeOf,
    ReturnInRangeClaim,
    ShortCircuitAnd,
    ShortCircuitOr,
    StringRef,
    Store,
    StructDef,
    StructInit,
    StructType,
    VoidType,
    While,
    Widen,
    WithArena,
)


I1 = ir.IntType(1)
I8 = ir.IntType(8)
I16 = ir.IntType(16)
I32 = ir.IntType(32)
I64 = ir.IntType(64)


def _type_to_llvm(
    t,
    struct_tys: dict[str, "ir.IdentifiedStructType"] | None = None,
    enum_tys: dict[str, "ir.IdentifiedStructType"] | None = None,
):
    """Lower a quod type to its LLVM equivalent.

    `struct_tys` is the per-module registry of identified struct types,
    threaded through every site that lowers a Type. None is allowed for
    legacy callers that operate on int-only contexts; passing None when
    the type IS a struct raises.

    `enum_tys` is the parallel registry for enums. Same None semantics.
    """
    match t:
        case I1Type():
            return I1
        case I8Type():
            return I8
        case I16Type():
            return I16
        case I32Type():
            return I32
        case I64Type():
            return I64
        case I8PtrType():
            return I8.as_pointer()
        case StructType(name=name):
            if struct_tys is None or name not in struct_tys:
                raise ValueError(f"struct type {name!r} not registered with the module")
            return struct_tys[name]
        case EnumType(name=name):
            if enum_tys is None or name not in enum_tys:
                raise ValueError(f"enum type {name!r} not registered with the module")
            return enum_tys[name]
        case VoidType():
            return ir.VoidType()
    raise ValueError(f"unhandled quod.Type: {t!r}")


# ---------- Lowering helpers ----------

def _get_or_declare_assume(module: ir.Module) -> ir.Function:
    if "llvm.assume" in module.globals:
        return module.globals["llvm.assume"]
    return ir.Function(module, ir.FunctionType(ir.VoidType(), [I1]), name="llvm.assume")


def _get_or_declare_abort(module: ir.Module) -> ir.Function:
    """libc abort(). Used by enforcement=verify claims when the predicate is false."""
    if "abort" in module.globals:
        return module.globals["abort"]
    fn = ir.Function(module, ir.FunctionType(ir.VoidType(), []), name="abort")
    fn.attributes.add("noreturn")
    return fn


def _get_or_declare_atoll(module: ir.Module) -> ir.Function:
    """libc atoll(const char*) -> long long. Used by the argv-parsing main
    wrapper to decode each argv slot to i64; the wrapper then trunc/sext's
    to whichever integer width the entry function's param actually takes."""
    if "atoll" in module.globals:
        return module.globals["atoll"]
    return ir.Function(module, ir.FunctionType(I64, [I8.as_pointer()]), name="atoll")


def _emit_for_enforcement(builder: ir.IRBuilder, cond: ir.Value, enforcement: str, llvm_fn: ir.Function, module: ir.Module) -> None:
    """Lower a single boolean predicate per the claim's effective enforcement.

    trust:  emit llvm.assume; falsity is undefined behaviour.
    verify: branch on the predicate; the false side calls abort + unreachable.
            The optimizer learns the predicate is true on the ok side, so
            downstream code still benefits — we get assertion + propagation.
    """
    if enforcement == "trust":
        builder.call(_get_or_declare_assume(module), [cond])
        return
    if enforcement == "verify":
        ok = llvm_fn.append_basic_block("check_ok")
        fail = llvm_fn.append_basic_block("check_fail")
        builder.cbranch(cond, ok, fail)
        builder.position_at_end(fail)
        builder.call(_get_or_declare_abort(module), [])
        builder.unreachable()
        builder.position_at_end(ok)
        return
    raise ValueError(f"unknown enforcement: {enforcement!r}")


# Map quod.BinOp.op -> the icmp predicate (cmp ops only).
_ICMP_SIGNED = {
    "slt": "<", "sle": "<=", "sgt": ">", "sge": ">=",
    "eq": "==", "ne": "!=",
}
_ICMP_UNSIGNED = {
    "ult": "<", "ule": "<=", "ugt": ">", "uge": ">=",
}


def _lower_expr(
    builder: ir.IRBuilder, expr, params: dict[str, ir.Value], module: ir.Module,
    *, constants: dict[str, ir.GlobalVariable], extern_sigs: dict[str, ExternFunction],
    locals_: dict[str, ir.AllocaInstr],
    struct_defs: dict[str, StructDef],
    struct_tys: dict[str, "ir.IdentifiedStructType"],
    enum_defs: dict[str, EnumDef],
    enum_tys: dict[str, "ir.IdentifiedStructType"],
) -> ir.Value:
    def go(e):
        return _lower_expr(
            builder, e, params, module,
            constants=constants, extern_sigs=extern_sigs, locals_=locals_,
            struct_defs=struct_defs, struct_tys=struct_tys,
            enum_defs=enum_defs, enum_tys=enum_tys,
        )

    match expr:
        case IntLit(type=t, value=v):
            return ir.Constant(_type_to_llvm(t), v)
        case ParamRef(name=n):
            return params[n]
        case LocalRef(name=n):
            if n not in locals_:
                raise ValueError(f"reference to undeclared local {n!r}")
            return builder.load(locals_[n])
        case BinOp(op="add", lhs=l, rhs=r):
            lv = go(l)
            rv = go(r)
            # Ptr-on-LHS pointer arithmetic: `(p + i64)` desugars to an
            # inbounds GEP. Mirrors what `quod.ptr_offset` does explicitly,
            # but lets straight-line script `(p + n)` work without forcing
            # the user to reach for the explicit form.
            if (
                isinstance(lv.type, ir.PointerType) and lv.type.pointee == I8
                and isinstance(rv.type, ir.IntType) and rv.type.width == 64
            ):
                return builder.gep(lv, [rv], inbounds=True)
            return builder.add(lv, rv)
        case BinOp(op="sub", lhs=l, rhs=r):
            return builder.sub(go(l), go(r))
        case BinOp(op="mul", lhs=l, rhs=r):
            return builder.mul(go(l), go(r))
        case BinOp(op="sdiv", lhs=l, rhs=r):
            return builder.sdiv(go(l), go(r))
        case BinOp(op="udiv", lhs=l, rhs=r):
            return builder.udiv(go(l), go(r))
        case BinOp(op="srem", lhs=l, rhs=r):
            return builder.srem(go(l), go(r))
        case BinOp(op=op, lhs=l, rhs=r) if op in _ICMP_SIGNED:
            return builder.icmp_signed(_ICMP_SIGNED[op], go(l), go(r))
        case BinOp(op=op, lhs=l, rhs=r) if op in _ICMP_UNSIGNED:
            return builder.icmp_unsigned(_ICMP_UNSIGNED[op], go(l), go(r))
        case BinOp(op="or", lhs=l, rhs=r):
            return builder.or_(go(l), go(r))
        case BinOp(op="and", lhs=l, rhs=r):
            return builder.and_(go(l), go(r))
        case ShortCircuitOr(lhs=l, rhs=r):
            return _lower_short_circuit(builder, l, r, kind="or", lower=go)
        case ShortCircuitAnd(lhs=l, rhs=r):
            return _lower_short_circuit(builder, l, r, kind="and", lower=go)
        case StringRef(name=n):
            gv = constants[n]
            return builder.bitcast(gv, I8.as_pointer())
        case Call(function=fname, args=args):
            callee = module.globals.get(fname)
            if callee is None:
                raise ValueError(f"call to undeclared function {fname!r}")
            # Coerce bare int literals at each fixed (non-vararg) parameter
            # position to the callee's declared type. Vararg slots (printf
            # etc.) keep their declared types since the callee has no
            # narrower type to coerce them to.
            param_tys = callee.function_type.args
            coerced: list = []
            for i, a in enumerate(args):
                if i < len(param_tys):
                    coerced.append(_coerce_int_lit(a, param_tys[i]))
                else:
                    coerced.append(a)
            arg_vals = [go(a) for a in coerced]
            return builder.call(callee, arg_vals)
        case StructInit(type=tname, fields=field_inits):
            sd = struct_defs.get(tname)
            sty = struct_tys.get(tname)
            if sd is None or sty is None:
                raise ValueError(f"struct_init for undefined struct {tname!r}")
            init_by_name = {fi.name: fi.value for fi in field_inits}
            val: ir.Value = ir.Constant(sty, ir.Undefined)
            for i, f in enumerate(sd.fields):
                if f.name not in init_by_name:
                    raise ValueError(
                        f"struct_init for {tname!r} missing field {f.name!r}"
                    )
                field_dest_ty = _type_to_llvm(f.type, struct_tys, enum_tys)
                coerced = _coerce_int_lit(init_by_name[f.name], field_dest_ty)
                val = builder.insert_value(val, go(coerced), i)
            return val
        case FieldRead(value=inner, name=fname):
            inner_val = go(inner)
            inner_ty = inner_val.type
            if not isinstance(inner_ty, ir.IdentifiedStructType):
                raise ValueError(
                    f"field read {fname!r} on non-struct value of type {inner_ty}"
                )
            sd = struct_defs.get(inner_ty.name)
            if sd is None:
                raise ValueError(f"field read on unknown struct {inner_ty.name!r}")
            idx = sd.field_index(fname)
            return builder.extract_value(inner_val, idx)
        case PtrOffset(base=b, offset=o):
            base_val = go(b)
            off_val = go(o)
            if not (isinstance(base_val.type, ir.PointerType) and base_val.type.pointee == I8):
                raise ValueError(
                    f"ptr_offset base must be i8*, got {base_val.type}"
                )
            if not (isinstance(off_val.type, ir.IntType) and off_val.type.width == 64):
                raise ValueError(
                    f"ptr_offset offset must be i64, got {off_val.type}"
                )
            return builder.gep(base_val, [off_val], inbounds=True)
        case Widen(value=v, target=t, signed=signed):
            val = go(v)
            target_ty = _type_to_llvm(t)
            if not isinstance(val.type, ir.IntType):
                raise ValueError(f"widen source must be an integer, got {val.type}")
            if not isinstance(target_ty, ir.IntType):
                raise ValueError(f"widen target must be an int type, got {t!r}")
            src_w, dst_w = val.type.width, target_ty.width
            if src_w == dst_w:
                return val
            if src_w < dst_w:
                return builder.sext(val, target_ty) if signed else builder.zext(val, target_ty)
            return builder.trunc(val, target_ty)
        case Load(ptr=p, type=t):
            base = go(p)
            if not (isinstance(base.type, ir.PointerType) and base.type.pointee == I8):
                raise ValueError(f"load base must be i8*, got {base.type}")
            target_ty = _type_to_llvm(t, struct_tys, enum_tys)
            casted = builder.bitcast(base, target_ty.as_pointer())
            return builder.load(casted)
        case NullPtr():
            return ir.Constant(I8.as_pointer(), None)
        case CharLit(value=v):
            return ir.Constant(I8, ord(v))
        case SizeOf(type=t):
            size, _align = _size_of_quod_type(t, struct_defs, enum_defs)
            return ir.Constant(I64, size)
        case EnumInit(enum=ename, variant=vname, fields=field_inits):
            ed = enum_defs.get(ename)
            ety = enum_tys.get(ename)
            if ed is None or ety is None:
                raise ValueError(f"enum_init for undefined enum {ename!r}")
            var = ed.variant(vname)
            if var is None:
                raise ValueError(f"enum_init: enum {ename!r} has no variant {vname!r}")
            init_by_name = {fi.name: fi.value for fi in field_inits}
            val: ir.Value = ir.Constant(ety, ir.Undefined)
            tag = ir.Constant(I8, ed.variant_index(vname))
            val = builder.insert_value(val, tag, 0)
            for i, f in enumerate(var.fields):
                if f.name not in init_by_name:
                    raise ValueError(
                        f"enum_init for {ename}::{vname} missing field {f.name!r}"
                    )
                field_dest_ty = _type_to_llvm(f.type, struct_tys, enum_tys)
                coerced = _coerce_int_lit(init_by_name[f.name], field_dest_ty)
                field_val = go(coerced)
                slot_val = _pack_to_i64_slot(builder, field_val, f.type)
                val = builder.insert_value(val, slot_val, [1, i])
            return val
    raise ValueError(f"unhandled expr: {expr!r}")


_LLVM_INT_TO_QUOD: dict[int, type] = {
    1: I1Type, 8: I8Type, 16: I16Type, 32: I32Type, 64: I64Type,
}


def _coerce_int_lit(expr, dest_ty):
    """If `expr` is a bare IntLit and `dest_ty` is a different-width
    integer type, return a new IntLit retyped to `dest_ty`. Otherwise
    return `expr` unchanged.

    The rule is bare-literals-only: `parser.had_error = 1` retypes 1 to
    i8 because the field is i8, but `parser.had_error = (x + 1)` does
    NOT — composite expressions don't auto-narrow (silent truncation
    would hide bugs). Same shape as how `return 0` adopts the function's
    return type.
    """
    if not isinstance(expr, IntLit):
        return expr
    if not isinstance(dest_ty, ir.IntType):
        return expr
    src_w = int_type_width(expr.type)
    if src_w == dest_ty.width:
        return expr
    cls = _LLVM_INT_TO_QUOD.get(dest_ty.width)
    if cls is None:
        return expr
    return expr.model_copy(update={"type": cls()})


def _size_of_quod_type(
    t,
    struct_defs: dict[str, StructDef],
    enum_defs: dict[str, EnumDef],
) -> tuple[int, int]:
    """Return (abi_size, abi_alignment) in bytes for a quod type.

    Assumes a 64-bit data model (i8* is 8 bytes, 8-byte aligned). Mirrors
    the LLVM data-layout rules for our v1 type system: integers are
    naturally aligned to their width; structs accumulate fields with
    per-field alignment + tail padding to the struct's max-alignment;
    enums lower to `{i8, [N x i64]}` so they're always 8-aligned with
    `8 + 8*max_payload_slots` bytes total.
    """
    match t:
        case I1Type() | I8Type():
            return (1, 1)
        case I16Type():
            return (2, 2)
        case I32Type():
            return (4, 4)
        case I64Type() | I8PtrType():
            return (8, 8)
        case StructType(name=name):
            sd = struct_defs.get(name)
            if sd is None:
                raise ValueError(f"sizeof: undefined struct {name!r}")
            offset = 0
            max_align = 1
            for f in sd.fields:
                fsize, falign = _size_of_quod_type(f.type, struct_defs, enum_defs)
                offset = _align_to(offset, falign)
                offset += fsize
                if falign > max_align:
                    max_align = falign
            offset = _align_to(offset, max_align)
            return (offset, max_align)
        case EnumType(name=name):
            ed = enum_defs.get(name)
            if ed is None:
                raise ValueError(f"sizeof: undefined enum {name!r}")
            return (8 + 8 * ed.max_payload_slots(), 8)
    raise ValueError(f"sizeof: unhandled type {t!r}")


def _align_to(offset: int, alignment: int) -> int:
    """Round `offset` up to the next multiple of `alignment`."""
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def _pack_to_i64_slot(builder: ir.IRBuilder, val: ir.Value, declared_ty) -> ir.Value:
    """Coerce a payload field value into the i64 slot it's stored in within
    an enum's payload array. Integers narrower than i64 zero-extend (the
    bit-pattern is what we round-trip on extract); pointers ptrtoint."""
    if isinstance(val.type, ir.IntType):
        if val.type.width == 64:
            return val
        return builder.zext(val, I64)
    if isinstance(val.type, ir.PointerType):
        return builder.ptrtoint(val, I64)
    raise ValueError(f"unsupported enum payload field type {declared_ty!r}: lowered to {val.type}")


def _unpack_from_i64_slot(builder: ir.IRBuilder, slot: ir.Value, declared_ty, struct_tys, enum_tys) -> ir.Value:
    """Inverse of `_pack_to_i64_slot`: pull the declared field value back
    out of an i64 slot. Trunc for narrower ints, inttoptr for pointers."""
    target_ty = _type_to_llvm(declared_ty, struct_tys, enum_tys)
    if isinstance(target_ty, ir.IntType):
        if target_ty.width == 64:
            return slot
        return builder.trunc(slot, target_ty)
    if isinstance(target_ty, ir.PointerType):
        return builder.inttoptr(slot, target_ty)
    raise ValueError(f"unsupported enum payload field type {declared_ty!r}: lowers to {target_ty}")


def _lower_short_circuit(builder: ir.IRBuilder, lhs, rhs, *, kind: str, lower) -> ir.Value:
    """Lower `lhs || rhs` (or-style) / `lhs && rhs` (and-style) with C semantics:
    skip evaluating `rhs` when `lhs` already determines the result. Branches are
    appended to the current function; result is materialized via phi."""
    fn = builder.block.parent
    rhs_bb = fn.append_basic_block(f"sc{kind}.rhs")
    end_bb = fn.append_basic_block(f"sc{kind}.end")

    lhs_val = lower(lhs)
    lhs_block = builder.block
    if kind == "or":
        # If lhs true, skip rhs.
        builder.cbranch(lhs_val, end_bb, rhs_bb)
        short_circuit_const = ir.Constant(I1, 1)
    else:  # "and"
        # If lhs false, skip rhs.
        builder.cbranch(lhs_val, rhs_bb, end_bb)
        short_circuit_const = ir.Constant(I1, 0)

    builder.position_at_end(rhs_bb)
    rhs_val = lower(rhs)
    rhs_block = builder.block  # rhs eval may have spawned more blocks
    builder.branch(end_bb)

    builder.position_at_end(end_bb)
    phi = builder.phi(I1)
    phi.add_incoming(short_circuit_const, lhs_block)
    phi.add_incoming(rhs_val, rhs_block)
    return phi


def _collect_local_bindings(
    stmts,
    struct_tys: dict[str, "ir.IdentifiedStructType"],
    enum_tys: dict[str, "ir.IdentifiedStructType"],
    enum_defs: dict[str, EnumDef],
) -> list[tuple[str, "ir.Type"]]:
    """Pre-walk the body and return every (name, llvm_type) pair introduced by
    `Let`, `For`, or a `Match` arm binding. Allocas for these are emitted at
    the top of the function's entry block, the canonical mem2reg layout:
    `alloca` lives in entry; `store` happens at the binding point. Names must
    be unique within the function (no shadowing — match arms with the same
    binding name across arms collide too; rename them differently for now)."""
    out: list[tuple[str, ir.Type]] = []
    seen: set[str] = set()

    def visit(body) -> None:
        for s in body:
            match s:
                case Let(name=name, type=ty):
                    if name in seen:
                        raise ValueError(f"local {name!r} declared twice in the same function")
                    seen.add(name)
                    out.append((name, _type_to_llvm(ty, struct_tys, enum_tys)))
                case For(var=var, type=ty, body=for_body):
                    if var in seen:
                        raise ValueError(f"for-loop var {var!r} conflicts with another local")
                    seen.add(var)
                    out.append((var, _type_to_llvm(ty, struct_tys, enum_tys)))
                    visit(for_body)
                case If(then_body=t, else_body=e):
                    visit(t); visit(e)
                case While(body=w_body):
                    visit(w_body)
                case Match(arms=arms):
                    # Match bindings are NOT pre-collected here — they're
                    # arm-scoped, allocated inline at each arm's entry block
                    # by the Match handler in _lower_stmt. Two arms binding
                    # the same name (e.g. Array(items, count) and
                    # Object(keys, values, count) both binding `count`) get
                    # independent allocas. Just recurse into arm bodies for
                    # any nested Let / For / etc.
                    for arm in arms:
                        visit(arm.body)
    visit(stmts)
    return out


def _lower_stmt(
    builder: ir.IRBuilder,
    stmt,
    *,
    llvm_fn: ir.Function,
    params: dict[str, ir.Value],
    locals_: dict[str, ir.AllocaInstr],
    entry_bb: ir.Block,
    constants: dict[str, ir.GlobalVariable],
    module: ir.Module,
    return_claims: tuple,
    overrides: dict[str, str],
    extern_sigs: dict[str, ExternFunction],
    struct_defs: dict[str, StructDef],
    struct_tys: dict[str, "ir.IdentifiedStructType"],
    enum_defs: dict[str, EnumDef],
    enum_tys: dict[str, "ir.IdentifiedStructType"],
) -> None:
    """Lower a statement. `return_claims` are emitted as llvm.assume / runtime
    check at every ret, so callers (after inlining) see the bound."""
    def lower_expr(e):
        return _lower_expr(
            builder, e, params, module,
            constants=constants, extern_sigs=extern_sigs, locals_=locals_,
            struct_defs=struct_defs, struct_tys=struct_tys,
            enum_defs=enum_defs, enum_tys=enum_tys,
        )

    def lower_body(body):
        for s in body:
            _lower_stmt(
                builder, s, llvm_fn=llvm_fn, params=params, locals_=locals_,
                entry_bb=entry_bb, constants=constants, module=module,
                return_claims=return_claims, overrides=overrides,
                extern_sigs=extern_sigs,
                struct_defs=struct_defs, struct_tys=struct_tys,
                enum_defs=enum_defs, enum_tys=enum_tys,
            )

    match stmt:
        case ReturnExpr(value=expr):
            ret_val = lower_expr(expr)
            _emit_return_claims(builder, ret_val, return_claims, llvm_fn, module, overrides)
            builder.ret(ret_val)
            return
        case Return():
            if not isinstance(llvm_fn.function_type.return_type, ir.VoidType):
                raise ValueError(
                    f"function {llvm_fn.name!r} returns "
                    f"{llvm_fn.function_type.return_type}, not void; "
                    "use return_expr"
                )
            builder.ret_void()
            return
        case ExprStmt(value=expr):
            lower_expr(expr)
            return
        case Let(name=name, init=init):
            # Alloca was pre-emitted at the entry block; just store the init value.
            dest_ty = locals_[name].type.pointee
            init_val = lower_expr(_coerce_int_lit(init, dest_ty))
            builder.store(init_val, locals_[name])
            return
        case Assign(name=name, value=v):
            if name not in locals_:
                raise ValueError(f"assign to undeclared local {name!r}")
            dest_ty = locals_[name].type.pointee
            val = lower_expr(_coerce_int_lit(v, dest_ty))
            builder.store(val, locals_[name])
            return
        case Store(ptr=p, value=v):
            base = lower_expr(p)
            val = lower_expr(v)
            if not (isinstance(base.type, ir.PointerType) and base.type.pointee == I8):
                raise ValueError(f"store base must be i8*, got {base.type}")
            casted = builder.bitcast(base, val.type.as_pointer())
            builder.store(val, casted)
            return
        case FieldSet(local=lname, name=fname, value=v):
            if lname not in locals_:
                raise ValueError(f"field-set on undeclared local {lname!r}")
            alloca = locals_[lname]
            pointee = alloca.type.pointee
            if not isinstance(pointee, ir.IdentifiedStructType):
                raise ValueError(
                    f"field-set {fname!r} on non-struct local {lname!r} "
                    f"(local type {pointee})"
                )
            sd = struct_defs.get(pointee.name)
            if sd is None:
                raise ValueError(f"field-set on unknown struct {pointee.name!r}")
            idx = sd.field_index(fname)
            dest_ty = _type_to_llvm(sd.fields[idx].type, struct_tys, enum_tys)
            val = lower_expr(_coerce_int_lit(v, dest_ty))
            field_ptr = builder.gep(
                alloca, [ir.Constant(I32, 0), ir.Constant(I32, idx)],
            )
            builder.store(val, field_ptr)
            return
        case If(cond=cond, then_body=then_body, else_body=else_body):
            then_bb = llvm_fn.append_basic_block("then")
            else_bb = llvm_fn.append_basic_block("else")
            cond_val = lower_expr(cond)
            builder.cbranch(cond_val, then_bb, else_bb)

            merge_bb: ir.Block | None = None

            def ensure_merge() -> ir.Block:
                nonlocal merge_bb
                if merge_bb is None:
                    merge_bb = llvm_fn.append_basic_block("ifmerge")
                return merge_bb

            builder.position_at_end(then_bb)
            for s in then_body:
                _lower_stmt(
                    builder, s, llvm_fn=llvm_fn, params=params, locals_=locals_,
                    entry_bb=entry_bb, constants=constants, module=module,
                    return_claims=return_claims, overrides=overrides,
                    extern_sigs=extern_sigs,
                    struct_defs=struct_defs, struct_tys=struct_tys,
                    enum_defs=enum_defs, enum_tys=enum_tys,
                )
            if not builder.block.is_terminated:
                builder.branch(ensure_merge())

            builder.position_at_end(else_bb)
            for s in else_body:
                _lower_stmt(
                    builder, s, llvm_fn=llvm_fn, params=params, locals_=locals_,
                    entry_bb=entry_bb, constants=constants, module=module,
                    return_claims=return_claims, overrides=overrides,
                    extern_sigs=extern_sigs,
                    struct_defs=struct_defs, struct_tys=struct_tys,
                    enum_defs=enum_defs, enum_tys=enum_tys,
                )
            if not builder.block.is_terminated:
                builder.branch(ensure_merge())

            if merge_bb is not None:
                builder.position_at_end(merge_bb)
            return
        case While(cond=cond, body=body):
            header_bb = llvm_fn.append_basic_block("while.header")
            body_bb = llvm_fn.append_basic_block("while.body")
            exit_bb = llvm_fn.append_basic_block("while.exit")
            builder.branch(header_bb)

            builder.position_at_end(header_bb)
            cond_val = lower_expr(cond)
            builder.cbranch(cond_val, body_bb, exit_bb)

            builder.position_at_end(body_bb)
            lower_body(body)
            if not builder.block.is_terminated:
                builder.branch(header_bb)

            builder.position_at_end(exit_bb)
            return
        case For(var=var, lo=lo, hi=hi, body=body):
            # Snapshot lo/hi once before the loop. The slot for `var` was
            # alloca'd at entry; we re-init it on each For (loop init).
            lo_val = lower_expr(lo)
            hi_val = lower_expr(hi)
            alloca = locals_[var]
            var_ty = alloca.type.pointee  # the loop var's iN type
            builder.store(lo_val, alloca)

            header_bb = llvm_fn.append_basic_block("for.header")
            body_bb = llvm_fn.append_basic_block("for.body")
            exit_bb = llvm_fn.append_basic_block("for.exit")
            builder.branch(header_bb)

            builder.position_at_end(header_bb)
            cur = builder.load(alloca)
            cmp = builder.icmp_signed("<", cur, hi_val)
            builder.cbranch(cmp, body_bb, exit_bb)

            builder.position_at_end(body_bb)
            lower_body(body)
            if not builder.block.is_terminated:
                # increment + back-edge — step constant matches the var's width
                cur2 = builder.load(alloca)
                nxt = builder.add(cur2, ir.Constant(var_ty, 1))
                builder.store(nxt, alloca)
                builder.branch(header_bb)

            builder.position_at_end(exit_bb)
            return
        case Match(scrutinee=scrut, arms=arms):
            scrut_val = lower_expr(scrut)
            scrut_ty = scrut_val.type
            if not isinstance(scrut_ty, ir.IdentifiedStructType):
                raise ValueError(f"match scrutinee must be an enum value, got {scrut_ty}")
            ed = enum_defs.get(scrut_ty.name)
            if ed is None:
                raise ValueError(f"match scrutinee of unknown enum type {scrut_ty.name!r}")
            tag = builder.extract_value(scrut_val, 0)
            # Order arms with wildcard last for predictable codegen — the
            # switch's default block becomes the wildcard arm if present,
            # else an unreachable block.
            wildcard_arm = next((a for a in arms if a.variant == "_"), None)
            named_arms = [a for a in arms if a.variant != "_"]
            if wildcard_arm is not None:
                wildcard_bb = llvm_fn.append_basic_block("match_default")
                sw = builder.switch(tag, wildcard_bb)
            else:
                unreachable_bb = llvm_fn.append_basic_block("match_unreach")
                sw = builder.switch(tag, unreachable_bb)
                builder.position_at_end(unreachable_bb)
                builder.unreachable()
            # Lazily create the merge block — only needed if some arm falls
            # through. If every arm terminates (ret/unreachable), the match
            # statement has no successor and we leave the builder pointing
            # at an empty trailing block (placed there for any subsequent
            # statements; if none follow, _lower_function_body's
            # is_terminated check covers it).
            end_bb: ir.Block | None = None
            def ensure_end() -> ir.Block:
                nonlocal end_bb
                if end_bb is None:
                    end_bb = llvm_fn.append_basic_block("match_end")
                return end_bb

            def lower_arm_body(arm_obj):
                for s in arm_obj.body:
                    _lower_stmt(
                        builder, s, llvm_fn=llvm_fn, params=params, locals_=locals_,
                        entry_bb=entry_bb, constants=constants, module=module,
                        return_claims=return_claims, overrides=overrides,
                        extern_sigs=extern_sigs,
                        struct_defs=struct_defs, struct_tys=struct_tys,
                        enum_defs=enum_defs, enum_tys=enum_tys,
                    )
                if not builder.block.is_terminated:
                    builder.branch(ensure_end())

            for arm in named_arms:
                var = ed.variant(arm.variant)
                if var is None:
                    raise ValueError(f"match arm references unknown variant {ed.name}::{arm.variant}")
                arm_bb = llvm_fn.append_basic_block(f"match_{arm.variant}")
                sw.add_case(ir.Constant(I8, ed.variant_index(arm.variant)), arm_bb)
                builder.position_at_end(arm_bb)
                # Bind payload fields with arm-scoped allocas. We allocate
                # in the arm's basic block (not the function entry) so each
                # arm's bindings are physically distinct — different arms
                # can reuse names without colliding. Save/restore locals_
                # across the body so the bindings don't leak past the arm.
                saved: dict[str, ir.AllocaInstr | None] = {}
                for i, (binding, field) in enumerate(zip(arm.bindings, var.fields)):
                    saved[binding] = locals_.get(binding)
                    field_ll_ty = _type_to_llvm(field.type, struct_tys, enum_tys)
                    alloca = builder.alloca(field_ll_ty, name=binding)
                    slot = builder.extract_value(scrut_val, [1, i])
                    bound = _unpack_from_i64_slot(builder, slot, field.type, struct_tys, enum_tys)
                    builder.store(bound, alloca)
                    locals_[binding] = alloca
                lower_arm_body(arm)
                for b, prior in saved.items():
                    if prior is None:
                        locals_.pop(b, None)
                    else:
                        locals_[b] = prior
            if wildcard_arm is not None:
                builder.position_at_end(wildcard_bb)
                lower_arm_body(wildcard_arm)
            if end_bb is not None:
                builder.position_at_end(end_bb)
            return
    raise ValueError(f"unhandled stmt: {stmt!r}")


def _icmp_for_bound(
    builder: ir.IRBuilder, predicate: str, val: ir.Value, bound: int,
) -> ir.Value:
    """Emit an icmp comparing `val` against an integer bound from a claim.

    i1 uses unsigned comparison (the boolean {0, 1} interpretation that
    matches clang's _Bool). At signed-1-bit, bit pattern 1 sign-extends to
    -1, so `signed >= 0` on an i1 would assert "value is 0 (false)" — a
    silent corruption of any non_negative or return_in_range(min=0) claim
    on an i1 function.

    Wider ints stay signed: quod programs use signed arithmetic (slt, sge,
    srem) by convention, and a claim's min=-N is meant in signed terms.
    """
    const = ir.Constant(val.type, bound)
    if val.type.width == 1:
        return builder.icmp_unsigned(predicate, val, const)
    return builder.icmp_signed(predicate, val, const)


def _emit_return_claims(
    builder: ir.IRBuilder, ret_val: ir.Value, return_claims: tuple,
    llvm_fn: ir.Function, module: ir.Module, overrides: dict[str, str],
) -> None:
    """Emit llvm.assume / runtime-check predicates against the return value
    just before `ret`. The optimizer learns the bound; after inlining, callers
    learn it too."""
    for claim in return_claims:
        if not isinstance(claim, ReturnInRangeClaim):
            continue
        enforcement = overrides.get(claim.regime, claim.enforcement)
        if claim.min is not None:
            cmp = _icmp_for_bound(builder, ">=", ret_val, claim.min)
            _emit_for_enforcement(builder, cmp, enforcement, llvm_fn, module)
        if claim.max is not None:
            cmp = _icmp_for_bound(builder, "<=", ret_val, claim.max)
            _emit_for_enforcement(builder, cmp, enforcement, llvm_fn, module)


def _lower_claim(
    builder: ir.IRBuilder,
    claim,
    params: dict[str, ir.Value],
    llvm_fn: ir.Function,
    module: ir.Module,
    *,
    overrides: dict[str, str],
) -> None:
    # The build's per-regime override (if any) replaces the claim's stored
    # enforcement; otherwise the stored value wins.
    enforcement = overrides.get(claim.regime, claim.enforcement)
    match claim:
        case NonNegativeClaim(param=name):
            cmp = _icmp_for_bound(builder, ">=", params[name], 0)
            _emit_for_enforcement(builder, cmp, enforcement, llvm_fn, module)
            return
        case IntRangeClaim(param=name, min=lo, max=hi):
            val = params[name]
            if lo is not None:
                cmp = _icmp_for_bound(builder, ">=", val, lo)
                _emit_for_enforcement(builder, cmp, enforcement, llvm_fn, module)
            if hi is not None:
                cmp = _icmp_for_bound(builder, "<=", val, hi)
                _emit_for_enforcement(builder, cmp, enforcement, llvm_fn, module)
            return
        case ReturnInRangeClaim():
            # Function-scoped — handled per-ret in _emit_return_claims, not at
            # function entry. _lower_function_body filters these out before
            # calling _lower_claim, so we should never reach here.
            raise AssertionError("ReturnInRangeClaim should be handled per-ret")
    raise ValueError(f"unhandled claim: {claim!r}")


def _declare_function(
    module: ir.Module, fn: Function,
    struct_tys: dict[str, "ir.IdentifiedStructType"],
    enum_tys: dict[str, "ir.IdentifiedStructType"],
) -> ir.Function:
    param_tys = [_type_to_llvm(p.type, struct_tys, enum_tys) for p in fn.params]
    ret_ty = _type_to_llvm(fn.return_type, struct_tys, enum_tys)
    fn_ty = ir.FunctionType(ret_ty, param_tys)
    return ir.Function(module, fn_ty, name=fn.name)


def _declare_extern(
    module: ir.Module, ext: ExternFunction,
    struct_tys: dict[str, "ir.IdentifiedStructType"],
    enum_tys: dict[str, "ir.IdentifiedStructType"],
) -> ir.Function:
    param_types = [_type_to_llvm(t, struct_tys, enum_tys) for t in ext.effective_param_types()]
    return_type = _type_to_llvm(ext.return_type, struct_tys, enum_tys)
    fn_ty = ir.FunctionType(return_type, param_types, var_arg=ext.varargs)
    return ir.Function(module, fn_ty, name=ext.name)


def _lower_function_body(
    module: ir.Module, fn: Function, *,
    constants: dict, overrides: dict[str, str],
    extern_sigs: dict[str, ExternFunction],
    struct_defs: dict[str, StructDef],
    struct_tys: dict[str, "ir.IdentifiedStructType"],
    enum_defs: dict[str, EnumDef],
    enum_tys: dict[str, "ir.IdentifiedStructType"],
) -> None:
    llvm_fn = module.globals[fn.name]
    for arg, p in zip(llvm_fn.args, fn.params):
        arg.name = p.name
    params = {p.name: arg for p, arg in zip(fn.params, llvm_fn.args)}

    # Split claims by scope: param-scoped at function entry, return-scoped
    # at every ret site (so callers benefit after inlining).
    entry_claims = tuple(c for c in fn.claims if not isinstance(c, ReturnInRangeClaim))
    return_claims = tuple(c for c in fn.claims if isinstance(c, ReturnInRangeClaim))

    entry_bb = llvm_fn.append_basic_block(name="entry")
    builder = ir.IRBuilder(entry_bb)

    # Allocas at the very top of entry, before any other instruction. mem2reg
    # promotes them to SSA values during the optimize pass.
    locals_: dict[str, ir.AllocaInstr] = {}
    for name, ty in _collect_local_bindings(fn.body, struct_tys, enum_tys, enum_defs):
        if name in params:
            raise ValueError(f"local {name!r} shadows parameter of {fn.name!r}")
        locals_[name] = builder.alloca(ty, name=name)

    for claim in entry_claims:
        _lower_claim(builder, claim, params, llvm_fn, module, overrides=overrides)

    for stmt in fn.body:
        _lower_stmt(
            builder, stmt,
            llvm_fn=llvm_fn, params=params, locals_=locals_, entry_bb=entry_bb,
            constants=constants, module=module,
            return_claims=return_claims, overrides=overrides,
            extern_sigs=extern_sigs,
            struct_defs=struct_defs, struct_tys=struct_tys,
            enum_defs=enum_defs, enum_tys=enum_tys,
        )

    # Void functions get an implicit `ret void` if the body falls through;
    # non-void functions left without a terminator surface as a verifier
    # error (correct — the user owes a return).
    if not builder.block.is_terminated and isinstance(
        llvm_fn.function_type.return_type, ir.VoidType
    ):
        builder.ret_void()


_ARENA_NEW = "quod_arena_new"
_ARENA_DROP = "quod_arena_drop"


def _desugar_with_arena(program: Program) -> Program:
    """Rewrite every `WithArena` block into the equivalent `Let` + `Call`
    sequence, threading `arena_drop` calls before every `return` reachable
    from the body.

    Auto-declares the `quod_arena_new` / `quod_arena_drop` externs when any
    block is present, so downstream lowering sees a regular Program with
    regular calls. Idempotent on programs that contain no `WithArena`.
    """
    has_block = any(_function_uses_with_arena(fn) for fn in program.functions)
    if not has_block:
        return program

    new_functions = tuple(
        fn.model_copy(update={"body": _desugar_stmts(fn.body)})
        for fn in program.functions
    )
    new_externs = _ensure_arena_externs(program.externs)
    return program.model_copy(update={
        "functions": new_functions,
        "externs": new_externs,
    })


def _function_uses_with_arena(fn: Function) -> bool:
    return any(_stmt_contains_with_arena(s) for s in fn.body)


def _stmt_contains_with_arena(s) -> bool:
    match s:
        case WithArena():
            return True
        case If(then_body=t, else_body=e):
            return any(_stmt_contains_with_arena(x) for x in (*t, *e))
        case While(body=b) | For(body=b):
            return any(_stmt_contains_with_arena(x) for x in b)
    return False


def _desugar_stmts(stmts) -> tuple:
    out: list = []
    for s in stmts:
        match s:
            case WithArena(name=name, capacity=cap, body=body):
                inner = _desugar_stmts(body)
                drop_stmt = ExprStmt(value=Call(
                    function=_ARENA_DROP, args=(LocalRef(name=name),),
                ))
                inner_with_drops = _prepend_drop_before_returns(inner, drop_stmt)
                out.append(Let(
                    name=name, type=I8PtrType(),
                    init=Call(function=_ARENA_NEW, args=(cap,)),
                ))
                out.extend(inner_with_drops)
                # If every path through the body returns, the fall-through
                # drop and anything that follows in the outer block is
                # unreachable. The lowering pass leaves the IR builder in a
                # terminated block after such an If, so we must trim here
                # instead of emitting dead instructions on top of the ret.
                if _always_terminates(inner_with_drops):
                    return tuple(out)
                out.append(drop_stmt)
            case If(then_body=t, else_body=e):
                out.append(s.model_copy(update={
                    "then_body": _desugar_stmts(t),
                    "else_body": _desugar_stmts(e),
                }))
            case While(body=b):
                out.append(s.model_copy(update={"body": _desugar_stmts(b)}))
            case For(body=b):
                out.append(s.model_copy(update={"body": _desugar_stmts(b)}))
            case _:
                out.append(s)
    return tuple(out)


def _prepend_drop_before_returns(stmts, drop_stmt) -> tuple:
    """Walk `stmts` and emit `drop_stmt` immediately before each
    `ReturnExpr` / bare `Return`. Recurses into branches and loop bodies;
    nested `WithArena`s have already been desugared (their own drops
    already in place), so we only need to add ours."""
    out: list = []
    for s in stmts:
        match s:
            case ReturnExpr() | Return():
                out.append(drop_stmt)
                out.append(s)
            case If(then_body=t, else_body=e):
                out.append(s.model_copy(update={
                    "then_body": _prepend_drop_before_returns(t, drop_stmt),
                    "else_body": _prepend_drop_before_returns(e, drop_stmt),
                }))
            case While(body=b):
                out.append(s.model_copy(update={
                    "body": _prepend_drop_before_returns(b, drop_stmt),
                }))
            case For(body=b):
                out.append(s.model_copy(update={
                    "body": _prepend_drop_before_returns(b, drop_stmt),
                }))
            case _:
                out.append(s)
    return tuple(out)


def _always_terminates(stmts) -> bool:
    """Conservative: True only when the last reachable statement is provably
    a terminator (a `return` or an `if` whose branches both terminate). Used
    to suppress an unreachable trailing drop after `with_arena` lowering."""
    if not stmts:
        return False
    last = stmts[-1]
    match last:
        case ReturnExpr() | Return():
            return True
        case If(then_body=t, else_body=e):
            return _always_terminates(t) and _always_terminates(e)
    return False


def _ensure_arena_externs(externs: tuple[ExternFunction, ...]) -> tuple[ExternFunction, ...]:
    by_name = {e.name: e for e in externs}
    additions: list[ExternFunction] = []
    if _ARENA_NEW not in by_name:
        additions.append(ExternFunction(
            name=_ARENA_NEW,
            param_types=(I64Type(),),
            return_type=I8PtrType(),
        ))
    if _ARENA_DROP not in by_name:
        additions.append(ExternFunction(
            name=_ARENA_DROP,
            param_types=(I8PtrType(),),
            return_type=I64Type(),
        ))
    return externs + tuple(additions)


def lower(
    program: Program, *,
    target: str | None = None,
    overrides: dict[str, str] | None = None,
    entry: str | None = None,
) -> ir.Module:
    """Lower `program` to LLVM IR.

    `entry` names the function that should serve as the binary's entry point.
    If `entry` is "main" or None and the program has a function called "main",
    no wrapping happens. Otherwise a synthetic `main` is appended that calls
    `entry` and returns its result.
    """
    program = _desugar_with_arena(program)

    module = ir.Module(name="quod")
    module.triple = target or llvm.get_default_triple()
    overrides = overrides or {}

    # Pass 0: register named struct types. Two phases (allocate, then set
    # body) so a struct can mention another that's defined later in the
    # list. Cycles are already rejected by the model validator, so the
    # second pass terminates.
    struct_defs: dict[str, StructDef] = {sd.name: sd for sd in program.structs}
    struct_tys: dict[str, ir.IdentifiedStructType] = {}
    for sd in program.structs:
        struct_tys[sd.name] = module.context.get_identified_type(sd.name)
    for sd in program.structs:
        ty = struct_tys[sd.name]
        # llvmlite shares one LLVMContext across every Module in a Python
        # process, so identified types interned by name persist between
        # builds. The same struct identity coming back is fine — the body
        # was set on a prior pass, and we've already verified our model
        # rejects redefinitions with conflicting layouts.
        if not ty.is_opaque:
            continue
        body = [_type_to_llvm(f.type, struct_tys) for f in sd.fields]
        ty.set_body(*body)

    # Enums: each lowers to an identified `{i8, [N x i64]}` struct, where
    # N is the max payload-field count across the enum's variants. Each
    # payload field occupies one i64 slot — int values zero-extend to fit,
    # pointers ptrtoint. Variant payload structures are virtual: there's
    # no per-variant LLVM struct; encode/decode happens at the slot level.
    enum_defs: dict[str, EnumDef] = {ed.name: ed for ed in program.enums}
    enum_tys: dict[str, ir.IdentifiedStructType] = {}
    for ed in program.enums:
        enum_tys[ed.name] = module.context.get_identified_type(ed.name)
    for ed in program.enums:
        ty = enum_tys[ed.name]
        if not ty.is_opaque:
            continue
        payload_slots = ed.max_payload_slots()
        ty.set_body(I8, ir.ArrayType(I64, payload_slots))

    constants: dict[str, ir.GlobalVariable] = {}
    for c in program.constants:
        data = bytearray(c.value.encode("utf-8") + b"\0")
        ty = ir.ArrayType(I8, len(data))
        gv = ir.GlobalVariable(module, ty, name=c.name)
        gv.linkage = "private"
        gv.global_constant = True
        gv.initializer = ir.Constant(ty, data)
        constants[c.name] = gv

    # Pass 1: declare every user function and every extern so calls can
    # resolve regardless of order or definedness.
    for fn in program.functions:
        _declare_function(module, fn, struct_tys, enum_tys)
    for ext in program.externs:
        _declare_extern(module, ext, struct_tys, enum_tys)

    extern_sigs: dict[str, ExternFunction] = {ext.name: ext for ext in program.externs}

    # Pass 2: lower bodies of user functions only (externs have no body here).
    for fn in program.functions:
        _lower_function_body(
            module, fn, constants=constants, overrides=overrides,
            extern_sigs=extern_sigs,
            struct_defs=struct_defs, struct_tys=struct_tys,
            enum_defs=enum_defs, enum_tys=enum_tys,
        )

    if entry is not None:
        _emit_main_wrapper(module, program, entry)

    return module


def _emit_main_wrapper(module: ir.Module, program: Program, entry: str) -> None:
    """Append a synthesized `main` calling the user's entry function.

    Three cases:
      - entry is 'main' and nullary: nothing to do — user's main IS the C main.
      - entry is nullary (any name): emit `i32 main() { return ext(entry()); }`,
        where `ext` is sext/trunc as needed to match i32.
      - entry has params: emit `i32 main(i32 argc, i8** argv)` that
        bounds-checks argc, calls atoll on each argv slot, trunc/sext's to
        each param's type, and forwards. The result is similarly converted
        to i32. Auto-declares atoll / abort if absent.
    """
    fn = next((f for f in program.functions if f.name == entry), None)
    if fn is None:
        raise ValueError(f"entry function {entry!r} not found in program")

    if entry == "main" and not fn.params:
        return  # user's nullary main is already the C main

    if entry == "main" and fn.params:
        raise ValueError(
            "entry function 'main' cannot have parameters; the synthesized "
            "argv wrapper would collide. Rename your entry (e.g. to 'app' "
            "or 'run') and quod will wrap it."
        )

    for p in fn.params:
        if not isinstance(p.type, (I1Type, I8Type, I16Type, I32Type, I64Type)):
            raise ValueError(
                f"entry function {entry!r} param {p.name!r} has non-int type "
                f"{p.type!r}; the argv wrapper only knows how to parse integers. "
                f"Use a nullary entry that constructs richer args internally."
            )
    if not isinstance(fn.return_type, (I1Type, I8Type, I16Type, I32Type, I64Type)):
        raise ValueError(
            f"entry function {entry!r} returns non-int type {fn.return_type!r}; "
            f"main must return an integer exit code"
        )

    if any(f.name == "main" for f in program.functions):
        raise ValueError(
            f"cannot use {entry!r} as entry: program already defines a function "
            f"named 'main'; remove one or rename the conflict"
        )

    target_fn = module.globals[entry]

    if not fn.params:
        # Nullary entry: simple forward.
        main_fn = ir.Function(module, ir.FunctionType(I32, []), name="main")
        bb = main_fn.append_basic_block("entry")
        builder = ir.IRBuilder(bb)
        result = builder.call(target_fn, [])
        builder.ret(_resize_int(builder, result, I32))
        return

    atoll = _get_or_declare_atoll(module)
    abort = _get_or_declare_abort(module)
    n = len(fn.params)

    main_ty = ir.FunctionType(I32, [I32, I8.as_pointer().as_pointer()])
    main_fn = ir.Function(module, main_ty, name="main")
    main_fn.args[0].name = "argc"
    main_fn.args[1].name = "argv"
    argc, argv = main_fn.args

    entry_bb = main_fn.append_basic_block("entry")
    parse_bb = main_fn.append_basic_block("parse")
    fail_bb = main_fn.append_basic_block("fail")
    builder = ir.IRBuilder(entry_bb)

    required = ir.Constant(I32, n + 1)
    too_few = builder.icmp_signed("<", argc, required)
    builder.cbranch(too_few, fail_bb, parse_bb)

    builder.position_at_end(fail_bb)
    builder.call(abort, [])
    builder.unreachable()

    builder.position_at_end(parse_bb)
    parsed_args = []
    for i, p in enumerate(fn.params):
        # argv[i+1]: GEP on i8** then load to get i8*
        idx = ir.Constant(I64, i + 1)
        arg_slot = builder.gep(argv, [idx])         # i8**
        arg_ptr = builder.load(arg_slot)            # i8*
        parsed_i64 = builder.call(atoll, [arg_ptr]) # i64
        target_ty = _type_to_llvm(p.type)
        parsed_args.append(_resize_int(builder, parsed_i64, target_ty))

    result = builder.call(target_fn, parsed_args)
    builder.ret(_resize_int(builder, result, I32))


def _resize_int(builder: ir.IRBuilder, val: ir.Value, target_ty: ir.IntType) -> ir.Value:
    """Sign-extend, truncate, or no-op a value to match `target_ty`."""
    src_w = val.type.width
    dst_w = target_ty.width
    if src_w == dst_w:
        return val
    if src_w < dst_w:
        return builder.sext(val, target_ty)
    return builder.trunc(val, target_ty)


# ---------- Backend pipeline ----------

_native_initialized = False
_all_initialized = False


def _ensure_initialized(*, cross: bool) -> None:
    """Initialize LLVM target backends. `cross=True` brings in every target
    LLVM was built with, needed when emitting code for a non-host triple."""
    global _native_initialized, _all_initialized
    if not _native_initialized:
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        _native_initialized = True
    if cross and not _all_initialized:
        llvm.initialize_all_targets()
        llvm.initialize_all_asmprinters()
        _all_initialized = True


def parse_and_verify(module: ir.Module):
    parsed = llvm.parse_assembly(str(module))
    parsed.verify()
    return parsed


def optimize_module(parsed_module, target_machine, *, speed_level: int) -> None:
    pto = llvm.PipelineTuningOptions(speed_level=speed_level)
    pb = llvm.PassBuilder(target_machine, pto)
    mpm = pb.getModulePassManager()
    mpm.run(parsed_module, pb)


def make_target_machine(target: str | None = None):
    triple = target or llvm.get_default_triple()
    _ensure_initialized(cross=triple != llvm.get_default_triple())
    return llvm.Target.from_triple(triple).create_target_machine(
        reloc="pic", codemodel="default",
    )


@dataclass(frozen=True)
class BinResult:
    name: str
    entry: str
    ir_unopt: Path
    ir_opt: Path | None
    object_path: Path
    binary: Path | None


@dataclass(frozen=True)
class CompileResult:
    bins: tuple[BinResult, ...]


def has_function(program: Program, name: str) -> bool:
    return any(fn.name == name for fn in program.functions)


def compile_program(
    program: Program,
    *,
    build_dir: Path,
    bins: tuple[tuple[str, str], ...] = (("main", "main"),),
    profile: int = 2,
    link: bool = True,
    libraries: tuple[str, ...] = (),
    target: str | None = None,
    overrides: dict[str, str] | None = None,
) -> CompileResult:
    """Compile `program` into one binary per bin.

    `bins` is a tuple of (name, entry) pairs: `name` is the output binary
    filename, `entry` is the program function used as the entry point. The
    default ((`"main"`, `"main"`),) preserves pre-config behavior.
    """
    if not 0 <= profile <= 3:
        raise ValueError(f"profile must be 0..3, got {profile}")
    build_dir.mkdir(parents=True, exist_ok=True)

    # Resolve imports first: stdlib functions (e.g. std.str) need to be
    # visible to the analysis pass and to lowering, just like user functions.
    program = resolve_imports(program)

    # Elaborate: derive lattice claims and merge them into the program before
    # lowering. Override flags (--enforce-lattice etc.) apply uniformly to
    # both stored and derived claims via the override map.
    derived = derive_lattice_claims(program)
    program = elaborate(program, derived)

    target_machine = make_target_machine(target=target)
    results: list[BinResult] = []

    for bin_name, entry in bins:
        if not has_function(program, entry):
            raise ValueError(f"bin {bin_name!r}: entry function {entry!r} not found")

        module = lower(program, target=target, overrides=overrides, entry=entry)
        ir_unopt = build_dir / f"{bin_name}.unopt.ll"
        ir_unopt.write_text(str(module))

        parsed = parse_and_verify(module)
        ir_opt: Path | None = None
        if profile > 0:
            optimize_module(parsed, target_machine, speed_level=profile)
            ir_opt = build_dir / f"{bin_name}.opt.ll"
            ir_opt.write_text(str(parsed))

        object_path = build_dir / f"{bin_name}.o"
        object_path.write_bytes(target_machine.emit_object(parsed))

        binary: Path | None = None
        if link:
            binary = build_dir / bin_name
            # Build the runtime archive (arena allocator etc.) into the same
            # build_dir, matching the user's target. Archive linking is
            # by-reference, so unused runtime symbols stay stripped.
            runtime_archive = build_runtime_archive(build_dir, target=target)
            cmd = ["clang"]
            if target:
                cmd += ["-target", target]
            cmd += [str(object_path), str(runtime_archive), "-o", str(binary)]
            cmd += [f"-l{lib}" for lib in libraries]
            subprocess.run(cmd, check=True)

        results.append(BinResult(
            name=bin_name, entry=entry,
            ir_unopt=ir_unopt, ir_opt=ir_opt,
            object_path=object_path, binary=binary,
        ))

    return CompileResult(bins=tuple(results))
