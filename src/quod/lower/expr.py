"""Expression lowering: `_lower_expr` + the if-expr / short-circuit
helpers it dispatches into. Every quod expression that can appear inside
a function body lands here; the result is an `ir.Value`.

Mutual recursion with statement lowering: the statement walker calls
`_lower_expr` for embedded expressions; conversely, `_lower_expr` only
calls into claims (`_emit_extern_call_postconditions` for extern-call
return bounds), not stmt — there is no expr→stmt edge.
"""

from __future__ import annotations

from llvmlite import ir

from quod.lower.claims import _emit_extern_call_postconditions
from quod.lower.runtime_decls import (
    _get_or_declare_fptosi_sat,
    _get_or_declare_fptoui_sat,
)
from quod.lower.types import (
    F32,
    F64,
    I1,
    I8,
    I32,
    I64,
    _ICMP_SIGNED,
    _ICMP_UNSIGNED,
    _coerce_int_lit,
    _size_of_quod_type,
    _type_to_llvm,
    _variant_struct_ty,
)
from quod.model import (
    BinOp,
    Call,
    Cast,
    CharLit,
    EnumDef,
    EnumInit,
    ExternFunction,
    F32Type,
    F64Type,
    FieldRead,
    Function,
    IfExpr,
    IntLit,
    Load,
    LoadField,
    LocalRef,
    Not,
    NullPtr,
    ParamRef,
    PtrOffset,
    ReturnRef,
    ShortCircuitAnd,
    ShortCircuitOr,
    SizeOf,
    StringRef,
    StructDef,
    StructInit,
    TryExpr,
    int_type_signed,
    int_type_width,
)


def _lower_expr(
    builder: ir.IRBuilder, expr, params: dict[str, ir.Value], module: ir.Module,
    *, constants: dict[str, ir.GlobalVariable], extern_sigs: dict[str, ExternFunction],
    locals_: dict[str, ir.AllocaInstr],
    struct_defs: dict[str, StructDef],
    struct_tys: dict[str, "ir.IdentifiedStructType"],
    enum_defs: dict[str, EnumDef],
    enum_tys: dict[str, "ir.IdentifiedStructType"],
    fn: Function | None = None,
    local_qtypes: dict[str, object] | None = None,
    fn_returns: dict[str, object] | None = None,
) -> ir.Value:
    def go(e):
        return _lower_expr(
            builder, e, params, module,
            constants=constants, extern_sigs=extern_sigs, locals_=locals_,
            struct_defs=struct_defs, struct_tys=struct_tys,
            enum_defs=enum_defs, enum_tys=enum_tys,
            fn=fn, local_qtypes=local_qtypes, fn_returns=fn_returns,
        )

    match expr:
        case IntLit(type=t, value=v):
            return ir.Constant(_type_to_llvm(t), v)
        case ParamRef(name=n):
            return params[n]
        case LocalRef(name=n):
            assert n in locals_, (
                f"validator invariant: LocalRef({n!r}) lowered without "
                f"a matching local — quod.validate should have caught this"
            )
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
        case BinOp(op="urem", lhs=l, rhs=r):
            return builder.urem(go(l), go(r))
        case BinOp(op=op, lhs=l, rhs=r) if op in _ICMP_SIGNED:
            return builder.icmp_signed(_ICMP_SIGNED[op], go(l), go(r))
        case BinOp(op=op, lhs=l, rhs=r) if op in _ICMP_UNSIGNED:
            return builder.icmp_unsigned(_ICMP_UNSIGNED[op], go(l), go(r))
        case BinOp(op="or", lhs=l, rhs=r):
            return builder.or_(go(l), go(r))
        case BinOp(op="and", lhs=l, rhs=r):
            return builder.and_(go(l), go(r))
        case BinOp(op="xor", lhs=l, rhs=r):
            return builder.xor(go(l), go(r))
        case BinOp(op="shl", lhs=l, rhs=r):
            return builder.shl(go(l), go(r))
        case BinOp(op="ashr", lhs=l, rhs=r):
            return builder.ashr(go(l), go(r))
        case BinOp(op="lshr", lhs=l, rhs=r):
            return builder.lshr(go(l), go(r))
        case ShortCircuitOr(lhs=l, rhs=r):
            return _lower_short_circuit(builder, l, r, kind="or", lower=go)
        case ShortCircuitAnd(lhs=l, rhs=r):
            return _lower_short_circuit(builder, l, r, kind="and", lower=go)
        case IfExpr(cond=cond, then_value=t, else_value=e):
            return _lower_if_expr(builder, cond, t, e, lower=go)
        case Not(operand=op):
            return builder.xor(go(op), ir.Constant(I1, 1))
        case ReturnRef():
            raise AssertionError(
                "ReturnRef encountered outside a predicate-lowering context "
                "— ReturnRef is only valid inside PredicateClaim.expr"
            )
        case StringRef(name=n):
            gv = constants[n]
            return builder.bitcast(gv, I8.as_pointer())
        case Call(function=fname, args=args):
            callee = module.globals.get(fname)
            assert callee is not None, (
                f"validator invariant: call to undeclared function {fname!r} "
                f"— quod.validate should have caught this"
            )
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
            ret = builder.call(callee, arg_vals)
            # If the callee is an extern with declared return claims, emit
            # the postcondition assumes against this call's return value
            # so the optimizer can exploit the bound at this site.
            ext = extern_sigs.get(fname)
            if ext is not None and ext.claims:
                _emit_extern_call_postconditions(builder, module, ret, ext.claims)
            return ret
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
        case LoadField(ptr=p, struct_type=tname, name=fname):
            base = go(p)
            if not (isinstance(base.type, ir.PointerType) and base.type.pointee == I8):
                raise ValueError(f"load_field base must be i8*, got {base.type}")
            sd = struct_defs.get(tname)
            sty = struct_tys.get(tname)
            if sd is None or sty is None:
                raise ValueError(f"load_field on undefined struct {tname!r}")
            idx = sd.field_index(fname)
            casted = builder.bitcast(base, sty.as_pointer())
            field_ptr = builder.gep(
                casted, [ir.Constant(I32, 0), ir.Constant(I32, idx)],
                inbounds=True,
            )
            return builder.load(field_ptr)
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
        case Cast(value=v, target_type=tgt):
            val = go(v)
            src_qty = _quod_type_of(
                v, fn=fn, local_qtypes=local_qtypes,
                extern_sigs=extern_sigs, fn_returns=fn_returns,
                struct_defs=struct_defs, enum_defs=enum_defs,
            )
            return _lower_cast(builder, module, val, src_qty, tgt)
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
        case TryExpr(value=inner):
            inner_val = go(inner)
            inner_ty = inner_val.type
            if not isinstance(inner_ty, ir.IdentifiedStructType):
                raise ValueError(
                    f"? requires an enum value, got {inner_ty}"
                )
            src_ed = enum_defs.get(inner_ty.name)
            if src_ed is None:
                raise ValueError(f"? on unknown enum type {inner_ty.name!r}")
            happy, sad = src_ed.try_variants()
            if happy is None:
                raise ValueError(
                    f"? on enum {src_ed.name!r}: not ?-eligible "
                    "(needs exactly two variants — one with a single "
                    "payload field, one with no payload)"
                )
            llvm_fn_local = builder.block.parent
            ret_ty = llvm_fn_local.function_type.return_type
            if not (isinstance(ret_ty, ir.IdentifiedStructType)
                    and ret_ty.name == src_ed.name):
                raise ValueError(
                    f"? on {src_ed.name!r} requires the enclosing function "
                    f"to return {src_ed.name!r}, got {ret_ty}"
                )
            # Spill, switch on tag.
            val_alloca = builder.alloca(inner_ty)
            builder.store(inner_val, val_alloca)
            tag_ptr = builder.gep(
                val_alloca, [ir.Constant(I32, 0), ir.Constant(I32, 0)],
            )
            tag = builder.load(tag_ptr)
            sad_bb = llvm_fn_local.append_basic_block("try_sad")
            happy_bb = llvm_fn_local.append_basic_block("try_happy")
            sw = builder.switch(tag, sad_bb)
            sw.add_case(ir.Constant(I8, src_ed.variant_index(sad.name)), sad_bb)
            sw.add_case(ir.Constant(I8, src_ed.variant_index(happy.name)), happy_bb)
            # Sad path: build the same enum's sad variant and ret it.
            builder.position_at_end(sad_bb)
            sad_alloca = builder.alloca(inner_ty)
            sad_tag_ptr = builder.gep(
                sad_alloca, [ir.Constant(I32, 0), ir.Constant(I32, 0)],
            )
            builder.store(
                ir.Constant(I8, src_ed.variant_index(sad.name)), sad_tag_ptr,
            )
            sad_val = builder.load(sad_alloca)
            builder.ret(sad_val)
            # Happy path: bitcast payload to variant struct, GEP+load
            # the single field. Builder is left at happy_bb so callers
            # continue from here.
            builder.position_at_end(happy_bb)
            payload_ptr = builder.gep(
                val_alloca, [ir.Constant(I32, 0), ir.Constant(I32, 1)],
            )
            variant_ty = _variant_struct_ty(happy, struct_tys, enum_tys)
            variant_ptr = builder.bitcast(payload_ptr, variant_ty.as_pointer())
            field_ptr = builder.gep(
                variant_ptr, [ir.Constant(I32, 0), ir.Constant(I32, 0)],
            )
            return builder.load(field_ptr)
        case EnumInit(enum=ename, variant=vname, fields=field_inits):
            ed = enum_defs.get(ename)
            ety = enum_tys.get(ename)
            if ed is None or ety is None:
                raise ValueError(f"enum_init for undefined enum {ename!r}")
            var = ed.variant(vname)
            if var is None:
                raise ValueError(f"enum_init: enum {ename!r} has no variant {vname!r}")
            init_by_name = {fi.name: fi.value for fi in field_inits}
            # Build the variant's LLVM struct value via insertvalue chain.
            variant_ty = _variant_struct_ty(var, struct_tys, enum_tys)
            variant_val: ir.Value = ir.Constant(variant_ty, ir.Undefined)
            for i, f in enumerate(var.fields):
                if f.name not in init_by_name:
                    raise ValueError(
                        f"enum_init for {ename}::{vname} missing field {f.name!r}"
                    )
                field_dest_ty = _type_to_llvm(f.type, struct_tys, enum_tys)
                coerced = _coerce_int_lit(init_by_name[f.name], field_dest_ty)
                field_val = go(coerced)
                variant_val = builder.insert_value(variant_val, field_val, i)
            # Pack into the enum: alloca, store tag, bitcast payload to
            # variant_struct*, store variant value, load enum back. The
            # alloca lives in the current block; the optimizer's pipeline
            # (mem2reg/SROA + instcombine/GVN) folds it out for simple cases.
            enum_alloca = builder.alloca(ety)
            tag_ptr = builder.gep(
                enum_alloca, [ir.Constant(I32, 0), ir.Constant(I32, 0)],
            )
            builder.store(ir.Constant(I8, ed.variant_index(vname)), tag_ptr)
            payload_ptr = builder.gep(
                enum_alloca, [ir.Constant(I32, 0), ir.Constant(I32, 1)],
            )
            variant_ptr = builder.bitcast(payload_ptr, variant_ty.as_pointer())
            builder.store(variant_val, variant_ptr)
            return builder.load(enum_alloca)
    raise ValueError(f"unhandled expr: {expr!r}")


def _lower_if_expr(builder: ir.IRBuilder, cond, then_value, else_value, *, lower) -> ir.Value:
    """Lower an `IfExpr` (ternary `cond ? a : b`) via branch + phi.
    Both arms must produce values of the same LLVM type; only the
    arm whose condition matches is evaluated, so side effects in the
    unselected arm are correctly skipped."""
    fn = builder.block.parent
    then_bb = fn.append_basic_block("ifexpr.then")
    else_bb = fn.append_basic_block("ifexpr.else")
    end_bb = fn.append_basic_block("ifexpr.end")

    cond_val = lower(cond)
    builder.cbranch(cond_val, then_bb, else_bb)

    builder.position_at_end(then_bb)
    then_val = lower(then_value)
    then_block = builder.block  # nested branches may have repositioned
    builder.branch(end_bb)

    builder.position_at_end(else_bb)
    else_val = lower(else_value)
    else_block = builder.block
    builder.branch(end_bb)

    builder.position_at_end(end_bb)
    if then_val.type != else_val.type:
        raise ValueError(
            f"if_expr branches produced mismatched LLVM types: "
            f"{then_val.type} vs {else_val.type} — quod.validate should "
            f"have caught this"
        )
    phi = builder.phi(then_val.type)
    phi.add_incoming(then_val, then_block)
    phi.add_incoming(else_val, else_block)
    return phi


def _is_quod_float(t) -> bool:
    return isinstance(t, (F32Type, F64Type))


def _is_quod_int(t) -> bool:
    try:
        int_type_signed(t)
        return True
    except (ValueError, TypeError):
        return False


def _lower_cast(
    builder: ir.IRBuilder, module: ir.Module,
    val: ir.Value, src_qty, tgt_qty,
) -> ir.Value:
    """Dispatch the nine numeric-conversion arms of `Cast`. `val` is the
    already-lowered LLVM value; `src_qty` and `tgt_qty` are the quod-side
    source and target types (signedness lives there, not on the LLVM type).
    """
    target_ty = _type_to_llvm(tgt_qty)
    src_is_int = _is_quod_int(src_qty)
    tgt_is_int = _is_quod_int(tgt_qty)
    src_is_float = _is_quod_float(src_qty)
    tgt_is_float = _is_quod_float(tgt_qty)

    if src_is_int and tgt_is_int:
        src_w = int_type_width(src_qty)
        dst_w = int_type_width(tgt_qty)
        if src_w == dst_w:
            return val
        if src_w < dst_w:
            return (builder.sext(val, target_ty)
                    if int_type_signed(src_qty)
                    else builder.zext(val, target_ty))
        return builder.trunc(val, target_ty)

    if src_is_int and tgt_is_float:
        return (builder.sitofp(val, target_ty)
                if int_type_signed(src_qty)
                else builder.uitofp(val, target_ty))

    if src_is_float and tgt_is_int:
        if int_type_signed(tgt_qty):
            intrinsic = _get_or_declare_fptosi_sat(module, target_ty, val.type)
        else:
            intrinsic = _get_or_declare_fptoui_sat(module, target_ty, val.type)
        return builder.call(intrinsic, [val])

    if src_is_float and tgt_is_float:
        src_w = 32 if isinstance(src_qty, F32Type) else 64
        dst_w = 32 if isinstance(tgt_qty, F32Type) else 64
        if src_w == dst_w:
            return val
        if src_w < dst_w:
            return builder.fpext(val, target_ty)
        return builder.fptrunc(val, target_ty)

    raise ValueError(
        f"cast source/target must both be numeric (int or float); "
        f"got source={src_qty!r}, target={tgt_qty!r}"
    )


def _quod_type_of(
    expr,
    *,
    fn,
    local_qtypes,
    extern_sigs,
    fn_returns,
    struct_defs,
    enum_defs,
):
    """Best-effort source quod-type inference used by Cast lowering.

    The lowerer normally strips quod types — `_type_to_llvm` collapses
    IXType / UXType to LLVM iN with no signedness. Cast lowering needs
    to recover the source signedness to choose sext vs zext, sitofp vs
    uitofp. This helper walks the value expression and returns its quod
    Type. Assumes the program already passed `validate_or_raise` —
    cases that can't be inferred raise (a Cast wrapping such an
    expression is a Cast-extension request, not an existing-program
    issue).
    """
    match expr:
        case IntLit(type=t):
            return t
        case CharLit():
            from quod.model import I8Type
            return I8Type()
        case SizeOf():
            from quod.model import I64Type
            return I64Type()
        case ParamRef(name=n):
            if fn is None:
                raise ValueError(f"cannot infer ParamRef {n!r} type — no Function in lower context")
            for p in fn.params:
                if p.name == n:
                    return p.type
            raise ValueError(f"ParamRef {n!r} not found in {fn.name!r}")
        case LocalRef(name=n):
            if local_qtypes is None or n not in local_qtypes:
                raise ValueError(f"cannot infer LocalRef {n!r} type — not in local_qtypes")
            return local_qtypes[n]
        case Cast(target_type=t):
            return t
        case Load(type=t):
            return t
        case BinOp(op=op, lhs=l):
            if op in ("eq", "ne", "slt", "sle", "sgt", "sge",
                      "ult", "ule", "ugt", "uge"):
                from quod.model import I1Type
                return I1Type()
            return _quod_type_of(
                l, fn=fn, local_qtypes=local_qtypes,
                extern_sigs=extern_sigs, fn_returns=fn_returns,
                struct_defs=struct_defs, enum_defs=enum_defs,
            )
        case ShortCircuitAnd() | ShortCircuitOr() | Not():
            from quod.model import I1Type
            return I1Type()
        case IfExpr(then_value=t):
            return _quod_type_of(
                t, fn=fn, local_qtypes=local_qtypes,
                extern_sigs=extern_sigs, fn_returns=fn_returns,
                struct_defs=struct_defs, enum_defs=enum_defs,
            )
        case Call(function=fname):
            if fn_returns is not None and fname in fn_returns:
                return fn_returns[fname]
            ext = extern_sigs.get(fname) if extern_sigs is not None else None
            if ext is not None:
                return ext.return_type
            raise ValueError(f"cannot infer return type of call to {fname!r}")
        case ReturnRef():
            if fn is None:
                raise ValueError("cannot infer ReturnRef type — no Function in lower context")
            return fn.return_type
        case LoadField(struct_type=tname, name=fname):
            if struct_defs is None or tname not in struct_defs:
                raise ValueError(f"cannot infer LoadField on {tname!r} — struct not in lower context")
            f = struct_defs[tname].field(fname)
            if f is None:
                raise ValueError(f"struct {tname!r} has no field {fname!r}")
            return f.type
        case FieldRead(value=v, name=fname):
            inner_ty = _quod_type_of(
                v, fn=fn, local_qtypes=local_qtypes,
                extern_sigs=extern_sigs, fn_returns=fn_returns,
                struct_defs=struct_defs, enum_defs=enum_defs,
            )
            from quod.model import StructType
            if not isinstance(inner_ty, StructType):
                raise ValueError(f"FieldRead on non-struct type {inner_ty!r}")
            if struct_defs is None or inner_ty.name not in struct_defs:
                raise ValueError(f"FieldRead on unknown struct {inner_ty.name!r}")
            f = struct_defs[inner_ty.name].field(fname)
            if f is None:
                raise ValueError(f"struct {inner_ty.name!r} has no field {fname!r}")
            return f.type
    raise NotImplementedError(
        f"_quod_type_of: cannot infer source type for {type(expr).__name__} "
        f"— extend the helper if a new Cast value-shape is needed"
    )


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
