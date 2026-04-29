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
from quod.model import (
    Assign,
    BinOp,
    Call,
    ExprStmt,
    ExternFunction,
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
    Let,
    LocalRef,
    NonNegativeClaim,
    ParamRef,
    Program,
    ReturnExpr,
    ReturnInRangeClaim,
    ReturnInt,
    ShortCircuitAnd,
    ShortCircuitOr,
    StringRef,
    While,
)


I1 = ir.IntType(1)
I8 = ir.IntType(8)
I16 = ir.IntType(16)
I32 = ir.IntType(32)
I64 = ir.IntType(64)


def _type_to_llvm(t):
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
) -> ir.Value:
    def go(e):
        return _lower_expr(
            builder, e, params, module,
            constants=constants, extern_sigs=extern_sigs, locals_=locals_,
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
            return builder.add(go(l), go(r))
        case BinOp(op="sub", lhs=l, rhs=r):
            return builder.sub(go(l), go(r))
        case BinOp(op="mul", lhs=l, rhs=r):
            return builder.mul(go(l), go(r))
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
            arg_vals = [go(a) for a in args]
            return builder.call(callee, arg_vals)
    raise ValueError(f"unhandled expr: {expr!r}")


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


def _collect_local_bindings(stmts) -> list[tuple[str, "ir.Type"]]:
    """Pre-walk the body and return every (name, llvm_type) pair introduced by
    `Let` or `For`. Allocas for these are emitted at the top of the function's
    entry block, the canonical mem2reg layout: `alloca` lives in entry; `store`
    happens at the binding point. Names must be unique within the function
    (no shadowing between Lets, between Fors, or across scopes); two `For`s
    with the same loop variable name aren't supported in this round."""
    out: list[tuple[str, ir.Type]] = []
    seen: set[str] = set()

    def visit(body) -> None:
        for s in body:
            match s:
                case Let(name=name, type=ty):
                    if name in seen:
                        raise ValueError(f"local {name!r} declared twice in the same function")
                    seen.add(name)
                    out.append((name, _type_to_llvm(ty)))
                case For(var=var, type=ty, body=for_body):
                    if var in seen:
                        raise ValueError(f"for-loop var {var!r} conflicts with another local")
                    seen.add(var)
                    out.append((var, _type_to_llvm(ty)))
                    visit(for_body)
                case If(then_body=t, else_body=e):
                    visit(t); visit(e)
                case While(body=w_body):
                    visit(w_body)
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
) -> None:
    """Lower a statement. `return_claims` are emitted as llvm.assume / runtime
    check at every ret, so callers (after inlining) see the bound."""
    def lower_expr(e):
        return _lower_expr(
            builder, e, params, module,
            constants=constants, extern_sigs=extern_sigs, locals_=locals_,
        )

    def lower_body(body):
        for s in body:
            _lower_stmt(
                builder, s, llvm_fn=llvm_fn, params=params, locals_=locals_,
                entry_bb=entry_bb, constants=constants, module=module,
                return_claims=return_claims, overrides=overrides,
                extern_sigs=extern_sigs,
            )

    match stmt:
        case ReturnInt(value=v):
            ret_val = ir.Constant(llvm_fn.function_type.return_type, v)
            _emit_return_claims(builder, ret_val, return_claims, llvm_fn, module, overrides)
            builder.ret(ret_val)
            return
        case ReturnExpr(value=expr):
            ret_val = lower_expr(expr)
            _emit_return_claims(builder, ret_val, return_claims, llvm_fn, module, overrides)
            builder.ret(ret_val)
            return
        case ExprStmt(value=expr):
            lower_expr(expr)
            return
        case Let(name=name, init=init):
            # Alloca was pre-emitted at the entry block; just store the init value.
            init_val = lower_expr(init)
            builder.store(init_val, locals_[name])
            return
        case Assign(name=name, value=v):
            if name not in locals_:
                raise ValueError(f"assign to undeclared local {name!r}")
            val = lower_expr(v)
            builder.store(val, locals_[name])
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
    raise ValueError(f"unhandled stmt: {stmt!r}")


def _emit_return_claims(
    builder: ir.IRBuilder, ret_val: ir.Value, return_claims: tuple,
    llvm_fn: ir.Function, module: ir.Module, overrides: dict[str, str],
) -> None:
    """Emit llvm.assume / runtime-check predicates against the return value
    just before `ret`. The optimizer learns the bound; after inlining, callers
    learn it too."""
    ret_ty = ret_val.type
    for claim in return_claims:
        if not isinstance(claim, ReturnInRangeClaim):
            continue
        enforcement = overrides.get(claim.regime, claim.enforcement)
        if claim.min is not None:
            cmp = builder.icmp_signed(">=", ret_val, ir.Constant(ret_ty, claim.min))
            _emit_for_enforcement(builder, cmp, enforcement, llvm_fn, module)
        if claim.max is not None:
            cmp = builder.icmp_signed("<=", ret_val, ir.Constant(ret_ty, claim.max))
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
            val = params[name]
            cmp = builder.icmp_signed(">=", val, ir.Constant(val.type, 0))
            _emit_for_enforcement(builder, cmp, enforcement, llvm_fn, module)
            return
        case IntRangeClaim(param=name, min=lo, max=hi):
            val = params[name]
            if lo is not None:
                cmp = builder.icmp_signed(">=", val, ir.Constant(val.type, lo))
                _emit_for_enforcement(builder, cmp, enforcement, llvm_fn, module)
            if hi is not None:
                cmp = builder.icmp_signed("<=", val, ir.Constant(val.type, hi))
                _emit_for_enforcement(builder, cmp, enforcement, llvm_fn, module)
            return
        case ReturnInRangeClaim():
            # Function-scoped — handled per-ret in _emit_return_claims, not at
            # function entry. _lower_function_body filters these out before
            # calling _lower_claim, so we should never reach here.
            raise AssertionError("ReturnInRangeClaim should be handled per-ret")
    raise ValueError(f"unhandled claim: {claim!r}")


def _declare_function(module: ir.Module, fn: Function) -> ir.Function:
    param_tys = [_type_to_llvm(p.type) for p in fn.params]
    ret_ty = _type_to_llvm(fn.return_type)
    fn_ty = ir.FunctionType(ret_ty, param_tys)
    return ir.Function(module, fn_ty, name=fn.name)


def _declare_extern(module: ir.Module, ext: ExternFunction) -> ir.Function:
    param_types = [_type_to_llvm(t) for t in ext.effective_param_types()]
    return_type = _type_to_llvm(ext.return_type)
    fn_ty = ir.FunctionType(return_type, param_types, var_arg=ext.varargs)
    return ir.Function(module, fn_ty, name=ext.name)


def _lower_function_body(
    module: ir.Module, fn: Function, *,
    constants: dict, overrides: dict[str, str],
    extern_sigs: dict[str, ExternFunction],
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
    for name, ty in _collect_local_bindings(fn.body):
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
        )


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
    module = ir.Module(name="quod")
    module.triple = target or llvm.get_default_triple()
    overrides = overrides or {}

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
        _declare_function(module, fn)
    for ext in program.externs:
        _declare_extern(module, ext)

    extern_sigs: dict[str, ExternFunction] = {ext.name: ext for ext in program.externs}

    # Pass 2: lower bodies of user functions only (externs have no body here).
    for fn in program.functions:
        _lower_function_body(
            module, fn, constants=constants, overrides=overrides,
            extern_sigs=extern_sigs,
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
            cmd = ["clang"]
            if target:
                cmd += ["-target", target]
            cmd += [str(object_path), "-o", str(binary)]
            subprocess.run(cmd, check=True)

        results.append(BinResult(
            name=bin_name, entry=entry,
            ir_unopt=ir_unopt, ir_opt=ir_opt,
            object_path=object_path, binary=binary,
        ))

    return CompileResult(bins=tuple(results))
