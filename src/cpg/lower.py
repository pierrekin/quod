"""Lowering: Program -> LLVM IR -> object/binary, plus the optimization pass.

Two-pass over functions: first declare every user function (so user-level
calls in expressions resolve regardless of definition order), then lower
each function body.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from llvmlite import binding as llvm
from llvmlite import ir

from cpg.analysis import derive_lattice_claims, elaborate
from cpg.model import (
    BinOp,
    Call,
    CallPuts,
    Function,
    If,
    IntLit,
    IntRangeClaim,
    NonNegativeClaim,
    ParamRef,
    Program,
    ReturnExpr,
    ReturnInt,
)


I8 = ir.IntType(8)
I32 = ir.IntType(32)
I1 = ir.IntType(1)


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


def _lower_expr(builder: ir.IRBuilder, expr, params: dict[str, ir.Value], module: ir.Module) -> ir.Value:
    match expr:
        case IntLit(value=v):
            return ir.Constant(I32, v)
        case ParamRef(name=n):
            return params[n]
        case BinOp(op="add", lhs=l, rhs=r):
            return builder.add(_lower_expr(builder, l, params, module), _lower_expr(builder, r, params, module))
        case BinOp(op="slt", lhs=l, rhs=r):
            return builder.icmp_signed("<", _lower_expr(builder, l, params, module), _lower_expr(builder, r, params, module))
        case Call(function=fname, args=args):
            callee = module.globals.get(fname)
            if callee is None:
                raise ValueError(f"call to undeclared function {fname!r}")
            arg_vals = [_lower_expr(builder, a, params, module) for a in args]
            return builder.call(callee, arg_vals)
    raise ValueError(f"unhandled expr: {expr!r}")


def _lower_stmt(
    builder: ir.IRBuilder,
    stmt,
    *,
    llvm_fn: ir.Function,
    params: dict[str, ir.Value],
    constants: dict[str, ir.GlobalVariable],
    puts: ir.Function,
    module: ir.Module,
) -> None:
    match stmt:
        case CallPuts(target=name):
            gv = constants[name]
            str_ptr = builder.bitcast(gv, I8.as_pointer())
            builder.call(puts, [str_ptr])
            return
        case ReturnInt(value=v):
            builder.ret(ir.Constant(I32, v))
            return
        case ReturnExpr(value=expr):
            builder.ret(_lower_expr(builder, expr, params, module))
            return
        case If(cond=cond, then_body=then_body, else_body=else_body):
            then_bb = llvm_fn.append_basic_block("then")
            else_bb = llvm_fn.append_basic_block("else")
            builder.cbranch(_lower_expr(builder, cond, params, module), then_bb, else_bb)
            # Both branches are required to terminate (return) in this round;
            # we'll add a merge block when we have an If whose branches fall
            # through.
            builder.position_at_end(then_bb)
            for s in then_body:
                _lower_stmt(builder, s, llvm_fn=llvm_fn, params=params, constants=constants, puts=puts, module=module)
            builder.position_at_end(else_bb)
            for s in else_body:
                _lower_stmt(builder, s, llvm_fn=llvm_fn, params=params, constants=constants, puts=puts, module=module)
            return
    raise ValueError(f"unhandled stmt: {stmt!r}")


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
            cmp = builder.icmp_signed(">=", params[name], ir.Constant(I32, 0))
            _emit_for_enforcement(builder, cmp, enforcement, llvm_fn, module)
            return
        case IntRangeClaim(param=name, min=lo, max=hi):
            val = params[name]
            if lo is not None:
                cmp = builder.icmp_signed(">=", val, ir.Constant(I32, lo))
                _emit_for_enforcement(builder, cmp, enforcement, llvm_fn, module)
            if hi is not None:
                cmp = builder.icmp_signed("<=", val, ir.Constant(I32, hi))
                _emit_for_enforcement(builder, cmp, enforcement, llvm_fn, module)
            return
    raise ValueError(f"unhandled claim: {claim!r}")


def _declare_function(module: ir.Module, fn: Function) -> ir.Function:
    fn_ty = ir.FunctionType(I32, [I32] * len(fn.params))
    return ir.Function(module, fn_ty, name=fn.name)


def _lower_function_body(
    module: ir.Module, fn: Function, *,
    constants: dict, puts: ir.Function, overrides: dict[str, str],
) -> None:
    llvm_fn = module.globals[fn.name]
    for arg, name in zip(llvm_fn.args, fn.params):
        arg.name = name
    params = {name: arg for name, arg in zip(fn.params, llvm_fn.args)}

    builder = ir.IRBuilder(llvm_fn.append_basic_block(name="entry"))
    for claim in fn.claims:
        _lower_claim(builder, claim, params, llvm_fn, module, overrides=overrides)
    for stmt in fn.body:
        _lower_stmt(
            builder, stmt,
            llvm_fn=llvm_fn, params=params, constants=constants, puts=puts, module=module,
        )


def lower(
    program: Program, *,
    target: str | None = None,
    overrides: dict[str, str] | None = None,
) -> ir.Module:
    module = ir.Module(name="cpg")
    module.triple = target or llvm.get_default_triple()
    overrides = overrides or {}

    # `puts` is the only extern we need today; lift it into the graph as an
    # ExternFunction node when a second extern shows up.
    puts = ir.Function(module, ir.FunctionType(I32, [I8.as_pointer()]), name="puts")

    constants: dict[str, ir.GlobalVariable] = {}
    for c in program.constants:
        data = bytearray(c.value.encode("utf-8") + b"\0")
        ty = ir.ArrayType(I8, len(data))
        gv = ir.GlobalVariable(module, ty, name=c.name)
        gv.linkage = "private"
        gv.global_constant = True
        gv.initializer = ir.Constant(ty, data)
        constants[c.name] = gv

    # Pass 1: declare every user function so calls can resolve regardless of order.
    for fn in program.functions:
        _declare_function(module, fn)

    # Pass 2: lower bodies.
    for fn in program.functions:
        _lower_function_body(module, fn, constants=constants, puts=puts, overrides=overrides)

    return module


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
class CompileResult:
    ir_unopt: Path
    ir_opt: Path | None
    object_path: Path
    binary: Path | None


def has_main(program: Program) -> bool:
    return any(fn.name == "main" for fn in program.functions)


def compile_program(
    program: Program,
    *,
    build_dir: Path,
    profile: int = 2,
    link: bool = True,
    target: str | None = None,
    overrides: dict[str, str] | None = None,
) -> CompileResult:
    if not 0 <= profile <= 3:
        raise ValueError(f"profile must be 0..3, got {profile}")
    build_dir.mkdir(parents=True, exist_ok=True)

    # Elaborate: derive lattice claims and merge them into the program before
    # lowering. Override flags (--enforce-lattice etc.) apply uniformly to
    # both stored and derived claims via the override map.
    derived = derive_lattice_claims(program)
    program = elaborate(program, derived)

    module = lower(program, target=target, overrides=overrides)
    ir_unopt = build_dir / "program.unopt.ll"
    ir_unopt.write_text(str(module))

    target_machine = make_target_machine(target=target)
    parsed = parse_and_verify(module)

    ir_opt: Path | None = None
    if profile > 0:
        optimize_module(parsed, target_machine, speed_level=profile)
        ir_opt = build_dir / "program.opt.ll"
        ir_opt.write_text(str(parsed))

    object_path = build_dir / "program.o"
    object_path.write_bytes(target_machine.emit_object(parsed))

    binary: Path | None = None
    if link and has_main(program):
        binary = build_dir / "program"
        cmd = ["clang"]
        if target:
            cmd += ["-target", target]
        cmd += [str(object_path), "-o", str(binary)]
        subprocess.run(cmd, check=True)

    return CompileResult(ir_unopt=ir_unopt, ir_opt=ir_opt, object_path=object_path, binary=binary)
