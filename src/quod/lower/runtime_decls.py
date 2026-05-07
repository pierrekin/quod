"""Runtime declarations: per-module LLVM Function declarations for the
intrinsics and libc symbols the lowering needs (`llvm.assume`, `abort`,
`atoll`), plus the enforcement-aware predicate emitter and the
process-global LLVM-target initialization.
"""

from __future__ import annotations

from llvmlite import binding as llvm
from llvmlite import ir

from quod.lower.types import I1, I8, I64


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
