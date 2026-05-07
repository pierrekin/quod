"""Function-shell glue: per-function LLVM declaration (`_declare_function`,
`_declare_extern`) and per-function body lowering
(`_lower_function_body`). This is the bridge between the program-level
`lower()` orchestrator and the statement walker.
"""

from __future__ import annotations

from llvmlite import ir

from quod.lower.claims import _lower_claim
from quod.lower.stmt import (
    _collect_local_bindings,
    _collect_local_qtypes,
    _lower_stmt,
)
from quod.lower.types import _type_to_llvm
from quod.model import (
    Block,
    EnumDef,
    ExternFunction,
    Function,
    StructDef,
)
from quod.predicate.proof import predicate_uses_return


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
    fn_returns: dict[str, object] | None = None,
) -> None:
    # `quod.lower` operates on layer C only. Family wrappers and
    # family-extension statements must be stripped/lowered by the
    # c-family lowering pass first; refuse here rather than crash deep
    # in the statement walker.
    if not isinstance(fn.body, Block):
        raise ValueError(
            f"function {fn.name!r} body is wrapped in {type(fn.body).__name__!r} — "
            f"layer C must be pure core. Run the c-family lowering pass "
            f"(lower/c_family.py) before lower."
        )
    llvm_fn = module.globals[fn.name]
    for arg, p in zip(llvm_fn.args, fn.params):
        arg.name = p.name
    params = {p.name: arg for p, arg in zip(fn.params, llvm_fn.args)}

    # Split claims by scope. A predicate is a postcondition iff it
    # references ReturnRef; otherwise it's a precondition and is
    # injected at function entry.
    entry_claims = tuple(c for c in fn.claims if not predicate_uses_return(c.expr))
    return_claims = tuple(c for c in fn.claims if predicate_uses_return(c.expr))

    entry_bb = llvm_fn.append_basic_block(name="entry")
    builder = ir.IRBuilder(entry_bb)

    # Allocas at the very top of entry, before any other instruction. mem2reg
    # promotes them to SSA values during the optimize pass.
    locals_: dict[str, ir.AllocaInstr] = {}
    for name, ty in _collect_local_bindings(fn.body, struct_tys, enum_tys, enum_defs):
        assert name not in params, (
            f"validator invariant: local {name!r} shadows parameter of "
            f"{fn.name!r} — quod.validate should have caught this"
        )
        locals_[name] = builder.alloca(ty, name=name)

    # Parallel quod-side type lookup for Cast lowering (the LLVM-side
    # locals_ has no signedness info).
    local_qtypes = _collect_local_qtypes(fn.body)

    for claim in entry_claims:
        _lower_claim(builder, claim, params, llvm_fn, module, overrides=overrides)

    for stmt in fn.body.stmts:
        _lower_stmt(
            builder, stmt,
            llvm_fn=llvm_fn, params=params, locals_=locals_, entry_bb=entry_bb,
            constants=constants, module=module,
            return_claims=return_claims, overrides=overrides,
            extern_sigs=extern_sigs,
            struct_defs=struct_defs, struct_tys=struct_tys,
            enum_defs=enum_defs, enum_tys=enum_tys,
            fn=fn, local_qtypes=local_qtypes, fn_returns=fn_returns,
        )

    # Void functions get an implicit `ret void` if the body falls through;
    # non-void functions left without a terminator surface as a verifier
    # error (correct — the user owes a return).
    if not builder.block.is_terminated and isinstance(
        llvm_fn.function_type.return_type, ir.VoidType
    ):
        builder.ret_void()

    # Sweep: any basic block left without a terminator is unreachable from
    # the well-formed flow (e.g. the dead exit_bb after a `while(true) { ...
    # return ... }` loop, where every body path returns and subsequent
    # statements never attach to exit_bb). LLVM rejects unterminated blocks;
    # `unreachable` documents the intent without introducing wrong behavior.
    for bb in llvm_fn.basic_blocks:
        if not bb.is_terminated:
            ir.IRBuilder(bb).unreachable()
