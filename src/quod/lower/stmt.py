"""Statement lowering: `_collect_local_bindings` (entry-block alloca
discovery) and `_lower_stmt` (the per-statement walker that emits
basic blocks and stores). Calls into `_lower_expr` for embedded
expressions and into `_emit_return_claims` at every ret site.
"""

from __future__ import annotations

from llvmlite import ir

from quod.lower.claims import _emit_return_claims
from quod.lower.expr import _lower_expr
from quod.lower.types import (
    I8,
    I32,
    _coerce_int_lit,
    _type_to_llvm,
    _variant_struct_ty,
)
from quod.model import (
    Assign,
    Break,
    Continue,
    DoWhile,
    EnumDef,
    ExprStmt,
    ExternFunction,
    FieldSet,
    For,
    If,
    Let,
    Match,
    Return,
    ReturnExpr,
    Store,
    StoreField,
    StructDef,
    Unreachable,
    While,
)


def _collect_local_bindings(
    body,
    struct_tys: dict[str, "ir.IdentifiedStructType"],
    enum_tys: dict[str, "ir.IdentifiedStructType"],
    enum_defs: dict[str, EnumDef],
) -> list[tuple[str, "ir.Type"]]:
    """Pre-walk the body and return every (name, llvm_type) pair introduced by
    `Let`, `For`, or a `Match` arm binding. Allocas for these are emitted at
    the top of the function's entry block, the canonical mem2reg layout:
    `alloca` lives in entry; `store` happens at the binding point. Names must
    be unique within the function (no shadowing — match arms with the same
    binding name across arms collide too; rename them differently for now).
    `body` is a `Block`."""
    out: list[tuple[str, ir.Type]] = []
    seen: set[str] = set()

    def visit(blk) -> None:
        for s in blk.stmts:
            match s:
                case Let(name=name, type=ty):
                    assert name not in seen, (
                        f"validator invariant: local {name!r} declared twice "
                        f"— quod.validate should have caught this"
                    )
                    seen.add(name)
                    out.append((name, _type_to_llvm(ty, struct_tys, enum_tys)))
                case For(var=var, type=ty, body=for_body):
                    assert var not in seen, (
                        f"validator invariant: for-loop var {var!r} conflicts "
                        f"with another local — quod.validate should have caught this"
                    )
                    seen.add(var)
                    out.append((var, _type_to_llvm(ty, struct_tys, enum_tys)))
                    visit(for_body)
                case If(then_body=t, else_body=e):
                    visit(t); visit(e)
                case While(body=w_body) | DoWhile(body=w_body):
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
    visit(body)
    return out


def _collect_local_qtypes(body) -> dict[str, "object"]:
    """Parallel to `_collect_local_bindings`, but yields the quod-side
    `Type` for each binding (not the lowered LLVM type). Used by `Cast`
    lowering to determine source signedness — the LLVM type alone has
    no signedness, but quod's IXType / UXType partition does. Match-arm
    bindings are handled inline (not pre-collected) — same as
    `_collect_local_bindings`."""
    out: dict[str, object] = {}

    def visit(blk) -> None:
        for s in blk.stmts:
            match s:
                case Let(name=name, type=ty):
                    out[name] = ty
                case For(var=var, type=ty, body=for_body):
                    out[var] = ty
                    visit(for_body)
                case If(then_body=t, else_body=e):
                    visit(t); visit(e)
                case While(body=w_body) | DoWhile(body=w_body):
                    visit(w_body)
                case Match(arms=arms):
                    for arm in arms:
                        visit(arm.body)
    visit(body)
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
    loop_stack: list[tuple[ir.Block, ir.Block]] | None = None,
    fn=None,
    local_qtypes: dict[str, object] | None = None,
    fn_returns: dict[str, object] | None = None,
) -> None:
    """Lower a statement. `return_claims` are emitted as llvm.assume / runtime
    check at every ret, so callers (after inlining) see the bound. The
    `loop_stack` carries (continue_target, break_target) for each enclosing
    loop, so `Break` / `Continue` can branch to the right basic block.

    `fn`, `local_qtypes`, and `fn_returns` are quod-side type lookups
    needed by `Cast` lowering to recover source signedness (the LLVM
    type alone is signedness-free)."""
    if loop_stack is None:
        loop_stack = []

    def lower_expr(e):
        return _lower_expr(
            builder, e, params, module,
            constants=constants, extern_sigs=extern_sigs, locals_=locals_,
            struct_defs=struct_defs, struct_tys=struct_tys,
            enum_defs=enum_defs, enum_tys=enum_tys,
            fn=fn, local_qtypes=local_qtypes, fn_returns=fn_returns,
        )

    def lower_body(body):
        for s in body.stmts:
            _lower_stmt(
                builder, s, llvm_fn=llvm_fn, params=params, locals_=locals_,
                entry_bb=entry_bb, constants=constants, module=module,
                return_claims=return_claims, overrides=overrides,
                extern_sigs=extern_sigs,
                struct_defs=struct_defs, struct_tys=struct_tys,
                enum_defs=enum_defs, enum_tys=enum_tys,
                loop_stack=loop_stack,
                fn=fn, local_qtypes=local_qtypes, fn_returns=fn_returns,
            )

    match stmt:
        case ReturnExpr(value=expr):
            ret_val = lower_expr(expr)
            _emit_return_claims(builder, ret_val, return_claims, llvm_fn, module, params, overrides)
            builder.ret(ret_val)
            return
        case Return():
            assert isinstance(llvm_fn.function_type.return_type, ir.VoidType), (
                f"validator invariant: bare Return in non-void function "
                f"{llvm_fn.name!r} — quod.validate should have caught this"
            )
            builder.ret_void()
            return
        case Unreachable():
            # Synthesized e.g. by the C ingest for fall-through off a non-main
            # int-returning function (UB per C99 §6.9.1/12). If the current
            # block is already terminated — say, after an `if (c) return 0;
            # else return 1;` — there's nothing to do; emitting an instruction
            # would error.
            if not builder.block.is_terminated:
                builder.unreachable()
            return
        case Break():
            assert loop_stack, (
                "validator invariant: Break outside a loop — "
                "quod.validate should have caught this"
            )
            _, break_target = loop_stack[-1]
            builder.branch(break_target)
            return
        case Continue():
            assert loop_stack, (
                "validator invariant: Continue outside a loop — "
                "quod.validate should have caught this"
            )
            continue_target, _ = loop_stack[-1]
            builder.branch(continue_target)
            return
        case ExprStmt(value=expr):
            lower_expr(expr)
            return
        case Let(name=name, init=init):
            # Alloca was pre-emitted at the entry block. Store the init
            # value if there is one; for an uninitialized local
            # (init=None), the alloca's contents are undef until a
            # later Assign — the validator ensures no read happens
            # before that.
            if init is None:
                return
            dest_ty = locals_[name].type.pointee
            init_val = lower_expr(_coerce_int_lit(init, dest_ty))
            builder.store(init_val, locals_[name])
            return
        case Assign(name=name, value=v):
            assert name in locals_, (
                f"validator invariant: assign to undeclared local {name!r} "
                f"— quod.validate should have caught this"
            )
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
        case StoreField(ptr=p, struct_type=tname, name=fname, value=v):
            base = lower_expr(p)
            if not (isinstance(base.type, ir.PointerType) and base.type.pointee == I8):
                raise ValueError(f"store_field base must be i8*, got {base.type}")
            sd = struct_defs.get(tname)
            sty = struct_tys.get(tname)
            if sd is None or sty is None:
                raise ValueError(f"store_field on undefined struct {tname!r}")
            idx = sd.field_index(fname)
            dest_ty = _type_to_llvm(sd.fields[idx].type, struct_tys, enum_tys)
            val = lower_expr(_coerce_int_lit(v, dest_ty))
            casted = builder.bitcast(base, sty.as_pointer())
            field_ptr = builder.gep(
                casted, [ir.Constant(I32, 0), ir.Constant(I32, idx)],
                inbounds=True,
            )
            builder.store(val, field_ptr)
            return
        case FieldSet(local=lname, name=fname, value=v):
            assert lname in locals_, (
                f"validator invariant: field-set on undeclared local "
                f"{lname!r} — quod.validate should have caught this"
            )
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
            for s in then_body.stmts:
                _lower_stmt(
                    builder, s, llvm_fn=llvm_fn, params=params, locals_=locals_,
                    entry_bb=entry_bb, constants=constants, module=module,
                    return_claims=return_claims, overrides=overrides,
                    extern_sigs=extern_sigs,
                    struct_defs=struct_defs, struct_tys=struct_tys,
                    enum_defs=enum_defs, enum_tys=enum_tys,
                    loop_stack=loop_stack,
                    fn=fn, local_qtypes=local_qtypes, fn_returns=fn_returns,
                )
            if not builder.block.is_terminated:
                builder.branch(ensure_merge())

            builder.position_at_end(else_bb)
            for s in else_body.stmts:
                _lower_stmt(
                    builder, s, llvm_fn=llvm_fn, params=params, locals_=locals_,
                    entry_bb=entry_bb, constants=constants, module=module,
                    return_claims=return_claims, overrides=overrides,
                    extern_sigs=extern_sigs,
                    struct_defs=struct_defs, struct_tys=struct_tys,
                    enum_defs=enum_defs, enum_tys=enum_tys,
                    loop_stack=loop_stack,
                    fn=fn, local_qtypes=local_qtypes, fn_returns=fn_returns,
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
            # Push (continue_target=header, break_target=exit) for the
            # body's nested Break/Continue. Pop when the body finishes.
            loop_stack.append((header_bb, exit_bb))
            try:
                lower_body(body)
            finally:
                loop_stack.pop()
            if not builder.block.is_terminated:
                builder.branch(header_bb)

            builder.position_at_end(exit_bb)
            return
        case DoWhile(body=body, cond=cond):
            # Body runs unconditionally first, then we test cond and
            # loop back. `continue` jumps to the cond block (per C
            # semantics); `break` jumps to the exit.
            body_bb = llvm_fn.append_basic_block("dowhile.body")
            cond_bb = llvm_fn.append_basic_block("dowhile.cond")
            exit_bb = llvm_fn.append_basic_block("dowhile.exit")
            builder.branch(body_bb)

            builder.position_at_end(body_bb)
            loop_stack.append((cond_bb, exit_bb))
            try:
                lower_body(body)
            finally:
                loop_stack.pop()
            if not builder.block.is_terminated:
                builder.branch(cond_bb)

            builder.position_at_end(cond_bb)
            cond_val = lower_expr(cond)
            builder.cbranch(cond_val, body_bb, exit_bb)

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
            inc_bb = llvm_fn.append_basic_block("for.inc")
            exit_bb = llvm_fn.append_basic_block("for.exit")
            builder.branch(header_bb)

            builder.position_at_end(header_bb)
            cur = builder.load(alloca)
            cmp = builder.icmp_signed("<", cur, hi_val)
            builder.cbranch(cmp, body_bb, exit_bb)

            builder.position_at_end(body_bb)
            # Continue jumps to the inc step (matches C for-loop
            # semantics); break jumps to the exit.
            loop_stack.append((inc_bb, exit_bb))
            try:
                lower_body(body)
            finally:
                loop_stack.pop()
            if not builder.block.is_terminated:
                builder.branch(inc_bb)
            builder.position_at_end(inc_bb)
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
            # Spill the enum value to memory so we can read the tag and
            # bitcast the payload to a per-variant LLVM struct pointer.
            # All field reads in arms go through this same alloca.
            scrut_alloca = builder.alloca(scrut_ty)
            builder.store(scrut_val, scrut_alloca)
            tag_ptr = builder.gep(
                scrut_alloca, [ir.Constant(I32, 0), ir.Constant(I32, 0)],
            )
            tag = builder.load(tag_ptr)
            payload_ptr = builder.gep(
                scrut_alloca, [ir.Constant(I32, 0), ir.Constant(I32, 1)],
            )
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
                for s in arm_obj.body.stmts:
                    _lower_stmt(
                        builder, s, llvm_fn=llvm_fn, params=params, locals_=locals_,
                        entry_bb=entry_bb, constants=constants, module=module,
                        return_claims=return_claims, overrides=overrides,
                        extern_sigs=extern_sigs,
                        struct_defs=struct_defs, struct_tys=struct_tys,
                        enum_defs=enum_defs, enum_tys=enum_tys,
                        fn=fn, local_qtypes=local_qtypes, fn_returns=fn_returns,
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
                # Bind payload fields with arm-scoped allocas. Bitcast the
                # enum's payload bytes to this variant's LLVM struct type;
                # extract each field via GEP+load and store into the
                # binding's alloca. Save/restore locals_ across the body
                # so bindings don't leak past the arm.
                variant_ty = _variant_struct_ty(var, struct_tys, enum_tys)
                variant_ptr = builder.bitcast(payload_ptr, variant_ty.as_pointer())
                saved: dict[str, ir.AllocaInstr | None] = {}
                saved_qtypes: dict[str, object | None] = {}
                for i, (binding, field) in enumerate(zip(arm.bindings, var.fields)):
                    saved[binding] = locals_.get(binding)
                    field_ll_ty = _type_to_llvm(field.type, struct_tys, enum_tys)
                    binding_alloca = builder.alloca(field_ll_ty, name=binding)
                    field_ptr = builder.gep(
                        variant_ptr,
                        [ir.Constant(I32, 0), ir.Constant(I32, i)],
                    )
                    field_val = builder.load(field_ptr)
                    builder.store(field_val, binding_alloca)
                    locals_[binding] = binding_alloca
                    if local_qtypes is not None:
                        saved_qtypes[binding] = local_qtypes.get(binding)
                        local_qtypes[binding] = field.type
                lower_arm_body(arm)
                for b, prior in saved.items():
                    if prior is None:
                        locals_.pop(b, None)
                    else:
                        locals_[b] = prior
                if local_qtypes is not None:
                    for b, prior in saved_qtypes.items():
                        if prior is None:
                            local_qtypes.pop(b, None)
                        else:
                            local_qtypes[b] = prior
            if wildcard_arm is not None:
                builder.position_at_end(wildcard_bb)
                lower_arm_body(wildcard_arm)
            if end_bb is not None:
                builder.position_at_end(end_bb)
            return
    # `c.*` extension nodes (CStyleFor, etc.) reach `quod.lower` only when a
    # layer-B program slipped past the c-family lowering pass. Surface a
    # readable error pointing at the right place to fix it.
    if getattr(stmt, "kind", None) and str(stmt.kind).startswith("c."):
        raise ValueError(
            f"lower refuses {stmt.kind!r}: layer C must be pure core "
            f"quod. Run the c-family lowering pass (lower/c_family.py) "
            f"before lower."
        )
    raise ValueError(f"unhandled stmt: {stmt!r}")
