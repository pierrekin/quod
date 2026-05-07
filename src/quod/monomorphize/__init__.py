"""Monomorphization pass — concrete instantiations of generic types.

Walks the program for every `StructType` / `EnumType` / `StructInit` /
`EnumInit` whose `type_args` is non-empty, generates one fresh nominal
`StructDef` / `EnumDef` per unique `(template, args)` tuple, and rewrites
every reference to use the mangled name with empty `type_args`.

Postconditions on the returned Program:

- No `StructDef.type_params` or `EnumDef.type_params` is non-empty.
- No `StructType.type_args` / `EnumType.type_args` is non-empty.
- No `StructInit.type_args` / `EnumInit.type_args` is non-empty.
- No `TypeParamRef` remains in any field, param, return, or
  expression position.

The resulting Program is what the lowerer consumes. The generic
templates are dropped from the output entirely — they never lower.
"""

from __future__ import annotations

from ..model import (
    EnumDef,
    EnumPayloadField,
    EnumVariant,
    Function,
    Param,
    Program,
    StructDef,
    StructField,
)

from .discovery import _collect_in_stmt, _collect_instantiations
from .mangling import _mangle, _type_to_name
from .rewriting import _rewrite_type, _walk_types_in_stmt
from .substitution import _substitute_in_stmt, _substitute_type
from .traits import (
    _build_impl_index,
    _index_generic_impls,
    _instantiate_generic_impl,
    _promote_impls,
    _resolve_trait_calls_in_stmt,
)


def monomorphize(program: Program) -> Program:
    """Rewrite generic instantiations into fresh nominal defs.

    Pre-pass: each `ImplDef`'s methods become top-level Functions with
    mangled names like `Arena::alloc`. The impl_index built from the
    same impls is later used to resolve TraitCalls.

    Post-pass: any remaining `TraitCall` (which by this point has a
    concrete `dispatch_type`) is rewritten to a direct `Call` to the
    matching impl method.
    """
    impl_index = _build_impl_index(program.impls)
    promoted_fns = _promote_impls(program.impls)
    generic_impls_by_template = _index_generic_impls(program.impls)
    if program.impls or promoted_fns:
        program = program.model_copy(update={
            "functions": program.functions + tuple(promoted_fns),
            "impls":     (),
        })

    generic_structs = {sd.name: sd for sd in program.structs if sd.type_params}
    generic_enums   = {ed.name: ed for ed in program.enums   if ed.type_params}
    generic_fns     = {fn.name: fn for fn in program.functions if fn.type_params}

    # No fast-path return: even programs with no generics may still have
    # trait calls to resolve (and a malformed TraitCall without a
    # matching impl needs to surface a clear error, not silently pass).
    # The cost of running the full pass on a non-generic, non-trait
    # program is microseconds.

    # Carry non-generic defs forward unchanged; collect generic defs as
    # templates to instantiate from.
    out_structs: dict[str, StructDef] = {
        sd.name: sd for sd in program.structs if not sd.type_params
    }
    out_enums: dict[str, EnumDef] = {
        ed.name: ed for ed in program.enums   if not ed.type_params
    }
    out_fns: dict[str, Function] = {}  # only monomorphized instances; non-generic carry through later

    # Seed the worklist by walking every type ref in the input program,
    # excluding generic templates' own bodies (TypeParamRefs there
    # aren't valid instantiations until substituted).
    seeds: set[tuple[str, tuple]] = set()
    for sd in program.structs:
        if sd.type_params:
            continue
        for f in sd.fields:
            _collect_instantiations(f.type, seeds)
    for ed in program.enums:
        if ed.type_params:
            continue
        for v in ed.variants:
            for f in v.fields:
                _collect_instantiations(f.type, seeds)
    for fn in program.functions:
        if fn.type_params:
            continue
        _collect_instantiations(fn.return_type, seeds)
        for p in fn.params:
            _collect_instantiations(p.type, seeds)
        for stmt in fn.body.stmts:
            _collect_in_stmt(stmt, seeds)
    for ext in program.externs:
        _collect_instantiations(ext.return_type, seeds)
        for t in ext.param_types:
            _collect_instantiations(t, seeds)

    pending: list[tuple[str, tuple]] = list(seeds)
    seen: set[tuple[str, tuple]] = set(seeds)

    def push(key):
        if key not in seen:
            seen.add(key)
            pending.append(key)

    def _check_bounds(kind: str, template_name: str, type_params, args) -> None:
        """Reject an instantiation whose concrete type lacks an
        `impl <bound> for <T>` for any bounded type parameter. The error
        names the binding site so the user sees `<i64> doesn't implement
        Allocator` at the call/use site, not later when a TraitCall
        inside the body tries to dispatch."""
        for tp, arg in zip(type_params, args):
            if tp.bound is None:
                continue
            arg_name = _type_to_name(arg)
            if (tp.bound, arg_name) not in impl_index:
                raise ValueError(
                    f"in instantiation of {kind} {template_name}<...>: type "
                    f"parameter {tp.name!r} is bound by {tp.bound!r}, but "
                    f"no `impl {tp.bound} for {arg_name}` is in scope"
                )

    while pending:
        template, args_keys = pending.pop()
        args = tuple(args_keys)
        mangled = _mangle(template, args)
        if mangled in out_structs or mangled in out_enums or mangled in out_fns:
            continue

        if template in generic_structs:
            sd = generic_structs[template]
            if len(sd.type_params) != len(args):
                raise ValueError(
                    f"generic struct {template!r} takes {len(sd.type_params)} "
                    f"type args, got {len(args)}: {args}"
                )
            _check_bounds("struct", template, sd.type_params, args)
            sub = dict(zip([tp.name for tp in sd.type_params], args))
            new_fields = []
            for f in sd.fields:
                substituted = _substitute_type(f.type, sub)
                fresh: set[tuple[str, tuple]] = set()
                _collect_instantiations(substituted, fresh)
                for k in fresh:
                    push(k)
                rewritten = _rewrite_type(substituted)
                new_fields.append(StructField(name=f.name, type=rewritten))
            out_structs[mangled] = StructDef(
                name=mangled, type_params=(), fields=tuple(new_fields),
            )
            # Generic impls targeting this template (e.g. `impl<T> Drop
            # for Box<T>`) get instantiated alongside.
            for g_impl in generic_impls_by_template.get(template, ()):
                _instantiate_generic_impl(g_impl, args, impl_index, out_fns)
        elif template in generic_enums:
            ed = generic_enums[template]
            if len(ed.type_params) != len(args):
                raise ValueError(
                    f"generic enum {template!r} takes {len(ed.type_params)} "
                    f"type args, got {len(args)}: {args}"
                )
            _check_bounds("enum", template, ed.type_params, args)
            sub = dict(zip([tp.name for tp in ed.type_params], args))
            new_variants = []
            for v in ed.variants:
                new_v_fields = []
                for f in v.fields:
                    substituted = _substitute_type(f.type, sub)
                    fresh: set[tuple[str, tuple]] = set()
                    _collect_instantiations(substituted, fresh)
                    for k in fresh:
                        push(k)
                    rewritten = _rewrite_type(substituted)
                    new_v_fields.append(EnumPayloadField(name=f.name, type=rewritten))
                new_variants.append(EnumVariant(name=v.name, fields=tuple(new_v_fields)))
            out_enums[mangled] = EnumDef(
                name=mangled, type_params=(), variants=tuple(new_variants),
            )
            # Generic impls targeting this enum's template — same as the
            # struct path.
            for g_impl in generic_impls_by_template.get(template, ()):
                _instantiate_generic_impl(g_impl, args, impl_index, out_fns)
        elif template in generic_fns:
            fn = generic_fns[template]
            if len(fn.type_params) != len(args):
                raise ValueError(
                    f"generic function {template!r} takes {len(fn.type_params)} "
                    f"type args, got {len(args)}: {args}"
                )
            _check_bounds("function", template, fn.type_params, args)
            sub = dict(zip([tp.name for tp in fn.type_params], args))
            # Three-pass on the function body: substitute → collect → rewrite.
            sub_return = _substitute_type(fn.return_type, sub)
            sub_params = tuple(
                Param(name=p.name, type=_substitute_type(p.type, sub)) for p in fn.params
            )
            sub_body = tuple(_substitute_in_stmt(s, sub) for s in fn.body.stmts)
            # Discover instantiations from substituted (still has type_args).
            fresh: set[tuple[str, tuple]] = set()
            _collect_instantiations(sub_return, fresh)
            for p in sub_params:
                _collect_instantiations(p.type, fresh)
            for stmt in sub_body:
                _collect_in_stmt(stmt, fresh)
            for k in fresh:
                push(k)
            # Rewrite to mangled names.
            rewritten_body = tuple(_walk_types_in_stmt(s, _rewrite_type) for s in sub_body)
            rewritten_params = tuple(
                Param(name=p.name, type=_rewrite_type(p.type)) for p in sub_params
            )
            rewritten_return = _rewrite_type(sub_return)
            out_fns[mangled] = fn.model_copy(update={
                "name":        mangled,
                "type_params": (),
                "params":      rewritten_params,
                "return_type": rewritten_return,
                "body":        fn.body.model_copy(update={"stmts": rewritten_body}),
                # Claims propagate as-is — they reference parameter names,
                # which are unchanged. If a claim references a type-param-typed
                # parameter, the claim's semantics travel with the
                # monomorphized instance untouched. (Lattice analysis runs
                # AFTER mono so it sees the concrete types.)
            })
        else:
            raise ValueError(
                f"reference to unknown generic template {template!r} "
                f"with type_args {args}"
            )

    # Final program-wide rewrite: every concrete reference (non-generic
    # functions, externs, the struct/enum bodies we just generated) gets
    # any leftover `(name, type_args)` references mangled to their final form.
    rewritten_structs: dict[str, StructDef] = {}
    for name, sd in out_structs.items():
        new_fields = tuple(
            StructField(name=f.name, type=_rewrite_type(f.type))
            for f in sd.fields
        )
        rewritten_structs[name] = sd.model_copy(update={"fields": new_fields})

    rewritten_enums: dict[str, EnumDef] = {}
    for name, ed in out_enums.items():
        new_variants = tuple(
            EnumVariant(
                name=v.name,
                fields=tuple(
                    EnumPayloadField(name=f.name, type=_rewrite_type(f.type))
                    for f in v.fields
                ),
            )
            for v in ed.variants
        )
        rewritten_enums[name] = ed.model_copy(update={"variants": new_variants})

    # Non-generic functions get the rewrite pass; generic templates are dropped.
    nongeneric_rewritten = tuple(
        fn.model_copy(update={
            "return_type": _rewrite_type(fn.return_type),
            "params": tuple(
                Param(name=p.name, type=_rewrite_type(p.type)) for p in fn.params
            ),
            "body": fn.body.model_copy(update={
                "stmts": tuple(_walk_types_in_stmt(s, _rewrite_type) for s in fn.body.stmts),
            }),
        })
        for fn in program.functions
        if not fn.type_params
    )
    new_functions = nongeneric_rewritten + tuple(out_fns.values())

    new_externs = tuple(
        ext.model_copy(update={
            "return_type": _rewrite_type(ext.return_type),
            "param_types": tuple(_rewrite_type(t) for t in ext.param_types),
        })
        for ext in program.externs
    )

    # After all type-rewriting, resolve every remaining TraitCall to a
    # direct Call on the impl method's mangled symbol. Always run, even
    # if impl_index is empty — that way a TraitCall without a matching
    # impl surfaces the clear `no impl of trait <X> for <Y>` error
    # rather than silently passing through to the lowerer.
    new_functions = tuple(
        fn.model_copy(update={
            "body": fn.body.model_copy(update={
                "stmts": tuple(_resolve_trait_calls_in_stmt(s, impl_index) for s in fn.body.stmts),
            }),
        })
        for fn in new_functions
    )

    return program.model_copy(update={
        "structs": tuple(rewritten_structs.values()),
        "enums":   tuple(rewritten_enums.values()),
        "functions": new_functions,
        "externs":   new_externs,
    })


__all__ = ["monomorphize"]
