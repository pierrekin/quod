"""Impl promotion + trait-call resolution.

Two concerns colocated:
- Pre-pass: promote `impl Trait for T { fn f(...) ... }` to top-level
  Functions named `<T>::<f>`, and instantiate generic impls on demand
  alongside their target template's monomorphization.
- Post-pass: rewrite every remaining TraitCall to a direct Call on the
  resolved impl-method symbol, once `dispatch_type` is concrete.
"""

from __future__ import annotations

from ..model import (
    Assign,
    BinOp,
    Break,
    Call,
    CharLit,
    Continue,
    DoWhile,
    EnumInit,
    EnumType,
    ExprStmt,
    FieldInit,
    FieldRead,
    FieldSet,
    For,
    Function,
    If,
    IfExpr,
    ImplDef,
    IntLit,
    Let,
    Load,
    LoadField,
    LocalRef,
    Match,
    Not,
    NullPtr,
    Param,
    ParamRef,
    PtrOffset,
    Return,
    ReturnExpr,
    ReturnRef,
    ShortCircuitAnd,
    ShortCircuitOr,
    SizeOf,
    Store,
    StoreField,
    StringRef,
    Cast,
    FloatLit,
    FNeg,
    StructInit,
    StructType,
    TraitCall,
    TryExpr,
    TypeParamRef,
    Unreachable,
    While,
    WithArena,
)

from .mangling import _type_to_name
from .rewriting import _rewrite_type, _walk_types_in_stmt
from .substitution import _substitute_in_stmt, _substitute_type


def _impl_method_name(for_type, method_name: str) -> str:
    """The mangled symbol for an impl method: e.g. `Arena::alloc`."""
    return f"{_type_to_name(for_type)}::{method_name}"


def _build_impl_index(impls) -> dict[tuple[str, str], ImplDef]:
    """Index impls by (trait_name, _type_to_name(for_type)). Used by
    trait-call resolution to find the impl that satisfies a given
    dispatch. Concrete impls only — generic impls are indexed
    separately and instantiated on demand."""
    index: dict[tuple[str, str], ImplDef] = {}
    for impl in impls:
        if impl.type_params:
            continue  # generic; handled by _index_generic_impls
        key = (impl.trait, _type_to_name(impl.for_type))
        if key in index:
            raise ValueError(
                f"duplicate impl for trait {impl.trait!r} on type "
                f"{key[1]!r}: only one allowed (coherence)"
            )
        index[key] = impl
    return index


def _promote_impls(impls) -> list[Function]:
    """Each non-generic `impl Trait for T { fn f(...) ... }` produces
    one top-level Function named `<T>::<f>`. The bodies pass through
    verbatim; `ImplDef`'s validator already substituted Self.

    Generic impls (with `type_params`) are NOT promoted here — their
    bodies still contain TypeParamRefs that the lowerer can't handle.
    The mono pass instantiates them on demand when their target
    template is monomorphized.
    """
    out: list[Function] = []
    for impl in impls:
        if impl.type_params:
            continue
        for method in impl.methods:
            name = _impl_method_name(impl.for_type, method.name)
            out.append(method.model_copy(update={"name": name}))
    return out


def _index_generic_impls(impls) -> dict[str, list[ImplDef]]:
    """Index generic impls by the template name in their `for_type`.
    `impl<T> Drop for Box<T>` indexes under "Box". When `Box<i64>` is
    monomorphized, mono looks up "Box" here, computes the substitution
    (T → i64), and instantiates the impl.

    Current restriction: `for_type` must be a `StructType` or `EnumType`,
    and each position in `for_type.type_args` is either a TypeParamRef
    (one of the impl's type-params) or a concrete type. No nested
    patterns like `Box<List<T>>` — `for_type` is one level deep.
    """
    out: dict[str, list[ImplDef]] = {}
    for impl in impls:
        if not impl.type_params:
            continue
        for_type = impl.for_type
        if isinstance(for_type, (StructType, EnumType)):
            out.setdefault(for_type.name, []).append(impl)
        else:
            raise ValueError(
                f"generic impl {impl.trait!r} for {for_type!r}: only "
                f"StructType/EnumType for_types are supported"
            )
    return out


def _instantiate_generic_impl(g_impl: ImplDef, args: tuple, impl_index, out_fns) -> None:
    """Concretize a generic impl `impl<...> Trait for Tmpl<...>` for one
    instantiation `(Tmpl, args)`. Builds the substitution by walking
    `g_impl.for_type.type_args`: TypeParamRef positions bind to the
    corresponding `args[i]`; concrete positions in for_type are
    sanity-checked (must equal args[i] or it's a non-match — caller
    should've filtered by template name).

    Adds the resulting concrete impl to `impl_index` and promotes its
    methods to mangled top-level Functions in `out_fns`. Also returns
    the concrete impl so the caller can discover further
    instantiations from substituted bodies.
    """
    # Build substitution from for_type.type_args ↔ args.
    impl_type_args = g_impl.for_type.type_args
    if len(impl_type_args) != len(args):
        raise ValueError(
            f"generic impl {g_impl.trait!r} for {g_impl.for_type.name!r}: "
            f"for_type has {len(impl_type_args)} type_args but instantiation "
            f"has {len(args)} args"
        )
    sub: dict[str, object] = {}
    for slot, concrete in zip(impl_type_args, args):
        if isinstance(slot, TypeParamRef):
            if slot.name in sub and sub[slot.name] != concrete:
                raise ValueError(
                    f"generic impl {g_impl.trait!r}: type-param "
                    f"{slot.name!r} bound twice to different types"
                )
            sub[slot.name] = concrete
        elif slot != concrete:
            # Concrete slot in for_type doesn't match this instantiation.
            return None
    # Concrete for_type for this instantiation.
    concrete_for_type = _substitute_type(g_impl.for_type, sub)
    concrete_for_type = _rewrite_type(concrete_for_type)
    concrete_target = _type_to_name(concrete_for_type)

    # Coherence check.
    coherence_key = (g_impl.trait, concrete_target)
    if coherence_key in impl_index:
        # Already instantiated (or a non-generic impl shadowed it). Skip.
        return None

    # Substitute + rewrite the methods. Each becomes a top-level Function.
    type_fn = lambda t: _rewrite_type(_substitute_type(t, sub))

    # Build the substituted+rewritten methods. Two flavors are needed:
    # (a) the impl_index entry keeps method names UNMANGLED (the trait
    #     method names) so `_resolve_trait_call` can look them up by
    #     trait-method-name; (b) the top-level Functions promoted into
    #     `out_fns` use the mangled name `<for_type>::<method>` so the
    #     emitted Call resolves to a unique symbol at lower time.
    impl_methods_for_index: list[Function] = []
    for method in g_impl.methods:
        new_params = tuple(
            Param(name=p.name, type=type_fn(p.type)) for p in method.params
        )
        new_return = type_fn(method.return_type)
        new_stmts = tuple(_walk_types_in_stmt(s, _rewrite_type)
                          for s in _substitute_type_in_method_body(method.body, sub))
        new_body = method.body.model_copy(update={"stmts": new_stmts})
        method_mangled = f"{concrete_target}::{method.name}"
        # (a) impl-index entry: original trait-method name on the inner Function.
        impl_methods_for_index.append(method.model_copy(update={
            "params":      new_params,
            "return_type": new_return,
            "body":        new_body,
        }))
        # (b) promoted top-level function with mangled symbol.
        out_fns[method_mangled] = method.model_copy(update={
            "name":        method_mangled,
            "params":      new_params,
            "return_type": new_return,
            "body":        new_body,
        })

    impl_index[coherence_key] = ImplDef.model_construct(
        trait=g_impl.trait,
        for_type=concrete_for_type,
        methods=tuple(impl_methods_for_index),
    )


def _substitute_type_in_method_body(body, sub):
    """Substitute TypeParamRefs in a method body using the shared
    traversal walker. Returns the substituted statement tuple (with
    nested type_args still un-mangled — caller mangles in a second
    pass). Takes a Block; returns a stmts tuple."""
    return tuple(_substitute_in_stmt(s, sub) for s in body.stmts)


def _resolve_trait_call(expr: TraitCall, impl_index) -> Call:
    """Rewrite `TraitCall` → `Call` after all type substitution and
    mangling are done. dispatch_type must be concrete by this point;
    `impl_index` resolves it to the impl whose method symbol we call."""
    target = _type_to_name(expr.dispatch_type)
    key = (expr.trait, target)
    if key not in impl_index:
        raise ValueError(
            f"no impl of trait {expr.trait!r} for type {target!r} "
            f"(known impls: {sorted(impl_index)})"
        )
    impl = impl_index[key]
    # Sanity: the trait method must exist on the impl.
    method_names = {m.name for m in impl.methods}
    if expr.method not in method_names:
        raise ValueError(
            f"impl {impl.trait!r} for {target!r} has no method "
            f"{expr.method!r} (has: {sorted(method_names)})"
        )
    return Call(
        function=_impl_method_name(impl.for_type, expr.method),
        args=expr.args,
    )


def _resolve_trait_calls_in_expr(expr, impl_index):
    if isinstance(expr, TraitCall):
        # First resolve any nested TraitCalls in the args, then rewrite this one.
        resolved_args = tuple(_resolve_trait_calls_in_expr(a, impl_index) for a in expr.args)
        return _resolve_trait_call(expr.model_copy(update={"args": resolved_args}), impl_index)
    if isinstance(expr, IntLit):
        return expr
    if isinstance(expr, FloatLit):
        return expr
    if isinstance(expr, FNeg):
        return expr.model_copy(update={"operand": _resolve_trait_calls_in_expr(expr.operand, impl_index)})
    if isinstance(expr, Load):
        return expr.model_copy(update={"ptr": _resolve_trait_calls_in_expr(expr.ptr, impl_index)})
    if isinstance(expr, SizeOf):
        return expr
    if isinstance(expr, Cast):
        return expr.model_copy(update={"value": _resolve_trait_calls_in_expr(expr.value, impl_index)})
    if isinstance(expr, StructInit):
        return expr.model_copy(update={
            "fields": tuple(
                FieldInit(name=fi.name, value=_resolve_trait_calls_in_expr(fi.value, impl_index))
                for fi in expr.fields
            ),
        })
    if isinstance(expr, EnumInit):
        return expr.model_copy(update={
            "fields": tuple(
                FieldInit(name=fi.name, value=_resolve_trait_calls_in_expr(fi.value, impl_index))
                for fi in expr.fields
            ),
        })
    if isinstance(expr, BinOp):
        return expr.model_copy(update={
            "lhs": _resolve_trait_calls_in_expr(expr.lhs, impl_index),
            "rhs": _resolve_trait_calls_in_expr(expr.rhs, impl_index),
        })
    if isinstance(expr, ShortCircuitOr):
        return expr.model_copy(update={
            "lhs": _resolve_trait_calls_in_expr(expr.lhs, impl_index),
            "rhs": _resolve_trait_calls_in_expr(expr.rhs, impl_index),
        })
    if isinstance(expr, ShortCircuitAnd):
        return expr.model_copy(update={
            "lhs": _resolve_trait_calls_in_expr(expr.lhs, impl_index),
            "rhs": _resolve_trait_calls_in_expr(expr.rhs, impl_index),
        })
    if isinstance(expr, IfExpr):
        return expr.model_copy(update={
            "cond": _resolve_trait_calls_in_expr(expr.cond, impl_index),
            "then_value": _resolve_trait_calls_in_expr(expr.then_value, impl_index),
            "else_value": _resolve_trait_calls_in_expr(expr.else_value, impl_index),
        })
    if isinstance(expr, Not):
        return expr.model_copy(update={
            "operand": _resolve_trait_calls_in_expr(expr.operand, impl_index),
        })
    if isinstance(expr, Call):
        return expr.model_copy(update={
            "args": tuple(_resolve_trait_calls_in_expr(a, impl_index) for a in expr.args),
        })
    if isinstance(expr, FieldRead):
        return expr.model_copy(update={
            "value": _resolve_trait_calls_in_expr(expr.value, impl_index),
        })
    if isinstance(expr, LoadField):
        return expr.model_copy(update={
            "ptr": _resolve_trait_calls_in_expr(expr.ptr, impl_index),
        })
    if isinstance(expr, PtrOffset):
        return expr.model_copy(update={
            "base":   _resolve_trait_calls_in_expr(expr.base,   impl_index),
            "offset": _resolve_trait_calls_in_expr(expr.offset, impl_index),
        })
    if isinstance(expr, TryExpr):
        return expr.model_copy(update={
            "value": _resolve_trait_calls_in_expr(expr.value, impl_index),
        })
    if isinstance(expr, (ParamRef, LocalRef, StringRef, NullPtr, CharLit, ReturnRef)):
        return expr
    raise AssertionError(f"unhandled expr in trait-call resolve: {type(expr).__name__}")


def _resolve_trait_calls_in_stmt(stmt, impl_index):
    if isinstance(stmt, ReturnExpr):
        return stmt.model_copy(update={
            "value": _resolve_trait_calls_in_expr(stmt.value, impl_index),
        })
    if isinstance(stmt, (Return, Unreachable, Break, Continue)):
        return stmt
    if isinstance(stmt, If):
        return stmt.model_copy(update={
            "cond":      _resolve_trait_calls_in_expr(stmt.cond, impl_index),
            "then_body": stmt.then_body.model_copy(update={
                "stmts": tuple(_resolve_trait_calls_in_stmt(s, impl_index) for s in stmt.then_body.stmts),
            }),
            "else_body": stmt.else_body.model_copy(update={
                "stmts": tuple(_resolve_trait_calls_in_stmt(s, impl_index) for s in stmt.else_body.stmts),
            }),
        })
    if isinstance(stmt, Let):
        if stmt.init is None:
            return stmt
        return stmt.model_copy(update={
            "init": _resolve_trait_calls_in_expr(stmt.init, impl_index),
        })
    if isinstance(stmt, Assign):
        return stmt.model_copy(update={
            "value": _resolve_trait_calls_in_expr(stmt.value, impl_index),
        })
    if isinstance(stmt, While):
        return stmt.model_copy(update={
            "cond": _resolve_trait_calls_in_expr(stmt.cond, impl_index),
            "body": stmt.body.model_copy(update={
                "stmts": tuple(_resolve_trait_calls_in_stmt(s, impl_index) for s in stmt.body.stmts),
            }),
        })
    if isinstance(stmt, DoWhile):
        return stmt.model_copy(update={
            "cond": _resolve_trait_calls_in_expr(stmt.cond, impl_index),
            "body": stmt.body.model_copy(update={
                "stmts": tuple(_resolve_trait_calls_in_stmt(s, impl_index) for s in stmt.body.stmts),
            }),
        })
    if isinstance(stmt, For):
        return stmt.model_copy(update={
            "lo":   _resolve_trait_calls_in_expr(stmt.lo, impl_index),
            "hi":   _resolve_trait_calls_in_expr(stmt.hi, impl_index),
            "body": stmt.body.model_copy(update={
                "stmts": tuple(_resolve_trait_calls_in_stmt(s, impl_index) for s in stmt.body.stmts),
            }),
        })
    if isinstance(stmt, ExprStmt):
        return stmt.model_copy(update={
            "value": _resolve_trait_calls_in_expr(stmt.value, impl_index),
        })
    if isinstance(stmt, FieldSet):
        return stmt.model_copy(update={
            "value": _resolve_trait_calls_in_expr(stmt.value, impl_index),
        })
    if isinstance(stmt, Store):
        return stmt.model_copy(update={
            "ptr":   _resolve_trait_calls_in_expr(stmt.ptr,   impl_index),
            "value": _resolve_trait_calls_in_expr(stmt.value, impl_index),
        })
    if isinstance(stmt, StoreField):
        return stmt.model_copy(update={
            "ptr":   _resolve_trait_calls_in_expr(stmt.ptr,   impl_index),
            "value": _resolve_trait_calls_in_expr(stmt.value, impl_index),
        })
    if isinstance(stmt, WithArena):
        return stmt.model_copy(update={
            "capacity": _resolve_trait_calls_in_expr(stmt.capacity, impl_index),
            "body":     stmt.body.model_copy(update={
                "stmts": tuple(_resolve_trait_calls_in_stmt(s, impl_index) for s in stmt.body.stmts),
            }),
        })
    if isinstance(stmt, Match):
        return stmt.model_copy(update={
            "scrutinee": _resolve_trait_calls_in_expr(stmt.scrutinee, impl_index),
            "arms": tuple(
                arm.model_copy(update={
                    "body": arm.body.model_copy(update={
                        "stmts": tuple(_resolve_trait_calls_in_stmt(s, impl_index) for s in arm.body.stmts),
                    }),
                })
                for arm in stmt.arms
            ),
        })
    raise AssertionError(f"unhandled stmt in trait-call resolve: {type(stmt).__name__}")
