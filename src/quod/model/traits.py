"""Trait + impl definitions.

`ImplDef`'s post-construction validator eagerly substitutes `SelfType`
→ `for_type` inside every method (params, return, body), so by the
time any consumer sees an impl, no `SelfType` survives.
"""

from __future__ import annotations

from pydantic import model_validator

from quod.model.base import _Node
from quod.model.top_level import Function, Param, TypeParam
from quod.model.types import EnumType, ReturnType, SelfType, StructType, Type


class TraitMethodSig(_Node):
    """One method signature in a `TraitDef`. No body; impls supply the
    body. Param/return types may reference `SelfType` (the implementing
    type) and any `TypeParamRef`s declared by the trait itself
    (currently always empty — generic traits aren't yet supported)."""
    name: str
    params: tuple[Param, ...] = ()
    return_type: ReturnType


class TraitDef(_Node):
    """A trait: a name plus a set of method signatures any conforming
    `ImplDef` must provide. Pure declaration; no runtime cost.

    A `<T: TraitName>` bound on a generic type parameter constrains
    instantiations to types that have an `impl TraitName for ...`
    visible at mono time. The mono pass rewrites `TraitCall` nodes to
    direct `Call`s of the impl's mangled method symbols.
    """
    name: str
    methods: tuple[TraitMethodSig, ...]


def _substitute_self_in_type(t, for_type):
    """Eagerly substitute `SelfType` → `for_type` inside a Type tree.
    Used by `ImplDef`'s post-construction validator so impls store
    Self-free methods.
    """
    if isinstance(t, SelfType):
        return for_type
    if isinstance(t, StructType) and t.type_args:
        return t.model_copy(update={
            "type_args": tuple(_substitute_self_in_type(a, for_type) for a in t.type_args),
        })
    if isinstance(t, EnumType) and t.type_args:
        return t.model_copy(update={
            "type_args": tuple(_substitute_self_in_type(a, for_type) for a in t.type_args),
        })
    return t


class ImplDef(_Node):
    """`impl<...> <trait> for <for_type> { <methods> }`.

    Provides concrete bodies for the named trait's methods on the named
    type. Multiple impls of distinct traits for the same type are
    allowed; two impls of the same trait for the same type are not
    (coherence — checked at Program-level validation).

    On construction, every `SelfType` inside `methods` is rewritten to
    `for_type` — the lowerer and the mono pass never see Self.

    `type_params`: a generic impl introduces type variables that appear
    in `for_type` (e.g. `impl<T> Drop for Box<T>`). When the
    corresponding template (`Box`) is instantiated, the mono pass
    generates one concrete impl per instantiation by binding the
    impl's type-params from positions in `for_type.type_args`. Current
    restriction: each `for_type.type_args[i]` must be either a
    `TypeParamRef` naming one of the impl's `type_params`, or a
    concrete type — no nested patterns like `Box<List<T>>`.

    Methods are stored as full `Function`s so they can be promoted to
    top-level by the monomorphization pass with mangled names like
    `<for_type>::<method>` (e.g., `Arena::alloc`, `Box<i64>::drop`).
    """
    trait: str
    type_params: tuple[TypeParam, ...] = ()
    for_type: Type
    methods: tuple[Function, ...]

    @model_validator(mode="after")
    def _resolve_self(self) -> "ImplDef":
        # Eagerly substitute Self → for_type in every method's params,
        # return_type, AND body. After this validator runs, no SelfType
        # survives in the impl methods; the lowerer and the mono pass
        # never see Self.
        from quod.traversal import substitute_in_stmt
        type_fn = lambda t: _substitute_self_in_type(t, self.for_type)
        new_methods = []
        for fn in self.methods:
            new_params = tuple(
                Param(name=p.name, type=type_fn(p.type)) for p in fn.params
            )
            new_return = type_fn(fn.return_type)
            new_stmts = tuple(substitute_in_stmt(s, type_fn) for s in fn.body.stmts)
            new_body = fn.body.model_copy(update={"stmts": new_stmts})
            new_methods.append(fn.model_copy(update={
                "params":      new_params,
                "return_type": new_return,
                "body":        new_body,
            }))
        object.__setattr__(self, "methods", tuple(new_methods))
        return self
