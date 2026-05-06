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

from typing import Iterable

from .model import (
    Assign,
    BinOp,
    Call,
    CharLit,
    EnumDef,
    EnumInit,
    EnumPayloadField,
    EnumType,
    EnumVariant,
    ExprStmt,
    ExternFunction,
    FieldInit,
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
    U8Type,
    U16Type,
    U32Type,
    U64Type,
    IsizeType,
    UsizeType,
    If,
    IfExpr,
    ImplDef,
    IntLit,
    Let,
    Load,
    LoadField,
    LocalRef,
    Match,
    MatchArm,
    Not,
    NullPtr,
    Param,
    ParamRef,
    Program,
    PtrOffset,
    Break,
    Continue,
    DoWhile,
    Return,
    ReturnExpr,
    ReturnRef,
    ShortCircuitAnd,
    ShortCircuitOr,
    SizeOf,
    Store,
    StoreField,
    StringRef,
    StructDef,
    StructField,
    StructInit,
    StructType,
    TraitCall,
    TraitDef,
    TryExpr,
    TypeParamRef,
    Unreachable,
    VoidType,
    While,
    Widen,
    WithArena,
)


# ---------- Mangling ----------

def _type_to_name(t) -> str:
    """Stable, human-readable string for a fully-concrete type. Used as
    a component in mangled struct/enum names. Must be deterministic and
    distinct for distinct types — `i64` ≠ `i32` ≠ `i8*` ≠ `core.str.String`.
    """
    if isinstance(t, I1Type):
        return "i1"
    if isinstance(t, I8Type):
        return "i8"
    if isinstance(t, I16Type):
        return "i16"
    if isinstance(t, I32Type):
        return "i32"
    if isinstance(t, I64Type):
        return "i64"
    if isinstance(t, U8Type):
        return "u8"
    if isinstance(t, U16Type):
        return "u16"
    if isinstance(t, U32Type):
        return "u32"
    if isinstance(t, U64Type):
        return "u64"
    if isinstance(t, IsizeType):
        return "isize"
    if isinstance(t, UsizeType):
        return "usize"
    if isinstance(t, I8PtrType):
        return "i8ptr"
    if isinstance(t, StructType):
        # Post-mono StructType has empty type_args, so the name is
        # already mangled. Pre-mono uses with type_args shouldn't appear
        # here — substitute resolves them first.
        if t.type_args:
            inner = ",".join(_type_to_name(a) for a in t.type_args)
            return f"{t.name}<{inner}>"
        return t.name
    if isinstance(t, EnumType):
        if t.type_args:
            inner = ",".join(_type_to_name(a) for a in t.type_args)
            return f"{t.name}<{inner}>"
        return t.name
    if isinstance(t, VoidType):
        return "void"
    if isinstance(t, TypeParamRef):
        # Substitute should have resolved this before mangling.
        raise AssertionError(f"unsubstituted TypeParamRef {t.name!r} in mangling")
    raise AssertionError(f"unhandled type for mangling: {t!r}")


def _mangle(template: str, args: tuple) -> str:
    """`Box<i64>`, `List<core.str.String>`, `Result<i64,ParseError>`. The
    raw form is kept literal — angle brackets and commas survive into
    LLVM identified-type names (llvmlite quotes as needed)."""
    inner = ",".join(_type_to_name(a) for a in args)
    return f"{template}<{inner}>"


# ---------- Type substitution ----------

def _substitute_type(t, sub: dict[str, object]):
    """Replace any TypeParamRef in `t` via `sub`. Recursively descends
    into StructType/EnumType `type_args`."""
    if isinstance(t, TypeParamRef):
        if t.name not in sub:
            raise ValueError(
                f"unbound type parameter {t.name!r} (available: {sorted(sub)})"
            )
        return sub[t.name]
    if isinstance(t, StructType):
        if not t.type_args:
            return t
        new_args = tuple(_substitute_type(a, sub) for a in t.type_args)
        return t.model_copy(update={"type_args": new_args})
    if isinstance(t, EnumType):
        if not t.type_args:
            return t
        new_args = tuple(_substitute_type(a, sub) for a in t.type_args)
        return t.model_copy(update={"type_args": new_args})
    # Scalar / void: no params possible.
    return t


# ---------- Reference rewriting (post-substitution) ----------

def _rewrite_type(t):
    """Concrete-types pass: rewrite any `(name, type_args)` to its
    mangled name. Assumes all TypeParamRefs are already substituted."""
    if isinstance(t, StructType) and t.type_args:
        return StructType(name=_mangle(t.name, t.type_args), type_args=())
    if isinstance(t, EnumType) and t.type_args:
        return EnumType(name=_mangle(t.name, t.type_args), type_args=())
    return t


def _walk_types_in_expr(expr, fn):
    """Apply `fn` to every Type reference in `expr` (returns rewritten
    expr). Also handles type-bearing inits like StructInit/EnumInit."""
    if isinstance(expr, IntLit):
        return expr.model_copy(update={"type": fn(expr.type)})
    if isinstance(expr, Load):
        return expr.model_copy(update={
            "ptr":  _walk_types_in_expr(expr.ptr,  fn),
            "type": fn(expr.type),
        })
    if isinstance(expr, SizeOf):
        return expr.model_copy(update={"type": fn(expr.type)})
    if isinstance(expr, Widen):
        # target is IntType — no struct/enum/typeparam.
        return expr.model_copy(update={
            "value": _walk_types_in_expr(expr.value, fn),
        })
    if isinstance(expr, StructInit):
        new_fields = tuple(
            FieldInit(name=fi.name, value=_walk_types_in_expr(fi.value, fn))
            for fi in expr.fields
        )
        if expr.type_args:
            # Apply fn to each type_arg first — this lets substitution
            # passes resolve TypeParamRefs in nested type_args before
            # they get mangled into the enclosing name.
            args = tuple(fn(a) for a in expr.type_args)
            mangled = _mangle(expr.type, args)
            return expr.model_copy(update={
                "type": mangled,
                "type_args": (),
                "fields": new_fields,
            })
        return expr.model_copy(update={"fields": new_fields})
    if isinstance(expr, EnumInit):
        new_fields = tuple(
            FieldInit(name=fi.name, value=_walk_types_in_expr(fi.value, fn))
            for fi in expr.fields
        )
        if expr.type_args:
            args = tuple(fn(a) for a in expr.type_args)
            mangled = _mangle(expr.enum, args)
            return expr.model_copy(update={
                "enum": mangled,
                "type_args": (),
                "fields": new_fields,
            })
        return expr.model_copy(update={"fields": new_fields})
    if isinstance(expr, BinOp):
        return expr.model_copy(update={
            "lhs": _walk_types_in_expr(expr.lhs, fn),
            "rhs": _walk_types_in_expr(expr.rhs, fn),
        })
    if isinstance(expr, ShortCircuitOr):
        return expr.model_copy(update={
            "lhs": _walk_types_in_expr(expr.lhs, fn),
            "rhs": _walk_types_in_expr(expr.rhs, fn),
        })
    if isinstance(expr, ShortCircuitAnd):
        return expr.model_copy(update={
            "lhs": _walk_types_in_expr(expr.lhs, fn),
            "rhs": _walk_types_in_expr(expr.rhs, fn),
        })
    if isinstance(expr, IfExpr):
        return expr.model_copy(update={
            "cond": _walk_types_in_expr(expr.cond, fn),
            "then_value": _walk_types_in_expr(expr.then_value, fn),
            "else_value": _walk_types_in_expr(expr.else_value, fn),
        })
    if isinstance(expr, Not):
        return expr.model_copy(update={
            "operand": _walk_types_in_expr(expr.operand, fn),
        })
    if isinstance(expr, Call):
        new_args = tuple(_walk_types_in_expr(a, fn) for a in expr.args)
        if expr.type_args:
            type_args = tuple(fn(a) for a in expr.type_args)
            mangled = _mangle(expr.function, type_args)
            return expr.model_copy(update={
                "function":  mangled,
                "type_args": (),
                "args":      new_args,
            })
        return expr.model_copy(update={"args": new_args})
    if isinstance(expr, FieldRead):
        return expr.model_copy(update={
            "value": _walk_types_in_expr(expr.value, fn),
        })
    if isinstance(expr, LoadField):
        new_ptr = _walk_types_in_expr(expr.ptr, fn)
        if expr.type_args:
            args = tuple(fn(a) for a in expr.type_args)
            mangled = _mangle(expr.struct_type, args)
            return expr.model_copy(update={
                "ptr": new_ptr,
                "struct_type": mangled,
                "type_args": (),
            })
        return expr.model_copy(update={"ptr": new_ptr})
    if isinstance(expr, PtrOffset):
        return expr.model_copy(update={
            "base":   _walk_types_in_expr(expr.base,   fn),
            "offset": _walk_types_in_expr(expr.offset, fn),
        })
    if isinstance(expr, TryExpr):
        return expr.model_copy(update={
            "value": _walk_types_in_expr(expr.value, fn),
        })
    if isinstance(expr, TraitCall):
        # The trait-call resolution pass runs separately, after all type
        # rewriting is done; here we just walk the dispatch_type and the
        # args so they get substituted/mangled along with everything else.
        return expr.model_copy(update={
            "dispatch_type": fn(expr.dispatch_type),
            "args": tuple(_walk_types_in_expr(a, fn) for a in expr.args),
        })
    # Leaf expressions with no nested Type or Expr.
    if isinstance(expr, (ParamRef, LocalRef, StringRef, NullPtr, CharLit, ReturnRef)):
        return expr
    raise AssertionError(f"unhandled expr in type rewrite: {type(expr).__name__}")


def _walk_types_in_stmt(stmt, fn):
    if isinstance(stmt, ReturnExpr):
        return stmt.model_copy(update={
            "value": _walk_types_in_expr(stmt.value, fn),
        })
    if isinstance(stmt, (Return, Unreachable, Break, Continue)):
        return stmt
    if isinstance(stmt, If):
        return stmt.model_copy(update={
            "cond":      _walk_types_in_expr(stmt.cond, fn),
            "then_body": stmt.then_body.model_copy(update={
                "stmts": tuple(_walk_types_in_stmt(s, fn) for s in stmt.then_body.stmts),
            }),
            "else_body": stmt.else_body.model_copy(update={
                "stmts": tuple(_walk_types_in_stmt(s, fn) for s in stmt.else_body.stmts),
            }),
        })
    if isinstance(stmt, Let):
        return stmt.model_copy(update={
            "type": fn(stmt.type),
            "init": (_walk_types_in_expr(stmt.init, fn) if stmt.init is not None else None),
        })
    if isinstance(stmt, Assign):
        return stmt.model_copy(update={
            "value": _walk_types_in_expr(stmt.value, fn),
        })
    if isinstance(stmt, While):
        return stmt.model_copy(update={
            "cond": _walk_types_in_expr(stmt.cond, fn),
            "body": stmt.body.model_copy(update={
                "stmts": tuple(_walk_types_in_stmt(s, fn) for s in stmt.body.stmts),
            }),
        })
    if isinstance(stmt, DoWhile):
        return stmt.model_copy(update={
            "cond": _walk_types_in_expr(stmt.cond, fn),
            "body": stmt.body.model_copy(update={
                "stmts": tuple(_walk_types_in_stmt(s, fn) for s in stmt.body.stmts),
            }),
        })
    if isinstance(stmt, For):
        return stmt.model_copy(update={
            "lo":   _walk_types_in_expr(stmt.lo, fn),
            "hi":   _walk_types_in_expr(stmt.hi, fn),
            "body": stmt.body.model_copy(update={
                "stmts": tuple(_walk_types_in_stmt(s, fn) for s in stmt.body.stmts),
            }),
        })
    if isinstance(stmt, ExprStmt):
        return stmt.model_copy(update={
            "value": _walk_types_in_expr(stmt.value, fn),
        })
    if isinstance(stmt, FieldSet):
        return stmt.model_copy(update={
            "value": _walk_types_in_expr(stmt.value, fn),
        })
    if isinstance(stmt, Store):
        return stmt.model_copy(update={
            "ptr":   _walk_types_in_expr(stmt.ptr,   fn),
            "value": _walk_types_in_expr(stmt.value, fn),
        })
    if isinstance(stmt, StoreField):
        new_ptr   = _walk_types_in_expr(stmt.ptr,   fn)
        new_value = _walk_types_in_expr(stmt.value, fn)
        if stmt.type_args:
            args = tuple(fn(a) for a in stmt.type_args)
            mangled = _mangle(stmt.struct_type, args)
            return stmt.model_copy(update={
                "ptr": new_ptr,
                "value": new_value,
                "struct_type": mangled,
                "type_args": (),
            })
        return stmt.model_copy(update={"ptr": new_ptr, "value": new_value})
    if isinstance(stmt, WithArena):
        return stmt.model_copy(update={
            "capacity": _walk_types_in_expr(stmt.capacity, fn),
            "body":     stmt.body.model_copy(update={
                "stmts": tuple(_walk_types_in_stmt(s, fn) for s in stmt.body.stmts),
            }),
        })
    if isinstance(stmt, Match):
        new_arms = tuple(
            arm.model_copy(update={
                "body": arm.body.model_copy(update={
                    "stmts": tuple(_walk_types_in_stmt(s, fn) for s in arm.body.stmts),
                }),
            })
            for arm in stmt.arms
        )
        return stmt.model_copy(update={
            "scrutinee": _walk_types_in_expr(stmt.scrutinee, fn),
            "arms":      new_arms,
        })
    raise AssertionError(f"unhandled stmt in type rewrite: {type(stmt).__name__}")


# ---------- Substitute-only walker ----------
#
# Inside a generic function template, every TypeParamRef must be resolved
# to a concrete Type before the rewrite pass runs. The walker is in
# `traversal` (shared with model.py's ImplDef Self substitution); here
# we just thunk it with a sub-dict-driven `type_fn`.

from .traversal import substitute_in_expr as _substitute_in_expr_walker
from .traversal import substitute_in_stmt as _substitute_in_stmt_walker


def _substitute_in_expr(expr, sub):
    return _substitute_in_expr_walker(expr, lambda t: _substitute_type(t, sub))


def _substitute_in_stmt(stmt, sub):
    return _substitute_in_stmt_walker(stmt, lambda t: _substitute_type(t, sub))


# ---------- Discovery: collect all (template, args) instantiations ----------

def _collect_instantiations(t, sink: set):
    """Recurse through a Type and add every (name, args) instantiation
    encountered. Args are normalized into a tuple of `_TypeKey`s for
    set membership."""
    if isinstance(t, StructType) and t.type_args:
        sink.add((t.name, tuple(_type_key(a) for a in t.type_args)))
        for a in t.type_args:
            _collect_instantiations(a, sink)
    elif isinstance(t, EnumType) and t.type_args:
        sink.add((t.name, tuple(_type_key(a) for a in t.type_args)))
        for a in t.type_args:
            _collect_instantiations(a, sink)


def _type_key(t):
    """Hashable, value-equality canonical form of a Type. Used as a
    dedup key for the worklist. Pydantic models are frozen and hashable
    via their dict shape — but Type instances created at different
    times may not be `==` for our purposes if they differ in field
    ordering; safest is to canonicalize through model_dump JSON."""
    # Pydantic's frozen instances ARE hashable by default; rely on that.
    # Fall back to model_dump if needed.
    return t


def _collect_in_expr(expr, sink: set):
    if isinstance(expr, IntLit):
        _collect_instantiations(expr.type, sink)
    elif isinstance(expr, SizeOf):
        _collect_instantiations(expr.type, sink)
    elif isinstance(expr, Load):
        _collect_instantiations(expr.type, sink)
        _collect_in_expr(expr.ptr, sink)
    elif isinstance(expr, StructInit):
        if expr.type_args:
            sink.add((expr.type, tuple(_type_key(a) for a in expr.type_args)))
            for a in expr.type_args:
                _collect_instantiations(a, sink)
        for fi in expr.fields:
            _collect_in_expr(fi.value, sink)
    elif isinstance(expr, EnumInit):
        if expr.type_args:
            sink.add((expr.enum, tuple(_type_key(a) for a in expr.type_args)))
            for a in expr.type_args:
                _collect_instantiations(a, sink)
        for fi in expr.fields:
            _collect_in_expr(fi.value, sink)
    elif isinstance(expr, BinOp):
        _collect_in_expr(expr.lhs, sink)
        _collect_in_expr(expr.rhs, sink)
    elif isinstance(expr, ShortCircuitOr):
        _collect_in_expr(expr.lhs, sink)
        _collect_in_expr(expr.rhs, sink)
    elif isinstance(expr, ShortCircuitAnd):
        _collect_in_expr(expr.lhs, sink)
        _collect_in_expr(expr.rhs, sink)
    elif isinstance(expr, IfExpr):
        _collect_in_expr(expr.cond, sink)
        _collect_in_expr(expr.then_value, sink)
        _collect_in_expr(expr.else_value, sink)
    elif isinstance(expr, Not):
        _collect_in_expr(expr.operand, sink)
    elif isinstance(expr, Call):
        if expr.type_args:
            sink.add((expr.function, tuple(_type_key(a) for a in expr.type_args)))
            for a in expr.type_args:
                _collect_instantiations(a, sink)
        for a in expr.args:
            _collect_in_expr(a, sink)
    elif isinstance(expr, FieldRead):
        _collect_in_expr(expr.value, sink)
    elif isinstance(expr, LoadField):
        _collect_in_expr(expr.ptr, sink)
    elif isinstance(expr, PtrOffset):
        _collect_in_expr(expr.base,   sink)
        _collect_in_expr(expr.offset, sink)
    elif isinstance(expr, TryExpr):
        _collect_in_expr(expr.value, sink)
    elif isinstance(expr, Widen):
        _collect_in_expr(expr.value, sink)
    elif isinstance(expr, TraitCall):
        _collect_instantiations(expr.dispatch_type, sink)
        for a in expr.args:
            _collect_in_expr(a, sink)
    elif isinstance(expr, (ParamRef, LocalRef, StringRef, NullPtr, CharLit, ReturnRef)):
        return
    else:
        raise AssertionError(f"unhandled expr in collect: {type(expr).__name__}")


def _collect_in_stmt(stmt, sink: set):
    if isinstance(stmt, ReturnExpr):
        _collect_in_expr(stmt.value, sink)
    elif isinstance(stmt, (Return, Unreachable, Break, Continue)):
        return
    elif isinstance(stmt, If):
        _collect_in_expr(stmt.cond, sink)
        for s in stmt.then_body.stmts:
            _collect_in_stmt(s, sink)
        for s in stmt.else_body.stmts:
            _collect_in_stmt(s, sink)
    elif isinstance(stmt, Let):
        _collect_instantiations(stmt.type, sink)
        if stmt.init is not None:
            _collect_in_expr(stmt.init, sink)
    elif isinstance(stmt, Assign):
        _collect_in_expr(stmt.value, sink)
    elif isinstance(stmt, While):
        _collect_in_expr(stmt.cond, sink)
        for s in stmt.body.stmts:
            _collect_in_stmt(s, sink)
    elif isinstance(stmt, DoWhile):
        for s in stmt.body.stmts:
            _collect_in_stmt(s, sink)
        _collect_in_expr(stmt.cond, sink)
    elif isinstance(stmt, For):
        _collect_in_expr(stmt.lo, sink)
        _collect_in_expr(stmt.hi, sink)
        for s in stmt.body.stmts:
            _collect_in_stmt(s, sink)
    elif isinstance(stmt, ExprStmt):
        _collect_in_expr(stmt.value, sink)
    elif isinstance(stmt, FieldSet):
        _collect_in_expr(stmt.value, sink)
    elif isinstance(stmt, Store):
        _collect_in_expr(stmt.ptr,   sink)
        _collect_in_expr(stmt.value, sink)
    elif isinstance(stmt, StoreField):
        _collect_in_expr(stmt.ptr,   sink)
        _collect_in_expr(stmt.value, sink)
    elif isinstance(stmt, WithArena):
        _collect_in_expr(stmt.capacity, sink)
        for s in stmt.body.stmts:
            _collect_in_stmt(s, sink)
    elif isinstance(stmt, Match):
        _collect_in_expr(stmt.scrutinee, sink)
        for arm in stmt.arms:
            for s in arm.body.stmts:
                _collect_in_stmt(s, sink)
    else:
        raise AssertionError(f"unhandled stmt in collect: {type(stmt).__name__}")


# ---------- Impl promotion + trait-call resolution ----------

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

    v1 restriction: `for_type` must be a `StructType` or `EnumType`,
    and each position in `for_type.type_args` is either a TypeParamRef
    (one of the impl's type-params) or a concrete type. No nested
    patterns like `Box<List<T>>` for v1 — `for_type` is one level deep.
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
                f"StructType/EnumType for_types supported in v1"
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
    if isinstance(expr, Load):
        return expr.model_copy(update={"ptr": _resolve_trait_calls_in_expr(expr.ptr, impl_index)})
    if isinstance(expr, SizeOf):
        return expr
    if isinstance(expr, Widen):
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


# ---------- Main pass ----------

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
