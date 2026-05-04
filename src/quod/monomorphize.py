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
    If,
    IntLit,
    Let,
    Load,
    LoadField,
    LocalRef,
    Match,
    MatchArm,
    NullPtr,
    Param,
    ParamRef,
    Program,
    PtrOffset,
    Return,
    ReturnExpr,
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
        return expr.model_copy(update={
            "ptr": _walk_types_in_expr(expr.ptr, fn),
        })
    if isinstance(expr, PtrOffset):
        return expr.model_copy(update={
            "base":   _walk_types_in_expr(expr.base,   fn),
            "offset": _walk_types_in_expr(expr.offset, fn),
        })
    if isinstance(expr, TryExpr):
        return expr.model_copy(update={
            "value": _walk_types_in_expr(expr.value, fn),
        })
    # Leaf expressions with no nested Type or Expr.
    if isinstance(expr, (ParamRef, LocalRef, StringRef, NullPtr, CharLit)):
        return expr
    raise AssertionError(f"unhandled expr in type rewrite: {type(expr).__name__}")


def _walk_types_in_stmt(stmt, fn):
    if isinstance(stmt, ReturnExpr):
        return stmt.model_copy(update={
            "value": _walk_types_in_expr(stmt.value, fn),
        })
    if isinstance(stmt, Return):
        return stmt
    if isinstance(stmt, Unreachable):
        return stmt
    if isinstance(stmt, If):
        return stmt.model_copy(update={
            "cond":      _walk_types_in_expr(stmt.cond, fn),
            "then_body": tuple(_walk_types_in_stmt(s, fn) for s in stmt.then_body),
            "else_body": tuple(_walk_types_in_stmt(s, fn) for s in stmt.else_body),
        })
    if isinstance(stmt, Let):
        return stmt.model_copy(update={
            "type": fn(stmt.type),
            "init": _walk_types_in_expr(stmt.init, fn),
        })
    if isinstance(stmt, Assign):
        return stmt.model_copy(update={
            "value": _walk_types_in_expr(stmt.value, fn),
        })
    if isinstance(stmt, While):
        return stmt.model_copy(update={
            "cond": _walk_types_in_expr(stmt.cond, fn),
            "body": tuple(_walk_types_in_stmt(s, fn) for s in stmt.body),
        })
    if isinstance(stmt, For):
        return stmt.model_copy(update={
            "lo":   _walk_types_in_expr(stmt.lo, fn),
            "hi":   _walk_types_in_expr(stmt.hi, fn),
            "body": tuple(_walk_types_in_stmt(s, fn) for s in stmt.body),
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
        return stmt.model_copy(update={
            "ptr":   _walk_types_in_expr(stmt.ptr,   fn),
            "value": _walk_types_in_expr(stmt.value, fn),
        })
    if isinstance(stmt, WithArena):
        return stmt.model_copy(update={
            "capacity": _walk_types_in_expr(stmt.capacity, fn),
            "body":     tuple(_walk_types_in_stmt(s, fn) for s in stmt.body),
        })
    if isinstance(stmt, Match):
        new_arms = tuple(
            arm.model_copy(update={
                "body": tuple(_walk_types_in_stmt(s, fn) for s in arm.body),
            })
            for arm in stmt.arms
        )
        return stmt.model_copy(update={
            "scrutinee": _walk_types_in_expr(stmt.scrutinee, fn),
            "arms":      new_arms,
        })
    raise AssertionError(f"unhandled stmt in type rewrite: {type(stmt).__name__}")


# ---------- Substitute-only walker (for generic-function bodies) ----------
#
# Inside a generic function template, every TypeParamRef must be resolved
# to a concrete Type before the rewrite pass runs. This walker does
# substitution only — it does NOT mangle StructType/EnumType.type_args
# nor Init/Call type_args. After it runs, the body has concrete types
# but still carries non-empty type_args at instantiation sites; the
# subsequent collect-then-rewrite pass discovers those and mangles them.

def _substitute_in_expr(expr, sub):
    if isinstance(expr, IntLit):
        return expr.model_copy(update={"type": _substitute_type(expr.type, sub)})
    if isinstance(expr, Load):
        return expr.model_copy(update={
            "ptr":  _substitute_in_expr(expr.ptr, sub),
            "type": _substitute_type(expr.type, sub),
        })
    if isinstance(expr, SizeOf):
        return expr.model_copy(update={"type": _substitute_type(expr.type, sub)})
    if isinstance(expr, Widen):
        return expr.model_copy(update={
            "value": _substitute_in_expr(expr.value, sub),
        })
    if isinstance(expr, StructInit):
        new_fields = tuple(
            FieldInit(name=fi.name, value=_substitute_in_expr(fi.value, sub))
            for fi in expr.fields
        )
        new_type_args = tuple(_substitute_type(a, sub) for a in expr.type_args)
        return expr.model_copy(update={
            "type_args": new_type_args,
            "fields":    new_fields,
        })
    if isinstance(expr, EnumInit):
        new_fields = tuple(
            FieldInit(name=fi.name, value=_substitute_in_expr(fi.value, sub))
            for fi in expr.fields
        )
        new_type_args = tuple(_substitute_type(a, sub) for a in expr.type_args)
        return expr.model_copy(update={
            "type_args": new_type_args,
            "fields":    new_fields,
        })
    if isinstance(expr, BinOp):
        return expr.model_copy(update={
            "lhs": _substitute_in_expr(expr.lhs, sub),
            "rhs": _substitute_in_expr(expr.rhs, sub),
        })
    if isinstance(expr, ShortCircuitOr):
        return expr.model_copy(update={
            "lhs": _substitute_in_expr(expr.lhs, sub),
            "rhs": _substitute_in_expr(expr.rhs, sub),
        })
    if isinstance(expr, ShortCircuitAnd):
        return expr.model_copy(update={
            "lhs": _substitute_in_expr(expr.lhs, sub),
            "rhs": _substitute_in_expr(expr.rhs, sub),
        })
    if isinstance(expr, Call):
        new_args = tuple(_substitute_in_expr(a, sub) for a in expr.args)
        new_type_args = tuple(_substitute_type(a, sub) for a in expr.type_args)
        return expr.model_copy(update={
            "type_args": new_type_args,
            "args":      new_args,
        })
    if isinstance(expr, FieldRead):
        return expr.model_copy(update={
            "value": _substitute_in_expr(expr.value, sub),
        })
    if isinstance(expr, LoadField):
        return expr.model_copy(update={
            "ptr": _substitute_in_expr(expr.ptr, sub),
        })
    if isinstance(expr, PtrOffset):
        return expr.model_copy(update={
            "base":   _substitute_in_expr(expr.base, sub),
            "offset": _substitute_in_expr(expr.offset, sub),
        })
    if isinstance(expr, TryExpr):
        return expr.model_copy(update={
            "value": _substitute_in_expr(expr.value, sub),
        })
    if isinstance(expr, (ParamRef, LocalRef, StringRef, NullPtr, CharLit)):
        return expr
    raise AssertionError(f"unhandled expr in substitute: {type(expr).__name__}")


def _substitute_in_stmt(stmt, sub):
    if isinstance(stmt, ReturnExpr):
        return stmt.model_copy(update={"value": _substitute_in_expr(stmt.value, sub)})
    if isinstance(stmt, (Return, Unreachable)):
        return stmt
    if isinstance(stmt, If):
        return stmt.model_copy(update={
            "cond":      _substitute_in_expr(stmt.cond, sub),
            "then_body": tuple(_substitute_in_stmt(s, sub) for s in stmt.then_body),
            "else_body": tuple(_substitute_in_stmt(s, sub) for s in stmt.else_body),
        })
    if isinstance(stmt, Let):
        return stmt.model_copy(update={
            "type": _substitute_type(stmt.type, sub),
            "init": _substitute_in_expr(stmt.init, sub),
        })
    if isinstance(stmt, Assign):
        return stmt.model_copy(update={"value": _substitute_in_expr(stmt.value, sub)})
    if isinstance(stmt, While):
        return stmt.model_copy(update={
            "cond": _substitute_in_expr(stmt.cond, sub),
            "body": tuple(_substitute_in_stmt(s, sub) for s in stmt.body),
        })
    if isinstance(stmt, For):
        return stmt.model_copy(update={
            "lo":   _substitute_in_expr(stmt.lo, sub),
            "hi":   _substitute_in_expr(stmt.hi, sub),
            "body": tuple(_substitute_in_stmt(s, sub) for s in stmt.body),
        })
    if isinstance(stmt, ExprStmt):
        return stmt.model_copy(update={"value": _substitute_in_expr(stmt.value, sub)})
    if isinstance(stmt, FieldSet):
        return stmt.model_copy(update={"value": _substitute_in_expr(stmt.value, sub)})
    if isinstance(stmt, Store):
        return stmt.model_copy(update={
            "ptr":   _substitute_in_expr(stmt.ptr,   sub),
            "value": _substitute_in_expr(stmt.value, sub),
        })
    if isinstance(stmt, StoreField):
        return stmt.model_copy(update={
            "ptr":   _substitute_in_expr(stmt.ptr,   sub),
            "value": _substitute_in_expr(stmt.value, sub),
        })
    if isinstance(stmt, WithArena):
        return stmt.model_copy(update={
            "capacity": _substitute_in_expr(stmt.capacity, sub),
            "body":     tuple(_substitute_in_stmt(s, sub) for s in stmt.body),
        })
    if isinstance(stmt, Match):
        new_arms = tuple(
            arm.model_copy(update={
                "body": tuple(_substitute_in_stmt(s, sub) for s in arm.body),
            })
            for arm in stmt.arms
        )
        return stmt.model_copy(update={
            "scrutinee": _substitute_in_expr(stmt.scrutinee, sub),
            "arms":      new_arms,
        })
    raise AssertionError(f"unhandled stmt in substitute: {type(stmt).__name__}")


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
    elif isinstance(expr, (ParamRef, LocalRef, StringRef, NullPtr, CharLit)):
        return
    else:
        raise AssertionError(f"unhandled expr in collect: {type(expr).__name__}")


def _collect_in_stmt(stmt, sink: set):
    if isinstance(stmt, ReturnExpr):
        _collect_in_expr(stmt.value, sink)
    elif isinstance(stmt, (Return, Unreachable)):
        return
    elif isinstance(stmt, If):
        _collect_in_expr(stmt.cond, sink)
        for s in stmt.then_body:
            _collect_in_stmt(s, sink)
        for s in stmt.else_body:
            _collect_in_stmt(s, sink)
    elif isinstance(stmt, Let):
        _collect_instantiations(stmt.type, sink)
        _collect_in_expr(stmt.init, sink)
    elif isinstance(stmt, Assign):
        _collect_in_expr(stmt.value, sink)
    elif isinstance(stmt, While):
        _collect_in_expr(stmt.cond, sink)
        for s in stmt.body:
            _collect_in_stmt(s, sink)
    elif isinstance(stmt, For):
        _collect_in_expr(stmt.lo, sink)
        _collect_in_expr(stmt.hi, sink)
        for s in stmt.body:
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
        for s in stmt.body:
            _collect_in_stmt(s, sink)
    elif isinstance(stmt, Match):
        _collect_in_expr(stmt.scrutinee, sink)
        for arm in stmt.arms:
            for s in arm.body:
                _collect_in_stmt(s, sink)
    else:
        raise AssertionError(f"unhandled stmt in collect: {type(stmt).__name__}")


# ---------- Main pass ----------

def monomorphize(program: Program) -> Program:
    """Rewrite generic instantiations into fresh nominal defs."""
    generic_structs = {sd.name: sd for sd in program.structs if sd.type_params}
    generic_enums   = {ed.name: ed for ed in program.enums   if ed.type_params}
    generic_fns     = {fn.name: fn for fn in program.functions if fn.type_params}

    # If no generics anywhere, short-circuit.
    has_generics = (
        any(sd.type_params for sd in program.structs)
        or any(ed.type_params for ed in program.enums)
        or any(fn.type_params for fn in program.functions)
    )
    if not has_generics:
        return program

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
        for stmt in fn.body:
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
            sub = dict(zip(sd.type_params, args))
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
        elif template in generic_enums:
            ed = generic_enums[template]
            if len(ed.type_params) != len(args):
                raise ValueError(
                    f"generic enum {template!r} takes {len(ed.type_params)} "
                    f"type args, got {len(args)}: {args}"
                )
            sub = dict(zip(ed.type_params, args))
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
        elif template in generic_fns:
            fn = generic_fns[template]
            if len(fn.type_params) != len(args):
                raise ValueError(
                    f"generic function {template!r} takes {len(fn.type_params)} "
                    f"type args, got {len(args)}: {args}"
                )
            sub = dict(zip(fn.type_params, args))
            # Three-pass on the function body: substitute → collect → rewrite.
            sub_return = _substitute_type(fn.return_type, sub)
            sub_params = tuple(
                Param(name=p.name, type=_substitute_type(p.type, sub)) for p in fn.params
            )
            sub_body = tuple(_substitute_in_stmt(s, sub) for s in fn.body)
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
                "body":        rewritten_body,
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
            "body": tuple(_walk_types_in_stmt(s, _rewrite_type) for s in fn.body),
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

    return program.model_copy(update={
        "structs": tuple(rewritten_structs.values()),
        "enums":   tuple(rewritten_enums.values()),
        "functions": new_functions,
        "externs":   new_externs,
    })
