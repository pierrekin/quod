"""Substitution walker shared between the monomorphizer and the
ImplDef Self-substitution validator.

Both passes need to walk an entire function body, applying a
`Type → Type` rewriting function at every Type position — including
nested type_args slots inside `StructInit`, `EnumInit`, `Call`, and
`TraitCall`. The walker recurses through every Expr/Stmt shape and
calls the user-supplied `type_fn` on the leaves it cares about; the
caller is responsible for the recursion *into* compound types
(StructType.type_args, EnumType.type_args), since how that recurses
varies by use case.

Use cases:

- Monomorphization: `type_fn = lambda t: _substitute_type(t, sub)`
  resolves TypeParamRefs from a substitution dict.
- ImplDef Self resolution: `type_fn = lambda t: _substitute_self_in_type(t, for_type)`
  rewrites `SelfType` → the impl's `for_type`.

This walker does NOT mangle. The mono pass calls `_walk_types_in_*`
with `_rewrite_type` afterwards to mangle `(template, args)` into
final names.
"""

from __future__ import annotations

from typing import Callable

from .model import (
    Assign,
    BinOp,
    Call,
    CharLit,
    EnumInit,
    ExprStmt,
    FieldInit,
    FieldRead,
    FieldSet,
    For,
    If,
    IntLit,
    Let,
    Load,
    LoadField,
    LocalRef,
    Match,
    NullPtr,
    ParamRef,
    PtrOffset,
    Return,
    ReturnExpr,
    ShortCircuitAnd,
    ShortCircuitOr,
    SizeOf,
    Store,
    StoreField,
    StringRef,
    StructInit,
    TraitCall,
    TryExpr,
    Unreachable,
    Widen,
    While,
    WithArena,
)


def substitute_in_expr(expr, type_fn: Callable):
    """Recurse `expr`, applying `type_fn` to every Type-valued field
    (including nested type_args on Init/Call/TraitCall nodes)."""
    if isinstance(expr, IntLit):
        return expr.model_copy(update={"type": type_fn(expr.type)})
    if isinstance(expr, Load):
        return expr.model_copy(update={
            "ptr":  substitute_in_expr(expr.ptr, type_fn),
            "type": type_fn(expr.type),
        })
    if isinstance(expr, SizeOf):
        return expr.model_copy(update={"type": type_fn(expr.type)})
    if isinstance(expr, Widen):
        return expr.model_copy(update={
            "value": substitute_in_expr(expr.value, type_fn),
        })
    if isinstance(expr, StructInit):
        new_fields = tuple(
            FieldInit(name=fi.name, value=substitute_in_expr(fi.value, type_fn))
            for fi in expr.fields
        )
        new_type_args = tuple(type_fn(a) for a in expr.type_args)
        return expr.model_copy(update={
            "type_args": new_type_args,
            "fields":    new_fields,
        })
    if isinstance(expr, EnumInit):
        new_fields = tuple(
            FieldInit(name=fi.name, value=substitute_in_expr(fi.value, type_fn))
            for fi in expr.fields
        )
        new_type_args = tuple(type_fn(a) for a in expr.type_args)
        return expr.model_copy(update={
            "type_args": new_type_args,
            "fields":    new_fields,
        })
    if isinstance(expr, BinOp):
        return expr.model_copy(update={
            "lhs": substitute_in_expr(expr.lhs, type_fn),
            "rhs": substitute_in_expr(expr.rhs, type_fn),
        })
    if isinstance(expr, ShortCircuitOr):
        return expr.model_copy(update={
            "lhs": substitute_in_expr(expr.lhs, type_fn),
            "rhs": substitute_in_expr(expr.rhs, type_fn),
        })
    if isinstance(expr, ShortCircuitAnd):
        return expr.model_copy(update={
            "lhs": substitute_in_expr(expr.lhs, type_fn),
            "rhs": substitute_in_expr(expr.rhs, type_fn),
        })
    if isinstance(expr, Call):
        new_args = tuple(substitute_in_expr(a, type_fn) for a in expr.args)
        new_type_args = tuple(type_fn(a) for a in expr.type_args)
        return expr.model_copy(update={
            "type_args": new_type_args,
            "args":      new_args,
        })
    if isinstance(expr, FieldRead):
        return expr.model_copy(update={
            "value": substitute_in_expr(expr.value, type_fn),
        })
    if isinstance(expr, LoadField):
        return expr.model_copy(update={
            "ptr": substitute_in_expr(expr.ptr, type_fn),
            "type_args": tuple(type_fn(a) for a in expr.type_args),
        })
    if isinstance(expr, PtrOffset):
        return expr.model_copy(update={
            "base":   substitute_in_expr(expr.base,   type_fn),
            "offset": substitute_in_expr(expr.offset, type_fn),
        })
    if isinstance(expr, TryExpr):
        return expr.model_copy(update={
            "value": substitute_in_expr(expr.value, type_fn),
        })
    if isinstance(expr, TraitCall):
        return expr.model_copy(update={
            "dispatch_type": type_fn(expr.dispatch_type),
            "args":          tuple(substitute_in_expr(a, type_fn) for a in expr.args),
        })
    if isinstance(expr, (ParamRef, LocalRef, StringRef, NullPtr, CharLit)):
        return expr
    raise AssertionError(f"unhandled expr in substitute: {type(expr).__name__}")


def substitute_in_stmt(stmt, type_fn: Callable):
    if isinstance(stmt, ReturnExpr):
        return stmt.model_copy(update={"value": substitute_in_expr(stmt.value, type_fn)})
    if isinstance(stmt, (Return, Unreachable)):
        return stmt
    if isinstance(stmt, If):
        return stmt.model_copy(update={
            "cond":      substitute_in_expr(stmt.cond, type_fn),
            "then_body": stmt.then_body.model_copy(update={
                "stmts": tuple(substitute_in_stmt(s, type_fn) for s in stmt.then_body.stmts),
            }),
            "else_body": stmt.else_body.model_copy(update={
                "stmts": tuple(substitute_in_stmt(s, type_fn) for s in stmt.else_body.stmts),
            }),
        })
    if isinstance(stmt, Let):
        return stmt.model_copy(update={
            "type": type_fn(stmt.type),
            "init": substitute_in_expr(stmt.init, type_fn),
        })
    if isinstance(stmt, Assign):
        return stmt.model_copy(update={"value": substitute_in_expr(stmt.value, type_fn)})
    if isinstance(stmt, While):
        return stmt.model_copy(update={
            "cond": substitute_in_expr(stmt.cond, type_fn),
            "body": stmt.body.model_copy(update={
                "stmts": tuple(substitute_in_stmt(s, type_fn) for s in stmt.body.stmts),
            }),
        })
    if isinstance(stmt, For):
        return stmt.model_copy(update={
            "lo":   substitute_in_expr(stmt.lo, type_fn),
            "hi":   substitute_in_expr(stmt.hi, type_fn),
            "body": stmt.body.model_copy(update={
                "stmts": tuple(substitute_in_stmt(s, type_fn) for s in stmt.body.stmts),
            }),
        })
    if isinstance(stmt, ExprStmt):
        return stmt.model_copy(update={"value": substitute_in_expr(stmt.value, type_fn)})
    if isinstance(stmt, FieldSet):
        return stmt.model_copy(update={"value": substitute_in_expr(stmt.value, type_fn)})
    if isinstance(stmt, Store):
        return stmt.model_copy(update={
            "ptr":   substitute_in_expr(stmt.ptr,   type_fn),
            "value": substitute_in_expr(stmt.value, type_fn),
        })
    if isinstance(stmt, StoreField):
        return stmt.model_copy(update={
            "ptr":   substitute_in_expr(stmt.ptr,   type_fn),
            "value": substitute_in_expr(stmt.value, type_fn),
            "type_args": tuple(type_fn(a) for a in stmt.type_args),
        })
    if isinstance(stmt, WithArena):
        return stmt.model_copy(update={
            "capacity": substitute_in_expr(stmt.capacity, type_fn),
            "body":     stmt.body.model_copy(update={
                "stmts": tuple(substitute_in_stmt(s, type_fn) for s in stmt.body.stmts),
            }),
        })
    if isinstance(stmt, Match):
        new_arms = tuple(
            arm.model_copy(update={
                "body": arm.body.model_copy(update={
                    "stmts": tuple(substitute_in_stmt(s, type_fn) for s in arm.body.stmts),
                }),
            })
            for arm in stmt.arms
        )
        return stmt.model_copy(update={
            "scrutinee": substitute_in_expr(stmt.scrutinee, type_fn),
            "arms":      new_arms,
        })
    raise AssertionError(f"unhandled stmt in substitute: {type(stmt).__name__}")
