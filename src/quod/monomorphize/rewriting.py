"""Reference rewriting (post-substitution).

Rewrites every `(name, type_args)` in a Type/Expr/Stmt to its mangled
flat name. Assumes all TypeParamRefs are already substituted.
"""

from __future__ import annotations

from ..model import (
    Assign,
    BinOp,
    Break,
    Call,
    CharLit,
    Cast,
    Continue,
    DoWhile,
    EnumInit,
    EnumType,
    ExprStmt,
    FieldInit,
    FieldRead,
    FieldSet,
    FloatLit,
    FNeg,
    For,
    If,
    IfExpr,
    IntLit,
    Let,
    Load,
    LoadField,
    LocalRef,
    Match,
    Not,
    NullPtr,
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
    StructInit,
    StructType,
    TraitCall,
    TryExpr,
    Unreachable,
    While,
    WithArena,
)

from .mangling import _mangle


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
    if isinstance(expr, FloatLit):
        return expr.model_copy(update={"type": fn(expr.type)})
    if isinstance(expr, FNeg):
        return expr.model_copy(update={
            "operand": _walk_types_in_expr(expr.operand, fn),
        })
    if isinstance(expr, Load):
        return expr.model_copy(update={
            "ptr":  _walk_types_in_expr(expr.ptr,  fn),
            "type": fn(expr.type),
        })
    if isinstance(expr, SizeOf):
        return expr.model_copy(update={"type": fn(expr.type)})
    if isinstance(expr, Cast):
        # target_type is the full Type union — may carry TypeParamRef
        # pre-monomorphization.
        return expr.model_copy(update={
            "value": _walk_types_in_expr(expr.value, fn),
            "target_type": fn(expr.target_type),
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
