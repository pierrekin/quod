"""Discovery: collect all (template, args) instantiations.

Walks the program seeding a worklist of every concrete `(name, args)`
pair the rest of the pass needs to monomorphize.
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
    FieldRead,
    FieldSet,
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
    Cast,
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
    elif isinstance(expr, Cast):
        _collect_instantiations(expr.target_type, sink)
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
