"""With-arena desugaring: a Program -> Program pre-pass that rewrites
each `WithArena` block into the equivalent `Let` + `Call` sequence and
threads `arena_drop` calls before every reachable `return`. Touches no
LLVM — runs before `lower()` proper begins.
"""

from __future__ import annotations

from quod.model import (
    Call,
    DoWhile,
    ExprStmt,
    For,
    Function,
    I8PtrType,
    If,
    Let,
    LocalRef,
    Match,
    Program,
    Return,
    ReturnExpr,
    While,
    WithArena,
    body_always_terminates,
)
from quod.resolve import resolve_imports


_ARENA_MODULE = "mem.arena"
_ARENA_NEW = "mem.arena.new"
_ARENA_DROP = "mem.arena.drop"


def _desugar_with_arena(program: Program) -> Program:
    """Rewrite every `WithArena` block into the equivalent `Let` + `Call`
    sequence, threading `arena_drop` calls before every `return` reachable
    from the body.

    The arena allocator (`mem.arena.new` / `mem.arena.alloc` /
    `mem.arena.drop` / `mem.arena.used`) is shipped as the
    `mem.arena` stdlib module. When a program uses `WithArena`, we
    auto-inject the import and re-run `resolve_imports` so the desugared
    Call expressions resolve to real Function nodes by the time lowering
    proper begins. Idempotent: programs with no `WithArena`, or programs
    that already import `mem.arena`, pass through unchanged (any inner
    `resolve_imports` call is itself idempotent — it clears imports after
    the first run).
    """
    has_block = any(_function_uses_with_arena(fn) for fn in program.functions)
    if not has_block:
        return program

    from quod.model import Import
    if not any(imp.module == _ARENA_MODULE for imp in program.imports):
        program = program.model_copy(update={
            "imports": program.imports + (Import(module=_ARENA_MODULE),),
        })
    program = resolve_imports(program)

    new_functions = []
    for fn in program.functions:
        # Per-function counter for `__arena_retval_N` locals introduced
        # by hoisting return-expr values across an arena drop.
        next_id = [0]
        new_stmts = _desugar_stmts(fn.body.stmts, fn.return_type, next_id)
        new_body = fn.body.model_copy(update={"stmts": new_stmts})
        new_functions.append(fn.model_copy(update={"body": new_body}))
    return program.model_copy(update={"functions": tuple(new_functions)})


def _function_uses_with_arena(fn: Function) -> bool:
    return any(_stmt_contains_with_arena(s) for s in fn.body.stmts)


def _stmt_contains_with_arena(s) -> bool:
    match s:
        case WithArena():
            return True
        case If(then_body=t, else_body=e):
            return any(_stmt_contains_with_arena(x) for x in (*t.stmts, *e.stmts))
        case While(body=b) | DoWhile(body=b) | For(body=b):
            return any(_stmt_contains_with_arena(x) for x in b.stmts)
    return False


def _desugar_stmts(stmts, return_type, next_id) -> tuple:
    out: list = []
    for s in stmts:
        match s:
            case WithArena(name=name, capacity=cap, body=body):
                inner = _desugar_stmts(body.stmts, return_type, next_id)
                drop_stmt = ExprStmt(value=Call(
                    function=_ARENA_DROP, args=(LocalRef(name=name),),
                ))
                inner_with_drops = _prepend_drop_before_returns(
                    inner, drop_stmt, return_type, next_id,
                )
                out.append(Let(
                    name=name, type=I8PtrType(),
                    init=Call(function=_ARENA_NEW, args=(cap,)),
                ))
                out.extend(inner_with_drops)
                # If every path through the body returns, the fall-through
                # drop and anything that follows in the outer block is
                # unreachable. The lowering pass leaves the IR builder in a
                # terminated block after such an If, so we must trim here
                # instead of emitting dead instructions on top of the ret.
                if body_always_terminates(inner_with_drops):
                    return tuple(out)
                out.append(drop_stmt)
            case If(then_body=t, else_body=e):
                out.append(s.model_copy(update={
                    "then_body": t.model_copy(update={
                        "stmts": _desugar_stmts(t.stmts, return_type, next_id),
                    }),
                    "else_body": e.model_copy(update={
                        "stmts": _desugar_stmts(e.stmts, return_type, next_id),
                    }),
                }))
            case While(body=b) | DoWhile(body=b):
                out.append(s.model_copy(update={
                    "body": b.model_copy(update={
                        "stmts": _desugar_stmts(b.stmts, return_type, next_id),
                    }),
                }))
            case For(body=b):
                out.append(s.model_copy(update={
                    "body": b.model_copy(update={
                        "stmts": _desugar_stmts(b.stmts, return_type, next_id),
                    }),
                }))
            case _:
                out.append(s)
    return tuple(out)


def _prepend_drop_before_returns(stmts, drop_stmt, return_type, next_id) -> tuple:
    """Walk `stmts` and emit `drop_stmt` between each `ReturnExpr` /
    bare `Return` and the actual exit. Recurses into branches and loop
    bodies; nested `WithArena`s have already been desugared (their own
    drops already in place), so we only need to add ours.

    For `ReturnExpr(value=v)`, the value is hoisted into a fresh local
    BEFORE the drop runs:
        let __arena_retval_N: T = v
        drop(arena)
        ret __arena_retval_N
    Otherwise the drop would invalidate any arena-resident memory that
    `v` reads (e.g. a struct loaded out of the arena, or a List<T>'s
    backing buffer). Without this hoist, the IR is morally
    `drop; ret v` and `v`'s loads happen AFTER drop — UB. (Optimized
    builds happen to hoist the load themselves; unoptimized builds
    blow up.)
    """
    out: list = []
    for s in stmts:
        match s:
            case ReturnExpr(value=v):
                next_id[0] += 1
                tmp = f"__arena_retval_{next_id[0]}"
                out.append(Let(name=tmp, type=return_type, init=v))
                out.append(drop_stmt)
                out.append(ReturnExpr(value=LocalRef(name=tmp)))
            case Return():
                out.append(drop_stmt)
                out.append(s)
            case If(then_body=t, else_body=e):
                out.append(s.model_copy(update={
                    "then_body": t.model_copy(update={
                        "stmts": _prepend_drop_before_returns(t.stmts, drop_stmt, return_type, next_id),
                    }),
                    "else_body": e.model_copy(update={
                        "stmts": _prepend_drop_before_returns(e.stmts, drop_stmt, return_type, next_id),
                    }),
                }))
            case While(body=b) | DoWhile(body=b):
                out.append(s.model_copy(update={
                    "body": b.model_copy(update={
                        "stmts": _prepend_drop_before_returns(b.stmts, drop_stmt, return_type, next_id),
                    }),
                }))
            case For(body=b):
                out.append(s.model_copy(update={
                    "body": b.model_copy(update={
                        "stmts": _prepend_drop_before_returns(b.stmts, drop_stmt, return_type, next_id),
                    }),
                }))
            case Match(arms=arms):
                new_arms = tuple(
                    arm.model_copy(update={
                        "body": arm.body.model_copy(update={
                            "stmts": _prepend_drop_before_returns(
                                arm.body.stmts, drop_stmt, return_type, next_id,
                            ),
                        }),
                    })
                    for arm in arms
                )
                out.append(s.model_copy(update={"arms": new_arms}))
            case _:
                out.append(s)
    return tuple(out)
