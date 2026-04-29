"""Analyses that derive claims from the graph.

Today: literal-only interprocedural range propagation. For each callee
parameter, if every call site passes an IntLit at that position, derive
an IntRangeClaim(regime=lattice) covering [min, max] of those literals.
Any non-literal arg poisons the parameter (no claim emitted).

The analysis is deliberately limited — no fixpoint, no expression evaluation,
no flow sensitivity. It's the smallest move that makes regime=lattice mean
something concrete and exercises the override flag (--enforce-lattice).
"""

from __future__ import annotations

from collections.abc import Iterator

from cpg.model import (
    BinOp,
    Call,
    Claim,
    If,
    IntLit,
    IntRangeClaim,
    Program,
    ReturnExpr,
)


def derive_lattice_claims(program: Program) -> dict[str, tuple[Claim, ...]]:
    """Return derived (regime=lattice) claims keyed by function name."""
    # arg_buckets[fn_name][param_index] is either a list of int literals
    # observed at that arg position, or None once poisoned.
    defined = {fn.name for fn in program.functions}
    arg_buckets: dict[str, list[list[int] | None]] = {
        fn.name: [[] for _ in fn.params] for fn in program.functions
    }

    for caller in program.functions:
        for stmt in caller.body:
            for call in _walk_calls_in_stmt(stmt):
                if call.function not in defined:
                    continue  # extern or dangling; lower-time error
                buckets = arg_buckets[call.function]
                for i, arg in enumerate(call.args):
                    if i >= len(buckets) or buckets[i] is None:
                        continue
                    if isinstance(arg, IntLit):
                        buckets[i].append(arg.value)
                    else:
                        buckets[i] = None  # poison: non-literal arg

    result: dict[str, tuple[Claim, ...]] = {}
    for fn in program.functions:
        derived: list[Claim] = []
        buckets = arg_buckets[fn.name]
        for i, p_name in enumerate(fn.params):
            literals = buckets[i]
            if literals is None or not literals:
                continue
            lo, hi = min(literals), max(literals)
            n = len(literals)
            derived.append(IntRangeClaim(
                regime="lattice",
                param=p_name,
                min=lo,
                max=hi,
                justification=f"derived: literal range over {n} call site(s)",
            ))
        if derived:
            result[fn.name] = tuple(derived)
    return result


def elaborate(program: Program, derived: dict[str, tuple[Claim, ...]]) -> Program:
    """Return a Program with `derived` claims appended to each function's claims."""
    new_fns = tuple(
        fn.model_copy(update={"claims": fn.claims + derived.get(fn.name, ())})
        for fn in program.functions
    )
    return program.model_copy(update={"functions": new_fns})


# ---------- Call-walking helpers ----------

def _walk_calls_in_stmt(stmt) -> Iterator[Call]:
    match stmt:
        case ReturnExpr(value=expr):
            yield from _walk_calls_in_expr(expr)
        case If(cond=cond, then_body=t_body, else_body=e_body):
            yield from _walk_calls_in_expr(cond)
            for s in t_body:
                yield from _walk_calls_in_stmt(s)
            for s in e_body:
                yield from _walk_calls_in_stmt(s)


def _walk_calls_in_expr(expr) -> Iterator[Call]:
    match expr:
        case Call() as call:
            yield call
            for a in call.args:
                yield from _walk_calls_in_expr(a)
        case BinOp(lhs=l, rhs=r):
            yield from _walk_calls_in_expr(l)
            yield from _walk_calls_in_expr(r)
