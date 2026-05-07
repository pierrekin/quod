"""Name-keyed lookups + immutable-update helpers on `Program`."""

from __future__ import annotations

from quod.model.claims import PredicateClaim, claim_param
from quod.model.program import Program
from quod.model.top_level import Function


def find_function(program: Program, name: str) -> Function | None:
    for fn in program.functions:
        if fn.name == name:
            return fn
    return None


def require_function(program: Program, name: str) -> Function:
    fn = find_function(program, name)
    if fn is None:
        raise KeyError(f"no function named {name!r}")
    return fn


def replace_function(program: Program, new_fn: Function) -> Program:
    """Return a new Program with the same-named function replaced."""
    updated = tuple(new_fn if fn.name == new_fn.name else fn for fn in program.functions)
    if updated == program.functions:
        raise KeyError(f"no function named {new_fn.name!r}")
    return program.model_copy(update={"functions": updated})


def remove_function(program: Program, function_name: str) -> Program:
    """Return a new Program with the named function removed.

    Permissive about dangling calls — if other functions reference this one,
    the dangling-callee error surfaces at lower time (matches the
    callgraph.json example with `ghost`). Use `quod fn callers` first if you
    want to know who'd be affected.
    """
    kept = tuple(fn for fn in program.functions if fn.name != function_name)
    if kept == program.functions:
        raise KeyError(f"no function named {function_name!r}")
    return program.model_copy(update={"functions": kept})


def add_claim(program: Program, function: str, claim: PredicateClaim) -> Program:
    """Append a claim to a function. Refuses duplicate predicates
    (canonical-equal) — relax first to overwrite."""
    fn = require_function(program, function)
    target = claim_param(claim)
    if target is not None and fn.param(target) is None:
        raise KeyError(f"function {function!r} has no parameter {target!r}")
    for existing in fn.claims:
        if existing.expr == claim.expr:
            raise ValueError(
                f"predicate already present on {function}; "
                f"relax it first if you need to change bounds"
            )
    new_fn = fn.model_copy(update={"claims": fn.claims + (claim,)})
    return replace_function(program, new_fn)


def add_extern_claim(program: Program, extern: str, claim: PredicateClaim) -> Program:
    """Append a claim to the named extern. Mirrors `add_claim`'s
    duplicate-check contract — re-adding the same predicate requires
    `relax` first."""
    new_externs = []
    found = False
    for ext in program.externs:
        if ext.name != extern:
            new_externs.append(ext)
            continue
        found = True
        for existing in ext.claims:
            if existing.expr == claim.expr:
                raise ValueError(
                    f"predicate already present on extern {extern}; "
                    f"relax it first if you need to change bounds"
                )
        new_externs.append(ext.model_copy(update={"claims": ext.claims + (claim,)}))
    if not found:
        raise KeyError(f"no extern named {extern!r}")
    return program.model_copy(update={"externs": tuple(new_externs)})


def relax_extern_claim(program: Program, extern: str, expr) -> Program:
    """Remove an extern claim by exact predicate match."""
    new_externs = []
    found_extern = False
    removed = False
    for ext in program.externs:
        if ext.name != extern:
            new_externs.append(ext)
            continue
        found_extern = True
        kept = tuple(c for c in ext.claims if c.expr != expr)
        if len(kept) != len(ext.claims):
            removed = True
        new_externs.append(ext.model_copy(update={"claims": kept}))
    if not found_extern:
        raise KeyError(f"no extern named {extern!r}")
    if not removed:
        raise KeyError(f"no matching predicate on extern {extern!r}")
    return program.model_copy(update={"externs": tuple(new_externs)})


def relax_claim(program: Program, function: str, expr) -> Program:
    """Remove a claim by exact predicate match (no-op disallowed:
    the predicate must exist)."""
    fn = require_function(program, function)
    kept = tuple(c for c in fn.claims if c.expr != expr)
    if len(kept) == len(fn.claims):
        raise KeyError(f"no matching predicate on {function!r}")
    new_fn = fn.model_copy(update={"claims": kept})
    return replace_function(program, new_fn)
