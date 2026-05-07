"""Claim/predicate lowering.

Predicates are restricted to the side-effect-free expression vocabulary
admitted by `quod.predicate_validate.assert_is_predicate`: int literals, param/
return references, comparisons, integer arithmetic on terms, boolean
combinators, negation. The walker here is intentionally separate from
`_lower_expr` — it doesn't need constants, locals, externs, or aggregate
machinery.

`_lower_claim` injects preconditions at function entry; `_emit_return_claims`
emits postconditions at every ret site; `_emit_extern_call_postconditions`
emits assumes against an extern call's return value at the call site.
"""

from __future__ import annotations

from llvmlite import ir

from quod.lower.runtime_decls import _emit_for_enforcement, _get_or_declare_assume
from quod.lower.types import I1, _ICMP_SIGNED, _ICMP_UNSIGNED, _type_to_llvm
from quod.model import (
    BinOp,
    IntLit,
    Not,
    ParamRef,
    ReturnRef,
)
from quod.proof import predicate_uses_return


def _lower_predicate(
    builder: ir.IRBuilder, expr,
    params: dict[str, ir.Value], ret_val: ir.Value | None,
) -> ir.Value:
    """Lower a predicate body to its IR value. Binds `ReturnRef` to
    `ret_val` (None for preconditions; required when the predicate
    references the return value).

    Predicates are restricted to the side-effect-free expression
    vocabulary admitted by `quod.predicate_validate.assert_is_predicate`: int
    literals, param/return references, comparisons, integer
    arithmetic on terms, boolean combinators, negation. This walker
    is intentionally separate from `_lower_expr` — it doesn't need
    constants, locals, externs, or aggregate machinery.
    """
    def go(e):
        return _lower_predicate(builder, e, params, ret_val)

    match expr:
        case IntLit(type=t, value=v):
            return ir.Constant(_type_to_llvm(t), v)
        case ParamRef(name=n):
            return params[n]
        case ReturnRef():
            if ret_val is None:
                raise AssertionError(
                    "ReturnRef in predicate but no return value in scope"
                )
            return ret_val
        case Not(operand=op):
            return builder.xor(go(op), ir.Constant(I1, 1))
        case BinOp(op="add", lhs=l, rhs=r):
            return builder.add(go(l), go(r))
        case BinOp(op="sub", lhs=l, rhs=r):
            return builder.sub(go(l), go(r))
        case BinOp(op="mul", lhs=l, rhs=r):
            return builder.mul(go(l), go(r))
        case BinOp(op=op, lhs=l, rhs=r) if op in _ICMP_SIGNED:
            return builder.icmp_signed(_ICMP_SIGNED[op], go(l), go(r))
        case BinOp(op=op, lhs=l, rhs=r) if op in _ICMP_UNSIGNED:
            return builder.icmp_unsigned(_ICMP_UNSIGNED[op], go(l), go(r))
        case BinOp(op="or", lhs=l, rhs=r):
            return builder.or_(go(l), go(r))
        case BinOp(op="and", lhs=l, rhs=r):
            return builder.and_(go(l), go(r))
        case BinOp(op="xor", lhs=l, rhs=r):
            return builder.xor(go(l), go(r))
    raise ValueError(f"unhandled predicate expr: {expr!r}")


def _emit_extern_call_postconditions(
    builder: ir.IRBuilder, module: ir.Module,
    ret_val: ir.Value, claims: tuple,
) -> None:
    """Emit `llvm.assume` against an extern call's return value, one
    assume per claim that references `ReturnRef`. Preconditions on
    extern params are emitted at the call site by the caller.

    Only `enforcement="trust"` is supported here. The verify path needs
    `llvm_fn` for basic-block creation, which the call-site lowering
    doesn't carry. Extern claims default to axiom + trust today, so this
    is sufficient; add the verify path when there's a real demand.
    """
    assume = None
    for claim in claims:
        if not predicate_uses_return(claim.expr):
            continue
        if claim.enforcement != "trust":
            raise NotImplementedError(
                f"extern return-claim with enforcement={claim.enforcement!r}: "
                f"only 'trust' is supported"
            )
        if assume is None:
            assume = _get_or_declare_assume(module)
        cmp = _lower_predicate(builder, claim.expr, params={}, ret_val=ret_val)
        builder.call(assume, [cmp])


def _emit_return_claims(
    builder: ir.IRBuilder, ret_val: ir.Value, return_claims: tuple,
    llvm_fn: ir.Function, module: ir.Module, params: dict[str, ir.Value],
    overrides: dict[str, str],
) -> None:
    """Emit llvm.assume / runtime-check predicates against the return value
    just before `ret`. The optimizer learns the bound; after inlining, callers
    learn it too.

    `params` are in scope for postconditions that mention both ReturnRef
    and parameters.
    """
    for claim in return_claims:
        if not predicate_uses_return(claim.expr):
            continue
        enforcement = overrides.get(claim.regime, claim.enforcement)
        cmp = _lower_predicate(builder, claim.expr, params, ret_val)
        _emit_for_enforcement(builder, cmp, enforcement, llvm_fn, module)


def _lower_claim(
    builder: ir.IRBuilder,
    claim,
    params: dict[str, ir.Value],
    llvm_fn: ir.Function,
    module: ir.Module,
    *,
    overrides: dict[str, str],
) -> None:
    """Lower a precondition (no `ReturnRef`) at function entry.
    Postconditions are emitted per return site by `_emit_return_claims`."""
    if predicate_uses_return(claim.expr):
        raise AssertionError(
            "postcondition (ReturnRef-bearing) reached entry-injection path; "
            "_lower_function_body should split these into return_claims"
        )
    enforcement = overrides.get(claim.regime, claim.enforcement)
    cmp = _lower_predicate(builder, claim.expr, params, ret_val=None)
    _emit_for_enforcement(builder, cmp, enforcement, llvm_fn, module)
