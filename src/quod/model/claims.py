"""Claim nodes — predicate-bearing assertions on functions and externs.

Single canonical claim form: `PredicateClaim`. The named claim shapes
(`non_negative`, `int_range`, `return_in_range`) are CLI sugar that
desugar to canonicalized `PredicateClaim`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_serializer

from quod.model.base import _Node
from quod.model.expressions import Expr, _collect_predicate_refs
from quod.model.justifications import Justification


# Epistemic source of a claim (who/what is making the assertion):
#   axiom   = the programmer asserts it (typically: no justification, or manual)
#   witness = a proof was produced out-of-band (typically: z3/coq/lean/...)
#   lattice = derived by an analysis pass (typically: derived)
Regime = Literal["axiom", "witness", "lattice"]

# Enforcement: do we trust the source named by `regime`, or verify at runtime?
#   trust  = lowered to llvm.assume; falsity is undefined behaviour
#   verify = lowered to a runtime branch + abort; falsity aborts the program
Enforcement = Literal["trust", "verify"]


class _Claim(_Node):
    """Common metadata carried by every claim.

    Defaults: a programmer assertion (regime=axiom), trusted unconditionally
    (enforcement=trust), without a justification (justification=None).
    """
    regime: Regime = "axiom"
    enforcement: Enforcement = "trust"
    justification: Justification | None = None

    # Drop metadata fields from serialized JSON when they're at default. This
    # keeps program.json compact for the common case while preserving the
    # discriminator `kind` (which is also default-valued but must round-trip).
    @model_serializer(mode="wrap")
    def _drop_default_metadata(self, handler, info):
        data = handler(self)
        if self.regime == "axiom":
            data.pop("regime", None)
        if self.enforcement == "trust":
            data.pop("enforcement", None)
        if self.justification is None:
            data.pop("justification", None)
        return data


class PredicateClaim(_Claim):
    """Single canonical claim form: a predicate over the function's
    parameters and (optionally) `ReturnRef`.

    Pre/post is implicit: a predicate that mentions `ReturnRef` is a
    postcondition, otherwise a precondition. The body says everything
    about scope — there is no separate scope field.

    `expr` must satisfy `assert_is_predicate`: i1-typed, side-effect-free,
    references resolve against the enclosing function. Stored in
    canonical form (see `quod.canonicalize`) so identical predicates
    produce identical hashes for proof pinning, dedup, and equivalence
    edges.

    The named claim kinds (`non_negative`, `int_range`, `return_in_range`)
    are CLI sugar that desugar to canonicalized PredicateClaim at parse
    time and resugar via a recognizer at render time. Sugar shapes are
    never stored.
    """
    kind: Literal["predicate"] = "predicate"
    expr: "Expr"


Claim = PredicateClaim


def claim_param(claim: PredicateClaim) -> str | None:
    """The sole parameter referenced by a claim's predicate, or None.

    Returns None if the predicate references zero params, multiple
    distinct params, or any `ReturnRef`. Used by introspection that
    only makes sense for single-param preconditions (e.g.
    `quod fn unconstrained`); callers that want all referenced params
    should walk `claim.expr` directly.
    """
    refs: set[str] = set()
    has_return = _collect_predicate_refs(claim.expr, refs)
    if has_return or len(refs) != 1:
        return None
    return next(iter(refs))
