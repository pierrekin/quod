"""Predicate domain — the bool-valued tree restricted to PredicateClaim.expr.

A predicate in quod is a bool-valued expression built from `BinOp`, `Not`,
`ShortCircuitAnd` / `ShortCircuitOr`, `IntLit`, and `ParamRef` / `ReturnRef`
leaves. It is the vocabulary of claim bodies — `non_negative(x)`,
`int_range(x, lo, hi)`, `return_in_range(lo, hi)` and their generalizations.

This package consolidates everything that operates on that vocabulary:

- `predicate_validate` — admits-or-refuses a predicate (rejects non-predicate
  expressions like `Call`, `FieldRead`, etc.).
- `predicate_canonical` — normal-form rewriter and the sugar-shape builders
  that the CLI emits when adding `non_negative` / `int_range` claims.
- `predicate_render` — the inverse of `predicate_canonical`: recognizes
  canonical sugar shapes and emits the human-readable form for `quod show`.
- `predicate_proof` — Z3 / SMT-LIB lowering. Encodes a predicate as an
  SMT-LIB goal and runs Z3.
- `predicate_providers` — the registry of claim-derivation backends
  (lattice literal-range, Z3 QF_LIA) keyed by `(regime, mode)`.
"""
from __future__ import annotations
