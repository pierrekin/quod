"""Pluggable claim providers.

quod's claim system is structured around regimes (axiom, witness, lattice).
A provider is a named source of claims for a given regime. Two built-ins:

  - lattice.literal_range  (regime=lattice, mode=derive)
        runs interprocedural literal-only range propagation; a batch
        analysis that emits all derivable claims at once.

  - z3.qf_lia              (regime=witness, mode=prove)
        encodes a single goal as SMT-LIB QF_LIA/QF_UFLIA, asks z3
        whether the negation is unsat, attaches a Z3Justification.

Either mode takes Program (the CPG) as input and returns Claim objects
keyed by function/param name — i.e., everything is framed in the same
CPG terms the user wrote, never in IR-level SSA names.

External providers can be registered as Python callables; future plugin
discovery (entry points, quod.toml [[provider]] subprocess specs) plugs
in here without changing the consumer surface.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from quod.analysis import derive_lattice_claims
from quod.model import (
    Claim,
    IntRangeClaim,
    NonNegativeClaim,
    Program,
    Regime,
    ReturnInRangeClaim,
    Z3Justification,
)
from quod.proof import Z3NotInstalled, goal_smt_lib, run_z3_on_smt


# ---------- Request / result shapes ----------

@dataclass(frozen=True)
class ClaimRequest:
    """A specific goal to prove. Distinct from `Claim`: at request time, the
    regime, enforcement, and justification are decided by the provider."""
    function: str
    kind: str                  # "non_negative" | "int_range" | "return_in_range"
    target: str | None         # param name; None for return-scoped claims
    min: int | None = None
    max: int | None = None
    enforcement: str = "trust"


@dataclass(frozen=True)
class ProviderResult:
    """`claim` is populated when status == 'proven'."""
    status: Literal["proven", "refuted", "unknown", "error"]
    detail: str
    claim: Claim | None = None
    artifact_path: Path | None = None
    artifact_hash: str | None = None


# ---------- Provider type ----------

# Two provider modes:
#   batch — derive(program) -> {fn_name: (Claim, ...)}, run over the whole program.
#   goal  — prove(program, request, proofs_dir) -> ProviderResult, one claim at a time.
# A provider implements one or the other (or both); the registry routes.
DeriveFn = Callable[[Program], dict[str, tuple[Claim, ...]]]
ProveFn = Callable[[Program, ClaimRequest, Path], ProviderResult]


@dataclass(frozen=True)
class Provider:
    name: str
    regime: Regime
    description: str
    derive: DeriveFn | None = None
    prove: ProveFn | None = None

    @property
    def modes(self) -> tuple[str, ...]:
        m: list[str] = []
        if self.derive is not None:
            m.append("derive")
        if self.prove is not None:
            m.append("prove")
        return tuple(m)


# ---------- Built-in: lattice literal-range propagation ----------

def _lattice_literal_range_derive(program: Program) -> dict[str, tuple[Claim, ...]]:
    return derive_lattice_claims(program)


LATTICE_LITERAL_RANGE = Provider(
    name="lattice.literal_range",
    regime="lattice",
    description=(
        "Interprocedural literal-only range propagation: per parameter, "
        "if every call site passes an IntLit, derive int_range over the "
        "min/max of those literals."
    ),
    derive=_lattice_literal_range_derive,
)


# ---------- Built-in: Z3 / QF_LIA ----------

def _build_claim_from_request(req: ClaimRequest) -> Claim:
    if req.kind == "non_negative":
        if req.target is None:
            raise ValueError("non_negative requires a parameter target")
        return NonNegativeClaim(regime="witness", enforcement=req.enforcement, param=req.target)
    if req.kind == "int_range":
        if req.target is None:
            raise ValueError("int_range requires a parameter target")
        return IntRangeClaim(
            regime="witness", enforcement=req.enforcement,
            param=req.target, min=req.min, max=req.max,
        )
    if req.kind == "return_in_range":
        if req.target is not None:
            raise ValueError("return_in_range takes no parameter target")
        return ReturnInRangeClaim(
            regime="witness", enforcement=req.enforcement,
            min=req.min, max=req.max,
        )
    raise ValueError(f"unknown claim kind: {req.kind!r}")


def _z3_qf_lia_prove(
    program: Program, request: ClaimRequest, proofs_dir: Path,
) -> ProviderResult:
    fn = next((f for f in program.functions if f.name == request.function), None)
    if fn is None:
        return ProviderResult(status="error", detail=f"function {request.function!r} not found")
    try:
        goal = _build_claim_from_request(request)
    except ValueError as e:
        return ProviderResult(status="error", detail=str(e))

    try:
        smt = goal_smt_lib(fn, goal, hypotheses=fn.claims, program=program)
    except NotImplementedError as e:
        return ProviderResult(status="error", detail=f"SMT lowering refused: {e}")

    try:
        z3_result = run_z3_on_smt(smt)
    except Z3NotInstalled as e:
        return ProviderResult(status="error", detail=str(e))

    artifact_hash = hashlib.sha256(smt.encode("utf-8")).hexdigest()
    target_part = request.target or "return"
    proofs_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = proofs_dir / f"{request.function}_{request.kind}_{target_part}_{artifact_hash[:12]}.smt2"
    artifact_path.write_text(smt)

    if z3_result.status != "unsat":
        status: Literal["refuted", "unknown"] = (
            "refuted" if z3_result.status == "sat" else "unknown"
        )
        return ProviderResult(
            status=status,
            detail=f"z3 returned {z3_result.status!r}",
            artifact_path=artifact_path,
            artifact_hash=artifact_hash,
        )

    proven = goal.model_copy(update={
        "justification": Z3Justification(
            artifact_path=str(artifact_path),
            artifact_hash=artifact_hash,
        ),
    })
    return ProviderResult(
        status="proven",
        detail=f"z3 unsat ({len(smt.splitlines())}-line problem)",
        claim=proven,
        artifact_path=artifact_path,
        artifact_hash=artifact_hash,
    )


Z3_QF_LIA = Provider(
    name="z3.qf_lia",
    regime="witness",
    description=(
        "Z3 SMT-LIB over QF_LIA / QF_UFLIA (cross-procedural via uninterpreted "
        "functions). Handles loop-free helpers without srem/sdiv."
    ),
    prove=_z3_qf_lia_prove,
)


# ---------- Registry ----------

_BUILT_IN: tuple[Provider, ...] = (LATTICE_LITERAL_RANGE, Z3_QF_LIA)


def all_providers() -> dict[str, Provider]:
    """The provider registry, keyed by name. Today: just the built-ins.
    Future: merge in user-registered externals (entry points or
    quod.toml [[provider]] subprocess specs)."""
    return {p.name: p for p in _BUILT_IN}


def get_provider(name: str) -> Provider:
    reg = all_providers()
    if name not in reg:
        raise KeyError(f"unknown provider {name!r}; known: {sorted(reg)}")
    return reg[name]


def providers_for(*, regime: Regime | None = None, mode: str | None = None) -> list[Provider]:
    out: list[Provider] = []
    for p in all_providers().values():
        if regime is not None and p.regime != regime:
            continue
        if mode is not None and mode not in p.modes:
            continue
        out.append(p)
    return out


def default_for(*, regime: Regime, mode: str) -> Provider:
    """Return the default provider for (regime, mode). Today: first match in
    the built-in tuple order. Errors if no provider can serve the pair."""
    candidates = providers_for(regime=regime, mode=mode)
    if not candidates:
        raise KeyError(f"no provider available for regime={regime!r}, mode={mode!r}")
    return candidates[0]
