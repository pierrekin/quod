"""Range-hint extraction from layer-A binary subtrees (`Program.binary_units`).

The first cross-layer claim flow per `.scratch/ghidra/04-providers-and-the-bet.md`:

> The provider runs interprocedurally over the binary and writes
> candidate claims **on the paired source function** via the
> equivalence. The verifier then runs `claim suggest`-style — does the
> candidate hold on the source side? If yes, promote to witness.

`derive_binary_range_hints(program)` is the derive function: it walks
every `BinaryProvenance` equivalence in the program, finds the paired
source `Function`, scans the binary function's p-code for signed-
compare constants, and emits one candidate `PredicateClaim` per
unique constant K — `int_range(param, …, K-1)` and
`int_range(param, K, …)` — keyed by source function name.

**Unsound by design.** Ghidra's heuristics produce hints, not proofs:
a comparison the binary performs is evidence that *some* execution
path reasons about that constant, not evidence that the parameter is
constrained to that range over all calls. The regime stays `lattice`
because the *analysis* derived these claims (the `Regime` enum
distinguishes who/what asserts a claim, not how sound the assertion
is). The cross-layer flow filters: agents pick candidates, run
`quod claim prove` on the source side, and the verifier upgrades
survivors to `witness`.

V1 limitations (each is a polish item, see `.scratch/ghidra/06-polish.md`):
- Only single-int-parameter source functions get hints. Multi-param
  attribution requires varnode→param ABI mapping (x86_64 SysV: RDI is
  param[0], RSI is param[1], etc.).
- Only `INT_SLESS` / `INT_SLESSEQUAL` p-code ops are scanned. Compilers
  often lower `v < K` to `(v - K) < 0`; a v2 should walk INT_SUB →
  INT_SLESS chains to recover the original constant.
- No control-flow refinement. The design memo describes the cleaner
  pattern "if (x < 0) return early; otherwise proceed" — recovering
  the implicit "x ≥ 0 in the body" requires CFG analysis the v1
  doesn't yet do.
"""

from __future__ import annotations

from quod.model import (
    BinaryProvenance,
    BinFunction,
    DerivedJustification,
    Function,
    I1Type,
    I8Type,
    I16Type,
    I32Type,
    I64Type,
    IntType,
    IsizeType,
    PredicateClaim,
    Program,
    U8Type,
    U16Type,
    U32Type,
    U64Type,
    UsizeType,
)
from quod.model.claims import Claim
from quod.predicate.canonical import predicate_for_param_range


_INT_TYPE_CLASSES = (
    I1Type, I8Type, I16Type, I32Type, I64Type,
    U8Type, U16Type, U32Type, U64Type, IsizeType, UsizeType,
)


_ANALYSIS_NAME = "ghidra.range_hints"

# Cap candidates per source function so a binary with many comparisons
# doesn't drown the user / agent in nearly-identical lattice claims.
# Each unique constant K produces up to two candidates (≤ K-1 and ≥ K),
# so 6 caps roughly to "the three most interesting thresholds."
_MAX_HINTS_PER_FUNCTION = 6

# Constants that are almost never user-meaningful range thresholds —
# they're compiler-emitted artifacts. Filtering them up front keeps
# the candidate stream cleaner without affecting recall.
_ARTIFACT_CONSTANTS = frozenset({
    0xFF,         # byte mask after INT_AND
    0x1,          # flag-bit / increment / popcount
    0xFFFFFFFF,   # 32-bit all-ones
    0xFFFFFFFFFFFFFFFF,  # 64-bit all-ones
})

# Ghidra's signed-comparison p-code opcodes. INT_SLESS is `<`,
# INT_SLESSEQUAL is `<=`. The unsigned variants (INT_LESS,
# INT_LESSEQUAL) appear too — typically from compiler-emitted unsigned
# comparisons (e.g. array bounds with size_t indices) — but a positive
# unsigned compare doesn't directly imply a signed range, so we skip
# them in v1 to keep the claim type sound for the predicate sugar
# (`int_range` is signed-int).
_SIGNED_CMP_OPCODES = frozenset({"INT_SLESS", "INT_SLESSEQUAL"})


def derive_binary_range_hints(program: Program) -> dict[str, tuple[Claim, ...]]:
    """Return derived (regime=lattice) claims keyed by source `Function.name`.

    Same shape as `quod.analysis.derive_lattice_claims` so the existing
    `elaborate(program, derived)` / `quod claim derive` pipeline picks
    them up without changes.

    Returns `{}` when the program has no `BinaryProvenance` equivalences
    (i.e., no binary ingest has run, or no source-binary name matches).
    """
    bin_fns = {fn.id: fn for u in program.binary_units for fn in u.functions}
    if not bin_fns:
        return {}

    pairings = _pair_function_to_bin(program, bin_fns)
    if not pairings:
        return {}

    result: dict[str, tuple[Claim, ...]] = {}
    for fn, bin_fn in pairings:
        int_params = [p for p in fn.params if isinstance(p.type, _INT_TYPE_CLASSES)]
        if len(int_params) != 1:
            # Multi-int-param attribution requires varnode→param ABI
            # mapping, deferred to v2 (see module docstring).
            continue
        target = int_params[0]

        constants = _signed_compare_constants(bin_fn)
        if not constants:
            continue

        candidates = _candidates_for_constants(
            param_name=target.name,
            param_type=target.type,
            constants=constants,
            bin_fn_id=bin_fn.id,
            bin_fn_label=bin_fn.demangled_name or bin_fn.mangled_name,
        )
        if candidates:
            result[fn.name] = candidates

    return result


def _pair_function_to_bin(
    program: Program,
    bin_fns: dict[str, BinFunction],
) -> list[tuple[Function, BinFunction]]:
    """Walk `BinaryProvenance` equivalences and resolve each binary
    endpoint to a layer-C `Function` in the same program.

    The seeder (`quod.ingest.binary.driver.seed_binary_equivalences`)
    pairs `bin.fn ↔ CFn` (Layer A) by symtab name match. To project the
    pairing to a Layer-C `Function`, we use name match: the c-ingester
    preserves `Function.name == CFn.name == BinFunction.demangled_name`
    end-to-end, so `Function.name == CFn.name` is reliable for the
    common case.

    A Layer-A-only program (CFn but no Function — e.g. before the
    c-family lowering pass runs) returns no pairings; the verifier
    runs against `Function`, so a hint without a `Function` endpoint
    has nowhere to land.
    """
    cfn_to_name: dict[str, str] = {}
    for unit in program.source_units:
        for cfn in unit.functions:
            cfn_to_name[cfn.id] = cfn.name

    fn_by_id: dict[str, Function] = {fn.id: fn for fn in program.functions}
    fn_by_name: dict[str, Function] = {fn.name: fn for fn in program.functions}

    pairs: list[tuple[Function, BinFunction]] = []
    seen: set[tuple[str, str]] = set()
    for eq in program.equivalences:
        if not isinstance(eq.justification, BinaryProvenance):
            continue
        # Identify which side is the bin.fn and which is the source side.
        if eq.b_node_id in bin_fns:
            bin_id, src_id = eq.b_node_id, eq.a_node_id
        elif eq.a_node_id in bin_fns:
            bin_id, src_id = eq.a_node_id, eq.b_node_id
        else:
            continue
        bin_fn = bin_fns[bin_id]

        # Resolve src_id to a Layer-C Function. Direct hit if the
        # seeder paired to a Function id; otherwise project the
        # Layer-A CFn endpoint through the c-ingester's
        # CFn.name == Function.name invariant.
        fn = fn_by_id.get(src_id)
        if fn is None and src_id in cfn_to_name:
            fn = fn_by_name.get(cfn_to_name[src_id])
        if fn is None:
            continue

        key = (fn.id, bin_fn.id)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((fn, bin_fn))
    return pairs


def _signed_compare_constants(bin_fn: BinFunction) -> set[int]:
    """Collect distinct constant operands seen in `INT_SLESS` /
    `INT_SLESSEQUAL` p-code ops across every basic block of `bin_fn`.

    Constants in a varnode are stored in the `const` address space —
    `(space="const", offset=K, size=N)` where `K` is the literal
    value. Filters out compiler artifacts (`_ARTIFACT_CONSTANTS`).
    """
    out: set[int] = set()
    for bb in bin_fn.basic_blocks:
        for op in bb.pcode_ops:
            if op.opcode not in _SIGNED_CMP_OPCODES:
                continue
            for vn in op.inputs:
                if vn.space != "const":
                    continue
                k = vn.offset
                if k in _ARTIFACT_CONSTANTS:
                    continue
                out.add(_signed_const(k, vn.size))
    return out


def _signed_const(raw: int, size_bytes: int) -> int:
    """Reinterpret a varnode constant offset as a signed integer of
    its declared bit-width. Ghidra stores constants as unsigned ints;
    `(const, 0xffffffff, 4)` is the encoding for `-1` in a 32-bit
    comparison, but reaches us as the unsigned 4294967295. Re-sign
    so candidate `int_range` bounds make sense to humans and to the
    SMT prover."""
    if size_bytes <= 0 or size_bytes > 8:
        return raw
    bits = size_bytes * 8
    sign_bit = 1 << (bits - 1)
    mask = (1 << bits) - 1
    raw_masked = raw & mask
    if raw_masked & sign_bit:
        return raw_masked - (1 << bits)
    return raw_masked


def _candidates_for_constants(
    *,
    param_name: str,
    param_type: IntType,
    constants: set[int],
    bin_fn_id: str,
    bin_fn_label: str,
) -> tuple[Claim, ...]:
    """One candidate per unique signed-compare constant K, in two
    directions (`param ≤ K-1` and `param ≥ K`). Capped at
    `_MAX_HINTS_PER_FUNCTION` so a function with many thresholds
    doesn't flood the candidate list."""
    candidates: list[Claim] = []
    seen_bounds: set[tuple[int | None, int | None]] = set()
    # Sort for determinism; smaller magnitudes first (intuitively more
    # likely to be value-domain thresholds rather than address arithmetic).
    for k in sorted(constants, key=lambda v: (abs(v), v)):
        for lo, hi in ((None, k - 1), (k, None)):
            if (lo, hi) in seen_bounds:
                continue
            try:
                expr = predicate_for_param_range(param_name, param_type, lo, hi)
            except ValueError:
                # `predicate_for_param_range` rejects (None, None); other
                # ValueErrors mean the bounds didn't fit the type — skip
                # the candidate rather than crashing the whole derive.
                continue
            seen_bounds.add((lo, hi))
            candidates.append(PredicateClaim(
                regime="lattice",
                expr=expr,
                justification=DerivedJustification(
                    analysis=_ANALYSIS_NAME,
                    inputs=(bin_fn_id,),
                    note=f"signed-compare constant K={k} in {bin_fn_label}",
                ),
            ))
            if len(candidates) >= _MAX_HINTS_PER_FUNCTION:
                return tuple(candidates)
    return tuple(candidates)
