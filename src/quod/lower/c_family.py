"""C-family lowering pass — strips `c.*` extensions to produce layer C.

Reads `Program.structured_functions` (layer B, extension-bearing
transcription) and produces `Program.functions` (layer C, pure core)
plus the cross-layer provenance metadata:

  - One block-level `ProvenanceEdge` per (B-block, C-block) pair. Block
    granularity is sufficient — every lowering rule operates inside
    an enclosing block, and per-statement edges can be added later
    when a use case needs them.
  - One function-level `Equivalence` per function with `FamilyLowering`
    justification citing the rule(s) used. Rules with a pinned proof
    artifact (under `c_family_proofs/`) emit `regime="witness"` with
    `artifact_path`/`artifact_hash` filled in; `quod equiv verify`
    re-checks both the hash and the Z3 result. Rules without an
    artifact (today: `identity` and `c.scoped_block`) emit
    `regime="axiom"` — honest about what hasn't been proved.

Supported rules:

  - `c.for_general` — `for(init; cond; inc) body` becomes
    `init; while (cond) { body; inc }`. Per-iteration equivalence
    pinned via `c_family_proofs/c_for_general.smt2` (unsat under Z3).
    Whole-loop equivalence is the meta-theoretic inductive lift — see
    the artifact's header comment.
  - `c.scoped_block` — drops the wrapper, surfacing the inner core
    Block. `scope_locals` is currently unused (all decls in a C
    scope are already lexically scoped at layer C); the data is
    preserved in the layer-B subtree for downstream analyses. No
    proof artifact — the rule is structurally a no-op.

Other `c.*` extensions refuse — they'll grow rules as the C subset
expands. Refusal mirrors `quod.lower`'s discipline: rather than silently
ignore, surface a clear error naming what's missing.

The pass is **idempotent on layer C**: a Program whose `functions`
collection is already populated and whose `structured_functions` is
empty passes through unchanged. Running the pass twice on the same
input is safe and produces the same output.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from quod.model import (
    Assign,
    Block,
    BlockOrScoped,
    Continue,
    CScopedBlock,
    CStyleFor,
    Equivalence,
    FamilyLowering,
    For,
    Function,
    I1Type,
    If,
    IntLit,
    Match,
    MatchArm,
    ProvenanceEdge,
    Program,
    Statement,
    While,
    WithArena,
)


# ---------- Proof-artifact registry ----------
#
# Per-rule proof artifacts ship with the package under
# `c_family_proofs/<rule_name>.smt2`. The lowering pass pins each
# artifact's sha256 in the FamilyLowering justification it emits, and
# `quod equiv verify` re-checks (a) the hash matches the file's bytes
# and (b) Z3 returns `unsat` on the artifact.
#
# Rules without an artifact in the registry emit FamilyLowering with
# `artifact_path=None`/`artifact_hash=None` and `regime="axiom"` — the
# equivalence is asserted but unproved. When a proof lands, drop a new
# .smt2 in `c_family_proofs/`, register its rule_name here, and the
# regime auto-bumps to `witness`.

_PROOFS_DIR = Path(__file__).parent / "c_family_proofs"

# Rule name → relative artifact path under the package's `quod/` dir.
# Stored relative so the path persists meaningfully across machines;
# verify resolves it against the package's installed location.
_RULE_PROOFS: dict[str, str] = {
    "c.for_general":  "lower/c_family_proofs/c_for_general.smt2",
    "c.scoped_block": "lower/c_family_proofs/c_scoped_block.smt2",
    "identity":       "lower/c_family_proofs/identity.smt2",
}


def _proof_for(rule_name: str) -> tuple[str, str] | None:
    """Return (artifact_path, artifact_hash) for `rule_name`, or None
    if no proof is registered. The path is package-relative (rooted
    at `src/quod/`); verify resolves it against the package install."""
    rel = _RULE_PROOFS.get(rule_name)
    if rel is None:
        return None
    full = Path(__file__).parent.parent / rel
    if not full.exists():
        # Registered but missing on disk — surface clearly. Misconfiguration
        # rather than a soundness issue.
        raise FileNotFoundError(
            f"c-family proof artifact missing: {full} "
            f"(registered for rule {rule_name!r})"
        )
    digest = hashlib.sha256(full.read_bytes()).hexdigest()
    return rel, digest


def lower_c_family(program: Program) -> Program:
    """Apply c-family lowering: structured_functions → functions.

    No-op when `structured_functions` is empty (the program is already
    pure core). Otherwise produces layer C in `Program.functions`,
    plus the block-level B→C edges and function-level FamilyLowering
    Equivalence claims.
    """
    if not program.structured_functions:
        return program

    layer_c_functions: list[Function] = []
    new_edges: list[ProvenanceEdge] = list(program.edges)
    new_equivalences: list[Equivalence] = list(program.equivalences)

    for b_fn in program.structured_functions:
        ctx = _LowerContext(b_fn.name)
        c_body = _lower_block_or_scoped(b_fn.body, ctx)
        c_fn = b_fn.model_copy(update={
            "id": _layer_c_fn_id(b_fn.name),
            "body": c_body,
        })
        layer_c_functions.append(c_fn)
        new_edges.extend(ctx.edges)
        # Function-level edge.
        new_edges.append(ProvenanceEdge(source=b_fn.id, target=c_fn.id))
        # One Equivalence per rule used (or one identity claim if the
        # function was already pure core). Rules with a registered
        # proof artifact emit regime=witness with the artifact pinned;
        # unproven rules stay regime=axiom — honest about which
        # citations have evidence and which don't.
        rules = sorted(ctx.rules_used) if ctx.rules_used else ["identity"]
        for rule in rules:
            proof = _proof_for(rule)
            if proof is None:
                justification = FamilyLowering(rule_name=rule)
                regime = "axiom"
            else:
                artifact_path, artifact_hash = proof
                justification = FamilyLowering(
                    rule_name=rule,
                    artifact_path=artifact_path,
                    artifact_hash=artifact_hash,
                )
                regime = "witness"
            new_equivalences.append(Equivalence(
                a_node_id=b_fn.id,
                b_node_id=c_fn.id,
                regime=regime,
                justification=justification,
            ))

    return program.model_copy(update={
        "functions": tuple(layer_c_functions),
        "edges": tuple(new_edges),
        "equivalences": tuple(new_equivalences),
    })


class _LowerContext:
    """Per-function bookkeeping during the lowering walk.

    Tracks edges produced for the function's blocks and the set of
    rules actually applied (so the function-level Equivalence can cite
    them). Block IDs are minted with a deterministic per-function
    counter so re-running the pass on identical input produces
    byte-identical output.
    """
    def __init__(self, fn_name: str) -> None:
        self._fn_name = fn_name
        self._block_counter = 0
        self.edges: list[ProvenanceEdge] = []
        self.rules_used: set[str] = set()

    def mint_block_id(self) -> str:
        self._block_counter += 1
        return f"@blk_c_lowered_{self._fn_name}_{self._block_counter}"


def _layer_c_fn_id(name: str) -> str:
    """Layer-C function ID derived from the function's name. Stable
    across re-ingest of the same source (matches the `@fn_c_*`
    convention from the ingester, with `_lowered_` to keep the layer-B
    and layer-C IDs distinct)."""
    return f"@fn_c_lowered_{name}"


def _lower_block_or_scoped(b: BlockOrScoped, ctx: _LowerContext) -> Block:
    """Strip a `CScopedBlock` wrapper if present, then lower the inner
    block. The wrapper contributes `c.scoped_block` to rules_used —
    even though stripping is a no-op semantically, we record that the
    rule fired so the equivalence claim cites it."""
    if isinstance(b, CScopedBlock):
        ctx.rules_used.add("c.scoped_block")
        return _lower_block(b.block, ctx)
    return _lower_block(b, ctx)


def _lower_block(b: Block, ctx: _LowerContext) -> Block:
    """Lower a block in-place: walk its statements, applying rules,
    and return a new Block with a fresh ID. Edge from `b.id` to the
    new block's ID records the pairing."""
    new_stmts: list[Statement] = []
    for stmt in b.stmts:
        new_stmts.extend(_lower_statement(stmt, ctx))
    new_block_id = ctx.mint_block_id()
    ctx.edges.append(ProvenanceEdge(source=b.id, target=new_block_id))
    return Block(id=new_block_id, stmts=tuple(new_stmts))


def _rewrite_continue_with_inc(
    stmts: tuple[Statement, ...], inc_stmts: tuple[Statement, ...],
) -> tuple[Statement, ...]:
    """Replace each `Continue` in `stmts` with `inc; continue`, preserving
    C for-loop semantics. Recurses into `If` branches (nested in the same
    loop scope) but stops at any nested loop (`While`, `For`, `CStyleFor`,
    `DoWhile`) — those loops have their own `continue` target."""
    out: list[Statement] = []
    for s in stmts:
        if isinstance(s, Continue):
            # Replace with `inc; continue`. Note inc_stmts may be empty
            # (sparse for `for (init; cond; ) body` — then continue is
            # plain).
            out.extend(inc_stmts)
            out.append(s)
        elif isinstance(s, If):
            out.append(s.model_copy(update={
                "then_body": s.then_body.model_copy(update={
                    "stmts": _rewrite_continue_with_inc(s.then_body.stmts, inc_stmts),
                }),
                "else_body": s.else_body.model_copy(update={
                    "stmts": _rewrite_continue_with_inc(s.else_body.stmts, inc_stmts),
                }),
            }))
        elif isinstance(s, (While, For, CStyleFor)):
            # Nested loop has its own continue target — don't recurse.
            out.append(s)
        else:
            out.append(s)
    return tuple(out)


def _lower_statement(stmt: Statement, ctx: _LowerContext) -> tuple[Statement, ...]:
    """Lower one layer-B statement; may produce 0..N replacement
    statements at layer C. Recurses into nested body slots so any
    extensions inside `if`/`while`/etc. are also stripped."""
    # ----- c.for_general → init; while (cond) { body; inc } -----
    if isinstance(stmt, CStyleFor):
        ctx.rules_used.add("c.for_general")
        # Init and inc are themselves statements that may contain
        # further extensions; lower them recursively. (For sum.c they
        # are a Let and an Assign, both pure core.)
        init_stmts = _lower_statement(stmt.init, ctx) if stmt.init is not None else ()
        inc_stmts = _lower_statement(stmt.inc, ctx) if stmt.inc is not None else ()
        # Body lowers via the block path (handles CScopedBlock too).
        body_block = _lower_block_or_scoped(stmt.body, ctx)
        # C semantics: `continue` inside a for-loop jumps to the *inc*
        # step, not the cond. The naïve rewrite to `while (cond) { body;
        # inc; }` would have continue jump to the cond, skipping inc —
        # a silent miscompilation. Pre-rewrite each `Continue` in the
        # body (not inside a nested loop, which has its own continue
        # target) to `inc; continue`, so the inc executes before the
        # while-loop's natural continue.
        body_stmts = _rewrite_continue_with_inc(body_block.stmts, tuple(inc_stmts))
        # Append `inc` statements at the end of body so fall-through
        # iterations also run inc. New block ID since we materially
        # changed the contents — keeps the edge graph honest about
        # which contents go with which ID.
        with_inc_id = ctx.mint_block_id()
        ctx.edges.append(ProvenanceEdge(source=body_block.id, target=with_inc_id))
        with_inc = Block(
            id=with_inc_id,
            stmts=body_stmts + tuple(inc_stmts),
        )
        # An absent cond (`for (init;;inc) body`) means "loop forever
        # unless body breaks out" — we model it as `while (true)` with
        # an i1-typed `IntLit(1)` cond. The while-loop preserves the
        # for-form's per-iteration semantics; the equivalence proof
        # is identical to the cond-present case at the per-iteration
        # level.
        cond = stmt.cond if stmt.cond is not None else IntLit(type=I1Type(), value=1)
        return (*init_stmts, While(cond=cond, body=with_inc))

    # ----- Recursive cases — extensions may live inside body slots. -----
    if isinstance(stmt, While):
        return (stmt.model_copy(update={
            "body": _lower_block_or_scoped(stmt.body, ctx),
        }),)
    if isinstance(stmt, If):
        return (stmt.model_copy(update={
            "then_body": _lower_block_or_scoped(stmt.then_body, ctx),
            "else_body": _lower_block_or_scoped(stmt.else_body, ctx),
        }),)
    if isinstance(stmt, For):
        return (stmt.model_copy(update={
            "body": _lower_block_or_scoped(stmt.body, ctx),
        }),)
    if isinstance(stmt, WithArena):
        return (stmt.model_copy(update={
            "body": _lower_block_or_scoped(stmt.body, ctx),
        }),)
    if isinstance(stmt, Match):
        new_arms = tuple(
            arm.model_copy(update={
                "body": _lower_block_or_scoped(arm.body, ctx),
            })
            for arm in stmt.arms
        )
        return (stmt.model_copy(update={"arms": new_arms}),)

    # Refuse any unknown `c.*` extension that reached this far — the
    # rules table needs a corresponding entry.
    kind = getattr(stmt, "kind", None)
    if kind and str(kind).startswith("c."):
        raise ValueError(
            f"c-family lowering: no rule for {kind!r}. Add a rule to "
            f"lower/c_family.py or refuse the construct at ingest time."
        )

    # Default: pure core leaf statements pass through unchanged. (The
    # *containing* block's edge already records the pairing; per-
    # statement edges are deferred until a use case demands them.)
    return (stmt,)
