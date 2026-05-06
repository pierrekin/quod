"""C-family lowering pass — strips `c.*` extensions to produce layer C.

Reads `Program.structured_functions` (layer B, extension-bearing
transcription) and produces `Program.functions` (layer C, pure core)
plus the cross-layer provenance metadata:

  - One block-level `ProvenanceEdge` per (B-block, C-block) pair. Block
    granularity is sufficient at v5 — every lowering rule operates
    inside an enclosing block, and per-statement edges can be added
    later when a use case needs them.
  - One function-level `Equivalence` per function with `FamilyLowering`
    justification citing the rule(s) used. Functions that needed no
    lowering get a `rule_name="identity"` claim so the data model is
    uniform across C-derived programs.

Step-5 supported rules:

  - `c.for_general` — `for(init; cond; inc) body` becomes
    `init; while (cond) { body; inc }`. The init declaration is
    hoisted out of the for-header into the enclosing block; the inc
    statement is appended to every iteration of the resulting while.
    Step 6 pins the rule's correctness proof in Z3 and bumps the
    Equivalence's regime from `axiom` to `witness`.
  - `c.scoped_block` — drops the wrapper, surfacing the inner core
    Block. v5 doesn't yet exploit `scope_locals` (all decls in a C
    scope are already lexically scoped at layer C); the data is
    preserved in the layer-B subtree for future analyses.

Other `c.*` extensions refuse — they'll grow rules as the C subset
expands. Refusal mirrors `lower.py`'s discipline: rather than silently
ignore, surface a clear error naming what's missing.

The pass is **idempotent on layer C**: a Program whose `functions`
collection is already populated and whose `structured_functions` is
empty passes through unchanged. Running the pass twice on the same
input is safe and produces the same output.
"""
from __future__ import annotations

from quod.model import (
    Assign,
    Block,
    BlockOrScoped,
    CScopedBlock,
    CStyleFor,
    Equivalence,
    FamilyLowering,
    For,
    Function,
    If,
    Match,
    MatchArm,
    ProvenanceEdge,
    Program,
    Statement,
    While,
    WithArena,
)


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
        # function was already pure core). regime=axiom for v5; step 6
        # pins the SMT artifact and bumps to witness.
        rules = sorted(ctx.rules_used) if ctx.rules_used else ["identity"]
        for rule in rules:
            new_equivalences.append(Equivalence(
                a_node_id=b_fn.id,
                b_node_id=c_fn.id,
                justification=FamilyLowering(rule_name=rule),
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
    even though stripping is a no-op semantically at v5, we record
    that the rule fired so the equivalence claim cites it."""
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
        # Append `inc` to every iteration. If `cond` is None, we'd need
        # `while (true)` which v5 doesn't model — refuse rather than
        # silently miscompile. (Sum.c always has a cond.)
        if stmt.cond is None:
            raise ValueError(
                "c.for_general lowering: `cond` is None — `for (init;;inc)` "
                "form requires a `while (true)` analogue at layer C, which "
                "v5 doesn't model. Add a CStyleFor refusal at ingest time "
                "or extend the lowering rule to emit a constant-true cond."
            )
        # Append `inc` statements at the end of body. New block ID
        # since we materially changed the contents — keeps the edge
        # graph honest about which contents go with which ID.
        with_inc_id = ctx.mint_block_id()
        ctx.edges.append(ProvenanceEdge(source=body_block.id, target=with_inc_id))
        with_inc = Block(
            id=with_inc_id,
            stmts=body_block.stmts + tuple(inc_stmts),
        )
        return (*init_stmts, While(cond=stmt.cond, body=with_inc))

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
