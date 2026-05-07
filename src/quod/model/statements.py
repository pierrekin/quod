"""Statement nodes — Block, control flow, mutation, match.

The Statement union forward-references `CStyleFor` (defined in layer_b)
because the family-extension lives outside the core statement family;
the union resolves once `CStyleFor` is bound into this module's
namespace by `model/__init__.py`.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field

from quod.model.base import _Node, _mint_block_id
from quod.model.expressions import Expr
from quod.model.types import IntType, Type


class ReturnExpr(_Node):
    kind: Literal["quod.return_expr"] = "quod.return_expr"
    value: Expr


class Return(_Node):
    """Bare return for void functions. The enclosing function's return_type
    must be llvm.void; non-void functions must use return_expr."""
    kind: Literal["quod.return"] = "quod.return"


class Unreachable(_Node):
    """A statement that must not be executed at runtime. Lowers to LLVM
    `unreachable`. Used to terminate a basic block when the source
    language's semantics for reaching this point are undefined — e.g. C's
    fall-through off a non-`main` int-returning function (UB per C99
    §6.9.1/12). Lets downstream analysis flag the path as a finding rather
    than silently fabricating a return value."""
    kind: Literal["quod.unreachable"] = "quod.unreachable"


class Break(_Node):
    """Exit the innermost enclosing loop (`while` or `for`). The
    validator refuses any `Break` outside a loop body. Lowers to
    `br loop-exit`."""
    kind: Literal["quod.break"] = "quod.break"


class Continue(_Node):
    """Skip to the next iteration of the innermost enclosing loop.
    The validator refuses any `Continue` outside a loop body. Lowers
    to `br loop-cond` (or, when the source is a c-family
    `for (init; cond; inc) body`, the c-family lowering pre-rewrites
    `continue` to `inc; continue` so the inc step is preserved per
    C semantics)."""
    kind: Literal["quod.continue"] = "quod.continue"


class Block(_Node):
    """Identified container for a sequence of statements.

    Endpoint of provenance and equivalence edges across language-family
    layers. The `id` is opaque, minted at construction time, and
    persisted in JSON so reloads are deterministic. `Program.edges` and
    `Equivalence` claims anchor on these IDs.
    """
    id: str = Field(default_factory=_mint_block_id)
    stmts: tuple["Statement", ...] = ()


class If(_Node):
    """Two-branch conditional. Branches may both terminate (return), or both
    fall through to the next statement, or mix — a merge block is created
    on demand by the lowering pass."""
    kind: Literal["quod.if"] = "quod.if"
    cond: Expr  # must lower to i1
    then_body: Block
    else_body: Block


class Let(_Node):
    """Introduce a mutable local variable. `name` must not shadow a parameter
    or another local in the same function. Lowered to alloca-at-entry plus
    a store of `init` at the binding point. When `init` is None, the
    local starts out **uninitialized** — alloca only, no store; reading
    such a local is undefined behaviour, and the validator's forward
    must-init analysis refuses any program that lets such a read be
    reachable. The C ingester uses this for `int x;` (no initializer);
    other source languages can use it too."""
    kind: Literal["quod.let"] = "quod.let"
    name: str
    type: Type
    init: Expr | None = None


class Assign(_Node):
    """Mutate an existing local. `name` must reference a local previously
    introduced by `Let` (or a `For` loop variable in scope)."""
    kind: Literal["quod.assign"] = "quod.assign"
    name: str
    value: Expr


class While(_Node):
    """Pre-test loop. Evaluates `cond` each iteration; runs `body` if true."""
    kind: Literal["quod.while"] = "quod.while"
    cond: Expr  # must lower to i1
    body: Block


class DoWhile(_Node):
    """Post-test loop. Runs `body` unconditionally, then evaluates
    `cond` and loops back if true. The body always executes at least
    once. Inside `body`, `continue` jumps to the cond check (matching
    C's `do { ... } while (...);` semantics)."""
    kind: Literal["quod.do_while"] = "quod.do_while"
    body: Block
    cond: Expr  # must lower to i1


class For(_Node):
    """Bounded iteration: `var` runs from `lo` (inclusive) to `hi` (exclusive),
    incrementing by 1 each iteration. `lo` and `hi` are evaluated once before
    the loop (snapshot semantics, not C-style re-evaluation). `var` is a local
    of type `type`, scoped to `body` only. `lo` and `hi` must produce values
    of the same type as `var`."""
    kind: Literal["quod.for"] = "quod.for"
    var: str
    type: IntType
    lo: Expr
    hi: Expr
    body: Block


class ExprStmt(_Node):
    """Evaluate an expression for its side effects, discard the result.
    The natural shape for `printf(...)` and other void-effect calls."""
    kind: Literal["quod.expr_stmt"] = "quod.expr_stmt"
    value: Expr


class FieldSet(_Node):
    """Mutate one field of a struct-typed local. Same scoping as `Assign`:
    `local` must reference a Let-introduced local of some `StructType("X")`,
    and `name` must be a field of that struct's def."""
    kind: Literal["quod.field_set"] = "quod.field_set"
    local: str
    name: str
    value: Expr


class Store(_Node):
    """Write `value` to the memory at the i8* pointer `ptr`.

    Lowered as `bitcast(ptr, T*)` + an LLVM `store` of `value` (which has
    type T). The store happens at the natural alignment of T; mismatched
    underlying memory is undefined behaviour, same caveat as `quod.load`.

    Pair with `quod.ptr_offset` to write at a non-zero offset of a buffer:
    `store(ptr_offset(buf, k), 0x42 as i8)`.
    """
    kind: Literal["quod.store"] = "quod.store"
    ptr: Expr     # must lower to i8*
    value: Expr   # iN or named struct value


class StoreField(_Node):
    """Write a value into a named field of a struct stored at an i8*
    pointer. Same shape and lowering rationale as `LoadField` but
    mutating: lowered as `bitcast(ptr, T*)` + `getelementptr` + `store`.

    Equivalent in effect to loading the whole struct, mutating one
    field, and storing it back, but emits a targeted GEP+store instead.
    The point is the same as `LoadField` — make struct-on-heap mutation
    expressible without forcing the whole struct through a register.

    `type_args` populates the type parameters when `struct_type` names a
    generic StructDef. The monomorphizer mangles `struct_type` using the
    args (same pattern as StructInit/EnumInit/Call) and clears `type_args`.
    Empty tuple is the non-generic case."""
    kind: Literal["quod.store_field"] = "quod.store_field"
    ptr: Expr                                  # must lower to i8*
    struct_type: str                           # name of a StructDef
    type_args: tuple["Type", ...] = ()         # generic instantiation, if any
    name: str                                  # field name
    value: Expr                                # the field's declared type


class WithArena(_Node):
    """Bracket a body with an arena that's freed automatically.

    Lowering is a desugar: at block entry `mem.arena.new` is called with
    `capacity` (i64) and the result (i8*) bound to a local named `name`
    for the duration of `body`. On every exit edge — fall-through and
    every `return` reachable from `body` — `mem.arena.drop` is called
    on that handle.

    The desugaring auto-injects `imports: ["mem.arena"]` if the program
    doesn't already declare it, so a `with_arena` block is one-stop sugar;
    `mem.arena.alloc` (and the other arena functions) are visible to
    code inside the body via the same import.
    """
    kind: Literal["quod.with_arena"] = "quod.with_arena"
    name: str
    capacity: Expr   # must lower to i64
    body: Block


class MatchArm(_Node):
    """One arm of a `Match`. Names a variant, binds its payload fields to
    locals (one name per field, in declaration order), and runs `body`
    with those locals in scope.

    The literal variant name `_` is a wildcard — it matches every variant
    not handled by another arm of the same match. Wildcards take no
    bindings (use a normal variant arm if you need the payload). At most
    one wildcard per match. With a wildcard present, the named-variant
    arms don't have to be exhaustive.

    Bindings are scoped to `body` only — they don't leak into sibling
    arms or out of the match.
    """
    variant: str
    bindings: tuple[str, ...] = ()
    body: Block


class Match(_Node):
    """Pattern-match on an enum value. One arm per variant, exhaustive.

    `scrutinee` must lower to a value of some `EnumType("E")`. Arms must
    cover every variant of `E` exactly once, in any order. No wildcards,
    no guards, no nested patterns yet.

    Lowered to a `switch` on the discriminant byte; each arm's body
    runs in its own basic block with the variant's payload fields
    bound as locals.
    """
    kind: Literal["quod.match"] = "quod.match"
    scrutinee: Expr
    arms: tuple[MatchArm, ...]


Statement = Annotated[
    Union[
        ReturnExpr, Return, Unreachable, Break, Continue, If, Let, Assign,
        While, DoWhile, For, ExprStmt, FieldSet, Store, StoreField, WithArena,
        Match,
        # Forward-declared `c.*` family extension. CStyleFor is defined
        # in `quod.model.layer_b`; the union uses a string forward-ref
        # which is resolved by `model/__init__.py` after both modules
        # have been imported.
        "CStyleFor",
    ],
    Field(discriminator="kind"),
]


def body_always_terminates(stmts) -> bool:
    """Conservative: True only when the last reachable statement is provably
    a terminator — a `return`, an `unreachable`, a `break` or `continue`
    (which exit the enclosing loop without falling through), an `if`
    whose branches both terminate, or a `match` whose arms (and wildcard,
    if present) all terminate. Used by the C ingest to decide whether a
    fall-through needs synthesizing, and by the lowering pass to suppress
    dead trailing instructions (e.g. arena drops after a body that never
    falls through)."""
    if not stmts:
        return False
    last = stmts[-1]
    match last:
        case ReturnExpr() | Return() | Unreachable() | Break() | Continue():
            return True
        case If(then_body=t, else_body=e):
            return body_always_terminates(t.stmts) and body_always_terminates(e.stmts)
        case Match(arms=arms):
            return all(body_always_terminates(arm.body.stmts) for arm in arms)
    return False
