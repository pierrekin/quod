"""`z3.bin_relational` — relational SMT proof of bin↔src equivalence.

The first relational prover in the project: takes a paired
`(BinFunction, Function)` plus an explicit varnode↔param binding
(`BinSrcSignatureBinding` from `lift_v2.signature_binding`), encodes
both sides as bitvector SMT, and asks z3 whether they always agree
on the return value.

v0 universe (deliberately tiny — the goal is to land the pipeline):

  source side
    - body is `Block(stmts=(ReturnExpr(value=expr),))` — exactly one
      statement, the return; no Let, Assign, If, While, etc.
    - `expr` is built from `IntLit`, `ParamRef`, and
      `BinOp(op∈{add,sub,mul}, …)`.
    - all params and the return are int types (i8 / i16 / i32 / i64
      and unsigned counterparts).

  binary side
    - all int types via fixed-width bitvectors keyed by varnode size.
    - p-code op whitelist:
        COPY, INT_ADD, INT_SUB, INT_MULT,
        INT_NEGATE, INT_2COMP,
        INT_AND, INT_OR, INT_XOR,
        INT_SEXT, INT_ZEXT, SUBPIECE, CAST,
        RETURN (terminator).
    - LOAD/STORE (memory) and INDIRECT are silently ignored — for the
      v0 universe (clang -O1 straight-line int arithmetic) the return
      value isn't computed via memory, and ignoring memory ops is
      sound for that subset.
    - any other op (CBRANCH, CALL, BRANCHIND, comparison ops) bails
      with status="unknown".

Anything outside the v0 universe yields `status="unknown"`. Refusal
beats silence: a generated witness must be tied to an SMT proof, and
a missing encoding rule is a hole, not an oversight to paper over.

Cross-procedural extension and branch/loop support are deferred —
they're the next slices, but each adds enough surface that pinning
v0's pipeline first is more useful than designing the union in
advance.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from quod.model import (
    BinaryProvenance,
    BinBasicBlock,
    BinFunction,
    BinPCodeOp,
    BinSrcSignatureBinding,
    BinVarnode,
    BinOp,
    Block,
    Equivalence,
    Function,
    I1Type,
    I8Type,
    I16Type,
    I32Type,
    I64Type,
    If,
    IntLit,
    IsizeType,
    IntType,
    Not,
    ParamRef,
    Program,
    ReturnExpr,
    U8Type,
    U16Type,
    U32Type,
    U64Type,
    UsizeType,
    Z3Justification,
)
from quod.predicate.proof import Z3NotInstalled, run_z3_on_smt


_INT_TYPE_BITS: dict[type, int] = {
    I1Type: 1, I8Type: 8, U8Type: 8,
    I16Type: 16, U16Type: 16,
    I32Type: 32, U32Type: 32,
    I64Type: 64, U64Type: 64,
    IsizeType: 64, UsizeType: 64,
}


def _int_bits(t: IntType) -> int:
    return _INT_TYPE_BITS[type(t)]


# Pcode opcodes we know how to encode. Anything else bails.
# `RETURN` is a terminator; we stop iterating when we hit one.
# `INDIRECT` (Ghidra's "may-have-other-effects" marker) and memory
# `LOAD`/`STORE` are silent skips — for the v0 universe (clang -O1
# register-only arithmetic) the return value isn't computed via
# memory, and ignoring those ops doesn't change the encoded answer.
_ABI_REGISTER_BITS: dict[str, int] = {
    # Physical register width for the ABI's int-class parameter regs.
    # Used to pre-populate cells when the binding records a sub-register
    # view (e.g. a 4-byte int param in the low half of an 8-byte register).
    "x86_64-sysv": 64,
}


def _abi_register_bits(abi: str) -> int:
    bits = _ABI_REGISTER_BITS.get(abi)
    if bits is None:
        raise _UnsupportedConstruct(f"unknown ABI {abi!r} for register width")
    return bits


_HANDLED_OPCODES = frozenset({
    "COPY",
    "INT_ADD", "INT_SUB", "INT_MULT",
    "INT_NEGATE", "INT_2COMP",
    "INT_AND", "INT_OR", "INT_XOR",
    "INT_SEXT", "INT_ZEXT", "SUBPIECE", "CAST",
    # Comparison ops produce a 1-bit BV `#b1` / `#b0`. Promoted from
    # silently-skipped (v0) to handled (v0.1) because branch
    # conditions consume their output. EFLAGS-only-use cases are
    # still benign: the bound 1-bit BV just doesn't appear in the
    # return path.
    "INT_EQUAL", "INT_NOTEQUAL",
    "INT_LESS", "INT_SLESS",
    "INT_LESSEQUAL", "INT_SLESSEQUAL",
    # Boolean ops. Inputs and outputs are guaranteed to be 0 or 1 in
    # the declared width (Ghidra's pcode semantics). Bitwise BV ops
    # give the correct logical answer on {0, 1} operands.
    "BOOL_NEGATE", "BOOL_AND", "BOOL_OR", "BOOL_XOR",
})
# Block terminators. A block ending in any of these has its
# successor-edge graph followed by the path-walker.
_TERMINATOR_OPCODES = frozenset({"RETURN", "BRANCH", "CBRANCH"})
_SILENTLY_SKIPPED_OPCODES = frozenset({
    "LOAD", "STORE", "INDIRECT", "PIECE",
    # x86 EFLAGS tracking. Ghidra synthesizes carry/overflow bits to
    # keep the flag varnodes in lockstep with the native
    # instruction's effects. They don't feed the value-path in our
    # v0.1 universe; silent skip means: don't write any cell. If a
    # later op *does* read a flag-output varnode, the cell map
    # lazy-declares a fresh symbolic, making the goal depend on an
    # unknown — z3 returns "unknown" rather than a silently-wrong
    # proof. The skip is sound: it can never turn a real
    # disagreement into a false proof, only into "unknown".
    "INT_CARRY", "INT_SCARRY", "INT_SBORROW",
    # POPCOUNT is x86's parity-flag input (PF = parity of low 8 bits
    # of the result). Same flag-tracking story.
    "POPCOUNT",
})
# Out-of-universe ops we know about by name — bail loud rather than
# silently treating as "unhandled."
_REFUSED_OPCODES = frozenset({
    "BRANCHIND", "CALL", "CALLIND", "CALLOTHER",
})


# ---------- Result shape ----------

@dataclass(frozen=True)
class RelationalResult:
    """Outcome of one (bin, src) relational proof attempt.

    `status` follows the existing provider vocabulary:
      - `proven`   — z3 returned unsat on the negated goal; the bin and
                     src functions are equivalent under the binding.
      - `refuted`  — z3 returned sat; counterexample available in `detail`.
      - `unknown`  — z3 returned unknown, or the encoding ran into an
                     opcode/expression outside the v0 universe.
      - `error`    — toolchain-side failure (no z3 binary, etc.).
    """
    status: Literal["proven", "refuted", "unknown", "error"]
    detail: str
    bin_fn_id: str
    src_fn_id: str
    smt: str | None = None
    artifact_path: Path | None = None
    artifact_hash: str | None = None
    equivalence: Equivalence | None = None  # populated on `proven`


# ---------- Encoder state ----------

@dataclass
class _BinState:
    """Mutable state collected while lowering pcode to SMT-LIB.

    Cells, not varnodes. Each unique `(space, offset)` is a "memory
    cell" that holds a single SMT BV term whose width is the cell's
    current width in bits. Reads at smaller widths extract low bits;
    reads at larger widths bail (would require ambiguous widening).
    Writes replace the cell contents wholesale at the write width.

    The cell-based shape handles register aliasing on x86-64: Ghidra's
    pcode reads RDI as `(register, 0x38, 8)` and EDI as
    `(register, 0x38, 4)` — same physical register, two views. The
    binding for an `int` source param pre-populates the cell at the
    full 8-byte width, then `(register, 0x38, 4)` reads via extract.

    Reads of unbound cells lazily declare a fresh symbolic value so
    the encoder doesn't fail on irrelevant prologue/epilogue ops
    (e.g. RSP load on `ret`). Soundness comes from the goal: if the
    return value depends on this fresh symbolic, z3 finds a model and
    refutes the goal; if it cancels out, we still get unsat.
    """
    cell_term: dict[tuple[str, int], str] = field(default_factory=dict)
    cell_bits: dict[tuple[str, int], int] = field(default_factory=dict)
    decls: list[str] = field(default_factory=list)
    asserts: list[str] = field(default_factory=list)
    next_id: int = 0

    def fresh(self, prefix: str) -> str:
        n = self.next_id
        self.next_id += 1
        return f"{prefix}_{n}"

    def read(self, vn: BinVarnode) -> str:
        """Resolve a non-const input varnode to an SMT term.

        const-space lookup is handled by the caller (`_input_term`);
        this method only handles addressable cells.
        """
        cell = (vn.space, vn.offset)
        wanted_bits = vn.size * 8
        cur_term = self.cell_term.get(cell)
        cur_bits = self.cell_bits.get(cell, 0)
        if cur_term is None:
            fresh = self.fresh(
                f"unbound_{vn.space}_{vn.offset:x}_{wanted_bits}"
            )
            self.decls.append(f"(declare-const {fresh} (_ BitVec {wanted_bits}))")
            self.cell_term[cell] = fresh
            self.cell_bits[cell] = wanted_bits
            return fresh
        if cur_bits == wanted_bits:
            return cur_term
        if cur_bits > wanted_bits:
            # Sub-view: take the low `wanted_bits` of the wider cell.
            return f"((_ extract {wanted_bits - 1} 0) {cur_term})"
        raise _UnsupportedConstruct(
            f"read of {wanted_bits} bits at cell ({vn.space}, "
            f"0x{vn.offset:x}) but cell currently holds only {cur_bits} bits — "
            f"v0 doesn't widen ambiguously (sign- vs zero-extend), "
            f"emit explicit INT_SEXT / INT_ZEXT in pcode"
        )

    def write(self, vn: BinVarnode, computed_term: str, prefix: str) -> str:
        """Assign `computed_term` to the cell named by `vn`. Returns the
        fresh SMT name introduced for the write (a synonym for the
        computed term, used for readable SMT output)."""
        cell = (vn.space, vn.offset)
        wanted_bits = vn.size * 8
        out_name = self.fresh(prefix)
        self.decls.append(f"(declare-const {out_name} (_ BitVec {wanted_bits}))")
        self.asserts.append(f"(assert (= {out_name} {computed_term}))")
        self.cell_term[cell] = out_name
        self.cell_bits[cell] = wanted_bits
        return out_name


@dataclass
class _SrcState:
    """Source-side encoder state. Tracks the SMT vars used for source
    params (one per param) so the binary side can bind to them via the
    signature binding. Source params are declared once up front."""
    param_smt: dict[str, str] = field(default_factory=dict)


# ---------- Source AST → SMT ----------

def _src_return_term_for(fn: Function, state: _SrcState) -> str:
    body = fn.body
    if not isinstance(body, Block):
        raise _UnsupportedConstruct(
            f"function {fn.name!r} body is not a Block "
            f"(got {type(body).__name__})"
        )
    return _src_stmts_to_return_term(body.stmts, fn, state)


def _src_stmts_to_return_term(stmts, fn: Function, state: _SrcState) -> str:
    """Walk a stmt list and return the SMT term for the function's
    eventual return value.

    Two patterns supported in v0.1:
      - `ReturnExpr(expr)` directly returns the encoded expression.
      - `If(cond, then_body, else_body)` produces an ITE; either or
        both branches may be empty (early-return idiom — clang's
        usual output for `if (c) return x; return y;`). When a
        branch is empty, the post-If fallthrough provides that arm's
        return term.

    Anything else (Let, Assign, While, For, ExprStmt with unknown
    side effects, …) bails with `_UnsupportedConstruct`."""
    if not stmts:
        raise _UnsupportedConstruct(
            f"function {fn.name!r}: stmt sequence has no terminating return"
        )
    head, *rest = stmts
    if isinstance(head, ReturnExpr):
        return _src_expr_to_smt(head.value, fn, state)
    if isinstance(head, If):
        cond_term = _src_predicate_to_smt(head.cond, fn, state)
        then_stmts = list(head.then_body.stmts)
        else_stmts = list(head.else_body.stmts)
        # The post-If tail acts as the fall-through arm when one of
        # the If's branches is empty. We compute it lazily; if both
        # branches terminate, we never look at it.
        def tail_term() -> str:
            return _src_stmts_to_return_term(rest, fn, state)
        if then_stmts and else_stmts:
            then_t = _src_stmts_to_return_term(then_stmts, fn, state)
            else_t = _src_stmts_to_return_term(else_stmts, fn, state)
            return f"(ite {cond_term} {then_t} {else_t})"
        if then_stmts and not else_stmts:
            then_t = _src_stmts_to_return_term(then_stmts, fn, state)
            return f"(ite {cond_term} {then_t} {tail_term()})"
        if else_stmts and not then_stmts:
            else_t = _src_stmts_to_return_term(else_stmts, fn, state)
            return f"(ite {cond_term} {tail_term()} {else_t})"
        # Both branches empty — degenerate If, skip and continue.
        return tail_term()
    raise _UnsupportedConstruct(
        f"function {fn.name!r}: unsupported stmt {type(head).__name__} "
        f"in v0.1 universe (Let/Assign/loops not yet handled)"
    )


# Source comparison BinOps and their bitvector-comparison SMT names.
_SRC_CMP_OP: dict[str, str] = {
    "slt": "bvslt", "sle": "bvsle", "sgt": "bvsgt", "sge": "bvsge",
    "ult": "bvult", "ule": "bvule", "ugt": "bvugt", "uge": "bvuge",
    "eq":  "=",     "ne":  "distinct",
}


def _src_predicate_to_smt(expr, fn: Function, state: _SrcState) -> str:
    """Encode an i1-typed source expression as a Bool SMT term (for
    use as the condition in an `(ite ...)` or as a path condition).

    Branch conditions can also be raw arithmetic expressions whose
    truth is "non-zero," but the C ingester always wraps comparison
    BinOps as the cond; this implementation handles only those.
    Other shapes raise `_UnsupportedConstruct`."""
    if isinstance(expr, BinOp) and expr.op in _SRC_CMP_OP:
        smt_op = _SRC_CMP_OP[expr.op]
        l = _src_expr_to_smt(expr.lhs, fn, state)
        r = _src_expr_to_smt(expr.rhs, fn, state)
        return f"({smt_op} {l} {r})"
    raise _UnsupportedConstruct(
        f"can't encode predicate {type(expr).__name__} "
        f"(op={getattr(expr, 'op', None)!r}) for v0.1; "
        f"branches must use comparison BinOps as their condition"
    )


def _src_expr_to_smt(expr, fn: Function, state: _SrcState) -> str:
    match expr:
        case IntLit(type=t, value=v):
            bits = _int_bits(t)
            # SMT-LIB BV literals are unsigned; reinterpret negative
            # ints as their two's-complement representation in the
            # declared width. (`value & ((1<<bits)-1)` is correct for
            # both positive and negative ints.)
            unsigned = v & ((1 << bits) - 1)
            return f"(_ bv{unsigned} {bits})"
        case ParamRef(name=n):
            return state.param_smt[n]
        case BinOp(op=op, lhs=l, rhs=r) if op in ("add", "sub", "mul"):
            ls = _src_expr_to_smt(l, fn, state)
            rs = _src_expr_to_smt(r, fn, state)
            smt_op = {"add": "bvadd", "sub": "bvsub", "mul": "bvmul"}[op]
            return f"({smt_op} {ls} {rs})"
        case Not(operand=op):
            # `!x` on i1: lift to BV by asserting the inner is zero.
            inner_bool = _src_predicate_to_smt(op, fn, state)
            return f"(ite {inner_bool} (_ bv0 1) (_ bv1 1))"
    raise _UnsupportedConstruct(
        f"can't encode source expression {type(expr).__name__} for v0.1"
    )


# ---------- Pcode → SMT ----------

def _input_term(vn: BinVarnode, state: _BinState) -> str:
    """Resolve an input varnode to an SMT term.

    `const`-space varnodes are immediate operands. All other spaces
    delegate to `state.read`, which handles cell-based register
    aliasing and lazy declaration of unbound reads.
    """
    if vn.space == "const":
        bits = vn.size * 8
        unsigned = vn.offset & ((1 << bits) - 1)
        return f"(_ bv{unsigned} {bits})"
    return state.read(vn)


def _emit_op_assert(state: _BinState, op: BinPCodeOp) -> None:
    """Encode one pcode op into SMT and update the cell map."""
    if op.opcode in _SILENTLY_SKIPPED_OPCODES:
        return
    if op.opcode in _TERMINATOR_OPCODES:
        return
    if op.opcode not in _HANDLED_OPCODES:
        raise _UnsupportedConstruct(
            f"opcode {op.opcode!r} not in v0 universe"
        )
    if op.output is None:
        raise _UnsupportedConstruct(
            f"opcode {op.opcode!r} appeared without an output varnode"
        )

    rhs = _encode_op_body(op, state)
    state.write(op.output, rhs, prefix=f"v_{op.opcode.lower()}")


def _encode_op_body(op: BinPCodeOp, state: _BinState) -> str:
    """Build the right-hand-side SMT term for an op (no `(assert ...)`
    wrapper). Caller wraps via `(= out_name <rhs>)`."""
    inputs = [_input_term(vn, state) for vn in op.inputs]
    out_bits = (op.output.size * 8) if op.output is not None else 0

    match op.opcode:
        case "COPY" | "CAST":
            (a,) = inputs
            # CAST is a type-only reinterpretation; for matching widths
            # it's a no-op. Different widths land here when the
            # decompiler relabels a varnode without bit changes — bail
            # if Ghidra emitted a width change since v0 doesn't have
            # rules for that.
            in_bits = op.inputs[0].size * 8
            if in_bits != out_bits:
                raise _UnsupportedConstruct(
                    f"{op.opcode} between {in_bits}-bit and {out_bits}-bit "
                    f"varnodes (only same-width allowed in v0)"
                )
            return a
        case "INT_ADD":
            return f"(bvadd {inputs[0]} {inputs[1]})"
        case "INT_SUB":
            return f"(bvsub {inputs[0]} {inputs[1]})"
        case "INT_MULT":
            return f"(bvmul {inputs[0]} {inputs[1]})"
        case "INT_AND":
            return f"(bvand {inputs[0]} {inputs[1]})"
        case "INT_OR":
            return f"(bvor {inputs[0]} {inputs[1]})"
        case "INT_XOR":
            return f"(bvxor {inputs[0]} {inputs[1]})"
        case "INT_NEGATE" | "INT_2COMP":
            # Ghidra's INT_NEGATE is bitwise NOT; INT_2COMP is two's
            # complement (arithmetic negation). Both are unary on input[0].
            (a,) = inputs
            return f"(bvnot {a})" if op.opcode == "INT_NEGATE" else f"(bvneg {a})"
        case "INT_SEXT":
            (a,) = inputs
            in_bits = op.inputs[0].size * 8
            extra = out_bits - in_bits
            if extra <= 0:
                raise _UnsupportedConstruct(
                    f"INT_SEXT from {in_bits} to {out_bits} bits is not a widening"
                )
            return f"((_ sign_extend {extra}) {a})"
        case "INT_ZEXT":
            (a,) = inputs
            in_bits = op.inputs[0].size * 8
            extra = out_bits - in_bits
            if extra <= 0:
                raise _UnsupportedConstruct(
                    f"INT_ZEXT from {in_bits} to {out_bits} bits is not a widening"
                )
            return f"((_ zero_extend {extra}) {a})"
        case "SUBPIECE":
            # SUBPIECE(value, offset_bytes) takes the slice starting at
            # `offset_bytes` from the LSB of `value`, output width
            # determined by output varnode. SMT extract uses bit indices.
            value, offset_const = inputs
            if op.inputs[1].space != "const":
                raise _UnsupportedConstruct(
                    "SUBPIECE with non-const offset not supported in v0"
                )
            offset_bytes = op.inputs[1].offset
            lo = offset_bytes * 8
            hi = lo + out_bits - 1
            return f"((_ extract {hi} {lo}) {value})"
        case "INT_EQUAL":
            # Comparison ops produce a boolean varnode whose width is
            # whatever Ghidra declared (typically 1 byte = 8 bits on
            # x86-64). The result is 0 or 1 in that width — same value,
            # zero-extended from the logical 1-bit truth.
            return (
                f"(ite (= {inputs[0]} {inputs[1]}) "
                f"(_ bv1 {out_bits}) (_ bv0 {out_bits}))"
            )
        case "INT_NOTEQUAL":
            return (
                f"(ite (distinct {inputs[0]} {inputs[1]}) "
                f"(_ bv1 {out_bits}) (_ bv0 {out_bits}))"
            )
        case "INT_LESS":
            return (
                f"(ite (bvult {inputs[0]} {inputs[1]}) "
                f"(_ bv1 {out_bits}) (_ bv0 {out_bits}))"
            )
        case "INT_LESSEQUAL":
            return (
                f"(ite (bvule {inputs[0]} {inputs[1]}) "
                f"(_ bv1 {out_bits}) (_ bv0 {out_bits}))"
            )
        case "INT_SLESS":
            return (
                f"(ite (bvslt {inputs[0]} {inputs[1]}) "
                f"(_ bv1 {out_bits}) (_ bv0 {out_bits}))"
            )
        case "INT_SLESSEQUAL":
            return (
                f"(ite (bvsle {inputs[0]} {inputs[1]}) "
                f"(_ bv1 {out_bits}) (_ bv0 {out_bits}))"
            )
        case "BOOL_NEGATE":
            (a,) = inputs
            return f"(bvxor {a} (_ bv1 {out_bits}))"
        case "BOOL_AND":
            return f"(bvand {inputs[0]} {inputs[1]})"
        case "BOOL_OR":
            return f"(bvor {inputs[0]} {inputs[1]})"
        case "BOOL_XOR":
            return f"(bvxor {inputs[0]} {inputs[1]})"
    raise _UnsupportedConstruct(f"unhandled opcode {op.opcode!r}")


class _UnsupportedConstruct(Exception):
    """Raised by encoders when they encounter something outside the v0
    universe. The caller catches and converts to status='unknown' with
    the exception's message as the detail."""


# ---------- Path-walker (multi-block, branch-aware) ----------

def _walk_bin_paths(
    bin_fn: BinFunction,
    binding: BinSrcSignatureBinding,
    state: _BinState,
) -> list[tuple[str, str]]:
    """DFS over the binary function's CFG, collecting one
    `(path_condition, return_term)` per execution path that reaches
    `RETURN`.

    Position-aware: a "position" is `(bb_id, op_index)`, so a CBRANCH
    whose target is within the SAME basic block (clang -O1's
    common shape on x86-64) is handled the same way as one that
    crosses to a sibling block. Successors-based traversal only
    kicks in when a block exhausts its ops.

    Per-path state isolation: cell-map snapshots are taken before
    branching and restored when the recursion unwinds. The shared
    `decls`/`asserts` lists accumulate across all paths because every
    fresh SMT name is unique (the next_id counter is shared).

    Loop detection: each (bb_id, op_index) may be visited at most
    once per path. The visited set is passed by value (frozenset)
    so siblings don't see each other's history.
    """
    bb_by_id = {bb.id: bb for bb in bin_fn.basic_blocks}

    # Map each pcode op's source address to its (bb_id, op_index) so
    # branch targets in `ram` space resolve to walker positions.
    # Multiple ops can share a source_address (Ghidra emits several
    # pcode ops per native instruction); the FIRST op at that
    # address is the natural jump target.
    address_to_pos: dict[int, tuple[str, int]] = {}
    for bb in bin_fn.basic_blocks:
        for i, op in enumerate(bb.pcode_ops):
            address_to_pos.setdefault(op.source_address, (bb.id, i))

    results: list[tuple[str, str]] = []

    def lookup_target(vn: BinVarnode) -> tuple[str, int] | None:
        """Resolve a CBRANCH/BRANCH target varnode to a walker
        position. Today only `ram` and `const` jumps are supported;
        register-indirect and stack jumps surface as None."""
        if vn.space not in ("ram", "const"):
            return None
        return address_to_pos.get(vn.offset)

    def visit(
        bb_id: str, op_index: int,
        path_cond: str, visited: frozenset[tuple[str, int]],
    ) -> None:
        pos = (bb_id, op_index)
        if pos in visited:
            raise _UnsupportedConstruct(
                f"loop detected: re-entered position {bb_id}@{op_index} "
                f"along the same path"
            )
        visited = visited | {pos}

        bb = bb_by_id[bb_id]
        ops = bb.pcode_ops
        i = op_index
        while i < len(ops):
            op = ops[i]
            if op.opcode in _REFUSED_OPCODES:
                raise _UnsupportedConstruct(
                    f"opcode {op.opcode!r} not in v0.1 universe "
                    f"(at 0x{op.source_address:x})"
                )
            if op.opcode == "RETURN":
                ret_term = state.read(binding.return_binding)
                results.append((path_cond, ret_term))
                return
            if op.opcode == "BRANCH":
                target = lookup_target(op.inputs[0]) if op.inputs else None
                if target is None:
                    raise _UnsupportedConstruct(
                        f"BRANCH at 0x{op.source_address:x} has "
                        f"unresolved target"
                    )
                visit(target[0], target[1], path_cond, visited)
                return
            if op.opcode == "CBRANCH":
                if len(op.inputs) < 2:
                    raise _UnsupportedConstruct(
                        f"CBRANCH at 0x{op.source_address:x} has "
                        f"{len(op.inputs)} inputs (expected 2)"
                    )
                cond_vn = op.inputs[1]
                cond_bv = _input_term(cond_vn, state)
                cond_bool = f"(distinct {cond_bv} (_ bv0 {cond_vn.size * 8}))"

                target = lookup_target(op.inputs[0])
                if target is None:
                    raise _UnsupportedConstruct(
                        f"CBRANCH at 0x{op.source_address:x} target "
                        f"{op.inputs[0].space}:0x{op.inputs[0].offset:x} "
                        f"resolves to no walker position"
                    )

                # False arm starts at the position physically after
                # the CBRANCH: another op in the same block if there
                # is one (clang -O1's intra-block branching shape),
                # OR a sibling block reached via a "false" /
                # "fallthrough" edge (the synthetic / clean
                # inter-block shape).
                if i + 1 < len(ops):
                    false_pos: tuple[str, int] | None = (bb_id, i + 1)
                else:
                    false_edge = next(
                        (e for e in bb.successors
                         if e.edge_kind in ("false", "fallthrough")),
                        None,
                    )
                    if false_edge is None or false_edge.successor_id not in bb_by_id:
                        raise _UnsupportedConstruct(
                            f"CBRANCH at 0x{op.source_address:x} has no "
                            f"reachable false-arm position"
                        )
                    false_pos = (false_edge.successor_id, 0)

                # Snapshot cell map; walk true arm first, restore,
                # walk false arm.
                snap = (dict(state.cell_term), dict(state.cell_bits))
                visit(
                    target[0], target[1],
                    _conj(path_cond, cond_bool),
                    visited,
                )
                state.cell_term, state.cell_bits = (
                    dict(snap[0]), dict(snap[1]),
                )
                visit(
                    false_pos[0], false_pos[1],
                    _conj(path_cond, f"(not {cond_bool})"),
                    visited,
                )
                state.cell_term, state.cell_bits = snap[0], snap[1]
                return

            _emit_op_assert(state, op)
            i += 1

        # Ops exhausted without a terminator. Try a single fallthrough
        # successor.
        fallthrough = [
            e for e in bb.successors
            if e.edge_kind in ("fallthrough", "unconditional", "call_return")
        ]
        if len(fallthrough) == 1:
            target_id = fallthrough[0].successor_id
            if target_id not in bb_by_id:
                raise _UnsupportedConstruct(
                    f"fallthrough target {target_id} not in this function"
                )
            visit(target_id, 0, path_cond, visited)
            return
        raise _UnsupportedConstruct(
            f"block at 0x{bb.start_address:x} exhausted without a "
            f"terminator and {len(fallthrough)} fallthrough successors"
        )

    visit(bin_fn.basic_blocks[0].id, 0, "true", frozenset())
    return results


def _conj(a: str, b: str) -> str:
    """Bool conjunction with simple identity simplification — `(and
    true x)` collapses to `x`, etc. Keeps the generated SMT readable."""
    if a == "true":
        return b
    if b == "true":
        return a
    if a == "false" or b == "false":
        return "false"
    return f"(and {a} {b})"


def _fold_paths_to_ite(paths: list[tuple[str, str]]) -> str:
    """Fold a list of `(path_cond, return_term)` tuples into a
    nested ITE chain. The last path becomes the unconditional default
    — this is sound when the path conditions are mutually exclusive
    and exhaustive (the guarantee that follows from a well-formed CFG
    walked via `_walk_bin_paths`)."""
    if len(paths) == 1:
        return paths[0][1]
    head_cond, head_ret = paths[0]
    rest = _fold_paths_to_ite(paths[1:])
    return f"(ite {head_cond} {head_ret} {rest})"


# ---------- Top-level: encode one binding, run z3 ----------

def _encode_binding(
    binding: BinSrcSignatureBinding,
    bin_fn: BinFunction,
    src_fn: Function,
) -> tuple[str, _SrcState, _BinState]:
    """Build a complete SMT-LIB problem for one bin↔src pair.

    Returns `(smt_text, src_state, bin_state)`. The text is `unsat`
    iff the two functions agree on the return value for all inputs.

    Raises `_UnsupportedConstruct` if either side fell outside v0.
    """
    src_state = _SrcState()

    # Source params declared once; they're the only "free variables"
    # in the joint encoding. The binary side aliases its input
    # varnodes to these via the signature binding.
    param_decls: list[str] = []
    for p in src_fn.params:
        bits = _int_bits(p.type)
        smt_name = f"src_{p.name}"
        src_state.param_smt[p.name] = smt_name
        param_decls.append(f"(declare-const {smt_name} (_ BitVec {bits}))")

    # Encode the source return.
    src_return_term = _src_return_term_for(src_fn, src_state)

    # Pre-bind input registers to source params per the binding.
    #
    # The binding's varnode size matches the *source param* width
    # (e.g. 4 bytes for `int`). The *physical register* is wider on
    # x86-64 (RDI is 8 bytes; the low 4 bytes are EDI). Real pcode
    # often reads the full 8-byte register, so we pre-populate the
    # cell at register width and constrain only the low bits to equal
    # the source param — leaving the high bits as a fresh symbolic.
    # Per the SysV ABI the high bits of an int-passing register are
    # undefined on entry, so this captures the "free variable" the
    # binary must not depend on.
    bin_state = _BinState()
    reg_bits = _abi_register_bits(binding.abi)
    for pb in binding.param_bindings:
        smt_var = src_state.param_smt.get(pb.param_name)
        if smt_var is None:
            raise _UnsupportedConstruct(
                f"binding references unknown source param {pb.param_name!r}"
            )
        binding_bits = pb.varnode.size * 8
        cell = (pb.varnode.space, pb.varnode.offset)
        if pb.varnode.space != "register" or binding_bits >= reg_bits:
            bin_state.cell_term[cell] = smt_var
            bin_state.cell_bits[cell] = binding_bits
            continue
        # Sub-register binding: declare a fresh full-width register,
        # constrain low bits to the source param, leave high bits free.
        full = bin_state.fresh(f"reg_init_{pb.param_name}")
        bin_state.decls.append(f"(declare-const {full} (_ BitVec {reg_bits}))")
        bin_state.asserts.append(
            f"(assert (= ((_ extract {binding_bits - 1} 0) {full}) {smt_var}))"
        )
        bin_state.cell_term[cell] = full
        bin_state.cell_bits[cell] = reg_bits

    # Multi-path walk from entry block. Each (path_cond, return_term)
    # tuple captures one execution path's final RETURN. Mutually-
    # exclusive path conditions guarantee well-defined ITE folding
    # at the end.
    if not bin_fn.basic_blocks:
        raise _UnsupportedConstruct(
            f"binary function {bin_fn.demangled_name!r} has no basic blocks"
        )
    paths = _walk_bin_paths(bin_fn, binding, bin_state)
    if not paths:
        raise _UnsupportedConstruct(
            f"binary function {bin_fn.demangled_name!r} has no return paths"
        )
    bin_return_term = _fold_paths_to_ite(paths)

    lines: list[str] = []
    lines.append(
        f"; auto-generated by quod.predicate.binary_relational for "
        f"({bin_fn.demangled_name!r}, {src_fn.name!r})"
    )
    lines.append(f"; binding ABI: {binding.abi}")
    lines.append("(set-logic QF_BV)")
    lines.append("")
    lines.append("; source parameters (also serve as binary input registers)")
    lines.extend(param_decls)
    lines.append("")
    lines.append("; binary pcode encoding")
    lines.extend(bin_state.decls)
    lines.append("")
    lines.extend(bin_state.asserts)
    lines.append("")
    lines.append("; goal (negated): bin and src return the same value")
    lines.append(f"(assert (not (= {bin_return_term} {src_return_term})))")
    lines.append("")
    lines.append("(check-sat)")
    lines.append("(get-model)")
    lines.append("(exit)")
    return "\n".join(lines) + "\n", src_state, bin_state


def prove_bin_relational_pair(
    program: Program,
    binding: BinSrcSignatureBinding,
    *,
    proofs_dir: Path,
) -> RelationalResult:
    """Encode and attempt to prove one (bin.fn, src.fn) pair.

    Looks up both endpoints in `program`. Returns a `RelationalResult`
    describing the outcome; on `proven`, `equivalence` carries an
    `Equivalence` node ready to merge into the program.
    """
    bin_fn = _find_bin_fn(program, binding.bin_fn_id)
    src_fn = _find_src_fn(program, binding.src_fn_id)
    if bin_fn is None or src_fn is None:
        return RelationalResult(
            status="error",
            detail=(
                f"binding endpoints not found in program: "
                f"bin={binding.bin_fn_id}, src={binding.src_fn_id}"
            ),
            bin_fn_id=binding.bin_fn_id,
            src_fn_id=binding.src_fn_id,
        )

    try:
        smt, _src_state, _bin_state = _encode_binding(binding, bin_fn, src_fn)
    except _UnsupportedConstruct as e:
        return RelationalResult(
            status="unknown",
            detail=f"encoding refused: {e}",
            bin_fn_id=binding.bin_fn_id,
            src_fn_id=binding.src_fn_id,
        )

    try:
        z3_result = run_z3_on_smt(smt)
    except Z3NotInstalled as e:
        return RelationalResult(
            status="error",
            detail=str(e),
            bin_fn_id=binding.bin_fn_id,
            src_fn_id=binding.src_fn_id,
            smt=smt,
        )

    artifact_hash = hashlib.sha256(smt.encode("utf-8")).hexdigest()
    proofs_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = proofs_dir / (
        f"bin_relational_{src_fn.name}_{artifact_hash[:12]}.smt2"
    )
    artifact_path.write_text(smt)

    if z3_result.status == "unsat":
        equivalence = Equivalence(
            a_node_id=src_fn.id,
            b_node_id=bin_fn.id,
            regime="witness",
            justification=Z3Justification(
                artifact_path=str(artifact_path),
                artifact_hash=artifact_hash,
                body_smt_hash=artifact_hash,
                note=(
                    f"z3.bin_relational: bin {bin_fn.demangled_name!r} "
                    f"~ src {src_fn.name!r} under {binding.abi}"
                ),
            ),
        )
        return RelationalResult(
            status="proven",
            detail=f"z3 unsat ({len(smt.splitlines())}-line problem)",
            bin_fn_id=binding.bin_fn_id,
            src_fn_id=binding.src_fn_id,
            smt=smt,
            artifact_path=artifact_path,
            artifact_hash=artifact_hash,
            equivalence=equivalence,
        )
    if z3_result.status == "sat":
        return RelationalResult(
            status="refuted",
            detail=(
                f"z3 sat — bin and src disagree on at least one input.\n"
                f"counterexample model:\n{z3_result.raw}"
            ),
            bin_fn_id=binding.bin_fn_id,
            src_fn_id=binding.src_fn_id,
            smt=smt,
            artifact_path=artifact_path,
            artifact_hash=artifact_hash,
        )
    return RelationalResult(
        status="unknown",
        detail=f"z3 returned {z3_result.status!r}",
        bin_fn_id=binding.bin_fn_id,
        src_fn_id=binding.src_fn_id,
        smt=smt,
        artifact_path=artifact_path,
        artifact_hash=artifact_hash,
    )


def prove_all_bin_relational(
    program: Program, *, proofs_dir: Path,
) -> tuple[RelationalResult, ...]:
    """Run the relational prover over every signature binding in `program`.

    Returns one `RelationalResult` per binding in the order they appear
    in `program.signature_bindings`. Callers fold the results into the
    program (e.g. by appending each `proven` result's `equivalence` to
    `program.equivalences`)."""
    return tuple(
        prove_bin_relational_pair(program, b, proofs_dir=proofs_dir)
        for b in program.signature_bindings
    )


# ---------- Internal lookup helpers ----------

def _find_bin_fn(program: Program, bin_fn_id: str) -> BinFunction | None:
    for u in program.binary_units:
        for f in u.functions:
            if f.id == bin_fn_id:
                return f
    return None


def _find_src_fn(program: Program, src_fn_id: str) -> Function | None:
    for f in program.functions:
        if f.id == src_fn_id:
            return f
    return None
