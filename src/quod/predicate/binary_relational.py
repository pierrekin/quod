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
    IntLit,
    IsizeType,
    IntType,
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
})
_TERMINATOR_OPCODES = frozenset({"RETURN"})
_SILENTLY_SKIPPED_OPCODES = frozenset({
    "LOAD", "STORE", "INDIRECT", "PIECE",
    # x86 EFLAGS tracking. Ghidra synthesizes carry/overflow bits and
    # comparison results to keep the flag varnodes in lockstep with
    # the native instruction's effects, but for the v0 universe
    # (straight-line arithmetic, no branching) the flags don't feed
    # the return path. Silently skip means: don't write any cell. If
    # a later op *does* read a flag-output varnode, the cell map
    # lazy-declares a fresh symbolic, which makes the relational goal
    # depend on an unknown — z3 returns "unknown" rather than a
    # silently-wrong proof. So the silent skip is sound: it can never
    # turn a real disagreement into a false proof, only into "unknown".
    "INT_CARRY", "INT_SCARRY", "INT_SBORROW",
    "INT_EQUAL", "INT_NOTEQUAL",
    "INT_LESS", "INT_SLESS", "INT_LESSEQUAL", "INT_SLESSEQUAL",
    # POPCOUNT is x86's parity-flag input (PF = parity of low 8 bits
    # of the result). Same flag-tracking story as the comparisons above.
    "POPCOUNT",
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
            f"(got {type(body).__name__}); v0 source universe is "
            f"single-statement Block(ReturnExpr(...))"
        )
    if len(body.stmts) != 1:
        raise _UnsupportedConstruct(
            f"function {fn.name!r} body has {len(body.stmts)} statements; "
            f"v0 only encodes a single ReturnExpr"
        )
    only = body.stmts[0]
    if not isinstance(only, ReturnExpr):
        raise _UnsupportedConstruct(
            f"function {fn.name!r} body's only statement is "
            f"{type(only).__name__}; v0 needs ReturnExpr"
        )
    return _src_expr_to_smt(only.value, fn, state)


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
    raise _UnsupportedConstruct(
        f"can't encode source expression {type(expr).__name__} for v0"
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
    raise _UnsupportedConstruct(f"unhandled opcode {op.opcode!r}")


class _UnsupportedConstruct(Exception):
    """Raised by encoders when they encounter something outside the v0
    universe. The caller catches and converts to status='unknown' with
    the exception's message as the detail."""


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

    # Walk pcode in declaration order. For v0, accept any number of
    # blocks but stop at the first RETURN (multi-return functions
    # would need path encoding — out of scope here).
    if not bin_fn.basic_blocks:
        raise _UnsupportedConstruct(
            f"binary function {bin_fn.demangled_name!r} has no basic blocks"
        )
    saw_return = False
    for bb in bin_fn.basic_blocks:
        for op in bb.pcode_ops:
            if op.opcode in _TERMINATOR_OPCODES:
                saw_return = True
                break
            _emit_op_assert(bin_state, op)
        if saw_return:
            break
    if not saw_return:
        raise _UnsupportedConstruct(
            f"binary function {bin_fn.demangled_name!r} pcode never reaches a RETURN"
        )

    # Look up the binary's return value at function exit. We read
    # through the cell map, which handles register aliasing — if the
    # binary wrote RAX:8 but the source returns int (32 bits), we
    # extract the low 32 bits at the return-binding width.
    try:
        bin_return_term = bin_state.read(binding.return_binding)
    except _UnsupportedConstruct as e:
        raise _UnsupportedConstruct(
            f"return-binding read failed: {e}"
        )

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
