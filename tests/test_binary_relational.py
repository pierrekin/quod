"""`z3.bin_relational` — relational equivalence prover, BV-encoded.

Synthetic-pcode tests: each test hand-constructs a small `BinFunction`
with a few p-code ops, pairs it with a matching source `Function`,
seeds the binding, and runs the prover. No Ghidra dependency.

The end-to-end against a real `.so` lives in `test_binary_e2e.py`.
"""
from __future__ import annotations

import shutil

import pytest

from quod.model import (
    BinaryProvenance,
    BinBasicBlock,
    BinFunction,
    BinFunctionParam,
    BinPCodeOp,
    BinSrcParamBinding,
    BinSrcSignatureBinding,
    BinUnit,
    BinVarnode,
    Block,
    BinOp,
    Equivalence,
    Function,
    I32Type,
    IntLit,
    Param,
    ParamRef,
    Program,
    ReturnExpr,
)
from quod.predicate.binary_relational import (
    prove_all_bin_relational,
    prove_bin_relational_pair,
)


pytestmark = pytest.mark.skipif(
    shutil.which("z3") is None, reason="z3 binary not on PATH"
)


# ---------- Helpers ----------

def _vn(space: str, offset: int, size: int = 4) -> BinVarnode:
    return BinVarnode(space=space, offset=offset, size=size)


def _rdi() -> BinVarnode:
    return _vn("register", 0x38)


def _rsi() -> BinVarnode:
    return _vn("register", 0x30)


def _rax() -> BinVarnode:
    return _vn("register", 0x00)


def _const(value: int, size: int = 4) -> BinVarnode:
    """A pcode const-space varnode. SMT BV literals are unsigned, so
    Ghidra encodes negative consts via two's-complement in `offset`."""
    return BinVarnode(space="const", offset=value & ((1 << (size * 8)) - 1), size=size)


def _bb(*ops: BinPCodeOp, addr: int = 0x401000) -> BinBasicBlock:
    return BinBasicBlock(
        start_address=addr,
        end_address=addr + len(ops) * 4,
        pcode_ops=tuple(ops),
    )


def _bin_fn(name: str, n_params: int, *blocks: BinBasicBlock) -> BinFunction:
    return BinFunction(
        address=0x401000,
        mangled_name=name,
        demangled_name=name,
        return_type_name="int",
        params=tuple(
            BinFunctionParam(name=f"param_{i+1}", type_name="int")
            for i in range(n_params)
        ),
        calling_convention="x86_64-sysv",
        basic_blocks=tuple(blocks),
    )


def _src_return_expr(expr) -> Block:
    return Block(stmts=(ReturnExpr(value=expr),))


def _bind(src: Function, bin_fn: BinFunction) -> BinSrcSignatureBinding:
    """Build a binding mirroring lift_v2's output for SysV int params."""
    sysv_offsets = (0x38, 0x30, 0x10, 0x08, 0x80, 0x88)
    pbs = tuple(
        BinSrcParamBinding(
            varnode=BinVarnode(space="register", offset=sysv_offsets[i], size=4),
            param_name=p.name,
        )
        for i, p in enumerate(src.params)
    )
    return BinSrcSignatureBinding(
        bin_fn_id=bin_fn.id,
        src_fn_id=src.id,
        abi="x86_64-sysv",
        param_bindings=pbs,
        return_binding=BinVarnode(space="register", offset=0x00, size=4),
    )


def _program(src: Function, bin_fn: BinFunction, *, with_binding: bool = True) -> Program:
    bu = BinUnit(
        path="/tmp/x.so", sha256="0" * 64,
        arch="x86_64", file_format="elf",
        functions=(bin_fn,),
    )
    eq = Equivalence(
        a_node_id=src.id,
        b_node_id=bin_fn.id,
        justification=BinaryProvenance(
            binary_path="/tmp/x.so", binary_sha256="0" * 64,
            binary_symbol=bin_fn.mangled_name, source_evidence="symtab",
        ),
    )
    bindings = (_bind(src, bin_fn),) if with_binding else ()
    return Program(
        functions=(src,),
        binary_units=(bu,),
        equivalences=(eq,),
        signature_bindings=bindings,
    )


# ---------- Identity ----------

def test_identity_function_proves(tmp_path):
    """`int ident(int x) { return x; }` ≡ `mov eax, edi; ret`.

    Pcode: COPY EDI:4 → EAX:4 ; RETURN.
    """
    src = Function(
        name="ident",
        params=(Param(name="x", type=I32Type()),),
        return_type=I32Type(),
        body=_src_return_expr(ParamRef(name="x")),
    )
    bin_fn = _bin_fn("ident", 1, _bb(
        BinPCodeOp(opcode="COPY", inputs=(_rdi(),), output=_rax(), source_address=0x401000),
        BinPCodeOp(opcode="RETURN", inputs=(), output=None, source_address=0x401004),
    ))
    prog = _program(src, bin_fn)

    results = prove_all_bin_relational(prog, proofs_dir=tmp_path)
    assert len(results) == 1
    r = results[0]
    assert r.status == "proven", f"expected proven; got {r.status}: {r.detail}"
    assert r.equivalence is not None
    assert r.equivalence.regime == "witness"
    assert r.equivalence.justification.kind == "z3"
    assert r.artifact_path is not None and r.artifact_path.exists()


# ---------- Add ----------

def test_add_function_proves(tmp_path):
    """`int add(int a, int b) { return a + b; }`."""
    src = Function(
        name="add",
        params=(Param(name="a", type=I32Type()), Param(name="b", type=I32Type())),
        return_type=I32Type(),
        body=_src_return_expr(BinOp(
            op="add", lhs=ParamRef(name="a"), rhs=ParamRef(name="b"),
        )),
    )
    bin_fn = _bin_fn("add", 2, _bb(
        BinPCodeOp(
            opcode="INT_ADD",
            inputs=(_rdi(), _rsi()),
            output=_rax(),
            source_address=0x401000,
        ),
        BinPCodeOp(opcode="RETURN", inputs=(), output=None, source_address=0x401004),
    ))
    prog = _program(src, bin_fn)

    [r] = prove_all_bin_relational(prog, proofs_dir=tmp_path)
    assert r.status == "proven", r.detail


# ---------- Affine: 3*x + 5 ----------

def test_affine_function_proves(tmp_path):
    """`int affine(int x) { return 3 * x + 5; }` via two ops."""
    src = Function(
        name="affine",
        params=(Param(name="x", type=I32Type()),),
        return_type=I32Type(),
        body=_src_return_expr(BinOp(
            op="add",
            lhs=BinOp(op="mul", lhs=IntLit(type=I32Type(), value=3), rhs=ParamRef(name="x")),
            rhs=IntLit(type=I32Type(), value=5),
        )),
    )
    # Pcode: tmp = 3 * EDI; EAX = tmp + 5; RETURN
    tmp_var = _vn("unique", 0x100, 4)
    bin_fn = _bin_fn("affine", 1, _bb(
        BinPCodeOp(
            opcode="INT_MULT",
            inputs=(_const(3), _rdi()),
            output=tmp_var,
            source_address=0x401000,
        ),
        BinPCodeOp(
            opcode="INT_ADD",
            inputs=(tmp_var, _const(5)),
            output=_rax(),
            source_address=0x401004,
        ),
        BinPCodeOp(opcode="RETURN", inputs=(), output=None, source_address=0x401008),
    ))
    prog = _program(src, bin_fn)

    [r] = prove_all_bin_relational(prog, proofs_dir=tmp_path)
    assert r.status == "proven", r.detail


# ---------- Refutation: source says +, binary says - ----------

def test_disagreeing_function_refuted(tmp_path):
    """Source returns `a + b`; binary returns `a - b`. z3 must refute."""
    src = Function(
        name="add",
        params=(Param(name="a", type=I32Type()), Param(name="b", type=I32Type())),
        return_type=I32Type(),
        body=_src_return_expr(BinOp(
            op="add", lhs=ParamRef(name="a"), rhs=ParamRef(name="b"),
        )),
    )
    bin_fn = _bin_fn("add", 2, _bb(
        BinPCodeOp(
            opcode="INT_SUB",  # NB: subtract, not add
            inputs=(_rdi(), _rsi()),
            output=_rax(),
            source_address=0x401000,
        ),
        BinPCodeOp(opcode="RETURN", inputs=(), output=None, source_address=0x401004),
    ))
    prog = _program(src, bin_fn)

    [r] = prove_all_bin_relational(prog, proofs_dir=tmp_path)
    assert r.status == "refuted", r.detail
    assert r.equivalence is None
    # Counterexample appears in detail.
    assert "sat" in r.detail or "model" in r.detail


# ---------- Out-of-universe: bail with unknown ----------

def test_unknown_opcode_yields_unknown(tmp_path):
    """An op outside both the handled and silently-skipped sets
    bails with status='unknown' — `CBRANCH` (conditional branch)
    sits squarely there: v0 is straight-line, branching breaks the
    encoder's path-free assumption."""
    src = Function(
        name="ident",
        params=(Param(name="x", type=I32Type()),),
        return_type=I32Type(),
        body=_src_return_expr(ParamRef(name="x")),
    )
    bin_fn = _bin_fn("ident", 1, _bb(
        BinPCodeOp(
            opcode="CBRANCH",  # control flow — out of v0 universe
            inputs=(_const(0x401010, size=8), _rdi()),
            output=None,
            source_address=0x401000,
        ),
        BinPCodeOp(opcode="RETURN", inputs=(), output=None, source_address=0x401004),
    ))
    prog = _program(src, bin_fn)

    [r] = prove_all_bin_relational(prog, proofs_dir=tmp_path)
    assert r.status == "unknown"
    assert "CBRANCH" in r.detail


def test_no_signature_bindings_yields_no_results(tmp_path):
    """A program with no bindings yields zero relational results."""
    src = Function(
        name="ident",
        params=(Param(name="x", type=I32Type()),),
        return_type=I32Type(),
        body=_src_return_expr(ParamRef(name="x")),
    )
    bin_fn = _bin_fn("ident", 1, _bb(
        BinPCodeOp(opcode="COPY", inputs=(_rdi(),), output=_rax(), source_address=0x401000),
        BinPCodeOp(opcode="RETURN", inputs=(), output=None, source_address=0x401004),
    ))
    prog = _program(src, bin_fn, with_binding=False)

    assert prove_all_bin_relational(prog, proofs_dir=tmp_path) == ()


# ---------- Two functions in one program ----------

def test_two_pairs_each_proven_independently(tmp_path):
    """`ident` and `add` in the same program; each gets its own
    binding, prover, and result."""
    ident = Function(
        name="ident",
        params=(Param(name="x", type=I32Type()),),
        return_type=I32Type(),
        body=_src_return_expr(ParamRef(name="x")),
    )
    ident_bin = _bin_fn("ident", 1, _bb(
        BinPCodeOp(opcode="COPY", inputs=(_rdi(),), output=_rax(), source_address=0x401000),
        BinPCodeOp(opcode="RETURN", inputs=(), output=None, source_address=0x401004),
    ))
    add = Function(
        name="add",
        params=(Param(name="a", type=I32Type()), Param(name="b", type=I32Type())),
        return_type=I32Type(),
        body=_src_return_expr(BinOp(
            op="add", lhs=ParamRef(name="a"), rhs=ParamRef(name="b"),
        )),
    )
    add_bin = _bin_fn("add", 2, _bb(
        BinPCodeOp(opcode="INT_ADD", inputs=(_rdi(), _rsi()), output=_rax(), source_address=0x402000),
        BinPCodeOp(opcode="RETURN", inputs=(), output=None, source_address=0x402004),
    ))

    bu = BinUnit(
        path="/tmp/x.so", sha256="0" * 64,
        arch="x86_64", file_format="elf",
        functions=(ident_bin, add_bin),
    )
    eqs = (
        Equivalence(
            a_node_id=ident.id, b_node_id=ident_bin.id,
            justification=BinaryProvenance(
                binary_path="/tmp/x.so", binary_sha256="0" * 64,
                binary_symbol="ident", source_evidence="symtab",
            ),
        ),
        Equivalence(
            a_node_id=add.id, b_node_id=add_bin.id,
            justification=BinaryProvenance(
                binary_path="/tmp/x.so", binary_sha256="0" * 64,
                binary_symbol="add", source_evidence="symtab",
            ),
        ),
    )
    bindings = (_bind(ident, ident_bin), _bind(add, add_bin))
    prog = Program(
        functions=(ident, add),
        binary_units=(bu,),
        equivalences=eqs,
        signature_bindings=bindings,
    )

    results = prove_all_bin_relational(prog, proofs_dir=tmp_path)
    assert len(results) == 2
    assert {r.status for r in results} == {"proven"}


# ---------- Branch encoding (v0.1) ----------

# Helpers for branched functions: we model `if (x < 0) return -x; return x;`
# as three basic blocks: entry (compare + CBRANCH), then-arm
# (negate + RETURN), and else-arm (RETURN). Successor edges tag the
# CBRANCH's two arms.

from quod.model import BinBlockEdge, If


def _bb_with_succ(
    *ops: BinPCodeOp,
    addr: int = 0x401000,
    successors: tuple[BinBlockEdge, ...] = (),
    bb_id: str | None = None,
) -> BinBasicBlock:
    bb = BinBasicBlock(
        start_address=addr,
        end_address=addr + len(ops) * 4,
        pcode_ops=tuple(ops),
        successors=successors,
    )
    if bb_id is not None:
        bb = bb.model_copy(update={"id": bb_id})
    return bb


def test_branch_function_proves_under_explicit_else(tmp_path):
    """`int sign_only(int x) { if (x < 0) return -1; else return 1; }`.
    Both arms terminate explicitly; the source encoder produces an
    ITE directly; the binary side has a single CBRANCH between two
    return blocks."""
    src = Function(
        name="sign_only",
        params=(Param(name="x", type=I32Type()),),
        return_type=I32Type(),
        body=Block(stmts=(If(
            cond=BinOp(op="slt", lhs=ParamRef(name="x"),
                       rhs=IntLit(type=I32Type(), value=0)),
            then_body=Block(stmts=(ReturnExpr(value=IntLit(
                type=I32Type(), value=-1,
            )),)),
            else_body=Block(stmts=(ReturnExpr(value=IntLit(
                type=I32Type(), value=1,
            )),)),
        ),)),
    )

    # Synthetic pcode: entry block does INT_SLESS into a 1-bit
    # cond, then CBRANCH to "then" if true; else fallthrough to
    # "else". Both arms write a constant to RAX and RETURN.
    cond_var = BinVarnode(space="unique", offset=0x100, size=1)
    entry = _bb_with_succ(
        BinPCodeOp(
            opcode="INT_SLESS",
            inputs=(_rdi(), _const(0)),
            output=cond_var,
            source_address=0x401000,
        ),
        BinPCodeOp(
            opcode="CBRANCH",
            inputs=(_const(0x401020, size=8), cond_var),
            output=None,
            source_address=0x401004,
        ),
        addr=0x401000,
        bb_id="@bb_entry",
    )
    then_bb = _bb_with_succ(
        BinPCodeOp(
            opcode="COPY",
            inputs=(_const(-1),),
            output=_rax(),
            source_address=0x401020,
        ),
        BinPCodeOp(opcode="RETURN", inputs=(), output=None,
                   source_address=0x401024),
        addr=0x401020,
        bb_id="@bb_then",
    )
    else_bb = _bb_with_succ(
        BinPCodeOp(
            opcode="COPY",
            inputs=(_const(1),),
            output=_rax(),
            source_address=0x401010,
        ),
        BinPCodeOp(opcode="RETURN", inputs=(), output=None,
                   source_address=0x401014),
        addr=0x401010,
        bb_id="@bb_else",
    )
    entry = entry.model_copy(update={"successors": (
        BinBlockEdge(successor_id="@bb_then", edge_kind="true"),
        BinBlockEdge(successor_id="@bb_else", edge_kind="false"),
    )})
    bin_fn = _bin_fn("sign_only", 1, entry, then_bb, else_bb)

    prog = _program(src, bin_fn)
    [r] = prove_all_bin_relational(prog, proofs_dir=tmp_path)
    assert r.status == "proven", r.detail


def test_branch_function_proves_under_early_return(tmp_path):
    """`int sign_only(int x) { if (x < 0) return -1; return 1; }`.
    The source has Block(stmts=(If(..., empty), ReturnExpr)) — early
    return idiom. The encoder treats post-If as the else arm."""
    src = Function(
        name="sign_only",
        params=(Param(name="x", type=I32Type()),),
        return_type=I32Type(),
        body=Block(stmts=(
            If(
                cond=BinOp(op="slt", lhs=ParamRef(name="x"),
                           rhs=IntLit(type=I32Type(), value=0)),
                then_body=Block(stmts=(ReturnExpr(value=IntLit(
                    type=I32Type(), value=-1,
                )),)),
                else_body=Block(stmts=()),
            ),
            ReturnExpr(value=IntLit(type=I32Type(), value=1)),
        )),
    )

    cond_var = BinVarnode(space="unique", offset=0x100, size=1)
    entry = _bb_with_succ(
        BinPCodeOp(opcode="INT_SLESS",
                   inputs=(_rdi(), _const(0)),
                   output=cond_var, source_address=0x401000),
        BinPCodeOp(opcode="CBRANCH",
                   inputs=(_const(0x401020, size=8), cond_var),
                   output=None, source_address=0x401004),
        addr=0x401000, bb_id="@bb_entry",
    )
    then_bb = _bb_with_succ(
        BinPCodeOp(opcode="COPY", inputs=(_const(-1),),
                   output=_rax(), source_address=0x401020),
        BinPCodeOp(opcode="RETURN", inputs=(), output=None,
                   source_address=0x401024),
        addr=0x401020, bb_id="@bb_then",
    )
    else_bb = _bb_with_succ(
        BinPCodeOp(opcode="COPY", inputs=(_const(1),),
                   output=_rax(), source_address=0x401010),
        BinPCodeOp(opcode="RETURN", inputs=(), output=None,
                   source_address=0x401014),
        addr=0x401010, bb_id="@bb_else",
    )
    entry = entry.model_copy(update={"successors": (
        BinBlockEdge(successor_id="@bb_then", edge_kind="true"),
        BinBlockEdge(successor_id="@bb_else", edge_kind="false"),
    )})
    bin_fn = _bin_fn("sign_only", 1, entry, then_bb, else_bb)

    prog = _program(src, bin_fn)
    [r] = prove_all_bin_relational(prog, proofs_dir=tmp_path)
    assert r.status == "proven", r.detail


def test_branch_function_refuted_when_then_arm_disagrees(tmp_path):
    """Source returns -1 in the then-arm, binary returns 0. z3 must
    refute, because for any x < 0 the two disagree."""
    src = Function(
        name="sign_only",
        params=(Param(name="x", type=I32Type()),),
        return_type=I32Type(),
        body=Block(stmts=(If(
            cond=BinOp(op="slt", lhs=ParamRef(name="x"),
                       rhs=IntLit(type=I32Type(), value=0)),
            then_body=Block(stmts=(ReturnExpr(value=IntLit(
                type=I32Type(), value=-1,
            )),)),
            else_body=Block(stmts=(ReturnExpr(value=IntLit(
                type=I32Type(), value=1,
            )),)),
        ),)),
    )

    # Binary has 0 in the then-arm, not -1.
    cond_var = BinVarnode(space="unique", offset=0x100, size=1)
    entry = _bb_with_succ(
        BinPCodeOp(opcode="INT_SLESS",
                   inputs=(_rdi(), _const(0)),
                   output=cond_var, source_address=0x401000),
        BinPCodeOp(opcode="CBRANCH",
                   inputs=(_const(0x401020, size=8), cond_var),
                   output=None, source_address=0x401004),
        addr=0x401000, bb_id="@bb_entry",
    )
    then_bb = _bb_with_succ(
        BinPCodeOp(opcode="COPY", inputs=(_const(0),),  # WRONG — should be -1
                   output=_rax(), source_address=0x401020),
        BinPCodeOp(opcode="RETURN", inputs=(), output=None,
                   source_address=0x401024),
        addr=0x401020, bb_id="@bb_then",
    )
    else_bb = _bb_with_succ(
        BinPCodeOp(opcode="COPY", inputs=(_const(1),),
                   output=_rax(), source_address=0x401010),
        BinPCodeOp(opcode="RETURN", inputs=(), output=None,
                   source_address=0x401014),
        addr=0x401010, bb_id="@bb_else",
    )
    entry = entry.model_copy(update={"successors": (
        BinBlockEdge(successor_id="@bb_then", edge_kind="true"),
        BinBlockEdge(successor_id="@bb_else", edge_kind="false"),
    )})
    bin_fn = _bin_fn("sign_only", 1, entry, then_bb, else_bb)

    prog = _program(src, bin_fn)
    [r] = prove_all_bin_relational(prog, proofs_dir=tmp_path)
    assert r.status == "refuted", r.detail


def test_loop_back_edge_yields_unknown(tmp_path):
    """Two blocks looping back at each other — the path-walker
    detects the back-edge and bails with status='unknown'."""
    src = Function(
        name="ident",
        params=(Param(name="x", type=I32Type()),),
        return_type=I32Type(),
        body=_src_return_expr(ParamRef(name="x")),
    )
    # entry → loop → entry → ... (no RETURN reachable).
    loop_a = _bb_with_succ(
        BinPCodeOp(opcode="COPY", inputs=(_rdi(),),
                   output=_rax(), source_address=0x401000),
        BinPCodeOp(opcode="BRANCH",
                   inputs=(_const(0x401010, size=8),),
                   output=None, source_address=0x401004),
        addr=0x401000, bb_id="@bb_a",
    )
    loop_b = _bb_with_succ(
        BinPCodeOp(opcode="BRANCH",
                   inputs=(_const(0x401000, size=8),),
                   output=None, source_address=0x401010),
        addr=0x401010, bb_id="@bb_b",
    )
    loop_a = loop_a.model_copy(update={"successors": (
        BinBlockEdge(successor_id="@bb_b", edge_kind="unconditional"),
    )})
    loop_b = loop_b.model_copy(update={"successors": (
        BinBlockEdge(successor_id="@bb_a", edge_kind="unconditional"),
    )})
    bin_fn = _bin_fn("ident", 1, loop_a, loop_b)

    prog = _program(src, bin_fn)
    [r] = prove_all_bin_relational(prog, proofs_dir=tmp_path)
    assert r.status == "unknown"
    assert "loop" in r.detail.lower()
