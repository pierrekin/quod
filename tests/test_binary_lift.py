"""`lift_v2.signature_binding` — pure-ABI varnode↔param mapping.

No Ghidra dependency: every test builds a tiny synthetic program with
hand-authored Function + BinFunction nodes, seeds a BinaryProvenance
equivalence, runs the lifter, and asserts on the binding it produces.
"""
from __future__ import annotations

import pytest

from quod.model import (
    BinaryProvenance,
    BinFunction,
    BinFunctionParam,
    BinSrcSignatureBinding,
    BinUnit,
    BinVarnode,
    Block,
    Equivalence,
    F32Type,
    Function,
    I32Type,
    I64Type,
    IntLit,
    Param,
    Program,
    ReturnExpr,
    VoidType,
)
from quod.predicate.binary_lift import derive_signature_bindings


def _src_fn(
    name: str, *, params: tuple[Param, ...], return_type=I32Type()
) -> Function:
    return Function(
        name=name,
        params=params,
        return_type=return_type,
        body=Block(stmts=(ReturnExpr(value=IntLit(type=I32Type(), value=0)),)),
    )


def _bin_fn(name: str, *, n_params: int) -> BinFunction:
    return BinFunction(
        address=0x401000,
        mangled_name=name,
        demangled_name=name,
        return_type_name="int",
        params=tuple(
            BinFunctionParam(name=f"param_{i+1}", type_name="int")
            for i in range(n_params)
        ),
        calling_convention="unknown",
    )


def _program_with_pair(src: Function, bin_fn: BinFunction) -> Program:
    bu = BinUnit(
        path="/tmp/x.so",
        sha256="0" * 64,
        arch="x86_64",
        file_format="elf",
        functions=(bin_fn,),
    )
    eq = Equivalence(
        a_node_id=src.id,
        b_node_id=bin_fn.id,
        justification=BinaryProvenance(
            binary_path="/tmp/x.so",
            binary_sha256="0" * 64,
            binary_symbol=bin_fn.mangled_name,
            source_evidence="symtab",
        ),
    )
    return Program(
        functions=(src,),
        binary_units=(bu,),
        equivalences=(eq,),
    )


def test_single_int_param_maps_to_RDI():
    src = _src_fn("ident", params=(Param(name="x", type=I32Type()),))
    bin_fn = _bin_fn("ident", n_params=1)
    prog = _program_with_pair(src, bin_fn)

    bindings = derive_signature_bindings(prog)

    assert len(bindings) == 1
    sb = bindings[0]
    assert sb.bin_fn_id == bin_fn.id
    assert sb.src_fn_id == src.id
    assert sb.abi == "x86_64-sysv"
    assert len(sb.param_bindings) == 1
    pb = sb.param_bindings[0]
    assert pb.param_name == "x"
    # RDI = register-space offset 0x38, size 4 (int = 4 bytes)
    assert pb.varnode == BinVarnode(space="register", offset=0x38, size=4)
    # RAX = register-space offset 0x00, size 4 for `int` return
    assert sb.return_binding == BinVarnode(space="register", offset=0x00, size=4)


def test_two_int_params_map_to_RDI_then_RSI():
    src = _src_fn(
        "add",
        params=(
            Param(name="a", type=I32Type()),
            Param(name="b", type=I32Type()),
        ),
    )
    bin_fn = _bin_fn("add", n_params=2)
    prog = _program_with_pair(src, bin_fn)

    bindings = derive_signature_bindings(prog)

    assert len(bindings) == 1
    pbs = bindings[0].param_bindings
    assert pbs[0].param_name == "a"
    assert pbs[0].varnode.offset == 0x38  # RDI
    assert pbs[1].param_name == "b"
    assert pbs[1].varnode.offset == 0x30  # RSI


def test_int64_param_uses_8_byte_varnode_view():
    src = _src_fn("f", params=(Param(name="x", type=I64Type()),), return_type=I64Type())
    bin_fn = _bin_fn("f", n_params=1)
    prog = _program_with_pair(src, bin_fn)

    bindings = derive_signature_bindings(prog)

    assert len(bindings) == 1
    sb = bindings[0]
    assert sb.param_bindings[0].varnode.size == 8
    assert sb.return_binding.size == 8


def test_void_return_is_rejected_in_v0():
    src = _src_fn("f", params=(Param(name="x", type=I32Type()),), return_type=VoidType())
    bin_fn = _bin_fn("f", n_params=1)
    prog = _program_with_pair(src, bin_fn)

    assert derive_signature_bindings(prog) == ()


def test_float_param_is_rejected_in_v0():
    src = _src_fn("f", params=(Param(name="x", type=F32Type()),))
    bin_fn = _bin_fn("f", n_params=1)
    prog = _program_with_pair(src, bin_fn)

    assert derive_signature_bindings(prog) == ()


def test_seven_int_params_overflows_register_pool():
    # SysV passes the first 6 in regs; a 7th would spill to stack.
    src = _src_fn(
        "f",
        params=tuple(Param(name=f"p{i}", type=I32Type()) for i in range(7)),
    )
    bin_fn = _bin_fn("f", n_params=7)
    prog = _program_with_pair(src, bin_fn)

    assert derive_signature_bindings(prog) == ()


def test_param_count_mismatch_is_rejected():
    src = _src_fn("f", params=(Param(name="x", type=I32Type()),))
    bin_fn = _bin_fn("f", n_params=2)  # ghidra recovered 2, source says 1
    prog = _program_with_pair(src, bin_fn)

    assert derive_signature_bindings(prog) == ()


def test_no_pairing_yields_no_bindings():
    src = _src_fn("f", params=(Param(name="x", type=I32Type()),))
    bin_fn = _bin_fn("f", n_params=1)
    bu = BinUnit(
        path="/tmp/x.so",
        sha256="0" * 64,
        arch="x86_64",
        file_format="elf",
        functions=(bin_fn,),
    )
    prog = Program(functions=(src,), binary_units=(bu,))  # no equivalence

    assert derive_signature_bindings(prog) == ()


def test_round_trips_through_save_load(tmp_path):
    """A signature binding survives JSON I/O byte-for-byte."""
    from quod.model import load_program, save_program

    src = _src_fn("ident", params=(Param(name="x", type=I32Type()),))
    bin_fn = _bin_fn("ident", n_params=1)
    prog = _program_with_pair(src, bin_fn)
    bindings = derive_signature_bindings(prog)
    prog2 = prog.model_copy(update={"signature_bindings": bindings})

    out = tmp_path / "p.json"
    save_program(prog2, out)
    loaded = load_program(out)
    assert loaded == prog2
