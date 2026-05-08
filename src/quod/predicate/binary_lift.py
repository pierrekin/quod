"""Decompile-lift v0 — signature-level binding between binary and source.

The smallest useful slice of "lift Ghidra's recovered binary into the
source-language graph": pure ABI-driven, no decompile-text re-parse,
no body inspection, no type recovery. For each `BinaryProvenance`
equivalence the seeder produced, attach an explicit varnode↔param
mapping so the relational SMT prover (`z3.bin_relational`) has a
structural alignment to encode against.

`derive_signature_bindings(program)` returns one `BinSrcSignatureBinding`
per (bin.fn, src.fn) pair where:
  - the source function's parameters and return are all int-typed
    (the prover's v0 universe);
  - the binary function is x86-64 SysV ABI;
  - the parameter count matches, none of the params spill past the
    first 6 int-class registers (RDI, RSI, RDX, RCX, R8, R9);

Pairs that don't fit the v0 universe are silently skipped — the lift
never speculates: a binding that can't be derived from the ABI rules
alone simply isn't emitted. v1 will widen to floats (XMM regs), pointer
params, struct-by-value (SysV's "first 8 bytes per class" decomposition),
and stack-passed params after the 7th int.

The lift is **structural only**: it asserts "if this binary obeys the
SysV ABI for this signature, then varnode V holds parameter P." It does
not claim the binary is correct, that the body matches the source, or
that the types are right. Those obligations belong to the relational
prover.
"""

from __future__ import annotations

from dataclasses import dataclass

from quod.model import (
    BinaryProvenance,
    BinFunction,
    BinSrcParamBinding,
    BinSrcSignatureBinding,
    BinVarnode,
    Function,
    I1Type,
    I8Type,
    I16Type,
    I32Type,
    I64Type,
    IntType,
    IsizeType,
    Program,
    U8Type,
    U16Type,
    U32Type,
    U64Type,
    UsizeType,
)


_INT_TYPE_CLASSES = (
    I1Type, I8Type, I16Type, I32Type, I64Type,
    U8Type, U16Type, U32Type, U64Type, IsizeType, UsizeType,
)


# x86-64 SysV ABI: integer-class arguments in this register sequence.
# Ghidra's register-space offsets follow the layout in the processor
# spec — for x86-64, that's the SLEIGH `register` space with these
# offsets:
#   RDI=0x38, RSI=0x30, RDX=0x10, RCX=0x08, R8=0x80, R9=0x88
# (These are *byte* offsets into the register address space, not the
# x86-64 register encoding. The varnode size determines whether we get
# a 32-bit or 64-bit view of the same physical register; the decompiler
# typically uses a 4-byte view for `int` params and 8-byte for pointers
# or `long`.)
_X86_64_SYSV_INT_REGS: tuple[tuple[str, int], ...] = (
    ("RDI", 0x38),
    ("RSI", 0x30),
    ("RDX", 0x10),
    ("RCX", 0x08),
    ("R8", 0x80),
    ("R9", 0x88),
)

# Return value: RAX in x86-64 SysV (same physical reg as for the
# Ghidra register-space offset 0).
_X86_64_SYSV_RETURN_OFFSET = 0x00

_X86_64_SYSV = "x86_64-sysv"


# Maximum int-class params passed in registers under SysV; after this,
# they spill onto the stack — supportable but deferred to v1 of the
# lifter so the prover can stay register-only for now.
_MAX_REG_INT_PARAMS = len(_X86_64_SYSV_INT_REGS)


@dataclass(frozen=True)
class _Pairing:
    bin_fn: BinFunction
    src_fn: Function


def derive_signature_bindings(
    program: Program,
) -> tuple[BinSrcSignatureBinding, ...]:
    """Return one `BinSrcSignatureBinding` per bin↔src pair where the
    ABI mapping is unambiguous under the v0 rules.

    Idempotent: existing `program.signature_bindings` are not consulted;
    the caller is expected to merge results into the program (the merge
    logic dedups by (bin_fn_id, src_fn_id, abi)).
    """
    pairings = _collect_pairings(program)
    if not pairings:
        return ()

    out: list[BinSrcSignatureBinding] = []
    for p in pairings:
        binding = _signature_binding_for_pair(p)
        if binding is not None:
            out.append(binding)
    return tuple(out)


def _collect_pairings(program: Program) -> list[_Pairing]:
    """Walk `BinaryProvenance` equivalences and produce `(bin.fn, src.fn)`
    pairs where the source endpoint is a Layer-C `Function`.

    Mirrors `binary_hints._pair_function_to_bin` but we only need
    the Layer-C side (ParamRef.name resolves against `Function.params`,
    and the relational prover's source encoder consumes Layer-C bodies)."""
    bin_fns = {fn.id: fn for u in program.binary_units for fn in u.functions}
    if not bin_fns:
        return []

    cfn_to_name: dict[str, str] = {}
    for unit in program.source_units:
        for cfn in unit.functions:
            cfn_to_name[cfn.id] = cfn.name

    fn_by_id: dict[str, Function] = {fn.id: fn for fn in program.functions}
    fn_by_name: dict[str, Function] = {fn.name: fn for fn in program.functions}

    seen: set[tuple[str, str]] = set()
    pairs: list[_Pairing] = []
    for eq in program.equivalences:
        if not isinstance(eq.justification, BinaryProvenance):
            continue
        if eq.b_node_id in bin_fns:
            bin_id, src_id = eq.b_node_id, eq.a_node_id
        elif eq.a_node_id in bin_fns:
            bin_id, src_id = eq.a_node_id, eq.b_node_id
        else:
            continue
        bin_fn = bin_fns[bin_id]

        fn = fn_by_id.get(src_id)
        if fn is None and src_id in cfn_to_name:
            fn = fn_by_name.get(cfn_to_name[src_id])
        if fn is None:
            continue

        key = (fn.id, bin_fn.id)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(_Pairing(bin_fn=bin_fn, src_fn=fn))
    return pairs


def _signature_binding_for_pair(p: _Pairing) -> BinSrcSignatureBinding | None:
    """Build a binding for one pair, or None if it falls outside v0's rules.

    Rejection conditions (each surfaces nothing rather than raising):
      - Any source param isn't an int type.
      - More than 6 int params (would spill onto the stack).
      - Source param count doesn't match the binary's recovered param
        count. Mismatch means Ghidra recovered a different signature
        than the source declares — we don't speculate which is right.
      - Source return type isn't an int.
    """
    int_params: list = []
    for sp in p.src_fn.params:
        if not isinstance(sp.type, _INT_TYPE_CLASSES):
            return None
        int_params.append(sp)
    if len(int_params) > _MAX_REG_INT_PARAMS:
        return None
    if len(int_params) != len(p.bin_fn.params):
        return None

    rt = p.src_fn.return_type
    # `Function.return_type` is `ReturnType = Type | VoidType`; we want
    # only the int subset for v0. Use isinstance so we don't have to
    # import the union sentinel.
    if not isinstance(rt, _INT_TYPE_CLASSES):
        return None

    param_bindings: list[BinSrcParamBinding] = []
    for i, sp in enumerate(int_params):
        _, reg_offset = _X86_64_SYSV_INT_REGS[i]
        param_bindings.append(BinSrcParamBinding(
            varnode=BinVarnode(
                space="register",
                offset=reg_offset,
                size=_int_type_bytes(sp.type),
            ),
            param_name=sp.name,
        ))

    return BinSrcSignatureBinding(
        bin_fn_id=p.bin_fn.id,
        src_fn_id=p.src_fn.id,
        abi=_X86_64_SYSV,
        param_bindings=tuple(param_bindings),
        return_binding=BinVarnode(
            space="register",
            offset=_X86_64_SYSV_RETURN_OFFSET,
            size=_int_type_bytes(rt),
        ),
    )


def _int_type_bytes(t: IntType) -> int:
    """Width in bytes of an int type. Used to size the varnode the ABI
    rule says holds the value: the same physical register can hold a
    32-bit `int` (4 bytes) or a 64-bit `long` (8 bytes), and the
    varnode's `size` records which view the binary actually uses."""
    match t:
        case I1Type():
            return 1
        case I8Type() | U8Type():
            return 1
        case I16Type() | U16Type():
            return 2
        case I32Type() | U32Type():
            return 4
        case I64Type() | U64Type() | IsizeType() | UsizeType():
            return 8
    raise ValueError(f"not an int type: {t!r}")
