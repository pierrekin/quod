"""Synthesised C-main wrapper.

When the program's entry function isn't a nullary `main`, append an
`i32 main(...)` that bounds-checks argc, parses each argv slot via
`atoll`, trunc/sext's to the entry function's param types, forwards,
and returns the entry's result widened/narrowed to i32.
"""

from __future__ import annotations

from llvmlite import ir

from quod.lower.runtime_decls import _get_or_declare_abort, _get_or_declare_atoll
from quod.lower.types import I8, I32, I64, _type_to_llvm
from quod.model import (
    I1Type,
    I8Type,
    I16Type,
    I32Type,
    I64Type,
    Program,
)


def _emit_main_wrapper(module: ir.Module, program: Program, entry: str) -> None:
    """Append a synthesized `main` calling the user's entry function.

    Three cases:
      - entry is 'main' and nullary: nothing to do — user's main IS the C main.
      - entry is nullary (any name): emit `i32 main() { return ext(entry()); }`,
        where `ext` is sext/trunc as needed to match i32.
      - entry has params: emit `i32 main(i32 argc, i8** argv)` that
        bounds-checks argc, calls atoll on each argv slot, trunc/sext's to
        each param's type, and forwards. The result is similarly converted
        to i32. Auto-declares atoll / abort if absent.
    """
    fn = next((f for f in program.functions if f.name == entry), None)
    if fn is None:
        raise ValueError(f"entry function {entry!r} not found in program")

    if entry == "main" and not fn.params:
        return  # user's nullary main is already the C main

    if entry == "main" and fn.params:
        raise ValueError(
            "entry function 'main' cannot have parameters; the synthesized "
            "argv wrapper would collide. Rename your entry (e.g. to 'app' "
            "or 'run') and quod will wrap it."
        )

    for p in fn.params:
        if not isinstance(p.type, (I1Type, I8Type, I16Type, I32Type, I64Type)):
            raise ValueError(
                f"entry function {entry!r} param {p.name!r} has non-int type "
                f"{p.type!r}; the argv wrapper only knows how to parse integers. "
                f"Use a nullary entry that constructs richer args internally."
            )
    if not isinstance(fn.return_type, (I1Type, I8Type, I16Type, I32Type, I64Type)):
        raise ValueError(
            f"entry function {entry!r} returns non-int type {fn.return_type!r}; "
            f"main must return an integer exit code"
        )

    if any(f.name == "main" for f in program.functions):
        raise ValueError(
            f"cannot use {entry!r} as entry: program already defines a function "
            f"named 'main'; remove one or rename the conflict"
        )

    target_fn = module.globals[entry]

    if not fn.params:
        # Nullary entry: simple forward.
        main_fn = ir.Function(module, ir.FunctionType(I32, []), name="main")
        bb = main_fn.append_basic_block("entry")
        builder = ir.IRBuilder(bb)
        result = builder.call(target_fn, [])
        builder.ret(_resize_int(builder, result, I32))
        return

    atoll = _get_or_declare_atoll(module)
    abort = _get_or_declare_abort(module)
    n = len(fn.params)

    main_ty = ir.FunctionType(I32, [I32, I8.as_pointer().as_pointer()])
    main_fn = ir.Function(module, main_ty, name="main")
    main_fn.args[0].name = "argc"
    main_fn.args[1].name = "argv"
    argc, argv = main_fn.args

    entry_bb = main_fn.append_basic_block("entry")
    parse_bb = main_fn.append_basic_block("parse")
    fail_bb = main_fn.append_basic_block("fail")
    builder = ir.IRBuilder(entry_bb)

    required = ir.Constant(I32, n + 1)
    too_few = builder.icmp_signed("<", argc, required)
    builder.cbranch(too_few, fail_bb, parse_bb)

    builder.position_at_end(fail_bb)
    builder.call(abort, [])
    builder.unreachable()

    builder.position_at_end(parse_bb)
    parsed_args = []
    for i, p in enumerate(fn.params):
        # argv[i+1]: GEP on i8** then load to get i8*
        idx = ir.Constant(I64, i + 1)
        arg_slot = builder.gep(argv, [idx])         # i8**
        arg_ptr = builder.load(arg_slot)            # i8*
        parsed_i64 = builder.call(atoll, [arg_ptr]) # i64
        target_ty = _type_to_llvm(p.type)
        parsed_args.append(_resize_int(builder, parsed_i64, target_ty))

    result = builder.call(target_fn, parsed_args)
    builder.ret(_resize_int(builder, result, I32))


def _resize_int(builder: ir.IRBuilder, val: ir.Value, target_ty: ir.IntType) -> ir.Value:
    """Sign-extend, truncate, or no-op a value to match `target_ty`."""
    src_w = val.type.width
    dst_w = target_ty.width
    if src_w == dst_w:
        return val
    if src_w < dst_w:
        return builder.sext(val, target_ty)
    return builder.trunc(val, target_ty)
