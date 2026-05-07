"""Lowering: Program -> LLVM IR -> object/binary, plus the optimization pass.

Two-pass over functions: first declare every user function (so user-level
calls in expressions resolve regardless of definition order), then lower
each function body.

Locals are alloca'd at the function's entry block (the canonical mem2reg
shape). Loops, ExprStmt, short-circuit booleans, and fall-through Ifs
each get their own basic-block layout.
"""

from __future__ import annotations

from llvmlite import binding as llvm
from llvmlite import ir

from quod.lower.desugar import _desugar_with_arena
from quod.lower.function import (
    _declare_extern,
    _declare_function,
    _lower_function_body,
)
from quod.lower.main_wrapper import _emit_main_wrapper
from quod.lower.runtime_decls import _ensure_initialized
from quod.lower.types import (
    I8,
    I64,
    _type_to_llvm,
    _variant_struct_layout,
)
from quod.model import (
    EnumDef,
    ExternFunction,
    Program,
    StructDef,
)


def lower(
    program: Program, *,
    target: str | None = None,
    overrides: dict[str, str] | None = None,
    entry: str | None = None,
) -> ir.Module:
    """Lower `program` to LLVM IR.

    `entry` names the function that should serve as the binary's entry point.
    If `entry` is "main" or None and the program has a function called "main",
    no wrapping happens. Otherwise a synthetic `main` is appended that calls
    `entry` and returns its result.
    """
    program = _desugar_with_arena(program)

    # Fresh context per module: llvmlite's default `ir.Module()` shares
    # one process-global LLVMContext, so identified types interned by
    # name (struct/enum bodies) leak between independent `lower()`
    # calls in the same Python process. That's invisible in production
    # (one program per CLI invocation) but breaks tests under
    # pytest-xdist when two cases name a type the same and give it a
    # different layout — the second build sees the first's body and
    # emits insertvalue at the wrong field type. Isolating the context
    # makes every lower() build a clean type universe.
    module = ir.Module(name="quod", context=ir.Context())
    module.triple = target or llvm.get_default_triple()
    # CRITICAL: set the data layout from the target. With an empty
    # datalayout string, LLVM falls back to its "neutral" defaults
    # which use 4-byte alignment for i64 — breaking any aggregate
    # layout that assumes natural 8-byte alignment. The optimizer
    # silently miscompiles bitcast-then-load through structs whose
    # alignment differs from the neutral default.
    _ensure_initialized(cross=module.triple != llvm.get_default_triple())
    _tm = llvm.Target.from_triple(module.triple).create_target_machine()
    module.data_layout = str(_tm.target_data)
    overrides = overrides or {}

    # Pass 0: register named struct AND enum types. Four phases:
    # (a) allocate identified types for every struct,
    # (b) allocate identified types for every enum,
    # (c) set struct bodies (may reference any allocated type),
    # (d) set enum bodies.
    # Allocating both type sets up-front lets a struct reference an
    # enum-typed field (or vice versa) without forcing a topological
    # ordering. Cycles in struct nesting are already rejected by the
    # model validator, so body-setting terminates.
    struct_defs: dict[str, StructDef] = {sd.name: sd for sd in program.structs}
    enum_defs: dict[str, EnumDef] = {ed.name: ed for ed in program.enums}
    struct_defs_for_layout: dict[str, StructDef] = {sd.name: sd for sd in program.structs}

    struct_tys: dict[str, ir.IdentifiedStructType] = {}
    for sd in program.structs:
        struct_tys[sd.name] = module.context.get_identified_type(sd.name)
    enum_tys: dict[str, ir.IdentifiedStructType] = {}
    for ed in program.enums:
        enum_tys[ed.name] = module.context.get_identified_type(ed.name)

    for sd in program.structs:
        ty = struct_tys[sd.name]
        body = [_type_to_llvm(f.type, struct_tys, enum_tys) for f in sd.fields]
        ty.set_body(*body)

    # Enums: each lowers to an identified `{i8 tag, [N x i64] payload}`
    # struct. N is the smallest count such that `N * 8` covers the
    # largest variant's payload size. Using i64 for the payload array
    # gives us 8-byte alignment, which is the maximum alignment any
    # variant field can require in quod's current type system (i64,
    # i8*, and any struct/enum built from them are at most 8-aligned).
    # EnumInit and Match access fields by bitcasting the payload bytes
    # to a per-variant literal LLVM struct type — variants can carry
    # arbitrary types (other structs, even other enums).
    for ed in program.enums:
        ty = enum_tys[ed.name]
        # Largest variant payload size in bytes. ceil-divide by 8.
        max_payload = max(
            (_variant_struct_layout(v, struct_defs_for_layout, enum_defs)[0]
             for v in ed.variants),
            default=0,
        )
        n_slots = (max_payload + 7) // 8 or 1
        ty.set_body(I8, ir.ArrayType(I64, n_slots))

    constants: dict[str, ir.GlobalVariable] = {}
    for c in program.constants:
        data = bytearray(c.value.encode("utf-8") + b"\0")
        ty = ir.ArrayType(I8, len(data))
        gv = ir.GlobalVariable(module, ty, name=c.name)
        gv.linkage = "private"
        gv.global_constant = True
        gv.initializer = ir.Constant(ty, data)
        constants[c.name] = gv

    # Pass 1: declare every user function and every extern so calls can
    # resolve regardless of order or definedness.
    for fn in program.functions:
        _declare_function(module, fn, struct_tys, enum_tys)
    for ext in program.externs:
        _declare_extern(module, ext, struct_tys, enum_tys)

    extern_sigs: dict[str, ExternFunction] = {ext.name: ext for ext in program.externs}

    # Pass 2: lower bodies of user functions only (externs have no body here).
    for fn in program.functions:
        _lower_function_body(
            module, fn, constants=constants, overrides=overrides,
            extern_sigs=extern_sigs,
            struct_defs=struct_defs, struct_tys=struct_tys,
            enum_defs=enum_defs, enum_tys=enum_tys,
        )

    if entry is not None:
        _emit_main_wrapper(module, program, entry)

    return module


# Pipeline imports come AFTER lower() is defined so pipeline.py can
# reach `quod.lower.lower` via the partially-initialized package.
from quod.lower.pipeline import (  # noqa: E402
    BinResult,
    CompileResult,
    compile_program,
    has_function,
    make_target_machine,
    optimize_module,
    parse_and_verify,
    prepare_program,
)


__all__ = [
    "BinResult",
    "CompileResult",
    "compile_program",
    "has_function",
    "lower",
    "make_target_machine",
    "optimize_module",
    "parse_and_verify",
    "prepare_program",
]
