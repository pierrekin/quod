"""Build pipeline driver: wraps `lower()` with the validate/monomorphize/
elaborate pre-passes, the LLVM verify/optimize/codegen passes, and the
clang link step. `compile_program` is the build entry-point used by the
CLI and tests; `prepare_program` is the validate-only gate used by `quod
check`.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from llvmlite import binding as llvm
from llvmlite import ir

from quod.analysis import derive_lattice_claims, elaborate
from quod.lower.desugar import _function_uses_with_arena
from quod.lower.runtime_decls import _ensure_initialized
from quod.model import Block, Program
from quod.resolve import resolve_imports
from quod.runtime import build_runtime_archive
from quod.validate import validate_or_raise


def parse_and_verify(module: ir.Module):
    parsed = llvm.parse_assembly(str(module))
    parsed.verify()
    return parsed


def optimize_module(parsed_module, target_machine, *, speed_level: int) -> None:
    pto = llvm.PipelineTuningOptions(speed_level=speed_level)
    pb = llvm.PassBuilder(target_machine, pto)
    mpm = pb.getModulePassManager()
    mpm.run(parsed_module, pb)


def make_target_machine(target: str | None = None):
    triple = target or llvm.get_default_triple()
    _ensure_initialized(cross=triple != llvm.get_default_triple())
    return llvm.Target.from_triple(triple).create_target_machine(
        reloc="pic", codemodel="default",
    )


@dataclass(frozen=True)
class BinResult:
    name: str
    entry: str
    ir_unopt: Path
    ir_opt: Path | None
    object_path: Path
    binary: Path | None


@dataclass(frozen=True)
class CompileResult:
    bins: tuple[BinResult, ...]


def has_function(program: Program, name: str) -> bool:
    return any(fn.name == name for fn in program.functions)


def _refuse_non_core_layer(program: Program) -> None:
    """Refuse layer-B (`c.*` extension) constructs at the build entry
    point. `quod.lower` and its supporting validators (`quod.validate`,
    `quod.monomorphize`, …) operate on layer C — pure core quod. The
    c-family lowering pass strips wrappers and rewrites extension
    statements before this point; if any survive, surface a clear error
    naming the offending kind so the fix is obvious.
    """
    for fn in program.functions:
        if not isinstance(fn.body, Block):
            raise ValueError(
                f"function {fn.name!r}: body is wrapped in "
                f"{type(fn.body).__name__!r} — layer C must be pure core. "
                f"Run the c-family lowering pass (lower/c_family.py) "
                f"before building."
            )
        for stmt in fn.body.stmts:
            kind = getattr(stmt, "kind", None)
            if kind and str(kind).startswith("c."):
                raise ValueError(
                    f"function {fn.name!r}: statement {kind!r} is a "
                    f"`c.*` family extension — layer C must be pure core. "
                    f"Run the c-family lowering pass (lower/c_family.py) "
                    f"before building."
                )


def prepare_program(
    program: Program,
    *,
    disabled_tiers: frozenset[str] = frozenset(),
) -> Program:
    """Run the standard pre-lowering pipeline and return a program ready
    to lower. Used by both `compile_program` (build) and `quod check`
    (validate without emitting artifacts) so the two share one
    "is this a valid program?" gate.

    Pipeline:
        1. `--no-alloc` short-circuit on `with_arena`. Build-config
           friendlier than letting the import resolver fail.
        2. resolve_imports — pull in stdlib + user imports.
        3. validate (pre-mono) — error locations point at user-written
           names, not mangled post-mono identifiers.
        4. monomorphize — specialize generics to concrete types.
        5. validate (post-mono) — canonical correctness gate;
           everything is concrete and every reference should resolve.
        6. elaborate — derive lattice claims and merge into the program.

    After this, the program is ready for `lower()` and downstream
    LLVM-only steps (codegen, verify, optimize, link). If `prepare`
    succeeds, the program is semantically valid; build failures from
    here on are LLVM/linker concerns, not language correctness.
    """
    _refuse_non_core_layer(program)

    if "alloc" in disabled_tiers:
        for fn in program.functions:
            if _function_uses_with_arena(fn):
                raise ValueError(
                    f"function {fn.name!r} uses `with_arena`, which requires "
                    f"the 'alloc' tier — disabled by --no-alloc"
                )

    program = resolve_imports(program, disabled_tiers=disabled_tiers)
    validate_or_raise(program)

    from quod.monomorphize import monomorphize as _monomorphize
    program = _monomorphize(program)
    validate_or_raise(program)

    derived = derive_lattice_claims(program)
    program = elaborate(program, derived)
    return program


def compile_program(
    program: Program,
    *,
    build_dir: Path,
    bins: tuple[tuple[str, str], ...] = (("main", "main"),),
    profile: int = 2,
    link: bool = True,
    libraries: tuple[str, ...] = (),
    target: str | None = None,
    overrides: dict[str, str] | None = None,
    disabled_tiers: frozenset[str] = frozenset(),
) -> CompileResult:
    """Compile `program` into one binary per bin.

    `bins` is a tuple of (name, entry) pairs: `name` is the output binary
    filename, `entry` is the program function used as the entry point. The
    default ((`"main"`, `"main"`),) preserves pre-config behavior.

    `disabled_tiers` is forwarded to import resolution; it short-circuits
    builds that try to import from a tier the profile excludes (e.g.
    `--no-std` → frozenset({"std"})).
    """
    from quod.lower import lower  # break the __init__-vs-pipeline import cycle
    if not 0 <= profile <= 3:
        raise ValueError(f"profile must be 0..3, got {profile}")
    build_dir.mkdir(parents=True, exist_ok=True)

    program = prepare_program(program, disabled_tiers=disabled_tiers)

    target_machine = make_target_machine(target=target)
    results: list[BinResult] = []

    for bin_name, entry in bins:
        if not has_function(program, entry):
            raise ValueError(f"bin {bin_name!r}: entry function {entry!r} not found")

        module = lower(program, target=target, overrides=overrides, entry=entry)
        ir_unopt = build_dir / f"{bin_name}.unopt.ll"
        ir_unopt.write_text(str(module))

        parsed = parse_and_verify(module)
        ir_opt: Path | None = None
        if profile > 0:
            optimize_module(parsed, target_machine, speed_level=profile)
            ir_opt = build_dir / f"{bin_name}.opt.ll"
            ir_opt.write_text(str(parsed))

        object_path = build_dir / f"{bin_name}.o"
        object_path.write_bytes(target_machine.emit_object(parsed))

        binary: Path | None = None
        if link:
            binary = build_dir / bin_name
            # Build the optional runtime archive into the same build_dir,
            # matching the user's target. Empty by default (the stdlib
            # arena lives in quod now); when a user adds a runtime/*.c,
            # archive linking is by-reference so unused symbols stay
            # stripped.
            runtime_archive = build_runtime_archive(build_dir, target=target)
            cmd = ["clang"]
            if target:
                cmd += ["-target", target]
            cmd += [str(object_path)]
            if runtime_archive is not None:
                cmd += [str(runtime_archive)]
            cmd += ["-o", str(binary)]
            cmd += [f"-l{lib}" for lib in libraries]
            subprocess.run(cmd, check=True)

        results.append(BinResult(
            name=bin_name, entry=entry,
            ir_unopt=ir_unopt, ir_opt=ir_opt,
            object_path=object_path, binary=binary,
        ))

    return CompileResult(bins=tuple(results))
