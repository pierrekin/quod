"""Import resolution — the language's module system.

Programs declare what they need via `imports`; this module resolves
those imports by loading modules, applying `wire` bindings (including
transitive forwarding), and merging the result into a flat Program.

The merge is first-wins by name: if a user-declared struct/extern/function
shadows an imported one (same `name`), the user's wins. Signature mismatch
is left for the LLVM verifier — we don't try to type-check across the
boundary at the model layer.

After resolution `program.imports` is cleared, so a resolved program is
indistinguishable from one the user wrote flat. Don't `save_program` on
a resolved program — it would inline imported modules into the user's source.

Module loading: by default, modules are looked up as `<name>.json` under
the shipped-modules directory (`src/quod/stdlib/`). The loader is a plain
function — tests or alternative toolchains can supply their own.
"""

from __future__ import annotations

from pathlib import Path

from quod.model import InputProgram, Program


_MODULE_DIR = Path(__file__).parent / "stdlib"


def module_dir() -> Path:
    """The default module search path. Modules live here as `<name>.json`."""
    return _MODULE_DIR


class ImportError_(Exception):
    """Module name doesn't resolve, or its file is malformed.

    Trailing underscore avoids shadowing the builtin `ImportError`; raising
    that one would lose the quod-specific error context."""


def module_tier(name: str) -> str:
    """Classify a module name by its top-level namespace.

    `core.*`  -> "core"  (pure quod, no runtime deps)
    `alloc.*` -> "alloc" (needs an allocator)
    `std.*`   -> "std"   (needs a hosted OS / libc)
    anything else -> "core" (treated as user-facing / unrestricted)
    """
    head = name.split(".", 1)[0]
    if head in ("core", "alloc", "std"):
        return head
    return "core"


def resolve_imports(program: Program, *, disabled_tiers: frozenset[str] = frozenset()) -> Program:
    """Walk `program.imports` (and any nested imports declared by those
    modules), fold their constants/structs/enums/externs/functions into
    `program`, and clear `program.imports`. First-wins dedupe by name —
    user-declared items always shadow imports.

    For imports with a `wire` clause, the resolver substitutes the
    imported module's wirables to the bound types throughout the
    module's body BEFORE merging it. No transitive forwarding —
    wirables in a stdlib module that's transitively imported (rather
    than directly wired by the consumer) are an error.

    `disabled_tiers` lists tiers that must NOT be resolved (e.g.
    `frozenset({"std"})` for --no-std). If a transitive import lives in
    a disabled tier, raise ImportError_ pointing at the offender."""
    if not program.imports:
        return program

    constants = list(program.constants)
    structs = list(program.structs)
    enums = list(program.enums)
    externs = list(program.externs)
    functions = list(program.functions)
    traits = list(program.traits)
    impls = list(program.impls)
    seen_const = {c.name for c in constants}
    seen_struct = {s.name for s in structs}
    seen_enum = {e.name for e in enums}
    seen_extern = {e.name for e in externs}
    seen_fn = {f.name for f in functions}
    seen_trait = {t.name for t in traits}
    # Impls are deduped by (trait, for_type-shape) — same trait + same
    # concrete type from two different stdlib modules would be a
    # coherence violation, but most commonly the same library transit-
    # ively imported twice is fine.
    seen_impl: set[tuple[str, str]] = {(i.trait, repr(i.for_type)) for i in impls}

    # Queue of Import objects. Each import's `wire` clauses are applied
    # to the imported module when it's loaded — regardless of whether
    # the import is the program's direct import or another module's
    # nested import. (Whoever wrote `import X wire A=Y` chose the wiring
    # in their own source.)
    queue: list = list(program.imports)
    visited: set[str] = set()
    while queue:
        imp = queue.pop(0)
        name = imp.module
        if name in visited:
            continue
        visited.add(name)
        tier = module_tier(name)
        if tier in disabled_tiers:
            raise ImportError_(
                f"import {name!r} requires the {tier!r} tier, which is "
                f"disabled by the build profile (e.g. --no-{tier})"
            )
        mod = _load_module(name)
        # If the module declares wirables, the import must wire all of
        # them. Substitute the wirables throughout the module before
        # merging.
        if mod.wirables:
            mod = _apply_wires(mod, imp)
        elif imp.wire:
            raise ImportError_(
                f"import {name!r} carries `wire` clauses, but the module "
                f"declares no wirables: {[w.name for w in imp.wire]}"
            )
        for nested in mod.imports:
            if nested.module not in visited:
                queue.append(nested)
        for c in mod.constants:
            if c.name not in seen_const:
                constants.append(c)
                seen_const.add(c.name)
        for s in mod.structs:
            if s.name not in seen_struct:
                structs.append(s)
                seen_struct.add(s.name)
        for ed in mod.enums:
            if ed.name not in seen_enum:
                enums.append(ed)
                seen_enum.add(ed.name)
        for e in mod.externs:
            if e.name not in seen_extern:
                externs.append(e)
                seen_extern.add(e.name)
        for f in mod.functions:
            if f.name not in seen_fn:
                functions.append(f)
                seen_fn.add(f.name)
        for t in mod.traits:
            if t.name not in seen_trait:
                traits.append(t)
                seen_trait.add(t.name)
        for i in mod.impls:
            key = (i.trait, repr(i.for_type))
            if key not in seen_impl:
                impls.append(i)
                seen_impl.add(key)

    # Construct rather than model_copy so the Program validator runs on the
    # merged result — catches dangling struct refs that were deferred while
    # imports were unresolved.
    return Program(
        constants=tuple(constants),
        structs=tuple(structs),
        enums=tuple(enums),
        externs=tuple(externs),
        functions=tuple(functions),
        traits=tuple(traits),
        impls=tuple(impls),
        imports=(),
    )


def _apply_wires(mod: InputProgram, imp) -> InputProgram:
    """Bind a module's `wirables` from an `Import.wire` clause and
    substitute throughout the module's body. Returns a new InputProgram
    with `wirables=()` and every TypeParamRef-named-by-a-wirable
    replaced.

    Errors out when:
      - any of `mod.wirables` is unbound by `imp.wire`
      - `imp.wire` names a binding for a wirable that doesn't exist
      - a wire RHS is still a `TypeParamRef` at apply time. By the time
        we get here, the parent (if any) should have substituted the
        forwarding chain to a concrete type. A remaining `TypeParamRef`
        means either (a) the program itself wrote a non-substitutable
        wire (no enclosing wirable to forward from), or (b) a parent
        module's wire forwards a name that isn't one of its wirables.
    """
    from .model import TypeParamRef
    for w in imp.wire:
        if isinstance(w.type, TypeParamRef):
            raise ImportError_(
                f"import {imp.module!r}: wire {w.name!r} binds to "
                f"unresolved TypeParamRef({w.type.name!r}). The wire RHS "
                f"must be a concrete type by the time the import is "
                f"applied — either bind it directly here, or ensure the "
                f"enclosing module declares a wirable named "
                f"{w.type.name!r} that gets substituted at *its* import."
            )
    wire_map = {w.name: w.type for w in imp.wire}
    wirable_names = {w.name for w in mod.wirables}
    missing = wirable_names - set(wire_map)
    extra = set(wire_map) - wirable_names
    if missing:
        raise ImportError_(
            f"import {imp.module!r}: missing `wire` for wirable(s) "
            f"{sorted(missing)}"
        )
    if extra:
        raise ImportError_(
            f"import {imp.module!r}: `wire` for unknown wirable(s) "
            f"{sorted(extra)} (declared wirables: {sorted(wirable_names)})"
        )
    return _substitute_wirables(mod, wire_map)


def mod_name(mod: InputProgram, imp) -> str:
    return imp.module


def _substitute_wirables(mod: InputProgram, wire_map: dict) -> InputProgram:
    """Walk every def in `mod`, substituting TypeParamRefs that name a
    wirable with the bound Type. Generic structs/enums/functions/impls
    that *shadow* a wirable name with their own type-param keep their
    own ref intact (no substitution at the inner scope).
    """
    from .traversal import substitute_in_stmt
    from .model import (
        TypeParamRef, StructType, EnumType, StructField, EnumPayloadField,
        EnumVariant, Param, Function, StructDef, EnumDef, ImplDef,
    )

    def sub_type(t, banned: set[str]):
        if isinstance(t, TypeParamRef) and t.name in wire_map and t.name not in banned:
            return wire_map[t.name]
        if isinstance(t, StructType) and t.type_args:
            return t.model_copy(update={
                "type_args": tuple(sub_type(a, banned) for a in t.type_args),
            })
        if isinstance(t, EnumType) and t.type_args:
            return t.model_copy(update={
                "type_args": tuple(sub_type(a, banned) for a in t.type_args),
            })
        return t

    def sub_in_def_with_params(type_params, fn_under):
        # Inner type_params shadow the wirable.
        own = {tp.name for tp in type_params}
        return fn_under(own)

    new_structs = []
    for sd in mod.structs:
        own = {tp.name for tp in sd.type_params}
        new_fields = tuple(
            StructField(name=f.name, type=sub_type(f.type, own))
            for f in sd.fields
        )
        new_structs.append(sd.model_copy(update={"fields": new_fields}))

    new_enums = []
    for ed in mod.enums:
        own = {tp.name for tp in ed.type_params}
        new_variants = tuple(
            EnumVariant(
                name=v.name,
                fields=tuple(
                    EnumPayloadField(name=f.name, type=sub_type(f.type, own))
                    for f in v.fields
                ),
            )
            for v in ed.variants
        )
        new_enums.append(ed.model_copy(update={"variants": new_variants}))

    new_functions = []
    for fn in mod.functions:
        own = {tp.name for tp in fn.type_params}
        type_fn = lambda t, banned=own: sub_type(t, banned)
        new_params = tuple(
            Param(name=p.name, type=type_fn(p.type)) for p in fn.params
        )
        new_return = type_fn(fn.return_type)
        new_stmts = tuple(substitute_in_stmt(s, type_fn) for s in fn.body.stmts)
        new_body = fn.body.model_copy(update={"stmts": new_stmts})
        new_functions.append(fn.model_copy(update={
            "params":      new_params,
            "return_type": new_return,
            "body":        new_body,
        }))

    new_impls = []
    for impl in mod.impls:
        own = {tp.name for tp in impl.type_params}
        type_fn = lambda t, banned=own: sub_type(t, banned)
        # for_type lives at the impl scope; impl's own type_params shadow.
        new_for_type = type_fn(impl.for_type)
        new_methods = []
        for method in impl.methods:
            method_own = own | {tp.name for tp in method.type_params}
            method_fn = lambda t, banned=method_own: sub_type(t, banned)
            m_params = tuple(
                Param(name=p.name, type=method_fn(p.type)) for p in method.params
            )
            m_return = method_fn(method.return_type)
            m_stmts = tuple(substitute_in_stmt(s, method_fn) for s in method.body.stmts)
            m_body = method.body.model_copy(update={"stmts": m_stmts})
            new_methods.append(method.model_copy(update={
                "params":      m_params,
                "return_type": m_return,
                "body":        m_body,
            }))
        new_impls.append(impl.model_copy(update={
            "for_type": new_for_type,
            "methods":  tuple(new_methods),
        }))

    new_externs = []
    for ext in mod.externs:
        new_externs.append(ext.model_copy(update={
            "param_types": tuple(sub_type(t, set()) for t in ext.param_types),
            "return_type": sub_type(ext.return_type, set()),
        }))

    # Substitute inside the module's own imports — this is what makes
    # transitive forwarding work. If this module declares `wirable A`
    # and one of its imports says `wire X=A` (a TypeParamRef("A")),
    # the substitution replaces it with the concrete type the parent
    # bound A to. When the resolver later processes that nested import,
    # its wire RHS is concrete.
    from .model import Import, WireBinding
    new_imports = []
    for imp in mod.imports:
        if imp.wire:
            new_wire = tuple(
                WireBinding(name=w.name, type=sub_type(w.type, set()))
                for w in imp.wire
            )
            new_imports.append(Import(module=imp.module, wire=new_wire))
        else:
            new_imports.append(imp)

    return mod.model_copy(update={
        "wirables":  (),
        "structs":   tuple(new_structs),
        "enums":     tuple(new_enums),
        "functions": tuple(new_functions),
        "impls":     tuple(new_impls),
        "externs":   tuple(new_externs),
        "imports":   tuple(new_imports),
    })


def _load_module(name: str) -> InputProgram:
    """Load a stdlib module by name (e.g. `"std.str"` -> `stdlib/std.str.json`).

    Modules are validated through `InputProgram` — the same gate user files
    pass. They may declare their own `imports`, which the caller queues
    for the recursive resolution."""
    path = _MODULE_DIR / f"{name}.json"
    if not path.is_file():
        available = sorted(p.stem for p in _MODULE_DIR.glob("*.json"))
        raise ImportError_(
            f"unknown stdlib module {name!r}; available: "
            f"{', '.join(available) or '(none)'}"
        )
    try:
        return InputProgram.model_validate_json(path.read_text())
    except Exception as exc:
        raise ImportError_(f"failed to load {name!r} from {path}: {exc}") from exc
