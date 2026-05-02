"""Higher-level Program operations: ref resolution + construction primitives.

`find_function_ref` is the unified addressing entry point: a function ref can
be a name OR a content-hash prefix. The CLI uses it everywhere a function is
referenced, so an agent can address by either.

Construction primitives accept JSON specs (dicts) because that's the natural
shape an agent emits via tool-call arguments. Pydantic validates the spec
at the editor boundary; structural errors never reach the lowering pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from quod.hashing import node_hash
from quod.model import (
    Function,
    Program,
    Statement,
    StringConstant,
    StructDef,
    replace_function,
)


# Discriminated-union sum types can't be Pydantic-validated as bare types,
# so we use a TypeAdapter to validate dicts against the union.
_StatementAdapter: TypeAdapter[Statement] = TypeAdapter(Statement)


# ---------- Reference resolution ----------

def find_function_ref(program: Program, ref: str) -> Function:
    """Resolve a function ref (name OR content-hash prefix) to a Function."""
    by_name = [fn for fn in program.functions if fn.name == ref]
    if by_name:
        return by_name[0]
    by_hash = [fn for fn in program.functions if node_hash(fn).startswith(ref)]
    if not by_hash:
        raise KeyError(f"no function matches {ref!r} (by name or hash prefix)")
    if len({node_hash(fn) for fn in by_hash}) > 1:
        names = sorted(fn.name for fn in by_hash)
        raise ValueError(f"ref {ref!r} is ambiguous across functions: {names}")
    return by_hash[0]


def find_statement_index(fn: Function, ref: str) -> int:
    """Return the index of the statement in fn.body matching `ref` (hash prefix)."""
    matches = [(i, s) for i, s in enumerate(fn.body) if node_hash(s).startswith(ref)]
    if not matches:
        raise KeyError(f"no statement in {fn.name!r} matches hash prefix {ref!r}")
    if len({node_hash(s) for _, s in matches}) > 1:
        raise ValueError(f"prefix {ref!r} is ambiguous within {fn.name!r}")
    return matches[0][0]


# ---------- Construction ----------

def add_function_to_program(program: Program, function: Function) -> Program:
    if any(fn.name == function.name for fn in program.functions):
        raise ValueError(f"function {function.name!r} already exists")
    return program.model_copy(update={"functions": program.functions + (function,)})


def add_statement_in_function(
    program: Program,
    function: Function,
    new_stmt: Statement,
    *,
    where: str,                   # "end" | "before" | "after" | "start"
    anchor_ref: str | None = None,
) -> Program:
    body = list(function.body)
    match where:
        case "end":
            body.append(new_stmt)
        case "start":
            body.insert(0, new_stmt)
        case "before":
            assert anchor_ref is not None, "anchor_ref required for 'before'"
            body.insert(find_statement_index(function, anchor_ref), new_stmt)
        case "after":
            assert anchor_ref is not None, "anchor_ref required for 'after'"
            body.insert(find_statement_index(function, anchor_ref) + 1, new_stmt)
        case _:
            raise ValueError(f"unknown anchor mode {where!r}")
    new_fn = function.model_copy(update={"body": tuple(body)})
    return replace_function(program, new_fn)


def remove_statement_in_function(
    program: Program,
    function: Function,
    hash_prefix: str,
) -> Program:
    """Remove the statement matching `hash_prefix` from the function body."""
    idx = find_statement_index(function, hash_prefix)
    body = function.body[:idx] + function.body[idx + 1:]
    new_fn = function.model_copy(update={"body": body})
    return replace_function(program, new_fn)


def add_constant_to_program(program: Program, constant: StringConstant) -> Program:
    """Append a string constant. Errors if the name collides."""
    if any(c.name == constant.name for c in program.constants):
        raise ValueError(f"constant {constant.name!r} already declared")
    return program.model_copy(update={"constants": program.constants + (constant,)})


def remove_constant_from_program(program: Program, name: str) -> Program:
    """Drop a string constant. Permissive about dangling StringRefs — those
    surface at lower time. Mirrors `remove_function`'s permissive stance."""
    kept = tuple(c for c in program.constants if c.name != name)
    if kept == program.constants:
        raise KeyError(f"no constant named {name!r}")
    return program.model_copy(update={"constants": kept})


def remove_extern_from_program(program: Program, name: str) -> Program:
    """Drop an extern declaration. Permissive about dangling Calls — those
    surface at lower time as 'call to undeclared function'."""
    kept = tuple(e for e in program.externs if e.name != name)
    if kept == program.externs:
        raise KeyError(f"no extern named {name!r}")
    return program.model_copy(update={"externs": kept})


def add_struct_to_program(program: Program, struct_def: StructDef) -> Program:
    """Append a struct definition. Errors if the name collides with an
    existing struct. Cross-cutting validation (dangling refs, cycles) is
    enforced by the Program-level model_validator on the returned object."""
    if any(sd.name == struct_def.name for sd in program.structs):
        raise ValueError(f"struct {struct_def.name!r} already declared")
    return program.model_copy(update={"structs": program.structs + (struct_def,)})


def remove_struct_from_program(program: Program, name: str) -> Program:
    """Drop a struct definition. Strict: refuses if any function param,
    return type, local, struct field, or extern signature references it.
    The model validator would catch some of these on save, but reporting
    the specific reference here is much friendlier."""
    if not any(sd.name == name for sd in program.structs):
        raise KeyError(f"no struct named {name!r}")
    refs = _struct_references(program, name)
    if refs:
        raise ValueError(
            f"refusing to remove struct {name!r}: still referenced by "
            f"{', '.join(refs)}"
        )
    kept = tuple(sd for sd in program.structs if sd.name != name)
    return program.model_copy(update={"structs": kept})


def _struct_references(program: Program, name: str) -> tuple[str, ...]:
    """Return human-readable references to struct `name` in the program.

    Covers: other structs' fields, function params/returns, extern
    params/returns. StructInit/FieldRead/FieldSet inside function bodies
    aren't enumerated here — they'd be unreachable without a typed
    container, and the Program validator catches dangling StructInit names.
    """
    out: list[str] = []
    for sd in program.structs:
        if sd.name == name:
            continue
        for f in sd.fields:
            if _type_names(f.type) == name:
                out.append(f"struct {sd.name}.{f.name}")
    for fn in program.functions:
        if _type_names(fn.return_type) == name:
            out.append(f"function {fn.name} return type")
        for p in fn.params:
            if _type_names(p.type) == name:
                out.append(f"function {fn.name} param {p.name}")
    for ext in program.externs:
        if _type_names(ext.return_type) == name:
            out.append(f"extern {ext.name} return type")
        for t in ext.param_types:
            if _type_names(t) == name:
                out.append(f"extern {ext.name} param")
    return tuple(out)


def _type_names(t) -> str | None:
    """Return the struct name if `t` is a StructType, else None."""
    from quod.model import StructType
    return t.name if isinstance(t, StructType) else None


# ---------- JSON ingest ----------

# Hint shown after pydantic validation errors so an agent knows where to look.
_SCHEMA_HINT = (
    "\n\n(see `quod schema` for the canonical shape of any node — "
    "e.g. `quod schema quod.let` or `quod schema --category statement`)"
)


def parse_function_spec(raw: str) -> Function:
    """Parse a JSON Function spec.

    Uses model_validate_json (not model_validate on a dict) so JSON arrays
    legitimately coerce into tuple-typed fields, even though _Node has
    strict=True. JSON-mode coercion is the right surface for the agent.
    """
    try:
        return Function.model_validate_json(raw)
    except ValidationError as e:
        raise ValueError(f"invalid function spec:\n{e}{_SCHEMA_HINT}") from e


def parse_statement_spec(raw: str) -> Statement:
    try:
        return _StatementAdapter.validate_json(raw)
    except ValidationError as e:
        raise ValueError(f"invalid statement spec:\n{e}{_SCHEMA_HINT}") from e


def read_json_arg(arg: str) -> str:
    """Return raw JSON content from a file path, or from stdin if arg == '-'."""
    return sys.stdin.read() if arg == "-" else Path(arg).read_text()
