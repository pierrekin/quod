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
