"""Unit tests for the `wirable`/`wire` import-time substitution.

The resolver in stdlib.py walks each Import, applies any `wire` clauses
to the imported module's wirables, and merges the substituted module
into the program. These tests exercise the resolver's behavior against
the real `alloc.list` stdlib module (which declares `wirable A:
mem.Allocator`).
"""

from __future__ import annotations

import json

import pytest

from quod.model import (
    EnumType,
    I32Type,
    Import,
    Program,
    StructType,
    TypeParamRef,
    WireBinding,
)
from quod.stdlib import resolve_imports, ImportError_


def _arena_wire_binding() -> WireBinding:
    return WireBinding(
        name="A",
        type=StructType(name="mem.arena.Arena"),
    )


def test_wire_substitutes_wirables_in_function_signatures():
    """After resolution, `alloc.list.list_new` should have its
    formerly-wirable references to `A` replaced by `mem.arena.Arena`
    everywhere it appears in the body."""
    prog = Program(imports=(Import(module="alloc.list", wire=(_arena_wire_binding(),)),
                            Import(module="mem.arena")))
    resolved = resolve_imports(prog)

    list_new = next(fn for fn in resolved.functions if fn.name == "alloc.list.list_new")
    # type_params should still have T (only A was wirable).
    assert [tp.name for tp in list_new.type_params] == ["T"]
    # No TypeParamRef("A") should remain in the body — A was substituted
    # to mem.arena.Arena. Walk via model_dump and assert the dispatch
    # type for at least one trait_call is the substituted Arena.
    body_dump = json.dumps([s.model_dump(mode="json") for s in list_new.body])
    assert '"name": "A"' not in body_dump.replace('"name": "alloc"', ""), \
        "wirable A should have been substituted everywhere in the body"
    assert "mem.arena.Arena" in body_dump, \
        "expected mem.arena.Arena to appear post-substitution"


def test_wire_missing_binding_errors_clearly():
    """Importing a module that declares wirables without binding them
    must error — the wirables are required parameters."""
    prog = Program(imports=(Import(module="alloc.list"),))  # no wire
    with pytest.raises(ImportError_, match="missing `wire`"):
        resolve_imports(prog)


def test_wire_binding_for_nonexistent_wirable_errors():
    """Binding a name that the module doesn't declare as a wirable is
    an error — surfaces typos at the import site. (We bind `A` too so
    we don't trip the missing-wirable check first.)"""
    prog = Program(imports=(Import(
        module="alloc.list",
        wire=(
            _arena_wire_binding(),
            WireBinding(name="NoSuchWirable", type=I32Type()),
        ),
    ),))
    with pytest.raises(ImportError_, match="unknown wirable"):
        resolve_imports(prog)


def test_wire_to_wirable_on_module_with_no_wirables_errors():
    """Wiring a module that doesn't declare wirables is an error —
    catches stale `wire` clauses left after a refactor."""
    prog = Program(imports=(Import(
        module="mem.arena",  # has no wirables
        wire=(_arena_wire_binding(),),
    ),))
    with pytest.raises(ImportError_, match="declares no wirables"):
        resolve_imports(prog)


def test_wire_typeparamref_rhs_rejected_v1():
    """v1 doesn't support forwarding-through-your-own-wirable. A wire
    RHS that's a TypeParamRef (rather than a concrete type) errors
    with a clear message."""
    prog = Program(imports=(Import(
        module="alloc.list",
        wire=(WireBinding(name="A", type=TypeParamRef(name="A")),),
    ),))
    with pytest.raises(ImportError_, match="forwarding"):
        resolve_imports(prog)
