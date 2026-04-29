"""Content-addressable hashes for CPG nodes (Merkle-style).

Every _Node has a content hash derived from canonical-JSON of its fields,
including the kind discriminator. Hashes propagate up: a Function's hash
covers its name, params, body, and claims, so any deep edit re-hashes the
function (and the program).

Hashes are computed on demand. Address by full hex hash or any unambiguous
prefix; the CLI displays the first HASH_DISPLAY_LEN chars.

Content-equivalence note: two structurally-identical subtrees share a hash.
`find_by_prefix` treats them as one (a single match), and `replace_node`
replaces every occurrence. That's the right semantic for a content-addressed
store; if a caller wants to disambiguate physically-distinct same-content
nodes, they should address the parent instead.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import NamedTuple

from cpg.model import _Node


HASH_DISPLAY_LEN = 12


def node_hash(node: _Node) -> str:
    """Full hex SHA-256 of the node's canonical JSON content."""
    payload = node.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def short_hash(node: _Node) -> str:
    return node_hash(node)[:HASH_DISPLAY_LEN]


class HashedNode(NamedTuple):
    hash: str
    node: _Node


def walk(node: _Node) -> Iterator[HashedNode]:
    """Pre-order walk: yield (hash, node) for the node and every nested descendant."""
    yield HashedNode(node_hash(node), node)
    for _name, value in node:
        yield from _walk_value(value)


def _walk_value(value) -> Iterator[HashedNode]:
    if isinstance(value, _Node):
        yield from walk(value)
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from _walk_value(v)


def find_by_prefix(root: _Node, prefix: str) -> _Node:
    """Return the unique node whose full hash starts with `prefix`.

    Treats content-equivalent occurrences as one match. Errors on miss
    (KeyError) or on prefix matching multiple distinct hashes (ValueError).
    """
    matches: dict[str, _Node] = {}
    for hn in walk(root):
        if hn.hash.startswith(prefix):
            matches.setdefault(hn.hash, hn.node)
    if not matches:
        raise KeyError(f"no node matches hash prefix {prefix!r}")
    if len(matches) > 1:
        sample = sorted(h[:HASH_DISPLAY_LEN] for h in matches)
        raise ValueError(f"prefix {prefix!r} is ambiguous; matches: {sample}")
    return next(iter(matches.values()))


def replace_node(root: _Node, target_ref: str, new_node: _Node) -> _Node:
    """Return a new root with every occurrence of the target node replaced.

    `target_ref` may be a full hash or a prefix; it's resolved via
    find_by_prefix, so ambiguity is rejected before any mutation occurs.
    """
    target = find_by_prefix(root, target_ref)
    full_hash = node_hash(target)

    def _go(node: _Node) -> _Node:
        if node_hash(node) == full_hash:
            return new_node
        updates: dict[str, object] = {}
        for name, value in node:
            replaced = _go_value(value)
            if replaced is not value:
                updates[name] = replaced
        return node.model_copy(update=updates) if updates else node

    def _go_value(value):
        if isinstance(value, _Node):
            return _go(value)
        if isinstance(value, tuple):
            new = tuple(_go_value(v) for v in value)
            return new if any(a is not b for a, b in zip(new, value)) else value
        return value

    return _go(root)
