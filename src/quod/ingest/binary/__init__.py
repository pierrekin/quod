"""Binary frontend — Ghidra-driven ingest of `.so`/`.exe`/`.o` artifacts
into the same CPG as C source.

The binary frontend's purpose is *not* to recompile binaries — Layer-A
binary ingest is inert by contract (see
`.scratch/ghidra/01-layer-a-nodes.md`). The aim is to land binary
artifacts as nodes in the same `Program` graph as C source so cross-
layer `Equivalence` claims have a binary-side endpoint to anchor on,
and so future claim providers (range hints from Ghidra control flow,
the relational SMT prover the strategy memo names "the bet") have an
input.

Two entry points:
- `ingest_binary(path, ...)` spawns `ghidra-analyzeHeadless`, parses
  the JSON dump, builds a `BinUnit`, returns the updated `Program`.
- `ingest_binary_dump(json_path, ...)` skips the subprocess and parses
  an existing JSON dump produced by `binary_ghidra_export.py`. The
  test path; also useful when a Ghidra project already exists and
  re-running analysis would be wasteful.

Both write `BinUnit` to `Program.binary_units` and run the equivalence
seeder before returning.
"""

from quod.ingest.binary.driver import (
    BinaryIngestError,
    ingest_binary,
    ingest_binary_dump,
    seed_binary_equivalences,
)

__all__ = [
    "BinaryIngestError",
    "ingest_binary",
    "ingest_binary_dump",
    "seed_binary_equivalences",
]
