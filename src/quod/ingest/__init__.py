"""Ingest source code from other languages into a quod Program.

C is the only supported source today. Each language gets its own submodule.
The output is a normal `Program` — every existing CLI command (`show`,
`fn ls`, `claim add`, `build`, `run`) works on it without special cases.
"""

from quod.ingest.c import IngestError, ingest_c

__all__ = ["IngestError", "ingest_c"]
