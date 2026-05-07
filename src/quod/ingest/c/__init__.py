"""C source → quod Program (staged-lift). See c_driver for the full overview."""

from __future__ import annotations

from quod.ingest.c.c_driver import ingest_c, ingest_header
from quod.ingest.c.c_helpers import IngestError

__all__ = ["IngestError", "ingest_c", "ingest_header"]
