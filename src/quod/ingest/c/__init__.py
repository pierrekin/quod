"""C-source ingestion via libclang.

`c_layer_a` produces the verbatim C subtree (the `C*` source-language
node family); `c_layer_b` produces the c-like-quod transcription with
`c.*` extensions; `c_helpers` is shared utility (op tables, type
predicates, AST navigation); `c_driver` is the top-level entry-point
(`ingest_c` / `ingest_header`) and the libclang-translation-unit
driver. The full staged-lift overview lives in `c_driver`'s docstring.
"""

from __future__ import annotations

from quod.ingest.c.driver import ingest_c, ingest_header
from quod.ingest.c.helpers import IngestError

__all__ = ["IngestError", "ingest_c", "ingest_header"]
