"""`quod lsp` — language server (JSON-RPC over stdio).

The handler is intentionally tiny: open the LSP loop on stdin/stdout and
exit with whatever code it returns. Project state isn't loaded eagerly —
the server starts blank and learns about a project from the client (via
`initialize` params or, later, a `quod/loadProject` request).
"""

from __future__ import annotations

import sys

import typer

from quod.cli.app import app
from quod.lsp import serve


@app.command(name="lsp")
def lsp_cmd() -> None:
    """Run the quod language server (JSON-RPC over stdio)."""
    raise typer.Exit(serve())
