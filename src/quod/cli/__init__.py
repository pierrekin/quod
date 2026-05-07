"""Typer CLI. Noun-first sub-apps; each leaf command maps 1:1 to a tool call.

Layout:
    quod init / check / build / run     # lifecycle
    quod show [--hashes] / find PREFIX  # whole-program inspection
    quod fn ...                         # functions
    quod claim ...                      # claims
    quod stmt ...                       # statements
    quod extern ...                     # externs
    quod note ...                       # notes

Every command except `init` requires a quod.toml. `--config PATH` (default
./quod.toml) selects which one. Paths inside quod.toml resolve relative to
its parent dir, so `quod run -c /elsewhere/quod.toml` works regardless of
CWD; the launched binary inherits CWD from the invocation.

Function and statement references accept either a name (functions only) or
a content-hash prefix (any node).
"""

from __future__ import annotations

# Import the root app first so `app` and the sub-app instances exist
# before any `@<sub_app>.command(...)` decorator fires.
from quod.cli.cli_app import app

# Import each command-registering module; the decorators inside attach
# commands to their respective sub-apps. Order is irrelevant — each
# module references already-defined sub-app instances.
from quod.cli import cli_inspect  # noqa: F401
from quod.cli import cli_ingest  # noqa: F401
from quod.cli import cli_fn  # noqa: F401
from quod.cli import cli_claim  # noqa: F401
from quod.cli import cli_equiv  # noqa: F401
from quod.cli import cli_stmt  # noqa: F401
from quod.cli import cli_const  # noqa: F401
from quod.cli import cli_extern  # noqa: F401
from quod.cli import cli_struct  # noqa: F401
from quod.cli import cli_enum  # noqa: F401
from quod.cli import cli_note  # noqa: F401
from quod.cli import cli_provider  # noqa: F401

# Re-exports for tests that reach into private CLI internals.
from quod.cli.cli_claim import _verify_justification  # noqa: F401
from quod.cli.cli_equiv import _verify_equivalence_justification  # noqa: F401
from quod.cli.cli_state import _state  # noqa: F401


if __name__ == "__main__":
    app()
