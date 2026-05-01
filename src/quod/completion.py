"""Shell completion callbacks for the quod CLI.

The shell completion script (installed via `quod --install-completion`)
re-invokes `quod` in a subprocess at completion time, with magic env vars
that say "give me candidates for argument N of this command line." Click
parses what it can, then calls the matching completer here.

Completers must:
  - be fast (every <Tab> press runs them)
  - never raise — any exception turns into an empty completion list, which
    is fine for the user; a stack trace shoved into their shell is not
  - prefix-filter their results (Click does final filtering too, but pre-
    filtering keeps the wire format small for big programs)

The protocol is per-argument: each `typer.Argument(autocompletion=...)` /
`typer.Option(autocompletion=...)` wires one of these in. There's no
single CLI-wide table; the CLI imports specific completers below.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typer

from quod.model import CLAIM_KINDS
from quod.templates import TEMPLATES


_STORED_REGIMES = ("axiom", "witness")
_ENFORCEMENTS = ("trust", "verify")


def _root_params(ctx) -> dict:
    """Walk to the root context and return its parsed params (config, program, ...)."""
    cur = ctx
    while cur.parent is not None:
        cur = cur.parent
    return cur.params or {}


def _safe_load_program(ctx):
    """Load the selected program. Swallow all errors → return None."""
    try:
        from quod.config import load_config
        from quod.model import load_program
        params = _root_params(ctx)
        config_path = params.get("config") or Path("quod.toml")
        program_name = params.get("program")
        cfg = load_config(config_path)
        prog_spec = cfg.select(program_name)
        return load_program(cfg.resolve(prog_spec.file))
    except Exception:
        return None


def _safe_load_config(ctx):
    try:
        from quod.config import load_config
        params = _root_params(ctx)
        config_path = params.get("config") or Path("quod.toml")
        return load_config(config_path)
    except Exception:
        return None


# ---------- Static enums ----------

def claim_kinds(incomplete: str) -> list[str]:
    return [k for k in CLAIM_KINDS if k.startswith(incomplete)]


def stored_regimes(incomplete: str) -> list[str]:
    return [r for r in _STORED_REGIMES if r.startswith(incomplete)]


def enforcements(incomplete: str) -> list[str]:
    return [e for e in _ENFORCEMENTS if e.startswith(incomplete)]


def template_names(incomplete: str) -> list[str]:
    return [t for t in TEMPLATES if t.startswith(incomplete)]


# ---------- Program-derived ----------

def function_names(ctx, incomplete: str) -> list[str]:
    program = _safe_load_program(ctx)
    if program is None:
        return []
    return [fn.name for fn in program.functions if fn.name.startswith(incomplete)]


def function_or_hash(ctx, incomplete: str) -> list[str]:
    """Function names + node hash prefixes — for args that accept either."""
    program = _safe_load_program(ctx)
    if program is None:
        return []
    from quod.hashing import HASH_DISPLAY_LEN, walk
    out = [fn.name for fn in program.functions if fn.name.startswith(incomplete)]
    seen: set[str] = set()
    for hn in walk(program):
        h = hn.hash[:HASH_DISPLAY_LEN]
        if h.startswith(incomplete) and h not in seen:
            seen.add(h)
            out.append(h)
    return out


def hash_prefixes(ctx, incomplete: str) -> list[str]:
    program = _safe_load_program(ctx)
    if program is None:
        return []
    from quod.hashing import HASH_DISPLAY_LEN, walk
    seen: set[str] = set()
    out: list[str] = []
    for hn in walk(program):
        h = hn.hash[:HASH_DISPLAY_LEN]
        if h.startswith(incomplete) and h not in seen:
            seen.add(h)
            out.append(h)
    return out


def extern_names(ctx, incomplete: str) -> list[str]:
    program = _safe_load_program(ctx)
    if program is None:
        return []
    return [e.name for e in program.externs if e.name.startswith(incomplete)]


def constant_names(ctx, incomplete: str) -> list[str]:
    program = _safe_load_program(ctx)
    if program is None:
        return []
    return [c.name for c in program.constants if c.name.startswith(incomplete)]


def param_names_for_function(ctx, incomplete: str) -> list[str]:
    """For `claim add FN KIND TARGET` — list FN's params.

    Reads `function` from the current command's already-parsed params, then
    resolves it (by name or hash prefix) and lists its Param.name values.
    """
    program = _safe_load_program(ctx)
    if program is None:
        return []
    fn_ref = (ctx.params or {}).get("function")
    if not fn_ref:
        return []
    try:
        from quod.editor import find_function_ref
        fn = find_function_ref(program, fn_ref)
    except Exception:
        return []
    return [p.name for p in fn.params if p.name.startswith(incomplete)]


# ---------- Config-derived ----------

def program_names(ctx, incomplete: str) -> list[str]:
    cfg = _safe_load_config(ctx)
    if cfg is None:
        return []
    return [p.name for p in cfg.programs if p.name.startswith(incomplete)]


def bin_names(ctx, incomplete: str) -> list[str]:
    cfg = _safe_load_config(ctx)
    if cfg is None:
        return []
    program_name = _root_params(ctx).get("program")
    try:
        targets = cfg.programs if program_name is None else (cfg.select(program_name),)
    except Exception:
        return []
    out: list[str] = []
    for prog in targets:
        for b in prog.bins:
            if b.name.startswith(incomplete):
                out.append(b.name)
    return out


# ---------- Provider registry ----------

def provider_names_for(regime: str) -> Callable[[object, str], list[str]]:
    """Curried completer: only providers whose regime matches `regime`."""
    def _go(ctx, incomplete: str) -> list[str]:
        try:
            from quod.providers import all_providers
            return [
                p.name for p in all_providers().values()
                if p.regime == regime and p.name.startswith(incomplete)
            ]
        except Exception:
            return []
    return _go
