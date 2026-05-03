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

import functools
import inspect
import os
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typer

from quod.model import CLAIM_KINDS
from quod.templates import TEMPLATES


_STORED_REGIMES = ("axiom", "witness")
_ENFORCEMENTS = ("trust", "verify")


# ---------- Debug logging ----------
#
# Writing to stderr would corrupt the completion protocol (the shell parses
# stderr too in some setups), so debug logs go to a file. Enable with:
#
#     export QUOD_COMPLETION_DEBUG=/tmp/quod-comp.log
#     touch ~/.zshrc && exec zsh   # reload completion
#     quod -c examples/quod.toml -p <Tab>
#     tail -f /tmp/quod-comp.log
#
# Each completer call emits: env vars, ctx params (for every level), the
# completer name, the incomplete prefix, and the returned candidates (or
# the exception, with traceback).

def _debug_path() -> str | None:
    return os.environ.get("QUOD_COMPLETION_DEBUG") or None


def _log(msg: str) -> None:
    path = _debug_path()
    if not path:
        return
    try:
        with open(path, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass


def _ctx_chain(ctx) -> list[dict]:
    """Snapshot params at each context level, root-first."""
    chain = []
    cur = ctx
    while cur is not None:
        info = getattr(cur, "info_name", None)
        chain.append({"info_name": info, "params": dict(cur.params or {})})
        cur = cur.parent
    chain.reverse()
    return chain


def _traced(fn: Callable):
    """Decorator: log a completer's inputs/outputs when QUOD_COMPLETION_DEBUG is set.

    Typer inspects the callback's parameter NAMES to decide what to pass
    ("ctx" / "args" / "incomplete"). We use `functools.wraps` so
    `inspect.signature(wrapper)` follows `__wrapped__` and reports the
    original signature — Typer sees `(incomplete)` or `(ctx, incomplete)`
    just as the underlying completer declared.
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not _debug_path():
            return fn(*args, **kwargs)
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError:
            bound = None
        ctx = bound.arguments.get("ctx") if bound else None
        incomplete = (bound.arguments.get("incomplete") if bound else "") or ""
        _log(f"{fn.__name__}(incomplete={incomplete!r})")
        if ctx is not None:
            _log(f"  env _TYPER_COMPLETE_ARGS={os.environ.get('_TYPER_COMPLETE_ARGS')!r}")
            _log(f"  cwd={os.getcwd()}")
            for level in _ctx_chain(ctx):
                _log(f"  ctx {level['info_name']}: params={level['params']}")
        try:
            result = fn(*args, **kwargs)
            _log(f"  -> {result!r}")
            return result
        except Exception as e:
            _log(f"  EXC {type(e).__name__}: {e}\n{traceback.format_exc()}")
            return []
    return wrapper


def _root_params(ctx) -> dict:
    """Walk to the root context and return its parsed params (config, program, ...)."""
    cur = ctx
    while cur.parent is not None:
        cur = cur.parent
    return cur.params or {}


def _config_path_from_ctx(ctx) -> Path:
    """Pull the --config value from the root context. During completion Click
    hasn't applied the click.Path type coercion, so values may be raw strings."""
    raw = _root_params(ctx).get("config")
    return Path(raw) if raw else Path("quod.toml")


def _safe_load_program(ctx):
    """Load the selected program. Swallow all errors → return None."""
    try:
        from quod.config import load_config
        from quod.model import load_program
        program_name = _root_params(ctx).get("program")
        cfg = load_config(_config_path_from_ctx(ctx))
        prog_spec = cfg.select(program_name)
        return load_program(cfg.resolve(prog_spec.file))
    except Exception as e:
        _log(f"  _safe_load_program failed: {type(e).__name__}: {e}")
        return None


def _safe_load_config(ctx):
    try:
        from quod.config import load_config
        return load_config(_config_path_from_ctx(ctx))
    except Exception as e:
        _log(f"  _safe_load_config failed: {type(e).__name__}: {e}")
        return None


# ---------- Static enums ----------

@_traced
def claim_kinds(incomplete: str) -> list[str]:
    return [k for k in CLAIM_KINDS if k.startswith(incomplete)]


@_traced
def stored_regimes(incomplete: str) -> list[str]:
    return [r for r in _STORED_REGIMES if r.startswith(incomplete)]


@_traced
def enforcements(incomplete: str) -> list[str]:
    return [e for e in _ENFORCEMENTS if e.startswith(incomplete)]


@_traced
def template_names(incomplete: str) -> list[str]:
    return [t for t in TEMPLATES if t.startswith(incomplete)]


# ---------- Program-derived ----------

@_traced
def function_names(ctx, incomplete: str) -> list[str]:
    program = _safe_load_program(ctx)
    if program is None:
        return []
    return [fn.name for fn in program.functions if fn.name.startswith(incomplete)]


@_traced
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


@_traced
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


@_traced
def extern_names(ctx, incomplete: str) -> list[str]:
    program = _safe_load_program(ctx)
    if program is None:
        return []
    return [e.name for e in program.externs if e.name.startswith(incomplete)]


@_traced
def linkage_names(ctx, incomplete: str) -> list[str]:
    """The fixed enum of valid `--linkage` / set-linkage values. No program
    load needed; these are language-level, not program-state."""
    return [n for n in ("libc", "runtime") if n.startswith(incomplete)]


@_traced
def constant_names(ctx, incomplete: str) -> list[str]:
    program = _safe_load_program(ctx)
    if program is None:
        return []
    return [c.name for c in program.constants if c.name.startswith(incomplete)]


@_traced
def struct_names(ctx, incomplete: str) -> list[str]:
    program = _safe_load_program(ctx)
    if program is None:
        return []
    return [sd.name for sd in program.structs if sd.name.startswith(incomplete)]


@_traced
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

@_traced
def program_names(ctx, incomplete: str) -> list[str]:
    cfg = _safe_load_config(ctx)
    if cfg is None:
        return []
    return [p.name for p in cfg.programs if p.name.startswith(incomplete)]


@_traced
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
    @_traced
    def _go(ctx, incomplete: str) -> list[str]:
        try:
            from quod.providers import all_providers
            return [
                p.name for p in all_providers().values()
                if p.regime == regime and p.name.startswith(incomplete)
            ]
        except Exception:
            return []
    _go.__name__ = f"provider_names_for[{regime}]"
    return _go
