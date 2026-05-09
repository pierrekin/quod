"""quod LSP server, built on pygls.

Workspace model: the LSP `workspaceFolders` is a *set* of quod projects
(each a directory containing quod.toml). qui's "workspace" of multiple
projects maps directly onto this. Programs across all projects are
exposed as a flat, project-qualified list under
`experimental.quod.programs`.

Lifecycle:
  initialize          — scan each workspaceFolder for quod.toml; advertise
                        the union of programs.
  workspace/didChangeWorkspaceFolders
                      — react to add/remove; push quod/programsChanged
                        notification with the updated flat list.
  quod/listPrograms   — pull alternative for the same data.
  quod/setActiveProgram
                      — params {project, name} for a workspace program;
                        params {file} for a standalone program.json
                        (no project). Returns summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from lsprotocol import types
from pygls.lsp.server import LanguageServer
from pygls.protocol.language_server import LanguageServerProtocol, lsp_method

from quod.config import CONFIG_FILENAME, Config, ProgramSpec, load_config
from quod.hashing import short_hash
from quod.model import Program, load_program
from quod.model.pretty import (
    format_c_fn,
    format_equivalence_metadata,
    format_function,
    format_struct_def,
)
from quod.model.pretty import _format_type as _fmt_type  # private but reusable

PROTOCOL_VERSION = 1
_CUSTOM_METHODS = [
    "quod/listPrograms",
    "quod/setActiveProgram",
    "quod/getProgramOutline",
    "quod/getLiftTrace",
]
_PROGRAMS_CHANGED = "quod/programsChanged"

try:
    _QUOD_VERSION = version("quod")
except PackageNotFoundError:
    _QUOD_VERSION = "unknown"


@dataclass
class ActiveProgram:
    label: str
    project: str | None        # project name (basename of root) or None for file mode
    project_path: Path | None  # project root or None
    summary: dict[str, int]
    via_name: str | None
    via_file: Path | None
    program: Program           # cached loaded model so the outline isn't disk-bound


def _file_uri_to_path(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path))


def _project_name(root: Path) -> str:
    return root.name or str(root)


def _program_payload(spec: ProgramSpec, root: Path) -> dict[str, Any]:
    return {
        "project": _project_name(root),
        "projectPath": str(root),
        "name": spec.name,
        "file": str(spec.file),
        "absFile": str(root / spec.file),
        "version": spec.version,
    }


def _summarize(prog: Program) -> dict[str, int]:
    claim_count = sum(len(fn.claims) for fn in prog.functions)
    claim_count += sum(len(fn.claims) for fn in prog.structured_functions)
    claim_count += sum(len(ext.claims) for ext in prog.externs)
    return {
        "functions": len(prog.functions),
        "structuredFunctions": len(prog.structured_functions),
        "externs": len(prog.externs),
        "structs": len(prog.structs),
        "claims": claim_count,
        "equivalences": len(prog.equivalences),
    }


class QuodServer(LanguageServer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Project root → loaded Config. Insertion order matters for the
        # advertised programs list.
        self.projects: dict[Path, Config] = {}
        self.active: ActiveProgram | None = None

    def add_project(self, root: Path) -> bool:
        """Returns True iff a project was newly added (not a no-op)."""
        if not root.is_dir():
            return False
        if root in self.projects:
            return False
        toml = root / CONFIG_FILENAME
        if not toml.is_file():
            return False
        try:
            self.projects[root] = load_config(toml)
            return True
        except Exception:
            return False

    def remove_project(self, root: Path) -> bool:
        if root not in self.projects:
            return False
        del self.projects[root]
        # If the active program was in this project, drop it.
        if self.active is not None and self.active.project_path == root:
            self.active = None
        return True

    def programs_payload(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for root, config in self.projects.items():
            for spec in config.programs:
                out.append(_program_payload(spec, root))
        return out


class QuodProtocol(LanguageServerProtocol):
    @lsp_method(types.INITIALIZE)
    def lsp_initialize(self, params):  # type: ignore[override]
        # Add every workspaceFolder that contains a quod.toml.
        for folder in (params.workspace_folders or []):
            path = _file_uri_to_path(folder.uri)
            if path is not None:
                self._server.add_project(path)

        gen = super().lsp_initialize(params)
        try:
            value = None
            while True:
                value = gen.send(value)
                value = yield value
        except StopIteration as stop:
            result: types.InitializeResult = stop.value
            result.capabilities.experimental = {
                "quod": {
                    "version": PROTOCOL_VERSION,
                    "methods": list(_CUSTOM_METHODS),
                    "programs": self._server.programs_payload(),
                }
            }
            return result


def build_server() -> QuodServer:
    server = QuodServer(
        name="quod-lsp",
        version=_QUOD_VERSION,
        protocol_cls=QuodProtocol,
    )

    def _push_programs_changed() -> None:
        server.protocol.notify(
            _PROGRAMS_CHANGED,
            {"programs": server.programs_payload()},
        )

    @server.feature(types.WORKSPACE_DID_CHANGE_WORKSPACE_FOLDERS)
    def did_change_folders(
        ls: QuodServer, params: types.DidChangeWorkspaceFoldersParams
    ) -> None:
        for added in params.event.added or []:
            path = _file_uri_to_path(added.uri)
            if path is not None:
                ls.add_project(path)
        for removed in params.event.removed or []:
            path = _file_uri_to_path(removed.uri)
            if path is not None:
                ls.remove_project(path)
        _push_programs_changed()

    @server.feature("quod/listPrograms")
    def list_programs(ls: QuodServer, _params: Any) -> dict[str, Any]:
        return {"programs": ls.programs_payload()}

    @server.feature("quod/setActiveProgram")
    def set_active_program(ls: QuodServer, params: Any) -> dict[str, Any]:
        def _get(key: str) -> Any:
            if params is None:
                return None
            if hasattr(params, key):
                return getattr(params, key)
            if hasattr(params, "get"):
                return params.get(key)
            return None

        project_name = _get("project")
        name = _get("name")
        file = _get("file")

        if file is not None:
            if project_name is not None or name is not None:
                raise ValueError("setActiveProgram: pass {project, name} or {file}, not both")
            file_path = Path(file)
            program = load_program(file_path)
            ls.active = ActiveProgram(
                label=str(file_path),
                project=None,
                project_path=None,
                summary=_summarize(program),
                via_name=None,
                via_file=file_path,
                program=program,
            )
            return {
                "label": ls.active.label,
                "project": None,
                "summary": ls.active.summary,
                "via": "file",
            }

        if name is None:
            raise ValueError("setActiveProgram: needs {project, name} or {file}")

        # Resolve the project. If `project` is given, match by basename;
        # otherwise require a single open project.
        candidates = [
            (root, cfg)
            for root, cfg in ls.projects.items()
            if project_name is None or _project_name(root) == project_name
        ]
        if not candidates:
            raise ValueError(f"no open project named {project_name!r}")
        if project_name is None and len(candidates) > 1:
            raise ValueError("multiple projects open; pass {project, name}")
        root, cfg = candidates[0]
        spec = next((p for p in cfg.programs if p.name == name), None)
        if spec is None:
            avail = ", ".join(p.name for p in cfg.programs) or "(none)"
            raise ValueError(
                f"no program named {name!r} in {_project_name(root)}; available: {avail}"
            )
        program_path = root / spec.file
        program = load_program(program_path)
        ls.active = ActiveProgram(
            label=name,
            project=_project_name(root),
            project_path=root,
            summary=_summarize(program),
            via_name=name,
            via_file=None,
            program=program,
        )
        return {
            "label": name,
            "project": _project_name(root),
            "summary": ls.active.summary,
            "via": "name",
        }

    @server.feature("quod/getLiftTrace")
    def get_lift_trace(ls: QuodServer, _params: Any) -> dict[str, Any]:
        """Cross-layer view: one row per function name, with the layer-A
        (CFn), layer-B (structured_functions), and layer-C (functions)
        renderings side-by-side. The hash anchor is the layer-C function's
        short hash; rows missing layer-C use the first available hash.

        First-cut join: name match. Honest provenance (via `edges` /
        `equivalences`) follows once the lifters wire those routinely."""
        if ls.active is None:
            return {"rows": []}
        prog = ls.active.program

        layer_c_by_name: dict[str, Any] = {fn.name: fn for fn in prog.functions}
        layer_b_by_name: dict[str, Any] = {fn.name: fn for fn in prog.structured_functions}
        layer_a_by_name: dict[str, Any] = {}
        for unit in prog.source_units:
            for cfn in getattr(unit, "functions", ()):
                layer_a_by_name.setdefault(cfn.name, cfn)

        names: list[str] = []
        seen: set[str] = set()
        # Iterate in order of layer-A → B → C so the first-seen layer
        # determines row order. Layer-C functions tend to be the densest
        # set and the canonical anchor, but other layers may have entries
        # without a layer-C counterpart in incomplete programs.
        for src in (layer_a_by_name, layer_b_by_name, layer_c_by_name):
            for name in src.keys():
                if name in seen:
                    continue
                seen.add(name)
                names.append(name)

        rows: list[dict[str, Any]] = []
        for name in names:
            a = layer_a_by_name.get(name)
            b = layer_b_by_name.get(name)
            c = layer_c_by_name.get(name)
            anchor = c or b or a
            rows.append({
                "name": name,
                "hash": short_hash(anchor) if anchor is not None else "",
                "layerA": format_c_fn(a) if a is not None else "",
                "layerB": format_function(b) if b is not None else "",
                "layerC": format_function(c) if c is not None else "",
            })
        return {"rows": rows}

    @server.feature("quod/getProgramOutline")
    def get_program_outline(ls: QuodServer, _params: Any) -> dict[str, Any]:
        """Structured outline of the active program — one category per
        node-cluster, with one-line summary strings per item. Returned in
        a fixed category order so the client can render directly without
        sorting. Empty categories are reported with `count: 0` and
        `items: []` so the outline shape is stable."""
        if ls.active is None:
            return {"categories": []}
        return {"categories": _build_outline(ls.active.program)}

    return server


def _fn_signature(fn: Any) -> str:
    """One-line `name<TPs>(p: T, q: U) -> R` for Functions / structured Functions."""
    type_params = ""
    if getattr(fn, "type_params", ()):
        tp = ", ".join(p.name for p in fn.type_params)
        type_params = f"<{tp}>"
    sig_params = ", ".join(
        f"{p.name}: {_fmt_type(p.type)}" for p in fn.params
    )
    return f"{fn.name}{type_params}({sig_params}) -> {_fmt_type(fn.return_type)}"


def _extern_signature(ext: Any) -> str:
    """Best-effort one-liner for an ExternFunction. Externs use either an
    `arity` shortcut (all-i32) or explicit `param_types` + `return_type`,
    optionally with `varargs`."""
    if getattr(ext, "arity", 0):
        params = ", ".join(["i32"] * ext.arity)
        ret = "i32"
    else:
        param_types = getattr(ext, "param_types", ()) or ()
        params = ", ".join(_fmt_type(t) for t in param_types)
        rt = getattr(ext, "return_type", None)
        ret = _fmt_type(rt) if rt is not None else "?"
    if getattr(ext, "varargs", False):
        params = f"{params}, ..." if params else "..."
    return f"{ext.name}({params}) -> {ret}"


def _build_outline(prog: Program) -> list[dict[str, Any]]:
    cats: list[dict[str, Any]] = []

    def add(key: str, label: str, items: list[str]) -> None:
        cats.append({"key": key, "label": label, "count": len(items), "items": items})

    add("fns", "fns", [_fn_signature(fn) for fn in prog.functions])
    add("structuredFns", "structured fns",
        [_fn_signature(fn) for fn in prog.structured_functions])
    add("externs", "externs", [_extern_signature(ext) for ext in prog.externs])
    add("structs", "structs", [format_struct_def(sd) for sd in prog.structs])

    enum_items: list[str] = []
    for ed in prog.enums:
        variant_names = ", ".join(v.name for v in ed.variants)
        enum_items.append(f"enum {ed.name} {{ {variant_names} }}")
    add("enums", "enums", enum_items)

    # Claims live on functions / structured_functions / externs. Aggregate
    # with provenance so the user knows where each came from.
    claim_items: list[str] = []
    for fn in prog.functions:
        for c in fn.claims:
            claim_items.append(f"fn {fn.name}: {getattr(c, 'kind', type(c).__name__)}")
    for fn in prog.structured_functions:
        for c in fn.claims:
            claim_items.append(f"sfn {fn.name}: {getattr(c, 'kind', type(c).__name__)}")
    for ext in prog.externs:
        for c in ext.claims:
            claim_items.append(f"ext {ext.name}: {getattr(c, 'kind', type(c).__name__)}")
    add("claims", "claims", claim_items)

    # Statements live inside functions; aggregate with `fn:short_hash kind`.
    stmt_items: list[str] = []
    for fn in prog.functions:
        for s in fn.body.stmts:
            kind = getattr(s, "kind", type(s).__name__)
            short = (getattr(s, "_hash", "") or "")[:8]
            stmt_items.append(f"{fn.name}:{short} {kind}" if short else f"{fn.name} {kind}")
    add("stmts", "stmts", stmt_items)

    const_items: list[str] = []
    for c in prog.constants:
        # StringConstant: name + value (truncated).
        val = getattr(c, "value", "")
        if len(val) > 40:
            val = val[:37] + "…"
        const_items.append(f"{c.name} = {val!r}")
    add("constants", "constants", const_items)

    equiv_items: list[str] = []
    for eq in prog.equivalences:
        equiv_items.append(f"{eq.a_node_id} ~ {eq.b_node_id}{format_equivalence_metadata(eq)}")
    add("equivalences", "equivalences", equiv_items)

    src_items = [getattr(u, "source_path", str(getattr(u, "id", "?"))) for u in prog.source_units]
    add("sourceUnits", "source units", src_items)

    bin_items = [getattr(u, "source_path", str(getattr(u, "id", "?"))) for u in prog.binary_units]
    add("binaryUnits", "binary units", bin_items)

    add("traits", "traits", [t.name for t in prog.traits])
    add("impls", "impls", [
        f"impl {getattr(i, 'trait_name', '?')} for {getattr(i, 'target_type_name', '?')}"
        for i in prog.impls
    ])

    return cats


def serve() -> int:
    """Run the LSP loop on stdin/stdout. Returns a process exit code."""
    server = build_server()
    server.start_io()
    return 0
