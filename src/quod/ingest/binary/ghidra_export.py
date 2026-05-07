"""Ghidra → JSON exporter using PyGhidra.

Ghidra 12 dropped Jython 2.7 (the design-doc memo's preferred host
runtime); the modern Python integration is **PyGhidra** — a JPype
bridge that runs Ghidra's JVM in-process from CPython 3. This module
is the CPython side of that integration: it loads a binary into an
ephemeral Ghidra project, drives auto-analysis and the decompiler,
and emits the same `schema_version=1` JSON contract documented in
`.scratch/ghidra/02-ghidra-export.md`.

The JSON contract is preserved deliberately — even though we now run
in the same Python process, keeping a stable JSON shape means swapping
Ghidra for radare2 / BAP / angr later means producing the same shape
elsewhere, not rewriting the ingester. The `subprocess+JSON` boundary
is replaced by an `in-process+JSON` boundary; the *contract* is what
matters.

Requires the `pyghidra` package and a Ghidra install discoverable via
the `GHIDRA_INSTALL_DIR` env var (or wherever pyghidra's launcher
finds Ghidra by default). Both checks fire on the *first* call to
`export_to_json` — module import is dependency-free so other ingest
paths don't pay the import cost.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


# JSON contract version emitted by this exporter. Bump when:
#  - A field is removed or renamed.
#  - A field's type changes incompatibly (e.g. the hex-string → int
#    flip we did for varnode offsets — that one happened before any
#    external consumers existed; future changes of that flavor MUST
#    bump).
#  - A field's semantics change (same shape, different meaning).
# Do NOT bump for purely additive optional fields — the parser uses
# `.get(...)` throughout, so new fields never break old readers.
# Update `SUPPORTED_SCHEMA_VERSION` in driver.py in lockstep.
SCHEMA_VERSION = 1


_PYGHIDRA_MISSING_MSG = (
    "pyghidra is required for binary ingest but is not installed. "
    "Install it with: pip install pyghidra (or `uv add pyghidra`). "
    "Also make sure GHIDRA_INSTALL_DIR points at your Ghidra install "
    "(e.g. /opt/ghidra) — pyghidra needs a Ghidra distribution at runtime."
)


def export_to_json(binary_path: Path, output_path: Path) -> None:
    """Drive Ghidra over `binary_path` and write a `schema_version=1`
    JSON dump to `output_path`. Auto-analysis runs with default
    settings; the decompiler is invoked once per non-thunk function.

    The Ghidra project is ephemeral — created in a temp dir, deleted
    on completion. Re-running on the same binary therefore re-analyzes
    from scratch (deterministic but not cheap; a real `.so` can take
    minutes). Caching across runs is a v2 concern.
    """
    try:
        import pyghidra
    except ImportError as e:
        raise RuntimeError(_PYGHIDRA_MISSING_MSG) from e

    if "GHIDRA_INSTALL_DIR" not in os.environ:
        # PyGhidra's launcher tries to auto-detect Ghidra in a few
        # standard places; if it can't, the start() call below raises a
        # cryptic JPype error. Surfacing the env-var requirement up
        # front gives a much clearer message.
        common = ("/opt/ghidra", "/usr/share/ghidra", "/usr/local/ghidra")
        for candidate in common:
            if Path(candidate).is_dir():
                os.environ["GHIDRA_INSTALL_DIR"] = candidate
                break
        else:
            raise RuntimeError(
                "GHIDRA_INSTALL_DIR is not set and Ghidra was not found in "
                f"any of {common}; set GHIDRA_INSTALL_DIR or install Ghidra."
            )

    pyghidra.start()  # idempotent; first call starts the JVM (~5s)

    binary_path = Path(binary_path).resolve()
    output_path = Path(output_path)

    with tempfile.TemporaryDirectory(prefix="quod-pyghidra-proj-") as proj_dir:
        project = pyghidra.open_project(proj_dir, "quod_ingest", create=True)
        try:
            loader = (
                pyghidra.program_loader()
                .source(str(binary_path))
                .project(project)
            )
            with loader.load() as load_results:
                program = load_results.getPrimaryDomainObject()
                pyghidra.analyze(program)
                dump = _build_dump(program, str(binary_path))
        finally:
            project.close()

    with open(output_path, "w") as fh:
        json.dump(dump, fh, indent=2, sort_keys=True)


# ---------- Top-level dump shape ----------


def _build_dump(program, binary_path: str) -> dict[str, Any]:
    from ghidra.app.decompiler import DecompInterface
    from ghidra.util.task import ConsoleTaskMonitor

    monitor = ConsoleTaskMonitor()
    decompiler = DecompInterface()
    decompiler.openProgram(program)
    try:
        functions = []
        fm = program.getFunctionManager()
        fn_iter = fm.getFunctions(True)
        while fn_iter.hasNext():
            fn = fn_iter.next()
            if fn.isExternal() or fn.isThunk():
                # Thunks are linker-emitted dispatch shims (PLT
                # entries, ifunc resolvers, etc.) — they appear as
                # call targets via `BinExternRef`/`BinCallEdge`, not
                # as substantive `BinFunction`s of their own.
                continue
            functions.append(_function_dump(program, fn, decompiler, monitor))

        return {
            "schema_version": SCHEMA_VERSION,
            "binary": _binary_meta(program, binary_path),
            "functions": functions,
            "data": _data_dump(program),
            "externs": _extern_dump(program),
            "type_refs": _type_ref_dump(program, _used_type_names(functions)),
        }
    finally:
        decompiler.dispose()


def _binary_meta(program, binary_path: str) -> dict[str, Any]:
    return {
        "path": binary_path,
        "sha256": _file_sha256(binary_path),
        "arch": str(program.getLanguageID() or ""),
        "format": _file_format(program),
        "build_id": _build_id(program),
    }


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _file_format(program) -> str:
    """Map Ghidra's verbose `getExecutableFormat()` to one of the four
    strings the model's `BinUnit.file_format` enum accepts."""
    fmt = str(program.getExecutableFormat() or "").lower()
    if "elf" in fmt:
        return "elf"
    if "portable executable" in fmt or "pe" in fmt.split():
        return "pe"
    if "mach-o" in fmt or "macho" in fmt:
        return "mach-o"
    return "raw"


def _build_id(program) -> str | None:
    """Extract GNU build-id from the program's metadata. Ghidra surfaces
    it under `Program Information` properties when DWARF or section
    headers expose it; absent on stripped binaries."""
    try:
        opts = program.getOptions("Program Information")
        for key in opts.getOptionNames():
            sk = str(key)
            if "build" in sk.lower() and "id" in sk.lower():
                v = opts.getString(sk, None)
                if v:
                    return str(v)
    except Exception:
        pass
    return None


# ---------- Function dump ----------


def _function_dump(program, function, decompiler, monitor) -> dict[str, Any]:
    decl_file, decl_line = _decl_location(program, function)
    return {
        "address": _hex(function.getEntryPoint()),
        "name_mangled": function.getName(True),
        "name_demangled": function.getName(False),
        "signature": _signature(function),
        "calling_convention": str(function.getCallingConventionName() or ""),
        "decompile": _decompile(decompiler, function, monitor),
        "basic_blocks": _basic_blocks(program, function, monitor),
        "calls": _calls(program, function, monitor),
        "decl_file": decl_file,
        "decl_line": decl_line,
    }


def _decl_location(program, function) -> tuple[str | None, int | None]:
    """Extract DWARF `DW_AT_decl_file` / `DW_AT_decl_line` for a
    function's entry point, via Ghidra's source-file manager.

    `clang -g` populates a `SourceMap` for every instruction it
    associates with a source line. The entry-point's first source-map
    entry is the function's opening line (typically the line of the
    `int foo(...)` signature). Stripped binaries return None for both
    fields; compiler-emitted glue (`_init`, `frame_dummy`) likewise
    has no source-map entry.

    Returns the path as Ghidra recorded it (typically the compile-time
    absolute path); the seeder compares basenames so different working
    directories don't break attribution.
    """
    try:
        sfm = program.getSourceFileManager()
    except Exception:
        return None, None
    if sfm is None:
        return None, None
    try:
        entries = list(sfm.getSourceMapEntries(function.getEntryPoint()))
    except Exception:
        return None, None
    if not entries:
        return None, None
    entry = entries[0]
    sf = entry.getSourceFile()
    return str(sf.getPath()), int(entry.getLineNumber())


def _signature(function) -> dict[str, Any]:
    sig = function.getSignature()
    params = []
    for p in function.getParameters():
        params.append({
            "name": str(p.getName() or ""),
            "type": str(p.getDataType().getName()),
        })
    return {
        "return_type": str(sig.getReturnType().getName()),
        "params": params,
    }


def _decompile(decompiler, function, monitor) -> str:
    """Run the Ghidra decompiler on `function`, return the C-like text.

    60-second per-function timeout matches Ghidra's documented
    recommendation for headless-style use; on real binaries this is
    rarely hit but a pathological function shouldn't stall the whole
    ingest. Empty string when the decompiler refuses or times out —
    the field is optional in the JSON contract."""
    res = decompiler.decompileFunction(function, 60, monitor)
    if res is None or not res.decompileCompleted():
        return ""
    fn = res.getDecompiledFunction()
    return str(fn.getC()) if fn is not None else ""


def _basic_blocks(program, function, monitor) -> list[dict[str, Any]]:
    from ghidra.program.model.block import BasicBlockModel

    bbm = BasicBlockModel(program)
    body = function.getBody()
    blocks: list[dict[str, Any]] = []
    iterator = bbm.getCodeBlocksContaining(body, monitor)
    while iterator.hasNext():
        block = iterator.next()
        succs = []
        dests = block.getDestinations(monitor)
        while dests.hasNext():
            d = dests.next()
            target = d.getDestinationAddress()
            # Skip out-of-function destinations — those surface as
            # `calls` entries instead. We only want intra-function CFG
            # edges on `successors`.
            if not body.contains(target):
                continue
            succs.append({
                "address": _hex(target),
                "kind": _block_edge_kind(d),
            })
        listing = program.getListing()
        instrs = listing.getInstructions(block, True)
        pcode = []
        while instrs.hasNext():
            ins = instrs.next()
            for op in ins.getPcode():
                pcode.append(_pcode_op(op))
        blocks.append({
            "address": _hex(block.getFirstStartAddress()),
            "end": _hex(block.getMaxAddress()),
            "successors": succs,
            "pcode": pcode,
        })
    return blocks


def _block_edge_kind(dest) -> str:
    """Map a Ghidra `CodeBlockReference.getFlowType()` to one of our
    six edge_kind labels. The Ghidra side has more than six categories
    (function calls, terminators, exceptions); we collapse them onto
    the model's enum and let downstream analyses recover detail from
    p-code if they need it."""
    flow = dest.getFlowType()
    if flow is None:
        return "unconditional"
    if flow.isFallthrough():
        return "fallthrough"
    if flow.isConditional() and flow.isJump():
        return "true"
    if flow.isUnConditional() and flow.isJump():
        return "unconditional"
    if flow.isComputed():
        return "indirect"
    if flow.isCall():
        return "call_return"
    return "unconditional"


def _pcode_op(op) -> dict[str, Any]:
    inputs = []
    for vn in op.getInputs():
        inputs.append(_varnode(vn))
    return {
        "opcode": str(op.getMnemonic()),
        "inputs": inputs,
        "output": _varnode(op.getOutput()),
        "instr_address": _hex(op.getSeqnum().getTarget()),
    }


def _varnode(vn) -> dict[str, Any] | None:
    if vn is None:
        return None
    space = str(vn.getAddress().getAddressSpace().getName())
    # Varnode offsets are signed integers — stack-space varnodes
    # routinely have negative offsets ("-8 from the frame pointer").
    # Emit as int rather than hex string so negative values round-trip
    # correctly; instruction addresses (memory addresses) keep hex.
    return {
        "space": space,
        "offset": int(vn.getOffset()),
        "size": int(vn.getSize()),
    }


def _calls(program, function, monitor) -> list[dict[str, Any]]:
    refs = program.getReferenceManager()
    body = function.getBody()
    out: list[dict[str, Any]] = []
    for addr_range in body:
        addr_iter = refs.getReferenceSourceIterator(addr_range.getMinAddress(), True)
        while addr_iter.hasNext():
            from_addr = addr_iter.next()
            if not body.contains(from_addr):
                break
            for ref in refs.getReferencesFrom(from_addr):
                rtype = ref.getReferenceType()
                if not rtype.isCall():
                    continue
                target = ref.getToAddress()
                target_fn = program.getFunctionManager().getFunctionAt(target)
                if target_fn is None:
                    to: dict[str, Any] = {
                        "kind": "external",
                        "address": _hex(target),
                        "name": None,
                    }
                elif target_fn.isExternal() or target_fn.isThunk():
                    to = {
                        "kind": "external",
                        "name": str(target_fn.getName(True)),
                        "address": _hex(target),
                    }
                else:
                    to = {
                        "kind": "internal",
                        "address": _hex(target_fn.getEntryPoint()),
                    }
                kind_str = str(rtype.toString()).upper()
                if rtype.isComputed():
                    call_kind = "indirect"
                elif "THUNK" in kind_str or "TAIL" in kind_str:
                    call_kind = "tail"
                else:
                    call_kind = "direct"
                out.append({
                    "from_block": _hex(from_addr),
                    "instr_address": _hex(from_addr),
                    "to": to,
                    "call_kind": call_kind,
                })
    return out


# ---------- Data items ----------


def _data_dump(program) -> list[dict[str, Any]]:
    listing = program.getListing()
    out: list[dict[str, Any]] = []
    it = listing.getDefinedData(True)
    while it.hasNext():
        data = it.next()
        if data.hasStringValue():
            value = data.getValue()
            out.append({
                "address": _hex(data.getAddress()),
                "kind": "string",
                "value": str(value) if value is not None else "",
            })
            continue
        # Skip non-string defined data for v1 — Ghidra's
        # `getDefinedData` includes import directories, jump tables,
        # and other structural data, much of which the binary frontend
        # has no use for at Layer A. A future export-script flag can
        # opt in.
    return out


# ---------- Externs ----------


def _extern_dump(program) -> list[dict[str, Any]]:
    sm = program.getSymbolTable()
    out: list[dict[str, Any]] = []
    it = sm.getExternalSymbols()
    while it.hasNext():
        s = it.next()
        out.append({
            "name": str(s.getName(True)),
            "address": _hex(s.getAddress()),
        })
    return out


# ---------- Type refs ----------


def _used_type_names(functions: list[dict[str, Any]]) -> set[str]:
    """Collect every type-name string that appears on a function's
    return type or parameter list in this dump. Used by
    `_type_ref_dump` to filter Ghidra's DataType universe down to
    types that are actually referenced — without this, the dump emits
    every type in `generic_clib_64.gdt` (~150 types for a toy `.so`)
    even when nothing in the binary uses them."""
    names: set[str] = set()
    for fn in functions:
        sig = fn.get("signature") or {}
        rt = sig.get("return_type")
        if rt:
            names.add(rt)
        for p in sig.get("params") or []:
            t = p.get("type")
            if t:
                names.add(t)
    return names


def _type_ref_dump(program, used_type_names: set[str]) -> list[dict[str, Any]]:
    """Dump only the DataTypes whose name appears on a function
    signature in this dump (return type or param). `BinTypeRef` is
    opaque-by-contract (see `quod.model.layer_a_bin.BinTypeRef`); the
    point of these entries is to record size + structural hash for
    types downstream consumers might reason about, not to enumerate
    Ghidra's full type catalog.

    A v2 could follow composite/typedef references transitively
    (struct → field types, typedef → base) for higher recall on
    structural-hint providers. v1 keeps it simple: name match against
    what we already emit on signatures."""
    out: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()
    dtm = program.getDataTypeManager()
    it = dtm.getAllDataTypes()
    while it.hasNext():
        dt = it.next()
        name = str(dt.getName())
        if name not in used_type_names:
            continue
        try:
            size = int(dt.getLength())
        except Exception:
            size = 0
        key = (name, size)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append({
            "name": name,
            "size": size,
            # v1 hash: name + size. A future export can compute a real
            # structural hash over Ghidra's DataType representation.
            "structural_hash": "ghidra:%s:%d" % (name, size),
        })
    return out


# ---------- Helpers ----------


def _hex(addr) -> str | None:
    """Render a Ghidra `Address` (or anything int-like) as `0x` hex."""
    if addr is None:
        return None
    try:
        return "0x%x" % int(addr.getOffset())
    except AttributeError:
        return "0x%x" % int(addr)
