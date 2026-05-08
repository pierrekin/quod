"""Cross-cluster pretty-printer.

Renders programs, functions, types, expressions, statements, claims,
justifications, and layer-A C subtrees. The printer pattern-matches
across full discriminated unions, so its dispatchers necessarily span
every cluster.
"""

from __future__ import annotations

from collections.abc import Callable

from quod.model.base import _Node
from quod.model.claims import Claim
from quod.model.expressions import (
    BinOp,
    Call,
    CharLit,
    EnumInit,
    FieldRead,
    IntLit,
    Load,
    LoadField,
    LocalRef,
    Cast,
    FloatLit,
    FNeg,
    NullPtr,
    ParamRef,
    PtrOffset,
    ShortCircuitAnd,
    ShortCircuitOr,
    SizeOf,
    StringRef,
    StructInit,
    TryExpr,
)
from quod.model.justifications import (
    BinaryProvenance,
    DecompileLift,
    DerivedJustification,
    FamilyLowering,
    Justification,
    LiftEquivalence,
    ManualJustification,
    Z3Justification,
)
from quod.model.layer_a import (
    CAddressOf,
    CArraySubscript,
    CAssign,
    CBinOp,
    CCall,
    CDeref,
    CDerefStore,
    CEnumConstRef,
    CExprStmt,
    CFn,
    CFor,
    CIf,
    CCast,
    CFloatLit,
    CIntLit,
    CNamedType,
    CPointerType,
    CReturn,
    CStringLit,
    CSubscriptStore,
    CUnit,
    CVarDecl,
    CVarRef,
    CWhile,
)
from quod.model.layer_a_bin import BinFunction, BinUnit
from quod.model.layer_b import CStyleFor
from quod.model.program import Program
from quod.model.relations import Equivalence, Import
from quod.model.statements import (
    Assign,
    Block,
    ExprStmt,
    FieldSet,
    For,
    If,
    Let,
    Match,
    Return,
    ReturnExpr,
    Store,
    StoreField,
    Unreachable,
    While,
    WithArena,
)
from quod.model.top_level import EnumDef, Function, StructDef, TypeParam
from quod.model.types import (
    EnumType,
    I1Type,
    I8PtrType,
    I8Type,
    I16Type,
    F32Type,
    F64Type,
    I32Type,
    I64Type,
    IsizeType,
    bits_to_python_float,
    SelfType,
    StructType,
    TypeParamRef,
    U8Type,
    U16Type,
    U32Type,
    U64Type,
    UsizeType,
    VoidType,
)


# A "label" is an optional prefix the formatter inserts before each addressable
# node. The default is empty; the CLI passes a function returning [hashprefix]
# so each node prints with its addressable identity inline.
NodeLabel = Callable[[_Node], str]
_NO_LABEL: NodeLabel = lambda _node: ""


def _format_import(imp: Import) -> str:
    s = imp.module
    if imp.wire:
        bindings = ", ".join(f"{w.name}={_format_type(w.type)}" for w in imp.wire)
        s += f" wire {bindings}"
    return s


def _format_type_param(tp: TypeParam) -> str:
    s = tp.name
    if tp.bound:
        s += f": {tp.bound}"
    return s


def _format_type_params(tps: tuple) -> str:
    if not tps:
        return ""
    return "<" + ", ".join(_format_type_param(tp) for tp in tps) + ">"


def format_program(program: Program, *, label: NodeLabel = _NO_LABEL) -> str:
    lines: list[str] = ["program {"]
    if program.source_units:
        lines.append("  source_units:")
        for unit in program.source_units:
            lines.extend("    " + line for line in _format_c_unit(unit, label=label).splitlines())
    if program.binary_units:
        lines.append("  binary_units:")
        for unit in program.binary_units:
            lines.extend("    " + line for line in _format_bin_unit(unit, label=label).splitlines())
    if program.wirables:
        lines.append("  wirables:")
        for w in program.wirables:
            lines.append(f"    {_format_type_param(w)}")
    if program.imports:
        lines.append("  imports:")
        for imp in program.imports:
            lines.append(f"    {_format_import(imp)}")
    if program.constants:
        lines.append("  constants:")
        for c in program.constants:
            lines.append(f"    {label(c)}{c.name} = {c.value!r}")
    if program.structs:
        lines.append("  structs:")
        for sd in program.structs:
            lines.append(f"    {label(sd)}{format_struct_def(sd)}")
    if program.enums:
        lines.append("  enums:")
        for ed in program.enums:
            lines.extend("    " + line for line in format_enum_def(ed, label=label).splitlines())
    if program.externs:
        lines.append("  externs:")
        for ext in program.externs:
            sig_parts = [_format_type(t) for t in ext.effective_param_types()]
            if ext.varargs:
                sig_parts.append("...")
            sig = ", ".join(sig_parts)
            ret = _format_type(ext.return_type)
            lines.append(f"    {label(ext)}extern {ext.name}({sig}) -> {ret}")
    if program.structured_functions:
        lines.append("  structured_functions:")
        for fn in program.structured_functions:
            lines.extend("    " + line for line in format_function(fn, label=label).splitlines())
    if program.functions:
        lines.append("  functions:")
        for fn in program.functions:
            lines.extend("    " + line for line in format_function(fn, label=label).splitlines())
    if program.edges:
        lines.append("  edges:")
        for e in program.edges:
            lines.append(f"    {e.source} -> {e.target}")
    if program.equivalences:
        lines.append("  equivalences:")
        for eq in program.equivalences:
            lines.append(f"    {eq.a_node_id} ~ {eq.b_node_id}{format_equivalence_metadata(eq)}")
    if (
        not program.constants and not program.functions
        and not program.externs and not program.structs and not program.enums
        and not program.imports and not program.edges
        and not program.equivalences and not program.source_units
        and not program.binary_units
        and not program.structured_functions
    ):
        lines.append("  (empty)")
    lines.append("}")
    return "\n".join(lines)


def _format_c_unit(unit: "CUnit", *, label: NodeLabel) -> str:
    """Render a layer-A C translation unit. Surface form approximates C
    source so the unit reads naturally in `quod show` output; no
    semantic processing — just structural pretty-printing."""
    head = f'{label(unit)}c_unit "{unit.source_path}" {{'
    lines = [head]
    for fn in unit.functions:
        lines.extend("  " + line for line in format_c_fn(fn, label=label).splitlines())
    lines.append("}")
    return "\n".join(lines)


def format_c_fn(fn: "CFn", *, label: NodeLabel = _NO_LABEL) -> str:
    """Render a layer-A `CFn` as C-flavored text. Used by
    `format_program` for the `source_units` section and by the CLI's
    `quod fn show --source` for single-function rendering."""
    params = ", ".join(f"{_format_c_type(p.type)} {p.name}" for p in fn.params)
    head = f"{label(fn)}{_format_c_type(fn.return_type)} {fn.name}({params}) {{"
    lines = [head]
    for s in fn.body:
        lines.append(_format_c_stmt(s, indent=2, label=label))
    lines.append("}")
    return "\n".join(lines)


def _format_bin_unit(unit: "BinUnit", *, label: NodeLabel) -> str:
    """Render a layer-A binary translation unit. Surface form is
    structural metadata + per-function summaries — Ghidra output is not
    quod source, so coloring it with quod's vocabulary would lie about
    what it is. The decompile text and p-code are rendered as opaque
    quoted blocks by `format_bin_fn`."""
    head = f'{label(unit)}bin_unit "{unit.path}" ({unit.arch}, {unit.file_format}) {{'
    lines = [head]
    lines.append(f"  sha256: {unit.sha256}")
    if unit.build_id:
        lines.append(f"  build_id: {unit.build_id}")
    if unit.extern_refs:
        lines.append("  externs:")
        for ref in unit.extern_refs:
            tail = f" -> {ref.linked_extern_name}" if ref.linked_extern_name else ""
            lines.append(f"    {ref.symbol}{tail}")
    for fn in unit.functions:
        lines.extend("  " + line for line in format_bin_fn(fn, label=label).splitlines())
    lines.append("}")
    return "\n".join(lines)


def _clean_decompile_text(text: str) -> str:
    """Display-only filter for Ghidra decompile output.

    - Strip `/* ... */` block comments (Ghidra emits these as
      `/* WARNING: Removing unreachable block ... */`,
      `/* Subroutine type: ... */`, etc.). They're informational
      diagnostics, not part of the decompiled C.
    - Trim leading and trailing blank-only lines (Ghidra prefixes the
      function header with a blank line and pads the trailer).

    Does **not** modify the underlying `BinFunction.decompile_text`
    field; layer-A's preserve-verbatim rule still holds. Callers that
    need the unaltered text pass `raw_decompile=True` to
    `format_bin_fn`.
    """
    import re
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def format_bin_fn(
    fn: "BinFunction",
    *,
    label: NodeLabel = _NO_LABEL,
    raw_decompile: bool = False,
    detail: bool = False,
) -> str:
    """Render a layer-A `BinFunction` as a structural summary plus the
    decompile text. Used by `_format_bin_unit` for the `binary_units`
    section and by the CLI's `quod fn show --binary` for single-
    function rendering.

    `BinFunction.decompile_text` is opaque by contract — the underlying
    string in the node is preserved verbatim. By default this renderer
    elides Ghidra's `/* ... */` block comments (warnings about
    unreachable blocks, type-recovery notes) and trims surrounding
    blank lines for readability; pass `raw_decompile=True` to keep the
    full decompile text untouched.

    With `detail=False` (default), per-block pcode is summarized as
    `addr [n ops]` — the structural skeleton without the noisy
    opcode listing. With `detail=True`, each block also lists every
    opcode (`addr [n ops: INT_LESS, INT_SBORROW, ...]`). The
    structured pcode is always available via `--json` regardless of
    this flag."""
    params = ", ".join(f"{p.type_name} {p.name}" for p in fn.params)
    head = (
        f"{label(fn)}bin.fn 0x{fn.address:x} "
        f"{fn.return_type_name} {fn.demangled_name}({params}) "
        f"[{fn.calling_convention}] {{"
    )
    lines = [head]
    if fn.basic_blocks:
        lines.append("  blocks:")
        for bb in fn.basic_blocks:
            head_bb = (
                f"    0x{bb.start_address:x}-0x{bb.end_address:x} "
                f"[{len(bb.pcode_ops)} ops"
            )
            if detail:
                opcodes = ", ".join(op.opcode for op in bb.pcode_ops)
                lines.append(f"{head_bb}: {opcodes}]")
            else:
                lines.append(f"{head_bb}]")
    if fn.call_edges:
        lines.append("  calls:")
        for c in fn.call_edges:
            lines.append(
                f"    0x{c.instruction_address:x} -> {c.callee_id or '<unresolved>'} "
                f"({c.call_kind})"
            )
    decompile_text = (
        fn.decompile_text if raw_decompile else _clean_decompile_text(fn.decompile_text)
    )
    if decompile_text:
        lines.append("  decompile:")
        for raw in decompile_text.splitlines() or [""]:
            lines.append(f"    {raw}")
    lines.append("}")
    return "\n".join(lines)


def _format_c_type(t) -> str:
    match t:
        case CNamedType(name=n):
            return n
        case CPointerType(pointee=p):
            return f"{_format_c_type(p)}*"
    raise ValueError(f"unhandled c.* type: {t!r}")


def _format_c_stmt(stmt, indent: int, *, label: NodeLabel) -> str:
    pad = " " * indent
    prefix = label(stmt)
    match stmt:
        case CVarDecl(type=ty, name=n, init=None):
            return f"{pad}{prefix}{_format_c_type(ty)} {n};"
        case CVarDecl(type=ty, name=n, init=init):
            return f"{pad}{prefix}{_format_c_type(ty)} {n} = {_format_c_expr(init)};"
        case CAssign(target=t, value=v):
            return f"{pad}{prefix}{t} = {_format_c_expr(v)};"
        case CReturn(value=None):
            return f"{pad}{prefix}return;"
        case CReturn(value=v):
            return f"{pad}{prefix}return {_format_c_expr(v)};"
        case CFor(init=init, cond=cond, inc=inc, body=body):
            init_s = _format_c_for_part(init) if init is not None else ""
            cond_s = _format_c_expr(cond) if cond is not None else ""
            inc_s = _format_c_for_part(inc) if inc is not None else ""
            head = f"{pad}{prefix}for ({init_s}; {cond_s}; {inc_s}) {{"
            body_lines = "\n".join(_format_c_stmt(s, indent + 2, label=label) for s in body)
            return f"{head}\n{body_lines}\n{pad}}}"
        case CIf(cond=cond, then_body=tb, else_body=eb):
            head = f"{pad}{prefix}if ({_format_c_expr(cond)}) {{"
            then_lines = "\n".join(_format_c_stmt(s, indent + 2, label=label) for s in tb)
            if not eb:
                return f"{head}\n{then_lines}\n{pad}}}"
            else_lines = "\n".join(_format_c_stmt(s, indent + 2, label=label) for s in eb)
            return f"{head}\n{then_lines}\n{pad}}} else {{\n{else_lines}\n{pad}}}"
        case CWhile(cond=cond, body=body):
            head = f"{pad}{prefix}while ({_format_c_expr(cond)}) {{"
            body_lines = "\n".join(_format_c_stmt(s, indent + 2, label=label) for s in body)
            return f"{head}\n{body_lines}\n{pad}}}"
        case CExprStmt(value=v):
            return f"{pad}{prefix}{_format_c_expr(v)};"
        case CDerefStore(operand=op, value=v):
            return f"{pad}{prefix}*{_format_c_expr(op)} = {_format_c_expr(v)};"
        case CSubscriptStore(base=b, index=i, value=v):
            return f"{pad}{prefix}{_format_c_expr(b)}[{_format_c_expr(i)}] = {_format_c_expr(v)};"
    raise ValueError(f"unhandled c.* statement: {stmt!r}")


def _format_c_for_part(s) -> str:
    """Render a CForInit (CVarDecl or CAssign) inline — no trailing
    semicolon, since the for-header supplies its own."""
    match s:
        case CVarDecl(type=ty, name=n, init=None):
            return f"{_format_c_type(ty)} {n}"
        case CVarDecl(type=ty, name=n, init=init):
            return f"{_format_c_type(ty)} {n} = {_format_c_expr(init)}"
        case CAssign(target=t, value=v):
            return f"{t} = {_format_c_expr(v)}"
    raise ValueError(f"unhandled c.* for-part: {s!r}")


def _format_c_expr(e) -> str:
    match e:
        case CIntLit(value=v):
            return str(v)
        case CFloatLit(type=t, bits=b):
            return _format_float_bits(b, t)
        case CStringLit(value=v):
            return repr(v)
        case CVarRef(name=n):
            return n
        case CEnumConstRef(name=n, value=v):
            return f"{n}={v}"
        case CBinOp(op=op, lhs=l, rhs=r):
            return f"({_format_c_expr(l)} {op} {_format_c_expr(r)})"
        case CCall(callee=callee, args=args):
            args_s = ", ".join(_format_c_expr(a) for a in args)
            return f"{callee}({args_s})"
        case CArraySubscript(base=b, index=i):
            return f"{_format_c_expr(b)}[{_format_c_expr(i)}]"
        case CAddressOf(target=t):
            return f"&{_format_c_expr(t)}"
        case CDeref(operand=op):
            return f"*{_format_c_expr(op)}"
        case CCast(target_type=t, value=v):
            return f"({_format_c_type(t)}){_format_c_expr(v)}"
    raise ValueError(f"unhandled c.* expression: {e!r}")


def format_equivalence_metadata(eq: "Equivalence") -> str:
    """Render the regime/enforcement/justification metadata trailing an
    equivalence, mirroring `format_claim_metadata` for plain claims.
    Returns " {…}" with a leading space when any non-default field is
    present, or "" when everything is at default."""
    bits: list[str] = []
    if eq.regime != "axiom":
        bits.append(f"regime={eq.regime}")
    if eq.enforcement != "trust":
        bits.append(f"enforcement={eq.enforcement}")
    if eq.justification is not None:
        bits.append(f"justification={_format_justification(eq.justification)}")
    return " {" + ", ".join(bits) + "}" if bits else ""


def format_struct_def(sd: StructDef) -> str:
    tp = _format_type_params(sd.type_params)
    body = ", ".join(f"{f.name}: {_format_type(f.type)}" for f in sd.fields)
    return f"struct {sd.name}{tp} {{ {body} }}"


def format_enum_def(ed: EnumDef, *, label: NodeLabel = _NO_LABEL) -> str:
    lines = [f"{label(ed)}enum {ed.name} {{"]
    for v in ed.variants:
        if v.fields:
            args = ", ".join(f"{f.name}: {_format_type(f.type)}" for f in v.fields)
            lines.append(f"  {label(v)}{v.name}({args})")
        else:
            lines.append(f"  {label(v)}{v.name}")
    lines.append("}")
    return "\n".join(lines)


def format_function(fn: Function, *, label: NodeLabel = _NO_LABEL) -> str:
    tp = _format_type_params(fn.type_params)
    sig_params = ", ".join(f"{p.name}: {_format_type(p.type)}" for p in fn.params)
    header = f"{label(fn)}{fn.name}{tp}({sig_params}) -> {_format_type(fn.return_type)}"
    if fn.claims:
        header += "  [claims: " + ", ".join(format_claim(c) for c in fn.claims) + "]"
    lines: list[str] = []
    for note in fn.notes:
        lines.append(f"// {note}")
    lines.append(header + " {")
    for s in fn.body.stmts:
        lines.append(_format_stmt(s, indent=2, label=label))
    lines.append("}")
    return "\n".join(lines)


def _format_type(t) -> str:
    match t:
        case I1Type():
            return "i1"
        case I8Type():
            return "i8"
        case I16Type():
            return "i16"
        case I32Type():
            return "i32"
        case I64Type():
            return "i64"
        case U8Type():
            return "u8"
        case U16Type():
            return "u16"
        case U32Type():
            return "u32"
        case U64Type():
            return "u64"
        case IsizeType():
            return "isize"
        case UsizeType():
            return "usize"
        case F32Type():
            return "f32"
        case F64Type():
            return "f64"
        case I8PtrType():
            return "i8*"
        case StructType(name=n, type_args=ta) if ta:
            args = ", ".join(_format_type(a) for a in ta)
            return f"{n}<{args}>"
        case StructType(name=n):
            return n
        case EnumType(name=n, type_args=ta) if ta:
            args = ", ".join(_format_type(a) for a in ta)
            return f"{n}<{args}>"
        case EnumType(name=n):
            return n
        case TypeParamRef(name=n):
            return n
        case SelfType():
            return "Self"
        case VoidType():
            return "void"
    raise ValueError(f"unhandled type: {t!r}")


def format_claim_metadata(c: Claim) -> str:
    """Return ` {regime,enforcement,justification}` if any field is non-default, else ''."""
    bits: list[str] = []
    if c.regime != "axiom":
        bits.append(f"regime={c.regime}")
    if c.enforcement != "trust":
        bits.append(f"enforcement={c.enforcement}")
    if c.justification is not None:
        bits.append(f"justification={_format_justification(c.justification)}")
    return " {" + ", ".join(bits) + "}" if bits else ""


def _format_justification(j: Justification) -> str:
    match j:
        case Z3Justification(artifact_path=p, artifact_hash=h):
            return f"z3({p}@{h[:12]})"
        case ManualJustification(signed_by=s):
            return f"manual(signed_by={s!r})"
        case DerivedJustification(analysis=a, inputs=i):
            return f"derived({a}, {len(i)} input(s))"
        case LiftEquivalence(artifact_path=p, artifact_hash=h):
            return f"lift_equivalence({p}@{h[:12]})"
        case FamilyLowering(rule_name=r, artifact_hash=h):
            tail = f"@{h[:12]}" if h else ""
            return f"family_lowering({r}{tail})"
        case BinaryProvenance(binary_symbol=sym, source_evidence=ev, binary_sha256=h):
            return f"binary_provenance({sym}, evidence={ev}, sha256={h[:12]})"
        case DecompileLift(decompile_text_sha256=h):
            return f"decompile_lift(sha256={h[:12]})"
    raise ValueError(f"unhandled justification: {j!r}")


def _format_stmt(stmt, indent: int, *, label: NodeLabel) -> str:
    pad = " " * indent
    prefix = label(stmt)
    match stmt:
        case ReturnExpr(value=expr):
            return f"{pad}{prefix}return {_format_expr(expr)}"
        case Return():
            return f"{pad}{prefix}return"
        case Unreachable():
            return f"{pad}{prefix}unreachable"
        case If(cond=cond, then_body=t_body, else_body=e_body):
            then_lines = "\n".join(_format_stmt(s, indent + 2, label=label) for s in t_body.stmts)
            head = f"{pad}{prefix}if ({_format_expr(cond)}) {{"
            if not e_body.stmts:
                return f"{head}\n{then_lines}\n{pad}}}"
            else_lines = "\n".join(_format_stmt(s, indent + 2, label=label) for s in e_body.stmts)
            return f"{head}\n{then_lines}\n{pad}}} else {{\n{else_lines}\n{pad}}}"
        case Let(name=n, type=ty, init=init):
            return f"{pad}{prefix}let {n}: {_format_type(ty)} = {_format_expr(init)}"
        case Assign(name=n, value=v):
            return f"{pad}{prefix}{n} = {_format_expr(v)}"
        case FieldSet(local=loc, name=fname, value=v):
            return f"{pad}{prefix}{loc}.{fname} = {_format_expr(v)}"
        case Store(ptr=p, value=v):
            return f"{pad}{prefix}store({_format_expr(p)}, {_format_expr(v)})"
        case StoreField(ptr=p, struct_type=tname, name=fname, value=v):
            return f"{pad}{prefix}{_format_expr(p)}->{tname}.{fname} = {_format_expr(v)}"
        case While(cond=cond, body=body):
            body_lines = "\n".join(_format_stmt(s, indent + 2, label=label) for s in body.stmts)
            return f"{pad}{prefix}while ({_format_expr(cond)}) {{\n{body_lines}\n{pad}}}"
        case For(var=var, type=ty, lo=lo, hi=hi, body=body):
            body_lines = "\n".join(_format_stmt(s, indent + 2, label=label) for s in body.stmts)
            return (
                f"{pad}{prefix}for {var}: {_format_type(ty)} in "
                f"{_format_expr(lo)}..{_format_expr(hi)} {{\n"
                f"{body_lines}\n{pad}}}"
            )
        case ExprStmt(value=v):
            return f"{pad}{prefix}{_format_expr(v)}"
        case WithArena(name=n, capacity=cap, body=body):
            body_lines = "\n".join(_format_stmt(s, indent + 2, label=label) for s in body.stmts)
            return (
                f"{pad}{prefix}with_arena {n} = arena_new({_format_expr(cap)}) {{\n"
                f"{body_lines}\n{pad}}}"
            )
        case Match(scrutinee=scrut, arms=arms):
            arm_lines = []
            for arm in arms:
                arm_pad = " " * (indent + 2)
                inner_pad = " " * (indent + 4)
                if arm.bindings:
                    head = f"{arm_pad}{arm.variant}({', '.join(arm.bindings)}) => {{"
                else:
                    head = f"{arm_pad}{arm.variant} => {{"
                arm_lines.append(head)
                for s in arm.body.stmts:
                    arm_lines.append(_format_stmt(s, indent + 4, label=label))
                arm_lines.append(f"{arm_pad}}}")
            arms_text = "\n".join(arm_lines)
            return f"{pad}{prefix}match {_format_expr(scrut)} {{\n{arms_text}\n{pad}}}"
        case CStyleFor(init=init, cond=cond, inc=inc, body=body):
            init_s = _format_stmt(init, indent=0, label=label).strip().rstrip(";") if init is not None else ""
            cond_s = _format_expr(cond) if cond is not None else ""
            inc_s = _format_stmt(inc, indent=0, label=label).strip().rstrip(";") if inc is not None else ""
            inner = body if isinstance(body, Block) else body.block
            body_lines = "\n".join(_format_stmt(s, indent + 2, label=label) for s in inner.stmts)
            return (
                f"{pad}{prefix}c.for_general ({init_s}; {cond_s}; {inc_s}) {{\n"
                f"{body_lines}\n{pad}}}"
            )
    raise ValueError(f"unhandled stmt: {stmt!r}")


_BINOP_SYMBOL = {
    "add": "+", "sub": "-", "mul": "*", "sdiv": "/", "udiv": "/u", "srem": "%",
    "urem": "%u",
    "slt": "<", "sle": "<=", "sgt": ">", "sge": ">=", "eq": "==", "ne": "!=",
    "ult": "<u", "ule": "<=u", "ugt": ">u", "uge": ">=u",
    "or": "|", "and": "&", "xor": "^",
    "shl": "<<", "ashr": ">>", "lshr": ">>l",
    "fadd": "+f", "fsub": "-f", "fmul": "*f", "fdiv": "/f", "frem": "%f",
    "feq": "==f", "fne": "!=f",
    "flt": "<f", "fle": "<=f", "fgt": ">f", "fge": ">=f",
}


def _format_float_bits(bits: int, t) -> str:
    """Human-readable rendering of a `FloatLit`/`CFloatLit` bit
    pattern. Special values render as `+inf`/`-inf`/`nan`/`-nan`;
    finite values use Python's repr (shortest decimal that round-trips
    at the type's precision)."""
    f = bits_to_python_float(bits, t)
    import math
    if math.isnan(f):
        sign_bit = 1 << (31 if isinstance(t, F32Type) else 63)
        return "-nan" if (bits & sign_bit) else "nan"
    if math.isinf(f):
        return "-inf" if f < 0 else "+inf"
    return repr(f)


def _format_expr(expr) -> str:
    match expr:
        case IntLit(value=v):
            return str(v)
        case FloatLit(type=t, bits=b):
            return _format_float_bits(b, t)
        case FNeg(operand=op):
            return f"fneg({_format_expr(op)})"
        case ParamRef(name=n):
            return n
        case LocalRef(name=n):
            return n
        case BinOp(op=op, lhs=l, rhs=r):
            return f"({_format_expr(l)} {_BINOP_SYMBOL[op]} {_format_expr(r)})"
        case ShortCircuitOr(lhs=l, rhs=r):
            return f"({_format_expr(l)} || {_format_expr(r)})"
        case ShortCircuitAnd(lhs=l, rhs=r):
            return f"({_format_expr(l)} && {_format_expr(r)})"
        case Call(function=fn_name, args=args):
            return f"{fn_name}({', '.join(_format_expr(a) for a in args)})"
        case StringRef(name=n):
            return f"&{n}"
        case FieldRead(value=inner, name=fname):
            return f"{_format_expr(inner)}.{fname}"
        case LoadField(ptr=p, struct_type=tname, name=fname):
            return f"{_format_expr(p)}->{tname}.{fname}"
        case StructInit(type=tname, fields=field_inits):
            inner = ", ".join(f"{fi.name}: {_format_expr(fi.value)}" for fi in field_inits)
            return f"{tname} {{ {inner} }}"
        case PtrOffset(base=b, offset=o):
            return f"({_format_expr(b)} + {_format_expr(o)})"
        case Cast(value=v, target_type=t):
            return f"cast({_format_expr(v)} to {_format_type(t)})"
        case Load(ptr=p, type=t):
            return f"load[{_format_type(t)}]({_format_expr(p)})"
        case NullPtr():
            return "null"
        case CharLit(value=v):
            return repr(v)
        case SizeOf(type=t):
            return f"sizeof[{_format_type(t)}]"
        case TryExpr(value=v):
            return f"{_format_expr(v)}?"
        case EnumInit(enum=ename, variant=vname, fields=field_inits):
            if field_inits:
                inner = ", ".join(f"{fi.name}: {_format_expr(fi.value)}" for fi in field_inits)
                return f"{ename}::{vname}({inner})"
            return f"{ename}::{vname}"
    raise ValueError(f"unhandled expr: {expr!r}")
