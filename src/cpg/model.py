"""CPG data model + pretty-printer + immutable update helpers.

The graph is the asset. Nodes are frozen Pydantic models; mutators return
new Programs via model_copy. Addressing is by name *or* content-hash prefix
(the latter implemented in cpg.hashing / cpg.editor).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer


# ---------- Base ----------

class _Node(BaseModel):
    # strict=True: no silent coercion. frozen=True: graph is read-only;
    # mutators must build new instances via model_copy.
    model_config = ConfigDict(strict=True, frozen=True)


# ---------- Constants ----------

class StringConstant(_Node):
    name: str
    value: str


# ---------- Expressions ----------

class IntLit(_Node):
    kind: Literal["int_lit"] = "int_lit"
    value: int


class ParamRef(_Node):
    kind: Literal["param_ref"] = "param_ref"
    name: str


class BinOp(_Node):
    kind: Literal["binop"] = "binop"
    op: Literal["add", "slt"]  # slt = signed less-than (yields i1)
    lhs: "Expr"
    rhs: "Expr"


class Call(_Node):
    """Call a user-defined function in the same Program. All i32 today."""
    kind: Literal["call"] = "call"
    function: str  # name of a Function in the Program
    args: tuple["Expr", ...] = ()


Expr = Annotated[Union[IntLit, ParamRef, BinOp, Call], Field(discriminator="kind")]


# ---------- Statements ----------

class CallPuts(_Node):
    kind: Literal["call_puts"] = "call_puts"
    target: str  # name of a StringConstant


class ReturnInt(_Node):
    """Return a constant integer. Shorthand kept for hello-world brevity."""
    kind: Literal["return_int"] = "return_int"
    value: int


class ReturnExpr(_Node):
    kind: Literal["return_expr"] = "return_expr"
    value: Expr


class If(_Node):
    kind: Literal["if"] = "if"
    cond: Expr  # must lower to i1
    then_body: tuple["Statement", ...]
    else_body: tuple["Statement", ...]


Statement = Annotated[
    Union[CallPuts, ReturnInt, ReturnExpr, If],
    Field(discriminator="kind"),
]


# ---------- Justifications ----------

# Polymorphic evidence channel attached to a claim. The kind discriminator
# tells you what flavor of evidence is on offer; the regime field on the
# claim is a coarse epistemic label (loosely correlated, not enforced).
#
#   z3        — external proof in SMT-LIB; verifiable by re-running Z3
#               or, in MVP2, just by re-hashing the artifact
#   manual    — a human signed off; no machine-checkable evidence
#   derived   — produced by an analysis pass; reproducible from `inputs`
#               (content-hash refs to the graph nodes the analysis read)


class Z3Justification(_Node):
    kind: Literal["z3"] = "z3"
    artifact_path: str
    artifact_hash: str
    note: str | None = None


class ManualJustification(_Node):
    kind: Literal["manual"] = "manual"
    signed_by: str
    rationale: str


class DerivedJustification(_Node):
    kind: Literal["derived"] = "derived"
    analysis: str                       # name of the analysis pass
    inputs: tuple[str, ...] = ()        # content hashes of nodes the pass read
    note: str | None = None


Justification = Annotated[
    Union[Z3Justification, ManualJustification, DerivedJustification],
    Field(discriminator="kind"),
]


# ---------- Claims ----------

# Epistemic source of a claim (who/what is making the assertion):
#   axiom   = the programmer asserts it (typically: no justification, or manual)
#   witness = a proof was produced out-of-band (typically: z3/coq/lean/...)
#   lattice = derived by an analysis pass (typically: derived)
Regime = Literal["axiom", "witness", "lattice"]

# Enforcement: do we trust the source named by `regime`, or verify at runtime?
#   trust  = lowered to llvm.assume; falsity is undefined behaviour
#   verify = lowered to a runtime branch + abort; falsity aborts the program
Enforcement = Literal["trust", "verify"]


class _Claim(_Node):
    """Common metadata carried by every claim.

    Defaults: a programmer assertion (regime=axiom), trusted unconditionally
    (enforcement=trust), without a justification (justification=None).
    """
    regime: Regime = "axiom"
    enforcement: Enforcement = "trust"
    justification: Justification | None = None

    # Drop metadata fields from serialized JSON when they're at default. This
    # keeps program.json compact for the common case while preserving the
    # discriminator `kind` (which is also default-valued but must round-trip).
    @model_serializer(mode="wrap")
    def _drop_default_metadata(self, handler, info):
        data = handler(self)
        if self.regime == "axiom":
            data.pop("regime", None)
        if self.enforcement == "trust":
            data.pop("enforcement", None)
        if self.justification is None:
            data.pop("justification", None)
        return data


class NonNegativeClaim(_Claim):
    """Asserts param >= 0. Subsumed by IntRangeClaim(min=0); kept as a convenience."""
    kind: Literal["non_negative"] = "non_negative"
    param: str


class IntRangeClaim(_Claim):
    """Asserts `min <= param <= max` (either bound optional).

    Lowered to one or two predicates (llvm.assume or runtime branch, per enforcement).
    min=None / max=None means unbounded on that side.
    """
    kind: Literal["int_range"] = "int_range"
    param: str
    min: int | None = None
    max: int | None = None


class ReturnInRangeClaim(_Claim):
    """Asserts the function's return value is in [min, max] (either bound optional).

    Function-scoped, not param-scoped — there's no `param` field. Today this
    is metadata only: the claim is provable via Z3 (cpg prove) and verifiable
    (cpg verify-claims) but not yet exploited by the LLVM lowering pass.
    """
    kind: Literal["return_in_range"] = "return_in_range"
    min: int | None = None
    max: int | None = None


Claim = Annotated[
    Union[NonNegativeClaim, IntRangeClaim, ReturnInRangeClaim],
    Field(discriminator="kind"),
]


CLAIM_KINDS: tuple[str, ...] = ("non_negative", "int_range", "return_in_range")
PARAM_CLAIM_KINDS: tuple[str, ...] = ("non_negative", "int_range")
RETURN_CLAIM_KINDS: tuple[str, ...] = ("return_in_range",)


def claim_param(claim: Claim) -> str | None:
    """The parameter a claim targets, or None for function-scoped (return-value) claims."""
    match claim:
        case NonNegativeClaim(param=p) | IntRangeClaim(param=p):
            return p
        case ReturnInRangeClaim():
            return None
    raise ValueError(f"unhandled claim: {claim!r}")


def function_callees(fn: "Function") -> tuple[str, ...]:
    """Names of user functions called from fn's body, deduplicated, first-seen order.

    Walks expressions inside statements (Call can hide in ReturnExpr, If.cond,
    BinOp arms, and another Call's args). Excludes `puts` (CallPuts is the
    extern-print path, not a user-function edge).
    """
    seen: dict[str, None] = {}

    def visit_expr(e) -> None:
        match e:
            case Call(function=name, args=args):
                seen.setdefault(name, None)
                for a in args:
                    visit_expr(a)
            case BinOp(lhs=l, rhs=r):
                visit_expr(l)
                visit_expr(r)
            case _:
                pass

    def visit_stmt(s) -> None:
        match s:
            case ReturnExpr(value=expr):
                visit_expr(expr)
            case If(cond=cond, then_body=t_body, else_body=e_body):
                visit_expr(cond)
                for x in t_body:
                    visit_stmt(x)
                for x in e_body:
                    visit_stmt(x)
            case _:
                pass

    for stmt in fn.body:
        visit_stmt(stmt)
    return tuple(seen)


# ---------- Top-level ----------

class Function(_Node):
    name: str
    params: tuple[str, ...] = ()      # all i32 in this round
    body: tuple[Statement, ...]
    claims: tuple[Claim, ...] = ()


class _ProgramBase(_Node):
    """Shared shape for Program and InputProgram."""
    constants: tuple[StringConstant, ...] = ()
    functions: tuple[Function, ...] = ()


class Program(_ProgramBase):
    """The fully-elaborated graph: stored claims + any derived (lattice) claims.

    Permissive: any regime is allowed in fn.claims. This is what `lower()`
    consumes and what editor mutators return.
    """


class InputProgram(_ProgramBase):
    """The graph as authored. Only stored claims (axiom, witness) allowed.

    Used as the validation gate at the JSON I/O boundary: load parses through
    InputProgram (rejects lattice in stored), save round-trips through it
    before writing. Lattice claims live in memory only — they're derived by
    the analysis pass and lowered each build.
    """

    @field_validator("functions")
    @classmethod
    def _no_lattice_in_stored(cls, fns: tuple[Function, ...]) -> tuple[Function, ...]:
        for fn in fns:
            for c in fn.claims:
                if c.regime == "lattice":
                    raise ValueError(
                        f"lattice claims are derived, not stored; "
                        f"function {fn.name!r} has stored claim {c!r}"
                    )
        return fns


# ---------- File I/O ----------

def load_program(path: Path) -> Program:
    """Parse program.json. Validates as InputProgram (no lattice in stored)
    then returns the permissive Program type for in-memory editing."""
    raw = path.read_text()
    InputProgram.model_validate_json(raw)
    return Program.model_validate_json(raw)


def save_program(program: Program, path: Path) -> None:
    """Validate as InputProgram (raises if any lattice claims slipped into
    stored), then write JSON."""
    InputProgram(constants=program.constants, functions=program.functions)
    path.write_text(program.model_dump_json(indent=2))


# ---------- Lookups + immutable updates (by name) ----------

def find_function(program: Program, name: str) -> Function | None:
    for fn in program.functions:
        if fn.name == name:
            return fn
    return None


def require_function(program: Program, name: str) -> Function:
    fn = find_function(program, name)
    if fn is None:
        raise KeyError(f"no function named {name!r}")
    return fn


def replace_function(program: Program, new_fn: Function) -> Program:
    """Return a new Program with the same-named function replaced."""
    updated = tuple(new_fn if fn.name == new_fn.name else fn for fn in program.functions)
    if updated == program.functions:
        raise KeyError(f"no function named {new_fn.name!r}")
    return program.model_copy(update={"functions": updated})


def add_claim(program: Program, function: str, claim: Claim) -> Program:
    fn = require_function(program, function)
    target = claim_param(claim)
    if target is not None and target not in fn.params:
        raise KeyError(f"function {function!r} has no parameter {target!r}")
    for existing in fn.claims:
        if existing.kind == claim.kind and claim_param(existing) == target:
            scope = f"on {target!r}" if target is not None else "on return value"
            raise ValueError(
                f"{claim.kind} claim {scope} already present on {function}; "
                f"relax it first if you need to change bounds"
            )
    new_fn = fn.model_copy(update={"claims": fn.claims + (claim,)})
    return replace_function(program, new_fn)


def relax_claim(program: Program, function: str, kind: str, target: str | None) -> Program:
    """Remove the matching claim (no-op disallowed: must exist).

    target=None matches return-value claims (which have no parameter scope)."""
    fn = require_function(program, function)
    kept = tuple(
        c for c in fn.claims
        if not (c.kind == kind and claim_param(c) == target)
    )
    if len(kept) == len(fn.claims):
        scope = f"targeting {target!r}" if target is not None else "(return-value)"
        raise KeyError(f"no {kind} claim on {function} {scope}")
    new_fn = fn.model_copy(update={"claims": kept})
    return replace_function(program, new_fn)


# ---------- Pretty-printer ----------

# A "label" is an optional prefix the formatter inserts before each addressable
# node. The default is empty; the CLI passes a function returning [hashprefix]
# so each node prints with its addressable identity inline.
NodeLabel = Callable[[_Node], str]
_NO_LABEL: NodeLabel = lambda _node: ""


def format_program(program: Program, *, label: NodeLabel = _NO_LABEL) -> str:
    lines: list[str] = ["program {"]
    if program.constants:
        lines.append("  constants:")
        for c in program.constants:
            lines.append(f"    {label(c)}{c.name} = {c.value!r}")
    if program.functions:
        lines.append("  functions:")
        for fn in program.functions:
            lines.extend("    " + line for line in format_function(fn, label=label).splitlines())
    if not program.constants and not program.functions:
        lines.append("  (empty)")
    lines.append("}")
    return "\n".join(lines)


def format_function(fn: Function, *, label: NodeLabel = _NO_LABEL) -> str:
    sig_params = ", ".join(f"{p}: i32" for p in fn.params)
    header = f"{label(fn)}{fn.name}({sig_params}) -> i32"
    if fn.claims:
        header += "  [claims: " + ", ".join(format_claim(c) for c in fn.claims) + "]"
    body_lines = [_format_stmt(s, indent=2, label=label) for s in fn.body]
    return header + " {\n" + "\n".join(body_lines) + "\n}"


def format_claim(c: Claim) -> str:
    match c:
        case NonNegativeClaim(param=p):
            head = f"non_negative({p})"
        case IntRangeClaim(param=p, min=lo, max=hi):
            lo_s = "-inf" if lo is None else str(lo)
            hi_s = "+inf" if hi is None else str(hi)
            head = f"int_range({p}, [{lo_s}, {hi_s}])"
        case ReturnInRangeClaim(min=lo, max=hi):
            lo_s = "-inf" if lo is None else str(lo)
            hi_s = "+inf" if hi is None else str(hi)
            head = f"return_in_range([{lo_s}, {hi_s}])"
        case _:
            raise ValueError(f"unhandled claim: {c!r}")
    return head + format_claim_metadata(c)


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
    raise ValueError(f"unhandled justification: {j!r}")


def _format_stmt(stmt, indent: int, *, label: NodeLabel) -> str:
    pad = " " * indent
    prefix = label(stmt)
    match stmt:
        case CallPuts(target=t):
            return f"{pad}{prefix}puts({t})"
        case ReturnInt(value=v):
            return f"{pad}{prefix}return {v}"
        case ReturnExpr(value=expr):
            return f"{pad}{prefix}return {_format_expr(expr)}"
        case If(cond=cond, then_body=t_body, else_body=e_body):
            then_lines = "\n".join(_format_stmt(s, indent + 2, label=label) for s in t_body)
            else_lines = "\n".join(_format_stmt(s, indent + 2, label=label) for s in e_body)
            return (
                f"{pad}{prefix}if ({_format_expr(cond)}) {{\n"
                f"{then_lines}\n"
                f"{pad}}} else {{\n"
                f"{else_lines}\n"
                f"{pad}}}"
            )
    raise ValueError(f"unhandled stmt: {stmt!r}")


def _format_expr(expr) -> str:
    match expr:
        case IntLit(value=v):
            return str(v)
        case ParamRef(name=n):
            return n
        case BinOp(op=op, lhs=l, rhs=r):
            sym = {"add": "+", "slt": "<"}[op]
            return f"({_format_expr(l)} {sym} {_format_expr(r)})"
        case Call(function=fn_name, args=args):
            return f"{fn_name}({', '.join(_format_expr(a) for a in args)})"
    raise ValueError(f"unhandled expr: {expr!r}")
