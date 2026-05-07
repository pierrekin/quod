"""Proof artifact generation and verification via Z3 + SMT-LIB.

Two paths:
  - generate: claim + function body → SMT-LIB problem statement (.smt2)
  - check:    .smt2 → invoke Z3 → confirm `unsat` (= claim holds)

The encoding lives entirely in this module — `model` knows nothing about SMT.

Coverage:
  expressions: IntLit, ParamRef, BinOp(add, sub, mul, slt, eq), Call (cross-procedural)
  statements:  ReturnExpr, If (both branches return), ExprStmt (skipped)
  claims:      PredicateClaim (canonical predicate body)
                 - as hypotheses on the function under analysis (via `hypotheses=`)
                 - as hypotheses on calls to *other* user functions (via `program=`),
                   so the callee's return claims constrain the call result

Cross-procedural strategy (uninterpreted functions in QF_UFLIA):
  Each user function `f` becomes an opaque SMT symbol `(declare-fun f (Int...) Int)`.
  A call `f(arg)` is the SMT term `(f arg)`. Two calls with the same arg yield the
  same term — referential transparency is preserved. The callee's return claims
  are asserted per call site as hypotheses on `(f arg)`. Without a return claim
  on the callee, the call's result is unconstrained.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field

from quod.model import (
    Assign,
    BinOp,
    Call,
    Claim,
    ExprStmt,
    For,
    Function,
    I1Type,
    I8Type,
    I16Type,
    I32Type,
    I64Type,
    U8Type,
    U16Type,
    U32Type,
    U64Type,
    IsizeType,
    UsizeType,
    If,
    IfExpr,
    IntLit,
    IntType,
    Let,
    LocalRef,
    Not,
    ParamRef,
    PredicateClaim,
    Program,
    ReturnExpr,
    ReturnRef,
    ShortCircuitAnd,
    ShortCircuitOr,
    StringRef,
    While,
)


def _type_universe(t: IntType) -> tuple[int, int]:
    """Inclusive [min, max] for the SMT Int sort modeling a quod int type.

    LLVM int types carry no signedness — quod's signed/unsigned op variants
    decide interpretation. SMT's Int is unbounded; the universe constraint
    keeps Z3 honest about widths instead of finding counterexamples that
    couldn't appear in our actual codegen. Signed two's-complement bounds
    apply for i8..i64; u8..u64 use [0, 2^N - 1]; i1 is the unsigned
    boolean {0, 1}.
    """
    match t:
        case I1Type():
            return (0, 1)
        case I8Type():
            return (-(2**7), 2**7 - 1)
        case I16Type():
            return (-(2**15), 2**15 - 1)
        case I32Type():
            return (-(2**31), 2**31 - 1)
        case I64Type():
            return (-(2**63), 2**63 - 1)
        case U8Type():
            return (0, 2**8 - 1)
        case U16Type():
            return (0, 2**16 - 1)
        case U32Type():
            return (0, 2**32 - 1)
        case U64Type():
            return (0, 2**64 - 1)
        case IsizeType():
            return (-(2**63), 2**63 - 1)
        case UsizeType():
            return (0, 2**64 - 1)
    raise ValueError(f"not an int type: {t!r}")


# Map quod.BinOp.op -> SMT-LIB operator. `ne` is special-cased (distinct).
# Unsigned comparisons (ult/ule/ugt/uge) are intentionally absent: QF_LIA's
# Int sort is signed, so unsigned semantics don't translate cleanly.
_SMT_BINOP = {
    "add": "+", "sub": "-", "mul": "*",
    "slt": "<", "sle": "<=", "sgt": ">", "sge": ">=", "eq": "=",
}


# ---------- Body → SMT ----------

@dataclass
class _SmtState:
    """Mutable state collected while lowering a body to SMT.

    Calls to user functions add `declare-fun` lines and per-call assertions
    of the callee's return claims. The state is emitted alongside the body
    term in goal_smt_lib.
    """
    declared_fns: set[str] = field(default_factory=set)
    extra_decls: list[str] = field(default_factory=list)
    extra_asserts: list[str] = field(default_factory=list)
    asserted_preds: set[str] = field(default_factory=set)
    fn_return_claims: dict[str, tuple[PredicateClaim, ...]] = field(default_factory=dict)


def _expr_to_smt(expr, state: _SmtState) -> str:
    match expr:
        case IntLit(value=v):
            return str(v)
        case ParamRef(name=n):
            return n
        case LocalRef():
            raise NotImplementedError(
                "can't lower LocalRef for SMT — function under proof uses "
                "mutable local state (Let/Assign); the SMT model is "
                "pure-expression-only"
            )
        case StringRef():
            raise NotImplementedError(
                "can't lower StringRef for SMT — function under proof contains "
                "an i8* expression; SMT model is Int-only"
            )
        case ShortCircuitOr() | ShortCircuitAnd():
            raise NotImplementedError(
                "can't lower short-circuit Or/And for SMT; "
                "rewrite as nested If if you need a proof"
            )
        case IfExpr():
            raise NotImplementedError(
                "can't lower IfExpr (ternary) for SMT; "
                "rewrite as a statement-level If if you need a proof"
            )
        case BinOp(op=op, lhs=l, rhs=r) if op in _SMT_BINOP:
            return f"({_SMT_BINOP[op]} {_expr_to_smt(l, state)} {_expr_to_smt(r, state)})"
        case BinOp(op="ne", lhs=l, rhs=r):
            return f"(distinct {_expr_to_smt(l, state)} {_expr_to_smt(r, state)})"
        case BinOp(op=op):
            # sdiv/udiv/srem, or, and, ult/ule/ugt/uge: skipped for SMT —
            # semantic mismatch with QF_LIA (LLVM's sdiv truncates toward
            # zero while SMT div floors; srem ≠ SMT mod; signed/unsigned
            # cmp; boolean-vs-integer ambiguity for or/and).
            raise NotImplementedError(
                f"can't lower BinOp(op={op!r}) for SMT"
            )
        case Call(function=fname, args=args):
            arg_terms = [_expr_to_smt(a, state) for a in args]
            if fname not in state.declared_fns:
                state.declared_fns.add(fname)
                arg_sort = " ".join(["Int"] * len(args))
                state.extra_decls.append(f"(declare-fun {fname} ({arg_sort}) Int)")
            call_term = (
                f"({fname} {' '.join(arg_terms)})" if arg_terms else f"({fname})"
            )
            # Assert callee's return claims about THIS call. Multiple calls with
            # the same args produce the same SMT term, so duplicate assertions
            # are filtered via asserted_preds.
            for rc in state.fn_return_claims.get(fname, ()):
                pred = predicate_to_smt(rc.expr, call_term)
                if pred not in state.asserted_preds:
                    state.asserted_preds.add(pred)
                    state.extra_asserts.append(
                        f";   {fname}'s return claim, on {call_term}"
                    )
                    state.extra_asserts.append(f"(assert {pred})")
            return call_term
    raise NotImplementedError(f"can't lower expr {expr!r} for SMT")


def _stmts_to_return_smt(stmts, state: _SmtState) -> str:
    """Translate a sequence of statements into the SMT term for the eventual
    return value. Side-effect-only statements (ExprStmt) are skipped."""
    for stmt in stmts:
        match stmt:
            case ReturnExpr(value=expr):
                return _expr_to_smt(expr, state)
            case If(cond=cond, then_body=t, else_body=e):
                return (
                    f"(ite {_expr_to_smt(cond, state)} "
                    f"{_stmts_to_return_smt(list(t.stmts), state)} "
                    f"{_stmts_to_return_smt(list(e.stmts), state)})"
                )
            case ExprStmt():
                continue  # side effect only; doesn't influence return value
            case Let() | Assign() | While() | For():
                raise NotImplementedError(
                    f"can't prove things about functions with mutable state or "
                    f"loops yet (saw {type(stmt).__name__}); rewrite as a "
                    f"recursive helper if you want a proof"
                )
            case _:
                raise NotImplementedError(f"can't lower stmt {stmt!r} for SMT")
    raise NotImplementedError("function body has no terminating return")


def _build_fn_return_claims_index(program: Program | None) -> dict[str, tuple[PredicateClaim, ...]]:
    if program is None:
        return {}
    out: dict[str, tuple[PredicateClaim, ...]] = {}
    for fn in program.functions:
        rcs = tuple(c for c in fn.claims if predicate_uses_return(c.expr))
        if rcs:
            out[fn.name] = rcs
    return out


def function_return_term(fn: Function, *, program: Program | None = None) -> tuple[str, _SmtState]:
    """SMT-LIB Int-valued term for fn's return value, plus the state collected
    while walking the body (call decls + per-call hypotheses).

    `program` is needed to look up callees' return claims; pass None to skip
    cross-procedural reasoning (calls become unconstrained).
    """
    state = _SmtState(fn_return_claims=_build_fn_return_claims_index(program))
    term = _stmts_to_return_smt(list(fn.body.stmts), state)
    return term, state


# Bool-valued BinOp ops used inside predicates. Map to SMT-LIB native names.
# `and`/`or`/`xor` are treated as Bool combinators here — predicates use
# i1 operands, where the bitwise and Boolean readings coincide.
_PRED_BOOL_BINOP = {"and": "and", "or": "or", "xor": "xor"}


def predicate_to_smt(expr, return_term: str | None = None) -> str:
    """Translate a `PredicateClaim.expr` into an SMT-LIB Bool predicate.

    Recursive walk over the expression vocabulary admitted by the
    predicate validator (i1-typed, side-effect-free). Unknown shapes
    raise NotImplementedError so we fail loudly rather than silently
    losing soundness.

    `return_term` names the SMT term standing for the function's
    return value; required iff the predicate references `ReturnRef`.
    """
    match expr:
        case IntLit(type=I1Type(), value=v):
            return "true" if v else "false"
        case IntLit(value=v):
            return str(v)
        case ParamRef(name=n):
            return n
        case ReturnRef():
            if return_term is None:
                raise ValueError(
                    "ReturnRef in predicate but no return_term provided"
                )
            return return_term
        case Not(operand=op):
            return f"(not {predicate_to_smt(op, return_term)})"
        case BinOp(op=op, lhs=l, rhs=r) if op in _PRED_BOOL_BINOP:
            return (
                f"({_PRED_BOOL_BINOP[op]} "
                f"{predicate_to_smt(l, return_term)} "
                f"{predicate_to_smt(r, return_term)})"
            )
        case BinOp(op="ne", lhs=l, rhs=r):
            return (
                f"(distinct {predicate_to_smt(l, return_term)} "
                f"{predicate_to_smt(r, return_term)})"
            )
        case BinOp(op=op, lhs=l, rhs=r) if op in _SMT_BINOP:
            return (
                f"({_SMT_BINOP[op]} "
                f"{predicate_to_smt(l, return_term)} "
                f"{predicate_to_smt(r, return_term)})"
            )
        case BinOp(op=op):
            raise NotImplementedError(
                f"can't lower BinOp(op={op!r}) for SMT in predicate "
                f"(unsigned comparisons and div/rem don't translate to QF_LIA)"
            )
    raise NotImplementedError(f"can't lower expr {expr!r} for SMT predicate")


def predicate_uses_return(expr) -> bool:
    """True iff `expr` references `ReturnRef` anywhere in its subtree.

    Used to decide whether a `PredicateClaim` is a precondition (over
    params only) or a postcondition (over params and the return value).
    """
    if isinstance(expr, ReturnRef):
        return True
    if isinstance(expr, BinOp):
        return predicate_uses_return(expr.lhs) or predicate_uses_return(expr.rhs)
    if isinstance(expr, ShortCircuitOr) or isinstance(expr, ShortCircuitAnd):
        return predicate_uses_return(expr.lhs) or predicate_uses_return(expr.rhs)
    if isinstance(expr, Not):
        return predicate_uses_return(expr.operand)
    if isinstance(expr, IfExpr):
        return (
            predicate_uses_return(expr.cond)
            or predicate_uses_return(expr.then_value)
            or predicate_uses_return(expr.else_value)
        )
    return False


# ---------- Full SMT-LIB problem ----------

def goal_smt_lib(
    fn: Function, goal: Claim, *,
    hypotheses: tuple[Claim, ...] = (),
    program: Program | None = None,
) -> str:
    """Build a full SMT-LIB problem that's `unsat` iff `goal` holds for `fn`.

    Strategy: assert hypotheses as facts; assert NEGATION of goal; ask Z3 to
    find a model. unsat = no counterexample = goal holds.

    `program` enables cross-procedural reasoning: calls in fn's body become
    SMT terms over uninterpreted function symbols, with the callee's return
    claims (looked up in `program`) asserted per call site.
    """
    return_term, state = function_return_term(fn, program=program)

    # If any cross-procedural calls were lowered, switch to QF_UFLIA.
    logic = "QF_UFLIA" if state.declared_fns else "QF_LIA"

    lines: list[str] = []
    lines.append(f"; auto-generated by quod.predicate.predicate_proof for function {fn.name}")
    lines.append(f"; goal: {goal!r}")
    lines.append(f"(set-logic {logic})")
    lines.append("")

    for p in fn.params:
        lo, hi = _type_universe(p.type)
        lines.append(f"(declare-const {p.name} Int)  ; {p.type.kind}")
        # Bound params to the type's universe; otherwise Z3 finds counterexamples
        # in the unbounded integers that don't apply to our actual codegen.
        lines.append(f"(assert (>= {p.name} {lo}))")
        lines.append(f"(assert (<= {p.name} {hi}))")

    if state.extra_decls:
        lines.append("")
        lines.append("; callee declarations (cross-procedural)")
        lines.extend(state.extra_decls)

    if hypotheses:
        lines.append("")
        lines.append("; hypotheses (existing claims on this function)")
        for h in hypotheses:
            scope = "return" if predicate_uses_return(h.expr) else "params"
            lines.append(f";   predicate ({scope}-scoped)")
            lines.append(f"(assert {predicate_to_smt(h.expr, return_term)})")

    if state.extra_asserts:
        lines.append("")
        lines.append("; hypotheses (callees' return claims, applied per call)")
        lines.extend(state.extra_asserts)

    lines.append("")
    lines.append("; goal (negated; we ask Z3 to find a counterexample)")
    lines.append(f"(assert (not {predicate_to_smt(goal.expr, return_term)}))")
    lines.append("")
    lines.append("(check-sat)")
    lines.append("(exit)")
    return "\n".join(lines) + "\n"


# ---------- Z3 invocation ----------

@dataclass(frozen=True)
class Z3Result:
    status: str   # "unsat" | "sat" | "unknown"
    raw: str      # full stdout for debugging


class Z3NotInstalled(RuntimeError):
    pass


def run_z3_on_smt(smt: str, *, timeout_s: float = 10.0) -> Z3Result:
    """Pipe SMT-LIB content to z3 via stdin and parse its first-line answer."""
    if shutil.which("z3") is None:
        raise Z3NotInstalled("z3 binary not found in PATH (install with `pacman -S z3`)")
    proc = subprocess.run(
        ["z3", "-in"],
        input=smt,
        capture_output=True, text=True, timeout=timeout_s, check=False,
    )
    out = (proc.stdout or "").strip()
    first = out.splitlines()[0].strip() if out else ""
    if first not in ("sat", "unsat", "unknown"):
        raise RuntimeError(
            f"z3 returned unexpected output:\nstdout: {proc.stdout!r}\n"
            f"stderr: {proc.stderr!r}\nrc: {proc.returncode}"
        )
    return Z3Result(status=first, raw=proc.stdout)


def run_z3_on_file(path) -> Z3Result:
    """Run z3 on an .smt2 file directly. Used by verify-claims."""
    if shutil.which("z3") is None:
        raise Z3NotInstalled("z3 binary not found in PATH (install with `pacman -S z3`)")
    proc = subprocess.run(
        ["z3", str(path)],
        capture_output=True, text=True, timeout=10.0, check=False,
    )
    out = (proc.stdout or "").strip()
    first = out.splitlines()[0].strip() if out else ""
    if first not in ("sat", "unsat", "unknown"):
        raise RuntimeError(
            f"z3 returned unexpected output:\nstdout: {proc.stdout!r}\n"
            f"stderr: {proc.stderr!r}\nrc: {proc.returncode}"
        )
    return Z3Result(status=first, raw=proc.stdout)
