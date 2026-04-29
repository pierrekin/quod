"""Proof artifact generation and verification via Z3 + SMT-LIB.

Two paths:
  - generate: claim + function body → SMT-LIB problem statement (.smt2)
  - check:    .smt2 → invoke Z3 → confirm `unsat` (= claim holds)

The encoding lives entirely in this module — `model` knows nothing about SMT.

Coverage:
  expressions: IntLit, ParamRef, BinOp(add, slt), Call (cross-procedural)
  statements:  ReturnInt, ReturnExpr, If (both branches return), CallPuts (skipped)
  claims:      NonNegativeClaim, IntRangeClaim, ReturnInRangeClaim
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

from cpg.model import (
    BinOp,
    Call,
    CallPuts,
    Claim,
    Function,
    If,
    IntLit,
    IntRangeClaim,
    NonNegativeClaim,
    ParamRef,
    Program,
    ReturnExpr,
    ReturnInRangeClaim,
    ReturnInt,
    StringRef,
    claim_param,
)


# i32 bounds — used as the universe for parameters when no claim narrows them.
I32_MIN = -2**31
I32_MAX = 2**31 - 1


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
    fn_return_claims: dict[str, tuple[ReturnInRangeClaim, ...]] = field(default_factory=dict)


def _expr_to_smt(expr, state: _SmtState) -> str:
    match expr:
        case IntLit(value=v):
            return str(v)
        case ParamRef(name=n):
            return n
        case StringRef():
            raise NotImplementedError(
                "can't lower StringRef for SMT — function under proof contains "
                "an i8* expression; SMT model is Int-only"
            )
        case BinOp(op="add", lhs=l, rhs=r):
            return f"(+ {_expr_to_smt(l, state)} {_expr_to_smt(r, state)})"
        case BinOp(op="slt", lhs=l, rhs=r):
            return f"(< {_expr_to_smt(l, state)} {_expr_to_smt(r, state)})"
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
                pred = claim_smt_predicate(rc, call_term)
                if pred not in state.asserted_preds:
                    state.asserted_preds.add(pred)
                    state.extra_asserts.append(
                        f";   {fname}'s {rc.kind} return claim, on {call_term}"
                    )
                    state.extra_asserts.append(f"(assert {pred})")
            return call_term
    raise NotImplementedError(f"can't lower expr {expr!r} for SMT")


def _stmts_to_return_smt(stmts, state: _SmtState) -> str:
    """Translate a sequence of statements into the SMT term for the eventual
    return value. Side-effect-only statements (CallPuts) are skipped."""
    for stmt in stmts:
        match stmt:
            case ReturnInt(value=v):
                return str(v)
            case ReturnExpr(value=expr):
                return _expr_to_smt(expr, state)
            case If(cond=cond, then_body=t, else_body=e):
                return (
                    f"(ite {_expr_to_smt(cond, state)} "
                    f"{_stmts_to_return_smt(list(t), state)} "
                    f"{_stmts_to_return_smt(list(e), state)})"
                )
            case CallPuts():
                continue  # side effect only; doesn't influence return value
            case _:
                raise NotImplementedError(f"can't lower stmt {stmt!r} for SMT")
    raise NotImplementedError("function body has no terminating return")


def _build_fn_return_claims_index(program: Program | None) -> dict[str, tuple[ReturnInRangeClaim, ...]]:
    if program is None:
        return {}
    out: dict[str, tuple[ReturnInRangeClaim, ...]] = {}
    for fn in program.functions:
        rcs = tuple(c for c in fn.claims if isinstance(c, ReturnInRangeClaim))
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
    term = _stmts_to_return_smt(list(fn.body), state)
    return term, state


# ---------- Claim → SMT predicate ----------

def claim_smt_predicate(c: Claim, return_term: str) -> str:
    """SMT-LIB Bool-valued predicate the claim *asserts*.

    Param-scoped claims read the param by name. Return-scoped claims read
    the supplied return-value term.
    """
    match c:
        case NonNegativeClaim(param=p):
            return f"(>= {p} 0)"
        case IntRangeClaim(param=p, min=lo, max=hi):
            return _range_pred(p, lo, hi)
        case ReturnInRangeClaim(min=lo, max=hi):
            return _range_pred(return_term, lo, hi)
    raise NotImplementedError(f"can't lower claim {c!r} for SMT")


def _range_pred(term: str, lo: int | None, hi: int | None) -> str:
    parts: list[str] = []
    if lo is not None:
        parts.append(f"(>= {term} {lo})")
    if hi is not None:
        parts.append(f"(<= {term} {hi})")
    if not parts:
        return "true"
    if len(parts) == 1:
        return parts[0]
    return f"(and {' '.join(parts)})"


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
    lines.append(f"; auto-generated by cpg.proof for function {fn.name}")
    lines.append(f"; goal: {goal!r}")
    lines.append(f"(set-logic {logic})")
    lines.append("")

    for p in fn.params:
        lines.append(f"(declare-const {p} Int)")
        # Bound params to the i32 universe; otherwise Z3 finds counterexamples
        # in the unbounded integers that don't apply to our actual codegen.
        lines.append(f"(assert (>= {p} {I32_MIN}))")
        lines.append(f"(assert (<= {p} {I32_MAX}))")

    if state.extra_decls:
        lines.append("")
        lines.append("; callee declarations (cross-procedural)")
        lines.extend(state.extra_decls)

    if hypotheses:
        lines.append("")
        lines.append("; hypotheses (existing claims on this function)")
        for h in hypotheses:
            lines.append(f";   {h.kind}({claim_param(h) or 'return'})")
            lines.append(f"(assert {claim_smt_predicate(h, return_term)})")

    if state.extra_asserts:
        lines.append("")
        lines.append("; hypotheses (callees' return claims, applied per call)")
        lines.extend(state.extra_asserts)

    lines.append("")
    lines.append("; goal (negated; we ask Z3 to find a counterexample)")
    lines.append(f"(assert (not {claim_smt_predicate(goal, return_term)}))")
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
