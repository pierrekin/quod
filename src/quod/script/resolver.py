"""quod-script type resolver: phase-2 of parsing.

Walks the parsed AST, retyping every poison-marked IntLit using the
type of its operand context (the enclosing let's declared type, the
function's return type, the param being compared against, etc.).
Operates by reflection at the leaves (ParamRef, LocalRef, ReturnRef
pull their types from scope) plus a same-type constraint at every
BinOp / shift / IfExpr / boolean combinator.

Limits, deliberate: type inference doesn't reach into struct fields,
call args, store/load destinations, or widen targets — those still
resolve at lower time via `_coerce_int_lit`. The script doesn't have
the program in scope, so it can't see struct/extern signatures; pushing
a full program-aware resolver would be a different feature."""

from __future__ import annotations

from quod.model import (
    Assign,
    BinOp,
    Block,
    CharLit,
    Call,
    EnumInit,
    ExprStmt,
    FieldRead,
    FieldSet,
    For,
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
    IfExpr,
    If,
    IntLit,
    Let,
    Load,
    LocalRef,
    Match,
    Not,
    ParamRef,
    PtrOffset,
    Cast,
    F32Type,
    F64Type,
    ReturnExpr,
    ReturnRef,
    ShortCircuitAnd,
    ShortCircuitOr,
    Store,
    StructInit,
    TryExpr,
    While,
    WithArena,
)

from quod.script.tokens import ScriptError


_INT_TYPE_CLASSES = (
    I1Type, I8Type, I16Type, I32Type, I64Type,
    U8Type, U16Type, U32Type, U64Type,
    IsizeType, UsizeType,
)

_NUMERIC_TYPE_CLASSES = _INT_TYPE_CLASSES + (F32Type, F64Type)

_BINOP_CMP = frozenset({"slt", "sle", "sgt", "sge", "eq", "ne",
                        "ult", "ule", "ugt", "uge"})


class _Scope:
    """Per-function (or per-predicate) lookup table for ParamRef /
    LocalRef / ReturnRef types. Locals extend as the resolver walks
    into Let / For; the resolver discards them on the way out."""

    __slots__ = ("params", "locals_", "return_type")

    def __init__(self, params: dict, return_type, locals_=None):
        self.params: dict[str, object] = dict(params)
        self.return_type = return_type
        self.locals_: dict[str, object] = dict(locals_) if locals_ else {}

    def lookup_local(self, name: str):
        return self.locals_.get(name)

    def lookup_param(self, name: str):
        return self.params.get(name)


class _Resolver:
    """Walks the parsed tree, retyping bare IntLits in place via
    model_copy. Carries the parser's `poison_locs` side table so it can
    drop a literal's poison marker the moment it's typed and so the
    final-scan can report the source location of any unresolved
    literal."""

    def __init__(self, poison_locs: dict[int, tuple[int, int]]):
        self.poison_locs = poison_locs

    # -- expression walk --

    def expr(self, e, expected, scope: _Scope):
        """Resolve `e` under `scope`, with `expected` providing the
        outer context's int type (or None). Returns (new_e, type_or_None)
        where the type is the resolved type if known."""
        match e:
            case IntLit():
                return self._int_lit(e, expected)
            case ParamRef(name=name):
                return e, scope.lookup_param(name)
            case LocalRef(name=name):
                return e, scope.lookup_local(name)
            case ReturnRef():
                rt = scope.return_type
                return e, rt if isinstance(rt, _INT_TYPE_CLASSES) else None
            case BinOp(op=op, lhs=lhs, rhs=rhs):
                return self._binop(e, op, lhs, rhs, expected, scope)
            case ShortCircuitOr(lhs=lhs, rhs=rhs):
                new_lhs, _ = self.expr(lhs, I1Type(), scope)
                new_rhs, _ = self.expr(rhs, I1Type(), scope)
                return e.model_copy(update={"lhs": new_lhs, "rhs": new_rhs}), I1Type()
            case ShortCircuitAnd(lhs=lhs, rhs=rhs):
                new_lhs, _ = self.expr(lhs, I1Type(), scope)
                new_rhs, _ = self.expr(rhs, I1Type(), scope)
                return e.model_copy(update={"lhs": new_lhs, "rhs": new_rhs}), I1Type()
            case Not(operand=op):
                new_op, _ = self.expr(op, I1Type(), scope)
                return e.model_copy(update={"operand": new_op}), I1Type()
            case IfExpr(cond=cond, then_value=tv, else_value=ev):
                new_cond, _ = self.expr(cond, I1Type(), scope)
                new_tv, t_ty = self.expr(tv, expected, scope)
                new_ev, e_ty = self.expr(ev, expected or t_ty, scope)
                # Cross-propagate: if one branch resolved and the other is
                # still bare, retype the bare branch to match.
                if t_ty is None and e_ty is not None:
                    new_tv, t_ty = self.expr(tv, e_ty, scope)
                inferred = expected or t_ty or e_ty
                return e.model_copy(
                    update={"cond": new_cond, "then_value": new_tv,
                            "else_value": new_ev}
                ), inferred
            case Call(args=args):
                # The script doesn't carry callee signatures, so we
                # can't type call args from context. Resolve their
                # subtrees (so e.g. nested let-init works) and then
                # release any leftover poison to the lower pass's
                # `_coerce_int_lit`.
                new_args = []
                changed = False
                for a in args:
                    new_a, _ = self.expr(a, None, scope)
                    if new_a is not a:
                        changed = True
                    new_args.append(new_a)
                    self._drop_poison_in(new_a)
                if changed:
                    return e.model_copy(update={"args": tuple(new_args)}), None
                return e, None
            case StructInit(fields=fields):
                # Struct field types live on the StructDef, which the
                # script doesn't see; fall through to lower-time coercion.
                new_fields, changed = self._field_inits(fields, scope, drop_poison=True)
                if changed:
                    return e.model_copy(update={"fields": new_fields}), None
                return e, None
            case EnumInit(fields=fields):
                new_fields, changed = self._field_inits(fields, scope, drop_poison=True)
                if changed:
                    return e.model_copy(update={"fields": new_fields}), None
                return e, None
            case FieldRead(value=inner):
                new_inner, _ = self.expr(inner, None, scope)
                if new_inner is not inner:
                    return e.model_copy(update={"value": new_inner}), None
                return e, None
            case PtrOffset(base=b, offset=o):
                new_b, _ = self.expr(b, None, scope)
                # Offset is i64 by lowering convention.
                new_o, _ = self.expr(o, I64Type(), scope)
                if new_b is not b or new_o is not o:
                    return e.model_copy(update={"base": new_b, "offset": new_o}), None
                return e, None
            case Cast(value=v, target_type=target):
                # Source type natural to operand; result is target_type.
                # _NUMERIC_TYPE_CLASSES is the int + float union.
                new_v, _ = self.expr(v, None, scope)
                result_ty = target if isinstance(target, _NUMERIC_TYPE_CLASSES) else None
                if new_v is not v:
                    return e.model_copy(update={"value": new_v}), result_ty
                return e, result_ty
            case Load(ptr=p):
                new_p, _ = self.expr(p, None, scope)
                if new_p is not p:
                    return e.model_copy(update={"ptr": new_p}), None
                return e, None
            case TryExpr(value=v):
                new_v, _ = self.expr(v, None, scope)
                if new_v is not v:
                    return e.model_copy(update={"value": new_v}), None
                return e, None
            case _:
                # Leaf nodes with no children to walk (CharLit, NullPtr,
                # SizeOf, StringRef) and any other Expr we don't
                # specifically handle. CharLit returns i8.
                if isinstance(e, CharLit):
                    return e, I8Type()
                return e, None

    def _int_lit(self, lit: IntLit, expected):
        if id(lit) not in self.poison_locs:
            return lit, lit.type
        if expected is not None and isinstance(expected, _INT_TYPE_CLASSES):
            new = lit.model_copy(update={"type": expected})
            del self.poison_locs[id(lit)]
            return new, expected
        return lit, None

    def _drop_poison_in(self, e):
        """Mark every IntLit reachable from `e` as 'lower-time will fix
        this'. Used for contexts the script-time resolver can't reach
        (Call args without a callee signature, struct/enum field
        initializers without a struct def, store/field-set
        destinations). The literal keeps its i64 placeholder; the
        lower pass's `_coerce_int_lit` retypes it once the destination
        type is in scope."""
        for lit in _iter_int_lits(e):
            self.poison_locs.pop(id(lit), None)

    def _binop(self, e, op, lhs, rhs, expected, scope):
        if op in _BINOP_CMP:
            # cmp: result is i1; operand width is shared. Resolve LHS
            # first with no outer expectation (since `expected` here is
            # the *cmp result's* type, not the operands'); then use LHS's
            # inferred type as the expectation for RHS. Re-resolve LHS if
            # RHS pinned a type that LHS missed.
            new_lhs, l_ty = self.expr(lhs, None, scope)
            new_rhs, r_ty = self.expr(rhs, l_ty, scope)
            if l_ty is None and r_ty is not None:
                new_lhs, l_ty = self.expr(lhs, r_ty, scope)
            return e.model_copy(update={"lhs": new_lhs, "rhs": new_rhs}), I1Type()
        # arith / bitwise / shift: operands and result share a type.
        # `expected` may flow from the surrounding context (the let's
        # declared type, the return type, etc.).
        new_lhs, l_ty = self.expr(lhs, expected, scope)
        new_rhs, r_ty = self.expr(rhs, expected or l_ty, scope)
        if expected is None and l_ty is None and r_ty is not None:
            new_lhs, l_ty = self.expr(lhs, r_ty, scope)
        result_ty = expected or l_ty or r_ty
        return e.model_copy(update={"lhs": new_lhs, "rhs": new_rhs}), result_ty

    def _field_inits(self, fields, scope, *, drop_poison: bool = False):
        """Walk a tuple of FieldInit; struct/enum field types aren't
        visible at script time so we resolve children with no
        expectation. With `drop_poison=True`, leftover bare literals are
        released to lower-time coercion (StructInit / EnumInit reach
        here). Returns (new_fields_tuple, any_changed)."""
        out = []
        changed = False
        for fi in fields:
            new_v, _ = self.expr(fi.value, None, scope)
            if drop_poison:
                self._drop_poison_in(new_v)
            if new_v is not fi.value:
                out.append(fi.model_copy(update={"value": new_v}))
                changed = True
            else:
                out.append(fi)
        return tuple(out), changed

    # -- statement walk --

    def block(self, b: Block, scope: _Scope) -> Block:
        new_stmts = []
        for s in b.stmts:
            new_stmts.append(self.stmt(s, scope))
        return b.model_copy(update={"stmts": tuple(new_stmts)})

    def stmt(self, s, scope: _Scope):
        match s:
            case Let(name=name, type=ty, init=init):
                new_init = init
                if init is not None:
                    expected = ty if isinstance(ty, _INT_TYPE_CLASSES) else None
                    new_init, _ = self.expr(init, expected, scope)
                # Bind the local in scope for subsequent statements.
                scope.locals_[name] = ty
                if new_init is not init:
                    return s.model_copy(update={"init": new_init})
                return s
            case Assign(name=name, value=v):
                ty = scope.lookup_local(name) or scope.lookup_param(name)
                expected = ty if isinstance(ty, _INT_TYPE_CLASSES) else None
                new_v, _ = self.expr(v, expected, scope)
                if new_v is not v:
                    return s.model_copy(update={"value": new_v})
                return s
            case ReturnExpr(value=v):
                rt = scope.return_type
                expected = rt if isinstance(rt, _INT_TYPE_CLASSES) else None
                new_v, _ = self.expr(v, expected, scope)
                if new_v is not v:
                    return s.model_copy(update={"value": new_v})
                return s
            case ExprStmt(value=v):
                new_v, _ = self.expr(v, None, scope)
                if new_v is not v:
                    return s.model_copy(update={"value": new_v})
                return s
            case If(cond=cond, then_body=tb, else_body=eb):
                new_cond, _ = self.expr(cond, I1Type(), scope)
                new_tb = self.block(tb, _Scope(scope.params, scope.return_type, scope.locals_))
                new_eb = self.block(eb, _Scope(scope.params, scope.return_type, scope.locals_))
                return s.model_copy(update={
                    "cond": new_cond, "then_body": new_tb, "else_body": new_eb,
                })
            case While(cond=cond, body=b):
                new_cond, _ = self.expr(cond, I1Type(), scope)
                new_body = self.block(b, _Scope(scope.params, scope.return_type, scope.locals_))
                return s.model_copy(update={"cond": new_cond, "body": new_body})
            case For(var=var, type=ty, lo=lo, hi=hi, body=b):
                expected = ty if isinstance(ty, _INT_TYPE_CLASSES) else None
                new_lo, _ = self.expr(lo, expected, scope)
                new_hi, _ = self.expr(hi, expected, scope)
                inner = _Scope(scope.params, scope.return_type, scope.locals_)
                inner.locals_[var] = ty
                new_body = self.block(b, inner)
                return s.model_copy(update={
                    "lo": new_lo, "hi": new_hi, "body": new_body,
                })
            case FieldSet(value=v):
                # Struct field types aren't visible to the script;
                # rely on lower-time coercion.
                new_v, _ = self.expr(v, None, scope)
                self._drop_poison_in(new_v)
                if new_v is not v:
                    return s.model_copy(update={"value": new_v})
                return s
            case Store(ptr=p, value=v):
                new_p, _ = self.expr(p, None, scope)
                new_v, _ = self.expr(v, None, scope)
                self._drop_poison_in(new_v)
                if new_p is not p or new_v is not v:
                    return s.model_copy(update={"ptr": new_p, "value": new_v})
                return s
            case WithArena(capacity=cap, body=b):
                # capacity is i64 by convention.
                new_cap, _ = self.expr(cap, I64Type(), scope)
                new_body = self.block(b, _Scope(scope.params, scope.return_type, scope.locals_))
                return s.model_copy(update={"capacity": new_cap, "body": new_body})
            case Match(scrutinee=sc, arms=arms):
                new_sc, _ = self.expr(sc, None, scope)
                new_arms = []
                for arm in arms:
                    new_body = self.block(
                        arm.body,
                        _Scope(scope.params, scope.return_type, scope.locals_),
                    )
                    new_arms.append(arm.model_copy(update={"body": new_body}))
                return s.model_copy(update={
                    "scrutinee": new_sc, "arms": tuple(new_arms),
                })
            case _:
                return s

    # -- final scan: any leftover poison is an error --

    def assert_no_poison(self, root):
        """Walk `root` (Function | Block | Expr) and raise ScriptError if
        any IntLit's id is still in the poison set."""
        for lit_id, (line, col) in list(self.poison_locs.items()):
            # Only literals reachable from `root` matter; entries for
            # original-tree IntLits that the resolver has copied away
            # have already been removed via `del`. The rest are real
            # unresolved bare literals.
            if _find_intlit_with_id(root, lit_id):
                raise ScriptError(
                    "bare integer literal needs a width suffix here — "
                    "no operand context to infer the type from",
                    line, col,
                )


def _find_intlit_with_id(node, target_id: int) -> bool:
    """True if some IntLit reachable from `node` has the given id()."""
    for lit in _iter_int_lits(node):
        if id(lit) == target_id:
            return True
    return False


def _iter_int_lits(node):
    """Yield every IntLit reachable from `node`. Walks frozen-Pydantic
    nodes via `__pydantic_fields__` so it stays generic over the model."""
    if isinstance(node, IntLit):
        yield node
        return
    fields = getattr(node, "__pydantic_fields__", None)
    if fields is None:
        return
    for fname in fields:
        v = getattr(node, fname, None)
        if v is None:
            continue
        if isinstance(v, tuple):
            for item in v:
                yield from _iter_int_lits(item)
        elif hasattr(v, "__pydantic_fields__"):
            yield from _iter_int_lits(v)
