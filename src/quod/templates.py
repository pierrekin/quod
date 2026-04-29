"""Starter programs that `quod init` can write to program.json."""

from quod.model import (
    BinOp,
    Call,
    ExprStmt,
    ExternFunction,
    Function,
    I8PtrType,
    I32Type,
    If,
    IntLit,
    Param,
    ParamRef,
    Program,
    ReturnExpr,
    ReturnInt,
    StringConstant,
    StringRef,
)


_I32 = I32Type()


HELLO_WORLD = Program(
    constants=(StringConstant(name=".str.greeting", value="hello, world"),),
    externs=(ExternFunction(name="puts", param_types=(I8PtrType(),)),),
    functions=(
        Function(
            name="main",
            return_type=_I32,
            body=(
                ExprStmt(value=Call(function="puts", args=(StringRef(name=".str.greeting"),))),
                ReturnInt(value=0),
            ),
        ),
    ),
)


# f(x: i32) -> i32 = if x < 0 then -1 else x + 1.
# Designed to demonstrate claim exploitation: with a non_negative(x) claim,
# the negative branch is unreachable and the optimizer eliminates it.
GUARDED_INC = Program(
    functions=(
        Function(
            name="f",
            params=(Param(name="x", type=_I32),),
            return_type=_I32,
            body=(
                If(
                    cond=BinOp(
                        op="slt",
                        lhs=ParamRef(name="x"),
                        rhs=IntLit(type=_I32, value=0),
                    ),
                    then_body=(ReturnExpr(value=IntLit(type=_I32, value=-1)),),
                    else_body=(
                        ReturnExpr(
                            value=BinOp(
                                op="add",
                                lhs=ParamRef(name="x"),
                                rhs=IntLit(type=_I32, value=1),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
)


EMPTY = Program()


TEMPLATES: dict[str, Program] = {
    "hello": HELLO_WORLD,
    "guarded": GUARDED_INC,
    "empty": EMPTY,
}
