"""Starter programs that `cpg init` can write to program.json."""

from cpg.model import (
    BinOp,
    CallPuts,
    Function,
    If,
    IntLit,
    ParamRef,
    Program,
    ReturnExpr,
    ReturnInt,
    StringConstant,
)


HELLO_WORLD = Program(
    constants=(StringConstant(name=".str.greeting", value="hello, world"),),
    functions=(
        Function(
            name="main",
            body=(
                CallPuts(target=".str.greeting"),
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
            params=("x",),
            body=(
                If(
                    cond=BinOp(op="slt", lhs=ParamRef(name="x"), rhs=IntLit(value=0)),
                    then_body=(ReturnExpr(value=IntLit(value=-1)),),
                    else_body=(
                        ReturnExpr(
                            value=BinOp(
                                op="add",
                                lhs=ParamRef(name="x"),
                                rhs=IntLit(value=1),
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
