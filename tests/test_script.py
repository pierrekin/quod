"""Tests for `quod.script` — the textual surface for authoring functions."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from quod.model import (
    Assign,
    BinOp,
    CharLit,
    Call,
    ExprStmt,
    FieldRead,
    FieldSet,
    For,
    I1Type,
    I8PtrType,
    I8Type,
    I32Type,
    I64Type,
    If,
    IntLit,
    Let,
    Load,
    LocalRef,
    NullPtr,
    ParamRef,
    PtrOffset,
    Return,
    ReturnExpr,
    ReturnInt,
    ShortCircuitAnd,
    ShortCircuitOr,
    Store,
    StringRef,
    StructInit,
    StructType,
    VoidType,
    While,
    Widen,
    WithArena,
)
from quod.script import ScriptError, parse_function, tokenize


# ---------- Tokeniser ----------

def test_tokens_basic_identifiers_keywords_ops():
    toks = tokenize("fn add(a, b) { return a + b }")
    kinds = [(t.kind, t.value) for t in toks if t.kind != "EOF"]
    assert kinds == [
        ("KW", "fn"), ("IDENT", "add"), ("OP", "("),
        ("IDENT", "a"), ("OP", ","), ("IDENT", "b"), ("OP", ")"),
        ("OP", "{"), ("KW", "return"), ("IDENT", "a"), ("OP", "+"),
        ("IDENT", "b"), ("OP", "}"),
    ]


def test_tokens_multi_char_ops():
    toks = [t for t in tokenize("a -> b == c <= d || e && f .. g <u h") if t.kind != "EOF"]
    op_values = [t.value for t in toks if t.kind == "OP"]
    assert op_values == ["->", "==", "<=", "||", "&&", "..", "<u"]


def test_tokens_char_literal_and_escapes():
    toks = tokenize(r"'l' '\n' '\t' '\\' '\''")
    chars = [t.value for t in toks if t.kind == "CHAR"]
    assert chars == ["l", "\n", "\t", "\\", "'"]


def test_tokens_int_and_negative():
    toks = [t for t in tokenize("42 -7") if t.kind != "EOF"]
    assert [(t.kind, t.value) for t in toks] == [
        ("INT", "42"), ("OP", "-"), ("INT", "7"),
    ]


def test_tokens_comment_and_whitespace():
    toks = [t for t in tokenize("# leading comment\n  fn  # trailing\n") if t.kind != "EOF"]
    assert [t.value for t in toks] == ["fn"]


def test_token_unterminated_char_raises():
    with pytest.raises(ScriptError, match="closing"):
        tokenize("'l")


# ---------- Parser: signatures ----------

def test_function_signature_basic():
    fn = parse_function("fn noop() -> i32 { return 0 }")
    assert fn.name == "noop"
    assert fn.params == ()
    assert isinstance(fn.return_type, I32Type)


def test_function_signature_typed_params_and_struct_return():
    fn = parse_function(
        "fn make(a: i64, b: i8*) -> Parser { return load[Parser](b) }"
    )
    assert [p.name for p in fn.params] == ["a", "b"]
    assert isinstance(fn.params[0].type, I64Type)
    assert isinstance(fn.params[1].type, I8PtrType)
    assert isinstance(fn.return_type, StructType) and fn.return_type.name == "Parser"


def test_function_void_return_with_bare_return():
    fn = parse_function("fn yo(p: i8*) -> void { return }")
    assert isinstance(fn.return_type, VoidType)
    assert isinstance(fn.body[0], Return)


def test_void_outside_return_position_rejected():
    with pytest.raises(ScriptError, match="void only valid"):
        parse_function("fn bad() -> i32 { let x: void = 1 return 0 }")


# ---------- Parser: statements ----------

def test_let_with_struct_type_and_load():
    fn = parse_function(
        "fn f(p: i8*) -> i32 { let q: Parser = load[Parser](p) return 0 }"
    )
    let = fn.body[0]
    assert isinstance(let, Let) and let.name == "q"
    assert isinstance(let.type, StructType) and let.type.name == "Parser"
    assert isinstance(let.init, Load)


def test_assign_distinct_from_field_set():
    fn = parse_function(
        "fn f() -> i32 { let x: i32 = 1 x = 2 return x }"
    )
    assert isinstance(fn.body[1], Assign)
    assert fn.body[1].name == "x"


def test_field_set():
    fn = parse_function(
        "fn f(p: i8*) -> i32 { let v: Parser = load[Parser](p) "
        "v.cursor = v.cursor + 1 return 0 }"
    )
    fs = fn.body[1]
    assert isinstance(fs, FieldSet) and fs.local == "v" and fs.name == "cursor"


def test_if_with_else():
    fn = parse_function(
        "fn f(x: i32) -> i32 { if (x == 0) { return 1 } else { return 2 } }"
    )
    iff = fn.body[0]
    assert isinstance(iff, If)
    assert isinstance(iff.then_body[0], ReturnInt) and iff.then_body[0].value == 1
    assert isinstance(iff.else_body[0], ReturnInt) and iff.else_body[0].value == 2


def test_if_without_else():
    fn = parse_function("fn f(x: i32) -> i32 { if (x == 0) { return 1 } return 0 }")
    iff = fn.body[0]
    assert iff.else_body == ()


def test_while_loop():
    fn = parse_function(
        "fn loop_(n: i32) -> i32 { let i: i32 = 0 while (i <  n) { i = i + 1 } return i }"
    )
    w = fn.body[1]
    assert isinstance(w, While)
    assert isinstance(w.cond, BinOp) and w.cond.op == "slt"


def test_for_loop():
    fn = parse_function(
        "fn f(n: i32) -> i32 { let s: i32 = 0 for i: i32 in 0 .. n { s = s + i } return s }"
    )
    fl = fn.body[1]
    assert isinstance(fl, For) and fl.var == "i"
    assert isinstance(fl.type, I32Type)


def test_return_void_vs_int_vs_expr():
    fn = parse_function("fn a() -> void { return }")
    assert isinstance(fn.body[0], Return)

    fn = parse_function("fn a() -> i32 { return 7 }")
    assert isinstance(fn.body[0], ReturnInt) and fn.body[0].value == 7

    fn = parse_function("fn a() -> i32 { return -7 }")
    assert isinstance(fn.body[0], ReturnInt) and fn.body[0].value == -7

    fn = parse_function("fn a(x: i32) -> i32 { return x + 1 }")
    assert isinstance(fn.body[0], ReturnExpr)


def test_store_statement():
    fn = parse_function(
        "fn f(p: i8*) -> i32 { store(p, 65) return 0 }"
    )
    s = fn.body[0]
    assert isinstance(s, Store)
    assert isinstance(s.value, IntLit) and s.value.value == 65


def test_with_arena():
    fn = parse_function(
        "fn f() -> i32 { with_arena a (capacity = 64) { let x: i32 = 1 } return 0 }"
    )
    wa = fn.body[0]
    assert isinstance(wa, WithArena) and wa.name == "a"
    assert isinstance(wa.capacity, IntLit) and wa.capacity.value == 64
    assert len(wa.body) == 1


def test_expression_statement_call():
    fn = parse_function(
        'fn f() -> i32 { printf(&.fmt, 1) return 0 }'
    )
    es = fn.body[0]
    assert isinstance(es, ExprStmt)
    assert isinstance(es.value, Call) and es.value.function == "printf"


# ---------- Parser: expressions ----------

def test_expr_int_lit_default_i64():
    fn = parse_function("fn f() -> i64 { return 42 + 1 }")
    body = fn.body[0]
    assert isinstance(body, ReturnExpr)
    assert isinstance(body.value.lhs, IntLit) and isinstance(body.value.lhs.type, I64Type)


def test_expr_char_lit_and_null_and_bools():
    fn = parse_function(
        "fn f() -> i1 { let c: i8 = 'l' let p: i8* = null let b: i1 = true return false }"
    )
    assert isinstance(fn.body[0].init, CharLit) and fn.body[0].init.value == "l"
    assert isinstance(fn.body[1].init, NullPtr)
    blit = fn.body[2].init
    assert isinstance(blit, IntLit) and isinstance(blit.type, I1Type) and blit.value == 1
    # `return false` sugars to ReturnInt (the model's value-less return
    # shortcut whose width follows the function's declared return_type).
    ret = fn.body[3]
    assert isinstance(ret, ReturnInt) and ret.value == 0


def test_expr_string_ref_dotted():
    fn = parse_function("fn f() -> i32 { printf(&.str.greeting) return 0 }")
    arg = fn.body[0].value.args[0]
    assert isinstance(arg, StringRef) and arg.name == ".str.greeting"


def test_expr_field_read_chained():
    fn = parse_function(
        "fn f(p: i8*) -> i64 { let v: Parser = load[Parser](p) return v.cursor }"
    )
    ret = fn.body[1]
    assert isinstance(ret.value, FieldRead) and ret.value.name == "cursor"


def test_expr_call_with_args_uses_param_ref():
    fn = parse_function(
        "fn f(x: i32) -> i32 { return helper(x, 7) }"
    )
    call = fn.body[0].value
    assert isinstance(call, Call) and call.function == "helper"
    assert isinstance(call.args[0], ParamRef) and call.args[0].name == "x"
    assert isinstance(call.args[1], IntLit)


def test_expr_struct_init_with_trailing_comma():
    fn = parse_function(
        "fn f() -> i32 { let p: Parser = "
        "Parser { input_ptr: null, input_len: 0, cursor: 0, arena: null, had_error: 0, } "
        "return 0 }"
    )
    init = fn.body[0].init
    assert isinstance(init, StructInit) and init.type == "Parser"
    assert [fi.name for fi in init.fields] == [
        "input_ptr", "input_len", "cursor", "arena", "had_error",
    ]


def test_expr_load_widen_uwiden_ptr_offset():
    fn = parse_function(
        "fn f(p: i8*) -> i32 { "
        "let b: i8 = load[i8](ptr_offset(p, 0)) "
        "let w: i32 = uwiden(b to i32) "
        "let s: i32 = widen(b to i32) "
        "return w + s }"
    )
    assert isinstance(fn.body[0].init, Load)
    load = fn.body[0].init
    assert isinstance(load.ptr, PtrOffset)
    uw = fn.body[1].init
    assert isinstance(uw, Widen) and uw.signed is False
    sw = fn.body[2].init
    assert isinstance(sw, Widen) and sw.signed is True


def test_expr_short_circuit_and_or():
    fn = parse_function(
        "fn f(x: i1, y: i1, z: i1) -> i1 { return x || y && z }"
    )
    # `&&` binds tighter than `||`, so the tree is `x || (y && z)`.
    e = fn.body[0].value
    assert isinstance(e, ShortCircuitOr)
    assert isinstance(e.rhs, ShortCircuitAnd)


def test_expr_cmp_and_arithmetic_precedence():
    fn = parse_function("fn f(x: i32) -> i1 { return x + 1 == 2 }")
    e = fn.body[0].value
    assert isinstance(e, BinOp) and e.op == "eq"
    assert isinstance(e.lhs, BinOp) and e.lhs.op == "add"


def test_expr_paren_grouping():
    fn = parse_function("fn f(a: i32, b: i32, c: i32) -> i32 { return (a + b) * c }")
    e = fn.body[0].value
    assert e.op == "mul"
    assert isinstance(e.lhs, BinOp) and e.lhs.op == "add"


# ---------- Param vs local disambiguation ----------

def test_param_names_become_param_ref():
    fn = parse_function(
        "fn f(x: i32) -> i32 { let y: i32 = x + 1 return y }"
    )
    let = fn.body[0]
    add = let.init  # x + 1
    assert isinstance(add.lhs, ParamRef) and add.lhs.name == "x"
    ret = fn.body[1]
    assert isinstance(ret.value, LocalRef) and ret.value.name == "y"


# ---------- Trailing tokens / errors ----------

def test_trailing_tokens_raise():
    with pytest.raises(ScriptError, match="trailing tokens"):
        parse_function("fn f() -> i32 { return 0 } extra")


def test_missing_arrow_raises():
    with pytest.raises(ScriptError):
        parse_function("fn f() i32 { return 0 }")


def test_missing_brace_raises():
    with pytest.raises(ScriptError):
        parse_function("fn f() -> i32 ( return 0 )")


# ---------- End-to-end: author -> build -> run via the real CLI ----------

def _quod(args: list[str], cwd: Path, stdin: str = "") -> subprocess.CompletedProcess:
    """Run the installed quod CLI (via `uv run`) inside `cwd`. Failures
    surface raw stderr so a broken CLI step doesn't silently look like a
    script-parse bug."""
    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        ["uv", "run", "--project", str(repo_root), "quod", *args],
        cwd=cwd, input=stdin, capture_output=True, text=True, check=False,
    )
    return proc


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_end_to_end_authoring(tmp_path: Path):
    """Author a small program entirely through `quod fn add --script ...`,
    then build and run it. Exercises the full handoff: script -> JSON ->
    LLVM lowering -> binary."""
    # Bootstrap from the hello template (gives us a [[program.bin]] entry
    # plus a placeholder main). Replace main with a script-authored body.
    init = _quod(["init", "-t", "hello"], cwd=tmp_path)
    assert init.returncode == 0, init.stderr
    rm = _quod(["fn", "rm", "main"], cwd=tmp_path)
    assert rm.returncode == 0, rm.stderr
    ex = _quod(["extern", "add", "printf", "--param-type", "i8_ptr", "--varargs"], cwd=tmp_path)
    assert ex.returncode == 0, ex.stderr
    co = _quod(["const", "add", ".fmt", "answer=%d\n"], cwd=tmp_path)
    assert co.returncode == 0, co.stderr

    script = (
        'fn main() -> i32 {\n'
        '  printf(&.fmt, 42)\n'
        '  return 0\n'
        '}\n'
    )
    add = _quod(["fn", "add", "--script", script], cwd=tmp_path)
    assert add.returncode == 0, add.stderr

    # Build via uv-spawned CLI; running the produced binary requires the
    # quod runtime archive and clang, both already on PATH for the test
    # suite (this test is a peer of the existing case-driven suite).
    run = _quod(["run"], cwd=tmp_path)
    assert run.returncode == 0, run.stderr
    # `quod run` echoes a build banner first, then `--- NAME ---`, then
    # `stdout: '...'` and `exit:   N`. We just need to confirm the program
    # actually printed our format — a substring check survives banner churn.
    assert "answer=42" in run.stdout, run.stdout
