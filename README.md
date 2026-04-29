# quod

A small programming-language toolchain. Programs are represented as JSON
trees of typed nodes (no textual source); the CLI builds, inspects, and
compiles them through LLVM. Claims attached to functions can be discharged
via Z3 and exploited by the optimizer.

For a tour of what quod is and how to use it, see [GUIDE.md](GUIDE.md).

## Prerequisites

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) — for env / dependency management
- `clang` — used as the linker driver
- `z3` — optional, only needed for `quod claim prove` / `quod claim verify`

## Setup

```sh
git clone <this-repo> quod && cd quod
uv sync
```

## Running the CLI

The CLI entry point is `quod` (declared in `pyproject.toml` as a
`[project.scripts]`).

```sh
uv run quod --help
uv run quod init -t hello
uv run quod run
```

## Project layout

```
src/quod/
    cli.py          Typer CLI (noun-first sub-apps)
    config.py       quod.toml loader
    model.py        Pydantic node types (the Program AST)
    editor.py       Mutation primitives (add/replace/insert nodes)
    hashing.py      Content-addressable node hashes
    analysis.py     Call graph, data flow, lattice claim derivation
    lower.py        Program → LLVM IR → object/binary
    proof.py        SMT-LIB lowering for `quod claim prove`
    templates.py    Starter programs `quod init` can write
examples/           Hand-rolled program.json files for various features
```

## A taste

```sh
mkdir hello && cd hello
uv run quod init -t hello
uv run quod run
# stdout: 'hello, world\n'
# exit:   0
```

quod programs are JSON. The `program.json` for hello-world contains a
`main` function whose body is a `puts` call wrapped in `quod.expr_stmt`,
followed by `quod.return_int 0`. There is no parser — you author the tree
directly (or via the CLI's mutation commands), and quod lowers it to
LLVM IR.

The interesting part is **claims**: assertions attached to functions
(`non_negative(x)`, `int_range(x, 0, 100)`, `return_in_range`, …) which
the optimizer will trust. Claims can be added as axioms (you assert, the
compiler trusts), proven via Z3 (`quod claim prove`), or derived from a
fixed-point lattice analysis. See [GUIDE.md](GUIDE.md) for the full walk.
