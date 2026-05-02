# C ingestion examples

Each subdirectory pairs a C source file with the quod program that `quod
ingest` produces from it.

```
hello/
  hello.c        ← input: the original C source
  program.json   ← output: what `quod ingest hello.c` writes
```

The `.c` file is the source of truth. The `program.json` is committed so
you can inspect the result without re-running ingest:

```sh
# from the examples/ directory:
quod -p c_hello show          # see the ingested program
quod -p c_hello run           # build and run the binary
```

To regenerate a `program.json` after editing a `.c` file, delete the old
one and re-run ingest from inside the example dir:

```sh
cd c_ingest/fizzbuzz
rm program.json quod.toml
quod ingest fizzbuzz.c
```

(Then copy the new `program.json` back over the committed one. `quod
ingest` always writes a sibling `quod.toml`; delete it before committing
since the umbrella `examples/quod.toml` is what wires these into the
workspace.)

## What each example demonstrates

| Example         | Constructs                                                  |
| --------------- | ----------------------------------------------------------- |
| `hello`         | Minimal `printf`. String literal + variadic extern.         |
| `arithmetic`    | User functions, params, calls between user functions.       |
| `control_flow`  | `if` / `else if` / `else`, comparisons, `&&`, i1→int widen. |
| `loops`         | `while`, `int` locals, assignment.                          |
| `fizzbuzz`      | Loops + nested `if/else if/else` + `%` + mixed `printf`.    |
| `string_offset` | `char*` pointer arithmetic — `p + n` and `&p[n]` patterns.  |
| `curl_fetch`    | Pointer locals, opaque handles, enum constants, `[link]`.   |

### Note on `curl_fetch`

This one's standalone — it has its own `quod.toml` and isn't in the
umbrella `examples/quod.toml`. Two reasons:

1. It needs libcurl at link time (`[link] libraries = ["curl"]`), and we
   don't want every other example dragging that dependency in.
2. The C source has a `#define CURL_DISABLE_TYPECHECK` before the include
   to disable libcurl's GCC-statement-expression typecheck macros, which
   our ingester can't walk. Plain function calls work fine.

Run it with:

```sh
cd examples/c_ingest/curl_fetch
quod run
```

## v1 subset reminder

Only `int`-typed function params/returns are ingested. Locals may also be
pointer-typed (mapped to `i8*`). Pointer arithmetic is supported on
`char*` bases (byte stride matches quod's GEP) with literal offsets.
Structs, floats, `unsigned`, `long`, `short`, multi-dimensional arrays,
`for` loops, `goto`, dereference (`*p`), and `switch` are refused with a
clear error pointing at the offending source location. See
`src/quod/ingest/c.py` for the full list of supported AST kinds.
