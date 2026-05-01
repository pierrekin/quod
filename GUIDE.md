# quod — a guided tour

This walks through quod end-to-end: a hello-world program, then a function
with a non-trivial body, then claims and Z3-discharged proofs. Every block
shows the actual command and its actual output.

## What is quod?

quod is a small programming language whose source code IS data. There's
no parser; programs are stored directly as a tree of typed JSON nodes,
and the CLI is how you author and edit them. That tree is lowered to LLVM
IR and from there to a native binary.

Two layers of nodes:

- `llvm.*` — thin wrappers over LLVM IR ops (`llvm.call`, `llvm.binop`,
  `llvm.const_int`, `llvm.param_ref`, …). One node = roughly one IR
  instruction.
- `quod.*` — higher-level sugar (`quod.if`, `quod.while`, `quod.for`,
  `quod.let`, `quod.assign`, `quod.return_int`, `quod.expr_stmt`, …) that
  lowers to multi-step IR — control flow with basic blocks, allocas at the
  entry block for locals, etc.

You write at the `quod.*` layer mostly, and drop to `llvm.*` for the
primitives.

The other distinguishing feature is **claims**: assertions attached to
functions (`non_negative(x)`, `int_range(x, 0, 100)`, `return_in_range`)
that the optimizer can trust. Claims come from three sources:

- **axiom** — you said so. The compiler trusts it. UB if false.
- **witness** — proven by Z3 (`quod claim prove`). Backed by an `.smt2`
  artifact whose hash is stored in the program.
- **lattice** — derived by quod's own fixed-point analysis on each compile.
  Re-derived from scratch every time, never stored.

## 1. Hello world

`quod init` writes two files: `quod.toml` (project config — what's the
program file, what binaries to build) and `program.json` (the program
itself).

```sh
mkdir quod-demo && cd quod-demo
quod init -t hello
```

```
wrote /tmp/demo/quod.toml
wrote /tmp/demo/program.json (hello starter)
```

The `quod.toml` is short and explicit:

```toml
[build]
profile = 2

[[program]]
name    = "hello"
version = "0.1.0"
file    = "program.json"

  [[program.bin]]
  name  = "hello"
  entry = "main"
```

A `quod.toml` lists one or more `[[program]]` entries — each is a quod
program with a `name`, a `version`, and a `file` pointing at its JSON.
Each `[[program.bin]]` is a thing-to-build for that program: `name` is
the output binary filename, `entry` names the function inside the
program JSON to use as the entry point. Multiple programs in one
`quod.toml` give you a workspace; pick one with `quod -p NAME ...`.

The `program.json` itself:

```json
{
    "constants": [
        {"name": ".str.greeting", "value": "hello, world"}
    ],
    "functions": [
        {
            "name": "main",
            "params": [],
            "return_type": {"kind": "llvm.i32"},
            "body": [
                {
                    "kind": "quod.expr_stmt",
                    "value": {
                        "kind": "llvm.call",
                        "function": "puts",
                        "args": [{"kind": "quod.string_ref", "name": ".str.greeting"}]
                    }
                },
                {"kind": "quod.return_int", "value": 0}
            ],
            "claims": []
        }
    ],
    "externs": [
        {"name": "puts", "param_types": [{"kind": "llvm.i8_ptr"}]}
    ]
}
```

That's the whole language: a `Program` with `constants`, `externs`, and
`functions`. Each function declares typed `params` (each a `{name, type}`
node) and a `return_type` chosen from `llvm.i1` / `i8` / `i16` / `i32` /
`i64`. The `body` is a list of statements; `claims` are optional. Every
node has a `kind` discriminator.

## 2. Inspect

`quod show` prints the program in a more readable form. The `[…]` prefixes
are short content-hashes — every node has a stable hash derived from its
content, and the CLI accepts hash prefixes anywhere a name is accepted.

```sh
quod show
```

```
[d39337f21f95]  program {
                  constants:
[cd8f4e38d2c2]      .str.greeting = 'hello, world'
                  externs:
[a271725532b7]      extern puts(i8*) -> i32
                  functions:
[740bab5f21fd]      main() -> i32 {
[e90990997573]        puts(&.str.greeting)
[8220f593d9a1]        return 0
                    }
                }
```

`quod fn ls` lists functions with their signatures:

```sh
quod fn ls
```

```
[740bab5f21fd] main() -> i32
```

`quod fn show NAME` prints just one function (accepts a name or a hash
prefix):

```sh
quod fn show main
```

```
[740bab5f21fd]  main() -> i32 {
[e90990997573]    puts(&.str.greeting)
[8220f593d9a1]    return 0
                }
```

Hashes are content-addressable. If you have a hash from `show` and want
to know what it points to:

```sh
quod find e909
```

```
hash:  e90990997573c55e3418e3600faedc5860a24d865a9825a7e772f040b8b73e08
short:  e90990997573
type:  ExprStmt
json:  {"kind":"quod.expr_stmt","value":{"kind":"llvm.call","function":"puts","args":[...]}}
```

You can also dump every node and its hash with `quod show --hashes`:

```sh
quod show --hashes
```

```
d39337f21f95  Program
cd8f4e38d2c2  StringConstant
740bab5f21fd  Function
79cac65d1b69  I32Type
e90990997573  ExprStmt
55f87d55fb52  Call
adfa48c7f0bc  StringRef
8220f593d9a1  ReturnInt
a271725532b7  ExternFunction
617dda8608f6  I8PtrType
```

## 3. Build and run

`quod build` lowers each program to LLVM IR, optimizes it, emits an
object file, and links a binary — once per `[[program.bin]]` entry in
`quod.toml`. Artifacts are grouped by program name:

```sh
quod build
```

```
[hello/hello] entry=main
  unopt IR -> /tmp/demo/build/hello/hello.unopt.ll
  opt IR   -> /tmp/demo/build/hello/hello.opt.ll
  object   -> /tmp/demo/build/hello/hello.o
  binary   -> /tmp/demo/build/hello/hello
```

All four artifacts are written; you can read the IR if you want to see
what came out of llvmlite. `quod run` is the same as `build`-then-execute:

```sh
quod run
```

```
[hello/hello] entry=main
  unopt IR -> /tmp/demo/build/hello/hello.unopt.ll
  opt IR   -> /tmp/demo/build/hello/hello.opt.ll
  object   -> /tmp/demo/build/hello/hello.o
  binary   -> /tmp/demo/build/hello/hello

--- hello ---
stdout: 'hello, world\n'
exit:   0
```

Build is rooted at `quod.toml` (so `quod -c /elsewhere/quod.toml run`
puts artifacts in `/elsewhere/build/`), but the launched binary inherits
your CWD.

## 4. A program with branches: the `guarded` template

Hello world has nothing to prove or analyze. The `guarded` template gives
us a function with a parameter and a conditional return:

```sh
cd .. && mkdir guarded-demo && cd guarded-demo
quod init -t guarded
quod fn show f
```

```
[b7b7f0d8d936]  f(x: i32) -> i32 {
[1b872194c743]    if ((x < 0)) {
[8f183d93c201]      return -1
                  } else {
[75271b08dd67]      return (x + 1)
                  }
                }
```

`f(x) = if x < 0 then -1 else x + 1`. The function takes a parameter, so
it can't be a binary entry point on its own — the starter `quod.toml` for
`guarded` deliberately has no `[[program.bin]]`. We can still inspect,
validate, and prove things about it.

`quod fn unconstrained` is a helper that lists params with no claims —
useful to see "what does this function not yet know about its inputs":

```sh
quod fn unconstrained
```

```
f.x
```

`x` is unconstrained — `f` accepts any `i32`, including negatives.

## 5. Adding a claim

We can attach a claim that the optimizer will trust. Three kinds:

- `non_negative(param)` — the param is `>= 0`.
- `int_range(param, min, max)` — bounded int.
- `return_in_range(min, max)` — function-scoped, about the return value.

Claim regimes:

- `axiom` — you assert it. Default. Compiled into an `llvm.assume(...)`
  the optimizer trusts. UB if violated.
- `witness` — proven by Z3 and stored with a hash-pinned `.smt2` artifact.
- `lattice` — derived by quod's analysis on each compile.

Add an axiom claim — "I promise `x` is non-negative":

```sh
quod claim add f non_negative x
quod claim ls
```

```
added non_negative(x) on f [regime=axiom, enforcement=trust]
f: non_negative(x)
```

If a future version of `f` is called with `x = -5`, the program is wrong:
the optimizer will have eliminated the `x < 0` branch entirely on the
strength of the claim, and behavior is undefined.

To remove a claim:

```sh
quod claim relax f non_negative x
```

```
relaxed non_negative(x) on f
```

## 6. Proving a claim instead of asserting it

`axiom` is "trust me." `witness` is "I'll prove it." `quod claim prove`
generates an SMT-LIB encoding of the function and the goal, ships it to
Z3, and on success attaches the resulting `.smt2` artifact as evidence.

For `f`, can we prove the return value is always `>= -1`?

- If `x < 0`: returns `-1`. ✓
- Otherwise: returns `x + 1`, which is `>= 1` since `x >= 0`. ✓

So `return_in_range(min=-1)` should hold:

```sh
quod claim prove f return_in_range --min -1
```

```
proved return_in_range([-1, +inf]) {regime=witness, justification=z3(/tmp/demo3/proofs/guarded/f_return_in_range_return_e7b5f9f88a3d.smt2@e7b5f9f88a3d)} via z3.qf_lia
  artifact: /tmp/demo3/proofs/guarded/f_return_in_range_return_e7b5f9f88a3d.smt2 (sha256=e7b5f9f88a3d)
```

Proof artifacts are written under `<proofs_dir>/<program-name>/`
(`proofs_dir` defaults to `proofs/`, configurable in `quod.toml`).

The `.smt2` file is real — you can open it and read the encoding Z3
solved. Its sha256 is now baked into the program:

```sh
quod claim ls
```

```
f: return_in_range([-1, +inf]) {regime=witness, justification=z3(/tmp/demo3/proofs/guarded/f_return_in_range_return_e7b5f9f88a3d.smt2@e7b5f9f88a3d)}
```

If you try to prove something that isn't true, Z3 returns `sat` (a model
exists where the claim fails), and the prover refuses:

```sh
quod claim prove f return_in_range --min 100
```

```
could not prove return_in_range: z3.qf_lia reported refuted (z3 returned 'sat')
(provider found a counterexample; the claim does not hold)
```

`quod claim verify` re-runs every stored proof — re-hashes the artifact
file and re-runs Z3 to confirm `unsat`:

```sh
quod claim verify
```

```
ok   f: return_in_range([-1, +inf]) {regime=witness, justification=z3(...)}
```

## 7. Other useful things

A few commands that round out the tour:

- `quod claim suggest` — speculatively compiles candidate claims and
  reports which ones would shrink optimized IR if true. A scout for what's
  worth proving.
- `quod claim derive` — runs the lattice analysis and prints the derived
  (`regime=lattice`) claims. Re-derived every compile, never stored.
- `quod fn callers TARGET` — every call site of `TARGET`.
- `quod fn data-flow FN PARAM` — where in `FN` does `PARAM` get read.
- `quod fn call-graph` — caller→callees, with `@extern` and `!` markers.
- `quod note add FN TEXT` — attach free-form notes to a function. Pure
  metadata; doesn't affect codegen.
- `quod stmt add FN SPEC --at-end` — append a statement (read JSON spec
  from stdin or a path).
- `quod fn add SPEC` — append a whole function.
- `quod extern add NAME --param-type i8_ptr --varargs` — declare a libc
  symbol like `printf`.

## 8. The CLI tree at a glance

```
quod init                           # write quod.toml + program.json
quod ingest SOURCE                  # ingest a C source file into a fresh project
quod -p NAME ...                    # select a [[program]] (omit if only one)
quod check                          # parse, lower, LLVM-verify
quod build [--profile N] [--show-ir]  # → object → linked binary, per [[program.bin]]
quod run [--bin NAME] [-- ARGS...]  # build then exec
quod show [--hashes]                # whole program
quod find PREFIX                    # resolve a hash prefix
quod schema [--category C | KIND]   # node-shape introspection

quod fn ls / show REF / add - / rm REF
quod fn callers TARGET
quod fn data-flow FN PARAM
quod fn call-graph
quod fn unconstrained

quod claim ls [FN]
quod claim add FN KIND [TARGET]   [--min N] [--max N]
                                  [--regime axiom|witness] [--enforcement trust|verify]
                                  [--justification '<json>']
quod claim relax FN KIND [TARGET]
quod claim prove FN KIND [TARGET] [--min N] [--max N]
quod claim verify
quod claim suggest [--top-n N]
quod claim derive

quod stmt add FN - [--at-end | --at-start | --before HASH | --after HASH]
quod stmt rm FN HASH_PREFIX

quod const ls / add NAME VALUE / rm NAME

quod extern ls
quod extern add NAME [--arity N | --param-type T ...] [--return-type T] [--varargs]
quod extern rm NAME

quod note add FN TEXT
quod note rm FN INDEX

quod provider ls                    # registered claim providers (regime + modes)
```

`fn add` and `stmt add` read the JSON spec from stdin (`-`).
