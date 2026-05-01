---
name: quod
description: Use when working with quod — a small programming language whose source is a JSON tree of typed nodes (no parser), lowered to LLVM IR and a native binary. Triggers when the user references the `quod` CLI, files named `quod.toml` or `program.json`, claims/proofs/Z3, or asks to author, build, run, or optimize a quod program.
allowed-tools: Bash(quod *) Bash(uv run quod *) Read Write Edit
---

# quod

quod is a programming language whose source code IS data. There is no
textual syntax — programs are stored as a JSON tree of typed nodes
(`program.json`), and the `quod` CLI is how you author and edit them.
The tree lowers to LLVM IR and from there to a native binary. The
distinguishing feature is **claims** (`non_negative`, `int_range`,
`return_in_range`) attached to functions, which can be discharged by
Z3 and trusted by the optimizer.

A quod project is any directory containing `quod.toml`. All commands
read project state from there.

## Two layers of nodes

- `llvm.*` — thin wrappers over LLVM IR ops (`llvm.call`, `llvm.binop`,
  `llvm.const_int`, `llvm.param_ref`, …). One node ≈ one IR instruction.
- `quod.*` — higher-level sugar (`quod.if`, `quod.while`, `quod.for`,
  `quod.let`, `quod.assign`, `quod.return_int`, `quod.expr_stmt`, …)
  that lowers to multi-step IR with basic blocks and entry-block allocas.

Author at the `quod.*` layer mostly; drop to `llvm.*` for primitives.

## Three claim regimes

- **axiom** — you assert it; the compiler trusts it; UB if false.
- **witness** — proven by Z3; backed by a hash-pinned `.smt2` artifact.
- **lattice** — derived by quod's fixed-point analysis on each compile;
  re-derived from scratch every time, never stored.

## Always start with the schema

Before constructing any JSON node, run `quod schema` to discover the
exact shape — this avoids round-trips that fail validation:

```sh
quod schema                              # list categories
quod schema --category statement         # kinds in a category
quod schema quod.let                     # full schema + minimal example
quod schema Function                     # whole-function shape
```

## CLI tree at a glance

```
quod init -t {hello|guarded|empty}    # writes quod.toml + program.json
quod ingest SOURCE [-n NAME]          # ingest a C source file into a fresh project
quod -p NAME ...                      # select a [[program]] (omit if only one)
quod check                            # parse, lower, LLVM-verify (no artifacts)
quod build [--profile N] [--show-ir]  # → object → linked binary, per [[program.bin]]
quod run [--bin NAME] [-- ARGS...]    # build then exec
quod show [--hashes]                  # whole program, canonical form
quod find PREFIX                      # resolve a content-hash prefix
quod schema [--category C | KIND]     # node shapes

quod fn ls / show REF / add - / rm REF
quod fn callers TARGET
quod fn data-flow FN PARAM
quod fn call-graph
quod fn unconstrained                 # params with no claim attached

quod claim ls [FN]
quod claim add   FN KIND [TARGET] [--min N] [--max N] [--regime ...] [--enforcement ...]
quod claim relax FN KIND [TARGET]
quod claim prove FN KIND [TARGET] [--min N] [--max N]
quod claim verify                     # re-hash + re-run every stored proof
quod claim suggest [--top-n N]        # rank candidate claims by codegen impact
quod claim derive                     # show lattice-derived claims

quod stmt add FN - [--at-end | --at-start | --before HASH | --after HASH]
quod stmt rm FN HASH_PREFIX

quod const ls / add NAME VALUE / rm NAME
quod extern ls / add NAME [--arity N | --param-type T ...] [--return-type T] [--varargs]
quod extern rm NAME

quod note add FN TEXT
quod note rm  FN INDEX

quod provider ls                      # registered claim providers (regime + modes)
```

For `fn add` and `stmt add`, pass JSON on stdin (`-`).

## Authoring workflows

### Starting a new project

`quod init` writes `quod.toml` and `program.json`. Pick the template by
goal:

- `hello` — runnable hello-world with `puts`. Use to demonstrate end-to-end.
- `guarded` — a function `f(x: i32)` with a conditional return; deliberately
  has no `[[program.bin]]`. Use as a claim/proof playground.
- `empty` — blank slate.

Typical next step depends on the template: `quod show` to inspect,
`quod run` for hello, `quod fn unconstrained` → `quod claim suggest`
for guarded.

### Adding code

Hashes are content-addressable. `quod show` prints them as `[abc123]`
prefixes; the CLI accepts any unique prefix anywhere a name is taken.

To add a function or statement:

1. `quod schema Function` (or `quod schema --category statement`) to
   get the canonical shape.
2. Construct the JSON.
3. `quod fn add -` (stdin) or `quod stmt add FN - --at-end` (or
   `--before HASH` / `--after HASH` for precise placement).
4. `quod check` to validate.

Removals (`fn rm`, `stmt rm`, `extern rm`, `const rm`) are permissive
— they don't refuse if other code still references the target. The
dangling reference surfaces at build time. Use `quod fn callers` first
to gauge blast radius.

### Building and running

`quod build` lowers → optimizes → emits objects → links a binary for
every `[[program.bin]]` in `quod.toml`. With multiple `[[program]]`
entries, all of them are built unless you pass `quod -p NAME ...` to
narrow it. `quod run` is build-then-execute.

For an entry function with `int` params, pass them via `quod run -- ARGS...`
— the synthesized `main` wrapper parses each via `atoll`, then trunc/sext's
to the param's width.

Use `quod check` as a fast sanity check after edits — no artifacts emitted.

## Claim / proof workflows

### When the user asserts a fact about inputs

If the user has stated something like "x is always non-negative", attach
an axiom claim:

```sh
quod claim add FN non_negative x
```

The optimizer trusts axiom claims unconditionally — UB at runtime if
violated. **Prefer `quod claim prove` over `quod claim add` when the
claim should be derivable from the function body.** Axioms are
trust-me; witnesses are proven.

Claim-add gotcha: `non_negative` and `int_range` need a `target`
parameter; `return_in_range` must omit it (it's function-scoped).

### Proving a claim with Z3

```sh
quod claim prove FN KIND [TARGET] [--min N] [--max N]
```

On `unsat` → success: a `.smt2` artifact is written under `proofs/` and
the claim is attached as `regime=witness` with the artifact's sha256
pinned.

On `sat` → Z3 found a counterexample; the claim is FALSE. Do NOT fall
back to `quod claim add` as axiom — revisit the claim or the function.

On `unknown` or `NotImplementedError` → the claim is beyond the current
SMT lowering (mutable locals, srem, unsigned cmps). Refactor the
function into a pure-expression form, or skip proving that particular
claim.

After editing a function with witness claims, run `quod claim verify`
— it re-hashes each artifact and re-runs Z3 to confirm `unsat`. If a
proof breaks, re-prove with `quod claim prove` or relax the claim.

### Optimizing a quod program

The "make it faster" / "shrink the IR" workflow:

1. `quod claim suggest [--top-n N]` — speculatively compiles candidate
   claims and reports which ones would shrink optimized IR if proven.
   Read-only; doesn't mutate the program.
2. For each suggestion that should genuinely hold, run `quod claim prove`.
3. `quod build --show-ir` to confirm the optimizer used the new claim.

If `quod claim suggest` reports nothing, the codegen is already tight or
the candidate set is exhausted. Not every program benefits from
claim-driven optimization.

## Common pitfalls

- **Skipping `quod schema`** before constructing JSON. The schemas are
  precise and short — read them.
- **Asserting axioms instead of proving witnesses.** Every axiom is a
  trust hole. Prefer `quod claim prove` whenever the function body
  could discharge the claim.
- **Forgetting `quod claim verify` after edits.** Witness proofs are
  pinned to function-body hashes; editing the body invalidates them.
- **Wrong target shape on `claim add`.** `return_in_range` is
  function-scoped (no target); `non_negative` / `int_range` need a
  parameter name.

## Project layout in this repo

The quod source lives at `src/quod/` (Typer CLI, Pydantic node types,
LLVM lowering, SMT-LIB encoding). Examples are under `examples/`,
grouped by topic (`basics/`, `claims/`, `project_euler/`, …). The
authoritative end-to-end tour is `GUIDE.md`.
