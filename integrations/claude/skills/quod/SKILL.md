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
  `quod.let`, `quod.assign`, `quod.return_expr`, `quod.return`,
  `quod.expr_stmt`, `quod.match`, `quod.with_arena`, `quod.try`,
  `quod.struct_init`, `quod.enum_init`, `quod.field_set`, `quod.store`,
  …) that lowers to multi-step IR with basic blocks and entry-block
  allocas.

Author at the `quod.*` layer mostly; drop to `llvm.*` for primitives.

Structured types: programs may define `StructDef`s (records, by-value)
and `EnumDef`s (tagged unions, one variant per arm). 2-variant enums
where exactly one variant has a single payload field and the other has
none are `?`-eligible — postfix `?` propagates the sad variant up.

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
quod schema EnumDef                      # whole-enum shape
```

`quod <verb> --help` is the authoritative reference for every flag
(LANGUAGE.md and GUIDE.md don't repeat the help text). When in doubt
about a flag's semantics, ask the CLI.

## Stdlib tiers

Programs declare `imports` (top-level array). Modules merge in at
build time, first-wins by name:

- `core.*` — pure quod, no runtime deps. Always available.
- `alloc.*` — needs the arena allocator. Disable with `--no-alloc`.
- `std.*` — needs hosted OS / libc. Disable with `--no-std`.

`--no-alloc` also refuses `with_arena` (it desugars to alloc-tier
externs). Common modules: `core.bytes`, `core.str`, `alloc.arena`,
`alloc.str`, `alloc.json`, `std.io`. See LANGUAGE.md for the list and
their entry points; `quod -f src/quod/stdlib/<name>.json show` opens
any module standalone.

## CLI tree at a glance

Globals (apply before any subcommand):

```
quod -c PATH ...     # quod.toml at non-default path
quod -p NAME ...     # select a [[program]] in a workspace
quod -f PATH ...     # bypass quod.toml; inspect a standalone program.json
quod --no-color ...  # ANSI off
```

Lifecycle / inspection:

```
quod init -t {hello|guarded|empty} [--force]
quod ingest SOURCE [-n NAME] [--import MOD]...
quod check
quod build [--profile N] [--target T] [--link/--no-link] [--show-ir]
           [--enforce-axiom V] [--enforce-witness V] [--enforce-lattice V]
           [--no-std] [--no-alloc]
quod run   [--bin NAME] [-- ARGS...]
quod show  [--hashes] [--json]
quod find  PREFIX [--json]
quod schema [--category C | KIND]
```

Functions, claims, statements:

```
quod fn ls / show REF / add SPEC / add --script "..." / add --script-file F / rename OLD NEW / rm REF
quod fn callers TARGET
quod fn data-flow FN PARAM
quod fn call-graph
quod fn unconstrained

quod claim ls [FN]
quod claim add   FN KIND [TARGET] [--min N] [--max N] [--regime ...] [--enforcement ...]
quod claim relax FN KIND [TARGET]
quod claim prove FN KIND [TARGET] [--min N] [--max N] [--provider NAME]
quod claim verify
quod claim suggest [--top-n N]
quod claim derive

quod stmt add FN - [--at-end | --at-start | --before HASH | --after HASH]
quod stmt rm FN HASH_PREFIX
```

Top-level declarations:

```
quod const  ls / add NAME VALUE / rm NAME / rename OLD NEW
quod struct ls / show NAME / add NAME FIELDS... / rm NAME / rename OLD NEW
quod enum   ls / show NAME / add SPEC / rm NAME / rename OLD NEW / rename-variant ENUM OLD NEW
quod extern ls / add NAME [...] / rm NAME / ingest HEADER

quod note add FN TEXT
quod note rm  FN INDEX

quod provider ls
```

For `fn add`, `stmt add`, `enum add`, pass JSON on stdin (`-`) or a
path. For function bodies of any complexity, prefer
`quod fn add --script "fn ... { ... }"` — see "Authoring code" below.

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

### Authoring code

Hashes are content-addressable. `quod show` prints them as `[abc123]`
prefixes; the CLI accepts any unique prefix anywhere a name is taken.

**Whole functions — preferred:** quod-script via
`quod fn add --script "fn name(p: T, ...) -> T { ... }"`. Compact
textual surface that emits the same JSON nodes. The grammar covers:

- statements: `let` / `if`/`else` / `while` / `for X: T in lo..hi` /
  `return [expr]` / `store(p, v)` / `match expr { Enum::Variant a, b
  { ... } _ { ... } }` / `with_arena name (capacity = N) { ... }`.
- expressions: int/char/`null`/`true`/`false` literals (with width
  suffixes `0i8`/`42i32`), `&.const_name`, field reads, calls (dotted
  names like `core.str.eq`), `Struct { f: e, ... }`,
  `Enum::Variant { f: e, ... }`, postfix `?`, `sizeof[T]`,
  `load[T](p)`, `widen(e to T)` / `uwiden(e to T)`,
  `ptr_offset(b, o)`, all binops, `&&` / `||`.

Use `--script-file path` (or `-` for stdin) for longer bodies. Struct
and enum literals are disabled in `if`/`while`/`for` cond positions to
avoid `{`-ambiguity — wrap in parens if needed:
`if ((Parser { … }).had_error == 0) { ... }`.

Out of script's scope: claims, struct/enum/extern/const/import
declarations. Use the dedicated CLI verbs for those.

**Whole functions — JSON:** `quod schema Function` → construct JSON →
`quod fn add -` (stdin). Use this when you want claims/notes attached
inline.

**Editing one statement:** the script surface is whole-function only.
For statement-level edits use `quod stmt add FN - --at-end` (or
`--at-start` / `--before HASH` / `--after HASH`) with a JSON spec on
stdin. `quod schema --category statement` for valid kinds.

**Top-level declarations:** structs, enums, externs, string
constants, and notes have their own verbs (`struct add NAME f1:t1 f2:t2`,
`enum add -` with a JSON `EnumDef`, etc.). Use `rename` / `rename-variant`
to refactor — they update every reference in the program. Use
`quod fn callers TARGET` and `quod show` to gauge blast radius before
removing things.

Removals (`fn rm`, `stmt rm`, `extern rm`, `const rm`) are permissive
— they don't refuse if other code still references the target.
`struct rm` and `enum rm` are strict — they refuse if anything still
references the type.

### Building and running

`quod build` lowers → optimizes → emits objects → links a binary for
every `[[program.bin]]` in `quod.toml`. With multiple `[[program]]`
entries, all of them are built unless you pass `quod -p NAME ...` to
narrow it. `quod run` is build-then-execute.

Useful build flags (see `quod build --help` for the full list):
`--profile N` (LLVM opt level 0..3, 0 skips optimize), `--show-ir`
(print optimized IR), `--enforce-axiom verify` (etc., turn `trust` →
runtime branch + abort for that regime), `--no-std` /
`--no-alloc` (refuse to resolve those tier imports).

For an entry function with `int` params, pass them via
`quod run -- ARGS...` — the synthesized `main` wrapper parses each
via `atoll`, then trunc/sext's to the param's width.

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

## Where to look for more

- **`GUIDE.md`** — end-to-end tour with real CLI sessions. Read first
  for hello-world + first proof.
- **`LANGUAGE.md`** — programmer-facing reference. Types (incl. enums),
  expressions (incl. `?`/`sizeof`/struct+enum init), statements (incl.
  `match`/`with_arena`), claims, the three stdlib tiers with module
  contents, the script grammar, the CLI command-by-command. Reach for
  this when authoring real programs.
- **`DEVELOPING.md`** — internals + extension points. Module layout,
  lowering pipeline, claim providers, stdlib resolution, runtime
  archive, how to add a node kind / CLI command / stdlib module /
  provider, the case-driven test harness, hashing.
- **`quod <verb> --help`** — authoritative for flags and arguments.

The quod source lives at `src/quod/` (Typer CLI, Pydantic node types,
LLVM lowering, SMT-LIB encoding). Examples are under `examples/`,
grouped by topic (`basics/`, `claims/`, `json_v3/`, `project_euler/`,
…); `examples/json_demo/` is the canonical "consumer of the stdlib"
program.
