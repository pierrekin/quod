# Developing quod

A guide for people working on the quod toolchain itself. Audience: us.
Read `GUIDE.md` and `LANGUAGE.md` first if you want the user-facing
picture; this document is internals + extension points.

## Project layout

```
src/quod/
    __init__.py
    cli.py             Typer CLI; noun-first sub-apps; one leaf per tool call
    config.py          quod.toml loader (no implicit defaults, no walk-up)
    templates.py       Programs that `quod init` can write
    model.py           Pydantic node types — the Program AST
    schema.py          Node-shape introspection (powers `quod schema`)
    hashing.py         Content-addressable node hashes (Merkle-style)
    editor.py          Mutation primitives over Program (used by CLI)
    analysis.py        Call graph, data flow, lattice claim derivation
    proof.py           SMT-LIB lowering for `quod claim prove`
    providers.py       Pluggable claim providers (lattice / Z3)
    lower.py           Program → LLVM IR → object/binary
    runtime.py         Compiles the C runtime archive (libquodrt-vN.a)
    runtime/
        quod_arena.c   Bump allocator runtime helper
    stdlib.py          Import resolution + tier classification
    stdlib/
        core.bytes.json
        core.str.json
        alloc.arena.json
        alloc.str.json
        alloc.json.json   ← JSON parser, the keystone module
        std.io.json
    ingest/
        __init__.py
        c.py           libclang → Program for the int-only C subset
    script.py          quod-script: textual surface → Program nodes
    render.py          Spans + theme-driven syntax highlighting
    completion.py      Semantic shell completion
examples/              Each subdir is its own [[program]] in examples/quod.toml
tests/                 Case-driven; *.json under tests/cases/ becomes a test
    conftest.py        Auto-collects every JSON case
    cases/{cli,lang,analysis,c_ingest}/...
integrations/
    pi/                pi-coding-agent extension exposing quod as tools
    claude/            Claude Code plugin (skill + slash commands)
```

Architectural rule, in order of strictness:

- `model.py` knows nothing about LLVM IR or SMT.
- `proof.py` knows nothing about IR.
- `lower.py` knows nothing about claim provenance (it sees regimes and
  enforcements, never `Z3Justification`).
- `analysis.py` and `proof.py` are sibling consumers of `model.py`;
  neither imports the other.
- `providers.py` is where they meet; it's the only place that decides
  "this regime + mode + goal → that callable."

When you add code, ask: which module owns it? If the answer is "more
than one," look harder.

## The graph is the asset

Every node in `model.py` is a frozen Pydantic model. To "edit" a node,
build a new one via `model_copy(update={...})` — never mutate in place.
This guarantees:

- Content hashes (`hashing.node_hash`) are stable.
- `editor.py` operations compose without aliasing surprises.
- The on-disk JSON round-trips through Pydantic validation cleanly.

Pydantic discriminator pattern: each node has a `kind: Literal[...]`
field; the `Expr`, `Statement`, `Type`, `ReturnType`, `Claim`, and
`Justification` unions are `Annotated[Union[...], Field(discriminator="kind")]`.
Adding a new kind means adding it to the right union(s).

`InputProgram` and `Program` share the same shape (both extend
`_ProgramBase`). `InputProgram` is what `_load_module` reads from disk;
`Program` is what `compile_program` consumes. The split is the seam
for input-only fields (e.g. imports that don't appear in the resolved
program).

## The lowering pipeline

`compile_program(program, build_dir, bins, ...)` orchestrates:

1. **`--no-alloc` precheck** — if the alloc tier is disabled, refuse
   early on any `with_arena` use.
2. **`resolve_imports(program, disabled_tiers=...)`** — fold stdlib
   modules into the program (first-wins by name); clear
   `program.imports`. Tier-disabled imports are an error.
3. **`elaborate(program, derive_lattice_claims(program))`** — run the
   lattice analysis; merge derived claims into each function's
   `claims` tuple with `regime=lattice` and a `DerivedJustification`.
4. **Per `[[program.bin]]` entry, `lower(...)`** lowers the (resolved,
   elaborated) program to an `ir.Module`:
   - register every struct/enum as an LLVM identified type;
   - declare every user function and extern;
   - lower each function body — `_lower_function_body` walks the
     statement tree, allocas every local at the function entry block,
     and emits IR via `_lower_stmt` / `_lower_expr`;
   - emit the synthesized `main` wrapper if this bin's entry isn't
     already named `main` (decodes argv via `atoll`, trunc/sext to
     each int param's actual width).
5. **`parse_and_verify(module)`** — `llvmlite.binding.parse_assembly`,
   then `verify`. Failures here are bugs in our lowering — every
   internal-consistency error should die before user code runs.
6. **`optimize_module(parsed, target_machine, speed_level=profile)`**
   — runs the LLVM new pass manager at the requested `-O`. Skipped at
   profile 0.
7. **`target_machine.emit_object(parsed)`** → `<bin>.o`.
8. **`build_runtime_archive(build_dir, target=target)`** — compiles
   `runtime/*.c` into `libquodrt-<TAG>.a` if stale (cached by mtime);
   linked unconditionally. Archive linking is by-reference, so unused
   runtime code stays stripped.
9. **`clang -target ... <obj> <archive> -o <bin>`** for each bin.

Artifacts (`<bin>.unopt.ll`, `<bin>.opt.ll`, `<bin>.o`, `<bin>`) are
returned in a `BinResult`.

### Why a static archive for the runtime?

A bare `.o` would always pull every runtime symbol into the binary.
A static `.a` is by-reference — programs that don't call
`quod_arena_alloc` don't drag the arena code into their image.

The archive is rebuilt per-program inside `build_dir`, not once at
install time, because the runtime has to match `--target`; the
install-time host triple is wrong for cross-compiles.

`_ARCHIVE_TAG` is encoded into the archive filename and bumps with
any runtime ABI change so old caches invalidate without a manual
`rm -rf build`.

## Claims: the three regimes plus enforcement

`Regime` is the *epistemic* source of a claim:

- `axiom` — programmer assertion; no machine evidence.
- `witness` — proof exists out-of-band (e.g. a Z3 `.smt2` artifact).
- `lattice` — derived by an analysis pass each compile.

`Enforcement` is the *runtime* consequence if the predicate is false:

- `trust` — lowered to `llvm.assume(predicate)`. Falsity = UB.
- `verify` — lowered to a runtime branch + `abort()`. Falsity = abort.

Regime and enforcement are independent. You can build with `--enforce-axiom verify`
to ship the optimizer-helpful axioms while keeping the safety net.
This is the wedge that makes `regime=lattice` interesting: it says "I
trust my own analysis to derive this; check me at runtime if I'm
wrong."

`_emit_for_enforcement` is the lowering site for both modes. `_lower_claim`
is its caller — it converts a `Claim` into the boolean predicate(s) and
delegates.

### Claim providers

`providers.py` owns the registry. A `Provider` has:

- `name` — registry key.
- `regime` — the regime it produces claims under.
- `description` — what `quod provider ls` shows.
- `derive: (Program) -> {fn: (Claim, ...)}` — batch mode.
- `prove: (Program, ClaimRequest, proofs_dir) -> ProviderResult` — goal mode.

Either or both. The built-in registry contains:

- `lattice.literal_range` (regime=lattice, mode=derive) — wraps
  `analysis.derive_lattice_claims`.
- `z3.qf_lia` (regime=witness, mode=prove) — wraps `proof.goal_smt_lib`
  + `proof.run_z3_on_smt`. On `unsat` writes a `.smt2` artifact under
  `<proofs_dir>/<program-name>/`, sha256-hashes it, attaches the
  hash to the `Z3Justification`. On `sat`/`unknown` the artifact is
  still written (audit trail) but no claim is attached.

The CLI routes via `default_for(regime, mode)` (first match), or
`get_provider(name)` if `--provider` is set.

### To add a new provider

1. Implement `derive` and/or `prove` against `model.Program` and
   `model.Claim`. Keep the implementation in its own module if it has
   non-trivial deps (e.g. a SMT lib other than Z3).
2. Construct a `Provider(...)` instance; add it to `_BUILT_IN` in
   `providers.py`.
3. `quod provider ls` should pick it up automatically — the registry
   is built from `_BUILT_IN`.
4. `quod claim prove` and `quod claim derive` route to the first
   provider matching `(regime, mode)`. If your provider should be
   default for that pair, put it ahead of the others in `_BUILT_IN`;
   if not, expose it via `--provider NAME` only.

The registry is built from `_BUILT_IN`. External provider discovery
(Python entry points, `[[provider]]` subprocess specs in `quod.toml`)
is unimplemented; the comment in `all_providers()` is the design
seam.

## stdlib resolution

`stdlib.py` has two responsibilities: classifying a module by its
top-level namespace into a tier (`core` / `alloc` / `std`), and
recursively merging an import set into a Program before lowering.

- Tier classification is just `name.split(".", 1)[0]`. Anything
  outside the three tiers is treated as `core` for permissions
  (so user-coded modules in non-stdlib dirs can't escape via tiering).
- Resolution is a BFS over `program.imports`, loading each module via
  `_load_module(name)` (which validates as `InputProgram`, so module
  files pass the same validators user code does), queueing nested
  imports, and merging constants/structs/enums/externs/functions into
  the user program with first-wins-by-name. User-declared items always
  shadow imports.
- After resolution, `program.imports` is empty and the resolved program
  is structurally identical to a flat user program. `save_program` on
  a resolved program would inline the stdlib into the user file — don't.

### To add a stdlib module

1. Pick a tier: pure quod / no externs → `core.*`; needs the arena → `alloc.*`;
   needs OS / libc → `std.*`. The namespace IS the tier (`module_tier`
   reads the first segment).
2. Author the module as a JSON `InputProgram` under
   `src/quod/stdlib/<name>.json`. It can declare its own
   `imports`, `constants`, `structs`, `enums`, `externs`, and
   `functions`. Validation runs at load via `InputProgram.model_validate_json`,
   so structural errors are caught at the boundary.
3. **Name collisions are silent.** Pick fully-qualified names
   (`alloc.json.parse`, not `parse`) so user code that imports the
   module can't accidentally shadow a name that another module also
   uses.
4. If the module requires runtime helpers, declare the externs in this
   JSON file and arrange the runtime to provide them (see
   `runtime/quod_arena.c` for the model).
5. There's no test runner specific to stdlib modules — write a
   `tests/cases/lang/...` case that imports the module and exercises
   the entry points end-to-end. The conftest auto-collects it.

To inspect a stdlib module without a project, use `quod -f
src/quod/stdlib/<name>.json show`.

## quod-script

`script.py` is a single-pass tokenizer + recursive-descent parser
emitting model nodes directly. The grammar is documented in the
module docstring (and re-stated in `LANGUAGE.md`). Touch points:

- `tokenize` is the lexer — keyword set, operator set, integer suffix
  handling, string/char literals, `&.dotted.name` lookahead.
- `Parser` is the parser; `parse_function(src, enum_names=...)` is the
  external entrypoint. The `enum_names` hint disambiguates
  `Foo::Bar { ... }` (enum_init) from `Foo { ... }` (struct_init); the
  CLI passes the program's enum names so the script doesn't need its
  own type system.
- `_split_int_suffix` and `_int_lit_from_token` handle the `42i32`
  syntax + bare-`return`-adopts-return-type rule.

When extending the grammar:

- Add the keyword to the keyword set and the lexer's identifier-vs-keyword
  branch.
- Pick the right precedence level (`or_expr`, `and_expr`, `cmp_expr`,
  `add_expr`, `mul_expr`, `unary_expr`, `postfix`, `primary`).
- Emit the model node directly. Don't introduce intermediate AST
  types — the model IS the AST.
- Extend `tests/cases/lang/...` with cases covering the new shape;
  most existing script-driven cases are in `tests/cases/lang/struct/`,
  `lang/arena/`, `lang/strings/`. Round-trip script→model→IR→behavior
  is the test.

What script does not do: claims, struct definitions, enum
definitions, externs, string constants, imports. Those are CLI verbs
(`quod claim add`, `quod struct add`, etc.) — they need flags the
grammar doesn't model.

## Adding a node kind

Worked example: adding a hypothetical `quod.bitcast` expression.

1. **Define the model node.** Add a `Bitcast(_Node)` class in
   `model.py` with `kind: Literal["quod.bitcast"]` and the fields it
   needs. Frozen, like every other node.
2. **Add it to the right union(s).** For an expression: append to the
   `Expr = Annotated[Union[...], ...]`. For a statement: append to
   `Statement`. (Don't forget the discriminator picks the class by
   the `kind` literal, so the literal is what matters.)
3. **Validate as needed.** If the node has cross-references (e.g.
   names a struct), extend the validators in `_validate_structs` /
   `_check_struct_uses_in_expr` so dangling refs are caught at load
   time, not at lowering time.
4. **Lower it.** Add a case in `_lower_expr` (or `_lower_stmt`).
   Follow existing examples — extract IR types from the registries
   threaded through the call (`struct_tys`, `enum_tys`), use
   `builder.bitcast` etc.
5. **Schema-up.** `schema.py` is data-driven from the model classes,
   so the new kind appears in `quod schema --category expression`
   automatically. Confirm `quod schema quod.bitcast` shows what you
   expect.
6. **Author surface.** Decide whether the node deserves a script
   shape; if yes, extend `script.py`. If no (rare), users construct
   it via JSON only.
7. **Ingest support.** If the C-ingest subset can produce this shape,
   wire it in `ingest/c.py`. If not, it's not a regression — `c.py`
   is intentionally narrow.
8. **Test.** Drop one or more `tests/cases/lang/<topic>/*.json`
   cases. End-to-end (compile a program that uses the node, assert
   stdout/exit) is the standard.

Good worked examples in the codebase: `EnumDef` / `EnumInit` /
`Match` for a structured-type kind with custom validators, `TryExpr`
(`?`) for postfix-syntax + control-flow desugar, `SizeOf` for a
target-data-driven constant expression, `WithArena` for a statement
that desugars + auto-declares externs, `PtrOffset` for an integer-on-
pointer binop modelled as its own node. All follow the pattern above.

## Adding a CLI command

`cli.py` is a Typer tree. Subcommands are `noun_app = typer.Typer(...)`,
attached to the root with `app.add_typer(noun_app, name="noun")`.

To add `quod foo bar`:

1. Find or create a `foo_app = typer.Typer(...)` in `cli.py` and
   `app.add_typer(foo_app, name="foo")`.
2. Implement `@foo_app.command("bar")` against the editor / model
   surfaces. Don't reach into LLVM directly — go through the
   `lower`/`editor` modules.
3. Use `find_function_ref(program, ref)` for any user-supplied
   function reference; that's what makes name-or-hash-prefix work
   uniformly.
4. Output: print plain text to stdout for human consumption. Add a
   `--json` flag for machine-readable output if it makes sense for
   this command (`fn ls`, `claim ls`, `find` already do).
5. **Test.** Drop a `tests/cases/cli/foo/bar.json` (single-step) or
   a multi-step case if the command modifies and you want to assert
   final state. The conftest treats `cli` lists or `steps` lists
   uniformly.

Schema introspection for the new command's input shape: `quod schema`
data is generated from the Pydantic model — if your command takes a
JSON spec for a node kind, point users at `quod schema KIND` instead
of writing prose.

## C ingest

`ingest/c.py` is a libclang AST walker. The supported subset is
deliberately narrow:

- int-only types, `i8*` for `char*` / `void*`,
- arithmetic and comparison binops,
- short-circuit `&&` / `||` (the only short-circuit ops),
- `if`, `while`, `return`,
- locals via `int x = ...;` declarations,
- calls to other ingested functions and to extern functions.

What's deliberately out: structs, floats, `for` (rewrite to `while`),
`switch`, function pointers, varargs. Anything outside the subset
raises `IngestError` with the source location.

When extending: add the AST node class to the `_visit_*` dispatcher,
mirror the C semantics in quod model nodes, raise `IngestError` for
cases you don't want to handle. Rerun `tests/cases/c_ingest/` cases
to confirm regressions don't appear.

## Tests

Cases are JSON files under `tests/cases/`. The conftest auto-collects
every `*.json` and dispatches by shape:

- `cli` / `steps` → CLI test, run via Typer's `CliRunner` against a
  sandbox copy of `in_program`. Single step or multi-step.
- `program` / `program_file` / `c_file` → behavior test: compile to a
  binary, optionally pass `args` / `stdin`, assert
  `expect.stdout` (exact) / `expect.stdout_json` (structural) and
  `expect.exit`.

A case file may hold a JSON list — each element becomes its own item
named by `name` or positionally.

Run them with `uv run pytest`. There's no separate fast/slow split —
the corpus is small enough to run end-to-end in a few seconds.

The two Python test files in `tests/` (`test_enum_codegen.py`,
`test_script.py`) are unit tests for the script parser and enum
lowering. Behavior coverage lives in cases.

## Hashing and addressing

`hashing.py` computes content hashes via canonical-JSON SHA-256 over
each node's `model_dump`. Hashes propagate up: a Function's hash
covers every field including the body.

The CLI displays the first 12 hex chars (`HASH_DISPLAY_LEN`) and
accepts any unambiguous prefix. `find_by_prefix` rejects ambiguous
prefixes and treats content-equivalent occurrences as one match
(content addressing is content-equality, by definition).

`replace_node(root, target_ref, new_node)` replaces every occurrence
of the matched node — appropriate for content-addressable semantics.
If a caller wants to disambiguate physically-distinct same-content
nodes, address the *parent*.

## quod.toml semantics

`config.py` loads with no walk-up, no implicit defaults. Every
invocation needs a `quod.toml`; `quod init` writes one.

Paths inside `quod.toml` resolve relative to the file's parent dir
— `quod build -c /elsewhere/quod.toml` works regardless of CWD, but
the launched binary inherits CWD from the invocation.

`build_dir` and `proofs_dir` are top-level (sibling of `[build]`).
`[build]` carries `profile`, `target`, and `link`. `[enforce]` carries
the per-regime enforcement default; `--enforce-{axiom,witness,lattice}`
on the command line overrides it.

`[[program]]` is repeatable — that's the workspace mode. Each
`[[program]]` may have multiple `[[program.bin]]` entries (one binary
per entry function).

When extending the config, keep it explicit. We don't infer; we read.

## Common authoring rules (apply when working in this repo)

- **Always run `quod schema` before constructing a JSON node.** The
  schemas are precise and short. Guessing produces validation errors.
- **Prefer `quod claim prove` over `quod claim add` (axiom).** Axioms
  are trust holes — `llvm.assume` with UB if false. Witnesses come
  with a Z3-backed `.smt2` artifact pinned by sha256.
- **After editing a function with witness claims, run `quod claim
  verify`.** Proofs are pinned to body hashes; edits invalidate them.
- **Don't bypass `sat`/`unknown` from `quod claim prove` by falling
  back to axiom.** `sat` means the claim is false; `unknown` means
  it's outside the SMT lowering — refactor or skip, don't trust-me-bro.
- **Frozen Pydantic models — never mutate in place.** Build new
  instances via `model_copy(update=...)`.
- **Keep boundaries.** `model.py` knows nothing about SMT;
  `proof.py` knows nothing about IR; `lower.py` knows nothing about
  claim provenance.

## Invariants worth checking when something breaks

- **Hash stability.** If a test fails with "no node matches hash
  prefix X", an edit somewhere upstream changed a node's serialization.
  Run `quod show --hashes` before/after to see what shifted.
- **Import resolution.** If a stdlib symbol "isn't found," confirm
  the program lists the module in `imports` AND that the module
  declares the symbol. Resolution is first-wins; a user-declared
  shadow with the same name will silently mask the import.
- **Witness verification.** If `quod claim verify` reports the proof
  diverges, `quod show` the function and check the body hash; an
  edit may have invalidated it. Re-run `quod claim prove` or relax
  the claim.
- **Tier-disabled builds.** `--no-std` and `--no-alloc` short-circuit
  before lowering. If a build fails with "tier disabled," the
  offending module is in the error message — either lift the
  dependency or stop disabling the tier.

## Where to look for more

- `GUIDE.md` — the introductory tour with real CLI sessions.
- `LANGUAGE.md` — programmer-facing reference (types, statements,
  expressions, claims, stdlib tiers, script grammar, CLI tree).
- `integrations/claude/skills/quod/SKILL.md` — concise CLI reference
  + workflow recipes; loaded automatically by Claude Code when the
  `quod` skill is installed.
- `examples/` — the corpus. Especially `examples/json_v3/` (the
  JSON parser self-hosting milestone) and `examples/json_demo/`
  (the same parser consumed via `import alloc.json`).
