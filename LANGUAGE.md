# quod — language reference

A programmer-facing reference: types (including enums), expressions
(including `?`, struct/enum literals, `sizeof`), statements (including
`match` and `with_arena`), claims, the three stdlib tiers (`core` /
`alloc` / `std`), the quod-script surface, and the CLI.

For a gentle introductory tour with hello-world and a first proof, read
`GUIDE.md` first. This document is the deeper reference you reach for
once you're past hello-world and want to write real programs.

## Mental model

A quod program is a tree of typed JSON nodes (`program.json`). There is
no parser. You author the tree directly, or via the CLI's mutation
commands, or via quod-script (a textual surface that emits the same
nodes).

Two layers:

- `llvm.*` — primitives that map ~1:1 to LLVM IR ops (`llvm.binop`,
  `llvm.const_int`, `llvm.call`, `llvm.param_ref`, …).
- `quod.*` — sugar over LLVM: control flow (`quod.if`, `quod.while`,
  `quod.for`, `quod.match`), bindings (`quod.let`, `quod.assign`),
  effectful statements (`quod.expr_stmt`, `quod.store`, `quod.store_field`,
  `quod.field_set`, `quod.with_arena`), structured types
  (`quod.struct_init`, `quod.enum_init`, `quod.field`), and a few
  computed expressions (`quod.sizeof`, `quod.ptr_offset`, `quod.widen`,
  `quod.load`, `quod.load_field`, `quod.null_ptr`, `quod.char_lit`,
  `quod.try`).

You write at the `quod.*` layer, dropping to `llvm.*` for primitives.

**Always start with `quod schema`.** Before authoring any node, ask the
schema for its shape — guessing produces validation errors:

```sh
quod schema                          # categories overview
quod schema --category statement     # all statement kinds
quod schema quod.match               # full schema + minimal example
quod schema EnumDef                  # whole-enum shape
```

## Types

| Kind          | What it is                              |
|---------------|------------------------------------------|
| `llvm.i1`     | bit, used for booleans                   |
| `llvm.i8`     | byte                                     |
| `llvm.i16`    | 16-bit int                               |
| `llvm.i32`    | 32-bit int                               |
| `llvm.i64`    | 64-bit int                               |
| `llvm.i8_ptr` | opaque byte pointer                      |
| `llvm.struct` | reference to a `StructDef` by name       |
| `llvm.enum`   | reference to an `EnumDef` by name        |
| `llvm.void`   | only valid as a function return type     |

LLVM convention: types carry no signedness; the operation does. Use
`slt` vs `ult` on `BinOp`, `widen` (sext) vs `uwiden` (zext) on
extension.

`int_type_width` is the canonical size table — i1 is 1 bit, i64 is 64.

### Structs

Records, by-value, no by-value recursion (a struct can't contain itself
directly or transitively as a value field). Pointer-to-struct uses
`i8*` as the universal pointer type — fine for opaque handles, and for
self-referential shapes (e.g. linked-list nodes) where the next-pointer
field is typed `i8*` and accessed via `quod.load_field` /
`quod.store_field`.

```sh
quod struct add Point  x:i32 y:i32
quod struct add Parser input_ptr:i8_ptr cur:i64 had_error:i8
```

Author a struct value with `quod.struct_init`:

```json
{"kind": "quod.struct_init", "type": "Point",
 "fields": [{"name": "x", "value": ...}, {"name": "y", "value": ...}]}
```

Two ways to read/write fields:

- **By value** — `quod.field` (read) and `quod.field_set` (write). The
  read takes a struct-typed expression and returns the field's value;
  the write mutates a Let-introduced struct local. Use these when the
  whole struct lives in a register.
- **Through a pointer** — `quod.load_field` (read) and
  `quod.store_field` (write). Both take an `i8*` plus a struct-type
  name and a field name; lowered as `bitcast(ptr, T*)` + `getelementptr`
  + `load` / `store`. Use these for struct-on-heap access (arena-
  allocated records, linked-list nodes, anything whose lifetime
  outlives a single function).

Lowered to an LLVM identified struct; values are passed and returned
by value, and the field-pointer accessors emit targeted GEPs.

### Enums (tagged unions)

Sum types. Each variant has a name and zero or more typed payload
fields. Payload fields are unrestricted — int widths, `i8*`, named
structs, or even other enums.

```sh
quod enum add < /tmp/json_value.json
```

```json
{
  "name": "JsonValue",
  "variants": [
    {"name": "Null"},
    {"name": "Bool", "fields": [{"name": "value", "type": {"kind": "llvm.i1"}}]},
    {"name": "Number", "fields": [{"name": "value", "type": {"kind": "llvm.i64"}}]},
    {"name": "String", "fields": [{"name": "text", "type": {"kind": "llvm.struct", "name": "core.str.String"}}]}
  ]
}
```

Lowered to `{i8 tag, [N x i64] payload}` where N is the largest
variant's slot count. `EnumInit` bitcasts the payload to a per-variant
struct shape; `Match` does the inverse to bind locals.

Construct with `quod.enum_init`:

```json
{"kind": "quod.enum_init", "enum": "JsonValue", "variant": "Number",
 "fields": [{"name": "value", "value": <i64-expr>}]}
```

Or via script:

```
JsonValue::Number { value: 42 }
JsonValue::Null { }
```

### `match` — pattern dispatch

Exhaustive arm-per-variant. Each arm names the variant, binds its
payload fields to locals (one name per field, in declaration order),
and runs its body. Bindings are scoped to the arm body only.

A wildcard arm `_` matches every variant not handled elsewhere; with a
wildcard present, named arms don't have to be exhaustive. At most one
wildcard per match. Wildcard takes no bindings — use a normal variant
arm if you need the payload.

```
match v {
  JsonValue::Null         { return -1 }
  JsonValue::Bool   value { return widen(value to i64) }
  JsonValue::Number value { return value }
  _                       { return 0 }
}
```

Lowered to a `switch` on the discriminant byte; one basic block per
arm. Per-arm bindings live in their own scope, so two arms can both
bind a local named `value` without conflict.

### `?` — early-return on the sad variant

Postfix operator that propagates "errors" up the stack. Defined for any
2-variant enum where exactly one variant has a single payload field
(the *happy* variant) and the other has no payload (the *sad* variant).
Variant names don't matter — `Ok`/`Err`, `Some`/`None`, `Found`/
`Missing` all qualify by shape.

```
fn lookup(map: i8_ptr, key: i8_ptr) -> JsonOpt { ... }

fn deep(map: i8_ptr, k1: i8_ptr, k2: i8_ptr) -> JsonOpt {
  let inner: alloc.json.JsonValue = lookup(map, k1)?
  let leaf:  alloc.json.JsonValue = lookup_in(inner, k2)?
  return JsonOpt::Some { value: leaf }
}
```

Semantics: at runtime, evaluate `value`. If it's the sad variant, the
enclosing function immediately returns the sad variant of its declared
return type (the function's return type must therefore be the same
enum). If it's the happy variant, the expression evaluates to the
single payload field.

Lowered to: spill to alloca, switch on tag; the sad branch ret's the
sad-variant constructor of the function's return type; the happy branch
bitcasts and loads the payload field.

## Expressions

Beyond `IntLit`, `ParamRef`, `LocalRef`, `Call`, `BinOp`,
`ShortCircuitOr`, `ShortCircuitAnd`, `StringRef`:

- **`quod.field { value, name }`** — read field of a struct-typed expr.
- **`quod.struct_init { type, fields[] }`** — build a struct value.
  Must cover every field, no extras.
- **`quod.enum_init { enum, variant, fields[] }`** — build an enum
  value. Fields must match exactly the variant's payload.
- **`quod.try { value }`** — `?` propagation, see above.
- **`quod.sizeof { type }`** — bytes occupied by a quod type, computed
  from LLVM's target data layout. Returns `i64`. Useful for
  stride-correct pointer arithmetic over arena-allocated arrays.
- **`quod.ptr_offset { base, offset }`** — `base + offset` in bytes;
  base must be `i8*`, offset is `i64`. Pointer-arith over `i8*` (the
  byte stride). For `T*` strides, multiply offset by `quod.sizeof T`.
  Plain `BinOp(add, ptr, int)` also works for `i8*` + `i64`.
- **`quod.widen { value, target_type, sign? }`** — sext or zext an
  integer to a wider type. `sign` defaults to "signed".
- **`quod.load { type, ptr }`** — load `T` from `i8*` (with bitcast).
- **`quod.load_field { ptr, struct_type, name }`** — read one named
  field of a struct stored at `ptr`. Lowered as bitcast + GEP + load,
  so only the field is read (no whole-struct copy). Pair with
  `quod.store_field` for round-trip access.
- **`quod.null_ptr`** — `null` of `i8*`. Useful as the placeholder
  field value when an unused pointer field is required by `struct_init`.
- **`quod.char_lit { value }`** — a single-byte literal written as a
  string of length 1. Lowers to `const_int i8 ord(value)`.

## Statements

- **`quod.let { name, type, init }`** — introduce a local. Allocas
  at function entry; mem2reg promotes most into SSA.
- **`quod.assign { name, value }`** — mutate a Let-introduced local.
- **`quod.return_expr { value }`** / **`quod.return`** — return.
  `quod.return` is bare and only valid for `void` functions.
- **`quod.if { cond, then_body, else_body }`** — conditional. Both
  branches may terminate (return) or fall through; the merge block is
  added on demand.
- **`quod.while { cond, body }`** — pre-test loop.
- **`quod.for { var, type, lo, hi, body }`** — bounded iteration `var
  ∈ [lo, hi)`. Snapshot semantics: `lo`/`hi` are evaluated once
  before the loop. `var` is local to `body`.
- **`quod.expr_stmt { value }`** — evaluate for side effects, discard.
- **`quod.field_set { local, name, value }`** — mutate one field of a
  struct-typed local.
- **`quod.store { ptr, value }`** — write `value` (any scalar) to memory
  at `ptr` (an `i8*`). Bitcast + LLVM `store`. Pair with `ptr_offset`
  to write at non-zero offsets.
- **`quod.store_field { ptr, struct_type, name, value }`** — write
  `value` into one named field of a struct stored at `ptr`. Mutating
  counterpart of `quod.load_field`; lowered as bitcast + GEP + store.
- **`quod.match { scrutinee, arms[] }`** — see above.
- **`quod.with_arena { name, capacity, body }`** — bracket a body with
  an arena that's allocated at entry and dropped on every exit edge.
  See "Arenas" below.

## Claims

Three kinds:

- **`non_negative(param)`** — `param >= 0`.
- **`int_range(param, min?, max?)`** — `min <= param <= max` (either
  bound optional). Subsumes `non_negative` (use `int_range(p, min=0)`).
- **`return_in_range(min?, max?)`** — function-scoped, on the return
  value. Lowered as `llvm.assume` at every return site of the function,
  so callers (after inlining or interprocedural propagation) see the
  bound. Also valid on `extern` declarations — there the assume fires
  at every call site so the optimizer learns the range without needing
  the body in scope.

Three regimes:

- **`axiom`** — you assert it. Compiled as `llvm.assume`. UB if false.
- **`witness`** — proven by Z3 (`quod claim prove`); backed by a
  hash-pinned `.smt2` artifact. Re-checked by `quod claim verify`.
- **`lattice`** — derived by quod's analysis on each compile.
  Re-derived from scratch every time, never stored.

Two enforcements:

- **`trust`** — lowered to `llvm.assume`. Default.
- **`verify`** — lowered to a runtime branch + `abort()`. Costs an
  instruction per check, but turns "UB if false" into "abort if false".

Provider routing: `quod claim prove` and `quod claim derive` route to
the first registered provider for the requested (regime, mode); see
`quod provider ls` for what's available.

## Stdlib tiers

Three tiers, identified by top-level namespace. Imports are declared
per-program in the `imports` array of `program.json`; they merge in at
build time (first-wins by name).

| Tier   | Namespace  | Needs                     | Disable with |
|--------|------------|---------------------------|--------------|
| core   | `core.*`   | nothing — pure quod       | (always on)  |
| alloc  | `alloc.*`  | runtime allocator (arena) | `--no-alloc` |
| std    | `std.*`    | hosted OS / libc          | `--no-std`   |

`--no-std` allows core + alloc but refuses to resolve std imports.
`--no-alloc` is bare-metal mode: only core. `--no-alloc` also forbids
`with_arena` (it desugars to alloc-tier externs).

### Modules

- `core.bytes` — `core.bytes.eq(a, alen, b, blen)`,
  `core.bytes.copy(dst, src, n)`, `core.bytes.cstr_len(p)`. Pure-quod
  byte ops; no allocator dependency.
- `core.str` — `core.str.String { ptr: i8*, len: i64 }`,
  `core.str.eq(a, b)`, `core.str.slice(s, lo, hi)`,
  `core.str.from_cstr(c)`. Slice-by-pointer; no copies.
- `alloc.arena` — quod-authored bump allocator with chunk-list growth.
  Defines `alloc.arena.{new,alloc,drop,used}` and the underlying
  `Chunk` / `Arena` structs. Calls libc `malloc` / `free` / `memset`
  via `linkage.libc` externs. Imported transitively by anything that
  allocates; auto-imported by `quod.with_arena`.
- `alloc.str` — `alloc.str.to_cstr_in(s, arena)` — copy a `String`
  into the arena and zero-terminate, returning `i8*`. The bridge from
  quod strings to libc-shaped externs.
- `alloc.json` — JSON parser. `alloc.json.parse(text, len, arena)` →
  `JsonValue`, plus accessors (`object_get`, `index`, `as_number`,
  `as_string`, …). Returns option-like enums (`JsonValue` with an
  `Error` variant; `JsonOpt` for nullable lookups) — pair with `?` and
  `match`.
- `std.io` — `std.io.read_file_to_arena(path, arena)` →
  `ReadResult::Ok { text } | Err`. Wraps libc `open` / `read` / `close`.
  `std.io.file_size(fd) -> IoResult` returns the file size via lseek.

### Importing in your program

Add an import:

```json
{
  "imports": ["alloc.json", "std.io"],
  "functions": [...]
}
```

Edit `program.json` directly to add an import, or pass the imports
list to `quod ingest --import` when starting from a C source.

After resolution: `program.imports` is empty in memory; the merged
program is indistinguishable from one written flat. The on-disk
`program.json` keeps your imports as you wrote them.

## Arenas

quod's allocator-of-record. The model surface is `quod.with_arena`;
the implementation is the quod-authored `alloc.arena` stdlib module.

```
fn parse_one(text: i8_ptr, len: i64) -> i64 {
  with_arena scratch (capacity = 65536) {
    let v: alloc.json.JsonValue = alloc.json.parse(text, len, scratch)
    return alloc.json.as_number(v)
  }
  // arena is dropped on every exit edge — fall-through and every reachable return
}
```

`with_arena` desugars to `alloc.arena.new(capacity)` at block entry and
`alloc.arena.drop(handle)` on every exit edge. The desugaring
auto-injects `imports: ["alloc.arena"]` when the program doesn't
already declare it, so a `with_arena` block is one-stop sugar — call
`alloc.arena.alloc(handle, n)` directly inside the body for typed
buffer allocations.

Arena pointers are valid until the matching drop. The allocator never
relocates: when a bump chunk runs out, a fresh chunk is added to a
singly linked list of chunks owned by the arena. Bytes returned are
zeroed (calloc-like). 16-byte alignment for every allocation.

`--no-alloc` refuses `with_arena` and refuses to resolve any `alloc.*`
or `std.*` import. Suitable for bare-metal targets where libc `malloc`
isn't available.

### Optional native runtime

quod also supports symbols defined in `src/quod/runtime/*.c`, compiled
on demand into `libquodrt-vN.a` and linked by-reference. Declare them
as externs with `linkage.runtime` and they're picked up automatically.
The directory ships empty by default — the arena allocator lives in
quod, not C — so this hook is reserved for primitives that genuinely
can't be expressed at the language level (SIMD intrinsics, panic
abort, etc.).

## quod-script — the textual surface

Authoring full function bodies as JSON gets verbose fast. **quod-script**
is a compact textual surface for the authoring subset: signatures,
statements, expressions. One-way (script → JSON nodes); the JSON
remains the asset.

```
quod fn add --script "fn add_two(a: i32, b: i32) -> i32 { return a + b }"
quod fn add --script-file path/to/body.q
cat body.q | quod fn add --script-file -
```

### Grammar

```
function   := 'fn' DOT_IDENT '(' params? ')' '->' type body
params     := param (',' param)*
param      := IDENT ':' type
body       := '{' stmt* '}'

type       := 'i1' | 'i8' '*'? | 'i16' | 'i32' | 'i64' | 'void' | DOT_IDENT

stmt       := let_stmt | if_stmt | while_stmt | for_stmt | return_stmt
            | with_arena | store_stmt | match_stmt
            | assign_or_field_set_or_expr
let_stmt   := 'let' IDENT ':' type '=' expr
if_stmt    := 'if' '(' expr ')' block ('else' block)?
while_stmt := 'while' '(' expr ')' block
for_stmt   := 'for' IDENT ':' type 'in' expr '..' expr block
return_stmt:= 'return' expr?
store_stmt := 'store' '(' expr ',' expr ')'
match_stmt := 'match' expr '{' arm+ '}'
arm        := (ENUM '::' VARIANT | '_') (IDENT (',' IDENT)*)? block
with_arena := 'with_arena' IDENT '(' 'capacity' '=' expr ')' block
assign_or_field_set_or_expr
           := IDENT '=' expr                       # assign
            | IDENT '.' IDENT '=' expr             # field_set
            | expr                                  # expr stmt

expr       := or_expr
or_expr    := and_expr ('||' and_expr)*
and_expr   := cmp_expr ('&&' cmp_expr)*
cmp_expr   := add_expr (CMPOP add_expr)?
add_expr   := mul_expr (('+' | '-') mul_expr)*
mul_expr   := unary_expr (('*' | '/' | '%' | '/u') unary_expr)*
unary_expr := postfix
postfix    := primary ('.' IDENT | '?')*

primary    := INT | CHAR | 'null' | 'true' | 'false'
            | '&' DOT_IDENT
            | 'load' '[' type ']' '(' expr ')'
            | 'widen' '(' expr 'to' type ')'
            | 'uwiden' '(' expr 'to' type ')'
            | 'ptr_offset' '(' expr ',' expr ')'
            | 'sizeof' '[' type ']'
            | DOT_IDENT '(' args? ')'      # call
            | DOT_IDENT '{' field_inits '}' # struct/enum init (constructor by name)
            | ENUM '::' VARIANT '{' field_inits '}'  # enum_init
            | IDENT                         # local/param ref
            | '(' expr ')'

CMPOP : == != < <= > >= <u <=u >u >=u
```

`DOT_IDENT` allows dotted names (e.g. `core.str.eq`,
`alloc.json.parse`), so stdlib call sites read naturally.

### Lexical notes

- Integer literals default to `i64`. Use a width suffix to opt in:
  `0i8`, `42i32`, `-3i8`. Essential when writing into narrower struct
  fields. Bare `return 0` is special-cased — the literal adopts the
  function's declared return type, so `return 0` works in any
  int-returning function.
- A bare integer literal as an `init`/`assign`/`store` value
  *auto-narrows* to the destination type. `let b: i8 = 42` doesn't
  need a suffix.
- Statements may be terminated by newlines or `;`; either is optional
  at end of block.
- `&.const_name` is a `quod.string_ref` — the dot is part of the
  syntax. The constant must already exist (`quod const add`).

### Out of scope

Claims, struct definitions, externs, string constants, imports, and
enum definitions are CLI verbs (`quod claim add`, `quod struct add`,
`quod extern add`, `quod const add`, `quod enum add`) — they need
flags the grammar doesn't model. Script is one-way: an input format
only, with no round-trip from `quod show`.

### Disambiguation

When the grammar can't disambiguate a `{` (e.g. `if (Foo { … })` —
struct literal or block?), Rust-style: struct/enum literals are
disabled in the cond position of `if` / `while` / `for`. Wrap in parens
to force one in: `if ((Parser { … }).had_error == 0) { ... }`.

## Workspaces

A `quod.toml` lists one or more `[[program]]` entries; each is a quod
program with a `name`, a `version`, and a `file` pointing at its JSON.
Each `[[program.bin]]` is a thing-to-build for that program: `name` is
the output binary filename, `entry` names the function inside the
program JSON to use as the entry point.

```toml
[build]
profile = 2

build_dir  = "build"        # default: ./build
proofs_dir = "proofs"       # default: ./proofs

[[program]]
name = "demo"
version = "0.1.0"
file = "program.json"

  [[program.bin]]
  name  = "demo"
  entry = "main"
```

Multiple `[[program]]` entries → workspace. Pick one with `quod -p NAME ...`,
or build all with `quod build`.

`examples/quod.toml` is an end-to-end workspace example with ~30 programs.

## CLI reference

### Lifecycle

| Command            | Purpose                                                                |
|--------------------|------------------------------------------------------------------------|
| `quod init -t T`   | Write `quod.toml` + `program.json`. Templates: `hello`, `guarded`, `empty`. |
| `quod ingest C`    | Ingest a C source file into a fresh project. `--import MOD` adds an stdlib import; `-n NAME` overrides the program name. |
| `quod check`       | Parse, lower, LLVM-verify. No artifacts emitted.                       |
| `quod build`       | Lower → optimize → emit object → link, per `[[program.bin]]`.          |
| `quod run`         | Build then exec. `--bin NAME` to pick one. `-- ARGS...` are passed to the binary, and int parameters are parsed via `atoll`. |

`build` flags worth knowing:

- `--profile N` — LLVM optimization level (0..3). 0 skips the optimize pass.
- `--target TRIPLE` — LLVM target triple. Defaults to host or the
  `target` field in `quod.toml`.
- `--show-ir` — print optimized IR to stdout.
- `--enforce-axiom trust|verify` (and `--enforce-witness`,
  `--enforce-lattice`) — override claim enforcement at build time
  (e.g. compile in `verify` for a debug build, `trust` for release).
- `--no-std` — refuse to resolve `std.*` imports.
- `--no-alloc` — refuse to resolve `alloc.*` and `std.*` imports;
  refuse `with_arena`. Bare-metal mode.

Top-level `quod -c PATH` selects a non-default `quod.toml`. `quod -p NAME`
selects one program from a workspace. `quod -f PATH` bypasses
`quod.toml` entirely — useful for inspecting standalone JSON modules
(e.g. stdlib files in `src/quod/stdlib/`).

### Inspection

| Command                  | Purpose                                              |
|--------------------------|------------------------------------------------------|
| `quod show`              | Whole program in canonical form. `--hashes` dumps every node + its short hash. `--json` for machine output. |
| `quod find PREFIX`       | Resolve a content-hash prefix to a node and print it. |
| `quod schema [KIND]`     | Schemas. No args → categories. `--category C` → kinds in C. `KIND` → fields, types, minimal example. |

### Functions

| Command                      | Purpose                                       |
|------------------------------|-----------------------------------------------|
| `quod fn ls`                 | List functions with signatures and hashes.    |
| `quod fn show REF`           | Print one function (name or hash prefix).     |
| `quod fn add SPEC` / `--script` / `--script-file` | Append a function. JSON via stdin (`-`) or a path; or quod-script. |
| `quod fn rename OLD NEW`     | Rename + update every call site that names it. |
| `quod fn rm REF`             | Remove a function. (Permissive — doesn't refuse if other code still references it.) |
| `quod fn callers TARGET`     | Every call site to `target`.                  |
| `quod fn data-flow FN PARAM` | Every statement in `fn` that reads `param`.   |
| `quod fn call-graph`         | Caller→callees, with `@extern` and `!` markers. |
| `quod fn unconstrained`      | Params with no claim attached.                |

### Claims

| Command                                  | Purpose                            |
|------------------------------------------|------------------------------------|
| `quod claim ls [FN]`                     | List stored claims (axiom + witness). |
| `quod claim add FN KIND [TARGET] [--min N] [--max N] [--regime ...] [--enforcement ...]` | Attach a claim. |
| `quod claim relax FN KIND [TARGET]`      | Remove a claim.                    |
| `quod claim prove FN KIND [TARGET] [--min N] [--max N]` | Prove via a witness provider. On `unsat`: writes `.smt2` + attaches the claim. On `sat`/`unknown`: refuses. |
| `quod claim verify`                      | Re-hash + re-run every stored proof.  |
| `quod claim suggest [--top-n N]`         | Speculatively compile candidates and rank by IR shrinkage. Read-only. |
| `quod claim derive`                      | Run a lattice provider; print derived claims. Re-derived every compile. |

### Schema-first authoring

| Command                                  | Purpose                            |
|------------------------------------------|------------------------------------|
| `quod stmt add FN [SPEC] --at-end \| --at-start \| --before HASH \| --after HASH` | Insert a statement (JSON, stdin or path). |
| `quod stmt rm FN HASH_PREFIX`            | Remove a statement by hash prefix. |
| `quod const ls / add NAME VALUE / rm NAME / rename OLD NEW` | String constants. |
| `quod struct ls / show NAME / add NAME FIELDS... / rm NAME / rename OLD NEW` | Structs. `add` takes `field:type` tokens. |
| `quod enum ls / show NAME / add SPEC / rm NAME / rename OLD NEW / rename-variant ENUM OLD NEW` | Enums. `add` takes a JSON `EnumDef` on stdin. |
| `quod extern ls / add NAME [...] / set-linkage NAME L / rm NAME / ingest HEADER` | Externs. `add` takes `--arity N` for all-i32, or `--param-type T...` + `--return-type T` for explicit, plus `--varargs` for printf-shaped, plus `--linkage libc\|runtime` (default libc). `set-linkage` flips the declared provenance in place. |
| `quod extern claim ls / add NAME KIND [...] / relax NAME KIND` | Claims attached to extern declarations. Currently restricted to return-scoped kinds (e.g. `return_in_range`); lowered as `llvm.assume` after every call site so the optimizer learns the bound. |
| `quod note add FN TEXT` / `note rm FN INDEX` | Free-form developer notes; metadata only. |

### Providers

| Command          | Purpose                                  |
|------------------|------------------------------------------|
| `quod provider ls` | Registered claim providers (regime + modes). Today: `lattice.literal_range` (derive) and `z3.qf_lia` (prove). |

## Common workflows

### Bootstrap a new project

```sh
mkdir my-thing && cd my-thing
quod init -t hello
quod run
```

For something with parameters and proofs to play with:

```sh
quod init -t guarded
quod fn show f
quod claim suggest
```

### Author a function with the script surface

```sh
quod fn add --script "fn clamp(x: i64, lo: i64, hi: i64) -> i64 {
  if (x < lo) { return lo }
  if (x > hi) { return hi }
  return x
}"
quod check
```

### Use the stdlib

Edit `program.json` directly to add imports:

```json
{"imports": ["std.io"], "functions": [...]}
```

Then call `std.io.read_file_to_arena(path, arena)` and match on the
result:

```
fn read_or(path: i8_ptr, arena: i8_ptr) -> core.str.String {
  let r: std.io.ReadResult = std.io.read_file_to_arena(path, arena)
  match r {
    std.io.ReadResult::Ok text { return text }
    std.io.ReadResult::Err     { return core.str.from_cstr(&.fallback) }
  }
}
```

### Prove a claim instead of asserting it

```sh
quod claim suggest               # what would help if proven?
quod claim prove f return_in_range --min -1
quod claim verify                # re-run after function edits
```

`unsat` → claim is true; `.smt2` artifact attached. `sat` → counterexample
exists; the claim is false (don't fall back to `claim add` axiom).
`unknown` → outside the SMT lowering (mutable locals, srem, unsigned cmp,
…) — refactor or skip.

### Inspect by content hash

Every node has a stable content-derived hash. `quod show` prints them
as `[abc123]` prefixes; the CLI accepts any unique prefix anywhere a
name is taken (so you can address a function by hash without naming it).

```sh
quod show --hashes
quod find e909
quod fn show 740b           # hash prefix instead of name
```

### Inspect a stdlib module standalone

`-f` bypasses `quod.toml`. Useful for reading a stdlib module:

```sh
quod -f src/quod/stdlib/std.io.json show
quod -f src/quod/stdlib/alloc.json.json fn ls
```
