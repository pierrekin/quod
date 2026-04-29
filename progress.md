# cpg-agent — multi-track progress log

This file tracks state across the 5 directions the user picked. Read it cold;
each section says what's done, what's mid-flight, and what's next.

## Order

User asked for all of (1)–(5), with (1) last. Picked path:

```
   (3) annotations          ← quick, independent, sets up intent capture
   (4) language expansion   ← needed by (5); minimum viable: pointers, more ops
   (5) C interop            ← demo target: file I/O via libc
   (2) more CPG views       ← higher value once programs get bigger via (5)
   (1) feedback loop        ← last per user; needs the rest to demo against
```

## Status legend

```
   [ ] not started
   [~] mid-flight
   [x] landed
```

## Tracks

### (3) Annotation / intention layer  [x]

Landed:
- `Function.notes: tuple[str, ...] = ()` with default-drop serializer.
- `format_function` renders notes as `// ...` lines before the signature.
- CLI: `cpg add-note -f FN "text"`, `cpg remove-note -f FN -i N`.
- Smoke: round-trips through JSON; `show` displays them.

Not done (deferred until needed):
- Notes on individual claims/statements (justification.note covers claims partially).
- Structured intent kinds (perf-critical / stub / etc.) — kept flat strings.


Small structural addition. Plan:
- `Annotation` node attached to claims/functions/statements? Or a free `notes: tuple[str, ...]` field on `_Node` directly?
- Decision pending: per-node free `notes` is simpler; structured `Annotation` with kind (perf-critical / stub / hot-path / unproven-but-believed) is richer.
- Going to start with: `notes: tuple[str, ...] = ()` on `_Node`, default-drops in JSON, renders in pretty-printer.
- CLI: `cpg add-note --to-hash PREFIX "text"` and `cpg list-notes`.

### (4) Language expansion  [x] phase 4a + 4b

Phase 4a (i32 externs):
- `ExternFunction(name, arity)` node, `Program.externs` tuple.
- LLVM declares externs in pass 1 alongside user functions.
- `cpg add-extern NAME --arity N` CLI.
- `show-call-graph` decorates extern callees with `@extern`.
- Demo: `examples/libc_abs.json` — `f(x): return abs(x)`, main returns abs(-5)=5.

Phase 4b (types + strings):
- `I32Type`, `I8PtrType`, `Type` discriminated union.
- `ExternFunction.param_types` / `.return_type` (typed sigs); `arity` kept as i32 shorthand.
- `StringRef(name)` expression — i8* pointer to a `StringConstant`.
- `Expr` union extended with `StringRef`.
- LLVM lowering: `_lower_expr` threads `extern_sigs` and `constants`; StringRef bitcasts the global to i8*.
- Pretty-printer renders typed sigs (`i32`, `i8*`).
- Demo: `examples/run_ls.json` — calls `system("ls ...")`, exit 0, real stdout.

Not done (deferred):
- ExprStmt (call as side-effect statement). Today you have to use the call's return as the function's return.
- Local variable bindings.
- Loops.
- `mul`, `sub`, etc. BinOps.
- Typed Function (params/return type annotations); user functions still all-i32.
- SMT lowering for typed Calls/StringRef — explicitly raises NotImplementedError.

Picking a minimum viable set keyed on what (5) needs.

For libc file I/O we need:
- pointer type (`i8*`) — for filenames, buffers
- some integer type ≥ i32 (for file sizes, but i32 enough for small demos)
- `null` literal
- maybe `eq` / `ne` BinOps for null checks
- `sub` BinOp (useful in general; cheap to add)

Punting on (for now): structs, arrays, loops, mul/div, i64.

Plan:
- Add `Type` discriminated union (`I32Type`, `I8PtrType`) — start narrow.
- Update `Function.params`, `Function.return_type` to be typed.
- `BinOp.op`: extend with `sub`, `eq`, `ne`.
- `NullLit` expression.
- LLVM lowering updates.
- SMT lowering: refuse non-Int types for now; cross-procedural over typed calls still works for Int-only.

### (5) C interop  [x] (subsumed by 4b)

Demo: `examples/run_ls.json` — `main(): return system("ls /etc/...")` runs the
shell command and returns its exit status. Real libc call, real stdout output.

The demo lands on top of the same machinery as 4b: typed `ExternFunction`,
`StringRef`, and the existing two-pass declaration. No special "interop layer"
beyond getting the type/pointer story right.

Open: getenv-then-puts-style demos require either an ExprStmt (so `puts(...)`
can stand alone as a side-effect statement) or a way for the i8* return of an
extern to flow into another extern's i8* arg in expression position. Today
the demo runs via `system()` because both args and return are simple enough
to chain into `return ...`.

Demo target: a program that **reads a file and prints its contents** (or
reads/writes via `fopen` / `fread` / `fwrite` / `fclose`). Smaller starting
point: just `puts(getenv("PATH"))` style — call libc, get a string back, print.

Plan:
- `ExternFunction` node — name + typed signature.
- `CallExtern` (or just reuse `Call` with the callee resolving to extern in the lowering step).
- Lowering: declare extern as LLVM function with ABI-correct types; link with `-lc` (libc is default) or `-lreadline` etc.
- Demo example: `read_env.json` that reads `getenv("HOME")` and `puts`-es it.

### (2) More CPG views  [x]

Landed:
- `cpg find-callers FN` — reverse call graph for one function. Walks every
  Call across the program; deduped on call-hash within a statement (so
  content-equivalent calls in if/else branches collapse to one entry).
- `cpg show-data-flow PARAM -f FN` — counts how many times the param is read
  per body-statement.

Not done (deferred — tooling layer is fine without them):
- Full reverse-adjacency graph view.
- Dominance / control-flow-graph view.
- Reach analysis (which paths can reach a statement).

Pick 2–3 useful views:
- `cpg show-data-flow PARAM` — where does this param's value flow? List of statement/expression hashes that read it.
- `cpg find-callers FN` — reverse call graph (every Call node referencing FN).
- Maybe: `cpg show-callers-graph` — full reverse adjacency.

### (1) Feedback loop  [x]

Landed:
- `cpg suggest-claims` — generates candidate claims (non_negative on each
  unconstrained param, return_in_range with bounds in {-1, 0}), speculatively
  compiles each through `derive → elaborate → lower → optimize`, measures
  the optimized-IR line count, and surfaces candidates that *shrink* IR.
- Doesn't require candidates to be true; doesn't mutate the program. Output
  is a TODO list for the agent to feed into `cpg prove`.

End-to-end demo (run on `examples/witness_changes_codegen.json` with no claims):
```
   step 1: suggest-claims
     baseline: 25 IR lines; 6 candidates
     -2 lines on f: return_in_range([-1, +inf])
     -2 lines on g: return_in_range([0, +inf])
   step 2: cpg prove return_in_range -f f --min=-1
     proved (z3 unsat). IR drops 25 → 23.
   step 3: suggest-claims (re-run)
     no candidates shrink IR — current codegen is already tight
   step 4: cpg prove return_in_range -f g --min=0  (the false suggestion)
     refused: z3 returned 'sat' (counterexample found)
```

The loop closes: suggester proposes, prover validates, system tightens. False
suggestions (the >=0 one for g, which is actually false because g can return -1
via f(y)) get caught at the prove step.

Heuristic limitations (open):
- Candidate set is small (handful of bounds per fn). A richer suggester would
  use lattice-derivable info, callee return claims, or LLVM's own analyses to
  propose more interesting candidates.
- Metric is opt-IR line count — coarse. Could refine to instruction count or
  to specific patterns (presence of `range` attribute, `nuw`, etc.).

The "if only we could prove this" suggester. Plan:
- Identify candidate claims (e.g., for each function, propose `return_in_range` and `non_negative` candidates).
- For each candidate: speculative compile WITH and WITHOUT, diff IR.
- Surface candidates whose presence shrinks IR.
- CLI: `cpg suggest-claims` printing a ranked list.

## Decisions / opens

- Notes vs structured annotations: starting with flat strings; promote to structured when there's a second consumer.
- Type system surface: keeping it in Pydantic discriminated union shape (`kind=`) for consistency with the rest of the model.
- Extern functions: do they live as graph nodes (yes, I think so — content-addressable, editable, etc.).

## Last touched

All 5 tracks landed in this round. Final smoke:
- 9 of 10 examples validate (callgraph.json's dangling `ghost` is by design).
- 3 witness-bearing examples re-verify clean (real Z3 invocation).
- Total source ~2.9k lines across 8 modules.

Demos that exist for each track:
- (3) any example via `cpg add-note -f FN "text"`
- (4a) `examples/libc_abs.json` (i32 extern)
- (4b) `examples/run_ls.json` (typed extern + StringRef)
- (5) same — `examples/run_ls.json` calls `system("ls ...")`, real stdout
- (2) `cpg find-callers` and `cpg show-data-flow` work on any example
- (1) end-to-end loop on `examples/witness_changes_codegen.json` (suggest → prove → tighter IR)
- core: `examples/cross_demo.json` for cross-procedural witness use
