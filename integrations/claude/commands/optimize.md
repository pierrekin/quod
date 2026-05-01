---
description: Find claim-driven optimization opportunities in the current quod project, prove the worthwhile ones, and confirm IR shrinkage
allowed-tools: Bash(quod *) Read
---

Drive the claim-driven optimization workflow on the current quod
project. The user has asked to make their program faster, shrink its
IR, or "find what's worth proving."

Steps:

1. Sanity-check the project: run `quod check`. If it fails, stop and
   surface the error — there's no point optimizing a broken program.

2. Capture the current optimized-IR baseline:
   `quod build --show-ir` and note the byte count or line count of the
   optimized IR. This is the "before."

3. Run `quod claim suggest --top-n 10`. This speculatively compiles
   candidate claims and ranks them by how much optimized IR each would
   eliminate if proven. Read-only.

4. If the suggester returns nothing, stop and tell the user the codegen
   is already tight — not every program benefits from claim-driven
   optimization.

5. For each suggestion, decide: is the claim *actually true* given the
   function body and the user's intent? Skim the function with
   `quod fn show FN` if needed. Then:
   - If the claim should hold: run `quod claim prove FN KIND [TARGET]
     [--min N] [--max N]`.
   - If `prove` returns `sat`: Z3 found a counterexample — the claim is
     false. Skip it. Do NOT fall back to `quod claim add` as axiom.
   - If `prove` returns `unknown` or hits a NotImplementedError: the
     claim is beyond the current SMT lowering. Skip it.

6. Re-run `quod build --show-ir` and compare optimized-IR size against
   the baseline. Report the delta.

7. Run `quod claim verify` to confirm every newly-attached witness
   still holds end-to-end.

Be conservative: only prove claims that are actually consequences of
the function body. The point of `regime=witness` is to NOT have to
trust the human — don't undermine that by skipping past `sat`/`unknown`
results.
