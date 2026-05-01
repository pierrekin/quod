---
description: Prove a claim about a quod function via Z3 and attach it as a witness
argument-hint: "<function> <kind> [target] [--min N] [--max N]"
allowed-tools: Bash(quod *) Read
---

Discharge a claim about a quod function via Z3 and, on success, attach
it as a witness with a hash-pinned `.smt2` artifact.

Parse `$ARGUMENTS` as `<function> <kind> [target] [--min N] [--max N]`.
If the user gave a vaguer ask (e.g. "prove that f always returns
non-negative"), translate it: that example is `f return_in_range
--min 0`.

Claim-kind shape:

- `non_negative <param>` — that parameter is `>= 0`.
- `int_range <param> --min N --max N` — bounded int.
- `return_in_range --min N [--max N]` — function-scoped; do NOT pass a
  target.

Steps:

1. Confirm the function exists: `quod fn show <function>`. If the user
   gave a hash prefix, that works too.

2. Run `quod claim prove <function> <kind> [target] [--min N] [--max N]`.

3. Interpret the result:
   - **proved** (Z3 returned `unsat`): success. The claim is now
     attached as `regime=witness` with a hash-pinned `.smt2` artifact.
     Show the user the artifact path. Mention they can inspect the
     SMT-LIB if curious, and that `quod claim verify` will re-validate
     it after future edits.
   - **`sat`**: Z3 found a counterexample. The claim is FALSE. Do NOT
     fall back to `quod claim add` as axiom. Tell the user the claim
     does not hold and suggest either revising the claim (e.g. tighter
     bounds) or fixing the function.
   - **`unknown`** or `NotImplementedError`: the claim is beyond the
     current SMT lowering (mutable locals, `srem`, unsigned compares).
     Tell the user this is a prover limitation, not a falsification.
     Options: refactor the function into pure-expression form, or
     attach as `regime=axiom` ONLY if the user explicitly accepts that
     trust hole.
