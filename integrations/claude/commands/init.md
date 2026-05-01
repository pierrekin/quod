---
description: Initialize a new quod project (writes quod.toml + program.json)
argument-hint: "[hello|guarded|empty]"
allowed-tools: Bash(quod *) Read
---

Initialize a new quod project in the current directory.

Template selection:

- `hello` — runnable hello-world with `puts`. Use when the user wants
  to "try quod" or see end-to-end build+run.
- `guarded` — a function `f(x: i32)` with a conditional return.
  Deliberately has no `[[program.bin]]`. Use when the user wants to explore
  claims, proofs, or the optimizer.
- `empty` — blank slate. Use when the user is going to author their
  own program from scratch.

Steps:

1. If the user provided a template in `$ARGUMENTS`, use it. Otherwise
   ask which template (or pick `hello` if the user said something like
   "just give me a quod project to play with").
2. Run `quod init -t <template>`. Add `--force` only if the user
   explicitly asked to overwrite.
3. Run `quod show` to display what was written.
4. Suggest a sensible next step based on the template:
   - `hello` → `quod run`
   - `guarded` → `quod fn unconstrained` then `quod claim suggest`
   - `empty` → `quod schema Function` to start authoring
