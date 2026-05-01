# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this is

quod is a small programming-language toolchain. Programs are JSON trees
of typed Pydantic nodes (no parser); the `quod` CLI authors and edits
them, lowers them through llvmlite to LLVM IR, and links a native binary
via `clang`. Functions can carry **claims** (`non_negative`,
`int_range`, `return_in_range`) that are either asserted (`axiom`),
proven by Z3 (`witness`), or derived by a fixed-point analysis
(`lattice`).

## Read these before authoring

- `GUIDE.md` — end-to-end tour with real commands and outputs. The
  authoritative source for "what does quod do."
- `integrations/claude/skills/quod/SKILL.md` — concise CLI reference and
  workflow recipes (init / build / claim-prove / optimize). Loaded
  automatically by Claude Code when the `quod` skill is installed.
- `README.md` — setup and project layout.

## Authoring rules

- **Always run `quod schema` before constructing a JSON node.** Schemas
  are precise and short (`quod schema KIND`, `quod schema --category
  statement`). Guessing the shape produces validation errors.
- **Prefer `quod claim prove` over `quod claim add` (axiom).** Axioms
  are trust holes — `llvm.assume` with UB if false. Witnesses come with
  a Z3-backed `.smt2` artifact pinned by sha256.
- **After editing a function with witness claims, run `quod claim
  verify`.** Proofs are pinned to body hashes; edits invalidate them.
- **Don't bypass `sat` / `unknown` from `quod claim prove` by falling
  back to axiom.** `sat` means the claim is false; `unknown` means it's
  outside the SMT lowering — refactor or skip, don't trust-me-bro.

## Editing the code itself

- The graph is the asset. Nodes (`src/quod/model.py`) are frozen
  Pydantic models — never mutate in place; build new instances via
  `model_copy`.
- `model.py` knows nothing about SMT; `proof.py` knows nothing about
  IR; `lower.py` knows nothing about claim provenance. Keep those
  boundaries.
- New claim sources plug in via `src/quod/providers.py` (a `Provider`
  with `derive` and/or `prove`). The CLI routes by `(regime, mode)`.
- C-ingest support lives in `src/quod/ingest/c.py`; the supported
  subset is intentionally narrow (int-only, no structs / floats / for /
  switch). Refusals raise `IngestError` with a source location.

## Running anything

`quod` requires `uv` and Python 3.14+. Either:

```sh
uv run quod <args>                     # from inside the repo
uv run --project <repo-path> quod <args>  # from anywhere
```

`clang` must be on PATH (linker driver). `z3` is optional but required
for `quod claim prove` / `quod claim verify`.
