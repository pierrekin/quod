# Claude Code plugin for quod

A Claude Code plugin that teaches Claude how to drive the `quod` CLI.

## What it ships

- **`skills/quod/SKILL.md`** — the meat. Prose loaded on demand when
  Claude detects relevance (the description gates auto-loading).
  Distills the CLI surface, the claim/proof regimes, and the standard
  authoring/optimization workflows.
- **`commands/init.md`** — `/quod:init` to bootstrap a project with
  template choice and a sensible next-step suggestion.
- **`commands/optimize.md`** — `/quod:optimize` drives the
  suggest → prove → rebuild loop and reports the IR-size delta.
- **`commands/prove.md`** — `/quod:prove` discharges a single claim via
  Z3 with proper handling of `sat` / `unknown` / `unsat`.

There is intentionally no MCP server. Claude already has Bash, the
`quod` CLI is self-describing (`quod --help`, `quod schema`), and the
skill keeps the CLI as the single source of truth.

## Install

### Dev / try it ad-hoc (no install)

Load the plugin for one session via the `--plugin-dir` flag:

```sh
claude --plugin-dir ./integrations/claude
```

Good for iterating on the skill or commands — every session re-reads
the files.

### Permanent install

Claude Code installs plugins from "marketplaces", which can be a local
directory. From inside Claude Code:

```
/plugin marketplace add /absolute/path/to/quod/integrations/claude
/plugin install quod@<marketplace-name>
```

Then `/plugin` to confirm `quod` is in the Installed tab.

### Verify it loaded

- `/plugin` → Installed tab should list `quod`.
- In a fresh session, ask "what is quod?" — Claude should pick up
  `SKILL.md` (description-gated) and answer from it.
- Type `/quod:` and the three slash commands (`init`, `optimize`,
  `prove`) should autocomplete.

## Requirements

- `quod` on `$PATH` (or available via `uv run quod` from a quod
  project's working directory).
- For `claim prove` / `claim verify`: `z3` installed.
