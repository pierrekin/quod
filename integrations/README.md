# integrations

Bundles that expose quod to coding agents.

- `pi/` — extension for [pi-coding-agent](https://github.com/mariozechner/pi-coding-agent).
  Each `quod` subcommand is registered as a typed tool that shells out
  to the CLI. See `pi/extensions/quod.ts`.
- `claude/` — [Claude Code](https://claude.com/claude-code) plugin.
  Ships a skill that teaches Claude how to drive the `quod` CLI plus a
  few workflow slash commands (`/quod:init`, `/quod:optimize`,
  `/quod:prove`). No MCP server — Claude calls `quod` via Bash directly.

Both assume `quod` is on `$PATH` (or, in the Python-managed case,
`uv run quod`).
