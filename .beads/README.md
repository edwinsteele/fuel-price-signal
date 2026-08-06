# .beads/

This directory holds [Beads](https://github.com/gastownhall/beads) (`bd`) — the issue tracker for this repo, replacing GitHub Issues as of 2026-08-06.

See [AGENTS.md § Beads](../AGENTS.md#beads) for the actual conventions used here (commands, label taxonomy, the decision-pointer convention) — that section is the source of truth, not this file.

What's in here:
- `config.yaml`, `metadata.json` — git-tracked project config
- `embeddeddolt/` — the Dolt database itself (gitignored; sync via `bd dolt push`/`bd dolt pull`, not git)
