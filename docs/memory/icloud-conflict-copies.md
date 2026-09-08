---
name: icloud-conflict-copies
description: OBSOLETE since the repo moved out of ~/Documents — kept so nobody re-derives the detection recipe.
metadata:
  type: reference
---

OBSOLETE AS OF 2026-08-18 — kept only so a future session doesn't re-derive it. The repo MOVED to ~/Code/fuel-price-signal, which is NOT under ~/Documents, so iCloud Desktop & Documents sync no longer covers it. Verified 2026-08-19: repo realpath /Users/esteele/Code/fuel-price-signal, zero conflict copies anywhere in the tree. Do not go hunting for these; do not trust the old ~/Documents/Claude/fuel-price-signal path in any doc or memory (the `reference_repo_path_migration` note in Claude's private memory dir records the move; Codex has no access to that, hence this file).

IF the repo is ever moved back under an iCloud-synced dir, the detection recipe was:
  - conflict copies insert ' 2' (or ' 3', ...) BEFORE the extension: 'freeze 2.json', 'fuel_signal 2.db', even 'fuel_signal 2.db-wal' (the ' N' landing before the whole '.db-wal' is the giveaway — no process writes that name).
  - they are mode 0600 with ZERO xattrs; files our own code writes are 0644 with com.apple.provenance.
  - git status shows only a fraction: *.parquet, *.db and .venv/ are gitignored, so a 500MB 'fuel_signal 2.db' hides completely. Use find, not git status.
