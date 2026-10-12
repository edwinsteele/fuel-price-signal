---
name: venv-corruption-concurrent-uv
description: ModuleNotFoundError for a package uv.lock lists (e.g. 'six') means two concurrent uv runs half-synced .venv; rm -rf and re-sync.
metadata:
  type: reference
---

ModuleNotFoundError for a locked package (first seen as 'six' or 'py'; 'py' has since left uv.lock) mid-session (even though uv sync reports satisfied) means .venv is half-synced from two concurrent uv run invocations racing. Only reliable fix: rm -rf .venv && uv sync. Avoid running uv run commands concurrently against this project; run pipeline stages sequentially.
