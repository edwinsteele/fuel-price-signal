"""The agent-instruction byte budget — enforced, because the overrun is silent.

Codex loads `AGENTS.md` (root first, then one file per directory down to the
working directory) up to `project_doc_max_bytes`, default 32 KiB, then stops.
There is no warning, no log line and nothing in the TUI: instructions past the
limit are simply absent, and the agent behaves as though they were never written.

This bit us once already. On 2026-09-10 the root AGENTS.md had reached 43,994
bytes, so the last 11 KB — the technical-memory write protocol, issue tracking,
the label taxonomy, branch/PR conventions and the PR-review section — were
invisible to Codex while every doc still cited them as authoritative.

The cap is on the *combined* size of every AGENTS.md that loads, so this sums
them rather than checking the root file alone: adding `fuel_signal/AGENTS.md`
does not buy extra budget, it spends the same budget from a second file.
"""
from __future__ import annotations

import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# codex's project_doc_max_bytes default. Not a style preference — the bytes past
# this point do not reach the model.
PROJECT_DOC_MAX_BYTES = 32 * 1024

SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".claude"}


def _instruction_files() -> list[pathlib.Path]:
    """Every AGENTS.md / AGENTS.override.md that Codex would load, root-down."""
    found = []
    for path in sorted(REPO_ROOT.rglob("AGENTS*.md")):
        if path.name not in {"AGENTS.md", "AGENTS.override.md"}:
            continue
        if SKIP_DIRS & set(path.relative_to(REPO_ROOT).parts):
            continue
        found.append(path)
    return found


def test_agent_instructions_fit_in_the_codex_context_budget():
    files = _instruction_files()
    assert files, "no AGENTS.md found — has the file been renamed?"

    sizes = {p.relative_to(REPO_ROOT): p.stat().st_size for p in files}
    total = sum(sizes.values())

    breakdown = "\n".join(f"    {name}: {size:,} bytes" for name, size in sizes.items())
    biggest = max(sizes, key=lambda k: sizes[k])
    assert total <= PROJECT_DOC_MAX_BYTES, (
        f"Agent instructions total {total:,} bytes, over the {PROJECT_DOC_MAX_BYTES:,}-byte "
        f"budget by {total - PROJECT_DOC_MAX_BYTES:,}.\n"
        f"{breakdown}\n"
        f"Codex will silently drop everything past the limit, so the tail of {biggest} "
        f"stops reaching it.\n"
        "Fix by MOVING content out, not deleting it — these files are for architecture and "
        "review rules only:\n"
        "    how-we-work rules      -> docs/CONVENTIONS.md\n"
        "    data shape / semantics -> docs/data-semantics.md\n"
        "    atomic technical traps -> docs/memory/ (one fact per file, hook in INDEX.md)\n"
        "    current model state    -> docs/STATUS.md\n"
        "and leave a one-line pointer behind. Linked docs cost nothing; only AGENTS.md is "
        "loaded eagerly."
    )
