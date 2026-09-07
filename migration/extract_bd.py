"""Phase 1 of the Beads -> GitHub cutover: freeze everything bd holds.

Re-runnable and deterministic. PLAN_beads_cutover.md 5.3 requires re-running this
immediately before the phase-3 migration and diffing, because every worktree writes
through to the primary checkout's Dolt DB and another session may have written since.

Writes, under migration/snapshot/:
    issues.json     all 161 issues, every field bd exposes, including inline `dependencies`
    comments.json   issue_id -> [comment, ...]  (bd list omits comment bodies)
    memories.json   the 53 `bd remember` key -> text pairs
    edges.json      dependency edges, flattened, each tagged live/closed on both ends
    MANIFEST.json   counts + sha256 of each file, for the diff in phase 3

A single `bd list` read has been observed returning torn JSON, so every read is taken
twice and compared; a mismatch is a hard failure rather than a silently-half-captured
snapshot.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

SNAP = pathlib.Path(__file__).parent / "snapshot"
LIVE_STATUSES = {"open", "in_progress", "blocked"}


def bd_json(*args: str):
    """Run a bd command twice and require byte-identical output."""
    outs = []
    for _ in range(2):
        r = subprocess.run(["bd", *args, "--json"], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"FAIL: bd {' '.join(args)} exited {r.returncode}\n{r.stderr}")
        outs.append(r.stdout)
    if outs[0] != outs[1]:
        sys.exit(f"FAIL: unstable read from `bd {' '.join(args)}` -- DB written concurrently?")
    try:
        return json.loads(outs[0])
    except json.JSONDecodeError as exc:
        sys.exit(f"FAIL: `bd {' '.join(args)}` returned invalid JSON: {exc}")


def write(name: str, obj) -> str:
    path = SNAP / name
    blob = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(blob, encoding="utf-8")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def main() -> None:
    SNAP.mkdir(parents=True, exist_ok=True)

    issues = bd_json("list", "--status", "all")
    by_id = {i["id"]: i for i in issues}
    live = {i for i, r in by_id.items() if r["status"] in LIVE_STATUSES}

    # Comments are not in `bd list` output; pull them per issue that reports any.
    comments = {}
    for rec in issues:
        if rec.get("comment_count"):
            comments[rec["id"]] = bd_json("comments", rec["id"])

    # `bd memories --json` returns the key->text map with bd's JSON envelope key
    # `schema_version` mixed in at the top level. Left in, it inflates the count to 54
    # and would be migrated as a 54th "memory" whose body is the integer 1.
    memories = {k: v for k, v in bd_json("memories").items() if k != "schema_version"}

    edges = []
    for rec in issues:
        for dep in rec.get("dependencies") or []:
            src, tgt = dep["issue_id"], dep["depends_on_id"]
            edges.append({
                "source": src,
                "type": dep["type"],
                "target": tgt,
                "source_status": by_id.get(src, {}).get("status", "MISSING"),
                "target_status": by_id.get(tgt, {}).get("status", "MISSING"),
                "source_live": src in live,
                "target_live": tgt in live,
            })
    edges.sort(key=lambda e: (e["type"], e["source"], e["target"]))

    manifest = {
        "counts": {
            "issues_total": len(issues),
            "issues_live": len(live),
            "issues_closed": len(issues) - len(live),
            "memories": len(memories),
            "issues_with_comments": len(comments),
            "comments_total": sum(len(v) for v in comments.values()),
            "live_issues_with_comments": sum(1 for k in comments if k in live),
            "live_comments_total": sum(len(v) for k, v in comments.items() if k in live),
            "edges_total": len(edges),
            "edges_from_live": sum(1 for e in edges if e["source_live"]),
            "edges_live_to_live": sum(1 for e in edges if e["source_live"] and e["target_live"]),
        },
        "live_ids": sorted(live),
        "sha256": {
            "issues.json": write("issues.json", issues),
            "comments.json": write("comments.json", comments),
            "memories.json": write("memories.json", memories),
            "edges.json": write("edges.json", edges),
        },
    }
    write("MANIFEST.json", manifest)

    c = manifest["counts"]
    print(f"snapshot -> {SNAP}")
    for k, v in c.items():
        print(f"  {k:<22} {v}")

    # Acceptance gate (PLAN 3, phase 1): every live issue must round-trip whole.
    required = ("id", "title", "description", "status", "priority", "labels")
    bad = [i for i in issues if i["id"] in live and any(i.get(f) in (None, "") for f in required)]
    if bad:
        sys.exit(f"FAIL: {len(bad)} live issues missing required fields: {[i['id'] for i in bad]}")
    if len(memories) != 53:
        sys.exit(f"FAIL: expected 53 memories, got {len(memories)} -- envelope key leaking again?")
    print("\nOK: all live issues carry id/title/description/status/priority/labels")
    print("OK: 53 memories, no envelope key")


if __name__ == "__main__":
    main()
