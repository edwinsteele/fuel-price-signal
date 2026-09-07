"""Phase 3 of the Beads -> GitHub cutover: create the live issues on GitHub.

Two passes, because a dependency cannot be wired until both endpoints exist:

    pass A  create issues + comments, recording bd-id -> issue-number in MAP_PATH
    pass B  wire the 9 native edges (blocked-by / parent) from that map

Resumable: pass A skips any bd id already present in MAP_PATH, so a partial run can
be re-run without creating duplicates. --dry-run prints the plan and touches nothing.

What is deliberately NOT migrated (PLAN 2.1):
  - the 138 closed issues, except `fps-3jj` (PLAN 3.1: it is a closed parent with four
    OPEN children, and GitHub sub-issues need a live parent)
  - `blocks` edges whose target is closed -- a closed blocker is a discharged gate, not
    a live constraint; recorded as a provenance line in the body instead
  - `relates-to` edges -- no GitHub equivalent, and they carry no semantics
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
SNAP = HERE / "snapshot"
MAP_PATH = HERE / "id_map.json"
LIVE_STATUSES = {"open", "in_progress", "blocked"}
PARENT_EXCEPTION = "fps-3jj"        # PLAN 3.1
CLOSE_AS_MOOT = "fps-sk0"           # PLAN 3, pass A step 5 -- this cutover *is* its fix


def load(name):
    return json.loads((SNAP / name).read_text(encoding="utf-8"))


def gh(*args: str, dry: bool = False, capture: bool = True) -> str:
    if dry:
        print(f"    $ gh {' '.join(args)}")
        return ""
    r = subprocess.run(["gh", *args], capture_output=capture, text=True)
    if r.returncode != 0:
        sys.exit(f"FAIL: gh {' '.join(args)}\n{r.stderr}")
    return r.stdout.strip()


def build_body(rec: dict, by_id: dict, live: set) -> str:
    """bd description, plus a provenance footer carrying what GitHub cannot express."""
    parts = [rec.get("description") or "_(no description in bd)_", "", "---", ""]
    prov = [f"Migrated from bd `{rec['id']}` (created {rec['created_at'][:10]})."]

    if rec.get("issue_type") and rec["issue_type"] != "task":
        prov.append(f"bd issue type: `{rec['issue_type']}`.")
    if rec.get("external_ref"):
        prov.append(f"Pre-Beads GitHub issue: {rec['external_ref']}")
    if rec["id"] == PARENT_EXCEPTION:
        prov.append(
            "**Reopened during the cutover.** bd had this closed, but it has four open "
            "children; GitHub sub-issues require a live parent, so it is migrated as a "
            "tracking issue (PLAN_beads_cutover.md 3.1)."
        )

    dropped = []
    for dep in rec.get("dependencies") or []:
        tgt, typ = dep["depends_on_id"], dep["type"]
        if typ == "parent-child" or tgt in live:
            continue  # wired natively in pass B
        t = by_id.get(tgt)
        title = t["title"] if t else "(unknown)"
        when = (t.get("closed_at") or t.get("updated_at") or "")[:10] if t else ""
        ext = f" — {t['external_ref']}" if t and t.get("external_ref") else ""
        if typ == "blocks":
            dropped.append(f"- was blocked by `{tgt}` — {title} (completed {when}){ext}")
        else:
            dropped.append(f"- `{typ}` `{tgt}` — {title}{ext}")
    if dropped:
        prov.append(
            "\nDependencies on closed bd issues, discharged and not recreated "
            "(full corpus in `docs/bd-archive/`):\n" + "\n".join(dropped)
        )
    return "\n".join(parts) + "\n".join(prov) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pass", dest="which", choices=["A", "B"], required=True)
    args = ap.parse_args()
    dry = args.dry_run

    issues = load("issues.json")
    comments = load("comments.json")
    by_id = {i["id"]: i for i in issues}
    live = {i["id"] for i in issues if i["status"] in LIVE_STATUSES}
    migrating = sorted(live | {PARENT_EXCEPTION})

    id_map = json.loads(MAP_PATH.read_text()) if MAP_PATH.exists() else {}

    if args.which == "A":
        print(f"pass A — create {len(migrating)} issues "
              f"({len(live)} live + 1 parent exception)\n")
        for bid in migrating:
            rec = by_id[bid]
            if bid in id_map:
                print(f"  skip {bid} -> #{id_map[bid]} (already created)")
                continue
            labels = list(rec.get("labels") or []) + [f"P{rec['priority']}"]
            body = build_body(rec, by_id, live)
            print(f"  {bid:<16} P{rec['priority']} [{','.join(labels)}]")
            print(f"      {rec['title'][:80]}")
            if dry:
                print(f"      body {len(body)} chars, "
                      f"{len(comments.get(bid, []))} comment(s) to follow")
                continue
            url = gh("issue", "create", "--title", rec["title"], "--body", body,
                     *sum((["--label", l] for l in labels), []))
            num = int(url.rstrip("/").rsplit("/", 1)[-1])
            id_map[bid] = num
            MAP_PATH.write_text(json.dumps(id_map, indent=2, sort_keys=True) + "\n")
            print(f"      -> #{num}")
            for c in comments.get(bid, []):
                gh("issue", "comment", str(num), "--body",
                   f"_bd comment by {c['author']}, {c['created_at'][:10]}_\n\n{c['text']}")
            if comments.get(bid):
                print(f"      + {len(comments[bid])} comment(s)")
        if not dry:
            print(f"\nmap -> {MAP_PATH}")
        return

    # pass B
    print("pass B — wire native edges\n")
    wired = 0
    for bid in migrating:
        for dep in by_id[bid].get("dependencies") or []:
            tgt, typ = dep["depends_on_id"], dep["type"]
            if tgt not in id_map or bid not in id_map:
                continue
            if typ == "blocks":
                print(f"  #{id_map[bid]} ({bid}) blocked-by #{id_map[tgt]} ({tgt})")
                gh("issue", "edit", str(id_map[bid]),
                   "--add-blocked-by", str(id_map[tgt]), dry=dry)
                wired += 1
            elif typ == "parent-child":
                print(f"  #{id_map[bid]} ({bid}) parent #{id_map[tgt]} ({tgt})")
                gh("issue", "edit", str(id_map[bid]),
                   "--parent", str(id_map[tgt]), dry=dry)
                wired += 1
    print(f"\n{wired} edges wired (expected 9)")

    if CLOSE_AS_MOOT in id_map:
        n = id_map[CLOSE_AS_MOOT]
        print(f"\nclosing #{n} ({CLOSE_AS_MOOT}) as moot — this cutover is its fix")
        gh("issue", "close", str(n), "--reason", "not planned", "--comment",
           "Closed by the Beads -> GitHub cutover. This issue *was* the blocker: "
           "`bd dolt push` could not authenticate from a Routine sandbox because Dolt's "
           "git-remote push cannot satisfy an interactive credential prompt. Removing bd "
           "removes the failure mode entirely -- `gh` already authenticates there. "
           "See PLAN_beads_cutover.md phase 7.", dry=dry)


if __name__ == "__main__":
    main()
