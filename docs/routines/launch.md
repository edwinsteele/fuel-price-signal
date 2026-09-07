# Launch routine — prompt source

Canonical, tracked home for the prompt used by the "fuel-price-signal launch" local scheduled
task (`fuel-price-signal-launch`, cron `0 21 * * *` local — 9:00 PM AEST daily, chosen to finish
this routine's short active burst before the 10pm–4am AEST weekday Claude peak window).

## This is a LOCAL routine, not a cloud one

Registered in the Claude app on the owner's Mac. It needs the Mac awake (parent design: fps-3jj)
and relies on the ambient `gh`/`git` credentials that already work locally.

Since the 2026-09 tracker cutover it needs nothing else: `launch.py` talks to GitHub Issues
through `gh` and has no second store to sync. CLAUDE.md's cloud-worker Dolt-push token
gymnastics (fps-sk0) are gone with the thing they worked around — if you find that workaround
still described anywhere, it is stale.

It must also be a **separate** scheduled task from the chore/polish worker: that worker exits
immediately whenever any `claude-authored` PR is open, so sharing one routine between both jobs
would mean one un-merged chore PR silently stops every experiment for days.

## The shim

This is the exact text that should be the scheduled task's stored prompt (the `SKILL.md` body):

```text
You are the fuel-price-signal launch routine.
Your working directory is /Users/esteele/Code/fuel-price-signal (the
fuel-price-signal repo, primary worktree — this is a persistent local checkout, not
a fresh clone).
Follow docs/routines/launch.md exactly.
```

Three lines, nothing else — who you are, where the repo is, where the rules live. Same pattern
as `docs/routines/worker.md` (fps-6zj): instructions duplicated into the scheduler are
instructions nobody remembers to update.

## What to do when this fires

1. Run:
   ```bash
   PYTHONPATH=. uv run python -m experiments.pipeline.launch
   ```
   This single command does everything: stale-claim recovery (see below), claim the oldest
   ready `experiment`-labelled GitHub issue, validate its candidate module, launch the runner
   detached, and exit. Implementation: `experiments/pipeline/launch.py`.
2. Read its output:
   - `[launch] no experiment work ready` — the queue is empty. Nothing to do; exit quietly.
   - `[launch] #<N>: launched detached pid=<pid> log=<path>` — success. The run continues for
     hours with no Claude involvement; the dossier routine (fps-3jj.6, not yet built) picks up
     the finished artifacts later.
   - `[launch] #<N>: aborted before launch (release|block) — <reason>` — the candidate failed
     validation (PIT leak, missing columns, bad `INPUTS`/`COLUMNS` declaration, etc.). The
     claim was already released (or blocked, if its retry budget was spent) and commented on
     the issue; nothing further to do.
   - `[launch] recovered stale claim #<N>` — a claim went back on the queue. Two causes, both
     commented on the issue with the reason (fps-g31):
     - the run crashed mid-flight (no `results.json`, `run.log` ends in a traceback, claimed
       more than 12h ago), or
     - the run **finished** in a retryable status — `aborted_pipeline` (the pipeline itself was
       misconfigured) or `aborted_environment` (DB/disk/OOM). No age gate on this one: a
       `results.json` existing is proof the run is over. The candidate was never actually
       tested, so it is re-queued rather than written up; the dossier routine skips these too.

     A candidate cannot keep reappearing here indefinitely — fps-rtd fixed that, and PR #304
     hardened it. `claim_next_candidate` takes the oldest unassigned issue, and a released
     issue keeps its original creation date, so an *unbounded* release would always be
     re-claimed ahead of everything else and a persistent fault would starve the whole queue.
     `MAX_RETRIES = 1` bounds it: the second give-up on the same claim blocks instead of
     releasing. See the two output lines below.
   - `[launch] blocked stale claim #<N>` — that claim had already spent its one retry and
     failed again. It is now assigned to the owner and carries the `blocked` label, which
     together drop it out of the claim query (see **Stale-claim recovery**). **This one wants a
     human.** Fix the underlying fault, then hand it back:
     ```bash
     gh issue edit <N> --remove-label blocked --remove-label retried --remove-assignee @me
     ```
   - `[launch] reset retry budget on stale claim #<N>` — a claim that had spent a retry
     eventually reached a real verdict, so the `retried` label was cleared. Bookkeeping only;
     nothing to do.
3. On an unexpected non-zero exit (a genuine bug, not one of the above), surface the traceback —
   don't retry blindly, and don't fall back to running `gh`/`git` commands by hand to route around
   it. This routine's only job is a 10-minute burst; if it's broken, report it and stop.
4. **Nothing to commit — and that is now structural, not luck.** This routine writes no repo
   files of its own, and since the cutover its claim state lives entirely on GitHub (assignees,
   labels, comments), not in a tracked file. Under bd it did dirty the tree: `bd` recorded status
   transitions as rows in the git-tracked `.beads/interactions.jsonl`, so a claim or a release
   left an uncommitted file behind in a routine whose whole design is "launch and exit", with
   nobody watching. That failure mode is gone with `.beads/`. A `git status --short` on the way
   out still costs nothing if you want the reassurance.

   **Known lag, not a fault:** `gh issue list` takes ~7s to show a *newly created* issue
   (measured 2026-09-08). A candidate filed in the last few seconds before this routine fires
   is therefore picked up the following night, not this one. What was measured is issue
   creation and **assignee** changes — assignee writes were reflected at the first read. Label
   writes (`retried`, `blocked`) were not measured, and the retry budget does not rely on them
   being fast: `release_stale_claim` adds the `retried` label *before* it unassigns, so the
   write that makes an issue claimable again is always the later of the two. Ordering is what
   protects that, not latency. See `ISSUE_LIST_LIMIT` in `launch.py` for the measurement and
   for why the query is the `--label` form rather than `--search`.

## Candidate-issue convention

`experiments/pipeline/launch.py` is the consumer of `experiment`-labelled GitHub issues; the
generator session (fps-3jj.7, not yet written) is the producer and must file issues this way. An
issue's **body** must contain two lines, parsed by `parse_candidate_ref()`:

```text
Batch: experiments/batches/<batch-name>
Module: experiments/candidates/<batch-name>/<candidate-name>.py
```

Paths are repo-root-relative, and both must resolve inside `experiments/` — `parse_candidate_ref()`
rejects a `..` traversal or an absolute path pointing anywhere else, since `Module` gets
`exec_module`'d unattended. The runner's output directory is `default_out_dir(Module)` —
`Module`'s path with its `.py` suffix stripped, a per-candidate subdirectory (e.g.
`experiments/candidates/<batch-name>/<candidate-name>.py` ->
`experiments/candidates/<batch-name>/<candidate-name>/`) — **not** `Module`'s parent dir, which
is the whole batch directory shared by every candidate filed against it (fps-icv: candidate 2+
in a batch used to overwrite candidate 1's artifacts there). That per-candidate directory is
where `run.log`, `results.json`, `rowpreds.parquet`, and `fills.parquet` land, and where
stale-claim recovery looks for them.

## Queue isolation

Experiment issues carry a fourth label, `experiment`, alongside whatever type label they'd
otherwise get. This needs no code on the worker side: the chore/polish worker's pickup rule only
ever queries the `chore` and `polish` labels (CLAUDE.md) — an issue carrying only `experiment` is
structurally invisible to it. `launch.py` lists open `experiment`-labelled issues, keeps the
unassigned and un-`blocked` ones, takes the oldest by creation date, and claims it with
`gh issue edit <N> --add-assignee "@me"`.

**Assignee is the claim.** bd had an explicit `in_progress` status; GitHub does not, and does not
need one — an `experiment` issue with an assignee is being worked, and one without is queued.

⚠️ **Do not self-assign an `experiment` issue to investigate it. Add the `blocked` label
instead.** bd's claim marker was a status, so assigning a bead to yourself left the routine's
sweep alone. GitHub's marker is an identity, and this routine runs as the owner — so an issue
you assign to yourself is indistinguishable from one the routine claimed. The stale sweep will
treat it as its own stale claim: release it, spend a retry, and relaunch the detached runner
over the top of whatever you were doing. The sweep compares assignee logins against its own, so
*other* people's claims are safe from this; yours cannot be, because the routine runs as you.
`blocked` is the one signal both the sweep and the claim query honour:

```bash
gh issue edit <N> --add-label blocked          # park it; the routine will not touch it
gh issue edit <N> --remove-label blocked       # hand it back
```

Two labels carry what bd held as status and metadata: `blocked` (retry budget spent, a human must
clear the fault) and `retried` (this claim has spent its one retry). Neither excludes anything on
its own, so a blocked issue is *also* assigned to the owner — that assignment, not the label, is
what actually keeps it out of the claim query.

## Stale-claim recovery

Mirrors CLAUDE.md's chore/polish worker pickup rule 4, adapted to the experiment queue. For every
open, un-`blocked` `experiment` issue **assigned to this routine's own login** (see the warning
above for what that does and doesn't protect): resolve its `(batch_dir, candidate_path)`, look at
`default_out_dir(candidate_path)`. It's stale iff `results.json` is absent, `run.log` exists and its tail
looks like a Python traceback, and the claim hasn't been touched for more than 12 hours (long
enough to cover a real multi-hour run; short enough that a crash isn't lost for days). Recovery
posts the traceback to the issue and unassigns it — the next night's launch (or a manual re-run)
will pick it back up.

The 12h clock reads GitHub's `updatedAt`, the only timestamp the list endpoint carries; bd had a
dedicated `started_at` that a comment did not move. So a comment on a live claim restarts the
clock. That errs in the safe direction — it delays a release, never causes an early one — and a
claim nobody is touching still ages out normally.

A second give-up on the same claim **blocks** instead of releasing (`MAX_RETRIES = 1`, fps-rtd),
and a claim that reaches a real verdict has its `retried` label cleared so a later, unrelated
re-run of the same issue gets the full budget again.

## Detachment implementation note

`launch_detached()` uses `subprocess.Popen(..., start_new_session=True)`, not a shelled-out
`setsid nohup ... &`. The parent design (fps-3jj) describes the shell recipe as prose, not a
literal requirement, and `setsid` the CLI tool is a Linux util-linux binary not guaranteed present
on macOS — where this routine actually runs. `start_new_session=True` calls `os.setsid()` in the
forked child before exec, giving the same guarantee (new session, no controlling terminal, immune
to the parent's SIGHUP) via the standard library, portably.
