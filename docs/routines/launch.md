# Launch routine — prompt source

Canonical, tracked home for the prompt used by the "fuel-price-signal launch" local scheduled
task (`fuel-price-signal-launch`, cron `0 21 * * *` local — 9:00 PM AEST daily, chosen to finish
this routine's short active burst before the 10pm–4am AEST weekday Claude peak window).

## This is a LOCAL routine, not a cloud one

Registered in the Claude app on the owner's Mac. It needs the Mac awake (parent design: fps-3jj)
and relies on ambient `bd`/`git` credentials that already work locally — none of CLAUDE.md's
cloud-worker Dolt-push token gymnastics (fps-sk0) apply here. Don't import that workaround.

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
   This single command does everything: `bd dolt pull`, stale-claim recovery (see below), claim
   the oldest ready `experiment`-labelled bd issue, validate its candidate module, launch the
   runner detached, `bd dolt push`, and exit. Implementation: `experiments/pipeline/launch.py`.
2. Read its output:
   - `[launch] no experiment work ready` — the queue is empty. Nothing to do; exit quietly.
   - `[launch] <id>: launched detached pid=<pid> log=<path>` — success. The run continues for
     hours with no Claude involvement; the dossier routine (fps-3jj.6, not yet built) picks up
     the finished artifacts later.
   - `[launch] <id>: aborted — <reason>` — the candidate failed validation (PIT leak, missing
     columns, bad `INPUTS`/`COLUMNS` declaration, etc.). The claim was already released and
     commented on the bead; nothing further to do.
   - `[launch] recovered stale claim <id>` — a claim went back on the queue. Two causes, both
     commented on the bead with the reason (fps-g31):
     - the run crashed mid-flight (no `results.json`, `run.log` ends in a traceback, claimed
       more than 12h ago), or
     - the run **finished** in a retryable status — `aborted_pipeline` (the pipeline itself was
       misconfigured) or `aborted_environment` (DB/disk/OOM). No age gate on this one: a
       `results.json` existing is proof the run is over. The candidate was never actually
       tested, so it is re-queued rather than written up; the dossier routine skips these too.

     **Watch for a candidate that keeps reappearing here (fps-rtd, P1, open).** There is no
     retry budget yet, and `claim_next_candidate` claims with `bd ready --sort oldest`, so a
     released bead — which keeps its original creation date — is always re-claimed ahead of
     everything else. A *persistent* fault therefore starves the whole queue: the same doomed
     candidate consumes the nightly slot indefinitely and nothing else runs. If you see the same
     id released on consecutive nights, fix the underlying fault or park the bead by hand; don't
     wait for the routine to move on, because it won't.
3. On an unexpected non-zero exit (a genuine bug, not one of the above), surface the traceback —
   don't retry blindly, and don't fall back to running `bd`/`git` commands by hand to route around
   it. This routine's only job is a 10-minute burst; if it's broken, report it and stop.

## Candidate-bead convention

`experiments/pipeline/launch.py` is the consumer of `experiment`-labelled bd issues; the
generator session (fps-3jj.7, not yet written) is the producer and must file beads this way. An
issue's **description** must contain two lines, parsed by `parse_candidate_ref()`:

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

Experiment beads carry a fourth label, `experiment`, alongside whatever type label they'd
otherwise get. This needs no code on the worker side: the chore/polish worker's pickup rule only
ever queries `bd ready --label chore` and `bd ready --label polish` (CLAUDE.md) — a bead carrying
only `experiment` is structurally invisible to it. `launch.py` queries
`bd ready --label experiment --unassigned --sort oldest -n 1 --claim`.

## Stale-claim recovery

Mirrors CLAUDE.md's chore/polish worker pickup rule 4, adapted to the experiment queue. For every
`in_progress` `experiment` bead: resolve its `(batch_dir, candidate_path)`, look at
`default_out_dir(candidate_path)`. It's stale iff `results.json` is absent, `run.log` exists and its tail
looks like a Python traceback, and the bead was claimed more than 12 hours ago (long enough to
cover a real multi-hour run; short enough that a crash isn't lost for days). Recovery posts the
traceback to the bead, unassigns it, and reopens it — the next night's launch (or a manual
re-run) will pick it back up.

## Detachment implementation note

`launch_detached()` uses `subprocess.Popen(..., start_new_session=True)`, not a shelled-out
`setsid nohup ... &`. The parent design (fps-3jj) describes the shell recipe as prose, not a
literal requirement, and `setsid` the CLI tool is a Linux util-linux binary not guaranteed present
on macOS — where this routine actually runs. `start_new_session=True` calls `os.setsid()` in the
forked child before exec, giving the same guarantee (new session, no controlling terminal, immune
to the parent's SIGHUP) via the standard library, portably.
