---
name: test-discrimination-claims-need-mutation
description: A claim about a test's discriminating power needs the mutation run, and needs you to read the line that answers the question.
metadata:
  type: feedback
---

A claim about a TEST's discriminating power ('this test catches X', 'N cases fail if you break Y') can only be established by RUNNING the mutation — and running it is only half the step. You must also read the part of the output that ANSWERS the question, not the part that happens to be on screen. In PR #361 the same docstring carried a wrong discrimination claim twice: 'this test fails if anyone reintroduces the second mechanism' (it does not; the alternative form decides identically), then 'three of these cases fail' (five do). The second was written WHILE the correct '5 failed' summary was printed one line below a 'tail -4' that had truncated the FAILED list to three. The CODE was right both times; the prose about what the code proves was wrong both times. Three habits. (1) Never write a discrimination claim you have not just run the mutation for. (2) 'tail -n' is a lossy summary chosen BEFORE you know what the answer looks like, and a count is exactly what it truncates — for any question of the form 'how many', grep the summary line or the full match set, never a fixed tail. This generalises past pytest to ruff, grep and long pipeline output. (3) When the claim carries a NUMBER, embed the mutation command in the docstring so the next reader re-derives it — a number in prose has no way to fail when the parametrize list grows underneath it. See [[prose-justifications-fail-silently]] for the general case.

**A fourth habit: clear `__pycache__` between the mutation and the restore.** CPython
invalidates cached bytecode on (mtime, size), so a mutation that is byte-identical in
LENGTH to what it replaced, restored inside the same one-second mtime tick, leaves a
STALE `.pyc` that pytest happily imports — the source on disk reads correct while the
test executes the mutant. Hit on 2026-09-10 mutating
`tests/test_agents_md_budget.py`'s `PROJECT_DOC_MAX_BYTES = 32 * 1024` to `20 * 1024`
and back: both spellings are 11 characters, `grep` confirmed the restore, the test alone
passed, and the very next full run failed reporting the 20,480 budget. The tell is an
assertion message quoting a value that is not in the file. `find . -name __pycache__
-type d -exec rm -rf {} +` before re-running, or make the mutated value a different
length. Note this also cuts the other way: a mutation that never took effect looks like
a test that fails to discriminate.
