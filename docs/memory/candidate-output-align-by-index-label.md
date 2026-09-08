---
name: candidate-output-align-by-index-label
description: Candidate pipeline output must be aligned by index label; positional .to_numpy() silently pairs rows with the wrong station.
metadata:
  type: project
---

Candidate-pipeline (fps-3jj) output must be aligned by INDEX LABEL, never positionally. differential_pit_test (experiments/lib/pit_test.py) compares by index label and its docstring explicitly permits a candidate's add_columns/add_axis to return rows in a different order (a df.sort_values() inside the candidate is blessed). So any consumer that does axis_series.to_numpy() / .values and zips it against frame's columns silently pairs each row with the WRONG station_code/date — validation passes, the run completes, and only the verdict is wrong. Use .reindex(frame.index) before .to_numpy(), or .loc[]. This shipped as a real bug in runner.py's _build_axis_lookup (found in the fps-hvi review of PR #299); note DataFrame['col'] = series is already label-aligned by pandas, which is why the two sites in the same function disagreed.
