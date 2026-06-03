# <experiment name>

- **Date:** YYYY-MM-DD
- **Branch:** main (or `scratch/<topic>` if iterating in isolation)
- **SHA:** <git rev-parse HEAD when the run happened>
- **Status:** open | done | graduated | abandoned

## Hypothesis
One or two sentences: what you expected and why.

## Setup
Features, model, CV protocol, seeds, anything non-default. Link to the run script in this dir if there is one.

## Results
Key metrics and delta vs the relevant baseline. Reference plots/CSVs by filename — they live in this dir but are gitignored.

## Conclusion
One line: graduate / abandon / iterate-next. If graduated, link the PR that landed the code.

## Followups
- Linked issue or next experiment dir, if any.
