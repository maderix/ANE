# Community Benchmark Submissions

This folder is for reproducible hardware benchmark submissions from the community.

## Goals

- Make cross-chip results easy to compare.
- Keep raw logs attached so numbers are auditable.
- Keep submissions lightweight and low-maintenance.

## Submission Layout

Use one directory per machine/date:

`benchmarks/submissions/<chip>-<machine>-<YYYY-MM-DD>/`

Required files:

- `README.md` — short summary of machine, commands, and key results
- `metrics.json` — machine-readable summary of key metrics
- `raw/` — raw command outputs (`*.log`, `system_info.txt`, `upstream_commit.txt`)

## Privacy

Please redact machine serial numbers, UUIDs, and other unique identifiers before committing logs.

## Minimal Repro Guidance

Each submission should include:

- exact upstream commit hash tested
- exact commands run
- fixed step counts for training comparisons (for example, `--steps 20`)
- clear pass/fail status for each benchmark
