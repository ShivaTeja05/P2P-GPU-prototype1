# Contributing

## Running the tests

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
```

```bash
python -m pytest -q && ruff check p2pgpu tests examples
```

111 tests, no GPU required — everything that needs CUDA is skipped or verified
against a CPU reference, so the suite runs anywhere.

## What this project cares about

**Failures must be loud.** The bugs that hurt here are the silent ones: a split
model that produces plausible-but-wrong output, a cluster where each node trains
alone while the loss falls convincingly on both screens. If a change can fail
quietly, add the check that makes it fail loudly instead.

**Claims need a control.** "It works" means something measured it, and the
measurement had a control that could have failed. This repo has retracted
conclusions twice for want of one — see `docs/PROGRESS.md`, which keeps the
retractions rather than quietly editing them out.

**Comments explain why, not what.** The code says what. Reserve comments for the
reasoning a future reader cannot reconstruct — why this bind address, why this
split, why the optimizer state stays local.

## Reporting something broken

Include `p2pgpu doctor` output. Nearly every report is environment-specific and
that output answers most of the questions we would otherwise have to ask.
