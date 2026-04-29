# Research Lab (rlab)

Manage dirty research runs with reproducible Git state capture.

## Why Research Lab?

Research code often gets run before commit. Later, reproducing exact code state is hard.
`rlab` solves this by storing:

1. Base commit hash.
2. Patch for staged, unstaged, and untracked changes.

Everything is logged to MLflow per run.

## Installation

```bash
uv add researchlab
```

## Tracking Module (`researchlab.tracking`)

### Usage

```python
from researchlab import ExperimentTracker

with ExperimentTracker(experiment_name="my_project") as tracker:
    tracker.log_config("params.yaml")
    print(f"Running experiment: {tracker.run_name}")
```

### CLI

```bash
rlab list
rlab restore 2026-02-15_radiant-octopus
rlab diff run_id_1 run_id_2
```
