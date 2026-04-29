# Welcome to Research Lab

**Research Lab (rlab)** makes ML experiments reproducible by tracking exact Git state per run.

It addresses one core challenge:
1. **Reproducibility of "Dirty" Runs:** experiments often run before commit. `rlab` captures exact state
   (Git SHA + uncommitted patch) for each run.

## Key Features

### Tracking Module (`researchlab.tracking`)
* **Context Manager:** `ExperimentTracker` logs Git state to MLflow.
* **CLI:** `rlab restore` and `rlab diff` to manage and inspect experiment code.
* **Dirty State:** captures staged, unstaged, and untracked files.

## Installation

```bash
uv add researchlab
```
