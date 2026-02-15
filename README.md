# Research Lab (rlab)

Manage dirty research runs by automatically tracking Git state and restoring code from MLflow.

## Why Research Lab?

Research code is often "dirty" — you make quick changes, run a script, and continue editing before the run finished. Eventually, you have results in MLflow but no easy way to get the exact code that produced them.

`rlab` solves this by:

1.  **Capturing Git State:** Every run logs the base commit SHA and a `.patch` file (staged + unstaged + untracked changes).
2.  **Restoring Runs:** A CLI to checkout a new branch at the base commit and apply the saved patch.
3.  **Comparing Runs:** A CLI to see the code diff between any two MLflow runs.

## Installation

```bash
pip install .
# or using uv
uv pip install .
```

## Usage

### 1. In your training script

```python
from researchlab import ExperimentTracker

# Use as a context manager
with ExperimentTracker(experiment_name="my_awesome_project") as tracker:
    # (Optional) Log your YAML configuration
    # This logs params to MLflow and captures changes to the config file in the patch
    tracker.log_config("params.yaml")

    # Your training logic here...
    print(f"Running experiment: {tracker.run_name}")
```

### 2. CLI for Code Management

`rlab` provides a command-line interface to manage your experiment branches.

#### List experiment branches

```bash
rlab list
```

#### Restore a run

Find the `run_id` (e.g., `2026-02-15_radiant-octopus`) from the MLflow UI or the console output.

```bash
rlab restore 2026-02-15_radiant-octopus
```

This will:

1.  Find the run in MLflow.
2.  Create a new branch `experiment/2026-02-15_radiant-octopus` from the recorded base commit.
3.  Apply the saved patch to that branch.

#### Diff two runs

Compare the code state of two different runs:

```bash
rlab diff run_id_1 run_id_2
```

#### Delete an experiment branch

```bash
rlab delete run_id_1
```

## How it works

- **Run ID:** Generated as `YYYY-MM-DD_cool-slug` for readability.
- **Git Patch:** We use `git add -N` (intent-to-add) temporarily to ensure untracked files are included in the `git diff` output.
- **MLflow Tags:** We store `rlab.base_commit` and `rlab.run_id` as tags in MLflow for easy searching.
- **Diffing:** The `diff` command clones the repository into temporary directories to ensure a clean "three-way" comparison between the base commits and their respective patches.

## Project Evolution & Naming

As this project grows beyond Git state tracking to a full-featured research management suite, we might consider renaming it to better reflect its specific focus. Current candidates include:

- **RunSnap:** Highlights the "snapshot" nature of capturing code at runtime.
- **CodeTrace:** Emphasizes the lineage between results and source.
- **PatchLab:** Focuses on the unique three-way diff/patch restoration solution.
- **DirtyRun:** A name that addresses the core problem head-on.
