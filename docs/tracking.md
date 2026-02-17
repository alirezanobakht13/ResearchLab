# Tracking Module

The `researchlab.tracking` module provides tools to ensure that every experiment you run can be exactly reproduced, even if you didn't commit your code.

## Experiment Tracker

The core of the tracking module is the `ExperimentTracker` context manager. It wraps your training code and automatically:
1.  Starts an MLflow run.
2.  Captures the current Git commit hash.
3.  Generates a patch file containing all uncommitted changes (staged, unstaged, and untracked).
4.  Logs these as tags and artifacts to MLflow.

### Usage

```python
from researchlab import ExperimentTracker

def train():
    with ExperimentTracker(experiment_name="my_experiment") as tracker:
        # Code execution here is now tracked.
        # If you change a file and run this, the change is captured in the patch.
        print(f"Run ID: {tracker.run_name}")
        
        # Log hyperparameters (automatically flattened)
        tracker.log_config("config.yaml")
        
        # ... training loop ...
```

## CLI Reference

`rlab` provides a command-line interface to interact with these tracked experiments.

### `rlab list`
Lists all local branches that have been restored from experiments.

### `rlab restore <run_id>`
Restores the code state of a specific run into a new git branch.
*   Creates a branch `experiment/<run_id>` from the base commit.
*   Applies the captured patch.

### `rlab diff <run_id_1> <run_id_2>`
Performs a three-way diff between the code states of two runs. This is incredibly useful for debugging regressions ("It worked yesterday, what changed?").
