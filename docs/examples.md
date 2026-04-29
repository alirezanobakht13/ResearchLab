# Examples

## Tracking Run Lifecycle

This example shows one tracked experiment run with config logging and metric logging.

```python
import mlflow
from researchlab import ExperimentTracker

with ExperimentTracker(experiment_name="demo_project") as tracker:
    tracker.log_config("config.yaml")

    for step in range(10):
        mlflow.log_metric("loss", 1.0 / (step + 1), step=step)
        mlflow.log_metric("accuracy", 0.5 + (step * 0.03), step=step)

print(f"Run captured: {tracker.run_name}")
```
