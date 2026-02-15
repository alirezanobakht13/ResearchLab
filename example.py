import random
import time
from pathlib import Path

import mlflow
import yaml

from researchlab import ExperimentTracker


def main():
    # 1. Prepare a dummy configuration file
    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "model": "ResNet-18",
        "optimizer": "Adam",
    }
    config_path = Path("config.yaml")
    with config_path.open("w") as f:
        yaml.dump(config, f)

    print("--- Starting Research Lab Example ---")

    # 2. Use ExperimentTracker as a context manager
    # This automatically:
    # - Starts an MLflow run
    # - Generates a readable run ID
    # - Captures Git state (base commit + patch of all changes)
    with ExperimentTracker(experiment_name="example_project") as tracker:
        print(f"Run ID: {tracker.run_name}")

        # 3. Log the configuration
        # This logs the YAML content as MLflow parameters and as an artifact
        tracker.log_config(str(config_path))

        # 4. Simulate a training loop with metric logging
        print("Simulating training...")
        for epoch in range(10):
            # Simulate metrics
            loss = 1.0 / (epoch + 1) + random.uniform(0, 0.1)
            accuracy = 0.5 + (epoch * 0.04) + random.uniform(0, 0.02)

            # Log metrics using standard mlflow calls (since we are inside the tracker context)
            mlflow.log_metric("loss", loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)

            time.sleep(0.1)  # Simulate work
            print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.4f}")

    print("\n--- Example Finished ---")
    print("Code state and metrics have been logged to MLflow.")
    print(f"To restore this exact code later, use: rlab restore {tracker.run_name}")

    # Clean up dummy config
    if config_path.exists():
        config_path.unlink()


if __name__ == "__main__":
    main()
