from pathlib import Path

import mlflow
import yaml

from .utils import generate_run_id, get_git_state, log_flattened_params


class ExperimentTracker:
    """A context manager for tracking research experiments with Git state.

    This class automates the process of starting an MLflow run, capturing the
    current Git state (base commit and diff), and logging artifacts and
    parameters.

    Example:
        ```python
        from researchlab import ExperimentTracker

        with ExperimentTracker(experiment_name="my_project") as tracker:
            tracker.log_config("config.yaml")
            # Your training logic here
            print(f"Run ID: {tracker.run_name}")
        ```

    Attributes:
        run_name (str): The unique, readable ID for this run.
        active_run (Optional[Run]): The currently active MLflow run.
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
    ):
        """Initializes the ExperimentTracker.

        Args:
            experiment_name: The name of the MLflow experiment.
            run_name: Optional custom name for the run. If not provided,
                a readable ID is generated automatically.
            tracking_uri: Optional MLflow tracking URI to use.
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        
        self.run_name = run_name or generate_run_id()
        self.active_run = None

    def __enter__(self) -> "ExperimentTracker":
        """Starts the MLflow run and captures the Git state.

        Returns:
            ExperimentTracker: The instance of the tracker.
        """
        self.active_run = mlflow.start_run(run_name=self.run_name)
        
        # Capture and log Git state
        git_state = get_git_state()
        mlflow.set_tag("rlab.base_commit", git_state["base_commit"])
        mlflow.set_tag("rlab.run_id", self.run_name)
        
        # Save patch file as an artifact
        if git_state["patch"]:
            patch_path = Path("run.patch")
            with patch_path.open("w") as f:
                f.write(git_state["patch"])
            mlflow.log_artifact(str(patch_path))
            patch_path.unlink()
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ends the MLflow run."""
        mlflow.end_run()

    def log_config(self, config_path: str):
        """Logs a YAML configuration file as parameters and an artifact.

        Recursively parses the YAML file and logs each entry as an MLflow
        parameter. Also uploads the file itself as an artifact.

        Args:
            config_path: Path to the YAML configuration file.
        """
        path = Path(config_path)
        if not path.exists():
            print(f"Warning: Config file {config_path} not found.")
            return

        with path.open() as f:
            try:
                config = yaml.safe_load(f)
                if isinstance(config, dict):
                    log_flattened_params(config)
                mlflow.log_artifact(str(path))
            except yaml.YAMLError as e:
                print(f"Error parsing YAML config: {e}")
