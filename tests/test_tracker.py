import mlflow
import yaml

from researchlab import ExperimentTracker
from researchlab.tracking.utils import get_git_state


def test_git_state_capture(mock_repo):
    repo_path, repo = mock_repo

    # Create some changes
    new_file = repo_path / "new_file.txt"
    new_file.write_text("new content")
    repo.index.add([str(new_file)])  # Staged change

    modified_file = repo_path / "base.txt"
    modified_file.write_text("modified content")  # Unstaged change

    state = get_git_state(str(repo_path))

    assert state["base_commit"] == repo.head.commit.hexsha
    assert "new content" in state["patch"]
    assert "modified content" in state["patch"]
    assert state["is_dirty"] is True


def test_experiment_tracker_logging(mock_repo, tmp_path):
    repo_path, repo = mock_repo

    # Create a config file
    config_dict = {"lr": 0.01, "layers": [10, 20]}
    config_path = repo_path / "config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_dict, f)

    # Create a change to be patched
    (repo_path / "change.txt").write_text("patch me")

    with ExperimentTracker(experiment_name="test_exp") as tracker:
        tracker.log_config(str(config_path))
        run_id = mlflow.active_run().info.run_id

    # Verify MLflow logs
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    assert run.data.tags["rlab.base_commit"] == repo.head.commit.hexsha
    assert run.data.params["lr"] == "0.01"
    assert run.data.params["layers"] == "[10, 20]"

    # Verify artifacts
    artifacts = [a.path for a in client.list_artifacts(run_id)]
    assert "run.patch" in artifacts
    assert "config.yaml" in artifacts
