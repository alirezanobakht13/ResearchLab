from typer.testing import CliRunner

from researchlab import ExperimentTracker
from researchlab.tracking.cli import app

runner = CliRunner()


def test_cli_lifecycle(mock_repo):
    repo_path, repo = mock_repo

    # 1. Create a run with a patch
    exp_file = repo_path / "experiment.txt"
    exp_file.write_text("experimental code")

    with ExperimentTracker(experiment_name="test_cli") as tracker:
        run_id = tracker.run_name

    # Reset repo to clean state to test restoration
    repo.git.reset("--hard", "HEAD")
    if exp_file.exists():
        exp_file.unlink()

    # 2. Test List (should be empty initially)
    result = runner.invoke(app, ["list"])
    assert "No experiment branches found" in result.stdout

    # 3. Test Restore
    result = runner.invoke(app, ["restore", run_id])
    assert result.exit_code == 0
    assert f"Successfully restored run {run_id}" in result.stdout

    assert f"experiment/{run_id}" in [b.name for b in repo.branches]
    assert exp_file.read_text() == "experimental code"

    # 4. Test List again
    result = runner.invoke(app, ["list"])
    assert f"experiment/{run_id}" in result.stdout

    # 5. Test Delete
    result = runner.invoke(app, ["delete", run_id])
    assert result.exit_code == 0
    assert f"experiment/{run_id}" not in [b.name for b in repo.branches]


def test_cli_diff(mock_repo):
    repo_path, repo = mock_repo
    file_path = repo_path / "file.txt"

    # Create run 1
    file_path.write_text("version 1")
    with ExperimentTracker(experiment_name="diff_test") as tracker1:
        run_id_1 = tracker1.run_name

    # Create run 2
    file_path.write_text("version 2")
    with ExperimentTracker(experiment_name="diff_test") as tracker2:
        run_id_2 = tracker2.run_name

    result = runner.invoke(app, ["diff", run_id_1, run_id_2])
    assert result.exit_code == 0
    assert "-version 1" in result.stdout
    assert "+version 2" in result.stdout
