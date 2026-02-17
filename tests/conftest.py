import os
from pathlib import Path

import git
import mlflow
import pytest


@pytest.fixture
def mock_repo(tmp_path):
    """Creates a temporary git repository for testing."""
    repo_path = tmp_path / "mock_repo"
    repo_path.mkdir()
    repo = git.Repo.init(repo_path)

    # Configure git user
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test User")
        cw.set_value("user", "email", "test@example.com")
        cw.release()

    # Create an initial commit
    base_file = repo_path / "base.txt"
    base_file.write_text("initial content")
    repo.index.add([str(base_file)])
    repo.index.commit("Initial commit")

    # Change current working directory to the repo path
    old_cwd = Path.cwd()
    os.chdir(repo_path)

    yield repo_path, repo

    # Restore CWD
    os.chdir(old_cwd)


@pytest.fixture(autouse=True)
def mlflow_setup(tmp_path):
    """Sets up a local MLflow tracking URI for tests."""
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri
