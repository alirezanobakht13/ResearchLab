import contextlib
import datetime
from typing import Any

import coolname
import git
import mlflow
from mlflow.entities import Run


def generate_run_id() -> str:
    """Generates a unique and readable run ID.

    Returns:
        str: A string in the format YYYY-MM-DD_slug (e.g., 2026-02-15_radiant-octopus).
    """
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    slug = coolname.generate_slug(2)
    return f"{date_str}_{slug}"

def find_run_by_rlab_id(run_id: str) -> Run | None:
    """Finds an MLflow run by the custom rlab.run_id tag.

    Args:
        run_id: The readable rlab run ID to search for.

    Returns:
        Optional[Run]: The MLflow Run object if found, else None.
    """
    runs = mlflow.search_runs(
        filter_string=f"tags.'rlab.run_id' = '{run_id}'",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        output_format="list"
    )
    return runs[0] if runs else None

def get_git_state(repo_path: str = ".") -> dict[str, Any]:
    """Captures the current Git state including base commit and patch.

    This captures staged, unstaged, and untracked changes by temporarily 
    using git's "intent-to-add" feature. It restores the untracked state
    after generating the patch.

    Args:
        repo_path: Path to the git repository. Defaults to ".".

    Returns:
        dict[str, Any]: A dictionary containing:
            - base_commit (str): The HEX SHA of the current HEAD.
            - patch (str): The git diff output.
            - is_dirty (bool): True if there are any changes (including untracked).
    """
    repo = git.Repo(repo_path, search_parent_directories=True)
    
    # Base commit (HEAD)
    base_commit = repo.head.commit.hexsha
    
    # Identify untracked files to restore them later
    untracked_files = repo.untracked_files
    
    try:
        # To include untracked files in the patch, we use 'git add -N' (intent-to-add)
        # which makes them appear in the diff without actually staging their content.
        # This DOES NOT modify the actual file content on disk.
        for f in untracked_files:
            repo.git.add(f, intent_to_add=True)
        
        # Patch of all changes (staged, unstaged, and untracked-via-intent-to-add)
        patch = repo.git.diff(repo.head.commit)
        
        # Check if dirty (including untracked files)
        is_dirty = repo.is_dirty(untracked_files=True)
        
    finally:
        # Restore the untracked state in the index for files we touched.
        # repo.git.reset(f) is equivalent to 'git reset <file>', which only
        # affects the index and does NOT revert working tree changes.
        for f in untracked_files:
            with contextlib.suppress(git.exc.GitCommandError):
                repo.git.reset(f)
    
    return {
        "base_commit": base_commit,
        "patch": patch,
        "is_dirty": is_dirty
    }

def log_flattened_params(d: dict[str, Any], prefix: str = "") -> None:
    """Recursively logs a dictionary as flattened MLflow parameters.

    Args:
        d: The dictionary to log.
        prefix: Optional prefix for parameter keys (used for recursion).
    """
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            log_flattened_params(v, prefix=f"{key}.")
        else:
            mlflow.log_param(key, v)
