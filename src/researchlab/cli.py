import tempfile
from pathlib import Path

import git
import mlflow
import typer

from .utils import find_run_by_rlab_id

app = typer.Typer(help="Research Lab CLI - Manage dirty runs and experiment branches.")

@app.command()
def restore(run_id: str):
    """Restore a run by creating a new branch and applying the saved patch.

    This command finds the run in MLflow, retrieves the base commit and patch,
    creates a new branch from the base commit, and applies the patch to it.

    Args:
        run_id: The readable rlab run ID to restore.
    """
    run = find_run_by_rlab_id(run_id)
    if not run:
        typer.echo(f"Error: Run {run_id} not found in MLflow.", err=True)
        raise typer.Exit(code=1)

    base_commit = run.data.tags.get("rlab.base_commit")
    if not base_commit:
        typer.echo(f"Error: Base commit not found for run {run_id}.", err=True)
        raise typer.Exit(code=1)

    repo = git.Repo(".", search_parent_directories=True)
    branch_name = f"experiment/{run_id}"

    if branch_name in repo.branches:
        typer.echo(f"Error: Branch {branch_name} already exists.")
        raise typer.Exit(code=1)

    # Create new branch from base commit
    typer.echo(f"Creating branch {branch_name} from commit {base_commit}...")
    new_branch = repo.create_head(branch_name, base_commit)
    new_branch.checkout()

    # Download and apply patch
    client = mlflow.tracking.MlflowClient()
    local_path = Path(client.download_artifacts(run.info.run_id, "run.patch"))
    
    if local_path.exists():
        typer.echo(f"Applying patch from {local_path}...")
        try:
            repo.git.apply(str(local_path))
            typer.echo(f"Successfully restored run {run_id} to branch {branch_name}")
        except git.exc.GitCommandError as e:
            typer.echo(f"Error applying patch: {e}", err=True)
            raise typer.Exit(code=1) from e
    else:
        typer.echo("Note: No patch file found for this run (clean run).")

@app.command(name="list")
def list_branches():
    """List all rlab experiment branches.

    Scans the local git repository for branches starting with 'experiment/'.
    """
    repo = git.Repo(".", search_parent_directories=True)
    exp_branches = [b.name for b in repo.branches if b.name.startswith("experiment/")]
    
    if not exp_branches:
        typer.echo("No experiment branches found.")
    else:
        for b in exp_branches:
            typer.echo(b)

@app.command()
def delete(run_id: str, force: bool = False):
    """Delete an experiment branch.

    Args:
        run_id: The readable rlab run ID of the branch to delete.
        force: If True, force delete the branch even if it's not merged.
    """
    repo = git.Repo(".", search_parent_directories=True)
    branch_name = f"experiment/{run_id}"
    
    if branch_name not in repo.branches:
        typer.echo(f"Branch {branch_name} not found.")
        raise typer.Exit(code=1)

    typer.echo(f"Deleting branch {branch_name}...")
    
    if repo.active_branch.name == branch_name:
        # Try to find a safe branch to switch to
        for fallback in ["main", "master"]:
            if fallback in repo.branches:
                typer.echo(f"Switching to {fallback} before deletion...")
                repo.branches[fallback].checkout()
                break
        else:
            # If no main/master, just checkout the first non-experiment branch
            for b in repo.branches:
                if b.name != branch_name:
                    typer.echo(f"Switching to {b.name} before deletion...")
                    b.checkout()
                    break
    
    repo.delete_head(branch_name, force=force)
    typer.echo("Done.")

@app.command()
def diff(run_id_1: str, run_id_2: str):
    """Compare the code states of two runs.

    Downloads artifacts for both runs, restores them in temporary directories,
    and shows the git diff between them.

    Args:
        run_id_1: The first readable rlab run ID.
        run_id_2: The second readable rlab run ID.
    """
    run1 = find_run_by_rlab_id(run_id_1)
    run2 = find_run_by_rlab_id(run_id_2)

    if not run1 or not run2:
        missing = []
        if not run1:
            missing.append(run_id_1)
        if not run2:
            missing.append(run_id_2)
        typer.echo(f"Error: Run(s) {', '.join(missing)} not found.", err=True)
        raise typer.Exit(code=1)

    client = mlflow.tracking.MlflowClient()

    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = Path(tmpdir) / "run1"
        path2 = Path(tmpdir) / "run2"

        repo = git.Repo(".", search_parent_directories=True)
        
        for run, path, rid in [(run1, path1, run_id_1), (run2, path2, run_id_2)]:
            path.mkdir()
            # Clone current repo to temp path to have the history
            temp_repo = git.Repo.clone_from(repo.working_dir, path)
            base_commit = run.data.tags.get("rlab.base_commit")
            temp_repo.head.reference = temp_repo.commit(base_commit)
            temp_repo.head.reset(index=True, working_tree=True)
            
            patch_path = Path(client.download_artifacts(run.info.run_id, "run.patch"))
            if patch_path.exists() and patch_path.stat().st_size > 0:
                try:
                    temp_repo.git.apply(str(patch_path))
                except git.exc.GitCommandError as e:
                    typer.echo(f"Warning: Could not apply patch for {rid}: {e}")

        # Run git diff between the two directories
        try:
            diff_output = repo.git.diff("--no-index", str(path1), str(path2))
            diff_output = diff_output.replace(str(path1), f"run/{run_id_1}")
            diff_output = diff_output.replace(str(path2), f"run/{run_id_2}")
            typer.echo(diff_output)
        except git.exc.GitCommandError as e:
            if e.status == 1:
                diff_output = e.stdout
                diff_output = diff_output.replace(str(path1), f"run/{run_id_1}")
                diff_output = diff_output.replace(str(path2), f"run/{run_id_2}")
                typer.echo(diff_output)
            else:
                typer.echo(f"Error calculating diff: {e}", err=True)

if __name__ == "__main__":
    app()
