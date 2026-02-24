from invoke import task


@task
def build_docs(c):
    """Build the documentation."""
    c.run("mkdocs build")


@task(pre=[build_docs])
def serve_docs(c):
    """Serve the documentation locally."""
    c.run("mkdocs serve")
