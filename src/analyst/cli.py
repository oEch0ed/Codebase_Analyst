from pathlib import Path
from typing import Annotated

import typer

from .config import Settings
from .logging_setup import setup_logging
from .scan import scan_repo

app = typer.Typer()


@app.callback(no_args_is_help=True)
def main() -> None:
    """Codebase Analyst: Analyze your codebase with AI."""
    setup_logging(Settings().log_level)


@app.command()
def scan(root: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    """List the Python files that would be analysed."""
    settings = Settings()
    files = scan_repo(root=root, max_file_lines=settings.max_file_lines)
    for info in files:
        typer.echo(f"{info.lines:>6}  {info.path}")
    total = sum(info.lines for info in files)
    typer.echo(f"{len(files)} files, {total} lines")
