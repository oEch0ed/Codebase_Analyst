from pathlib import Path
from typing import Annotated

import typer

from .chunk import chunk_repo
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


@app.command()
def chunk(root: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    """List the chunks each Python file splits into, and files that don't parse."""
    settings = Settings()
    files = scan_repo(root=root, max_file_lines=settings.max_file_lines)
    chunks, skipped = chunk_repo(root, files)
    for c in chunks:
        location = f"{c.path}:{c.start_line}-{c.end_line}"
        typer.echo(f"{location:<40}  {c.kind:<8}  {c.name}")

    typer.echo(f"{len(chunks)} chunks from {len(files) - len(skipped)} files")
    if skipped:
        typer.echo(f"Skipped {len(skipped)} files that don't parse:")
        for path, reason in skipped.items():
            typer.echo(f"  {path}  {reason}")
    
