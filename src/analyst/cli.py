from pathlib import Path
from typing import Annotated

import typer

from .chunk import chunk_repo
from .config import Settings, StaleError
from .index import build_index, save_index
from .llm import ask_model
from .logging_setup import setup_logging
from .retrieve import build_prompt, retrieve
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
def index(root: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    settings = Settings()
    if settings.gemini_api_key is None:
        raise typer.BadParameter("api key not set, add to .env")
    api_key = settings.gemini_api_key.get_secret_value()
    files = scan_repo(root,max_file_lines=settings.max_file_lines)
    chunks, skipped = chunk_repo(root,files)
    if skipped:
        typer.echo(f"Skipped {len(skipped)} files that don't parse")

    idx = build_index(chunks, api_key=api_key, model=settings.embedding_model)
    path = save_index(root, idx)
    typer.echo(f"Indexed {len(idx.entries)} chunks from {len(files)} files -> {path}")


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

@app.command()
def ask(
    question: str,
    root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path("."),
    k: int = 20
    ) -> None:
    """ask the model a question about the code base"""
    settings = Settings()
    if settings.gemini_api_key is None:
            raise typer.BadParameter("api key not set, add to .env")
    api_key = settings.gemini_api_key.get_secret_value()
    try:
        chunks = retrieve(question,
                                   root,settings.embedding_model,api_key,k)
    except FileNotFoundError:
        typer.echo(f"No index found in {root}; run `analyst index` first", err=True)
        raise typer.Exit(1) from None
    except StaleError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    prompt = build_prompt(question,chunks)
    answer = ask_model(prompt,api_key=api_key,model=settings.model)
    typer.echo(answer)




    
