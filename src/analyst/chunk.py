"""given a .py file, parse it into an ast code"""

import ast
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .scan import FileInfo

logger = logging.getLogger(__name__)

ChunkKind = Literal["function", "class", "method", "module"]
DefNode = ast.FunctionDef | ast.AsyncFunctionDef


class Chunk(BaseModel):
    path: Path  # relative to the repo root, same as FileInfo.path
    kind: ChunkKind
    name: str  # "scan_repo", "FileInfo", "Settings.model_config", or "<module>"
    start_line: int  # 1-based, inclusive (includes decorators)
    end_line: int  # 1-based, inclusive
    text: str


def _parse_error_summary(e: SyntaxError | ValueError) -> str:
    """Short reason a file didn't parse, e.g. 'line 12: invalid syntax'."""
    lineno = getattr(e, "lineno", None)
    msg = getattr(e, "msg", None) or str(e)
    return f"line {lineno}: {msg}" if lineno else msg


def chunk_repo(
    root: Path, files: list[FileInfo]
) -> tuple[list[Chunk], dict[Path, str]]:
    """Chunk every scanned file, in the same order as `files`.
    returns the chunks and a summary about failed attemps"""
    chunks = []
    skipped = {}
    for info in files:
        try:
            curr_chunks = chunk_file(root, info)
        except (SyntaxError, ValueError) as e:
            cause = _parse_error_summary(e)
            logger.warning("Skipping %s: %s", info.path, cause)
            skipped[info.path] = cause
            continue
        chunks.extend(curr_chunks)
    return chunks, skipped


def chunk_file(root: Path, info: FileInfo) -> list[Chunk]:
    """Read one file from disk and chunk it."""
    full_path = root / info.path
    source_code = full_path.read_text(encoding="utf-8", errors="replace")
    return chunk_source(source_code, info.path)


def chunk_source(source: str, path: Path) -> list[Chunk]:
    lines = source.splitlines()
    chunks = []
    tree = ast.parse(source, filename=str(path))

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            chunks.extend(_class_chunks(node, lines, path))

        elif isinstance(node, DefNode):
            chunks.append(_def_chunks(node, lines, path))

    module = _module_chunk(tree, lines, path)
    if module is not None:
        chunks.append(module)

    return chunks


def _class_chunks(
    class_node: ast.ClassDef, lines: list[str], path: Path
) -> list[Chunk]:
    """top class chunk plus one more for each method"""
    start_line = _start_line(class_node)
    methods = [
        s
        for s in class_node.body
        if isinstance(s, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    assert class_node.end_lineno is not None  # always set for parsed code
    end_header = (
        min([_start_line(n) for n in methods]) if methods else class_node.end_lineno + 1
    )  # noqa: E501

    chunks = [
        Chunk(
            path=path,
            kind="class",
            name=class_node.name,
            start_line=start_line,
            end_line=end_header - 1,
            text="\n".join(lines[start_line - 1 : end_header - 1]),
        )
    ]

    for method_node in methods:
        start_method = _start_line(method_node)
        end_method = method_node.end_lineno
        chunks.append(
            Chunk(
                path=path,
                kind="method",
                name=f"{class_node.name}.{method_node.name}",
                start_line=start_method,
                end_line=end_method,
                text="\n".join(lines[start_method - 1 : end_method]),
            )
        )

    return chunks


def _def_chunks(node: DefNode, lines: list[str], path: Path) -> Chunk:
    """Top-level function chunk."""
    start_line = _start_line(node)
    end_line = node.end_lineno
    return Chunk(
        path=path,
        kind="function",
        name=node.name,
        start_line=start_line,
        end_line=end_line,
        text="\n".join(lines[start_line - 1 : end_line]),
    )


def _module_chunk(tree: ast.Module, lines: list[str], path: Path) -> Chunk | None:
    """Leftover top-level code (docstring, imports, constants) as one chunk,.

    The leftover statements can sit on both sides of defs/classes,so start_line/end_line
    span from the first to the last of them, text holds only the leftovers themselves.
    """
    leftovers = [n for n in tree.body if not isinstance(n, DefNode | ast.ClassDef)]
    if not leftovers:
        return None

    start_line = leftovers[0].lineno
    end_line = leftovers[-1].end_lineno
    text = "\n".join("\n".join(lines[n.lineno - 1 : n.end_lineno]) for n in leftovers)
    return Chunk(
        path=path,kind="module",name="<module>",
        start_line=start_line,end_line=end_line,
        text=text,)


def _start_line(node: DefNode | ast.ClassDef) -> int:
    """First line of a def/class, counting its decorators."""
    if node.decorator_list:
        return node.decorator_list[0].lineno  # decorators come first, top to bottom
    return node.lineno
