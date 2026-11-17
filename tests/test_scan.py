from pathlib import Path

import pytest
from typer.testing import CliRunner

from analyst.cli import app
from analyst.scan import SKIP_DIRS, scan_repo


def write(path: Path, lines: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x = 1\n" * lines, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A small fake repo with source files, junk dirs and non-Python files."""
    write(tmp_path / "main.py", 3)
    write(tmp_path / "pkg" / "b.py", 2)
    write(tmp_path / "pkg" / "a.py", 1)
    write(tmp_path / "pkg" / "deep" / "c.py", 4)
    write(tmp_path / "README.md", 5)
    write(tmp_path / "script.pyc", 1)
    for skipped in SKIP_DIRS:
        write(tmp_path / skipped / "hidden.py", 1)
    return tmp_path


def paths(root: Path, max_file_lines: int = 1000) -> list[str]:
    return [info.path.as_posix() for info in scan_repo(root, max_file_lines)]


def test_finds_python_files_sorted_and_relative(repo: Path) -> None:
    assert paths(repo) == ["main.py", "pkg/a.py", "pkg/b.py", "pkg/deep/c.py"]


def test_counts_lines(repo: Path) -> None:
    lines = {info.path.as_posix(): info.lines for info in scan_repo(repo, 1000)}
    assert lines == {"main.py": 3, "pkg/a.py": 1, "pkg/b.py": 2, "pkg/deep/c.py": 4}


def test_skips_skip_dirs_at_any_depth(tmp_path: Path) -> None:
    write(tmp_path / "pkg" / "node_modules" / "x.py", 1)
    write(tmp_path / "pkg" / "ok.py", 1)
    assert paths(tmp_path) == ["pkg/ok.py"]


def test_excludes_files_over_the_limit(repo: Path) -> None:
    # the limit is inclusive: a 3-line file passes a 3-line limit
    assert paths(repo, max_file_lines=3) == ["main.py", "pkg/a.py", "pkg/b.py"]


def test_empty_file_counts_as_zero_lines(tmp_path: Path) -> None:
    (tmp_path / "empty.py").touch()
    [info] = scan_repo(tmp_path, 1000)
    assert info.lines == 0


def test_tolerates_non_utf8_files(tmp_path: Path) -> None:
    (tmp_path / "latin1.py").write_bytes(b"s = '\xe9'\n")
    [info] = scan_repo(tmp_path, 1000)
    assert info.lines == 1


def test_empty_repo(tmp_path: Path) -> None:
    assert scan_repo(tmp_path, 1000) == []


def test_cli_scan_prints_summary(repo: Path) -> None:
    result = CliRunner().invoke(app, ["scan", str(repo)])
    assert result.exit_code == 0, result.output
    assert "4 files, 10 lines" in result.output
    assert "hidden.py" not in result.output
