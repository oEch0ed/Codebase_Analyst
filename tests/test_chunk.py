import ast
import logging
from pathlib import Path

import pytest

from analyst import chunk
from analyst.chunk import chunk_repo, chunk_source
from analyst.scan import FileInfo

# Covers every case the chunker has to handle; line numbers are on the left
# of the printed output so you can check start/end lines by eye.
SAMPLE = '''\
"""Module docstring."""

import os
from pathlib import Path

LIMIT = 10


@decorator
def decorated(a: int) -> int:
    return a + 1


async def fetch(url: str) -> str:
    def helper() -> None:  # nested: should stay inside fetch
        pass

    return url


class Retriever:
    """Finds relevant chunks."""

    default_k = 5

    def __init__(self, index: list[str], top_k: int = default_k) -> None:
        self.index = index
        self.top_k = top_k

    def search(
        self,
        question: str,
    ) -> list[str]:
        return self.index[: self.top_k]

    @staticmethod
    def _rerank(items: list[str]) -> list[str]:
        return sorted(items)


print(os.getcwd(), Path.cwd())
'''


# Line numbers:
#  1 @dataclass
#  2 class Greeter:
#  3     """Says hello."""
#  4
#  5     greeting = "hi"
#  6
#  7     def hello(self, name: str) -> str:
#  8         return f"{self.greeting} {name}"
#  9
# 10     @staticmethod
# 11     def shout(text: str) -> str:
# 12         return text.upper()
GREETER = '''\
@dataclass
class Greeter:
    """Says hello."""

    greeting = "hi"

    def hello(self, name: str) -> str:
        return f"{self.greeting} {name}"

    @staticmethod
    def shout(text: str) -> str:
        return text.upper()
'''


def test_class_chunks() -> None:
    lines = GREETER.splitlines()
    node = ast.parse(GREETER).body[0]
    chunks = chunk._class_chunks(node, lines, Path("greeter.py"))

    assert [c.kind for c in chunks] == ["class", "method", "method"]
    assert [c.name for c in chunks] == ["Greeter", "Greeter.hello", "Greeter.shout"]

    # header: from the decorator up to (not including) the first method
    header = chunks[0]
    assert (header.start_line, header.end_line) == (1, 6)
    assert header.text == "\n".join(lines[0:6])
    assert header.text.startswith("@dataclass\nclass Greeter:")
    assert "def hello" not in header.text

    hello = chunks[1]
    assert (hello.start_line, hello.end_line) == (7, 8)
    assert hello.text.splitlines()[0] == "    def hello(self, name: str) -> str:"

    # decorator counts as the method's first line
    shout = chunks[2]
    assert (shout.start_line, shout.end_line) == (10, 12)
    assert shout.text.splitlines()[0] == "    @staticmethod"
    assert shout.text.splitlines()[-1] == "        return text.upper()"
    assert all(c.path == Path("greeter.py") for c in chunks)


# Line numbers:
#  1 @dataclass
#  2 class Point:
#  3     """A 2D point."""
#  4
#  5     x: int
#  6     y: int
NO_METHODS = '''\
@dataclass
class Point:
    """A 2D point."""

    x: int
    y: int
'''


def test_class_without_methods() -> None:
    """With no methods, the header chunk is the whole class, last line included."""
    lines = NO_METHODS.splitlines()
    node = ast.parse(NO_METHODS).body[0]
    chunks = chunk._class_chunks(node, lines, Path("point.py"))

    assert len(chunks) == 1
    point = chunks[0]
    assert point.kind == "class"
    assert point.name == "Point"
    assert (point.start_line, point.end_line) == (1, 6)
    assert point.text == NO_METHODS.rstrip("\n")


# Line numbers:
#  1 import os
#  2
#  3
#  4 @first
#  5 @second(arg=1)
#  6 def decorated(
#  7     a: int,
#  8     b: int = 2,
#  9 ) -> int:
# 10     def helper() -> int:  # nested: stays inside decorated
# 11         return a
# 12
# 13     return helper() + b
# 14
# 15
# 16 async def fetch(url: str) -> str:
# 17     return url
FUNCS = """\
import os


@first
@second(arg=1)
def decorated(
    a: int,
    b: int = 2,
) -> int:
    def helper() -> int:  # nested: stays inside decorated
        return a

    return helper() + b


async def fetch(url: str) -> str:
    return url
"""


def test_def_chunks() -> None:
    lines = FUNCS.splitlines()
    tree = ast.parse(FUNCS)
    path = Path("funcs.py")

    # both decorators count, multi-line signature and nested def are included
    decorated = chunk._def_chunks(tree.body[1], lines, path)
    assert decorated.kind == "function"
    assert decorated.name == "decorated"
    assert (decorated.start_line, decorated.end_line) == (4, 13)
    assert decorated.text.splitlines()[0] == "@first"
    assert decorated.text.splitlines()[-1] == "    return helper() + b"
    assert "def helper" in decorated.text
    assert "import os" not in decorated.text
    assert decorated.path == path

    # async def works the same way
    fetch = chunk._def_chunks(tree.body[2], lines, path)
    assert fetch.name == "fetch"
    assert (fetch.start_line, fetch.end_line) == (16, 17)
    assert fetch.text == "async def fetch(url: str) -> str:\n    return url"


def test_module_chunk() -> None:
    """Everything that isn't a def/class, in order, as one '<module>' chunk."""
    lines = SAMPLE.splitlines()
    module = chunk._module_chunk(ast.parse(SAMPLE), lines, Path("sample.py"))

    assert module is not None
    assert module.kind == "module"
    assert module.name == "<module>"
    # docstring on line 1 through the print() call on the last line
    assert (module.start_line, module.end_line) == (1, len(lines))
    assert module.text.splitlines() == [
        '"""Module docstring."""',
        "import os",
        "from pathlib import Path",
        "LIMIT = 10",
        "print(os.getcwd(), Path.cwd())",
    ]


def test_module_chunk_empty() -> None:
    """A file with only defs/classes has no module chunk."""
    lines = GREETER.splitlines()
    assert chunk._module_chunk(ast.parse(GREETER), lines, Path("greeter.py")) is None


def test_chunk_source_covers_sample() -> None:
    """Every top-level def/class gets chunks, plus one module chunk at the end."""
    chunks = chunk_source(SAMPLE, Path("sample.py"))
    assert [(c.kind, c.name) for c in chunks] == [
        ("function", "decorated"),
        ("function", "fetch"),
        ("class", "Retriever"),
        ("method", "Retriever.__init__"),
        ("method", "Retriever.search"),
        ("method", "Retriever._rerank"),
        ("module", "<module>"),
    ]


def test_chunk_source_line_ranges() -> None:
    """Line ranges for SAMPLE (see print_numbered(SAMPLE) to check by eye)."""
    chunks = chunk_source(SAMPLE, Path("sample.py"))
    assert {c.name: (c.start_line, c.end_line) for c in chunks} == {
        "decorated": (9, 11),  # starts at @decorator
        "fetch": (14, 18),  # nested helper stays inside
        "Retriever": (21, 25),  # header stops before __init__
        "Retriever.__init__": (26, 28),
        "Retriever.search": (30, 34),  # multi-line signature
        "Retriever._rerank": (36, 38),  # starts at @staticmethod
        "<module>": (1, 41),  # docstring through the final print()
    }


def test_chunk_source_text_matches_lines() -> None:
    """Each def/class chunk's text is exactly its line range; path is passed through."""
    lines = SAMPLE.splitlines()
    path = Path("pkg/sample.py")
    for c in chunk_source(SAMPLE, path):
        assert c.path == path
        if c.kind != "module":
            assert c.text == "\n".join(lines[c.start_line - 1 : c.end_line]), c.name


def test_chunk_source_nested_def_not_chunked() -> None:
    chunks = chunk_source(SAMPLE, Path("sample.py"))
    assert all("helper" not in c.name for c in chunks)


def test_chunk_source_empty() -> None:
    assert chunk_source("", Path("empty.py")) == []


def test_chunk_source_only_code() -> None:
    """A script with no defs/classes is a single module chunk."""
    source = "import sys\n\nprint(sys.argv)\n"
    chunks = chunk_source(source, Path("script.py"))
    assert [(c.kind, c.start_line, c.end_line) for c in chunks] == [("module", 1, 3)]
    assert chunks[0].text == "import sys\nprint(sys.argv)"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Two good files (one nested) and one that doesn't parse."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "main.py").write_text("import sys\n\n\ndef main() -> None:\n    pass\n")
    (tmp_path / "pkg" / "util.py").write_text("def helper() -> int:\n    return 1\n")
    (tmp_path / "broken.py").write_text("def f(:\n    pass\n")
    return tmp_path


def info(path: str) -> FileInfo:
    return FileInfo(path=Path(path), lines=0)  # lines isn't used by the chunker


def test_chunk_repo_good_files(repo: Path) -> None:
    """Chunks come back flat, in the order of `files`, with repo-relative paths."""
    chunks, skipped = chunk_repo(repo, [info("main.py"), info("pkg/util.py")])

    assert skipped == {}
    assert [(c.path, c.kind, c.name) for c in chunks] == [
        (Path("main.py"), "function", "main"),
        (Path("main.py"), "module", "<module>"),
        (Path("pkg/util.py"), "function", "helper"),
    ]


def test_chunk_repo_skips_unparsable_file(repo: Path) -> None:
    """A broken file is reported in `skipped` and doesn't stop the others."""
    files = [info("main.py"), info("broken.py"), info("pkg/util.py")]
    chunks, skipped = chunk_repo(repo, files)

    assert skipped == {Path("broken.py"): "line 1: invalid syntax"}
    assert {c.path for c in chunks} == {Path("main.py"), Path("pkg/util.py")}


def test_chunk_repo_logs_skipped_file(
    repo: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="analyst.chunk"):
        chunk_repo(repo, [info("broken.py")])

    assert "broken.py" in caplog.text
    assert "invalid syntax" in caplog.text


def test_chunk_repo_no_files(repo: Path) -> None:
    assert chunk_repo(repo, []) == ([], {})


def test_chunk_source_raises_on_invalid_python() -> None:
    """chunk_source lets parse errors through; chunk_repo is what catches them."""
    with pytest.raises(SyntaxError):
        chunk_source("def f(:\n    pass\n", Path("bad.py"))


def print_numbered(text: str) -> None:
    for number, line in enumerate(text.splitlines(), start=1):
        print(f"{number:>3} | {line}")
