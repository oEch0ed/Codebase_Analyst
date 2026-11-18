from pathlib import Path

import pytest

from analyst import index
from analyst.chunk import Chunk
from analyst.embed import Vector
from analyst.index import (
    INDEX_DIR,
    INDEX_FILE,
    Index,
    IndexEntry,
    WrongModelError,
    build_index,
    index_path,
    load_index,
    save_index,
)


def make_chunk(name: str) -> Chunk:
    return Chunk(
        path=Path(f"{name}.py"),
        kind="function",
        name=name,
        start_line=1,
        end_line=2,
        text=f"def {name}(): ...",
    )


class FakeEmbed:
    """Stands in for embed_chunks: chunk i gets the vector [i, i, i]."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[Chunk], str, str]] = []  # (chunks, api_key, model)

    def __call__(
        self, chunks: list[Chunk], *, api_key: str, model: str
    ) -> list[Vector]:
        self.calls.append((chunks, api_key, model))
        return [[float(i)] * 3 for i in range(len(chunks))]


@pytest.fixture
def fake_embed(monkeypatch: pytest.MonkeyPatch) -> FakeEmbed:
    """Replace embed_chunks inside index.py so no test touches the network."""
    fake = FakeEmbed()
    monkeypatch.setattr(index, "embed_chunks", fake)
    return fake


def test_build_index_pairs_each_chunk_with_its_vector(fake_embed: FakeEmbed) -> None:
    chunks = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
    idx = build_index(chunks, api_key="k", model="m")

    assert [e.chunk for e in idx.entries] == chunks
    assert [e.vector for e in idx.entries] == [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ]


def test_build_index_records_model_and_dimensions(fake_embed: FakeEmbed) -> None:
    idx = build_index([make_chunk("a"), make_chunk("b")], api_key="k", model="m")

    assert idx.embedding_model == "m"
    assert idx.dimensions == 3  # length of each vector, not the number of entries


def test_build_index_passes_key_and_model_to_embed(fake_embed: FakeEmbed) -> None:
    chunks = [make_chunk("a")]
    build_index(chunks, api_key="secret", model="gemini-embedding-2")

    assert fake_embed.calls == [(chunks, "secret", "gemini-embedding-2")]


def test_build_index_no_chunks(fake_embed: FakeEmbed) -> None:
    """A repo with no Python code still gets a valid (empty) index."""
    idx = build_index([], api_key="k", model="m")

    assert idx.entries == []
    assert idx.dimensions == 0
    assert idx.embedding_model == "m"


# ---------- index_path / save_index / load_index ----------


def sample_index() -> Index:
    return Index(
        embedding_model="m",
        dimensions=2,
        entries=[
            IndexEntry(chunk=make_chunk("a"), vector=[0.1, 0.2]),
            IndexEntry(chunk=make_chunk("b"), vector=[0.3, 0.4]),
        ],
    )


def test_index_path(tmp_path: Path) -> None:
    assert index_path(tmp_path) == tmp_path / ".analyst" / "index.json"


def test_save_index_creates_dir_and_file(tmp_path: Path) -> None:
    path = save_index(tmp_path, sample_index())

    assert path == tmp_path / INDEX_DIR / INDEX_FILE
    assert path.is_file()


def test_save_index_overwrites_existing(tmp_path: Path) -> None:
    """Re-indexing replaces the old index rather than failing or appending."""
    save_index(tmp_path, sample_index())
    newer = Index(embedding_model="m", dimensions=0, entries=[])
    save_index(tmp_path, newer)

    assert load_index(tmp_path, model="m") == newer


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    """What you load is exactly what you saved, Paths and vectors included."""
    original = sample_index()
    save_index(tmp_path, original)
    loaded = load_index(tmp_path, model="m")

    assert loaded == original
    assert isinstance(loaded.entries[0].chunk.path, Path)


def test_load_index_not_indexed_yet(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path, model="m")


def test_load_index_wrong_model(tmp_path: Path) -> None:
    save_index(tmp_path, sample_index())  # built with "m"
    with pytest.raises(WrongModelError, match="re-index"):
        load_index(tmp_path, model="other-model")
