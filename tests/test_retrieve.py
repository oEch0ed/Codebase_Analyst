from pathlib import Path

import pytest

from analyst import retrieve as retrieve_module
from analyst.chunk import Chunk
from analyst.config import StaleError
from analyst.embed import Vector
from analyst.index import Index, IndexEntry, save_index
from analyst.retrieve import retrieve, topk


def make_entry(name: str, vector: list[float]) -> IndexEntry:
    chunk = Chunk(
        path=Path("a.py"),
        kind="function",
        name=name,
        start_line=1,
        end_line=2,
        text=f"def {name}(): ...",
    )
    return IndexEntry(chunk=chunk, vector=vector)


def make_index() -> Index:
    return Index(
        embedding_model="test-model",
        dimensions=2,
        entries=[
            make_entry("opposite", [-1.0, 0.0]),
            make_entry("diagonal", [10.0, 10.0]),  # long, but 45° off
            make_entry("same", [2.0, 0.0]),  # same direction as the query
            make_entry("orthogonal", [0.0, 1.0]),
        ],
    )


# ---------- topk ----------


def test_topk_returns_indices_most_similar_first() -> None:
    assert topk(make_index(), [1.0, 0.0], k=2) == [2, 1]


def test_topk_k_larger_than_index_returns_everything() -> None:
    assert topk(make_index(), [1.0, 0.0], k=10) == [2, 1, 3, 0]


def test_topk_empty_index_returns_nothing() -> None:
    empty = Index(embedding_model="test-model", dimensions=0, entries=[])
    assert topk(empty, [1.0, 0.0], k=2) == []


@pytest.mark.parametrize("k", [0, -1])
def test_topk_k_below_one_raises(k: int) -> None:
    with pytest.raises(ValueError):
        topk(make_index(), [1.0, 0.0], k=k)


def test_topk_wrong_dimensions_raises_stale_error() -> None:
    with pytest.raises(StaleError):
        topk(make_index(), [1.0, 0.0, 0.0], k=2)


# ---------- retrieve, with a fake embed_query ----------


class FakeEmbedQuery:
    """Stands in for embed_query: always returns [1, 0] and records each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []  # (question, api_key, model)

    def __call__(self, question: str, *, api_key: str, model: str) -> Vector:
        self.calls.append((question, api_key, model))
        return [1.0, 0.0]


@pytest.fixture
def fake_embed_query(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedQuery:
    """Replace embed_query inside retrieve.py so no test touches the network."""
    fake = FakeEmbedQuery()
    monkeypatch.setattr(retrieve_module, "embed_query", fake)
    return fake


def test_retrieve_returns_best_chunks(
    tmp_path: Path, fake_embed_query: FakeEmbedQuery
) -> None:
    save_index(tmp_path, make_index())

    result = retrieve("how?", tmp_path, api_key="test-key", 
                      embedding_model="test-model", k=2)

    assert [c.name for c in result] == ["same", "diagonal"]
    assert fake_embed_query.calls == [("how?", "test-key", "test-model")]


def test_retrieve_wrong_model_raises_stale_error(
    tmp_path: Path, fake_embed_query: FakeEmbedQuery
) -> None:
    save_index(tmp_path, make_index())

    with pytest.raises(StaleError):
        retrieve("how?", tmp_path, api_key="test-key", embedding_model="other-model",
                  k=2)
    assert fake_embed_query.calls == []  # fails before spending an API call
