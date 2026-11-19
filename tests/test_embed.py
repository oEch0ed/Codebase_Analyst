from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import types

from analyst import embed
from analyst.chunk import Chunk
from analyst.embed import batched, chunk_to_text, embed_chunks, embed_query


def make_chunk(path: str, kind: str, name: str, text: str) -> Chunk:
    return Chunk(
        path=Path(path), kind=kind, name=name, start_line=1, end_line=2, text=text
    )


def test_chunk_to_text_adds_header() -> None:
    c = make_chunk(
        "pkg/util.py", "method", "Retriever.search", "def search(self):\n    pass"
    )
    assert (
        chunk_to_text(c)
        == "# pkg/util.py · Retriever.search\ndef search(self):\n    pass"
    )


def test_chunk_to_text_module_chunk() -> None:
    c = make_chunk("main.py", "module", "<module>", "import sys")
    assert chunk_to_text(c) == "# main.py · <module>\nimport sys"


# ---------- embed_chunks / embed_query, with a fake Gemini client ----------


class FakeModels:
    """Stands in for client.models: records each request, returns numbered vectors.

    The n-th text ever sent gets the vector [n], so tests can check order.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str | None]] = []  # (texts, task_type)
        self.drop = 0  # return this many fewer embeddings than texts sent
        self.counter = 0

    def embed_content(
        self,
        *,
        model: str,
        contents: list[types.Content],
        config: types.EmbedContentConfig,
    ) -> types.EmbedContentResponse:
        texts = [c.parts[0].text for c in contents]
        self.calls.append((texts, config.task_type))
        embeddings = []
        for _ in texts:
            embeddings.append(types.ContentEmbedding(values=[float(self.counter)]))
            self.counter += 1
        return types.EmbedContentResponse(
            embeddings=embeddings[: len(embeddings) - self.drop]
        )


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeModels:
    """Replace make_client so no test touches the network."""
    models = FakeModels()
    monkeypatch.setattr(
        embed, "make_client", lambda api_key: SimpleNamespace(models=models)
    )
    return models


def chunks(n: int) -> list[Chunk]:
    return [
        make_chunk(f"f{i}.py", "function", f"f{i}", f"def f{i}(): ...")
        for i in range(n)
    ]


def test_embed_chunks_one_vector_per_chunk_in_order(fake: FakeModels) -> None:
    vectors = embed_chunks(chunks(3), api_key="k", model="m")
    assert vectors == [[0.0], [1.0], [2.0]]


def test_embed_chunks_sends_header_text_as_documents(fake: FakeModels) -> None:
    embed_chunks(chunks(2), api_key="k", model="m")
    texts, task_type = fake.calls[0]
    assert texts == ["# f0.py · f0\ndef f0(): ...", "# f1.py · f1\ndef f1(): ..."]
    assert task_type == "RETRIEVAL_DOCUMENT"


def test_embed_chunks_batches_requests(
    fake: FakeModels, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(embed, "BATCH_SIZE", 2)
    vectors = embed_chunks(chunks(5), api_key="k", model="m")

    assert [len(texts) for texts, _ in fake.calls] == [2, 2, 1]
    assert vectors == [[0.0], [1.0], [2.0], [3.0], [4.0]]  # still in chunk order


def test_embed_chunks_empty_makes_no_request(fake: FakeModels) -> None:
    assert embed_chunks([], api_key="k", model="m") == []
    assert fake.calls == []


def test_embed_chunks_raises_on_missing_embeddings(fake: FakeModels) -> None:
    """Fewer vectors than chunks would misalign every vector after the gap."""
    fake.drop = 1
    with pytest.raises(RuntimeError, match="embedding"):
        embed_chunks(chunks(3), api_key="k", model="m")


def test_embed_query_returns_one_vector(fake: FakeModels) -> None:
    assert embed_query("where is search?", api_key="k", model="m") == [0.0]


def test_embed_query_sends_question_as_query(fake: FakeModels) -> None:
    embed_query("where is search?", api_key="k", model="m")
    assert fake.calls == [(["where is search?"], "RETRIEVAL_QUERY")]


def test_embed_query_raises_on_missing_embedding(fake: FakeModels) -> None:
    fake.drop = 1
    with pytest.raises(RuntimeError, match="embedding"):
        embed_query("where is search?", api_key="k", model="m")


# ---------- batched ----------


def test_batched_even_split() -> None:
    assert batched([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_batched_last_batch_shorter() -> None:
    assert batched([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_batched_size_bigger_than_list() -> None:
    assert batched([1, 2, 3], 10) == [[1, 2, 3]]


def test_batched_size_one() -> None:
    assert batched(["a", "b", "c"], 1) == [["a"], ["b"], ["c"]]


def test_batched_empty() -> None:
    assert batched([], 3) == []


def test_batched_returns_lists() -> None:
    """Lists, not tuples (itertools.batched gives tuples)."""
    result = batched([1, 2, 3], 2)
    assert result == [[1, 2], [3]]
    assert all(type(b) is list for b in result)


def test_batched_rejects_size_below_one() -> None:
    with pytest.raises(ValueError):
        batched([1, 2, 3], 0)
