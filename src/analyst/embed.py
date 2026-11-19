"""Turn chunks and questions into embedding vectors with the Gemini API."""

import logging

from google.genai import types

from .chunk import Chunk
from .llm import make_client

logger = logging.getLogger(__name__)

BATCH_SIZE = 100  # chunks per request; check the model's per-request limit

Vector = list[float]


def embed_chunks(chunks: list[Chunk], *, api_key: str, model: str) -> list[Vector]:
    """One vector per chunk, in the same order as `chunks`.

    Embeds with task_type RETRIEVAL_DOCUMENT, BATCH_SIZE chunks per request.
    """
    client = make_client(api_key)
    config = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")

    vectors: list[Vector] = []
    for batch in batched(chunks, BATCH_SIZE):           
        contents = [types.Content(parts=[types.Part(text=chunk_to_text(c))]) for c in batch]  # noqa: E501
        response = client.models.embed_content(model=model, 
                                               contents=contents, 
                                               config=config)  # noqa: E501
        if response.embeddings is None or len(response.embeddings) != len(batch):
            raise RuntimeError(f"Expected {len(batch)} embeddings from {model}, got {response.embeddings and len(response.embeddings)}")  # noqa: E501
        for e in response.embeddings:
            if e.values is None:
                raise RuntimeError(f"{model} returned an embedding with no values")
            vectors.append(e.values)         
    return vectors


def embed_query(question: str, *, api_key: str, model: str) -> Vector:
    """Vector for a user question, with task_type RETRIEVAL_QUERY."""
    client = make_client(api_key)
    config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    response = client.models.embed_content(model=model, 
                                           contents=[types.Content(parts=[types.Part(text=question)])], 
                                           config=config)  
    if response.embeddings is None or len(response.embeddings) != 1:
        raise RuntimeError("no embedding returned")
    values = response.embeddings[0].values
    if values is None:
        raise RuntimeError(f"{model} returned an embedding with no values")
    return values


def chunk_to_text(chunk: Chunk) -> str:
    """Text sent for embedding: a '# path · name' header line, then the chunk's code."""
    return f"# {chunk.path.as_posix()} · {chunk.name}\n{chunk.text}"


def batched[T](items: list[T], size: int) -> list[list[T]]:
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    return [items[i:i + size] for i in range(0, len(items), size)]
