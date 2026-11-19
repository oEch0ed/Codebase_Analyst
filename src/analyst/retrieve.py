"""Place Holder for now"""
# i want to define a user query : {topk:list[str], query:str, model:str}
# topk - text of the relevant chunks
# query - the user's question
# model - the embedding model for the user question, later check for wrong model error
# i need to retrieve the top-k cosine simillar items 

from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .chunk import Chunk
from .config import StaleError
from .embed import Vector, embed_query
from .index import Index, load_index


def retrieve(query:str, root:Path,
              embedding_model: str, api_key:str, k:int) -> list[Chunk]:
    """ embed user query, load index, get neighbors, win battles"""
    index = load_index(root,model=embedding_model)
    query_embedding = embed_query(query,api_key=api_key,model=embedding_model)
    indices = topk(index,query_embedding,k)
    return [index.entries[i].chunk for i in indices]

def topk(index:Index, query:Vector, k:int) -> list[int]:
    """return the top k neighbors of the user's query"""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if not index.entries:
        return []
    if len(query) != index.dimensions:
        raise StaleError(
            f"Query has {len(query)} dimensions but the index has "
            f"{index.dimensions}; re-index"
        )
    k = min(k,len(index.entries))
    matrix = [entry.vector for entry in index.entries]
    cosines = cosine_similarity(matrix,[query])[:, 0]
    top = np.argpartition(-cosines,k - 1)[:k]
    return top[np.argsort(-cosines[top])].tolist()

def build_prompt(question: str, chunks: list[Chunk]) -> str:
    """Question plus the retrieved code, each chunk headed by its location."""
    context = "\n\n".join(
        f"### {c.path.as_posix()}:{c.start_line}-{c.end_line} ({c.kind} {c.name})\n```python\n{c.text}\n```"  # noqa: E501
        for c in chunks
    )
    return (
        "You are answering questions about a Python codebase.\n"
        "Use only the code below. Cite file paths and line numbers. "
        "If the code shown isn't enough to answer, say so.\n\n"
        f"{context}\n\n"
        f"Question: {question}"
    )

    


