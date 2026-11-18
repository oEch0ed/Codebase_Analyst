"""Build, save and load a repo's index: every chunk with its embedding vector."""

import logging
from pathlib import Path

from pydantic import BaseModel

from .chunk import Chunk
from .embed import Vector, embed_chunks

logger = logging.getLogger(__name__)

INDEX_DIR = ".analyst"  
INDEX_FILE = "index.json"


class IndexEntry(BaseModel):
    chunk: Chunk
    vector: Vector


class Index(BaseModel):
    embedding_model: str  # embedding model that made the vectors
    dimensions: int  # length of every vector
    entries: list[IndexEntry]

class WrongModelError(Exception):
    """Index was saved with a different embedding model"""

def index_path(root: Path) -> Path:
    """Where the index for the repo at `root` lives: <root>/.analyst/index.json.
    so both save_index and load_index agree on the same path/file"""
    return root / INDEX_DIR / INDEX_FILE


def build_index(chunks: list[Chunk], *, api_key: str, model: str) -> Index:
    """Embed every chunk and pair each one with its vector."""
    embeddings = embed_chunks(chunks,api_key=api_key,model=model)
    entries = [IndexEntry(chunk=chunks[i],vector=embeddings[i]) for i in range(len(embeddings))]
    dimensions = len(embeddings[0]) if embeddings else 0
    return Index(embedding_model=model,dimensions=dimensions,entries=entries)


def save_index(root: Path, index: Index) -> Path:
    """Write `index` to index_path(root), creating .analyst/ if needed."""
    path = index_path(root)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(index.model_dump_json(),encoding="utf-8")
    return path
    

def load_index(root: Path, *, model: str) -> Index:
    """"raises FileNotFoundError, WrongModelError."""
    index = Index.model_validate_json(index_path(root).read_text(encoding="utf-8"))
    if index.embedding_model != model:
        raise WrongModelError(
            f"Index was built with {index.embedding_model}, but the current model is {model}; re-index"
        )
    return index
    
