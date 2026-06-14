"""Embedding pipeline using sentence-transformers (all-MiniLM-L6-v2).

Produces 384-dimensional vectors from text or structured attributes.
Used by relay-api for pgvector cosine matching on product units and wishes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_EMBEDDING_DIM = 384
_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingUnavailable(Exception):
    """Raised when the embedding model cannot be loaded."""


@lru_cache(maxsize=1)
def _load_model():
    """Load the sentence-transformer model once and cache it."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(f"sentence-transformers/{_MODEL_NAME}")
        return model
    except Exception as exc:
        raise EmbeddingUnavailable(
            f"Failed to load embedding model '{_MODEL_NAME}': {exc}"
        ) from exc


def _build_text(
    text: str | None = None,
    category: str | None = None,
    grade: str | None = None,
    size: str | None = None,
    vertical: str | None = None,
) -> str:
    """Build a text string from structured attributes or raw text.

    If `text` is provided, it's used directly.
    Otherwise, structured fields are joined into a descriptive sentence.
    """
    if text:
        return text

    parts: list[str] = []
    if category:
        parts.append(f"category: {category}")
    if grade:
        parts.append(f"grade: {grade}")
    if size:
        parts.append(f"size: {size}")
    if vertical:
        parts.append(f"vertical: {vertical}")

    if not parts:
        return "unknown item"

    return ", ".join(parts)


def compute_embedding(
    text: str | None = None,
    category: str | None = None,
    grade: str | None = None,
    size: str | None = None,
    vertical: str | None = None,
) -> tuple[list[float], str]:
    """Compute a 384-d embedding vector.

    Returns:
        Tuple of (vector as list of floats, model name string).
    """
    model = _load_model()
    input_text = _build_text(
        text=text, category=category, grade=grade, size=size, vertical=vertical
    )
    embedding = model.encode(input_text, normalize_embeddings=True)
    vector = embedding.tolist()

    # Ensure correct dimension
    assert len(vector) == _EMBEDDING_DIM, (
        f"Expected {_EMBEDDING_DIM}-d vector, got {len(vector)}"
    )

    return vector, _MODEL_NAME
