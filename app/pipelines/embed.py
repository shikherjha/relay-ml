"""Embedding pipeline.

Supports two backends:
- sentence-transformers/all-MiniLM-L6-v2 (local, requires torch)
- Amazon Bedrock Titan Embeddings (cloud, no torch needed)

Set EMBEDDING_MODEL env var:
- "all-MiniLM-L6-v2" for local (default)
- "bedrock-titan" for AWS Bedrock Titan Embeddings V2
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_EMBEDDING_DIM = 384
_MODEL_NAME_LOCAL = "all-MiniLM-L6-v2"
_MODEL_NAME_TITAN = "amazon.titan-embed-text-v2:0"


class EmbeddingUnavailable(Exception):
    """Raised when the embedding model cannot be loaded."""


@lru_cache(maxsize=1)
def _load_local_model():
    """Load the sentence-transformer model once and cache it."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(f"sentence-transformers/{_MODEL_NAME_LOCAL}")
        return model
    except Exception as exc:
        raise EmbeddingUnavailable(
            f"Failed to load embedding model '{_MODEL_NAME_LOCAL}': {exc}"
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


def _embed_with_titan(input_text: str) -> tuple[list[float], str]:
    """Compute embedding using Bedrock Titan Embeddings V2.
    
    Titan V2 supports dimensions: 1024, 512, 256.
    We request 512 and truncate to 384 for pgvector compatibility.
    Titan V2 uses Matryoshka representation learning so truncation preserves quality.
    """
    try:
        import json
        import boto3
        from app.core.config import settings

        client = boto3.client("bedrock-runtime", region_name=settings.aws_region)

        response = client.invoke_model(
            modelId=_MODEL_NAME_TITAN,
            body=json.dumps({
                "inputText": input_text,
                "dimensions": 512,
                "normalize": True,
            }),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())
        vector_512 = result["embedding"]

        # Truncate to 384 and re-normalize
        vector_384 = vector_512[:384]
        # L2 normalize after truncation
        norm = sum(x * x for x in vector_384) ** 0.5
        if norm > 0:
            vector_384 = [x / norm for x in vector_384]

        return vector_384, "bedrock-titan-v2"
    except Exception as exc:
        raise EmbeddingUnavailable(f"Bedrock Titan embedding failed: {exc}") from exc


def _embed_with_local(input_text: str) -> tuple[list[float], str]:
    """Compute embedding using local sentence-transformers."""
    model = _load_local_model()
    embedding = model.encode(input_text, normalize_embeddings=True)
    vector = embedding.tolist()
    assert len(vector) == _EMBEDDING_DIM, (
        f"Expected {_EMBEDDING_DIM}-d vector, got {len(vector)}"
    )
    return vector, _MODEL_NAME_LOCAL


def compute_embedding(
    text: str | None = None,
    category: str | None = None,
    grade: str | None = None,
    size: str | None = None,
    vertical: str | None = None,
) -> tuple[list[float], str]:
    """Compute a 384-d embedding vector.

    Uses Bedrock Titan if EMBEDDING_MODEL=bedrock-titan, else local model.

    Returns:
        Tuple of (vector as list of floats, model name string).
    """
    from app.core.config import settings

    input_text = _build_text(
        text=text, category=category, grade=grade, size=size, vertical=vertical
    )

    if settings.embedding_model == "bedrock-titan":
        return _embed_with_titan(input_text)

    return _embed_with_local(input_text)
