"""Embed passages and queries with the same multilingual retrieval model."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-small"
EMBEDDING_MODEL_REVISION = "fd1525a9fd15316a2d503bf26ab031a61d056e98"


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    """Load the model once per API process, on its first embedding request."""

    return SentenceTransformer(
        EMBEDDING_MODEL_ID,
        revision=EMBEDDING_MODEL_REVISION,
    )


def embed_passages(passages: list[str]) -> list[list[float]]:
    """Embed chunks with the document prefix required by E5 retrieval training."""

    return _encode([f"passage: {passage}" for passage in passages])


def embed_query(query: str) -> list[float]:
    """Embed a user question with the query prefix required by E5."""

    return _encode([f"query: {query}"])[0]


def _encode(texts: list[str]) -> list[list[float]]:
    """Normalize vectors so cosine similarity compares their directions only."""

    vectors = get_embedding_model().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()
