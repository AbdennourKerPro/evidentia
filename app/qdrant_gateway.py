"""Small boundary between the HTTP API and the Qdrant client library."""

from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from app.settings import get_qdrant_url
from app.schemas import EvidenceChunk, IndexedDocument, SearchResult


DEMO_COLLECTION_NAME = "evidence_chunks_e5_small"
ARXIV_COLLECTION_NAME = "arxiv_chunks_e5_small"


def _new_client() -> QdrantClient:
    """Create a short-lived HTTP client for one Qdrant operation."""

    return QdrantClient(url=get_qdrant_url(), timeout=10)


def qdrant_is_available() -> bool:
    """Check that Qdrant accepts a simple request within two seconds."""

    client: QdrantClient | None = None

    try:
        client = QdrantClient(url=get_qdrant_url(), timeout=2)
        client.get_collections()
    except Exception:
        # Network and protocol failures both mean that the API is not ready.
        return False
    finally:
        if client is not None:
            client.close()

    return True


def index_chunks(
    chunks: Sequence[EvidenceChunk],
    vectors: Sequence[list[float]],
    *,
    collection_name: str,
) -> int:
    """Create one named collection if needed, then upsert its chunk vectors."""

    if len(chunks) != len(vectors):
        raise ValueError("Each evidence chunk must have exactly one vector.")
    if not chunks:
        return 0

    client = _new_client()

    try:
        _ensure_collection(
            client,
            collection_name=collection_name,
            vector_size=len(vectors[0]),
        )
        points = [
            models.PointStruct(
                id=chunk.id,
                vector=vector,
                payload={
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "page": chunk.page,
                    "section": chunk.section,
                    "language": chunk.language,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        client.upsert(collection_name=collection_name, points=points, wait=True)
    finally:
        client.close()

    return len(chunks)


def search_chunks(
    query_vector: list[float],
    *,
    limit: int,
    collection_name: str,
    document_ids: Sequence[str] | None = None,
) -> list[SearchResult]:
    """Return the nearest chunks, optionally restricted to selected articles."""

    client = _new_client()

    try:
        if not client.collection_exists(collection_name):
            return []

        hits = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=_document_filter(document_ids),
        ).points
    finally:
        client.close()

    return [
        SearchResult(
            chunk_id=int(hit.id),
            score=hit.score,
            document_id=str(hit.payload["document_id"]),
            title=str(hit.payload["title"]),
            page=int(hit.payload["page"]),
            section=str(hit.payload["section"]),
            language=str(hit.payload["language"]),
            text=str(hit.payload["text"]),
        )
        for hit in hits
    ]


def list_indexed_documents(collection_name: str) -> list[IndexedDocument]:
    """Read unique article metadata and chunk counts from one collection."""

    client = _new_client()

    try:
        if not client.collection_exists(collection_name):
            return []

        documents: dict[str, IndexedDocument] = {}
        next_page_offset = None

        while True:
            points, next_page_offset = client.scroll(
                collection_name=collection_name,
                offset=next_page_offset,
                limit=512,
                with_payload=["document_id", "title"],
                with_vectors=False,
            )

            for point in points:
                document_id = str(point.payload["document_id"])
                title = str(point.payload["title"])
                existing = documents.get(document_id)

                if existing is None:
                    documents[document_id] = IndexedDocument(
                        document_id=document_id,
                        title=title,
                        indexed_chunks=1,
                    )
                else:
                    existing.indexed_chunks += 1

            if next_page_offset is None:
                break
    finally:
        client.close()

    return sorted(documents.values(), key=lambda document: document.title)


def _document_filter(document_ids: Sequence[str] | None) -> models.Filter | None:
    """Build one Qdrant keyword filter only when a scope was explicitly chosen."""

    if document_ids is None:
        return None

    return models.Filter(
        must=[
            models.FieldCondition(
                key="document_id",
                match=models.MatchAny(any=list(document_ids)),
            )
        ]
    )


def _ensure_collection(
    client: QdrantClient, *, collection_name: str, vector_size: int
) -> None:
    """Create a cosine-similarity collection for this specific embedding model."""

    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )
