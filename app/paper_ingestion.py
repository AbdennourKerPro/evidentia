"""Download, convert, chunk, embed and index one arXiv paper on demand."""

from __future__ import annotations

import json
import re
import shutil
import threading
from pathlib import Path
from typing import Callable

from app import arxiv_client, document_ingestion, paper_registry
from app.embeddings import embed_passages
from app.hybrid_retrieval import clear_bm25_cache
from app.qdrant_gateway import (
    ARXIV_COLLECTION_NAME,
    delete_document_chunks,
    index_chunks,
)
from app.settings import get_arxiv_cache_dir, get_dynamic_limits
from app.text_normalization import normalize_text


ProgressCallback = Callable[[str, dict], None]
MAX_TITLE_ALIAS_CHARS = 30
MIN_ALIAS_CHARS = 3

_locks_guard = threading.Lock()
_document_locks: dict[str, threading.Lock] = {}
# Docling is memory hungry: never run two conversions at once.
_conversion_lock = threading.Lock()


class IngestionError(RuntimeError):
    """A paper could not be ingested; the message is safe to show a model."""


def document_id_for(arxiv_id: str, version: str) -> str:
    return f"arxiv-{arxiv_id}{version}"


def _is_stale(paper: paper_registry.PaperRecord) -> bool:
    """A cached dynamic paper converted before formula decoding existed must be redone."""

    if paper.pinned:
        return False
    manifest = (
        Path(paper.processed_dir) if paper.processed_dir
        else get_arxiv_cache_dir() / "processed" / paper.document_id
    ) / "conversion-manifest.json"
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("formula_enrichment") is not True
    except (OSError, ValueError):
        return True


def _lock_for(document_id: str) -> threading.Lock:
    with _locks_guard:
        return _document_locks.setdefault(document_id, threading.Lock())


def build_aliases(title: str, short_name: str | None) -> list[str]:
    """Regex aliases used to detect the paper's name in a question.

    Aliases apply to the accent-free lowercase question (see ``normalize_text``).
    Very short aliases are refused: they would match ordinary words.
    """

    candidates: list[str] = []
    if short_name:
        candidates.append(short_name)
    prefix = title.split(":", 1)[0].strip() if ":" in title else ""
    if prefix and len(prefix) <= MAX_TITLE_ALIAS_CHARS:
        candidates.append(prefix)
    aliases: list[str] = []
    for candidate in candidates:
        normalized = normalize_text(candidate)
        if len(normalized) < MIN_ALIAS_CHARS:
            continue
        pattern = r"\b" + re.escape(normalized).replace(r"\ ", " ") + r"\b"
        if pattern not in aliases:
            aliases.append(pattern)
    return aliases


def ingest_arxiv_paper(
    arxiv_id: str,
    version: str | None = None,
    *,
    short_name: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> paper_registry.PaperRecord:
    """Return a ready paper, ingesting it only if it is not already cached.

    Raises ``IngestionError``; on failure the registry row is marked ``failed`` and
    partial artifacts are removed, so a later retry starts clean.
    """

    notify = on_progress or (lambda stage, payload: None)
    limits = get_dynamic_limits()

    cached = paper_registry.find_paper(arxiv_id, version)
    if cached is not None and cached.status == "ready" and not _is_stale(cached):
        return cached

    try:
        metadata = arxiv_client.lookup([arxiv_id])
    except arxiv_client.ArxivError as error:
        raise IngestionError(str(error)) from None
    if not metadata:
        raise IngestionError(f"arXiv has no paper with identifier {arxiv_id}.")
    paper = metadata[0]
    resolved_version = version or paper.latest_version
    document_id = document_id_for(paper.arxiv_id, resolved_version)

    with _lock_for(document_id):
        existing = paper_registry.get_paper(document_id)
        if existing is not None and existing.status == "ready" and not _is_stale(existing):
            return existing  # another request finished while we waited

        cache_dir = get_arxiv_cache_dir()
        pdf_path = cache_dir / "raw" / f"{document_id}.pdf"
        processed_dir = cache_dir / "processed" / document_id
        aliases = build_aliases(paper.title, short_name)
        paper_registry.upsert_paper(
            document_id=document_id,
            arxiv_id=paper.arxiv_id,
            version=resolved_version,
            title=paper.title,
            short_name=short_name,
            aliases=aliases,
            status="pending",
        )
        try:
            notify("downloading", {"document_id": document_id, "title": paper.title})
            paper_registry.set_paper_status(document_id, "downloading")
            sha256 = arxiv_client.download_pdf(
                paper.arxiv_id,
                resolved_version,
                pdf_path,
                max_megabytes=limits.max_pdf_megabytes,
            )

            notify("converting", {"document_id": document_id})
            paper_registry.set_paper_status(document_id, "converting", pdf_path=str(pdf_path))
            with _conversion_lock:
                docling_document = document_ingestion.convert_pdf_to_artifacts(
                    pdf_path=pdf_path,
                    output_dir=processed_dir,
                    source={
                        "source_id": document_id,
                        "arxiv_id": paper.arxiv_id,
                        "version": resolved_version,
                        "title": paper.title,
                        "sha256": sha256,
                    },
                    max_pages=limits.max_pages,
                )
            chunks = document_ingestion.chunk_document(
                document_id=document_id,
                title=paper.title,
                document=docling_document,
                output_dir=processed_dir,
            )

            notify("indexing", {"document_id": document_id, "chunks": len(chunks)})
            paper_registry.set_paper_status(
                document_id, "indexing", processed_dir=str(processed_dir)
            )
            vectors = embed_passages([chunk.text for chunk in chunks])
            # A re-conversion may produce a different number of chunks: start clean.
            delete_document_chunks(ARXIV_COLLECTION_NAME, document_id)
            index_chunks(chunks, vectors, collection_name=ARXIV_COLLECTION_NAME)
            clear_bm25_cache()
            paper_registry.set_paper_status(document_id, "ready", chunk_count=len(chunks))
        except Exception as error:
            _discard_artifacts(document_id, pdf_path, processed_dir)
            message = (
                str(error)
                if isinstance(error, (arxiv_client.ArxivError, IngestionError))
                else f"Conversion failed ({type(error).__name__})."
            )
            paper_registry.set_paper_status(document_id, "failed", error=message)
            raise IngestionError(message) from None

    ready = paper_registry.get_paper(document_id)
    assert ready is not None
    notify("ready", {"document_id": document_id, "chunks": ready.chunk_count})
    return ready


def _discard_artifacts(document_id: str, pdf_path: Path, processed_dir: Path) -> None:
    pdf_path.unlink(missing_ok=True)
    shutil.rmtree(processed_dir, ignore_errors=True)
    try:
        delete_document_chunks(ARXIV_COLLECTION_NAME, document_id)
        clear_bm25_cache()
    except Exception:
        pass  # Qdrant may be the very thing that failed; the registry still records it


def purge_document(document_id: str) -> bool:
    """Delete a paper everywhere. Pinned and in-progress papers are never purged."""

    paper = paper_registry.get_paper(document_id)
    if paper is None or paper.pinned or paper.status not in ("ready", "failed"):
        return False
    with _lock_for(document_id):
        delete_document_chunks(ARXIV_COLLECTION_NAME, document_id)
        clear_bm25_cache()
        cache_dir = get_arxiv_cache_dir()
        (cache_dir / "raw" / f"{document_id}.pdf").unlink(missing_ok=True)
        shutil.rmtree(cache_dir / "processed" / document_id, ignore_errors=True)
        paper_registry.delete_paper_row(document_id)
    return True


def collect_garbage(min_age_seconds: int = 600) -> list[str]:
    """Purge every unpinned paper that no conversation uses; return their ids."""

    return [
        document_id
        for document_id in paper_registry.orphan_documents(min_age_seconds)
        if purge_document(document_id)
    ]
