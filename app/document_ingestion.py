"""Convert declared local PDFs into inspectable Docling artifacts."""

from __future__ import annotations

import json
from functools import lru_cache
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument
from transformers import AutoTokenizer

from app.embeddings import EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION
from app.schemas import ConversionReport, EvidenceChunk


RAW_DIR = Path("/data/raw")
PROCESSED_DIR = Path("/data/processed")
DOWNLOAD_MANIFEST_PATH = RAW_DIR / "download-manifest.json"
CHUNK_TOKEN_LIMIT = 350


class UnknownSourceError(ValueError):
    """Raised when a requested source is absent from the download manifest."""


def convert_source(source_id: str) -> ConversionReport:
    """Convert one declared PDF and persist Markdown plus lossless Docling JSON."""

    source = _find_source(source_id)
    source_pdf = RAW_DIR / source["filename"]

    if not source_pdf.is_file():
        raise FileNotFoundError(f"Downloaded PDF is missing: {source_pdf.name}")

    document = DocumentConverter().convert(source_pdf).document
    output_dir = PROCESSED_DIR / source_id
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / "document.md"
    docling_json_path = output_dir / "document.json"
    conversion_manifest_path = output_dir / "conversion-manifest.json"

    document.save_as_markdown(markdown_path)
    document.save_as_json(docling_json_path)
    _write_conversion_manifest(
        conversion_manifest_path=conversion_manifest_path,
        source=source,
        markdown_path=markdown_path,
        docling_json_path=docling_json_path,
    )

    return ConversionReport(
        source_id=source_id,
        source_pdf=f"data/raw/{source_pdf.name}",
        markdown_file=f"data/processed/{source_id}/{markdown_path.name}",
        docling_json_file=f"data/processed/{source_id}/{docling_json_path.name}",
    )


def chunk_source(source_id: str) -> tuple[list[EvidenceChunk], Path]:
    """Turn one converted Docling document into page-aware RAG passages.

    The generated JSON is deliberately saved before embedding. It makes the
    otherwise invisible retrieval input inspectable without looking at Qdrant.
    """

    source = _find_source(source_id)
    document_path = PROCESSED_DIR / source_id / "document.json"

    if not document_path.is_file():
        raise FileNotFoundError(
            f"Converted Docling JSON is missing: data/processed/{source_id}/document.json"
        )

    document = DoclingDocument.load_from_json(document_path)
    chunks = _build_evidence_chunks(source_id=source_id, source=source, document=document)

    chunks_path = document_path.with_name("chunks.json")
    chunks_path.write_text(
        json.dumps(
            [chunk.model_dump() for chunk in chunks],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return chunks, chunks_path


def _find_source(source_id: str) -> dict[str, str]:
    """Find a downloaded source by its stable source identifier."""

    manifest = json.loads(DOWNLOAD_MANIFEST_PATH.read_text(encoding="utf-8"))
    papers = manifest.get("papers", [])

    for paper in papers:
        if paper.get("source_id") == source_id:
            return {key: str(value) for key, value in paper.items()}

    raise UnknownSourceError(f"Unknown source: {source_id}")


@lru_cache
def _get_chunker() -> HybridChunker:
    """Load the E5 tokenizer once and make chunks fit its input budget."""

    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL_ID,
        revision=EMBEDDING_MODEL_REVISION,
    )
    return HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=tokenizer,
            max_tokens=CHUNK_TOKEN_LIMIT,
        )
    )


def _build_evidence_chunks(
    *, source_id: str, source: dict[str, str], document: DoclingDocument
) -> list[EvidenceChunk]:
    """Keep Docling's structural context and map it to the RAG schema."""

    evidence_chunks: list[EvidenceChunk] = []
    chunker = _get_chunker()

    for position, docling_chunk in enumerate(chunker.chunk(dl_doc=document), start=1):
        contextual_text = chunker.contextualize(docling_chunk).strip()
        if not contextual_text:
            continue

        evidence_chunks.append(
            EvidenceChunk(
                id=_stable_chunk_id(source_id, position),
                document_id=source_id,
                title=source["title"],
                page=_page_number(docling_chunk),
                section=_section_name(docling_chunk),
                language="en",
                text=contextual_text,
            )
        )

    if not evidence_chunks:
        raise ValueError(f"No text chunks were produced for source: {source_id}")

    return evidence_chunks


def _stable_chunk_id(source_id: str, position: int) -> int:
    """Create a reproducible positive Qdrant point identifier for one chunk."""

    digest = sha256(f"{source_id}:{position}".encode("utf-8")).digest()
    return int.from_bytes(digest[:7], byteorder="big")


def _page_number(docling_chunk: Any) -> int:
    """Read the first known source page from Docling provenance metadata."""

    pages = [
        provenance.page_no
        for item in getattr(docling_chunk.meta, "doc_items", [])
        for provenance in (getattr(item, "prov", []) or [])
        if getattr(provenance, "page_no", None) is not None
    ]
    return min(pages) if pages else 1


def _section_name(docling_chunk: Any) -> str:
    """Represent the hierarchy retained by Docling as a readable path."""

    headings = getattr(docling_chunk.meta, "headings", []) or []
    section = " > ".join(heading.strip() for heading in headings if heading.strip())
    return section or "Document"


def _write_conversion_manifest(
    conversion_manifest_path: Path,
    source: dict[str, str],
    markdown_path: Path,
    docling_json_path: Path,
) -> None:
    """Record the exact source hash and artifacts produced by this conversion."""

    manifest = {
        "converted_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "artifacts": {
            "markdown": markdown_path.name,
            "docling_json": docling_json_path.name,
        },
    }
    conversion_manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
