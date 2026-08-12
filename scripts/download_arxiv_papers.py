"""Download the version-pinned arXiv PDFs declared in data/sources."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_DIR / "data" / "sources" / "arxiv_papers.json"
RAW_DIR = PROJECT_DIR / "data" / "raw"
DOWNLOAD_MANIFEST_PATH = RAW_DIR / "download-manifest.json"
REQUIRED_FIELDS = {
    "source_id",
    "arxiv_id",
    "version",
    "title",
    "pdf_url",
    "filename",
}
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
USER_AGENT = "Evidentia/0.1 (local research RAG prototype)"


def load_catalog() -> list[dict[str, object]]:
    """Read and validate the version-pinned list of arXiv sources."""

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    papers = catalog.get("papers")

    if not isinstance(papers, list):
        raise ValueError("The source catalog must contain a 'papers' list.")

    for paper in papers:
        if not isinstance(paper, dict):
            raise ValueError("Each paper declaration must be a JSON object.")

        missing_fields = REQUIRED_FIELDS - paper.keys()
        if missing_fields:
            raise ValueError(f"A paper declaration is missing: {sorted(missing_fields)}")

    return papers


def download_pdf(url: str, destination: Path) -> None:
    """Download one PDF atomically so an interrupted file is never accepted."""

    temporary_path = destination.with_name(f".{destination.name}.part")
    request = Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urlopen(request) as response, temporary_path.open("wb") as output_file:
            while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                output_file.write(chunk)

        ensure_pdf(temporary_path)
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def ensure_pdf(path: Path) -> None:
    """Reject an HTML error page or another unexpected response type."""

    with path.open("rb") as input_file:
        header = input_file.read(4)

    if header != b"%PDF":
        raise ValueError(f"Downloaded file is not a PDF: {path.name}")


def sha256(path: Path) -> str:
    """Compute the content hash recorded in the local download manifest."""

    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        while chunk := input_file.read(DOWNLOAD_CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


def build_manifest(papers: list[dict[str, object]]) -> list[dict[str, str]]:
    """Download missing PDFs and return their provenance plus content hashes."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_papers: list[dict[str, str]] = []

    for paper in papers:
        filename = str(paper["filename"])
        destination = RAW_DIR / filename

        if not destination.exists():
            print(f"Downloading {paper['arxiv_id']}{paper['version']}...")
            download_pdf(str(paper["pdf_url"]), destination)
        else:
            print(f"Using existing file: {filename}")

        ensure_pdf(destination)
        downloaded_papers.append(
            {
                "source_id": str(paper["source_id"]),
                "arxiv_id": str(paper["arxiv_id"]),
                "version": str(paper["version"]),
                "title": str(paper["title"]),
                "filename": filename,
                "sha256": sha256(destination),
            }
        )

    return downloaded_papers


def write_download_manifest(papers: list[dict[str, str]]) -> None:
    """Record exactly which local PDF bytes are ready for ingestion."""

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "papers": papers,
    }
    DOWNLOAD_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Download all declared papers and create a local provenance manifest."""

    papers = load_catalog()
    downloaded_papers = build_manifest(papers)
    write_download_manifest(downloaded_papers)
    print(f"Ready for ingestion: {len(downloaded_papers)} PDF(s).")


if __name__ == "__main__":
    main()
