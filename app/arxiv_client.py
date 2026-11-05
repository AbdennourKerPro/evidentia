"""Minimal, polite client for the public arXiv API and PDF endpoint."""

from __future__ import annotations

import hashlib
import re
import threading
import time
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Callable, Literal

import httpx
from pydantic import BaseModel, Field


API_URL = "https://export.arxiv.org/api/query"
PDF_URL = "https://arxiv.org/pdf/{arxiv_id}{version}"
USER_AGENT = "Evidentia/0.1 (local research RAG prototype)"
MIN_SECONDS_BETWEEN_API_CALLS = 3.0
REQUEST_TIMEOUT_SECONDS = 20.0
SEARCH_CACHE_TTL_SECONDS = 3600
ABSTRACT_LIMIT = 600
ARXIV_ID_PATTERN = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
CATEGORY_PATTERN = re.compile(r"^[a-z\-]+(\.[A-Za-z\-]{2,})?$")
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"


class ArxivError(RuntimeError):
    """A network, format or size problem; the message is safe to show a model."""


class ArxivPaperMeta(BaseModel):
    """Metadata of one arXiv paper, as returned to the tool-calling agent."""

    arxiv_id: str
    latest_version: str
    title: str
    authors: list[str] = Field(default_factory=list)
    published: str = ""
    primary_category: str = ""
    abstract: str = ""


def split_arxiv_id(value: str) -> tuple[str, str | None]:
    """Split ``2304.02643v4`` into (``2304.02643``, ``v4``); version may be None."""

    match = ARXIV_ID_PATTERN.match(value.strip())
    if match is None:
        raise ArxivError(f"Not a valid modern arXiv identifier: {value!r}")
    return match.group(1), match.group(2)


# ------------------------------------------------------------------ rate limit

_api_lock = threading.Lock()
_last_api_call = 0.0
_search_cache: dict[tuple, tuple[float, list[ArxivPaperMeta]]] = {}


def _wait_for_turn() -> None:
    """arXiv asks automated clients to leave about three seconds between calls."""

    global _last_api_call
    delay = MIN_SECONDS_BETWEEN_API_CALLS - (time.monotonic() - _last_api_call)
    if delay > 0:
        time.sleep(delay)
    _last_api_call = time.monotonic()


def _http_client() -> httpx.Client:
    """Overridable in tests with an in-memory transport."""

    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
        follow_redirects=True,
    )


def _query(params: dict[str, str]) -> list[ArxivPaperMeta]:
    with _api_lock:
        _wait_for_turn()
        try:
            with _http_client() as client:
                response = client.get(API_URL, params=params)
        except httpx.HTTPError as error:
            raise ArxivError(f"arXiv is unreachable: {type(error).__name__}") from None
    if response.status_code != 200:
        raise ArxivError(f"arXiv answered HTTP {response.status_code}.")
    return parse_atom_feed(response.text)


def parse_atom_feed(xml_text: str) -> list[ArxivPaperMeta]:
    """Parse an arXiv Atom feed; entries without a usable id are skipped."""

    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        raise ArxivError("arXiv returned an unreadable response.") from None
    papers: list[ArxivPaperMeta] = []
    for entry in root.findall(f"{_ATOM}entry"):
        identifier = (entry.findtext(f"{_ATOM}id") or "").strip()
        match = re.search(r"/abs/(\d{4}\.\d{4,5})(v\d+)?$", identifier)
        if match is None:
            continue  # error entries and legacy ids
        category = entry.find(f"{_ARXIV}primary_category")
        abstract = " ".join((entry.findtext(f"{_ATOM}summary") or "").split())
        papers.append(
            ArxivPaperMeta(
                arxiv_id=match.group(1),
                latest_version=match.group(2) or "v1",
                title=" ".join((entry.findtext(f"{_ATOM}title") or "").split()),
                authors=[
                    " ".join((author.findtext(f"{_ATOM}name") or "").split())
                    for author in entry.findall(f"{_ATOM}author")
                ][:6],
                published=(entry.findtext(f"{_ATOM}published") or "")[:10],
                primary_category=category.get("term", "") if category is not None else "",
                abstract=abstract[:ABSTRACT_LIMIT],
            )
        )
    return papers


def search(
    query: str,
    *,
    max_results: int = 5,
    category: str | None = None,
    sort_by: Literal["relevance", "submittedDate"] = "relevance",
) -> list[ArxivPaperMeta]:
    """Full-text search; results are cached for an hour."""

    query = " ".join(query.split())
    if not 3 <= len(query) <= 200:
        raise ArxivError("The query must contain between 3 and 200 characters.")
    if category is not None and not CATEGORY_PATTERN.match(category):
        raise ArxivError("Invalid arXiv category.")
    if sort_by not in ("relevance", "submittedDate"):
        raise ArxivError("sort_by must be 'relevance' or 'submittedDate'.")
    max_results = max(1, min(int(max_results), 8))
    key = (query.lower(), max_results, category, sort_by)
    cached = _search_cache.get(key)
    if cached and time.monotonic() - cached[0] < SEARCH_CACHE_TTL_SECONDS:
        return list(cached[1])
    tokens = _search_tokens(query)
    if not tokens:
        raise ArxivError("The query contains no searchable word.")
    base = {"start": "0", "max_results": str(max_results), "sortBy": sort_by, "sortOrder": "descending"}
    suffix = f" AND cat:{category}" if category else ""
    # An exact-phrase search misses almost every natural query: require every word
    # first, then fall back to any word (arXiv ranks by how many match).
    results = _query({**base, "search_query": " AND ".join(f"all:{t}" for t in tokens) + suffix})
    if not results and len(tokens) > 1:
        results = _query({**base, "search_query": "(" + " OR ".join(f"all:{t}" for t in tokens) + ")" + suffix})
    _search_cache[key] = (time.monotonic(), results)
    return list(results)


_STOP_WORDS = frozenset("a an and are as at by for from in into is of on or the to with via using".split())


def _search_tokens(query: str) -> list[str]:
    """Distinct alphanumeric words, without filler words, capped to keep URLs short."""

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", query)
    kept = [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 1]
    return list(dict.fromkeys(kept))[:12]


def lookup(arxiv_ids: list[str]) -> list[ArxivPaperMeta]:
    """Exact metadata for known identifiers (versions are ignored for the lookup)."""

    bare = [split_arxiv_id(value)[0] for value in arxiv_ids][:5]
    if not bare:
        return []
    return _query({"id_list": ",".join(bare), "max_results": str(len(bare))})


# -------------------------------------------------------------------- download


def download_pdf(
    arxiv_id: str,
    version: str,
    destination: Path,
    *,
    max_megabytes: int,
    on_bytes: Callable[[int], None] | None = None,
) -> str:
    """Download one PDF atomically, enforce the size cap, return its SHA-256."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    limit = max_megabytes * 1024 * 1024
    digest = hashlib.sha256()
    received = 0
    try:
        with _http_client() as client, client.stream(
            "GET", PDF_URL.format(arxiv_id=arxiv_id, version=version)
        ) as response:
            if response.status_code != 200:
                raise ArxivError(f"arXiv answered HTTP {response.status_code} for the PDF.")
            declared = int(response.headers.get("content-length") or 0)
            if declared > limit:
                raise ArxivError(f"The PDF is larger than {max_megabytes} MB.")
            with temporary.open("wb") as output:
                for chunk in response.iter_bytes(1024 * 1024):
                    received += len(chunk)
                    if received > limit:
                        raise ArxivError(f"The PDF is larger than {max_megabytes} MB.")
                    digest.update(chunk)
                    output.write(chunk)
                    if on_bytes:
                        on_bytes(received)
        with temporary.open("rb") as check:
            if check.read(4) != b"%PDF":
                raise ArxivError("The downloaded file is not a PDF.")
        temporary.replace(destination)
    except httpx.HTTPError as error:
        raise ArxivError(f"PDF download failed: {type(error).__name__}") from None
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()
