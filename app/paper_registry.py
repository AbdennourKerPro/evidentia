"""SQLite registry: which papers exist, and which conversation may use them.

The registry is the source of truth for isolation. Qdrant holds every chunk in one
collection; a conversation only ever searches the document ids listed here.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, Field

from app.settings import get_state_db_path


PaperStatus = Literal["pending", "downloading", "converting", "indexing", "ready", "failed"]
ATTACHMENT_SOURCES = ("agent", "user", "seed")
CONVERSATION_ID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    document_id TEXT PRIMARY KEY,
    arxiv_id TEXT NOT NULL,
    version TEXT NOT NULL,
    title TEXT NOT NULL,
    short_name TEXT,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    pdf_path TEXT,
    processed_dir TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN
        ('pending','downloading','converting','indexing','ready','failed')),
    error TEXT,
    pinned INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS papers_arxiv_version ON papers (arxiv_id, version);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL,
    owner TEXT
);
CREATE TABLE IF NOT EXISTS conversation_papers (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES papers(document_id) ON DELETE CASCADE,
    added_at TEXT NOT NULL,
    added_by TEXT NOT NULL CHECK (added_by IN ('agent','user','seed')),
    reason TEXT,
    PRIMARY KEY (conversation_id, document_id)
);
CREATE TABLE IF NOT EXISTS invites (
    id TEXT PRIMARY KEY,
    code_hash TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    quota INTEGER NOT NULL,
    used INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL,
    manual_papers_used INTEGER NOT NULL DEFAULT 0,
    max_manual_papers INTEGER NOT NULL,
    expires_at TEXT,
    revoked INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    last_used_at TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    invite_id TEXT NOT NULL REFERENCES invites(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invite_id TEXT,
    conversation_id TEXT NOT NULL,
    rating TEXT NOT NULL CHECK (rating IN ('up','down')),
    comment TEXT,
    question TEXT,
    answer TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversation_turns (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    question TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (conversation_id, position)
);
"""


class PaperRecord(BaseModel):
    """One paper known to the registry, ready or not."""

    document_id: str
    arxiv_id: str
    version: str
    title: str
    short_name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    pdf_path: str | None = None
    processed_dir: str | None = None
    chunk_count: int = 0
    status: PaperStatus
    error: str | None = None
    pinned: bool = False


class ConversationPaper(BaseModel):
    """A paper attached to one conversation, as shown in the interface."""

    document_id: str
    arxiv_id: str
    version: str
    title: str
    short_name: str | None = None
    chunk_count: int
    status: PaperStatus
    added_by: Literal["agent", "user", "seed"]
    reason: str | None = None
    pinned: bool = False


def is_valid_conversation_id(conversation_id: str) -> bool:
    """Conversation ids are browser-generated UUIDs; reject anything else."""

    return bool(CONVERSATION_ID_PATTERN.match(conversation_id))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """One short-lived connection per operation; WAL allows concurrent readers."""

    path = get_state_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=15)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    """Create the tables if needed. Safe to call repeatedly."""

    with _connect() as connection:
        connection.executescript(SCHEMA)
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(conversations)")}
        if "owner" not in columns:  # database created before invitations existed
            connection.execute("ALTER TABLE conversations ADD COLUMN owner TEXT")


def _paper_from_row(row: sqlite3.Row) -> PaperRecord:
    return PaperRecord(
        document_id=row["document_id"],
        arxiv_id=row["arxiv_id"],
        version=row["version"],
        title=row["title"],
        short_name=row["short_name"],
        aliases=json.loads(row["aliases_json"]),
        pdf_path=row["pdf_path"],
        processed_dir=row["processed_dir"],
        chunk_count=row["chunk_count"],
        status=row["status"],
        error=row["error"],
        pinned=bool(row["pinned"]),
    )


# --------------------------------------------------------------------- papers


def upsert_paper(
    *,
    document_id: str,
    arxiv_id: str,
    version: str,
    title: str,
    short_name: str | None = None,
    aliases: list[str] | None = None,
    status: PaperStatus = "pending",
    pinned: bool = False,
    chunk_count: int = 0,
    pdf_path: str | None = None,
    processed_dir: str | None = None,
) -> PaperRecord:
    """Insert a paper or refresh its descriptive fields without losing its state."""

    now = _now()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO papers (document_id, arxiv_id, version, title, short_name,
                aliases_json, pdf_path, processed_dir, chunk_count, status, pinned,
                created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                title = excluded.title,
                short_name = COALESCE(excluded.short_name, papers.short_name),
                aliases_json = CASE WHEN excluded.aliases_json = '[]'
                    THEN papers.aliases_json ELSE excluded.aliases_json END,
                pinned = MAX(papers.pinned, excluded.pinned),
                updated_at = excluded.updated_at
            """,
            (
                document_id, arxiv_id, version, title, short_name,
                json.dumps(aliases or []), pdf_path, processed_dir, chunk_count,
                status, int(pinned), now, now,
            ),
        )
    paper = get_paper(document_id)
    assert paper is not None
    return paper


def set_paper_status(
    document_id: str,
    status: PaperStatus,
    *,
    error: str | None = None,
    chunk_count: int | None = None,
    pdf_path: str | None = None,
    processed_dir: str | None = None,
    aliases: list[str] | None = None,
) -> None:
    """Record an ingestion transition; unspecified columns keep their value."""

    with _connect() as connection:
        connection.execute(
            """
            UPDATE papers SET status = ?, error = ?, updated_at = ?,
                chunk_count = COALESCE(?, chunk_count),
                pdf_path = COALESCE(?, pdf_path),
                processed_dir = COALESCE(?, processed_dir),
                aliases_json = COALESCE(?, aliases_json)
            WHERE document_id = ?
            """,
            (
                status, error, _now(), chunk_count, pdf_path, processed_dir,
                json.dumps(aliases) if aliases is not None else None, document_id,
            ),
        )


def get_paper(document_id: str) -> PaperRecord | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM papers WHERE document_id = ?", (document_id,)
        ).fetchone()
    return _paper_from_row(row) if row else None


def find_paper(arxiv_id: str, version: str | None = None) -> PaperRecord | None:
    """Find a paper by arXiv id; without a version prefer a ready one, then the newest."""

    with _connect() as connection:
        if version:
            row = connection.execute(
                "SELECT * FROM papers WHERE arxiv_id = ? AND version = ?",
                (arxiv_id, version),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT * FROM papers WHERE arxiv_id = ?
                ORDER BY (status = 'ready') DESC, pinned DESC, version DESC LIMIT 1
                """,
                (arxiv_id,),
            ).fetchone()
    return _paper_from_row(row) if row else None


def list_pinned_papers() -> list[PaperRecord]:
    """The historical corpus used by the classic pipeline and the benchmark."""

    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM papers WHERE pinned = 1 AND status = 'ready' ORDER BY document_id"
        ).fetchall()
    return [_paper_from_row(row) for row in rows]


def delete_paper_row(document_id: str) -> None:
    with _connect() as connection:
        connection.execute("DELETE FROM papers WHERE document_id = ?", (document_id,))


# -------------------------------------------------------------- conversations


def ensure_conversation(conversation_id: str) -> None:
    """Create a conversation or refresh its activity time (idempotent)."""

    now = _now()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO conversations (id, created_at, last_active_at) VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET last_active_at = excluded.last_active_at
            """,
            (conversation_id, now, now),
        )


def claim_conversation(conversation_id: str, owner: str) -> bool:
    """Create the conversation for ``owner`` or check that ``owner`` already owns it.

    A conversation without an owner (created before invitations existed) is never
    claimable: its UUID would otherwise be enough to take it over.
    """

    now = _now()
    with _connect() as connection:
        connection.execute(
            "INSERT OR IGNORE INTO conversations (id, created_at, last_active_at, owner) VALUES (?, ?, ?, ?)",
            (conversation_id, now, now, owner),
        )
        row = connection.execute(
            "SELECT owner FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row["owner"] == owner:
            connection.execute(
                "UPDATE conversations SET last_active_at = ? WHERE id = ?", (now, conversation_id)
            )
            return True
    return False


def count_papers_created_since(since_iso: str) -> int:
    """Papers added by conversations (not the pinned corpus) since a given instant."""

    with _connect() as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM papers WHERE pinned = 0 AND created_at >= ?", (since_iso,)
        ).fetchone()[0]


def attach_paper(
    conversation_id: str,
    document_id: str,
    *,
    added_by: Literal["agent", "user", "seed"],
    reason: str | None = None,
) -> bool:
    """Attach a paper to a conversation. Returns False if it was already attached."""

    if added_by not in ATTACHMENT_SOURCES:
        raise ValueError(f"Unknown attachment source: {added_by}")
    ensure_conversation(conversation_id)
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO conversation_papers
                (conversation_id, document_id, added_at, added_by, reason)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, document_id, _now(), added_by, reason),
        )
        return cursor.rowcount == 1


def detach_paper(conversation_id: str, document_id: str) -> bool:
    with _connect() as connection:
        cursor = connection.execute(
            "DELETE FROM conversation_papers WHERE conversation_id = ? AND document_id = ?",
            (conversation_id, document_id),
        )
        return cursor.rowcount == 1


def list_conversation_papers(
    conversation_id: str, *, only_ready: bool = False
) -> list[ConversationPaper]:
    query = """
        SELECT p.*, cp.added_by, cp.reason FROM conversation_papers cp
        JOIN papers p ON p.document_id = cp.document_id
        WHERE cp.conversation_id = ?
    """
    if only_ready:
        query += " AND p.status = 'ready'"
    query += " ORDER BY cp.added_at, p.document_id"
    with _connect() as connection:
        rows = connection.execute(query, (conversation_id,)).fetchall()
    return [
        ConversationPaper(
            document_id=row["document_id"],
            arxiv_id=row["arxiv_id"],
            version=row["version"],
            title=row["title"],
            short_name=row["short_name"],
            chunk_count=row["chunk_count"],
            status=row["status"],
            added_by=row["added_by"],
            reason=row["reason"],
            pinned=bool(row["pinned"]),
        )
        for row in rows
    ]


def conversation_exists(conversation_id: str) -> bool:
    with _connect() as connection:
        return connection.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone() is not None


def record_turn(conversation_id: str, question: str) -> None:
    """Keep the questions asked so far: the agent's minimal conversational memory."""

    ensure_conversation(conversation_id)
    with _connect() as connection:
        position = connection.execute(
            "SELECT COALESCE(MAX(position), 0) + 1 FROM conversation_turns WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO conversation_turns (conversation_id, position, question, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, position, question, _now()),
        )


def recent_questions(conversation_id: str, count: int = 3) -> list[str]:
    """Most recent questions, oldest first."""

    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT question FROM conversation_turns WHERE conversation_id = ?
            ORDER BY position DESC LIMIT ?
            """,
            (conversation_id, count),
        ).fetchall()
    return [row["question"] for row in reversed(rows)]


def delete_conversation(conversation_id: str) -> None:
    """Delete a conversation, its turns and its attachments (papers stay cached)."""

    with _connect() as connection:
        connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def orphan_documents(min_age_seconds: int = 0) -> list[str]:
    """Unpinned, settled papers that no conversation uses any more.

    ``min_age_seconds`` protects a paper that was just ingested and is about to be
    attached to a conversation.
    """

    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=min_age_seconds)).isoformat()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT document_id FROM papers
            WHERE pinned = 0 AND status IN ('ready', 'failed')
              AND document_id NOT IN (SELECT document_id FROM conversation_papers)
              AND updated_at <= ?
            """,
            (cutoff,),
        ).fetchall()
    return [row["document_id"] for row in rows]


def stale_conversations(older_than_days: int) -> list[str]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
    with _connect() as connection:
        rows = connection.execute(
            "SELECT id FROM conversations WHERE last_active_at < ?", (cutoff,)
        ).fetchall()
    return [row["id"] for row in rows]


# ----------------------------------------------------------------- legacy seed


def seed_legacy_papers(
    *, catalog: dict[str, dict], sources_path: Path
) -> list[PaperRecord]:
    """Register the five historical papers as pinned, ready, with their exact aliases.

    The aliases are the regular expressions the classic pipeline always used, so
    question routing is identical for them.
    """

    declared = json.loads(sources_path.read_text(encoding="utf-8")).get("papers", [])
    seeded: list[PaperRecord] = []
    for paper in declared:
        document_id = str(paper["source_id"])
        metadata = catalog.get(document_id)
        if metadata is None:
            continue
        record = upsert_paper(
            document_id=document_id,
            arxiv_id=str(paper["arxiv_id"]),
            version=str(paper["version"]),
            title=str(metadata["title"]),
            aliases=list(metadata["aliases"]),
            status="ready",
            pinned=True,
        )
        # The chunk count is informational; keep whatever indexing recorded.
        set_paper_status(document_id, "ready")
        seeded.append(record)
    return seeded
