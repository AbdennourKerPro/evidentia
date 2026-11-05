"""Invitation-based access for test users: codes, sessions, quotas, limits, feedback.

Disabled by default (``ACCESS_MODE=open``, the local behaviour). With
``ACCESS_MODE=invite`` every conversation route needs a session obtained by
redeeming an invitation code. Codes and session tokens are stored hashed, so a
copy of the database does not give access.

A question is *reserved* before it runs (this also bounds parallel requests) and
*settled* afterwards: only an answered question keeps its quota unit, an abstention
or a server failure gives it back. ``attempts`` is never refunded, so that a tester
who only triggers abstentions cannot cost an unbounded amount.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Callable, Literal

from pydantic import BaseModel

from app.paper_registry import _connect, _now


DEFAULT_QUOTA = 20
ATTEMPTS_FACTOR = 3  # default attempt cap = quota x 3
DEFAULT_MANUAL_PAPERS = 3
SESSION_COOKIE = "evidentia_session"
ReservationError = Literal["revoked", "expired", "quota_exhausted", "attempts_exhausted"]


def access_mode() -> str:
    """``invite`` or ``open``; read at call time so tests and restarts can switch it."""

    return "invite" if os.getenv("ACCESS_MODE", "open").strip().lower() == "invite" else "open"


def trust_proxy_headers() -> bool:
    """Believe ``CF-Connecting-IP`` / ``X-Forwarded-For`` (only behind a tunnel you control)."""

    return os.getenv("TRUST_PROXY_HEADERS", "").strip().lower() in {"1", "true", "yes"}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class Invite(BaseModel):
    """One invitation and its consumption counters."""

    id: str
    label: str
    quota: int
    used: int
    attempts: int
    max_attempts: int
    manual_papers_used: int
    max_manual_papers: int
    expires_at: str | None = None
    revoked: bool = False
    created_at: str
    last_used_at: str | None = None

    @property
    def remaining(self) -> int:
        return max(self.quota - self.used, 0)

    @property
    def expired(self) -> bool:
        return bool(self.expires_at and self.expires_at <= _now())

    def public(self) -> dict:
        return {
            "label": self.label,
            "quota": self.quota,
            "used": self.used,
            "remaining": self.remaining,
            "expires_at": self.expires_at,
            "max_manual_papers": self.max_manual_papers,
            "manual_papers_used": self.manual_papers_used,
        }


def _invite(row) -> Invite:
    return Invite(**{key: row[key] for key in Invite.model_fields} | {"revoked": bool(row["revoked"])})


# ---------------------------------------------------------------- administration


def create_invite(
    label: str,
    *,
    quota: int = DEFAULT_QUOTA,
    max_attempts: int | None = None,
    max_manual_papers: int = DEFAULT_MANUAL_PAPERS,
    days: int | None = None,
) -> tuple[Invite, str]:
    """Create an invitation; the clear code is returned once and never stored."""

    if quota < 1:
        raise ValueError("quota must be at least 1")
    code = secrets.token_urlsafe(16)  # 128 bits
    invite_id = secrets.token_hex(4)
    expires_at = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat() if days else None
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO invites (id, code_hash, label, quota, max_attempts, max_manual_papers,
                                 expires_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (invite_id, _digest(code), label.strip()[:80] or "sans nom", quota,
             max_attempts or quota * ATTEMPTS_FACTOR, max_manual_papers, expires_at, _now()),
        )
    invite = get_invite(invite_id)
    assert invite is not None
    return invite, code


def get_invite(invite_id: str) -> Invite | None:
    with _connect() as connection:
        row = connection.execute("SELECT * FROM invites WHERE id = ?", (invite_id,)).fetchone()
    return _invite(row) if row else None


def list_invites() -> list[Invite]:
    with _connect() as connection:
        rows = connection.execute("SELECT * FROM invites ORDER BY created_at").fetchall()
    return [_invite(row) for row in rows]


def revoke_invite(invite_id: str) -> bool:
    """Revoke an invitation and end its sessions."""

    with _connect() as connection:
        changed = connection.execute("UPDATE invites SET revoked = 1 WHERE id = ?", (invite_id,)).rowcount == 1
        connection.execute("DELETE FROM sessions WHERE invite_id = ?", (invite_id,))
    return changed


def add_quota(invite_id: str, extra: int) -> bool:
    with _connect() as connection:
        return connection.execute(
            "UPDATE invites SET quota = quota + ?, max_attempts = max_attempts + ? WHERE id = ?",
            (extra, extra * ATTEMPTS_FACTOR, invite_id),
        ).rowcount == 1


# ----------------------------------------------------------------------- sessions


def redeem(code: str) -> tuple[Invite, str] | None:
    """Exchange a valid invitation code for a session token; None if unusable."""

    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM invites WHERE code_hash = ?", (_digest(code.strip()),)
        ).fetchone()
        if row is None:
            return None
        invite = _invite(row)
        if invite.revoked or invite.expired:
            return None
        token = secrets.token_urlsafe(32)
        connection.execute(
            "INSERT INTO sessions (token_hash, invite_id, created_at) VALUES (?, ?, ?)",
            (_digest(token), invite.id, _now()),
        )
        connection.execute("UPDATE invites SET last_used_at = ? WHERE id = ?", (_now(), invite.id))
    return invite, token


def invite_for_session(token: str) -> Invite | None:
    """The valid invitation behind a session cookie, or None."""

    with _connect() as connection:
        row = connection.execute(
            "SELECT i.* FROM sessions s JOIN invites i ON i.id = s.invite_id WHERE s.token_hash = ?",
            (_digest(token),),
        ).fetchone()
    if row is None:
        return None
    invite = _invite(row)
    return None if invite.revoked or invite.expired else invite


def end_session(token: str) -> None:
    with _connect() as connection:
        connection.execute("DELETE FROM sessions WHERE token_hash = ?", (_digest(token),))


# -------------------------------------------------------------------------- quota


def reserve_question(invite_id: str) -> ReservationError | None:
    """Atomically take one quota unit and one attempt; return why it failed, if it did."""

    with _connect() as connection:
        changed = connection.execute(
            """
            UPDATE invites SET used = used + 1, attempts = attempts + 1, last_used_at = ?
            WHERE id = ? AND revoked = 0 AND used < quota AND attempts < max_attempts
              AND (expires_at IS NULL OR expires_at > ?)
            """,
            (_now(), invite_id, _now()),
        ).rowcount
    if changed == 1:
        return None
    invite = get_invite(invite_id)
    if invite is None or invite.revoked:
        return "revoked"
    if invite.expired:
        return "expired"
    return "quota_exhausted" if invite.used >= invite.quota else "attempts_exhausted"


def settle_question(invite_id: str, answered: bool) -> Invite | None:
    """Keep the quota unit for an answered question, give it back otherwise."""

    if not answered:
        with _connect() as connection:
            connection.execute("UPDATE invites SET used = MAX(used - 1, 0) WHERE id = ?", (invite_id,))
    return get_invite(invite_id)


def reserve_manual_paper(invite_id: str) -> bool:
    with _connect() as connection:
        return connection.execute(
            """
            UPDATE invites SET manual_papers_used = manual_papers_used + 1
            WHERE id = ? AND revoked = 0 AND manual_papers_used < max_manual_papers
            """,
            (invite_id,),
        ).rowcount == 1


def release_manual_paper(invite_id: str) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE invites SET manual_papers_used = MAX(manual_papers_used - 1, 0) WHERE id = ?",
            (invite_id,),
        )


# ---------------------------------------------------------------------- feedback


def add_feedback(
    *, invite_id: str | None, conversation_id: str, rating: Literal["up", "down"],
    comment: str | None, question: str | None, answer: str | None,
) -> None:
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO feedback (invite_id, conversation_id, rating, comment, question, answer, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (invite_id, conversation_id, rating, (comment or "")[:1000] or None,
             (question or "")[:500] or None, (answer or "")[:3000] or None, _now()),
        )


def list_feedback() -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT f.created_at, COALESCE(i.label, '') AS label, f.rating, f.comment,
                   f.question, f.answer, f.conversation_id
            FROM feedback f LEFT JOIN invites i ON i.id = f.invite_id ORDER BY f.id
            """
        ).fetchall()
    return [dict(row) for row in rows]


# ------------------------------------------------------------- rate and load limits


class RateLimiter:
    """In-memory sliding window: at most ``limit`` events per ``window`` seconds per key."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._clock = clock

    def allow(self, key: str, limit: int, window: float) -> bool:
        now = self._clock()
        with self._lock:
            events = self._events[key]
            while events and now - events[0] >= window:
                events.popleft()
            if len(events) >= limit:
                return False
            events.append(now)
            return True


rate_limiter = RateLimiter()


class QuestionGate:
    """Bound the number of questions processed at the same time (CPU and API budget)."""

    def __init__(self, slots: int) -> None:
        self._semaphore = threading.BoundedSemaphore(slots)

    def acquire(self, on_wait: Callable[[], None], timeout: float = 180.0) -> bool:
        if self._semaphore.acquire(blocking=False):
            return True
        on_wait()
        return self._semaphore.acquire(timeout=timeout)

    def release(self) -> None:
        self._semaphore.release()


_gates: dict[int, QuestionGate] = {}


def question_gate() -> QuestionGate | None:
    """Gate for ``MAX_CONCURRENT_QUESTIONS`` (default 2 in invite mode, unlimited otherwise)."""

    try:
        slots = int(os.getenv("MAX_CONCURRENT_QUESTIONS", "2" if access_mode() == "invite" else "0"))
    except ValueError:
        slots = 0
    if slots <= 0:
        return None
    return _gates.setdefault(slots, QuestionGate(slots))
