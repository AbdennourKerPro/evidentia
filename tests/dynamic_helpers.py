"""Shared isolation for tests that touch the paper registry and cache."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import paper_registry


class RegistryTestCase(unittest.TestCase):
    """Each test gets a private SQLite file and cache directory."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = patch.dict(os.environ, {
            "STATE_DB_PATH": str(Path(self.tmp.name) / "state" / "db.sqlite3"),
            "ARXIV_CACHE_DIR": str(Path(self.tmp.name) / "cache"),
        })
        env.start()
        self.addCleanup(env.stop)
        paper_registry.init_db()

    def make_paper(self, arxiv_id="2312.00752", version="v2", title="Mamba: Linear-Time Sequence Modeling",
                   status="ready", pinned=False, aliases=None, chunks=10):
        return paper_registry.upsert_paper(
            document_id=f"arxiv-{arxiv_id}{version}", arxiv_id=arxiv_id, version=version,
            title=title, aliases=aliases or [], status=status, pinned=pinned, chunk_count=chunks,
        )
