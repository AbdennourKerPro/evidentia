import sqlite3
import threading
import unittest
from datetime import datetime, timedelta, timezone

from app import access_control as access
from app import paper_registry
from tests.dynamic_helpers import RegistryTestCase

CONV = "11111111-2222-3333-4444-555555555555"


class InviteTests(RegistryTestCase):
    def test_code_is_returned_once_and_only_its_hash_is_stored(self):
        invite, code = access.create_invite("Alice", quota=20)
        self.assertGreaterEqual(len(code), 20)
        with paper_registry._connect() as c:
            dump = " ".join(str(tuple(row)) for row in c.execute("SELECT * FROM invites"))
        self.assertNotIn(code, dump)
        self.assertEqual((invite.quota, invite.max_attempts, invite.max_manual_papers), (20, 60, 3))

    def test_redeem_valid_invalid_expired_revoked(self):
        invite, code = access.create_invite("Alice")
        redeemed = access.redeem(code)
        self.assertIsNotNone(redeemed)
        self.assertEqual(access.invite_for_session(redeemed[1]).id, invite.id)
        self.assertIsNone(access.redeem("wrong-code-value"))
        self.assertIsNone(access.invite_for_session("not-a-token"))
        expired, expired_code = access.create_invite("Old", days=1)
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        with paper_registry._connect() as c:
            c.execute("UPDATE invites SET expires_at = ? WHERE id = ?", (past, expired.id))
        self.assertIsNone(access.redeem(expired_code))
        access.revoke_invite(invite.id)
        self.assertIsNone(access.invite_for_session(redeemed[1]))  # sessions end with the invitation
        self.assertIsNone(access.redeem(code))

    def test_reserve_and_settle_count_only_answered_questions(self):
        invite, _ = access.create_invite("A", quota=2, max_attempts=4)
        self.assertIsNone(access.reserve_question(invite.id))
        self.assertEqual(access.settle_question(invite.id, answered=True).used, 1)
        self.assertIsNone(access.reserve_question(invite.id))
        after = access.settle_question(invite.id, answered=False)  # abstention: quota unit returned
        self.assertEqual((after.used, after.attempts, after.remaining), (1, 2, 1))

    def test_quota_exhaustion_and_attempt_cap(self):
        invite, _ = access.create_invite("A", quota=1, max_attempts=3)
        self.assertIsNone(access.reserve_question(invite.id))
        access.settle_question(invite.id, True)
        self.assertEqual(access.reserve_question(invite.id), "quota_exhausted")
        spammer, _ = access.create_invite("B", quota=5, max_attempts=2)
        for _ in range(2):
            self.assertIsNone(access.reserve_question(spammer.id))
            access.settle_question(spammer.id, False)  # only abstentions: quota never runs out
        self.assertEqual(access.reserve_question(spammer.id), "attempts_exhausted")

    def test_in_flight_reservations_cannot_overshoot_the_quota(self):
        invite, _ = access.create_invite("A", quota=3, max_attempts=30)
        results = []

        def attempt():
            results.append(access.reserve_question(invite.id))

        threads = [threading.Thread(target=attempt) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(results.count(None), 3)
        self.assertEqual(access.get_invite(invite.id).used, 3)

    def test_revoked_and_expired_reasons(self):
        invite, _ = access.create_invite("A")
        access.revoke_invite(invite.id)
        self.assertEqual(access.reserve_question(invite.id), "revoked")

    def test_manual_paper_budget(self):
        invite, _ = access.create_invite("A", max_manual_papers=1)
        self.assertTrue(access.reserve_manual_paper(invite.id))
        self.assertFalse(access.reserve_manual_paper(invite.id))
        access.release_manual_paper(invite.id)
        self.assertTrue(access.reserve_manual_paper(invite.id))

    def test_add_quota_also_raises_the_attempt_cap(self):
        invite, _ = access.create_invite("A", quota=2)
        access.add_quota(invite.id, 5)
        updated = access.get_invite(invite.id)
        self.assertEqual((updated.quota, updated.max_attempts), (7, 6 + 15))

    def test_feedback_roundtrip(self):
        invite, _ = access.create_invite("Alice")
        access.add_feedback(invite_id=invite.id, conversation_id=CONV, rating="down", comment="faux", question="q?", answer="a")
        rows = access.list_feedback()
        self.assertEqual([(r["label"], r["rating"], r["comment"]) for r in rows], [("Alice", "down", "faux")])


class OwnershipTests(RegistryTestCase):
    def test_conversation_belongs_to_its_first_invitation(self):
        self.assertTrue(paper_registry.claim_conversation(CONV, "inv-a"))
        self.assertTrue(paper_registry.claim_conversation(CONV, "inv-a"))
        self.assertFalse(paper_registry.claim_conversation(CONV, "inv-b"))

    def test_ownerless_legacy_conversation_cannot_be_claimed(self):
        paper_registry.ensure_conversation(CONV)  # created before invitations: no owner
        self.assertFalse(paper_registry.claim_conversation(CONV, "inv-a"))

    def test_old_database_without_owner_column_is_migrated(self):
        import os
        path = os.environ["STATE_DB_PATH"]
        with sqlite3.connect(path) as c:
            c.executescript("DROP TABLE conversation_turns; DROP TABLE conversation_papers; DROP TABLE conversations;"
                            "CREATE TABLE conversations (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, last_active_at TEXT NOT NULL);")
        paper_registry.init_db()
        self.assertTrue(paper_registry.claim_conversation(CONV, "inv-a"))


class LimiterTests(unittest.TestCase):
    def test_sliding_window(self):
        now = [0.0]
        limiter = access.RateLimiter(clock=lambda: now[0])
        self.assertTrue(all(limiter.allow("k", 3, 60) for _ in range(3)))
        self.assertFalse(limiter.allow("k", 3, 60))
        self.assertTrue(limiter.allow("other", 3, 60))
        now[0] = 61
        self.assertTrue(limiter.allow("k", 3, 60))

    def test_gate_reports_waiting_and_times_out(self):
        gate = access.QuestionGate(1)
        waits = []
        self.assertTrue(gate.acquire(lambda: waits.append(1)))
        self.assertFalse(gate.acquire(lambda: waits.append(1), timeout=0.05))
        self.assertEqual(waits, [1])
        gate.release()
        self.assertTrue(gate.acquire(lambda: waits.append(1)))


if __name__ == "__main__":
    unittest.main()
