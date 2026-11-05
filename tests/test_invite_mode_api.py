import json
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import access_control as access
from app import conversation_service, paper_registry
from app.main import app
from app.schemas import AgenticAskResponse
from tests.dynamic_helpers import RegistryTestCase

A = "11111111-2222-3333-4444-555555555555"
B = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def answer(abstained=False):
    return AgenticAskResponse.model_validate({
        "question": "q", "answer": "ok [S1]", "citations": [], "evidence": [], "retrieved_chunks": 0,
        "abstained": abstained,
        "execution": {"strategy": "global", "expected_language": "en", "detected_document_ids": [],
                      "planned_queries": [], "candidate_chunks": 0, "selected_chunks": 0,
                      "correction_attempted": False, "response_contract_valid": True, "steps": []}})


def events(response_text):
    out = []
    for block in response_text.strip().split("\n\n"):
        lines = [l for l in block.splitlines() if not l.startswith(":")]
        if len(lines) >= 2:
            out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


class InviteModeTests(RegistryTestCase):
    def setUp(self):
        super().setUp()
        env = patch.dict(os.environ, {"ACCESS_MODE": "invite", "OPENAI_API_KEY": "k", "MAX_CONCURRENT_QUESTIONS": "0"})
        env.start()
        self.addCleanup(env.stop)
        access.rate_limiter._events.clear()

    def client(self):
        c = TestClient(app)
        self.addCleanup(c.close)
        return c

    def logged_in(self, label="Alice", **kwargs):
        invite, code = access.create_invite(label, **kwargs)
        c = self.client()
        r = c.post("/access/redeem", json={"code": code})
        self.assertEqual(r.status_code, 200)
        return c, invite

    def ask(self, c, cid=A, **body):
        with c.stream("POST", f"/conversations/{cid}/ask", json={"question": "What is Mamba?", "discovery": "off", **body}) as r:
            return r.status_code, r.headers.get("content-type", ""), "".join(r.iter_text())

    def test_dangerous_routes_do_not_exist_in_invite_mode(self):
        c, _ = self.logged_in()
        for method, path in [("post", "/ask"), ("post", "/agentic/ask"), ("get", "/review/"), ("get", "/docs"),
                             ("get", "/openapi.json"), ("get", "/evaluation/cases"), ("get", "/arxiv/search?query=abc"),
                             ("get", "/arxiv/documents"), ("post", "/documents/clip-2021/convert"),
                             ("post", "/demo/index"), ("get", "/llm/status"), ("get", "/ready")]:
            self.assertEqual(getattr(c, method)(path).status_code, 404, path)
        for path in ("/health", "/access/me", "/ui/"):
            self.assertEqual(c.get(path).status_code, 200, path)

    def test_open_mode_keeps_everything(self):
        with patch.dict(os.environ, {"ACCESS_MODE": "open"}):
            c = self.client()
            self.assertEqual(c.get("/access/me").json(), {"mode": "open", "authenticated": True})
            self.assertEqual(c.get("/docs").status_code, 200)
            self.assertEqual(c.post("/access/redeem", json={"code": "whatever-code"}).status_code, 404)

    def test_login_flow_and_cookie_flags(self):
        invite, code = access.create_invite("Alice")
        c = self.client()
        self.assertEqual(c.get("/access/me").json()["authenticated"], False)
        self.assertEqual(c.post("/access/redeem", json={"code": "nope-nope-nope"}).status_code, 403)
        behind_tunnel = self.client().post("/access/redeem", json={"code": code}, headers={"x-forwarded-proto": "https"})
        cookie = behind_tunnel.headers["set-cookie"].lower()
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=lax", cookie)
        self.assertIn("secure", cookie)  # served over HTTPS by the tunnel
        r = c.post("/access/redeem", json={"code": code})  # plain HTTP: no Secure flag, cookie usable
        self.assertNotIn("secure", r.headers["set-cookie"].lower())
        me = c.get("/access/me").json()
        self.assertEqual((me["authenticated"], me["invite"]["remaining"], me["invite"]["quota"]), (True, 20, 20))
        self.assertNotIn("code_hash", json.dumps(me))
        c.post("/access/logout")
        self.assertEqual(c.get("/access/me").json()["authenticated"], False)

    def test_redeem_is_rate_limited(self):
        c = self.client()
        codes = [c.post("/access/redeem", json={"code": "wrong-code-xx"}).status_code for _ in range(10)]
        self.assertEqual(codes[:8], [403] * 8)
        self.assertEqual(codes[8:], [429, 429])

    def test_conversation_routes_need_a_session(self):
        c = self.client()
        self.assertEqual(c.get(f"/conversations/{A}/papers").status_code, 401)
        self.assertEqual(c.post(f"/conversations/{A}/ask", json={"question": "abc def"}).status_code, 401)
        self.assertEqual(c.post(f"/conversations/{A}/feedback", json={"rating": "up"}).status_code, 401)

    def test_a_tester_cannot_touch_another_testers_conversation(self):
        alice, _ = self.logged_in("Alice")
        bob, _ = self.logged_in("Bob")
        self.assertEqual(alice.get(f"/conversations/{A}/papers").status_code, 200)
        for call in (lambda: bob.get(f"/conversations/{A}/papers"), lambda: bob.delete(f"/conversations/{A}"),
                     lambda: bob.put(f"/conversations/{A}"), lambda: bob.post(f"/conversations/{A}/feedback", json={"rating": "up"})):
            self.assertEqual(call().status_code, 404)
        self.assertEqual(self.ask(bob)[0], 404)
        self.assertEqual(alice.get(f"/conversations/{A}/papers").status_code, 200)

    def test_answered_question_consumes_quota_and_reports_it(self):
        c, invite = self.logged_in(quota=2)
        with patch.object(conversation_service, "answer_agentic_question", return_value=answer()):
            status, ctype, body = self.ask(c)
        self.assertEqual(status, 200)
        self.assertTrue(ctype.startswith("text/event-stream"))
        quota = dict(events(body))["quota"]
        self.assertEqual((quota["used"], quota["remaining"]), (1, 1))
        self.assertEqual(c.get("/access/me").json()["invite"]["remaining"], 1)

    def test_abstention_and_failures_give_the_quota_back(self):
        c, invite = self.logged_in(quota=2)
        with patch.object(conversation_service, "answer_agentic_question", return_value=answer(abstained=True)):
            self.assertEqual(dict(events(self.ask(c)[2]))["quota"]["remaining"], 2)
        with patch.object(conversation_service, "answer_agentic_question", side_effect=RuntimeError("boom")):
            body = self.ask(c)[2]
        self.assertEqual(dict(events(body))["quota"]["remaining"], 2)
        self.assertIn("error", dict(events(body)))
        self.assertEqual(access.get_invite(invite.id).attempts, 2)  # attempts are never refunded

    def test_exhausted_quota_is_refused_before_any_work(self):
        c, invite = self.logged_in(quota=1)
        with patch.object(conversation_service, "answer_agentic_question", return_value=answer()) as work:
            self.ask(c)
            status, _, body = self.ask(c)
        self.assertEqual(status, 429)
        self.assertIn("épuisé", json.loads(body)["detail"])
        self.assertEqual(work.call_count, 1)

    def test_attempt_cap_bounds_cost_even_when_everything_abstains(self):
        c, _ = self.logged_in(quota=5, max_attempts=2)
        with patch.object(conversation_service, "answer_agentic_question", return_value=answer(abstained=True)):
            self.assertEqual([self.ask(c)[0] for _ in range(3)], [200, 200, 429])

    def test_per_invitation_rate_limit(self):
        c, _ = self.logged_in(quota=20)
        with patch.object(conversation_service, "answer_agentic_question", return_value=answer()):
            statuses = [self.ask(c)[0] for _ in range(7)]
        self.assertEqual(statuses, [200] * 6 + [429])

    def test_revoking_ends_access_immediately(self):
        c, invite = self.logged_in()
        access.revoke_invite(invite.id)
        self.assertEqual(c.get(f"/conversations/{A}/papers").status_code, 401)

    def test_feedback_is_stored_with_the_invitation_label(self):
        c, _ = self.logged_in("Alice")
        r = c.post(f"/conversations/{A}/feedback", json={"rating": "down", "comment": "formule fausse", "question": "q?", "answer": "a"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual([(f["label"], f["rating"], f["comment"]) for f in access.list_feedback()], [("Alice", "down", "formule fausse")])
        self.assertEqual(c.post(f"/conversations/{A}/feedback", json={"rating": "meh"}).status_code, 422)

    def test_manual_paper_additions_are_capped_and_refunded_on_failure(self):
        from app import paper_ingestion
        c, invite = self.logged_in(max_manual_papers=1)
        with patch.object(paper_ingestion, "ingest_arxiv_paper", side_effect=paper_ingestion.IngestionError("nope")):
            self.assertEqual(c.post(f"/conversations/{A}/papers", json={"arxiv_id": "2312.00752"}).status_code, 422)
        self.assertEqual(access.get_invite(invite.id).manual_papers_used, 0)
        paper = self.make_paper()
        with patch.object(paper_ingestion, "ingest_arxiv_paper", return_value=paper):
            self.assertEqual(c.post(f"/conversations/{A}/papers", json={"arxiv_id": "2312.00752"}).status_code, 200)
            self.assertEqual(c.post(f"/conversations/{A}/papers", json={"arxiv_id": "2312.00752"}).status_code, 429)

    def test_busy_server_queues_then_gives_up_and_refunds(self):
        c, invite = self.logged_in(quota=2)
        gate = access.QuestionGate(1)
        gate.acquire(lambda: None)  # someone else holds the only slot
        gate_timeout = access.QuestionGate.acquire

        def quick(self_, on_wait, timeout=180.0):
            return gate_timeout(self_, on_wait, timeout=0.05)

        with patch.dict(os.environ, {"MAX_CONCURRENT_QUESTIONS": "1"}), \
             patch.dict(access._gates, {1: gate}), patch.object(access.QuestionGate, "acquire", quick), \
             patch.object(conversation_service, "answer_agentic_question", return_value=answer()) as work:
            _, _, body = self.ask(c)
        names = [n for n, _ in events(body)]
        self.assertEqual(names[:2], ["queue.waiting", "error"])
        self.assertEqual(dict(events(body))["error"]["code"], "server_busy")
        work.assert_not_called()
        self.assertEqual(access.get_invite(invite.id).used, 0)

    def test_daily_download_cap_in_discovery(self):
        from app import source_discovery
        with patch.dict(os.environ, {"MAX_NEW_PAPERS_PER_DAY": "1"}):
            self.make_paper("2401.00001", "v1")  # one paper already fetched today
            session = source_discovery._Session(A, lambda *_: None)
            result = session.add_paper_to_conversation({"arxiv_id": "2401.00002", "short_name": None, "reason": "r"})
        self.assertEqual(result["error"], "daily_download_cap")


if __name__ == "__main__":
    unittest.main()
