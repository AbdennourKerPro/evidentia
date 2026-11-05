import json
import os
import unittest
from unittest.mock import patch

import httpx

from app.llm_gateway import LlmGatewayError, _get_client, capture_llm_calls, run_tool_loop
from tests.openai_fixtures import TEST_ENV, function_call_body, mock_client, response_body

TOOLS = [{"type": "function", "name": "echo", "description": "d", "strict": True,
          "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}]


class ToolLoopTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, TEST_ENV)
        env.start()
        self.addCleanup(env.stop)
        _get_client.cache_clear()
        self.addCleanup(_get_client.cache_clear)
        self.requests = []

    def install(self, bodies):
        queue = list(bodies)

        def handler(request):
            self.requests.append(json.loads(request.content))
            return httpx.Response(200, json=queue.pop(0))

        client = mock_client(handler)
        self.addCleanup(client.close)
        p = patch("app.llm_gateway._get_client", return_value=client)
        p.start()
        self.addCleanup(p.stop)

    def run_loop(self, execute, budget=4, events=None):
        return run_tool_loop(
            system_message="sys", user_message="hello", tools=TOOLS, execute_tool=execute,
            final_tool_name="finish", max_model_calls=budget,
            on_event=(lambda k, p: events.append((k, p))) if events is not None else None,
        )

    def test_search_then_finish_replays_history_with_matching_call_ids(self):
        self.install([
            function_call_body("echo", {"x": 1}, "call_a"),
            function_call_body("finish", {"done": True}, "call_b"),
        ])
        events = []
        with capture_llm_calls() as calls:
            result = self.run_loop(lambda n, a: {"ok": True, "seen": a}, events=events)
        self.assertEqual(result.stopped_reason, "finished")
        self.assertEqual([c.name for c in result.calls], ["echo", "finish"])
        self.assertEqual(result.final_call.arguments, {"done": True})
        self.assertEqual(len(calls), 2)  # both model calls counted in usage metrics
        first, second = self.requests
        self.assertEqual(first["tool_choice"], "required")
        self.assertFalse(first["parallel_tool_calls"])
        self.assertFalse(first["store"])
        self.assertEqual(first["tools"][0]["name"], "echo")
        outputs = [i for i in second["input"] if i.get("type") == "function_call_output"]
        self.assertEqual([o["call_id"] for o in outputs], ["call_a"])
        self.assertEqual(json.loads(outputs[0]["output"]), {"ok": True, "seen": {"x": 1}})
        self.assertTrue(any(i.get("type") == "function_call" for i in second["input"]))
        self.assertEqual([k for k, _ in events], ["tool_call", "tool_result"] * 2)

    def test_invalid_json_arguments_are_returned_to_the_model(self):
        bad = function_call_body("echo", {})
        bad["output"][0]["arguments"] = "{not json"
        self.install([bad, function_call_body("finish", {})])
        executed = []
        result = self.run_loop(lambda n, a: executed.append(n) or {"ok": True})
        self.assertEqual(result.calls[0].result["error"], "invalid_arguments")
        self.assertEqual(executed, ["finish"])  # the broken call never reached a tool

    def test_budget_exhaustion(self):
        self.install([function_call_body("echo", {}, f"c{i}") for i in range(3)])
        result = self.run_loop(lambda n, a: {"ok": True}, budget=3)
        self.assertEqual(result.stopped_reason, "budget")
        self.assertEqual(result.model_calls, 3)
        self.assertIsNone(result.final_call)

    def test_text_answer_without_tool_ends_loop(self):
        self.install([response_body("I will just answer.")])
        result = self.run_loop(lambda n, a: {"ok": True})
        self.assertEqual(result.stopped_reason, "no_tool_call")

    def test_failed_final_call_does_not_finish(self):
        self.install([function_call_body("finish", {}, "c1"), function_call_body("finish", {}, "c2")])
        answers = iter([{"ok": False, "error": "bad"}, {"ok": True}])
        result = self.run_loop(lambda n, a: next(answers))
        self.assertEqual(result.stopped_reason, "finished")
        self.assertEqual(len(result.calls), 2)

    def test_incomplete_response_raises_safe_error(self):
        body = function_call_body("echo", {})
        body["status"] = "incomplete"
        self.install([body])
        with self.assertRaises(LlmGatewayError):
            self.run_loop(lambda n, a: {"ok": True})


if __name__ == "__main__":
    unittest.main()
