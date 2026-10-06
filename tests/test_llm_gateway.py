"""Contract and failure tests for the real OpenAI SDK, without billed calls."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

import httpx

from app.llm_gateway import (
    LlmGatewayError,
    _get_client,
    capture_llm_calls,
    generate_chat,
    get_llm_configuration,
    summarize_llm_calls,
)
from app.settings import get_openai_settings
from tests.openai_fixtures import FAKE_API_KEY, TEST_ENV, mock_client, response_body


class LlmGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env_patch = patch.dict(os.environ, TEST_ENV)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        _get_client.cache_clear()
        self.addCleanup(_get_client.cache_clear)
        self.requests: list[dict] = []

    def install_transport(self, *, body: dict | None = None, status: int = 200) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url), "https://api.openai.com/v1/responses")
            self.requests.append(json.loads(request.content))
            return httpx.Response(status, json=body if body is not None else response_body())

        client = mock_client(handler)
        self.addCleanup(client.close)
        client_patch = patch("app.llm_gateway._get_client", return_value=client)
        client_patch.start()
        self.addCleanup(client_patch.stop)

    def generate(self, **kwargs) -> str:
        return generate_chat(
            system_message="Private system instruction.",
            user_message="Private question and evidence.",
            **kwargs,
        )

    def test_uses_responses_luna_and_supported_parameters(self) -> None:
        self.install_transport()
        answer = self.generate(max_output_tokens=3072)
        request = self.requests[0]
        self.assertIn("[S1]", answer)
        self.assertEqual(request["model"], "gpt-6-luna")
        self.assertEqual(request["reasoning"], {"effort": "low"})
        self.assertEqual(request["instructions"], "Private system instruction.")
        self.assertEqual(request["input"][0]["role"], "user")
        self.assertEqual(request["max_output_tokens"], 3072)
        self.assertFalse(request["store"])
        self.assertNotIn("temperature", request)
        self.assertNotIn("top_p", request)
        self.assertNotIn("max_new_tokens", request)

    def test_applies_server_output_cap(self) -> None:
        self.install_transport()
        with patch.dict(os.environ, {"OPENAI_MAX_OUTPUT_TOKENS": "128"}):
            self.generate(max_output_tokens=2048)
        self.assertEqual(self.requests[0]["max_output_tokens"], 128)

    def test_supports_no_reasoning_for_fast_experiments(self) -> None:
        self.install_transport()
        with patch.dict(os.environ, {"OPENAI_REASONING_EFFORT": "none"}):
            self.generate()
        self.assertEqual(self.requests[0]["reasoning"], {"effort": "none"})

    def test_model_configuration_is_shared_and_explicit(self) -> None:
        self.install_transport()
        with patch.dict(os.environ, {"OPENAI_MODEL": "explicit-test-model"}):
            self.generate()
            self.assertEqual(get_llm_configuration()["model_id"], "explicit-test-model")
        self.assertEqual(self.requests[0]["model"], "explicit-test-model")

    def test_missing_key_fails_before_constructing_client(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with patch("app.llm_gateway.OpenAI") as constructor:
                with self.assertRaises(LlmGatewayError) as caught:
                    self.generate()
                constructor.assert_not_called()
        self.assertEqual(caught.exception.code, "missing_api_key")

    def test_reuses_client_with_explicit_timeout_and_retry_limit(self) -> None:
        with patch("app.llm_gateway.OpenAI") as constructor:
            settings = get_openai_settings()
            self.assertIs(_get_client(settings), _get_client(settings))
            constructor.assert_called_once_with(
                api_key=FAKE_API_KEY,
                base_url="https://api.openai.com/v1",
                timeout=60.0,
                max_retries=0,
            )

    def test_configuration_and_repr_never_expose_key(self) -> None:
        self.assertNotIn(FAKE_API_KEY, repr(get_openai_settings()))
        self.assertNotIn(FAKE_API_KEY, json.dumps(get_llm_configuration()))
        self.assertNotIn("api_key", get_llm_configuration())

    def test_invalid_configuration_is_rejected(self) -> None:
        invalid_values = {
            "OPENAI_REASONING_EFFORT": "unsupported",
            "OPENAI_TIMEOUT_SECONDS": "nan",
            "OPENAI_MAX_RETRIES": "-1",
            "OPENAI_MAX_OUTPUT_TOKENS": "not-an-integer",
            "OPENAI_MODEL": "",
        }
        for name, value in invalid_values.items():
            with self.subTest(name=name), patch.dict(os.environ, {name: value}):
                with self.assertRaises(LlmGatewayError) as caught:
                    get_llm_configuration()
                self.assertEqual(caught.exception.code, "invalid_llm_configuration")

    def test_captures_usage_not_prompt_contents(self) -> None:
        self.install_transport()
        with capture_llm_calls() as calls:
            self.generate()
        usage = summarize_llm_calls(calls)
        self.assertEqual(usage.call_count, 1)
        self.assertEqual(usage.total_tokens, 46)
        self.assertEqual(usage.reasoning_tokens, 7)
        self.assertEqual(usage.cached_input_tokens, 4)
        serialized = usage.model_dump_json()
        self.assertNotIn(FAKE_API_KEY, serialized)
        self.assertNotIn("Private", serialized)

    def test_nested_collectors_do_not_mix_requests(self) -> None:
        self.install_transport()
        with capture_llm_calls() as outer:
            self.generate()
            with capture_llm_calls() as inner:
                self.generate()
            self.generate()
        self.assertEqual(len(outer), 2)
        self.assertEqual(len(inner), 1)

    def test_no_call_has_zero_usage(self) -> None:
        usage = summarize_llm_calls([])
        self.assertEqual(usage.call_count, 0)
        self.assertEqual(usage.total_tokens, 0)

    def test_missing_provider_usage_remains_unknown(self) -> None:
        body = response_body()
        body["usage"] = None
        self.install_transport(body=body)
        with capture_llm_calls() as calls:
            self.generate()
        self.assertIsNone(summarize_llm_calls(calls).total_tokens)

    def test_rejects_truncated_response_and_keeps_usage(self) -> None:
        body = response_body("Partial answer [S")
        body["status"] = "incomplete"
        body["incomplete_details"] = {"reason": "max_output_tokens"}
        self.install_transport(body=body)
        with capture_llm_calls() as calls, self.assertRaises(LlmGatewayError) as caught:
            self.generate()
        self.assertEqual(caught.exception.code, "incomplete_llm_response")
        self.assertEqual(calls[0].total_tokens, 46)

    def test_rejects_empty_response(self) -> None:
        self.install_transport(body=response_body(" "))
        with self.assertRaises(LlmGatewayError) as caught:
            self.generate()
        self.assertEqual(caught.exception.code, "empty_llm_response")

    def test_rejects_refusal_without_echoing_provider_text(self) -> None:
        body = response_body()
        body["output"][0]["content"] = [{"type": "refusal", "refusal": "Private refusal."}]
        self.install_transport(body=body)
        with self.assertRaises(LlmGatewayError) as caught:
            self.generate()
        self.assertEqual(caught.exception.code, "llm_refusal")
        self.assertNotIn("Private", str(caught.exception))

    def test_rejects_failed_response(self) -> None:
        body = response_body()
        body["status"] = "failed"
        self.install_transport(body=body)
        with self.assertRaises(LlmGatewayError) as caught:
            self.generate()
        self.assertEqual(caught.exception.code, "failed_llm_response")

    def test_sanitizes_authentication_permission_quota_and_server_errors(self) -> None:
        for status, code in [
            (401, "invalid_api_key"),
            (403, "model_access_denied"),
            (404, "model_unavailable"),
            (429, "upstream_rate_limit"),
            (500, "upstream_api_error"),
            (400, "upstream_api_error"),
        ]:
            with self.subTest(status=status):
                client = mock_client(lambda request: httpx.Response(
                    status,
                    json={"error": {"message": FAKE_API_KEY + " Private evidence."}},
                ))
                with client, patch("app.llm_gateway._get_client", return_value=client):
                    with capture_llm_calls() as calls, self.assertRaises(LlmGatewayError) as caught:
                        self.generate()
                self.assertEqual(caught.exception.code, code)
                self.assertNotIn(FAKE_API_KEY, str(caught.exception))
                self.assertIsNone(summarize_llm_calls(calls).total_tokens)

    def test_sanitizes_timeout_and_connection_errors(self) -> None:
        for error_type, code, expected_status in [
            (httpx.ReadTimeout, "llm_timeout", 504),
            (httpx.ConnectError, "llm_connection_error", 502),
        ]:
            with self.subTest(code=code):
                def handler(request):
                    raise error_type("Private connection details.", request=request)
                client = mock_client(handler)
                with client, patch("app.llm_gateway._get_client", return_value=client):
                    with self.assertRaises(LlmGatewayError) as caught:
                        self.generate()
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(caught.exception.http_status, expected_status)


if __name__ == "__main__":
    unittest.main()
