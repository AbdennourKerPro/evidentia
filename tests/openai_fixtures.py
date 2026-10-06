"""Real SDK response fixtures for tests using an in-memory HTTP transport."""

from __future__ import annotations

import httpx
from openai import OpenAI


FAKE_API_KEY = "unit-test-key-never-use-in-production"
TEST_ENV = {
    "OPENAI_API_KEY": FAKE_API_KEY,
    "OPENAI_MODEL": "gpt-6-luna",
    "OPENAI_REASONING_EFFORT": "low",
    "OPENAI_TIMEOUT_SECONDS": "60",
    "OPENAI_MAX_RETRIES": "0",
    "OPENAI_MAX_OUTPUT_TOKENS": "4096",
}


def response_body(text: str = "The answer is supported by the evidence [S1].") -> dict:
    """Supply actual Responses message and usage shapes, not a fake SDK class."""

    return {
        "id": "resp_unit_test",
        "object": "response",
        "created_at": 1780000000,
        "model": "gpt-6-luna",
        "status": "completed",
        "output": [{
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }],
        "usage": {
            "input_tokens": 17,
            "output_tokens": 29,
            "total_tokens": 46,
            "input_tokens_details": {"cached_tokens": 4},
            "output_tokens_details": {"reasoning_tokens": 7},
        },
    }


def mock_client(handler) -> OpenAI:
    """Never access the network: every request is resolved by MockTransport."""

    return OpenAI(
        api_key=FAKE_API_KEY,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
