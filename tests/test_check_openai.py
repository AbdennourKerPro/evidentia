"""The smoke script is offline by default; --live must be explicitly supplied."""

from __future__ import annotations

import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import httpx

from scripts.check_openai import main
from tests.openai_fixtures import FAKE_API_KEY, TEST_ENV, mock_client, response_body


class CheckOpenaiTests(unittest.TestCase):
    def test_default_does_not_contact_provider(self) -> None:
        output = io.StringIO()
        with patch.dict(os.environ, TEST_ENV), patch("sys.argv", ["check_openai"]), patch(
            "app.llm_gateway._get_client"
        ) as factory, redirect_stdout(output):
            result = main()
        factory.assert_not_called()
        self.assertEqual(result, 0)
        self.assertIn("No API call made", output.getvalue())
        self.assertNotIn(FAKE_API_KEY, output.getvalue())

    def test_live_flag_makes_exactly_one_mocked_request(self) -> None:
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=response_body("API opérationnelle."))

        client = mock_client(handler)
        output = io.StringIO()
        with client, patch.dict(os.environ, TEST_ENV), patch(
            "sys.argv", ["check_openai", "--live"]
        ), patch("app.llm_gateway._get_client", return_value=client), redirect_stdout(output):
            result = main()
        self.assertEqual(result, 0)
        self.assertEqual(len(requests), 1)
        self.assertIn("API opérationnelle.", output.getvalue())
        self.assertNotIn(FAKE_API_KEY, output.getvalue())

    def test_missing_key_live_returns_controlled_error(self) -> None:
        output = io.StringIO()
        with patch.dict(os.environ, dict(TEST_ENV, OPENAI_API_KEY="")), patch(
            "sys.argv", ["check_openai", "--live"]
        ), redirect_stdout(output):
            result = main()
        self.assertEqual(result, 1)
        self.assertIn("missing_api_key", output.getvalue())
