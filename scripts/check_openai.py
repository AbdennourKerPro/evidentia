"""Check configuration without billing, or explicitly test one live Responses call."""

from __future__ import annotations

import argparse
import json

from app.llm_gateway import (
    LlmGatewayError,
    capture_llm_calls,
    generate_chat,
    get_llm_configuration,
    summarize_llm_calls,
)
from app.settings import get_openai_settings


def main() -> int:
    """Require --live for a network request; this script never loads retrieval models."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Make one small, billable API call.")
    args = parser.parse_args()
    try:
        configuration = get_llm_configuration()
        configured = bool(get_openai_settings().api_key)
        print(json.dumps({"configuration": configuration, "configured": configured}, indent=2))
        if not args.live:
            print("No API call made. Use --live to verify model access.")
            return 0
        with capture_llm_calls() as calls:
            answer = generate_chat(
                system_message="Follow the user's requested output format exactly.",
                user_message="Réponds uniquement par : API opérationnelle.",
                max_output_tokens=2048,
            )
        print(json.dumps(
            {"answer": answer, "llm": summarize_llm_calls(calls).model_dump()},
            indent=2,
            ensure_ascii=False,
        ))
        return 0
    except LlmGatewayError as error:
        print(json.dumps({"error": error.code, "detail": str(error)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
