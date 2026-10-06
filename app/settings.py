"""Runtime configuration read from environment variables."""

import os
import math
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_OPENAI_MODEL = "gpt-6-luna"
REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}


@dataclass(frozen=True)
class OpenAISettings:
    """Validated API configuration; the secret is excluded from repr and reports."""

    api_key: str = field(repr=False)
    model_id: str
    reasoning_effort: str
    timeout_seconds: float
    max_retries: int
    max_output_tokens: int

    def public_configuration(self) -> dict[str, str | int | float]:
        """Describe an experiment without serializing its API key."""

        return {
            "provider": "openai",
            "endpoint": "responses",
            "model_id": self.model_id,
            "reasoning_effort": self.reasoning_effort,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_output_tokens": self.max_output_tokens,
        }


def get_openai_settings() -> OpenAISettings:
    """Read settings without contacting OpenAI or requiring a key at startup."""

    model_id = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip()
    effort = os.getenv("OPENAI_REASONING_EFFORT", "low").strip()
    if not model_id:
        raise ValueError("OPENAI_MODEL must not be empty.")
    if effort not in REASONING_EFFORTS:
        raise ValueError("OPENAI_REASONING_EFFORT must be none, low, medium, high, xhigh or max.")
    try:
        timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
    except ValueError:
        raise ValueError("OPENAI_TIMEOUT_SECONDS must be a number between 1 and 300.") from None
    if not math.isfinite(timeout) or not 1 <= timeout <= 300:
        raise ValueError("OPENAI_TIMEOUT_SECONDS must be a number between 1 and 300.")
    return OpenAISettings(
        api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        model_id=model_id,
        reasoning_effort=effort,
        timeout_seconds=timeout,
        max_retries=_integer_setting("OPENAI_MAX_RETRIES", default=1, minimum=0, maximum=2),
        max_output_tokens=_integer_setting(
            "OPENAI_MAX_OUTPUT_TOKENS", default=4096, minimum=16, maximum=32768
        ),
    )


def _integer_setting(name: str, *, default: int, minimum: int, maximum: int) -> int:
    """Reject invalid numeric options without echoing environment contents."""

    message = f"{name} must be an integer between {minimum} and {maximum}."
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        raise ValueError(message) from None
    if not minimum <= value <= maximum:
        raise ValueError(message)
    return value


def get_qdrant_url() -> str:
    """Return the internal Qdrant URL used by the API container."""

    return os.getenv("QDRANT_URL", "http://qdrant:6333")


def get_llm_model_path() -> Path:
    """Legacy downloader path only; the active LLM no longer reads these files."""

    return Path(
        os.getenv(
            "LLM_MODEL_PATH",
            "/models/huggingface/qwen2.5-7b-instruct-int4-ov",
        )
    )


def get_reranker_model_path() -> Path:
    """Return the persistent local path of the multilingual cross-encoder."""

    return Path(
        os.getenv(
            "RERANKER_MODEL_PATH",
            "/models/huggingface/mmarco-minilmv2-l12-h384-v1",
        )
    )
