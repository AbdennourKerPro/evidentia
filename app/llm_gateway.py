"""Single OpenAI Responses API boundary for every generative LLM call."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from time import perf_counter

import openai
from openai import OpenAI
from openai.types.responses import Response

from app.schemas import LlmCallMetrics, LlmUsage
from app.settings import OpenAISettings, get_openai_settings


logger = logging.getLogger(__name__)
_call_collector: ContextVar[list[LlmCallMetrics] | None] = ContextVar(
    "llm_call_collector", default=None
)


class LlmGatewayError(RuntimeError):
    """Safe application error: never expose the provider's raw response or key."""

    def __init__(self, message: str, *, code: str, http_status: int = 503) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status


def _read_settings() -> OpenAISettings:
    """Translate configuration errors into a predictable service response."""

    try:
        return get_openai_settings()
    except ValueError as error:
        raise LlmGatewayError(str(error), code="invalid_llm_configuration") from None


def get_llm_configuration() -> dict[str, str | int | float]:
    """Return public model/experiment metadata, without an API request."""

    return _read_settings().public_configuration()


@lru_cache(maxsize=1)
def _get_client(settings: OpenAISettings) -> OpenAI:
    """Reuse HTTP connections; do not load any local model or tokenizer."""

    if not settings.api_key:
        raise LlmGatewayError(
            "OPENAI_API_KEY is missing. Configure it on the API server and recreate the api container.",
            code="missing_api_key",
        )
    return OpenAI(
        api_key=settings.api_key,
        base_url="https://api.openai.com/v1",
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )


@contextmanager
def capture_llm_calls() -> Iterator[list[LlmCallMetrics]]:
    """Collect calls for one HTTP request or evaluation case, not globally."""

    calls: list[LlmCallMetrics] = []
    token = _call_collector.set(calls)
    try:
        yield calls
    finally:
        _call_collector.reset(token)


def summarize_llm_calls(calls: list[LlmCallMetrics]) -> LlmUsage:
    """Keep missing usage unknown; it must not be mistaken for zero billed tokens."""

    def total(field: str) -> int | None:
        values = [getattr(call, field) for call in calls]
        return None if any(value is None for value in values) else sum(values)

    settings = _read_settings()
    return LlmUsage(
        model_id=settings.model_id,
        reasoning_effort=settings.reasoning_effort,
        call_count=len(calls),
        input_tokens=total("input_tokens"),
        output_tokens=total("output_tokens"),
        total_tokens=total("total_tokens"),
        reasoning_tokens=total("reasoning_tokens"),
        cached_input_tokens=total("cached_input_tokens"),
        calls=list(calls),
    )


def generate_chat(
    *, system_message: str, user_message: str, max_output_tokens: int = 4096
) -> str:
    """Generate text with Luna; reject refusals, empty and truncated responses.

    Unlike OpenVINO's old max_new_tokens, the API limit includes reasoning tokens.
    Prompts, retrieval and validators are deliberately unchanged in this milestone.
    """

    if max_output_tokens < 16:
        raise ValueError("max_output_tokens must be at least 16.")
    settings = _read_settings()
    client = _get_client(settings)
    token_limit = min(max_output_tokens, settings.max_output_tokens)
    started_at = perf_counter()
    try:
        response = client.responses.create(
            model=settings.model_id,
            instructions=system_message,
            input=[{"role": "user", "content": user_message}],
            reasoning={"effort": settings.reasoning_effort},
            max_output_tokens=token_limit,
            store=False,
        )
    except openai.APIError as error:
        safe_error = _translate_api_error(error)
        _record_call(
            settings=settings,
            started_at=started_at,
            token_limit=token_limit,
            error_code=safe_error.code,
        )
        raise safe_error from None

    _record_call(
        settings=settings,
        started_at=started_at,
        token_limit=token_limit,
        response=response,
    )
    if response.status == "incomplete":
        raise LlmGatewayError(
            "The LLM response is incomplete. Check the output-token budget; no truncated answer was published.",
            code="incomplete_llm_response",
            http_status=502,
        )
    if response.status != "completed":
        raise LlmGatewayError(
            "The LLM did not complete its response.",
            code="failed_llm_response",
            http_status=502,
        )
    for item in response.output:
        if item.type == "message" and any(part.type == "refusal" for part in item.content):
            raise LlmGatewayError(
                "The LLM declined to generate this response.",
                code="llm_refusal",
                http_status=502,
            )
    text = response.output_text.strip()
    if not text:
        raise LlmGatewayError(
            "The LLM returned no answer text.", code="empty_llm_response", http_status=502
        )
    return text


def _translate_api_error(error: openai.APIError) -> LlmGatewayError:
    """Map upstream failures without including their potentially sensitive body."""

    if isinstance(error, openai.AuthenticationError):
        return LlmGatewayError("OpenAI rejected the API key.", code="invalid_api_key")
    if isinstance(error, openai.PermissionDeniedError):
        return LlmGatewayError(
            "This API project cannot access the requested model.", code="model_access_denied"
        )
    if isinstance(error, openai.NotFoundError):
        return LlmGatewayError(
            "The requested API model is unavailable to this project.", code="model_unavailable"
        )
    if isinstance(error, openai.RateLimitError):
        return LlmGatewayError(
            "OpenAI quota or rate limit reached. Check API billing and limits.",
            code="upstream_rate_limit",
        )
    if isinstance(error, openai.APITimeoutError):
        return LlmGatewayError(
            "The OpenAI request timed out.", code="llm_timeout", http_status=504
        )
    if isinstance(error, openai.APIConnectionError):
        return LlmGatewayError(
            "Cannot connect to OpenAI.", code="llm_connection_error", http_status=502
        )
    return LlmGatewayError(
        "The OpenAI request failed. Check model access and API configuration.",
        code="upstream_api_error",
        http_status=502,
    )


def _record_call(
    *,
    settings: OpenAISettings,
    started_at: float,
    token_limit: int,
    response: Response | None = None,
    error_code: str | None = None,
) -> None:
    """Record operational metadata only: no prompts, passages, answers or secrets."""

    usage = response.usage if response is not None else None
    output_details = usage.output_tokens_details if usage is not None else None
    input_details = usage.input_tokens_details if usage is not None else None
    metrics = LlmCallMetrics(
        model_id=response.model if response is not None else settings.model_id,
        response_id=response.id if response is not None else None,
        status=response.status if response is not None else "error",
        error_code=error_code,
        duration_ms=round((perf_counter() - started_at) * 1000),
        max_output_tokens=token_limit,
        input_tokens=usage.input_tokens if usage is not None else None,
        output_tokens=usage.output_tokens if usage is not None else None,
        total_tokens=usage.total_tokens if usage is not None else None,
        reasoning_tokens=(
            output_details.reasoning_tokens if output_details is not None else None
        ),
        cached_input_tokens=(
            input_details.cached_tokens if input_details is not None else None
        ),
    )
    calls = _call_collector.get()
    if calls is not None:
        calls.append(metrics)
    logger.info(
        "LLM model=%s status=%s duration_ms=%s input_tokens=%s output_tokens=%s",
        metrics.model_id, metrics.status, metrics.duration_ms,
        metrics.input_tokens, metrics.output_tokens,
    )
