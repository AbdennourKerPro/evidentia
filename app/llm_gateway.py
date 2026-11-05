"""Single OpenAI Responses API boundary for every generative LLM call."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

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


@dataclass
class ToolCallRecord:
    """One tool call requested by the model and what we answered."""

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    duration_ms: int


@dataclass
class ToolLoopResult:
    """Outcome of a bounded tool-calling loop."""

    calls: list[ToolCallRecord] = field(default_factory=list)
    final_call: ToolCallRecord | None = None
    stopped_reason: Literal["finished", "budget", "no_tool_call"] = "no_tool_call"
    model_calls: int = 0


def run_tool_loop(
    *,
    system_message: str,
    user_message: str,
    tools: list[dict[str, Any]],
    execute_tool: Callable[[str, dict[str, Any]], dict[str, Any]],
    final_tool_name: str,
    max_model_calls: int,
    max_output_tokens: int = 2048,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> ToolLoopResult:
    """Let the model call ``tools`` until it calls ``final_tool_name`` or the budget ends.

    Every model call goes through the same client, settings, error translation and
    metrics as ``generate_chat``. Responses are stored client-side (``store=False``),
    so the full item history is replayed on each turn. Tool failures must be returned
    by ``execute_tool`` as ``{"ok": False, ...}`` so the model can adapt; only gateway
    errors raise.
    """

    if max_model_calls < 1:
        raise ValueError("max_model_calls must be at least 1.")
    settings = _read_settings()
    client = _get_client(settings)
    token_limit = min(max_output_tokens, settings.max_output_tokens)
    result = ToolLoopResult()
    items: list[Any] = [{"role": "user", "content": user_message}]
    emit = on_event or (lambda kind, payload: None)

    for _ in range(max_model_calls):
        started_at = perf_counter()
        try:
            response = client.responses.create(
                model=settings.model_id,
                instructions=system_message,
                input=items,
                tools=tools,
                tool_choice="required",
                parallel_tool_calls=False,
                reasoning={"effort": settings.reasoning_effort},
                max_output_tokens=token_limit,
                store=False,
            )
        except openai.APIError as error:
            safe_error = _translate_api_error(error)
            _record_call(
                settings=settings, started_at=started_at, token_limit=token_limit,
                error_code=safe_error.code,
            )
            raise safe_error from None
        _record_call(
            settings=settings, started_at=started_at, token_limit=token_limit,
            response=response,
        )
        result.model_calls += 1
        if response.status != "completed":
            raise LlmGatewayError(
                "The tool-calling response was incomplete.",
                code="incomplete_llm_response", http_status=502,
            )
        function_calls = [item for item in response.output if item.type == "function_call"]
        if not function_calls:
            result.stopped_reason = "no_tool_call"
            return result
        items.extend(response.output)
        for call in function_calls:
            tool_started = perf_counter()
            try:
                arguments = json.loads(call.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be an object")
            except ValueError:
                arguments = {}
                output: dict[str, Any] = {
                    "ok": False, "error": "invalid_arguments",
                    "message": "The arguments were not a valid JSON object.",
                }
            else:
                emit("tool_call", {"name": call.name, "arguments": arguments})
                output = execute_tool(call.name, arguments)
            record = ToolCallRecord(
                name=call.name, arguments=arguments, result=output,
                duration_ms=round((perf_counter() - tool_started) * 1000),
            )
            result.calls.append(record)
            emit("tool_result", {"name": call.name, "ok": bool(output.get("ok"))})
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(output, ensure_ascii=False),
                }
            )
            if call.name == final_tool_name and output.get("ok"):
                result.final_call = record
                result.stopped_reason = "finished"
                return result
    result.stopped_reason = "budget"
    return result


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
