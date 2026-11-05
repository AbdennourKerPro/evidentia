"""Conversation-scoped question answering with live progress (Server-Sent Events)."""

from __future__ import annotations

import contextvars
import json
import queue
import threading
from collections.abc import Iterator
from typing import Any, Callable, Literal

from app import access_control, paper_ingestion, paper_registry
from app.agentic_rag import CatalogEntry, DOCUMENT_CATALOG, answer_agentic_question
from app.llm_gateway import LlmGatewayError, capture_llm_calls, summarize_llm_calls
from app.paper_registry import ConversationPaper
from app.qdrant_gateway import ARXIV_COLLECTION_NAME, count_document_chunks
from app.rag_service import answer_question
from app.schemas import AgenticAskResponse
from app.settings import get_dynamic_limits
from app.question_rewriting import rewrite_follow_up
from app.source_discovery import run_discovery

import logging
from pathlib import Path


logger = logging.getLogger(__name__)
KEEPALIVE_SECONDS = 15
SOURCES_PATH = Path("/data/sources/arxiv_papers.json")


def startup() -> None:
    """Prepare the registry, register the historical papers, drop expired chats."""

    paper_registry.init_db()
    sources = SOURCES_PATH if SOURCES_PATH.is_file() else Path("data/sources/arxiv_papers.json")
    if sources.is_file():
        paper_registry.seed_legacy_papers(catalog=DOCUMENT_CATALOG, sources_path=sources)
    for record in paper_registry.list_pinned_papers():
        if record.chunk_count == 0:
            try:
                count = count_document_chunks(ARXIV_COLLECTION_NAME, record.document_id)
            except Exception:
                continue  # Qdrant not up yet; counts are cosmetic
            if count:
                paper_registry.set_paper_status(record.document_id, "ready", chunk_count=count)
    limits = get_dynamic_limits()
    for conversation_id in paper_registry.stale_conversations(limits.conversation_ttl_days):
        paper_registry.delete_conversation(conversation_id)
    try:
        paper_ingestion.collect_garbage()
    except Exception:
        logger.exception("Garbage collection failed")


def build_catalog(papers: list[ConversationPaper]) -> dict[str, CatalogEntry]:
    """Name-detection catalogue of the papers a conversation may search."""

    catalog: dict[str, CatalogEntry] = {}
    for paper in papers:
        record = paper_registry.get_paper(paper.document_id)
        catalog[paper.document_id] = {
            "title": paper.title,
            "aliases": tuple(record.aliases) if record else (),
        }
    return catalog


def detach_and_collect(conversation_id: str, document_id: str) -> bool:
    """Detach a paper; purge it when no conversation uses it any more."""

    removed = paper_registry.detach_paper(conversation_id, document_id)
    if removed and document_id in paper_registry.orphan_documents(0):
        paper_ingestion.purge_document(document_id)
    return removed


def delete_conversation_and_collect(conversation_id: str) -> list[str]:
    """Delete a conversation; return the document ids that were purged."""

    documents = [p.document_id for p in paper_registry.list_conversation_papers(conversation_id)]
    paper_registry.delete_conversation(conversation_id)
    orphans = set(paper_registry.orphan_documents(0))
    return [d for d in documents if d in orphans and paper_ingestion.purge_document(d)]


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def answer_for_conversation(
    *,
    conversation_id: str,
    question: str,
    limit: int,
    pipeline: Literal["agentic", "baseline"],
    discovery: Literal["auto", "off"],
    document_ids: list[str] | None,
    emit,
) -> AgenticAskResponse:
    """Discovery (optional) followed by the usual RAG pipeline on the conversation's papers."""

    paper_registry.record_turn(conversation_id, question)
    # The last four recorded questions end with the current one.
    previous = paper_registry.recent_questions(conversation_id, 4)[:-1]
    original_question = question
    with capture_llm_calls() as calls:
        if previous:
            titles = [p.title for p in paper_registry.list_conversation_papers(conversation_id)]
            question = rewrite_follow_up(original_question, previous, titles)
            if question != original_question:
                emit("question.rewritten", {"question": question})
        audit = (
            run_discovery(conversation_id, question, emit, recorded_question=original_question)
            if discovery == "auto" else None
        )
        papers = paper_registry.list_conversation_papers(conversation_id, only_ready=True)
        corpus = [p.document_id for p in papers]
        selection = [d for d in (document_ids or []) if d in corpus] or None
        emit("answer.started", {"papers": len(corpus)})
        if pipeline == "agentic" or not corpus:
            response = answer_agentic_question(
                question=question,
                limit=limit,
                document_ids=selection,
                corpus_document_ids=corpus,
                catalog=build_catalog(papers),
                focus_document_ids=audit.focus_document_ids if audit else [],
                check_formulas=True,
                partial_answers=True,
            )
        else:
            baseline = answer_question(
                question=question, limit=limit, document_ids=selection or corpus
            )
            response = AgenticAskResponse.model_validate(
                {**baseline.model_dump(), "execution": _baseline_execution(question)}
            )
        response.llm = summarize_llm_calls(calls)
    if question != original_question:
        response.execution.rewritten_question = question
    if audit is not None:
        audit.conversation_document_ids = corpus
        response.execution.discovery = audit
        if not corpus:
            searched = any(call.name == "search_arxiv" for call in audit.tool_calls)
            failures = [call.error for call in audit.tool_calls if call.error]
            response.reason = (
                "The agent could not attach any paper "
                f"({', '.join(dict.fromkeys(failures))})." if failures
                else "The agent searched arXiv but found no suitable paper." if searched
                else "The agent did not select any paper for this question."
            )
    return response


def _baseline_execution(question: str) -> dict:
    from app.response_contract import detect_question_language
    from app.retrieval_policy import get_retrieval_profile

    return {
        "strategy": "global", "retrieval_profile": get_retrieval_profile(),
        "expected_language": detect_question_language(question),
        "detected_document_ids": [], "planned_queries": [], "candidate_chunks": 0,
        "selected_chunks": 0, "correction_attempted": False,
        "response_contract_valid": True, "steps": [],
    }


def ask_stream(
    *,
    conversation_id: str,
    question: str,
    limit: int,
    pipeline: Literal["agentic", "baseline"],
    discovery: Literal["auto", "off"],
    document_ids: list[str] | None = None,
    settle: Callable[[bool], Any] | None = None,
) -> Iterator[str]:
    """Start the worker now and return the SSE frames it produces.

    ``settle(answered)`` is called exactly once when the worker ends (quota accounting)
    and may return a dict, emitted as a ``quota`` event. The worker is started eagerly so
    that a client disconnecting early can never leave a reservation unsettled. If the
    client leaves, the worker still finishes: an ingested paper is never wasted.
    """

    events: queue.Queue[tuple[str, Any] | None] = queue.Queue()

    def emit(event: str, data: dict[str, Any]) -> None:
        events.put((event, data))

    def work() -> None:
        answered = False
        gate = access_control.question_gate()
        acquired = False
        try:
            if gate is not None:
                acquired = gate.acquire(lambda: emit("queue.waiting", {}))
                if not acquired:
                    events.put(("error", {"detail": "The server is busy: please retry in a moment.", "code": "server_busy"}))
                    return
            response = answer_for_conversation(
                conversation_id=conversation_id, question=question, limit=limit,
                pipeline=pipeline, discovery=discovery, document_ids=document_ids, emit=emit,
            )
            answered = not response.abstained
            events.put(("answer", response.model_dump()))
        except LlmGatewayError as error:
            events.put(("error", {"detail": str(error), "code": error.code}))
        except Exception:
            logger.exception("Conversation question failed")
            events.put(("error", {"detail": "The question could not be processed.", "code": "internal_error"}))
        finally:
            if acquired and gate is not None:
                gate.release()
            if settle is not None:
                try:
                    quota = settle(answered)
                    if quota:
                        events.put(("quota", quota))
                except Exception:
                    logger.exception("Quota settlement failed")
            events.put(("done", {}))
            events.put(None)

    context = contextvars.copy_context()
    threading.Thread(target=context.run, args=(work,), daemon=True).start()
    return _drain(events)


def _drain(events: "queue.Queue[tuple[str, Any] | None]") -> Iterator[str]:
    while True:
        try:
            item = events.get(timeout=KEEPALIVE_SECONDS)
        except queue.Empty:
            yield ": ping\n\n"
            continue
        if item is None:
            return
        yield sse(*item)
