"""HTTP entry point for the Evidentia prototype."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.benchmark_review import (
    ReviewCaseResponse,
    ReviewCollectionResponse,
    ReviewUpdate,
    list_review_cases,
    update_review_case,
)
from app.demo_corpus import DEMO_CHUNKS
from app.document_ingestion import UnknownSourceError, chunk_source, convert_source
from app.embeddings import EMBEDDING_MODEL_ID, embed_passages, embed_query
from app.llm_gateway import (
    LlmGatewayError,
    capture_llm_calls,
    summarize_llm_calls,
)
from app.hybrid_retrieval import clear_bm25_cache
from app.qdrant_gateway import qdrant_is_available
from app.qdrant_gateway import (
    ARXIV_COLLECTION_NAME,
    DEMO_COLLECTION_NAME,
    index_chunks,
    list_indexed_documents,
    search_chunks,
)
from app import access_control, conversation_service, paper_ingestion, paper_registry
from app.agentic_rag import DOCUMENT_CATALOG
from pydantic import BaseModel, Field
from app.schemas import (
    AgenticAskResponse,
    AskRequest,
    AskResponse,
    ConversionReport,
    DocumentIndexingReport,
    IndexingReport,
    IndexedDocumentResponse,
    LlmStatus,
    SearchResponse,
)
from app.agentic_rag import answer_agentic_question
from app.settings import get_openai_settings
from app.rag_service import answer_question


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Prepare the paper registry once; Qdrant may start later than the API."""

    try:
        conversation_service.startup()
    except Exception:
        import logging

        logging.getLogger(__name__).exception("Conversation registry startup failed")
    yield


app = FastAPI(
    title="Evidentia",
    version="0.1.0",
    description="Multimodal evidence exploration prototype.",
    lifespan=lifespan,
)

# The classic pipeline and the benchmark only see these historical papers.
CLASSIC_DOCUMENT_IDS = list(DOCUMENT_CATALOG)


@app.exception_handler(LlmGatewayError)
async def handle_llm_error(request: Request, error: LlmGatewayError) -> JSONResponse:
    """Expose a sanitized API failure, never an upstream body or a secret."""

    return JSONResponse(
        status_code=error.http_status,
        content={"detail": str(error), "code": error.code},
    )

# The lightweight local interface is served by the same container as the API.
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
REVIEW_STATIC_DIR = Path(__file__).parent / "review_static"
app.mount(
    "/review",
    StaticFiles(directory=REVIEW_STATIC_DIR, html=True),
    name="review",
)


@app.middleware("http")
async def disable_ui_cache(request: Request, call_next):
    """Prevent stale HTML, CSS, and JavaScript during local UI development."""

    response = await call_next(request)
    if request.url.path.startswith(("/ui", "/review")):
        response.headers["Cache-Control"] = "no-store"
    return response


# With ACCESS_MODE=invite only these routes exist. Everything else (the classic /ask API,
# the benchmark review UI, corpus conversion, API docs) answers 404: an invited tester
# must never be able to spend the API budget or write files through them.
INVITE_MODE_PREFIXES = ("/health", "/ui", "/access", "/conversations/")


@app.middleware("http")
async def restrict_routes_in_invite_mode(request: Request, call_next):
    if access_control.access_mode() == "invite":
        path = request.url.path
        if path not in ("/", "/favicon.ico") and not path.startswith(INVITE_MODE_PREFIXES):
            return JSONResponse(status_code=404, content={"detail": "Not found"})
    return await call_next(request)


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a small response proving that the API is running."""

    return {
        "service": "evidentia",
        "status": "running",
        "message": "The Evidentia API is running.",
    }


@app.get("/health")
def liveness_check() -> dict[str, str]:
    """Report whether the API process itself is alive."""

    return {"status": "ok"}


@app.get("/ready")
def readiness_check() -> dict[str, str]:
    """Report whether the API can reach its required vector database."""

    if not qdrant_is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant is unavailable.",
        )

    return {"status": "ready", "qdrant": "ok"}


@app.get("/llm/status", response_model=LlmStatus)
def llm_status() -> LlmStatus:
    """Check local configuration without any network request or billed tokens."""

    try:
        settings = get_openai_settings()
    except ValueError as error:
        raise LlmGatewayError(str(error), code="invalid_llm_configuration") from None
    return LlmStatus(
        model_id=settings.model_id,
        configured=bool(settings.api_key),
        reasoning_effort=settings.reasoning_effort,
        max_output_tokens=settings.max_output_tokens,
        configuration_error=None if settings.api_key else "OPENAI_API_KEY is missing.",
    )


@app.post("/demo/index", response_model=IndexingReport)
def index_demo_corpus() -> IndexingReport:
    """Embed the controlled demo corpus and upsert it into Qdrant."""

    vectors = embed_passages([chunk.text for chunk in DEMO_CHUNKS])
    indexed_chunks = index_chunks(
        DEMO_CHUNKS,
        vectors,
        collection_name=DEMO_COLLECTION_NAME,
    )

    return IndexingReport(
        collection=DEMO_COLLECTION_NAME,
        indexed_chunks=indexed_chunks,
        embedding_model=EMBEDDING_MODEL_ID,
    )


@app.post("/documents/{source_id}/convert", response_model=ConversionReport)
def convert_document(source_id: str) -> ConversionReport:
    """Convert a downloaded arXiv PDF into Markdown and Docling JSON artifacts."""

    try:
        return convert_source(source_id)
    except (FileNotFoundError, UnknownSourceError) as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


@app.post("/documents/{source_id}/index", response_model=DocumentIndexingReport)
def index_document(source_id: str) -> DocumentIndexingReport:
    """Chunk, embed, and store one converted scientific paper in Qdrant."""

    try:
        chunks, chunks_path = chunk_source(source_id)
    except (FileNotFoundError, UnknownSourceError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error

    vectors = embed_passages([chunk.text for chunk in chunks])
    indexed_chunks = index_chunks(
        chunks,
        vectors,
        collection_name=ARXIV_COLLECTION_NAME,
    )
    clear_bm25_cache()

    return DocumentIndexingReport(
        source_id=source_id,
        collection=ARXIV_COLLECTION_NAME,
        indexed_chunks=indexed_chunks,
        embedding_model=EMBEDDING_MODEL_ID,
        chunks_file=f"data/processed/{source_id}/{chunks_path.name}",
    )


@app.get("/search", response_model=SearchResponse)
def semantic_search(
    query: str = Query(min_length=3, max_length=500),
    limit: int = Query(default=3, ge=1, le=10),
) -> SearchResponse:
    """Retrieve evidence chunks semantically related to a user question."""

    results = search_chunks(
        embed_query(query),
        limit=limit,
        collection_name=DEMO_COLLECTION_NAME,
    )

    return SearchResponse(
        query=query,
        collection=DEMO_COLLECTION_NAME,
        results=results,
    )


@app.get("/arxiv/search", response_model=SearchResponse)
def search_arxiv(
    query: str = Query(min_length=3, max_length=500),
    limit: int = Query(default=3, ge=1, le=10),
) -> SearchResponse:
    """Retrieve evidence from the arXiv-paper collection only."""

    results = search_chunks(
        embed_query(query),
        limit=limit,
        collection_name=ARXIV_COLLECTION_NAME,
        document_ids=CLASSIC_DOCUMENT_IDS,
    )

    return SearchResponse(
        query=query,
        collection=ARXIV_COLLECTION_NAME,
        results=results,
    )


@app.get("/arxiv/documents", response_model=IndexedDocumentResponse)
def indexed_arxiv_documents() -> IndexedDocumentResponse:
    """List articles that can be selected as a scope in the local UI."""

    return IndexedDocumentResponse(
        collection=ARXIV_COLLECTION_NAME,
        documents=[
            document for document in list_indexed_documents(ARXIV_COLLECTION_NAME)
            if document.document_id in CLASSIC_DOCUMENT_IDS
        ],
    )


@app.get("/evaluation/cases", response_model=ReviewCollectionResponse)
def evaluation_cases() -> ReviewCollectionResponse:
    """Return the human-review queue enriched with its source chunks."""

    return list_review_cases()


@app.patch(
    "/evaluation/cases/{case_id}",
    response_model=ReviewCaseResponse,
)
def review_evaluation_case(case_id: str, update: ReviewUpdate) -> ReviewCaseResponse:
    """Persist corrections and the review status of one benchmark entry."""

    try:
        return update_review_case(case_id, update)
    except KeyError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown evaluation case: {case_id}",
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """Generate an evidence-grounded answer from the indexed arXiv collection."""

    with capture_llm_calls() as calls:
        response = answer_question(
            question=request.question,
            limit=request.limit,
            document_ids=request.document_ids or CLASSIC_DOCUMENT_IDS,
        )
    response.llm = summarize_llm_calls(calls)
    return response


@app.post("/agentic/ask", response_model=AgenticAskResponse)
def ask_agentic_question(request: AskRequest) -> AgenticAskResponse:
    """Run the observable LangGraph RAG workflow on the arXiv collection."""

    with capture_llm_calls() as calls:
        response = answer_agentic_question(
            question=request.question,
            limit=request.limit,
            document_ids=request.document_ids,
        )
    response.llm = summarize_llm_calls(calls)
    return response


# ----------------------------------------------------------------- conversations
# Tool-calling agent mode: every route is scoped to one conversation (a browser UUID).
# In invite mode a session cookie is required and conversations belong to one invitation.


class ConversationAskRequest(BaseModel):
    """A question asked inside a conversation that owns its own papers."""

    question: str = Field(min_length=3, max_length=500)
    limit: int = Field(default=5, ge=1, le=5)
    pipeline: Literal["agentic", "baseline"] = "agentic"
    discovery: Literal["auto", "off"] = "auto"
    document_ids: list[str] | None = Field(default=None, max_length=20)


class AddPaperRequest(BaseModel):
    arxiv_id: str = Field(min_length=9, max_length=20)


class RedeemRequest(BaseModel):
    code: str = Field(min_length=8, max_length=128)


class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)
    question: str | None = Field(default=None, max_length=500)
    answer: str | None = Field(default=None, max_length=6000)


QUOTA_MESSAGES = {
    "quota_exhausted": "Votre quota de questions est épuisé. Merci pour votre test !",
    "attempts_exhausted": "Trop de tentatives sur cette invitation. Merci pour votre test !",
    "expired": "Cette invitation a expiré.",
    "revoked": "Cette invitation n'est plus valide.",
}


def _valid_conversation(conversation_id: str) -> str:
    if not paper_registry.is_valid_conversation_id(conversation_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The conversation id must be a UUID.",
        )
    return conversation_id


def _client_key(request: Request) -> str:
    """Rate-limit key: the real client address, honouring a tunnel's header only if trusted."""

    if access_control.trust_proxy_headers():
        forwarded = request.headers.get("cf-connecting-ip") or request.headers.get(
            "x-forwarded-for", ""
        ).split(",")[0].strip()
        if forwarded:
            return forwarded
    return request.client.host if request.client else "unknown"


def _session_invite(request: Request) -> access_control.Invite | None:
    token = request.cookies.get(access_control.SESSION_COOKIE)
    return access_control.invite_for_session(token) if token else None


def _authorize(request: Request, conversation_id: str) -> access_control.Invite | None:
    """Validate the id; in invite mode also require a session that owns the conversation."""

    _valid_conversation(conversation_id)
    if access_control.access_mode() != "invite":
        return None
    invite = _session_invite(request)
    if invite is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Accès sur invitation : ouvrez le lien reçu.")
    if not paper_registry.claim_conversation(conversation_id, invite.id):
        # 404, not 403: do not reveal that someone else's conversation exists.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation inconnue.")
    return invite


@app.get("/access/me")
def access_me(request: Request) -> dict:
    """Who am I? Always 200 so that the interface can decide what to show."""

    mode = access_control.access_mode()
    if mode == "open":
        return {"mode": "open", "authenticated": True}
    invite = _session_invite(request)
    return {
        "mode": "invite",
        "authenticated": invite is not None,
        "retention_days": conversation_service.get_dynamic_limits().conversation_ttl_days,
        **({"invite": invite.public()} if invite else {}),
    }


@app.post("/access/redeem")
def access_redeem(request: Request, body: RedeemRequest, response: Response) -> dict:
    """Exchange an invitation code (from the link's fragment) for a session cookie."""

    if access_control.access_mode() != "invite":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if not access_control.rate_limiter.allow(f"redeem:{_client_key(request)}", 8, 60):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Trop d'essais : patientez une minute.")
    redeemed = access_control.redeem(body.code)
    if redeemed is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Invitation invalide, expirée ou révoquée.")
    invite, token = redeemed
    secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    response.set_cookie(
        access_control.SESSION_COOKIE, token, httponly=True, samesite="lax", secure=secure,
        max_age=60 * 60 * 24 * 30, path="/",
    )
    return {"authenticated": True, "invite": invite.public()}


@app.post("/access/logout")
def access_logout(request: Request, response: Response) -> dict:
    token = request.cookies.get(access_control.SESSION_COOKIE)
    if token:
        access_control.end_session(token)
    response.delete_cookie(access_control.SESSION_COOKIE, path="/")
    return {"authenticated": False}


@app.put("/conversations/{conversation_id}")
def touch_conversation(conversation_id: str, request: Request) -> dict[str, str]:
    """Create the conversation or refresh its activity time (idempotent)."""

    _authorize(request, conversation_id)
    paper_registry.ensure_conversation(conversation_id)
    return {"id": conversation_id, "status": "ok"}


@app.get("/conversations/{conversation_id}/papers")
def conversation_papers(conversation_id: str, request: Request) -> dict:
    """Papers attached to this conversation, with their ingestion state."""

    _authorize(request, conversation_id)
    papers = paper_registry.list_conversation_papers(conversation_id)
    return {"conversation_id": conversation_id, "papers": [p.model_dump() for p in papers]}


@app.post("/conversations/{conversation_id}/papers")
def add_conversation_paper(conversation_id: str, request: Request, body: AddPaperRequest) -> dict:
    """Manually attach an arXiv paper (same ingestion as the agent; may take a minute)."""

    from app.arxiv_client import ArxivError, split_arxiv_id

    invite = _authorize(request, conversation_id)
    reserved = False
    if invite is not None:
        if not access_control.reserve_manual_paper(invite.id):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Limite d'ajouts manuels d'articles atteinte pour cette invitation.")
        reserved = True
    try:
        arxiv_id, version = split_arxiv_id(body.arxiv_id)
        paper = paper_ingestion.ingest_arxiv_paper(arxiv_id, version)
        attached = paper_registry.list_conversation_papers(conversation_id)
        limit = conversation_service.get_dynamic_limits().max_papers_per_conversation
        if paper.document_id not in {p.document_id for p in attached} and len(attached) >= limit:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The conversation is full.")
        paper_registry.attach_paper(conversation_id, paper.document_id, added_by="user")
        reserved = False  # the addition is consumed
    except (ArxivError, paper_ingestion.IngestionError) as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from None
    finally:
        if reserved and invite is not None:
            access_control.release_manual_paper(invite.id)
    return {"document_id": paper.document_id, "title": paper.title, "chunks": paper.chunk_count}


@app.delete("/conversations/{conversation_id}/papers/{document_id}")
def remove_conversation_paper(conversation_id: str, document_id: str, request: Request) -> dict:
    """Detach a paper; it is deleted if no other conversation needs it."""

    _authorize(request, conversation_id)
    if not conversation_service.detach_and_collect(conversation_id, document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paper not attached.")
    return {"removed": document_id}


@app.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: str, request: Request) -> dict:
    """Delete a conversation and every paper only it used."""

    _authorize(request, conversation_id)
    purged = conversation_service.delete_conversation_and_collect(conversation_id)
    return {"deleted": conversation_id, "purged_documents": purged}


@app.post("/conversations/{conversation_id}/feedback")
def conversation_feedback(conversation_id: str, request: Request, body: FeedbackRequest) -> dict:
    """Record a thumbs up/down with an optional comment (tester feedback)."""

    invite = _authorize(request, conversation_id)
    if invite is not None and not access_control.rate_limiter.allow(f"feedback:{invite.id}", 30, 60):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Trop de retours d'affilée.")
    access_control.add_feedback(
        invite_id=invite.id if invite else None, conversation_id=conversation_id,
        rating=body.rating, comment=body.comment, question=body.question, answer=body.answer,
    )
    return {"status": "recorded"}


@app.post("/conversations/{conversation_id}/ask")
def ask_in_conversation(
    conversation_id: str, request: Request, body: ConversationAskRequest
) -> StreamingResponse:
    """Stream progress events, then the answer, as Server-Sent Events."""

    invite = _authorize(request, conversation_id)
    settle = None
    if invite is not None:
        if not access_control.rate_limiter.allow(f"ask:{invite.id}", 6, 60):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Trop de questions d'affilée : patientez une minute.")
        refusal = access_control.reserve_question(invite.id)
        if refusal is not None:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail=QUOTA_MESSAGES[refusal])

        def settle(answered: bool, invite_id: str = invite.id) -> dict | None:
            settled = access_control.settle_question(invite_id, answered)
            return settled.public() if settled else None

    return StreamingResponse(
        conversation_service.ask_stream(
            conversation_id=conversation_id,
            question=body.question,
            limit=body.limit,
            pipeline=body.pipeline,
            discovery=body.discovery,
            document_ids=body.document_ids,
            settle=settle,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
