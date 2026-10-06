"""HTTP entry point for the Evidentia prototype."""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
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


app = FastAPI(
    title="Evidentia",
    version="0.1.0",
    description="Multimodal evidence exploration prototype.",
)


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
        documents=list_indexed_documents(ARXIV_COLLECTION_NAME),
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
            document_ids=request.document_ids,
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
