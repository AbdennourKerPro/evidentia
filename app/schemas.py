"""Data structures exchanged between the API, models, and Qdrant."""

from typing import Literal

from pydantic import BaseModel, Field


class EvidenceChunk(BaseModel):
    """A retrievable passage together with its provenance metadata."""

    id: int
    document_id: str
    title: str
    page: int
    section: str
    language: str
    text: str


class IndexingReport(BaseModel):
    """Result returned after a corpus was embedded and stored."""

    collection: str
    indexed_chunks: int
    embedding_model: str


class DocumentIndexingReport(IndexingReport):
    """Result returned after one converted scientific document is indexed."""

    source_id: str
    chunks_file: str


class ConversionReport(BaseModel):
    """Artifacts produced when a source PDF is converted by Docling."""

    source_id: str
    source_pdf: str
    markdown_file: str
    docling_json_file: str


class SearchResult(BaseModel):
    """A passage retrieved from Qdrant for a user query."""

    chunk_id: int
    score: float
    document_id: str
    title: str
    page: int
    section: str
    language: str
    text: str


class SearchResponse(BaseModel):
    """The query and the evidence passages returned for it."""

    query: str
    collection: str
    results: list[SearchResult]


class AskRequest(BaseModel):
    """One user question passed through the evidence-grounded RAG pipeline."""

    question: str = Field(min_length=3, max_length=500)
    limit: int = Field(default=3, ge=1, le=5)
    document_ids: list[str] | None = Field(default=None, max_length=20)


class Citation(BaseModel):
    """A server-resolved source reference used by a generated answer."""

    reference: str
    document_id: str
    title: str
    page: int
    section: str
    score: float


class RetrievedEvidence(BaseModel):
    """One complete retrieved chunk, whether or not the LLM cited it."""

    reference: str
    document_id: str
    title: str
    page: int
    section: str
    score: float
    text: str


class PlannedFact(BaseModel):
    """One evidence-derived fact that the final answer must state and cite."""

    id: str
    claim: str
    anchor: str
    alternatives: list[str] = Field(default_factory=list)
    references: list[str]


class LlmCallMetrics(BaseModel):
    """Metadata for one Responses call; SDK-internal retries are not separate calls."""

    model_id: str
    response_id: str | None = None
    status: str
    error_code: str | None = None
    duration_ms: int
    max_output_tokens: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None


class LlmUsage(BaseModel):
    """Per-question usage, including planning, generation and any correction."""

    provider: Literal["openai"] = "openai"
    model_id: str
    reasoning_effort: str
    call_count: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    reasoning_tokens: int | None
    cached_input_tokens: int | None
    calls: list[LlmCallMetrics] = Field(default_factory=list)


class AskResponse(BaseModel):
    """A generated answer together with the evidence it is allowed to cite."""

    question: str
    answer: str
    citations: list[Citation]
    evidence: list[RetrievedEvidence]
    retrieved_chunks: int
    abstained: bool
    reason: str | None = None
    llm: LlmUsage | None = None


class AgenticQueryPlan(BaseModel):
    """One retrieval operation planned by the LangGraph workflow."""

    document_id: str | None = None
    query: str
    query_variants: list[str] = Field(default_factory=list)
    retrieval_intents: list[str] = Field(default_factory=list)
    retrieved_chunks: int = 0


class AgenticTraceStep(BaseModel):
    """One observable node execution in the LangGraph workflow."""

    node: str
    detail: str
    duration_ms: int


class RetrievalAudit(BaseModel):
    """Chunk identities before and after ranking, without benchmark knowledge."""

    document_id: str | None
    phase: Literal["initial", "supplementary"] = "initial"
    primary_query: str
    primary_dense_ids: list[int] = Field(default_factory=list)
    primary_lexical_ids: list[int] = Field(default_factory=list)
    candidate_ids: list[int] = Field(default_factory=list)
    ranked_ids: list[int] = Field(default_factory=list)


class CoverageRequirement(BaseModel):
    """One question aspect and its exact supporting excerpt, if present."""

    aspect: str
    document_id: str | None = None
    reference: str | None = None
    quote: str | None = None
    search_query: str | None = None


class EvidenceCoverage(BaseModel):
    """A fallible LLM assessment, not a guarantee of factual completeness."""

    sufficient: bool | None
    requirements: list[CoverageRequirement] = Field(default_factory=list)
    diagnostic: str | None = None


class CandidateRecoveryAudit(BaseModel):
    """Bounded candidate-only recovery; original evidence is never replaced."""

    attempted: bool = False
    considered_chunk_ids: list[int] = Field(default_factory=list)
    added_chunk_ids: list[int] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    diagnostic: str | None = None


class AgenticExecution(BaseModel):
    """Readable execution metadata returned with an agentic RAG answer."""

    strategy: Literal["global", "per_source"]
    retrieval_profile: str = "intent_baseline"
    retrieval_audit: list[RetrievalAudit] = Field(default_factory=list)
    coverage_checks: list[EvidenceCoverage] = Field(default_factory=list)
    candidate_recovery: CandidateRecoveryAudit | None = None
    supplementary_search_attempted: bool = False
    expected_language: Literal["en", "fr", "zh"]
    language_status: Literal["match", "mismatch", "uncertain"] | None = None
    language_check_method: str = "heuristic"
    detected_document_ids: list[str]
    planned_queries: list[AgenticQueryPlan]
    candidate_chunks: int
    selected_chunks: int
    fact_plan: list[PlannedFact] = Field(default_factory=list)
    fact_planning_fallback: bool = False
    covered_planned_facts: int = 0
    correction_attempted: bool
    response_contract_valid: bool
    response_publishable: bool = True
    fact_contract_valid: bool = True
    contract_issues: list[str] = Field(default_factory=list)
    steps: list[AgenticTraceStep]


class AgenticAskResponse(AskResponse):
    """A grounded answer enriched with the LangGraph execution trace."""

    execution: AgenticExecution


class LlmStatus(BaseModel):
    """Local configuration only: this route never verifies remote model access."""

    model_id: str
    provider: Literal["openai"] = "openai"
    endpoint: Literal["responses"] = "responses"
    configured: bool
    access_verified: bool = False
    reasoning_effort: str
    max_output_tokens: int
    configuration_error: str | None = None


class IndexedDocument(BaseModel):
    """One distinct article currently represented in an indexed collection."""

    document_id: str
    title: str
    indexed_chunks: int


class IndexedDocumentResponse(BaseModel):
    """The articles available for filtering a corpus question."""

    collection: str
    documents: list[IndexedDocument]
