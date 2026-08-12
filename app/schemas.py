"""Data structures exchanged between the API, models, and Qdrant."""

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


class AskResponse(BaseModel):
    """A generated answer together with the evidence it is allowed to cite."""

    question: str
    answer: str
    citations: list[Citation]
    evidence: list[RetrievedEvidence]
    retrieved_chunks: int
    abstained: bool
    reason: str | None = None


class LlmStatus(BaseModel):
    """Availability information without loading the large model into memory."""

    model_id: str
    model_path: str
    device: str
    downloaded: bool


class IndexedDocument(BaseModel):
    """One distinct article currently represented in an indexed collection."""

    document_id: str
    title: str
    indexed_chunks: int


class IndexedDocumentResponse(BaseModel):
    """The articles available for filtering a corpus question."""

    collection: str
    documents: list[IndexedDocument]
