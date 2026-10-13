from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=1000)]


class QueryRequest(BaseModel):
    question: Question
    top_k: int = Field(default=4, ge=1, le=20, description="Number of chunks to retrieve.")
    include_chunks: bool = Field(default=True, description="Include the full retrieved chunk text.")


class RetrievedChunk(BaseModel):
    rank: int
    source_file: str
    chunk_id: int
    score: float = Field(description="Cosine similarity between the question and the chunk.")
    text: str | None = None


class EvidenceItem(BaseModel):
    source: str
    chunk_id: int
    score: float
    evidence: str


class Citation(BaseModel):
    """Where one answer statement sits in the retrieved text."""

    statement: str
    found: bool = Field(description="False if the statement could not be located word for word.")
    rank: int | None = Field(default=None, description="Rank of the retrieved chunk that contains it.")
    chunk_id: int | None = None
    source_file: str | None = None
    start: int | None = Field(default=None, description="Character offset in the chunk text.")
    end: int | None = None


class Answer(BaseModel):
    summary: str
    key_insights: list[str]
    citations: list[Citation] = Field(default_factory=list)
    risks_limitations: list[str]
    supporting_evidence: list[EvidenceItem]
    confidence_score: float


class QueryResponse(BaseModel):
    question: str
    answer: Answer
    answerable: bool = Field(description="False when confidence is below the threshold.")
    confidence_threshold: float
    retrieved: list[RetrievedChunk]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    index_ready: bool


class IndexInfo(BaseModel):
    embedding_model: str
    dimension: int
    index_type: str
    documents: int
    chunks: int
    chunk_size: int
    chunk_overlap: int
    rebuilt_on_startup: bool


class DocumentInfo(BaseModel):
    name: str
    origin: str = Field(description="'sample' for bundled documents, 'uploaded' for user uploads.")
    chunks: int
    size_bytes: int


class DocumentList(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


class UploadResponse(BaseModel):
    document: DocumentInfo
    index: IndexInfo
