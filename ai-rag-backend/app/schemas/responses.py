from datetime import datetime

from pydantic import BaseModel, ConfigDict


class BaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# RAG Responses
class DocumentSource(BaseResponse):
    """Source document information."""

    document_id: str
    filename: str
    chunk_index: str

class ChunkResult(BaseResponse):
    """A single chunk result from similarity search."""

    content: str
    document_id: str
    filename: str
    chunk_index: int
    similarity_score: float


class RAGSearchResponse(BaseResponse):
    """Response from RAG similarity search."""

    chunks: list[ChunkResult]
