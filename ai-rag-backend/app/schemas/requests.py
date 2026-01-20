from pydantic import BaseModel, Field


class BaseRequest(BaseModel):
    # may define additional fields or config shared across requests
    pass


# RAG Requests
class RAGQueryRequest(BaseRequest):
    """Request to query the RAG system with natural language."""

    question: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=20, description="Number of chunks to retrieve")
    document_id: str | None = Field(
        default=None, description="Optional: limit search to a specific document"
    )


class RAGSearchRequest(BaseRequest):
    """Request to search for relevant chunks without generating an answer."""

    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)
    document_id: str | None = None
