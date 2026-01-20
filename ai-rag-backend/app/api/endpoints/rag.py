# RAG API endpoints for natural language querying

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.models import Document
from app.rag import RAGService, VectorStoreService
from app.schemas.requests import RAGQueryRequest, RAGSearchRequest
from app.schemas.responses import (
    ChunkResult,
    RAGSearchResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/query/stream",
    description="Query documents using natural language with streaming response",
)
async def query_documents_stream(request: RAGQueryRequest) -> StreamingResponse:
    """
    Query the document collection using natural language with streaming response.

    The system will:
    1. Find relevant document chunks
    2. Use an LLM to generate an answer based on the context
    3. Stream the answer tokens as they are generated
    4. Return sources at the end of the stream

    Response format (newline-delimited JSON):
    - {"type": "token", "content": "..."} - Answer tokens
    - {"type": "sources", "sources": [...]} - Source documents
    - {"type": "done"} - Stream complete
    """
    async def generate():
        try:
            rag_service = RAGService()
            async for chunk in rag_service.query_stream(
                question=request.question,
                k=request.k,
                document_id=request.document_id,
            ):
                yield chunk
        except Exception as e:
            logger.exception("RAG streaming query failed: %s", str(e))
            import json
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
    )


@router.post(
    "/search",
    response_model=RAGSearchResponse,
    description="Search for relevant document chunks without generating an answer",
)
async def search_documents(request: RAGSearchRequest) -> RAGSearchResponse:
    """
    Search for relevant document chunks using similarity search.

    Returns the most relevant chunks without LLM generation.
    Useful for exploring what information is available.
    """
    try:
        rag_service = RAGService()
        chunks = rag_service.get_relevant_chunks(
            question=request.query,
            k=request.k,
            document_id=request.document_id,
        )

        return RAGSearchResponse(
            chunks=[
                ChunkResult(
                    content=str(c["content"]),
                    document_id=str(c["document_id"]),
                    filename=str(c["filename"]),
                    chunk_index=int(c["chunk_index"]),
                    similarity_score=float(c["similarity_score"]),
                )
                for c in chunks
            ]
        )

    except Exception as e:
        logger.exception("Search failed: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        ) from e
