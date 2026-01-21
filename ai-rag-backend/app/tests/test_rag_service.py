# Tests for RAGService - handles RAG-based question answering with vector search

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document as LangchainDocument
from langchain_core.messages import AIMessageChunk
from openai import OpenAIError

from app.rag.rag_service import RAGService, VectorSearchError
from app.rag.vector_store import VectorStoreService


@pytest.fixture(autouse=True)
def reset_vector_store_singleton() -> None:
    """Reset VectorStoreService singleton before each test."""
    VectorStoreService._instance = None
    VectorStoreService._vector_store = None


@pytest.fixture
def mock_vector_store_service() -> MagicMock:
    """Create a mock VectorStoreService."""
    mock = MagicMock(spec=VectorStoreService)
    return mock


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock ChatOpenAI."""
    mock = MagicMock()
    return mock


@pytest.fixture
def rag_service(
    mock_vector_store_service: MagicMock, mock_llm: MagicMock
) -> RAGService:
    """Create a RAGService with mocked dependencies."""
    with (
        patch("app.rag.rag_service.get_settings") as mock_settings,
        patch("app.rag.rag_service.ChatOpenAI", return_value=mock_llm),
        patch("app.rag.rag_service.VectorStoreService", return_value=mock_vector_store_service),
    ):
        mock_settings.return_value.rag.chat_model = "gpt-4o-mini"
        mock_settings.return_value.rag.openai_api_key.get_secret_value.return_value = (
            "fake-key"
        )

        service = RAGService()
        service.vector_store_service = mock_vector_store_service
        service.llm = mock_llm

    return service


class TestQueryStream:
    """Tests for query_stream method."""

    @pytest.mark.asyncio
    async def test_streams_response_tokens(
        self, rag_service: RAGService, mock_vector_store_service: MagicMock
    ) -> None:
        """Should stream LLM response tokens."""
        # Setup mock retrieved documents
        mock_vector_store_service.similarity_search.return_value = [
            LangchainDocument(
                page_content="John has 5 years of Python experience",
                metadata={
                    "document_id": "doc-1",
                    "filename": "cv_john_doe.pdf",
                    "full_name": "John Doe",
                    "chunk_index": 0,
                },
            ),
        ]

        # Setup mock LLM streaming
        async def mock_stream(*args: object, **kwargs: object):
            yield AIMessageChunk(content="Based ")
            yield AIMessageChunk(content="on the ")
            yield AIMessageChunk(content="documents...")

        rag_service.llm.astream = mock_stream

        chunks = []
        async for chunk in rag_service.query_stream("What is John's experience?"):
            chunks.append(json.loads(chunk.strip()))

        # Should have token chunks, sources, and done
        token_chunks = [c for c in chunks if c.get("type") == "token"]
        assert len(token_chunks) == 3
        assert token_chunks[0]["content"] == "Based "
        assert token_chunks[1]["content"] == "on the "
        assert token_chunks[2]["content"] == "documents..."

    @pytest.mark.asyncio
    async def test_yields_sources_at_end(
        self, rag_service: RAGService, mock_vector_store_service: MagicMock
    ) -> None:
        """Should yield sources after streaming completes."""
        mock_vector_store_service.similarity_search.return_value = [
            LangchainDocument(
                page_content="Content",
                metadata={
                    "document_id": "doc-1",
                    "filename": "cv_john.pdf",
                    "full_name": "John Doe",
                    "chunk_index": 0,
                },
            ),
            LangchainDocument(
                page_content="More content",
                metadata={
                    "document_id": "doc-2",
                    "filename": "cv_jane.pdf",
                    "full_name": "Jane Smith",
                    "chunk_index": 1,
                },
            ),
        ]

        async def mock_stream(*args: object, **kwargs: object):
            yield AIMessageChunk(content="Answer")

        rag_service.llm.astream = mock_stream

        chunks = []
        async for chunk in rag_service.query_stream("Question"):
            chunks.append(json.loads(chunk.strip()))

        # Find sources chunk
        sources_chunk = next(c for c in chunks if c.get("type") == "sources")
        assert len(sources_chunk["sources"]) == 2
        assert sources_chunk["sources"][0]["document_id"] == "doc-1"
        assert sources_chunk["sources"][0]["full_name"] == "John Doe"
        assert sources_chunk["sources"][1]["document_id"] == "doc-2"
        assert sources_chunk["sources"][1]["full_name"] == "Jane Smith"

    @pytest.mark.asyncio
    async def test_yields_done_signal(
        self, rag_service: RAGService, mock_vector_store_service: MagicMock
    ) -> None:
        """Should yield done signal at the end."""
        mock_vector_store_service.similarity_search.return_value = []

        async def mock_stream(*args: object, **kwargs: object):
            yield AIMessageChunk(content="")

        rag_service.llm.astream = mock_stream

        chunks = []
        async for chunk in rag_service.query_stream("Question"):
            chunks.append(json.loads(chunk.strip()))

        assert chunks[-1] == {"type": "done"}

    @pytest.mark.asyncio
    async def test_deduplicates_sources(
        self, rag_service: RAGService, mock_vector_store_service: MagicMock
    ) -> None:
        """Should deduplicate sources from same document."""
        # Multiple chunks from same document
        mock_vector_store_service.similarity_search.return_value = [
            LangchainDocument(
                page_content="Chunk 1",
                metadata={
                    "document_id": "doc-1",
                    "filename": "cv_john.pdf",
                    "full_name": "John Doe",
                    "chunk_index": 0,
                },
            ),
            LangchainDocument(
                page_content="Chunk 2",
                metadata={
                    "document_id": "doc-1",
                    "filename": "cv_john.pdf",
                    "full_name": "John Doe",
                    "chunk_index": 1,
                },
            ),
        ]

        async def mock_stream(*args: object, **kwargs: object):
            yield AIMessageChunk(content="Answer")

        rag_service.llm.astream = mock_stream

        chunks = []
        async for chunk in rag_service.query_stream("Question"):
            chunks.append(json.loads(chunk.strip()))

        sources_chunk = next(c for c in chunks if c.get("type") == "sources")
        # Should only have one source despite two chunks from same doc
        assert len(sources_chunk["sources"]) == 1

    @pytest.mark.asyncio
    async def test_applies_document_filter(
        self, rag_service: RAGService, mock_vector_store_service: MagicMock
    ) -> None:
        """Should apply document_id filter when provided."""
        mock_vector_store_service.similarity_search.return_value = []

        async def mock_stream(*args: object, **kwargs: object):
            yield AIMessageChunk(content="")

        rag_service.llm.astream = mock_stream

        async for _ in rag_service.query_stream(
            "Question", document_id="specific-doc"
        ):
            pass

        mock_vector_store_service.similarity_search.assert_called_once_with(
            "Question", k=3, filter_dict={"document_id": "specific-doc"}
        )

    @pytest.mark.asyncio
    async def test_uses_custom_k_value(
        self, rag_service: RAGService, mock_vector_store_service: MagicMock
    ) -> None:
        """Should use custom k value for retrieval."""
        mock_vector_store_service.similarity_search.return_value = []

        async def mock_stream(*args: object, **kwargs: object):
            yield AIMessageChunk(content="")

        rag_service.llm.astream = mock_stream

        async for _ in rag_service.query_stream("Question", k=10):
            pass

        mock_vector_store_service.similarity_search.assert_called_once_with(
            "Question", k=10, filter_dict=None
        )

    @pytest.mark.asyncio
    async def test_emits_error_when_retrieval_fails(
        self, rag_service: RAGService, mock_vector_store_service: MagicMock
    ) -> None:
        """Should emit error chunk when retrieval raises."""
        mock_vector_store_service.similarity_search.side_effect = Exception(
            "vector store down"
        )

        chunks = []
        async for chunk in rag_service.query_stream("Question"):
            chunks.append(json.loads(chunk.strip()))

        assert chunks == [{"type": "error", "message": "Document retrieval failed"}]

    @pytest.mark.asyncio
    async def test_emits_error_on_openai_failure(
        self, rag_service: RAGService, mock_vector_store_service: MagicMock
    ) -> None:
        """Should emit error chunk when OpenAI streaming fails."""
        mock_vector_store_service.similarity_search.return_value = [
            LangchainDocument(
                page_content="Content",
                metadata={
                    "document_id": "doc-1",
                    "filename": "cv_john.pdf",
                    "full_name": "John Doe",
                    "chunk_index": 0,
                },
            ),
        ]

        async def mock_stream(*args: object, **kwargs: object):
            raise OpenAIError("boom")
            if False:
                yield  # pragma: no cover

        rag_service.llm.astream = mock_stream

        chunks = []
        async for chunk in rag_service.query_stream("Question"):
            chunks.append(json.loads(chunk.strip()))

        assert chunks == [
            {
                "type": "error",
                "message": "Failed to generate response. Please try again.",
            }
        ]

    @pytest.mark.asyncio
    async def test_emits_error_on_unexpected_stream_failure(
        self, rag_service: RAGService, mock_vector_store_service: MagicMock
    ) -> None:
        """Should emit error chunk when streaming fails unexpectedly."""
        mock_vector_store_service.similarity_search.return_value = [
            LangchainDocument(
                page_content="Content",
                metadata={
                    "document_id": "doc-1",
                    "filename": "cv_john.pdf",
                    "full_name": "John Doe",
                    "chunk_index": 0,
                },
            ),
        ]

        async def mock_stream(*args: object, **kwargs: object):
            raise RuntimeError("unexpected")
            if False:
                yield  # pragma: no cover

        rag_service.llm.astream = mock_stream

        chunks = []
        async for chunk in rag_service.query_stream("Question"):
            chunks.append(json.loads(chunk.strip()))

        assert chunks == [
            {
                "type": "error",
                "message": "An unexpected error occurred. Please try again.",
            }
        ]


class TestGetRelevantChunks:
    """Tests for get_relevant_chunks method."""

    def test_returns_chunks_with_scores(
        self, rag_service: RAGService, mock_vector_store_service: MagicMock
    ) -> None:
        """Should return chunks with similarity scores."""
        mock_vector_store_service.similarity_search_with_score.return_value = [
            (
                LangchainDocument(
                    page_content="Content about Python",
                    metadata={
                        "document_id": "doc-1",
                        "filename": "cv_john.pdf",
                        "full_name": "John Doe",
                        "chunk_index": 0,
                    },
                ),
                0.92,
            ),
            (
                LangchainDocument(
                    page_content="More Python content",
                    metadata={
                        "document_id": "doc-2",
                        "filename": "cv_jane.pdf",
                        "full_name": "Jane Smith",
                        "chunk_index": 2,
                    },
                ),
                0.85,
            ),
        ]

        result = rag_service.get_relevant_chunks("Python experience")

        assert len(result) == 2
        assert result[0]["content"] == "Content about Python"
        assert result[0]["document_id"] == "doc-1"
        assert result[0]["filename"] == "cv_john.pdf"
        assert result[0]["full_name"] == "John Doe"
        assert result[0]["chunk_index"] == 0
        assert result[0]["similarity_score"] == 0.92

        assert result[1]["similarity_score"] == 0.85

    def test_applies_document_filter(
        self, rag_service: RAGService, mock_vector_store_service: MagicMock
    ) -> None:
        """Should apply document_id filter."""
        mock_vector_store_service.similarity_search_with_score.return_value = []

        rag_service.get_relevant_chunks("query", document_id="doc-123")

        mock_vector_store_service.similarity_search_with_score.assert_called_once_with(
            "query", k=3, filter_dict={"document_id": "doc-123"}
        )

    def test_uses_custom_k_value(
        self, rag_service: RAGService, mock_vector_store_service: MagicMock
    ) -> None:
        """Should use custom k value."""
        mock_vector_store_service.similarity_search_with_score.return_value = []

        rag_service.get_relevant_chunks("query", k=7)

        mock_vector_store_service.similarity_search_with_score.assert_called_once_with(
            "query", k=7, filter_dict=None
        )

    def test_handles_missing_metadata(
        self, rag_service: RAGService, mock_vector_store_service: MagicMock
    ) -> None:
        """Should handle documents with missing metadata gracefully."""
        mock_vector_store_service.similarity_search_with_score.return_value = [
            (
                LangchainDocument(
                    page_content="Content",
                    metadata={},  # No metadata
                ),
                0.9,
            ),
        ]

        result = rag_service.get_relevant_chunks("query")

        assert len(result) == 1
        assert result[0]["content"] == "Content"
        assert result[0]["document_id"] == "unknown"
        assert result[0]["filename"] == "unknown"
        assert result[0]["full_name"] == "Unknown"
        assert result[0]["chunk_index"] == 0

    def test_returns_empty_list_for_no_results(
        self, rag_service: RAGService, mock_vector_store_service: MagicMock
    ) -> None:
        """Should return empty list when no results found."""
        mock_vector_store_service.similarity_search_with_score.return_value = []

        result = rag_service.get_relevant_chunks("obscure query")

        assert result == []

    def test_raises_vector_search_error_on_failure(
        self, rag_service: RAGService, mock_vector_store_service: MagicMock
    ) -> None:
        """Should raise VectorSearchError on retrieval failure."""
        mock_vector_store_service.similarity_search_with_score.side_effect = Exception(
            "boom"
        )

        with pytest.raises(VectorSearchError, match="Failed to retrieve relevant chunks"):
            rag_service.get_relevant_chunks("query")
