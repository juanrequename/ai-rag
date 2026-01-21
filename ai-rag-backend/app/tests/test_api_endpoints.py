# Tests for API endpoints and dependencies to achieve 100% coverage

import json
from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient

from app.api.deps import get_session
from app.core.config import get_settings
from app.rag.vector_store import VectorStoreService


@pytest.fixture(autouse=True)
def reset_vector_store_singleton() -> None:
    """Reset VectorStoreService singleton before each test."""
    VectorStoreService._instance = None
    VectorStoreService._vector_store = None


class TestGetSession:
    """Tests for the get_session dependency."""

    @pytest.mark.asyncio
    async def test_get_session_yields_session(self, session) -> None:
        """Test that get_session yields a valid database session."""
        # The session fixture from conftest already handles this,
        # but we need to explicitly test the get_session function
        gen = get_session()
        db_session = await gen.__anext__()
        assert db_session is not None
        # Clean up generator
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass


class TestConfigPsycopgUri:
    """Tests for config computed properties."""

    def test_psycopg_database_uri_format(self) -> None:
        """Test psycopg_database_uri returns properly formatted connection string."""
        settings = get_settings()
        uri = settings.psycopg_database_uri

        # Should be a properly formatted psycopg connection string
        assert uri.startswith("postgresql+psycopg://")
        assert settings.database.hostname in uri
        assert str(settings.database.port) in uri
        assert settings.database.username in uri
        assert settings.database.db in uri


class TestQueryDocumentsStreamEndpoint:
    """Tests for POST /api/rag/query/stream endpoint."""

    @pytest.mark.asyncio
    async def test_query_stream_returns_streaming_response(
        self, client: AsyncClient
    ) -> None:
        """Test that query/stream endpoint returns a streaming response."""
        # Mock the RAGService to avoid real LLM calls
        async def mock_query_stream(self, question, k=3, document_id=None):
            yield json.dumps({"type": "token", "content": "Hello"}) + "\n"
            yield json.dumps({"type": "token", "content": " world"}) + "\n"
            yield json.dumps({"type": "sources", "sources": []}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

        with patch("app.api.endpoints.rag.RAGService") as MockRAGService:
            mock_instance = MagicMock()
            mock_instance.query_stream = lambda **kwargs: mock_query_stream(
                mock_instance, **kwargs
            )
            MockRAGService.return_value = mock_instance

            response = await client.post(
                "/rag/query/stream",
                json={"question": "What is Python?", "k": 3},
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/x-ndjson"

            # Parse streaming response
            lines = response.text.strip().split("\n")
            chunks = [json.loads(line) for line in lines if line]

            assert any(c.get("type") == "token" for c in chunks)
            assert any(c.get("type") == "done" for c in chunks)

    @pytest.mark.asyncio
    async def test_query_stream_handles_exception(self, client: AsyncClient) -> None:
        """Test that query/stream endpoint handles exceptions gracefully."""

        async def mock_query_stream_error(self, **kwargs):
            raise Exception("LLM service unavailable")
            yield  # Make it a generator

        with patch("app.api.endpoints.rag.RAGService") as MockRAGService:
            mock_instance = MagicMock()
            mock_instance.query_stream = lambda **kwargs: mock_query_stream_error(
                mock_instance, **kwargs
            )
            MockRAGService.return_value = mock_instance

            response = await client.post(
                "/rag/query/stream",
                json={"question": "What is Python?", "k": 3},
            )

            assert response.status_code == 200
            # Error should be streamed back
            chunks = [json.loads(line) for line in response.text.strip().split("\n") if line]
            error_chunk = next((c for c in chunks if c.get("type") == "error"), None)
            assert error_chunk is not None
            assert "LLM service unavailable" in error_chunk["message"]

    @pytest.mark.asyncio
    async def test_query_stream_with_document_filter(
        self, client: AsyncClient
    ) -> None:
        """Test query/stream endpoint with document_id filter."""

        async def mock_query_stream(self, question, k=3, document_id=None):
            yield json.dumps({"type": "token", "content": "Filtered response"}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

        with patch("app.api.endpoints.rag.RAGService") as MockRAGService:
            mock_instance = MagicMock()
            mock_instance.query_stream = lambda **kwargs: mock_query_stream(
                mock_instance, **kwargs
            )
            MockRAGService.return_value = mock_instance

            response = await client.post(
                "/rag/query/stream",
                json={
                    "question": "What skills does John have?",
                    "k": 5,
                    "document_id": "doc-123",
                },
            )

            assert response.status_code == 200


class TestSearchDocumentsEndpoint:
    """Tests for POST /api/rag/search endpoint."""

    @pytest.mark.asyncio
    async def test_search_returns_chunks(self, client: AsyncClient) -> None:
        """Test that search endpoint returns relevant chunks."""
        mock_chunks = [
            {
                "content": "John has 5 years of Python experience",
                "document_id": "doc-1",
                "filename": "cv_john_doe.pdf",
                "full_name": "John Doe",
                "chunk_index": 0,
                "similarity_score": 0.95,
            },
            {
                "content": "Jane specializes in machine learning",
                "document_id": "doc-2",
                "filename": "cv_jane_smith.pdf",
                "full_name": "Jane Smith",
                "chunk_index": 1,
                "similarity_score": 0.87,
            },
        ]

        with patch("app.api.endpoints.rag.RAGService") as MockRAGService:
            mock_instance = MagicMock()
            mock_instance.get_relevant_chunks = MagicMock(return_value=mock_chunks)
            MockRAGService.return_value = mock_instance

            response = await client.post(
                "/rag/search",
                json={"query": "Python experience", "k": 3},
            )

            assert response.status_code == 200
            data = response.json()
            assert "chunks" in data
            assert len(data["chunks"]) == 2
            assert data["chunks"][0]["content"] == "John has 5 years of Python experience"
            assert data["chunks"][0]["similarity_score"] == 0.95

    @pytest.mark.asyncio
    async def test_search_with_document_filter(self, client: AsyncClient) -> None:
        """Test search endpoint with document_id filter."""
        mock_chunks = [
            {
                "content": "Filtered content",
                "document_id": "specific-doc",
                "filename": "cv_specific.pdf",
                "full_name": "Specific Person",
                "chunk_index": 0,
                "similarity_score": 0.90,
            },
        ]

        with patch("app.api.endpoints.rag.RAGService") as MockRAGService:
            mock_instance = MagicMock()
            mock_instance.get_relevant_chunks = MagicMock(return_value=mock_chunks)
            MockRAGService.return_value = mock_instance

            response = await client.post(
                "/rag/search",
                json={
                    "query": "skills",
                    "k": 5,
                    "document_id": "specific-doc",
                },
            )

            assert response.status_code == 200
            mock_instance.get_relevant_chunks.assert_called_once_with(
                question="skills",
                k=5,
                document_id="specific-doc",
            )

    @pytest.mark.asyncio
    async def test_search_returns_empty_list_when_no_results(
        self, client: AsyncClient
    ) -> None:
        """Test search endpoint returns empty list when no results found."""
        with patch("app.api.endpoints.rag.RAGService") as MockRAGService:
            mock_instance = MagicMock()
            mock_instance.get_relevant_chunks = MagicMock(return_value=[])
            MockRAGService.return_value = mock_instance

            response = await client.post(
                "/rag/search",
                json={"query": "nonexistent topic xyz", "k": 3},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["chunks"] == []

    @pytest.mark.asyncio
    async def test_search_handles_exception(self, client: AsyncClient) -> None:
        """Test search endpoint returns 500 on internal error."""
        with patch("app.api.endpoints.rag.RAGService") as MockRAGService:
            mock_instance = MagicMock()
            mock_instance.get_relevant_chunks = MagicMock(
                side_effect=Exception("Vector store connection failed")
            )
            MockRAGService.return_value = mock_instance

            response = await client.post(
                "/rag/search",
                json={"query": "Python", "k": 3},
            )

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Search failed" in data["detail"]
            assert "Vector store connection failed" in data["detail"]
