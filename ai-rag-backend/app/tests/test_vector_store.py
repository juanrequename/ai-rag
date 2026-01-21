# Tests for VectorStoreService - handles storing and retrieving document embeddings

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document as LangchainDocument

from app.rag.vector_store import VectorStoreService


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    """Reset the singleton instance before each test."""
    VectorStoreService._instance = None
    VectorStoreService._vector_store = None


@pytest.fixture
def mock_pgvector() -> MagicMock:
    """Create a mock PGVector instance."""
    return MagicMock()


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Create a mock OpenAIEmbeddings instance."""
    return MagicMock()


@pytest.fixture
def vector_store_service(
    mock_pgvector: MagicMock, mock_embeddings: MagicMock
) -> VectorStoreService:
    """Create a VectorStoreService with mocked dependencies."""
    with (
        patch("app.rag.vector_store.get_settings") as mock_settings,
        patch("app.rag.vector_store.OpenAIEmbeddings", return_value=mock_embeddings),
        patch("app.rag.vector_store.PGVector", return_value=mock_pgvector),
    ):
        mock_settings.return_value.rag.embedding_model = "text-embedding-3-small"
        mock_settings.return_value.rag.openai_api_key.get_secret_value.return_value = (
            "fake-key"
        )
        mock_settings.return_value.rag.collection_name = "test_collection"
        mock_settings.return_value.psycopg_database_uri = "postgresql://test"

        service = VectorStoreService()
        # Ensure the mock is set
        VectorStoreService._vector_store = mock_pgvector

    return service


class TestVectorStoreServiceSingleton:
    """Tests for singleton pattern."""

    def test_singleton_returns_same_instance(
        self, mock_pgvector: MagicMock, mock_embeddings: MagicMock
    ) -> None:
        """Should return the same instance on multiple instantiations."""
        with (
            patch("app.rag.vector_store.get_settings") as mock_settings,
            patch(
                "app.rag.vector_store.OpenAIEmbeddings", return_value=mock_embeddings
            ),
            patch("app.rag.vector_store.PGVector", return_value=mock_pgvector),
        ):
            mock_settings.return_value.rag.embedding_model = "test-model"
            mock_settings.return_value.rag.openai_api_key.get_secret_value.return_value = "key"
            mock_settings.return_value.rag.collection_name = "test"
            mock_settings.return_value.psycopg_database_uri = "postgresql://test"

            instance1 = VectorStoreService()
            instance2 = VectorStoreService()

            assert instance1 is instance2


class TestAddDocuments:
    """Tests for add_documents method."""

    def test_adds_documents_successfully(
        self, vector_store_service: VectorStoreService, mock_pgvector: MagicMock
    ) -> None:
        """Should add documents to vector store."""
        mock_pgvector.add_documents.return_value = ["id1", "id2"]

        documents = [
            LangchainDocument(page_content="Content 1", metadata={"doc_id": "1"}),
            LangchainDocument(page_content="Content 2", metadata={"doc_id": "2"}),
        ]

        result = vector_store_service.add_documents(documents)

        mock_pgvector.add_documents.assert_called_once_with(documents)
        assert result == ["id1", "id2"]

    def test_adds_empty_list(
        self, vector_store_service: VectorStoreService, mock_pgvector: MagicMock
    ) -> None:
        """Should handle empty document list."""
        mock_pgvector.add_documents.return_value = []

        result = vector_store_service.add_documents([])

        mock_pgvector.add_documents.assert_called_once_with([])
        assert result == []


class TestSimilaritySearch:
    """Tests for similarity_search method."""

    def test_search_without_filter(
        self, vector_store_service: VectorStoreService, mock_pgvector: MagicMock
    ) -> None:
        """Should perform similarity search without filter."""
        expected_docs = [
            LangchainDocument(page_content="Result 1", metadata={}),
            LangchainDocument(page_content="Result 2", metadata={}),
        ]
        mock_pgvector.similarity_search.return_value = expected_docs

        result = vector_store_service.similarity_search("test query", k=2)

        mock_pgvector.similarity_search.assert_called_once_with("test query", k=2)
        assert result == expected_docs

    def test_search_with_filter(
        self, vector_store_service: VectorStoreService, mock_pgvector: MagicMock
    ) -> None:
        """Should perform similarity search with filter."""
        expected_docs = [LangchainDocument(page_content="Filtered", metadata={})]
        mock_pgvector.similarity_search.return_value = expected_docs

        result = vector_store_service.similarity_search(
            "test query", k=3, filter_dict={"document_id": "doc-123"}
        )

        mock_pgvector.similarity_search.assert_called_once_with(
            "test query", k=3, filter={"document_id": "doc-123"}
        )
        assert result == expected_docs

    def test_search_default_k(
        self, vector_store_service: VectorStoreService, mock_pgvector: MagicMock
    ) -> None:
        """Should use default k=3 when not specified."""
        mock_pgvector.similarity_search.return_value = []

        vector_store_service.similarity_search("query")

        mock_pgvector.similarity_search.assert_called_once_with("query", k=3)


class TestSimilaritySearchWithScore:
    """Tests for similarity_search_with_score method."""

    def test_search_with_score_without_filter(
        self, vector_store_service: VectorStoreService, mock_pgvector: MagicMock
    ) -> None:
        """Should return documents with scores."""
        expected_results = [
            (LangchainDocument(page_content="Result 1", metadata={}), 0.95),
            (LangchainDocument(page_content="Result 2", metadata={}), 0.85),
        ]
        mock_pgvector.similarity_search_with_score.return_value = expected_results

        result = vector_store_service.similarity_search_with_score("query", k=2)

        mock_pgvector.similarity_search_with_score.assert_called_once_with(
            "query", k=2
        )
        assert result == expected_results

    def test_search_with_score_with_filter(
        self, vector_store_service: VectorStoreService, mock_pgvector: MagicMock
    ) -> None:
        """Should apply filter when searching with scores."""
        expected_results = [
            (LangchainDocument(page_content="Filtered", metadata={}), 0.9),
        ]
        mock_pgvector.similarity_search_with_score.return_value = expected_results

        result = vector_store_service.similarity_search_with_score(
            "query", k=1, filter_dict={"document_id": "doc-456"}
        )

        mock_pgvector.similarity_search_with_score.assert_called_once_with(
            "query", k=1, filter={"document_id": "doc-456"}
        )
        assert result == expected_results


class TestDeleteByDocumentId:
    """Tests for delete_by_document_id method."""

    def test_deletes_by_document_id(
        self, vector_store_service: VectorStoreService, mock_pgvector: MagicMock
    ) -> None:
        """Should delete chunks by document_id filter."""
        vector_store_service.delete_by_document_id("doc-to-delete")

        mock_pgvector.delete.assert_called_once_with(
            filter={"document_id": "doc-to-delete"}
        )


class TestGetRetriever:
    """Tests for get_retriever method."""

    def test_get_retriever_default(
        self, vector_store_service: VectorStoreService, mock_pgvector: MagicMock
    ) -> None:
        """Should create retriever with default settings."""
        mock_retriever = MagicMock()
        mock_pgvector.as_retriever.return_value = mock_retriever

        result = vector_store_service.get_retriever()

        mock_pgvector.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
        assert result == mock_retriever

    def test_get_retriever_custom_k(
        self, vector_store_service: VectorStoreService, mock_pgvector: MagicMock
    ) -> None:
        """Should create retriever with custom k value."""
        mock_retriever = MagicMock()
        mock_pgvector.as_retriever.return_value = mock_retriever

        result = vector_store_service.get_retriever(k=5)

        mock_pgvector.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
        assert result == mock_retriever

    def test_get_retriever_with_filter(
        self, vector_store_service: VectorStoreService, mock_pgvector: MagicMock
    ) -> None:
        """Should create retriever with filter."""
        mock_retriever = MagicMock()
        mock_pgvector.as_retriever.return_value = mock_retriever

        result = vector_store_service.get_retriever(
            k=4, filter_dict={"document_id": "specific-doc"}
        )

        mock_pgvector.as_retriever.assert_called_once_with(
            search_kwargs={"k": 4, "filter": {"document_id": "specific-doc"}}
        )
        assert result == mock_retriever


class TestVectorStoreProperty:
    """Tests for vector_store property."""

    def test_returns_vector_store(
        self, vector_store_service: VectorStoreService, mock_pgvector: MagicMock
    ) -> None:
        """Should return the vector store instance."""
        result = vector_store_service.vector_store
        assert result == mock_pgvector

    def test_raises_when_not_initialized(self) -> None:
        """Should raise RuntimeError when vector store not initialized."""
        # Reset singleton
        VectorStoreService._instance = None
        VectorStoreService._vector_store = None

        # Create instance without initializing vector store
        with patch.object(VectorStoreService, "__init__", lambda self: None):
            service = object.__new__(VectorStoreService)

        with pytest.raises(RuntimeError, match="Vector store not initialized"):
            _ = service.vector_store
