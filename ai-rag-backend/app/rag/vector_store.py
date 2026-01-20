# Vector store service using PGVector via langchain-postgres
# Handles storing and retrieving document embeddings

import logging

from langchain_core.documents import Document as LangchainDocument
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Service for managing document embeddings in PGVector."""

    _instance: "VectorStoreService | None" = None
    _vector_store: PGVector | None = None

    def __new__(cls) -> "VectorStoreService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if VectorStoreService._vector_store is not None:
            return

        settings = get_settings()

        self.embeddings = OpenAIEmbeddings(
            model=settings.rag.embedding_model,
            api_key=settings.rag.openai_api_key.get_secret_value(),
        )

        VectorStoreService._vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=settings.rag.collection_name,
            connection=settings.psycopg_database_uri,
            use_jsonb=True,
        )

        logger.info(
            "Initialized PGVector store with collection: %s",
            settings.rag.collection_name,
        )

    @property
    def vector_store(self) -> PGVector:
        """Get the vector store instance."""
        if VectorStoreService._vector_store is None:
            raise RuntimeError("Vector store not initialized")
        return VectorStoreService._vector_store

    def add_documents(self, documents: list[LangchainDocument]) -> list[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of LangchainDocument objects to store

        Returns:
            List of document IDs
        """
        ids = self.vector_store.add_documents(documents)
        logger.info("Added %d documents to vector store", len(documents))
        return ids

    def similarity_search(
        self, query: str, k: int = 4, filter_dict: dict[str, str] | None = None
    ) -> list[LangchainDocument]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            List of similar documents
        """
        if filter_dict:
            results = self.vector_store.similarity_search(
                query, k=k, filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search(query, k=k)

        logger.info("Found %d similar documents for query", len(results))
        return results

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter_dict: dict[str, str] | None = None
    ) -> list[tuple[LangchainDocument, float]]:
        """
        Search for similar documents with relevance scores.

        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            List of (document, score) tuples
        """
        if filter_dict:
            results = self.vector_store.similarity_search_with_score(
                query, k=k, filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search_with_score(query, k=k)

        logger.info("Found %d similar documents with scores", len(results))
        return results

    def delete_by_document_id(self, document_id: str) -> None:
        """
        Delete all chunks for a specific document.

        Args:
            document_id: The document ID to delete
        """
        # PGVector delete by filter
        self.vector_store.delete(filter={"document_id": document_id})
        logger.info("Deleted chunks for document: %s", document_id)

    def get_retriever(
        self, k: int = 4, filter_dict: dict[str, str] | None = None
    ) -> VectorStoreRetriever:
        """
        Get a retriever for use with LangChain chains.

        Args:
            k: Number of documents to retrieve
            filter_dict: Optional metadata filter

        Returns:
            A LangChain retriever
        """
        search_kwargs: dict[str, int | dict[str, str]] = {"k": k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
