# RAG service for querying documents with natural language
# Uses vector similarity search to retrieve relevant context for the LLM

import json
import logging
from typing import AsyncGenerator

from langchain_openai import ChatOpenAI
from openai import OpenAIError

from app.core.config import get_settings
from app.rag.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class RAGServiceError(Exception):
    """Base exception for RAG service errors."""

    pass


class VectorSearchError(RAGServiceError):
    """Exception raised when vector search fails."""

    pass


class RAGService:
    """Service for RAG-based question answering with vector search retrieval."""

    def __init__(self) -> None:
        settings = get_settings()

        self.llm = ChatOpenAI(
            model=settings.rag.chat_model,
            api_key=settings.rag.openai_api_key.get_secret_value(),
            temperature=0,
        )

        self.vector_store_service = VectorStoreService()

        logger.info(
            "Initialized RAG service with tool-based retrieval, model: %s",
            settings.rag.chat_model,
        )

    async def query_stream(
        self,
        question: str,
        k: int = 3,
        document_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the RAG query response token by token.

        Retrieves context synchronously, then streams the LLM response.
        Yields JSON chunks for streaming response.

        Args:
            question: The question to ask
            k: Number of relevant chunks to retrieve per search
            document_id: Optional - limit search to a specific document

        Yields:
            JSON string chunks with type and content


        """
        logger.info(
            "Processing RAG query: question_length=%d, k=%d, document_id=%s",
            len(question),
            k,
            document_id,
        )

        # Retrieve relevant documents with error handling
        try:
            filter_dict = {"document_id": document_id} if document_id else None
            retrieved_docs = self.vector_store_service.similarity_search(
                question, k=k, filter_dict=filter_dict
            )
            logger.debug("Retrieved %d documents from vector store", len(retrieved_docs))
        except Exception as e:
            error_msg = f"Failed to retrieve documents: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield json.dumps(
                {"type": "error", "message": "Document retrieval failed"}
            ) + "\n"
            return

        # Check if we got any results
        if not retrieved_docs:
            logger.warning("No relevant documents found for query")
            yield json.dumps(
                {
                    "type": "warning",
                    "message": "No relevant documents found. Please try a different question.",
                }
            ) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return

        # Build context from retrieved documents
        context_parts = []
        sources: list[dict[str, str]] = []
        seen_sources: set[str] = set()

        for doc in retrieved_docs:
            # Include full_name in the context so LLM knows who each chunk belongs to
            full_name = doc.metadata.get("full_name", "Unknown")
            context_parts.append(f"[CV: {full_name}]\n{doc.page_content}")
            source_id = doc.metadata.get("document_id", "unknown")
            if source_id not in seen_sources:
                sources.append(
                    {
                        "document_id": source_id,
                        "filename": doc.metadata.get("filename", "unknown"),
                        "full_name": full_name,
                        "chunk_index": str(doc.metadata.get("chunk_index", 0)),
                    }
                )
                seen_sources.add(source_id)

        context = "\n\n---\n\n".join(context_parts)

        # Create prompt with context
        system_prompt = f"""You are a helpful assistant that answers questions based on documents.

Use the following retrieved context to answer the question. If the context doesn't contain enough information, say so clearly.
Always cite which document(s) you used to answer the question. Be concise but thorough.

Context:
{context}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        # Stream the LLM response with error handling
        try:
            token_count = 0
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    token_count += 1
                    yield json.dumps({"type": "token", "content": chunk.content}) + "\n"

            logger.info(
                "LLM streaming completed: tokens=%d, sources=%d",
                token_count,
                len(sources),
            )

        except OpenAIError as e:
            error_msg = f"OpenAI API error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield json.dumps(
                {
                    "type": "error",
                    "message": "Failed to generate response. Please try again.",
                }
            ) + "\n"
            return
        except Exception as e:
            error_msg = f"Unexpected error during LLM streaming: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield json.dumps(
                {
                    "type": "error",
                    "message": "An unexpected error occurred. Please try again.",
                }
            ) + "\n"
            return

        # Yield sources at the end
        yield json.dumps({"type": "sources", "sources": sources}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

        logger.info(
            "RAG streaming query completed successfully. Retrieved %d sources.",
            len(sources),
        )

    def get_relevant_chunks(
        self,
        question: str,
        k: int = 3,
        document_id: str | None = None,
    ) -> list[dict[str, str | float]]:
        """
        Get relevant document chunks without generating an answer.

        Args:
            question: The search query
            k: Number of chunks to retrieve
            document_id: Optional - limit search to a specific document

        Returns:
            List of chunks with content and metadata

        Raises:
            VectorSearchError: If document retrieval fails
        """
        logger.info(
            "Retrieving relevant chunks: question_length=%d, k=%d, document_id=%s",
            len(question),
            k,
            document_id,
        )

        try:
            filter_dict = {"document_id": document_id} if document_id else None

            results = self.vector_store_service.similarity_search_with_score(
                question, k=k, filter_dict=filter_dict
            )

            chunks: list[dict[str, str | float]] = []
            for doc, score in results:
                chunks.append(
                    {
                        "content": doc.page_content,
                        "document_id": doc.metadata.get("document_id", "unknown"),
                        "filename": doc.metadata.get("filename", "unknown"),
                        "full_name": doc.metadata.get("full_name", "Unknown"),
                        "chunk_index": doc.metadata.get("chunk_index", 0),
                        "similarity_score": float(score),
                    }
                )

            logger.info("Successfully retrieved %d chunks", len(chunks))
            return chunks

        except Exception as e:
            error_msg = f"Failed to retrieve relevant chunks: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorSearchError(error_msg) from e
