# RAG service for querying documents with natural language
# Uses vector similarity search to retrieve relevant context for the LLM

import json
import logging

from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.rag.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


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
    ):
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
        # First, retrieve relevant documents synchronously
        filter_dict = {"document_id": document_id} if document_id else None
        retrieved_docs = self.vector_store_service.similarity_search(
            question, k=k, filter_dict=filter_dict
        )

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

        # Stream the LLM response
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield json.dumps({"type": "token", "content": chunk.content}) + "\n"

        # Yield sources at the end
        yield json.dumps({"type": "sources", "sources": sources}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

        logger.info(
            "RAG streaming query completed. Retrieved %d sources.",
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
        """
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

        return chunks
