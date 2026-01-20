# PDF service for processing PDF documents
# Handles loading, text extraction, and chunking of PDF files

import hashlib
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class PDFService:
    """Service for processing PDF documents."""

    def __init__(self) -> None:
        settings = get_settings()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag.chunk_size,
            chunk_overlap=settings.rag.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n"],
        )

        logger.info(
            "Initialized PDF service with chunk_size=%d, chunk_overlap=%d",
            settings.rag.chunk_size,
            settings.rag.chunk_overlap,
        )

    def _extract_full_name_from_filename(self, filename: str) -> str:
        """
        Extract full name from CV filename.

        Args:
            filename: PDF filename (e.g., "cv_john_doe.pdf")

        Returns:
            Full name with proper capitalization (e.g., "John Doe")
        """
        # Remove .pdf extension and cv_ prefix
        name_part = filename.lower().replace(".pdf", "").replace("cv_", "")
        # Split by underscore and capitalize each part
        name_parts = name_part.split("_")
        return " ".join(part.capitalize() for part in name_parts)

    def _generate_document_id(self, pdf_path: Path, file_size: int) -> str:
        """
        Generate a unique document ID (UUID format) based on filename and file size.

        Args:
            pdf_path: Path to the PDF file
            file_size: Size of the file in bytes

        Returns:
            Unique document ID as a UUID string
        """
        unique_string = f"{pdf_path.name}:{file_size}"
        # Take first 32 hex chars from hash and format as UUID
        hex_digest = hashlib.sha256(unique_string.encode()).hexdigest()[:32]
        # Format as UUID: 8-4-4-4-12
        return f"{hex_digest[:8]}-{hex_digest[8:12]}-{hex_digest[12:16]}-{hex_digest[16:20]}-{hex_digest[20:32]}"

    def get_pdfs_from_folder(self, folder_path: str) -> list[Path]:
        """
        Get all PDF files from a folder.

        Args:
            folder_path: Path to the folder to scan

        Returns:
            List of Path objects pointing to PDF files
        """
        folder = Path(folder_path)

        if not folder.exists():
            logger.warning("Folder does not exist: %s", folder_path)
            return []

        if not folder.is_dir():
            logger.warning("Path is not a directory: %s", folder_path)
            return []

        pdf_files = list(folder.glob("*.pdf"))
        logger.info("Found %d PDF files in %s", len(pdf_files), folder_path)

        return pdf_files

    def process_pdf_from_path(
        self, pdf_path: Path
    ) -> tuple[list[LangchainDocument], int, str, int]:
        """
        Process a PDF file and return chunks with metadata.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Tuple of (chunks, page_count, document_id, file_size)
        """
        # Get file size
        file_size = pdf_path.stat().st_size

        # Generate deterministic document ID from filename and file size
        # This ensures the same file always gets the same ID for duplicate detection
        document_id = self._generate_document_id(pdf_path, file_size)

        # Load PDF
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        page_count = len(pages)

        logger.info(
            "Loaded PDF %s: %d pages, %d bytes",
            pdf_path.name,
            page_count,
            file_size,
        )

        # Split into chunks
        chunks = self.text_splitter.split_documents(pages)

        # Extract full name from filename (e.g., "cv_john_doe.pdf" -> "John Doe")
        full_name = self._extract_full_name_from_filename(pdf_path.name)

        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "document_id": document_id,
                    "filename": pdf_path.name,
                    "full_name": full_name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            )

        logger.info(
            "Split %s into %d chunks",
            pdf_path.name,
            len(chunks),
        )

        return chunks, page_count, document_id, file_size

