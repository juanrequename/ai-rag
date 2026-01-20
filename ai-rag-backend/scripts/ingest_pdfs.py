#!/usr/bin/env python3
"""
Script to ingest all PDFs from the RAG__PDF_FOLDER setting.
"""
import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.core.config import get_settings, Settings
from app.models import Document
from app.rag import PDFService, VectorStoreService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class IngestResult(Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


def ingest_pdf(
    pdf_path: Path,
    session: Session,
    vector_store: VectorStoreService,
    pdf_service: PDFService,
) -> IngestResult:
    """
    Ingest a single PDF file.

    Args:
        pdf_path: Path to the PDF file to ingest
        session: SQLAlchemy database session
        vector_store: Vector store service for storing embeddings
        pdf_service: PDF service for processing PDF files

    Returns:
        IngestResult indicating success, skipped (already exists), or failed
    """
    try:
        # Process PDF
        from langchain_core.documents import Document as LangchainDocument
        
        chunks: list[LangchainDocument]
        page_count: int
        document_id: str
        file_size: int
        chunks, page_count, document_id, file_size = pdf_service.process_pdf_from_path(pdf_path)

        # Check if document already exists
        existing: Optional[Document] = session.query(Document).filter(
            Document.document_id == document_id
        ).first()
        if existing:
            logger.warning("Document %s already exists, skipping: %s", document_id, pdf_path.name)
            return IngestResult.SKIPPED

        # Create document record
        document: Document = Document(
            document_id=document_id,
            filename=pdf_path.name,
            file_size=file_size,
            page_count=page_count,
            chunk_count=len(chunks),
            status="completed",
        )
        session.add(document)

        # Store in vector database (this embeds the documents)
        vector_ids: list[str] = vector_store.add_documents(chunks)

        logger.info(
            "✓ Ingested %s: %d pages, %d chunks",
            pdf_path.name,
            page_count,
            len(chunks),
        )
        return IngestResult.SUCCESS

    except Exception as e:
        logger.error("✗ Failed to process %s: %s", pdf_path.name, str(e))
        # Rollback the session to clear the failed transaction state
        session.rollback()
        return IngestResult.FAILED


def main(
    pdf_folder: Optional[str] = None,
    skip_existing: bool = True,
) -> int:
    """
    Main function to ingest all PDFs from the configured folder.

    Args:
        pdf_folder: Optional override for the PDF folder path.
                   If None, uses the path from settings (RAG__PDF_FOLDER).
        skip_existing: If True, skip documents that already exist in the database.
                      If False, would re-ingest (currently always skips for safety).

    Returns:
        Exit code: 0 if all files processed successfully (or skipped),
                  1 if any files failed to process
    """
    settings: Settings = get_settings()
    target_path: Path = Path(pdf_folder) if pdf_folder else Path(settings.rag.pdf_folder)
    logger.info("Using PDF folder: %s", target_path)

    if not target_path.exists():
        logger.error("PDF folder does not exist: %s", target_path)
        return 1
    if not target_path.is_dir():
        logger.error("PDF path is not a directory: %s", target_path)
        return 1

    # Initialize services
    pdf_service: PDFService = PDFService()
    vector_store: VectorStoreService = VectorStoreService()

    # Create sync database session (using psycopg driver, not asyncpg)
    engine: Engine = create_engine(settings.psycopg_database_uri)

    pdf_files: list[Path] = pdf_service.get_pdfs_from_folder(str(target_path))

    if not pdf_files:
        logger.warning("No PDF files found in: %s", target_path)
        return 0

    logger.info("Found %d PDF file(s) to process", len(pdf_files))
    print("-" * 50)

    # Process files
    successful: int = 0
    failed: int = 0
    skipped: int = 0

    with Session(engine) as session:
        for pdf_path in pdf_files:
            result: IngestResult = ingest_pdf(pdf_path, session, vector_store, pdf_service)
            if result == IngestResult.SUCCESS:
                successful += 1
            elif result == IngestResult.SKIPPED:
                skipped += 1
            else:
                failed += 1

        session.commit()

    # Print summary
    print("-" * 50)
    print(f"Ingestion complete:")
    print(f"  ✓ Successful: {successful}")
    print(f"  ⊘ Skipped (already exists): {skipped}")
    print(f"  ✗ Failed: {failed}")
    print(f"  Total: {len(pdf_files)}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
