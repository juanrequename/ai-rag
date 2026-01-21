# Tests for PDFService - handles PDF processing, text extraction, and chunking

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document as LangchainDocument

from app.rag.pdf_service import PDFService


@pytest.fixture
def pdf_service() -> PDFService:
    """Create a PDFService instance with mocked settings."""
    with patch("app.rag.pdf_service.get_settings") as mock_settings:
        mock_settings.return_value.rag.chunk_size = 500
        mock_settings.return_value.rag.chunk_overlap = 50
        service = PDFService()
    return service


class TestExtractFullNameFromFilename:
    """Tests for _extract_full_name_from_filename method."""

    def test_standard_cv_filename(self, pdf_service: PDFService) -> None:
        """Should extract name from standard CV filename format."""
        result = pdf_service._extract_full_name_from_filename("cv_john_doe.pdf")
        assert result == "John Doe"


class TestGenerateDocumentId:
    """Tests for _generate_document_id method."""

    def test_generates_uuid_format(self, pdf_service: PDFService) -> None:
        """Should generate ID in UUID format (8-4-4-4-12)."""
        pdf_path = Path("/fake/path/cv_test.pdf")
        result = pdf_service._generate_document_id(pdf_path, 1024)

        # Check UUID format
        parts = result.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4
        assert len(parts[2]) == 4
        assert len(parts[3]) == 4
        assert len(parts[4]) == 12

    def test_deterministic_for_same_input(self, pdf_service: PDFService) -> None:
        """Should generate same ID for same file and size."""
        pdf_path = Path("/fake/path/cv_test.pdf")
        result1 = pdf_service._generate_document_id(pdf_path, 1024)
        result2 = pdf_service._generate_document_id(pdf_path, 1024)
        assert result1 == result2

    def test_different_for_different_sizes(self, pdf_service: PDFService) -> None:
        """Should generate different IDs for different file sizes."""
        pdf_path = Path("/fake/path/cv_test.pdf")
        result1 = pdf_service._generate_document_id(pdf_path, 1024)
        result2 = pdf_service._generate_document_id(pdf_path, 2048)
        assert result1 != result2

    def test_different_for_different_filenames(self, pdf_service: PDFService) -> None:
        """Should generate different IDs for different filenames."""
        path1 = Path("/fake/path/cv_john.pdf")
        path2 = Path("/fake/path/cv_jane.pdf")
        result1 = pdf_service._generate_document_id(path1, 1024)
        result2 = pdf_service._generate_document_id(path2, 1024)
        assert result1 != result2


class TestGetPdfsFromFolder:
    """Tests for get_pdfs_from_folder method."""

    def test_returns_pdf_files(
        self, pdf_service: PDFService, tmp_path: Path
    ) -> None:
        """Should return list of PDF files in folder."""
        # Create test PDF files
        (tmp_path / "doc1.pdf").touch()
        (tmp_path / "doc2.pdf").touch()
        (tmp_path / "other.txt").touch()

        result = pdf_service.get_pdfs_from_folder(str(tmp_path))

        assert len(result) == 2
        assert all(p.suffix == ".pdf" for p in result)

    def test_returns_empty_for_nonexistent_folder(
        self, pdf_service: PDFService
    ) -> None:
        """Should return empty list for nonexistent folder."""
        result = pdf_service.get_pdfs_from_folder("/nonexistent/path")
        assert result == []

    def test_returns_empty_for_file_path(
        self, pdf_service: PDFService, tmp_path: Path
    ) -> None:
        """Should return empty list when path is a file, not directory."""
        file_path = tmp_path / "test.pdf"
        file_path.touch()

        result = pdf_service.get_pdfs_from_folder(str(file_path))
        assert result == []

    def test_returns_empty_for_no_pdfs(
        self, pdf_service: PDFService, tmp_path: Path
    ) -> None:
        """Should return empty list when no PDFs in folder."""
        (tmp_path / "doc.txt").touch()
        (tmp_path / "image.png").touch()

        result = pdf_service.get_pdfs_from_folder(str(tmp_path))
        assert result == []


class TestProcessPdfFromPath:
    """Tests for process_pdf_from_path method."""

    @patch("app.rag.pdf_service.PyPDFLoader")
    def test_processes_pdf_successfully(
        self, mock_loader_class: MagicMock, pdf_service: PDFService, tmp_path: Path
    ) -> None:
        """Should process PDF and return chunks with metadata."""
        # Create a mock PDF file
        pdf_path = tmp_path / "cv_john_doe.pdf"
        pdf_path.write_bytes(b"fake pdf content for testing")

        # Mock the PDF loader
        mock_loader = MagicMock()
        mock_loader.load.return_value = [
            LangchainDocument(
                page_content="Page 1 content about John Doe's experience",
                metadata={"page": 0},
            ),
            LangchainDocument(
                page_content="Page 2 content about skills",
                metadata={"page": 1},
            ),
        ]
        mock_loader_class.return_value = mock_loader

        chunks, page_count, doc_id, file_size = pdf_service.process_pdf_from_path(
            pdf_path
        )

        assert page_count == 2
        assert file_size == len(b"fake pdf content for testing")
        assert len(chunks) >= 1

        # Check metadata on first chunk
        first_chunk = chunks[0]
        assert first_chunk.metadata["document_id"] == doc_id
        assert first_chunk.metadata["filename"] == "cv_john_doe.pdf"
        assert first_chunk.metadata["full_name"] == "John Doe"
        assert first_chunk.metadata["chunk_index"] == 0
        assert first_chunk.metadata["total_chunks"] == len(chunks)

    @patch("app.rag.pdf_service.PyPDFLoader")
    def test_chunk_indices_are_sequential(
        self, mock_loader_class: MagicMock, pdf_service: PDFService, tmp_path: Path
    ) -> None:
        """Should assign sequential chunk indices."""
        pdf_path = tmp_path / "cv_test.pdf"
        pdf_path.write_bytes(b"content")

        # Create multiple pages to get multiple chunks
        mock_loader = MagicMock()
        mock_loader.load.return_value = [
            LangchainDocument(page_content="Content " * 200, metadata={"page": 0}),
            LangchainDocument(page_content="More content " * 200, metadata={"page": 1}),
        ]
        mock_loader_class.return_value = mock_loader

        chunks, _, _, _ = pdf_service.process_pdf_from_path(pdf_path)

        # Verify sequential chunk indices
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    @patch("app.rag.pdf_service.PyPDFLoader")
    def test_total_chunks_in_metadata(
        self, mock_loader_class: MagicMock, pdf_service: PDFService, tmp_path: Path
    ) -> None:
        """Should include total_chunks in all chunk metadata."""
        pdf_path = tmp_path / "cv_test.pdf"
        pdf_path.write_bytes(b"content")

        mock_loader = MagicMock()
        mock_loader.load.return_value = [
            LangchainDocument(page_content="Content " * 200, metadata={"page": 0}),
        ]
        mock_loader_class.return_value = mock_loader

        chunks, _, _, _ = pdf_service.process_pdf_from_path(pdf_path)

        for chunk in chunks:
            assert chunk.metadata["total_chunks"] == len(chunks)
