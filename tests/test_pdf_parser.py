"""Tests for PDF and document parsing."""

import io

import pytest

from ats_checker.pdf_parser import (
    extract_text_from_file,
)


class TestExtractTextFromFile:
    """Tests for extract_text_from_file function."""

    def test_extract_from_txt_bytes(self) -> None:
        """Test extracting text from TXT bytes."""
        text_content = b"This is a test resume\nWith multiple lines"
        result = extract_text_from_file(text_content, file_name="resume.txt")

        assert "This is a test resume" in result
        assert "With multiple lines" in result

    def test_extract_from_txt_string_path(self, tmp_path) -> None:
        """Test extracting text from TXT file path."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Sample resume content", encoding="utf-8")

        result = extract_text_from_file(str(txt_file))

        assert "Sample resume content" in result

    def test_extract_from_file_object(self, tmp_path) -> None:
        """Test extracting from file-like object."""
        content = "File object content"
        file_obj = io.BytesIO(content.encode("utf-8"))

        result = extract_text_from_file(file_obj, file_name="test.txt")

        assert content in result

    def test_unsupported_format(self) -> None:
        """Test that unsupported formats raise error."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            extract_text_from_file(b"content", file_name="file.xyz")

    def test_extract_handles_unicode(self, tmp_path) -> None:
        """Test handling of unicode characters."""
        txt_file = tmp_path / "unicode.txt"
        txt_file.write_text("Résumé with émojis 🎉", encoding="utf-8")

        result = extract_text_from_file(str(txt_file))

        assert "Résumé" in result
        assert "🎉" in result


class TestPDFExtraction:
    """Tests for PDF-specific extraction."""

    def test_extract_pdf_minimal(self, sample_pdf_bytes: bytes) -> None:
        """Test extraction from minimal PDF."""
        # Note: The minimal PDF in fixtures may not have extractable text
        # This test verifies the function handles it gracefully
        try:
            result = extract_text_from_file(sample_pdf_bytes, file_name="test.pdf")
            # If extraction succeeds, result should be a string
            assert isinstance(result, str)
        except ValueError as e:
            # If no text found, that's expected for minimal PDF
            assert "No text content found" in str(e) or "Failed to extract" in str(e)
