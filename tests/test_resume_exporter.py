"""Tests for resume exporter functionality."""

import pytest

from ats_checker.resume_exporter import ResumeExporter, export_resume


class TestResumeExporter:
    """Tests for ResumeExporter class."""

    @pytest.fixture
    def sample_content(self) -> str:
        """Sample resume content for testing."""
        return """JOHN DOE
Software Engineer
john@email.com

SUMMARY
Experienced developer with Python expertise.

EXPERIENCE

Senior Developer | Company Inc | 2020-Present
- Built scalable applications
- Led team of 5 engineers

SKILLS
- Python
- JavaScript
- Docker
"""

    def test_to_markdown(self, sample_content: str) -> None:
        """Test markdown conversion."""
        exporter = ResumeExporter(sample_content)
        md = exporter.to_markdown()

        assert isinstance(md, str)
        assert len(md) > 0
        # Should contain some markdown formatting
        assert "##" in md or "-" in md

    def test_to_html(self, sample_content: str) -> None:
        """Test HTML conversion."""
        exporter = ResumeExporter(sample_content)
        html = exporter.to_html(include_style=True)

        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "<style>" in html
        assert "</body>" in html

    def test_to_html_no_style(self, sample_content: str) -> None:
        """Test HTML conversion without styles."""
        exporter = ResumeExporter(sample_content)
        html = exporter.to_html(include_style=False)

        assert "<style>" not in html

    def test_to_docx(self, sample_content: str) -> None:
        """Test DOCX conversion."""
        exporter = ResumeExporter(sample_content)
        docx_bytes = exporter.to_docx()

        assert isinstance(docx_bytes, bytes)
        assert len(docx_bytes) > 0
        # DOCX files start with PK (zip signature)
        assert docx_bytes[:2] == b"PK"

    def test_save_markdown(self, sample_content: str, tmp_path) -> None:
        """Test saving to markdown file."""
        exporter = ResumeExporter(sample_content)
        output = tmp_path / "resume"

        result_path = exporter.save(output, format="md")

        assert result_path.exists()
        assert result_path.suffix == ".md"
        assert result_path.read_text()

    def test_save_docx(self, sample_content: str, tmp_path) -> None:
        """Test saving to DOCX file."""
        exporter = ResumeExporter(sample_content)
        output = tmp_path / "resume"

        result_path = exporter.save(output, format="docx")

        assert result_path.exists()
        assert result_path.suffix == ".docx"

    def test_save_html(self, sample_content: str, tmp_path) -> None:
        """Test saving to HTML file."""
        exporter = ResumeExporter(sample_content)
        output = tmp_path / "resume"

        result_path = exporter.save(output, format="html")

        assert result_path.exists()
        assert result_path.suffix == ".html"

    def test_save_unsupported_format(self, sample_content: str, tmp_path) -> None:
        """Test saving with unsupported format raises error."""
        exporter = ResumeExporter(sample_content)
        output = tmp_path / "resume"

        with pytest.raises(ValueError, match="Unsupported format"):
            exporter.save(output, format="xyz")


class TestExportResumeFunction:
    """Tests for export_resume convenience function."""

    def test_export_markdown(self) -> None:
        """Test export to markdown."""
        content = "Sample resume"
        result = export_resume(content, format="md")

        assert isinstance(result, str)

    def test_export_docx(self) -> None:
        """Test export to DOCX."""
        content = "Sample resume"
        result = export_resume(content, format="docx")

        assert isinstance(result, bytes)

    def test_export_html(self) -> None:
        """Test export to HTML."""
        content = "Sample resume"
        result = export_resume(content, format="html")

        assert isinstance(result, str)
        assert "<" in result  # Contains HTML tags

    def test_export_unsupported(self) -> None:
        """Test unsupported format."""
        with pytest.raises(ValueError):
            export_resume("content", format="xyz")
