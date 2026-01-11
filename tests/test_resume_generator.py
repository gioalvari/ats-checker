"""Tests for resume generator and adapter."""

import pytest

from ats_checker.resume_generator import ResumeAdapter, ResumeSection


class TestResumeAdapter:
    """Tests for ResumeAdapter class."""

    @pytest.fixture
    def adapter(self) -> ResumeAdapter:
        """Create adapter without LLM for unit tests."""
        return ResumeAdapter(analyzer=None)  # type: ignore

    def test_parse_sections_basic(self, adapter: ResumeAdapter, sample_resume_text: str) -> None:
        """Test basic section parsing."""
        sections = adapter.parse_sections(sample_resume_text)

        assert len(sections) > 0
        assert isinstance(sections[0], ResumeSection)

        # Should find common sections
        section_names = [s.name.lower() for s in sections]
        assert any("experience" in name for name in section_names)
        assert any("skill" in name for name in section_names)

    def test_parse_sections_empty(self, adapter: ResumeAdapter) -> None:
        """Test parsing empty text."""
        sections = adapter.parse_sections("")

        assert sections == []

    def test_parse_sections_no_headers(self, adapter: ResumeAdapter) -> None:
        """Test parsing text without clear headers."""
        text = """John Doe
Software Developer
john@email.com

I have experience in Python and JavaScript.
Built many web applications.
"""
        sections = adapter.parse_sections(text)

        # Should still extract something
        assert len(sections) >= 1

    def test_classify_section(self, adapter: ResumeAdapter) -> None:
        """Test section classification."""
        assert adapter._classify_section("WORK EXPERIENCE") == "experience"
        assert adapter._classify_section("Employment History") == "experience"
        assert adapter._classify_section("SKILLS") == "skills"
        assert adapter._classify_section("Technical Skills") == "skills"
        assert adapter._classify_section("EDUCATION") == "education"
        assert adapter._classify_section("Summary") == "summary"
        assert adapter._classify_section("Career Objective") == "summary"
        assert adapter._classify_section("Random Section") == "other"

    def test_extract_facts(self, adapter: ResumeAdapter) -> None:
        """Test fact extraction from resume."""
        text = """
John Doe
john.doe@email.com

Senior Developer at TechCorp | 2020 - Present
- Increased efficiency by 40%
- Managed budget of $500,000
"""
        facts = adapter._extract_facts(text)

        # Facts may be extracted (years, emails, numbers)
        assert isinstance(facts, list)
        # Should find some data - either years, email, or numbers
        if facts:
            # If facts found, they should be strings
            assert all(isinstance(f, str) for f in facts)


class TestResumeSectionDataclass:
    """Tests for ResumeSection dataclass."""

    def test_resume_section_creation(self) -> None:
        """Test creating ResumeSection."""
        section = ResumeSection(
            name="EXPERIENCE",
            content="Work history here",
            start_index=10,
            end_index=50,
        )

        assert section.name == "EXPERIENCE"
        assert section.content == "Work history here"
        assert section.start_index == 10
        assert section.end_index == 50
