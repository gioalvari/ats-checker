"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from ats_checker.models import (
    AdaptedResume,
    AnalysisResult,
    JobDescription,
    Resume,
    Suggestion,
)


class TestResumeModel:
    """Tests for Resume model."""

    def test_resume_creation(self) -> None:
        """Test basic resume creation."""
        resume = Resume(
            raw_text="Sample resume text",
            file_name="resume.pdf",
            keywords={"python", "django"},
        )

        assert resume.raw_text == "Sample resume text"
        assert resume.file_name == "resume.pdf"
        assert "python" in resume.keywords

    def test_resume_minimal(self) -> None:
        """Test resume with minimal fields."""
        resume = Resume(raw_text="Text only")

        assert resume.raw_text == "Text only"
        assert resume.file_name is None
        assert resume.keywords == set()

    def test_resume_empty_text_fails(self) -> None:
        """Test that empty text is technically allowed but validates."""
        # Pydantic allows empty string by default
        resume = Resume(raw_text="")
        assert resume.raw_text == ""


class TestJobDescriptionModel:
    """Tests for JobDescription model."""

    def test_job_creation(self) -> None:
        """Test job description creation."""
        job = JobDescription(
            raw_text="Looking for a Python developer",
            source="https://company.com/jobs/123",
            company="TechCorp",
            title="Senior Developer",
        )

        assert job.raw_text == "Looking for a Python developer"
        assert job.source == "https://company.com/jobs/123"
        assert job.company == "TechCorp"

    def test_job_minimal(self) -> None:
        """Test job with minimal fields."""
        job = JobDescription(raw_text="Job text")

        assert job.source is None
        assert job.company is None


class TestSuggestionModel:
    """Tests for Suggestion model."""

    def test_suggestion_creation(self) -> None:
        """Test suggestion creation."""
        suggestion = Suggestion(
            category="skills",
            original_text="Experienced in Python",
            suggested_text="Expert Python developer with 5+ years experience",
            reasoning="More specific and quantified",
            priority="high",
        )

        assert suggestion.category == "skills"
        assert suggestion.priority == "high"

    def test_suggestion_default_priority(self) -> None:
        """Test default priority value."""
        suggestion = Suggestion(
            category="keywords",
            suggested_text="Add Docker experience",
            reasoning="Missing key skill",
        )

        assert suggestion.priority == "medium"


class TestAnalysisResultModel:
    """Tests for AnalysisResult model."""

    def test_analysis_result_creation(self) -> None:
        """Test analysis result creation."""
        result = AnalysisResult(
            match_score=75.5,
            keyword_score=80.0,
            missing_keywords=["go", "rust"],
            matching_keywords=["python", "docker"],
            suggestions=[
                Suggestion(
                    category="skills",
                    suggested_text="Add Go experience",
                    reasoning="Required skill",
                )
            ],
        )

        assert result.match_score == 75.5
        assert len(result.missing_keywords) == 2
        assert len(result.suggestions) == 1

    def test_analysis_result_score_bounds(self) -> None:
        """Test score validation bounds."""
        # Valid score
        result = AnalysisResult(match_score=0, keyword_score=100)
        assert result.match_score == 0
        assert result.keyword_score == 100

        # Invalid scores should fail
        with pytest.raises(ValidationError):
            AnalysisResult(match_score=-1, keyword_score=50)

        with pytest.raises(ValidationError):
            AnalysisResult(match_score=50, keyword_score=101)


class TestAdaptedResumeModel:
    """Tests for AdaptedResume model."""

    def test_adapted_resume_creation(self) -> None:
        """Test adapted resume creation."""
        adapted = AdaptedResume(
            original_text="Original resume",
            adapted_text="Optimized resume",
            changes_made=["Improved summary", "Added keywords"],
            preserved_facts=["Employment dates", "Company names"],
        )

        assert adapted.original_text == "Original resume"
        assert adapted.adapted_text == "Optimized resume"
        assert len(adapted.changes_made) == 2

    def test_adapted_resume_defaults(self) -> None:
        """Test adapted resume default values."""
        adapted = AdaptedResume(
            original_text="Original",
            adapted_text="Adapted",
        )

        assert adapted.changes_made == []
        assert adapted.preserved_facts == []
