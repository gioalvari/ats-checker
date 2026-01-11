"""Tests for keyword extraction and similarity calculation."""

import pytest

from ats_checker.keyword_extractor import (
    calculate_keyword_match,
    calculate_similarity,
    extract_keywords,
    get_keyword_importance,
)


class TestExtractKeywords:
    """Tests for extract_keywords function."""

    def test_extract_keywords_basic(self, sample_resume_text: str) -> None:
        """Test basic keyword extraction from resume."""
        keywords = extract_keywords(sample_resume_text)

        assert isinstance(keywords, set)
        assert len(keywords) > 0
        # Should contain common technical terms
        assert any("python" in kw for kw in keywords)
        assert any("engineer" in kw for kw in keywords)

    def test_extract_keywords_empty_text(self, empty_text: str) -> None:
        """Test keyword extraction from empty text."""
        keywords = extract_keywords(empty_text)
        assert keywords == set()

    def test_extract_keywords_whitespace_only(self) -> None:
        """Test keyword extraction from whitespace-only text."""
        keywords = extract_keywords("   \n\t   ")
        assert keywords == set()

    def test_extract_keywords_min_length(self) -> None:
        """Test minimum keyword length filter."""
        text = "I am a go pro at AI and ML work"
        keywords = extract_keywords(text, min_length=3)

        # Short words like 'go', 'ai', 'ml' should be filtered
        assert all(len(kw) >= 3 for kw in keywords)

    def test_extract_keywords_ngrams_disabled(self) -> None:
        """Test extraction without ngrams."""
        text = "Python developer with experience in machine learning"

        # With ngrams
        keywords_with = extract_keywords(text, include_ngrams=True)
        # Without ngrams
        keywords_without = extract_keywords(text, include_ngrams=False)

        # With ngrams should have more keywords (includes bigrams)
        assert len(keywords_with) >= len(keywords_without)


class TestCalculateSimilarity:
    """Tests for calculate_similarity function."""

    def test_similarity_identical_texts(self) -> None:
        """Test similarity of identical texts."""
        text = "Python developer with experience in Django and Flask"
        similarity = calculate_similarity(text, text)

        assert similarity == pytest.approx(1.0, rel=0.01)

    def test_similarity_different_texts(
        self, sample_resume_text: str, sample_job_description: str
    ) -> None:
        """Test similarity of different but related texts."""
        similarity = calculate_similarity(sample_resume_text, sample_job_description)

        # Should have some similarity (same domain)
        assert 0.0 < similarity < 1.0

    def test_similarity_unrelated_texts(self) -> None:
        """Test similarity of completely unrelated texts."""
        text1 = "Python programming machine learning data science"
        text2 = "Cooking recipes pasta sauce italian cuisine"

        similarity = calculate_similarity(text1, text2)

        # Should have low similarity
        assert similarity < 0.3

    def test_similarity_empty_text(self) -> None:
        """Test similarity with empty text."""
        similarity = calculate_similarity("", "some text")
        assert similarity == 0.0

        similarity = calculate_similarity("some text", "")
        assert similarity == 0.0


class TestCalculateKeywordMatch:
    """Tests for calculate_keyword_match function."""

    def test_keyword_match_full_overlap(self) -> None:
        """Test when all job keywords are in resume."""
        resume_kw = {"python", "javascript", "docker", "aws"}
        job_kw = {"python", "docker"}

        score, matching, missing, extra = calculate_keyword_match(resume_kw, job_kw)

        assert score == 100.0
        assert matching == {"python", "docker"}
        assert missing == set()
        assert extra == {"javascript", "aws"}

    def test_keyword_match_partial_overlap(self) -> None:
        """Test partial keyword overlap."""
        resume_kw = {"python", "javascript"}
        job_kw = {"python", "go", "rust"}

        score, matching, missing, extra = calculate_keyword_match(resume_kw, job_kw)

        assert score == pytest.approx(33.33, rel=0.1)
        assert matching == {"python"}
        assert missing == {"go", "rust"}
        assert extra == {"javascript"}

    def test_keyword_match_no_overlap(self) -> None:
        """Test no keyword overlap."""
        resume_kw = {"python", "django"}
        job_kw = {"java", "spring"}

        score, matching, missing, extra = calculate_keyword_match(resume_kw, job_kw)

        assert score == 0.0
        assert matching == set()
        assert missing == {"java", "spring"}
        assert extra == {"python", "django"}

    def test_keyword_match_empty_job(self) -> None:
        """Test with empty job keywords."""
        resume_kw = {"python", "django"}
        job_kw: set[str] = set()

        score, matching, missing, extra = calculate_keyword_match(resume_kw, job_kw)

        assert score == 100.0  # No requirements = full match


class TestGetKeywordImportance:
    """Tests for get_keyword_importance function."""

    def test_high_importance_keywords(self) -> None:
        """Test high importance technical keywords."""
        assert get_keyword_importance("python") == "high"
        assert get_keyword_importance("aws") == "high"
        assert get_keyword_importance("docker") == "high"

    def test_medium_importance_keywords(self) -> None:
        """Test medium importance keywords."""
        assert get_keyword_importance("development") == "medium"
        assert get_keyword_importance("implementation") == "medium"

    def test_low_importance_keywords(self) -> None:
        """Test low importance short keywords."""
        # Short keywords not in TECH_KEYWORDS
        assert get_keyword_importance("web") == "low"
        assert get_keyword_importance("it") == "low"
