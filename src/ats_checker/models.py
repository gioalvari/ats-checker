"""Pydantic models for ATS Resume Analyzer."""

from pydantic import BaseModel, Field


class Resume(BaseModel):
    """Represents a parsed resume."""

    raw_text: str = Field(..., description="Raw text extracted from the resume")
    file_name: str | None = Field(None, description="Original file name if uploaded")
    keywords: set[str] = Field(default_factory=set, description="Extracted keywords")

    model_config = {"arbitrary_types_allowed": True}


class JobDescription(BaseModel):
    """Represents a job description."""

    raw_text: str = Field(..., description="Raw text of the job description")
    source: str | None = Field(None, description="Source URL or file name")
    keywords: set[str] = Field(default_factory=set, description="Extracted keywords")
    company: str | None = Field(None, description="Company name if detected")
    title: str | None = Field(None, description="Job title if detected")

    model_config = {"arbitrary_types_allowed": True}


class Suggestion(BaseModel):
    """A single suggestion for improving resume-job fit."""

    category: str = Field(..., description="Category: skills, experience, keywords, formatting")
    original_text: str | None = Field(None, description="Original text from resume if applicable")
    suggested_text: str = Field(..., description="Suggested modification")
    reasoning: str = Field(..., description="Why this change improves ATS score")
    priority: str = Field("medium", description="Priority: high, medium, low")


class AnalysisResult(BaseModel):
    """Complete analysis result from resume-job comparison."""

    match_score: float = Field(..., ge=0, le=100, description="Overall match score 0-100")
    keyword_score: float = Field(..., ge=0, le=100, description="Keyword match percentage")
    missing_keywords: list[str] = Field(
        default_factory=list, description="Keywords in job but not in resume"
    )
    matching_keywords: list[str] = Field(default_factory=list, description="Keywords found in both")
    extra_keywords: list[str] = Field(
        default_factory=list, description="Keywords in resume but not in job"
    )
    suggestions: list[Suggestion] = Field(
        default_factory=list, description="List of improvement suggestions"
    )
    summary: str = Field("", description="LLM-generated summary of the analysis")
    strengths: list[str] = Field(default_factory=list, description="Resume strengths for this job")
    weaknesses: list[str] = Field(default_factory=list, description="Areas needing improvement")


class AdaptedResume(BaseModel):
    """An adapted/optimized version of the resume."""

    original_text: str = Field(..., description="Original resume text")
    adapted_text: str = Field(..., description="Optimized resume text")
    changes_made: list[str] = Field(default_factory=list, description="List of changes applied")
    preserved_facts: list[str] = Field(
        default_factory=list, description="Facts that were preserved unchanged"
    )
