"""Resume adaptation and generation with fact preservation."""

import logging
import re
from dataclasses import dataclass

from ats_checker.llm_analyzer import OllamaAnalyzer
from ats_checker.models import AdaptedResume, Suggestion

logger = logging.getLogger(__name__)


@dataclass
class ResumeSection:
    """Represents a section of a resume."""

    name: str
    content: str
    start_index: int
    end_index: int


class ResumeAdapter:
    """Adapts resumes to better match job descriptions while preserving facts."""

    # Common resume section headers
    SECTION_PATTERNS = [
        r"(?i)^(professional\s+)?summary",
        r"(?i)^(career\s+)?objective",
        r"(?i)^(work\s+)?experience",
        r"(?i)^employment(\s+history)?",
        r"(?i)^education",
        r"(?i)^skills",
        r"(?i)^(technical\s+)?skills",
        r"(?i)^certifications?",
        r"(?i)^projects?",
        r"(?i)^achievements?",
        r"(?i)^awards?",
        r"(?i)^publications?",
        r"(?i)^languages?",
        r"(?i)^interests?",
        r"(?i)^references?",
    ]

    def __init__(self, analyzer: OllamaAnalyzer | None = None):
        """
        Initialize the resume adapter.

        Args:
            analyzer: OllamaAnalyzer instance (creates new one if not provided)
        """
        self.analyzer = analyzer or OllamaAnalyzer()

    def parse_sections(self, resume_text: str) -> list[ResumeSection]:
        """
        Parse resume into sections.

        Args:
            resume_text: Full resume text

        Returns:
            List of ResumeSection objects
        """
        sections: list[ResumeSection] = []
        lines = resume_text.split("\n")

        current_section: ResumeSection | None = None
        current_content: list[str] = []

        for i, line in enumerate(lines):
            # Check if this line is a section header
            is_header = False
            header_name = ""

            for pattern in self.SECTION_PATTERNS:
                if re.match(pattern, line.strip()):
                    is_header = True
                    header_name = line.strip()
                    break

            if is_header:
                # Save previous section
                if current_section is not None:
                    current_section.content = "\n".join(current_content).strip()
                    current_section.end_index = i - 1
                    sections.append(current_section)

                # Start new section
                current_section = ResumeSection(
                    name=header_name,
                    content="",
                    start_index=i,
                    end_index=-1,
                )
                current_content = []
            elif current_section is not None:
                current_content.append(line)
            elif line.strip():
                # Content before first section (usually contact info)
                if not sections or sections[0].name != "Header":
                    sections.insert(
                        0,
                        ResumeSection(
                            name="Header",
                            content=line,
                            start_index=0,
                            end_index=i,
                        ),
                    )
                else:
                    sections[0].content += f"\n{line}"
                    sections[0].end_index = i

        # Don't forget the last section
        if current_section is not None:
            current_section.content = "\n".join(current_content).strip()
            current_section.end_index = len(lines) - 1
            sections.append(current_section)

        return sections

    def adapt_resume(
        self,
        resume_text: str,
        job_text: str,
        suggestions: list[Suggestion] | None = None,
        selected_suggestions: list[int] | None = None,
    ) -> AdaptedResume:
        """
        Create an adapted version of the resume.

        Args:
            resume_text: Original resume text
            job_text: Job description text
            suggestions: List of suggestions to potentially apply
            selected_suggestions: Indices of suggestions to apply (None = all)

        Returns:
            AdaptedResume with original and adapted text
        """
        sections = self.parse_sections(resume_text)
        adapted_sections: list[str] = []
        changes_made: list[str] = []
        preserved_facts: list[str] = []

        # Determine which sections to adapt
        adaptable_sections = {"summary", "objective", "experience", "skills"}

        for section in sections:
            section_type = self._classify_section(section.name)

            if section_type in adaptable_sections and section.content.strip():
                # Adapt this section
                adapted_content = self.analyzer.generate_adapted_section(
                    original_section=section.content,
                    job_context=job_text,
                    section_type=section_type,
                )

                if adapted_content != section.content:
                    changes_made.append(f"Adapted {section.name} section")
                    adapted_sections.append(f"{section.name}\n{adapted_content}")
                else:
                    adapted_sections.append(f"{section.name}\n{section.content}")
            else:
                # Preserve unchanged
                if section.name != "Header":
                    adapted_sections.append(f"{section.name}\n{section.content}")
                else:
                    adapted_sections.append(section.content)
                preserved_facts.append(f"{section.name}: preserved as-is")

        # Apply specific suggestions if provided
        adapted_text = "\n\n".join(adapted_sections)

        if suggestions and selected_suggestions:
            for idx in selected_suggestions:
                if 0 <= idx < len(suggestions):
                    suggestion = suggestions[idx]
                    if suggestion.original_text and suggestion.suggested_text:
                        if suggestion.original_text in adapted_text:
                            adapted_text = adapted_text.replace(
                                suggestion.original_text,
                                suggestion.suggested_text,
                                1,
                            )
                            changes_made.append(f"Applied: {suggestion.category} suggestion")

        return AdaptedResume(
            original_text=resume_text,
            adapted_text=adapted_text,
            changes_made=changes_made,
            preserved_facts=preserved_facts,
        )

    def _classify_section(self, section_name: str) -> str:
        """Classify a section name into a standard type."""
        name_lower = section_name.lower()

        if "summary" in name_lower or "objective" in name_lower:
            return "summary"
        elif "experience" in name_lower or "employment" in name_lower or "work" in name_lower:
            return "experience"
        elif "skill" in name_lower:
            return "skills"
        elif "education" in name_lower:
            return "education"
        elif "certification" in name_lower:
            return "certifications"
        elif "project" in name_lower:
            return "projects"
        else:
            return "other"

    def generate_full_adapted_resume(
        self,
        resume_text: str,
        job_text: str,
        embedding_gaps: list[str] | None = None,
        missing_keywords: list[str] | None = None,
    ) -> AdaptedResume:
        """
        Generate a fully adapted resume using LLM.

        This method sends the entire resume for comprehensive adaptation
        rather than section-by-section.

        Args:
            resume_text: Original resume text
            job_text: Job description
            embedding_gaps: Semantic gaps from embedding analysis
            missing_keywords: Keywords missing from resume

        Returns:
            AdaptedResume with comprehensive changes
        """
        try:
            # Use the new optimized method if available
            if hasattr(self.analyzer, "generate_optimized_resume"):
                adapted_text = self.analyzer.generate_optimized_resume(
                    resume_text=resume_text,
                    job_text=job_text,
                    embedding_gaps=embedding_gaps,
                    missing_keywords=missing_keywords,
                )
            else:
                # Fallback to old method
                prompt = f"""Adapt this resume to better match the job description.

ORIGINAL RESUME:
{resume_text}

JOB DESCRIPTION:
{job_text[:2500]}

ADAPTATION RULES:
1. PRESERVE all factual information: names, dates, companies, titles, degrees
2. PRESERVE all numerical achievements and metrics
3. REPHRASE descriptions to use relevant keywords from the job
4. EMPHASIZE transferable skills that match job requirements
5. IMPROVE action verbs to be more impactful and relevant
6. ENSURE ATS-friendly formatting
7. DO NOT add any skills, experiences, or qualifications not in the original

Return the complete adapted resume text."""

                adapted_text = self.analyzer._call_ollama(prompt)

            # Extract and verify facts from original
            preserved_facts = self._extract_facts(resume_text)
            verification = self._verify_facts_preserved(resume_text, adapted_text)

            changes_made = ["Full resume optimization for job alignment"]
            if verification["warnings"]:
                changes_made.extend(verification["warnings"])

            return AdaptedResume(
                original_text=resume_text,
                adapted_text=adapted_text.strip(),
                changes_made=changes_made,
                preserved_facts=preserved_facts + verification.get("verified", []),
            )
        except Exception as e:
            logger.error(f"Error generating adapted resume: {e}")
            return AdaptedResume(
                original_text=resume_text,
                adapted_text=resume_text,
                changes_made=[f"Adaptation failed: {e}"],
                preserved_facts=[],
            )

    def _verify_facts_preserved(self, original: str, adapted: str) -> dict:
        """Verify that key facts from original are preserved in adapted version."""
        result = {"verified": [], "warnings": []}

        # Extract company names (capitalized multi-word phrases)
        original.lower()
        adapted.lower()

        # Check years are preserved
        original_years = set(re.findall(r"\b(19|20)\d{2}\b", original))
        adapted_years = set(re.findall(r"\b(19|20)\d{2}\b", adapted))

        missing_years = original_years - adapted_years
        if missing_years:
            result["warnings"].append(f"⚠️ Years possibly removed: {', '.join(missing_years)}")
        else:
            result["verified"].append(f"✅ All {len(original_years)} years preserved")

        # Check percentages/metrics preserved
        original_metrics = set(re.findall(r"\d+(?:\.\d+)?%", original))
        adapted_metrics = set(re.findall(r"\d+(?:\.\d+)?%", adapted))

        missing_metrics = original_metrics - adapted_metrics
        if missing_metrics:
            result["warnings"].append(f"⚠️ Metrics possibly removed: {', '.join(missing_metrics)}")
        else:
            if original_metrics:
                result["verified"].append(
                    f"✅ All {len(original_metrics)} percentage metrics preserved"
                )

        # Check for potentially added content (new years not in original)
        added_years = adapted_years - original_years
        if added_years:
            result["warnings"].append(f"⚠️ Review: New years appeared: {', '.join(added_years)}")

        return result

    def _extract_facts(self, text: str) -> list[str]:
        """Extract key facts from resume text for verification."""
        facts: list[str] = []

        # Extract dates (years)
        years = re.findall(r"\b(19|20)\d{2}\b", text)
        if years:
            facts.append(f"Years mentioned: {', '.join(set(years))}")

        # Extract emails
        emails = re.findall(r"\b[\w.-]+@[\w.-]+\.\w+\b", text)
        if emails:
            facts.append(f"Email: {emails[0]}")

        # Extract percentages and numbers
        numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
        if numbers:
            facts.append(f"Numbers/metrics preserved: {len(numbers)} values")

        return facts
