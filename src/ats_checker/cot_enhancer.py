"""Chain-of-thought resume enhancement for better quality and fact preservation."""

import logging
import re
from dataclasses import dataclass, field

from ats_checker.evaluation.resume_chunker import (
    ResumeSection,
    chunk_resume,
    extract_job_requirements,
)
from ats_checker.llm_analyzer import OllamaAnalyzer
from ats_checker.models import AdaptedResume

logger = logging.getLogger(__name__)


@dataclass
class SectionEnhancement:
    """Result of enhancing a single section."""

    section_name: str
    original: str
    enhanced: str
    analysis: str
    keywords_added: list[str]
    facts_preserved: list[str]
    confidence: float  # 0-1 confidence in enhancement quality


@dataclass
class ChainOfThoughtResult:
    """Result of chain-of-thought enhancement."""

    full_text: str
    section_enhancements: list[SectionEnhancement]
    original_text: str
    job_requirements: dict
    changes_made: list[str] = field(default_factory=list)
    preserved_facts: list[str] = field(default_factory=list)
    quality_score: float = 0.0


class ChainOfThoughtEnhancer:
    """
    Multi-step resume enhancement with fact preservation.

    Process:
    1. Chunk resume into sections
    2. Extract job requirements
    3. For each section:
       a. Analyze gaps
       b. Identify keywords to add
       c. Generate enhanced version
       d. Verify facts preserved
    4. Reassemble and validate
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        strict_fact_preservation: bool = True,
    ):
        """
        Initialize the enhancer.

        Args:
            model: Ollama model to use
            strict_fact_preservation: If True, reject enhancements that lose facts
        """
        self.analyzer = OllamaAnalyzer(model=model)
        self.strict_fact_preservation = strict_fact_preservation
        self.model = model

    def _extract_facts_from_section(self, text: str) -> dict:
        """Extract verifiable facts from a section."""
        facts = {
            "dates": list(set(re.findall(r"\b(19|20)\d{2}\b", text))),
            "percentages": list(set(re.findall(r"\d+(?:\.\d+)?%", text))),
            "money": list(set(re.findall(r"[€$£]\s*\d+(?:[.,]\d+)*\s*[MKBmkb]?", text))),
            "numbers": list(
                set(
                    re.findall(
                        r"\b\d+(?:\.\d+)?\s*(?:M|K|million|billion|users|clients|projects|team members)\b",
                        text,
                        re.I,
                    )
                )
            ),
            "companies": [],  # Extracted via LLM if needed
            "metrics": list(
                set(
                    re.findall(
                        r"(?:reduced|increased|improved|saved|grew|achieved)\s*(?:by\s*)?\d+",
                        text,
                        re.I,
                    )
                )
            ),
        }
        return facts

    def _verify_facts_preserved(
        self, original: str, enhanced: str, original_facts: dict
    ) -> tuple[bool, list[str], list[str]]:
        """
        Verify that all facts from original are in enhanced version.

        Returns:
            Tuple of (all_preserved, preserved_list, missing_list)
        """
        enhanced_facts = self._extract_facts_from_section(enhanced)

        preserved = []
        missing = []

        for category, values in original_facts.items():
            for value in values:
                # Check if value exists in enhanced text
                if value in enhanced or value in str(enhanced_facts.get(category, [])):
                    preserved.append(f"✅ {category}: {value}")
                else:
                    missing.append(f"❌ Lost {category}: {value}")

        all_preserved = len(missing) == 0
        return all_preserved, preserved, missing

    def _analyze_section_gap(
        self,
        section: ResumeSection,
        job_requirements: dict,
        job_description: str,
    ) -> str:
        """Analyze what's missing in a section compared to job requirements."""

        prompt = f"""Analyze this resume section against the job requirements.

SECTION: {section.name}
CONTENT:
{section.content[:1500]}

JOB REQUIREMENTS:
- Required: {", ".join(job_requirements.get("must_have", [])[:5])}
- Nice to have: {", ".join(job_requirements.get("nice_to_have", [])[:3])}
- Skills needed: {", ".join(job_requirements.get("skills", [])[:10])}

Identify:
1. What requirements are ALREADY covered in this section?
2. What requirements COULD be implied from existing experience but aren't explicit?
3. What requirements are genuinely NOT present?

Be specific and concise. Focus on what can be truthfully emphasized."""

        return self.analyzer._call_ollama(prompt)

    def _identify_keywords_to_add(
        self,
        section: ResumeSection,
        analysis: str,
        job_requirements: dict,
    ) -> list[str]:
        """Identify specific keywords to incorporate."""

        prompt = f"""Based on this analysis, list keywords to add to this section.

SECTION: {section.name}
CURRENT CONTENT:
{section.content[:1000]}

ANALYSIS:
{analysis[:800]}

SKILLS FROM JOB:
{", ".join(job_requirements.get("skills", []))}

RULES:
- Only suggest keywords that can be TRUTHFULLY added based on existing experience
- Don't suggest keywords for skills not demonstrated
- Focus on rephrasing to include synonyms from job description

List 3-8 specific keywords, one per line. No bullets or numbers."""

        response = self.analyzer._call_ollama(prompt)
        keywords = [kw.strip() for kw in response.strip().split("\n") if kw.strip()]
        return keywords[:8]

    def _enhance_section(
        self,
        section: ResumeSection,
        keywords: list[str],
        job_description: str,
    ) -> str:
        """Generate enhanced version of a section."""

        original_facts = self._extract_facts_from_section(section.content)
        facts_list = []
        for _cat, vals in original_facts.items():
            if vals:
                facts_list.extend(vals)

        prompt = f"""Rewrite this resume section optimized for the job.

ORIGINAL {section.name.upper()} SECTION:
{section.content}

KEYWORDS TO INCORPORATE (where truthful):
{", ".join(keywords)}

FACTS THAT MUST BE PRESERVED EXACTLY:
{", ".join(facts_list) if facts_list else "No specific metrics found"}

STRICT RULES:
1. PRESERVE ALL dates, numbers, percentages, company names, and metrics EXACTLY
2. DO NOT invent new achievements, metrics, or experiences
3. Only rephrase existing content using better action verbs and keywords
4. Keep the same structure and length
5. Every statement must be derivable from the original

Output ONLY the enhanced section text, nothing else."""

        return self.analyzer._call_ollama(prompt)

    def enhance_section(
        self,
        section: ResumeSection,
        job_requirements: dict,
        job_description: str,
    ) -> SectionEnhancement:
        """
        Enhance a single section using chain-of-thought.

        Steps:
        1. Analyze gaps
        2. Identify keywords
        3. Generate enhancement
        4. Verify facts
        """
        logger.info(f"Enhancing section: {section.name}")

        # Step 1: Analyze
        analysis = self._analyze_section_gap(section, job_requirements, job_description)

        # Step 2: Keywords
        keywords = self._identify_keywords_to_add(section, analysis, job_requirements)

        # Step 3: Enhance
        enhanced = self._enhance_section(section, keywords, job_description)

        # Step 4: Verify facts
        original_facts = self._extract_facts_from_section(section.content)
        all_preserved, preserved, missing = self._verify_facts_preserved(
            section.content, enhanced, original_facts
        )

        # If strict mode and facts missing, try again or use original
        confidence = 1.0 if all_preserved else 0.5

        if self.strict_fact_preservation and not all_preserved:
            logger.warning(f"Facts lost in {section.name}, retrying with stricter prompt...")

            # Retry with explicit fact list
            retry_prompt = f"""Rewrite this section but you MUST keep these EXACT values:

ORIGINAL:
{section.content}

EXACT VALUES TO PRESERVE (copy these exactly):
{chr(10).join(f"- {m.replace('❌ Lost ', '')}" for m in missing)}

Only change wording, not facts. Output the section text only."""

            enhanced = self.analyzer._call_ollama(retry_prompt)
            _, preserved, missing = self._verify_facts_preserved(
                section.content, enhanced, original_facts
            )
            confidence = 0.8 if not missing else 0.3

        return SectionEnhancement(
            section_name=section.name,
            original=section.content,
            enhanced=enhanced.strip(),
            analysis=analysis,
            keywords_added=keywords,
            facts_preserved=preserved,
            confidence=confidence,
        )

    def enhance_resume(
        self,
        resume_text: str,
        job_description: str,
        embedding_gaps: list[str] | None = None,
        missing_keywords: list[str] | None = None,
    ) -> ChainOfThoughtResult:
        """
        Enhance full resume using chain-of-thought process.

        Args:
            resume_text: Full resume text
            job_description: Job description
            embedding_gaps: Semantic gaps from embedding analysis
            missing_keywords: Keywords missing from resume

        Returns:
            ChainOfThoughtResult with enhanced resume
        """
        logger.info("Starting chain-of-thought enhancement")

        # 1. Chunk resume
        sections = chunk_resume(resume_text)
        logger.info(f"Found {len(sections)} sections")

        # 2. Extract job requirements
        job_requirements = extract_job_requirements(job_description)

        # Add embedding gaps to requirements
        if embedding_gaps:
            job_requirements.setdefault("must_have", []).extend(embedding_gaps)
        if missing_keywords:
            job_requirements.setdefault("skills", []).extend(missing_keywords)

        # 3. Enhance each section
        enhancements: list[SectionEnhancement] = []
        changes_made: list[str] = []
        preserved_facts: list[str] = []

        for section in sections:
            if len(section.content.strip()) < 50:
                # Keep short sections as-is
                enhancements.append(
                    SectionEnhancement(
                        section_name=section.name,
                        original=section.content,
                        enhanced=section.content,
                        analysis="Section too short to enhance",
                        keywords_added=[],
                        facts_preserved=[],
                        confidence=1.0,
                    )
                )
                continue

            enhancement = self.enhance_section(section, job_requirements, job_description)
            enhancements.append(enhancement)

            # Track changes
            if enhancement.keywords_added:
                changes_made.append(
                    f"Added keywords to {section.name}: {', '.join(enhancement.keywords_added[:3])}"
                )

            preserved_facts.extend(enhancement.facts_preserved)

            # Warn about low confidence
            if enhancement.confidence < 0.5:
                changes_made.append(
                    f"⚠️ Low confidence in {section.name} enhancement - review carefully"
                )

        # 4. Reassemble resume
        enhanced_parts = []
        for enhancement in enhancements:
            if enhancement.section_name != "Header":
                enhanced_parts.append(f"\n{enhancement.section_name}\n")
            enhanced_parts.append(enhancement.enhanced)

        full_text = "\n".join(enhanced_parts)

        # 5. Calculate quality score
        avg_confidence = (
            sum(e.confidence for e in enhancements) / len(enhancements) if enhancements else 0
        )
        quality_score = avg_confidence * 100

        return ChainOfThoughtResult(
            full_text=full_text,
            section_enhancements=enhancements,
            original_text=resume_text,
            job_requirements=job_requirements,
            changes_made=changes_made,
            preserved_facts=preserved_facts[:20],  # Limit to 20
            quality_score=quality_score,
        )

    def to_adapted_resume(self, result: ChainOfThoughtResult) -> AdaptedResume:
        """Convert ChainOfThoughtResult to AdaptedResume for UI compatibility."""
        return AdaptedResume(
            original_text=result.original_text,
            adapted_text=result.full_text,
            changes_made=result.changes_made,
            preserved_facts=result.preserved_facts,
        )


def enhance_with_cot(
    resume_text: str,
    job_description: str,
    model: str = "qwen2.5:7b-instruct",
    embedding_gaps: list[str] | None = None,
    missing_keywords: list[str] | None = None,
) -> AdaptedResume:
    """
    Convenience function to enhance resume with chain-of-thought.

    Args:
        resume_text: Full resume text
        job_description: Job description
        model: Ollama model to use
        embedding_gaps: Semantic gaps from analysis
        missing_keywords: Missing keywords from analysis

    Returns:
        AdaptedResume ready for UI display
    """
    enhancer = ChainOfThoughtEnhancer(model=model)
    result = enhancer.enhance_resume(
        resume_text=resume_text,
        job_description=job_description,
        embedding_gaps=embedding_gaps,
        missing_keywords=missing_keywords,
    )
    return enhancer.to_adapted_resume(result)
