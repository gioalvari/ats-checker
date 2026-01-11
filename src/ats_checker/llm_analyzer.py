"""Ollama LLM integration for resume analysis and suggestions."""

import json
import logging
import re
from typing import Any

import ollama

from ats_checker.config import config
from ats_checker.models import AnalysisResult, Suggestion

logger = logging.getLogger(__name__)


class OllamaAnalyzer:
    """Analyzer using Ollama LLM for resume-job matching and suggestions."""

    def __init__(self, model: str | None = None):
        """
        Initialize the analyzer.

        Args:
            model: Ollama model to use (defaults to config setting)
        """
        self.model = model or config.ollama.default_model
        self._verify_model()

    def _verify_model(self) -> None:
        """Verify the model is available, warn if not."""
        if not config.ollama.is_model_available(self.model):
            available = config.ollama.get_available_models()
            logger.warning(
                f"Model '{self.model}' may not be available. "
                f"Available models: {available}. "
                f"Run 'ollama pull {self.model}' to download."
            )

    def _call_ollama(self, prompt: str, system_prompt: str | None = None) -> str:
        """Make a call to Ollama API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": config.ollama.temperature,
                    "num_predict": config.ollama.max_tokens,
                },
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise RuntimeError(f"Failed to get response from Ollama: {e}") from e

    def analyze_resume(
        self,
        resume_text: str,
        job_text: str,
        keyword_score: float,
        missing_keywords: list[str],
        matching_keywords: list[str],
    ) -> AnalysisResult:
        """
        Perform comprehensive resume analysis using LLM.

        Args:
            resume_text: Full resume text
            job_text: Full job description text
            keyword_score: Pre-calculated keyword match score
            missing_keywords: Keywords in job but not in resume
            matching_keywords: Keywords found in both

        Returns:
            Complete AnalysisResult with LLM-enhanced insights
        """
        system_prompt = """You are an expert ATS (Applicant Tracking System) analyst and career coach.
Your task is to analyze resumes against job descriptions and provide actionable feedback.
Always be specific, constructive, and focus on improvements that will help pass ATS screening."""

        prompt = f"""Analyze this resume against the job description and provide a detailed assessment.

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{job_text[:2000]}

KEYWORD ANALYSIS:
- Keyword Match Score: {keyword_score:.1f}%
- Matching Keywords: {", ".join(matching_keywords[:20])}
- Missing Keywords: {", ".join(missing_keywords[:20])}

Please provide your analysis in the following JSON format:
{{
    "overall_score": <0-100 integer>,
    "summary": "<2-3 sentence summary of the match quality>",
    "strengths": ["<strength 1>", "<strength 2>", ...],
    "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
    "suggestions": [
        {{
            "category": "<skills|experience|keywords|formatting>",
            "original_text": "<relevant text from resume or null>",
            "suggested_text": "<specific suggestion>",
            "reasoning": "<why this helps>",
            "priority": "<high|medium|low>"
        }}
    ]
}}

Focus on:
1. ATS compatibility and keyword optimization
2. Quantifiable achievements and impact
3. Skills alignment with job requirements
4. Experience relevance
5. Formatting and clarity improvements

Provide 5-10 specific, actionable suggestions."""

        try:
            response = self._call_ollama(prompt, system_prompt)
            parsed = self._parse_analysis_response(response)

            return AnalysisResult(
                match_score=parsed.get("overall_score", keyword_score),
                keyword_score=keyword_score,
                missing_keywords=missing_keywords,
                matching_keywords=matching_keywords,
                extra_keywords=[],
                suggestions=[
                    Suggestion(**s) for s in parsed.get("suggestions", [])[: config.max_suggestions]
                ],
                summary=parsed.get("summary", "Analysis completed."),
                strengths=parsed.get("strengths", []),
                weaknesses=parsed.get("weaknesses", []),
            )
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            # Return basic result without LLM insights
            return AnalysisResult(
                match_score=keyword_score,
                keyword_score=keyword_score,
                missing_keywords=missing_keywords,
                matching_keywords=matching_keywords,
                extra_keywords=[],
                suggestions=[],
                summary=f"Basic analysis completed. LLM analysis unavailable: {e}",
                strengths=[],
                weaknesses=[],
            )

    def _parse_analysis_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response into structured data."""
        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse the whole response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM response as JSON")
            return {"summary": response[:500]}

    def get_suggestions(
        self, resume_text: str, job_text: str, focus_areas: list[str] | None = None
    ) -> list[Suggestion]:
        """
        Get specific suggestions for improving the resume.

        Args:
            resume_text: Resume content
            job_text: Job description
            focus_areas: Specific areas to focus on

        Returns:
            List of suggestions
        """
        focus = ", ".join(focus_areas) if focus_areas else "all areas"

        prompt = f"""Provide specific suggestions to improve this resume for the given job.
Focus on: {focus}

RESUME:
{resume_text[:3000]}

JOB:
{job_text[:2000]}

Return a JSON array of suggestions:
[
    {{
        "category": "<skills|experience|keywords|formatting>",
        "original_text": "<text to change or null>",
        "suggested_text": "<specific improvement>",
        "reasoning": "<why this helps with ATS>",
        "priority": "<high|medium|low>"
    }}
]

Provide 5-8 specific, actionable suggestions that maintain factual accuracy."""

        try:
            response = self._call_ollama(prompt)
            suggestions_data = self._parse_suggestions_response(response)
            return [Suggestion(**s) for s in suggestions_data[: config.max_suggestions]]
        except Exception as e:
            logger.error(f"Error getting suggestions: {e}")
            return []

    def _parse_suggestions_response(self, response: str) -> list[dict[str, Any]]:
        """Parse suggestions from LLM response."""
        # Try to extract JSON array
        json_match = re.search(r"\[[\s\S]*\]", response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return []

    def generate_adapted_section(
        self,
        original_section: str,
        job_context: str,
        section_type: str = "experience",
    ) -> str:
        """
        Generate an adapted version of a resume section.

        Args:
            original_section: Original resume section text
            job_context: Relevant job description context
            section_type: Type of section (experience, skills, summary, etc.)

        Returns:
            Adapted section text
        """
        prompt = f"""Adapt this {section_type} section to better match the job while STRICTLY preserving all factual information.

ORIGINAL {section_type.upper()}:
{original_section}

JOB CONTEXT:
{job_context[:1500]}

STRICT RULES - VIOLATIONS ARE NOT ACCEPTABLE:
1. ❌ NEVER invent, add, or fabricate ANY facts, dates, company names, job titles, or achievements
2. ❌ NEVER add skills, certifications, or experiences not explicitly present in the original
3. ❌ NEVER change numerical values, percentages, or metrics
4. ❌ NEVER modify company names, dates, locations, or job titles
5. ✅ ONLY rephrase existing content to better highlight relevant keywords
6. ✅ ONLY improve sentence structure and action verbs
7. ✅ ONLY reorganize bullet points to prioritize most relevant experience first
8. ✅ ONLY add industry keywords that describe EXISTING skills in the original

WHAT YOU CAN DO:
- Change "built ML models" to "designed and deployed machine learning models" (if they built ML models)
- Reorder bullet points to put job-relevant ones first
- Replace weak verbs with strong action verbs (managed → spearheaded, helped → facilitated)
- Add implicit keywords (if they mention "Python scripts", you can say "Python development")

Return ONLY the adapted text, no explanations."""

        try:
            response = self._call_ollama(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Error adapting section: {e}")
            return original_section

    def generate_optimized_resume(
        self,
        resume_text: str,
        job_text: str,
        embedding_gaps: list[str] | None = None,
        missing_keywords: list[str] | None = None,
    ) -> str:
        """
        Generate a fully optimized resume preserving all real experience.

        Args:
            resume_text: Original resume text
            job_text: Job description
            embedding_gaps: Semantic gaps identified by embedding analysis
            missing_keywords: Keywords missing from resume

        Returns:
            Optimized resume text with real facts preserved
        """
        gaps_info = ""
        if embedding_gaps:
            gaps_info = f"\nSEMANTIC GAPS TO ADDRESS (if the experience exists in resume):\n{chr(10).join(f'- {g}' for g in embedding_gaps[:5])}"

        keywords_info = ""
        if missing_keywords:
            keywords_info = f"\nMISSING KEYWORDS (incorporate ONLY if skill actually exists):\n{', '.join(missing_keywords[:15])}"

        prompt = f"""You are an expert resume writer. Optimize this resume for the job description.

ORIGINAL RESUME:
{resume_text}

JOB DESCRIPTION:
{job_text[:2500]}
{gaps_info}
{keywords_info}

═══════════════════════════════════════════════════════════════════════════════
ABSOLUTE RULES - YOUR OUTPUT WILL BE VERIFIED AGAINST THE ORIGINAL:
═══════════════════════════════════════════════════════════════════════════════

🚫 PROHIBITED ACTIONS (will be rejected):
1. Adding ANY company, job title, or position not in the original
2. Adding ANY degree, certification, or education not in the original
3. Adding ANY skill or technology not mentioned or implied in the original
4. Changing dates, durations, or numerical achievements
5. Inventing metrics, percentages, or results (like "$2M savings")
6. Adding awards, publications, or projects that don't exist
7. Changing job titles (Senior → Director, Engineer → Lead)

✅ PERMITTED ACTIONS:
1. Rewriting bullet points with stronger action verbs
2. Reordering content to prioritize job-relevant experience
3. Adding keywords that describe EXISTING stated skills differently
4. Improving sentence structure and clarity
5. Making implicit skills explicit (e.g., "used SQL" → "SQL database querying")
6. Adding a Summary/Objective if one doesn't exist (based ONLY on stated experience)
7. Improving formatting for ATS compatibility

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT:
═══════════════════════════════════════════════════════════════════════════════

Return the COMPLETE optimized resume in clean text format.
Keep the same overall structure. Use clear section headers.
Every fact in your output MUST be traceable to the original resume.

START YOUR RESPONSE WITH THE OPTIMIZED RESUME:"""

        try:
            response = self._call_ollama(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating optimized resume: {e}")
            return resume_text
