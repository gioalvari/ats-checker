"""Semantic embedding analyzer for resume-job matching using Ollama embeddings."""

import logging
import re
from dataclasses import dataclass

import numpy as np
import ollama
from ollama import Client

from .config import OllamaConfig

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding-based analysis."""

    overall_similarity: float  # 0-100 score
    section_scores: dict[str, float]  # Scores per resume section
    semantic_gaps: list[str]  # Concepts in JD but not semantically matched in resume
    strong_matches: list[str]  # Well-matched concepts
    embedding_model: str
    keyword_overlap: float  # 0-100 keyword overlap score
    domain_match: bool  # Whether domains align
    adjusted_score: float  # Final score after adjustments


# Domain keywords for mismatch detection
DOMAIN_KEYWORDS = {
    "data_science": [
        "machine learning",
        "data science",
        "ml",
        "ai",
        "deep learning",
        "neural network",
        "nlp",
        "computer vision",
        "statistics",
        "modeling",
        "prediction",
        "forecasting",
        "pandas",
        "scikit-learn",
        "tensorflow",
        "pytorch",
        "spark",
        "analytics",
    ],
    "software_engineering": [
        "software engineer",
        "backend",
        "frontend",
        "full stack",
        "api",
        "microservices",
        "rest",
        "graphql",
        "database",
        "devops",
        "ci/cd",
        "agile",
        "scrum",
    ],
    "web_development": [
        "react",
        "angular",
        "vue",
        "javascript",
        "typescript",
        "html",
        "css",
        "node.js",
        "frontend",
        "web developer",
        "ui",
        "ux",
    ],
    "marketing": [
        "marketing",
        "campaign",
        "brand",
        "advertising",
        "seo",
        "content",
        "social media",
        "digital marketing",
        "lead generation",
        "crm",
    ],
    "finance": [
        "finance",
        "accounting",
        "investment",
        "trading",
        "risk",
        "portfolio",
        "banking",
        "financial analysis",
        "audit",
        "compliance",
    ],
    "engineering_other": [
        "mechanical",
        "electrical",
        "civil",
        "chemical",
        "manufacturing",
        "cad",
        "autocad",
        "solidworks",
        "engineering design",
    ],
}


class EmbeddingAnalyzer:
    """
    Analyzes semantic similarity between resume and job description using embeddings.

    This provides a more accurate matching than TF-IDF by understanding:
    - Synonyms (ML = Machine Learning = AI)
    - Related concepts (Python → programming → coding)
    - Contextual meaning
    """

    # Common resume sections to analyze separately
    RESUME_SECTIONS = [
        "experience",
        "education",
        "skills",
        "projects",
        "summary",
        "certifications",
    ]

    def __init__(
        self,
        embedding_model: str = "nomic-embed-text",
        config: OllamaConfig | None = None,
    ):
        """
        Initialize the embedding analyzer.

        Args:
            embedding_model: Ollama model for embeddings (nomic-embed-text recommended)
            config: Ollama configuration
        """
        self.config = config or OllamaConfig()
        self.embedding_model = embedding_model
        self._embedding_cache: dict[int, list[float]] = {}

    def _detect_domain(self, text: str) -> str | None:
        """Detect the primary domain of a text based on keyword density."""
        text_lower = text.lower()
        domain_scores: dict[str, int] = {}

        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                domain_scores[domain] = score

        if not domain_scores:
            return None

        return max(domain_scores, key=domain_scores.get)

    def _calculate_keyword_overlap(
        self, resume: str, job: str
    ) -> tuple[float, list[str], list[str]]:
        """
        Calculate keyword overlap between resume and job description.

        Returns:
            Tuple of (overlap_percentage, matched_keywords, missing_keywords)
        """
        # Extract technical keywords from job description
        job_lower = job.lower()
        resume_lower = resume.lower()

        # Common technical skills to look for
        tech_keywords = [
            # Programming languages
            "python",
            "java",
            "scala",
            "r",
            "sql",
            "javascript",
            "typescript",
            "c++",
            "go",
            "rust",
            # ML/Data
            "machine learning",
            "deep learning",
            "nlp",
            "computer vision",
            "pytorch",
            "tensorflow",
            "scikit-learn",
            "pandas",
            "numpy",
            "spark",
            "hadoop",
            "kafka",
            "airflow",
            # Cloud
            "aws",
            "gcp",
            "azure",
            "sagemaker",
            "s3",
            "ec2",
            "lambda",
            "kubernetes",
            "docker",
            # Tools
            "git",
            "mlops",
            "ci/cd",
            "jenkins",
            "terraform",
            "mlflow",
            "dvc",
            # Concepts
            "forecasting",
            "time series",
            "classification",
            "regression",
            "clustering",
            "a/b testing",
            "statistics",
            "experimentation",
            "etl",
            "data pipeline",
        ]

        # Find keywords in job description
        job_keywords = [kw for kw in tech_keywords if kw in job_lower]

        if not job_keywords:
            return 100.0, [], []  # No specific keywords to match

        # Check which are in resume
        matched = [kw for kw in job_keywords if kw in resume_lower]
        missing = [kw for kw in job_keywords if kw not in resume_lower]

        overlap = len(matched) / len(job_keywords) * 100
        return overlap, matched, missing

    def _check_seniority_match(self, resume: str, job: str) -> tuple[bool, str]:
        """
        Check if seniority levels match between resume and job.

        Returns:
            Tuple of (is_match, explanation)
        """
        resume_lower = resume.lower()
        job_lower = job.lower()

        # Extract years of experience from resume
        resume_years = 0
        years_patterns = [
            r"(\d+)\+?\s*years?\s*(?:of\s*)?experience",
            r"experience.*?(\d+)\+?\s*years?",
            r"(\d{4})\s*[-–]\s*(?:present|current|\d{4})",
        ]

        for pattern in years_patterns:
            matches = re.findall(pattern, resume_lower)
            if matches:
                for match in matches:
                    try:
                        year = int(match)
                        if year > 1990:  # It's a year range
                            resume_years = max(resume_years, 2025 - year)
                        else:
                            resume_years = max(resume_years, year)
                    except ValueError:
                        continue

        # Extract required years from job
        job_years = 0
        job_years_match = re.search(r"(\d+)\+?\s*years?", job_lower)
        if job_years_match:
            job_years = int(job_years_match.group(1))

        # Check seniority keywords
        junior_keywords = [
            "junior",
            "entry level",
            "entry-level",
            "graduate",
            "intern",
            "0-2 years",
        ]
        senior_keywords = [
            "senior",
            "lead",
            "principal",
            "staff",
            "architect",
            "director",
            "8+ years",
            "10+ years",
        ]

        resume_is_junior = any(kw in resume_lower for kw in junior_keywords)
        resume_is_senior = any(kw in resume_lower for kw in senior_keywords)
        job_is_junior = any(kw in job_lower for kw in junior_keywords)
        job_is_senior = any(kw in job_lower for kw in senior_keywords)

        # Mismatch cases
        if job_is_senior and resume_is_junior:
            return False, "Junior candidate for senior role"
        if job_is_junior and resume_is_senior:
            return False, "Overqualified for junior role"
        if job_years > 0 and resume_years > 0:
            if resume_years < job_years * 0.6:  # Less than 60% of required experience
                return False, f"Experience gap: {resume_years} years vs {job_years}+ required"

        return True, "Seniority match OK"

    def get_embedding(self, text: str, max_chars: int = 2000) -> list[float]:
        """
        Get embedding vector for text using Ollama.

        Args:
            text: Text to embed
            max_chars: Maximum characters to embed (nomic-embed-text has token limits)

        Returns:
            Embedding vector as list of floats
        """
        # Truncate text if needed (nomic-embed-text has ~8192 token limit, ~2000 chars safe)
        truncated_text = text[:max_chars] if len(text) > max_chars else text

        # Check cache first
        cache_key = hash(truncated_text[:500])  # Use first 500 chars for cache key
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        try:
            # Use ollama module directly instead of Client
            response = ollama.embed(
                model=self.embedding_model,
                input=truncated_text,
            )

            # Handle different response formats
            if hasattr(response, "embeddings"):
                embedding = response.embeddings[0]
            elif isinstance(response, dict) and "embeddings" in response:
                embedding = response["embeddings"][0]
            elif isinstance(response, dict) and "embedding" in response:
                embedding = response["embedding"]
            else:
                raise ValueError(f"Unexpected embedding response format: {type(response)}")

            # Cache the result
            self._embedding_cache[cache_key] = embedding
            return embedding

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score between -1 and 1
        """
        a = np.array(vec1)
        b = np.array(vec2)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def extract_sections(self, text: str) -> dict[str, str]:
        """
        Extract sections from resume text.

        Args:
            text: Full resume text

        Returns:
            Dictionary of section name to content
        """
        sections: dict[str, str] = {}
        text_lower = text.lower()

        # Define section keywords
        section_keywords = {
            "experience": ["experience", "work history", "employment", "professional background"],
            "education": ["education", "academic", "degree", "university", "school"],
            "skills": ["skills", "technologies", "tools", "competencies", "technical skills"],
            "projects": ["projects", "portfolio", "personal projects"],
            "summary": ["summary", "objective", "profile", "about me", "introduction"],
            "certifications": ["certifications", "certificates", "credentials", "licenses"],
        }

        # Find section positions
        section_positions: list[tuple[str, int]] = []
        for section_name, keywords in section_keywords.items():
            for keyword in keywords:
                pos = text_lower.find(keyword)
                if pos != -1:
                    section_positions.append((section_name, pos))
                    break

        # Sort by position
        section_positions.sort(key=lambda x: x[1])

        # Extract content between sections
        for i, (section_name, start_pos) in enumerate(section_positions):
            end_pos = section_positions[i + 1][1] if i + 1 < len(section_positions) else len(text)

            content = text[start_pos:end_pos].strip()
            if content:
                sections[section_name] = content

        # If no sections found, treat entire text as content
        if not sections:
            sections["full_resume"] = text

        return sections

    def extract_key_requirements(self, job_description: str) -> list[str]:
        """
        Extract key requirements/concepts from job description.

        Args:
            job_description: Job description text

        Returns:
            List of key requirement phrases
        """
        # Split into sentences/phrases
        import re

        # Split on common delimiters
        phrases = re.split(r"[.•\n;]", job_description)

        # Filter to meaningful phrases
        requirements = []
        for phrase in phrases:
            phrase = phrase.strip()
            # Keep phrases with 3-50 words that contain relevant content
            word_count = len(phrase.split())
            if 3 <= word_count <= 50:
                # Skip generic phrases
                generic_words = ["we are", "you will", "the company", "our team", "apply now"]
                if not any(g in phrase.lower() for g in generic_words):
                    requirements.append(phrase)

        return requirements[:20]  # Limit to top 20 requirements

    def analyze(
        self,
        resume_text: str,
        job_description: str,
    ) -> EmbeddingResult:
        """
        Perform semantic analysis of resume against job description.

        Uses a hybrid approach:
        1. Embedding similarity for semantic understanding
        2. Keyword overlap for technical skill matching
        3. Domain detection for obvious mismatches
        4. Seniority matching

        Args:
            resume_text: Full resume text
            job_description: Job description text

        Returns:
            EmbeddingResult with similarity scores and analysis
        """
        logger.info(f"Analyzing with embedding model: {self.embedding_model}")

        # 1. Get overall embeddings
        resume_embedding = self.get_embedding(resume_text)
        jd_embedding = self.get_embedding(job_description)

        # Calculate raw embedding similarity
        overall_sim = self.cosine_similarity(resume_embedding, jd_embedding)
        raw_score = max(0, min(100, (overall_sim + 1) / 2 * 100))  # Map [-1,1] to [0,100]

        # 2. Domain matching
        resume_domain = self._detect_domain(resume_text)
        job_domain = self._detect_domain(job_description)
        domain_match = (
            (resume_domain == job_domain) or (resume_domain is None) or (job_domain is None)
        )

        # 3. Keyword overlap
        keyword_overlap, matched_kw, missing_kw = self._calculate_keyword_overlap(
            resume_text, job_description
        )

        # 4. Seniority check
        seniority_match, seniority_reason = self._check_seniority_match(
            resume_text, job_description
        )

        # 5. Calculate adjusted score with penalties
        adjusted_score = raw_score

        # Penalty for domain mismatch (up to -40 points)
        if not domain_match and resume_domain and job_domain:
            domain_penalty = 40
            adjusted_score -= domain_penalty
            logger.info(
                f"Domain mismatch penalty: -{domain_penalty} ({resume_domain} vs {job_domain})"
            )

        # Penalty for low keyword overlap (up to -25 points)
        if keyword_overlap < 30:
            keyword_penalty = 25 * (1 - keyword_overlap / 30)
            adjusted_score -= keyword_penalty
            logger.info(
                f"Low keyword overlap penalty: -{keyword_penalty:.1f} ({keyword_overlap:.0f}%)"
            )

        # Penalty for seniority mismatch (up to -20 points)
        if not seniority_match:
            seniority_penalty = 20
            adjusted_score -= seniority_penalty
            logger.info(f"Seniority mismatch penalty: -{seniority_penalty} ({seniority_reason})")

        # Ensure score is in valid range
        adjusted_score = max(0, min(100, adjusted_score))

        # 6. Analyze sections
        sections = self.extract_sections(resume_text)
        section_scores: dict[str, float] = {}

        for section_name, section_content in sections.items():
            if len(section_content) > 50:  # Only analyze substantial sections
                section_emb = self.get_embedding(section_content)
                sim = self.cosine_similarity(section_emb, jd_embedding)
                section_scores[section_name] = max(0, min(100, (sim + 1) / 2 * 100))

        # 7. Analyze specific requirements for gaps
        requirements = self.extract_key_requirements(job_description)
        semantic_gaps: list[str] = []
        strong_matches: list[str] = []

        for req in requirements:
            req_emb = self.get_embedding(req)
            req_sim = self.cosine_similarity(req_emb, resume_embedding)

            if req_sim > 0.5:  # Good semantic match
                strong_matches.append(req)
            elif req_sim < 0.3:  # Poor match - potential gap
                semantic_gaps.append(req)

        # Add missing keywords to gaps
        for kw in missing_kw[:5]:
            if kw not in " ".join(semantic_gaps).lower():
                semantic_gaps.append(f"Missing skill: {kw}")

        return EmbeddingResult(
            overall_similarity=round(raw_score, 1),
            section_scores={k: round(v, 1) for k, v in section_scores.items()},
            semantic_gaps=semantic_gaps[:8],
            strong_matches=strong_matches[:5],
            embedding_model=self.embedding_model,
            keyword_overlap=round(keyword_overlap, 1),
            domain_match=domain_match,
            adjusted_score=round(adjusted_score, 1),
        )

    def get_improvement_suggestions(
        self,
        resume_text: str,
        job_description: str,
        result: EmbeddingResult,
    ) -> list[str]:
        """
        Generate improvement suggestions based on embedding analysis.

        Args:
            resume_text: Resume text
            job_description: Job description text
            result: EmbeddingResult from analysis

        Returns:
            List of actionable suggestions
        """
        suggestions: list[str] = []

        # Overall score suggestions
        if result.overall_similarity < 50:
            suggestions.append(
                "🔴 Low semantic match. Consider rewriting your resume to use similar "
                "language and terminology as the job description."
            )
        elif result.overall_similarity < 70:
            suggestions.append(
                "🟡 Moderate match. Focus on adding keywords and experiences that "
                "directly relate to the job requirements."
            )

        # Section-specific suggestions
        if result.section_scores:
            weakest_section = min(result.section_scores, key=result.section_scores.get)
            weakest_score = result.section_scores[weakest_section]
            if weakest_score < 60:
                suggestions.append(
                    f"📝 Your '{weakest_section}' section has low relevance ({weakest_score:.0f}%). "
                    f"Consider enhancing it with more relevant content."
                )

        # Gap-based suggestions
        if result.semantic_gaps:
            gaps_text = ", ".join(
                f'"{g[:50]}..."' if len(g) > 50 else f'"{g}"' for g in result.semantic_gaps[:3]
            )
            suggestions.append(
                f"⚠️ Missing concepts from JD: {gaps_text}. "
                "Consider addressing these in your resume."
            )

        # Strength-based feedback
        if result.strong_matches:
            matches_text = ", ".join(
                f'"{m[:40]}..."' if len(m) > 40 else f'"{m}"' for m in result.strong_matches[:3]
            )
            suggestions.append(f"✅ Strong matches: {matches_text}. Keep these prominent!")

        return suggestions


def get_available_embedding_models() -> list[str]:
    """Get list of available Ollama models suitable for embeddings."""
    embedding_models = [
        "nomic-embed-text",
        "mxbai-embed-large",
        "all-minilm",
        "snowflake-arctic-embed",
    ]

    try:
        client = Client()
        response = client.list()

        # Get installed model names
        installed: set[str] = set()
        if hasattr(response, "models"):
            for model in response.models:
                name = model.model if hasattr(model, "model") else str(model.get("name", ""))
                # Remove tag suffix
                base_name = name.split(":")[0]
                installed.add(base_name)

        # Return only installed embedding models
        available = [m for m in embedding_models if m in installed]
        return available if available else ["nomic-embed-text"]  # Default

    except Exception as e:
        logger.warning(f"Could not check available models: {e}")
        return ["nomic-embed-text"]
