"""Keyword extraction and similarity calculation using regex and TF-IDF."""

import logging
import re
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Common English stop words
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "do",
    "does",
    "doing",
    "for",
    "from",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "also",
    "am",
    "any",
    "because",
    "before",
    "below",
    "between",
    "both",
    "but",
    "can",
    "cannot",
    "could",
    "did",
    "each",
    "few",
    "get",
    "got",
    "however",
    "like",
    "make",
    "many",
    "may",
    "might",
    "much",
    "must",
    "need",
    "new",
    "now",
    "one",
    "onto",
    "per",
    "shall",
    "since",
    "still",
    "two",
    "way",
    "well",
    "yet",
    "able",
    "along",
    "already",
    "around",
}

# Technical skills and keywords commonly found in tech jobs
TECH_KEYWORDS = {
    # Programming languages
    "python",
    "java",
    "javascript",
    "typescript",
    "c++",
    "c#",
    "ruby",
    "go",
    "golang",
    "rust",
    "swift",
    "kotlin",
    "scala",
    "php",
    "perl",
    "r",
    "matlab",
    "sql",
    "nosql",
    "bash",
    "shell",
    "powershell",
    # Frameworks & Libraries
    "react",
    "angular",
    "vue",
    "nodejs",
    "node.js",
    "django",
    "flask",
    "fastapi",
    "spring",
    "springboot",
    "express",
    "nextjs",
    "next.js",
    "tensorflow",
    "pytorch",
    "keras",
    "scikit-learn",
    "pandas",
    "numpy",
    "jquery",
    "bootstrap",
    "tailwind",
    "svelte",
    # Cloud & DevOps
    "aws",
    "azure",
    "gcp",
    "google cloud",
    "docker",
    "kubernetes",
    "k8s",
    "terraform",
    "ansible",
    "jenkins",
    "gitlab",
    "github",
    "ci/cd",
    "cicd",
    "devops",
    "mlops",
    "dataops",
    "linux",
    "unix",
    "nginx",
    "apache",
    # Databases
    "mysql",
    "postgresql",
    "postgres",
    "mongodb",
    "redis",
    "elasticsearch",
    "dynamodb",
    "cassandra",
    "oracle",
    "sqlserver",
    "sqlite",
    "firebase",
    # Data & AI
    "machine learning",
    "deep learning",
    "nlp",
    "natural language",
    "computer vision",
    "ai",
    "artificial intelligence",
    "data science",
    "data engineering",
    "etl",
    "data pipeline",
    "big data",
    "hadoop",
    "spark",
    "kafka",
    "airflow",
    "snowflake",
    "databricks",
    # Methodologies & Tools
    "agile",
    "scrum",
    "kanban",
    "jira",
    "confluence",
    "git",
    "svn",
    "rest",
    "api",
    "graphql",
    "microservices",
    "serverless",
    # Certifications
    "aws certified",
    "azure certified",
    "gcp certified",
    "pmp",
    "scrum master",
    "cissp",
    "ceh",
    "comptia",
}


def _tokenize(text: str) -> list[str]:
    """Tokenize text into words."""
    # Convert to lowercase and extract word tokens
    text = text.lower()
    # Keep alphanumeric chars, dots, hashes, pluses (for C++, C#, .NET, etc.)
    tokens = re.findall(r"[a-z0-9]+(?:[.+#][a-z0-9]+)*", text)
    return tokens


def _extract_ngrams(tokens: list[str], n: int = 2) -> list[str]:
    """Extract n-grams from tokens."""
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def extract_keywords(
    text: str,
    min_length: int = 2,
    include_ngrams: bool = True,
) -> set[str]:
    """
    Extract keywords from text using regex-based tokenization.

    Args:
        text: Input text to extract keywords from
        min_length: Minimum keyword length
        include_ngrams: Whether to include bigrams

    Returns:
        Set of extracted keywords (lowercased)
    """
    if not text.strip():
        return set()

    tokens = _tokenize(text)
    keywords: set[str] = set()

    # Add single tokens (excluding stop words)
    for token in tokens:
        if len(token) >= min_length and token not in STOP_WORDS and not token.isdigit():
            keywords.add(token)

    # Add bigrams
    if include_ngrams:
        bigrams = _extract_ngrams(tokens, 2)
        for bigram in bigrams:
            parts = bigram.split()
            # Keep bigrams where at least one word is not a stop word
            if any(p not in STOP_WORDS for p in parts):
                if bigram in TECH_KEYWORDS or any(p in TECH_KEYWORDS for p in parts):
                    keywords.add(bigram)

    return keywords


def extract_skills(text: str) -> set[str]:
    """
    Extract technical skills and competencies from text.

    This is optimized for finding skills commonly listed in resumes
    and job descriptions.
    """
    if not text.strip():
        return set()

    text_lower = text.lower()
    skills: set[str] = set()

    # Find direct matches with tech keywords
    for keyword in TECH_KEYWORDS:
        if keyword in text_lower:
            skills.add(keyword)

    # Also extract using general keyword extraction
    general_keywords = extract_keywords(text, min_length=2, include_ngrams=True)
    skills.update(general_keywords)

    return skills


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two texts using TF-IDF.

    Args:
        text1: First text (e.g., resume)
        text2: Second text (e.g., job description)

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1.strip() or not text2.strip():
        return 0.0

    try:
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0


def calculate_keyword_match(
    resume_keywords: set[str], job_keywords: set[str]
) -> tuple[float, set[str], set[str], set[str]]:
    """
    Calculate keyword match statistics.

    Args:
        resume_keywords: Keywords from resume
        job_keywords: Keywords from job description

    Returns:
        Tuple of (match_percentage, matching, missing, extra)
    """
    if not job_keywords:
        return 100.0, set(), set(), resume_keywords

    matching = resume_keywords & job_keywords
    missing = job_keywords - resume_keywords
    extra = resume_keywords - job_keywords

    match_percentage = (len(matching) / len(job_keywords)) * 100 if job_keywords else 0.0

    return match_percentage, matching, missing, extra


@lru_cache(maxsize=100)
def get_keyword_importance(keyword: str) -> str:
    """
    Estimate the importance of a keyword for ATS matching.

    Returns: 'high', 'medium', or 'low'
    """
    keyword_lower = keyword.lower()

    # High-importance: known tech skills
    if keyword_lower in TECH_KEYWORDS:
        return "high"

    # High-importance patterns (seniority, roles)
    high_patterns = {
        "engineer",
        "developer",
        "manager",
        "director",
        "senior",
        "lead",
        "architect",
        "analyst",
        "scientist",
        "specialist",
        "consultant",
    }

    if keyword_lower in high_patterns:
        return "high"

    # Medium importance: longer keywords likely to be specific
    if len(keyword) > 5:
        return "medium"

    return "low"
