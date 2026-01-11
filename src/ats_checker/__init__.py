"""ATS Resume Analyzer - Score and optimize your resume for job descriptions using AI."""

__version__ = "0.1.0"

from ats_checker.cot_enhancer import ChainOfThoughtEnhancer, enhance_with_cot
from ats_checker.embedding_analyzer import EmbeddingAnalyzer, EmbeddingResult
from ats_checker.keyword_extractor import calculate_similarity, extract_keywords
from ats_checker.llm_analyzer import OllamaAnalyzer
from ats_checker.models import AnalysisResult, JobDescription, Resume, Suggestion
from ats_checker.pdf_parser import extract_text_from_pdf
from ats_checker.resume_exporter import ResumeExporter
from ats_checker.resume_generator import ResumeAdapter

__all__ = [
    "AnalysisResult",
    "JobDescription",
    "Resume",
    "Suggestion",
    "extract_text_from_pdf",
    "extract_keywords",
    "calculate_similarity",
    "OllamaAnalyzer",
    "ResumeAdapter",
    "ResumeExporter",
    "EmbeddingAnalyzer",
    "EmbeddingResult",
    "ChainOfThoughtEnhancer",
    "enhance_with_cot",
]
