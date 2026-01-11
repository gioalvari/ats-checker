"""Evaluation metrics for resume enhancement quality."""

import re
from dataclasses import dataclass

from ats_checker.embedding_analyzer import EmbeddingAnalyzer


@dataclass
class EnhancementMetrics:
    """Metrics for evaluating resume enhancement quality."""

    # Similarity improvement
    original_similarity: float
    enhanced_similarity: float
    similarity_improvement: float

    # Fact preservation
    dates_preserved: bool
    numbers_preserved: bool
    percentages_preserved: bool
    fact_preservation_score: float  # 0-100

    # Quality metrics
    keyword_incorporation: float  # % of target keywords incorporated
    keywords_found: int
    keywords_total: int

    # Length metrics
    original_length: int
    enhanced_length: int
    length_change_percent: float

    # Detailed facts
    original_facts: dict
    enhanced_facts: dict
    missing_facts: dict


def extract_facts(text: str) -> dict:
    """
    Extract verifiable facts from text.

    Returns dict with:
    - dates: years found (e.g., 2020, 2024)
    - percentages: percentage values (e.g., 35%, 99.9%)
    - money: monetary values (e.g., $5M, €100K)
    - numbers_with_units: numbers with context (e.g., 5 years, 10M users)
    - emails: email addresses
    - metrics: achievement metrics (e.g., "reduced by 35%", "increased 2x")
    """
    facts = {
        "dates": list(set(re.findall(r"\b(19|20)\d{2}\b", text))),
        "percentages": list(set(re.findall(r"\d+(?:\.\d+)?%", text))),
        "money": list(set(re.findall(r"[€$£]\s*\d+(?:[.,]\d+)*\s*[MKBmkb]?", text))),
        "numbers_with_units": list(
            set(
                re.findall(
                    r"\b\d+(?:\.\d+)?\s*(?:years?|months?|million|M|K|users?|clients?|projects?)\b",
                    text,
                    re.I,
                )
            )
        ),
        "emails": list(set(re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", text))),
        "phone_numbers": list(set(re.findall(r"\+?\d[\d\s\-]{8,}\d", text))),
        "urls": list(set(re.findall(r"https?://[^\s]+|www\.[^\s]+", text))),
    }
    return facts


def check_facts_preserved(original_facts: dict, enhanced_facts: dict) -> dict:
    """
    Check which facts from original are preserved in enhanced.

    Returns dict with preservation status for each category.
    """
    missing = {}

    for category, original_items in original_facts.items():
        if not original_items:
            continue

        enhanced_items = enhanced_facts.get(category, [])
        missing_items = [item for item in original_items if item not in enhanced_items]

        if missing_items:
            missing[category] = missing_items

    return missing


def evaluate_enhancement(
    original: str,
    enhanced: str,
    job_description: str,
    target_keywords: list[str] | None = None,
    embedding_model: str = "nomic-embed-text",
) -> EnhancementMetrics:
    """
    Evaluate the quality of resume enhancement.

    Args:
        original: Original resume text
        enhanced: Enhanced resume text
        job_description: Job description used for enhancement
        target_keywords: Keywords that should be incorporated
        embedding_model: Model to use for similarity calculation

    Returns:
        EnhancementMetrics with detailed evaluation
    """
    analyzer = EmbeddingAnalyzer(embedding_model=embedding_model)

    # 1. Calculate similarity improvement
    original_result = analyzer.analyze(original, job_description)
    enhanced_result = analyzer.analyze(enhanced, job_description)

    original_similarity = original_result.overall_similarity
    enhanced_similarity = enhanced_result.overall_similarity
    similarity_improvement = enhanced_similarity - original_similarity

    # 2. Extract and verify facts
    original_facts = extract_facts(original)
    enhanced_facts = extract_facts(enhanced)
    missing_facts = check_facts_preserved(original_facts, enhanced_facts)

    # Calculate preservation scores
    dates_preserved = len(missing_facts.get("dates", [])) == 0
    numbers_preserved = len(missing_facts.get("numbers_with_units", [])) == 0
    percentages_preserved = len(missing_facts.get("percentages", [])) == 0

    # Overall fact preservation score
    total_original_facts = sum(len(v) for v in original_facts.values())
    total_missing_facts = sum(len(v) for v in missing_facts.values())

    if total_original_facts > 0:
        fact_preservation_score = (1 - total_missing_facts / total_original_facts) * 100
    else:
        fact_preservation_score = 100.0

    # 3. Keyword incorporation
    if target_keywords:
        enhanced_lower = enhanced.lower()
        keywords_found = sum(1 for kw in target_keywords if kw.lower() in enhanced_lower)
        keyword_incorporation = keywords_found / len(target_keywords) * 100
    else:
        keywords_found = 0
        keyword_incorporation = 0.0

    # 4. Length metrics
    original_length = len(original)
    enhanced_length = len(enhanced)
    length_change_percent = (
        (enhanced_length - original_length) / original_length * 100 if original_length > 0 else 0
    )

    return EnhancementMetrics(
        original_similarity=original_similarity,
        enhanced_similarity=enhanced_similarity,
        similarity_improvement=similarity_improvement,
        dates_preserved=dates_preserved,
        numbers_preserved=numbers_preserved,
        percentages_preserved=percentages_preserved,
        fact_preservation_score=fact_preservation_score,
        keyword_incorporation=keyword_incorporation,
        keywords_found=keywords_found,
        keywords_total=len(target_keywords) if target_keywords else 0,
        original_length=original_length,
        enhanced_length=enhanced_length,
        length_change_percent=length_change_percent,
        original_facts=original_facts,
        enhanced_facts=enhanced_facts,
        missing_facts=missing_facts,
    )


def print_evaluation_report(metrics: EnhancementMetrics) -> None:
    """Print a detailed enhancement evaluation report."""
    print("\n" + "=" * 60)
    print("ENHANCEMENT EVALUATION REPORT")
    print("=" * 60)

    # Similarity section
    print("\n📈 SIMILARITY IMPROVEMENT")
    print(f"  Original Score:  {metrics.original_similarity:.1f}%")
    print(f"  Enhanced Score:  {metrics.enhanced_similarity:.1f}%")

    improvement_icon = (
        "🟢"
        if metrics.similarity_improvement > 5
        else "🟡"
        if metrics.similarity_improvement > 0
        else "🔴"
    )
    print(f"  Improvement:     {improvement_icon} {metrics.similarity_improvement:+.1f}%")

    # Fact preservation section
    print(f"\n🔒 FACT PRESERVATION: {metrics.fact_preservation_score:.0f}%")
    print(f"  Dates preserved:       {'✅' if metrics.dates_preserved else '❌'}")
    print(f"  Numbers preserved:     {'✅' if metrics.numbers_preserved else '❌'}")
    print(f"  Percentages preserved: {'✅' if metrics.percentages_preserved else '❌'}")

    if metrics.missing_facts:
        print("\n  ⚠️ Missing facts:")
        for category, items in metrics.missing_facts.items():
            print(f"    {category}: {', '.join(str(i) for i in items[:5])}")

    # Keyword section
    if metrics.keywords_total > 0:
        print(f"\n🎯 KEYWORD INCORPORATION: {metrics.keyword_incorporation:.0f}%")
        print(f"  Keywords found: {metrics.keywords_found}/{metrics.keywords_total}")

    # Length section
    print(f"\n📏 LENGTH CHANGE: {metrics.length_change_percent:+.1f}%")
    print(f"  Original: {metrics.original_length:,} chars")
    print(f"  Enhanced: {metrics.enhanced_length:,} chars")

    # Overall score calculation
    # Weights: similarity improvement (30), fact preservation (50), keywords (20)
    similarity_score = min(max(metrics.similarity_improvement, 0), 20) / 20 * 30
    fact_score = metrics.fact_preservation_score / 100 * 50
    keyword_score = metrics.keyword_incorporation / 100 * 20 if metrics.keywords_total > 0 else 20

    overall_score = similarity_score + fact_score + keyword_score

    print("\n" + "=" * 60)
    print("SCORING BREAKDOWN")
    print(f"  Similarity Improvement: {similarity_score:.1f}/30")
    print(f"  Fact Preservation:      {fact_score:.1f}/50")
    print(f"  Keyword Incorporation:  {keyword_score:.1f}/20")
    print("=" * 60)

    # Grade
    if overall_score >= 90:
        grade = "A+ Excellent"
        emoji = "🌟"
    elif overall_score >= 80:
        grade = "A Good"
        emoji = "✨"
    elif overall_score >= 70:
        grade = "B Acceptable"
        emoji = "👍"
    elif overall_score >= 60:
        grade = "C Needs Work"
        emoji = "⚠️"
    else:
        grade = "D Poor"
        emoji = "❌"

    print(f"\n{emoji} OVERALL ENHANCEMENT SCORE: {overall_score:.0f}/100 ({grade})")
    print("=" * 60)


def evaluate_enhancement_chain(
    original: str,
    enhanced: str,
    job_description: str,
    target_keywords: list[str] | None = None,
    print_report: bool = True,
) -> EnhancementMetrics:
    """
    Convenience function to evaluate and optionally print report.

    Args:
        original: Original resume
        enhanced: Enhanced resume
        job_description: Job description
        target_keywords: Target keywords
        print_report: Whether to print the report

    Returns:
        EnhancementMetrics
    """
    metrics = evaluate_enhancement(
        original=original,
        enhanced=enhanced,
        job_description=job_description,
        target_keywords=target_keywords,
    )

    if print_report:
        print_evaluation_report(metrics)

    return metrics


if __name__ == "__main__":
    # Example usage
    original = """
    John Doe
    Data Scientist | 5 Years Experience

    EXPERIENCE:
    Senior Data Scientist at TechCorp (2020-2024)
    - Built ML models for sales forecasting
    - Reduced prediction error by 35%
    - Managed team of 3 analysts
    - Saved company $2M annually

    SKILLS: Python, SQL, TensorFlow, AWS
    """

    enhanced = """
    John Doe
    Senior Data Scientist | 5 Years Experience | ML & Forecasting Expert

    EXPERIENCE:
    Senior Data Scientist at TechCorp (2020-2024)
    - Designed and deployed production ML models for demand forecasting using Prophet and LSTM
    - Reduced prediction error by 35% through advanced feature engineering and model optimization
    - Led and mentored team of 3 junior data scientists
    - Delivered $2M annual cost savings through improved inventory optimization

    SKILLS: Python, SQL, TensorFlow, PyTorch, AWS (S3, SageMaker), MLOps, Time Series Analysis
    """

    job = """
    Senior Data Scientist

    Requirements:
    - 5+ years experience
    - Production ML systems
    - Time series forecasting
    - AWS experience
    - Team leadership
    """

    keywords = ["mlops", "sagemaker", "forecasting", "production", "mentoring"]

    metrics = evaluate_enhancement_chain(
        original=original,
        enhanced=enhanced,
        job_description=job,
        target_keywords=keywords,
    )
