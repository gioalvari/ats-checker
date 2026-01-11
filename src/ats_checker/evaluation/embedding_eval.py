"""Evaluation metrics for embedding quality."""

from dataclasses import dataclass

import numpy as np

from ats_checker.embedding_analyzer import EmbeddingAnalyzer
from ats_checker.evaluation.test_dataset import TEST_CASES, TestCase


@dataclass
class EvaluationResult:
    """Results from embedding evaluation."""

    mae: float  # Mean Absolute Error between predicted and expected scores
    rmse: float  # Root Mean Square Error
    correlation: float  # Pearson correlation
    precision: float  # Precision for match/no-match classification
    recall: float  # Recall for match/no-match classification
    f1: float  # F1 score
    accuracy: float  # Overall accuracy
    predictions: list[dict]  # Individual predictions for analysis


def evaluate_embeddings(
    test_cases: list[TestCase] | None = None,
    threshold: float = 50.0,
    embedding_model: str = "nomic-embed-text",
    verbose: bool = True,
) -> EvaluationResult:
    """
    Evaluate embedding quality on a test set.

    Args:
        test_cases: List of test cases (uses default TEST_CASES if None)
        threshold: Score threshold for classifying as "match" (default 50%)
        embedding_model: Embedding model to use
        verbose: Print progress

    Returns:
        EvaluationResult with metrics and predictions
    """
    if test_cases is None:
        test_cases = TEST_CASES

    analyzer = EmbeddingAnalyzer(embedding_model=embedding_model)

    predicted_scores: list[float] = []
    expected_scores: list[float] = []
    predicted_match: list[bool] = []
    expected_match: list[bool] = []
    predictions: list[dict] = []

    for i, case in enumerate(test_cases):
        if verbose:
            print(f"  Testing case {i + 1}/{len(test_cases)}: {case.category}...", end=" ")

        try:
            result = analyzer.analyze(case.resume, case.job)
            # Use adjusted_score if available (includes penalties), otherwise overall_similarity
            pred_score = getattr(result, "adjusted_score", result.overall_similarity)
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            continue

        predicted_scores.append(pred_score)
        expected_scores.append(case.expected_score)

        pred_is_match = pred_score >= threshold
        predicted_match.append(pred_is_match)
        expected_match.append(case.expected_match)

        error = abs(pred_score - case.expected_score)

        predictions.append(
            {
                "category": case.category,
                "expected_score": case.expected_score,
                "predicted_score": pred_score,
                "raw_score": result.overall_similarity,
                "error": error,
                "expected_match": case.expected_match,
                "predicted_match": pred_is_match,
                "correct_classification": pred_is_match == case.expected_match,
                "notes": case.notes,
                "keyword_overlap": getattr(result, "keyword_overlap", None),
                "domain_match": getattr(result, "domain_match", None),
            }
        )

        if verbose:
            status = "✅" if error < 15 else "⚠️" if error < 25 else "❌"
            print(
                f"{status} Expected: {case.expected_score:.0f}, Got: {pred_score:.1f}, Error: {error:.1f}"
            )

    if not predicted_scores:
        raise ValueError("No successful predictions")

    # Calculate metrics
    pred_arr = np.array(predicted_scores)
    exp_arr = np.array(expected_scores)

    mae = float(np.mean(np.abs(pred_arr - exp_arr)))
    rmse = float(np.sqrt(np.mean((pred_arr - exp_arr) ** 2)))

    # Correlation (handle edge case of constant values)
    if np.std(pred_arr) > 0 and np.std(exp_arr) > 0:
        correlation = float(np.corrcoef(pred_arr, exp_arr)[0, 1])
    else:
        correlation = 0.0

    # Classification metrics
    tp = sum(1 for p, e in zip(predicted_match, expected_match, strict=False) if p and e)
    fp = sum(1 for p, e in zip(predicted_match, expected_match, strict=False) if p and not e)
    fn = sum(1 for p, e in zip(predicted_match, expected_match, strict=False) if not p and e)
    tn = sum(1 for p, e in zip(predicted_match, expected_match, strict=False) if not p and not e)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(predicted_match) if predicted_match else 0.0

    return EvaluationResult(
        mae=mae,
        rmse=rmse,
        correlation=correlation,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        predictions=predictions,
    )


def compare_embedding_models(
    embedding_models: list[str],
    test_cases: list[TestCase] | None = None,
    threshold: float = 50.0,
) -> dict[str, EvaluationResult]:
    """
    Compare multiple embedding models.

    Args:
        embedding_models: List of model names to compare
        test_cases: Test cases to use
        threshold: Match threshold

    Returns:
        Dictionary mapping model name to EvaluationResult
    """
    results = {}

    for model_name in embedding_models:
        print(f"\n{'=' * 50}")
        print(f"Testing model: {model_name}")
        print("=" * 50)

        try:
            result = evaluate_embeddings(
                test_cases=test_cases,
                threshold=threshold,
                embedding_model=model_name,
                verbose=True,
            )
            results[model_name] = result
        except Exception as e:
            print(f"❌ Failed to evaluate {model_name}: {e}")
            continue

    return results


def print_evaluation_report(result: EvaluationResult, model_name: str = "Embedding Model") -> None:
    """Print a detailed evaluation report."""
    print("\n" + "=" * 60)
    print(f"EVALUATION REPORT: {model_name}")
    print("=" * 60)

    print("\n📊 REGRESSION METRICS (Score Prediction)")
    print(f"  MAE (Mean Absolute Error):  {result.mae:.2f}")
    print(f"  RMSE (Root Mean Square):    {result.rmse:.2f}")
    print(f"  Correlation:                {result.correlation:.3f}")

    print("\n🎯 CLASSIFICATION METRICS (Match/No-Match)")
    print(f"  Accuracy:   {result.accuracy:.1%}")
    print(f"  Precision:  {result.precision:.1%}")
    print(f"  Recall:     {result.recall:.1%}")
    print(f"  F1 Score:   {result.f1:.1%}")

    # Analysis by category
    categories: dict[str, list[dict]] = {}
    for pred in result.predictions:
        cat = pred["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(pred)

    print("\n📁 PERFORMANCE BY CATEGORY")
    for cat, preds in sorted(categories.items()):
        avg_error = sum(p["error"] for p in preds) / len(preds)
        correct = sum(1 for p in preds if p["correct_classification"])
        print(f"  {cat}:")
        print(f"    Avg Error: {avg_error:.1f}, Classification: {correct}/{len(preds)} correct")

    # Worst predictions
    print("\n⚠️ LARGEST ERRORS")
    sorted_preds = sorted(result.predictions, key=lambda x: x["error"], reverse=True)
    for pred in sorted_preds[:3]:
        print(
            f"  - {pred['category']}: Expected {pred['expected_score']:.0f}, Got {pred['predicted_score']:.1f}"
        )
        print(f"    Error: {pred['error']:.1f} | {pred['notes'][:50]}...")

    # Overall grade
    print("\n" + "=" * 60)
    if result.mae < 10 and result.f1 > 0.8:
        grade = "A - Excellent"
    elif result.mae < 15 and result.f1 > 0.7:
        grade = "B - Good"
    elif result.mae < 20 and result.f1 > 0.6:
        grade = "C - Acceptable"
    elif result.mae < 25:
        grade = "D - Needs Improvement"
    else:
        grade = "F - Poor"

    print(f"OVERALL GRADE: {grade}")
    print("=" * 60)


def run_full_evaluation() -> None:
    """Run complete evaluation with all test cases."""
    print("\n" + "=" * 60)
    print("EMBEDDING EVALUATION - FULL TEST SUITE")
    print("=" * 60)

    result = evaluate_embeddings(verbose=True)
    print_evaluation_report(result, "nomic-embed-text")


if __name__ == "__main__":
    run_full_evaluation()
