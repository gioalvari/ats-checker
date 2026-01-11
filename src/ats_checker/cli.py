"""CLI entry point for ATS Resume Analyzer."""

import argparse
import json
import logging
import sys
from pathlib import Path

from ats_checker.config import config
from ats_checker.keyword_extractor import (
    calculate_keyword_match,
    calculate_similarity,
    extract_keywords,
)
from ats_checker.llm_analyzer import OllamaAnalyzer
from ats_checker.pdf_parser import extract_text_from_file
from ats_checker.resume_exporter import ResumeExporter
from ats_checker.resume_generator import ResumeAdapter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ATS Resume Analyzer - Analyze and optimize your resume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze resume against job description
  ats-checker analyze resume.pdf job.txt

  # Analyze with specific model
  ats-checker analyze resume.pdf job.txt --model llama3.2

  # Generate optimized resume
  ats-checker optimize resume.pdf job.txt -o optimized.pdf

  # Start web interface
  ats-checker web
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze resume against job")
    analyze_parser.add_argument("resume", type=Path, help="Path to resume file (PDF, DOCX, TXT)")
    analyze_parser.add_argument("job", type=Path, help="Path to job description file")
    analyze_parser.add_argument(
        "--model", "-m", default=config.ollama.default_model, help="Ollama model to use"
    )
    analyze_parser.add_argument("--export", "-e", type=Path, help="Export report to JSON file")
    analyze_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Generate optimized resume")
    optimize_parser.add_argument("resume", type=Path, help="Path to resume file")
    optimize_parser.add_argument("job", type=Path, help="Path to job description file")
    optimize_parser.add_argument(
        "--output", "-o", type=Path, default=Path("optimized_resume.pdf"), help="Output file path"
    )
    optimize_parser.add_argument(
        "--format",
        "-f",
        choices=["pdf", "docx", "md", "html"],
        default="pdf",
        help="Output format",
    )
    optimize_parser.add_argument(
        "--model", "-m", default=config.ollama.default_model, help="Ollama model to use"
    )

    # Web command
    web_parser = subparsers.add_parser("web", help="Start web interface")
    web_parser.add_argument("--port", "-p", type=int, default=8501, help="Port number")
    web_parser.add_argument("--host", default="localhost", help="Host address")

    # Models command
    subparsers.add_parser("models", help="List available Ollama models")

    args = parser.parse_args()

    if args.command == "analyze":
        run_analyze(args)
    elif args.command == "optimize":
        run_optimize(args)
    elif args.command == "web":
        run_web(args)
    elif args.command == "models":
        run_models()
    else:
        parser.print_help()
        sys.exit(1)


def run_analyze(args: argparse.Namespace) -> None:
    """Run analysis command."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Reading resume from {args.resume}")
    resume_text = extract_text_from_file(args.resume)

    logger.info(f"Reading job description from {args.job}")
    job_text = extract_text_from_file(args.job)

    # Extract keywords
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_text)

    # Calculate scores
    keyword_score, matching, missing, extra = calculate_keyword_match(resume_keywords, job_keywords)
    similarity = calculate_similarity(resume_text, job_text)

    # LLM analysis
    logger.info(f"Running AI analysis with model: {args.model}")
    analyzer = OllamaAnalyzer(model=args.model)
    result = analyzer.analyze_resume(
        resume_text=resume_text,
        job_text=job_text,
        keyword_score=keyword_score,
        missing_keywords=sorted(missing),
        matching_keywords=sorted(matching),
    )

    # Print report
    print("\n" + "=" * 60)
    print("📊 ATS RESUME ANALYSIS REPORT")
    print("=" * 60)

    print(f"\n🎯 Overall Match Score: {result.match_score:.0f}%")
    print(f"🔑 Keyword Match: {result.keyword_score:.0f}%")
    print(f"📐 Content Similarity: {similarity * 100:.0f}%")

    print(f"\n📋 Summary:\n{result.summary}")

    if result.strengths:
        print("\n✅ Strengths:")
        for s in result.strengths:
            print(f"  • {s}")

    if result.weaknesses:
        print("\n⚠️ Areas to Improve:")
        for w in result.weaknesses:
            print(f"  • {w}")

    if result.missing_keywords:
        print(f"\n❌ Missing Keywords ({len(result.missing_keywords)}):")
        print(f"  {', '.join(result.missing_keywords[:20])}")

    if result.suggestions:
        print(f"\n💡 Suggestions ({len(result.suggestions)}):")
        for i, s in enumerate(result.suggestions[:5], 1):
            print(f"\n  {i}. [{s.priority.upper()}] {s.category}")
            print(f"     {s.suggested_text[:100]}...")
            print(f"     Why: {s.reasoning[:80]}...")

    print("\n" + "=" * 60)

    if args.export:
        report_data = result.model_dump()
        args.export.write_text(json.dumps(report_data, indent=2))
        logger.info(f"Report exported to {args.export}")


def run_optimize(args: argparse.Namespace) -> None:
    """Run optimization command."""
    logger.info(f"Reading resume from {args.resume}")
    resume_text = extract_text_from_file(args.resume)

    logger.info(f"Reading job description from {args.job}")
    job_text = extract_text_from_file(args.job)

    logger.info(f"Generating optimized resume with model: {args.model}")
    analyzer = OllamaAnalyzer(model=args.model)
    adapter = ResumeAdapter(analyzer=analyzer)

    adapted = adapter.generate_full_adapted_resume(resume_text, job_text)

    # Export
    exporter = ResumeExporter(adapted.adapted_text)
    output_path = exporter.save(args.output.with_suffix(""), args.format)

    print(f"\n✅ Optimized resume saved to: {output_path}")
    print("\n📝 Changes made:")
    for change in adapted.changes_made:
        print(f"  • {change}")


def run_web(args: argparse.Namespace) -> None:
    """Start web interface."""
    import subprocess

    logger.info(f"Starting web interface at http://{args.host}:{args.port}")

    # Get the app module path
    import ats_checker.app

    app_path = ats_checker.app.__file__

    subprocess.run(
        [
            "streamlit",
            "run",
            app_path,
            "--server.port",
            str(args.port),
            "--server.address",
            args.host,
        ]
    )


def run_models() -> None:
    """List available models."""
    print("\n🤖 Available Ollama Models:")
    print("-" * 40)

    models = config.ollama.get_available_models()

    if models:
        for model in models:
            default = " (default)" if config.ollama.default_model in model else ""
            print(f"  • {model}{default}")
    else:
        print("  No models found. Make sure Ollama is running.")
        print("\n  To start Ollama:")
        print("    ollama serve")
        print("\n  To download a model:")
        print("    ollama pull llama3.2")

    print()


if __name__ == "__main__":
    main()
