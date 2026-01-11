"""ATS Resume Analyzer - Main Streamlit Application."""

import logging

import streamlit as st

from ats_checker.config import config
from ats_checker.embedding_analyzer import EmbeddingAnalyzer, get_available_embedding_models
from ats_checker.keyword_extractor import (
    calculate_keyword_match,
    calculate_similarity,
    extract_keywords,
)
from ats_checker.llm_analyzer import OllamaAnalyzer
from ats_checker.models import AdaptedResume, AnalysisResult
from ats_checker.resume_generator import ResumeAdapter
from ats_checker.ui.editor import (
    render_generate_button,
    render_resume_editor,
)
from ats_checker.ui.panels import render_job_panel, render_model_selector, render_resume_panel
from ats_checker.ui.results import (
    render_analysis_results,
    render_export_options,
    render_score_card,
    render_suggestions,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "adapted_resume" not in st.session_state:
        st.session_state.adapted_resume = None
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = None
    if "job_text" not in st.session_state:
        st.session_state.job_text = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = config.ollama.default_model
    if "embedding_result" not in st.session_state:
        st.session_state.embedding_result = None
    if "use_embeddings" not in st.session_state:
        st.session_state.use_embeddings = True


def main() -> None:
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title=config.page_title,
        page_icon=config.page_icon,
        layout=config.layout,
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    init_session_state()

    # Header
    st.title(f"{config.page_icon} {config.app_name}")
    st.markdown(
        "Analyze your resume against job descriptions, get AI-powered suggestions, "
        "and generate optimized versions tailored for ATS systems."
    )

    # Sidebar - Model Selection
    available_models = config.ollama.get_available_models()
    selected_model = render_model_selector(
        available_models=available_models,
        default_model=config.ollama.default_model,
    )
    st.session_state.selected_model = selected_model

    # Sidebar - Embedding Settings
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🧠 Semantic Analysis")

        use_embeddings = st.checkbox(
            "Use Embeddings",
            value=st.session_state.use_embeddings,
            help="Use semantic embeddings for more accurate matching (recommended)",
        )
        st.session_state.use_embeddings = use_embeddings

        if use_embeddings:
            embedding_models = get_available_embedding_models()
            if embedding_models:
                selected_emb_model = st.selectbox(
                    "Embedding Model",
                    options=embedding_models,
                    index=0,
                    help="Model for semantic similarity calculation",
                )
                st.session_state.embedding_model = selected_emb_model
            else:
                st.warning("⚠️ No embedding model found. Run: `ollama pull nomic-embed-text`")
                st.session_state.embedding_model = "nomic-embed-text"

    # Sidebar - App Info
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            """
            This tool helps you:
            1. **Analyze** resume-job compatibility
            2. **Identify** missing keywords
            3. **Get** AI-powered suggestions
            4. **Generate** optimized resumes

            All analysis happens locally using Ollama.
            """
        )

        st.markdown("---")
        st.markdown("### 🔧 Quick Setup")
        with st.expander("First time setup"):
            st.code(
                """# Install Ollama
brew install ollama

# Start Ollama server
ollama serve

# Pull a model
ollama pull llama3.2""",
                language="bash",
            )

    # Main content - Two column layout
    st.markdown("---")
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        resume_text, resume_file = render_resume_panel()
        if resume_text:
            st.session_state.resume_text = resume_text

    with col_right:
        job_text, job_source = render_job_panel()
        if job_text:
            st.session_state.job_text = job_text

    # Analysis button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        analyze_disabled = not (st.session_state.resume_text and st.session_state.job_text)

        if analyze_disabled:
            st.info("👆 Upload or paste both your resume and the job description to begin analysis")

        if st.button(
            "🔍 Analyze Resume",
            type="primary",
            disabled=analyze_disabled,
            use_container_width=True,
            key="analyze_btn",
        ):
            with st.spinner("Analyzing resume with AI... This may take a moment."):
                try:
                    # Embedding analysis first (if enabled)
                    embedding_result = None
                    if st.session_state.use_embeddings:
                        emb_model = st.session_state.get("embedding_model", "nomic-embed-text")
                        embedding_analyzer = EmbeddingAnalyzer(embedding_model=emb_model)
                        embedding_result = embedding_analyzer.analyze(
                            st.session_state.resume_text,
                            st.session_state.job_text,
                        )
                        st.session_state.embedding_result = embedding_result

                    # Traditional + LLM analysis
                    result = analyze_resume(
                        st.session_state.resume_text,
                        st.session_state.job_text,
                        selected_model,
                        embedding_result=embedding_result,
                    )
                    st.session_state.analysis_result = result
                    st.session_state.adapted_resume = None  # Reset adapted resume
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    logger.error(f"Analysis error: {e}")

    # Results section
    if st.session_state.analysis_result:
        result: AnalysisResult = st.session_state.analysis_result

        # Score card
        render_score_card(result)

        # Embedding analysis results (if available)
        if st.session_state.embedding_result:
            emb_result = st.session_state.embedding_result
            st.markdown("### 🧠 Semantic Analysis")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Semantic Similarity",
                    f"{emb_result.overall_similarity:.1f}%",
                    help="Based on deep semantic understanding of both texts",
                )
            with col2:
                st.caption(f"Model: `{emb_result.embedding_model}`")

            # Section scores
            if emb_result.section_scores:
                with st.expander("📊 Section-by-Section Scores", expanded=False):
                    for section, score in sorted(
                        emb_result.section_scores.items(), key=lambda x: -x[1]
                    ):
                        st.progress(score / 100, text=f"{section.title()}: {score:.0f}%")

            # Strong matches and gaps
            col1, col2 = st.columns(2)
            with col1:
                if emb_result.strong_matches:
                    st.markdown("**✅ Strong Matches:**")
                    for match in emb_result.strong_matches[:3]:
                        st.markdown(f"- {match[:80]}..." if len(match) > 80 else f"- {match}")
            with col2:
                if emb_result.semantic_gaps:
                    st.markdown("**⚠️ Missing Concepts:**")
                    for gap in emb_result.semantic_gaps[:3]:
                        st.markdown(f"- {gap[:80]}..." if len(gap) > 80 else f"- {gap}")

            st.markdown("---")

        # Detailed results
        render_analysis_results(result)

        # Suggestions with selection
        selected_indices = render_suggestions(result.suggestions)

        # Generate optimized resume button
        if render_generate_button(
            st.session_state.resume_text,
            st.session_state.job_text,
            key="generate_optimized_btn",
        ):
            with st.spinner("Generating optimized resume... This may take 30-60 seconds."):
                try:
                    # Get semantic gaps if available
                    embedding_gaps = None
                    if st.session_state.embedding_result:
                        embedding_gaps = st.session_state.embedding_result.semantic_gaps

                    adapted = generate_adapted_resume(
                        st.session_state.resume_text,
                        st.session_state.job_text,
                        result.suggestions,
                        selected_indices if selected_indices else None,
                        selected_model,
                        embedding_gaps=embedding_gaps,
                        missing_keywords=result.missing_keywords[:20],
                    )
                    st.session_state.adapted_resume = adapted
                    st.success("✅ Optimized resume generated!")
                except Exception as e:
                    st.error(f"❌ Generation failed: {e}")
                    logger.error(f"Generation error: {e}")

    # Resume editor section
    if st.session_state.adapted_resume:
        edited_content = render_resume_editor(
            st.session_state.adapted_resume,
            key_prefix="main_editor",
        )

        if edited_content:
            render_export_options(
                content=edited_content,
                filename_base="optimized_resume",
            )


def analyze_resume(
    resume_text: str,
    job_text: str,
    model: str,
    embedding_result=None,
) -> AnalysisResult:
    """
    Perform full resume analysis.

    Args:
        resume_text: Resume content
        job_text: Job description content
        model: Ollama model to use
        embedding_result: Optional EmbeddingResult for semantic scoring

    Returns:
        AnalysisResult with scores and suggestions
    """
    # Extract keywords
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_text)

    # Calculate keyword match
    keyword_score, matching, missing, extra = calculate_keyword_match(resume_keywords, job_keywords)

    # Calculate TF-IDF similarity
    similarity = calculate_similarity(resume_text, job_text)

    # Combined base score - weight embeddings if available
    # Note: base_score is computed but the actual score comes from LLM analysis
    if embedding_result:
        # Use embedding similarity as primary score (more accurate)
        # Keywords as secondary verification
        _base_score = (
            embedding_result.overall_similarity * 0.5  # Semantic (embeddings)
            + keyword_score * 0.3  # Keyword match
            + similarity * 100 * 0.2  # TF-IDF
        )
    else:
        # Original scoring without embeddings
        _base_score = keyword_score * 0.6 + similarity * 100 * 0.4

    # LLM-enhanced analysis
    analyzer = OllamaAnalyzer(model=model)
    result = analyzer.analyze_resume(
        resume_text=resume_text,
        job_text=job_text,
        keyword_score=keyword_score,
        missing_keywords=sorted(missing),
        matching_keywords=sorted(matching),
    )

    # Update with extra keywords
    result.extra_keywords = sorted(extra)

    return result


def generate_adapted_resume(
    resume_text: str,
    job_text: str,
    suggestions: list,
    selected_indices: list[int] | None,
    model: str,
    embedding_gaps: list[str] | None = None,
    missing_keywords: list[str] | None = None,
) -> AdaptedResume:
    """
    Generate an adapted version of the resume.

    Args:
        resume_text: Original resume
        job_text: Job description
        suggestions: List of suggestions
        selected_indices: Indices of suggestions to apply
        model: Ollama model to use
        embedding_gaps: Semantic gaps from embedding analysis
        missing_keywords: Keywords missing from resume

    Returns:
        AdaptedResume with original and adapted text
    """
    analyzer = OllamaAnalyzer(model=model)
    adapter = ResumeAdapter(analyzer=analyzer)

    if selected_indices:
        # Apply specific suggestions
        return adapter.adapt_resume(
            resume_text=resume_text,
            job_text=job_text,
            suggestions=suggestions,
            selected_suggestions=selected_indices,
        )
    else:
        # Full adaptation with semantic context
        return adapter.generate_full_adapted_resume(
            resume_text=resume_text,
            job_text=job_text,
            embedding_gaps=embedding_gaps,
            missing_keywords=missing_keywords,
        )


if __name__ == "__main__":
    main()
