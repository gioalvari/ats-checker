"""Results display components for analysis output."""

import streamlit as st

from ats_checker.models import AnalysisResult, Suggestion


def render_score_card(result: AnalysisResult) -> None:
    """
    Render the main score card with metrics.

    Args:
        result: Analysis result to display
    """
    st.markdown("---")
    st.subheader("📊 Analysis Results")

    # Main metrics row
    col1, col2, col3 = st.columns(3)

    with col1:
        _get_score_color(result.match_score)
        st.metric(
            label="🎯 Overall Match Score",
            value=f"{result.match_score:.0f}%",
            delta=None,
            help="AI-assessed overall compatibility with the job",
        )
        st.progress(result.match_score / 100)

    with col2:
        st.metric(
            label="🔑 Keyword Match",
            value=f"{result.keyword_score:.0f}%",
            delta=None,
            help="Percentage of job keywords found in resume",
        )
        st.progress(result.keyword_score / 100)

    with col3:
        missing_count = len(result.missing_keywords)
        matching_count = len(result.matching_keywords)
        st.metric(
            label="📝 Keywords Found",
            value=f"{matching_count}/{matching_count + missing_count}",
            delta=f"-{missing_count} missing" if missing_count else "All matched!",
            delta_color="inverse" if missing_count else "normal",
        )

    # Score interpretation
    _render_score_interpretation(result.match_score)


def _get_score_color(score: float) -> str:
    """Get color based on score."""
    if score >= 80:
        return "green"
    elif score >= 60:
        return "orange"
    else:
        return "red"


def _render_score_interpretation(score: float) -> None:
    """Render score interpretation message."""
    if score >= 80:
        st.success(
            "🌟 **Excellent Match!** Your resume is well-aligned with this job. "
            "Minor optimizations may still help."
        )
    elif score >= 60:
        st.warning(
            "👍 **Good Match** with room for improvement. "
            "Review the suggestions below to strengthen your application."
        )
    elif score >= 40:
        st.warning(
            "⚠️ **Moderate Match**. Consider significant revisions to better "
            "align your resume with this job's requirements."
        )
    else:
        st.error(
            "❌ **Low Match**. This role may require skills or experience "
            "not reflected in your current resume. Review carefully."
        )


def render_analysis_results(result: AnalysisResult) -> None:
    """
    Render full analysis results.

    Args:
        result: Analysis result to display
    """
    # Summary
    if result.summary:
        st.markdown("### 📋 Summary")
        st.info(result.summary)

    # Strengths and Weaknesses
    col1, col2 = st.columns(2)

    with col1:
        if result.strengths:
            st.markdown("### ✅ Strengths")
            for strength in result.strengths:
                st.markdown(f"- {strength}")

    with col2:
        if result.weaknesses:
            st.markdown("### ⚠️ Areas to Improve")
            for weakness in result.weaknesses:
                st.markdown(f"- {weakness}")

    # Keywords section
    st.markdown("### 🔑 Keyword Analysis")

    kw_col1, kw_col2, kw_col3 = st.columns(3)

    with kw_col1:
        st.markdown("**✅ Matching Keywords**")
        if result.matching_keywords:
            st.write(", ".join(sorted(result.matching_keywords)[:20]))
            if len(result.matching_keywords) > 20:
                st.caption(f"...and {len(result.matching_keywords) - 20} more")
        else:
            st.caption("None found")

    with kw_col2:
        st.markdown("**❌ Missing Keywords**")
        if result.missing_keywords:
            # Highlight important missing keywords
            for kw in sorted(result.missing_keywords)[:15]:
                st.markdown(f"- `{kw}`")
            if len(result.missing_keywords) > 15:
                st.caption(f"...and {len(result.missing_keywords) - 15} more")
        else:
            st.success("All keywords covered!")

    with kw_col3:
        st.markdown("**ℹ️ Extra Keywords**")
        if result.extra_keywords:
            st.caption(", ".join(sorted(result.extra_keywords)[:15]))
        else:
            st.caption("None")


def render_suggestions(suggestions: list[Suggestion]) -> list[int]:
    """
    Render improvement suggestions with selection checkboxes.

    Args:
        suggestions: List of suggestions to display

    Returns:
        List of indices of selected suggestions
    """
    st.markdown("### 💡 Improvement Suggestions")

    if not suggestions:
        st.info("No specific suggestions generated. Your resume looks good!")
        return []

    selected_indices: list[int] = []

    # Group by category
    categories: dict[str, list[tuple[int, Suggestion]]] = {}
    for i, suggestion in enumerate(suggestions):
        cat = suggestion.category.title()
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((i, suggestion))

    # Render by category
    for category, items in categories.items():
        with st.expander(f"**{category}** ({len(items)} suggestions)", expanded=True):
            for idx, suggestion in items:
                col1, col2 = st.columns([0.05, 0.95])

                with col1:
                    selected = st.checkbox(
                        "Select",
                        key=f"suggestion_{idx}",
                        label_visibility="collapsed",
                    )
                    if selected:
                        selected_indices.append(idx)

                with col2:
                    priority_emoji = {
                        "high": "🔴",
                        "medium": "🟡",
                        "low": "🟢",
                    }.get(suggestion.priority, "⚪")

                    st.markdown(f"{priority_emoji} **{suggestion.priority.upper()}**")
                    st.markdown(f"_{suggestion.reasoning}_")

                    if suggestion.original_text:
                        st.markdown("**Original:**")
                        st.code(suggestion.original_text, language=None)

                    st.markdown("**Suggested:**")
                    st.code(suggestion.suggested_text, language=None)

                st.markdown("---")

    return selected_indices


def render_export_options(
    content: str,
    filename_base: str = "optimized_resume",
) -> None:
    """
    Render export buttons for different formats.

    Args:
        content: Resume content to export
        filename_base: Base filename without extension
    """
    st.markdown("### 📥 Export Resume")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.download_button(
            label="📄 Markdown",
            data=content,
            file_name=f"{filename_base}.md",
            mime="text/markdown",
            key="download_md",
        )

    with col2:
        try:
            from ats_checker.resume_exporter import ResumeExporter

            exporter = ResumeExporter(content)
            pdf_bytes = exporter.to_pdf()
            st.download_button(
                label="📕 PDF",
                data=pdf_bytes,
                file_name=f"{filename_base}.pdf",
                mime="application/pdf",
                key="download_pdf",
            )
        except Exception as e:
            st.button("📕 PDF", disabled=True, help=f"PDF export unavailable: {e}")

    with col3:
        try:
            from ats_checker.resume_exporter import ResumeExporter

            exporter = ResumeExporter(content)
            docx_bytes = exporter.to_docx()
            st.download_button(
                label="📘 DOCX",
                data=docx_bytes,
                file_name=f"{filename_base}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="download_docx",
            )
        except Exception as e:
            st.button("📘 DOCX", disabled=True, help=f"DOCX export unavailable: {e}")

    with col4:
        try:
            from ats_checker.resume_exporter import ResumeExporter

            exporter = ResumeExporter(content)
            html_content = exporter.to_html()
            st.download_button(
                label="🌐 HTML",
                data=html_content,
                file_name=f"{filename_base}.html",
                mime="text/html",
                key="download_html",
            )
        except Exception as e:
            st.button("🌐 HTML", disabled=True, help=f"HTML export unavailable: {e}")
