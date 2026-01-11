"""Resume editor component for modifying adapted resumes."""

import difflib

import streamlit as st

from ats_checker.models import AdaptedResume


def render_resume_editor(
    adapted_resume: AdaptedResume | None = None,
    key_prefix: str = "editor",
) -> str | None:
    """
    Render an editable resume editor.

    Args:
        adapted_resume: AdaptedResume object with original and adapted text
        key_prefix: Prefix for widget keys

    Returns:
        Current editor content or None
    """
    st.markdown("### ✏️ Optimized Resume")

    if adapted_resume is None:
        st.info(
            "Click 'Generate Optimized Resume' above to create an adapted version "
            "of your resume tailored to the job description."
        )
        return None

    # Show important notice
    st.info(
        "📋 **Review carefully**: The AI optimizes wording and keyword placement "
        "while preserving your real experience. Always verify the output before using."
    )

    # Show changes made (including any warnings)
    if adapted_resume.changes_made:
        with st.expander("📝 Changes Applied", expanded=True):
            for change in adapted_resume.changes_made:
                if "⚠️" in change:
                    st.warning(change)
                else:
                    st.markdown(f"- {change}")

    # Show preserved facts
    if adapted_resume.preserved_facts:
        with st.expander("✅ Fact Verification", expanded=False):
            for fact in adapted_resume.preserved_facts:
                if "✅" in fact:
                    st.success(fact)
                elif "⚠️" in fact:
                    st.warning(fact)
                else:
                    st.markdown(f"- {fact}")

    # Initialize edited content in session state if not present
    editor_key = f"{key_prefix}_adapted_text"
    if editor_key not in st.session_state:
        st.session_state[editor_key] = adapted_resume.adapted_text

    # Tabs for viewing
    tab_edit, tab_original, tab_diff = st.tabs(["📝 Edit Adapted", "📄 Original", "🔄 Compare"])

    # Track edited content
    edited_content = st.session_state.get(editor_key, adapted_resume.adapted_text)

    with tab_edit:
        edited_content = st.text_area(
            "Adapted Resume (editable)",
            value=st.session_state[editor_key],
            height=500,
            key=f"{key_prefix}_text_input",
            help="Edit the adapted resume. Changes are reflected in exports.",
        )
        # Update session state with edited content
        st.session_state[editor_key] = edited_content

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset to Adapted", key=f"{key_prefix}_reset_adapted"):
                st.session_state[editor_key] = adapted_resume.adapted_text
                st.rerun()
        with col2:
            if st.button("↩️ Reset to Original", key=f"{key_prefix}_reset_original"):
                st.session_state[editor_key] = adapted_resume.original_text
                st.rerun()

    with tab_original:
        st.markdown("#### 📄 Your Original Resume")
        st.markdown("This is your original resume before any AI optimization:")
        _render_formatted_text(adapted_resume.original_text, key=f"{key_prefix}_orig_view")

    with tab_diff:
        st.markdown("#### 🔍 Compare Changes")
        st.markdown("See what was changed between your original and the optimized version:")
        _render_diff_view(adapted_resume.original_text, edited_content)

    return edited_content


def _render_formatted_text(text: str, key: str = "formatted") -> None:
    """Render text with proper formatting in a scrollable container."""
    if not text:
        st.warning("No text available to display.")
        return

    # Use a text_area in disabled mode for reliable display
    st.text_area(
        "Resume Content",
        value=text,
        height=500,
        disabled=True,
        key=f"{key}_display",
        label_visibility="collapsed",
    )


def _render_diff_view(original: str, adapted: str) -> None:
    """Render a side-by-side diff view with highlighted changes."""

    if not original or not adapted:
        st.warning("Both original and adapted text are required for comparison.")
        return

    st.caption("🟢 Added text | 🔴 Removed text | ⚪ Unchanged")

    # Generate unified diff
    original_lines = original.splitlines(keepends=True)
    adapted_lines = adapted.splitlines(keepends=True)

    # Create side-by-side columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📄 Original**")
        _render_highlighted_text(original, adapted, is_original=True)

    with col2:
        st.markdown("**✨ Optimized**")
        _render_highlighted_text(original, adapted, is_original=False)

    st.markdown("---")

    # Detailed diff view
    with st.expander("📊 Detailed Line-by-Line Changes", expanded=False):
        _render_line_diff(original_lines, adapted_lines)

    # Word statistics
    _render_diff_stats(original, adapted)


def _render_highlighted_text(original: str, adapted: str, is_original: bool) -> None:
    """Render text with highlighting for changes."""

    # CSS for highlighted text
    css = """
    <style>
    .diff-container {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        max-height: 400px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 12px;
        line-height: 1.5;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .diff-added {
        background-color: #d4edda;
        color: #155724;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .diff-removed {
        background-color: #f8d7da;
        color: #721c24;
        padding: 2px 4px;
        border-radius: 3px;
        text-decoration: line-through;
    }
    .diff-line-added {
        background-color: #e6ffec;
        border-left: 3px solid #28a745;
        padding-left: 10px;
        margin: 2px 0;
    }
    .diff-line-removed {
        background-color: #ffebe9;
        border-left: 3px solid #dc3545;
        padding-left: 10px;
        margin: 2px 0;
    }
    </style>
    """

    text = original if is_original else adapted
    # Escape HTML and format
    text_escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text_formatted = text_escaped.replace("\n", "<br>")

    st.markdown(css + f'<div class="diff-container">{text_formatted}</div>', unsafe_allow_html=True)


def _render_line_diff(original_lines: list[str], adapted_lines: list[str]) -> None:
    """Render detailed line-by-line diff with colors."""

    differ = difflib.unified_diff(
        original_lines, adapted_lines, fromfile="Original", tofile="Optimized", lineterm=""
    )

    diff_html = []
    diff_html.append(
        '<div style="font-family: monospace; font-size: 12px; background: #f6f8fa; padding: 10px; border-radius: 6px; max-height: 400px; overflow-y: auto;">'
    )

    for line in differ:
        line_escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if line.startswith("+") and not line.startswith("+++"):
            diff_html.append(
                f'<div style="background-color: #e6ffec; color: #22863a; padding: 2px 5px;">{line_escaped}</div>'
            )
        elif line.startswith("-") and not line.startswith("---"):
            diff_html.append(
                f'<div style="background-color: #ffebe9; color: #cb2431; padding: 2px 5px;">{line_escaped}</div>'
            )
        elif line.startswith("@@"):
            diff_html.append(
                f'<div style="background-color: #f1f8ff; color: #0366d6; padding: 2px 5px; font-weight: bold;">{line_escaped}</div>'
            )
        else:
            diff_html.append(f'<div style="padding: 2px 5px; color: #586069;">{line_escaped}</div>')

    diff_html.append("</div>")

    if len(diff_html) > 2:  # More than just container tags
        st.markdown("".join(diff_html), unsafe_allow_html=True)
    else:
        st.info("No significant line-by-line differences detected.")


def _render_diff_stats(original: str, adapted: str) -> None:
    """Render statistics about the differences."""

    # Word-level analysis
    original_words = original.lower().split()
    adapted_words = adapted.lower().split()

    original_set = set(original_words)
    adapted_set = set(adapted_words)

    added_words = adapted_set - original_set
    removed_words = original_set - adapted_set
    common_words = original_set & adapted_set

    # Character counts
    original_chars = len(original)
    adapted_chars = len(adapted)
    char_diff = adapted_chars - original_chars

    st.markdown("### 📈 Change Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Words Added", len(added_words), delta=f"+{len(added_words)}", delta_color="normal"
        )
    with col2:
        st.metric(
            "Words Removed",
            len(removed_words),
            delta=f"-{len(removed_words)}",
            delta_color="inverse",
        )
    with col3:
        st.metric("Words Preserved", len(common_words))
    with col4:
        delta_str = f"+{char_diff}" if char_diff > 0 else str(char_diff)
        st.metric("Character Change", f"{adapted_chars:,}", delta=delta_str)

    # Show added/removed words
    col1, col2 = st.columns(2)

    with col1:
        if added_words:
            with st.expander(f"🟢 New Keywords ({len(added_words)})", expanded=False):
                # Sort by length (longer words often more meaningful)
                sorted_added = sorted(added_words, key=len, reverse=True)[:40]
                st.markdown(" • ".join(f"`{w}`" for w in sorted_added))

    with col2:
        if removed_words:
            with st.expander(f"🔴 Removed Words ({len(removed_words)})", expanded=False):
                sorted_removed = sorted(removed_words, key=len, reverse=True)[:40]
                st.markdown(" • ".join(f"`{w}`" for w in sorted_removed))


def render_generate_button(
    resume_text: str | None,
    job_text: str | None,
    key: str = "generate_btn",
) -> bool:
    """
    Render the generate optimized resume button.

    Args:
        resume_text: Resume text (enables button if present)
        job_text: Job description text (enables button if present)
        key: Button key

    Returns:
        True if button was clicked
    """
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        disabled = not (resume_text and job_text)

        if disabled:
            st.warning("⬆️ Upload both resume and job description to generate optimized resume")

        return st.button(
            "🚀 Generate Optimized Resume",
            type="primary",
            disabled=disabled,
            use_container_width=True,
            key=key,
            help="Generate an ATS-optimized version of your resume",
        )


def render_apply_suggestions_button(
    selected_count: int,
    key: str = "apply_suggestions_btn",
) -> bool:
    """
    Render button to apply selected suggestions.

    Args:
        selected_count: Number of selected suggestions
        key: Button key

    Returns:
        True if button was clicked
    """
    return st.button(
        f"✨ Apply {selected_count} Selected Suggestion{'s' if selected_count != 1 else ''}",
        disabled=selected_count == 0,
        key=key,
        help="Apply the selected suggestions to the adapted resume",
    )
