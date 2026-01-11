"""Input panel components for resume and job description."""

import streamlit as st

from ats_checker.job_scraper import is_valid_url, scrape_job_description
from ats_checker.pdf_parser import extract_text_from_file


def render_resume_panel() -> tuple[str | None, str | None]:
    """
    Render the resume input panel (left side).

    Returns:
        Tuple of (extracted_text, file_name) or (None, None) if no input
    """
    st.subheader("📄 Your Resume")

    tab_upload, tab_paste = st.tabs(["📁 Upload File", "📝 Paste Text"])

    extracted_text: str | None = None
    file_name: str | None = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload your resume",
            type=["pdf", "docx", "txt"],
            key="resume_upload",
            help="Supported formats: PDF, DOCX, TXT",
        )

        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.read()
                extracted_text = extract_text_from_file(file_bytes, file_name=uploaded_file.name)
                file_name = uploaded_file.name
                st.success(f"✅ Loaded: {uploaded_file.name}")

                with st.expander("Preview extracted text", expanded=False):
                    st.text_area(
                        "Extracted content",
                        value=extracted_text[:2000] + ("..." if len(extracted_text) > 2000 else ""),
                        height=200,
                        disabled=True,
                        key="resume_preview",
                    )
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")

    with tab_paste:
        pasted_text = st.text_area(
            "Paste your resume text",
            height=300,
            placeholder="Paste your resume content here...",
            key="resume_paste",
        )

        if pasted_text.strip():
            extracted_text = pasted_text
            file_name = "pasted_resume.txt"

    # Show character count
    if extracted_text:
        st.caption(
            f"📊 {len(extracted_text):,} characters | ~{len(extracted_text.split()):,} words"
        )

    return extracted_text, file_name


def render_job_panel() -> tuple[str | None, str | None]:
    """
    Render the job description input panel (right side).

    Returns:
        Tuple of (extracted_text, source) or (None, None) if no input
    """
    st.subheader("💼 Job Description")

    tab_upload, tab_paste, tab_url = st.tabs(["📁 Upload File", "📝 Paste Text", "🔗 From URL"])

    extracted_text: str | None = None
    source: str | None = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload job description",
            type=["pdf", "docx", "txt"],
            key="job_upload",
            help="Supported formats: PDF, DOCX, TXT",
        )

        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.read()
                extracted_text = extract_text_from_file(file_bytes, file_name=uploaded_file.name)
                source = uploaded_file.name
                st.success(f"✅ Loaded: {uploaded_file.name}")

                with st.expander("Preview extracted text", expanded=False):
                    st.text_area(
                        "Extracted content",
                        value=extracted_text[:2000] + ("..." if len(extracted_text) > 2000 else ""),
                        height=200,
                        disabled=True,
                        key="job_preview",
                    )
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")

    with tab_paste:
        pasted_text = st.text_area(
            "Paste job description",
            height=300,
            placeholder="Paste the job description here...\n\nTip: Copy from LinkedIn, Indeed, or company career pages.",
            key="job_paste",
        )

        if pasted_text.strip():
            extracted_text = pasted_text
            source = "pasted"

    with tab_url:
        st.markdown(
            "🔗 **Paste a LinkedIn or job site URL** to automatically extract the job description."
        )

        url_input = st.text_input(
            "Job posting URL",
            placeholder="https://linkedin.com/jobs/view/...",
            key="job_url",
            help="Supports LinkedIn, Indeed, and most job sites.",
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            fetch_btn = st.button("🔍 Fetch", key="fetch_job_btn", type="primary")

        if url_input and fetch_btn:
            if is_valid_url(url_input):
                with st.spinner("Fetching job description..."):
                    try:
                        scraped_text = scrape_job_description(url_input)
                        if scraped_text:
                            extracted_text = scraped_text
                            source = url_input
                            st.success("✅ Job description extracted!")

                            # Store in session state for persistence
                            st.session_state.scraped_job_text = scraped_text
                            st.session_state.scraped_job_url = url_input

                            with st.expander("Preview extracted text", expanded=True):
                                st.text_area(
                                    "Extracted content",
                                    value=scraped_text[:3000]
                                    + ("..." if len(scraped_text) > 3000 else ""),
                                    height=300,
                                    disabled=True,
                                    key="scraped_job_preview",
                                )
                    except ValueError as e:
                        st.error(f"❌ {e}")
                        st.info(
                            "💡 Try copying the job description manually and pasting it in the 'Paste Text' tab."
                        )
            else:
                st.warning("⚠️ Please enter a valid URL")

        # Check if we have previously scraped content
        if "scraped_job_text" in st.session_state and st.session_state.scraped_job_text:
            if not extracted_text:  # Only use if no new extraction
                extracted_text = st.session_state.scraped_job_text
                source = st.session_state.get("scraped_job_url", "scraped")
                st.caption(f"📎 Using previously fetched: {source[:50]}...")

        st.markdown("---")
        st.caption(
            "⚠️ **Note**: Some LinkedIn jobs require login. "
            "If fetching fails, copy the description manually."
        )

    # Show character count
    if extracted_text:
        st.caption(
            f"📊 {len(extracted_text):,} characters | ~{len(extracted_text.split()):,} words"
        )

    return extracted_text, source


def render_model_selector(available_models: list[str], default_model: str) -> str:
    """
    Render Ollama model selector in sidebar.

    Args:
        available_models: List of available model names
        default_model: Default model to select

    Returns:
        Selected model name
    """
    import subprocess

    st.sidebar.subheader("🤖 AI Model Settings")

    # Check Ollama status and add start button
    if not available_models:
        st.sidebar.warning("⚠️ No Ollama models found. Make sure Ollama is running.")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🚀 Start Ollama", key="start_ollama_btn"):
                try:
                    # Start ollama serve in background
                    subprocess.Popen(
                        ["ollama", "serve"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    st.success("Starting Ollama...")
                    st.info("Please refresh the page in a few seconds.")
                except FileNotFoundError:
                    st.error("Ollama not installed. Install from ollama.ai")
                except Exception as e:
                    st.error(f"Failed to start: {e}")

        with col2:
            if st.button("🔄 Refresh", key="refresh_models_btn"):
                st.rerun()

        st.sidebar.code("ollama serve", language="bash")
        return default_model

    # Find default model index
    default_idx = 0
    for i, model in enumerate(available_models):
        if default_model in model:
            default_idx = i
            break

    selected = st.sidebar.selectbox(
        "Select Model",
        options=available_models,
        index=default_idx,
        key="model_selector",
        help="Choose the Ollama model for analysis",
    )

    # Model info
    st.sidebar.caption(f"Using: {selected}")

    # Pull new model option
    with st.sidebar.expander("Download new model"):
        new_model = st.text_input(
            "Model name",
            placeholder="llama3.2",
            key="new_model_input",
        )
        if st.button("Pull Model", key="pull_model_btn"):
            st.info(f"Run in terminal: `ollama pull {new_model}`")

    return selected or default_model
