# ats_resume_checker/main.py

import spacy
import argparse
import logging
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

def extract_keywords(text: str):
    logger.debug("Extracting keywords")
    doc = nlp(text)
    return set([
        token.lemma_.lower()
        for token in doc
        if token.pos_ in {"NOUN", "PROPN", "VERB"} and not token.is_stop
    ])

def compute_similarity(resume_text: str, job_text: str) -> float:
    logger.debug("Computing cosine similarity")
    tfidf = TfidfVectorizer().fit_transform([resume_text, job_text])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def analyze_resume(resume_text: str, job_text: str) -> dict:
    logger.info("Analyzing resume against job description")
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_text)
    missing = job_keywords - resume_keywords
    extra = resume_keywords - job_keywords
    match_score = compute_similarity(resume_text, job_text)
    return {
        "missing_keywords": sorted(missing),
        "extra_keywords": sorted(extra),
        "match_score": round(match_score * 100, 2),
    }

def print_report(result: dict):
    print("\n--- Resume Match Report ---")
    print(f"Match Score: {result['match_score']}%")
    print("\nMissing Keywords:")
    for kw in result["missing_keywords"]:
        print(f"  - {kw}")
    print("\nExtra Keywords:")
    for kw in result["extra_keywords"]:
        print(f"  - {kw}")

def export_report(result: dict, output_path: Path):
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(f"Report saved to {output_path}")

def run_cli():
    parser = argparse.ArgumentParser(description="ATS Resume Checker")
    parser.add_argument("resume", type=Path, help="Path to resume text file")
    parser.add_argument("job", type=Path, help="Path to job description text file")
    parser.add_argument("--export", type=Path, help="Optional output report JSON path")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Reading resume from {args.resume}")
    logger.info(f"Reading job description from {args.job}")
    resume_text = args.resume.read_text(encoding="utf-8")
    job_text = args.job.read_text(encoding="utf-8")

    result = analyze_resume(resume_text, job_text)
    print_report(result)

    if args.export:
        export_report(result, args.export)

def run_web():
    st.set_page_config(page_title="ATS Resume Checker", layout="centered")
    st.title("ATS Resume Checker")
    st.markdown("Compare your resume against a job description and get insights to optimize it for Applicant Tracking Systems (ATS).")

    resume_file = st.file_uploader("Upload your resume (.txt)", type=["txt"])
    job_file = st.file_uploader("Upload job description (.txt)", type=["txt"])

    if resume_file and job_file:
        logger.info("Processing uploaded files in web app")
        resume_text = resume_file.read().decode("utf-8")
        job_text = job_file.read().decode("utf-8")
        result = analyze_resume(resume_text, job_text)

        st.metric(label="Match Score", value=f"{result['match_score']}%")
        st.subheader("Missing Keywords")
        st.write(result["missing_keywords"])
        st.subheader("Extra Keywords in Resume")
        st.write(result["extra_keywords"])

        st.download_button(
            label="Download Report as JSON",
            data=json.dumps(result, indent=2),
            file_name="ats_report.json",
            mime="application/json"
        )

if __name__ == "__main__":
    import sys
    if "streamlit" in sys.argv:
        logger.info("Launching web interface")
        run_web()
    else:
        logger.info("Running CLI interface")
        run_cli()
