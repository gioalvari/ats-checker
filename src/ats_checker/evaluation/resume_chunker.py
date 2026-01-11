"""Intelligent resume chunking for better LLM processing."""

import re
from dataclasses import dataclass


@dataclass
class ResumeSection:
    """A section of a resume."""

    name: str
    content: str
    start_line: int
    end_line: int
    section_type: str  # header, experience, education, skills, etc.


# Common section header patterns
SECTION_PATTERNS = {
    "summary": [
        r"(?i)^(summary|profile|objective|about\s*me|professional\s*summary)",
        r"(?i)^(career\s*objective|executive\s*summary)",
    ],
    "experience": [
        r"(?i)^(experience|work\s*experience|employment|professional\s*experience)",
        r"(?i)^(work\s*history|career\s*history|relevant\s*experience)",
    ],
    "education": [
        r"(?i)^(education|academic|qualifications|academic\s*background)",
        r"(?i)^(degrees?|educational\s*background)",
    ],
    "skills": [
        r"(?i)^(skills|technical\s*skills|competencies|technologies)",
        r"(?i)^(core\s*competencies|areas\s*of\s*expertise|tech\s*stack)",
    ],
    "projects": [
        r"(?i)^(projects|portfolio|personal\s*projects|key\s*projects)",
    ],
    "certifications": [
        r"(?i)^(certifications?|licenses?|credentials)",
        r"(?i)^(professional\s*certifications?)",
    ],
    "publications": [
        r"(?i)^(publications?|papers?|research)",
    ],
    "languages": [
        r"(?i)^(languages?|language\s*skills)",
    ],
    "awards": [
        r"(?i)^(awards?|honors?|achievements?|recognition)",
    ],
    "references": [
        r"(?i)^(references?|referees?)",
    ],
}


def identify_section_type(line: str) -> str | None:
    """
    Identify if a line is a section header and return its type.

    Returns section type or None if not a header.
    """
    line_clean = line.strip()

    # Skip empty or very short lines
    if len(line_clean) < 3:
        return None

    # Skip lines that look like content (too long, contain typical content patterns)
    if len(line_clean) > 50:
        return None
    if re.search(r"\d{4}\s*[-–]\s*\d{4}|\d{4}\s*[-–]\s*present", line_clean, re.I):
        return None

    for section_type, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, line_clean):
                return section_type

    return None


def chunk_resume(text: str) -> list[ResumeSection]:
    """
    Divide a resume into logical sections.

    Args:
        text: Full resume text

    Returns:
        List of ResumeSection objects
    """
    lines = text.split("\n")
    sections: list[ResumeSection] = []

    current_section_name: str | None = None
    current_section_type: str = "header"
    current_content: list[str] = []
    current_start: int = 0

    for i, line in enumerate(lines):
        section_type = identify_section_type(line)

        if section_type:
            # Save previous section if exists
            if current_content or current_section_name:
                content = "\n".join(current_content).strip()
                if content or current_section_name:
                    sections.append(
                        ResumeSection(
                            name=current_section_name or "Header",
                            content=content,
                            start_line=current_start,
                            end_line=i - 1,
                            section_type=current_section_type,
                        )
                    )

            # Start new section
            current_section_name = line.strip()
            current_section_type = section_type
            current_content = []
            current_start = i
        else:
            current_content.append(line)

    # Don't forget last section
    if current_content or current_section_name:
        content = "\n".join(current_content).strip()
        sections.append(
            ResumeSection(
                name=current_section_name or "Header",
                content=content,
                start_line=current_start,
                end_line=len(lines) - 1,
                section_type=current_section_type,
            )
        )

    return sections


def extract_job_requirements(job_description: str) -> dict:
    """
    Extract structured requirements from a job description.

    Returns dict with:
    - must_have: Required qualifications
    - nice_to_have: Preferred qualifications
    - skills: Technical skills mentioned
    - experience_years: Years of experience required
    - education: Education requirements
    - responsibilities: Key responsibilities
    """
    requirements: dict = {
        "must_have": [],
        "nice_to_have": [],
        "skills": [],
        "experience_years": None,
        "education": [],
        "responsibilities": [],
    }

    lines = job_description.split("\n")
    current_category = "must_have"

    for line in lines:
        line_lower = line.lower().strip()
        line_clean = line.strip()

        # Detect category changes
        if any(
            kw in line_lower
            for kw in [
                "required",
                "must have",
                "requirements",
                "qualifications",
                "you have",
                "you bring",
            ]
        ):
            current_category = "must_have"
            continue
        elif any(
            kw in line_lower
            for kw in ["nice to have", "preferred", "bonus", "ideally", "plus", "advantage"]
        ):
            current_category = "nice_to_have"
            continue
        elif any(
            kw in line_lower
            for kw in ["responsibilities", "you will", "what you", "role", "duties"]
        ):
            current_category = "responsibilities"
            continue

        # Extract years of experience
        years_match = re.search(r"(\d+)\+?\s*years?", line_lower)
        if years_match and requirements["experience_years"] is None:
            requirements["experience_years"] = int(years_match.group(1))

        # Extract education
        if any(
            edu in line_lower
            for edu in ["bachelor", "master", "phd", "degree", "bs ", "ms ", "msc", "bsc"]
        ):
            requirements["education"].append(line_clean)

        # Extract bullet points
        if line_clean.startswith(("•", "-", "*", "·", "–")):
            clean_line = re.sub(r"^[•\-*·–]\s*", "", line_clean)
            if clean_line:
                requirements[current_category].append(clean_line)

        # Extract skills (look for technical terms)
        skill_patterns = [
            r"\b(python|java|scala|sql|r|c\+\+|javascript|typescript)\b",
            r"\b(aws|gcp|azure|kubernetes|docker)\b",
            r"\b(pytorch|tensorflow|keras|scikit-learn|pandas|spark)\b",
            r"\b(machine learning|deep learning|nlp|computer vision)\b",
            r"\b(mlops|ci/cd|devops|airflow|mlflow)\b",
        ]
        for pattern in skill_patterns:
            matches = re.findall(pattern, line_lower)
            for match in matches:
                if match not in requirements["skills"]:
                    requirements["skills"].append(match)

    return requirements


def print_resume_structure(sections: list[ResumeSection]) -> None:
    """Print the structure of a chunked resume."""
    print("\n" + "=" * 50)
    print("RESUME STRUCTURE")
    print("=" * 50)

    for i, section in enumerate(sections, 1):
        content_preview = section.content[:100].replace("\n", " ")
        if len(section.content) > 100:
            content_preview += "..."

        print(f"\n{i}. {section.name} ({section.section_type})")
        print(f"   Lines: {section.start_line}-{section.end_line}")
        print(f"   Content: {content_preview}")


def print_job_requirements(requirements: dict) -> None:
    """Print extracted job requirements."""
    print("\n" + "=" * 50)
    print("JOB REQUIREMENTS")
    print("=" * 50)

    if requirements["experience_years"]:
        print(f"\n📅 Experience: {requirements['experience_years']}+ years")

    if requirements["education"]:
        print("\n🎓 Education:")
        for edu in requirements["education"]:
            print(f"  - {edu}")

    if requirements["must_have"]:
        print("\n✅ Must Have:")
        for req in requirements["must_have"][:10]:
            print(f"  - {req[:80]}")

    if requirements["nice_to_have"]:
        print("\n🌟 Nice to Have:")
        for req in requirements["nice_to_have"][:10]:
            print(f"  - {req[:80]}")

    if requirements["skills"]:
        print("\n💻 Skills Mentioned:")
        print(f"  {', '.join(requirements['skills'])}")


if __name__ == "__main__":
    # Example usage
    sample_resume = """
John Doe
john.doe@email.com | +1-555-0123

Summary
Experienced data scientist with 5 years in ML and analytics.

Experience

Senior Data Scientist | TechCorp | 2020-2024
- Built ML models for forecasting
- Led team of 3 analysts
- Reduced costs by 35%

Data Scientist | StartupX | 2018-2020
- Developed recommendation systems
- Improved user engagement by 25%

Education

MSc Machine Learning, Stanford University, 2018
BSc Computer Science, MIT, 2016

Skills

Python, TensorFlow, PyTorch, AWS, SQL, Spark

Certifications

AWS Certified Machine Learning Specialty
    """

    sample_job = """
Senior Data Scientist

About the Role:
We're looking for a Senior Data Scientist to join our team.

Responsibilities:
- Design and deploy ML models in production
- Build forecasting systems
- Mentor junior team members
- Work with stakeholders

Requirements:
- 5+ years of experience in data science
- Strong Python and SQL skills
- Experience with cloud platforms (AWS/GCP)
- Machine learning expertise
- MS or PhD in relevant field

Nice to Have:
- Experience with Spark
- MLOps experience
- Publications in top venues
    """

    print("Testing Resume Chunker...")
    sections = chunk_resume(sample_resume)
    print_resume_structure(sections)

    print("\n\nTesting Job Requirements Extractor...")
    requirements = extract_job_requirements(sample_job)
    print_job_requirements(requirements)
