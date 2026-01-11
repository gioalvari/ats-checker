"""Pytest configuration and fixtures for ATS Resume Analyzer tests."""

import pytest


@pytest.fixture
def sample_resume_text() -> str:
    """Sample resume text for testing."""
    return """
John Doe
Senior Software Engineer
john.doe@email.com | (555) 123-4567 | linkedin.com/in/johndoe

PROFESSIONAL SUMMARY
Experienced software engineer with 8+ years of expertise in Python, JavaScript, and cloud technologies.
Proven track record of delivering scalable applications and leading cross-functional teams.

WORK EXPERIENCE

Senior Software Engineer | TechCorp Inc. | 2020 - Present
- Led development of microservices architecture serving 10M+ daily users
- Reduced system latency by 40% through optimization of database queries
- Mentored team of 5 junior developers and conducted code reviews
- Implemented CI/CD pipelines using Jenkins and GitHub Actions

Software Engineer | StartupXYZ | 2016 - 2020
- Built RESTful APIs using Python Flask and Django frameworks
- Developed React frontend applications with TypeScript
- Managed AWS infrastructure including EC2, S3, and Lambda
- Increased test coverage from 45% to 90%

EDUCATION
Bachelor of Science in Computer Science | State University | 2016

SKILLS
Programming: Python, JavaScript, TypeScript, SQL, Go
Frameworks: Django, Flask, React, Node.js
Cloud: AWS, Docker, Kubernetes
Tools: Git, Jenkins, Jira, Confluence
"""


@pytest.fixture
def sample_job_description() -> str:
    """Sample job description for testing."""
    return """
Senior Backend Engineer

About the Role
We are looking for an experienced Senior Backend Engineer to join our growing team.
You will be responsible for designing and implementing scalable backend services
that power our platform serving millions of users.

Requirements
- 5+ years of experience in software development
- Strong proficiency in Python and Go
- Experience with microservices architecture
- Familiarity with cloud platforms (AWS, GCP, or Azure)
- Experience with containerization (Docker, Kubernetes)
- Strong understanding of databases (PostgreSQL, Redis)
- Excellent problem-solving and communication skills

Nice to Have
- Experience with machine learning systems
- Knowledge of GraphQL
- Contributions to open source projects

What We Offer
- Competitive salary and equity
- Remote work flexibility
- Health insurance and 401k
- Learning and development budget
"""


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Create minimal PDF bytes for testing."""
    # Minimal valid PDF
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test Resume) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000214 00000 n 
trailer
<< /Size 5 /Root 1 0 R >>
startxref
306
%%EOF"""


@pytest.fixture
def empty_text() -> str:
    """Empty text for edge case testing."""
    return ""


@pytest.fixture
def minimal_resume() -> str:
    """Minimal resume for testing."""
    return """
Jane Smith
Software Developer
jane@email.com

SKILLS
Python, JavaScript

EXPERIENCE
Developer at Company | 2022-Present
- Built web applications
"""
