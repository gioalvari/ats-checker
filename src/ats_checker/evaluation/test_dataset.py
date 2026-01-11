"""Test dataset for embedding and enhancement evaluation."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TestCase:
    """A test case for embedding evaluation."""

    resume: str
    job: str
    expected_score: float  # 0-100, human judgment
    expected_match: bool  # Should this be considered a good match?
    category: str  # e.g., "exact_match", "partial_match", "mismatch", "domain_switch"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TestCase":
        return cls(**data)


# Pre-built test cases covering various scenarios
TEST_CASES: list[TestCase] = [
    # ===== EXACT MATCH CASES (expected 80-95) =====
    TestCase(
        resume="""
Senior Data Scientist | 6 Years Experience

EXPERIENCE:
- Lead Data Scientist at TechCorp (2020-2024)
  • Built ML models for demand forecasting using Prophet and LSTM
  • Deployed models on AWS SageMaker with CI/CD pipelines
  • Mentored team of 3 junior data scientists
  • Reduced forecast error by 35% saving $2M annually

SKILLS:
Python, PyTorch, TensorFlow, AWS (S3, SageMaker, Lambda), Spark, SQL, MLOps, Docker, Kubernetes

EDUCATION:
MSc Machine Learning, Stanford University
        """,
        job="""
Senior Data Scientist - ML Platform

We're looking for a Senior Data Scientist to join our ML team.

Responsibilities:
- Design and deploy ML models in production
- Build forecasting systems for demand planning
- Work with AWS (S3, SageMaker, Lambda)
- Develop MLOps pipelines with CI/CD
- Mentor junior data scientists

Requirements:
- 5+ years in data science
- Strong Python (pandas, scikit-learn, PyTorch)
- Time series forecasting experience
- AWS cloud services knowledge
- Spark and distributed computing
        """,
        expected_score=90,
        expected_match=True,
        category="exact_match",
        notes="Perfect alignment - same role, skills, and experience level",
    ),
    TestCase(
        resume="""
Machine Learning Engineer | 5 Years

EXPERIENCE:
- ML Engineer at DataCo (2019-2024)
  • Developed recommendation systems serving 10M users
  • Built real-time inference pipelines with Kafka and Flink
  • Implemented A/B testing framework
  • Reduced model latency by 60%

SKILLS:
Python, Java, Scala, PyTorch, MLflow, Kubernetes, AWS, GCP, Airflow

EDUCATION:
PhD Computer Science, MIT
        """,
        job="""
Machine Learning Engineer

Join our ML infrastructure team!

Requirements:
- 4+ years ML engineering experience
- Production ML systems experience
- Python and one of Java/Scala
- Kubernetes and cloud platforms
- Real-time systems knowledge
        """,
        expected_score=88,
        expected_match=True,
        category="exact_match",
        notes="Strong match for ML Engineer role",
    ),
    # ===== PARTIAL MATCH CASES (expected 50-75) =====
    TestCase(
        resume="""
Data Analyst | 4 Years Experience

EXPERIENCE:
- Senior Data Analyst at RetailCo (2020-2024)
  • Created dashboards in Tableau and Power BI
  • SQL queries for business reporting
  • Basic Python scripting for data cleaning
  • A/B test analysis for marketing team

SKILLS:
SQL, Python (pandas, matplotlib), Tableau, Power BI, Excel, Statistics

EDUCATION:
BSc Statistics, UCLA
        """,
        job="""
Data Scientist

We need a Data Scientist for our analytics team.

Requirements:
- 3+ years data experience
- Strong Python and SQL
- Machine learning knowledge
- Statistical analysis skills
- Visualization tools experience
        """,
        expected_score=65,
        expected_match=True,
        category="partial_match",
        notes="Data Analyst applying for Data Scientist - some overlap but missing ML",
    ),
    TestCase(
        resume="""
Software Engineer | 3 Years

EXPERIENCE:
- Backend Developer at StartupX (2021-2024)
  • Built REST APIs with Python Flask
  • PostgreSQL database optimization
  • Basic ML model integration
  • Docker and CI/CD pipelines

SKILLS:
Python, Java, PostgreSQL, MongoDB, Docker, Git, AWS EC2

EDUCATION:
BSc Computer Science
        """,
        job="""
ML Engineer

Looking for ML Engineer to productionize models.

Requirements:
- 2+ years ML/software engineering
- Python expertise
- Model deployment experience
- Docker and Kubernetes
- Cloud platforms (AWS/GCP)
        """,
        expected_score=55,
        expected_match=True,
        category="partial_match",
        notes="SWE with some ML exposure - could transition",
    ),
    TestCase(
        resume="""
Business Intelligence Analyst | 5 Years

EXPERIENCE:
- BI Analyst at FinanceCorp (2019-2024)
  • Built executive dashboards
  • ETL pipelines with SSIS
  • Financial forecasting models in Excel
  • Stakeholder presentations

SKILLS:
SQL, Excel (advanced), Tableau, Power BI, SSIS, Basic Python

EDUCATION:
MBA, Finance concentration
        """,
        job="""
Senior Data Scientist

Join our data science team!

Requirements:
- 5+ years experience
- Advanced Python and ML
- Deep learning frameworks
- Cloud ML platforms
- Research background preferred
        """,
        expected_score=40,
        expected_match=False,
        category="partial_match",
        notes="BI background - significant skill gap for DS role",
    ),
    # ===== MISMATCH CASES (expected 10-30) =====
    TestCase(
        resume="""
Frontend Developer | 4 Years

EXPERIENCE:
- Senior Frontend Dev at WebAgency (2020-2024)
  • React and TypeScript applications
  • UI/UX implementation
  • Performance optimization
  • Responsive design

SKILLS:
JavaScript, TypeScript, React, Vue.js, CSS, HTML, Figma, Jest

EDUCATION:
BSc Web Development
        """,
        job="""
Senior Data Scientist

ML team needs experienced data scientist.

Requirements:
- 5+ years data science
- Python, R, SQL
- Machine learning expertise
- Statistics and math background
- PhD preferred
        """,
        expected_score=15,
        expected_match=False,
        category="mismatch",
        notes="Frontend dev - completely different domain",
    ),
    TestCase(
        resume="""
Marketing Manager | 7 Years

EXPERIENCE:
- Marketing Director at BrandCo (2018-2024)
  • Led team of 10 marketers
  • Campaign strategy and execution
  • Budget management ($5M annually)
  • Brand partnerships

SKILLS:
Marketing Strategy, Team Leadership, Budget Management, Salesforce, HubSpot

EDUCATION:
MBA Marketing
        """,
        job="""
Machine Learning Engineer

Build ML infrastructure at scale.

Requirements:
- 3+ years ML engineering
- Python, C++
- Distributed systems
- Model optimization
- GPU programming
        """,
        expected_score=10,
        expected_match=False,
        category="mismatch",
        notes="Marketing background - no technical overlap",
    ),
    TestCase(
        resume="""
Mechanical Engineer | 6 Years

EXPERIENCE:
- Senior Mechanical Engineer at AutoCorp (2018-2024)
  • CAD design for automotive parts
  • FEA simulations
  • Manufacturing process optimization
  • Patent holder (3 patents)

SKILLS:
AutoCAD, SolidWorks, ANSYS, MATLAB, Manufacturing, Project Management

EDUCATION:
MSc Mechanical Engineering
        """,
        job="""
Data Engineer

Build data pipelines at scale.

Requirements:
- 4+ years data engineering
- Python, Scala, SQL
- Spark, Kafka, Airflow
- Cloud data platforms
- Data modeling expertise
        """,
        expected_score=12,
        expected_match=False,
        category="mismatch",
        notes="Mechanical engineer - different engineering domain",
    ),
    # ===== DOMAIN SWITCH CASES (expected 30-50) =====
    TestCase(
        resume="""
PhD Researcher - Physics | 4 Years Postdoc

EXPERIENCE:
- Postdoctoral Researcher at CERN (2020-2024)
  • Statistical analysis of particle collision data
  • Python for data analysis (numpy, scipy)
  • Large dataset processing
  • Published 8 papers

SKILLS:
Python, C++, ROOT, Statistical Analysis, Scientific Computing, LaTeX

EDUCATION:
PhD Particle Physics, Cambridge
        """,
        job="""
Data Scientist

Analytics team needs data scientist.

Requirements:
- Strong statistics background
- Python proficiency
- Data analysis experience
- Communication skills
- Industry experience preferred
        """,
        expected_score=60,
        expected_match=True,
        category="domain_switch",
        notes="PhD physicist - strong analytical skills, lacks industry exp",
    ),
    TestCase(
        resume="""
Quantitative Analyst | 5 Years

EXPERIENCE:
- Quant at HedgeFund (2019-2024)
  • Built trading algorithms
  • Risk modeling and backtesting
  • Time series analysis
  • Python and C++ development

SKILLS:
Python, C++, R, SQL, Statistics, Time Series, Risk Modeling, Bloomberg

EDUCATION:
MSc Financial Engineering
        """,
        job="""
Senior Data Scientist - Forecasting

Build demand forecasting models.

Requirements:
- 5+ years quantitative experience
- Time series expertise
- Python and SQL
- Statistical modeling
- Business stakeholder communication
        """,
        expected_score=75,
        expected_match=True,
        category="domain_switch",
        notes="Quant to DS - strong quantitative overlap",
    ),
    # ===== JUNIOR VS SENIOR MISMATCH (expected 35-55) =====
    TestCase(
        resume="""
Junior Data Scientist | 1 Year

EXPERIENCE:
- Data Science Intern at TechStartup (2023-2024)
  • Built classification models
  • Data cleaning and EDA
  • Jupyter notebooks and reports

SKILLS:
Python, pandas, scikit-learn, SQL basics, Jupyter

EDUCATION:
BSc Data Science (2023)
        """,
        job="""
Staff Data Scientist

Lead our ML initiatives.

Requirements:
- 8+ years data science
- Team leadership experience
- System design skills
- PhD preferred
- Published research
        """,
        expected_score=25,
        expected_match=False,
        category="seniority_mismatch",
        notes="Junior applying for Staff level - major experience gap",
    ),
    TestCase(
        resume="""
Principal Data Scientist | 12 Years

EXPERIENCE:
- Principal DS at FAANG (2018-2024)
  • Led team of 15 data scientists
  • Company-wide ML strategy
  • Research publications (20+ papers)
  • $50M revenue impact

SKILLS:
Python, Deep Learning, NLP, Computer Vision, Team Leadership, Strategy

EDUCATION:
PhD Machine Learning, Stanford
        """,
        job="""
Junior Data Analyst

Entry-level analytics role.

Requirements:
- 0-2 years experience
- Basic SQL and Excel
- Eagerness to learn
- Bachelor's degree
        """,
        expected_score=30,
        expected_match=False,
        category="seniority_mismatch",
        notes="Overqualified - Principal applying for Junior role",
    ),
]


def load_test_cases(filepath: str | Path) -> list[TestCase]:
    """Load test cases from a JSON file."""
    path = Path(filepath)
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    return [TestCase.from_dict(case) for case in data]


def save_test_cases(cases: list[TestCase], filepath: str | Path) -> None:
    """Save test cases to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump([case.to_dict() for case in cases], f, indent=2)


def get_cases_by_category(category: str) -> list[TestCase]:
    """Get test cases filtered by category."""
    return [case for case in TEST_CASES if case.category == category]


def print_test_case_summary() -> None:
    """Print summary of available test cases."""
    categories = {}
    for case in TEST_CASES:
        if case.category not in categories:
            categories[case.category] = []
        categories[case.category].append(case)

    print("\n" + "=" * 50)
    print("TEST DATASET SUMMARY")
    print("=" * 50)
    print(f"\nTotal test cases: {len(TEST_CASES)}")
    print("\nBy category:")
    for cat, cases in categories.items():
        avg_score = sum(c.expected_score for c in cases) / len(cases)
        match_count = sum(1 for c in cases if c.expected_match)
        print(
            f"  {cat}: {len(cases)} cases (avg expected: {avg_score:.0f}%, {match_count} matches)"
        )


if __name__ == "__main__":
    print_test_case_summary()
