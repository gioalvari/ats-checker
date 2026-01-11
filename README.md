# ATS Resume Analyzer 📄

An AI-powered resume analyzer that scores your resume against job descriptions and generates optimized versions tailored for Applicant Tracking Systems (ATS).

## ✨ Features

- **📊 Resume Analysis**: Score your resume against job descriptions using AI
- **🧠 Semantic Embeddings**: Deep semantic matching with Ollama embeddings
- **🔑 Keyword Matching**: Identify missing keywords and skills with TF-IDF
- **💡 Smart Suggestions**: Get AI-powered recommendations to improve your resume
- **✏️ Resume Optimization**: Generate ATS-optimized versions while preserving your real information
- **🔗 LinkedIn Scraping**: Automatically extract job descriptions from LinkedIn URLs
- **📥 Multiple Export Formats**: Download as PDF, DOCX, Markdown, or HTML
- **🤖 Local AI**: Uses Ollama for privacy-focused, local LLM processing
- **📓 Jupyter Notebooks**: Experiment with embeddings and enhancement strategies

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.ai/) for local LLM

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ats-checker.git
cd ats-checker

# Setup environment and install dependencies
make setup

# Pull required Ollama models
make pull-models
```

### Start Ollama

```bash
# Install Ollama (macOS)
brew install ollama

# Start the server
ollama serve

# Pull models (in another terminal)
ollama pull qwen2.5:7b-instruct   # LLM for analysis
ollama pull nomic-embed-text      # Embeddings
```

### Run the App

```bash
# Start the web interface
make run

# Or directly:
uv run streamlit run main.py
```

Open http://localhost:8501 in your browser.

## 📖 Usage

### Web Interface

1. **Upload/Paste Resume**: Use the left panel to upload a PDF/DOCX or paste your resume text
2. **Add Job Description**: Paste job posting or enter LinkedIn URL to auto-scrape
3. **Enable Embeddings**: Check "Use Embeddings" for semantic analysis (recommended)
4. **Analyze**: Click "Analyze Resume" to get your match score and suggestions
5. **Optimize**: Review suggestions, select which to apply, and generate an optimized resume
6. **Export**: Download your optimized resume in PDF, DOCX, Markdown, or HTML

### CLI

```bash
# Analyze a resume
uv run ats-checker analyze resume.pdf job.txt

# Generate optimized resume
uv run ats-checker optimize resume.pdf job.txt -o optimized.pdf

# List available models
uv run ats-checker models

# Show help
uv run ats-checker --help
```

### Jupyter Notebooks

```bash
# Start notebook server
make notebook

# Or directly:
uv run jupyter notebook notebooks/
```

Available notebooks:
- `01_embedding_analysis.ipynb` - Experiment with embeddings, compare models, tune thresholds

## 📁 Project Structure

```
ats-checker/
├── src/ats_checker/
│   ├── __init__.py              # Package exports
│   ├── py.typed                 # PEP 561 type marker
│   ├── app.py                   # Streamlit web application
│   ├── cli.py                   # Command-line interface
│   ├── config.py                # Configuration settings
│   ├── models.py                # Pydantic data models
│   ├── pdf_parser.py            # PDF/DOCX text extraction
│   ├── keyword_extractor.py     # TF-IDF keyword extraction
│   ├── embedding_analyzer.py    # Semantic embedding analysis
│   ├── llm_analyzer.py          # Ollama LLM integration
│   ├── job_scraper.py           # LinkedIn job scraping
│   ├── resume_generator.py      # Resume adaptation logic
│   ├── resume_exporter.py       # Export to PDF/DOCX/MD/HTML
│   ├── cot_enhancer.py          # Chain-of-Thought enhancement
│   ├── evaluation/              # Evaluation framework
│   │   ├── embedding_eval.py    # Embedding quality metrics
│   │   ├── enhancement_eval.py  # Enhancement quality metrics
│   │   ├── resume_chunker.py    # Resume section parsing
│   │   └── test_dataset.py      # Test cases for evaluation
│   └── ui/                      # Streamlit UI components
│       ├── panels.py            # Input panels
│       ├── results.py           # Results display
│       └── editor.py            # Resume editor
├── tests/                       # Pytest test suite
├── notebooks/                   # Jupyter notebooks
├── data/                        # User data (gitignored)
├── outputs/                     # Generated files (gitignored)
├── main.py                      # Streamlit entry point
├── pyproject.toml               # Project configuration
├── Makefile                     # Development commands
└── README.md
```

## 🛠️ Development

### Available Commands

```bash
make help         # Show all commands
make setup        # Initial setup
make pull-models  # Pull Ollama models
make run          # Start Streamlit app
make notebook     # Start Jupyter server
make lint         # Run ruff linter
make format       # Format code
make typecheck    # Run mypy
make test         # Run tests
make test-cov     # Run tests with coverage
make clean        # Clean build artifacts
make check-ollama # Check if Ollama is running
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
uv run pytest tests/test_keyword_extractor.py -v
```

### Code Quality

```bash
# Lint code
make lint

# Format code
make format

# Type check
make typecheck
```

## ⚙️ Configuration

### Ollama Models

| Type | Model | Purpose |
|------|-------|---------|
| **LLM** | `qwen2.5:7b-instruct` | Analysis & generation (default) |
| **LLM** | `llama3.2` | Fast alternative |
| **Embedding** | `nomic-embed-text` | Semantic similarity (default) |
| **Embedding** | `mxbai-embed-large` | Higher quality, slower |

Select models in the sidebar or set defaults in `src/ats_checker/config.py`.

### Export Formats

| Format | Best For |
|--------|----------|
| PDF | Job applications, printing |
| DOCX | Further editing in Word |
| Markdown | Version control, simple editing |
| HTML | Web viewing, email |

## 🔒 Privacy

All processing happens locally:
- Resume data never leaves your machine
- Ollama runs entirely on your computer
- No external API calls for analysis
- LinkedIn scraping only fetches public job descriptions

## 🧰 Tech Stack

- **Python 3.13** - Modern Python with type hints
- **Streamlit** - Web interface
- **Ollama** - Local LLM inference & embeddings
- **scikit-learn** - TF-IDF similarity
- **pdfplumber** - PDF text extraction
- **python-docx** - DOCX generation
- **WeasyPrint** - PDF generation
- **BeautifulSoup4** - Web scraping
- **Pydantic** - Data validation
- **Jupyter** - Notebooks for experimentation
- **uv** - Package management
- **ruff** - Linting and formatting
- **mypy** - Type checking
- **pytest** - Testing

## 📄 License

MIT License - see LICENSE for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting: `make test && make lint`
5. Submit a pull request

---

Built with ❤️ for job seekers everywhere.
