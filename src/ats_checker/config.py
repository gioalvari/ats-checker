"""Configuration settings for ATS Resume Analyzer."""

from dataclasses import dataclass, field

import ollama


@dataclass
class OllamaConfig:
    """Configuration for Ollama LLM integration."""

    default_model: str = "qwen2.5:7b-instruct"
    host: str = "http://localhost:11434"
    timeout: int = 120
    temperature: float = 0.7
    max_tokens: int = 4096

    def get_available_models(self) -> list[str]:
        """Fetch list of available models from Ollama."""
        try:
            response = ollama.list()
            # Handle different response formats
            if hasattr(response, "models"):
                models = response.models
            elif isinstance(response, dict):
                models = response.get("models", [])
            else:
                models = []

            result = []
            for model in models:
                if hasattr(model, "model"):
                    result.append(model.model)
                elif hasattr(model, "name"):
                    result.append(model.name)
                elif isinstance(model, dict):
                    result.append(model.get("model") or model.get("name", ""))
            return [m for m in result if m]
        except Exception:
            return []

    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        available = self.get_available_models()
        return any(model_name in m for m in available)


@dataclass
class AppConfig:
    """Main application configuration."""

    app_name: str = "ATS Resume Analyzer"
    version: str = "0.1.0"

    # Ollama settings
    ollama: OllamaConfig = field(default_factory=OllamaConfig)

    # Analysis settings
    min_keyword_length: int = 2
    max_suggestions: int = 10

    # Export settings
    default_export_format: str = "pdf"
    supported_export_formats: list[str] = field(default_factory=lambda: ["markdown", "pdf", "docx"])

    # PDF parsing settings
    pdf_extract_tables: bool = True
    pdf_extract_images: bool = False

    # UI settings
    page_title: str = "ATS Resume Analyzer"
    page_icon: str = "📄"
    layout: str = "wide"


# Global config instance
config = AppConfig()
