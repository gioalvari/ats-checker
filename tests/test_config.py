"""Tests for application configuration."""

from ats_checker.config import AppConfig, OllamaConfig, config


class TestOllamaConfig:
    """Tests for OllamaConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        ollama_config = OllamaConfig()

        assert ollama_config.default_model == "qwen2.5:7b-instruct"
        assert ollama_config.host == "http://localhost:11434"
        assert ollama_config.timeout == 120
        assert 0 <= ollama_config.temperature <= 1

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        ollama_config = OllamaConfig(
            default_model="mistral",
            timeout=60,
            temperature=0.5,
        )

        assert ollama_config.default_model == "mistral"
        assert ollama_config.timeout == 60
        assert ollama_config.temperature == 0.5

    def test_get_available_models_fallback(self) -> None:
        """Test fallback when Ollama not available."""
        ollama_config = OllamaConfig()
        # When Ollama is not running, returns empty or default
        models = ollama_config.get_available_models()

        assert isinstance(models, list)
        # May be empty if Ollama not running, or contain models if running
        assert isinstance(models, list)


class TestAppConfig:
    """Tests for AppConfig."""

    def test_default_values(self) -> None:
        """Test default app configuration."""
        app_config = AppConfig()

        assert app_config.app_name == "ATS Resume Analyzer"
        assert app_config.version == "0.1.0"
        assert app_config.layout == "wide"
        assert app_config.min_keyword_length == 2

    def test_ollama_nested_config(self) -> None:
        """Test nested Ollama config."""
        app_config = AppConfig()

        assert isinstance(app_config.ollama, OllamaConfig)
        assert app_config.ollama.default_model == "qwen2.5:7b-instruct"

    def test_export_formats(self) -> None:
        """Test supported export formats."""
        app_config = AppConfig()

        assert "pdf" in app_config.supported_export_formats
        assert "docx" in app_config.supported_export_formats
        assert "markdown" in app_config.supported_export_formats


class TestGlobalConfig:
    """Tests for global config instance."""

    def test_global_config_exists(self) -> None:
        """Test that global config is available."""
        assert config is not None
        assert isinstance(config, AppConfig)

    def test_global_config_immutable_access(self) -> None:
        """Test accessing global config values."""
        # Should be able to read values
        assert config.app_name
        assert config.ollama.default_model
