"""Tests for CLI functionality."""

import pytest
from typer.testing import CliRunner
from llm_cli.cli import app, get_model_provider, is_reasoning_model


runner = CliRunner()


class TestModelProviderDetection:
    """Test model provider detection logic."""

    def test_openai_detection(self):
        assert get_model_provider("gpt-4o") == "openai"
        assert get_model_provider("gpt-4o-mini") == "openai"
        assert get_model_provider("openai/gpt-4") == "openai"

    def test_anthropic_detection(self):
        assert get_model_provider("claude-3-opus") == "anthropic"
        assert get_model_provider("anthropic/claude-3-sonnet") == "anthropic"

    def test_gemini_detection(self):
        assert get_model_provider("gemini/gemini-2.5-flash") == "gemini"
        assert get_model_provider("gemini-pro") == "gemini"

    def test_ollama_detection(self):
        assert get_model_provider("ollama/llama2") == "ollama"
        assert get_model_provider("ollama_chat/mistral") == "ollama"

    def test_deepseek_detection(self):
        assert get_model_provider("deepseek-r1") == "deepseek"
        assert get_model_provider("deepseek-reasoner") == "deepseek"

    def test_xai_detection(self):
        assert get_model_provider("xai/grok-3") == "xai"
        assert get_model_provider("grok-4") == "xai"

    def test_openrouter_detection(self):
        assert get_model_provider("openrouter/anthropic/claude") == "openrouter"


class TestReasoningModelDetection:
    """Test reasoning model detection logic."""

    def test_deepseek_reasoning_models(self):
        assert is_reasoning_model("deepseek-r1")
        assert is_reasoning_model("deepseek/deepseek-reasoner")
        assert is_reasoning_model("DeepSeek-R1")

    def test_openai_o1_models(self):
        assert is_reasoning_model("o1-preview")
        assert is_reasoning_model("o1-mini")
        assert is_reasoning_model("o1-pro")

    def test_grok_models(self):
        assert is_reasoning_model("grok-3")
        assert is_reasoning_model("xai/grok-4")

    def test_non_reasoning_models(self):
        assert not is_reasoning_model("gpt-4o")
        assert not is_reasoning_model("claude-3-opus")
        assert not is_reasoning_model("gemini-pro")


class TestCLIVersion:
    """Test CLI version command."""

    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "streamlm version" in result.stdout
        assert "commit:" in result.stdout


class TestCLIValidation:
    """Test CLI input validation."""

    def test_missing_prompt(self):
        # Chat command requires a prompt argument
        result = runner.invoke(app, [])
        assert result.exit_code != 0
