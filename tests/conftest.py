"""Pytest configuration and fixtures for streamlm tests."""

import pytest
import os
from io import StringIO


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to set up mock environment variables for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")


@pytest.fixture
def string_output():
    """Fixture to provide a StringIO object for capturing output."""
    return StringIO()


@pytest.fixture
def sample_markdown():
    """Fixture providing sample markdown content for testing."""
    return {
        "simple": "Hello, world!",
        "paragraph": "This is a paragraph.\n\nThis is another paragraph.",
        "code_block": "```python\nprint('hello')\n```",
        "header": "# Header\n\nSome content.",
        "list": "- Item 1\n- Item 2\n- Item 3",
        "mixed": "# Header\n\nSome text.\n\n```python\ncode()\n```\n\n- List item",
    }
