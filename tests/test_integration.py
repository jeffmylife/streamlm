"""Integration tests for streamlm CLI."""

import pytest
import subprocess
import sys
from pathlib import Path


# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class TestCLIIntegration:
    """Integration tests that verify end-to-end CLI functionality."""

    def test_version_command_integration(self):
        """Test that the version command works end-to-end."""
        result = subprocess.run(
            [sys.executable, "-m", "src.llm_cli.cli", "--version"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "streamlm version" in result.stdout
        assert "0.1.10" in result.stdout

    def test_help_command_integration(self):
        """Test that the help command works."""
        result = subprocess.run(
            [sys.executable, "-m", "src.llm_cli.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "Usage:" in result.stdout or "help" in result.stdout.lower()


class TestMarkdownRendering:
    """Integration tests for markdown rendering."""

    def test_streaming_renderer_imports(self):
        """Test that streaming markdown renderer can be imported."""
        from llm_cli.streaming_markdown import StreamingMarkdownRenderer, LoadingIndicator

        assert StreamingMarkdownRenderer is not None
        assert LoadingIndicator is not None

    def test_renderer_can_handle_simple_text(self):
        """Test that renderer can process simple text."""
        from llm_cli.streaming_markdown import StreamingMarkdownRenderer
        from io import StringIO

        output = StringIO()
        renderer = StreamingMarkdownRenderer(output)
        renderer.add_text("Hello, world!")
        renderer.finalize()

        result = output.getvalue()
        assert len(result) > 0
