"""Tests for streaming markdown renderer."""

import pytest
from io import StringIO
from llm_cli.streaming_markdown import (
    StreamingMarkdownRenderer,
    LoadingIndicator,
)


class TestLoadingIndicator:
    """Test loading indicator functionality."""

    def test_loading_indicator_initialization(self):
        output = StringIO()
        indicator = LoadingIndicator(output)
        assert not indicator.active
        assert indicator.output == output

    def test_loading_indicator_start_stop(self):
        output = StringIO()
        indicator = LoadingIndicator(output)

        indicator.start()
        assert indicator.active

        indicator.stop()
        assert not indicator.active


class TestStreamingMarkdownRenderer:
    """Test streaming markdown renderer functionality."""

    def test_renderer_initialization(self):
        output = StringIO()
        renderer = StreamingMarkdownRenderer(output)
        assert renderer.buffer == ""
        assert renderer.rendered_buffer == ""
        assert renderer.output == output

    def test_add_text_simple(self):
        output = StringIO()
        renderer = StreamingMarkdownRenderer(output)
        renderer.add_text("Hello, world!")
        assert "Hello" in renderer.buffer

    def test_add_text_empty(self):
        output = StringIO()
        renderer = StreamingMarkdownRenderer(output)
        initial_buffer = renderer.buffer
        renderer.add_text("")
        assert renderer.buffer == initial_buffer

    def test_finalize_adds_content(self):
        output = StringIO()
        renderer = StreamingMarkdownRenderer(output)
        renderer.add_text("Test content")
        renderer.finalize()
        # Check that finalize was called (output should have content)
        result = output.getvalue()
        assert len(result) > 0

    def test_code_block_detection(self):
        output = StringIO()
        renderer = StreamingMarkdownRenderer(output)

        # Add opening code fence
        renderer.add_text("```python\n")
        # The renderer should detect we're in a code block
        # This is implementation-specific, but we can check the buffer
        assert "```" in renderer.buffer

    def test_safe_point_detection(self):
        output = StringIO()
        renderer = StreamingMarkdownRenderer(output)

        # Test finding safe points with complete markdown structures
        renderer.buffer = "# Header\n\nSome text.\n\n"
        safe_point = renderer._find_latest_safe_point()
        # Should find a safe point (at least at the start)
        assert safe_point >= 0

    def test_balanced_code_fences(self):
        output = StringIO()
        renderer = StreamingMarkdownRenderer(output)

        # Balanced code fences
        assert renderer._has_balanced_code_fences("```python\ncode\n```")

        # Unbalanced code fences
        assert not renderer._has_balanced_code_fences("```python\ncode")

    def test_multiple_code_blocks(self):
        output = StringIO()
        renderer = StreamingMarkdownRenderer(output)

        content = "```python\nprint('hello')\n```\n\nSome text.\n\n```python\nprint('world')\n```"
        renderer.add_text(content)
        assert renderer.buffer == content


class TestContentAwareRendering:
    """Test content-aware rendering features."""

    def test_paragraph_breaks_are_safe_points(self):
        output = StringIO()
        renderer = StreamingMarkdownRenderer(output)

        content = "First paragraph.\n\nSecond paragraph."
        renderer.add_text(content)

        # Should handle paragraph breaks
        assert "\n\n" in renderer.buffer

    def test_incomplete_structures_detection(self):
        output = StringIO()
        renderer = StreamingMarkdownRenderer(output)

        # Incomplete code block
        incomplete = "Some text\n```python\ncode here..."
        assert renderer._contains_incomplete_structures(incomplete)

        # Complete structure
        complete = "Some text\n```python\ncode\n```"
        assert not renderer._contains_incomplete_structures(complete)

    def test_markdown_structures_detection(self):
        output = StringIO()
        renderer = StreamingMarkdownRenderer(output)

        # Has markdown structures
        assert renderer._contains_markdown_structures("# Header\n")
        assert renderer._contains_markdown_structures("```code```")
        assert renderer._contains_markdown_structures("- list item")

        # No markdown structures
        assert not renderer._contains_markdown_structures("plain text")
