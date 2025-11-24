# StreamLM Test Suite

This directory contains the test suite for streamlm, organized using pytest.

## Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Shared fixtures and test configuration
├── test_cli.py              # CLI functionality tests
├── test_streaming_markdown.py  # Streaming markdown renderer tests
└── test_integration.py      # End-to-end integration tests
```

## Running Tests

All tests use `uv run` for consistency with the UV workflow:

### Run all tests
```bash
uv run pytest tests/ -v
```

### Run with coverage
```bash
uv run pytest tests/ -v --cov=src --cov-report=term-missing
```

### Run unit tests only
```bash
uv run pytest tests/test_cli.py tests/test_streaming_markdown.py -v
```

### Run integration tests only
```bash
uv run pytest tests/test_integration.py -v
```

### Run specific test file
```bash
uv run pytest tests/test_cli.py -v
```

### Run specific test class
```bash
uv run pytest tests/test_cli.py::TestModelProviderDetection -v
```

### Run specific test
```bash
uv run pytest tests/test_cli.py::TestModelProviderDetection::test_openai_detection -v
```

## Test Categories

### Unit Tests
- `test_cli.py`: Tests for CLI argument parsing, model detection, and provider logic
- `test_streaming_markdown.py`: Tests for markdown rendering, safe point detection, and code block handling

### Integration Tests
- `test_integration.py`: End-to-end tests that verify the complete CLI workflow

## Writing New Tests

1. Create a new test file following the pattern `test_*.py`
2. Import pytest and the modules you want to test
3. Organize tests into classes using `Test*` naming convention
4. Use fixtures from `conftest.py` for common setup
5. Follow the existing patterns for assertions and test structure

Example:
```python
class TestMyFeature:
    def test_feature_works(self):
        assert my_function() == expected_result
```

## Coverage

Current coverage focuses on:
- Model provider detection
- Reasoning model identification
- CLI version command
- Streaming markdown renderer logic
- Code block detection and safe point finding
- Loading indicators

Areas with lower coverage (intentionally):
- API integration (requires API keys and network access)
- Full streaming response handling (requires live API calls)
- Terminal rendering (depends on terminal capabilities)
