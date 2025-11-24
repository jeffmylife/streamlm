# StreamLM

[![Downloads](https://static.pepy.tech/badge/streamlm)](https://pepy.tech/project/streamlm)
[![PyPI version](https://badge.fury.io/py/streamlm.svg)](https://badge.fury.io/py/streamlm)
[![GitHub Release](https://img.shields.io/github/v/release/jeffmylife/streamlm)](https://github.com/jeffmylife/streamlm/releases)
[![Build Status](https://github.com/jeffmylife/streamlm/workflows/Test/badge.svg)](https://github.com/jeffmylife/streamlm/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A command-line interface for interacting with various Large Language Models with beautiful markdown-formatted responses.

## Installation

### uv (recommended)

```bash
uv tool install streamlm
```

### PyPI
```bash
pip install streamlm
```

### Homebrew (macOS/Linux)
```bash
brew install jeffmylife/streamlm/streamlm
```

## Usage

After installation, you can use the `lm` command:

```bash
lm explain quantum computing
lm -m gpt-4o "write a Python function"
lm -m claude-3-5-sonnet "analyze this data"
```

### Raw Markdown Output

StreamLM includes beautiful built-in markdown formatting, but you can also output raw markdown for piping to other tools:

```bash
# Output raw markdown without Rich formatting
lm --md "explain machine learning" > output.md

# Pipe to your favorite markdown formatter (like glow)
lm --md "write a Python tutorial" | glow

# Use with other markdown tools
lm --raw "create documentation" | pandoc -f markdown -t html
```

### Supported Models

StreamLM provides access to various Large Language Models including:

- **OpenAI**: GPT-4o, o1, o3-mini, GPT-4o-mini
- **Anthropic**: Claude-3-7-sonnet, Claude-3-5-sonnet, Claude-3-5-haiku
- **Google**: Gemini-2.5-flash, Gemini-2.5-pro, Gemini-2.0-flash-thinking
- **DeepSeek**: DeepSeek-R1, DeepSeek-V3
- **xAI**: Grok-4, Grok-3-beta, Grok-3-mini-beta
- **Local models**: Via Ollama (Llama3.3, Qwen2.5, DeepSeek-Coder, etc.)

### Options

- `--model` / `-m`: Choose the LLM model
- `--image` / `-i`: Include image files for vision models
- `--context` / `-c`: Add context from a file
- `--max-tokens` / `-t`: Set maximum response length
- `--temperature` / `-temp`: Control response creativity (0.0-1.0)
- `--think`: Show reasoning process (for reasoning models)
- `--debug` / `-d`: Enable debug mode
- `--raw` / `--md`: Output raw markdown without Rich formatting

## Features

- 🎨 Beautiful markdown-formatted responses
- 🖼️ Image input support for compatible models
- 📁 Context file support
- 🧠 Reasoning model support (DeepSeek, OpenAI o1, etc.)
- 🔧 Extensive model support across providers
- ⚡ Fast and lightweight
- 🛠️ Easy configuration

## Links

- [PyPI Package](https://pypi.org/project/streamlm/)
- [Homebrew Tap](https://github.com/jeffmylife/homebrew-streamlm)
- [Issues](https://github.com/jeffmylife/streamlm/issues)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/jeffmylife/streamlm.git
cd streamlm

# Install with dev dependencies
uv pip install -e ".[dev]"
```

### Running Tests

All tests use `uv run` for consistency:

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test types
uv run pytest tests/test_cli.py -v                    # Unit tests only
uv run pytest tests/test_integration.py -v            # Integration tests only
```

### Release Process

```bash
# Make your changes
uv version --bump patch
git add .
git commit -m "feat: your changes"
git push

# Create GitHub release (this triggers everything automatically)
gh release create v0.1.11 --generate-notes
```
