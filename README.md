# StreamLM

[![Downloads](https://static.pepy.tech/badge/streamlm)](https://pepy.tech/project/streamlm)
[![PyPI version](https://badge.fury.io/py/streamlm.svg)](https://badge.fury.io/py/streamlm)
[![GitHub Release](https://img.shields.io/github/v/release/jeffmylife/streamlm)](https://github.com/jeffmylife/streamlm/releases)
[![Build Status](https://github.com/jeffmylife/streamlm/workflows/Test/badge.svg)](https://github.com/jeffmylife/streamlm/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A command-line interface for interacting with various Large Language Models with beautiful markdown-formatted responses.

**Design Principle**: Frictionless interaction. Just type `lm hello` - no need for subcommands. The CLI defaults to chat mode for the fastest possible workflow.

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

### Basic Usage

After installation, you can use the `lm` command. **The CLI defaults to chat mode** - just type your prompt:

```bash
lm explain quantum computing
lm -m gpt-4o "write a Python function"
lm -m claude-3-5-sonnet "analyze this data"

# Explicit 'chat' command also works
lm chat "hello world"
```

### Gateway Routing

StreamLM supports routing requests through different gateways for cost optimization and flexibility:

```bash
# Route through Vercel AI Gateway (no markup, low latency)
lm --gateway vercel "explain quantum computing"

# Route through OpenRouter (model discovery, transparent pricing)
lm --gateway openrouter -m gpt-4o "write a function"

# Direct provider access (default, supports reasoning models)
lm --gateway direct "analyze this data"
```

**Gateway Benefits:**
- **Vercel**: $5/month free credits, no token markup, <20ms latency
- **OpenRouter**: Model discovery, pricing transparency, bring-your-own-key
- **Direct**: Full provider feature support, reasoning models, lowest latency

### Configuration

StreamLM can be configured via config file (`~/.streamlm/config.yaml`), environment variables, or CLI flags:

```bash
# Interactive setup wizard
lm config setup

# Set default gateway
lm config set gateway.default vercel

# Configure gateway API keys
lm config set gateway.vercel.api_key sk-your-ai-gateway-key

# Set default model
lm config set models.default gpt-4o

# View current configuration
lm config get

# Validate configuration and API keys
lm config validate

# List available gateways
lm config list-gateways
```

**Configuration Priority** (highest to lowest):
1. CLI flags (`--gateway`, `--model`)
2. Environment variables (`STREAMLM_GATEWAY`, provider API keys)
3. Config file (`~/.streamlm/config.yaml`)
4. Defaults (direct gateway, gemini-2.5-flash model)

### Model Aliases

Define shortcuts for your favorite models in the config:

```yaml
# ~/.streamlm/config.yaml
models:
  aliases:
    gpt: "gpt-4o"
    claude: "claude-3-5-sonnet"
    fast: "gemini/gemini-2.5-flash"
    smart: "gpt-4o"
```

Then use them:

```bash
lm -m fast "quick question"    # Uses gemini-2.5-flash
lm -m smart "complex analysis"  # Uses gpt-4o
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

### Chat Command Options

- `--model` / `-m`: Choose the LLM model (or use alias from config)
- `--gateway` / `-g`: Route through gateway (direct, vercel, openrouter)
- `--image` / `-i`: Include image files for vision models
- `--context` / `-c`: Add context from a file
- `--max-tokens` / `-t`: Set maximum response length
- `--temperature` / `-temp`: Control response creativity (0.0-1.0)
- `--think`: Show reasoning process (reasoning models, direct gateway only)
- `--session` / `-s`: Session ID to continue conversation
- `--session-name`: Name for a new session (only when creating)
- `--debug` / `-d`: Enable debug mode
- `--raw` / `--md`: Output raw markdown without Rich formatting

### Session Management

StreamLM supports conversation sessions to maintain context across multiple messages:

```bash
# Create a new session or continue an existing one
lm --session my-project "How do I implement authentication?"
lm --session my-project "Can you show me an example?"  # Continues with context

# Name your session when creating it
lm --session dev-2025 --session-name "Development Session" "Let's start coding"

# List all sessions
lm sessions --list

# Show session details and conversation history
lm sessions --show my-project

# Export session to JSON
lm sessions --export my-project > session.json

# Clear messages from a session (keeps session metadata)
lm sessions --clear my-project

# Delete a session completely
lm sessions --delete my-project
```

**Session Features:**
- Automatic conversation history - context is maintained across messages
- Token usage tracking per session
- Local-first storage using libSQL (SQLite compatible)
- Optional remote sync with Turso (not required)
- Export/import sessions for backup or sharing
- Metadata support for images and context files

### Config Command Actions

- `lm config setup`: Interactive configuration wizard
- `lm config get [key]`: Get configuration value
- `lm config set <key> <value>`: Set configuration value
- `lm config validate`: Validate configuration and API keys
- `lm config list-gateways`: Show available gateways and their status

## Features

- 🎨 Beautiful markdown-formatted responses
- 💬 **Conversation sessions** with persistent history
- 🌐 **Gateway routing** (Vercel AI Gateway, OpenRouter, or direct)
- ⚙️ **Flexible configuration** (config file, env vars, CLI flags)
- 🔑 **Model aliases** for quick access to favorite models
- 🖼️ Image input support for compatible models
- 📁 Context file support
- 🧠 Reasoning model support (DeepSeek, OpenAI o1, etc.)
- 📊 Token usage tracking per session
- 💾 Local-first database storage (no cloud required)
- 🔧 Extensive model support across providers
- ⚡ Fast and lightweight
- 🛠️ Easy configuration management

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
