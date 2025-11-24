"""Tests for gateway routing functionality."""

import os
import pytest
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from llm_cli.gateways import (
    GatewayRouter,
    GatewayRoute,
    get_model_provider,
    is_reasoning_model,
)
from llm_cli.config import ConfigManager, Config, GatewayConfig


class TestModelProviderDetection:
    """Test model provider detection."""

    def test_openai_detection(self):
        assert get_model_provider("gpt-4o") == "openai"
        assert get_model_provider("gpt-4o-mini") == "openai"
        assert get_model_provider("openai/gpt-4") == "openai"
        assert get_model_provider("o1-preview") == "openai"

    def test_anthropic_detection(self):
        assert get_model_provider("claude-3-opus") == "anthropic"
        assert get_model_provider("anthropic/claude-3-sonnet") == "anthropic"
        assert get_model_provider("claude-3-5-sonnet") == "anthropic"

    def test_gemini_detection(self):
        assert get_model_provider("gemini-pro") == "gemini"
        assert get_model_provider("gemini/gemini-2.5-flash") == "gemini"

    def test_deepseek_detection(self):
        assert get_model_provider("deepseek-chat") == "deepseek"
        assert get_model_provider("deepseek-r1") == "deepseek"

    def test_xai_detection(self):
        assert get_model_provider("grok-3") == "xai"
        assert get_model_provider("xai/grok-4") == "xai"

    def test_ollama_detection(self):
        assert get_model_provider("ollama/llama3.3") == "ollama"

    def test_openrouter_detection(self):
        assert get_model_provider("openrouter/anthropic/claude-3-opus") == "openrouter"

    def test_unknown_provider(self):
        assert get_model_provider("unknown-model") == "unknown"


class TestReasoningModelDetection:
    """Test reasoning model detection."""

    def test_deepseek_reasoning_models(self):
        assert is_reasoning_model("deepseek-r1") is True
        assert is_reasoning_model("deepseek-reasoner") is True
        assert is_reasoning_model("deepseek-chat") is False

    def test_openai_reasoning_models(self):
        assert is_reasoning_model("o1-preview") is True
        assert is_reasoning_model("o1-mini") is True
        assert is_reasoning_model("o1-pro") is True
        assert is_reasoning_model("gpt-4o") is False

    def test_grok_reasoning_models(self):
        assert is_reasoning_model("grok-3") is True
        assert is_reasoning_model("grok-4") is True
        assert is_reasoning_model("grok-2") is False

    def test_non_reasoning_models(self):
        assert is_reasoning_model("claude-3-opus") is False
        assert is_reasoning_model("gemini-pro") is False


class TestGatewayRouter:
    """Test GatewayRouter functionality."""

    @pytest.fixture
    def temp_config_manager(self):
        """Create a temporary config manager for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            mgr = ConfigManager(config_path)
            yield mgr

    def test_router_initialization(self, temp_config_manager):
        """Test router initialization."""
        router = GatewayRouter(temp_config_manager)
        assert router.config_manager is temp_config_manager

    def test_route_direct(self, temp_config_manager):
        """Test routing with direct gateway (default)."""
        router = GatewayRouter(temp_config_manager)
        route = router.route_request("gpt-4o", provider="openai")

        assert route.gateway == "direct"
        assert route.base_url is None
        assert route.api_key is None
        assert route.model_name == "gpt-4o"

    def test_route_vercel_with_override(self, temp_config_manager, monkeypatch):
        """Test routing to Vercel gateway with CLI override."""
        monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-vercel-key")

        router = GatewayRouter(temp_config_manager)
        route = router.route_request(
            "gpt-4o", gateway_override="vercel", provider="openai"
        )

        assert route.gateway == "vercel"
        assert route.base_url == "https://ai-gateway.vercel.sh/v1"
        assert route.api_key == "test-vercel-key"
        assert "openai/gpt-4o" in route.model_name  # Should be normalized

    def test_route_openrouter_with_override(self, temp_config_manager, monkeypatch):
        """Test routing to OpenRouter with CLI override."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

        router = GatewayRouter(temp_config_manager)
        route = router.route_request(
            "claude-3-opus", gateway_override="openrouter", provider="anthropic"
        )

        assert route.gateway == "openrouter"
        assert route.base_url == "https://openrouter.ai/api/v1"
        assert route.api_key == "test-openrouter-key"
        assert "anthropic/claude-3-opus" in route.model_name

    def test_route_vercel_missing_key(self, temp_config_manager, monkeypatch):
        """Test routing to Vercel without API key raises error."""
        # Ensure env var is not set
        monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)

        router = GatewayRouter(temp_config_manager)

        with pytest.raises(ValueError, match="AI_GATEWAY_API_KEY"):
            router.route_request("gpt-4o", gateway_override="vercel", provider="openai")

    def test_route_openrouter_missing_key(self, temp_config_manager, monkeypatch):
        """Test routing to OpenRouter without API key raises error."""
        # Ensure env var is not set
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        router = GatewayRouter(temp_config_manager)

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            router.route_request(
                "gpt-4o", gateway_override="openrouter", provider="openai"
            )

    def test_normalize_model_with_provider_prefix(self, temp_config_manager):
        """Test model normalization when model already has provider prefix."""
        router = GatewayRouter(temp_config_manager)

        # Model already has prefix
        normalized = router._normalize_model_for_gateway(
            "openai/gpt-4o", provider="openai"
        )
        assert normalized == "openai/gpt-4o"

    def test_normalize_model_add_prefix(self, temp_config_manager):
        """Test adding provider prefix to model."""
        router = GatewayRouter(temp_config_manager)

        # Add openai prefix
        normalized = router._normalize_model_for_gateway("gpt-4o", provider="openai")
        assert normalized == "openai/gpt-4o"

        # Add anthropic prefix
        normalized = router._normalize_model_for_gateway(
            "claude-3-opus", provider="anthropic"
        )
        assert normalized == "anthropic/claude-3-opus"

        # Map gemini to google
        normalized = router._normalize_model_for_gateway(
            "gemini-pro", provider="gemini"
        )
        assert normalized == "google/gemini-pro"

        # Map xai to x-ai
        normalized = router._normalize_model_for_gateway("grok-3", provider="xai")
        assert normalized == "x-ai/grok-3"

    def test_supports_reasoning_direct_gateway(self, temp_config_manager):
        """Test reasoning support with direct gateway."""
        router = GatewayRouter(temp_config_manager)

        # Reasoning models via direct gateway
        assert router.supports_reasoning("deepseek-r1", "direct") is True
        assert router.supports_reasoning("o1-preview", "direct") is True
        assert router.supports_reasoning("grok-3", "direct") is True

        # Non-reasoning models
        assert router.supports_reasoning("gpt-4o", "direct") is False

    def test_supports_reasoning_gateway(self, temp_config_manager):
        """Test reasoning not supported via gateways."""
        router = GatewayRouter(temp_config_manager)

        # Reasoning not supported via gateways
        assert router.supports_reasoning("deepseek-r1", "vercel") is False
        assert router.supports_reasoning("o1-preview", "openrouter") is False

    def test_create_client_for_direct_route(self, temp_config_manager):
        """Test creating client for direct route returns None."""
        router = GatewayRouter(temp_config_manager)
        route = GatewayRoute(
            gateway="direct",
            base_url=None,
            api_key=None,
            headers={},
            model_name="gpt-4o",
        )

        client = router.create_client_for_route(route)
        assert client is None

    def test_create_client_for_gateway_route(self, temp_config_manager, monkeypatch):
        """Test creating client for gateway route."""
        monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")

        router = GatewayRouter(temp_config_manager)
        route = router.route_request(
            "gpt-4o", gateway_override="vercel", provider="openai"
        )

        client = router.create_client_for_route(route)
        assert client is not None
        assert client.api_key == "test-key"

    def test_configure_litellm_for_vercel(self, temp_config_manager, monkeypatch):
        """Test configuring litellm for Vercel gateway."""
        monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")

        router = GatewayRouter(temp_config_manager)
        route = router.route_request(
            "gpt-4o", gateway_override="vercel", provider="openai"
        )

        router.configure_litellm_for_route(route)

        # Check env vars are set
        assert os.environ.get("OPENAI_API_BASE") == "https://ai-gateway.vercel.sh/v1"
        assert os.environ.get("OPENAI_API_KEY") == "test-key"

    def test_configure_litellm_for_openrouter(self, temp_config_manager, monkeypatch):
        """Test configuring litellm for OpenRouter gateway."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        router = GatewayRouter(temp_config_manager)
        route = router.route_request(
            "gpt-4o", gateway_override="openrouter", provider="openai"
        )

        router.configure_litellm_for_route(route)

        # Check env vars are set
        assert os.environ.get("OPENAI_API_BASE") == "https://openrouter.ai/api/v1"
        assert os.environ.get("OPENAI_API_KEY") == "test-key"

    def test_route_from_config_default(self, temp_config_manager):
        """Test routing uses config default gateway."""
        # Set config to use vercel by default
        config = Config(default_gateway="vercel")
        config.vercel.api_key = "test-key"
        temp_config_manager.save(config)
        temp_config_manager._config = None  # Reset cache

        router = GatewayRouter(temp_config_manager)
        route = router.route_request("gpt-4o", provider="openai")

        # Should use vercel from config
        assert route.gateway == "vercel"

    def test_route_from_env_var(self, temp_config_manager, monkeypatch):
        """Test routing uses STREAMLM_GATEWAY env var."""
        # Set env var
        monkeypatch.setenv("STREAMLM_GATEWAY", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # Config has different default
        config = Config(default_gateway="direct")
        temp_config_manager.save(config)
        temp_config_manager._config = None

        router = GatewayRouter(temp_config_manager)
        route = router.route_request("gpt-4o", provider="openai")

        # Should use openrouter from env var (higher priority)
        assert route.gateway == "openrouter"

    def test_route_priority_cli_over_env(self, temp_config_manager, monkeypatch):
        """Test CLI override has highest priority."""
        monkeypatch.setenv("STREAMLM_GATEWAY", "openrouter")
        monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")

        router = GatewayRouter(temp_config_manager)
        route = router.route_request(
            "gpt-4o", gateway_override="vercel", provider="openai"
        )

        # Should use vercel from CLI (highest priority)
        assert route.gateway == "vercel"
