"""Tests for configuration management."""

import os
import pytest
from pathlib import Path
import tempfile
import yaml

from llm_cli.config import Config, ConfigManager, GatewayConfig, get_config_manager


class TestConfig:
    """Test Config dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = Config()
        assert config.default_gateway == "direct"
        assert config.vercel.enabled is True
        assert config.openrouter.enabled is True
        assert config.default_model == "gemini/gemini-2.5-flash"
        assert "gpt" in config.model_aliases
        assert "fast" in config.model_aliases

    def test_gateway_config_defaults(self):
        """Test GatewayConfig defaults."""
        gateway = GatewayConfig()
        assert gateway.enabled is True
        assert gateway.api_key is None
        assert gateway.base_url is None
        assert gateway.models == []
        assert gateway.extra == {}


class TestConfigManager:
    """Test ConfigManager functionality."""

    def test_init_with_default_path(self):
        """Test initialization with default config path."""
        mgr = ConfigManager()
        assert mgr.config_path is not None
        assert "streamlm" in str(mgr.config_path)
        assert mgr.config_path.name == "config.yaml"

    def test_init_with_custom_path(self):
        """Test initialization with custom path."""
        custom_path = Path("/tmp/test_config.yaml")
        mgr = ConfigManager(custom_path)
        assert mgr.config_path == custom_path

    def test_load_nonexistent_config(self):
        """Test loading config when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            mgr = ConfigManager(config_path)
            config = mgr.load()

            # Should return default config
            assert isinstance(config, Config)
            assert config.default_gateway == "direct"

    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            mgr = ConfigManager(config_path)

            # Create and save config
            config = Config(default_gateway="vercel", default_model="gpt-4o")
            config.vercel.api_key = "test-key"
            mgr.save(config)

            # Verify file was created
            assert config_path.exists()

            # Load and verify
            mgr._config = None  # Reset cache
            loaded_config = mgr.load()
            assert loaded_config.default_gateway == "vercel"
            assert loaded_config.default_model == "gpt-4o"

    def test_interpolate_env_vars(self):
        """Test environment variable interpolation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # Create config with env var reference
            config_data = {
                "version": "1.0",
                "gateway": {
                    "default": "vercel",
                    "vercel": {
                        "enabled": True,
                        "api_key": "${TEST_API_KEY}",
                        "base_url": "https://ai-gateway.vercel.sh/v1",
                        "models": [],
                    },
                    "openrouter": {
                        "enabled": True,
                        "api_key": "${OPENROUTER_API_KEY}",
                        "base_url": "https://openrouter.ai/api/v1",
                        "models": [],
                        "app_name": "streamlm",
                        "app_url": "https://github.com/jeffmylife/streamlm",
                    },
                },
                "providers": {},
                "models": {
                    "default": "gemini/gemini-2.5-flash",
                    "aliases": {},
                },
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            # Set env var and load
            os.environ["TEST_API_KEY"] = "test-value"
            mgr = ConfigManager(config_path)
            config = mgr.load()

            assert config.vercel.api_key == "test-value"

            # Cleanup
            del os.environ["TEST_API_KEY"]

    def test_get_gateway_from_env(self, monkeypatch):
        """Test getting gateway from environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            mgr = ConfigManager(config_path)

            # Set env var
            monkeypatch.setenv("STREAMLM_GATEWAY", "vercel")

            # Should return env var value
            assert mgr.get_gateway() == "vercel"

    def test_get_gateway_from_config(self):
        """Test getting gateway from config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            mgr = ConfigManager(config_path)

            # Save config with gateway
            config = Config(default_gateway="openrouter")
            mgr.save(config)

            # Should return config value
            mgr._config = None  # Reset cache
            assert mgr.get_gateway() == "openrouter"

    def test_get_gateway_default(self):
        """Test default gateway when not configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nonexistent.yaml"
            mgr = ConfigManager(config_path)

            # Should return default
            assert mgr.get_gateway() == "direct"

    def test_resolve_model_alias(self):
        """Test resolving model aliases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            mgr = ConfigManager(config_path)

            config = mgr.load()

            # Test built-in aliases
            assert mgr.resolve_model_alias("gpt") == "gpt-4o"
            assert mgr.resolve_model_alias("fast") == "gemini/gemini-2.5-flash"

            # Test non-alias (returns as-is)
            assert mgr.resolve_model_alias("claude-3-opus") == "claude-3-opus"

    def test_set_config_values(self):
        """Test setting individual config values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            mgr = ConfigManager(config_path)

            # Set various keys
            mgr.set("gateway.default", "vercel")
            mgr.set("gateway.vercel.api_key", "test-key")
            mgr.set("models.default", "gpt-4o")

            # Verify
            config = mgr.load()
            assert config.default_gateway == "vercel"
            assert config.vercel.api_key == "test-key"
            assert config.default_model == "gpt-4o"

    def test_set_invalid_key(self):
        """Test setting invalid config key raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            mgr = ConfigManager(config_path)

            with pytest.raises(ValueError, match="Unknown config key"):
                mgr.set("invalid.key", "value")

    def test_validate(self, monkeypatch):
        """Test configuration validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            mgr = ConfigManager(config_path)

            # Clear any existing env vars
            monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
            monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
            monkeypatch.delenv("GEMINI_API_KEY", raising=False)
            monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
            monkeypatch.delenv("XAI_API_KEY", raising=False)

            # Set some env vars
            monkeypatch.setenv("OPENAI_API_KEY", "test-key")
            monkeypatch.setenv("AI_GATEWAY_API_KEY", "vercel-key")

            # Validate
            results = mgr.validate()

            # Check structure
            assert "config_file" in results
            assert "gateway" in results
            assert "providers" in results
            assert "models" in results

            # Check gateway validation
            assert results["gateway"]["default"] == "direct"
            assert results["gateway"]["vercel"]["api_key_set"] is True
            assert results["gateway"]["openrouter"]["api_key_set"] is False

            # Check provider validation
            assert results["providers"]["openai"]["env_key_set"] is True
            assert results["providers"]["openai"]["available"] is True
            assert results["providers"]["anthropic"]["available"] is False

    def test_get_full_config(self):
        """Test getting full config as dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            mgr = ConfigManager(config_path)

            config = Config(default_gateway="vercel")
            config.vercel.api_key = "test-key"
            mgr.save(config)

            # Get full config
            mgr._config = None  # Reset cache
            result = mgr.get()

            # Verify structure
            assert isinstance(result, dict)
            assert "gateway" in result
            assert "models" in result
            assert result["gateway"]["default"] == "vercel"
            # API key should be masked
            assert result["gateway"]["vercel"]["api_key"] == "***"

    def test_get_specific_key(self):
        """Test getting specific config key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            mgr = ConfigManager(config_path)

            config = Config(default_gateway="vercel")
            mgr.save(config)

            mgr._config = None  # Reset cache

            # Get specific key
            assert mgr.get("default_gateway") == "vercel"

    def test_global_config_manager(self):
        """Test global config manager instance."""
        mgr1 = get_config_manager()
        mgr2 = get_config_manager()

        # Should return same instance
        assert mgr1 is mgr2
