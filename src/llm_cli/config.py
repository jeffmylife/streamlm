"""Configuration management for streamlm."""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class GatewayConfig:
    """Configuration for a specific gateway."""

    enabled: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    models: list[str] = field(default_factory=list)
    # Additional gateway-specific settings
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration for streamlm."""

    # Gateway settings
    default_gateway: str = "direct"  # direct, vercel, openrouter

    # Gateway-specific configs
    vercel: GatewayConfig = field(
        default_factory=lambda: GatewayConfig(
            enabled=True,
            api_key=None,
            base_url="https://ai-gateway.vercel.sh/v1",
        )
    )

    openrouter: GatewayConfig = field(
        default_factory=lambda: GatewayConfig(
            enabled=True,
            api_key=None,
            base_url="https://openrouter.ai/api/v1",
            extra={
                "app_name": "streamlm",
                "app_url": "https://github.com/jeffmylife/streamlm",
            },
        )
    )

    # Provider API keys (for direct access)
    provider_keys: Dict[str, Optional[str]] = field(
        default_factory=lambda: {
            "openai": None,
            "anthropic": None,
            "gemini": None,
            "deepseek": None,
            "xai": None,
        }
    )

    # Model preferences
    default_model: str = "gemini/gemini-2.5-flash"
    model_aliases: Dict[str, str] = field(
        default_factory=lambda: {
            "gpt": "gpt-4o",
            "claude": "claude-3-5-sonnet",
            "fast": "gemini/gemini-2.5-flash",
            "smart": "gpt-4o",
        }
    )


class ConfigManager:
    """Manages loading, saving, and accessing configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize config manager.

        Args:
            config_path: Path to config file. If None, uses default location.
        """
        if config_path is None:
            # Try XDG_CONFIG_HOME first, fallback to ~/.streamlm
            xdg_config = os.getenv("XDG_CONFIG_HOME")
            if xdg_config:
                self.config_path = Path(xdg_config) / "streamlm" / "config.yaml"
            else:
                self.config_path = Path.home() / ".streamlm" / "config.yaml"
        else:
            self.config_path = config_path

        self._config: Optional[Config] = None

    def load(self) -> Config:
        """Load configuration from file, or return default if not exists."""
        if self._config is not None:
            return self._config

        if not self.config_path.exists():
            self._config = Config()
            return self._config

        try:
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f) or {}

            # Parse gateway configs
            gateway_data = data.get("gateway", {})
            vercel_data = gateway_data.get("vercel", {})
            openrouter_data = gateway_data.get("openrouter", {})

            # Interpolate environment variables
            vercel_api_key = self._interpolate_env(vercel_data.get("api_key"))
            openrouter_api_key = self._interpolate_env(
                openrouter_data.get("api_key")
            )

            vercel_config = GatewayConfig(
                enabled=vercel_data.get("enabled", True),
                api_key=vercel_api_key,
                base_url=vercel_data.get(
                    "base_url", "https://ai-gateway.vercel.sh/v1"
                ),
                models=vercel_data.get("models", []),
            )

            openrouter_config = GatewayConfig(
                enabled=openrouter_data.get("enabled", True),
                api_key=openrouter_api_key,
                base_url=openrouter_data.get(
                    "base_url", "https://openrouter.ai/api/v1"
                ),
                models=openrouter_data.get("models", []),
                extra={
                    "app_name": openrouter_data.get("app_name", "streamlm"),
                    "app_url": openrouter_data.get(
                        "app_url", "https://github.com/jeffmylife/streamlm"
                    ),
                },
            )

            # Parse provider keys
            providers_data = data.get("providers", {})
            provider_keys = {}
            for provider in ["openai", "anthropic", "gemini", "deepseek", "xai"]:
                key_data = providers_data.get(provider, {})
                if isinstance(key_data, dict):
                    key = self._interpolate_env(key_data.get("api_key"))
                else:
                    key = self._interpolate_env(key_data)
                provider_keys[provider] = key

            # Parse model settings
            models_data = data.get("models", {})
            default_model = models_data.get("default", "gemini/gemini-2.5-flash")
            model_aliases = models_data.get(
                "aliases",
                {
                    "gpt": "gpt-4o",
                    "claude": "claude-3-5-sonnet",
                    "fast": "gemini/gemini-2.5-flash",
                    "smart": "gpt-4o",
                },
            )

            self._config = Config(
                default_gateway=gateway_data.get("default", "direct"),
                vercel=vercel_config,
                openrouter=openrouter_config,
                provider_keys=provider_keys,
                default_model=default_model,
                model_aliases=model_aliases,
            )

            return self._config

        except Exception as e:
            # If there's an error loading, return default config
            print(f"Warning: Error loading config: {e}. Using defaults.")
            self._config = Config()
            return self._config

    def save(self, config: Optional[Config] = None) -> None:
        """Save configuration to file.

        Args:
            config: Config to save. If None, saves current config.
        """
        if config is None:
            config = self._config or Config()

        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Build config dict
        data = {
            "version": "1.0",
            "gateway": {
                "default": config.default_gateway,
                "vercel": {
                    "enabled": config.vercel.enabled,
                    "api_key": config.vercel.api_key or "${AI_GATEWAY_API_KEY}",
                    "base_url": config.vercel.base_url,
                    "models": config.vercel.models,
                },
                "openrouter": {
                    "enabled": config.openrouter.enabled,
                    "api_key": config.openrouter.api_key
                    or "${OPENROUTER_API_KEY}",
                    "base_url": config.openrouter.base_url,
                    "models": config.openrouter.models,
                    "app_name": config.openrouter.extra.get("app_name", "streamlm"),
                    "app_url": config.openrouter.extra.get(
                        "app_url", "https://github.com/jeffmylife/streamlm"
                    ),
                },
            },
            "providers": {},
            "models": {
                "default": config.default_model,
                "aliases": config.model_aliases,
            },
        }

        # Add provider keys (only if set)
        for provider, key in config.provider_keys.items():
            if key:
                data["providers"][provider] = {"api_key": key}

        # Write to file
        with open(self.config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get(self, key: Optional[str] = None) -> Any:
        """Get a config value by dot-notation key.

        Args:
            key: Dot-notation key (e.g., 'gateway.default'). If None, returns full config.

        Returns:
            Config value or full config dict.
        """
        config = self.load()

        if key is None:
            # Return full config as dict
            return {
                "gateway": {
                    "default": config.default_gateway,
                    "vercel": {
                        "enabled": config.vercel.enabled,
                        "api_key": "***" if config.vercel.api_key else None,
                        "base_url": config.vercel.base_url,
                        "models": config.vercel.models,
                    },
                    "openrouter": {
                        "enabled": config.openrouter.enabled,
                        "api_key": "***" if config.openrouter.api_key else None,
                        "base_url": config.openrouter.base_url,
                        "models": config.openrouter.models,
                    },
                },
                "models": {
                    "default": config.default_model,
                    "aliases": config.model_aliases,
                },
            }

        # Parse dot notation
        parts = key.split(".")
        value: Any = config

        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            elif isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None

        return value

    def set(self, key: str, value: Any) -> None:
        """Set a config value by dot-notation key.

        Args:
            key: Dot-notation key (e.g., 'gateway.default')
            value: Value to set
        """
        config = self.load()

        # Handle specific keys
        if key == "gateway" or key == "gateway.default":
            config.default_gateway = value
        elif key == "gateway.vercel.api_key":
            config.vercel.api_key = value
        elif key == "gateway.openrouter.api_key":
            config.openrouter.api_key = value
        elif key == "gateway.vercel.enabled":
            config.vercel.enabled = value
        elif key == "gateway.openrouter.enabled":
            config.openrouter.enabled = value
        elif key == "models.default":
            config.default_model = value
        elif key.startswith("models.aliases."):
            alias = key.split(".")[-1]
            config.model_aliases[alias] = value
        elif key.startswith("providers."):
            provider = key.split(".")[1]
            if provider in config.provider_keys:
                config.provider_keys[provider] = value
        else:
            raise ValueError(f"Unknown config key: {key}")

        self._config = config
        self.save(config)

    def get_gateway(self) -> str:
        """Get the effective gateway to use.

        Priority:
        1. Environment variable STREAMLM_GATEWAY
        2. Config file gateway.default
        3. Fallback to 'direct'

        Returns:
            Gateway name (direct, vercel, openrouter)
        """
        # Check environment variable first
        env_gateway = os.getenv("STREAMLM_GATEWAY")
        if env_gateway and env_gateway in ["direct", "vercel", "openrouter"]:
            return env_gateway

        # Check config file
        config = self.load()
        return config.default_gateway

    def resolve_model_alias(self, model: str) -> str:
        """Resolve a model alias to its actual model name.

        Args:
            model: Model name or alias

        Returns:
            Resolved model name
        """
        config = self.load()
        return config.model_aliases.get(model, model)

    def _interpolate_env(self, value: Optional[str]) -> Optional[str]:
        """Interpolate environment variables in config values.

        Supports ${VAR_NAME} syntax.

        Args:
            value: String value that may contain env var references

        Returns:
            Interpolated string or None
        """
        if not value or not isinstance(value, str):
            return value

        # Handle ${ENV_VAR} syntax
        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var)

        return value

    def validate(self) -> Dict[str, Any]:
        """Validate configuration and check API key availability.

        Returns:
            Dict with validation results
        """
        config = self.load()
        results = {
            "config_file": str(self.config_path),
            "exists": self.config_path.exists(),
            "gateway": {
                "default": config.default_gateway,
                "vercel": {
                    "enabled": config.vercel.enabled,
                    "api_key_set": bool(config.vercel.api_key),
                },
                "openrouter": {
                    "enabled": config.openrouter.enabled,
                    "api_key_set": bool(config.openrouter.api_key),
                },
            },
            "providers": {},
            "models": {
                "default": config.default_model,
                "aliases": config.model_aliases,
            },
        }

        # Check provider API keys (both config and env)
        for provider in ["openai", "anthropic", "gemini", "deepseek", "xai"]:
            config_key = config.provider_keys.get(provider)
            env_key = os.getenv(f"{provider.upper()}_API_KEY")
            results["providers"][provider] = {
                "config_key_set": bool(config_key),
                "env_key_set": bool(env_key),
                "available": bool(config_key or env_key),
            }

        # Check gateway-specific env vars
        vercel_env = os.getenv("AI_GATEWAY_API_KEY")
        openrouter_env = os.getenv("OPENROUTER_API_KEY")

        if vercel_env:
            results["gateway"]["vercel"]["api_key_set"] = True
        if openrouter_env:
            results["gateway"]["openrouter"]["api_key_set"] = True

        return results


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
