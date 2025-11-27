"""Gateway routing logic for streamlm."""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import litellm
from openai import OpenAI

from .config import ConfigManager, GatewayConfig


@dataclass
class GatewayRoute:
    """Represents a routing decision for a model request."""

    gateway: str  # direct, vercel, openrouter
    base_url: Optional[str]
    api_key: Optional[str]
    headers: Dict[str, str]
    model_name: str  # May be modified for gateway routing


class GatewayRouter:
    """Routes model requests through appropriate gateways."""

    def __init__(self, config_manager: ConfigManager):
        """Initialize gateway router.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager

    def route_request(
        self,
        model: str,
        gateway_override: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> GatewayRoute:
        """Determine routing for a model request.

        Args:
            model: Model name (may include provider prefix)
            gateway_override: CLI-specified gateway override
            provider: Detected provider (openai, anthropic, etc.)

        Returns:
            GatewayRoute with routing information
        """
        config = self.config_manager.load()

        # Determine which gateway to use
        # Priority: CLI override > env var > config > default
        if gateway_override:
            gateway = gateway_override
        else:
            gateway = self.config_manager.get_gateway()

        # Route based on gateway choice
        if gateway == "vercel":
            return self._route_vercel(model, config.vercel, provider)
        elif gateway == "openrouter":
            return self._route_openrouter(model, config.openrouter, provider)
        else:
            # Direct routing (current behavior)
            return self._route_direct(model, provider)

    def _route_vercel(
        self, model: str, gateway_config: GatewayConfig, provider: Optional[str]
    ) -> GatewayRoute:
        """Route request through Vercel AI Gateway.

        Args:
            model: Model name
            gateway_config: Vercel gateway configuration
            provider: Provider name

        Returns:
            GatewayRoute configured for Vercel
        """
        # Get Vercel AI Gateway API key (config or env)
        api_key = gateway_config.api_key or os.getenv("AI_GATEWAY_API_KEY")

        if not api_key:
            raise ValueError(
                "AI_GATEWAY_API_KEY not set. Set it via config or environment variable."
            )

        # Vercel uses OpenAI-compatible format
        # Model names should be in format: provider/model-name
        model_name = self._normalize_model_for_gateway(model, provider)

        return GatewayRoute(
            gateway="vercel",
            base_url=gateway_config.base_url,
            api_key=api_key,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            model_name=model_name,
        )

    def _route_openrouter(
        self, model: str, gateway_config: GatewayConfig, provider: Optional[str]
    ) -> GatewayRoute:
        """Route request through OpenRouter.

        Args:
            model: Model name
            gateway_config: OpenRouter gateway configuration
            provider: Provider name

        Returns:
            GatewayRoute configured for OpenRouter
        """
        # Get OpenRouter API key (config or env)
        api_key = gateway_config.api_key or os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Set it via config or environment variable."
            )

        # OpenRouter expects models in format: provider/model-name
        model_name = self._normalize_model_for_gateway(model, provider)

        # OpenRouter-specific headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": gateway_config.extra.get(
                "app_url", "https://github.com/jeffmylife/streamlm"
            ),
            "X-Title": gateway_config.extra.get("app_name", "streamlm"),
        }

        return GatewayRoute(
            gateway="openrouter",
            base_url=gateway_config.base_url,
            api_key=api_key,
            headers=headers,
            model_name=model_name,
        )

    def _route_direct(
        self, model: str, provider: Optional[str]
    ) -> GatewayRoute:
        """Route request directly to provider (current behavior).

        Args:
            model: Model name
            provider: Provider name

        Returns:
            GatewayRoute configured for direct access
        """
        # For direct access, API keys are handled per-provider in cli.py
        return GatewayRoute(
            gateway="direct",
            base_url=None,  # Uses provider's default base URL
            api_key=None,  # Handled by litellm/provider SDK
            headers={},
            model_name=model,  # Keep original model name
        )

    def _normalize_model_for_gateway(
        self, model: str, provider: Optional[str]
    ) -> str:
        """Normalize model name for gateway routing.

        Args:
            model: Original model name
            provider: Detected provider

        Returns:
            Normalized model name (e.g., openai/gpt-4o)
        """
        # Model ID mappings for friendly names -> actual API IDs
        model_id_map = {
            "claude-opus-4.5": "claude-opus-4-20250514",
            "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
        }

        # If model already has provider prefix, extract and map the model part
        if "/" in model:
            parts = model.split("/", 1)
            if len(parts) == 2:
                prefix, model_name = parts
                # Map the model name if needed
                mapped_model = model_id_map.get(model_name, model_name)
                return f"{prefix}/{mapped_model}"
            return model

        # Map model name if needed
        model_clean = model_id_map.get(model, model)

        # Add provider prefix if known
        if provider and provider != "unknown":
            # Map internal provider names to gateway names
            provider_map = {
                "openai": "openai",
                "anthropic": "anthropic",
                "gemini": "google",
                "deepseek": "deepseek",
                "xai": "x-ai",
                "ollama": "ollama",
            }

            gateway_provider = provider_map.get(provider, provider)

            # For certain models, clean up the name
            model_clean = model_clean.replace("gemini/", "").replace("ollama/", "")

            return f"{gateway_provider}/{model_clean}"

        # If no provider detected, return as-is
        return model_clean

    def supports_reasoning(self, model: str, gateway: str) -> bool:
        """Check if reasoning features are supported for this model/gateway combo.

        Args:
            model: Model name
            gateway: Gateway being used

        Returns:
            True if reasoning is supported
        """
        # Reasoning features like DeepSeek's reasoning_content are only
        # supported via direct API access currently
        if gateway != "direct":
            return False

        # Check if model supports reasoning (from cli.py logic)
        model_lower = model.lower()

        # DeepSeek reasoning models
        if any(name in model_lower for name in ["deepseek-reasoner", "deepseek-r1"]):
            return True

        # OpenAI o1 models
        if any(name in model_lower for name in ["o1-preview", "o1-mini", "o1-pro"]):
            return True

        # xAI Grok models with reasoning
        if any(name in model_lower for name in ["grok-3", "grok-4"]):
            return True

        return False

    def create_client_for_route(self, route: GatewayRoute) -> Optional[OpenAI]:
        """Create an OpenAI-compatible client for a gateway route.

        Args:
            route: Gateway route information

        Returns:
            OpenAI client configured for the gateway, or None for direct access
        """
        if route.gateway == "direct":
            # Let litellm handle it
            return None

        # Create OpenAI client configured for gateway
        return OpenAI(
            api_key=route.api_key,
            base_url=route.base_url,
            default_headers=route.headers,
        )

    def configure_litellm_for_route(self, route: GatewayRoute) -> None:
        """Configure litellm for a gateway route.

        Args:
            route: Gateway route information
        """
        if route.gateway == "vercel":
            # Configure litellm to use Vercel gateway
            os.environ["OPENAI_API_BASE"] = route.base_url or ""
            os.environ["OPENAI_API_KEY"] = route.api_key or ""
        elif route.gateway == "openrouter":
            # Configure litellm to use OpenRouter
            # Set as custom OpenAI base URL
            os.environ["OPENAI_API_BASE"] = route.base_url or ""
            os.environ["OPENAI_API_KEY"] = route.api_key or ""
        # For direct, let existing logic handle it


def get_model_provider(model: str) -> str:
    """Determine the provider for a given model.

    This is moved from cli.py and enhanced for gateway support.

    Args:
        model: Model name

    Returns:
        Provider name
    """
    model_lower = model.lower()

    if model_lower.startswith("openrouter/"):
        return "openrouter"
    elif model_lower.startswith("xai/") or any(
        name in model_lower for name in ["grok"]
    ):
        return "xai"
    elif any(name in model_lower for name in ["gpt", "openai", "o1-"]):
        return "openai"
    elif any(name in model_lower for name in ["claude", "anthropic"]):
        return "anthropic"
    elif "gemini" in model_lower:
        return "gemini"
    elif "ollama" in model_lower:
        return "ollama"
    elif "deepseek" in model_lower:
        return "deepseek"
    else:
        return "unknown"


def is_reasoning_model(model: str) -> bool:
    """Check if a model supports reasoning/thinking capabilities.

    This is moved from cli.py for consistency.

    Args:
        model: Model name

    Returns:
        True if model supports reasoning
    """
    model_lower = model.lower()

    # DeepSeek reasoning models
    if any(name in model_lower for name in ["deepseek-reasoner", "deepseek-r1"]):
        return True

    # OpenAI o1 models
    if any(name in model_lower for name in ["o1-preview", "o1-mini", "o1-pro"]):
        return True

    # xAI Grok models with reasoning
    if any(name in model_lower for name in ["grok-3", "grok-4"]):
        return True

    return False
