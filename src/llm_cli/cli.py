import sys
import warnings
from typing import Optional, List, cast, Any
import os
import base64
import subprocess
import time

# Suppress Pydantic warning about config keys
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:*")

import typer
import litellm
from litellm import completion
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from rich.console import Console
from rich.traceback import install
from .streaming_markdown import StreamingMarkdownRenderer
from .config import get_config_manager
from .gateways import GatewayRouter, get_model_provider, is_reasoning_model
from .database import ConversationDatabase

install()

# Initialize Typer app and Rich console
app = typer.Typer(
    help="A CLI tool for interacting with various LLMs",
    name="llm",
    no_args_is_help=False,  # Allow calling without args to trigger default command
)
console = Console()


def version_callback_global(value: bool):
    """Global callback for --version option."""
    if value:
        from . import __version__

        commit_hash = get_git_commit_hash()
        console.print(f"streamlm version {__version__} (commit: {commit_hash})")
        raise typer.Exit()


# Callback to handle global options including --version
@app.callback(invoke_without_command=True)
def global_options(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback_global,
        is_eager=True,
        help="Show version information and exit",
    ),
):
    """A CLI tool for interacting with various LLMs with streaming markdown output.

    Design principle: The CLI should be frictionless. Users should be able to type
    'lm hello' without needing to specify a subcommand. The chat command is the default.
    """
    # If no subcommand is provided and not asking for version, default to chat
    if ctx.invoked_subcommand is None and not version:
        # We'll handle this by making the arguments pass through to chat
        pass


def get_git_commit_hash() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]  # Short hash
        else:
            return "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return "unknown"


def version_callback(value: bool):
    """Callback for --version option."""
    if value:
        from . import __version__

        commit_hash = get_git_commit_hash()
        console.print(f"streamlm version {__version__} (commit: {commit_hash})")
        raise typer.Exit()


litellm.suppress_debug_info = True
litellm.drop_params = True


def encode_image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def stream_llm_response(
    model: str,
    prompt: str,
    messages: List[dict],
    images: Optional[List[str]] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    show_reasoning: bool = False,
    is_being_piped: bool = False,
    raw_output: bool = False,
) -> tuple[str, str, Optional[dict]]:
    """Stream responses from the LLM and format them using Rich.

    Returns:
        Tuple of (accumulated_content, accumulated_reasoning, usage_info)
    """
    try:
        # Add images if provided
        if images:
            # For models that expect base64
            if any(
                name in model.lower()
                for name in ["gpt-4", "gemini", "claude-3", "deepseek", "o1-", "grok"]
            ):
                image_contents = []
                for img_path in images:
                    if img_path.startswith(("http://", "https://")):
                        image_contents.append(
                            {"type": "image_url", "image_url": img_path}
                        )
                    else:
                        base64_image = encode_image_to_base64(img_path)
                        image_contents.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            }
                        )
                # Update the content of the first message instead of replacing messages
                messages[0]["content"] = [
                    {"type": "text", "text": prompt},
                    *image_contents,
                ]
            # For Ollama vision models
            elif "ollama" in model.lower():
                for img_path in images:
                    if img_path.startswith(("http://", "https://")):
                        console.print(
                            "[red]Error: Ollama vision models only support local image files[/red]"
                        )
                        sys.exit(1)
                    else:
                        base64_image = encode_image_to_base64(img_path)
                        # Update the content of the first message
                        messages[0]["content"] = [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ]
            else:
                console.print(
                    "[red]Error: This model doesn't support image input[/red]"
                )
                sys.exit(1)

        # Initialize strings to accumulate the response
        accumulated_reasoning = ""
        accumulated_content = ""
        usage_info = None
        in_reasoning_phase = True

        # Check if this is a reasoning model and reasoning is requested
        supports_reasoning = is_reasoning_model(model)
        provider = get_model_provider(model)

        # Use our flicker-free streaming renderer for non-piped output (unless raw mode)
        if not is_being_piped and not raw_output:
            renderer = StreamingMarkdownRenderer()

            # Create a loading indicator and connect it to the renderer
            from .streaming_markdown import LoadingIndicator

            loading_indicator = LoadingIndicator(sys.stdout)
            renderer.set_loading_indicator(loading_indicator)

            # Initialize timing for first loading indicator show
            last_chunk_time = time.time()
            first_content_received = False

            try:
                # For reasoning models with reasoning enabled
                if supports_reasoning and show_reasoning:
                    # For direct DeepSeek API (non-OpenRouter)
                    if provider == "deepseek" and not model.lower().startswith(
                        "openrouter/"
                    ):
                        client = OpenAI(
                            api_key=os.getenv("DEEPSEEK_API_KEY"),
                            base_url="https://api.deepseek.com",
                        )

                        # Convert messages to proper type
                        typed_messages_live: List[ChatCompletionMessageParam] = []
                        for msg in messages:
                            typed_messages_live.append(msg)  # type: ignore

                        response_stream = client.chat.completions.create(
                            model=model.split("/")[-1],  # Remove 'deepseek/' prefix
                            messages=typed_messages_live,
                            stream=True,
                        )

                        # Start loading indicator initially
                        loading_indicator.start("initial")

                        for chunk in response_stream:
                            if (
                                hasattr(chunk.choices[0].delta, "reasoning_content")
                                and chunk.choices[0].delta.reasoning_content  # type: ignore
                            ):
                                reasoning = chunk.choices[0].delta.reasoning_content  # type: ignore
                                accumulated_reasoning += reasoning
                                # For reasoning content, we could show it differently
                                # but for now, we'll focus on the main content
                            elif chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                # Transition from reasoning to content phase
                                if in_reasoning_phase and accumulated_reasoning:
                                    in_reasoning_phase = False

                                accumulated_content += content

                                if raw_output:
                                    # Raw output: write content directly
                                    sys.stdout.write(content)
                                    sys.stdout.flush()
                                else:
                                    renderer.add_text(content)

                                    # Stop initial loading indicator once content starts
                                    if not first_content_received:
                                        loading_indicator.stop()
                                        first_content_received = True
                    else:
                        # Use litellm for OpenRouter and other reasoning models
                        response_stream = completion(
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stream=True,
                        )

                        # Start loading indicator
                        loading_indicator.start("initial")

                        for chunk in response_stream:
                            has_content = False

                            # Extract content from the chunk
                            delta = chunk.choices[0].delta  # type: ignore
                            content = delta.get("content", "")

                            # Check for reasoning content (may vary by provider)
                            reasoning_content = delta.get("reasoning_content", "")

                            if reasoning_content:
                                accumulated_reasoning += reasoning_content
                                has_content = True
                                # For reasoning content, we could show it differently
                                # but for now, we'll focus on the main content
                            elif content:
                                # Transition from reasoning to content phase
                                if in_reasoning_phase and accumulated_reasoning:
                                    in_reasoning_phase = False

                                accumulated_content += content

                                if raw_output:
                                    # Raw output: write content directly
                                    sys.stdout.write(content)
                                    sys.stdout.flush()
                                else:
                                    renderer.add_text(content)
                                has_content = True

                            # Only stop loading indicator and update time when we get actual content
                            if has_content and not raw_output:
                                if not first_content_received:
                                    loading_indicator.stop()
                                    first_content_received = True
                else:
                    # Use litellm for all non-reasoning models or when reasoning is disabled
                    response_stream = completion(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    )

                    # Start loading indicator
                    loading_indicator.start("initial")

                    for chunk in response_stream:
                        # Extract content from the chunk
                        delta = chunk.choices[0].delta  # type: ignore
                        content = delta.get("content", "")

                        if content:
                            # Stop loading indicator once content starts
                            if not first_content_received and not raw_output:
                                loading_indicator.stop()
                                first_content_received = True

                            accumulated_content += content

                            if raw_output:
                                # Raw output: write content directly
                                sys.stdout.write(content)
                                sys.stdout.flush()
                            else:
                                renderer.add_text(content)

            except KeyboardInterrupt:
                if not raw_output:
                    loading_indicator.stop()
                    console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
                else:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
            except Exception as e:
                if not raw_output:
                    loading_indicator.stop()
                    console.print(f"[red]❌ Error: {e}[/red]")
                else:
                    sys.stderr.write(f"Error: {e}\n")
                    sys.stderr.flush()
                raise
            finally:
                # Always finalize the renderer and stop loading indicator (if not raw mode)
                if not raw_output:
                    loading_indicator.stop()
                    renderer.finalize()
                elif accumulated_content and not accumulated_content.endswith("\n"):
                    # Add final newline for raw output
                    sys.stdout.write("\n")
                    sys.stdout.flush()

        elif is_being_piped or raw_output:
            # Direct streaming for piped output or raw output (plain text, no markdown rendering)
            if supports_reasoning and show_reasoning:
                # For direct DeepSeek API (non-OpenRouter)
                if provider == "deepseek" and not model.lower().startswith(
                    "openrouter/"
                ):
                    client = OpenAI(
                        api_key=os.getenv("DEEPSEEK_API_KEY"),
                        base_url="https://api.deepseek.com",
                    )

                    # Convert messages to proper type
                    typed_messages: List[ChatCompletionMessageParam] = []
                    for msg in messages:
                        typed_messages.append(msg)  # type: ignore

                    response_stream = client.chat.completions.create(
                        model=model.split("/")[-1],
                        messages=typed_messages,
                        stream=True,
                    )

                    for chunk in response_stream:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            # Write content directly, preserving line breaks
                            sys.stdout.write(content.replace("\\n", "\n"))
                            sys.stdout.flush()
                            accumulated_content += content
                else:
                    # Use litellm for OpenRouter and other reasoning models
                    response_stream = completion(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    )

                    for chunk in response_stream:
                        delta = chunk.choices[0].delta  # type: ignore
                        content = delta.get("content", "")
                        if content:
                            # Write content directly, preserving line breaks
                            sys.stdout.write(content.replace("\\n", "\n"))
                            sys.stdout.flush()
                            accumulated_content += content
            else:
                response_stream = completion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )

                for chunk in response_stream:
                    delta = chunk.choices[0].delta  # type: ignore
                    content = delta.get("content", "")
                    if content:
                        # Write content directly, preserving line breaks
                        sys.stdout.write(content.replace("\\n", "\n"))
                        sys.stdout.flush()
                        accumulated_content += content

            # Add final newline for piped output
            if not accumulated_content.endswith("\n") and is_being_piped:
                sys.stdout.write("\n")
                sys.stdout.flush()

        # Return accumulated content and reasoning
        return accumulated_content, accumulated_reasoning, usage_info

    except Exception as e:
        if not is_being_piped:
            console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


@app.command(context_settings={"ignore_unknown_options": True})
def chat(
    prompt: list[str] = typer.Argument(..., help="The prompt to send to the LLM"),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="The LLM model to use. Examples: gpt-4o, claude-3-sonnet-20240229, ollama/llama2",
    ),
    images: Optional[List[str]] = typer.Option(
        None,
        "--image",
        "-i",
        help="Path to image file or URL. Can be specified multiple times for multiple images.",
    ),
    context: Optional[str] = typer.Option(
        None,
        "--context",
        "-c",
        help="Path to a file to use as context for the prompt",
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", "-t", help="Maximum number of tokens to generate"
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", "-temp", help="Sampling temperature (0.0 to 1.0)"
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
    think: bool = typer.Option(
        False,
        "--think",
        help="Show the model's reasoning process (works with reasoning models like DeepSeek, OpenAI o1, etc.)",
    ),
    gateway: Optional[str] = typer.Option(
        None,
        "--gateway",
        "-g",
        help="Gateway to route request through: direct, vercel, openrouter",
    ),
    raw: bool = typer.Option(
        False,
        "--raw",
        "--md",
        help="Output raw markdown without Rich formatting",
    ),
    session: Optional[str] = typer.Option(
        None,
        "--session",
        "-s",
        help="Session ID to continue conversation. Creates new session if it doesn't exist.",
    ),
    session_name: Optional[str] = typer.Option(
        None,
        "--session-name",
        help="Name for the session (only used when creating a new session)",
    ),
):
    """Chat with an LLM model and get markdown-formatted responses. Supports image input for compatible models."""

    # Check if we're being piped to another command
    is_being_piped = not sys.stdout.isatty()

    # Initialize config manager and gateway router
    config_mgr = get_config_manager()
    router = GatewayRouter(config_mgr)

    # Initialize database for session management
    config = config_mgr.load()
    db_path = os.path.expanduser("~/.streamlm/conversations.db")
    db = ConversationDatabase(db_path)

    # Handle session management
    db_session = None
    if session:
        # Check if session exists
        db_session = db.get_session(session)
        if db_session:
            if not is_being_piped:
                console.print(f"[dim]Continuing session '{db_session.name}'[/dim]")
        else:
            # Create new session
            session_display_name = session_name or session
            db_session = db.create_session(
                session_id=session,
                name=session_display_name,
                model=model or config.default_model,
            )
            if not is_being_piped:
                console.print(f"[dim]Created new session '{session_display_name}'[/dim]")

    # Resolve model from config if not specified
    if model is None:
        model = config.default_model
    else:
        # Resolve model aliases
        model = config_mgr.resolve_model_alias(model)

    # Only show debug info if we're not being piped
    if not is_being_piped:
        print("Starting chat function...")  # Debug print

        if debug:
            print("Debug mode enabled")  # Basic print for debugging
            # litellm.set_verbose = True  # Not available in current version

    # Join the prompt list into a single string
    prompt_text = " ".join(prompt)
    display_text = prompt_text

    # Prepare the message content
    message_content = prompt_text

    # Check for piped input
    if not sys.stdin.isatty():
        piped_input = sys.stdin.read().strip()
        if piped_input:
            # Format the message with prompt first, then previous output
            message_content = f"{prompt_text}\n\n{piped_input}"
            display_text = f"{prompt_text}\n\n<Previous output>\n"

    # If context file is provided, read it and append to both display and message
    if context:
        try:
            with open(context, "r") as f:
                context_content = f.read()
                display_text = f"{display_text}\n\n# {os.path.basename(context)}\n..."
                message_content = f"{message_content}\n\nHere's the content of {os.path.basename(context)}:\n\n{context_content}"
        except Exception as e:
            console.print(f"[red]Error reading context file: {str(e)}[/red]")
            sys.exit(1)

    # Create the messages list with session history if available
    if db_session:
        # Load conversation history
        messages = db.get_message_history_for_llm(session)
        # Add the new user message
        messages.append({"role": "user", "content": message_content})
    else:
        messages = [{"role": "user", "content": message_content}]

    # Only show prompt info if we're not being piped
    if not is_being_piped:
        print(f"Prompt: {display_text}")  # Debug print

    # Determine provider and route through gateway
    provider = get_model_provider(model)

    # Route the request
    try:
        route = router.route_request(model, gateway_override=gateway, provider=provider)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    # Update model name if gateway routing modified it
    model = route.model_name

    # Configure litellm if using a gateway
    if route.gateway != "direct":
        router.configure_litellm_for_route(route)

        if not is_being_piped:
            console.print(f"[dim]Routing through {route.gateway} gateway[/dim]")

    # Check if reasoning is supported with this gateway
    reasoning_supported = router.supports_reasoning(model, route.gateway)
    if think and not reasoning_supported:
        console.print(
            f"[yellow]Warning: Reasoning mode not supported via {route.gateway} gateway. "
            f"Use --gateway direct for reasoning features.[/yellow]"
        )
        think = False

    # Validate and check API keys based on the model (only for direct access)
    # For gateway routing, the gateway handles authentication
    if route.gateway == "direct":
        if provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                console.print(
                    "[red]Error: OPENROUTER_API_KEY environment variable is not set[/red]"
                )
                sys.exit(1)
            if debug and not is_being_piped:
                console.print(
                    f"[dim]Found OpenRouter API key: {api_key[:4]}...{api_key[-4:]}[/dim]"
                )
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                console.print(
                    "[red]Error: OPENAI_API_KEY environment variable is not set[/red]"
                )
                sys.exit(1)
            if debug and not is_being_piped:
                console.print(
                    f"[dim]Found OpenAI API key: {api_key[:4]}...{api_key[-4:]}[/dim]"
                )
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                console.print(
                    "[red]Error: ANTHROPIC_API_KEY environment variable is not set[/red]"
                )
                sys.exit(1)
            if debug and not is_being_piped:
                console.print(
                    f"[dim]Found Anthropic API key: {api_key[:4]}...{api_key[-4:]}[/dim]"
                )
        elif provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                console.print(
                    "[red]Error: GEMINI_API_KEY environment variable is not set[/red]"
                )
                sys.exit(1)
            if debug and not is_being_piped:
                console.print(
                    f"[dim]Found Gemini API key: {api_key[:4]}...{api_key[-4:]}[/dim]"
                )
        elif provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                console.print(
                    "[red]Error: DEEPSEEK_API_KEY environment variable is not set[/red]"
                )
                sys.exit(1)
            if debug and not is_being_piped:
                console.print(
                    f"[dim]Found DeepSeek API key: {api_key[:4]}...{api_key[-4:]}[/dim]"
                )
        elif provider == "xai":
            api_key = os.getenv("XAI_API_KEY")
            if not api_key:
                console.print(
                    "[red]Error: XAI_API_KEY environment variable is not set[/red]"
                )
                sys.exit(1)
            if debug and not is_being_piped:
                console.print(
                    f"[dim]Found xAI API key: {api_key[:4]}...{api_key[-4:]}[/dim]"
                )
        elif provider == "ollama":
            # Check if Ollama server is running
            try:
                import requests

                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code != 200:
                    console.print(
                        "[red]Error: Ollama server is not running. Please start it with 'ollama serve'[/red]"
                    )
                    sys.exit(1)
            except requests.exceptions.ConnectionError:
                console.print(
                    "[red]Error: Cannot connect to Ollama server. Please start it with 'ollama serve'[/red]"
                )
                sys.exit(1)

    # Show what model we're using (only if not being piped)
    if not is_being_piped:
        console.print(f"[dim]Using model: {model}[/dim]")
        if images:
            console.print(
                f"[dim]With {len(images)} image{'s' if len(images) > 1 else ''}[/dim]"
            )
        console.print()  # Add a blank line for cleaner output

    # Configure model-specific settings
    if "ollama" in model.lower():
        # litellm.set_verbose = False  # Not available in current version
        os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
        # Format for litellm's Ollama support
        model = f"ollama/{model.split('/')[-1]}"

    # Stream the response
    try:
        response_content, reasoning_content, usage = stream_llm_response(
            model=model,
            prompt=prompt_text,
            messages=messages,
            images=images,
            max_tokens=max_tokens,
            temperature=temperature,
            show_reasoning=think,
            is_being_piped=is_being_piped,  # Pass pipe status to response handler
            raw_output=raw,
        )

        # Save to database if session is active
        if db_session and response_content:
            # Save user message
            user_metadata = {}
            if images:
                user_metadata["images"] = images
            if context:
                user_metadata["context_file"] = context

            db.add_message(
                session_id=session,
                role="user",
                content=message_content,
                model=model,
                metadata=user_metadata if user_metadata else None,
            )

            # Save assistant response
            prompt_tokens = usage.get("prompt_tokens") if usage else None
            completion_tokens = usage.get("completion_tokens") if usage else None

            db.add_message(
                session_id=session,
                role="assistant",
                content=response_content,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_content=reasoning_content if reasoning_content else None,
            )

    except Exception as e:
        if not is_being_piped:
            print(f"Error occurred: {str(e)}")  # Basic print for errors
            if debug:
                import traceback

                traceback.print_exc()
        sys.exit(1)


@app.command(name="config")
def config_cmd(
    action: str = typer.Argument(
        help="Action to perform: get, set, validate, list-gateways, setup"
    ),
    key: Optional[str] = typer.Argument(None, help="Config key (for get/set)"),
    value: Optional[str] = typer.Argument(None, help="Config value (for set)"),
):
    """Manage streamlm configuration."""
    config_mgr = get_config_manager()

    if action == "get":
        # Get config value
        result = config_mgr.get(key)
        if result is None:
            console.print(f"[yellow]Config key '{key}' not found[/yellow]")
        else:
            import json

            console.print(json.dumps(result, indent=2))

    elif action == "set":
        # Set config value
        if not key or value is None:
            console.print("[red]Error: 'set' requires both key and value[/red]")
            console.print("Example: lm config set gateway.default vercel")
            sys.exit(1)

        try:
            config_mgr.set(key, value)
            console.print(f"[green]✓ Set {key} = {value}[/green]")
            console.print(f"[dim]Config saved to {config_mgr.config_path}[/dim]")
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    elif action == "validate":
        # Validate configuration
        results = config_mgr.validate()

        console.print("\n[bold]Configuration Validation[/bold]")
        console.print(f"Config file: {results['config_file']}")
        console.print(f"Exists: {'✓' if results['exists'] else '✗'}\n")

        console.print("[bold]Gateway Configuration:[/bold]")
        console.print(f"  Default: {results['gateway']['default']}")
        console.print(
            f"  Vercel: {'✓' if results['gateway']['vercel']['api_key_set'] else '✗'} API key"
        )
        console.print(
            f"  OpenRouter: {'✓' if results['gateway']['openrouter']['api_key_set'] else '✗'} API key\n"
        )

        console.print("[bold]Provider API Keys:[/bold]")
        for provider, info in results["providers"].items():
            status = "✓" if info["available"] else "✗"
            source = []
            if info["config_key_set"]:
                source.append("config")
            if info["env_key_set"]:
                source.append("env")
            source_str = f" ({', '.join(source)})" if source else ""
            console.print(f"  {provider}: {status}{source_str}")

        console.print(f"\n[bold]Models:[/bold]")
        console.print(f"  Default: {results['models']['default']}")
        console.print(f"  Aliases: {len(results['models']['aliases'])} defined")

    elif action == "list-gateways":
        # List available gateways
        config = config_mgr.load()

        console.print("\n[bold]Available Gateways:[/bold]\n")

        console.print("[cyan]direct[/cyan] (default)")
        console.print("  Route directly to provider APIs")
        console.print("  Supports: All providers, reasoning models")
        console.print("  Requires: Provider-specific API keys\n")

        console.print("[cyan]vercel[/cyan]")
        console.print("  Route through Vercel AI Gateway")
        console.print("  Benefits: No markup, $5/month free, low latency (<20ms)")
        console.print(
            f"  Status: {'✓ Configured' if config.vercel.api_key else '✗ Not configured'}"
        )
        console.print("  Requires: AI_GATEWAY_API_KEY\n")

        console.print("[cyan]openrouter[/cyan]")
        console.print("  Route through OpenRouter")
        console.print("  Benefits: Model discovery, pricing transparency, BYOK")
        console.print(
            f"  Status: {'✓ Configured' if config.openrouter.api_key else '✗ Not configured'}"
        )
        console.print("  Requires: OPENROUTER_API_KEY\n")

        console.print("[dim]Current default: " + config_mgr.get_gateway() + "[/dim]")

    elif action == "setup":
        # Interactive setup wizard
        console.print("\n[bold]StreamLM Configuration Setup[/bold]\n")

        config = config_mgr.load()

        # Gateway selection
        console.print("[bold]1. Default Gateway[/bold]")
        console.print("Which gateway would you like to use by default?")
        console.print("  [cyan]direct[/cyan]    - Direct provider access (current behavior)")
        console.print(
            "  [cyan]vercel[/cyan]    - Vercel AI Gateway (no markup, low latency)"
        )
        console.print(
            "  [cyan]openrouter[/cyan] - OpenRouter (model discovery, BYOK)"
        )

        gateway_choice = typer.prompt(
            "Gateway", default=config.default_gateway, type=str
        )

        if gateway_choice in ["direct", "vercel", "openrouter"]:
            config.default_gateway = gateway_choice
        else:
            console.print(
                f"[yellow]Invalid gateway '{gateway_choice}', keeping '{config.default_gateway}'[/yellow]"
            )

        # Gateway API keys
        if gateway_choice == "vercel":
            console.print("\n[bold]2. Vercel AI Gateway Configuration[/bold]")
            console.print("Enter your Vercel AI Gateway API key (or leave blank to skip):")
            vercel_key = typer.prompt("AI_GATEWAY_API_KEY", default="", type=str)
            if vercel_key:
                config.vercel.api_key = vercel_key

        elif gateway_choice == "openrouter":
            console.print("\n[bold]2. OpenRouter Configuration[/bold]")
            console.print("Enter your OpenRouter API key (or leave blank to skip):")
            openrouter_key = typer.prompt("OPENROUTER_API_KEY", default="", type=str)
            if openrouter_key:
                config.openrouter.api_key = openrouter_key

        # Default model
        console.print("\n[bold]3. Default Model[/bold]")
        console.print(f"Current default: {config.default_model}")
        new_model = typer.prompt("Default model", default=config.default_model, type=str)
        config.default_model = new_model

        # Save configuration
        config_mgr.save(config)

        console.print(
            f"\n[green]✓ Configuration saved to {config_mgr.config_path}[/green]"
        )
        console.print("\nRun 'lm config validate' to check your configuration.")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print(
            "Valid actions: get, set, validate, list-gateways, setup"
        )
        sys.exit(1)


@app.command(name="sessions")
def sessions_cmd(
    list_all: bool = typer.Option(False, "--list", "-l", help="List all sessions"),
    show: Optional[str] = typer.Option(None, "--show", help="Show session details and messages"),
    delete: Optional[str] = typer.Option(None, "--delete", help="Delete a session"),
    clear: Optional[str] = typer.Option(None, "--clear", help="Clear messages in a session"),
    export: Optional[str] = typer.Option(None, "--export", help="Export session to JSON"),
):
    """Manage conversation sessions."""
    from datetime import datetime

    db_path = os.path.expanduser("~/.streamlm/conversations.db")
    db = ConversationDatabase(db_path)

    if list_all:
        # List all sessions
        sessions = db.list_sessions()
        if not sessions:
            console.print("[yellow]No sessions found.[/yellow]")
            return

        console.print("\n[bold]Conversation Sessions[/bold]\n")

        from rich.table import Table

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("ID", style="green")
        table.add_column("Name", style="white")
        table.add_column("Model", style="blue")
        table.add_column("Messages", justify="right", style="yellow")
        table.add_column("Tokens", justify="right", style="magenta")
        table.add_column("Updated", style="dim")

        for sess in sessions:
            updated = datetime.fromtimestamp(sess.updated_at).strftime("%Y-%m-%d %H:%M")
            table.add_row(
                sess.id,
                sess.name,
                sess.model,
                str(sess.message_count),
                str(sess.total_tokens) if sess.total_tokens > 0 else "-",
                updated,
            )

        console.print(table)
        console.print()

    elif show:
        # Show session details
        session = db.get_session(show)
        if not session:
            console.print(f"[red]Session '{show}' not found.[/red]")
            sys.exit(1)

        messages = db.get_messages(show)

        console.print(f"\n[bold]Session: {session.name}[/bold]")
        console.print(f"[dim]ID: {session.id}[/dim]")
        console.print(f"[dim]Model: {session.model}[/dim]")
        console.print(f"[dim]Messages: {session.message_count}[/dim]")
        console.print(f"[dim]Total tokens: {session.total_tokens}[/dim]")
        created = datetime.fromtimestamp(session.created_at).strftime("%Y-%m-%d %H:%M:%S")
        updated = datetime.fromtimestamp(session.updated_at).strftime("%Y-%m-%d %H:%M:%S")
        console.print(f"[dim]Created: {created}[/dim]")
        console.print(f"[dim]Updated: {updated}[/dim]")
        console.print()

        if messages:
            console.print("[bold]Conversation History:[/bold]\n")
            for msg in messages:
                role_color = "green" if msg.role == "user" else "blue"
                timestamp = datetime.fromtimestamp(msg.created_at).strftime("%H:%M:%S")
                console.print(f"[{role_color}]■ {msg.role}[/{role_color}] [dim]({timestamp})[/dim]")
                console.print(f"  {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}\n")

                if msg.reasoning_content:
                    console.print(f"  [dim italic]💭 Reasoning: {msg.reasoning_content[:100]}...[/dim italic]\n")
        else:
            console.print("[dim]No messages in this session.[/dim]")

    elif delete:
        # Delete session
        session = db.get_session(delete)
        if not session:
            console.print(f"[red]Session '{delete}' not found.[/red]")
            sys.exit(1)

        if db.delete_session(delete):
            console.print(f"[green]✓[/green] Session '{session.name}' deleted.")
        else:
            console.print(f"[red]Failed to delete session '{delete}'.[/red]")
            sys.exit(1)

    elif clear:
        # Clear messages in session
        session = db.get_session(clear)
        if not session:
            console.print(f"[red]Session '{clear}' not found.[/red]")
            sys.exit(1)

        if db.clear_messages(clear):
            console.print(f"[green]✓[/green] Cleared {session.message_count} messages from session '{session.name}'.")
        else:
            console.print(f"[red]Failed to clear messages from session '{clear}'.[/red]")
            sys.exit(1)

    elif export:
        # Export session to JSON
        data = db.export_session(export)
        if not data:
            console.print(f"[red]Session '{export}' not found.[/red]")
            sys.exit(1)

        import json

        print(json.dumps(data, indent=2))

    else:
        console.print("[yellow]Please specify an action: --list, --show, --delete, --clear, or --export[/yellow]")
        console.print("Usage: lm sessions --list")
        sys.exit(1)


def main():
    """Entry point for the CLI.

    Design principle: Make chat the default command for frictionless UX.
    Users should be able to type 'lm hello' without 'chat' subcommand.
    """
    import sys

    # If first arg is not a known command, inject 'chat'
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        known_commands = ['chat', 'config', 'sessions']
        global_flags = ['--version', '-v', '--help', '--install-completion', '--show-completion']

        # If first arg is not a command and not a global flag, default to chat
        if first_arg not in known_commands and first_arg not in global_flags:
            sys.argv.insert(1, 'chat')

    app()


if __name__ == "__main__":
    main()
