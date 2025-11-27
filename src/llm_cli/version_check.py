"""Version check and upgrade utilities for streamlm."""

import os
import sys
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
import requests
from rich.console import Console

console = Console()


def get_cache_path() -> Path:
    """Get the path to the version check cache file."""
    cache_dir = Path.home() / ".streamlm"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / "version_check.json"


def get_latest_version() -> Optional[str]:
    """Fetch the latest version from GitHub releases.

    Returns:
        Latest version string (e.g., "0.1.15") or None if failed
    """
    try:
        response = requests.get(
            "https://api.github.com/repos/jeffmylife/streamlm/releases/latest",
            timeout=2  # Quick timeout to not block CLI
        )
        if response.status_code == 200:
            data = response.json()
            # Tag name is like "v0.1.15", strip the 'v'
            tag_name = data.get("tag_name", "")
            return tag_name.lstrip("v")
        return None
    except Exception:
        # Silently fail - don't block the CLI if network is down
        return None


def should_check_version() -> bool:
    """Check if we should check for a new version (once per day).

    Returns:
        True if we should check, False otherwise
    """
    cache_path = get_cache_path()

    if not cache_path.exists():
        return True

    try:
        with open(cache_path, "r") as f:
            cache_data = json.load(f)

        last_check = datetime.fromisoformat(cache_data.get("last_check", ""))
        # Check if it's been more than 24 hours
        if datetime.now() - last_check > timedelta(days=1):
            return True

        return False
    except (json.JSONDecodeError, ValueError, KeyError):
        # Invalid cache, check again
        return True


def save_version_check(latest_version: Optional[str]) -> None:
    """Save the version check result to cache.

    Args:
        latest_version: The latest version found, or None if check failed
    """
    cache_path = get_cache_path()

    cache_data = {
        "last_check": datetime.now().isoformat(),
        "latest_version": latest_version,
    }

    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2)


def get_cached_latest_version() -> Optional[str]:
    """Get the latest version from cache.

    Returns:
        Latest version string or None if not cached
    """
    cache_path = get_cache_path()

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r") as f:
            cache_data = json.load(f)
        return cache_data.get("latest_version")
    except (json.JSONDecodeError, ValueError, KeyError):
        return None


def check_for_updates(current_version: str, is_being_piped: bool = False) -> None:
    """Check for updates and show notification if available.

    Args:
        current_version: Current installed version
        is_being_piped: Whether output is being piped (suppress notification)
    """
    # Don't show notifications when piped
    if is_being_piped:
        return

    # Only check once per day
    if not should_check_version():
        # Use cached version if available
        latest_version = get_cached_latest_version()
    else:
        # Fetch latest version
        latest_version = get_latest_version()
        save_version_check(latest_version)

    # Compare versions and show notification
    if latest_version and latest_version != current_version:
        # Parse versions for comparison
        try:
            current_parts = [int(x) for x in current_version.split(".")]
            latest_parts = [int(x) for x in latest_version.split(".")]

            # Only notify if latest is actually newer
            if latest_parts > current_parts:
                console.print(
                    f"[dim]💡 Update available: v{latest_version} "
                    f"(current: v{current_version}). Run 'lm upgrade' to update.[/dim]"
                )
        except (ValueError, AttributeError):
            # Skip if version parsing fails
            pass


def detect_installation_method() -> Tuple[Optional[str], Optional[str]]:
    """Detect how streamlm was installed.

    Returns:
        Tuple of (method, details) where method is one of:
        - 'uv-tool': Installed via uv tool install
        - 'pipx': Installed via pipx
        - 'pip-user': Installed via pip --user
        - 'pip-system': Installed via pip (system-wide)
        - 'editable': Installed in development mode (-e)
        - 'unknown': Could not detect
    """
    # Get the path to the current executable
    exe_path = sys.argv[0]

    # Also check sys.executable for the Python interpreter path
    python_path = sys.executable

    # Check for editable install
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "streamlm"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "Editable project location:" in result.stdout:
            return ("editable", "Development install")
    except Exception:
        pass

    # Check for uv tool install (check both exe and python paths)
    if ".local/share/uv/tools/streamlm" in exe_path or ".local/share/uv/tools/streamlm" in python_path:
        return ("uv-tool", "uv tool install")

    # Check for pipx
    if ".local/pipx/venvs/streamlm" in exe_path or "pipx" in exe_path or \
       ".local/pipx/venvs/streamlm" in python_path or "pipx" in python_path:
        return ("pipx", "pipx install")

    # Check for pip user install
    if ".local/lib/python" in exe_path or ".local/lib/python" in python_path:
        return ("pip-user", "pip install --user")

    # Check for system pip install
    if "/usr/local" in exe_path or "/usr/lib" in exe_path or \
       "/usr/local" in python_path or "/usr/lib" in python_path:
        return ("pip-system", "pip install (system)")

    return ("unknown", None)


def upgrade_streamlm() -> bool:
    """Upgrade streamlm to the latest version.

    Returns:
        True if upgrade was successful, False otherwise
    """
    from . import __version__

    method, details = detect_installation_method()

    console.print(f"\n[bold]streamlm upgrade[/bold]")
    console.print(f"Current version: [cyan]v{__version__}[/cyan]")
    console.print(f"Installation method: [yellow]{details or method}[/yellow]\n")

    # Don't upgrade editable installs
    if method == "editable":
        console.print(
            "[yellow]⚠️  You are using a development install. "
            "Please upgrade manually with git pull.[/yellow]"
        )
        return False

    # Get latest version
    console.print("[dim]Checking for updates...[/dim]")
    latest_version = get_latest_version()

    if not latest_version:
        console.print("[red]❌ Could not fetch latest version from GitHub.[/red]")
        return False

    console.print(f"Latest version: [green]v{latest_version}[/green]\n")

    # Check if already up to date
    if latest_version == __version__:
        console.print("[green]✓[/green] You are already on the latest version!")
        return True

    # Determine upgrade command
    if method == "uv-tool":
        upgrade_cmd = ["uv", "tool", "install", "--force", "streamlm"]
    elif method == "pipx":
        upgrade_cmd = ["pipx", "upgrade", "streamlm"]
    elif method == "pip-user":
        upgrade_cmd = [sys.executable, "-m", "pip", "install", "--user", "--upgrade", "streamlm"]
    elif method == "pip-system":
        upgrade_cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "streamlm"]
    else:
        console.print(
            f"[red]❌ Unknown installation method: {method}\n"
            "Please upgrade manually with: pip install --upgrade streamlm[/red]"
        )
        return False

    # Show what we'll run
    console.print(f"[dim]Running: {' '.join(upgrade_cmd)}[/dim]\n")

    # Run upgrade
    try:
        result = subprocess.run(
            upgrade_cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            console.print(f"[green]✓ Successfully upgraded to v{latest_version}![/green]")

            # Show release notes URL
            console.print(
                f"\n[dim]Release notes: "
                f"https://github.com/jeffmylife/streamlm/releases/tag/v{latest_version}[/dim]"
            )

            # Clear version cache
            save_version_check(latest_version)

            return True
        else:
            console.print(f"[red]❌ Upgrade failed:[/red]\n{result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        console.print("[red]❌ Upgrade timed out.[/red]")
        return False
    except Exception as e:
        console.print(f"[red]❌ Upgrade failed: {e}[/red]")
        return False
