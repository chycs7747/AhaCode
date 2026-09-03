"""Where AhaCode works and where it writes: PROJECT_ROOT and the .ahacode layout.

PROJECT_ROOT is the directory AhaCode was launched in (AHACODE_ROOT overrides it),
resolved once at import. Consumers read these through the module (workspace.X), so
a test can redirect every path by patching one place.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("AHACODE_ROOT") or Path.cwd()).resolve()
AHACODE_DIR = PROJECT_ROOT / ".ahacode"
SESSIONS_DIR = AHACODE_DIR / "sessions"
PLANS_DIR = AHACODE_DIR / "plans"
CONFIG_PATH = AHACODE_DIR / "config.toml"  # optional per-project override
GLOBAL_CONFIG_PATH = Path.home() / ".ahacode" / "config.toml"


def resolve_path(path: str) -> Path:
    """Resolve a tool path against PROJECT_ROOT; absolute paths pass through.

    Args:
        path: A path as the model wrote it.

    Returns:
        The absolute path.
    """
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def display_path(path: Path) -> str:
    """Render a path the way the model and the user should see it.

    Project-relative when inside the project, and always with forward slashes: the
    string goes straight back into read() and bash, where a backslash escapes.

    Args:
        path: An absolute path.

    Returns:
        The posix-style path string.
    """
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
