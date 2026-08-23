"""The tool contract: a name, a JSON-Schema parameter spec, and an execute()
callable. Concrete tools live in their own modules (read.py, bash.py, ...) and
are assembled in __init__.py — mirroring Pi's harness/tools/ directory (one file
per tool + an index) and Roo Code's core/tools/. A tool is a *value* of this
dataclass, never a subclass, so adding one never touches this file.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Tools resolve relative paths / run commands against the project root, so the
# agent's "workspace" matches where AhaCode was launched. Defined once here.
# tools/base.py -> tools -> ahacode -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_path(path: str) -> Path:
    """Resolve a tool path against the project root (absolute paths pass through)."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass(frozen=True)
class Tool:
    """One callable the model may invoke by name."""

    name: str
    description: str
    parameters: dict  # JSON Schema describing the arguments object
    execute: Callable[[dict], str]
    # Tools with side effects (bash, write) must be confirmed before they run.
    requires_approval: bool = False
    # Optional safety gate, checked *before* approval: return a reason to hard-block
    # the call (it never runs, never prompts), or None to allow it through.
    validate: Callable[[dict], str | None] | None = None
    # A tool that spawns a sub-agent (task) needs the running context, so the loop
    # calls execute(args, ctx) instead of execute(args). Plain tools leave this False.
    wants_ctx: bool = False
    # A turn's tool calls run in parallel only when EVERY runnable tool is
    # parallelizable — safe for delegation (task), off for side-effecting tools.
    parallelizable: bool = False
