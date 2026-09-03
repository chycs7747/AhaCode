"""The tool contract. A tool is a value of this dataclass, never a subclass, so
adding one never touches this file."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    """One callable the model may invoke by name."""

    name: str
    description: str
    parameters: dict  # JSON Schema for the arguments object
    execute: Callable[[dict], str]
    requires_approval: bool = False  # side effects: confirmed before it runs
    # A safety gate checked before approval: a reason hard-blocks the call.
    validate: Callable[[dict], str | None] | None = None
    wants_ctx: bool = False  # the loop calls execute(args, ctx) instead of execute(args)
    # A turn's calls run in parallel only when every runnable tool allows it.
    parallelizable: bool = False


def clamp_timeout(requested, default: int, maximum: int) -> int:
    """Seconds a call may run: what it asked for, clamped to [1, maximum].

    Args:
        requested: The call's own timeout argument, possibly missing or junk.
        default: Used when `requested` is missing or not a number.
        maximum: The ceiling a single call may ask for.

    Returns:
        The timeout in seconds.
    """
    try:
        return max(1, min(int(requested), maximum))
    except (TypeError, ValueError):
        return default
