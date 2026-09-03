"""Rule-based pre-approval: which tool calls may run without asking.

A rule is `tool:pattern`, matched with fnmatch against the call's subject (bash →
the command, grep/glob → the pattern, everything else → the path). Rules never
widen what is possible: the dangerous-command denylist runs before approval.
"""

from __future__ import annotations

from fnmatch import fnmatchcase

from ahacode import config
from ahacode.tools.bash import split_chain


def subject(name: str, args: dict) -> str:
    """The one string that identifies a call, which a rule is matched against.

    Also what a tool card's title shows, so a rule reads like the line on screen.

    Args:
        name: The tool name.
        args: The call's arguments.

    Returns:
        The command, pattern or path, stripped; "" when absent.
    """
    if name == "bash":
        raw = args.get("command", "")
    elif name in ("grep", "glob"):
        raw = args.get("pattern", "")
    else:
        raw = args.get("path", "")
    return str(raw or "").strip()


def _split_rule(rule: str) -> tuple[str, str]:
    """`"bash:git status*"` → ("bash", "git status*"); a bare `"read"` → ("read", "*")."""
    tool, _, pattern = rule.partition(":")
    return tool.strip(), (pattern.strip() or "*")


def allowed(name: str, args: dict, rules: list[str] | tuple[str, ...] | None = None) -> bool:
    """Whether a rule pre-approves this call.

    For bash the command is split on its chain operators and every part must be
    allowed on its own, so `git status && rm -rf ~` cannot ride in under a
    `git status*` rule.

    Args:
        name: The tool name.
        args: The call's arguments.
        rules: The rules to match; the configured allow_rules when omitted.

    Returns:
        True when a rule matches.
    """
    if rules is None:
        rules = config.load().allow_rules
    patterns = [pat for rule in rules for tool, pat in [_split_rule(rule)] if tool == name]
    if not patterns:
        return False
    if name == "bash":
        command = subject(name, args)
        parts = split_chain(command) or [command]
        return bool(parts) and all(
            any(fnmatchcase(part, pat) for pat in patterns) for part in parts
        )
    return any(fnmatchcase(subject(name, args), pat) for pat in patterns)
