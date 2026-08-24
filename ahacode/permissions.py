"""Rule-based pre-approval: which tool calls may run without asking.

The approval modal is the last line of defence, not the first. With no rules, every
`ls` and every test run costs a dialog — and because only one modal can be on screen
at a time, a fan-out of parallel sub-agents queues them one behind another: the
approval, not the work, becomes the bottleneck. A rule decided once, ahead of time,
takes the call out of that path entirely.

A rule is `tool:pattern`, matched against the call's SUBJECT — the one thing that
identifies it (bash → the command, grep/glob → the pattern, everything else → the
path). Matching is fnmatch, not regex: rules are hand-written and `*` is what people
expect there.

This never widens what is possible. The dangerous-command denylist runs BEFORE
approval (see agent._gate_tool), so a rule cannot authorise something the safety
gate already blocked.
"""

from __future__ import annotations

from fnmatch import fnmatchcase

from ahacode import config
from ahacode.tools.bash import split_chain


def subject(name: str, args: dict) -> str:
    """The one string that identifies a call — what a rule is matched against.

    Also what the UI puts in a tool card's title, so the rule a user writes reads
    like the line they saw on screen.
    """
    if name == "bash":
        raw = args.get("command", "")
    elif name in ("grep", "glob"):
        raw = args.get("pattern", "")
    else:
        raw = args.get("path", "")
    return str(raw or "").strip()


def _split_rule(rule: str) -> tuple[str, str]:
    """`"bash:git status*"` -> ("bash", "git status*"); `"read"` -> ("read", "*")."""
    tool, _, pattern = rule.partition(":")
    return tool.strip(), (pattern.strip() or "*")


def allowed(name: str, args: dict, rules: list[str] | tuple[str, ...] | None = None) -> bool:
    """Does a rule pre-approve this call?

    bash is special: the command is split on its chain operators and EVERY part must
    be allowed on its own. Matching the whole line would let `git status && rm -rf ~`
    ride in under a `git status*` rule — the same reason the denylist checks each
    sub-command separately.
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
