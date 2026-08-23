"""bash: run a shell command in the project root. requires_approval=True — an
approved command runs with the user's own privileges.

Danger filtering: a command is split
on chain operators (&&, ||, ;, |, &, newlines) and each sub-command is matched
against a denylist of catastrophic patterns. A match hard-blocks the call before
it can even be offered for approval. This is defense in depth, NOT a guarantee —
a denylist can be worded around; the real safeguard is the human approval modal.
"""

from __future__ import annotations

import re
import subprocess

from ahacode.tools.base import PROJECT_ROOT, Tool

_TIMEOUT = 30  # seconds; a hung command must not freeze the agent

# Roo splits a command into sub-commands before checking each one, so that
# `ls && rm -rf /` cannot slip a dangerous half past the check.
_CHAIN = re.compile(r"&&|\|\||[;|&\n]")

# Only clearly catastrophic, system-wrecking patterns. Deliberately small:
# everything else still goes through the approval modal.
_DENYLIST: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\brm\b.*\s-\w*[rf]\w*.*\s(/|~|/\*|\$HOME)(\s|/|$)"),
     "recursive/force delete targeting / or ~"),
    (re.compile(r":\(\)\s*\{.*:.*\|.*:.*&.*\}\s*;\s*:"), "fork bomb"),
    (re.compile(r"\bmkfs(\.\w+)?\b"), "filesystem format (mkfs)"),
    (re.compile(r"\bdd\b[^|;&]*\bof=/dev/"), "raw write to a device (dd of=/dev/...)"),
    (re.compile(r">\s*/dev/(sd|nvme|hd|disk|mmcblk)"), "redirect to a block device"),
    (re.compile(r"\bchmod\s+-R\s+0*777\s+/(\s|$)"), "recursive chmod 777 on /"),
]


def _split_chain(command: str) -> list[str]:
    """Break a command line into its chained sub-commands (Roo parseCommand idea)."""
    return [part.strip() for part in _CHAIN.split(command) if part.strip()]


def _check_dangerous(args: dict) -> str | None:
    """Return a block reason if the denylist matches, else None.

    Checked against the whole command *and* each chained sub-command: the whole
    line catches patterns that span operators (a fork bomb uses ; | &), while the
    per-sub-command pass mirrors Roo's parseCommand so a dangerous half of
    `ls && rm -rf /` cannot hide behind a harmless first command.
    """
    command = args.get("command", "")
    for segment in (command, *_split_chain(command)):
        for pattern, reason in _DENYLIST:
            if pattern.search(segment):
                return reason
    return None


def _bash(args: dict) -> str:
    # shell=True: the model writes ordinary shell (pipes, globs). text=True
    # decodes bytes to str; stderr folds into stdout so the model sees errors too.
    proc = subprocess.run(
        args["command"],
        shell=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        out += f"\n(exit code {proc.returncode})"
    return out.strip() or "(no output)"


BASH = Tool(
    name="bash",
    description="Run a shell command in the project root and return its output.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run"},
        },
        "required": ["command"],
    },
    execute=_bash,
    requires_approval=True,   # arbitrary command -> confirm before running
    validate=_check_dangerous,  # catastrophic patterns are hard-blocked before that
)
