"""bash: run a shell command in the project root.

Approval-gated, with a denylist of catastrophic commands checked first. The
denylist is defense in depth, not a guarantee: the approval modal is the real
safeguard.
"""

from __future__ import annotations

import re
import subprocess

from ahacode import config, shell, workspace
from ahacode.tools import spill
from ahacode.tools.base import Tool, clamp_timeout

# The default timeout lives in config; a call may ask for more, up to this ceiling.
MAX_TIMEOUT = 600

# Chain operators; each sub-command is checked on its own.
_CHAIN = re.compile(r"&&|\|\||[;|&\n]")

# Only clearly catastrophic patterns; everything else goes through approval.
_DENYLIST: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\brm\b.*\s-\w*[rf]\w*.*\s(/|~|/\*|\$HOME)(\s|/|$)"),
     "recursive/force delete targeting / or ~"),
    (re.compile(r":\(\)\s*\{.*:.*\|.*:.*&.*\}\s*;\s*:"), "fork bomb"),
    (re.compile(r"\bmkfs(\.\w+)?\b"), "filesystem format (mkfs)"),
    (re.compile(r"\bdd\b[^|;&]*\bof=/dev/"), "raw write to a device (dd of=/dev/...)"),
    (re.compile(r">\s*/dev/(sd|nvme|hd|disk|mmcblk)"), "redirect to a block device"),
    (re.compile(r"\bchmod\s+-R\s+0*777\s+/(\s|$)"), "recursive chmod 777 on /"),
]


def split_chain(command: str) -> list[str]:
    """Break a command line into its chained sub-commands.

    Public because the allow rules use it too: matching the whole line would let a
    dangerous half ride in behind an allowed first command.

    Args:
        command: The shell command.

    Returns:
        The non-empty sub-commands, stripped.
    """
    return [part.strip() for part in _CHAIN.split(command) if part.strip()]


def _check_dangerous(args: dict) -> str | None:
    """The denylist reason for a command, or None when it is allowed through.

    Checked against the whole line (a fork bomb spans operators) and against each
    sub-command (so `ls && rm -rf /` cannot hide behind a harmless first half).
    """
    command = args.get("command", "")
    for segment in (command, *split_chain(command)):
        for pattern, reason in _DENYLIST:
            if pattern.search(segment):
                return reason
    return None


def _bash(args: dict) -> str:
    """Run the command, killing its whole process tree on timeout."""
    seconds = clamp_timeout(args.get("timeout"), config.load().bash_timeout, MAX_TIMEOUT)
    proc = shell.popen(args["command"], cwd=workspace.PROJECT_ROOT)
    try:
        out, _ = proc.communicate(timeout=seconds)
        return _finish(out, proc.returncode)
    except subprocess.TimeoutExpired:
        shell.kill_tree(proc)
        # Keep the partial output: it tells the model how far the command got.
        out, _ = proc.communicate()
        out += (
            f"\n[timed out after {seconds}s and was killed — the output above is "
            f"partial. Re-run with a longer timeout, e.g. "
            f"timeout={min(seconds * 2, MAX_TIMEOUT)}]"
        )
        return _finish(out, None)


def _finish(out: str | None, returncode: int | None) -> str:
    """Spill if oversized, note a failing exit code, and hand back the text."""
    # communicate() hands back None for a pipe whose reader thread died.
    out = spill.preview(out or "", prefix="bash", noun="output")
    if returncode:
        out += f"\n(exit code {returncode})"
    return out.strip() or "(no output)"


BASH = Tool(
    name="bash",
    description="Run a shell command in the project root and return its output.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run"},
            "timeout": {
                "type": "integer",
                "description": (
                    "Seconds to allow before the command is killed. Raise it for a "
                    f"test suite or a build; the maximum is {MAX_TIMEOUT}."
                ),
            },
        },
        "required": ["command"],
    },
    execute=_bash,
    requires_approval=True,
    validate=_check_dangerous,
)
