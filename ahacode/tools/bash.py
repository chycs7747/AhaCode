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

from ahacode import config, shell, workspace
from ahacode.tools import spill
from ahacode.tools.base import Tool, clamp_timeout

# The default timeout lives in config; a call may ask for more, up to this ceiling.
MAX_TIMEOUT = 600

# A command is split into sub-commands before each one is checked, so that
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


def split_chain(command: str) -> list[str]:
    """Break a command line into its chained sub-commands.

    Public because the allow-rules use it too: an allowlist that matched the whole
    line would let a dangerous half ride in behind an allowed first command."""
    return [part.strip() for part in _CHAIN.split(command) if part.strip()]


def _check_dangerous(args: dict) -> str | None:
    """Return a block reason if the denylist matches, else None.

    Checked against the whole command *and* each chained sub-command: the whole
    line catches patterns that span operators (a fork bomb uses ; | &), while the
    per-sub-command pass means a dangerous half of `ls && rm -rf /` cannot hide
    behind a harmless first command.
    """
    command = args.get("command", "")
    for segment in (command, *split_chain(command)):
        for pattern, reason in _DENYLIST:
            if pattern.search(segment):
                return reason
    return None


def _bash(args: dict) -> str:
    seconds = clamp_timeout(args.get("timeout"), config.load().bash_timeout, MAX_TIMEOUT)
    # The model writes ordinary shell (pipes, globs), stderr folds into stdout so it
    # sees errors too, and the command gets its own process group so a timeout can
    # kill the whole tree. Which shell that is — and how to kill it — is per-platform;
    # shell.py owns that.
    proc = shell.popen(args["command"], cwd=workspace.PROJECT_ROOT)
    try:
        out, _ = proc.communicate(timeout=seconds)
        return _finish(out, proc.returncode)
    except subprocess.TimeoutExpired:
        shell.kill_tree(proc)
        # Keep what it managed to produce. Discarding it tells the model nothing about
        # how far the command got — which is the one thing that makes a timeout
        # actionable (a suite that printed 200 passing lines before being killed is
        # very different from one that printed nothing).
        out, _ = proc.communicate()
        out += (
            f"\n[timed out after {seconds}s and was killed — the output above is "
            f"partial. Re-run with a longer timeout, e.g. "
            f"timeout={min(seconds * 2, MAX_TIMEOUT)}]"
        )
        return _finish(out, None)


def _finish(out: str | None, returncode: int | None) -> str:
    """Spill if oversized, note a failing exit code, and hand back the text."""
    # communicate() hands back None for a pipe whose reader thread died. shell.py
    # fixes the cause (a locale-decoding crash on non-ASCII output); this makes the
    # symptom a turn that says "(no output)" rather than one that raises.
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
    requires_approval=True,   # arbitrary command -> confirm before running
    validate=_check_dangerous,  # catastrophic patterns are hard-blocked before that
)
