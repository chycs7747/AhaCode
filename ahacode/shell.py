"""Which shell a bash tool call runs in, and how to kill everything it started.

The model writes bash. On Windows `shell=True` means cmd.exe, so Git for Windows'
bash is preferred and cmd is the fallback; the model is told which one it got.
"""

from __future__ import annotations

import locale
import os
import shutil
import signal
import subprocess
from pathlib import Path

WINDOWS = os.name == "nt"


def _find_bash() -> str | None:
    """A usable POSIX shell on Windows, or None (always None elsewhere)."""
    if not WINDOWS:
        return None
    for var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        root = os.environ.get(var)
        if root and (exe := Path(root) / "Git" / "bin" / "bash.exe").is_file():
            return str(exe)
    found = shutil.which("bash")
    # System32\bash.exe is the WSL launcher: it runs inside the Linux VM, where a
    # Windows path is not a path, so it does not count.
    if found and "system32" not in found.lower():
        return found
    return None


BASH_EXE = _find_bash()
# What the model is told it is writing for: the shell it actually gets.
NAME = "cmd" if (WINDOWS and BASH_EXE is None) else "bash"
# A POSIX shell and Git bash's coreutils emit UTF-8 whatever the locale; cmd.exe's
# tools write the machine's codepage. So: follow the shell, not the platform.
ENCODING = "utf-8" if NAME == "bash" else locale.getpreferredencoding(False)


def popen(command: str, cwd) -> subprocess.Popen:
    """Start `command` in its own process group, so a timeout can kill the whole
    tree it spawned (see kill_tree).

    Args:
        command: The shell command.
        cwd: The working directory.

    Returns:
        The running process, with stderr folded into stdout.
    """
    # errors="replace": a byte nobody can decode still comes back as text, which
    # tells the model more than a dead reader thread.
    kw = dict(cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
              text=True, encoding=ENCODING, errors="replace")
    if not WINDOWS:
        return subprocess.Popen(command, shell=True, start_new_session=True, **kw)
    kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    if BASH_EXE:
        # A list, not shell=True: routing bash through cmd would put cmd's quoting
        # rules in front of bash's.
        return subprocess.Popen([BASH_EXE, "-c", command], **kw)
    return subprocess.Popen(command, shell=True, **kw)


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill the command and everything it started; killing only the shell would
    orphan its children.

    Args:
        proc: The process from popen.
    """
    if WINDOWS:
        # No signal reaches a Windows process group; walk the tree by PID. /F because
        # a shell that is being killed will not pass a polite request on.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
        if proc.poll() is None:  # taskkill declined (already gone, or no rights)
            proc.kill()
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:  # already gone, or not permitted
        proc.kill()
