"""Which shell a bash tool call runs in, and how to stop it.

The `bash` tool is named bash and the model is prompted for bash, so it writes
bash: `[ -f x ]`, `$(...)`, `cmd && cmd`, heredocs. On Linux and macOS `shell=True`
delivers exactly that. On Windows it delivers cmd.exe, which shares almost none of
that syntax — a bare `if` is already a parse error — so most of what the model emits
comes back as "i was unexpected at this time" rather than a result.

Git for Windows ships a real bash, and anyone who cloned this repo on Windows has
it. So: prefer that bash, fall back to cmd only when there is none, and tell the
model which one it got (prompts.environment_block) instead of claiming bash and
handing it cmd.

Process groups differ too, and the timeout path depends on them — killing only the
shell orphans everything the shell started. start_new_session is POSIX;
CREATE_NEW_PROCESS_GROUP plus taskkill /T is the Windows equivalent.
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
    """A usable POSIX shell on Windows, or None. Always None elsewhere, where
    shell=True is already the right shell and no path is needed."""
    if not WINDOWS:
        return None
    for var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        root = os.environ.get(var)
        if root and (exe := Path(root) / "Git" / "bin" / "bash.exe").is_file():
            return str(exe)
    found = shutil.which("bash")
    # System32\bash.exe is the WSL launcher, not a shell for this machine: it runs
    # the command inside the Linux VM, where C:\Users\... is not a path. A shell
    # that cannot see the project is worse than cmd, so it does not count.
    if found and "system32" not in found.lower():
        return found
    return None


BASH_EXE = _find_bash()
# What the model is told it is writing for. The only honest values are the two we
# can actually deliver.
NAME = "cmd" if (WINDOWS and BASH_EXE is None) else "bash"
# How to decode what the child writes. A POSIX shell and Git bash's coreutils emit
# UTF-8 whatever the machine's locale is, so decoding by locale is what broke here:
# on Korean Windows that is cp949, and the first UTF-8 byte from `cat` on a Korean
# file killed the reader thread. cmd.exe is the other way round — its tools write
# the machine's codepage, and forcing UTF-8 there would trade the crash for silent
# mojibake. So: follow the shell, not the platform.
ENCODING = "utf-8" if NAME == "bash" else locale.getpreferredencoding(False)


def popen(command: str, cwd) -> subprocess.Popen:
    """Start `command` in its own process group, so a timeout can kill the tree it
    spawned and not just the shell at the top of it (see kill_tree)."""
    # encoding is spelled out rather than left to text=True (which picks the locale's
    # — see ENCODING above). errors="replace" is the other half: a byte nobody can
    # decode still comes back as text, because one mangled line tells the model more
    # than a dead reader thread, which is what the crash actually was.
    kw = dict(cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
              text=True, encoding=ENCODING, errors="replace")
    if not WINDOWS:
        return subprocess.Popen(command, shell=True, start_new_session=True, **kw)
    kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    if BASH_EXE:
        # A list, not shell=True: shell=True on Windows means cmd.exe, and routing
        # bash through cmd would put cmd's own quoting rules in front of bash's.
        return subprocess.Popen([BASH_EXE, "-c", command], **kw)
    return subprocess.Popen(command, shell=True, **kw)


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill the command AND everything it started.

    A timeout that kills only the direct child — the shell — orphans whatever that
    shell launched, and it keeps running. Not theoretical: a few timed-out
    `uv run pytest` calls once left ~140 stray processes behind and took the load
    average to 198.
    """
    if WINDOWS:
        # No signal reaches a Windows process group the way SIGKILL does, so walk
        # the child tree by PID instead. /F because a shell that is being killed
        # will not pass a polite request on to its children.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
        if proc.poll() is None:  # taskkill declined (already gone, or no rights)
            proc.kill()
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:  # already gone, or not permitted — settle for the direct child
        proc.kill()
