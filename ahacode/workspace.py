"""Where the agent works — the one definition of "project root".

PROJECT_ROOT is the directory AhaCode was *launched* in, not the directory its
source happens to sit in. The two are the same while you run AhaCode from its own
checkout, which is exactly why the distinction went unnoticed: config.py,
storage.py and tools/base.py each derived the root from `__file__` and each was
right by coincidence.

It stops being a coincidence the moment AhaCode is installed as a tool. The code
then lives in site-packages, so a `__file__`-derived root would point there — and
read/glob/grep/bash would search the installed package instead of the project you
opened. Launch directory is what those tools always meant; this is that.

Resolved once, at import: a later os.chdir cannot move the workspace out from
under a session that is already writing transcripts into it. AHACODE_ROOT overrides
it, for pointing a run at another tree without cd-ing there first.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("AHACODE_ROOT") or Path.cwd()).resolve()

# The per-machine home for settings that are not about any one project — the
# endpoint you configure once. Deliberately NOT under PROJECT_ROOT.
GLOBAL_DIR = Path.home() / ".ahacode"
