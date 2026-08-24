"""glob: find files by path pattern, newest first (read-only, no approval needed).

The "where is it?" half of navigating a codebase: `**/*.py`, `ahacode/**/test_*.py`.
Results are ordered by modification time descending, so the files a task is actually
about surface first.
"""

from __future__ import annotations

from ahacode.tools.base import PROJECT_ROOT, Tool, resolve_path
from ahacode.tools.walk import iter_files

_MAX_RESULTS = 200  # guard rail so one search can't flood the model's context


def _glob(args: dict) -> str:
    root = resolve_path(args["path"]) if args.get("path") else PROJECT_ROOT
    matches = [p for p in iter_files(root, args["pattern"]) if p.is_file()]
    # Newest first: recency is the best cheap proxy for "relevant to this task".
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    shown = matches[:_MAX_RESULTS]
    # Report paths relative to the project root — that's how the model refers to
    # them everywhere else (and what `read` accepts straight back).
    lines = [str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p)
             for p in shown]
    if len(matches) > len(shown):
        lines.append(f"... ({len(matches) - len(shown)} more; narrow the pattern)")
    return "\n".join(lines) or "(no files matched)"


GLOB = Tool(
    name="glob",
    description=(
        "Find files by path pattern (e.g. '**/*.py', 'ahacode/widgets/*.py'), "
        "most recently modified first. Use this to locate files by name; use grep "
        "to search their contents."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern, '**' matches any depth (e.g. '**/*.py')",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: the project root)",
            },
        },
        "required": ["pattern"],
    },
    execute=_glob,
)
