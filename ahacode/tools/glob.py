"""glob: find files by path pattern, newest first."""

from __future__ import annotations

from ahacode import workspace
from ahacode.tools.base import Tool
from ahacode.tools.walk import iter_files

_MAX_RESULTS = 200  # so one search cannot flood the model's context


def _glob(args: dict) -> str:
    """List the matches as project-relative posix paths, newest first."""
    root = workspace.resolve_path(args["path"]) if args.get("path") else workspace.PROJECT_ROOT
    matches = [p for p in iter_files(root, args["pattern"]) if p.is_file()]
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)  # recency ≈ relevance

    shown = matches[:_MAX_RESULTS]
    lines = [workspace.display_path(p) for p in shown]
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
    parallelizable=True,
)
