"""grep: search file contents by regular expression, reported as `path:line:text`."""

from __future__ import annotations

import re

from ahacode import workspace
from ahacode.tools.base import Tool
from ahacode.tools.walk import iter_files, read_text_or_none

_MAX_MATCHES = 100  # so one search cannot flood the model's context
_MAX_LINE_CHARS = 300  # a minified line must not blow up a whole result


def _grep(args: dict) -> str:
    """Search the files under the given root and list the matching lines."""
    regex = re.compile(args["pattern"])  # a bad regex surfaces as a tool error here
    root = workspace.resolve_path(args["path"]) if args.get("path") else workspace.PROJECT_ROOT
    pattern = args.get("glob") or "**/*"

    hits: list[str] = []
    files_with_hits = 0
    truncated = False
    for path in iter_files(root, pattern):
        if not path.is_file():
            continue
        text = read_text_or_none(path)
        if text is None:
            continue
        name = workspace.display_path(path)
        matched_here = False
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not regex.search(line):
                continue
            matched_here = True
            body = line.strip()
            if len(body) > _MAX_LINE_CHARS:
                body = body[:_MAX_LINE_CHARS] + "…"
            hits.append(f"{name}:{lineno}:{body}")
            if len(hits) >= _MAX_MATCHES:
                truncated = True
                break
        files_with_hits += matched_here
        if truncated:
            break

    if not hits:
        return "(no matches)"
    if truncated:
        hits.append(f"... (stopped at {_MAX_MATCHES} matches; narrow the pattern or the glob)")
    else:
        hits.append(f"({len(hits)} matches in {files_with_hits} files)")
    return "\n".join(hits)


GREP = Tool(
    name="grep",
    description=(
        "Search file contents with a regular expression and return matching lines as "
        "'path:line:text'. Narrow the search with glob (e.g. '**/*.py') or path. "
        "Use this to find where something is defined or used."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression to search for"},
            "path": {
                "type": "string",
                "description": "Directory to search in (default: the project root)",
            },
            "glob": {
                "type": "string",
                "description": "Only search files matching this pattern (e.g. '**/*.py')",
            },
        },
        "required": ["pattern"],
    },
    execute=_grep,
    parallelizable=True,
)
