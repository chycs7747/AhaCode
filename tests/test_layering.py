"""The layer boundary, enforced instead of documented.

The README says the harness is widget-free and that all provider traffic goes
through client.py. Both were true by discipline alone — nothing failed if a
later edit imported a widget into the agent loop, and the cost of that mistake
is not cosmetic: it is what makes `agent.run` testable without a terminal, and
what lets a sub-agent run on a worker thread at all.

A folder per layer would not have caught it either. This does.
"""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "ahacode"

# The layer that must not know about screens, widgets, or Textual. These modules
# reach the UI only through the injected `emit` callback.
HARNESS = {
    "agent.py", "subagent.py", "context.py", "client.py", "storage.py",
    "config.py", "permissions.py", "prompts.py", "render.py", "text.py",
    "shell.py", "workspace.py", "session.py", "events.py",
}

# What the harness may not import, by module prefix. NOT rich: render.py exists
# to build Rich renderables, and a renderable is data, not a mounted widget —
# which is exactly why the chat and the approval modal can share one.
FORBIDDEN_FOR_HARNESS = ("textual", "ahacode.widgets", "ahacode.app")

# Only client.py may reach a MODEL provider. Plain HTTP is not on this list:
# webfetch is a tool that fetches web pages, and bounding it by the model
# gateway's concurrency gate would be meaningless.
MAY_IMPORT_PROVIDER = {"client.py"}
PROVIDER_PACKAGES = ("openai", "anthropic", "google.generativeai", "mistralai", "cohere")


def _imports(path: pathlib.Path) -> list[tuple[str, int]]:
    """Every module name this file imports, with its line number."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name, node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append((node.module, node.lineno))
    return out


def _files(where: pathlib.Path) -> list[pathlib.Path]:
    return [p for p in sorted(where.rglob("*.py")) if "__pycache__" not in str(p)]


@pytest.mark.parametrize("name", sorted(HARNESS))
def test_the_harness_never_imports_the_ui(name):
    """agent.run must stay runnable with no terminal attached."""
    path = SRC / name
    bad = [
        f"{name}:{line} imports {mod}"
        for mod, line in _imports(path)
        if mod.startswith(FORBIDDEN_FOR_HARNESS)
    ]
    assert not bad, (
        "the harness reached into the UI layer:\n  " + "\n  ".join(bad) +
        "\nReach the UI through the injected emit callback instead."
    )


@pytest.mark.parametrize(
    "path", _files(SRC / "tools"), ids=lambda p: f"tools/{p.name}"
)
def test_tools_never_import_the_ui(path):
    """A tool runs on a worker thread, often several at once. It has no screen."""
    bad = [
        f"{path.name}:{line} imports {mod}"
        for mod, line in _imports(path)
        if mod.startswith(FORBIDDEN_FOR_HARNESS)
    ]
    assert not bad, "a tool reached into the UI layer:\n  " + "\n  ".join(bad)


def test_only_client_talks_to_a_provider():
    """Every request funnels through client.py — that is what makes the one
    concurrency gate an actual bound, and what lets the provider be swapped."""
    bad = []
    for path in _files(SRC):
        if path.name in MAY_IMPORT_PROVIDER:
            continue
        for mod, line in _imports(path):
            if mod.split(".")[0] in {q.split(".")[0] for q in PROVIDER_PACKAGES}:
                bad.append(f"{path.relative_to(SRC)}:{line} imports {mod}")
    assert not bad, (
        "a module other than client.py talks to a provider:\n  " + "\n  ".join(bad) +
        "\nRoute it through client.py so it is bounded by the concurrency gate."
    )


def test_widgets_never_import_the_app():
    """A widget that reaches back into AhaCodeApp cannot be mounted anywhere else,
    and cannot be tested on its own. Widgets take what they need as arguments."""
    bad = []
    for path in _files(SRC / "widgets"):
        for mod, line in _imports(path):
            if mod.startswith(("ahacode.app", "ahacode.runner", "ahacode.plan_run",
                               "ahacode.session_ctl", "ahacode.turn_view")):
                bad.append(f"widgets/{path.name}:{line} imports {mod}")
    assert not bad, "a widget reached back into the app:\n  " + "\n  ".join(bad)


@pytest.mark.xfail(
    strict=True,
    reason="storage.py:384 imports ahacode.tools.plan from inside a function to "
           "dodge the cycle. The fix is to move the plan DATA MODEL out of the "
           "tool layer (plan_model.py); then drop this marker.",
)
def test_storage_does_not_import_the_tool_layer():
    """Persistence sits UNDER the tools, not beside them.

    A local import inside a function still IS the dependency — it only hides
    when the cycle would be noticed. Counting it is the point.
    """
    top_level = [
        f"storage.py:{line} imports {mod}"
        for mod, line in _imports(SRC / "storage.py")
        if mod.startswith("ahacode.tools")
    ]
    assert not top_level, (
        "storage imports the tool layer at load:\n  " + "\n  ".join(top_level)
    )
