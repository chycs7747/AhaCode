"""System prompts, assembled from layers.

One base per mode plus optional addenda, composed in a fixed order and flattened
to a single module for our scale. Callers use the FUNCTIONS (act_system,
plan_system, …) — never the raw constants — so the internal composition can grow
(per-model deltas, env injection, sub-agent roles) without touching call sites.

Style is qwen-informed and *measured* (2026-08-24, gateway qwen38):
- No few-shot examples — they anchor a reasoning model's output; length is scaled
  by a declarative rule instead (A/B: 0 preamble, ~10x short↔conceptual scaling).
- Delegation is kept OUT of the always-on prompt — qwen never self-delegates
  (0/120), and pushing it only biases atomic tasks; the `task` tool's own
  description holds the (conservative) when-to guidance instead.
"""

from __future__ import annotations

import platform

from ahacode import config, storage

# --- raw layers -----------------------------------------------------------

# The base act-mode prompt: staged sections + terse bullets. "coding agent … also
# answer questions" keeps general Q&A un-muzzled (validated on qwen).
ACT_SYSTEM = """You are AhaCode, a TUI coding agent working in this project. Most tasks are software work with the available tools; you also answer questions directly.

# Output
- Terminal Markdown. Reply in the user's language. Code references as `path:line`.
- Length follows the task — one line for a lookup; a full explanation when the question is conceptual or the user asks for depth.
- No preamble or postamble.

# Editing code
- Read first; mimic the file's language, libraries, and style. Don't assume a library exists.
- Make the smallest correct edit. Verify with the project's own tests; report failures plainly.

# Never (IMPORTANT)
- Print, log, or commit secrets; `config.toml` and `sessions/` are private.
- Do anything irreversible (`git push`, force-push, deleting data) without an explicit go-ahead."""

# Plan mode: read-only, produce a plan rather than act.
PLAN_SYSTEM = (
    "You are in PLAN MODE. Do not change anything or run commands. If needed, "
    "investigate with the read tool, then call todo_write to lay out a clear, "
    "step-by-step plan for the user to review. Do not carry out the plan."
)

# A worker sub-agent's framing. Deliberately short: it inherits the same tools, so
# it only needs to know its job is one delegated task and to end with a
# self-contained result. Kept lean on purpose — when many sub-agents share this
# prefix the gateway's prefix cache reuses the prefill (~67% saving, measured).
SUBAGENT_SYSTEM = (
    "You are a focused sub-agent spawned to complete ONE delegated task. "
    "Work autonomously with the tools available, then finish with a concise, "
    "self-contained result the caller can use directly — no filler, no questions."
)

# Injected as a user turn when the agent loop hits its turn cap. The wrap-up turn is
# sent with NO tools (so the model physically cannot call one and must answer), and
# this primes a useful close instead of a bare truncation.
MAX_TURNS_PROMPT = (
    "You've reached the step limit for this task and tools are no longer available. "
    "Give your best final answer now, as text only: briefly summarize what you "
    "accomplished, what remains unfinished, and the recommended next step."
)

TITLE_SYSTEM = (
    "You write a very short title (2-5 words) for a conversation. "
    "Reply with ONLY the title — no quotes, no trailing punctuation."
)

# --- extension points (empty today; filled when a 2nd model family / role lands) ---
# Different model families want different deltas (e.g. examples-on for Claude,
# delegation emphasis where the model is trained to calibrate it). One model today
# (qwen), so these stay empty; a future entry, e.g.
# BY_MODEL["claude"] = {"examples": …, "delegation": …}, layers in through
# act_system() with no call-site change.
BY_MODEL: dict[str, dict[str, str]] = {}
ROLE_ADDENDA: dict[str, str] = {}  # e.g. "debug": "Reflect on 5-7 sources before fixing…"


# --- assembly -------------------------------------------------------------

def _family(model: str | None = None) -> str:
    """Map a concrete model id to a prompt family. Returns 'qwen' today; the other
    branches are the seam for BYOK Claude/GPT prompts later."""
    name = (model or config.load().name).lower()
    if "claude" in name:
        return "claude"
    if "gpt" in name or "codex" in name:
        return "gpt"
    return "qwen"


def environment_block(model: str | None = None) -> str:
    """The live facts the model needs to emit valid commands: real OS/shell/cwd and
    the active model. Rebuilt each call so a /model switch or a different cwd is
    always reflected."""
    cfg = config.load()
    return (
        "# Environment\n"
        f"- OS: {platform.system()} · shell: bash · cwd: {storage.PROJECT_ROOT}\n"
        f"- model: {model or cfg.name}"
    )


def act_system(model: str | None = None) -> str:
    """Full act-mode system prompt: base + any per-model deltas + live environment.
    BY_MODEL is empty today so this is ACT_SYSTEM + environment; a family entry
    (examples-on for Claude, delegation emphasis, …) layers in here transparently."""
    parts = [ACT_SYSTEM]
    profile = BY_MODEL.get(_family(model), {})
    if profile.get("examples"):
        parts.append(profile["examples"])
    if profile.get("delegation"):
        parts.append(profile["delegation"])
    parts.append(environment_block(model))
    return "\n\n".join(parts)


def plan_system(model: str | None = None) -> str:
    """Plan mode prompt. (model param mirrors act_system so per-model tuning slots in.)"""
    return PLAN_SYSTEM


def subagent_system(role: str | None = None) -> str:
    """A worker sub-agent's framing, plus an optional role addendum (debug/explore…).
    ROLE_ADDENDA is empty today; a role slots in with no change to subagent.run,
    which already accepts a `system=` argument."""
    addendum = ROLE_ADDENDA.get(role or "")
    return f"{SUBAGENT_SYSTEM}\n\n{addendum}" if addendum else SUBAGENT_SYSTEM


def title_system() -> str:
    return TITLE_SYSTEM


def max_turns_prompt() -> str:
    """The user turn injected to force a tool-free wrap-up when the loop hits its cap."""
    return MAX_TURNS_PROMPT
