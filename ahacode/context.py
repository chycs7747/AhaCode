"""Context-window management: prune old tool output, then summarize the oldest
stretch, before a request outgrows the window. Pure apart from the default
summarizer, so the boundary logic is unit-testable with plain dicts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ahacode import client, config, prompts
from ahacode.text import elide

SummarizeFn = Callable[[list[dict]], str]  # (older messages) -> a condensed summary

# What replaces a pruned tool result. The message stays and only its content goes,
# so an assistant/tool pairing can never be broken by pruning.
PRUNED_STUB = "[older tool output dropped to free context — re-run the tool if needed]"

PRUNE_PROTECT_CHARS = 40_000  # the newest tool output, kept verbatim
PRUNE_MIN_GAIN_CHARS = 8_000  # below this, pruning is churn rather than relief


@dataclass
class Compaction:
    """What a compaction pass did; pruning is free, summarizing costs a model call."""

    summarized: int = 0  # messages replaced by one summary
    pruned_chars: int = 0  # characters of old tool output blanked out

    def __bool__(self) -> bool:
        return bool(self.summarized or self.pruned_chars)


# The transcript handed to the summarizer: each message elided to a per-message
# share of a budget that scales with the window, so the whole stretch is covered
# rather than only its oldest end.
_MAX_MESSAGE_CHARS = 1200
_MIN_MESSAGE_CHARS = 200
_TRANSCRIPT_WINDOW_SHARE = 0.35
_CHARS_PER_TOKEN = 3  # matches estimate_tokens
_MIN_TRANSCRIPT_CHARS = 24_000
_MAX_TRANSCRIPT_CHARS = 300_000  # bounds the summarizing call's own prefill


def transcript_budget(cfg: config.ModelConfig | None = None) -> int:
    """How much of the condensed stretch the summarizer may be shown, scaled to
    the window.

    Args:
        cfg: The config; loaded when omitted.

    Returns:
        A character budget.
    """
    cfg = cfg or config.load()
    share = int(cfg.context_window * _CHARS_PER_TOKEN * _TRANSCRIPT_WINDOW_SHARE)
    return max(_MIN_TRANSCRIPT_CHARS, min(_MAX_TRANSCRIPT_CHARS, share))


def estimate_tokens(messages: list[dict]) -> int:
    """A rough token count, used only until the server reports a real one.

    Pessimistic at ~3 characters per token: Korean runs closer to 1-1.5, and
    over-counting merely condenses a little early.

    Args:
        messages: The history.

    Returns:
        The estimate.
    """
    chars = 0
    for msg in messages:
        chars += len(str(msg.get("content") or ""))
        for call in msg.get("tool_calls") or []:
            chars += len(str(call))
    return chars // 3


def find_split(messages: list[dict], keep_recent: int) -> int:
    """The index to condense up to: messages[head:split] is summarized.

    Only a `user` message is a legal boundary: cutting elsewhere can separate a
    `tool` message from the `assistant` entry that introduced it, which an
    OpenAI-compatible server rejects.

    Args:
        messages: The history.
        keep_recent: The newest messages that must stay verbatim.

    Returns:
        The split index, or 0 when there is no legal boundary.
    """
    head = 1 if messages and messages[0].get("role") == "system" else 0
    candidate = len(messages) - keep_recent
    for i in range(min(candidate, len(messages) - 1), head, -1):  # newest boundary wins
        if messages[i].get("role") == "user":
            return i
    return 0


def _render_one(msg: dict, per_message: int) -> str:
    """One message as a transcript line, elided to `per_message` characters."""
    body = str(msg.get("content") or "")
    for call in msg.get("tool_calls") or []:
        fn = call.get("function", {})
        body += f"\n[called {fn.get('name')} {fn.get('arguments', '')}]"
    return f"{msg.get('role')}: {elide(body, per_message)}"


def render_transcript(messages: list[dict], budget: int | None = None) -> str:
    """Flatten messages into the plain transcript handed to the summarizer.

    The budget is shared out per message rather than spent from the oldest end, so
    the summary covers the whole stretch. If it still does not fit, both ends are
    kept and the middle is dropped with a note.

    Args:
        messages: The stretch being condensed.
        budget: A character budget; transcript_budget() when omitted.

    Returns:
        The transcript text.
    """
    if not messages:
        return ""
    budget = transcript_budget() if budget is None else budget
    per = min(_MAX_MESSAGE_CHARS, max(_MIN_MESSAGE_CHARS, budget // len(messages)))
    lines = [_render_one(m, per) for m in messages]
    if sum(len(line) for line in lines) <= budget:
        return "\n\n".join(lines)

    half = budget // 2
    head, used = [], 0
    for line in lines:
        if used + len(line) > half:
            break
        head.append(line)
        used += len(line)
    tail, used = [], 0
    for line in reversed(lines[len(head):]):
        if used + len(line) > half:
            break
        tail.append(line)
        used += len(line)
    tail.reverse()
    dropped = len(lines) - len(head) - len(tail)
    if not dropped:
        return "\n\n".join([*head, *tail])
    return "\n\n".join([*head, f"…[{dropped} messages omitted from the middle]…", *tail])


def llm_summarize(messages: list[dict]) -> str:
    """The default summarizer: one non-streaming model call.

    Args:
        messages: The stretch being condensed.

    Returns:
        The summary text.
    """
    return client.complete([
        {"role": "system", "content": prompts.side.COMPACT},
        {"role": "user", "content": render_transcript(messages)},
    ])


def prune_tool_output(messages: list[dict], cfg: config.ModelConfig | None = None) -> int:
    """Blank the content of the oldest tool results in place.

    The cheap half of compaction, and the only one that works on a sub-agent's
    history: that has a single user message (the task), so there is never a legal
    boundary to summarize at. All-or-nothing below a floor.

    Args:
        messages: The history, mutated in place.
        cfg: The config; loaded when omitted.

    Returns:
        Characters freed; 0 when nothing was pruned.
    """
    cfg = cfg or config.load()
    kept = 0
    victims: list[dict] = []
    for msg in reversed(messages):
        if msg.get("role") != "tool":
            continue
        content = msg.get("content") or ""
        if content == PRUNED_STUB:
            continue
        if kept + len(content) <= PRUNE_PROTECT_CHARS:
            kept += len(content)
            continue
        victims.append(msg)
    gain = sum(len(m.get("content") or "") for m in victims)
    if gain < PRUNE_MIN_GAIN_CHARS:
        return 0
    for msg in victims:
        msg["content"] = PRUNED_STUB
    return gain


def maybe_compact(
    messages: list[dict],
    prompt_tokens: int | None = None,
    *,
    summarize: SummarizeFn | None = None,
    cfg: config.ModelConfig | None = None,
) -> Compaction:
    """Shrink `messages` in place when it is close to the window.

    Cheapest first: prune old tool output; only if that frees nothing, replace the
    oldest stretch with one summary.

    Args:
        messages: The history, mutated in place.
        prompt_tokens: The server's count for the previous request; estimated when None.
        summarize: The summarizer; llm_summarize when omitted.
        cfg: The config; loaded when omitted.

    Returns:
        What was done; falsy when nothing was.
    """
    cfg = cfg or config.load()
    done = Compaction()
    if not cfg.context_window:
        return done
    used = prompt_tokens or estimate_tokens(messages)
    if used < cfg.context_window * cfg.compact_threshold:
        return done

    done.pruned_chars = prune_tool_output(messages, cfg)
    if done.pruned_chars:
        return done

    split = find_split(messages, cfg.keep_recent_messages)
    if split <= 0:
        return done
    head = 1 if messages[0].get("role") == "system" else 0
    older = messages[head:split]
    summary = (summarize or llm_summarize)(older)
    if not summary.strip():
        return done  # a failed summary must not silently delete the history

    messages[head:split] = [{"role": "user", "content": prompts.injected.SUMMARY_PREFIX + summary}]
    done.summarized = len(older) - 1  # the summary itself takes one slot back
    return done
