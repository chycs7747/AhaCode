"""Context-window management: condense an over-long conversation before it is sent.

A chat history only grows. Left alone it eventually exceeds the model's window and
the server rejects the whole request — or, worse, silently drops the front of it.
The fix here is to *summarize* the oldest stretch and put one condensed message in
its place, rather than dropping messages outright: an agent that forgets a decision
it already made will happily make the opposite one on the next turn.

Pure and network-free apart from the default summarizer, so the boundary logic (the
part that is easy to get wrong) is unit-testable with plain dicts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ahacode import client, config, prompts
from ahacode.text import elide

# (older messages) -> a condensed summary of them. Injected so tests stay offline.
SummarizeFn = Callable[[list[dict]], str]

SUMMARY_PREFIX = "# Condensed summary of the earlier conversation\n\n"

# What replaces a pruned tool result. The MESSAGE stays — only its content goes —
# which is the whole point: an assistant/tool pairing can never be broken by this.
PRUNED_STUB = "[older tool output dropped to free context — re-run the tool if needed]"

# Pruning protects the newest tool output and only touches what is older, because
# the recent results are the ones the next turn is actually working from.
PRUNE_PROTECT_CHARS = 40_000
# Below this there is nothing worth reclaiming; leave the transcript alone.
PRUNE_MIN_GAIN_CHARS = 8_000


@dataclass
class Compaction:
    """What a compaction pass actually did. Two mechanisms, reported apart because
    they cost very differently: pruning is free, summarizing is a model call."""

    summarized: int = 0     # messages replaced by one summary
    pruned_chars: int = 0   # characters of old tool output blanked out

    def __bool__(self) -> bool:
        return bool(self.summarized or self.pruned_chars)

# Caps for the text handed to the summarizer — the stretch being condensed is by
# definition large, so summarizing it verbatim would hit the same wall it is meant
# to avoid. Long messages are elided in the middle, keeping both ends.
_MAX_MESSAGE_CHARS = 1200
_MAX_TRANSCRIPT_CHARS = 24_000


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token count, used only until the server reports a real one.

    Deliberately pessimistic (~3 chars/token): a token is ~4 chars of English but
    closer to 1-1.5 for Korean, so a generous divisor would under-count a Korean
    conversation and let the request blow the window. Over-counting merely condenses
    a little early, which is the cheaper mistake. Every turn after the first uses the
    server's own `usage.prompt_tokens` instead.
    """
    chars = 0
    for msg in messages:
        chars += len(str(msg.get("content") or ""))
        for call in msg.get("tool_calls") or []:
            chars += len(str(call))
    return chars // 3


def find_split(messages: list[dict], keep_recent: int) -> int:
    """The index to condense up to: messages[head:split] is summarized, the rest kept.

    Only a `user` message is a legal boundary. Cutting anywhere else can separate a
    `tool` message from the `assistant` entry whose tool_calls introduced it, and an
    OpenAI-compatible server rejects the entire request when that pairing is broken —
    the classic trap in naive history trimming. A `user` turn can never sit between
    the two, so it is always safe.

    Returns 0 when there is no legal boundary (nothing to condense yet).
    """
    head = 1 if messages and messages[0].get("role") == "system" else 0
    candidate = len(messages) - keep_recent
    # Walk backwards from the candidate so the newest legal boundary wins — that
    # condenses as little as the threshold allows, keeping the most detail.
    for i in range(min(candidate, len(messages) - 1), head, -1):
        if messages[i].get("role") == "user":
            return i
    return 0


def render_transcript(messages: list[dict]) -> str:
    """Flatten messages into the plain transcript handed to the summarizer."""
    parts: list[str] = []
    total = 0
    for msg in messages:
        body = str(msg.get("content") or "")
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            body += f"\n[called {fn.get('name')} {fn.get('arguments', '')}]"
        body = elide(body, _MAX_MESSAGE_CHARS)
        line = f"{msg.get('role')}: {body}"
        if total + len(line) > _MAX_TRANSCRIPT_CHARS:
            parts.append("…[earlier messages omitted]…")
            break
        parts.append(line)
        total += len(line)
    return "\n\n".join(parts)


def llm_summarize(messages: list[dict]) -> str:
    """Default summarizer — one non-streaming call, like the session auto-titler."""
    return client.complete([
        {"role": "system", "content": prompts.compact_system()},
        {"role": "user", "content": render_transcript(messages)},
    ])


def prune_tool_output(messages: list[dict], cfg: config.ModelConfig | None = None) -> int:
    """Blank the content of the OLDEST tool results, newest-first. Returns chars freed.

    This is the cheap half of context management, and it reaches a case summarizing
    cannot. Summarizing has to cut somewhere, and the only safe cut is a `user`
    message — but a sub-agent's history is [system, task, assistant, tool, assistant,
    tool, ...] with exactly one user message, which is its task and must survive. So
    find_split always returns 0 there and a sub-agent could never be compacted at all,
    however much tool output it piled up.

    Pruning has no such problem: the tool MESSAGE stays and only its content is
    replaced, so the assistant/tool pairing the server checks is untouched, and no
    boundary is needed. It also costs no model call.

    All-or-nothing below a floor: blanking a few hundred characters is churn, not
    relief.
    """
    cfg = cfg or config.load()
    kept = 0
    victims: list[dict] = []
    for msg in reversed(messages):
        if msg.get("role") != "tool":
            continue
        content = msg.get("content") or ""
        if content == PRUNED_STUB:  # already pruned — idempotent
            continue
        if kept + len(content) <= PRUNE_PROTECT_CHARS:
            kept += len(content)   # recent enough to keep verbatim
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
    """Shrink `messages` IN PLACE when it is close to the window.

    Two mechanisms, cheapest first:

    1. Prune — blank the oldest tool results. No model call, cannot break the
       assistant/tool pairing, and needs no cut point, so it works on a sub-agent's
       history where no legal one exists. If it frees enough, we stop here.
    2. Summarize — replace the oldest stretch with one condensed message. Costs a
       request, and only possible where there is a `user` boundary to cut on.

    `prompt_tokens` is the server's own count for the previous request — the most
    accurate signal available, and free (it already arrives in the stream's usage
    trailer). Only the first turn of a fresh process falls back to the estimate.
    """
    cfg = cfg or config.load()
    done = Compaction()
    if not cfg.context_window:  # 0 disables the whole mechanism
        return done
    used = prompt_tokens or estimate_tokens(messages)
    if used < cfg.context_window * cfg.compact_threshold:
        return done

    # Cheap first: old tool output is usually where the bulk actually is, and
    # dropping it costs nothing.
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

    messages[head:split] = [{"role": "user", "content": SUMMARY_PREFIX + summary}]
    done.summarized = len(older) - 1  # the summary itself takes one slot back
    return done
