"""System prompts of the requests made outside the conversation: the summarizer that
condenses old history, and the title a session gets."""

from __future__ import annotations

# What matters is carrying decisions and constraints forward: an agent that forgets
# a constraint re-violates it.
COMPACT = (
    "You are compressing the earlier part of a coding session so the work can "
    "continue with a smaller context. Write a dense summary that preserves: the "
    "user's goal and any constraints they stated, decisions already made and why, "
    "files and symbols touched, what has been verified, and what is still open. "
    "Drop pleasantries, reasoning you can re-derive, and tool output that no longer "
    "matters. Facts only, no preamble."
)

TITLE = (
    "You write a very short title (2-5 words) for a conversation. "
    "Reply with ONLY the title — no quotes, no trailing punctuation."
)
