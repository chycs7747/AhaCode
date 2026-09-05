"""Every prompt the harness sends, one module per moment it is sent.

system: the first message of every request, assembled per mode. injected: user
turns the harness writes into the conversation on the user's behalf. side: the
system prompts of requests made outside the conversation.
"""

from __future__ import annotations

# Imported here so `from ahacode import prompts` reaches all three as attributes.
from ahacode.prompts import injected, side, system
