"""The concurrency gate has to recover from leaked permits without supervision —
and without cutting off a backend that is merely slow."""

import threading
import time
from dataclasses import replace

import pytest

from ahacode import client, config
from ahacode.events import Notice, TextDelta


@pytest.fixture(autouse=True)
def gate_of(monkeypatch):
    """A two-permit gate over a fake endpoint, with the request timeout dialled
    down so the stuck threshold is reachable in a test."""
    monkeypatch.setattr(client, "_ensure_client",
                        lambda: (object(), replace(config.DEFAULTS, timeout=1.0)))
    monkeypatch.setattr(client, "GATE_STUCK_MARGIN", 1.0)
    monkeypatch.setattr(client, "_GATE_POLL", 0.05)
    config.save(replace(config.DEFAULTS, max_parallel_agents=2))
    client.reset()
    yield
    client.reset()


def _endless(_c, _k):
    for i in range(10_000):
        yield TextDelta(f"{i} ")


def _leak(n):
    """Start n turns and abandon them mid-stream, holding the generators open."""
    held = []
    for _ in range(n):
        gen = client.stream_chat([{"role": "user", "content": "x"}])
        next(gen)
        held.append(gen)
    return held


def test_a_leaked_gate_heals_itself(monkeypatch):
    monkeypatch.setattr(client, "_stream_with_budget_fallback", _endless)
    held = _leak(2)                                   # both permits gone
    assert client._ensure_gate()._value == 0

    events = []
    done = threading.Event()

    def next_request():
        for event in client.stream_chat([{"role": "user", "content": "y"}]):
            events.append(event)
            break
        done.set()

    threading.Thread(target=next_request, daemon=True).start()
    assert done.wait(timeout=10), "the request never got a permit — still deadlocked"
    assert any(isinstance(e, Notice) and "초기화" in e.text for e in events), \
        "it recovered silently; a rebuilt gate is a defect worth reporting"
    held.clear()


def test_a_real_queue_says_it_is_queueing(monkeypatch):
    """A normal wait and a hang look identical on screen, and that is exactly why a
    stuck app was hard to recognise. max_parallel_agents = 1 is the recommended
    setting for a single GPU, so every fan-out queues — it has to be legible."""
    monkeypatch.setattr(client, "_stream_with_budget_fallback", _endless)
    monkeypatch.setattr(client, "WAIT_NOTICE_AFTER", 0.2)
    held = _leak(2)                                   # both permits busy

    events = []
    started = threading.Event()

    def queued():
        for event in client.stream_chat([{"role": "user", "content": "y"}]):
            events.append(event)
            if isinstance(event, TextDelta):
                break
        started.set()

    threading.Thread(target=queued, daemon=True).start()
    assert started.wait(timeout=10)
    waiting = [e for e in events if isinstance(e, Notice) and "차례를 기다리는" in e.text]
    assert waiting, "queued in silence — indistinguishable from a hang"
    held.clear()


def test_a_permit_that_is_free_says_nothing(monkeypatch):
    """No bubble for a wait nobody noticed."""
    monkeypatch.setattr(client, "_stream_with_budget_fallback",
                        lambda c, k: iter([TextDelta("hi")]))
    events = list(client.stream_chat([{"role": "user", "content": "y"}]))
    assert not [e for e in events if isinstance(e, Notice)]


def test_a_busy_gate_is_left_alone(monkeypatch):
    """Permits changing hands means work is happening, however slow. The gate must
    not be rebuilt underneath it — that is what would cut off a cold model load."""
    monkeypatch.setattr(client, "_stream_with_budget_fallback", _endless)
    rebuilt = []
    real_reset = client._reset_gate
    monkeypatch.setattr(client, "_reset_gate",
                        lambda: (rebuilt.append(True), real_reset())[1])

    stop = threading.Event()

    def churn():                                       # a queue that keeps moving
        while not stop.is_set():
            list(zip(range(3), client.stream_chat([{"role": "user", "content": "z"}])))

    workers = [threading.Thread(target=churn, daemon=True) for _ in range(3)]
    for w in workers:
        w.start()
    time.sleep(3.0)                                    # well past timeout + margin
    stop.set()
    for w in workers:
        w.join(timeout=5)
    assert not rebuilt, "a moving queue was mistaken for a deadlock"
