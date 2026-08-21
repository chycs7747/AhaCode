<div align="center">

<img src="assets/logo.svg" alt="AhaCode" width="440" />

**A terminal-native chat client for local LLMs, built with [Textual](https://github.com/Textualize/textual).**

streaming chat · visible thinking · multi-turn memory · persistent sessions · in-chat commands

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Built with Textual](https://img.shields.io/badge/built%20with-Textual-5967ff.svg)](https://github.com/Textualize/textual)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

[Features](#features) · [Getting started](#getting-started) · [Commands](#commands) · [Sessions](#sessions-on-disk) · [Architecture](#architecture) · [Development](#development)

</div>

---

> ⚠️ **Alpha.** AhaCode is under active development — expect rough edges and
> breaking changes between versions.

AhaCode talks to any OpenAI-compatible endpoint (vLLM, Ollama, gateways, …) and
renders the conversation in your terminal — with the model's reasoning streamed
live into a separate, dimmed bubble as it thinks.

## Features

- **Token-by-token streaming** in a responsive TUI — typing stays smooth while
  the model answers, and a new message cancels the previous stream.
- **Visible thinking** — reasoning deltas stream into their own dimmed bubble,
  which only appears for models that actually think.
- **Multi-turn memory** — the full conversation history rides along with every
  request.
- **Persistent sessions** — every message is appended to a plain-text JSONL
  file under `sessions/` in the project root; the latest session is restored
  on startup. Your history is always `cat`-able.
- **Model picker & status bar** — the bar under the prompt shows the current
  endpoint and a dropdown of models fetched from the server's `/v1/models`;
  pick one and it persists. `/model` and `/url` do the same from the keyboard.
- **Smart auto-scroll** — follows the stream only while you are at the bottom;
  scrolling up to read is never interrupted.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- An OpenAI-compatible LLM server (developed against a local
  [vLLM](https://github.com/vllm-project/vllm) instance)

## Getting started

```bash
git clone https://github.com/chycs7747/AhaCode && cd AhaCode
uv sync
uv run textual run ahacode.app
```

Type a message and press <kbd>Enter</kbd>. Quit with <kbd>Ctrl+Q</kbd>.

The first run creates `config.toml` in the project root — edit it to point at
your server:

```toml
[model]
base_url = "http://127.0.0.1:8078/v1"
name = "qwen38-nvfp4"
api_key = "EMPTY"   # many local servers ignore this, but the SDK requires one
timeout = 60.0
```

…or configure it without leaving the chat — see [Commands](#commands).

## Commands

Messages starting with `/` are handled locally — they never reach the model
and are not recorded in your session.

| Command | Effect |
|---|---|
| `/model` | Show the current model and endpoint |
| `/model <name>` | Switch model (persisted to `config.toml`; the dropdown under the prompt does the same) |
| `/url <base_url>` | Switch endpoint (persisted to `config.toml`) |
| `/help` | List available commands |

## Sessions on disk

One session per file, one message per line, append-only:

```bash
ls sessions/
cat sessions/2026-08-18_212512.jsonl
tail -f sessions/*.jsonl   # watch messages land in real time
```

Both `config.toml` and `sessions/` are git-ignored — your key and your
conversations never end up in a commit.

## Architecture

```
ahacode/
├── app.py            # Textual App: layout, event wiring, streaming worker
├── client.py         # LLM I/O — the only module that talks to a provider;
│                     # emits unified ("thinking" | "text", fragment) deltas
├── config.py         # config.toml in the project root (endpoint, model, key)
├── session.py        # conversation state (plain Python, widget-free)
├── storage.py        # JSONL persistence (./sessions/)
├── ahacode.tcss      # styles (no inline CSS)
└── widgets/
    ├── chatbox.py    # a single chat bubble
    └── model_bar.py  # endpoint display + model dropdown (fed by /v1/models)
```

Design rules the codebase sticks to:

- **All LLM traffic goes through `client.py`.** The UI only ever sees unified
  `(kind, fragment)` deltas, so providers can be swapped without touching
  widgets. Provider quirks (e.g. which key thinking deltas arrive under) are
  absorbed there.
- **Session state is a plain Python object** — testable without a terminal,
  serializable without ceremony.
- **UI is only touched from the main thread.** The streaming worker hands
  fragments over via `call_from_thread` (built-in backpressure) and announces
  completion with a message, never by mutating shared state.

## Development

```bash
uv run pytest -v            # unit + headless TUI tests (Textual Pilot)
uv run textual console      # live logs in a second terminal (run app with --dev)
```

Tests replace the LLM with an offline fake via `monkeypatch`, so the suite
runs without a server.

## Acknowledgments

AhaCode's design borrows ideas from some excellent open source projects:

- [Textual](https://github.com/Textualize/textual) — the TUI framework
- [elia](https://github.com/darrenburns/elia) — TUI chat architecture:
  worker-based streaming, pre-mounted bubbles, smart auto-scroll
- [Roo Code](https://github.com/RooCodeInc/Roo-Code) &
  [Kilo Code](https://github.com/Kilo-Org/kilocode) — agent-loop and
  provider-abstraction patterns

## License

[MIT](LICENSE)
