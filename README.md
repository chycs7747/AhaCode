<div align="center">

<img src="assets/logo.svg" alt="AhaCode" width="440" />

**A terminal-native coding agent for local LLMs, built with [Textual](https://github.com/Textualize/textual).**

streaming chat · visible thinking · tool-using agent loop · plan mode · sub-agents · persistent sessions

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Built with Textual](https://img.shields.io/badge/built%20with-Textual-5967ff.svg)](https://github.com/Textualize/textual)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

[Features](#features) · [Getting started](#getting-started) · [Commands](#commands) · [Configuration](#configuration) · [Sessions](#sessions-on-disk) · [Architecture](#architecture) · [Development](#development)

</div>

---

> ⚠️ **Alpha.** AhaCode is under active development — expect rough edges and
> breaking changes between versions.

AhaCode talks to any OpenAI-compatible endpoint (vLLM, Ollama, gateways, …) and
renders the conversation in your terminal — with the model's reasoning streamed
live into a separate, dimmed bubble as it thinks, and its tool calls rendered as
foldable cards you approve before they run.

## Features

### Chat

- **Token-by-token streaming** in a responsive TUI — typing stays smooth while
  the model answers, and a new message cancels the previous stream.
- **Visible thinking** — reasoning deltas stream into their own foldable, dimmed
  block, which only appears for models that actually think and auto-collapses
  once the answer begins.
- **Multi-turn memory** — the full conversation history rides along with every
  request.
- **Markdown answers** — fenced code blocks are syntax-highlighted in place.
- **Smart auto-scroll** — follows the stream only while you are at the bottom;
  scrolling up to read is never interrupted.

### Agent

- **Tool-using agent loop** over native function calling: the model calls a tool,
  the result is fed back, and the loop runs until a turn arrives with no tool
  calls — with a turn cap that forces a tool-free wrap-up instead of truncating
  mid-task.

  | Tool | What it does | Asks first? |
  |---|---|---|
  | `read` | Read a file (paged with offset/limit) | — |
  | `glob` | Find files by path pattern, newest first | — |
  | `grep` | Search contents by regex → `path:line:text` | — |
  | `write` | Create or overwrite a file | ✅ |
  | `edit` | Replace a unique snippet, shown as a `-`/`+` diff | ✅ |
  | `bash` | Run a shell command in the project root | ✅ |
  | `todo_write` | Keep a working checklist in the pinned panel | — |
  | `plan_submit` | Plan mode only: submit the finished plan for approval | — |
  | `task` | Delegate a subtask to a fresh sub-agent | ✅ |

- **Approval with a real preview** — side-effecting calls open a modal showing
  *what will happen*: the file content as highlighted code, the edit as a diff,
  the command as a command. Auto-approve is one toggle away for a trusted run.
- **A denylist below the approval** — catastrophic commands (`rm -rf /`, fork
  bombs, `mkfs`, `dd of=/dev/…`) are hard-blocked before they can even be
  offered, and each half of a chained command is checked separately. Defense in
  depth, not a guarantee — the human prompt is still the real safeguard.
- **Plan mode** — a read-only mode (`read`/`glob`/`grep` + `plan_submit`) where
  the model investigates, settles questions with you, and ends its turn by
  *submitting* the plan. The harness writes it to `plans/<session>.md` and fills
  the pinned checklist from it; an empty plan is refused, and steps that do not
  read as executable come back noted for the model to reconsider.
- **A plan gate** — `plan_submit` is the trigger: the loop stops *between turns*
  with the plan on screen and asks ▶ 실행 or ✎ 수정. No heuristics about step
  counts — the model said it is done, and the pause is a harness decision, not a
  prompt asking nicely. Choosing 수정 keeps you in plan mode; your next message
  revises the plan and it is submitted again.
- **Approval hands the plan to a child session** — ▶ (or an empty Enter) opens
  a new session parented to the planning one, in act mode, seeded with a single
  message naming the plan file. One continuous context reads the plan, mirrors
  it into the checklist, and works it step by step — no per-step sub-agents,
  nothing summarised and threaded forward by hand. Stop it and it is a session
  like any other: open it and carry on. Approving a revised plan makes a new
  sibling, never a deeper child. The plan file itself is never edited by the
  run; after every turn the harness snapshots the checklist and the latest
  summary to `plans/<session>.result.md` beside it, and says on screen whether
  steps are still owed or the plan is complete.
- **Sub-agents** — `task` spawns a child agent that renders into a nested,
  foldable 🤖 card and gets its own linked session file. Nesting depth is capped
  by config, a fan-out of tasks runs concurrently, and one process-wide gate
  bounds total requests so a single-GPU backend is never oversubscribed.
- **Thinking controls** — a per-turn reasoning budget (`/think`), and reasoning
  switched off on turns that only need to act on a tool result, which is what
  stops a multi-turn loop from spiralling.
- **Context management, cheapest first** — when a request approaches the model's
  window, old *tool output* is blanked first: it costs no model call, and because
  only the content goes (the message stays) it can never separate a `tool` message
  from the `assistant` tool_calls entry that introduced it. Only if that is not
  enough is the oldest stretch replaced by a summary — which can only cut on a
  user turn, since anywhere else breaks that same pairing and an
  OpenAI-compatible server rejects the whole request. Either way you get a note in
  the chat, and the trigger is the server's *own* `usage.prompt_tokens`, not a
  guessed token count. Your session file keeps the full transcript; only the
  request in flight is shortened.

  The two-layer order matters for sub-agents in particular: their history is
  `[system, task, assistant, tool, …]` with exactly one user message — the task,
  which must survive — so there is no legal cut point and summarizing alone could
  never compact them at all.
- **Big output spills to a file** — `bash` is the one tool whose output size
  nobody can predict, so anything past a few KB is written whole to
  `sessions/<id>-out/` and only a header plus a both-ends preview comes back.
  Nothing is lost and the way back already exists: the spilled file is an
  ordinary text file, so `read` pages through it and `grep` searches it.

### Sessions & config

- **Persistent sessions** — every message is appended to a plain-text JSONL
  file under `.ahacode/sessions/` in the project you launched in; the latest
  session is restored on startup. Your history is always `cat`-able.
- **A session tree** — sub-agent transcripts are linked to the session that
  spawned them, and the picker shows the whole tree. Machine-authored child
  sessions open read-only, so you can read one without derailing it.
- **Model picker & status bar** — the bar under the prompt holds the mode,
  model, and auto-approve controls plus a live token/throughput readout; the
  model list is fetched from the server's `/v1/models`.
- **A settings screen** — ⚙ Settings opens every `config.toml` field worth changing,
  on four tabs down the left edge: 연결 (endpoint, key, model, timeout), 에이전트,
  컨텍스트, 사고. The 연결 tab asks *the address in the box* for its `/v1/models`,
  so the endpoint and the model name cannot drift apart — a model name only means
  anything on the server that serves it. Listing is a plain GET and loads nothing.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- An OpenAI-compatible LLM server (developed against a local
  [vLLM](https://github.com/vllm-project/vllm) instance)
- On Windows, [Git for Windows](https://git-scm.com/download/win) — the `bash`
  tool runs commands through its bash. Without it AhaCode falls back to `cmd`
  and says so in the model's prompt rather than sending it bash it cannot run.

## Getting started

Install it once; run it inside whatever project you want to work on.

```bash
uv tool install git+https://github.com/chycs7747/AhaCode

cd ~/some/project
ahacode
```

**The directory you launch from is the workspace.** `read`, `glob`, `grep` and
`bash` all resolve against it, and everything the run generates lands in that
project's `.ahacode/` — so add `.ahacode/` to its `.gitignore`. (`AHACODE_ROOT`
overrides the choice if you would rather not cd.)

To try it without installing anything permanent:

```bash
uvx --from git+https://github.com/chycs7747/AhaCode ahacode
```

To work on AhaCode itself, see [Development](#development).

Type a message and press <kbd>Enter</kbd> — see [Keys](#keys) for the rest.

The first run writes `~/.ahacode/config.toml` pointing at `localhost:8888`. Point
it at your own server with `/url` — Ollama serves `localhost:11434`, vLLM
`localhost:8000` — or edit the file; see [Configuration](#configuration).

## Commands

Messages starting with `/` are handled locally — they never reach the model
and are not recorded in your session.

| Command | Effect |
|---|---|
| `/model` | Show the current model and endpoint |
| `/model <name>` | Switch model (persisted; the dropdown under the prompt does the same) |
| `/url <base_url>` | Switch endpoint (persisted) |
| `/think <n>` \| `/think off` | Per-turn reasoning budget in tokens |
| `/new` | Start a new session |
| `/sessions` | Open the session tree and switch |
| `/help` | List available commands |

### Keys

| Key | Action |
|---|---|
| <kbd>Enter</kbd> | Send |
| <kbd>Shift</kbd>+<kbd>Enter</kbd> / <kbd>Alt</kbd>+<kbd>Enter</kbd> / <kbd>Ctrl</kbd>+<kbd>J</kbd> | Newline |
| <kbd>Esc</kbd> | Stop the turn in flight |
| <kbd>Ctrl</kbd>+<kbd>Y</kbd> | Copy the last answer (OSC 52 — works over SSH) |
| <kbd>y</kbd> / <kbd>n</kbd> | Approve / skip, in the approval modal |
| <kbd>Ctrl</kbd>+<kbd>D</kbd> | Quit |

> <kbd>Shift</kbd>+<kbd>Enter</kbd> needs a terminal that speaks the Kitty keyboard
> protocol (kitty, WezTerm, Ghostty, …) — historically a terminal sends the *same*
> bytes for <kbd>Enter</kbd> and <kbd>Shift</kbd>+<kbd>Enter</kbd>, so it cannot tell
> them apart. <kbd>Alt</kbd>+<kbd>Enter</kbd> and <kbd>Ctrl</kbd>+<kbd>J</kbd> work
> everywhere.

## Configuration

Settings live in two layers. Which server to talk to is a fact about your machine,
not about any one project, so it lives once in `~/.ahacode/config.toml` — written
with commented defaults on first run, and the file `/url` and `/model` update. A
project that wants something different can drop its own `.ahacode/config.toml`
beside its sessions; it is merged **key by key** over the global one, so overriding
the model does not mean restating the endpoint. There is no project file unless you
write one.

```toml
[model]
base_url = "http://localhost:8888/v1"    # Ollama is :11434 · vLLM is :8000
name = "qwen3.8-flash-next"
api_key = "EMPTY"            # many local servers ignore this, but the SDK requires one
timeout = 60.0               # seconds; caps how long a read may block between chunks
thinking_token_budget = 4096 # per-turn reasoning cap; 0 = unbounded
reasoning_effort = "medium"  # low|medium|high|xhigh — a hint; effect is server-dependent
no_think_after_tools = true  # skip reasoning on turns that only act on a tool result

context_window = 32768       # your model's window, in tokens; 0 disables compaction

[agent]
subagent_depth = 1        # generations of sub-agents that may nest (0 = none)
max_parallel_agents = 8   # cap on concurrent requests (1 = serialise sub-agents)
# Every field on this page is editable live from the ⚙ Settings button — set
# max_parallel_agents to 1 to keep a single GPU from being double-loaded, and give
# plan a bigger thinking budget than impl/subagent.
compact_threshold = 0.8   # condense once a request reaches this fraction of the window
keep_recent_messages = 6  # newest messages always kept verbatim
```

Both reasoning knobs are vendor extensions, so they are *hints*: a server without
a reasoning config simply refuses the budget, and AhaCode retries once without it
rather than failing every turn. `no_think_after_tools` is the one that matters most
in practice — a reasoning model re-deliberates on every turn of an agent loop, and a
per-turn budget only caps *one* turn, so the re-thinking stacks across a long loop.

## Sessions on disk

One session per file, one message per line, append-only:

```bash
ls .ahacode/sessions/
cat .ahacode/sessions/2026-08-18_212512.jsonl
tail -f .ahacode/sessions/*.jsonl   # watch messages land in real time
```

The first line of each file is a header — `{"type":"header","id","parent_id",
"kind","depth","model","title"}` — so a sub-agent's transcript points back at the
session that spawned it. The tree in the picker is derived by scanning those
headers; a parent never stores a child list, which keeps every write an append.

Everything a run generates — `sessions/`, `plans/`, `scratch/`, and a project
`config.toml` if you wrote one — sits under the single hidden `.ahacode/`, so one
`.gitignore` line keeps your key and your conversations out of every commit.

## Architecture

Three layers, with one vocabulary between them:

```
ahacode/
├── app.py            # the Textual App itself: layout, key/button wiring, and the
│                     # state a session has (which file, what depth, which mode).
│                     # Behaviour lives in the five collaborators it holds:
│   ├─ session_ctl.py #   SessionControl — new / switch / repair / replay history
│   ├─ plan_run.py    #   PlanRun — the gate, the handoff to an impl session, and
│   │                 #     the stall detection that ends an unattended run
│   ├─ runner.py      #   TurnRunner — the worker thread: the agent loop, tool
│   │                 #     approval, sub-agent spawning, and what a turn cost
│   ├─ turn_view.py   #   TurnView — canonical events -> mounted bubbles and cards
│   └─ commands.py    #   Commands — /model /url /allow /think (config only)
│
├── events.py         # the canonical event union every layer speaks
│                     #   ThinkingDelta · TextDelta · ToolCallDelta
│                     #   ToolCall · ToolResult · Notice · Phase · Usage
│
├── agent.py          # the agent loop: stream a turn → run its tools → feed the
│                     # results back → until a turn has no tool calls
├── context.py        # prune, then condense, before the history outgrows the window
├── subagent.py       # one delegated task as a fresh child loop (sub-agent-as-a-tool)
├── prompts.py        # system prompts, assembled per mode/model
├── render.py         # widget-free previews (diffs, syntax) shared by chat + modal
├── tools/
│   ├── base.py       #   the Tool contract (name · JSON Schema · execute)
│   ├── walk.py       #   shared traversal + skip rules for the search tools
│   ├── spill.py      #   oversized tool output -> a file the tools can read back
│   ├── read.py glob.py grep.py write.py edit.py bash.py plan.py task.py
│   ├── plan_submit.py #   plan mode's way out: validate, write plans/<session>.md
│   └── __init__.py   #   the registry, and the depth gate that hands out `task`
│
├── client.py         # LLM I/O — the only module that talks to a provider
├── workspace.py      # PROJECT_ROOT = the directory AhaCode was launched in
├── shell.py          # which shell a bash call gets, and how to kill its tree
├── config.py         # ~/.ahacode/config.toml + the project's optional override
├── session.py        # conversation state (plain Python, widget-free)
├── storage.py        # JSONL persistence + the session tree (./sessions/)
├── ahacode.tcss      # styles (no inline CSS)
└── widgets/          # one file per widget: chatbox, thinking, tool_result,
                      # subagent_card, todo_panel, plan_gate, approval_modal,
                      # model_bar, header_bar, prompt_input, session_picker
```

Design rules the codebase sticks to:

- **All LLM traffic goes through `client.py`.** The UI only ever sees the
  canonical events from `events.py`, so providers can be swapped without touching
  widgets. Provider quirks — which key reasoning deltas arrive under, the
  usage-only trailer chunk, tool-call arguments arriving as JSON fragments
  spread across chunks — are absorbed there.
- **The harness is widget-free.** `agent.run` reaches the UI only through an
  injected `emit` callback, and takes its stream, tool registry, and approval
  hook as arguments — so the whole agent loop is tested offline, without a
  terminal or a server.
- **Session state is a plain Python object** — testable without a terminal,
  serializable without ceremony.
- **UI is only touched from the main thread.** The worker hands events over via
  `call_from_thread` (built-in backpressure) and announces completion with a
  message, never by mutating shared state.
- **One file per tool and per widget**, styles in `.tcss`, never inline.
- **The App owns state; collaborators own behaviour.** `app.py` holds the
  session (which file, what depth, which mode) because everything reads it, and
  wires the keys and buttons Textual dispatches to it. Everything those handlers
  then *do* lives in one of the five collaborators above, each named for the
  question it answers — so "when does a run stop?" is `plan_run.py`, start to
  finish, rather than six pieces of state spread through the App.

## Development

```bash
git clone https://github.com/chycs7747/AhaCode && cd AhaCode
uv sync

uv run pytest -q                        # unit + headless TUI tests (Textual Pilot)
uv run textual run --dev ahacode.app    # run with devtools attached
uv run textual console                  # live logs in a second terminal

uv tool install -e .                    # `ahacode` everywhere, live off this checkout
```

`uv tool install -e` links the installed command at this working copy, so edits
land on the next run with no reinstall. Only a change to `dependencies` or to
`[project.scripts]` needs `--reinstall`.

Tests replace the LLM with an offline fake via `monkeypatch`, so the suite runs
without a server — including the TUI tests, which drive the real app through
Textual's `Pilot` (keystrokes in, mounted widgets out).

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
