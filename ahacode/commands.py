"""The slash commands typed into the chat: they edit the config and return the
text to show. None reaches the model or the session file."""

from __future__ import annotations

from dataclasses import replace

from ahacode import client, config, tools

HELP = """Commands:
  /model           show the current model and endpoint
  /model <name>    switch to a different model
  /url <base_url>  switch to a different endpoint
  /think <n>       per-turn thinking budget in tokens (off = unbounded)
  /allow           list the rules that skip the approval prompt
  /allow <rule>    add one, e.g. /allow bash:uv run pytest*
  /new             start a new session
  /sessions        switch between sessions
  /help            this message"""


class Commands:
    """Runs one slash command against the config.

    Holds the app only to announce that settings moved. /new and /sessions never
    arrive here: they change what the running worker persists into, so the app
    handles them itself.
    """

    def __init__(self, app) -> None:
        self.app = app

    def handle(self, text: str) -> str:
        """Run one command line.

        Args:
            text: The full line, starting with "/".

        Returns:
            The text to show the user.
        """
        parts = text.split()
        cmd, args = parts[0].lower(), parts[1:]
        run = {
            "/help": lambda: HELP,
            "/model": lambda: self._model(args),
            "/url": lambda: self._url(args),
            "/allow": lambda: self._allow(args, text),
            "/think": lambda: self._think(args),
        }.get(cmd)
        if run is None:
            return f"unknown command: {cmd} — try /help"
        return run()

    def switch_model(self, name: str) -> str:
        """Persist a model choice; the server loads it on the next request.

        Args:
            name: The model id.

        Returns:
            The text to show the user.
        """
        config.save(replace(config.load(), name=name))
        client.reset()
        self.app.refresh_config_ui()
        return f"model switched to: {name} (loads on the next message — may take a while)"

    def _model(self, args: list[str]) -> str:
        cfg = config.load()
        if not args:
            return f"model: {cfg.name}\nendpoint: {cfg.base_url}"
        return self.switch_model(args[0])

    def _url(self, args: list[str]) -> str:
        cfg = config.load()
        if not args:
            return f"endpoint: {cfg.base_url}"
        config.save(replace(cfg, base_url=args[0]))
        client.reset()
        self.app.refresh_config_ui(reload_models=True)
        return f"endpoint switched to: {args[0]}"

    def _allow(self, args: list[str], text: str) -> str:
        cfg = config.load()
        if not args:
            if not cfg.allow_rules:
                return ("no allow rules — every side-effecting tool asks first.\n"
                        "add one with e.g.  /allow bash:uv run pytest*")
            return "allow rules:\n" + "\n".join(f"  {r}" for r in cfg.allow_rules)
        # Re-split the raw text: a rule keeps its spaces ("bash:git status*" is one rule).
        rule = text.split(maxsplit=1)[1].strip()
        if ":" not in rule and rule not in tools.REGISTRY:
            return f"unknown tool: {rule} — use  tool:pattern  (e.g. bash:ls*)"
        if rule in cfg.allow_rules:
            return f"already allowed: {rule}"
        config.save(replace(cfg, allow_rules=(*cfg.allow_rules, rule)))
        client.reset()
        return (f"allowed without asking: {rule}\n"
                "(the denylist still blocks dangerous commands)")

    def _think(self, args: list[str]) -> str:
        cfg = config.load()
        if not args:
            shown = (f"{cfg.thinking_token_budget} tokens/turn"
                     if cfg.thinking_token_budget else "off (unbounded)")
            return f"thinking budget: {shown}  ·  reasoning_effort: {cfg.reasoning_effort}"
        val = args[0].lower()
        if val in ("off", "0", "none"):
            budget = 0
        elif val.isdigit():
            budget = int(val)
        else:
            return "usage: /think <tokens>  |  /think off"
        config.save(replace(cfg, thinking_token_budget=budget))
        client.reset()
        if not budget:
            return "thinking budget: off (unbounded)"
        return (f"thinking budget: {budget} tokens/turn  "
                "(needs the server's reasoning-config to take effect)")
