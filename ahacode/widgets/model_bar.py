from dataclasses import dataclass

from textual import on, work
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Checkbox, Select, Static

from ahacode import client, config


class ModelBar(Horizontal):
    """Status bar under the prompt: current endpoint plus a model picker.

    The picker is filled from the server's /v1/models. Picking an entry posts
    ModelChosen so the app can persist it — the app owns config and client state,
    the bar only displays and reports.
    """

    @dataclass
    class ModelChosen(Message):
        name: str

    @dataclass
    class ModeChosen(Message):
        mode: str

    @dataclass
    class AutoApproveChanged(Message):
        value: bool

    def __init__(self) -> None:
        super().__init__()
        self._names: list[str] = []

    def compose(self):
        yield Static(id="endpoint")
        yield Static(id="status")  # live turn status (Thinking / Running tool / …)
        yield Select([], prompt="model", id="model-select")
        yield Select(
            [("act", "act"), ("plan", "plan")],
            value="act",
            allow_blank=False,
            id="mode-select",
        )
        yield Checkbox("auto-approve", value=False, id="auto-approve")

    def on_mount(self) -> None:
        self.refresh_state()
        self.load_models()

    def set_status(self, text: str) -> None:
        """Show live turn status in the bar (empty string = idle)."""
        self.query_one("#status", Static).update(text)

    def refresh_state(self, names: list[str] | None = None) -> None:
        """Sync the bar with config.toml; optionally replace the option list."""
        cfg = config.load()
        self.query_one("#endpoint", Static).update(cfg.base_url)
        if names is not None:
            self._names = list(names)
        if cfg.name not in self._names:  # the configured model is always selectable
            self._names = [cfg.name, *self._names]
        select = self.query_one("#model-select", Select)
        select.set_options([(n, n) for n in self._names])  # resets the selection...
        select.value = cfg.name  # ...so re-select the current model

    @work(thread=True, exit_on_error=False)
    def load_models(self) -> None:
        """Fetch /v1/models off the main thread; an unreachable server leaves the list short."""
        try:
            names = client.list_models()
        except Exception:
            names = []
        self.app.call_from_thread(self.refresh_state, names)

    @on(Select.Changed, "#model-select")
    def model_changed(self, event: Select.Changed) -> None:
        event.stop()  # the raw Select event stays inside the bar
        if event.value is Select.NULL or event.value == config.load().name:
            return  # programmatic re-sync, not a user choice
        self.post_message(self.ModelChosen(str(event.value)))

    @on(Select.Changed, "#mode-select")
    def mode_changed(self, event: Select.Changed) -> None:
        event.stop()  # the raw Select event stays inside the bar
        self.post_message(self.ModeChosen(str(event.value)))

    @on(Checkbox.Changed, "#auto-approve")
    def auto_approve_changed(self, event: Checkbox.Changed) -> None:
        event.stop()  # the raw Checkbox event stays inside the bar
        self.post_message(self.AutoApproveChanged(event.value))
