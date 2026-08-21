from dataclasses import dataclass

from textual import on, work
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Select, Static

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

    def __init__(self) -> None:
        super().__init__()
        self._names: list[str] = []

    def compose(self):
        yield Static(id="endpoint")
        yield Select([], prompt="model", id="model-select")

    def on_mount(self) -> None:
        self.refresh_state()
        self.load_models()

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

    @on(Select.Changed)
    def model_changed(self, event: Select.Changed) -> None:
        event.stop()  # the raw Select event stays inside the bar
        if event.value is Select.NULL or event.value == config.load().name:
            return  # programmatic re-sync, not a user choice
        self.post_message(self.ModelChosen(str(event.value)))
