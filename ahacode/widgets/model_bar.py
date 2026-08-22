from dataclasses import dataclass

from textual import on, work
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button, Checkbox, Select, Static

from ahacode import client, config


class ModelBar(Horizontal):
    """Composer footer under the prompt: mode + model + auto-approve controls, a
    live status/stats readout, and the Send button.

    The model picker is filled from the server's /v1/models. Picking an entry (or
    changing mode / auto-approve) posts a message so the app can persist it — the
    app owns config and client state, the bar only displays and reports.
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
        # Composer footer: settings on the left, live status in the flexible
        # middle, the send/stop action on the right (Claude Code's shape). The
        # endpoint moved to the HeaderBar — it's connection identity, not a control.
        yield Select(
            [("act", "act"), ("plan", "plan")],
            value="act",
            allow_blank=False,
            id="mode-select",
        )
        # Seed with the configured model + allow_blank=False so the dropdown never
        # shows an empty "model" placeholder entry (refresh_state fills the rest).
        cfg = config.load()
        yield Select(
            [(cfg.name, cfg.name)], value=cfg.name, allow_blank=False, id="model-select"
        )
        yield Checkbox("auto-approve", value=False, id="auto-approve")
        yield Static(id="status")  # live status / token stats (flexible middle)
        # Send button: a click alternative to Enter; the app flips it to "Stop"
        # while a turn is streaming. Button.Pressed bubbles to the app.
        yield Button("↑ Send", id="send-btn", variant="primary")

    def on_mount(self) -> None:
        self.refresh_state()
        self.load_models()

    def set_status(self, text: str) -> None:
        """Show live turn status in the bar (empty string = idle)."""
        self.query_one("#status", Static).update(text)

    def refresh_state(self, names: list[str] | None = None) -> None:
        """Sync the bar with config.toml; optionally replace the option list."""
        cfg = config.load()
        if names is not None:
            self._names = list(names)
        # cfg.name always first: set_options (allow_blank=False) transiently selects
        # the first option, so keeping the current model there avoids a spurious
        # Select.Changed that would look like the user picked another model.
        self._names = [cfg.name, *[n for n in self._names if n != cfg.name]]
        select = self.query_one("#model-select", Select)
        select.set_options([(n, n) for n in self._names])
        select.value = cfg.name  # explicit, in case set_options left it elsewhere

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
