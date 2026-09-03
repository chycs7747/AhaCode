"""The composer footer: model, mode and auto-approve controls, the live status,
and the Send button."""

from __future__ import annotations

from dataclasses import dataclass

from textual import on, work
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button, Checkbox, Select, Static
from textual.widgets._select import SelectCurrent, SelectOverlay

from ahacode import client, config


class Checkmark(Checkbox):
    """A Checkbox that shows ✓ when on and a truly empty box when off; Textual's
    default only dims the mark."""

    BUTTON_INNER = "✓"

    @property
    def _button(self):
        from textual.content import Content
        from textual.style import Style

        inner = self.BUTTON_INNER if self.value else " "
        button_style = self.get_visual_style("toggle--button")
        side_style = Style(
            foreground=button_style.background,
            background=self.background_colors[1],
        )
        return Content.assemble(
            (self.BUTTON_LEFT, side_style),
            (inner, button_style),
            (self.BUTTON_RIGHT, side_style),
        )


class ToggleSelect(Select):
    """A Select whose button also closes the open menu when clicked again.

    Textual's default double-fires: the click blurs the overlay (Dismiss) and then
    the button's own Toggle re-opens it. A just-happened lost-focus dismiss is
    remembered and the Toggle that follows it swallowed.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._suppress_reopen = False

    @on(SelectOverlay.Dismiss)
    def _select_overlay_dismiss(self, event: SelectOverlay.Dismiss) -> None:
        event.stop()
        event.prevent_default()  # replaces Select's own handler
        self.expanded = False
        if event.lost_focus:
            self._suppress_reopen = True
            self.call_after_refresh(self._allow_reopen)  # the next click is a fresh open
        else:
            self.focus()

    def _allow_reopen(self) -> None:
        self._suppress_reopen = False

    @on(SelectCurrent.Toggle)
    def _select_current_toggle(self, event: SelectCurrent.Toggle) -> None:
        event.stop()
        event.prevent_default()
        if self._suppress_reopen:
            self._suppress_reopen = False
            return
        self.expanded = not self.expanded


class ModelBar(Horizontal):
    """The composer footer. It only displays and reports: picking a model, a mode
    or auto-approve posts a message, and the app owns the state."""

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
        cfg = config.load()
        yield ToggleSelect(
            [(cfg.name, cfg.name)], value=cfg.name, allow_blank=False, id="model-select"
        )
        yield Static(id="status")
        yield ToggleSelect(
            [("act", "act"), ("plan", "plan")],
            value="act",
            allow_blank=False,
            id="mode-select",
        )
        yield Checkmark("auto-approve", value=False, id="auto-approve")
        yield Button("↑ Send", id="send-btn", variant="primary")

    def on_mount(self) -> None:
        self.refresh_state()
        self.load_models()

    def set_status(self, text: str) -> None:
        """Show live turn status; "" is idle.

        Args:
            text: The status line.
        """
        self.query_one("#status", Static).update(text)

    def refresh_state(self, names: list[str] | None = None) -> None:
        """Sync the bar with config.toml.

        Args:
            names: A replacement model list, when one was fetched.
        """
        cfg = config.load()
        if names is not None:
            self._names = list(names)
        # cfg.name first: set_options transiently selects the first option, and a
        # different one there would look like the user picking another model.
        self._names = [cfg.name, *[n for n in self._names if n != cfg.name]]
        select = self.query_one("#model-select", Select)
        select.set_options([(n, n) for n in self._names])
        select.value = cfg.name

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
            return  # a programmatic re-sync, not a user choice
        self.post_message(self.ModelChosen(str(event.value)))

    @on(Select.Changed, "#mode-select")
    def mode_changed(self, event: Select.Changed) -> None:
        event.stop()
        self.post_message(self.ModeChosen(str(event.value)))

    @on(Checkbox.Changed, "#auto-approve")
    def auto_approve_changed(self, event: Checkbox.Changed) -> None:
        event.stop()
        self.post_message(self.AutoApproveChanged(event.value))
