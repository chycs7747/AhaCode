from dataclasses import dataclass

from textual import on, work
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button, Checkbox, Select, Static
from textual.widgets._select import SelectCurrent, SelectOverlay

from ahacode import client, config


class Checkmark(Checkbox):
    """Checkbox that shows a ✓ when on and a truly empty box when off.

    Textual always draws BUTTON_INNER (its default off-state just dims it, so a
    mark still shows); we render a space when off so the box is genuinely empty
    until checked.
    """

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
    """A Select whose button also *closes* the open menu when clicked again.

    Textual's default double-fires: clicking the button while the menu is open
    first blurs the overlay (Dismiss → close) and then the button's own Toggle
    re-opens it, so a second click never closes it. We remember a just-happened
    lost-focus dismiss and swallow the Toggle that immediately follows it.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._suppress_reopen = False

    @on(SelectOverlay.Dismiss)
    def _select_overlay_dismiss(self, event: SelectOverlay.Dismiss) -> None:
        # prevent_default() breaks the MRO dispatch loop before Select's own
        # handler runs (ToggleSelect is first in the MRO), so ours fully replaces it.
        event.stop()
        event.prevent_default()
        self.expanded = False
        if event.lost_focus:
            # A click on the button blurs the overlay first; block the reopen the
            # button's Toggle would do next. Cleared on the next frame, so the very
            # next click (a fresh open) and click-away both still work.
            self._suppress_reopen = True
            self.call_after_refresh(self._allow_reopen)
        else:
            self.focus()

    def _allow_reopen(self) -> None:
        self._suppress_reopen = False

    @on(SelectCurrent.Toggle)
    def _select_current_toggle(self, event: SelectCurrent.Toggle) -> None:
        event.stop()
        event.prevent_default()  # replace Select's base toggle (see above)
        if self._suppress_reopen:
            self._suppress_reopen = False
            return  # the blur already closed it — don't reopen
        self.expanded = not self.expanded


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
        # Composer footer: model on the left, live status in the flexible middle,
        # and the mode + auto-approve controls next to Send on the right (Claude
        # Code's shape — the "what happens when I send" cluster is together). The
        # endpoint moved to the HeaderBar; it's connection identity, not a control.
        cfg = config.load()  # seed the model + allow_blank=False → no empty entry
        yield ToggleSelect(
            [(cfg.name, cfg.name)], value=cfg.name, allow_blank=False, id="model-select"
        )
        yield Static(id="status")  # live status / token stats (flexible middle)
        yield ToggleSelect(
            [("act", "act"), ("plan", "plan")],
            value="act",
            allow_blank=False,
            id="mode-select",
        )
        yield Checkmark("auto-approve", value=False, id="auto-approve")
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
