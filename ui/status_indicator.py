"""On-screen status indicator.

A small always-on-top pill -- a colored dot and one short word -- that shows what
the monitor is doing without the main window being visible. Owned by the Tk main
thread, like the score overlay: the controller only emits "state" events and the
app forwards them here.

It is static on purpose (no blinking): it is part of the screen the monitor
captures, so it should not keep changing pixels.
"""

import logging
import tkinter as tk
import tkinter.font as tkfont
from typing import Any, Callable, Dict, Optional

from core.settings_manager import config_value

# state -> (dot color, label)
STATES = {
    "stopped": ("#8a8f98", "Stopped"),
    "waiting": ("#3b82f6", "Waiting"),
    "reading": ("#f5a524", "Reading"),
    "checking": ("#f5a524", "Checking"),
    "sending": ("#f5a524", "Sending key"),
    "match": ("#22c55e", "Match"),
    "review": ("#ef4444", "Review"),
    "unreadable": ("#d946ef", "Can't read Rx"),
    "error": ("#d946ef", "Check failed"),
}

CORNERS = ("top-left", "top-center", "top-right", "bottom-left", "bottom-right")

PILL_COLOR = "#1f2329"
TEXT_COLOR = "#ffffff"
# Windows-only color key that makes the canvas corners see-through. Elsewhere Tk
# raises TclError and the indicator is simply drawn as a rectangle.
TRANSPARENT_COLOR = "#010203"

HEIGHT = 22
DOT = 12
PAD = 6


class StatusIndicator:
    def __init__(self, master: tk.Misc, on_click: Optional[Callable[[], None]] = None):
        self._master = master
        self._on_click = on_click
        self._window: Optional[tk.Toplevel] = None
        self._canvas: Optional[tk.Canvas] = None
        self._font: Optional[tkfont.Font] = None
        self._rounded = False

        self._state = "stopped"
        self._enabled = True
        self._corner = "top-center"
        self._show_label = True
        self._opacity = 0.85
        self._margin = 8

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply the "indicator" section of the app config and redraw."""
        self._enabled = bool(config_value(config, "indicator", "enabled"))
        corner = config_value(config, "indicator", "corner")
        self._corner = corner if corner in CORNERS else "top-center"
        self._show_label = bool(config_value(config, "indicator", "show_label"))
        self._opacity = float(config_value(config, "indicator", "opacity"))
        self._margin = int(config_value(config, "indicator", "margin_px"))
        self._redraw()

    def set_state(self, state: str) -> None:
        if state not in STATES:
            logging.debug(f"Indicator: unknown state '{state}'")
            return
        if state != self._state:
            self._state = state
            self._redraw()

    def _ensure_window(self) -> bool:
        if self._window is not None:
            return True
        try:
            window = tk.Toplevel(self._master)
            window.overrideredirect(True)
            window.wm_attributes("-topmost", True)

            canvas_bg = PILL_COLOR
            try:
                window.wm_attributes("-transparentcolor", TRANSPARENT_COLOR)
                canvas_bg = TRANSPARENT_COLOR
                self._rounded = True
            except tk.TclError:
                logging.debug("Indicator: -transparentcolor unsupported, drawing a rectangle")

            self._canvas = tk.Canvas(
                window, bg=canvas_bg, highlightthickness=0, height=HEIGHT, cursor="hand2"
            )
            self._canvas.pack()
            self._canvas.bind("<Button-1>", self._clicked)
            self._font = tkfont.Font(family="Segoe UI", size=9, weight="bold")
            self._window = window
            return True
        except Exception as e:
            logging.error(f"Indicator: failed to create window: {e}")
            return False

    def _clicked(self, event=None):
        if self._on_click:
            self._on_click()

    def _redraw(self) -> None:
        if not self._enabled:
            if self._window is not None:
                self._window.withdraw()
            return
        if not self._ensure_window():
            return

        try:
            color, label = STATES[self._state]
            width = HEIGHT
            if self._show_label:
                width = PAD + DOT + PAD + self._font.measure(label) + PAD + 2

            canvas = self._canvas
            canvas.delete("all")
            canvas.config(width=width)

            if self._rounded:
                # A pill: two end caps joined by a rectangle.
                canvas.create_oval(0, 0, HEIGHT, HEIGHT, fill=PILL_COLOR, outline=PILL_COLOR)
                canvas.create_oval(
                    width - HEIGHT, 0, width, HEIGHT, fill=PILL_COLOR, outline=PILL_COLOR
                )
                canvas.create_rectangle(
                    HEIGHT // 2, 0, width - HEIGHT // 2, HEIGHT, fill=PILL_COLOR, outline=PILL_COLOR
                )

            dot_x = (HEIGHT - DOT) // 2 if not self._show_label else PAD
            dot_y = (HEIGHT - DOT) // 2
            canvas.create_oval(
                dot_x, dot_y, dot_x + DOT, dot_y + DOT, fill=color, outline=color
            )
            if self._show_label:
                canvas.create_text(
                    PAD + DOT + PAD, HEIGHT // 2, text=label, anchor=tk.W,
                    fill=TEXT_COLOR, font=self._font,
                )

            self._window.geometry(f"{width}x{HEIGHT}+{self._x(width)}+{self._y()}")
            try:
                self._window.wm_attributes("-alpha", max(0.2, min(1.0, self._opacity)))
            except tk.TclError:
                pass
            self._window.deiconify()
            self._window.lift()
        except Exception as e:
            logging.error(f"Indicator: failed to draw: {e}")

    def _x(self, width: int) -> int:
        screen = self._window.winfo_screenwidth()
        if self._corner.endswith("left"):
            return self._margin
        if self._corner.endswith("center"):
            return (screen - width) // 2
        return screen - width - self._margin

    def _y(self) -> int:
        if self._corner.startswith("top"):
            return self._margin
        return self._window.winfo_screenheight() - HEIGHT - self._margin

    def destroy(self) -> None:
        if self._window is None:
            return
        try:
            self._window.destroy()
        except tk.TclError:
            pass
        self._window = None
        self._canvas = None
