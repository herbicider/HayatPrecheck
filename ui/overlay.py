"""Score overlay window.

Draws colored rectangles over the verified screen regions. Owned by the Tk main
thread: the verification controller runs in a worker thread and never touches Tk,
it only emits result events that the app forwards here.

The window is created once and reused (withdraw/deiconify) rather than rebuilt
per verification.
"""

import logging
import tkinter as tk
from typing import Any, Dict

# Windows-only window attributes. On other platforms Tk raises TclError; the
# overlay still draws, just without click-through or color-keyed transparency.
TRANSPARENT_COLOR = "white"


def score_color(score: float, threshold: float) -> str:
    """Map a match score to a fill color.

    100 is dark green, threshold..100 ramps light->dark green, 1..threshold ramps
    red->light red, and 0 is pure red.
    """
    if score >= 100:
        return "#006400"

    if score >= threshold:
        # Light green (144,238,144) -> dark green (0,100,0)
        ratio = (score - threshold) / (100 - threshold) if threshold < 100 else 1
        r = int(144 * (1 - ratio))
        g = int(238 * (1 - ratio) + 100 * ratio)
        b = int(144 * (1 - ratio))
        return f"#{r:02x}{g:02x}{b:02x}"

    if score > 0:
        # Red (255,0,0) -> light red (255,182,193)
        ratio = score / threshold if threshold > 0 else 0
        return f"#{255:02x}{int(182 * ratio):02x}{int(193 * ratio):02x}"

    return "#FF0000"


class ScoreOverlay:
    """A borderless, always-on-top, click-through window of score rectangles."""

    def __init__(self, master: tk.Misc):
        self._master = master
        self._window: tk.Toplevel = None
        self._canvas: tk.Canvas = None
        self._visible = False

    @property
    def visible(self) -> bool:
        return self._visible

    def _ensure_window(self) -> bool:
        """Create the Toplevel on first use. Returns False if unavailable."""
        if self._window is not None:
            return True

        try:
            window = tk.Toplevel(self._master)
            window.overrideredirect(True)
            window.geometry(
                f"{window.winfo_screenwidth()}x{window.winfo_screenheight()}+0+0"
            )

            # Best-effort: each of these is unsupported on some platforms.
            for attribute, value in (
                ("-topmost", True),
                ("-disabled", True),
                ("-transparentcolor", TRANSPARENT_COLOR),
            ):
                try:
                    window.wm_attributes(attribute, value)
                except tk.TclError:
                    logging.debug(f"Overlay: {attribute} unsupported on this platform")

            self._canvas = tk.Canvas(
                window, bg=TRANSPARENT_COLOR, highlightthickness=0
            )
            self._canvas.pack(fill="both", expand=True)

            window.withdraw()
            self._window = window
            return True
        except Exception as e:
            logging.error(f"Overlay: failed to create window: {e}")
            return False

    def show(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Draw one rectangle per result that carries valid coordinates."""
        if not self._ensure_window():
            return

        try:
            self._canvas.delete("all")

            drawn = 0
            for field_name, result in results.items():
                coords = result.get("coords")
                if not coords or len(coords) != 4:
                    logging.debug(f"Overlay: no valid coords for {field_name}, skipping")
                    continue

                color = score_color(result.get("score", 0), result.get("threshold", 80))
                self._canvas.create_rectangle(*coords, outline=color, width=3)
                drawn += 1

            if not drawn:
                logging.warning("Overlay: no fields had coordinates, nothing to draw")
                return

            self._window.deiconify()
            self._window.lift()
            self._visible = True
            logging.info(f"Overlay displayed ({drawn} field(s))")
        except Exception as e:
            logging.error(f"Overlay: failed to draw: {e}")

    def hide(self) -> None:
        if self._window is None or not self._visible:
            return
        try:
            self._window.withdraw()
        except tk.TclError:
            pass
        self._visible = False

    def destroy(self) -> None:
        if self._window is None:
            return
        try:
            self._window.destroy()
        except tk.TclError:
            pass
        self._window = None
        self._canvas = None
        self._visible = False
