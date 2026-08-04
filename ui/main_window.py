"""The application window.

Owns the Tk main thread and everything hanging off it: the tabs, the score
overlay, and the lifecycle of the verification worker thread.

Threading contract
------------------
Tk is not thread-safe, so only this class touches widgets. The verification
controller runs an asyncio loop on a worker thread and reports back through
``VerificationController(on_event=...)``. Those callbacks fire on the worker
thread, so they only put a tuple on ``self._events``; ``_drain_events`` runs on
the Tk thread via ``after()`` and does the actual UI work.

This is the fix for the old design, where the controller built its own ``tk.Tk()``
from a background thread and called ``update()`` once, so the overlay never
pumped events.
"""

import asyncio
import copy
import logging
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Dict, Optional

from core.settings_manager import SettingsManager
from ui.overlay import ScoreOverlay
from ui.run_tab import RunTab

APP_TITLE = "Pharmacy Pre-Check"
EVENT_POLL_MS = 150
THREAD_JOIN_TIMEOUT = 10.0


class MainWindow:
    def __init__(self, root: tk.Tk, config_file: str = "config/config.json"):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1700x950")
        self.root.minsize(1200, 760)

        self.settings_manager = SettingsManager(config_file)
        if not self.settings_manager.load_config():
            raise RuntimeError(
                f"Could not load {config_file}. Check that the file exists and is valid JSON."
            )
        self.config = self.settings_manager.config

        self._dirty = False
        self._events: "queue.Queue[tuple]" = queue.Queue()
        self._controller = None
        self._worker: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._closing = False

        self.overlay = ScoreOverlay(self.root)

        self._build_menu()
        self._build_tabs()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Control-s>", lambda e: self.save_config())
        self.root.after(EVENT_POLL_MS, self._drain_events)

    # ----------------------------------------------------------------- layout

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="Save", command=self.save_config, accelerator="Ctrl+S"
        )
        file_menu.add_command(label="Reload from disk", command=self.reload_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)

    def _build_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._tabs: Dict[str, ttk.Frame] = {}

        self.run_tab = RunTab(
            self.notebook,
            self.config,
            on_start=self.start_monitoring,
            on_stop=self.stop_monitoring,
            on_config_changed=self.mark_dirty,
            on_goto_tab=self.select_tab,
        )
        self._add_tab("run", self.run_tab, "Run")

        # Region editor and the settings panels come from the (embeddable)
        # settings GUI, so the canvas/drag/zoom code is reused unchanged.
        from ui.settings_gui import SettingsGUI

        self.settings = SettingsGUI(
            self.notebook,
            settings_manager=self.settings_manager,
            on_dirty=self.mark_dirty,
        )
        self._add_tab("regions", self.settings.regions_frame, "Regions")
        self._add_tab("matching", self.settings.matching_frame, "Matching")

        from ui.vlm_tab import VlmTab

        self.vlm_tab = VlmTab(self.notebook, on_dirty=self.mark_dirty)
        self._add_tab("ai", self.vlm_tab, "AI (VLM)")

        from ui.legacy_ocr_tab import LegacyOcrTab

        self.legacy_tab = LegacyOcrTab(
            self.notebook, self.config, on_dirty=self.mark_dirty
        )
        self._add_tab("legacy", self.legacy_tab, "Legacy OCR")

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _add_tab(self, key: str, frame: ttk.Frame, label: str):
        self._tabs[key] = frame
        self.notebook.add(frame, text=f"  {label}  ")

    def select_tab(self, key: str):
        frame = self._tabs.get(key)
        if frame is not None:
            self.notebook.select(frame)

    def _on_tab_changed(self, event=None):
        current = self.notebook.select()
        if current and current == str(self._tabs["regions"]):
            # Deferred so startup does not flash the window during screen capture.
            self.settings.ensure_screenshot()

    # ------------------------------------------------------------------ config

    def mark_dirty(self):
        self._dirty = True
        self.root.title(f"{APP_TITLE} *")

    def save_config(self):
        """Explicit save. The only place config is written to disk."""
        try:
            self.settings_manager.config = self.config
            if not self.settings_manager.save_config(create_backup=True):
                raise RuntimeError("save_config() returned False")
            self.settings_manager.cleanup_old_backups()
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
            messagebox.showerror(APP_TITLE, f"Could not save settings:\n\n{e}")
            return

        self._dirty = False
        self.root.title(APP_TITLE)
        self.settings.update_status("Settings saved")
        if self._controller is not None:
            self.run_tab.set_status(
                "Saved — restart monitoring for changes to take effect"
            )

    def reload_config(self):
        if self._dirty and not messagebox.askyesno(
            APP_TITLE, "Discard unsaved changes and reload from disk?"
        ):
            return
        if not self.settings_manager.load_config():
            messagebox.showerror(APP_TITLE, "Could not reload the config file.")
            return

        # Keep the same dict object so tabs holding a reference stay valid.
        self.config.clear()
        self.config.update(self.settings_manager.config)
        self.settings_manager.config = self.config

        self.run_tab.reload_from_config()
        self.settings.reload_from_config()
        self.legacy_tab.reload_from_config()
        self._dirty = False
        self.root.title(APP_TITLE)

    # -------------------------------------------------------------- monitoring

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def start_monitoring(self):
        if self.is_running:
            return

        if self._dirty and messagebox.askyesno(
            APP_TITLE, "Save your changes before starting?"
        ):
            self.save_config()

        from core.readiness import check

        state = check(self.config)
        if not state.ready:
            messagebox.showwarning(
                APP_TITLE,
                "Not ready to start:\n\n• " + "\n• ".join(state.problems),
            )
            self.run_tab.refresh_readiness()
            self.select_tab(state.fix_tab)
            return

        from core.verification_controller import VerificationController

        # Snapshot the config so edits made while running cannot change behavior
        # mid-prescription. Restarting picks up the new values.
        run_config = copy.deepcopy(self.config)

        loop = asyncio.new_event_loop()
        try:
            controller = VerificationController(run_config, loop, on_event=self._on_event)
        except Exception as e:
            loop.close()
            logging.error(f"Failed to create verification controller: {e}")
            messagebox.showerror(APP_TITLE, f"Could not start monitoring:\n\n{e}")
            return

        self._loop = loop
        self._controller = controller

        def run():
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(controller.async_run())
            except Exception as e:
                logging.error(f"Monitoring thread crashed: {e}", exc_info=True)
                self._events.put(("status", f"Stopped after an error: {e}"))
            finally:
                try:
                    loop.close()
                except Exception:
                    pass
                self._events.put(("finished", None))

        self._worker = threading.Thread(target=run, name="verification", daemon=True)
        self._worker.start()

        self.run_tab.set_running(True)
        self.run_tab.set_status("Starting…")
        logging.info("Monitoring started from the UI")

    def stop_monitoring(self):
        if self._controller is None:
            return
        self.run_tab.set_status("Stopping…")
        self._controller.stop()

    def _on_event(self, kind: str, payload: Any):
        """Called on the worker thread. Must not touch Tk."""
        self._events.put((kind, payload))

    def _drain_events(self):
        """Runs on the Tk thread; the only place events become UI changes."""
        try:
            while True:
                kind, payload = self._events.get_nowait()
                try:
                    self._handle_event(kind, payload)
                except Exception as e:
                    logging.error(f"Error handling '{kind}' event: {e}")
        except queue.Empty:
            pass

        if not self._closing:
            self.root.after(EVENT_POLL_MS, self._drain_events)

    def _handle_event(self, kind: str, payload: Any):
        if kind == "results":
            self.overlay.show(payload)
            rx = getattr(self._controller, "last_rx_number", "") or ""
            self.run_tab.show_results(rx, payload)
        elif kind == "clear_overlay":
            self.overlay.hide()
        elif kind == "status":
            self.run_tab.set_status(payload)
        elif kind == "finished":
            self._join_worker()
            self.run_tab.set_running(False)
            self.overlay.hide()

    def _join_worker(self):
        if self._worker is not None:
            self._worker.join(timeout=THREAD_JOIN_TIMEOUT)
            if self._worker.is_alive():
                logging.warning("Verification thread did not exit within the timeout")
        self._worker = None
        self._controller = None
        self._loop = None

    # ------------------------------------------------------------------ close

    def on_close(self):
        """Tear down in order so no thread outlives the window."""
        if self._dirty:
            answer = messagebox.askyesnocancel(
                APP_TITLE, "Save your settings before closing?"
            )
            if answer is None:
                return
            if answer:
                self.save_config()

        self._closing = True

        if self._controller is not None:
            logging.info("Window closing - stopping monitoring")
            self._controller.stop()
            self._join_worker()

        self.overlay.destroy()
        logging.info("Application closed")
        self.root.destroy()

    def run(self):
        self.root.mainloop()
