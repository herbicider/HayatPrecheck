"""The Run tab: start/stop monitoring and watch it work.

This is the tab the user sits on all day, so it answers only the questions that
matter while working: is it running, is it configured, what did it just score,
and what is in the log.
"""

import os
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict

from core import readiness

METHOD_CHOICES = (
    (
        "vlm_ai",
        "AI vision (VLM)",
        "Sends one screenshot to a vision model. Handles handwriting and odd layouts.",
    ),
    (
        "local_ocr_fuzzy",
        "Legacy OCR + fuzzy match",
        "Reads each field with local OCR and compares text. Offline, CPU only, no AI.",
    ),
)

PRESET_KEYS = (
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
    "enter", "tab", "space", "escape",
)

LOG_TAIL_LINES = 200
LOG_POLL_MS = 1000


class RunTab(ttk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        config: Dict[str, Any],
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_config_changed: Callable[[], None],
        on_goto_tab: Callable[[str], None],
        log_file: str = "verification.log",
    ):
        super().__init__(parent, padding=12)
        self.config_data = config
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_config_changed = on_config_changed
        self._on_goto_tab = on_goto_tab
        self.log_file = log_file

        self._log_size = 0
        self._log_job = None

        self._build()
        self.refresh_readiness()
        self.set_running(False)

    # ---------------------------------------------------------------- layout

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        self._build_method(row=0)
        self._build_controls(row=1)
        self._build_scores(row=2)
        self._build_automation(row=3)
        self._build_log(row=4)

    def _build_method(self, row: int):
        frame = ttk.LabelFrame(self, text="Verification method", padding=10)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))

        current = self.config_data.get("verification_method", "local_ocr_fuzzy")
        self.method_var = tk.StringVar(value=current)

        for value, label, blurb in METHOD_CHOICES:
            ttk.Radiobutton(
                frame,
                text=label,
                value=value,
                variable=self.method_var,
                command=self._on_method_change,
            ).pack(anchor=tk.W)
            ttk.Label(
                frame, text=blurb, foreground="#555", wraplength=760, justify=tk.LEFT
            ).pack(anchor=tk.W, padx=(22, 0), pady=(0, 6))

    def _build_controls(self, row: int):
        frame = ttk.Frame(self)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))

        self.start_button = ttk.Button(frame, text="▶  Start", command=self._on_start)
        self.start_button.pack(side=tk.LEFT)

        self.stop_button = ttk.Button(frame, text="■  Stop", command=self._on_stop)
        self.stop_button.pack(side=tk.LEFT, padx=(6, 16))

        self.status_label = ttk.Label(frame, text="Stopped", font=("Arial", 11, "bold"))
        self.status_label.pack(side=tk.LEFT)

        # Readiness sits next to the buttons because it is the reason Start fails.
        self.readiness_frame = ttk.Frame(self)
        self.readiness_frame.grid(row=row, column=0, sticky="ew", pady=(34, 0))

        self.readiness_label = ttk.Label(
            self.readiness_frame, wraplength=640, justify=tk.LEFT
        )
        self.readiness_label.pack(side=tk.LEFT, anchor=tk.N)

        self.readiness_button = ttk.Button(
            self.readiness_frame, text="Fix this", command=self._goto_fix_tab
        )

    def _build_scores(self, row: int):
        frame = ttk.LabelFrame(self, text="Last verification", padding=10)
        frame.grid(row=row, column=0, sticky="ew", pady=(16, 10))

        self.rx_label = ttk.Label(frame, text="No prescription verified yet")
        self.rx_label.pack(anchor=tk.W, pady=(0, 6))

        self.score_tree = ttk.Treeview(
            frame,
            columns=("score", "threshold", "match"),
            show="tree headings",
            height=5,
        )
        self.score_tree.heading("#0", text="Field")
        self.score_tree.heading("score", text="Score")
        self.score_tree.heading("threshold", text="Threshold")
        self.score_tree.heading("match", text="Result")
        self.score_tree.column("#0", width=200)
        for col in ("score", "threshold", "match"):
            self.score_tree.column(col, width=90, anchor=tk.CENTER)
        self.score_tree.pack(fill=tk.X)

    def _build_automation(self, row: int):
        frame = ttk.LabelFrame(self, text="On full match", padding=10)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))

        automation = self.config_data.setdefault("automation", {})

        self.auto_var = tk.BooleanVar(
            value=bool(automation.get("send_key_on_all_match", False))
        )
        ttk.Checkbutton(
            frame,
            text="Press a key automatically when every field matches",
            variable=self.auto_var,
            command=self._on_automation_change,
        ).pack(anchor=tk.W)

        key_row = ttk.Frame(frame)
        key_row.pack(anchor=tk.W, pady=(6, 0))

        ttk.Label(key_row, text="Key:").pack(side=tk.LEFT)
        self.key_var = tk.StringVar(value=automation.get("key_on_all_match", "f12"))
        key_combo = ttk.Combobox(
            key_row,
            textvariable=self.key_var,
            values=list(PRESET_KEYS),
            width=10,
            state="readonly",
        )
        key_combo.pack(side=tk.LEFT, padx=(6, 0))
        key_combo.bind("<<ComboboxSelected>>", lambda e: self._on_automation_change())

        mode = automation.get("mode", "preset_key")
        if mode == "autohotkey_v2":
            ttk.Label(
                key_row,
                text="(overridden by the custom script on the Matching tab)",
                foreground="#a15c00",
            ).pack(side=tk.LEFT, padx=(10, 0))

    def _build_log(self, row: int):
        frame = ttk.LabelFrame(self, text="Activity", padding=10)
        frame.grid(row=row, column=0, sticky="nsew")
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        controls = ttk.Frame(frame)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.follow_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Follow", variable=self.follow_var).pack(
            side=tk.LEFT
        )
        ttk.Button(controls, text="Reload", command=lambda: self._read_log(force=True)).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        text_wrap = ttk.Frame(frame)
        text_wrap.grid(row=1, column=0, sticky="nsew")
        text_wrap.rowconfigure(0, weight=1)
        text_wrap.columnconfigure(0, weight=1)

        self.log_text = tk.Text(
            text_wrap, height=12, wrap=tk.NONE, font=("Consolas", 9), state=tk.DISABLED
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        yscroll = ttk.Scrollbar(text_wrap, orient="vertical", command=self.log_text.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(
            text_wrap, orient="horizontal", command=self.log_text.xview
        )
        xscroll.grid(row=1, column=0, sticky="ew")
        self.log_text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

    # --------------------------------------------------------------- updates

    def set_running(self, running: bool):
        self.start_button.state(["disabled"] if running else ["!disabled"])
        self.stop_button.state(["!disabled"] if running else ["disabled"])
        if running:
            self._schedule_log_poll()
        else:
            self._cancel_log_poll()

    def set_status(self, text: str):
        self.status_label.config(text=text)

    def show_results(self, rx_number: str, results: Dict[str, Dict[str, Any]]):
        self.score_tree.delete(*self.score_tree.get_children())

        matched = sum(1 for r in results.values() if r.get("match"))
        label = f"Rx#{rx_number}" if rx_number else "Prescription"
        self.rx_label.config(text=f"{label} — {matched} of {len(results)} fields matched")

        for field, result in results.items():
            display = readiness.FIELD_LABELS.get(
                field, field.replace("_", " ").title()
            )
            self.score_tree.insert(
                "",
                tk.END,
                text=display,
                values=(
                    f"{result.get('score', 0):.0f}",
                    f"{result.get('threshold', 0):.0f}",
                    "match" if result.get("match") else "REVIEW",
                ),
            )

        self._read_log()

    def refresh_readiness(self):
        """Re-check config and update the banner. Call after any config edit."""
        state = readiness.check(self.config_data)

        if state.ready:
            self.readiness_label.config(text="✔  Ready to start", foreground="#0a6b2d")
            self.readiness_button.pack_forget()
            return

        shown = state.problems[:4]
        if len(state.problems) > len(shown):
            shown.append(f"…and {len(state.problems) - len(shown)} more")

        self.readiness_label.config(
            text=state.summary + ":\n• " + "\n• ".join(shown), foreground="#a15c00"
        )
        self._fix_tab = state.fix_tab
        self.readiness_button.pack(side=tk.LEFT, padx=(12, 0), anchor=tk.N)

    def reload_from_config(self):
        """Pull widget values back from config (after an import or reset)."""
        self.method_var.set(self.config_data.get("verification_method", "local_ocr_fuzzy"))
        automation = self.config_data.get("automation", {})
        self.auto_var.set(bool(automation.get("send_key_on_all_match", False)))
        self.key_var.set(automation.get("key_on_all_match", "f12"))
        self.refresh_readiness()

    # --------------------------------------------------------------- handlers

    def _on_method_change(self):
        self.config_data["verification_method"] = self.method_var.get()
        self.refresh_readiness()
        self._on_config_changed()

    def _on_automation_change(self):
        automation = self.config_data.setdefault("automation", {})
        automation["send_key_on_all_match"] = self.auto_var.get()
        automation["key_on_all_match"] = self.key_var.get()
        self._on_config_changed()

    def _goto_fix_tab(self):
        self._on_goto_tab(getattr(self, "_fix_tab", "regions"))

    # -------------------------------------------------------------- log tail

    def _schedule_log_poll(self):
        self._cancel_log_poll()
        self._read_log()
        self._log_job = self.after(LOG_POLL_MS, self._schedule_log_poll)

    def _cancel_log_poll(self):
        if self._log_job is not None:
            self.after_cancel(self._log_job)
            self._log_job = None

    def _read_log(self, force: bool = False):
        """Render the log tail, skipping the read when the file has not grown."""
        try:
            size = os.path.getsize(self.log_file)
        except OSError:
            return

        if not force and size == self._log_size:
            return
        self._log_size = size

        try:
            with open(self.log_file, "r", encoding="utf-8", errors="replace") as f:
                # Seek near the end rather than reading a multi-megabyte log.
                if size > 200_000:
                    f.seek(size - 200_000)
                    f.readline()
                lines = f.readlines()[-LOG_TAIL_LINES:]
        except OSError:
            return

        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert("1.0", "".join(lines))
        self.log_text.config(state=tk.DISABLED)
        if self.follow_var.get():
            self.log_text.see(tk.END)
