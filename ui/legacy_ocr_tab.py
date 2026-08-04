"""Legacy OCR tab: engine selection for the offline CPU-only path.

The OCR engines and their settings are unchanged -- this tab only gives them a
clearer home. Note that OCR is not purely legacy: trigger detection and
Rx-number extraction always run Tesseract, in both verification methods, so
Tesseract is required even when using AI vision.
"""

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict

from core.ocr_provider import check_ocr_availability
from core.settings_manager import config_value

ENGINES = (
    ("auto", "Auto", "Picks the best available engine, using the GPU when present."),
    ("tesseract", "Tesseract", "Fast and predictable on CPU. Needs the Tesseract binary installed."),
    ("easyocr", "EasyOCR", "More accurate on awkward text, slower to start."),
    ("paddleocr", "PaddleOCR", "Strong text detection, CPU optimised."),
)

INSTALL_HINTS = {
    "tesseract": "pip install pytesseract  (plus the Tesseract binary)",
    "easyocr": "pip install easyocr",
    "paddleocr": "pip install paddlepaddle paddleocr",
}


class LegacyOcrTab(ttk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        config: Dict[str, Any],
        on_dirty: Callable[[], None],
    ):
        super().__init__(parent, padding=12)
        self.config_data = config
        self._on_dirty = on_dirty

        self._available = self._detect_available()
        self._build()

    def _detect_available(self) -> Dict[str, bool]:
        # Returns a list of provider names, and raises ImportError when none are
        # installed -- which is a state this tab needs to render, not crash on.
        try:
            installed = set(check_ocr_availability() or [])
        except Exception:
            installed = set()
        return {name: name in installed for name, _, _ in ENGINES if name != "auto"}

    def _build(self):
        ttk.Label(
            self,
            text="Offline OCR engine",
            font=("Arial", 12, "bold"),
        ).pack(anchor=tk.W)
        ttk.Label(
            self,
            text=(
                "Used by the legacy OCR + fuzzy matching method, and always used for "
                "trigger detection and Rx-number reading regardless of which "
                "verification method is selected."
            ),
            wraplength=700,
            justify=tk.LEFT,
            foreground="#555",
        ).pack(anchor=tk.W, pady=(2, 12))

        self.engine_var = tk.StringVar(
            value=config_value(self.config_data, "ocr_provider")
        )

        for name, label, blurb in ENGINES:
            row = ttk.Frame(self)
            row.pack(fill=tk.X, anchor=tk.W)

            installed = name == "auto" or self._available.get(name, False)
            radio = ttk.Radiobutton(
                row,
                text=label,
                value=name,
                variable=self.engine_var,
                command=self._on_engine_change,
            )
            radio.pack(side=tk.LEFT)
            if not installed:
                radio.state(["disabled"])
                ttk.Label(row, text="not installed", foreground="#a15c00").pack(
                    side=tk.LEFT, padx=(8, 0)
                )

            detail = blurb if installed else f"{blurb}  —  {INSTALL_HINTS.get(name, '')}"
            ttk.Label(
                self, text=detail, wraplength=700, justify=tk.LEFT, foreground="#555"
            ).pack(anchor=tk.W, padx=(22, 0), pady=(0, 8))

        ttk.Separator(self, orient="horizontal").pack(fill=tk.X, pady=10)
        self._build_engine_settings()

    def _build_engine_settings(self):
        frame = ttk.LabelFrame(self, text="Engine settings", padding=10)
        frame.pack(fill=tk.X)

        tesseract = self.config_data.setdefault("tesseract", {})
        easyocr_cfg = self.config_data.setdefault("easyocr", {})

        grid = ttk.Frame(frame)
        grid.pack(fill=tk.X)

        ttk.Label(grid, text="Tesseract options:", width=20).grid(
            row=0, column=0, sticky=tk.W, pady=3
        )
        self.psm_var = tk.StringVar(
            value=config_value(self.config_data, "tesseract", "config_options")
        )
        entry = ttk.Entry(grid, textvariable=self.psm_var, width=18)
        entry.grid(row=0, column=1, sticky=tk.W, pady=3)
        entry.bind("<FocusOut>", lambda e: self._on_settings_change())
        entry.bind("<Return>", lambda e: self._on_settings_change())

        ttk.Label(grid, text="Fallback options:", width=20).grid(
            row=1, column=0, sticky=tk.W, pady=3
        )
        self.psm_fallback_var = tk.StringVar(
            value=config_value(self.config_data, "tesseract", "fallback_config")
        )
        entry2 = ttk.Entry(grid, textvariable=self.psm_fallback_var, width=18)
        entry2.grid(row=1, column=1, sticky=tk.W, pady=3)
        entry2.bind("<FocusOut>", lambda e: self._on_settings_change())
        entry2.bind("<Return>", lambda e: self._on_settings_change())

        self.gpu_var = tk.BooleanVar(
            value=bool(config_value(self.config_data, "easyocr", "use_gpu"))
        )
        ttk.Checkbutton(
            frame,
            text="EasyOCR: use GPU when available",
            variable=self.gpu_var,
            command=self._on_settings_change,
        ).pack(anchor=tk.W, pady=(8, 0))

    def _on_engine_change(self):
        self.config_data["ocr_provider"] = self.engine_var.get()
        self._on_dirty()

    def _on_settings_change(self):
        self.config_data.setdefault("tesseract", {})["config_options"] = self.psm_var.get()
        self.config_data["tesseract"]["fallback_config"] = self.psm_fallback_var.get()
        self.config_data.setdefault("easyocr", {})["use_gpu"] = self.gpu_var.get()
        self._on_dirty()

    def reload_from_config(self):
        self.engine_var.set(config_value(self.config_data, "ocr_provider"))
        self.psm_var.set(config_value(self.config_data, "tesseract", "config_options"))
        self.psm_fallback_var.set(
            config_value(self.config_data, "tesseract", "fallback_config")
        )
        self.gpu_var.set(bool(config_value(self.config_data, "easyocr", "use_gpu")))
