#!/usr/bin/env pythonw
"""Pharmacy Pre-Check — single-window desktop app.

The .pyw extension makes Windows launch this with pythonw.exe, so a
double-click opens the window with no console. Everything lives in this one
process: the Tk UI on the main thread, the verification loop on a worker
thread, and the score overlay as a child window of the app root.

    python app.pyw          # or just double-click on Windows

IMPORTANT: this module must stay import-safe. ProcessPoolExecutor uses spawn on
Windows, which re-imports __main__ in every OCR worker. Nothing may run at
import time -- no Tk, no config loading -- or each worker would try to build its
own UI.
"""

import multiprocessing
import os
import sys

# Make the repo root importable regardless of the working directory the app was
# launched from (double-clicking on Windows sets cwd to the file's folder, but
# shortcuts and scheduled tasks do not).
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    # Config paths throughout the codebase are relative ("config/config.json"),
    # so anchor the process at the repo root before anything reads them.
    os.chdir(_ROOT)

    from core.logger_config import setup_logging

    # No console when launched via pythonw, so stream logging would write to a
    # dead handle. The Run tab tails the log file instead.
    setup_logging(add_stream=sys.stderr is not None and sys.stderr.isatty())

    import tkinter as tk
    from tkinter import messagebox

    from ui.main_window import MainWindow

    root = tk.Tk()
    try:
        window = MainWindow(root)
    except Exception as e:
        import logging
        import traceback

        logging.error(f"Failed to start: {e}\n{traceback.format_exc()}")
        messagebox.showerror(
            "Pharmacy Pre-Check",
            f"Failed to start:\n\n{e}\n\nSee verification.log for details.",
        )
        return 1

    window.run()
    return 0


if __name__ == "__main__":
    # Required before creating a ProcessPoolExecutor in a frozen/spawned context.
    multiprocessing.freeze_support()
    sys.exit(main())
