"""AI (VLM) tab: endpoint, model, API key and prompts.

Ported from the Streamlit VLM page. The storage convention is unchanged: the key
itself goes to .env, and vlm_config.json holds a ${VAR} reference to it, so the
config file stays safe to commit.
"""

import json
import logging
import os
import re
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Callable, Dict, Optional

from dotenv import load_dotenv

from core.settings_manager import substitute_env_vars

VLM_CONFIG_FILE = os.path.join("config", "vlm_config.json")
ENV_FILE = ".env"

# Kept for the three historical profile slots so existing .env files keep working.
PROFILE_ENV_VARS = {
    "local": "VLM_API_KEY_1",
    "online1": "VLM_API_KEY_2",
    "online2": "VLM_API_KEY_3",
    "profile1": "VLM_API_KEY_1",
    "profile2": "VLM_API_KEY_2",
    "profile3": "VLM_API_KEY_3",
}

COMMON_ENDPOINTS = (
    ("Ollama (local)", "http://localhost:11434/v1"),
    ("LM Studio (local)", "http://localhost:1234/v1"),
    ("Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    ("OpenAI", "https://api.openai.com/v1"),
    ("OpenRouter", "https://openrouter.ai/api/v1"),
)


def env_var_for_profile(profile_name: str) -> str:
    if profile_name in PROFILE_ENV_VARS:
        return PROFILE_ENV_VARS[profile_name]
    suffix = re.sub(r"[^A-Z0-9]", "_", profile_name.upper())
    return f"VLM_API_KEY_{suffix}"


def update_env_file(key: str, value: str, env_path: str = ENV_FILE) -> None:
    """Set key=value in .env, replacing an existing entry in place."""
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    replacement = f'{key}="{value}"\n'
    out, found = [], False
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in line:
            if line.split("=", 1)[0].strip() == key:
                out.append(replacement)
                found = True
                continue
        out.append(line)

    if not found:
        if out and not out[-1].endswith("\n"):
            out.append("\n")
        out.append(replacement)

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(out)

    load_dotenv(dotenv_path=env_path, override=True)


class VlmTab(ttk.Frame):
    def __init__(self, parent: tk.Misc, on_dirty: Callable[[], None]):
        super().__init__(parent, padding=12)
        self._on_dirty = on_dirty
        self.raw_config: Dict[str, Any] = {}
        self._testing = False
        # Guards the var traces below. Populating the fields from config is not a
        # user edit, so it must not mark the config dirty -- otherwise merely
        # opening the app would show unsaved changes and prompt on close.
        self._loading = True

        self._load()
        self._build()
        self._show_profile(self.raw_config.get("current_profile", ""))
        self._loading = False

    # ------------------------------------------------------------------- state

    def _load(self):
        """Load the raw config -- unsubstituted, so ${VAR} refs stay intact."""
        try:
            with open(VLM_CONFIG_FILE, "r", encoding="utf-8") as f:
                self.raw_config = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logging.error(f"Could not read {VLM_CONFIG_FILE}: {e}")
            self.raw_config = {"current_profile": "", "profiles": {}, "prompts": {}}

    def _save(self) -> bool:
        try:
            with open(VLM_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.raw_config, f, indent=2, ensure_ascii=False)
            return True
        except OSError as e:
            logging.error(f"Could not write {VLM_CONFIG_FILE}: {e}")
            messagebox.showerror("AI settings", f"Could not save:\n\n{e}")
            return False

    @property
    def _profiles(self) -> Dict[str, Any]:
        return self.raw_config.setdefault("profiles", {})

    # ------------------------------------------------------------------ layout

    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Profile:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.profile_var = tk.StringVar()
        self.profile_combo = ttk.Combobox(
            top,
            textvariable=self.profile_var,
            values=list(self._profiles.keys()),
            state="readonly",
            width=28,
        )
        self.profile_combo.pack(side=tk.LEFT, padx=(6, 10))
        self.profile_combo.bind(
            "<<ComboboxSelected>>", lambda e: self._show_profile(self.profile_var.get())
        )

        self.active_label = ttk.Label(top, text="", foreground="#0a6b2d")
        self.active_label.pack(side=tk.LEFT)

        ttk.Button(top, text="Use this profile", command=self._set_active).pack(
            side=tk.RIGHT
        )

        ttk.Separator(self, orient="horizontal").pack(fill=tk.X, pady=10)

        self._build_fields()
        self._build_prompts()
        self._build_actions()

    def _build_fields(self):
        frame = ttk.LabelFrame(self, text="Connection", padding=10)
        frame.pack(fill=tk.X)
        frame.columnconfigure(1, weight=1)

        def row(index: int, label: str, hint: str = "") -> tk.StringVar:
            ttk.Label(frame, text=label, width=18).grid(
                row=index, column=0, sticky=tk.W, pady=4
            )
            var = tk.StringVar()
            entry = ttk.Entry(frame, textvariable=var, width=60)
            entry.grid(row=index, column=1, sticky="ew", pady=4)
            if hint:
                ttk.Label(frame, text=hint, foreground="#555").grid(
                    row=index, column=2, sticky=tk.W, padx=(8, 0)
                )
            var.trace_add("write", lambda *_: self._mark_dirty())
            return var

        self.name_var = row(0, "Display name:")
        self.url_var = row(1, "Endpoint URL:")
        self.model_var = row(2, "Model name:")

        ttk.Label(frame, text="API key:", width=18).grid(
            row=3, column=0, sticky=tk.W, pady=4
        )
        self.key_var = tk.StringVar()
        self.key_entry = ttk.Entry(
            frame, textvariable=self.key_var, width=60, show="•"
        )
        self.key_entry.grid(row=3, column=1, sticky="ew", pady=4)
        self.key_hint = ttk.Label(frame, text="", foreground="#555")
        self.key_hint.grid(row=3, column=2, sticky=tk.W, padx=(8, 0))

        params = ttk.Frame(frame)
        params.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))

        ttk.Label(params, text="Max tokens:").pack(side=tk.LEFT)
        self.max_tokens_var = tk.StringVar()
        ttk.Entry(params, textvariable=self.max_tokens_var, width=8).pack(
            side=tk.LEFT, padx=(6, 18)
        )

        ttk.Label(params, text="Temperature:").pack(side=tk.LEFT)
        self.temp_var = tk.StringVar()
        ttk.Entry(params, textvariable=self.temp_var, width=8).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        hints = ttk.Frame(frame)
        hints.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))
        ttk.Label(hints, text="Common endpoints:", foreground="#555").pack(side=tk.LEFT)
        for label, url in COMMON_ENDPOINTS:
            ttk.Button(
                hints,
                text=label,
                width=len(label) + 2,
                command=lambda u=url: self.url_var.set(u),
            ).pack(side=tk.LEFT, padx=2)

    def _build_prompts(self):
        frame = ttk.LabelFrame(
            self, text="Prompts (shared by every profile)", padding=10
        )
        frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        ttk.Label(frame, text="System prompt:").pack(anchor=tk.W)
        self.system_text = scrolledtext.ScrolledText(frame, height=4, wrap=tk.WORD)
        self.system_text.pack(fill=tk.X, pady=(2, 8))

        ttk.Label(frame, text="User prompt:").pack(anchor=tk.W)
        self.user_text = scrolledtext.ScrolledText(frame, height=12, wrap=tk.WORD)
        self.user_text.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

        prompts = self.raw_config.get("prompts", {})
        self.system_text.insert("1.0", prompts.get("oneshot_system_prompt", ""))
        self.user_text.insert("1.0", prompts.get("oneshot_user_prompt", ""))

        for widget in (self.system_text, self.user_text):
            widget.edit_modified(False)
            widget.bind("<<Modified>>", self._on_text_modified)

    def _build_actions(self):
        frame = ttk.Frame(self)
        frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(frame, text="Save AI settings", command=self.save).pack(side=tk.LEFT)
        self.test_button = ttk.Button(
            frame, text="Test connection", command=self._test_connection
        )
        self.test_button.pack(side=tk.LEFT, padx=(8, 0))

        self.test_label = ttk.Label(frame, text="")
        self.test_label.pack(side=tk.LEFT, padx=(12, 0))

        ttk.Label(
            self,
            text=(
                "The API key is written to .env; this file stores only a ${VAR} "
                "reference to it, so it stays safe to commit."
            ),
            foreground="#555",
            wraplength=760,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(8, 0))

    def _mark_dirty(self):
        if not self._loading:
            self._on_dirty()

    def _on_text_modified(self, event):
        widget = event.widget
        if widget.edit_modified():
            widget.edit_modified(False)
            self._mark_dirty()

    # ----------------------------------------------------------------- profile

    def _show_profile(self, name: str):
        profile = self._profiles.get(name, {})
        self._current = name

        was_loading = self._loading
        self._loading = True
        try:
            self._populate(name, profile)
        finally:
            self._loading = was_loading

    def _populate(self, name: str, profile: Dict[str, Any]):

        self.profile_var.set(name)
        self.name_var.set(profile.get("name", ""))
        self.url_var.set(profile.get("base_url", ""))
        self.model_var.set(profile.get("model_name", ""))
        self.max_tokens_var.set(str(profile.get("max_tokens", 1500)))
        self.temp_var.set(str(profile.get("temperature", 0.1)))

        env_var = env_var_for_profile(name) if name else ""
        raw_key = str(profile.get("api_key", ""))

        # Show the resolved key so the user sees whether .env actually has one.
        if raw_key.startswith("${") and raw_key.endswith("}"):
            env_var = raw_key[2:-1]
            resolved = os.getenv(env_var, "")
            self.key_hint.config(
                text=f"from {env_var}" if resolved else f"{env_var} is empty in .env",
                foreground="#555" if resolved else "#a15c00",
            )
            self.key_var.set(resolved)
        else:
            self.key_var.set(raw_key)
            self.key_hint.config(text=f"will be saved as {env_var}", foreground="#555")

        active = self.raw_config.get("current_profile")
        self.active_label.config(text="● in use" if name == active else "")

    def _set_active(self):
        if not self._current:
            return
        self.raw_config["current_profile"] = self._current
        if self._save():
            self._show_profile(self._current)

    def save(self):
        """Write the profile, sending the key to .env and a ${VAR} ref to config."""
        if not self._current:
            return

        profile = self._profiles.setdefault(self._current, {})
        env_var = env_var_for_profile(self._current)
        key = self.key_var.get().strip()

        if key:
            try:
                update_env_file(env_var, key)
            except OSError as e:
                messagebox.showerror("AI settings", f"Could not write .env:\n\n{e}")
                return
            profile["api_key"] = f"${{{env_var}}}"
        else:
            # Preserve an existing ${VAR} reference when the field is left blank.
            # The old Streamlit page overwrote it with "", which silently broke
            # the profile's link to .env.
            existing = str(profile.get("api_key", ""))
            if not (existing.startswith("${") and existing.endswith("}")):
                profile["api_key"] = ""

        profile["name"] = self.name_var.get().strip()
        profile["base_url"] = self.url_var.get().strip()
        profile["model_name"] = self.model_var.get().strip()

        try:
            profile["max_tokens"] = int(float(self.max_tokens_var.get()))
        except ValueError:
            messagebox.showwarning("AI settings", "Max tokens must be a number.")
            return
        try:
            profile["temperature"] = float(self.temp_var.get())
        except ValueError:
            messagebox.showwarning("AI settings", "Temperature must be a number.")
            return

        prompts = self.raw_config.setdefault("prompts", {})
        prompts["oneshot_system_prompt"] = self.system_text.get("1.0", tk.END).rstrip()
        prompts["oneshot_user_prompt"] = self.user_text.get("1.0", tk.END).rstrip()

        if self._save():
            self.test_label.config(text="Saved", foreground="#0a6b2d")
            self._show_profile(self._current)

    # -------------------------------------------------------------------- test

    def _test_connection(self):
        """Run the round-trip off the UI thread so the window stays responsive."""
        if self._testing:
            return
        self._testing = True
        self.test_button.state(["disabled"])
        self.test_label.config(text="Testing…", foreground="#555")

        resolved = substitute_env_vars(self.raw_config)
        resolved["current_profile"] = self._current

        def work():
            try:
                from ai.vlm_verifier import VLM_Verifier

                result = VLM_Verifier(resolved).test_vlm_connection()
            except Exception as e:
                result = {"success": False, "message": str(e)}
            self.after(0, lambda: self._test_done(result))

        threading.Thread(target=work, name="vlm-test", daemon=True).start()

    def _test_done(self, result: Optional[Dict[str, Any]]):
        self._testing = False
        self.test_button.state(["!disabled"])

        result = result or {}
        ok = bool(result.get("success"))
        message = result.get("message") or result.get("error") or (
            "Connection OK" if ok else "Failed"
        )
        self.test_label.config(
            text=str(message)[:120], foreground="#0a6b2d" if ok else "#b00020"
        )

    def reload_from_config(self):
        self._loading = True
        try:
            self._load()
            self.profile_combo.config(values=list(self._profiles.keys()))
            self._show_profile(self.raw_config.get("current_profile", ""))
        finally:
            self._loading = False
