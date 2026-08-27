from __future__ import annotations

import os
import sys
from typing import Optional

from .controller import Controller
from .tui import TUI
from .events import EventBus
from .workspace import Workspace
from .models import ModelRegistry


class App:
    def __init__(self, workspace: str, runtime_count: int = 1, model: Optional[str] = None, debug: bool = False):
        self.workspace = workspace
        self.runtime_count = runtime_count
        self.model = model
        self.debug = debug
        self.ws = Workspace(workspace)
        self.registry = ModelRegistry()
        self.bus = EventBus()
        self.controller = Controller(
            workspace=self.ws,
            runtime_count=runtime_count,
            model=model,
            bus=self.bus,
            registry=self.registry,
        )

    def run(self, prompt: Optional[str] = None, *, plain: bool = False) -> int:
        if prompt:
            return self._run_headless(prompt, plain=plain)
        return self._run_interactive()

    def _run_headless(self, prompt: str, *, plain: bool = False) -> int:
        self.bus.emit("session_started", {"mode": "headless"})
        try:
            # A slash command is a command in every mode. Routing it through
            # execute() sends `hcli /help` to the model, which is both wrong
            # and expensive.
            if prompt.startswith("/"):
                # Print what the TUI would render, not the structured payload
                # handle_command returns to programmatic callers. Both surfaces
                # must show the operator the same text.
                rendered: list = []
                self.bus.subscribe(
                    lambda e: rendered.append(
                        str(
                            (e.data or {}).get("content")
                            or (e.data or {}).get("message")
                            or ""
                        )
                    )
                    if e.type in ("final_response", "warning", "transcript_cleared")
                    else None
                )
                result = self.controller.handle_command(prompt)
                text = "\n".join(x for x in rendered if x).strip()
                if text:
                    print(text)
                    return 0
                if result is None:
                    print(f"Unknown command: {prompt.split()[0]}", file=sys.stderr)
                    return 1
                if isinstance(result, bool):
                    return 0
                print(result if isinstance(result, str) else str(result))
                return 0

            if plain:
                text = self.controller.complete_text(prompt)
                if text:
                    print(text)
                return 0

            result = self.controller.execute(prompt)

            if isinstance(result, int):
                return result

            if not isinstance(result, dict):
                print(str(result))
                return 0

            content = str(result.get("content") or "")
            error = str(result.get("error") or "")

            if content:
                print(content)
            elif error:
                print(error, file=sys.stderr)

            if result.get("cancelled") or str(
                result.get("status") or ""
            ) == "cancelled":
                return 130

            if str(result.get("status") or "") != "completed":
                return 1

            return 0
        finally:
            self.controller.shutdown()

    def _run_interactive(self) -> int:
        model_name = self.controller.model_name or "local"
        tui = TUI(
            event_bus=self.bus,
            workspace=self.ws.root,
            model_name=model_name,
            runtime_count=self.runtime_count,
        )
        self.bus.emit("session_started", {"mode": "interactive"})
        try:
            return tui.run(on_input=self._handle_input)
        finally:
            self.controller.shutdown()

    def _handle_input(self, text: str):
        self.bus.emit("user_message", {"text": text})
        if text.startswith("/"):
            return self.controller.handle_command(text)
        return self.controller.execute(text)
