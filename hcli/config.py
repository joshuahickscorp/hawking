from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

from .context_budget import resolve as resolve_context_budget
from .persist import atomic_write_json


def coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def coerce_on_off(value: Any, default: str = "on") -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value).strip().lower()
    if text in {"off", "0", "false", "no"}:
        return "off"
    if text in {"on", "1", "true", "yes"}:
        return "on"
    return default


class Config:
    def __init__(self, workspace: str, global_path: Optional[str] = None):
        self.workspace = workspace
        self.global_path = global_path or os.path.expanduser("~/.config/hcli/config.json")
        self.project_path = os.path.join(workspace, ".hcli", "config.json")

    def _read(self, path: str) -> Dict[str, Any]:
        if not os.path.isfile(path):
            return {}
        try:
            with open(path) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write(self, path: str, data: Dict[str, Any]) -> None:
        atomic_write_json(path, data)

    def load(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        cfg.update(self._read(self.global_path))
        cfg.update(self._read(self.project_path))
        return cfg

    def layer_model(self, path: str) -> Optional[str]:
        model = self._read(path).get("model")
        if not model:
            return None
        return os.path.expanduser(str(model))

    def save_project(self, data: Dict[str, Any]):
        atomic_write_json(self.project_path, data)

    def save_global(self, data: Dict[str, Any]) -> None:
        existing = self._read(self.global_path)
        existing.update(data)
        self._write(self.global_path, existing)

    def value(
        self,
        key: str,
        env_key: Optional[str] = None,
        default: Any = None,
    ) -> Any:
        """Resolve a key: env wins, then project, then global, then default."""
        if env_key is not None and env_key in os.environ:
            return os.environ[env_key]
        loaded = self.load()
        if key in loaded:
            return loaded[key]
        return default

    def enable_thinking(self, default: bool = False) -> bool:
        return coerce_bool(
            self.value("enable_thinking", "HCLI_ENABLE_THINKING", None),
            default=default,
        )

    def response_schema_on(self, default: bool = True) -> bool:
        raw = self.value("response_schema", "HCLI_RESPONSE_SCHEMA", None)
        if raw is None:
            return default
        return coerce_on_off(raw, default="on" if default else "off") == "on"

    def model_tokens(self) -> Tuple[Optional[int], Optional[str]]:
        """Explicit completion budget and its source (env | config).

        Returns (None, None) when neither env nor config set a value so the
        caller can derive a budget from the context window.
        """
        raw_env = os.environ.get("HCLI_MODEL_TOKENS")
        if raw_env is not None and str(raw_env).strip() != "":
            try:
                return int(raw_env), "env"
            except ValueError:
                pass
        loaded = self.load()
        if "model_tokens" in loaded and loaded["model_tokens"] is not None:
            try:
                return int(loaded["model_tokens"]), "config"
            except (TypeError, ValueError):
                pass
        return None, None

    def ctx_size(self, default: Optional[int] = None) -> int:
        """Return the authority's total_ctx (--ctx-size).

        `default` is accepted for call-site compatibility and ignored:
        the conservative fallback lives in context_budget.DEFAULT_PER_SLOT_CTX.
        """
        del default
        return int(resolve_context_budget().total_ctx)
