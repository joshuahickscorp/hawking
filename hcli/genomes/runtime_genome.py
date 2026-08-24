"""RuntimeGenome: learned per-backend performance science.

Not admission (that is ``resolve_runtime_limits``) and not a live
re-measure. The live MLX profile is copied from
``receipts/headless/CONVENTIONAL_CONTROL_SET.json``. llama.cpp Q5_K is
archived science; the deleted GGUF is not opened.

MachineGenome remains the machine/admission bag. This class records
runtime profiles *into* a MachineGenome via ``set_profile`` on the
compatibility bag, never by hand-editing
``~/.config/hcli/machine_genome.json`` or rewriting the probe's
``MACHINE_GENOME.json`` admission numbers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hcli.machine import MachineGenome, live_machine_identity
from hcli.runtime_iface import (
    ARCHIVED_Q5K_GGUF_NAME,
    archived_q5k_gguf_path,
    extract_live_mlx_profile,
    load_control_set,
    q5k_gguf_required,
    runtime_interface_census,
)

SCHEMA = "hawking.headless.runtime_genome.v1"
RECEIPT_REL = Path("receipts") / "headless" / "RUNTIME_GENOME.json"
CONTROL_REL = Path("receipts") / "headless" / "CONVENTIONAL_CONTROL_SET.json"


def _repo_root(start: Optional[Path] = None) -> Path:
    cur = Path(start) if start is not None else Path(__file__).resolve()
    for parent in (cur, *cur.parents):
        if (parent / "receipts" / "headless").is_dir() and (
            parent / "hcli" / "runtime.py"
        ).is_file():
            return parent
    return Path.cwd()


@dataclass
class RuntimeGenome:
    """Per-runtime science. Load from the control set; do not re-measure."""

    data: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None

    @classmethod
    def from_control_set(
        cls,
        repo: Optional[Union[str, Path]] = None,
        *,
        control: Optional[Dict[str, Any]] = None,
    ) -> "RuntimeGenome":
        root = _repo_root(Path(repo) if repo is not None else None)
        payload = control if control is not None else load_control_set(root)
        live = payload.get("live") if isinstance(payload.get("live"), dict) else {}
        archived = (
            payload.get("archived") if isinstance(payload.get("archived"), dict) else {}
        )
        metrics = live.get("metrics") if isinstance(live.get("metrics"), dict) else {}
        mlx_profile = extract_live_mlx_profile(
            metrics,
            source_receipt=str(CONTROL_REL),
        )
        llama = {
            "status": archived.get("status") or "ARCHIVED",
            "remeasured": False,
            "artifact": ARCHIVED_Q5K_GGUF_NAME,
            "artifact_path": str(archived_q5k_gguf_path()),
            "artifact_present": archived_q5k_gguf_path().is_file(),
            "required": q5k_gguf_required(),
            "source_receipt": str(CONTROL_REL),
            "note": (
                "llama.cpp Q5_K is archived science. The GGUF is gone. "
                "Numbers survive labelled ARCHIVED. No code path may require "
                "the file to exist."
            ),
            "metrics": {
                k: (archived.get(k) if not isinstance(archived.get(k), dict)
                    else {
                        "status": (archived.get(k) or {}).get("status"),
                        "value": (archived.get(k) or {}).get("value"),
                        "source_receipt": (archived.get(k) or {}).get("source_receipt"),
                    })
                for k in (
                    "startup", "prefill", "decode_tps", "context_limit",
                    "peak_memory", "quant",
                )
                if k in archived
            },
        }
        # Prefer the archived arm wrapper if present (conventional_control_set).
        if isinstance(archived.get("metrics"), dict):
            llama["metrics"] = {
                k: {
                    "status": (archived["metrics"].get(k) or {}).get("status"),
                    "value": (archived["metrics"].get(k) or {}).get("value"),
                    "source_receipt": (archived["metrics"].get(k) or {}).get(
                        "source_receipt"
                    ),
                }
                for k in archived["metrics"]
            }
        data = {
            "schema": SCHEMA,
            "producer": "hcli.genomes.runtime_genome.RuntimeGenome.from_control_set",
            "remeasured": False,
            "q5k_gguf_required": False,
            "q5k_gguf_opened": False,
            "machine": payload.get("machine") or live_machine_identity(),
            "live": {
                "runtime": "mlx",
                "status": live.get("status") or "LIVE",
                "artifact": live.get("artifact"),
                "profile": mlx_profile,
            },
            "archived": {
                "runtime": "llamacpp",
                "quant": "Q5_K",
                **llama,
            },
            "future": {
                "noetic_native": {
                    "status": "INTERFACE_ONLY",
                    "note": (
                        "Noetic native is a RuntimeBackend kind on the one "
                        "interface. complete() refuses to invent tokens."
                    ),
                }
            },
            "census": runtime_interface_census(),
            "control_set_commit": payload.get("commit"),
            "control_set_generated_at": payload.get("generated_at"),
        }
        return cls(data=data, path=root / RECEIPT_REL)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)

    def mlx_headline(self) -> Dict[str, Any]:
        live = self.data.get("live") or {}
        profile = live.get("profile") or {}
        return dict(profile.get("headline") or {})

    def record_into_machine_genome(self, genome: MachineGenome) -> None:
        """Attach profiles on the compatibility bag. Not admission. Not a save
        to ~/.config/hcli/machine_genome.json."""
        genome.set_profile("runtime_genome", {
            "schema": SCHEMA,
            "remeasured": False,
            "live_mlx": (self.data.get("live") or {}).get("profile"),
            "archived_llamacpp_q5k": self.data.get("archived"),
        })

    def save_receipt(self, path: Optional[Union[str, Path]] = None) -> Path:
        from hcli.persist import atomic_write_json

        dest = Path(path) if path is not None else self.path
        if dest is None:
            dest = _repo_root() / RECEIPT_REL
        dest.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(dest, self.data)
        self.path = dest
        return dest


def load_runtime_genome(repo: Optional[Union[str, Path]] = None) -> RuntimeGenome:
    return RuntimeGenome.from_control_set(repo)
