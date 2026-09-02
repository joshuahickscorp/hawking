"""Metadata-only IO for Doctor. Weight bytes are a hard refuse.

A diagnosis that opened a safetensors shard, a pytorch .bin, a GGUF, or any
other weight body is not a metadata diagnosis. The access log is the proof:
every path Doctor opened is recorded, and anything that looks like a weight
file raises before a byte is read.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".npz",
    ".npy",
    ".onnx",
    ".pkl",
    ".pickle",
    ".msgpack",
    ".h5",
    ".tflite",
    ".ggml",
)

# Index JSON names contain "safetensors" but they are maps, not payloads.
INDEX_SUFFIXES = (
    ".safetensors.index.json",
    ".index.json",
)

SPECIMEN_METADATA_NAMES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "processor_config.json",
    }
)


class WeightBytesForbidden(RuntimeError):
    """Doctor attempted to open a weight body. The diagnosis is invalid."""


def is_weight_file(path: Path | str) -> bool:
    """True for payload shards. False for the JSON index that names them."""
    p = Path(path)
    name = p.name.lower()
    if any(name.endswith(suf) for suf in INDEX_SUFFIXES):
        return False
    if any(name.endswith(suf) for suf in WEIGHT_SUFFIXES):
        return True
    return False


def is_specimen_metadata(path: Path | str) -> bool:
    return Path(path).name in SPECIMEN_METADATA_NAMES


@dataclass
class AccessLog:
    """Every open Doctor performs. Weight-byte count stays at zero or we raise."""

    files_opened: list[str] = field(default_factory=list)
    metadata_bytes_read: int = 0
    refused: list[str] = field(default_factory=list)
    weight_bytes_loaded: int = 0

    def _record_refuse(self, path: Path) -> None:
        self.refused.append(str(path))

    def _guard(self, path: Path) -> Path:
        path = Path(path)
        if is_weight_file(path):
            self._record_refuse(path)
            raise WeightBytesForbidden(
                f"refused weight payload {path}; Doctor reads config/index/receipts only"
            )
        return path

    def open_bytes(self, path: Path | str, *, max_bytes: int | None = None) -> bytes:
        path = self._guard(Path(path))
        data = path.read_bytes()
        if max_bytes is not None and len(data) > max_bytes:
            raise WeightBytesForbidden(
                f"refused {path}: {len(data)} bytes exceeds metadata cap {max_bytes}"
            )
        self.files_opened.append(str(path))
        self.metadata_bytes_read += len(data)
        return data

    def open_text(self, path: Path | str, *, max_bytes: int | None = None) -> str:
        return self.open_bytes(path, max_bytes=max_bytes).decode("utf-8")

    def open_json(self, path: Path | str, *, max_bytes: int | None = None) -> Any:
        raw = self.open_text(path, max_bytes=max_bytes)
        return json.loads(raw)

    def report(self) -> dict[str, Any]:
        return {
            "weight_bytes_loaded": self.weight_bytes_loaded,
            "weight_files_opened": [],
            "weight_files_refused": list(self.refused),
            "metadata_files_opened": list(self.files_opened),
            "metadata_bytes_read": self.metadata_bytes_read,
            "evidence_tier": "STATIC",
            "rule": "config.json + *.index.json + receipts only; never a weight shard",
        }
