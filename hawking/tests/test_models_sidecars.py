from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hawking.models import (
    discover_models,
    resolve_model,
    selectable_models,
)


def _write_gguf(path: Path, kvs: list) -> None:
    parts = [b"GGUF", struct.pack("<I", 3), struct.pack("<QQ", 0, len(kvs))]

    def enc_str(s: str) -> bytes:
        raw = s.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    for key, value in kvs:
        parts.append(enc_str(key))
        parts.append(struct.pack("<I", 8))
        parts.append(enc_str(value))
    path.write_bytes(b"".join(parts))


class TestSidecars(unittest.TestCase):
    def test_retired_gguf_and_mmproj_are_not_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf").write_bytes(b"x" * 100)
            (Path(tmp) / "mmproj-model-bf16.gguf").write_bytes(b"x" * 100)
            found = discover_models([tmp])
            assert found == []
            assert selectable_models(found) == []
            assert resolve_model(discovered=found) is None

    def test_two_genuine_models_stay_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "alpha-7B-Q4.gguf").write_bytes(b"x" * 100)
            (Path(tmp) / "beta-7B-Q4.gguf").write_bytes(b"x" * 100)
            found = discover_models([tmp])
            self.assertEqual(selectable_models(found), [])
            self.assertIsNone(resolve_model(discovered=found))

    def test_gguf_metadata_clip_is_projector(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "real-7B-Q4.gguf"
            proj = Path(tmp) / "vision.gguf"
            _write_gguf(model, [("general.architecture", "qwen35"), ("general.type", "model")])
            _write_gguf(proj, [("general.architecture", "clip"), ("general.type", "mmproj")])
            found = discover_models([tmp])
            self.assertEqual(found, [])
            self.assertIsNone(resolve_model(discovered=found))


if __name__ == "__main__":
    unittest.main()
