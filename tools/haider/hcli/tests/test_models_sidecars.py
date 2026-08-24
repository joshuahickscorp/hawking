from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

from hcli.models import (
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
    def test_filename_heuristic_mmproj_not_selectable(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf").write_bytes(b"x" * 100)
            (Path(tmp) / "mmproj-model-bf16.gguf").write_bytes(b"x" * 100)
            found = discover_models([tmp])
            self.assertEqual(len(found), 2)
            selectable = selectable_models(found)
            self.assertEqual(len(selectable), 1)
            self.assertFalse(selectable[0].is_projector)
            self.assertIn("Qwen", selectable[0].name)
            projectors = [m for m in found if m.is_projector]
            self.assertEqual(len(projectors), 1)
            self.assertTrue(projectors[0].is_projector)
            chosen = resolve_model(discovered=found)
            self.assertIsNotNone(chosen)
            self.assertEqual(chosen.path, selectable[0].path)

    def test_two_genuine_models_stay_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "alpha-7B-Q4.gguf").write_bytes(b"x" * 100)
            (Path(tmp) / "beta-7B-Q4.gguf").write_bytes(b"x" * 100)
            found = discover_models([tmp])
            self.assertEqual(len(selectable_models(found)), 2)
            self.assertIsNone(resolve_model(discovered=found))

    def test_gguf_metadata_clip_is_projector(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "real-7B-Q4.gguf"
            proj = Path(tmp) / "vision.gguf"
            _write_gguf(model, [("general.architecture", "qwen35"), ("general.type", "model")])
            _write_gguf(proj, [("general.architecture", "clip"), ("general.type", "mmproj")])
            found = discover_models([tmp])
            by_name = {m.name: m for m in found}
            self.assertFalse(by_name["real-7B-Q4.gguf"].is_projector)
            self.assertTrue(by_name["vision.gguf"].is_projector)
            chosen = resolve_model(discovered=found)
            self.assertIsNotNone(chosen)
            self.assertEqual(chosen.name, "real-7B-Q4.gguf")


if __name__ == "__main__":
    unittest.main()
