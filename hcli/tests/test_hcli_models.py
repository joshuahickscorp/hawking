from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from hcli.models import (
    ModelInfo,
    discover_models,
    resolve_model,
    _infer_family,
    _infer_param_class,
    _infer_quantization,
)


class TestInference(unittest.TestCase):
    def test_family_qwen(self):
        self.assertEqual(_infer_family("Qwen3.8-27B-Q5_K_M.gguf"), "Qwen")

    def test_family_llama(self):
        self.assertEqual(_infer_family("llama-3-8b-instruct.gguf"), "Llama")

    def test_param_class(self):
        self.assertEqual(_infer_param_class("Qwen3.8-27B-Q5.gguf"), "27B")

    def test_param_class_float(self):
        self.assertEqual(_infer_param_class("model-7.5B.gguf"), "7.5B")

    def test_quantization(self):
        self.assertEqual(_infer_quantization("Qwen3.8-27B-Q5_K_M.gguf"), "Q5_K_M")

    def test_quantization_f16(self):
        self.assertEqual(_infer_quantization("model-f16.gguf"), "F16")


class TestDiscover(unittest.TestCase):
    def test_discovers_gguf(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test-model-7B-Q4.gguf"
            p.write_bytes(b"x" * 100)
            models = discover_models([tmp])
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0].family, "Unknown")
            self.assertEqual(models[0].param_class, "7B")
            self.assertEqual(models[0].quantization, "Q4")

    def test_ignores_non_gguf(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "readme.txt").write_text("hi")
            models = discover_models([tmp])
            self.assertEqual(len(models), 0)

    def test_dedup_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp, "real.gguf")
            real.write_bytes(b"x")
            link = Path(tmp, "link.gguf")
            os.symlink(real, link)
            models = discover_models([tmp])
            self.assertEqual(len(models), 1)


class TestResolve(unittest.TestCase):
    def test_explicit_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, "m.gguf")
            p.write_bytes(b"x")
            info = resolve_model(explicit=str(p))
            self.assertIsNotNone(info)
            self.assertEqual(info.path, str(p.resolve()))

    def test_single_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "only.gguf").write_bytes(b"x")
            info = resolve_model(discovered=discover_models([tmp]))
            self.assertIsNotNone(info)

    def test_ambiguous_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.gguf").write_bytes(b"x")
            Path(tmp, "b.gguf").write_bytes(b"x")
            info = resolve_model(discovered=discover_models([tmp]))
            self.assertIsNone(info)

    def test_never_silently_uses_model_gguf(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "model.gguf").write_bytes(b"x")
            Path(tmp, "other.gguf").write_bytes(b"x")
            info = resolve_model(discovered=discover_models([tmp]))
            self.assertIsNone(info)


if __name__ == "__main__":
    unittest.main()
