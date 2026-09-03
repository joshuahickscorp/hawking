from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[2]

from hcli.config import Config
from hcli.controller import Controller
from hcli.events import EventBus
from hcli.models import ModelRegistry, discover_models


class TestConfigPersist(unittest.TestCase):
    def test_project_overrides_global(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            global_path = root / "global.json"
            ws = root / "ws"
            ws.mkdir()
            (root / "global.gguf").write_bytes(b"x")
            (root / "project.gguf").write_bytes(b"x")
            global_path.write_text(json.dumps({"model": str(root / "global.gguf")}))
            cfg = Config(str(ws), global_path=str(global_path))
            cfg.save_project({"model": str(root / "project.gguf")})
            loaded = cfg.load()
            self.assertEqual(loaded["model"], str(root / "project.gguf"))
            self.assertEqual(cfg.layer_model(cfg.global_path), str(root / "global.gguf"))
            self.assertEqual(cfg.layer_model(cfg.project_path), str(root / "project.gguf"))

    def test_select_model_writes_global_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir()
            (models_dir / "only-7B-Q4.gguf").write_bytes(b"x" * 50)
            global_path = root / "home" / ".config" / "hcli" / "config.json"
            ws = root / "ws"
            ws.mkdir()
            bus = EventBus()
            controller = Controller(
                workspace=str(ws),
                runtime_count=1,
                model=None,
                bus=bus,
                registry=ModelRegistry([str(models_dir)]),
            )
            controller.config.global_path = str(global_path)
            found = discover_models([str(models_dir)])
            self.assertEqual(len(found), 1)
            ok = controller.select_model("1")
            self.assertTrue(ok)
            self.assertTrue(global_path.is_file())
            data = json.loads(global_path.read_text())
            self.assertEqual(data["model"], found[0].path)
            controller.shutdown()

    def test_resolve_reads_global_when_discovery_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir()
            a = models_dir / "alpha-7B-Q4.gguf"
            b = models_dir / "beta-7B-Q4.gguf"
            a.write_bytes(b"x" * 50)
            b.write_bytes(b"x" * 50)
            global_path = root / "cfg.json"
            global_path.write_text(json.dumps({"model": str(a.resolve())}))
            ws = root / "ws"
            ws.mkdir()
            registry = ModelRegistry([str(models_dir)])
            controller = Controller(
                workspace=str(ws),
                runtime_count=1,
                model=None,
                bus=MagicMock(),
                registry=registry,
            )
            controller.config.global_path = str(global_path)
            resolved = registry.resolve(
                project_config=controller.config.layer_model(controller.config.project_path),
                global_config=controller.config.layer_model(controller.config.global_path),
            )
            self.assertIsNotNone(resolved)
            self.assertEqual(Path(resolved.path), a.resolve())
            controller.shutdown()


if __name__ == "__main__":
    unittest.main()
