from __future__ import annotations

import importlib
import pkgutil
import tempfile
import unittest
from pathlib import Path

import hcli
from hcli.app import App
from hcli.controller import Controller
from hcli.events import EventBus
from hcli.models import ModelRegistry


class TestLiveStartBoundary(unittest.TestCase):
    def test_every_hcli_module_imports(self):
        failures = []

        for item in pkgutil.iter_modules(hcli.__path__):
            name = "hcli." + item.name

            try:
                importlib.import_module(name)
            except BaseException as exc:
                failures.append(
                    (name, type(exc).__name__, str(exc))
                )

        self.assertEqual(failures, [])

    def test_model_registry_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry([tmp])

            self.assertEqual(
                registry.discover(),
                [],
            )

            self.assertIsNone(
                registry.resolve(),
            )

    def test_controller_matches_app_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            bus = EventBus()

            controller = Controller(
                workspace=tmp,
                runtime_count=3,
                model=None,
                bus=bus,
                registry=ModelRegistry([tmp]),
            )

            self.assertEqual(
                controller.runtime_count,
                3,
            )

            self.assertIsNone(controller.runtime_pool)
            self.assertEqual(
                controller.status()["model_name"],
                controller.model_name,
            )

            status = controller.status()

            self.assertEqual(
                status["requested_runtimes"],
                3,
            )

            self.assertEqual(
                status["admitted_runtimes"],
                0,
            )

            controller.shutdown()

    def test_app_constructs_without_loading_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = App(
                workspace=tmp,
                runtime_count=3,
                model=None,
                debug=False,
            )

            self.assertEqual(
                app.runtime_count,
                3,
            )

            self.assertEqual(
                app.controller.runtime_count,
                3,
            )

            # Construction must not load multi-GB models merely to open HCLI.
            self.assertIsNone(
                app.controller.runtime_pool,
            )

            app.controller.shutdown()


if __name__ == "__main__":
    unittest.main()
