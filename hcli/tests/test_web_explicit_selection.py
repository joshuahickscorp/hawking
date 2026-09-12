"""An explicit Web artifact may never silently reuse a different resident."""

from __future__ import annotations

import unittest

from hcli.catalog import Body
from hcli.web import reconcile_requested_resident


class TestExplicitWebSelection(unittest.TestCase):
    def test_same_exact_path_reuses_without_switch(self):
        existing = {"resident": "loaded", "model": "/tmp/model.json"}
        self.assertIs(
            reconcile_requested_resident(
                existing, "/tmp/model.json", "http://127.0.0.1:8011/v1", 1.0
            ),
            existing,
        )

    def test_different_admitted_artifact_switches_before_reuse(self):
        import hcli.catalog as catalog_module
        import hcli.use as use_module

        target = Body(
            name="wanted@revision",
            path="/tmp/wanted.json",
            kind="noetic_native",
            revision="revision",
            supported_actions=("execute", "serve", "web"),
        )
        original_catalog = catalog_module.catalog
        original_resolve = catalog_module.resolve
        original_post = use_module.post_json
        calls = []
        catalog_module.catalog = lambda extra_roots=None: [target]
        catalog_module.resolve = lambda name, bodies=None: target
        use_module.post_json = lambda url, body, timeout: (
            calls.append((url, body, timeout)) or (200, {"resident": target.name, "switched": True})
        )
        try:
            updated = reconcile_requested_resident(
                {"resident": "wrong", "model": "/tmp/wrong.json"},
                target.name,
                "http://127.0.0.1:8011/v1",
                7.0,
            )
        finally:
            catalog_module.catalog = original_catalog
            catalog_module.resolve = original_resolve
            use_module.post_json = original_post
        self.assertEqual(updated["resident"], target.name)
        self.assertEqual(updated["model"], target.path)
        self.assertEqual(
            calls,
            [("http://127.0.0.1:8011/v1/switch", {"model": target.path}, 7.0)],
        )

    def test_unknown_artifact_refuses_instead_of_reusing_wrong_resident(self):
        import hcli.catalog as catalog_module

        original_catalog = catalog_module.catalog
        original_resolve = catalog_module.resolve
        catalog_module.catalog = lambda extra_roots=None: []
        catalog_module.resolve = lambda name, bodies=None: None
        try:
            with self.assertRaisesRegex(RuntimeError, "not an admitted catalog identity"):
                reconcile_requested_resident(
                    {"resident": "wrong", "model": "/tmp/wrong.json"},
                    "unknown",
                    "http://127.0.0.1:8011/v1",
                    1.0,
                )
        finally:
            catalog_module.catalog = original_catalog
            catalog_module.resolve = original_resolve


if __name__ == "__main__":
    unittest.main()
