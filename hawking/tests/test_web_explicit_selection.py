"""An explicit Web artifact must re-prove its live binding before reuse."""

from __future__ import annotations

import unittest

from hawking.catalog import Body, resolve_action
from hawking.web import reconcile_requested_resident, revalidate_resident_for_web


def _body(name: str, path: str, revision: str) -> Body:
    return Body(
        name=name,
        path=path,
        kind="mlx",
        revision=revision,
        source="fixture",
        admitted=True,
        supported_actions=("execute", "serve", "web"),
    )


class _CatalogFixture:
    def __init__(self, *bodies: Body):
        self.bodies = list(bodies)
        self.original = None

    def __enter__(self):
        import hawking.catalog as catalog_module

        self.original = catalog_module.catalog
        catalog_module.catalog = lambda extra_roots=None, *, research=False: list(self.bodies)
        return self

    def __exit__(self, *_args):
        import hawking.catalog as catalog_module

        catalog_module.catalog = self.original


class TestExplicitWebSelection(unittest.TestCase):
    def test_same_current_binding_reuses_without_switch(self):
        body = _body("loaded", "/tmp/model.json", "a" * 64)
        action = resolve_action(body.name, "web", [body])
        existing = {
            "resident": body.name,
            "model": body.path,
            "resolved_action": action.to_wire(),
        }
        with _CatalogFixture(body):
            self.assertIs(
                reconcile_requested_resident(
                    existing, action, "http://127.0.0.1:8011/v1", 1.0
                ),
                existing,
            )

    def test_same_binding_serve_contract_posts_the_web_action(self):
        import hawking.use as use_module

        body = _body("loaded", "/tmp/model.json", "a" * 64)
        serve_action = resolve_action(body.name, "serve", [body])
        web_action = resolve_action(body.name, "web", [body])
        calls = []
        original_post = use_module.post_json
        use_module.post_json = lambda url, envelope, timeout: (
            calls.append((url, envelope, timeout)) or (
                200,
                {
                    "resident": body.name,
                    "switched": False,
                    "resolved_action": web_action.to_wire(),
                },
            )
        )
        try:
            with _CatalogFixture(body):
                updated = reconcile_requested_resident(
                    {
                        "resident": body.name,
                        "model": body.path,
                        "resolved_action": serve_action.to_wire(),
                    },
                    web_action,
                    "http://127.0.0.1:8011/v1",
                    7.0,
                )
        finally:
            use_module.post_json = original_post

        self.assertEqual(
            calls,
            [
                (
                    "http://127.0.0.1:8011/v1/switch",
                    {"resolved_action": web_action.to_wire()},
                    7.0,
                )
            ],
        )
        self.assertEqual(updated["resolved_action"], web_action.to_wire())

    def test_implicit_web_reuse_transitions_a_serve_contract(self):
        import hawking.use as use_module

        body = _body("loaded", "/tmp/model.json", "a" * 64)
        serve_action = resolve_action(body.name, "serve", [body])
        web_action = resolve_action(body.name, "web", [body])
        calls = []
        original_post = use_module.post_json
        use_module.post_json = lambda url, envelope, timeout: (
            calls.append((url, envelope, timeout)) or (
                200,
                {
                    "resident": body.name,
                    "switched": False,
                    "resolved_action": web_action.to_wire(),
                },
            )
        )
        try:
            with _CatalogFixture(body):
                updated = revalidate_resident_for_web(
                    {
                        "resident": body.name,
                        "model": body.path,
                        "resolved_action": serve_action.to_wire(),
                    },
                    base_url="http://127.0.0.1:8011/v1",
                    timeout=7.0,
                )
        finally:
            use_module.post_json = original_post

        self.assertEqual(
            calls,
            [
                (
                    "http://127.0.0.1:8011/v1/switch",
                    {"resolved_action": web_action.to_wire()},
                    7.0,
                )
            ],
        )
        self.assertEqual(updated["resolved_action"], web_action.to_wire())

    def test_different_admitted_artifact_switches_with_full_envelope(self):
        import hawking.use as use_module

        loaded = _body("loaded", "/tmp/loaded.json", "a" * 64)
        target = _body("wanted", "/tmp/wanted.json", "b" * 64)
        target_action = resolve_action(target.name, "web", [target])
        calls = []
        original_post = use_module.post_json
        use_module.post_json = lambda url, body, timeout: (
            calls.append((url, body, timeout)) or (
                200,
                {
                    "resident": target.name,
                    "switched": True,
                    "resolved_action": target_action.to_wire(),
                },
            )
        )
        try:
            with _CatalogFixture(loaded, target):
                updated = reconcile_requested_resident(
                    {
                        "resident": loaded.name,
                        "model": loaded.path,
                        "resolved_action": resolve_action(
                            loaded.name, "serve", [loaded]
                        ).to_wire(),
                    },
                    target_action,
                    "http://127.0.0.1:8011/v1",
                    7.0,
                )
        finally:
            use_module.post_json = original_post
        self.assertEqual(updated["resident"], target.name)
        self.assertEqual(updated["model"], target_action.path)
        self.assertEqual(
            calls,
            [
                (
                    "http://127.0.0.1:8011/v1/switch",
                    {"resolved_action": target_action.to_wire()},
                    7.0,
                )
            ],
        )

    def test_same_path_with_stale_binding_switches_instead_of_reusing(self):
        import hawking.use as use_module

        old = _body("loaded", "/tmp/model.json", "a" * 64)
        current = _body("loaded", "/tmp/model.json", "b" * 64)
        old_action = resolve_action(old.name, "web", [old])
        current_action = resolve_action(current.name, "web", [current])
        calls = []
        original_post = use_module.post_json
        use_module.post_json = lambda url, body, timeout: (
            calls.append((url, body, timeout)) or (200, {"resident": current.name})
        )
        try:
            with _CatalogFixture(current):
                reconcile_requested_resident(
                    {
                        "resident": old.name,
                        "model": old.path,
                        "resolved_action": old_action.to_wire(),
                    },
                    current_action,
                    "http://127.0.0.1:8011/v1",
                    1.0,
                )
        finally:
            use_module.post_json = original_post
        self.assertEqual(calls[0][1], {"resolved_action": current_action.to_wire()})

    def test_unknown_artifact_refuses_instead_of_reusing_wrong_resident(self):
        with _CatalogFixture():
            with self.assertRaisesRegex(RuntimeError, "not an admitted catalog identity"):
                reconcile_requested_resident(
                    {"resident": "wrong", "model": "/tmp/wrong.json"},
                    "unknown",
                    "http://127.0.0.1:8011/v1",
                    1.0,
                )


if __name__ == "__main__":
    unittest.main()
