"""Picking a body, and becoming it.

The failures guarded here are all SILENT ones: a menu that resolves a name to
whichever body was walked first, a switch that answers as a different model than
the caller selected, a swap that holds two multi-gigabyte residents at once, and
our own catalog name leaking into a backend that reads it as a HuggingFace repo
id and goes to the network for it.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from hcli.catalog import Body, _deduplicate, catalog, resolve
from hcli.serve import Resident


def _body(name, path="/tmp/x", kind="mlx", size=0, revision="", source="modellake"):
    return Body(name=name, path=path, kind=kind, bytes=size,
                revision=revision, source=source)


class _FakeBackend:
    def __init__(self, name="a", fail_ready=False):
        self.name = name
        self.stopped = False
        self.spawned = False
        self.seen = []
        self.fail_ready = fail_ready

    def spawn(self, **kw):
        self.spawned = True

    def ready(self, timeout):
        return not self.fail_ready

    def stop(self):
        self.stopped = True
        return {}

    def complete(self, payload, timeout=None):
        self.seen.append(payload)
        return type("R", (), {"text": f"from {self.name}", "finish_reason": "stop",
                              "prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2, "degraded": [], "raw": {}})()


class TestNamesDoNotCollideSilently(unittest.TestCase):
    def test_a_collision_qualifies_both_rather_than_shadowing_one(self):
        rows = _deduplicate([
            _body("Qwen3-4B", revision="aaaaaaaaaaaa"),
            _body("Qwen3-4B", revision="bbbbbbbbbbbb"),
            _body("Qwen3-14B", revision="cccccccccccc"),
        ])
        names = [b.name for b in rows]
        self.assertEqual(len(set(names)), 3, names)
        self.assertIn("Qwen3-14B", names)
        self.assertNotIn("Qwen3-4B", names, "one revision silently won the name")

    def test_exact_name_beats_prefix(self):
        bodies = [_body("Qwen3-4B"), _body("Qwen3-4B-Instruct")]
        self.assertEqual(resolve("Qwen3-4B", bodies).name, "Qwen3-4B")

    def test_an_ambiguous_prefix_is_refused_and_names_the_candidates(self):
        bodies = [_body("Qwen3-14B"), _body("Qwen3-30B")]
        with self.assertRaises(LookupError) as caught:
            resolve("Qwen3", bodies)
        self.assertIn("Qwen3-14B", str(caught.exception))
        self.assertIn("Qwen3-30B", str(caught.exception))

    def test_an_unknown_name_is_none_not_a_guess(self):
        self.assertIsNone(resolve("nope", [_body("Qwen3-14B")]))

    def test_a_path_resolves_to_its_body(self):
        bodies = [_body("Qwen3-14B", path="/tmp")]
        self.assertEqual(resolve("/tmp", bodies).name, "Qwen3-14B")


class TestGravityCatalogBoundary(unittest.TestCase):
    def test_normal_catalog_exposes_admitted_gravity_bodies_only(self):
        import tempfile
        import hcli.catalog as catalog_mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = root / "kimi"
            body.mkdir()
            (body / "config.json").write_text("{}")
            (body / "model.safetensors").write_bytes(b"fixture")
            registry = root / "gravity-artifacts.json"
            registry.write_text(json.dumps({"artifacts": [{
                "id": "KIMI_P0_OPERATIONAL",
                "path": str(body),
                "kind": "mlx",
                "status": "OPERATIONAL_DEVELOPMENTAL",
                "role": "developmental resident",
            }]}))
            lake = root / "lake"
            lake.mkdir()
            specimen = lake / "raw-model@abc"
            specimen.mkdir()
            (specimen / "config.json").write_text("{}")
            (specimen / "model.safetensors").write_bytes(b"fixture")
            old_registry, old_lake = catalog_mod.GRAVITY_REGISTRY, catalog_mod.MODELLAKE
            catalog_mod.GRAVITY_REGISTRY, catalog_mod.MODELLAKE = registry, lake
            try:
                normal = catalog()
                research = catalog(research=True)
            finally:
                catalog_mod.GRAVITY_REGISTRY, catalog_mod.MODELLAKE = old_registry, old_lake
            normal_names = [row.name for row in normal]
            research_names = [row.name for row in research]
            self.assertIn("KIMI_P0_OPERATIONAL", normal_names)
            self.assertNotIn("raw-model", normal_names)
            self.assertIn("KIMI_P0_OPERATIONAL", research_names)
            self.assertIn("raw-model", research_names)


class TestSwitching(unittest.TestCase):
    def setUp(self):
        self.backend = _FakeBackend("first")
        self.resident = Resident(_body("first", path="/tmp/first"), self.backend)

    def test_our_catalog_name_never_reaches_the_backend(self):
        # THE DEFECT THIS EXISTS FOR: mlx_lm.server reads `model` as a repo id
        # and fetched https://huggingface.co/api/models/Qwen3-0.6B, so a correct
        # switch came back "Repository Not Found".
        self.resident.complete({"model": "first", "messages": [{"role": "user",
                                                                "content": "hi"}]})
        self.assertNotIn("model", self.backend.seen[-1])
        self.assertIn("messages", self.backend.seen[-1])

    def test_an_unknown_model_is_refused_not_answered_by_the_loaded_one(self):
        with self.assertRaises(LookupError) as caught:
            self.resident.complete({"model": "gpt-4o", "messages": []})
        self.assertIn("first", str(caught.exception),
                      "the refusal does not say what IS loaded")
        self.assertEqual(self.backend.seen, [], "it answered as the wrong body")

    def test_no_model_field_uses_the_loaded_body(self):
        self.resident.complete({"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(len(self.backend.seen), 1)

    def test_a_body_too_big_for_the_machine_is_refused_with_both_numbers(self):
        huge = _body("Enormous", size=10 ** 15)
        refusal = self.resident.admits(huge)
        self.assertIsNotNone(refusal)
        self.assertIn("Enormous", refusal)
        self.assertIn("GB", refusal)

    def test_a_body_that_fits_is_admitted(self):
        self.assertIsNone(self.resident.admits(_body("Small", size=10 ** 8)))

    def test_switching_to_the_loaded_body_is_a_no_op(self):
        # Re-selecting the model already answering must not tear down a loaded
        # resident and pay the load again -- which is what a browser does every
        # time someone opens the dropdown and picks the current entry.
        import hcli.catalog as catalog_mod
        real = catalog_mod.resolve
        catalog_mod.resolve = lambda name, bodies=None: _body("first", path="/tmp/first")
        try:
            got = self.resident.switch("first")
        finally:
            catalog_mod.resolve = real
        self.assertFalse(got["switched"])
        self.assertFalse(self.backend.stopped, "a no-op switch stopped the resident")


class TestSwitchStopsBeforeStarting(unittest.TestCase):
    """Two residents alive at once would page this machine into the ground."""

    def test_the_old_backend_is_stopped_before_the_new_one_spawns(self):
        import hcli.serve as serve
        order = []
        old = _FakeBackend("old")
        new = _FakeBackend("new")

        class _Watched(_FakeBackend):
            pass

        old.stop = lambda: order.append("stop old")            # type: ignore
        target = _body("new", path="/tmp/new")

        def fake_resolve(name, bodies=None):
            return target

        def fake_make(path, **kw):
            order.append("spawn new")
            return new

        resident = Resident(_body("old", path="/tmp/old"), old)
        import hcli.catalog as catalog_mod
        import hcli.runtime_iface as iface
        real_resolve, real_make = catalog_mod.resolve, iface.make_backend_for_model
        catalog_mod.resolve, iface.make_backend_for_model = fake_resolve, fake_make
        try:
            got = resident.switch("new")
        finally:
            catalog_mod.resolve, iface.make_backend_for_model = real_resolve, real_make
        self.assertTrue(got["switched"])
        self.assertEqual(order, ["stop old", "spawn new"],
                         "the new resident spawned while the old one was still loaded")
        self.assertEqual(resident.identity, "new")

    def test_research_switch_resolves_only_in_research_catalog(self):
        import hcli.catalog as catalog_mod
        import hcli.runtime_iface as iface
        old = _FakeBackend("old")
        new = _FakeBackend("new")
        target = _body("LFM2-24B-A2B", path="/tmp/lfm2")
        seen = []

        def fake_resolve(name, bodies=None, *, research=False):
            seen.append((name, research))
            return target if research else None

        real_resolve, real_make = catalog_mod.resolve, iface.make_backend_for_model
        catalog_mod.resolve = fake_resolve
        iface.make_backend_for_model = lambda path, **kw: new
        resident = Resident(_body("old", path="/tmp/old"), old)
        try:
            got = resident.switch("LFM2-24B-A2B", research=True)
        finally:
            catalog_mod.resolve, iface.make_backend_for_model = real_resolve, real_make
        self.assertTrue(got["switched"])
        self.assertTrue(got["research"])
        self.assertEqual(seen, [("LFM2-24B-A2B", True)])
        self.assertFalse(resident.tools_verdict["qualified"])
        self.assertIn("research body", resident.tools_verdict["reason"])


if __name__ == "__main__":
    unittest.main()
