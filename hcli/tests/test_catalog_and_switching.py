"""Picking a body, and becoming it.

The failures guarded here are all SILENT ones: a menu that resolves a name to
whichever body was walked first, a switch that answers as a different model than
the caller selected, a swap that holds two multi-gigabyte residents at once, and
our own catalog name leaking into a backend that reads it as a HuggingFace repo
id and goes to the network for it.
"""
from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import unittest
from pathlib import Path

from hcli.catalog import Body, _deduplicate, _modellake, _native_profiles, catalog, resolve
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
    def test_native_admission_binds_actions_to_exact_profile_bytes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            profile = Path(root) / "hawking-native.test.json"
            profile.write_text(json.dumps({
                "resident_identity": "sealed-test",
                "profile_schema": "hcli.provider.profile.v1",
                "provider": "native",
                "runtime": "hawking-native",
                "qualification": "QUALIFIED_REFERENCE_PATH",
                "admission": {
                    "status": "ADMITTED",
                    "contract": "reference_path",
                    "evidence": ["fixtures/reference.json"],
                },
            }), encoding="utf-8")
            body = _native_profiles(Path(root))[0]
            self.assertTrue(body.admitted)
            self.assertEqual(body.revision, hashlib.sha256(profile.read_bytes()).hexdigest())
            self.assertEqual(body.supported_actions, ("execute", "serve", "web"))

    def test_rejected_native_profile_has_no_actions(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            profile = Path(root) / "hawking-native.rejected.json"
            profile.write_text(json.dumps({
                "resident_identity": "rejected-test",
                "profile_schema": "hcli.provider.profile.v1",
                "provider": "native",
                "runtime": "hawking-native",
                "qualification": "UNQUALIFIED_CANDIDATE",
                "admission": {
                    "status": "REJECTED",
                    "contract": "reference_path",
                    "evidence": ["fixtures/rejected.json"],
                },
            }), encoding="utf-8")
            body = _native_profiles(Path(root))[0]
            self.assertFalse(body.admitted)
            self.assertEqual(body.supported_actions, ())

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

    def test_raw_modellake_directory_is_a_specimen_not_an_admitted_body(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            specimen = Path(root) / "Org--Model@abcdef123456"
            specimen.mkdir()
            (specimen / "config.json").write_text("{}", encoding="utf-8")
            rows = _modellake(Path(root))
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0].admitted)
            self.assertIn("no qualified Hawking execution binding", rows[0].admission_reason)

    def test_explicit_unqualified_directory_is_excluded_from_normal_catalog(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            specimen = Path(root) / "candidate"
            specimen.mkdir()
            (specimen / "config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(catalog([str(specimen)]), [
                body for body in catalog() if body.admitted
            ])


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


class TestTerminologyGuard(unittest.TestCase):
    def test_guard_selfcheck_runs_in_normal_python_suite(self):
        repo = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, str(repo / "tools/verify/hawking_terminology.py"), "--selfcheck"],
            cwd=repo,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
