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
import urllib.error
from pathlib import Path

from hawking.catalog import Body, _deduplicate, _modellake, _native_profiles, catalog, resolve
from hawking.serve import Resident


def _body(name, path="/tmp/x", kind="mlx", size=0, revision="", source="modellake"):
    return Body(name=name, path=path, kind=kind, bytes=size,
                revision=revision, source=source)


def _admitted_body(name, path="/tmp/x", kind="mlx", size=0, revision="a" * 64):
    return Body(
        name=name,
        path=path,
        kind=kind,
        bytes=size,
        revision=revision,
        source="fixture",
        admitted=True,
        supported_actions=("execute", "serve", "web"),
    )


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
        return {"gone": True}

    def complete(self, payload, timeout=None):
        self.seen.append(payload)
        return type("R", (), {"text": f"from {self.name}", "finish_reason": "stop",
                              "prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2, "degraded": [], "raw": {}})()


class _OwnedProcess:
    def __init__(self, exit_code=None):
        self.exit_code = exit_code

    def poll(self):
        return self.exit_code


class _OwnedFakeBackend(_FakeBackend):
    def __init__(self, name="a", *, exit_code=None, fail_request=False,
                 fail_ready=False):
        super().__init__(name, fail_ready=fail_ready)
        self.process = _OwnedProcess(exit_code)
        self.fail_request = fail_request

    def complete(self, payload, timeout=None):
        self.seen.append(payload)
        if self.fail_request:
            raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
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
                "profile_schema": "hawking.provider.profile.v1",
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
                "profile_schema": "hawking.provider.profile.v1",
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


class TestGravityCatalogBoundary(unittest.TestCase):
    def test_normal_catalog_exposes_admitted_gravity_bodies_only(self):
        import tempfile
        import hawking.catalog as catalog_mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = root / "kimi"
            body.mkdir()
            (body / "config.json").write_text("{}")
            (body / "model.safetensors").write_bytes(b"fixture")
            registry = root / "gravity-artifacts.json"
            registry.write_text(json.dumps({
                "schema": "hawking.gravity.admitted_artifact_registry.v1",
                "status": "ACTIVE",
                "artifacts": [{
                "id": "KIMI_P0_OPERATIONAL",
                "path": str(body),
                "kind": "mlx",
                "status": "OPERATIONAL_DEVELOPMENTAL",
                "revision": "a" * 64,
                "revision_basis": "fixture artifact bytes SHA-256",
                "supported_actions": ["execute", "serve", "web"],
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
            # Local Kimi P0 remains a research/review identity, not an
            # admitted executable body.  The normal catalog must stay honest.
            self.assertNotIn("KIMI_P0_OPERATIONAL", normal_names)
            self.assertNotIn("raw-model", normal_names)
            # MLX registry rows remain historical source data even in the
            # explicit research catalog; only ModelLake specimens are exposed
            # there, and neither path grants a live execution identity.
            self.assertNotIn("KIMI_P0_OPERATIONAL", research_names)
            self.assertIn("raw-model", research_names)
            self.assertEqual(normal, [])

    def test_registry_entry_without_exact_revision_is_not_executable(self):
        import tempfile
        import hawking.catalog as catalog_mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = root / "body"
            body.mkdir()
            (body / "config.json").write_text("{}")
            (body / "model.safetensors").write_bytes(b"fixture")
            registry = root / "gravity-artifacts.json"
            registry.write_text(json.dumps({
                "schema": "hawking.gravity.admitted_artifact_registry.v1",
                "status": "ACTIVE",
                "artifacts": [{
                "id": "UNBOUND",
                "path": str(body),
                "kind": "mlx",
                "status": "ADMITTED",
                "revision_basis": "fixture artifact bytes SHA-256",
                "supported_actions": ["serve"],
            }]}))
            self.assertEqual(catalog_mod._gravity_artifacts(registry), [])


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

    def test_model_lock_refuses_a_switch_before_catalog_resolution(self):
        with self.assertRaises(PermissionError):
            self.resident.complete(
                {"model": "other", "messages": [{"role": "user", "content": "hi"}]},
                model_lock="first",
            )
        self.assertEqual(self.backend.seen, [])

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
        import hawking.catalog as catalog_mod
        from hawking.catalog import resolve_action

        body = _admitted_body("first", path="/tmp/first")
        action = resolve_action("first", "serve", [body])
        self.resident = Resident(body, self.backend, resolved_action=action)
        real = catalog_mod.catalog
        catalog_mod.catalog = lambda extra_roots=None, *, research=False: [body]
        try:
            got = self.resident.switch("first")
        finally:
            catalog_mod.catalog = real
        self.assertFalse(got["switched"])
        self.assertFalse(self.backend.stopped, "a no-op switch stopped the resident")


class TestSwitchStopsBeforeStarting(unittest.TestCase):
    """Two residents alive at once would page this machine into the ground."""

    def test_the_old_backend_is_stopped_before_the_new_one_spawns(self):
        import hawking.serve as serve
        order = []
        old = _FakeBackend("old")
        new = _FakeBackend("new")

        class _Watched(_FakeBackend):
            pass

        def _stop_old():
            order.append("stop old")
            return {"gone": True}

        old.stop = _stop_old                                      # type: ignore
        target = _admitted_body("new", path="/tmp/new")

        def fake_make(path, **kw):
            order.append("spawn new")
            return new

        resident = Resident(_body("old", path="/tmp/old"), old)
        import hawking.catalog as catalog_mod
        import hawking.runtime_iface as iface
        real_catalog, real_make = catalog_mod.catalog, iface.make_backend_for_model
        catalog_mod.catalog = lambda extra_roots=None, *, research=False: [target]
        iface.make_backend_for_model = fake_make
        try:
            got = resident.switch("new")
        finally:
            catalog_mod.catalog, iface.make_backend_for_model = real_catalog, real_make
        self.assertTrue(got["switched"])
        self.assertEqual(order, ["stop old", "spawn new"],
                         "the new resident spawned while the old one was still loaded")
        self.assertEqual(resident.identity, "new")

    def test_unproven_old_stop_refuses_to_construct_a_candidate(self):
        import hawking.catalog as catalog_mod
        import hawking.runtime_iface as iface

        old = _FakeBackend("old")
        old.stop = lambda: {"gone": False}                       # type: ignore
        target = _admitted_body("new", path="/tmp/new")
        resident = Resident(_admitted_body("old", path="/tmp/old"), old)
        original_catalog, original_make = catalog_mod.catalog, iface.make_backend_for_model
        catalog_mod.catalog = lambda extra_roots=None, *, research=False: [target]
        made = []
        iface.make_backend_for_model = lambda path, **kw: made.append(path)
        try:
            with self.assertRaisesRegex(RuntimeError, "did not prove exit"):
                resident.switch("new")
        finally:
            catalog_mod.catalog, iface.make_backend_for_model = original_catalog, original_make

        self.assertEqual(made, [])
        self.assertIs(resident.backend, old)
        self.assertEqual(resident.identity, "old")

    def test_old_stop_exception_refuses_to_construct_a_candidate(self):
        import hawking.catalog as catalog_mod
        import hawking.runtime_iface as iface

        old = _FakeBackend("old")

        def _broken_stop():
            raise OSError("still alive")

        old.stop = _broken_stop                                    # type: ignore
        target = _admitted_body("new", path="/tmp/new")
        resident = Resident(_admitted_body("old", path="/tmp/old"), old)
        original_catalog, original_make = catalog_mod.catalog, iface.make_backend_for_model
        catalog_mod.catalog = lambda extra_roots=None, *, research=False: [target]
        made = []
        iface.make_backend_for_model = lambda path, **kw: made.append(path)
        try:
            with self.assertRaisesRegex(RuntimeError, "stop failed"):
                resident.switch("new")
        finally:
            catalog_mod.catalog, iface.make_backend_for_model = original_catalog, original_make

        self.assertEqual(made, [])
        self.assertIs(resident.backend, old)
        self.assertEqual(resident.identity, "old")

    def test_research_switch_resolves_only_in_research_catalog(self):
        import hawking.catalog as catalog_mod
        import hawking.runtime_iface as iface
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


class TestOwnedProviderRecovery(unittest.TestCase):
    """A dead daemon-owned child must not masquerade as a ready resident."""

    def _with_same_body_factory(self, resident, replacement):
        import hawking.catalog as catalog_mod
        import hawking.runtime_iface as iface

        original_body = resident.body
        original_catalog = catalog_mod.catalog
        original_make = iface.make_backend_for_model
        catalog_mod.catalog = lambda extra_roots=None, *, research=False: [original_body]
        iface.make_backend_for_model = lambda path, **kw: replacement
        self.addCleanup(setattr, catalog_mod, "catalog", original_catalog)
        self.addCleanup(setattr, iface, "make_backend_for_model", original_make)
        # Qualification is separately covered; this lifecycle test must only
        # prove provider ownership and request recovery.
        resident.qualify_tools = lambda *args, **kw: resident.tools_verdict

    def test_exited_same_body_child_is_recreated_before_completion(self):
        stale = _OwnedFakeBackend("stale", exit_code=17)
        replacement = _OwnedFakeBackend("replacement")
        resident = Resident(
            _admitted_body("first", path="/tmp/first"), stale, provider_owned=True)
        self._with_same_body_factory(resident, replacement)

        result = resident.complete({"model": "first", "messages": [{"role": "user", "content": "hi"}]})

        self.assertEqual(result.text, "from replacement")
        self.assertTrue(stale.stopped)
        self.assertTrue(replacement.spawned)
        self.assertEqual(len(replacement.seen), 1)
        self.assertEqual(resident.identity, "first")
        self.assertEqual(resident.state, "ready")

    def test_refused_loopback_request_restarts_owned_child_once(self):
        stale_listener = _OwnedFakeBackend("stale-listener", fail_request=True)
        replacement = _OwnedFakeBackend("replacement")
        resident = Resident(
            _admitted_body("first", path="/tmp/first"), stale_listener,
            provider_owned=True)
        self._with_same_body_factory(resident, replacement)

        result = resident.complete({"model": "first", "messages": [{"role": "user", "content": "hi"}]})

        self.assertEqual(result.text, "from replacement")
        self.assertTrue(stale_listener.stopped)
        self.assertTrue(replacement.spawned)
        self.assertEqual(len(stale_listener.seen), 1)
        self.assertEqual(len(replacement.seen), 1)

    def test_recovery_never_reloads_an_unowned_observed_process(self):
        class _ObservedButUnowned:
            process = _OwnedProcess(exit_code=17)

            def __init__(self):
                self.seen = []
                self.spawned = False
                self.stopped = False

            def spawn(self):
                self.spawned = True

            def ready(self, _timeout):
                return True

            def stop(self):
                self.stopped = True

            def complete(self, payload, timeout=None):
                self.seen.append(payload)
                return type("R", (), {
                    "text": "remote result", "finish_reason": "stop",
                    "prompt_tokens": 1, "completion_tokens": 1,
                    "total_tokens": 2, "degraded": [], "raw": {},
                })()

        backend = _ObservedButUnowned()
        body = _admitted_body("first", path="/tmp/first")
        from hawking.catalog import resolve_action
        resident = Resident(body, backend, resolved_action=resolve_action("first", "serve", [body]))
        import hawking.catalog as catalog_mod
        import hawking.runtime_iface as iface
        original_catalog, original_make = catalog_mod.catalog, iface.make_backend_for_model
        catalog_mod.catalog = lambda extra_roots=None, *, research=False: [resident.body]
        iface.make_backend_for_model = lambda path, **kw: (_ for _ in ()).throw(
            AssertionError("an unowned observed process must not be restarted"))
        try:
            result = resident.complete(
                {"model": "first", "messages": [{"role": "user", "content": "hi"}]}
            )
            same = resident.switch("first")
        finally:
            catalog_mod.catalog, iface.make_backend_for_model = original_catalog, original_make

        self.assertEqual(result.text, "remote result")
        self.assertEqual(len(backend.seen), 1)
        self.assertIs(resident.backend, backend)
        self.assertFalse(backend.spawned)
        self.assertFalse(backend.stopped)
        self.assertFalse(same["switched"])
        self.assertFalse(backend.stopped)

    def test_failed_listener_recovery_is_bounded_to_one_replacement_attempt(self):
        stale_listener = _OwnedFakeBackend("stale-listener", fail_request=True)
        replacement = _OwnedFakeBackend("replacement", fail_request=True)
        resident = Resident(
            _admitted_body("first", path="/tmp/first"), stale_listener,
            provider_owned=True)
        self._with_same_body_factory(resident, replacement)

        with self.assertRaises(urllib.error.URLError):
            resident.complete({"model": "first", "messages": [{"role": "user", "content": "hi"}]})

        self.assertTrue(stale_listener.stopped)
        self.assertTrue(replacement.spawned)
        self.assertEqual(len(stale_listener.seen), 1)
        self.assertEqual(len(replacement.seen), 1)

    def test_dead_child_and_failed_replacement_use_one_forced_reload_budget(self):
        import hawking.catalog as catalog_mod
        import hawking.runtime_iface as iface

        stale = _OwnedFakeBackend("stale", exit_code=17)
        failed_replacement = _OwnedFakeBackend("replacement", fail_request=True)
        must_not_start = _OwnedFakeBackend("second-replacement")
        resident = Resident(
            _admitted_body("first", path="/tmp/first"), stale, provider_owned=True)
        original_catalog = catalog_mod.catalog
        original_make = iface.make_backend_for_model
        catalog_mod.catalog = lambda extra_roots=None, *, research=False: [resident.body]
        made = []

        def _make(path, **_kw):
            made.append(path)
            return [failed_replacement, must_not_start][len(made) - 1]

        iface.make_backend_for_model = _make
        self.addCleanup(setattr, catalog_mod, "catalog", original_catalog)
        self.addCleanup(setattr, iface, "make_backend_for_model", original_make)
        resident.qualify_tools = lambda *args, **kw: resident.tools_verdict

        with self.assertRaises(urllib.error.URLError):
            resident.complete({"model": "first", "messages": [{"role": "user", "content": "hi"}]})

        self.assertEqual(made, ["/tmp/first"])
        self.assertTrue(failed_replacement.spawned)
        self.assertFalse(must_not_start.spawned)
        self.assertEqual(len(failed_replacement.seen), 1)

    def test_constrained_request_never_recovers_a_child_that_dies_after_poll(self):
        import hawking.catalog as catalog_mod
        import hawking.runtime_iface as iface

        class _RacyProcess:
            def __init__(self):
                self.polls = 0

            def poll(self):
                self.polls += 1
                return None if self.polls == 1 else 17

        class _RacyBackend(_OwnedFakeBackend):
            def __init__(self):
                super().__init__("racy")
                self.process = _RacyProcess()

            def complete(self, payload, timeout=None):
                self.seen.append(payload)
                if self.process.poll() is not None:
                    raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
                return super().complete(payload, timeout)

        stale = _RacyBackend()
        resident = Resident(_admitted_body("first", path="/tmp/first"), stale,
                            provider_owned=True)
        original_catalog, original_make = catalog_mod.catalog, iface.make_backend_for_model
        catalog_mod.catalog = lambda extra_roots=None, *, research=False: [resident.body]
        iface.make_backend_for_model = lambda path, **kw: (_ for _ in ()).throw(
            AssertionError("a constrained request must not construct recovery"))
        self.addCleanup(setattr, catalog_mod, "catalog", original_catalog)
        self.addCleanup(setattr, iface, "make_backend_for_model", original_make)

        with self.assertRaises(urllib.error.URLError):
            resident.complete(
                {"model": "first", "messages": [{"role": "user", "content": "hi"}]},
                allow_owned_recovery=False,
            )

        self.assertEqual(stale.process.polls, 2)
        self.assertFalse(stale.spawned)
        self.assertFalse(stale.stopped)

    def test_failed_recovery_restores_the_previous_owned_body(self):
        import hawking.catalog as catalog_mod
        import hawking.runtime_iface as iface

        stale = _OwnedFakeBackend("stale", exit_code=17)
        rejected_replacement = _OwnedFakeBackend("rejected", fail_ready=True)
        restored = _OwnedFakeBackend("restored")
        resident = Resident(
            _admitted_body("first", path="/tmp/first"), stale, provider_owned=True)
        original_catalog = catalog_mod.catalog
        original_make = iface.make_backend_for_model
        catalog_mod.catalog = lambda extra_roots=None, *, research=False: [resident.body]
        candidates = [rejected_replacement, restored]
        iface.make_backend_for_model = lambda path, **kw: candidates.pop(0)
        self.addCleanup(setattr, catalog_mod, "catalog", original_catalog)
        self.addCleanup(setattr, iface, "make_backend_for_model", original_make)
        resident.qualify_tools = lambda *args, **kw: resident.tools_verdict

        with self.assertRaises(RuntimeError):
            resident.complete({"model": "first", "messages": [{"role": "user", "content": "hi"}]})

        self.assertTrue(stale.stopped)
        self.assertTrue(rejected_replacement.spawned)
        self.assertTrue(rejected_replacement.stopped)
        self.assertTrue(restored.spawned)
        self.assertIs(resident.backend, restored)
        self.assertEqual(resident.state, "ready")
        self.assertIn("restored first", resident.error)

    def test_failed_qualification_restores_the_original_body_not_the_target(self):
        import hawking.catalog as catalog_mod
        import hawking.runtime_iface as iface

        old_body = _admitted_body("old", path="/tmp/old")
        target_body = _admitted_body("new", path="/tmp/new")
        old = _OwnedFakeBackend("old")
        candidate = _OwnedFakeBackend("candidate")
        restored = _OwnedFakeBackend("restored")
        resident = Resident(old_body, old, provider_owned=True)
        resident.tools_verdict = {"qualified": True, "reason": "old proof"}
        original_catalog = catalog_mod.catalog
        original_make = iface.make_backend_for_model
        catalog_mod.catalog = lambda extra_roots=None, *, research=False: [target_body]
        made = []

        def _make(path, **_kw):
            made.append(path)
            return [candidate, restored][len(made) - 1]

        iface.make_backend_for_model = _make
        self.addCleanup(setattr, catalog_mod, "catalog", original_catalog)
        self.addCleanup(setattr, iface, "make_backend_for_model", original_make)
        resident.qualify_tools = lambda *args, **kw: (_ for _ in ()).throw(
            RuntimeError("qualification failed"))

        with self.assertRaises(RuntimeError):
            resident.switch("new")

        self.assertEqual(made, ["/tmp/new", "/tmp/old"])
        self.assertTrue(candidate.stopped)
        self.assertIs(resident.backend, restored)
        self.assertIs(resident.body, old_body)
        self.assertEqual(resident.tools_verdict["reason"], "old proof")
        self.assertEqual(resident.state, "ready")


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
