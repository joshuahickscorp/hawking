"""ONE Runtime interface: Hawking-native first; legacy runtimes fail closed."""
from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]

from hawking.backends import NoeticNativeBackend
from hawking.genomes import RuntimeGenome
from hawking.machine import MachineGenome
from hawking.models import discover_models, resolve_model
from hawking.runtime import RuntimePool
from hawking.runtime_iface import (
    FOREIGN_AUTHORITIES,
    RuntimeInterface,
    archived_q5k_gguf_path,
    artifact_present,
    classify_backend,
    llamacpp_allowed,
    make_canonical_backend_for_model,
    make_backend_for_model,
    q5k_gguf_required,
    runtime_interface_census,
)
from hawking.session import Session, SessionStore
from hawking.scheduler import Scheduler


def _fake_mlx_dir(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
                "text_config": {"max_position_embeddings": 262144},
            }
        ),
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"x" * 64)
    return str(root)


def _fake_mlx_vlm_dir(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({
            "model_type": "kimi_vl",
            "architectures": ["KimiVLForConditionalGeneration"],
        }),
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"x" * 64)
    return str(root)


class TestClassifyAndFactory(unittest.TestCase):
    def test_default_without_path_is_hawking_native_not_legacy_runtime(self):
        self.assertEqual(classify_backend(None), "noetic_native")
        self.assertFalse(q5k_gguf_required())

    def test_missing_q5k_is_classified_by_suffix_but_not_required(self):
        path = str(archived_q5k_gguf_path())
        self.assertEqual(classify_backend(path), "llamacpp")
        self.assertFalse(llamacpp_allowed())
        self.assertFalse(artifact_present(path) and q5k_gguf_required())
        self.assertFalse(q5k_gguf_required())

    def test_mlx_dir_is_refused_without_spawning(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _fake_mlx_dir(Path(tmp) / "4bit")
            self.assertEqual(classify_backend(model), "mlx")
            with self.assertRaisesRegex(RuntimeError, "runtime kind 'mlx' is retired"):
                make_backend_for_model(model, port=9, n_slots=2)

    def test_kimi_vl_dir_is_refused_without_spawning(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _fake_mlx_vlm_dir(Path(tmp) / "kimi-vl")
            self.assertEqual(classify_backend(model), "mlx")
            with self.assertRaisesRegex(RuntimeError, "runtime kind 'mlx' is retired"):
                make_backend_for_model(model, port=9, n_slots=2)

    def test_gguf_suffix_selects_llama_without_opening_missing_file(self):
        path = "/nonexistent/Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf"
        with self.assertRaisesRegex(RuntimeError, "runtime kind 'llamacpp' is retired"):
            make_backend_for_model(path, port=9, n_slots=1, ctx_size=128)
        self.assertFalse(os.path.isfile(path))

    def test_env_override_noetic(self):
        kind = classify_backend(
            "/any.gguf", env={"HAWKING_RUNTIME_BACKEND": "noetic_native"}
        )
        self.assertEqual(kind, "noetic_native")
        self.assertEqual(classify_backend("/tmp/foo.gravity"), "noetic_native")
        native = make_backend_for_model("/tmp/foo.gravity", port=1)
        self.assertIsInstance(native, NoeticNativeBackend)

    def test_canonical_factory_refuses_mlx_but_accepts_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _fake_mlx_dir(Path(tmp) / "legacy")
            with self.assertRaisesRegex(RuntimeError, "canonical Hawking execution"):
                make_canonical_backend_for_model(model, port=9)
        native = make_canonical_backend_for_model("/tmp/pulsar.gravity", port=9)
        self.assertIsInstance(native, NoeticNativeBackend)

    def test_noetic_complete_refuses_to_invent_tokens(self):
        backend = NoeticNativeBackend(model_path="reserved")
        backend.spawn()
        self.assertTrue(backend.ready(0.01))
        with self.assertRaises(RuntimeError) as ctx:
            backend.complete({"prompt": "hi"})
        self.assertIn("interface reservation", str(ctx.exception))
        self.assertFalse(backend.supports("response_format"))


class TestDoesNotDuplicateSchedulerOrSession(unittest.TestCase):
    def test_interface_source_defines_none_of_the_foreign_classes(self):
        src = inspect.getsource(RuntimeInterface)
        for name in FOREIGN_AUTHORITIES:
            self.assertNotIn(f"class {name}", src)
        self.assertIs(Scheduler, Scheduler)
        self.assertIs(SessionStore, SessionStore)

    def test_bind_session_does_not_write_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(tmp)
            session = Session(goal="x")
            iface = RuntimeInterface.from_artifact(None)
            iface.bind_session(session.id)
            self.assertEqual(iface.session_id, session.id)
            self.assertIsNone(store.load(session.id))


class TestPoolPicksMlxForMlxDir(unittest.TestCase):
    # These pins used to leak: set inline with no teardown, a 64 GiB swap
    # ceiling escaped into the rest of the process and every later test that
    # relied on the 2 GiB default silently measured the wrong gate. Save and
    # restore the same way RuntimePoolTestCase does.
    _ENV = (
        "HAWKING_DISABLE_SIGNAL_HOOKS",
        "HAWKING_SWAP_CEILING_GIB",
        "HAWKING_RESIDENT_RUNTIME_LIMIT",
    )

    def setUp(self):
        saved = {k: os.environ.get(k) for k in self._ENV}

        def restore():
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        self.addCleanup(restore)

    def test_make_backend_via_pool_refuses_retired_mlx(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _fake_mlx_dir(Path(tmp) / "4bit")
            os.environ["HAWKING_DISABLE_SIGNAL_HOOKS"] = "1"
            os.environ["HAWKING_SWAP_CEILING_GIB"] = "64"
            os.environ["HAWKING_RESIDENT_RUNTIME_LIMIT"] = "1"
            pool = RuntimePool(
                model,
                requested_n=1,
                workspace=tmp,
                topology="slot",
                repo_root=tmp,
            )
            with self.assertRaisesRegex(RuntimeError, "canonical Hawking execution"):
                pool._make_backend(0, 1, 9999)
            pool.stop()

    def test_start_missing_gguf_raises_without_requiring_q5k_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["HAWKING_DISABLE_SIGNAL_HOOKS"] = "1"
            missing = str(Path(tmp) / "no-such.gguf")
            pool = RuntimePool(
                missing,
                requested_n=1,
                workspace=tmp,
                topology="slot",
                repo_root=tmp,
            )
            with self.assertRaisesRegex(RuntimeError, "accepts only an admitted Hawking-native"):
                pool.start()


class TestMlxDiscovery(unittest.TestCase):
    def test_discover_and_resolve_explicit_mlx_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _fake_mlx_dir(Path(tmp) / "4bit")
            found = discover_models([tmp])
            self.assertEqual(found, [])
            chosen = resolve_model(explicit=model)
            self.assertIsNone(chosen)


class TestRuntimeGenomeRecordsControlSet(unittest.TestCase):
    def test_mlx_headline_from_receipt_not_remeasured(self):
        control = json.loads(
            (REPO / "receipts" / "headless" / "CONVENTIONAL_CONTROL_SET.json").read_text(
                encoding="utf-8"
            )
        )
        genome = RuntimeGenome.from_control_set(REPO, control=control)
        headline = genome.mlx_headline()
        self.assertEqual(headline.get("startup_s"), 1.329)
        self.assertEqual(headline.get("prefill_tps"), 309.94)
        self.assertEqual(headline.get("decode_tps"), 38.06)
        self.assertEqual(headline.get("context_tokens"), 262144)
        self.assertEqual(headline.get("peak_memory_gb"), 18.21)
        self.assertFalse(genome.data.get("remeasured"))
        self.assertFalse(genome.data.get("q5k_gguf_required"))
        iface = RuntimeInterface.from_control_set(control)
        self.assertEqual(iface.backend_kind, "retired")
        self.assertFalse((iface.profile or {}).get("remeasured"))
        with tempfile.TemporaryDirectory() as tmp:
            bag = Path(tmp) / "machine-genome.json"
            mg = MachineGenome(bag)
            genome.record_into_machine_genome(mg)
            profile = mg.get_profile("runtime_genome")
            self.assertIsNotNone(profile)
            self.assertFalse(profile.get("remeasured"))
            canonical = Path.home() / ".config" / "hawking" / "machine_genome.json"
            self.assertNotEqual(bag, canonical)

    def test_census_q5k_not_required(self):
        census = runtime_interface_census()
        self.assertFalse(census["q5k_gguf_required"])
        self.assertEqual(census["default_kind_without_path"], "noetic_native")
        self.assertFalse(census["scheduler_duplicated"])
        self.assertEqual(census["foreign_classes_defined_here"], [])


if __name__ == "__main__":
    unittest.main()
