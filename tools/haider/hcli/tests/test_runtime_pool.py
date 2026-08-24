from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[4]

from hcli.backends import CompletionResult, LlamaServerBackend
from hcli.machine import (
    GIB,
    MemGate,
    live_machine_identity,
    resolve_decode_topology,
    resolve_runtime_limits,
)
from hcli.resources import pid_is_alive, process_start_token
from hcli.runtime import RuntimePool, allocate_port


def _alive(pid: int) -> bool:
    return pid_is_alive(pid)


class FakeBackend:
    def __init__(self, model_path, port, n_slots=1, index=0, **_kwargs):
        self.model_path = model_path
        self.port = port
        self.n_slots = n_slots
        self.index = index
        self.process = None
        self.pid = None
        self.start_time = None
        self._stopped = False

    def identity(self):
        return {
            "backend": "fake",
            "binary": "sleep",
            "version": "0",
            "runtime_build": "sleep 0",
            "model_path": self.model_path,
            "model_bytes": 1,
            "model_identity": f"{self.model_path}:1:none",
            "context": 128,
            "quantisation": "none",
        }

    def spawn(self, **kwargs):
        if kwargs.get("port") is not None:
            self.port = int(kwargs["port"])
        if kwargs.get("n_slots") is not None:
            self.n_slots = int(kwargs["n_slots"])
        if self.process is not None and self.process.poll() is None:
            return
        self.process = subprocess.Popen(
            ["sleep", "120"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.pid = self.process.pid
        self.start_time = process_start_token(self.pid)
        self._stopped = False

    def ready(self, timeout):
        return self.process is not None and self.process.poll() is None

    def endpoint(self):
        return f"http://127.0.0.1:{self.port}"

    def supports(self, feature):
        return feature in {"prefix_cache", "slots", "chat_template_kwargs"}

    def complete(self, payload, timeout=None):
        time.sleep(0.12)
        return CompletionResult(
            raw={"ok": True, "payload": payload},
            finish_reason="stop",
            text="ok",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )

    def stop(self):
        from hcli.backends import terminate_pid

        report = {"pid": self.pid, "gone": True, "unreaped": []}
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                try:
                    self.process.wait(timeout=3)
                except Exception:
                    pass
        if self.pid and _alive(self.pid):
            killed = terminate_pid(int(self.pid), term_timeout=2, kill_timeout=2)
            report.update(killed)
            if not killed.get("gone"):
                report["unreaped"] = [self.pid]
                report["gone"] = False
        self.process = None
        if report.get("gone"):
            self.pid = None
        self._stopped = True
        return report


def _gate(**kwargs):
    defaults = dict(
        reserve_bytes=1,
        model_bytes=100,
        per_runtime_overhead_bytes=100,
        headroom_frac=0.1,
        metal_info={
            "recommendedMaxWorkingSetSize": 80 * GIB,
            "currentAllocatedSize": 0,
            "source": "test-inject",
        },
        topology="slot",
    )
    defaults.update(kwargs)
    return MemGate(**defaults)


def _dummy_model(root: Path) -> str:
    path = root / "dummy.gguf"
    path.write_bytes(b"x" * 64)
    return str(path)


LIMIT_ENV = (
    "HCLI_RESIDENT_RUNTIME_LIMIT",
    "HCLI_ACTIVE_DECODE_LIMIT",
    "RESIDENT_RUNTIME_LIMIT",
    "ACTIVE_DECODE_LIMIT",
    "HCLI_MAX_RUNTIMES",
    "HCLI_OBSERVED_MODEL_OVERLAP",
    "HCLI_DECODE_TOPOLOGY",
    "HCLI_DISABLE_SIGNAL_HOOKS",
    "HCLI_SWAP_CEILING_GIB",
)


class RuntimePoolTestCase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in LIMIT_ENV}
        os.environ["HCLI_DISABLE_SIGNAL_HOOKS"] = "1"
        os.environ.pop("HCLI_MAX_RUNTIMES", None)
        os.environ.pop("HCLI_OBSERVED_MODEL_OVERLAP", None)
        # Pin the swap ceiling so admission is decided by the logic under test
        # and not by whatever else this machine happens to be running. Left
        # unset, MemGate reads the host's live swap: seven of these tests
        # admitted ZERO runtimes and failed, and the assertions then compared
        # 0 against 0 while claiming to measure admission width. A test whose
        # outcome depends on ambient memory pressure passes on a quiet box and
        # fails on a busy one, which is the same as not testing.
        # This does not weaken refusal coverage: the refusal path is exercised
        # by a deliberately constructed gate in test_memgate_refuses_and_admits,
        # not by ambient pressure.
        os.environ["HCLI_SWAP_CEILING_GIB"] = "64"

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestLimitResolution(RuntimePoolTestCase):
    def test_env_wins_independently(self):
        os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "3"
        os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            limits = resolve_runtime_limits(repo_root=tmp, start_dir=tmp)
        self.assertEqual(limits.resident_limit, 3)
        self.assertEqual(limits.active_decode_limit, 1)
        self.assertTrue(limits.resident_source.startswith("env:"))
        self.assertTrue(limits.active_source.startswith("env:"))
        self.assertNotEqual(limits.resident_limit, limits.active_decode_limit)

    def test_genome_then_receipt_then_equilibrium_then_fallback(self):
        os.environ.pop("HCLI_RESIDENT_RUNTIME_LIMIT", None)
        os.environ.pop("HCLI_ACTIVE_DECODE_LIMIT", None)
        os.environ.pop("RESIDENT_RUNTIME_LIMIT", None)
        os.environ.pop("ACTIVE_DECODE_LIMIT", None)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            genome_dir = home / ".config" / "hcli"
            genome_dir.mkdir(parents=True)
            (genome_dir / "machine_genome.json").write_text(
                json.dumps(
                    {
                        "schema": "hcli.machine_genome.v1",
                        "generated_at": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                        "measured_by": "tools/headless/machine_probe.py",
                        "resident_runtime_limit": 4,
                        "active_decode_limit": 2,
                        "machine": live_machine_identity(),
                    }
                )
            )
            receipt_dir = root / "receipts" / "headless"
            receipt_dir.mkdir(parents=True)
            (receipt_dir / "MACHINE_GENOME.json").write_text(
                json.dumps(
                    {"RESIDENT_RUNTIME_LIMIT": 9, "ACTIVE_DECODE_LIMIT": 9}
                )
            )
            eq_dir = root / ".haider" / "bootstrap-director-v6"
            eq_dir.mkdir(parents=True)
            (eq_dir / "worker-equilibrium.json").write_text(
                json.dumps({"bootstrap_workers": 8, "active_decode_limit": 8})
            )
            with patch.object(
                Path, "home", return_value=home
            ):
                limits = resolve_runtime_limits(repo_root=root, start_dir=root)
            self.assertEqual(limits.resident_limit, 4)
            self.assertEqual(limits.active_decode_limit, 2)

            (genome_dir / "machine_genome.json").unlink()
            with patch.object(Path, "home", return_value=home):
                limits = resolve_runtime_limits(repo_root=root, start_dir=root)
            self.assertEqual(limits.resident_limit, 9)
            self.assertEqual(limits.active_decode_limit, 9)

            (receipt_dir / "MACHINE_GENOME.json").unlink()
            with patch.object(Path, "home", return_value=home):
                limits = resolve_runtime_limits(repo_root=root, start_dir=root)
            self.assertEqual(limits.resident_limit, 8)
            self.assertEqual(limits.active_decode_limit, 8)
            self.assertEqual(limits.resident_source, "worker-equilibrium.json")

            (eq_dir / "worker-equilibrium.json").unlink()
            empty = root / "empty"
            empty.mkdir()
            with patch.object(Path, "home", return_value=home):
                limits = resolve_runtime_limits(
                    repo_root=empty, start_dir=empty
                )
            self.assertEqual(limits.resident_limit, 1)
            self.assertEqual(limits.active_decode_limit, 1)
            self.assertEqual(limits.resident_source, "fallback")
            self.assertEqual(limits.active_source, "fallback")

    def test_topology_receipt_prefers_slot(self):
        os.environ.pop("HCLI_DECODE_TOPOLOGY", None)
        topo, source = resolve_decode_topology(repo_root=REPO)
        receipt = REPO / "receipts" / "headless" / "DECODE_TOPOLOGY.json"
        if receipt.is_file():
            self.assertEqual(topo, "slot")
            self.assertIn("DECODE_TOPOLOGY.json", source)
        else:
            self.assertEqual(topo, "process")
            self.assertEqual(source, "fallback")


class TestMemGate(RuntimePoolTestCase):
    def test_gpu_gate_before_host(self):
        # Usable = 0.9 * 20 GiB = 18 GiB. One process at 19 GiB must refuse
        # even though host free RAM is huge.
        gate = _gate(
            model_bytes=18 * GIB,
            per_runtime_overhead_bytes=int(1.6 * GIB),
            metal_info={
                "recommendedMaxWorkingSetSize": 20 * GIB,
                "currentAllocatedSize": 0,
                "source": "test-inject",
            },
            topology="process",
            reserve_bytes=1,
        )
        snap = {
            "total_bytes": 96 * GIB,
            "free_bytes": 80 * GIB,
            "swap_used_bytes": 0,
            "pressure": "normal",
        }
        decision = gate.consider(0, extra=1, snapshot=snap, refresh_metal=False)
        self.assertFalse(decision.allow)
        self.assertEqual(decision.gate, "gpu")

    def test_absurd_reserve_refuses_on_host(self):
        gate = _gate(reserve_bytes=10**18, topology="slot")
        snap = {
            "total_bytes": 96 * GIB,
            "free_bytes": 40 * GIB,
            "swap_used_bytes": 0,
            "pressure": "normal",
        }
        decision = gate.consider(0, extra=1, snapshot=snap, refresh_metal=False)
        self.assertFalse(decision.allow)
        self.assertEqual(decision.gate, "host")
        self.assertIn("reserve", decision.reason)

    def test_slot_second_runtime_is_not_a_full_model_copy(self):
        gate = _gate(
            model_bytes=18 * GIB,
            per_runtime_overhead_bytes=int(1.6 * GIB),
            topology="slot",
        )
        first = gate.gpu_cost_bytes(0, extra=1)
        second = gate.gpu_cost_bytes(1, extra=1)
        increment = second - first
        self.assertLess(increment, 18 * GIB * 0.5)
        self.assertEqual(increment, int(1.6 * GIB))


class TestPoolLifecycle(RuntimePoolTestCase):
    def test_two_limits_and_in_flight_cap(self):
        os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "3"
        os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = RuntimePool(
                _dummy_model(root),
                requested_n=8,
                workspace=root,
                backend_factory=FakeBackend,
                mem_gate=_gate(topology="slot"),
                topology="slot",
                repo_root=root,
                observed_overlap=3,
            )
            self.assertEqual(pool.resident_limit, 3)
            self.assertEqual(pool.active_decode_limit, 1)
            try:
                pool.start()
                self.assertEqual(pool.admitted_n, 3)
                self.assertEqual(len(pool.runtimes), 3)
                errors = []

                def worker():
                    try:
                        pool.complete({"prompt": "x"}, prefix_key="k")
                    except Exception as exc:  # noqa: BLE001
                        errors.append(exc)

                threads = [threading.Thread(target=worker) for _ in range(3)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=10)
                self.assertFalse(errors)
                self.assertEqual(pool.max_in_flight_observed, 1)
            finally:
                pool.stop()

    def test_memgate_refuses_and_admits(self):
        os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "3"
        os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = _dummy_model(root)
            refused = RuntimePool(
                model,
                requested_n=3,
                workspace=root,
                backend_factory=FakeBackend,
                mem_gate=_gate(reserve_bytes=10**18, topology="slot"),
                topology="slot",
                repo_root=root,
            )
            try:
                refused.start()
                self.assertIn(refused.admitted_n, (0, 1))
                self.assertTrue(refused.refusal_reason)
            finally:
                refused.stop()
            sane = RuntimePool(
                model,
                requested_n=3,
                workspace=root / "sane",
                backend_factory=FakeBackend,
                mem_gate=_gate(reserve_bytes=1, topology="slot"),
                topology="slot",
                repo_root=root,
                observed_overlap=3,
            )
            try:
                sane.start()
                self.assertGreater(sane.admitted_n, refused.admitted_n)
            finally:
                sane.stop()

    def test_second_runtime_marginal_not_full_model(self):
        os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "2"
        os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = _dummy_model(root)
            pool = RuntimePool(
                model,
                requested_n=2,
                workspace=root,
                backend_factory=FakeBackend,
                mem_gate=_gate(topology="slot"),
                topology="slot",
                repo_root=root,
                observed_overlap=2,
            )
            try:
                pool.start()
                self.assertGreaterEqual(len(pool.admission_records), 2)
                cost = pool.admission_records[1]["marginal_free_ram_cost_bytes"]
                self.assertLess(int(cost), 64 * 0.5 + 1)
            finally:
                pool.stop()

    def test_clean_stop_reaps_children(self):
        os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "1"
        os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = RuntimePool(
                _dummy_model(root),
                requested_n=1,
                workspace=root,
                backend_factory=FakeBackend,
                mem_gate=_gate(),
                topology="slot",
                repo_root=root,
            )
            try:
                pool.start()
                pids = [r.pid for r in pool.runtimes if r.pid]
                self.assertTrue(pids)
                pool.stop()
                pool.stop()
                for pid in pids:
                    self.assertFalse(_alive(pid), pid)
            finally:
                pool.stop()

    def test_reaper_kills_recorded_orphan_only(self):
        os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "1"
        os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orphan = subprocess.Popen(
                ["sleep", "120"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            foreign = subprocess.Popen(
                ["bash", "-lc", "exec -a llama-server sleep 120"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            dead_owner = subprocess.Popen(["true"])
            dead_owner.wait()
            try:
                time.sleep(0.15)
                hcli = root / ".hcli"
                hcli.mkdir()
                (hcli / "runtime_pool.json").write_text(
                    json.dumps(
                        {
                            "schema": "hcli.runtime_pool.v1",
                            "pool_pid": dead_owner.pid,
                            "pool_start_time": "dead-owner",
                            "children": [
                                {
                                    "pid": orphan.pid,
                                    "start_time": process_start_token(orphan.pid),
                                    "port": 1,
                                    "model": "/x.gguf",
                                }
                            ],
                        }
                    )
                )
                pool = RuntimePool(
                    _dummy_model(root),
                    requested_n=1,
                    workspace=root,
                    backend_factory=FakeBackend,
                    mem_gate=_gate(),
                    topology="slot",
                    repo_root=root,
                )
                try:
                    reports = pool.reap_orphans()
                    self.assertFalse(_alive(orphan.pid), reports)
                    self.assertTrue(_alive(foreign.pid))
                finally:
                    pool.stop()
            finally:
                for proc in (orphan, foreign):
                    if proc.poll() is None:
                        try:
                            os.kill(proc.pid, signal.SIGKILL)
                        except OSError:
                            pass
                        try:
                            proc.wait(timeout=3)
                        except Exception:
                            pass

    def test_prefix_affinity_hits_same_runtime(self):
        os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "2"
        os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = RuntimePool(
                _dummy_model(root),
                requested_n=2,
                workspace=root,
                backend_factory=FakeBackend,
                mem_gate=_gate(topology="slot"),
                topology="slot",
                repo_root=root,
                observed_overlap=2,
            )
            try:
                pool.start()
                a = pool.complete({"prompt": "a"}, prefix_key="stable")
                b = pool.complete({"prompt": "b"}, prefix_key="stable")
                self.assertEqual(a.runtime_index, b.runtime_index)
                self.assertGreaterEqual(pool.prefix_hits, 1)
            finally:
                pool.stop()

    def test_unmeasured_overlap_admits_one_runtime(self):
        os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "4"
        os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "2"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = RuntimePool(
                _dummy_model(root),
                requested_n=4,
                workspace=root,
                backend_factory=FakeBackend,
                mem_gate=_gate(topology="process"),
                topology="process",
                repo_root=root,
            )
            try:
                pool.start()
                self.assertEqual(pool.admitted_n, 1)
                self.assertEqual(pool.overlap_admit_cap, 1)
            finally:
                pool.stop()

    def test_stored_overlap_raises_admission_width(self):
        os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "4"
        os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "2"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from hcli.runtime import store_observed_overlap

            store_observed_overlap(root, 2)
            pool = RuntimePool(
                _dummy_model(root),
                requested_n=4,
                workspace=root,
                backend_factory=FakeBackend,
                mem_gate=_gate(topology="slot"),
                topology="slot",
                repo_root=root,
            )
            try:
                pool.start()
                self.assertEqual(pool.admitted_n, 2)
                self.assertEqual(pool.overlap_admit_cap, 2)
            finally:
                pool.stop()

    def test_payload_degrade_strips_response_format(self):
        backend = LlamaServerBackend(model_path="/no.gguf", port=allocate_port())

        def no_schema(feature):
            if feature == "response_format":
                return False
            return True

        backend.supports = no_schema  # type: ignore[method-assign]
        prepared, degraded = backend._prepare_payload(
            {
                "messages": [{"role": "user", "content": "give data"}],
                "response_format": {"type": "json_object"},
            }
        )
        self.assertIn("response_format", degraded)
        self.assertNotIn("response_format", prepared)
        self.assertIn("JSON object", prepared["messages"][-1]["content"])


class TestLlamaIdentity(unittest.TestCase):
    def test_identity_fields_and_supports_against_binary(self):
        import shutil

        if not shutil.which("llama-server") and not os.environ.get(
            "HCLI_LLAMA_SERVER"
        ):
            self.skipTest("llama-server not on PATH")
        backend = LlamaServerBackend(
            model_path="/nonexistent/Huihui-Q5_K.gguf", port=allocate_port()
        )
        ident = backend.identity()
        for key in (
            "backend",
            "binary",
            "version",
            "model_path",
            "model_bytes",
            "context",
            "quantisation",
        ):
            self.assertIn(key, ident)
        self.assertEqual(ident["quantisation"], "Q5_K")
        self.assertTrue(backend.supports("response_format"))
        self.assertTrue(backend.supports("chat_template_kwargs"))


if __name__ == "__main__":
    unittest.main()
