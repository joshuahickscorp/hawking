"""Stale-genome detection, honest runtime topology receipts, oversized slots.

A genome is a prior, not present truth. These tests require the pool to
refuse stale numbers, to record only observed topology, and to surface the
single-stream cost of allocating more slots than a deep decode can use.
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]

from hcli.backends import CompletionResult
from hcli.machine import (
    GIB,
    MemGate,
    resolve_runtime_limits,
)
from hcli import machine as machine_mod
from hcli.resources import pid_is_alive, process_start_token
from hcli.runtime import RuntimePool


TOPOLOGY_KEYS = (
    "model_path",
    "artifact_identity",
    "pid",
    "port",
    "ctx_size",
    "parallel",
    "per_slot_context",
    "active_sequences",
    "kv_configuration",
)

LIMIT_ENV = (
    "HCLI_RESIDENT_RUNTIME_LIMIT",
    "HCLI_ACTIVE_DECODE_LIMIT",
    "RESIDENT_RUNTIME_LIMIT",
    "ACTIVE_DECODE_LIMIT",
    "HCLI_MAX_RUNTIMES",
    "HCLI_DECODE_TOPOLOGY",
    "HCLI_DISABLE_SIGNAL_HOOKS",
    "HCLI_CTX_SIZE",
    "HCLI_GENOME_STALENESS_HORIZON_S",
    "HCLI_SWAP_CEILING_GIB",
)

LIVE_MACHINE = {
    "hw_model": "Mac15,14",
    "cpu": "Apple M3 Ultra",
    "ncpu": 28,
    "mem_bytes": 103079215104,
}

MODEL_PATH = "/models/Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf"
MODEL_BYTES = 19535701280


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _genome(**overrides):
    data = {
        "schema": "hcli.machine_genome.v1",
        "generated_at": _now_iso(),
        "measured_by": "tools/headless/machine_probe.py (metal-gated)",
        "resident_runtime_limit": 4,
        "active_decode_limit": 2,
        "machine": dict(LIVE_MACHINE),
        "runtime_identity": {
            "model_path": MODEL_PATH,
            "model_size_bytes": MODEL_BYTES,
        },
    }
    data.update(overrides)
    return data


class FakeBackend:
    def __init__(self, model_path, port, n_slots=1, index=0, **_kwargs):
        self.model_path = model_path
        self.port = port
        self.n_slots = n_slots
        self.index = index
        self.ctx_size = int(os.environ.get("HCLI_CTX_SIZE", "8192"))
        self.process = None
        self.pid = None
        self.start_time = None
        self._stopped = False

    def identity(self):
        try:
            model_bytes = os.path.getsize(self.model_path) if os.path.isfile(self.model_path) else 1
        except OSError:
            model_bytes = 1
        return {
            "backend": "fake",
            "binary": "sleep",
            "version": "0",
            "runtime_build": "sleep 0",
            "model_path": self.model_path,
            "model_bytes": model_bytes,
            "model_identity": f"{self.model_path}:{model_bytes}:none",
            "context": self.ctx_size,
            "quantisation": "none",
            "n_slots": self.n_slots,
        }

    def command(self, port=None, n_slots=None):
        use_port = int(port if port is not None else (self.port or 0))
        slots = max(1, int(n_slots if n_slots is not None else self.n_slots))
        return [
            "sleep",
            "120",
            "--ctx-size",
            str(self.ctx_size),
            "--parallel",
            str(slots),
            "--port",
            str(use_port),
        ]

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

    def complete(self, payload, timeout=None):
        return CompletionResult(
            raw={"ok": True},
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
        if self.pid and pid_is_alive(self.pid):
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


def _field(envelope):
    if not isinstance(envelope, dict):
        return envelope, None
    return envelope.get("value"), envelope.get("reason")


def _status(report):
    if report is None:
        return None
    if isinstance(report, dict):
        return report.get("status"), list(report.get("reasons") or [])
    return getattr(report, "status", None), list(getattr(report, "reasons", None) or [])


class AuthorityTestCase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in LIMIT_ENV}
        os.environ["HCLI_DISABLE_SIGNAL_HOOKS"] = "1"
        os.environ.pop("HCLI_MAX_RUNTIMES", None)
        os.environ.pop("HCLI_RESIDENT_RUNTIME_LIMIT", None)
        os.environ.pop("HCLI_ACTIVE_DECODE_LIMIT", None)
        os.environ.pop("RESIDENT_RUNTIME_LIMIT", None)
        os.environ.pop("ACTIVE_DECODE_LIMIT", None)
        os.environ.pop("HCLI_GENOME_STALENESS_HORIZON_S", None)
        os.environ["HCLI_CTX_SIZE"] = "8192"
        # Pin the swap ceiling: unset, MemGate consults the host's live swap and
        # admits zero runtimes, so assertions like admitted_n == 5 fail for a
        # reason that has nothing to do with the authority under test. Ambient
        # memory pressure must not decide the outcome of an admission test.
        os.environ["HCLI_SWAP_CEILING_GIB"] = "64"

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestGenomeFreshness(AuthorityTestCase):
    def _resolve(self, home: Path, root: Path, genome: dict, receipt=None, **kwargs):
        genome_dir = home / ".config" / "hcli"
        genome_dir.mkdir(parents=True)
        (genome_dir / "machine_genome.json").write_text(json.dumps(genome))
        if receipt is not None:
            receipt_dir = root / "receipts" / "headless"
            receipt_dir.mkdir(parents=True)
            (receipt_dir / "MACHINE_GENOME.json").write_text(json.dumps(receipt))
        sig = inspect.signature(resolve_runtime_limits)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        with patch.object(Path, "home", return_value=home):
            return resolve_runtime_limits(
                repo_root=root,
                start_dir=root,
                **accepted,
            )

    def test_stale_genome_older_than_horizon_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            genome = _genome(generated_at="2020-01-01T00:00:00Z")
            genome["resident_runtime_limit"] = 9
            genome["active_decode_limit"] = 9
            limits = self._resolve(
                home,
                root,
                genome,
                receipt={"RESIDENT_RUNTIME_LIMIT": 3, "ACTIVE_DECODE_LIMIT": 1},
                live_machine=LIVE_MACHINE,
                model_path=MODEL_PATH,
                model_bytes=MODEL_BYTES,
                now="2026-08-23T00:00:00Z",
                horizon_s=7 * 24 * 3600,
            )
        self.assertEqual(limits.resident_limit, 3)
        self.assertEqual(limits.active_decode_limit, 1)
        self.assertIn("MACHINE_GENOME.json", limits.resident_source)
        reports = getattr(limits, "genome_reports", None)
        self.assertTrue(reports, "stale genome must be reported, not silently skipped")
        status, reasons = _status(reports[0])
        self.assertEqual(status, "STALE")
        blob = " ".join(reasons).lower()
        self.assertTrue(
            "horizon" in blob or "older" in blob or "age" in blob,
            reasons,
        )

    def test_stale_genome_wrong_machine_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            genome = _genome(machine={"hw_model": "Mac99,99", "mem_bytes": 1})
            genome["resident_runtime_limit"] = 9
            genome["active_decode_limit"] = 9
            limits = self._resolve(
                home,
                root,
                genome,
                receipt={"RESIDENT_RUNTIME_LIMIT": 3, "ACTIVE_DECODE_LIMIT": 1},
                live_machine=LIVE_MACHINE,
                model_path=MODEL_PATH,
                model_bytes=MODEL_BYTES,
            )
        self.assertEqual(limits.resident_limit, 3)
        reports = getattr(limits, "genome_reports", None)
        self.assertTrue(reports)
        status, reasons = _status(reports[0])
        self.assertEqual(status, "STALE")
        blob = " ".join(reasons).lower()
        self.assertIn("machine", blob)

    def test_stale_genome_wrong_model_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            genome = _genome()
            genome["resident_runtime_limit"] = 9
            genome["active_decode_limit"] = 9
            limits = self._resolve(
                home,
                root,
                genome,
                receipt={"RESIDENT_RUNTIME_LIMIT": 3, "ACTIVE_DECODE_LIMIT": 1},
                live_machine=LIVE_MACHINE,
                model_path="/models/other.gguf",
                model_bytes=123,
            )
        self.assertEqual(limits.resident_limit, 3)
        reports = getattr(limits, "genome_reports", None)
        self.assertTrue(reports)
        status, reasons = _status(reports[0])
        self.assertEqual(status, "STALE")
        blob = " ".join(reasons).lower()
        self.assertIn("model", blob)

    def test_fresh_genome_is_trusted_and_not_reported_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            genome = _genome()
            limits = self._resolve(
                home,
                root,
                genome,
                receipt={"RESIDENT_RUNTIME_LIMIT": 9, "ACTIVE_DECODE_LIMIT": 9},
                live_machine=LIVE_MACHINE,
                model_path=MODEL_PATH,
                model_bytes=MODEL_BYTES,
                now=_now_iso(),
                horizon_s=7 * 24 * 3600,
            )
        self.assertEqual(limits.resident_limit, 4)
        self.assertEqual(limits.active_decode_limit, 2)
        reports = getattr(limits, "genome_reports", None)
        self.assertTrue(reports)
        status, reasons = _status(reports[0])
        self.assertEqual(status, "FRESH")
        self.assertEqual(list(reasons), [])

    def test_assess_genome_freshness_names_the_reason(self):
        assess = getattr(machine_mod, "assess_genome_freshness", None)
        self.assertTrue(callable(assess), "assess_genome_freshness is not defined")
        old = assess(
            _genome(generated_at="2020-01-01T00:00:00Z"),
            live_machine=LIVE_MACHINE,
            model_path=MODEL_PATH,
            model_bytes=MODEL_BYTES,
            now="2026-08-23T00:00:00Z",
            horizon_s=3600,
        )
        self.assertEqual(old.status, "STALE")
        self.assertTrue(old.reasons)
        fresh = assess(
            _genome(),
            live_machine=LIVE_MACHINE,
            model_path=MODEL_PATH,
            model_bytes=MODEL_BYTES,
            now=_now_iso(),
            horizon_s=7 * 24 * 3600,
        )
        self.assertEqual(fresh.status, "FRESH")
        self.assertEqual(list(fresh.reasons), [])


class TestRuntimeTopologyReceipt(AuthorityTestCase):
    def test_spawned_runtime_receipt_has_observable_fields_and_honest_nulls(self):
        os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "2"
        os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = _dummy_model(root)
                # Admission width now follows MEASURED model-call overlap,
                # not the caller's constant: a mission whose model calls never
                # overlap does not get extra 19.79 GiB processes idling. These
                # tests are about per-runtime topology and oversized SLOT
                # allocation, not about admission width, so they declare the
                # overlap they need rather than relying on a default that has
                # since been measured down to 1.
            pool = RuntimePool(
                model,
                requested_n=2,
                observed_overlap=2,
                workspace=root,
                backend_factory=FakeBackend,
                mem_gate=_gate(topology="slot"),
                topology="slot",
                repo_root=root,
            )
            try:
                pool.start()
                self.assertGreaterEqual(len(pool.runtimes), 1)
                runtime = pool.runtimes[0]
                receipt_fn = getattr(runtime, "topology_receipt", None)
                if callable(receipt_fn):
                    topo = receipt_fn()
                else:
                    recs = [r for r in pool.admission_records if r.get("admitted")]
                    self.assertTrue(recs)
                    topo = recs[0].get("topology")
                self.assertTrue(isinstance(topo, dict), topo)
                for key in TOPOLOGY_KEYS:
                    self.assertIn(key, topo, key)
                    field = topo[key]
                    self.assertIsInstance(field, dict, key)
                    self.assertIn("value", field, key)
                    self.assertIn("reason", field, key)
                    if field["value"] is None:
                        self.assertTrue(
                            field["reason"],
                            f"{key} is null without a reason",
                        )
                    else:
                        self.assertIsNone(
                            field["reason"],
                            f"{key} is observed but still has a reason: {field['reason']}",
                        )
                model_path, _ = _field(topo["model_path"])
                self.assertEqual(os.path.realpath(model_path), os.path.realpath(model))
                artifact, _ = _field(topo["artifact_identity"])
                self.assertTrue(artifact)
                pid, _ = _field(topo["pid"])
                self.assertTrue(pid)
                port, _ = _field(topo["port"])
                self.assertTrue(port)
                ctx, _ = _field(topo["ctx_size"])
                self.assertEqual(int(ctx), 8192)
                parallel, _ = _field(topo["parallel"])
                self.assertEqual(int(parallel), 2)
                per_slot, _ = _field(topo["per_slot_context"])
                self.assertEqual(int(per_slot), 8192 // 2)
                active, _ = _field(topo["active_sequences"])
                self.assertEqual(int(active), 0)
                kv, kv_reason = _field(topo["kv_configuration"])
                self.assertIsNone(kv)
                self.assertTrue(kv_reason)
            finally:
                pool.stop()


class TestOversizedSlotAllocation(AuthorityTestCase):
    def test_five_slots_for_one_decode_surfaces_single_stream_cost(self):
        os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "5"
        os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
                # Admission width now follows MEASURED model-call overlap,
                # not the caller's constant: a mission whose model calls never
                # overlap does not get extra 19.79 GiB processes idling. These
                # tests are about per-runtime topology and oversized SLOT
                # allocation, not about admission width, so they declare the
                # overlap they need rather than relying on a default that has
                # since been measured down to 1.
            pool = RuntimePool(
                _dummy_model(root),
                requested_n=5,
                observed_overlap=5,
                workspace=root,
                backend_factory=FakeBackend,
                mem_gate=_gate(topology="slot"),
                topology="slot",
                repo_root=root,
            )
            try:
                pool.start()
                self.assertEqual(pool.admitted_n, 5)
                decision = getattr(pool, "allocation_decision", None)
                self.assertIsNotNone(
                    decision,
                    "oversized allocation must be recorded at the decision, not silent",
                )
                self.assertTrue(
                    decision.get("oversized_vs_single_stream")
                    or decision.get("oversized"),
                    decision,
                )
                self.assertEqual(int(decision.get("planned_slots")), 5)
                cost = decision.get("single_stream_cost")
                self.assertIsInstance(cost, dict, cost)
                loss = cost.get("relative_loss")
                if loss is None and isinstance(cost.get("value"), dict):
                    loss = cost["value"].get("relative_loss")
                self.assertIsNotNone(loss, cost)
                self.assertGreater(float(loss), 0.04)
                self.assertLess(float(loss), 0.08)
                guidance = str(decision.get("caller_should") or "")
                self.assertTrue(guidance)
                self.assertIn("requested_n", guidance)
            finally:
                pool.stop()

    def test_single_slot_is_not_oversized(self):
        os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "1"
        os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = RuntimePool(
                _dummy_model(root),
                requested_n=1,
                workspace=root,
                backend_factory=FakeBackend,
                mem_gate=_gate(topology="slot"),
                topology="slot",
                repo_root=root,
            )
            try:
                pool.start()
                decision = getattr(pool, "allocation_decision", None)
                self.assertIsNotNone(decision)
                self.assertFalse(
                    decision.get("oversized_vs_single_stream")
                    or decision.get("oversized"),
                    decision,
                )
            finally:
                pool.stop()


if __name__ == "__main__":
    unittest.main()
