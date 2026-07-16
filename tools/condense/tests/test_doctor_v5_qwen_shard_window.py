from __future__ import annotations

import hashlib
import json
import copy
import datetime as dt
from pathlib import Path
import sys
import tempfile
import textwrap
import time
import unittest


HERE = Path(__file__).resolve().parent
CONDENSE = HERE.parent
sys.path.insert(0, str(CONDENSE))

import doctor_v5_qwen_shard_window as window


STUB = r'''
import argparse, hashlib, json, os, pathlib, sys, time

def canon(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def artifact(path):
    p = pathlib.Path(path).resolve()
    data = p.read_bytes()
    return {"path": str(p), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

p = argparse.ArgumentParser()
p.add_argument("--phase", required=True)
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--receipt", required=True)
p.add_argument("--events", required=True)
p.add_argument("--fail-finalize-ordinal", type=int, default=-1)
p.add_argument("--corrupt-receipt-ordinal", type=int, default=-1)
a = p.parse_args()
ordinal = int(os.environ["HAWKING_WINDOW_ORDINAL"])
with open(a.events, "a") as h:
    h.write(f"{time.monotonic()} {a.phase} start {ordinal}\n")
if a.phase == "finalize":
    time.sleep(0.25)
else:
    time.sleep(0.04)
if a.phase == "finalize" and ordinal == a.fail_finalize_ordinal:
    sys.exit(17)
raw = pathlib.Path(a.input).read_bytes()
pathlib.Path(a.output).write_bytes(a.phase.encode() + b":" + raw)
out = artifact(a.output)
receipt = {
    "schema": "hawking.doctor_v5_qwen_shard_window_child_receipt.v1",
    "status": "complete", "stage": a.phase, "ordinal": ordinal,
    "attempt": int(os.environ["HAWKING_WINDOW_ATTEMPT"]),
    "token": os.environ["HAWKING_WINDOW_TOKEN"],
    "request_sha256": os.environ["HAWKING_WINDOW_REQUEST_SHA256"],
    "request_file_sha256": os.environ["HAWKING_WINDOW_REQUEST_FILE_SHA256"],
    "source_sha256": os.environ["HAWKING_WINDOW_SOURCE_SHA256"],
    "shard_request_sha256": os.environ["HAWKING_WINDOW_SHARD_REQUEST_SHA256"],
    "program_sha256": os.environ["HAWKING_WINDOW_PROGRAM_SHA256"],
    "input_sha256": os.environ["HAWKING_WINDOW_INPUT_SHA256"],
    "output": out,
}
if ordinal == a.corrupt_receipt_ordinal:
    receipt["output"]["sha256"] = "0" * 64
receipt["receipt_sha256"] = hashlib.sha256(canon(receipt)).hexdigest()
pathlib.Path(a.receipt).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
with open(a.events, "a") as h:
    h.write(f"{time.monotonic()} {a.phase} end {ordinal}\n")
'''


def probe(_pids: set[int]) -> dict[str, object]:
    return {"status": "sampled", "aggregate_process_tree_rss_bytes": 0,
            "swap_used_bytes": 0, "memory_pressure": "normal",
            "thermal_state": "nominal"}


class QwenShardWindowTests(unittest.TestCase):
    def _qualified_overlay(self, root: Path) -> Path:
        policy = window.admission_policy
        contract = policy._load_thread_contract()
        binary = root / "quantizer"
        binary.write_bytes(b"exact-qwen-window-test-binary")
        binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
        receipts = []
        for threads, wall in {8: 8.0, 12: 6.0, 16: 4.0, 20: 5.0}.items():
            receipt = {
                "schema": contract.RECEIPT_SCHEMA, "status": "pass",
                "scope": "production", "synthetic": False,
                "tier": "14B", "rate": "q3", "threads": threads,
                "binary_sha256": binary_sha, "source_sha256": "b" * 64,
                "canonical_output_sha256": "c" * 64,
                "output_sha256": "c" * 64, "exact_output": True,
                "wall_seconds": wall, "peak_rss_bytes": 10_000 + threads,
                "scratch_budget_bytes": 268_435_456, "mode": "block_parallel",
            }
            path = root / f"thread-{threads}.json"
            path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
            receipts.append(path)
        profile = contract.build_profile(
            receipts, expected_binary_sha256=binary_sha, rss_limit_bytes=1_000_000
        )
        profile_path = root / "thread-profile.json"
        profile_path.write_text(json.dumps(profile, sort_keys=True) + "\n")
        cell = {
            "cell_id": "qwen-14b-q3", "model_family": "qwen2.5-dense",
            "model_label": "14B", "rate_id": "q3",
            "branch": "codec_control", "priority": 0,
            "admission": {"whole_parent_residency_assumed": False},
        }
        plan = {
            "schema": "hawking.doctor_v5_ultra_campaign_plan.v1", "cells": [cell]
        }
        plan["plan_sha256"] = policy._hash_value(plan)
        state = {
            "schema": "hawking.doctor_v5_ultra_queue_state.v1",
            "plan_sha256": plan["plan_sha256"], "status": "running",
            "active_children": {},
            "cells": {cell["cell_id"]: {
                "status": "pending", "request_sha256": "d" * 64,
            }},
        }
        state["state_sha256"] = policy._hash_value(state)
        start = dt.datetime(2026, 7, 14, tzinfo=dt.timezone.utc)
        samples = []
        for index in range(policy.MIN_AUTHENTICATED_SAMPLES):
            pgid = 20_000
            samples.append({
                "cell_id": cell["cell_id"], "plan_sha256": plan["plan_sha256"],
                "request_sha256": "d" * 64,
                "process_budget_bytes": policy.PROCESS_BUDGET_BYTES,
                "sampled_at": (start + dt.timedelta(seconds=8 * index)).isoformat(),
                "root_pid": pgid, "pgid": pgid, "process_count": 2,
                "processes": [
                    {"pid": pgid, "ppid": 1, "pgid": pgid,
                     "rss_bytes": 1_000_000_000, "state": "S"},
                    {"pid": pgid + 1, "ppid": pgid, "pgid": pgid,
                     "rss_bytes": 9_000_000_000, "state": "R"},
                ],
                "tree_rss_bytes": 10_000_000_000,
                "max_tree_rss_bytes": 10_000_000_000,
                "at_or_over_budget": False,
            })
        overlay = policy.build_overlay(
            plan, state, samples,
            baseline_snapshot={"pressure_level": 1, "swap_used_mb": 0.0},
            thread_profile_path=profile_path, thread_binary_path=binary,
        )
        self.assertEqual([], policy.validate_overlay(overlay))
        overlay_path = root / "aggressive-overlay.json"
        window._atomic_json(overlay_path, overlay)
        return overlay_path

    def _setup(
        self, directory: str, *, aggressive: bool = True,
        fail_finalize_ordinal: int = -1, corrupt_receipt_ordinal: int = -1,
        execution_mode: str = "stub-test-only",
    ) -> tuple[Path, Path]:
        root = Path(directory)
        stub = root / "stub.py"
        stub.write_text(textwrap.dedent(STUB), encoding="utf-8")
        events = root / "events.log"
        shards = []
        for ordinal in range(3):
            source = root / f"source-{ordinal}.bin"
            source.write_bytes((f"source-{ordinal}-" * 100).encode())
            shards.append({
                "ordinal": ordinal, "source": window._artifact(source),
                "shard_request_sha256": hashlib.sha256(
                    f"request-{ordinal}".encode()
                ).hexdigest(),
            })
        program = {
            "program": window._artifact(stub),
            "launcher_argv": [
                sys.executable, str(stub), "--phase", "{phase}",
                "--input", "{input}", "--output", "{output}",
                "--receipt", "{receipt}", "--events", str(events),
                "--fail-finalize-ordinal", str(fail_finalize_ordinal),
                "--corrupt-receipt-ordinal", str(corrupt_receipt_ordinal),
            ],
        }
        if execution_mode == "production-parity-gated":
            overlay_path = self._qualified_overlay(root)
            thread_profile = window.load_qualified_thread_profile(
                overlay_path, tier="14B", rate="q3"
            )
            admission_binding = window.load_production_admission_binding(overlay_path)
        else:
            thread_profile = (window.stub_thread_profile()
                              if aggressive else window.serial_thread_profile())
            admission_binding = window.stub_admission_binding()
        request = window.build_request(
            shards=shards, output_root=root / "output",
            prepare_program=program, finalize_program=program,
            thread_profile=thread_profile, admission_binding=admission_binding,
            process_budget_bytes=78_000_000_000,
            prepare_reservation_bytes=32_000_000_000,
            finalize_reservation_bytes=30_000_000_000,
            execution_mode=execution_mode,
        )
        request_path = root / "request.json"
        window.write_request(request_path, request)
        return request_path, events

    def test_two_shard_window_overlaps_finalize_n_with_prepare_n_plus_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request, events = self._setup(directory)
            state = window.WindowCoordinator(
                request, resource_probe=probe
            ).run(timeout_seconds=10)
            self.assertEqual("complete", state["status"])
            self.assertEqual([0, 1, 2], [row["ordinal"] for row in state["shards"]
                                         if row["commit"]["status"] == "complete"])
            parsed = [line.split() for line in events.read_text().splitlines()]
            times = {(row[1], row[2], int(row[3])): float(row[0]) for row in parsed}
            self.assertLess(times[("prepare", "start", 1)],
                            times[("finalize", "end", 0)])
            for ordinal in range(1, 3):
                self.assertGreaterEqual(
                    times[("prepare", "start", ordinal)],
                    times[("prepare", "end", ordinal - 1)],
                )
            active = peak = 0
            for row in sorted(parsed, key=lambda value: float(value[0])):
                active += 1 if row[2] == "start" else -1
                peak = max(peak, active)
            self.assertLessEqual(peak, 2)
            receipts = sorted((Path(directory) / "output/committed-shards").glob(
                "*.receipt.json"
            ))
            self.assertEqual(3, len(receipts))
            for ordinal, path in enumerate(receipts):
                receipt = json.loads(path.read_text())
                self.assertEqual(ordinal, receipt["ordinal"])
                self.assertTrue(receipt["validators"]["prepare_child"])
                self.assertTrue(receipt["validators"]["finalize_child"])
                self.assertFalse(receipt["source_files_deleted"])

    def test_serial_fallback_never_has_two_shards_in_flight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request, events = self._setup(directory, aggressive=False)
            state = window.WindowCoordinator(
                request, resource_probe=probe
            ).run(timeout_seconds=10)
            self.assertEqual("complete", state["status"])
            parsed = [line.split() for line in events.read_text().splitlines()]
            times = {(row[1], row[2], int(row[3])): float(row[0]) for row in parsed}
            self.assertGreaterEqual(times[("prepare", "start", 1)],
                                    times[("finalize", "end", 0)])

    def test_crash_resume_adopts_valid_completed_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request, _events = self._setup(directory)
            first = window.WindowCoordinator(request, resource_probe=probe)
            first._launch(0, "prepare")
            first.processes[(0, "prepare")].wait(timeout=5)
            resumed = window.WindowCoordinator(request, resource_probe=probe)
            self.assertEqual("complete", resumed.state["shards"][0]["prepare"]["status"])
            self.assertTrue(resumed.state["shards"][0]["prepare"][
                "adopted_after_coordinator_restart"
            ])
            state = resumed.run(timeout_seconds=10)
            self.assertEqual("complete", state["status"])

    def test_crash_resume_adopts_existing_canonical_commits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request, _events = self._setup(directory)
            first = window.WindowCoordinator(request, resource_probe=probe)
            first.run(timeout_seconds=10)
            state = window._read_json(first.state_path)
            state["status"] = "running"
            state["next_commit_ordinal"] = 0
            for shard in state["shards"]:
                shard["commit"] = {"status": "pending"}
            state["state_sha256"] = window._hash_value(
                window._without(state, "state_sha256")
            )
            window._atomic_json(first.state_path, state)
            resumed = window.WindowCoordinator(request, resource_probe=probe)
            final = resumed.run(timeout_seconds=5)
            self.assertEqual("complete", final["status"])
            self.assertTrue(all(row["commit"]["status"] == "complete"
                                for row in final["shards"]))

    def test_failure_cancels_later_uncommitted_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request, _events = self._setup(directory, fail_finalize_ordinal=0)
            coordinator = window.WindowCoordinator(request, resource_probe=probe)
            with self.assertRaises(window.WindowError):
                coordinator.run(timeout_seconds=10)
            self.assertEqual("failed", coordinator.state["status"])
            for shard in coordinator.state["shards"][1:]:
                self.assertEqual("cancelled", shard["commit"]["status"])
            self.assertTrue(all(Path(row["source"]["path"]).exists()
                                for row in coordinator.request["shards"]))

    def test_adversarial_child_receipt_prevents_commit_and_cancels_later(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request, _events = self._setup(directory, corrupt_receipt_ordinal=0)
            coordinator = window.WindowCoordinator(request, resource_probe=probe)
            with self.assertRaises(window.WindowError):
                coordinator.run(timeout_seconds=10)
            self.assertFalse(any((Path(directory) / "output/committed-shards").glob(
                "*.strand"
            )))
            self.assertEqual("failed", coordinator.state["status"])

    def test_aggregate_resource_gate_fails_closed(self) -> None:
        def denied(_pids: set[int]) -> dict[str, object]:
            return {"status": "sampled",
                    "aggregate_process_tree_rss_bytes": 77_000_000_000,
                    "swap_used_bytes": 0, "memory_pressure": "normal",
                    "thermal_state": "nominal"}
        with tempfile.TemporaryDirectory() as directory:
            request, _events = self._setup(directory)
            with self.assertRaisesRegex(window.WindowError, "resource gate"):
                window.WindowCoordinator(request, resource_probe=denied).run(
                    timeout_seconds=2
                )

    def test_production_mode_remains_blocked_on_owner_free_real_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request, _events = self._setup(
                directory, execution_mode="production-parity-gated"
            )
            with self.assertRaisesRegex(window.WindowError, "owner-free parity"):
                window.WindowCoordinator(request, resource_probe=probe).run()
            caps = window.capabilities()
            self.assertFalse(caps["reviewed_for_live_execution"])
            self.assertFalse(caps["source_deletion_permitted"])

    def test_production_request_requires_qualified_profile_and_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request, _events = self._setup(directory)
            doc = window._read_json(request)
            doc["execution_mode"] = "production-parity-gated"
            doc["request_sha256"] = window._hash_value(
                window._without(doc, "request_sha256")
            )
            errors = window.validate_request(doc, verify_files=False)
            self.assertTrue(any("production thread profile" in row for row in errors))
            self.assertTrue(any("production admission binding" in row for row in errors))

    def test_qualified_profile_is_exact_and_rejects_unmeasured_thread_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request, _events = self._setup(
                directory, execution_mode="production-parity-gated"
            )
            doc = window._read_json(request)
            profile = doc["thread_profile"]
            self.assertEqual([8, 12, 16, 20], profile["required_threads"])
            self.assertEqual(16, profile["prepare_threads"])
            self.assertTrue(profile["exact_parity_approved"])
            profile["prepare_threads"] = 24
            profile["profile_sha256"] = window._hash_value(
                window._without(profile, "profile_sha256")
            )
            doc["request_sha256"] = window._hash_value(
                window._without(doc, "request_sha256")
            )
            errors = window.validate_request(doc, verify_files=False)
            self.assertTrue(any("exact qualified tier/rate" in row for row in errors))

    def test_swap_baseline_cannot_ratchet_and_controller_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request, _events = self._setup(
                directory, execution_mode="production-parity-gated"
            )
            doc = window._read_json(request)
            binding = doc["admission_binding"]
            self.assertFalse(binding["baseline_can_ratchet"])
            self.assertEqual(window.admission_policy.swap_policy(),
                             binding["swap_policy"])
            ratcheted = copy.deepcopy(doc)
            ratcheted_binding = ratcheted["admission_binding"]
            ratcheted_binding["sealed_baseline_swap_mb"] = 900.0
            ratcheted_binding["baseline_seal_sha256"] = window._baseline_seal(900.0)
            ratcheted_binding["binding_sha256"] = window._hash_value(
                window._without(ratcheted_binding, "binding_sha256")
            )
            ratcheted["request_sha256"] = window._hash_value(
                window._without(ratcheted, "request_sha256")
            )
            errors = window.validate_request(ratcheted, verify_files=False)
            self.assertTrue(any("re-baselined" in row for row in errors))

            drifted = copy.deepcopy(doc)
            drifted_binding = drifted["admission_binding"]
            drifted_binding["controller"] = drifted["programs"]["prepare"]["program"]
            drifted_binding["binding_sha256"] = window._hash_value(
                window._without(drifted_binding, "binding_sha256")
            )
            drifted["request_sha256"] = window._hash_value(
                window._without(drifted, "request_sha256")
            )
            window._atomic_json(request, drifted)
            with self.assertRaisesRegex(window.WindowError, "controller"):
                window.WindowCoordinator(request, resource_probe=probe)

    def test_overlay_artifact_drift_is_rejected_before_production_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request, _events = self._setup(
                directory, execution_mode="production-parity-gated"
            )
            doc = window._read_json(request)
            overlay = Path(doc["admission_binding"]["overlay"]["path"])
            overlay.write_text(overlay.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(window.WindowError, "overlay"):
                window.WindowCoordinator(request, resource_probe=probe)

    def test_relative_swap_growth_uses_soft_and_hard_controller_modes(self) -> None:
        def soft(_pids: set[int]) -> dict[str, object]:
            return {"status": "sampled", "aggregate_process_tree_rss_bytes": 0,
                    "swap_used_bytes": 600 * 1024 * 1024,
                    "memory_pressure": "normal", "thermal_state": "nominal"}

        def hard(_pids: set[int]) -> dict[str, object]:
            return {"status": "sampled", "aggregate_process_tree_rss_bytes": 0,
                    "swap_used_bytes": 1600 * 1024 * 1024,
                    "memory_pressure": "normal", "thermal_state": "nominal"}

        with tempfile.TemporaryDirectory() as directory:
            request, _events = self._setup(directory)
            state = window.WindowCoordinator(request, resource_probe=soft).run(
                timeout_seconds=10
            )
            decisions = [
                shard[phase].get("last_admission_sample", {}).get(
                    "aggressive_admission_decision", {}
                ).get("mode")
                for shard in state["shards"] for phase in ("prepare", "finalize")
            ]
            self.assertIn("soft_throttle", decisions)
        with tempfile.TemporaryDirectory() as directory:
            request, _events = self._setup(directory)
            coordinator = window.WindowCoordinator(request, resource_probe=hard)
            with self.assertRaisesRegex(window.WindowError, "resource gate"):
                coordinator.run(timeout_seconds=2)
            persisted = window._read_json(coordinator.state_path)
            self.assertFalse(persisted["last_admission_decision"]["admitted"])
            self.assertEqual(
                "hard_stop",
                persisted["last_admission_decision"]["sample"][
                    "aggressive_admission_decision"
                ]["mode"],
            )
            self.assertEqual(
                persisted["state_sha256"],
                window._hash_value(window._without(persisted, "state_sha256")),
            )

    def test_source_and_program_drift_are_rejected_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request, _events = self._setup(directory)
            doc = window._read_json(request)
            Path(doc["shards"][0]["source"]["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(window.WindowError, "source file differs"):
                window.WindowCoordinator(request, resource_probe=probe)
        with tempfile.TemporaryDirectory() as directory:
            request, _events = self._setup(directory)
            doc = window._read_json(request)
            program = Path(doc["programs"]["prepare"]["program"]["path"])
            program.write_text(program.read_text() + "\n# drift\n")
            with self.assertRaisesRegex(window.WindowError, "program file differs"):
                window.WindowCoordinator(request, resource_probe=probe)


if __name__ == "__main__":
    unittest.main()
