from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import doctor_v5_controlled_swap_activation as activation
import doctor_v5_controlled_swap_autoresume as autoresume


class FakeProbe:
    def __init__(self, swap: float = 6.44): self.swap = swap
    def sample(self):
        return {"pressure_level": 1, "pressure_name": "normal",
                "swap_used_mb": self.swap, "power_source": "AC",
                "thermal_green": True}


class FakeService:
    def __init__(self): self.loaded = True; self.events: list[str] = []
    def is_loaded(self, _label): return self.loaded
    def bootout(self, _label): self.loaded = False; self.events.append("bootout")
    def bootstrap(self, _plist): self.loaded = True; self.events.append("bootstrap")
    def kickstart(self, _label): self.events.append("kickstart")


class HardCut(BaseException):
    pass


def seal(value: dict, field: str) -> dict:
    value[field] = activation._hash_value(value)
    return value


class Fixture:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.paths = activation.production_paths(
            self.root, launch_agent=self.root / "Library/LaunchAgents/service.plist")
        self.service = FakeService(); self.probe = FakeProbe()
        self._build()

    def close(self): self.temporary.cleanup()

    def _write(self, path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes): path.write_bytes(value)
        else: path.write_text(json.dumps(value, sort_keys=True) + "\n")

    def _build(self):
        p = self.paths
        for path, raw in (
            (p.accelerated_queue, b"# accelerated\n"),
            (p.stacked_admission, b"# stacked\n"),
            (p.successor_autoresume, b"# autoresume\n"),
            (p.activation_source, Path(activation.__file__).read_bytes()),
        ):
            if path != Path(activation.__file__).resolve(): self._write(path, raw)
        phase = self.root / "tools/condense/doctor_v5_phase_aware_disk_gate.py"
        ledger = self.root / "tools/condense/doctor_v5_remaining_scratch_ledger.py"
        self._write(phase, b"def install_phase_gate(base_module, predecessor_gate, successor_policy, policy_root):\n    return predecessor_gate\n")
        self._write(ledger, b"# ledger\n")
        queue_source = f'''from pathlib import Path
import hashlib
ROOT = Path(__file__).resolve().parents[2]
PHASE_GATE_PATH = Path({str(phase)!r})
def phase_gate_declaration(module_path):
    raw = module_path.read_bytes()
    return {{"api_schema": {activation.PHASE_API_SCHEMA!r}, "module": {{"path": str(module_path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}}, "install_callable": "install_phase_gate"}}
def validate_policy(policy, policy_path=None, deep_predecessor=False):
    return []
'''
        self._write(p.successor_queue, queue_source.encode())

        cell_id = "model__4bpw__codec-control"
        runtime = p.ultra / f"runtime_specs/{cell_id}.json"
        self._write(runtime, {
            "schema": "hawking.doctor_v5_strand_ladder_spec.v1",
            "adapter_id": "doctor-v5-strand-ladder-qwen25-dense",
            "operation": "condense_control", "program_spec_sha256": "9" * 64})
        cell = {
            "cell_id": cell_id, "cell_identity_sha256": "8" * 64,
            "runtime_spec_schema": "hawking.doctor_v5_strand_ladder_spec.v1",
            "adapter_id": "doctor-v5-strand-ladder-qwen25-dense",
            "command": "condense_control", "branch": "codec_control",
            "runtime_spec_path": str(runtime.relative_to(self.root)),
            "projected_output_bytes": 9_000_000_000,
            "admission": {"disk_reserve_bytes": 150_000_000_000,
                          "recommended_scratch_bytes": 48_000_000_000},
        }
        plan = seal({"schema": "hawking.doctor_v5_ultra_campaign_plan.v1",
                     "cells": [cell]}, "plan_sha256")
        self._write(p.plan, plan)
        state = seal({"schema": "hawking.doctor_v5_ultra_queue_state.v1",
                      "plan_sha256": plan["plan_sha256"], "status": "drained",
                      "active_cells": [], "active_children": {},
                      "cells": {cell_id: {"status": "pending"}}}, "state_sha256")
        campaign = seal({"schema": "hawking.doctor_v5_ultra_campaign.v1",
                         "plan_sha256": plan["plan_sha256"]}, "campaign_sha256")
        control = seal({"schema": "hawking.doctor_v5_ultra_control.v1",
                        "plan_sha256": plan["plan_sha256"], "mode": "drain"},
                       "control_sha256")
        pid = seal({"schema": "hawking.doctor_v5_ultra_queue_pid.v1",
                    "plan_sha256": plan["plan_sha256"], "pid": 4242},
                   "pid_record_sha256")
        for path, value in ((p.state, state), (p.campaign, campaign),
                            (p.control, control), (p.pid_record, pid)):
            self._write(path, value)
        result = p.results / cell_id
        self._write(result / "checkpoint.json", {"unit": 7, "sha256": "7" * 64})
        self._write(result / "reconstruction/model.safetensors", b"payload" * 100)

        overlay = seal({"schema": "hawking.doctor_v5_stacked_admission_overlay.v1"},
                       "overlay_sha256")
        self._write(p.overlay, overlay)
        pending = seal({"schema": "hawking.doctor_v5_acceleration_pending_runtime.v1",
                        "plan_sha256": plan["plan_sha256"]}, "packet_sha256")
        self._write(p.pending_runtime_packet, pending)
        old_marker = seal({
            "schema": "hawking.doctor_v5_acceleration_active_marker.v1",
            "overlay_path": str(p.overlay.resolve()),
            "overlay_sha256": overlay["overlay_sha256"],
            "pending_runtime_generation_sha256": pending["packet_sha256"],
            "accelerated_queue": activation._artifact(p.accelerated_queue),
        }, "marker_sha256")
        self._write(p.old_marker, old_marker)
        backup = p.ultra / "staged_acceleration/forward_recovery_v1/backup/test"
        head = seal({
            "schema": "hawking.doctor_v5_forward_recovery_wal_entry.v1",
            "index": 1, "forward_packet_sha256": "6" * 64,
            "phase": "active", "operation": "supervisor-active",
            "previous_entry_sha256": None, "details": {},
        }, "entry_sha256")
        self._write(backup / "wal/00000001.json", head)
        forward = seal({
            "schema": "hawking.doctor_v5_forward_recovery_journal.v2",
            "status": "active", "phase": "active",
            "plan_sha256": plan["plan_sha256"], "backup_root": str(backup),
            "wal_index": 1, "wal_entry_sha256": head["entry_sha256"],
            "forward_packet_sha256": "6" * 64,
        }, "journal_sha256")
        self._write(p.forward_journal, forward)
        self._write(p.launch_agent, b"old-service\n")

    def prepare(self):
        activation.issue_phase_gate(paths=self.paths, probe=self.probe)
        return activation.stage(paths=self.paths, service=self.service, probe=self.probe)

    def protected_bytes(self):
        paths = [self.paths.plan, self.paths.state, self.paths.campaign,
                 *sorted(path for path in self.paths.results.rglob("*") if path.is_file())]
        return {str(path): path.read_bytes() for path in paths}


class ControlledSwapActivationTests(unittest.TestCase):
    def test_activate_preserves_plan_state_campaign_and_results(self):
        fx = Fixture(); self.addCleanup(fx.close)
        packet = fx.prepare(); before = fx.protected_bytes()
        journal = activation.activate(
            packet_sha256=packet["packet_sha256"],
            generation_id=packet["generation_id"], paths=fx.paths,
            service=fx.service, probe=fx.probe)
        self.assertEqual("active", journal["status"])
        self.assertEqual(before, fx.protected_bytes())
        activation.validate_active_marker(paths=fx.paths)

    def test_all_hard_cuts_resolve_on_the_correct_side_of_marker(self):
        for label in activation.HARD_CUT_LABELS:
            with self.subTest(label=label):
                fx = Fixture()
                try:
                    packet = fx.prepare(); before = fx.protected_bytes()
                    def fault(observed):
                        if observed == label: raise HardCut(label)
                    with mock.patch.object(activation, "_fault", side_effect=fault):
                        with self.assertRaises(HardCut):
                            activation.activate(
                                packet_sha256=packet["packet_sha256"],
                                generation_id=packet["generation_id"], paths=fx.paths,
                                service=fx.service, probe=fx.probe)
                    recovered = activation.recover(paths=fx.paths, service=fx.service)
                    if label in activation.HARD_CUT_LABELS[:3]:
                        self.assertEqual("rolled-back", recovered["status"])
                        self.assertEqual(b"old-service\n", fx.paths.launch_agent.read_bytes())
                    else:
                        self.assertEqual("active", recovered["status"])
                        activation.validate_active_marker(paths=fx.paths)
                    self.assertEqual(before, fx.protected_bytes())
                finally:
                    fx.close()

    def test_old_new_marker_and_service_mixtures_fail_closed(self):
        fx = Fixture(); self.addCleanup(fx.close)
        packet = fx.prepare()
        activation.activate(packet_sha256=packet["packet_sha256"],
                            generation_id=packet["generation_id"], paths=fx.paths,
                            service=fx.service, probe=fx.probe)
        fx.paths.launch_agent.write_bytes(b"old-service\n")
        with self.assertRaises(activation.ActivationError):
            activation.validate_active_marker(paths=fx.paths)
        fx.paths.launch_agent.write_bytes(fx.paths.staged_service.read_bytes())
        marker = json.loads(fx.paths.active_marker.read_text())
        marker["packet"]["sha256"] = "0" * 64
        marker["marker_sha256"] = activation._hash_value(
            activation._without(marker, "marker_sha256"))
        fx.paths.active_marker.write_text(json.dumps(marker))
        with self.assertRaises(activation.ActivationError):
            activation.validate_active_marker(paths=fx.paths)

    def test_autoresume_accepts_exact_clean_drained_handoff(self):
        fx = Fixture(); self.addCleanup(fx.close)
        packet = fx.prepare()
        activation.activate(packet_sha256=packet["packet_sha256"],
                            generation_id=packet["generation_id"], paths=fx.paths,
                            service=fx.service, probe=fx.probe)
        calls = []
        class Result: returncode = 0
        def runner(*args, **kwargs): calls.append((args, kwargs)); return Result()
        self.assertEqual(0, autoresume.run_once(paths=fx.paths, runner=runner))
        self.assertEqual(1, len(calls))
        env = calls[0][1]["env"]
        self.assertEqual(packet["policy_sha256"], env[autoresume.ENV_POLICY_SHA256])

    def test_absolute_swap_cap_and_receipt_hash_are_fail_closed(self):
        fx = Fixture(); self.addCleanup(fx.close)
        with self.assertRaises(activation.ActivationError):
            activation.issue_phase_gate(paths=fx.paths, probe=FakeProbe(512.001))
        before = activation.result_tree_identity(fx.paths.results)
        checkpoint = next(fx.paths.results.rglob("checkpoint.json"))
        checkpoint.write_text('{"tampered":true}\n')
        self.assertNotEqual(before, activation.result_tree_identity(fx.paths.results))


if __name__ == "__main__":
    unittest.main()
