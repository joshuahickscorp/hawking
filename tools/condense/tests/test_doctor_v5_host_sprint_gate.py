#!/usr/bin/env python3.12
"""Cheap adversarial tests for current host/swap/owner evidence gates."""
from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path
import sys
import unittest


CONDENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONDENSE))
import doctor_v5_aggressive_admission_policy as aggressive
import doctor_v5_host_sprint_plan as host


class HostSprintGateTests(unittest.TestCase):
    PROBE_EPOCH = 1_000.0
    SWAP_EPOCH = 1_001.0
    OWNER_EPOCH = 1_002.0
    NOW_EPOCH = 1_003.0

    def setUp(self) -> None:
        self.probe = self._probe(self.PROBE_EPOCH)
        self.plan = host.build_plan(self.probe)
        initial = aggressive.initial_swap_state(
            {"pressure_level": 1, "swap_used_mb": 0.0}, now_epoch=990.0
        )
        self.swap_state, decision = aggressive.advance_swap_state(
            initial, {"pressure_level": 1, "swap_used_mb": 0.0},
            now_epoch=self.SWAP_EPOCH, sealed_baseline_swap_mb=0.0,
        )
        self.swap_binding = host.bind_aggressive_swap_decision(
            self.swap_state, decision
        )
        self.lease = self._lease()
        self.owner_snapshot = host.build_owner_lease_snapshot(
            self.plan, self.probe, [self.lease],
            sampled_at=self._iso(self.OWNER_EPOCH),
        )

    def test_exact_swap_and_expected_owner_lease_pass(self) -> None:
        gate = self._gate()
        self.assertTrue(gate["ok"], gate["blockers"])
        self.assertFalse(gate["production_authorized"])
        self.assertEqual("caller-attested-test-only", gate["evidence_authority"])
        self.assertEqual(self.swap_state["state_sha256"],
                         gate["aggressive_swap_state_sha256"])
        self.assertEqual(self.swap_binding["binding_sha256"],
                         gate["aggressive_swap_binding_sha256"])
        self.assertEqual([self.lease["lease_sha256"]],
                         gate["expected_owner_lease_sha256s"])
        self.assertFalse(gate["automatic_actions_executed"])
        self.assertFalse(gate["fan_control_touched"])
        self.assertFalse(gate["os_services_mutated"])
        self.assertEqual(host._hash_value(host._without(gate, "gate_sha256")),
                         gate["gate_sha256"])

    def test_owner_free_gate_requires_an_exact_empty_snapshot(self) -> None:
        empty = host.build_owner_lease_snapshot(
            self.plan, self.probe, [], sampled_at=self._iso(self.OWNER_EPOCH)
        )
        gate = self._gate(snapshot=empty, expected=())
        self.assertTrue(gate["ok"], gate["blockers"])
        self.assertFalse(gate["production_authorized"])
        unexpected = self._gate(snapshot=self.owner_snapshot, expected=())
        self.assertFalse(unexpected["ok"])
        self.assertIn("owner-free gate observed one or more phase owners",
                      unexpected["blockers"])
        missing = self._gate(snapshot=empty, expected=(self.lease,))
        self.assertFalse(missing["ok"])
        self.assertIn("current owner leases differ from the exact expected owner set",
                      missing["blockers"])

    def test_raw_or_forged_swap_output_cannot_pass(self) -> None:
        raw = {"mode": "green", "allow_launch": True}
        gate = self._gate(binding=raw)
        self.assertFalse(gate["ok"])
        self.assertTrue(any("swap" in row for row in gate["blockers"]))

        forged = copy.deepcopy(self.swap_binding)
        forged["controller_policy_sha256"] = "f" * 64
        forged["binding_sha256"] = host._hash_value(
            host._without(forged, "binding_sha256")
        )
        gate = self._gate(binding=forged)
        self.assertFalse(gate["ok"])
        self.assertIn("aggressive swap controller binding is invalid or stale",
                      gate["blockers"])

        forged = copy.deepcopy(self.swap_binding)
        forged["decision"]["launch_limit"] = 999
        forged["binding_sha256"] = host._hash_value(
            host._without(forged, "binding_sha256")
        )
        gate = self._gate(binding=forged)
        self.assertFalse(gate["ok"])
        self.assertTrue(any("successor state" in row for row in gate["blockers"]))

    def test_swap_state_baseline_and_freshness_are_external_gates(self) -> None:
        wrong_baseline = self._gate(baseline=512.0)
        self.assertFalse(wrong_baseline["ok"])
        self.assertTrue(any("baseline" in row for row in wrong_baseline["blockers"]))

        initial = aggressive.initial_swap_state(
            {"pressure_level": 1, "swap_used_mb": 0.0}, now_epoch=800.0
        )
        stale_state, decision = aggressive.advance_swap_state(
            initial, {"pressure_level": 1, "swap_used_mb": 0.0},
            now_epoch=801.0, sealed_baseline_swap_mb=0.0,
        )
        stale_binding = host.bind_aggressive_swap_decision(stale_state, decision)
        stale = self._gate(state=stale_state, binding=stale_binding)
        self.assertFalse(stale["ok"])
        self.assertIn("aggressive swap controller sample is stale", stale["blockers"])

    def test_stale_future_and_replayed_host_probes_fail(self) -> None:
        stale_probe = self._probe(900.0)
        stale = self._gate(probe=stale_probe)
        self.assertFalse(stale["ok"])
        self.assertIn("current host probe is stale", stale["blockers"])

        future_probe = self._probe(self.NOW_EPOCH + 1.0)
        future = self._gate(probe=future_probe)
        self.assertFalse(future["ok"])
        self.assertIn("current host probe is from the future", future["blockers"])

        next_probe = self._probe(self.PROBE_EPOCH + 1.0)
        replay = self._gate(probe=next_probe)
        self.assertFalse(replay["ok"])
        self.assertIn("owner/lease snapshot is replayed across plan/probe/topology",
                      replay["blockers"])

    def test_forged_or_stale_owner_snapshot_fails_even_when_rehashed(self) -> None:
        forged_lease = copy.deepcopy(self.lease)
        forged_lease["process_identity_sha256"] = "e" * 64
        forged_lease["lease_sha256"] = host._hash_value(
            host._without(forged_lease, "lease_sha256")
        )
        forged_snapshot = host.build_owner_lease_snapshot(
            self.plan, self.probe, [forged_lease],
            sampled_at=self._iso(self.OWNER_EPOCH),
        )
        forged = self._gate(snapshot=forged_snapshot)
        self.assertFalse(forged["ok"])
        self.assertIn("current owner leases differ from the exact expected owner set",
                      forged["blockers"])

        stale_snapshot = copy.deepcopy(self.owner_snapshot)
        stale_snapshot["sampled_at"] = self._iso(900.0)
        stale_snapshot["snapshot_sha256"] = host._hash_value(
            host._without(stale_snapshot, "snapshot_sha256")
        )
        stale = self._gate(snapshot=stale_snapshot)
        self.assertFalse(stale["ok"])
        self.assertIn("owner/lease snapshot is stale", stale["blockers"])

        future_snapshot = copy.deepcopy(self.owner_snapshot)
        future_snapshot["sampled_at"] = self._iso(self.NOW_EPOCH + 1.0)
        future_snapshot["snapshot_sha256"] = host._hash_value(
            host._without(future_snapshot, "snapshot_sha256")
        )
        future = self._gate(snapshot=future_snapshot)
        self.assertFalse(future["ok"])
        self.assertIn("owner/lease snapshot is from the future", future["blockers"])

    def test_rehashed_topology_forgery_is_rejected(self) -> None:
        forged_probe = copy.deepcopy(self.probe)
        forged_probe["topology"]["performance_cores"] = 28
        forged_probe["topology"]["topology_sha256"] = host._hash_value(
            host._without(forged_probe["topology"], "topology_sha256")
        )
        forged_probe["probe_sha256"] = host._hash_value(
            host._without(forged_probe, "probe_sha256")
        )
        gate = self._gate(probe=forged_probe)
        self.assertFalse(gate["ok"])
        self.assertTrue(any("topology" in row for row in gate["blockers"]))

    def test_malformed_evidence_fails_closed_without_an_exception(self) -> None:
        self.assertEqual(["host topology is not an object"],
                         host._topology_errors(None))
        malformed_state = {"mode": "green"}
        gate = self._gate(state=malformed_state)
        self.assertFalse(gate["ok"])
        self.assertTrue(any("swap" in row for row in gate["blockers"]))

        malformed_probe = copy.deepcopy(self.probe)
        malformed_probe["topology"] = []
        malformed_probe["resource_snapshot"] = []
        malformed_probe["probe_sha256"] = host._hash_value(
            host._without(malformed_probe, "probe_sha256")
        )
        gate = self._gate(probe=malformed_probe)
        self.assertFalse(gate["ok"])
        self.assertTrue(any("topology" in row for row in gate["blockers"]))

        malformed_binding = copy.deepcopy(self.swap_binding)
        malformed_binding["decision"]["swap_rate_mb_min"] = float("nan")
        malformed_binding["binding_sha256"] = "f" * 64
        gate = self._gate(binding=malformed_binding)
        self.assertFalse(gate["ok"])
        self.assertTrue(any("swap" in row for row in gate["blockers"]))

    def test_campaign_wide_heavy_patterns_cover_every_release_family(self) -> None:
        commands = (
            "python doctor_v5_ultra_accelerated_queue.py run",
            "python doctor_v5_x_adapter.py run",
            "/tmp/quantize-model-block-parallel --in shard",
            "python appendix_device_runner.py --run",
            "/tmp/hawking-tq-device-probe --artifact x",
            "/tmp/probe-metal-rht --artifact x",
            "python native_probe.py --release",
            "python mop_generation1_campaign.py run",
            "python generation1_cognitive_corpus.py",
        )
        for command in commands:
            self.assertTrue(any(
                pattern.search(command.lower())
                for pattern in host.local_observer.HEAVY_COMMAND_PATTERNS
            ), command)

    def _gate(self, *, probe: dict | None = None, state: dict | None = None,
              binding: dict | None = None, snapshot: dict | None = None,
              expected: tuple[dict, ...] | list[dict] | None = None,
              baseline: float = 0.0) -> dict:
        return host.evaluate_gate(
            self.plan, self.probe if probe is None else probe,
            self.swap_state if state is None else state,
            self.swap_binding if binding is None else binding,
            self.owner_snapshot if snapshot is None else snapshot,
            now_epoch=self.NOW_EPOCH, sealed_baseline_swap_mb=baseline,
            expected_owner_leases=(self.lease,) if expected is None else expected,
        )

    def _lease(self) -> dict:
        lease = {
            "schema": host.OWNER_LEASE_SCHEMA, "version": host.VERSION,
            "contract_sha256": "a" * 64, "role": "encoder",
            "cell_id": "primary", "process_identity_sha256": "b" * 64,
            "lease_generation": 3, "state_generation_at_acquire": 7,
            "acquired_epoch": 999.0,
        }
        lease["lease_sha256"] = host._hash_value(lease)
        return lease

    @staticmethod
    def _iso(epoch: float) -> str:
        return dt.datetime.fromtimestamp(
            epoch, tz=dt.timezone.utc
        ).isoformat(timespec="seconds")

    @classmethod
    def _probe(cls, epoch: float) -> dict:
        receipts = {
            field: {
                "argv": ["/usr/sbin/sysctl", "-n", key], "returncode": 0,
                "output": str(host.REVIEWED_TOPOLOGY[field]), "timed_out": False,
            }
            for field, key in host.TOPOLOGY_SYSCTLS.items()
        }
        topology = {
            **host.REVIEWED_TOPOLOGY,
            "reviewed_expected": host.REVIEWED_TOPOLOGY,
            "verified_for_doctor_v5": True, "source": "read-only-sysctl",
            "sysctl_receipts": receipts,
        }
        topology["topology_sha256"] = host._hash_value(topology)
        probe = {
            "schema": host.PROBE_SCHEMA, "version": host.VERSION,
            "sampled_at": cls._iso(epoch), "topology": topology,
            "resource_snapshot": {
                "power_source": "AC Power", "pressure_level": 1,
                "swap_used_mb": 0.0,
            },
            "thermal_probe": {"returncode": 0}, "thermal_green": True,
            "tools": {}, "spotlight_status_read_only": {"available": False},
            "backup_status_read_only": {"available": False},
            "fan_control_read_or_write_attempted": False,
            "os_service_mutation_attempted": False,
        }
        probe["probe_sha256"] = host._hash_value(probe)
        return probe


if __name__ == "__main__":
    unittest.main()
