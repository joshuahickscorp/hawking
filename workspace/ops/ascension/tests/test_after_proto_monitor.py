"""Unit tests for the after-proto monitor façade and hardening gate."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_OPS = Path(__file__).resolve().parents[1]
_ROOT = _OPS.parents[2]
for p in (str(_OPS.parent), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ascension.after_proto_monitor import (  # noqa: E402
    MONITOR_SCHEMA,
    PROTO_OFFLOAD_ENDPOINT,
    monitor_after_proto,
    validate_proto_offload_receipt,
)
from ascension.garbage_ecosystem import classify_object
from ascension.notifications import NotificationKind
from ascension.signals import HostSignals


def _sig(
    *,
    free_disk_gib: float = 100.0,
    available_ram_gib: float = 24.0,
    total_ram_gib: float = 32.0,
    swap_used_gib: float = 0.0,
    swap_total_gib: float = 0.0,
    gpu_util_pct: int | None = 10,
    thermal_throttling: bool | None = False,
    foreground_user_active: bool | None = False,
) -> HostSignals:
    g = 1024**3
    return HostSignals(
        free_disk_bytes=int(free_disk_gib * g),
        total_disk_bytes=int(500 * g),
        available_ram_bytes=int(available_ram_gib * g),
        total_ram_bytes=int(total_ram_gib * g),
        swap_used_bytes=int(swap_used_gib * g),
        swap_total_bytes=int(swap_total_gib * g),
        gpu_util_pct=gpu_util_pct,
        thermal_throttling=thermal_throttling,
        foreground_user_active=foreground_user_active,
        source="test",
    )


class TestProtoReceiptGate(unittest.TestCase):
    def test_validate_proto_receipt_passes(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td, "PROTO_FRANKENSTEIN_RUN_RECEIPT.json")
            path.write_text(
                json.dumps(
                    {
                        "schema": "hawking.frankenstein.proto_frankenstein_run.v1",
                        "endpoint": "PROTO_FRANKENSTEIN_V0_FULL_LATENT_SEALED",
                        "recorded_at": "2026-08-05T00:00:00Z",
                        "dry_run": False,
                        "runtime_storage": {
                            "storage": {"donor_weights_retained": False}
                        },
                    }
                )
            )
            result = validate_proto_offload_receipt(path)
            self.assertTrue(result.allowed)
            self.assertTrue(result.exists)
            self.assertEqual(result.endpoint, "PROTO_FRANKENSTEIN_V0_FULL_LATENT_SEALED")

    def test_validate_proto_receipt_accepts_terminal_endpoint_alias(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td, "PROTO_FRANKENSTEIN_RUN_RECEIPT.json")
            path.write_text(
                json.dumps(
                    {
                        "terminal_endpoint": PROTO_OFFLOAD_ENDPOINT,
                        "recorded_at": "2026-08-05T00:00:00Z",
                        "runtime_storage": {
                            "storage": {"donor_weights_retained": False}
                        },
                    }
                )
            )
            result = validate_proto_offload_receipt(path)
            self.assertTrue(result.allowed)
            self.assertEqual(result.endpoint, PROTO_OFFLOAD_ENDPOINT)

    def test_validate_proto_receipt_rejects_invalid_recorded_at(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td, "PROTO_FRANKENSTEIN_RUN_RECEIPT.json")
            path.write_text(
                json.dumps(
                    {
                        "endpoint": "PROTO_FRANKENSTEIN_V0_FULL_LATENT_SEALED",
                        "recorded_at": "not-a-timestamp",
                        "runtime_storage": {"storage": {"donor_weights_retained": False}},
                    }
                )
            )
            result = validate_proto_offload_receipt(path)
            self.assertFalse(result.allowed)
            self.assertTrue(any("recorded_at" in r for r in result.reasons))

    def test_validate_proto_receipt_rejects_wrong_endpoint(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td, "PROTO_FRANKENSTEIN_RUN_RECEIPT.json")
            path.write_text(json.dumps({"endpoint": "WRONG"}))
            result = validate_proto_offload_receipt(path)
            self.assertFalse(result.allowed)
            self.assertTrue(any("endpoint mismatch" in r for r in result.reasons))

    def test_validate_proto_receipt_rejects_dry_run(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td, "PROTO_FRANKENSTEIN_RUN_RECEIPT.json")
            path.write_text(json.dumps({"endpoint": "PROTO_FRANKENSTEIN_V0_FULL_LATENT_SEALED", "dry_run": True}))
            result = validate_proto_offload_receipt(path)
            self.assertFalse(result.allowed)
            self.assertTrue(any("dry_run" in r for r in result.reasons))

    def test_validate_proto_receipt_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td, "missing.json")
            result = validate_proto_offload_receipt(path)
            self.assertFalse(result.allowed)
            self.assertFalse(result.exists)


class TestAfterProtoMonitor(unittest.TestCase):
    def _frank_record(self, sandbox: Path, *, size_bytes: int = 3 * 1024**3):
        path = sandbox / "hf-cache" / "proto_donor" / "shard"
        return classify_object(
            str(path),
            sandbox_roots=[str(sandbox)],
            evictable_paths=[str(path)],
            active_references=0,
            receipt_sealed=True,
            successor_or_rejection_verified=True,
            rollback_preserved=True,
            remote_hash_verified=True,
            size_bytes=size_bytes,
        )

    def test_monitor_hardening_blocks_without_valid_proto(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            monitor = monitor_after_proto(
                proto_receipt_path=td_p / "missing.json",
                signals=_sig(free_disk_gib=100.0),
                cleanup_records=[
                    self._frank_record(td_p / "sandbox")
                ],
            )
            self.assertFalse(monitor.can_advance_after_proto)
            self.assertFalse(monitor.proto_check.allowed)
            self.assertEqual(len(monitor.notifications_sent), 0)
            self.assertFalse(monitor.cleanup_decisions)
            self.assertIn("proto offload seal not valid for after-proto actions", monitor.blockers)

    def test_monitor_pressure_red_blocks_advance(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            rec_path = td_p / "PROTO_FRANKENSTEIN_RUN_RECEIPT.json"
            rec_path.write_text(
                json.dumps(
                    {
                        "endpoint": PROTO_OFFLOAD_ENDPOINT,
                        "schema": "x",
                        "recorded_at": "2026-08-05T00:00:00Z",
                        "dry_run": False,
                        "runtime_storage": {"storage": {"donor_weights_retained": False}},
                    }
                )
            )
            sandbox = td_p / "sandbox"
            rec = self._frank_record(sandbox)
            monitor = monitor_after_proto(
                proto_receipt_path=rec_path,
                # The governor's RED disk band is below 25 GiB.  Exercise
                # the real RED path rather than the 25–40 GiB YELLOW band.
                signals=_sig(free_disk_gib=20.0),
                cleanup_records=[rec],
                cleanup_apply=False,
            )
            self.assertFalse(monitor.can_advance_after_proto)
            self.assertTrue(monitor.proto_check.allowed)
            self.assertIn("storage pressure RED blocks advancement", monitor.blockers)
            self.assertEqual(len(monitor.cleanup_decisions), 1)

    def test_monitor_allow_when_proto_valid_and_pressure_green(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            rec_path = td_p / "PROTO_FRANKENSTEIN_RUN_RECEIPT.json"
            rec_path.write_text(
                json.dumps(
                    {
                        "endpoint": "PROTO_FRANKENSTEIN_V0_FULL_LATENT_SEALED",
                        "schema": "x",
                        "recorded_at": "2026-08-05T00:00:00Z",
                        "dry_run": False,
                        "runtime_storage": {"storage": {"donor_weights_retained": False}},
                    }
                )
            )
            sandbox = td_p / "sandbox"
            rec = self._frank_record(sandbox)
            monitor = monitor_after_proto(
                proto_receipt_path=rec_path,
                signals=_sig(free_disk_gib=100),
                cleanup_records=[rec],
                cleanup_apply=True,
                free_bytes_before=10 * 1024**3,
                free_bytes_after=20 * 1024**3,
            )
            self.assertTrue(monitor.proto_check.allowed)
            self.assertTrue(monitor.can_advance_after_proto)
            self.assertEqual(len(monitor.cleanup_decisions), 1)
            self.assertEqual(monitor.cleanup_decisions[0].as_dict()["object_class"], "EVICTABLE")
            self.assertEqual(monitor.cleanup_receipt["schema"], "hawking.ascension.cleanup_receipt.v1")
            self.assertEqual(monitor.as_receipt()["schema"], MONITOR_SCHEMA)
            self.assertEqual(len(monitor.as_receipt()["notifications"]["sent"]), 1)

    def test_monitor_pressure_critical_blocks_cleanup_and_requires_human(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            rec_path = td_p / "PROTO_FRANKENSTEIN_RUN_RECEIPT.json"
            rec_path.write_text(
                json.dumps(
                    {
                        "endpoint": "PROTO_FRANKENSTEIN_V0_FULL_LATENT_SEALED",
                        "schema": "x",
                        "recorded_at": "2026-08-05T00:00:00Z",
                        "dry_run": False,
                        "runtime_storage": {"storage": {"donor_weights_retained": False}},
                    }
                )
            )
            sandbox = td_p / "sandbox"
            rec = self._frank_record(sandbox)
            monitor = monitor_after_proto(
                proto_receipt_path=rec_path,
                signals=_sig(free_disk_gib=5.0),
                cleanup_records=[rec],
                cleanup_apply=False,
            )
            self.assertFalse(monitor.can_advance_after_proto)
            self.assertFalse(monitor.cleanup_decisions)
            self.assertTrue(monitor.proto_check.allowed)
            kinds = {e["kind"] for e in monitor.notifications_sent}
            self.assertIn(NotificationKind.MEMORY_DISK_PRESSURE.value, kinds)
            self.assertIn(NotificationKind.HUMAN_DECISION_REQUIRED.value, kinds)

    def test_monitor_invalid_proto_event_authority_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            rec_path = td_p / "PROTO_FRANKENSTEIN_RUN_RECEIPT.json"
            rec_path.write_text(
                json.dumps(
                    {
                        "endpoint": "PROTO_FRANKENSTEIN_V0_FULL_LATENT_SEALED",
                        "schema": "x",
                        "recorded_at": "2026-08-05T00:00:00Z",
                        "dry_run": False,
                        "runtime_storage": {"storage": {"donor_weights_retained": False}},
                    }
                )
            )
            monitor = monitor_after_proto(
                proto_receipt_path=rec_path,
                signals=_sig(free_disk_gib=100),
                proto_event_authority="definitely-not-a-authority",
            )
            self.assertTrue(monitor.proto_check.allowed)
            self.assertIn(
                "invalid proto event authority requested; defaulted to sealed receipt",
                monitor.blockers,
            )
            self.assertTrue(
                any(
                    event["kind"] == NotificationKind.PROTO_SEALED.value
                    and event["authority"] == "sealed_receipt"
                    for event in monitor.notifications_sent
                )
            )


if __name__ == "__main__":
    unittest.main()
