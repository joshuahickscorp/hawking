from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import doctor_v5_remaining_scratch_ledger as ledger


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


class Fixture:
    def __init__(self, root: Path, *, mode: str = "resident",
                 scratch: int = 30) -> None:
        self.root = root
        self.output = root / "run"
        self.source = root / "source"
        self.request_path = self.output / "request.json"
        self.checkpoint_path = self.output / "checkpoint.json"
        (self.output / "bundle/shards").mkdir(parents=True)
        (self.output / "evaluation/reconstruction").mkdir(parents=True)
        self.source.mkdir()
        source_rows = []
        for ordinal, size in enumerate((3, 4)):
            path = self.source / f"model-{ordinal:05d}.safetensors"
            path.write_bytes(bytes([ordinal + 1]) * size)
            source_rows.append({
                "bytes": size, "name": path.name, "ordinal": ordinal,
                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        self.request = {
            "schema": ledger.REQUEST_SCHEMA,
            "request_id": "strand-ladder-test-fixture",
            "label": "0.5B", "model_family": "qwen2.5-dense",
            "campaign_binding": {}, "codec": {},
            "source": {
                "census_path": str(root / "census.json"),
                "census_sha256": "1" * 64, "model_dir": str(self.source),
                "shards": source_rows, "source_manifest_sha256": "2" * 64,
            },
            "parameter_manifest": {}, "execution": {},
            "evaluation": {"mode": mode, "retain_dense_reconstruction": False},
            "doctor_hook": {},
            "resources": {
                "disk_reserve_bytes": ledger.DISK_RESERVE_BYTES,
                "scratch_budget_bytes": scratch,
            },
            "output_root": str(self.output), "evidence_policy": {},
        }
        _write_json(self.request_path, self.request)

        self.packed = []
        self.recon = []
        for ordinal, (packed_size, recon_size) in enumerate(((5, 13), (7, 17))):
            packed = self.output / f"bundle/shards/{ordinal:05d}.strand"
            packed.write_bytes(bytes([10 + ordinal]) * packed_size)
            recon = self.output / (
                f"evaluation/reconstruction/{ordinal:05d}.safetensors"
            )
            # Ordinal 1 exists but is not checkpoint-finalized in the resident
            # baseline.  Its bytes therefore must not reduce scratch.
            recon.write_bytes(bytes([20 + ordinal]) * recon_size)
            self.packed.append(packed); self.recon.append(recon)

        plan = ["preflight", "metadata"]
        for ordinal in range(2):
            plan.extend([
                f"passthrough:{ordinal:05d}", f"encode:{ordinal:05d}",
                f"attest:{ordinal:05d}",
            ])
            if mode == "resident":
                plan.append(f"decode:{ordinal:05d}")
        plan.extend(ledger.RESIDENT_SUFFIX if mode == "resident"
                    else ledger.DEFERRED_SUFFIX)
        if mode == "resident":
            completed = plan[:plan.index("decode:00001")]
        else:
            completed = plan[:plan.index("bundle_manifest")]
        units: dict[str, dict[str, object]] = {}
        for unit in completed:
            units[unit] = {"completed_at": "2026-07-14T00:00:00+00:00"}
            match = ledger.ORDINAL_UNIT_RE.fullmatch(unit)
            if match is None:
                continue
            phase, raw = match.groups(); ordinal = int(raw)
            if phase == "encode":
                units[unit]["artifact"] = _artifact(self.packed[ordinal])
            elif phase == "attest":
                units[unit]["archive"] = _artifact(self.packed[ordinal])
            elif phase == "decode":
                units[unit]["artifact"] = _artifact(self.recon[ordinal])
        self.checkpoint = {
            "schema": ledger.CHECKPOINT_SCHEMA,
            "request_sha256": _file_sha(self.request_path),
            "created_at": "2026-07-14T00:00:00+00:00",
            "updated_at": "2026-07-14T00:01:00+00:00", "status": "running",
            "plan": plan, "completed_units": completed, "units": units,
            "stop_requested": False,
        }
        _write_json(self.checkpoint_path, self.checkpoint)

    def rewrite_request(self) -> None:
        _write_json(self.request_path, self.request)
        self.checkpoint["request_sha256"] = _file_sha(self.request_path)
        _write_json(self.checkpoint_path, self.checkpoint)

    def rewrite_checkpoint(self) -> None:
        _write_json(self.checkpoint_path, self.checkpoint)


class RemainingScratchLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build(self, fixture: Fixture, projected: int = 20) -> dict[str, object]:
        return ledger.build_ledger(
            fixture.request_path, projected_packed_output_bytes=projected,
            workspace_root=self.root,
        )

    def test_resident_counts_only_exact_completed_decode_chain(self) -> None:
        fixture = Fixture(self.root)
        receipt = self._build(fixture)
        self.assertEqual([], ledger.validate_receipt(receipt))
        self.assertEqual(13, receipt["durable_materialized_bytes"])
        self.assertEqual(17, receipt["remaining_scratch_bytes"])
        self.assertEqual(12, receipt["durable_attested_packed_bytes"])
        self.assertEqual(8, receipt["projected_remaining_packed_output_bytes"])
        self.assertEqual(
            ledger.DISK_RESERVE_BYTES + 25, receipt["required_free_bytes"]
        )
        self.assertTrue(receipt["ordinals"][0]["reconstruction_counted"])
        self.assertFalse(receipt["ordinals"][1]["reconstruction_counted"])
        self.assertFalse(receipt["validation_contract"][
            "artifact_payload_content_rehashed"
        ])
        self.assertFalse(receipt["isolation"]["activation_permitted"])

    def test_deferred_mode_never_subtracts_reconstruction_scratch(self) -> None:
        fixture = Fixture(self.root, mode="deferred")
        receipt = self._build(fixture)
        self.assertEqual(0, receipt["durable_materialized_bytes"])
        self.assertEqual(30, receipt["remaining_scratch_bytes"])
        self.assertTrue(all(not row["decode_completed"] for row in receipt["ordinals"]))

    def test_request_and_checkpoint_are_not_mutated(self) -> None:
        fixture = Fixture(self.root)
        before = {
            path.relative_to(self.root): (
                path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_size
            )
            for path in self.root.rglob("*") if path.is_file()
        }
        self._build(fixture)
        after = {
            path.relative_to(self.root): (
                path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_size
            )
            for path in self.root.rglob("*") if path.is_file()
        }
        self.assertEqual(before, after)

    def test_checkpoint_request_hash_mismatch_is_refused(self) -> None:
        fixture = Fixture(self.root)
        fixture.checkpoint["request_sha256"] = "0" * 64
        fixture.rewrite_checkpoint()
        with self.assertRaisesRegex(ledger.ScratchLedgerError, "request-file hash"):
            self._build(fixture)

    def test_unknown_mode_is_refused(self) -> None:
        fixture = Fixture(self.root)
        fixture.request["evaluation"]["mode"] = "streaming-ish"
        fixture.rewrite_request()
        with self.assertRaisesRegex(ledger.ScratchLedgerError, "evaluation mode"):
            self._build(fixture)

    def test_disk_reserve_must_be_exactly_150_decimal_gb(self) -> None:
        fixture = Fixture(self.root)
        fixture.request["resources"]["disk_reserve_bytes"] -= 1
        fixture.rewrite_request()
        with self.assertRaisesRegex(ledger.ScratchLedgerError, "exactly 150"):
            self._build(fixture)

    def test_duplicate_source_ordinal_is_refused(self) -> None:
        fixture = Fixture(self.root)
        fixture.request["source"]["shards"][1]["ordinal"] = 0
        fixture.rewrite_request()
        with self.assertRaisesRegex(ledger.ScratchLedgerError, "ordinals"):
            self._build(fixture)

    def test_duplicate_plan_unit_is_refused(self) -> None:
        fixture = Fixture(self.root)
        fixture.checkpoint["plan"].append("receipt")
        fixture.rewrite_checkpoint()
        with self.assertRaisesRegex(ledger.ScratchLedgerError, "unique string list"):
            self._build(fixture)

    def test_malformed_checkpoint_artifact_hash_is_refused(self) -> None:
        fixture = Fixture(self.root)
        fixture.checkpoint["units"]["attest:00000"]["archive"]["sha256"] = "bad"
        fixture.rewrite_checkpoint()
        with self.assertRaisesRegex(ledger.ScratchLedgerError, "identity syntax"):
            self._build(fixture)

    def test_valid_basename_metadata_identity_is_not_a_filesystem_artifact(self) -> None:
        fixture = Fixture(self.root)
        fixture.checkpoint["units"]["metadata"]["source_metadata"] = [{
            "name": "config.json", "sha256": "a" * 64, "bytes": 663,
        }]
        fixture.rewrite_checkpoint()
        receipt = self._build(fixture)
        self.assertEqual([], ledger.validate_receipt(receipt))
        observed = receipt["artifact_identity_observations"]
        self.assertFalse(any(row["path"].endswith("config.json") for row in observed))

    def test_auxiliary_workspace_artifact_is_verified_but_not_counted(self) -> None:
        fixture = Fixture(self.root)
        auxiliary = self.root / "shared-cache/receipt.json"
        auxiliary.parent.mkdir()
        auxiliary.write_bytes(b"sealed auxiliary receipt")
        fixture.checkpoint["units"]["metadata"]["shared_cache"] = _artifact(auxiliary)
        fixture.rewrite_checkpoint()
        receipt = self._build(fixture)
        self.assertEqual([], ledger.validate_receipt(receipt))
        self.assertEqual(13, receipt["durable_materialized_bytes"])
        self.assertEqual(12, receipt["durable_attested_packed_bytes"])
        self.assertTrue(any(row["path"] == "shared-cache/receipt.json"
                            for row in receipt["artifact_identity_observations"]))

    def test_malformed_basename_metadata_identity_is_refused(self) -> None:
        fixture = Fixture(self.root)
        fixture.checkpoint["units"]["metadata"]["source_metadata"] = [{
            "name": "../config.json", "sha256": "a" * 64, "bytes": 663,
        }]
        fixture.rewrite_checkpoint()
        with self.assertRaisesRegex(ledger.ScratchLedgerError, "identity syntax"):
            self._build(fixture)

    def test_conflicting_duplicate_archive_identity_is_refused(self) -> None:
        fixture = Fixture(self.root)
        fixture.checkpoint["units"]["attest:00000"]["archive"]["sha256"] = "f" * 64
        fixture.rewrite_checkpoint()
        with self.assertRaisesRegex(ledger.ScratchLedgerError, "conflicting identities"):
            self._build(fixture)

    def test_path_escape_is_refused(self) -> None:
        fixture = Fixture(self.root)
        fixture.checkpoint["units"]["decode:00000"]["artifact"] = _artifact(
            fixture.source / "model-00000.safetensors"
        )
        fixture.rewrite_checkpoint()
        with self.assertRaisesRegex(ledger.ScratchLedgerError, "escapes workspace"):
            self._build(fixture)

    def test_symlink_artifact_is_refused_without_following(self) -> None:
        fixture = Fixture(self.root)
        fixture.recon[0].unlink()
        os.symlink(fixture.source / "model-00000.safetensors", fixture.recon[0])
        with self.assertRaisesRegex(ledger.ScratchLedgerError, "symlink"):
            self._build(fixture)

    def test_partial_artifact_is_refused_even_when_not_checkpointed(self) -> None:
        fixture = Fixture(self.root)
        partial = fixture.output / (
            "evaluation/reconstruction/.00001.safetensors.partial.123"
        )
        partial.write_bytes(b"unfinished")
        with self.assertRaisesRegex(ledger.ScratchLedgerError, "partial artifact"):
            self._build(fixture)

    def test_size_change_is_refused_without_content_rehash(self) -> None:
        fixture = Fixture(self.root)
        fixture.recon[0].write_bytes(b"short")
        with self.assertRaisesRegex(ledger.ScratchLedgerError, "byte count"):
            self._build(fixture)

    def test_hard_link_alias_is_refused_as_duplicate_materialization(self) -> None:
        fixture = Fixture(self.root)
        os.link(fixture.recon[0], self.root / "second-link-to-reconstruction")
        with self.assertRaisesRegex(ledger.ScratchLedgerError, "hard-linked"):
            self._build(fixture)

    def test_observation_race_is_refused(self) -> None:
        fixture = Fixture(self.root)
        stable = (1, 2, 3, 5, 6, 7)
        changed = (1, 2, 3, 5, 8, 9)
        with mock.patch.object(
                ledger, "_stat_identity",
                side_effect=[stable, stable, stable, changed]):
            with self.assertRaisesRegex(ledger.ScratchLedgerError, "changed while"):
                ledger._stable_regular_identity(
                    fixture.packed[0], self.root, expected_bytes=5,
                    checkpoint_sha256="a" * 64,
                )

    def test_reconstruction_over_declared_scratch_is_refused(self) -> None:
        fixture = Fixture(self.root, scratch=12)
        with self.assertRaisesRegex(ledger.ScratchLedgerError, "exceed declared"):
            self._build(fixture)

    def test_packed_output_projection_cannot_undercount_durable_bytes(self) -> None:
        fixture = Fixture(self.root)
        with self.assertRaisesRegex(ledger.ScratchLedgerError, "exceed projected"):
            self._build(fixture, projected=11)

    def test_duplicate_json_keys_are_refused(self) -> None:
        fixture = Fixture(self.root)
        fixture.request_path.write_text(
            '{"schema":"one","schema":"two"}', encoding="utf-8"
        )
        with self.assertRaisesRegex(ledger.ScratchLedgerError, "duplicate JSON key"):
            self._build(fixture)

    def test_resealed_accounting_tamper_is_detected(self) -> None:
        fixture = Fixture(self.root)
        receipt = self._build(fixture)
        damaged = copy.deepcopy(receipt)
        damaged["remaining_scratch_bytes"] += 1
        damaged["receipt_sha256"] = ledger._hash_value(
            ledger._without(damaged, "receipt_sha256")
        )
        errors = ledger.validate_receipt(damaged)
        self.assertTrue(any("remaining-scratch equation" in row for row in errors))

    def test_resealed_unknown_fields_are_rejected_at_every_receipt_level(self) -> None:
        fixture = Fixture(self.root)
        receipt = self._build(fixture)
        for mutate in (
                lambda value: value.__setitem__("unknown", True),
                lambda value: value["ordinals"][0].__setitem__("unknown", True),
                lambda value: value["artifact_identity_observations"][0].__setitem__(
                    "unknown", True),
                lambda value: value["request"].__setitem__("unknown", True)):
            damaged = copy.deepcopy(receipt)
            mutate(damaged)
            damaged["receipt_sha256"] = ledger._hash_value(
                ledger._without(damaged, "receipt_sha256"))
            self.assertTrue(ledger.validate_receipt(damaged))

    def test_resealed_non_single_link_identity_is_rejected(self) -> None:
        fixture = Fixture(self.root)
        receipt = self._build(fixture)
        for row in (receipt["request"], receipt["checkpoint"],
                    receipt["artifact_identity_observations"][0]):
            damaged = copy.deepcopy(receipt)
            if row is receipt["request"]:
                target = damaged["request"]
            elif row is receipt["checkpoint"]:
                target = damaged["checkpoint"]
            else:
                target = damaged["artifact_identity_observations"][0]
            target["identity"]["links"] = 2
            damaged["receipt_sha256"] = ledger._hash_value(
                ledger._without(damaged, "receipt_sha256"))
            self.assertTrue(any("binding" in error or "observation" in error
                                for error in ledger.validate_receipt(damaged)))


if __name__ == "__main__":
    unittest.main()
