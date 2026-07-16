from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
CONDENSE = HERE.parent
ROOT = CONDENSE.parents[1]
sys.path.insert(0, str(CONDENSE))

import doctor_v5_source_seal as seal
import doctor_v5_strand_ladder_block_parallel_adapter as adapter
import doctor_v5_qwen_treatment_block_parallel_adapter as treatment_adapter


VOLUME = {
    "filesystem": "apfs",
    "kind": "darwin-apfs-volume-uuid",
    "uuid": "a180c02c-7336-4e5b-a4ca-f765a690a50f",
}


class _Lease:
    def close(self) -> None:
        pass


class SourceSealV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.bin"
        self.source.write_bytes(b"doctor-v5-source-seal-v2" * 4096)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _v2(self) -> dict[str, object]:
        with mock.patch.object(seal, "_volume_identity", return_value=VOLUME):
            digest, size, identity = seal._hash_regular(self.source)
        document: dict[str, object] = {
            "schema": seal.SCHEMA,
            "label": "fixture",
            "created_at": "2026-07-15T00:00:00+00:00",
            "source_manifest_sha256": "0" * 64,
            "parameter_manifest": {
                "path": str(self.source), "sha256": digest, "bytes": size,
            },
            "census": {"path": str(self.source), "sha256": digest, "bytes": size},
            "shards": [{
                "ordinal": 0, "name": self.source.name, "path": str(self.source),
                "sha256": digest, "bytes": size, "identity": identity,
            }],
            "verification": dict(seal.V2_VERIFICATION),
            "migration": None,
            "source_deletion_permitted": False,
        }
        document["seal_sha256"] = seal._hash_value(document)
        return document

    def _persist(self, document: dict[str, object]) -> Path:
        seal_root = self.root / "seals"
        seal._atomic_json(seal_root / "fixture.json", document)
        return seal_root

    def test_v2_accepts_only_device_number_drift_with_stable_apfs_uuid(self) -> None:
        document = self._v2()
        identity = document["shards"][0]["identity"]  # type: ignore[index]
        identity["st_dev_at_seal"] += 99  # type: ignore[index,operator]
        document["seal_sha256"] = seal._hash_value(
            seal._without(document, "seal_sha256")
        )
        seal_root = self._persist(document)
        row = document["shards"][0]  # type: ignore[index]
        with mock.patch.object(seal, "_volume_identity", return_value=VOLUME):
            self.assertEqual(
                (row["sha256"], row["bytes"]),  # type: ignore[index]
                seal.lookup(self.source, seal_root=seal_root),
            )

    def test_v2_rejects_volume_uuid_drift(self) -> None:
        document = self._v2()
        seal_root = self._persist(document)
        other = {**VOLUME, "uuid": "b180c02c-7336-4e5b-a4ca-f765a690a50f"}
        with mock.patch.object(seal, "_volume_identity", return_value=other):
            with self.assertRaises(seal.SourceSealError):
                seal.lookup(self.source, seal_root=seal_root)

    def test_v2_rejects_every_nondevice_identity_drift(self) -> None:
        for field in ("st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
            with self.subTest(field=field):
                document = self._v2()
                row = document["shards"][0]  # type: ignore[index]
                row["identity"][field] += 1  # type: ignore[index,operator]
                if field == "st_size":
                    row["bytes"] += 1  # type: ignore[index,operator]
                document["seal_sha256"] = seal._hash_value(
                    seal._without(document, "seal_sha256")
                )
                seal_root = self.root / f"seals-{field}"
                seal._atomic_json(seal_root / "fixture.json", document)
                with mock.patch.object(seal, "_volume_identity", return_value=VOLUME):
                    with self.assertRaises(seal.SourceSealError):
                        seal.lookup(self.source, seal_root=seal_root)

    def test_symlink_never_reuses_sealed_hash(self) -> None:
        document = self._v2()
        seal_root = self._persist(document)
        alias = self.root / "alias.bin"
        alias.symlink_to(self.source)
        with mock.patch.object(seal, "_volume_identity", return_value=VOLUME):
            self.assertIsNone(seal.lookup(alias, seal_root=seal_root))

    def test_v1_migration_full_revalidates_and_archives_exact_bytes(self) -> None:
        data = self.source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        stat_row = self.source.stat()
        old_identity = seal._identity(stat_row)
        old_identity["st_dev"] += 7
        old: dict[str, object] = {
            "schema": seal.SCHEMA_V1,
            "label": "fixture",
            "created_at": "2026-07-14T00:00:00+00:00",
            "source_manifest_sha256": "1" * 64,
            "parameter_manifest": {
                "path": str(self.source), "sha256": digest, "bytes": len(data),
            },
            "census": {"path": str(self.source), "sha256": digest, "bytes": len(data)},
            "shards": [{
                "ordinal": 0, "name": self.source.name, "path": str(self.source),
                "sha256": digest, "bytes": len(data), "identity": old_identity,
            }],
            "verification": dict(seal.V1_VERIFICATION),
            "source_deletion_permitted": False,
        }
        old["seal_sha256"] = seal._hash_value(old)
        old_path = self.root / "fixture-v1.json"
        output = self.root / "fixture-v2.json"
        archive = self.root / "archive" / "fixture-v1.json"
        seal._atomic_json(old_path, old)
        exact_old = old_path.read_bytes()
        with mock.patch.object(seal, "_volume_identity", return_value=VOLUME), \
                mock.patch.object(seal, "_acquire_exclusive_heavy_lease",
                                  return_value=_Lease()):
            migrated = seal.migrate_v1(
                old_path, output=output, archive=archive, workers=2
            )
            self.assertEqual([], seal.validate_document(migrated, verify_structural=True))
        self.assertEqual(exact_old, archive.read_bytes())
        self.assertEqual(seal.SCHEMA, migrated["schema"])
        transition = migrated["migration"]["device_transitions"][0]
        self.assertEqual(old_identity["st_dev"], transition["st_dev_v1"])
        self.assertEqual(stat_row.st_dev, transition["st_dev_at_migration"])
        self.assertTrue(migrated["migration"]["content_revalidated"])

    def test_v1_migration_rejects_content_drift(self) -> None:
        data = self.source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        identity = seal._identity(self.source.stat())
        old: dict[str, object] = {
            "schema": seal.SCHEMA_V1, "label": "fixture",
            "created_at": "2026-07-14T00:00:00+00:00",
            "source_manifest_sha256": "2" * 64,
            "parameter_manifest": {
                "path": str(self.source), "sha256": digest, "bytes": len(data),
            },
            "census": {"path": str(self.source), "sha256": digest, "bytes": len(data)},
            "shards": [{
                "ordinal": 0, "name": self.source.name, "path": str(self.source),
                "sha256": digest, "bytes": len(data), "identity": identity,
            }],
            "verification": dict(seal.V1_VERIFICATION),
            "source_deletion_permitted": False,
        }
        old["seal_sha256"] = seal._hash_value(old)
        old_path = self.root / "tampered-v1.json"
        seal._atomic_json(old_path, old)
        self.source.write_bytes(data + b"tamper")
        with mock.patch.object(seal, "_volume_identity", return_value=VOLUME), \
                mock.patch.object(seal, "_acquire_exclusive_heavy_lease",
                                  return_value=_Lease()):
            with self.assertRaises(seal.SourceSealError):
                seal.migrate_v1(
                    old_path, output=self.root / "never.json",
                    archive=self.root / "archive.json", workers=1,
                )

    def test_generic_hash_reuse_supports_outer_abi_attribute(self) -> None:
        document = self._v2()
        seal_root = self._persist(document)
        calls: list[Path] = []

        def original(path: Path) -> tuple[str, int]:
            calls.append(Path(path))
            return "f" * 64, 1

        module = SimpleNamespace(_sha_file=original)
        seal.install_hash_reuse(module, seal_root=seal_root, attribute="_sha_file")
        seal.install_hash_reuse(module, seal_root=seal_root, attribute="_sha_file")
        with mock.patch.object(seal, "_volume_identity", return_value=VOLUME):
            row = document["shards"][0]  # type: ignore[index]
            self.assertEqual(
                (row["sha256"], row["bytes"]), module._sha_file(self.source)
            )
        self.assertEqual([], calls)
        self.assertEqual(("f" * 64, 1), module._sha_file(self.root / "unsealed.bin"))
        self.assertEqual([self.root / "unsealed.bin"], calls)


class AdapterSealAndTransitionTests(unittest.TestCase):
    def test_outer_abi_is_patched_before_request_file_validation(self) -> None:
        abi = adapter._load_module("doctor_v5_source_seal_test_abi", adapter._BASE.ABI_PATH)
        self.assertTrue(getattr(abi._sha_file, "_doctor_v5_source_seal_reuse", False))
        treatment_abi = treatment_adapter._load_module(
            "doctor_v5_source_seal_test_treatment_abi", treatment_adapter._BASE.ABI_PATH
        )
        self.assertTrue(
            getattr(treatment_abi._sha_file, "_doctor_v5_source_seal_reuse", False)
        )

    def test_resume_transition_retry_is_exact_and_never_rewritten(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            internal = Path(raw)
            outer_sha = "b" * 64
            old_id = "strand-ladder-14b-4-" + "a" * 16
            new_id = "strand-ladder-14b-4-" + outer_sha[:16]
            existing = {
                "schema": "fixture", "request_id": old_id, "label": "14B",
                "codec": {"rate_id": "4", "tensor_scope": "all-2d"},
                "campaign_binding": {"branch": "codec_control"},
                "semantics": {"threads": 20},
            }
            candidate = copy.deepcopy(existing)
            candidate["request_id"] = new_id
            inner_path = internal / "request.json"
            seal._atomic_json(inner_path, existing)
            inner_sha = adapter._BASE._hash_file(inner_path)[0]
            checkpoint = {
                "request_sha256": inner_sha, "status": "running",
                "completed_units": ["preflight"], "units": {},
            }
            seal._atomic_json(internal / "checkpoint.json", checkpoint)

            class Worker:
                @staticmethod
                def _validate_request(path: Path):
                    return existing, adapter._BASE._hash_file(path)[0], []

                @staticmethod
                def _plan(request, stats):
                    return ["preflight"]

                @staticmethod
                def _paths(output, count):
                    return {}

                @staticmethod
                def _checkpoint(path, request_sha, plan, paths, stats):
                    self.assertEqual(inner_sha, request_sha)
                    self.assertEqual(["preflight"], plan)
                    return json.loads(path.read_text(encoding="utf-8"))

            abi = SimpleNamespace(atomic_json=seal._atomic_json)
            with mock.patch.object(adapter, "_ORIGINAL_BUILD_INTERNAL",
                                   return_value=candidate), \
                    mock.patch.object(adapter, "_LOADED_WORKER", Worker), \
                    mock.patch.object(adapter, "_LOADED_ABI", abi):
                first = adapter._build_internal(
                    {"request_sha256": outer_sha}, {}, {}, internal
                )
                transition = internal / adapter.TRANSITION_NAME
                first_bytes = transition.read_bytes()
                second = adapter._build_internal(
                    {"request_sha256": outer_sha}, {}, {}, internal
                )
                second_bytes = transition.read_bytes()
                progressed = {
                    **checkpoint,
                    "status": "checkpointed-stop",
                    "completed_units": ["preflight", "metadata"],
                    "units": {"preflight": {}, "metadata": {}},
                }
                seal._atomic_json(internal / "checkpoint.json", progressed)
                third = adapter._build_internal(
                    {"request_sha256": outer_sha}, {}, {}, internal
                )
                third_bytes = transition.read_bytes()
            self.assertEqual(existing, first)
            self.assertEqual(existing, second)
            self.assertEqual(existing, third)
            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(first_bytes, third_bytes)
            receipt = json.loads(first_bytes)
            self.assertEqual(inner_sha, receipt["preserved_inner_request"]["sha256"])
            self.assertTrue(receipt["verification"]["only_request_id_changed"])


if __name__ == "__main__":
    unittest.main()
