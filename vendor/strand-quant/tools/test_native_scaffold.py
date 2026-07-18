from __future__ import annotations

import json
import fcntl
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import native_build
import native_probe


class NativeBuildSafetyTests(unittest.TestCase):
    @staticmethod
    def _seal(path: Path, document: dict[str, object]) -> dict[str, object]:
        document = dict(document)
        document["document_sha256"] = native_build.canonical_sha256(document)
        path.write_text(
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return document

    @staticmethod
    def _identity(path: Path) -> dict[str, object]:
        path = path.resolve()
        return {
            "path": str(path),
            "sha256": native_build.sha256_file(path),
            "bytes": path.stat().st_size,
        }

    def _execution_fixture(
        self, root: Path
    ) -> tuple[Path, dict[str, object], dict[str, object]]:
        program = root / "quantize-model-native"
        program.write_bytes(b"instrumented-program")
        program.chmod(0o700)
        build_receipt = root / "build.json"
        build_receipt.write_bytes(b"sealed-build-receipt")
        build: dict[str, object] = {
            "path": str(build_receipt.resolve()),
            "receipt_sha256": native_build.sha256_file(build_receipt),
            "program": str(program.resolve()),
            "program_sha256": native_build.sha256_file(program),
            "source_manifest_sha256": "1" * 64,
            "host_sha256": "2" * 64,
        }

        cwd = root / "run"
        profiles_dir = root / "raw"
        cwd.mkdir()
        profiles_dir.mkdir()
        input_bundle = root / "input.bundle"
        input_bundle.write_bytes(b"sealed representative input")
        output_bundle = root / "output.bundle"
        output_bundle.write_bytes(b"exact training output")
        profile = profiles_dir / "training-123.profraw"
        profile.write_bytes(b"physical profile counters")
        profile_stat = profile.stat()
        started = profile_stat.st_mtime_ns - 1_000_000
        finished = profile_stat.st_mtime_ns + 1_000_000

        environment = {
            "LLVM_PROFILE_FILE": str(profiles_dir / "training-%p.profraw"),
            "RUST_BACKTRACE": "1",
        }
        invocation_body = {
            "argv": [str(program.resolve()), "--fixture", str(input_bundle.resolve())],
            "cwd": str(cwd.resolve()),
            "environment": environment,
        }
        invocation = dict(invocation_body)
        invocation["sha256"] = native_build.canonical_sha256(invocation_body)

        admission_path = root / "admission.json"
        self._seal(
            admission_path,
            {
                "schema": native_build.TRAINING_ADMISSION_SCHEMA,
                "status": "admitted",
                "generated_unix_ns": started - 1_000_000,
                "instrumented_build_receipt_sha256": build["receipt_sha256"],
                "instrumented_program_sha256": build["program_sha256"],
                "invocation_sha256": invocation["sha256"],
                "active_heavy_owner_count": 0,
                "owners_rechecked_under_lease": True,
                "exclusive_heavy_lease_held": True,
                "resource_health_ok": True,
                "owner_pattern_source": self._identity(
                    native_build.OWNER_PATTERN_SOURCE
                ),
                "owner_snapshot_sha256": "3" * 64,
                "resource_snapshot_sha256": "4" * 64,
            },
        )
        output_path = root / "output.json"
        self._seal(
            output_path,
            {
                "schema": native_build.TRAINING_OUTPUT_SCHEMA,
                "status": "pass",
                "program_sha256": build["program_sha256"],
                "invocation_sha256": invocation["sha256"],
                "input_bundle_sha256": native_build.sha256_file(input_bundle),
                "output_bundle": self._identity(output_bundle),
            },
        )
        parity_path = root / "parity.json"
        self._seal(
            parity_path,
            {
                "schema": native_build.TRAINING_PARITY_SCHEMA,
                "status": "pass",
                "program_sha256": build["program_sha256"],
                "invocation_sha256": invocation["sha256"],
                "input_bundle_sha256": native_build.sha256_file(input_bundle),
                "output_bundle_sha256": native_build.sha256_file(output_bundle),
                "exact_output": True,
                "skipped_cases": 0,
            },
        )
        execution: dict[str, object] = {
            "schema": native_build.EXECUTION_SCHEMA,
            "status": "pass",
            "instrumented_build_receipt": {
                "path": build["path"],
                "sha256": build["receipt_sha256"],
            },
            "program": self._identity(program),
            "invocation": invocation,
            "input_bundle": self._identity(input_bundle),
            "admission_receipt": self._identity(admission_path),
            "output_receipt": self._identity(output_path),
            "parity_receipt": self._identity(parity_path),
            "run": {
                "started_unix_ns": started,
                "finished_unix_ns": finished,
                "exit_code": 0,
                "skipped": False,
            },
            "profile_generation": {
                "directory": str(profiles_dir.resolve()),
                "before_entries": [],
                "after_entries": [
                    {
                        **self._identity(profile),
                        "mtime_ns": profile_stat.st_mtime_ns,
                    }
                ],
            },
            "resources": {
                "owner_free_before": True,
                "owner_free_after": True,
                "active_heavy_owner_count_before": 0,
                "active_heavy_owner_count_after": 0,
                "exclusive_heavy_lease_held_throughout": True,
                "memory_pressure_start": "normal",
                "memory_pressure_end": "normal",
                "thermal_start": "nominal",
                "thermal_end": "fair",
                "swap_start_bytes": 0,
                "swap_end_bytes": 0,
                "peak_rss_bytes": 1024,
                "cpu_seconds": 0.5,
                "wall_seconds": 1.0,
                "disk_free_start_bytes": 10_000,
                "disk_free_end_bytes": 9_000,
                "scratch_peak_bytes": 512,
            },
            "source_files_deleted": False,
            "runtime_defaults_changed": False,
        }
        execution_path = root / "execution.json"
        self._seal(execution_path, execution)
        return execution_path, build, execution

    def test_staging_escape_is_rejected(self) -> None:
        with self.assertRaises(native_build.BuildError):
            native_build.admitted_path(Path("/tmp/strand-native-escape"), label="escape")

    def test_existing_build_target_is_rejected(self) -> None:
        native_build.ADMITTED_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=native_build.ADMITTED_ROOT) as directory:
            with self.assertRaises(native_build.BuildError):
                native_build.admitted_target(Path(directory))

    def test_atomic_receipt_and_profile_publication(self) -> None:
        native_build.ADMITTED_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=native_build.ADMITTED_ROOT) as directory:
            root = Path(directory)
            receipt = root / "receipt.json"
            native_build.atomic_write(receipt, {"status": "pass"})
            self.assertEqual(json.loads(receipt.read_text()), {"status": "pass"})
            with self.assertRaises(native_build.BuildError):
                native_build.atomic_write(receipt, {"status": "replacement"})

            temporary = root / "worker.profdata"
            output = root / "merged.profdata"
            temporary.write_bytes(b"profile")
            native_build.publish_profile(temporary, output)
            self.assertEqual(output.read_bytes(), b"profile")
            self.assertFalse(temporary.exists())

    def test_unsealed_training_receipt_is_rejected(self) -> None:
        native_build.ADMITTED_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=native_build.ADMITTED_ROOT) as directory:
            receipt = Path(directory) / "training.json"
            receipt.write_text('{"schema":"wrong"}\n', encoding="utf-8")
            with self.assertRaises(native_build.BuildError):
                native_build.validate_training_receipt(receipt)

    def test_duplicate_raw_profiles_are_rejected(self) -> None:
        native_build.ADMITTED_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=native_build.ADMITTED_ROOT) as directory:
            profile = Path(directory) / "sample.profraw"
            profile.write_bytes(b"profile")
            with self.assertRaises(native_build.BuildError):
                native_build.profile_identities([profile, profile])

    def test_unsealed_merge_receipt_is_rejected_before_profile_use(self) -> None:
        native_build.ADMITTED_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=native_build.ADMITTED_ROOT) as directory:
            root = Path(directory)
            profile = root / "merged.profdata"
            profile.write_bytes(b"profile")
            receipt = root / "merge.json"
            receipt.write_text(
                json.dumps(
                    {
                        "schema": native_build.MERGE_SCHEMA,
                        "status": "pass",
                        "output": str(profile),
                        "output_sha256": native_build.sha256_file(profile),
                        "document_sha256": "0" * 64,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(native_build.BuildError):
                native_build.validate_profile_authority(profile, receipt)

    def test_hash_bound_execution_receipt_is_accepted(self) -> None:
        native_build.ADMITTED_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=native_build.ADMITTED_ROOT) as directory:
            execution, build, _ = self._execution_fixture(Path(directory))
            evidence = native_build.validate_pgo_execution_receipt(execution, build=build)
            self.assertEqual(evidence["program"]["sha256"], build["program_sha256"])
            self.assertEqual(len(evidence["profiles"]), 1)

    def test_execution_receipt_rejects_resealed_adversarial_claims(self) -> None:
        native_build.ADMITTED_ROOT.mkdir(parents=True, exist_ok=True)
        mutations = (
            ("failed-run", lambda document: document["run"].update(exit_code=9)),
            (
                "reused-profile-directory",
                lambda document: document["profile_generation"].update(
                    before_entries=["old.profraw"]
                ),
            ),
            (
                "owner-present-after",
                lambda document: document["resources"].update(
                    active_heavy_owner_count_after=1
                ),
            ),
            (
                "malformed-thermal-type",
                lambda document: document["resources"].update(
                    thermal_start=["nominal"]
                ),
            ),
            (
                "wrong-program-invocation",
                lambda document: document["invocation"].update(
                    argv=["/tmp/not-the-instrumented-program"]
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                dir=native_build.ADMITTED_ROOT
            ) as directory:
                root = Path(directory)
                _, build, document = self._execution_fixture(root)
                mutate(document)
                path = root / f"execution-{label}.json"
                self._seal(path, document)
                with self.assertRaises(native_build.BuildError):
                    native_build.validate_pgo_execution_receipt(path, build=build)

    def test_execution_receipt_rejects_resealed_inexact_parity(self) -> None:
        native_build.ADMITTED_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=native_build.ADMITTED_ROOT) as directory:
            root = Path(directory)
            _, build, document = self._execution_fixture(root)
            original_parity = Path(document["parity_receipt"]["path"])
            parity = json.loads(original_parity.read_text())
            parity.pop("document_sha256")
            parity["exact_output"] = False
            bad_parity = root / "parity-inexact.json"
            self._seal(bad_parity, parity)
            document["parity_receipt"] = self._identity(bad_parity)
            bad_execution = root / "execution-inexact.json"
            self._seal(bad_execution, document)
            with self.assertRaises(native_build.BuildError):
                native_build.validate_pgo_execution_receipt(bad_execution, build=build)


class NativeProbeSafetyTests(unittest.TestCase):
    def test_probe_staging_escape_is_rejected(self) -> None:
        with self.assertRaises(native_probe.ProbeAdmissionError):
            native_probe.confined(Path("/tmp/strand-probe-escape"), label="escape")

    def test_admission_receipt_is_exclusive(self) -> None:
        native_probe.ADMITTED_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=native_probe.ADMITTED_ROOT) as directory:
            receipt = Path(directory) / "admission.json"
            native_probe.atomic_json_exclusive(receipt, {"status": "admitted"})
            self.assertEqual(json.loads(receipt.read_text()), {"status": "admitted"})
            with self.assertRaises(native_probe.ProbeAdmissionError):
                native_probe.atomic_json_exclusive(receipt, {"status": "replacement"})

    def test_owner_snapshot_failure_is_fail_closed(self) -> None:
        with mock.patch.object(native_probe.subprocess, "run", side_effect=OSError("ps failed")):
            with self.assertRaises(native_probe.ProbeAdmissionError):
                native_probe.active_heavy_owners_fail_closed()

    def test_campaign_wide_owner_inventory_detects_mop_and_corpus(self) -> None:
        snapshot = "\n".join(
            (
                "4796 /Users/test/mop/.venv/bin/python /Users/test/mop/scripts/mop_generation1_campaign.py run --execute",
                "4797 /usr/bin/caffeinate -ims /Users/test/mop/scripts/mop_generation1_campaign.py run --execute",
                "81377 .venv/bin/python scripts/generation1_competence_atlas/generation1_cognitive_corpus.py --config test.json",
                "90001 /tmp/appendix_device_runner.py --run-raw artifact",
                "90002 /tmp/probe-metal-rht --dispatch",
                "90003 /tmp/quantize-model-native --in fixture",
            )
        )
        completed = native_probe.subprocess.CompletedProcess(
            args=["ps"], returncode=0, stdout=snapshot, stderr=""
        )
        with mock.patch.object(native_probe.subprocess, "run", return_value=completed):
            owners = native_probe.active_heavy_owners_fail_closed()
        self.assertEqual(
            {4796, 4797, 81377, 90001, 90002, 90003},
            {row["pid"] for row in owners},
        )

    def test_lease_must_be_held_before_verification(self) -> None:
        native_probe.ADMITTED_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=native_probe.ADMITTED_ROOT) as directory:
            path = Path(directory) / "lease.lock"
            with path.open("a+") as lease:
                self.assertFalse(native_probe.lease_is_already_owned(lease.fileno(), path))
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertTrue(native_probe.lease_is_already_owned(lease.fileno(), path))
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    unittest.main()
