from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import doctor_v5_qwen_thread_profile_runner as runner


class QwenThreadProfileRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output_root = self.root / "profile_qualification"
        self.context = self._context()
        self.lease = self._lease()
        self.canonical = self._canonical()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _file(self, name: str, raw: bytes) -> dict:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return runner._artifact(path)

    def _context(self) -> dict:
        binary = self._file("bin/quantize-model-block-parallel", b"binary-v1")
        serial_binary = self._file("bin/quantize-model-serial", b"serial-binary-v1")
        runtime = self._file("runtime/spec.json", b"{}\n")
        seal = self._file("seals/14B.json", b"{}\n")
        codec = {
            "adaptive_scales": True, "allow_over_ceiling_control": True,
            "artifact_mode": "packed_scalar_control", "block_len": 256,
            "c2f_outl": True, "learned_codebook": False, "outlier_bits": 8,
            "outlier_channel_pct": 1, "quality": True, "ragged_v2": True,
            "rate_id": "3", "rht_cols": True, "sdsq_sideinfo": True,
            "symbol_bits": 3, "tensor_scope": "all-2d", "vector_dim": 1,
        }
        source = {
            "runtime_spec": runtime, "program_spec_sha256": "1" * 64,
            "resource_admission_sha256": "2" * 64,
            "cell_id": "qwen2-5-14b__3bpw__codec-control",
            "tier": "14B", "rate": "3", "branch": "codec_control",
            "codec": codec, "codec_sha256": runner._hash_value(codec),
            "source_seal": seal, "source_seal_schema": "hawking.doctor_v5_source_seal.v2",
            "source_shard": {"path": str(self.root / "source.safetensors"),
                             "sha256": "3" * 64, "bytes": 100,
                             "identity": {"volume_uuid": "A", "st_ino": 1,
                                          "st_size": 100, "st_mtime_ns": 2,
                                          "st_ctime_ns": 3, "st_dev_at_seal": 4}},
            "tensor": {"name": "model.layers.0.self_attn.q_proj.weight",
                       "shape": [2048, 2048], "elements": 4194304,
                       "dtype": "F16"},
            "selection": "smallest-real-qwen-projection>=4M-elements;prefer<=16M;name-tiebreak",
        }
        return {
            "path": Path(runtime["path"]), "document": {}, "tier": "14B", "rate": "3",
            "branch": "codec_control", "cell_id": source["cell_id"],
            "runtime_spec": runtime, "codec": codec, "binary": binary,
            "binary_path": Path(binary["path"]), "source_seal": seal,
            "serial_binary": serial_binary,
            "serial_binary_path": Path(serial_binary["path"]),
            "source_shard": source["source_shard"], "tensor": source["tensor"],
            "source_binding": source, "source_binding_sha256": runner._hash_value(source),
        }

    def _lease(self) -> dict:
        artifact = self._file("locks/studio_heavy.lock", b"")
        info = Path(artifact["path"]).stat()
        lease = {
            "path": artifact["path"], "acquired_at": "2026-07-15T00:00:00+00:00",
            "holder_pid": 123, "device": info.st_dev, "inode": info.st_ino,
            "exclusive_nonblocking_flock": True, "inherited_by_child": False,
            "parent_retained_for_entire_child_lifetime": True,
        }
        lease["lease_evidence_sha256"] = runner._hash_value(lease)
        return lease

    def _resource(self) -> dict:
        return {
            "sampled_at": "2026-07-15T00:00:00+00:00", "pressure_level": 1,
            "swap_used_mb": 0.0, "thermal_nominal": True, "ac_power": True,
            "disk_free_bytes": 100_000_000_000, "host_cpu_cores": 0.5,
            "logical_cpu_count": 32, "load_average": [0.1, 0.1, 0.1],
            "probe_sha256s": {"pressure": "a" * 64},
        }

    def _canonical(self) -> dict:
        base = runner.cell_root(self.context, output_root=self.output_root)
        output = self._file(str((base / "canonical-output.strand").relative_to(self.root)),
                            b"exact-output")
        log = self._file(str((base / "canonical.log").relative_to(self.root)), b"CPU serial\n")
        monitor = self._file(str((base / "canonical-monitor.jsonl").relative_to(self.root)),
                            b"{}\n")
        env = runner._launch_env(base / "tmp")
        argv = runner.build_argv(self.context, Path(output["path"]), threads=None)
        execution = {
            "started_at": "2026-07-15T00:00:00+00:00",
            "completed_at": "2026-07-15T00:00:01+00:00", "wall_seconds": 1.0,
            "peak_rss_bytes": 1024, "rss_sample_count": 4, "returncode": 0,
            "guard_trip": None, "argv": argv, "argv_sha256": runner._hash_value(argv),
            "environment": env, "environment_sha256": runner._hash_value(env),
            "log": log, "monitor": monitor, "output": output,
        }
        document = {
            "schema": runner.CANONICAL_SCHEMA, "version": runner.VERSION,
            "status": "pass", "created_at": "2026-07-15T00:00:02+00:00",
            "scope": "production", "synthetic": False, "tier": "14B", "rate": "3",
            "cell_id": self.context["cell_id"], "branch": "codec_control",
            "runtime_spec": self.context["runtime_spec"],
            "source_seal": self.context["source_seal"],
            "binary": self.context["serial_binary"],
            "source_binding": self.context["source_binding"],
            "source_binding_sha256": self.context["source_binding_sha256"],
            "execution": execution,
            "resource_evidence": {"before": self._resource(), "after": self._resource()},
            "lease_evidence": self.lease, "cpu_only": True,
            "real_production_tensor": True, "runtime_defaults_mutated": False,
            "live_queue_mutated": False, "source_deletion_permitted": False,
        }
        document["receipt_sha256"] = runner._hash_value(document)
        path = runner.canonical_path(self.context, output_root=self.output_root)
        runner._write_exclusive_json(path, document)
        self.assertFalse(runner.validate_canonical(document, self.context, verify_files=True))
        return document

    def _receipt(self, threads: int, *, wall: float = 1.0) -> dict:
        base = runner.cell_root(self.context, output_root=self.output_root)
        output = self._file(
            str((base / f"candidate-{threads}-output.strand").relative_to(self.root)),
            b"exact-output",
        )
        phrase = (f"feature-gated block-parallel CPU encode: {threads} block workers, "
                  "256 MiB aggregate Viterbi scratch cap; forcing one outer tensor worker\n")
        log = self._file(str((base / f"candidate-{threads}.log").relative_to(self.root)),
                         phrase.encode())
        monitor = self._file(
            str((base / f"candidate-{threads}-monitor.jsonl").relative_to(self.root)), b"{}\n")
        env = runner._launch_env(base / f"tmp-{threads}")
        argv = runner.build_argv(self.context, Path(output["path"]), threads=threads)
        execution = {
            "started_at": "2026-07-15T00:00:00+00:00",
            "completed_at": "2026-07-15T00:00:01+00:00", "wall_seconds": wall,
            "peak_rss_bytes": 1_000_000 + threads, "rss_sample_count": 4,
            "returncode": 0, "guard_trip": None, "argv": argv,
            "argv_sha256": runner._hash_value(argv), "environment": env,
            "environment_sha256": runner._hash_value(env), "log": log,
            "monitor": monitor, "output": output,
        }
        document = {
            "schema": runner.RECEIPT_SCHEMA, "version": runner.VERSION,
            "status": "pass", "scope": "production", "synthetic": False,
            "created_at": "2026-07-15T00:00:02+00:00", "tier": "14B", "rate": "3",
            "threads": threads, "binary_sha256": self.context["binary"]["sha256"],
            "source_sha256": self.context["source_binding_sha256"],
            "canonical_output_sha256": self.canonical["execution"]["output"]["sha256"],
            "output_sha256": output["sha256"], "exact_output": True,
            "wall_seconds": wall, "peak_rss_bytes": execution["peak_rss_bytes"],
            "scratch_budget_bytes": runner.BLOCK_SCRATCH_BUDGET_BYTES,
            "mode": "block_parallel", "cell_id": self.context["cell_id"],
            "branch": "codec_control", "runtime_spec": self.context["runtime_spec"],
            "source_seal": self.context["source_seal"], "binary": self.context["binary"],
            "source_binding": self.context["source_binding"],
            "source_binding_sha256": self.context["source_binding_sha256"],
            "canonical": {
                "receipt": runner._artifact(runner.canonical_path(
                    self.context, output_root=self.output_root)),
                "output": self.canonical["execution"]["output"],
            },
            "candidate": execution,
            "resource_evidence": {"before": self._resource(), "after": self._resource()},
            "lease_evidence": self.lease, "cpu_only": True,
            "real_production_tensor": True, "runtime_defaults_mutated": False,
            "live_queue_mutated": False, "source_deletion_permitted": False,
        }
        document["receipt_sha256"] = runner._hash_value(document)
        return document

    def test_matrix_is_deterministic_exact_and_tamper_evident(self) -> None:
        # The fixture spec is not semantically Qwen, so patch only the identity reader.
        identity = {"path": Path(self.context["runtime_spec"]["path"]), "document": {},
                    "tier": "14B", "rate": "3", "branch": "codec_control",
                    "cell_id": self.context["cell_id"]}
        with mock.patch.object(runner, "_spec_identity", return_value=identity):
            first = runner.build_matrix_manifest([identity["path"]])
            second = runner.build_matrix_manifest([identity["path"]])
        self.assertEqual(first, second)
        self.assertFalse(runner.validate_matrix(first, verify_files=True))
        first["required_threads"] = [8, 12, 16, 24]
        self.assertTrue(runner.validate_matrix(first, verify_files=False))

    def test_argv_is_exact_codec_cpu_block_path(self) -> None:
        output = self.root / "out.strand"
        argv = runner.build_argv(self.context, output, threads=16)
        self.assertEqual(argv[-4:], ["--block-threads", "16",
                                    "--block-scratch-budget-bytes", "268435456"])
        self.assertIn("--only", argv)
        self.assertIn(self.context["tensor"]["name"], argv)
        self.assertNotIn("sh", argv)
        serial = runner.build_argv(self.context, output, threads=None)
        self.assertEqual(serial[0], str(self.context["serial_binary_path"]))
        self.assertNotIn("--block-threads", serial)
        env = runner._launch_env(self.root / "tmp")
        self.assertEqual(env["STRAND_NO_GPU"], "1")
        self.assertTrue(all(env[key] == "1" for key in runner.THREAD_ENV_KEYS))

    def test_strict_receipt_accepts_real_and_rejects_adversarial_changes(self) -> None:
        receipt = self._receipt(16)
        self.assertFalse(runner.validate_receipt(receipt, self.context, threads=16,
                                                verify_files=True))
        for mutate in (
            lambda row: row.__setitem__("synthetic", True),
            lambda row: row.__setitem__("source_sha256", "f" * 64),
            lambda row: row["candidate"]["argv"].__setitem__(-3, "24"),
            lambda row: row["lease_evidence"].__setitem__("exclusive_nonblocking_flock", False),
        ):
            changed = copy.deepcopy(receipt); mutate(changed)
            changed["receipt_sha256"] = runner._hash_value(
                runner._without(changed, "receipt_sha256"))
            self.assertTrue(runner.validate_receipt(changed, self.context, threads=16,
                                                    verify_files=False))

    def test_file_tamper_is_rejected_even_with_unchanged_receipt(self) -> None:
        receipt = self._receipt(12)
        self.assertFalse(runner.validate_receipt(receipt, self.context, threads=12,
                                                verify_files=True))
        Path(receipt["candidate"]["output"]["path"]).chmod(0o600)
        Path(receipt["candidate"]["output"]["path"]).write_bytes(b"tampered")
        self.assertTrue(runner.validate_receipt(receipt, self.context, threads=12,
                                               verify_files=True))

    def test_profile_build_requires_all_four_and_is_idempotent(self) -> None:
        for threads, wall in zip(runner.THREADS, (4.0, 3.0, 2.0, 2.5)):
            document = self._receipt(threads, wall=wall)
            runner._write_exclusive_json(
                runner.receipt_path(self.context, threads, output_root=self.output_root),
                document,
            )
        qualification = runner.build_profile([self.context], output_root=self.output_root)
        again = runner.build_profile([self.context], output_root=self.output_root)
        self.assertEqual(qualification, again)
        profile = json.loads(Path(qualification["profile"]["path"]).read_text())
        entry = profile["entries"][json.dumps(["14B", "3"], separators=(",", ":"))]
        self.assertEqual(entry["selected_threads"], 16)
        self.assertFalse(qualification["automatic_runtime_promotion_permitted"])

    def test_profile_missing_candidate_fails_closed(self) -> None:
        with self.assertRaisesRegex(runner.QualificationError, "missing exact production receipt"):
            runner.build_profile([self.context], output_root=self.output_root)

    def test_status_estimate_is_unknown_until_physical_measurement(self) -> None:
        result = runner.status([self.context], output_root=self.output_root)
        self.assertIsNone(result["estimated_remaining_seconds"])
        self.assertEqual(result["arm_count_missing"], 4)  # canonical fixture is valid
        self.assertEqual(result["execution_default"], "off")

    def test_run_without_explicit_flag_refuses_before_context_or_model_read(self) -> None:
        with mock.patch.object(runner, "load_context") as load:
            with self.assertRaisesRegex(runner.QualificationError, "default-off"):
                runner.main(["run", "--runtime-spec", str(self.root / "absent.json"),
                             "--threads", "8"])
        load.assert_not_called()

    def test_invalid_existing_candidate_is_never_overwritten_or_rerun(self) -> None:
        path = runner.receipt_path(self.context, 8, output_root=self.output_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        with mock.patch.object(runner, "_run_binary") as execute:
            with self.assertRaisesRegex(runner.QualificationError, "existing candidate is invalid"):
                runner.run_candidate(self.context, self.canonical, 8, self.lease,
                                     output_root=self.output_root)
        execute.assert_not_called()
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
