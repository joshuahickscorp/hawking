#!/usr/bin/env python3.12
"""Focused regression tests for the generalized STRAND ladder runtime."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = ROOT / "tools/condense/doctor_v5_strand_ladder_worker.py"
TREATMENT_PATH = ROOT / "tools/condense/doctor_v5_qwen_treatment_adapter.py"


def _load_worker():
    spec = importlib.util.spec_from_file_location("strand_ladder_worker_test", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


W = _load_worker()


def _load_treatment():
    spec = importlib.util.spec_from_file_location("qwen_treatment_adapter_test", TREATMENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = _load_treatment()


class LadderRuntimeTests(unittest.TestCase):
    def require_release_binaries(self) -> None:
        required = (
            ROOT / "vendor/strand-quant/target/release/quantize-model",
            ROOT / "vendor/strand-decode-kernel/target/release/attest-strand",
            ROOT / "vendor/strand-decode-kernel/target/release/archive-to-safetensors",
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            self.skipTest(
                "detached worktree has no release binaries: "
                + ", ".join(path.name for path in missing)
            )

    def test_pre_admission_gc_v2_receipt_binds_successor_program(self) -> None:
        scratch = ROOT / "scratch"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as raw:
            root = Path(raw)
            successor_path = root / "successor.json"
            reporter_path = root / "reporter.json"
            receipt_path = root / "packed_gc_receipt.json"
            successor_spec = {
                "label": "0.5B", "program_spec_sha256": "a" * 64,
                "campaign_binding": {
                    "cell_id": "static-cell",
                    "cell_identity_sha256": "b" * 64,
                    "branch": "doctor_static", "target_rate_id": "4",
                    "label": "0.5B",
                },
            }
            successor_path.write_text(json.dumps(successor_spec, sort_keys=True))
            reporter_path.write_text(json.dumps({"sealed": True}, sort_keys=True))
            successor_sha, successor_bytes = T._hash_file(successor_path)
            reporter_sha, reporter_bytes = T._hash_file(reporter_path)
            packed_rows = [{
                "role": "bundle_shard:00000.strand",
                "path": str(root / "already_deleted.strand"),
                "sha256": "c" * 64, "bytes": 123,
            }]
            receipt = {
                "schema": "hawking.doctor_v5_packed_gc_receipt.v2",
                "cell_id": "codec-cell", "cell_identity_sha256": "d" * 64,
                "result_sha256": "e" * 64,
                "successor": {
                    "cell_id": "static-cell", "cell_identity_sha256": "b" * 64,
                    "runtime_spec_path": str(successor_path),
                    "runtime_spec_sha256": successor_sha,
                    "runtime_spec_bytes": successor_bytes,
                    "program_spec_sha256": "a" * 64,
                },
                "reporter_sync": {"path": str(reporter_path),
                                  "sha256": reporter_sha, "bytes": reporter_bytes},
                "deleted_artifacts": packed_rows,
                "retained_evidence_roles": [
                    "bundle_manifest", "worker_request", "worker_checkpoint",
                    "worker_receipt", "outer_result", "outer_execution_receipt",
                ],
                "parent_source_deleted": False,
                "completed_at": "2026-07-13T00:00:00+00:00",
            }
            receipt["receipt_sha256"] = T._sha_value(receipt)
            receipt_path.write_text(json.dumps(receipt, sort_keys=True))
            validated = T._validate_gc_receipt(
                receipt_path,
                binding={"branch": "codec_control", "cell_id": "codec-cell",
                         "cell_identity_sha256": "d" * 64},
                result={"result_sha256": "e" * 64}, packed_rows=packed_rows,
                consumer_spec={"label": "0.5B", "campaign_binding": {
                    "target_rate_id": "4"
                }},
            )
            self.assertEqual("static-cell", validated["successor"]["cell_id"])
            successor_spec["program_spec_sha256"] = "f" * 64
            successor_path.write_text(json.dumps(successor_spec, sort_keys=True))
            with self.assertRaises(T.TreatmentError):
                T._validate_gc_receipt(
                    receipt_path,
                    binding={"branch": "codec_control", "cell_id": "codec-cell",
                             "cell_identity_sha256": "d" * 64},
                    result={"result_sha256": "e" * 64}, packed_rows=packed_rows,
                    consumer_spec={"label": "0.5B", "campaign_binding": {
                        "target_rate_id": "4"
                    }},
                )

    def test_resume_rehashes_every_completed_shard_artifact(self) -> None:
        scratch = ROOT / "scratch"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as raw:
            root = Path(raw)
            paths = W._paths(root, 1)
            paths["bundle"].mkdir(parents=True)
            (paths["bundle"] / "shards").mkdir(parents=True)
            paths["logs"].mkdir(parents=True)
            paths["evaluation"].mkdir(parents=True)
            artifact = paths["shards"][0]["passthrough"]
            artifact.write_bytes(b"sealed passthrough")
            plan = ["preflight", "metadata", "passthrough:00000"]
            cp = W._initial_checkpoint("a" * 64, plan)
            cp["completed_units"] = list(plan)
            cp["units"] = {
                "preflight": {"completed_at": W._now()},
                "metadata": {"completed_at": W._now(), "artifacts": []},
                "passthrough:00000": {"completed_at": W._now(),
                                      "artifact": W._artifact(artifact)},
            }
            W.BASE._atomic_json(paths["checkpoint"], cp)
            stats = [{"quantized_tensors": 0}]
            W._checkpoint(paths["checkpoint"], "a" * 64, plan, paths, stats)
            artifact.unlink()
            with self.assertRaises(W.LadderError):
                W._checkpoint(paths["checkpoint"], "a" * 64, plan, paths, stats)

    def test_bundle_manifest_refuses_missing_expected_packed_shard(self) -> None:
        scratch = ROOT / "scratch"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as raw:
            root = Path(raw)
            paths = W._paths(root, 1)
            paths["bundle"].mkdir(parents=True)
            (paths["bundle"] / "shards").mkdir(parents=True)
            paths["shards"][0]["passthrough"].write_bytes(b"pass")
            cp = {"units": {"preflight": {"shard_stats": [
                {"quantized_tensors": 1}
            ]}}}
            request = {
                "codec": {"rate_id": "4"},
                "campaign_binding": {}, "label": "tiny",
                "model_family": W.MODEL_FAMILY, "doctor_hook": {},
                "source": {"source_manifest_sha256": "b" * 64},
            }
            totals = {"stored_parameters": 8, "quantized_parameters": 8,
                      "passthrough_parameters": 0}
            with self.assertRaises(W.LadderError):
                W._bundle_manifest(request, paths, cp, totals)

    def test_shared_baseline_cache_receipt_is_content_bound(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            result = root / "baseline.json"
            log = root / "baseline.log"
            result.write_text(json.dumps({
                "mode": "ppl", "model": "/model", "override_manifest": None,
                "label": "3B-shared-baseline",
            }))
            log.write_text("completed\n")
            identity = {"mode": "ppl", "model_dir": "/model", "label": "3B"}
            receipt = {
                "schema": W.BASELINE_CACHE_SCHEMA,
                "cache_key_sha256": W._sha_value(identity), "identity": identity,
                "result": W._artifact(result), "log": W._artifact(log),
                "created_at": W._now(),
            }
            receipt["receipt_sha256"] = W._sha_value(receipt)
            W._validate_baseline_cache(receipt, identity, result, log)
            log.write_text("tampered\n")
            with self.assertRaises(W.LadderError):
                W._validate_baseline_cache(receipt, identity, result, log)

    def test_all_canonical_rates_have_packed_candidates(self) -> None:
        caps = W.capabilities()
        self.assertEqual(("0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"),
                         W.SUPPORTED_LABELS)
        self.assertIn("3B", W.RESIDENT_LABELS)
        self.assertEqual(10, len(caps["rates"]))
        self.assertEqual(set(W.CANONICAL_RATES), {row["rate_id"] for row in caps["rates"]})
        for row in caps["rates"]:
            self.assertTrue(row["packed_output_supported"])
            self.assertTrue(row["candidate_within_payload_ceiling"])
            candidate = row["nominal_payload_fraction"]
            target = row["target_fraction"]
            self.assertLessEqual(candidate[0] * target[1], target[0] * candidate[1])
            self.assertEqual("candidate_only_until_measured",
                             row["all_in_model_target_supported"])
        ultra = W._rate_geometry("0.1")
        self.assertFalse(ultra["adaptive_scales"])
        codec = {
            "rate_id": "0.1", "artifact_mode": ultra["artifact_mode"],
            "symbol_bits": ultra["symbol_bits"], "vector_dim": ultra["vector_dim"],
            "block_len": ultra["block_len"], "tensor_scope": "all-2d",
            "quality": True, "rht_cols": True, "outlier_channel_pct": 0,
            "outlier_bits": 8, "sdsq_sideinfo": True, "c2f_outl": False,
            "ragged_v2": True, "allow_over_ceiling_control": True,
            "learned_codebook": False, "adaptive_scales": False,
        }
        argv = W._quantizer_argv({"codec": codec, "execution": {
            "quantizer_path": "quantize-model", "threads": 1}}, Path("in"), Path("out"))
        self.assertIn("--no-adaptive-scales", argv)

    def test_streamed_passthrough_excludes_every_2d_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source, output = root / "source.safetensors", root / "pass.safetensors"
            save_file({
                "model.embed_tokens.weight": torch.randn(32, 16, dtype=torch.bfloat16),
                "model.layers.0.self_attn.q_proj.weight": torch.randn(
                    16, 16, dtype=torch.bfloat16),
                "model.layers.0.input_layernorm.weight": torch.ones(16, dtype=torch.bfloat16),
            }, str(source))
            evidence = W._stream_passthrough(source, output, "all-2d")
            self.assertEqual(1, evidence["tensor_count"])
            self.assertEqual(16, evidence["parameter_count"])
            with safe_open(str(output), framework="pt") as handle:
                self.assertEqual(["model.layers.0.input_layernorm.weight"], list(handle.keys()))
                self.assertEqual(torch.bfloat16, handle.get_tensor(list(handle.keys())[0]).dtype)

    def test_vector_packed_all2d_attest_and_bf16_decode(self) -> None:
        self.require_release_binaries()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.safetensors"
            archive = root / "model.strand"
            decoded = root / "decoded.safetensors"
            save_file({
                "model.embed_tokens.weight": torch.randn(32, 32, dtype=torch.bfloat16),
                "model.layers.0.self_attn.q_proj.weight": torch.randn(
                    32, 32, dtype=torch.bfloat16),
                "lm_head.weight": torch.randn(32, 32, dtype=torch.bfloat16),
                "model.norm.weight": torch.ones(32, dtype=torch.bfloat16),
            }, str(source))
            geometry = W._rate_geometry("0.33")
            codec = {
                "rate_id": "0.33", "artifact_mode": geometry["artifact_mode"],
                "symbol_bits": geometry["symbol_bits"], "vector_dim": geometry["vector_dim"],
                "block_len": geometry["block_len"], "tensor_scope": "all-2d",
                "quality": True, "rht_cols": True, "outlier_channel_pct": 0,
                "outlier_bits": 8, "sdsq_sideinfo": True, "c2f_outl": False,
                "ragged_v2": True, "allow_over_ceiling_control": True,
                "learned_codebook": False, "adaptive_scales": True,
            }
            request = {"codec": codec, "execution": {
                "quantizer_path": str(ROOT / "vendor/strand-quant/target/release/quantize-model"),
                "threads": 1,
            }}
            subprocess.run(W._quantizer_argv(request, source, archive), check=True,
                           capture_output=True, text=True, env=W.BASE._fixed_env())
            attest = subprocess.run([
                str(ROOT / "vendor/strand-decode-kernel/target/release/attest-strand"),
                str(archive), "--roots",
            ], check=True, capture_output=True, text=True)
            self.assertIn("self-verify", attest.stdout)
            self.assertIn("model_root", attest.stdout)
            subprocess.run([
                str(ROOT / "vendor/strand-decode-kernel/target/release/archive-to-safetensors"),
                str(archive), str(decoded), "--dtype", "bf16",
            ], check=True, capture_output=True, text=True)
            validation = W._validate_decoded(source, decoded, scope="all-2d")
            self.assertEqual(3, validation["tensor_count"])
            with safe_open(str(decoded), framework="pt") as handle:
                self.assertEqual(torch.bfloat16,
                                 handle.get_tensor("model.embed_tokens.weight").dtype)
                self.assertNotIn("model.norm.weight", list(handle.keys()))
            # A tiny archive is framing-dominated.  This guards against ever
            # equating the candidate's 1/8 symbol payload with physical bpw.
            physical_bpw = archive.stat().st_size * 8 / (3 * 32 * 32 + 32)
            self.assertGreater(physical_bpw, float(W.CANONICAL_RATES["0.33"]))

    def test_full_vector_treatment_is_packed_and_decodable(self) -> None:
        self.require_release_binaries()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.safetensors"
            archive = root / "full.strand"
            decoded = root / "decoded.safetensors"
            save_file({
                "model.layers.0.self_attn.q_proj.weight": torch.randn(
                    32, 32, dtype=torch.bfloat16),
            }, str(source))
            geometry = W._rate_geometry("0.5")
            recipe = W.treatment_recipe("doctor_full", "0.5")
            self.assertTrue(recipe["learned_codebook"])
            self.assertGreater(recipe["outlier_channel_pct"], 0)
            codec = {
                "rate_id": "0.5", "artifact_mode": geometry["artifact_mode"],
                "symbol_bits": geometry["symbol_bits"],
                "vector_dim": geometry["vector_dim"],
                "block_len": geometry["block_len"], "tensor_scope": "all-2d",
                "quality": True, "rht_cols": True,
                "outlier_channel_pct": recipe["outlier_channel_pct"],
                "outlier_bits": recipe["outlier_bits"], "sdsq_sideinfo": True,
                "c2f_outl": recipe["c2f_outl"], "ragged_v2": True,
                "allow_over_ceiling_control": True,
                "learned_codebook": recipe["learned_codebook"],
                "adaptive_scales": geometry["adaptive_scales"],
            }
            request = {"codec": codec, "execution": {
                "quantizer_path": str(
                    ROOT / "vendor/strand-quant/target/release/quantize-model"),
                "threads": 1,
            }}
            subprocess.run(W._quantizer_argv(request, source, archive), check=True,
                           capture_output=True, text=True, env=W.BASE._fixed_env())
            subprocess.run([
                str(ROOT / "vendor/strand-decode-kernel/target/release/attest-strand"),
                str(archive), "--roots",
            ], check=True, capture_output=True, text=True)
            subprocess.run([
                str(ROOT / "vendor/strand-decode-kernel/target/release/archive-to-safetensors"),
                str(archive), str(decoded), "--dtype", "bf16",
            ], check=True, capture_output=True, text=True)
            validation = W._validate_decoded(source, decoded, scope="all-2d")
            self.assertEqual(1, validation["tensor_count"])


if __name__ == "__main__":
    unittest.main()
