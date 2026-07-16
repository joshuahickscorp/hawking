from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
CONDENSE = HERE.parent
sys.path.insert(0, str(CONDENSE))

import doctor_v5_gptoss_parallel_scaffold as parallel
import doctor_v5_higher_tier_scaffold as higher
import doctor_v5_streaming_source as streaming


class StreamingSourceTests(unittest.TestCase):
    def test_bounded_range_and_full_hash_bind_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.bin"
            payload = bytes(range(251)) * 1000
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            info = path.stat()
            with streaming.ImmutableSourceReader(
                path, expected_bytes=len(payload), expected_sha256=digest,
                expected_device=info.st_dev, expected_inode=info.st_ino,
                expected_mtime_ns=info.st_mtime_ns,
            ) as reader:
                receipt = reader.hash_range(17, 10_001, chunk_bytes=997)
                self.assertEqual(
                    hashlib.sha256(payload[17:10_018]).hexdigest(),
                    receipt["range_sha256"],
                )
                self.assertFalse(receipt["whole_file_materialized"])
                self.assertLessEqual(receipt["maximum_buffer_bytes"], 997)
                full = reader.hash_all(chunk_bytes=4096)
                self.assertTrue(full["content_authority_verified"])

    def test_reader_rejects_wrong_authority_and_unsafe_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.bin"
            path.write_bytes(b"abcdef")
            with self.assertRaises(streaming.StreamingSourceError):
                streaming.ImmutableSourceReader(
                    path, expected_bytes=5, expected_sha256="0" * 64
                )
            with streaming.ImmutableSourceReader(
                path, expected_bytes=6,
                expected_sha256=hashlib.sha256(b"abcdef").hexdigest(),
            ) as reader:
                with self.assertRaises(streaming.StreamingSourceError):
                    reader.read_exact(5, 2)


class GptOssParallelPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = parallel.build_work_plan(created_at="2026-07-14T00:00:00+00:00")

    def test_exact_parallel_work_graph_is_fail_closed(self) -> None:
        self.assertEqual([], parallel.validate_work_plan(self.plan))
        self.assertEqual(615, len(self.plan["source_units"]))
        self.assertEqual(6_150, len(self.plan["output_units"]))
        counts: dict[str, int] = {}
        for unit in self.plan["source_units"]:
            counts[unit["kind"]] = counts.get(unit["kind"], 0) + 1
        self.assertEqual({
            "expert_batch": 576, "dense_layer": 36,
            "embedding": 1, "output_head": 1, "lossless_sidecar": 1,
        }, counts)
        self.assertEqual(
            116_829_156_672,
            sum(row["logical_parameters"] for row in self.plan["source_units"]),
        )
        self.assertFalse(self.plan["execution_gate"]["executable"])

    def test_tampered_source_binding_and_output_coverage_are_rejected(self) -> None:
        damaged = copy.deepcopy(self.plan)
        damaged["source_units"][0]["source_extents"][0]["bytes"] += 1
        damaged["work_plan_sha256"] = parallel._hash_value(
            parallel._without(damaged, "work_plan_sha256")
        )
        errors = parallel.validate_work_plan(damaged)
        self.assertTrue(any("source binding hash" in error for error in errors))
        damaged = copy.deepcopy(self.plan)
        damaged["output_units"].pop()
        damaged["work_plan_sha256"] = parallel._hash_value(
            parallel._without(damaged, "work_plan_sha256")
        )
        self.assertTrue(any("coverage" in error for error in
                            parallel.validate_work_plan(damaged)))

    def test_one_output_receipt_and_structural_moe_lookup(self) -> None:
        output = next(row for row in self.plan["output_units"]
                      if row["attestation_required"])
        source = next(row for row in self.plan["source_units"]
                      if row["unit_id"] == output["source_unit_id"])
        traversal = parallel.build_source_traversal_receipt(
            self.plan, source_unit_id=source["unit_id"],
            staging_artifact={"path": "/ephemeral/stage", "sha256": "1" * 64,
                              "bytes": 100},
            range_sha256=["4" * 64 for _extent in source["source_extents"]],
        )
        self.assertEqual([], parallel.validate_source_traversal_receipt(
            self.plan, traversal
        ))
        receipt = parallel.build_output_receipt(
            self.plan, output_unit_id=output["output_unit_id"],
            source_traversal_receipt=traversal,
            archive={"path": "/ephemeral/archive", "sha256": "2" * 64,
                     "bytes": 50},
            attestation={"root_sha256": "3" * 64},
        )
        self.assertEqual([], parallel.validate_output_receipt(self.plan, receipt))
        with self.assertRaises(parallel.ParallelScaffoldError):
            parallel.build_merge_manifest(self.plan, [receipt])
        route = parallel.MoeArchiveIndex(self.plan).resolve_expert(
            rate_id="0.25", layer=7, expert=119
        )
        self.assertEqual(
            "expert/layer=007/experts=112-119/rate=0.25",
            route["output_unit_id"],
        )
        self.assertEqual("pending-output-receipt", route["status"])

    def test_pending_wiring_is_exact_10_by_4_and_never_live(self) -> None:
        packet = parallel.build_pending_wiring_packet(self.plan)
        self.assertEqual([], parallel.validate_pending_wiring(packet))
        self.assertEqual(40, len(packet["cell_bindings"]))
        self.assertFalse(packet["live_registry_mutated"])
        self.assertFalse(packet["live_runtime_specs_written"])
        self.assertFalse(packet["promotion_gate"]["currently_permitted"])

    def test_tokenizer_binding_is_hash_bound_but_still_unreviewed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            path.write_text('{"version":"synthetic"}', encoding="utf-8")
            binding = parallel.build_tokenizer_binding(
                model_source_manifest_sha256="4" * 64, files=[path],
                chat_template_sha256="5" * 64,
            )
            self.assertEqual([], parallel.validate_tokenizer_binding(
                binding, verify_files=True
            ))
            self.assertFalse(binding["quality_evaluation_permitted"])


class HigherTierScaffoldTests(unittest.TestCase):
    def _manifest(self) -> dict[str, object]:
        doc: dict[str, object] = {
            "schema": higher.SOURCE_MANIFEST_SCHEMA,
            "created_at": "2026-07-14T00:00:00+00:00", "status": "sealed",
            "model": {
                "label": "synthetic-200b", "hf_id_or_source_id": "test/synthetic",
                "family": "synthetic-moe", "architecture_kind": "moe",
                "logical_parameters": 200_000_000_000,
                "parameter_authority_sha256": "a" * 64,
            },
            "sources": [
                {"source_id": "shard-0", "transport": "object_range",
                 "uri": "object://immutable/shard-0", "bytes": 1000,
                 "sha256": "b" * 64, "immutable_content": True,
                 "range_reads_supported": True},
                {"source_id": "shard-1", "transport": "https_range",
                 "uri": "https://example.invalid/shard-1", "bytes": 2000,
                 "sha256": "c" * 64, "immutable_content": True,
                 "range_reads_supported": True},
            ],
            "work_units": [
                {"unit_id": "expert-0", "kind": "expert_batch",
                 "logical_parameters": 150_000_000_000,
                 "estimated_peak_resident_bytes": 8_000_000_000,
                 "threads_per_lane": 8,
                 "source_ranges": [
                     {"source_id": "shard-0", "absolute_byte_range": [0, 1000],
                      "range_role": "packed-experts", "range_sha256": "e" * 64,
                      "tensor_keys": ["layer.0.experts"]}
                 ]},
                {"unit_id": "dense-0", "kind": "dense_tensor_batch",
                 "logical_parameters": 50_000_000_000,
                 "estimated_peak_resident_bytes": 6_000_000_000,
                 "threads_per_lane": 6,
                 "source_ranges": [
                     {"source_id": "shard-1", "absolute_byte_range": [0, 2000],
                      "range_role": "dense-weights", "range_sha256": "f" * 64,
                      "tensor_keys": ["layer.0.dense"]}
                 ]},
            ],
            "coverage": {"work_unit_count": 2,
                         "logical_parameters": 200_000_000_000,
                         "all_model_tensors_assigned_exactly_once": True,
                         "tensor_count": 2,
                         "tensor_layout_sha256": "d" * 64},
            "tokenizer_binding": None, "source_deletion_permitted": False,
        }
        doc["manifest_sha256"] = higher._hash_value(doc)
        return doc

    def test_explicit_manifest_builds_dynamic_but_blocked_admission(self) -> None:
        manifest = self._manifest()
        self.assertEqual([], higher.validate_source_manifest(manifest))
        plan = higher.build_admission_plan(
            manifest, total_memory_bytes=96_000_000_000,
            process_budget_bytes=78_000_000_000,
            control_resident_bytes=8_000_000_000,
            safety_margin_bytes=14_000_000_000, logical_cpu_count=28,
            maximum_lanes=8,
        )
        self.assertEqual([], higher.validate_admission_plan(plan, manifest))
        self.assertEqual(20, len(plan["output_units"]))
        self.assertEqual(3, plan["estimated_caps"]["proposed_lane_cap"])
        self.assertFalse(plan["execution_gate"]["executable"])
        aggressive = plan["execution_gate"]["aggressive_admission"]
        self.assertFalse(aggressive["qualified_overlay_artifact_bound"])
        self.assertFalse(aggressive["sealed_swap_baseline_bound"])
        self.assertFalse(aggressive["controller_requirement"]["baseline_can_ratchet"])
        self.assertEqual(
            higher.aggressive_admission.swap_policy(),
            aggressive["controller_requirement"]["relative_growth_and_rate_policy"],
        )

    def test_missing_exact_source_coverage_fails_closed(self) -> None:
        damaged = self._manifest()
        damaged["work_units"][1]["source_ranges"] = []
        damaged["manifest_sha256"] = higher._hash_value(
            higher._without(damaged, "manifest_sha256")
        )
        errors = higher.validate_source_manifest(damaged)
        self.assertTrue(any("no source ranges" in error for error in errors))

    def test_requirements_packet_makes_no_model_claim(self) -> None:
        requirements = higher.build_requirements_packet()
        self.assertEqual("generic-scaffold-only-no-model-admitted",
                         requirements["status"])
        self.assertFalse(requirements["execution_permitted"])
        self.assertFalse(requirements["unsupported_or_negative_outcomes_synthesized"])
        admission = requirements["admission"]
        self.assertNotIn("zero_swap_required", admission)
        controller = admission["aggressive_swap_controller"]
        self.assertFalse(controller["baseline_can_ratchet"])
        self.assertTrue(
            controller["sealed_baseline_swap_mb_required_at_quiescent_promotion"]
        )
        self.assertEqual(higher.aggressive_admission.swap_policy(),
                         controller["relative_growth_and_rate_policy"])

    def test_higher_tier_controller_contract_tamper_fails_closed(self) -> None:
        manifest = self._manifest()
        plan = higher.build_admission_plan(
            manifest, total_memory_bytes=96_000_000_000,
            process_budget_bytes=78_000_000_000,
            control_resident_bytes=8_000_000_000,
            safety_margin_bytes=14_000_000_000, logical_cpu_count=28,
            maximum_lanes=8,
        )
        damaged = copy.deepcopy(plan)
        damaged["execution_gate"]["aggressive_admission"][
            "controller_requirement"
        ]["baseline_can_ratchet"] = True
        damaged["admission_plan_sha256"] = higher._hash_value(
            higher._without(damaged, "admission_plan_sha256")
        )
        errors = higher.validate_admission_plan(damaged, manifest)
        self.assertTrue(any("aggressive admission" in row for row in errors))

    def test_higher_tier_admission_resealed_semantic_forgery_is_rejected(self) -> None:
        manifest = self._manifest()
        plan = higher.build_admission_plan(
            manifest, total_memory_bytes=96_000_000_000,
            process_budget_bytes=78_000_000_000,
            control_resident_bytes=8_000_000_000,
            safety_margin_bytes=14_000_000_000, logical_cpu_count=28,
            maximum_lanes=8,
        )
        damaged = copy.deepcopy(plan)
        damaged["model"]["label"] = "substituted"
        damaged["estimated_caps"]["proposed_lane_cap"] = 8
        damaged["waves"][0]["estimated_peak_resident_bytes"] = 1
        damaged["output_units"][0]["source_manifest_sha256"] = "f" * 64
        damaged["execution_gate"]["thermal_guard_armed"] = True
        damaged["admission_plan_sha256"] = higher._hash_value(
            higher._without(damaged, "admission_plan_sha256")
        )
        errors = higher.validate_admission_plan(damaged, manifest)
        self.assertTrue(any("deterministic" in row for row in errors), errors)


if __name__ == "__main__":
    unittest.main()
