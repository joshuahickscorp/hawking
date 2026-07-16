from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
CONDENSE = HERE.parent
sys.path.insert(0, str(CONDENSE))

import doctor_v5_distributed_transport as transport
import doctor_v5_gptoss_parallel_scaffold as parallel
import doctor_v5_gptoss_reuse_fanout as fanout
import doctor_v5_higher_tier_scaffold as higher


COORDINATOR_KEY = bytes(range(32))
HOST_KEY = bytes(range(32, 64))


class ReuseFanoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.work = parallel.build_work_plan(created_at="2026-07-14T00:00:00+00:00")
        cls.wiring = parallel.build_pending_wiring_packet(cls.work)
        cls.plan = fanout.build_fanout_plan(
            cls.work, cls.wiring, created_at="2026-07-14T00:00:00+00:00"
        )

    def test_exact_shared_preprocess_isolated_evidence_matrix(self) -> None:
        self.assertEqual([], fanout.validate_fanout_plan(
            self.plan, self.work, self.wiring
        ))
        self.assertEqual(24_600, len(self.plan["jobs"]))
        self.assertEqual(615, self.plan["matrix"]["source_preprocess_units"])
        namespaces = {row["evidence_namespace_sha256"] for row in self.plan["jobs"]}
        self.assertEqual(24_600, len(namespaces))
        self.assertFalse(self.plan["reuse_contract"]["scientific_evidence_may_be_shared"])

    def test_branch_receipts_share_only_traversal_and_chain_exact_dependencies(self) -> None:
        source = self.work["source_units"][0]
        traversal = parallel.build_source_traversal_receipt(
            self.work, source_unit_id=source["unit_id"],
            staging_artifact={"sha256": "1" * 64, "bytes": 100,
                              "artifact_instance_id": "shared-preprocess-1"},
            range_sha256=["2" * 64 for _row in source["source_extents"]],
        )
        codec_job = fanout._job_id(source["unit_id"], "4", "codec_control")
        codec = fanout.build_branch_receipt(
            self.plan, self.work, job_id=codec_job,
            source_traversal_receipt=traversal, dependency_receipts=[],
            output_artifact={"sha256": "3" * 64, "bytes": 50,
                             "artifact_instance_id": "codec-output-1"},
            method_evidence={"sha256": "4" * 64, "bytes": 10,
                             "artifact_instance_id": "codec-evidence-1"},
            attestation_root_sha256="5" * 64,
        )
        static_job = fanout._job_id(source["unit_id"], "4", "doctor_static")
        static = fanout.build_branch_receipt(
            self.plan, self.work, job_id=static_job,
            source_traversal_receipt=traversal,
            dependency_receipts=[{"job_id": codec_job,
                                  "receipt_sha256": codec["receipt_sha256"]}],
            output_artifact={"sha256": "6" * 64, "bytes": 50,
                             "artifact_instance_id": "static-output-1"},
            method_evidence={"sha256": "7" * 64, "bytes": 10,
                             "artifact_instance_id": "static-evidence-1"},
            attestation_root_sha256="8" * 64,
        )
        self.assertEqual([], fanout.validate_branch_receipt(self.plan, codec))
        self.assertEqual([], fanout.validate_branch_receipt(self.plan, static))
        self.assertEqual(codec["source_traversal_receipt_sha256"],
                         static["source_traversal_receipt_sha256"])
        self.assertNotEqual(codec["evidence_namespace_sha256"],
                            static["evidence_namespace_sha256"])
        with self.assertRaises(fanout.FanoutError):
            fanout.build_merge_manifest(self.plan, [codec, static])

    def test_tampered_evidence_namespace_fails_even_with_resealed_plan(self) -> None:
        damaged = copy.deepcopy(self.plan)
        damaged["jobs"][1]["evidence_namespace_sha256"] = damaged["jobs"][0][
            "evidence_namespace_sha256"
        ]
        damaged["fanout_plan_sha256"] = fanout._hash_value(
            fanout._without(damaged, "fanout_plan_sha256")
        )
        errors = fanout.validate_fanout_plan(damaged, self.work, self.wiring)
        self.assertTrue(any("aliased" in error for error in errors))


class DistributedTransportTests(unittest.TestCase):
    OBSERVED_AT = "2026-07-14T00:00:00+00:00"
    EXPIRES_AT = "2099-07-14T00:00:00+00:00"

    def _higher_manifest(self) -> dict[str, object]:
        doc: dict[str, object] = {
            "schema": higher.SOURCE_MANIFEST_SCHEMA,
            "created_at": "2026-07-14T00:00:00+00:00", "status": "sealed",
            "model": {"label": "synthetic-200b",
                      "hf_id_or_source_id": "test/synthetic",
                      "family": "synthetic", "architecture_kind": "moe",
                      "logical_parameters": 200_000_000_000,
                      "parameter_authority_sha256": "a" * 64},
            "sources": [{"source_id": "source", "transport": "object_range",
                         "uri": "object://source", "bytes": 100,
                         "sha256": "b" * 64, "immutable_content": True,
                         "range_reads_supported": True}],
            "work_units": [{"unit_id": "expert-0", "kind": "expert_batch",
                            "logical_parameters": 200_000_000_000,
                            "estimated_peak_resident_bytes": 8_000_000_000,
                            "threads_per_lane": 8,
                            "source_ranges": [{"source_id": "source",
                                               "absolute_byte_range": [0, 100],
                                               "range_role": "weights",
                                               "range_sha256": "c" * 64,
                                               "tensor_keys": ["expert.0"]}]}],
            "coverage": {"work_unit_count": 1,
                         "logical_parameters": 200_000_000_000,
                         "all_model_tensors_assigned_exactly_once": True,
                         "tensor_count": 1, "tensor_layout_sha256": "d" * 64},
            "tokenizer_binding": None, "source_deletion_permitted": False,
        }
        doc["manifest_sha256"] = higher._hash_value(doc)
        return doc

    def _parent(self) -> dict[str, object]:
        return higher.build_admission_plan(
            self._higher_manifest(), total_memory_bytes=96_000_000_000,
            process_budget_bytes=78_000_000_000,
            control_resident_bytes=8_000_000_000,
            safety_margin_bytes=14_000_000_000, logical_cpu_count=28,
        )

    def _resource(
        self, *, swap_used_bytes: int = 0, sealed_baseline_swap_bytes: int = 0,
        previous_swap_used_bytes: int | None = None,
        prior_state: dict[str, object] | None = None,
    ) -> dict[str, object]:
        authority = transport.build_swap_baseline_authority(
            host_id="second-host", instance_nonce="boot-1",
            sealed_baseline_swap_bytes=sealed_baseline_swap_bytes,
            authorized_at=self.OBSERVED_AT, expires_at=self.EXPIRES_AT,
            coordinator_key_id="coordinator-key", coordinator_key=COORDINATOR_KEY,
        )
        observed_epoch = transport._parse_time(self.OBSERVED_AT).timestamp()
        baseline_mb = authority["sealed_baseline_swap_mb"]
        if prior_state is None:
            previous = (sealed_baseline_swap_bytes if previous_swap_used_bytes is None
                        else previous_swap_used_bytes)
            state = transport.aggressive_admission.initial_swap_state(
                {"pressure_level": 1, "swap_used_mb": baseline_mb},
                now_epoch=observed_epoch - 120.0,
            )
            if previous != sealed_baseline_swap_bytes:
                state, _decision = transport.aggressive_admission.advance_swap_state(
                    state,
                    {"pressure_level": 1,
                     "swap_used_mb": previous / (1024.0 * 1024.0)},
                    now_epoch=observed_epoch - 60.0,
                    sealed_baseline_swap_mb=baseline_mb,
                )
            prior_state = state
        return transport.build_resource_attestation(
            baseline_authority=authority, prior_controller_state=prior_state,
            swap_used_bytes=swap_used_bytes, sampled_epoch=observed_epoch,
        )

    def _host(self, tools: list[dict[str, object]], *,
              resource_state: dict[str, object] | None = None) -> dict[str, object]:
        if resource_state is None:
            resource_state = self._resource()
        return transport.build_host_capability(
            host_id="second-host", instance_nonce="boot-1", architecture="arm64",
            logical_cpu_count=28, memory_bytes=96_000_000_000,
            process_budget_bytes=78_000_000_000, free_disk_bytes=300_000_000_000,
            tool_artifacts=tools,
            resource_state=resource_state,
            transport_certificate_sha256="e" * 64,
            observed_at=self.OBSERVED_AT, expires_at=self.EXPIRES_AT,
            key_id="host-key", key=HOST_KEY,
        )

    def test_chunk_resume_dedup_and_full_reassembly_verification(self) -> None:
        payload = bytes(range(251)) * 1000
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.bin"
            path.write_bytes(payload)
            manifest = transport.chunk_local_source(
                source_id="source", path=path,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_bytes=len(payload), chunk_bytes=64 * 1024,
            )
            store = transport.ContentAddressedChunkStore(Path(directory) / "chunks")
            empty = transport.build_resume_plan(manifest, store.verified_index(manifest))
            self.assertEqual(len(payload), empty["missing_bytes"])
            for row in manifest["chunks"]:
                start, end = row["absolute_byte_range"]
                store.accept(payload[start:end], expected_sha256=row["sha256"])
            first = manifest["chunks"][0]
            start, end = first["absolute_byte_range"]
            dedup = store.accept(payload[start:end], expected_sha256=first["sha256"])
            self.assertTrue(dedup["deduplicated"])
            resumed = transport.build_resume_plan(manifest, store.verified_index(manifest))
            self.assertEqual(0, resumed["missing_bytes"])
            verified = store.verify_source(manifest)
            self.assertFalse(verified["whole_source_materialized"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(),
                             verified["source"]["sha256"])

    def test_no_host_is_local_only_and_does_not_change_defaults(self) -> None:
        payload = b"transport-test"
        chunks = [{"index": 0, "absolute_byte_range": [0, len(payload)],
                   "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]
        manifest = transport.build_chunk_manifest(
            source_id="source", source_sha256=hashlib.sha256(payload).hexdigest(),
            source_bytes=len(payload), chunks=chunks,
        )
        tools = [{"role": "worker", "sha256": "f" * 64, "bytes": 123}]
        plan = transport.build_transport_plan(
            parent=self._parent(), chunk_manifests=[manifest], host_capabilities=[],
            trusted_host_keys={"host-key": HOST_KEY}, required_tool_artifacts=tools,
            nominal_link_bps=1_000_000_000, coordinator_key_id="coordinator-key",
            coordinator_key=COORDINATOR_KEY,
        )
        self.assertEqual("local-only-no-eligible-host", plan["status"])
        self.assertFalse(plan["activation"]["remote_execution_enabled"])
        self.assertFalse(plan["activation"]["runtime_defaults_changed"])
        self.assertTrue(plan["activation"]["local_only_fallback"])
        self.assertEqual([], transport.validate_transport_plan(
            plan, parent=self._parent(),
            coordinator_keys={"coordinator-key": COORDINATOR_KEY},
        ))

    def test_signed_host_lease_result_merge_and_recovery(self) -> None:
        payload = b"transport-test"
        manifest = transport.build_chunk_manifest(
            source_id="source", source_sha256=hashlib.sha256(payload).hexdigest(),
            source_bytes=len(payload),
            chunks=[{"index": 0, "absolute_byte_range": [0, len(payload)],
                     "bytes": len(payload),
                     "sha256": hashlib.sha256(payload).hexdigest()}],
        )
        tools = [{"role": "worker", "sha256": "f" * 64, "bytes": 123}]
        host = self._host(tools)
        self.assertEqual([], transport.validate_host_capability(
            host, keys={"host-key": HOST_KEY},
            baseline_authority_keys={"coordinator-key": COORDINATOR_KEY},
            now="2026-07-14T01:00:00+00:00"
        ))
        parent = self._parent()
        plan = transport.build_transport_plan(
            parent=parent, chunk_manifests=[manifest], host_capabilities=[host],
            trusted_host_keys={"host-key": HOST_KEY}, required_tool_artifacts=tools,
            nominal_link_bps=1_000_000_000, coordinator_key_id="coordinator-key",
            coordinator_key=COORDINATOR_KEY,
        )
        self.assertEqual("remote-lease-eligible", plan["status"])
        authority_sha = host["resource_state"]["aggressive_admission"][
            "baseline_authority"
        ]["baseline_authority_sha256"]
        self.assertEqual(authority_sha, plan["eligible_hosts"][0][
            "swap_baseline_authority_sha256"
        ])
        work_ids = [parent["output_units"][0]["output_unit_id"]]
        lease = transport.build_lease(
            transport_plan=plan, parent=parent, host_capability=host,
            work_ids=work_ids, attempt=1,
            issued_at="2026-07-14T01:00:00+00:00",
            expires_at="2026-07-14T02:00:00+00:00",
            coordinator_key_id="coordinator-key", coordinator_key=COORDINATOR_KEY,
        )
        self.assertEqual(authority_sha, lease["swap_baseline_authority_sha256"])
        self.assertEqual([], transport.validate_lease(
            lease, transport_plan=plan, parent=parent,
            coordinator_keys={"coordinator-key": COORDINATOR_KEY},
            now="2026-07-14T01:30:00+00:00",
        ))
        result_row = {"work_id": work_ids[0], "artifact_sha256": "1" * 64,
                      "artifact_bytes": 42, "artifact_instance_id": "remote-output-1",
                      "canonical_receipt_sha256": "2" * 64}
        result = transport.build_result_receipt(
            lease=lease, results=[result_row],
            source_verifications=[{
                "chunk_manifest_sha256": manifest["chunk_manifest_sha256"],
                "source_verification_sha256": "3" * 64,
            }],
            completed_at="2026-07-14T01:45:00+00:00",
            host_key_id="host-key", host_key=HOST_KEY,
        )
        self.assertEqual([], transport.validate_result_receipt(
            result, lease=lease, host_keys={"host-key": HOST_KEY}
        ))
        acceptance = transport.build_result_acceptance(
            result_receipt=result,
            verified_results=[{
                "work_id": work_ids[0], "artifact_sha256": "1" * 64,
                "artifact_bytes": 42, "canonical_receipt_sha256": "2" * 64,
                "transfer_verification_sha256": "4" * 64,
            }],
            coordinator_key_id="coordinator-key", coordinator_key=COORDINATOR_KEY,
        )
        self.assertEqual([], transport.validate_result_acceptance(
            acceptance, result_receipt=result,
            coordinator_keys={"coordinator-key": COORDINATOR_KEY},
        ))
        merged = transport.merge_result_receipts(
            target_work_ids=work_ids,
            receipt_leases=[(result, lease, acceptance)],
            host_keys={"host-key": HOST_KEY}, transport_plan=plan, parent=parent,
            coordinator_keys={"coordinator-key": COORDINATOR_KEY},
            coordinator_key_id="coordinator-key", coordinator_key=COORDINATOR_KEY,
        )
        self.assertEqual(1, merged["result_count"])
        self.assertFalse(merged["scientific_evidence_validated_by_parent_contract"])
        recovery = transport.build_recovery_plan(
            lease=lease, completed_work_ids=[],
            verified_chunk_sha256=[manifest["chunks"][0]["sha256"]],
            reason="host-disconnected", coordinator_key_id="coordinator-key",
            coordinator_key=COORDINATOR_KEY,
        )
        self.assertEqual(work_ids, recovery["requeue_work_ids"])
        self.assertTrue(recovery["local_only_fallback_permitted"])
        overlapping = transport.build_lease(
            transport_plan=plan, parent=parent, host_capability=host,
            work_ids=work_ids, attempt=2,
            issued_at="2026-07-14T01:15:00+00:00",
            expires_at="2026-07-14T02:15:00+00:00",
            coordinator_key_id="coordinator-key", coordinator_key=COORDINATOR_KEY,
        )
        overlap_errors = transport.validate_nonoverlapping_active_leases(
            [lease, overlapping], transport_plan=plan, parent=parent,
            coordinator_keys={"coordinator-key": COORDINATOR_KEY},
            now="2026-07-14T01:30:00+00:00",
        )
        self.assertTrue(any("overlapping" in error for error in overlap_errors))

    def test_tampered_host_attestation_is_rejected(self) -> None:
        tools = [{"role": "worker", "sha256": "f" * 64, "bytes": 123}]
        host = self._host(tools)
        host["memory_bytes"] += 1
        errors = transport.validate_host_capability(
            host, keys={"host-key": HOST_KEY},
            baseline_authority_keys={"coordinator-key": COORDINATOR_KEY},
            now="2026-07-14T01:00:00+00:00"
        )
        self.assertTrue(any("hash" in error.lower() or "HMAC" in error for error in errors))

    def test_signed_bounded_nonzero_swap_capability_is_admitted(self) -> None:
        tools = [{"role": "worker", "sha256": "f" * 64, "bytes": 123}]
        mib = 1024 * 1024
        resource = self._resource(
            swap_used_bytes=600 * mib, previous_swap_used_bytes=500 * mib
        )
        host = self._host(tools, resource_state=resource)
        self.assertEqual("soft_throttle", resource["aggressive_admission"][
            "decision"
        ]["mode"])
        self.assertEqual([], transport.validate_host_capability(
            host, keys={"host-key": HOST_KEY},
            baseline_authority_keys={"coordinator-key": COORDINATOR_KEY},
            now="2026-07-14T01:00:00+00:00"
        ))

    def test_remote_swap_threshold_rate_and_policy_drift_fail_closed(self) -> None:
        tools = [{"role": "worker", "sha256": "f" * 64, "bytes": 123}]
        mib = 1024 * 1024
        cases = [
            self._resource(swap_used_bytes=1600 * mib,
                           previous_swap_used_bytes=1500 * mib),
            self._resource(swap_used_bytes=1200 * mib,
                           sealed_baseline_swap_bytes=1100 * mib,
                           previous_swap_used_bytes=0),
        ]
        for resource in cases:
            host = self._host(tools, resource_state=resource)
            errors = transport.validate_host_capability(
                host, keys={"host-key": HOST_KEY},
                baseline_authority_keys={"coordinator-key": COORDINATOR_KEY},
                now="2026-07-14T01:00:00+00:00",
            )
            self.assertTrue(any("does not admit" in row for row in errors), errors)

        drifted = self._resource(swap_used_bytes=100 * mib)
        admission = drifted["aggressive_admission"]
        admission["policy"]["soft_growth_mb"] = 999.0
        admission["policy_sha256"] = transport._hash_value(admission["policy"])
        admission["attestation_sha256"] = transport._hash_value(
            transport._without(admission, "attestation_sha256")
        )
        host = self._host(tools, resource_state=drifted)
        errors = transport.validate_host_capability(
            host, keys={"host-key": HOST_KEY},
            baseline_authority_keys={"coordinator-key": COORDINATOR_KEY},
            now="2026-07-14T01:00:00+00:00",
        )
        self.assertTrue(any("policy/controller" in row for row in errors), errors)

    def test_changed_baseline_without_new_coordinator_authority_is_rejected(self) -> None:
        tools = [{"role": "worker", "sha256": "f" * 64, "bytes": 123}]
        mib = 1024 * 1024
        resource = self._resource(swap_used_bytes=100 * mib)
        admission = resource["aggressive_admission"]
        authority = admission["baseline_authority"]
        authority["sealed_baseline_swap_bytes"] = 900 * mib
        authority["sealed_baseline_swap_mb"] = 900.0
        authority["baseline_authority_sha256"] = transport._hash_value(
            transport._without(
                authority, "baseline_authority_sha256", "signature"
            )
        )
        admission["attestation_sha256"] = transport._hash_value(
            transport._without(admission, "attestation_sha256")
        )
        host = self._host(tools, resource_state=resource)
        errors = transport.validate_host_capability(
            host, keys={"host-key": HOST_KEY},
            baseline_authority_keys={"coordinator-key": COORDINATOR_KEY},
            now="2026-07-14T01:00:00+00:00",
        )
        self.assertTrue(any("HMAC" in row or "baseline" in row for row in errors), errors)

    def test_remote_capability_preserves_hard_stop_hysteresis(self) -> None:
        tools = [{"role": "worker", "sha256": "f" * 64, "bytes": 123}]
        observed_epoch = transport._parse_time(self.OBSERVED_AT).timestamp()
        authority = transport.build_swap_baseline_authority(
            host_id="second-host", instance_nonce="boot-1",
            sealed_baseline_swap_bytes=0, authorized_at=self.OBSERVED_AT,
            expires_at=self.EXPIRES_AT, coordinator_key_id="coordinator-key",
            coordinator_key=COORDINATOR_KEY,
        )
        state = transport.aggressive_admission.initial_swap_state(
            {"pressure_level": 1, "swap_used_mb": 0.0},
            now_epoch=observed_epoch - 300.0,
        )
        state, hard = transport.aggressive_admission.advance_swap_state(
            state, {"pressure_level": 2, "swap_used_mb": 0.0},
            now_epoch=observed_epoch - 100.0, sealed_baseline_swap_mb=0.0,
        )
        self.assertEqual("hard_stop", hard["mode"])
        resource = transport.build_resource_attestation(
            baseline_authority=authority, prior_controller_state=state,
            swap_used_bytes=0, sampled_epoch=observed_epoch,
        )
        decision = resource["aggressive_admission"]["decision"]
        self.assertEqual("hard_stop", decision["mode"])
        self.assertIn("hysteresis", decision["reason"])
        host = self._host(tools, resource_state=resource)
        errors = transport.validate_host_capability(
            host, keys={"host-key": HOST_KEY},
            baseline_authority_keys={"coordinator-key": COORDINATOR_KEY},
            now="2026-07-14T01:00:00+00:00",
        )
        self.assertTrue(any("does not admit" in row for row in errors), errors)


if __name__ == "__main__":
    unittest.main()
