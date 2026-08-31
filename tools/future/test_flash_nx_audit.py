import json

from tools.future import flash_nx_audit as audit
from tools.future._common import RECEIPTS, HardwareClaimError, write_receipt


def _docs():
    return audit._load_all()


def test_build_emits_sealed_receipt():
    out = audit.build()
    assert out.name == "FLASH_NX_COMPLETENESS_AUDIT.json"
    assert out.parent == RECEIPTS
    doc = json.loads(out.read_text())
    assert doc["schema"] == "hawking.future.flash_nx_audit.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    import hashlib
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]


def test_selftest_aliases_build():
    assert audit.selftest == audit.build


def test_seven_requirements_named_and_cited_from_receipts():
    docs = _docs()
    rows = audit.seven_from_disk(docs)
    names = [r["requirement"] for r in rows]
    assert names == list(audit.SEVEN_REQUIREMENTS)
    for row in rows:
        assert row["state"] == "NOT_MET"
        assert row["cited"], f"{row['requirement']} must cite at least one field"
        for cite in row["cited"]:
            assert cite["path"]
            assert cite["field"]
        checker = row["checker_on_FLASH_COMPLETE_V0_nx"]
        assert checker["ok"] is False


def test_fourteen_blocked_flash_candidates_verbatim_reasons():
    docs = _docs()
    blocked = audit.flash_blocked_candidates(docs["queue"])
    # Disk is authority. The frontier still says 12-of-14; the live queue has
    # grown by flash-pipeline-id-resolution and every BLOCKED row is Flash.
    assert len(blocked) == docs["queue"]["counts"]["by_status"]["BLOCKED"]
    assert len(blocked) >= 14
    reasons = {c["candidate_id"]: c["blocked_reason"] for c in blocked}
    assert reasons["flash-attention-gate-fusion"] == (
        "Flash complete-token and source-independent capability gates remain open"
    )
    assert reasons["flash-compact-moe-bf16-vec4"] == (
        "source-independent Flash NX and protected complete-token capability are not qualified"
    )
    assert reasons["flash-compact-moe-epilogue"] == (
        "Flash source-independent compact expert consumer is not yet qualified"
    )
    assert reasons["flash-encoder-label-elision"] == (
        "source-independent Flash NX and protected complete-token capability are not qualified"
    )
    assert reasons["flash-fullseq-catalog-cache"] == (
        "source-independent Flash NX and protected complete-token capability are not qualified"
    )
    assert reasons["flash-fullseq-ordered-encoder"] == (
        "fullseq source path is not a complete source-independent Flash NX executable"
    )
    assert reasons["flash-hc-staged-threadgroup"] == (
        "Flash complete source-independent NX executable is not qualified"
    )
    assert reasons["flash-meta-sub1-coherent"] == (
        "meta budget is only a prospective function description; no serialized functional "
        "artifact, source-independent Flash NX consumer, or protected complete-token path exists"
    )
    assert reasons["flash-pipeline-cache-reuse"] == (
        "source-independent Flash NX and protected complete-token capability are not qualified"
    )
    assert reasons["flash-pipeline-id-resolution"] == (
        "source-independent Flash NX and protected complete-token capability are not qualified"
    )
    assert reasons["flash-qkv-gqa-rope-fusion"] == (
        "source-independent Flash NX and protected complete-token capability are not qualified"
    )
    assert reasons["flash-routed-fp4-gate-up-swiglu-fused"] == (
        "Flash source-independent NX and protected complete-token capability are not qualified"
    )
    assert reasons["flash-router-topk-fusion"] == (
        "source-independent Flash route consumer and protected capability are not qualified"
    )
    assert reasons["flash-shared-fp8-gate-up-swiglu-fused"] == (
        "Flash source-independent NX and protected complete-token capability are not qualified"
    )
    assert reasons["flash-source-bf16-simd"] == (
        "source oracle is a control; Flash NX and protected complete-token path are open"
    )
    for reason in reasons.values():
        assert isinstance(reason, str) and reason.strip()


def test_twelve_of_fourteen_collapse_to_the_dominant_dependency():
    docs = _docs()
    chain = audit.dependency_chain(audit.flash_blocked_candidates(docs["queue"]))
    assert chain["additional_count"] == 2
    assert set(chain["additional_candidates"]) == {
        "flash-compact-moe-epilogue",
        "flash-meta-sub1-coherent",
    }
    assert chain["dominant_count"] == chain["blocked_count"] - 2
    assert chain["blocked_count"] >= 14
    assert "flash-pipeline-id-resolution" in chain["dominant_candidates"]
    assert chain["dominant_dependency"] == (
        "source-independent Flash NX with a protected complete-token measurement"
    )
    window = chain["gpu_window_worth_the_most"]
    assert window["piece"] == "protected_complete_token_measurement"
    assert window["unblocks_count"] == chain["dominant_count"]
    assert set(window["unblocks_to_qualifiable"]) == set(chain["dominant_candidates"])
    assert set(window["still_blocked_after_window"]) == set(chain["additional_candidates"])


def test_topological_order_cpu_chain_before_gpu_window():
    docs = _docs()
    chain = audit.dependency_chain(audit.flash_blocked_candidates(docs["queue"]))
    order = [row["id"] for row in chain["topological_order"]]
    assert order[:4] == [
        "serialized_source_independent_nx_artifact",
        "physical_loader",
        "whole_model_native_kernel_binding",
        "protected_complete_token_measurement",
    ]
    ranks = {row["id"]: row["rank"] for row in chain["topological_order"]}
    assert ranks["serialized_source_independent_nx_artifact"] < ranks["physical_loader"]
    assert ranks["physical_loader"] < ranks["whole_model_native_kernel_binding"]
    assert ranks["whole_model_native_kernel_binding"] < ranks["protected_complete_token_measurement"]
    gpu = next(r for r in chain["topological_order"] if r["id"] == "protected_complete_token_measurement")
    assert gpu["lane"] == "GPU"
    assert gpu["unblocks_count"] == chain["dominant_count"]
    assert gpu["unblocks_count"] >= 12


def test_runtime_audit_distinguishes_designed_from_built():
    docs = _docs()
    rows = {r["need"]: r for r in audit.runtime_dependency_audit(docs)}
    assert rows["serialized_nx_artifact"]["designed"] is True
    assert rows["serialized_nx_artifact"]["built"] is False
    assert rows["physical_loader"]["designed"] is True
    assert rows["physical_loader"]["built"] is False
    assert rows["native_kernel_catalog"]["designed"] is True
    assert rows["native_kernel_catalog"]["built"] is False
    assert rows["whole_model_nx_consumer"]["built"] is False
    # Built for a different artifact class, not wired as Flash NX.
    assert rows["packed_qwen38_greedy_as_flash_loader"]["built"] is True
    assert rows["packed_qwen38_greedy_as_flash_loader"]["wired"] is False
    assert rows["source_oracle_executors"]["built"] is True
    assert rows["source_oracle_executors"]["wired"] is False
    assert docs["executable"]["native_loader"]["status"] == "NOT_IMPLEMENTED"
    assert docs["executable"]["native_kernels"]["status"] == "PLAN_ONLY"
    assert docs["meta"]["measurement_state"]["serialized_artifact"] == "NOT_BUILT"
    assert docs["meta"]["measurement_state"]["physical_loader"] == "NOT_BUILT"
    assert docs["meta"]["measurement_state"]["native_kernel"] == "NOT_BUILT"


def test_dense_rematerialization_verification_vs_production():
    docs = _docs()
    dense = audit.dense_rematerialization_audit(docs)
    assert dense["production_may_not_reconstruct_dense_weights"] is True
    assert dense["verification_may_reconstruct"] is True
    assert dense["policy"]["meta_program.dense_weight_materialization"] == "forbidden"
    assert dense["policy"]["meta_accelerator_contract.dense_rematerialization"] is False
    assert dense["verification_reconstructs"]["found"] is True
    assert dense["production_path"]["whole_model_nx_production"] == "NOT_BUILT"
    assert dense["verdict"]["built_production_compact_path"] == "NOT_BUILT"
    assert "eliminate dense source-bank reloads" in (docs["token_accepted"].get("next") or "")


def test_checker_rejects_real_v0_nx_and_names_the_reason():
    """Negative control: the refusal must actually fire on the real metadata NX."""
    path = audit.evidence_path(audit.REL_NX_V0)
    assert path is not None, "FLASH_COMPLETE_V0.nx.json must be recoverable from disk"
    nx = json.loads(path.read_text())
    assert nx["status"] == audit.METADATA_ONLY
    ledger = audit.load_evidence(audit.REL_LEDGER)
    result = audit.check_nx(nx, context={"byte_ledger": ledger})
    assert result["promotable"] is False
    assert result["status"] == audit.METADATA_ONLY
    blob = " ".join(result["reasons"])
    assert "SEALED_METADATA_ONLY_NOT_FOR_PROMOTION" in blob
    assert "self_contained_dependencies" in result["failed_requirements"]
    assert "protected_performance" in result["failed_requirements"]
    assert "accepted_generation" in result["failed_requirements"]
    assert "capability" in result["failed_requirements"]
    assert "complete_byte_ledger" in result["failed_requirements"]
    assert "reproducibility" in result["failed_requirements"]
    assert "no_forbidden_fallback" in result["failed_requirements"]
    assert set(result["failed_requirements"]) == set(audit.SEVEN_REQUIREMENTS)


def test_checker_accepts_synthetic_that_satisfies_all_seven():
    synth = audit.synthetic_promotable_nx()
    result = audit.check_nx(synth)
    assert result["promotable"] is True
    assert result["failed_requirements"] == []
    assert all(result["requirements"][r]["ok"] for r in audit.SEVEN_REQUIREMENTS)


def test_checker_discriminates_when_one_requirement_is_stripped():
    synth = audit.synthetic_promotable_nx()
    synth["status"] = audit.METADATA_ONLY
    result = audit.check_nx(synth)
    assert result["promotable"] is False
    assert any("SEALED_METADATA_ONLY_NOT_FOR_PROMOTION" in r for r in result["reasons"])

    synth = audit.synthetic_promotable_nx()
    del synth["capability"]
    result = audit.check_nx(synth)
    assert result["promotable"] is False
    assert "capability" in result["failed_requirements"]

    synth = audit.synthetic_promotable_nx()
    synth["fallback"]["fallback_count"] = 1
    result = audit.check_nx(synth)
    assert result["promotable"] is False
    assert "no_forbidden_fallback" in result["failed_requirements"]

    synth = audit.synthetic_promotable_nx()
    synth["protected_performance"]["measurement_class"] = "DIAGNOSTIC_RELATIVE"
    synth["bench"]["state"] = "UNKNOWN"
    result = audit.check_nx(synth)
    assert result["promotable"] is False
    assert "protected_performance" in result["failed_requirements"]


def test_receipt_refuses_hardware_claims():
    raised = False
    try:
        write_receipt(
            "_flash_nx_audit_hardware_probe.json",
            {"schema": "probe", "tps": 12.0},
            "tools/future/test_flash_nx_audit.py",
        )
    except HardwareClaimError:
        raised = True
    assert raised, "write_receipt must refuse a numeric tps field"


def test_nr_v2_and_nx_v0_statuses_match_the_contract():
    docs = _docs()
    assert docs["nr_v2"]["status"] == audit.NR_NOT_FOR_PROMOTION
    assert docs["nx_v0"]["status"] == audit.METADATA_ONLY
    assert docs["nx_next"]["status"] == audit.METADATA_ONLY
    assert docs["meta"]["measurement_state"]["serialized_artifact"] == "NOT_BUILT"
    assert docs["meta"]["measurement_state"]["physical_loader"] == "NOT_BUILT"
    assert docs["meta"]["measurement_state"]["native_kernel"] == "NOT_BUILT"
    assert docs["meta"]["measurement_state"]["complete_token"] == "NOT_MEASURED"
    assert docs["meta"]["measurement_state"]["capability"] == "NOT_MEASURED"
    assert docs["executable"]["status"] == "SCAFFOLD_ONLY"
    assert docs["token_attempt"]["status"] == "BLOCKED"


def test_assembled_receipt_has_required_sections():
    docs = _docs()
    doc = audit.assemble(docs)
    for key in (
        "schema",
        "version",
        "seven_requirements",
        "dependency_chain",
        "runtime_dependency_audit",
        "dense_rematerialization",
        "nx_completeness_checker",
        "recovered_implementation",
        "gaps_closed",
        "negative_findings",
    ):
        assert key in doc
    assert doc["seven_all_met"] is False
    assert doc["nx_completeness_checker"]["discriminator_holds"] is True
    assert doc["dependency_chain"]["dominant_count"] == doc["dependency_chain"]["blocked_count"] - 2
    assert doc["dependency_chain"]["blocked_count"] >= 14
