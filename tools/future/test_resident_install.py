import json

from tools.future import resident_install as ri
from tools.future._common import RECEIPTS, _assert_no_hardware_claims


def test_build_emits_sealed_receipt():
    out = ri.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "RESIDENT_INSTALL_CONTRACT.json"
    assert doc["schema"] == "hawking.future.resident_install.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["phases"] == list(ri.PHASES)
    _assert_no_hardware_claims(doc)


def test_selftest_emits_sealed_receipt():
    out = ri.selftest()
    assert json.loads(out.read_text())["seal_sha256"]


def test_unbound_contract_fails_validation():
    unbound = ri.empty_contract()
    problems = ri.validate_contract(unbound)
    assert problems
    assert unbound["winner_id"] is None
    assert unbound["generic"] is True
    assert ri.bound(unbound) is False


def test_contract_has_every_required_phase():
    required = (
        "nx_identity",
        "executable_identity",
        "tokenizer_session",
        "memory_requirements",
        "launch_args",
        "readiness_probe",
        "shutdown",
        "unload",
        "protected_benchmark_evacuation",
        "restart",
        "crash_recovery",
        "fallback_policy",
        "capability_receipt",
        "performance_receipt",
    )
    assert ri.PHASES == required
    unbound = ri.empty_contract()
    assert list(unbound["slots"]) == list(required)


def test_bind_winner_is_generic_over_identity_shape():
    flashish = ri.bind_winner(
        "FLASH_SINGULARITY.NX",
        {
            "nx_kind": "hawking.nos.flash_noetic_executable_genome",
            "status": "SEALED_METADATA_ONLY_NOT_FOR_PROMOTION",
            "seal_sha256": "abc",
            "physical_program": {"executor": [{"path": "/tmp/flash-exec"}]},
            "lowers_nr": {"path": "receipts/headless/FLASH_COMPLETE_V0.nr.json"},
        },
        identity_path="receipts/headless/FLASH_COMPLETE_V0.nx.json",
        extra={
            "tokenizer_path": "/tmp/tok.json",
            "prompt_contract": {"renderer": "flash"},
            "quoted_artifact_bytes": 1,
            "capability_receipt_path": "receipts/headless/CAPABILITY_flash.json",
            "performance_receipt_path": "receipts/headless/PERF_flash.json",
        },
    )
    qwenish = ri.bind_winner(
        "QWEN27_SINGULARITY.NX",
        {
            "resident_identity": "sealed-3.14",
            "protocol": "hawking.qwen38.resident.v1",
            "resident_binary": "/tmp/qwen-res",
            "artifact_root": "/tmp/art",
            "tokenizer": "/tmp/qwen-tok.json",
            "prompt_contract": {"renderer": "qwen"},
            "fusion_env": {"HAWKING_QWEN38_FUSE_MLP": "swiglu"},
        },
        identity_path="hcli/hawking-native.sealed-3.14.json",
        extra={
            "quoted_artifact_bytes": 2,
            "capability_receipt_path": "receipts/headless/CAPABILITY_noetic-sealed-3.14.json",
            "performance_receipt_path": "receipts/headless/PERF_qwen.json",
        },
    )
    assert not ri.validate_contract(flashish)
    assert not ri.validate_contract(qwenish)
    assert flashish["winner_id"] == "FLASH_SINGULARITY.NX"
    assert qwenish["winner_id"] == "QWEN27_SINGULARITY.NX"
    assert flashish["slots"]["nx_identity"]["nx_kind"].endswith("flash_noetic_executable_genome")
    assert qwenish["slots"]["nx_identity"]["nx_kind"] == "hawking.qwen38.resident.v1"
    assert flashish["slots"]["protected_benchmark_evacuation"]["stop_before_closing_quiescence"] is True
    assert qwenish["slots"]["protected_benchmark_evacuation"]["stop_before_closing_quiescence"] is True
    assert qwenish["slots"]["performance_receipt"]["required_bench_state"] == "PROTECTED_ABSOLUTE"


def test_default_template_does_not_hardcode_a_winner():
    unbound = ri.empty_contract()
    assert unbound["winner_id"] is None
    assert unbound["policy"]["hardcoded_winner_forbidden"] is True
    text = json.dumps(unbound)
    assert "FLASH_SINGULARITY.NX" not in text
    assert "QWEN27_SINGULARITY.NX" not in text
    assert unbound["slots"]["performance_receipt"]["required_bench_state"] == "PROTECTED_ABSOLUTE"


def test_incomplete_binding_is_rejected():
    partial = ri.bind_winner("X", {"nx_kind": "example"}, identity_path="x.json")
    problems = ri.validate_contract(partial)
    assert any(p.startswith("executable_identity.") for p in problems)
    assert any(p.startswith("tokenizer_session.") for p in problems)
    assert ri.bound(partial) is False
