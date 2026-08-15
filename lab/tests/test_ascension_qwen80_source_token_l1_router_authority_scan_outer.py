from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from lab.operators import ascension_qwen80_source_token_l1_router_authority_scan_outer as outer
from lab.operators import ascension_qwen80_source_token_l0_route_plan as source_identity
from lab.receipts import seal, verify


# These are the Rust child's accepted outer authority identifiers.  Keep this
# literal regression beside the fake-child lifecycle test so a Python-only
# rename cannot mint a launch receipt that the real CPU child rejects before
# its single admitted catalog scan.
RUST_OUTER_LAUNCH_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l1_all_ten_route_authority_"
    "outer_launch_authority.v1"
)
RUST_OUTER_LAUNCH_STATUS = (
    "AUTHORIZED_QWEN80_SOURCE_TOKEN_L1_ALL_TEN_ROUTE_AUTHORITY_CPU_CHILD_"
    "ONE_SHOT"
)
HISTORICAL_RUST_INNER = (
    outer.REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-runtime"
    / "QWEN80_SOURCE_TOKEN_L1_ROUTE_AUTHORITY_CPU_SCAN_20260809T130548Z/inner"
    / "l1-source-token-route-authority.json"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha(value: object) -> str:
    payload = value if isinstance(value, bytes) else _canonical(value)
    return hashlib.sha256(payload).hexdigest()


def _write_sealed(path: Path, value: dict[str, object]) -> dict[str, object]:
    document = seal(value)
    path.write_bytes(_canonical(document))
    return document


def _binding(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    document = json.loads(raw)
    return {
        "path": str(path.resolve()),
        "present": True,
        "bytes": len(raw),
        "sha256": _sha(raw),
        "raw_sha256": _sha(raw),
        # Rust's document identity is the seal of the unsigned canonical
        # document; raw on-disk identity remains ``sha256``.
        "document_sha256": document["seal_sha256"],
        "document_seal_sha256": document["seal_sha256"],
    }


def _identity(path: Path) -> dict[str, object]:
    binding = _binding(path)
    return {
        "present": True,
        "document_sha256": binding["document_sha256"],
        "document_seal_sha256": binding["document_seal_sha256"],
    }


def _write_fake_producer(path: Path, *, counter: Path | None = None, valid: bool = True) -> None:
    counter_code = ""
    if counter is not None:
        counter_code = f"""
counter = pathlib.Path({str(counter)!r})
count = int(counter.read_text() if counter.exists() else '0')
counter.write_text(str(count + 1))
"""
    if not valid:
        body = "pathlib.Path(value('--out')).write_text('{}')\n"
    else:
        body = """
def evidence(path):
    path = pathlib.Path(path)
    raw = path.read_bytes()
    document = json.loads(raw)
    return {
        'path': str(path.resolve()), 'present': True, 'bytes': len(raw),
        'sha256': sha_bytes(raw), 'raw_sha256': sha_bytes(raw),
        'document_sha256': document['seal_sha256'],
        'document_seal_sha256': document['seal_sha256'],
    }

def binary_evidence(path):
    path = pathlib.Path(path)
    raw = path.read_bytes()
    return {
        'path': str(path.resolve()), 'present': True, 'bytes': len(raw),
        'sha256': sha_bytes(raw),
    }

producer_preflight = evidence(value('--producer-preflight'))
producer_binary = binary_evidence(value('--producer-binary'))
launch_path = value('--outer-launch-authority')
launch = evidence(launch_path)
launch_document = json.loads(pathlib.Path(launch_path).read_text())
admission_current = evidence(value('--admission-current'))
versioned_current_admission = dict(launch_document['versioned_current_admission'])
versioned_current_admission['terminal_observed'] = admission_current
manifest = evidence(value('--manifest'))
assessment = evidence(value('--joint-assessment'))
admission = json.loads(pathlib.Path(value('--admission-current')).read_text())
admission_receipt = evidence(admission['admission_receipt']['path'])
ids = list(range(10))
weights = [0.1] * 10
counter = 0
def descriptor():
    global counter
    counter += 1
    return {
      'artifact_sha256': f'{counter:064x}',
      'direct_packed_payload_sha256': f'{counter + 100:064x}',
      'header_sha256': f'{counter + 200:064x}',
      'payload_bytes': 1,
      'layout': {'magic':'HQ30G1B1','version':1,'group_size':128,'scale_dtype':'float16','sign_bit_order':'little'},
    }
waves = []
for index, expert_id in enumerate(ids):
    waves.append({
      'wave_index': index, 'layer': 1, 'expert_id': expert_id,
      'normalized_weight': weights[index],
      'normalized_weight_bits_hex': f\"0x{struct.unpack('!Q', struct.pack('!d', weights[index]))[0]:016x}\",
      'gate': descriptor(), 'up': descriptor(), 'down': descriptor(),
    })
document = {
  'schema': 'hawking.ascension.qwen80_source_token_l1_all_ten_route_authority.v1',
  'status': 'SEALED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L1_ALL_TEN_ROUTE_AUTHORITY_READY_FOR_SAME_RUNTIME_MOE_SUFFIX',
  'fixture_or_synthetic': False,
  'metal_or_gpu_activity_performed': False,
  'metal_device_or_dispatch_performed': False,
  'producer_preflight': producer_preflight,
  'producer_binary': producer_binary,
  'outer_launch_authority_binding': launch,
  'versioned_current_admission': versioned_current_admission,
  'cpu_outer_capture': {
      'capture_dir': value('--capture-dir'),
      'output_authority_path': value('--out'),
      'workers': int(value('--workers')),
      'one_current_admitted_catalog_scan_performed': True,
      'raw_bf16_or_safetensors_reopened': False,
      'outer_terminal_receipt_written_by_parent_last': True,
  },
  'source_binding': {
      'manifest_document_sha256': manifest['sha256'],
      'manifest_seal_sha256': manifest['document_seal_sha256'],
      'admission_receipt_seal_sha256': admission_receipt['document_seal_sha256'],
      'joint_l0_l1_assessment': {
          'document_sha256': assessment['document_sha256'],
          'document_seal_sha256': assessment['document_seal_sha256'],
      },
      'prior_joint_assessment_is_provenance_only': True,
      'cross_process_pinned_buffer_import_allowed': False,
  },
  'source_token_l1_cpu_oracle': {
      'source_token_id': 1, 'layer': 1, 'linear_state_slot': 1,
      'fresh_l0_reencode_dispatches': 23, 'fresh_l1_prefix_dispatches': 9,
      'cpu_oracle_reencodes_l0_then_l1_prefix': True,
      'zero_initial_l0_state': True, 'zero_initial_l1_slot1_state': True,
      **{name: f'{index + 300:064x}' for index, name in enumerate([
          'source_input_f32le_sha256', 'l0_second_residual_cpu_f32le_sha256',
          'l1_prefix_input_cpu_f32le_sha256', 'l1_first_residual_cpu_f32le_sha256',
          'l1_post_attention_normalized_hidden_cpu_f32le_sha256',
          'l1_router_logits_cpu_f32le_sha256', 'l1_post_conv_state_cpu_f32le_sha256',
          'l1_post_recurrent_state_cpu_f32le_sha256'])},
  },
  'source_token_router_evidence': {
      'logit_count': 512, 'top_k': 10, 'selection': 'source_qwen80_topk_router',
      'tie_break': 'lowest_expert_id_within_route_tie_epsilon',
      'softmax': 'subtract_max_exp_f32', 'route_tie_epsilon_source': 'HAWKING_DS_ROUTE_TIE_EPS',
      'selected_probabilities_renormalized': True, 'route_tie_epsilon': 0.0,
      'route_tie_epsilon_f32_bits_hex': '0x00000000',
      'source_stable_route_ids': ids, 'source_stable_normalized_weights': weights,
      'weights_sum': sum(weights),
  },
  'fixed_l1_payloads': [descriptor() for _ in range(6)],
  'deterministic_waves': waves,
  'rawls_real_all_ten_provenance_gate': {
      'all_ten_source_bindings_complete': True,
      'execution_receipt_required_for_each_wave': True,
      'direct_packed_execution_required_for_each_wave': True,
      'source_bound_input_required_for_each_wave': True,
      'route_combine_receipt_required_separately': True,
      'shared_expert_receipt_required_separately': True,
      'first_and_second_residual_receipts_required_separately': True,
      'rejects_tensor_substitution': True, 'rejects_route_reorder': True,
      'rejects_duplicate_experts': True, 'rejects_missing_tensor_or_weight': True,
      'expected_layer': 1,
  },
  'route_execution_performed': False, 'route_combine_performed': False,
  'shared_expert_performed': False, 'residual_combine_performed': False,
  'model_execution_performed': False, 'hcli_execution_performed': False,
  'tps_or_tg_measurement_performed': False, 'complete_layer_or_decoder_claim_earned': False,
}
document['seal_sha256'] = sha_json(document)
pathlib.Path(value('--out')).write_text(json.dumps(document, sort_keys=True, separators=(',', ':')))
"""
    script = f"""#!/usr/bin/env python3
import hashlib
import json
import pathlib
import struct
import sys

def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()

def sha_json(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode())

def value(flag):
    index = sys.argv.index(flag)
    return sys.argv[index + 1]
{counter_code}
{body}
"""
    path.write_text(script)
    path.chmod(0o755)


def _fixture(tmp_path: Path, *, valid_child: bool = True) -> tuple[outer.PreflightConfig, Path]:
    manifest_path = tmp_path / "manifest.json"
    manifest = _write_sealed(manifest_path, {"schema": source_identity.MANIFEST_SCHEMA})
    admission_receipt_path = tmp_path / "admission-receipt.json"
    admission_receipt = _write_sealed(
        admission_receipt_path,
        {
            "schema": source_identity.ADMISSION_RECEIPT_SCHEMA,
            "status": source_identity.ADMISSION_RECEIPT_STATUS,
        },
    )
    admission_path = tmp_path / "admission-current.json"
    _write_sealed(
        admission_path,
        {
            "schema": source_identity.ADMISSION_SCHEMA,
            "status": source_identity.ADMISSION_STATUS,
            "complete_manifest": {
                "document_sha256": _sha(manifest_path.read_bytes()),
                "seal_sha256": manifest["seal_sha256"],
            },
            "admission_receipt": {
                "path": str(admission_receipt_path.resolve()),
                "seal_sha256": admission_receipt["seal_sha256"],
            },
        },
    )
    assessment_path = tmp_path / "joint-assessment.json"
    _write_sealed(
        assessment_path,
        {"schema": outer.JOINT_ASSESSMENT_SCHEMA, "status": outer.JOINT_ASSESSMENT_STATUS},
    )
    completion_path = tmp_path / "completion-preflight.json"
    _write_sealed(
        completion_path,
        {
            "schema": outer.COMPLETION_PREFLIGHT_SCHEMA,
            "status": outer.COMPLETION_PREFLIGHT_STATUS,
            "preflight_ready_for_future_outer_authority_only": False,
            "antecedent_l0_l1_component": _identity(assessment_path),
        },
    )
    producer_binary = tmp_path / outer.PRODUCER_BINARY_NAME
    _write_fake_producer(producer_binary, valid=valid_child)
    initial = outer.PreflightConfig(
        producer_preflight=tmp_path / "future-producer-preflight.json",
        manifest=manifest_path,
        admission_current=admission_path,
        joint_assessment=assessment_path,
        completion_preflight=completion_path,
        producer_binary=producer_binary,
    )
    current = outer._read_current_source(initial)
    producer_preflight_path = initial.producer_preflight
    _write_sealed(
        producer_preflight_path,
        {
            "schema": outer.PRODUCER_PREFLIGHT_SCHEMA,
            "status": outer.PRODUCER_PREFLIGHT_STATUS,
            "source_binding": {
                **{name: outer._binding(bound) for name, bound in current.items()},
                "manifest_seal_sha256": current["manifest"].document_seal_sha256,
                "admission_receipt_seal_sha256": current[
                    "admission_receipt"
                ].document_seal_sha256,
                "admission_current_pointer_seal_sha256": current[
                    "admission_current"
                ].document_seal_sha256,
                "source_audit_seal_sha256": "a" * 64,
                "source_revision": "test-source-revision",
            },
            "producer_binary": outer._file_evidence(producer_binary, "producer", executable=True),
            "versioned_current_admission": outer._versioned_current_admission(current),
            "dynamic_authority_contract": {
                "schema": outer.DYNAMIC_AUTHORITY_SCHEMA,
                "status": outer.DYNAMIC_AUTHORITY_STATUS,
                "all_ten_dynamic_router_ids_and_weights_required": True,
                "exact_fixed_payload_requirements": [{} for _ in range(6)],
                "exact_route_payloads_required": 30,
                "source_token_id": 1,
                "l0_reencode_dispatches": 23,
                "l1_layer": 1,
                "l1_linear_state_slot": 1,
                "l1_moe_suffix_dispatches": 14,
                "l1_prefix_dispatches": 9,
                "no_fixture_or_cross_process_buffer_substitution": True,
                "one_current_admitted_cpu_catalog_scan_required": True,
                "outer_launch_authority_binding_required": True,
                "planned_output_must_be_new_under_outer_capture_dir": True,
            },
            "claim_boundary": {
                "strict_catalog_admission_scan_performed": False,
                "admitted_payload_snapshot_opened": False,
                "child_started": False,
                "metal_or_gpu_activity_performed": False,
                "lease_issued_or_consumed": False,
                "watcher_or_server_changed": False,
                "model_token_or_tps_claim_earned": False,
                "complete_layer_or_decoder_claim_earned": False,
                "preflight_only": True,
            },
        },
    )
    return initial, producer_binary


def test_preflight_is_file_only_and_never_spawns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, _ = _fixture(tmp_path)

    def unexpected_spawn(*args: object, **kwargs: object) -> object:
        raise AssertionError("preflight must not create a subprocess")

    monkeypatch.setattr(outer.subprocess, "Popen", unexpected_spawn)
    document = outer.build_outer_preflight(config)
    assert document["status"] == outer.OUTER_PREFLIGHT_STATUS
    assert document["child_spawned"] is False
    assert document["catalog_or_payload_scan_performed"] is False
    verify(document, label="outer preflight")


def test_child_command_is_exact_cpu_oracle_interface(tmp_path: Path) -> None:
    config, binary = _fixture(tmp_path)
    out = tmp_path / "outer-preflight.json"
    outer.write_outer_preflight(config, out)
    preflight = outer._read_outer_preflight(out)
    source = outer._source_from_outer(preflight)
    producer = outer._read_bound(
        config.producer_preflight,
        "producer",
        outer.PRODUCER_PREFLIGHT_SCHEMA,
        outer.PRODUCER_PREFLIGHT_STATUS,
    )
    command = outer._child_command(
        producer_binary=outer._file_evidence(binary, "producer", executable=True),
        source=source,
        producer_preflight=producer,
        launch_authority_path=tmp_path / "capture" / outer.OUTER_LAUNCH_FILENAME,
        capture_dir=tmp_path / "capture",
        workers=2,
    )
    assert command[0] == str(binary.resolve())
    assert command[1:3] == ["--mode", "cpu-oracle"]
    for flag in (
        "--manifest",
        "--admission-current",
        "--joint-assessment",
        "--completion-preflight",
        "--producer-preflight",
        "--producer-binary",
        "--outer-launch-authority",
        "--capture-dir",
        "--out",
        "--workers",
    ):
        assert flag in command
    assert not any("metal" in item.lower() or "lease" in item.lower() for item in command)


def test_explicit_one_shot_reaps_one_fake_child_and_replays_terminal(tmp_path: Path) -> None:
    config, binary = _fixture(tmp_path)
    counter = tmp_path / "counter.txt"
    _write_fake_producer(binary, counter=counter)
    # Refresh the producer preflight so its full executable evidence is exact.
    current = outer._read_current_source(config)
    producer_preflight = json.loads(config.producer_preflight.read_text())
    producer_preflight.pop("seal_sha256")
    producer_preflight["producer_binary"] = outer._file_evidence(binary, "producer", executable=True)
    _write_sealed(config.producer_preflight, producer_preflight)
    preflight_path = tmp_path / "outer-preflight.json"
    outer.write_outer_preflight(config, preflight_path)
    capture = tmp_path / "capture"
    attempt = outer.CaptureConfig(
        outer_preflight=preflight_path,
        producer_binary=binary,
        capture_dir=capture,
        replay_guard_dir=tmp_path / "replay",
        workers=2,
        timeout_seconds=30.0,
    )
    terminal = outer.run_attempt(attempt)
    assert terminal["status"] == outer.CAPTURED_STATUS
    assert terminal["child"]["terminal"]["reaped"] is True
    assert counter.read_text() == "1"
    launch = json.loads((capture / outer.OUTER_LAUNCH_FILENAME).read_text())
    assert outer.OUTER_LAUNCH_SCHEMA == RUST_OUTER_LAUNCH_SCHEMA
    assert outer.OUTER_LAUNCH_STATUS == RUST_OUTER_LAUNCH_STATUS
    assert launch["schema"] == outer.OUTER_LAUNCH_SCHEMA
    assert launch["status"] == outer.OUTER_LAUNCH_STATUS
    for field in (
        "producer_preflight",
        "source_binding",
        "producer_binary",
        "planned_capture_dir",
        "planned_output_authority",
        "workers",
        "execution_policy",
        "replay_guard",
    ):
        assert field in launch
    assert launch["execution_policy"]["exact_catalog_admission_scans"] == 1
    assert launch["execution_policy"]["metal_or_gpu_allowed"] is False
    assert launch["replay_guard"] == {"capture_dir_unique": True, "one_child_maximum": True}
    dynamic = json.loads(
        (capture / outer.INNER_DIRNAME / outer.DYNAMIC_AUTHORITY_FILENAME).read_text()
    )
    assert "capture_dir" not in dynamic
    assert "workers" not in dynamic
    assert dynamic["cpu_outer_capture"]["capture_dir"] == str(
        capture / outer.INNER_DIRNAME
    )
    assert dynamic["cpu_outer_capture"]["workers"] == 2
    terminal_path = capture / outer.OUTER_TERMINAL_FILENAME
    assert terminal_path.is_file()
    verify(json.loads(terminal_path.read_text()), label="outer terminal")
    replay = outer.run_attempt(attempt)
    assert replay == terminal
    assert counter.read_text() == "1"


def test_versioned_admission_pointer_reseal_keeps_exact_immutable_lineage(tmp_path: Path) -> None:
    config, binary = _fixture(tmp_path)
    counter = tmp_path / "counter.txt"
    _write_fake_producer(binary, counter=counter)
    producer_preflight = json.loads(config.producer_preflight.read_text())
    producer_preflight.pop("seal_sha256")
    producer_preflight["producer_binary"] = outer._file_evidence(
        binary, "producer", executable=True
    )
    _write_sealed(config.producer_preflight, producer_preflight)
    preflight_path = tmp_path / "outer-preflight.json"
    outer.write_outer_preflight(config, preflight_path)
    preflight = json.loads(preflight_path.read_text())
    historical = preflight["versioned_current_admission"]["preflight_observed"]

    # This models only a canonical current-pointer reseal: its manifest and
    # immutable receipt still bind exactly, while its raw and seal identities
    # become historical facts rather than false replay failures.
    pointer = json.loads(config.admission_current.read_text())
    pointer.pop("seal_sha256")
    pointer["versioned_current_reseal"] = "after-outer-preflight"
    _write_sealed(config.admission_current, pointer)

    capture = tmp_path / "capture"
    terminal = outer.run_attempt(
        outer.CaptureConfig(
            outer_preflight=preflight_path,
            producer_binary=binary,
            capture_dir=capture,
            replay_guard_dir=tmp_path / "replay",
            workers=1,
            timeout_seconds=30.0,
        )
    )
    assert terminal["status"] == outer.CAPTURED_STATUS
    assert terminal["terminal_current_pointer_valid"] is True
    assert counter.read_text() == "1"
    launch = json.loads((capture / outer.OUTER_LAUNCH_FILENAME).read_text())
    launch_observed = launch["versioned_current_admission"]["launch_observed"]
    terminal_observed = terminal["versioned_current_admission"]["terminal_observed"]
    assert launch_observed["path"] == historical["path"]
    assert launch_observed["sha256"] != historical["sha256"]
    assert launch_observed["document_seal_sha256"] != historical["document_seal_sha256"]
    assert terminal_observed == launch_observed
    assert launch["versioned_current_admission"]["acceptance"] == outer._versioned_current_acceptance()


def test_historical_rust_authority_uses_nested_cpu_outer_capture_shape() -> None:
    """Keep the exact reaped Rust receipt shape from regressing to a fake alias."""
    if not HISTORICAL_RUST_INNER.is_file():
        pytest.skip("historical reaped Rust authority is unavailable in this checkout")
    document = json.loads(HISTORICAL_RUST_INNER.read_text())
    verify(document, label="historical Rust L1 authority")
    assert document["schema"] == outer.DYNAMIC_AUTHORITY_SCHEMA
    assert document["status"] == outer.DYNAMIC_AUTHORITY_STATUS
    assert "capture_dir" not in document
    assert "workers" not in document
    capture = document["cpu_outer_capture"]
    assert capture["capture_dir"] == str(HISTORICAL_RUST_INNER.parent)
    assert capture["output_authority_path"] == str(HISTORICAL_RUST_INNER)
    assert capture["workers"] == 1
    assert capture["one_current_admitted_catalog_scan_performed"] is True
    assert capture["raw_bf16_or_safetensors_reopened"] is False
    assert capture["outer_terminal_receipt_written_by_parent_last"] is True
    assert set(document["source_binding"]["joint_l0_l1_assessment"]) == {
        "document_sha256",
        "document_seal_sha256",
    }


def test_zero_exit_without_dynamic_authority_is_terminal_refusal(tmp_path: Path) -> None:
    config, binary = _fixture(tmp_path, valid_child=False)
    preflight_path = tmp_path / "outer-preflight.json"
    outer.write_outer_preflight(config, preflight_path)
    terminal = outer.run_attempt(
        outer.CaptureConfig(
            outer_preflight=preflight_path,
            producer_binary=binary,
            capture_dir=tmp_path / "capture",
            replay_guard_dir=tmp_path / "replay",
            timeout_seconds=30.0,
        )
    )
    assert terminal["status"] == (
        outer.REFUSED_PREFIX + "ZERO_EXIT_WITHOUT_VALID_SEALED_DYNAMIC_AUTHORITY"
    )
    assert terminal["child"]["terminal"]["exit_code"] == 0
    assert terminal["child"]["terminal"]["reaped"] is True
