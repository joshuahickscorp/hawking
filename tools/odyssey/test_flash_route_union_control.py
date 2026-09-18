from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.odyssey import flash_route_union_control as control


PROMPT = list(control.PROMPT_IDS)
CHAIN = [*PROMPT, 271, 248045]
ATTENTION_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47}


def _seal(document: dict) -> dict:
    document["seal_sha256"] = control._compact_utf8_sorted_seal(document)
    return document


def _reseal(document: dict) -> dict:
    return _seal({key: value for key, value in document.items() if key != "seal_sha256"})


def _terminal_checks(tokens: list[int]) -> list[dict]:
    prompt_length = len(PROMPT)
    return [
        {
            "generation_index": index,
            "input_state_index": prompt_length - 1 + index,
            "expected_token_id": token,
            "predicted_token_id": token,
            "accepted": True,
        }
        for index, token in enumerate(tokens[prompt_length:])
    ]


def _token_major_segments(tokens: list[int], *, expert_bank_mode: str) -> list[dict]:
    segments: list[dict] = []
    for layer in range(48):
        segments.append({
            "layers": [layer, layer],
            "species": "full_attention" if layer in ATTENTION_LAYERS else "linear_attention",
            "expert_bank_mode": expert_bank_mode,
            "steps": [
                {
                    "step": step,
                    "token_id": token,
                    "route_ids": list(range(10)),
                    "final_state_sha256": f"{layer:02x}{step:02x}" + "a" * 60,
                }
                for step, token in enumerate(tokens)
            ],
        })
    return segments


def _doc(*, compact: bool, source_bytes: int, token_major: bool = True, tokens: list[int] | None = None) -> dict:
    token_ids = list(CHAIN if tokens is None else tokens)
    execution = {
        "process_boundary": "one native process",
        "source_reset_or_reprefill": False,
        "source_payload_bytes_read": source_bytes,
        "expert_bank_mode": "route_union_compact_teacher_bound" if compact else "dense",
    }
    if token_major:
        execution["token_major_resident_banks"] = {
            "status": control.TOKEN_MAJOR_STATUS,
            "resident_layers": 48,
            "expected_layers": 48,
            "cross_layer_activation_handoff": "device buffers",
            "source_reset_or_reprefill": False,
        }
    document = {
        "schema": control.SESSION_SCHEMA,
        "status": control.SESSION_STATUS,
        "model": control.REPO_ID,
        "pinned_revision": control.PINNED_REVISION,
        "token_ids": token_ids,
        "accepted_generation_tokens": len(token_ids) - len(PROMPT),
        "execution": execution,
        "terminal": {"reference_checks": _terminal_checks(token_ids)},
    }
    if token_major:
        document["segments"] = _token_major_segments(
            token_ids,
            expert_bank_mode="route_union_compact_teacher_bound" if compact else "dense",
        )
    return _seal(document)


def _resident_candidate() -> dict:
    candidate = _doc(compact=True, source_bytes=100)
    candidate["execution"].update({
        "linear_compact_bank_residency": {
            "status": "ALL_LINEAR_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE",
            "immutable_weight_ownership": "device_only_after_source_upload",
            "retained_linear_layers": 36,
            "retained_linear_source_weight_bytes": 0,
            "retained_linear_device_weight_bytes": 24_230_661_888,
            "observed_process_rss_bytes": 27_905_114_112,
        },
        "full_attention_compact_bank_residency": {
            "status": "ALL_FULL_ATTENTION_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE",
            "resident_full_attention_layers": 12,
            "expected_full_attention_layers": 12,
            "device_weight_bytes": 7_678_144_512,
            "host_source_weights_retained_during_token_loop": False,
            "process_lifetime_all_banks_retained": True,
        },
    })
    for segment in candidate["segments"]:
        if segment["species"] == "full_attention":
            segment.update({
                "immutable_weight_ownership": "device_only_after_source_upload",
                "host_source_weights_retained_during_token_loop": False,
            })
    return _reseal(candidate)


def test_compact_control_requires_lower_bytes_and_selected_terminal_chain():
    teacher = _doc(compact=False, source_bytes=1_000)
    candidate = _doc(compact=True, source_bytes=100)
    control._verify_candidate(candidate, teacher)

    candidate["execution"]["source_payload_bytes_read"] = 1_000
    candidate = _reseal(candidate)
    with pytest.raises(ValueError, match="reduce source payload"):
        control._verify_candidate(candidate, teacher)


def test_route_receipt_loader_rejects_duplicate_raw_object_keys(tmp_path):
    path = tmp_path / "ambiguous.json"
    path.write_text('{"status":"one","status":"two"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        control._load(path)


def test_teacher_rejects_segmented_historical_control_even_if_terminal_tokens_pass():
    segmented = _doc(compact=False, source_bytes=1_000, token_major=False)
    with pytest.raises(ValueError, match="segmented"):
        control._verify_teacher(segmented)


def test_teacher_rejects_compact_candidate_as_its_own_dense_route_authority():
    compact = _doc(compact=True, source_bytes=100)
    with pytest.raises(ValueError, match="exact dense teacher"):
        control._verify_teacher(compact)


def test_teacher_source_identity_must_match_the_owner_authorized_v2_contract():
    identity = {
        "schema": "hawking.flash.source_input_identity.v1",
        "model_root": "/canonical/modellake/specimen",
        "config": {"file": "config.json", "sha256": "a" * 64, "bytes": 1},
    }
    teacher = _doc(compact=False, source_bytes=1_000)
    teacher["root"] = identity["model_root"]
    teacher["source_input_identity"] = json.loads(json.dumps(identity))
    teacher = _reseal(teacher)
    control._verify_teacher(teacher, source_input_identity=identity)

    teacher["source_input_identity"]["config"]["sha256"] = "b" * 64
    teacher = _reseal(teacher)
    with pytest.raises(ValueError, match="source input identity"):
        control._verify_teacher(teacher, source_input_identity=identity)


def test_candidate_cannot_silently_substitute_a_different_native_chain():
    teacher = _doc(compact=False, source_bytes=1_000)
    candidate = _doc(compact=True, source_bytes=100, token_major=False, tokens=[*PROMPT, 2972, 96125])
    with pytest.raises(ValueError, match="token sequence drifted"):
        control._verify_candidate(candidate, teacher)


def test_reference_contract_must_match_the_explicit_teacher_chain():
    teacher = _doc(compact=False, source_bytes=1_000)
    chain = control._verify_teacher(teacher)
    control._verify_reference_matches_teacher(
        {"prompt_token_ids": PROMPT, "generated_token_ids": [271, 248045]}, chain
    )
    with pytest.raises(ValueError, match="independently sealed"):
        control._verify_reference_matches_teacher(
            {"prompt_token_ids": PROMPT, "generated_token_ids": [2972, 96125]}, chain
        )


def test_device_only_resident_control_requires_explicit_single_owner_accounting():
    teacher = _doc(compact=False, source_bytes=1_000)
    candidate = _resident_candidate()
    control._verify_candidate(
        candidate, teacher, device_only_resident=True, token_major_resident=True
    )

    candidate["execution"]["linear_compact_bank_residency"]["retained_linear_source_weight_bytes"] = 1
    candidate = _reseal(candidate)
    with pytest.raises(ValueError, match="source-side"):
        control._verify_candidate(
            candidate, teacher, device_only_resident=True, token_major_resident=True
        )

    candidate["execution"]["linear_compact_bank_residency"]["retained_linear_source_weight_bytes"] = 0
    attention_segment = next(
        segment for segment in candidate["segments"] if segment["species"] == "full_attention"
    )
    attention_segment["host_source_weights_retained_during_token_loop"] = True
    candidate = _reseal(candidate)
    with pytest.raises(ValueError, match="full-attention"):
        control._verify_candidate(
            candidate, teacher, device_only_resident=True, token_major_resident=True
        )


def test_token_major_control_requires_all_layers_and_device_handoffs():
    teacher = _doc(compact=False, source_bytes=1_000)
    candidate = _resident_candidate()
    control._verify_candidate(
        candidate,
        teacher,
        device_only_resident=True,
        token_major_resident=True,
    )

    candidate["execution"]["token_major_resident_banks"]["resident_layers"] = 47
    candidate = _reseal(candidate)
    with pytest.raises(ValueError, match="complete token-major state ownership"):
        control._verify_candidate(
            candidate,
            teacher,
            device_only_resident=True,
            token_major_resident=True,
        )


@pytest.mark.parametrize("field, replacement", [
    ("route_ids", list(range(1, 11))),
    ("final_state_sha256", "b" * 64),
])
def test_compact_candidate_must_match_every_dense_teacher_route_and_state_hash(
    field: str, replacement: object
):
    teacher = _doc(compact=False, source_bytes=1_000)
    candidate = _doc(compact=True, source_bytes=100)
    candidate["segments"][17]["steps"][3][field] = replacement
    candidate = _reseal(candidate)
    with pytest.raises(ValueError, match="route/state transcript drifted"):
        control._verify_candidate(candidate, teacher)


def test_dry_run_without_authoritative_inputs_is_withheld_and_does_not_spawn(capsys):
    assert control.main(["--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "DRY_RUN_WITHHELD_MISSING_AUTHORITATIVE_INPUTS"
    assert payload["required"] == [
        "--teacher", "--reference-contract", "--owner-authorization", "--out"
    ]
    assert "No model, GPU, native executor" in payload["claim_boundary"]


def test_existing_output_is_rejected_before_source_or_model_access(tmp_path):
    out = tmp_path / "already-exists.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        control.main([
            "--teacher", str(tmp_path / "teacher.json"),
            "--reference-contract", str(tmp_path / "reference.json"),
            "--owner-authorization", str(tmp_path / "owner-auth.json"),
            "--out", str(out),
            "--retain-linear-banks",
            "--device-only-compact-banks",
            "--token-major-resident-banks",
        ])


def test_existing_session_artifacts_are_rejected_before_source_or_model_access(tmp_path):
    out = tmp_path / "fresh.json"
    (tmp_path / "fresh.session_artifacts").mkdir()
    with pytest.raises(ValueError, match="session-artifact"):
        control.main([
            "--teacher", str(tmp_path / "teacher.json"),
            "--reference-contract", str(tmp_path / "reference.json"),
            "--owner-authorization", str(tmp_path / "owner-auth.json"),
            "--out", str(out),
            "--retain-linear-banks",
            "--device-only-compact-banks",
            "--token-major-resident-banks",
        ])


def test_output_nested_in_teacher_session_artifacts_is_rejected_before_source_or_model_access(
    tmp_path,
):
    teacher = tmp_path / "teacher.json"
    teacher_artifacts = tmp_path / "teacher.session_artifacts"
    teacher_artifacts.mkdir()
    out = teacher_artifacts / "candidate.json"

    with pytest.raises(ValueError, match="teacher receipt/session-artifact namespace"):
        control.main([
            "--teacher", str(teacher),
            "--reference-contract", str(tmp_path / "reference.json"),
            "--owner-authorization", str(tmp_path / "owner-auth.json"),
            "--out", str(out),
            "--retain-linear-banks",
            "--device-only-compact-banks",
            "--token-major-resident-banks",
        ])


def test_dangling_output_symlink_is_rejected_before_source_or_model_access(tmp_path):
    out = tmp_path / "dangling.json"
    out.symlink_to(tmp_path / "missing-target.json")

    with pytest.raises(ValueError, match="already exists"):
        control.main([
            "--teacher", str(tmp_path / "teacher.json"),
            "--reference-contract", str(tmp_path / "reference.json"),
            "--owner-authorization", str(tmp_path / "owner-auth.json"),
            "--out", str(out),
            "--retain-linear-banks",
            "--device-only-compact-banks",
            "--token-major-resident-banks",
        ])


def test_non_dry_route_control_requires_the_exact_all_48_device_only_path(tmp_path):
    with pytest.raises(ValueError, match="all-48 device-only token-major path"):
        control.main([
            "--teacher", str(tmp_path / "teacher.json"),
            "--reference-contract", str(tmp_path / "reference.json"),
            "--owner-authorization", str(tmp_path / "owner-auth.json"),
            "--out", str(tmp_path / "candidate.json"),
        ])


def test_provenance_write_never_overwrites_a_racing_existing_artifact(tmp_path):
    teacher_path = tmp_path / "teacher.json"
    candidate_path = tmp_path / "candidate.json"
    teacher_path.write_text("{}", encoding="utf-8")
    candidate_path.write_text("{}", encoding="utf-8")
    provenance = tmp_path / "candidate.route_union_provenance.json"
    provenance.write_text("preserve me", encoding="utf-8")

    with pytest.raises(ValueError, match="appeared during execution"):
        control._write_run_provenance(
            provenance,
            command=["/tmp/flash-session"],
            teacher_path=teacher_path,
            teacher={"seal_sha256": "a" * 64},
            candidate_path=candidate_path,
            candidate={"seal_sha256": "b" * 64},
            reference={"mode": "withheld"},
            build={"observed": True},
            lane_preflight={"clean": True},
        )
    assert provenance.read_text(encoding="utf-8") == "preserve me"


def test_route_command_invokes_an_explicit_prebuilt_binary_not_cargo():
    command = control._build_command(
        native_binary=Path("/tmp/flash-session"),
        model_root=Path("/tmp/specimen"),
        teacher_path=Path("/tmp/teacher.json"),
        out=Path("/tmp/candidate.json"),
        chain=control.AcceptedChain(tuple(CHAIN), len(PROMPT)),
        retain_linear_banks=True,
        device_only_compact_banks=True,
        token_major_resident_banks=True,
    )
    assert command[0] == "/tmp/flash-session"
    assert "cargo" not in command
    assert "--wrapper-admitted-source-control" in command
    assert command[command.index("--supervising-launcher-pid") + 1] == str(control.os.getpid())
    assert command[command.index("--token-ids") + 1] == ",".join(map(str, CHAIN))


def test_lane_preflight_ignores_its_own_process_and_refuses_other_flash_worker(monkeypatch):
    own = str(control.os.getpid())
    monkeypatch.setattr(
        control.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=(
                f"{own} S python3 tools/odyssey/flash_route_union_control.py\n"
                "7001 S flash_stateful_complete_token_session --root /specimen\n"
            )
        ),
    )
    lane = control._protected_native_lane()
    assert lane["clean"] is False
    assert len(lane["matches"]) == 1
    assert "flash_stateful_complete_token_session" in lane["matches"][0]["command"]
