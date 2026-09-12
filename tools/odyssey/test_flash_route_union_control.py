from __future__ import annotations

from tools.odyssey.flash_route_union_control import TOKENS, _verify_candidate


def _doc(*, compact: bool, source_bytes: int) -> dict:
    return {
        "status": "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE",
        "model": "Qwen/Qwen3.8-Flash-Next",
        "pinned_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
        "token_ids": TOKENS,
        "execution": {
            "source_reset_or_reprefill": False,
            "source_payload_bytes_read": source_bytes,
            "expert_bank_mode": "route_union_compact_teacher_bound" if compact else "dense",
        },
        "terminal": {"reference_checks": [
            {"expected_token_id": 2972, "predicted_token_id": 2972, "accepted": True},
            {"expected_token_id": 96125, "predicted_token_id": 96125, "accepted": True},
        ]},
    }


def test_compact_control_requires_lower_bytes_and_exact_terminal_chain():
    teacher = _doc(compact=False, source_bytes=1000)
    candidate = _doc(compact=True, source_bytes=100)
    _verify_candidate(candidate, teacher)

    candidate["execution"]["source_payload_bytes_read"] = 1000
    try:
        _verify_candidate(candidate, teacher)
    except ValueError as exc:
        assert "reduce source payload" in str(exc)
    else:
        raise AssertionError("a compact label without reduced reads must fail")


def test_device_only_resident_control_requires_explicit_single_owner_accounting():
    teacher = _doc(compact=False, source_bytes=1000)
    candidate = _doc(compact=True, source_bytes=100)
    candidate["execution"]["linear_compact_bank_residency"] = {
        "status": "PARTIAL_PROCESS_LIFETIME_DEVICE_ONLY_LINEAR_BANKS_RETAINED",
        "immutable_weight_ownership": "device_only_after_source_upload",
        "retained_linear_layers": 36,
        "retained_linear_source_weight_bytes": 0,
        "retained_linear_device_weight_bytes": 24230661888,
        "observed_process_rss_bytes": 27905114112,
    }
    candidate["execution"]["full_attention_compact_bank_residency"] = {
        "status": "PER_LAYER_DEVICE_ONLY_FULL_ATTENTION_BANKS_REUSED_ACROSS_SESSION_TOKENS",
        "resident_full_attention_layers": 1,
        "expected_full_attention_layers": 1,
        "device_weight_bytes": 123456,
        "host_source_weights_retained_during_token_loop": False,
        "process_lifetime_all_banks_retained": False,
    }
    candidate["segments"] = [{
        "species": "full_attention",
        "immutable_weight_ownership": "device_only_after_source_upload",
        "host_source_weights_retained_during_token_loop": False,
    }]
    _verify_candidate(candidate, teacher, device_only_resident=True)

    candidate["execution"]["linear_compact_bank_residency"]["retained_linear_source_weight_bytes"] = 1
    try:
        _verify_candidate(candidate, teacher, device_only_resident=True)
    except ValueError as exc:
        assert "source-side" in str(exc)
    else:
        raise AssertionError("host-side retained weights must fail the device-only control")

    candidate["execution"]["linear_compact_bank_residency"]["retained_linear_source_weight_bytes"] = 0
    candidate["segments"][0]["host_source_weights_retained_during_token_loop"] = True
    try:
        _verify_candidate(candidate, teacher, device_only_resident=True)
    except ValueError as exc:
        assert "full-attention" in str(exc)
    else:
        raise AssertionError("full-attention host retention must fail the device-only control")

    candidate["segments"][0]["host_source_weights_retained_during_token_loop"] = False
    candidate["execution"]["full_attention_compact_bank_residency"]["device_weight_bytes"] = 0
    try:
        _verify_candidate(candidate, teacher, device_only_resident=True)
    except ValueError as exc:
        assert "device weight accounting" in str(exc)
    else:
        raise AssertionError("unbilled full-attention resident weights must fail the control")


def test_token_major_control_requires_all_layers_and_device_handoffs():
    teacher = _doc(compact=False, source_bytes=1000)
    candidate = _doc(compact=True, source_bytes=100)
    candidate["execution"].update({
        "linear_compact_bank_residency": {
            "status": "ALL_LINEAR_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE",
            "immutable_weight_ownership": "device_only_after_source_upload",
            "retained_linear_layers": 36,
            "retained_linear_source_weight_bytes": 0,
            "retained_linear_device_weight_bytes": 24230661888,
            "observed_process_rss_bytes": 27905114112,
        },
        "full_attention_compact_bank_residency": {
            "status": "ALL_FULL_ATTENTION_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE",
            "resident_full_attention_layers": 12,
            "expected_full_attention_layers": 12,
            "device_weight_bytes": 7678144512,
            "host_source_weights_retained_during_token_loop": False,
            "process_lifetime_all_banks_retained": True,
        },
        "token_major_resident_banks": {
            "status": "ALL_48_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE",
            "resident_layers": 48,
            "expected_layers": 48,
            "cross_layer_activation_handoff": "device buffers",
            "source_reset_or_reprefill": False,
        },
    })
    candidate["segments"] = [
        {
            "layers": [layer, layer],
            "species": "linear_attention",
            "steps": [
                {"step": step, "token_id": token}
                for step, token in enumerate(TOKENS)
            ],
        }
        for layer in range(48)
    ]
    # Mark the 12 attention layers exactly as the production receipt does.
    for layer in (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47):
        candidate["segments"][layer].update({
            "species": "full_attention",
            "immutable_weight_ownership": "device_only_after_source_upload",
            "host_source_weights_retained_during_token_loop": False,
        })
    _verify_candidate(
        candidate,
        teacher,
        device_only_resident=True,
        token_major_resident=True,
    )

    candidate["execution"]["token_major_resident_banks"]["resident_layers"] = 47
    try:
        _verify_candidate(
            candidate,
            teacher,
            device_only_resident=True,
            token_major_resident=True,
        )
    except ValueError as exc:
        assert "48-layer accounting" in str(exc)
    else:
        raise AssertionError("partial bank inventory must fail token-major verification")
