from __future__ import annotations

from hcli.agentos.protected_accelerator_benchmark import (
    _aggregate,
    _request_record,
    _required_metrics,
)


def test_packed_active_byte_metrics_survive_provider_normalization():
    raw = {
        "hawking": {
            "generated_tokens": 4,
            "new_token_ids": [10, 11, 12, 13],
            "fallbacks": 0,
            "prompt_tokens": 7,
            "resident_health": {"pid": 42},
            "native_metrics": {
                "gpu_ns": 80,
                "gpu_ns_per_generated_token": 20,
                "dispatches": 40,
                "dispatches_per_generated_token": 10,
                "active_weight_bytes_per_generated_token": 400,
                "active_bytes_scope": "packed_weight_payloads_per_complete_request_generated_token",
                "resident_weight_bytes": 1000,
                "workspace_resident_bytes": 200,
                "actual_read_bytes_per_token": None,
                "transient_bytes_per_token": None,
                "capability": {"complete_token_accounting": True},
            },
        }
    }
    row = _request_record(raw, elapsed_ns=100, index=1, phase="measure")
    summary = _aggregate([row])
    required = _required_metrics([row])

    assert row["active_bytes_per_token"] == 400
    assert row["active_bytes_source"] == "native_active_weight_bytes_per_generated_token"
    assert row["resident_weight_bytes"] == 1000
    assert row["workspace_resident_bytes"] == 200
    assert summary["active_bytes_per_token"]["median"] == 400
    assert required["optional_physical_fields"]["active_bytes_per_token"] is True
    assert required["optional_physical_fields"]["actual_read_bytes_per_token"] is False
