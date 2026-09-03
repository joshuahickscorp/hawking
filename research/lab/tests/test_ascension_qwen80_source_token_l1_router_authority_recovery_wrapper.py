from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.operators import ascension_qwen80_source_token_l1_router_authority_recovery_wrapper as recovery
from lab.receipts import verify


COMPLETE_RUNTIME = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/"
    "physical/qwen80/complete-runtime"
)
HISTORICAL_CAPTURE = COMPLETE_RUNTIME / "QWEN80_SOURCE_TOKEN_L1_ROUTE_AUTHORITY_CPU_SCAN_20260809T130548Z"
HISTORICAL_OUTER_PREFLIGHT = (
    COMPLETE_RUNTIME
    / "QWEN80_SOURCE_TOKEN_L1_ROUTE_AUTHORITY_OUTER_PREFLIGHT_20260809T130432Z"
    / "l1-router-authority-scan-outer-preflight.json"
)
HISTORICAL_INNER = HISTORICAL_CAPTURE / "inner" / "l1-source-token-route-authority.json"
HISTORICAL_BINARY = (
    Path("/Users/scammermike/Downloads/hawking/workspace/ops/build/rust/debug/examples")
    / "ascension_qwen80_source_token_l1_all_ten_route_authority_cpu"
)


def _config(out: Path) -> recovery.RecoveryConfig:
    return recovery.RecoveryConfig(
        outer_preflight=HISTORICAL_OUTER_PREFLIGHT,
        outer_launch_authority=HISTORICAL_CAPTURE / "outer-launch-authority.json",
        outer_terminal=HISTORICAL_CAPTURE / "outer-terminal-receipt.json",
        child_record=HISTORICAL_CAPTURE / "child.json",
        inner_authority=HISTORICAL_INNER,
        producer_binary=HISTORICAL_BINARY,
        out=out,
    )


def _require_historical_chain() -> None:
    required = (
        HISTORICAL_OUTER_PREFLIGHT,
        HISTORICAL_CAPTURE / "outer-launch-authority.json",
        HISTORICAL_CAPTURE / "outer-terminal-receipt.json",
        HISTORICAL_CAPTURE / "child.json",
        HISTORICAL_INNER,
        HISTORICAL_BINARY,
    )
    if not all(path.is_file() for path in required):
        pytest.skip("historical reaped L1 route-authority chain is unavailable")


def test_historical_recovery_wrapper_binds_valid_inner_without_relabeling_outer(
    tmp_path: Path,
) -> None:
    _require_historical_chain()
    document = recovery.build_recovery_wrapper(_config(tmp_path / "recovery.json"))
    verify(document, label="historical recovery wrapper")
    assert document["schema"] == recovery.RECOVERY_SCHEMA
    assert document["status"] == recovery.RECOVERY_STATUS
    assert document["historical_inner_authority"]["seal_sha256"] == (
        "1be012d736659b4c0d761c6643be590e43dde495a2d05a3ad715928bac642722"
    )
    assert document["historical_chain"]["refused_outer_terminal"]["status"] == (
        recovery.HISTORICAL_REFUSED_STATUS
    )
    assert document["historical_chain"]["refused_outer_terminal"]["capture_error"] == (
        recovery.HISTORICAL_CAPTURE_ERROR
    )
    canonicalization = document["canonicalization"]
    assert canonicalization["historical_outer_remains_refused"] is True
    assert canonicalization["historical_outer_status_relabelled"] is False
    assert canonicalization["no_new_scan_or_child"] is True
    assert document["downstream_authority"]["consume_historical_inner_directly"] is True
    assert document["downstream_authority"]["recovery_wrapper_is_not_a_dynamic_route_authority_substitute"] is True


def test_recovery_wrapper_is_create_new_and_preserves_the_raw_inner_authority(
    tmp_path: Path,
) -> None:
    _require_historical_chain()
    out = tmp_path / "recovery.json"
    document = recovery.write_recovery_wrapper(_config(out))
    persisted = json.loads(out.read_text())
    verify(persisted, label="persisted historical recovery wrapper")
    assert persisted == document
    with pytest.raises(recovery.RecoveryCanonicalizationError, match="new absolute file"):
        recovery.write_recovery_wrapper(_config(out))
    assert json.loads(HISTORICAL_INNER.read_text())["seal_sha256"] == document[
        "historical_inner_authority"
    ]["seal_sha256"]
