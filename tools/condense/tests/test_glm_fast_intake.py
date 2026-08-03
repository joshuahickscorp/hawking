"""Fail-closed admission tests for the speed-sensitive GLM handoff."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import glm_fast_intake as intake  # noqa: E402


REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"


def write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def source(path: Path) -> Path:
    return write_json(path, {
        "revision": REVISION,
        "admission_gates": {"body_stream": True},
        "source": {"complete_source_resident": True, "logical_bytes": 10},
    })


def lock(path: Path, authorized: bool = True) -> Path:
    return write_json(path, {"RAMANUJAN_RESEARCH_AUTHORIZED": authorized})


def artifact(path: Path) -> tuple[Path, str]:
    path.mkdir()
    index = path / "model.gravity.index.json"
    index.write_text(json.dumps({"schema": "hawking.gravity.model_index.v1"}), encoding="utf-8")
    return path, hashlib.sha256(index.read_bytes()).hexdigest()


def parity(path: Path, index_sha: str, *, status: str = "PASS") -> Path:
    return write_json(path, {
        "status": status,
        "artifact_index_sha256": index_sha,
        "source_revision": REVISION,
    })


def measurement(path: Path, index_sha: str, *, fast: bool = True, tps: float = 87.0) -> Path:
    resolved = {name: fast for name in intake.FAST_FLAGS}
    resolved["full_logits_readback"] = False
    values = [11.0] * 32
    waits = [2] * 32 if fast else [None] * 32
    return write_json(path, {
        "verify_hash": True,
        "artifact": {"index_sha256": index_sha},
        "run_configuration": {"resolved": resolved},
        "measurements": [{
            "context_tokens": 512,
            "decode_tokens": 32,
            "base_true_decode_tps": tps,
            "decode_ms_per_token_all": values,
            "output_modes": ["token_plus_topk_diagnostics"],
            "device_execution": {
                "backend": "metal",
                "device_name": "Apple Test GPU",
                "resident_state": fast,
                "command_buffer_waits_per_token_all": waits,
            },
        }],
    })


def evaluate(tmp_path: Path, target: intake.Target, **overrides: Path | None) -> dict:
    artifact_dir, index_sha = artifact(tmp_path / "artifact")
    values = {
        "artifact_dir": artifact_dir,
        "source_admission_path": source(tmp_path / "source.json"),
        "parity_path": parity(tmp_path / "parity.json", index_sha),
        "measurement_path": measurement(tmp_path / "measurement.json", index_sha),
        "ramanujan_lock_path": lock(tmp_path / "lock.json"),
    }
    values.update(overrides)
    return intake.evaluate(target=target, **values)


def strict_target() -> intake.Target:
    return intake.Target(min_decode_tps=60.0, max_decode_p99_ms=20.0, min_context_tokens=512, min_decode_tokens=32)


def test_preflight_refuses_an_undefined_godly_speed_target(tmp_path: Path) -> None:
    result = evaluate(tmp_path, intake.Target(None, None, None, 32))
    assert result["status"] == "BLOCKED"
    assert result["gates"]["TARGET_CONTRACT"]["status"] == "BLOCKED"
    assert result["gates"]["HIDE_HANDOFF"]["status"] == "BLOCKED"


def test_bound_fast_artifact_can_reach_hide_and_authorized_sandbox(tmp_path: Path) -> None:
    result = evaluate(tmp_path, strict_target())
    assert result["status"] == "PASS"
    assert result["gates"]["GPU_FAST_DECODE"]["status"] == "PASS"
    assert result["gates"]["DECODE_PERFORMANCE"]["status"] == "PASS"


def test_historical_host_state_shape_cannot_qualify_as_gpu_decode(tmp_path: Path) -> None:
    artifact_dir, index_sha = artifact(tmp_path / "artifact")
    result = intake.evaluate(
        target=strict_target(),
        artifact_dir=artifact_dir,
        source_admission_path=source(tmp_path / "source.json"),
        parity_path=parity(tmp_path / "parity.json", index_sha),
        measurement_path=measurement(tmp_path / "measurement.json", index_sha, fast=False, tps=0.15),
        ramanujan_lock_path=lock(tmp_path / "lock.json"),
    )
    assert result["status"] == "BLOCKED"
    assert result["gates"]["GPU_FAST_DECODE"]["status"] == "BLOCKED"
    assert result["gates"]["DECODE_PERFORMANCE"]["status"] == "BLOCKED"


def test_unbound_or_synthetic_parity_receipt_cannot_be_reused(tmp_path: Path) -> None:
    artifact_dir, index_sha = artifact(tmp_path / "artifact")
    result = intake.evaluate(
        target=strict_target(),
        artifact_dir=artifact_dir,
        source_admission_path=source(tmp_path / "source.json"),
        parity_path=parity(tmp_path / "parity.json", index_sha, status="PASS_SYNTHETIC_MAIN_AND_MTP_SELF_CONSISTENCY_SOURCE_PARENT_PENDING"),
        measurement_path=measurement(tmp_path / "measurement.json", index_sha),
        ramanujan_lock_path=lock(tmp_path / "lock.json"),
    )
    assert result["gates"]["ORACLE_PARITY"]["status"] == "BLOCKED"
    assert result["gates"]["HIDE_HANDOFF"]["status"] == "BLOCKED"


def test_missing_artifact_path_is_a_clean_block_not_a_preflight_crash(tmp_path: Path) -> None:
    artifact_dir, index_sha = artifact(tmp_path / "present-artifact")
    missing = tmp_path / "missing-artifact"
    result = intake.evaluate(
        target=strict_target(),
        artifact_dir=missing,
        source_admission_path=source(tmp_path / "source.json"),
        parity_path=parity(tmp_path / "parity.json", index_sha),
        measurement_path=measurement(tmp_path / "measurement.json", index_sha),
        ramanujan_lock_path=lock(tmp_path / "lock.json"),
    )
    assert artifact_dir.is_dir()
    assert result["gates"]["ARTIFACT_ASSEMBLY"]["status"] == "BLOCKED"


def test_cli_verify_demands_explicit_operator_target() -> None:
    # argparse itself accepts the flags; main is responsible for the speed
    # contract so a silent default can never sneak in.
    assert intake.main(["verify"]) == 2
