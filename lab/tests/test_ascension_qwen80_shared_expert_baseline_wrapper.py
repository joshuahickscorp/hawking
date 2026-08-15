from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.receipts import verify
from lab.operators import ascension_qwen80_shared_expert_baseline_wrapper as wrapper


ROOT = Path("/Users/scammermike/Downloads/hawking")
RUNTIME = ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-runtime"
GRAVITY = ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-gravity"
INNER = RUNTIME / "QWEN80_SHARED_EXPERT_CPU_CAPTURE_20260808T000002Z/receipt.json"
MANIFEST = GRAVITY / "QWEN80_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
ADMISSION = GRAVITY / "QWEN80_COMPLETE_BINARY_GRAVITY_ADMISSION_CURRENT.json"


def test_real_cpu_baseline_builds_a_sealed_component_only_wrapper(tmp_path: Path) -> None:
    output = tmp_path / "shared-expert-baseline-wrapper.json"
    body = wrapper.build_wrapper(
        baseline_receipt=INNER,
        manifest=MANIFEST,
        admission_current=ADMISSION,
    )
    sealed = wrapper._write_new_sealed(output, body)
    verify(sealed, label="shared-expert baseline wrapper")
    assert sealed["schema"] == wrapper.WRAPPER_SCHEMA
    assert sealed["status"] == wrapper.WRAPPER_STATUS
    assert sealed["cpu_inner_receipt"]["path"] == str(INNER)
    assert sealed["source_binding"]["manifest"]["path"] == str(MANIFEST)
    assert sealed["source_binding"]["admission_current"]["path"] == str(ADMISSION)
    assert sealed["claim_boundary"]["does_not_perform_metal_device_execution"] is True


def test_unsigned_inner_drift_fails_closed_before_wrapper_write(tmp_path: Path) -> None:
    drifted = tmp_path / "drifted-inner.json"
    document = json.loads(INNER.read_text(encoding="utf-8"))
    document["mode"] = "metal"
    drifted.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="inner mode"):
        wrapper.build_wrapper(
            baseline_receipt=drifted,
            manifest=MANIFEST,
            admission_current=ADMISSION,
        )


def test_immutable_output_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "already-present.json"
    output.write_text("{}\n", encoding="utf-8")
    body = wrapper.build_wrapper(
        baseline_receipt=INNER,
        manifest=MANIFEST,
        admission_current=ADMISSION,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        wrapper._write_new_sealed(output, body)
