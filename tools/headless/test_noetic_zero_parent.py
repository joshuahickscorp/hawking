"""G011: production Noetic inference does not load parent weights.

Proved by observed file access (DYLD interpose of open/openat) on a
COMPLETE native decode — not a truncated prefix — with a negative control
the detector must catch. See noetic_zero_parent.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noetic_zero_parent import (  # noqa: E402
    RECEIPT,
    REPO,
    classify_path,
    parent_dir,
    parse_open_log,
    run_and_write,
)

PARENT = parent_dir()


@pytest.fixture(scope="session")
def receipt() -> dict:
    rec = run_and_write()
    assert RECEIPT.is_file(), f"harness did not write {RECEIPT}"
    return rec


def _skip_if_metal_refused(receipt: dict) -> None:
    live = (receipt.get("production_run") or {}).get("live_native_decode") or {}
    if live.get("metal_refused"):
        pytest.skip(
            "native decode INCONCLUSIVE: no Metal-capable GPU on this host "
            f"(exit={live.get('exit_code')})"
        )


def test_classifier_distinguishes_weight_tokenizer_config():
    weight = PARENT / "model-00001-of-00018.safetensors"
    tok = PARENT / "tokenizer.json"
    cfg = PARENT / "config.json"
    idx = PARENT / "model.safetensors.index.json"
    artifact = Path.home() / "models" / "qwen38-gravity-uniform-q4-v1" / "manifest.json"
    assert classify_path(str(weight), parent=PARENT) == "parent_weight"
    assert classify_path(str(tok), parent=PARENT) == "parent_tokenizer"
    assert classify_path(str(cfg), parent=PARENT) == "parent_config"
    assert classify_path(str(idx), parent=PARENT) == "parent_config"
    assert classify_path(str(artifact), parent=PARENT) == "not_parent"
    assert classify_path("/etc/hosts", parent=PARENT) == "not_parent"


def test_detector_flags_a_parent_safetensor_in_a_synthetic_log(tmp_path: Path):
    """The detector, given a log that DID touch a parent shard, must say so.

    This is the cheap form of the negative-control obligation: a detector that
    cannot classify a safetensors open as a weight cannot certify the clean
    case either.
    """
    log = tmp_path / "control.log"
    log.write_text(
        "\n".join(
            [
                f"open\t{PARENT / 'tokenizer.json'}",
                f"open\t{PARENT / 'model.safetensors.index.json'}",
                f"open\t{PARENT / 'model-00001-of-00018.safetensors'}",
                "open\t/etc/hosts",
            ]
        )
        + "\n"
    )
    obs = parse_open_log(log, parent=PARENT, cwd=tmp_path)
    assert obs["parent_weight"], "detector missed a parent .safetensors open"
    assert any(p.endswith(".safetensors") for p in obs["parent_weight"])
    assert obs["parent_tokenizer"]
    assert obs["parent_config"]
    assert "/etc/hosts" not in obs["parent_weight"]


def test_receipt_written(receipt: dict):
    _skip_if_metal_refused(receipt)
    assert RECEIPT.is_file()
    assert receipt["schema"] == "hawking.headless.noetic_zero_parent.v1"
    assert receipt["verdict"] == "PASS", receipt.get("pass_rule")


def test_decode_binary_is_from_this_repo(receipt: dict):
    binary = Path(receipt["decode_binary"]).resolve()
    repo = REPO.resolve()
    assert str(binary).startswith(str(repo)), binary
    assert "hawking-copy" not in str(binary)
    ident = receipt["decode_binary_identity"]
    assert ident["from_this_repo"] is True
    assert ident["not_vestigial_hawking_copy"] is True


def test_decode_ran_to_completion_and_emitted_a_token(receipt: dict):
    _skip_if_metal_refused(receipt)
    live = receipt["production_run"]["live_native_decode"]
    assert live["complete"] is True, (
        "native decode did not run to completion; a truncated prefix is "
        "INCONCLUSIVE, not PASS. "
        f"exit={live.get('exit_code')} metal_refused={live.get('metal_refused')} "
        f"timed_out={live.get('timed_out')} tokens={live.get('new_token_ids')} "
        f"stderr={live.get('stderr_head', '')[-500:]}"
    )
    assert live["exit_code"] == 0
    assert live["timed_out"] is False
    assert live["metal_refused"] is False
    assert live["n_new_tokens"] >= 1, live.get("new_token_ids")
    assert live["generated_text"] is not None
    assert live["dylib_loaded"] is True


def test_observed_the_whole_process_including_catalog_reads(receipt: dict):
    _skip_if_metal_refused(receipt)
    live = receipt["production_run"]["live_native_decode"]
    obs = live["observation"]
    assert obs["n_events"] >= 755, (
        f"only {obs['n_events']} file-open events; a complete catalog load "
        "is 755 tensor reads plus tokenizer/manifest/dylibs"
    )
    assert live["n_artifact_tensor_opens"] >= 755, live.get("n_artifact_tensor_opens")
    assert live["saw_catalog_count"] is True
    assert live["catalog_count"] == 755


def test_production_run_opens_no_parent_weights(receipt: dict):
    _skip_if_metal_refused(receipt)
    prod = receipt["production_run"]
    assert prod["parent_weight_paths"] == []
    assert prod["n_parent_weight_opens"] == 0
    assert prod["verdict"] == "PASS"
    live = prod["live_native_decode"]
    assert live["n_parent_weight_opens"] == 0
    assert live["observation"]["parent_weight"] == []


def test_negative_control_caught_a_parent_weight_open(receipt: dict):
    ctrl = receipt["negative_control"]
    assert ctrl["detector_caught"] is True, (
        "detector missed the composition teacher-scoring path; it cannot "
        "certify the clean case either"
    )
    assert ctrl["parent_weight_opens"], ctrl
    assert any(p.endswith(".safetensors") for p in ctrl["parent_weight_opens"])
    assert ctrl["verdict"] == "PASS"
    assert ctrl["did_not_load_27b"] is True


def test_tokenizer_dependency_is_not_a_weight_dependency(receipt: dict):
    prod = receipt["production_run"]
    dist = receipt["distinctions"]
    assert prod["parent_tokenizer_paths"], (
        "expected the parent tokenizer.json open (prior work; tokenizer "
        "dependency, not a weight dependency)"
    )
    assert all(p.endswith("tokenizer.json") for p in prod["parent_tokenizer_paths"])
    assert dist["parent_weights_at_inference"]["observed"] is False
    assert dist["parent_config_or_tokenizer"]["observed"] is True
    assert dist["parent_at_compile_time"]["observed"] is False
    found = " ".join(prod["found"]).lower()
    assert "tokenizer" in found
    assert "violation" not in found
