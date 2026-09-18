from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.odyssey import flash_source_boundary_native as native


def _sealed(path: Path, document: dict) -> None:
    result = dict(document)
    result["seal_sha256"] = native.gate._compact_utf8_sorted_seal(result)
    path.write_text(json.dumps(result), encoding="utf-8")


def _post_ple_payload(path: Path, *, elements: int = 10_240) -> dict:
    raw = b"\0" * (elements * 4)
    path.write_bytes(raw)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dtype": "F32_LE",
        "elements": elements,
        "bytes": len(raw),
    }


def _layer0_trace_records(tmp_path: Path) -> list[dict]:
    records: list[dict] = []
    for declaration in native.preflight._required_layer0_trace():
        records.append({
            **declaration,
            "original_dtype": "mlx.core.bfloat16",
            "payload": _post_ple_payload(
                tmp_path / f"layer0-trace-{declaration['ordinal']:02d}.f32",
                elements=native.preflight._shape_elements(declaration["shape"]),
            ),
        })
    return records


def _source_capture(path: Path, *, post_ple: dict | None, shape: list[int]) -> None:
    seams = [
        {"ordinal": ordinal, "layer": None, "stage": "unused", "shape": []}
        for ordinal in range(14)
    ]
    seams[3] = {
        "ordinal": 3,
        "layer": 1,
        "stage": "ple_additive_output",
        "shape": shape,
    }
    if post_ple is not None:
        seams[3]["payload"] = post_ple
    _sealed(path, {
        "schema": native.source_capture.SCHEMA,
        "status": native.source_capture.STATUS,
        "source_reference": {
            "sha256": "a" * 64,
            "owner_authorization": {"sha256": "b" * 64},
        },
        "preflight": {"sha256": "c" * 64},
        "source_payloads": {
            "layer0_trace_contract_version": 5,
            "seams": seams,
            "layer0_trace": _layer0_trace_records(path.parent),
        },
        "full_body_executed": False,
        "native_body_executed": False,
        "parity_compared": False,
    })


def test_native_run_refuses_at_owner_gate_before_outputs_or_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "reference.json"
    authorization = tmp_path / "authorization.json"
    preflight = tmp_path / "preflight.json"
    source_capture = tmp_path / "source-capture.json"
    for path in (reference, authorization, preflight, source_capture):
        path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        native.gate,
        "load_admitted_reference_for_boundary",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("owner refused")),
    )
    monkeypatch.setattr(
        native,
        "_fresh_outputs",
        lambda path: (_ for _ in ()).throw(AssertionError("outputs inspected before owner admission")),
    )
    monkeypatch.setattr(
        native.oracle,
        "_regular",
        lambda path, label: (
            (_ for _ in ()).throw(AssertionError("binary inspected before owner admission"))
            if "executable" in label
            else Path(path).resolve()
        ),
    )
    out = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="owner refused"):
        native.run(
            model_root=tmp_path,
            reference=reference,
            owner_authorization=authorization,
            preflight_path=preflight,
            source_capture_path=source_capture,
            native_binary=tmp_path / "missing-binary",
            out=out,
        )
    assert not out.exists()
    assert not native._native_artifact_dir(out).exists()


def test_source_capture_requires_exact_hc4_post_ple_payload(tmp_path: Path) -> None:
    capture = tmp_path / "capture.json"
    payload = _post_ple_payload(tmp_path / "post-ple.f32")
    _source_capture(capture, post_ple=payload, shape=[1, 1, 10_240])
    _, binding = native._source_capture(
        capture,
        reference_sha256="a" * 64,
        authorization_sha256="b" * 64,
        preflight_sha256="c" * 64,
    )
    assert binding["post_ple_state"]["elements"] == 10_240
    assert binding["post_ple_state"]["sha256"] == payload["sha256"]
    assert len(binding["layer0_trace"]) == len(native.preflight._required_layer0_trace())

    trace_only_capture = tmp_path / "trace-only-capture.json"
    _source_capture(trace_only_capture, post_ple=None, shape=[1, 1, 10_240])
    _, trace_only_binding = native._source_capture(
        trace_only_capture,
        reference_sha256="a" * 64,
        authorization_sha256="b" * 64,
        preflight_sha256="c" * 64,
        require_post_ple_state=False,
    )
    assert "post_ple_state" not in trace_only_binding
    assert len(trace_only_binding["layer0_trace"]) == len(native.preflight._required_layer0_trace())
    with pytest.raises(ValueError, match="payload binding is absent"):
        native._source_capture(
            trace_only_capture,
            reference_sha256="a" * 64,
            authorization_sha256="b" * 64,
            preflight_sha256="c" * 64,
        )

    trace = _layer0_trace_records(tmp_path)
    verified_trace = native._verify_layer0_trace(
        trace,
        out_dir=tmp_path,
        label_prefix="native test receipt",
        source_runtime_dtype_required=False,
    )
    assert [record["stage"] for record in verified_trace] == [
        declaration["stage"] for declaration in native.preflight._required_layer0_trace()
    ]
    trace[0]["shape"] = [1, 1, 7]
    with pytest.raises(ValueError, match="layer-0 trace order, name, layer, or shape differs"):
        native._verify_layer0_trace(
            trace,
            out_dir=tmp_path,
            label_prefix="native test receipt",
            source_runtime_dtype_required=False,
        )

    wrong_shape = tmp_path / "wrong-shape.json"
    _source_capture(wrong_shape, post_ple=payload, shape=[1, 1, 7_680])
    with pytest.raises(ValueError, match="exact post-PLE"):
        native._source_capture(
            wrong_shape,
            reference_sha256="a" * 64,
            authorization_sha256="b" * 64,
            preflight_sha256="c" * 64,
        )

    Path(payload["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="payload bytes differ"):
        native._source_capture(
            capture,
            reference_sha256="a" * 64,
            authorization_sha256="b" * 64,
            preflight_sha256="c" * 64,
        )


def test_hc_precision_candidate_is_explicit_in_native_command(tmp_path: Path) -> None:
    binding = {
        "capture": {"sha256": "a" * 64},
        "post_ple_state": {"path": str(tmp_path / "post-ple.f32"), "sha256": "b" * 64},
    }
    default = native._command(
        tmp_path / "native",
        model_root=tmp_path,
        out=tmp_path / "native-out.json",
        source_capture_binding=binding,
        preflight_sha256="c" * 64,
    )
    candidate = native._command(
        tmp_path / "native",
        model_root=tmp_path,
        out=tmp_path / "native-out.json",
        source_capture_binding=binding,
        preflight_sha256="c" * 64,
        diagnostic_attention_hc_source_bf16_norm=True,
        diagnostic_attention_hc_norm_threadgroup=128,
    )
    flag = "--diagnostic-attention-hc-source-bf16-norm"
    assert flag not in default
    assert candidate.count(flag) == 1
    threadgroup_flag = "--diagnostic-attention-hc-norm-threadgroup"
    assert candidate.count(threadgroup_flag) == 1
    assert candidate[candidate.index(threadgroup_flag) + 1] == "128"
    full_pairwise = native._command(
        tmp_path / "native",
        model_root=tmp_path,
        out=tmp_path / "full-pairwise.json",
        source_capture_binding=binding,
        preflight_sha256="c" * 64,
        diagnostic_attention_hc_source_bf16_norm=True,
        diagnostic_attention_hc_norm_full_pairwise=True,
    )
    full_pairwise_flag = "--diagnostic-attention-hc-norm-full-pairwise"
    assert full_pairwise.count(flag) == 1
    assert full_pairwise.count(full_pairwise_flag) == 1
    assert threadgroup_flag not in full_pairwise
    trace_only = native._command(
        tmp_path / "native",
        model_root=tmp_path,
        out=tmp_path / "layer0-trace.json",
        source_capture_binding={"capture": {"sha256": "a" * 64}},
        preflight_sha256="c" * 64,
        diagnostic_attention_hc_source_bf16_norm=True,
        diagnostic_attention_hc_norm_full_pairwise=True,
        layer0_hc_trace_only=True,
    )
    assert "--capture-layer0-hc-trace-only" in trace_only
    assert "--capture-source-boundary" not in trace_only
    assert "--source-post-ple-state" not in trace_only
    assert "--source-post-ple-state-sha256" not in trace_only
    assert full_pairwise_flag in trace_only
    with pytest.raises(ValueError, match="requires --diagnostic"):
        native._command(
            tmp_path / "native",
            model_root=tmp_path,
            out=tmp_path / "invalid.json",
            source_capture_binding=binding,
            preflight_sha256="c" * 64,
            diagnostic_attention_hc_norm_threadgroup=128,
        )
    with pytest.raises(ValueError, match="requires --diagnostic"):
        native._command(
            tmp_path / "native",
            model_root=tmp_path,
            out=tmp_path / "invalid-full-pairwise.json",
            source_capture_binding=binding,
            preflight_sha256="c" * 64,
            diagnostic_attention_hc_norm_full_pairwise=True,
        )
    with pytest.raises(ValueError, match="may not specify a threadgroup"):
        native._command(
            tmp_path / "native",
            model_root=tmp_path,
            out=tmp_path / "invalid-full-pairwise-threadgroup.json",
            source_capture_binding=binding,
            preflight_sha256="c" * 64,
            diagnostic_attention_hc_source_bf16_norm=True,
            diagnostic_attention_hc_norm_threadgroup=128,
            diagnostic_attention_hc_norm_full_pairwise=True,
        )


def test_layer0_trace_receipt_requires_exact_native_scope_and_lineage(tmp_path: Path) -> None:
    out = tmp_path / "layer0-trace.json"
    artifact_dir = native._native_artifact_dir(out)
    artifact_dir.mkdir()
    seams = []
    for declaration in native.preflight._required_seams()[:1]:
        seams.append({
            **declaration,
            "payload": _post_ple_payload(
                artifact_dir / f"seam-{declaration['ordinal']:02d}.f32",
                elements=native.preflight._shape_elements(declaration["shape"]),
            ),
        })
    document = {
        "schema": native.NATIVE_LAYER0_TRACE_SCHEMA,
        "status": native.NATIVE_LAYER0_TRACE_STATUS,
        "capture_scope": "LAYER0_ATTENTION_HC_TRACE_ONLY",
        "root": str(tmp_path),
        "token_index": 0,
        "token_id": native.gate.PROMPT_IDS[0],
        "layers_executed": [0],
        "layer0_trace_contract_version": 5,
        "source_capture_lineage": {
            "source_capture_sha256": "a" * 64,
            "preflight_sha256": "b" * 64,
            "source_ple_state_consumed": False,
        },
        "seams": seams,
        "layer0_trace": _layer0_trace_records(artifact_dir),
        "execution": {
            "attention_hc_norm_precision": {"mode": "NATIVE_F32_NO_DIAGNOSTIC_CAST"},
            "mlp_executed": False,
            "full_body_executed": False,
            "lm_head_executed": False,
        },
        "promotion_allowed": False,
    }
    _sealed(out, document)
    verified = native._verify_native_receipt(
        out,
        model_root=tmp_path,
        source_capture_sha256="a" * 64,
        preflight_sha256="b" * 64,
        post_ple_sha256=None,
        diagnostic_attention_hc_source_bf16_norm=False,
        layer0_hc_trace_only=True,
    )
    assert verified["capture_scope"] == "LAYER0_ATTENTION_HC_TRACE_ONLY"
    assert verified["seam_count"] == 1
    assert verified["layer0_trace_count"] == len(native.preflight._required_layer0_trace())

    document["source_capture_lineage"]["source_ple_state_consumed"] = True
    document.pop("seal_sha256", None)
    _sealed(out, document)
    with pytest.raises(ValueError, match="native layer-0 trace receipt identity, source lineage, or scope differs"):
        native._verify_native_receipt(
            out,
            model_root=tmp_path,
            source_capture_sha256="a" * 64,
            preflight_sha256="b" * 64,
            post_ple_sha256=None,
            diagnostic_attention_hc_source_bf16_norm=False,
            layer0_hc_trace_only=True,
        )


def test_native_run_trace_scope_omits_the_post_ple_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "reference.json"
    reference.write_text("{}", encoding="utf-8")
    binary = tmp_path / "native"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(binary.stat().st_mode | 0o111)
    out = tmp_path / "layer0-trace.json"

    monkeypatch.setattr(
        native.gate,
        "load_admitted_reference_for_boundary",
        lambda *args, **kwargs: {
            "authorization_binding_sha256": None,
            "seal_sha256": "reference-seal",
            "admission_class": "LOCAL_CANONICAL",
            "owner_authorization": None,
        },
    )
    monkeypatch.setattr(
        native.source_capture,
        "_preflight",
        lambda *args, **kwargs: ({}, {"sha256": "preflight-sha256"}),
    )

    def source_capture(*args: object, **kwargs: object) -> tuple[dict, dict]:
        assert kwargs["require_post_ple_state"] is False
        return {}, {"capture": {"sha256": "capture-sha256"}, "layer0_trace": []}

    monkeypatch.setattr(native, "_source_capture", source_capture)
    monkeypatch.setattr(native, "_fresh_outputs", lambda path: path.with_suffix(".runner.json"))
    monkeypatch.setattr(native.gate, "_protected_native_lane", lambda: {"clean": True, "matches": []})
    monkeypatch.setattr(native.gate, "_preexecution_build_binding", lambda path: {"binary": str(path)})
    monkeypatch.setattr(native.gate, "_same_executable_closure", lambda *args: True)
    monkeypatch.setattr(native.gate, "_write_new_runner_sidecar", lambda *args, **kwargs: None)

    def verify(*args: object, **kwargs: object) -> dict:
        assert kwargs["post_ple_sha256"] is None
        assert kwargs["layer0_hc_trace_only"] is True
        return {"capture_scope": "LAYER0_ATTENTION_HC_TRACE_ONLY"}

    monkeypatch.setattr(native, "_verify_native_receipt", verify)
    result = native.run(
        model_root=tmp_path,
        reference=reference,
        owner_authorization=None,
        preflight_path=tmp_path / "preflight.json",
        source_capture_path=tmp_path / "source.json",
        native_binary=binary,
        out=out,
        layer0_hc_trace_only=True,
    )

    assert result["status"] == native.LAYER0_TRACE_STATUS
    assert result["source_ple_state"]["consumed"] is False
    assert "source_post_ple_state" not in result
    assert "--capture-layer0-hc-trace-only" in result["command"]
    assert "--source-post-ple-state" not in result["command"]


def test_native_run_keeps_full_pairwise_effective_width_distinct_from_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The profile's fixed width must not re-enter the raw-option validator."""
    reference = tmp_path / "reference.json"
    reference.write_text("{}", encoding="utf-8")
    binary = tmp_path / "native"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(binary.stat().st_mode | 0o111)
    out = tmp_path / "native-out.json"

    monkeypatch.setattr(
        native.gate,
        "load_admitted_reference_for_boundary",
        lambda *args, **kwargs: {
            "authorization_binding_sha256": None,
            "seal_sha256": "reference-seal",
            "admission_class": "LOCAL_CANONICAL",
            "owner_authorization": None,
        },
    )
    monkeypatch.setattr(
        native.source_capture,
        "_preflight",
        lambda *args, **kwargs: ({}, {"sha256": "preflight-sha256"}),
    )
    monkeypatch.setattr(
        native,
        "_source_capture",
        lambda *args, **kwargs: (
            {},
            {
                "capture": {"sha256": "capture-sha256"},
                "post_ple_state": {"sha256": "post-ple-sha256"},
            },
        ),
    )
    monkeypatch.setattr(native, "_fresh_outputs", lambda path: path.with_suffix(".runner.json"))
    monkeypatch.setattr(native.gate, "_protected_native_lane", lambda: {"clean": True, "matches": []})
    monkeypatch.setattr(native.gate, "_preexecution_build_binding", lambda path: {"binary": str(path)})
    monkeypatch.setattr(native.gate, "_same_executable_closure", lambda *args: True)
    monkeypatch.setattr(native.gate, "_write_new_runner_sidecar", lambda *args, **kwargs: None)

    def command(*args, **kwargs):
        assert kwargs["diagnostic_attention_hc_norm_threadgroup"] is None
        assert kwargs["diagnostic_attention_hc_norm_full_pairwise"] is True
        return ["/usr/bin/true"]

    def verify(*args, **kwargs):
        assert kwargs["diagnostic_attention_hc_norm_threadgroup"] is None
        assert kwargs["diagnostic_attention_hc_norm_full_pairwise"] is True
        return {"capture": "verified"}

    monkeypatch.setattr(native, "_command", command)
    monkeypatch.setattr(native, "_verify_native_receipt", verify)

    result = native.run(
        model_root=tmp_path,
        reference=reference,
        owner_authorization=None,
        preflight_path=tmp_path / "preflight.json",
        source_capture_path=tmp_path / "source.json",
        native_binary=binary,
        out=out,
        diagnostic_attention_hc_source_bf16_norm=True,
        diagnostic_attention_hc_norm_full_pairwise=True,
    )

    assert result["diagnostic_attention_hc_norm_threadgroup"] == 256
