from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from tools.odyssey import flash_source_boundary_compare as compare


def _payload(path: Path, values: list[float] | list[int], *, dtype: str) -> dict:
    if dtype == "F32_LE":
        raw = b"".join(struct.pack("<f", float(value)) for value in values)
    else:
        raw = b"".join(struct.pack("<i", int(value)) for value in values)
    path.write_bytes(raw)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dtype": dtype,
        "elements": len(values),
        "bytes": len(raw),
    }


def _layer0_trace_records(directory: Path) -> list[dict]:
    records: list[dict] = []
    for declaration in compare.preflight._required_layer0_trace():
        records.append({
            **declaration,
            "payload": _payload(
                directory / f"layer0-trace-{declaration['ordinal']:02d}.f32",
                [0.0] * compare.preflight._shape_elements(declaration["shape"]),
                dtype="F32_LE",
            ),
        })
    return records


def test_metrics_are_finite_and_do_not_invent_zero_vector_cosine() -> None:
    exact = compare._metrics([0.0, 0.0], [0.0, 0.0])
    assert exact["relative_l2"] == 0.0
    assert exact["cosine"] is None
    assert exact["reference_l2_zero"] is True
    shifted = compare._metrics([1.0, 2.0], [1.0, 3.0])
    assert shifted["max_abs"] == 1.0
    assert shifted["relative_l2"] == pytest.approx(1.0 / 5.0**0.5)


def test_seam_comparison_stops_at_first_exact_payload_difference(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    native_dir = tmp_path / "native"
    source_dir.mkdir()
    native_dir.mkdir()
    declarations = compare.preflight._required_seams()
    source_seams = [dict(declaration) for declaration in declarations]
    native_seams = [dict(declaration) for declaration in declarations]
    zeros = [0.0] * 10_240
    source_seams[0]["payload"] = _payload(source_dir / "seam-0.f32", zeros, dtype="F32_LE")
    native_seams[0]["payload"] = _payload(native_dir / "seam-0.f32", zeros, dtype="F32_LE")
    shifted = list(zeros)
    shifted[0] = 1.0
    source_seams[1]["payload"] = _payload(source_dir / "seam-1.f32", zeros, dtype="F32_LE")
    native_seams[1]["payload"] = _payload(native_dir / "seam-1.f32", shifted, dtype="F32_LE")
    compared, first = compare._compare_seams(
        source_seams, native_seams, source_dir=source_dir, native_dir=native_dir
    )
    assert len(compared) == 2
    assert first == {
        "ordinal": 1,
        "layer": 0,
        "stage": "mlp_injection",
        "observed_difference_kind": "F32_PAYLOAD_BYTES",
        "semantic_attribution": "WITHHELD_NO_PREDECLARED_NUMERICAL_ACCEPTANCE_BOUND",
    }
    assert compared[1]["comparison"]["metrics"]["max_abs"] == 1.0


def test_layer0_trace_localizes_before_the_coarse_layer0_seam(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    native_dir = tmp_path / "native"
    source_dir.mkdir()
    native_dir.mkdir()
    source_trace = _layer0_trace_records(source_dir)
    native_trace = _layer0_trace_records(native_dir)
    shifted = [0.0] * compare.preflight._shape_elements(native_trace[1]["shape"])
    shifted[1] = 1.0
    native_trace[1]["payload"] = _payload(
        native_dir / "layer0-trace-shifted.f32", shifted, dtype="F32_LE"
    )

    compared, first = compare._compare_layer0_trace(
        source_trace,
        native_trace,
        source_dir=source_dir,
        native_dir=native_dir,
    )

    assert len(compared) == 2
    assert first == {
        "comparison_scope": "layer0_trace",
        "ordinal": 1,
        "layer": 0,
        "stage": "attention_hyper_connection_down_projection",
        "observed_difference_kind": "F32_PAYLOAD_BYTES",
        "semantic_attribution": "WITHHELD_NO_PREDECLARED_NUMERICAL_ACCEPTANCE_BOUND",
    }
    assert compared[1]["comparison"]["metrics"]["max_abs"] == 1.0


def test_route_comparison_keeps_order_exact_and_weights_diagnostic(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    native_dir = tmp_path / "native"
    source_dir.mkdir()
    native_dir.mkdir()
    ids = list(range(10))
    reversed_ids = list(reversed(ids))
    weights = [float(10 - index) / 55.0 for index in range(10)]
    source = {
        "ids_payload": _payload(source_dir / "ids.i32", ids, dtype="I32_LE"),
        "weights_payload": _payload(source_dir / "weights.f32", weights, dtype="F32_LE"),
    }
    candidate = {
        "ids_payload": _payload(native_dir / "ids.i32", reversed_ids, dtype="I32_LE"),
        "weights_payload": _payload(native_dir / "weights.f32", weights, dtype="F32_LE"),
    }
    result = compare._route_record(
        source, candidate, source_dir=source_dir, native_dir=native_dir
    )
    assert result["ordered_route_ids_exact"] is False
    assert result["route_weights_exact_sha256"] is True
    assert result["route_weight_metrics"]["relative_l2"] == 0.0


def test_cache_comparison_requires_named_source_to_native_state_mapping(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    native_dir = tmp_path / "native"
    source_dir.mkdir()
    native_dir.mkdir()
    source_cache = {
        "pre_attention": [{
            "state_role": "conv_state",
            "shape": [1],
            "payload": _payload(source_dir / "source.f32", [0.0], dtype="F32_LE"),
        }],
        "post_attention": [],
    }
    native_cache = {
        "pre_attention": [{
            "name": "recurrent-state",
            "shape": [1],
            "payload": _payload(native_dir / "native.f32", [0.0], dtype="F32_LE"),
        }],
        "post_attention": [],
    }

    with pytest.raises(ValueError, match="state-role mapping differs"):
        compare._compare_cache(
            source_cache,
            native_cache,
            source_dir=source_dir,
            native_dir=native_dir,
        )


def test_compare_refuses_at_owner_gate_before_capture_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "reference.json"
    authorization = tmp_path / "authorization.json"
    for path in (reference, authorization):
        path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        compare.gate,
        "load_admitted_reference_for_boundary",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("owner refused")),
    )
    monkeypatch.setattr(
        compare.native,
        "_source_capture",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("capture consumed before owner admission")
        ),
    )
    monkeypatch.setattr(
        compare.oracle,
        "_regular",
        lambda path, label: Path(path).resolve(),
    )
    out = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="owner refused"):
        compare.compare(
            model_root=tmp_path,
            reference=reference,
            owner_authorization=authorization,
            preflight_path=tmp_path / "missing-preflight.json",
            source_capture_path=tmp_path / "missing-source.json",
            native_capture_path=tmp_path / "missing-native.json",
            hawking_binary=tmp_path / "missing-hawking",
            out=out,
        )
    assert not out.exists()


def test_compare_forwards_explicit_hc_precision_mode_to_native_receipt_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class VerifiedPrecision(Exception):
        pass

    reference = tmp_path / "reference.json"
    reference.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(compare.oracle, "_regular", lambda path, label: Path(path).resolve())
    monkeypatch.setattr(compare.oracle, "_sha256", lambda path: "sha256")
    monkeypatch.setattr(
        compare.gate,
        "load_admitted_reference_for_boundary",
        lambda *args, **kwargs: {
            "authorization_binding_sha256": None,
            "seal_sha256": "reference-seal",
            "admission_class": "LOCAL_CANONICAL",
            "owner_authorization": None,
        },
    )
    monkeypatch.setattr(
        compare.source_capture,
        "_preflight",
        lambda *args, **kwargs: ({}, {"sha256": "preflight-sha256"}),
    )
    monkeypatch.setattr(
        compare.native,
        "_source_capture",
        lambda *args, **kwargs: (
            {},
            {
                "capture": {"sha256": "source-capture-sha256"},
                "post_ple_state": {"sha256": "post-ple-sha256"},
            },
        ),
    )

    def verify_native_receipt(*args, **kwargs):
        assert kwargs["diagnostic_attention_hc_source_bf16_norm"] is True
        assert kwargs["diagnostic_attention_hc_norm_threadgroup"] is None
        assert kwargs["diagnostic_attention_hc_norm_full_pairwise"] is True
        raise VerifiedPrecision

    monkeypatch.setattr(compare.native, "_verify_native_receipt", verify_native_receipt)

    with pytest.raises(VerifiedPrecision):
        compare.compare(
            model_root=tmp_path,
            reference=reference,
            owner_authorization=None,
            preflight_path=tmp_path / "preflight.json",
            source_capture_path=tmp_path / "source.json",
            native_capture_path=tmp_path / "native.json",
            hawking_binary=tmp_path / "hawking",
            out=tmp_path / "comparison.json",
            diagnostic_attention_hc_source_bf16_norm=True,
            diagnostic_attention_hc_norm_full_pairwise=True,
        )
