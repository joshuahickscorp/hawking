from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.odyssey import flash_source_boundary_capture_mlx as capture
from tools.odyssey import flash_source_boundary_preflight as preflight


def _raw_binding(path: Path, values: int, *, dtype: str = "F32_LE") -> dict:
    raw = b"\0" * (values * 4)
    path.write_bytes(raw)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dtype": dtype,
        "elements": values,
        "bytes": len(raw),
    }


def _cache_records(tmp_path: Path, boundary: str) -> list[dict]:
    records: list[dict] = []
    for index, declaration in enumerate(preflight._required_layer4_cache()[boundary]):
        records.append({
            **declaration,
            "path_in_provider_cache": f"test_cache[{index}]",
            "original_dtype": "mlx.core.float32",
            "payload": _raw_binding(
                tmp_path / f"cache-{boundary}-{index}.f32",
                preflight._shape_elements(declaration["shape"]),
            ),
        })
    return records


def _layer0_trace_records(tmp_path: Path) -> list[dict]:
    records: list[dict] = []
    for declaration in preflight._required_layer0_trace():
        records.append({
            **declaration,
            "original_dtype": "mlx.core.bfloat16",
            "payload": _raw_binding(
                tmp_path / f"layer0-trace-{declaration['ordinal']:02d}.f32",
                preflight._shape_elements(declaration["shape"]),
            ),
        })
    return records


def test_capture_refuses_before_output_reservation_when_owner_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "reference.json"
    authorization = tmp_path / "authorization.json"
    boundary = tmp_path / "preflight.json"
    reference.write_text("{}", encoding="utf-8")
    authorization.write_text("{}", encoding="utf-8")
    boundary.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        capture.gate,
        "load_admitted_reference_for_boundary",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("owner refused")),
    )
    monkeypatch.setattr(
        capture.oracle,
        "_reserve_output",
        lambda path: (_ for _ in ()).throw(AssertionError("output reserved before owner admission")),
    )
    with pytest.raises(ValueError, match="owner refused"):
        capture.capture(
            model_root=tmp_path,
            reference=reference,
            owner_authorization=authorization,
            preflight_path=boundary,
            out_dir=tmp_path / "must-not-exist",
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_worker_manifest_verifier_requires_every_declared_payload(tmp_path: Path) -> None:
    declarations = preflight._required_seams()
    seams = []
    for declaration in declarations:
        record = dict(declaration)
        if declaration["ordinal"] == 11:
            record.update({
                "expert_ids": list(range(10)),
                "weights": [0.1] * 10,
                "ids_payload": _raw_binding(tmp_path / "ids.i32", 10, dtype="I32_LE"),
                "weights_payload": _raw_binding(tmp_path / "weights.f32", 10),
            })
        else:
            elements = 1
            for value in declaration["shape"]:
                elements *= value
            record["payload"] = _raw_binding(
                tmp_path / f"seam-{declaration['ordinal']:02d}.f32", elements
            )
        seams.append(record)
    pre_cache = _cache_records(tmp_path, "pre_attention")
    post_cache = _cache_records(tmp_path, "post_attention")
    layer0_trace = _layer0_trace_records(tmp_path)
    document = {
        "schema": capture.WORKER_SCHEMA,
        "status": capture.WORKER_STATUS,
        "root": str(tmp_path),
        "token_index": 0,
        "token_id": preflight.gate.PROMPT_IDS[0],
        "reference_contract_sha256": "a" * 64,
        "owner_authorization_sha256": "b" * 64,
        "preflight_sha256": "c" * 64,
        "layers_executed": [0, 1, 2, 3, 4],
        "ple_layer_ids": [1],
        "full_body_executed": False,
        "lm_head_executed": False,
        "layer0_trace_contract_version": 5,
        "layer0_trace": layer0_trace,
        "seams": seams,
        "layer4_cache": {"pre_attention": pre_cache, "post_attention": post_cache},
    }
    manifest = tmp_path / "source-boundary-manifest.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    verified = capture._verify_worker_manifest(
        manifest,
        out_dir=tmp_path,
        model_root=tmp_path,
        reference_sha256="a" * 64,
        authorization_sha256="b" * 64,
        preflight_sha256="c" * 64,
    )
    assert len(verified["seams"]) == 14
    assert len(verified["layer0_trace"]) == 9
    assert [record["state_role"] for record in verified["layer4_cache"]["pre_attention"]] == [
        "conv_state",
        "recurrent_state",
    ]
    document["layer0_trace"] = document["layer0_trace"][:-1]
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="did not emit every predeclared layer-0 trace"):
        capture._verify_worker_manifest(
            manifest,
            out_dir=tmp_path,
            model_root=tmp_path,
            reference_sha256="a" * 64,
            authorization_sha256="b" * 64,
            preflight_sha256="c" * 64,
        )
    document["layer0_trace"] = _layer0_trace_records(tmp_path)
    document["layer0_trace"][0]["shape"] = [1, 1, 7]
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="layer-0 trace order, name, layer, or shape differs"):
        capture._verify_worker_manifest(
            manifest,
            out_dir=tmp_path,
            model_root=tmp_path,
            reference_sha256="a" * 64,
            authorization_sha256="b" * 64,
            preflight_sha256="c" * 64,
        )
    document["layer0_trace"] = _layer0_trace_records(tmp_path)
    document["layer4_cache"]["pre_attention"][0]["initialization"] = "untrusted"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="cache record is malformed"):
        capture._verify_worker_manifest(
            manifest,
            out_dir=tmp_path,
            model_root=tmp_path,
            reference_sha256="a" * 64,
            authorization_sha256="b" * 64,
            preflight_sha256="c" * 64,
        )
    document["layer4_cache"]["pre_attention"][0]["initialization"] = (
        preflight._required_layer4_cache()["pre_attention"][0]["initialization"]
    )
    document["seams"][11].pop("shape")
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="seam order, name, layer, or shape differs"):
        capture._verify_worker_manifest(
            manifest,
            out_dir=tmp_path,
            model_root=tmp_path,
            reference_sha256="a" * 64,
            authorization_sha256="b" * 64,
            preflight_sha256="c" * 64,
        )
    document["seams"][11]["shape"] = [10]
    manifest.write_text(json.dumps(document), encoding="utf-8")
    Path(seams[8]["payload"]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="payload bytes differ"):
        capture._verify_worker_manifest(
            manifest,
            out_dir=tmp_path,
            model_root=tmp_path,
            reference_sha256="a" * 64,
            authorization_sha256="b" * 64,
            preflight_sha256="c" * 64,
        )
