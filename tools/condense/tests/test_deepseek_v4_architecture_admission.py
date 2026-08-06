#!/usr/bin/env python3.12
"""Fail-closed controls for the bounded DeepSeek-V4 architecture fixture."""
from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import deepseek_v4_architecture_admission as admission


REVISION = "a" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(dtype: str) -> dict[str, object]:
    return {"dtype": dtype, "shape": [1], "data_offsets": [0, 1]}


def _write_header(path: Path, *, wrong_expert_name: bool = False, body: bytes = b"") -> None:
    expert = (
        "model.layers.4.mlp.experts.0.w1.weight"
        if wrong_expert_name
        else "layers.4.ffn.experts.0.w1.weight"
    )
    rows: dict[str, dict[str, object]] = {
        expert: _descriptor("I8"),
        f"{expert[:-len('.weight')]}.scale": _descriptor("F8_E8M0"),
        "layers.4.ffn.shared_experts.w1.weight": _descriptor("F8_E4M3"),
        "layers.4.ffn.shared_experts.w1.scale": _descriptor("F8_E8M0"),
        "layers.4.ffn.gate.weight": _descriptor("BF16"),
        "layers.4.attn.indexer.wq_b.weight": _descriptor("F8_E4M3"),
        "layers.4.attn.indexer.wq_b.scale": _descriptor("F8_E8M0"),
        "layers.4.attn.compressor.wkv.weight": _descriptor("BF16"),
        "layers.4.attn.indexer.compressor.wkv.weight": _descriptor("BF16"),
    }
    for suffix in (
        "hc_attn_base",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_ffn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
    ):
        rows[f"layers.4.{suffix}"] = _descriptor("BF16")
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + body)


def _write_inputs(tmp_path: Path, *, wrong_expert_name: bool = False, body: bytes = b"") -> tuple[Path, Path, Path]:
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "expert_dtype": "fp4",
        "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "scale_fmt": "ue8m0"},
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "n_shared_experts": 1,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "qk_rope_head_dim": 64,
        "compress_rope_theta": 160000,
        "compress_ratios": [0, 0, 4, 128, 0],
        "index_topk": 512,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "num_hash_layers": 3,
    }
    tokenizer = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": {"content": "<bos>"},
        "eos_token": {"content": "<eos>"},
        "chat_template": "{{ messages }}",
    }
    config_path = tmp_path / "config.json"
    tokenizer_path = tmp_path / "tokenizer_config.json"
    header_path = tmp_path / "model-00001-of-00046.header"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")
    _write_header(header_path, wrong_expert_name=wrong_expert_name, body=body)
    return config_path, tokenizer_path, header_path


def _source(config: Path, tokenizer: Path, header: Path) -> dict[str, str]:
    return {
        "repo": admission.EXPECTED_REPOSITORY,
        "revision": REVISION,
        "immutable_tree_url": f"https://huggingface.co/{admission.EXPECTED_REPOSITORY}/tree/{REVISION}",
        "config_sha256": _sha256(config),
        "tokenizer_config_sha256": _sha256(tokenizer),
        "safetensors_header_sha256": _sha256(header),
    }


def _write_authority_and_envelope(
    tmp_path: Path, config: Path, tokenizer: Path, header: Path, *, result_status: str = "PASS"
) -> tuple[Path, Path]:
    source = _source(config, tokenizer, header)
    results = {
        behavior: {
            "status": result_status,
            "fixture_receipt_sha256": hashlib.sha256(f"receipt:{behavior}".encode()).hexdigest(),
            "test_vector_sha256": hashlib.sha256(f"vector:{behavior}".encode()).hexdigest(),
            "implementation_sha256": hashlib.sha256(f"implementation:{behavior}".encode()).hexdigest(),
            "output_sha256": hashlib.sha256(f"output:{behavior}".encode()).hexdigest(),
            "source_binding": source,
        }
        for behavior in admission.REQUIRED_BEHAVIORS
    }
    authority = admission.seal_document(
        {
            "schema": admission.SOURCE_AUTHORITY_SCHEMA,
            "status": "PASS",
            "source": source,
            "verified_blobs": {
                "config": {"sha256": source["config_sha256"], "blob_id": "b" * 40},
                "tokenizer_config": {
                    "sha256": source["tokenizer_config_sha256"],
                    "blob_id": "c" * 40,
                },
            },
            "owning_shard": {
                "filename": admission.FIXTURE_SHARD_FILENAME,
                "lfs_sha256": admission.FIXTURE_SHARD_LFS_SHA256,
                "full_size_bytes": admission.FIXTURE_SHARD_FULL_SIZE_BYTES,
                "header_capture_sha256": source["safetensors_header_sha256"],
                "header_range": {
                    "length_prefix_inclusive": [0, 7],
                    "json_inclusive": [8, header.stat().st_size - 1],
                },
            },
            "authority": {
                "kind": "official_manifest_control_plane",
                "independently_verified": True,
                "verification_receipt_sha256": "e" * 64,
            },
            "freshness": {
                "verified_at": "2026-08-04T00:00:00Z",
                "expires_at": "2099-08-05T00:00:00Z",
            },
            "campaign_nonce_sha256": "a" * 64,
            "fixture_attestations": {
                behavior: {
                    key: result[key]
                    for key in (
                        "fixture_receipt_sha256",
                        "test_vector_sha256",
                        "implementation_sha256",
                        "output_sha256",
                    )
                }
                for behavior, result in results.items()
            },
        }
    )
    authority_path = tmp_path / "official-source-authority.json"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    envelope = admission.seal_document(
        {
            "schema": admission.SOURCE_ENVELOPE_SCHEMA,
            "status": "SOURCE_EXACT",
            "source": source,
            "official_source_authority_sha256": _sha256(authority_path),
            "fixture_results": results,
        }
    )
    envelope_path = tmp_path / "source-envelope.json"
    envelope_path.write_text(json.dumps(envelope), encoding="utf-8")
    return authority_path, envelope_path


def _trusted_test_verifier(authority: dict[str, object]) -> dict[str, str]:
    """Test-only stand-in for a future independent control-plane verifier."""

    return {
        "status": "PASS",
        "verifier_id": "test-control-plane",
        "verifier_receipt_sha256": "f" * 64,
        "scope": "official_source_authority",
        "fixture_attestations_sha256": admission._sha256(authority["fixture_attestations"]),
        "campaign_nonce_sha256": authority["campaign_nonce_sha256"],
    }


def test_structural_fixture_without_source_evidence_is_not_admitted(tmp_path: Path) -> None:
    config, tokenizer, header = _write_inputs(tmp_path)
    receipt = admission.evaluate(
        config_path=config,
        tokenizer_config_path=tokenizer,
        safetensors_header_path=header,
    )
    assert receipt["status"] == "NOT_ADMITTED"
    assert receipt["structural_fixture"]["status"] == "STRUCTURAL_PASS"
    assert receipt["source_exact_evidence"]["reason"] == "MISSING_SOURCE_EXACT_EVIDENCE"


def test_source_owned_chat_protocol_covers_null_jinja_template(tmp_path: Path) -> None:
    config, tokenizer, header = _write_inputs(tmp_path)
    tokenizer_data = json.loads(tokenizer.read_text())
    tokenizer_data["chat_template"] = None
    tokenizer.write_text(json.dumps(tokenizer_data), encoding="utf-8")
    protocol = tmp_path / "encoding_dsv4.py"
    protocol.write_text("# source-owned encoding protocol\n", encoding="utf-8")
    receipt = admission.evaluate(
        config_path=config,
        tokenizer_config_path=tokenizer,
        chat_protocol_path=protocol,
        safetensors_header_path=header,
    )
    assert receipt["status"] == "NOT_ADMITTED"
    assert receipt["structural_fixture"]["status"] == "STRUCTURAL_PASS"
    check = receipt["structural_fixture"]["checks"]["tokenizer_template_coverage"]
    assert check["status"] == "STRUCTURAL_PASS"
    assert check["chat_template_state"] == "source_owned_protocol"


def test_self_sealed_complete_documents_cannot_admit_through_cli(tmp_path: Path) -> None:
    config, tokenizer, header = _write_inputs(tmp_path)
    authority, envelope = _write_authority_and_envelope(tmp_path, config, tokenizer, header)
    script = REPO_ROOT / "tools" / "deepseek_v4_architecture_admission.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--tokenizer-config",
            str(tokenizer),
            "--safetensors-header",
            str(header),
            "--source-authority",
            str(authority),
            "--source-envelope",
            str(envelope),
            "--json",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "NOT_ADMITTED"
    assert (
        receipt["source_exact_evidence"]["official_source_authority"]["verifier"]["reason"]
        == "INDEPENDENT_AUTHORITY_VERIFIER_UNAVAILABLE"
    )


def test_verified_authority_control_can_admit_architecture_only(tmp_path: Path, monkeypatch) -> None:
    config, tokenizer, header = _write_inputs(tmp_path)
    authority, envelope = _write_authority_and_envelope(tmp_path, config, tokenizer, header)
    monkeypatch.setattr(admission, "_OFFICIAL_SOURCE_AUTHORITY_VERIFIER", _trusted_test_verifier)
    receipt = admission.evaluate(
        config_path=config,
        tokenizer_config_path=tokenizer,
        safetensors_header_path=header,
        source_authority_path=authority,
        source_envelope_path=envelope,
    )
    assert receipt["status"] == "ARCHITECTURE_ADMITTED"
    assert "native FP4 or FP8 codec implementation" in receipt["does_not_establish"]
    assert "runtime residency, latency, or TPS result" in receipt["does_not_establish"]
    verifier = receipt["source_exact_evidence"]["official_source_authority"]["verifier"]
    assert verifier["verifier_id"] == "test-control-plane"


def test_tampered_envelope_fails_closed(tmp_path: Path) -> None:
    config, tokenizer, header = _write_inputs(tmp_path)
    authority, envelope = _write_authority_and_envelope(tmp_path, config, tokenizer, header)
    raw = json.loads(envelope.read_text())
    raw["fixture_results"]["native_fp4_expert_decode"]["status"] = "PASS_BUT_TAMPERED"
    envelope.write_text(json.dumps(raw), encoding="utf-8")
    receipt = admission.evaluate(
        config_path=config,
        tokenizer_config_path=tokenizer,
        safetensors_header_path=header,
        source_authority_path=authority,
        source_envelope_path=envelope,
    )
    assert receipt["status"] == "NOT_ADMITTED"
    assert "SOURCE_EXACT_EVIDENCE_INVALID" == receipt["source_exact_evidence"]["reason"]
    assert any("seal_sha256" in error for error in receipt["source_exact_evidence"]["errors"])


def test_forged_authority_header_binding_fails_closed(tmp_path: Path) -> None:
    config, tokenizer, header = _write_inputs(tmp_path)
    authority, envelope = _write_authority_and_envelope(tmp_path, config, tokenizer, header)
    raw = json.loads(authority.read_text())
    raw["owning_shard"]["header_capture_sha256"] = "0" * 64
    authority.write_text(json.dumps(admission.seal_document(raw)), encoding="utf-8")
    # An attacker can reseal their own authority document, so also update the
    # envelope's file binding. The independent verifier does not bless the
    # source detail because the authority's header evidence is contradictory.
    envelope_raw = json.loads(envelope.read_text())
    envelope_raw["official_source_authority_sha256"] = _sha256(authority)
    envelope.write_text(json.dumps(admission.seal_document(envelope_raw)), encoding="utf-8")
    receipt = admission.evaluate(
        config_path=config,
        tokenizer_config_path=tokenizer,
        safetensors_header_path=header,
        source_authority_path=authority,
        source_envelope_path=envelope,
    )
    assert receipt["status"] == "NOT_ADMITTED"
    authority_errors = receipt["source_exact_evidence"]["official_source_authority"]["errors"]
    assert any("header_capture_sha256" in error for error in authority_errors)


def test_unknown_fixture_result_and_wrong_tensor_grammar_fail_closed(tmp_path: Path) -> None:
    config, tokenizer, header = _write_inputs(tmp_path, wrong_expert_name=True)
    authority, envelope = _write_authority_and_envelope(
        tmp_path, config, tokenizer, header, result_status="UNKNOWN"
    )
    receipt = admission.evaluate(
        config_path=config,
        tokenizer_config_path=tokenizer,
        safetensors_header_path=header,
        source_authority_path=authority,
        source_envelope_path=envelope,
    )
    assert receipt["status"] == "NOT_ADMITTED"
    assert receipt["structural_fixture"]["checks"]["native_fp4_expert_decode"]["status"] == "BLOCKED"
    assert any(
        "native_fp4_expert_decode: source-exact fixture result must be PASS" in error
        for error in receipt["source_exact_evidence"]["errors"]
    )


def test_header_body_is_refused_without_reading_model_payload(tmp_path: Path) -> None:
    config, tokenizer, header = _write_inputs(tmp_path, body=b"not-a-tensor-body")
    receipt = admission.evaluate(
        config_path=config,
        tokenizer_config_path=tokenizer,
        safetensors_header_path=header,
    )
    assert receipt["status"] == "NOT_ADMITTED"
    assert any("no tensor body" in error for error in receipt["structural_fixture"]["input_errors"])
