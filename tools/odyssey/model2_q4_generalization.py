#!/usr/bin/env python3
"""Record the bounded model-2 Qwen30 Q4 generalization checks.

This is a source-and-wiring receipt, not a physical artifact or runtime pass.
It proves that the existing Qwen30 uniform-Q4 path can be parameterized for the
base sibling without weakening source admission, while leaving the expensive
current-shard revalidation, full pack, and token execution explicitly open.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from lab.receipts import seal, verify

SOURCE = Path(
    "/Volumes/corpdrive/hawking-modellake/specimens/"
    "Qwen--Qwen3-30B-A3B@ad44e777bcd1"
)
AUDIT = Path("/Users/scammermike/noetic/MODEL2_AUDIT/QWEN30BASE_SOURCE_BODY_AUDIT_CANDIDATE.json")
RUNTIME = REPO / "crates/hawking-core/src/model/qwen30_complete_runtime.rs"
NATIVE_CLI = REPO / "crates/hawking-core/examples/ascension_qwen30_complete_native_runtime.rs"
UNIFORM = REPO / "crates/hawking-core/src/model/qwen_complete_binary/uniform_q4.rs"
REPACK = REPO / "lab/operators/ascension_qwen30_uniform_q4_repack.py"
OUT = REPO / "receipts/headless/NOETIC_MODEL2_Q4_GENERALIZATION.json"
SOURCE_REVALIDATION = Path(
    "/Users/scammermike/noetic/MODEL2_Q4_CONTROL/"
    "QWEN30BASE_Q4_CONTROL_CURRENT_SOURCE_SHARD_REVALIDATION.json"
)


def main() -> int:
    config = json.loads((SOURCE / "config.json").read_text())
    tokenizer_config = json.loads((SOURCE / "tokenizer_config.json").read_text())
    audit = json.loads(AUDIT.read_text())
    source = audit["source"]
    runtime = RUNTIME.read_text()
    native_cli = NATIVE_CLI.read_text()
    uniform = UNIFORM.read_text()
    repack = REPACK.read_text()
    revalidation = None
    if SOURCE_REVALIDATION.is_file():
        try:
            candidate = json.loads(SOURCE_REVALIDATION.read_text())
            checked = verify(candidate, label=str(SOURCE_REVALIDATION))
            if (
                checked.get("status") == "EARNED_CURRENT_SOURCE_SHARDS_REVALIDATED"
                and checked.get("source_repository") == source.get("repository")
                and checked.get("source_revision") == source.get("revision")
                and len(checked.get("shards", {})) == source.get("shard_count")
            ):
                revalidation = checked
        except Exception:
            # This source-contract receipt remains useful before the expensive
            # physical pass; an unverified optional receipt must not be trusted.
            revalidation = None

    template = tokenizer_config.get("chat_template", "")
    checks = {
        "source_directory_present": SOURCE.is_dir(),
        "source_config_is_qwen3_moe": (
            config.get("architectures") == ["Qwen3MoeForCausalLM"]
            and config.get("model_type") == "qwen3_moe"
        ),
        "sealed_source_audit_matches_base": (
            source.get("repository") == "Qwen/Qwen3-30B-A3B"
            and source.get("revision") == "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
            and source.get("shard_count") == 16
            and len(source.get("shards", {})) == 16
            and all(
                isinstance(row, dict)
                and isinstance(row.get("sha256"), str)
                and len(row["sha256"]) == 64
                for row in source.get("shards", {}).values()
            )
        ),
        "embedded_chat_template_is_recognized": (
            not (SOURCE / "chat_template.jinja").exists()
            and "for message in messages" in template
            and "add_generation_prompt" in template
            and "<|im_start|>" in template
        ),
        "uniform_admission_resolves_base_variant": (
            "Qwen/Qwen3-30B-A3B" in uniform
            and "model_for_manifest_source" in uniform
            and "QwenCompleteBinaryModel::Qwen30Base" in uniform
        ),
        "runtime_uses_admitted_repository": (
            runtime.count("artifact.model.source_repository()") >= 2
            and "direct.model.source_repository()" in runtime
        ),
        "uniform_q4_preflight_is_family_specific": (
            "pub fn preflight_uniform_q4_runtime" in runtime
            and "preflight_uniform_q4_runtime" in native_cli
            and "if is_uniform_q4" in native_cli
        ),
        "repacker_is_source_parameterized": all(
            token in repack
            for token in (
                "QWEN30_REPACK_MODEL_DIR",
                "QWEN30_REPACK_SOURCE_AUDIT",
                "QWEN30_REPACK_SOURCE_REVALIDATION",
                "QWEN30_REPACK_SOURCE_REPOSITORY",
            )
        ),
            "bounded_repacker_refuses_partial_manifest": (
                "PARTIAL_UNIFORM_Q4_GROUP64_PROGRESS" in repack
                and '"complete_candidate": False' in repack
                and "if max_tensors is not None and len(done) < len(all_names)" in repack
            ),
    }

    test = subprocess.run(
        [
            "cargo",
            "test",
            "--manifest-path",
            str(REPO / "crates/hawking-core/Cargo.toml"),
            "--lib",
            "uniform_q4",
            "--quiet",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
    )
    checks["native_uniform_q4_tests_pass"] = test.returncode == 0

    out = {
        "schema": "hawking.odyssey.noetic_model2_q4_generalization.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/model2_q4_generalization.py",
        "obligation": "G023 — model-2 Qwen30 uniform-Q4 generalization",
        "hand_authored": False,
        "specimen": {
            "repository": source.get("repository"),
            "revision": source.get("revision"),
            "model_dir": str(SOURCE),
            "source_audit": str(AUDIT),
            "source_audit_seal_sha256": audit.get("seal_sha256"),
            "source_revalidation": (
                {
                    "path": str(SOURCE_REVALIDATION),
                    "seal_sha256": revalidation.get("seal_sha256"),
                    "observed_total_bytes": revalidation.get("observed_total_bytes"),
                    "shard_count": len(revalidation.get("shards", {})),
                }
                if revalidation is not None
                else None
            ),
        },
        "checks": checks,
        "test_command": "cargo test --manifest-path crates/hawking-core/Cargo.toml --lib uniform_q4 --quiet",
        "test_returncode": test.returncode,
        "claim_boundary": {
            "current_source_shard_revalidation_completed": revalidation is not None,
            "full_qwen30base_q4_artifact_packed": False,
            "uniform_q4_artifact_admitted": False,
            "native_token_executed": False,
            "device_profiles_qualified": False,
            "cuda_hardware_execution": False,
            "meaning": "wiring and source-contract generalization only; no physical or capability claim",
        },
    }
    out["pass"] = all(checks.values())
    out = seal(out)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"receipt": str(OUT), "pass": out["pass"], "checks": checks}, indent=2))
    if test.returncode != 0:
        print(test.stderr[-4000:])
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
