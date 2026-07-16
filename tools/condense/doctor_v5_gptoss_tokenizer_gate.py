#!/usr/bin/env python3.12
"""Offline tokenizer provenance, dual-path parity, and round-trip gate for GPT-OSS."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import secrets
import stat
from typing import Any

from tokenizers import Tokenizer
from transformers import AutoTokenizer

import doctor_v5_gptoss_parallel_scaffold as scaffold


ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_ROOT = (
    ROOT / "reports/condense/doctor_v5_unbound/gptoss_120b_parallel/tokenizer"
)
OUTPUT = ROOT / "reports/condense/doctor_v5_unbound/gptoss_120b_parallel/tokenizer_gate.json"
BINDING_OUTPUT = (
    ROOT / "reports/condense/doctor_v5_unbound/gptoss_120b_parallel/tokenizer_binding.json"
)
MODEL_SOURCE_MANIFEST_SHA256 = (
    "9659fc607d6725354cd354e1868a772c86cd2044f70d86ffc07f8e88506dbafe"
)
REPOSITORY = "openai/gpt-oss-120b"
REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
SCHEMA = "hawking.doctor_v5_gptoss_tokenizer_gate.v1"
FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
         "chat_template.jinja")
CORPUS = (
    "The Doctor verifies exact bytes.",
    "Unicode: π λ 漢字 café — résumé",
    "def f(x: int) -> int:\n    return x * x\n",
    " leading  spaces\n\ntrailing spaces  ",
    "<analysis>not a control token unless the tokenizer says so</analysis>",
)


class TokenizerGateError(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _hash_file(path: Path) -> tuple[str, int]:
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _artifact(path: Path) -> dict[str, Any]:
    digest, size = _hash_file(path)
    return {"path": str(path.resolve()), "sha256": digest, "bytes": size}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush()
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    paths = [TOKENIZER_ROOT / name for name in FILES]
    for path in paths:
        info = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise TokenizerGateError(f"tokenizer asset is not immutable regular file: {path}")
    direct = Tokenizer.from_file(str(TOKENIZER_ROOT / "tokenizer.json"))
    fast = AutoTokenizer.from_pretrained(
        str(TOKENIZER_ROOT), local_files_only=True, trust_remote_code=False,
        use_fast=True,
    )
    vectors = []
    for text in CORPUS:
        direct_ids = direct.encode(text, add_special_tokens=False).ids
        fast_ids = fast.encode(text, add_special_tokens=False)
        if direct_ids != fast_ids:
            raise TokenizerGateError("tokenizers/transformers token IDs differ")
        direct_roundtrip = direct.encode(
            direct.decode(direct_ids, skip_special_tokens=False),
            add_special_tokens=False,
        ).ids
        fast_roundtrip = fast.encode(
            fast.decode(fast_ids, skip_special_tokens=False,
                        clean_up_tokenization_spaces=False),
            add_special_tokens=False,
        )
        if direct_roundtrip != direct_ids or fast_roundtrip != fast_ids:
            raise TokenizerGateError("token ID round-trip is not idempotent")
        vectors.append({
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "token_count": len(direct_ids), "token_ids": direct_ids,
            "token_ids_sha256": _hash_value(direct_ids),
        })
    messages = [
        {"role": "system", "content": "Answer exactly and cite evidence."},
        {"role": "user", "content": "Explain 2 + 2."},
    ]
    rendered = fast.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        current_date="2026-07-14",
    )
    encoded_chat = fast.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        current_date="2026-07-14",
    )
    # Transformers 5 returns BatchEncoding here, while older versions returned
    # the ID list directly.  Normalize without silently accepting a batch.
    if hasattr(encoded_chat, "get"):
        chat_ids = encoded_chat.get("input_ids")
    else:
        chat_ids = encoded_chat
    if not isinstance(rendered, str) or not rendered or not isinstance(chat_ids, list) \
            or not chat_ids:
        raise TokenizerGateError("chat template produced no reference vector")
    template_sha, _ = _hash_file(TOKENIZER_ROOT / "chat_template.jinja")
    binding = scaffold.build_tokenizer_binding(
        model_source_manifest_sha256=MODEL_SOURCE_MANIFEST_SHA256,
        files=paths, chat_template_sha256=template_sha,
    )
    binding_errors = scaffold.validate_tokenizer_binding(binding, verify_files=True)
    if binding_errors:
        raise TokenizerGateError("tokenizer binding invalid: " + "; ".join(binding_errors))
    doc = {
        "schema": SCHEMA, "created_at": _now(), "status": "pass",
        "repository": REPOSITORY, "revision": REVISION,
        "model_source_manifest_sha256": MODEL_SOURCE_MANIFEST_SHA256,
        "files": [_artifact(path) for path in paths],
        "implementations": {
            "tokenizers": __import__("tokenizers").__version__,
            "transformers": __import__("transformers").__version__,
            "local_files_only": True, "trust_remote_code": False,
        },
        "plain_text_vectors": vectors,
        "chat_template": {
            "template_sha256": template_sha,
            "messages_sha256": _hash_value(messages),
            "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "token_ids_sha256": _hash_value(chat_ids), "token_count": len(chat_ids),
        },
        "checks": {
            "dual_path_token_id_parity": True,
            "token_id_roundtrip_idempotence": True,
            "chat_template_reference_vector_present": True,
            "source_and_revision_bound": True,
        },
        "promotion_reviewed": False,
        "quality_evaluation_permitted": False,
        "source_deletion_permitted": False,
    }
    doc["gate_sha256"] = _hash_value(doc)
    return binding, doc


def validate(doc: Any, *, verify_files: bool = True) -> list[str]:
    errors = []
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA \
            or doc.get("status") != "pass":
        return ["tokenizer gate schema/status invalid"]
    if doc.get("gate_sha256") != _hash_value({key: row for key, row in doc.items()
                                               if key != "gate_sha256"}):
        errors.append("tokenizer gate hash mismatch")
    if doc.get("repository") != REPOSITORY or doc.get("revision") != REVISION \
            or doc.get("model_source_manifest_sha256") != MODEL_SOURCE_MANIFEST_SHA256:
        errors.append("tokenizer provenance binding mismatch")
    if doc.get("checks") != {
            "dual_path_token_id_parity": True,
            "token_id_roundtrip_idempotence": True,
            "chat_template_reference_vector_present": True,
            "source_and_revision_bound": True}:
        errors.append("tokenizer checks are incomplete")
    if doc.get("promotion_reviewed") is not False \
            or doc.get("quality_evaluation_permitted") is not False \
            or doc.get("source_deletion_permitted") is not False:
        errors.append("tokenizer gate overclaims readiness")
    if verify_files:
        for row in doc.get("files", []):
            try:
                if _artifact(Path(row["path"])) != row:
                    errors.append(f"tokenizer asset changed: {row.get('path')}")
            except (OSError, KeyError, TypeError):
                errors.append("tokenizer asset cannot be verified")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("verify")
    args = parser.parse_args(argv)
    if args.command == "build":
        binding, gate = build()
        _atomic_json(BINDING_OUTPUT, binding)
        _atomic_json(OUTPUT, gate)
    else:
        gate = json.loads(OUTPUT.read_text(encoding="utf-8"))
    errors = validate(gate, verify_files=True)
    print(json.dumps({"ok": not errors, "errors": errors,
                      "gate_sha256": gate.get("gate_sha256"),
                      "promotion_reviewed": gate.get("promotion_reviewed")},
                     indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TokenizerGateError,
            scaffold.ParallelScaffoldError) as exc:
        print(f"doctor_v5_gptoss_tokenizer_gate: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
