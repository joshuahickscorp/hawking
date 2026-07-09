#!/usr/bin/env python3.12
"""frontier_parity_runner.py - draft, sign, and verify frontier architecture parity receipts.

This is the signed integrity layer for `reports/condense/<LABEL>_parity.json`. The older
`frontier_parity.py` file remains the architecture readiness ledger; this file decides whether a parity
record is strong enough to unlock public quality/tok/s/RAM-cliff claims.

It does not run heavy reference backends on the laptop. It creates signed but blocked envelopes, and it
signs final measured parity records only after strict schema, command, trace, hash, threshold, and
family-feature checks pass.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "tools" / "condense"))

from studio_manifest import FRONTIER_MODELS, FrontierModel, frontier_by_label  # noqa: E402
import frontier_parity  # noqa: E402

SIGN_ALG = "sha256-json-v1"
SCHEMA = "hawking.frontier_parity.v1"
MAX_LOGIT_ABS_ERR_GATE = 1e-3
MIN_PROMPTS = 4
MIN_GREEDY_MATCH_TOKENS = 16
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
TOKENIZER_CONTRACT_SHA_KEYS = (
    "tokenizer_sha256",
    "chat_template_sha256",
    "special_tokens_sha256",
    "prompt_fixture_sha256",
)
CONTEXT_CONTRACT_TEXT_KEYS = (
    "rope_policy",
    "kv_cache_policy",
    "position_id_policy",
)
REQUIRED_PARITY_TESTS = ("logit_abs_error", "greedy_decode")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _git_commit(root: pathlib.Path = ROOT) -> str:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return p.stdout.strip() if p.returncode == 0 and p.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def _read_json(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        return json.load(open(path))
    except Exception:
        return None


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _canonical_digest(data: dict[str, Any]) -> str:
    unsigned = copy.deepcopy(data)
    unsigned.pop("signature", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _placeholder(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return not text or "<" in text or "TODO" in text or "..." in text


def _commands(record: dict[str, Any]) -> list[str]:
    out = []
    cmds = record.get("commands")
    if isinstance(cmds, list):
        out.extend(str(cmd) for cmd in cmds if cmd)
    if record.get("command"):
        out.append(str(record["command"]))
    return out


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.match(value))


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parity_path(root: pathlib.Path, label: str) -> pathlib.Path:
    model = frontier_by_label(label)
    if not model:
        raise ValueError(f"unknown frontier label: {label}")
    return pathlib.Path(frontier_parity.parity_status(model, root)["record"])


def _model_for_record(record: dict[str, Any], model: FrontierModel | None = None) -> FrontierModel | None:
    if model:
        return model
    label = record.get("model") or record.get("label")
    return frontier_by_label(str(label)) if label else None


def _family(model: FrontierModel | None) -> dict[str, Any]:
    if not model:
        return {"family": None, "required_native_features": []}
    return frontier_parity._family_spec(model)


def _verified_features(record: dict[str, Any]) -> set[str]:
    values = record.get("verified_native_features")
    if isinstance(values, list):
        return {str(v) for v in values}
    values = record.get("architecture_features_verified")
    if isinstance(values, list):
        return {str(v) for v in values}
    return set()


def _require_ref(record: dict[str, Any], key: str, problems: list[str]) -> None:
    if _placeholder(record.get(key)):
        problems.append(f"{key} missing or placeholder")


def _trace_pair_problems(record: dict[str, Any]) -> list[str]:
    problems = []
    if _placeholder(record.get("reference_trace")):
        problems.append("reference_trace missing or placeholder")
    if not _is_sha256(record.get("reference_trace_sha256")):
        problems.append("reference_trace_sha256 missing or invalid")

    native_trace = next(
        (record.get(key) for key in ("hawking_trace", "native_trace") if not _placeholder(record.get(key))),
        None,
    )
    native_sha = next(
        (record.get(key) for key in ("hawking_trace_sha256", "native_trace_sha256")
         if _is_sha256(record.get(key))),
        None,
    )
    if _placeholder(native_trace):
        problems.append("hawking_trace/native_trace missing or placeholder")
    if not _is_sha256(native_sha):
        problems.append("hawking_trace_sha256/native_trace_sha256 missing or invalid")
    return problems


def _tokenizer_contract_problems(record: dict[str, Any]) -> list[str]:
    problems = []
    contract = record.get("tokenizer_contract")
    if not isinstance(contract, dict):
        return ["tokenizer_contract missing or not an object"]
    if contract.get("tokenizer_sha256") != record.get("tokenizer_sha256"):
        problems.append("tokenizer_contract.tokenizer_sha256 must match tokenizer_sha256")
    for key in TOKENIZER_CONTRACT_SHA_KEYS:
        if not _is_sha256(contract.get(key)):
            problems.append(f"tokenizer_contract.{key} missing or invalid")
    return problems


def _context_contract_problems(record: dict[str, Any]) -> list[str]:
    problems = []
    contract = record.get("context_contract")
    if not isinstance(contract, dict):
        return ["context_contract missing or not an object"]
    if not _positive_int(contract.get("context_length")):
        problems.append("context_contract.context_length must be positive")
    for key in CONTEXT_CONTRACT_TEXT_KEYS:
        if _placeholder(contract.get(key)):
            problems.append(f"context_contract.{key} missing or placeholder")
    return problems


def _parity_test_problems(record: dict[str, Any]) -> list[str]:
    problems = []
    tests = record.get("parity_tests")
    if not isinstance(tests, list) or not tests:
        return ["parity_tests missing or empty"]
    by_name = {str(row.get("name")): row for row in tests if isinstance(row, dict)}
    for name in REQUIRED_PARITY_TESTS:
        row = by_name.get(name)
        if not row:
            problems.append(f"parity_tests.{name} missing")
            continue
        if row.get("status") != "pass":
            problems.append(f"parity_tests.{name}.status must be pass")
        if _placeholder(row.get("receipt")):
            problems.append(f"parity_tests.{name}.receipt missing or placeholder")
        if name == "logit_abs_error":
            err = row.get("max_abs_err", record.get("max_logit_abs_err"))
            if not _number(err) or err > MAX_LOGIT_ABS_ERR_GATE:
                problems.append(f"parity_tests.{name}.max_abs_err must be <= {MAX_LOGIT_ABS_ERR_GATE:g}")
        if name == "greedy_decode":
            matched = row.get("matched_tokens", record.get("greedy_match_tokens"))
            if not _positive_int(matched) or matched < MIN_GREEDY_MATCH_TOKENS:
                problems.append(f"parity_tests.{name}.matched_tokens must be >= {MIN_GREEDY_MATCH_TOKENS}")
    return problems


def _unsupported_exit_problems(record: dict[str, Any]) -> list[str]:
    problems = []
    exits = record.get("unsupported_by_design")
    if not isinstance(exits, list):
        return ["unsupported_by_design must be a list, even when empty"]
    for i, row in enumerate(exits):
        if not isinstance(row, dict):
            problems.append(f"unsupported_by_design[{i}] must be an object")
            continue
        for key in ("feature", "reason", "exit_receipt"):
            if _placeholder(row.get(key)):
                problems.append(f"unsupported_by_design[{i}].{key} missing or placeholder")
        if row.get("status") not in ("blocked", "unsupported_by_design"):
            problems.append(f"unsupported_by_design[{i}].status must be blocked or unsupported_by_design")
    return problems


def signature_status(record: dict[str, Any]) -> dict[str, Any]:
    sig = record.get("signature") if isinstance(record.get("signature"), dict) else {}
    expected = _canonical_digest(record)
    ok = sig.get("algorithm") == SIGN_ALG and sig.get("digest") == expected
    problems = []
    if sig.get("algorithm") != SIGN_ALG:
        problems.append(f"signature algorithm must be {SIGN_ALG}")
    if sig.get("digest") != expected:
        problems.append("signature digest mismatch")
    return {
        "ok": ok,
        "algorithm": sig.get("algorithm"),
        "digest": sig.get("digest"),
        "expected_digest": expected,
        "problems": problems,
    }


def _strict_problems(record: dict[str, Any], model: FrontierModel | None) -> list[str]:
    problems = []
    family = _family(model)
    label = model.label if model else str(record.get("model") or record.get("label") or "")

    if record.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA}")
    if not model:
        problems.append("model must match a frontier manifest label")
    else:
        if record.get("model") != model.label:
            problems.append("model label mismatch")
        if record.get("hf_id") != model.hf_id:
            problems.append("hf_id mismatch")
        if record.get("family") != family["family"]:
            problems.append(f"family must be {family['family']}")
    if record.get("receipt_state") != "final":
        problems.append("receipt_state must be final")
    if str(record.get("source") or "").lower() != "measured":
        problems.append("source must be measured")
    if record.get("status") != "pass":
        problems.append("status must be pass")
    if not record.get("reference_backend") or _placeholder(str(record.get("reference_backend"))):
        problems.append("reference_backend must be a concrete trusted backend")
    if not (record.get("git_commit") or record.get("hawking_commit")):
        problems.append("git_commit/hawking_commit missing")
    for key in ("tokenizer_sha256", "config_sha256"):
        if not _is_sha256(record.get(key)):
            problems.append(f"{key} is missing or invalid")
    prompt_count = record.get("prompt_count")
    if not _positive_int(prompt_count) or prompt_count < MIN_PROMPTS:
        problems.append(f"prompt_count must be >= {MIN_PROMPTS}")
    err = record.get("max_logit_abs_err")
    if not _number(err) or err > MAX_LOGIT_ABS_ERR_GATE:
        problems.append(f"max_logit_abs_err must be <= {MAX_LOGIT_ABS_ERR_GATE:g}")
    greedy = record.get("greedy_match_tokens")
    if not _positive_int(greedy) or greedy < MIN_GREEDY_MATCH_TOKENS:
        problems.append(f"greedy_match_tokens must be >= {MIN_GREEDY_MATCH_TOKENS}")

    cmds = _commands(record)
    if not cmds:
        problems.append("exact command(s) missing")
    elif any(_placeholder(cmd) for cmd in cmds):
        problems.append("command contains placeholder text")

    if _placeholder(record.get("reference_receipt")) and _placeholder(record.get("logits_receipt")):
        problems.append("reference_receipt or logits_receipt missing")
    if _placeholder(record.get("hawking_receipt")) and _placeholder(record.get("native_receipt")):
        problems.append("hawking_receipt or native_receipt missing")
    for key in ("architecture_adapter", "adapter_receipt", "tensor_map_receipt"):
        _require_ref(record, key, problems)
    if not _is_sha256(record.get("tensor_map_sha256")):
        problems.append("tensor_map_sha256 missing or invalid")
    problems.extend(_trace_pair_problems(record))
    problems.extend(_tokenizer_contract_problems(record))
    problems.extend(_context_contract_problems(record))
    problems.extend(_parity_test_problems(record))
    problems.extend(_unsupported_exit_problems(record))

    required = set(family.get("required_native_features") or [])
    verified = _verified_features(record)
    missing = sorted(required - verified)
    if missing:
        problems.append(f"required native feature(s) unverified for {label}: {', '.join(missing[:4])}")
    return problems


def record_status(record: dict[str, Any] | None, *, model: FrontierModel | None = None,
                  require_signature: bool = True) -> dict[str, Any]:
    if not record:
        return {
            "schema": "hawking.frontier_parity_receipt_status.v1",
            "ok": False,
            "model": model.label if model else None,
            "problems": ["record missing or unreadable"],
        }
    model = _model_for_record(record, model)
    problems = _strict_problems(record, model)
    sig = signature_status(record)
    if require_signature and not sig["ok"]:
        problems.extend(sig["problems"])
    return {
        "schema": "hawking.frontier_parity_receipt_status.v1",
        "ok": not problems,
        "model": model.label if model else record.get("model") or record.get("label"),
        "hf_id": model.hf_id if model else record.get("hf_id"),
        "receipt_state": record.get("receipt_state"),
        "signature": sig,
        "problems": problems,
    }


def sign_record(record: dict[str, Any], *, model: FrontierModel | None = None,
                allow_blocked_draft: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    signed = copy.deepcopy(record)
    signed.pop("signature", None)
    signed.setdefault("generated_at", _now())
    signed.setdefault("git_commit", _git_commit(ROOT))
    signed["signed_at"] = _now()
    status = record_status(signed, model=model, require_signature=False)
    if not status["ok"] and not allow_blocked_draft:
        return signed, status
    signed["signature"] = {"algorithm": SIGN_ALG, "digest": _canonical_digest(signed)}
    return signed, record_status(signed, model=model, require_signature=True)


def draft_record(model: FrontierModel, *, machine_class: str = "Studio-M1Ultra-128") -> dict[str, Any]:
    family = _family(model)
    return {
        "schema": SCHEMA,
        "model": model.label,
        "hf_id": model.hf_id,
        "family": family["family"],
        "generated_at": _now(),
        "git_commit": _git_commit(ROOT),
        "machine_class": machine_class,
        "receipt_state": "draft",
        "source": "TODO measured",
        "status": "TODO pass",
        "reference_backend": "<Transformers/vLLM/SGLang>",
        "tokenizer_sha256": "<64 hex>",
        "config_sha256": "<64 hex>",
        "prompt_count": f"TODO >= {MIN_PROMPTS}",
        "max_logit_abs_err": f"TODO <= {MAX_LOGIT_ABS_ERR_GATE:g}",
        "greedy_match_tokens": f"TODO >= {MIN_GREEDY_MATCH_TOKENS}",
        "commands": ["<exact reference command>", "<exact Hawking/native command>"],
        "reference_receipt": "<path>",
        "hawking_receipt": "<path>",
        "reference_trace": "<path>",
        "reference_trace_sha256": "<64 hex>",
        "hawking_trace": "<path>",
        "hawking_trace_sha256": "<64 hex>",
        "architecture_adapter": f"<project-native {family['family']} parity adapter>",
        "adapter_receipt": "<path>",
        "tensor_map_receipt": "<path>",
        "tensor_map_sha256": "<64 hex>",
        "tokenizer_contract": {
            "tokenizer_sha256": "<64 hex>",
            "chat_template_sha256": "<64 hex>",
            "special_tokens_sha256": "<64 hex>",
            "prompt_fixture_sha256": "<64 hex>",
        },
        "context_contract": {
            "context_length": "TODO positive int",
            "rope_policy": "<exact rope/position scaling policy>",
            "kv_cache_policy": "<exact KV cache policy>",
            "position_id_policy": "<exact position id policy>",
        },
        "parity_tests": [
            {
                "name": "logit_abs_error",
                "status": "TODO pass",
                "max_abs_err": f"TODO <= {MAX_LOGIT_ABS_ERR_GATE:g}",
                "receipt": "<path>",
            },
            {
                "name": "greedy_decode",
                "status": "TODO pass",
                "matched_tokens": f"TODO >= {MIN_GREEDY_MATCH_TOKENS}",
                "receipt": "<path>",
            },
        ],
        "unsupported_by_design": [
            {
                "feature": "<optional unsupported feature>",
                "status": "unsupported_by_design",
                "reason": "<why safe for this text-only parity target>",
                "exit_receipt": "<path>",
            }
        ],
        "required_native_features": family["required_native_features"],
        "verified_native_features": [],
    }


def _selected_models(labels: list[str]) -> list[FrontierModel]:
    if not labels:
        return list(FRONTIER_MODELS)
    out = []
    for label in labels:
        model = frontier_by_label(label)
        if not model:
            raise SystemExit(f"unknown frontier label: {label}")
        out.append(model)
    return out


def dispatch(args, root: pathlib.Path = ROOT) -> int:
    rows = []
    ok = True
    for model in _selected_models(args.label):
        path = parity_path(root, model.label)
        if getattr(args, "out_dir", ""):
            path = pathlib.Path(args.out_dir) / path.name
        if args.parity_mode == "draft":
            if path.exists() and not args.force:
                rows.append({"label": model.label, "path": str(path), "ok": False,
                             "problems": ["path exists; use --force to overwrite"]})
                ok = False
                continue
            record = draft_record(model, machine_class=args.machine_class)
            if args.sign_draft:
                record, status = sign_record(record, model=model, allow_blocked_draft=True)
            else:
                status = record_status(record, model=model, require_signature=False)
            _write_json(path, record)
        elif args.parity_mode == "sign":
            record = _read_json(path)
            record, status = sign_record(record or {}, model=model,
                                         allow_blocked_draft=args.allow_blocked_draft)
            if _read_json(path):
                _write_json(path, record)
        elif args.parity_mode == "verify":
            record = _read_json(path)
            status = record_status(record, model=model, require_signature=True)
        else:
            raise SystemExit(f"unknown parity mode: {args.parity_mode}")
        rows.append({"label": model.label, "path": str(path), "ok": status["ok"],
                     "problems": status["problems"]})
        ok = ok and status["ok"]
    result = {
        "schema": "hawking.frontier_parity_receipt_run.v1",
        "mode": args.parity_mode,
        "ok": ok,
        "rows": rows,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"# frontier parity receipts {args.parity_mode}: {'OK' if ok else 'BLOCKED'}")
        for row in rows:
            print(f"{row['label'][:18]:18s} {'OK' if row['ok'] else 'BLOCK':6s} {row['path']}")
            for problem in row["problems"][:6]:
                print(f"  - {problem}")
    return 0 if ok else 1


def _complete_record(model: FrontierModel) -> dict[str, Any]:
    family = _family(model)
    return {
        "schema": SCHEMA,
        "model": model.label,
        "hf_id": model.hf_id,
        "family": family["family"],
        "receipt_state": "final",
        "source": "measured",
        "machine_class": "Studio-M1Ultra-128",
        "status": "pass",
        "reference_backend": "selftest-transformers",
        "tokenizer_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "prompt_count": MIN_PROMPTS,
        "max_logit_abs_err": 0.0,
        "greedy_match_tokens": MIN_GREEDY_MATCH_TOKENS,
        "commands": ["selftest reference parity", "selftest hawking parity"],
        "git_commit": "selftest",
        "reference_receipt": "selftest://reference-logits",
        "hawking_receipt": "selftest://hawking-logits",
        "reference_trace": "selftest://reference-trace",
        "reference_trace_sha256": "c" * 64,
        "hawking_trace": "selftest://hawking-trace",
        "hawking_trace_sha256": "d" * 64,
        "architecture_adapter": f"selftest::{family['family']}::Adapter",
        "adapter_receipt": "selftest://adapter",
        "tensor_map_receipt": "selftest://tensor-map",
        "tensor_map_sha256": "e" * 64,
        "tokenizer_contract": {
            "tokenizer_sha256": "a" * 64,
            "chat_template_sha256": "f" * 64,
            "special_tokens_sha256": "1" * 64,
            "prompt_fixture_sha256": "2" * 64,
        },
        "context_contract": {
            "context_length": 4096,
            "rope_policy": "selftest rope policy",
            "kv_cache_policy": "selftest KV cache policy",
            "position_id_policy": "selftest position id policy",
        },
        "parity_tests": [
            {
                "name": "logit_abs_error",
                "status": "pass",
                "max_abs_err": 0.0,
                "receipt": "selftest://logit-test",
            },
            {
                "name": "greedy_decode",
                "status": "pass",
                "matched_tokens": MIN_GREEDY_MATCH_TOKENS,
                "receipt": "selftest://greedy-test",
            },
        ],
        "unsupported_by_design": [],
        "required_native_features": family["required_native_features"],
        "verified_native_features": family["required_native_features"],
    }


def selftest() -> bool:
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    model = FRONTIER_MODELS[0]
    draft = draft_record(model)
    draft_signed, draft_status = sign_record(draft, model=model, allow_blocked_draft=True)
    check("signed parity draft stays blocked", not draft_status["ok"] and draft_signed.get("signature"))
    complete_signed, complete_status = sign_record(_complete_record(model), model=model)
    check("complete parity signs and verifies", complete_status["ok"])
    complete_signed["max_logit_abs_err"] = 1.0
    check("tampered parity signature fails", not record_status(complete_signed, model=model)["ok"])
    missing_trace = _complete_record(model)
    missing_trace.pop("reference_trace")
    _, missing_status = sign_record(missing_trace, model=model)
    check("parity without reference trace hash is blocked", not missing_status["ok"])
    missing_contract = _complete_record(model)
    missing_contract.pop("tensor_map_sha256")
    _, contract_status = sign_record(missing_contract, model=model)
    check("parity without tensor map contract is blocked", not contract_status["ok"])
    loose = _complete_record(model)
    loose["max_logit_abs_err"] = MAX_LOGIT_ABS_ERR_GATE * 10
    _, loose_status = sign_record(loose, model=model)
    check("loose logit parity is blocked", not loose_status["ok"])

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        out_dir = root / "reports" / "condense"
        args = argparse.Namespace(parity_mode="draft", label=[model.label], out_dir=str(out_dir),
                                  force=True, sign_draft=True,
                                  machine_class="Studio-M1Ultra-128", json=True)
        check("draft command writes blocked parity receipt", dispatch(args, root=root) == 1)
        check("draft parity exists", (out_dir / f"{model.label}_parity.json").exists())
    print(f"\n# SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


def cmd_selftest(args) -> int:
    return 0 if selftest() else 1


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Draft/sign/verify signed frontier parity receipts.")
    sub = ap.add_subparsers(dest="parity_mode")
    for mode in ("draft", "sign", "verify"):
        p = sub.add_parser(mode, help=f"{mode} signed parity receipts")
        p.add_argument("label", nargs="*", help="frontier label(s); default all")
        p.add_argument("--out-dir", default="")
        p.add_argument("--json", action="store_true")
        if mode == "draft":
            p.add_argument("--force", action="store_true")
            p.add_argument("--sign-draft", action="store_true")
            p.add_argument("--machine-class", default="Studio-M1Ultra-128")
            p.set_defaults(allow_blocked_draft=False)
        else:
            p.set_defaults(force=False, sign_draft=False, machine_class="Studio-M1Ultra-128")
        if mode == "sign":
            p.add_argument("--allow-blocked-draft", action="store_true")
        else:
            p.set_defaults(allow_blocked_draft=False)
        p.set_defaults(func=dispatch)
    p = sub.add_parser("selftest", help="synthetic signed parity receipt tests")
    p.set_defaults(func=cmd_selftest)
    return ap


def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()
    if not args.parity_mode:
        args = ap.parse_args(["verify"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
