"""Hosted GLM-5.2 behavior capture (outputs-only).

Captures generation trajectories and, when the provider returns them, top-k
logprobs. Never fabricates tokens, logits, or activations. Live capture fails
closed when no API credential is configured.

Identity note: hosted ``model`` strings are provider-claimed. This harness pins
the *intended* open-weight identity (zai-org/GLM-5.2@b4734de4) for provenance,
and records the provider model id actually sent on the wire. Byte-equality with
the HF revision is not claimed.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from lab.layout import evidence_dir
from lab.operators.glm52_adapter import IMMUTABLE_REVISION, REPO_ID
from lab.operators.glm52_behavior_access import HOSTED_PROVIDERS
from lab.operators.glm52_common import (
    Glm52Error,
    atomic_json,
    canonical,
    seal,
    utc_now,
    verify_sealed,
)

SCHEMA_TRAJECTORY = "hawking.glm52.behavior_trajectory.v1"
SCHEMA_BUNDLE = "hawking.glm52.behavior_capture_bundle.v1"
SCHEMA_PREFLIGHT = "hawking.glm52.behavior_capture_preflight.v1"
SCHEMA_RECEIPT = "hawking.glm52.behavior_capture_receipt.v1"

GLM52_EVIDENCE = evidence_dir("glm52")
DEFAULT_OUT_DIR = Path(
    os.environ.get(
        "GLM52_BEHAVIOR_CAPTURE_DIR",
        str(
            Path.home()
            / "Library/Application Support/Hawking/GLM52Gravity/behavior_capture"
        ),
    )
)

# Default top-k request. Providers may ignore or cap this.
DEFAULT_TOP_LOGPROBS = 20
DEFAULT_MAX_TOKENS = 64
DEFAULT_TEMPERATURE = 0.0


class BehaviorCaptureError(Glm52Error):
    """Raised when behavior evidence cannot be captured honestly."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_provider(provider_id: str | None = None) -> dict[str, Any]:
    """Pick an explicit provider or the first with a live credential."""
    if provider_id:
        for row in HOSTED_PROVIDERS:
            if row["id"] == provider_id:
                env_hit = next((n for n in row["api_key_env"] if os.environ.get(n)), None)
                return {
                    **row,
                    "api_key": os.environ[env_hit] if env_hit else None,
                    "credential_env_hit": env_hit,
                }
        raise BehaviorCaptureError(f"unknown provider id: {provider_id!r}")
    for row in HOSTED_PROVIDERS:
        for name in row["api_key_env"]:
            key = os.environ.get(name)
            if key:
                return {**row, "api_key": key, "credential_env_hit": name}
    raise BehaviorCaptureError(
        "no hosted GLM-5.2 API credential found; set one of "
        "ZHIPU_API_KEY, BIGMODEL_API_KEY, ZAI_API_KEY, OPENROUTER_API_KEY, "
        "TOGETHER_API_KEY, FIREWORKS_API_KEY (or pass --dry-run)"
    )


def default_prompt_records(*, limit: int = 8) -> list[dict[str, Any]]:
    """Small sealed-ish prompt set from the existing capture corpus when possible.

    Falls back to a tiny built-in math/code set so the harness is usable without
    loading the full tokenizer (which may be multi-tens of MB).
    """
    records: list[dict[str, Any]] = []
    try:
        from lab.operators import glm52_capture_program as program

        # Hosted chat needs text prompts; token-id batches are local-only evidence.
        chosen = program.records_for("teacher_holdout")[:limit]
        for record in chosen:
            text = record.context_window if record.context_rung_tokens else record.prompt
            records.append(
                {
                    "record_id": record.record_id,
                    "domain": record.domain,
                    "partition": record.partition,
                    "prompt": text,
                    "prompt_sha256": _sha256_text(text),
                    "source": "glm52_capture_program.teacher_holdout",
                }
            )
        if records:
            return records
    except Exception:  # noqa: BLE001 - fail open to builtins; never fake a trajectory
        pass

    builtins = [
        ("builtin.math.01", "mathematics", "What is 17 * 19? Show only the integer."),
        ("builtin.math.02", "mathematics", "Simplify: (x^2 - 1)/(x - 1) for x != 1."),
        ("builtin.code.01", "coding", "Write a Python function is_palindrome(s: str) -> bool."),
        ("builtin.reason.01", "reasoning", "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?"),
        ("builtin.fact.01", "factual", "What is the atomic number of carbon?"),
        ("builtin.prose.01", "general prose", "In one sentence, define recursion."),
        ("builtin.science.01", "science", "State Ohm's law relating V, I, and R."),
        ("builtin.tool.01", "tool formatting", "Return JSON only: {\"ok\": true, \"n\": 3}"),
    ]
    for record_id, domain, prompt in builtins[:limit]:
        records.append(
            {
                "record_id": record_id,
                "domain": domain,
                "partition": "builtin_calibration",
                "prompt": prompt,
                "prompt_sha256": _sha256_text(prompt),
                "source": "builtin_fail_open_set",
            }
        )
    return records


def preflight(
    *,
    provider_id: str | None = None,
    limit: int = 8,
    require_credential: bool = False,
) -> dict[str, Any]:
    """Seal a preflight receipt. Does not call the network."""
    prompts = default_prompt_records(limit=limit)
    credential_error: str | None = None
    provider_view: dict[str, Any] | None = None
    try:
        resolved = resolve_provider(provider_id)
        provider_view = {
            "id": resolved["id"],
            "base_url": resolved["base_url"],
            "model_ids": list(resolved["model_ids"]),
            "first_party": resolved["first_party"],
            "credential_env_hit": resolved.get("credential_env_hit"),
            "credential_present": bool(resolved.get("api_key")),
            "notes": resolved["notes"],
        }
    except BehaviorCaptureError as exc:
        credential_error = str(exc)
        if require_credential:
            raise

    payload = {
        "schema": SCHEMA_PREFLIGHT,
        "built_at": utc_now(),
        "repo": REPO_ID,
        "revision": IMMUTABLE_REVISION,
        "intended_open_weight_identity": {
            "repo": REPO_ID,
            "revision": IMMUTABLE_REVISION,
            "byte_equality_with_hosted_not_claimed": True,
        },
        "provider": provider_view,
        "credential_error": credential_error,
        "prompt_count": len(prompts),
        "prompt_membership_sha256": hashlib.sha256(
            canonical({"prompts": [{"record_id": r["record_id"], "prompt_sha256": r["prompt_sha256"]} for r in prompts]})
        ).hexdigest(),
        "prompts": [
            {
                "record_id": r["record_id"],
                "domain": r["domain"],
                "partition": r["partition"],
                "prompt_sha256": r["prompt_sha256"],
                "source": r["source"],
                "prompt_chars": len(r["prompt"]),
            }
            for r in prompts
        ],
        "capture_contract": {
            "supplies": ["assistant text", "token trajectory when returned", "top-k logprobs when returned"],
            "never_supplies": [
                "hidden states",
                "router logits",
                "IndexShare selections",
                "MoE expert activations",
            ],
            "fail_closed": [
                "no API credential",
                "HTTP error",
                "empty choices",
                "fabricating logprobs",
            ],
            "temperature_default": DEFAULT_TEMPERATURE,
            "max_tokens_default": DEFAULT_MAX_TOKENS,
            "top_logprobs_default": DEFAULT_TOP_LOGPROBS,
        },
        "status": (
            "READY_FOR_LIVE_CAPTURE"
            if provider_view and provider_view.get("credential_present")
            else "DRY_RUN_ONLY_NO_CREDENTIAL"
        ),
    }
    return seal(payload)


def _chat_completion_request(
    *,
    provider: dict[str, Any],
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_logprobs: int,
    timeout_s: float,
) -> dict[str, Any]:
    """POST /chat/completions. Returns the raw JSON object or raises."""
    model = provider["model_ids"][0]
    url = provider["base_url"].rstrip("/") + "/chat/completions"
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }
    # OpenAI-style logprobs. Providers that do not support this may 400;
    # caller may retry without.
    if top_logprobs > 0:
        body["logprobs"] = True
        body["top_logprobs"] = int(top_logprobs)

    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {provider['api_key']}",
            "User-Agent": "hawking-glm52-behavior-capture/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
            status = getattr(response, "status", 200)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        # Retry once without logprobs if the provider rejects them.
        if top_logprobs > 0 and exc.code in {400, 422}:
            return _chat_completion_request(
                provider=provider,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_logprobs=0,
                timeout_s=timeout_s,
            ) | {"_logprobs_rejected": True, "_logprobs_reject_detail": detail}
        raise BehaviorCaptureError(
            f"HTTP {exc.code} from {provider['id']}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise BehaviorCaptureError(f"network error contacting {provider['id']}: {exc}") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise BehaviorCaptureError(f"non-JSON response from {provider['id']}") from exc
    if not isinstance(payload, dict):
        raise BehaviorCaptureError("chat completion root is not an object")
    payload["_http_status"] = status
    payload["_request_model"] = model
    payload["_request_url"] = url
    return payload


def _extract_trajectory(
    response: dict[str, Any],
    *,
    record: dict[str, Any],
    provider: dict[str, Any],
    top_logprobs_requested: int,
) -> dict[str, Any]:
    """Normalize a chat completion into a sealed trajectory record.

    Never invents tokens. Missing top-k is recorded as absent, not zero-filled.
    """
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise BehaviorCaptureError("completion has no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise BehaviorCaptureError("choice is not an object")

    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    text = message.get("content")
    if text is None:
        text = choice.get("text")
    if not isinstance(text, str):
        raise BehaviorCaptureError("completion choice has no text content")

    # OpenAI-style logprobs: choice.logprobs.content[].top_logprobs
    topk_steps: list[dict[str, Any]] | None = None
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        content = logprobs.get("content")
        if isinstance(content, list) and content:
            topk_steps = []
            for step in content:
                if not isinstance(step, dict):
                    continue
                token = step.get("token")
                token_logprob = step.get("logprob")
                top_list = step.get("top_logprobs")
                tops: list[dict[str, Any]] = []
                if isinstance(top_list, list):
                    for item in top_list:
                        if not isinstance(item, dict):
                            continue
                        if "token" in item and "logprob" in item:
                            tops.append(
                                {
                                    "token": item["token"],
                                    "logprob": item["logprob"],
                                }
                            )
                topk_steps.append(
                    {
                        "token": token,
                        "logprob": token_logprob,
                        "top_logprobs": tops,
                    }
                )

    finish_reason = choice.get("finish_reason")
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}

    trajectory = {
        "schema": SCHEMA_TRAJECTORY,
        "captured_at": utc_now(),
        "repo": REPO_ID,
        "revision": IMMUTABLE_REVISION,
        "record_id": record["record_id"],
        "domain": record["domain"],
        "partition": record["partition"],
        "prompt_sha256": record["prompt_sha256"],
        "prompt_source": record["source"],
        "provider": {
            "id": provider["id"],
            "base_url": provider["base_url"],
            "request_model": response.get("_request_model"),
            "first_party": provider["first_party"],
            "credential_env_hit": provider.get("credential_env_hit"),
        },
        "response": {
            "text": text,
            "text_sha256": _sha256_text(text),
            "finish_reason": finish_reason,
            "usage": usage,
        },
        "top_k": {
            "requested": top_logprobs_requested,
            "present": topk_steps is not None,
            "steps": topk_steps,
            "logprobs_rejected_by_provider": bool(response.get("_logprobs_rejected")),
            "note": (
                None
                if topk_steps is not None
                else "provider did not return per-token top_logprobs; trajectory text still sealed"
            ),
        },
        "activations_captured": False,
        "capability_claim_permitted": False,
        "weight_byte_equality_claimed": False,
    }
    return seal(trajectory)


def capture_live(
    *,
    provider_id: str | None = None,
    limit: int = 8,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_logprobs: int = DEFAULT_TOP_LOGPROBS,
    timeout_s: float = 120.0,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Capture real trajectories. Fails closed without credential or on HTTP error."""
    provider = resolve_provider(provider_id)
    if not provider.get("api_key"):
        raise BehaviorCaptureError("provider resolved without api_key")

    prompts = default_prompt_records(limit=limit)
    trajectories: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for record in prompts:
        try:
            response = _chat_completion_request(
                provider=provider,
                prompt=record["prompt"],
                max_tokens=max_tokens,
                temperature=temperature,
                top_logprobs=top_logprobs,
                timeout_s=timeout_s,
            )
            traj = _extract_trajectory(
                response,
                record=record,
                provider=provider,
                top_logprobs_requested=top_logprobs,
            )
            trajectories.append(traj)
        except BehaviorCaptureError as exc:
            errors.append(
                {
                    "record_id": record["record_id"],
                    "prompt_sha256": record["prompt_sha256"],
                    "error": str(exc),
                }
            )

    if not trajectories:
        raise BehaviorCaptureError(
            f"live capture produced zero trajectories; errors={errors!r}"
        )

    bundle = seal(
        {
            "schema": SCHEMA_BUNDLE,
            "captured_at": utc_now(),
            "repo": REPO_ID,
            "revision": IMMUTABLE_REVISION,
            "provider": {
                "id": provider["id"],
                "base_url": provider["base_url"],
                "model_ids": list(provider["model_ids"]),
                "first_party": provider["first_party"],
                "credential_env_hit": provider.get("credential_env_hit"),
            },
            "request": {
                "limit": limit,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_logprobs": top_logprobs,
            },
            "trajectory_count": len(trajectories),
            "error_count": len(errors),
            "errors": errors,
            "trajectories": trajectories,
            "activations_captured": False,
            "capability_claim_permitted": False,
            "weight_byte_equality_claimed": False,
            "status": "LIVE_CAPTURE_PARTIAL" if errors else "LIVE_CAPTURE_OK",
        }
    )

    target_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = target_dir / f"behavior_bundle_{bundle['seal_sha256'][:16]}.json"
    atomic_json(bundle_path, bundle)

    receipt = seal(
        {
            "schema": SCHEMA_RECEIPT,
            "captured_at": utc_now(),
            "repo": REPO_ID,
            "revision": IMMUTABLE_REVISION,
            "mode": "live",
            "bundle_path": str(bundle_path),
            "bundle_seal_sha256": bundle["seal_sha256"],
            "trajectory_count": bundle["trajectory_count"],
            "error_count": bundle["error_count"],
            "provider_id": provider["id"],
            "status": bundle["status"],
            "activations_captured": False,
            "capability_claim_permitted": False,
        }
    )
    receipt_path = target_dir / f"behavior_receipt_{receipt['seal_sha256'][:16]}.json"
    atomic_json(receipt_path, receipt)
    # Also drop a pointer under campaign evidence (small receipt only).
    evidence_receipt = GLM52_EVIDENCE / "GLM52_BEHAVIOR_CAPTURE_LATEST.json"
    atomic_json(evidence_receipt, receipt)
    return {
        "receipt": receipt,
        "receipt_path": str(receipt_path),
        "bundle_path": str(bundle_path),
        "evidence_receipt_path": str(evidence_receipt),
    }


def run_dry_run(
    *,
    provider_id: str | None = None,
    limit: int = 8,
    write_evidence: bool = True,
) -> dict[str, Any]:
    """Preflight only — no network, no fabricated trajectories."""
    pre = preflight(provider_id=provider_id, limit=limit, require_credential=False)
    receipt = seal(
        {
            "schema": SCHEMA_RECEIPT,
            "captured_at": utc_now(),
            "repo": REPO_ID,
            "revision": IMMUTABLE_REVISION,
            "mode": "dry_run",
            "preflight_seal_sha256": pre["seal_sha256"],
            "preflight_status": pre["status"],
            "trajectory_count": 0,
            "error_count": 0,
            "status": "DRY_RUN_NO_TRAJECTORIES",
            "activations_captured": False,
            "capability_claim_permitted": False,
            "honest_note": (
                "Dry-run seals prompt membership and provider gates only. "
                "Zero trajectories by construction; live capture requires an API key."
            ),
        }
    )
    paths: dict[str, str] = {}
    if write_evidence:
        pre_path = GLM52_EVIDENCE / "GLM52_BEHAVIOR_CAPTURE_PREFLIGHT.json"
        rec_path = GLM52_EVIDENCE / "GLM52_BEHAVIOR_CAPTURE_DRY_RUN.json"
        atomic_json(pre_path, pre)
        atomic_json(rec_path, receipt)
        paths = {
            "preflight_path": str(pre_path),
            "receipt_path": str(rec_path),
        }
    return {"preflight": pre, "receipt": receipt, **paths}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="preflight only; no network")
    parser.add_argument("--live", action="store_true", help="call hosted endpoint (needs API key)")
    parser.add_argument("--provider", default=None, help="provider id from HOSTED_PROVIDERS")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-logprobs", type=int, default=DEFAULT_TOP_LOGPROBS)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    if args.live and args.dry_run:
        raise SystemExit("pass only one of --live / --dry-run")
    if not args.live and not args.dry_run:
        args.dry_run = True

    if args.dry_run:
        result = run_dry_run(provider_id=args.provider, limit=args.limit)
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "status": result["receipt"]["status"],
                    "preflight_status": result["preflight"]["status"],
                    "preflight_seal_sha256": result["preflight"]["seal_sha256"],
                    "receipt_seal_sha256": result["receipt"]["seal_sha256"],
                    "paths": {
                        k: result[k]
                        for k in ("preflight_path", "receipt_path")
                        if k in result
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    result = capture_live(
        provider_id=args.provider,
        limit=args.limit,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_logprobs=args.top_logprobs,
        timeout_s=args.timeout,
        out_dir=args.out_dir,
    )
    print(
        json.dumps(
            {
                "mode": "live",
                "status": result["receipt"]["status"],
                "trajectory_count": result["receipt"]["trajectory_count"],
                "error_count": result["receipt"]["error_count"],
                "bundle_path": result["bundle_path"],
                "receipt_path": result["receipt_path"],
                "seal_sha256": result["receipt"]["seal_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
