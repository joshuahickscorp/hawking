#!/usr/bin/env python3
"""Q1 — can any available body emit the choice JSON this scheduler requires?

receipts/future/MODEL_BEARING_TORTURE_30M.json (3fee807a8) failed twice because
choose() turns produced markdown, never {"choice_id","reason"}. This probe is
the cheapest discriminator: exact trial ask, one lever at a time, then one
other sealed body.

Three named outcomes:
  ASK_WRONG          incumbent emits the object under a different ask
  INCUMBENT_CANNOT   a different sealed specimen emits it; the 27B body does not
  CONTRACT_UNUSABLE  nothing available emits the object

Not a prompt-engineering campaign. One pass. Hardware-named fields refused.
Does not write the autonomy-trial module. Parks on the GPU lane lock.

    python3 tools/future/choice_json_probe.py --selftest
    python3 tools/future/choice_json_probe.py --run
    python3 -m pytest tools/future/test_choice_json_probe.py -q
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import REPO, git, write_receipt
from tools.future import model_bearing as mb
from tools.future import model_bearing_torture as mbt

RECEIPT = "CHOICE_JSON_PROBE.json"
SCHEMA = "hawking.future.choice_json_probe.v1"
RECORDED_BY = "tools/future/choice_json_probe.py"
VERSION = 1

# THE HISTORICAL TIMELINE, FROM A PATH NOTHING WRITES.
#
# This probe reconstructs the ask that failed 0 of 43. It used to read the LIVE
# receipts/future/MODEL_BEARING_TIMELINE.json, which every subsequent run
# overwrites - so once the ask was FIXED and the resident went 51 of 51, this
# probe's assertions failed for the one reason a control never should: the
# artifact it exists to describe had been replaced by the thing it caused.
# Same trap, same repair, as the archived 477 s autonomy control.
TIMELINE_REL = "receipts/future/controls/MODEL_BEARING_TIMELINE_ARCHIVED_43turn.json"
TIMELINE_ARCHIVE_COMMIT = "3fee807a8"
LIVE_TIMELINE_REL = "receipts/future/MODEL_BEARING_TIMELINE.json"
TORTURE_REL = "receipts/future/MODEL_BEARING_TORTURE_30M.json"

# Landed trial control. Reconstruction must match this or the probe is answering
# a different question.
TIMELINE_CHOOSE_PROMPT_SHA256 = "31789943e9e699468140b0f5d97f7ea54e22782507c0c51b4c7a262a5cf0cb87"
TIMELINE_CHOOSE_REPLY_SHA256 = "ec9a514b40f32ca28d618dd882378d776ecd1b84adec72b86296f8d094f30b02"

# Copied from choose() in model_bearing.py. A test proves the source still
# contains this exact tail so we cannot silently paraphrase.
CHOOSE_PREAMBLE_PREFIX = "Pick one candidate. The scripted policy would pick "
CHOOSE_SCHEMA_TAIL_A = (
    '{"choice_id":"id","reason":"why this, citing a real difference","mechanism":"...",'
)
CHOOSE_SCHEMA_TAIL_B = '"surface":"...","hypothesis_family":"..."}'
CHOOSE_SCHEMA_TAIL = CHOOSE_SCHEMA_TAIL_A + CHOOSE_SCHEMA_TAIL_B

JSON_ONLY_INSTRUCTION = (
    "Return JSON only, no markdown, no prose, no fences. The entire reply must "
    "be one object: " + CHOOSE_SCHEMA_TAIL + "."
)

ONE_SHOT_EXAMPLE = (
    '{"choice_id":"WU.HAWKING.resident_identity_pin",'
    '"reason":"live Hawking-self work; the scripted policy pick; not a closed scar",'
    '"mechanism":"fusion-env identity pin","surface":"hawking.resident",'
    '"hypothesis_family":"resident_identity_pin"}'
)

SMALLER_IDS: tuple[str, ...] = (
    "WU.HAWKING.resident_identity_pin",
    "WU.HAWKING.fusion_env_applied",
)

INCUMBENT_BODY = "sealed-3.14"
QWEN06_BODY = "Qwen3-0.6B"
QWEN06_REPO = "Qwen/Qwen3-0.6B"
QWEN06_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
QWEN06_SPECIMEN = Path(
    "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3-0.6B@c1899de289a0"
)
MLX_PYTHON = Path("/Users/scammermike/.local/share/uv/tools/mlx-lm/bin/python")

# Greedy, matching ConnectorProvider.ask / sealed generation. Two repeats is
# enough to confirm determinism; the trial's 43 choose turns were one hash.
N_REPEATS = 2
ASK_TIMEOUT_S = mbt.ASK_TIMEOUT_S
READY_TIMEOUT_S = mbt.READY_TIMEOUT_S
MAX_ASK_TOKENS = mbt.MAX_ASK_TOKENS

OUTCOME_ASK = "ASK_WRONG"
OUTCOME_BODY = "INCUMBENT_CANNOT"
OUTCOME_NOTHING = "CONTRACT_UNUSABLE"
OUTCOME_DISCREPANCY = "TRIAL_DID_NOT_REPRODUCE"

CLAIM_BOUNDARY = (
    "Static sidecar artifact plus SELF_MEASURED_DIRTY process telemetry "
    "(pid, token counts, flock wait). No hardware measurement. Token counts "
    "are protocol fields, not throughput. Parse-rate is a fraction of raw "
    "replies, not a score."
)

# Native complete_payload request keys. A test greps hawking_native.py for this.
NATIVE_REQUEST_KEYS: tuple[str, ...] = ("id", "prompt", "max_new_tokens", "max_seq_len")
GRAMMAR_PAYLOAD_KEYS: tuple[str, ...] = (
    "grammar",
    "response_format",
    "json_schema",
    "guided_json",
    "guided_grammar",
    "schema",
)


class ProbeRefused(ValueError):
    def __init__(self, reason: str, *, missing: list[str] | None = None) -> None:
        self.reason = reason
        self.missing = list(missing or [])
        super().__init__(reason)


# ---------------------------------------------------------------------------
# Exact ask. Same constructor as model_bearing.choose, then the same clip.
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


def clip_ask(prompt: str) -> str:
    """The TRIAL-ERA clip, kept verbatim so the reconstruction still hashes.

    model_bearing no longer clips this way - this probe is why. Do not repoint
    this at the live clip: the sha256 that ties the reconstruction to the 43
    recorded choose turns is a hash of the ask as it WAS asked.
    """
    if len(prompt) <= mb.MAX_PROMPT_CHARS:
        return prompt
    return prompt[: mb.MAX_PROMPT_CHARS - 1] + "…"


def compact_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(c) for c in candidates if isinstance(c, Mapping) and mb._cid(c)]
    return mb._compact_entries(rows, cap=max(mb.PROMPT_ENTRY_CAP, len(rows)))


def choose_prompt_unclipped(candidates: Sequence[Mapping[str, Any]]) -> str:
    """The exact choose() ask, before MAX_PROMPT_CHARS. Do not paraphrase."""
    rows = [dict(c) for c in candidates if isinstance(c, Mapping) and mb._cid(c)]
    policy = mb.fixed_policy_choose(rows)
    compact = compact_candidates(rows)
    return (
        CHOOSE_PREAMBLE_PREFIX
        + json.dumps(policy.get("id"))
        + ".\nCandidates:\n"
        + json.dumps(compact, sort_keys=True)
        + "\nReturn JSON only: "
        + CHOOSE_SCHEMA_TAIL
    )


def control_prompt(candidates: Sequence[Mapping[str, Any]] | None = None) -> str:
    return clip_ask(choose_prompt_unclipped(candidates or mbt.live_catalog()))


def json_only_prompt(candidates: Sequence[Mapping[str, Any]] | None = None) -> str:
    """ONE lever from control: JSON-only instruction with no prose room.

    Placed first so the 1800-char clip cannot eat it. Candidates and clip
    budget are otherwise the control's.
    """
    rows = list(candidates or mbt.live_catalog())
    policy = mb.fixed_policy_choose(rows)
    compact = compact_candidates(rows)
    unclipped = (
        JSON_ONLY_INSTRUCTION
        + "\n"
        + CHOOSE_PREAMBLE_PREFIX
        + json.dumps(policy.get("id"))
        + ".\nCandidates:\n"
        + json.dumps(compact, sort_keys=True)
    )
    return clip_ask(unclipped)


def one_shot_prompt(candidates: Sequence[Mapping[str, Any]] | None = None) -> str:
    """ONE lever from control: a correct-reply example, prepended, then clip."""
    unclipped = (
        "Example of a correct reply:\n"
        + ONE_SHOT_EXAMPLE
        + "\n"
        + choose_prompt_unclipped(candidates or mbt.live_catalog())
    )
    return clip_ask(unclipped)


def smaller_catalog(candidates: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows = list(candidates or mbt.live_catalog())
    wanted = {str(i) for i in SMALLER_IDS}
    small = [dict(c) for c in rows if str(c.get("id") or "") in wanted]
    if len(small) < 2:
        raise ProbeRefused("smaller_choice_set needs the two live Hawking-self ids")
    return small


def smaller_choice_set_prompt(candidates: Sequence[Mapping[str, Any]] | None = None) -> str:
    """ONE lever from control: two live candidates so the schema survives clip."""
    return clip_ask(choose_prompt_unclipped(smaller_catalog(candidates)))


def parse_choice_json(text: str) -> dict[str, Any]:
    """Parseable means _extract_json plus a non-empty choice_id and reason.

    That is the object the scheduler admits. Markdown that names an id is not
    it. Interpret-shaped JSON ({reading, why}) is not it.
    """
    parsed = mb._extract_json(text or "")
    if not isinstance(parsed, dict):
        return {
            "parse_ok": False,
            "choice_id": None,
            "reason": None,
            "parsed": None,
            "why": "reply was not a JSON object",
        }
    choice_id = mb._field(parsed, "choice_id", "id")
    reason = mb._reason_of(parsed, "reason", "why")
    ok = bool(choice_id and reason)
    return {
        "parse_ok": ok,
        "choice_id": choice_id or None,
        "reason": reason or None,
        "parsed": parsed,
        "why": None if ok else (
            "JSON object missing choice_id"
            if not choice_id
            else "JSON object missing reason"
        ),
    }


def json_instruction_reached(prompt: str) -> bool:
    return "choice_id" in prompt and "Return JSON only" in prompt


def fraction(k: int, n: int) -> str:
    return f"{int(k)} of {int(n)}"


# ---------------------------------------------------------------------------
# Citations. The trial ask is reconstructed, then hashed against the timeline.
# ---------------------------------------------------------------------------


def choose_source_contains_literals() -> dict[str, Any]:
    src = Path(mb.__file__).read_text(encoding="utf-8")
    return {
        "path": "tools/future/model_bearing.py",
        "preamble_present": CHOOSE_PREAMBLE_PREFIX in src,
        "schema_tail_present": CHOOSE_SCHEMA_TAIL_A in src and CHOOSE_SCHEMA_TAIL_B in src,
        "max_prompt_chars": mb.MAX_PROMPT_CHARS,
        "max_prompt_chars_literal": "MAX_PROMPT_CHARS = 1800" in src,
        # Was: does the live clip still eat the tail? It did, and that was the
        # whole finding - schema off the end, 0 of 2 on a 27B AND on a 0.6B.
        # Now these two report which regime the live code is in.
        "clip_eats_tail": 'prompt[: MAX_PROMPT_CHARS - 1] + "…"' in src,
        "clip_keeps_tail": "_clip_keeping_tail" in src and "SCHEMA_TAIL_RESERVE" in src,
        "fits_by_dropping_candidates": "_fit_entries" in src,
    }


def extract_exact_ask() -> dict[str, Any]:
    """Reconstruct the trial choose ask and cite where every piece came from."""
    catalog = mbt.live_catalog()
    unclipped = choose_prompt_unclipped(catalog)
    clipped = clip_ask(unclipped)
    policy = mb.fixed_policy_choose(catalog)
    src = choose_source_contains_literals()
    sha = sha256_text(clipped)
    return {
        "cited_from": {
            "prompt_constructor": (
                "tools/future/model_bearing.py:choose — "
                "'Pick one candidate. The scripted policy would pick ' + "
                "json.dumps(policy id) + Candidates compact JSON + "
                "'Return JSON only: ' + choice schema. Same literals, not paraphrased."
            ),
            "clip": (
                "tools/future/model_bearing.py:_ask_json — "
                f"MAX_PROMPT_CHARS={mb.MAX_PROMPT_CHARS}; "
                "prompt[:MAX_PROMPT_CHARS-1] + ellipsis"
            ),
            "catalog": "tools/future/model_bearing_torture.py:live_catalog",
            "max_ask_tokens": (
                "tools/future/model_bearing_torture.py:MAX_ASK_TOKENS="
                f"{MAX_ASK_TOKENS} and ConnectorProvider.ask payload max_tokens"
            ),
            "temperature": (
                "sealed-3.14 generation block: do_sample=false, temperature=0.0, "
                "top_k=1 (receipts/future/MODEL_BEARING_TORTURE_30M.json "
                "sealed.identity_live.generation). ConnectorProvider.ask does not "
                "override temperature; the resident uses the sealed greedy decode."
            ),
            "enable_thinking": (
                "ConnectorProvider.ask chat_template_kwargs enable_thinking=False"
            ),
            "timeline": (
                f"{TIMELINE_REL} model_calls whose prompt starts with "
                "'Pick one candidate'; prompt_sha256 "
                f"{TIMELINE_CHOOSE_PROMPT_SHA256}"
            ),
        },
        "source_literals_still_in_choose": src,
        "policy_id": policy.get("id"),
        "n_candidates": len(catalog),
        "n_compact": len(compact_candidates(catalog)),
        "unclipped_chars": len(unclipped),
        "clipped_chars": len(clipped),
        "unclipped_sha256": sha256_text(unclipped),
        "clipped_sha256": sha,
        "matches_timeline_prompt_sha256": sha == TIMELINE_CHOOSE_PROMPT_SHA256,
        "json_instruction_in_unclipped": json_instruction_reached(unclipped),
        "json_instruction_in_clipped_control": json_instruction_reached(clipped),
        "choice_id_in_clipped_control": "choice_id" in clipped,
        "clip_eats_the_schema": (
            json_instruction_reached(unclipped) and not json_instruction_reached(clipped)
        ),
        "unclipped_prompt": unclipped,
        "clipped_prompt": clipped,
        "max_ask_tokens": MAX_ASK_TOKENS,
        "max_prompt_chars": mb.MAX_PROMPT_CHARS,
        "temperature": 0.0,
        "do_sample": False,
        "enable_thinking": False,
    }


def load_timeline() -> dict[str, Any]:
    path = REPO / TIMELINE_REL
    if not path.is_file():
        # Fall back to the NAMED COMMIT, never HEAD: HEAD holds whatever the most
        # recent run produced, which is not this control.
        import subprocess

        blob = subprocess.run(
            ["git", "show", f"{TIMELINE_ARCHIVE_COMMIT}:{LIVE_TIMELINE_REL}"],
            cwd=REPO, capture_output=True, text=True,
        )
        if blob.returncode == 0 and blob.stdout.strip():
            return json.loads(blob.stdout)
    if not path.is_file():
        raise ProbeRefused(f"{TIMELINE_REL} missing", missing=[TIMELINE_REL])
    return json.loads(path.read_text(encoding="utf-8"))


def timeline_choose_calls(doc: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    blob = doc or load_timeline()
    calls = blob.get("model_calls") if isinstance(blob, Mapping) else None
    if not isinstance(calls, list):
        return []
    out: list[dict[str, Any]] = []
    for row in calls:
        if not isinstance(row, Mapping):
            continue
        prompt = str(row.get("prompt") or "")
        if prompt.startswith(CHOOSE_PREAMBLE_PREFIX):
            out.append(dict(row))
    return out


def timeline_interpret_calls(doc: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    blob = doc or load_timeline()
    calls = blob.get("model_calls") if isinstance(blob, Mapping) else None
    if not isinstance(calls, list):
        return []
    out: list[dict[str, Any]] = []
    for row in calls:
        if not isinstance(row, Mapping):
            continue
        prompt = str(row.get("prompt") or "")
        if prompt.startswith("Live frontier entries"):
            out.append(dict(row))
    return out


def summarize_timeline_calls(calls: Sequence[Mapping[str, Any]], *, as_choice: bool) -> dict[str, Any]:
    n = len(calls)
    parsed_ok = 0
    unique: dict[str, str] = {}
    for row in calls:
        text = str(row.get("reply_text") or "")
        digest = str(row.get("reply_sha256") or sha256_text(text))
        unique.setdefault(digest, text)
        if as_choice:
            if parse_choice_json(text)["parse_ok"]:
                parsed_ok += 1
        else:
            if mb._extract_json(text) is not None:
                parsed_ok += 1
    replies = [
        {"reply_sha256": digest, "reply_text": text, "n_with_this_hash": sum(
            1 for row in calls if str(row.get("reply_sha256") or sha256_text(str(row.get("reply_text") or ""))) == digest
        )}
        for digest, text in unique.items()
    ]
    return {
        "n": n,
        "n_parseable": parsed_ok,
        "parse_rate": fraction(parsed_ok, n) if n else fraction(0, 0),
        "unique_reply_hashes": len(unique),
        "replies": replies,
    }


def timeline_observation() -> dict[str, Any]:
    doc = load_timeline()
    choose = timeline_choose_calls(doc)
    interpret = timeline_interpret_calls(doc)
    choose_sum = summarize_timeline_calls(choose, as_choice=True)
    interpret_sum = summarize_timeline_calls(interpret, as_choice=False)
    prompt_shas = {str(r.get("prompt_sha256")) for r in choose}
    return {
        "source": TIMELINE_REL,
        "choose": choose_sum,
        "interpret_json_object": interpret_sum,
        "choose_prompt_sha256_unique": sorted(prompt_shas),
        "choose_prompt_matches_reconstruction": TIMELINE_CHOOSE_PROMPT_SHA256 in prompt_shas,
        "why_interpret_matters": (
            "the same sealed-3.14 body, same session, same greedy decode, "
            "emitted fenced JSON on interpret() whose prompt still contained "
            "'Return JSON only' (1536 chars, under the 1800 clip). choose() "
            "did not, and its 1800-char clip cuts the schema off."
        ),
    }


# ---------------------------------------------------------------------------
# Variations. Each is one named lever off the control.
# ---------------------------------------------------------------------------


def variation_specs(candidates: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    catalog = list(candidates or mbt.live_catalog())
    control = control_prompt(catalog)
    json_only = json_only_prompt(catalog)
    one_shot = one_shot_prompt(catalog)
    smaller = smaller_choice_set_prompt(catalog)
    return [
        {
            "variation": "control",
            "lever": "none",
            "one_change": "verbatim trial ask after the same 1800-char clip",
            "prompt": control,
            "prompt_sha256": sha256_text(control),
            "json_instruction_reached_the_body": json_instruction_reached(control),
            "grammar": False,
            "n_candidates": len(catalog),
            "candidate_ids": [mb._cid(c) for c in catalog],
        },
        {
            "variation": "json_only",
            "lever": "instruction",
            "one_change": (
                "JSON-only instruction with no prose room, placed first so the "
                "clip cannot eat it; candidates and clip budget unchanged"
            ),
            "prompt": json_only,
            "prompt_sha256": sha256_text(json_only),
            "json_instruction_reached_the_body": json_instruction_reached(json_only),
            "grammar": False,
            "n_candidates": len(catalog),
            "candidate_ids": [mb._cid(c) for c in catalog],
        },
        {
            "variation": "grammar",
            "lever": "runtime_constraint",
            "one_change": (
                "same control prompt; add a schema/grammar constraint on the "
                "runtime request if the runtime supports one"
            ),
            "prompt": control,
            "prompt_sha256": sha256_text(control),
            "json_instruction_reached_the_body": json_instruction_reached(control),
            "grammar": True,
            "n_candidates": len(catalog),
            "candidate_ids": [mb._cid(c) for c in catalog],
        },
        {
            "variation": "one_shot",
            "lever": "example",
            "one_change": "one correct-reply example prepended; same clip, same catalog",
            "prompt": one_shot,
            "prompt_sha256": sha256_text(one_shot),
            "json_instruction_reached_the_body": json_instruction_reached(one_shot),
            "grammar": False,
            "n_candidates": len(catalog),
            "candidate_ids": [mb._cid(c) for c in catalog],
            "example": ONE_SHOT_EXAMPLE,
        },
        {
            "variation": "smaller_choice_set",
            "lever": "candidate_set_size",
            "one_change": (
                "two live Hawking-self candidates so the trial schema tail "
                "survives the 1800-char clip; constructor otherwise identical"
            ),
            "prompt": smaller,
            "prompt_sha256": sha256_text(smaller),
            "json_instruction_reached_the_body": json_instruction_reached(smaller),
            "grammar": False,
            "n_candidates": len(smaller_catalog(catalog)),
            "candidate_ids": list(SMALLER_IDS),
        },
    ]


def native_runtime_grammar_support(*, source_text: str | None = None) -> dict[str, Any]:
    """Logit-mask grammar is a runtime feature. Validate-and-retry is not one."""
    text = source_text
    path_used = None
    if text is None:
        native = mbt.hcli_root() / "hcli" / "hawking_native.py"
        if native.is_file():
            text = native.read_text(encoding="utf-8")
            path_used = str(native)
        else:
            text = git("show", "HEAD:hcli/hawking_native.py")
            path_used = "HEAD:hcli/hawking_native.py"
    payload_hits = [k for k in GRAMMAR_PAYLOAD_KEYS if k in (text or "")]
    request_literal = (
        '"id": request_id' in (text or "")
        and '"prompt": prompt.text' in (text or "")
        and '"max_new_tokens": max_new_tokens' in (text or "")
        and '"max_seq_len": max_seq_len' in (text or "")
    )
    supported = False
    return {
        "supported": supported,
        "why": (
            "HawkingNativeConnector.complete_payload sends the resident "
            "{id, prompt, max_new_tokens, max_seq_len} only. No grammar, "
            "response_format, or json_schema field on the wire. "
            "crates/hawking-orch/src/grammar.rs is validate-and-retry after "
            "the reply, not a logit mask; a retry loop would be a second ask."
        ),
        "inspected": path_used,
        "native_request_keys": list(NATIVE_REQUEST_KEYS),
        "grammar_keys_mentioned_anywhere_in_source": payload_hits,
        "request_literal_is_four_keys": request_literal,
        "shell_grammar_is_validate_and_retry_not_logit_mask": True,
    }


# ---------------------------------------------------------------------------
# Bodies.
# ---------------------------------------------------------------------------


def acquire_gpu_park() -> tuple[mbt.GpuPark, dict[str, Any]]:
    lock = mbt.GPU_LOCK
    if lock.is_dir() and not any(lock.iterdir()):
        lock.rmdir()
    park = mbt.GpuPark()
    rec = park.acquire()
    rec["widens_hcli_authority"] = False
    rec["gpu_authority"] = False
    return park, rec


def start_incumbent() -> tuple[Any, mbt.ConnectorProvider, dict[str, Any]]:
    mbt._ensure_hcli_on_path()
    from hcli.hawking_native import HawkingNativeConfig, HawkingNativeConnector

    descriptor = mbt.load_sealed_descriptor()
    pin = mbt.pin_sealed_body(descriptor)
    if not pin.get("sealed"):
        raise ProbeRefused(f"sealed pin failed: {pin.get('mismatches')}", missing=["sealed-3.14"])
    cfg = HawkingNativeConfig.from_file(str(mbt.sealed_profile_path()))
    connector = HawkingNativeConnector(cfg)
    connector.start(timeout=READY_TIMEOUT_S)
    provider = mbt.ConnectorProvider(connector)
    health = provider.health()
    if not health.get("ok"):
        raise ProbeRefused(f"incumbent health not ok: {health}")
    identity = mbt.strip_hardware(connector.identity())
    return connector, provider, {
        "body": INCUMBENT_BODY,
        "pin_sealed": True,
        "pid": connector.pid,
        "health": mbt.strip_hardware(health),
        "resident_identity": identity.get("resident_identity") or pin.get("resident_identity"),
        "generation": mbt.strip_hardware(identity.get("generation") or {}),
        "gpu_authority": False,
    }


def ask_incumbent(provider: mbt.ConnectorProvider, prompt: str) -> dict[str, Any]:
    rec = provider.ask(prompt, session="main")
    text = str(rec.get("text") or "")
    parsed = parse_choice_json(text)
    return {
        "ok": bool(rec.get("ok")),
        "reply_text": text,
        "reply_sha256": sha256_text(text),
        "prompt_tokens": rec.get("prompt_tokens"),
        "generated_tokens": rec.get("generated_tokens"),
        "resident_identity": rec.get("resident_identity"),
        "elapsed_s": rec.get("elapsed_s"),
        "elapsed_evidence_class": "SELF_MEASURED_DIRTY",
        **parsed,
    }


def qwen06_on_disk() -> dict[str, Any]:
    path = QWEN06_SPECIMEN
    weights = path / "model.safetensors"
    tok = path / "tokenizer.json"
    return {
        "body": QWEN06_BODY,
        "repo": QWEN06_REPO,
        "revision": QWEN06_REVISION,
        "specimen_path": str(path),
        "present": path.is_dir() and weights.is_file() and tok.is_file(),
        "weights_present": weights.is_file(),
        "tokenizer_present": tok.is_file(),
        "mlx_python_present": MLX_PYTHON.is_file(),
    }


_MLX_WORKER = r'''
import json, sys
from mlx_lm import load, generate

spec = json.load(sys.stdin)
path = spec["path"]
max_tokens = int(spec["max_tokens"])
prompts = spec["prompts"]
model, tok = load(path)
out = []
for prompt in prompts:
    try:
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        render_how = "apply_chat_template enable_thinking=False"
    except TypeError:
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        render_how = "apply_chat_template (no enable_thinking kwarg)"
    text = generate(model, tok, rendered, max_tokens=max_tokens, verbose=False)
    out.append({"text": text if isinstance(text, str) else str(text), "render_how": render_how})
json.dump({"ok": True, "replies": out}, sys.stdout)
'''


def ask_qwen06_batch(prompts: Sequence[str], *, max_tokens: int = MAX_ASK_TOKENS) -> dict[str, Any]:
    disk = qwen06_on_disk()
    if not disk["present"]:
        raise ProbeRefused("Qwen3-0.6B specimen not on disk", missing=[str(QWEN06_SPECIMEN)])
    if not disk["mlx_python_present"]:
        raise ProbeRefused("mlx-lm python missing", missing=[str(MLX_PYTHON)])
    payload = json.dumps(
        {"path": str(QWEN06_SPECIMEN), "max_tokens": int(max_tokens), "prompts": list(prompts)}
    )
    proc = subprocess.run(
        [str(MLX_PYTHON), "-c", _MLX_WORKER],
        input=payload,
        capture_output=True,
        text=True,
        timeout=max(600, 90 + 45 * max(1, len(prompts))),
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[-2000:]
        raise ProbeRefused(f"mlx Qwen3-0.6B failed: {err}")
    try:
        body = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeRefused(f"mlx worker returned non-JSON: {(proc.stdout or '')[:500]}") from exc
    replies = body.get("replies") if isinstance(body, dict) else None
    if not isinstance(replies, list) or len(replies) != len(prompts):
        raise ProbeRefused("mlx worker reply count mismatch")
    rows: list[dict[str, Any]] = []
    for raw in replies:
        text = str((raw or {}).get("text") or "")
        parsed = parse_choice_json(text)
        rows.append(
            {
                "ok": True,
                "reply_text": text,
                "reply_sha256": sha256_text(text),
                "prompt_tokens": None,
                "generated_tokens": None,
                "resident_identity": QWEN06_BODY,
                "render_how": (raw or {}).get("render_how"),
                "elapsed_evidence_class": "SELF_MEASURED_DIRTY",
                **parsed,
            }
        )
    return {"ok": True, "body": QWEN06_BODY, "runtime": "mlx_lm greedy default sampler (argmax)", "asks": rows}


def cell_from_asks(
    *,
    body: str,
    spec: Mapping[str, Any],
    asks: Sequence[Mapping[str, Any]],
    skipped: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    n = len(asks)
    k = sum(1 for a in asks if a.get("parse_ok"))
    unique = sorted({str(a.get("reply_sha256")) for a in asks})
    row: dict[str, Any] = {
        "body": body,
        "variation": spec["variation"],
        "lever": spec["lever"],
        "one_change": spec["one_change"],
        "prompt": spec["prompt"],
        "prompt_sha256": spec["prompt_sha256"],
        "json_instruction_reached_the_body": spec["json_instruction_reached_the_body"],
        "n_candidates": spec.get("n_candidates"),
        "n": n,
        "n_parseable": k,
        "parse_rate": fraction(k, n) if n else fraction(0, 0),
        "unique_reply_hashes": unique,
        "asks": [
            {
                "reply_text": a.get("reply_text"),
                "reply_sha256": a.get("reply_sha256"),
                "parse_ok": a.get("parse_ok"),
                "choice_id": a.get("choice_id"),
                "reason": a.get("reason"),
                "why": a.get("why"),
                "generated_tokens": a.get("generated_tokens"),
                "prompt_tokens": a.get("prompt_tokens"),
                "resident_identity": a.get("resident_identity"),
            }
            for a in asks
        ],
        "skipped": dict(skipped) if skipped else None,
        "gpu_authority": False,
    }
    if spec.get("example"):
        row["example"] = spec["example"]
    if spec.get("candidate_ids"):
        row["candidate_ids"] = list(spec["candidate_ids"])
    allowed = set(spec.get("candidate_ids") or [])
    if not allowed:
        allowed = {mb._cid(c) for c in mbt.live_catalog()}
    in_set = 0
    for ask, stored in zip(asks, row["asks"]):
        cid = ask.get("choice_id")
        hit = bool(cid) and cid in allowed
        stored["choice_id_in_candidate_set"] = hit
        if hit:
            in_set += 1
    row["n_choice_id_in_candidate_set"] = in_set
    return row


def judge(cells: Sequence[Mapping[str, Any]], *, control_reproduced: bool | None) -> dict[str, Any]:
    """Name exactly one of the three campaign outcomes, or a discrepancy."""
    def ok(cell: Mapping[str, Any]) -> bool:
        return int(cell.get("n_parseable") or 0) > 0 and not cell.get("skipped")

    incumbent = [c for c in cells if c.get("body") == INCUMBENT_BODY]
    alt = [c for c in cells if c.get("body") != INCUMBENT_BODY]
    inc_control = next((c for c in incumbent if c.get("variation") == "control"), None)
    inc_working = [
        c for c in incumbent
        if c.get("variation") != "control" and ok(c)
    ]
    alt_control_ok = [c for c in alt if c.get("variation") == "control" and ok(c)]
    alt_any_ok = [c for c in alt if ok(c)]

    if inc_control is not None and ok(inc_control):
        return {
            "outcome": OUTCOME_DISCREPANCY,
            "why": (
                "the incumbent emitted parseable choice JSON on the verbatim "
                "trial ask; the 30-minute trial's 0 of 43 did not reproduce. "
                "The trial's conditions differ from this probe, or the body "
                "is no longer the one that wrote the timeline."
            ),
            "fixed_by": None,
            "recommendation": (
                "stop treating the torture receipt as a format finding until "
                "the discrepancy is named: compare this probe's control reply "
                "hash to the timeline hash "
                f"{TIMELINE_CHOOSE_REPLY_SHA256}"
            ),
        }

    if inc_working:
        first = inc_working[0]
        names = [str(c.get("variation")) for c in inc_working]
        return {
            "outcome": OUTCOME_ASK,
            "why": (
                f"incumbent control was 0 parseable; variation(s) {names} "
                "produced at least one parseable {\"choice_id\",\"reason\"}. "
                "The body can emit the object. The trial ask cannot."
            ),
            "fixed_by": first.get("variation"),
            "fixed_by_lever": first.get("lever"),
            "one_change_that_worked": first.get("one_change"),
            "working_variations": names,
            "recommendation": (
                "this is a prompt/schema fix, not succession. "
                "The named defect is MAX_PROMPT_CHARS=1800 clipping "
                "'Return JSON only: {choice_id,...}' off the choose() ask; "
                "the body never saw the contract. "
                f"The cheapest working change was {first.get('variation')!r} "
                f"({first.get('lever')}: {first.get('one_change')}). "
                "Campaign action: stop eating the schema — raise the clip, "
                "reserve the tail, or put the JSON instruction first. "
                "Do not start G011 succession on this evidence."
            ),
        }

    if alt_control_ok:
        who = [str(c.get("body")) for c in alt_control_ok]
        return {
            "outcome": OUTCOME_BODY,
            "why": (
                "the incumbent produced 0 parseable choice JSON on the trial "
                f"ask and on every variation tried; {who} produced at least "
                "one parseable object on the same ask. The incumbent is the "
                "problem. G011 succession is about cognition, not "
                "capability-per-byte."
            ),
            "fixed_by": None,
            "working_bodies": who,
            "recommendation": (
                "stop spending campaign time making sealed-3.14 follow a JSON "
                "contract it does not emit. G011 succession is about cognition, "
                "not capability-per-byte. Qualify a different sealed specimen "
                "on the choose() object, then succession."
            ),
        }

    if alt_any_ok and not inc_working:
        who = [f"{c.get('body')}:{c.get('variation')}" for c in alt_any_ok]
        return {
            "outcome": OUTCOME_BODY,
            "why": (
                "the incumbent never emitted the object; a different sealed "
                f"body did under {who}. That is still the incumbent, not scale."
            ),
            "fixed_by": None,
            "working_bodies": who,
            "recommendation": (
                "G011 succession is about this body's cognition/format "
                "obedience. A 0.6B specimen emitting the object the 27B "
                "incumbent cannot is decisive about the incumbent."
            ),
        }

    return {
        "outcome": OUTCOME_NOTHING,
        "why": (
            "no available body, on the trial ask or on any one-lever "
            "variation, emitted a parseable {\"choice_id\",\"reason\"}. "
            "The scheduler's contract with the model is unusable as written."
        ),
        "fixed_by": None,
        "recommendation": (
            "replace structured-choice, do not prompt-tune it further. "
            "A classifier over the candidate ids, a constrained decoder the "
            "runtime does not have, or not asking the model to choose in JSON "
            "are the remaining designs. This probe was the cheap discriminator; "
            "another ask variation is not."
        ),
        "control_reproduced": control_reproduced,
    }


# ---------------------------------------------------------------------------
# Live pass.
# ---------------------------------------------------------------------------


def run_cell_incumbent(
    provider: mbt.ConnectorProvider,
    spec: Mapping[str, Any],
    *,
    n: int,
    grammar_info: Mapping[str, Any],
) -> dict[str, Any]:
    if spec["variation"] == "grammar" and not grammar_info.get("supported"):
        return cell_from_asks(
            body=INCUMBENT_BODY,
            spec=spec,
            asks=[],
            skipped={
                "reason": "runtime_does_not_support_logit_mask_grammar",
                "detail": grammar_info.get("why"),
            },
        )
    asks = [ask_incumbent(provider, spec["prompt"]) for _ in range(int(n))]
    return cell_from_asks(body=INCUMBENT_BODY, spec=spec, asks=asks)


def run_probe(*, n_repeats: int = N_REPEATS, skip_alt: bool = False) -> dict[str, Any]:
    t0 = time.time()
    errors: list[str] = []
    exact = extract_exact_ask()
    if not exact["matches_timeline_prompt_sha256"]:
        errors.append(
            "reconstructed control sha256 does not match the timeline; "
            "a probe against a different ask answers a different question"
        )
    timeline = timeline_observation()
    grammar_info = native_runtime_grammar_support()
    specs = variation_specs()
    cells: list[dict[str, Any]] = []
    incumbent_meta: dict[str, Any] = {}
    alt_meta: dict[str, Any] = qwen06_on_disk()
    park = None
    park_rec: dict[str, Any] = {"held": False}
    connector = None

    try:
        park, park_rec = acquire_gpu_park()
    except Exception as exc:
        errors.append(f"gpu park failed: {type(exc).__name__}: {exc}")
        park_rec = {"held": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        connector, provider, incumbent_meta = start_incumbent()
        for spec in specs:
            try:
                cells.append(
                    run_cell_incumbent(
                        provider, spec, n=n_repeats, grammar_info=grammar_info
                    )
                )
            except Exception as exc:
                errors.append(f"incumbent {spec['variation']}: {type(exc).__name__}: {exc}")
                cells.append(
                    cell_from_asks(
                        body=INCUMBENT_BODY,
                        spec=spec,
                        asks=[],
                        skipped={"reason": f"{type(exc).__name__}: {exc}"},
                    )
                )
    except Exception as exc:
        errors.append(f"incumbent start failed: {type(exc).__name__}: {exc}")
        incumbent_meta = {"body": INCUMBENT_BODY, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if connector is not None:
            try:
                connector.stop()
            except Exception as exc:
                errors.append(f"incumbent stop: {type(exc).__name__}: {exc}")

    if not skip_alt:
        alt_specs = [s for s in specs if s["variation"] in {"control", "json_only", "smaller_choice_set"}]
        try:
            batch_prompts: list[str] = []
            batch_index: list[int] = []
            for i, spec in enumerate(alt_specs):
                for _ in range(int(n_repeats)):
                    batch_prompts.append(spec["prompt"])
                    batch_index.append(i)
            bundled = ask_qwen06_batch(batch_prompts, max_tokens=MAX_ASK_TOKENS)
            alt_meta = {**alt_meta, "runtime": bundled.get("runtime"), "ok": True}
            grouped: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(alt_specs))}
            for spec_i, ask in zip(batch_index, bundled["asks"]):
                grouped[spec_i].append(ask)
            for i, spec in enumerate(alt_specs):
                cells.append(cell_from_asks(body=QWEN06_BODY, spec=spec, asks=grouped[i]))
        except Exception as exc:
            errors.append(f"Qwen3-0.6B: {type(exc).__name__}: {exc}")
            alt_meta = {**alt_meta, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            for spec in alt_specs:
                cells.append(
                    cell_from_asks(
                        body=QWEN06_BODY,
                        spec=spec,
                        asks=[],
                        skipped={"reason": f"{type(exc).__name__}: {exc}"},
                    )
                )

    if park is not None:
        try:
            park.release()
            park_rec = {
                **park_rec,
                "released": True,
                "held_during_run": bool(park_rec.get("held")),
            }
        except Exception as exc:
            errors.append(f"gpu park release: {type(exc).__name__}: {exc}")

    inc_control = next(
        (c for c in cells if c.get("body") == INCUMBENT_BODY and c.get("variation") == "control"),
        None,
    )
    reproduced_fail = False
    matched_trial_reply = False
    if inc_control and inc_control.get("n"):
        matched_trial_reply = TIMELINE_CHOOSE_REPLY_SHA256 in (inc_control.get("unique_reply_hashes") or [])
        reproduced_fail = int(inc_control.get("n_parseable") or 0) == 0
    verdict = judge(cells, control_reproduced=reproduced_fail)
    live_ran = any(int(c.get("n") or 0) > 0 for c in cells)
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "question": "CAN ANY AVAILABLE BODY EMIT THE CHOICE JSON THIS SCHEDULER REQUIRES?",
        "outcomes_named": [OUTCOME_ASK, OUTCOME_BODY, OUTCOME_NOTHING],
        "verdict": verdict.get("outcome"),
        "reason": verdict.get("why"),
        "recommendation": verdict.get("recommendation"),
        "judgment": verdict,
        "exact_ask": exact,
        "timeline_observation": timeline,
        "live_reproduction": {
            "ran": live_ran,
            "incumbent_control_parse_rate": (inc_control or {}).get("parse_rate"),
            "incumbent_control_n_parseable": (inc_control or {}).get("n_parseable"),
            "failure_reproduced": reproduced_fail,
            "matched_trial_reply_sha256": matched_trial_reply,
            "trial_choose_reply_sha256": TIMELINE_CHOOSE_REPLY_SHA256,
            "trial_choose_prompt_sha256": TIMELINE_CHOOSE_PROMPT_SHA256,
        },
        "grammar": grammar_info,
        "cells": cells,
        "bodies": {
            "incumbent": incumbent_meta,
            "alternative": alt_meta,
        },
        "n_repeats": int(n_repeats),
        "gpu_park": park_rec,
        "errors": errors,
        "elapsed_s": round(time.time() - t0, 3),
        "elapsed_evidence_class": "SELF_MEASURED_DIRTY",
        "gpu_authority": False,
        "evidence_class": "SELF_MEASURED_DIRTY" if live_ran else "STATIC_ONLY",
        "claim_boundary": CLAIM_BOUNDARY,
        "hcli_invoked_not_edited": True,
        "autonomy_trial_not_touched": True,
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
    }
    return doc


def recovered_implementation() -> list[str]:
    return [
        "tools/future/model_bearing.py:choose / _ask_json / _extract_json / MAX_PROMPT_CHARS=1800",
        "tools/future/model_bearing_torture.py:live_catalog / ConnectorProvider.ask / MAX_ASK_TOKENS=384 / GpuPark",
        "receipts/future/MODEL_BEARING_TIMELINE.json choose prompt_sha256 31789943…",
        "hcli/hawking_native.py complete_payload request {id,prompt,max_new_tokens,max_seq_len}",
        "ModelLake Qwen--Qwen3-0.6B@c1899de289a0 (curriculum role very_small_dense_procedural_speed)",
    ]


def gaps_closed() -> list[str]:
    return [
        "the exact trial choose ask is reconstructed and hashed against the timeline, not paraphrased",
        "the 1800-char clip is named as a property of the control (schema never reached the body)",
        "each live variation is one lever off that control",
        "parse-rate is a fraction with raw replies, not a score",
        "grammar is reported unsupported rather than faked as a retry loop",
        "Qwen3-0.6B is the cheapest second sealed body",
    ]


def negative_findings() -> list[str]:
    return [
        "a JSON-only instruction that the clip eats is not an instruction the body saw",
        "interpret() JSON on the same timeline is not choose() JSON",
        "hawking-orch grammar.rs validate-and-retry is not a runtime constraint",
        "this sidecar has no GPU authority; token counts are protocol fields",
    ]


def static_document() -> dict[str, Any]:
    """Reconstruction only. --build does not mint a live outcome."""
    exact = extract_exact_ask()
    timeline = timeline_observation()
    specs = variation_specs()
    grammar_info = native_runtime_grammar_support()
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "question": "CAN ANY AVAILABLE BODY EMIT THE CHOICE JSON THIS SCHEDULER REQUIRES?",
        "outcomes_named": [OUTCOME_ASK, OUTCOME_BODY, OUTCOME_NOTHING],
        "verdict": "NOT_RUN",
        "reason": (
            "--build reconstructs the exact trial ask and does not start a body. "
            "Invoke --run. A static receipt is not a discriminator."
        ),
        "recommendation": "run tools/future/choice_json_probe.py --run on the GPU lane lock",
        "exact_ask": exact,
        "timeline_observation": timeline,
        "grammar": grammar_info,
        "variation_specs": [
            {k: v for k, v in s.items() if k != "prompt"} | {"prompt_chars": len(s["prompt"])}
            for s in specs
        ],
        "cells": [],
        "live_ran": False,
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
        "claim_boundary": CLAIM_BOUNDARY,
        "hcli_invoked_not_edited": True,
        "autonomy_trial_not_touched": True,
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "qwen06": qwen06_on_disk(),
    }


def selftest() -> None:
    exact = extract_exact_ask()
    assert exact["matches_timeline_prompt_sha256"], exact["clipped_sha256"]
    assert exact["clip_eats_the_schema"] is True
    assert exact["json_instruction_in_clipped_control"] is False
    src = choose_source_contains_literals()
    assert src["preamble_present"] and src["schema_tail_present"]
    md = (
        "The candidate is:\n\n**WU.HAWKING.resident_identity_pin**\n\n"
        "**Reasoning:**\nThe prompt explicitly states that the scripted policy would pick this."
    )
    assert parse_choice_json(md)["parse_ok"] is False
    good = '{"choice_id":"WU.HAWKING.resident_identity_pin","reason":"live work, not a closed scar"}'
    got = parse_choice_json(good)
    assert got["parse_ok"] is True
    assert got["choice_id"] == "WU.HAWKING.resident_identity_pin"
    fenced = "```json\n" + good + "\n```"
    assert parse_choice_json(fenced)["parse_ok"] is True
    interpret = (
        '```json\n{"reading":"x","worth_doing_next":["WU.DEAD.mlp_function_replacement"],'
        '"why":"highest gain"}\n```'
    )
    assert mb._extract_json(interpret) is not None
    assert parse_choice_json(interpret)["parse_ok"] is False
    specs = {s["variation"]: s for s in variation_specs()}
    assert specs["control"]["prompt_sha256"] == TIMELINE_CHOOSE_PROMPT_SHA256
    assert specs["json_only"]["json_instruction_reached_the_body"] is True
    assert specs["smaller_choice_set"]["json_instruction_reached_the_body"] is True
    assert specs["one_shot"]["prompt"].startswith("Example of a correct reply")
    assert specs["grammar"]["prompt_sha256"] == specs["control"]["prompt_sha256"]
    grammar = native_runtime_grammar_support()
    assert grammar["supported"] is False
    cells = [
        {"body": INCUMBENT_BODY, "variation": "control", "n_parseable": 0, "n": 2, "skipped": None},
        {
            "body": INCUMBENT_BODY,
            "variation": "json_only",
            "lever": "instruction",
            "one_change": "JSON-only first",
            "n_parseable": 2,
            "n": 2,
            "skipped": None,
        },
    ]
    v = judge(cells, control_reproduced=True)
    assert v["outcome"] == OUTCOME_ASK, v
    empty = [
        {"body": INCUMBENT_BODY, "variation": "control", "n_parseable": 0, "n": 2, "skipped": None},
        {"body": QWEN06_BODY, "variation": "control", "n_parseable": 0, "n": 2, "skipped": None},
    ]
    nothing = judge(empty, control_reproduced=True)
    assert nothing["outcome"] == OUTCOME_NOTHING, nothing


def build() -> Path:
    return write_receipt(RECEIPT, static_document(), RECORDED_BY)


def main() -> int:
    parser = argparse.ArgumentParser(description="Choice-JSON body-vs-ask probe")
    parser.add_argument("--run", action="store_true", help="park on the GPU lock and run the bodies")
    parser.add_argument("--build", action="store_true", help="static reconstruction receipt; not a discriminator")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--skip-alt", action="store_true", help="incumbent only")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("selftest ok")
        return 0
    if args.run:
        doc = run_probe(n_repeats=int(args.n_repeats), skip_alt=bool(args.skip_alt))
        path = write_receipt(RECEIPT, doc, RECORDED_BY)
        print(
            json.dumps(
                {
                    "verdict": doc.get("verdict"),
                    "reason": doc.get("reason"),
                    "recommendation": doc.get("recommendation"),
                    "receipt": str(path),
                    "cells": [
                        {
                            "body": c.get("body"),
                            "variation": c.get("variation"),
                            "parse_rate": c.get("parse_rate"),
                            "skipped": c.get("skipped"),
                        }
                        for c in doc.get("cells") or []
                    ],
                    "errors": doc.get("errors"),
                },
                indent=2,
            )
        )
        return 0
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
