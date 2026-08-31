"""DRIVE THE REAL RESIDENT — own one body; return the model's ANSWER, not a transcript.

The sealed-3.14 resident already starts and generates. This module owns the
process AND the prompt/stop contract the live probe never had: apply the
artifact's own chat_template.jinja, honour EOS and turn boundaries, detect
degenerate repetition, and label CLEAN | TRUNCATED | DEGENERATE so a decision
is extractable without a human fishing it out of loop garbage.

The cognition is there (the model chose hbm_doctor.py). The harness was wrong:
raw concatenated text made the model continue a fabricated transcript, and
generation ran until max_new_tokens instead of stopping at the turn.

Refuses: a second 9.9GB body; ask before ready; dead pid as healthy-with-zero;
malformed reply as an answer; hardware-named receipt fields; stdin=DEVNULL;
inventing a chat format when the jinja is missing or fails its anchors; a
User:/Assistant: concatenation; calling a repeating fragment CLEAN; calling a
budget-hit mid-sentence CLEAN; regex-fishing an answer out of loop garbage.

Cannot establish: a qualified throughput number (no GPU lease; timings are
SELF_MEASURED_DIRTY); that the 9.9GB sealed body, once templated, answers the
three quality probes CLEAN — build() does not spawn it; a live campaign holds
the machine. Observed thinking-arm obedience is UNPROBED on the sealed body.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from hcli.resources import pid_is_alive, process_start_token
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    REPO,
    git,
    sha256_file,
    write_receipt,
)
from tools.future.resident_health import rss_bytes_of
from tools.future.resident_identity import SEALED_REL, load_authority
from tools.future.workunit_species import emit_hcli_workunit, validate_emitted_unit

RECEIPT = "RESIDENT_PROVIDER.json"
SCHEMA = "hawking.future.resident_provider.v1"
RECORDED_BY = "tools/future/resident_provider.py"
VERSION = 1
PROTOCOL = "hawking.qwen38.resident.v1"
LIVE_PROBE_REL = "receipts/future/evidence/RESIDENT_LIVE_PROBE.json"
PROBE_BINARY_REL = (
    "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_resident"
)
MAX_HASH_BYTES = 64 << 20
DEFAULT_READY_TIMEOUT_S = 180.0
DEFAULT_ASK_TIMEOUT_S = 30.0
DOUBLE_READY_TIMEOUT_S = 4.0
STOP_GRACE_S = 0.4
SPAWN_TOKEN_S = 1.0

# Keys a caller or a receipt is allowed to keep from a protocol record.
# Everything else is either hardware, a nested metric blob, or unused.
REPLY_KEEP: tuple[str, ...] = (
    "id",
    "status",
    "protocol",
    "text",
    "generated_text",
    "generated_tokens",
    "prompt_tokens",
    "prompt_len",
    "fallbacks",
    "dense_w_materialized",
    "model_open_count",
    "weight_upload_count",
    "resident_identity",
    "resident_pid",
    "max_seq_len",
    "error",
    "new_token_ids",
)
READY_KEEP: tuple[str, ...] = (
    "status",
    "protocol",
    "resident_identity",
    "resident_pid",
    "max_seq_len",
    "model_open_count",
    "weight_upload_count",
    "dense_w_materialized",
    "fallbacks",
)
# write_receipt already bans HARDWARE_FIELDS; these extra names are rates the
# resident puts on the wire and must never be copied into a sidecar document.
RATE_FIELDS = frozenset({"decode_tps", "complete_tps"})
FORBIDDEN_WIRE = HARDWARE_FIELDS | RATE_FIELDS

# Quality is computed, never assumed. A validator nobody has watched reject
# is a validator that will silently drift into fiction.
QUALITY_CLEAN = "CLEAN"
QUALITY_TRUNCATED = "TRUNCATED"
QUALITY_DEGENERATE = "DEGENERATE"
QUALITY_LABELS = (QUALITY_CLEAN, QUALITY_TRUNCATED, QUALITY_DEGENERATE)

# tokenizer_config.json / generation_config.json / qwen38_geometry.rs agree.
# Used only to recognise a stop, never as a hardware claim.
EOS_IM_END_ID = 248046
EOS_EOT_ID = 248044
EOS_TOKEN_IDS = (EOS_IM_END_ID, EOS_EOT_ID)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EOT = "<|endoftext|>"
CLOSED_THINK = "<think>\n\n</think>\n\n"
OPEN_THINK = "<think>\n"
CHAT_TEMPLATE_NAME = "chat_template.jinja"

# Jinja arm when enable_thinking is true / undefined. Copied from the artifact
# template; not an invented system prompt.
REASONING_XHIGH = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer."
)

# Anchors the renderer requires of the artifact jinja. A foreign/rewritten
# template fails closed rather than silently changing the token stream.
TEMPLATE_ANCHORS: tuple[str, ...] = (
    "{%- if add_generation_prompt %}",
    "{{- '<|im_start|>assistant\\n' }}",
    "enable_thinking is defined and enable_thinking is false",
    "{{- '<think>\\n\\n</think>\\n\\n' }}",
    "{{- '<think>\\n' }}",
    "{%- if enable_thinking is undefined or enable_thinking is true %}",
    "{{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}",
)

# Stub used only when the sealed artifact is not on disk. Must carry every
# anchor so the renderer still refuses a foreign dialect.
QWEN_TEMPLATE_STUB = (
    "{%- if enable_thinking is undefined or enable_thinking is true %}\n"
    "{%- endif %}\n"
    "{%- for message in messages %}\n"
    "    {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|im_start|>assistant\\n' }}\n"
    "    {%- if enable_thinking is defined and enable_thinking is false %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- else %}\n"
    "        {{- '<think>\\n' }}\n"
    "    {%- endif %}\n"
    "{%- endif %}\n"
)

PROMPT_FRANCE = "What is the capital of France? Answer in one word."
PROMPT_CHOICE = (
    "Choose ONE and reply with only its name: freshness.py, "
    "negative_index.py, hbm_doctor.py. Which most likely reveals a stale "
    "artifact?"
)
PROMPT_STATUS = "In one sentence: why is a status label not the same as the world state?"

# Measured on the real resident with the old harness (raw prompt, no stop).
# Negative control: this must never be classified CLEAN, and hbm_doctor.py
# must not be regex-fished out of it.
MEASURED_DEGENERATE_CHOICE = (
    "\nAssistant:\n\nAssistant:\n\nhbm_doctor.py\n\nAssistant:\n\nhbm_do\n\nAssistant:\n\nh"
)
MEASURED_DEGENERATE_STATUS = (
    "\nAssistant:\n\nAssistant:\n\nh\n\nAssistant:\n\nh\n\nAssistant:\n\nh"
)
MEASURED_CLEAN_FRANCE = "\n\nParis"

_FABRICATED_TURN = re.compile(r"(?:^|\n)(?:Assistant|User|System)\s*:")
_ASSISTANT_TURN = re.compile(r"(?:^|\n)\s*Assistant\s*:")
_USER_TURN = re.compile(r"(?:^|\n)\s*User\s*:")

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. "
    "A provider-side elapsed_s is SELF_MEASURED_DIRTY process telemetry on a "
    "contaminated machine with no lease; it ranks nothing."
)
ERAS = (
    "I Genesis of the Laboratory",
    "II Compounding Civilization",
    "III Autonomous Science Civilization",
    "IV Synthetic Machine Civilization",
    "V Released Hawking Civilization",
)
ODYSSEYS = (
    "I WHAT IS TRUE?",
    "II WHAT DID HAWKING ALREADY LEARN?",
    "III WHERE IS HAWKING WRONG?",
)

# Protocol double. Speaks hawking.qwen38.resident.v1 so the driver is tested
# without loading 9.9GB or taking a GPU. Modes via HAWKING_PROVIDER_DOUBLE.
PROTOCOL_DOUBLE_SOURCE = """\
import json, os, sys, time
PROTOCOL = "hawking.qwen38.resident.v1"
mode = os.environ.get("HAWKING_PROVIDER_DOUBLE", "ok")
sleep_s = float(os.environ.get("HAWKING_PROVIDER_DOUBLE_SLEEP_S", "0") or 0)
identity = "sealed-3.14"
args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "--resident-identity" and i + 1 < len(args):
        identity = args[i + 1]
        i += 2
        continue
    i += 1
pid = os.getpid()

def emit(obj):
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\\n")
    sys.stdout.flush()

if mode == "die":
    sys.exit(0)
if mode == "never_ready":
    time.sleep(60)
    sys.exit(0)

emit({
    "status": "ready",
    "protocol": PROTOCOL,
    "resident_identity": identity,
    "resident_pid": pid,
    "max_seq_len": 8192,
    "model_open_count": 1,
    "weight_upload_count": 1,
    "dense_w_materialized": 0,
    "fallbacks": 0,
})
opens = 1
for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    if sleep_s:
        time.sleep(sleep_s)
    if mode == "malformed":
        sys.stdout.write("this is not json\\n")
        sys.stdout.flush()
        continue
    try:
        body = json.loads(line)
    except Exception as exc:
        emit({"id": "", "status": "error", "protocol": PROTOCOL, "error": str(exc)})
        continue
    if not isinstance(body, dict):
        emit({"id": "", "status": "error", "protocol": PROTOCOL, "error": "not an object"})
        continue
    rid = body.get("id") or ""
    if mode == "reload":
        opens += 1
        emit({
            "id": rid,
            "status": "ok",
            "protocol": PROTOCOL,
            "text": "reload",
            "generated_tokens": 1,
            "model_open_count": opens,
            "weight_upload_count": opens,
            "fallbacks": 0,
            "resident_identity": identity,
            "resident_pid": pid,
        })
        continue
    if mode == "error_status":
        emit({"id": rid, "status": "error", "protocol": PROTOCOL, "error": "forced"})
        continue
    if mode == "dirty_metrics":
        emit({
            "id": rid,
            "status": "ok",
            "protocol": PROTOCOL,
            "text": "dirty",
            "generated_tokens": 1,
            "new_token_ids": [248046],
            "model_open_count": 1,
            "weight_upload_count": 1,
            "fallbacks": 0,
            "resident_identity": identity,
            "resident_pid": pid,
            "decode_tps": 99.0,
            "complete_tps": 12.0,
            "wall_ns": 123,
            "gpu_ns": 45,
            "tps": 12.0,
        })
        continue
    prompt = body.get("prompt") or ""
    try:
        n = int(body.get("max_new_tokens") or 1)
    except (TypeError, ValueError):
        n = 1
    if mode == "degenerate":
        emit({
            "id": rid,
            "status": "ok",
            "protocol": PROTOCOL,
            "text": (
                "\\nAssistant:\\n\\nAssistant:\\n\\nhbm_doctor.py\\n\\n"
                "Assistant:\\n\\nhbm_do\\n\\nAssistant:\\n\\nh"
            ),
            "generated_tokens": n,
            "model_open_count": 1,
            "weight_upload_count": 1,
            "fallbacks": 0,
            "resident_identity": identity,
            "resident_pid": pid,
        })
        continue
    if mode == "truncated":
        emit({
            "id": rid,
            "status": "ok",
            "protocol": PROTOCOL,
            "text": "A status label is a claim about",
            "generated_tokens": n,
            "model_open_count": 1,
            "weight_upload_count": 1,
            "fallbacks": 0,
            "resident_identity": identity,
            "resident_pid": pid,
        })
        continue
    if mode == "thinking_echo":
        emit({
            "id": rid,
            "status": "ok",
            "protocol": PROTOCOL,
            "text": "<think>\\nsecret\\n</think>\\n\\nParis",
            "generated_tokens": 8,
            "new_token_ids": [248046],
            "model_open_count": 1,
            "weight_upload_count": 1,
            "fallbacks": 0,
            "resident_identity": identity,
            "resident_pid": pid,
        })
        continue
    if mode == "quality_scripted":
        low = str(prompt)
        if "capital of France" in low:
            out = "Paris"
        elif "hbm_doctor.py" in low and "freshness.py" in low:
            out = "hbm_doctor.py"
        elif "status label" in low.lower() and "world state" in low.lower():
            out = "A status label is a claim; the world state is independently true."
        else:
            out = "ok"
        emit({
            "id": rid,
            "status": "ok",
            "protocol": PROTOCOL,
            "text": out,
            "generated_text": out,
            "generated_tokens": 1,
            "new_token_ids": [248046],
            "model_open_count": 1,
            "weight_upload_count": 1,
            "fallbacks": 0,
            "resident_identity": identity,
            "resident_pid": pid,
            "dense_w_materialized": 0,
        })
        continue
    emit({
        "id": rid,
        "status": "ok",
        "protocol": PROTOCOL,
        "text": "ok",
        "generated_text": "ok",
        "generated_tokens": 1,
        "new_token_ids": [248046],
        "model_open_count": 1,
        "weight_upload_count": 1,
        "fallbacks": 0,
        "resident_identity": identity,
        "resident_pid": pid,
        "dense_w_materialized": 0,
    })
"""


class ProviderRefuse(RuntimeError):
    """Operational refusal with a reason. Never a success-shaped default."""

    def __init__(self, reason: str, *, fault: str = "refused") -> None:
        self.reason = reason
        self.fault = fault
        super().__init__(f"REFUSED [{fault}]: {reason}")


class NotReady(ProviderRefuse):
    def __init__(self, reason: str) -> None:
        super().__init__(reason, fault="not_ready")


class DeadProcess(ProviderRefuse):
    def __init__(self, reason: str) -> None:
        super().__init__(reason, fault="dead")


class MalformedReply(ProviderRefuse):
    def __init__(self, reason: str) -> None:
        super().__init__(reason, fault="malformed_reply")


class AskFailed(ProviderRefuse):
    def __init__(self, reason: str) -> None:
        super().__init__(reason, fault="ask_failed")


class WeightReload(ProviderRefuse):
    def __init__(self, reason: str) -> None:
        super().__init__(reason, fault="weight_reload")


class TemplateRefuse(ProviderRefuse):
    def __init__(self, reason: str) -> None:
        super().__init__(reason, fault="template_absent")


class QualityUnproven(ProviderRefuse):
    def __init__(self, reason: str) -> None:
        super().__init__(reason, fault="quality_unproven")


# ---------------------------------------------------------------------------
# Artifact chat template. The model's own jinja, not an invented format.
# We do not run a general Jinja engine (jinja2 is not a sidecar dependency).
# The renderer implements the text-only, no-tools path of THIS template and
# refuses any other branch; missing anchors fail closed.
# ---------------------------------------------------------------------------


def load_declared_thinking() -> dict[str, Any]:
    """Sealed generation.enable_thinking. Absence is UNDECLARED, not a default."""
    src, ident = load_authority(SEALED_REL)
    if not isinstance(ident, dict):
        return {
            "present": False,
            "enable_thinking": None,
            "source": src,
            "reason": f"{SEALED_REL} was not locatable as a JSON object ({src})",
        }
    gen = ident.get("generation")
    if not isinstance(gen, dict) or "enable_thinking" not in gen:
        return {
            "present": False,
            "enable_thinking": None,
            "source": f"{SEALED_REL}#generation",
            "reason": "sealed generation.enable_thinking is absent; jinja default is thinking ON",
        }
    val = gen.get("enable_thinking")
    if not isinstance(val, bool):
        return {
            "present": False,
            "enable_thinking": None,
            "source": f"{SEALED_REL}#generation.enable_thinking",
            "reason": f"sealed enable_thinking is {val!r}, not a bool; refusing to coerce",
        }
    return {
        "present": True,
        "enable_thinking": val,
        "source": f"{SEALED_REL}#generation.enable_thinking",
        "reason": None,
        "jinja_default_if_undefined": "thinking_on_with_xhigh_reasoning_system",
    }


def _artifact_root_from_launch(spec: Mapping[str, Any] | None) -> Path | None:
    if not isinstance(spec, Mapping):
        return None
    raw = spec.get("artifact_root")
    if isinstance(raw, str) and raw.strip():
        return Path(raw)
    argv = spec.get("argv")
    if isinstance(argv, list):
        for i, tok in enumerate(argv):
            if str(tok) == "--artifact-root" and i + 1 < len(argv):
                return Path(str(argv[i + 1]))
    return None


def load_chat_template(artifact_root: str | Path | None = None) -> dict[str, Any]:
    """Load chat_template.jinja from the sealed artifact. Missing → present=False."""
    ident_src, ident = load_authority(SEALED_REL)
    tried: list[dict[str, str]] = []
    candidates: list[tuple[str, Path]] = []
    if artifact_root is not None:
        candidates.append(("explicit_artifact_root", Path(artifact_root) / CHAT_TEMPLATE_NAME))
    if isinstance(ident, dict) and isinstance(ident.get("artifact_root"), str):
        candidates.append(
            (f"sealed:{ident_src}", Path(ident["artifact_root"]) / CHAT_TEMPLATE_NAME)
        )
    chosen: tuple[str, Path] | None = None
    for source, path in candidates:
        tried.append({"source": source, "path": str(path)})
        if path.is_file():
            chosen = (source, path)
            break
    if chosen is None:
        return {
            "present": False,
            "filename": CHAT_TEMPLATE_NAME,
            "reason": (
                f"{CHAT_TEMPLATE_NAME} was not locatable at artifact_root; "
                "refusing to invent a chat format"
            ),
            "tried": tried,
            "text": None,
            "sha256": None,
            "path": None,
            "gpu_authority": False,
        }
    source, path = chosen
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        return {
            "present": False,
            "filename": CHAT_TEMPLATE_NAME,
            "path": str(path),
            "source": source,
            "reason": f"{path} unreadable: {type(exc).__name__}: {exc}",
            "tried": tried,
            "text": None,
            "sha256": None,
            "gpu_authority": False,
        }
    missing = [a for a in TEMPLATE_ANCHORS if a not in text]
    if missing:
        return {
            "present": False,
            "filename": CHAT_TEMPLATE_NAME,
            "path": str(path),
            "source": source,
            "reason": (
                f"{CHAT_TEMPLATE_NAME} is missing required anchors {missing}; "
                "refusing to invent a format"
            ),
            "missing_anchors": missing,
            "tried": tried,
            "text": None,
            "sha256": None,
            "gpu_authority": False,
        }
    digest = hashlib.sha256(raw).hexdigest()
    tok_path = path.parent / "tokenizer_config.json"
    tok_match: bool | None = None
    tok_reason: str | None = None
    if tok_path.is_file():
        try:
            cfg = json.loads(tok_path.read_text(encoding="utf-8"))
            embedded = cfg.get("chat_template") if isinstance(cfg, dict) else None
            if not isinstance(embedded, str):
                tok_match = False
                tok_reason = "tokenizer_config.json has no string chat_template"
            else:
                emb = embedded.encode("utf-8")
                tok_match = emb == raw or emb + b"\n" == raw or emb == raw + b"\n"
                if not tok_match:
                    tok_reason = (
                        "tokenizer_config chat_template differs from chat_template.jinja "
                        "by more than a trailing LF"
                    )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            tok_match = False
            tok_reason = f"tokenizer_config.json unreadable: {type(exc).__name__}"
    else:
        tok_reason = "tokenizer_config.json absent beside the jinja"
    return {
        "present": True,
        "filename": CHAT_TEMPLATE_NAME,
        "path": str(path),
        "source": source,
        "sha256": digest,
        "n_bytes": len(raw),
        "text": text,
        "missing_anchors": [],
        "tried": tried,
        "tokenizer_config_path": str(tok_path) if tok_path.is_file() else None,
        "tokenizer_config_chat_template_match": tok_match,
        "tokenizer_config_reason": tok_reason,
        "gpu_authority": False,
    }


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, dict) and (
                "image" in item
                or "image_url" in item
                or item.get("type") == "image"
                or "video" in item
                or item.get("type") == "video"
            ):
                raise TemplateRefuse(
                    "vision content is not rendered by this path; refuse rather than guess"
                )
            else:
                raise TemplateRefuse(
                    f"unexpected item in message content: {type(item).__name__}"
                )
        return "".join(parts)
    raise TemplateRefuse(f"unexpected content type {type(content).__name__}")


def _assert_template_anchors(template_text: str) -> None:
    missing = [a for a in TEMPLATE_ANCHORS if a not in template_text]
    if missing:
        raise TemplateRefuse(
            f"chat template is not the Qwen artifact jinja (missing {missing}); "
            "refusing to render"
        )


def render_chat(
    messages: Sequence[Mapping[str, Any]],
    *,
    enable_thinking: bool,
    template_text: str,
    template_sha256: str = "",
) -> dict[str, Any]:
    """Apply the artifact template's text-only, no-tools path.

    Matches the jinja byte-for-byte for ordinary system/user/assistant turns
    (gold-checked against jinja2 ImmutableSandboxedEnvironment with
    trim_blocks/lstrip_blocks). Tool/vision branches are refused.
    """
    _assert_template_anchors(template_text)
    if not messages:
        raise TemplateRefuse("chat template requires at least one message")
    rows: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, Mapping):
            raise TemplateRefuse("message must be an object")
        role = str(item.get("role") or "")
        if role == "tool":
            raise TemplateRefuse(
                "tool/observation turns are not rendered; refuse rather than guess a tool render"
            )
        if role not in {"system", "user", "assistant"}:
            raise TemplateRefuse(
                f"unexpected message role {role!r} (expected system/user/assistant)"
            )
        rows.append(dict(item))
        rows[-1]["role"] = role
        rows[-1]["content"] = _message_text(item.get("content"))
    if not any(r["role"] == "user" for r in rows):
        raise TemplateRefuse("No user query found in messages.")

    parts: list[str] = []
    first_role = rows[0]["role"]
    if enable_thinking:
        if first_role == "system":
            sys_content = str(rows[0]["content"]).strip()
            body = (REASONING_XHIGH + "\n\n" + sys_content) if sys_content else REASONING_XHIGH
            parts.append(f"{IM_START}system\n{body}{IM_END}\n")
        else:
            parts.append(f"{IM_START}system\n{REASONING_XHIGH}{IM_END}\n")
    elif first_role == "system":
        sys_content = str(rows[0]["content"]).strip()
        if sys_content:
            parts.append(f"{IM_START}system\n{sys_content}{IM_END}\n")

    for i, message in enumerate(rows):
        role = message["role"]
        content = str(message["content"]).strip()
        if role == "system":
            if i != 0:
                raise TemplateRefuse("System message must be at the beginning.")
            continue
        if role == "user":
            parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")
            continue
        reasoning_content = message.get("reasoning_content")
        rc = reasoning_content.strip() if isinstance(reasoning_content, str) else ""
        # preserve_thinking is undefined in this path, so history keeps the
        # think wrapper the jinja emits by default.
        parts.append(
            f"{IM_START}{role}\n<think>\n{rc}\n</think>\n\n{content}{IM_END}\n"
        )

    parts.append(f"{IM_START}assistant\n")
    if enable_thinking:
        parts.append(OPEN_THINK)
        applied_arm = "thinking_on"
        applied_prefix = OPEN_THINK
    else:
        parts.append(CLOSED_THINK)
        applied_arm = "thinking_off"
        applied_prefix = CLOSED_THINK
    text = "".join(parts)
    return {
        "text": text,
        "enable_thinking": enable_thinking,
        "applied_arm": applied_arm,
        "applied_prefix": applied_prefix,
        "n_messages": len(rows),
        "templated": True,
        "concatenated": False,
        "source_filename": CHAT_TEMPLATE_NAME,
        "source_sha256": template_sha256,
        "contains_im_start": IM_START in text,
        "contains_user_colon_blob": "User: " in text and IM_START not in text,
    }


# ---------------------------------------------------------------------------
# Stop + quality. Cut at the turn boundary; never fish an answer out of junk.
# ---------------------------------------------------------------------------


def cut_at_stop(text: str) -> tuple[str, str | None]:
    """Keep bytes before the first EOS / next-turn / fabricated role marker."""
    if not isinstance(text, str) or not text:
        return "", None
    hits: list[tuple[int, int, str]] = []
    for marker, reason in (
        (IM_END, "eos_im_end"),
        (EOT, "eos_endoftext"),
        (IM_START, "turn_boundary"),
    ):
        idx = text.find(marker)
        if idx >= 0:
            hits.append((idx, 0 if reason.startswith("eos") else 1, reason))
    match = _FABRICATED_TURN.search(text)
    if match is not None:
        hits.append((match.start(), 2, "fabricated_turn"))
    if not hits:
        return text, None
    hits.sort()
    idx, _prio, reason = hits[0]
    return text[:idx], reason


def is_degenerate(text: str) -> bool:
    """True when a short fragment repeats. Empty is a different condition."""
    if not isinstance(text, str) or not text:
        return False
    if len(_ASSISTANT_TURN.findall(text)) >= 2:
        return True
    if len(_USER_TURN.findall(text)) >= 2:
        return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    run = 1
    for prev, cur in zip(lines, lines[1:]):
        if prev == cur and 1 <= len(cur) <= 80:
            run += 1
            if run >= 3:
                return True
        else:
            run = 1
    compact = re.sub(r"\s+", "", text)
    n = len(compact)
    if n >= 8:
        for size in range(1, 17):
            need = size * 4
            if need > n:
                break
            limit = n - need + 1
            for i in range(limit):
                chunk = compact[i : i + size]
                if chunk and compact[i : i + need] == chunk * 4:
                    return True
    return False


def observe_thinking(raw: str) -> str:
    """What the generated text did, not what the config said it would do."""
    if not isinstance(raw, str) or not raw:
        return "no_text"
    stripped = raw.lstrip()
    if stripped.startswith("<think>") and "</think>" in stripped[:200]:
        head = stripped.split("</think>", 1)[0]
        inner = head[len("<think>") :].strip()
        if not inner:
            return "closed_think_echo"
        return "thinking_opened"
    if stripped.startswith("<think>"):
        return "thinking_opened"
    if "<think>" in raw:
        return "think_tag_present"
    return "no_think_tag"


def finalize_generation(
    *,
    text: Any,
    generated_tokens: Any,
    max_new_tokens: int,
    new_token_ids: Any = None,
) -> dict[str, Any]:
    """Cut, classify, and refuse to call junk CLEAN.

    Does not regex-extract a plausible token from the middle of a loop.
    Text before the first turn marker is the reply; a repeating fragment
    is DEGENERATE even when a prefix looks like an answer.
    """
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
        raise ProviderRefuse(
            f"max_new_tokens must be a positive integer, got {max_new_tokens!r}",
            fault="bad_request",
        )
    raw = text if isinstance(text, str) else ""
    cut, stop_reason = cut_at_stop(raw)
    answer = cut.strip()
    ids = new_token_ids if isinstance(new_token_ids, list) else []
    eos_hit = False
    if ids:
        last = ids[-1]
        if last in EOS_TOKEN_IDS:
            eos_hit = True
            if stop_reason is None:
                stop_reason = "eos_token_id"
    n_gen: int | None
    if isinstance(generated_tokens, int) and not isinstance(generated_tokens, bool):
        n_gen = generated_tokens
    else:
        n_gen = None
    budget_hit = n_gen is not None and n_gen >= max_new_tokens
    early_stop = n_gen is not None and n_gen < max_new_tokens
    if early_stop and stop_reason is None:
        stop_reason = "early_stop"

    degenerate = is_degenerate(raw) or is_degenerate(answer)
    if not answer and stop_reason in {"fabricated_turn", "turn_boundary"}:
        degenerate = True

    if degenerate:
        label = QUALITY_DEGENERATE
    elif eos_hit or stop_reason in {
        "eos_im_end",
        "eos_endoftext",
        "eos_token_id",
        "early_stop",
        "turn_boundary",
        "fabricated_turn",
    }:
        if answer:
            label = QUALITY_CLEAN
        elif stop_reason == "fabricated_turn":
            label = QUALITY_DEGENERATE
        elif budget_hit:
            label = QUALITY_TRUNCATED
        else:
            label = QUALITY_CLEAN
    elif budget_hit:
        label = QUALITY_TRUNCATED
    elif n_gen is None and stop_reason is None:
        raise QualityUnproven(
            "quality refused: no stop marker and generated_tokens omitted; "
            "cannot call CLEAN"
        )
    else:
        label = QUALITY_TRUNCATED

    return {
        "text": answer,
        "raw_text": raw,
        "quality": label,
        "stopped_at": stop_reason if stop_reason is not None else ("max_new_tokens" if budget_hit else None),
        "generated_tokens": n_gen,
        "max_new_tokens": max_new_tokens,
        "eos_token_seen": eos_hit,
        "degenerate": degenerate,
        "extractable": label == QUALITY_CLEAN and bool(answer),
        "observed_thinking": observe_thinking(raw),
    }


def quality(reply: Mapping[str, Any]) -> str:
    """CLEAN | TRUNCATED | DEGENERATE, recomputed from evidence. Never trusted."""
    if not isinstance(reply, Mapping):
        raise ProviderRefuse("quality() needs a mapping reply", fault="bad_request")
    raw = reply.get("raw_text")
    if not isinstance(raw, str):
        raw = reply.get("text")
    if not isinstance(raw, str):
        raise QualityUnproven(
            "quality() refused: reply has no text/raw_text; cannot call CLEAN"
        )
    max_new = reply.get("max_new_tokens")
    if not isinstance(max_new, int) or isinstance(max_new, bool) or max_new <= 0:
        raise QualityUnproven(
            "quality() refused: max_new_tokens missing; cannot distinguish "
            "CLEAN from TRUNCATED"
        )
    got = finalize_generation(
        text=raw,
        generated_tokens=reply.get("generated_tokens"),
        max_new_tokens=max_new,
        new_token_ids=reply.get("new_token_ids"),
    )
    return str(got["quality"])


def thinking_arm_record(
    *,
    declared: Mapping[str, Any],
    applied: bool,
    observed: str,
    observed_on: str,
) -> dict[str, Any]:
    """Declared vs applied vs observed. Config obedience is not assumed."""
    declared_val = declared.get("enable_thinking") if isinstance(declared, Mapping) else None
    declared_present = bool(isinstance(declared, Mapping) and declared.get("present"))
    if observed_on != "sealed_resident":
        obeyed: bool | str = "UNPROBED_ON_SEALED_BODY"
    elif observed in {"no_text"}:
        obeyed = "UNPROBED"
    elif applied is False:
        obeyed = observed in {"no_think_tag", "closed_think_echo"}
    else:
        obeyed = observed in {"thinking_opened", "think_tag_present", "closed_think_echo"}
    return {
        "declared": declared_val,
        "declared_present": declared_present,
        "declared_source": None if not isinstance(declared, Mapping) else declared.get("source"),
        "applied": applied,
        "applied_arm": "thinking_on" if applied else "thinking_off",
        "applied_prefix": OPEN_THINK if applied else CLOSED_THINK,
        "observed": observed,
        "observed_on": observed_on,
        "config_obeyed": obeyed,
        "jinja_default_if_undefined": "thinking_on_with_xhigh_reasoning_system",
        "campaign_note": (
            "same body scored 0/43 with thinking on and 14/43 with it off; "
            "the closed-think generation prefix is load-bearing"
        ),
    }


# ---------------------------------------------------------------------------
# Wire sanitizer. A resident reply carries wall_ns / gpu_ns / *tps; the
# receipt must never see a numeric hardware-named field.
# ---------------------------------------------------------------------------


def _public_record(body: Mapping[str, Any], keep: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keep:
        if key not in body:
            continue
        if key in FORBIDDEN_WIRE:
            continue
        out[key] = body[key]
    return out


def _hardware_numeric_keys(node: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key in FORBIDDEN_WIRE and isinstance(value, (int, float)) and not isinstance(value, bool):
                hits.append(here)
            hits.extend(_hardware_numeric_keys(value, here))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            hits.extend(_hardware_numeric_keys(value, f"{path}[{i}]"))
    return hits


def _assert_one_body(payload: Mapping[str, Any], *, where: str) -> tuple[int, int]:
    open_c = payload.get("model_open_count")
    up_c = payload.get("weight_upload_count")
    if not isinstance(open_c, int) or isinstance(open_c, bool):
        raise ProviderRefuse(
            f"{where} omitted a numeric model_open_count; cannot prove one body",
            fault="one_body_unproven",
        )
    if not isinstance(up_c, int) or isinstance(up_c, bool):
        raise ProviderRefuse(
            f"{where} omitted a numeric weight_upload_count; cannot prove one body",
            fault="one_body_unproven",
        )
    if open_c != 1 or up_c != 1:
        raise WeightReload(
            f"{where} model_open_count={open_c} weight_upload_count={up_c}; "
            "the provider reloaded weights and that is a defect"
        )
    return open_c, up_c


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProviderRefuse(f"{name} must be a positive integer, got {value!r}", fault="bad_request")
    return value


def _nonempty_str(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProviderRefuse(f"{name} must be a non-empty string", fault="bad_request")
    return value


def _json_digest(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _file_digest(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        if path.stat().st_size > MAX_HASH_BYTES:
            return None
        return sha256_file(path)
    except OSError:
        return None


def _load_json_rel(rel: str) -> tuple[dict[str, Any] | None, str]:
    path = REPO / rel
    if path.is_file():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None, f"{rel}:UNPARSEABLE"
        if isinstance(doc, dict):
            return doc, rel
    raw = git("show", f"HEAD:{rel}")
    if raw:
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError:
            return None, f"HEAD:{rel}:UNPARSEABLE"
        if isinstance(doc, dict):
            return doc, f"HEAD:{rel}"
    return None, "ABSENT"


def _checkout_parent() -> Path | None:
    common = git("rev-parse", "--git-common-dir")
    if not common:
        return None
    gd = Path(common)
    gd = gd.resolve() if gd.is_absolute() else (REPO / gd).resolve()
    parent = gd.parent if gd.name == ".git" else gd
    return parent


def _binary_candidates(ident: Mapping[str, Any], probe: Mapping[str, Any] | None) -> list[Path]:
    out: list[Path] = []
    rb = ident.get("resident_binary")
    if isinstance(rb, str) and rb.strip():
        out.append(Path(rb).expanduser())
    rel = None
    if isinstance(probe, Mapping):
        raw = probe.get("binary")
        if isinstance(raw, str) and raw.strip():
            rel = raw.strip()
    if rel:
        out.append(REPO / rel)
        parent = _checkout_parent()
        if parent is not None:
            out.append(parent / rel)
    seen: set[str] = set()
    uniq: list[Path] = []
    for path in out:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(path)
    return uniq


# ---------------------------------------------------------------------------
# Launch spec. Resolved from the live probe + sealed identity, or injected.
# ---------------------------------------------------------------------------


def load_live_probe() -> tuple[dict[str, Any] | None, str]:
    return _load_json_rel(LIVE_PROBE_REL)


def resolve_launch(overlay: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Name the sealed argv/env. Does not spawn. Missing inputs refuse later at start()."""
    if overlay is not None and overlay.get("explicit"):
        return dict(overlay)
    ident_src, ident = load_authority(SEALED_REL)
    probe, probe_src = load_live_probe()
    if not isinstance(ident, dict):
        return {
            "present": False,
            "source": ident_src,
            "reason": f"{SEALED_REL} was not locatable as a JSON object ({ident_src})",
            "protocol": PROTOCOL,
            "argv": None,
            "gpu_authority": False,
            "started_model_process": False,
        }
    probe_doc = probe if isinstance(probe, dict) else {}
    identity = str(ident.get("resident_identity") or "sealed-3.14")
    protocol = str(ident.get("protocol") or PROTOCOL)
    if protocol != PROTOCOL:
        return {
            "present": False,
            "source": ident_src,
            "reason": f"sealed protocol {protocol!r} is not {PROTOCOL}",
            "protocol": protocol,
            "argv": None,
            "gpu_authority": False,
            "started_model_process": False,
        }
    artifact_root = ident.get("artifact_root")
    tokenizer = ident.get("tokenizer")
    max_seq_len = ident.get("max_seq_len")
    fusion = ident.get("fusion_env") if isinstance(ident.get("fusion_env"), dict) else {}
    require_fusion = ident.get("require_fusion_env")
    binaries = _binary_candidates(ident, probe_doc)
    binary: Path | None = None
    for cand in binaries:
        if cand.is_file():
            binary = cand
            break
    art_ok = isinstance(artifact_root, str) and Path(artifact_root).is_dir()
    tok_ok = isinstance(tokenizer, str) and Path(tokenizer).is_file()
    seq_ok = isinstance(max_seq_len, int) and not isinstance(max_seq_len, bool) and max_seq_len > 0
    fusion_ok = True
    if require_fusion is True:
        needed = (
            "HAWKING_QWEN38_FUSE_ADD_RMSNORM",
            "HAWKING_QWEN38_FUSE_GQA_QKV",
            "HAWKING_QWEN38_FUSE_DN_INPROJ",
            "HAWKING_QWEN38_FUSE_MLP",
        )
        fusion_ok = all(isinstance(fusion.get(k), str) and fusion[k] for k in needed)
    missing: list[str] = []
    if binary is None:
        missing.append("resident_binary")
    if not art_ok:
        missing.append("artifact_root")
    if not tok_ok:
        missing.append("tokenizer")
    if not seq_ok:
        missing.append("max_seq_len")
    if not fusion_ok:
        missing.append("fusion_env")
    present = not missing
    argv = None
    if present and binary is not None:
        argv = [
            str(binary.resolve()),
            "--artifact-root",
            str(artifact_root),
            "--tokenizer",
            str(tokenizer),
            "--max-seq-len",
            str(max_seq_len),
            "--resident-identity",
            identity,
        ]
    env = {str(k): str(v) for k, v in fusion.items() if isinstance(k, str) and isinstance(v, str)}
    digest = _file_digest(binary) if binary is not None else None
    reason = None if present else (
        "sealed launch is not startable; missing " + ", ".join(missing)
    )
    return {
        "present": present,
        "source": f"sealed:{ident_src}+probe:{probe_src}",
        "reason": reason,
        "missing": missing,
        "identity": identity,
        "protocol": protocol,
        "model_id": ident.get("model_id"),
        "binary": str(binary) if binary is not None else (str(ident.get("resident_binary") or "") or None),
        "binary_candidates": [str(p) for p in binaries],
        "artifact_root": artifact_root if isinstance(artifact_root, str) else None,
        "tokenizer": tokenizer if isinstance(tokenizer, str) else None,
        "max_seq_len": max_seq_len if seq_ok else None,
        "argv": argv,
        "env": env,
        "cwd": str(REPO),
        "artifact_digest": digest,
        "artifact_digest_kind": "resident_binary_sha256" if digest else "UNHASHED",
        "require_fusion_env": require_fusion,
        "probe": {
            "source": probe_src,
            "verdict": probe_doc.get("verdict"),
            "protocol": probe_doc.get("protocol"),
            "resident_identity": probe_doc.get("resident_identity"),
            "reached_ready": (probe_doc.get("observed") or {}).get("reached_ready")
            if isinstance(probe_doc.get("observed"), dict)
            else None,
            "requests_served": (probe_doc.get("observed") or {}).get("requests_served")
            if isinstance(probe_doc.get("observed"), dict)
            else None,
            "binary_rel": probe_doc.get("binary"),
        },
        "stdin": "PIPE",
        "stdin_why": (
            "the first live probe died because nohup gave it /dev/null on stdin; "
            "a request-driven resident reaching EOF is not a resident that failed to start"
        ),
        "gpu_authority": False,
        "started_model_process": False,
        "evidence_class": "STATIC_ONLY",
    }


def explicit_launch(
    *,
    argv: list[str],
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    identity: str = "sealed-3.14",
    protocol: str = PROTOCOL,
    binary: str | None = None,
) -> dict[str, Any]:
    """Injected launch (protocol double or a named binary). Relative argv[0] is refused."""
    if not argv or not isinstance(argv, list):
        raise ProviderRefuse("explicit argv must be a non-empty list", fault="bad_launch")
    argv0 = Path(str(argv[0]))
    if not argv0.is_absolute():
        raise ProviderRefuse(
            "argv[0] is not absolute; refusing PATH resolution at start",
            fault="runtime_resolution_refused",
        )
    body_path = Path(binary) if binary else None
    if body_path is None:
        for tok in argv[1:]:
            token = Path(str(tok))
            if str(tok).endswith(".py"):
                body_path = token
                break
            if token.is_file() and os.access(str(token), os.X_OK):
                body_path = token
                break
    child_env = {str(k): str(v) for k, v in (env or {}).items()}
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    cwd_s = str(Path(cwd).resolve()) if cwd is not None else str(REPO)
    present = argv0.exists()
    if body_path is not None:
        present = present and body_path.is_file()
    digest = _file_digest(body_path) if body_path is not None else None
    return {
        "present": present,
        "source": "explicit",
        "reason": None if present else "explicit argv[0] or binary is not on disk",
        "missing": [] if present else ["explicit_binary"],
        "identity": identity,
        "protocol": protocol,
        "binary": str(body_path) if body_path is not None else str(argv0),
        "argv": [str(x) for x in argv],
        "env": child_env,
        "cwd": cwd_s,
        "artifact_digest": digest,
        "artifact_digest_kind": "named_binary_sha256" if digest else "UNHASHED",
        "stdin": "PIPE",
        "gpu_authority": False,
        "started_model_process": False,
        "evidence_class": "STATIC_ONLY",
        "explicit": True,
    }


def _seed_chat_template(art: Path) -> None:
    """Put the artifact jinja next to the double so ask() does not invent a format."""
    dest = art / CHAT_TEMPLATE_NAME
    _, ident = load_authority(SEALED_REL)
    src: Path | None = None
    if isinstance(ident, dict) and isinstance(ident.get("artifact_root"), str):
        cand = Path(ident["artifact_root"]) / CHAT_TEMPLATE_NAME
        if cand.is_file():
            src = cand
    if src is not None:
        dest.write_bytes(src.read_bytes())
        for name in ("tokenizer_config.json", "generation_config.json"):
            side = src.parent / name
            if side.is_file():
                (art / name).write_bytes(side.read_bytes())
        return
    dest.write_text(QWEN_TEMPLATE_STUB)


def write_protocol_double(root: str | os.PathLike[str], *, mode: str = "ok", sleep_s: float = 0.0) -> dict[str, Any]:
    """Materialize the protocol double. Tests and proofs inject this, never the 9.9GB body."""
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    script = path / "resident_protocol_double.py"
    script.write_text(PROTOCOL_DOUBLE_SOURCE)
    art = path / "artifact"
    art.mkdir(exist_ok=True)
    tok = art / "tokenizer.json"
    if not tok.is_file():
        tok.write_text("{}\n")
    _seed_chat_template(art)
    env = {
        "HAWKING_PROVIDER_DOUBLE": str(mode),
        "PYTHONUNBUFFERED": "1",
    }
    if sleep_s:
        env["HAWKING_PROVIDER_DOUBLE_SLEEP_S"] = str(float(sleep_s))
    return explicit_launch(
        argv=[
            str(Path(_sys.executable).resolve()),
            str(script.resolve()),
            "--artifact-root",
            str(art.resolve()),
            "--tokenizer",
            str(tok.resolve()),
            "--max-seq-len",
            "8192",
            "--resident-identity",
            "sealed-3.14",
        ],
        env=env,
        cwd=str(path.resolve()),
        identity="sealed-3.14",
        binary=str(script.resolve()),
    )


# ---------------------------------------------------------------------------
# Inference slot: held only while a request is on the wire.
# ---------------------------------------------------------------------------


class InferenceSlot:
    """One in-flight JSONL request. Idle does not pin the resident."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._busy = False
        self._in_flight: str | None = None
        self._waiters = 0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "in_flight": self._in_flight,
                "waiters": self._waiters,
                "queue_depth": self._waiters + (1 if self._busy else 0),
                "busy": self._busy,
            }

    @contextmanager
    def hold(self, request_id: str, *, deadline: float) -> Iterator[None]:
        with self._cv:
            self._waiters += 1
            while self._busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._waiters -= 1
                    raise ProviderRefuse(
                        f"inference slot wait exceeded for {request_id}; not queued forever",
                        fault="slot_timeout",
                    )
                self._cv.wait(timeout=min(0.05, remaining))
            self._waiters -= 1
            self._busy = True
            self._in_flight = request_id
        try:
            yield
        finally:
            with self._cv:
                self._busy = False
                self._in_flight = None
                self._cv.notify()


# ---------------------------------------------------------------------------
# Provider. One Popen, many logical sessions.
# ---------------------------------------------------------------------------


class ResidentProvider:
    """Owns one resident process. Logical sessions share its weights."""

    def __init__(self) -> None:
        self._life = threading.RLock()
        self._slot = InferenceSlot()
        self._proc: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._ready_payload: dict[str, Any] | None = None
        self._spec: dict[str, Any] | None = None
        self._handle: dict[str, Any] | None = None
        self._sessions: dict[str, dict[str, Any]] = {}
        self._requests_served = 0
        self._failures = 0
        self._generation = 0
        self._stderr_path: str | None = None
        self._spawned_at: float | None = None
        self._ready_at: float | None = None
        self._start_token: str | None = None
        self._template: dict[str, Any] | None = None
        self._thinking_declared: dict[str, Any] | None = None
        self._last_wire_prompt: str | None = None
        self._last_render: dict[str, Any] | None = None
        self._applied_thinking: bool = False

    def _alive(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def _stderr_tail(self, limit: int = 2000) -> str:
        if not self._stderr_path:
            return ""
        try:
            return Path(self._stderr_path).read_text(encoding="utf-8", errors="replace")[-limit:]
        except OSError:
            return ""

    def _reader_loop(self, proc: subprocess.Popen[str]) -> None:
        stream = proc.stdout
        try:
            if stream is not None:
                for line in stream:
                    self._lines.put(line.rstrip("\r\n"))
        finally:
            self._lines.put(None)

    def _next_line(self, timeout_s: float) -> str | None:
        try:
            return self._lines.get(timeout=max(0.001, timeout_s))
        except queue.Empty:
            raise ProviderRefuse(
                f"protocol read exceeded {timeout_s:.3f}s",
                fault="timeout",
            )

    def _parse_line(self, line: str) -> dict[str, Any]:
        try:
            body = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise MalformedReply(f"stdout is not JSONL: {line[:400]!r} ({exc})") from exc
        if not isinstance(body, dict):
            raise MalformedReply("JSONL record is not an object")
        return body

    def start(
        self,
        spec: Mapping[str, Any] | None = None,
        *,
        ready_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Spawn one body, wait for status ready. Reuses a live ready process."""
        with self._life:
            if self._alive() and self._ready_payload is not None:
                return dict(self._handle or {})
            if self._alive() and self._ready_payload is None:
                raise NotReady("load in progress; refusing a second model body")
            launch = dict(spec) if spec is not None else resolve_launch()
            if not launch.get("present"):
                raise ProviderRefuse(
                    launch.get("reason") or "launch spec is not present on disk",
                    fault="launch_absent",
                )
            argv = launch.get("argv")
            if not isinstance(argv, list) or not argv:
                raise ProviderRefuse("launch spec has no argv", fault="launch_absent")
            argv0 = Path(str(argv[0]))
            if not argv0.is_absolute():
                raise ProviderRefuse(
                    "argv[0] is not absolute; refusing PATH resolution at start",
                    fault="runtime_resolution_refused",
                )
            art_root = _artifact_root_from_launch(launch)
            template = load_chat_template(art_root)
            if not template.get("present"):
                raise TemplateRefuse(
                    template.get("reason") or "chat_template.jinja is absent; refusing a raw prompt"
                )
            thinking = load_declared_thinking()
            # Sealed body declares false. If undeclared, still apply closed-think:
            # the jinja default (thinking on) is the 0/43 arm. Record the choice.
            applied = False
            if thinking.get("present") is True and thinking.get("enable_thinking") is True:
                applied = True
            self._template = template
            self._thinking_declared = thinking
            self._applied_thinking = applied
            timeout = float(
                DOUBLE_READY_TIMEOUT_S
                if launch.get("source") == "explicit" and ready_timeout_s is None
                else (DEFAULT_READY_TIMEOUT_S if ready_timeout_s is None else ready_timeout_s)
            )
            cwd = str(launch.get("cwd") or REPO)
            child_env = os.environ.copy()
            extra = launch.get("env") if isinstance(launch.get("env"), dict) else {}
            for key, value in extra.items():
                if isinstance(key, str) and isinstance(value, str):
                    child_env[key] = value
            err_fh = tempfile.NamedTemporaryFile(
                prefix="hawking-resident-provider-",
                suffix=".stderr.log",
                mode="w",
                delete=False,
            )
            self._stderr_path = err_fh.name
            self._lines = queue.Queue()
            self._ready_payload = None
            self._spawned_at = time.monotonic()
            try:
                proc = subprocess.Popen(
                    [str(x) for x in argv],
                    cwd=cwd,
                    env=child_env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=err_fh,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    start_new_session=True,
                )
            except OSError as exc:
                err_fh.close()
                raise ProviderRefuse(
                    f"named artifact could not be spawned: {type(exc).__name__}: {exc}",
                    fault="spawn_failed",
                ) from exc
            finally:
                try:
                    err_fh.close()
                except OSError as close_exc:
                    _ = close_exc
            self._proc = proc
            self._spec = launch
            self._generation += 1
            reader = threading.Thread(
                target=self._reader_loop,
                args=(proc,),
                name="future-resident-provider-reader",
                daemon=True,
            )
            self._reader = reader
            reader.start()
            token = None
            token_deadline = time.monotonic() + SPAWN_TOKEN_S
            while time.monotonic() < token_deadline:
                token = process_start_token(proc.pid)
                if token:
                    break
                if proc.poll() is not None:
                    break
                time.sleep(0.02)
            self._start_token = token
            try:
                self._await_ready(timeout)
            except Exception:
                self._failures += 1
                self._kill_unlocked()
                raise
            ready = dict(self._ready_payload or {})
            _assert_one_body(ready, where="ready")
            ready_elapsed = None
            if self._spawned_at is not None and self._ready_at is not None:
                ready_elapsed = round(self._ready_at - self._spawned_at, 6)
            handle = {
                "pid": proc.pid,
                "pgid": proc.pid,
                "start_token": token,
                "identity": launch.get("identity"),
                "protocol": ready.get("protocol") or launch.get("protocol") or PROTOCOL,
                "artifact_digest": launch.get("artifact_digest"),
                "artifact_digest_kind": launch.get("artifact_digest_kind"),
                "binary": launch.get("binary"),
                "argv": list(argv),
                "ready_elapsed_s": ready_elapsed,
                "ready_elapsed_evidence_class": "SELF_MEASURED_DIRTY",
                "model_open_count": ready.get("model_open_count"),
                "weight_upload_count": ready.get("weight_upload_count"),
                "generation": self._generation,
                "source": launch.get("source"),
                "gpu_authority": False,
                "sealed_body": launch.get("source", "").startswith("sealed"),
            }
            self._handle = handle
            return dict(handle)

    def _await_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise NotReady(
                    "resident did not emit status ready before timeout; "
                    f"stderr_tail={self._stderr_tail()!r}"
                )
            try:
                item = self._next_line(remaining)
            except ProviderRefuse as exc:
                if exc.fault == "timeout":
                    raise NotReady(
                        "resident did not emit status ready before timeout; "
                        f"stderr_tail={self._stderr_tail()!r}"
                    ) from exc
                raise
            if item is None:
                code = self._proc.returncode if self._proc is not None else None
                raise NotReady(
                    f"resident exited before ready (returncode={code}); "
                    f"stderr_tail={self._stderr_tail()!r}"
                )
            body = self._parse_line(item)
            if body.get("status") == "ready":
                self._ready_payload = _public_record(body, READY_KEEP)
                self._ready_at = time.monotonic()
                return
            raise MalformedReply(f"non-ready startup record: {body!r}")

    def handle(self) -> dict[str, Any]:
        if self._handle is None:
            raise NotReady("provider has no handle; start() was not called")
        return dict(self._handle)

    def open_session(self, session_id: str | None = None) -> str:
        """Logical session over the current process. Does not spawn."""
        sid = session_id if session_id is not None else f"s-{uuid.uuid4().hex[:12]}"
        sid = _nonempty_str(sid, name="session")
        with self._life:
            if sid not in self._sessions:
                self._sessions[sid] = {
                    "id": sid,
                    "turns": [],
                    "created_at_unix": time.time(),
                    "process_generation": self._generation,
                }
        return sid

    def sessions(self) -> dict[str, Any]:
        """Independent logical session states. One pid. No second 9.9GB body."""
        with self._life:
            rows = []
            for sid, rec in self._sessions.items():
                turns = rec.get("turns") or []
                rows.append(
                    {
                        "id": sid,
                        "n_turns": len(turns),
                        "last_request_id": None if not turns else turns[-1].get("id"),
                        "process_generation": rec.get("process_generation"),
                        "templated": bool(rec.get("templated")),
                    }
                )
            pid = self._proc.pid if self._proc is not None else None
            return {
                "n": len(rows),
                "pid": pid,
                "generation": self._generation,
                "same_process": True,
                "second_model_body": False,
                "templated": True,
                "concatenated": False,
                "sessions": rows,
            }

    def _compose_prompt(self, session: Mapping[str, Any], text: str) -> str:
        """Template session turns as real chat turns. Never concatenate a blob."""
        tmpl = self._template
        if not isinstance(tmpl, dict) or not tmpl.get("present"):
            raise TemplateRefuse(
                (tmpl or {}).get("reason")
                if isinstance(tmpl, dict)
                else "chat template is not loaded; refusing a raw concatenated prompt"
            )
        template_text = tmpl.get("text")
        if not isinstance(template_text, str) or not template_text:
            raise TemplateRefuse("loaded chat template has no text")
        messages: list[dict[str, Any]] = []
        for row in session.get("turns") or []:
            if not isinstance(row, dict):
                continue
            prompt = row.get("prompt")
            answer = row.get("text")
            if isinstance(prompt, str) and prompt.strip():
                messages.append({"role": "user", "content": prompt})
                if isinstance(answer, str):
                    messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": text})
        rendered = render_chat(
            messages,
            enable_thinking=self._applied_thinking,
            template_text=template_text,
            template_sha256=str(tmpl.get("sha256") or ""),
        )
        self._last_wire_prompt = rendered["text"]
        self._last_render = rendered
        return rendered["text"]

    def ask(
        self,
        session: str,
        text: str,
        max_new_tokens: int,
        *,
        timeout_s: float = DEFAULT_ASK_TIMEOUT_S,
    ) -> dict[str, Any]:
        """One correlated request/reply. Refuses before ready. Does not queue forever."""
        sid = _nonempty_str(session, name="session")
        prompt_text = _nonempty_str(text, name="text")
        n_new = _positive_int(max_new_tokens, name="max_new_tokens")
        if self._ready_payload is None or not self._alive():
            self._failures += 1
            if not self._alive() and self._proc is not None:
                raise DeadProcess("resident is not alive; ask refused")
            raise NotReady("ask before ready is refused, not queued")
        request_id = uuid.uuid4().hex
        deadline = time.monotonic() + float(timeout_s)
        with self._slot.hold(request_id, deadline=deadline):
            with self._life:
                if self._ready_payload is None:
                    self._failures += 1
                    raise NotReady("ask before ready is refused, not queued")
                if not self._alive():
                    self._failures += 1
                    raise DeadProcess("resident died before the request was written")
                rec = self._sessions.get(sid)
                if rec is None:
                    rec = {
                        "id": sid,
                        "turns": [],
                        "created_at_unix": time.time(),
                        "process_generation": self._generation,
                    }
                    self._sessions[sid] = rec
                wire_prompt = self._compose_prompt(rec, prompt_text)
                body = {
                    "id": request_id,
                    "prompt": wire_prompt,
                    "max_new_tokens": n_new,
                }
                proc = self._proc
                if proc is None or proc.stdin is None:
                    self._failures += 1
                    raise DeadProcess("resident stdin is unavailable")
                t0 = time.monotonic()
                try:
                    proc.stdin.write(json.dumps(body, separators=(",", ":")) + "\n")
                    proc.stdin.flush()
                except (BrokenPipeError, OSError) as exc:
                    self._failures += 1
                    raise DeadProcess(f"resident write failed: {exc}") from exc
            remaining = deadline - time.monotonic()
            try:
                item = self._next_line(remaining)
            except ProviderRefuse:
                self._failures += 1
                raise
            if item is None:
                self._failures += 1
                raise DeadProcess("resident closed stdout during ask")
            try:
                reply = self._parse_line(item)
            except MalformedReply:
                self._failures += 1
                raise
            rid = reply.get("id")
            if rid != request_id:
                self._failures += 1
                raise MalformedReply(
                    f"reply id {rid!r} does not match request {request_id}; "
                    "not treated as an answer"
                )
            status = reply.get("status")
            if status == "error":
                self._failures += 1
                raise AskFailed(str(reply.get("error") or "resident status=error"))
            if status != "ok":
                self._failures += 1
                raise MalformedReply(
                    f"reply status {status!r} is not ok; not treated as an answer"
                )
            try:
                open_c, up_c = _assert_one_body(reply, where="ask")
            except ProviderRefuse:
                self._failures += 1
                raise
            elapsed = round(time.monotonic() - t0, 6)
            public = _public_record(reply, REPLY_KEEP)
            leaked = _hardware_numeric_keys(public)
            if leaked:
                self._failures += 1
                raise ProviderRefuse(
                    f"public reply still carried hardware fields {leaked}",
                    fault="hardware_leak",
                )
            raw_text = public.get("text") or public.get("generated_text") or ""
            if not isinstance(raw_text, str):
                raw_text = str(raw_text)
            try:
                final = finalize_generation(
                    text=raw_text,
                    generated_tokens=public.get("generated_tokens"),
                    max_new_tokens=n_new,
                    new_token_ids=public.get("new_token_ids"),
                )
            except ProviderRefuse:
                self._failures += 1
                raise
            sealed_body = bool((self._spec or {}).get("source", "").startswith("sealed"))
            arm = thinking_arm_record(
                declared=self._thinking_declared or {},
                applied=self._applied_thinking,
                observed=str(final["observed_thinking"]),
                observed_on="sealed_resident" if sealed_body else "protocol_double",
            )
            turn = {
                "id": request_id,
                "prompt": prompt_text,
                "text": final["text"],
                "raw_text": final["raw_text"],
                "quality": final["quality"],
                "stopped_at": final["stopped_at"],
                "generated_tokens": public.get("generated_tokens"),
                "fallbacks": public.get("fallbacks"),
                "templated": True,
            }
            with self._life:
                rec["turns"].append(turn)
                rec["process_generation"] = self._generation
                rec["templated"] = True
                self._requests_served += 1
            return {
                "id": request_id,
                "session": sid,
                "status": "ok",
                "text": turn["text"],
                "raw_text": turn["raw_text"],
                "quality": final["quality"],
                "stopped_at": final["stopped_at"],
                "extractable": final["extractable"],
                "templated": True,
                "template_sha256": (self._template or {}).get("sha256"),
                "template_path": (self._template or {}).get("path"),
                "thinking_arm": arm,
                "generated_tokens": public.get("generated_tokens"),
                "max_new_tokens": n_new,
                "new_token_ids": public.get("new_token_ids"),
                "fallbacks": public.get("fallbacks"),
                "model_open_count": open_c,
                "weight_upload_count": up_c,
                "pid": self._proc.pid if self._proc is not None else None,
                "generation": self._generation,
                "cost": {
                    "evidence_class": "SELF_MEASURED_DIRTY",
                    "gpu_authority": False,
                    "elapsed_s": elapsed,
                    "generated_tokens": public.get("generated_tokens"),
                    "fallbacks": public.get("fallbacks"),
                    "ranks_nothing": True,
                },
            }

    def health(self) -> dict[str, Any]:
        """Snapshot. A dead pid is ABSENT with rss_bytes null, never healthy-with-zero."""
        slot = self._slot.snapshot()
        proc = self._proc
        pid = proc.pid if proc is not None else None
        if pid is None:
            presence = "UNDECLARED"
            alive = False
            rss = None
            reason = "no resident pid; will not invent one from the largest RSS neighbour"
        elif proc is not None and proc.poll() is not None:
            # poll() reaps our child, including a zombie os.kill(pid,0) still sees.
            presence = "ABSENT"
            alive = False
            rss = None
            reason = "resident pid is not alive"
        elif not pid_is_alive(pid):
            presence = "ABSENT"
            alive = False
            rss = None
            reason = "resident pid is not alive"
        else:
            presence = "PRESENT"
            alive = True
            rss = rss_bytes_of(int(pid))
            reason = None
            if rss is None:
                reason = "rss unreadable; left null, not 0"
        ready = bool(self._ready_payload is not None and alive)
        ready_pub = dict(self._ready_payload or {})
        return {
            "presence": presence,
            "alive": alive,
            "ready": ready,
            "dead": presence == "ABSENT",
            "pid": pid,
            "start_token": self._start_token,
            "rss_bytes": rss,
            "requests_served": self._requests_served,
            "failures": self._failures,
            "queue_depth": slot["queue_depth"],
            "in_flight": slot["in_flight"],
            "waiters": slot["waiters"],
            "sessions": len(self._sessions),
            "generation": self._generation,
            "model_open_count": ready_pub.get("model_open_count"),
            "weight_upload_count": ready_pub.get("weight_upload_count"),
            "identity": (self._spec or {}).get("identity"),
            "reason": reason,
            "evidence_class": "SELF_MEASURED_DIRTY",
            "gpu_authority": False,
            "authorizes": "liveness_only",
        }

    def _kill_unlocked(self) -> dict[str, Any]:
        proc = self._proc
        pid = proc.pid if proc is not None else None
        needed = "already_dead"
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except OSError as exc:
                _ = exc
            if proc.poll() is None:
                needed = "stdin_eof"
                try:
                    proc.wait(timeout=STOP_GRACE_S)
                except subprocess.TimeoutExpired:
                    needed = "cooperative"
                    try:
                        proc.terminate()
                        proc.wait(timeout=STOP_GRACE_S)
                    except (OSError, subprocess.TimeoutExpired):
                        needed = "escalated"
                        try:
                            if pid:
                                try:
                                    os.killpg(pid, signal.SIGKILL)
                                except OSError:
                                    proc.kill()
                            else:
                                proc.kill()
                            proc.wait(timeout=1.0)
                        except (OSError, subprocess.TimeoutExpired):
                            needed = "refused"
        gone = proc is None or proc.poll() is not None
        self._proc = None
        self._ready_payload = None
        self._handle = None
        self._ready_at = None
        reader = self._reader
        self._reader = None
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1.0)
        return {
            "stopped": gone,
            "needed": needed,
            "pid": pid,
            "result": "ok" if gone else "refused",
        }

    def stop(self) -> dict[str, Any]:
        """Close stdin (protocol EOF), then SIGTERM, then SIGKILL."""
        with self._life:
            out = self._kill_unlocked()
            out["gpu_authority"] = False
            return out

    def restart(self, *, ready_timeout_s: float | None = None) -> dict[str, Any]:
        """Stop, spawn the same spec, wait for ready, serve again. Sessions keep history."""
        with self._life:
            spec = dict(self._spec) if self._spec is not None else None
            if spec is None:
                raise ProviderRefuse(
                    "restart has no remembered launch spec; start() first",
                    fault="launch_absent",
                )
            stopped = self._kill_unlocked()
        handle = self.start(spec, ready_timeout_s=ready_timeout_s)
        return {
            "stopped": stopped,
            "handle": handle,
            "generation": self._generation,
            "pid": handle.get("pid"),
            "ready": True,
            "gpu_authority": False,
        }


# Module-level names the lane contract quotes. Each needs a provider.
def start(spec: Mapping[str, Any] | None = None, **kwargs: Any) -> tuple[ResidentProvider, dict[str, Any]]:
    provider = ResidentProvider()
    handle = provider.start(spec, **kwargs)
    return provider, handle


# ---------------------------------------------------------------------------
# Proofs. Declared capability is not evidence; these actually fire.
# ---------------------------------------------------------------------------


def _demand(name: str, cond: bool, detail: Any) -> dict[str, Any]:
    if not cond:
        raise ProviderRefuse(f"proof {name} failed: {detail}", fault="proof_failed")
    return {"fires": True, "detail": detail}


def run_proofs(tmp: Path | None = None) -> dict[str, Any]:
    """Hermetic protocol-double proofs. Does not spawn the 9.9GB sealed body."""
    own = tmp is None
    root = Path(tmp) if tmp is not None else Path(tempfile.mkdtemp(prefix="resident-provider-"))
    proofs: dict[str, Any] = {}
    living: list[ResidentProvider] = []

    def _prov() -> ResidentProvider:
        inst = ResidentProvider()
        living.append(inst)
        return inst

    try:
        spec = write_protocol_double(root / "ok", mode="ok")
        p = _prov()
        handle = p.start(spec, ready_timeout_s=DOUBLE_READY_TIMEOUT_S)
        proofs["start_reaches_ready"] = _demand(
            "start_reaches_ready",
            isinstance(handle.get("pid"), int) and handle.get("model_open_count") == 1,
            {"pid": handle.get("pid"), "open": handle.get("model_open_count")},
        )
        pid1 = handle["pid"]
        r1 = p.ask("s1", "What is the capital of France?", 3)
        r2 = p.ask("s1", "Name one reason a status label can be wrong.", 8)
        proofs["reuse_same_pid"] = _demand(
            "reuse_same_pid",
            r1.get("pid") == pid1 and r2.get("pid") == pid1 and p.health()["requests_served"] == 2,
            {"pid": pid1, "served": p.health()["requests_served"]},
        )
        proofs["one_body_across_asks"] = _demand(
            "one_body_across_asks",
            r1["model_open_count"] == 1
            and r2["model_open_count"] == 1
            and r1["weight_upload_count"] == 1
            and r2["weight_upload_count"] == 1,
            {"r1": r1["model_open_count"], "r2": r2["model_open_count"]},
        )
        p.ask("s2", "second session", 2)
        sess = p.sessions()
        proofs["two_sessions_one_pid"] = _demand(
            "two_sessions_one_pid",
            sess["n"] == 2 and sess["pid"] == pid1 and sess["second_model_body"] is False,
            sess,
        )
        idle = p.health()
        proofs["slot_idle_when_nothing_in_flight"] = _demand(
            "slot_idle_when_nothing_in_flight",
            idle["queue_depth"] == 0 and idle["in_flight"] is None and idle["presence"] == "PRESENT",
            {"queue_depth": idle["queue_depth"], "in_flight": idle["in_flight"]},
        )
        restarted = p.restart(ready_timeout_s=DOUBLE_READY_TIMEOUT_S)
        pid2 = restarted["handle"]["pid"]
        r3 = p.ask("s1", "after restart", 2)
        proofs["restart_reaches_ready_and_serves"] = _demand(
            "restart_reaches_ready_and_serves",
            pid2 != pid1 and r3.get("status") == "ok" and r3.get("model_open_count") == 1,
            {"old": pid1, "new": pid2, "text": r3.get("text")},
        )
        p.stop()
        proofs["stop_kills"] = _demand(
            "stop_kills",
            not pid_is_alive(int(pid2)),
            {"pid": pid2},
        )

        cold = _prov()
        try:
            cold.ask("s", "too early", 1)
            proofs["ask_before_ready"] = _demand("ask_before_ready", False, "ask succeeded before start")
        except NotReady as exc:
            proofs["ask_before_ready"] = _demand(
                "ask_before_ready", exc.fault == "not_ready", exc.reason
            )
        never = write_protocol_double(root / "never", mode="never_ready")
        loading = _prov()
        try:
            loading.start(never, ready_timeout_s=0.25)
            proofs["never_ready_refuses"] = _demand("never_ready_refuses", False, "start returned")
        except NotReady as exc:
            proofs["never_ready_refuses"] = _demand("never_ready_refuses", True, exc.reason)

        mortal = _prov()
        mortal.start(write_protocol_double(root / "diehealth", mode="ok"), ready_timeout_s=DOUBLE_READY_TIMEOUT_S)
        pid_m = mortal.handle()["pid"]
        os.kill(int(pid_m), signal.SIGKILL)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if mortal._proc is not None and mortal._proc.poll() is not None:
                break
            time.sleep(0.02)
        h_dead = mortal.health()
        proofs["dead_is_absent_rss_null"] = _demand(
            "dead_is_absent_rss_null",
            h_dead["presence"] == "ABSENT"
            and h_dead["rss_bytes"] is None
            and h_dead["alive"] is False
            and h_dead["dead"] is True,
            h_dead,
        )

        bad = _prov()
        bad.start(write_protocol_double(root / "malformed", mode="malformed"), ready_timeout_s=DOUBLE_READY_TIMEOUT_S)
        try:
            bad.ask("s", "ping", 1)
            proofs["malformed_raises"] = _demand("malformed_raises", False, "ask accepted non-JSON")
        except MalformedReply as exc:
            proofs["malformed_raises"] = _demand("malformed_raises", True, exc.reason)

        rel = _prov()
        rel.start(write_protocol_double(root / "reload", mode="reload"), ready_timeout_s=DOUBLE_READY_TIMEOUT_S)
        try:
            rel.ask("s", "ping", 1)
            proofs["reload_is_defect"] = _demand("reload_is_defect", False, "climbing open_count accepted")
        except WeightReload as exc:
            proofs["reload_is_defect"] = _demand("reload_is_defect", True, exc.reason)

        errp = _prov()
        errp.start(write_protocol_double(root / "err", mode="error_status"), ready_timeout_s=DOUBLE_READY_TIMEOUT_S)
        try:
            errp.ask("s", "ping", 1)
            proofs["error_status_not_answer"] = _demand(
                "error_status_not_answer", False, "status=error treated as answer"
            )
        except AskFailed as exc:
            proofs["error_status_not_answer"] = _demand("error_status_not_answer", True, exc.reason)

        dirty = _prov()
        dirty.start(write_protocol_double(root / "dirty", mode="dirty_metrics"), ready_timeout_s=DOUBLE_READY_TIMEOUT_S)
        got = dirty.ask("s", "ping", 1)
        leaked = _hardware_numeric_keys(got)
        proofs["dirty_metrics_stripped"] = _demand(
            "dirty_metrics_stripped", not leaked, {"leaked": leaked, "keys": sorted(got)}
        )

        missing = explicit_launch(
            argv=[str(Path(_sys.executable).resolve()), str(root / "no-such-double.py")],
            cwd=str(root),
            binary=str(root / "no-such-double.py"),
        )
        proofs["missing_binary_spec_not_present"] = _demand(
            "missing_binary_spec_not_present",
            missing["present"] is False,
            missing.get("reason"),
        )
        ghost = _prov()
        try:
            ghost.start(missing)
            proofs["missing_binary_start_refuses"] = _demand(
                "missing_binary_start_refuses", False, "start of missing binary succeeded"
            )
        except ProviderRefuse as exc:
            proofs["missing_binary_start_refuses"] = _demand(
                "missing_binary_start_refuses", exc.fault == "launch_absent", exc.reason
            )

        undeclared = _prov().health()
        proofs["undeclared_not_healthy_zero"] = _demand(
            "undeclared_not_healthy_zero",
            undeclared["presence"] == "UNDECLARED" and undeclared["rss_bytes"] is None,
            undeclared,
        )

        tmpl = load_chat_template()
        proofs["template_from_artifact_jinja"] = _demand(
            "template_from_artifact_jinja",
            bool(tmpl.get("present"))
            and str(tmpl.get("filename")) == CHAT_TEMPLATE_NAME
            and str(tmpl.get("path") or "").endswith(CHAT_TEMPLATE_NAME),
            {
                "path": tmpl.get("path"),
                "sha256": tmpl.get("sha256"),
                "reason": tmpl.get("reason"),
                "match": tmpl.get("tokenizer_config_chat_template_match"),
            },
        )
        if tmpl.get("present"):
            rendered = render_chat(
                [{"role": "user", "content": PROMPT_FRANCE}],
                enable_thinking=False,
                template_text=str(tmpl["text"]),
                template_sha256=str(tmpl.get("sha256") or ""),
            )
            gold = (
                f"{IM_START}user\n{PROMPT_FRANCE}{IM_END}\n"
                f"{IM_START}assistant\n{CLOSED_THINK}"
            )
            proofs["template_matches_artifact_gold"] = _demand(
                "template_matches_artifact_gold",
                rendered["text"] == gold
                and "User: " not in rendered["text"]
                and rendered["templated"] is True
                and rendered["concatenated"] is False,
                {"head": rendered["text"][:120], "n": len(rendered["text"])},
            )
            hist = render_chat(
                [
                    {"role": "user", "content": "one"},
                    {"role": "assistant", "content": "two"},
                    {"role": "user", "content": "three"},
                ],
                enable_thinking=False,
                template_text=str(tmpl["text"]),
                template_sha256=str(tmpl.get("sha256") or ""),
            )
            proofs["session_turns_templated_not_concatenated"] = _demand(
                "session_turns_templated_not_concatenated",
                f"{IM_START}user\none{IM_END}" in hist["text"]
                and f"{IM_START}assistant\n<think>\n\n</think>\n\ntwo{IM_END}" in hist["text"]
                and "User: one" not in hist["text"]
                and "Assistant: two" not in hist["text"],
                {"head": hist["text"][:220]},
            )
        else:
            proofs["template_matches_artifact_gold"] = _demand(
                "template_matches_artifact_gold", False, tmpl.get("reason")
            )
            proofs["session_turns_templated_not_concatenated"] = _demand(
                "session_turns_templated_not_concatenated", False, tmpl.get("reason")
            )

        deg = finalize_generation(
            text=MEASURED_DEGENERATE_CHOICE, generated_tokens=64, max_new_tokens=64
        )
        proofs["degenerate_never_clean"] = _demand(
            "degenerate_never_clean",
            deg["quality"] == QUALITY_DEGENERATE
            and deg["text"] != "hbm_doctor.py"
            and quality(
                {
                    "raw_text": MEASURED_DEGENERATE_CHOICE,
                    "generated_tokens": 64,
                    "max_new_tokens": 64,
                }
            )
            == QUALITY_DEGENERATE,
            {"text": deg["text"], "quality": deg["quality"]},
        )
        trunc = finalize_generation(
            text="A status label is a claim about",
            generated_tokens=16,
            max_new_tokens=16,
        )
        proofs["truncated_not_clean"] = _demand(
            "truncated_not_clean",
            trunc["quality"] == QUALITY_TRUNCATED,
            {"text": trunc["text"], "stopped_at": trunc["stopped_at"]},
        )
        clean = finalize_generation(
            text=MEASURED_CLEAN_FRANCE,
            generated_tokens=3,
            max_new_tokens=32,
            new_token_ids=[11, 12, EOS_IM_END_ID],
        )
        proofs["clean_paris"] = _demand(
            "clean_paris",
            clean["quality"] == QUALITY_CLEAN and clean["text"] == "Paris",
            clean,
        )

        thinking = load_declared_thinking()
        proofs["thinking_arm_declared_false"] = _demand(
            "thinking_arm_declared_false",
            thinking.get("present") is True and thinking.get("enable_thinking") is False,
            thinking,
        )

        qspec = write_protocol_double(root / "qscript", mode="quality_scripted")
        qp = _prov()
        qp.start(qspec, ready_timeout_s=DOUBLE_READY_TIMEOUT_S)
        r_fr = qp.ask("s1", PROMPT_FRANCE, 32)
        r_ch = qp.ask("s2", PROMPT_CHOICE, 32)
        r_st = qp.ask("s3", PROMPT_STATUS, 32)
        proofs["scripted_three_prompts_clean"] = _demand(
            "scripted_three_prompts_clean",
            r_fr["quality"] == QUALITY_CLEAN
            and r_fr["text"] == "Paris"
            and r_ch["quality"] == QUALITY_CLEAN
            and r_ch["text"] == "hbm_doctor.py"
            and r_st["quality"] == QUALITY_CLEAN
            and bool(r_st["text"]),
            {
                "france": r_fr["text"],
                "choice": r_ch["text"],
                "status": r_st["text"],
                "arm": r_fr.get("thinking_arm"),
            },
        )
        wire = qp._last_wire_prompt or ""
        proofs["ask_sends_templated_prompt"] = _demand(
            "ask_sends_templated_prompt",
            IM_START + "user" in wire
            and CLOSED_THINK in wire
            and "User: " not in wire
            and qp._applied_thinking is False,
            {"head": wire[:160], "applied_thinking": qp._applied_thinking},
        )
        qp.ask("hist", "one", 8)
        qp.ask("hist", "two", 8)
        hist_wire = qp._last_wire_prompt or ""
        proofs["ask_session_history_templated"] = _demand(
            "ask_session_history_templated",
            f"{IM_START}user\none{IM_END}" in hist_wire
            and "User: one" not in hist_wire
            and "Assistant: " not in hist_wire,
            {"head": hist_wire[:260]},
        )

        dspec = write_protocol_double(root / "deg", mode="degenerate")
        dp = _prov()
        dp.start(dspec, ready_timeout_s=DOUBLE_READY_TIMEOUT_S)
        drep = dp.ask("s", PROMPT_CHOICE, 64)
        proofs["ask_degenerate_labelled"] = _demand(
            "ask_degenerate_labelled",
            drep["quality"] == QUALITY_DEGENERATE and drep["text"] != "hbm_doctor.py",
            {"text": drep["text"], "quality": drep["quality"]},
        )
        tspec = write_protocol_double(root / "trunc", mode="truncated")
        tp = _prov()
        tp.start(tspec, ready_timeout_s=DOUBLE_READY_TIMEOUT_S)
        trep = tp.ask("s", PROMPT_STATUS, 16)
        proofs["ask_truncated_labelled"] = _demand(
            "ask_truncated_labelled",
            trep["quality"] == QUALITY_TRUNCATED,
            {"text": trep["text"], "stopped_at": trep["stopped_at"]},
        )

        sealed = resolve_launch()
        argv = sealed.get("argv") or []
        flags = [t for t in argv if str(t).startswith("--")]
        expected_flags = [
            "--artifact-root",
            "--tokenizer",
            "--max-seq-len",
            "--resident-identity",
        ]
        if argv:
            proofs["resolve_launch_names_probe_argv"] = _demand(
                "resolve_launch_names_probe_argv",
                flags == expected_flags
                and sealed.get("stdin") == "PIPE"
                and sealed.get("started_model_process") is False
                and sealed.get("identity") == "sealed-3.14",
                {"flags": flags, "present": sealed.get("present")},
            )
        else:
            proofs["resolve_launch_names_probe_argv"] = _demand(
                "resolve_launch_names_probe_argv",
                sealed.get("present") is False and sealed.get("stdin") == "PIPE",
                {"reason": sealed.get("reason"), "missing": sealed.get("missing")},
            )
    finally:
        for inst in living:
            try:
                inst.stop()
            except Exception as stop_exc:
                _ = stop_exc
        if own:
            try:
                import shutil

                shutil.rmtree(root, ignore_errors=True)
            except OSError as exc:
                _ = exc
    return {
        "n": len(proofs),
        "all_passed": all(row.get("fires") for row in proofs.values()),
        "proofs": proofs,
        "started_sealed_resident": False,
        "started_protocol_double": True,
        "gpu_authority": False,
    }


def emit_workunits() -> list[dict[str, Any]]:
    unit = emit_hcli_workunit(
        id="future.resident_provider.drive",
        role="science",
        description=(
            "Own one hawking.qwen38.resident.v1 process: start, templated ask "
            "with stop+quality, logical sessions over shared weights, health, "
            "stop, restart. CPU/protocol-double proven; sealed 9.9GB spawn is "
            "a later MODEL_SESSION."
        ),
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier="future.resident_provider.run_proofs",
        provider="future.resident_provider",
        effect_class="READ_ONLY",
        status="pending",
        classification="STATIC_ONLY",
        extras={
            "output_receipt_path": f"receipts/future/{RECEIPT}",
            "command": "python3 tools/future/resident_provider.py --build",
            "species": "CPU_ANALYSIS",
            "claim_boundary": (
                "WorkUnit is a proposal; receipt remains authoritative; "
                "this unit cannot take a GPU lease or rank a kernel."
            ),
            "may_promote": False,
            "may_modify_verifier": False,
        },
    )
    validate_emitted_unit(unit)
    return [unit]


def recovered_implementation() -> list[str]:
    return [
        "tools/future/resident_provider.py (start/ask/sessions/health/stop/restart; model_open_count=1; protocol double)",
        "receipts/future/evidence/RESIDENT_LIVE_PROBE.json (resident starts and generates; r1='\\n\\nParis' 3 tokens; r2 truncated at 40)",
        "hcli/hawking-native.sealed-3.14.json (generation.enable_thinking=false; prompt_contract.fallback_template=qwen_closed_think)",
        "/Users/scammermike/noetic/NOETIC_PARENT_A/chat_template.jinja (artifact template; thinking-off arm emits closed <think>)",
        "/Users/scammermike/noetic/NOETIC_PARENT_A/tokenizer_config.json (chat_template byte-identical to the jinja; eos=<|im_end|>)",
        "/Users/scammermike/noetic/NOETIC_PARENT_A/generation_config.json (eos_token_id 248046/248044)",
        "hcli/hawking_native.py _TokenizerRenderer._qwen_fallback_render (cited, not imported: closed-think generation prefix)",
        "crates/hawking-core/examples/ascension_qwen38_resident.rs (JSONL; raw prompt; generate_greedy)",
        "crates/hawking-core/src/model/qwen38_hybrid_decode.rs generate_greedy (stops on IM_END/EOT unless HAWKING_QWEN38_IGNORE_EOS)",
        "crates/hawking-serve/src/glm_chat.rs (pattern: validate artifact jinja anchors, render the no-tools path, refuse tools)",
        "tools/future/resident_identity.py (load_authority for the sealed profile)",
        "tools/future/resident_health.py (rss_bytes_of; ABSENT rss_bytes null)",
    ]


def gaps_closed() -> list[str]:
    return [
        "ask() applies the artifact chat_template.jinja; session history is real turns not a User:/Assistant: blob",
        "generation is cut at <|im_end|>, <|endoftext|>, <|im_start|>, and fabricated Assistant:/User: markers",
        "quality(reply) computes CLEAN | TRUNCATED | DEGENERATE; a repeating fragment is never CLEAN",
        "a budget-hit mid-sentence is TRUNCATED, not CLEAN",
        "thinking arm is recorded as declared/applied/observed; sealed enable_thinking=false is applied as closed-think",
        "a decision-style reply is extractable as text when CLEAN; hbm_doctor.py is not regex-fished from loop garbage",
        "a future-partition driver that spawns the JSONL resident, waits for ready, and correlates ask/reply",
        "logical sessions multiplexed over one process (no second 9.9GB body)",
        "inference slot held only while a request is in flight",
        "reuse proven: several asks through one start, same pid, model_open_count stays 1",
        "stop/restart that return to ready and serve",
        "stdin=PIPE (the live probe's first death was DEVNULL EOF)",
    ]


def negative_findings() -> list[str]:
    return [
        "build() did not spawn the sealed 9.9GB resident: a live campaign is running and this sidecar has no GPU lease",
        "the three quality probes were not re-run on the sealed body; CLEAN on those prompts is proven on the harness and a protocol double, not on this resident",
        "observed thinking-arm obedience on the sealed body is UNPROBED; a raw prompt producing <think> is consistent with the jinja default (thinking on) when the template is unused",
        "the rust resident still decodes with skip_special_tokens and does not accept stop strings on the wire; stop handling here is post-decode plus the kernel's existing EOS break",
        "timings are SELF_MEASURED_DIRTY and rank nothing",
        "orchestration.py BINDINGS was not edited (not in this lane's WRITE list)",
        "hcli/hawking_native.ResidentProcess remains the HCLI driver; this module does not replace it",
        "the rust resident resets KV per request; logical sessions are prompt-side history over shared weights",
        "restart_supervisor.restart cannot drive this protocol (stdin=DEVNULL)",
        "jinja2 is not a sidecar dependency; the renderer is the artifact template's text-only path, gold-checked, and refuses tools/vision",
    ]


def build() -> Path:
    sealed = resolve_launch()
    proofs = run_proofs()
    units = emit_workunits()
    probe, probe_src = load_live_probe()
    tmpl = load_chat_template()
    thinking = load_declared_thinking()
    template_public = {
        k: v for k, v in tmpl.items() if k != "text"
    }
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Durable provider that owns one hawking.qwen38.resident.v1 process "
            "and returns the model's ANSWER: artifact chat template applied, "
            "stopped at the turn boundary, quality labelled CLEAN|TRUNCATED|"
            "DEGENERATE."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "started_model_process": False,
        "started_sealed_resident": False,
        "started_protocol_double": True,
        "took_gpu_lease": False,
        "flock": False,
        "protocol": PROTOCOL,
        "live_probe": {
            "source": probe_src,
            "verdict": None if not isinstance(probe, dict) else probe.get("verdict"),
            "reached_ready": (
                (probe.get("observed") or {}).get("reached_ready")
                if isinstance(probe, dict) and isinstance(probe.get("observed"), dict)
                else None
            ),
        },
        "launch": {
            "present": sealed.get("present"),
            "source": sealed.get("source"),
            "reason": sealed.get("reason"),
            "missing": sealed.get("missing"),
            "identity": sealed.get("identity"),
            "protocol": sealed.get("protocol"),
            "binary": sealed.get("binary"),
            "argv_flags": None
            if not sealed.get("argv")
            else [t for t in sealed["argv"] if str(t).startswith("--")],
            "stdin": sealed.get("stdin"),
            "stdin_why": sealed.get("stdin_why"),
            "artifact_digest_kind": sealed.get("artifact_digest_kind"),
            "started_model_process": False,
        },
        "proofs": proofs,
        "work_units": units,
        "chat_template": template_public,
        "thinking_arm": {
            "declared": thinking.get("enable_thinking"),
            "declared_present": thinking.get("present"),
            "declared_source": thinking.get("source"),
            "declared_reason": thinking.get("reason"),
            "applied_by_ask": True,
            "applied_on_sealed_body": False,
            "applied_arm": "thinking_off",
            "applied_prefix": CLOSED_THINK,
            "applied_why": (
                "sealed generation.enable_thinking is false; the jinja arm "
                "emits <think>\\n\\n</think>\\n\\n so the model continues the "
                "answer instead of opening a think block"
            ),
            "observed_on_sealed_body": "UNPROBED",
            "observed_on_sealed_body_reason": (
                "build() did not spawn the 9.9GB body; config obedience is not assumed"
            ),
            "jinja_default_if_undefined": "thinking_on_with_xhigh_reasoning_system",
            "campaign_note": (
                "same body scored 0/43 with thinking on and 14/43 with it off"
            ),
        },
        "quality_probes": {
            "france": PROMPT_FRANCE,
            "choice": PROMPT_CHOICE,
            "status": PROMPT_STATUS,
            "proved_on": "protocol_double_plus_artifact_template",
            "proved_on_sealed_body": False,
            "measured_defect_replayed": {
                "choice": MEASURED_DEGENERATE_CHOICE,
                "classification": QUALITY_DEGENERATE,
                "did_not_extract": "hbm_doctor.py",
            },
        },
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "resident_callable": {
            "entry_point": "tools.future.resident_provider.ResidentProvider.start()",
            "workunit": (
                "one CPU_ANALYSIS unit; protocol-double start/templated-ask/"
                "stop+quality/sessions/health/stop/restart; does not take a GPU lease"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.CHILD_RESIDENT.launch",
            "fails_closed": (
                "ask before ready → NotReady; dead pid → health presence=ABSENT "
                "rss_bytes=null; climbing model_open_count → WeightReload; "
                "malformed JSONL → MalformedReply; missing binary → launch_absent; "
                "missing/foreign chat_template.jinja → TemplateRefuse; "
                "repeating fragment → DEGENERATE not CLEAN; "
                "budget-hit mid-sentence → TRUNCATED not CLEAN; "
                "hardware-named fields stripped and refused in receipts"
            ),
            "python_api": {
                "start": "ResidentProvider.start(spec=None) -> handle",
                "ask": "ResidentProvider.ask(session, text, max_new_tokens) -> reply+quality+cost",
                "quality": "quality(reply) -> CLEAN|TRUNCATED|DEGENERATE, recomputed",
                "render_chat": "render_chat(messages, enable_thinking, template_text) -> templated prompt",
                "sessions": "ResidentProvider.sessions() -> logical states, one pid",
                "health": "ResidentProvider.health() -> presence/rss/queue_depth",
                "stop": "ResidentProvider.stop()",
                "restart": "ResidentProvider.restart() -> ready handle",
                "resolve_launch": "resolve_launch() -> sealed argv, does not spawn",
            },
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    leaked = _hardware_numeric_keys(doc)
    if leaked:
        raise ProviderRefuse(f"receipt would carry hardware fields {leaked}", fault="hardware_leak")
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
