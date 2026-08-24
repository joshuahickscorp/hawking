"""Single authority for HCLI llama.cpp context-window arithmetic.

llama-server takes `--ctx-size TOTAL --parallel N` and then gives each
slot `ceil(TOTAL / N / 256) * 256` tokens. Every budget in HCLI must be
computed against that per-request ceiling, not against TOTAL.
"""
from __future__ import annotations

import json
import os
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

LLAMA_KV_PAD = 256
DEFAULT_PER_SLOT_CTX = 32768
DEFAULT_FRAMING_RESERVE = 4096
DEFAULT_GENERATION_RESERVE = 4096
# Matches engine._CHARS_PER_TOKEN. Packet preflight must use the same
# estimator the HTTP path uses or a "fits" verdict here still overflows there.
CHARS_PER_TOKEN = 3

_GGUF_MAGIC = b"GGUF"
_GGUF_STRING = 8
_GGUF_ARRAY = 9
_GGUF_SCALAR = {
    0: (1, "B"),
    1: (1, "b"),
    2: (2, "H"),
    3: (2, "h"),
    4: (4, "I"),
    5: (4, "i"),
    6: (4, "f"),
    7: (1, "?"),
    10: (8, "Q"),
    11: (8, "q"),
    12: (8, "d"),
}


class ContextBudgetError(RuntimeError):
    """Raised when a context budget cannot be applied to a request."""


class PacketBudgetError(ContextBudgetError):
    """Packet + evidence does not fit. Refused rather than silently sliced."""

    def __init__(
        self,
        message: str = "",
        *,
        omitted: tuple = (),
        shortfall: int = 0,
        result: Optional["PreflightResult"] = None,
        prompt_len: int = 0,
        cap: int = 0,
    ) -> None:
        self.omitted = tuple(omitted)
        self.result = result
        self.prompt_len = int(prompt_len)
        self.cap = int(cap)
        if result is not None:
            self.shortfall = int(result.shortfall)
            if not message:
                message = (
                    f"worker packet refused: demand {result.demand} exceeds "
                    f"per-request ctx {result.per_request_ctx} by {result.shortfall} "
                    f"(kind={result.kind}, omitted={list(self.omitted) or '(none)'}). "
                    f"{result.remedy}"
                )
        else:
            self.shortfall = int(shortfall)
            if not message:
                message = (
                    f"worker packet refused: prompt {self.prompt_len} chars "
                    f"exceeds cap {self.cap} by {self.shortfall}; "
                    f"omitted={list(self.omitted) or '(none)'}. "
                    f"Silent truncation is not allowed."
                )
        super().__init__(message)


@dataclass(frozen=True)
class ContextBudget:
    total_ctx: int
    n_parallel: int
    per_request_ctx: int
    generation_reserve: int
    framing_reserve: int
    usable_input_tokens: int
    source: str
    model_ceiling: Optional[int]
    provenance: Dict[str, Any]


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    demand: int
    usable: int
    per_request_ctx: int
    shortfall: int
    kind: str
    remedy: str
    budget: ContextBudget


def per_seq_context(
    total_ctx: int, n_parallel: int, pad: int = LLAMA_KV_PAD
) -> int:
    """llama.cpp per-sequence KV: ceil(total_ctx / n_parallel / pad) * pad."""
    total_ctx = int(total_ctx)
    n_parallel = max(1, int(n_parallel))
    pad = max(1, int(pad))
    if total_ctx <= 0:
        return 0
    quantum = n_parallel * pad
    return ((total_ctx + quantum - 1) // quantum) * pad


def solve_parallel(total_ctx: int, observed_per_seq: int) -> Optional[int]:
    """Recover n_parallel from (total_ctx, observed per-seq). Unique or None."""
    try:
        total_ctx = int(total_ctx)
        observed_per_seq = int(observed_per_seq)
    except (TypeError, ValueError):
        return None
    if total_ctx <= 0 or observed_per_seq <= 0:
        return None
    pad = LLAMA_KV_PAD
    upper = min(4096, max(1, total_ctx // pad)) + 2
    hits = [
        n
        for n in range(1, upper)
        if per_seq_context(total_ctx, n, pad) == observed_per_seq
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def _round_up(n: int, pad: int = LLAMA_KV_PAD) -> int:
    n = max(0, int(n))
    pad = max(1, int(pad))
    if n <= 0:
        return 0
    return ((n + pad - 1) // pad) * pad


def _generation_reserve(explicit: Optional[int]) -> int:
    if explicit is not None:
        try:
            value = int(explicit)
        except (TypeError, ValueError):
            value = None
        else:
            if value >= 0:
                return value
    raw = os.environ.get("HCLI_MODEL_TOKENS")
    if raw is not None and str(raw).strip() != "":
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = None
        else:
            if value >= 0:
                return value
    return DEFAULT_GENERATION_RESERVE


def _profile_genome_path() -> Path:
    return Path.home() / ".config" / "hcli" / "machine_genome.json"


def _read_profile_ctx() -> tuple[Optional[int], Dict[str, Any]]:
    path = _profile_genome_path()
    meta: Dict[str, Any] = {"path": str(path)}
    if not path.is_file():
        meta["reason"] = f"{path} is absent"
        return None, meta
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        meta["reason"] = f"{path} is unreadable"
        return None, meta
    if not isinstance(data, dict) or "context_ctx_size" not in data:
        meta["reason"] = "machine_genome.json has no context_ctx_size"
        return None, meta
    try:
        value = int(data["context_ctx_size"])
    except (TypeError, ValueError):
        meta["reason"] = "context_ctx_size is not an int"
        return None, meta
    if value <= 0:
        meta["reason"] = "context_ctx_size is not positive"
        return None, meta
    meta["value"] = value
    return value, meta


def _read_exact(handle, n: int) -> bytes:
    blob = handle.read(n)
    if blob is None or len(blob) != n:
        raise EOFError("truncated GGUF")
    return blob


def _gguf_skip_value(handle, value_type: int) -> None:
    scalar = _GGUF_SCALAR.get(value_type)
    if scalar is not None:
        _read_exact(handle, scalar[0])
        return
    if value_type == _GGUF_STRING:
        length = struct.unpack("<Q", _read_exact(handle, 8))[0]
        if length > 50_000_000:
            raise ValueError("GGUF string too large")
        handle.seek(length, os.SEEK_CUR)
        return
    if value_type == _GGUF_ARRAY:
        elem_type = struct.unpack("<I", _read_exact(handle, 4))[0]
        count = struct.unpack("<Q", _read_exact(handle, 8))[0]
        if count > 10_000_000:
            raise ValueError("GGUF array too large")
        elem_scalar = _GGUF_SCALAR.get(elem_type)
        if elem_scalar is not None:
            handle.seek(elem_scalar[0] * count, os.SEEK_CUR)
            return
        for _ in range(count):
            _gguf_skip_value(handle, elem_type)
        return
    raise ValueError(f"unknown GGUF type {value_type}")


def _gguf_read_scalar(handle, value_type: int) -> int:
    spec = _GGUF_SCALAR.get(value_type)
    if spec is None:
        raise ValueError("context_length is not a scalar")
    size, fmt = spec
    raw = struct.unpack("<" + fmt, _read_exact(handle, size))[0]
    if isinstance(raw, bool):
        raise ValueError("context_length is bool")
    return int(raw)


def gguf_context_length(model_path: str) -> Optional[int]:
    """Return GGUF `*.context_length`, or None on any parse problem."""
    try:
        path = os.path.realpath(os.path.expanduser(str(model_path)))
        with open(path, "rb") as handle:
            magic = _read_exact(handle, 4)
            if magic != _GGUF_MAGIC:
                return None
            version = struct.unpack("<I", _read_exact(handle, 4))[0]
            if version < 2:
                return None
            _n_tensors, n_kv = struct.unpack("<QQ", _read_exact(handle, 16))
            if n_kv > 1_000_000:
                return None
            found: Optional[int] = None
            for _ in range(n_kv):
                key_len = struct.unpack("<Q", _read_exact(handle, 8))[0]
                if key_len > 4096:
                    return None
                key = _read_exact(handle, key_len).decode("utf-8", errors="strict")
                value_type = struct.unpack("<I", _read_exact(handle, 4))[0]
                if key.endswith(".context_length") and value_type in _GGUF_SCALAR:
                    found = _gguf_read_scalar(handle, value_type)
                else:
                    _gguf_skip_value(handle, value_type)
            return found if found and found > 0 else None
    except Exception:
        return None


def probe_server_context(
    port: int, timeout: float = 3.0
) -> Optional[Dict[str, Any]]:
    """GET /props. Never raises. Returns per-slot n_ctx / slots / model_path."""
    try:
        url = f"http://127.0.0.1:{int(port)}/props"
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError, json.JSONDecodeError, TypeError):
        return None
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    dgs = payload.get("default_generation_settings")
    n_ctx = None
    if isinstance(dgs, dict):
        n_ctx = dgs.get("n_ctx")
        params = dgs.get("params")
        if n_ctx is None and isinstance(params, dict):
            n_ctx = params.get("n_ctx")
    try:
        per_slot = int(n_ctx) if n_ctx is not None else None
    except (TypeError, ValueError):
        per_slot = None
    if per_slot is not None and per_slot <= 0:
        per_slot = None
    try:
        total_slots = (
            int(payload["total_slots"])
            if payload.get("total_slots") is not None
            else None
        )
    except (TypeError, ValueError):
        total_slots = None
    model_path = payload.get("model_path")
    if not isinstance(model_path, str):
        model_path = None
    if per_slot is None and total_slots is None and model_path is None:
        return None
    return {
        "per_slot_n_ctx": per_slot,
        "total_slots": total_slots,
        "model_path": model_path,
    }


def _discover_ceiling(
    model_path: Optional[str], port: Optional[int]
) -> tuple[Optional[int], Optional[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    gguf = None
    if model_path:
        gguf = gguf_context_length(model_path)
        meta["gguf_context_length"] = gguf
        meta["model_path"] = model_path
    props = None
    if port is not None:
        props = probe_server_context(int(port))
        meta["props"] = props
        meta["port"] = int(port)
    props_ceiling = None
    if props and props.get("per_slot_n_ctx"):
        props_ceiling = int(props["per_slot_n_ctx"])
    if gguf:
        return gguf, "discovered:gguf", meta
    if props_ceiling:
        return props_ceiling, "discovered:props", meta
    reasons = []
    if not model_path:
        reasons.append("no model_path")
    elif gguf is None:
        reasons.append("GGUF has no usable context_length")
    if port is None:
        reasons.append("no port to probe")
    elif not props_ceiling:
        reasons.append("/props did not report per_slot_n_ctx")
    meta["reason"] = "; ".join(reasons) if reasons else "no discovered capability"
    return None, None, meta


def _safe_policy_total(
    model_ceiling: int,
    n_parallel: int,
    demand_tokens: Optional[int],
    generation_reserve: int,
    framing_reserve: int,
) -> tuple[int, int]:
    """Return (total_ctx, per_slot_target).

    Without demand_tokens this is today's spawn: --ctx-size is the
    conservative per-slot default (capped by the model), correctly divided
    later. With demand_tokens a large root ingress raises the allocation
    so the goal is not truncated: total_ctx = per_slot_target * n_parallel.
    """
    ceiling = max(1, int(model_ceiling))
    if demand_tokens is None:
        per_slot_target = min(ceiling, DEFAULT_PER_SLOT_CTX)
        return per_slot_target, per_slot_target
    needed = int(demand_tokens) + int(generation_reserve) + int(framing_reserve)
    rounded = _round_up(needed, LLAMA_KV_PAD)
    per_slot_target = min(ceiling, max(DEFAULT_PER_SLOT_CTX, rounded))
    total_ctx = per_slot_target * max(1, int(n_parallel))
    return total_ctx, per_slot_target


def _lost(reason: str, extra: Optional[Dict[str, Any]] = None, **more: Any) -> Dict[str, Any]:
    payload = dict(extra or {})
    payload.update(more)
    payload.pop("reason", None)
    entry: Dict[str, Any] = {"won": False, "reason": reason}
    entry.update(payload)
    return entry


def resolve(
    *,
    model_path: Optional[str] = None,
    n_parallel: int = 1,
    repo_root: Optional[str] = None,
    demand_tokens: Optional[int] = None,
    ctx_size: Optional[int] = None,
    port: Optional[int] = None,
    generation_reserve: Optional[int] = None,
) -> ContextBudget:
    del repo_root  # reserved for callers; profile lives on the machine genome
    n_parallel = max(1, int(n_parallel))
    framing_reserve = DEFAULT_FRAMING_RESERVE
    generation_reserve = _generation_reserve(generation_reserve)

    override_val: Optional[int] = None
    override_source: Optional[str] = None
    if ctx_size is not None:
        try:
            parsed = int(ctx_size)
        except (TypeError, ValueError):
            parsed = None
        if parsed is not None and parsed > 0:
            override_val = parsed
            override_source = "override:arg"
    if override_val is None:
        raw = os.environ.get("HCLI_CTX_SIZE")
        if raw is not None and str(raw).strip() != "":
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                parsed = None
            if parsed is not None and parsed > 0:
                override_val = parsed
                override_source = "override:HCLI_CTX_SIZE"

    profile_val, profile_meta = _read_profile_ctx()
    ceiling, discovered_source, discovered_meta = _discover_ceiling(
        model_path, port
    )

    provenance: Dict[str, Any] = {}
    source: str
    total_ctx: int
    model_ceiling = ceiling

    if override_val is not None and override_source is not None:
        total_ctx = override_val
        source = override_source
        provenance["override"] = {
            "won": True,
            "value": override_val,
            "via": override_source,
        }
        provenance["profile"] = _lost(
            "lost to explicit override", profile_meta
        )
        provenance["discovered"] = _lost(
            "lost to explicit override", discovered_meta
        )
        provenance["fallback"] = _lost(
            "lost to explicit override",
            value=DEFAULT_PER_SLOT_CTX,
        )
    elif profile_val is not None:
        total_ctx = profile_val
        source = "profile:context_ctx_size"
        provenance["override"] = _lost(
            "no ctx_size argument and HCLI_CTX_SIZE unset"
        )
        provenance["profile"] = {
            "won": True,
            "value": profile_val,
            **profile_meta,
        }
        provenance["discovered"] = _lost(
            "lost to qualified profile", discovered_meta
        )
        provenance["fallback"] = _lost(
            "lost to qualified profile",
            value=DEFAULT_PER_SLOT_CTX,
        )
    elif ceiling is not None and discovered_source is not None:
        total_ctx, per_slot_target = _safe_policy_total(
            ceiling,
            n_parallel,
            demand_tokens,
            generation_reserve,
            framing_reserve,
        )
        source = discovered_source
        provenance["override"] = _lost(
            "no ctx_size argument and HCLI_CTX_SIZE unset"
        )
        provenance["profile"] = _lost(
            profile_meta.get("reason") or "no context_ctx_size",
            profile_meta,
        )
        provenance["discovered"] = {
            "won": True,
            "via": discovered_source,
            "model_ceiling": ceiling,
            "per_slot_target": per_slot_target,
            "demand_tokens": demand_tokens,
            **discovered_meta,
        }
        provenance["fallback"] = _lost(
            "lost to discovered capability",
            value=DEFAULT_PER_SLOT_CTX,
        )
    else:
        total_ctx = DEFAULT_PER_SLOT_CTX
        source = "fallback:DEFAULT_PER_SLOT_CTX"
        provenance["override"] = _lost(
            "no ctx_size argument and HCLI_CTX_SIZE unset"
        )
        provenance["profile"] = _lost(
            profile_meta.get("reason") or "no context_ctx_size",
            profile_meta,
        )
        provenance["discovered"] = _lost(
            discovered_meta.get("reason")
            or "no GGUF context_length and no live /props",
            discovered_meta,
        )
        provenance["fallback"] = {
            "won": True,
            "value": DEFAULT_PER_SLOT_CTX,
        }

    per_request_ctx = per_seq_context(total_ctx, n_parallel)
    usable = per_request_ctx - int(generation_reserve) - int(framing_reserve)
    if usable < 0:
        usable = 0
    return ContextBudget(
        total_ctx=int(total_ctx),
        n_parallel=n_parallel,
        per_request_ctx=int(per_request_ctx),
        generation_reserve=int(generation_reserve),
        framing_reserve=int(framing_reserve),
        usable_input_tokens=int(usable),
        source=source,
        model_ceiling=int(model_ceiling) if model_ceiling else None,
        provenance=provenance,
    )


def apply_observed_slot(
    budget: ContextBudget,
    observed_per_slot: int,
    *,
    props: Optional[Dict[str, Any]] = None,
) -> ContextBudget:
    """Reconcile a planned budget with the live server's /props n_ctx.

    The observed per-slot value wins when it differs. Spawn is not failed.
    """
    observed = int(observed_per_slot)
    planned = int(budget.per_request_ctx)
    provenance = dict(budget.provenance)
    provenance["reconcile"] = {
        "planned_per_request_ctx": planned,
        "observed_per_slot_n_ctx": observed,
        "diverged": observed != planned,
        "winner": "observed" if observed != planned else "planned",
        "props": props,
    }
    usable = observed - int(budget.generation_reserve) - int(budget.framing_reserve)
    if usable < 0:
        usable = 0
    return ContextBudget(
        total_ctx=budget.total_ctx,
        n_parallel=budget.n_parallel,
        per_request_ctx=observed,
        generation_reserve=budget.generation_reserve,
        framing_reserve=budget.framing_reserve,
        usable_input_tokens=int(usable),
        source=budget.source,
        model_ceiling=budget.model_ceiling,
        provenance=provenance,
    )


def preflight(
    budget: ContextBudget,
    prompt_tokens: int,
    *,
    kind: str = "root",
) -> PreflightResult:
    prompt_tokens = max(0, int(prompt_tokens))
    demand = (
        prompt_tokens
        + int(budget.generation_reserve)
        + int(budget.framing_reserve)
    )
    per_request_ctx = int(budget.per_request_ctx)
    usable = int(budget.usable_input_tokens)
    shortfall = max(0, demand - per_request_ctx)
    ok = demand <= per_request_ctx
    if ok:
        remedy = (
            f"request fits: demand {demand} <= per_request_ctx {per_request_ctx} "
            f"(n_parallel={budget.n_parallel}, total_ctx={budget.total_ctx}, "
            f"source={budget.source})"
        )
    else:
        n_one = per_seq_context(budget.total_ctx, 1)
        remedy = (
            f"demand {demand} exceeds per-request context {per_request_ctx} "
            f"by {shortfall} tokens (kind={kind}). "
            f"Levers: reduce n_parallel (currently {budget.n_parallel}; "
            f"n_parallel=1 yields per_request_ctx {n_one}); "
            f"raise HCLI_CTX_SIZE/--ctx-size (currently {budget.total_ctx}); "
            f"or shrink inlined evidence so prompt_tokens fall below "
            f"usable_input_tokens {usable}."
        )
    return PreflightResult(
        ok=ok,
        demand=demand,
        usable=usable,
        per_request_ctx=per_request_ctx,
        shortfall=shortfall,
        kind=str(kind or "root"),
        remedy=remedy,
        budget=budget,
    )


def estimate_tokens(
    text: str, *, chars_per_token: int = CHARS_PER_TOKEN
) -> int:
    """Character-proxy token count. Empty is 0 so evidence can add cleanly."""
    n = len(str(text or ""))
    if n <= 0:
        return 0
    cpt = max(1, int(chars_per_token))
    return max(1, n // cpt)


def preflight_packet(
    budget: ContextBudget,
    packet_prompt: str,
    *,
    evidence_text: str = "",
    evidence_tokens: int = 0,
    kind: str = "worker",
) -> PreflightResult:
    """Preflight a compiled worker packet plus inlined evidence.

    The caller-visible signal is PreflightResult.ok / shortfall / remedy.
    Does not truncate. Use fit_or_refuse to turn a miss into an error.
    """
    tokens = estimate_tokens(packet_prompt)
    if evidence_text:
        tokens += estimate_tokens(evidence_text)
    tokens += max(0, int(evidence_tokens))
    return preflight(budget, tokens, kind=str(kind or "worker"))


def fit_or_refuse(
    budget: ContextBudget,
    packet_prompt: str,
    *,
    evidence_text: str = "",
    evidence_tokens: int = 0,
    kind: str = "worker",
    omitted: tuple = (),
) -> PreflightResult:
    """Refuse an over-budget packet. Never silently drop fields."""
    result = preflight_packet(
        budget,
        packet_prompt,
        evidence_text=evidence_text,
        evidence_tokens=evidence_tokens,
        kind=kind,
    )
    if not result.ok:
        raise PacketBudgetError(
            result=result,
            omitted=tuple(omitted),
            prompt_len=len(str(packet_prompt or "")),
            cap=int(result.per_request_ctx),
        )
    return result


__all__ = [
    "LLAMA_KV_PAD",
    "DEFAULT_PER_SLOT_CTX",
    "DEFAULT_FRAMING_RESERVE",
    "DEFAULT_GENERATION_RESERVE",
    "CHARS_PER_TOKEN",
    "ContextBudgetError",
    "PacketBudgetError",
    "ContextBudget",
    "PreflightResult",
    "per_seq_context",
    "solve_parallel",
    "gguf_context_length",
    "probe_server_context",
    "resolve",
    "apply_observed_slot",
    "preflight",
    "estimate_tokens",
    "preflight_packet",
    "fit_or_refuse",
]
