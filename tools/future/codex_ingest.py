"""CODEX_INGEST — read-only watcher that turns Codex receipts into sidecar deltas.

`receipts/headless/` is Codex's live artifact stream. This module never writes
there. It hashes every file, classifies genuinely new or changed artifacts as
LAW / SCAR / NEUTRAL from their content, and emits the downstream deltas that
Odyssey II, Odyssey III, the Architecture Atlas, PhysicalGraph, the Learned
Physical Compiler and HWIR consume. A second consecutive run must produce no
new deltas; `--once --assert-idempotent` exits non-zero if it found anything.

    python3 tools/future/codex_ingest.py --once
    python3 tools/future/codex_ingest.py --once --assert-idempotent
    python3 tools/future/codex_ingest.py --selftest
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO, git, RECEIPTS


import argparse
import hashlib
import json
import os
import re
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RECEIPT = "CODEX_INGEST_STATE.json"
SCHEMA = "hawking.future.codex_ingest.v1"
HEADLESS = REPO / "receipts" / "headless"
RECORDED_BY = "tools/future/codex_ingest.py"

LABELS = ("LAW", "SCAR", "NEUTRAL")

# Odyssey II scope. There is no fourth Odyssey and no Era VI.
SCOPE_MODEL_LOCAL = "MODEL_LOCAL"
SCOPE_FAMILY = "FAMILY"
SCOPE_GENERIC_VERIFIED = "GENERIC_VERIFIED"

# Distinctive tokens. Substring match is reserved for these; we do not match
# the English word "negative" in prose (too many false positives).
SCAR_TOKENS = (
    "PROTECTED_REJECT",
    "DIAGNOSTIC_REJECT",
    "NOT_FOR_PROMOTION",
)
LAW_TOKENS = (
    "PROTECTED_PASS",
    "DIAGNOSTIC_PASS",
)
REJECT_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9]*_REJECT\b")

SCAR_STATUS = frozenset(
    {
        "BLOCKED",
        "REJECTED",
        "FAILED",
        "FAIL",
        "REFUTED",
        "ERROR",
        "REFUSED",
        "PROTECTED_REJECT",
        "DIAGNOSTIC_REJECT",
        "NOT_FOR_PROMOTION",
        "NEGATIVE",
    }
)
LAW_STATUS = frozenset(
    {
        "VERIFIED",
        "PASSED",
        "PASS",
        "PROTECTED_PASS",
        "DIAGNOSTIC_PASS",
    }
)
SCAR_VERDICTS = frozenset(
    {
        "REFUTED",
        "REJECT",
        "REJECTED",
        "FAIL",
        "FAILED",
        "NO-GO",
        "NOGO",
        "NOT_WORTH_BUILDING",
        "PROTECTED_REJECT",
        "DIAGNOSTIC_REJECT",
    }
)
LAW_VERDICTS = frozenset(
    {
        "PASS",
        "PASSED",
        "VERIFIED",
        "CONFIRMED",
        "HOLD",
        "PROTECTED_PASS",
        "DIAGNOSTIC_PASS",
    }
)

# Catalog / plan receipts without a top-level verdict stay NEUTRAL.
NEUTRAL_SCHEMA_MARKERS = (
    "census",
    "scoreboard",
    "queue",
    "ledger",
    "atlas",
    "nomenclature",
    "handoff",
    "akb",
    "index",
    "schema_requirement",
    "membership",
)
CATALOG_TOP_KEYS = frozenset(
    {"entries", "corpus_size", "candidates", "queue", "columns", "frontier_points"}
)

SPATIAL_HINTS = (
    "attention",
    "gemm",
    "matmul",
    "mlp",
    "convolution",
    "softmax",
    "reduction",
    "scan",
    "barrier",
    "simdgroup",
    "occupancy",
    "hbm",
    "bandwidth",
    "dispatch",
    "fusion",
    "moe",
    "router",
    "kv_cache",
    "rmsnorm",
    "normalization",
    "organ",
)

# Paths the contract asked us to recover. A missing path is a finding, not a crash.
RECOVERY_PROBES: tuple[tuple[str, str], ...] = (
    ("tools/future/_common.py", "write_receipt / sha256_file / bench_block — used, not reimplemented"),
    ("tools/future/mutation_surface.py", "ownership map; this module writes only inside the sidecar glob"),
    ("tools/future/global_frontier.py", "F015 is this lane: Codex receipts currently die where they land"),
    ("tools/headless/negative_science.py", "scar representation: nine fields, three levels; not a watcher"),
    ("tools/headless/lane_watch.py", "closest existing watcher; watches grok worktrees, not receipt content"),
    ("tools/odyssey/ingest.py", "training-corpus ingest; content-addressed but a different pipeline"),
    ("tools/odyssey/modellake_watch.py", "named in the contract; absent from HEAD"),
    ("hcli/agentos/protected_benchmark_watcher.py", "GPU-window watcher; takes locks/signals — we must not copy that"),
    ("hcli/physical_graph.py", "PhysicalGraph plan schema; we emit candidate semantics only"),
    ("tools/odyssey/physical_graph_compiler.py", "existing compiler; sidecar does not invoke it"),
    ("receipts/headless/ACCELERATOR_SCOREBOARD.json", "named in the contract; absent from HEAD"),
    ("receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json", "named in the contract; absent from HEAD"),
    ("receipts/headless/NOETIC_SCOREBOARD.json", "present representation scoreboard; not an ingest cursor"),
    ("receipts/headless/ACCELERATOR_LAW_BASE.json", "AKB law store — shape reference for LAW entries"),
    ("receipts/headless/NOETIC_NEGATIVE_SCIENCE.json", "scar corpus — shape reference for SCAR entries"),
    ("receipts/headless/PHYSICAL_GRAPH_COMPILER.json", "organ-graph evidence the HWIR/PhysicalGraph deltas cite"),
    ("crates/hawking-research/src/ingest.rs", "arxiv/web ingest; unrelated to Codex receipts"),
)


@dataclass(frozen=True)
class Classification:
    label: str
    confidence: float
    field: str
    token: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "field": self.field,
            "token": self.token,
            "reason": self.reason,
        }


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_readonly(path: Path) -> bytes:
    """Open a Codex artifact with O_RDONLY. Never create, never truncate."""
    fd = os.open(path, os.O_RDONLY)
    with os.fdopen(fd, "rb") as fh:
        return fh.read()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _upper_token(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return None
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s or len(s) > 80:
        return None
    return s.upper().replace(" ", "_")


def _truthy_flag(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    if isinstance(value, (int, float)) and value == 1:
        return True
    if isinstance(value, str) and value.strip().upper() in {"TRUE", "YES", "1"}:
        return True
    return False


def _nonempty(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (str, bytes)) and not value:
        return False
    if isinstance(value, (list, dict)) and not value:
        return False
    return True


def _dotted(obj: Any, path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _schema_looks_neutral(schema: Any) -> bool:
    if not isinstance(schema, str):
        return False
    s = schema.lower()
    return any(m in s for m in NEUTRAL_SCHEMA_MARKERS)


def _looks_like_catalog(obj: dict[str, Any]) -> bool:
    if _schema_looks_neutral(obj.get("schema")):
        return True
    if CATALOG_TOP_KEYS & obj.keys() and not any(
        k in obj for k in ("status", "pass", "verdict", "NOT_FOR_PROMOTION", "result")
    ):
        return True
    return False


def _looks_like_census(obj: dict[str, Any]) -> bool:
    """A census/index with pass:true is still a catalog, not a law."""
    if any("census" in str(k).lower() or "index" == str(k).lower() for k in obj):
        return True
    for key in ("experiment_class", "obligation", "headline", "schema"):
        val = obj.get(key)
        if isinstance(val, str) and ("census" in val.lower() or val.upper().endswith("_INDEX")):
            return True
    return False


def _token_in_text(text: str, tokens: tuple[str, ...]) -> str | None:
    for tok in tokens:
        if tok in text:
            return tok
    return None


def _scar_reject_in_text(text: str) -> str | None:
    m = REJECT_TOKEN_RE.search(text)
    return m.group(0) if m else None


def _first_token_path(
    node: Any, tokens: tuple[str, ...], *, also_reject_tokens: bool, path: str = ""
) -> tuple[str, str] | None:
    """Return (dotted-path, token) for the first LAW/SCAR token, sorted-key walk."""
    if isinstance(node, dict):
        for key in sorted(node):
            child = f"{path}.{key}" if path else str(key)
            hit = _first_token_path(
                node[key], tokens, also_reject_tokens=also_reject_tokens, path=child
            )
            if hit:
                return hit
        return None
    if isinstance(node, list):
        for i, item in enumerate(node):
            hit = _first_token_path(
                item, tokens, also_reject_tokens=also_reject_tokens, path=f"{path}[{i}]"
            )
            if hit:
                return hit
        return None
    if isinstance(node, str):
        tok = _token_in_text(node, tokens)
        if tok is None and also_reject_tokens:
            tok = _scar_reject_in_text(node)
        if tok:
            return (path or "<document>", tok)
    return None


def _classify_text(text: str) -> Classification:
    scar = _token_in_text(text, SCAR_TOKENS) or _scar_reject_in_text(text)
    if scar:
        return Classification(
            "SCAR", 0.9, "<text>", scar, f"text contains {scar}"
        )
    law = _token_in_text(text, LAW_TOKENS)
    if law:
        return Classification(
            "LAW", 0.9, "<text>", law, f"text contains {law}"
        )
    return Classification(
        "NEUTRAL", 0.7, "<text>", "no_verdict", "non-JSON artifact with no LAW/SCAR token"
    )


def _failed_gate(value: Any, field: str) -> Classification | None:
    tok = _upper_token(value)
    if tok in SCAR_STATUS or tok in SCAR_VERDICTS:
        return Classification(
            "SCAR", 0.9, field, tok or str(value), f"{field} is a failed/rejected gate"
        )
    if isinstance(value, dict):
        for k in ("pass", "ok", "accepted"):
            if k in value and value[k] is False:
                return Classification(
                    "SCAR", 0.85, f"{field}.{k}", "false", f"{field} reports failure"
                )
        nested = value.get("status") or value.get("verdict") or value.get("decision")
        tok = _upper_token(nested)
        if tok in SCAR_STATUS or tok in SCAR_VERDICTS:
            return Classification(
                "SCAR", 0.9, f"{field}.status", tok or str(nested), f"{field} rejected"
            )
    return None


def classify_obj(obj: dict[str, Any]) -> Classification:
    """Pure classification of a parsed receipt. Reproducible: no path, no mtime."""
    # --- SCAR, highest priority ---
    if "NOT_FOR_PROMOTION" in obj and _truthy_flag(obj.get("NOT_FOR_PROMOTION")):
        return Classification(
            "SCAR",
            1.0,
            "NOT_FOR_PROMOTION",
            "true",
            "explicit NOT_FOR_PROMOTION flag; this artifact must not become a law",
        )

    status_tok = _upper_token(obj.get("status"))
    if status_tok in SCAR_STATUS:
        return Classification(
            "SCAR", 1.0, "status", status_tok, f"status {status_tok} is a negative result"
        )

    gate_hit = _failed_gate(obj.get("gate"), "gate") if "gate" in obj else None
    if gate_hit:
        return gate_hit

    if obj.get("pass") is False:
        return Classification(
            "SCAR", 0.95, "pass", "false", "top-level pass is false (failed gate)"
        )

    if _truthy_flag(obj.get("negative_result")):
        return Classification(
            "SCAR", 0.95, "negative_result", "true", "explicit negative_result"
        )

    if "negative_science" in obj and _nonempty(obj.get("negative_science")):
        return Classification(
            "SCAR",
            0.95,
            "negative_science",
            "negative_science",
            "explicit negative_science block",
        )

    verdict_tok = _upper_token(obj.get("verdict"))
    if verdict_tok in SCAR_VERDICTS:
        return Classification(
            "SCAR", 0.95, "verdict", verdict_tok, f"verdict {verdict_tok}"
        )

    result = obj.get("result")
    if isinstance(result, dict):
        rv = _upper_token(result.get("verdict"))
        if rv in SCAR_VERDICTS:
            return Classification(
                "SCAR", 0.95, "result.verdict", rv, f"result.verdict {rv}"
            )

    ev = _upper_token(obj.get("experiment_verdict"))
    if ev in SCAR_VERDICTS:
        return Classification(
            "SCAR", 0.9, "experiment_verdict", ev, f"experiment_verdict {ev}"
        )

    for field in ("claim_boundary", "reason", "headline", "qualification"):
        val = obj.get(field)
        if isinstance(val, str):
            tok = _token_in_text(val, SCAR_TOKENS) or _scar_reject_in_text(val)
            if tok:
                return Classification(
                    "SCAR", 0.85, field, tok, f"{field} contains {tok}"
                )

    # Nested NOT_FOR_PROMOTION / *_REJECT on well-known child dicts only
    # (do not walk catalog `entries` — a law store listing a REFUTED law is
    # still a catalog, not itself a scar).
    for child_key in ("arms", "checks", "qualification", "promotion_gate"):
        child = obj.get(child_key)
        if isinstance(child, dict) and _truthy_flag(child.get("NOT_FOR_PROMOTION")):
            return Classification(
                "SCAR",
                1.0,
                f"{child_key}.NOT_FOR_PROMOTION",
                "true",
                "nested NOT_FOR_PROMOTION",
            )
        if isinstance(child, list):
            for i, item in enumerate(child):
                if isinstance(item, dict) and _truthy_flag(item.get("NOT_FOR_PROMOTION")):
                    return Classification(
                        "SCAR",
                        1.0,
                        f"{child_key}[{i}].NOT_FOR_PROMOTION",
                        "true",
                        "nested NOT_FOR_PROMOTION",
                    )

    # --- LAW ---
    if status_tok in LAW_STATUS:
        return Classification(
            "LAW",
            0.9,
            "status",
            status_tok,
            f"status {status_tok} is an evidence-backed positive result",
        )

    if isinstance(result, dict):
        rv = _upper_token(result.get("verdict"))
        if rv in LAW_VERDICTS:
            return Classification(
                "LAW", 0.85, "result.verdict", rv, f"result.verdict {rv}"
            )

    if verdict_tok in LAW_VERDICTS:
        return Classification(
            "LAW", 0.85, "verdict", verdict_tok, f"verdict {verdict_tok}"
        )

    for field in ("claim_boundary", "reason", "headline", "status"):
        val = obj.get(field)
        if isinstance(val, str):
            tok = _token_in_text(val, LAW_TOKENS)
            if tok:
                return Classification(
                    "LAW", 0.95, field, tok, f"{field} contains {tok}"
                )

    if obj.get("pass") is True and not _looks_like_catalog(obj) and not _looks_like_census(obj):
        # pass:true on a census/index is a "the census ran" bit, not a law.
        has_body = any(k in obj for k in ("result", "finding", "measurements", "answer"))
        conf = 0.7 if has_body else 0.55
        return Classification(
            "LAW",
            conf,
            "pass",
            "true",
            "top-level pass is true"
            + (" with a result/finding body" if has_body else ""),
        )

    # Content-wide token search so a receipt whose body *says* PROTECTED_REJECT
    # (or PROTECTED_PASS) is classified even if the token is not in a well-known
    # field. Sorted-key walk keeps this reproducible.
    scar_hit = _first_token_path(obj, SCAR_TOKENS, also_reject_tokens=True)
    if scar_hit:
        field, tok = scar_hit
        return Classification(
            "SCAR", 0.85, field, tok, f"{field} contains {tok}"
        )
    law_hit = _first_token_path(obj, LAW_TOKENS, also_reject_tokens=False)
    if law_hit:
        field, tok = law_hit
        return Classification(
            "LAW", 0.85, field, tok, f"{field} contains {tok}"
        )

    # --- NEUTRAL ---
    if _looks_like_catalog(obj):
        schema = obj.get("schema")
        return Classification(
            "NEUTRAL",
            0.8,
            "schema" if isinstance(schema, str) else "<catalog>",
            str(schema) if isinstance(schema, str) else "catalog",
            "metadata / census / index / schema / plan with no verdict",
        )

    status = obj.get("status")
    if isinstance(status, str) and any(
        p in status.upper() for p in ("PLANNED", "SCAFFOLD", "SEALED", "READY")
    ):
        return Classification(
            "NEUTRAL",
            0.75,
            "status",
            status,
            "plan or scaffold status, no verdict",
        )

    return Classification(
        "NEUTRAL", 0.6, "<default>", "no_verdict", "no LAW or SCAR driver in well-known fields"
    )


def classify_bytes(raw: bytes) -> Classification:
    """Pure function of file bytes. Same bytes ⇒ same classification."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return Classification(
            "NEUTRAL", 1.0, "<bytes>", "undecodable", "binary or non-UTF-8 artifact"
        )
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return _classify_text(text)
    if isinstance(obj, dict):
        return classify_obj(obj)
    if isinstance(obj, list):
        return Classification(
            "NEUTRAL", 0.7, "<list>", "json_array", "JSON array with no object-level verdict"
        )
    return _classify_text(text)


def classify_path(path: Path) -> Classification:
    """Read-only classify. Provided for callers; class is still a function of bytes."""
    return classify_bytes(_read_readonly(path))


def _clip(text: str, n: int = 240) -> str:
    text = " ".join(text.split())
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _string_at(obj: dict[str, Any], *paths: str) -> str | None:
    for path in paths:
        v = _dotted(obj, path) if "." in path else obj.get(path)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _guess_organ(obj: dict[str, Any]) -> str:
    v = _string_at(
        obj,
        "organ",
        "identities.organ",
        "applicability.ORGAN",
        "experiment_class",
        "architecture",
    )
    if v:
        return _clip(v, 80)
    nodes = _dotted(obj, "organ_graph.nodes")
    if isinstance(nodes, list):
        organs = [
            n.get("organ")
            for n in nodes
            if isinstance(n, dict) and isinstance(n.get("organ"), str)
        ]
        if organs:
            return ",".join(organs[:8])
    return "UNKNOWN"


def _guess_model(obj: dict[str, Any]) -> str:
    v = _string_at(
        obj,
        "model",
        "model_id",
        "identities.model.id",
        "identities.model.name",
        "applicability.MODEL",
        "base_identity.model_id",
    )
    model = obj.get("identities", {}).get("model") if isinstance(obj.get("identities"), dict) else None
    if isinstance(model, dict) and model.get("status") == "ABSENT":
        return "NONE"
    return v or "UNKNOWN"


def _knowledge_level(obj: dict[str, Any]) -> str:
    v = obj.get("knowledge_level")
    return str(v) if isinstance(v, str) and v else "UNKNOWN"


def proposed_scope(obj: dict[str, Any]) -> tuple[str, str]:
    """Odyssey II scope proposal. Sidecar never promotes."""
    kl = _knowledge_level(obj).upper()
    if kl == "GENERAL":
        return (
            SCOPE_GENERIC_VERIFIED,
            "source knowledge_level is GENERAL; Odyssey II must still refuse unevidenced promotion",
        )
    if kl in {"MODEL_FAMILY", "ARCHITECTURE", "SOC_FAMILY", "REPRESENTATION", "FAMILY"}:
        return SCOPE_FAMILY, f"source knowledge_level is {kl}"
    return (
        SCOPE_MODEL_LOCAL,
        "a single Codex receipt is MODEL_LOCAL; the sidecar has no promotion authority",
    )


def _statement_sketch(obj: dict[str, Any], cls: Classification) -> str:
    v = _string_at(
        obj,
        "headline",
        "result.hypothesis",
        "result.the_named_gap_is_closed",
        "finding.reason",
        "question",
        "one_line",
        "answer",
        "reason",
        "obligation",
    )
    if v:
        return _clip(v)
    return _clip(cls.reason)


def _spatially_meaningful(obj: dict[str, Any], organ: str) -> tuple[bool, str]:
    blob = " ".join(
        str(x).lower()
        for x in (
            organ,
            obj.get("experiment_class") or "",
            obj.get("schema") or "",
            obj.get("kernel") or "",
        )
    )
    for hint in SPATIAL_HINTS:
        if hint in blob:
            return True, f"organ/schema/experiment mentions {hint}"
    return False, "no spatial organ/primitive hint; HWIR projection is a stub"


def _contamination_tag(obj: dict[str, Any]) -> str:
    bc = obj.get("benchmark_class")
    if isinstance(bc, str) and bc:
        return bc
    if "contamination" in obj:
        return "RECORDED_NOT_INTERPRETED"
    pw = obj.get("protected_window")
    if pw is True:
        return "PROTECTED_WINDOW_DECLARED"
    if pw is False:
        return "NO_PROTECTED_WINDOW"
    return "UNKNOWN"


def _law_delta(relpath: str, sha: str, obj: dict[str, Any], cls: Classification) -> dict[str, Any]:
    organ = _guess_organ(obj)
    model = _guess_model(obj)
    scope, scope_why = proposed_scope(obj)
    spatial, spatial_why = _spatially_meaningful(obj, organ)
    sketch = _statement_sketch(obj, cls)
    kl = _knowledge_level(obj)
    return {
        "source": relpath,
        "source_sha256": sha,
        "classification": "LAW",
        "driver": cls.as_dict(),
        "odyssey_ii_law_candidate": {
            "proposed_scope": scope,
            "scope_reason": scope_why,
            "sidecar_promotion_authority": False,
            "statement_sketch": sketch,
            "model": model,
            "organ": organ,
            "knowledge_level": kl,
            "evidence_class": "STATIC_ONLY",
            "action": "admit as a candidate; do not promote past MODEL_LOCAL without independent evidence",
        },
        "odyssey_iii_attack_target": {
            "target": relpath,
            "attack": "refute, bound, or find the contamination in the claimed result",
            "suggested_angle": (
                "transfer to a second model"
                if scope == SCOPE_MODEL_LOCAL
                else "seek a counter-organ or counter-machine"
            ),
            "action": "register as an Odyssey III attack; do not treat the candidate as settled",
        },
        "architecture_atlas_behaviour_reference": {
            "behaviour": organ if organ != "UNKNOWN" else kl,
            "action": "cite as behaviour evidence; do not rewrite the atlas",
            "atlas_path": "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json",
        },
        "physical_graph_candidate_semantic": {
            "semantic_type": "PhysicalGraphPlan",
            "qualification": "PLAN_ONLY",
            "organ": organ,
            "action": "consider as a candidate semantic; sidecar does not compile a graph",
        },
        "learned_physical_compiler_row": {
            "source": relpath,
            "source_sha256": sha,
            "label": "LAW",
            "organ": organ,
            "model": model,
            "technique": _string_at(obj, "technique", "experiment_class") or "UNKNOWN",
            "representation": _string_at(obj, "representation", "identities.representation.name") or "UNKNOWN",
            "machine": "UNKNOWN",
            "contamination": _contamination_tag(obj),
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
            # Hardware fields stay null. Copying a measured number here would
            # raise HardwareClaimError, which is the point of that guard.
            "measured": {
                "tps": None,
                "token_ns": None,
                "gpu_ns": None,
                "joules_per_token": None,
                "bandwidth_gbps": None,
            },
            "action": "append as a dataset-row skeleton; numbers remain UNKNOWN until a protected measurement",
        },
        "hwir_projection": {
            "spatially_meaningful": spatial,
            "reason": spatial_why,
            "organ": organ,
            "backend": "UNKNOWN",
            "action": (
                "project onto HWIR once an IR exists"
                if spatial
                else "no spatial mapping suggested"
            ),
        },
    }


def _scar_delta(relpath: str, sha: str, obj: dict[str, Any], cls: Classification) -> dict[str, Any]:
    organ = _guess_organ(obj)
    sketch = _statement_sketch(obj, cls)
    reopen = _string_at(obj, "reopen_condition", "finding.reopen_condition") or "UNKNOWN"
    kills = [sketch]
    if cls.token in SCAR_TOKENS or cls.token.endswith("_REJECT"):
        kills.append(
            "any hypothesis that this artifact is promotion-grade PROTECTED_ABSOLUTE evidence"
        )
    if cls.field in {"pass", "gate"} or cls.token in {"BLOCKED", "FAILED", "REFUTED"}:
        kills.append(f"the hypothesis this receipt was testing ({cls.field}={cls.token})")
    redundant = [
        "retrying the same hypothesis on the same model/organ/machine without a new reopen condition",
        "promoting a DIAGNOSTIC_RELATIVE number derived from this artifact",
    ]
    if cls.token == "NOT_FOR_PROMOTION" or cls.field == "NOT_FOR_PROMOTION":
        redundant.append("writing this result into a scoreboard as a deciding measurement")
    return {
        "source": relpath,
        "source_sha256": sha,
        "classification": "SCAR",
        "driver": cls.as_dict(),
        "invalidation": {
            "kills": kills[:6],
            "makes_redundant": redundant,
            "reopen_condition": reopen,
            "level": "MODEL_SPECIFIC",
            "sidecar_must_not_promote": True,
            "organ": organ,
            "action": "feed the negative index; a single model's scar never globally prunes a technique",
        },
        "consumers_notified": [
            "odyssey_ii_law_store",
            "odyssey_iii",
            "negative_index",
            "learned_physical_compiler",
        ],
    }


def emit_delta(relpath: str, sha: str, raw: bytes, cls: Classification) -> dict[str, Any] | None:
    if cls.label == "NEUTRAL":
        return None
    obj: dict[str, Any] = {}
    try:
        parsed = json.loads(raw.decode("utf-8"))
        if isinstance(parsed, dict):
            obj = parsed
    except (UnicodeDecodeError, json.JSONDecodeError):
        obj = {}
    if cls.label == "LAW":
        return _law_delta(relpath, sha, obj, cls)
    if cls.label == "SCAR":
        return _scar_delta(relpath, sha, obj, cls)
    return None


def _relkey(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return resolved.relative_to(root.resolve()).as_posix()


def list_artifacts(root: Path) -> list[Path]:
    """Sorted, followlinks=False, skip junk. Read-only walk."""
    if not root.is_dir():
        return []
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in {".git", "__pycache__", ".pytest_cache"} and not d.startswith(".")
        )
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            p = Path(dirpath) / fn
            try:
                st = p.lstat()
            except OSError:
                continue
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                continue
            out.append(p)
    out.sort()
    return out


def _load_previous_cursor(previous: dict[str, Any] | None) -> dict[str, Any]:
    if not previous:
        return {}
    cur = previous.get("cursor")
    if isinstance(cur, dict):
        return dict(cur)
    # A bare relpath->record map is also accepted (tests).
    if previous and all(isinstance(v, dict) and "sha256" in v for v in previous.values()):
        return dict(previous)
    return {}


def load_state(path: Path | None = None) -> dict[str, Any]:
    p = path or (RECEIPTS / RECEIPT)
    if not p.is_file():
        return {}
    try:
        return load_json(p)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _probe_path(rel: str) -> dict[str, Any]:
    p = REPO / rel
    in_git = bool(git("ls-tree", "--name-only", "HEAD", rel).strip())
    return {
        "path": rel,
        "on_disk": p.exists(),
        "in_git": in_git,
    }


def recovered_implementation() -> dict[str, Any]:
    rows = []
    for path, note in RECOVERY_PROBES:
        row = _probe_path(path)
        row["note"] = note
        rows.append(row)
    present = [r["path"] for r in rows if r["on_disk"] or r["in_git"]]
    absent = [r["path"] for r in rows if not r["on_disk"] and not r["in_git"]]
    return {
        "already_existed": rows,
        "present": present,
        "absent_from_head_and_disk": absent,
        "adequate_existing_watcher": None,
        "why_not_redundant": (
            "lane_watch.py watches grok worktrees; odyssey ingest.py watches training "
            "corpora; protected_benchmark_watcher.py governs a GPU window. None of them "
            "hash receipts/headless and emit LAW/SCAR deltas. F015 still holds."
        ),
    }


def gaps_closed() -> list[str]:
    return [
        "durable sha256 cursor over receipts/headless keyed by relpath",
        "pure-content LAW/SCAR/NEUTRAL classifier with recorded driver field/token",
        "downstream deltas for Odyssey II/III, Architecture Atlas, PhysicalGraph, LPC, HWIR",
        "SCAR invalidation deltas (kills / makes_redundant / reopen_condition)",
        "--once --assert-idempotent refusal that is capable of firing",
        "read-only open; writes only receipts/future/CODEX_INGEST_STATE.json",
    ]


def negative_findings(recovered: dict[str, Any]) -> list[str]:
    findings = [
        f"absent: {p}" for p in recovered.get("absent_from_head_and_disk") or []
    ]
    findings.extend(
        [
            "ACCELERATOR_SCOREBOARD.json was named as the current scoreboard and is not in HEAD",
            "ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json (status_transitions/funnel/candidate_statuses) is not in HEAD",
            "tools/odyssey/modellake_watch.py was named as the style template and is not in HEAD; lane_watch.py was used instead",
            "ACCELERATOR_ARCHITECTURE_ATLAS.json is not materialized in this sparse checkout (and is absent from HEAD name-search)",
            "this sidecar produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE; every delta is STATIC_ONLY / bench UNKNOWN",
            "hardware numbers in source receipts are deliberately not copied; LPC measured.* is null",
        ]
    )
    return findings


def ingest(
    *,
    root: Path,
    previous: dict[str, Any] | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Scan `root` read-only and produce the receipt body (unsealed).

    Change detection is sha256 of file bytes, never mtime. Classification is a
    pure function of those bytes. `now` is recorded only on new/changed entries.
    """
    clock = now or _utc_now()
    prev_cursor = _load_previous_cursor(previous)
    artifacts = list_artifacts(root)

    cursor: dict[str, Any] = {}
    classified_this_scan: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    active_deltas: list[dict[str, Any]] = []
    unreadable: list[str] = []
    n_new = n_changed = n_unchanged = 0
    by_scan = {"LAW": 0, "SCAR": 0, "NEUTRAL": 0}

    seen_keys: set[str] = set()
    for path in artifacts:
        rel = _relkey(path, root)
        seen_keys.add(rel)
        try:
            st = path.lstat()
            raw = _read_readonly(path)
        except OSError:
            unreadable.append(rel)
            if rel in prev_cursor:
                cursor[rel] = prev_cursor[rel]
            continue
        sha = _sha256_bytes(raw)
        rec_prev = prev_cursor.get(rel) if isinstance(prev_cursor.get(rel), dict) else None
        prev_sha = rec_prev.get("sha256") if rec_prev else None
        is_new = prev_sha is None
        is_changed = (not is_new) and prev_sha != sha
        # Classification is a function of the bytes we just hashed, so we always
        # recompute it. Deltas fire only on new/changed hashes — never on mtime.
        cls = classify_bytes(raw)

        delta = emit_delta(rel, sha, raw, cls)
        if delta is not None:
            active_deltas.append(delta)

        if is_new or is_changed:
            first_seen = clock if is_new else rec_prev.get("first_seen", clock)
            last_classified = clock
            if is_new:
                n_new += 1
            else:
                n_changed += 1
            by_scan[cls.label] = by_scan.get(cls.label, 0) + 1
            classified_this_scan.append(
                {"relpath": rel, "event": "new" if is_new else "changed", **cls.as_dict()}
            )
            if delta is not None:
                deltas.append(delta)
        else:
            n_unchanged += 1
            first_seen = rec_prev.get("first_seen", clock)
            last_classified = rec_prev.get("last_classified", clock)

        cursor[rel] = {
            "sha256": sha,
            "size": int(st.st_size),
            "mtime": float(st.st_mtime),
            "first_seen": first_seen,
            "last_classified": last_classified,
            "classification": cls.label,
            "confidence": cls.confidence,
            "driver_field": cls.field,
            "driver_token": cls.token,
            "reason": cls.reason,
        }

    missing = sorted(k for k in prev_cursor if k not in seen_keys)
    for rel in missing:
        # Keep last known record so a reappearing file with the same hash is
        # unchanged rather than "new". Missing is not "new".
        rec = dict(prev_cursor[rel])
        rec["present"] = False
        cursor[rel] = rec

    by_cursor = {"LAW": 0, "SCAR": 0, "NEUTRAL": 0}
    for rec in cursor.values():
        lab = rec.get("classification")
        if lab in by_cursor:
            by_cursor[lab] += 1

    recovered = recovered_implementation()
    try:
        root_rel = root.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        root_rel = str(root)

    classified_this_scan.sort(key=lambda r: r["relpath"])
    deltas.sort(key=lambda d: d["source"])
    active_deltas.sort(key=lambda d: d["source"])
    cursor = {k: cursor[k] for k in sorted(cursor)}

    return {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Read-only ingest of receipts/headless. Detect new/changed artifacts by "
            "sha256, classify LAW/SCAR/NEUTRAL from content, emit downstream deltas. "
            "Produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE."
        ),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "root": root_rel,
        "scan": {
            "n_on_disk": len(artifacts),
            "n_new": n_new,
            "n_changed": n_changed,
            "n_unchanged": n_unchanged,
            "n_missing_from_disk": len(missing),
            "n_unreadable": len(unreadable),
            "n_deltas": len(deltas),
            "n_active_deltas": len(active_deltas),
            "by_class_this_scan": by_scan,
            "by_class_cursor": by_cursor,
            "missing_from_disk": missing,
            "unreadable": unreadable,
        },
        "cursor": cursor,
        "classified_this_scan": classified_this_scan,
        "deltas_this_scan": deltas,
        "active_deltas": active_deltas,
        "idempotence": {
            "change_detector": "sha256 of file bytes; mtime is recorded and never consulted",
            "second_run_must_yield": {"n_new": 0, "n_changed": 0, "n_deltas": 0},
            "assert_idempotent": "exit non-zero iff n_new + n_changed > 0",
        },
        "vocabulary": {
            "eras": ["I", "II", "III", "IV", "V"],
            "odysseys": ["I WHAT IS TRUE?", "II WHAT DID HAWKING ALREADY LEARN?", "III WHERE IS HAWKING WRONG?"],
            "no_era_vi": True,
            "no_odyssey_iv": True,
            "fpga_is": "part of Accelerator / Physical Compiler / Fusion, not its own civilization",
            "evidence_classes_we_do_not_emit": ["DIAGNOSTIC_RELATIVE", "PROTECTED_ABSOLUTE"],
            "evidence_class_we_emit": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
        },
        "recovered_implementation": recovered,
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(recovered),
        "claim_boundary": (
            "Static sidecar artifact. No hardware measurement. "
            "Does not produce DIAGNOSTIC_RELATIVE or PROTECTED_ABSOLUTE. "
            "Deltas are proposals for downstream modules; disk evidence stays in receipts/headless."
        ),
    }


def _delta_count(doc: dict[str, Any]) -> int:
    scan = doc.get("scan") or {}
    return int(scan.get("n_new") or 0) + int(scan.get("n_changed") or 0)


def build(
    root: Path | None = None,
    *,
    previous: dict[str, Any] | None = None,
    now: str | None = None,
) -> Path:
    """Scan and write the sealed CODEX_INGEST_STATE receipt."""
    scan_root = Path(root) if root is not None else HEADLESS
    if previous is None:
        previous = load_state()
    doc = ingest(root=scan_root, previous=previous, now=now)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="scan receipts/headless and emit deltas")
    ap.add_argument(
        "--assert-idempotent",
        action="store_true",
        help="exit non-zero if the scan found any new or changed artifact",
    )
    ap.add_argument("--selftest", action="store_true", help="alias for a full build()")
    ap.add_argument("--root", default=None, help="artifact directory (default: receipts/headless)")
    a = ap.parse_args(argv)
    root = Path(a.root) if a.root else None
    out = build(root=root)
    print(out)
    if a.assert_idempotent:
        doc = load_json(out)
        n = _delta_count(doc)
        if n:
            scan = doc.get("scan") or {}
            print(
                f"not idempotent: n_new={scan.get('n_new')} "
                f"n_changed={scan.get('n_changed')} n_deltas={scan.get('n_deltas')}",
                file=sys.stderr,
            )
            return 1
        print("idempotent: no new or changed artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
