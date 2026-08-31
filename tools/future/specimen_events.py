"""G077 — a completed download is a scientific event, not a log line.

The ModelLake watcher watches. This module is what happens next: the
transition DOWNLOADING -> COMPLETE_UNSEALED -> SEALED_SOURCE_SPECIMEN is a
durable, replayable event. On SEALED_SOURCE_SPECIMEN it wakes the Odyssey
scheduler and ADDS a bounded WorkGraph to the running mission. It never
restarts Odyssey. Early metadata is not a sealed specimen; a partial-weight
experiment cannot be recorded as specimen science.

    python3 tools/future/specimen_events.py --build
    python3 -m pytest tools/future/test_specimen_events.py -q

STATIC_ONLY. No GPU lease. Does not re-hash a sealed specimen. Does not
touch live downloads. Does not rewrite ODYSSEY_I_LAUNCH.json.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from hcli.persist import atomic_write_json
from tools.future import negative_index as ni
from tools.future import workgraph as wg
from tools.future._common import RECEIPTS, REPO, load_json, write_receipt


RECEIPT = "SPECIMEN_EVENTS.json"
SCHEMA = "hawking.future.specimen_events.v1"
EVENT_SCHEMA = "hawking.future.specimen_event.v1"
LOG_SCHEMA = "hawking.future.specimen_event_log.v1"
VERSION = 1
RECORDED_BY = "tools/future/specimen_events.py"

LAUNCH_RECEIPT = RECEIPTS / "ODYSSEY_I_LAUNCH.json"
VERIFY_RECEIPT = RECEIPTS / "SPECIMEN_VERIFICATION.json"
LAW_STORE_RECEIPT = RECEIPTS / "ODYSSEY2_LAW_STORE.json"
CURRICULUM_RECEIPT = RECEIPTS / "SPECIMEN_CURRICULUM.json"

DOWNLOADING = "DOWNLOADING"
COMPLETE_UNSEALED = "COMPLETE_UNSEALED"
SEALED_SOURCE_SPECIMEN = "SEALED_SOURCE_SPECIMEN"

SPECIMEN_STATES: tuple[str, ...] = (
    DOWNLOADING,
    COMPLETE_UNSEALED,
    SEALED_SOURCE_SPECIMEN,
)

LEGAL_TRANSITIONS: frozenset[tuple[str, str]] = frozenset(
    {
        (DOWNLOADING, COMPLETE_UNSEALED),
        (COMPLETE_UNSEALED, SEALED_SOURCE_SPECIMEN),
        (DOWNLOADING, SEALED_SOURCE_SPECIMEN),
    }
)

KIND_TRANSITION = "SPECIMEN_STATE_TRANSITION"
KIND_SCHEDULER_WAKE = "SCHEDULER_WAKE"
KIND_WORKGRAPH_ADDED = "WORKGRAPH_ADDED"
KIND_EARLY_METADATA = "EARLY_METADATA"

CAMPAIGN_TRANSFER = "transfer_and_scars"
CAMPAIGN_NEW_SEARCH = "new_search"

# Campaign scars the next proposer must try before inventing a family.
# negative_index now keys these; a miss is a regression, not a skip.
WAVE_DEAD_FAMILIES: tuple[str, ...] = (
    "MLP_FUNCTION_REPLACEMENT",
    "MONARCH",
    "BUTTERFLY",
    "FACTORIZE_THE_FACTORS",
    "PRODUCT_DICTIONARY",
    "CONDITIONAL_PROGRAM",
    "GENERATED_BLOCK",
    "NONLINEAR_GENERATOR",
)

# Experiments that need tensor values. Config/index/filenames do not.
WEIGHT_REQUIRING_STAGES: frozenset[str] = frozenset(
    {
        "gravity",
        "doctor",
        "representation_search",
        "physical_profiling",
        "nr",
        "nx",
        "native_execution",
        "capability",
        "mlp_function_replacement",
        "monarch",
        "butterfly",
        "weight_read",
        "organ_fit",
    }
)

ORGAN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("mlp", re.compile(r"(mlp|feed_forward|ffn|gate_proj|up_proj|down_proj)", re.I)),
    ("attention", re.compile(r"(self_attn|attention|q_proj|k_proj|v_proj|o_proj)", re.I)),
    ("mamba_ssm", re.compile(r"(mamba|ssm|conv1d|A_log|dt_bias|in_proj|out_proj)", re.I)),
    ("moe_expert", re.compile(r"experts?\.\d+|moe", re.I)),
    ("moe_router", re.compile(r"(router|gate\.weight)", re.I)),
    ("embed", re.compile(r"(embed_tokens|word_embeddings|wte)", re.I)),
    ("lm_head", re.compile(r"lm_head", re.I)),
    ("vision", re.compile(r"(visual|vision_tower|vision_model|patch_embed)", re.I)),
    ("norm", re.compile(r"(layernorm|rms_norm|final_layernorm)", re.I)),
)

# Huge specimens still get cite + fingerprint + transfer + scars. New search
# is what they do not earn. Four units is not a 13-stage explosion.
MAX_UNITS_HUGE = 4
MAX_UNITS_MID = 6
MAX_UNITS_SMALL = 8
HUGE_BYTES = 50_000_000_000
MID_BYTES = 5_000_000_000

CLAIM_BOUNDARY = (
    "Static sidecar artifact. A specimen-seal event ADDS a WorkGraph to the "
    "running Odyssey I mission. It does not restart Odyssey, does not rewrite "
    "ODYSSEY_I_LAUNCH.json, does not re-hash a sealed source, and does not "
    "record a partial-weight experiment as specimen science."
)


class SpecimenEventError(ValueError):
    """Base error for the seal-transition event path."""


class IllegalTransition(SpecimenEventError):
    """A state jump that is not on the acquisition continuum."""


class PartialWeightScienceError(SpecimenEventError):
    """Weights are required and the source is not a sealed specimen."""


class MissionRestartForbidden(SpecimenEventError):
    """A model arriving must ADD a WorkGraph. Restarting Odyssey is illegal."""


class UnsealedSourceError(SpecimenEventError):
    """Architecture/size/filenames may be early metadata; science needs a seal."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _canonical_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _atomic_write(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, dict(doc))


def slug(repo: str, revision: str | None) -> str:
    name = (repo or "specimen").split("/")[-1]
    name = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    rev = (revision or "unpinned")[:12]
    return f"{name}@{rev}"


def lake_tag(repo: str, revision: str) -> str:
    return repo.replace("/", "--") + "@" + revision[:12]


def restart_odyssey(*_args: Any, **_kwargs: Any) -> None:
    """Named so a caller who reaches for a restart hits a wall, not a launch."""
    raise MissionRestartForbidden(
        "a sealed specimen ADDS a WorkGraph to the running mission; "
        "restarting Odyssey is forbidden"
    )


# ---------------------------------------------------------------------------
# Durable event log. Survives process death. Replay is the authority, not a
# callback table.
# ---------------------------------------------------------------------------


def event_id_for(payload: Mapping[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k not in {"event_id", "ts", "seal_sha256"}}
    return _canonical_hash(body)


def make_transition_event(
    *,
    from_state: str,
    to_state: str,
    specimen: Mapping[str, Any],
    replay: bool,
    source: str,
    extras: Mapping[str, Any] | None = None,
    ts: str | None = None,
) -> dict[str, Any]:
    if from_state not in SPECIMEN_STATES or to_state not in SPECIMEN_STATES:
        raise IllegalTransition(f"unknown state {from_state!r} -> {to_state!r}")
    if from_state == to_state:
        raise IllegalTransition(f"not a transition: {from_state}")
    if (from_state, to_state) not in LEGAL_TRANSITIONS:
        raise IllegalTransition(f"illegal transition {from_state} -> {to_state}")
    row: dict[str, Any] = {
        "schema": EVENT_SCHEMA,
        "kind": KIND_TRANSITION,
        "from_state": from_state,
        "to_state": to_state,
        "specimen": dict(specimen),
        "replay": bool(replay),
        "live_arrival": (not replay) and source == "live_watcher",
        "source": source,
        "ts": ts or _now(),
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
    }
    if extras:
        row.update(dict(extras))
    row["event_id"] = event_id_for(row)
    return row


class SpecimenEventLog:
    """Append-only JSON document. An in-memory callback is not this."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.events: list[dict[str, Any]] = []
        if self.path is not None and self.path.is_file():
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
        except (OSError, json.JSONDecodeError) as exc:
            raise SpecimenEventError(f"corrupt event log {self.path}: {exc}") from exc
        if not isinstance(data, dict):
            raise SpecimenEventError(f"{self.path} root is not an object")
        if data.get("schema") not in {LOG_SCHEMA, EVENT_SCHEMA, SCHEMA}:
            raise SpecimenEventError(f"{self.path} schema {data.get('schema')!r} is not an event log")
        rows = data.get("events")
        if not isinstance(rows, list):
            raise SpecimenEventError(f"{self.path} missing events list")
        self.events = [dict(r) for r in rows if isinstance(r, dict)]

    def to_durable(self) -> dict[str, Any]:
        return {
            "schema": LOG_SCHEMA,
            "version": VERSION,
            "n_events": len(self.events),
            "events": list(self.events),
            "durable": True,
            "replayable": True,
            "in_memory_callback_is_not_this": True,
        }

    def save(self) -> Path | None:
        if self.path is None:
            return None
        _atomic_write(self.path, self.to_durable())
        return self.path

    def known_ids(self) -> set[str]:
        return {str(e.get("event_id")) for e in self.events if e.get("event_id")}

    def append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(event)
        eid = str(row.get("event_id") or event_id_for(row))
        row["event_id"] = eid
        if eid in self.known_ids():
            return {"kind": "idempotent", "event": row, "inserted": False}
        self.events.append(row)
        self.save()
        return {"kind": "inserted", "event": row, "inserted": True}

    def replay(self) -> list[dict[str, Any]]:
        """Return events in persist order. Restart-safe: disk is authority."""
        if self.path is not None and self.path.is_file():
            self._load()
        return list(self.events)

    def transitions_to(self, state: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("kind") == KIND_TRANSITION and e.get("to_state") == state]


# ---------------------------------------------------------------------------
# Running mission. Identity is the launch receipt. Arrival must not mint a new
# one and must not rewrite phase_transition.
# ---------------------------------------------------------------------------


def load_launch_doc(path: Path | None = None) -> dict[str, Any]:
    target = path or LAUNCH_RECEIPT
    if not target.is_file():
        raise SpecimenEventError(f"launch receipt missing: {target}")
    doc = load_json(target)
    if not isinstance(doc, dict):
        raise SpecimenEventError("launch receipt is not an object")
    return doc


def mission_id_of(doc: Mapping[str, Any]) -> str:
    seal = str(doc.get("seal_sha256") or "")
    if len(seal) != 64:
        raise SpecimenEventError("launch receipt has no seal; cannot name the mission")
    return f"odyssey-i/{seal}"


def phase_transition_of(doc: Mapping[str, Any]) -> str:
    phase = str(doc.get("phase_transition") or "")
    if not phase:
        raise SpecimenEventError("launch receipt has no phase_transition")
    return phase


@dataclass
class RunningMission:
    """The live Odyssey I mission. Adding work does not rebind identity."""

    mission_id: str
    phase_transition: str
    launch_seal_sha256: str
    launch_path: str
    existing_unit_ids: tuple[str, ...]
    first_specimen: dict[str, Any]
    added_graphs: list[dict[str, Any]] = field(default_factory=list)
    wakes: list[dict[str, Any]] = field(default_factory=list)
    _identity_frozen: bool = field(default=True, repr=False)

    def identity(self) -> dict[str, str]:
        return {
            "mission_id": self.mission_id,
            "phase_transition": self.phase_transition,
            "launch_seal_sha256": self.launch_seal_sha256,
        }

    def rebind_identity(self, **_fields: Any) -> None:
        raise MissionRestartForbidden(
            "mission identity is the launch receipt; a specimen arrival cannot rebind it"
        )

    def add_workgraph(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        if not snapshot.get("units"):
            raise SpecimenEventError("refusing to add an empty WorkGraph")
        rec = dict(snapshot)
        rec.setdefault("added_to_mission_id", self.mission_id)
        rec.setdefault("phase_transition_unchanged", self.phase_transition)
        rec.setdefault("restarts_odyssey", False)
        self.added_graphs.append(rec)
        return rec

    def record_wake(self, wake: Mapping[str, Any]) -> None:
        self.wakes.append(dict(wake))


def load_running_mission(path: Path | None = None) -> RunningMission:
    target = path or LAUNCH_RECEIPT
    doc = load_launch_doc(target)
    units = []
    graphs = doc.get("first_workgraphs") if isinstance(doc.get("first_workgraphs"), Mapping) else {}
    if isinstance(graphs, Mapping):
        units = [str(u.get("id")) for u in (graphs.get("units") or []) if isinstance(u, Mapping) and u.get("id")]
        specimen = dict(graphs.get("specimen") or {})
    else:
        specimen = {}
    return RunningMission(
        mission_id=mission_id_of(doc),
        phase_transition=phase_transition_of(doc),
        launch_seal_sha256=str(doc.get("seal_sha256")),
        launch_path=str(target),
        existing_unit_ids=tuple(units),
        first_specimen=specimen,
    )


def launch_receipt_untouched(mission: RunningMission) -> dict[str, Any]:
    """Re-read the launch file. Arrival is illegal if either field moved."""
    path = Path(mission.launch_path)
    doc = load_launch_doc(path)
    return {
        "mission_id_unchanged": mission_id_of(doc) == mission.mission_id,
        "phase_transition_unchanged": phase_transition_of(doc) == mission.phase_transition,
        "seal_unchanged": str(doc.get("seal_sha256")) == mission.launch_seal_sha256,
        "mission_id": mission_id_of(doc),
        "phase_transition": phase_transition_of(doc),
        "seal_sha256": str(doc.get("seal_sha256")),
        "path": str(path),
    }


# ---------------------------------------------------------------------------
# Sealed identity. Cite the existing whole-tree receipt. Do not re-hash.
# ---------------------------------------------------------------------------


def load_verification_rows() -> list[dict[str, Any]]:
    if not VERIFY_RECEIPT.is_file():
        return []
    try:
        doc = load_json(VERIFY_RECEIPT)
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    rows = doc.get("results") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        return []
    return [dict(r) for r in rows if isinstance(r, Mapping)]


def sealed_row_for(tag: str, rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any] | None:
    pool = list(rows) if rows is not None else load_verification_rows()
    for row in pool:
        if str(row.get("specimen") or "") == tag and row.get("whole_tree_verified") is True:
            if str(row.get("status") or "") == "WHOLE_TREE_VERIFIED":
                return dict(row)
    return None


def cite_existing_seal(row: Mapping[str, Any], *, expected_revision: str | None) -> dict[str, Any]:
    """Verify revision against the already-sealed receipt. Never re-hash."""
    cited_rev = (
        str(row.get("resolved_sha") or row.get("revision") or "")
        or None
    )
    # The verification receipt rows in this campaign carry the lake tag, not
    # always a revision field. The tag suffix is the first 12 of the pin.
    tag = str(row.get("specimen") or "")
    tag_rev = tag.rsplit("@", 1)[-1] if "@" in tag else ""
    matches = True
    if expected_revision:
        pin = expected_revision[:12]
        matches = bool(
            (cited_rev and (cited_rev.startswith(pin) or pin.startswith(cited_rev[:12])))
            or (tag_rev and tag_rev.startswith(pin[: len(tag_rev)]))
            or pin.startswith(tag_rev)
        )
    return {
        "action": "CITED_EXISTING_SEAL",
        "rehashed": False,
        "whole_tree_verified": True,
        "status": row.get("status"),
        "specimen": row.get("specimen"),
        "bytes_hashed_cited": row.get("bytes_hashed"),
        "n_files_cited": row.get("n_files"),
        "n_sha256_verified_cited": row.get("n_sha256_verified") or row.get("verified"),
        "revision_matches": matches,
        "expected_revision": expected_revision,
        "tag_revision_prefix": tag_rev,
        "verification_source": "receipts/future/SPECIMEN_VERIFICATION.json",
        "rule": "a sealed source is not re-hashed; the existing whole-tree receipt is the seal",
    }


# ---------------------------------------------------------------------------
# Architecture fingerprint and organ census. Config + index only. No weights.
# ---------------------------------------------------------------------------


def _text_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    inner = cfg.get("text_config")
    if isinstance(inner, Mapping):
        return inner
    return cfg


def fingerprint_from_config(
    cfg: Mapping[str, Any],
    *,
    weight_names: Sequence[str] = (),
    total_size: int | None = None,
) -> dict[str, Any]:
    """Metadata fingerprint. Opening a safetensors shard is out of scope."""
    text = _text_cfg(cfg)
    arches = cfg.get("architectures") if isinstance(cfg.get("architectures"), list) else []
    model_type = str(cfg.get("model_type") or text.get("model_type") or "")
    hidden = text.get("hidden_size")
    layers = text.get("num_hidden_layers")
    heads = text.get("num_attention_heads")
    kv = text.get("num_key_value_heads")
    intermediate = text.get("intermediate_size")
    vocab = text.get("vocab_size")
    families: list[str] = []
    if weight_names:
        blob = " ".join(weight_names)
        for name, pat in ORGAN_PATTERNS:
            if pat.search(blob) and name not in families:
                families.append(name)
    else:
        # Config-only: still name organs the architecture declares.
        if cfg.get("vision_config") or "vl" in model_type.lower() or any(
            "vl" in str(a).lower() for a in arches
        ):
            families.append("vision")
        if text.get("num_experts") or cfg.get("num_experts") or text.get("n_routed_experts"):
            families.append("moe_expert")
            families.append("moe_router")
        families.extend(["attention", "mlp", "embed", "lm_head", "norm"])
        if "mamba" in model_type.lower() or "falcon_h1" in model_type.lower() or any(
            "mamba" in str(a).lower() or "falconh1" in str(a).lower() for a in arches
        ):
            if "mamba_ssm" not in families:
                families.insert(0, "mamba_ssm")
    architecture_family = model_type or "UNKNOWN"
    if "falcon_h1" in model_type.lower() or any("falconh1" in str(a).lower() for a in arches):
        architecture_family = "falcon_h1"
    elif "qwen3_vl" in model_type.lower() or any("qwen3vl" in str(a).lower() for a in arches):
        architecture_family = "qwen3_vl"
    elif "inkling" in model_type.lower() or any("inkling" in str(a).lower() for a in arches):
        architecture_family = "inkling_mm"
    elif any("glm" in str(a).lower() for a in arches) or "glm" in model_type.lower():
        architecture_family = "glm"
    multimodal = bool(cfg.get("vision_config")) or "vl" in architecture_family or architecture_family in {
        "qwen3_vl",
        "inkling_mm",
    }
    moe = "moe_expert" in families or bool(text.get("num_experts") or cfg.get("n_routed_experts"))
    return {
        "architectures": [str(a) for a in arches],
        "model_type": model_type,
        "architecture_family": architecture_family,
        "hidden_size": hidden if isinstance(hidden, int) else None,
        "num_hidden_layers": layers if isinstance(layers, int) else None,
        "num_attention_heads": heads if isinstance(heads, int) else None,
        "num_key_value_heads": kv if isinstance(kv, int) else None,
        "intermediate_size": intermediate if isinstance(intermediate, int) else None,
        "vocab_size": vocab if isinstance(vocab, int) else None,
        "multimodal": multimodal,
        "moe": moe,
        "organ_families": families,
        "n_named_tensors": len(weight_names),
        "size_bytes": total_size if isinstance(total_size, int) else None,
        "weights_opened": False,
        "source": "config.json+optional_index",
        "evidence_class": "STATIC_ONLY",
    }


def read_config(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    return doc if isinstance(doc, dict) else None


def read_index_names(path: Path) -> tuple[list[str], int | None]:
    if not path.is_file():
        return [], None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return [], None
    if not isinstance(doc, dict):
        return [], None
    weight_map = doc.get("weight_map") if isinstance(doc.get("weight_map"), dict) else {}
    names = [str(k) for k in weight_map]
    meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    total = meta.get("total_size")
    return names, (int(total) if isinstance(total, int) else None)


def fingerprint_on_disk(specimen_path: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    if not specimen_path:
        return None
    root = Path(specimen_path)
    cfg = read_config(root / "config.json")
    if cfg is None:
        return None
    names, total = read_index_names(root / "model.safetensors.index.json")
    return fingerprint_from_config(cfg, weight_names=names, total_size=total)


def candidate_curriculum_roles(fingerprint: Mapping[str, Any], *, size_bytes: int | None) -> list[dict[str, Any]]:
    """CANDIDATE roles. Ready-ness stays with specimen_curriculum.py."""
    family = str(fingerprint.get("architecture_family") or "")
    hidden = fingerprint.get("hidden_size")
    multimodal = bool(fingerprint.get("multimodal"))
    moe = bool(fingerprint.get("moe"))
    out: list[dict[str, Any]] = []

    def add(role: str, why: str) -> None:
        out.append({"role": role, "status": "CANDIDATE", "why": why})

    if family == "falcon_h1":
        add(
            "small_dense_alternate_architecture_transfer",
            "falcon_h1 is the alternate-architecture first-wave role",
        )
    if family in {"qwen4_exp"} or "flash" in family.lower():
        add("flash_heterogeneous_frontier", f"architecture_family={family}")
    if isinstance(hidden, int) and hidden <= 1536 and not multimodal and not moe:
        add("very_small_dense_procedural_speed", f"hidden_size={hidden} is in the very-small dense band")
    if (
        isinstance(hidden, int)
        and 2048 <= hidden <= 6144
        and not multimodal
        and not moe
        and family not in {"falcon_h1"}
        and isinstance(size_bytes, int)
        and size_bytes >= 20_000_000_000
    ):
        add("mid_size_dense_compiler", f"hidden_size={hidden} mid-size dense")
    if multimodal:
        add("multimodal_vl_candidate", "config declares a vision tower or VL architecture")
    if moe:
        add("moe_candidate", "config/index names experts or a router")
    if family in {"glm", "inkling_mm"} or "flash" in family:
        add("heterogeneous_frontier_candidate", f"architecture_family={family}")
    if not out:
        add("deferred_lake_entry", "no first-wave role matches; recorded and deferred")
    return out


# ---------------------------------------------------------------------------
# Laws and scars FIRST. New search only where transfer does not apply / is dead.
# ---------------------------------------------------------------------------


def load_laws() -> list[dict[str, Any]]:
    if not LAW_STORE_RECEIPT.is_file():
        return []
    try:
        doc = load_json(LAW_STORE_RECEIPT)
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    rows = doc.get("laws") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        return []
    return [dict(r) for r in rows if isinstance(r, Mapping)]


def transferable_laws(laws: Sequence[Mapping[str, Any]], fingerprint: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Laws this specimen may test before opening new search.

    MODEL_LOCAL on a different parent is not a transfer. GENERIC and matching
    ARCHITECTURE_FAMILY laws are. That is the cheap path and Odyssey II leverage.
    """
    family = str(fingerprint.get("architecture_family") or "")
    hits: list[dict[str, Any]] = []
    for law in laws:
        scope = str(law.get("scope") or "")
        law_fam = str(law.get("architecture_family") or "")
        if scope in {"GENERIC_CANDIDATE", "GENERIC_VERIFIED", "GENERIC"}:
            hits.append(dict(law))
            continue
        if scope == "ARCHITECTURE_FAMILY" and law_fam and family and law_fam == family:
            hits.append(dict(law))
            continue
        # A MACHINE_LOCAL or BACKEND_FAMILY law is still cheaper than a cold
        # search: it is a claimed regularity, not a new family.
        if scope in {"MACHINE_LOCAL", "BACKEND_FAMILY"}:
            hits.append(dict(law))
    return hits


def relevant_scars(
    *,
    fingerprint: Mapping[str, Any],
    scars: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Query the campaign's closed schools before proposing them again."""
    pool = list(scars) if scars is not None else ni.ingest()
    organ = "mlp" if "mlp" in (fingerprint.get("organ_families") or ["mlp"]) else "unrecorded"
    model = "qwen3.8-27b"
    out: list[dict[str, Any]] = []
    for family in WAVE_DEAD_FAMILIES:
        got = ni.refuse_if_dead(
            {"hypothesis_family": family, "organ": organ, "model": model},
            scars=pool,
        )
        out.append(
            {
                "hypothesis_family": family,
                "refused": bool(got),
                "scar": ({k: got.get(k) for k in ("scar_id", "source_path", "verdict", "level")} if got else None),
            }
        )
    return out


def cheapest_first_experiments(
    *,
    fingerprint: Mapping[str, Any],
    laws: Sequence[Mapping[str, Any]],
    scars: Sequence[Mapping[str, Any]],
    size_bytes: int | None,
) -> list[dict[str, Any]]:
    """Prioritized, bounded. Transfer and scars before new search."""
    budget = experiment_budget(size_bytes)
    experiments: list[dict[str, Any]] = []
    for law in laws:
        experiments.append(
            {
                "kind": "transfer_law",
                "campaign_phase": CAMPAIGN_TRANSFER,
                "law_id": law.get("law_id"),
                "scope": law.get("scope"),
                "organ_class": law.get("organ_class"),
                "cost_units": 1,
                "expected_information_gain": wg.INFO_HIGH,
                "requires_weights": False,
                "why": "transferable law is cheaper than cold search",
            }
        )
    closed = [s for s in scars if s.get("refused")]
    open_dead = [s for s in scars if not s.get("refused")]
    experiments.append(
        {
            "kind": "scar_lookup",
            "campaign_phase": CAMPAIGN_TRANSFER,
            "n_closed": len(closed),
            "closed_families": [s["hypothesis_family"] for s in closed],
            "still_visible_gap": [s["hypothesis_family"] for s in open_dead],
            "cost_units": 1,
            "expected_information_gain": wg.INFO_HIGH,
            "requires_weights": False,
            "why": "a scar the index can see prunes the family before a unit is emitted",
        }
    )
    if budget["allow_new_search"] and open_dead:
        # A school the index cannot see is a bug, not an invitation. Do not
        # emit new search for WAVE_DEAD even if a lookup missed.
        pass
    if budget["allow_new_search"]:
        experiments.append(
            {
                "kind": "new_search_where_transfer_inapplicable",
                "campaign_phase": CAMPAIGN_NEW_SEARCH,
                "cost_units": 3,
                "expected_information_gain": wg.INFO_MEDIUM,
                "requires_weights": False,
                "why": (
                    "open new search only after transfer and scar lookup; "
                    "this unit depends on both and is skipped if they already cover the organ"
                ),
                "pruned_families": [s["hypothesis_family"] for s in closed],
            }
        )
    # Bound.
    return experiments[: int(budget["max_units"])]


def experiment_budget(size_bytes: int | None) -> dict[str, Any]:
    n = int(size_bytes) if isinstance(size_bytes, int) else 0
    if n >= HUGE_BYTES:
        return {
            "max_units": MAX_UNITS_HUGE,
            "allow_new_search": False,
            "allow_gpu": False,
            "band": "huge",
            "size_bytes": n,
            "why": "a giant specimen earns cite+fingerprint+transfer+scars, not a full-depth graph",
        }
    if n >= MID_BYTES:
        return {
            "max_units": MAX_UNITS_MID,
            "allow_new_search": True,
            "allow_gpu": False,
            "band": "mid",
            "size_bytes": n,
            "why": "mid-size: transfer/scars first, at most one new-search unit, no GPU depth",
        }
    return {
        "max_units": MAX_UNITS_SMALL,
        "allow_new_search": True,
        "allow_gpu": False,
        "band": "small",
        "size_bytes": n,
        "why": "small: still bounded; GPU stages stay out until economics and a role earn them",
    }


# ---------------------------------------------------------------------------
# WorkGraph construction. Transfer/scar units have no edge onto new_search;
# new_search depends on them. That is the order test.
# ---------------------------------------------------------------------------


def _unit(
    *,
    uid: str,
    role: str,
    description: str,
    dependencies: Sequence[str],
    lane: str,
    info: int,
    cost: int,
    phase: str,
    extras: Mapping[str, Any] | None = None,
    requires_hardware: bool = False,
) -> dict[str, Any]:
    extra = {
        "campaign_phase": phase,
        "species": "odyssey_i_specimen_arrival",
        "odyssey": "I",
        "era": "I",
        "restarts_odyssey": False,
    }
    if extras:
        extra.update(dict(extras))
    return wg.make_unit(
        id=uid,
        role=role,
        description=description,
        dependencies=list(dependencies),
        resource_lane=lane,
        mutation_scope=[f"specimen:{uid.rsplit('.', 1)[0]}"],
        verifier=f"future.specimen_events.{uid}",
        expected_information_gain=info,
        cost_units=cost,
        requires_hardware=requires_hardware,
        species="odyssey_i_specimen_arrival",
        effect_class="READ_ONLY",
        extras=extra,
    )


def plan_bounded_workgraph(
    *,
    specimen: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    experiments: Sequence[Mapping[str, Any]],
    cite: Mapping[str, Any],
    roles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    repo = str(specimen.get("repo") or "unknown")
    revision = specimen.get("revision")
    tag = str(specimen.get("tag") or lake_tag(repo, str(revision or "")))
    prefix = f"odyssey-i.arrival.{slug(repo, str(revision) if revision else None)}"
    graph = wg.WorkGraph(ncpu=8)
    budget = experiment_budget(fingerprint.get("size_bytes") or specimen.get("bytes_hashed"))

    cite_id = f"{prefix}.cite_existing_seal"
    fp_id = f"{prefix}.architecture_fingerprint"
    transfer_id = f"{prefix}.transfer_laws"
    scars_id = f"{prefix}.relevant_scars"
    search_id = f"{prefix}.new_search"

    inserted: list[str] = []

    def admit(unit: dict[str, Any]) -> None:
        if len(inserted) >= int(budget["max_units"]):
            return
        result = graph.admit(unit)
        if result.get("kind") in {"inserted", "idempotent"} and result.get("unit"):
            inserted.append(str(result["unit"]["id"]))

    admit(
        _unit(
            uid=cite_id,
            role="verify",
            description=f"cite existing whole-tree seal for {tag}; do not re-hash",
            dependencies=[],
            lane="CPU_VERIFY",
            info=wg.INFO_HIGH,
            cost=1,
            phase=CAMPAIGN_TRANSFER,
            extras={"cite": {"rehashed": cite.get("rehashed"), "action": cite.get("action")}},
        )
    )
    admit(
        _unit(
            uid=fp_id,
            role="science",
            description=f"fingerprint architecture and organ families from config/index for {tag}",
            dependencies=[cite_id],
            lane="CPU_ANALYSIS",
            info=wg.INFO_HIGH,
            cost=1,
            phase=CAMPAIGN_TRANSFER,
            extras={
                "architecture_family": fingerprint.get("architecture_family"),
                "organ_families": list(fingerprint.get("organ_families") or []),
                "weights_opened": False,
                "candidate_roles": [r.get("role") for r in roles],
            },
        )
    )
    admit(
        _unit(
            uid=transfer_id,
            role="science",
            description=f"test transferable laws on {tag} before opening new search",
            dependencies=[fp_id],
            lane="CPU_ANALYSIS",
            info=wg.INFO_HIGH,
            cost=1,
            phase=CAMPAIGN_TRANSFER,
            extras={
                "n_transferable_laws": sum(1 for e in experiments if e.get("kind") == "transfer_law"),
                "law_ids": [e.get("law_id") for e in experiments if e.get("kind") == "transfer_law"],
            },
        )
    )
    admit(
        _unit(
            uid=scars_id,
            role="science",
            description=f"query relevant scars (including WAVE_DEAD families) for {tag} before new search",
            dependencies=[fp_id],
            lane="CPU_ANALYSIS",
            info=wg.INFO_HIGH,
            cost=1,
            phase=CAMPAIGN_TRANSFER,
            extras={
                "wave_dead_families": list(WAVE_DEAD_FAMILIES),
                "scar_lookup": [e for e in experiments if e.get("kind") == "scar_lookup"],
            },
        )
    )
    want_search = any(e.get("kind") == "new_search_where_transfer_inapplicable" for e in experiments)
    if want_search and budget["allow_new_search"]:
        admit(
            _unit(
                uid=search_id,
                role="science",
                description=(
                    f"open new search on {tag} only where transfer is inapplicable "
                    "and scars have not closed the family"
                ),
                dependencies=[transfer_id, scars_id],
                lane="CPU_ANALYSIS",
                info=wg.INFO_MEDIUM,
                cost=3,
                phase=CAMPAIGN_NEW_SEARCH,
                extras={"pruned_families": list(WAVE_DEAD_FAMILIES)},
            )
        )

    units = [dict(graph.units[uid]) for uid in sorted(graph.units)]
    transfer_ids = [u["id"] for u in units if u.get("campaign_phase") == CAMPAIGN_TRANSFER]
    search_ids = [u["id"] for u in units if u.get("campaign_phase") == CAMPAIGN_NEW_SEARCH]
    snapshot = {
        "schema": "hawking.future.specimen_arrival_workgraph.v1",
        "specimen": {"repo": repo, "revision": revision, "tag": tag, "slug": slug(repo, str(revision) if revision else None)},
        "n_units": len(units),
        "unit_ids": [u["id"] for u in units],
        "units": units,
        "transfer_and_scar_ids": transfer_ids,
        "new_search_ids": search_ids,
        "budget": budget,
        "bounded": True,
        "not_exhaustive": True,
        "restarts_odyssey": False,
        "durable_graph": graph.to_durable(),
    }
    snapshot["transfer_before_new_search"] = transfer_runs_before_new_search(snapshot)
    snapshot["transfer_does_not_depend_on_new_search"] = all(
        not (set(search_ids) & set(u.get("dependencies") or []))
        for u in units
        if u["id"] in transfer_ids
    )
    return snapshot


def transfer_runs_before_new_search(snapshot: Mapping[str, Any]) -> bool:
    units = snapshot.get("units") or []
    by_id = {u["id"]: u for u in units if isinstance(u, Mapping) and u.get("id")}
    transfer_ids = [u["id"] for u in units if u.get("campaign_phase") == CAMPAIGN_TRANSFER]
    search_ids = [u["id"] for u in units if u.get("campaign_phase") == CAMPAIGN_NEW_SEARCH]
    if not transfer_ids:
        return False
    for sid in search_ids:
        deps = set(by_id[sid].get("dependencies") or [])
        if not (deps & set(transfer_ids)):
            return False
    for tid in transfer_ids:
        deps = set(by_id[tid].get("dependencies") or [])
        if deps & set(search_ids):
            return False
    return True


# ---------------------------------------------------------------------------
# Science recording. Early metadata is a different kind. Partial weights fail.
# ---------------------------------------------------------------------------


def record_early_metadata(
    *,
    specimen: Mapping[str, Any],
    state: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if state not in SPECIMEN_STATES:
        raise SpecimenEventError(f"unknown state {state}")
    return {
        "kind": KIND_EARLY_METADATA,
        "specimen_state": state,
        "is_specimen_science": False,
        "weights_required": False,
        "specimen": dict(specimen),
        "payload": dict(payload),
        "rule": "architecture, size and filenames may be learned during download; this is not a sealed specimen",
    }


def record_specimen_science(
    *,
    specimen_state: str,
    experiment: Mapping[str, Any],
    sealed: bool,
) -> dict[str, Any]:
    """Refuse a weight-requiring (or any) science record unless the source is sealed."""
    requires_weights = bool(experiment.get("requires_weights"))
    stage = str(experiment.get("stage") or experiment.get("kind") or "")
    if stage.lower() in WEIGHT_REQUIRING_STAGES:
        requires_weights = True
    if specimen_state != SEALED_SOURCE_SPECIMEN or not sealed:
        raise PartialWeightScienceError(
            f"refusing specimen science in state {specimen_state}: "
            "early metadata is not a sealed specimen; a partial-weight experiment "
            "cannot be recorded as specimen science"
        )
    if requires_weights and specimen_state != SEALED_SOURCE_SPECIMEN:
        raise PartialWeightScienceError(
            f"experiment {stage!r} requires weights and the source is not sealed"
        )
    return {
        "kind": "SPECIMEN_SCIENCE",
        "specimen_state": specimen_state,
        "is_specimen_science": True,
        "requires_weights": requires_weights,
        "experiment": dict(experiment),
        "sealed": True,
    }


# ---------------------------------------------------------------------------
# Scheduler wake. Named consumer, not a process spawn.
# ---------------------------------------------------------------------------


WakeFn = Callable[[Mapping[str, Any], RunningMission], None]


class SchedulerWake:
    """Wake the running scheduler. Does not launch Odyssey."""

    def __init__(self) -> None:
        self.invocations: list[dict[str, Any]] = []

    def wake(self, event: Mapping[str, Any], mission: RunningMission) -> dict[str, Any]:
        if event.get("to_state") != SEALED_SOURCE_SPECIMEN:
            raise UnsealedSourceError("scheduler wakes on SEALED_SOURCE_SPECIMEN only")
        rec = {
            "kind": KIND_SCHEDULER_WAKE,
            "event_id": event.get("event_id"),
            "mission_id": mission.mission_id,
            "phase_transition": mission.phase_transition,
            "restarts_odyssey": False,
            "human_reminder": False,
            "ts": _now(),
        }
        self.invocations.append(rec)
        mission.record_wake(rec)
        return rec


# ---------------------------------------------------------------------------
# Arrival: the scientific event. Adds a WorkGraph. Never restarts.
# ---------------------------------------------------------------------------


def apply_sealed_arrival(
    event: Mapping[str, Any],
    mission: RunningMission,
    *,
    log: SpecimenEventLog | None = None,
    wake: SchedulerWake | None = None,
    fingerprint: Mapping[str, Any] | None = None,
    cite: Mapping[str, Any] | None = None,
    laws: Sequence[Mapping[str, Any]] | None = None,
    scar_rows: Sequence[Mapping[str, Any]] | None = None,
    roles: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if event.get("to_state") != SEALED_SOURCE_SPECIMEN:
        raise UnsealedSourceError(
            f"arrival requires SEALED_SOURCE_SPECIMEN, got {event.get('to_state')}"
        )
    before = mission.identity()
    identity_check = launch_receipt_untouched(mission)
    if not identity_check["phase_transition_unchanged"] or not identity_check["mission_id_unchanged"]:
        raise MissionRestartForbidden("launch receipt identity moved under us")

    specimen = dict(event.get("specimen") or {})
    fp = dict(fingerprint or {})
    cited = dict(cite or {})
    law_hits = list(laws) if laws is not None else transferable_laws(load_laws(), fp)
    scars = list(scar_rows) if scar_rows is not None else relevant_scars(fingerprint=fp)
    role_rows = list(roles) if roles is not None else candidate_curriculum_roles(
        fp, size_bytes=fp.get("size_bytes") or specimen.get("bytes_hashed")
    )
    experiments = cheapest_first_experiments(
        fingerprint=fp,
        laws=law_hits,
        scars=scars,
        size_bytes=fp.get("size_bytes") or specimen.get("bytes_hashed"),
    )
    snapshot = plan_bounded_workgraph(
        specimen=specimen,
        fingerprint=fp,
        experiments=experiments,
        cite=cited,
        roles=role_rows,
    )
    if not transfer_runs_before_new_search(snapshot):
        raise SpecimenEventError("new search is not gated on transfer-and-scar lookup")

    waker = wake or SchedulerWake()
    wake_rec = waker.wake(event, mission)
    added = mission.add_workgraph(snapshot)
    after = mission.identity()
    if after != before:
        raise MissionRestartForbidden(
            f"mission identity changed across arrival: before={before} after={after}"
        )
    post = launch_receipt_untouched(mission)
    if not (post["mission_id_unchanged"] and post["phase_transition_unchanged"] and post["seal_unchanged"]):
        raise MissionRestartForbidden("launch receipt was rewritten during arrival")

    added_event = {
        "schema": EVENT_SCHEMA,
        "kind": KIND_WORKGRAPH_ADDED,
        "event_id": event_id_for(
            {
                "kind": KIND_WORKGRAPH_ADDED,
                "parent": event.get("event_id"),
                "unit_ids": snapshot["unit_ids"],
            }
        ),
        "parent_event_id": event.get("event_id"),
        "mission_id": mission.mission_id,
        "phase_transition": mission.phase_transition,
        "restarts_odyssey": False,
        "n_units": snapshot["n_units"],
        "unit_ids": snapshot["unit_ids"],
        "ts": _now(),
    }
    if log is not None:
        log.append(event)
        log.append(wake_rec)
        log.append(added_event)

    return {
        "kind": "sealed_arrival",
        "replay": bool(event.get("replay")),
        "live_arrival": bool(event.get("live_arrival")),
        "event_id": event.get("event_id"),
        "mission_id": mission.mission_id,
        "phase_transition": mission.phase_transition,
        "mission_id_unchanged": before["mission_id"] == after["mission_id"],
        "phase_transition_unchanged": before["phase_transition"] == after["phase_transition"],
        "launch_seal_unchanged": post["seal_unchanged"],
        "restarts_odyssey": False,
        "wake": wake_rec,
        "cite": cited,
        "fingerprint": fp,
        "candidate_roles": role_rows,
        "transferable_laws": [
            {"law_id": l.get("law_id"), "scope": l.get("scope"), "organ_class": l.get("organ_class")}
            for l in law_hits
        ],
        "scars": scars,
        "experiments": experiments,
        "workgraph": snapshot,
        "added_workgraph": {
            "n_units": added.get("n_units"),
            "unit_ids": added.get("unit_ids"),
        },
        "transfer_before_new_search": transfer_runs_before_new_search(snapshot),
        "identity_before": before,
        "identity_after": after,
    }


def replay_log_onto_mission(
    log: SpecimenEventLog,
    mission: RunningMission,
    *,
    handler: Callable[[Mapping[str, Any], RunningMission], Any] | None = None,
) -> list[dict[str, Any]]:
    """Replay persisted transitions. A second process can pick this up."""
    out: list[dict[str, Any]] = []
    for event in log.replay():
        if event.get("kind") != KIND_TRANSITION:
            continue
        if event.get("to_state") != SEALED_SOURCE_SPECIMEN:
            continue
        if handler is not None:
            out.append(handler(event, mission))
        else:
            out.append({"replayed_event_id": event.get("event_id"), "mission_id": mission.mission_id})
    return out


# ---------------------------------------------------------------------------
# Demonstration: live arrival if one sealed while we ran; else replay.
# ---------------------------------------------------------------------------


def sealed_specimens_from_receipts() -> list[dict[str, Any]]:
    """Identity of already-sealed specimens. Does not re-hash them."""
    rows = load_verification_rows()
    launch = load_launch_doc()
    curriculum = launch.get("first_specimen_set") if isinstance(launch.get("first_specimen_set"), Mapping) else {}
    roles = list((curriculum or {}).get("roles") or [])
    by_tag: dict[str, dict[str, Any]] = {}
    for role in roles:
        if not isinstance(role, Mapping):
            continue
        vs = role.get("verified_specimen") if isinstance(role.get("verified_specimen"), Mapping) else {}
        path = vs.get("specimen_path") or (role.get("modellake") or {}).get("specimen_path")
        repo = str(role.get("repo") or vs.get("repo") or "")
        revision = role.get("revision") or vs.get("revision") or vs.get("resolved_sha")
        if not repo:
            continue
        tag = lake_tag(repo, str(revision or "unpinned"))
        # External Qwen27 uses a local tag that does not match lake_tag().
        if vs.get("identity_kind") == "external_tree_digest":
            tag = "qwen3.8-27b-abliterated-bf16@local"
        by_tag[tag] = {
            "repo": repo,
            "revision": revision,
            "tag": tag,
            "role": role.get("role"),
            "architecture_family": role.get("architecture_family"),
            "specimen_path": path,
            "bytes_hashed": vs.get("bytes_hashed"),
            "n_files": vs.get("n_files"),
        }
    out: list[dict[str, Any]] = []
    for row in rows:
        tag = str(row.get("specimen") or "")
        ident = by_tag.get(tag) or {
            "repo": tag.split("@")[0].replace("--", "/", 1) if tag else "unknown",
            "revision": None,
            "tag": tag,
            "specimen_path": None,
            "bytes_hashed": row.get("bytes_hashed"),
            "n_files": row.get("n_files"),
        }
        ident = dict(ident)
        ident["verification"] = {
            "status": row.get("status"),
            "whole_tree_verified": row.get("whole_tree_verified"),
            "bytes_hashed": row.get("bytes_hashed"),
            "n_files": row.get("n_files"),
        }
        out.append(ident)
    return out


def pick_demonstration(
    *,
    live_complete_tags: Sequence[str] = (),
) -> dict[str, Any]:
    """Prefer a live ModelLake completion that is already sealed. Else replay."""
    sealed = sealed_specimens_from_receipts()
    sealed_tags = {s["tag"] for s in sealed}
    for tag in live_complete_tags:
        if tag in sealed_tags:
            specimen = next(s for s in sealed if s["tag"] == tag)
            return {
                "mode": "live_arrival",
                "replay": False,
                "source": "live_watcher",
                "specimen": specimen,
                "why": f"watcher reported {tag} complete and SPECIMEN_VERIFICATION already has a whole-tree seal",
            }
    # Prefer a sealed specimen that is NOT the first WorkGraph patient so the
    # added graph is visibly an addition, not a re-emit of the launch graph.
    # Prefer mid/small so the cheap path (laws AND scars) both become units.
    launch = load_launch_doc()
    first = ((launch.get("first_workgraphs") or {}) if isinstance(launch.get("first_workgraphs"), Mapping) else {})
    first_repo = str((first.get("specimen") or {}).get("repo") or "")
    others = [s for s in sealed if s.get("repo") != first_repo]
    def _bytes(s: Mapping[str, Any]) -> int:
        n = s.get("bytes_hashed")
        return int(n) if isinstance(n, int) else 0
    mid = [s for s in others if _bytes(s) < HUGE_BYTES]
    pick = (mid[0] if mid else None) or (others[0] if others else None)
    if pick is None and sealed:
        pick = sealed[0]
    if pick is None:
        raise SpecimenEventError("no sealed specimen exists to replay")
    return {
        "mode": "replay",
        "replay": True,
        "source": "replay_sealed_specimen",
        "specimen": pick,
        "why": (
            "no live ModelLake completion sealed while this lane ran; "
            f"replaying the already-sealed transition of {pick.get('tag')}"
        ),
    }


def demonstrate_arrival(
    *,
    log_path: Path | None = None,
    live_complete_tags: Sequence[str] = (),
) -> dict[str, Any]:
    demo = pick_demonstration(live_complete_tags=live_complete_tags)
    specimen = dict(demo["specimen"])
    tag = str(specimen.get("tag") or "")
    row = sealed_row_for(tag)
    if row is None:
        raise UnsealedSourceError(f"{tag} is not whole-tree verified; refusing to treat it as sealed")
    cite = cite_existing_seal(row, expected_revision=str(specimen.get("revision") or "") or None)
    if cite.get("rehashed"):
        raise SpecimenEventError("cite_existing_seal must not re-hash")
    fp = fingerprint_on_disk(specimen.get("specimen_path")) or fingerprint_from_config(
        {"architectures": [], "model_type": specimen.get("architecture_family") or ""}
    )
    if fp.get("size_bytes") is None and isinstance(specimen.get("bytes_hashed"), int):
        fp["size_bytes"] = specimen["bytes_hashed"]
    roles = candidate_curriculum_roles(fp, size_bytes=fp.get("size_bytes"))
    scar_pool = ni.ingest()
    scars = relevant_scars(fingerprint=fp, scars=scar_pool)
    laws = transferable_laws(load_laws(), fp)

    log = SpecimenEventLog(log_path)
    downloading_to_complete = make_transition_event(
        from_state=DOWNLOADING,
        to_state=COMPLETE_UNSEALED,
        specimen=specimen,
        replay=bool(demo["replay"]),
        source=str(demo["source"]),
    )
    complete_to_sealed = make_transition_event(
        from_state=COMPLETE_UNSEALED,
        to_state=SEALED_SOURCE_SPECIMEN,
        specimen=specimen,
        replay=bool(demo["replay"]),
        source=str(demo["source"]),
    )
    log.append(downloading_to_complete)
    mission = load_running_mission()
    before = mission.identity()
    arrival = apply_sealed_arrival(
        complete_to_sealed,
        mission,
        log=log,
        fingerprint=fp,
        cite=cite,
        laws=laws,
        scar_rows=scars,
        roles=roles,
    )
    after = mission.identity()
    untouched = launch_receipt_untouched(mission)
    # Prove replay: a fresh log object reading the same path sees both transitions.
    replayed = []
    if log.path is not None:
        replayed = SpecimenEventLog(log.path).replay()
    return {
        "demonstration": demo,
        "transitions": [downloading_to_complete, complete_to_sealed],
        "arrival": arrival,
        "identity_before": before,
        "identity_after": after,
        "launch_untouched": untouched,
        "n_log_events": len(log.events),
        "replay_sees_n_events": len(replayed),
        "durable_log_path": str(log.path) if log.path else None,
        "fingerprint_weights_opened": bool(fp.get("weights_opened")),
        "cite_rehashed": bool(cite.get("rehashed")),
    }


def _live_complete_tags_from_watcher() -> list[str]:
    """Tags that were in-flight (download_started / active) and are now complete.

    `already_complete` for the rest of the lake is not a landing that happened
    while this lane ran. A live arrival is a pending acquisition that finished.
    """
    try:
        from tools.future.sleeping_specimens import (
            _read_jsonl_tail,
            read_latest_watcher_sample,
            watcher_log_path,
        )
    except Exception:
        return []
    try:
        sample = read_latest_watcher_sample()
    except Exception:
        return []
    if not isinstance(sample, dict):
        return []
    in_flight: set[str] = set(str(t) for t in (sample.get("active_jobs") or []))
    log = watcher_log_path()
    if log is not None:
        try:
            for row in _read_jsonl_tail(log):
                if row.get("event") in {
                    "download_started",
                    "download_exit",
                    "download_recovery_refresh",
                    "download_stall",
                } and row.get("job"):
                    in_flight.add(str(row["job"]))
        except Exception:
            pass
    complete: set[str] = set()
    for row in sample.get("states") or []:
        if isinstance(row, Mapping) and row.get("state") == "complete" and row.get("job"):
            complete.add(str(row["job"]))
    # Still-active jobs have not landed. Intersection of recent in-flight
    # with now-complete is a real completion this cycle.
    landed = sorted(t for t in in_flight if t in complete and t not in set(sample.get("active_jobs") or []))
    return landed


def build() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="specimen-events-"))
    log_path = tmp / "specimen_events.jsonl.json"
    live_complete = _live_complete_tags_from_watcher()
    demo = demonstrate_arrival(log_path=log_path, live_complete_tags=live_complete)
    arrival = demo["arrival"]
    snapshot = arrival.get("workgraph") or {}
    # Partial-weight refusal is a watched negative, recorded as evidence.
    partial_refused = False
    partial_why = ""
    try:
        record_specimen_science(
            specimen_state=DOWNLOADING,
            experiment={"kind": "gravity", "stage": "gravity", "requires_weights": True},
            sealed=False,
        )
    except PartialWeightScienceError as exc:
        partial_refused = True
        partial_why = str(exc)
    early = record_early_metadata(
        specimen=demo["demonstration"]["specimen"],
        state=DOWNLOADING,
        payload={"note": "filenames and config may be learned while bytes are still arriving"},
    )
    restart_refused = False
    restart_why = ""
    try:
        restart_odyssey()
    except MissionRestartForbidden as exc:
        restart_refused = True
        restart_why = str(exc)

    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Durable replayable event for DOWNLOADING -> COMPLETE_UNSEALED -> "
            "SEALED_SOURCE_SPECIMEN. On seal, wake the Odyssey scheduler and ADD "
            "a bounded WorkGraph to the running mission. Never restart Odyssey."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "states": list(SPECIMEN_STATES),
        "legal_transitions": [list(t) for t in sorted(LEGAL_TRANSITIONS)],
        "demonstration_mode": demo["demonstration"]["mode"],
        "replay": bool(demo["demonstration"]["replay"]),
        "live_arrival": demo["demonstration"]["mode"] == "live_arrival",
        "demonstration_why": demo["demonstration"]["why"],
        "specimen": demo["demonstration"]["specimen"],
        "mission": {
            "mission_id": arrival["mission_id"],
            "phase_transition": arrival["phase_transition"],
            "launch_receipt": "receipts/future/ODYSSEY_I_LAUNCH.json",
            "identity_before": demo["identity_before"],
            "identity_after": demo["identity_after"],
            "launch_untouched": demo["launch_untouched"],
            "restarts_odyssey": False,
        },
        "durable_event": {
            "n_events": demo["n_log_events"],
            "replay_sees_n_events": demo["replay_sees_n_events"],
            "transitions": [
                {
                    "event_id": e.get("event_id"),
                    "from_state": e.get("from_state"),
                    "to_state": e.get("to_state"),
                    "replay": e.get("replay"),
                    "kind": e.get("kind"),
                }
                for e in demo["transitions"]
            ],
            "in_memory_callback_is_not_this": True,
            "survives_restart": True,
        },
        "scheduler_wake": arrival.get("wake"),
        "cite_existing_seal": arrival.get("cite"),
        "fingerprint": {
            k: arrival.get("fingerprint", {}).get(k)
            for k in (
                "architectures",
                "model_type",
                "architecture_family",
                "hidden_size",
                "num_hidden_layers",
                "organ_families",
                "n_named_tensors",
                "size_bytes",
                "multimodal",
                "moe",
                "weights_opened",
                "source",
            )
        },
        "candidate_roles": arrival.get("candidate_roles"),
        "transferable_laws": arrival.get("transferable_laws"),
        "scars": arrival.get("scars"),
        "wave_dead_families": list(WAVE_DEAD_FAMILIES),
        "experiments": arrival.get("experiments"),
        "workgraph": {
            "n_units": snapshot.get("n_units"),
            "unit_ids": snapshot.get("unit_ids"),
            "transfer_and_scar_ids": snapshot.get("transfer_and_scar_ids"),
            "new_search_ids": snapshot.get("new_search_ids"),
            "transfer_before_new_search": arrival.get("transfer_before_new_search"),
            "transfer_does_not_depend_on_new_search": snapshot.get("transfer_does_not_depend_on_new_search"),
            "budget": snapshot.get("budget"),
            "bounded": True,
            "restarts_odyssey": False,
            "units": [
                {
                    "id": u.get("id"),
                    "role": u.get("role"),
                    "campaign_phase": u.get("campaign_phase"),
                    "dependencies": u.get("dependencies"),
                    "resource_lane": u.get("resource_lane"),
                    "status": u.get("status"),
                    "expected_information_gain": u.get("expected_information_gain"),
                    "cost_units": u.get("cost_units"),
                }
                for u in (snapshot.get("units") or [])
            ],
        },
        "proofs": {
            "durable_replayable_event": demo["n_log_events"] >= 2 and demo["replay_sees_n_events"] >= 2,
            "mission_id_unchanged": arrival.get("mission_id_unchanged") is True,
            "phase_transition_unchanged": arrival.get("phase_transition_unchanged") is True,
            "launch_seal_unchanged": arrival.get("launch_seal_unchanged") is True,
            "restarts_odyssey": False,
            "restart_helper_refused": restart_refused,
            "restart_why": restart_why,
            "transfer_before_new_search": arrival.get("transfer_before_new_search") is True,
            "partial_weight_science_refused": partial_refused,
            "partial_weight_why": partial_why,
            "early_metadata_is_not_science": early.get("is_specimen_science") is False,
            "cite_did_not_rehash": arrival.get("cite", {}).get("rehashed") is False,
            "fingerprint_did_not_open_weights": arrival.get("fingerprint", {}).get("weights_opened") is False,
            "workgraph_added": int(snapshot.get("n_units") or 0) > 0,
            "workgraph_bounded": int(snapshot.get("n_units") or 0)
            <= int((snapshot.get("budget") or {}).get("max_units") or 0),
        },
        "recovered_implementation": [
            "tools/odyssey/modellake_watch.py emits download_started / already_complete / watcher_sample; it does not emit a scientific event",
            "tools/future/wakeup.py is the receipt-completion bus; this module is the specimen-seal event on that idea, keyed on ModelLake state not a receipt poll",
            "tools/future/workgraph.py WorkGraph.admit inserts units; identity conflict refuses a silent overwrite",
            "tools/future/odyssey_launch.py emit_first_workgraphs is the first-wave graph; this module ADDS a later graph and does not call launch()",
            "tools/future/negative_index.py refuse_if_dead keys MLP_FUNCTION_REPLACEMENT, MONARCH, BUTTERFLY, FACTORIZE_THE_FACTORS, PRODUCT_DICTIONARY, CONDITIONAL_PROGRAM, GENERATED_BLOCK, NONLINEAR_GENERATOR",
            "tools/future/odyssey2_law_store.py is the transferable-law store; MODEL_LOCAL on another parent is not a transfer",
            "tools/future/specimen_verify.py already sealed the demonstration specimen; this lane cites that receipt and does not re-hash",
        ],
        "gaps_closed": [
            "DOWNLOADING -> COMPLETE_UNSEALED -> SEALED_SOURCE_SPECIMEN is a durable replayable event",
            "SEALED_SOURCE_SPECIMEN wakes the scheduler with no human reminder",
            "arrival ADDS a WorkGraph; mission_id and phase_transition are unchanged",
            "transferable laws and relevant scars are queried and scheduled before new search",
            "WorkGraphs are prioritized and bounded by size-band economics, not an optimization explosion",
            "a partial-weight experiment cannot be recorded as specimen science",
            "already-sealed specimens are cited, not re-hashed",
        ],
        "negative_findings": [
            "this lane did not observe a live ModelLake seal during the run unless demonstration_mode is live_arrival",
            "tools/odyssey/modellake_watch.py is not edited; it still only logs",
            "ODYSSEY_I_LAUNCH.json is not rewritten; the gate is not re-run",
            "GPU stages are omitted from the arrival graph: depth is economics, and this sidecar has no GPU authority",
        ],
        "resident_callable": {
            "entry_point": "tools.future.specimen_events.apply_sealed_arrival(event, mission)",
            "workunit": (
                "one CPU_ANALYSIS unit: persist the seal transition and admit the "
                "bounded arrival WorkGraph onto the running mission"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.HCLI_SELF.specimen-events",
            "fails_closed": (
                "partial-weight science is refused; restart_odyssey() raises; "
                "an illegal state jump raises; an empty WorkGraph is not added"
            ),
        },
        "no_era_vi": True,
        "no_odyssey_iv": True,
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    if args.build or True:
        out = build()
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
