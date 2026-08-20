#!/usr/bin/env python3
"""Odyssey-I patient controller — queue, harvest, packets, frontier.

Queue skeleton follows tools/ascent_controller.py (load/save/value/loop shape)
but uses a new state file and a patient/work schema. Does not touch Genesis
ascent state or tools/odyssey/ (training-data Odyssey).

    python3 tools/odyssey_ctl.py --self-check
    python3 tools/odyssey_ctl.py status
    python3 tools/odyssey_ctl.py queue
    python3 tools/odyssey_ctl.py value
    python3 tools/odyssey_ctl.py harvest
    python3 tools/odyssey_ctl.py harvest --dry-run
    python3 tools/odyssey_ctl.py packet O005
    python3 tools/odyssey_ctl.py admit <slug> <est_gib>
    python3 tools/odyssey_ctl.py completions --rebuild
    python3 tools/odyssey_ctl.py run --dry-run
    python3 tools/odyssey_ctl.py run --go [--max-lanes N]
    python3 tools/odyssey_ctl.py cycle --dry-run
    python3 tools/odyssey_ctl.py cycle --go [--max-lanes N]
    python3 tools/odyssey_ctl.py retire <OXX>
    python3 tools/odyssey_ctl.py acquire-next [--go]
    python3 tools/odyssey_ctl.py economics
    bash tools/odyssey_driver.sh
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import doctor_seal  # noqa: E402
import worker_gate  # noqa: E402
import odyssey_candgen as candgen  # noqa: E402
import odyssey_costmodel as costmodel  # noqa: E402
import odyssey_memgate as memgate  # noqa: E402
import odyssey_novelty as novelty  # noqa: E402

ODYSSEY = REPO / "workspace" / "campaign" / "odyssey"
STATE = ODYSSEY / "ODYSSEY_STATE.json"
COMPLETIONS = ODYSSEY / "ODYSSEY_COMPLETIONS.json"
LEDGER = ODYSSEY / "ODYSSEY.md"
SCHEMA_PATH = ODYSSEY / "patient_packet_schema.json"
PATIENTS_DIR = ODYSSEY / "patients"
RECEIPT_DIR = REPO / "receipts" / "odyssey-i"
ESCALATIONS = ODYSSEY / "OPUS_ESCALATIONS.jsonl"
RULEBASE = ODYSSEY / "GRAVITY_RULEBASE.json"
TRANSFER = ODYSSEY / "TRANSFER_MATRIX.json"
NEGATIVE = ODYSSEY / "NEGATIVE_SCIENCE.json"
A3B_RECON = REPO / "receipts" / "ascent-2026-08-18" / "A3B_RECON.json"
GROK_TASKS = Path.home() / ".claude-grok" / "tasks"
GROK_WORKTREES = Path.home() / ".claude-grok" / "worktrees"
REVIEW_QUEUE = ODYSSEY / "REVIEW_QUEUE.jsonl"
RECLAIM = TOOLS / "reclaim_safe.sh"
AUTO_DIR = ODYSSEY / "contracts" / "auto"
RUN_LOG = ODYSSEY / "RUN_LOG.jsonl"
DOWNLOADS = ODYSSEY / "downloads"
GROK_BIN = Path.home() / ".claude-grok" / "bin" / "grok-run"
LINT_JS = Path.home() / ".claude-grok" / "v2" / "lint.mjs"
NODE_BIN = Path("/opt/homebrew/bin/node")
HAWKING_REPO = Path("/Users/scammermike/Downloads/hawking")
PREFERRED_PY = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
HF_BIN = Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/hf")
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"

DISK_FLOOR_GIB = 15.0
DISK_WARN_GIB = 40.0
DISK_RUN_GIB = 45.0
# Disk-aware model-lane throttle: never launch a NEW model lane that would push
# free disk below this floor, budgeting ~one 4-bit body per new gravity spec.
DISK_MIN_FLOOR = 10.0
TYPICAL_SPEC_GIB = 12.0
# Evict cold spec caches when free disk drops below the trigger, up to the
# target. Only-when-low + bounded target keeps ticks fast (no per-tick full sweep).
DISK_EVICT_TRIGGER = 45.0
DISK_EVICT_TARGET = 65.0
# Memgate (swap<=30 GiB) is the real multi-model bound. This cap is only a
# safety rail so a stuck driver cannot spawn unbounded grok-run processes.
HARD_LANE_CAP = 14
DEFAULT_MAX_LANES = 2
SCHEMA = "hawking.odyssey.controller.v1"
RUN_LOG_SCHEMA = "hawking.odyssey.run_log.v1"
HARVEST_SCHEMA = "hawking.odyssey.harvest.v2"
COMPLETION_SCHEMA = "hawking.odyssey.completions.v1"
SEAL_SCHEMA = "hawking.odyssey.patient_seal.v1"
CYCLE_SCHEMA = "hawking.odyssey.cycle.v1"
ACQUIRE_SCHEMA = "hawking.odyssey.acquire.v1"

PHASES = (
    "INGEST", "BASELINE", "CENSUS", "ROUTEMAP", "SENSITIVITY",
    "GRAVITY", "NX", "KERNEL", "DOCTOR", "PACKET", "TRANSFER", "SEAL",
    "SEALED",
)
PHASE_INDEX = {name: i for i, name in enumerate(PHASES)}
TRANSFER_REF = {"O006": "O005"}
NOVELTY_TEMPLATES = tuple(f"novelty-{lane}" for lane in novelty.LANES)
TEMPLATES = (
    "external-science-moe",
    "external-science-dense",
    "route-map",
    "sensitivity-map",
    "transfer-control",
    "gravity-moe",
    "gravity-dense",
    "gravity-hybrid",
    "gravity-aggressive-moe",
    "gravity-aggressive-dense",
    "gravity-aggressive-hybrid",
    "nx-gather-moe",
    "nx-state-hybrid",
    "nx-dense",
) + NOVELTY_TEMPLATES
# Known runner invocations. No novelty, no code change — the orchestrator
# execs tools/odyssey_patient_runner.py itself (S004 §52/§55).
DETERMINISTIC_TEMPLATES = frozenset({
    "external-science-moe",
    "external-science-dense",
    "route-map",
    "sensitivity-map",
    "gravity-moe",
    "gravity-dense",
    "gravity-hybrid",
    "gravity-aggressive-moe",
    "gravity-aggressive-dense",
    "gravity-aggressive-hybrid",
    "nx-gather-moe",
    "nx-state-hybrid",
    "nx-dense",
    "transfer-control",
})
DEFAULT_LANE_TIMEOUT_MIN = 30
LANES_DIR = ODYSSEY / "lanes"
# Every current template is DATA-PRODUCING: it RUNS the existing runner and
# delivers a receipt + packet fields. Incidental tools/*.py diffs are noise.
# Empty: no template claims the runner, so lanes are parallel-safe.
# Kept so evaluate_gates still has a serial hook if a future template needs it.
CODE_EDIT_TEMPLATES = frozenset()
RUNNER_WRITE_TEMPLATES = frozenset()
DATA_PRODUCING_TEMPLATES = frozenset(TEMPLATES)
RUNNER_REL = "tools/odyssey_patient_runner.py"
RUNNER_DO_NOT_EDIT = (
    "The runner ALREADY has this mode — RUN it, do NOT modify "
    "tools/odyssey_patient_runner.py."
)
TRANSFER_REL = "workspace/campaign/odyssey/TRANSFER_MATRIX.json"
TEMPLATE_MECHANISM = {
    "external-science-moe": "external-science",
    "external-science-dense": "external-science",
    "route-map": "route-map",
    "sensitivity-map": "sensitivity-map",
    "transfer-control": "transfer-control",
    "gravity-moe": "gravity-moe",
    "gravity-dense": "gravity-dense",
    "gravity-hybrid": "gravity-hybrid",
    "gravity-aggressive-moe": "gravity-aggressive-moe",
    "gravity-aggressive-dense": "gravity-aggressive-dense",
    "gravity-aggressive-hybrid": "gravity-aggressive-hybrid",
    "nx-gather-moe": "nx-gather-moe",
    "nx-state-hybrid": "nx-state-hybrid",
    "nx-dense": "nx-dense",
    **{t: t for t in NOVELTY_TEMPLATES},
}
# Agreed runner flags (parallel lane owns the runner; these names are the contract).
GRAVITY_SPEC = {
    "gravity-moe": "q3-g32-experts",
    "gravity-dense": "q4-g64",
    "gravity-hybrid": "q4-g64-attn-mlp",
    "gravity-aggressive-moe": "q2-g32-experts",
    "gravity-aggressive-dense": "q2-g64",
    "gravity-aggressive-hybrid": "q2-g64-attn-mlp",
}
NX_FLAG = {
    "nx-gather-moe": "--nx-gather",
    "nx-state-hybrid": "--nx-state",
    "nx-dense": "--nx-dense",
}
# Aggressive descent LADDER: many gravity specs per patient beyond the single
# canonical aggressive probe, so the deterministic well is deep enough to keep
# the box working and to MAP the sub-frontier (S004 anti-complacency: don't stop
# at the first aggressive point). Each rung is its own per-spec mechanism
# (independent completion) and its own receipt; rungs serialize per patient on
# the packet lock. All specs are runner-grammar-valid. The runner reuses the
# per-spec quant cache, so a repeated rung is seconds; a new one is a real run.
# Deep ladders: the full (bits x group x organ-target x mixed) grid the runner
# grammar accepts, ordered aggressive-first (q2 -> mixed -> q3 -> q4) so the
# interesting sub-2-bit / mixed science and frontier boundary run before the
# conventional anchors. ~30 rungs/patient maps the BPW-vs-Doctor surface finely
# and keeps the box saturated. Cold spec caches are evicted; disk is throttled.
# mlx quantize supports group sizes 32/64/128 ONLY (g16 is rejected at runtime).
GRAVITY_LADDER = {
    "moe": [
        "q2-g32-experts", "q2-g64-experts", "q2-g128-experts",
        "q2-g32", "q2-g64", "q2-g128",
        "mixed-q2q3-experts", "mixed-q2q4-experts", "mixed-q2q3", "mixed-q2q4",
        "q3-g32-experts", "q3-g64-experts", "q3-g128-experts",
        "q3-g32", "q3-g64", "q3-g128", "mixed-q3q4-experts", "mixed-q3q4",
        "q4-g32-experts", "q4-g64-experts", "q4-g128-experts",
        "q4-g32", "q4-g64", "q4-g128",
    ],
    "dense": [
        "q2-g32", "q2-g64", "q2-g128",
        "q2-g32-attn-mlp", "q2-g64-attn-mlp", "q2-g128-attn-mlp",
        "mixed-q2q3", "mixed-q2q4", "mixed-q2q3-attn-mlp", "mixed-q2q4-attn-mlp",
        "q3-g32", "q3-g64", "q3-g128",
        "q3-g32-attn-mlp", "q3-g64-attn-mlp", "q3-g128-attn-mlp",
        "mixed-q3q4", "mixed-q3q4-attn-mlp",
        "q4-g32", "q4-g64", "q4-g128",
        "q4-g32-attn-mlp", "q4-g64-attn-mlp", "q4-g128-attn-mlp",
    ],
    "hybrid": [
        "q2-g32", "q2-g64", "q2-g128",
        "q2-g32-attn-mlp", "q2-g64-attn-mlp", "q2-g128-attn-mlp",
        "mixed-q2q3", "mixed-q2q4", "mixed-q2q3-attn-mlp", "mixed-q2q4-attn-mlp",
        "q3-g32", "q3-g64", "q3-g128",
        "q3-g32-attn-mlp", "q3-g64-attn-mlp", "q3-g128-attn-mlp",
        "mixed-q3q4", "mixed-q3q4-attn-mlp",
        "q4-g32", "q4-g64", "q4-g128",
        "q4-g32-attn-mlp", "q4-g64-attn-mlp", "q4-g128-attn-mlp",
    ],
}
GRAVITY_LADDER_TEMPLATE = {
    "moe": "gravity-aggressive-moe",
    "dense": "gravity-aggressive-dense",
    "hybrid": "gravity-aggressive-hybrid",
}
GRAVITY_RECEIPT = {
    "gravity-moe": "{oxx}_GRAVITY_q3-g32-experts.json",
    "gravity-dense": "{oxx}_GRAVITY_q4-g64.json",
    "gravity-hybrid": "{oxx}_GRAVITY_q4-g64-attn-mlp.json",
    "gravity-aggressive-moe": "{oxx}_GRAVITY_q2-g32-experts.json",
    "gravity-aggressive-dense": "{oxx}_GRAVITY_q2-g64.json",
    "gravity-aggressive-hybrid": "{oxx}_GRAVITY_q2-g64-attn-mlp.json",
}
NX_RECEIPT = {
    "nx-gather-moe": "{oxx}_NX_gather.json",
    "nx-state-hybrid": "{oxx}_NX_state.json",
    "nx-dense": "{oxx}_NX_dense.json",
}
# Mechanism/template → expected receipts/odyssey-i/<OXX>_*.json leaf.
# Harvest classifies by this, not by whether the lane also touched tools/*.py.
RECEIPT_PATTERN = {
    "external-science": "{oxx}_EXTERNAL.json",
    "external-science-moe": "{oxx}_EXTERNAL.json",
    "external-science-dense": "{oxx}_EXTERNAL.json",
    "route-map": "{oxx}_EXTERNAL.json",
    "ssm-accounting": "{oxx}_EXTERNAL.json",
    "sensitivity-map": "{oxx}_SENSITIVITY.json",
    "transfer-control": "{oxx}_TRANSFER.json",
    **GRAVITY_RECEIPT,
    **NX_RECEIPT,
    **{f"novelty-{lane}": "{oxx}_NOVELTY_" + lane + ".json" for lane in novelty.LANES},
    "patient-sealed": "{oxx}_PATIENT_SEAL.json",
}
# Bounded required set per class (steer S002 — do not over-deepen).
REQUIRED_MOE = (
    "external-science", "route-map", "sensitivity-map",
    "gravity-moe", "gravity-aggressive-moe", "nx-gather-moe",
)
REQUIRED_DENSE = (
    "external-science", "sensitivity-map", "gravity-dense",
    "gravity-aggressive-dense", "nx-dense",
)
REQUIRED_HYBRID = (
    "external-science", "ssm-accounting", "sensitivity-map",
    "gravity-hybrid", "gravity-aggressive-hybrid", "nx-state-hybrid",
)
AGGRESSIVE_GRAVITY_TEMPLATES = frozenset({
    "gravity-aggressive-moe", "gravity-aggressive-dense", "gravity-aggressive-hybrid",
})
CONVENTIONAL_GRAVITY_TEMPLATES = frozenset({
    "gravity-moe", "gravity-dense", "gravity-hybrid",
})
POLICY_PATH = ODYSSEY / "ODYSSEY_POLICY.json"
MANIFEST_PATH = ODYSSEY / "ODYSSEY_MANIFEST.json"
_POLICY_CACHE = None
_MANIFEST_CACHE = None
_MANIFEST_BY_OXX = None
SENSITIVITY_SKIP_KEYS = frozenset({
    "baseline", "_label", "_evidence", "treatments", "summary",
})
RETIRE_TERMINAL = frozenset({"VERIFIED", "REFUTED"})
# Conservative download estimates (GiB). Overridden by census/hf used_storage.
PATIENT_EST_GIB = {
    "O000": 3.0, "O001": 16.0, "O002": 9.0, "O003": 32.0, "O004": 48.0,
    "O005": 61.0, "O006": 62.0, "O007": 100.0, "O008": 55.0, "O009": 145.0,
    "O010": 220.0, "O011": 0.0, "O012": 720.0, "O013": 80.0,
}
TERMINAL_COMPLETION = frozenset({
    "VERIFIED", "REFUTED", "SUPERSEDED", "ARCHIVED",
})
# Explicit receipt → completion map. Not a glob: O005_SENSITIVITY.json may
# exist on disk without being sealed science (stays PENDING).
COMPLETION_BACKFILL = (
    ("O001", "external-science", "O001_EXTERNAL.json"),
    ("O001", "ssm-accounting", "O001_EXTERNAL.json"),
    ("O001", "sensitivity-map", "O001_SENSITIVITY.json"),
    ("O003", "external-science", "O003_EXTERNAL.json"),
    ("O003", "route-map", "O003_EXTERNAL.json"),
    ("O005", "external-science", "O005_EXTERNAL.json"),
    ("O005", "route-map", "O005_EXTERNAL.json"),
    ("O005", "gravity-moe", "O005_GRAVITY_q3-g32-experts.json"),
    ("O005", "nx-gather-moe", "O005_NX_gather.json"),
    ("O006", "external-science", "O006_EXTERNAL.json"),
    ("O006", "route-map", "O006_EXTERNAL.json"),
    ("O006", "transfer-control", "O006_TRANSFER.json"),
)

STATES = (
    "READY", "RUNNING", "BLOCKED", "LANDED",
    "VERIFYING", "VERIFIED", "REVIEW", "REFUTED", "ARCHIVED",
    "RETIRED", "ACQUIRING",
)
EVIDENCE = (
    "VERIFIED", "MEASURED", "DERIVED", "INFERRED",
    "HYPOTHESIS", "SPECULATIVE", "REFUTED", "STALE", "UNKNOWN",
)
PACKET_SECTIONS = (
    "identity", "architecture", "doctor", "tabula", "routing",
    "representation", "execution", "gravity", "nx", "transfer", "next",
)
MOE_PATIENTS = {
    "O003", "O005", "O006", "O007", "O008",
    "O010", "O011", "O012", "O013",
}

# §22 high-value kinds get higher `info`. Costs are coarse units, not money.
SEED_WORK = [
    {"id": "A1", "oxx": "O005", "title": "route/state map (instrument router)",
     "status": "READY", "info": 10, "wall_cost": 2, "gpu_cost": 1, "opus_cost": 0,
     "kind": "router-sensitivity"},
    {"id": "A2", "oxx": "O005", "title": "baseline TPS via external runtime",
     "status": "READY", "info": 4, "wall_cost": 1, "gpu_cost": 1, "opus_cost": 0,
     "kind": "architecture-first"},
    {"id": "A3", "oxx": "O005", "title": "per-organ/per-expert sensitivity map",
     "status": "READY", "info": 10, "wall_cost": 2, "gpu_cost": 1, "opus_cost": 0,
     "kind": "representation-discriminator"},
    {"id": "A5", "oxx": "O001", "title": "SSM organ bucket + state-vs-KV",
     "status": "READY", "info": 8, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0,
     "kind": "architecture-first"},
    {"id": "A6", "oxx": "O000", "title": "HF-token unblock O000/O002/O004",
     "status": "BLOCKED", "info": 5, "wall_cost": 0, "gpu_cost": 0, "opus_cost": 0,
     "kind": "acquisition"},
    {"id": "ACQ-O003", "oxx": "O003", "title": "acquire Kimi-VL-A3B",
     "status": "READY", "info": 3, "wall_cost": 3, "gpu_cost": 0, "opus_cost": 0,
     "kind": "acquisition"},
    {"id": "ACQ-O006", "oxx": "O006", "title": "acquire Qwen3-VL sibling (transfer ctrl)",
     "status": "READY", "info": 4, "wall_cost": 3, "gpu_cost": 0, "opus_cost": 0,
     "kind": "acquisition"},
    {"id": "ACQ-O007", "oxx": "O007", "title": "acquire Kimi-Linear-48B-A3B",
     "status": "READY", "info": 3, "wall_cost": 4, "gpu_cost": 0, "opus_cost": 0,
     "kind": "acquisition"},
    {"id": "ACQ-O008", "oxx": "O008", "title": "acquire Jamba-Mini-1.5",
     "status": "READY", "info": 3, "wall_cost": 3, "gpu_cost": 0, "opus_cost": 0,
     "kind": "acquisition"},
    {"id": "ACQ-O009", "oxx": "O009", "title": "acquire Qwen2.5-72B",
     "status": "READY", "info": 2, "wall_cost": 5, "gpu_cost": 0, "opus_cost": 0,
     "kind": "acquisition"},
    {"id": "ACQ-O010", "oxx": "O010", "title": "acquire GLM-4.5-Air",
     "status": "READY", "info": 4, "wall_cost": 5, "gpu_cost": 0, "opus_cost": 0,
     "kind": "acquisition"},
    {"id": "ACQ-O011", "oxx": "O011", "title": "DSV4F legacy replay from receipts",
     "status": "READY", "info": 6, "wall_cost": 2, "gpu_cost": 1, "opus_cost": 0,
     "kind": "false-win-discovery"},
    {"id": "ACQ-O012", "oxx": "O012", "title": "acquire GLM-4.5 full (partial-residency)",
     "status": "READY", "info": 2, "wall_cost": 6, "gpu_cost": 0, "opus_cost": 0,
     "kind": "acquisition"},
    {"id": "ACQ-O013", "oxx": "O013", "title": "acquire Kimi-K3 streamed capstone",
     "status": "READY", "info": 2, "wall_cost": 8, "gpu_cost": 0, "opus_cost": 0,
     "kind": "acquisition"},
]

ROW_RE = re.compile(
    r"^\|\s*(O\d{3})\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|"
)
OXX_RE = re.compile(r"\bO(\d{3})\b", re.I)
SLUG_OXX_RE = re.compile(r"odyssey-o(\d{3})", re.I)
JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)
COMPLETION_RE = re.compile(
    r"(?:\*\*Completion report\*\*|#{1,3}\s+Completion report)\s*(.*)",
    re.S | re.I,
)
RESULT_RE = re.compile(r"^RESULT\s*:", re.M)
FALLBACK_PATIENTS = [
    ("O000", "gemma-3-1b-it", "tiny dense (compiler lab)",
     "google/gemma-3-1b-it", "HF-gated", "BLOCKED-auth", "—"),
    ("O001", "Falcon-H1-7B-Instruct", "small hybrid attn+Mamba",
     "tiiuae/Falcon-H1-7B-Instruct", "open", "on-disk", "CENSUS✓"),
    ("O002", "gemma-3-4b-it", "small dense multimodal",
     "google/gemma-3-4b-it", "HF-gated", "BLOCKED-auth", "—"),
    ("O003", "Kimi-VL-A3B-Instruct", "small multimodal MoE",
     "moonshotai/Kimi-VL-A3B-Instruct", "open", "queued", "—"),
    ("O004", "Mistral-Small-3.1-24B", "medium dense multimodal",
     "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "HF-gated", "BLOCKED-auth", "—"),
    ("O005", "Qwen3-30B-A3B", "small-active MoE",
     "Qwen/Qwen3-30B-A3B", "open", "on-disk", "CENSUS✓"),
    ("O006", "Qwen3-VL-30B-A3B", "multimodal MoE sibling",
     "Qwen/Qwen3-VL-30B-A3B-Instruct", "open", "queued (transfer ctrl)", "—"),
    ("O007", "Kimi-Linear-48B-A3B", "linear-attn MoE (KDA+MLA)",
     "moonshotai/Kimi-Linear-48B-A3B-Instruct", "open", "queued", "—"),
    ("O008", "Jamba-Mini-1.5", "Mamba+attn MoE",
     "ai21labs/AI21-Jamba-Mini-1.5", "license?", "queued", "—"),
    ("O009", "Qwen2.5-72B-Instruct", "large dense",
     "Qwen/Qwen2.5-72B-Instruct", "open", "queued (large)", "—"),
    ("O010", "GLM-4.5-Air", "106B/12B MoE",
     "zai-org/GLM-4.5-Air", "open", "queued (~220GB)", "—"),
    ("O011", "DSV4F", "mandatory legacy replay",
     "reconstruct from receipts", "local", "queued (control)", "—"),
    ("O012", "GLM-4.5 full", "355B/32B very-large MoE",
     "zai-org/GLM-4.5", "open", "queued (partial-residency)", "—"),
    ("O013", "Kimi-K3", "frontier 2.8T/104B streamed",
     "moonshotai/Kimi-K3", "open", "queued (streamed capstone)", "—"),
]


def strip_md(s: str) -> str:
    return (s or "").replace("**", "").strip()


def evidence_class(label) -> str | None:
    if label is None:
        return None
    text = str(label)
    for ev in EVIDENCE:
        if ev in text:
            return ev
    return None


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def load_odyssey_policy(path: Path | None = None) -> dict:
    """Machine-readable Odyssey-I policy. Orchestrator consumes this; do not fork thresholds."""
    global _POLICY_CACHE
    dest = Path(path) if path else POLICY_PATH
    if _POLICY_CACHE is not None and path is None:
        return _POLICY_CACHE
    if not dest.is_file():
        doc = {}
    else:
        doc = read_json(dest)
    if path is None:
        _POLICY_CACHE = doc
    return doc


def load_odyssey_manifest(path: Path | None = None) -> list:
    """Predeclared patient ladder (canonical_source / targets / search_class)."""
    global _MANIFEST_CACHE, _MANIFEST_BY_OXX
    dest = Path(path) if path else MANIFEST_PATH
    if _MANIFEST_CACHE is not None and path is None:
        return _MANIFEST_CACHE
    rows: list = []
    if dest.is_file():
        try:
            data = read_json(dest)
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, list):
            rows = [e for e in data if isinstance(e, dict)]
        elif isinstance(data, dict):
            for key in ("patients", "entries", "manifest"):
                if isinstance(data.get(key), list):
                    rows = [e for e in data[key] if isinstance(e, dict)]
                    break
    if path is None:
        _MANIFEST_CACHE = rows
        _MANIFEST_BY_OXX = {
            str(e.get("oxx") or ""): e for e in rows if e.get("oxx")
        }
    return rows


def manifest_entry(oxx: str) -> dict:
    oxx = norm_oxx(oxx) if oxx else ""
    if not oxx:
        return {}
    load_odyssey_manifest()
    by = _MANIFEST_BY_OXX or {}
    return dict(by.get(oxx) or {})


def overlay_manifest(meta: dict | None, oxx: str | None = None) -> dict:
    """Copy of a patient record with manifest source/targets overlaid."""
    rec = dict(meta or {})
    oid = oxx or rec.get("oxx") or ""
    man = manifest_entry(oid)
    if not man:
        return rec
    rec["canonical_source"] = man.get("canonical_source") or rec.get("canonical_source")
    if man.get("canonical_source"):
        rec["source"] = man["canonical_source"]
    if man.get("gated") is not None:
        rec["gated"] = bool(man.get("gated"))
    if man.get("gated_reason"):
        rec["gated_reason"] = man["gated_reason"]
    for key in (
        "est_source_gib", "est_4bit_gib", "doctor_bar",
        "stored_bpw_pressure", "active_bpw_pressure", "tps_pressure_rel",
        "search_class", "kernel_effort", "info_budget", "arch_objective",
        "reference_sibling", "canonical_revision", "source_precision",
        "reopen_if",
    ):
        if man.get(key) is not None:
            rec[key] = man[key]
    return rec


def gravity_spec_meta(spec: str) -> dict:
    table = (load_odyssey_policy().get("gravity_specs") or {})
    meta = table.get(spec) or {}
    return dict(meta)


def parse_gravity_grammar(spec: str) -> dict | None:
    """Parse a runner --gravity spec. Prefers candgen; falls back to the published grammar."""
    raw = str(spec or "").strip()
    if not raw:
        return None
    try:
        return candgen.parse_spec(raw)
    except (ValueError, TypeError):
        pass
    base, *rest = raw.split("+", 1)
    suffixes = "+" + rest[0] if rest else ""
    has_corr = bool(re.search(r"\+(correction|c\d)", suffixes, re.I))
    m = re.fullmatch(r"q(\d+)-g(\d+)(?:-(experts|attn-mlp))?", base, re.I)
    if m:
        return {
            "form": "uniform",
            "bits": int(m.group(1)),
            "group": int(m.group(2)),
            "target": (m.group(3) or "").lower() or None,
            "correction_budget": 0.02 if has_corr else 0.0,
            "correction_token": has_corr,
        }
    m = re.fullmatch(r"mixed-q(\d+)q(\d+)(?:-(experts|attn-mlp))?", base, re.I)
    if m:
        return {
            "form": "mixed",
            "mixed_lo": int(m.group(1)),
            "mixed_hi": int(m.group(2)),
            "bits": int(m.group(1)),
            "group": 32,
            "target": (m.group(3) or "").lower() or None,
            "correction_budget": 0.02 if has_corr else 0.0,
            "correction_token": has_corr,
        }
    if base.lower().startswith("tiers-") or "tier" in raw.lower():
        return {
            "form": "tiers",
            "bits": 1,
            "correction_budget": 0.02 if has_corr else 0.0,
            "correction_token": has_corr,
        }
    m = re.fullmatch(r"scale-joint-q(\d+)-g(\d+)(?:-(experts|attn-mlp))?", base, re.I)
    if m:
        return {
            "form": "scale_joint",
            "bits": int(m.group(1)),
            "group": int(m.group(2)),
            "target": (m.group(3) or "").lower() or None,
            "correction_budget": 0.02 if has_corr else 0.0,
            "correction_token": has_corr,
        }
    return None


def classify_gravity_spec(spec: str) -> dict:
    """Grammar-based spec → candidate_class. policy.gravity_specs is an exact-key override.

    q>=3 affine → CONVENTIONAL_ANCHOR; q<=2 affine → AGGRESSIVE_QUANT;
    mixed / +correction / tiers / scale-joint → STRUCTURAL_GRAVITY.
    No per-candidate model reasoning.
    """
    spec = str(spec or "")
    meta = gravity_spec_meta(spec)
    classes = load_odyssey_policy().get("candidate_classes") or []
    parsed = parse_gravity_grammar(spec)
    klass = None
    conv = None
    bits = None
    mech = None
    if meta.get("candidate_class"):
        klass = meta["candidate_class"]
        conv = meta.get("conventionality")
        bits = meta.get("nominal_bits")
        mech = meta.get("mechanism")
    elif parsed:
        form = parsed.get("form")
        corr = bool(parsed.get("correction_token")) or float(
            parsed.get("correction_budget") or 0
        ) > 0
        bits = parsed.get("bits")
        if form in {"mixed", "tiers", "scale_joint"} or corr:
            klass = "STRUCTURAL_GRAVITY"
            if form == "mixed":
                mech = "per-organ/layer/expert/sensitivity-driven bit allocation"
            elif form == "tiers":
                mech = "matryoshka-tiers"
            elif form == "scale_joint":
                mech = "scale-codec-joint"
            else:
                mech = "base+correction"
        elif bits is not None and int(bits) <= 2:
            klass = "AGGRESSIVE_QUANT"
            mech = "affine-quant"
        elif bits is not None and int(bits) >= 3:
            klass = "CONVENTIONAL_ANCHOR"
            mech = "affine-quant"
    if not klass:
        klass = "BASELINE"
    if classes and klass not in classes:
        klass = "BASELINE"
    if not conv:
        conv = "conventional" if klass == "CONVENTIONAL_ANCHOR" else "nonconventional"
    if bits is None and parsed and parsed.get("bits") is not None:
        bits = parsed.get("bits")
    return {
        "candidate_class": klass,
        "conventionality": conv,
        "mechanism": mech or meta.get("mechanism"),
        "nominal_bits": bits if bits is not None else meta.get("nominal_bits"),
        "spec": spec,
    }


def classify_gravity_receipt(rec: dict | None) -> dict:
    rec = rec if isinstance(rec, dict) else {}
    spec = rec.get("spec") or ""
    tagged = classify_gravity_spec(spec) if spec else {
        "candidate_class": rec.get("candidate_class") or "BASELINE",
        "conventionality": rec.get("conventionality") or "conventional",
        "mechanism": None,
        "nominal_bits": rec.get("nominal_bits"),
        "spec": spec,
    }
    if rec.get("candidate_class"):
        tagged["candidate_class"] = rec["candidate_class"]
    if rec.get("conventionality"):
        tagged["conventionality"] = rec["conventionality"]
    if rec.get("nominal_bits") is not None:
        tagged["nominal_bits"] = rec.get("nominal_bits")
    return tagged


def gravity_pass_threshold() -> int:
    gv = load_odyssey_policy().get("gravity_verdict") or {}
    return int(gv.get("candidate_pass_min_delta_hits", -1))


def _as_int(val, default=0) -> int:
    if val is None or isinstance(val, bool):
        return default
    if isinstance(val, (int, float)):
        return int(val)
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def organ_sensitivity_delta(entry) -> int:
    """Worse (more negative) delta = more sensitive. Prefers round8, then zero."""
    if not isinstance(entry, dict):
        return _as_int(entry, 0)
    r = entry.get("round8")
    if isinstance(r, dict) and r.get("delta_hits") is not None:
        return _as_int(r.get("delta_hits"), 0)
    z = entry.get("zero")
    if isinstance(z, dict) and z.get("delta_hits") is not None:
        return _as_int(z.get("delta_hits"), 0)
    if entry.get("delta_hits") is not None:
        return _as_int(entry.get("delta_hits"), 0)
    return 0


def organ_sensitivity_rank(per_organ_sensitivity) -> list[dict]:
    """Rank organs by MEASURED sensitivity delta (more negative first)."""
    pos = per_organ_sensitivity if isinstance(per_organ_sensitivity, dict) else {}
    rows = []
    for name, entry in pos.items():
        if not name or name in SENSITIVITY_SKIP_KEYS:
            continue
        if not isinstance(entry, dict) and not isinstance(entry, (int, float)):
            continue
        delta = organ_sensitivity_delta(entry)
        treatment = "round8"
        if isinstance(entry, dict):
            if not (isinstance(entry.get("round8"), dict) and entry["round8"].get("delta_hits") is not None):
                treatment = "zero" if isinstance(entry.get("zero"), dict) else "delta_hits"
        rows.append({
            "organ": str(name),
            "sensitivity_delta": delta,
            "treatment": treatment,
        })
    rows.sort(key=lambda r: (r["sensitivity_delta"], r["organ"]))
    return rows


def select_protected_components(spec: str, per_organ_sensitivity=None) -> list[str]:
    """Sensitivity-driven mix: low-bit base, promote the worst organs. Deterministic."""
    parsed = parse_gravity_grammar(spec or "")
    is_mixed = (parsed or {}).get("form") == "mixed" or str(spec or "").startswith("mixed-")
    if not is_mixed:
        return []
    ranked = organ_sensitivity_rank(per_organ_sensitivity)
    usable = [r for r in ranked if r["organ"] not in {"norm", "ssm"}]
    if not usable:
        return ["attn", "router"]
    dropped = [r["organ"] for r in usable if r["sensitivity_delta"] < 0]
    if dropped:
        return dropped
    return [usable[0]["organ"]]


def localize_gravity_failure(delta_hits, per_organ_sensitivity=None,
                             threshold: int | None = None) -> dict | None:
    """Cheap failure localization: rank organs by sensitivity delta. No global retreat."""
    thresh = gravity_pass_threshold() if threshold is None else int(threshold)
    if delta_hits is None:
        return None
    try:
        dh = int(delta_hits)
    except (TypeError, ValueError):
        return None
    if dh >= thresh:
        return None
    ranked = organ_sensitivity_rank(per_organ_sensitivity)
    top = ranked[0]["organ"] if ranked else "unknown"
    anti = (load_odyssey_policy().get("failure_policy") or {}).get("anti_retreat") or (
        "global precision retreat is last resort; first protect the sensitive component"
    )
    return {
        "failed": True,
        "delta_hits": dh,
        "threshold": thresh,
        "ranked_organs": ranked,
        "most_likely_component": top,
        "targeted_repair": f"protect {top} (do not globally raise bits)",
        "anti_retreat": anti,
        "_evidence": "DERIVED (rank organs by MEASURED sensitivity delta)",
    }


def _policy_mech_list(key: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    gate = load_odyssey_policy().get("retirement_gate") or {}
    vals = gate.get(key)
    if isinstance(vals, list) and vals:
        return tuple(str(x) for x in vals)
    return fallback


def aggressive_mechanism_ids() -> tuple[str, ...]:
    return _policy_mech_list(
        "aggressive_probe_mechanisms",
        tuple(sorted(AGGRESSIVE_GRAVITY_TEMPLATES)),
    )


def conventional_anchor_mechanism_ids() -> tuple[str, ...]:
    return _policy_mech_list(
        "conventional_anchor_mechanisms",
        tuple(sorted(CONVENTIONAL_GRAVITY_TEMPLATES)),
    )


def is_aggressive_mechanism(mechanism_id: str) -> bool:
    m = mechanism_id or ""
    if m in aggressive_mechanism_ids() or m in AGGRESSIVE_GRAVITY_TEMPLATES:
        return True
    if m.startswith("gravity-aggressive-"):
        return True
    # ladder rungs: a per-spec gravity mechanism at <=2 bits or a mixed codec is
    # an aggressive probe (satisfies anti-complacency the same as the canonical).
    return m.startswith("gravity-") and (
        "q2" in m or "q1" in m or "mixed" in m
    )


def conventional_anchor_exists(oxx: str, entries: list | None = None) -> bool:
    pool = entries if entries is not None else _completions_entries(None)
    for mech in conventional_anchor_mechanism_ids():
        ent = current_completion(oxx, mech, pool)
        if not ent or ent.get("status") not in RETIRE_TERMINAL:
            continue
        klass = ent.get("candidate_class")
        if klass in (None, "", "CONVENTIONAL_ANCHOR"):
            return True
    for ent in pool or []:
        if ent.get("patient_id") != oxx:
            continue
        if ent.get("status") not in RETIRE_TERMINAL:
            continue
        if ent.get("candidate_class") == "CONVENTIONAL_ANCHOR":
            return True
    return False


def aggressive_probe_attempted(oxx: str, entries: list | None = None) -> bool:
    """VERIFIED or REFUTED gravity-aggressive-* still counts as attempted."""
    pool = entries if entries is not None else _completions_entries(None)
    for mech in aggressive_mechanism_ids():
        ent = current_completion(oxx, mech, pool)
        if ent and ent.get("status") in RETIRE_TERMINAL:
            return True
    for ent in pool or []:
        if ent.get("patient_id") != oxx:
            continue
        if ent.get("status") not in RETIRE_TERMINAL:
            continue
        if is_aggressive_mechanism(ent.get("mechanism_id") or ""):
            return True
    return False


def has_low_information_value(oxx: str, receipt_dir: Path | None = None) -> bool:
    root = Path(receipt_dir) if receipt_dir else RECEIPT_DIR
    if not root.is_dir():
        return False
    for path in sorted(root.glob(f"{oxx}*.json")):
        try:
            rec = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(rec, dict):
            continue
        if rec.get("oxx") not in (None, oxx) and rec.get("patient_id") not in (None, oxx):
            continue
        schema = str(rec.get("schema") or "")
        verdict = str(rec.get("verdict") or rec.get("status") or "")
        if rec.get("low_information_value") is True:
            return True
        if "LOW_INFORMATION_VALUE" in schema or verdict == "LOW_INFORMATION_VALUE":
            return True
    return False


def cheap_credible_mechanisms_remain(oxx: str, entries: list | None = None) -> bool:
    """If the aggressive probe has not been attempted, cheap mechanisms remain."""
    return not aggressive_probe_attempted(oxx, entries)


def ladder_rungs_remain(oxx: str, entries: list | None = None) -> list[str]:
    """Descent-ladder rungs for this patient not yet terminal (VERIFIED/REFUTED).

    A REFUTED rung counts as descended (frontier located), so a rung that keeps
    failing does not block forever. The ladder is bounded, so this delays
    retirement by a bounded amount — the point is to descend the sub-frontier
    before sealing, not to keep a patient open indefinitely.
    """
    arch = patient_arch_kind(oxx)
    pool = entries if entries is not None else _completions_entries(None)
    ladder_tmpl = GRAVITY_LADDER_TEMPLATE.get(arch)
    # the canonical aggressive spec is sealed under the gravity-aggressive-*
    # mechanism (a required mechanism), NOT gravity-<spec>; skip it here or it
    # would show as forever-remaining and the patient could never retire.
    picked = pick_aggressive_spec(oxx, ladder_tmpl) if ladder_tmpl else None
    remaining = []
    for spec in GRAVITY_LADDER.get(arch, []):
        if spec == picked:
            continue
        mech = f"gravity-{spec}"
        if not mechanism_retire_done(oxx, mech, pool):
            remaining.append(mech)
    return remaining


def retirement_gate_reason(oxx: str, entries: list | None = None,
                           receipt_dir: Path | None = None) -> str | None:
    """DEFAULT REFUSE when policy.retirement_gate.default_refuse_if holds.

    Exception: explicit LOW_INFORMATION_VALUE receipt for this patient.
    """
    if has_low_information_value(oxx, receipt_dir=receipt_dir):
        return None
    conv = conventional_anchor_exists(oxx, entries)
    agg = aggressive_probe_attempted(oxx, entries)
    cheap = cheap_credible_mechanisms_remain(oxx, entries)
    if conv and (not agg) and cheap:
        return (
            "retirement_gate: conventional_anchor_exists AND "
            "aggressive_probe_attempted==false AND cheap_credible_mechanisms_remain"
        )
    # Anti-complacency: descend the full aggressive ladder before sealing. One
    # aggressive probe is not the frontier — keep the patient (and the box)
    # working through every rung until each is terminal.
    rem = ladder_rungs_remain(oxx, entries)
    if rem:
        return (
            f"retirement_gate: descent ladder not exhausted "
            f"({len(rem)} rungs remain: {', '.join(r.split('gravity-')[-1] for r in rem[:4])})"
        )
    return None


def count_retired_without_nonconventional_probe(
    state: dict | None = None, entries: list | None = None,
) -> int:
    st = state if state is not None else ensure_state()
    pool = entries if entries is not None else _completions_entries(None)
    n = 0
    for p in st.get("patients") or []:
        if p.get("state") != "RETIRED":
            continue
        oxx = p.get("oxx")
        if not oxx:
            continue
        if not aggressive_probe_attempted(oxx, pool):
            n += 1
    return n


def gravity_tags_from_receipt(path: Path | None) -> dict:
    """candidate_class / conventionality from a gravity receipt, else from spec."""
    if path is None or not Path(path).is_file():
        return {}
    try:
        rec = read_json(Path(path))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(rec, dict):
        return {}
    schema = str(rec.get("schema") or "")
    if schema and schema != "odyssey.patient.gravity.v1" and not rec.get("spec"):
        return {}
    if not rec.get("spec") and not rec.get("candidate_class"):
        return {}
    tagged = classify_gravity_receipt(rec)
    out = {}
    if tagged.get("candidate_class"):
        out["candidate_class"] = tagged["candidate_class"]
    if tagged.get("conventionality"):
        out["conventionality"] = tagged["conventionality"]
    return out


def apply_gravity_tags_to_packet(oxx: str, tags: dict, dest_root: Path | None = None) -> None:
    if not tags or not oxx:
        return
    root = Path(dest_root) if dest_root else REPO
    dest = root / packet_rel(oxx)
    if not dest.is_file():
        return
    try:
        pkt = read_json(dest)
    except (OSError, json.JSONDecodeError):
        return
    g = pkt.setdefault("gravity", {})
    if tags.get("candidate_class"):
        g["candidate_class"] = tags["candidate_class"]
    if tags.get("conventionality"):
        g["conventionality"] = tags["conventionality"]
    last = g.get("last")
    if isinstance(last, dict):
        if tags.get("candidate_class"):
            last["candidate_class"] = tags["candidate_class"]
        if tags.get("conventionality"):
            last["conventionality"] = tags["conventionality"]
    write_json(dest, pkt)


# ---------------------------------------------------------------------------
# completion index — workflow state (what science is sealed). Not a packet.
# Flow: experiment → receipt → verification → completion index → packet → scheduler.
# ---------------------------------------------------------------------------

def git_head(repo: Path | None = None) -> str:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo or REPO), capture_output=True, text=True,
    )
    return (r.stdout or "").strip() if r.returncode == 0 else ""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receipt_stamp(path: Path) -> str:
    """Historical timestamp for a receipt. Never wall-clock now."""
    try:
        rel = str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        rel = str(path)
    r = subprocess.run(
        ["git", "log", "-1", "--format=%cI", "--", rel],
        cwd=str(REPO), capture_output=True, text=True,
    )
    iso = (r.stdout or "").strip()
    if iso:
        if iso.endswith("+00:00"):
            iso = iso[:-6] + "Z"
        return iso
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def empty_completions() -> dict:
    return {
        "schema": COMPLETION_SCHEMA,
        "entries": [],
        "_evidence": "DERIVED (completion index)",
    }


def load_completions(path: Path | None = None) -> dict:
    dest = Path(path) if path else COMPLETIONS
    if not dest.is_file():
        return empty_completions()
    try:
        doc = read_json(dest)
    except (OSError, json.JSONDecodeError):
        return empty_completions()
    if isinstance(doc, list):
        return {
            "schema": COMPLETION_SCHEMA,
            "entries": doc,
            "_evidence": "DERIVED (completion index)",
        }
    if not isinstance(doc, dict):
        return empty_completions()
    doc.setdefault("schema", COMPLETION_SCHEMA)
    doc.setdefault("entries", [])
    return doc


def save_completions(doc: dict, path: Path | None = None) -> Path:
    dest = Path(path) if path else COMPLETIONS
    if isinstance(doc, list):
        doc = {
            "schema": COMPLETION_SCHEMA,
            "entries": doc,
            "_evidence": "DERIVED (completion index)",
        }
    write_json(dest, doc)
    return dest


def mechanism_for_template(template: str) -> str:
    if not template:
        return ""
    if template in TEMPLATE_MECHANISM:
        return TEMPLATE_MECHANISM[template]
    if template.startswith("external-science"):
        return "external-science"
    if template.startswith("gravity-") or template.startswith("nx-"):
        return template
    return template


def packet_rel(oxx: str) -> str:
    return f"workspace/campaign/odyssey/patients/{oxx}/ODYSSEY_PATIENT_{oxx}.json"


def receipt_filename(oxx: str, template_or_mech: str, spec: str | None = None) -> str | None:
    """Leaf name of the expected receipts/odyssey-i receipt for a template."""
    if not oxx or not template_or_mech:
        return None
    tmpl = template_or_mech
    if spec and (
        tmpl in GRAVITY_SPEC
        or tmpl.startswith("gravity-")
        or tmpl in GRAVITY_RECEIPT
    ):
        return f"{oxx}_GRAVITY_{spec}.json"
    pat = RECEIPT_PATTERN.get(tmpl)
    if not pat:
        pat = RECEIPT_PATTERN.get(mechanism_for_template(tmpl))
    if not pat:
        return None
    return pat.format(oxx=oxx)


def expected_receipt_rel(oxx: str, template: str, spec: str | None = None) -> str | None:
    name = receipt_filename(oxx, template, spec=spec)
    return f"receipts/odyssey-i/{name}" if name else None


def write_scope(ob: dict) -> dict:
    """Files this obligation may edit + exclusive resources. Honest per template.

    Data-producing templates RUN the existing runner; they do not claim it.
    write_set is {patient packet, that patient's receipts} only. The runner
    is claimed solely by RUNNER_WRITE_TEMPLATES (empty: no current template
    builds the runner). Different patients → disjoint → parallel.
    """
    oxx = ob.get("oxx") or ob.get("patient_id") or ""
    template = ob.get("template") or ""
    # Descent-ladder rungs write ONLY their per-spec receipt, not the shared
    # patient packet, so many rungs of the same patient run concurrently instead
    # of serializing on the packet lock. The packet is a derived human summary;
    # completions are receipt-driven and rebuild_completions recovers any packet
    # update lost to a concurrent write. This is what lets the box saturate.
    if ob.get("ladder"):
        rec = expected_receipt_rel(oxx, template, spec=ob.get("gravity_spec"))
        return {"write_set": [rec] if rec else [], "exclusive_resources": []}
    if "write_set" in ob:
        excl = list(ob.get("exclusive_resources") or [])
        if ob.get("timing") and "protected-timing" not in excl:
            excl.append("protected-timing")
        return {
            "write_set": list(ob.get("write_set") or []),
            "exclusive_resources": excl,
        }
    packet = packet_rel(oxx) if oxx else ""
    spec = ob.get("gravity_spec")
    rec = expected_receipt_rel(oxx, template, spec=spec)
    writes: list[str] = []
    if template in RUNNER_WRITE_TEMPLATES or template in CODE_EDIT_TEMPLATES:
        writes = [RUNNER_REL, packet, rec]
    elif template.startswith("novelty-"):
        writes = [rec] if rec else []
    elif template == "transfer-control":
        writes = [
            packet,
            f"receipts/odyssey-i/{oxx}_EXTERNAL.json",
            rec,
        ]
    elif template in {
        "external-science-moe", "external-science-dense", "route-map",
        "sensitivity-map",
    } or template in GRAVITY_SPEC or template in NX_FLAG or template in DATA_PRODUCING_TEMPLATES:
        writes = [packet, rec]
    else:
        writes = [packet] if packet else []
    writes = [p for p in writes if p]
    if template not in RUNNER_WRITE_TEMPLATES:
        writes = [p for p in writes if p != RUNNER_REL]
    excl = list(ob.get("exclusive_resources") or [])
    if ob.get("timing") and "protected-timing" not in excl:
        excl.append("protected-timing")
    return {
        "write_set": writes,
        "exclusive_resources": excl,
    }


def scopes_conflict(a: dict, b: dict) -> bool:
    wa = set(a.get("write_set") or [])
    wb = set(b.get("write_set") or [])
    if wa & wb:
        return True
    ea = set(a.get("exclusive_resources") or [])
    eb = set(b.get("exclusive_resources") or [])
    return bool(ea & eb)


def scope_conflict_reason(scope: dict, occupied: list[dict]) -> str | None:
    for other in occupied:
        if not scopes_conflict(scope, other):
            continue
        files = sorted(set(scope.get("write_set") or []) & set(other.get("write_set") or []))
        excl = sorted(
            set(scope.get("exclusive_resources") or [])
            & set(other.get("exclusive_resources") or [])
        )
        if files:
            return ",".join(files)
        if excl:
            return "exclusive:" + ",".join(excl)
        return "collision"
    return None


def reopen_if_satisfied(entry: dict, *, source_revision: str | None = None) -> bool:
    pred = entry.get("reopen_if") if entry else None
    if pred is None:
        return False
    text = str(pred).strip()
    if not text or text.lower() == "null":
        return False
    low = text.lower()
    if low in {"true", "1", "yes"}:
        return True
    if low in {"false", "0", "no"}:
        return False
    m = re.fullmatch(r"source_revision\s*(==|!=)\s*(.+)", text)
    if not m:
        return False
    op = m.group(1)
    rev = m.group(2).strip().strip("<>\"'")
    cur = source_revision if source_revision is not None else git_head()
    if op == "!=":
        return cur != rev
    return cur == rev


def current_completion(patient_id: str, mechanism_id: str,
                       entries: list | None = None) -> dict | None:
    if entries is None:
        entries = load_completions().get("entries") or []
    hits = [
        e for e in entries
        if e.get("patient_id") == patient_id and e.get("mechanism_id") == mechanism_id
    ]
    if not hits:
        return None
    superseded = {e.get("supersedes") for e in hits if e.get("supersedes")}
    live = [e for e in hits if e.get("obligation_id") not in superseded]
    pool = live or hits
    pool.sort(key=lambda e: str(e.get("completed_at") or ""))
    return pool[-1]


def science_is_done(patient_id: str, mechanism_id: str,
                    entries: list | None = None, *,
                    source_revision: str | None = None) -> bool:
    """Terminal completion blocks relaunch unless reopen_if is mechanically true."""
    entry = current_completion(patient_id, mechanism_id, entries)
    if not entry:
        return False
    if entry.get("status") not in TERMINAL_COMPLETION:
        return False
    if reopen_if_satisfied(entry, source_revision=source_revision):
        return False
    return True


def selection_verdict(patient_id: str, mechanism_id: str,
                      entries: list | None = None, *,
                      source_revision: str | None = None) -> str:
    if science_is_done(
        patient_id, mechanism_id, entries, source_revision=source_revision,
    ):
        return "REFUSE"
    return "LAUNCH"


def _completions_entries(completions) -> list:
    if completions is not None:
        if isinstance(completions, dict):
            return list(completions.get("entries") or [])
        return list(completions)
    if not COMPLETIONS.is_file():
        return list((rebuild_completions(persist=True).get("entries") or []))
    return list(load_completions().get("entries") or [])


def complete(*, obligation_id: str, patient_id: str, mechanism_id: str,
             status: str, completed_at: str,
             receipt_ref: str | None = None, receipt_sha256: str | None = None,
             source_revision: str | None = None, supersedes=None,
             reopen_if=None, index: dict | None = None,
             persist: bool = True, path: Path | None = None,
             candidate_class: str | None = None,
             conventionality: str | None = None) -> dict:
    """Write a terminal completion. `completed_at` is required (never Date.now)."""
    if not completed_at:
        raise ValueError("completed_at must be passed in (do not use wall clock here)")
    if status not in TERMINAL_COMPLETION:
        raise ValueError(
            f"status must be terminal {sorted(TERMINAL_COMPLETION)}, got {status}"
        )
    doc = index if index is not None else load_completions(path)
    entries = doc.setdefault("entries", [])
    row = {
        "obligation_id": obligation_id,
        "patient_id": patient_id,
        "mechanism_id": mechanism_id,
        "status": status,
        "receipt_ref": receipt_ref,
        "receipt_sha256": receipt_sha256,
        "source_revision": (
            source_revision if source_revision is not None else git_head()
        ),
        "completed_at": completed_at,
        "supersedes": supersedes,
        "reopen_if": reopen_if,
    }
    if candidate_class is not None:
        row["candidate_class"] = candidate_class
    if conventionality is not None:
        row["conventionality"] = conventionality
    match_i = None
    for i, existing in enumerate(entries):
        if existing.get("patient_id") != patient_id:
            continue
        if existing.get("mechanism_id") != mechanism_id:
            continue
        if existing.get("receipt_sha256") != receipt_sha256:
            continue
        if existing.get("status") != status:
            continue
        if obligation_id and existing.get("obligation_id") not in {None, obligation_id}:
            continue
        match_i = i
        break
    if match_i is not None:
        merged = dict(entries[match_i])
        merged.update(row)
        entries[match_i] = merged
        row = merged
    else:
        for i, existing in enumerate(entries):
            if existing.get("patient_id") != patient_id:
                continue
            if existing.get("mechanism_id") != mechanism_id:
                continue
            if existing.get("status") == "SUPERSEDED":
                continue
            if not row.get("supersedes"):
                row["supersedes"] = existing.get("obligation_id")
            old = dict(existing)
            old["status"] = "SUPERSEDED"
            entries[i] = old
        entries.append(row)
    doc["schema"] = COMPLETION_SCHEMA
    doc["_evidence"] = "DERIVED (completion index)"
    if persist:
        save_completions(doc, path)
    return row


def rebuild_completions(*, completed_at: str | None = None,
                        path: Path | None = None,
                        receipt_dir: Path | None = None,
                        persist: bool = True) -> dict:
    """Idempotent VERIFIED backfill from the explicit receipt map."""
    rec_dir = Path(receipt_dir) if receipt_dir else RECEIPT_DIR
    dest = Path(path) if path else COMPLETIONS
    doc = load_completions(dest) if dest.is_file() else empty_completions()
    have_backfill_files = any(
        (rec_dir / fname).is_file() for _, _, fname in COMPLETION_BACKFILL
    )
    backfill_keys = {(p, m) for p, m, _ in COMPLETION_BACKFILL}
    # Sparse worktrees omit receipts/; do not strip sealed backfill entries.
    if have_backfill_files:
        kept = [
            e for e in (doc.get("entries") or [])
            if (e.get("patient_id"), e.get("mechanism_id")) not in backfill_keys
        ]
        doc["entries"] = kept
    head = git_head()
    for patient_id, mechanism_id, fname in COMPLETION_BACKFILL:
        if not have_backfill_files:
            break
        rec_path = rec_dir / fname
        if not rec_path.is_file():
            continue
        stamp = completed_at or receipt_stamp(rec_path)
        try:
            rel = str(rec_path.resolve().relative_to(REPO.resolve()))
        except ValueError:
            rel = f"receipts/odyssey-i/{fname}"
        tags = gravity_tags_from_receipt(rec_path) if mechanism_id.startswith("gravity-") else {}
        complete(
            obligation_id=f"{patient_id}:{mechanism_id}",
            patient_id=patient_id,
            mechanism_id=mechanism_id,
            status="VERIFIED",
            completed_at=stamp,
            receipt_ref=rel.replace("\\", "/"),
            receipt_sha256=file_sha256(rec_path),
            source_revision=head,
            supersedes=None,
            reopen_if=None,
            index=doc,
            persist=False,
            path=dest,
            candidate_class=tags.get("candidate_class"),
            conventionality=tags.get("conventionality"),
        )
    rec_root = rec_dir
    for e in doc.get("entries") or []:
        if not str(e.get("mechanism_id") or "").startswith("gravity-"):
            continue
        if e.get("candidate_class"):
            continue
        ref = e.get("receipt_ref") or ""
        rec_path = None
        if ref:
            rec_path = (REPO / ref) if not Path(ref).is_absolute() else Path(ref)
            if not rec_path.is_file():
                rec_path = rec_root / Path(ref).name
        if rec_path is None or not rec_path.is_file():
            continue
        tags = gravity_tags_from_receipt(rec_path)
        if tags.get("candidate_class"):
            e["candidate_class"] = tags["candidate_class"]
        if tags.get("conventionality"):
            e["conventionality"] = tags["conventionality"]
        if persist and tags:
            apply_gravity_tags_to_packet(e.get("patient_id") or "", tags)
    doc["_evidence"] = "DERIVED (receipts/odyssey-i backfill)"
    if persist:
        save_completions(doc, dest)
    return doc


def parse_science_task(name: str) -> tuple[str, str] | None:
    m = re.match(
        r"^odyssey-o(\d{3})-("
        r"external-science-moe|external-science-dense|route-map|"
        r"sensitivity-map|transfer-control|"
        r"gravity-aggressive-moe|gravity-aggressive-dense|gravity-aggressive-hybrid|"
        r"gravity-moe|gravity-dense|gravity-hybrid|"
        r"nx-gather-moe|nx-state-hybrid|nx-dense|"
        r"novelty-representation|novelty-numerical|novelty-arch|novelty-kernel|"
        r"novelty-adversarial-falsifier|novelty-compression"
        r")(?:-\d{8}-\d{6})?$",
        name or "",
        re.I,
    )
    if not m:
        return None
    return f"O{m.group(1)}", m.group(2).lower()


def science_done_for_template(oxx: str, template: str,
                              entries: list | None = None, *,
                              source_revision: str | None = None) -> bool:
    mech = mechanism_for_template(template)
    if mech and science_is_done(
        oxx, mech, entries, source_revision=source_revision,
    ):
        return True
    # moe also produces a route-map; a sealed route-map still blocks relaunch
    if template == "external-science-moe" and science_is_done(
        oxx, "route-map", entries, source_revision=source_revision,
    ):
        return True
    return False


def machine_snapshot() -> dict:
    """Same resource-governor shape as ascent_controller.machine_snapshot."""
    try:
        from agentos.machine_state import clean_box_ok, snapshot

        snap = snapshot()
        ok, why = clean_box_ok(snap, min_free_gib=DISK_FLOOR_GIB)
        snap["clean_box_ok"], snap["clean_box_reason"] = ok, why
        return snap
    except Exception as exc:  # pragma: no cover - environment drift
        st = os.statvfs(REPO)
        gib = st.f_bavail * st.f_frsize / 1024**3
        return {
            "disk_free_gib": round(gib, 1),
            "clean_box_ok": gib >= DISK_FLOOR_GIB,
            "clean_box_reason": f"machine_state unavailable ({exc})",
        }


def reclaim_if_tight(snap: dict) -> None:
    if (snap.get("disk_free_gib") or 999) < DISK_WARN_GIB and RECLAIM.is_file():
        subprocess.run(["bash", str(RECLAIM)], cwd=REPO, check=False)


def value(work: dict) -> float:
    """§22 proxy: expected reusable-compiler-info / (wall+gpu+opus cost).

    Ordering only. No fake precision.
    """
    info = float(work.get("info") or 0.0)
    cost = (
        float(work.get("wall_cost") or 0.0)
        + float(work.get("gpu_cost") or 0.0)
        + float(work.get("opus_cost") or 0.0)
    )
    return info / max(cost, 0.1)


def map_state(ledger: str) -> str:
    raw = strip_md(ledger)
    low = raw.lower()
    if low.startswith("blocked"):
        return "BLOCKED"
    if raw.upper() in STATES:
        return raw.upper()
    # on-disk and queued both have READY work (measure / acquire)
    return "READY"


def norm_phase(phase: str) -> str:
    p = strip_md(phase).replace("✓", "").replace("✔", "").strip()
    return "" if p in {"", "—", "-"} else p


def norm_oxx(s: str) -> str:
    s = strip_md(s).upper()
    if s.startswith("O"):
        s = s[1:]
    return f"O{int(s):03d}"


def parse_ledger_rows() -> list[tuple]:
    if LEDGER.is_file():
        rows = []
        for line in LEDGER.read_text().splitlines():
            m = ROW_RE.match(line)
            if not m:
                continue
            oxx = m.group(1)
            if oxx.startswith("O") and oxx[1:].isdigit():
                rows.append(tuple(strip_md(g) for g in m.groups()))
        if len(rows) >= 14:
            return rows
    return list(FALLBACK_PATIENTS)


def patient_record(row: tuple) -> dict:
    oxx, model, klass, source, gate, ledger, phase = row
    state = map_state(ledger)
    on_disk = strip_md(ledger).lower().startswith("on-disk")
    blocked = state == "BLOCKED"
    return {
        "oxx": oxx,
        "model": model,
        "class": klass,
        "source": source,
        "gate": gate,
        "ledger": ledger,
        "state": state,
        "phase": norm_phase(phase),
        "on_disk": on_disk,
        "blocked_reason": "HF-gated / BLOCKED-auth" if blocked else None,
        "_evidence": "VERIFIED (ODYSSEY.md patient table)",
    }


def empty_state() -> dict:
    return {
        "schema": SCHEMA,
        "patients": [patient_record(r) for r in parse_ledger_rows()],
        "work": [dict(w) for w in SEED_WORK],
        "history": [],
        "harvested": [],
        "metrics": {
            "opus_calls": 1,
            "grok_lanes_bootstrap": 3,
            "gpu_seconds": 0,
            "gpu_owner": "none",
            "_evidence": "INFERRED (ODYSSEY.md §72 session bootstrap)",
        },
    }


def load_state() -> dict:
    if STATE.is_file():
        st = read_json(STATE)
        if not st.get("patients"):
            st["patients"] = empty_state()["patients"]
        if not st.get("work"):
            st["work"] = [dict(w) for w in SEED_WORK]
        return st
    st = empty_state()
    save_state(st)
    return st


def save_state(state: dict) -> None:
    write_json(STATE, state)


def ensure_state() -> dict:
    return load_state()


def packet_path(oxx: str) -> Path:
    return PATIENTS_DIR / oxx / f"ODYSSEY_PATIENT_{oxx}.json"


def census_path(oxx: str) -> Path:
    return PATIENTS_DIR / oxx / "census.json"


def load_packet(oxx: str) -> dict | None:
    p = packet_path(oxx)
    return read_json(p) if p.is_file() else None


def section_defaults(oxx: str, meta: dict) -> dict:
    kind = "moe" if oxx in MOE_PATIENTS else (
        "hybrid" if "hybrid" in (meta.get("class") or "").lower() else "dense"
    )
    return {
        "identity": {
            "source_repo": meta.get("source") or "",
            "revision": None,
            "content_hashes": None,
            "tokenizer": None,
            "source_precision": None,
            "source_type": None,
            "license": None,
            "model_family": None,
            "_evidence": "UNKNOWN",
        },
        "architecture": {
            "total_params": None,
            "active_params": None,
            "kind": kind,
            "layers": None,
            "attention": None,
            "state_ssm": None,
            "moe_topology": None,
            "experts": None,
            "experts_per_tok": None,
            "shared_experts": None,
            "modality": None,
            "context": None,
            "census_ref": None,
            "_evidence": "UNKNOWN",
        },
        "doctor": {
            "capability_vector": None,
            "controls": None,
            "long_context_result": None,
            "coding_reasoning": None,
            "behavioral_profile": None,
            "known_weaknesses": None,
            "blind_spots": None,
            "fast_doctor_seal_ref": None,
            "full_doctor_seal_ref": None,
            "_evidence": "UNKNOWN",
        },
        "tabula": {
            "status": "N/A",
            "behavioral_identity": None,
            "drift_monitor": None,
            "_evidence": "UNKNOWN",
        },
        "routing": {
            "entropy": None,
            "expert_frequency": None,
            "transitions": None,
            "co_occurrence": None,
            "hot_set": None,
            "cold_set": None,
            "route_predictability": None,
            "P(E_t|E_t-1)": None,
            "_evidence": "UNKNOWN",
        },
        "representation": {
            "source_bytes": None,
            "best_stored_bpw_eq": None,
            "active_bpw_eq": None,
            "metadata_bytes": None,
            "corrections": None,
            "reconstruction": None,
            "per_organ_sensitivity": None,
            "per_expert_sensitivity": None,
            "_evidence": "UNKNOWN",
        },
        "execution": {
            "token_ns": None,
            "tps": None,
            "ttft": None,
            "prefill": None,
            "long_context_slope": None,
            "active_learned_bytes_per_token": None,
            "dram_per_token": None,
            "cache": None,
            "state": None,
            "route_overhead": None,
            "baseline_runtime": None,
            "baseline_tps": None,
            "_evidence": "UNKNOWN",
        },
        "gravity": {
            "tried_mechanisms": [],
            "wins": [],
            "kills": [],
            "architecture_specific_findings": [],
            "_evidence": "UNKNOWN",
        },
        "nx": {
            "primitive_set": None,
            "machine_lowering": None,
            "kernel_bindings": None,
            "fallback_count": None,
            "best_preliminary_nx": None,
            "nx_hash": None,
            "_evidence": "UNKNOWN",
        },
        "transfer": {
            "inherited_rules": [],
            "unchanged": [],
            "retuned": [],
            "failed": [],
            "harmful": [],
            "_evidence": "UNKNOWN",
        },
        "next": [],
    }


def patient_meta(oxx: str, state: dict | None = None) -> dict:
    st = state or ensure_state()
    for p in st.get("patients") or []:
        if p.get("oxx") == oxx:
            return overlay_manifest(p, oxx)
    man = manifest_entry(oxx)
    if man:
        return overlay_manifest({"oxx": oxx, "source": "", "class": man.get("class") or "",
                                 "phase": "", "model": man.get("model") or oxx}, oxx)
    return {"oxx": oxx, "source": "", "class": "", "phase": "", "model": oxx}


def ensure_sections(pkt: dict, oxx: str, meta: dict) -> dict:
    defaults = section_defaults(oxx, meta)
    pkt.setdefault("oxx", oxx)
    if meta.get("class") and "class" not in pkt:
        pkt["class"] = meta["class"]
    if meta.get("phase") and "phase" not in pkt:
        pkt["phase"] = meta["phase"]
    for key, default in defaults.items():
        if key not in pkt:
            pkt[key] = json.loads(json.dumps(default))
        elif isinstance(default, dict) and isinstance(pkt[key], dict):
            for dk, dv in default.items():
                pkt[key].setdefault(dk, dv)
        elif key == "next" and pkt[key] is None:
            pkt[key] = []
    return pkt


def _stamp(section: dict, stamp: str) -> None:
    prev = str(section.get("_evidence") or "")
    if stamp in prev:
        return
    if not prev or prev == "UNKNOWN":
        section["_evidence"] = stamp
    else:
        section["_evidence"] = f"{stamp} + {prev}"


def apply_census(pkt: dict, census: dict) -> None:
    arch = pkt.setdefault("architecture", {})
    cfg = census.get("config") or {}
    arch["arch"] = census.get("arch") or arch.get("arch")
    arch["total_params"] = census.get("total_params")
    active = census.get("active_params_per_token")
    arch["active_params"] = active
    arch["active_params_per_token"] = active
    total = census.get("total_params")
    if total and active:
        arch["active_pct"] = round(100.0 * active / total, 1)
    arch["layers"] = cfg.get("num_hidden_layers", arch.get("layers"))
    arch["hidden_size"] = cfg.get("hidden_size", arch.get("hidden_size"))
    if census.get("is_moe"):
        arch["kind"] = "moe"
        arch["experts"] = cfg.get("num_experts") or cfg.get("n_routed_experts") or cfg.get("num_local_experts")
        arch["experts_per_tok"] = (
            cfg.get("num_experts_per_tok") or cfg.get("moe_topk") or cfg.get("n_experts_per_tok")
        )
        arch["shared_experts"] = cfg.get("n_shared_experts") or arch.get("shared_experts") or 0
    elif not arch.get("kind"):
        arch["kind"] = "dense"
    arch["census_ref"] = f"patients/{pkt['oxx']}/census.json"
    _stamp(arch, "MEASURED (census)")

    rep = pkt.setdefault("representation", {})
    rep["source_bytes"] = census.get("total_bytes")
    rep["best_stored_bpw_eq"] = census.get("stored_bpw")
    rep["stored_bpw"] = census.get("stored_bpw")
    obytes = census.get("organs_bytes") or {}
    organs_gb = {k: round(v / 1e9, 2) for k, v in obytes.items() if v}
    prev_organs = rep.get("organs_bytes_GB")
    if isinstance(prev_organs, dict) and prev_organs.get("ssm") and "ssm" not in organs_gb:
        organs_gb["ssm"] = prev_organs["ssm"]
        if "other" in organs_gb and prev_organs.get("other") is not None:
            organs_gb["other"] = prev_organs["other"]
    if organs_gb:
        rep["organs_bytes_GB"] = organs_gb
    if census.get("active_bytes_per_token"):
        rep["active_bytes_per_token_bf16"] = census["active_bytes_per_token"]
    _stamp(rep, "MEASURED (census)")


def apply_a3b(pkt: dict, recon: dict) -> None:
    if pkt.get("oxx") != "O005":
        return
    rm = recon.get("route_map") or {}
    routing = pkt.setdefault("routing", {})

    def _fill(dst, key, val):
        if val is not None and dst.get(key) is None:
            dst[key] = val

    if rm:
        _fill(routing, "entropy", rm.get("avg_layer_route_entropy_bits"))
        _fill(routing, "expert_frequency", {
            "most_popular_expert_share_pct": rm.get("most_popular_expert_share_pct"),
            "pct_mass_top16_experts": rm.get("pct_mass_top16_experts"),
            "never_routed_experts": rm.get("never_routed_experts"),
        })
        _fill(
            routing,
            "cold_set",
            "empty" if rm.get("never_routed_experts") == 0 else rm.get("never_routed_experts"),
        )
        _fill(routing, "hot_set", "none (near-uniform)")
        if evidence_class(routing.get("_evidence")) in (None, "UNKNOWN"):
            routing["_evidence"] = (
                "MEASURED (A3B_RECON specimen, short prompt set; A1 still required)"
            )
    ex = pkt.setdefault("execution", {})
    tps = (recon.get("A3B_baseline") or {}).get("tps_specimen")
    if tps is not None:
        ex.setdefault("tps_specimen_mlx", tps)
        # specimen is not a canonical baseline (foreign-runtime gate, bible §60)
        if evidence_class(ex.get("_evidence")) in (None, "UNKNOWN"):
            if "specimen" not in str(ex.get("_evidence") or "").lower():
                ex["_evidence"] = (
                    "UNKNOWN (canonical baseline pending; mlx specimen is not a Hawking measurement, §60)"
                )
    g = pkt.setdefault("gravity", {})
    findings = g.setdefault("architecture_specific_findings", [])
    for key, val in (recon.get("CLASSIFICATION") or {}).items():
        note = f"{key}: {val}"
        if note not in findings:
            findings.append(note)
    if findings and evidence_class(g.get("_evidence")) in (None, "UNKNOWN"):
        g["_evidence"] = "INFERRED (A3B_RECON CLASSIFICATION)"


def apply_transfer(pkt: dict) -> None:
    if not TRANSFER.is_file():
        return
    try:
        matrix = read_json(TRANSFER)
    except (OSError, json.JSONDecodeError):
        return
    oxx = pkt.get("oxx")
    tr = pkt.setdefault("transfer", {})
    inherited, unchanged, retuned, failed, harmful = [], [], [], [], []
    for row in matrix.get("rows") or []:
        rid = row.get("rule")
        cell = (row.get("cells") or {}).get(oxx)
        if not rid or not cell or cell == "NOT_TESTED":
            continue
        inherited.append(rid)
        if cell == "TRANSFERRED_UNCHANGED":
            unchanged.append(rid)
        elif cell in ("TRANSFERRED_RETUNED", "RETUNED"):
            retuned.append(rid)
        elif cell == "FAILED":
            failed.append(rid)
        elif cell == "HARMFUL":
            harmful.append(rid)
    if inherited:
        tr["inherited_rules"] = inherited
        tr["unchanged"] = unchanged
        tr["retuned"] = retuned
        tr["failed"] = failed
        tr["harmful"] = harmful
        if evidence_class(tr.get("_evidence")) in (None, "UNKNOWN"):
            tr["_evidence"] = "INFERRED (TRANSFER_MATRIX.json)"


def apply_receipts(pkt: dict, oxx: str) -> None:
    if not RECEIPT_DIR.is_dir():
        return
    findings = []
    for path in sorted(RECEIPT_DIR.glob("*.json")):
        try:
            rec = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("oxx") != oxx or rec.get("verdict") != "ACCEPTED":
            continue
        body = rec.get("outputs") or {}
        if isinstance(body, dict) and body.get("finding"):
            findings.append(str(body["finding"]))
        ev = rec.get("evidence")
        payload = body.get("structured") if isinstance(body, dict) else None
        if isinstance(payload, dict):
            if "routing" in payload and isinstance(payload["routing"], dict):
                pkt.setdefault("routing", {}).update(payload["routing"])
            if "gravity" in payload and isinstance(payload["gravity"], dict):
                g = pkt.setdefault("gravity", {})
                for key in ("tried_mechanisms", "wins", "kills"):
                    extra = payload["gravity"].get(key) or []
                    bucket = g.setdefault(key, [])
                    for item in extra:
                        if item not in bucket:
                            bucket.append(item)
        if ev and ev not in (pkt.get("harvest_evidence") or []):
            pkt.setdefault("harvest_evidence", []).append(ev)
    nxt = pkt.get("next")
    if isinstance(nxt, list):
        for f in findings:
            line = f"harvest: {f}"
            if line not in nxt:
                nxt.append(line)


def seal_packet(pkt: dict) -> None:
    """Reuse doctor_seal.seal for every patient seal (structural only)."""
    doctor = pkt.get("doctor") or {}
    candidate = {
        "tabula_drift": (pkt.get("tabula") or {}).get("drift_monitor") or doctor.get("tabula_drift"),
        "observed_controls": doctor.get("controls") or doctor.get("observed_controls"),
        "stated_test_width": doctor.get("stated_test_width"),
        "known_blind_spots": doctor.get("blind_spots") or doctor.get("known_blind_spots"),
    }
    verdict, reasons = doctor_seal.seal(candidate)
    pkt["receipt_seal"] = {
        "verdict": verdict,
        "reasons": reasons,
        "_evidence": "DERIVED (doctor_seal.seal structural)",
    }


def validate_packet(pkt: dict) -> list[str]:
    errs = []
    if not pkt.get("oxx"):
        errs.append("missing oxx")
    for sec in PACKET_SECTIONS:
        if sec not in pkt:
            errs.append(f"missing section {sec}")
            continue
        val = pkt[sec]
        if isinstance(val, dict):
            ev = evidence_class(val.get("_evidence"))
            if ev is None:
                errs.append(f"{sec} missing §18 _evidence")
        elif sec == "next" and not isinstance(val, (list, dict, str)):
            errs.append("next must be list/dict/str")
    return errs


def assemble_packet(oxx: str, state: dict | None = None) -> dict:
    oxx = norm_oxx(oxx)
    meta = patient_meta(oxx, state)
    pkt = load_packet(oxx) or {"oxx": oxx}
    pkt = ensure_sections(pkt, oxx, meta)
    cp = census_path(oxx)
    if cp.is_file():
        apply_census(pkt, read_json(cp))
    if A3B_RECON.is_file():
        try:
            apply_a3b(pkt, read_json(A3B_RECON))
        except (OSError, json.JSONDecodeError):
            pass
    apply_transfer(pkt)
    apply_receipts(pkt, oxx)
    # §18: every section needs a recognised evidence class. Preserve notes
    # like "N/A — dense/hybrid" by wrapping them, do not invent a stronger class.
    for sec in PACKET_SECTIONS:
        val = pkt.get(sec)
        if not isinstance(val, dict):
            continue
        if evidence_class(val.get("_evidence")) is None:
            prev = val.get("_evidence")
            val["_evidence"] = f"UNKNOWN ({prev})" if prev else "UNKNOWN"
    if not pkt.get("next"):
        st = state or ensure_state()
        pkt["next"] = [
            f"{w['id']} {w['title']}"
            for w in st.get("work") or []
            if w.get("oxx") == oxx and w.get("status") == "READY"
        ]
    seal_packet(pkt)
    return pkt


def write_packet(oxx: str, state: dict | None = None) -> Path:
    oxx = norm_oxx(oxx)
    pkt = assemble_packet(oxx, state)
    errs = validate_packet(pkt)
    if errs:
        raise SystemExit(f"packet {oxx} invalid: {errs}")
    dest = packet_path(oxx)
    write_json(dest, pkt)
    return dest


# ---------------------------------------------------------------------------
# harvest (§12)
# ---------------------------------------------------------------------------

def classify_evidence(text: str, structured=None) -> str:
    if isinstance(structured, dict):
        ev = evidence_class(structured.get("evidence") or structured.get("_evidence"))
        if ev:
            return ev
    hits = [ev for ev in EVIDENCE if re.search(rf"\b{ev}\b", text)]
    if not hits:
        return "UNKNOWN"
    # strongest first (EVIDENCE is already strongest-first)
    for ev in EVIDENCE:
        if ev in hits:
            return ev
    return "UNKNOWN"


def parse_report(text: str) -> dict:
    structured = None
    m = JSON_FENCE_RE.search(text)
    if m:
        try:
            structured = json.loads(m.group(1))
        except json.JSONDecodeError:
            structured = None
    completion = None
    cm = COMPLETION_RE.search(text)
    if cm:
        completion = cm.group(1).strip()[:2000]
    has_result = bool(RESULT_RE.search(text))
    ok = structured is not None or completion is not None or has_result
    oxx = None
    if isinstance(structured, dict) and structured.get("oxx"):
        try:
            oxx = norm_oxx(str(structured["oxx"]))
        except (TypeError, ValueError):
            oxx = None
    if oxx is None:
        hm = OXX_RE.search(text)
        if hm:
            oxx = f"O{hm.group(1)}"
    return {
        "structured": structured,
        "completion": completion,
        "ok": ok,
        "oxx": oxx,
        "evidence": classify_evidence(text, structured),
    }


def oxx_from_task(name: str, parsed: dict) -> str | None:
    if parsed.get("oxx"):
        return parsed["oxx"]
    m = SLUG_OXX_RE.search(name)
    return f"O{m.group(1)}" if m else None


def should_escalate(text: str, evidence: str) -> tuple[str, str] | None:
    if evidence == "REFUTED":
        return "CONTRADICTION", "evidence classified REFUTED"
    if re.search(r"two (high-quality )?receipts disagree", text, re.I):
        return "CONTRADICTION", "receipts disagree"
    if re.search(r"\bANOMALY\b|\bESCALATE(?: TO OPUS)?\b", text):
        return "ANOMALY", "explicit nomination"
    if re.search(r"\bHARMFUL\b.*\btransfer\b", text, re.I):
        return "MAJOR FALSE WIN", "harmful transfer"
    return None


def _task_status(task_dir: Path) -> str:
    p = task_dir / "status"
    if not p.is_file():
        return "unknown"
    return p.read_text(errors="replace").strip().splitlines()[0].strip().lower()


def harvest(*, tasks_root: Path | None = None, receipt_dir: Path | None = None,
            escalate_path: Path | None = None, state: dict | None = None,
            classify: bool = False, dry_run: bool = False,
            worktrees_root: Path | None = None, dest_root: Path | None = None,
            review_queue: Path | None = None, cleanup_fn=None,
            persist: bool | None = None) -> list[dict]:
    """Scan completed odyssey-* grok lanes. Reject reports with no structured result.

    classify/dry_run: hardened lane harvest (DATA-ONLY vs CODE). Default stays
    the original report harvester so --self-check fixtures keep working.
    """
    if classify or dry_run:
        return harvest_lanes(
            tasks_root=tasks_root, receipt_dir=receipt_dir,
            escalate_path=escalate_path, state=state, dry_run=dry_run,
            worktrees_root=worktrees_root, dest_root=dest_root,
            review_queue=review_queue, cleanup_fn=cleanup_fn, persist=persist,
        )
    root = Path(tasks_root) if tasks_root else GROK_TASKS
    out_dir = Path(receipt_dir) if receipt_dir else RECEIPT_DIR
    esc_path = Path(escalate_path) if escalate_path else ESCALATIONS
    st = state if state is not None else ensure_state()
    already = set(st.get("harvested") or [])
    rows = []
    if not root.is_dir():
        return rows
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("odyssey-")):
        report = d / "grok-report.md"
        stat = _task_status(d)
        if stat in {"running", "queued", "starting"}:
            continue
        rec = {
            "schema": "hawking.odyssey.harvest.v1",
            "task": d.name,
            "slug": re.sub(r"-\d{8}-\d{6}$", "", d.name),
            "task_status": stat,
            "verdict": "REJECTED",
            "reason": "",
            "evidence": "UNKNOWN",
            "oxx": None,
            "patient_hash": None,
            "representation_hash": None,
            "nx_hash": None,
            "machine_state": None,
            "command": "harvest",
            "inputs": {"task_dir": str(d)},
            "outputs": {},
            "controls": [],
            "assumptions": ["only odyssey-* slugs; running lanes skipped"],
            "reopen_if": "structured result later appears in grok-report.md",
            "_evidence": "DERIVED (harvester)",
        }
        if not report.is_file():
            rec["reason"] = "malformed: no grok-report.md"
        else:
            try:
                text = report.read_text(errors="replace")
            except OSError as exc:
                rec["reason"] = f"malformed: unreadable report ({exc})"
                text = ""
            if text:
                parsed = parse_report(text)
                rec["oxx"] = oxx_from_task(d.name, parsed)
                rec["evidence"] = parsed["evidence"]
                rec["_evidence"] = parsed["evidence"]
                if not parsed["ok"]:
                    rec["reason"] = "malformed: no structured result"
                else:
                    rec["verdict"] = "ACCEPTED"
                    rec["reason"] = "ok"
                    rec["outputs"] = {
                        "finding": (parsed.get("completion") or "")[:400],
                        "structured": parsed.get("structured"),
                    }
                    trig = should_escalate(text, parsed["evidence"])
                    if trig:
                        rec["escalation"] = {"trigger": trig[0], "note": trig[1]}
                        esc_path.parent.mkdir(parents=True, exist_ok=True)
                        with esc_path.open("a") as fh:
                            fh.write(json.dumps({
                                "task": d.name,
                                "oxx": rec["oxx"],
                                "trigger": trig[0],
                                "note": trig[1],
                                "evidence": parsed["evidence"],
                                "_evidence": "INFERRED (harvester nomination, bible §9/§12)",
                            }) + "\n")
                    if receipt_dir is None:
                        record_ctl_event(
                            rec["oxx"], "grok", 0.0,
                            grok_lane=d.name,
                            opus=bool(trig),
                            extra={"verdict": rec["verdict"], "harvest": "report"},
                            persist=True,
                        )
        dest = out_dir / f"{d.name}.json"
        write_json(dest, rec)
        rows.append({"task": d.name, "verdict": rec["verdict"],
                     "reason": rec["reason"], "oxx": rec["oxx"],
                     "receipt": str(dest)})
        if rec["verdict"] == "ACCEPTED" and rec["oxx"] and receipt_dir is None:
            # only mutate real packets when writing to the real receipt dir
            try:
                write_packet(rec["oxx"], st)
            except SystemExit:
                pass
        if d.name not in already:
            already.add(d.name)
    st["harvested"] = sorted(already)
    if receipt_dir is None and escalate_path is None and tasks_root is None:
        save_state(st)
    return rows


# ---------------------------------------------------------------------------
# hardened harvest — classify by template + RECEIPT_PATTERN, not by
# tools/*.py inspection. Known data-producing + matching receipt is
# DATA-ONLY (drop runner diffs). Cleanup only after a DATA-ONLY copy succeeds.
# ---------------------------------------------------------------------------

DATA_RECEIPT_PREFIX = "receipts/"
DATA_PACKET_PREFIX = "workspace/campaign/odyssey/patients/"
DIFF_PLUS_RE = re.compile(r"^\+\+\+ b/(.+?)(?:\t|$)")


def norm_rel(path: str) -> str:
    p = (path or "").replace("\\", "/").strip().strip('"')
    while p.startswith("./"):
        p = p[2:]
    return p


def is_data_path(path: str) -> bool:
    """DATA-ONLY: receipts/** or workspace/campaign/odyssey/patients/*/*.json."""
    p = norm_rel(path)
    if p.startswith(DATA_RECEIPT_PREFIX):
        return True
    if p.startswith(DATA_PACKET_PREFIX) and p.endswith(".json"):
        rest = p[len(DATA_PACKET_PREFIX):]
        return "/" in rest and not rest.endswith("/")
    return False


def classify_paths(paths: list[str]) -> str:
    cleaned = []
    for raw in paths:
        p = norm_rel(raw)
        if not p or p == "/dev/null":
            continue
        cleaned.append(p)
    if not cleaned:
        return "DATA-ONLY"
    if all(is_data_path(p) for p in cleaned):
        return "DATA-ONLY"
    return "CODE"


def is_expected_receipt(rel: str, oxx: str, template: str) -> bool:
    """True if rel is receipts/odyssey-i/<OXX>_*.json matching RECEIPT_PATTERN."""
    n = norm_rel(rel)
    want = expected_receipt_rel(oxx, template)
    if want and n == want:
        return True
    if not n.startswith(DATA_RECEIPT_PREFIX) or not n.endswith(".json"):
        return False
    if not oxx:
        return False
    leaf = n.rsplit("/", 1)[-1]
    if template in AGGRESSIVE_GRAVITY_TEMPLATES:
        prefix = f"{oxx}_GRAVITY_"
        if leaf.startswith(prefix) and leaf.endswith(".json"):
            spec = leaf[len(prefix):-5]
            klass = classify_gravity_spec(spec).get("candidate_class")
            return klass in {
                "AGGRESSIVE_QUANT", "STRUCTURAL_GRAVITY", "FRONTIER", "ACTIVE_NX",
            }
        return False
    if template and template.startswith("novelty-"):
        lane = template.split("novelty-", 1)[-1]
        return leaf == f"{oxx}_NOVELTY_{lane}.json"
    pat = RECEIPT_PATTERN.get(template) or RECEIPT_PATTERN.get(
        mechanism_for_template(template or "")
    )
    if not pat:
        return False
    return fnmatch.fnmatch(leaf, pat.format(oxx=oxx))


def lane_matching_receipts(files: list[str], worktree: Path | None,
                           dest_root: Path | None, oxx: str,
                           template: str) -> list[str]:
    """Receipts this data-producing lane produced (listed in the diff or on disk)."""
    want = expected_receipt_rel(oxx, template)
    listed = {norm_rel(p) for p in files if norm_rel(p)}
    found: list[str] = []
    seen: set[str] = set()
    candidates = list(files)
    if want:
        candidates.append(want)
    for raw in candidates:
        n = norm_rel(raw)
        if not n or n in seen or n == "/dev/null":
            continue
        if not is_expected_receipt(n, oxx, template):
            continue
        on_disk = _path_exists(worktree, n) or _path_exists(dest_root, n)
        if n in listed or on_disk:
            seen.add(n)
            found.append(n)
    return found


def resolve_lane_identity(task_name: str, state: dict | None,
                          parsed: dict | None = None) -> tuple[str | None, str | None]:
    """(oxx, template) from the task slug first, then the work item."""
    parsed_task = parse_science_task(task_name) or parse_science_task(
        re.sub(r"-\d{8}-\d{6}$", "", task_name)
    )
    oxx = tmpl = None
    if parsed_task:
        oxx, tmpl = parsed_task
    w = _work_for_task(state, task_name) if state is not None else None
    if w:
        oxx = oxx or w.get("oxx")
        tmpl = tmpl or w.get("template")
    if parsed and parsed.get("oxx"):
        oxx = oxx or parsed.get("oxx")
    if not oxx:
        m = SLUG_OXX_RE.search(task_name or "")
        if m:
            oxx = f"O{m.group(1)}"
    return oxx, tmpl


def classify_lane(files: list[str], *, oxx: str | None = None,
                  template: str | None = None, worktree: Path | None = None,
                  dest_root: Path | None = None) -> tuple[str, list[str]]:
    """Classify by template + expected receipt, not by tools/*.py inspection.

    Known data-producing template + matching receipt → DATA-ONLY (runner
    diffs are noise). Known template with no receipt, or unknown template
    with a non-data path → CODE (REVIEW). Unknown template with only data
    paths → DATA-ONLY (legacy path-based).
    """
    if template in DATA_PRODUCING_TEMPLATES and oxx:
        matching = lane_matching_receipts(
            files, worktree, dest_root, oxx, template,
        )
        if matching:
            return "DATA-ONLY", matching
        return "CODE", []
    return classify_paths(files), []


def parse_diff_paths(text: str) -> list[str]:
    out = []
    for line in (text or "").splitlines():
        m = DIFF_PLUS_RE.match(line)
        if not m:
            if line.startswith("+++ "):
                rest = line[4:]
                if rest.startswith("b/"):
                    rest = rest[2:]
                rest = rest.split("\t", 1)[0].strip()
                if rest and rest != "/dev/null":
                    out.append(norm_rel(rest))
            continue
        p = norm_rel(m.group(1))
        if p and p != "/dev/null":
            out.append(p)
    return out


def parse_porcelain(text: str) -> list[str]:
    out = []
    for raw in (text or "").splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if " -> " in line:
            out.append(norm_rel(line.split(" -> ", 1)[1].strip().strip('"')))
            continue
        path = line[3:] if len(line) >= 3 else line
        path = path.strip().strip('"')
        if path:
            out.append(norm_rel(path))
    return out


def worktree_porcelain_paths(worktree: Path | None) -> list[str]:
    if worktree is None or not worktree.is_dir():
        return []
    r = subprocess.run(
        ["git", "-C", str(worktree), "status", "--porcelain"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return []
    return parse_porcelain(r.stdout or "")


def resolve_worktree(task_dir: Path, worktrees_root: Path | None = None) -> Path | None:
    meta = task_dir / "metadata.json"
    if meta.is_file():
        try:
            doc = read_json(meta)
        except (OSError, json.JSONDecodeError):
            doc = {}
        wd = doc.get("workdir") or doc.get("worktree")
        if wd:
            p = Path(wd)
            if p.is_dir():
                return p
    root = Path(worktrees_root) if worktrees_root else GROK_WORKTREES
    cand = root / task_dir.name
    return cand if cand.is_dir() else None


def lane_file_list(task_dir: Path, worktree: Path | None) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    patch = task_dir / "diff.patch"
    if patch.is_file():
        try:
            paths.extend(parse_diff_paths(patch.read_text(errors="replace")))
        except OSError:
            pass
    paths.extend(worktree_porcelain_paths(worktree))
    out = []
    for p in paths:
        n = norm_rel(p)
        if not n or n == "/dev/null" or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _work_for_task(state: dict, task_name: str) -> dict | None:
    """Match a work item to a task. Timestamped names match exactly so two
    odyssey-o005-sensitivity-map-<ts> lanes do not steal each other's row."""
    works = state.get("work") or []
    for w in works:
        if w.get("task") == task_name:
            return w
    slug = re.sub(r"-\d{8}-\d{6}$", "", task_name)
    if slug != task_name:
        return None
    for w in works:
        t = str(w.get("task") or "")
        if not t:
            continue
        t_slug = re.sub(r"-\d{8}-\d{6}$", "", t)
        if t == task_name or t_slug == slug:
            return w
    return None


def _recorded_running(state: dict, task_name: str) -> bool:
    w = _work_for_task(state, task_name)
    return bool(w and w.get("status") == "RUNNING")


def _review_queue_has(path: Path | None, task_name: str) -> bool:
    if path is None or not path.is_file():
        return False
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return False
    for line in lines:
        if not line.strip():
            continue
        try:
            if json.loads(line).get("task") == task_name:
                return True
        except json.JSONDecodeError:
            continue
    return False


def _lane_already_resolved(state: dict, task_name: str,
                           receipt_dir: Path, review_path: Path | None) -> str | None:
    rec = receipt_dir / f"harvest_{task_name}.json"
    if rec.is_file():
        return "already harvested"
    w = _work_for_task(state, task_name)
    if w and w.get("status") in ("VERIFIED", "REVIEW", "REFUTED"):
        return f"already {w['status']}"
    if _review_queue_has(review_path, task_name):
        return "already in REVIEW_QUEUE"
    return None


def _mark_lane(state: dict, task_name: str, status: str, **extra) -> dict | None:
    w = _work_for_task(state, task_name)
    if w is not None:
        w["status"] = status
        w["task"] = task_name
        for key, val in extra.items():
            if val is not None:
                w[key] = val
    harvested = list(state.get("harvested") or [])
    if task_name not in harvested:
        harvested.append(task_name)
    state["harvested"] = harvested
    return w


def append_review_queue(path: Path, row: dict) -> bool:
    if _review_queue_has(path, row.get("task")):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


def default_cleanup(task_id: str) -> tuple[bool, str]:
    if not GROK_BIN.is_file():
        return False, "grok-run missing"
    r = subprocess.run(
        [str(GROK_BIN), "cleanup", "--id", task_id],
        capture_output=True, text=True,
    )
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    return r.returncode == 0, out[-400:]


def _path_exists(root: Path | None, rel: str) -> bool:
    return bool(root and (root / rel).is_file())


def copy_data_files(worktree: Path | None, dest_root: Path,
                    files: list[str]) -> tuple[list[str], list[str]]:
    copied, missing = [], []
    for rel in files:
        if not is_data_path(rel):
            continue
        src = (worktree / rel) if worktree is not None else None
        if src is None or not src.is_file():
            missing.append(rel)
            continue
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(rel)
    return copied, missing


def _receipt_present(files: list[str], worktree: Path | None,
                     dest_root: Path | None) -> bool:
    for rel in files:
        n = norm_rel(rel)
        if not n.startswith(DATA_RECEIPT_PREFIX):
            continue
        if _path_exists(worktree, n) or _path_exists(dest_root, n):
            return True
    return False


def planned_action(classification: str | None, reason: str) -> str:
    if reason.startswith("malformed"):
        return f"REFUTE ({reason})"
    if classification == "CODE":
        return "REVIEW (do not merge, do not cleanup)"
    if classification == "DATA-ONLY":
        return "MERGE+COMPLETE (copy data, drop tools/, mark VERIFIED, complete, grok-run cleanup)"
    return f"REFUTE ({reason or 'unclassified'})"


def _dest_completions(dest: Path) -> Path:
    try:
        if dest.resolve() == REPO.resolve():
            return COMPLETIONS
    except OSError:
        pass
    return dest / "workspace" / "campaign" / "odyssey" / "ODYSSEY_COMPLETIONS.json"


def _dest_receipt_dir(dest: Path) -> Path:
    try:
        if dest.resolve() == REPO.resolve():
            return RECEIPT_DIR
    except OSError:
        pass
    return dest / "receipts" / "odyssey-i"


def _complete_harvested_lane(task_name: str, oxx: str | None, work: dict | None,
                             *, completed_at: str | None = None,
                             persist: bool = True, path: Path | None = None,
                             receipt_dir: Path | None = None,
                             dest_root: Path | None = None) -> None:
    """Record a VERIFIED harvest in the completion index. No-op if undetermined."""
    if not oxx:
        return
    tmpl = (work or {}).get("template")
    parsed = parse_science_task(task_name) or parse_science_task(
        re.sub(r"-\d{8}-\d{6}$", "", task_name)
    )
    if not tmpl and parsed:
        oxx = oxx or parsed[0]
        tmpl = parsed[1]
    if not tmpl:
        return
    mech = mechanism_for_template(tmpl)
    if not mech:
        return
    rec_name = receipt_filename(oxx, mech) or receipt_filename(oxx, tmpl)
    rec_root = Path(receipt_dir) if receipt_dir else RECEIPT_DIR
    rec_path = rec_root / rec_name if rec_name else None
    if rec_path is not None and not rec_path.is_file() and rec_name:
        fallback = RECEIPT_DIR / rec_name
        if fallback.is_file():
            rec_path = fallback
    if rec_path is None or not rec_path.is_file():
        return
    stamp = completed_at or os.environ.get("ODYSSEY_COMPLETED_AT")
    if not stamp:
        stamp = receipt_stamp(rec_path)
    sha = file_sha256(rec_path)
    ref = f"receipts/odyssey-i/{rec_name}" if rec_name else None
    mechs = [mech]
    if tmpl == "external-science-moe" and "route-map" not in mechs:
        mechs.append("route-map")
    if tmpl == "external-science-dense":
        kind = arch_kind(oxx, load_packet(oxx), load_census(oxx))
        if kind == "hybrid" and "ssm-accounting" not in mechs:
            mechs.append("ssm-accounting")
    head = git_head()
    tags = gravity_tags_from_receipt(rec_path) if str(mech).startswith("gravity-") else {}
    for mechanism_id in mechs:
        complete(
            obligation_id=f"{oxx}:{mechanism_id}",
            patient_id=oxx,
            mechanism_id=mechanism_id,
            status="VERIFIED",
            completed_at=stamp,
            receipt_ref=ref,
            receipt_sha256=sha,
            source_revision=head,
            persist=persist,
            path=path,
            candidate_class=tags.get("candidate_class"),
            conventionality=tags.get("conventionality"),
        )
    if tags:
        apply_gravity_tags_to_packet(oxx, tags, dest_root=dest_root)


def harvest_lanes(*, tasks_root: Path | None = None,
                  receipt_dir: Path | None = None,
                  escalate_path: Path | None = None,
                  state: dict | None = None, dry_run: bool = False,
                  worktrees_root: Path | None = None,
                  dest_root: Path | None = None,
                  review_queue: Path | None = None,
                  cleanup_fn=None, persist: bool | None = None) -> list[dict]:
    """Classify finished odyssey-* lanes and (unless dry-run) reap DATA-ONLY.

    Classification is by template + expected receipt, not by tools/*.py diffs.
    Known data-producing template with a matching receipt is DATA-ONLY even
    if the lane also tweaked the runner: copy receipts + packet, drop tools/,
    write the completion, cleanup. CODE → REVIEW only for unknown templates
    or known templates that produced no valid receipt. Malformed (no report)
    → REFUTED. DATA-ONLY apply does not require the work item to be RUNNING.
    Dry-run classifies every finished lane and changes nothing.
    """
    del escalate_path  # reserved; report harvest already nominates
    root = Path(tasks_root) if tasks_root else GROK_TASKS
    out_dir = Path(receipt_dir) if receipt_dir else RECEIPT_DIR
    dest = Path(dest_root) if dest_root else REPO
    qpath = Path(review_queue) if review_queue else REVIEW_QUEUE
    st = state if state is not None else ensure_state()
    do_persist = persist if persist is not None else (
        tasks_root is None and receipt_dir is None
        and dest_root is None and review_queue is None
        and not dry_run
    )
    cleaner = cleanup_fn or default_cleanup
    rows: list[dict] = []
    mutated = False
    if not root.is_dir():
        return rows

    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("odyssey-")):
        stat = _task_status(d)
        if stat != "done":
            continue
        wt = resolve_worktree(d, worktrees_root)
        files = lane_file_list(d, wt)
        report = d / "grok-report.md"
        parsed = None
        reason = ""
        oxx, tmpl = resolve_lane_identity(d.name, st)
        matching_receipts: list[str] = []
        if not report.is_file():
            reason = "malformed: no grok-report.md"
            classification, matching_receipts = classify_lane(
                files, oxx=oxx, template=tmpl, worktree=wt, dest_root=dest,
            )
        else:
            try:
                text = report.read_text(errors="replace")
            except OSError as exc:
                text = ""
                reason = f"malformed: unreadable report ({exc})"
            if text:
                parsed = parse_report(text)
                oxx2 = oxx_from_task(d.name, parsed)
                if oxx2:
                    oxx = oxx2
                slug_m = SLUG_OXX_RE.search(d.name)
                if slug_m:
                    oxx = f"O{slug_m.group(1)}"
                oxx, tmpl = resolve_lane_identity(d.name, st, parsed)
                if slug_m:
                    oxx = f"O{slug_m.group(1)}"
            classification, matching_receipts = classify_lane(
                files, oxx=oxx, template=tmpl, worktree=wt, dest_root=dest,
            )
            if tmpl in DATA_PRODUCING_TEMPLATES:
                # known template + no expected receipt stays CODE (REVIEW),
                # not malformed — that is the infra/broken-lane signal.
                pass
            elif classification == "DATA-ONLY" and not _receipt_present(files, wt, dest):
                reason = "malformed: no receipt"
            elif classification == "DATA-ONLY" and parsed and not parsed.get("ok"):
                if not _receipt_present(files, wt, dest):
                    reason = "malformed: no structured result"
        action = planned_action(classification, reason)
        already = _lane_already_resolved(st, d.name, out_dir, qpath)
        in_scope = _recorded_running(st, d.name)
        data_apply = (
            classification == "DATA-ONLY"
            and not reason.startswith("malformed")
        )
        row = {
            "schema": HARVEST_SCHEMA,
            "task": d.name,
            "slug": re.sub(r"-\d{8}-\d{6}$", "", d.name),
            "task_status": stat,
            "classification": classification,
            "action": action,
            "reason": reason or (already or ("ok" if classification else "unclassified")),
            "files": files,
            "matching_receipts": matching_receipts,
            "template": tmpl,
            "oxx": oxx,
            "worktree": str(wt) if wt else None,
            "report": str(report) if report.is_file() else None,
            "in_scope": in_scope,
            "applied": False,
            "cleanup": False,
            "copied": [],
            "dropped": [],
            "verdict": None,
            "dry_run": dry_run,
            "_evidence": "DERIVED (harvester classification)",
        }
        if already:
            row["reason"] = already
            row["verdict"] = "SKIP"
            rows.append(row)
            continue

        apply_ok = in_scope or data_apply
        if dry_run or not apply_ok:
            if not apply_ok and not dry_run:
                row["reason"] = (reason + "; " if reason else "") + "not RUNNING in state"
            rows.append(row)
            continue

        # ---- apply ----
        dropped = [p for p in files if not is_data_path(p)]
        row["dropped"] = dropped
        if reason.startswith("malformed"):
            _mark_lane(st, d.name, "REFUTED", oxx=oxx, harvest_reason=reason)
            row["verdict"] = "REFUTED"
            rec_verdict = "REFUTED"
        elif classification == "CODE":
            append_review_queue(qpath, {
                "task": d.name,
                "files": files,
                "report": str(report) if report.is_file() else None,
                "worktree": str(wt) if wt else None,
            })
            _mark_lane(st, d.name, "REVIEW", oxx=oxx, harvest_reason="CODE")
            row["verdict"] = "REVIEW"
            rec_verdict = "REVIEW"
        else:
            copied, _missing = copy_data_files(wt, dest, files)
            row["copied"] = copied
            # packet-field changes ride along; tools/ diffs are dropped
            ok, msg = cleaner(d.name)
            row["cleanup"] = bool(ok)
            if not ok:
                row["cleanup_note"] = msg
            _mark_lane(st, d.name, "VERIFIED", oxx=oxx, harvest_reason="DATA-ONLY")
            row["verdict"] = "VERIFIED"
            rec_verdict = "VERIFIED"
            _complete_harvested_lane(
                d.name, oxx, _work_for_task(st, d.name) or {"template": tmpl},
                persist=True,
                path=_dest_completions(dest),
                receipt_dir=_dest_receipt_dir(dest),
                dest_root=dest,
            )
        rec = {
            "schema": HARVEST_SCHEMA,
            "task": d.name,
            "slug": row["slug"],
            "classification": classification,
            "action": action,
            "verdict": rec_verdict,
            "reason": reason or row["reason"],
            "files": files,
            "copied": row["copied"],
            "dropped": row["dropped"],
            "matching_receipts": matching_receipts,
            "template": tmpl,
            "oxx": oxx,
            "worktree": row["worktree"],
            "report": row["report"],
            "cleanup": row["cleanup"],
            "command": "harvest",
            "inputs": {"task_dir": str(d), "worktree": row["worktree"]},
            "outputs": {
                "copied": row["copied"],
                "dropped": row["dropped"],
                "verdict": rec_verdict,
            },
            "controls": [
                "never auto-merge CODE",
                "cleanup only DATA-ONLY",
                "drop tools/ diffs on known data-producing templates",
            ],
            "assumptions": [
                "DATA-ONLY = known data-producing template + matching RECEIPT_PATTERN",
                "tools/*.py diffs on those templates are noise, not a deliverable",
                "CODE = unknown template with a non-data path, or known template with no receipt",
            ],
            "reopen_if": "worktree still holds uncopied data files",
            "_evidence": "DERIVED (harvester classification)",
        }
        write_json(out_dir / f"harvest_{d.name}.json", rec)
        row["applied"] = True
        mutated = True
        rows.append(row)
        record_ctl_event(
            oxx, "grok" if rec_verdict == "VERIFIED" else "harvest",
            0.0, grok_lane=d.name,
            opus=rec_verdict == "REVIEW",
            extra={
                "verdict": rec_verdict,
                "template": tmpl,
                "classification": classification,
            },
            persist=do_persist,
        )

    if do_persist and mutated and not dry_run:
        save_state(st)
    return rows


# ---------------------------------------------------------------------------
# run loop (§21/§22) — select, render, gate, maybe launch; harvest reaps
# ---------------------------------------------------------------------------

ACTIVE_OB_RE = re.compile(
    r"^\s*[-*]\s+\*\*(A\d+|ACQ-O\d{3})\*\*\s+(O\d{3})\b",
    re.M,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def launch_repo() -> Path:
    if HAWKING_REPO.is_dir() and (HAWKING_REPO / ".git").exists():
        return HAWKING_REPO
    return REPO


def patient_on_disk(meta: dict) -> bool:
    if meta.get("on_disk"):
        return True
    return str(meta.get("ledger") or "").lower().startswith("on-disk")


def arch_kind(oxx: str, pkt: dict | None, census: dict | None) -> str:
    if census and census.get("is_moe"):
        return "moe"
    kind = str(((pkt or {}).get("architecture") or {}).get("kind") or "").lower()
    if kind in {"moe", "dense", "hybrid"}:
        return kind
    if oxx in MOE_PATIENTS:
        return "moe"
    klass = str(((pkt or {}).get("class") or "")).lower()
    if "hybrid" in klass or "mamba" in klass:
        return "hybrid"
    if "moe" in klass:
        return "moe"
    return "dense"


def has_census(oxx: str) -> bool:
    return census_path(oxx).is_file()


def load_census(oxx: str) -> dict | None:
    p = census_path(oxx)
    if not p.is_file():
        return None
    try:
        return read_json(p)
    except (OSError, json.JSONDecodeError):
        return None


def _unknownish(val) -> bool:
    if val is None or val == "":
        return True
    if isinstance(val, str) and val.upper().startswith("UNKNOWN"):
        return True
    return False


def has_routing(pkt: dict | None) -> bool:
    r = (pkt or {}).get("routing") or {}
    if _unknownish(r.get("entropy")):
        return False
    ev = evidence_class(r.get("_evidence"))
    return ev not in (None, "UNKNOWN")


def has_baseline(pkt: dict | None) -> bool:
    ex = (pkt or {}).get("execution") or {}
    tps = ex.get("baseline_tps")
    if _unknownish(tps):
        tps = ex.get("tps_specimen") or ex.get("tps")
    if _unknownish(tps):
        return False
    ev = evidence_class(ex.get("_evidence"))
    if ev in (None, "UNKNOWN") and _unknownish(ex.get("baseline_tps")):
        # specimen tps still counts as a baseline for sensitivity gating
        return not _unknownish(ex.get("tps_specimen"))
    return True


def missing_sensitivity(pkt: dict | None) -> bool:
    rep = (pkt or {}).get("representation") or {}
    return rep.get("per_organ_sensitivity") in (None, "", {}, [])


def needs_ssm_accounting(pkt: dict | None, kind: str) -> bool:
    if kind != "hybrid":
        return False
    rep = (pkt or {}).get("representation") or {}
    organs = rep.get("organs_bytes_GB") or {}
    if "ssm" in organs:
        return False
    return True


def phase_rank(meta: dict, census_exists: bool) -> int:
    raw = norm_phase(meta.get("phase") or "")
    key = raw.upper().replace("✓", "").strip()
    if key in PHASE_INDEX:
        return PHASE_INDEX[key]
    if census_exists:
        return PHASE_INDEX["CENSUS"]
    if patient_on_disk(meta):
        return PHASE_INDEX["INGEST"]
    return -1


def prereq_ok(template: str, meta: dict, pkt: dict | None, census_exists: bool) -> bool:
    """Obligation is READY when the patient is on disk and phase prereqs hold."""
    if not patient_on_disk(meta):
        return False
    if meta.get("state") in {"BLOCKED", "RETIRED", "ACQUIRING"}:
        return False
    rank = phase_rank(meta, census_exists)
    if template in {
        "external-science-moe", "external-science-dense", "route-map",
        "transfer-control",
    }:
        return census_exists or rank >= PHASE_INDEX["CENSUS"]
    if template == "sensitivity-map":
        return census_exists and has_baseline(pkt)
    if template in GRAVITY_SPEC or template in NX_FLAG:
        return census_exists
    if template.startswith("novelty-"):
        return census_exists
    return False


def weights_dir(oxx: str, pkt: dict | None, census: dict | None) -> str:
    if census and census.get("model_dir"):
        return str(census["model_dir"])
    ident = (pkt or {}).get("identity") or {}
    on_disk = ident.get("on_disk")
    if on_disk:
        return str(on_disk)
    return ""


def sg_lint(contract: Path, repo: Path | None = None) -> tuple[bool, str]:
    """Dry-check ~/.claude-grok/v2/lint.mjs. Never sets SG_OFF."""
    lint = LINT_JS
    if not lint.is_file():
        return False, "ERROR LINT_MISSING: ~/.claude-grok/v2/lint.mjs not found"
    node = str(NODE_BIN) if NODE_BIN.is_file() else "node"
    r = subprocess.run(
        [node, str(lint), str(contract), str(repo or REPO)],
        capture_output=True, text=True,
    )
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    if r.returncode not in (0, None) and out.startswith("ERROR"):
        return False, out
    if out.startswith("ERROR"):
        return False, out
    return True, out


def append_run_log(row: dict, path: Path | None = None) -> None:
    dest = Path(path) if path else RUN_LOG
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def record_ctl_event(patient, event, wall_s=0.0, *, grok_lane=None,
                     opus=False, extra=None, persist=True):
    """Append a compile-economics event. Never raises into the loop."""
    if not persist:
        return None
    oxx = patient or ""
    if not oxx:
        return None
    try:
        return costmodel.record(
            oxx, event, float(wall_s or 0.0),
            grok_lane=grok_lane, opus=bool(opus), extra=extra or {},
        )
    except Exception:
        return None


def parse_active_obligations() -> list[tuple[str, str]]:
    if not LEDGER.is_file():
        return []
    text = LEDGER.read_text()
    idx = text.lower().find("## active obligations")
    chunk = text[idx:] if idx >= 0 else text
    stop = chunk.find("\n## ", 3)
    if stop > 0:
        chunk = chunk[:stop]
    return [(m.group(1), m.group(2)) for m in ACTIVE_OB_RE.finditer(chunk)]


def _seed_by_id() -> dict[str, dict]:
    return {w["id"]: dict(w) for w in SEED_WORK}


def template_for_work(work: dict, meta: dict, pkt: dict | None,
                      census: dict | None) -> str | None:
    """Map a work item onto one of the four run templates, or None (skip).

    Done-ness is NOT inferred from packet fields. Completions is the only
    source; the caller filters with science_done_for_template().
    """
    oxx = work.get("oxx")
    if not oxx:
        return None
    if work.get("status") not in (None, "READY"):
        return None
    kind = work.get("kind") or ""
    if kind in {"acquisition", "false-win-discovery"}:
        return None
    if work.get("template") in TEMPLATES:
        tmpl = work["template"]
        return tmpl if prereq_ok(tmpl, meta, pkt, census is not None) else None
    arch = arch_kind(oxx, pkt, census)
    census_ok = census is not None
    if kind == "router-sensitivity":
        if arch != "moe":
            return None
        tmpl = "external-science-moe"
        return tmpl if prereq_ok(tmpl, meta, pkt, census_ok) else None
    if kind == "representation-discriminator":
        tmpl = "sensitivity-map"
        return tmpl if prereq_ok(tmpl, meta, pkt, census_ok) else None
    if kind == "architecture-first":
        if arch in {"dense", "hybrid"}:
            tmpl = "external-science-dense"
            return tmpl if prereq_ok(tmpl, meta, pkt, census_ok) else None
        if arch == "moe":
            tmpl = "external-science-moe"
            return tmpl if prereq_ok(tmpl, meta, pkt, census_ok) else None
    return None


def _tried_gravity_specs(oxx: str, pkt: dict | None = None) -> set[str]:
    tried: set[str] = set()
    blob = pkt if pkt is not None else load_packet(oxx)
    grav = (blob or {}).get("gravity") or {}
    for key in ("wins", "kills"):
        for item in grav.get(key) or []:
            if isinstance(item, dict) and item.get("spec"):
                tried.add(str(item["spec"]))
            elif isinstance(item, str) and item:
                tried.add(item)
    last = grav.get("last")
    if isinstance(last, dict) and last.get("spec"):
        tried.add(str(last["spec"]))
    if RECEIPT_DIR.is_dir():
        prefix = f"{oxx}_GRAVITY_"
        for path in RECEIPT_DIR.glob(f"{prefix}*.json"):
            spec = path.name[len(prefix):-5]
            if spec:
                tried.add(spec)
    return tried


def pick_aggressive_spec(oxx: str, template: str,
                         census: dict | None = None,
                         pkt: dict | None = None) -> str:
    """Next-highest-EV aggressive/structural spec from candgen. Fallback: table."""
    default = GRAVITY_SPEC.get(template) or "q2-g32-experts"
    klass = "moe" if "moe" in template else (
        "hybrid" if "hybrid" in template else "dense"
    )
    census_d = census if census is not None else load_census(oxx) or {}
    pkt_d = pkt if pkt is not None else load_packet(oxx) or {}
    sens = ((pkt_d.get("representation") or {}).get("per_organ_sensitivity")
            if isinstance(pkt_d, dict) else None)
    policy = load_odyssey_policy()
    try:
        cands = candgen.generate(klass, census_d or {}, sens, policy)
    except (TypeError, ValueError, OSError, json.JSONDecodeError):
        return default
    tried = _tried_gravity_specs(oxx, pkt_d)
    for cand in cands:
        spec = cand.get("spec") if isinstance(cand, dict) else None
        if not spec or spec in tried:
            continue
        tagged = classify_gravity_spec(str(spec))
        klass_c = cand.get("candidate_class") or tagged.get("candidate_class")
        if klass_c in {None, "", "CONVENTIONAL_ANCHOR", "BASELINE"}:
            continue
        return str(spec)
    return default


def _families_tried(oxx: str, pkt: dict | None, entries: list | None) -> list:
    tried: list = []
    grav = (pkt or {}).get("gravity") or {}
    for key in ("wins", "kills"):
        for item in grav.get(key) or []:
            tried.append(item)
    last = grav.get("last")
    if isinstance(last, dict):
        tried.append(last)
    for e in entries or []:
        if e.get("patient_id") != oxx:
            continue
        mech = str(e.get("mechanism_id") or "")
        if mech.startswith("gravity") or is_aggressive_mechanism(mech):
            tried.append(e)
    return tried


def _patient_best_class(oxx: str, pkt: dict | None, entries: list | None) -> str:
    grav = (pkt or {}).get("gravity") or {}
    if grav.get("candidate_class"):
        return str(grav["candidate_class"])
    last = grav.get("last") if isinstance(grav.get("last"), dict) else {}
    if last.get("candidate_class"):
        return str(last["candidate_class"])
    for e in reversed(list(entries or [])):
        if e.get("patient_id") == oxx and e.get("candidate_class"):
            return str(e["candidate_class"])
    if conventional_anchor_exists(oxx, entries):
        return "CONVENTIONAL_ANCHOR"
    return ""


def _patient_target_delta(oxx: str, pkt: dict | None):
    man = manifest_entry(oxx)
    pressure = None
    raw_p = man.get("stored_bpw_pressure")
    try:
        if raw_p is not None and raw_p != "UNKNOWN":
            pressure = float(raw_p)
    except (TypeError, ValueError):
        pressure = None
    if pressure is None:
        zones = load_odyssey_policy().get("target_pressure_zones_bpw") or {}
        try:
            pressure = float(zones.get("pressure") or 2.5)
        except (TypeError, ValueError):
            pressure = 2.5
    stored = None
    grav = (pkt or {}).get("gravity") or {}
    last = grav.get("last") if isinstance(grav.get("last"), dict) else {}
    for src in (last, (pkt or {}).get("representation") or {}):
        if not isinstance(src, dict):
            continue
        for key in ("complete_bpw", "stored_bpw", "best_stored_bpw_eq"):
            if src.get(key) is None:
                continue
            try:
                stored = float(src[key])
                break
            except (TypeError, ValueError):
                continue
        if stored is not None:
            break
    if stored is None:
        return None
    return stored - float(pressure)


def should_novelty_escalate(oxx: str, pkt: dict | None,
                            entries: list | None,
                            state: dict | None = None) -> bool:
    if not aggressive_probe_attempted(oxx, entries):
        return False
    tried = _families_tried(oxx, pkt, entries)
    xfer_done = science_is_done(oxx, "transfer-control", entries) or not reference_sibling(
        oxx, state
    )
    patient = {
        "oxx": oxx,
        "deterministic_search": True,
        "deterministic_exhausted": True,
        "rule_transfer": xfer_done,
        "stages_completed": ["deterministic_search"] + (
            ["rule_transfer"] if xfer_done else []
        ),
    }
    return novelty.should_escalate(
        patient,
        _patient_best_class(oxx, pkt, entries),
        _patient_target_delta(oxx, pkt),
        tried,
        load_odyssey_policy(),
    )


def _frontier_novelty_packet(oxx: str) -> dict:
    pkt = load_packet(oxx) or {}
    recs: list = []
    if RECEIPT_DIR.is_dir():
        for path in sorted(RECEIPT_DIR.glob(f"{oxx}_*.json")):
            try:
                recs.append(read_json(path))
            except (OSError, json.JSONDecodeError):
                continue
    xfer_done = science_is_done(oxx, "transfer-control") or not reference_sibling(oxx)
    patient = {
        "oxx": oxx,
        "kind": arch_kind(oxx, pkt, load_census(oxx)),
        "deterministic_search": True,
        "rule_transfer": xfer_done,
        "stages_completed": ["deterministic_search"] + (
            ["rule_transfer"] if xfer_done else []
        ),
    }
    return novelty.build_packet(
        patient,
        pkt,
        recs,
        RULEBASE if RULEBASE.is_file() else {},
        TRANSFER if TRANSFER.is_file() else {},
        NEGATIVE if NEGATIVE.is_file() else {},
        load_odyssey_policy(),
    )


def _ob_record(work: dict, template: str, *, source: str) -> dict:
    oxx = work["oxx"]
    man = manifest_entry(oxx)
    rec = {
        "id": work.get("id") or f"AUTO-{oxx}-{template}",
        "oxx": oxx,
        "title": work.get("title") or template,
        "status": "READY",
        "info": float(work.get("info") or 0),
        "wall_cost": float(work.get("wall_cost") or 0),
        "gpu_cost": float(work.get("gpu_cost") or 0),
        "opus_cost": float(work.get("opus_cost") or 0),
        "kind": work.get("kind") or template,
        "template": template,
        "mechanism_id": mechanism_for_template(template),
        "model_loading": not str(template).startswith("novelty-"),
        "timing": False,
        "download": False,
        "source": source,
        "reference": man.get("reference_sibling") or TRANSFER_REF.get(oxx),
        "search_class": man.get("search_class"),
        "info_budget": man.get("info_budget"),
        "arch_objective": man.get("arch_objective"),
        "stored_bpw_pressure": man.get("stored_bpw_pressure"),
        "active_bpw_pressure": man.get("active_bpw_pressure"),
        "_evidence": "HYPOTHESIS (§22 ranking; READY if on_disk + phase prereqs)",
    }
    if template in AGGRESSIVE_GRAVITY_TEMPLATES:
        rec["gravity_spec"] = work.get("gravity_spec") or pick_aggressive_spec(oxx, template)
        if rec["gravity_spec"] and rec["gravity_spec"] not in (rec.get("title") or ""):
            rec["title"] = f"{rec['title']} [{rec['gravity_spec']}]"
    elif template in GRAVITY_SPEC:
        rec["gravity_spec"] = GRAVITY_SPEC[template]
    scope = write_scope(rec)
    rec["write_set"] = scope["write_set"]
    rec["exclusive_resources"] = scope["exclusive_resources"]
    if template == "transfer-control":
        rec["reference"] = (
            work.get("reference")
            or man.get("reference_sibling")
            or TRANSFER_REF.get(oxx, "O005")
        )
    return rec


def synthesize_for_patient(oxx: str, meta: dict, pkt: dict | None,
                           census: dict | None, covered: set[tuple],
                           entries: list | None = None) -> list[dict]:
    """Fill gaps the seed queue does not name: on-disk patients still missing science.

    Completions is the only done-check. Packet fields are prerequisites
    (e.g. sensitivity needs a baseline), never completion markers.
    """
    if not patient_on_disk(meta) or meta.get("state") in {
        "BLOCKED", "RETIRED", "ACQUIRING",
    }:
        return []
    if census is None:
        return []
    arch = arch_kind(oxx, pkt, census)
    out = []

    def add(template, info, wall, gpu, title, kind, gravity_spec=None):
        if (oxx, template) in covered:
            return
        if science_done_for_template(oxx, template, entries):
            return
        if not prereq_ok(template, meta, pkt, True):
            return
        work = {
            "id": f"AUTO-{oxx}-{template}",
            "oxx": oxx, "title": title, "info": info,
            "wall_cost": wall, "gpu_cost": gpu, "opus_cost": 0, "kind": kind,
        }
        if gravity_spec:
            work["gravity_spec"] = gravity_spec
        out.append(_ob_record(work, template, source="synthesized"))

    if arch == "moe":
        ref = TRANSFER_REF.get(oxx)
        if (
            ref
            and science_is_done(ref, "external-science", entries)
            and not science_is_done(oxx, "transfer-control", entries)
        ):
            add("transfer-control", 9, 2, 1,
                f"transfer control vs {ref} (route/representation delta)",
                "transfer-control")
        elif not science_done_for_template(oxx, "external-science-moe", entries):
            add("external-science-moe", 10, 2, 1,
                "route/state map + baseline + fast-doctor (external)",
                "router-sensitivity")
    if arch in {"dense", "hybrid"} and not science_is_done(
        oxx, "external-science", entries,
    ):
        wall, gpu = (1, 0) if arch == "hybrid" else (2, 1)
        add("external-science-dense", 8, wall, gpu,
            "baseline TPS + fast-doctor"
            + (" + SSM-state-vs-KV" if arch == "hybrid" else ""),
            "architecture-first")
    if has_baseline(pkt) and not science_is_done(oxx, "sensitivity-map", entries):
        add("sensitivity-map", 10, 2, 1,
            "per-organ / per-expert Doctor sensitivity",
            "representation-discriminator")
    if arch == "moe":
        add("gravity-moe", 8, 2, 1,
            "modest gravity q3-g32-experts (SPECIMEN)",
            "gravity")
        agg = pick_aggressive_spec(oxx, "gravity-aggressive-moe", census, pkt)
        add("gravity-aggressive-moe", 8, 2, 1,
            f"aggressive gravity {agg} (SPECIMEN; anti-complacency)",
            "gravity", gravity_spec=agg)
        add("nx-gather-moe", 7, 2, 1,
            "NX gather accounting (selected-expert bytes/token)",
            "nx")
    elif arch == "hybrid":
        add("gravity-hybrid", 8, 2, 1,
            "modest gravity q4-g64-attn-mlp (protect SSM/norm, SPECIMEN)",
            "gravity")
        agg = pick_aggressive_spec(oxx, "gravity-aggressive-hybrid", census, pkt)
        add("gravity-aggressive-hybrid", 8, 2, 1,
            f"aggressive gravity {agg} (protect SSM/norm, SPECIMEN)",
            "gravity", gravity_spec=agg)
        add("nx-state-hybrid", 7, 2, 1,
            "NX state accounting (SSM-vs-KV residency)",
            "nx")
    else:
        add("gravity-dense", 8, 2, 1,
            "modest gravity q4-g64 (SPECIMEN)",
            "gravity")
        agg = pick_aggressive_spec(oxx, "gravity-aggressive-dense", census, pkt)
        add("gravity-aggressive-dense", 8, 2, 1,
            f"aggressive gravity {agg} (SPECIMEN; anti-complacency)",
            "gravity", gravity_spec=agg)
        add("nx-dense", 7, 2, 1,
            "NX dense floor (full-weight-sweep bytes/token)",
            "nx")
    # Aggressive descent ladder: deep, per-spec deterministic work so the box
    # keeps running and the sub-frontier is mapped. Skips the single canonical
    # aggressive spec (already added above) and any rung already sealed.
    ladder_tmpl = GRAVITY_LADDER_TEMPLATE.get(arch)
    if ladder_tmpl and prereq_ok(ladder_tmpl, meta, pkt, True):
        picked = pick_aggressive_spec(oxx, ladder_tmpl, census, pkt)
        for spec in GRAVITY_LADDER.get(arch, []):
            if spec == picked:
                continue  # canonical aggressive obligation covers this spec
            mech = f"gravity-{spec}"
            if science_is_done(oxx, mech, entries):
                continue
            rec = _ob_record({
                "id": f"AUTO-{oxx}-gravity-{spec}",
                "oxx": oxx,
                "title": f"gravity descent {spec} (SPECIMEN; ladder)",
                "info": 6, "wall_cost": 2, "gpu_cost": 1, "opus_cost": 0,
                "kind": "gravity", "gravity_spec": spec,
            }, ladder_tmpl, source="ladder")
            rec["mechanism_id"] = mech
            rec["mechanism"] = mech
            # receipt-only scope so same-patient rungs run concurrently
            rec["ladder"] = True
            scope = write_scope(rec)
            rec["write_set"] = scope["write_set"]
            rec["exclusive_resources"] = scope["exclusive_resources"]
            out.append(rec)
    if should_novelty_escalate(oxx, pkt, entries):
        for lane in novelty.LANES:
            add(f"novelty-{lane}", 9, 1, 0,
                f"frontier novelty / {lane} (Grok; nonconventional)",
                "novelty")
    return out


def select_ready_obligations(state: dict | None = None,
                             completions=None) -> list[dict]:
    """READY science work, ranked by existing value(). Acquisition is not a template.

    Completions is the only done-check: a terminal (patient, mechanism) with
    reopen_if not satisfied is never selected.
    """
    st = state if state is not None else ensure_state()
    patients = {p["oxx"]: p for p in st.get("patients") or []}
    cache_pkt: dict[str, dict | None] = {}
    cache_cen: dict[str, dict | None] = {}
    entries = _completions_entries(completions)

    def pkt_of(oxx: str):
        if oxx not in cache_pkt:
            cache_pkt[oxx] = load_packet(oxx)
        return cache_pkt[oxx]

    def cen_of(oxx: str):
        if oxx not in cache_cen:
            cache_cen[oxx] = load_census(oxx)
        return cache_cen[oxx]

    selected = []
    seen_ids = set()
    covered: set[tuple] = set()
    seeds = _seed_by_id()
    flying = {
        (w.get("oxx"), w.get("template"))
        for w in st.get("work") or []
        if w.get("status") in {"RUNNING", "REVIEW"} and w.get("oxx") and w.get("template")
    }

    def consider(work: dict, source: str) -> None:
        wid = work.get("id")
        if not wid or wid in seen_ids or wid == "A6":
            return
        if work.get("status") not in (None, "READY"):
            return
        oxx = work.get("oxx")
        meta = patients.get(oxx) or patient_meta(oxx, st)
        tmpl = template_for_work(work, meta, pkt_of(oxx), cen_of(oxx))
        if not tmpl:
            return
        if science_done_for_template(oxx, tmpl, entries):
            return
        if (oxx, tmpl) in flying:
            return
        seen_ids.add(wid)
        covered.add((oxx, tmpl))
        selected.append(_ob_record(work, tmpl, source=source))

    for w in st.get("work") or []:
        consider(w, "state")
    for oid, oxx in parse_active_obligations():
        if oid in seen_ids:
            continue
        base = seeds.get(oid) or {
            "id": oid, "oxx": oxx, "title": oid, "status": "READY",
            "info": 5, "wall_cost": 2, "gpu_cost": 1, "opus_cost": 0,
            "kind": "architecture-first",
        }
        consider(base, "ledger")
    for p in st.get("patients") or []:
        oxx = p.get("oxx")
        if not oxx:
            continue
        pkt = pkt_of(oxx)
        nxt = (pkt or {}).get("next") or []
        if isinstance(nxt, str):
            nxt = [nxt]
        for line in nxt:
            text = str(line)
            # packet-next is advisory; synthesis below covers the science
            if "sensitivity" in text.lower() and (oxx, "sensitivity-map") not in covered:
                consider({
                    "id": f"NEXT-{oxx}-SENS", "oxx": oxx, "title": text[:80],
                    "status": "READY", "info": 10, "wall_cost": 2, "gpu_cost": 1,
                    "opus_cost": 0, "kind": "representation-discriminator",
                }, "packet-next")
        selected.extend(synthesize_for_patient(
            oxx, p, pkt, cen_of(oxx), covered, entries=entries,
        ))
        for rec in selected:
            covered.add((rec["oxx"], rec["template"]))

    # de-dupe by (oxx, mechanism_id), keep highest value. Keying on mechanism
    # (not template) is what lets many descent-ladder rungs — all sharing one
    # gravity-aggressive-* template but each its own per-spec mechanism —
    # coexist instead of collapsing to a single obligation.
    def _rec_done(rec: dict) -> bool:
        mech = rec.get("mechanism_id")
        tmpl = rec.get("template") or ""
        if mech and mech != mechanism_for_template(tmpl):
            # per-spec (ladder) mechanism: check it directly, NOT the template's
            return science_is_done(rec["oxx"], mech, entries)
        return science_done_for_template(rec["oxx"], tmpl, entries)

    best: dict[tuple, dict] = {}
    for rec in selected:
        if _rec_done(rec):
            continue
        # flying is per-patient/template: a rung in flight holds the packet lock,
        # so no sibling rung of the same template is selected until it completes.
        if (rec["oxx"], rec["template"]) in flying:
            continue
        key = (rec["oxx"], rec.get("mechanism_id") or rec["template"])
        prev = best.get(key)
        if prev is None or value(rec) > value(prev):
            best[key] = rec
    out = list(best.values())
    out.sort(key=lambda w: (-value(w), 0 if w.get("source") == "state" else 1, w["oxx"], w["id"]))
    return out


def auto_contract_path(oxx: str, template: str, auto_dir: Path | None = None) -> Path:
    return (auto_dir or AUTO_DIR) / f"{oxx.lower()}_{template}.md"


def _scope_block(writes: list[str], reads: list[str], verify_path: str,
                 verify_cmd: str) -> str:
    lines = ["## SCOPE"]
    for w in writes:
        lines.append(f"WRITE {w}")
    lines.append("READ " + ", ".join(reads))
    lines.append(
        f"VERIFY {verify_path} by running the unfenced command below; "
        "must pass, exit 0."
    )
    lines.append(verify_cmd)
    lines.append("Do not touch Genesis state or tools/odyssey/.")
    return "\n".join(lines)


def _patient_facts(oxx: str) -> dict:
    meta = patient_meta(oxx)
    pkt = load_packet(oxx) or {}
    census = load_census(oxx) or {}
    kind = arch_kind(oxx, pkt, census)
    weights = weights_dir(oxx, pkt, census)
    ident = pkt.get("identity") or {}
    arch = pkt.get("architecture") or {}
    man = manifest_entry(oxx)
    source = (
        man.get("canonical_source")
        or ident.get("source_repo")
        or meta.get("source")
        or ""
    )
    return {
        "oxx": oxx,
        "meta": meta,
        "pkt": pkt,
        "census": census,
        "kind": kind,
        "weights": weights,
        "model": meta.get("model") or ident.get("source_repo") or man.get("model") or oxx,
        "source": source,
        "canonical_source": man.get("canonical_source") or source,
        "arch_name": arch.get("arch") or census.get("arch") or "",
        "layers": arch.get("layers") or (census.get("config") or {}).get("num_hidden_layers"),
        "experts": arch.get("experts") or (census.get("config") or {}).get("num_experts"),
        "experts_per_tok": arch.get("experts_per_tok") or (census.get("config") or {}).get("num_experts_per_tok"),
        "total_params": arch.get("total_params") or census.get("total_params"),
        "organs": (pkt.get("representation") or {}).get("organs_bytes_GB")
                  or {k: round(v / 1e9, 2) for k, v in (census.get("organs_bytes") or {}).items() if v},
        "packet_rel": f"workspace/campaign/odyssey/patients/{oxx}/ODYSSEY_PATIENT_{oxx}.json",
        "census_rel": f"workspace/campaign/odyssey/patients/{oxx}/census.json",
        "receipt_rel": f"receipts/odyssey-i/{oxx}_EXTERNAL.json",
        "search_class": man.get("search_class"),
        "info_budget": man.get("info_budget"),
        "arch_objective": man.get("arch_objective"),
        "stored_bpw_pressure": man.get("stored_bpw_pressure"),
        "active_bpw_pressure": man.get("active_bpw_pressure"),
        "doctor_bar": man.get("doctor_bar"),
        "reference_sibling": man.get("reference_sibling") or TRANSFER_REF.get(oxx),
    }


def _runner_cmd(oxx: str, weights: str, receipt: str, packet: str,
                route_tokens: int, extra: str = "") -> str:
    w = weights or f"<weights-from-{oxx}-census>"
    cmd = (
        f"python3 tools/odyssey_patient_runner.py --oxx {oxx} "
        f"--weights {w} --runtime mlx --route-tokens {route_tokens} "
        f"--out {receipt} --packet {packet}"
    )
    if extra:
        cmd += " " + extra
    return cmd


def render_external_science_moe(f: dict) -> str:
    oxx = f["oxx"]
    organs = ", ".join(f"{k}={v}GB" for k, v in (f["organs"] or {}).items())
    verify = _runner_cmd(oxx, f["weights"], f["receipt_rel"], f["packet_rel"], 512)
    body = f"""# DELEGATION — {oxx} EXTERNAL SCIENCE (MoE; gate profile: MLX/Metal)

Patient {oxx} = `{f['source'] or f['model']}` ({f['kind']}; {f['arch_name']}),
on disk at `{f['weights']}`. Repo: `/Users/scammermike/Downloads/hawking`.
This is the O005-style runner: route map + baseline TPS + fast-Doctor.

Native Hawking `load_engine` is not the path. Use `tools/odyssey_patient_runner.py`
(mlx_lm EXTERNAL SPECIMEN). SPECIMEN labels everywhere; this is NOT BASE_TRUE_TPS (§14).

Census (MEASURED): layers={f['layers']} experts={f['experts']} topk={f['experts_per_tok']}
total_params={f['total_params']} organs: {organs or 'see census.json'}.

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ {f['census_rel']}
READ {f['packet_rel']}
READ workspace/campaign/odyssey/contracts/o005_external_science.md

Call worker_gate.observe()/gate() before load (the runner already does this). Abort on REFUSE.
If REFUSE, convert to 4-bit mlx and LABEL `quant=4bit-mlx`; prefer bf16 if admitted.

## BUILD
{RUNNER_DO_NOT_EDIT}
Reuse tools/odyssey_patient_runner.py. Do not start from scratch.
If this patient is multimodal, tap the language-MoE router (skip the vision tower).
If the runner assumes Qwen3-MoE config assertions, keep them as recorded pass/fail —
do not fail the receipt solely because a Qwen-specific assertion is N/A; label N/A.

Outputs:
- {f['receipt_rel']} with quant, tps_specimen, ttft, route{{entropy_avg,entropy_max,cold_experts,top16_mass_pct,most_popular_share,transition_stability}}, doctor{{battery,refusals,seal_ref}}.
- Refresh {f['packet_rel']} routing + execution + doctor from the receipt (schema-valid).

## ACCEPTANCE
- {f['receipt_rel']} exists with route.entropy_avg>0 and doctor.battery. Must pass, exit 0.
- {f['packet_rel']} still schema-valid after the refresh.

{_scope_block(
    ["receipts/odyssey-i/", f"workspace/campaign/odyssey/patients/{oxx}/"],
    ["tools/odyssey_patient_runner.py", "tools/worker_gate.py", f['census_rel'], f['packet_rel']],
    f['receipt_rel'],
    verify,
)}
"""
    return body


def render_external_science_dense(f: dict) -> str:
    oxx = f["oxx"]
    hybrid = f["kind"] == "hybrid"
    organs = ", ".join(f"{k}={v}GB" for k, v in (f["organs"] or {}).items())
    verify = _runner_cmd(oxx, f["weights"], f["receipt_rel"], f["packet_rel"], 0, extra="--skip-route")
    ssm = ""
    if hybrid:
        ssm = (
            "Also measure hybrid SSM-state-vs-KV byte accounting across ctx "
            "(short / moderate / long). Census currently buckets Mamba tensors as "
            "`other`; write an `ssm` organ bucket and state-vs-KV bytes into the packet "
            "representation + execution. No route map."
        )
    body = f"""# DELEGATION — {oxx} EXTERNAL SCIENCE (dense/hybrid; gate profile: MLX/Metal)

Patient {oxx} = `{f['source'] or f['model']}` ({f['kind']}; {f['arch_name']}),
on disk at `{f['weights']}`. Repo: `/Users/scammermike/Downloads/hawking`.
Baseline TPS + fast-Doctor{(' + SSM-state-vs-KV' if hybrid else '')}. NO route map.

This is a DENSE/HYBRID path. There is no MoE router. The runner already skips
the route tap via `--route-tokens 0` and `--skip-route` (no-ops RouteRecorder
when no layer has gate+switch_mlp, writes `route_skipped=true` instead of
failing). Do not reimplement that flag.

SPECIMEN labels everywhere; mlx TPS is NOT BASE_TRUE_TPS (§14, §60 foreign-runtime).

Census (MEASURED): layers={f['layers']} total_params={f['total_params']}
organs: {organs or 'see census.json'}.

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ {f['census_rel']}
READ {f['packet_rel']}

Call worker_gate.observe()/gate() before load. Abort on REFUSE; 4-bit fallback is allowed
and must be labelled.

## BUILD
{RUNNER_DO_NOT_EDIT}
Reuse tools/odyssey_patient_runner.py `--skip-route`. {ssm}
Keep the model loaded once (one load, memory-safe). Do not delete the canonical weights.

Outputs:
- {f['receipt_rel']} with quant, tps_specimen, ttft, doctor{{battery,refusals,seal_ref}}, route_skipped=true{', ssm_vs_kv{{ctx,state_bytes,kv_bytes}}' if hybrid else ''}.
- Refresh {f['packet_rel']} execution + doctor{(' + representation.ssm' if hybrid else '')}.

## ACCEPTANCE
- {f['receipt_rel']} exists with tps_specimen, ttft, doctor.battery, route_skipped true. Must pass, exit 0.
- {f['packet_rel']} still schema-valid.

{_scope_block(
    ["receipts/odyssey-i/", f"workspace/campaign/odyssey/patients/{oxx}/"],
    ["tools/odyssey_patient_runner.py", "tools/worker_gate.py", f['census_rel'], f['packet_rel']],
    f['receipt_rel'],
    verify,
)}
"""
    return body


def render_sensitivity_map(f: dict) -> str:
    oxx = f["oxx"]
    organs = list((f["organs"] or {}).keys()) or ["embed", "attn", "mlp_dense", "lm_head"]
    organ_list = ", ".join(organs)
    receipt = f"receipts/odyssey-i/{oxx}_SENSITIVITY.json"
    verify = (
        f"python3 tools/odyssey_patient_runner.py --oxx {oxx} "
        f"--weights {f['weights'] or '<weights>'} --runtime mlx --route-tokens 0 "
        f"--out {receipt} --packet {f['packet_rel']} --sensitivity"
    )
    body = f"""# DELEGATION — {oxx} PER-ORGAN / PER-EXPERT SENSITIVITY (§17)

Patient {oxx} = `{f['source'] or f['model']}` ({f['kind']}; {f['arch_name']}),
on disk at `{f['weights']}`. Repo: `/Users/scammermike/Downloads/hawking`.
A baseline already exists on this patient. Measure Doctor capability delta when
each organ (and each expert, if MoE) is zeroed or rounded.

Organs from census (MEASURED): {organ_list}.

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ tools/doctor_seal.py
READ {f['census_rel']}
READ {f['packet_rel']}

Call worker_gate.observe()/gate() before load. Abort on REFUSE.

## BUILD
{RUNNER_DO_NOT_EDIT}
Reuse tools/odyssey_patient_runner.py `--sensitivity`. After the fast battery baseline:
1. For each organ in {{{organ_list}}}, zero (and separately round-to-zero-bpw / 8-bit round) that organ.
2. Re-run the same fast battery + refusal controls.
3. Record capability delta vs the unablated battery (hits, refusals, seal verdict).
For MoE patients, also ablate a hot expert and a random expert and record per-expert delta.
Non-MoE: skip the expert loop; skip the route tap (`--skip-route` / `--route-tokens 0`).

Write {receipt} and fill representation.per_organ_sensitivity (and
per_expert_sensitivity when MoE) on {f['packet_rel']}. Label every delta MEASURED.

Do not delete canonical weights. Keep one load if possible; if reload is required,
re-admit via worker_gate each time.

## ACCEPTANCE
- {receipt} exists with per_organ_sensitivity entries for each named organ and a
  baseline battery. Must pass, exit 0.
- {f['packet_rel']} representation.per_organ_sensitivity is non-null and schema-valid.

{_scope_block(
    ["receipts/odyssey-i/", f"workspace/campaign/odyssey/patients/{oxx}/"],
    ["tools/odyssey_patient_runner.py", "tools/worker_gate.py", "tools/doctor_seal.py", f['census_rel'], f['packet_rel']],
    receipt,
    verify,
)}
"""
    return body


def render_transfer_control(f: dict, reference: str) -> str:
    oxx = f["oxx"]
    ref = _patient_facts(reference)
    receipt = f"receipts/odyssey-i/{oxx}_TRANSFER.json"
    verify = _runner_cmd(oxx, f["weights"], f["receipt_rel"], f["packet_rel"], 512)
    body = f"""# DELEGATION — {oxx} TRANSFER CONTROL vs {reference} (§41)

Patient {oxx} = `{f['source'] or f['model']}` ({f['kind']}; {f['arch_name']}),
on disk at `{f['weights']}`. Reference {reference} = `{ref['source'] or ref['model']}`
({ref['kind']}; {ref['arch_name']}). Repo: `/Users/scammermike/Downloads/hawking`.

Run the O005-style runner on the sibling, then diff route/representation against
the named reference and write a transfer-matrix delta.

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ {f['census_rel']}
READ {f['packet_rel']}
READ {ref['census_rel']}
READ {ref['packet_rel']}
READ receipts/odyssey-i/{reference}_EXTERNAL.json
READ workspace/campaign/odyssey/TRANSFER_MATRIX.json

Call worker_gate.observe()/gate() before load. Abort on REFUSE.

## BUILD
{RUNNER_DO_NOT_EDIT}
Reuse tools/odyssey_patient_runner.py on {oxx} (route map + baseline + fast-Doctor).
If {oxx} is multimodal, tap the language-MoE router; skip the vision tower.
Then diff against {reference}:
- route entropy / cold-expert count / top16 mass / transition stability
- organs_bytes_GB and stored_bpw
- doctor battery delta
Write {receipt} with `reference={reference}`, `delta`, and a `transfer_cells` block
mapping each GRAVITY_RULEBASE rule id to one of TRANSFERRED_UNCHANGED / RETUNED /
ARCHITECTURE_SPECIFIC / PATIENT_SPECIFIC / FAILED / HARMFUL / NOT_TESTED.
Merge those cells into workspace/campaign/odyssey/TRANSFER_MATRIX.json for {oxx}
(do not blank other patients). Refresh {f['packet_rel']} transfer + routing + execution.

## ACCEPTANCE
- {f['receipt_rel']} exists (sibling external science) AND {receipt} exists with
  reference, delta, transfer_cells. Must pass, exit 0.
- workspace/campaign/odyssey/TRANSFER_MATRIX.json has non-NOT_TESTED cells for {oxx}.

{_scope_block(
    ["receipts/odyssey-i/",
     f"workspace/campaign/odyssey/patients/{oxx}/",
     "workspace/campaign/odyssey/TRANSFER_MATRIX.json"],
    ["tools/odyssey_patient_runner.py", "tools/worker_gate.py",
     f['census_rel'], f['packet_rel'], ref['census_rel'], ref['packet_rel'],
     "workspace/campaign/odyssey/TRANSFER_MATRIX.json"],
    receipt,
    verify,
)}
"""
    return body


def render_gravity(f: dict, template: str, spec: str | None = None) -> str:
    oxx = f["oxx"]
    if template in AGGRESSIVE_GRAVITY_TEMPLATES:
        spec = spec or pick_aggressive_spec(oxx, template, f.get("census"), f.get("pkt"))
    spec = spec or GRAVITY_SPEC.get(template) or "q4-g64"
    receipt = f"receipts/odyssey-i/{oxx}_GRAVITY_{spec}.json"
    extra = f"--gravity {spec}"
    if template not in {"gravity-moe", "gravity-aggressive-moe"}:
        extra = f"--skip-route --gravity {spec}"
    verify = (
        f"python3 tools/odyssey_patient_runner.py --oxx {oxx} "
        f"--weights {f['weights'] or '<weights>'} --runtime mlx "
        f"--out {receipt} --packet {f['packet_rel']} {extra}"
    )
    protect = ""
    aggressive = template in AGGRESSIVE_GRAVITY_TEMPLATES
    if template == "gravity-hybrid":
        protect = (
            "Protect SSM/conv/norm at full precision; quantize attn+MLP only "
            "(`q4-g64-attn-mlp`)."
        )
    elif template == "gravity-moe":
        protect = (
            "Experts → 3-bit group32; attention/router → 4-bit group64; norms full "
            "(`q3-g32-experts`)."
        )
    elif template == "gravity-aggressive-moe":
        protect = (
            f"AGGRESSIVE candgen spec `{spec}` (experts low-bit; attention/router "
            "protected; norms full). candidate_class from grammar. "
            "Complete bpw (payload+scales+biases+metadata); record nominal_bits AND "
            "complete_bpw. On Doctor fail: failure_localization naming the organ; "
            "do NOT globally retreat."
        )
    elif template == "gravity-aggressive-hybrid":
        protect = (
            f"AGGRESSIVE candgen spec `{spec}`: attn+mlp (+embed/lm_head); protect "
            "SSM/conv/norm full. candidate_class from grammar. "
            "Complete bpw (payload+scales+biases+metadata). On fail: localize, do not "
            "globally retreat."
        )
    elif template == "gravity-aggressive-dense":
        protect = (
            f"AGGRESSIVE candgen spec `{spec}`. candidate_class from grammar. "
            "Complete bpw (payload+scales+biases+metadata). On fail: localize, do not "
            "globally retreat."
        )
    else:
        protect = "Uniform 4-bit group64 (`q4-g64`)."
    kind_label = "AGGRESSIVE GRAVITY" if aggressive else "MODEST GRAVITY"
    body = f"""# DELEGATION — {oxx} {kind_label} ({spec}; gate profile: MLX/Metal)

Patient {oxx} = `{f['source'] or f['model']}` ({f['kind']}; {f['arch_name']}),
on disk at `{f['weights']}`. Repo: `/Users/scammermike/Downloads/hawking`.
One bounded Gravity candidate (steer S004 anti-complacency). SPECIMEN-labelled mlx quant; this
is NOT a Hawking NX win (§15).

{protect}

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ tools/doctor_seal.py
READ {f['census_rel']}
READ {f['packet_rel']}
READ {f['receipt_rel']}

Call worker_gate.observe()/gate() before load. Abort on REFUSE.

## BUILD
{RUNNER_DO_NOT_EDIT}
Reuse tools/odyssey_patient_runner.py `--gravity {spec}`. One candidate, do not sweep.
Reload the quantized model, run the SAME fast-Doctor battery + refusal controls.
Measure stored_bytes, stored_bpw (bytes*8/params), complete_bpw (payload+scales+biases+
metadata+headers), nominal_bits, active_bytes_per_token + active_bpw
(census active-param split for MoE) and battery/refusal DELTA vs {f['receipt_rel']}.
Write {receipt} (schema odyssey.patient.gravity.v1): spec, stored/active bpw, complete_bpw,
nominal_bits, candidate_class, conventionality, battery, delta_hits, doctor_seal,
SPECIMEN + quant caveat, verdict CANDIDATE_PASS if delta_hits>=-1 else DEGRADED
(and failure_localization naming the responsible organ — do not globally retreat).
Refresh {f['packet_rel']} gravity.wins/kills + candidate_class/conventionality.
Do not delete canonical weights.

## ACCEPTANCE
- {receipt} exists with stored_bpw < 16, complete_bpw, nominal_bits, candidate_class,
  active_bpw, battery, delta vs baseline, SPECIMEN label. Must pass, exit 0.
- {f['packet_rel']} still schema-valid.

{_scope_block(
    ["receipts/odyssey-i/",
     f"workspace/campaign/odyssey/patients/{oxx}/"],
    ["tools/odyssey_patient_runner.py", "tools/worker_gate.py", "tools/doctor_seal.py",
     f['census_rel'], f['packet_rel'], f['receipt_rel']],
    receipt,
    verify,
)}
"""
    return body


def render_nx(f: dict, template: str) -> str:
    oxx = f["oxx"]
    flag = NX_FLAG[template]
    receipt = f"receipts/odyssey-i/{NX_RECEIPT[template].format(oxx=oxx)}"
    extra = flag
    if template != "nx-gather-moe":
        extra = f"--skip-route {flag}"
    verify = (
        f"python3 tools/odyssey_patient_runner.py --oxx {oxx} "
        f"--weights {f['weights'] or '<weights>'} --runtime mlx "
        f"--out {receipt} --packet {f['packet_rel']} {extra}"
    )
    if template == "nx-gather-moe":
        build = (
            "MoE `--nx-gather`: from the router over N tokens, compute THEORETICAL "
            "selected-expert bytes/token = topk/n_experts × expert_body_bytes; contrast "
            "with full-expert-body bytes and the dense-MLP-equivalent; report the ratio "
            "(the NX opportunity). Note whether mlx actually gathers or densely computes."
        )
        schema = "odyssey.patient.nx.v1"
    elif template == "nx-state-hybrid":
        build = (
            "Hybrid `--nx-state`: reuse SSM-vs-KV accounting already on the packet/"
            "external receipt; frame fixed-state residency as the NX lever. Emit "
            "state_bytes vs kv_bytes across ctx."
        )
        schema = "odyssey.patient.nx.v1"
    else:
        build = (
            "Dense `--nx-dense`: report full-weight-sweep bytes/token as the dense NX "
            "floor and note there is no sparsity lever."
        )
        schema = "odyssey.patient.nx.v1"
    body = f"""# DELEGATION — {oxx} NX ACCOUNTING ({flag}; gate profile: MLX/Metal)

Patient {oxx} = `{f['source'] or f['model']}` ({f['kind']}; {f['arch_name']}),
on disk at `{f['weights']}`. Repo: `/Users/scammermike/Downloads/hawking`.
Bounded NX/execution attempt (steer S002). ACCOUNTING + minimal-primitive-design,
not a full Rust runtime (§14). Label DERIVED/MEASURED.

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ {f['census_rel']}
READ {f['packet_rel']}

Call worker_gate.observe()/gate() before load. Abort on REFUSE.

## BUILD
{RUNNER_DO_NOT_EDIT}
Reuse tools/odyssey_patient_runner.py {flag}. {build}
Write {receipt} (schema {schema}) and refresh {f['packet_rel']} nx.
Do not delete canonical weights. Never call this a Hawking NX win.

## ACCEPTANCE
- {receipt} exists with the theoretical-vs-measured (or state-vs-KV / dense-floor)
  accounting and §18 labels. Must pass, exit 0.
- {f['packet_rel']} still schema-valid.

{_scope_block(
    ["receipts/odyssey-i/",
     f"workspace/campaign/odyssey/patients/{oxx}/"],
    ["tools/odyssey_patient_runner.py", "tools/worker_gate.py",
     f['census_rel'], f['packet_rel']],
    receipt,
    verify,
)}
"""
    return body


RENDERERS = {
    "external-science-moe": lambda f, ob: render_external_science_moe(f),
    "external-science-dense": lambda f, ob: render_external_science_dense(f),
    "route-map": lambda f, ob: render_external_science_moe(f),
    "sensitivity-map": lambda f, ob: render_sensitivity_map(f),
    "transfer-control": lambda f, ob: render_transfer_control(
        f, ob.get("reference") or TRANSFER_REF.get(ob["oxx"], "O005"),
    ),
    "gravity-moe": lambda f, ob: render_gravity(f, "gravity-moe", ob.get("gravity_spec")),
    "gravity-dense": lambda f, ob: render_gravity(f, "gravity-dense", ob.get("gravity_spec")),
    "gravity-hybrid": lambda f, ob: render_gravity(f, "gravity-hybrid", ob.get("gravity_spec")),
    "gravity-aggressive-moe": lambda f, ob: render_gravity(
        f, "gravity-aggressive-moe", ob.get("gravity_spec"),
    ),
    "gravity-aggressive-dense": lambda f, ob: render_gravity(
        f, "gravity-aggressive-dense", ob.get("gravity_spec"),
    ),
    "gravity-aggressive-hybrid": lambda f, ob: render_gravity(
        f, "gravity-aggressive-hybrid", ob.get("gravity_spec"),
    ),
    "nx-gather-moe": lambda f, ob: render_nx(f, "nx-gather-moe"),
    "nx-state-hybrid": lambda f, ob: render_nx(f, "nx-state-hybrid"),
    "nx-dense": lambda f, ob: render_nx(f, "nx-dense"),
}


def render_novelty_lane(f: dict, ob: dict) -> str:
    oxx = f["oxx"]
    lane = (ob.get("template") or "").removeprefix("novelty-")
    packet = _frontier_novelty_packet(oxx)
    dest_dir = f.get("_auto_dir")
    novelty.render_lane_contracts(packet, auto_dir=dest_dir)
    path = Path(dest_dir or AUTO_DIR) / f"{oxx.lower()}_novelty-{lane}.md"
    if path.is_file():
        return path.read_text()
    return novelty._lane_contract_text(packet, lane)  # noqa: SLF001 — module API


for _nov in NOVELTY_TEMPLATES:
    RENDERERS[_nov] = lambda f, ob, _t=_nov: render_novelty_lane(f, ob)


def render_contract(ob: dict, auto_dir: Path | None = None) -> Path:
    template = ob["template"]
    if template not in RENDERERS:
        raise ValueError(f"unknown template {template}")
    facts = _patient_facts(ob["oxx"])
    facts["_auto_dir"] = auto_dir or AUTO_DIR
    text = RENDERERS[template](facts, ob)
    dest = auto_contract_path(ob["oxx"], template, auto_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    return dest


def call_worker_gate(observe_fn=None, gate_fn=None) -> dict:
    obs_fn = observe_fn or worker_gate.observe
    g_fn = gate_fn or worker_gate.gate
    try:
        obs = obs_fn()
        g = g_fn(obs)
        g = dict(g)
        g.setdefault("decision", "REFUSE")
        return g
    except Exception as exc:
        return {
            "decision": "REFUSE",
            "note": f"worker_gate failed: {exc}",
            "reasons": [str(exc)],
            "_evidence": "INFERRED (observe/gate raised)",
        }


SCIENCE_TASK_RE = re.compile(r"^odyssey-o\d{3}-", re.I)


def _lane_still_running(w: dict, now_epoch: float, pid_alive_fn=None) -> bool:
    """RUNNING and (subprocess PID live | grok status) within timeout."""
    if w.get("status") != "RUNNING":
        return False
    timeout_s = float(w.get("timeout_s") or lane_timeout_s())
    started = _started_epoch(w)
    if started is not None and (float(now_epoch) - float(started)) > timeout_s:
        return False
    if w.get("kind") == "subprocess":
        alive_fn = pid_alive_fn or _pid_alive
        return bool(w.get("pid")) and bool(alive_fn(w.get("pid")))
    return True


def odyssey_running_ids(
    state: dict,
    now_epoch: float | None = None,
    pid_alive_fn=None,
) -> set[str]:
    """Concurrent science lanes: live subprocess PIDs + live grok-novelty.

    Age-capped. Dead subprocess PIDs do not consume --max-lanes.
    The controller itself may run as odyssey-autonomous-loop-*; that is not a
    patient science lane and must not consume --max-lanes.
    """
    now = float(now_epoch) if now_epoch is not None else time.time()
    ids: set[str] = set()
    for name in live_odyssey_lanes():
        if SCIENCE_TASK_RE.match(name):
            ids.add(name)
    for w in state.get("work") or []:
        if not _lane_still_running(w, now, pid_alive_fn=pid_alive_fn):
            continue
        ids.add(str(w.get("task") or w.get("id") or w.get("pid")))
    return ids


def grok_argv(task: str, contract: Path, *, model_loading: bool,
              repo: Path | None = None) -> list[str]:
    cmd = [
        str(GROK_BIN), "delegate",
        "--task", task,
        "--contract", str(contract),
        "--repo", str(repo or launch_repo()),
        "--background",
    ]
    if model_loading:
        cmd.extend(["--profile", "gate"])
    return cmd


def default_launch(task: str, contract: Path, *, model_loading: bool,
                   repo: Path | None = None) -> tuple[int, str, str]:
    cmd = grok_argv(task, contract, model_loading=model_loading, repo=repo)
    r = subprocess.run(cmd, capture_output=True, text=True)
    stdout, stderr = r.stdout or "", r.stderr or ""
    task_id = None
    for line in stdout.splitlines()[::-1]:
        t = line.strip()
        if t.startswith("odyssey-") or t.startswith(task):
            task_id = t
            break
    return r.returncode, task_id or "", (stdout + "\n" + stderr).strip()


def _mark_running(state: dict, ob: dict, task_id: str, contract: str,
                  started: str) -> None:
    found = False
    for w in state.setdefault("work", []):
        if w.get("id") == ob["id"]:
            w["status"] = "RUNNING"
            w["task"] = task_id
            w["started"] = started
            w["contract"] = contract
            w["template"] = ob["template"]
            found = True
            break
    if not found:
        state["work"].append({
            "id": ob["id"], "oxx": ob["oxx"], "title": ob["title"],
            "status": "RUNNING", "info": ob["info"], "wall_cost": ob["wall_cost"],
            "gpu_cost": ob["gpu_cost"], "opus_cost": ob["opus_cost"],
            "kind": ob.get("kind"), "template": ob["template"],
            "task": task_id, "started": started, "contract": contract,
        })


def is_deterministic_obligation(ob: dict | str) -> bool:
    """True iff this obligation is a known runner invocation (not Grok)."""
    if isinstance(ob, str):
        template = ob
        code_building = False
    else:
        template = (ob or {}).get("template") or ""
        code_building = bool(
            (ob or {}).get("code_building")
            or (ob or {}).get("kind") in {"code", "code-building"}
        )
    if code_building:
        return False
    if template in CODE_EDIT_TEMPLATES or template in RUNNER_WRITE_TEMPLATES:
        return False
    if str(template).startswith("novelty-"):
        return False
    return template in DETERMINISTIC_TEMPLATES


def lane_timeout_min() -> float:
    pol = load_odyssey_policy()
    det = pol.get("detachment") if isinstance(pol.get("detachment"), dict) else {}
    mem = det.get("memory") if isinstance(det.get("memory"), dict) else {}
    for src in (pol, det, mem):
        if not isinstance(src, dict):
            continue
        val = src.get("lane_timeout_min")
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
        if isinstance(val, str):
            try:
                n = float(val.strip())
            except ValueError:
                continue
            if n > 0:
                return n
    return float(DEFAULT_LANE_TIMEOUT_MIN)


def lane_timeout_s() -> int:
    return int(lane_timeout_min() * 60)


def epoch_iso(epoch: float) -> str:
    return datetime.fromtimestamp(float(epoch), timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _started_epoch(w: dict) -> float | None:
    if w.get("started_epoch") is not None:
        try:
            return float(w["started_epoch"])
        except (TypeError, ValueError):
            pass
    started = w.get("started")
    if not started:
        return None
    text = str(started).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def resolve_patient_weights(oxx: str, pkt: dict | None = None,
                            census: dict | None = None) -> str:
    """HF snapshot / census model_dir for --weights. Empty if unknown."""
    census_d = census if census is not None else load_census(oxx)
    pkt_d = pkt if pkt is not None else load_packet(oxx)
    w = weights_dir(oxx, pkt_d, census_d)
    if w:
        p = Path(os.path.expanduser(str(w)))
        if p.exists():
            return str(p)
    man = manifest_entry(oxx)
    src = (man.get("canonical_source") or "") if man else ""
    snap = hf_cache_snapshot(src)
    if snap is not None:
        return str(snap)
    ident = (pkt_d or {}).get("identity") or {}
    on_disk = ident.get("on_disk")
    if on_disk:
        p = Path(os.path.expanduser(str(on_disk)))
        if p.is_dir():
            return str(p)
    return w or ""


def runner_argv(ob: dict, *, weights: str | None = None,
                out: str | None = None) -> list[str]:
    """Resolve the patient-runner argv for a deterministic template."""
    oxx = ob.get("oxx") or ob.get("patient_id") or ""
    template = ob.get("template") or ""
    spec = ob.get("gravity_spec")
    w = weights if weights is not None else resolve_patient_weights(oxx)
    rec_rel = expected_receipt_rel(oxx, template, spec=spec)
    if not rec_rel:
        rec_rel = f"receipts/odyssey-i/{oxx}_{template}.json"
    out_path = out if out is not None else rec_rel
    packet = packet_rel(oxx)
    argv = [
        PREFERRED_PY,
        RUNNER_REL,
        "--oxx", oxx,
        "--weights", w or f"<weights-from-{oxx}-census>",
        "--runtime", "mlx",
        "--out", out_path,
        "--packet", packet,
    ]
    if template == "sensitivity-map":
        argv.extend(["--route-tokens", "0", "--sensitivity"])
        kind = arch_kind(oxx, load_packet(oxx), load_census(oxx))
        if kind != "moe":
            argv.append("--skip-route")
    elif template in GRAVITY_SPEC or str(template).startswith("gravity-"):
        gspec = spec or GRAVITY_SPEC.get(template) or "q4-g64"
        argv.extend(["--gravity", gspec])
        if template not in {"gravity-moe", "gravity-aggressive-moe"}:
            argv.append("--skip-route")
    elif template in NX_FLAG:
        argv.append(NX_FLAG[template])
        if template != "nx-gather-moe":
            argv.append("--skip-route")
    elif template == "external-science-dense":
        argv.extend(["--route-tokens", "0", "--skip-route"])
    elif template in {
        "external-science-moe", "route-map", "transfer-control",
    }:
        argv.extend(["--route-tokens", "512"])
    return argv


def lane_log_path(oxx: str, mechanism: str) -> Path:
    mech = (mechanism or "lane").replace("/", "-")
    return LANES_DIR / f"{oxx}-{mech}.log"


def default_spawn_runner(argv: list[str], log_path: Path, cwd: Path) -> int:
    """Detached setsid spawn; stdout/err append to log_path. Returns PID."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "ab")
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            argv,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(cwd),
            start_new_session=True,
            close_fds=True,
            env=env,
        )
    finally:
        log_f.close()
    return int(proc.pid)


def kill_process_group(pid, *, kill_fn=None, sig=None) -> bool:
    if kill_fn is not None:
        kill_fn(pid)
        return True
    try:
        ipid = int(pid)
    except (TypeError, ValueError):
        return False
    sig = signal.SIGTERM if sig is None else sig
    try:
        os.killpg(ipid, sig)
        return True
    except (OSError, ProcessLookupError, PermissionError):
        try:
            os.kill(ipid, sig)
            return True
        except (OSError, ProcessLookupError, PermissionError):
            return False


def _receipt_path_obj(rec_path) -> Path | None:
    if not rec_path:
        return None
    p = Path(rec_path)
    if p.is_file():
        return p
    if not p.is_absolute():
        alt = REPO / rec_path
        if alt.is_file():
            return alt
    return p


def _complete_subprocess_work(w: dict, *, now_epoch: float, persist: bool,
                              status: str) -> None:
    oxx = w.get("oxx") or ""
    tmpl = w.get("template") or ""
    mech = (
        w.get("mechanism")
        or w.get("mechanism_id")
        or mechanism_for_template(tmpl)
    )
    if not oxx or not mech:
        return
    rec = _receipt_path_obj(w.get("receipt_path"))
    sha = None
    ref = None
    if rec is not None and rec.is_file():
        sha = file_sha256(rec)
        try:
            ref = str(rec.resolve().relative_to(REPO.resolve())).replace("\\", "/")
        except ValueError:
            name = rec.name
            ref = name if name.startswith("receipts/") else f"receipts/odyssey-i/{name}"
    stamp = epoch_iso(now_epoch)
    tags = {}
    if str(mech).startswith("gravity-") and rec is not None:
        tags = gravity_tags_from_receipt(rec)
    mechs = [mech]
    if tmpl == "external-science-moe" and "route-map" not in mechs:
        mechs.append("route-map")
    if tmpl == "external-science-dense":
        kind = arch_kind(oxx, load_packet(oxx), load_census(oxx))
        if kind == "hybrid" and "ssm-accounting" not in mechs:
            mechs.append("ssm-accounting")
    for mechanism_id in mechs:
        grav = str(mechanism_id).startswith("gravity-")
        complete(
            obligation_id=str(w.get("id") or f"{oxx}:{mechanism_id}"),
            patient_id=oxx,
            mechanism_id=mechanism_id,
            status=status,
            completed_at=stamp,
            receipt_ref=ref,
            receipt_sha256=sha,
            persist=persist,
            candidate_class=tags.get("candidate_class") if grav else None,
            conventionality=tags.get("conventionality") if grav else None,
        )
    if tags and status == "VERIFIED":
        apply_gravity_tags_to_packet(oxx, tags)


def _mark_subprocess_running(state: dict, ob: dict, *, pid: int,
                             now_epoch: float, receipt_path: str,
                             timeout_s: int, argv: list[str],
                             log_path: str) -> dict:
    mech = (
        ob.get("mechanism")
        or ob.get("mechanism_id")
        or mechanism_for_template(ob.get("template") or "")
    )
    started = epoch_iso(now_epoch)
    entry = {
        "id": ob.get("id"),
        "oxx": ob.get("oxx"),
        "title": ob.get("title"),
        "status": "RUNNING",
        "info": ob.get("info"),
        "wall_cost": ob.get("wall_cost"),
        "gpu_cost": ob.get("gpu_cost"),
        "opus_cost": ob.get("opus_cost"),
        "kind": "subprocess",
        "template": ob.get("template"),
        "mechanism": mech,
        "mechanism_id": mech,
        "pid": int(pid),
        "started": started,
        "started_epoch": float(now_epoch),
        "receipt_path": receipt_path,
        "timeout_s": int(timeout_s),
        "argv": list(argv),
        "log": log_path,
        "task": f"pid:{pid}",
        "retries": int(ob.get("retries") or 0),
    }
    found = False
    for w in state.setdefault("work", []):
        if w.get("id") == ob.get("id") or (
            w.get("oxx") == ob.get("oxx")
            and w.get("template") == ob.get("template")
            and w.get("status") in {None, "READY", "FAILED", "RUNNING"}
        ):
            w.update(entry)
            found = True
            return w
    if not found:
        state["work"].append(entry)
    return entry


def launch_deterministic(
    oblig: dict,
    *,
    now_epoch: float | None = None,
    in_flight_gib: float = 0.0,
    state: dict | None = None,
    persist: bool = True,
    spawn: bool = True,
    spawn_fn=None,
    cwd: Path | None = None,
) -> dict:
    """Memgate + detached runner spawn. Caller may pass wall time in."""
    if now_epoch is None:
        now_epoch = time.time()
    oxx = oblig.get("oxx") or ""
    template = oblig.get("template") or ""
    spec = oblig.get("gravity_spec")
    argv = runner_argv(oblig)
    rec_rel = expected_receipt_rel(oxx, template, spec=spec) or (
        f"receipts/odyssey-i/{oxx}_{template}.json"
    )
    timeout = int(oblig.get("timeout_s") or lane_timeout_s())
    mech = (
        oblig.get("mechanism")
        or oblig.get("mechanism_id")
        or mechanism_for_template(template)
    )
    logp = lane_log_path(oxx, mech)
    row = {
        "verdict": "DRY-RUN",
        "kind": "subprocess",
        "argv": argv,
        "receipt_path": rec_rel,
        "timeout_s": timeout,
        "oxx": oxx,
        "template": template,
        "mechanism": mech,
        "log": str(logp),
        "pid": None,
        "skip_reason": None,
    }
    mg = memgate_admit_lane(oblig, in_flight_gib)
    row["memgate"] = mg.get("decision")
    row["est_gib"] = mg.get("est_gib")
    row["in_flight_gib"] = mg.get("in_flight_gib")
    if mg.get("decision") == "REFUSE":
        row["verdict"] = "SKIP"
        row["skip_reason"] = mg.get("note") or "memgate REFUSE"
        return row
    if not spawn:
        row["verdict"] = "DRY-RUN"
        return row
    weights = None
    try:
        wi = argv.index("--weights")
        weights = argv[wi + 1] if wi + 1 < len(argv) else None
    except ValueError:
        weights = None
    if weights and str(weights).startswith("<"):
        row["verdict"] = "SKIP"
        row["skip_reason"] = "weights not resolved"
        return row
    if weights and not Path(os.path.expanduser(str(weights))).exists():
        row["verdict"] = "SKIP"
        row["skip_reason"] = f"weights not found: {weights}"
        return row
    dest_cwd = Path(cwd) if cwd is not None else REPO
    spawner = spawn_fn or default_spawn_runner
    try:
        pid = spawner(argv, logp, dest_cwd)
    except (OSError, TypeError, ValueError) as exc:
        row["verdict"] = "SKIP"
        row["skip_reason"] = f"spawn failed: {exc}"
        return row
    row["pid"] = int(pid)
    row["verdict"] = "LAUNCH"
    st = state if state is not None else ensure_state()
    _mark_subprocess_running(
        st, oblig, pid=int(pid), now_epoch=float(now_epoch),
        receipt_path=rec_rel, timeout_s=timeout, argv=argv, log_path=str(logp),
    )
    if persist:
        save_state(st)
    return row


def reap_lanes(
    now_epoch: float,
    *,
    state: dict | None = None,
    persist: bool = True,
    dry_run: bool = False,
    pid_alive_fn=None,
    kill_fn=None,
) -> list[dict]:
    """Self-heal RUNNING subprocess lanes. Call at the top of every tick.

    PID alive & age<timeout → still RUNNING (counts toward cap).
    PID dead & receipt exists → complete() VERIFIED.
    PID dead & no receipt → FAILED, retry once, else REFUTED.
    PID alive & age>timeout → kill process group, FAILED (retry once).
    """
    st = state if state is not None else ensure_state()
    alive_fn = pid_alive_fn or _pid_alive
    default_timeout = float(lane_timeout_s())
    rows: list[dict] = []
    mutated = False

    def receipt_exists(w: dict) -> bool:
        rec = _receipt_path_obj(w.get("receipt_path"))
        return bool(rec is not None and rec.is_file())

    def fail_or_retry(w: dict, reason: str) -> str:
        retries = int(w.get("retries") or 0)
        w["fail_reason"] = reason
        if retries < 1:
            w["retries"] = retries + 1
            w["status"] = "READY"
            w["pid"] = None
            w["task"] = None
            return "retry"
        w["status"] = "REFUTED"
        w["pid"] = None
        if not dry_run:
            _complete_subprocess_work(
                w, now_epoch=float(now_epoch), persist=persist, status="REFUTED",
            )
        return "refute"

    for w in list(st.get("work") or []):
        kind = w.get("kind")
        status = w.get("status")
        if kind != "subprocess":
            continue
        if status not in {"RUNNING", "FAILED"}:
            continue
        pid = w.get("pid")
        started = _started_epoch(w)
        timeout_s = float(w.get("timeout_s") or default_timeout)
        age = (
            float(now_epoch) - float(started)
            if started is not None
            else 0.0
        )
        is_alive = bool(pid) and bool(alive_fn(pid))
        has_receipt = receipt_exists(w)
        action = None
        note = None
        if dry_run:
            retries = int(w.get("retries") or 0)
            if status == "FAILED":
                if is_alive:
                    action = "kill-zombie"
                elif has_receipt:
                    action = "verified"
                else:
                    action = "retry" if retries < 1 else "refute"
            elif is_alive and age <= timeout_s:
                action = "keep"
            elif (not is_alive) and has_receipt:
                action = "verified"
            elif is_alive and age > timeout_s:
                action = "timeout"
            else:
                action = "retry" if retries < 1 else "refute"
            rows.append({
                "id": w.get("id"), "oxx": w.get("oxx"),
                "template": w.get("template"), "pid": pid,
                "action": action, "status": status, "dry_run": True,
                "retries": retries,
            })
            continue

        if status == "FAILED":
            if is_alive:
                if not dry_run:
                    kill_process_group(pid, kill_fn=kill_fn, sig=signal.SIGKILL)
                action = "kill-zombie"
                note = "FAILED pid still alive; SIGKILL"
            elif has_receipt:
                w["status"] = "VERIFIED"
                action = "verified"
                if not dry_run:
                    _complete_subprocess_work(
                        w, now_epoch=float(now_epoch), persist=persist,
                        status="VERIFIED",
                    )
                mutated = True
            else:
                action = fail_or_retry(w, w.get("fail_reason") or "exited-no-receipt")
                mutated = True
            rows.append({
                "id": w.get("id"), "oxx": w.get("oxx"),
                "template": w.get("template"), "pid": pid,
                "action": action, "status": w.get("status"), "note": note,
            })
            continue

        # RUNNING
        if is_alive and age <= timeout_s:
            action = "keep"
        elif (not is_alive) and has_receipt:
            w["status"] = "VERIFIED"
            action = "verified"
            if not dry_run:
                _complete_subprocess_work(
                    w, now_epoch=float(now_epoch), persist=persist,
                    status="VERIFIED",
                )
            mutated = True
        elif is_alive and age > timeout_s:
            if not dry_run:
                kill_process_group(pid, kill_fn=kill_fn)
            w["status"] = "FAILED"
            w["fail_reason"] = "timeout"
            mutated = True
            action = "timeout"
            # Retry is the next tick: FAILED + dead PID → fail_or_retry.
        else:
            # dead, no receipt
            w["status"] = "FAILED"
            mutated = True
            action = fail_or_retry(w, "exited-no-receipt")

        rows.append({
            "id": w.get("id"), "oxx": w.get("oxx"),
            "template": w.get("template"), "pid": pid,
            "action": action, "status": w.get("status"),
            "retries": w.get("retries"), "note": note,
        })

    if persist and mutated and not dry_run:
        save_state(st)
    return rows


def _sys_free_ram_pct():
    """System-wide free RAM %, via macOS `memory_pressure`. None if unavailable.
    Used to override worker_gate's over-strict stale-swap REFUSE (reuse_surface
    flagged its coefficients as needing retune)."""
    try:
        import subprocess, re as _re
        out = subprocess.run(["memory_pressure"], capture_output=True,
                             text=True, timeout=5).stdout
        m = _re.search(r"free percentage:\s*(\d+)%", out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def obligation_est_gib(ob: dict) -> float:
    """Resident model estimate for memgate. 4-bit footprint; default 16 GiB."""
    oxx = ob.get("oxx") or ""
    man = manifest_entry(oxx)
    v = man.get("est_4bit_gib")
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
        return float(v)
    return float(getattr(memgate, "DEFAULT_EST_GIB", 16.0))


def _template_loads_model(template: str, ob: dict | None = None) -> bool:
    if ob is not None and ob.get("model_loading") is False:
        return False
    t = template or ((ob or {}).get("template") or "")
    if not t or str(t).startswith("novelty-"):
        return False
    if ob is not None and ob.get("model_loading") is True:
        return True
    return (
        t in DATA_PRODUCING_TEMPLATES
        or t in GRAVITY_SPEC
        or t in NX_FLAG
        or t in {
            "external-science-moe", "external-science-dense",
            "sensitivity-map", "transfer-control", "route-map",
        }
    )


def _running_model_gib(
    state: dict,
    now_epoch: float | None = None,
    pid_alive_fn=None,
) -> float:
    total = 0.0
    seen: set[tuple] = set()
    now = float(now_epoch) if now_epoch is not None else time.time()
    for w in state.get("work") or []:
        if not _lane_still_running(w, now, pid_alive_fn=pid_alive_fn):
            continue
        tmpl = w.get("template") or ""
        if not _template_loads_model(tmpl, w):
            continue
        # Per-LANE identity, not (oxx, template): multiple ladder rungs of the
        # same patient share a template but each is its own resident model, so
        # each must count toward RAM or memgate under-counts and over-admits.
        key = w.get("pid") or w.get("task") or w.get("id") or (
            w.get("oxx"), w.get("mechanism_id") or tmpl,
        )
        if key in seen:
            continue
        seen.add(key)
        total += obligation_est_gib(w)
    return total


MLX_ODYSSEY_CACHE = Path.home() / ".cache" / "mlx" / "odyssey"


def _inflight_gravity_cache_dirs(
    state: dict, now_epoch: float | None = None, pid_alive_fn=None,
) -> set[str]:
    """Absolute cache dirs a RUNNING lane is reading (must not be evicted)."""
    now = float(now_epoch) if now_epoch is not None else time.time()
    keep: set[str] = set()
    for w in state.get("work") or []:
        if not _lane_still_running(w, now, pid_alive_fn=pid_alive_fn):
            continue
        argv = w.get("argv") or []
        if "--gravity" in argv:
            try:
                spec = argv[argv.index("--gravity") + 1]
            except (ValueError, IndexError):
                continue
            keep.add(str(MLX_ODYSSEY_CACHE / f"{w.get('oxx')}-gravity-{spec}"))
    return keep


def evict_gravity_caches(
    target_free_gib: float,
    *,
    state: dict,
    now_epoch: float | None = None,
    pid_alive_fn=None,
) -> float:
    """LRU-evict cold `*-gravity-*` spec caches until free disk >= target.

    Never removes a base `*-4bit` parent (reused constantly) or a spec an
    in-flight lane is reading. Returns GiB freed. Coldest (oldest atime) first.
    """
    if not MLX_ODYSSEY_CACHE.is_dir():
        return 0.0
    keep = _inflight_gravity_cache_dirs(state, now_epoch, pid_alive_fn)
    specs = []
    for d in MLX_ODYSSEY_CACHE.iterdir():
        if not d.is_dir() or "-gravity-" not in d.name:
            continue
        if str(d) in keep:
            continue
        try:
            atime = d.stat().st_atime
        except OSError:
            continue
        specs.append((atime, d))
    specs.sort()  # coldest first
    freed = 0.0
    for _atime, d in specs:
        if float(machine_snapshot().get("disk_free_gib") or 0.0) >= target_free_gib:
            break
        try:
            sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        except OSError:
            sz = 0
        shutil.rmtree(d, ignore_errors=True)
        freed += sz / (1024 ** 3)
    return freed


def _running_model_lane_count(
    state: dict,
    now_epoch: float | None = None,
    pid_alive_fn=None,
) -> int:
    """Distinct RUNNING model lanes (subprocess). Grok/novelty lanes excluded."""
    seen: set[tuple] = set()
    now = float(now_epoch) if now_epoch is not None else time.time()
    for w in state.get("work") or []:
        if not _lane_still_running(w, now, pid_alive_fn=pid_alive_fn):
            continue
        tmpl = w.get("template") or ""
        if not _template_loads_model(tmpl, w):
            continue
        # per-lane, not (oxx, template): concurrent same-patient rungs each count
        seen.add(w.get("pid") or w.get("task") or w.get("id") or (
            w.get("oxx"), w.get("mechanism_id") or tmpl,
        ))
    return len(seen)


def _headroom_override_note() -> str | None:
    if os.environ.get("ODYSSEY_HEADROOM_ADMIT") != "1":
        return None
    free = _sys_free_ram_pct()
    if free is not None and free >= 70:
        return f"free RAM {free}%"
    return None


def memgate_admit_lane(ob: dict, in_flight_gib: float) -> dict:
    """Primary model-lane admission. HEADROOM_ADMIT is an opt-in fallback."""
    if not _template_loads_model(ob.get("template") or "", ob):
        return {
            "decision": "n/a",
            "note": "not a model lane",
            "projected_swap_gib": None,
            "est_gib": 0.0,
            "in_flight_gib": float(in_flight_gib or 0.0),
            "clean_room": False,
        }
    est = obligation_est_gib(ob)
    clean = bool(ob.get("timing"))
    try:
        verdict = memgate.admit(est, in_flight_gib=in_flight_gib, clean_room=clean)
    except Exception as exc:
        verdict = {
            "decision": "REFUSE",
            "note": f"memgate failed: {exc}",
            "projected_swap_gib": None,
        }
    out = dict(verdict)
    out["est_gib"] = est
    out["in_flight_gib"] = float(in_flight_gib or 0.0)
    out["clean_room"] = clean
    if out.get("decision") == "REFUSE" and not clean:
        note = _headroom_override_note()
        if note:
            out["decision"] = "GO"
            out["headroom_fallback"] = True
            out["note"] = (
                f"memgate REFUSE overridden (opt-in HEADROOM_ADMIT): {note}; "
                f"{out.get('note') or ''}"
            )
    return out


def evaluate_gates(ob: dict, *, go: bool, running_n: int, cap: int,
                   snap: dict, worker: dict | None,
                   lint_ok: bool, lint_msg: str,
                   disk_after_reclaim: float | None = None,
                   code_edit_busy: bool = False,
                   scope_conflict: str | None = None) -> dict:
    """Return a gate bundle + verdict/skip_reason. Does not launch."""
    disk = float(snap.get("disk_free_gib") or 0.0)
    clean_ok = bool(snap.get("clean_box_ok"))
    clean_why = snap.get("clean_box_reason") or ""
    worker_decision = (worker or {}).get("decision") if ob.get("model_loading") else "n/a"
    reasons = []
    worker_override = None
    if cap <= 0 or running_n >= cap:
        reasons.append(f"max-lanes cap {cap} (running={running_n})")
    elif ob.get("template") in CODE_EDIT_TEMPLATES and code_edit_busy:
        reasons.append(
            "code-edit serial: template edits tools/odyssey_patient_runner.py "
            "while another lane is RUNNING"
        )
    elif scope_conflict:
        reasons.append(f"write-scope collision: {scope_conflict}")
    elif not lint_ok:
        reasons.append(f"SG-rejected: {lint_msg}")
    elif ob.get("model_loading") and worker_decision == "REFUSE":
        # Prefer memgate. ODYSSEY_HEADROOM_ADMIT=1 is an opt-in fallback for
        # stale-swap REFUSE (worker_gate or memgate swap formula). Never override
        # clean_room / protected-timing exclusivity.
        _note = str((worker or {}).get("note") or "")
        _clean = bool(ob.get("timing")) or "clean_room" in _note
        _free = _sys_free_ram_pct() if os.environ.get("ODYSSEY_HEADROOM_ADMIT") == "1" else None
        if (not _clean) and _free is not None and _free >= 70:
            worker_override = (
                f"model-gate REFUSE overridden (opt-in): free RAM {_free}% "
                f"(stale swap); runner self-gates load to 4-bit; prefer memgate"
            )
        else:
            reasons.append(
                f"model-gate REFUSE: {_note or 'REFUSE'}"
            )
    elif ob.get("download") and disk < DISK_RUN_GIB and (
        disk_after_reclaim is None or disk_after_reclaim < DISK_RUN_GIB
    ):
        d = disk_after_reclaim if disk_after_reclaim is not None else disk
        reasons.append(
            f"disk {d:.1f} GiB < {DISK_RUN_GIB} after reclaim; download-implying"
        )
    elif ob.get("timing") and not clean_ok:
        reasons.append(f"clean_box_ok false (§14 protected-GPU): {clean_why}")
    if go:
        verdict = "SKIP" if reasons else "LAUNCH"
        skip_reason = "; ".join(reasons) if reasons else None
    else:
        verdict = "DRY-RUN"
        skip_reason = ("would skip: " + "; ".join(reasons)) if reasons else None
    return {
        "verdict": verdict,
        "skip_reason": skip_reason,
        "worker": worker_decision,
        "worker_note": (worker or {}).get("note"),
        "worker_override": worker_override,
        "disk_free_gib": disk,
        "disk_floor_run": DISK_RUN_GIB,
        "would_reclaim": disk < DISK_RUN_GIB,
        "clean_box_ok": clean_ok,
        "clean_box_reason": clean_why,
        "sg": "ok" if lint_ok else "ERROR",
        "sg_msg": lint_msg,
        "model_loading": bool(ob.get("model_loading")),
        "timing": bool(ob.get("timing")),
        "download": bool(ob.get("download")),
        "running": running_n,
        "cap": cap,
        "_evidence": "MEASURED (gates) + DERIVED (verdict)" if worker else "DERIVED (verdict)",
    }


def _running_scopes(state: dict, running_ids: set[str]) -> list[dict]:
    """write_set/exclusive_resources of currently RUNNING science lanes."""
    scopes = []
    seen = set()
    for w in state.get("work") or []:
        if w.get("status") != "RUNNING":
            continue
        oxx, tmpl = w.get("oxx"), w.get("template")
        if not oxx or not tmpl:
            parsed = parse_science_task(str(w.get("task") or ""))
            if parsed:
                oxx, tmpl = parsed
        if not oxx or not tmpl:
            continue
        key = (oxx, tmpl)
        if key in seen:
            continue
        seen.add(key)
        scopes.append(write_scope(w if w.get("template") else {"oxx": oxx, "template": tmpl}))
    for name in running_ids:
        parsed = parse_science_task(name) or parse_science_task(
            re.sub(r"-\d{8}-\d{6}$", "", name)
        )
        if not parsed:
            continue
        if parsed in seen:
            continue
        seen.add(parsed)
        scopes.append(write_scope({"oxx": parsed[0], "template": parsed[1]}))
    return scopes


def run_loop(*, go: bool, max_lanes: int, grok_lanes: int = 0,
             state: dict | None = None,
             observe_fn=None, gate_fn=None, snapshot_fn=None,
             launch_fn=None, lint_fn=None, reclaim_fn=None,
             log_path: Path | None = None, auto_dir: Path | None = None,
             persist: bool = True, consider_limit: int | None = None,
             completions=None, spawn_fn=None, now_epoch: float | None = None,
             pid_alive_fn=None, kill_fn=None) -> list[dict]:
    """Select, render, gate; launch only when go=True. Idempotent. Never blocks.

    max_lanes = MODEL subprocess concurrency ceiling (memgate is the real
    limiter under it). grok_lanes = grok/novelty ceiling, DEFAULT 0: the
    autonomous odyssey loop runs deterministic model science only, no grok.
    Grok novelty is opt-in scaffolding (grok_lanes>0), dispatched deliberately.
    """
    st = state if state is not None else ensure_state()
    now = float(now_epoch) if now_epoch is not None else time.time()
    reap_lanes(
        now, state=st, persist=persist and go, dry_run=not go,
        pid_alive_fn=pid_alive_fn, kill_fn=kill_fn,
    )
    ranked = select_ready_obligations(st, completions=completions)
    cap = min(max(int(max_lanes), 0), HARD_LANE_CAP)
    running_ids = odyssey_running_ids(st, now_epoch=now, pid_alive_fn=pid_alive_fn)
    running_n = len(running_ids)
    snap = (snapshot_fn or machine_snapshot)()
    disk = float(snap.get("disk_free_gib") or 0.0)
    reclaimed = False
    disk_after = None
    if go and cap > 0 and disk < DISK_EVICT_TRIGGER:
        # Evict COLD (sealed, not-in-flight) gravity-spec caches ONLY when disk
        # is actually low, and only up to DISK_EVICT_TARGET — not everything
        # every tick (that made each tick take ~90s of du/rmtree I/O, starving
        # the box between waves). A sealed spec's science is in its receipt, so
        # its cache is pure reclaimable disk. Keeping disk near the target lets
        # the throttle admit more concurrent lanes without slow ticks.
        freed = evict_gravity_caches(DISK_EVICT_TARGET, state=st, now_epoch=now,
                                     pid_alive_fn=pid_alive_fn)
        if disk < DISK_RUN_GIB:
            fn = reclaim_fn or (lambda: subprocess.run(
                ["bash", str(RECLAIM)], cwd=str(REPO), check=False,
            ))
            fn()
        reclaimed = True
        snap = (snapshot_fn or machine_snapshot)()
        disk_after = float(snap.get("disk_free_gib") or 0.0)
        disk = disk_after

    worker_cache = None
    rows = []
    launched = 0
    occupied = 0  # concurrent admits this tick (real or planned)
    limit = consider_limit if consider_limit is not None else max(cap, DEFAULT_MAX_LANES, 8)
    occupied_scopes = _running_scopes(st, running_ids)
    in_flight_gib = _running_model_gib(st, now_epoch=now, pid_alive_fn=pid_alive_fn)
    # Concurrency is decoupled by lane kind. Grok/novelty lanes are external
    # reasoning that does NO GPU/model work, so they must not consume the
    # model-experiment budget — they stay bounded by max_lanes (`cap`). Model
    # subprocess lanes are the whole point of memgate: admit as many concurrent
    # small-model experiments as swap<SWAP_MAX allows, under HARD_LANE_CAP. Two
    # grok lanes were starving all six on-disk models behind cap=2.
    running_model_n = _running_model_lane_count(st, now, pid_alive_fn)
    running_grok_n = max(0, running_n - running_model_n)
    # max_lanes is the MODEL concurrency ceiling; memgate (cumulative
    # swap<SWAP_MAX) is the real limiter under it. grok_lanes is the grok/novelty
    # ceiling, default 0 so the autonomous loop spends no grok usage — grok
    # novelty is deliberate scaffolding, not an every-tick reflex.
    model_cap = cap
    grok_cap = min(max(int(grok_lanes), 0), HARD_LANE_CAP)
    grok_launched = 0
    model_launched = 0
    grok_slots = 0 if not go else max(0, grok_cap - running_grok_n)
    model_slots = 0 if not go else max(0, model_cap - running_model_n)
    # Disk-aware throttle on NEW model lanes: each new gravity spec materializes
    # ~one 4-bit body (~5-16 GiB) to the mlx cache. memgate governs RAM/swap but
    # not disk; without this, N parallel new-spec writes fill the disk between
    # per-tick evictions (in-flight specs cannot be evicted). Cap new launches to
    # what fits above DISK_MIN_FLOOR; running lanes finish, evict, and free slots
    # next tick. No deadlock: running lanes always terminate.
    if go and model_slots > 0:
        headroom_lanes = int(max(0.0, disk - DISK_MIN_FLOOR) / TYPICAL_SPEC_GIB)
        model_slots = min(model_slots, headroom_lanes)

    for ob in ranked:
        if go and grok_launched >= grok_slots and model_launched >= model_slots:
            break
        if not go and model_cap == 0 and grok_cap == 0:
            break
        if len(rows) >= limit:
            break

        # grok disabled (grok_cap==0, the autonomous default): don't even
        # consider grok/novelty obligations — they cannot launch and would only
        # crowd the plan and the consider window ahead of real model science.
        if grok_cap == 0 and not is_deterministic_obligation(ob):
            continue

        already = None
        for w in st.get("work") or []:
            if w.get("id") == ob["id"] and w.get("status") == "RUNNING":
                already = w.get("task") or w.get("id")
                break
            if (
                w.get("status") == "RUNNING"
                and w.get("oxx") == ob.get("oxx")
                and w.get("template") == ob.get("template")
            ):
                already = w.get("task") or w.get("id")
                break
        dest = render_contract(ob, auto_dir=auto_dir)
        try:
            rel = str(dest.relative_to(REPO))
        except ValueError:
            rel = str(dest)
        det = is_deterministic_obligation(ob)
        planned_argv = runner_argv(ob) if det else None
        if det:
            lint_ok, lint_msg = True, "n/a (subprocess)"
        else:
            lint_ok, lint_msg = (lint_fn or sg_lint)(dest)

        worker = None
        mg = memgate_admit_lane(ob, in_flight_gib)
        if _template_loads_model(ob.get("template") or "", ob):
            if mg.get("decision") == "REFUSE":
                worker = {
                    "decision": "REFUSE",
                    "note": mg.get("note") or "memgate REFUSE",
                    "projected_swap_gib": mg.get("projected_swap_gib"),
                }
            else:
                if worker_cache is None:
                    worker_cache = call_worker_gate(observe_fn, gate_fn)
                worker = dict(worker_cache)
                worker["memgate"] = mg.get("decision")
                worker["projected_swap_gib"] = mg.get("projected_swap_gib")
                if mg.get("headroom_fallback"):
                    worker["worker_override"] = mg.get("note")
        elif ob.get("model_loading"):
            if worker_cache is None:
                worker_cache = call_worker_gate(observe_fn, gate_fn)
            worker = worker_cache

        scope = write_scope(ob)
        conflict = scope_conflict_reason(scope, occupied_scopes)
        # cap is per-kind: model subprocesses gated by HARD_LANE_CAP (memgate is
        # the real limiter under it), grok/novelty by max_lanes. dry-run listing
        # is informational; collision still evaluated against occupied slots.
        if det:
            kind_cap = model_cap if go else max(model_cap, limit)
            kind_running = running_model_n + model_launched
        else:
            kind_cap = grok_cap if go else max(grok_cap, limit)
            kind_running = running_grok_n + grok_launched
        gates = evaluate_gates(
            ob, go=go, running_n=kind_running, cap=kind_cap,
            snap=snap, worker=worker, lint_ok=lint_ok, lint_msg=lint_msg,
            disk_after_reclaim=disk_after,
            code_edit_busy=False,
            scope_conflict=conflict,
        )
        if already and go:
            gates["verdict"] = "SKIP"
            gates["skip_reason"] = f"already RUNNING ({already})"

        row = {
            "schema": RUN_LOG_SCHEMA,
            "obligation": ob["id"],
            "oxx": ob["oxx"],
            "title": ob["title"],
            "template": ob["template"],
            "mechanism_id": ob.get("mechanism_id") or mechanism_for_template(ob["template"]),
            "write_set": scope["write_set"],
            "exclusive_resources": scope["exclusive_resources"],
            "proxy": round(value(ob), 2),
            "contract": rel,
            "verdict": gates["verdict"],
            "skip_reason": gates["skip_reason"],
            "task_id": None,
            "gates": gates,
            "model_loading": ob["model_loading"],
            "timing": ob["timing"],
            "download": ob["download"],
            "est_gib": mg.get("est_gib"),
            "in_flight_gib": mg.get("in_flight_gib"),
            "memgate": mg.get("decision"),
            "reclaimed": reclaimed,
            "go": go,
            "kind": "subprocess" if det else "grok",
            "launch": "subprocess" if det else "grok-delegate",
            "argv": planned_argv,
            "_evidence": "DERIVED (§18 run-loop decision)",
        }

        admitted = False
        if go and gates["verdict"] == "LAUNCH":
            if det:
                launched_row = launch_deterministic(
                    ob, now_epoch=now, in_flight_gib=in_flight_gib,
                    state=st, persist=persist, spawn=True, spawn_fn=spawn_fn,
                )
                row["argv"] = launched_row.get("argv") or planned_argv
                row["kind"] = "subprocess"
                row["launch"] = "subprocess"
                if launched_row.get("pid") and launched_row.get("verdict") == "LAUNCH":
                    row["task_id"] = f"pid:{launched_row['pid']}"
                    row["pid"] = launched_row["pid"]
                    launched += 1
                    model_launched += 1
                    occupied += 1
                    admitted = True
                    gates["running"] = running_n + launched
                    record_ctl_event(
                        ob.get("oxx"), "cpu", 0.0,
                        extra={
                            "template": ob.get("template"),
                            "verdict": "LAUNCH",
                            "kind": "subprocess",
                            "pid": launched_row["pid"],
                        },
                        persist=persist,
                    )
                else:
                    row["verdict"] = "SKIP"
                    row["skip_reason"] = (
                        launched_row.get("skip_reason") or "subprocess spawn failed"
                    )
                    gates["verdict"] = "SKIP"
                    gates["skip_reason"] = row["skip_reason"]
            else:
                slug = f"odyssey-{ob['oxx'].lower()}-{ob['template']}"
                fn = launch_fn or default_launch
                try:
                    rc, task_id, output = fn(
                        slug, dest, model_loading=ob["model_loading"],
                    )
                except TypeError:
                    rc, task_id, output = fn(slug, dest)
                if rc != 0 or not task_id:
                    row["verdict"] = "SKIP"
                    row["skip_reason"] = f"grok-run failed rc={rc} {output[-400:]}"
                    gates["verdict"] = "SKIP"
                    gates["skip_reason"] = row["skip_reason"]
                else:
                    started = utc_now()
                    row["task_id"] = task_id
                    if persist:
                        _mark_running(st, ob, task_id, rel, started)
                    launched += 1
                    grok_launched += 1
                    occupied += 1
                    admitted = True
                    running_n_display = running_n + launched
                    gates["running"] = running_n_display
                    record_ctl_event(
                        ob.get("oxx"), "grok", 0.0,
                        grok_lane=task_id,
                        extra={"template": ob.get("template"), "verdict": "LAUNCH"},
                        persist=persist,
                    )

        if not go and gates["verdict"] == "DRY-RUN" and not gates.get("skip_reason"):
            occupied += 1
            if det:
                model_launched += 1
            else:
                grok_launched += 1
            admitted = True

        if admitted:
            occupied_scopes.append(scope)
            if _template_loads_model(ob.get("template") or "", ob):
                in_flight_gib += float(mg.get("est_gib") or obligation_est_gib(ob))

        if go or log_path is not None:
            append_run_log(row, path=log_path)
        rows.append(row)

    if persist and go and launched:
        save_state(st)
    return rows


def print_run_plan(rows: list[dict], *, go: bool, max_lanes: int,
                   snap: dict, running_n: int) -> None:
    mode = "GO" if go else "dry-run"
    disk = snap.get("disk_free_gib")
    print(
        f"ODYSSEY RUN  {mode}  max_lanes={max_lanes}  hard_cap={HARD_LANE_CAP}  "
        f"running={running_n}  disk={disk}GiB"
    )
    if not go:
        print("launch NOTHING (dry-run is the default; pass --go to spawn)")
    if not rows:
        print("PLAN: (empty)")
        return
    print("PLAN:")
    for i, r in enumerate(rows, 1):
        g = r.get("gates") or {}
        print(
            f"  {i}. {r['obligation']}  {r['oxx']}  {r['template']}  "
            f"proxy={r['proxy']:.1f}  verdict={r['verdict']}"
        )
        print(f"     title: {r.get('title')}")
        print(f"     contract: {r['contract']}")
        print(
            f"     model-loading: {'yes' if r.get('model_loading') else 'no'}  "
            f"timing: {'yes' if r.get('timing') else 'no'}  "
            f"download: {'yes' if r.get('download') else 'no'}"
            + (
                f"  memgate={r.get('memgate')} est={r.get('est_gib')}GiB "
                f"in_flight={r.get('in_flight_gib')}GiB"
                if r.get("memgate") not in (None, "n/a")
                else ""
            )
        )
        print(
            f"     gates: worker={g.get('worker')}  "
            f"disk={g.get('disk_free_gib')}{' <45 would-reclaim' if g.get('would_reclaim') else ''}  "
            f"clean_box={g.get('clean_box_ok')}  sg={g.get('sg')}"
        )
        if r.get("kind") or r.get("launch"):
            print(
                f"     launch: {r.get('launch') or r.get('kind')}  "
                f"kind={r.get('kind') or '—'}"
            )
        if r.get("argv"):
            argv = r["argv"]
            printed = " ".join(str(a) for a in argv) if isinstance(argv, list) else str(argv)
            print(f"     argv: {printed}")
        if r.get("write_set"):
            print(f"     write_set: {', '.join(r['write_set'])}")
        if r.get("skip_reason"):
            print(f"     skip: {r['skip_reason']}")
        if r.get("task_id"):
            print(f"     task: {r['task_id']}")
        if r.get("pid"):
            print(f"     pid: {r['pid']}")
        print(f"     _evidence={r.get('_evidence')}")


def cmd_run(*, go: bool, max_lanes: int, grok_lanes: int = 0, **hooks) -> int:
    st = hooks.pop("state", None) or ensure_state()
    snap = (hooks.get("snapshot_fn") or machine_snapshot)()
    hooks.setdefault("persist", go)
    rows = run_loop(go=go, max_lanes=max_lanes, grok_lanes=grok_lanes,
                    state=st, **hooks)
    running_n = len(odyssey_running_ids(st))
    print_run_plan(rows, go=go, max_lanes=max_lanes, snap=snap, running_n=running_n)
    if go:
        print()
        cmd_status()
    return 0


# ---------------------------------------------------------------------------
# patient class → required obligation set + retirement (steer S002)
# ---------------------------------------------------------------------------

def _family_key(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "", (text or "").lower())
    for tok in ("instruct", "chat", "it", "vl"):
        s = s.replace(tok, "")
    return s


def reference_sibling(oxx: str, state: dict | None = None) -> str | None:
    """Named transfer *reference* for this patient (the sibling, not the original).

    Manifest.reference_sibling is authority; TRANSFER_REF / ledger text are fallback.
    """
    man = manifest_entry(oxx)
    if man.get("reference_sibling"):
        return str(man["reference_sibling"])
    if oxx in TRANSFER_REF:
        return TRANSFER_REF[oxx]
    st = state if state is not None else ensure_state()
    meta = patient_meta(oxx, st)
    klass = (meta.get("class") or "").lower()
    ledger = (meta.get("ledger") or "").lower()
    note = f"{klass} {ledger}"
    is_sib = (
        "sibling" in note
        or "transfer ctrl" in note
        or re.search(r"\breference\b", note)
    )
    if not is_sib and LEDGER.is_file():
        for line in LEDGER.read_text().splitlines():
            if not re.search(rf"\b{re.escape(oxx)}\b", line):
                continue
            if re.search(r"\b(sibling|transfer ctrl)\b", line, re.I):
                is_sib = True
                break
    if not is_sib:
        return None
    src = meta.get("source") or meta.get("model") or ""
    key = _family_key(src.split("/")[-1] if "/" in src else src)
    for p in st.get("patients") or []:
        other = p.get("oxx")
        if not other or other == oxx:
            continue
        # the reference is the non-sibling family member
        oklass = (p.get("class") or "").lower()
        if "sibling" in oklass:
            continue
        osrc = p.get("source") or p.get("model") or ""
        okey = _family_key(osrc.split("/")[-1] if "/" in osrc else osrc)
        if key and okey and (key in okey or okey in key):
            return other
    return None


def patient_arch_kind(oxx: str, state: dict | None = None) -> str:
    return arch_kind(oxx, load_packet(oxx), load_census(oxx))


def required_mechanisms(oxx: str, state: dict | None = None) -> list[str]:
    """Bounded required set for this patient's class. Do not over-deepen."""
    kind = patient_arch_kind(oxx, state)
    if kind == "moe":
        req = list(REQUIRED_MOE)
        if reference_sibling(oxx, state):
            req.append("transfer-control")
        return req
    if kind == "hybrid":
        return list(REQUIRED_HYBRID)
    return list(REQUIRED_DENSE)


def mechanism_retire_done(patient_id: str, mechanism_id: str,
                          entries: list | None = None, *,
                          source_revision: str | None = None) -> bool:
    entry = current_completion(patient_id, mechanism_id, entries)
    if not entry:
        return False
    if entry.get("status") not in RETIRE_TERMINAL:
        return False
    if reopen_if_satisfied(entry, source_revision=source_revision):
        return False
    return True


def missing_required(oxx: str, state: dict | None = None,
                     entries: list | None = None,
                     receipt_dir: Path | None = None) -> list[str]:
    pool = entries if entries is not None else _completions_entries(None)
    miss = [
        m for m in required_mechanisms(oxx, state)
        if not mechanism_retire_done(oxx, m, pool)
    ]
    # LOW_INFORMATION_VALUE is the sole exception that waives the aggressive probe.
    if miss and has_low_information_value(oxx, receipt_dir=receipt_dir):
        miss = [m for m in miss if not is_aggressive_mechanism(m)]
    return miss


def retire_eligible(oxx: str, state: dict | None = None,
                    entries: list | None = None,
                    receipt_dir: Path | None = None) -> bool:
    """True iff every required mechanism is terminal AND the retirement_gate passes.

    DEFAULT REFUSE when a conventional gravity anchor exists and no aggressive
    probe has been attempted (VERIFIED or REFUTED still counts), unless an
    explicit LOW_INFORMATION_VALUE receipt exists for the patient.
    """
    st = state if state is not None else ensure_state()
    meta = patient_meta(oxx, st)
    if meta.get("state") == "RETIRED":
        return False
    if science_is_done(oxx, "patient-sealed", entries):
        return False
    miss = missing_required(oxx, st, entries, receipt_dir=receipt_dir)
    if miss:
        return False
    if retirement_gate_reason(oxx, entries, receipt_dir=receipt_dir):
        return False
    return True


def _patient_row(state: dict, oxx: str) -> dict | None:
    for p in state.setdefault("patients", []):
        if p.get("oxx") == oxx:
            return p
    return None


def retire_patient(oxx: str, *, dry_run: bool = False, persist: bool = True,
                   state: dict | None = None, index: dict | None = None,
                   completed_at: str | None = None,
                   receipt_dir: Path | None = None) -> dict:
    """Seal a retire-eligible patient. Does NOT delete weights."""
    oxx = norm_oxx(oxx)
    st = state if state is not None else ensure_state()
    entries = (index.get("entries") if isinstance(index, dict) else index)
    if entries is None:
        entries = _completions_entries(index)
    meta = patient_meta(oxx, st)
    if meta.get("state") == "RETIRED" or science_is_done(oxx, "patient-sealed", entries):
        return {
            "schema": SEAL_SCHEMA,
            "oxx": oxx,
            "verdict": "SKIP",
            "reason": "already RETIRED",
            "_evidence": "DERIVED (patient-sealed)",
        }
    miss = missing_required(oxx, st, entries, receipt_dir=receipt_dir)
    if miss:
        return {
            "schema": SEAL_SCHEMA,
            "oxx": oxx,
            "verdict": "REFUSE",
            "reason": "not retire-eligible; missing: " + ",".join(miss),
            "missing": miss,
            "_evidence": "DERIVED (required-obligation set)",
        }
    gate_reason = retirement_gate_reason(oxx, entries, receipt_dir=receipt_dir)
    if gate_reason:
        return {
            "schema": SEAL_SCHEMA,
            "oxx": oxx,
            "verdict": "REFUSE",
            "reason": "not retire-eligible; " + gate_reason,
            "missing": ["aggressive_probe"],
            "_evidence": "DERIVED (ODYSSEY_POLICY.retirement_gate)",
        }
    req = required_mechanisms(oxx, st)
    refs = []
    for m in req:
        ent = current_completion(oxx, m, entries)
        if ent and ent.get("receipt_ref"):
            refs.append(ent["receipt_ref"])
    head = git_head()
    stamp = (
        completed_at
        or os.environ.get("ODYSSEY_COMPLETED_AT")
        or utc_now()
    )
    out_dir = Path(receipt_dir) if receipt_dir else RECEIPT_DIR
    dest = out_dir / f"{oxx}_PATIENT_SEAL.json"
    try:
        rec_rel = str(dest.resolve().relative_to(REPO.resolve())).replace("\\", "/")
    except ValueError:
        rec_rel = f"receipts/odyssey-i/{oxx}_PATIENT_SEAL.json"
    seal = {
        "schema": SEAL_SCHEMA,
        "oxx": oxx,
        "status": "VERIFIED",
        "phase": "SEALED",
        "state": "RETIRED",
        "sealed_mechanisms": req,
        "receipt_refs": refs,
        "source_revision": head,
        "reclaimable": True,
        "completed_at": stamp,
        "weights_deleted": False,
        "_evidence": "DERIVED (patient-sealed from terminal completions)",
    }
    if dry_run:
        return {
            "schema": SEAL_SCHEMA,
            "oxx": oxx,
            "verdict": "DRY-RUN",
            "reason": "would retire",
            "receipt": rec_rel,
            "seal": seal,
            "_evidence": "DERIVED (retire dry-run)",
        }
    write_json(dest, seal)
    complete(
        obligation_id=f"{oxx}:patient-sealed",
        patient_id=oxx,
        mechanism_id="patient-sealed",
        status="VERIFIED",
        completed_at=stamp,
        receipt_ref=rec_rel,
        receipt_sha256=file_sha256(dest),
        source_revision=head,
        index=index,
        persist=persist,
    )
    row = _patient_row(st, oxx)
    if row is not None:
        row["state"] = "RETIRED"
        row["phase"] = "SEALED"
        row["reclaimable"] = True
        row["_evidence"] = "DERIVED (patient-sealed)"
    pkt = load_packet(oxx)
    if pkt is None:
        pkt = assemble_packet(oxx, st)
    pkt["phase"] = "SEALED"
    pkt["state"] = "RETIRED"
    pkt["reclaimable"] = True
    if evidence_class(pkt.get("_evidence")) is None:
        pkt["_evidence"] = "DERIVED (patient-sealed)"
    dest_pkt = packet_path(oxx)
    dest_pkt.parent.mkdir(parents=True, exist_ok=True)
    write_json(dest_pkt, pkt)
    if persist:
        save_state(st)
    record_ctl_event(oxx, "retirement", 0.0, persist=persist)
    return {
        "schema": SEAL_SCHEMA,
        "oxx": oxx,
        "verdict": "VERIFIED",
        "reason": "sealed",
        "receipt": rec_rel,
        "reclaimable": True,
        "_evidence": "DERIVED (patient-sealed)",
    }


def cmd_retire(oxx: str) -> int:
    rec = retire_patient(oxx)
    ev = rec.get("_evidence") or "DERIVED"
    if rec.get("verdict") == "REFUSE":
        print(f"REFUSE  retire {oxx}  {rec.get('reason')}  _evidence={ev}")
        return 1
    if rec.get("verdict") == "SKIP":
        print(f"SKIP  retire {oxx}  {rec.get('reason')}  _evidence={ev}")
        return 0
    print(
        f"VERIFIED  retire {oxx}  sealed  receipt={rec.get('receipt')}  "
        f"reclaimable=true  _evidence={ev}"
    )
    return 0


# ---------------------------------------------------------------------------
# next-patient acquisition
# ---------------------------------------------------------------------------

def hf_executable() -> str:
    if HF_BIN.is_file():
        return str(HF_BIN)
    return "hf"


def hf_model_info(repo: str, *, timeout: int = 45) -> tuple[bool, dict | None, str]:
    """Attempt `hf models info`. Success ⇒ token+license is enough to see metadata."""
    if not repo or "/" not in repo or repo.lower().startswith("reconstruct"):
        return False, None, "not an hf repo"
    try:
        r = subprocess.run(
            [hf_executable(), "models", "info", repo, "--format", "json"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, None, f"hf metadata failed: {exc}"
    if r.returncode != 0:
        err = ((r.stderr or "") + (r.stdout or "")).strip()[-400:]
        return False, None, err or "hf metadata failed"
    try:
        info = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return True, None, "hf metadata ok (non-json)"
    return True, info if isinstance(info, dict) else None, "ok"


def hf_cache_snapshot(repo: str) -> Path | None:
    if not repo or "/" not in repo:
        return None
    slug = "models--" + repo.replace("/", "--")
    snaps = HF_HUB / slug / "snapshots"
    if not snaps.is_dir():
        return None
    hub = HF_HUB / slug
    if any(hub.rglob("*.incomplete")):
        return None
    ranked = sorted(
        (p for p in snaps.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for snap in ranked:
        if not (snap / "config.json").is_file():
            continue
        if list(snap.glob("*.safetensors")) or list(snap.glob("*.bin")):
            return snap
    return None


def patient_est_gib(oxx: str, meta: dict | None = None,
                    info: dict | None = None) -> float:
    if info and info.get("used_storage"):
        try:
            return float(info["used_storage"]) / 1024**3
        except (TypeError, ValueError):
            pass
    census = load_census(oxx)
    if census and census.get("total_bytes"):
        return float(census["total_bytes"]) / 1024**3
    man = overlay_manifest(meta, oxx) if meta or oxx else manifest_entry(oxx)
    for key in ("est_source_gib", "est_gib_hf"):
        v = (man or {}).get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            return float(v)
    return float(PATIENT_EST_GIB.get(oxx, 50.0))


def _hf_gated(meta: dict) -> bool:
    if meta.get("gated") is True:
        return True
    gate = str(meta.get("gate") or "")
    return "HF-gated" in gate or "HF-gated" in str(meta.get("blocked_reason") or "")


def _downloadable_repo(meta: dict) -> str:
    src = strip_md(
        meta.get("canonical_source") or meta.get("source") or ""
    )
    if not src or "/" not in src or src.lower().startswith("reconstruct"):
        return ""
    if src.startswith("http"):
        return ""
    return src


def _manifest_acquire_fields(oxx: str, meta: dict | None = None) -> dict:
    man = overlay_manifest(meta, oxx)
    return {
        "canonical_source": man.get("canonical_source") or _downloadable_repo(man),
        "est_source_gib": man.get("est_source_gib"),
        "est_4bit_gib": man.get("est_4bit_gib"),
        "gated": man.get("gated"),
        "reference_sibling": man.get("reference_sibling"),
        "search_class": man.get("search_class"),
        "info_budget": man.get("info_budget"),
        "arch_objective": man.get("arch_objective"),
        "stored_bpw_pressure": man.get("stored_bpw_pressure"),
        "active_bpw_pressure": man.get("active_bpw_pressure"),
        "doctor_bar": man.get("doctor_bar"),
        "kernel_effort": man.get("kernel_effort"),
    }


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def weights_dir_for_reclaim(oxx: str) -> Path | None:
    census = load_census(oxx)
    if census and census.get("model_dir"):
        p = Path(census["model_dir"])
        if p.is_dir():
            return p
    pkt = load_packet(oxx) or {}
    on_disk = (pkt.get("identity") or {}).get("on_disk")
    if on_disk:
        p = Path(os.path.expanduser(str(on_disk)))
        if p.is_dir():
            return p
    meta = patient_meta(oxx)
    snap = hf_cache_snapshot(_downloadable_repo(meta))
    return snap


def reclaim_retired_weights(oxx: str, *, dry_run: bool = False,
                            persist: bool = True,
                            state: dict | None = None) -> dict:
    """Delete RETIRED patient weights. Record provenance. Separate from retire()."""
    st = state if state is not None else ensure_state()
    meta = patient_meta(oxx, st)
    if meta.get("state") != "RETIRED" and not meta.get("reclaimable"):
        return {
            "verdict": "REFUSE",
            "oxx": oxx,
            "reason": "not RETIRED/reclaimable",
            "_evidence": "DERIVED (reclaim gate)",
        }
    path = weights_dir_for_reclaim(oxx)
    if path is None or not path.exists():
        return {
            "verdict": "SKIP",
            "oxx": oxx,
            "reason": "no on-disk weights to reclaim",
            "_evidence": "MEASURED (weights missing)",
        }
    size = 0
    if path.is_dir():
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    size += f.stat().st_size
                except OSError:
                    pass
    rec = {
        "schema": ACQUIRE_SCHEMA,
        "oxx": oxx,
        "action": "reclaim-weights",
        "path": str(path),
        "bytes": size,
        "at": utc_now(),
        "source_revision": git_head(),
        "dry_run": dry_run,
        "_evidence": "MEASURED (reclaim of RETIRED patient weights)",
    }
    dest = RECEIPT_DIR / f"{oxx}_WEIGHTS_RECLAIM.json"
    if dry_run:
        rec["verdict"] = "DRY-RUN"
        return rec
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.is_file():
        path.unlink(missing_ok=True)
    write_json(dest, rec)
    row = _patient_row(st, oxx)
    if row is not None:
        row["on_disk"] = False
        row["ledger"] = "reclaimed"
        row["reclaim_ref"] = f"receipts/odyssey-i/{oxx}_WEIGHTS_RECLAIM.json"
    st.setdefault("reclaim_log", []).append({
        "oxx": oxx, "path": str(path), "bytes": size, "at": rec["at"],
        "_evidence": rec["_evidence"],
    })
    if persist:
        save_state(st)
    rec["verdict"] = "VERIFIED"
    rec["receipt"] = str(dest)
    return rec


def run_patient_census(oxx: str, model_dir: str) -> tuple[bool, str]:
    dest = census_path(oxx)
    dest.parent.mkdir(parents=True, exist_ok=True)
    py = PREFERRED_PY if Path(PREFERRED_PY).is_file() else sys.executable
    r = subprocess.run(
        [py, str(TOOLS / "odyssey_census.py"), str(model_dir), "--out", str(dest)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return False, ((r.stderr or "") + (r.stdout or ""))[-400:]
    return dest.is_file(), str(dest)


def finalize_acquisitions(state: dict, *, dry_run: bool = False,
                          persist: bool = True) -> list[dict]:
    """If an ACQUIRING download landed, census + seed packet + mark on-disk."""
    rows = []
    for p in state.get("patients") or []:
        if p.get("state") != "ACQUIRING":
            continue
        oxx = p.get("oxx")
        repo = _downloadable_repo(p)
        snap = hf_cache_snapshot(repo) if repo else None
        rec = {
            "oxx": oxx, "repo": repo, "snapshot": str(snap) if snap else None,
            "verdict": "WAIT",
            "_evidence": "MEASURED (hf cache)",
        }
        if snap is None:
            pid = p.get("acquire_pid")
            if pid and not _pid_alive(pid):
                rec["verdict"] = "STALE"
                rec["reason"] = f"download pid {pid} dead and snapshot incomplete"
            rows.append(rec)
            continue
        if dry_run:
            rec["verdict"] = "DRY-RUN"
            rec["reason"] = "would census + seed packet"
            rows.append(rec)
            continue
        ok, note = run_patient_census(oxx, str(snap))
        if not ok:
            rec["verdict"] = "WAIT"
            rec["reason"] = f"census failed: {note}"
            rows.append(rec)
            continue
        try:
            write_packet(oxx, state)
        except SystemExit as exc:
            rec["verdict"] = "WAIT"
            rec["reason"] = f"packet seed failed: {exc}"
            rows.append(rec)
            continue
        p["on_disk"] = True
        p["state"] = "READY"
        p["ledger"] = "on-disk"
        p["phase"] = "CENSUS"
        p["blocked_reason"] = None
        p["_evidence"] = "MEASURED (hf snapshot + census)"
        rec["verdict"] = "READY"
        rec["reason"] = "on-disk; obligations registered via synthesis"
        rows.append(rec)
    if persist and any(r.get("verdict") == "READY" for r in rows) and not dry_run:
        save_state(state)
    return rows


def pick_acquire_candidate(state: dict, *,
                           hf_info_fn=None,
                           mutate: bool = True) -> tuple[dict | None, dict]:
    """Lowest-numbered ladder patient not on disk, not RETIRED, not blocked.

    HF-gated patients are probed via hf metadata; failure marks BLOCKED and skips.
    """
    info_fn = hf_info_fn or hf_model_info
    skipped = []
    mutated = False
    for p in state.get("patients") or []:
        oxx = p.get("oxx")
        if not oxx:
            continue
        man = manifest_entry(oxx)
        if man:
            if man.get("canonical_source"):
                p["canonical_source"] = man["canonical_source"]
                p["source"] = man["canonical_source"]
            if man.get("gated") is not None:
                p["gated"] = bool(man["gated"])
            if man.get("est_source_gib") is not None:
                p["est_source_gib"] = man["est_source_gib"]
                p["est_gib_hf"] = man["est_source_gib"]
            for key in (
                "search_class", "info_budget", "arch_objective",
                "reference_sibling", "stored_bpw_pressure", "active_bpw_pressure",
                "doctor_bar", "kernel_effort", "est_4bit_gib",
            ):
                if man.get(key) is not None:
                    p[key] = man[key]
        if p.get("state") == "RETIRED":
            skipped.append((oxx, "RETIRED"))
            continue
        if p.get("state") == "ACQUIRING":
            skipped.append((oxx, "already ACQUIRING"))
            continue
        if patient_on_disk(p) or p.get("on_disk"):
            skipped.append((oxx, "on-disk"))
            continue
        repo = _downloadable_repo(p)
        gated = _hf_gated(p) or p.get("state") == "BLOCKED"
        if gated:
            if not repo:
                if mutate:
                    p["state"] = "BLOCKED"
                    p["blocked_reason"] = p.get("blocked_reason") or "HF-gated / no repo"
                    mutated = True
                skipped.append((oxx, "HF-gated, no repo"))
                continue
            ok, info, why = info_fn(repo)
            if not ok:
                if mutate:
                    p["state"] = "BLOCKED"
                    p["blocked_reason"] = f"HF-gated / hf metadata failed: {why}"
                    p["_evidence"] = "INFERRED (hf metadata)"
                    mutated = True
                skipped.append((oxx, f"HF-gated blocked: {why}"))
                continue
            return p, {"skipped": skipped, "mutated": mutated, "hf_info": info}
        if p.get("state") == "BLOCKED":
            skipped.append((oxx, "BLOCKED"))
            continue
        if not repo:
            skipped.append((oxx, "no downloadable hf repo"))
            continue
        return p, {"skipped": skipped, "mutated": mutated, "hf_info": None}
    return None, {"skipped": skipped, "mutated": mutated, "hf_info": None}


def start_hf_download(repo: str, log_path: Path) -> tuple[int | None, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("w")
    env = dict(os.environ)
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    try:
        proc = subprocess.Popen(
            [hf_executable(), "download", repo],
            stdout=fh, stderr=subprocess.STDOUT,
            cwd=str(REPO), env=env, start_new_session=True,
        )
    except OSError as exc:
        fh.close()
        return None, str(exc)
    return proc.pid, str(log_path)


def acquire_next(*, go: bool = False, dry_run: bool | None = None,
                 state: dict | None = None, persist: bool = True,
                 snapshot_fn=None, hf_info_fn=None, download_fn=None,
                 reclaim_fn=None, log_path: Path | None = None) -> dict:
    """Pick + (optionally) start the next ladder patient download.

    Never blocks the caller on a long download. Marks ACQUIRING and returns.
    """
    st = state if state is not None else ensure_state()
    planning = (not go) if dry_run is None else bool(dry_run)
    finalize_acquisitions(st, dry_run=planning, persist=persist and not planning)
    snap = (snapshot_fn or machine_snapshot)()
    disk = float(snap.get("disk_free_gib") or 0.0)
    cand, meta = pick_acquire_candidate(
        st, hf_info_fn=hf_info_fn, mutate=not planning,
    )
    if persist and meta.get("mutated") and not planning:
        save_state(st)
    if cand is None:
        rec = {
            "schema": ACQUIRE_SCHEMA,
            "verdict": "REFUSE",
            "reason": "no eligible patient (all on-disk, RETIRED, ACQUIRING, or blocked)",
            "skipped": meta.get("skipped"),
            "disk_free_gib": disk,
            "_evidence": "DERIVED (acquire-next)",
        }
        append_run_log({**rec, "command": "acquire-next"}, path=log_path)
        return rec
    oxx = cand["oxx"]
    repo = _downloadable_repo(cand)
    est = cand.get("est_gib_hf") or patient_est_gib(oxx, cand, meta.get("hf_info"))
    need = est + DISK_RUN_GIB
    man_fields = _manifest_acquire_fields(oxx, cand)
    reclaimed = []
    # Auto-deleting RETIRED patients' downloaded weights to free disk for a new
    # acquisition is OFF by default: it destroyed active-campaign weights during
    # a disk-pressure event (14G Falcon-H1 gone, Kimi-VL snapshot broken). Weight
    # deletion is a HUMAN decision now. When disk is short, acquisition simply
    # waits. Opt in with ODYSSEY_AUTO_RECLAIM_WEIGHTS=1 only if you want it back.
    auto_reclaim = os.environ.get("ODYSSEY_AUTO_RECLAIM_WEIGHTS") == "1"
    if disk < need and auto_reclaim:
        for p in list(st.get("patients") or []):
            if p.get("state") != "RETIRED":
                continue
            if not (p.get("on_disk") or p.get("reclaimable")):
                continue
            fn = reclaim_fn or (
                lambda o, **kw: reclaim_retired_weights(
                    o, dry_run=planning, persist=persist and not planning, state=st,
                )
            )
            rec_r = fn(p.get("oxx"), dry_run=planning, persist=persist and not planning)
            reclaimed.append(rec_r)
            if not planning and rec_r.get("verdict") == "VERIFIED":
                snap = (snapshot_fn or machine_snapshot)()
                disk = float(snap.get("disk_free_gib") or 0.0)
                if disk >= need:
                    break
    # Disk-hold REFUSE always applies when disk is short (independent of the
    # opt-in reclaim loop above): never start a download that will not fit.
    if disk < need:
        rec = {
            "schema": ACQUIRE_SCHEMA,
            "verdict": "REFUSE",
            "reason": (
                f"disk-hold: free {disk:.1f} GiB < est {est:.1f} + "
                f"floor {DISK_RUN_GIB:.0f}"
            ),
            "oxx": oxx,
            "repo": repo,
            "est_gib": round(est, 2),
            "need_gib": round(need, 2),
            "disk_free_gib": disk,
            "reclaimed": reclaimed,
            **man_fields,
            "_evidence": "MEASURED (disk) + DERIVED (disk-hold)",
        }
        append_run_log({**rec, "command": "acquire-next"}, path=log_path)
        return rec
    rec = {
        "schema": ACQUIRE_SCHEMA,
        "oxx": oxx,
        "repo": repo,
        "est_gib": round(est, 2),
        "need_gib": round(need, 2),
        "disk_free_gib": disk,
        "reclaimed": reclaimed,
        **man_fields,
        "_evidence": "HYPOTHESIS (acquire plan) + MEASURED (disk)",
    }
    if planning:
        rec["verdict"] = "DRY-RUN"
        rec["reason"] = f"would acquire {oxx} ({repo})"
        append_run_log({**rec, "command": "acquire-next"}, path=log_path)
        return rec
    # Auto-DOWNLOAD is OFF by default. The autonomous loop re-downloaded models
    # the user had just deleted for storage (and started pulling a 72B) the
    # moment disk freed up. Model downloads are a HUMAN decision — the loop plans
    # the acquisition but never fetches unless explicitly opted in.
    if os.environ.get("ODYSSEY_AUTO_ACQUIRE") != "1":
        rec["verdict"] = "HOLD"
        rec["reason"] = (
            f"auto-acquire disabled: would download {oxx} ({repo}, ~{est:.0f} GiB); "
            "set ODYSSEY_AUTO_ACQUIRE=1 or acquire manually to proceed"
        )
        append_run_log({**rec, "command": "acquire-next"}, path=log_path)
        return rec
    logf = DOWNLOADS / f"{oxx}_{(repo or 'repo').replace('/', '_')}.log"
    fn = download_fn or start_hf_download
    pid, note = fn(repo, logf)
    if not pid:
        rec["verdict"] = "REFUSE"
        rec["reason"] = f"hf download failed to start: {note}"
        append_run_log({**rec, "command": "acquire-next"}, path=log_path)
        return rec
    cand["state"] = "ACQUIRING"
    cand["ledger"] = "acquiring"
    cand["acquire_pid"] = pid
    cand["acquire_log"] = str(logf)
    cand["acquire_started"] = utc_now()
    cand["_evidence"] = "MEASURED (hf download launched)"
    st.setdefault("acquisitions", []).append({
        "oxx": oxx, "repo": repo, "pid": pid, "log": str(logf),
        "started": cand["acquire_started"],
        "_evidence": cand["_evidence"],
    })
    if persist:
        save_state(st)
    rec["verdict"] = "ACQUIRING"
    rec["reason"] = f"download started pid={pid}; cycle will pick up when on-disk"
    rec["pid"] = pid
    rec["log"] = str(logf)
    append_run_log({**rec, "command": "acquire-next"}, path=log_path)
    record_ctl_event(oxx, "acquisition", 0.0, persist=persist)
    return rec


def cmd_acquire_next(*, go: bool = False) -> int:
    rec = acquire_next(go=go)
    ev = rec.get("_evidence") or "DERIVED"
    verdict = rec.get("verdict") or "REFUSE"
    reason = rec.get("reason") or ""
    print(f"{verdict}  acquire-next  {reason}  _evidence={ev}")
    if rec.get("oxx"):
        print(
            f"  oxx={rec.get('oxx')} repo={rec.get('repo')}  "
            f"est={rec.get('est_gib')}GiB need={rec.get('need_gib')}GiB  "
            f"disk={rec.get('disk_free_gib')}GiB"
        )
        if rec.get("canonical_source"):
            print(
                f"  manifest: source={rec.get('canonical_source')}  "
                f"search_class={rec.get('search_class')}  "
                f"info_budget={rec.get('info_budget')}  "
                f"arch_objective={rec.get('arch_objective')}  "
                f"gated={rec.get('gated')}"
            )
    return 0 if verdict in {"ACQUIRING", "DRY-RUN", "SKIP"} else 1


# ---------------------------------------------------------------------------
# cycle — one unattended tick: reap → harvest → complete → retire → acquire → admit
# ---------------------------------------------------------------------------

def cycle_tick(*, go: bool, max_lanes: int, grok_lanes: int = 0,
               state: dict | None = None, persist: bool = True,
               **hooks) -> dict:
    """One driver tick. Idempotent, event-safe, no model calls in the controller."""
    st = state if state is not None else ensure_state()
    now = hooks.get("now_epoch")
    if now is None:
        now = time.time()
    reap_rows = reap_lanes(
        float(now), state=st, persist=persist and go, dry_run=not go,
        pid_alive_fn=hooks.get("pid_alive_fn"),
        kill_fn=hooks.get("kill_fn"),
    )
    harvest_hooks = {
        k: hooks[k] for k in (
            "tasks_root", "receipt_dir", "worktrees_root", "dest_root",
            "review_queue", "cleanup_fn",
        ) if k in hooks
    }
    harvest_rows = harvest_lanes(
        dry_run=not go, state=st, persist=persist and go, **harvest_hooks,
    )
    completions = rebuild_completions(persist=persist)
    entries = list(completions.get("entries") or [])

    eligible = []
    missing_map = {}
    for p in st.get("patients") or []:
        oxx = p.get("oxx")
        if not oxx or p.get("state") in {"RETIRED", "BLOCKED"}:
            continue
        if not (patient_on_disk(p) or p.get("on_disk")):
            continue
        miss = missing_required(oxx, st, entries)
        gate_reason = retirement_gate_reason(oxx, entries) if not miss else None
        if gate_reason:
            miss = list(miss) + ["aggressive_probe"]
        missing_map[oxx] = miss
        if not miss:
            eligible.append(oxx)

    retired = []
    for oxx in eligible:
        rec = retire_patient(
            oxx, dry_run=not go, persist=persist and go, state=st,
            index=completions,
        )
        retired.append(rec)
        if rec.get("verdict") == "VERIFIED":
            entries = list((completions.get("entries") or entries))

    finalized = finalize_acquisitions(st, dry_run=not go, persist=persist and go)

    ranked = select_ready_obligations(st, completions=completions)
    acquire_row = None
    if not ranked:
        acquire_row = acquire_next(
            go=go, dry_run=not go, state=st, persist=persist and go,
            snapshot_fn=hooks.get("snapshot_fn"),
            hf_info_fn=hooks.get("hf_info_fn"),
            download_fn=hooks.get("download_fn"),
            reclaim_fn=hooks.get("reclaim_fn"),
            log_path=hooks.get("log_path"),
        )
        ranked = select_ready_obligations(st, completions=completions)

    run_hooks = {
        k: hooks[k] for k in (
            "observe_fn", "gate_fn", "snapshot_fn", "launch_fn",
            "lint_fn", "reclaim_fn", "log_path", "auto_dir",
            "spawn_fn", "now_epoch", "pid_alive_fn", "kill_fn",
        ) if k in hooks
    }
    run_hooks.setdefault("now_epoch", now)
    rows = run_loop(
        go=go, max_lanes=max_lanes, grok_lanes=grok_lanes, state=st,
        persist=persist, completions=completions, consider_limit=24, **run_hooks,
    )
    without_probe = count_retired_without_nonconventional_probe(st, entries)
    tick_without = 0
    for rec in retired:
        if rec.get("verdict") not in {"VERIFIED", "DRY-RUN"}:
            continue
        if not aggressive_probe_attempted(rec.get("oxx"), entries):
            tick_without += 1
    return {
        "schema": CYCLE_SCHEMA,
        "go": go,
        "harvest": harvest_rows,
        "retire_eligible": eligible,
        "missing": missing_map,
        "retired": retired,
        "finalized": finalized,
        "acquire": acquire_row,
        "ready": ranked,
        "admitted": rows,
        "reap": reap_rows,
        "patients_retired_without_nonconventional_probe": without_probe,
        "tick_retired_without_nonconventional_probe": tick_without,
        "_evidence": "DERIVED (cycle tick)",
    }


def _retire_none_reason(missing_map: dict) -> str:
    if not missing_map:
        return "(none)"
    need_gn = True
    for miss in missing_map.values():
        if not any(m.startswith("gravity-") or m.startswith("nx-") for m in miss):
            need_gn = False
            break
    if need_gn:
        return "(none — all need gravity/nx)"
    return "(none)"


def print_cycle(plan: dict, *, go: bool, max_lanes: int,
                snap: dict, running_n: int) -> None:
    mode = "GO" if go else "dry-run"
    disk = snap.get("disk_free_gib")
    eligible = plan.get("retire_eligible") or []
    acquire = plan.get("acquire")
    ready = plan.get("ready") or []
    harvest = plan.get("harvest") or []
    acq_s = "skipped (ready frontier non-empty)"
    if acquire:
        acq_s = f"{acquire.get('verdict')}: {acquire.get('reason')}"
    print(
        f"ODYSSEY-I  {mode}  on-disk-ready={len(plan.get('missing') or {})}  "
        f"running={running_n}  ready={len(ready)}  retire-eligible={len(eligible)}  "
        f"disk={disk}GiB  _evidence=DERIVED (§97)"
    )
    without = plan.get("patients_retired_without_nonconventional_probe")
    desired = (
        (load_odyssey_policy().get("retirement_gate") or {}).get("cycle_metric")
        or "PATIENTS_RETIRED_WITHOUT_NONCONVENTIONAL_PROBE desired 0"
    )
    print(
        f"ODYSSEY CYCLE  {mode}  max_lanes={max_lanes}  harvest={len(harvest)}  "
        f"retire-eligible={len(eligible)}  acquire={acq_s}  ready={len(ready)}  "
        f"disk={disk}GiB  patients_retired_without_nonconventional_probe={without}  "
        f"({desired})"
    )
    if not go:
        print("launch NOTHING (dry-run is the default; pass --go to spawn)")
    if eligible:
        print("RETIRE-ELIGIBLE: " + ",".join(eligible))
    else:
        print("RETIRE-ELIGIBLE: " + _retire_none_reason(plan.get("missing") or {}))
    for oxx, miss in (plan.get("missing") or {}).items():
        if miss:
            print(f"  {oxx} missing: {','.join(miss)}")
    if acquire:
        print(
            f"ACQUIRE: {acquire.get('verdict')}  {acquire.get('reason')}  "
            f"_evidence={acquire.get('_evidence')}"
        )
    print_run_plan(
        plan.get("admitted") or [], go=go, max_lanes=max_lanes,
        snap=snap, running_n=running_n,
    )


def cmd_cycle(*, go: bool, max_lanes: int, grok_lanes: int = 0,
              loop_secs: float = 0.0, inner_sleep: float = 3.0,
              **hooks) -> int:
    hooks.setdefault("persist", go)
    # Tight internal loop: reap + fill lanes every inner_sleep seconds inside ONE
    # process for loop_secs, so lanes stay CONSTANTLY full — no per-tick Python
    # startup, no 12s gap between waves. The resident relaunches on exit.
    if go and loop_secs and loop_secs > 0:
        start = time.time()
        n = 0
        while time.time() - start < loop_secs:
            st = ensure_state()
            snap = (hooks.get("snapshot_fn") or machine_snapshot)()
            plan = cycle_tick(go=True, max_lanes=max_lanes,
                              grok_lanes=grok_lanes, state=st, **hooks)
            n += 1
            running_n = len(odyssey_running_ids(st))
            admitted = [r for r in (plan.get("admitted") or [])
                        if r.get("verdict") == "LAUNCH"]
            print(f"tick {n} running={running_n} launched={len(admitted)} "
                  f"disk={snap.get('disk_free_gib')}GiB", flush=True)
            time.sleep(max(0.5, float(inner_sleep)))
        return 0
    st = hooks.pop("state", None) or ensure_state()
    snap = (hooks.get("snapshot_fn") or machine_snapshot)()
    plan = cycle_tick(go=go, max_lanes=max_lanes, grok_lanes=grok_lanes,
                      state=st, **hooks)
    running_n = len(odyssey_running_ids(st))
    print_cycle(plan, go=go, max_lanes=max_lanes, snap=snap, running_n=running_n)
    return 0


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def live_odyssey_lanes() -> list[str]:
    if not GROK_TASKS.is_dir():
        return []
    live = []
    for d in GROK_TASKS.iterdir():
        if d.is_dir() and d.name.startswith("odyssey-") and _task_status(d) == "running":
            live.append(d.name)
    return sorted(live)


def frontier_counts() -> dict:
    out = {
        "universal_rules": 0,
        "architecture_rules": 0,
        "new_negatives": 0,
        "transfer_score": "0/0",
        "_evidence": "INFERRED",
    }
    rules = []
    if RULEBASE.is_file():
        try:
            rules = read_json(RULEBASE).get("rules") or []
        except (OSError, json.JSONDecodeError):
            rules = []
    if NEGATIVE.is_file():
        try:
            out["new_negatives"] = len(read_json(NEGATIVE).get("entries") or [])
        except (OSError, json.JSONDecodeError):
            pass
    tested = total = 0
    arch_rules = set()
    families_ok = {}
    if TRANSFER.is_file():
        try:
            matrix = read_json(TRANSFER)
        except (OSError, json.JSONDecodeError):
            matrix = {}
        for row in matrix.get("rows") or []:
            rid = row.get("rule")
            cells = row.get("cells") or {}
            fams = set()
            for oxx, cell in cells.items():
                total += 1
                if cell != "NOT_TESTED":
                    tested += 1
                if cell == "ARCHITECTURE_SPECIFIC":
                    arch_rules.add(rid)
                if cell == "TRANSFERRED_UNCHANGED":
                    fams.add("moe" if oxx in MOE_PATIENTS else "other")
            families_ok[rid] = fams
    out["architecture_rules"] = len(arch_rules) or sum(
        1 for r in rules if "moe" in json.dumps(r).lower()
    )
    out["universal_rules"] = sum(1 for fams in families_ok.values() if len(fams) >= 2)
    out["transfer_score"] = f"{tested}/{total}" if total else "0/0"
    return out


def ready_work(state: dict) -> list[dict]:
    blocked = {p["oxx"] for p in state.get("patients") or [] if p.get("state") == "BLOCKED"}
    out = []
    for w in state.get("work") or []:
        if w.get("status") != "READY":
            continue
        if w.get("oxx") in blocked and w.get("kind") != "acquisition":
            continue
        if w.get("id") == "A6":
            continue
        out.append(w)
    out.sort(key=value, reverse=True)
    return out


def cmd_status() -> int:
    state = ensure_state()
    patients = {p["oxx"]: p for p in state.get("patients") or []}
    on_disk = [p["oxx"] for p in state["patients"] if p.get("on_disk") or
               str(p.get("ledger", "")).lower().startswith("on-disk")]
    blocked = [p["oxx"] for p in state["patients"] if p.get("state") == "BLOCKED"]
    queued = [p["oxx"] for p in state["patients"]
              if str(p.get("ledger", "")).lower().startswith("queued")]
    # PATIENT of record = already-in-motion teacher (O005), not the cheapest next action.
    # NEXT is where §22 ranking lives.
    ranked = ready_work(state)
    if "O005" in on_disk:
        current = "O005"
    elif on_disk:
        current = on_disk[0]
    elif ranked:
        current = ranked[0]["oxx"]
    else:
        current = "O005"
    meta = patients.get(current) or patient_meta(current, state)
    pkt = load_packet(current) or assemble_packet(current, state)
    doctor = pkt.get("doctor") or {}
    doctor_s = doctor.get("fast_doctor_seal_ref") or "UNKNOWN"
    if doctor_s in (None, ""):
        doctor_s = evidence_class(doctor.get("_evidence")) or "UNKNOWN"
    rep = pkt.get("representation") or {}
    bpw = rep.get("stored_bpw") or rep.get("best_stored_bpw_eq")
    bpw_ev = evidence_class(rep.get("_evidence")) or "UNKNOWN"
    active = rep.get("active_bytes_per_token_bf16") or rep.get("active_learned_bytes_per_token")
    ex = pkt.get("execution") or {}
    tps = ex.get("baseline_tps") or ex.get("tps")
    tps_ev = evidence_class(ex.get("_evidence")) or "UNKNOWN"
    finding = (
        (rep.get("lever") if isinstance(rep.get("lever"), str) else None)
        or (meta.get("class") or "")
    )
    fc = frontier_counts()
    lanes = live_odyssey_lanes()
    esc_n = 0
    if ESCALATIONS.is_file():
        esc_n = sum(1 for line in ESCALATIONS.read_text().splitlines() if line.strip())
    nxt = [w for w in ranked[:3]]
    next_on_disk = [x for x in on_disk if x != current]
    print("HAWKING ODYSSEY-I")
    print()
    print("QUEUE:")
    print(f"    on-disk: {','.join(on_disk) or '—'}")
    print(f"    BLOCKED-auth: {','.join(blocked) or '—'}")
    print(f"    queued: {','.join(queued) or '—'}")
    print()
    print("PATIENT:")
    print(f"    {current}")
    print(f"    phase: {pkt.get('phase') or meta.get('phase') or '—'}")
    print(f"    source: {(pkt.get('identity') or {}).get('source_repo') or meta.get('source') or '—'}")
    print(f"    Doctor: {doctor_s}")
    print(f"    stored BPW: {bpw if bpw is not None else 'UNKNOWN'}"
          + (f" {bpw_ev}" if bpw is not None else ""))
    print(f"    active bytes/token: {active if active is not None else 'UNKNOWN'}"
          + (f" {bpw_ev}" if active is not None else ""))
    tps_s = "UNKNOWN" if tps in (None, "", "UNKNOWN") or (
        isinstance(tps, str) and tps.startswith("UNKNOWN")
    ) else tps
    print(f"    TPS: {tps_s}" + (f" {tps_ev}" if tps_s != "UNKNOWN" else ""))
    print(f"    top finding: {finding}")
    print()
    print("NEXT PATIENT:")
    acq = ", ".join(next_on_disk) if next_on_disk else "—"
    print(f"    acquisition: {acq}; queued: {','.join(queued) or '—'}")
    print(f"    static recon: {'/'.join(blocked) or '—'} BLOCKED-auth")
    print()
    print("COMPILER:")
    print(f"    universal rules: {fc['universal_rules']} {fc['_evidence']}")
    print(f"    architecture rules: {fc['architecture_rules']} {fc['_evidence']}")
    print(f"    new negatives: {fc['new_negatives']} {fc['_evidence']}")
    print(f"    transfer score: {fc['transfer_score']} {fc['_evidence']}")
    print()
    print("RESEARCH:")
    print(f"    Grok lanes: {len(lanes)} MEASURED (live odyssey-*)")
    print("    deterministic jobs: 0")
    print(f"    GPU owner: {(state.get('metrics') or {}).get('gpu_owner', 'none')}")
    print(f"    Opus escalations: {esc_n}")
    print()
    print("NEXT:")
    if not nxt:
        print("    (empty READY queue)")
    for i, w in enumerate(nxt, 1):
        print(f"    {i}. {w['id']} {w['oxx']} {w['title']}  "
              f"proxy={value(w):.1f} HYPOTHESIS")
    return 0


def cmd_queue() -> int:
    state = ensure_state()
    print(f"{'Oxx':<6}{'state':<12}{'phase':<10}{'ledger':<28}model")
    for p in state.get("patients") or []:
        st = p.get("state") or "?"
        if st not in STATES:
            continue
        print(f"{p['oxx']:<6}{st:<12}{(p.get('phase') or '—'):<10}"
              f"{(p.get('ledger') or '—'):<28}{p.get('model') or ''}")
    return 0


def cmd_value() -> int:
    state = ensure_state()
    ranked = ready_work(state)
    print("info-value proxy = info / (wall+gpu+opus)   HYPOTHESIS (§22, ordering only)")
    print(f"{'proxy':>6}  {'id':<10}{'oxx':<6}{'kind':<32}title")
    for w in ranked:
        print(f"{value(w):6.1f}  {w['id']:<10}{w['oxx']:<6}{(w.get('kind') or ''):<32}{w['title']}")
    return 0


def cmd_harvest(*, dry_run: bool = False) -> int:
    st = ensure_state()
    reap_lanes(time.time(), state=st, persist=not dry_run, dry_run=dry_run)
    rows = harvest_lanes(dry_run=dry_run, state=st)
    mode = "dry-run" if dry_run else "apply"
    if not rows:
        print(f"harvest ({mode}): no finished odyssey-* lanes")
        return 0
    print(f"HARVEST  mode={mode}  finished={len(rows)}")
    for r in rows:
        files = ", ".join(r.get("files") or []) or "—"
        scope = "RUNNING" if r.get("in_scope") else "not-RUNNING"
        if r.get("applied"):
            apply = "applied"
        elif dry_run and (
            r.get("in_scope")
            or (
                r.get("classification") == "DATA-ONLY"
                and str(r.get("action") or "").startswith("MERGE")
                and r.get("verdict") != "SKIP"
            )
        ):
            apply = "would-apply"
        elif dry_run:
            apply = "would-skip-apply"
        else:
            apply = "skip-apply"
        print(f"  {r['task']}")
        print(
            f"    class={r.get('classification') or '—'}  "
            f"action={r.get('action')}  {scope}  {apply}"
        )
        print(f"    files={files}")
        print(
            f"    worktree={r.get('worktree') or 'missing'}  "
            f"report={'yes' if r.get('report') else 'no'}  "
            f"oxx={r.get('oxx') or '—'}  cleanup={r.get('cleanup')}"
        )
        if r.get("reason"):
            print(f"    reason={r['reason']}")
        print(f"    _evidence={r.get('_evidence')}")
    return 0


def cmd_packet(oxx: str) -> int:
    dest = write_packet(oxx)
    print(f"wrote {dest.relative_to(REPO)}  valid")
    return 0


def cmd_completions(*, rebuild: bool = False, completed_at: str | None = None) -> int:
    stamp = completed_at or os.environ.get("ODYSSEY_COMPLETED_AT") or None
    if rebuild:
        doc = rebuild_completions(completed_at=stamp)
        entries = doc.get("entries") or []
        print(f"ODYSSEY COMPLETIONS  rebuilt  n={len(entries)}  path={COMPLETIONS}")
        for e in entries:
            sha = (e.get("receipt_sha256") or "")[:12]
            print(
                f"  {e.get('patient_id')}  {e.get('mechanism_id')}  "
                f"{e.get('status')}  receipt={e.get('receipt_ref')}  "
                f"sha256={sha}  at={e.get('completed_at')}"
            )
        return 0
    doc = load_completions()
    entries = doc.get("entries") or []
    print(f"ODYSSEY COMPLETIONS  n={len(entries)}  path={COMPLETIONS}")
    if not entries:
        print("  (empty — run `completions --rebuild` to backfill from receipts)")
        return 0
    for e in entries:
        print(
            f"  {e.get('patient_id')}  {e.get('mechanism_id')}  {e.get('status')}  "
            f"reopen_if={e.get('reopen_if')!r}  receipt={e.get('receipt_ref')}"
        )
    return 0


def cmd_economics() -> int:
    """Print costmodel.derive() per patient + detachment_metrics()."""
    patients: list[str] = []
    if STATE.is_file():
        try:
            for p in (read_json(STATE).get("patients") or []):
                if p.get("oxx"):
                    patients.append(p["oxx"])
        except (OSError, json.JSONDecodeError):
            patients = []
    if not patients:
        patients = [e.get("oxx") for e in load_odyssey_manifest() if e.get("oxx")]
    derived = {}
    for oxx in patients:
        try:
            derived[oxx] = costmodel.derive(oxx)
        except Exception as exc:
            derived[oxx] = {"patient": oxx, "error": str(exc)}
    payload = {
        "schema": "hawking.odyssey.ctl.economics.v1",
        "detachment_metrics": costmodel.detachment_metrics(),
        "derive": derived,
        "_evidence": "DERIVED (odyssey_costmodel)",
    }
    json.dump(payload, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


def cmd_admit(slug: str, est_gib: float) -> int:
    """Memgate (preferred) + worker_gate. Abort on REFUSE."""
    mg = memgate.admit(est_gib, in_flight_gib=0.0, clean_room=False)
    try:
        obs = worker_gate.observe()
        g = worker_gate.gate(obs)
    except Exception as exc:
        g = {"decision": "REFUSE", "note": f"worker_gate failed: {exc}"}
    snap = machine_snapshot()
    disk = snap.get("disk_free_gib")
    decision = "GO" if mg.get("decision") == "GO" else "REFUSE"
    notes = [mg.get("note") or "", g.get("note") or ""]
    if g.get("decision") == "REFUSE" and decision == "GO":
        override = _headroom_override_note()
        if not override:
            decision = "REFUSE"
            notes.append(g.get("note") or "worker_gate REFUSE")
        else:
            notes.append(f"worker_gate REFUSE overridden ({override})")
    if disk is not None and disk < DISK_FLOOR_GIB:
        decision = "REFUSE"
        notes.append(f"disk {disk} GiB below floor {DISK_FLOOR_GIB}")
        reclaim_if_tight(snap)
    elif disk is not None and (disk - est_gib) < DISK_FLOOR_GIB:
        decision = "REFUSE"
        notes.append(
            f"est {est_gib} GiB would leave {disk - est_gib:.1f} GiB (< {DISK_FLOOR_GIB} floor)"
        )
    note = "; ".join(n for n in notes if n)
    print(f"{decision}  slug={slug} est_gib={est_gib}  {note}")
    print(
        f"  memgate={mg.get('decision')} projected_swap={mg.get('projected_swap_gib')}  "
        f"gate={g.get('decision')} wired={g.get('current_wired_gb')} "
        f"headroom={g.get('projected_headroom_gb')}  _evidence=MEASURED (memgate+worker_gate)"
    )
    return 0 if decision == "GO" else 1


def _self_check() -> int:
    # 1. state round-trip
    st = ensure_state()
    save_state(st)
    st2 = read_json(STATE)
    assert st2.get("schema") == SCHEMA, st2.get("schema")
    assert len(st2.get("patients") or []) >= 14, len(st2.get("patients") or [])
    oxxs = [p["oxx"] for p in st2["patients"]]
    assert oxxs == [f"O{i:03d}" for i in range(14)], oxxs
    by = {p["oxx"]: p for p in st2["patients"]}
    # Gated (HF-auth) patients are not on disk: BLOCKED or mid-acquisition.
    assert not by["O000"]["on_disk"] and by["O000"]["state"] in {"BLOCKED", "ACQUIRING"}
    assert not by["O002"]["on_disk"] and by["O002"]["state"] in {"BLOCKED", "ACQUIRING"}
    # On-disk patients advance READY -> RUNNING -> RETIRED; patients whose weights
    # were deleted for storage (or bug-reclaimed) are off-disk and re-acquirable
    # (BLOCKED/ACQUIRING). Both are valid — assert the state matches on_disk.
    for _oxx in ("O001", "O004", "O005"):
        if by[_oxx]["on_disk"]:
            assert by[_oxx]["state"] in {"READY", "RUNNING", "RETIRED"}, (_oxx, by[_oxx]["state"])
        else:
            assert by[_oxx]["state"] in {"BLOCKED", "ACQUIRING"}, (_oxx, by[_oxx]["state"])
    if not by["O003"]["on_disk"]:
        assert str(by["O003"]["ledger"]).lower().startswith("queued")
    save_state(st2)
    assert read_json(STATE) == st2, "state did not round-trip"

    # 2. value ranking: cheap architecture-first > expensive acquisition
    cheap = {"info": 8, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0}
    dear = {"info": 3, "wall_cost": 8, "gpu_cost": 0, "opus_cost": 0}
    assert value(cheap) > value(dear), (value(cheap), value(dear))
    assert value({"info": 1, "wall_cost": 0, "gpu_cost": 0, "opus_cost": 0}) == 10.0

    # 3. status renders the seed queue
    from io import StringIO
    buf = StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        rc = cmd_status()
    finally:
        sys.stdout = old
    assert rc == 0
    text = buf.getvalue()
    assert "HAWKING ODYSSEY-I" in text, text[:200]
    assert "O001" in text
    assert "on-disk" in text

    # 4. packet builder validates against schema (refresh seed + stub)
    for oxx in ("O005", "O001"):
        pkt = assemble_packet(oxx, st2)
        errs = validate_packet(pkt)
        assert not errs, (oxx, errs)
        dest = write_packet(oxx, st2)
        again = read_json(dest)
        assert validate_packet(again) == []
        # idempotent: second assemble matches
        assert assemble_packet(oxx, st2) == again
    stub = assemble_packet("O003", st2)
    assert validate_packet(stub) == []
    assert stub["architecture"]["kind"] == "moe"

    schema = read_json(SCHEMA_PATH)
    for sec in schema.get("fields") or {}:
        assert sec in assemble_packet("O005", st2), sec

    # 5. harvest tolerates a malformed task dir
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tasks, recs, esc = td / "tasks", td / "recs", td / "esc.jsonl"
        mal = tasks / "odyssey-malformed-xyz"
        mal.mkdir(parents=True)
        (mal / "status").write_text("done\n")
        (mal / "grok-report.md").write_text("just chatting, no result\n")
        empty = tasks / "odyssey-empty-xyz"
        empty.mkdir()
        (empty / "status").write_text("done\n")
        running = tasks / "odyssey-running-xyz"
        running.mkdir()
        (running / "status").write_text("running\n")
        ignore = tasks / "not-odyssey-xyz"
        ignore.mkdir()
        (ignore / "grok-report.md").write_text("**Completion report**\n\nok\n")
        good = tasks / "odyssey-o005-fixture-xyz"
        good.mkdir()
        (good / "status").write_text("done\n")
        (good / "grok-report.md").write_text(
            "**Completion report**\n\n"
            "```json\n"
            + json.dumps({"oxx": "O005", "finding": "fixture", "evidence": "HYPOTHESIS"})
            + "\n```\n"
        )
        rows = harvest(tasks_root=tasks, receipt_dir=recs, escalate_path=esc, state=dict(st2))
        by_task = {r["task"]: r for r in rows}
        assert "odyssey-running-xyz" not in by_task
        assert by_task["odyssey-malformed-xyz"]["verdict"] == "REJECTED"
        assert by_task["odyssey-empty-xyz"]["verdict"] == "REJECTED"
        assert by_task["odyssey-o005-fixture-xyz"]["verdict"] == "ACCEPTED"
        assert by_task["odyssey-o005-fixture-xyz"]["oxx"] == "O005"
        assert (recs / "odyssey-malformed-xyz.json").is_file()

    # 6. governors: worker_gate refuses injected pressure; doctor_seal refuses empty
    obs = {
        "total_gb": 100.0, "wired_gb": 4.63, "free_gb": 50.0, "inactive_gb": 10.0,
        "compressed_gb": 0.05, "swap_used_mb": 512.0, "workers_resident": 0,
        "worker_rss_total_gb": 0.0,
    }
    g = worker_gate.gate(obs)
    assert g["decision"] == "REFUSE", g
    v, _ = doctor_seal.seal({})
    assert v == "REFUSED", v

    # 7. frontier files exist, parse, non-empty, labelled
    for path, key in ((RULEBASE, "rules"), (TRANSFER, "rows"), (NEGATIVE, "entries")):
        assert path.is_file(), path
        doc = read_json(path)
        assert evidence_class(doc.get("_evidence")), path
        assert doc.get(key), path
    cells0 = (read_json(TRANSFER)["rows"][0].get("cells") or {})
    for i in range(14):
        assert f"O{i:03d}" in cells0, f"missing O{i:03d}"

    # 8. run-loop: select, render an SG-valid contract, honor max-lanes=0 and
    #    injected worker_gate REFUSE (no launch). An empty frontier is a VALID
    #    state (all on-disk patients' science mapped / ladders exhausted) — assert
    #    only the invariants that must hold for whatever IS selected.
    selected = select_ready_obligations(st2)
    assert all(s["template"] in TEMPLATES for s in selected), selected
    assert all(patient_on_disk(patient_meta(s["oxx"], st2)) for s in selected)

    samples = [
        {"id": "T-MOE", "oxx": "O003", "template": "external-science-moe",
         "title": "t", "info": 1, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0},
        {"id": "T-DENSE", "oxx": "O001", "template": "external-science-dense",
         "title": "t", "info": 1, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0},
        {"id": "T-DENSE-O004", "oxx": "O004", "template": "external-science-dense",
         "title": "t", "info": 1, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0},
        {"id": "T-SENS", "oxx": "O005", "template": "sensitivity-map",
         "title": "t", "info": 1, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0},
        {"id": "T-XFER", "oxx": "O006", "template": "transfer-control",
         "title": "t", "info": 1, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0,
         "reference": "O005"},
        {"id": "T-GRAV", "oxx": "O005", "template": "gravity-moe",
         "title": "t", "info": 1, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0},
        {"id": "T-AGGR", "oxx": "O006", "template": "gravity-aggressive-moe",
         "title": "t", "info": 1, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0},
        {"id": "T-AGGR-D", "oxx": "O004", "template": "gravity-aggressive-dense",
         "title": "t", "info": 1, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0},
        {"id": "T-AGGR-H", "oxx": "O001", "template": "gravity-aggressive-hybrid",
         "title": "t", "info": 1, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0},
        {"id": "T-NX", "oxx": "O001", "template": "nx-state-hybrid",
         "title": "t", "info": 1, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0},
    ]
    for ob in samples:
        dest = render_contract(ob)
        assert dest.is_file(), dest
        ok, msg = sg_lint(dest)
        assert ok, (ob["template"], dest, msg)

    launches: list = []

    def no_launch(*_a, **_k):
        launches.append((_a, _k))
        return (0, "should-not-launch", "")

    permit_obs = {
        "total_gb": 100.0, "wired_gb": 4.63, "free_gb": 50.0, "inactive_gb": 10.0,
        "compressed_gb": 0.05, "swap_used_mb": 0.0, "workers_resident": 0,
        "worker_rss_total_gb": 0.0,
    }
    refuse_obs = dict(permit_obs, swap_used_mb=512.0)
    fat_snap = {
        "disk_free_gib": 80.0, "clean_box_ok": True, "clean_box_reason": "injected ok",
    }

    iso_state = dict(st2)
    iso_state["work"] = [
        dict(w) for w in (st2.get("work") or [])
        if w.get("status") not in {"RUNNING", "REVIEW"}
    ]
    mod = sys.modules[__name__]
    _orig_live = mod.live_odyssey_lanes
    mod.live_odyssey_lanes = lambda: []
    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            logp = td / "RUN_LOG.jsonl"
            rows = run_loop(
                go=False, max_lanes=2, state=dict(iso_state),
                observe_fn=lambda: permit_obs, gate_fn=worker_gate.gate,
                snapshot_fn=lambda: dict(fat_snap),
                launch_fn=no_launch, persist=False, log_path=logp,
                reclaim_fn=lambda: None,
            )
            assert launches == [], launches
            assert rows, "dry-run plan empty"
            assert all(r["verdict"] == "DRY-RUN" for r in rows), rows
            for r in rows:
                cpath = Path(r["contract"])
                if not cpath.is_file():
                    cpath = REPO / r["contract"]
                assert cpath.is_file(), r["contract"]
                ok, msg = sg_lint(cpath)
                assert ok, (r["contract"], msg)

            rows0 = run_loop(
                go=True, max_lanes=0, state=dict(iso_state),
                observe_fn=lambda: permit_obs, gate_fn=worker_gate.gate,
                snapshot_fn=lambda: dict(fat_snap),
                launch_fn=no_launch, persist=False, log_path=logp,
                reclaim_fn=lambda: None,
            )
            assert launches == [], "max-lanes 0 launched"
            assert all(r["verdict"] != "LAUNCH" for r in rows0)

            rows_r = run_loop(
                go=True, max_lanes=2, state=dict(iso_state),
                observe_fn=lambda: refuse_obs, gate_fn=worker_gate.gate,
                snapshot_fn=lambda: dict(fat_snap),
                launch_fn=no_launch, persist=False, log_path=logp,
                reclaim_fn=lambda: None,
            )
            # Under a model-memory REFUSE the safety invariant is that NO model
            # lane launches — memgate/worker gate is the boundary. Grok/novelty
            # lanes load no model, so they are orthogonal to the memory gate and
            # may still launch; anything that launched here must be grok-kind.
            # (Which model lanes are in the live frontier drifts as patients
            # retire, so the frontier-independent proof is the direct
            # launch_deterministic check below, not this one.)
            model_rows = [
                r for r in rows_r
                if _template_loads_model(r.get("template") or "", r)
            ]
            launched_rows = [r for r in rows_r if r["verdict"] == "LAUNCH"]
            assert rows_r, "injected REFUSE produced no decisions"
            assert all(r["verdict"] == "SKIP" for r in model_rows), model_rows
            assert all(r["kind"] == "grok" for r in launched_rows), launched_rows
            assert logp.is_file() and logp.read_text().strip()
    finally:
        mod.live_odyssey_lanes = _orig_live

    # Frontier-independent safety proof: memgate REFUSE (huge in_flight ->
    # projected swap over SWAP_MAX) makes launch_deterministic SKIP without ever
    # spawning. This is the boundary that keeps concurrent model lanes off swap.
    spawned: list = []
    refuse_row = launch_deterministic(
        {"id": "REF", "oxx": "O001", "template": "gravity-aggressive-hybrid",
         "title": "t", "gravity_spec": "q2-g32-attn-mlp",
         "model_loading": True, "timing": False, "download": False},
        now_epoch=time.time(), in_flight_gib=10_000.0,
        state={"work": []}, persist=False, spawn=True,
        spawn_fn=lambda *a, **k: (spawned.append(a) or 424242),
    )
    assert refuse_row["verdict"] == "SKIP", refuse_row
    assert refuse_row.get("pid") is None, refuse_row
    assert spawned == [], "memgate REFUSE still spawned a model subprocess"

    argv = grok_argv(
        "odyssey-o001-external-science-dense",
        AUTO_DIR / "o001_external-science-dense.md",
        model_loading=True,
    )
    assert argv[:2] == [str(GROK_BIN), "delegate"]
    assert "--profile" in argv and argv[argv.index("--profile") + 1] == "gate"
    assert "--background" in argv
    assert "SG_OFF" not in argv

    # 9. harden harvest: DATA-ONLY fake lane harvests + would-cleanup;
    #    data-producing + tools/*.py tweak + valid receipt is still DATA-ONLY
    #    (drop the code, auto-complete); non-template or no-receipt → REVIEW.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tasks = td / "tasks"
        trees = td / "worktrees"
        dest = td / "dest"
        recs = td / "recs"
        qpath = td / "REVIEW_QUEUE.jsonl"
        fake_state = {
            "schema": SCHEMA,
            "patients": list(st2.get("patients") or []),
            "work": [
                {"id": "H-DATA", "oxx": "O001", "title": "data lane",
                 "status": "RUNNING",
                 "task": "odyssey-o001-data-fake",
                 "template": "external-science-moe",
                 "info": 1, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0},
                {"id": "H-CODE", "oxx": "O005", "title": "code lane",
                 "status": "RUNNING",
                 "task": "odyssey-o005-code-fake",
                 "template": "sensitivity-map",
                 "info": 1, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0},
                {"id": "H-MAL", "oxx": "O003", "title": "malformed lane",
                 "status": "RUNNING",
                 "task": "odyssey-o003-malformed-fake",
                 "template": "transfer-control",
                 "info": 1, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0},
                {"id": "H-TWEAK", "oxx": "O005", "title": "data+runner tweak",
                 "status": "RUNNING",
                 "task": "odyssey-o005-sensitivity-map-20260819-000001",
                 "template": "sensitivity-map",
                 "info": 1, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0},
                {"id": "H-UNK", "oxx": "O009", "title": "unknown template",
                 "status": "RUNNING",
                 "task": "odyssey-o009-custom-hack",
                 "info": 1, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0},
            ],
            "history": [], "harvested": [], "metrics": {},
        }
        cleaned: list[str] = []

        def fake_cleanup(task_id: str):
            cleaned.append(task_id)
            return True, "ok"

        def _mk_lane(name: str, report: str, file_map: dict[str, str],
                     patch_files: list[str] | None = None) -> Path:
            tdir = tasks / name
            tdir.mkdir(parents=True)
            (tdir / "status").write_text("done\n")
            if report is not None:
                (tdir / "grok-report.md").write_text(report)
            wt = trees / name
            wt.mkdir(parents=True)
            (tdir / "metadata.json").write_text(json.dumps({"workdir": str(wt)}))
            listed = patch_files if patch_files is not None else list(file_map)
            patch_lines = []
            for rel, body in file_map.items():
                p = wt / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(body)
            for rel in listed:
                patch_lines.append(f"diff --git a/{rel} b/{rel}")
                patch_lines.append(f"+++ b/{rel}")
            (tdir / "diff.patch").write_text("\n".join(patch_lines) + "\n")
            return tdir

        data_body = json.dumps({
            "oxx": "O001", "finding": "data-only fixture",
            "evidence": "MEASURED",
        })
        _mk_lane(
            "odyssey-o001-data-fake",
            "**Completion report**\n\n```json\n" + data_body + "\n```\n",
            {
                "receipts/odyssey-i/O001_EXTERNAL.json": data_body + "\n",
                "workspace/campaign/odyssey/patients/O001/ODYSSEY_PATIENT_O001.json":
                    json.dumps({"oxx": "O001", "identity": {"_evidence": "MEASURED"}}) + "\n",
            },
        )
        code_marker = "CODE_FAKE_MARKER_DO_NOT_MERGE\n"
        _mk_lane(
            "odyssey-o005-code-fake",
            "**Completion report**\n\nRESULT: code fixture\n",
            {
                "tools/odyssey_patient_runner.py": code_marker,
                "receipts/odyssey-i/O005_FAKE.json": '{"oxx":"O005"}\n',
            },
        )
        tweak_body = json.dumps({
            "oxx": "O005", "finding": "sensitivity with runner tweak",
            "evidence": "MEASURED", "per_organ_sensitivity": {"embed": 0.1},
        })
        _mk_lane(
            "odyssey-o005-sensitivity-map-20260819-000001",
            "**Completion report**\n\n```json\n" + tweak_body + "\n```\n",
            {
                "tools/odyssey_patient_runner.py": code_marker,
                "receipts/odyssey-i/O005_SENSITIVITY.json": tweak_body + "\n",
                "workspace/campaign/odyssey/patients/O005/ODYSSEY_PATIENT_O005.json":
                    json.dumps({"oxx": "O005", "representation": {
                        "per_organ_sensitivity": {"embed": 0.1},
                    }}) + "\n",
            },
        )
        _mk_lane(
            "odyssey-o009-custom-hack",
            "**Completion report**\n\nRESULT: unknown template fixture\n",
            {"tools/odyssey_patient_runner.py": code_marker},
        )
        mal = tasks / "odyssey-o003-malformed-fake"
        mal.mkdir(parents=True)
        (mal / "status").write_text("done\n")
        (mal / "metadata.json").write_text(json.dumps({
            "workdir": str(trees / "odyssey-o003-malformed-fake"),
        }))
        (trees / "odyssey-o003-malformed-fake").mkdir(parents=True)

        dry_rows = harvest_lanes(
            tasks_root=tasks, receipt_dir=recs, state=dict(fake_state),
            dry_run=True, worktrees_root=trees, dest_root=dest,
            review_queue=qpath, cleanup_fn=fake_cleanup, persist=False,
        )
        dry_by = {r["task"]: r for r in dry_rows}
        assert dry_by["odyssey-o001-data-fake"]["classification"] == "DATA-ONLY"
        assert "MERGE+COMPLETE" in dry_by["odyssey-o001-data-fake"]["action"]
        assert dry_by["odyssey-o005-code-fake"]["classification"] == "CODE"
        assert "REVIEW" in dry_by["odyssey-o005-code-fake"]["action"]
        tweak_name = "odyssey-o005-sensitivity-map-20260819-000001"
        assert dry_by[tweak_name]["classification"] == "DATA-ONLY", dry_by[tweak_name]
        assert "MERGE+COMPLETE" in dry_by[tweak_name]["action"]
        assert dry_by["odyssey-o009-custom-hack"]["classification"] == "CODE"
        assert "REVIEW" in dry_by["odyssey-o009-custom-hack"]["action"]
        assert not dry_by["odyssey-o001-data-fake"]["applied"]
        assert cleaned == []
        assert not (dest / "receipts" / "odyssey-i" / "O001_EXTERNAL.json").exists()
        assert not qpath.exists()

        rows_h = harvest_lanes(
            tasks_root=tasks, receipt_dir=recs, state=fake_state,
            dry_run=False, worktrees_root=trees, dest_root=dest,
            review_queue=qpath, cleanup_fn=fake_cleanup, persist=False,
        )
        by_h = {r["task"]: r for r in rows_h}
        data_row = by_h["odyssey-o001-data-fake"]
        code_row = by_h["odyssey-o005-code-fake"]
        mal_row = by_h["odyssey-o003-malformed-fake"]
        tweak_row = by_h[tweak_name]
        unk_row = by_h["odyssey-o009-custom-hack"]
        assert data_row["classification"] == "DATA-ONLY"
        assert data_row["applied"] and data_row["verdict"] == "VERIFIED"
        assert data_row["cleanup"] is True
        assert "odyssey-o001-data-fake" in cleaned
        dest_receipt = dest / "receipts" / "odyssey-i" / "O001_EXTERNAL.json"
        assert dest_receipt.is_file(), dest_receipt
        assert "data-only fixture" in dest_receipt.read_text()
        dest_pkt = dest / "workspace" / "campaign" / "odyssey" / "patients" / "O001" / "ODYSSEY_PATIENT_O001.json"
        assert dest_pkt.is_file()
        assert (recs / "harvest_odyssey-o001-data-fake.json").is_file()
        assert _work_for_task(fake_state, "odyssey-o001-data-fake")["status"] == "VERIFIED"

        assert code_row["classification"] == "CODE"
        assert code_row["applied"] and code_row["verdict"] == "REVIEW"
        assert "odyssey-o005-code-fake" not in cleaned
        dest_code = dest / "tools" / "odyssey_patient_runner.py"
        assert not dest_code.exists(), "CODE lane must not merge tools/*.py"
        assert qpath.is_file()
        qrows = [json.loads(x) for x in qpath.read_text().splitlines() if x.strip()]
        assert any(q["task"] == "odyssey-o005-code-fake" for q in qrows), qrows
        assert _work_for_task(fake_state, "odyssey-o005-code-fake")["status"] == "REVIEW"
        assert (recs / "harvest_odyssey-o005-code-fake.json").is_file()

        assert tweak_row["classification"] == "DATA-ONLY"
        assert tweak_row["applied"] and tweak_row["verdict"] == "VERIFIED"
        assert tweak_name in cleaned
        assert RUNNER_REL in (tweak_row.get("dropped") or []), tweak_row.get("dropped")
        dest_sens = dest / "receipts" / "odyssey-i" / "O005_SENSITIVITY.json"
        assert dest_sens.is_file(), dest_sens
        assert not dest_code.exists(), "DATA-ONLY must drop tools/*.py even with a runner tweak"
        dest_comp = dest / "workspace" / "campaign" / "odyssey" / "ODYSSEY_COMPLETIONS.json"
        assert dest_comp.is_file(), dest_comp
        ckeys = {
            (e["patient_id"], e["mechanism_id"])
            for e in (read_json(dest_comp).get("entries") or [])
            if e.get("status") == "VERIFIED"
        }
        assert ("O005", "sensitivity-map") in ckeys, ckeys
        assert ("O001", "external-science") in ckeys, ckeys
        assert _work_for_task(fake_state, tweak_name)["status"] == "VERIFIED"

        assert unk_row["classification"] == "CODE"
        assert unk_row["applied"] and unk_row["verdict"] == "REVIEW"
        assert "odyssey-o009-custom-hack" not in cleaned
        qrows = [json.loads(x) for x in qpath.read_text().splitlines() if x.strip()]
        assert any(q["task"] == "odyssey-o009-custom-hack" for q in qrows), qrows

        assert mal_row["verdict"] == "REFUTED"
        assert "malformed" in (mal_row.get("reason") or "")
        assert "odyssey-o003-malformed-fake" not in cleaned
        assert _work_for_task(fake_state, "odyssey-o003-malformed-fake")["status"] == "REFUTED"

        # idempotent re-run: no second cleanup, no second queue line
        n_q = len(qpath.read_text().splitlines())
        cleaned.clear()
        rows_h2 = harvest_lanes(
            tasks_root=tasks, receipt_dir=recs, state=fake_state,
            dry_run=False, worktrees_root=trees, dest_root=dest,
            review_queue=qpath, cleanup_fn=fake_cleanup, persist=False,
        )
        assert cleaned == []
        assert len(qpath.read_text().splitlines()) == n_q
        assert all(r.get("verdict") == "SKIP" for r in rows_h2), rows_h2

    # 10. data-producing templates are RUN-only: code_edit_busy does not serialize
    g_serial = evaluate_gates(
        {"template": "sensitivity-map", "model_loading": False,
         "timing": False, "download": False},
        go=True, running_n=1, cap=2, snap=fat_snap, worker=None,
        lint_ok=True, lint_msg="", code_edit_busy=True,
    )
    assert g_serial["verdict"] == "LAUNCH", g_serial
    g_runonly = evaluate_gates(
        {"template": "external-science-moe", "model_loading": False,
         "timing": False, "download": False},
        go=True, running_n=1, cap=2, snap=fat_snap, worker=None,
        lint_ok=True, lint_msg="", code_edit_busy=True,
    )
    assert g_runonly["verdict"] == "LAUNCH", g_runonly

    # 11. completion index rebuild + A-F idempotence battery + write-scope
    try:
        complete(
            obligation_id="x", patient_id="O001", mechanism_id="x",
            status="VERIFIED", completed_at="", persist=False,
        )
        raise AssertionError("complete() must require completed_at")
    except ValueError:
        pass

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        idx_path = td / "ODYSSEY_COMPLETIONS.json"
        recs = td / "recs"
        recs.mkdir()
        # rebuild from the real receipt dir into a temp index
        rec_src = RECEIPT_DIR
        alt_recs = HAWKING_REPO / "receipts" / "odyssey-i"
        if not (RECEIPT_DIR / "O001_EXTERNAL.json").is_file() and (
            alt_recs / "O001_EXTERNAL.json"
        ).is_file():
            rec_src = alt_recs
        rebuilt = rebuild_completions(
            path=idx_path, receipt_dir=rec_src, persist=True,
        )
        keys = {
            (e["patient_id"], e["mechanism_id"])
            for e in rebuilt.get("entries") or []
            if e.get("status") == "VERIFIED"
        }
        assert ("O001", "external-science") in keys, keys
        assert ("O001", "sensitivity-map") in keys, keys
        assert ("O003", "external-science") in keys, keys
        assert ("O005", "external-science") in keys, keys
        assert ("O005", "route-map") in keys, keys
        assert ("O005", "sensitivity-map") not in keys, keys
        for e in rebuilt.get("entries") or []:
            assert e.get("receipt_sha256"), e
            assert e.get("source_revision"), e
            assert e.get("completed_at"), e
            assert e.get("status") == "VERIFIED"
        rebuilt2 = rebuild_completions(
            path=idx_path, receipt_dir=rec_src, persist=True,
        )
        sig = lambda d: {
            (e["patient_id"], e["mechanism_id"], e.get("receipt_sha256"), e.get("status"))
            for e in (d.get("entries") or [])
        }
        assert sig(rebuilt) == sig(rebuilt2), "rebuild is not idempotent"

    # persist the canonical index from real receipts (scheduler source of truth)
    # Sparse worktrees may omit receipts/; never strip the live index then.
    rec_live = RECEIPT_DIR
    alt_recs = HAWKING_REPO / "receipts" / "odyssey-i"
    if not (RECEIPT_DIR / "O001_EXTERNAL.json").is_file() and (
        alt_recs / "O001_EXTERNAL.json"
    ).is_file():
        rec_live = alt_recs
    if rec_live.resolve() == RECEIPT_DIR.resolve():
        live = rebuild_completions(persist=True)
    else:
        live = load_completions()
    live_keys = {
        (e["patient_id"], e["mechanism_id"])
        for e in live.get("entries") or []
        if e.get("status") == "VERIFIED"
    }
    o005_sens_done = ("O005", "sensitivity-map") in live_keys
    o003_sens_done = ("O003", "sensitivity-map") in live_keys
    o006_sens_done = ("O006", "sensitivity-map") in live_keys
    flying_now = {
        (w.get("oxx"), w.get("template"))
        for w in (st2.get("work") or [])
        if w.get("status") in {"RUNNING", "REVIEW"}
        and w.get("oxx") and w.get("template")
    }
    if o005_sens_done:
        assert (
            (RECEIPT_DIR / "O005_SENSITIVITY.json").is_file()
            or (rec_live / "O005_SENSITIVITY.json").is_file()
        )
    ranked_live = select_ready_obligations(st2)
    live_pair = {(r["oxx"], r["template"]) for r in ranked_live}
    assert ("O001", "external-science-dense") not in live_pair, live_pair
    assert ("O001", "sensitivity-map") not in live_pair, live_pair
    assert ("O003", "external-science-moe") not in live_pair, live_pair
    if o005_sens_done or ("O005", "sensitivity-map") in flying_now:
        assert ("O005", "sensitivity-map") not in live_pair, live_pair
    else:
        assert any(
            r["oxx"] == "O005" and r["template"] == "sensitivity-map" for r in ranked_live
        ), live_pair
    o004_ext_done = science_is_done(
        "O004", mechanism_for_template("external-science-dense"),
        live.get("entries") or [],
    )
    if o004_ext_done or ("O004", "external-science-dense") in flying_now:
        # sealed (replay-proof) or in flight -> must not be re-selected
        assert ("O004", "external-science-dense") not in live_pair, live_pair
    else:
        assert any(
            r["oxx"] == "O004" and r["template"] == "external-science-dense" for r in ranked_live
        ), live_pair
    # O006 transfer-control is now SEALED (completions) -> must be refused (replay-proof).
    assert ("O006", "transfer-control") not in live_pair, live_pair
    # a genuinely-pending O006 obligation is still selectable — UNLESS O006's
    # science is fully mapped (ladder + required all sealed), a valid end state.
    o006_exhausted = not any(r["oxx"] == "O006" for r in ranked_live)
    if o006_sens_done or ("O006", "sensitivity-map") in flying_now:
        assert ("O006", "sensitivity-map") not in live_pair, live_pair
    elif not o006_exhausted:
        assert any(
            r["oxx"] == "O006" and r["template"] == "sensitivity-map" for r in ranked_live
        ), live_pair

    # A-F: synthetic completions + queue. Watch fail AND pass.
    af_entries = [
        {"obligation_id": "A", "patient_id": "PA", "mechanism_id": "mech",
         "status": "VERIFIED", "reopen_if": None, "completed_at": "t0"},
        {"obligation_id": "B", "patient_id": "PB", "mechanism_id": "mech",
         "status": "REFUTED", "reopen_if": None, "completed_at": "t0"},
        {"obligation_id": "C", "patient_id": "PC", "mechanism_id": "mech",
         "status": "SUPERSEDED", "reopen_if": None, "completed_at": "t0"},
        # D pending: no entry
        {"obligation_id": "E", "patient_id": "PE", "mechanism_id": "mech",
         "status": "VERIFIED", "reopen_if": "true", "completed_at": "t0"},
        # F: same patient as A, different mechanism — no entry
    ]
    af_queue = [
        ("A", "PA", "mech"),
        ("B", "PB", "mech"),
        ("C", "PC", "mech"),
        ("D", "PD", "mech"),
        ("E", "PE", "mech"),
        ("F", "PA", "other"),
    ]
    af_got = {
        label: selection_verdict(pid, mech, af_entries)
        for label, pid, mech in af_queue
    }
    assert af_got["A"] == "REFUSE", af_got
    assert af_got["B"] == "REFUSE", af_got
    assert af_got["C"] == "REFUSE", af_got
    assert af_got["D"] == "LAUNCH", af_got
    assert af_got["E"] == "LAUNCH", af_got
    assert af_got["F"] == "LAUNCH", af_got
    # reopen_if source_revision != <other> is mechanically TRUE; == HEAD is FALSE
    head = git_head()
    assert selection_verdict(
        "PE2", "mech",
        [{"obligation_id": "E2", "patient_id": "PE2", "mechanism_id": "mech",
          "status": "VERIFIED", "reopen_if": "source_revision != deadbeef",
          "completed_at": "t0"}],
        source_revision=head,
    ) == "LAUNCH"
    assert selection_verdict(
        "PA2", "mech",
        [{"obligation_id": "A2", "patient_id": "PA2", "mechanism_id": "mech",
          "status": "VERIFIED", "reopen_if": f"source_revision != {head}",
          "completed_at": "t0"}],
        source_revision=head,
    ) == "REFUSE"

    # write-scope: data-producing templates no longer claim the runner.
    # Different patients run in parallel; same-patient packet still serializes.
    sens = {"oxx": "O005", "template": "sensitivity-map"}
    dense = {"oxx": "O001", "template": "external-science-dense"}
    moe = {"oxx": "O003", "template": "external-science-moe"}
    xfer = {"oxx": "O006", "template": "transfer-control"}
    grav = {"oxx": "O005", "template": "gravity-moe"}
    nxh = {"oxx": "O001", "template": "nx-state-hybrid"}
    assert RUNNER_REL not in write_scope(sens)["write_set"], write_scope(sens)
    assert RUNNER_REL not in write_scope(grav)["write_set"], write_scope(grav)
    assert RUNNER_REL not in write_scope(dense)["write_set"], write_scope(dense)
    assert not scopes_conflict(write_scope(sens), write_scope(dense))
    assert not scopes_conflict(write_scope(sens), write_scope(moe))
    assert not scopes_conflict(write_scope(sens), write_scope(xfer)), write_scope(xfer)
    assert not scopes_conflict(write_scope(dense), write_scope(xfer))
    assert not scopes_conflict(write_scope(grav), write_scope(nxh))
    assert scopes_conflict(write_scope(sens), write_scope(grav)), "same-patient packet"
    admitted, occupied = [], []
    for ob in (sens, dense, moe, xfer):
        sc = write_scope(ob)
        if scope_conflict_reason(sc, occupied):
            continue
        admitted.append((ob["oxx"], ob["template"]))
        occupied.append(sc)
    assert ("O005", "sensitivity-map") in admitted
    assert ("O001", "external-science-dense") in admitted
    assert ("O003", "external-science-moe") in admitted
    assert ("O006", "transfer-control") in admitted
    g_collide = evaluate_gates(
        {"template": "sensitivity-map", "model_loading": False,
         "timing": False, "download": False},
        go=True, running_n=1, cap=2, snap=fat_snap, worker=None,
        lint_ok=True, lint_msg="", scope_conflict=RUNNER_REL,
    )
    assert g_collide["verdict"] == "SKIP", g_collide
    assert "write-scope collision" in (g_collide.get("skip_reason") or ""), g_collide

    # 12. retire-eligible (all required VERIFIED vs one missing) + cycle dry-run
    #     + retire/acquire-next refuse when preconditions fail.
    o001_req = required_mechanisms("O001", st2)
    assert "external-science" in o001_req and "ssm-accounting" in o001_req
    assert "gravity-hybrid" in o001_req and "nx-state-hybrid" in o001_req
    assert "gravity-aggressive-hybrid" in o001_req
    o005_req = required_mechanisms("O005", st2)
    assert "gravity-moe" in o005_req and "nx-gather-moe" in o005_req
    assert "gravity-aggressive-moe" in o005_req
    assert "transfer-control" not in o005_req
    o006_req = required_mechanisms("O006", st2)
    assert "transfer-control" in o006_req, o006_req
    assert "gravity-aggressive-moe" in o006_req
    o004_req = required_mechanisms("O004", st2)
    assert "gravity-dense" in o004_req and "nx-dense" in o004_req
    assert "gravity-aggressive-dense" in o004_req
    # eligible now requires the full descent ladder too (anti-complacency gate),
    # so the "all done" set is required mechanisms PLUS every ladder rung.
    o001_ladder = [
        f"gravity-{s}"
        for s in GRAVITY_LADDER.get(patient_arch_kind("O001", st2), [])
    ]
    full_o001 = [
        {"obligation_id": f"t:{m}", "patient_id": "O001", "mechanism_id": m,
         "status": "VERIFIED", "reopen_if": None, "completed_at": "t0"}
        for m in (o001_req + o001_ladder)
    ]
    if by["O001"].get("state") == "RETIRED":
        # already retired -> not re-eligible (guard holds regardless of a full set)
        assert not retire_eligible("O001", st2, full_o001)
    else:
        assert retire_eligible("O001", st2, full_o001), o001_req
        # dropping any rung (last ladder rung here) -> ladder not exhausted -> refuse
        assert not retire_eligible("O001", st2, full_o001[:-1]), full_o001[:-1]
    o005_miss = missing_required("O005", st2)
    if by["O005"].get("state") == "RETIRED" or science_is_done("O005", "patient-sealed"):
        assert not retire_eligible("O005", st2)
        skipped = retire_patient("O005", dry_run=True, persist=False, state=dict(st2))
        assert skipped.get("verdict") == "SKIP", skipped
    elif o005_miss:
        assert not retire_eligible("O005", st2), o005_miss
        refused = retire_patient("O005", dry_run=True, persist=False, state=dict(st2))
        assert refused.get("verdict") == "REFUSE", refused
        assert "not retire-eligible" in (refused.get("reason") or ""), refused
    elif ladder_rungs_remain("O005"):
        # required done but descent ladder not exhausted -> gate holds it open
        assert not retire_eligible("O005", st2)
        refused5 = retire_patient("O005", dry_run=True, persist=False, state=dict(st2))
        assert refused5.get("verdict") == "REFUSE", refused5
    else:
        assert retire_eligible("O005", st2)
        would = retire_patient("O005", dry_run=True, persist=False, state=dict(st2))
        assert would.get("verdict") == "DRY-RUN", would
        assert "would retire" in (would.get("reason") or ""), would
    refused = retire_patient("O004", dry_run=True, persist=False, state=dict(st2))
    assert refused.get("verdict") == "REFUSE", refused
    assert "not retire-eligible" in (refused.get("reason") or ""), refused

    acq_hold = acquire_next(
        go=False, persist=False, state=dict(st2),
        snapshot_fn=lambda: {
            "disk_free_gib": 1.0, "clean_box_ok": True, "clean_box_reason": "ok",
        },
        hf_info_fn=lambda _repo: (True, {"used_storage": 3 * 1024**3}, "ok"),
    )
    assert acq_hold.get("verdict") == "REFUSE", acq_hold
    assert "disk-hold" in (acq_hold.get("reason") or ""), acq_hold

    frozen = {
        "schema": SCHEMA,
        "patients": [
            dict(p, on_disk=True, state="RETIRED", reclaimable=False)
            for p in (st2.get("patients") or [])
        ],
        "work": [], "history": [], "harvested": [], "metrics": {},
    }
    acq_none = acquire_next(
        go=False, persist=False, state=frozen,
        snapshot_fn=lambda: dict(fat_snap),
        hf_info_fn=lambda _repo: (True, {}, "ok"),
    )
    assert acq_none.get("verdict") == "REFUSE", acq_none
    assert "no eligible" in (acq_none.get("reason") or ""), acq_none

    launches.clear()
    with tempfile.TemporaryDirectory() as td_c:
        td_c = Path(td_c)
        cplan = cycle_tick(
            go=False, max_lanes=2, state=dict(st2), persist=False,
            observe_fn=lambda: permit_obs, gate_fn=worker_gate.gate,
            snapshot_fn=lambda: dict(fat_snap),
            launch_fn=no_launch, reclaim_fn=lambda *a, **k: None,
            log_path=td_c / "RUN_LOG.jsonl",
            auto_dir=td_c / "auto",
        )
    assert launches == [], launches
    assert "O005" not in (cplan.get("retire_eligible") or []), cplan.get("retire_eligible")
    ready_pair = {(r["oxx"], r["template"]) for r in (cplan.get("ready") or [])}

    def _od(o):  # on-disk with usable weights
        return patient_on_disk(patient_meta(o, st2))

    # SEALED science is never re-planned (replay-proof), regardless of roster.
    assert ("O006", "transfer-control") not in ready_pair, ready_pair
    if o005_sens_done or ("O005", "sensitivity-map") in flying_now or not _od("O005"):
        assert ("O005", "sensitivity-map") not in ready_pair, ready_pair
    if o006_sens_done or ("O006", "sensitivity-map") in flying_now:
        assert ("O006", "sensitivity-map") not in ready_pair, ready_pair
    elif _od("O006"):
        assert ("O006", "sensitivity-map") in ready_pair, ready_pair
    if o004_ext_done or ("O004", "external-science-dense") in flying_now or not _od("O004"):
        assert ("O004", "external-science-dense") not in ready_pair, ready_pair
    if o003_sens_done or ("O003", "sensitivity-map") in flying_now:
        assert ("O003", "sensitivity-map") not in ready_pair, ready_pair
    elif _od("O003"):
        assert ("O003", "sensitivity-map") in ready_pair, ready_pair
    # If any on-disk patient still has work, the plan must contain gravity/nx work.
    if ready_pair:
        assert any(t.startswith("gravity-") or t.startswith("nx-") for _, t in ready_pair), ready_pair
    admitted = cplan.get("admitted") or []
    # an empty plan is valid when no on-disk patient has remaining work
    assert all(r.get("verdict") != "LAUNCH" for r in admitted), admitted
    assert all(r.get("task_id") in (None, "") for r in admitted)

    # 13. anti-complacency / conventionality / failure-localization (steer S004)
    # Use an O003-active state copy: O003 may be RETIRED in live state (it can
    # descend its ladder and re-seal), which would short-circuit retire_eligible
    # via the RETIRED guard and mask the gate logic these assertions exercise.
    st2_o003 = json.loads(json.dumps(st2))
    for _p in st2_o003.get("patients") or []:
        if _p.get("oxx") == "O003":
            _p["state"] = "READY"
    o003_req = required_mechanisms("O003", st2_o003)
    assert "gravity-moe" in o003_req and "gravity-aggressive-moe" in o003_req
    conv_only = [
        {"obligation_id": f"t:{m}", "patient_id": "O003", "mechanism_id": m,
         "status": "VERIFIED", "reopen_if": None, "completed_at": "t0",
         "candidate_class": "CONVENTIONAL_ANCHOR" if m == "gravity-moe" else None}
        for m in o003_req if m != "gravity-aggressive-moe"
    ]
    assert not retire_eligible("O003", st2_o003, conv_only), "conventional gravity alone must not retire"
    assert "gravity-aggressive-moe" in missing_required("O003", st2_o003, conv_only)
    # descend the full ladder (terminal) so this isolates the REFUTED-aggressive
    # check from the separate ladder-exhaustion gate.
    o003_ladder_done = [
        {"obligation_id": f"t:gravity-{s}", "patient_id": "O003",
         "mechanism_id": f"gravity-{s}", "status": "VERIFIED",
         "reopen_if": None, "completed_at": "t1"}
        for s in GRAVITY_LADDER.get(patient_arch_kind("O003", st2_o003), [])
    ]
    with_agg = conv_only + o003_ladder_done + [{
        "obligation_id": "t:gravity-aggressive-moe",
        "patient_id": "O003",
        "mechanism_id": "gravity-aggressive-moe",
        "status": "REFUTED",
        "reopen_if": None,
        "completed_at": "t1",
        "candidate_class": "AGGRESSIVE_QUANT",
        "conventionality": "nonconventional",
    }]
    assert retire_eligible("O003", st2_o003, with_agg), "REFUTED aggressive probe still counts as attempted"
    assert aggressive_probe_attempted("O003", with_agg)
    assert conventional_anchor_exists("O003", conv_only)

    q3_rec = {
        "schema": "odyssey.patient.gravity.v1",
        "spec": "q3-g32-experts",
        "quant": "mlx-q3-g32-experts",
    }
    q3_cls = classify_gravity_receipt(q3_rec)
    assert q3_cls["candidate_class"] == "CONVENTIONAL_ANCHOR", q3_cls
    mixed_cls = classify_gravity_receipt({
        "schema": "odyssey.patient.gravity.v1",
        "spec": "mixed-q2q3-experts",
    })
    assert mixed_cls["candidate_class"] == "STRUCTURAL_GRAVITY", mixed_cls
    real_q3 = RECEIPT_DIR / "O005_GRAVITY_q3-g32-experts.json"
    if real_q3.is_file():
        assert classify_gravity_receipt(read_json(real_q3))["candidate_class"] == "CONVENTIONAL_ANCHOR"

    loc = localize_gravity_failure(
        -3,
        {
            "attn": {"round8": {"delta_hits": -4}, "zero": {"delta_hits": -10}},
            "expert": {"round8": {"delta_hits": 0}, "zero": {"delta_hits": -2}},
            "router": {"round8": {"delta_hits": -1}, "zero": {"delta_hits": -8}},
        },
    )
    assert loc, loc
    assert loc.get("most_likely_component") == "attn", loc
    assert loc.get("targeted_repair")
    assert "attn" in str(loc.get("targeted_repair"))
    assert localize_gravity_failure(0, {"attn": {"round8": {"delta_hits": -4}}}) is None

    # 14. G2 integration: memgate multi-model, grammar classify, write_set, manifest.
    low_snap = {
        "free_ram_gib": 48.0, "wired_gib": 8.0, "compressor_gib": 0.25,
        "swap_used_gib": 0.25, "swap_total_gib": 2.0, "cpu_load": 1.2,
    }
    with memgate.using_snapshot(low_snap):
        g0 = memgate.admit(16, 0)
        g1 = memgate.admit(16, 16.0)
        assert g0["decision"] == "GO" and g1["decision"] == "GO", (g0, g1)
        inflight = 0.0
        admitted_m = []
        occupied_m = []
        for ob_m in (
            {"oxx": "O001", "template": "external-science-dense",
             "model_loading": True, "timing": False},
            {"oxx": "O003", "template": "external-science-moe",
             "model_loading": True, "timing": False},
            {"oxx": "O004", "template": "gravity-dense",
             "model_loading": True, "timing": False},
        ):
            sc = write_scope(ob_m)
            if scope_conflict_reason(sc, occupied_m):
                continue
            v = memgate_admit_lane(ob_m, inflight)
            if v.get("decision") != "GO":
                continue
            admitted_m.append(ob_m["oxx"])
            occupied_m.append(sc)
            inflight += float(v.get("est_gib") or obligation_est_gib(ob_m))
        assert len(admitted_m) >= 2, admitted_m
        assert len(set(admitted_m)) == len(admitted_m), admitted_m

    for tmpl, oxx in (
        ("sensitivity-map", "O005"),
        ("gravity-aggressive-moe", "O003"),
        ("nx-gather-moe", "O006"),
        ("external-science-dense", "O001"),
        ("route-map", "O003"),
        ("transfer-control", "O006"),
        ("novelty-representation", "O005"),
    ):
        ws = write_scope({"oxx": oxx, "template": tmpl})
        assert RUNNER_REL not in ws["write_set"], (tmpl, ws)

    gram_q4 = classify_gravity_spec("q4-g128")
    assert gram_q4["candidate_class"] == "CONVENTIONAL_ANCHOR", gram_q4
    gram_q1 = classify_gravity_spec("q1-g32-experts")
    assert gram_q1["candidate_class"] == "AGGRESSIVE_QUANT", gram_q1
    gram_corr = classify_gravity_spec("q2-g32-experts+correction")
    assert gram_corr["candidate_class"] == "STRUCTURAL_GRAVITY", gram_corr
    gram_tier = classify_gravity_spec("tiers-t0t1-experts")
    assert gram_tier["candidate_class"] == "STRUCTURAL_GRAVITY", gram_tier
    gram_mixed = classify_gravity_spec("mixed-q1q3-experts")
    assert gram_mixed["candidate_class"] == "STRUCTURAL_GRAVITY", gram_mixed
    assert classify_gravity_spec("q3-g32-experts")["candidate_class"] == "CONVENTIONAL_ANCHOR"

    man7 = manifest_entry("O007")
    assert man7.get("canonical_source") == "moonshotai/Kimi-Linear-48B-A3B-Instruct", man7
    assert acq_hold.get("oxx")
    hold_man = manifest_entry(acq_hold["oxx"])
    assert acq_hold.get("canonical_source") == hold_man.get("canonical_source"), acq_hold
    assert acq_hold.get("arch_objective") == hold_man.get("arch_objective"), acq_hold
    assert acq_hold.get("search_class") == hold_man.get("search_class"), acq_hold
    assert "info_budget" in acq_hold, acq_hold

    spec_moe = pick_aggressive_spec("O003", "gravity-aggressive-moe")
    assert spec_moe, spec_moe
    assert classify_gravity_spec(spec_moe)["candidate_class"] != "CONVENTIONAL_ANCHOR", spec_moe

    # 15. deterministic direct execution: plan subprocess, reap, timeout, cap
    assert is_deterministic_obligation("gravity-moe")
    assert is_deterministic_obligation({
        "template": "external-science-dense", "oxx": "O001",
    })
    assert not is_deterministic_obligation("novelty-arch")
    assert not is_deterministic_obligation({
        "template": "gravity-moe", "code_building": True,
    })
    det_ob = {
        "id": "T-DET", "oxx": "O001", "template": "external-science-dense",
        "title": "t", "info": 1, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0,
        "model_loading": True, "timing": False, "download": False,
    }
    argv_det = runner_argv(det_ob)
    joined = " ".join(argv_det)
    assert argv_det[0] == PREFERRED_PY, argv_det
    assert RUNNER_REL in argv_det, argv_det
    assert "grok-run" not in joined and "delegate" not in joined
    assert "--oxx" in argv_det and "O001" in argv_det
    assert "--weights" in argv_det
    assert "--out" in argv_det
    assert any("EXTERNAL.json" in str(a) for a in argv_det), argv_det
    grav_argv = runner_argv({
        "oxx": "O003", "template": "gravity-moe",
        "gravity_spec": "q3-g32-experts",
    })
    assert "--gravity" in grav_argv
    assert "q3-g32-experts" in grav_argv
    assert any("GRAVITY" in str(a) for a in grav_argv)

    # gravity-cache eviction: never remove a base 4-bit parent or an in-flight
    # spec; evict cold specs. target huge so free-disk never satisfies -> evicts
    # everything evictable.
    _orig_cache = mod.MLX_ODYSSEY_CACHE
    with tempfile.TemporaryDirectory() as tdc:
        tdc = Path(tdc)
        base = tdc / "O001-Falcon-H1-7B-Instruct-4bit"
        hot = tdc / "O003-gravity-q2-g32-experts"
        cold = tdc / "O001-gravity-q4-g64-attn-mlp"
        for d in (base, hot, cold):
            d.mkdir()
            (d / "w.bin").write_bytes(b"x" * 1024)
        mod.MLX_ODYSSEY_CACHE = tdc
        try:
            evict_state = {"work": [{
                "oxx": "O003", "status": "RUNNING",
                "kind": "subprocess", "pid": os.getpid(),
                "started_epoch": time.time(),
                "argv": ["py", "--oxx", "O003", "--gravity", "q2-g32-experts"],
            }]}
            evict_gravity_caches(
                10 ** 9, state=evict_state, now_epoch=time.time(),
                pid_alive_fn=lambda _p: True,
            )
            assert base.is_dir(), "evicted a base 4-bit parent"
            assert hot.is_dir(), "evicted an in-flight gravity spec"
            assert not cold.is_dir(), "cold gravity spec not evicted"
        finally:
            mod.MLX_ODYSSEY_CACHE = _orig_cache

    # dry-run run_loop plans subprocess, not grok
    with tempfile.TemporaryDirectory() as td_det:
        td_det = Path(td_det)
        plan_rows = run_loop(
            go=False, max_lanes=2, state=dict(iso_state),
            observe_fn=lambda: permit_obs, gate_fn=worker_gate.gate,
            snapshot_fn=lambda: dict(fat_snap),
            launch_fn=no_launch, persist=False, log_path=td_det / "RUN_LOG.jsonl",
            reclaim_fn=lambda: None,
        )
        assert plan_rows, "deterministic dry-run plan empty"
        det_rows = [
            r for r in plan_rows
            if r.get("template") in DETERMINISTIC_TEMPLATES
        ]
        assert det_rows, [r.get("template") for r in plan_rows]
        for r in det_rows:
            assert r.get("kind") == "subprocess", r
            assert r.get("launch") == "subprocess", r
            assert r.get("argv"), r
            argv_s = " ".join(str(a) for a in r["argv"])
            assert RUNNER_REL in argv_s or "odyssey_patient_runner.py" in argv_s
            assert "grok-run" not in argv_s
            assert "--oxx" in r["argv"]
            assert "--out" in r["argv"]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rec = td / "O001_EXTERNAL.json"
        rec.write_text(json.dumps({"oxx": "O001", "schema": "fixture"}) + "\n")
        fake_st = {
            "schema": SCHEMA,
            "work": [{
                "id": "R1", "oxx": "O001",
                "template": "external-science-dense",
                "mechanism": "external-science",
                "kind": "subprocess",
                "pid": 999991, "started_epoch": 1000.0, "timeout_s": 1800,
                "receipt_path": str(rec), "status": "RUNNING", "retries": 0,
            }],
            "patients": [], "history": [], "harvested": [], "metrics": {},
        }
        reap_ok = reap_lanes(
            2000.0, state=fake_st, persist=False,
            pid_alive_fn=lambda _p: False,
        )
        assert fake_st["work"][0]["status"] == "VERIFIED", (fake_st["work"][0], reap_ok)
        assert any(r.get("action") == "verified" for r in reap_ok), reap_ok

        fake_st2 = {
            "schema": SCHEMA,
            "work": [{
                "id": "R2", "oxx": "O004", "template": "nx-dense",
                "mechanism": "nx-dense", "kind": "subprocess",
                "pid": 999992, "started_epoch": 1.0, "timeout_s": 1800,
                "receipt_path": str(td / "missing.json"),
                "status": "RUNNING", "retries": 0,
            }],
            "patients": [], "history": [], "harvested": [], "metrics": {},
        }
        reap_retry = reap_lanes(
            10.0, state=fake_st2, persist=False,
            pid_alive_fn=lambda _p: False,
        )
        assert fake_st2["work"][0]["status"] == "READY", fake_st2["work"][0]
        assert fake_st2["work"][0]["retries"] == 1, fake_st2["work"][0]
        assert any(r.get("action") == "retry" for r in reap_retry), reap_retry
        fake_st2["work"][0]["status"] = "RUNNING"
        fake_st2["work"][0]["pid"] = 999992
        fake_st2["work"][0]["kind"] = "subprocess"
        reap_ref = reap_lanes(
            20.0, state=fake_st2, persist=False,
            pid_alive_fn=lambda _p: False,
        )
        assert fake_st2["work"][0]["status"] == "REFUTED", fake_st2["work"][0]
        assert any(r.get("action") == "refute" for r in reap_ref), reap_ref

        killed: list = []
        fake_st3 = {
            "schema": SCHEMA,
            "work": [{
                "id": "R3", "oxx": "O003", "template": "gravity-moe",
                "mechanism": "gravity-moe", "kind": "subprocess",
                "pid": 999993, "started_epoch": 0.0, "timeout_s": 10,
                "receipt_path": str(td / "nope.json"),
                "status": "RUNNING", "retries": 0,
            }],
            "patients": [], "history": [], "harvested": [], "metrics": {},
        }
        reap_to = reap_lanes(
            100.0, state=fake_st3, persist=False,
            pid_alive_fn=lambda _p: True,
            kill_fn=lambda p: killed.append(p),
        )
        assert killed == [999993], killed
        assert fake_st3["work"][0]["status"] == "FAILED", fake_st3["work"][0]
        assert any(r.get("action") == "timeout" for r in reap_to), reap_to

        live_pids = {999001, 999002}
        cap_st = {
            "schema": SCHEMA,
            "work": [
                {
                    "status": "RUNNING", "kind": "subprocess", "pid": 999001,
                    "started_epoch": 50, "timeout_s": 100, "id": "a",
                    "template": "nx-dense", "oxx": "O004", "task": "pid:999001",
                },
                {
                    "status": "RUNNING", "kind": "subprocess", "pid": 999002,
                    "started_epoch": 50, "timeout_s": 100, "id": "b",
                    "template": "gravity-dense", "oxx": "O001", "task": "pid:999002",
                },
                {
                    "status": "RUNNING", "kind": "subprocess", "pid": 999003,
                    "started_epoch": 50, "timeout_s": 100, "id": "c",
                    "template": "nx-gather-moe", "oxx": "O003", "task": "pid:999003",
                },
                {
                    "status": "RUNNING", "kind": "grok",
                    "task": "odyssey-o005-novelty-arch", "id": "d",
                    "template": "novelty-arch", "oxx": "O005",
                    "started_epoch": 50, "timeout_s": 100,
                },
                {
                    "status": "RUNNING", "kind": "subprocess", "pid": 999004,
                    "started_epoch": 0, "timeout_s": 10, "id": "e",
                    "template": "sensitivity-map", "oxx": "O006",
                    "task": "pid:999004",
                },
            ],
        }
        _orig_live2 = mod.live_odyssey_lanes
        mod.live_odyssey_lanes = lambda: []
        try:
            ids_cap = odyssey_running_ids(
                cap_st, now_epoch=60.0,
                pid_alive_fn=lambda p: p in live_pids,
            )
        finally:
            mod.live_odyssey_lanes = _orig_live2
        # live PIDs 999001/999002 + live grok novelty; dead 999003 and
        # over-timeout 999004 (even if we marked it live) excluded.
        assert "pid:999001" in ids_cap, ids_cap
        assert "pid:999002" in ids_cap, ids_cap
        assert "pid:999003" not in ids_cap, ids_cap
        assert "pid:999004" not in ids_cap, ids_cap
        assert "odyssey-o005-novelty-arch" in ids_cap, ids_cap
        assert len(ids_cap) == 3, ids_cap

    print("self-check ok")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="odyssey_ctl")
    ap.add_argument("--self-check", action="store_true",
                    help="assert state/status/packet/harvest; no network, no model")
    sp = ap.add_subparsers(dest="cmd")
    sp.add_parser("status")
    sp.add_parser("queue")
    sp.add_parser("value")
    p_hv = sp.add_parser("harvest")
    p_hv.add_argument("--dry-run", action="store_true",
                      help="classify finished lanes and print planned action; change nothing")
    p_pkt = sp.add_parser("packet")
    p_pkt.add_argument("oxx")
    p_ad = sp.add_parser("admit")
    p_ad.add_argument("slug")
    p_ad.add_argument("est_gib", type=float)
    p_run = sp.add_parser("run")
    p_run.add_argument("--dry-run", action="store_true",
                       help="plan and render only (default when --go is absent)")
    p_run.add_argument("--go", action="store_true",
                       help="actually launch grok-run lanes; required to spawn")
    p_run.add_argument("--max-lanes", type=int, default=DEFAULT_MAX_LANES,
                       help="MODEL subprocess concurrency ceiling (hard cap 8; memgate is the real limiter)")
    p_run.add_argument("--grok-lanes", type=int, default=0,
                       help="grok/novelty ceiling (default 0: autonomous loop spends no grok usage)")
    p_comp = sp.add_parser("completions")
    p_comp.add_argument("--rebuild", action="store_true",
                        help="idempotent VERIFIED backfill from receipts/odyssey-i")
    p_comp.add_argument("--completed-at", default=None,
                        help="ISO timestamp passed into complete(); default: receipt git/mtime")
    p_cy = sp.add_parser("cycle")
    p_cy.add_argument("--dry-run", action="store_true",
                      help="plan only (default when --go is absent)")
    p_cy.add_argument("--go", action="store_true",
                      help="harvest/retire/acquire/launch for real")
    p_cy.add_argument("--max-lanes", type=int, default=DEFAULT_MAX_LANES,
                      help="MODEL subprocess concurrency ceiling (hard cap 8; memgate is the real limiter)")
    p_cy.add_argument("--grok-lanes", type=int, default=0,
                      help="grok/novelty ceiling (default 0: autonomous loop spends no grok usage)")
    p_cy.add_argument("--loop-secs", type=float, default=0.0,
                      help="tight internal reap+fill loop for N seconds (constant uptime, no per-tick startup)")
    p_cy.add_argument("--inner-sleep", type=float, default=3.0,
                      help="seconds between iterations of the tight loop (default 3)")
    p_ret = sp.add_parser("retire")
    p_ret.add_argument("oxx")
    p_acq = sp.add_parser("acquire-next")
    p_acq.add_argument("--dry-run", action="store_true",
                       help="plan only (default when --go is absent)")
    p_acq.add_argument("--go", action="store_true",
                       help="start hf download in the background")
    sp.add_parser("economics")
    args = ap.parse_args(argv)
    if args.self_check or args.cmd in {"self-check", "selfcheck"}:
        return _self_check()
    if args.cmd == "status":
        return cmd_status()
    if args.cmd == "queue":
        return cmd_queue()
    if args.cmd == "value":
        return cmd_value()
    if args.cmd == "harvest":
        return cmd_harvest(dry_run=bool(getattr(args, "dry_run", False)))
    if args.cmd == "packet":
        return cmd_packet(args.oxx)
    if args.cmd == "admit":
        return cmd_admit(args.slug, args.est_gib)
    if args.cmd == "completions":
        return cmd_completions(
            rebuild=bool(getattr(args, "rebuild", False)),
            completed_at=getattr(args, "completed_at", None),
        )
    if args.cmd == "run":
        go = bool(args.go) and not bool(args.dry_run)
        return cmd_run(go=go, max_lanes=args.max_lanes, grok_lanes=args.grok_lanes)
    if args.cmd == "cycle":
        go = bool(args.go) and not bool(args.dry_run)
        return cmd_cycle(go=go, max_lanes=args.max_lanes, grok_lanes=args.grok_lanes,
                         loop_secs=args.loop_secs, inner_sleep=args.inner_sleep)
    if args.cmd == "retire":
        return cmd_retire(args.oxx)
    if args.cmd == "acquire-next":
        go = bool(args.go) and not bool(args.dry_run)
        return cmd_acquire_next(go=go)
    if args.cmd == "economics":
        return cmd_economics()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
