"""Honest assessment of existing teacher traces vs T1 / T3 needs."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.odyssey._paths import TEACHER_MANIFEST


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def assess_teacher_traces() -> dict[str, Any]:
    if not TEACHER_MANIFEST.is_file():
        return {
            "status": "MANIFEST_MISSING",
            "gap": "no teacher trace manifest",
        }
    manifest = json.loads(TEACHER_MANIFEST.read_text(encoding="utf-8"))
    existing = manifest.get("existing") or {}
    meta = existing.get("glm52_teacher_capsules") or {}
    ledger_path = Path(meta["ledger"]) if meta.get("ledger") else None

    if ledger_path is None or not ledger_path.is_file():
        return {
            "status": "LEDGER_NOT_PRESENT",
            "manifest_status": manifest.get("status"),
            "declared_ledger": meta.get("ledger"),
            "note": meta.get("note"),
            "what_exists": None,
            "t1_need": _t1_need(),
            "t3_need": _t3_need(),
            "gap": "declared teacher ledger path is not on disk",
        }

    rows = _load_ledger(ledger_path)
    events = Counter(r.get("event") for r in rows)
    captured = [r for r in rows if r.get("event") == "TEACHER_CAPTURED"]
    layers: set[int] = set()
    capsules: set[str] = set()
    windows: set[str] = set()
    provenance = Counter()
    total_capsule_bytes = 0
    for r in captured:
        for L in r.get("layers") or []:
            layers.add(int(L))
        if r.get("capsule_id"):
            capsules.add(str(r["capsule_id"]))
        for w in r.get("window_ids") or []:
            windows.add(str(w))
        provenance[r.get("input_provenance") or "?"] += 1
        total_capsule_bytes += int(r.get("capsule_bytes") or 0)

    what = {
        "ledger_path": str(ledger_path),
        "ledger_bytes": ledger_path.stat().st_size,
        "n_lines": len(rows),
        "events": dict(events),
        "n_captured": len(captured),
        "n_failed": events.get("TEACHER_CAPTURE_FAILED", 0),
        "unique_capsule_ids": len(capsules),
        "layers_covered": sorted(layers),
        "n_layers_covered": len(layers),
        "window_ids": sorted(windows),
        "n_windows": len(windows),
        "input_provenance": dict(provenance),
        "sum_capsule_bytes": total_capsule_bytes,
        "scope": "layer-scoped teacher capsules (hidden states / router / MoE organ dumps)",
        "not_present": [
            "token-level trajectory traces over natural prompts",
            "full forward-chain teacher logits for sequence distillation",
            "multi-step reasoning rollouts with parent/student divergence labels",
            "T3 trajectory stabilization pairs",
        ],
        "calibration_note": (
            "GLM52_TEACHER_EVIDENCE_POLICY: calibration is sealed synthetic token-id "
            "probes (8 tokens), not natural text; routing stats are not natural-text routing."
        ),
        "manifest_note": meta.get("note"),
    }

    t1 = _t1_need()
    t3 = _t3_need()

    # Concrete gaps
    gap = {
        "for_t1_primary_training": (
            "Teacher capsules are representation-fitting evidence, not a text/math train set. "
            "T1 still needs the math-core and support-language corpora (0/2 present). "
            "Capsules do not substitute for those corpora."
        ),
        "for_t2_qat": (
            f"Layer-scoped capsules exist for {len(layers)} layers across {len(capsules)} capsule ids "
            f"and {len(windows)} windows ({len(captured)} successful captures of {len(rows)} ledger lines). "
            "Manifest marks this usable for T2-style representation work; coverage is partial "
            f"(failed captures={events.get('TEACHER_CAPTURE_FAILED', 0)}; "
            f"EMBEDDING_SEEDED_NOT_CHAINED={provenance.get('EMBEDDING_SEEDED_NOT_CHAINED', 0)})."
        ),
        "for_t3_trajectory": (
            "T3 requires full trajectory traces from the parent over real long-horizon tasks. "
            f"Have: {len(captured)} layer-capsule events, {len(windows)} synthetic windows, "
            "no trajectory length, no token sequences, no divergence labels. "
            "Gap: ~0 trajectory traces of the required kind; need on the order of thousands of "
            "parent rollouts (task-length sequences) spanning the profile's long-horizon set, "
            "not 122 layer-local organ dumps."
        ),
        "numeric_gap": {
            "have_ledger_lines": len(rows),
            "have_successful_captures": len(captured),
            "have_trajectory_traces": 0,
            "have_natural_text_windows": 0,
            "t3_target_order_of_magnitude_traces": "1e3–1e5 parent trajectories (not yet collected)",
            "scope_have": "per-layer organs over synthetic token probes",
            "scope_need_t3": "full-sequence parent traces over long-horizon tasks",
        },
    }

    return {
        "status": "PARTIAL",
        "manifest_status": manifest.get("status"),
        "required_for": manifest.get("required_for"),
        "gate": manifest.get("gate"),
        "what_exists": what,
        "t1_need": t1,
        "t3_need": t3,
        "gap": gap,
    }


def _t1_need() -> dict[str, Any]:
    return {
        "primary": "math-core content-addressed corpus (permissive licence)",
        "support": "support-language corpus (technical language, coding, tools)",
        "teacher_role": (
            "T1 is capability-conditioned continued training (CE / profile-weighted). "
            "Teacher capsules are optional aids, not the training set."
        ),
    }


def _t3_need() -> dict[str, Any]:
    return {
        "primary": "full trajectory traces from the parent (token + semantic divergence depth)",
        "manifest_gate": "T3 traces require the flagship source, which is released after M18",
        "shape": (
            "per-example: prompt, parent token sequence (or top-k logits), optional intermediate "
            "states, horizon length, domain tag; disjoint from evaluation memberships"
        ),
        "why_current_ledger_fails": (
            "current ledger records layer organ dumps on synthetic calibration windows — "
            "different object type from trajectory traces"
        ),
    }
