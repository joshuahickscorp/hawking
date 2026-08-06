#!/usr/bin/env python3.12
"""Hidden evaluation memberships with mechanical hiding.

Hiding is not a naming convention. The training-visible path may only read the
public selection set and the commitment (hash of the hidden set). The hidden
items live in a separate file under odyssey/evaluation/hidden/ and are never
imported by training objective or data loaders.

Commitment protocol:
  commitment = sha256(canonical_jsonl of hidden items)
  Training code that opens the hidden items file is a contract violation;
  the training path API only exposes `commitment` and `n_hidden`, not item ids.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from tools.odyssey._paths import HIDDEN_DIR, PUBLIC_EVAL_DIR, ROOT, TRAINING_DIR

SCHEMA = "hawking.odyssey.hidden_memberships.v1"
HIDDEN_ITEMS = HIDDEN_DIR / "hidden_items.jsonl"
COMMITMENT = HIDDEN_DIR / "HIDDEN_MEMBERSHIP_COMMITMENT.json"
PUBLIC_SELECTION = PUBLIC_EVAL_DIR / "selection_items.jsonl"
TRAINING_VISIBLE = TRAINING_DIR / "TRAINING_VISIBLE_EVAL.json"

# Seed hidden items for T0 (synthetic held-out probes; not training data).
_SEED_HIDDEN = [
    {
        "id": "hid_math_01",
        "set": "hidden",
        "domain": "math",
        "prompt": "Evaluate the contour integral of 1/z around the unit circle (answer as 2*pi*i or equivalent).",
        "expect_contains": ["2", "pi", "i"],
    },
    {
        "id": "hid_halo_tech_01",
        "set": "hidden",
        "domain": "technical_language",
        "prompt": "Define 'idempotent' for a database write in one sentence.",
        "expect_contains": ["idempotent"],
    },
    {
        "id": "hid_halo_code_01",
        "set": "hidden",
        "domain": "coding",
        "prompt": "Write a Python function is_palindrome(s) that returns True iff s equals its reverse.",
        "expect_contains": ["def is_palindrome"],
    },
    {
        "id": "hid_reason_01",
        "set": "hidden",
        "domain": "general_reasoning",
        "prompt": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?",
        "expect_contains": ["0.05", "5 cent"],
    },
    {
        "id": "hid_uncert_01",
        "set": "hidden",
        "domain": "uncertainty_calibration",
        "prompt": "What is the capital of a country that does not exist: Atlantis? If unknown, say UNKNOWN.",
        "expect_contains": ["UNKNOWN"],
    },
]

_SEED_PUBLIC = [
    {
        "id": "sel_math_01",
        "set": "selection",
        "domain": "math",
        "prompt": "What is 2+2?",
        "expect_contains": ["4"],
    },
    {
        "id": "sel_halo_tech_01",
        "set": "selection",
        "domain": "technical_language",
        "prompt": "What does BPW mean in compression?",
        "expect_contains": ["bits per weight"],
    },
]


def _canonical_line(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256_lines(lines: Iterable[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def write_seed_sets() -> dict[str, Any]:
    HIDDEN_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    hidden_lines = [_canonical_line(x) for x in _SEED_HIDDEN]
    public_lines = [_canonical_line(x) for x in _SEED_PUBLIC]
    HIDDEN_ITEMS.write_text("\n".join(hidden_lines) + "\n")
    PUBLIC_SELECTION.write_text("\n".join(public_lines) + "\n")

    commitment_hash = _sha256_lines(hidden_lines)
    commitment = {
        "schema": SCHEMA,
        "hidden_path": str(HIDDEN_ITEMS.relative_to(ROOT)),
        "n_hidden": len(_SEED_HIDDEN),
        "commitment_sha256": commitment_hash,
        "algorithm": "sha256 over canonical JSONL lines (sort_keys, no spaces) each terminated by \\n",
        "public_selection_path": str(PUBLIC_SELECTION.relative_to(ROOT)),
        "n_public_selection": len(_SEED_PUBLIC),
        "invariants": [
            "training path may read commitment and public selection only",
            "training path must not open hidden_items.jsonl",
            "a support-halo regression visible only on hidden items is protected by this split",
        ],
    }
    COMMITMENT.write_text(json.dumps(commitment, indent=2, sort_keys=True) + "\n")

    # Training-visible surface: no item ids from hidden set.
    training_visible = {
        "schema": "hawking.odyssey.training_visible_eval.v1",
        "public_selection_path": str(PUBLIC_SELECTION.relative_to(ROOT)),
        "hidden_commitment_path": str(COMMITMENT.relative_to(ROOT)),
        "hidden_commitment_sha256": commitment_hash,
        "n_hidden_committed": len(_SEED_HIDDEN),
        "hidden_item_ids_visible": False,
        "note": "This file is the only eval membership surface training code may import.",
    }
    TRAINING_VISIBLE.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_VISIBLE.write_text(json.dumps(training_visible, indent=2, sort_keys=True) + "\n")
    return commitment


def load_training_visible() -> dict[str, Any]:
    """API for the training path: commitment + public selection, never hidden ids."""
    if not TRAINING_VISIBLE.is_file():
        write_seed_sets()
    visible = json.loads(TRAINING_VISIBLE.read_text())
    public = []
    for line in PUBLIC_SELECTION.read_text().splitlines():
        if line.strip():
            public.append(json.loads(line))
    # Strip any accidental hidden leakage.
    public = [p for p in public if p.get("set") == "selection"]
    return {
        "public_selection": public,
        "hidden_commitment_sha256": visible["hidden_commitment_sha256"],
        "n_hidden_committed": visible["n_hidden_committed"],
        "hidden_item_ids": None,  # mechanical refusal
    }


def verify_commitment() -> dict[str, Any]:
    if not HIDDEN_ITEMS.is_file() or not COMMITMENT.is_file():
        write_seed_sets()
    lines = [ln for ln in HIDDEN_ITEMS.read_text().splitlines() if ln.strip()]
    # Re-canonicalize
    objs = [json.loads(ln) for ln in lines]
    recomputed = _sha256_lines(_canonical_line(o) for o in objs)
    committed = json.loads(COMMITMENT.read_text())
    match = recomputed == committed["commitment_sha256"]
    # Training-visible must not contain hidden ids.
    tv = load_training_visible()
    hidden_ids = {o["id"] for o in objs}
    leaked = [p["id"] for p in tv["public_selection"] if p["id"] in hidden_ids]
    return {
        "status": "PASS" if match and not leaked and tv["hidden_item_ids"] is None else "FAIL",
        "commitment_match": match,
        "recomputed": recomputed,
        "committed": committed["commitment_sha256"],
        "n_hidden": len(objs),
        "hidden_ids_leaked_to_training_visible": leaked,
        "training_hidden_item_ids_is_none": tv["hidden_item_ids"] is None,
    }


def main(argv: list[str] | None = None) -> int:
    write_seed_sets()
    result = verify_commitment()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())
