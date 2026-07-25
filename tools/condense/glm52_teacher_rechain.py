#!/usr/bin/env python3.12
"""Re-seed the teacher capsule chain after the calibration corpus changed.

The eviction gate had authorized nothing for three consecutive windows, so the source
traversal was accumulating every shard it fetched against a bounded disk floor it would
have hit around window six. The gate was right to refuse. The capsules on disk were sealed
against eight ids drawn from a SHA-256 stream -- uniform over the vocabulary, and therefore
evidence of no domain, no language, no router margin and no context length. Calibration has
since moved to 256 real corpus tokens, and chaining a 256-token capture onto an 8-token
carry-out would produce a trajectory that is arithmetically fine and means nothing.

So the fix is not to relax the membership check. It is to retire the superseded capsules
and rebuild the chain under the current calibration.

Retire, not delete: the old capsules move to an archive directory with a receipt naming
the calibration they were sealed under and why they were withdrawn. They stop being the
chain's parent because `_previous_capsule` only scans the live directory, and they remain
readable as the evidence of what Generation A actually measured.

The rebuild starts at layer 0 deliberately. A capsule with no parent seeds itself from the
embedding, which is exact at layer 0 and off-distribution anywhere deeper -- the
embedding-seed confound this project has already measured once. Capturing a contiguous run
from zero is what makes the chain real rather than merely present.

    python3.12 tools/condense/glm52_teacher_rechain.py --dry-run
    python3.12 tools/condense/glm52_teacher_rechain.py --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import glm52_teacher_capture as teacher  # noqa: E402

STATE = Path.home() / "Library/Application Support/Hawking/GLM52Gravity/source_fetch"
SOURCE_ROOT = STATE.parent / "source"
CAPSULES = STATE / "teacher/capsules"
ARCHIVE = STATE / "teacher/archive"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def current_membership() -> tuple[str, int, int]:
    vocab = int(teacher.official_config()["vocab_size"])
    ids = teacher.calibration_ids("teacher_fit", vocab_size=vocab,
                                  tokens=teacher.CALIBRATION_TOKENS)
    return teacher.membership_sha256(ids, "teacher_fit"), vocab, teacher.CALIBRATION_TOKENS


def survey() -> dict:
    """What is on disk, and which capsules disagree with the current calibration."""
    membership, vocab, tokens = current_membership()
    live, stale = [], []
    for path in sorted(CAPSULES.glob("*.json")):
        receipt = json.loads(path.read_text())
        row = {
            "capsule_id": receipt.get("capsule_id"),
            "layers": receipt.get("layers"),
            "membership": receipt.get("calibration_membership_sha256"),
            "split": receipt.get("calibration_split"),
            "tokens": receipt.get("calibration_tokens"),
        }
        (live if row["membership"] == membership else stale).append(row)

    graph = teacher._graph()
    shards = teacher.organ_shards(graph)
    resident = {p.name for p in SOURCE_ROOT.glob("model-*.safetensors")}
    ready = [
        layer for layer in range(int(teacher.official_config()["num_hidden_layers"]))
        if (need := shards.get(f"text_layer_{layer:02d}", set())) and need <= resident
    ]
    runs = teacher.contiguous_runs(ready)
    chainable = runs[0] if runs and runs[0][0] == 0 else []
    return {
        "at": _now(),
        "current_calibration": {"membership_sha256": membership,
                                "vocab_size": vocab, "tokens": tokens,
                                "split": "teacher_fit"},
        "capsules_live": live,
        "capsules_stale": stale,
        "resident_layers": ready,
        "chainable_from_zero": chainable,
    }


def apply(limit: int | None = None) -> dict:
    """Archive stale capsules, then capture one contiguous chain starting at layer 0."""
    state = survey()
    if not state["capsules_stale"]:
        return {**state, "action": "NOTHING_TO_DO",
                "note": "every live capsule already matches the current calibration"}
    chain = state["chainable_from_zero"]
    if not chain:
        return {**state, "action": "REFUSED",
                "note": "layer 0 is not resident, so a chain from zero cannot be built; "
                        "capturing deeper would seed from the embedding off-distribution"}
    if limit is not None:
        chain = chain[:limit]

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    stamp = _now().replace(":", "").replace("-", "")
    dest = ARCHIVE / f"precalibration_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    moved = []
    for row in state["capsules_stale"]:
        cid = row["capsule_id"]
        for suffix in (".json", ".npz"):
            src = CAPSULES / f"{cid}{suffix}"
            if src.exists():
                shutil.move(str(src), str(dest / src.name))
                moved.append(src.name)
    (dest / "WITHDRAWAL_RECEIPT.json").write_text(json.dumps({
        "schema": "hawking.glm52.capsule_withdrawal.v1",
        "at": _now(),
        "reason": "sealed against 8 ids from a SHA-256 stream, uniform over the vocabulary; "
                  "calibration has since moved to 256 real corpus tokens, and chaining "
                  "across the two would produce a trajectory that means nothing",
        "withdrawn_from_chain": True,
        "deleted": False,
        "capsules": state["capsules_stale"],
        "superseded_by_calibration": state["current_calibration"],
        "files": sorted(moved),
    }, indent=1) + "\n")

    outcome = teacher.ensure_captured(chain, source_root=SOURCE_ROOT, capsule_dir=CAPSULES)
    after = survey()
    return {
        "action": "RECHAINED",
        "archive": str(dest),
        "archived_files": sorted(moved),
        "captured_layers": chain,
        "capture_outcome": outcome,
        "capsules_live_after": [c["capsule_id"] for c in after["capsules_live"]],
        "capsules_stale_after": [c["capsule_id"] for c in after["capsules_stale"]],
        "current_calibration": after["current_calibration"],
        "at": _now(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="archive stale capsules and rebuild the chain")
    ap.add_argument("--limit", type=int, default=None,
                    help="capture only the first N layers of the chainable run")
    args = ap.parse_args()
    result = apply(args.limit) if args.apply else survey()
    print(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
