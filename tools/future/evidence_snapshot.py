"""Pinned evidence snapshot for sidecar lanes.

Codex writes its authoritative receipts into `receipts/headless/`, and most of
the current ones are NOT committed. A sidecar lane running in a sparse git
worktree therefore cannot see them at all: not on disk, not in HEAD. Building
against a fixture instead would quietly turn evidence-grounded work into
made-up work, which is the exact failure this campaign exists to prevent.

So: copy the receipts lanes need into the sidecar partition, with a manifest
recording each file's source path, size, sha256 and mtime at capture time. That
is strictly better than reading the live directory anyway -- it pins the
evidence, so a lane's conclusion stays attached to the exact bytes it saw while
Codex keeps moving underneath.

The snapshot is a COPY OF EVIDENCE, never an edit of it. Codex's originals are
opened read-only and are never modified.

    python3 tools/future/evidence_snapshot.py --build
    python3 tools/future/evidence_snapshot.py --verify
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import shutil
import sys
import time
from pathlib import Path

from tools.future._common import REPO, sha256_file, write_receipt

SNAP = REPO / "receipts" / "future" / "evidence"

# Curated: exactly what the sidecar lanes were told to read. Not the whole 65 MB
# directory -- a snapshot nobody consumes is just a second copy of the problem.
WANTED = [
    # physical qualification frontier
    "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
    "receipts/headless/ACCELERATOR_SCOREBOARD.json",
    "receipts/headless/ACCELERATOR_REPATRIATION_QUEUE.json",
    "receipts/headless/ACCELERATOR_REPATRIATION_AUDIT.json",
    "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json",
    "receipts/headless/ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json",
    "receipts/headless/ACCELERATOR_TRANSFER_VERIFIED.json",
    "receipts/headless/ACCELERATOR_FRONT_F_ODYSSEY_PASS.json",
    # Flash frontier
    "receipts/headless/FLASH_COMPLETE_V0.nx.json",
    "receipts/headless/FLASH_COMPLETE_V2.nr.json",
    "receipts/headless/FLASH_NEXT_MACHINE.nx.json",
    "receipts/headless/FLASH_META_REPRESENTATION_SUB1.json",
    "receipts/headless/FLASH_META_COHERENCE_SCREEN_L4.json",
    "receipts/headless/FLASH_ORGAN_CENSUS.json",
    "receipts/headless/FLASH_ROUTER_SENSITIVITY_MAP_L3_L4.json",
    "receipts/headless/FLASH_ROUTE_STABILITY.json",
    "receipts/headless/FLASH_NOETIC_ROUTER_SELECTION.json",
    "receipts/headless/FLASH_DOCTOR_EXPERT_BANK_SCREEN_FULL_L44.json",
    "receipts/headless/FLASH_LAYER30_CRITICAL_PATH.json",
    "receipts/headless/FLASH_LAYER10_CRITICAL_PATH.json",
    "receipts/headless/FLASH_LAYER46_DISPATCH_LEDGER.json",
    "receipts/headless/FLASH_STATEFUL_TPS_GATE_V14.json",
    "receipts/headless/FLASH_NEXT_FPGA_ORGAN_MAP.json",
    "receipts/headless/FLASH_ATTENTION_ROUTE_UNION_PARITY.json",
    # Qwen27
    "receipts/headless/QWEN27_TOKEN_NS_BUDGET.json",
    "receipts/headless/QWEN27_FPGA_ORGAN_MAP.json",
    "receipts/headless/QWEN38_ACCELERATOR_TRANSFER_MAP.json",
    "receipts/headless/QWEN80_BIT_BUDGET_LEDGER.json",
    # ANE / FPGA / device
    "receipts/headless/APPLE_ANE_ATLAS.json",
    "receipts/headless/APPLE_ANE_DEVICE_PROFILE.json",
    "receipts/headless/HCLI_FPGA_PREBOARD.json",
    "receipts/headless/FLASH_FPGA_PREBOARD_CURRENT.json",
    # Odyssey / transfer / negative science
    "receipts/headless/ODYSSEY_TRANSFER_PROVEN.json",
    "receipts/headless/ODYSSEY_ADVERSARIAL_SWEEP.json",
    "receipts/headless/ODYSSEY_LEARNING_CURVE.json",
    "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
    "receipts/headless/DOCTOR_TRANSFER.json",
    "receipts/headless/DENSE_SUBBIT_TRANSFER.json",
    "receipts/headless/HCLI_ACCELERATOR_REGRESSION.json",
    "receipts/headless/HCLI_MODELLAKE_FLASH_CENSUS.json",
    "receipts/QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json",
    "receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
    "tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
    "workspace/campaign/odyssey/NEGATIVE_SCIENCE.json",
    # Added 2026-08-29 in compounding mode: Codex produced these after the
    # scaffold campaign closed, and downstream lanes reason over both.
    "receipts/headless/ACCELERATOR_REPATRIATION_EFFECTS.json",
    "receipts/headless/FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json",
    "receipts/headless/ACCELERATOR_FRONT_G_P6.json",
    # 2026-08-30 phase transition: Codex sealed its Accelerator handoff at the
    # repo root. It is the training trace the whole resident metabolism is
    # extracted from, so it must be pinned like any other authority.
    "CODEX_ACCELERATOR_HANDOFF.json",
]


def build() -> Path:
    SNAP.mkdir(parents=True, exist_ok=True)
    captured, missing = [], []
    for rel in WANTED:
        src = REPO / rel
        if not src.exists():
            missing.append(rel)
            continue
        # Flatten: the manifest keeps the true source path, so lanes need only a name.
        dst = SNAP / Path(rel).name
        shutil.copy2(src, dst)
        st = src.stat()
        captured.append(
            {
                "name": dst.name,
                "source_path": rel,
                "snapshot_path": f"receipts/future/evidence/{dst.name}",
                "bytes": st.st_size,
                "sha256": sha256_file(src),
                "source_mtime_epoch": round(st.st_mtime, 3),
                "source_mtime_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)
                ),
            }
        )
    doc = {
        "schema": "hawking.future.evidence_snapshot.v1",
        "version": 1,
        "purpose": (
            "Pinned copy of the Codex receipts sidecar lanes read. Lanes run in sparse "
            "git worktrees where the live, uncommitted receipts are invisible; without "
            "this snapshot a lane would silently fall back to a fixture and its "
            "conclusions would be attached to nothing."
        ),
        "read_only_guarantee": "Codex originals are opened read-only and never modified.",
        "pinning_rationale": (
            "Pinned bytes are better than a live read even when a live read is possible: "
            "Codex keeps writing, and a conclusion must stay attached to the exact "
            "evidence that produced it."
        ),
        "counts": {"wanted": len(WANTED), "captured": len(captured), "missing": len(missing)},
        "captured": captured,
        "missing": missing,
        "missing_meaning": (
            "A wanted receipt absent from disk. That is a real negative finding about "
            "the project's evidence, not a snapshot bug."
        ),
    }
    return write_receipt("EVIDENCE_SNAPSHOT.json", doc, "tools/future/evidence_snapshot.py")


def verify() -> int:
    """Every captured file must still hash to what the manifest recorded."""
    import json

    man = REPO / "receipts" / "future" / "EVIDENCE_SNAPSHOT.json"
    if not man.exists():
        print("no manifest; run --build", file=sys.stderr)
        return 1
    doc = json.loads(man.read_text())
    bad = []
    for row in doc["captured"]:
        p = REPO / row["snapshot_path"]
        if not p.exists():
            bad.append((row["name"], "snapshot file missing"))
            continue
        if sha256_file(p) != row["sha256"]:
            bad.append((row["name"], "sha256 mismatch against manifest"))
    if bad:
        for n, why in bad:
            print(f"  {n}: {why}", file=sys.stderr)
        return 1
    print(f"evidence snapshot verified: {len(doc['captured'])} files hash-match the manifest")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        return verify()
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
