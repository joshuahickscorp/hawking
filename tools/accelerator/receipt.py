"""Accelerator receipt schema. FRONT A (G043, steer S015 §79).

The steer's rule is blunt: NO RESULT WITHOUT PHYSICAL IDENTITY. Every Accelerator
receipt carries eight identities. Where one genuinely does not apply -- there is no
TransportIdentity for a single-device Metal run -- it is recorded ABSENT with a
reason, never omitted and never invented. That is the same discipline the kernel
library already uses, and it is what keeps a missing field from reading as a
covered one.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "hawking.accelerator.receipt.v1"

IDENTITIES = ("experiment", "machine", "device", "model",
              "representation", "kernel", "runtime", "transport")

# The steer's canonical experiment classes. A receipt may not invent a class.
EXPERIMENT_CLASSES = {
    "ACCEL-KERNEL", "ACCEL-FUSION", "ACCEL-LAYOUT", "ACCEL-MEMORY",
    "ACCEL-DISPATCH", "ACCEL-SCHEDULING", "ACCEL-REPRESENTATION", "ACCEL-STATE",
    "ACCEL-DEVICE", "ACCEL-C2M", "ACCEL-EGB", "ACCEL-HUMF", "ACCEL-SUSTAINED",
}

# Steer §80. Never promote too early.
KNOWLEDGE_LEVELS = ("INSTANCE", "MODEL_FAMILY", "ARCHITECTURE", "REPRESENTATION",
                    "SOC_FAMILY", "DEVICE_CLASS", "APPLE_GENERAL", "EGB_TOPOLOGY",
                    "GENERAL_PHYSICAL")


def absent(reason: str) -> dict[str, str]:
    return {"status": "ABSENT", "reason": reason}


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return None


def build(*, experiment_class: str, knowledge_level: str, identities: dict[str, Any],
          result: dict[str, Any], claim_boundary: str, passed: bool) -> dict[str, Any]:
    if experiment_class not in EXPERIMENT_CLASSES:
        raise ValueError(f"{experiment_class!r} is not a canonical class; "
                         f"known: {sorted(EXPERIMENT_CLASSES)}")
    if knowledge_level not in KNOWLEDGE_LEVELS:
        raise ValueError(f"{knowledge_level!r} is not a knowledge level")
    missing = [k for k in IDENTITIES if k not in identities]
    if missing:
        raise ValueError(f"receipt is missing identities {missing}; record them "
                         f"ABSENT with a reason rather than omitting them")
    for k in IDENTITIES:
        v = identities[k]
        if isinstance(v, dict) and v.get("status") == "ABSENT" and not v.get("reason"):
            raise ValueError(f"identity {k!r} is ABSENT without a reason")
    return {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_class": experiment_class,
        "knowledge_level": knowledge_level,
        "git_head": git_head(),
        "identities": {k: identities[k] for k in IDENTITIES},
        "result": result,
        "claim_boundary": claim_boundary,
        "pass": bool(passed),
    }


def write(receipt: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=1))
    return path
