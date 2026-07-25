#!/usr/bin/env python3.12
"""Prove the sealed rollback actually rolls back, against the receipt that promised it.

`GLM52_SOURCE_RELEASE_RECEIPT.json` released 405.4 GB on one promise: "re-fetch
zai-org/GLM-5.2 @ b4734de4... and verify against the rehydration receipt." A promise is
not a green gate. This reads the live fetch ledger and grades the rehydration against
`GLM52_REHYDRATION_RECEIPT.json` shard by shard, so the claim survives contact with the
bytes that came back.

Three things are graded and reported separately, because they fail differently:

* **coverage** -- every shard the receipt names has been fetched and verified;
* **fidelity** -- each fetched shard's sha256 equals the pre-release sha256, so the
  re-fetched source is the same source and not merely a source;
* **packing** -- every verified shard produced a `.gravity`, so the artifact's tensor
  coverage is complete rather than merely large.

Partial rehydration reports PARTIAL with exact counts. It never reports GREEN early: a
rollback that is 90% proven is a rollback nobody should rely on.

    python3.12 tools/condense/glm52_rollback_seal.py [--out RECEIPT.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATE = Path.home() / "Library/Application Support/Hawking/GLM52Gravity"
LEDGER = STATE / "source_fetch/SOURCE_FETCH_LEDGER.jsonl"
COMPACT = Path.home() / "Desktop/GLM52-Gravity-SubBit"
REHYDRATION = REPO / "GLM52_REHYDRATION_RECEIPT.json"
RELEASE = REPO / "GLM52_SOURCE_RELEASE_RECEIPT.json"


def _rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]


def seal() -> dict:
    receipt = json.loads(REHYDRATION.read_text())
    release = json.loads(RELEASE.read_text())
    # The receipt hashes every manifest file; only the weight shards are what the fetcher
    # streams and the packer consumes. Grading coverage against the index file too would
    # hold the seal at PARTIAL forever for a file that was never in scope.
    expected = {name: row for name, row in receipt["per_file_sha256"].items()
                if name.endswith(".safetensors")}
    rows = _rows()

    verified = {r["shard"]: r for r in rows if r.get("status") == "VERIFIED"}
    packed_names = {p.stem + ".safetensors" for p in COMPACT.glob("*.gravity")} \
        if COMPACT.exists() else set()

    # Fidelity is graded on the hash the fetcher recorded, not re-read from disk: the
    # bodies are evicted by design, so the ledger receipt is the only durable witness.
    mismatched = sorted(
        name for name, row in verified.items()
        if row.get("sha256") and name in expected
        and row["sha256"] != expected[name]["sha256"]
    )
    unhashed = sorted(name for name, row in verified.items() if not row.get("sha256"))
    missing = sorted(set(expected) - set(verified))
    unpacked = sorted(set(verified) - packed_names)

    complete = not missing and not mismatched and not unpacked and not unhashed
    state = "GREEN" if complete else "PARTIAL"
    out = {
        "schema": "hawking.glm52.rollback_seal.v1",
        "state": state,
        "repo": receipt.get("blobs_api_url", "").split("/models/")[-1].split("/revision")[0],
        "revision": release.get("rollback", ""),
        "shards_expected": len(expected),
        "shards_verified": len(verified),
        "shards_packed": len(packed_names),
        "coverage_fraction": round(len(verified) / max(1, len(expected)), 6),
        "packing_fraction": round(len(packed_names) / max(1, len(expected)), 6),
        "hash_mismatches": mismatched,
        "verified_without_recorded_hash": unhashed,
        "shards_missing": len(missing),
        "shards_verified_but_unpacked": len(unpacked),
        "released_gb": release.get("reclaimed_gb"),
        "grades": {
            "coverage": "GREEN" if not missing else "PARTIAL",
            "fidelity": "GREEN" if not (mismatched or unhashed) else "FAILED",
            "packing": "GREEN" if not unpacked and packed_names else "PARTIAL",
        },
        "meaning": ("the released source is provably recoverable and every recovered shard "
                    "is byte-identical to the one that was deleted"
                    if complete else
                    "rehydration in progress or incomplete; this is not a rollback guarantee"),
    }
    out["seal_sha256"] = hashlib.sha256(
        json.dumps(out, sort_keys=True).encode()).hexdigest()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = seal()
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report["grades"]["fidelity"] != "FAILED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
