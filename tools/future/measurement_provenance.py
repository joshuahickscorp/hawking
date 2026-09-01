#!/usr/bin/env python3
"""Every TPS number measured today came from source that is in no commit.

The release binary that produced 628 dispatches, 35.158 TPS, the region
timeline and the FUSE_BA_DELTA A/B was built from a working tree carrying 2,679
uncommitted lines across three crate files. `TokenPipelineCache` and the batched
dispatch helpers those measurements depend on appear 16 times in the working
tree and ZERO times in HEAD.

This is the artifact-identity problem one level up. artifact_identity.py refuses
a binary older than the commit introducing a field it reads. It does not catch a
binary built from source that was never committed at all — there is no commit to
compare against, and the check passes precisely because nothing is there.

The measurements are not wrong. The build is reproducible only from a working
tree that exists on one machine, in one directory, unversioned. If that tree is
lost, no receipt in this campaign can be regenerated.

It surfaced when a lane's own Metal batching patch collided with it: n1 built
cleanly in a worktree from HEAD, and its patch would not apply here, because
both implement the same symbols independently. Preserved unapplied at
receipts/future/patches/n1-region-timing.crate.patch.

    python3 tools/future/measurement_provenance.py --record
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, git  # noqa: E402, require_known_flags

RECEIPT = REPO / "receipts" / "future" / "MEASUREMENT_PROVENANCE.json"
BINARY = REPO / "workspace/ops/build/rust/release/examples/ascension_qwen38_resident"
CRATE_FILES = (
    "crates/hawking-core/src/metal/mod.rs",
    "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
    "crates/hawking-core/examples/ascension_qwen38_resident.rs",
)
# A symbol the measurements depend on, to make the gap concrete.
WITNESS = "TokenPipelineCache"

MEASUREMENTS_AT_RISK = (
    "receipts/future/RESIDENT_TOKEN_BUDGET.json",
    "receipts/future/ORGAN_BANDWIDTH.json",
    "receipts/future/BA_DELTA_AB.json",
    "receipts/future/RESIDENT_BINARY_DRIFT.json",
)


def _witness_counts() -> dict[str, Any]:
    src = (REPO / CRATE_FILES[0])
    in_tree = src.read_text().count(WITNESS) if src.exists() else 0
    head = git("show", f"HEAD:{CRATE_FILES[0]}")
    return {
        "symbol": WITNESS,
        "file": CRATE_FILES[0],
        "occurrences_in_working_tree": in_tree,
        "occurrences_in_HEAD": head.count(WITNESS),
        "conclusion": "the measurements depend on a symbol that exists only in "
                      "the working tree" if in_tree and not head.count(WITNESS)
                      else "symbol is committed",
    }


def build() -> dict[str, Any]:
    stat = git("diff", "--numstat", "--", *CRATE_FILES).splitlines()
    added = sum(int(r.split("\t")[0]) for r in stat if r.split("\t")[0].isdigit())
    removed = sum(int(r.split("\t")[1]) for r in stat if r.split("\t")[1].isdigit())
    digest = None
    if BINARY.exists():
        h = hashlib.sha256()
        with BINARY.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
    return {
        "schema": "hawking.future.measurement_provenance.v1",
        "version": 1,
        "recorded_by": "tools/future/measurement_provenance.py",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "severity": "HIGH",
        "defect": {
            "class": "MEASURED_FROM_UNCOMMITTED_SOURCE",
            "what": "the binary every measurement in this campaign came from was "
                    "built from a working tree that is in no commit",
            "uncommitted_lines_added": added,
            "uncommitted_lines_removed": removed,
            "files": list(CRATE_FILES),
            "dirty_crate_files_total": len(
                [x for x in git("status", "--porcelain", "--", "crates/").splitlines() if x]),
            "last_commit_touching_metal_mod": git(
                "log", "-1", "--format=%h %ad %s", "--date=short", "--", CRATE_FILES[0]),
        },
        "witness": _witness_counts(),
        "binary": {"path": str(BINARY.relative_to(REPO)) if BINARY.exists() else None,
                   "sha256": digest},
        "measurements_that_inherit_this": list(MEASUREMENTS_AT_RISK),
        "why_artifact_identity_does_not_catch_it": (
            "artifact_identity.py refuses a binary OLDER than the commit that "
            "introduced a field it reads. Here there is no such commit. The "
            "check passes because the source was never committed, which is the "
            "worse case, not the safe one."
        ),
        "what_is_and_is_not_at_risk": {
            "not_at_risk": [
                "the numbers themselves: they were measured, repeatedly, on a "
                "binary that runs with fallbacks 0 and produces coherent text",
                "the relative findings: the fusion A/B, the BA_DELTA A/B and the "
                "organ table are all paired or ratio measurements",
            ],
            "at_risk": [
                "reproducibility: the build exists only in one working directory "
                "on one machine",
                "review: 2,679 lines of Metal batching and decode changes have "
                "not been read by anyone",
                "collision: an independently written lane patch for the same "
                "symbols could not be applied, and was preserved unapplied",
            ],
        },
        "collision": {
            "lane": "n1regiontime",
            "what": "built cleanly in a worktree from HEAD and implements the "
                    "same Metal batching symbols independently",
            "preserved_at": "receipts/future/patches/n1-region-timing.crate.patch",
            "not_applied_because": "it collides with the uncommitted "
                                   "implementation the measurements were taken on",
            "its_measurement_is_kept": "receipts/future/ORGAN_BANDWIDTH.json — the "
                                       "region timeline it produced is recorded "
                                       "even though its instrumentation is not "
                                       "landed",
        },
        "obligation": (
            "Read, review and commit the uncommitted crate work, or quarantine "
            "it explicitly. Committing 2,679 unreviewed lines of Metal code to "
            "close a provenance gap would trade one defect for a worse one, so "
            "this receipt records the gap rather than papering it."
        ),
        "claim_boundary": (
            "A diff stat, a symbol count against HEAD, and a binary digest. It "
            "asserts that the source is uncommitted, not that the source is "
            "wrong. No measurement recorded elsewhere is retracted."
        ),
    }


def record() -> Path:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(build(), indent=1, sort_keys=True, default=str) + "\n")
    return RECEIPT


if __name__ == "__main__":
    from _common import require_known_flags
    require_known_flags(["--build", "--record"])
    d = build()
    if "--record" in sys.argv:
        print(f"wrote {record()}")
    print(f"uncommitted crate lines: +{d['defect']['uncommitted_lines_added']} "
          f"-{d['defect']['uncommitted_lines_removed']}")
    print(f"witness {d['witness']['symbol']}: "
          f"{d['witness']['occurrences_in_working_tree']} in tree, "
          f"{d['witness']['occurrences_in_HEAD']} in HEAD")
