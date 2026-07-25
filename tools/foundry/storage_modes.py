#!/usr/bin/env python3.12
"""Storage/Vulture lifecycle for the deep architecture foundry.

Three storage modes, chosen from BYTES, never from parameter count.

    FULL_DISK_RESIDENT    the whole official manifest lands on disk, then the
                          harvest/pack pass runs over local files.
    VULTURE_SHARD_SERIAL  one shard (or range group) at a time: fetch, verify,
                          harvest the raw references/statistics, pack every
                          tensor the shard contains, append to the compact
                          artifact, seal dependencies, release the raw bytes,
                          continue. Full parent coverage is still achieved.
    BOUNDED_REMOTE_RANGE  not even one shard may land: per-tensor HTTP range
                          windows, bounded by the working window.

Why bytes and not params: on-disk size is precision-driven. A 397B bf16 parent
(~2.0 B/param, ~794 GB) is LARGER on disk than a 1T int4 parent (~0.60 B/param,
~595 GB). Any rule that reads parameter count picks the wrong mode for both.
`choose_mode` therefore takes no parameter-count argument at all.

Release policy (the vulture-release predicate) is deliberately SEPARATE from the
capability gate in quality_contract.py: raw source may be released once it is
re-downloadable from a pinned immutable revision and the rehydration metadata
(config, index, tokenizer) is retained. It does not require a passing artifact.
GPT-OSS-120B (F0) was released under exactly this predicate after harvest.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCHEMA_DECISION = "hawking.foundry.storage_decision.v1"

FULL_DISK_RESIDENT = "FULL_DISK_RESIDENT"
VULTURE_SHARD_SERIAL = "VULTURE_SHARD_SERIAL"
BOUNDED_REMOTE_RANGE = "BOUNDED_REMOTE_RANGE"
MODES = (FULL_DISK_RESIDENT, VULTURE_SHARD_SERIAL, BOUNDED_REMOTE_RANGE)

# Working window a shard-serial pass needs resident at once: one shard being
# harvested plus one being fetched, plus the growing compact artifact.
# ponytail: single scalar knob, not a per-parent model. Real shards vary
# (F0 ships 7 shards of ~9.3 GB); raise this if a parent ships fatter shards.
DEFAULT_WORKING_WINDOW_BYTES = 25 * 10**9

# Measured, do not raise: a 64 GB expert cache gave ZERO evictions on a single
# lockstep pass (no cross-layer reuse) and drove RAM 70 -> 18 GB with swap at
# 906 MB free. Aggressive RAM only where real reuse exists.
EXPERT_CACHE_CAP_BYTES = 20 * 10**9

REQUIRED_REHYDRATION_METADATA = frozenset({"config", "index", "tokenizer"})
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_MUTABLE_LABELS = frozenset({"main", "master", "latest", "head", "dev"})


class CoverageError(AssertionError):
    """A vulture pass would drop a tensor. Never downgrade this to a warning."""


# ── mode choice ───────────────────────────────────────────────────────────────

def choose_mode(
    official_manifest_bytes: int,
    live_free_bytes: int,
    reserve_bytes: int,
    parent_id: str,
    *,
    working_window_bytes: int = DEFAULT_WORKING_WINDOW_BYTES,
) -> dict:
    """Pick a storage mode from the official manifest byte total and live disk.

    `live_free_bytes` must be measured AFTER the parent-side state that already
    exists on disk (post-parent), not from a stale pre-run reading.
    """
    for name, value in (
        ("official_manifest_bytes", official_manifest_bytes),
        ("live_free_bytes", live_free_bytes),
        ("reserve_bytes", reserve_bytes),
        ("working_window_bytes", working_window_bytes),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an int number of bytes, got {value!r}")
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    if official_manifest_bytes == 0:
        raise ValueError("official_manifest_bytes must be > 0 (no manifest, no decision)")
    if not parent_id:
        raise ValueError("parent_id is required")

    usable = max(0, live_free_bytes - reserve_bytes)

    if official_manifest_bytes <= usable:
        mode = FULL_DISK_RESIDENT
        reason = (
            f"official manifest {official_manifest_bytes} B fits in usable "
            f"{usable} B (free {live_free_bytes} - reserve {reserve_bytes})"
        )
    elif usable >= working_window_bytes:
        mode = VULTURE_SHARD_SERIAL
        reason = (
            f"official manifest {official_manifest_bytes} B exceeds usable {usable} B, "
            f"but usable covers the {working_window_bytes} B shard-serial working window"
        )
    else:
        mode = BOUNDED_REMOTE_RANGE
        reason = (
            f"usable {usable} B is below the {working_window_bytes} B shard-serial "
            "working window: no shard may land, range windows only"
        )

    return {
        "schema": SCHEMA_DECISION,
        "parent_id": parent_id,
        "mode": mode,
        "reason": reason,
        "official_manifest_bytes": official_manifest_bytes,
        "live_free_bytes": live_free_bytes,
        "reserve_bytes": reserve_bytes,
        "usable_bytes": usable,
        "headroom_bytes": usable - official_manifest_bytes,
        "working_window_bytes": working_window_bytes,
        "expert_cache_cap_bytes": EXPERT_CACHE_CAP_BYTES,
        "decided_from": ["official_manifest_bytes", "live_free_bytes", "reserve_bytes"],
        "never_decided_from": ["parameter_count", "active_params", "expert_count"],
        "full_parent_coverage": True,
    }


# ── vulture coverage invariant ────────────────────────────────────────────────

def vulture_coverage(manifest_tensors, packed_tensors) -> dict:
    """Shard-serial is a streaming order, NOT a subset. Coverage must be total.

    Every tensor named in the official manifest index must appear in the compact
    artifact; nothing outside the manifest may appear.
    """
    manifest = set(manifest_tensors)
    packed = set(packed_tensors)
    missing = sorted(manifest - packed)
    unexpected = sorted(packed - manifest)
    return {
        "invariant": "vulture_shard_serial_achieves_full_parent_coverage",
        "manifest_count": len(manifest),
        "packed_count": len(packed),
        "missing": missing,
        "unexpected": unexpected,
        "complete": not missing and not unexpected,
    }


def assert_full_coverage(manifest_tensors, packed_tensors) -> dict:
    result = vulture_coverage(manifest_tensors, packed_tensors)
    if not result["complete"]:
        raise CoverageError(
            f"vulture pass is not total: {len(result['missing'])} missing "
            f"{result['missing'][:5]}, {len(result['unexpected'])} unexpected "
            f"{result['unexpected'][:5]}"
        )
    return result


def assert_shard_release_ordered(shard_tensors, packed_at_release) -> None:
    """Raw bytes of a shard may only be released after ALL its tensors are packed
    and their dependencies sealed. Releasing early is how coverage silently rots."""
    unpacked = sorted(set(shard_tensors) - set(packed_at_release))
    if unpacked:
        raise CoverageError(
            f"raw shard released with {len(unpacked)} tensor(s) still unpacked: {unpacked[:5]}"
        )


# ── vulture release predicate ─────────────────────────────────────────────────

def vulture_release_ok(pinned_revision, retained_metadata) -> dict:
    """May the raw source bytes be deleted after harvest?

    Two conditions, both necessary: the source is re-downloadable from a pinned
    IMMUTABLE revision, and the rehydration metadata is retained locally.
    This predicate is independent of the capability gate: a NEGATIVE quality
    result does not block release, and a release does not imply a passing
    artifact.
    """
    reasons = []
    rev = (pinned_revision or "").strip().lower()
    if not rev:
        reasons.append("no pinned revision recorded")
    elif rev in _MUTABLE_LABELS or not _IMMUTABLE_REVISION.match(rev):
        reasons.append(f"revision {pinned_revision!r} is not an immutable 40-hex commit sha")

    retained = set(retained_metadata or ())
    absent = sorted(REQUIRED_REHYDRATION_METADATA - retained)
    if absent:
        reasons.append(f"rehydration metadata not retained: {absent}")

    return {
        "predicate": "vulture_release",
        "ok": not reasons,
        "reasons": reasons,
        "pinned_revision": pinned_revision,
        "retained_metadata": sorted(retained),
        "required_metadata": sorted(REQUIRED_REHYDRATION_METADATA),
        "independent_of_capability_gate": True,
    }


# ── known ladder ──────────────────────────────────────────────────────────────
# There is exactly ONE ladder authority: HAWKING_LADDER_V3.json, whose rungs and byte totals were
# resolved live against the official repositories (repo, license, immutable revision, shard manifest).
# A hardcoded copy here was stale within one wave (it had F2=DeepSeek-V3.2, F3=Kimi, F4=V4-Pro against
# the real F2=Qwen3.5-397B, F3=MiniMax-M3, F4=DeepSeek-V3.2, F5=GLM, F6=Kimi, F7=V4-Pro, and an
# ESTIMATED F1 byte total where a measured manifest exists). Duplicating the ladder is how the two
# copies silently disagree, so this reads the authority instead of restating it.
LADDER_AUTHORITY = Path(__file__).resolve().parents[2] / "HAWKING_LADDER_V3.json"


def load_ladder(path: Path | None = None) -> list[dict]:
    """Rungs from the one ladder authority. Rungs with no released weights (no bytes) are skipped:
    a storage mode cannot be assigned to bytes that do not exist."""
    src = Path(path) if path is not None else LADDER_AUTHORITY
    doc = json.loads(src.read_text())
    out: list[dict] = []
    for r in doc.get("rungs") or []:
        if not isinstance(r, dict):
            continue
        size = r.get("size")
        if not isinstance(size, dict):
            continue  # unresolved rung records size as a note string (e.g. F8), not a manifest
        total_gb = size.get("total_main_gb")
        if not total_gb:
            continue  # unresolved / unreleased rung: no bytes, no decision
        out.append({
            "parent_id": f"{r.get('rung', '?')}:{r.get('official_repo', '?')}",
            "bytes": int(round(float(total_gb) * 10**9)),
            "basis": size.get("evidence", "HAWKING_LADDER_V3 size.total_main_gb"),
        })
    return out


def ladder_decisions(live_free_bytes: int, reserve_bytes: int,
                     ladder: list[dict] | None = None) -> list:
    rungs = ladder if ladder is not None else load_ladder()
    return [
        dict(choose_mode(p["bytes"], live_free_bytes, reserve_bytes, p["parent_id"]), basis=p["basis"])
        for p in rungs
    ]


if __name__ == "__main__":
    import shutil

    free = shutil.disk_usage("/").free if len(sys.argv) < 2 else int(sys.argv[1])
    reserve = 50 * 10**9 if len(sys.argv) < 3 else int(sys.argv[2])
    print(json.dumps(ladder_decisions(free, reserve), indent=1))
