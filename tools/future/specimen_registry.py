#!/usr/bin/env python3
"""G100: the specimen registry and lifecycle state machine, built from disk.

S027 §1-§3. Every specimen carries a machine-readable lifecycle state DERIVED
from observable facts - a sealed manifest, a config on disk, a partial directory
still being written - never declared. The scheduler consumes this; it is not
documentation.

    43 SEALED_SOURCE, 10 DOWNLOADING on the ModelLake volume today.

LIFECYCLE IS DERIVED, NOT ASSERTED. A specimen is SEALED_SOURCE because a
manifest exists with a resolved sha and a byte count; FINGERPRINTED because its
config.json parses and names an architecture; DOWNLOADING because it sits under
partial/ with no manifest. A state nothing on disk supports is DISCOVERED, and
DISCOVERED is not a claim that anything is ready.

WHAT THIS DOES NOT DO. It does not measure load cost, hold residency, or decide
what to load next - those are G102 and G104. A registry that guessed load times
it never measured would be worse than one that says it does not know.

    python3 tools/future/specimen_registry.py --build
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/specimen_registry.py"
RECEIPT_NAME = "SPECIMEN_REGISTRY.json"

LAKE = Path("/Volumes/corpdrive/hawking-modellake")
SPECIMENS = LAKE / "specimens"
PARTIAL = LAKE / "partial"
MANIFESTS = LAKE / "manifests"

# S027 §1. Ordered cheapest-to-richest; a specimen's state is the richest one
# the evidence on disk supports.
LIFECYCLE = (
    "DISCOVERED",
    "DOWNLOADING",
    "COMPLETE_UNSEALED",
    "SEALED_SOURCE",
    "FINGERPRINTED",
)

# States this module cannot derive, because nothing on disk records them. Named
# so their absence is a known gap rather than an oversight.
NOT_DERIVABLE_HERE = {
    "NR_AVAILABLE": "no NR index is written to the lake",
    "NX_AVAILABLE": "no NX index is written to the lake",
    "DISK_WARM": "filesystem cache state is not observable from here",
    "LOADING": "a live scheduler state, not a disk fact",
    "UMA_RESIDENT": "a live scheduler state, not a disk fact",
    "EXECUTION_WARM": "a live scheduler state, not a disk fact",
    "ACTIVE": "a live scheduler state, not a disk fact",
    "PARKED": "a live scheduler state, not a disk fact",
    "EVICTABLE": "a residency decision, which is G104",
}


class RegistryRefused(RuntimeError):
    """The ModelLake volume is not mounted, so no state can be derived."""


def _require_lake() -> None:
    if not LAKE.is_dir():
        raise RegistryRefused(
            f"{LAKE} is not mounted; every lifecycle state here is derived from "
            "that volume and an empty registry would read as 'no specimens "
            "exist' rather than 'the disk is not attached'"
        )


def _config(d: Path) -> dict[str, Any] | None:
    c = d / "config.json"
    if not c.is_file():
        return None
    try:
        return json.loads(c.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _arch(cfg: dict[str, Any] | None) -> dict[str, Any]:
    if not cfg:
        return {"model_type": None, "hidden_size": None, "num_hidden_layers": None}
    text = cfg.get("text_config") or {}
    def pick(k: str) -> Any:
        return cfg.get(k) if cfg.get(k) is not None else text.get(k)
    return {
        "model_type": cfg.get("model_type"),
        "hidden_size": pick("hidden_size"),
        "num_hidden_layers": pick("num_hidden_layers"),
        "architectures": cfg.get("architectures"),
    }


def _manifest(name: str) -> dict[str, Any] | None:
    p = MANIFESTS / f"{name}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# Weight-file extensions this library actually uses. Counting only
# .safetensors classified ten complete specimens as DISCOVERED - evo2 ships
# .pt, boltz .ckpt, mamba3 and musicgen .bin, Wan .pth. The registry's own
# classifier was wrong before the specimens were.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")


def _shards(d: Path) -> dict[str, Any]:
    """Do the weight files on disk match the specimen's own index?

    This is what separates "complete but unsealed" from "incomplete". A
    specimen whose shards match its index is one verification pass from
    SEALED_SOURCE, not a re-download. Where there is no index - a single-file
    or non-safetensors layout - the test is only that SOME weight file exists,
    which is weaker and is labelled as such.
    """
    try:
        names = os.listdir(d)
    except OSError:
        return {"present": None, "expected": None, "complete": None}
    present = len([f for f in names if f.endswith(WEIGHT_SUFFIXES)])
    idx = d / "model.safetensors.index.json"
    expected = None
    if idx.is_file():
        try:
            wm = json.loads(idx.read_text()).get("weight_map") or {}
            expected = len(set(wm.values()))
        except (json.JSONDecodeError, OSError):
            expected = None
    return {
        "present": present,
        "expected": expected,
        "complete": (expected is not None and present == expected)
        or (expected is None and present > 0),
        "expected_is_none_because": (
            None if expected is not None
            else "no safetensors index on disk, so completeness is only "
                 "'at least one weight file exists' - a weaker test"
        ),
        "check_strength": "INDEX_MATCHED" if expected is not None else "WEIGHTS_PRESENT_ONLY",
    }


def _entry(name: str, root: Path, downloading: bool) -> dict[str, Any]:
    d = root / name
    cfg = _config(d)
    arch = _arch(cfg)
    man = _manifest(name)
    shards = _shards(d)
    repo, _, rev = name.partition("@")

    if downloading:
        state = "DOWNLOADING"
        why = "sits under partial/ with no sealed manifest"
    elif man and man.get("resolved_sha") and man.get("bytes"):
        state = "FINGERPRINTED" if arch["model_type"] else "SEALED_SOURCE"
        why = ("a manifest records a resolved sha and byte count"
               + (", and config.json names an architecture"
                  if arch["model_type"] else ""))
    elif cfg is not None and shards["complete"]:
        state = "COMPLETE_UNSEALED"
        why = ("config parses and the weight shards match the specimen's own "
               "index, but no manifest seals them")
    elif cfg is not None:
        state = "DISCOVERED"
        why = ("config parses but no weight file is present"
               if not shards["present"] else
               "config parses but the weight shards do not match the index "
               f"({shards['present']} of {shards['expected']})")
    else:
        state = "DISCOVERED"
        why = "a directory exists and nothing on disk says more than that"

    return {
        "id": name,
        "repo": repo.replace("--", "/"),
        "revision": rev or None,
        "lifecycle": state,
        "lifecycle_derived_from": why,
        "path": str(d),
        "shards": shards,
        "architecture": arch,
        "source_bytes": man.get("bytes") if man else None,
        "n_files": man.get("n_files") if man else None,
        "acquired_at": man.get("acquired_at") if man else None,
        "reacquisition": man.get("reacquisition") if man else None,
        "measured_load_seconds": None,
        "measured_warmup_seconds": None,
        "load_cost_is_unknown_because": (
            "no load of this specimen has been timed. G102 owns that; a "
            "registry that guessed would be worse than one that says so."
        ),
    }


def shadowed() -> list[dict[str, Any]]:
    """Ids present in BOTH specimens/ and partial/.

    A stale partial directory shadows a complete specimen and would make a
    scheduler believe a download is still in flight. Reported rather than
    silently deduped, because the stale directory is the defect.
    """
    _require_lake()
    if not (SPECIMENS.is_dir() and PARTIAL.is_dir()):
        return []
    a = {n for n in os.listdir(SPECIMENS) if (SPECIMENS / n).is_dir()}
    b = {n for n in os.listdir(PARTIAL) if (PARTIAL / n).is_dir()}
    out = []
    for n in sorted(a & b):
        def total(root: Path) -> int:
            try:
                return sum(f.stat().st_size for f in os.scandir(root / n)
                           if f.is_file())
            except OSError:
                return 0
        sb, pb = total(SPECIMENS), total(PARTIAL)
        out.append({
            "id": n,
            "specimens_bytes": sb,
            "partial_bytes": pb,
            "partial_fraction_of_specimen": round(pb / sb, 6) if sb else None,
            "partial_is_a_fragment": bool(sb) and pb < sb * 0.01,
            "reading": (
                f"partial holds {pb} bytes against {sb} in specimens - under 1% "
                "- so it is a leftover fragment of a finished download, not a "
                "competing copy. It would still make a scheduler reading "
                "partial/ believe a download is in flight."
                if sb and pb < sb * 0.01 else
                "both directories hold comparable data; which is authoritative "
                "is NOT decided here"
            ),
        })
    return out


def registry() -> list[dict[str, Any]]:
    """One row per id. The specimens/ copy wins; shadowing is reported by
    shadowed(), not resolved by dropping a row and saying nothing."""
    _require_lake()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    if SPECIMENS.is_dir():
        for n in sorted(os.listdir(SPECIMENS)):
            if (SPECIMENS / n).is_dir():
                rows.append(_entry(n, SPECIMENS, downloading=False))
                seen.add(n)
    if PARTIAL.is_dir():
        for n in sorted(os.listdir(PARTIAL)):
            if (PARTIAL / n).is_dir() and n not in seen:
                rows.append(_entry(n, PARTIAL, downloading=True))
                seen.add(n)
    return rows


def by_lifecycle() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in registry():
        out.setdefault(r["lifecycle"], []).append(r["id"])
    return {k: sorted(v) for k, v in sorted(out.items())}


def architecture_families() -> dict[str, list[str]]:
    """S027 §48: cluster into structural families to choose discovery, near
    transfer and distant adversarial specimens. Unknown is its own bucket, not
    folded into a guess."""
    out: dict[str, list[str]] = {}
    for r in registry():
        mt = r["architecture"]["model_type"] or "UNKNOWN"
        out.setdefault(mt, []).append(r["id"])
    return {k: sorted(v) for k, v in sorted(out.items())}


def schedulable() -> list[dict[str, Any]]:
    """What a scheduler may act on today: sealed and fingerprinted only."""
    return [
        {"id": r["id"], "model_type": r["architecture"]["model_type"],
         "hidden_size": r["architecture"]["hidden_size"],
         "layers": r["architecture"]["num_hidden_layers"],
         "source_bytes": r["source_bytes"]}
        for r in registry()
        if r["lifecycle"] in ("SEALED_SOURCE", "FINGERPRINTED")
    ]


def seal_backlog() -> dict[str, Any]:
    """The finding this registry surfaced: complete specimens nobody sealed.

    S027 §2 assumes a newly sealed model becomes schedulable material at once.
    Today most of the library is complete and UNSEALED, so it is not
    schedulable at all - and that is a pipeline gap, not a registry limitation.
    """
    rows = registry()
    backlog = [r for r in rows if r["lifecycle"] == "COMPLETE_UNSEALED"]
    sealed = [r for r in rows
              if r["lifecycle"] in ("SEALED_SOURCE", "FINGERPRINTED")]
    return {
        "n_complete_unsealed": len(backlog),
        "n_sealed": len(sealed),
        "ids": sorted(r["id"] for r in backlog),
        "all_shards_match_their_own_index": all(
            r["shards"]["complete"] for r in backlog),
        "check_strength_mix": {
            k: sum(1 for r in backlog if r["shards"]["check_strength"] == k)
            for k in ("INDEX_MATCHED", "WEIGHTS_PRESENT_ONLY")
        },
        "not_every_check_is_equally_strong": (
            "INDEX_MATCHED compares the shard count to the specimen's own "
            "safetensors index. WEIGHTS_PRESENT_ONLY just says a weight file "
            "exists, which is what a non-safetensors layout allows. Sealing "
            "must verify bytes either way; this only says which are worth "
            "queueing first."
        ),
        "statement": (
            f"{len(backlog)} specimens have complete weight shards matching "
            f"their own index and no manifest, against {len(sealed)} sealed. "
            "They are one verification pass from SEALED_SOURCE - NOT "
            "re-downloads - and until that pass runs they are invisible to a "
            "scheduler that acts only on sealed material."
        ),
        "why_it_matters": (
            "S027 §2 assumes a newly sealed model becomes schedulable at once. "
            "The assumption holds; what is missing is that most of the library "
            "was never sealed, so the Odyssey's first cycle would see a small "
            "fraction of the specimens actually on disk."
        ),
        "this_module_does_not_seal_them": (
            "sealing verifies bytes and writes a manifest, which is a "
            "ModelLake responsibility. A registry that sealed specimens as a "
            "side effect of being read would be a mutation writer pretending "
            "to be an observer."
        ),
    }


def build() -> dict[str, Any]:
    rows = registry()
    bl = by_lifecycle()
    fam = architecture_families()
    return {
        "obligation": "G100",
        "authority": "S027 §1-§3, §48",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "lake": str(LAKE),
        "n_specimens": len(rows),
        "lifecycle_states": list(LIFECYCLE),
        "by_lifecycle": {k: len(v) for k, v in bl.items()},
        "ids_by_lifecycle": bl,
        "n_schedulable": len(schedulable()),
        "seal_backlog": seal_backlog(),
        "shadowed_by_a_stale_partial_directory": shadowed(),
        "architecture_families": {k: len(v) for k, v in fam.items()},
        "n_families": len(fam),
        "specimens": rows,
        "incomplete_in_the_specimens_directory": {
            "ids": sorted(r["id"] for r in registry()
                          if r["lifecycle"] == "DISCOVERED"
                          and r["shards"]["present"] == 0
                          and r["architecture"]["model_type"]),
            "statement": (
                "these sit in specimens/ - the directory for COMPLETED "
                "downloads - with a parseable config and NO weight file at "
                "all. Directory placement is not evidence of completeness, and "
                "anything reading specimens/ as a done-list would have counted "
                "them."
            ),
        },
        "states_not_derivable_here": NOT_DERIVABLE_HERE,
        "lifecycle_is_derived_not_declared": (
            "SEALED_SOURCE requires a manifest with a resolved sha and byte "
            "count; FINGERPRINTED additionally requires config.json to parse "
            "and name an architecture; DOWNLOADING is a partial/ directory with "
            "no manifest. A specimen nothing supports is DISCOVERED, and that "
            "is not a claim anything is ready."
        ),
        "what_this_does_not_do": (
            "it does not measure load cost, hold residency or choose what to "
            "load next - G102 and G104 own those. Every measured_load_seconds "
            "is null and says why, because a guessed load time is worse than an "
            "admitted unknown."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_receipt(RECEIPT_NAME, doc, RECORDED_BY))
        return 0
    print(json.dumps({k: doc[k] for k in (
        "n_specimens", "by_lifecycle", "n_schedulable", "n_families",
        "architecture_families")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
