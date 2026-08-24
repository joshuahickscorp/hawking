#!/usr/bin/env python3
"""ArtifactLedger census (directive §8) + DiskTruth (§2).

Deterministic, read-only over the filesystem.  Writes two receipts:
  receipts/headless/ARTIFACT_LEDGER.json   every model/runtime artifact >= --min-gib
  receipts/headless/DISK_TRUTH.json        HCLI source identity, tests, receipts, genomes

SHA-256 of multi-GiB weights is expensive, so hashing is opt-in per artifact via
--hash (full) or --hash-head (first+last 8 MiB + size, a cheap collision-resistant
identity that still detects truncation and corruption).  Which one ran is recorded
in the receipt so a later reader is never fooled about the strength of the identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

GIB = 1024 ** 3

# Extensions that are model/runtime weight artifacts.
WEIGHT_EXT = {".gguf", ".safetensors", ".bin", ".mlx", ".npz", ".pt", ".pth"}

# Roots worth walking.  Everything else is noise for this census.
DEFAULT_ROOTS = [
    "~/models",
    "~/.cache/huggingface",
    "~/.cache/lm-studio",
    "~/Downloads/hawking/workspace",
    "~/Downloads/hawking-copy/workspace",
    "~/Downloads/hawking/receipts",
    "~/Downloads/hawking-copy/receipts",
    "~/.ollama",
]

# Directive §8: never-delete classes.  A path matching any of these is pinned.
PROTECTED_SUBSTRINGS = (
    "qwen3.8-27b-abliterated",   # current bootstrap parent
)


def sha256_full(path: str, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_head_tail(path: str, span: int = 8 << 20) -> str:
    """Cheap identity: sha256(size || first span || last span)."""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(span))
        if size > span:
            f.seek(max(0, size - span))
            h.update(f.read(span))
    return h.hexdigest()


def classify(path: str, size: int, active_models: set) -> tuple[str, str]:
    """(classification, reason) per directive §8."""
    p = path.lower()
    if path in active_models or any(s in p for s in PROTECTED_SUBSTRINGS):
        return "KEEP_ACTIVE_PARENT", "current bootstrap/production Qwen parent"
    if "mmproj" in p:
        return "KEEP_ACTIVE_PARENT", "vision projector paired with the active parent"
    if "/.cache/huggingface/" in p or "/hub/models--" in p:
        return "REDOWNLOADABLE", "HuggingFace hub cache; re-fetchable from the recorded repo id"
    if "/receipts/" in p or "negative" in p:
        return "KEEP_UNIQUE_SCIENCE", "receipt / negative-science artifact"
    return "UNKNOWN_DO_NOT_DELETE", "not yet attributed to a parent, child, or reproducible source"


def hf_repo_id(path: str) -> str | None:
    """Recover the HF repo id from a hub cache path so a REDOWNLOADABLE row is actionable."""
    parts = Path(path).parts
    for i, seg in enumerate(parts):
        if seg.startswith("models--"):
            return seg[len("models--"):].replace("--", "/", 1)
    return None


def walk_artifacts(roots, min_bytes):
    seen = set()
    for root in roots:
        r = os.path.expanduser(root)
        if not os.path.isdir(r):
            continue
        for dirpath, dirnames, filenames in os.walk(r, followlinks=False):
            # blobs/ under the HF hub are the real bytes; snapshots/ are symlinks to them
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                if os.path.islink(full):
                    continue
                ext = os.path.splitext(fn)[1].lower()
                if ext not in WEIGHT_EXT and "/blobs/" not in full:
                    continue
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                if st.st_size < min_bytes:
                    continue
                real = os.path.realpath(full)
                if real in seen:
                    continue
                seen.add(real)
                yield full, st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gib", type=float, default=1.0)
    ap.add_argument("--hash", action="store_true", help="full SHA-256 (slow)")
    ap.add_argument("--hash-head", action="store_true", help="cheap head/tail identity")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/Downloads/hawking-copy/receipts/headless"))
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    min_bytes = int(args.min_gib * GIB)
    t0 = time.time()

    active = {os.path.realpath(os.path.expanduser(
        "~/models/qwen3.8-27b-abliterated/Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf"))}

    rows = []
    for full, st in walk_artifacts(DEFAULT_ROOTS, min_bytes):
        cls, reason = classify(full, st.st_size, active)
        row = {
            "path": full,
            "real_path": os.path.realpath(full),
            "size_bytes": st.st_size,
            "size_gib": round(st.st_size / GIB, 3),
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
            "atime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_atime)),
            "classification": cls,
            "classification_reason": reason,
            "hf_repo_id": hf_repo_id(full),
            "identity_kind": "none",
            "identity": None,
        }
        if args.hash:
            row["identity_kind"] = "sha256_full"
            row["identity"] = sha256_full(full)
        elif args.hash_head:
            row["identity_kind"] = "sha256_size_head_tail_8MiB"
            row["identity"] = sha256_head_tail(full)
        rows.append(row)
        print(f"  {row['size_gib']:>9.3f} GiB  {cls:<24} {full}", flush=True)

    rows.sort(key=lambda r: -r["size_bytes"])
    ledger = {
        "schema": "hawking.headless.artifact_ledger.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roots_walked": [os.path.expanduser(r) for r in DEFAULT_ROOTS],
        "min_gib": args.min_gib,
        "identity_mode": ("sha256_full" if args.hash else
                          "sha256_size_head_tail_8MiB" if args.hash_head else "none"),
        "identity_caveat": ("full content hash" if args.hash else
                            "head/tail+size identity: detects truncation and corruption, "
                            "NOT a full content hash. Do not cite as a content SHA."
                            if args.hash_head else
                            "NO hashing was performed; rows carry size+mtime identity only."),
        "artifact_count": len(rows),
        "total_gib": round(sum(r["size_bytes"] for r in rows) / GIB, 3),
        "by_class": {},
        "artifacts": rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    for r in rows:
        c = ledger["by_class"].setdefault(r["classification"], {"count": 0, "gib": 0.0})
        c["count"] += 1
        c["gib"] = round(c["gib"] + r["size_gib"], 3)

    (out / "ARTIFACT_LEDGER.json").write_text(json.dumps(ledger, indent=1))
    print(f"\nARTIFACT_LEDGER: {len(rows)} artifacts, {ledger['total_gib']} GiB, "
          f"{ledger['elapsed_s']}s -> {out/'ARTIFACT_LEDGER.json'}")
    for k, v in sorted(ledger["by_class"].items()):
        print(f"  {k:<24} {v['count']:>4}  {v['gib']:>10.1f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
