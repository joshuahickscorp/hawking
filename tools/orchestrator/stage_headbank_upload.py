#!/usr/bin/env python3
"""stage_headbank_upload — organize locally-prepared Eagle5 inputs into a
single Drive-upload bundle the corrected-headbank Colab expects.

Produces `_headbank_upload/<slug>/`:
  corpus_shards/shard_*.parquet   (from _capture/<slug>_corpus_shards)
  frozen_gguf.npz                 (from frozen/<slug>_frozen_gguf.npz)

Upload the whole `_headbank_upload/` folder to Drive and point the notebook's
DRIVE_ROOT at it. Only the corpus (quantized residuals) truly must come from
the local Metal runtime; the frozen npz is included so the Colab needs no GGUF.

Usage:
    python3 tools/orchestrator/stage_headbank_upload.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLUGS = ["q05b", "q1p5b", "q3b", "q7b"]


def main() -> int:
    out_root = ROOT / "_headbank_upload"
    out_root.mkdir(exist_ok=True)
    staged = []
    for slug in SLUGS:
        shards = ROOT / "_capture" / f"{slug}_corpus_shards"
        frozen = ROOT / "frozen" / f"{slug}_frozen_gguf.npz"
        if not shards.is_dir() or not any(shards.glob("shard_*.parquet")):
            print(f"{slug:6s} SKIP — no corpus shards at {shards}")
            continue
        if not frozen.is_file():
            print(f"{slug:6s} SKIP — no frozen npz at {frozen}")
            continue
        dst = out_root / slug
        (dst / "corpus_shards").mkdir(parents=True, exist_ok=True)
        for p in shards.glob("shard_*.parquet"):
            shutil.copy2(p, dst / "corpus_shards" / p.name)
        shutil.copy2(frozen, dst / "frozen_gguf.npz")
        nshards = len(list((dst / "corpus_shards").glob("*.parquet")))
        print(f"{slug:6s} staged {nshards} shards + frozen_gguf.npz")
        staged.append(slug)
    print(f"\nstaged {len(staged)} models -> {out_root}")
    print("Upload this folder to Drive; set the notebook DRIVE_ROOT to it.")
    return 0 if staged else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
