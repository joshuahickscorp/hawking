#!/usr/bin/env python3
"""Export only the essential Qwen reconciliation artifacts.

Intended for Colab after the q3b/q1p5 reconciliation runs. It copies final
deployable heads plus small JSON/quantization artifacts into a clean export
folder and optionally writes a zip next to it.

Example:
  python colab/export_reconciliation_essentials.py \
      --drive-root /content/drive/MyDrive/dismantle \
      --export-dir /content/drive/MyDrive/dismantle_export \
      --zip
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import zipfile
from pathlib import Path


def copy_if_exists(src: Path, dst: Path) -> dict | None:
    if not src.exists() or not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return {
        "source": str(src),
        "exported": str(dst),
        "bytes": int(dst.stat().st_size),
    }


def add_artifact(
    artifacts: list[tuple[str, Path, Path]],
    name: str,
    root: Path,
    rel_src: str,
    rel_dst: str,
) -> None:
    artifacts.append((name, root / rel_src, Path(rel_dst)))


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(src_dir))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive/dismantle"),
        help="Root folder containing qwen_reconciliation and corpus folders.",
    )
    p.add_argument(
        "--export-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/dismantle_export"),
        help="Clean destination folder for essential artifacts.",
    )
    p.add_argument("--zip", action="store_true", help="Also create .zip archive.")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if either final q3b or q1p5 head is missing.",
    )
    args = p.parse_args()

    root = args.drive_root
    export_dir = args.export_dir
    rec = root / "qwen_reconciliation"
    ckpt = rec / "checkpoints"

    artifacts: list[tuple[str, Path, Path]] = []

    # Final deployable heads.
    add_artifact(
        artifacts,
        "q3b_head",
        root,
        "qwen_reconciliation/checkpoints/"
        "q3b_e6_b1_wide_b1_h16_ff60_lr5e-4_rd020_cw20_long/"
        "head_final.safetensors",
        "heads/q3b_eagle6_long.safetensors",
    )
    add_artifact(
        artifacts,
        "q1p5_head",
        root,
        "qwen_reconciliation/checkpoints/"
        "q1p5_e6_b2_wide_b2_h16_ff60_lr5e-4_rd020_cw20_long/"
        "head_final.safetensors",
        "heads/q1p5_eagle6_long.safetensors",
    )

    # Small evaluation outputs, if present.
    for target, folder in (
        ("q3b", "q3b_e6_b1_wide_b1_h16_ff60_lr5e-4_rd020_cw20_long"),
        ("q1p5", "q1p5_e6_b2_wide_b2_h16_ff60_lr5e-4_rd020_cw20_long"),
    ):
        for filename in ("tau.json", "frontier.json", "log.jsonl"):
            add_artifact(
                artifacts,
                f"{target}_{filename}",
                root,
                f"qwen_reconciliation/checkpoints/{folder}/{filename}",
                f"eval/{target}_{filename}",
            )

    # Quantization/calibration inventory. These are tiny compared with corpus
    # shards and useful for local follow-up work.
    for filename in (
        "qwen3b_awq.json",
        "qwen3b_awq_per_channel.json",
        "qwen3b_q2_importance.npz",
        "qwen1p5_awq.json",
        "qwen1p5_awq_per_channel.json",
        "qwen1p5_q2_importance.npz",
        "q1p5_resume_summary.json",
        "reconciliation_frontier_winners.json",
        "reconciliation_simulation.json",
        "reconciliation_summary.md",
    ):
        add_artifact(
            artifacts,
            filename,
            root,
            f"qwen_reconciliation/{filename}",
            f"metadata/{filename}",
        )

    manifest = {
        "schema": "dismantle-qwen-reconciliation-export-v1",
        "created_at_unix": int(time.time()),
        "drive_root": str(root),
        "export_dir": str(export_dir),
        "files": {},
        "missing": {},
    }

    export_dir.mkdir(parents=True, exist_ok=True)
    for name, src, rel_dst in artifacts:
        copied = copy_if_exists(src, export_dir / rel_dst)
        if copied is None:
            manifest["missing"][name] = str(src)
        else:
            manifest["files"][name] = copied
            print("copied", f"{copied['bytes'] / 1e9:.2f} GB", rel_dst)

    required = ("q3b_head", "q1p5_head")
    missing_required = [name for name in required if name in manifest["missing"]]
    if args.strict and missing_required:
        raise SystemExit(f"missing required final heads: {missing_required}")

    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest_path}")

    if args.zip:
        zip_path = export_dir.with_suffix(".zip")
        zip_dir(export_dir, zip_path)
        print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")

    total = sum(item["bytes"] for item in manifest["files"].values())
    print(
        f"exported {len(manifest['files'])} file(s), "
        f"{total / 1e9:.2f} GB; missing optional={len(manifest['missing'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
