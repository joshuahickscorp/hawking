#!/usr/bin/env python3.12
"""hawking.deepseek_v4_flash.source.v1 - admit and fetch the next parent.

DeepSeek-V4-Flash is the contraction-first primary: the smallest frontier MoE on
the ladder, the same deepseek_v4 family as the 864 GB F7, and native fp4-in-INT8
with E8M0 scales rather than BF16. That last fact matters for the pilot: the
routed experts are already four bits, so the sub-bit question here is a ~4x
reduction, not the ~16x a BF16 parent implies.

Three commands:

    admit   fetch config.json, the safetensors index, and the immutable HF blobs
            manifest live (metadata only). Seal an admission + rehydration
            receipt, and check the complete source plus a 100 GiB operational
            reserve fit the free disk.
    fetch   download one complete verified physical copy at the pinned revision
            into the support root, then verify every shard byte-for-byte against
            the blobs manifest sha256. Refuses if admission says it will not fit.
    status  print what is resident and verified.

The heavy shards live outside the repository and outside MOP, under the Hawking
application-support tree, exactly as the GLM body did.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

REPO = "deepseek-ai/DeepSeek-V4-Flash"
REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
GB = 10 ** 9
GIB = 1024 ** 3
OPERATIONAL_RESERVE_BYTES = 100 * GIB

CONDENSE = Path(__file__).resolve().parent
REPO_ROOT = CONDENSE.parents[1]
SUPPORT = Path(os.environ.get(
    "DEEPSEEK_V4_SUPPORT_ROOT",
    str(Path.home() / "Library/Application Support/Hawking/DeepSeekV4Flash")))
SOURCE = SUPPORT / "source"
META = SUPPORT / "meta"

ADMISSION = REPO_ROOT / "DEEPSEEK_V4_FLASH_SOURCE_ADMISSION.json"
REHYDRATION = REPO_ROOT / "DEEPSEEK_V4_FLASH_REHYDRATION_RECEIPT.json"
FETCH_RECEIPT = REPO_ROOT / "DEEPSEEK_V4_FLASH_FETCH_RECEIPT.json"

API = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}"
BLOBS_API = API + "?blobs=true"
RESOLVE = f"https://huggingface.co/{REPO}/resolve/{REVISION}"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _get(url: str, *, binary: bool = False):
    request = urllib.request.Request(url, headers={"User-Agent": "hawking-admit"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read() if binary else json.loads(response.read())


def _seal(document: dict) -> dict:
    document["seal_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in document.items() if k != "seal_sha256"},
                   sort_keys=True).encode()).hexdigest()
    return document


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def admit() -> dict:
    """Resolve the source live and decide whether a complete copy fits."""
    META.mkdir(parents=True, exist_ok=True)
    manifest = _get(BLOBS_API)
    if manifest.get("sha") != REVISION:
        raise SystemExit(f"live sha {manifest.get('sha')} != pinned {REVISION}")
    if manifest.get("gated") or manifest.get("private"):
        raise SystemExit("repo is gated or private; not admissible unattended")

    config = _get(f"{RESOLVE}/config.json")
    (META / "config.json").write_bytes(json.dumps(config, indent=2).encode())
    index = _get(f"{RESOLVE}/model.safetensors.index.json")
    (META / "model.safetensors.index.json").write_bytes(json.dumps(index).encode())

    lfs, total = {}, 0
    for entry in manifest.get("siblings", []):
        name = entry.get("rfilename")
        blob = entry.get("lfs") or {}
        size = blob.get("size") or entry.get("size")
        sha = blob.get("sha256") or blob.get("oid")
        if name and name.endswith(".safetensors"):
            lfs[name] = {"sha256": sha, "size": size}
            total += int(size or 0)

    index_total = int(index.get("metadata", {}).get("total_size", 0))
    free = shutil.disk_usage(str(SUPPORT if SUPPORT.exists() else SUPPORT.parent)).free
    fits = free - total >= OPERATIONAL_RESERVE_BYTES

    # Dtype breakdown from the safetensors parameter-type totals HF reports.
    dtypes = (manifest.get("safetensors") or {}).get("parameters", {})

    admission = _seal({
        "schema": "hawking.deepseek_v4_flash.source_admission.v1",
        "admitted_at": _now(),
        "repo": REPO,
        "revision": REVISION,
        "immutable_tree_url": f"https://huggingface.co/{REPO}/tree/{REVISION}",
        "license": manifest.get("cardData", {}).get("license") or "mit",
        "gated": bool(manifest.get("gated")),
        "private": bool(manifest.get("private")),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "n_routed_experts": config.get("n_routed_experts"),
        "num_experts_per_tok": config.get("num_experts_per_tok"),
        "n_shared_experts": config.get("n_shared_experts"),
        "index_topk": config.get("index_topk"),
        "quantization_config": config.get("quantization_config"),
        "source_precision": {
            "dtype_parameter_counts": dtypes,
            "note": "native fp4-in-I8 packed experts with F8_E8M0 (ue8m0) scales; "
                    "NOT BF16. Sub-bit here is a ~4x reduction, not ~16x.",
        },
        "safetensors_files": len(lfs),
        "source_bytes_from_blobs": total,
        "source_gb_from_blobs": round(total / GB, 1),
        "index_total_size": index_total,
        "free_bytes": free,
        "free_gb": round(free / GB, 1),
        "operational_reserve_bytes": OPERATIONAL_RESERVE_BYTES,
        "operational_reserve_gib": 100,
        "fits_with_reserve": bool(fits),
        "headroom_after_source_and_reserve_gb": round(
            (free - total - OPERATIONAL_RESERVE_BYTES) / GB, 1),
        "download_decision": "DOWNLOAD_COMPLETE_VERIFIED_COPY" if fits
        else "PARTIAL_WINDOWS_ONLY",
    })
    ADMISSION.write_text(json.dumps(admission, indent=2, sort_keys=True))

    rehydration = _seal({
        "schema": "hawking.deepseek_v4_flash.rehydration_receipt.v1",
        "sealed_at": _now(),
        "repo": REPO,
        "revision": REVISION,
        "route": "huggingface content-addressable git-lfs at the pinned revision",
        "files": len(lfs),
        "per_file_sha256": {name: lfs[name] for name in sorted(lfs)},
    })
    REHYDRATION.write_text(json.dumps(rehydration, indent=2, sort_keys=True))
    return {"fits": fits, "source_gb": round(total / GB, 1),
            "free_gb": round(free / GB, 1),
            "headroom_gb": admission["headroom_after_source_and_reserve_gb"],
            "decision": admission["download_decision"], "files": len(lfs)}


def fetch() -> dict:
    if not ADMISSION.exists():
        raise SystemExit("run `admit` first")
    admission = json.loads(ADMISSION.read_text())
    if not admission.get("fits_with_reserve"):
        raise SystemExit("admission says a complete copy does not fit; partial only")

    from huggingface_hub import snapshot_download
    SOURCE.mkdir(parents=True, exist_ok=True)
    started = time.time()
    snapshot_download(
        repo_id=REPO, revision=REVISION, local_dir=str(SOURCE),
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "tokenizer*"],
        max_workers=8)
    elapsed = time.time() - started

    rehydration = json.loads(REHYDRATION.read_text())["per_file_sha256"]
    verified, mismatched, missing = [], [], []
    for name, meta in sorted(rehydration.items()):
        path = SOURCE / name
        if not path.exists():
            missing.append(name)
            continue
        got = sha256_file(path)
        (verified if got == meta["sha256"] else mismatched).append(name)

    resident = sorted(p.name for p in SOURCE.glob("*.safetensors"))
    total = sum(p.stat().st_size for p in SOURCE.rglob("*") if p.is_file())
    receipt = _seal({
        "schema": "hawking.deepseek_v4_flash.fetch_receipt.v1",
        "fetched_at": _now(),
        "repo": REPO,
        "revision": REVISION,
        "source_root": str(SOURCE),
        "resident_safetensors": len(resident),
        "resident_bytes": total,
        "resident_gb": round(total / GB, 1),
        "verified_against_blobs_sha256": len(verified),
        "mismatched": mismatched,
        "missing": missing,
        "complete_and_verified": not mismatched and not missing
        and len(verified) == len(rehydration),
        "download_seconds": round(elapsed, 1),
        "free_after_gb": round(shutil.disk_usage(str(SUPPORT)).free / GB, 1),
    })
    FETCH_RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return {"verified": len(verified), "mismatched": len(mismatched),
            "missing": len(missing),
            "complete": receipt["complete_and_verified"],
            "resident_gb": receipt["resident_gb"]}


def status() -> dict:
    resident = sorted(p.name for p in SOURCE.glob("*.safetensors")) if SOURCE.exists() else []
    total = sum(p.stat().st_size for p in SOURCE.rglob("*") if p.is_file()) \
        if SOURCE.exists() else 0
    return {
        "source_root": str(SOURCE),
        "resident_safetensors": len(resident),
        "resident_gb": round(total / GB, 1),
        "admission": ADMISSION.exists(),
        "fetch_receipt": json.loads(FETCH_RECEIPT.read_text())
        if FETCH_RECEIPT.exists() else None,
        "free_gb": round(shutil.disk_usage(str(SUPPORT if SUPPORT.exists()
                                              else SUPPORT.parent)).free / GB, 1),
    }


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "admit":
        print(json.dumps(admit(), indent=2))
    elif command == "fetch":
        print(json.dumps(fetch(), indent=2))
    elif command == "status":
        print(json.dumps(status(), indent=2))
    else:
        raise SystemExit(f"unknown command: {command}")
