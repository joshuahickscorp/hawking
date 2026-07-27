#!/usr/bin/env python3.12
"""Re-pack shard 00001 with lm_head/embed_tokens native instead of gravity-pq.

Cause found by reading data already on disk, no re-fetch needed for the diagnosis: the
packer's own per-tensor `relative_frobenius_error` field (stored for every tensor's
production rung, whether or not that tensor's own descriptor survives) shows lm_head at
0.965 and embed_tokens at 0.823 -- both classified `COMPRESSIBLE_CANDIDATE` and PQ-packed
at R0 like everything else, both reconstructing far worse than any other category
(attention/dense_mlp/shared_expert/routed_expert all cluster at 0.62-0.65). An embedding
table's rows are 154,880 largely-distinct semantic directions; a shared 128-entry
codebook over 8-wide subvectors cannot represent that the way it represents the more
redundant structure of an MLP or expert weight matrix.

This is exactly what Prometheus's general-v1 and math-v1 profiles already specify
(embedding/lm_head at tier T0, native) and what the uniform production packer's
budget-class heuristic did not apply. So the fix is not "spend more bits everywhere" --
it is "protect the two tensors the data already says are badly reconstructed," at a cost
of about 3.3 GB on an 83 GB model (whole-model BPW moves from 0.883 to about 0.918,
nowhere near the H10 ceiling that a uniform re-pack would require).

Both tensors live in one shard, so this needs one re-fetched file (5.3 GB), not 282.

    python3.12 tools/condense/glm52_protect_head_embed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gravity_format as gravity  # noqa: E402

REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
SHARD_NAME = "model-00001-of-00282.safetensors"
GRAVITY_NAME = "model-00001-of-00282.gravity"
NATIVE_PROTECT = {"lm_head.weight", "model.embed_tokens.weight"}

MODEL_DIR = (
    Path.home() / "Library/Application Support/Hawking/Models/GLM-5.2"
    f"/{REVISION}/General-R0")


def _read_safetensors_index(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def _bf16_bytes_for(safet_path: Path, header: dict, base: int, name: str) -> bytes:
    meta = header[name]
    start, end = (int(v) for v in meta["data_offsets"])
    with open(safet_path, "rb") as f:
        f.seek(base + start)
        return f.read(end - start)


def main() -> int:
    from huggingface_hub import hf_hub_download

    safet_path = Path(
        hf_hub_download("zai-org/GLM-5.2", SHARD_NAME, revision=REVISION)
    )
    header, base = _read_safetensors_index(safet_path)

    gravity_path = MODEL_DIR / GRAVITY_NAME
    old_header = gravity.read_header(gravity_path)
    old_tensors = {t["name"]: t for t in old_header["tensors"]}
    for name in NATIVE_PROTECT:
        if name not in old_tensors:
            raise SystemExit(f"{name} not found in {gravity_path}")
        if name not in header:
            raise SystemExit(f"{name} not found in re-fetched {SHARD_NAME}")

    payloads: list[tuple[dict, bytes]] = []
    before_bytes = after_bytes = 0
    for name in sorted(old_tensors):
        old = old_tensors[name]
        before_bytes += old["bytes"]
        if name in NATIVE_PROTECT:
            blob = _bf16_bytes_for(safet_path, header, base, name)
            expected_elements = 1
            for d in old["shape"]:
                expected_elements *= int(d)
            if len(blob) != expected_elements * 2:
                raise SystemExit(
                    f"{name}: refetched bf16 blob is {len(blob)} bytes, "
                    f"expected {expected_elements * 2} for shape {old['shape']}"
                )
            descriptor = {
                "name": name,
                "category": old["category"],
                "layer": old.get("layer"),
                "expert": old.get("expert"),
                "shape": old["shape"],
                "codec": "native.bf16",
                "terminal_state": "PROTECTED_SOURCE_NATIVE",
                "elements": expected_elements,
                "bpw": len(blob) * 8 / expected_elements,
                "reason": "GLM52_HEAD_EMBED_RECONSTRUCTION_FAILURE: relative_frobenius_error "
                          "0.965/0.823 at R0 vs 0.62-0.65 for every other category; "
                          "reclassified from COMPRESSIBLE_CANDIDATE, matching Prometheus "
                          "general-v1/math-v1 T0 tier for embedding/lm_head",
            }
        else:
            blob = gravity.read_tensor(gravity_path, name, verify_hash=True)
            descriptor = {k: v for k, v in old.items()
                          if k not in ("offset", "bytes", "sha256")}
        after_bytes += len(blob)
        payloads.append((descriptor, blob))

    pq_bits = sum(len(b) * 8 for d, b in payloads if d["codec"] == "gravity-pq")
    native_bits = sum(len(b) * 8 for d, b in payloads if d["codec"].startswith("native."))
    pq_weights = sum(d["elements"] for d, b in payloads if d["codec"] == "gravity-pq")
    native_weights = sum(d["elements"] for d, b in payloads if d["codec"].startswith("native."))
    total_weights = pq_weights + native_weights

    tmp_path = gravity_path.with_name(gravity_path.name + ".protected")
    gravity.write_shard(
        tmp_path, payloads,
        model=old_header["model"],
        architecture=old_header["architecture"],
        tokenizer=old_header.get("tokenizer"),
        compression={
            **old_header["compression"],
            "complete_bpw": (pq_bits + native_bits) / max(1, total_weights),
            "packed_bpw": pq_bits / max(1, pq_weights) if pq_weights else 0.0,
            "native_bytes": native_bits // 8,
            "native_tensors": sum(1 for d, _ in payloads if d["codec"].startswith("native.")),
            "compressed_tensors": sum(1 for d, _ in payloads if d["codec"] == "gravity-pq"),
            "note": "lm_head/embed_tokens reclassified PROTECTED_SOURCE_NATIVE after "
                    "R0 reconstruction failure; see tools/condense/glm52_protect_head_embed.py",
        },
        shard=old_header["shard"],
    )

    backup = gravity_path.with_name(gravity_path.name + ".pre_protect_backup")
    gravity_path.rename(backup)
    tmp_path.rename(gravity_path)

    result = {
        "shard": GRAVITY_NAME,
        "protected": sorted(NATIVE_PROTECT),
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "delta_bytes": after_bytes - before_bytes,
        "backup": str(backup),
    }
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
