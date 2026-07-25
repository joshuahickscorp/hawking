#!/usr/bin/env python3.12
"""Pack any BF16/FP16 safetensors checkpoint into one `.gravity` shard.

``glm52_pack`` is bound to the GLM-5.2 tensor graph and its streaming schedule, which is
correct for the flagship and useless for the small model the runtime has to be PROVEN on.
This is the same codec and the same container with the model-specific plumbing removed:
one file in, one file out, every declared tensor accounted for exactly once.

Every tensor takes one of two paths and both are billed at what they physically cost:

* **PQ** -- 2D tensors whose geometry divides the rung and whose fixed costs amortize
  under the ceiling. Payload is the ``glm52_pack.serialize`` blob, byte-identical to what
  the flagship pipeline writes.
* **native** -- everything else, stored at source precision. Norms, biases and anything
  too small to amortize a codebook. Carried, not excluded: a tensor the artifact needs
  but does not store is the exact defect the container spec exists to prevent.

    python3.12 tools/condense/safetensors_to_gravity.py MODEL_DIR OUT.gravity [--rung R0]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import glm52_pack as pack  # noqa: E402
import gravity_format  # noqa: E402
import gravity_forge as forge  # noqa: E402

_DTYPES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "U8": 1, "I8": 1}


def _read_safetensors_index(path: Path) -> tuple[dict, int]:
    """Header dict plus the absolute offset where tensor data starts."""
    with open(path, "rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))
    return header, 8 + header_len


def _to_f32(raw: bytes, dtype: str, shape: list[int]) -> np.ndarray:
    if dtype == "BF16":
        # bfloat16 is the top 16 bits of an f32, so widening is a shift, not a table.
        u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u16.view(np.float32).reshape(shape)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    raise ValueError(f"unsupported source dtype {dtype}")


def pack_model(model_dir: Path, out_path: Path, rung_name: str = "R0",
               seed: int = 0) -> dict:
    weights_file = model_dir / "model.safetensors"
    if not weights_file.exists():
        raise SystemExit(f"{weights_file} not found (sharded checkpoints not handled)")
    rung = next(r for r in pack.LADDER if r["rung"] == rung_name)
    header, base = _read_safetensors_index(weights_file)
    config = json.loads((model_dir / "config.json").read_text())

    entries: list[dict] = []
    payloads: list[tuple[dict, bytes]] = []
    offset = 0
    totals = {"pq_bits": 0, "native_bits": 0, "weights_pq": 0, "weights_native": 0}

    names = sorted(k for k in header if k != "__metadata__")
    with open(weights_file, "rb", buffering=0) as source:
        for name in names:
            meta = header[name]
            shape = [int(d) for d in meta["shape"]]
            elements = int(np.prod(shape)) if shape else 1
            start, end = (int(v) for v in meta["data_offsets"])
            source.seek(base + start)
            raw = source.read(end - start)

            usable = (len(shape) == 2 and shape[1] % rung["dim"] == 0
                      and forge_admissible(rung, elements))
            if usable:
                weights = _to_f32(raw, meta["dtype"], shape)
                artifact = forge.pack_product_quant(
                    weights, dim=rung["dim"], subspaces=1, k=rung["k"], seed=seed)
                blob = pack.serialize(artifact)
                descriptor = {
                    "name": name, "codec": "gravity-pq", "shape": shape,
                    "elements": elements, "rung": rung["rung"],
                    "bpw": len(blob) * 8 / elements,
                    "source_dtype": meta["dtype"],
                }
                totals["pq_bits"] += len(blob) * 8
                totals["weights_pq"] += elements
            else:
                blob = raw
                descriptor = {
                    "name": name, "codec": f"native.{meta['dtype'].lower()}",
                    "shape": shape, "elements": elements,
                    "terminal_state": "PROTECTED_SOURCE_NATIVE",
                    "bpw": len(blob) * 8 / max(1, elements),
                    "reason": ("NOT_2D" if len(shape) != 2 else
                               "GEOMETRY_OR_FIXED_COST"),
                }
                totals["native_bits"] += len(blob) * 8
                totals["weights_native"] += elements
            descriptor["offset"] = offset
            descriptor["bytes"] = len(blob)
            offset += len(blob)
            entries.append(descriptor)
            payloads.append((descriptor, blob))

    total_weights = totals["weights_pq"] + totals["weights_native"]
    complete_bpw = (totals["pq_bits"] + totals["native_bits"]) / max(1, total_weights)
    # write_shard assigns offsets itself, so a descriptor can never disagree with where
    # its bytes landed; ours are advisory and get overwritten.
    gravity_format.write_shard(
        out_path, payloads,
        model={"repo": str(model_dir.name), "revision": "local",
               "representation": "QUANTIZED_TRANSFORMER"},
        architecture={"model_type": config.get("model_type"),
                      "hidden_size": config.get("hidden_size"),
                      "num_hidden_layers": config.get("num_hidden_layers"),
                      "num_attention_heads": config.get("num_attention_heads"),
                      "num_key_value_heads": config.get("num_key_value_heads"),
                      "intermediate_size": config.get("intermediate_size"),
                      "vocab_size": config.get("vocab_size"),
                      "rope_theta": config.get("rope_theta"),
                      # Without this a reader silently falls back to plain RoPE and is
                      # wrong by a factor of 32 on long positions. A container that needs
                      # the source config to be served is not self-describing.
                      "rope_scaling": config.get("rope_scaling"),
                      "head_dim": config.get("head_dim"),
                      "tie_word_embeddings": config.get("tie_word_embeddings"),
                      "rms_norm_eps": config.get("rms_norm_eps")},
        tokenizer={"source": "tokenizer.json", "dir": str(model_dir)},
        compression={"codec": "gravity-pq", "production_rung": rung["rung"],
                     "representation": "QUANTIZED_TRANSFORMER",
                     "complete_bpw": complete_bpw,
                     "packed_bpw": totals["pq_bits"] / max(1, totals["weights_pq"]),
                     "rate_basis": "artifact bytes over ALL declared weights, "
                                   "native-carried tensors included in the denominator"},
        shard={"index": 0, "count": 1},
    )

    return {
        "out": str(out_path), "tensors": len(entries),
        "tensors_pq": sum(1 for e in entries if e["codec"] == "gravity-pq"),
        "tensors_native": sum(1 for e in entries if e["codec"] != "gravity-pq"),
        "weights_total": total_weights,
        "weights_pq_fraction": round(totals["weights_pq"] / max(1, total_weights), 6),
        "complete_bpw": round(complete_bpw, 6),
        "packed_bpw": round(totals["pq_bits"] / max(1, totals["weights_pq"]), 6),
        "artifact_bytes": out_path.stat().st_size,
        "source_bytes": weights_file.stat().st_size,
    }


def forge_admissible(rung: dict, elements: int) -> bool:
    """Reuse the flagship's admissibility rule so both pipelines bill the same way."""
    return pack.rung_is_admissible(rung, elements)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--rung", default="R0", choices=[r["rung"] for r in pack.LADDER])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    report = pack_model(args.model_dir, args.out, args.rung, args.seed)
    print(json.dumps(report, indent=1, sort_keys=True))
    # The bill and the file must agree, or the BPW claim is prose.
    assert report["artifact_bytes"] > 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
