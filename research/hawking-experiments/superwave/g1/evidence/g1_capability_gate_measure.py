#!/usr/bin/env python3
"""G1 capability-gate measurements. No GPU. Streams tensors. Peak << 20 GB."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from collections import Counter

import numpy as np

G0 = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1"
BF16 = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
MIXED = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-2p0-v1"
NATIVE_GEN = "/tmp/QWEN38_NATIVE_MIXED_2P0_GENERATE.json"

GROUP = 64


def labeled_sha(label: str) -> str:
    return hashlib.sha256(f"hawking.lineage/{label}".encode()).hexdigest()


def sha256_file(path: str, buf: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_q4_header(path: str) -> dict:
    with open(path, "rb") as f:
        head = f.read(32)
    if head[:8] != b"HQ30UQ4\0":
        raise ValueError(f"not HQ30UQ4: {path}")
    version, group, rank, reserved = struct.unpack_from("<IIHH", head, 8)
    elements, reserved_tail = struct.unpack_from("<QI", head, 20)
    if version != 1 or group != 64 or reserved != 0 or reserved_tail != 0:
        raise ValueError(f"bad q4 header {path} v={version} g={group}")
    with open(path, "rb") as f:
        f.seek(32)
        dims = struct.unpack("<" + "I" * rank, f.read(4 * rank))
    groups = (elements + 63) // 64
    dim_off = 32
    scale_off = dim_off + 4 * rank
    code_off = scale_off + 2 * groups
    payload = os.path.getsize(path)
    expect = code_off + 32 * groups
    if payload != expect:
        raise ValueError(f"size {payload} != {expect} for {path}")
    return {
        "shape": list(dims),
        "elements": int(elements),
        "groups": int(groups),
        "scale_off": scale_off,
        "code_off": code_off,
        "bytes": payload,
    }


def load_safetensors_index() -> dict:
    idx = json.load(open(os.path.join(BF16, "model.safetensors.index.json")))
    return idx["weight_map"]


_ST_CACHE: dict[str, dict] = {}


def safetensor_loc(shard_name: str, tensor_name: str) -> tuple[str, int, int, list[int], str]:
    path = os.path.join(BF16, shard_name)
    if shard_name not in _ST_CACHE:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        _ST_CACHE[shard_name] = {"n": n, "hdr": hdr, "data0": 8 + n}
    rec = _ST_CACHE[shard_name]
    meta = rec["hdr"][tensor_name]
    start, end = meta["data_offsets"]
    return path, rec["data0"] + start, end - start, list(meta["shape"]), meta["dtype"]


def widen_bf16(raw: bytes) -> np.ndarray:
    bits = np.frombuffer(raw, dtype="<u2")
    return (bits.astype(np.uint32) << 16).view(np.float32)


def dequant_q4_group(scales_u16: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """scales_u16: (G,) uint16 f16 bits; codes: (G, 32) uint8. Returns (G, 64) f32."""
    scale = scales_u16.view(np.float16).astype(np.float32)
    low = (codes & 0x0F).astype(np.int16) - 8
    high = (codes >> 4).astype(np.int16) - 8
    q = np.empty((codes.shape[0], 64), dtype=np.float32)
    q[:, 0::2] = low
    q[:, 1::2] = high
    return q * scale[:, None]


def streamed_q4_cosine(q4_path: str, shard: str, tensor_name: str, chunk_groups: int = 4096) -> dict:
    hdr = parse_q4_header(q4_path)
    path, off, nbytes, shape, dtype = safetensor_loc(shard, tensor_name)
    elements = int(np.prod(shape))
    if elements != hdr["elements"]:
        raise ValueError(f"{tensor_name} bf16 elems {elements} != q4 {hdr['elements']}")
    if dtype.upper() not in {"BF16", "BFLOAT16"}:
        raise ValueError(f"{tensor_name} dtype {dtype}")
    if nbytes != elements * 2:
        raise ValueError(f"{tensor_name} bf16 bytes {nbytes} != {elements*2}")

    g = hdr["groups"]
    dot = 0.0
    src_sq = 0.0
    rec_sq = 0.0
    err_sq = 0.0
    n_nonfinite = 0
    t0 = time.perf_counter()
    with open(q4_path, "rb") as qf, open(path, "rb") as sf:
        for g0 in range(0, g, chunk_groups):
            gn = min(chunk_groups, g - g0)
            qf.seek(hdr["scale_off"] + g0 * 2)
            scales = np.frombuffer(qf.read(gn * 2), dtype="<u2").copy()
            qf.seek(hdr["code_off"] + g0 * 32)
            codes = np.frombuffer(qf.read(gn * 32), dtype=np.uint8).copy().reshape(gn, 32)
            recon = dequant_q4_group(scales, codes)  # (gn, 64)
            e0 = g0 * 64
            e1 = min(e0 + gn * 64, elements)
            take = e1 - e0
            sf.seek(off + e0 * 2)
            src = widen_bf16(sf.read(take * 2))
            flat = recon.reshape(-1)[:take]
            # f64 accumulators
            src64 = src.astype(np.float64, copy=False)
            rec64 = flat.astype(np.float64, copy=False)
            n_nonfinite += int((~np.isfinite(src64)).sum() + (~np.isfinite(rec64)).sum())
            dot += float(np.dot(src64, rec64))
            src_sq += float(np.dot(src64, src64))
            rec_sq += float(np.dot(rec64, rec64))
            diff = src64 - rec64
            err_sq += float(np.dot(diff, diff))
    src_n = src_sq**0.5
    rec_n = rec_sq**0.5
    cosine = None if src_n == 0.0 or rec_n == 0.0 else dot / (src_n * rec_n)
    return {
        "name": tensor_name,
        "shape": shape,
        "elements": elements,
        "q4_bytes": hdr["bytes"],
        "cosine": cosine,
        "rel_l2": None if src_n == 0.0 else err_sq**0.5 / src_n,
        "rmse": (err_sq / max(elements, 1)) ** 0.5,
        "src_norm": src_n,
        "recon_norm": rec_n,
        "n_nonfinite": n_nonfinite,
        "wall_s": time.perf_counter() - t0,
        "kind": "q4_dequant_vs_bf16_weight",
    }


def streamed_f32v2_vs_bf16(f32_path: str, shard: str, tensor_name: str, delta: bool) -> dict:
    raw = open(f32_path, "rb").read()
    n = struct.unpack_from("<Q", raw, 0)[0]
    stored = np.frombuffer(raw, dtype="<f4", count=n, offset=8).astype(np.float64)
    path, off, nbytes, shape, dtype = safetensor_loc(shard, tensor_name)
    elements = int(np.prod(shape))
    with open(path, "rb") as sf:
        sf.seek(off)
        src = widen_bf16(sf.read(nbytes)).astype(np.float64)
    if delta:
        src = src - 1.0
    if stored.size != src.size:
        raise ValueError(f"{tensor_name} f32 {stored.size} vs bf16 {src.size}")
    dot = float(np.dot(stored, src))
    sn = float(np.dot(src, src)) ** 0.5
    rn = float(np.dot(stored, stored)) ** 0.5
    err = stored - src
    return {
        "name": tensor_name,
        "shape": shape,
        "elements": elements,
        "cosine": None if sn == 0 or rn == 0 else dot / (sn * rn),
        "rel_l2": None if sn == 0 else float(np.dot(err, err)) ** 0.5 / sn,
        "compare_to": "bf16_minus_1" if delta else "bf16_raw",
        "stored_first": float(stored[0]),
        "src_first": float(src[0]),
        "kind": "f32v2_vs_bf16",
    }


def self_check_q4_roundtrip() -> dict:
    rng = np.random.default_rng(0)
    src = rng.normal(0, 0.02, size=256).astype(np.float32)
    # pack
    groups = 256 // 64
    scales = []
    codes = np.zeros((groups, 32), dtype=np.uint8)
    recon = np.zeros(256, dtype=np.float32)
    for g in range(groups):
        sl = src[g * 64 : (g + 1) * 64]
        max_abs = float(np.max(np.abs(sl)))
        scale = np.float16(max_abs / 7.0).astype(np.float32)
        scales.append(np.float16(scale).view(np.uint16))
        for i, v in enumerate(sl):
            q = 0 if scale == 0 else int(np.rint(np.clip(v / scale, -8, 7)))
            # ties-to-even via numpy rint
            code = q + 8
            if i & 1 == 0:
                codes[g, i // 2] |= code
            else:
                codes[g, i // 2] |= code << 4
            recon[g * 64 + i] = q * scale
    scales_u16 = np.array(scales, dtype=np.uint16)
    recon2 = dequant_q4_group(scales_u16, codes).reshape(-1)
    if not np.allclose(recon, recon2):
        raise SystemExit("dequant helper disagrees with pack recon")
    src64 = src.astype(np.float64)
    rec64 = recon.astype(np.float64)
    cos = float(np.dot(src64, rec64) / (np.linalg.norm(src64) * np.linalg.norm(rec64)))
    return {"self_check_cosine": cos, "max_abs_err": float(np.max(np.abs(src - recon)))}


# --- generation classifiers (no GPU; operate on sealed receipts) ---

NEWLINE = 198
CLOSE_PAREN = 8
SPACE = 220
DOT = 13


def token_class_stats(ids: list[int]) -> dict:
    n = len(ids)
    if n == 0:
        return {"n": 0, "unique": 0, "unique_ratio": 0.0, "top_id": None, "top_frac": 0.0, "entropy": 0.0}
    c = Counter(ids)
    top_id, top_n = c.most_common(1)[0]
    # unigram entropy bits
    ent = 0.0
    for v in c.values():
        p = v / n
        ent -= p * (np.log2(p) if p > 0 else 0.0)
    return {
        "n": n,
        "unique": len(c),
        "unique_ratio": len(c) / n,
        "top_id": top_id,
        "top_frac": top_n / n,
        "entropy_bits": ent,
        "newline_frac": c.get(NEWLINE, 0) / n,
        "punct_only_frac": sum(c[t] for t in (NEWLINE, CLOSE_PAREN, DOT, SPACE, 1076, 578) if t in c) / n,
    }


def classify_generation(prompt: str, ids: list[int], text: str, parent_ids: list[int] | None = None) -> dict:
    stats = token_class_stats(ids)
    text_l = (text or "").lower()
    reasons = []
    klass = "COHERENT"

    if stats["n"] == 0:
        klass = "INCOHERENT"
        reasons.append("empty_generation")
    elif stats["unique"] <= 2 and stats["n"] >= 8:
        klass = "INCOHERENT"
        reasons.append("degenerate_cycle")
    elif stats["top_frac"] >= 0.70 and stats["n"] >= 8:
        klass = "INCOHERENT"
        reasons.append("majority_token_collapse")
    elif stats["newline_frac"] >= 0.50:
        klass = "INCOHERENT"
        reasons.append("newline_collapse")
    elif stats["punct_only_frac"] >= 0.85:
        klass = "INCOHERENT"
        reasons.append("punctuation_only")
    elif stats["entropy_bits"] < 1.0 and stats["n"] >= 8:
        klass = "INCOHERENT"
        reasons.append("low_entropy")

    # echo: generated text is a prefix of the prompt or equals the prompt
    prompt_l = prompt.lower().strip()
    gen_stripped = (text or "").strip().lower()
    if gen_stripped and (gen_stripped == prompt_l or prompt_l.startswith(gen_stripped) and len(gen_stripped) >= 8):
        if klass == "COHERENT":
            klass = "ECHO"
            reasons.append("prompt_echo")

    # task probes
    task = None
    task_ok = None
    if "capital of france" in prompt_l:
        task = "france"
        task_ok = "paris" in text_l
        if task_ok is False and klass == "COHERENT":
            # grammatical-looking failure
            if stats["unique_ratio"] > 0.4 and stats["newline_frac"] < 0.2:
                klass = "FLUENT_NONSENSE"
                reasons.append("france_missing_paris")
            else:
                klass = "INCOHERENT"
                reasons.append("france_missing_paris")
    elif "17 times 19" in prompt_l or "17 × 19" in prompt_l or "17 x 19" in prompt_l:
        task = "arithmetic_17_19"
        task_ok = "323" in (text or "")
        if task_ok is False and klass == "COHERENT":
            if stats["unique_ratio"] > 0.4:
                klass = "FLUENT_NONSENSE"
                reasons.append("arithmetic_missing_323")
            else:
                klass = "INCOHERENT"
                reasons.append("arithmetic_missing_323")
    elif "revers" in prompt_l:
        task = "reverse_string"
        task_ok = any(w in text_l for w in ("reverse", "reversed", "[::-1]", "slic"))
        if task_ok is False and klass == "COHERENT":
            klass = "DEGRADED"
            reasons.append("reverse_task_not_addressed")

    prefix_match = None
    if parent_ids is not None:
        n = min(len(parent_ids), len(ids))
        k = 0
        while k < n and parent_ids[k] == ids[k]:
            k += 1
        prefix_match = k

    return {
        "prompt": prompt,
        "class": klass,
        "reasons": reasons,
        "task": task,
        "task_ok": task_ok,
        "parent_prefix_match": prefix_match,
        "stats": stats,
        "text": text,
        "ids": ids,
    }


def fold_min_or_none(values):
    present = [v for v in values if v is not None]
    if len(present) != len(values) or not values:
        return {"min": None, "n_none": len(values) - len(present), "n": len(values), "status": "FAIL_NONE"}
    return {"min": min(present), "n_none": 0, "n": len(values), "status": "OK"}


def main() -> None:
    out: dict = {"schema": "hawking.g1.capability_gate.measure.v1"}
    out["self_check"] = self_check_q4_roundtrip()

    out["labeled_vs_content"] = {
        "labeled_artifact": labeled_sha("artifact/qwen38-27b/uniform-q4-v1"),
        "labeled_runtime": labeled_sha("runtime/ascension_qwen38_hybrid_greedy"),
        "labeled_genome": labeled_sha("genome/Qwen38HybridDecodeSession+qwen_uniform_q4_group64"),
        "g0_manifest_sha256": sha256_file(os.path.join(G0, "manifest.json")),
        "mixed_catalog_sha256": sha256_file(os.path.join(MIXED, "catalog.hq38m20")),
        "same": False,
    }

    man = json.load(open(os.path.join(G0, "manifest.json")))
    q4 = [t for t in man["tensors"] if t["kind"] == "q4"]
    fold = fold_min_or_none([t.get("cosine") for t in q4])
    out["g0_min_q4_cosine_fold"] = {
        "manifest_field": man["min_q4_cosine"],
        "q4_tensors": len(q4),
        **fold,
    }

    # catalog merkle of G0: sha256 of "name\\0sha256\\n" lines in manifest order
    t0 = time.perf_counter()
    h = hashlib.sha256()
    file_hashes = []
    for row in man["tensors"]:
        p = os.path.join(G0, "tensors", row["artifact"])
        digest = sha256_file(p)
        file_hashes.append({"name": row["name"], "artifact": row["artifact"], "sha256": digest, "bytes": row["bytes"]})
        h.update(row["name"].encode())
        h.update(b"\0")
        h.update(digest.encode())
        h.update(b"\n")
    out["g0_catalog_merkle"] = {
        "merkle_sha256": h.hexdigest(),
        "n_files": len(file_hashes),
        "wall_s": time.perf_counter() - t0,
        "bind": "sha256(concat(name || 0x00 || sha256(file_bytes) || newline) in manifest order)",
        "first3": file_hashes[:3],
        "last1": file_hashes[-1],
    }
    # bound identity
    ident = hashlib.sha256()
    ident.update(bytes.fromhex(out["labeled_vs_content"]["g0_manifest_sha256"]))
    ident.update(bytes.fromhex(out["g0_catalog_merkle"]["merkle_sha256"]))
    out["g0_artifact_content_sha"] = ident.hexdigest()

    weight_map = load_safetensors_index()
    targets = [
        "language_model.model.layers.0.linear_attn.out_proj.weight",
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.up_proj.weight",
        "language_model.model.layers.0.mlp.down_proj.weight",
        "language_model.model.layers.3.self_attn.o_proj.weight",
        "language_model.model.layers.3.self_attn.q_proj.weight",
        "language_model.model.layers.63.self_attn.o_proj.weight",
        "language_model.model.layers.63.mlp.down_proj.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
    ]
    by_name = {t["name"]: t for t in man["tensors"]}
    cos_rows = []
    for name in targets:
        row = by_name[name]
        q4_path = os.path.join(G0, "tensors", row["artifact"])
        shard = weight_map[name]
        rec = streamed_q4_cosine(q4_path, shard, name)
        rec["manifest_cosine"] = row.get("cosine")
        rec["role"] = (
            "out_proj"
            if "out_proj" in name or "o_proj" in name
            else "q_proj"
            if "q_proj" in name
            else "gate"
            if "gate_proj" in name
            else "up"
            if "up_proj" in name
            else "down"
            if "down_proj" in name
            else "embed"
            if "embed" in name
            else "lm_head"
        )
        cos_rows.append(rec)
        print(f"COS {rec['role']:8} {name.split('language_model.')[-1]:60} {rec['cosine']:.8f} rel_l2={rec['rel_l2']:.6e} {rec['wall_s']:.2f}s", flush=True)

    out["g0_dequant_cosine_rows"] = cos_rows
    out["g0_dequant_cosine_min"] = min(r["cosine"] for r in cos_rows)
    out["g0_dequant_cosine_max"] = max(r["cosine"] for r in cos_rows)

    # f32v2 norm: stored is (mlx-1); packer writes cosine=1.0 without measuring
    f32_name = "language_model.model.layers.0.input_layernorm.weight"
    f32_row = by_name[f32_name]
    f32_path = os.path.join(G0, "tensors", f32_row["artifact"])
    out["f32v2_norm_vs_bf16_raw"] = streamed_f32v2_vs_bf16(f32_path, weight_map[f32_name], f32_name, delta=False)
    out["f32v2_norm_vs_bf16_minus_1"] = streamed_f32v2_vs_bf16(f32_path, weight_map[f32_name], f32_name, delta=True)
    print("F32 raw", out["f32v2_norm_vs_bf16_raw"])
    print("F32 dlt", out["f32v2_norm_vs_bf16_minus_1"])

    # mixed-2p0 generation classifier
    gen_path = NATIVE_GEN
    gen = json.load(open(gen_path))
    prompts = gen.get("prompts") or []
    classified = []
    for p in prompts:
        classified.append(
            classify_generation(
                p.get("prompt") or "",
                list(p.get("new_token_ids") or []),
                p.get("generated_text") or "",
            )
        )
    out["mixed_2p0_generate"] = {
        "receipt": gen_path,
        "engine_field": gen.get("engine"),
        "fallbacks_total": gen.get("fallbacks_total"),
        "dense_w_materialized_total": gen.get("dense_w_materialized_total"),
        "n_prompts": len(classified),
        "classes": Counter(c["class"] for c in classified),
        "rows": classified,
        "gate_verdict": "FAIL" if any(c["class"] != "COHERENT" for c in classified) else "PASS",
    }
    print("MIXED CLASS", dict(Counter(c["class"] for c in classified)), "verdict", out["mixed_2p0_generate"]["gate_verdict"])

    # G0 oracle-32 from remasure (sealed ids) — classify as if generated
    g0_ids = [
        248068, 198, 760, 1156, 4777, 6587, 728, 310, 1910, 328, 5834, 1149,
        1061, 369, 264, 1546, 4145, 11, 2050, 1622, 13, 353, 3172, 1066, 1910,
        15131, 303, 264, 11321, 11, 5629, 1560,
    ]
    g0_text = (
        '<think>\nThe user simply wants me to say "hi." This is a very simple, '
        "direct request. I'll just say hi in a friendly, natural way"
    )
    out["g0_oracle32_class"] = classify_generation("Say hi.", g0_ids, g0_text)
    out["g0_oracle32_ids"] = g0_ids

    dest = "/tmp/g1_capability_gate_measure.json"
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", dest)
    print("MIN_COS", out["g0_dequant_cosine_min"], "MAX_COS", out["g0_dequant_cosine_max"])
    print("ARTIFACT_CONTENT_SHA", out["g0_artifact_content_sha"])
    print("MERKLE", out["g0_catalog_merkle"]["merkle_sha256"], "wall", out["g0_catalog_merkle"]["wall_s"])


if __name__ == "__main__":
    main()
