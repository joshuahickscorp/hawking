#!/usr/bin/env python3
"""FIRST_NOETIC_EXECUTABLE: compile a sub-2-bit MLP mix and decode it natively.

Partial mixes are legitimate. The native runtime already has a fused
binary-group kernel (HGRAVB01) and the production HQ30UQ4 kernel. This
harness hardlinks the sealed uniform-q4 catalog, re-encodes selected MLP
tensors as mean-abs binary (the operator that survived L0 down_proj in
NOETIC_COMPOSITION), writes catalog.hq38m20, and runs
ascension_qwen38_hybrid_greedy.

Does not load a second 27B. Streams one parent tensor at a time. Does not
write under ~/models. Does not touch receipts/ascent-2026-08-16 or
workspace/campaign.

    python3 tools/headless/first_noetic_executable.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RECEIPT = REPO / "receipts" / "headless" / "FIRST_NOETIC_EXECUTABLE.json"
SCHEMA = "hawking.headless.first_noetic_executable.v1"
Q4_INCUMBENT_EBPW = 4.252735126866492
PARENT_PARAMS = 26_895_998_464

Q4_ROOT = Path(
    os.environ.get(
        "QWEN38_Q4_ARTIFACT",
        str(Path.home() / "models/qwen38-gravity-uniform-q4-v1"),
    )
)
PARENT_BF16 = Path(
    os.environ.get(
        "QWEN38_PARENT_BF16",
        str(Path.home() / "models/qwen3.8-27b-abliterated-bf16"),
    )
)
TOKENIZER = PARENT_BF16 / "tokenizer.json"
ARTIFACTS_ROOT = Path(
    os.environ.get(
        "QWEN38_NOETIC_ARTIFACT_ROOT",
        str(REPO / "artifacts" / "qwen38-noetic-exec"),
    )
)

LAYERS = 64
HIDDEN = 5120
INTERMEDIATE = 17408
GROUP_BINARY_CANON = 64
GROUP_BINARY_SURVIVOR = 1024
SCALE_BITS = 16

CATALOG_MAGIC = b"HQ38M20\0"
CATALOG_VERSION = 1
RECORD_SIZE = 128
CODEC_BINARY = 0
CODEC_Q4 = 3
CODEC_F32 = 4
ORGAN_GATE = 0
ORGAN_UP = 1
ORGAN_DOWN = 2
ORGAN_ATTN = 3
ORGAN_EMB = 4
ORGAN_HEAD = 5
ORGAN_SMALL = 6

MAGIC_BINARY = b"HGRAVB01"
SCHEMA_BINARY = "hawking.gravity.binary_sign_scale.v1"

PROMPT = (
    "Explain, in ordinary prose and at length, how a compiler turns a "
    "for-loop into basic blocks and then into machine code."
)
MAX_NEW = 16
MAX_SEQ = 128

Q4_INCUMBENT = {
    "complete_physical_bpw": Q4_INCUMBENT_EBPW,
    "artifact": str(Q4_ROOT),
    "schema": "hawking.ascent.qwen38_language_uniform_q4.v1",
}


class PackError(RuntimeError):
    pass


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def artifact_filename(name: str, ext: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest() + f".{ext}"


def catalog_name(layer: int, suffix: str) -> str:
    return f"language_model.model.layers.{layer}.{suffix}"


def parent_name(layer: int, suffix: str) -> str:
    return f"model.language_model.layers.{layer}.{suffix}"


def organ_of(name: str) -> int:
    if name.endswith("embed_tokens.weight"):
        return ORGAN_EMB
    if name.endswith("lm_head.weight"):
        return ORGAN_HEAD
    if name.endswith("mlp.gate_proj.weight"):
        return ORGAN_GATE
    if name.endswith("mlp.up_proj.weight"):
        return ORGAN_UP
    if name.endswith("mlp.down_proj.weight"):
        return ORGAN_DOWN
    if ".mlp." in name:
        return ORGAN_SMALL
    if "self_attn." in name or "linear_attn." in name:
        if name.endswith(".weight") and any(
            s in name
            for s in (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "in_proj",
                "out_proj",
            )
        ):
            return ORGAN_ATTN
        return ORGAN_SMALL
    return ORGAN_SMALL


def binary_storage_bpw(group: int) -> float:
    return 1.0 + (SCALE_BITS / float(group))


def pack_hgravb01(weights: np.ndarray, group_size: int) -> bytes:
    """HGRAVB01 mean-abs sign code. Matches hawking-core pack_binary_group."""
    if weights.ndim != 2:
        raise PackError(f"binary packer wants rank-2, got {weights.shape}")
    rows, cols = (int(weights.shape[0]), int(weights.shape[1]))
    if group_size <= 0 or cols % group_size != 0:
        raise PackError(f"cols={cols} is not a multiple of group_size={group_size}")
    flat = np.ascontiguousarray(weights, dtype=np.float32)
    if not np.isfinite(flat).all():
        raise PackError("binary source is non-finite")
    groups_per_row = cols // group_size
    grouped = flat.reshape(rows, groups_per_row, group_size)
    # f64 mean then f16, matching the rust packer (f64 sum / n → f32 → f16).
    mean_abs = np.abs(grouped.astype(np.float64)).mean(axis=-1)
    scales = mean_abs.astype(np.float32).astype(np.float16)
    signs = (flat.reshape(-1) >= 0.0).astype(np.uint8)
    sign_bytes = np.packbits(signs, bitorder="little")
    groups = rows * groups_per_row
    header = {
        "schema": SCHEMA_BINARY,
        "representation": "binary_sign_scale",
        "shape": [rows, cols],
        "elements": rows * cols,
        "group_size": int(group_size),
        "groups": int(groups),
        "scale_bytes": int(groups * 2),
        "sign_bytes": int(sign_bytes.size),
    }
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    body = scales.tobytes() + sign_bytes.tobytes()
    if len(body) != header["scale_bytes"] + header["sign_bytes"]:
        raise PackError("HGRAVB01 body ledger drifted")
    return MAGIC_BINARY + struct.pack("<I", len(header_bytes)) + header_bytes + body


def parse_hgravb01(payload: bytes) -> dict[str, Any]:
    if payload[:8] != MAGIC_BINARY:
        raise PackError(f"magic {payload[:8]!r} is not HGRAVB01")
    header_len = struct.unpack_from("<I", payload, 8)[0]
    header = json.loads(payload[12 : 12 + header_len])
    body = payload[12 + header_len :]
    if len(body) != int(header["scale_bytes"]) + int(header["sign_bytes"]):
        raise PackError("HGRAVB01 body length disagrees with ledger")
    return header


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def hardlink_or_copy(src: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        if dest.stat().st_ino == src.stat().st_ino:
            return "reused"
        dest.unlink()
    try:
        os.link(src, dest)
        return "hardlink"
    except OSError:
        dest.write_bytes(src.read_bytes())
        return "copy"


class SourceBF16:
    def __init__(self, root: Path):
        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._hdr: dict[str, tuple[Path, int, dict]] = {}

    def _header(self, shard: str) -> tuple[Path, int, dict]:
        if shard in self._hdr:
            return self._hdr[shard]
        path = self.root / shard
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen))
        self._hdr[shard] = (path, hlen, hdr)
        return self._hdr[shard]

    def load(self, src_name: str) -> np.ndarray:
        shard = self.weight_map[src_name]
        path, hlen, hdr = self._header(shard)
        meta = hdr[src_name]
        s, e = meta["data_offsets"]
        with open(path, "rb") as f:
            f.seek(8 + hlen + s)
            raw = f.read(e - s)
        dt = meta["dtype"]
        if dt in ("BF16", "BFLOAT16"):
            u16 = np.frombuffer(raw, dtype="<u2")
            arr = (u16.astype(np.uint32) << 16).view(np.float32).copy()
        elif dt in ("F32", "FLOAT32"):
            arr = np.frombuffer(raw, dtype="<f4").copy()
        else:
            raise PackError(f"{src_name} dtype {dt}")
        return arr.reshape(meta["shape"])


def write_catalog(
    path: Path,
    records: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> bytes:
    names = [r["name"] for r in records]
    name_blob = bytearray()
    offs: list[int] = []
    for name in names:
        raw = name.encode("utf-8")
        offs.append(len(name_blob))
        name_blob.extend(raw)
    table = bytearray()
    for rec, off in zip(records, offs):
        raw_name = rec["name"].encode("utf-8")
        dims = [0, 0, 0, 0]
        shape = rec["shape"]
        if len(shape) > 4:
            raise PackError(f"{rec['name']} rank {len(shape)} exceeds catalog")
        for i, d in enumerate(shape):
            dims[i] = int(d)
        digest = bytes.fromhex(rec["sha256"])
        if len(digest) != 32:
            raise PackError("catalog sha256 is not 32 bytes")
        rec_bytes = struct.pack(
            "<IHBBBB",
            off,
            len(raw_name),
            int(rec["codec"]),
            int(rec["organ"]),
            len(shape),
            0,
        )
        rec_bytes += b"\x00\x00"
        rec_bytes += struct.pack(
            "<IIIIQHHQQ32sIIf",
            dims[0],
            dims[1],
            dims[2],
            dims[3],
            int(rec["elements"]),
            int(rec["segment_id"]),
            int(rec.get("achieved_rank") or 0),
            int(rec["offset"]),
            int(rec["nbytes"]),
            digest,
            int(rec.get("flags") or 0),
            int(rec.get("n_fit_rows") or 0),
            float(rec["codec_bpw"]),
        )
        if len(rec_bytes) > RECORD_SIZE:
            raise PackError(f"record packed {len(rec_bytes)} > {RECORD_SIZE}")
        rec_bytes = rec_bytes + b"\x00" * (RECORD_SIZE - len(rec_bytes))
        table.extend(rec_bytes)
    seg_blob = bytearray()
    for seg in segments:
        name = str(seg["filename"]).encode("utf-8")
        digest = bytes.fromhex(seg["sha256"])
        if len(digest) != 32:
            raise PackError("segment sha256 is not 32 bytes")
        seg_blob.extend(
            struct.pack(
                "<HHQ32s",
                int(seg["id"]),
                len(name),
                int(seg["bytes"]),
                digest,
            )
        )
        seg_blob.extend(name)
    blob = (
        CATALOG_MAGIC
        + struct.pack(
            "<IIIIII",
            CATALOG_VERSION,
            len(records),
            len(segments),
            0,
            len(name_blob),
            0,
        )
        + bytes(seg_blob)
        + bytes(table)
        + bytes(name_blob)
    )
    write_atomic(path, blob)
    return blob


def load_q4_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise PackError(f"missing q4 manifest {path}")
    return json.loads(path.read_text())


def is_mlp_proj(name: str, which: str) -> bool:
    return name.endswith(f"mlp.{which}_proj.weight")


def mix_selects(name: str, mix_id: str) -> tuple[bool, int] | None:
    """Return (is_binary, group) if this catalog name is rewritten, else None."""
    if mix_id == "mix_a_l0_down_binary_g1024":
        if name == catalog_name(0, "mlp.down_proj.weight"):
            return True, GROUP_BINARY_SURVIVOR
        return None
    if mix_id == "mix_b_all_down_binary_g1024":
        if is_mlp_proj(name, "down"):
            return True, GROUP_BINARY_SURVIVOR
        return None
    if mix_id == "mix_c_all_mlp_binary_g64":
        if is_mlp_proj(name, "gate") or is_mlp_proj(name, "up") or is_mlp_proj(name, "down"):
            return True, GROUP_BINARY_CANON
        return None
    raise PackError(f"unknown mix {mix_id}")


def mix_recipe(mix_id: str) -> dict[str, Any]:
    if mix_id == "mix_a_l0_down_binary_g1024":
        return {
            "id": mix_id,
            "risk": "lowest",
            "tensors": "language_model.model.layers.0.mlp.down_proj.weight only",
            "codec": "HGRAVB01 mean-abs binary_sign_scale",
            "group": GROUP_BINARY_SURVIVOR,
            "layers": [0],
            "organs": ["mlp.down_proj"],
            "attention": "HQ30UQ4 g64 (hardlinked incumbent)",
            "other_mlp": "HQ30UQ4 g64 (hardlinked incumbent)",
            "why": (
                "NOETIC_COMPOSITION: binary grouped-1024 at 1.015625 bpw survived "
                "the L0 down_proj 64-layer loop (argmax 9714 = teacher). Partial mix."
            ),
        }
    if mix_id == "mix_b_all_down_binary_g1024":
        return {
            "id": mix_id,
            "risk": "medium",
            "tensors": "mlp.down_proj on layers 0..63",
            "codec": "HGRAVB01 mean-abs binary_sign_scale",
            "group": GROUP_BINARY_SURVIVOR,
            "layers": list(range(LAYERS)),
            "organs": ["mlp.down_proj"],
            "attention": "HQ30UQ4 g64 (hardlinked incumbent)",
            "other_mlp": "HQ30UQ4 g64 gate_proj and up_proj",
            "why": (
                "Doctor v2: down_proj is the most sensitive SwiGLU member. "
                "Sub-2-bit on down_proj only; attention left at q4."
            ),
        }
    if mix_id == "mix_c_all_mlp_binary_g64":
        return {
            "id": mix_id,
            "risk": "higher",
            "tensors": "mlp.gate_proj, mlp.up_proj, mlp.down_proj on layers 0..63",
            "codec": "HGRAVB01 mean-abs binary_sign_scale",
            "group": GROUP_BINARY_CANON,
            "layers": list(range(LAYERS)),
            "organs": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
            "attention": "HQ30UQ4 g64 (hardlinked incumbent)",
            "other_mlp": "none — every MLP GEMV is binary",
            "why": (
                "FRACTIONAL_BIT_CANON ternary/binary locally survived at g64. "
                "Whole-model mean-abs ternary already failed; this is the "
                "activation-unaware binary counterpart on every MLP tensor."
            ),
        }
    raise PackError(f"unknown mix {mix_id}")


def compile_mix(
    mix_id: str,
    *,
    q4_root: Path = Q4_ROOT,
    parent: Path = PARENT_BF16,
    out_root: Path | None = None,
) -> dict[str, Any]:
    recipe = mix_recipe(mix_id)
    dest = Path(out_root or (ARTIFACTS_ROOT / mix_id))
    dest.mkdir(parents=True, exist_ok=True)
    segments_dir = dest / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_q4_manifest(q4_root)
    rows = list(manifest["tensors"])
    src = SourceBF16(parent)
    records: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    binary_names: list[str] = []
    t0 = time.perf_counter()
    payload_bytes = 0
    binary_bytes = 0
    q4_bytes = 0
    f32_bytes = 0
    n_hardlink = 0
    n_binary = 0
    for i, row in enumerate(rows):
        name = row["name"]
        shape = [int(x) for x in row["shape"]]
        elements = int(row["elements"])
        src_artifact = q4_root / "tensors" / row["artifact"]
        if not src_artifact.is_file():
            raise PackError(f"incumbent missing {src_artifact}")
        selected = mix_selects(name, mix_id)
        if selected is None:
            filename = row["artifact"]
            dest_path = segments_dir / filename
            hardlink_or_copy(src_artifact, dest_path)
            n_hardlink += 1
            nbytes = int(dest_path.stat().st_size)
            codec = CODEC_Q4 if row["kind"] == "q4" else CODEC_F32
            codec_bpw = 8.0 * nbytes / max(elements, 1)
            digest = "00" * 32
            if codec == CODEC_Q4:
                q4_bytes += nbytes
            else:
                f32_bytes += nbytes
        else:
            _, group = selected
            parent_key = name.replace("language_model.model.", "model.language_model.")
            if parent_key == name and name.startswith("language_model."):
                parent_key = "model." + name
            print(f"  [{mix_id}] binary {name} group={group}", flush=True)
            w = src.load(parent_key)
            if list(w.shape) != shape:
                raise PackError(f"{name} parent shape {list(w.shape)} != catalog {shape}")
            payload = pack_hgravb01(w, group)
            del w
            filename = artifact_filename(name, "hgravb01")
            dest_path = segments_dir / filename
            write_atomic(dest_path, payload)
            nbytes = len(payload)
            digest = sha256_hex(payload)
            codec = CODEC_BINARY
            codec_bpw = binary_storage_bpw(group)
            binary_names.append(name)
            binary_bytes += nbytes
            n_binary += 1
        payload_bytes += nbytes
        seg_id = i
        segments.append(
            {
                "id": seg_id,
                "filename": filename,
                "bytes": nbytes,
                "sha256": digest if digest != "00" * 32 else sha256_hex(filename.encode()),
            }
        )
        records.append(
            {
                "name": name,
                "codec": codec,
                "organ": organ_of(name),
                "shape": shape,
                "elements": elements,
                "segment_id": seg_id,
                "offset": 0,
                "nbytes": nbytes,
                "sha256": segments[-1]["sha256"],
                "codec_bpw": codec_bpw,
            }
        )
    catalog_path = dest / "catalog.hq38m20"
    write_catalog(catalog_path, records, segments)
    complete_ebpw = 8.0 * payload_bytes / PARENT_PARAMS
    storage_bpw = complete_ebpw
    active_bpw = complete_ebpw
    codecs = Counter(int(r["codec"]) for r in records)
    report = {
        "mix_id": mix_id,
        "recipe": recipe,
        "artifact_root": str(dest),
        "catalog": str(catalog_path),
        "n_tensors": len(records),
        "n_binary": n_binary,
        "n_hardlink": n_hardlink,
        "codecs": {str(k): int(v) for k, v in sorted(codecs.items())},
        "binary_tensors": binary_names,
        "binary_group": recipe["group"],
        "binary_storage_bpw": binary_storage_bpw(int(recipe["group"])),
        "payload_bytes": payload_bytes,
        "binary_bytes": binary_bytes,
        "q4_bytes": q4_bytes,
        "f32_bytes": f32_bytes,
        "parent_params": PARENT_PARAMS,
        "storage_bpw": storage_bpw,
        "active_bpw": active_bpw,
        "active_fused_bpw": active_bpw,
        "complete_ebpw": complete_ebpw,
        "q4_incumbent_complete_physical_bpw": Q4_INCUMBENT_EBPW,
        "wall_s": time.perf_counter() - t0,
        "did_not_load_second_27b": True,
        "parent_streamed_one_tensor_at_a_time": True,
        "wrote_under_models": False,
    }
    write_atomic(
        dest / "MIX_REPORT.json",
        json.dumps(report, indent=2).encode(),
    )
    print(
        f"[{mix_id}] tensors={len(records)} binary={n_binary} "
        f"ebpw={complete_ebpw:.6f} (q4 {Q4_INCUMBENT_EBPW:.6f}) "
        f"in {report['wall_s']:.1f}s",
        flush=True,
    )
    return report


def find_decode_binary() -> Path:
    env = os.environ.get("QWEN38_DECODE_BIN")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    candidates = [
        REPO
        / "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy",
        Path.home()
        / "Downloads/hawking-copy/workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise PackError(
        "ascension_qwen38_hybrid_greedy is not built; "
        "cargo build --profile release-fast -p hawking-core "
        "--example ascension_qwen38_hybrid_greedy"
    )


def judge_coherence(text: str, token_ids: list[int]) -> dict[str, Any]:
    n = len(token_ids)
    if n == 0:
        return {
            "coherent": False,
            "reason": "no tokens generated",
            "repeated_single_token": False,
            "n_unique_ids": 0,
        }
    unique = len(set(token_ids))
    repeated_single = unique == 1
    stripped = (text or "").strip()
    whitespace_only = stripped == ""
    two_cycle = False
    if n >= 8 and unique == 2:
        a, b = token_ids[0], token_ids[1]
        two_cycle = all(
            token_ids[i] == (a if i % 2 == 0 else b) for i in range(n)
        )
    reason = "emits varied tokens"
    if n < 16:
        reason = f"only {n} new tokens (stopped early, likely EOS); not a 16-token sample"
        coherent = False
    elif repeated_single:
        reason = f"16 copies of the same token ({token_ids[0]})"
        coherent = False
    elif whitespace_only:
        reason = "generated text is whitespace only"
        coherent = False
    elif two_cycle:
        reason = f"two-token cycle {token_ids[0]}/{token_ids[1]}"
        coherent = False
    else:
        coherent = True
    return {
        "coherent": coherent,
        "reason": reason,
        "repeated_single_token": repeated_single,
        "whitespace_only": whitespace_only,
        "two_token_cycle": two_cycle,
        "n_unique_ids": unique,
        "n_new_tokens": n,
    }


def decode_mix(
    artifact_root: Path,
    *,
    binary: Path | None = None,
    prompt: str = PROMPT,
    max_new: int = MAX_NEW,
    max_seq: int = MAX_SEQ,
    tokenizer: Path = TOKENIZER,
) -> dict[str, Any]:
    exe = binary or find_decode_binary()
    out_json = artifact_root / "decode.json"
    cmd = [
        str(exe),
        "--artifact-root",
        str(artifact_root),
        "--tokenizer",
        str(tokenizer),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new),
        "--max-seq-len",
        str(max_seq),
        "--out",
        str(out_json),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    wall_s = time.perf_counter() - t0
    stdout = proc.stdout
    stderr = proc.stderr
    result: dict[str, Any] = {
        "command": cmd,
        "binary": str(exe),
        "exit_code": proc.returncode,
        "wall_s": wall_s,
        "stdout": stdout[-8000:],
        "stderr": stderr[-8000:],
    }
    if proc.returncode != 0:
        result["ok"] = False
        result["generated_text"] = None
        result["coherence"] = {
            "coherent": False,
            "reason": f"decode exit {proc.returncode}",
        }
        return result
    body: dict[str, Any] = {}
    if out_json.is_file():
        body = json.loads(out_json.read_text())
    text = body.get("generated_text")
    if text is None:
        for line in stdout.splitlines():
            if line.startswith("GENERATED_TEXT_VERBATIM: "):
                text = line[len("GENERATED_TEXT_VERBATIM: ") :]
                break
    ids = [int(x) for x in body.get("new_token_ids") or []]
    decode_steps = int(body.get("decode_steps") or max(len(ids), 1))
    decode_wall_ns = int(body.get("decode_wall_ns") or 0)
    tok_s = None
    if decode_wall_ns > 0 and decode_steps > 0:
        tok_s = decode_steps / (decode_wall_ns / 1e9)
    elif wall_s > 0 and ids:
        tok_s = len(ids) / wall_s
    native = "qwen38-decode mixed HQ38M20" in stderr or "mixed bind" in stderr
    # Census prints expanded_to_q4=0 on the native path; do not treat that as dequant.
    dequant = ("expanded_to_q4=" in stderr and "expanded_to_q4=0" not in stderr) or (
        "reconstruct-to-Q4" in stderr and "no reconstruct-to-Q4" not in stderr
    )
    fused_binary = "q80_binary_group_matvec" in stderr
    result.update(
        {
            "ok": True,
            "prompt": body.get("prompt") or prompt,
            "generated_text": text if text is not None else "",
            "generated_text_verbatim": text if text is not None else "",
            "new_token_ids": ids,
            "n_new_tokens": len(ids),
            "fallbacks": int(body.get("fallbacks") or 0),
            "dense_w_materialized": int(body.get("dense_w_materialized") or 0),
            "decode_wall_ns": decode_wall_ns,
            "decode_steps": decode_steps,
            "tok_s": tok_s,
            "median_gpu_ns_per_token": body.get("median_gpu_ns_per_token"),
            "native_kernel_ran": bool(native or fused_binary) and not dequant,
            "dequant_path": bool(dequant),
            "stderr_saw_mixed_catalog": "HQ38M20" in stderr,
            "stderr_saw_binary_kernel": fused_binary,
            "coherence": judge_coherence(text or "", ids),
        }
    )
    return result


def find_decode_binary_optional() -> Path | None:
    try:
        return find_decode_binary()
    except PackError:
        return None


MIX_ORDER = (
    "mix_a_l0_down_binary_g1024",
    "mix_b_all_down_binary_g1024",
    "mix_c_all_mlp_binary_g64",
)


def run_all(
    *,
    mixes: Iterable[str] = MIX_ORDER,
    decode: bool = True,
    out_receipt: Path = RECEIPT,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    for mix_id in mixes:
        print(f"== compile {mix_id} ==", flush=True)
        compiled = compile_mix(mix_id)
        attempt: dict[str, Any] = {"compile": compiled, "decode": None}
        if decode:
            print(f"== decode {mix_id} ==", flush=True)
            attempt["decode"] = decode_mix(Path(compiled["artifact_root"]))
            gen = (attempt["decode"] or {}).get("generated_text_verbatim")
            coh = (attempt["decode"] or {}).get("coherence") or {}
            print(
                f"[{mix_id}] exit={attempt['decode'].get('exit_code')} "
                f"coherent={coh.get('coherent')} text={gen!r}",
                flush=True,
            )
        attempts.append(attempt)

    decoded = [a for a in attempts if a.get("decode") and a["decode"].get("ok")]
    chosen = None
    for a in decoded:
        if a["decode"]["coherence"]["coherent"]:
            chosen = a
            break
    if chosen is None and decoded:
        chosen = decoded[0]

    not_tried = [
        {
            "id": "mix_d_activation_aware_ternary_all_mlp",
            "risk": "highest listed",
            "status": "NOT_RUN_NO_NATIVE_KERNEL",
            "reason": (
                "FRACTIONAL_BIT_CANON ternary 5-in-8 at g64 is 1.85 bpw locally. "
                "The native runtime has no ternary fused kernel. Packing it as "
                "reconstruct-then-GEMV would be an ORACLE path. Whole-model "
                "mean-abs ternary already failed (NOETIC_COMPOSITION_WHOLEMODEL_TERNARY). "
                "This lane reports the binary mixes that the HGRAVB01 kernel can run."
            ),
        }
    ]

    receipt = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "question": (
            "Produce the first artifact that is BOTH sub-2-bit somewhere real "
            "AND actually decodes on the native runtime."
        ),
        "q4_incumbent": Q4_INCUMBENT,
        "parent_bf16": str(PARENT_BF16),
        "parent_params": PARENT_PARAMS,
        "did_not_load_second_27b": True,
        "did_not_write_under_models": True,
        "mixes_attempted": attempts,
        "mixes_not_run": not_tried,
        "chosen": None,
        "elapsed_s": time.perf_counter() - t0,
    }
    if chosen is not None:
        c = chosen["compile"]
        d = chosen["decode"]
        receipt["chosen"] = {
            "mix_id": c["mix_id"],
            "recipe": c["recipe"],
            "artifact_root": c["artifact_root"],
            "exact_mix": {
                "tensors": c["recipe"]["tensors"],
                "codec": c["recipe"]["codec"],
                "group": c["recipe"]["group"],
                "layers": c["recipe"]["layers"],
                "attention": c["recipe"]["attention"],
                "binary_tensors": c["binary_tensors"],
            },
            "storage_bpw": c["storage_bpw"],
            "active_bpw": c["active_bpw"],
            "complete_ebpw": c["complete_ebpw"],
            "q4_incumbent_complete_physical_bpw": Q4_INCUMBENT_EBPW,
            "binary_tensor_storage_bpw": c["binary_storage_bpw"],
            "prompt": d.get("prompt"),
            "generated_text_verbatim": d.get("generated_text_verbatim"),
            "new_token_ids": d.get("new_token_ids"),
            "n_new_tokens": d.get("n_new_tokens"),
            "tok_s": d.get("tok_s"),
            "native_kernel_ran": d.get("native_kernel_ran"),
            "dequant_path": d.get("dequant_path"),
            "fallbacks": d.get("fallbacks"),
            "dense_w_materialized": d.get("dense_w_materialized"),
            "coherence": d.get("coherence"),
        }
    out_receipt.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_receipt.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(tmp, out_receipt)
    print(f"wrote {out_receipt}", flush=True)
    return receipt


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    decode = "--pack-only" not in args
    mixes = [a for a in args if a.startswith("mix_")]
    if "--all-mlp" in args:
        mixes = ["mix_c_all_mlp_binary_g64"]
    if "--l0-down" in args:
        mixes = ["mix_a_l0_down_binary_g1024"]
    if not mixes:
        mixes = list(MIX_ORDER)
    run_all(mixes=mixes, decode=decode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
