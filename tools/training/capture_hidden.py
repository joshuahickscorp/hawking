#!/usr/bin/env python3
"""
tools/training/capture_hidden.py — orchestrator for path-to-90 C2.

Path-to-90 Stage 3 / C2. Produces (final-norm hidden state, ground-truth
next token) tuples from teacher-forced text samples, suitable for off-
machine training of an EAGLE-3 / MTP-style draft head against the
dismantle target model (DeepSeek-V2-Lite Q4_K_M).

The heavy lifting (model forward + per-token hidden capture) lives in
the Rust `dismantle capture-hidden` subcommand; this script handles
dataset prep, shard orchestration, parquet conversion, and inspection.

Subcommands
-----------
  prep        Download a chunk of UltraChat (or other HF dataset),
              extract single-turn user prompts as plain text, write
              a deterministic JSONL slice (`tests/data/ultrachat_*.jsonl`).
  run         Invoke `dismantle capture-hidden` once per shard. Resume-
              capable — checks `<shard>.bin` for already-captured
              sample_ids and skips them. Writes `<out_dir>/manifest.json`
              tracking shard files + total records.
  to-parquet  Convert one .bin shard to a Parquet file with columns
              (sample_id, pos, prev_token, next_token, hidden_bytes).
              Hidden stored as raw f16 bytes (BinaryArray) — half the
              disk of f32, universally readable by HF datasets, no fp16
              parquet-version gotchas.
  inspect     Open a .bin file, print stats (record count, hidden_dim,
              first/last sample_ids), then optionally decode N random
              records back to text via the model tokenizer (sanity
              check: prev_token + next_token round-trip).

Architecture decision: see reports/path_to_90/stage3_c1/architecture.md
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import struct
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "models/deepseek-v2-lite-q4.gguf"
DEFAULT_PROFILE = REPO_ROOT / "profiles/deepseek-v2-lite-q4.m3pro18.json"
DEFAULT_BINARY = REPO_ROOT / "target/release/dismantle"
DEFAULT_OUT_DIR = REPO_ROOT / "training_data/c2_hidden"

# Binary file format constants (must match crates/dismantle/src/main.rs).
MAGIC = b"DCAP"
VERSION = 1
HEADER_SIZE = 16


# -----------------------------------------------------------------------------
# prep — slice an HF chat dataset deterministically into JSONL
# -----------------------------------------------------------------------------

def cmd_prep(args: argparse.Namespace) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "ERROR: `datasets` package not found. Install with\n"
            "  pip install datasets",
            file=sys.stderr,
        )
        return 2

    out = pathlib.Path(args.out)
    if out.exists() and not args.force:
        print(
            f"[prep] {out} already exists; pass --force to overwrite",
            file=sys.stderr,
        )
        return 0

    print(
        f"[prep] loading {args.dataset} (split={args.split}, streaming={args.streaming})",
        file=sys.stderr,
    )
    if args.streaming:
        ds = load_dataset(args.dataset, split=args.split, streaming=True)
    else:
        ds = load_dataset(args.dataset, split=args.split)

    # Shape of each row depends on dataset:
    #   - UltraChat: row["data"] = list[str], dialogue turns
    #   - ShareGPT: row["conversations"] = list[{"from","value"}]
    #   - Alpaca-style: row["instruction"] / row["input"] / row["output"]
    # We extract a single text field per row by trying common keys.
    def extract(row) -> Optional[str]:
        # UltraChat-200K
        if "data" in row and isinstance(row["data"], list) and row["data"]:
            return row["data"][0].strip()
        # ShareGPT-style
        if "conversations" in row and isinstance(row["conversations"], list):
            for turn in row["conversations"]:
                if isinstance(turn, dict) and "value" in turn:
                    return str(turn["value"]).strip()
        # generic text
        for k in ("text", "instruction", "prompt", "input"):
            if k in row and isinstance(row[k], str) and row[k].strip():
                return row[k].strip()
        return None

    rng = random.Random(args.seed)
    paragraphs: List[Tuple[int, str]] = []
    target_pool = max(args.n * 4, 2000)  # over-sample so we can filter + shuffle
    for i, row in enumerate(ds):
        if len(paragraphs) >= target_pool:
            break
        t = extract(row)
        if not t:
            continue
        if len(t) < args.min_chars or len(t) > args.max_chars:
            continue
        paragraphs.append((i, t))
    if len(paragraphs) < args.n:
        print(
            f"ERROR: only {len(paragraphs)} substantive samples available; "
            f"need {args.n}. Try a different dataset or relax --min-chars/"
            f"--max-chars.",
            file=sys.stderr,
        )
        return 2
    print(
        f"[prep] {len(paragraphs)} candidate samples; sampling {args.n} (seed={args.seed})",
        file=sys.stderr,
    )

    sampled = rng.sample(paragraphs, args.n)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for slot, (orig_idx, text) in enumerate(sampled):
            sid = f"{args.id_prefix}_{orig_idx}_{slot}"
            f.write(json.dumps({"id": sid, "text": text}, ensure_ascii=False) + "\n")
    chars = sum(len(p[1]) for p in sampled)
    print(
        f"[prep] wrote {out} ({args.n} samples, {chars:,} chars)",
        file=sys.stderr,
    )
    return 0


# -----------------------------------------------------------------------------
# run — shell out to `dismantle capture-hidden`, sharded + resumable
# -----------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    binary = pathlib.Path(args.binary)
    if not binary.exists():
        print(
            f"ERROR: dismantle binary not found at {binary}. "
            f"Run `cargo build --release --workspace` first.",
            file=sys.stderr,
        )
        return 2
    weights = pathlib.Path(args.weights)
    if not weights.exists():
        print(f"ERROR: weights not found at {weights}", file=sys.stderr)
        return 2
    samples = pathlib.Path(args.samples)
    if not samples.exists():
        print(f"ERROR: samples not found at {samples}", file=sys.stderr)
        return 2

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single-shard mode: a smoke run for C2 only writes one shard.
    # The orchestrator currently supports one shard per invocation.
    # Multiple invocations with different --shard-name accumulate.
    shard_path = out_dir / args.shard_name

    cmd = [
        str(binary),
        "capture-hidden",
        "--weights", str(weights),
        "--samples", str(samples),
        "--out", str(shard_path),
        "--max-tokens", str(args.max_tokens),
    ]
    if args.max_samples > 0:
        cmd += ["--max-samples", str(args.max_samples)]
    if args.resume:
        cmd += ["--resume"]
    if args.no_lm_head:
        cmd += ["--no-lm-head"]
    if args.kernel_profile:
        cmd += ["--kernel-profile", str(args.kernel_profile)]

    print(f"[run] $ {' '.join(cmd)}", file=sys.stderr)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[run] dismantle returned exit {rc}", file=sys.stderr)
        return rc

    # Update manifest.
    manifest_path = out_dir / "manifest.json"
    manifest: Dict = {"shards": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    bin_path = shard_path.with_suffix(".bin") if shard_path.suffix != ".bin" else shard_path
    meta_path = bin_path.parent / f"{bin_path.stem}.meta.json"
    if not meta_path.exists():
        print(f"[run] WARN: meta sidecar {meta_path} missing", file=sys.stderr)
    else:
        meta = json.loads(meta_path.read_text())
        # de-dup by file path
        manifest["shards"] = [
            s for s in manifest.get("shards", [])
            if s.get("bin") != str(bin_path.relative_to(REPO_ROOT))
        ]
        manifest["shards"].append({
            "bin": str(bin_path.relative_to(REPO_ROOT)),
            "meta": str(meta_path.relative_to(REPO_ROOT)),
            "records": meta.get("records", 0),
            "samples": meta.get("samples_processed", 0),
            "hidden_dim": meta.get("hidden_dim", 0),
            "model_id": meta.get("model_id", ""),
            "profile_id": meta.get("profile_id"),
        })
    total_records = sum(s["records"] for s in manifest["shards"])
    total_samples = sum(s["samples"] for s in manifest["shards"])
    manifest["total_records"] = total_records
    manifest["total_samples"] = total_samples
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"[run] manifest @ {manifest_path}: {len(manifest['shards'])} shard(s), "
        f"{total_records} records, {total_samples} samples",
        file=sys.stderr,
    )
    return 0


# -----------------------------------------------------------------------------
# .bin reader — used by to-parquet + inspect
# -----------------------------------------------------------------------------

def _read_header(f) -> int:
    """Return hidden_dim. Raises on bad magic/version."""
    header = f.read(HEADER_SIZE)
    if len(header) < HEADER_SIZE:
        raise ValueError("file too short for DCAP header")
    if header[:4] != MAGIC:
        raise ValueError(f"bad magic {header[:4]!r}, expected {MAGIC!r}")
    version = struct.unpack("<I", header[4:8])[0]
    if version != VERSION:
        raise ValueError(f"unsupported DCAP version {version}, expected {VERSION}")
    hidden_dim = struct.unpack("<I", header[8:12])[0]
    return hidden_dim


def _iter_records(path: pathlib.Path):
    """Yield (sample_id, pos, prev_token, next_token, hidden_bytes_f16)."""
    with open(path, "rb") as f:
        hd = _read_header(f)
        hidden_bytes = hd * 2
        while True:
            len_buf = f.read(2)
            if not len_buf:
                return
            if len(len_buf) < 2:
                raise ValueError("truncated record (id_len)")
            (id_len,) = struct.unpack("<H", len_buf)
            sid = f.read(id_len).decode("utf-8")
            (pos, prev_tok, next_tok) = struct.unpack("<III", f.read(12))
            hb = f.read(hidden_bytes)
            if len(hb) != hidden_bytes:
                raise ValueError(f"truncated hidden at sample {sid} pos {pos}")
            yield sid, pos, prev_tok, next_tok, hb


# -----------------------------------------------------------------------------
# to-parquet — convert .bin to parquet for HF datasets consumption
# -----------------------------------------------------------------------------

def cmd_to_parquet(args: argparse.Namespace) -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print(
            "ERROR: pyarrow not found. Install with\n"
            "  pip install pyarrow",
            file=sys.stderr,
        )
        return 2

    src = pathlib.Path(args.src)
    if not src.exists():
        print(f"ERROR: source .bin {src} not found", file=sys.stderr)
        return 2

    with open(src, "rb") as f:
        hidden_dim = _read_header(f)
    print(f"[to-parquet] {src.name} hidden_dim={hidden_dim}", file=sys.stderr)

    # STREAMING write — bounded memory regardless of shard size.
    # Prior implementation held the FULL record set (all hiddens) in Python
    # memory before writing → ~7.5 GB for our current shard, OOM'd when 4
    # concurrent conversions ran in parallel (~30 GB total need vs 18 GB).
    # Now: accumulate ROW_GROUP_SIZE records into a small list, flush to
    # ParquetWriter as a record batch, repeat. Peak Python memory: ~120 MB
    # per conversion (row-group size 32K × 4KB hidden).
    ROW_GROUP_SIZE = 32_000

    schema = pa.schema(
        [
            pa.field("sample_id", pa.string()),
            pa.field("pos", pa.int32()),
            pa.field("prev_token", pa.int32()),
            pa.field("next_token", pa.int32()),
            pa.field("hidden_f16", pa.binary()),
        ],
        # store hidden_dim + dtype in schema metadata so a downstream
        # training loader can shape the binary blob without a sidecar.
        metadata={
            b"hidden_dim": str(hidden_dim).encode(),
            b"hidden_dtype": b"float16",
            b"dcap_version": str(VERSION).encode(),
        },
    )

    dst = pathlib.Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Write to a tmp path then atomically rename — concurrent readers
    # (training loaders, other tools) see either the old file or the new,
    # never a partial.
    tmp_dst = dst.with_suffix(dst.suffix + ".tmp")

    writer = pq.ParquetWriter(str(tmp_dst), schema, compression=args.compression)

    buf_sids: List[str] = []
    buf_pos: List[int] = []
    buf_prev: List[int] = []
    buf_next: List[int] = []
    buf_hidden: List[bytes] = []
    n = 0

    def _flush():
        nonlocal buf_sids, buf_pos, buf_prev, buf_next, buf_hidden
        if not buf_sids:
            return
        batch = pa.record_batch(
            {
                "sample_id": pa.array(buf_sids, type=pa.string()),
                "pos": pa.array(buf_pos, type=pa.int32()),
                "prev_token": pa.array(buf_prev, type=pa.int32()),
                "next_token": pa.array(buf_next, type=pa.int32()),
                "hidden_f16": pa.array(buf_hidden, type=pa.binary()),
            },
            schema=schema,
        )
        writer.write_batch(batch)
        # Drop refs so Python frees the underlying bytes objects.
        buf_sids = []
        buf_pos = []
        buf_prev = []
        buf_next = []
        buf_hidden = []

    try:
        for sid, pos, p, nx, hb in _iter_records(src):
            buf_sids.append(sid)
            buf_pos.append(pos)
            buf_prev.append(p)
            buf_next.append(nx)
            buf_hidden.append(hb)
            n += 1
            if len(buf_sids) >= ROW_GROUP_SIZE:
                _flush()
        _flush()
    finally:
        writer.close()

    # Atomic rename: tmp → dst.
    import os as _os
    _os.replace(tmp_dst, dst)
    print(f"[to-parquet] {n} record(s) written via streaming write "
          f"(row-group {ROW_GROUP_SIZE})", file=sys.stderr)
    sz = dst.stat().st_size
    print(
        f"[to-parquet] wrote {dst} ({sz:,} bytes, compression={args.compression})",
        file=sys.stderr,
    )
    return 0


# -----------------------------------------------------------------------------
# inspect — peek inside a .bin shard, optionally tokenizer-decode samples
# -----------------------------------------------------------------------------

def cmd_inspect(args: argparse.Namespace) -> int:
    src = pathlib.Path(args.src)
    if not src.exists():
        print(f"ERROR: source .bin {src} not found", file=sys.stderr)
        return 2

    with open(src, "rb") as f:
        hidden_dim = _read_header(f)

    sids_seen: List[str] = []
    n_records = 0
    by_sample: Dict[str, List[Tuple[int, int, int, bytes]]] = {}
    for sid, pos, prev_tok, next_tok, hb in _iter_records(src):
        n_records += 1
        if sid not in by_sample:
            sids_seen.append(sid)
            by_sample[sid] = []
        by_sample[sid].append((pos, prev_tok, next_tok, hb))

    print(f"file: {src}")
    print(f"  hidden_dim       : {hidden_dim}")
    print(f"  total records    : {n_records}")
    print(f"  unique samples   : {len(sids_seen)}")
    if sids_seen:
        print(f"  first sample_id  : {sids_seen[0]}")
        print(f"  last  sample_id  : {sids_seen[-1]}")
    if by_sample:
        lens = [len(v) for v in by_sample.values()]
        print(
            f"  records/sample   : min={min(lens)} max={max(lens)} "
            f"mean={sum(lens)/len(lens):.1f}"
        )
    # hidden f16 sanity: read one record, compute a few stats
    if n_records > 0 and args.hidden_stats:
        try:
            import numpy as np
        except ImportError:
            print("(skipping hidden stats — numpy not installed)")
        else:
            first_sid = sids_seen[0]
            _, _, _, hb = by_sample[first_sid][0]
            arr = np.frombuffer(hb, dtype=np.float16)
            print(
                f"  hidden[0] stats  : len={arr.size}, min={arr.min():.4f}, "
                f"max={arr.max():.4f}, mean={arr.mean():.4f}, std={arr.std():.4f}"
            )

    if args.decode_n > 0:
        # Sanity: decode the prev/next tokens of N random sample windows
        # back to text via the model tokenizer. This catches off-by-one
        # bugs, dtype mismatches, or sample_id corruption.
        try:
            from transformers import AutoTokenizer
        except ImportError:
            print(
                "WARN: --decode-n N specified but `transformers` not available; "
                "skipping decode sanity check.",
                file=sys.stderr,
            )
            return 0
        tok_id = args.tokenizer or "deepseek-ai/DeepSeek-V2-Lite-Chat"
        print(f"\n[decode] using tokenizer {tok_id}", file=sys.stderr)
        try:
            tok = AutoTokenizer.from_pretrained(tok_id)
        except Exception as e:
            print(f"WARN: tokenizer load failed ({e}); skipping decode.", file=sys.stderr)
            return 0
        rng = random.Random(args.decode_seed)
        picked = rng.sample(sids_seen, min(args.decode_n, len(sids_seen)))
        print(f"\n[decode] {len(picked)} random sample windows:")
        for sid in picked:
            recs = by_sample[sid]
            # Reconstruct token stream: prev_token at pos i ⇒ tokens[i] = prev,
            # tokens[i+1] = next. Stitch: tokens = [recs[0].prev, recs[0].next,
            # recs[1].next, recs[2].next, ...]
            recs_sorted = sorted(recs, key=lambda r: r[0])
            tok_ids: List[int] = []
            if recs_sorted:
                tok_ids.append(recs_sorted[0][1])
            for _pos, _prev, nxt, _hb in recs_sorted:
                tok_ids.append(nxt)
            text = tok.decode(tok_ids, skip_special_tokens=False)
            preview = text.replace("\n", "\\n")[: args.decode_chars]
            print(f"  {sid} ({len(tok_ids)} tokens): {preview!r}")
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        prog="capture_hidden.py",
        description="Path-to-90 C2 — capture (hidden, next_token) tuples.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # prep
    pp = sub.add_parser("prep", help="Slice an HF chat dataset → JSONL")
    pp.add_argument("--out", default=str(REPO_ROOT / "tests/data/ultrachat_smoke.jsonl"))
    pp.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    pp.add_argument("--split", default="train_sft")
    pp.add_argument("--streaming", action="store_true",
                    help="Stream from HF (avoids full download). Recommended for UltraChat.")
    pp.add_argument("--n", type=int, default=10, help="Number of samples to keep.")
    pp.add_argument("--seed", type=int, default=20260515)
    pp.add_argument("--min-chars", type=int, default=200)
    pp.add_argument("--max-chars", type=int, default=2000)
    pp.add_argument("--id-prefix", default="ultrachat")
    pp.add_argument("--force", action="store_true")
    pp.set_defaults(func=cmd_prep)

    # run
    pr = sub.add_parser("run", help="Invoke dismantle capture-hidden, sharded.")
    pr.add_argument("--binary", default=str(DEFAULT_BINARY))
    pr.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    pr.add_argument("--samples", required=True)
    pr.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    pr.add_argument("--shard-name", default="shard_000.bin")
    pr.add_argument("--max-tokens", type=int, default=128)
    pr.add_argument("--max-samples", type=int, default=0)
    pr.add_argument("--resume", action="store_true")
    pr.add_argument("--no-lm-head", action="store_true",
                    help="Skip lm_head + argmax (~10-15% faster). "
                         "Use with teacher-forced training data where "
                         "next_token comes from the source corpus.")
    pr.add_argument("--kernel-profile", default=None)
    pr.set_defaults(func=cmd_run)

    # to-parquet
    pq_ = sub.add_parser("to-parquet", help="Convert .bin → parquet")
    pq_.add_argument("--src", required=True, help="Path to .bin shard.")
    pq_.add_argument("--dst", required=True, help="Path to write .parquet.")
    pq_.add_argument("--compression", default="zstd",
                     choices=["zstd", "snappy", "gzip", "none"])
    pq_.set_defaults(func=cmd_to_parquet)

    # inspect
    pi = sub.add_parser("inspect", help="Peek inside a .bin shard.")
    pi.add_argument("--src", required=True)
    pi.add_argument("--decode-n", type=int, default=0,
                    help="Decode N random sample-windows back to text via tokenizer.")
    pi.add_argument("--decode-seed", type=int, default=20260515)
    pi.add_argument("--decode-chars", type=int, default=160)
    pi.add_argument("--tokenizer", default=None,
                    help="HF tokenizer id (default deepseek-ai/DeepSeek-V2-Lite-Chat).")
    pi.add_argument("--hidden-stats", action="store_true",
                    help="Print hidden vector min/max/mean/std (requires numpy).")
    pi.set_defaults(func=cmd_inspect)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
