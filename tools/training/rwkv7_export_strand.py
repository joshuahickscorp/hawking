"""Export a QAT-trained RWKV-7 checkpoint to STR2/TQ format for low-bit serving.

Loads a state_dict.pt produced by the SFT or QAT path, maps projection weights
to GGUF names, writes a float32 safetensors staging file, then shells out to the
strand `quantize-model` binary to produce the packed STR2 reconstruction
(.recon.safetensors) and TQ tile-quantized output (.tq.safetensors).

Supports two source modes detected from run_config.json:
  - strand-pv: use the `.base` reconstruction buffer (not `.weight` deltas)
  - fake-quant or absent: apply round-to-nearest via quant_fn, then detach

Usage:
    python rwkv7_export_strand.py \\
      --checkpoint artifacts/rwkv7_posttrain/dpo_out/final \\
      --out artifacts/rwkv7_posttrain/strand_export \\
      --bits 2 --l 7 \\
      --strand-bin target/release/quantize-model \\
      --strand-flags "--tail-biting --affine-min off"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# ---------------------------------------------------------------------------
# GGUF name mapping
# ---------------------------------------------------------------------------

# Projection weights only — non-projection params (norms, lerp coeffs, LoRA,
# x_*, etc.) pass through to HF export unchanged and are not STR2-quantised.
_PROJ_PATTERNS = [
    ("attn.r_proj.weight", "time_mix_receptance.weight"),
    ("attn.k_proj.weight", "time_mix_key.weight"),
    ("attn.v_proj.weight", "time_mix_value.weight"),
    ("attn.o_proj.weight", "time_mix_output.weight"),
    ("ffn.key.weight",     "channel_mix_key.weight"),
    ("ffn.value.weight",   "channel_mix_value.weight"),
]

TORCH_TO_GGUF_GLOBAL = {
    "lm_head.weight": "output.weight",
}


def make_torch_to_gguf(n_layers: int) -> dict[str, str]:
    """Build the full torch-name → GGUF-name mapping for *n_layers* layers."""
    mapping: dict[str, str] = {}
    for i in range(n_layers):
        for torch_suffix, gguf_suffix in _PROJ_PATTERNS:
            mapping[f"layers.{i}.{torch_suffix}"] = f"blk.{i}.{gguf_suffix}"
    mapping.update(TORCH_TO_GGUF_GLOBAL)
    return mapping


# ---------------------------------------------------------------------------
# Quantisation helper (fake-quant path)
# ---------------------------------------------------------------------------

def quant_fn(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric min-max uniform fake-quantise to *bits* bits, round-to-nearest."""
    if bits <= 0:
        return weight
    qmax = (1 << (bits - 1)) - 1
    scale = weight.abs().max() / qmax
    if scale == 0:
        return weight
    return (weight / scale).round().clamp(-qmax - 1, qmax).mul(scale)


# ---------------------------------------------------------------------------
# Self-contained safetensors writer (no safetensors dep at write time)
# ---------------------------------------------------------------------------

def write_safetensors(
    path: str | Path,
    tensors: dict[str, "np.ndarray | torch.Tensor"],
) -> None:
    """Write *tensors* as a safetensors file with F32 dtype.

    Format:
      [8 bytes: uint64 header_len]
      [header_len bytes: JSON header, zero-padded to 8-byte alignment]
      [tensor bytes: contiguous, ordered by header insertion]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Canonicalize to numpy float32.
    arrays: dict[str, np.ndarray] = {}
    for name, t in tensors.items():
        if isinstance(t, torch.Tensor):
            arr = t.detach().float().cpu().contiguous().numpy()
        else:
            arr = np.ascontiguousarray(t, dtype=np.float32)
        arrays[name] = arr

    # Build header.
    header: dict[str, dict] = {}
    offset = 0
    for name, arr in arrays.items():
        nbytes = arr.nbytes
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    header_json = json.dumps(header, separators=(",", ":"))
    header_bytes = header_json.encode("utf-8")
    # Pad to 8-byte alignment with spaces (valid JSON whitespace).
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes = header_bytes + b" " * pad

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for arr in arrays.values():
            f.write(arr.tobytes())


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------

def _resolve_checkpoint(checkpoint: str) -> tuple[Path, Path | None]:
    """Return (state_dict_path, run_config_path_or_None)."""
    p = Path(checkpoint)
    if p.is_dir():
        sd_path = p / "state_dict.pt"
        if not sd_path.exists():
            # fall back to last checkpoint saved by sft_torch (step_*.pt)
            pts = sorted(p.glob("step_*.pt"))
            if pts:
                sd_path = pts[-1]
            else:
                raise FileNotFoundError(f"No state_dict.pt or step_*.pt in {p}")
        cfg_path = p / "run_config.json"
        return sd_path, cfg_path if cfg_path.exists() else None
    else:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        cfg_path = p.parent / "run_config.json"
        return p, cfg_path if cfg_path.exists() else None


def _detect_pv_mode(cfg_path: Path | None) -> bool:
    """Return True if run_config.json says quant == "strand-pv"."""
    if cfg_path is None:
        return False
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get("quant") == "strand-pv"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-tensor bits from mp-config
# ---------------------------------------------------------------------------

def _build_bits_map(mp_config: str | None, default_bits: int) -> list[tuple[str, int]]:
    """Return [(pattern, bits), ...] from mp_config JSON, or a default catch-all."""
    if not mp_config:
        return [(".*", default_bits)]
    with open(mp_config) as f:
        rules = json.load(f)  # [{pattern, bits}, ...]
    return [(r["pattern"], int(r["bits"])) for r in rules]


def _bits_for(gguf_name: str, rules: list[tuple[str, int]], default: int) -> int:
    import re
    for pattern, bits in rules:
        if re.search(pattern, gguf_name):
            return bits
    return default


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export RWKV-7 QAT checkpoint to STR2/TQ format."
    )
    ap.add_argument("--checkpoint", required=True,
                    help="Run dir with state_dict.pt OR path to a .pt file")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--bits", type=int, default=1,
                    help="Default quantisation bits (default: 1)")
    ap.add_argument("--l", type=int, default=7, dest="l_param",
                    help="Trellis L parameter; 0 = use quantize-model default")
    ap.add_argument("--strand-bin", default="target/release/quantize-model",
                    help="Path to the strand quantize-model binary")
    ap.add_argument("--strand-flags", default="",
                    help='Extra flags for quantize-model, e.g. "--tail-biting --affine-min off"')
    ap.add_argument("--mp-config", default=None,
                    help="JSON file [{pattern, bits}] for per-tensor bit overrides")
    ap.add_argument("--skip-quantize", action="store_true",
                    help="Write safetensors only; skip quantize-model invocation")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be exported and exit without writing")
    ap.add_argument("--n-layers", type=int, default=24,
                    help="RWKV-7 layer count (default: 24)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    strand_bin = Path(args.strand_bin)
    if not strand_bin.is_absolute():
        strand_bin = ROOT / strand_bin

    # -- resolve checkpoint ---------------------------------------------------
    sd_path, cfg_path = _resolve_checkpoint(args.checkpoint)
    pv_mode = _detect_pv_mode(cfg_path)
    print(f"[export] checkpoint : {sd_path}")
    print(f"[export] run_config : {cfg_path or '(none)'}")
    print(f"[export] strand-pv  : {pv_mode}")

    # -- build name mapping ---------------------------------------------------
    mapping = make_torch_to_gguf(args.n_layers)
    bits_rules = _build_bits_map(args.mp_config, args.bits)

    if args.dry_run:
        print(f"\n[dry-run] would export {len(mapping)} tensors:")
        for tname, gname in sorted(mapping.items()):
            b = _bits_for(gname, bits_rules, args.bits)
            src = "(base buffer)" if pv_mode else f"(fake-quant bits={b})"
            print(f"  {tname}  ->  {gname}  {src}")
        print(f"\n[dry-run] safetensors -> {out_dir/'rwkv7-lowbit-proj.safetensors'}")
        if not args.skip_quantize:
            print(f"[dry-run] quantize-model binary: {strand_bin}")
        return

    # -- load state_dict ------------------------------------------------------
    print(f"[export] loading {sd_path} ...")
    sd: dict[str, torch.Tensor] = torch.load(str(sd_path), map_location="cpu")

    # -- extract + remap tensors ----------------------------------------------
    out_tensors: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    total_params = 0

    for torch_name, gguf_name in sorted(mapping.items()):
        if pv_mode:
            base_name = torch_name.replace(".weight", ".base")
            if base_name in sd:
                t = sd[base_name].float().contiguous()
            elif torch_name in sd:
                # base buffer not found — fall back to weight (degraded)
                print(f"[export] WARNING: .base not found for {torch_name}, using .weight")
                t = sd[torch_name].float().contiguous()
            else:
                missing.append(torch_name)
                continue
        else:
            if torch_name not in sd:
                missing.append(torch_name)
                continue
            b = _bits_for(gguf_name, bits_rules, args.bits)
            t = quant_fn(sd[torch_name], b).detach().float().contiguous()

        out_tensors[gguf_name] = t
        total_params += t.numel()

    if missing:
        print(f"[export] WARNING: {len(missing)} keys not found in state_dict:")
        for m in missing[:10]:
            print(f"  {m}")

    n_tensors = len(out_tensors)
    effective_bpw = args.bits  # before strand trellis compression
    print(f"[export] tensors={n_tensors}  params={total_params/1e6:.2f}M  "
          f"bits={args.bits}  est_bpw={effective_bpw}")

    # -- write safetensors ----------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    proj_st = out_dir / "rwkv7-lowbit-proj.safetensors"
    print(f"[export] writing {proj_st} ...")
    write_safetensors(proj_st, out_tensors)

    # -- sha256 for manifest --------------------------------------------------
    h = hashlib.sha256()
    with open(proj_st, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    source_hash = h.hexdigest()

    # -- write manifest -------------------------------------------------------
    manifest = {
        "gguf_names": list(out_tensors.keys()),
        "shapes": {k: list(v.shape) for k, v in out_tensors.items()},
        "bits": args.bits,
        "l": args.l_param,
        "strand_flags": args.strand_flags,
        "pv_mode": pv_mode,
        "n_tensors": n_tensors,
        "total_params": total_params,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_hash": source_hash,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[export] manifest   -> {manifest_path}")

    if args.skip_quantize:
        print("[export] --skip-quantize: done (no quantize-model call)")
        _print_summary(n_tensors, total_params, args.bits)
        return

    # -- invoke strand quantize-model -----------------------------------------
    if not strand_bin.exists():
        print(f"[export] ERROR: strand binary not found: {strand_bin}")
        print("[export] Build it with: cargo build --release -p quantize-model")
        print("[export] Or pass --skip-quantize to write safetensors only")
        sys.exit(1)

    recon_path = out_dir / "rwkv7-lowbit-proj.recon.safetensors"
    tq_path    = out_dir / "rwkv7-lowbit-proj.tq.safetensors"

    cmd = [
        str(strand_bin),
        "--in",  str(proj_st),
        "--out", str(recon_path),
        "--bits", str(args.bits),
        "--packed-v2-out", str(tq_path),
    ]
    if args.l_param > 0:
        cmd += ["--l", str(args.l_param)]
    if args.strand_flags.strip():
        cmd += args.strand_flags.split()

    print(f"[export] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[export] ERROR: quantize-model exited {result.returncode}")
        sys.exit(result.returncode)

    print(f"[export] recon  -> {recon_path}")
    print(f"[export] tq     -> {tq_path}")
    _print_summary(n_tensors, total_params, args.bits)


def _print_summary(n_tensors: int, total_params: int, bits: int) -> None:
    param_bytes_f32 = total_params * 4
    param_bytes_lowbit = total_params * bits / 8
    print(
        f"\n[export] summary: {n_tensors} tensors | "
        f"{total_params/1e6:.2f}M params | "
        f"f32={param_bytes_f32/1e6:.1f}MB | "
        f"est {bits}bpw={param_bytes_lowbit/1e6:.1f}MB | "
        f"compression {param_bytes_f32/max(param_bytes_lowbit,1):.1f}x"
    )


if __name__ == "__main__":
    main()
