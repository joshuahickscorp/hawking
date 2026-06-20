"""Export a trained RWKV7Model state_dict back to HF safetensors layout, then to
GGUF for on-device serving via dismantle.

Inverse of rwkv7_load_weights.py: maps the pure-torch param names back to the
fla `RWKV7ForCausalLM` safetensors names (no transposes — same [out,in] layout),
copies the aux files the GGUF converter needs (config.json, the World vocab,
modeling/tokenizer shims), then runs convert_hf_to_gguf.py --outtype f16.

Usage:
    python rwkv7_export_hf.py \
      --state-dict artifacts/rwkv7_posttrain/sft_out/final/state_dict.pt \
      --hf-dir models/rwkv7-g1-04-hf \
      --out-dir artifacts/rwkv7_posttrain/sft_hf \
      --gguf models/rwkv7-g1-04-sft-f16.gguf
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from rwkv7_torch_model import RWKV7Config  # noqa: E402


def torch_key_to_hf(k: str, cfg: RWKV7Config) -> str:
    """Map a RWKV7Model state_dict key to its HF safetensors name."""
    if k == "embeddings.weight":
        return "model.embeddings.weight"
    if k == "norm_w":
        return "model.norm.weight"
    if k == "norm_b":
        return "model.norm.bias"
    if k == "lm_head.weight":
        return "lm_head.weight"
    assert k.startswith("layers."), k
    _, li, rest = k.split(".", 2)
    p = f"model.layers.{li}."
    simple = {
        "pre_norm_w": "pre_norm.weight", "pre_norm_b": "pre_norm.bias",
        "attn_norm_w": "attn_norm.weight", "attn_norm_b": "attn_norm.bias",
        "ffn_norm_w": "ffn_norm.weight", "ffn_norm_b": "ffn_norm.bias",
    }
    if rest in simple:
        return p + simple[rest]
    if rest.startswith("attn."):
        a = rest[len("attn."):]
        amap = {
            "r_proj.weight": "attn.r_proj.weight", "k_proj.weight": "attn.k_proj.weight",
            "v_proj.weight": "attn.v_proj.weight", "o_proj.weight": "attn.o_proj.weight",
            "w1": "attn.w_lora.lora.0.weight", "w2": "attn.w_lora.lora.2.weight", "w0": "attn.w_lora.lora.2.bias",
            "a1": "attn.a_lora.lora.0.weight", "a2": "attn.a_lora.lora.2.weight", "a0": "attn.a_lora.lora.2.bias",
            "v1": "attn.v_lora.lora.0.weight", "v2": "attn.v_lora.lora.2.weight", "v0": "attn.v_lora.lora.2.bias",
            "g1": "attn.g_lora.lora.0.weight", "g2": "attn.g_lora.lora.2.weight",
            "k_k": "attn.k_k", "k_a": "attn.k_a", "r_k": "attn.r_k",
            "g_norm_w": "attn.g_norm.weight", "g_norm_b": "attn.g_norm.bias",
        }
        if a.startswith("x_"):
            return p + f"attn.{a}"  # reshaped to [1,1,n] below
        return p + amap[a]
    if rest.startswith("ffn."):
        f = rest[len("ffn."):]
        if f == "x_k":
            return p + "ffn.x_k"
        return p + f"ffn.{f}"  # key.weight / value.weight
    raise KeyError(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dict", required=True)
    ap.add_argument("--hf-dir", default=str(ROOT / "models/rwkv7-g1-04-hf"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gguf", default="", help="final serving GGUF (Q4_K_M by default)")
    ap.add_argument("--converter", default=str(ROOT / "tools/strand/tools/gguf/convert_hf_to_gguf.py"))
    ap.add_argument("--quantize", default="Q4_K_M",
                    help="llama-quantize type for the serving GGUF; '' keeps raw f16 "
                         "(NOTE: dismantle's RWKV serving path is broken on raw f16 — use Q4_K_M)")
    ap.add_argument("--llama-quantize", default="/opt/homebrew/bin/llama-quantize")
    ap.add_argument("--no-gguf", action="store_true", help="write HF dir only, skip GGUF")
    args = ap.parse_args()

    cfg = RWKV7Config()
    sd = torch.load(args.state_dict, map_location="cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Re-key to HF names; x_* lerps go back to [1,1,n].
    hf_sd: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        hk = torch_key_to_hf(k, cfg)
        t = v.to(torch.float32).contiguous()
        if ".attn.x_" in hk:
            t = t.reshape(1, 1, -1).contiguous()
        hf_sd[hk] = t
    save_file(hf_sd, str(out_dir / "model.safetensors"))
    print(f"[export] wrote {len(hf_sd)} tensors -> {out_dir/'model.safetensors'}")

    # Aux files the converter + a valid HF dir need.
    hf_dir = Path(args.hf_dir)
    for fn in ("config.json", "rwkv_vocab_v20230424.txt", "modeling_rwkv7.py",
               "hf_rwkv_tokenizer.py", "tokenizer_config.json", "generation_config.json",
               "special_tokens_map.json", "added_tokens.json"):
        src = hf_dir / fn
        if src.exists():
            shutil.copy2(src, out_dir / fn)
    print(f"[export] copied aux files into {out_dir}")

    if args.no_gguf:
        return
    final = args.gguf or str(out_dir / "model-Q4_K_M.gguf")
    # Step 1: HF safetensors -> f16 GGUF (intermediate).
    f16 = str(out_dir / "model-f16.gguf")
    cmd = [sys.executable, args.converter, str(out_dir), "--outfile", f16, "--outtype", "f16"]
    print(f"[export] converting -> {f16}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[export] CONVERT FAILED:\n" + (r.stderr[-2000:] or ""))
        sys.exit(1)
    # Step 2: f16 -> Q4_K_M (dismantle's RWKV serving path needs Q4_K, not raw f16).
    if not args.quantize:
        print(f"[export] GGUF ready (raw f16): {f16}")
        return
    qcmd = [args.llama_quantize, f16, final, args.quantize]
    print(f"[export] quantizing -> {final} ({args.quantize})")
    r = subprocess.run(qcmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[export] QUANTIZE FAILED:\n" + (r.stderr[-2000:] or r.stdout[-2000:] or ""))
        sys.exit(1)
    print(f"[export] serving GGUF ready: {final}")


if __name__ == "__main__":
    main()
