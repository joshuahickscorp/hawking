#!/usr/bin/env python3
"""strand-qat.py — quantization-aware fine-tuning harness for STRAND low-bit (Act 2).

PTQ on STRAND floors near ~80 PPL at 2-bit on Qwen2.5-0.5B (the RHT closes the field's
Hessian-objective lever; see will.md §4). The ~80 -> ~bf16 jump is TRAINING's. This harness
proves the mechanism: load a pretrained CausalLM, wrap its projection Linears with a
fake-quant (straight-through estimator), fine-tune on WikiText-2 train, and eval PPL on
WikiText-2 test with the SAME protocol as tools/strand/scripts/strand-7b-ppl.sh
(non-overlapping ctx
windows, exp(sum nll / sum tok)) so the number is directly comparable to the PTQ canon.

Quantizers (all STE):
  uniform  — per-output-channel symmetric uniform at --bits (harness validation).
  ternary  — BitNet b1.58: {-1,0,+1} with scale = mean(|w|) per output channel.
  (strand-trellis comes next: periodic re-quant via the Rust encoder + STE.)

The fine-tuned fp32 shadow weights are saved (--save) so the REAL STRAND trellis quantizer
can be run on the *trained* weights afterward (the deployment artifact stays bit-exact STRAND).

Usage:
    strand-qat.py --model scratch/qwen-05b --quant uniform --bits 2 \
        --steps 300 --lr 2e-5 --ctx 1024 --train-chunks 256 --eval-chunks 32 \
        [--kd] [--device mps] [--save scratch/qwen-05b/qat-2bit.pt]
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJ_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------------------
# Quantizers (straight-through estimator: forward quantizes, backward is identity)
# ---------------------------------------------------------------------------
def quant_uniform(w, bits):
    """Per-output-channel (row) symmetric uniform. w: [out, in]."""
    qmax = 2 ** (bits - 1) - 1          # b=2->1, b=3->3, b=4->7
    qmin = -(2 ** (bits - 1))           # b=2->-2
    s = w.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / max(qmax, 1)
    q = torch.clamp(torch.round(w / s), qmin, qmax)
    wq = q * s
    return w + (wq - w).detach()        # STE

def quant_ternary(w, bits=None):
    """BitNet b1.58: ternary {-1,0,1}, per-output-channel scale = mean(|w|)."""
    s = w.detach().abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    q = torch.clamp(torch.round(w / s), -1, 1)
    wq = q * s
    return w + (wq - w).detach()        # STE

QUANTIZERS = {"uniform": quant_uniform, "ternary": quant_ternary, "strand": None}
# "strand" = PV mode: forward against an EXTERNAL recon buffer refreshed by the real
# Rust encoder (quantize-model) every --requant-every steps. The proxy-transfer
# verdict (3,013 vs 80.7, will.md §4) killed every train-with-a-different-quantizer
# shortcut: you train through what you ship, so the forward IS the deployment recon.


class QuantLinear(nn.Module):
    """Wraps an nn.Linear: forward quantizes the (trainable fp32) shadow weight via STE.
    quant_fn=None (strand mode): forward = STE against `wq_ext`, a buffer holding the
    real STRAND recon of the shadow, refreshed by periodic re-quantization."""
    def __init__(self, lin, quant_fn, bits):
        super().__init__()
        self.weight = nn.Parameter(lin.weight.data.detach().clone().float())
        self.bias = (nn.Parameter(lin.bias.data.detach().clone().float())
                     if lin.bias is not None else None)
        self.quant_fn = quant_fn
        self.bits = bits
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        if quant_fn is None:
            # base = recon - w_anchor (set at each requant). Forward = base + w:
            # exact recon at requant instants, recon + learned drift between them —
            # TRUE gradients every step. (Frozen STE vs a fixed recon gave the shadows
            # 75 steps of feedback-free drift -> requant exploded to 59e9 PPL.)
            self.register_buffer("base", torch.zeros_like(self.weight.data))

    def forward(self, x):
        if self.quant_fn is None:
            wq = self.base + self.weight
        else:
            wq = self.quant_fn(self.weight, self.bits)
        return F.linear(x, wq.to(x.dtype), None if self.bias is None
                        else self.bias.to(x.dtype))


def wrap_proj_linears(model, quant_fn, bits):
    """Replace every projection nn.Linear (q/k/v/o/gate/up/down_proj) with a QuantLinear.
    Embeddings, lm_head, and norms stay full precision."""
    n = 0
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and child_name in PROJ_SUFFIXES:
                setattr(parent, child_name, QuantLinear(child, quant_fn, bits))
                n += 1
    return n


# ---------------------------------------------------------------------------
# Data — WikiText-2, tokenized as "\n\n".join(text), non-overlapping ctx windows
# (identical preprocessing to strand-7b-ppl.sh, so train/eval share the tokenization).
# ---------------------------------------------------------------------------
def load_wikitext_ids(tok, split):
    from datasets import load_dataset
    ds = None
    for ds_id in ("wikitext", "Salesforce/wikitext"):
        try:
            ds = load_dataset(ds_id, "wikitext-2-raw-v1", split=split)
            break
        except Exception:
            continue
    if ds is None:
        raise SystemExit(f"[qat] failed to load wikitext-2-raw-v1 split={split}")
    return tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]


def chunks(ids, ctx, limit):
    n = ids.shape[0] // ctx
    if limit > 0:
        n = min(n, limit)
    return [ids[i * ctx:(i + 1) * ctx] for i in range(n)]


def shadow_digest(m):
    """sha256 of a QuantLinear's fp32 shadow weight bytes (CPU, deterministic)."""
    import hashlib
    t = m.weight.detach().to("cpu", torch.float32).contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def forward_digest(m):
    """sha256 of a QuantLinear's effective delta-forward weight (base + w), fp32 bytes."""
    import hashlib
    t = (m.base + m.weight).detach().to("cpu", torch.float32).contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


@torch.no_grad()
def pv_capture_frozen(model, args):
    """Selective PV invariant capture (design §4.2/§4.3): after the init requant, record
    per-FROZEN-tensor digests of (a) the shadow weight and (b) the delta-forward weight
    base+w == recon. Frozen tensors must keep BOTH for the rest of the run."""
    if not args.pv_tensors:
        return
    args._pv_frozen = {}
    for name, m in model.named_modules():
        if isinstance(m, QuantLinear) and not m.weight.requires_grad:
            args._pv_frozen[name] = (shadow_digest(m), forward_digest(m))
    print(f"[qat] selective-PV: captured digests for {len(args._pv_frozen)} frozen tensors",
          flush=True)


@torch.no_grad()
def pv_verify_frozen(model, args, tag):
    """Assert the frozen invariant: shadows unchanged (hash-equal) AND base+w still equals
    the init recon (hash-equal). Any drift = a bug in the freeze or the requant scope."""
    frozen = getattr(args, "_pv_frozen", None)
    if not frozen:
        return
    bad = []
    for name, m in model.named_modules():
        if isinstance(m, QuantLinear) and name in frozen:
            w0, f0 = frozen[name]
            if shadow_digest(m) != w0:
                bad.append((name, "shadow"))
            elif forward_digest(m) != f0:
                bad.append((name, "forward(base+w)"))
    assert not bad, f"[qat] FROZEN-INVARIANT VIOLATED at {tag}: {bad[:4]}"
    print(f"[qat] frozen-verify {tag}: {len(frozen)} frozen tensors hash-equal "
          f"(shadow + base+w==recon) OK", flush=True)


@torch.no_grad()
def strand_requant(model, args, tag):
    """PV refresh: dump shadow proj weights -> real Rust STRAND encoder -> reload the
    recon into every QuantLinear.base. Training pauses; the quantizer gets the CPU.
    P3 (design §4.4): the init requant is FULL (every frozen tensor's base anchors to its
    recon); subsequent requants dump + re-encode + reload ONLY the PV set — frozen recons
    cannot change (§4.2/§4.3), so re-encoding them is pure waste."""
    import subprocess
    from safetensors import safe_open
    from safetensors.torch import save_file
    t0 = time.time()
    model.eval()
    if args.device == "mps":
        torch.mps.empty_cache()
    os.makedirs(args.strand_dir, exist_ok=True)
    sel = (tag != "init") and bool(getattr(args, "pv_tensors", ""))
    if sel:
        pv_verify_frozen(model, args, tag)   # frozen invariant holds at every boundary
    # P6 (100x lever): the FULL init requant re-encodes all ~196 tensors (~2.5h, memory-bound).
    # But at init every tensor's recon == an EXISTING same-config PTQ recon -- shadows are still the
    # original weights and RHT is name-seeded, so the encode is byte-identical to re-running it. If a
    # matching recon dir is supplied, LOAD it instead of re-encoding -> init in seconds, not hours.
    # (Subsequent SELECTIVE requants still re-encode the trained PV tensors; their shadows changed.)
    if tag == "init" and getattr(args, "init_recon_dir", ""):
        import glob
        pre = sorted(glob.glob(os.path.join(args.init_recon_dir, "*.safetensors")))
        if pre:
            nrl = 0
            for rj in pre:
                with safe_open(rj, framework="pt") as f:
                    have = set(f.keys())
                    for name, m in model.named_modules():
                        if isinstance(m, QuantLinear) and (name + ".weight") in have:
                            rt = f.get_tensor(name + ".weight").to(torch.float32).to(args.device)
                            m.base.copy_(rt - m.weight.data); nrl += 1; del rt
            if args.device == "mps":
                torch.mps.empty_cache()
            secs = time.time() - t0
            print(f"[qat] requant init: {secs:.0f}s LOADED {nrl} tensors from {args.init_recon_dir} "
                  f"(100x lever: byte-identical to re-encode, no quant pass)", flush=True)
            if not hasattr(args, "_requants"):
                args._requants = []
            args._requants.append({"tag": tag, "secs": round(secs, 1),
                                   "scope": f"loaded({nrl})", "tensors": nrl, "bpw": None})
            return
        print(f"[qat] init-recon-dir {args.init_recon_dir} empty -> full requant fallback", flush=True)
    dump = os.path.join(args.strand_dir, "shadow.safetensors")
    recon = os.path.join(args.strand_dir, "recon.safetensors")
    sd = {}
    for name, m in model.named_modules():
        if isinstance(m, QuantLinear) and (not sel or m.weight.requires_grad):
            sd[name + ".weight"] = m.weight.detach().cpu().to(torch.bfloat16).contiguous()
    # P5 (wall-clock): SHARD the requant across parallel quantize-model processes. The FULL init
    # requant on one merged file was a single memory-bound process (~2.5h on 7B, only ~14/64 cores
    # effective); each tensor is quantized INDEPENDENTLY (same property the scorecard relies on when
    # it shards by model file), so splitting the dump into N size-balanced shards run in parallel is
    # RESULT-IDENTICAL and ~Nx faster wall-clock. Per-shard --threads = cores/N (no oversubscribe).
    keys = list(sd.keys())
    njobs = max(1, min(int(getattr(args, "requant_shards", 8)), len(keys), os.cpu_count() or 8))
    base_flags = []; _skip = False
    for tok in args.strand_flags.split():
        if _skip: _skip = False; continue
        if tok == "--threads": _skip = True; continue
        base_flags.append(tok)
    base_flags += ["--threads", str(max(1, (os.cpu_count() or 8) // njobs))]
    keys.sort(key=lambda k: sd[k].numel(), reverse=True)   # greedy longest-first size balance
    buckets = [[] for _ in range(njobs)]; loads = [0] * njobs
    for k in keys:
        j = loads.index(min(loads)); buckets[j].append(k); loads[j] += sd[k].numel()
    recons = []; procs = []
    for j, bucket in enumerate(buckets):
        if not bucket: continue
        dj = os.path.join(args.strand_dir, f"shadow.{j}.safetensors")
        rj = os.path.join(args.strand_dir, f"recon.{j}.safetensors")
        save_file({k: sd[k] for k in bucket}, dj, metadata={"format": "pt"})
        recons.append(rj)
        procs.append(subprocess.Popen([args.strand_bin, "--in", dj, "--out", rj] + base_flags,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    del sd
    for p in procs:
        out, _ = p.communicate()
        if p.returncode != 0:
            raise SystemExit(f"[qat] requant shard FAILED rc={p.returncode}:\n{(out or '')[-800:]}")
    nreloaded = 0
    for rj in recons:
        with safe_open(rj, framework="pt") as f:
            have = set(f.keys())
            for name, m in model.named_modules():
                if isinstance(m, QuantLinear) and (name + ".weight") in have:
                    rt = f.get_tensor(name + ".weight").to(torch.float32).to(args.device)
                    m.base.copy_(rt - m.weight.data)   # re-anchor: forward == recon right now
                    nreloaded += 1
                    del rt
    if args.device == "mps":
        torch.mps.empty_cache()   # the recon reload churns ~1.4GB of fresh blocks
    bpw = None
    try:
        bpw = json.load(open(recons[0] + ".json"))["aggregate"]["effective_bpw"]
    except Exception:
        pass
    secs = time.time() - t0
    scope = f"selective({nreloaded} tensors)" if sel else f"full({nreloaded} tensors)"
    print(f"[qat] requant {tag}: {secs:.0f}s  {scope}  bpw={bpw}", flush=True)
    if not hasattr(args, "_requants"):
        args._requants = []
    args._requants.append({"tag": tag, "secs": round(secs, 1), "scope": scope,
                           "tensors": nreloaded, "bpw": bpw})


# ---------------------------------------------------------------------------
# KD teacher-logit cache (--kd-cache <dir>, OFF BY DEFAULT — science flag).
#
# WHY: the KD teacher is FROZEN and the train chunks REPEAT across epochs/segments,
# so its logits per chunk are a pure function of the chunk — recomputing the teacher
# forward every epoch is wasted wall time. Full logits are too big to cache
# (vocab ~152k -> ~300MB/chunk fp16), so we cache the standard TOP-K distillation
# approximation: per position, the teacher's top-k (k=128 default) probabilities at
# temperature T plus the lumped tail mass m = 1 - sum(top-k).
#
# EXACT MATH (why this is principled, and exactly what it changes):
#   Full KD loss (the --kd path):    KL(P || Q) = sum_i p_i (log p_i - log q_i)
#   with P = softmax(t/T), Q = softmax(s/T), summed over rows / N * T^2
#   (identical to F.kl_div(log_softmax(s/T), softmax(t/T), "batchmean") * T^2).
#
#   Top-k bucketed form: partition the vocab into K = teacher's top-k ids and the
#   tail bucket. Define the (k+1)-event distributions
#     P' = (p_i for i in K,  m      = 1 - sum_{i in K} p_i)
#     Q' = (q_i for i in K,  q_tail = 1 - sum_{i in K} q_i)
#   and the cached loss is the EXACT KL between the bucketed distributions:
#     KL(P' || Q') = sum_{i in K} p_i (log p_i - log q_i) + m (log m - log q_tail).
#   By the data-processing inequality KL(P'||Q') <= KL(P||Q); the dropped term is
#   the within-tail conditional KL, weighted by m. The teacher's top-128 at T=2
#   carries almost all the mass (m is typically <2-5%), so the gradient signal is
#   nearly identical — but it is NOT bit-identical to full KD. Hence: OFF by
#   default, and adoption requires the A/B protocol in research/qat-step-profile.md
#   (10-step loss overlay + full-arm PPL A/B, kill bar |dPPL| < 0.5%).
#
# Cache keying: sha256(chunk token ids) + version + T + k -> one small .pt per chunk
# (~int32[N,k] + fp16[N,k] + fp16[N]). A miss runs the teacher forward as before and
# populates; a hit SKIPS the teacher forward entirely. When --kd-cache is on, the
# sparse loss is used on BOTH hits and misses so the loss math is uniform in-run.
# ---------------------------------------------------------------------------
KD_CACHE_VERSION = 1


def kd_cache_key(row_ids, T, k):
    """Content key for one train chunk: token ids + cache version + T + k."""
    import hashlib
    h = hashlib.sha256(row_ids.detach().to("cpu", torch.int32).contiguous()
                       .numpy().tobytes())
    h.update(f"|v{KD_CACHE_VERSION}|T={T}|k={k}".encode())
    return h.hexdigest()[:32]


@torch.no_grad()
def kd_targets_from_logits(tl_row, T, k):
    """tl_row: [N, V] fp32 teacher logits for one sample (N = ctx-1 positions).
    Returns CPU tensors (idx int32 [N,k], p fp16 [N,k], tail fp16 [N])."""
    pt = F.softmax(tl_row / T, dim=-1)
    p, idx = torch.topk(pt, k, dim=-1)
    tail = (1.0 - p.sum(dim=-1)).clamp_min(0.0)
    return (idx.to(torch.int32).cpu(), p.to(torch.float16).cpu(),
            tail.to(torch.float16).cpu())


def kd_loss_sparse(sl, idx, p, tail, T):
    """KL(P'||Q') of the bucketed top-k target (derivation above), chunked over rows
    like the dense path (softmax stays over the full vocab per row; row-chunking is
    exact because the sum is additive over rows)."""
    KD_CHUNK = 128
    N = sl.shape[0]
    kd_sum = sl.new_zeros(())
    for c0 in range(0, N, KD_CHUNK):
        lq = F.log_softmax(sl[c0:c0 + KD_CHUNK] / T, dim=-1)
        lqk = lq.gather(1, idx[c0:c0 + KD_CHUNK].long())        # log q_i, i in K
        q_in = lqk.exp().sum(dim=-1)
        lq_tail = torch.log1p(-q_in.clamp(max=1.0 - 1e-7))      # log q_tail, stable
        pk = p[c0:c0 + KD_CHUNK].float()
        m = tail[c0:c0 + KD_CHUNK].float()
        kd_sum = kd_sum + (pk * (pk.clamp_min(1e-12).log() - lqk)).sum() \
                        + (m * (m.clamp_min(1e-12).log() - lq_tail)).sum()
    return kd_sum / N * (T * T)


def kd_cached(teacher, ids, ids_cpu, sl, args):
    """--kd-cache path: per-chunk lookup; misses run ONE teacher forward for the
    micro-batch and populate; hits skip the teacher forward entirely."""
    T, k = args.kd_temp, args.kd_cache_topk
    B, ctx = ids_cpu.shape
    N = ctx - 1
    os.makedirs(args.kd_cache, exist_ok=True)
    keys = [kd_cache_key(ids_cpu[b], T, k) for b in range(B)]
    entries, miss = [], []
    for b, key in enumerate(keys):
        path = os.path.join(args.kd_cache, key + ".pt")
        e = None
        if os.path.exists(path):
            try:
                e = torch.load(path, map_location="cpu")
                if not (e.get("v") == KD_CACHE_VERSION and tuple(e["idx"].shape) == (N, k)):
                    e = None
            except Exception:
                e = None
        entries.append(e)
        if e is None:
            miss.append(b)
    if miss:
        with torch.no_grad():
            tl = teacher(ids, use_cache=False).logits[:, :-1, :].float()
        for b in miss:
            idx, p, tail = kd_targets_from_logits(tl[b], T, k)
            e = {"v": KD_CACHE_VERSION, "idx": idx, "p": p, "tail": tail}
            torch.save(e, os.path.join(args.kd_cache, keys[b] + ".pt"))
            entries[b] = e
        del tl
    dev = sl.device
    idx = torch.cat([e["idx"] for e in entries]).to(dev)     # [B*N, k]
    p = torch.cat([e["p"] for e in entries]).to(dev)
    tail = torch.cat([e["tail"] for e in entries]).to(dev)
    st = args._kd_cache_stats
    st["hit"] += B - len(miss)
    st["miss"] += len(miss)
    return kd_loss_sparse(sl, idx, p, tail, T)


@torch.no_grad()
def eval_ppl(model, eval_ch, device, tag=""):
    model.eval()
    if device == "mps":
        torch.mps.empty_cache()   # defragment: eval needs contiguous GBs next to AdamW state
    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    nll, ntok = 0.0, 0
    t0 = time.time()
    for i, ch in enumerate(eval_ch):
        ids = ch.unsqueeze(0).to(device)
        try:
            logits = model(ids, use_cache=False).logits
        except RuntimeError as e:   # MPS allocator fragmentation after long runs: defrag+retry
            if "MPS backend out of memory" not in str(e):
                raise
            torch.mps.empty_cache()
            logits = model(ids, use_cache=False).logits
        sl = logits[:, :-1, :].reshape(-1, logits.size(-1))
        lab = ids[:, 1:].reshape(-1)
        # Chunked CE: full-vocab log_softmax over 2047x152k rows is a ~2.5GB transient
        # (OOM'd the step-100 mid-eval at ctx2048). 512-row slices = identical Σnll
        # (CE-sum is additive over rows) at ~300MB peak.
        for j in range(0, lab.numel(), 512):
            nll += loss_fct(sl[j:j + 512].float(), lab[j:j + 512]).item()
        ntok += lab.numel()
        del logits, sl
    if device == "mps":
        torch.mps.empty_cache()
    ppl = math.exp(nll / ntok)
    print(f"[qat] eval{(' '+tag) if tag else ''}: ppl={ppl:.4f}  "
          f"({len(eval_ch)} ch, {ntok} tok, {time.time()-t0:.0f}s)", flush=True)
    return ppl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--quant", choices=list(QUANTIZERS), default="uniform")
    p.add_argument("--bits", type=int, default=2)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-5)
    # Apple compute-optimal QAT recipe (ADOPT, 2026-06-11 intel scorecard): a
    # warmup->hold->decay (WSD/trapezoid) schedule. Apple's published finding is that
    # for low-bit QAT a short final COOLDOWN matters more than total step count; their
    # other levers (balanced level set, weight_decay=0) are already-have (our frozen
    # LUT is Gaussian-optimal Lloyd-Max; wd=0 is already set) or dead-for-us (learnable
    # clip-init — RHT+outlier-channel already suppress the tail). --cooldown-frac 0
    # (default) reproduces the prior plain CosineAnnealingLR exactly — opt-in, no
    # default-path change.
    p.add_argument("--cooldown-frac", type=float, default=0.0,
                   help="WSD schedule: fraction of steps in the final linear decay-to-zero "
                        "(0 = legacy cosine; Apple-recipe value ~0.2)")
    p.add_argument("--warmup-frac", type=float, default=0.0,
                   help="WSD schedule: fraction of steps in the initial linear warmup")
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--train-chunks", type=int, default=256)
    p.add_argument("--eval-chunks", type=int, default=32)
    p.add_argument("--eval-ctx", type=int, default=2048)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--kd", action="store_true", help="distill from the frozen FP teacher (KL)")
    p.add_argument("--kd-temp", type=float, default=2.0)
    p.add_argument("--kd-cache", default="",
                   help="OFF-BY-DEFAULT science flag: dir for the teacher top-k logit "
                        "cache. Changes the KD loss to the bucketed top-k approximation "
                        "(exact derivation at kd_loss_sparse) — adopt only via the A/B "
                        "protocol in research/qat-step-profile.md (kill bar |dPPL|<0.5%%). "
                        "First epoch populates; later passes skip the teacher forward.")
    p.add_argument("--kd-cache-topk", type=int, default=128,
                   help="k for the cached top-k teacher target (tail mass = 1 bucket)")
    p.add_argument("--kd-weight", type=float, default=1.0)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=0, help="eval mid-training every N steps")
    p.add_argument("--log-every", type=int, default=1, help="print every N optimizer steps")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="HF gradient checkpointing: recompute activations in backward "
                        "(~30%% slower, several GB less transient MPS memory)")
    p.add_argument("--requant-every", type=int, default=75,
                   help="strand mode: re-quantize through the Rust encoder every N steps")
    p.add_argument("--requant-shards", type=int, default=8,
                   help="parallel quantize-model shards for the in-loop requant (wall-clock only; "
                        "result-identical since each tensor quantizes independently). The FULL init "
                        "requant on 7B was ~2.5h as one merged process; ~8 shards -> ~15-20min.")
    p.add_argument("--init-recon-dir", default="",
                   help="100x lever: dir of an EXISTING same-config PTQ recon (e.g. the scorecard "
                        "q2_l12_out1/recon). At init, load it instead of re-encoding all tensors -- "
                        "byte-identical (init shadows==original weights, name-seeded RHT), seconds not "
                        "hours. MUST match --strand-flags bits/l/outlier or the recon is wrong.")
    p.add_argument("--strand-bin", default="target/release/quantize-model")
    p.add_argument("--strand-flags", default="--bits 2 --l 12 --outlier-channel 1 --threads 8",
                   help="flags for the in-loop quantize-model calls (deployment config)")
    p.add_argument("--strand-dir", default="",
                   help="workdir for shadow/recon dumps (default: <model>/strand-pv)")
    p.add_argument("--skip-after", action="store_true",
                   help="segment mode: exit after training without the final requant/AFTER eval "
                        "(the next segment's fresh process does them on a pristine MPS pool)")
    p.add_argument("--chunk-offset", type=int, default=0,
                   help="skip the first N train chunks (segmented arms walk the data forward)")
    p.add_argument("--init-state", default="",
                   help="load a state_dict checkpoint (shadow weights + buffers) before "
                        "training — crash-resume / warm restart of a longer arm")
    p.add_argument("--train-all", action="store_true",
                   help="also train embeddings/norms (default: only the wrapped QuantLinears — "
                        "freezing the rest cuts ~2GB of grad+AdamW state; the 18GB box froze "
                        "twice training everything)")
    # P1 (rung-allocator design §4.4): selective PV.
    p.add_argument("--pv-tensors", default="",
                   help="selective PV: regex (re.search) over QuantLinear module names; "
                        "matching tensors train, the rest freeze at their requant recon "
                        "(delta forward: base+w is exact recon while w is frozen). "
                        "Empty = train all wrapped (today's behavior).")
    # P6 is deliberately DEFERRED (per-class LR via AdamW param groups): v1 runs the PV
    # set at a single --lr; an AMBER pass is a separate invocation at 3e-5.
    p.add_argument("--arm-name", default="",
                   help="lineage: arm name for the research/pv-lineage.jsonl record "
                        "(default: derived from quant/bits/pv)")
    p.add_argument("--lineage", default="",
                   help="lineage jsonl path (default: <repo>/research/pv-lineage.jsonl; "
                        "'none' disables)")
    p.add_argument("--lineage-label", default="",
                   help="free-text label for the lineage record, e.g. 'mechanism-smoke' "
                        "(NOT science) vs 'science'")
    p.add_argument("--save", default="")
    p.add_argument("--save-hf", default="",
                   help="write a drop-in HF model dir (config+tokenizer+model.safetensors, bf16, "
                        "tuned shadow weights) — feed it to strand-7b-ppl.sh as MODEL_DIR to get "
                        "the canon PPL of QAT'd-then-STRAND-quantized weights")
    p.add_argument("--out", default="", help="json results path")
    args = p.parse_args()

    if args.kd_cache and not args.kd:
        raise SystemExit("[qat] --kd-cache requires --kd (it caches the KD teacher)")
    args._kd_cache_stats = {"hit": 0, "miss": 0}
    if args.kd_cache:
        print(f"[qat] KD-CACHE ON (science flag, approximates the KD loss with the "
              f"bucketed top-{args.kd_cache_topk} target): {args.kd_cache}", flush=True)

    torch.manual_seed(args.seed)
    dev = args.device
    print(f"[qat] {args.quant} bits={args.bits} steps={args.steps} lr={args.lr} "
          f"ctx={args.ctx} kd={args.kd} dev={dev}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    # eager attention: MPS SDPA can't broadcast GQA (14 Q heads vs 2 KV heads); eager makes
    # repeat_kv explicit. Matches calibrate-hsdi.py.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager").to(dev)

    # Optional frozen teacher for KD (loaded before wrapping so it stays full-precision).
    teacher = None
    if args.kd:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="eager").to(dev)
        teacher.eval()
        for q in teacher.parameters():
            q.requires_grad_(False)

    nwrapped = wrap_proj_linears(model, QUANTIZERS[args.quant], args.bits)
    model.to(dev)
    print(f"[qat] wrapped {nwrapped} projection Linears as QuantLinear", flush=True)

    if not args.train_all:
        keep = set()
        for m in model.modules():
            if isinstance(m, QuantLinear):
                keep.add(id(m.weight))
                if m.bias is not None:
                    keep.add(id(m.bias))
        for q in model.parameters():
            if id(q) not in keep:
                q.requires_grad_(False)
    # P2 (rung-allocator design §4.4): selective PV — matching QuantLinears train, the
    # rest freeze. Under the delta forward (base + w, base = recon - w_anchor re-set per
    # requant) a frozen tensor's forward IS its requant recon exactly, forever (§4.2).
    npv = -1
    if args.pv_tensors:
        import re
        rx = re.compile(args.pv_tensors)
        npv = 0
        for name, m in model.named_modules():
            if isinstance(m, QuantLinear):
                sel = bool(rx.search(name))
                m.weight.requires_grad_(sel)
                if m.bias is not None:
                    m.bias.requires_grad_(sel)   # bias rides with its weight (v1; see E4d)
                npv += int(sel)
        assert npv > 0, f"--pv-tensors {args.pv_tensors!r} matched 0 QuantLinears"
        print(f"[qat] selective PV: {npv}/{nwrapped} QuantLinears trainable", flush=True)
    ntrain = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"[qat] trainable: {ntrain/1e6:.1f}M params "
          f"({'all' if args.train_all else 'QuantLinears only'})", flush=True)

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()   # frozen embeddings: give checkpoints a grad path
        print("[qat] gradient checkpointing ON", flush=True)

    if args.init_state and os.path.exists(args.init_state):
        sd = torch.load(args.init_state, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[qat] init-state <- {args.init_state} "
              f"(missing {len(missing)}, unexpected {len(unexpected)})", flush=True)

    train_ch = chunks(load_wikitext_ids(tok, "train"), args.ctx,
                      args.chunk_offset + args.train_chunks)[args.chunk_offset:]
    eval_ch = chunks(load_wikitext_ids(tok, "test"), args.eval_ctx, args.eval_chunks)
    print(f"[qat] train={len(train_ch)} ch @ctx{args.ctx}  eval={len(eval_ch)} ch @ctx{args.eval_ctx}",
          flush=True)

    args._evals = []

    def eval_parked(tag):
        """Evals never use the KD teacher — park it on CPU to free ~1GB+ of MPS during
        the eval's logits transients (the post-requant eval OOM'd at the 13.32 cap)."""
        if teacher is not None and dev == "mps":
            teacher.to("cpu"); torch.mps.empty_cache()
        p = eval_ppl(model, eval_ch, dev, tag=tag)
        if teacher is not None and dev == "mps":
            teacher.to(dev)
        args._evals.append({"tag": tag, "ppl": round(p, 4)})
        return p

    last_requant = -1
    pv_trained0 = {}
    if args.quant == "strand":
        if not args.strand_dir:
            args.strand_dir = os.path.join(args.model if os.path.isdir(args.model) else ".",
                                           "strand-pv")
        strand_requant(model, args, "init")   # BEFORE eval = the true PTQ-floor recon
        last_requant = 0
        if args.pv_tensors:
            pv_capture_frozen(model, args)
            # Trained-set digests at the start: the smoke's "trained ones move" check.
            pv_trained0 = {name: shadow_digest(m) for name, m in model.named_modules()
                           if isinstance(m, QuantLinear) and m.weight.requires_grad}

    ppl_before = eval_parked("BEFORE (quant, untuned)")

    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    if args.cooldown_frac > 0.0 or args.warmup_frac > 0.0:
        # WSD: linear warmup -> hold at 1.0 -> linear cooldown to 0 (Apple QAT recipe).
        n = max(args.steps, 1)
        n_warm = int(args.warmup_frac * n)
        n_cool = int(args.cooldown_frac * n)
        n_hold_end = n - n_cool
        def wsd(step):
            if step < n_warm:
                return (step + 1) / max(n_warm, 1)
            if step < n_hold_end:
                return 1.0
            return max(0.0, (n - step) / max(n_cool, 1))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, wsd)
        print(f"[qat] LR schedule: WSD warmup={n_warm} hold={n_hold_end - n_warm} cooldown={n_cool}", flush=True)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.steps, 1))

    model.train()
    bsz = max(args.batch, 1)
    # batch=B, accum=A/B is gradient-identical to batch=1, accum=A (full equal-length chunks,
    # mean-of-means == global mean) but launches fewer, larger MPS kernels.
    train_b = ([torch.stack(train_ch[i:i + bsz]) for i in range(0, len(train_ch) - bsz + 1, bsz)]
               if bsz > 1 else [ch.unsqueeze(0) for ch in train_ch])
    print(f"[qat] entering training loop ({args.steps} steps, batch {bsz}, "
          f"accum {args.grad_accum}, {len(train_b)} micro-batches/epoch)", flush=True)
    step, gi, t0 = 0, 0, time.time()
    opt.zero_grad()
    while step < args.steps:
        for ids_cpu in train_b:
            ids = ids_cpu.to(dev)
            out = model(ids, use_cache=False)
            logits = out.logits
            sl = logits[:, :-1, :].float().reshape(-1, logits.size(-1))
            lab = ids[:, 1:].reshape(-1)
            loss = args.ce_weight * F.cross_entropy(sl, lab)
            if teacher is not None and args.kd_cache:
                # Science-flag path (off by default): bucketed top-k KD against the
                # cached teacher target — hits skip the teacher forward entirely.
                loss = loss + args.kd_weight * kd_cached(teacher, ids, ids_cpu, sl, args)
            elif teacher is not None:
                with torch.no_grad():
                    tl = teacher(ids, use_cache=False).logits[:, :-1, :].float().reshape(-1, logits.size(-1))
                T = args.kd_temp
                # Chunked KD (footprint surgery 2026-06-10, will.md doctrine: "chunked
                # kl_div, not more cap"). Identical math to
                #   kl_div(log_softmax(sl/T), softmax(tl/T), reduction="batchmean") * T*T
                # — batchmean is sum/N over rows, so row-chunks (softmax stays over the
                # full vocab per row) reproduce it exactly up to fp summation order.
                # Cuts the ~1.2GB softmax/KL transient ~4x (step-26 OOM at the 13.32GB
                # cap, 2026-06-10 18:27).
                KD_CHUNK = 128
                kd_sum = sl.new_zeros(())
                for c0 in range(0, sl.shape[0], KD_CHUNK):
                    kd_sum = kd_sum + F.kl_div(
                        F.log_softmax(sl[c0:c0 + KD_CHUNK] / T, dim=-1),
                        F.softmax(tl[c0:c0 + KD_CHUNK] / T, dim=-1),
                        reduction="sum")
                kd = kd_sum / sl.shape[0] * (T * T)
                loss = loss + args.kd_weight * kd
            (loss / args.grad_accum).backward()
            gi += 1
            if gi % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if dev == "mps":
                    torch.mps.empty_cache()
                if step % max(args.log_every, 1) == 0 or step == args.steps:
                    mem = (f" mps={torch.mps.current_allocated_memory()/2**30:.2f}GB"
                           f" drv={torch.mps.driver_allocated_memory()/2**30:.2f}GB"
                           if dev == "mps" else "")
                    print(f"[qat] step {step}/{args.steps} loss={loss.item():.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s){mem}", flush=True)
                if args.quant == "strand" and args.requant_every \
                        and step % args.requant_every == 0:
                    strand_requant(model, args, f"@step{step}"); model.train()
                    last_requant = step
                if args.eval_every and step % args.eval_every == 0:
                    eval_parked(f"@step{step}"); model.train()
                    if args.save:   # checkpoint so a late crash never costs the whole run
                        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                                   args.save)
                        print(f"[qat] checkpoint -> {args.save} (step {step})", flush=True)
                if step >= args.steps:
                    break

    if args.steps == 0:
        ppl_after = ppl_before        # finisher invocation: requant->eval already done above
    elif args.skip_after:
        ppl_after = None              # segment exits here; next fresh process evals
        print("[qat] segment end (skip-after): no final requant/eval in this process", flush=True)
    else:
        if args.quant == "strand" and last_requant != step:
            strand_requant(model, args, "final")
        ppl_after = eval_parked("AFTER (quant, tuned)")

    # Selective-PV end-of-run invariants: frozen tensors hash-equal; trained ones moved.
    pv_moved = -1
    if args.pv_tensors and args.quant == "strand":
        pv_verify_frozen(model, args, "end-of-run")
        if args.steps > 0 and pv_trained0:
            pv_moved = sum(1 for name, m in model.named_modules()
                           if isinstance(m, QuantLinear) and name in pv_trained0
                           and shadow_digest(m) != pv_trained0[name])
            print(f"[qat] selective-PV: {pv_moved}/{len(pv_trained0)} trained shadows moved",
                  flush=True)
            assert pv_moved == len(pv_trained0), \
                "[qat] selective-PV: some TRAINED shadows did not move — freeze scope bug?"

    if args.kd_cache:
        st = args._kd_cache_stats
        tot = st["hit"] + st["miss"]
        print(f"[qat] kd-cache: {st['hit']}/{tot} micro-batch rows hit "
              f"({st['miss']} teacher forwards paid)", flush=True)

    print(f"\n[qat] ===== RESULT =====", flush=True)
    if ppl_after is not None:
        print(f"[qat]   {args.quant} {args.bits}-bit  BEFORE={ppl_before:.3f}  AFTER={ppl_after:.3f}  "
              f"(Δ {(ppl_after/ppl_before-1)*100:+.1f}%)", flush=True)
    else:
        print(f"[qat]   {args.quant} {args.bits}-bit  segment BEFORE={ppl_before:.3f}  (AFTER deferred)",
              flush=True)

    if args.save:
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(sd, args.save)
        print(f"[qat] saved tuned shadow weights -> {args.save}", flush=True)

    if args.save_hf:
        import glob, shutil
        from safetensors import safe_open
        from safetensors.torch import save_file
        os.makedirs(args.save_hf, exist_ok=True)
        for f in ("config.json", "generation_config.json", "tokenizer.json",
                  "tokenizer_config.json", "vocab.json", "merges.txt",
                  "special_tokens_map.json", "added_tokens.json"):
            src = os.path.join(args.model, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(args.save_hf, f))
        # Keep exactly the source file's tensor set (drops tied lm_head, non-persistent
        # buffers) so the output dir is layout-identical to the original model dir.
        src_keys = set()
        for f in glob.glob(os.path.join(args.model, "model*.safetensors")):
            with safe_open(f, framework="pt") as sf:
                src_keys.update(sf.keys())
        sd = {k: v.detach().cpu().to(torch.bfloat16).contiguous()
              for k, v in model.state_dict().items() if k in src_keys}
        missing = src_keys - set(sd)
        if missing:
            print(f"[qat] WARNING: {len(missing)} source tensors not in state_dict: "
                  f"{sorted(missing)[:4]}...", flush=True)
        save_file(sd, os.path.join(args.save_hf, "model.safetensors"), metadata={"format": "pt"})
        print(f"[qat] saved HF model dir -> {args.save_hf} ({len(sd)} tensors, bf16)", flush=True)
    if args.out:
        json.dump({"quant": args.quant, "bits": args.bits, "steps": args.steps, "lr": args.lr,
                   "kd": args.kd, "ppl_before": ppl_before, "ppl_after": ppl_after,
                   "wrapped": nwrapped, "eval_chunks": len(eval_ch), "eval_ctx": args.eval_ctx,
                   # P5: selective-PV provenance.
                   "pv_tensors": args.pv_tensors, "pv_count": npv},
                  open(args.out, "w"), indent=2)
        print(f"[qat] wrote {args.out}", flush=True)

    # ---- per-arm lineage record (append-only jsonl; audit training.md §2.4) ----
    if args.lineage != "none":
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lineage_path = args.lineage or os.path.join(repo, "research", "pv-lineage.jsonl")
        os.makedirs(os.path.dirname(lineage_path), exist_ok=True)
        arm = args.arm_name or (f"{args.quant}-{args.bits}bit"
                                + (f"-pv[{args.pv_tensors}]" if args.pv_tensors else ""))
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "arm": arm,
            "label": args.lineage_label,          # e.g. "mechanism-smoke" (NOT science)
            "model": os.path.abspath(args.model),
            "parent": os.path.abspath(args.init_state) if args.init_state else None,
            "quant": args.quant, "bits": args.bits,
            "steps": args.steps, "lr": args.lr, "kd": args.kd, "seed": args.seed,
            "ctx": args.ctx, "train_chunks": args.train_chunks,
            "chunk_offset": args.chunk_offset,
            "eval_chunks": len(eval_ch), "eval_ctx": args.eval_ctx,
            "pv_tensors": args.pv_tensors, "pv_count": npv,
            "pv_trained_moved": pv_moved,
            "kd_cache": os.path.abspath(args.kd_cache) if args.kd_cache else None,
            "kd_cache_topk": args.kd_cache_topk if args.kd_cache else None,
            "kd_cache_stats": args._kd_cache_stats if args.kd_cache else None,
            "strand_flags": args.strand_flags if args.quant == "strand" else None,
            "requant_every": args.requant_every if args.quant == "strand" else None,
            "requants": getattr(args, "_requants", []),
            "evals": args._evals,
            "ppl_before": ppl_before, "ppl_after": ppl_after,
            "checkpoints": {"save": os.path.abspath(args.save) if args.save else None,
                            "save_hf": os.path.abspath(args.save_hf) if args.save_hf else None,
                            "out_json": os.path.abspath(args.out) if args.out else None},
            "argv": sys.argv[1:],
        }
        with open(lineage_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[qat] lineage record appended -> {lineage_path}", flush=True)


if __name__ == "__main__":
    main()
