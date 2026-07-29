#!/usr/bin/env python3
"""qat_profile.py — per-phase wall-time decomposition of ONE strand-qat.py training step.

MECHANISM-NOT-SCIENCE: this runs a short (default 8 optimizer steps) instrumented QAT
smoke at a tiny lr, with the uniform-2bit STE quantizer (the cheapest forward that still
exercises the real step structure: QuantLinear fake-quant fwd, KD teacher fwd, CE+chunked
KD loss, backward through 168 wrapped Linears, AdamW). NO number printed here is a quality
result; the deliverable is the phase decomposition table:

    data | student fwd | teacher fwd | loss+KD | backward | optimizer

so the teacher-logit-cache projection (strand-qat.py --kd-cache) can be priced honestly.

MPS timing discipline: torch.mps.synchronize() before reading every timer boundary —
MPS kernels are async; without the sync the time lands in whichever phase forces the
flush (usually .item() in the loss). With gradient checkpointing ON (the real PV-arm
config) the student forward is cheap and its recompute cost lands in `backward` — the
table footnotes this.

Usage (matches the live PV-arm shape):
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.92 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7 \
  /usr/local/bin/python3 tools/qat_profile.py --model scratch/qwen-05b \
      --steps 8 --ctx 512 --grad-accum 4 --out research/qat-step-profile.json
"""
import argparse, importlib.util, json, os, sys, time

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_qat_module():
    """Import scripts/strand-qat.py (dash in the name -> importlib, not import)."""
    path = os.path.join(REPO, "scripts", "strand-qat.py")
    spec = importlib.util.spec_from_file_location("strand_qat", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.join(REPO, "scratch", "qwen-05b"))
    p.add_argument("--device", default="mps")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-7, help="tiny on purpose: mechanism, not science")
    p.add_argument("--bits", type=int, default=2)
    p.add_argument("--quant", choices=["uniform", "strand-sim"], default="strand-sim",
                   help="strand-sim = QuantLinear with quant_fn=None and base=0: the "
                        "delta forward (base + w) of the live PV arm, so the per-step "
                        "COMPUTE matches the real strand mode without the 17-min Rust "
                        "init requant (numerically it is the fp32 identity — timing "
                        "only, mechanism-not-science). uniform = the STE fake-quant "
                        "path (heavier forward: 168 per-channel quantize ops).")
    p.add_argument("--kd-temp", type=float, default=2.0)
    p.add_argument("--no-kd", action="store_true", help="profile without the teacher")
    p.add_argument("--no-grad-checkpoint", action="store_true",
                   help="default ON to match the live PV-arm config")
    p.add_argument("--out", default="", help="json results path")
    args = p.parse_args()

    qat = load_qat_module()
    dev = args.device
    use_kd = not args.no_kd

    def sync():
        if dev == "mps":
            torch.mps.synchronize()

    print(f"[profile] MECHANISM-NOT-SCIENCE smoke: {args.quant} bits={args.bits}, "
          f"steps={args.steps} ctx={args.ctx} accum={args.grad_accum} lr={args.lr} "
          f"kd={use_kd} grad_checkpoint={not args.no_grad_checkpoint} dev={dev}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager").to(dev)
    teacher = None
    if use_kd:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="eager").to(dev)
        teacher.eval()
        for q in teacher.parameters():
            q.requires_grad_(False)
    print(f"[profile] models loaded ({time.time()-t0:.0f}s)", flush=True)

    quant_fn = None if args.quant == "strand-sim" else qat.QUANTIZERS["uniform"]
    nwrapped = qat.wrap_proj_linears(model, quant_fn, args.bits)
    model.to(dev)
    keep = set()
    for m in model.modules():
        if isinstance(m, qat.QuantLinear):
            keep.add(id(m.weight))
            if m.bias is not None:
                keep.add(id(m.bias))
    for q in model.parameters():
        if id(q) not in keep:
            q.requires_grad_(False)
    if not args.no_grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    print(f"[profile] wrapped {nwrapped} QuantLinears", flush=True)

    n_micro = args.steps * args.grad_accum
    train_ch = qat.chunks(qat.load_wikitext_ids(tok, "train"), args.ctx, n_micro)
    assert len(train_ch) >= n_micro, "not enough train chunks for the requested steps"

    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    model.train()

    PHASES = ["data", "student_fwd", "teacher_fwd", "loss_kd", "backward", "optimizer"]
    per_step = []          # list of dicts, one per optimizer step
    cur = {ph: 0.0 for ph in PHASES}
    losses = []
    opt.zero_grad()
    step, gi = 0, 0
    wall0 = time.time()
    for ids_cpu in train_ch:
        # --- data ---
        sync(); t = time.time()
        ids = ids_cpu.unsqueeze(0).to(dev)
        sync(); cur["data"] += time.time() - t

        # --- student forward (with grad-checkpoint ON, the recompute lands in backward) ---
        t = time.time()
        logits = model(ids, use_cache=False).logits
        sync(); cur["student_fwd"] += time.time() - t

        # --- teacher forward ---
        t = time.time()
        tl = None
        if teacher is not None:
            with torch.no_grad():
                tl = teacher(ids, use_cache=False).logits
        sync(); cur["teacher_fwd"] += time.time() - t

        # --- loss + KD (identical math to strand-qat.py: CE + chunked KL) ---
        t = time.time()
        sl = logits[:, :-1, :].float().reshape(-1, logits.size(-1))
        lab = ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(sl, lab)
        if tl is not None:
            tlf = tl[:, :-1, :].float().reshape(-1, logits.size(-1))
            T = args.kd_temp
            KD_CHUNK = 128
            kd_sum = sl.new_zeros(())
            for c0 in range(0, sl.shape[0], KD_CHUNK):
                kd_sum = kd_sum + F.kl_div(
                    F.log_softmax(sl[c0:c0 + KD_CHUNK] / T, dim=-1),
                    F.softmax(tlf[c0:c0 + KD_CHUNK] / T, dim=-1),
                    reduction="sum")
            loss = loss + kd_sum / sl.shape[0] * (T * T)
        sync(); cur["loss_kd"] += time.time() - t

        # --- backward ---
        t = time.time()
        (loss / args.grad_accum).backward()
        sync(); cur["backward"] += time.time() - t
        loss_val = loss.item()
        del logits, sl, tl

        gi += 1
        if gi % args.grad_accum == 0:
            # --- optimizer (clip + step + zero + the per-step empty_cache the arm does) ---
            t = time.time()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad()
            if dev == "mps":
                torch.mps.empty_cache()
            sync(); cur["optimizer"] += time.time() - t
            step += 1
            cur["total"] = sum(cur[ph] for ph in PHASES)
            per_step.append(cur)
            losses.append(loss_val)
            mem = (f" mps={torch.mps.current_allocated_memory()/2**30:.2f}GB"
                   f" drv={torch.mps.driver_allocated_memory()/2**30:.2f}GB"
                   if dev == "mps" else "")
            print(f"[profile] step {step}/{args.steps} total={cur['total']:.2f}s "
                  f"loss={loss_val:.4f}{mem}", flush=True)
            cur = {ph: 0.0 for ph in PHASES}
            if step >= args.steps:
                break

    wall = time.time() - wall0
    # Mean over steps 2..N — step 1 carries MPS graph/JIT warmup.
    steady = per_step[1:] if len(per_step) > 1 else per_step
    mean = {ph: sum(s[ph] for s in steady) / len(steady) for ph in PHASES}
    mean_total = sum(mean.values())

    print("\n[profile] ===== STEP DECOMPOSITION (mean over steps 2..N; "
          "MECHANISM-NOT-SCIENCE) =====", flush=True)
    print(f"[profile] {'phase':<14}{'s/step':>10}{'share':>9}")
    for ph in PHASES:
        print(f"[profile] {ph:<14}{mean[ph]:>10.3f}{mean[ph]/mean_total*100:>8.1f}%")
    print(f"[profile] {'TOTAL':<14}{mean_total:>10.3f}{'100.0%':>9}")
    if not args.no_grad_checkpoint:
        print("[profile] note: grad-checkpoint ON -> the student-fwd recompute cost is "
              "inside `backward`; `student_fwd` is the cheap no-stash pass.", flush=True)
    no_teacher = mean_total - mean["teacher_fwd"]
    print(f"[profile] teacher fwd = {mean['teacher_fwd']/mean_total*100:.1f}% of the step; "
          f"a 100%-hit logit cache projects {mean_total:.2f} -> {no_teacher:.2f} s/step "
          f"({mean_total/no_teacher:.2f}x)", flush=True)
    print(f"[profile] wall total {wall:.0f}s for {step} steps "
          f"({step*args.grad_accum} micro-batches)", flush=True)

    if args.out:
        json.dump({
            "label": "mechanism-not-science",
            "config": {"model": os.path.abspath(args.model), "quant": args.quant,
                       "bits": args.bits, "steps": args.steps, "ctx": args.ctx,
                       "grad_accum": args.grad_accum, "lr": args.lr, "kd": use_kd,
                       "grad_checkpoint": not args.no_grad_checkpoint, "device": dev},
            "per_step_s": per_step, "losses": losses,
            "mean_steady_s": mean, "mean_total_s": mean_total,
            "projected_cached_step_s": no_teacher,
            "machine": "M3 Pro 18GB", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, open(args.out, "w"), indent=2)
        print(f"[profile] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
