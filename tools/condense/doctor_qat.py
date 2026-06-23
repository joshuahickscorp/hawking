#!/usr/bin/env python3.12
"""The DOCTOR — quality-infusion via quantization-aware fine-tuning (QAT).

Proves "condensation doesn't mean loss": PTQ collapses at 2-bit, the doctor heals it
back toward the f16 parent. Self-contained — uniform per-output-channel symmetric
N-bit fake-quant with a straight-through estimator (STE), so gradients flow to the
weights while the forward pass sees the quantized values. Trains the weights to be
quantization-robust, then reports held-out perplexity PTQ vs QAT.

Usage: doctor_qat.py [bits] [steps] [lr] [save.safetensors]
  bits  default 2 (the Hawking lead)   steps default 300   lr default 2e-5

Heavy (training) — run plugged in / via the cron, not on battery. A 5-step smoke
validates it runs.
"""
import sys, os, math, json, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "scratch/qwen-05b"
BITS  = int(sys.argv[1]) if len(sys.argv) > 1 else 2
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
LR    = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-5
SAVE  = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None
dev = "mps" if torch.backends.mps.is_available() else "cpu"

# calib (train) and eval (held-out) — DISTINCT passages so recovery is not memorization
CALIB = """A computing machine operates on symbols according to fixed rules, and the
arrangement of those rules is the program. The earliest engines were mechanical, built
from gears and levers, and their state was the position of physical wheels. As the
field matured, electrical relays replaced gears, then vacuum tubes replaced relays, and
finally transistors etched into silicon replaced tubes. Each step reduced the size and
energy of a single logical operation by orders of magnitude. Memory hierarchies emerged
because fast storage is expensive and slow storage is cheap, so designers arrange a
small fast cache near the processor and a large slow store further away. A program that
respects this hierarchy, touching nearby data repeatedly before moving on, runs far
faster than one that scatters its accesses across the whole memory. The same principle
governs modern accelerators, where the cost of moving a number from memory often exceeds
the cost of the arithmetic performed on it. Efficient computation is therefore as much
about the choreography of data movement as about the operations themselves."""
EVAL = """The science of operations, as derived from mathematics more especially, is a
science of itself, and has its own abstract truth and value. A new, a vast, and a
powerful language is developed for the future use of analysis, in which to wield its
truths so that these may become of more speedy and accurate practical application for
the purposes of mankind. The engine can arrange and combine its numerical quantities
exactly as if they were letters or any other general symbols, and in fact it might
bring out its results in algebraical notation, were provisions made accordingly."""


def fakequant(w, bits):
    qmax = 2 ** (bits - 1) - 1
    s = (w.abs().amax(dim=1, keepdim=True) / max(qmax, 1)).clamp(min=1e-8)
    q = torch.clamp(torch.round(w / s), -qmax - 1, qmax) * s
    return w + (q - w).detach()  # STE: forward=quantized, grad=identity


LINEARS = []


def patch(model, bits):
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.weight.shape[0] > 1 and m.weight.shape[1] >= 256:
            m._qbits = bits
            m.forward = (lambda x, mm=m: F.linear(x, fakequant(mm.weight, mm._qbits), mm.bias))
            LINEARS.append(m)


def ppl(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        return math.exp(model(ids, labels=ids).loss.item())


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager").to(dev)
    patch(model, BITS)
    print(f"# doctor: {BITS}-bit, {len(LINEARS)} linears, {STEPS} steps, lr {LR}, dev {dev}",
          file=sys.stderr)

    model.eval()
    ptq = ppl(model, tok, EVAL)
    print(f"# PTQ {BITS}-bit held-out ppl (no training): {ptq:.3f}", file=sys.stderr)

    # KD (distillation) mode: KL against a frozen f16 teacher — a richer signal than
    # self-CE, which the user explicitly wants. Env KD=1.
    KD = os.environ.get("KD") == "1"
    cids = tok(CALIB, return_tensors="pt").input_ids.to(dev)
    t_logp = None
    if KD:
        teacher = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float32, attn_implementation="eager").to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            t_logp = torch.log_softmax(teacher(cids).logits, dim=-1)
        print(f"# KD on: distilling frozen f16 teacher logits", file=sys.stderr)

    model.train()
    opt = torch.optim.AdamW([m.weight for m in LINEARS], lr=LR)
    for step in range(STEPS):
        opt.zero_grad()
        s_logits = model(cids).logits
        if KD:
            s_logp = torch.log_softmax(s_logits, dim=-1)
            loss = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()  # forward KL
        else:
            loss = F.cross_entropy(
                s_logits[:, :-1].reshape(-1, s_logits.size(-1)), cids[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == STEPS - 1:
            print(f"#  step {step:4d} {'kl' if KD else 'ce'}_loss {loss.item():.4f}", file=sys.stderr)

    model.eval()
    qat = ppl(model, tok, EVAL)
    rec = (ptq - qat) / ptq * 100 if ptq else 0.0
    print(json.dumps({"bits": BITS, "steps": STEPS, "lr": LR,
                      "ptq_ppl": ptq, "qat_ppl": qat, "recovery_pct": rec}))

    if SAVE:
        from safetensors.torch import save_file
        sd, raw = {}, {}
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and hasattr(m, "_qbits"):
                sd[name + ".weight"] = fakequant(m.weight.detach(), m._qbits).cpu().to(torch.float16)
                raw[name + ".weight"] = m.weight.detach().cpu().to(torch.float16)
        save_file(sd, SAVE)
        print(f"# saved healed (uniform-quantized) weights: {SAVE} ({len(sd)} tensors)", file=sys.stderr)
        # RAW healed shadow (un-quantized) so the cron can STRAND-bake it -> product TQ quality
        raw_path = SAVE.replace(".safetensors", ".raw.safetensors")
        save_file(raw, raw_path)
        print(f"# saved RAW healed shadow (for STRAND re-bake): {raw_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
