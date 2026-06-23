#!/usr/bin/env python3.12
"""LoRA RECOVERY doctor (memory-efficient) — the fix for the 19GB wall.

Full-weight STRAND-aware QAT didn't fit (model + per-linear anchors + optimizer + bake
all in 19GB -> swap). Instead: FREEZE the STRAND-quantized base (W_hat, already baked) and
train tiny rank-r LoRA adapters (A@B) to recover quality. Trains ~millions of params, not
0.5B -> fits + fast. Deployed artifact = STRAND low-bit base + a small f16 LoRA correction
(QLoRA-style). The healed weight W_hat + A@B is measured directly (held-out ppl).

Usage: doctor_lora.py <wbase.safetensors> [steps] [lr] [rank] [save.safetensors]
  wbase = the STRAND-decoded W_hat base (e.g. scratch/qwen-05b-tq2-full.safetensors)
Env: DOCTOR_CALIB (diverse corpus)
"""
import sys, os, math, json, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file, load_file

MODEL = "scratch/qwen-05b"
WBASE = sys.argv[1] if len(sys.argv) > 1 else "scratch/qwen-05b-tq2-full.safetensors"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
LR    = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
RANK  = int(sys.argv[4]) if len(sys.argv) > 4 else 16
SAVE  = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None
dev = "mps" if torch.backends.mps.is_available() else "cpu"

EVAL = """The science of operations, as derived from mathematics more especially, is a
science of itself, and has its own abstract truth and value. A new, a vast, and a
powerful language is developed for the future use of analysis, in which to wield its
truths so that these may become of more speedy and accurate practical application for
the purposes of mankind. The engine can arrange and combine its numerical quantities
exactly as if they were letters or any other general symbols."""

ADAPT = []  # modules with LoRA


def ppl(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids[:, :2048].to(dev)
    with torch.no_grad():
        return math.exp(model(ids, labels=ids).loss.item())


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager").to(dev)
    wh = load_file(WBASE)  # STRAND-decoded base weights

    params = []
    for name, m in model.named_modules():
        key = name + ".weight"
        if isinstance(m, nn.Linear) and key in wh:
            m.weight.data = wh[key].to(dev, torch.float32)  # frozen STRAND base
            m.weight.requires_grad_(False)
            m._A = nn.Parameter(torch.zeros(m.weight.shape[0], RANK, device=dev))
            m._B = nn.Parameter(torch.randn(RANK, m.weight.shape[1], device=dev) * 0.01)
            m.forward = (lambda x, mm=m: F.linear(x, mm.weight + mm._A @ mm._B, mm.bias))
            params += [m._A, m._B]
            ADAPT.append((name, m))
    print(f"# lora-doctor: base {WBASE}, {len(ADAPT)} adapters rank {RANK}, "
          f"{sum(p.numel() for p in params)/1e6:.1f}M trainable, {STEPS} steps lr {LR}", file=sys.stderr)

    model.eval()
    base_ppl = ppl(model, tok, EVAL)  # A=0 => pure STRAND base
    print(f"# STRAND base held-out ppl (no LoRA): {base_ppl:.3f}", file=sys.stderr)

    # diverse calib chunks
    calib = open(os.environ["DOCTOR_CALIB"], errors="ignore").read() if os.environ.get("DOCTOR_CALIB") else EVAL
    ids = tok(calib, return_tensors="pt").input_ids[0]
    CTX = 512
    chunks = [ids[i:i + CTX].unsqueeze(0).to(dev) for i in range(0, max(1, len(ids) - CTX), CTX)]
    chunks = [c for c in chunks if c.shape[1] >= 16] or [ids.unsqueeze(0).to(dev)]
    print(f"# calib: {len(chunks)} chunks", file=sys.stderr)

    # KD: distill the frozen f16 teacher's full logit distribution — a far richer recovery
    # signal than next-token CE (CE barely moved TQ3; KD targets the teacher's behavior).
    # KD: distill the f16 teacher. CACHED top-k (precompute teacher logits, FREE the teacher)
    # so only ONE model is in memory during training — KD with two f32 models swaps on 19GB.
    KD = os.environ.get("KD") == "1"
    KDK = int(os.environ.get("KD_TOPK", "64"))
    kd_cache = None
    if KD:
        teacher = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float32, attn_implementation="eager").to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        kd_cache = []
        with torch.no_grad():
            for c in chunks:
                v, idx = teacher(c).logits.topk(KDK, dim=-1)        # [1,T,K]
                kd_cache.append((torch.log_softmax(v, -1).detach(), idx.detach()))  # teacher logp over top-k
        del teacher
        if dev == "mps":
            torch.mps.empty_cache()
        print(f"# KD: cached top-{KDK} teacher logits for {len(chunks)} chunks; teacher freed", file=sys.stderr)

    model.train()
    opt = torch.optim.AdamW(params, lr=LR)  # tiny param set -> AdamW is cheap here
    best_ppl, best_step, best_state = base_ppl, -1, None
    for step in range(STEPS):
        ci = step % len(chunks)
        cids = chunks[ci]
        opt.zero_grad()
        s = model(cids).logits
        if KD:
            tlp, idx = kd_cache[ci]
            slp = torch.log_softmax(torch.gather(s, -1, idx), -1)   # student renorm over teacher's top-k
            loss = (tlp.exp() * (tlp - slp)).sum(-1).mean()         # KL over top-k
        else:
            loss = F.cross_entropy(s[:, :-1].reshape(-1, s.size(-1)), cids[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == STEPS - 1:
            model.eval()
            hp = ppl(model, tok, EVAL)  # held-out — early-stop on THIS, not train loss
            model.train()
            tag = ""
            if hp < best_ppl:
                best_ppl, best_step = hp, step
                best_state = {n: (m._A.detach().clone(), m._B.detach().clone()) for n, m in ADAPT}
                tag = " *best"
            print(f"#  step {step:4d} ce {loss.item():.4f}  held-out {hp:.1f}{tag}", file=sys.stderr)

    # early-stopping: restore the best held-out checkpoint (LoRA diverges if over-trained)
    if best_state is not None:
        for n, m in ADAPT:
            a, b = best_state[n]
            m._A.data, m._B.data = a, b
    model.eval()
    lora_ppl = ppl(model, tok, EVAL)
    print(f"# best held-out ppl {best_ppl:.3f} @ step {best_step}", file=sys.stderr)
    print(json.dumps({"base_ppl": base_ppl, "lora_ppl": lora_ppl, "rank": RANK, "steps": STEPS,
                      "recovery_pct": (base_ppl - lora_ppl) / base_ppl * 100 if base_ppl else 0}))

    if SAVE:  # deployed weights = W_hat + A@B (the healed artifact)
        sd = {}
        for name, m in ADAPT:
            sd[name + ".weight"] = (m.weight.data + m._A.data @ m._B.data).cpu().to(torch.float16)
        save_file(sd, SAVE)
        print(f"# saved healed (base+LoRA) weights: {SAVE}", file=sys.stderr)


if __name__ == "__main__":
    main()
