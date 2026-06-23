#!/usr/bin/env python3.12
"""STRAND-AWARE doctor — QAT with the REAL STRAND codec in the loop.

Verdict v2 proved uniform-proxy QAT is counterproductive for STRAND (uniform-healed
STRAND-TQ2 ppl 676 > 481 PTQ): optimizing weights for uniform mis-optimizes them for
STRAND's trellis. Fix: anchor the forward to the ACTUAL STRAND recon every `requant`
steps. base = STRAND_recon - weight (detached), forward = weight + base. Right after a
requant, weight+base == the exact STRAND recon; between requants the weights drift and
re-anchor at the next requant. So the model is tuned for STRAND, not a proxy.

Usage: doctor_strand.py [bits] [steps] [lr] [requant_every] [save.safetensors]
Env: DOCTOR_CALIB (corpus file for diverse chunks).
"""
import sys, os, math, json, subprocess, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file, load_file

MODEL = "scratch/qwen-05b"
BITS  = int(sys.argv[1]) if len(sys.argv) > 1 else 2
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
LR    = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-5
REQ   = int(sys.argv[4]) if len(sys.argv) > 4 else 50
SAVE  = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None
BAKER = "vendor/strand-quant/target/release/quantize-model"
TMP_IN, TMP_OUT = "/tmp/_dr_shadow.safetensors", "/tmp/_dr_recon.safetensors"
dev = "mps" if torch.backends.mps.is_available() else "cpu"

EVAL = """The science of operations, as derived from mathematics more especially, is a
science of itself, and has its own abstract truth and value. A new, a vast, and a
powerful language is developed for the future use of analysis, in which to wield its
truths so that these may become of more speedy and accurate practical application for
the purposes of mankind. The engine can arrange and combine its numerical quantities
exactly as if they were letters or any other general symbols."""

LINEARS = []


def collect(model):
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.weight.shape[0] > 1 and m.weight.shape[1] >= 256:
            m._base = torch.zeros_like(m.weight.data)
            m.forward = (lambda x, mm=m: F.linear(x, mm.weight + mm._base, mm.bias))
            LINEARS.append((name, m))


def strand_requant():
    """Bake the current shadow with the REAL STRAND codec; set base = recon - weight."""
    sd = {n + ".weight": m.weight.detach().cpu().to(torch.float16) for n, m in LINEARS}
    save_file(sd, TMP_IN)
    subprocess.run([BAKER, "--in", TMP_IN, "--out", TMP_OUT, "--bits", str(BITS),
                    "--quality", "--rht-cols", "--outlier-channel", "1", "--outlier-bits", "8"],
                   check=True, capture_output=True)
    recon = load_file(TMP_OUT)
    for n, m in LINEARS:
        r = recon[n + ".weight"].to(m.weight.device, torch.float32)
        m._base = (r - m.weight.detach()).detach()


def ppl(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids[:, :2048].to(dev)
    with torch.no_grad():
        return math.exp(model(ids, labels=ids).loss.item())


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager").to(dev)
    collect(model)
    print(f"# strand-doctor: {BITS}-bit, {len(LINEARS)} linears, {STEPS} steps, lr {LR}, requant/{REQ}", file=sys.stderr)

    # calib chunks (diverse — else overfit, see verdict v1)
    calib = open(os.environ["DOCTOR_CALIB"], errors="ignore").read() if os.environ.get("DOCTOR_CALIB") else EVAL
    ids = tok(calib, return_tensors="pt").input_ids[0]
    CTX = 512
    chunks = [ids[i:i + CTX].unsqueeze(0).to(dev) for i in range(0, max(1, len(ids) - CTX), CTX)]
    chunks = [c for c in chunks if c.shape[1] >= 16] or [ids.unsqueeze(0).to(dev)]
    print(f"# calib: {len(chunks)} chunks", file=sys.stderr)

    model.eval()
    strand_requant()  # base now = STRAND recon - weight  => forward == STRAND PTQ recon
    ptq = ppl(model, tok, EVAL)
    print(f"# STRAND PTQ {BITS}-bit held-out ppl: {ptq:.3f}", file=sys.stderr)

    model.train()
    # SGD+momentum (1 state tensor) not AdamW (2): the per-linear STRAND `_base` anchors
    # already cost ~weight-size each; AdamW's m+v tipped a 19GB Mac into swap (~25x slower).
    opt = torch.optim.SGD([m.weight for _, m in LINEARS], lr=LR, momentum=0.9)
    for step in range(STEPS):
        if step > 0 and step % REQ == 0:
            strand_requant()  # re-anchor to the real codec
        cids = chunks[step % len(chunks)]
        opt.zero_grad()
        s = model(cids).logits
        loss = F.cross_entropy(s[:, :-1].reshape(-1, s.size(-1)), cids[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        if step % REQ == 0 or step == STEPS - 1:
            print(f"#  step {step:4d} ce {loss.item():.4f}", file=sys.stderr)

    model.eval()
    strand_requant()  # final: forward == STRAND recon of the healed shadow
    qat = ppl(model, tok, EVAL)
    print(json.dumps({"bits": BITS, "steps": STEPS, "requant": REQ,
                      "strand_ptq_ppl": ptq, "strand_doctor_ppl": qat,
                      "recovery_pct": (ptq - qat) / ptq * 100 if ptq else 0}))

    if SAVE:  # save the healed shadow so quality_3way can STRAND-bake + bench it
        raw = {n + ".weight": m.weight.detach().cpu().to(torch.float16) for n, m in LINEARS}
        save_file(raw, SAVE.replace(".safetensors", ".raw.safetensors"))
        print(f"# saved healed shadow: {SAVE.replace('.safetensors','.raw.safetensors')}", file=sys.stderr)


if __name__ == "__main__":
    main()
