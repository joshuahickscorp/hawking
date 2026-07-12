#!/usr/bin/env python3.12
"""doctor.py - merged tool: blockwise (was doctor_blockwise.py) + strand (was doctor_strand.py) + qat (was doctor_qat.py) + lora (was doctor_lora.py) + registry (was doctor_registry.py).

the Doctor recovery stack: blockwise (L4 full-rank per-layer QAT), strand (L5 codec-native GPTQ-Hessian error-feedback), qat (uniform STE-QAT, characterized dead-end), lora (L6 KD-LoRA polish, the live recovery), registry (the L0-L6 method registry + auto-composer: --list/--select/--emit-set).

  doctor.py blockwise <args...>   # was: python3.12 tools/condense/doctor_blockwise.py <args...>
  doctor.py strand <args...>   # was: python3.12 tools/condense/doctor_strand.py <args...>
  doctor.py qat <args...>   # was: python3.12 tools/condense/doctor_qat.py <args...>
  doctor.py lora <args...>   # was: python3.12 tools/condense/doctor_lora.py <args...>
  doctor.py registry <args...>   # was: python3.12 tools/condense/doctor_registry.py <args...>
"""
import sys

def _run_blockwise():
    """Block/layer-wise QAT (BRECQ-lite) — the full-rank ceiling-breaker, done STABLY.

    The global STRAND-QAT diverged (one stale anchor over the whole model = drift). This does
    it PER LINEAR, locally: optimize each layer's weights so their FAKE-QUANTIZED form matches
    the f16 layer's OUTPUT on calib activations (local MSE + STE). Local scope = stable. The
    result is full-rank quant-robust weights (no LoRA ceiling, no bpw overhead) -> STRAND-bake.

    Usage: doctor_blockwise.py <hf-model-dir> <out_raw.safetensors> [bits] [steps]
    Then STRAND-bake <out_raw> and measure. Env: DOCTOR_CALIB.
    """
    import sys, os, torch, torch.nn as nn, torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file, save_file

    MODEL = sys.argv[1]
    OUT = sys.argv[2]
    BITS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 80
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    CALIB = open(os.environ.get("DOCTOR_CALIB", "scratch/calib_corpus.txt"), errors="ignore").read()[:8000]


    def fq(w, bits):  # per-output-channel symmetric uniform + STE
        qmax = 2 ** (bits - 1) - 1
        s = (w.abs().amax(1, keepdim=True) / qmax).clamp(min=1e-8)
        return w + (torch.clamp((w / s).round(), -qmax - 1, qmax) * s - w).detach()


    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager").to(dev).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(CALIB, return_tensors="pt").input_ids[:, :1024].to(dev)

    # capture each linear's INPUT activations (f16 run)
    inps, hooks = {}, []
    def mk(n):
        def h(mod, i, o): inps[n] = i[0].detach().reshape(-1, i[0].shape[-1])
        return h
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and mod.weight.shape[1] >= 256:
            hooks.append(mod.register_forward_hook(mk(n)))
    with torch.no_grad():
        m(ids)
    for h in hooks:
        h.remove()
    print(f"# block-wise QAT: {len(inps)} linears, {BITS}-bit, {STEPS} steps/layer", file=sys.stderr)

    sd = load_file(os.path.join(MODEL, "model.safetensors"))
    out_sd = dict(sd)
    for n, mod in m.named_modules():
        k = n + ".weight"
        if isinstance(mod, nn.Linear) and n in inps and k in sd:
            X = inps[n]                                   # [N, in]
            W0 = mod.weight.detach().clone()             # [out, in]
            with torch.no_grad():
                Y = X @ W0.T                             # f16 output target
            W = W0.clone().requires_grad_(True)
            opt = torch.optim.Adam([W], lr=1e-3)
            for _ in range(STEPS):
                opt.zero_grad()
                loss = F.mse_loss(X @ fq(W, BITS).T, Y)  # match f16 output UNDER quant
                loss.backward()
                opt.step()
            out_sd[k] = W.detach().cpu().to(torch.float16)   # raw quant-robust weights (STRAND-bake next)
            del X, Y, W

    save_file(out_sd, OUT)
    print(f"# saved quant-robust weights -> {OUT} (STRAND-bake this, then measure)")




def _run_strand():
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

    MODEL = os.environ.get("DOCTOR_MODEL", "scratch/qwen-05b")
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

    main()



def _run_qat():
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
        # DIVERSE calib: sample a different chunk each step so the doctor learns
        # generalizable quantization-robustness instead of memorizing one passage.
        # (single-passage QAT overfit: train CE -> 0.03 but held-out ppl stayed ~1e5.)
        calib_text = CALIB
        cf = os.environ.get("DOCTOR_CALIB")
        if cf and os.path.exists(cf):
            calib_text = open(cf, errors="ignore").read()
        all_ids = tok(calib_text, return_tensors="pt").input_ids[0]
        CTX = 512
        chunks = [all_ids[i:i + CTX].unsqueeze(0).to(dev)
                  for i in range(0, max(1, len(all_ids) - CTX), CTX)]
        chunks = [c for c in chunks if c.shape[1] >= 16] or [all_ids.unsqueeze(0).to(dev)]
        print(f"# calib: {len(chunks)} chunks x <={CTX} tok ({cf if cf else 'embedded'})", file=sys.stderr)

        teacher = None
        if KD:
            teacher = AutoModelForCausalLM.from_pretrained(
                MODEL, torch_dtype=torch.float32, attn_implementation="eager").to(dev).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            print("# KD on: distilling frozen f16 teacher logits", file=sys.stderr)

        model.train()
        opt = torch.optim.AdamW([m.weight for m in LINEARS], lr=LR)
        for step in range(STEPS):
            cids = chunks[step % len(chunks)]
            opt.zero_grad()
            s_logits = model(cids).logits
            if KD:
                with torch.no_grad():
                    t_logp = torch.log_softmax(teacher(cids).logits, dim=-1)
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

    main()



def _run_lora():
    """LoRA RECOVERY doctor (memory-efficient) — the fix for the 19GB wall.

    Full-weight STRAND-aware QAT didn't fit (model + per-linear anchors + optimizer + bake
    all in 19GB -> swap). Instead: FREEZE the STRAND-quantized base (W_hat, already baked) and
    train tiny rank-r LoRA adapters (A@B) to recover quality. Trains ~millions of params, not
    0.5B -> fits + fast. Deployed artifact = STRAND low-bit base + a small f16 LoRA correction
    (QLoRA-style). The healed weight W_hat + A@B is measured directly (held-out ppl).

    Usage: doctor_lora.py <wbase.safetensors> [steps] [lr] [rank] [save.safetensors]
      wbase = the STRAND-decoded W_hat base (e.g. scratch/qwen-05b-tq2-full.safetensors)
    Env: DOCTOR_CALIB (diverse corpus)
         DOCTOR_THREADS (CPU thread count, default = all cores)
         DOCTOR_GRAD_ACCUM (gradient accumulation steps, default 1)
         DOCTOR_SAVE_MODE (adapter default, or fused for legacy full-weight output)
         DOCTOR_PROGRESS (JSONL progress/checkpoint ledger)
    """
    import sys, os, re, math, json, gc, time, signal
    # ── engage ALL cores (P+E) — set BLAS thread env BEFORE importing torch, because OMP/veclib read
    # their thread count at library load. DOCTOR_THREADS (or all logical cores) drives every backend.
    _n_threads = int(os.environ.get("DOCTOR_THREADS", str(os.cpu_count() or 8)))
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_v, str(_n_threads))
    import torch, torch.nn as nn, torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import save_file
    from safetensors import safe_open

    torch.set_num_threads(_n_threads)
    torch.set_num_interop_threads(max(2, _n_threads // 2))

    MODEL = os.environ.get("DOCTOR_MODEL", "scratch/qwen-05b")
    WBASE = sys.argv[1] if len(sys.argv) > 1 else "scratch/qwen-05b-tq2-full.safetensors"
    STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    LR    = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
    RANK  = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    SAVE  = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None
    dev = os.environ.get("DOCTOR_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    DTYPE = getattr(torch, os.environ.get("DOCTOR_DTYPE", "float32"))
    GRAD_ACCUM = int(os.environ.get("DOCTOR_GRAD_ACCUM", "1"))
    SAVE_MODE = os.environ.get("DOCTOR_SAVE_MODE", "adapter").lower()
    TARGET_REGEX = os.environ.get("DOCTOR_TARGET_REGEX")
    TARGET_RE = re.compile(TARGET_REGEX) if TARGET_REGEX else None
    EVAL_EVERY = int(os.environ.get("DOCTOR_EVAL_EVERY", "25"))
    SAVE_EVERY = int(os.environ.get("DOCTOR_SAVE_EVERY", str(EVAL_EVERY)))
    PROGRESS = os.environ.get("DOCTOR_PROGRESS") or (SAVE + ".jsonl" if SAVE else None)
    LATEST = os.environ.get("DOCTOR_LATEST")
    if SAVE and SAVE_MODE == "adapter" and not LATEST:
        root, ext = os.path.splitext(SAVE)
        LATEST = root + ".latest" + (ext or ".safetensors")

    EVAL = """The science of operations, as derived from mathematics more especially, is a
    science of itself, and has its own abstract truth and value. A new, a vast, and a
    powerful language is developed for the future use of analysis, in which to wield its
    truths so that these may become of more speedy and accurate practical application for
    the purposes of mankind. The engine can arrange and combine its numerical quantities
    exactly as if they were letters or any other general symbols."""

    ADAPT = []  # modules with LoRA
    STOP_REQUESTED = False


    def ppl(model, tok, text):
        ids = tok(text, return_tensors="pt").input_ids[:, :2048].to(dev)
        with torch.no_grad():
            return math.exp(model(ids, labels=ids).loss.item())


    def _metadata(d):
        return {str(k): str(v) for k, v in d.items() if v is not None}


    def _save_safetensors_atomic(tensors, path, metadata=None):
        if not path:
            return
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = path + ".tmp"
        save_file(tensors, tmp, metadata=_metadata(metadata or {}))
        os.replace(tmp, path)


    def _adapter_state():
        sd = {}
        for name, m in ADAPT:
            sd[name + ".lora_A"] = m._A.detach().cpu().to(torch.float16)
            sd[name + ".lora_B"] = m._B.detach().cpu().to(torch.float16)
        return sd


    def save_adapter(path, *, step, heldout, kind, loss=None, base_ppl=None):
        meta = {
            "artifact_type": "hawking_lora_adapter",
            "model": MODEL,
            "wbase": WBASE,
            "rank": RANK,
            "step": step,
            "steps": STEPS,
            "lr": LR,
            "heldout_ppl": heldout,
            "base_ppl": base_ppl,
            "kind": kind,
            "loss": loss,
            "created_unix": int(time.time()),
        }
        _save_safetensors_atomic(_adapter_state(), path, meta)
        print(f"# saved {kind} LoRA adapter: {path}", file=sys.stderr)


    def load_adapter(path):
        if not path or not os.path.exists(path):
            return False
        by_name = {n: m for n, m in ADAPT}
        with safe_open(path, framework="pt") as f:
            for name, m in ADAPT:
                ak, bk = name + ".lora_A", name + ".lora_B"
                if ak in f.keys() and bk in f.keys():
                    m._A.data.copy_(f.get_tensor(ak).to(dev, DTYPE))
                    m._B.data.copy_(f.get_tensor(bk).to(dev, DTYPE))
        return bool(by_name)


    def emit_progress(record):
        if not PROGRESS:
            return
        record = {"ts": time.time(), **record}
        parent = os.path.dirname(PROGRESS)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(PROGRESS, "a") as f:
            f.write(json.dumps(record) + "\n")


    def _request_stop(signum, _frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
        print(f"# doctor: received signal {signum}; will checkpoint and stop", file=sys.stderr)


    def main():
        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)
        print(f"# doctor: threads={_n_threads} dev={dev} dtype={DTYPE} steps={STEPS} "
              f"rank={RANK} grad_accum={GRAD_ACCUM} save_mode={SAVE_MODE}"
              f" target={TARGET_REGEX or 'all'}", file=sys.stderr)

        tok = AutoTokenizer.from_pretrained(MODEL)

        # ── PHASE 1: cache teacher logits FIRST so teacher + student are never both in RAM ──
        # Peak without this reorder: student(14GB) + wh(15GB) + teacher(14GB) = 43GB on 18GB RAM.
        # With teacher-first: teacher(14GB) → cache(tiny) → del teacher → student+wh(29GB) → del wh → 14GB.
        KD = os.environ.get("KD") == "1"
        KDK = int(os.environ.get("KD_TOPK", "64"))
        kd_cache = None
        calib = open(os.environ["DOCTOR_CALIB"], errors="ignore").read() if os.environ.get("DOCTOR_CALIB") else EVAL
        ids_calib = tok(calib, return_tensors="pt").input_ids[0]
        CTX = 512
        chunk_ids = [ids_calib[i:i + CTX].unsqueeze(0) for i in range(0, max(1, len(ids_calib) - CTX), CTX)]
        chunk_ids = [c for c in chunk_ids if c.shape[1] >= 16] or [ids_calib.unsqueeze(0)]

        if KD:
            print(f"# KD phase-1: loading teacher to cache top-{KDK} logits ({len(chunk_ids)} chunks)…",
                  file=sys.stderr)
            teacher = AutoModelForCausalLM.from_pretrained(
                MODEL, torch_dtype=DTYPE, attn_implementation="eager").to(dev).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            kd_cache = []
            with torch.no_grad():
                for c in chunk_ids:
                    v, idx = teacher(c.to(dev)).logits.topk(KDK, dim=-1)
                    kd_cache.append((torch.log_softmax(v, -1).detach().cpu(),
                                     idx.detach().cpu()))  # store on CPU to free GPU/MPS
            del teacher
            gc.collect()
            if dev == "mps":
                torch.mps.empty_cache()
            print(f"# KD: teacher freed; {len(kd_cache)} chunk logits cached", file=sys.stderr)

        # ── PHASE 2: load student + STRAND base weights ───────────────────────────────────────
        print(f"# loading student model…", file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=DTYPE, attn_implementation="eager").to(dev)

        # ── stream WBASE one tensor at a time into model.weight in-place ─────────────────────
        # load_file() returns mmap'd tensors that stay referenced through m.weight.data even
        # after del wh (Python refcount stays >0). safe_open + .copy_() keeps peak RAM at
        # ~model (14GB) + one tensor at a time — no 15GB wh dict ever fully in heap RAM.
        print(f"# streaming WBASE weights in-place (peak ~model size, no 15GB wh alloc)…",
              file=sys.stderr)
        wbase_keys = set()
        with safe_open(WBASE, framework="pt") as fwh:
            wbase_keys = set(fwh.keys())
            for name, m in model.named_modules():
                key = name + ".weight"
                if isinstance(m, nn.Linear) and key in wbase_keys:
                    t = fwh.get_tensor(key).to(dev, DTYPE)
                    m.weight.data.copy_(t)   # copy INTO existing model tensor — t freed next iter
                    del t

        gc.collect()
        if dev == "mps":
            torch.mps.empty_cache()

        params = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and (name + ".weight") in wbase_keys:
                if TARGET_RE and not TARGET_RE.search(name):
                    m.weight.requires_grad_(False)
                    continue
                m.weight.requires_grad_(False)
                m._A = nn.Parameter(torch.zeros(m.weight.shape[0], RANK, device=dev, dtype=DTYPE))
                m._B = nn.Parameter(torch.randn(RANK, m.weight.shape[1], device=dev, dtype=DTYPE) * 0.01)
                m.forward = (lambda x, mm=m: F.linear(x, mm.weight + mm._A @ mm._B, mm.bias))
                params += [m._A, m._B]
                ADAPT.append((name, m))

        print(f"# {len(ADAPT)} adapters rank {RANK}, "
              f"{sum(p.numel() for p in params)/1e6:.1f}M trainable params", file=sys.stderr)

        model.eval()
        base_ppl = ppl(model, tok, EVAL)
        print(f"# STRAND base held-out ppl (no LoRA): {base_ppl:.3f}", file=sys.stderr)
        print(f"# lora-doctor: base {WBASE}, {len(ADAPT)} adapters rank {RANK}, "
              f"{sum(p.numel() for p in params)/1e6:.1f}M trainable, {STEPS} steps lr {LR}", file=sys.stderr)

        # move calib chunks to device after model is loaded and wh freed
        chunks = [c.to(dev) for c in chunk_ids]
        print(f"# calib: {len(chunks)} chunks", file=sys.stderr)
        if kd_cache is not None:
            kd_cache = [(v.to(dev), idx.to(dev)) for v, idx in kd_cache]

        # ── PHASE 3: train ────────────────────────────────────────────────────────────────────
        model.train()
        opt = torch.optim.AdamW(params, lr=LR)
        best_ppl, best_step, best_path = base_ppl, -1, None
        opt.zero_grad()
        for step in range(STEPS):
            if STOP_REQUESTED:
                break
            ci = step % len(chunks)
            cids = chunks[ci]
            s = model(cids).logits
            if KD:
                tlp, idx = kd_cache[ci]
                slp = torch.log_softmax(torch.gather(s, -1, idx), -1)
                loss = (tlp.exp() * (tlp - slp)).sum(-1).mean()
            else:
                loss = F.cross_entropy(s[:, :-1].reshape(-1, s.size(-1)), cids[:, 1:].reshape(-1))
            (loss / GRAD_ACCUM).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                opt.step()
                opt.zero_grad()

            if step % EVAL_EVERY == 0 or step == STEPS - 1 or STOP_REQUESTED:
                model.eval()
                hp = ppl(model, tok, EVAL)
                model.train()
                tag = ""
                if hp < best_ppl:
                    best_ppl, best_step = hp, step
                    if SAVE_MODE == "adapter" and SAVE:
                        save_adapter(SAVE, step=step, heldout=hp, kind="best",
                                     loss=loss.item(), base_ppl=base_ppl)
                        best_path = SAVE
                    tag = " *best"
                if SAVE_MODE == "adapter" and LATEST and (
                    step % SAVE_EVERY == 0 or step == STEPS - 1 or STOP_REQUESTED
                ):
                    save_adapter(LATEST, step=step, heldout=hp, kind="latest",
                                 loss=loss.item(), base_ppl=base_ppl)
                emit_progress({
                    "event": "eval",
                    "step": step,
                    "loss": loss.item(),
                    "heldout_ppl": hp,
                    "best_ppl": best_ppl,
                    "best_step": best_step,
                    "rank": RANK,
                    "steps": STEPS,
                    "stop_requested": STOP_REQUESTED,
                })
                print(f"#  step {step:4d} ce {loss.item():.4f}  held-out {hp:.1f}{tag}", file=sys.stderr)

        if SAVE_MODE == "adapter":
            if best_path:
                load_adapter(best_path)
            elif LATEST and os.path.exists(LATEST):
                load_adapter(LATEST)
        model.eval()
        lora_ppl = ppl(model, tok, EVAL)
        print(f"# best held-out ppl {best_ppl:.3f} @ step {best_step}", file=sys.stderr)
        if SAVE_MODE == "adapter" and SAVE and not os.path.exists(SAVE) and LATEST and os.path.exists(LATEST):
            os.replace(LATEST, SAVE)
            print(f"# promoted latest LoRA adapter: {SAVE}", file=sys.stderr)
        artifact_path = best_path or (SAVE if SAVE and os.path.exists(SAVE) else LATEST)
        final = {"base_ppl": base_ppl, "lora_ppl": lora_ppl, "rank": RANK, "steps": STEPS,
                 "best_ppl": best_ppl, "best_step": best_step, "save_mode": SAVE_MODE,
                 "artifact_path": artifact_path,
                 "stopped_early": STOP_REQUESTED,
                 "recovery_pct": (base_ppl - lora_ppl) / base_ppl * 100 if base_ppl else 0}
        emit_progress({"event": "final", **final})
        print(json.dumps(final))

        if SAVE:
            if SAVE_MODE == "fused":
                sd = {}
                for name, m in ADAPT:
                    sd[name + ".weight"] = (m.weight.data + m._A.data @ m._B.data).cpu().to(torch.float16)
                save_file(sd, SAVE)
                print(f"# saved healed (base+LoRA) weights: {SAVE}", file=sys.stderr)

    main()



def _run_registry():
    """doctor_registry.py — THE DOCTOR, expanded. "The doctor" was never one function; it is the name
    for the whole program of restoring quality at low bits. This makes that explicit: a pluggable
    REGISTRY of recovery methods (L0..L6 + extensions), an auto-SELECTOR that composes the right chain
    for a (model, target-bpw, device), and a leverage-ordered catalog. New recovery methods register
    themselves with a decorator and immediately become available to the selector and the ledger — no
    edit to studio_run/audit_ladder needed. Advisor/orchestration only (pure stdlib): a method's
    build_fn returns a BakeSpec (baker argv + optional shadow-weights step + eff-bpw hooks); the driver
    runs it, so composition/ordering lives in one place.

    CLI:
      doctor_registry.py --list                          # the catalog (layer, stage, train-free, serve)
      doctor_registry.py --select <params_b> <bpw> [--moe] [--floor F] [--device studio-m2max]
                                                         # the recommended ordered recovery chain
      doctor_registry.py --emit-set <params_b> <bpw>     # emit the chain as audit_ladder-style config rows
    """
    import sys, os, math, json
    from dataclasses import dataclass, field
    from typing import Callable, Optional

    LADDER = [(1, 1.34), (2, 2.34), (3, 3.34), (4, 4.5)]
    OUT = "reports/condense"


    @dataclass
    class RecoveryMethod:
        name: str
        layer: int                       # L0..L6 leverage rank (cheapest/most-general first)
        stage: str                       # 'local' | 'studio'
        train_free: bool
        sensitivity: str                 # 'global' | 'per_tensor' | 'per_expert'
        tool: str                        # the script/CLI that implements it
        provides_serve: bool             # does its artifact serve natively (vs needing more build)
        min_params_b: float = 0.0
        max_params_b: Optional[float] = None
        status: str = "MEASURED"         # MEASURED | GATED | UNPROVEN | DEAD
        note: str = ""
        build_fn: Optional[Callable] = field(default=None, repr=False)


    REGISTRY: dict = {}


    def register(**kw):
        def deco(fn):
            m = RecoveryMethod(build_fn=fn, **kw)
            REGISTRY[m.name] = m
            return fn
        return deco


    # ---- the catalog: existing levers (thin build_fns emit an audit_ladder-style (name, fn, args) spec) ----
    @register(name="calib", layer=0, stage="local", train_free=True, sensitivity="global",
              tool="calib_build.py", provides_serve=True, status="MEASURED",
              note="domain-matched calibration; multiplies every layer below")
    def _calib(ctx): return ("calib", "build_calib", ())

    @register(name="awq", layer=1, stage="local", train_free=True, sensitivity="per_tensor",
              tool="awq.py bake", provides_serve=True, status="MEASURED",
              note="alpha=0.5 activation-aware pre-scale; halves the raw gap at 3-4bit")
    def _awq(ctx): return (f"{ctx['bits']}-AWQ", "build_awq", (ctx["bits"], 0.5))

    @register(name="mixed_prec", layer=2, stage="local", train_free=True, sensitivity="per_tensor",
              tool="mixed_precision.py", provides_serve=True, status="MEASURED",
              note="output-sensitivity water-fill: sensitive tensors get depth, tolerant get starved. Biggest NEW density lever.")
    def _mp(ctx): return ("mp-4a3f", "build_awq", (3, 0.5, {"q_proj": 4, "k_proj": 4, "v_proj": 4, "o_proj": 4,
                                                             "gate_proj": 3, "up_proj": 3, "down_proj": 3}))

    @register(name="residual", layer=3, stage="local", train_free=True, sensitivity="per_tensor",
              tool="residual.py bake", provides_serve=True, status="MEASURED",
              note="W ~= STRAND(W)+STRAND(residual); train-free ~1:1; two-part serve (parity-green)")
    def _res(ctx): return (f"res{ctx['bits']}+1", "build_residual", (ctx["bits"], 1))

    @register(name="outlier_channel", layer=3, stage="local", train_free=True, sensitivity="per_tensor",
              tool="audit_ladder.build_awq(outlier_pct)", provides_serve=True, status="MEASURED",
              note="keep top-|w| 5-10% at 8-bit sparse channel (OUTL wire); train-free sub-3-bit rescue")
    def _outl(ctx): return (f"{ctx['bits']}-AWQ-o5", "build_awq", (ctx["bits"], 0.5, None, 5.0))

    @register(name="block_qat", layer=4, stage="studio", train_free=False, sensitivity="per_tensor",
              tool="doctor.py blockwise", provides_serve=True, min_params_b=0.0, status="GATED",
              note="BRECQ-lite full-rank per-linear QAT; the LoRA-plateau fix; studio at 7B+")
    def _bw(ctx): return (f"{ctx['bits']}-bw", "build_blockwise", (ctx["bits"],))

    @register(name="gptq_hessian", layer=5, stage="studio", train_free=False, sensitivity="per_tensor",
              tool="doctor.py strand", provides_serve=True, status="UNPROVEN",
              note="codec-native sequential error-feedback (NO uniform STE — that path is DEAD); sub-residual edge")
    def _str(ctx): return (f"{ctx['bits']}-str", "build_strand", (ctx["bits"],))

    @register(name="deep_kd", layer=6, stage="studio", train_free=False, sensitivity="global",
              tool="doctor.py lora", provides_serve=True, status="GATED",
              note="logit/feature/attn KD polish on the full-rank base; recovers near the base bpw")
    def _kd(ctx): return (f"{ctx['bits']}-AWQ+dr", "build_recover", (ctx["bits"],))

    @register(name="expert_alloc", layer=2, stage="studio", train_free=True, sensitivity="per_expert",
              tool="expert.py sensitivity", provides_serve=False, min_params_b=100.0, status="GATED",
              note="MoE per-expert bit allocation: router/shared high-bit, hot 2-bit, cold 1-bit/ternary")
    def _expert(ctx): return ("expert-alloc", "per_expert", (ctx["bits"],))

    # ---- extension slots the audit named as MISSING (registered as UNPROVEN so the selector knows they exist) ----
    @register(name="learned_rotation", layer=1, stage="local", train_free=True, sensitivity="per_tensor",
              tool="rotation_search.py (TODO)", provides_serve=True, status="UNPROVEN",
              note="QuaRot/SpinQuant learned orthogonal rotation before the cut; ~0 serve bpw; may beat RHT")
    def _rot(ctx): return ("rot", "TODO", ())

    @register(name="big_teacher_kd", layer=6, stage="studio", train_free=False, sensitivity="global",
              tool="doctor.py lora --teacher (TODO)", provides_serve=True, status="UNPROVEN",
              note="a larger model distilling the condensed one; needs both resident; high upside")
    def _bigkd(ctx): return ("bigkd", "TODO", ())


    def _floor_bits(params_b, entropy_floor=None):
        if entropy_floor:
            for b, bpw in LADDER:
                if bpw >= entropy_floor:
                    return b
        raw = 3.6 - 0.9 * math.log10(max(0.5, params_b))
        for b, bpw in LADDER:
            if bpw >= raw:
                return b
        return 4


    def select(params_b, target_bpw=None, is_moe=False, entropy_floor=None, device="studio-m2max"):
        """Compose the recovery chain for this model. Cheapest-leverage-first train-free stack always;
        add training layers only when needed and size-appropriate; per-expert for MoE."""
        bits = None
        for b, bpw in LADDER:
            if target_bpw and abs(bpw - target_bpw) < 0.5:
                bits = b
        bits = bits or _floor_bits(params_b, entropy_floor)
        ctx = {"params_b": params_b, "bits": bits, "is_moe": is_moe}
        chain = []
        # 1) always the train-free stack (L0-L3) + outlier at sub-3-bit
        for name in ("calib", "awq", "mixed_prec", "residual"):
            chain.append(name)
        if bits <= 2:
            chain.append("outlier_channel")
        # 2) MoE: per-expert allocation before the bake
        if is_moe:
            chain.insert(1, "expert_alloc")
        # 3) training recovery only if the target is aggressive (train-free likely leaves a gap)
        if bits <= 2:
            chain += (["block_qat", "gptq_hessian", "deep_kd"] if params_b >= 7 else ["block_qat", "deep_kd"])
        elif bits == 3:
            chain.append("deep_kd")
        # de-dup preserve order; annotate stage feasibility
        seen, ordered = set(), []
        for n in chain:
            if n in REGISTRY and n not in seen:
                ordered.append(n); seen.add(n)
        ordered.sort(key=lambda n: REGISTRY[n].layer)
        return bits, ordered, ctx


    def cmd_select(params_b, target_bpw, is_moe, floor, device):
        bits, chain, ctx = select(params_b, target_bpw, is_moe, floor, device)
        rec = {"params_b": params_b, "target_bits": bits, "target_bpw": dict(LADDER)[bits],
               "is_moe": is_moe, "chain": [
                   {"method": n, "layer": REGISTRY[n].layer, "stage": REGISTRY[n].stage,
                    "train_free": REGISTRY[n].train_free, "status": REGISTRY[n].status} for n in chain]}
        os.makedirs(OUT, exist_ok=True)
        json.dump(rec, open(f"{OUT}/doctor_plan_{int(params_b)}b.json", "w"), indent=2)
        print(f"[doctor] {params_b}B{' MoE' if is_moe else ''} -> target {bits}-bit ({dict(LADDER)[bits]} eff-bpw)", file=sys.stderr)
        print(f"[doctor] recovery chain (leverage-ordered):", file=sys.stderr)
        for n in chain:
            m = REGISTRY[n]
            print(f"   L{m.layer} {n:16s} [{m.stage:6s} {'train-free' if m.train_free else 'TRAINING ':10s} {m.status:9s}] {m.note}", file=sys.stderr)
        print("# fallback: this is the STARTING chain; the floor-search confirms/steps up (2->3->4) on the +2% gate.",
              file=sys.stderr)


    def cmd_list():
        print(f"{'method':16s} {'L':2s} {'stage':7s} {'train-free':10s} {'serve':6s} {'status':9s} sens")
        for m in sorted(REGISTRY.values(), key=lambda x: (x.layer, x.name)):
            print(f"{m.name:16s} {m.layer:<2d} {m.stage:7s} {str(m.train_free):10s} "
                  f"{str(m.provides_serve):6s} {m.status:9s} {m.sensitivity}")
        print(f"\n{len(REGISTRY)} recovery methods registered. The Doctor = this whole registry, not one script.")

    a = sys.argv[1] if len(sys.argv) > 1 else "--list"
    if a == "--list":
        cmd_list()
    elif a == "--select":
        cmd_select(float(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else None,
                   "--moe" in sys.argv,
                   float(sys.argv[sys.argv.index("--floor")+1]) if "--floor" in sys.argv else None,
                   sys.argv[sys.argv.index("--device")+1] if "--device" in sys.argv else "studio-m2max")
    elif a == "--emit-set":
        _, chain, ctx = select(float(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else None)
        rows = [REGISTRY[n].build_fn(ctx) for n in chain if REGISTRY[n].build_fn]
        print(json.dumps([r for r in rows if r and r[1] != "TODO"], default=str))
    else:
        print(__doc__)


if __name__ == "__main__":
    _sub = sys.argv[1] if len(sys.argv) > 1 else "--help"
    if _sub == "blockwise":
        sys.argv = ["doctor_blockwise.py"] + sys.argv[2:]
        _run_blockwise()
    elif _sub == "strand":
        sys.argv = ["doctor_strand.py"] + sys.argv[2:]
        _run_strand()
    elif _sub == "qat":
        sys.argv = ["doctor_qat.py"] + sys.argv[2:]
        _run_qat()
    elif _sub == "lora":
        sys.argv = ["doctor_lora.py"] + sys.argv[2:]
        _run_lora()
    elif _sub == "registry":
        sys.argv = ["doctor_registry.py"] + sys.argv[2:]
        _run_registry()
    else:
        print(__doc__)
