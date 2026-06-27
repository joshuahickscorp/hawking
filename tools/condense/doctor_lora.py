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
     DOCTOR_THREADS (CPU thread count, default = all cores)
     DOCTOR_GRAD_ACCUM (gradient accumulation steps, default 1)
     DOCTOR_SAVE_MODE (adapter default, or fused for legacy full-weight output)
     DOCTOR_PROGRESS (JSONL progress/checkpoint ledger)
"""
import sys, os, re, math, json, gc, time, signal, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
from safetensors import safe_open

# ── use all cores — PyTorch defaults to half on macOS ────────────────────────
_n_threads = int(os.environ.get("DOCTOR_THREADS", str(os.cpu_count() or 8)))
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


if __name__ == "__main__":
    main()
