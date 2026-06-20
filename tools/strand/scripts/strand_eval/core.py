# strand_eval.core — the eval engine + the by-construction identity helpers.
#
# Torch-free helpers (model_id_from_dir, harness_key, output naming, record build)
# live at module top so the unit tests and the ledger never import torch.
# Everything torch-bound is inside functions.

import hashlib
import json
import math
import os
import re
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from strand_eval import HARNESS_VERSION, SCHEMA

# Dataset identity: the canon fallback chain (older hubs take the bare legacy id —
# may be cached locally, no network; newer hubs REQUIRE namespace/name and raise on
# the bare name). Same wikitext-2-raw-v1 test split either way → comparable PPL,
# and the content fingerprint recorded per-result PROVES it.
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_IDS = ("wikitext", "Salesforce/wikitext")

DTYPES = ("bfloat16", "float16", "float32")
DEVICES = ("auto", "cpu", "mps")

# Leaf dir names that carry no model identity (recon dumps, HF export dirs, ...).
_GENERIC_LEAVES = {
    "recon", "model", "models", "hf", "out", "output", "shadow", "weights",
    "checkpoint", "checkpoints", "snapshot", "snapshots", "final", "best",
}


def sanitize(s):
    """Filename-safe identity component."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-._")
    return s or "unnamed"


def model_id_from_dir(load_dir):
    """Derive the model id from the model dir BY CONSTRUCTION (never caller-typed).

    Takes the first path component, walking up from the resolved leaf, that is not
    a generic container name ('recon', 'hf', ...). The write-time collision guard
    in output_path() covers the residual case of two different dirs reducing to
    the same id.
    """
    p = os.path.realpath(load_dir)
    while p and p != os.path.dirname(p):
        leaf = os.path.basename(p)
        if leaf.lower() not in _GENERIC_LEAVES and not leaf.isdigit():
            return sanitize(leaf)
        p = os.path.dirname(p)
    return sanitize(os.path.basename(os.path.realpath(load_dir)))


def path_fp(load_dir, n=6):
    return hashlib.sha256(os.path.realpath(load_dir).encode()).hexdigest()[:n]


def harness_key(device, dtype, ctx, chunks, dataset_id):
    """The comparability class, one canonical dict + an 8-hex key.

    Two results are directly comparable IFF their harness_key8 match. Fields per
    the audit: module version, RESOLVED device, dtype, ctx, chunk count, dataset id.
    """
    hk = {
        "harness_version": HARNESS_VERSION,
        "device": str(device),
        "dtype": str(dtype),
        "ctx": int(ctx),
        "chunks": int(chunks),
        "dataset_id": str(dataset_id),
    }
    canon = json.dumps(hk, sort_keys=True, separators=(",", ":"))
    return hk, hashlib.sha256(canon.encode()).hexdigest()[:8]


def output_path(out_dir, model_id, tag, model_path):
    """ppl_<model-id>_<tag>.json, derived INSIDE the writer — flat, caller-typed,
    or colliding names are unrepresentable.

    Collision guard: if the derived name already exists in out_dir and belongs to
    a DIFFERENT model_path, the name gains a 6-hex path fingerprint instead of
    overwriting (the llama2-overwrote-qwen incident, closed by construction)."""
    base = f"ppl_{sanitize(model_id)}_{sanitize(tag)}.json"
    p = os.path.join(out_dir, base)
    if os.path.exists(p):
        try:
            with open(p) as f:
                prev = json.load(f)
            prev_path = os.path.realpath(prev.get("model_path", prev.get("model", "")))
            if prev_path and prev_path != os.path.realpath(model_path):
                fp = path_fp(model_path)
                return os.path.join(out_dir, f"ppl_{sanitize(model_id)}-{fp}_{sanitize(tag)}.json")
        except (OSError, ValueError):
            pass  # unreadable previous file: same-name overwrite is the honest move
    return p


def build_record(*, model_path, tag, ppl, ctx, chunks, tokens, device_resolved,
                 device_mode, dtype, dataset_id, dataset_fp, out_json=None,
                 eff_bpw=None, extra=None):
    """One canonical result record (json file == ledger line, same shape)."""
    hk, hk8 = harness_key(device_resolved, dtype, ctx, chunks, dataset_id)
    rec = {
        "schema": SCHEMA,
        "harness_version": HARNESS_VERSION,
        "harness_key": hk,
        "harness_key8": hk8,
        "model": model_id_from_dir(model_path),
        "model_path": os.path.realpath(model_path),
        "tag": tag,
        "ppl": ppl,
        "ctx": ctx,
        "chunks": chunks,
        "tokens": tokens,
        "device": str(device_resolved),
        "device_mode": device_mode,
        "dtype": dtype,
        "dataset_id": dataset_id,
        "dataset_fp": dataset_fp,
        "eff_bpw": eff_bpw,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "epoch": int(time.time()),
        "out_json": out_json,
    }
    if extra:
        rec.update(extra)
    return rec


def sidecar_eff_bpw(load_dir):
    """Best-effort: mean aggregate.effective_bpw over quantize-model sidecars in
    the recon dir. None when absent (bf16 baselines, raw HF dirs)."""
    import glob
    vals = []
    for p in glob.glob(os.path.join(load_dir, "*.safetensors.json")):
        try:
            with open(p) as f:
                v = json.load(f).get("aggregate", {}).get("effective_bpw")
            if v is not None:
                vals.append(float(v))
        except (OSError, ValueError):
            continue
    return (sum(vals) / len(vals)) if vals else None


# ---------------------------------------------------------------------------
# torch-bound section
# ---------------------------------------------------------------------------

def resolve_device(device_s):
    """'auto'|'cpu'|'mps' -> (mode, resolved torch.device str).

    The RECORDED device is the resolved placement, never the mode string (the
    eval-ppl.py divergence the audit flagged)."""
    import torch
    if device_s == "auto":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "auto", "mps"
        return "auto", "cpu"
    return device_s, device_s


def load_model_and_tokenizer(load_dir, device_s, dtype_s, gpu_gb=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float16": torch.float16, "float32": torch.float32,
                   "bfloat16": torch.bfloat16}[dtype_s]
    if dtype_s == "float16":
        print("[ppl] WARNING: fp16 overflows to NaN on Qwen2.5 — the canon is bf16",
              file=sys.stderr, flush=True)

    mode, resolved = resolve_device(device_s)
    tok = AutoTokenizer.from_pretrained(load_dir, trust_remote_code=True)
    kw = dict(torch_dtype=torch_dtype, low_cpu_mem_usage=True, trust_remote_code=True)

    dev = resolved.split(":")[0]
    if dev == "mps":
        # MPS canon quirks (from strand-7b-ppl.sh): eager attention (SDPA cannot
        # broadcast Qwen GQA on MPS) + disable the whole-model allocator warmup
        # (fails on 7B even when tensor-by-tensor loading fits).
        kw["attn_implementation"] = "eager"
        import transformers.modeling_utils as mu
        mu.caching_allocator_warmup = lambda *a, **k: None
        model = AutoModelForCausalLM.from_pretrained(load_dir, device_map=resolved, **kw)
        input_device = torch.device(resolved)
    elif dev == "cpu":
        model = AutoModelForCausalLM.from_pretrained(load_dir, **kw)
        input_device = torch.device("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(load_dir, device_map=resolved, **kw)
        input_device = torch.device(resolved)

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model, tok, mode, resolved, input_device


def load_wikitext(tok, split="test"):
    """Canon dataset load with the fallback chain. Returns (ids, dataset_id, fp16hex).

    fp = sha256 of the joined raw text — the cheap content fingerprint that PROVES
    'same split' across hub versions / mirrors / caches."""
    from datasets import load_dataset
    ds, used, errs = None, None, []
    for ds_id in DATASET_IDS:
        try:
            ds = load_dataset(ds_id, DATASET_CONFIG, split=split)
            used = ds_id
            break
        except Exception as e:  # noqa: BLE001 — the fallback chain is the point
            errs.append(f"{ds_id}: {type(e).__name__}: {e}")
    if ds is None:
        raise SystemExit("[ppl] WikiText-2 load failed for all candidate ids:\n  "
                         + "\n  ".join(errs))
    text = "\n\n".join(ds["text"])
    fp = hashlib.sha256(text.encode()).hexdigest()[:16]
    ids = tok(text, return_tensors="pt").input_ids[0]
    return ids, used, fp


def eval_chunks(model, chunk_list, input_device, ce_slice=0, progress=True):
    """The ONE sum-CE loop. Σnll/Σtok over non-overlapping windows.

    ce_slice=0 → whole-window CE (bit-faithful to the canon heredoc).
    ce_slice=N → N-row slices (CE-sum is additive over rows: identical Σnll in
    exact arithmetic; the MPS memory fix from strand-qat.py — full-vocab
    log_softmax over 2047×152k rows is a ~2.5GB transient)."""
    import torch
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
    is_mps = getattr(input_device, "type", str(input_device)) == "mps" or \
        str(input_device).startswith("mps")
    nll, ntok = 0.0, 0
    t0 = time.time()
    n = len(chunk_list)
    with torch.no_grad():
        for i, ch in enumerate(chunk_list):
            ids = ch.unsqueeze(0).to(input_device)
            try:
                logits = model(ids, use_cache=False).logits
            except RuntimeError as e:
                # MPS allocator fragmentation after long runs: defrag + retry once
                if not (is_mps and "MPS backend out of memory" in str(e)):
                    raise
                torch.mps.empty_cache()
                logits = model(ids, use_cache=False).logits
            sl = logits[:, :-1, :].reshape(-1, logits.size(-1))
            lab = ids[:, 1:].reshape(-1).to(sl.device)
            if ce_slice and ce_slice > 0:
                for j in range(0, lab.numel(), ce_slice):
                    nll += loss_fct(sl[j:j + ce_slice].float(), lab[j:j + ce_slice]).item()
            else:
                nll += loss_fct(sl.float(), lab).item()
            ntok += lab.numel()
            del logits, sl
            if progress:
                print(f"[ppl] {i+1}/{n}  ppl={math.exp(nll/ntok):.4f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
    if is_mps:
        import torch as _t
        _t.mps.empty_cache()
    return nll, ntok


def run_eval(load_dir, tag, ctx=2048, limit_chunks=64, device="auto",
             dtype="bfloat16", out_dir=None, ledger_path=None, gpu_gb=None,
             ce_slice=None, no_ledger=False):
    """The canonical entrypoint. Returns (record, out_json_path).

    Output name + harness_key + ledger append are BY CONSTRUCTION — callers pass
    a model dir and a tag, nothing else identity-bearing."""
    from strand_eval.ledger import append_record
    from strand_eval import default_ledger_path

    if not os.path.isdir(load_dir):
        raise SystemExit(f"[ppl] model dir not found: {load_dir}")
    out_dir = out_dir or os.path.realpath(load_dir)
    os.makedirs(out_dir, exist_ok=True)

    model, tok, mode, resolved, input_device = load_model_and_tokenizer(
        load_dir, device, dtype, gpu_gb)
    print(f"[ppl] loaded '{load_dir}' mode={mode} resolved={resolved} ({dtype})", flush=True)

    enc, dataset_id, dataset_fp = load_wikitext(tok)
    n_chunks = enc.shape[0] // ctx
    if limit_chunks > 0:
        n_chunks = min(n_chunks, limit_chunks)
    if n_chunks == 0:
        raise SystemExit(f"[ppl] ctx={ctx} too large for {enc.shape[0]} tokens")
    chunk_list = [enc[i * ctx:(i + 1) * ctx] for i in range(n_chunks)]

    # MPS defaults to sliced CE (memory law); CPU keeps the canon whole-window CE.
    if ce_slice is None:
        ce_slice = 512 if str(resolved).startswith("mps") else 0

    nll, ntok = eval_chunks(model, chunk_list, input_device, ce_slice=ce_slice)
    ppl = math.exp(nll / ntok)

    import torch
    import transformers
    rec = build_record(
        model_path=load_dir, tag=tag, ppl=ppl, ctx=ctx, chunks=n_chunks,
        tokens=ntok, device_resolved=resolved, device_mode=device, dtype=dtype,
        dataset_id=dataset_id, dataset_fp=dataset_fp,
        eff_bpw=sidecar_eff_bpw(load_dir),
        extra={"ce_slice": ce_slice, "torch": torch.__version__,
               "transformers": transformers.__version__})

    out_json = output_path(out_dir, rec["model"], tag, load_dir)
    rec["out_json"] = out_json
    with open(out_json, "w") as f:
        json.dump(rec, f, indent=2)

    # Legacy-shape RESULT_JSON stdout line for older drivers that grep for it.
    legacy = {"tag": tag, "ppl": ppl, "ctx": ctx, "chunks": n_chunks, "tokens": ntok,
              "device": resolved, "dtype": dtype, "model": load_dir}
    print("RESULT_JSON " + json.dumps(legacy), flush=True)
    print(f"[ppl] wrote {out_json}  harness_key8={rec['harness_key8']}", flush=True)

    if not no_ledger:
        lp = ledger_path or default_ledger_path()
        append_record(lp, rec)
        print(f"[ppl] ledger += {lp}", flush=True)
    return rec, out_json
