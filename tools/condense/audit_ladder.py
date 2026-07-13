#!/usr/bin/env python3.12
"""4/3/2/1-bit STRAND ladder audit — per-model frontier (quality + REAL effective bpw).

Sweeps RHT / AWQ(alpha) / residual(b1+b2) / AWQ*residual across bit budgets, measures:
  - REAL effective bpw  (parsed from the baker's "AGGREGATE effective bpw" = RHT+outlier+
    residual-pass overhead included — the honest number, not nominal)
  - output-space degradation = ppl(condensed)/ppl(f16) - 1   (real forward passes)
Emits per-config JSONL (for the overlaid curve) + a markdown table (best method per tier).

Memory-safe for the 7B in 19GB: the model is FREED during every bake/build (so the Rust baker
and the Python parent are never both holding ~14GB) and RELOADED per measurement, with overrides
STREAMED into the model in-place (peak ~= one model copy). Slow but fits. DOCTOR_DEVICE/DTYPE
honored (0.5B: mps/float32; 7B: cpu/bfloat16 — fp16 overflows the 7B CPU forward -> nan).

Usage: audit_ladder.py <hf-dir> <label> <full|essential> [out_prefix]
"""
import sys, os, re, gc, json, math, time, hashlib, subprocess, shutil, atexit, fcntl, glob, torch, torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
from safetensors.torch import save_file
from tripwire_gate import compare as compare_tripwire
from tripwire_gate import policy as tripwire_policy
from tripwire_gate import validate_baseline as validate_tripwire_baseline
from tripwire_gate import validate_result as validate_tripwire_result
from adapter_contract import AdapterContractError, validate_for_model as validate_adapter_for_model

MODEL = sys.argv[1]; LABEL = sys.argv[2]
SETNAME = sys.argv[3] if len(sys.argv) > 3 else "essential"
OUTP = sys.argv[4] if len(sys.argv) > 4 else f"/tmp/audit_{LABEL}"
BAKER = "vendor/strand-quant/target/release/quantize-model"
DEV = os.environ.get("DOCTOR_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = getattr(torch, os.environ.get("DOCTOR_DTYPE", "float32"))
PT = os.environ.get("PPL_TEXT", "/tmp/ppl24k.txt")
HEAVY_LEASE_FD_ENV = "HAWKING_HEAVY_LEASE_FD"


def _lease_fds():
    try:
        fd = int(os.environ.get(HEAVY_LEASE_FD_ENV, ""))
        os.fstat(fd)
        return (fd,)
    except (TypeError, ValueError, OSError):
        return ()


def _resolve_src():
    """The baker needs ONE safetensors. 0.5B is single-file; 7B+ is sharded -> consolidate once.
    Streaming: peak ~one tensor, never loads the full 14GB dict into RAM simultaneously.  The cache
    is source-fingerprinted and atomically published; an interrupted or same-label/stale file is never
    trusted merely because it exists."""
    single = os.path.join(MODEL, "model.safetensors")
    if os.path.exists(single):
        return single
    fingerprint = _model_stat_fingerprint()
    cons = f"/tmp/aud_{LABEL}_src_{fingerprint[:16]}.safetensors"
    manifest_path = f"{cons}.manifest.json"
    try:
        manifest = json.load(open(manifest_path))
        stat = os.stat(cons)
        if not (
            manifest.get("schema") == "hawking.audit_consolidated_source.v1"
            and manifest.get("model_dir") == os.path.realpath(MODEL)
            and manifest.get("source_fingerprint") == fingerprint
            and manifest.get("bytes") == stat.st_size
            and manifest.get("mtime_ns") == stat.st_mtime_ns
            and isinstance(manifest.get("sha256"), str)
            and len(manifest["sha256"]) == 64
            and manifest.get("tensor_count", 0) > 0
        ):
            raise ValueError("manifest identity/stat mismatch")
        # SafeTensors validates that header offsets fit inside the file, catching a truncated cache.
        with safe_open(cons, framework="pt") as cached:
            if len(cached.keys()) != manifest["tensor_count"]:
                raise ValueError("cached tensor count mismatch")
        if os.environ.get("AUDIT_VERIFY_SOURCE_SHA", "1") != "0" \
                and _sha256_file(cons) != manifest["sha256"]:
            raise ValueError("cached consolidated source sha256 mismatch")
        return cons
    except Exception:
        pass

    idxp = os.path.join(MODEL, "model.safetensors.index.json")
    weight_map = json.load(open(idxp))["weight_map"]
    shards = sorted(set(weight_map.values()))
    shard_fhs = {sh: safe_open(os.path.join(MODEL, sh), framework="pt") for sh in shards}
    all_keys = []
    for sh in shards:
        all_keys.extend(shard_fhs[sh].keys())
    tmp = f"{cons}.tmp.{os.getpid()}"
    try:
        def _spec(k):
            sl = shard_fhs[weight_map[k]].get_slice(k)
            return (_str_dtype(sl.get_dtype()), tuple(sl.get_shape()))

        def _produce(k):
            return shard_fhs[weight_map[k]].get_tensor(k)

        stream_save(tmp, all_keys, _spec, _produce)
        with open(tmp, "rb") as staged:
            os.fsync(staged.fileno())
        with safe_open(tmp, framework="pt") as staged:
            if len(staged.keys()) != len(all_keys):
                raise RuntimeError("consolidated source tensor count mismatch")
        sha = _sha256_file(tmp)
        os.replace(tmp, cons)
        try:
            dfd = os.open(os.path.dirname(cons) or ".", os.O_RDONLY)
            try: os.fsync(dfd)
            finally: os.close(dfd)
        except OSError:
            pass
        stat = os.stat(cons)
        _atomic_json_durable(manifest_path, {
            "schema": "hawking.audit_consolidated_source.v1",
            "created_unix": time.time(),
            "model_dir": os.path.realpath(MODEL),
            "source_fingerprint": fingerprint,
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha,
            "tensor_count": len(all_keys),
        })
    finally:
        for handle in shard_fhs.values():
            close = getattr(handle, "close", None)
            if close:
                close()
        try: os.remove(tmp)
        except OSError: pass
    gc.collect()
    return cons


def log(m): print(m, file=sys.stderr); sys.stderr.flush()


# ---- streaming safetensors writer (peak ~one tensor, not the whole ~15GB dict) ----
_ST_DTYPE = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32",
             torch.float64: "F64", torch.int64: "I64", torch.int32: "I32",
             torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL"}
_TORCH_DTYPE = {v: k for k, v in _ST_DTYPE.items()}


def _str_dtype(s):
    """safetensors dtype string (e.g. 'BF16') -> torch dtype."""
    return _TORCH_DTYPE[s]


def _raw_bytes(t):
    """Raw little-endian bytes of a tensor in safetensors layout (works for bf16, which has no numpy dtype)."""
    return t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def stream_save(out_path, names, spec, produce):
    """Write a safetensors file one tensor at a time.

    names   : ordered list of tensor names (write order = data-blob order).
    spec(k) : -> (torch_dtype, shape_tuple) for the FINAL tensor (cheap, no materialization).
    produce(k): -> the final tensor for key k (built on demand, freed right after write).

    Transforms here preserve dtype+shape, so spec() reads them off the lazy source slices;
    only produce() materializes a tensor, and only one at a time. Peak ~= one tensor.
    """
    header, off = {}, 0
    for k in names:
        dt, shape = spec(k)
        nbytes = 1
        for d in shape:
            nbytes *= d
        nbytes *= torch.empty(0, dtype=dt).element_size()
        header[k] = {"dtype": _ST_DTYPE[dt], "shape": list(shape), "data_offsets": [off, off + nbytes]}
        off += nbytes
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-(len(hb) + 8)) % 8                      # 8-byte align the data blob (pad header w/ spaces)
    hb += b" " * pad
    with open(out_path, "wb") as fo:
        fo.write(len(hb).to_bytes(8, "little"))
        fo.write(hb)
        for k in names:
            t = produce(k)
            b = _raw_bytes(t)
            exp = header[k]["data_offsets"][1] - header[k]["data_offsets"][0]
            if len(b) != exp:
                raise RuntimeError(f"stream_save size mismatch {k}: wrote {len(b)} expected {exp}")
            fo.write(b)
            del t, b


SRC = None
_TEMP_ID = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{LABEL}_{SETNAME}")
# Lane-qualified scratch prevents a diagnostic/manual launch from deleting another lane's files.
# A model-wide flock below still serializes lanes because their consolidated source and RAM budgets
# are shared; the separate namespace is defense in depth for legacy processes that predate that lock.
T = f"/tmp/aud_{_TEMP_ID}"                  # reused temp prefix (overwritten per config)
SIGPATH = os.environ.get("LADDER_SIGMA", f"{OUTP}_sigma.safetensors")
LAST_BUILD_META = None
LAST_QUANTIZED_WEIGHTS = 0
LAST_ORACLE_META = None
AUDIT_RECIPE_VERSION = "hawking.audit.recipe.2026-07-12.v2"


def _free_gb(path="/tmp"):
    return shutil.disk_usage(path).free / 1e9


def _append_jsonl_durable(path, row):
    """Commit one completed-config checkpoint before deleting its temporary artifacts."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_durable(path, value):
    """Atomic+fsynced JSON for proof inputs that must survive an unplug."""
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    try:
        dfd = os.open(parent, os.O_RDONLY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    except OSError:
        pass


def _model_stat_fingerprint():
    """Fast identity for baseline reuse: metadata bytes plus every weight shard's stat tuple."""
    manifest = {"model_dir": os.path.realpath(MODEL), "files": []}
    try:
        names = sorted(os.listdir(MODEL))
    except OSError:
        names = []
    tokenizer_names = {
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model",
        "spiece.model", "sentencepiece.bpe.model",
    }
    for name in names:
        if not (name in {"config.json", "generation_config.json", *tokenizer_names}
                or name.endswith(".safetensors") or name.endswith(".safetensors.index.json")):
            continue
        path = os.path.join(MODEL, name)
        try:
            stat = os.stat(path)
        except OSError:
            continue
        item = {"name": name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if name.endswith(".json") or name in tokenizer_names:
            item["sha256"] = _sha256_file(path)
        manifest["files"].append(item)
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _audit_identity():
    """Identity for every resumable row, including parent, evaluator, and codec recipe."""
    evidence_files = {}
    for path in (
        BAKER, __file__, "tools/condense/doctor.py", "tools/condense/multi_eval.py",
        "tools/condense/adapter_contract.py",
        "tools/condense/tripwire_gate.py", "scratch/calib_corpus.txt",
    ):
        if os.path.isfile(path):
            evidence_files[path] = _sha256_file(path)
    actmean = os.environ.get("BAKE_ACTMEAN")
    return {
        "schema": "hawking.audit_identity.v1",
        "recipe_version": AUDIT_RECIPE_VERSION,
        "model": LABEL,
        "model_dir": os.path.realpath(MODEL),
        "model_fingerprint": _model_stat_fingerprint(),
        "lane": SETNAME,
        "eval_text_path": os.path.realpath(PT),
        "eval_text_sha256": _sha256_file(PT),
        "device": DEV,
        "dtype": str(DTYPE),
        "multiwindow": int(os.environ.get("MULTIWINDOW", "1")),
        "studio_tripwire": os.environ.get("STUDIO_TRIPWIRE") == "1",
        "bake_quality": os.environ.get("BAKE_QUALITY") == "1",
        "bake_actmean_path": os.path.realpath(actmean) if actmean and os.path.isfile(actmean) else None,
        "bake_actmean_sha256": _sha256_file(actmean) if actmean and os.path.isfile(actmean) else None,
        "strand_f32_metric": os.environ.get("STRAND_F32_METRIC"),
        "strand_f32_search": os.environ.get("STRAND_F32_SEARCH"),
        "doctor_grad_accum": 4,
        "doctor_kd_topk": 64,
        "doctor_target_regex": None,
        "evidence_files": evidence_files,
    }


def _prepare_audit_identity():
    """Quarantine all resumable evidence when its full recipe/source identity changes."""
    _ensure_ppl_text()
    path = f"{OUTP}.identity.json"
    expected = _audit_identity()
    try:
        existing = json.load(open(path))
    except Exception:
        existing = None
    if existing != expected:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        suffix = f".stale.{stamp}.{hashlib.sha256(json.dumps(existing, sort_keys=True, default=str).encode()).hexdigest()[:8]}"
        stale_paths = (
            f"{OUTP}.jsonl", f"{OUTP}.md", SIGPATH,
            f"{OUTP}_tripwire_baseline.json",
        )
        for old in stale_paths:
            if os.path.exists(old):
                destination = f"{old}{suffix}"
                os.replace(old, destination)
                log(f"# identity changed: quarantined {old} -> {destination}")
        _atomic_json_durable(path, expected)
    return expected, path


def _tripwire_baseline_path():
    # The task suite is lane-independent; reuse one expensive f16 pass across studio/subbit/verify.
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", LABEL)
    return os.environ.get("TRIPWIRE_BASELINE_PATH",
                          f"reports/cron/f16_tripwire_{safe_label}.json")


def _tripwire_suite_identity():
    policy = tripwire_policy()
    return {
        "schema": "hawking.tripwire_suite_identity.v1",
        "multi_eval_sha256": _sha256_file("tools/condense/multi_eval.py"),
        "tripwire_gate_sha256": _sha256_file("tools/condense/tripwire_gate.py"),
        "policy_sha256": hashlib.sha256(
            json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "suite": "hawking.multi_eval.v1",
    }


def _tripwire_baseline_valid(receipt, fingerprint):
    result = receipt.get("result", {}) if isinstance(receipt, dict) else {}
    return bool(
        isinstance(receipt, dict)
        and receipt.get("schema") == "hawking.tripwire_baseline.v1"
        and receipt.get("status") == "pass"
        and receipt.get("model") == LABEL
        and receipt.get("model_dir") == os.path.realpath(MODEL)
        and receipt.get("model_fingerprint") == fingerprint
        and receipt.get("suite_identity") == _tripwire_suite_identity()
        and result.get("model") == MODEL
        and validate_tripwire_baseline(result, f"{LABEL}-f16")["ok"]
    )


def _ensure_tripwire_baseline():
    """Capture/reuse the parent f16 task baseline that every floor candidate must match."""
    path = _tripwire_baseline_path()
    fingerprint = _model_stat_fingerprint()
    try:
        receipt = json.load(open(path))
    except Exception:
        receipt = None
    if _tripwire_baseline_valid(receipt, fingerprint):
        log(f"# tripwire f16 baseline = {receipt['result']['aggregate']:.4f} (checkpoint)")
        return receipt
    timeout = int(os.environ.get("TRIPWIRE_TIMEOUT", "14400"))
    label = f"{LABEL}-f16"
    cmd = ["python3.12", "tools/condense/multi_eval.py", MODEL, "-", label]
    log(f"# tripwire: capturing f16 task baseline ({timeout}s timeout)")
    run = subprocess.run(
        cmd, capture_output=True, text=True, env=os.environ, timeout=timeout,
        pass_fds=_lease_fds(),
    )
    last = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else ""
    try:
        result = json.loads(last)
    except Exception as exc:
        raise RuntimeError(f"f16 tripwire baseline emitted invalid JSON: {exc}; rc={run.returncode}")
    check = validate_tripwire_baseline(result, label)
    if isinstance(result, dict) and result.get("model") != MODEL:
        check["ok"] = False
        check["problems"].append(f"baseline model={result.get('model')!r}, expected {MODEL!r}")
    if run.returncode != 0 or not check["ok"]:
        raise RuntimeError(
            f"f16 tripwire baseline failed rc={run.returncode}: {check['problems']} "
            f"stderr={run.stderr[-200:]}"
        )
    receipt = {
        "schema": "hawking.tripwire_baseline.v1",
        "status": "pass",
        "created_unix": time.time(),
        "model": LABEL,
        "model_dir": os.path.realpath(MODEL),
        "model_fingerprint": fingerprint,
        "suite_identity": _tripwire_suite_identity(),
        "command": cmd,
        "policy": tripwire_policy(),
        "result": result,
    }
    _atomic_json_durable(path, receipt)
    log(f"# tripwire f16 baseline = {result['aggregate']:.4f} -> {path}")
    return receipt


def _ensure_ppl_text():
    """Recreate the small eval/capture corpus if /tmp was wiped by a reboot."""
    if os.path.exists(PT):
        return
    chunks = []
    if os.path.exists("README.md"):
        chunks.append(open("README.md", errors="ignore").read())
    import glob
    for path in sorted(glob.glob("docs/plans/*.md")):
        chunks.append(open(path, errors="ignore").read())
    text = "\n".join(chunks)[:24000]
    if not text:
        raise FileNotFoundError(PT)
    with open(PT, "w") as f:
        f.write(text)


def _clean_temps(keep=None):
    """Remove every {T}_* intermediate (b1/b2/rin/scaled/baked/ovr), keep SRC + sigma.
    Used to recover from an interrupted config before retrying."""
    import glob
    if keep is None:
        keep = (SRC, SIGPATH)
    for p in glob.glob(f"{T}_*.safetensors"):
        if p not in keep:
            try: os.remove(p)
            except OSError: pass


def _ensure_disk(need_gb, label):
    """Pre-flight: guarantee `need_gb` free on /tmp before a config writes intermediates.
    If short, sweep orphaned temps and recheck; if STILL short, raise so main() logs the
    config as skipped and the ladder CONTINUES — never let a write hit ENOSPC mid-stream."""
    free = _free_gb()
    if free < need_gb:
        log(f"  [disk] {label}: {free:.1f}GB < {need_gb}GB need — sweeping orphan temps")
        _clean_temps()
        free = _free_gb()
    if free < need_gb:
        raise RuntimeError(f"insufficient disk: {free:.1f}GB free < {need_gb}GB needed (skipped, ladder continues)")


def _bake_one(inp, out, bits, rung=None, outlier_pct=1.0, *, vec_dim=None,
              learned_codebook=False, block_len=256, require_oracle_accounting=False):
    """ONE raw baker invocation. --quality removed (L=k+4 fast, accurate for curve shape);
    STRAND_NO_GPU forces CPU so the baker never fights the model measurement for Metal memory.
    rung = path to a flat JSON {substr: bits} for MIXED-PRECISION (per-layer bit allocation,
    e.g. 4-bit attention / 3-bit FFN); --bits is the fallback for tensors matching no rule.
    outlier_pct = % of top-|w| weights kept as an 8-bit sparse channel — the train-free rescue
    for sub-3-bit (at 1-bit most signal dies; keeping the top 5-10% at 8-bit can recover a lot)."""
    nt = int(os.environ.get("BAKE_THREADS") or 4)
    env = {**os.environ, "STRAND_NO_GPU": "1"}
    cmd = [BAKER, "--in", inp, "--out", out, "--bits", str(bits), "--rht-cols",
           "--outlier-channel", str(outlier_pct), "--outlier-bits", "8", "--threads", str(nt)]
    if rung:
        cmd += ["--rung-config", rung]
    # ---- frontier quality levers (env-gated; inert unless a recipe turns them on) ----
    # --actmean: activation-mean output de-bias (baker comment: -28.7% PPL for ~0.014 bpw).
    #   c_i = -Σ_j (recon_ij - orig_ij)·E[x_j]; the actmean json is written by capture_sigma.
    # Named VTQ oracle identities deliberately pin the fast L=k+4/no-actmean recipe. Mutable
    # environment tuning is retained for legacy/scalar research only; otherwise a resumed
    # `vtq-...` row could silently mean a different codec than its config ID.
    am = None if require_oracle_accounting else os.environ.get("BAKE_ACTMEAN")
    if am and os.path.exists(am):
        cmd += ["--actmean", am]
    # --quality: deeper trellis (L=bits+6 vs +4). Lower PPL, ~4× slower on CPU. Chunked baking
    #   now bounds RAM regardless of L, so the old 7B OOM that forced this off no longer applies.
    if not require_oracle_accounting and os.environ.get("BAKE_QUALITY") == "1":
        cmd += ["--quality"]
    # Explicit builder arguments are load-bearing for resume identity. Mutable vector environment
    # overrides once allowed a scalar-named row to execute a different codec, so named builders now
    # default unconditionally to scalar and VTQ passes d/codebook explicitly.
    vd = 1 if vec_dim is None else int(vec_dim)
    if vd > 1:
        cmd += ["--vec-dim", str(vd)]
        if learned_codebook:
            cmd += ["--learned-codebook"]
    cmd += ["--block-len", str(int(block_len))]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       pass_fds=_lease_fds())
    if r.returncode != 0:
        raise RuntimeError(f"baker failed bits={bits}: {r.stderr[-200:]}")
    # parse the baker's TRUE quantized-weight count too — embed/lm_head pass through unquantized
    # (bpw 0 over 0 weights), so we weight the per-chunk aggregate by the baker's count, not our
    # own _isq guess (which would wrongly include the 1B+ embedding params and dilute eff-bpw).
    m = re.search(r"AGGREGATE effective bpw = ([0-9.]+) over ([0-9]+) quantized", r.stderr + r.stdout)
    accounting = None
    sidecar = f"{out}.json"
    try:
        sidecar_doc = json.load(open(sidecar))
        aggregate = sidecar_doc.get("aggregate", {})
        if aggregate.get("billing_complete") is True:
            accounting = {
                key: aggregate.get(key) for key in (
                    "quantized_weights", "effective_bpw", "oracle_effective_bpw",
                    "payload_bits", "trellis_side_bits", "outlier_side_bits",
                    "required_lut_bytes", "vector_lut_required_tensors",
                    "vector_lut_required_weights", "learned_lut_selected_tensors",
                    "learned_lut_selected_weights", "billing_complete", "billing_scope",
                    "artifact_class", "deployable",
                )
            }
            accounting["encoder_config"] = sidecar_doc.get("config")
    except Exception:
        accounting = None
    if require_oracle_accounting and accounting is None:
        raise RuntimeError(
            "VTQ oracle needs the rebuilt quantize-model exact-accounting sidecar; "
            "build vendor/strand-quant/target/release/quantize-model"
        )
    if accounting is not None:
        bpw = accounting.get("oracle_effective_bpw")
        qcount = accounting.get("quantized_weights")
        if isinstance(bpw, (int, float)) and isinstance(qcount, int):
            return float(bpw), int(qcount), accounting
    if m:
        return float(m.group(1)), int(m.group(2)), None
    return float("nan"), 0, None


#   CHUNKED baking — the memory fix for 7B on a 19GB Mac.
#   The Rust baker (quantize-model.rs write_safetensors) accumulates the ENTIRE recon as F32 in
#   RAM before writing — ~28GB for a 7B (F32 = 2× the bf16 model) — and OOM-dies partway
#   (observed 2026-06-23: killed at tensor 70/196, swap pinned ~40GB; thread-count independent).
#   Per-tensor trellis quant is INDEPENDENT, so we feed the baker ONE chunk of tensors at a time:
#   it only ever accumulates one chunk's F32 recon (RAM bounded), and we never materialize a full
#   scaled/baked/rin/ovr file (disk bounded) — each builder streams its transform per chunk into a
#   small bf16 "part", and measure() applies the parts together. Bit-identical per tensor.
def _src_specs(path=None):
    with safe_open(path or SRC, framework="pt") as f:
        names = list(f.keys())
        specs = {k: (_str_dtype(f.get_slice(k).get_dtype()), tuple(f.get_slice(k).get_shape())) for k in names}
    return names, specs


def _isq(specs, k):
    """Approx of the baker's is_quantizable_linear: 2D, both dims >= 256."""
    sh = specs[k][1]
    return len(sh) == 2 and min(sh) >= 256


def _chunks(names, specs):
    """Partition the ordered tensor list into ~BAKE_CHUNKS equal-BYTE groups (whole tensors)."""
    n = int(os.environ.get("BAKE_CHUNKS") or 6)
    esz = {dt: torch.empty(0, dtype=dt).element_size() for dt in {d for d, _ in specs.values()}}
    def nb(k):
        dt, sh = specs[k]; m = 1
        for d in sh: m *= d
        return m * esz[dt]
    target = sum(nb(k) for k in names) / max(1, n)
    out, cur, cb = [], [], 0
    for k in names:
        cur.append(k); cb += nb(k)
        if cb >= target and len(out) < n - 1:
            out.append(cur); cur, cb = [], 0
    if cur: out.append(cur)
    return out


def _bake_chunk(cnames, specs, produce, bits, rung=None, outlier_pct=1.0, *, vec_dim=None,
                learned_codebook=False, block_len=256, require_oracle_accounting=False):
    """Bake ONE chunk: stream its (pre-transformed, bf16) inputs, run the baker, return the recon
    as a small in-RAM dict (this chunk only) + the chunk's aggregate bpw. Cleans its own temps.
    This is the unit that bounds memory — the baker accumulates only one chunk's F32 recon."""
    cin, cout = f"{T}_ckin.safetensors", f"{T}_ckout.safetensors"
    try:
        stream_save(cin, cnames, lambda k: (DTYPE, specs[k][1]), produce)
        bpw, qcount, accounting = _bake_one(
            cin, cout, bits, rung=rung, outlier_pct=outlier_pct, vec_dim=vec_dim,
            learned_codebook=learned_codebook, block_len=block_len,
            require_oracle_accounting=require_oracle_accounting,
        )
        recon = {}
        with safe_open(cout, framework="pt") as fc:
            for k in fc.keys():
                recon[k] = fc.get_tensor(k).to(DTYPE)
    finally:
        for p in (cin, cout, f"{cout}.json"):
            try: os.remove(p)
            except OSError: pass
    return recon, bpw, qcount, accounting


def _emit_part(ci, cnames, specs, produce):
    """Write one chunk's final (transformed) tensors as a bf16 override part; return its path."""
    part = f"{T}_p{ci}.safetensors"
    stream_save(part, cnames, lambda k: (DTYPE, specs[k][1]), produce)
    return part


def keys_2d(path):
    with safe_open(path, framework="pt") as f:
        return [k for k in f.keys() if len(f.get_slice(k).get_shape()) == 2]


def quant_keys(base_decoded):
    """Tensors the baker actually quantized = 2D and changed vs SRC."""
    qk = set()
    with safe_open(SRC, framework="pt") as fs, safe_open(base_decoded, framework="pt") as fb:
        bk = set(fb.keys())
        for k in fs.keys():
            if k not in bk:
                continue
            v = fs.get_tensor(k)
            if v.dim() == 2 and not torch.equal(fb.get_tensor(k).to(v.dtype), v):
                qk.add(k)
    return qk


# ---- builders: each returns (list-of-override-parts, effective_bpw). Model NOT loaded here.
#      Every builder loops chunks so RAM stays ~one chunk and disk never holds a full temp. ----
def _wavg(pairs):
    w = sum(b * q for b, q in pairs if not math.isnan(b))
    q = sum(q for b, q in pairs if not math.isnan(b))
    return w / q if q else float("nan")


def build_rht(bits, outlier_pct=1.0):
    names, specs = _src_specs()
    chunks = _chunks(names, specs)
    log(f"  [bake] {bits}-RHT in {len(chunks)} chunks")
    parts, bpws = [], []
    with safe_open(SRC, framework="pt") as fs:
        for ci, cn in enumerate(chunks):
            _ensure_disk(16, f"{bits}-RHT ck{ci+1}/{len(chunks)}")
            recon, bpw, qc, _ = _bake_chunk(
                cn, specs, lambda k: fs.get_tensor(k).to(DTYPE), bits,
                outlier_pct=outlier_pct,
            )
            parts.append(_emit_part(ci, cn, specs, lambda k: recon[k]))
            bpws.append((bpw, qc))
            del recon; gc.collect()
            log(f"  [bake]  {bits}-RHT ck{ci+1}/{len(chunks)} bpw={bpw:.3f} q={qc} free={_free_gb():.0f}GB")
    return parts, _wavg(bpws)


def build_awq(bits, alpha=0.5, rung=None, outlier_pct=1.0, *, vec_dim=None,
              learned_codebook=False, block_len=256, oracle=False):
    """AWQ (scale cols by sigma^alpha, bake, unscale). rung = mixed-precision dict {substr: bits}
    written to a temp json and passed to the baker; `bits` is then the per-tensor FALLBACK.
    outlier_pct = top-|w| 8-bit sparse channel size (sub-3-bit train-free rescue)."""
    global LAST_QUANTIZED_WEIGHTS, LAST_ORACLE_META
    LAST_QUANTIZED_WEIGHTS = 0
    LAST_ORACLE_META = None
    sig = {}
    if alpha != 0.0:
        with safe_open(SIGPATH, framework="pt") as f:
            for k in f.keys():
                sig[k] = f.get_tensor(k)
    rpath = None
    if rung:
        rpath = f"{T}_rung.json"
        json.dump(rung, open(rpath, "w"))
    label = ("mp" if rung else f"{bits}") + f"-AWQ.{int(alpha*100)}"
    if vec_dim is not None and int(vec_dim) > 1:
        label += f"-d{int(vec_dim)}-b{int(block_len)}-{'learned' if learned_codebook else 'frozen'}"
    names, specs = _src_specs()
    chunks = _chunks(names, specs)
    log(f"  [bake] {label} in {len(chunks)} chunks{' (mixed-prec)' if rung else ''}")
    parts, bpws, accounting_rows = [], [], []
    with safe_open(SRC, framework="pt") as fs:
        for ci, cn in enumerate(chunks):
            _ensure_disk(16, f"{label} ck{ci+1}/{len(chunks)}")
            def scaled(k):                                   # scale cols by sigma^alpha pre-bake
                v = fs.get_tensor(k)
                return (v.float() * sig[k].pow(alpha)).to(DTYPE) \
                    if alpha != 0.0 and k in sig else v.to(DTYPE)
            recon, bpw, qc, accounting = _bake_chunk(
                cn, specs, scaled, bits, rung=rpath, outlier_pct=outlier_pct,
                vec_dim=vec_dim, learned_codebook=learned_codebook, block_len=block_len,
                require_oracle_accounting=oracle,
            )
            def unscaled(k):                                 # undo the scale post-bake
                return (recon[k].float() / sig[k].pow(alpha)).to(DTYPE) \
                    if alpha != 0.0 and k in sig else recon[k]
            parts.append(_emit_part(ci, cn, specs, unscaled))
            bpws.append((bpw, qc))
            if accounting is not None:
                accounting_rows.append(accounting)
            del recon; gc.collect()
            log(f"  [bake]  {label} ck{ci+1}/{len(chunks)} bpw={bpw:.3f} q={qc} free={_free_gb():.0f}GB")
    LAST_QUANTIZED_WEIGHTS = sum(qcount for _bpw, qcount in bpws)
    exact_bpw = None
    if oracle:
        if len(accounting_rows) != len(chunks):
            raise RuntimeError("VTQ oracle accounting missing one or more chunk sidecars")
        expected_encoder = {
            "bits": bits, "l": bits + 4, "k": bits, "rht": True, "rht_axis": "cols",
            "vec_dim": int(vec_dim or 1), "block_len": int(block_len),
            "learned_codebook": bool(learned_codebook),
            "learned_codebook_iters": 50,
            "learned_codebook_max_vectors": 16384,
        }
        encoder_workers = set()
        for row in accounting_rows:
            encoded = row.get("encoder_config")
            if not isinstance(encoded, dict) \
                    or any(encoded.get(key) != value for key, value in expected_encoder.items()):
                raise RuntimeError(f"VTQ encoder sidecar recipe mismatch: {encoded!r}")
            workers = encoded.get("encode_workers")
            if not isinstance(workers, int) or workers <= 0:
                raise RuntimeError(f"VTQ encoder worker evidence invalid: {workers!r}")
            encoder_workers.add(workers)
        if len(encoder_workers) != 1 or (learned_codebook and encoder_workers != {1}):
            raise RuntimeError(f"VTQ encoder worker recipe mismatch: {sorted(encoder_workers)}")
        sums = {
            key: sum(int(row.get(key) or 0) for row in accounting_rows)
            for key in ("quantized_weights", "payload_bits", "trellis_side_bits",
                        "outlier_side_bits", "required_lut_bytes",
                        "vector_lut_required_tensors", "vector_lut_required_weights",
                        "learned_lut_selected_tensors", "learned_lut_selected_weights")
        }
        billing_scopes = {row.get("billing_scope") for row in accounting_rows}
        expected_scope = "logical_codec_stream_plus_required_lut_not_physical_packed_artifact"
        if billing_scopes != {expected_scope}:
            raise RuntimeError(f"VTQ oracle accounting scope mismatch: {sorted(map(str, billing_scopes))}")
        if learned_codebook and sums["learned_lut_selected_tensors"] == 0:
            raise RuntimeError(
                "learned VTQ recipe selected no learned tensor LUTs; every tensor fell back "
                "to the frozen broadcast LUT, so this row cannot be labeled learned"
            )
        record_bytes = 52 + (1 << (bits + 4)) * int(vec_dim or 1) * 4
        if not (
            sums["vector_lut_required_tensors"] > 0
            and sums["vector_lut_required_weights"] == sums["quantized_weights"]
            and sums["required_lut_bytes"]
                == sums["vector_lut_required_tensors"] * record_bytes
            and 0 <= sums["learned_lut_selected_tensors"]
                <= sums["vector_lut_required_tensors"]
            and 0 <= sums["learned_lut_selected_weights"]
                <= sums["vector_lut_required_weights"]
        ):
            raise RuntimeError("VTQ oracle per-vector SDSC LUT accounting mismatch")
        total_bits = (sums["payload_bits"] + sums["trellis_side_bits"]
                      + sums["outlier_side_bits"] + sums["required_lut_bytes"] * 8)
        exact_bpw = total_bits / sums["quantized_weights"] if sums["quantized_weights"] else float("nan")
        learned_weight_fraction = (sums["learned_lut_selected_weights"]
                                   / sums["quantized_weights"]
                                   if sums["quantized_weights"] else 0.0)
        recipe = {
            "bits": bits, "l_bits": bits + 4,
            "vec_dim": int(vec_dim or 1), "block_len": int(block_len),
            "learned_codebook": bool(learned_codebook), "outlier_pct": float(outlier_pct),
            "awq_alpha": float(alpha), "rht": "cols",
            "trellis_quality": False, "actmean": False,
            "learned_codebook_iters": 50,
            "learned_codebook_max_vectors": 16384,
            "encode_workers": next(iter(encoder_workers)),
            "input_dtype": str(DTYPE),
            "source_fingerprint": _model_stat_fingerprint(),
            "quantizer_sha256": _sha256_file(BAKER),
        }
        recipe_sha256 = hashlib.sha256(
            json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        LAST_ORACLE_META = {
            "schema": "hawking.vtq_reconstruction_oracle.v1",
            "artifact_class": "reconstruction_oracle",
            "deployable": False,
            "packed_artifact": None,
            "recipe": recipe,
            "recipe_sha256": recipe_sha256,
            "accounting": {
                **sums,
                "logical_stream_bits_including_required_lut": total_bits,
                "oracle_effective_bpw": exact_bpw,
                "learned_lut_selected_weight_fraction": learned_weight_fraction,
                "billing_scope": expected_scope,
                "method": "exact encoder payload/trellis-side/OUTL bits + required per-tensor Q12 LUT bytes",
            },
            "limitations": [
                "dense reconstruction override, not a packed VTQ artifact",
                "learned is guarded per tensor; fallback tensors use the frozen broadcast LUT",
                "every vector tensor's required SDSC LUT record is billed; this baker path does not yet invoke the packed-v2 append hook",
                "billing covers the logical codec stream, not physical framing/alignment/container bytes",
                "Hawking vector runtime/GPU residency is not established",
                "cannot support a model-fit, source-deletion, or deployment claim",
            ],
        }
    return parts, exact_bpw if oracle else _wavg(bpws)


def build_vtq(bits, vec_dim, block_len=256, learned_codebook=False):
    """Explicit vector-trellis reconstruction oracle.

    The config name supplies every geometry lever and this wrapper passes them as arguments, so a
    resumed row cannot silently inherit BAKE_VECDIM/BAKE_LEARNED_CB from its environment. Outliers
    are deliberately disabled: at sub-bit density their bookkeeping can consume the entire budget.
    """
    return build_awq(
        bits, 0.0, None, 0.0, vec_dim=vec_dim, learned_codebook=learned_codebook,
        block_len=block_len, oracle=True,
    )


def build_residual(b1, b2):
    """W ≈ STRAND(W) + STRAND(W − STRAND(W)). Two bakes per chunk, fully streamed — never holds a
    full b1/rin/b2/ovr file (the old 52GB-peak that forced residual to skip on this disk)."""
    names, specs = _src_specs()
    chunks = _chunks(names, specs)
    log(f"  [bake] res{b1}+{b2} in {len(chunks)} chunks (2 bakes/chunk)")
    parts, bp1, bp2 = [], [], []
    with safe_open(SRC, framework="pt") as fs:
        for ci, cn in enumerate(chunks):
            _ensure_disk(16, f"res{b1}+{b2} ck{ci+1}/{len(chunks)}")
            r1, w1, qc1, _ = _bake_chunk(cn, specs, lambda k: fs.get_tensor(k).to(DTYPE), b1)
            def rin(k):                                      # residual = SRC − b1(SRC) on quant tensors
                v = fs.get_tensor(k)
                return (v.float() - r1[k].float()).to(DTYPE) if _isq(specs, k) else v.to(DTYPE)
            r2, w2, qc2, _ = _bake_chunk(cn, specs, rin, b2)
            def recon(k):                                    # reconstruct = b1 + b2
                return (r1[k].float() + r2[k].float()).to(DTYPE) if _isq(specs, k) else r1[k]
            parts.append(_emit_part(ci, cn, specs, recon))
            bp1.append((w1, qc1)); bp2.append((w2, qc2))
            del r1, r2; gc.collect()
            log(f"  [bake]  res ck{ci+1}/{len(chunks)} bpw={w1:.2f}+{w2:.2f} q={qc1} free={_free_gb():.0f}GB")
    return parts, _wavg(bp1) + _wavg(bp2)


def _merge_parts(parts, out):
    """Stream chunk parts into one safetensors (doctor_lora.py needs a single base file)."""
    fhs = {p: safe_open(p, framework="pt") for p in parts}
    try:
        order = [k for p in parts for k in fhs[p].keys()]
        loc = {k: p for p in parts for k in fhs[p].keys()}
        shp = {k: tuple(fhs[loc[k]].get_slice(k).get_shape()) for k in order}
        stream_save(out, order, lambda k: (DTYPE, shp[k]), lambda k: fhs[loc[k]].get_tensor(k))
    finally:
        for fh in fhs.values():
            if hasattr(fh, "close"):
                fh.close()


def _swap_mb():
    """Current macOS swap used in MB."""
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                             capture_output=True, text=True).stdout
        return float(out.split()[5].rstrip("M,"))
    except Exception:
        return 0.0


def _doctor_progress(path):
    """Read the newest durable Doctor event without trusting a partially written tail."""
    latest = None
    try:
        with open(path, errors="ignore") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    latest = row
    except OSError:
        pass
    return latest


def _doctor_accounting_valid(row):
    """A completed Doctor row must bind its rank and exact serialized adapter cost."""
    doctor = row.get("doctor") if isinstance(row, dict) else None
    accounting = doctor.get("adapter_accounting") if isinstance(doctor, dict) else None
    final = doctor.get("final") if isinstance(doctor, dict) else None
    if not (isinstance(doctor, dict) and doctor.get("complete") is True
            and isinstance(final, dict) and final.get("stopped_early") is False
            and isinstance(accounting, dict)
            and accounting.get("schema") == "hawking.doctor_adapter_accounting.v1"):
        return False
    numeric_positive = ("rank", "adapter_bytes", "quantized_weights", "adapter_effective_bpw",
                        "base_effective_bpw", "total_effective_bpw")
    if any(not isinstance(accounting.get(key), (int, float)) or accounting[key] <= 0
           for key in numeric_positive):
        return False
    expected_adapter = accounting["adapter_bytes"] * 8.0 / accounting["quantized_weights"]
    return (
        abs(float(accounting["adapter_effective_bpw"]) - expected_adapter) <= 1e-9
        and abs(float(accounting["total_effective_bpw"])
                - float(accounting["base_effective_bpw"])
                - float(accounting["adapter_effective_bpw"])) <= 1e-9
        and isinstance(row.get("eff_bpw"), (int, float))
        and abs(float(row["eff_bpw"]) - float(accounting["total_effective_bpw"])) <= 0.0011
    )


def _finite_number(value):
    """JSON measurement fields must be real finite numbers (``bool`` is not evidence)."""
    return not isinstance(value, bool) and isinstance(value, (int, float)) \
        and math.isfinite(float(value))


def _measurement_row_valid(config, row, baseline_ppl=None):
    """Fail closed on malformed generic checkpoints before treating them as resumable.

    Previously any row containing a ``ppl`` key entered ``completed``.  A truncated/manual row with
    NaN, a wrong model label, or missing density could therefore suppress the corresponding config
    forever.  Recipe-specific Doctor/VTQ checks remain stricter and run after this shape check.
    """
    if not isinstance(row, dict) or row.get("model") != LABEL or row.get("config") != config \
            or "error" in row:
        return False
    if not _finite_number(row.get("ppl")) or float(row["ppl"]) <= 0 \
            or not _finite_number(row.get("eff_bpw")) or float(row["eff_bpw"]) <= 0 \
            or not _finite_number(row.get("degr_pct")):
        return False
    if config == "f16":
        return abs(float(row["eff_bpw"]) - 16.0) <= 1e-9 \
            and abs(float(row["degr_pct"])) <= 1e-9
    if baseline_ppl is not None:
        if not _finite_number(baseline_ppl) or float(baseline_ppl) <= 0:
            return False
        expected = (float(row["ppl"]) / float(baseline_ppl) - 1.0) * 100.0
        # Both persisted PPLs are rounded to 0.001 and degradation to 0.01.  Propagate that
        # uncertainty instead of using a fixed tolerance: catastrophic low-bit PPL can be orders of
        # magnitude above f16, magnifying the baseline's final decimal without making the row invalid.
        base = float(baseline_ppl)
        ppl = float(row["ppl"])
        rounding_bound = 0.006 + 100.0 * (0.00051 / base + ppl * 0.00051 / (base * base))
        if abs(float(row["degr_pct"]) - expected) > rounding_bound:
            return False
    return True


def _vtq_row_valid_for_resume(config, row):
    identity = re.fullmatch(
        r"vtq-k(?P<bits>\d+)-d(?P<vec_dim>\d+)-b(?P<block_len>\d+)-"
        r"(?P<codebook>frozen|learned)(?:\+dr-r(?P<rank>\d+))?",
        config,
    )
    if identity is None or row.get("deployable") is not False \
            or row.get("artifact_class") != "reconstruction_oracle":
        return False
    oracle = row.get("oracle")
    if not isinstance(oracle, dict) or oracle.get("schema") != "hawking.vtq_reconstruction_oracle.v1" \
            or oracle.get("deployable") is not False or oracle.get("packed_artifact") is not None:
        return False
    recipe, accounting = oracle.get("recipe"), oracle.get("accounting")
    if not isinstance(recipe, dict) or not isinstance(accounting, dict):
        return False
    expected_recipe = {
        "bits": int(identity.group("bits")), "l_bits": int(identity.group("bits")) + 4,
        "vec_dim": int(identity.group("vec_dim")),
        "block_len": int(identity.group("block_len")),
        "learned_codebook": identity.group("codebook") == "learned",
        "outlier_pct": 0.0, "awq_alpha": 0.0, "rht": "cols",
        "trellis_quality": False, "actmean": False,
        "learned_codebook_iters": 50, "learned_codebook_max_vectors": 16384,
        "input_dtype": str(DTYPE),
        "source_fingerprint": _model_stat_fingerprint(),
        "quantizer_sha256": _sha256_file(BAKER),
    }
    if any(recipe.get(key) != value for key, value in expected_recipe.items()):
        return False
    workers = recipe.get("encode_workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0 \
            or (expected_recipe["learned_codebook"] and workers != 1):
        return False
    recipe_sha = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if oracle.get("recipe_sha256") != recipe_sha:
        return False
    try:
        qcount = int(accounting["quantized_weights"])
        payload = int(accounting["payload_bits"])
        side = int(accounting["trellis_side_bits"])
        outlier = int(accounting["outlier_side_bits"])
        lut_bytes = int(accounting["required_lut_bytes"])
        vector_tensors = int(accounting["vector_lut_required_tensors"])
        vector_weights = int(accounting["vector_lut_required_weights"])
        selected_tensors = int(accounting["learned_lut_selected_tensors"])
        selected_weights = int(accounting["learned_lut_selected_weights"])
        total = payload + side + outlier + lut_bytes * 8
    except (KeyError, TypeError, ValueError):
        return False
    if qcount <= 0 or min(payload, side, outlier, lut_bytes, vector_tensors, vector_weights,
                          selected_tensors, selected_weights) < 0 \
            or accounting.get("logical_stream_bits_including_required_lut") != total \
            or accounting.get("billing_scope") != "logical_codec_stream_plus_required_lut_not_physical_packed_artifact" \
            or accounting.get("method") != "exact encoder payload/trellis-side/OUTL bits + required per-tensor Q12 LUT bytes" \
            or abs(float(accounting.get("oracle_effective_bpw", float("nan"))) - total / qcount) > 1e-9:
        return False
    fraction = accounting.get("learned_lut_selected_weight_fraction")
    if not isinstance(fraction, (int, float)) or not math.isfinite(float(fraction)) \
            or abs(float(fraction) - selected_weights / qcount) > 1e-9:
        return False
    learned = expected_recipe["learned_codebook"]
    expected_lut = vector_tensors * (
        52 + (1 << expected_recipe["l_bits"]) * expected_recipe["vec_dim"] * 4
    )
    if not (
        vector_tensors > 0 and vector_weights == qcount and lut_bytes == expected_lut
        and selected_tensors <= vector_tensors and selected_weights <= vector_weights
    ):
        return False
    if learned:
        if not (selected_tensors > 0 and 0 < selected_weights <= qcount):
            return False
    elif any((selected_tensors, selected_weights)):
        return False
    rank = identity.group("rank")
    if rank is not None:
        if not _doctor_accounting_valid(row):
            return False
        da = row["doctor"]["adapter_accounting"]
        plus = accounting.get("oracle_plus_adapter_effective_bpw")
        if da.get("rank") != int(rank) or da.get("quantized_weights") != qcount \
                or not isinstance(plus, (int, float)) \
                or abs(float(da["base_effective_bpw"])
                       - float(accounting["oracle_effective_bpw"])) > 1e-9 \
                or abs(float(da["total_effective_bpw"]) - float(plus)) > 1e-9:
            return False
    expected_bpw = accounting.get("oracle_plus_adapter_effective_bpw") \
        if rank is not None else accounting.get("oracle_effective_bpw")
    return (
        isinstance(expected_bpw, (int, float)) and math.isfinite(float(expected_bpw))
        and isinstance(row.get("eff_bpw"), (int, float))
        and abs(float(row["eff_bpw"]) - float(expected_bpw)) <= 0.0011
        and isinstance(row.get("ppl"), (int, float)) and math.isfinite(float(row["ppl"]))
    )


def build_recover(bits, steps=60, rank=64, lr=1e-4, alpha=0.5, rung=None,
                  outlier_pct=1.0, target_regex=None, kd_topk=None, *, vec_dim=None,
                  learned_codebook=False, block_len=256, oracle=False):
    """RECOVERY track: AWQ-quantize, then HEAL with LoRA-KD (doctor_lora.py) — the only path to
    usable sub-3-bit, where PTQ alone collapses. The doctor caches+frees the teacher (one model
    in RAM, 19GB-safe) and saves LoRA adapters, so measurement applies base+adapter in-place
    without ever materializing a fused full-weight checkpoint.
    Returns ({"base": base, "adapter": adapter}, base_bpw + LoRA overhead).
    Slow on CPU — wall-clock is not the constraint.

    Self-managing resource guard: monitors doctor subprocess every 60s.
      DOCTOR_TIMEOUT          max wall-clock seconds before checkpointed stop
      DOCTOR_SWAP_CEIL        soft swap telemetry threshold
      DOCTOR_SWAP_HARD_CEIL   hard swap threshold before checkpointed stop
    On breach: preserve the best/latest adapter if one exists and ladder CONTINUES."""
    import time
    global LAST_BUILD_META, LAST_ORACLE_META
    LAST_BUILD_META = None
    # timeout: long leash by default; progress is adapter-checkpointed, so time is useful.
    TIMEOUT   = int(os.environ.get("DOCTOR_TIMEOUT", str(max(28800, steps * 480))))
    # soft swap is telemetry only; hard swap asks the doctor to checkpoint before SIGKILL.
    SWAP_CEIL = float(os.environ.get("DOCTOR_SWAP_CEIL", "12000"))
    HARD_SWAP = float(os.environ.get("DOCTOR_SWAP_HARD_CEIL", "18000"))
    GRACE     = int(os.environ.get("DOCTOR_TERMINATE_GRACE", "600"))
    # A checkpoint is useful recovery material, but it is not a completed experiment. Studio is
    # fail-closed by default; exploratory callers can explicitly opt into measuring partial work.
    USE_PARTIAL = os.environ.get("DOCTOR_USE_PARTIAL", "0").lower() in {"1", "true", "yes"}
    # Loading is exactly when a wrong model/dtype can exhaust unified memory; no blind warmup.
    WARMUP    = int(os.environ.get("DOCTOR_RESOURCE_WARMUP", "0"))

    parts, base_bpw = build_awq(
        bits, alpha, rung, outlier_pct, vec_dim=vec_dim,
        learned_codebook=learned_codebook, block_len=block_len, oracle=oracle,
    )
    base = f"{T}_rbase.safetensors"
    _merge_parts(parts, base)
    for p in parts:
        try: os.remove(p)
        except OSError: pass
    adapter = f"{T}_adapter.safetensors"
    latest = f"{T}_adapter.latest.safetensors"
    progress = f"{T}_doctor_progress.jsonl"
    dout = f"{T}_doctor_stdout.log"
    derr = f"{T}_doctor_stderr.log"
    # These are append-oriented inside Doctor. Remove stale evidence from an earlier recipe so a
    # crashed new worker can never inherit an old `final` event and appear complete.
    for stale in (progress, dout, derr):
        try: os.remove(stale)
        except OSError: pass
    env = {**os.environ, "DOCTOR_MODEL": MODEL, "DOCTOR_DTYPE": "bfloat16", "DOCTOR_DEVICE": DEV,
           "DOCTOR_THREADS": str(os.cpu_count() or 8),
           "DOCTOR_GRAD_ACCUM": "4",
           "DOCTOR_SAVE_MODE": "adapter",
           "DOCTOR_PROGRESS": progress,
           "DOCTOR_LATEST": latest,
           "KD": "1", "KD_TOPK": str(kd_topk or 64)}
    env.pop("DOCTOR_TARGET_REGEX", None)
    if target_regex:
        env["DOCTOR_TARGET_REGEX"] = target_regex
    cal = "scratch/calib_corpus.txt"
    if os.path.exists(cal):
        env["DOCTOR_CALIB"] = cal
    log(f"  [recover] {bits}b base bpw={base_bpw:.3f}; LoRA-KD r{rank} {steps} steps"
        f" (timeout={TIMEOUT}s soft_swap={SWAP_CEIL:.0f}MB hard_swap={HARD_SWAP:.0f}MB)…")
    out_fh, err_fh = open(dout, "w"), open(derr, "w")
    proc = subprocess.Popen(["python3.12", "tools/condense/doctor.py", "lora", base,
                             str(steps), str(lr), str(rank), adapter],
                            stdout=out_fh, stderr=err_fh,
                            # Keep Doctor in the audit/queue process group.  The detached processing
                            # supervisor drains that group; a separate session could survive as an
                            # unobserved writer after the lane lease had been released.
                            text=True, env=env, pass_fds=_lease_fds())
    t0 = time.monotonic()
    terminating_at = None
    terminated_reason = None
    while proc.poll() is None:
        time.sleep(60)
        elapsed = time.monotonic() - t0
        swap = _swap_mb()
        heartbeat = _doctor_progress(progress)
        step_note = ""
        if heartbeat:
            step = heartbeat.get("step")
            steps_total = heartbeat.get("steps", steps)
            eta_s = heartbeat.get("eta_s")
            phase = heartbeat.get("phase")
            if step is not None:
                step_note = f" step={int(step)+1}/{int(steps_total)}"
                if isinstance(eta_s, (int, float)) and math.isfinite(float(eta_s)):
                    step_note += f" eta={float(eta_s)/60:.0f}m"
                if heartbeat.get("heldout_ppl") is not None:
                    step_note += f" heldout={float(heartbeat['heldout_ppl']):.2f}"
                if heartbeat.get("best_ppl") is not None:
                    step_note += f" best={float(heartbeat['best_ppl']):.2f}"
            elif phase:
                step_note = f" phase={phase}"
        log(f"  [recover] elapsed={elapsed/60:.0f}m swap={swap:.0f}MB pid={proc.pid}{step_note}")
        if elapsed > WARMUP and swap > SWAP_CEIL:
            log(f"  [recover] soft swap leash crossed ({swap:.0f}>{SWAP_CEIL:.0f}MB); continuing under watch")
        if terminating_at is None and elapsed > WARMUP and swap > HARD_SWAP:
            terminating_at = time.monotonic()
            terminated_reason = f"hard swap {swap:.0f}MB > {HARD_SWAP:.0f}MB"
            log(f"  [recover] {terminated_reason}; SIGTERM doctor for checkpoint")
            proc.terminate()
        if terminating_at is None and elapsed > TIMEOUT:
            terminating_at = time.monotonic()
            terminated_reason = f"timeout {elapsed/60:.0f}m > {TIMEOUT/60:.0f}m"
            log(f"  [recover] {terminated_reason}; SIGTERM doctor for checkpoint")
            proc.terminate()
        if terminating_at is not None and time.monotonic() - terminating_at > GRACE:
            log(f"  [recover] checkpoint grace expired after {GRACE}s; SIGKILL doctor")
            proc.kill()
    proc.wait()
    out_fh.close(); err_fh.close()
    stdout = open(dout, errors="ignore").read() if os.path.exists(dout) else ""
    stderr = open(derr, errors="ignore").read() if os.path.exists(derr) else ""
    r = subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
    artifact = adapter if os.path.exists(adapter) else latest if os.path.exists(latest) else None
    final_event = None
    if os.path.exists(progress):
        try:
            for line in open(progress, errors="ignore"):
                try:
                    candidate = json.loads(line)
                except Exception:
                    continue
                if isinstance(candidate, dict) and candidate.get("event") == "final":
                    final_event = candidate
        except OSError:
            pass
    incomplete = bool(terminated_reason or not final_event or final_event.get("stopped_early"))
    LAST_BUILD_META = {
        "schema": "hawking.doctor_run_evidence.v1",
        "complete": not incomplete and r.returncode == 0,
        "returncode": r.returncode,
        "termination_reason": terminated_reason,
        "checkpoint_present": bool(artifact),
        "artifact": artifact,
        "artifact_bytes": os.path.getsize(artifact) if artifact else None,
        "artifact_sha256": _sha256_file(artifact) if artifact else None,
        "final": final_event,
    }
    if (r.returncode != 0 or incomplete) and not (USE_PARTIAL and artifact):
        try: os.remove(base)
        except OSError: pass
        why = terminated_reason or (
            "missing durable final event" if not final_event else "doctor stopped early"
        )
        raise RuntimeError(f"doctor incomplete bits={bits}: {why}; rc={r.returncode}; "
                           f"checkpoint={'present' if artifact else 'missing'}")
    try:
        rec = json.loads(r.stdout.strip().splitlines()[-1])
        log(f"  [recover] base_ppl={rec.get('base_ppl'):.1f} -> lora_ppl={rec.get('lora_ppl'):.1f}"
            f" (recovered {rec.get('recovery_pct',0):.1f}%)")
    except Exception:
        pass
    if (r.returncode != 0 or incomplete) and artifact:
        log(f"  [recover] EXPLORATORY PARTIAL: using checkpointed adapter after "
            f"{terminated_reason or 'incomplete doctor run'}")
    if not artifact:
        try: os.remove(base)
        except OSError: pass
        raise RuntimeError(f"doctor produced no adapter checkpoint bits={bits}")
    # Honest deployed density: charge the exact serialized adapter bytes (including metadata/header)
    # against the baker's exact aggregate quantized-weight count. The old rank formula used an
    # approximate tensor predicate and materially underbilled small/sub-bit models.
    quantized_weights = int(LAST_QUANTIZED_WEIGHTS)
    if quantized_weights <= 0:
        raise RuntimeError("doctor adapter accounting missing exact baker quantized-weight count")
    adapter_bytes = os.path.getsize(artifact)
    lora_add = adapter_bytes * 8.0 / quantized_weights
    LAST_BUILD_META["adapter_accounting"] = {
        "schema": "hawking.doctor_adapter_accounting.v1",
        "rank": rank,
        "adapter_bytes": adapter_bytes,
        "quantized_weights": quantized_weights,
        "adapter_effective_bpw": lora_add,
        "base_effective_bpw": base_bpw,
        "total_effective_bpw": base_bpw + lora_add,
        "method": "serialized_adapter_bytes*8 / baker_exact_quantized_weights",
    }
    if oracle and LAST_ORACLE_META is not None:
        LAST_ORACLE_META["doctor_adapter"] = dict(LAST_BUILD_META["adapter_accounting"])
        LAST_ORACLE_META["accounting"]["oracle_plus_adapter_effective_bpw"] = base_bpw + lora_add
    return {"base": base, "adapter": artifact}, base_bpw + lora_add


def build_vtq_recover(bits, vec_dim, rank, block_len=256, learned_codebook=True, steps=60):
    """Rank-bounded Doctor over an explicitly identified VTQ reconstruction oracle."""
    return build_recover(
        bits, steps, rank, 1e-4, 0.0, None, 0.0, None, None,
        vec_dim=vec_dim, learned_codebook=learned_codebook,
        block_len=block_len, oracle=True,
    )


def _bake_shadow(candidates, bits, tag):
    """STRAND-bake a doctor's healed-shadow weights (L4/L5 produce full-precision quant-robust
    weights that must be baked through the real codec). Picks the first shadow file that exists,
    bakes it whole, returns ([baked_override], effective_bpw). Whole-bake peaks at ~2x bf16 RAM —
    fine on the 128GB Studio (this is a Studio-tier stage)."""
    raw = next((c for c in candidates if os.path.exists(c)), None)
    if not raw:
        raise RuntimeError(f"{tag}: doctor produced no shadow ({candidates})")
    baked = f"{T}_{tag}_baked.safetensors"
    bpw, qc, _ = _bake_one(raw, baked, bits)
    for c in candidates:
        try: os.remove(c)
        except OSError: pass
    return [baked], bpw


def _doctor_env():
    env = {**os.environ, "DOCTOR_MODEL": MODEL, "DOCTOR_DEVICE": DEV, "DOCTOR_DTYPE": "bfloat16",
           "STRAND_NO_GPU": "1"}
    cal = "scratch/calib_corpus.txt"
    if os.path.exists(cal):
        env["DOCTOR_CALIB"] = cal
    return env


def build_blockwise(bits, steps=80):
    """L4 — full-rank per-layer QAT (doctor_blockwise): the LoRA-plateau fix. Heals the full weight
    matrix (no rank ceiling, no bpw overhead) so it survives the STRAND cut, then bakes it."""
    out = f"{T}_bw.safetensors"
    log(f"  [L4 blockwise] {bits}b QAT {steps} steps")
    r = subprocess.run(["python3.12", "tools/condense/doctor.py", "blockwise", MODEL, out, str(bits), str(steps)],
                       capture_output=True, text=True, env=_doctor_env(),
                       pass_fds=_lease_fds())
    if r.returncode != 0:
        raise RuntimeError(f"blockwise failed bits={bits}: {r.stderr[-200:]}")
    return _bake_shadow([out, out.replace(".safetensors", ".raw.safetensors")], bits, "bw")


def build_strand(bits, steps=200, lr=3e-5, req=50):
    """L5 — codec-native GPTQ-Hessian error-feedback (doctor_strand): the sub-residual ceiling
    breaker; quantizes sequentially through STRAND's trellis with Hessian error feedback (no STE)."""
    save = f"{T}_str.safetensors"
    log(f"  [L5 strand] {bits}b GPTQ-Hessian {steps} steps lr {lr} requant/{req}")
    r = subprocess.run(["python3.12", "tools/condense/doctor.py", "strand", str(bits), str(steps),
                        str(lr), str(req), save], capture_output=True, text=True,
                       env=_doctor_env(), pass_fds=_lease_fds())
    if r.returncode != 0:
        raise RuntimeError(f"strand failed bits={bits}: {r.stderr[-200:]}")
    return _bake_shadow([save.replace(".safetensors", ".raw.safetensors"), save], bits, "str")


def _mp(a, f):
    """mixed-precision rung: a-bit attention, f-bit FFN."""
    return {k: a for k in ("q_proj", "k_proj", "v_proj", "o_proj")} | \
           {k: f for k in ("gate_proj", "up_proj", "down_proj")}


# True sub-bit research starts with vector trellis: k symbol bits reconstruct d weights. Every row
# below is a reconstruction oracle (`deployable=false`) until the baker integrates the exact-coverage
# SDSC LUT hook, emits a packed-v2 round-trip, and Hawking vector serving clears independently. Names
# are the resume/cache identity.
VTQ_CONFIGS = [
    # Mandatory frozen iso-payload controls.
    ("vtq-k1-d2-b256-frozen", build_vtq, (1, 2, 256, False)),
    ("vtq-k2-d4-b256-frozen", build_vtq, (2, 4, 256, False)),
    ("vtq-k1-d4-b256-frozen", build_vtq, (1, 4, 256, False)),
    ("vtq-k2-d8-b256-frozen", build_vtq, (2, 8, 256, False)),
    ("vtq-k1-d8-b256-frozen", build_vtq, (1, 8, 256, False)),
    # Mandatory learned-codebook density curve, including d3 (~one-third payload).
    ("vtq-k1-d2-b256-learned", build_vtq, (1, 2, 256, True)),
    ("vtq-k1-d3-b256-learned", build_vtq, (1, 3, 256, True)),
    ("vtq-k1-d4-b256-learned", build_vtq, (1, 4, 256, True)),
    ("vtq-k1-d8-b256-learned", build_vtq, (1, 8, 256, True)),
    # Mandatory rank-8 Doctor points. Exact serialized adapter bytes are charged to bpw.
    ("vtq-k1-d2-b256-learned+dr-r8", build_vtq_recover, (1, 2, 8, 256, True, 60)),
    ("vtq-k1-d3-b256-learned+dr-r8", build_vtq_recover, (1, 3, 8, 256, True, 60)),
    ("vtq-k1-d4-b256-learned+dr-r8", build_vtq_recover, (1, 4, 8, 256, True, 60)),
    ("vtq-k1-d8-b256-learned+dr-r8", build_vtq_recover, (1, 8, 8, 256, True, 60)),
    # Exploratory learned iso-payload controls (same payload, larger input alphabet/state space).
    ("vtq-k2-d4-b256-learned", build_vtq, (2, 4, 256, True)),
    ("vtq-k2-d8-b256-learned", build_vtq, (2, 8, 256, True)),
    # Exploratory side-info amortization campaign. >256 is CPU/reconstruction research geometry.
    ("vtq-k1-d2-b2048-frozen", build_vtq, (1, 2, 2048, False)),
    ("vtq-k1-d4-b4096-frozen", build_vtq, (1, 4, 4096, False)),
    ("vtq-k1-d8-b8192-frozen", build_vtq, (1, 8, 8192, False)),
]


# ---- measurement: load model, stream override in-place, ppl, free ----
def _apply_weight_overrides(model, paths):
    if not paths:
        return
    sd = model.state_dict()
    for pth in paths:
        with safe_open(pth, framework="pt") as f:
            for k in f.keys():
                if k in sd and tuple(sd[k].shape) == tuple(f.get_slice(k).get_shape()):
                    sd[k].copy_(f.get_tensor(k).to(DEV, DTYPE))


def _attach_lora_adapter(model, adapter_path, *, expected_wbase):
    """Fail closed on the complete v1 contract before mutating the PPL model."""
    if not expected_wbase:
        raise AdapterContractError("Doctor PPL measurement requires the adapter's base override")
    modules = dict(model.named_modules())
    staged = []
    with safe_open(adapter_path, framework="pt") as f:
        contract = validate_adapter_for_model(
            f, model, expected_model=MODEL, expected_wbase=expected_wbase,
            expected_target_regex="all",
        )
        compatibility = []
        for entry in contract["entries"]:
            module = modules.get(entry["name"])
            if not isinstance(module, nn.Linear):
                compatibility.append(f"{entry['name']}: no matching nn.Linear")
                continue
            expected_a = (module.weight.shape[0], contract["rank"])
            expected_b = (contract["rank"], module.weight.shape[1])
            if entry["a_shape"] != expected_a or entry["b_shape"] != expected_b:
                compatibility.append(
                    f"{entry['name']}: A{entry['a_shape']} B{entry['b_shape']}, expected "
                    f"A{expected_a} B{expected_b}"
                )
        if compatibility:
            raise AdapterContractError(
                "adapter/model shape mismatch: " + "; ".join(compatibility)
            )
        # Stage all factors after metadata/pairs/orientation/rank/dtype/module shapes pass. Nothing
        # is attached if any read or conversion fails.
        for entry in contract["entries"]:
            staged.append((
                modules[entry["name"]],
                f.get_tensor(entry["a_key"]).to(DEV, DTYPE),
                f.get_tensor(entry["b_key"]).to(DEV, DTYPE),
            ))
    for module, factor_a, factor_b in staged:
        module._hawking_lora_A = factor_a
        module._hawking_lora_B = factor_b
        module.forward = (lambda x, mm=module: F.linear(x, mm.weight, mm.bias)
                          + F.linear(F.linear(x, mm._hawking_lora_B),
                                     mm._hawking_lora_A))
    log(f"  [measure] attached {len(staged)} LoRA adapters "
        f"schema={contract['metadata']['adapter_schema']} "
        f"v{contract['metadata']['adapter_version']} orientation={contract['orientation']}")
    return contract


def measure(override):
    _ensure_ppl_text()
    torch.set_num_threads(os.cpu_count() or 12)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE, attn_implementation="eager").to(DEV).eval()
    if override:
        adapter = None
        if isinstance(override, dict):
            paths = [override["base"]] if override.get("base") else []
            adapter = override.get("adapter")
        else:
            paths = [override] if isinstance(override, str) else list(override)
        _apply_weight_overrides(model, paths)
        if adapter:
            _attach_lora_adapter(model, adapter, expected_wbase=paths[0] if paths else None)
    text = open(PT, errors="ignore").read()
    all_ids = tok(text, return_tensors="pt").input_ids[0].to(DEV)
    # MULTIWINDOW>1 (env, default 1) evaluates N non-overlapping 2048-token windows and returns
    # the MEAN ppl — a single noisy slice is overfittable (doctor trains on a calib corpus), so the
    # verifier scores on multiple held-out windows to confirm a recovery is real, not memorized.
    # The running ladder leaves this at 1 (consistent fast search); only the verifier sets it.
    nwin = int(os.environ.get("MULTIWINDOW", "1"))
    W = 2048
    ppls = []
    with torch.no_grad():
        for w in range(max(1, nwin)):
            seg = all_ids[w * W:(w + 1) * W]
            if seg.numel() < 16:
                break
            ids = seg.unsqueeze(0)
            ppls.append(math.exp(model(ids, labels=ids).loss.item()))
    del model; gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    if len(ppls) > 1:
        log(f"  [measure] {len(ppls)} windows: min={min(ppls):.2f} mean={sum(ppls)/len(ppls):.2f} max={max(ppls):.2f}")
    return sum(ppls) / len(ppls)


def capture_sigma():
    if os.path.exists(SIGPATH):
        log(f"# sigma already exists at {SIGPATH}, skipping capture")
        return
    _ensure_ppl_text()
    torch.set_num_threads(os.cpu_count() or 12)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE, attn_implementation="eager").to(DEV).eval()
    sig, hooks = {}, []
    def mk(n):
        def h(m, i, o):
            x = i[0].detach().abs().reshape(-1, i[0].shape[-1]).float().mean(0)
            sig[n + ".weight"] = sig.get(n + ".weight", torch.zeros_like(x)) + x
        return h
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.weight.shape[1] >= 256:
            hooks.append(m.register_forward_hook(mk(n)))
    ids = tok(open(PT, errors="ignore").read(), return_tensors="pt").input_ids[:, :2048].to(DEV)
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()
    save_file({k: (v.cpu().float() + 1e-6) for k, v in sig.items()}, SIGPATH)
    del model; gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()


CONFIGS = {
    "full": [("4-RHT", build_rht, (4,)), ("4-AWQ", build_awq, (4,)),
             ("3-RHT", build_rht, (3,)), ("3-AWQ", build_awq, (3,)),
             ("3-AWQ.25", build_awq, (3, 0.25)), ("3-AWQ.75", build_awq, (3, 0.75)),
             ("2-RHT", build_rht, (2,)), ("2-AWQ", build_awq, (2,)),
             ("1-RHT", build_rht, (1,)), ("1-AWQ", build_awq, (1,)),
             ("res3+2", build_residual, (3, 2)), ("res2+2", build_residual, (2, 2)),
             ("res2+1", build_residual, (2, 1)), ("res1+1", build_residual, (1, 1))],
    "essential": [("4-AWQ", build_awq, (4,)), ("3-AWQ", build_awq, (3,)),
                  ("2-AWQ", build_awq, (2,)), ("1-AWQ", build_awq, (1,)), ("1-RHT", build_rht, (1,)),
                  ("res3+2", build_residual, (3, 2)), ("res2+2", build_residual, (2, 2)),
                  ("res2+1", build_residual, (2, 1)), ("res1+1", build_residual, (1, 1))],
    # comprehensive 4/3/2/1-bit local frontier: mixed-precision (measured density win) + the
    # 3-bit floor alpha-sweep + 2-bit rescue attempts. Known essential results are seeded into
    # the JSONL so they skip; ordered most-valuable-first so the best data lands while watched.
    "frontier": [
        ("mp-4a3f", build_awq, (3, 0.5, {"q_proj": 4, "k_proj": 4, "v_proj": 4, "o_proj": 4,
                                          "gate_proj": 3, "up_proj": 3, "down_proj": 3})),
        ("mp-4a2f", build_awq, (2, 0.5, {"q_proj": 4, "k_proj": 4, "v_proj": 4, "o_proj": 4,
                                          "gate_proj": 2, "up_proj": 2, "down_proj": 2})),
        ("mp-3a2f", build_awq, (2, 0.5, {"q_proj": 3, "k_proj": 3, "v_proj": 3, "o_proj": 3,
                                          "gate_proj": 2, "up_proj": 2, "down_proj": 2})),
        ("3-AWQ.25", build_awq, (3, 0.25)), ("3-AWQ.75", build_awq, (3, 0.75)),
        ("3-RHT", build_rht, (3,)),
        ("2-AWQ.25", build_awq, (2, 0.25)), ("2-AWQ.75", build_awq, (2, 0.75)),
        ("2-RHT", build_rht, (2,)),
        ("4-RHT", build_rht, (4,)),
        # --- sub-3-bit rescue suite (the most important: can 1/2-bit be made usable on 7B?) ---
        # train-free: keep top-|w| 5/10% at 8-bit (outlier channel)
        ("2-AWQ-o5", build_awq, (2, 0.5, None, 5.0)), ("2-AWQ-o10", build_awq, (2, 0.5, None, 10.0)),
        ("1-AWQ-o5", build_awq, (1, 0.5, None, 5.0)), ("1-AWQ-o10", build_awq, (1, 0.5, None, 10.0)),
        # recovery: AWQ base + LoRA-KD heal (doctor) — the real sub-3-bit lever
        ("3-AWQ+dr", build_recover, (3,)), ("2-AWQ+dr", build_recover, (2,)),
        ("2-AWQ-o5+dr", build_recover, (2, 60, 64, 1e-4, 0.5, None, 5.0)),
        ("1-AWQ+dr", build_recover, (1,)), ("1-AWQ-o10+dr", build_recover, (1, 60, 64, 1e-4, 0.5, None, 10.0)),
        # seeded/known (skip on resume):
        ("4-AWQ", build_awq, (4,)), ("3-AWQ", build_awq, (3,)), ("2-AWQ", build_awq, (2,)),
        ("1-AWQ", build_awq, (1,)), ("1-RHT", build_rht, (1,))],
    # STUDIO — interruption-safe/default set for the M3 Ultra ladder. L4 blockwise is excluded until
    # it supports sharded sources, requested dtype, and per-layer resume; L5 STRAND is excluded until
    # its requested-model and durable optimizer-state gates pass. `studio_full` retains those research
    # configs behind HAWKING_STUDIO_RESEARCH_FULL=1 in studio_run.py.
    "studio": [
        ("4-AWQ", build_awq, (4,)), ("3-AWQ", build_awq, (3,)),
        ("2-AWQ", build_awq, (2,)), ("1-AWQ", build_awq, (1,)),
        ("mp-4a3f", build_awq, (3, 0.5, _mp(4, 3))), ("mp-3a2f", build_awq, (2, 0.5, _mp(3, 2))),
        ("res3+2", build_residual, (3, 2)), ("res2+1", build_residual, (2, 1)),
        ("3-AWQ+dr", build_recover, (3,)), ("2-AWQ+dr", build_recover, (2,)),
        ("1-AWQ+dr", build_recover, (1,)),
    ],
    # Full L0-L6 research set. Not production-ready on sharded 7B+ parents; explicit opt-in only.
    "studio_full": [
        ("4-AWQ", build_awq, (4,)), ("3-AWQ", build_awq, (3,)),
        ("2-AWQ", build_awq, (2,)), ("1-AWQ", build_awq, (1,)),
        ("mp-4a3f", build_awq, (3, 0.5, _mp(4, 3))), ("mp-3a2f", build_awq, (2, 0.5, _mp(3, 2))),
        ("res3+2", build_residual, (3, 2)), ("res2+1", build_residual, (2, 1)),
        ("3-AWQ+dr", build_recover, (3,)), ("2-AWQ+dr", build_recover, (2,)),
        ("1-AWQ+dr", build_recover, (1,)),
        ("3-bw", build_blockwise, (3,)), ("2-bw", build_blockwise, (2,)),
        ("1-bw", build_blockwise, (1,)),
        ("2-str", build_strand, (2,)), ("1-str", build_strand, (1,)),
    ],
    # SUBBIT — vector-trellis reconstruction-oracle campaign. SUBBIT-0 is a non-gating theory
    # bound; each real row below must emit exact accounting and remains ineligible for floor claims.
    "subbit": list(VTQ_CONFIGS),
    "subbit_full": [
        *VTQ_CONFIGS,
        # Codec-native/full-rank recovery research remains opt-in and is not VTQ-core coverage.
        ("1-str",        build_strand,   (1,)),
        ("1-bw",         build_blockwise, (1,)),
    ],
}


def _try_inject():
    """Execute an inject script if present, then delete it. Drop a Python file at
    {OUTP}_inject.py to modify globals (e.g. change bake thread count, skip a config,
    redefine a build function) between configs without restarting the process."""
    ipath = f"{OUTP}_inject.py"
    if not os.path.exists(ipath):
        return
    if os.environ.get("STUDIO_TRIPWIRE") == "1":
        raise RuntimeError(
            f"proof-mode audit refuses mutable injection {ipath}; encode the recipe in source "
            "and restart so its hash is part of the audit identity"
        )
    code = open(ipath).read()
    os.remove(ipath)
    log(f"# INJECT: {ipath}")
    try:
        exec(code, globals())  # noqa: S102 — intentional, user-controlled
        log("# INJECT: ok")
    except Exception as e:
        log(f"# INJECT: error — {e}")


def _acquire_lock():
    """Serialize every lane for one model and interoperate with already-live legacy locks.

    The old lock was lane/output-prefix-specific while scratch was model-specific, so a Studio and
    sub-bit launch for the same label could both pass their locks and delete each other's files.  A
    model-wide advisory flock is the authority for new processes; the legacy lock scan keeps a new
    process from colliding with a worker started before this fix.
    """
    model_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", LABEL)
    model_lock_path = f"/tmp/hawking-audit-{model_id}.model.lock"
    model_lock = open(model_lock_path, "a+")
    try:
        fcntl.flock(model_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        model_lock.seek(0)
        owner = model_lock.read().strip() or "unknown"
        model_lock.close()
        log(f"# BLOCKED: model-wide audit lock held for {LABEL}: {owner}")
        sys.exit(75)

    # Current detached Studio workers were launched with the former per-output lock only.  Refuse
    # overlap while any such PID is still alive; stale files are harmless and left for their owner.
    legacy_locks = set(glob.glob(f"reports/cron/*_{glob.escape(LABEL)}.lock"))
    legacy_locks.add(f"{OUTP}.lock")

    def _blocked_legacy(detail):
        try:
            fcntl.flock(model_lock.fileno(), fcntl.LOCK_UN)
            model_lock.close()
        finally:
            log(f"# BLOCKED: {detail}")
        sys.exit(75)

    for legacy in sorted(legacy_locks):
        if not os.path.exists(legacy):
            continue
        try:
            other = int(open(legacy).read().strip())
        except FileNotFoundError:
            continue  # benign exists/read race with a normally exiting legacy worker
        except (ValueError, OSError) as exc:
            _blocked_legacy(f"cannot validate legacy audit lock {legacy}: {type(exc).__name__}: {exc}")
        if other == os.getpid():
            continue
        try:
            os.kill(other, 0)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            _blocked_legacy(f"cannot inspect live legacy PID {other} from {legacy}: {exc}")
        # Avoid a stale PID file blocking forever after PID reuse.  An unreadable command is still
        # uncertainty and therefore blocks; only a positively identified unrelated command is stale.
        try:
            probe = subprocess.run(
                ["ps", "-p", str(other), "-o", "command="],
                capture_output=True, text=True, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            _blocked_legacy(f"legacy PID {other} command probe failed ({legacy}): {exc}")
        if probe.returncode != 0 or not probe.stdout.strip():
            _blocked_legacy(f"legacy PID {other} is live but its command cannot be verified ({legacy})")
        command = probe.stdout.strip()
        if "audit_ladder.py" in command and LABEL in command:
            _blocked_legacy(f"legacy {LABEL} audit PID {other} holds {legacy}")
        log(f"# reclaiming stale legacy lock {legacy}: PID {other} now runs unrelated command")

    model_lock.seek(0)
    model_lock.truncate()
    model_lock.write(json.dumps({"pid": os.getpid(), "label": LABEL, "set": SETNAME,
                                 "out": OUTP}) + "\n")
    model_lock.flush()
    os.fsync(model_lock.fileno())

    lock = f"{OUTP}.lock"
    os.makedirs(os.path.dirname(lock) or ".", exist_ok=True)
    with open(lock, "w") as handle:
        handle.write(str(os.getpid()) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def _release():
        try:
            if os.path.exists(lock) and int(open(lock).read().strip()) == os.getpid():
                os.remove(lock)
        except Exception:
            pass
        try:
            fcntl.flock(model_lock.fileno(), fcntl.LOCK_UN)
            model_lock.close()
        except Exception:
            pass
    atexit.register(_release)


def _load_verify_set():
    """Build CONFIGS['verify'] from a JSON spec so the verifier can independently REPRODUCE a
    chosen set of top candidates. Spec = [[name, fn_name, [args...]], ...]; fn resolved from
    module globals (build_awq/build_recover/build_rht/build_residual). Lets the verifier re-bake
    + re-doctor the winners from scratch under MULTIWINDOW eval — independent reproduction is the
    strongest 'this result is real' guarantee (catches eval-noise, overfit, and non-determinism)."""
    spec_path = os.environ.get("VERIFY_SPEC")
    if not spec_path or not os.path.exists(spec_path):
        raise RuntimeError(f"SETNAME=verify needs VERIFY_SPEC json (got {spec_path!r})")
    spec = json.load(open(spec_path))
    out = []
    for name, fn_name, args in spec:
        fn = globals().get(fn_name)
        if fn is None:
            raise RuntimeError(f"verify spec: unknown build fn {fn_name!r} for {name}")
        out.append((name, fn, tuple(args)))
    CONFIGS["verify"] = out
    log(f"# verify set: {len(out)} candidates — {[c[0] for c in out]}")


def main():
    global SRC
    _acquire_lock()
    _prepare_audit_identity()
    SRC = _resolve_src()
    if SETNAME == "verify":
        _load_verify_set()
    log(f"# audit {LABEL} dev={DEV} dtype={DTYPE} set={SETNAME} cpu={os.cpu_count()} free={_free_gb():.0f}GB")

    # --- checkpoint resume: scan JSONL for already-completed configs ---
    records, completed, failed = {}, {}, {}
    if os.path.exists(f"{OUTP}.jsonl"):
        for line in open(f"{OUTP}.jsonl"):
            try:
                rec = json.loads(line.strip())
                cfg = rec.get("config")
                if cfg:
                    records[cfg] = rec
            except Exception:
                pass
    for cfg, rec in records.items():
        if "ppl" in rec:
            completed[cfg] = rec
        elif "error" in rec:
            failed[cfg] = rec
    if completed:
        log(f"# resume: {len(completed)} checkpointed — {sorted(completed)}")
    retry_errors = os.environ.get("LADDER_RETRY_ERRORS", "0").lower() in {"1", "true", "yes"}
    skip = dict(completed)
    if failed and not retry_errors:
        skip.update(failed)
        log(f"# resume: {len(failed)} error-checkpointed — {sorted(failed)}")

    # A mere ``ppl`` key is not a completion marker.  Remove malformed generic rows before the f16
    # branch reads them; otherwise a bad f16 row can poison every degradation calculation and a bad
    # scalar row can remain skipped forever.  The durable JSONL remains append-only; the fresh row
    # appended below becomes the newest record for that config.
    for config, row in list(completed.items()):
        if not _measurement_row_valid(config, row):
            completed.pop(config, None)
            skip.pop(config, None)
            records.pop(config, None)
            log(f"# resume: requeue {config} — malformed/non-finite measurement checkpoint")

    # --- f16 baseline (resumable) ---
    if "f16" in completed:
        hf = completed["f16"]["ppl"]
        log(f"# f16 ppl = {hf:.3f} (checkpoint)")
    else:
        hf = measure(None)
        log(f"# f16 ppl = {hf:.3f}")
        _append_jsonl_durable(
            f"{OUTP}.jsonl",
            {"model": LABEL, "config": "f16", "eff_bpw": 16.0,
             "ppl": round(hf, 3), "degr_pct": 0.0},
        )

    # Bind every scalar/Doctor/VTQ degradation checkpoint to this run's valid f16 baseline.  This
    # catches internally inconsistent hand-written or partially migrated rows without imposing any
    # codec-specific assumptions; exact Doctor and VTQ identity/accounting checks follow below.
    for config, row in list(completed.items()):
        if config != "f16" and not _measurement_row_valid(config, row, hf):
            completed.pop(config, None)
            skip.pop(config, None)
            records.pop(config, None)
            log(f"# resume: requeue {config} — measurement does not match f16 checkpoint")

    # The task tripwire is a relative gate, never an absolute score. Capture the same parent's f16
    # result once, bind it to the model fingerprint, and put it in the audit ledger. Missing baseline
    # evidence is fatal when Studio proof mode is enabled.
    tripwire_baseline = None
    if os.environ.get("STUDIO_TRIPWIRE") == "1":
        baseline_receipt = _ensure_tripwire_baseline()
        tripwire_baseline = baseline_receipt["result"]
        f16_row = {"model": LABEL, "config": "f16", "eff_bpw": 16.0,
                   "ppl": round(hf, 3), "degr_pct": 0.0,
                   "tripwire": tripwire_baseline,
                   "tripwire_gate": {"schema": "hawking.tripwire_gate.v1",
                                     "status": "baseline", "policy": tripwire_policy()}}
        if records.get("f16", {}).get("tripwire") != tripwire_baseline:
            _append_jsonl_durable(f"{OUTP}.jsonl", f16_row)
        records["f16"] = f16_row
        completed["f16"] = f16_row
        skip["f16"] = f16_row

        # Old runs did not tripwire Doctor dict artifacts. Requeue only a PPL-floor-eligible row
        # whose task result is missing/malformed; a valid task result that genuinely fails the gate
        # remains measured and is simply ineligible for promotion.
        floor_gate = float(os.environ.get("FLOOR_GATE_PCT", "2.0"))
        for config, row in list(completed.items()):
            if config == "f16" or not isinstance(row.get("degr_pct"), (int, float)):
                continue
            if float(row["degr_pct"]) <= floor_gate \
                    and not validate_tripwire_result(row.get("tripwire"))["ok"]:
                skip.pop(config, None)
                records.pop(config, None)
                log(f"# resume: requeue {config} — floor-eligible row lacks valid task tripwire")

    # Rows produced before exact adapter billing understate effective bpw (r64 was +1.574 bpw at
    # 0.5B and +0.902 at 1.5B). They are measurements, but not valid density evidence; rerun them
    # with the now-faster low-rank Doctor instead of silently carrying the old number forward.
    for config, row in list(completed.items()):
        if "+dr" in config and not _doctor_accounting_valid(row):
            skip.pop(config, None)
            records.pop(config, None)
            log(f"# resume: requeue {config} — missing complete rank/exact-adapter-bpw evidence")
        elif config.startswith("vtq-") and not _vtq_row_valid_for_resume(config, row):
            skip.pop(config, None)
            records.pop(config, None)
            log(f"# resume: requeue {config} — stale/malformed VTQ recipe or accounting")

    # Consume a saved inject before deciding whether expensive sigma capture is still needed.
    _try_inject()
    pending = [(name, fn, args) for name, fn, args in CONFIGS[SETNAME] if name not in skip]
    if any(fn in (build_awq, build_recover)
           for _, fn, _ in pending):
        capture_sigma()
    else:
        log("# sigma not needed; no pending AWQ/recovery configs")

    rows = list(records.values())
    for name, fn, args in CONFIGS[SETNAME]:
        global LAST_BUILD_META, LAST_ORACLE_META
        LAST_BUILD_META = None
        LAST_ORACLE_META = None
        _try_inject()  # execute inject script if present before each config

        if name in skip:
            rec = skip[name]
            if "ppl" in rec:
                log(f"  {name:10s} -> SKIP (checkpoint +{rec.get('degr_pct','?')}%)")
            else:
                log(f"  {name:10s} -> SKIP (error checkpoint: {rec.get('error','')[:60]})")
            continue

        _clean_temps()   # sweep any leftover chunk parts/inputs from an interrupted run

        try:
            path, bpw = fn(*args)
            p = measure(path)
            rec = {"model": LABEL, "config": name, "eff_bpw": round(bpw, 3),
                   "ppl": round(p, 3), "degr_pct": round((p / hf - 1) * 100, 2)}
            if LAST_ORACLE_META is not None:
                rec["artifact_class"] = "reconstruction_oracle"
                rec["deployable"] = False
                rec["oracle"] = LAST_ORACLE_META
            if LAST_BUILD_META is not None:
                rec["doctor"] = LAST_BUILD_META
            # Capability tripwire on floor candidates: a floor claim is void if ppl is ~1:1 but a
            # downstream task collapses (§6). The floor selector independently recomputes this gate.
            tw_gate = float(os.environ.get("STUDIO_TRIPWIRE_GATE", "8.0"))
            if (os.environ.get("STUDIO_TRIPWIRE") == "1"
                    and isinstance(path, (list, dict)) and rec["degr_pct"] <= tw_gate):
                merged = f"{T}_trip.safetensors"
                try:
                    adapter = None
                    if isinstance(path, dict):
                        override = path.get("base")
                        adapter = path.get("adapter")
                    else:
                        _merge_parts(path, merged)
                        override = merged
                    tw_cmd = ["python3.12", "tools/condense/multi_eval.py", MODEL,
                              override or "-", name]
                    if adapter:
                        tw_cmd.append(adapter)
                    tw = subprocess.run(tw_cmd,
                                        capture_output=True, text=True, env=os.environ,
                                        pass_fds=_lease_fds())
                    last = tw.stdout.strip().splitlines()[-1] if tw.stdout.strip() else ""
                    try: result = json.loads(last)
                    except Exception: result = None
                    if tw.returncode == 0 and validate_tripwire_result(result)["ok"]:
                        rec["tripwire"] = result
                    else:
                        rec["tripwire_error"] = {
                            "returncode": tw.returncode,
                            "stdout_tail": last[-200:],
                            "stderr_tail": tw.stderr[-200:],
                            "validation": validate_tripwire_result(result),
                        }
                    rec["tripwire_gate"] = compare_tripwire(tripwire_baseline, result)
                finally:
                    try: os.remove(merged)
                    except OSError: pass
        except Exception as e:
            rec = {"model": LABEL, "config": name, "error": str(e)[:140]}
        rows.append(rec)
        log(f"  {name:10s} -> {rec.get('eff_bpw','?')} bpw  +{rec.get('degr_pct','?')}%")
        _append_jsonl_durable(f"{OUTP}.jsonl", rec)
        _clean_temps()   # free this config's parts before the next one
    # markdown
    with open(f"{OUTP}.md", "w") as o:
        o.write(f"## {LABEL} ladder (f16 ppl {hf:.2f}) — effective bpw vs degradation\n\n")
        o.write("| config | eff bpw | degr vs f16 |\n|---|--:|--:|\n")
        for r in sorted([x for x in rows if "error" not in x], key=lambda x: x["eff_bpw"]):
            o.write(f"| {r['config']} | {r['eff_bpw']:.2f} | +{r['degr_pct']:.1f}% |\n")
        for r in [x for x in rows if "error" in x]:
            o.write(f"| {r['config']} | ERR | {r['error']} |\n")
    log(f"# done -> {OUTP}.md / {OUTP}.jsonl")


if __name__ == "__main__":
    main()
