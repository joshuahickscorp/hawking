#!/usr/bin/env python3.12
"""awq.py - merged tool: bake (was awq_bake.py) + plus (was awq_plus.py).

AWQ pre-scale + bake: bake (alpha-scaled STRAND bake, the L1 recovery step, bare script), plus (per-tensor AWQ alpha search on folded col-weighted OUTPUT error).

  awq.py bake <args...>   # was: python3.12 tools/condense/awq_bake.py <args...>
  awq.py plus <args...>   # was: python3.12 tools/condense/awq_plus.py <args...>
"""
import sys

def _run_bake():
    """AWQ-bake: activation-aware STRAND condense. Scale each weight's input columns by the
    activation magnitude^alpha BEFORE quant (protects high-activation channels), unscale after
    — folded into the weight so serving is unchanged. Proven in the output-space harness
    (3-bit 1.96x -> 1.28x). Training-free => the doctor then has a far smaller gap to close.

    Usage: awq_bake.py <hf-model-dir> <out.safetensors> [bits] [alpha]
    """
    import sys, os, subprocess, torch, torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file, save_file

    MODEL = sys.argv[1]
    OUT = sys.argv[2]
    TAG = os.path.basename(OUT).replace(".safetensors", "").replace("/", "_")  # unique temps per run
    BITS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    ALPHA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    BAKER = "vendor/strand-quant/target/release/quantize-model"
    # 7B+ won't fit 19GB in f32 and MPS-f16 has the GQA bug -> DOCTOR_DEVICE=cpu DOCTOR_DTYPE=float16
    # gives correct (CPU has no f16 bug) + fitting (14GB) at the cost of speed. Mac Studio: mps/float32.
    dev = os.environ.get("DOCTOR_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    DTYPE = getattr(torch, os.environ.get("DOCTOR_DTYPE", "float32"))
    CALIB = open(os.environ.get("DOCTOR_CALIB", "scratch/calib_corpus.txt"), errors="ignore").read()[:20000]

    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE, attn_implementation="eager").to(dev).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)

    # capture per-input-channel mean|x| per linear (the AWQ importance)
    sig, hooks = {}, []
    def mk(name):
        def h(mod, inp, out):
            x = inp[0].detach().abs().reshape(-1, inp[0].shape[-1]).mean(0)
            sig[name] = sig.get(name, torch.zeros_like(x)) + x
        return h
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and mod.weight.shape[1] >= 256:
            hooks.append(mod.register_forward_hook(mk(name)))
    ids = tok(CALIB, return_tensors="pt").input_ids[:, :2048].to(dev)
    with torch.no_grad():
        m(ids)
    for h in hooks:
        h.remove()
    print(f"# captured activation scale for {len(sig)} linears", file=sys.stderr)

    sd = load_file(os.path.join(MODEL, "model.safetensors"))
    scaled, scales = {}, {}
    for name, mod in m.named_modules():
        k = name + ".weight"
        if isinstance(mod, nn.Linear) and k in sd and name in sig:
            s = (sig[name].cpu().float() + 1e-6) ** ALPHA            # [in_features]
            scaled[k] = (sd[k].float() * s).to(torch.float16)        # scale columns
            scales[k] = s
        elif k in sd:
            scaled[k] = sd[k]
    save_file(scaled, f"/tmp/awq_scaled_{TAG}.safetensors")

    subprocess.run([BAKER, "--in", f"/tmp/awq_scaled_{TAG}.safetensors", "--out", f"/tmp/awq_baked_{TAG}.safetensors",
                    "--bits", str(BITS), "--quality", "--rht-cols", "--outlier-channel", "1",
                    "--outlier-bits", "8", "--threads", "10"], check=True, capture_output=True)

    baked = load_file(f"/tmp/awq_baked_{TAG}.safetensors")
    awq = {}
    for k, v in baked.items():
        awq[k] = (v.float() / scales[k]).to(torch.float16) if k in scales else v   # unscale -> fold
    save_file(awq, OUT)
    print(f"AWQ base ({BITS}-bit, alpha={ALPHA}) saved -> {OUT}")




def _run_plus():
    """AWQ-PLUS — per-TENSOR activation-aware scaling for STRAND condense.

    awq_bake.py uses ONE global alpha (0.5) for every linear: it scales each weight's
    input columns by σ_c^alpha (σ_c = mean_calib|x_c|) BEFORE the trellis baker, then
    unscales after — folded into the weight, so serving is unchanged. But the optimal
    alpha is NOT uniform: a tensor with a few super-outlier channels wants a HIGH alpha
    (aggressively protect them); a tensor with flat activation wants alpha≈0 (scaling
    only distorts the trellis grid for no protection gain). A single global alpha leaves
    quality on the table on both ends.

    This tool searches the best alpha PER TENSOR on a held-out output-space objective and
    bakes the per-tensor-alpha base. It reuses awq_bake's exact capture + scale + bake +
    unscale logic (copied here so the running ladder audit's awq_bake.py is untouched).

    ────────────────────────────────────────────────────────────────────────────────────
    OBJECTIVE  (per tensor t, candidate alpha a)
    ────────────────────────────────────────────────────────────────────────────────────
    What matters is OUTPUT-space error after the fold, not raw weight rel-RMS (memory:
    weight-space rel-RMS is "proxy-limbo"). For Y = W·X, with the per-input-channel
    activation magnitude σ as a diagonal proxy for X (the same output-space proxy
    mixed_precision.py endorses), the decode error that lands in the output stream is

        err(t,a) = || ( unscale_a(Q(W · s_a)) − W ) · diag(σ_t) ||_F  /  || W · diag(σ_t) ||_F
                   s_a = (σ_t + 1e-6)^a            (awq_bake's column scale)
                   unscale_a(·) = (·) / s_a        (awq_bake's fold-back)

    This is the TRUE folded reconstruction error in column-(=output)-weighted space — it
    needs the actual decoded Ŵ (a write-bake), not an analytic guess, and it correctly
    captures the protect-the-hot-channel trade-off that drives the alpha choice. We pick
    argmin_a err(t,a) per tensor (best alpha on this held-out objective).

    COST. We bake the FULL model once per candidate alpha (write mode), then read every
    decoded tensor back and score ALL tensors from that one bake — so the cost is
    N_alphas full bakes, NOT N_alphas × N_tensors. A final assembly bake stitches each
    tensor at its own best alpha and unscales (one more bake). The baker auto-selects
    Metal on macOS, so these probes run on the GPU and do NOT contend with the ladder
    audit's CPU baker (DOCTOR_DEVICE=cpu) — still, keep --threads modest beside a live run.

    ────────────────────────────────────────────────────────────────────────────────────
    MODES
    ────────────────────────────────────────────────────────────────────────────────────
      per-tensor (default): search --alphas per tensor, assemble best-alpha base, prove
                            ppl vs the global-alpha baseline (honest WIN/null).
      --sweep             : alpha-SWEEP — bake one global-alpha base per candidate alpha,
                            report ppl for each (the curve global awq_bake sits on).

    HONORS DOCTOR_DEVICE / DOCTOR_DTYPE for the ppl forward pass and σ capture (0.5B →
    mps/float32; 7B → cpu/bfloat16; NEVER float16 — MPS f16 GQA bug + 7B fp16 overflow→nan).

    Usage:
      # PLAN per-tensor alphas + assemble + prove (default).  3-bit base.
      python3.12 tools/condense/awq_plus.py scratch/qwen-05b out/awq_pt_3b.safetensors --bits 3

      # LIGHT self-test: few tensors, few alphas, fast (non-quality) bake — does not contend.
      python3.12 tools/condense/awq_plus.py scratch/qwen-05b /tmp/awqp_selftest.safetensors \
            --bits 3 --alphas 0.25,0.5,0.75 --limit-tensors 4 --fast

      # alpha-SWEEP (the global curve):
      python3.12 tools/condense/awq_plus.py scratch/qwen-05b /tmp/awqp_sweep.safetensors \
            --bits 3 --alphas 0.0,0.25,0.5,0.75,1.0 --sweep
    """
    import sys, os, re, gc, json, math, time, argparse, subprocess
    import torch, torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
    BAKER = os.path.join(ROOT, "vendor", "strand-quant", "target", "release", "quantize-model")
    # 7B+ won't fit 19GB f32 and MPS-f16 has the GQA bug -> DOCTOR_DEVICE=cpu DOCTOR_DTYPE=bfloat16.
    DEV = os.environ.get("DOCTOR_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    DTYPE = getattr(torch, os.environ.get("DOCTOR_DTYPE", "float32"))
    PT = os.environ.get("PPL_TEXT", "/tmp/ppl24k.txt")
    # capture corpus: calib_corpus + calib_multidomain (if present), matching the doctor toolchain.
    CALIB = os.environ.get("DOCTOR_CALIB", os.path.join(ROOT, "scratch", "calib_corpus.txt"))
    CALIB_MD = os.path.join(ROOT, "scratch", "calib_multidomain.txt")


    def log(*m):
        print(*m, file=sys.stderr); sys.stderr.flush()


    def audit_running():
        try:
            r = subprocess.run(["pgrep", "-fl", "quantize-model"], capture_output=True, text=True)
            return [ln for ln in r.stdout.splitlines() if "quantize-model" in ln]
        except Exception:
            return []


    def calib_text():
        txt = open(CALIB, errors="ignore").read() if os.path.exists(CALIB) else open(PT, errors="ignore").read()
        if os.path.exists(CALIB_MD):
            txt = txt + "\n" + open(CALIB_MD, errors="ignore").read()
        return txt[:20000]


    def model_src(model_dir):
        """Single-file or sharded HF dir → a path the baker can read. The baker reads one
        safetensors; for sharded models we materialize a merged file once (7B is sharded)."""
        one = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(one):
            return one, load_file(one)
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            merged = f"/tmp/awqp_merged_{os.path.basename(model_dir.rstrip('/'))}.safetensors"
            if not os.path.exists(merged):
                wm = json.load(open(idx))["weight_map"]
                sd = {}
                for shard in sorted(set(wm.values())):
                    sd.update(load_file(os.path.join(model_dir, shard)))
                save_file(sd, merged)
                log(f"# merged {len(set(wm.values()))} shards → {merged}")
            return merged, load_file(merged)
        raise FileNotFoundError(f"no model.safetensors[.index.json] in {model_dir}")


    # ── baker (copied baker invocation; Metal auto on macOS) ───────────────────────────────
    def bake(inp, out, bits, quality=True, threads=10, measure_only=False, only=None):
        cmd = [BAKER, "--in", inp, "--out", out, "--bits", str(bits), "--rht-cols",
               "--outlier-channel", "1", "--outlier-bits", "8", "--threads", str(threads)]
        if quality:
            cmd += ["--quality"]
        if measure_only:
            cmd += ["--measure-only"]
        if only:
            cmd += ["--only", only]
        r = subprocess.run(cmd, capture_output=True, text=True)
        blob = r.stderr + r.stdout
        if r.returncode != 0:
            raise RuntimeError(f"baker failed: {blob.strip().splitlines()[-4:]}")
        agg = re.search(r"AGGREGATE effective bpw = ([\d.]+)", blob)
        return float(agg.group(1)) if agg else float("nan")


    # ── capture per-input-channel σ (COPIED from awq_bake.py) ──────────────────────────────
    def capture_sigma(model_dir):
        """σ[name] = Σ_calib mean|x| over input channels for each Linear (the AWQ importance).
        Same hook / same corpus as awq_bake.py. Returns {tensor.weight: tensor[in_features]}."""
        m = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=DTYPE, attn_implementation="eager").to(DEV).eval()
        tok = AutoTokenizer.from_pretrained(model_dir)
        sig, hooks = {}, []

        def mk(name):
            def h(mod, inp, out):
                x = inp[0].detach().abs().reshape(-1, inp[0].shape[-1]).float().mean(0)
                sig[name] = sig.get(name, torch.zeros_like(x)) + x
            return h

        for name, mod in m.named_modules():
            if isinstance(mod, nn.Linear) and mod.weight.shape[1] >= 256:
                hooks.append(mod.register_forward_hook(mk(name)))
        ids = tok(calib_text(), return_tensors="pt").input_ids[:, :2048].to(DEV)
        with torch.no_grad():
            m(ids)
        for h in hooks:
            h.remove()
        out = {k + ".weight": v.cpu().float() for k, v in sig.items()}
        log(f"# captured activation σ for {len(out)} linears")
        del m
        gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()
        return out


    def scaled_weights(W, sigma, alpha_of):
        """Build the column-scaled state dict (awq_bake's scale step), alpha PER tensor.
        Returns (scaled_sd, scales{name: s_vector}). Non-quantized tensors pass through."""
        scaled, scales = {}, {}
        for k, v in W.items():
            a = alpha_of(k)
            if k in sigma and a is not None:
                s = (sigma[k] + 1e-6) ** a                 # [in_features]
                scaled[k] = (v.float() * s).to(torch.float16)
                scales[k] = s
            else:
                scaled[k] = v
        return scaled, scales


    def unscale_fold(baked, scales):
        """awq_bake's fold-back: divide decoded columns by the same s, fold into the weight."""
        out = {}
        for k, v in baked.items():
            out[k] = (v.float() / scales[k]).to(torch.float16) if k in scales else v
        return out


    # ── per-tensor output-space error of a decoded+unscaled tensor ─────────────────────────
    def col_weighted_relerr(Wt, Wht, sig_t):
        """|| (Ŵ − W)·diag(σ) ||_F / || W·diag(σ) ||_F  — output-space (column-weighted) error."""
        s = sig_t.to(Wt.dtype)
        num = ((Wht - Wt) * s).pow(2).sum().sqrt()
        den = (Wt * s).pow(2).sum().sqrt().clamp_min(1e-12)
        return float((num / den).item())


    # ── ppl (real forward pass; honors DEV/DTYPE) ──────────────────────────────────────────
    def ppl(model_dir, override):
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=DTYPE, attn_implementation="eager").to(DEV).eval()
        if override:
            sd = model.state_dict()
            with safe_open(override, framework="pt") as f:
                for k in f.keys():
                    if k in sd and tuple(sd[k].shape) == tuple(f.get_slice(k).get_shape()):
                        sd[k].copy_(f.get_tensor(k).to(DEV, DTYPE))
        txt = open(PT, errors="ignore").read()
        ids = tok(txt, return_tensors="pt").input_ids[:, :2048].to(DEV)
        with torch.no_grad():
            loss = model(ids, labels=ids).loss.item()
        del model
        gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()
        return math.exp(loss)


    # ── search: best alpha per tensor over the output-space objective ───────────────────────
    def search_alphas(src, W, sigma, alphas, bits, quality, threads, names):
        """For each candidate alpha: bake the FULL global-alpha-scaled model once (write),
        read decoded tensors back, unscale, score the col-weighted output error for every
        tensor. One full bake per alpha → score all tensors. Returns best_alpha{name}."""
        err = {n: {} for n in names}                    # err[name][alpha]
        Wf = {k: v.float() for k, v in W.items() if k in names}
        for a in alphas:
            scaled, scales = scaled_weights(W, sigma, lambda k, _a=a: _a if k in sigma else None)
            tmp_in = f"/tmp/awqp_scan_in_{a}.safetensors"
            tmp_out = f"/tmp/awqp_scan_out_{a}.safetensors"
            save_file(scaled, tmp_in)
            t0 = time.time()
            bake(tmp_in, tmp_out, bits, quality=quality, threads=threads)
            baked = load_file(tmp_out)
            folded = unscale_fold(baked, scales)
            for n in names:
                if n in folded and n in sigma:
                    err[n][a] = col_weighted_relerr(Wf[n], folded[n].float(), sigma[n])
            log(f"  alpha={a}: full bake {time.time()-t0:.0f}s · scored {sum(1 for n in names if a in err[n])} tensors")
            os.remove(tmp_in); os.remove(tmp_out)
        best = {}
        for n in names:
            if err[n]:
                best[n] = min(err[n], key=err[n].get)
        return best, err


    def main():
        ap = argparse.ArgumentParser(description="AWQ-plus: per-tensor alpha (+ alpha sweep)")
        ap.add_argument("model_dir")
        ap.add_argument("out")
        ap.add_argument("--bits", type=int, default=3)
        ap.add_argument("--alphas", default="0.0,0.25,0.5,0.75,1.0",
                        help="candidate alphas (comma). Global awq_bake uses 0.5.")
        ap.add_argument("--sweep", action="store_true",
                        help="alpha-SWEEP: bake a global base per alpha, report ppl each (no per-tensor search)")
        ap.add_argument("--baseline-alpha", type=float, default=0.5,
                        help="global alpha to compare the per-tensor base against (awq_bake default 0.5)")
        ap.add_argument("--limit-tensors", type=int, default=None,
                        help="search only N tensors (rest take --baseline-alpha) — LIGHT self-test")
        ap.add_argument("--fast", action="store_true", help="non-quality bake (L=k+4) — self-test speed")
        ap.add_argument("--threads", type=int, default=10)
        ap.add_argument("--no-prove", action="store_true", help="skip the ppl comparison bake")
        args = ap.parse_args()

        quality = not args.fast
        alphas = [float(x) for x in args.alphas.split(",")]
        src, W = model_src(args.model_dir)
        log(f"# awq_plus · {os.path.basename(args.model_dir.rstrip('/'))} · bits={args.bits} · "
            f"alphas={alphas} · {'SWEEP' if args.sweep else 'per-tensor'} · dev={DEV}/{DTYPE} · "
            f"{'fast' if args.fast else 'quality'}")
        run = audit_running()
        if run:
            log(f"# NOTE: {len(run)} baker proc running (ladder audit). These probes use Metal (GPU); "
                f"audit baker is CPU — keep --threads modest. {'self-test is light.' if args.limit_tensors else ''}")

        sigma = capture_sigma(args.model_dir)

        # ── alpha SWEEP: the global curve ──────────────────────────────────────────────────
        if args.sweep:
            f16 = ppl(args.model_dir, None)
            log(f"  f16 ppl = {f16:.3f}")
            rows = []
            for a in alphas:
                scaled, scales = scaled_weights(W, sigma, lambda k, _a=a: _a if k in sigma else None)
                ti, to = f"/tmp/awqp_sw_in_{a}.safetensors", f"/tmp/awqp_sw_out_{a}.safetensors"
                save_file(scaled, ti)
                ebpw = bake(ti, to, args.bits, quality=quality, threads=args.threads)
                folded = unscale_fold(load_file(to), scales)
                save_file(folded, args.out)                 # last one persists to --out
                p = ppl(args.model_dir, args.out)
                rows.append(dict(alpha=a, bpw=round(ebpw, 3), ppl=round(p, 3),
                                 degr_pct=round((p / f16 - 1) * 100, 2)))
                log(f"  alpha={a}: eff {ebpw:.3f} bpw  ppl {p:.3f}  (+{(p/f16-1)*100:.2f}%)")
                os.remove(ti); os.remove(to)
            best = min(rows, key=lambda r: r["ppl"])
            print(json.dumps({"mode": "sweep", "model": os.path.basename(args.model_dir.rstrip('/')),
                              "bits": args.bits, "f16_ppl": round(f16, 3), "sweep": rows,
                              "best_alpha": best["alpha"], "best_degr_pct": best["degr_pct"]}))
            return

        # ── per-TENSOR alpha ────────────────────────────────────────────────────────────────
        qkeys = [k for k in W if k in sigma]                # the scalable linears
        if args.limit_tensors:
            # representative slice across roles (same spirit as mixed_precision.py)
            roles = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            picked, seen = [], set()
            for role in roles:
                for k in qkeys:
                    if role in k and role not in seen:
                        picked.append(k); seen.add(role); break
            for k in qkeys:
                if k not in picked and len(picked) < args.limit_tensors:
                    picked.append(k)
            names = picked[:args.limit_tensors]
            log(f"# self-test slice: {len(names)} tensors searched; "
                f"the other {len(qkeys)-len(names)} take baseline alpha {args.baseline_alpha}")
        else:
            names = qkeys

        t0 = time.time()
        best, err = search_alphas(src, W, sigma, alphas, args.bits, quality, args.threads, names)
        log(f"# searched {len(best)} tensors in {time.time()-t0:.0f}s. Per-tensor best alpha:")
        dist = {}
        for n in sorted(best):
            dist[best[n]] = dist.get(best[n], 0) + 1
        for n in list(sorted(best))[:10]:
            es = "  ".join(f"a{a}={err[n][a]:.4f}" for a in alphas if a in err[n])
            log(f"    {n:<46s} best={best[n]:<4} | {es}")
        log(f"# best-alpha distribution: {dist}  (global awq_bake would force all = {args.baseline_alpha})")

        # assemble the per-tensor-alpha base (fall back to baseline alpha for un-searched)
        def alpha_of(k):
            if k not in sigma:
                return None
            return best.get(k, args.baseline_alpha)

        scaled, scales = scaled_weights(W, sigma, alpha_of)
        ti = "/tmp/awqp_assemble_in.safetensors"
        to = "/tmp/awqp_assemble_out.safetensors"
        save_file(scaled, ti)
        ebpw = bake(ti, to, args.bits, quality=quality, threads=args.threads)
        folded = unscale_fold(load_file(to), scales)
        save_file(folded, args.out)
        os.remove(ti); os.remove(to)
        log(f"# per-tensor-alpha base saved → {args.out}  (eff {ebpw:.3f} bpw)")

        result = {"mode": "per-tensor", "model": os.path.basename(args.model_dir.rstrip('/')),
                  "bits": args.bits, "eff_bpw": round(ebpw, 3), "alpha_dist": dist,
                  "n_searched": len(best), "n_total_scalable": len(qkeys), "out": args.out,
                  "alpha_map": {n: best[n] for n in sorted(best)}}

        # prove: per-tensor alpha vs the global baseline at the SAME bits (≈iso-bpw)
        if not args.no_prove:
            f16 = ppl(args.model_dir, None)
            pt_ppl = ppl(args.model_dir, args.out)
            gscaled, gscales = scaled_weights(W, sigma, lambda k: args.baseline_alpha if k in sigma else None)
            save_file(gscaled, ti)
            g_bpw = bake(ti, to, args.bits, quality=quality, threads=args.threads)
            gfolded = unscale_fold(load_file(to), gscales)
            gpath = args.out + ".globalbase.safetensors"
            save_file(gfolded, gpath)
            g_ppl = ppl(args.model_dir, gpath)
            os.remove(ti); os.remove(to)
            win = (g_ppl / f16) - (pt_ppl / f16)
            verdict = ("WIN: per-tensor alpha beats global at ~iso-bpw" if win > 0
                       else "no win: global alpha ties/beats per-tensor here (honest null)")
            log(f"  f16 ppl              = {f16:.3f}")
            log(f"  GLOBAL alpha={args.baseline_alpha:<4}     : {g_bpw:.3f} bpw  ppl {g_ppl:.3f}  (+{(g_ppl/f16-1)*100:.2f}%)")
            log(f"  PER-TENSOR alpha     : {ebpw:.3f} bpw  ppl {pt_ppl:.3f}  (+{(pt_ppl/f16-1)*100:.2f}%)")
            log(f"  Δ(degr) global − per-tensor = {win*100:+.2f} pts → {verdict}")
            result.update(f16_ppl=round(f16, 3),
                          per_tensor=dict(bpw=round(ebpw, 3), ppl=round(pt_ppl, 3),
                                          degr_pct=round((pt_ppl/f16-1)*100, 2)),
                          global_base=dict(alpha=args.baseline_alpha, bpw=round(g_bpw, 3),
                                           ppl=round(g_ppl, 3), degr_pct=round((g_ppl/f16-1)*100, 2)),
                          win_pts=round(win*100, 2), verdict=verdict)
        print(json.dumps(result))

    main()


if __name__ == "__main__":
    _sub = sys.argv[1] if len(sys.argv) > 1 else "--help"
    if _sub == "bake":
        sys.argv = ["awq_bake.py"] + sys.argv[2:]
        _run_bake()
    elif _sub == "plus":
        sys.argv = ["awq_plus.py"] + sys.argv[2:]
        _run_plus()
    else:
        print(__doc__)
