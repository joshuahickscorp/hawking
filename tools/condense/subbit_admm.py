#!/usr/bin/env python3.12
"""subbit_admm.py — SUBBIT-2: NanoQuant-style low-rank BINARY factorization via ADMM (PROBE).

WHAT THIS IS (and is NOT). This is a BOUNDED research PROBE, not a serving codec. It MEASURES
whether a sub-bit binary low-rank approximation generalizes; it makes NO serving-win claim and
emits no deployable artifact. It exists to RE-TEST a named resurrection risk in the kill ledger:
post-hoc low-rank (the ASVD / data-aware-SVD family, and L2_lowrank_heal_NOGO) is DEAD because
its held-out functional error runs ~2x its in-sample error — it overfits the calibration slice.
NanoQuant is the same shape dressed as binary factors, so it inherits the same kill risk.

THE LEVER. Approximate a weight matrix W (rows x cols) as

        W  ~=  s * (B1 @ B2)        with  B1 in {-1,+1}^(rows x r),  B2 in {-1,+1}^(r x cols)

solved by ADMM: alternate (a) binarize each factor by sign with a least-squares scale, (b) a
dual/residual update that pulls the relaxed real factors back toward {-1,+1}. r is the rank arg.

EFFECTIVE BPW (honest, side-info included; never nominal). The binary factors cost r*(rows+cols)
sign bits; the per-matrix scale s is one fp32 number => SCALE_BITS. So

        eff_bpw = ( r*(rows+cols)*1  +  SCALE_BITS ) / (rows*cols)

We then build the PLAIN-RESIDUAL baseline (two passes of sign+per-row-scale binary quant, b1+b2)
and TRUNCATE its rank so its effective bpw MATCHES the ADMM probe's eff_bpw — a fair shoot-out at
identical storage. Both are scored on the SAME held-out activations.

FUNCTIONAL ERROR (output space, the only honest metric). For input activations X (captured from
the real model, or synthetic), the error of a reconstruction W_hat is

        err = || (W - W_hat) @ X^T ||_F  /  || W @ X^T ||_F

measured on a FIT split (the ADMM fit it) and a HELD-OUT split (it did NOT). Held-out is the verdict.

THE KILL (the criterion that REFUTES the lever — printed in --help and in the JSON):
  KILL if  heldout_err / fit_err > 1.5            (the dead ASVD overfit signature), OR
  KILL if  admm_heldout >= residual_heldout       (loses to plain residual at matched eff bpw).
On KILL: print  'KILL: NanoQuant is a low-rank resurrection'  and exit 1.
Only a probe that BOTH generalizes (ratio <= 1.5) AND beats matched-bpw residual on held-out
survives (exit 0) — and even then it is a probe result, not a serving win.

Env (matches the condense tools): DOCTOR_DEVICE (cpu/mps), DOCTOR_DTYPE (bfloat16 for 7B+ CPU),
STRAND_NO_GPU=1 (honored — keeps Metal idle). Stdlib + torch + safetensors only.

Usage:
  python3.12 tools/condense/subbit_admm.py [--model DIR] [--tensor NAME] [--rank R]
                                           [--iters N] [--calib FILE] [--ctx T]
                                           [--out reports/condense/subbit2_admm.json]
  python3.12 tools/condense/subbit_admm.py --self-test     # synthetic low-rank+noise; checks kill logic
  python3.12 tools/condense/subbit_admm.py --help

Defaults: model = scratch/qwen-05b (fits 18GB), tensor = model.layers.0.mlp.down_proj.weight.
If the model is absent, a SYNTHETIC structured matrix is used and the path is gated (--dry runs here).
"""
import sys, os, json, math, argparse

# Honor the repo's no-GPU contract before importing torch backends do anything heavy.
os.environ.setdefault("STRAND_NO_GPU", "1")
import torch

SCALE_BITS = 32                       # bits per fp32 scale; a per-ROW scale vector is counted below
DEFAULT_MODEL = "scratch/qwen-05b"
DEFAULT_TENSOR = "model.layers.0.mlp.down_proj.weight"
DEFAULT_OUT = "reports/condense/subbit2_admm.json"
KILL_LINE = "KILL: NanoQuant is a low-rank resurrection"
OVERFIT_RATIO = 1.5                   # heldout/fit ratio above which the lever is the dead ASVD signature


def log(m):
    print(m, file=sys.stderr); sys.stderr.flush()


def _device():
    dev = os.environ.get("DOCTOR_DEVICE")
    if dev:
        return dev
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _dtype():
    return getattr(torch, os.environ.get("DOCTOR_DTYPE", "float32"))


# ----------------------------------------------------------------------------------------------
# effective-bpw accounting (side-info included; NEVER nominal)
# ----------------------------------------------------------------------------------------------
def eff_bpw_lowrank(rows, cols, rank, scale_bits=SCALE_BITS):
    """Binary low-rank W ~= B1 @ diag(a) @ B2: r*(rows+cols) sign bits + r per-COMPONENT fp32
    scales (the a diagonal), over rows*cols weights. Both are side-info and counted (never
    nominal)."""
    return (rank * (rows + cols) * 1 + rank * scale_bits) / (rows * cols)


def eff_bpw_residual(rows, cols, r1, r2, scale_bits=SCALE_BITS):
    """Plain 2-pass binary residual: ranks r1 + r2 of sign bits plus (r1+r2) per-component fp32
    scales. Side-info fully counted. Same per-component scale cost as the low-rank probe, so the
    two are compared at matched total storage (we never hide the scale cost to flatter either)."""
    return ((r1 + r2) * (rows + cols) * 1 + (r1 + r2) * scale_bits) / (rows * cols)


def residual_ranks_for_bpw(rows, cols, target_bpw, scale_bits=SCALE_BITS):
    """Pick r1,r2 for the 2-pass residual so its eff_bpw <= the low-rank probe's target (a fair,
    if anything generous-to-low-rank, match: the residual gets no MORE storage). Per-component
    sign+scale cost is (rows+cols)+scale_bits bits; the budget buys total rank, split across two
    passes."""
    per_comp_bits = (rows + cols) + scale_bits
    total_rank = max(2, int((target_bpw * rows * cols) // per_comp_bits))
    r1 = max(1, total_rank // 2)
    r2 = max(1, total_rank - r1)
    return r1, r2


# ----------------------------------------------------------------------------------------------
# functional (output-space) error: || (W - W_hat) X^T ||_F / || W X^T ||_F
# ----------------------------------------------------------------------------------------------
def func_err(W, W_hat, X):
    """X is (n_samples, cols). Returns relative Frobenius error of the layer's output."""
    num = torch.linalg.norm((W - W_hat) @ X.T)
    den = torch.linalg.norm(W @ X.T)
    return float(num / den) if float(den) > 0 else float("inf")


# ----------------------------------------------------------------------------------------------
# ADMM binary low-rank: W ~= B1 @ diag(a) @ B2,  B1,B2 in {-1,+1},  a = per-component scale
# ----------------------------------------------------------------------------------------------
def _joint_scale_ls(W, B1, B2):
    """Solve a = argmin ||W - B1 diag(a) B2||_F over the per-component scales (a linear LS whose
    design columns are the rank-1 binary outer products vec(b1_k outer b2_k))."""
    rank = B1.shape[1]
    design = torch.stack([torch.outer(B1[:, k], B2[k, :]).reshape(-1) for k in range(rank)], dim=1)
    sol = torch.linalg.lstsq(design, W.reshape(-1)).solution
    return sol


def admm_binary_lowrank(W, rank, iters=40, rho=1.0, polish=3):
    """Solve W ~= B1 @ diag(a) @ B2 with binary factors, scale-LS, and an ADMM consensus polish.

    A single global (or per-row) scalar CANNOT fit a binary product (its entries grow with rank);
    the working NanoQuant form gives each rank component its own scale a_k. Two stages:

      STAGE 1 - greedy rank-1 binary pursuit (the binarize-via-sign step). For each component,
        alternate u <- sign(R v), v <- sign(R^T u) on the residual R (this IS the discrete factor
        update: sign() is the projection onto {-1,+1}); take the LS scalar a_k = <R, u v^T>/||u v^T||^2;
        subtract a_k u v^T from R; repeat for r components. Monotone, stable.

      STAGE 2 - ADMM consensus polish (the dual-coupled refinement). Treat the joint scales `a` as
        the consensus variable and the per-component signs as the local variables. Each polish sweep:
          (1) a <- joint least-squares over all components            (scale least-squares)
          (2) re-binarize each (u_k, v_k) against its residual target  (sign projection)
          (3) dual feedback: damp the change by rho so the sweep is a contraction (bounded dual;
              an unclamped per-vector dual diverges for rank-1 LS, so the consensus form is used).
      `iters` drives stage-1 inner alternations; `polish` drives stage-2 sweeps.
    Returns (W_hat, a, B1, B2) where a is the length-rank scale vector (side-info, counted)."""
    rows, cols = W.shape
    R = W.clone()
    a = torch.zeros(rank, dtype=W.dtype, device=W.device)
    B1 = torch.empty(rows, rank, dtype=W.dtype, device=W.device)
    B2 = torch.empty(rank, cols, dtype=W.dtype, device=W.device)
    inner = max(4, iters)
    # ---- STAGE 1: greedy rank-1 binary pursuit ----
    for k in range(rank):
        try:                                        # SVD-seed the sign vectors (faster convergence)
            U, S, Vh = torch.linalg.svd(R, full_matrices=False)
            u = torch.sign(U[:, 0]); v = torch.sign(Vh[0])
        except Exception:
            u = torch.sign(R.sum(1)); v = torch.sign(R.sum(0))
        u[u == 0] = 1.0; v[v == 0] = 1.0
        for _ in range(inner):
            u = torch.sign(R @ v); u[u == 0] = 1.0
            v = torch.sign(R.T @ u); v[v == 0] = 1.0
        ok = torch.outer(u, v)
        denom = float((ok * ok).sum())
        sc = float((R * ok).sum() / denom) if denom > 0 else 0.0
        a[k] = sc; B1[:, k] = u; B2[k, :] = v
        R = R - sc * ok
    # ---- STAGE 2: ADMM consensus polish (bounded dual via rho-damping) ----
    for _ in range(max(0, polish)):
        a = _joint_scale_ls(W, B1, B2)
        Wr = B1 @ torch.diag(a) @ B2
        for k in range(rank):
            ok = torch.outer(B1[:, k], B2[k, :])
            Rk = W - (Wr - a[k] * ok)               # residual seen by component k
            u_new = torch.sign(Rk @ B2[k, :]); u_new[u_new == 0] = 1.0
            v_new = torch.sign(Rk.T @ u_new); v_new[v_new == 0] = 1.0
            # rho-damped consensus update: only flip toward the new sign (contraction, bounded)
            if rho >= 1.0:
                B1[:, k] = u_new; B2[k, :] = v_new
            else:
                B1[:, k] = torch.where(torch.rand_like(u_new) < rho, u_new, B1[:, k])
                B2[k, :] = torch.where(torch.rand_like(v_new) < rho, v_new, B2[k, :])
    a = _joint_scale_ls(W, B1, B2)
    W_hat = B1 @ torch.diag(a) @ B2
    return W_hat, a, B1, B2


# ----------------------------------------------------------------------------------------------
# baseline: plain 2-pass binary residual, truncated to the SAME sign-bit budget (matched eff bpw)
# ----------------------------------------------------------------------------------------------
def plain_residual_matched(W, r1, r2, iters=40, rho=1.0):
    """The honest baseline at MATCHED (or smaller) storage. Plain residual b1+b2 quant is the lever
    that low-rank must beat (the kill ledger entry). It is two binary-low-rank passes: factorize W
    (rank r1), then factorize the residual W - W_hat1 (rank r2). r1,r2 are chosen by
    residual_ranks_for_bpw so its eff_bpw <= the low-rank probe's. Unlike the probe, it does NOT
    whiten by calibration activations — it is data-FREE, the property the kill ledger says is what
    makes residual robust. If the data-aware single-pass low-rank cannot beat this, the lever is
    dead."""
    What1, _, _, _ = admm_binary_lowrank(W, r1, iters=iters, rho=rho)
    What2, _, _, _ = admm_binary_lowrank(W - What1, r2, iters=iters, rho=rho)
    return What1 + What2


# ----------------------------------------------------------------------------------------------
# activation capture (real model) and synthetic fallback
# ----------------------------------------------------------------------------------------------
def capture_activations(model_dir, tensor_name, calib_path, ctx, dev, dtype):
    """Forward the calib corpus once, hook the linear whose .weight == tensor_name, collect its
    INPUT activations (rows of x, shape cols). Returns (W, X) with X = (n_samples, cols)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch.nn as nn
    if tensor_name.endswith(".weight"):
        mod_name = tensor_name[:-len(".weight")]
    else:
        mod_name = tensor_name
    log(f"# loading {model_dir} on {dev}/{dtype} to capture activations for {mod_name}")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype, attn_implementation="eager").to(dev).eval()
    modules = dict(model.named_modules())
    target = modules.get(mod_name)
    if not isinstance(target, nn.Linear):
        raise RuntimeError(f"{mod_name} is not an nn.Linear (got {type(target).__name__})")
    W = target.weight.detach().to("cpu", torch.float32).clone()
    bucket = []

    def hook(m, inp, out):
        x = inp[0].detach().reshape(-1, inp[0].shape[-1]).to("cpu", torch.float32)
        bucket.append(x)

    h = target.register_forward_hook(hook)
    text = open(calib_path, errors="ignore").read() if os.path.exists(calib_path) else ""
    if not text:
        raise FileNotFoundError(f"calib corpus not found: {calib_path}")
    ids = tok(text, return_tensors="pt").input_ids[:, :ctx].to(dev)
    with torch.no_grad():
        model(ids)
    h.remove()
    X = torch.cat(bucket, dim=0)
    del model
    if dev == "mps":
        torch.mps.empty_cache()
    log(f"# captured W{tuple(W.shape)}  X{tuple(X.shape)}")
    return W, X


def synthetic_problem(rows=256, cols=512, true_rank=8, n_act=2048, noise=0.05, seed=0,
                      cov_shift=False):
    """A structured low-rank + noise matrix and matched activations — the case that SHOULD be
    recoverable. Used when no model is present (--dry) and for --self-test.

    cov_shift=True builds X so its FIRST half and SECOND half have DIFFERENT dominant input
    channels (a calibration->held distribution shift). A data-aware (whitened) fit then overfits
    the first half's directions and its held-out functional error blows up — the exact ASVD
    overfit signature the overfit-ratio kill is meant to catch."""
    g = torch.Generator().manual_seed(seed)
    L = torch.randn(rows, true_rank, generator=g)
    R = torch.randn(true_rank, cols, generator=g)
    W = (L @ R) / math.sqrt(true_rank)
    W = W + noise * torch.randn(rows, cols, generator=g)
    if not cov_shift:
        X = torch.randn(n_act, cols, generator=g)
    else:
        half = n_act // 2
        X = torch.randn(n_act, cols, generator=g) * 0.1
        ndom = max(2, cols // 64)
        idx = torch.randperm(cols, generator=g)
        dom_a, dom_b = idx[:ndom], idx[ndom:2 * ndom]     # disjoint dominant channels per half
        X[:half, dom_a] += torch.randn(half, ndom, generator=g) * 12.0
        X[half:, dom_b] += torch.randn(n_act - half, ndom, generator=g) * 12.0
    return W.float(), X.float()


# ----------------------------------------------------------------------------------------------
# core measurement: fit ADMM on FIT split, score on FIT + HELD-OUT, compare to matched residual
# ----------------------------------------------------------------------------------------------
def run_probe(W, X, rank, iters, fit_frac=0.5, shuffle=True):
    """Split activations into FIT/HELD-OUT, fit the (data-aware) low-rank on FIT, score on both.
    The ASVD failure mode is that a calibration-aware fit overfits the FIT activations' directions;
    we expose it by scoring functional error on the split the fit was tuned against vs a disjoint
    held-out split. shuffle=True (real-model path) removes token-position bias; shuffle=False keeps
    a deliberately distribution-shifted split intact (used by the self-test)."""
    rows, cols = W.shape
    n = X.shape[0]
    if shuffle:
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(1))
        X = X[perm]
    cut = max(1, int(n * fit_frac))
    X_fit, X_held = X[:cut], X[cut:]
    if X_held.shape[0] == 0:
        X_held = X_fit

    # Low-rank probe: fit factors weighted toward the FIT activations (data-aware, the ASVD shape).
    # We whiten W by the FIT-activation second moment so the factorization is calibration-aware —
    # this is exactly the post-hoc-low-rank move the kill ledger says overfits.
    cov = (X_fit.T @ X_fit) / X_fit.shape[0] + 1e-3 * torch.eye(cols)
    d = torch.sqrt(torch.clamp(torch.diagonal(cov), min=1e-8))     # per-input-channel importance
    W_w = W * d.unsqueeze(0)                                       # weight columns by FIT importance
    What_w, a_scales, B1, B2 = admm_binary_lowrank(W_w, rank, iters=iters)
    What_admm = What_w / d.unsqueeze(0)                            # un-whiten back to weight space

    admm_fit = func_err(W, What_admm, X_fit)
    admm_held = func_err(W, What_admm, X_held)

    bpw_admm = eff_bpw_lowrank(rows, cols, rank)
    # Matched-bpw plain residual baseline (no activation whitening — it is NOT data-aware). Its
    # ranks are sized so its eff_bpw <= the probe's; it never gets more storage than the lever.
    r1, r2 = residual_ranks_for_bpw(rows, cols, bpw_admm)
    What_res = plain_residual_matched(W, r1, r2, iters=iters)
    res_fit = func_err(W, What_res, X_fit)
    res_held = func_err(W, What_res, X_held)
    bpw_res = eff_bpw_residual(rows, cols, r1, r2)

    return {
        "rows": rows, "cols": cols, "rank": rank, "iters": iters,
        "residual_r1": r1, "residual_r2": r2,
        "n_act_fit": int(X_fit.shape[0]), "n_act_held": int(X_held.shape[0]),
        "scale_abs_mean": round(float(a_scales.abs().mean()), 6), "scale_is_per_component": True,
        "eff_bpw_admm": round(bpw_admm, 5),
        "eff_bpw_residual": round(bpw_res, 5),
        "eff_bpw_matched": bpw_res <= bpw_admm + 1e-6,   # residual gets no MORE storage than probe
        "admm_fit_err": round(admm_fit, 5),
        "admm_heldout_err": round(admm_held, 5),
        "residual_fit_err": round(res_fit, 5),
        "residual_heldout_err": round(res_held, 5),
        "heldout_over_fit_ratio": round(admm_held / admm_fit, 4) if admm_fit > 0 else float("inf"),
    }


def verdict(rec):
    """Apply the hard KILL criteria. Returns (killed: bool, reasons: list[str])."""
    reasons = []
    ratio = rec["heldout_over_fit_ratio"]
    if ratio > OVERFIT_RATIO:
        reasons.append(f"overfit: heldout/fit = {ratio:.2f} > {OVERFIT_RATIO} (dead ASVD signature)")
    if rec["admm_heldout_err"] >= rec["residual_heldout_err"]:
        reasons.append(
            f"loses to residual at matched bpw: admm_heldout {rec['admm_heldout_err']:.4f} "
            f">= residual_heldout {rec['residual_heldout_err']:.4f}")
    return (len(reasons) > 0), reasons


# ----------------------------------------------------------------------------------------------
# self-test: run on synthetic, confirm the kill logic FIRES correctly on a forced-overfit case
# ----------------------------------------------------------------------------------------------
def self_test():
    log("# --self-test: synthetic low-rank+noise; checking kill logic")
    ok = True

    # Case A: genuinely low-rank, generous rank -> probe should generalize (ratio low). The kill
    # may still fire on the residual-comparison leg (that is fine — it is a real, honest result).
    W, X = synthetic_problem(rows=256, cols=512, true_rank=8, n_act=4096, noise=0.02, seed=0)
    recA = run_probe(W, X, rank=16, iters=40, fit_frac=0.5)
    killedA, reasonsA = verdict(recA)
    log(f"#  [A clean low-rank] ratio={recA['heldout_over_fit_ratio']:.3f} "
        f"admm_held={recA['admm_heldout_err']:.4f} res_held={recA['residual_heldout_err']:.4f} "
        f"killed={killedA} reasons={reasonsA}")
    # The functional error should NOT blow up out-of-sample on a truly low-rank matrix.
    if not (recA["heldout_over_fit_ratio"] <= OVERFIT_RATIO):
        log("#  [A] FAIL: clean low-rank tripped the overfit ratio (kill logic too sensitive)")
        ok = False

    # Case B: forced overfit — a distribution SHIFT between the FIT and HELD activation halves
    # (disjoint dominant input channels). The data-aware (whitened) fit latches onto the FIT
    # half's directions, so its held-out functional error blows up => heldout/fit > 1.5 (the dead
    # ASVD signature). shuffle=False preserves the deliberate first-half/second-half split.
    Wb, Xb = synthetic_problem(rows=256, cols=512, true_rank=200, n_act=512, noise=0.8, seed=3,
                               cov_shift=True)
    recB = run_probe(Wb, Xb, rank=64, iters=20, fit_frac=0.5, shuffle=False)
    killedB, reasonsB = verdict(recB)
    log(f"#  [B forced overfit] ratio={recB['heldout_over_fit_ratio']:.3f} "
        f"admm_held={recB['admm_heldout_err']:.4f} res_held={recB['residual_heldout_err']:.4f} "
        f"killed={killedB} reasons={reasonsB}")
    if not killedB:
        log("#  [B] FAIL: forced-overfit case did NOT trip the kill (kill logic too lax)")
        ok = False

    log(f"# self-test {'PASS' if ok else 'FAIL'}: kill logic "
        f"{'fires correctly' if ok else 'is MISCALIBRATED'}")
    return 0 if ok else 1


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="SUBBIT-2 probe: binary low-rank ADMM (W~=s*B1@B2) vs matched-bpw residual on "
                    "HELD-OUT functional error. PROBE ONLY — measures, never claims a serving win. "
                    f"{KILL_LINE!r} on overfit (heldout/fit>1.5) or loss to residual.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"HF dir (default {DEFAULT_MODEL})")
    ap.add_argument("--tensor", default=DEFAULT_TENSOR, help=f"weight tensor (default {DEFAULT_TENSOR})")
    ap.add_argument("--calib", default="scratch/calib_corpus.txt", help="calibration corpus file")
    ap.add_argument("--ctx", type=int, default=1024, help="calib tokens to forward")
    ap.add_argument("--rank", type=int, default=32, help="binary low-rank r")
    ap.add_argument("--iters", type=int, default=40, help="ADMM iterations")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"JSON report (default {DEFAULT_OUT})")
    ap.add_argument("--dry", action="store_true",
                    help="force the synthetic problem (no model load) — runs anywhere")
    ap.add_argument("--self-test", action="store_true",
                    help="run synthetic self-test and report whether the kill logic fires correctly")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    dev, dtype = _device(), _dtype()
    log(f"# subbit_admm PROBE — dev={dev} dtype={dtype} STRAND_NO_GPU={os.environ.get('STRAND_NO_GPU')}")
    log(f"# KILL criteria: heldout/fit > {OVERFIT_RATIO}  OR  admm_heldout >= residual_heldout (matched bpw)")

    used_synthetic = False
    model_present = os.path.exists(os.path.join(args.model, "model.safetensors")) or \
        os.path.exists(os.path.join(args.model, "model.safetensors.index.json"))
    if args.dry or not model_present:
        if not args.dry:
            log(f"# model {args.model} absent — GATED: falling back to synthetic structured matrix")
        W, X = synthetic_problem()
        used_synthetic = True
        source = "synthetic(low-rank+noise)"
    else:
        W, X = capture_activations(args.model, args.tensor, args.calib, args.ctx, dev, dtype)
        source = f"{args.model}::{args.tensor}"

    rec = run_probe(W, X, rank=args.rank, iters=args.iters)
    rec["source"] = source
    rec["synthetic"] = used_synthetic
    rec["probe"] = True
    rec["disclaimer"] = ("PROBE: measures held-out functional error of binary low-rank vs "
                         "matched-bpw residual. NOT a serving win and emits no deployable artifact.")
    rec["kill_criteria"] = {"overfit_ratio_gt": OVERFIT_RATIO,
                            "or_admm_heldout_ge_residual_heldout": True}

    killed, reasons = verdict(rec)
    rec["killed"] = killed
    rec["kill_reasons"] = reasons

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(json.dumps(rec, indent=2) + "\n")
    log(f"# wrote {args.out}")
    log(f"# eff_bpw(admm)={rec['eff_bpw_admm']:.4f} eff_bpw(residual)={rec['eff_bpw_residual']:.4f} "
        f"matched={rec['eff_bpw_matched']}")
    log(f"# admm:     fit_err={rec['admm_fit_err']:.4f}  heldout_err={rec['admm_heldout_err']:.4f}  "
        f"ratio={rec['heldout_over_fit_ratio']:.3f}")
    log(f"# residual: fit_err={rec['residual_fit_err']:.4f}  heldout_err={rec['residual_heldout_err']:.4f}")

    if killed:
        for r in reasons:
            log(f"#   reason: {r}")
        print(KILL_LINE)
        sys.exit(1)
    log("# SURVIVES (probe only): generalizes AND beats matched-bpw residual on held-out. "
        "Still NOT a serving win — schedule independent reproduction before any claim.")
    sys.exit(0)


if __name__ == "__main__":
    main()
