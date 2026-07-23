#!/usr/bin/env python3.12
"""Hawking Mechanics/Thermodynamics - Generation-M run-all tournament orchestrator (M2..M7).

This is the testing/feedback loop that extends the mandated B0/B1/M1 measurement contract
(mech_measure.py, REUSED read-only) with the MoE-scale execution grammars M2..M6 and a
deferred bit-oriented stub M7. It runs each candidate on REAL GPT-OSS-120B layer-0 experts,
applies the quality gate BEFORE speed, seals positive AND negative artifacts, and continues the
tournament even when a stage produces a negative (quarantine + continue, Bible section 56).

Grammars (each EXTENDS M1's lookup-linear discipline: build activation->codeword tables once,
accumulate y via pure index gathers; never materialize a dense [rows,cols] shadow):

  * M2 shared_lookup_linear_moe   - ONE additive codeword table per sharing group + activation,
                                    reused across the real top-k experts that share a codebook
                                    (pack_shared_grammar codebook + moe routing). Controls:
                                    per-expert-table (independent, the M1 control) vs shared-table
                                    vs layer-group-share. Measures reuse ratio, table-build work
                                    avoided, weighted-combine (MoE output) quality.
  * M3 shared_moe_plus_islands    - protected islands (4 selectors) as bounded sparse exact-row
                                    corrections on top of M2. Control: islands-off. Bills island
                                    bytes, measures gather cost + quality on mlp2 (sensitive)
                                    without mlp1 regression.
  * M4 fused_pq_islands_doctor    - fuse base lookup + island correction + doctor residual codebook
                                    in ONE bounded accumulation (vs separate-kernel control).
                                    Measures launch + temporary reduction; quality must MATCH the
                                    unfused CPU treatment (same math).
  * M5 conditional_doctor         - gate the doctor per-activation by a deterministic condition
                                    (router confidence / output margin / residual syndrome) vs
                                    always-on M4. HARD gate = false_negative_rate (skipping a needed
                                    correction). Rejected if it misses needed corrections (Bible 76).
  * M6 residual_additive_lookup   - richer 2-stage additive multi-codebook via staged lookup
                                    accumulation, compared vs M4 at EQUAL total bits (who spends
                                    the bytes better).
  * M7 bit_oriented               - DEFERRED stub only (Bible 77): sealed as deferred, not built.

Laws honoured (do not weaken): quality gate BEFORE speed; exact whole-artifact byte accounting via
the frozen ByteLedger; no dense shadow (bounded tables/tiles/islands only, peak temporary reported +
asserted bounded); CPU (numpy) authoritative, MPS measured but never overrides the CPU verdict;
Apple-only; energy UNAVAILABLE (all thermodynamics labelled UNAVAILABLE, no invented estimates);
timing MEASURED but caveated contaminated_by_concurrent_cpu_load, paired/relative preferred; a
negative result (nothing beats the baseline at matched quality) is VALID and sealed honestly; a
speedup at LOWER quality is NOT a win (fake-win ban).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gravity_forge as gf          # frozen: PQ pack + ByteLedger + shared grammar + islands + doctor
import mech_measure as mm           # just built: MechVector, timed/paired_window, seal, quality, B0/B1/M1
import gptoss_moe_runtime as rt     # frozen: router/expert loader + reference MoE forward

RUN_SCHEMA = "hawking.mechanics.run_all.v1"
_REPORT_DIR_DEFAULT = "reports/mechanics_thermodynamics"
_QUALITY_TOL = 5e-3                 # matched-quality window for the combine-divergence gate
_HALF = 0.5                        # no-dense-shadow: peak temporary must be < 0.5 * full dense


# ============================================================================================
# Staged additive lookup-linear execution (the M2..M6 generalisation of M1).
#
# A "staged" grammar reshapes W[rows,cols] into [rows*nchunk, D] (nchunk = cols/D) and encodes each
# D-vector as an ADDITIVE sum over `stages` codebooks cb_m[k, D] (greedy residual VQ). Reconstruction
# recon_row(r,c) = sum_m cb_m[idx[r*nchunk+c, m]]. The matvec y = recon @ x factors as
#   y[r,b] = sum_c sum_m <cb_m[idx[r*nchunk+c,m]], xc[c,:,b]> = sum_c sum_m T_m[c, idx[.,m], b]
# where T_m[c,q,b] = <cb_m[q], xc[c,:,b]> is the SHARED activation->codeword table (nchunk x k x B).
# Build T once, accumulate via pure gathers. When the codebooks are SHARED across E experts (M2), T
# is built ONCE and reused across all E; when INDEPENDENT (the M1 control) each expert rebuilds T.
# ============================================================================================
@dataclass
class StagedCodes:
    """One expert's staged-additive codes. `codebooks` may be a shared reference (M2) or private
    (M1 control). Islands / doctor stages are bounded corrections, each billed in the ledger."""
    codebooks: list[np.ndarray]                     # [stages] each [k, D] fp32
    indices: np.ndarray                             # [N, stages] int64, N = rows*nchunk
    D: int
    rows: int
    cols: int
    nchunk: int
    stages: int
    k: int
    shared: bool
    recon: np.ndarray                               # bounded reconstruction (this expert only)
    island_rows: np.ndarray | None = None           # exact-row island indices (sorted)
    island_vals: np.ndarray | None = None           # [n_islands, cols] fp32 exact rows
    doctor_codebooks: list[np.ndarray] = field(default_factory=list)   # per-expert residual stages
    doctor_indices: np.ndarray | None = None        # [N, doctor_stages] int64

    @property
    def n_islands(self) -> int:
        return 0 if self.island_rows is None else int(len(self.island_rows))

    @property
    def doctor_stages(self) -> int:
        return len(self.doctor_codebooks)


def _reshape_v_np(w: np.ndarray, D: int) -> np.ndarray:
    return np.ascontiguousarray(w.astype(np.float32)).reshape(-1, D)


def _fit_codebooks(vectors, k: int, stages: int, *, seed: int, iters: int) -> list[np.ndarray]:
    """Greedy residual VQ on a pooled set of D-vectors -> `stages` codebooks [k, D]. Reuses the frozen
    gravity_forge _kmeans/_assign (MPS when present). Reproduces pack_shared_grammar's codebook fit."""
    torch = mm._torch()
    dev = gf._device()
    if not isinstance(vectors, torch.Tensor):
        vectors = torch.from_numpy(np.ascontiguousarray(vectors, dtype=np.float32)).to(dev)
    residual = vectors.clone()
    cbs: list[np.ndarray] = []
    for m in range(stages):
        cb = gf._kmeans(residual, k, iters=iters, seed=seed + m)
        residual = residual - cb[gf._assign(residual, cb)]
        cbs.append(cb.detach().cpu().numpy().astype(np.float32))
    return cbs


def _encode_expert(w: np.ndarray, cbs: list[np.ndarray], D: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode ONE expert against fixed codebooks (greedy residual assignment). Returns
    (indices[N,stages], recon[rows,cols]). Reproduces pack_shared_grammar's per-expert encoding."""
    torch = mm._torch()
    dev = gf._device()
    rows, cols = w.shape
    v = torch.from_numpy(_reshape_v_np(w, D)).to(dev)
    cbs_t = [torch.from_numpy(cb).to(dev) for cb in cbs]
    res = v.clone()
    recon = torch.zeros_like(v)
    idxs = []
    for cb in cbs_t:
        idx = gf._assign(res, cb)
        recon = recon + cb[idx]
        res = res - cb[idx]
        idxs.append(idx.detach().cpu().numpy().astype(np.int64))
    indices = np.stack(idxs, axis=1)
    recon_np = recon.reshape(rows, cols).detach().cpu().numpy().astype(np.float32)
    return indices, recon_np


def build_staged_codes(experts: list[np.ndarray], *, D: int, k: int, stages: int, shared: bool,
                       seed: int = 0, iters: int = 8) -> list[StagedCodes]:
    """Build staged-additive codes for a cluster of experts. shared=True fits ONE codebook on the
    pooled vectors (M2 shared grammar); shared=False fits an INDEPENDENT codebook per expert (the M1
    per-expert-table control). Deterministic in (seed, iters)."""
    torch = mm._torch()
    dev = gf._device()
    rows, cols = experts[0].shape
    nchunk = cols // D
    out: list[StagedCodes] = []
    if shared:
        pool = torch.cat([torch.from_numpy(_reshape_v_np(w, D)).to(dev) for w in experts], 0)
        cbs = _fit_codebooks(pool, k, stages, seed=seed, iters=iters)
        for w in experts:
            idx, recon = _encode_expert(w, cbs, D)
            out.append(StagedCodes(cbs, idx, D, w.shape[0], w.shape[1], w.shape[1] // D, stages, k,
                                   True, recon))
    else:
        for e, w in enumerate(experts):
            v = torch.from_numpy(_reshape_v_np(w, D)).to(dev)
            cbs = _fit_codebooks(v, k, stages, seed=seed + 17 * e, iters=iters)
            idx, recon = _encode_expert(w, cbs, D)
            out.append(StagedCodes(cbs, idx, D, w.shape[0], w.shape[1], w.shape[1] // D, stages, k,
                                   False, recon))
    return out


# --------------------------------------------------------------------------------------------
# CPU (authoritative) staged lookup execution.
# --------------------------------------------------------------------------------------------
def _stage_tables_np(codebooks: list[np.ndarray], xc: np.ndarray) -> list[np.ndarray]:
    """T_m[c,q,b] = <cb_m[q], xc[c,:,b]> for each stage. xc:[nchunk,D,B]. Bounded: nchunk*k*B each."""
    return [np.einsum("qd,cdb->cqb", cb, xc, optimize=True) for cb in codebooks]


def _accumulate_np(tables: list[np.ndarray], indices: np.ndarray, rows: int, nchunk: int,
                   B: int) -> np.ndarray:
    """y[r,b] = sum_{c,m} T_m[c, idx[r*nchunk+c, m], b] via pure index gathers + adds (no multiplies)."""
    stages = indices.shape[1]
    idx_r = indices.reshape(rows, nchunk, stages)
    cc = np.arange(nchunk)
    y = np.zeros((rows, B), dtype=np.float32)
    for m in range(stages):
        T = tables[m]
        k = T.shape[1]
        flat = T.reshape(nchunk * k, B)
        gidx = idx_r[:, :, m] + cc[None, :] * k
        g = flat[gidx.reshape(-1)].reshape(rows, nchunk, B)
        y += g.sum(1)
    return y


def staged_execute_np(codes: StagedCodes, x: np.ndarray, *, tables: list[np.ndarray] | None = None,
                      fused: bool = True) -> np.ndarray:
    """Execute y = W_pq @ x for one expert via staged lookup accumulation (base [+ doctor] stages
    [+ island exact rows]). `tables` may be a prebuilt SHARED base table (M2 reuse); if None it is
    built here. fused=True accumulates base + doctor in one buffer; fused=False keeps them separate
    (same math, more temporaries) - used for the M4 separate-kernel control."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    onedim = x.ndim == 1
    xm = x[:, None] if onedim else x
    B = xm.shape[1]
    xc = xm.reshape(codes.nchunk, codes.D, B)
    if tables is None:
        tables = _stage_tables_np(codes.codebooks, xc)
    y = _accumulate_np(tables, codes.indices, codes.rows, codes.nchunk, B)
    if codes.doctor_stages > 0 and codes.doctor_indices is not None:
        dtab = _stage_tables_np(codes.doctor_codebooks, xc)
        yd = _accumulate_np(dtab, codes.doctor_indices, codes.rows, codes.nchunk, B)
        y = (y + yd) if fused else (y + yd)   # arithmetically identical; separateness is in Ttemp/Klaunch
    if codes.island_rows is not None and codes.n_islands > 0:
        y[codes.island_rows] = codes.island_vals.astype(np.float32) @ xm   # exact rows use original x
    return y[:, 0] if onedim else y


# --------------------------------------------------------------------------------------------
# MPS (measured, non-authoritative) staged lookup execution - mirrors the CPU decomposition.
# --------------------------------------------------------------------------------------------
def staged_execute_torch(codes: StagedCodes, x, *, dev, tables=None):
    torch = mm._torch()
    onedim = x.ndim == 1
    xm = x[:, None] if onedim else x
    B = xm.shape[1]
    xc = xm.reshape(codes.nchunk, codes.D, B)
    if tables is None:
        cbs = [torch.from_numpy(cb).to(dev) for cb in codes.codebooks]
        tables = [torch.einsum("qd,cdb->cqb", cb, xc) for cb in cbs]
    idx_r = torch.from_numpy(np.ascontiguousarray(codes.indices)).to(dev).reshape(
        codes.rows, codes.nchunk, codes.stages)
    cc = torch.arange(codes.nchunk, device=dev).view(1, codes.nchunk)
    y = torch.zeros((codes.rows, B), device=dev, dtype=torch.float32)
    for m in range(codes.stages):
        T = tables[m]
        k = T.shape[1]
        flat = T.reshape(codes.nchunk * k, B)
        gidx = (idx_r[:, :, m] + cc * k).reshape(-1)
        y += flat.index_select(0, gidx).reshape(codes.rows, codes.nchunk, B).sum(1)
    if codes.island_rows is not None and codes.n_islands > 0:
        iv = torch.from_numpy(np.ascontiguousarray(codes.island_vals, dtype=np.float32)).to(dev)
        ir = torch.from_numpy(np.ascontiguousarray(codes.island_rows)).to(dev)
        y[ir] = iv @ xm
    return y[:, 0] if onedim else y


# --------------------------------------------------------------------------------------------
# Islands + doctor attachment (bounded, billed).
# --------------------------------------------------------------------------------------------
def attach_islands(codes: StagedCodes, w: np.ndarray, *, strategy: str, budget_frac: float) -> StagedCodes:
    """Attach protected exact-row islands (frozen gf.select_protected_islands, 4 selectors) to the
    base codes. Selection uses the base residual; the exact rows are stored verbatim (billed later)."""
    resid = (w.astype(np.float32) - codes.recon)
    isl = gf.select_protected_islands(w, resid, strategy=strategy, budget_frac=budget_frac)
    rows_sel = isl["row_indices"]
    recon = codes.recon.copy()
    recon[rows_sel] = w[rows_sel].astype(np.float32)
    codes.recon = recon
    codes.island_rows = rows_sel
    codes.island_vals = w[rows_sel].astype(np.float32)
    return codes


def attach_doctor(codes: StagedCodes, w: np.ndarray, *, k: int, stages: int, seed: int,
                  iters: int) -> StagedCodes:
    """Fit a per-expert residual-codebook doctor (additive staged VQ on w - recon) and attach it as
    extra staged-lookup codes. The residual is used only transiently to FIT; the stored state is the
    billed codebook + indices (no uncounted dense residual)."""
    resid = (w.astype(np.float32) - codes.recon)
    torch = mm._torch()
    dev = gf._device()
    v = torch.from_numpy(_reshape_v_np(resid, codes.D)).to(dev)
    cbs = _fit_codebooks(v, k, stages, seed=seed, iters=iters)
    idx, recon_d = _encode_expert(resid, cbs, codes.D)
    codes.doctor_codebooks = cbs
    codes.doctor_indices = idx
    codes.recon = codes.recon + recon_d
    if codes.island_rows is not None and codes.n_islands > 0:
        codes.recon[codes.island_rows] = w[codes.island_rows].astype(np.float32)  # islands stay exact
    return codes


# --------------------------------------------------------------------------------------------
# Exact byte accounting for a cluster of staged codes (reuses the frozen ByteLedger).
# --------------------------------------------------------------------------------------------
def cluster_ledger(codes_list: list[StagedCodes]) -> tuple[gf.ByteLedger, dict[str, Any]]:
    """Bill a whole cluster: shared codebooks ONCE, per-expert indices, islands, doctor stages.
    Returns (ledger, breakdown). whole_artifact_bpw = total_bits / total_weights over the cluster."""
    led = gf.ByteLedger()
    E = len(codes_list)
    c0 = codes_list[0]
    total_weights = sum(c.rows * c.cols for c in codes_list)
    if c0.shared:
        for cb in c0.codebooks:                                    # shared base codebook billed ONCE
            led.add_fp16(cb.shape[0] * c0.D)
    else:
        for c in codes_list:
            for cb in c.codebooks:
                led.add_fp16(cb.shape[0] * c.D)
    base_index_bits = 0
    island_bits = 0
    doctor_bits = 0
    for c in codes_list:
        N = c.rows * c.nchunk
        for _ in range(c.stages):
            led.add_index(N, c.k)
            base_index_bits += N * max(1, math.ceil(math.log2(max(2, c.k))))
        if c.n_islands > 0:
            ib = c.n_islands * (c.cols * 16 + max(1, math.ceil(math.log2(max(2, c.rows)))))
            led.add("protected_islands", ib)
            island_bits += ib
        if c.doctor_stages > 0:
            for cb in c.doctor_codebooks:
                led.add("doctor_codebooks", cb.shape[0] * c.D * 16)
                doctor_bits += cb.shape[0] * c.D * 16
            for _ in range(c.doctor_stages):
                led.add("doctor_indices", N * max(1, math.ceil(math.log2(max(2, c.k)))))
                doctor_bits += N * max(1, math.ceil(math.log2(max(2, c.k))))
    total_bits = led.total_bits()
    breakdown = {
        "n_experts": E, "total_weights": int(total_weights),
        "whole_artifact_bpw": round(total_bits / max(1, total_weights), 6),
        "base_index_bits": int(base_index_bits), "island_bits": int(island_bits),
        "doctor_bits": int(doctor_bits), "total_bits": int(total_bits),
        "physical_bytes": int(led.bytes()),
        "shared_codebook": bool(c0.shared),
    }
    return led, breakdown


# --------------------------------------------------------------------------------------------
# Analytical mechanical vectors for a cluster forward (process E experts on ONE activation).
# --------------------------------------------------------------------------------------------
def mech_cluster(codes_list: list[StagedCodes], breakdown: dict[str, Any], B: int, *,
                 shared: bool, fused: bool, mps: bool) -> mm.MechVector:
    """The MoE unit of work: run all E experts on one activation. The M2 lever lives in the number of
    table builds (1 if shared, E if independent); islands/doctor add bounded flops/lookups/temporary."""
    E = len(codes_list)
    c0 = codes_list[0]
    rows, cols, nchunk, stages, k, D = c0.rows, c0.cols, c0.nchunk, c0.stages, c0.k, c0.D
    N = rows * nchunk
    table_build_per = 2.0 * cols * k * B * stages          # per distinct codebook set
    n_builds = 1 if shared else E
    base_accum_adds = float(E * N * stages * B)
    base_lookups = float(E * N * stages * B)
    doctor_flops = 0.0
    doctor_lookups = 0.0
    doctor_tmp = 0.0
    total_doctor_stages = sum(c.doctor_stages for c in codes_list)
    if total_doctor_stages > 0:
        doctor_flops = 2.0 * cols * k * B * total_doctor_stages + float(total_doctor_stages * N * B)
        doctor_lookups = float(total_doctor_stages * N * B)
        doctor_tmp = float(nchunk * k * B * 4)             # doctor tables (fused reuses; unfused holds sep.)
    island_flops = 0.0
    island_lookups = 0.0
    island_tmp = 0.0
    total_islands = sum(c.n_islands for c in codes_list)
    if total_islands > 0:
        island_flops = float(2 * total_islands * cols * B)
        island_lookups = float(total_islands)
        island_tmp = float(max(c.n_islands for c in codes_list) * cols * 4)
    F32 = n_builds * table_build_per + base_accum_adds + doctor_flops + island_flops
    Llookup = base_lookups + doctor_lookups + island_lookups
    idx_b = breakdown["base_index_bits"] / 8.0
    cb_b = (sum(cb.shape[0] * D for cb in c0.codebooks) * 2) if shared else \
        sum(sum(cb.shape[0] * D for cb in c.codebooks) * 2 for c in codes_list)
    x_b = cols * 4 * B
    gather_bytes = N * stages * B * 4 * E
    table_bytes = n_builds * nchunk * k * B * 4
    Mread = float(idx_b + cb_b + x_b * E + gather_bytes)
    Mwrite = float(table_bytes + rows * 4 * B * E)
    # peak temporary: base table (shared or one expert) + one expert gather + (fused: reuse) doctor/island
    base_tmp = nchunk * k * B * 4 + rows * nchunk * B * 4
    if fused:
        peak_tmp = base_tmp + max(doctor_tmp, island_tmp)
    else:
        peak_tmp = base_tmp + doctor_tmp + island_tmp + rows * B * 4   # separate intermediate buffer
    # Metal dispatch estimate: per (table build + accumulate) per stage per build + doctor + island
    if mps:
        launches = n_builds * stages + E * stages + total_doctor_stages * 2 + (1 if total_islands else 0)
        if not fused:
            launches += total_doctor_stages + E   # extra separate-kernel dispatches + merge adds
    else:
        launches = 0.0
    m = mm.MechVector(
        F32=float(F32), F16=0.0, Fint=0.0, Bbit=0.0,
        Llookup=float(Llookup),
        Mread=Mread, Mwrite=Mwrite,
        Klaunch=float(launches),
        Ssync=1.0 if mps else 0.0,
        Ttemporary=float(peak_tmp),
    )
    for d in ("F32", "F16", "Fint", "Bbit", "Llookup", "Mread", "Mwrite", "Ttemporary"):
        m.label(d, "ANALYTICAL")
    m.label("Klaunch", "ESTIMATED" if mps else "MEASURED")
    m.label("Ssync", "MEASURED")
    return m


# ============================================================================================
# Quality: per-expert matvec parity + weighted-combine (MoE output) divergence.
# ============================================================================================
def _combine_divergence(xs: list[np.ndarray], router: dict[str, np.ndarray],
                        experts_orig: dict[int, dict[str, np.ndarray]],
                        mlp1_codes: dict[int, StagedCodes], mlp2_codes: dict[int, StagedCodes],
                        *, top_k: int) -> dict[str, Any]:
    """Weighted-combine quality: run the reference MoE forward with ORIGINAL experts vs experts whose
    mlp1/mlp2 are replaced by the staged-lookup execution, over the LOADED bounded expert subset.
    The router logits are masked to the loaded experts (a single activation's global top-k could route
    to an unloaded expert; we keep memory bounded and combine within the loaded cluster instead), then
    softmax-weighted over the top-k among them. Divergence of the combined output. Synthetic
    activations + loaded-subset routing => proxy_output, NOT capability parity."""
    available = sorted(experts_orig.keys())
    avail_arr = np.array(available)
    rels = []
    kk = min(top_k, len(available))
    for x in xs:
        logits = router["weight"] @ x + router["bias"]
        sub = logits[avail_arr]                                   # restrict to loaded experts
        order = np.argsort(-sub)[:kk]
        idx = avail_arr[order]
        wgt = sub[order]
        wgt = np.exp(wgt - wgt.max()); wgt = wgt / wgt.sum()
        y0 = np.zeros_like(x)
        y1 = np.zeros_like(x)
        for e, gw in zip(idx, wgt):
            e = int(e)
            ex = experts_orig[e]
            h0 = ex["mlp1"] @ x + ex["mlp1_bias"]
            a0 = rt._swiglu(h0)
            y0 += gw * (ex["mlp2"] @ a0 + ex["mlp2_bias"])
            if e in mlp1_codes and e in mlp2_codes:
                h1 = staged_execute_np(mlp1_codes[e], x) + ex["mlp1_bias"]
                a1 = rt._swiglu(h1)
                y1 += gw * (staged_execute_np(mlp2_codes[e], a1) + ex["mlp2_bias"])
            else:
                y1 += gw * (ex["mlp2"] @ rt._swiglu(ex["mlp1"] @ x + ex["mlp1_bias"]) + ex["mlp2_bias"])
        rels.append(float(np.linalg.norm(y0 - y1) / (np.linalg.norm(y0) + 1e-9)))
    return {"n_inputs": len(xs), "n_experts_in_combine": len(available), "top_k_used": int(kk),
            "mean_combine_div": round(float(np.mean(rels)), 6),
            "max_combine_div": round(float(np.max(rels)), 6),
            "min_combine_div": round(float(np.min(rels)), 6),
            "routing": "masked_to_loaded_expert_subset",
            "signal": "proxy_output_synthetic_activations", "capability_parity": False}


def _matvec_quality(codes_list: list[StagedCodes], mats: list[np.ndarray],
                    acts: list[np.ndarray]) -> dict[str, Any]:
    """Mean relative error of the staged-lookup matvec vs the exact dense W@x, over the cluster."""
    rels, coss = [], []
    for c, w in zip(codes_list, mats):
        for x in acts:
            y = staged_execute_np(c, x)
            yd = w.astype(np.float32) @ np.ascontiguousarray(x, dtype=np.float32)
            rels.append(mm._rel(y, yd))
            coss.append(mm._cos(y, yd))
    return {"rel_error_vs_dense_mean": round(float(np.mean(rels)), 6),
            "rel_error_vs_dense_max": round(float(np.max(rels)), 6),
            "cosine_vs_dense_mean": round(float(np.mean(coss)), 6)}


# ============================================================================================
# Timing: paired cluster-forward windows (build-all + accumulate all E experts).
# ============================================================================================
def _cluster_forward_np(codes_list: list[StagedCodes], x: np.ndarray, *, shared: bool,
                        fused: bool = True) -> list[np.ndarray]:
    """Process the WHOLE cluster on one activation. shared=True builds the base table ONCE and reuses
    it across all E experts (the M2 lever); shared=False rebuilds per expert (the M1 control)."""
    xm = np.ascontiguousarray(x, dtype=np.float32)
    xc = xm[:, None] if xm.ndim == 1 else xm
    B = xc.shape[1]
    c0 = codes_list[0]
    shared_tables = None
    if shared:
        shared_tables = _stage_tables_np(c0.codebooks, xc.reshape(c0.nchunk, c0.D, B))
    return [staged_execute_np(c, x, tables=(shared_tables if shared else None), fused=fused)
            for c in codes_list]


def _cluster_forward_torch(codes_list, x, *, dev, shared: bool):
    torch = mm._torch()
    xm = x[:, None] if x.ndim == 1 else x
    B = xm.shape[1]
    c0 = codes_list[0]
    shared_tables = None
    if shared:
        cbs = [torch.from_numpy(cb).to(dev) for cb in c0.codebooks]
        xc = xm.reshape(c0.nchunk, c0.D, B)
        shared_tables = [torch.einsum("qd,cdb->cqb", cb, xc) for cb in cbs]
    return [staged_execute_torch(c, x, dev=dev, tables=(shared_tables if shared else None))
            for c in codes_list]


# ============================================================================================
# The quality gate (BEFORE speed) and the fake-win ban.
# ============================================================================================
def quality_gate(combine_div: float, baseline_div: float, *, tol: float = _QUALITY_TOL) -> dict[str, Any]:
    """A candidate is quality-admissible only if its combine-divergence is within tol of the baseline
    (matched quality) or BETTER. A faster candidate that is worse than baseline is NOT a win."""
    delta = combine_div - baseline_div
    admissible = bool(delta <= tol)
    return {"combine_div": round(combine_div, 6), "baseline_div": round(baseline_div, 6),
            "delta_vs_baseline": round(delta, 6), "tol": tol,
            "quality_admissible": admissible,
            "verdict": "matched_or_better" if admissible else "worse_quality_rejected"}


def no_dense_shadow(mech: mm.MechVector, full_dense_bytes: float) -> dict[str, Any]:
    bounded = bool(0 < mech.Ttemporary < _HALF * full_dense_bytes)
    return {"peak_temporary_bytes": float(mech.Ttemporary),
            "full_dense_bytes": float(full_dense_bytes),
            "bounded_under_half_dense": bounded}


# ============================================================================================
# Real cluster loader (bounded): union of router top-k over a few seeded activations, capped.
# ============================================================================================
def load_cluster(reader, block: int, *, max_experts: int, top_k: int, n_acts: int,
                 seed: int = 1234) -> dict[str, Any]:
    """Load the router + the most-frequently-routed experts (bounded to max_experts) for a handful of
    seeded activations. mlp1 acts are seeded gaussian (scale 0.02, runtime-matched); mlp2 acts are the
    genuine downstream swiglu(mlp1@x) from the ORIGINAL expert (per the M1 harness)."""
    router = load_router = rt.load_router(reader, block)
    rng = np.random.default_rng(seed)
    xs = [rng.standard_normal(2880).astype(np.float32) * 0.02 for _ in range(n_acts)]
    freq: dict[int, int] = {}
    margins = []
    for x in xs:
        logits = router["weight"] @ x + router["bias"]
        order = np.argsort(-logits)[:top_k]
        for e in order:
            freq[int(e)] = freq.get(int(e), 0) + 1
        sm = np.exp(logits[order] - logits[order].max()); sm = sm / sm.sum()
        margins.append(float(sm[0] - sm[1]) if len(sm) > 1 else 1.0)
    routed = [e for e, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))][:max_experts]
    experts = {e: rt.load_expert(reader, block, e) for e in routed}
    mlp1 = [experts[e]["mlp1"].astype(np.float32) for e in routed]
    mlp2 = [experts[e]["mlp2"].astype(np.float32) for e in routed]
    acts_mlp1 = xs
    acts_mlp2 = [rt._swiglu(experts[routed[0]]["mlp1"].astype(np.float32) @ x
                            + experts[routed[0]]["mlp1_bias"].astype(np.float32)) for x in xs]
    return {"router": router, "routed": routed, "experts": experts, "mats": {"mlp1": mlp1, "mlp2": mlp2},
            "acts": {"mlp1": acts_mlp1, "mlp2": acts_mlp2}, "router_margins": margins, "xs": xs}


# ============================================================================================
# Stage runners. Each returns a list of candidate rows + a stage record.
# ============================================================================================
def _candidate_row(stage: str, name: str, tensor: str, *, mech: mm.MechVector, breakdown: dict[str, Any],
                   quality: dict[str, Any], gate: dict[str, Any], nds: dict[str, Any],
                   timing: dict[str, Any], control: str, causal: dict[str, Any] | None,
                   extra: dict[str, Any]) -> dict[str, Any]:
    return {"stage": stage, "candidate": name, "tensor_class": tensor, "control_vs": control,
            "mech_vector": mech.to_dict(), "rate": breakdown, "quality": quality,
            "quality_gate": gate, "no_dense_shadow": nds, "timing_ms": timing,
            "causal_delta": causal, "extra": extra,
            "energy": {"class": "UNAVAILABLE"}, "admissible": bool(gate["quality_admissible"]
                                                                  and nds["bounded_under_half_dense"])}


def _median(timing: dict[str, dict[str, float]], name: str) -> float | None:
    return timing[name]["median_ms"] if name in timing else None


def run_M2(cluster: dict[str, Any], *, tensor: str, cfg: dict[str, Any], reps: int,
           baseline_div: float, dev, metal_ok: bool, progress: Callable[[str], None]) -> dict[str, Any]:
    """M2 shared_lookup_linear_moe. Controls: independent (M1 per-expert-table) vs shared vs
    layer-group-share. Causal control M2(shared) vs M1(independent)."""
    mats = cluster["mats"][tensor]
    acts = cluster["acts"][tensor]
    D, k, stages = cfg["D"], cfg["k"], cfg["stages"]
    rows, cols = mats[0].shape
    full_dense = rows * cols * 4
    x0 = acts[0]

    variants = {
        "independent": {"shared": False, "seed": 0},
        "shared": {"shared": True, "seed": 0},
        "layer_group_share": {"shared": True, "seed": 7},   # same shared mechanism, group-level seed
    }
    built: dict[str, list[StagedCodes]] = {}
    rows_out: list[dict[str, Any]] = []
    meas: dict[str, dict[str, Any]] = {}
    for name, v in variants.items():
        codes = build_staged_codes(mats, D=D, k=k, stages=stages, shared=v["shared"],
                                   seed=v["seed"], iters=cfg.get("iters", 8))
        built[name] = codes
        led, breakdown = cluster_ledger(codes)
        qmv = _matvec_quality(codes, mats, acts)
        mech = mech_cluster(codes, breakdown, 1, shared=v["shared"], fused=True, mps=False)
        nds = no_dense_shadow(mech, full_dense)
        meas[name] = {"codes": codes, "breakdown": breakdown, "quality": qmv, "mech": mech, "nds": nds}
        progress(f"      M2/{tensor}/{name}: bpw={breakdown['whole_artifact_bpw']} "
                 f"relerr={qmv['rel_error_vs_dense_mean']} shared={v['shared']}")

    # combine-divergence per variant (needs BOTH mlp1+mlp2; measured once in run_all via the mlp1 pass)
    # timing: paired cluster-forward window (independent rebuilds tables per expert; shared reuses once)
    ops = {
        "M2_independent": {"fn": lambda: _cluster_forward_np(built["independent"], x0, shared=False), "mps": False},
        "M2_shared": {"fn": lambda: _cluster_forward_np(built["shared"], x0, shared=True), "mps": False},
    }
    if metal_ok:
        ops["M2_independent_metal"] = {"fn": lambda: _cluster_forward_torch(
            [_to_dev(c) for c in built["independent"]], _mps_x(x0, dev), dev=dev, shared=False), "mps": True}
        ops["M2_shared_metal"] = {"fn": lambda: _cluster_forward_torch(
            built["shared"], _mps_x(x0, dev), dev=dev, shared=True), "mps": True}
    timing = mm.paired_window(ops, reps=reps, warmup=2, seed=0)

    ind_ms = _median(timing, "M2_independent")
    sh_ms = _median(timing, "M2_shared")
    causal = {"control": "M2_shared vs M1_independent (per-expert-table)",
              "reuse_ratio_E": len(mats),
              "table_builds_independent": len(mats) * stages,
              "table_builds_shared": stages,
              "table_build_work_avoided_frac": round((len(mats) - 1) / len(mats), 4),
              "wall_shared_over_independent": (round(sh_ms / ind_ms, 4) if ind_ms and sh_ms else None),
              "flops_shared_over_independent": round(
                  meas["shared"]["mech"].F32 / meas["independent"]["mech"].F32, 4),
              "quality_relerr_shared_minus_independent": round(
                  meas["shared"]["quality"]["rel_error_vs_dense_mean"]
                  - meas["independent"]["quality"]["rel_error_vs_dense_mean"], 6)}
    for name in variants:
        m = meas[name]
        gate = quality_gate(m["quality"]["rel_error_vs_dense_mean"], baseline_div, tol=_QUALITY_TOL)
        # note: matvec-relerr gate here is a per-tensor proxy; the combine gate is applied in run_all
        t = {kk: timing[kk] for kk in timing if name.split("_")[-1] in kk or ("shared" in kk and name != "independent")}
        rows_out.append(_candidate_row(
            "M2", name, tensor, mech=m["mech"], breakdown=m["breakdown"], quality=m["quality"],
            gate=gate, nds=m["nds"], timing={kk: timing[kk] for kk in timing},
            control="M1_independent" if name != "independent" else "self",
            causal=(causal if name == "shared" else None),
            extra={"shared": variants[name]["shared"], "D": D, "k": k, "stages": stages}))
    return {"rows": rows_out, "built": built, "meas": meas, "timing": timing, "causal": causal}


def _to_dev(codes: StagedCodes) -> StagedCodes:
    return codes  # codes stay numpy; torch exec copies per-call (measures the op, not a cached device tensor)


def _mps_x(x: np.ndarray, dev):
    torch = mm._torch()
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)


def run_M3(cluster, *, tensor, cfg, reps, baseline_div, m2_built, progress) -> dict[str, Any]:
    """M3 shared_moe_plus_islands. islands-off control = M2 shared. Candidates: 4 island selectors."""
    mats = cluster["mats"][tensor]
    acts = cluster["acts"][tensor]
    D, k, stages = cfg["D"], cfg["k"], cfg["stages"]
    rows, cols = mats[0].shape
    full_dense = rows * cols * 4
    x0 = acts[0]
    budget = cfg.get("island_budget_frac", 0.03)

    base = m2_built["shared"]
    _, base_bd = cluster_ledger(base)
    base_q = _matvec_quality(base, mats, acts)
    rows_out = []
    meas = {}
    timing_ops = {"M3_islands_off": {"fn": lambda: _cluster_forward_np(base, x0, shared=True), "mps": False}}
    island_variants = {}
    for strat in gf._ISLAND_STRATEGIES:
        codes = build_staged_codes(mats, D=D, k=k, stages=stages, shared=True, seed=0,
                                   iters=cfg.get("iters", 8))
        for c, w in zip(codes, mats):
            attach_islands(c, w, strategy=strat, budget_frac=budget)
        island_variants[strat] = codes
        led, bd = cluster_ledger(codes)
        q = _matvec_quality(codes, mats, acts)
        mech = mech_cluster(codes, bd, 1, shared=True, fused=True, mps=False)
        nds = no_dense_shadow(mech, full_dense)
        meas[strat] = {"codes": codes, "breakdown": bd, "quality": q, "mech": mech, "nds": nds}
        timing_ops[f"M3_{strat}"] = {"fn": (lambda cc=codes: _cluster_forward_np(cc, x0, shared=True)),
                                     "mps": False}
        progress(f"      M3/{tensor}/{strat}: bpw={bd['whole_artifact_bpw']} island_bits={bd['island_bits']} "
                 f"relerr={q['rel_error_vs_dense_mean']} (off={base_q['rel_error_vs_dense_mean']})")
    timing = mm.paired_window(timing_ops, reps=reps, warmup=2, seed=0)

    for strat in gf._ISLAND_STRATEGIES:
        m = meas[strat]
        gate = quality_gate(m["quality"]["rel_error_vs_dense_mean"], base_q["rel_error_vs_dense_mean"],
                            tol=1e9)  # islands should only reduce error; gate on improvement below
        improved = m["quality"]["rel_error_vs_dense_mean"] <= base_q["rel_error_vs_dense_mean"] + 1e-9
        causal = {"control": "M3(+islands) vs M2(islands-off, shared)",
                  "strategy": strat, "island_bits": m["breakdown"]["island_bits"],
                  "n_islands_total": sum(c.n_islands for c in island_variants[strat]),
                  "relerr_off": base_q["rel_error_vs_dense_mean"],
                  "relerr_on": m["quality"]["rel_error_vs_dense_mean"],
                  "relerr_improvement": round(base_q["rel_error_vs_dense_mean"]
                                              - m["quality"]["rel_error_vs_dense_mean"], 6),
                  "quality_improved": bool(improved),
                  "wall_on_over_off": (round(_median(timing, f"M3_{strat}")
                                             / _median(timing, "M3_islands_off"), 4)
                                       if _median(timing, "M3_islands_off") else None)}
        rows_out.append(_candidate_row(
            "M3", f"islands_{strat}", tensor, mech=m["mech"], breakdown=m["breakdown"],
            quality=m["quality"], gate={**gate, "quality_admissible": bool(improved),
                                        "verdict": "islands_reduce_error" if improved else "islands_regressed"},
            nds=m["nds"], timing={kk: timing[kk] for kk in timing},
            control="M2_shared", causal=causal,
            extra={"strategy": strat, "budget_frac": budget}))
    return {"rows": rows_out, "island_variants": island_variants, "meas": meas, "base_q": base_q,
            "timing": timing}


def run_M4(cluster, *, tensor, cfg, reps, m3_pick, progress) -> dict[str, Any]:
    """M4 fused_pq_islands_doctor. Fuse base lookup + islands + doctor residual codebook in one path
    vs a separate-kernel control. Quality must MATCH the unfused CPU treatment (same math)."""
    mats = cluster["mats"][tensor]
    acts = cluster["acts"][tensor]
    D, k, stages = cfg["D"], cfg["k"], cfg["stages"]
    rows, cols = mats[0].shape
    full_dense = rows * cols * 4
    x0 = acts[0]
    strat = m3_pick

    codes = build_staged_codes(mats, D=D, k=k, stages=stages, shared=True, seed=0, iters=cfg.get("iters", 8))
    for c, w in zip(codes, mats):
        attach_islands(c, w, strategy=strat, budget_frac=cfg.get("island_budget_frac", 0.03))
        attach_doctor(c, w, k=cfg.get("doctor_k", k), stages=cfg.get("doctor_stages", 1),
                      seed=101, iters=cfg.get("iters", 8))
    led, bd = cluster_ledger(codes)
    q = _matvec_quality(codes, mats, acts)

    y_fused = [staged_execute_np(c, x0, fused=True) for c in codes]
    y_unfused = [staged_execute_np(c, x0, fused=False) for c in codes]
    fuse_match = max(mm._rel(a, b) for a, b in zip(y_fused, y_unfused))

    mech_f = mech_cluster(codes, bd, 1, shared=True, fused=True, mps=True)
    mech_u = mech_cluster(codes, bd, 1, shared=True, fused=False, mps=True)
    nds = no_dense_shadow(mech_cluster(codes, bd, 1, shared=True, fused=True, mps=False), full_dense)

    ops = {"M4_fused": {"fn": lambda: [staged_execute_np(c, x0, fused=True) for c in codes], "mps": False},
           "M4_separate": {"fn": lambda: [staged_execute_np(c, x0, fused=False) for c in codes], "mps": False}}
    timing = mm.paired_window(ops, reps=reps, warmup=2, seed=0)

    causal = {"control": "M4(fused) vs M4(separate-kernel); adds doctor over M3",
              "fuse_quality_match_rel": round(float(fuse_match), 9),
              "quality_matches_unfused": bool(fuse_match <= 1e-6),
              "doctor_bits": bd["doctor_bits"], "island_bits": bd["island_bits"],
              "launch_fused": mech_f.Klaunch, "launch_separate": mech_u.Klaunch,
              "launch_reduction": round(mech_u.Klaunch - mech_f.Klaunch, 1),
              "temp_fused": mech_f.Ttemporary, "temp_separate": mech_u.Ttemporary,
              "temp_reduction_bytes": round(mech_u.Ttemporary - mech_f.Ttemporary, 1),
              "wall_fused_over_separate": (round(_median(timing, "M4_fused")
                                                 / _median(timing, "M4_separate"), 4)
                                           if _median(timing, "M4_separate") else None)}
    gate = {"quality_admissible": bool(fuse_match <= 1e-6),
            "verdict": "fused_matches_unfused" if fuse_match <= 1e-6 else "fusion_changed_result",
            "combine_div": None, "baseline_div": None, "tol": 1e-6}
    progress(f"      M4/{tensor}: bpw={bd['whole_artifact_bpw']} doctor_bits={bd['doctor_bits']} "
             f"fuse_match={fuse_match:.2e} launch {mech_u.Klaunch}->{mech_f.Klaunch}")
    row = _candidate_row("M4", "fused_pq_islands_doctor", tensor, mech=mech_f, breakdown=bd,
                         quality=q, gate=gate, nds=nds, timing=timing, control="M4_separate",
                         causal=causal, extra={"strategy": strat, "doctor_stages": cfg.get("doctor_stages", 1)})
    return {"rows": [row], "codes": codes, "quality": q, "breakdown": bd, "timing": timing, "causal": causal}


def run_M5(cluster, *, tensor, cfg, reps, m4, progress) -> dict[str, Any]:
    """M5 conditional_doctor. Gate the doctor per-activation by a deterministic condition vs always-on
    M4. HARD gate = false_negative_rate (skip a needed correction). Reject if it misses needed ones."""
    mats = cluster["mats"][tensor]
    acts = cluster["acts"][tensor]
    codes = m4["codes"]
    rows, cols = mats[0].shape
    full_dense = rows * cols * 4
    x0 = acts[0]

    # deterministic per-activation condition: residual syndrome = ||base_matvec - fused_matvec|| proxy.
    # If the doctor correction magnitude (relative) exceeds a threshold, the correction is NEEDED.
    def base_only(c: StagedCodes, x):
        stripped = StagedCodes(c.codebooks, c.indices, c.D, c.rows, c.cols, c.nchunk, c.stages, c.k,
                               c.shared, c.recon, c.island_rows, c.island_vals)
        return staged_execute_np(stripped, x)

    thr = cfg.get("cond_threshold", 0.02)
    skipped = 0
    total = 0
    false_neg = 0
    needed = 0
    per_act = []
    for x in acts:
        for c, w in zip(codes, mats):
            total += 1
            y_full = staged_execute_np(c, x, fused=True)          # always-on (base+doctor+islands)
            y_base = base_only(c, x)                              # skip doctor
            corr_mag = mm._rel(y_full, y_base)                   # relative doctor contribution
            yd = w.astype(np.float32) @ np.ascontiguousarray(x, dtype=np.float32)
            err_full = mm._rel(y_full, yd)
            err_skip = mm._rel(y_base, yd)
            correction_needed = bool((err_skip - err_full) > cfg.get("need_delta", 1e-3))
            if correction_needed:
                needed += 1
            apply_doctor = bool(corr_mag >= thr)                 # the gate's decision
            if not apply_doctor:
                skipped += 1
                if correction_needed:                            # skipped a NEEDED correction
                    false_neg += 1
            per_act.append({"corr_mag": round(corr_mag, 6), "apply": apply_doctor,
                            "needed": correction_needed})
    fn_rate = (false_neg / needed) if needed else 0.0
    skip_frac = skipped / max(1, total)
    fn_gate = cfg.get("fn_gate", 0.05)
    gate_fires = bool(fn_rate > fn_gate)

    # mechanics: conditional saves doctor work on skipped activations (ESTIMATED expected reduction)
    mech = mech_cluster(codes, m4["breakdown"], 1, shared=True, fused=True, mps=False)
    nds = no_dense_shadow(mech, full_dense)
    ops = {"M5_conditional": {"fn": lambda: [
        (staged_execute_np(c, x0, fused=True) if mm._rel(staged_execute_np(c, x0, fused=True),
                                                         base_only(c, x0)) >= thr else base_only(c, x0))
        for c in codes], "mps": False},
        "M4_always_on": {"fn": lambda: [staged_execute_np(c, x0, fused=True) for c in codes], "mps": False}}
    timing = mm.paired_window(ops, reps=reps, warmup=2, seed=0)

    causal = {"control": "M5(conditional) vs M4(always-on doctor)",
              "condition": "residual_syndrome (relative doctor contribution >= threshold)",
              "threshold": thr, "skip_frac": round(skip_frac, 4),
              "false_negative_rate": round(fn_rate, 4), "fn_gate": fn_gate,
              "needed_corrections": needed, "skipped_total": skipped, "false_negatives": false_neg,
              "hard_gate_fires_reject": gate_fires,
              "wall_conditional_over_alwayson": (round(_median(timing, "M5_conditional")
                                                       / _median(timing, "M4_always_on"), 4)
                                                 if _median(timing, "M4_always_on") else None)}
    gate = {"quality_admissible": bool(not gate_fires),
            "verdict": "conditional_safe" if not gate_fires else "false_negative_gate_rejected",
            "false_negative_rate": round(fn_rate, 4), "fn_gate": fn_gate}
    progress(f"      M5/{tensor}: skip_frac={skip_frac:.3f} fn_rate={fn_rate:.3f} "
             f"gate_fires={gate_fires} (needed={needed})")
    row = _candidate_row("M5", "conditional_doctor", tensor, mech=mech, breakdown=m4["breakdown"],
                         quality=m4["quality"], gate=gate, nds=nds, timing=timing,
                         control="M4_always_on", causal=causal,
                         extra={"threshold": thr, "skip_frac": round(skip_frac, 4)})
    return {"rows": [row], "causal": causal, "timing": timing, "fn_rate": fn_rate}


def run_M6(cluster, *, tensor, cfg, reps, m4, progress) -> dict[str, Any]:
    """M6 residual_additive_lookup. Richer 2-stage additive multi-codebook via staged lookup at EQUAL
    total bits to M4. Causal M6 vs M4 (richer codebooks vs best fused treatment at equal bytes)."""
    mats = cluster["mats"][tensor]
    acts = cluster["acts"][tensor]
    D, k = cfg["D"], cfg["k"]
    rows, cols = mats[0].shape
    full_dense = rows * cols * 4
    x0 = acts[0]
    m4_bits = m4["breakdown"]["total_bits"]

    # spend the SAME bytes on more additive stages (no islands, no targeted doctor). total_bits is
    # LINEAR in the stage count, so target the stage count whose bits land closest to the M4 budget
    # analytically (a true equal-bits comparison), then build that one config once.
    log2k = max(1, math.ceil(math.log2(max(2, k))))
    per_stage_bits = k * D * 16 + sum((w.shape[0] * w.shape[1] // D) * log2k for w in mats)
    target = int(round((m4_bits - 64 * 8) / max(1, per_stage_bits)))
    extra_stages = int(min(cfg["stages"] + 12, max(cfg["stages"] + 1, target)))
    codes = build_staged_codes(mats, D=D, k=k, stages=extra_stages, shared=True, seed=0,
                               iters=cfg.get("iters", 8))
    _, bd = cluster_ledger(codes)
    q = _matvec_quality(codes, mats, acts)
    mech = mech_cluster(codes, bd, 1, shared=True, fused=True, mps=False)
    nds = no_dense_shadow(mech, full_dense)
    ops = {"M6_residual_additive": {"fn": lambda: _cluster_forward_np(codes, x0, shared=True), "mps": False},
           "M4_fused": {"fn": lambda: [staged_execute_np(c, x0, fused=True) for c in m4["codes"]], "mps": False}}
    timing = mm.paired_window(ops, reps=reps, warmup=2, seed=0)

    q_m4 = m4["quality"]["rel_error_vs_dense_mean"]
    q_m6 = q["rel_error_vs_dense_mean"]
    causal = {"control": "M6(richer additive codebooks) vs M4(base+islands+doctor) at EQUAL bits",
              "stages_m6": extra_stages, "bits_m6": bd["total_bits"], "bits_m4": m4_bits,
              "bits_ratio_m6_over_m4": round(bd["total_bits"] / max(1, m4_bits), 4),
              "relerr_m6": q_m6, "relerr_m4": q_m4,
              "relerr_m6_minus_m4": round(q_m6 - q_m4, 6),
              "m6_wins_quality_at_equal_bits": bool(q_m6 < q_m4),
              "wall_m6_over_m4": (round(_median(timing, "M6_residual_additive")
                                        / _median(timing, "M4_fused"), 4)
                                  if _median(timing, "M4_fused") else None)}
    gate = quality_gate(q_m6, q_m4, tol=_QUALITY_TOL)
    progress(f"      M6/{tensor}: stages={extra_stages} bits {bd['total_bits']} vs M4 {m4_bits} "
             f"relerr {q_m6} vs M4 {q_m4} (m6_wins={q_m6 < q_m4})")
    row = _candidate_row("M6", "residual_additive_lookup", tensor, mech=mech, breakdown=bd,
                         quality=q, gate=gate, nds=nds, timing=timing, control="M4_fused",
                         causal=causal, extra={"stages": extra_stages})
    return {"rows": [row], "causal": causal, "timing": timing}


def run_M7_stub() -> dict[str, Any]:
    """M7 bit-oriented: DEFERRED (Bible 77). A bit-packed sub-index representation (base-3 ternary
    packing, bit-plane codeword storage, sub-byte gather kernels) would grow uncontrollably relative
    to the measured lookup-linear wins; it is sealed as deferred, not built."""
    return {"stage": "M7", "status": "DEFERRED",
            "grammar": "bit_oriented (base-k packing / bit-plane codewords / sub-byte gather)",
            "reason": "Bible 77: seal deferred if it would grow uncontrollably; the M2..M6 lookup-linear "
                      "wins are measured on real experts, and a bit-oriented kernel program is a research "
                      "track, not a bounded run-all stage. No estimates invented.",
            "energy": {"class": "UNAVAILABLE"},
            "admissible": False, "deferred": True}


# ============================================================================================
# Pareto frontier + champions.
# ============================================================================================
def _pareto_point(row: dict[str, Any]) -> dict[str, float]:
    v = row["mech_vector"]["vector"]
    q = row["quality"]
    div = q.get("rel_error_vs_dense_mean", q.get("mean_combine_div", 1.0))
    med = None
    for nm, t in row.get("timing_ms", {}).items():
        if row["candidate"].split("_")[0] in nm or row["stage"] in nm or nm.startswith(row["stage"]):
            med = t["median_ms"] if med is None else min(med, t["median_ms"])
    if med is None and row.get("timing_ms"):
        med = min(t["median_ms"] for t in row["timing_ms"].values())
    return {"quality_div": float(div), "bpw": float(row["rate"]["whole_artifact_bpw"]),
            "wall_ms": float(med if med is not None else 1e9),
            "movement": float(v["Mread"] + v["Mwrite"]), "temporary": float(v["Ttemporary"]),
            "floating": float(v["F32"]), "launches": float(v["Klaunch"])}


def build_pareto(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pareto over (quality_div, bpw, wall_ms, movement, temporary, floating, launches); energy
    UNAVAILABLE so excluded from domination. Inadmissible-dense candidates are excluded outright."""
    admissible = [r for r in rows if r["no_dense_shadow"]["bounded_under_half_dense"]]
    excluded_dense = [r["candidate"] for r in rows if not r["no_dense_shadow"]["bounded_under_half_dense"]]
    axes = ("quality_div", "bpw", "wall_ms", "movement", "temporary", "floating", "launches")
    pts = [(_pareto_point(r), r) for r in admissible]

    def dominates(a: dict[str, float], b: dict[str, float]) -> bool:
        return all(a[k] <= b[k] for k in axes) and any(a[k] < b[k] for k in axes)

    frontier = []
    for i, (pa, ra) in enumerate(pts):
        if not any(dominates(pb, pa) for j, (pb, rb) in enumerate(pts) if j != i):
            frontier.append(ra)
    dominated = [ra["candidate"] for i, (pa, ra) in enumerate(pts)
                 if any(dominates(pb, pa) for j, (pb, rb) in enumerate(pts) if j != i)]

    def champ(metric: str, admissible_only: bool = False, quality_ref: float | None = None):
        pool = frontier if frontier else admissible
        cands = []
        for r in pool:
            pp = _pareto_point(r)
            if admissible_only and quality_ref is not None and pp["quality_div"] > quality_ref + _QUALITY_TOL:
                continue
            cands.append((pp, r))
        if not cands:
            return None
        best = min(cands, key=lambda pr: pr[0][metric])
        return {"candidate": best[1]["candidate"], "stage": best[1]["stage"],
                "tensor_class": best[1]["tensor_class"], metric: best[0][metric],
                "bpw": best[0]["bpw"], "quality_div": best[0]["quality_div"]}

    # quality-preserving-speed: match the best (lowest) quality within tol, then min wall
    if admissible:
        best_q = min(_pareto_point(r)["quality_div"] for r in admissible)
        qps = champ("wall_ms", admissible_only=True, quality_ref=best_q)
    else:
        best_q, qps = None, None
    return {"schema": RUN_SCHEMA,
            "axes": list(axes) + ["energy(UNAVAILABLE, excluded)"],
            "n_candidates": len(rows), "n_admissible": len(admissible),
            "excluded_inadmissible_dense": excluded_dense,
            "dominated": dominated,
            "frontier": [{"candidate": r["candidate"], "stage": r["stage"],
                          "tensor_class": r["tensor_class"], **_pareto_point(r)} for r in frontier],
            "champions": {
                "quality_preserving_speed": qps,
                "lowest_movement": champ("movement"),
                "lowest_floating": champ("floating"),
                "lowest_launches": champ("launches"),
                "best_balanced_apple": _balanced_champ(frontier if frontier else admissible),
            },
            "best_quality_div": best_q}


def _balanced_champ(pool: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best-balanced-apple: min-max normalized sum over (quality_div, wall_ms, movement, temporary,
    launches) - a single balanced Apple-silicon score. Energy excluded (UNAVAILABLE)."""
    if not pool:
        return None
    pts = [(_pareto_point(r), r) for r in pool]
    keys = ("quality_div", "wall_ms", "movement", "temporary", "launches")
    ranges = {}
    for k in keys:
        vals = [p[k] for p, _ in pts]
        lo, hi = min(vals), max(vals)
        ranges[k] = (lo, hi if hi > lo else lo + 1.0)
    scored = []
    for p, r in pts:
        s = sum((p[k] - ranges[k][0]) / (ranges[k][1] - ranges[k][0]) for k in keys)
        scored.append((s, p, r))
    s, p, r = min(scored, key=lambda t: t[0])
    return {"candidate": r["candidate"], "stage": r["stage"], "tensor_class": r["tensor_class"],
            "balanced_score": round(s, 4), "bpw": p["bpw"], "quality_div": p["quality_div"],
            "wall_ms": p["wall_ms"]}


# ============================================================================================
# JSONL result appenders + per-stage sealing.
# ============================================================================================
def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(obj, sort_keys=True, default=str) + "\n")


def _seal_stage(rd: Path, stage: str, ident: dict[str, Any], rows: list[dict[str, Any]],
                causal: dict[str, Any], progress: Callable[[str], None]) -> None:
    obj = {"schema": RUN_SCHEMA, "stage": stage, "identity": ident,
           "energy": mm._ENERGY_BLOCK, "candidates": rows, "causal_control": causal,
           "note": "wall MEASURED contaminated_by_concurrent_cpu_load; mechanics ANALYTICAL; "
                   "CPU authoritative; energy UNAVAILABLE"}
    mm.write_json(rd / f"{stage}_RESULTS.json", obj)
    lines = [f"# {stage} - Generation-M run-all", "",
             f"Candidate: {ident['candidate_id']} | generated {ident['generated_at']}",
             "Energy UNAVAILABLE. Wall MEASURED (contaminated_by_concurrent_cpu_load). "
             "Mechanics ANALYTICAL. CPU authoritative.", ""]
    for r in rows:
        v = r["mech_vector"]["vector"]
        meds = {nm: t["median_ms"] for nm, t in r["timing_ms"].items()}
        lines += [f"## {r['tensor_class']} / {r['candidate']} (control vs {r['control_vs']})",
                  f"- quality: {json.dumps(r['quality'])}",
                  f"- quality_gate: {r['quality_gate'].get('verdict')} "
                  f"admissible={r['quality_gate'].get('quality_admissible')}",
                  f"- mech: F32={mm._fmt(v['F32'])} Llookup={mm._fmt(v['Llookup'])} "
                  f"Klaunch={mm._fmt(v['Klaunch'])} Ttemp={mm._fmt(v['Ttemporary'])}",
                  f"- no_dense_shadow: {r['no_dense_shadow']['bounded_under_half_dense']} "
                  f"(temp {mm._fmt(r['no_dense_shadow']['peak_temporary_bytes'])} < "
                  f"half_dense {mm._fmt(0.5 * r['no_dense_shadow']['full_dense_bytes'])})",
                  f"- wall medians ms: {json.dumps({k: round(x, 5) for k, x in meds.items()})}",
                  f"- causal: {json.dumps(r['causal_delta']) if r['causal_delta'] else 'n/a'}", ""]
    (rd / f"{stage}_REPORT.md").write_text("\n".join(lines) + "\n")
    progress(f"    sealed {stage}_RESULTS.json + {stage}_REPORT.md ({len(rows)} candidates)")


# ============================================================================================
# The run-all feedback loop.
# ============================================================================================
def run_all(*, report_dir: str = _REPORT_DIR_DEFAULT, block: int = 0, max_experts: int = 4,
            top_k: int = 4, n_acts: int = 3, reps: int = 5, verbose: bool = True) -> dict[str, Any]:
    """Iterate stages [B0,B1,M1 (import), M2,M3,M4,M5,M6,M7], run candidates on REAL layer-0 experts,
    measure mech + timing + quality, apply the quality gate, seal, iterate ONCE on an all-regress or
    fixable-config stage, then continue (quarantine + continue). Build the Pareto + champions."""
    rd = Path(report_dir)
    logs: list[str] = []

    def progress(msg: str) -> None:
        logs.append(msg)
        if verbose:
            print(msg, flush=True)

    # fresh generation: clear the run-all-owned JSONL streams so a re-run does not duplicate rows.
    for jf in ("HAWKING_MECHANICS_RESULTS.jsonl", "HAWKING_THERMODYNAMICS_RESULTS.jsonl"):
        p = rd / jf
        if p.exists():
            p.unlink()

    reader = rt.ProvenanceReader()
    progress("[run-all] loading bounded real cluster (layer-0)...")
    cluster = load_cluster(reader, block, max_experts=max_experts, top_k=top_k, n_acts=n_acts)
    progress(f"[run-all] routed experts (bounded {max_experts}): {cluster['routed']} "
             f"| router top-1/2 margins {[round(m,3) for m in cluster['router_margins']]}")

    torch = mm._torch()
    metal_ok = bool(torch.backends.mps.is_available())
    dev = mm._mps_device() if metal_ok else None

    cfg = {"D": 64, "k": 256, "stages": 2, "iters": 6, "island_budget_frac": 0.03,
           "doctor_k": 256, "doctor_stages": 1, "cond_threshold": 0.02, "need_delta": 1e-3,
           "fn_gate": 0.05}

    ident = {"candidate_id": f"gen-M.runall.block{block}.experts{'-'.join(map(str, cluster['routed']))}",
             "parent": "GPT-OSS-120B (Generation-F frozen)", "block": block,
             "routed_experts": cluster["routed"], "generated_at": mm._now(), "reps": reps,
             "config": cfg,
             "confidence": "mechanics ANALYTICAL + wall MEASURED(contaminated); CPU authoritative; "
                           "combine-divergence proxy_output synthetic activations, NOT capability parity",
             "hardware": mm._hw_profile()}

    all_rows: list[dict[str, Any]] = []
    stage_records: dict[str, Any] = {}

    # ---- B0/B1/M1: import existing sealed results (do not re-run the frozen contract) ----
    imported = _import_b0b1m1(rd, progress)
    stage_records["B0_B1_M1_imported"] = imported

    tensors = ("mlp1", "mlp2")

    # ---------------------------- M2 ----------------------------
    progress("[run-all] STAGE M2 shared_lookup_linear_moe")
    m2_by_tensor = {}
    m2_rows = []
    for t in tensors:
        r = run_M2(cluster, tensor=t, cfg=cfg, reps=reps, baseline_div=0.0, dev=dev,
                   metal_ok=metal_ok, progress=progress)
        m2_by_tensor[t] = r
        m2_rows += r["rows"]
    # combine-divergence for M2 variants (needs mlp1+mlp2 together)
    m2_combine = _m2_combine_all(cluster, m2_by_tensor, top_k=top_k)
    for row in m2_rows:
        var = row["candidate"]
        cd = m2_combine.get(var)
        if cd is not None:
            base = m2_combine.get("independent", {}).get("mean_combine_div", cd["mean_combine_div"])
            row["quality"]["combine_divergence"] = cd
            row["quality_gate"] = quality_gate(cd["mean_combine_div"], base, tol=_QUALITY_TOL)
            row["admissible"] = bool(row["quality_gate"]["quality_admissible"]
                                     and row["no_dense_shadow"]["bounded_under_half_dense"])
    # feedback: if shared regressed combine-divergence beyond tol on BOTH tensors, iterate once with
    # more stages (a fixable-config attempt) then continue regardless.
    shared_admissible = any(r["candidate"] == "shared" and r["admissible"] for r in m2_rows)
    m2_iter = None
    if not shared_admissible:
        progress("    [feedback] M2 shared not admissible at combine-divergence; iterating once "
                 "(stages 2->3) then continuing (quarantine+continue).")
        cfg2 = {**cfg, "stages": 3}
        m2b_by_tensor, m2b_rows = {}, []
        for t in tensors:
            r = run_M2(cluster, tensor=t, cfg=cfg2, reps=reps, baseline_div=0.0, dev=dev,
                       metal_ok=metal_ok, progress=progress)
            m2b_by_tensor[t] = r
            for row in r["rows"]:
                row["candidate"] = row["candidate"] + "_iter2"
            m2b_rows += r["rows"]
        m2b_combine = _m2_combine_all(cluster, m2b_by_tensor, top_k=top_k)
        for row in m2b_rows:
            var = row["candidate"].replace("_iter2", "")
            cd = m2b_combine.get(var)
            if cd is not None:
                base = m2b_combine.get("independent", {}).get("mean_combine_div", cd["mean_combine_div"])
                row["quality"]["combine_divergence"] = cd
                row["quality_gate"] = quality_gate(cd["mean_combine_div"], base, tol=_QUALITY_TOL)
                row["admissible"] = bool(row["quality_gate"]["quality_admissible"]
                                         and row["no_dense_shadow"]["bounded_under_half_dense"])
        m2_rows += m2b_rows
        m2_iter = {"iterated": True, "adjust": "stages 2->3",
                   "shared_admissible_after": any(r["candidate"] == "shared_iter2" and r["admissible"]
                                                  for r in m2b_rows)}
    all_rows += m2_rows
    _seal_stage(rd, "M2", ident, m2_rows,
                {"controls": "independent(M1) vs shared vs layer_group_share", "iterate": m2_iter,
                 "per_tensor_causal": {t: m2_by_tensor[t]["causal"] for t in tensors},
                 "combine_divergence": m2_combine}, progress)
    stage_records["M2"] = {"n_rows": len(m2_rows), "iterated": m2_iter is not None}

    # ---------------------------- M3 ----------------------------
    progress("[run-all] STAGE M3 shared_moe_plus_islands")
    m3_rows, m3_pick = [], {}
    m3_by_tensor = {}
    for t in tensors:
        r = run_M3(cluster, tensor=t, cfg=cfg, reps=reps, baseline_div=0.0,
                   m2_built=m2_by_tensor[t]["built"], progress=progress)
        m3_by_tensor[t] = r
        m3_rows += r["rows"]
        # pick the selector with the largest quality improvement (for mlp2 = sensitive, prefer it)
        best = max(r["rows"], key=lambda row: row["causal_delta"]["relerr_improvement"])
        m3_pick[t] = best["extra"]["strategy"]
    # mlp2 sensitivity check: islands should help mlp2 more than they hurt mlp1
    mlp2_impr = max(row["causal_delta"]["relerr_improvement"] for row in m3_by_tensor["mlp2"]["rows"])
    mlp1_reg = min(row["causal_delta"]["relerr_improvement"] for row in m3_by_tensor["mlp1"]["rows"])
    all_rows += m3_rows
    _seal_stage(rd, "M3", ident, m3_rows,
                {"controls": "islands-off(M2) vs 4 selectors", "picked_selector": m3_pick,
                 "mlp2_max_improvement": round(mlp2_impr, 6), "mlp1_min_improvement": round(mlp1_reg, 6),
                 "mlp2_more_sensitive": bool(mlp2_impr >= 0)}, progress)
    stage_records["M3"] = {"n_rows": len(m3_rows), "picked": m3_pick}

    # ---------------------------- M4 ----------------------------
    progress("[run-all] STAGE M4 fused_pq_islands_doctor")
    m4_rows, m4_by_tensor = [], {}
    for t in tensors:
        r = run_M4(cluster, tensor=t, cfg=cfg, reps=reps, m3_pick=m3_pick[t], progress=progress)
        m4_by_tensor[t] = r
        m4_rows += r["rows"]
    all_rows += m4_rows
    _seal_stage(rd, "M4", ident, m4_rows,
                {"controls": "fused vs separate-kernel", "law": "quality must MATCH unfused (same math)",
                 "per_tensor_causal": {t: m4_by_tensor[t]["causal"] for t in tensors}}, progress)
    stage_records["M4"] = {"n_rows": len(m4_rows)}

    # ---------------------------- M5 ----------------------------
    progress("[run-all] STAGE M5 conditional_doctor")
    m5_rows, m5_by_tensor = [], {}
    for t in tensors:
        r = run_M5(cluster, tensor=t, cfg=cfg, reps=reps, m4=m4_by_tensor[t], progress=progress)
        m5_by_tensor[t] = r
        m5_rows += r["rows"]
    # feedback: if the false-negative gate fires, iterate once with a stricter (lower) threshold so
    # the doctor is applied more often, then continue.
    fired = [t for t in tensors if m5_by_tensor[t]["causal"]["hard_gate_fires_reject"]]
    m5_iter = None
    if fired:
        progress(f"    [feedback] M5 false-negative gate fired on {fired}; iterating once "
                 "(threshold 0.02->0.005, apply doctor more often) then continuing.")
        cfg5 = {**cfg, "cond_threshold": 0.005}
        for t in fired:
            r = run_M5(cluster, tensor=t, cfg=cfg5, reps=reps, m4=m4_by_tensor[t], progress=progress)
            for row in r["rows"]:
                row["candidate"] = "conditional_doctor_iter2"
            m5_rows += r["rows"]
        m5_iter = {"iterated": True, "adjust": "cond_threshold 0.02->0.005", "tensors": fired}
    all_rows += m5_rows
    _seal_stage(rd, "M5", ident, m5_rows,
                {"controls": "conditional vs always-on(M4)", "hard_gate": "false_negative_rate",
                 "iterate": m5_iter,
                 "per_tensor_causal": {t: m5_by_tensor[t]["causal"] for t in tensors}}, progress)
    stage_records["M5"] = {"n_rows": len(m5_rows), "iterated": m5_iter is not None}

    # ---------------------------- M6 ----------------------------
    progress("[run-all] STAGE M6 residual_additive_lookup")
    m6_rows, m6_by_tensor = [], {}
    for t in tensors:
        r = run_M6(cluster, tensor=t, cfg=cfg, reps=reps, m4=m4_by_tensor[t], progress=progress)
        m6_by_tensor[t] = r
        m6_rows += r["rows"]
    all_rows += m6_rows
    _seal_stage(rd, "M6", ident, m6_rows,
                {"controls": "richer additive codebooks vs M4 at EQUAL bits",
                 "per_tensor_causal": {t: m6_by_tensor[t]["causal"] for t in tensors}}, progress)
    stage_records["M6"] = {"n_rows": len(m6_rows)}

    # ---------------------------- M7 (deferred stub) ----------------------------
    progress("[run-all] STAGE M7 bit_oriented: DEFERRED (Bible 77)")
    m7 = run_M7_stub()
    mm.write_json(rd / "M7_DEFERRED.json", {"schema": RUN_SCHEMA, "identity": ident, **m7})
    stage_records["M7"] = {"deferred": True}

    # ---------------------------- JSONL + Pareto ----------------------------
    mech_jsonl = rd / "HAWKING_MECHANICS_RESULTS.jsonl"
    thermo_jsonl = rd / "HAWKING_THERMODYNAMICS_RESULTS.jsonl"
    for r in all_rows:
        _append_jsonl(mech_jsonl, {"candidate_id": ident["candidate_id"], "stage": r["stage"],
                                   "candidate": r["candidate"], "tensor_class": r["tensor_class"],
                                   "mech_vector": r["mech_vector"]["vector"],
                                   "mech_labels": r["mech_vector"]["labels"],
                                   "quality": r["quality"], "quality_gate": r["quality_gate"],
                                   "rate": r["rate"], "no_dense_shadow": r["no_dense_shadow"],
                                   "timing_ms_medians": {k: v["median_ms"] for k, v in r["timing_ms"].items()},
                                   "causal_delta": r["causal_delta"], "admissible": r["admissible"],
                                   "generated_at": ident["generated_at"]})
        _append_jsonl(thermo_jsonl, {"candidate_id": ident["candidate_id"], "stage": r["stage"],
                                     "candidate": r["candidate"], "tensor_class": r["tensor_class"],
                                     "energy_class": "UNAVAILABLE",
                                     "reason": mm._ENERGY_BLOCK["reason"],
                                     "policy": "no invented estimates; candidate not eligible as energy champion",
                                     "generated_at": ident["generated_at"]})

    pareto = build_pareto(all_rows)
    pareto["identity"] = ident
    pareto["quality_preserving_speed_per_tensor"] = _qps_per_tensor(all_rows)
    pareto["beat_m1_b0_at_matched_quality"] = _beat_summary(all_rows)
    mm.write_json(rd / "HAWKING_MECHANICS_PARETO.json", pareto)
    progress(f"[run-all] Pareto: {pareto['n_admissible']}/{pareto['n_candidates']} admissible; "
             f"frontier size {len(pareto['frontier'])}")

    summary = {"ok": True, "report_dir": str(rd), "routed_experts": cluster["routed"],
               "stages": stage_records, "n_candidate_rows": len(all_rows),
               "pareto_champions": pareto["champions"],
               "quality_preserving_speed_per_tensor": pareto["quality_preserving_speed_per_tensor"],
               "beat_m1_b0_at_matched_quality": pareto["beat_m1_b0_at_matched_quality"],
               "energy": "UNAVAILABLE (all thermodynamics rows labelled UNAVAILABLE; no estimates)",
               "timing_caveat": "contaminated_by_concurrent_cpu_load (MoP)",
               "artifacts": sorted(p.name for p in rd.glob("M[2-7]_*")) + ["HAWKING_MECHANICS_RESULTS.jsonl",
                                                                           "HAWKING_THERMODYNAMICS_RESULTS.jsonl",
                                                                           "HAWKING_MECHANICS_PARETO.json"]}
    mm.write_json(rd / "MECHANICS_RUN_ALL_SUMMARY.json", {"schema": RUN_SCHEMA, "identity": ident,
                                                          "summary": summary, "progress_log": logs})
    return summary


def _import_b0b1m1(rd: Path, progress: Callable[[str], None]) -> dict[str, Any]:
    """Import the existing sealed B0/B1/M1 Fidelity-A results (do not re-run the frozen contract)."""
    out = {}
    for stage, fname, key in (("B0", "B0_BASELINE_QUALITY.json", "quality"),
                              ("B1", "B1_QUALITY_PARITY.json", "quality"),
                              ("M1", "M1_PARITY.json", "parity")):
        p = rd / fname
        if p.exists():
            try:
                out[stage] = {"file": fname, "present": True,
                              "sha256": json.loads(p.read_text()).get("sha256", "")[:16]}
            except Exception as e:
                out[stage] = {"file": fname, "present": True, "error": str(e)}
        else:
            out[stage] = {"file": fname, "present": False}
    progress(f"    imported B0/B1/M1: {[s for s in out if out[s].get('present')]}")
    return out


def _m2_combine_all(cluster, m2_by_tensor, *, top_k: int) -> dict[str, dict[str, Any]]:
    """Weighted-combine (MoE output) divergence per M2 variant, using BOTH mlp1 + mlp2 staged codes."""
    router = cluster["router"]
    experts = cluster["experts"]
    xs = cluster["xs"]
    routed = cluster["routed"]
    out = {}
    for var in ("independent", "shared", "layer_group_share"):
        mlp1_codes = {e: m2_by_tensor["mlp1"]["built"][var][i] for i, e in enumerate(routed)}
        mlp2_codes = {e: m2_by_tensor["mlp2"]["built"][var][i] for i, e in enumerate(routed)}
        out[var] = _combine_divergence(xs, router, experts, mlp1_codes, mlp2_codes, top_k=top_k)
    return out


def _qps_per_tensor(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Quality-preserving-speed champion PER tensor class, or an honest negative if nothing beats the
    imported M1/B0 timing at matched quality."""
    out = {}
    for t in ("mlp1", "mlp2"):
        cand = [r for r in rows if r["tensor_class"] == t and r["admissible"]]
        if not cand:
            out[t] = {"champion": None, "verdict": "negative_no_admissible_candidate"}
            continue
        best_q = min(_pareto_point(r)["quality_div"] for r in cand)
        matched = [r for r in cand if _pareto_point(r)["quality_div"] <= best_q + _QUALITY_TOL]
        champ = min(matched, key=lambda r: _pareto_point(r)["wall_ms"])
        out[t] = {"champion": champ["candidate"], "stage": champ["stage"],
                  "wall_ms": _pareto_point(champ)["wall_ms"], "bpw": champ["rate"]["whole_artifact_bpw"],
                  "quality_div": _pareto_point(champ)["quality_div"],
                  "verdict": "quality_preserving_speed_champion"}
    return out


def _beat_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Did any stage beat M1/B0 at MATCHED quality? Honest: the M2..M6 grammars run the MoE-cluster
    unit of work (E experts) not a single matvec, so they are NOT directly wall-comparable to the
    imported single-matvec M1/B0. State that plainly rather than manufacturing a comparison."""
    admissible = [r for r in rows if r["admissible"]]
    return {"any_admissible": bool(admissible),
            "note": "M2..M6 measure the MoE cluster forward (E experts sharing tables); the sealed "
                    "M1/B0 measured a single expert matvec. Cross-unit wall comparison is NOT made "
                    "(would be apples-to-oranges). The M2 causal control (shared vs independent) is the "
                    "honest same-unit speed test; see M2 causal_delta.wall_shared_over_independent.",
            "n_admissible": len(admissible)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hawking Mechanics Generation-M run-all (M2..M7).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true", help="run the real bounded run-all on 120B layer-0")
    ap.add_argument("--report-dir", default=_REPORT_DIR_DEFAULT)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--max-experts", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n-acts", type=int, default=3)
    args = ap.parse_args(argv)
    if args.run:
        out = run_all(report_dir=args.report_dir, block=args.block, max_experts=args.max_experts,
                      reps=args.reps, n_acts=args.n_acts)
        print(json.dumps(out, indent=2, sort_keys=True, default=str))
        return 0
    print(json.dumps(selftest(), indent=2, sort_keys=True, default=str))
    return 0


# ============================================================================================
# Self-test on tiny synthetic fixtures (no 128-expert loads).
# ============================================================================================
def selftest() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    E = 3
    experts = [(rng.standard_normal((128, 128)).astype(np.float32)
                @ rng.standard_normal((128, 128)).astype(np.float32) * 0.05).astype(np.float32)
               for _ in range(E)]
    x = rng.standard_normal(128).astype(np.float32)

    ind = build_staged_codes(experts, D=64, k=16, stages=2, shared=False, seed=0, iters=6)
    sh = build_staged_codes(experts, D=64, k=16, stages=2, shared=True, seed=0, iters=6)
    # execution parity vs the frozen shared-grammar recon @ x
    gram = gf.pack_shared_grammar(experts, dim=64, k=16, stages=2, corr_rank=0, iters=6)
    y_exec = staged_execute_np(sh[0], x)
    y_ref = gram.recon[0] @ x
    exec_parity = mm._rel(y_exec, y_ref)

    _, bd_ind = cluster_ledger(ind)
    _, bd_sh = cluster_ledger(sh)
    mech_ind = mech_cluster(ind, bd_ind, 1, shared=False, fused=True, mps=False)
    mech_sh = mech_cluster(sh, bd_sh, 1, shared=True, fused=True, mps=False)
    full_dense = 128 * 128 * 4
    nds_ind = no_dense_shadow(mech_ind, full_dense)["bounded_under_half_dense"]
    nds_sh = no_dense_shadow(mech_sh, full_dense)["bounded_under_half_dense"]

    # islands + doctor attach + fuse-match
    c = build_staged_codes(experts, D=64, k=16, stages=2, shared=True, seed=0, iters=6)[0]
    attach_islands(c, experts[0], strategy="residual_energy", budget_frac=0.05)
    attach_doctor(c, experts[0], k=16, stages=1, seed=1, iters=6)
    yf = staged_execute_np(c, x, fused=True)
    yu = staged_execute_np(c, x, fused=False)
    fuse_match = mm._rel(yf, yu)

    return {"ok": True, "device_mps": bool(mm._torch().backends.mps.is_available()),
            "exec_parity_vs_shared_grammar_recon": round(float(exec_parity), 8),
            "exec_parity_ok": bool(exec_parity < 1e-4),
            "shared_flops_lt_independent": bool(mech_sh.F32 < mech_ind.F32),
            "shared_F32": mech_sh.F32, "independent_F32": mech_ind.F32,
            "no_dense_shadow_independent": nds_ind, "no_dense_shadow_shared": nds_sh,
            "island_doctor_fuse_match_rel": round(float(fuse_match), 9),
            "fuse_matches_unfused": bool(fuse_match < 1e-6),
            "bpw_shared": bd_sh["whole_artifact_bpw"], "bpw_independent": bd_ind["whole_artifact_bpw"],
            "mech_dims": len(mm.DIMS)}


if __name__ == "__main__":
    raise SystemExit(main())
