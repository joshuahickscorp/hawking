# STRAND ↔ dismantle — integrate / no-integrate close-out

**Status:** DRAFT, 2026-06-14. Grounded in dismantle's `paradigmshift.md` (code-audited 2026-06-01) + dismantle's loader code + STRAND's measured Metal kernel. The one pending input is the 7B selective-PV result (running); it sets the *quality* of the option but **does not change the core verdict**.

## The decision in one line
Integrating STRAND into dismantle is a **default-off, per-tensor FFN footprint + determinism option** that sits **off dismantle's tps↑/J↓ critical path**. Worth it only if "deterministic + compact deployment" is a product axis you want — it is **not** a speed or energy win for dismantle.

## Grounded in dismantle's own reality (not STRAND's hopes)
- dismantle's north star (paradigmshift.md): **tps↑, J/tok↓**. ~31 dec_tps today; the gap to llama.cpp's ~50 on the same HW is **recoverable *runtime* headroom** — dispatch fusion (~180/token → fewer), f16 activations + f16 KV, GPU sampling. **Not the format.**
- The custom format / sub-4-bit is **roadmap item #4, explicitly labeled "secondary"** and **gated**: *"Validate QTIP-on-Metal BEFORE designing the format — if trellis decode is compute-heavy on Apple's SIMD model, a 2-bit format could be slower than Q4_K, exactly the trap dismantle's own Q3_K kernel fell into (compute-bound at 24% peak)."*

## STRAND clears that gate — on the pessimistic side (this is the key convergence)
- STRAND **is** a QTIP-class trellis quantizer, with a deterministic float-free Metal decode kernel (`strand_bitslice.metal`).
- Measured on this M3: bitslice fused B=1 decode **34.6 Gw/s ≈ 18% of peak, ALU-bound** → hybrid 7B decode ceiling **~6–10 tok/s** (per-column RHT) / ~1 tok/s (per-row), **below Q4_K's 20–30.**
- **d=2 vector kernel MEASURED 2026-06-14 (gate-bitslice): 1.18× (40.78 Gw/s) = MARGINAL, below the ≥1.3× bar; identity-verified (864 cells).** Root cause: B=1 decode is ALU-bound, d=2 cuts index-reads not the ALU wall → it does NOT unlock B=1. (Decode primitive is 53% peak/healthy; prompt-GEMM 9.2× CPU @B=4 — only B=1 autoregressive decode is the wall.) So speed-via-STRAND-kernel is measured-hard; best-of-both = the hybrid (STRAND-FFN compression + Q4_K-attention speed + dismantle runtime levers), not STRAND-decode beating Q4_K.
- → On Apple, STRAND's trellis decode is **compute-bound, not bandwidth-bound** → **footprint-only, not a speed win.** dismantle's own roadmap-#4 asterisk, STRAND's measurement, and the dismantle-chat's analysis all converge on the same answer.

## The integration seam exists today
- dismantle has a per-tensor **load-time repack architecture**: `ffn_down_q4k` (+ predec / f16-scale twins), Q4_K LM-head, `q4k_fast` — each an opt-in `PinnedBuffer` + a kernel selector in `forward_token_greedy_tcb` (qwen_dense.rs).
- STRAND slots in cleanly as `ffn_down_strand: Option<PinnedBuffer>` + a `strand_bitslice` GEMV branch. `ffn_down` is the **largest per-layer weight** (Q6_K ~18.5 MB) → the natural footprint target for the per-tensor hybrid (attention/QKV stay Q4_K-fast; FFN goes STRAND-compact).
- Honest cost of that hybrid: the STRAND FFN GEMV is ALU-bound, so it **loses tps** vs all-Q4_K. Footprint/determinism is bought with throughput.

## What the PV result (pending) decides — and what it doesn't
- **Decides:** the *quality* of the footprint option — does 7B 2-bit selective-PV reach ≤0.15 loss-tax (vs the ~0.28–0.31 PTQ floor).
- **Does NOT decide:** the speed verdict (footprint-only stands regardless of PV quality).
- Already in hand: de-risk gate proved the PV mechanism (0.5B 2-bit 78.95 → 22.16, down-only beats full-PV); PTQ floor 7B q2 = 10.54 / 14B 8.92 / 32B 6.61.

## Recommendation
1. **For dismantle's metrics:** STRAND is off the critical path. The tps/J prize is the runtime levers (dispatch fusion, f16 KV) — do those first.
2. **For a deterministic-compact deployment axis (the moat):** STRAND is the right vehicle — 673-cell bit-identity verified on this M3; reproducible, portable, ~3-bit. Wire it via the `ffn_down` seam, **default-off**.
3. **The integrate call is strategic, not metric-driven:** is deterministic/compact deployment a product axis you want to own? If yes → integrate as a default-off feature. The PV result raises the option's quality; it doesn't change its strategic role.

## Pending to finalize this doc
- [ ] 7B selective-PV loss-tax number (running; result-identical sharded requant fixed the kill-loop).
- [ ] (optional) the exact `ffn_down_strand` kernel-dispatch diff in qwen_dense.rs — design only, no build/test in dismantle without a go.
