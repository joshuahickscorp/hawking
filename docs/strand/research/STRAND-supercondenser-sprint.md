# STRAND Supercondenser Sprint — the final push before dismantle/integration

Launched 2026-06-13 via an 18-agent frontier workflow (run `wf_f8917fca-d07`): design every
bpw/PPL/speed lever + cross-disciplinary research (astrophysics, info-theory, biology, ontology,
reversible compute), adversarially filter the redundant, synthesize one gated sprint.

## Targets (each MUST preserve bit-identical integer decode — the determinism moat)

| axis | now | target | physical floor |
|------|-----|--------|----------------|
| ship bpw (density) | 2.665 | **2.40** | Shannon side-info floor ~2.35 |
| 2-bit loss-tax `ln(PPLq/PPLbf16)` (fidelity) | 0.324 | **≤0.15** | bf16 (tax 0) |
| 3-bit loss-tax | 0.056 | **≤0.05** | indistinguishable |
| decode (speed) | 74% of mem-BW | **≥90%** | memory bandwidth |
| encode | 2.36× | **≥5×** | GPU Viterbi throughput |

## Walls map — the rigorous "becoming the densest"

**GENUINELY NEAR (~0.06 bpw away): Shannon rate-distortion / source coding** — the ONLY
physical-information bound that binds at STRAND's scale. The 2.0-bit *payload* already sits AT the
Gaussian R(D) floor (R(D)=½log₂(σ²/D) ⇒ D=0.0625σ² at 2 bits), because RHT deliberately pins the
weights to the Gaussian (Shapiro worst-case) source. The payload is **saturated** — this is exactly
why "entropy-code the indices" is dead (post-RHT indices are max-entropy/uniform). The entire
reachable gap is the **0.665-bit side-info**, and the bit-ledger localizes 0.232 bpw as provably
recoverable to source entropy: scale_q 0.084 ("stop billing 32 bits for a 10.5-bit symbol"),
outlier-positions 0.148 (gap-coding, H 21.0→7.82), sub_scale 0.022. Closing it lands ship at ~2.43
(C2+OUTL), approaching the 2.35 floor — zero change to the frozen-integer decode.

**PURE FRAMING (true but astronomically far — NEVER market, use only to debunk overclaims):**
Bekenstein bound — a ~2 kg, R~0.15 m card caps at 7.7e42 bits vs STRAND-32B's ~8.5e10 bits ⇒ **gap
~1e32 (32 orders)**. Holographic/Bekenstein-Hawking ~2.7e68 bits ⇒ gap ~1e57. Landauer — decode LUT
erases ~10 bits/weight at kT·ln2=2.87e-21 J but real cost ~5e-12 J/weight (HBM at ~2.5 pJ/bit) is
**~1e8× above** the floor; erasure thermodynamics is invisible, the binding term is data-movement
(which points the *same way* as the bpw target). The float-free integer decode IS at the
reversible/Landauer floor for its arithmetic (it erases nothing) — an honest defensive statement, not
a lever.

**DODGEABLE (not a wall STRAND faces):** sub-Landauer adiabatic logic (cryogenic-only);
Bremermann/Margolus-Levitin quantum op-rate (decode ~2e13 bit/s is ~37 orders below the ceiling —
correctly reframes decode as DRAM-bandwidth-bound, which the 74% figure already knew); analog/in-memory
MVM decode (provably non-bit-exact ⇒ auto-disqualified from the ship lane).

**Bottom line:** STRAND is ~0.06 bpw from its only real wall and 32–57 orders from every cosmic bound.
The honest "densest" story is the Shannon one.

## The 7 ranked levers (local-first during PV uptime; cloud gated on the 0.5B green light)

1. **[LOCAL] SDSQ side-info rANS (C2)** — entropy-code scale_q + sub_scale. **2.665→~2.559 bpw**, zero
   quality cost, zero determinism risk (integer-only rANS, static 14-bit CDF in the bytestream).
   `sideinfo_rans.rs` coder ACTIVATED 2026-06-13 (`lib.rs` lacked `pub mod sideinfo_rans` — it was an
   orphan) + **29/29 determinism tests green**. Remaining: `sideinfo_wire.rs` (~250-line copy of
   `outlier_wire.rs` append/read/EOF-chain), producer hook in `quantize-model.rs` (~:1087, behind
   `--sdsq-sideinfo`), scale_q-overwrite in the decode apply path, A/B gate on the real Qwen2.5-0.5B
   artifact. Deploy v2 seek table (`format.rs:330-331`) byte-UNTOUCHED.
2. **[LOCAL] DBIA deployment** — fold the −28.7% q2 de-bias win into the `.strand` format (DBIA section
   EOF-chained like OUTL/SPRV + decode MAC epilogue `y[o]+=bf16_to_f32(c[o])`). The hard part
   (`debias_wire.rs` codec, 65536-bf16 exhaustive oracle) is DONE. **LOAD-BEARING HAZARD — fix FIRST:**
   `outlier_wire.rs` `read_outl_bytes` (~:433) steps over SPRV only; with OUTL→DBIA→SPRV order it hits
   DBIA magic, returns `Ok(None)`, and **silently drops outliers** at decode. Teach `read_outl_bytes` +
   `provenance_io` to step over DBIA/RSLT magics; un-ignore `sprint_section_chain_audit`. Costs +0.01–0.02
   bpw (a 28.7%-PPL-for-<0.02-bpw trade). This is the deployment for the loss-tax target.
3. **[LOCAL, GPU-BLOCKED] Staged-tile bitslice decode** — coalesced Q12 flush kills the ~32-lane scatter
   store (85–88% of decode traffic) → ≥90% of decode-*primitive* peak (k2 L12) / high-80s–90s (k3 L7).
   Does NOT move the fused y=Wx token path (table-bound, separate lever). BLOCKED until PV frees MPS; then
   `cargo run -p strand-decode-kernel --release --bin gate-bitslice-staged -- --force` (green = staged
   byte-identical to `decode_tensor_fixed` AND %peak ≥ deployed).
4. **[LOCAL] Wolf-Ziv function-projection** — the one novel *research* survivor = "quantize the FUNCTION
   not the weights." Quantize Ŵ=P_F·W (top-k Fisher image), spending zero error budget in the loss
   null-space. **LOW confidence** (RHT may make per-block Fisher full-rank; Hessian-Viterbi already
   backfired +1.1% on this pipeline). **FALSIFY FIRST (~1 day CPU):** form per-256-block empirical Fisher
   from the y-space grad² STRAND already derives, eigendecompose ~50 blocks, plot rank-k for 99% of trace;
   **KILL if 99%-energy rank > ~h/2**. Decode byte-untouched (encoder-target change only).
5. **[LOCAL] Energy meter (diagnostic)** — replace the `energy.rs` stub (`Backend::detect()` always
   Unavailable) with real powermetrics/RAPL → pJ/decoded-weight per gate run. Measurement-only, never on
   the hot path. Lowest priority.
6. **[CLOUD] Selective-PV at scale, KL-routed** — train RED classes (up/v/gate) through the STRAND encoder
   recon, RED@4 + de-bias stacked, 7B then 32B. The single largest move on the loss-tax target
   (PV removed ~1.1 nats at 0.5B). Realistic landing: 32B tax ~0.12–0.18, 7B ~0.18–0.28.
   **CAVEAT:** mp-kl-routed lifts v/up/gate to 4-bit ⇒ ~**3.2–3.8 deploy bpw** — this buys 2-bit QUALITY
   at 3-bit DENSITY; it FIGHTS the 2.40-bpw target. NEVER report a combined "smaller AND better" from this
   arm; density comes from levers 1–2 separately. Code is ZERO (`cloud-selective-pv.sh` written +
   self-gated via `require_promoted`).
7. **[CLOUD] cloud-GPU block-batched encode** — wire `cloud-gpu_dispatch.rs` (per-thread host levels 30.06 GB→~16.8 MB
   at batch=512) to unblock >32B/70B/405B GPU encode (the exact wall the 70B hit). VALIDATE-ONLY; CPU stays
   canonical; parity oracle MUST be the f32 CPU reference (cloud-GPU computes f32 distance). Separate pod leg.

## Discarded as redundant (do NOT revisit)

- Entropy-coding post-RHT **indices** — max-entropy/uniform, dead.
- **SeedLM** (store LFSR seed + coeffs, regenerate) — its low-rank-in-pseudorandom fit is destroyed by RHT
  (post-RHT block error is near-white/full-rank); published 97.9% retention is on un-rotated Llama-3.
- **Hierarchical/temporal predictive coding** of de-bias residuals across layers — side-info is only
  ~0.014 bpw total (21× too small to matter); c_L/c_{L-1} live in different RHT-seeded spaces (no row
  correspondence). At most a 10-line correlation scout; kill if corr<0.1.
- **Combinatorial N-of-M** outlier-position coding — C(n,k) bound H(0.01)=0.0808 bpw is WORSE than the
  shipping gap-coded 0.0783 (i.i.d. mask discards intra-row clustering gap-coding captures).
- **Enumerative trellis-path** coding — the path object is init_state (near-incompressible H~10/12); the
  right tool is tail-biting (structural), not entropy. Bit-ledger: "init-state: do NOT entropy-code."
- **Per-RHT-block MDL routing** — RHT flattens the block structure it presumes; KL-routing already does
  this at tensor grain (rung-kl inverts naive down-protect). At most a cheap KL-vs-loss-delta A/B.
- **Protein-folding generative decode** — a heavy energy-minimization violates the cheap-LUT/float-free/
  ≥90%-BW moat; finding a cheap deterministic generator for arbitrary weights IS the whole problem.
- **Reversible/adiabatic/analog decode** — physically-irrelevant (sub-Landauer is cryogenic; ~1e8×
  dominated by HBM at 300K) or moat-breaking (analog MVM non-bit-exact).
- **Rank-sweep / eigen water-fill** across blocks — `error-spectrum-dp_d4_r2_down.json` proves error is
  near-white post-RHT (rank-1 captures 0.24%, rank-64 only 14%); rank-0 de-bias already banked the one
  structural direction. Banked-dead.
- **QTIP scale-folding** (targets sub_scale, the wrong 0.022-bpw stream) + per-tensor-derived RHT seed
  (saves 0.00003 bpw). Non-levers.

## Cloud arm

Pod `root@213.192.2.110:40078` (up 49 days, stable, idle, 0% GPU — billing for nothing). Selective-PV is
gated **two gates downstream** of the running PV: `pv_down_protect_q2` (PID 28690, ~step 130/300, ETA
~Jun-14) → `research/pv-dp/pv-dp.json` → `gate 10-pv-kl-routed.sh` → `research/pv-dp/pv-kl-routed.json` →
`promote.py` stamps `PROMOTE_CLOUD` (needs PPL≤30 AND <26.77) → `cloud-selective-pv.sh require_promoted`
authorizes. **Pre-flight 2026-06-13 found two real gaps** the gated run would have hit: the on-pod
`quantize-model` binary is broken (937 bytes) and there are 0 weight shards at `scratch/qwen-7b|32b`.
Background agent is rebuilding the binary + readying 7B weights + deploying the latest script; 32B deferred
(7B-first; volume ~68 GB free). At gate-flip: scp the stamped `pv-kl-routed.json` to `/workspace/gates/`,
`touch /workspace/SKIP-32B`, `nohup bash scripts/cloud-selective-pv.sh`. Success = `ppl_selpv_7b.json`
loss_tax ≤0.15.

## Constraint during PV uptime

PV (28690) OWNS the Apple GPU/MPS (no second MPS job) and the box is RAM-bound (18 GB, heavy swap — PV's
s/step is climbing under pressure). Local lever work must be **CPU-only, low-RAM, bounded `-j4`**. Lever 3
is GPU-blocked until PV exits; the conductor's gate queue (gate-20 CPU lowrank judge next) self-blocks
behind the `strand-qat.py` pgrep guard — do not pre-empt it.

---

## COMPLEMENTARY FINDINGS from the parallel 13-wave fleet (other session, 2026-06-13)
Two chats ran convergent sprints — same walls map, same C2/sideinfo approach, same de-bias law, same discard list (strong independent validation). The 13-wave fleet caught three things this sprint's wave should fold in before executing. Full detail in `docs/STRAND-quality-density-frontier.md` §12–§16.

1. **fp32 PV-shadow OOMs at 32B (strand-qat.py) — a landmine on the deferred 32B leg.** The frozen PV shadow is fp32 (8 B/param) → ~238 GB at 32B → OOM-killed (SAME class as the 70B encode OOM). The cloud-arm's pre-flight didn't surface this (it caught stale-checkout + missing-scripts — complementary). **Before enabling the 32B selective-PV leg: switch to a bf16 frozen shadow + 8-bit Adam.** 7B is unaffected (fits). (fleet wave wiohtqx42 / §13)

2. **DBIA (Lever 2) MUST be sealed in the attestation or it silently breaks the moat.** The audit found `descriptor_digest`/`verify_archive` have no DBIA slot — a tampered/dropped de-bias correction (a real f32 add per output row) passes SPRV verification unnoticed, breaking bit-identical attestation. **Step 1 of the DBIA deploy: add `dbia_digest` to `descriptor_digest` + read DBIA in `verify_archive` (mirror OUTL).** Not just the chain-walk fix. (fleet wave wxgehdvom / §15)

3. **C2/SDSQ needs whole-model concatenation to hit ~2.43 bpw.** The bake-off confirmed the mode-adaptive coder is byte-exact (87,381-stream cert) but per-tensor small-block (256 blocks) only recovers 0.053 bpw — the per-symbol table dominates. **The `sideinfo_wire` producer must concatenate streams model-wide (or use a shared frozen model)** so the table amortizes to ~0.0001 bpw and the win lands. (fleet wave wm521gic6 / §16)

Convergent (both sessions agree): the ~0.06-bpw-from-Shannon-floor walls map; cosmic limits are defensive-only; the chain-walk step-over silent-drop hazard; quality (PV) and density (C2/DBIA) are separate headlines; the Wolf-Ziv / function-projection bet is the one survivor worth a cheap falsification.

DIVISION OF LABOR (avoid collision on the shared tree/pod/PV): the sprint-wave session owns EXECUTION (Lever 1 wiring, DBIA deploy, cloud-arming, Wolf-Ziv probe — it's ahead and `lib.rs` is its edit). The fleet session owns SYNTHESIS (the §12–§16 findings, the master-synthesis) + does NOT re-wire/re-arm/edit lib.rs/touch the pod.
