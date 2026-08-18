# Handoff — 2026-08-16 ascent session

**12/25 verified, 5 blocked.** 128 commits on main, unpushed.

## Waiting on you — one decision

**DSV4F `hc_sha` reseal.** This single choice gates G004 and G007.

| option | bytes/token | floor | roof | clears 50 TPS? |
|---|---|---|---|---|
| C1 keep bit-identity | 8.839 GB | 21.48 ms | 46.56 TPS | **no**, short by 0.608 GB |
| C2 reseal | 7.042 GB | 17.11 ms | **58.44 TPS** | **yes** |

C2 replaces bit-identity with greedy-token identity + logit delta ≤ 0.05 + HC cosine
≥ 0.995 (a `wq_a` probe already measures 0.99653). The assert must be **reseated, not
dropped**. My view: worth taking — greedy-token identity is what matters for generation
and those bounds are real. But it changes what correctness *means* for that model, so
it's yours. Receipt: `DSV4F_UNBLOCK_REQUIRES_RESEAL.json`.

## Two lanes still running

- `qwen38-locate-coherence-floor` — bisecting between 2.0856 (incoherent) and 4.2527
  (coherent) BPW. A floor above 3.0 is a valid finding; it's told not to force a pass.
- `auto-q80-host-preparation-residual-embed` — **launched by the daemon, not by me.**
  Targets the ~25 ms embed `packed()` residual.

Harvest both with `python3 tools/lane_health.py` first — lanes finish uncommitted and
`grok-run status` reports dead ones as running.

## Next actions, in order

1. **Re-run the 14-component Q80 decomposition on current main.** The old table doesn't
   close against the measured 108.3 ms — 26 ms unattributed, partly cost moving between
   buckets. That instrument found all four of today's wins; it's the highest-value
   measurement available. 69.1 ms of the token sits outside `gpu_matvec`.
2. Harvest the two live lanes.
3. G011 needs a **full unattended cycle** — two consecutive non-null `launched` fields
   where the second traces to the first lane's `NEXT_BOTTLENECK`.

## Standing rules earned today

- `tools/merge_guard.py <branch>` before every merge. **Serialise merges against
  in-flight lanes** — the guard catches staleness, nothing stops you creating it.
- Every contract ends with `NEXT_BOTTLENECK: <top cost, measured ns>` or the loop starves.
- Contracts need `## ACCEPTANCE`, a backticked `cargo` command, and `## EDIT <path>` lines
  or the gate rejects them. Never `SG_OFF`.
- Check a metric's **ceiling** against its threshold before commissioning work.
- **Never trust a check you haven't watched fail.**

## Blocked, with named inputs

- **G004/G007** — DSV4F: needs ≤8.230 GB/token; see the decision above.
- **G006** — Qwen3.8: coherence floor above 2.0856 BPW. Needs an attention codec holding
  coherence below Q4. Attention is 74% of the artifact at 4.250 BPW; MLP is already 0.848.
- **G005** — capability half MET (displacement 0.5556 on a widened bench); speed half
  inherits G006.
- **G008** — 100 TPS floor reachable; TG3/333 needs ~0.805 BPW and is byte-blocked.

## Q80 today

1376.3 → **108.306 ms** (12.7x), measured across 3 reps under the lock. Four
eliminations, none of them compute. fs/weight 386,270 → ~30,400. Still 5.4x from the
20 ms gate; 69.1 ms of the token is non-`gpu_matvec`.
