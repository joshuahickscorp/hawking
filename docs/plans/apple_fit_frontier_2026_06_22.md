# Apple Fit Frontier (2026-06-22)

This is the Apple-Silicon-specific expansion lane for making Hawking the
obvious local AI runtime on Macs. The core product question is:

> Can Hawking discover and run the strongest usable configuration for this exact
> Apple machine without reducing the engine's maximum capability?

The goal is not a cautious manager that makes Hawking smaller. The goal is a
capability amplifier: inspect the Mac, understand memory, thermals, context,
model format, quant level, and task intent, then expose the best configuration
with evidence.

## Capability Doctrine

Every Apple Fit feature must obey these rules:

1. **No hidden throttles:** do not silently reduce tps, context length, batch
   size, precision, or quality to make an auto mode look stable.
2. **Auto chooses the strongest stable mode:** `--auto` means "find the best
   configuration for the declared intent," not "play it safe by default."
3. **Make downgrades explicit:** if Hawking changes context, KV policy, quant,
   batch, or model choice, it must say why and show the stronger alternatives.
4. **Always allow expert override:** users can force max speed, max quality,
   max context, max battery, or max capability even when Hawking recommends
   another mode.
5. **Protect against hard failure only:** memory-pressure actions exist to avoid
   OOM, swap collapse, request failure, or thermal collapse. They are not an
   excuse to cap performance preemptively.
6. **Restore capability after pressure clears:** any temporary downgrade must be
   reversible and visible.
7. **Gate auto modes against manual best:** no auto policy ships if it loses
   material speed, quality, or context versus the best known manual
   configuration without a stated user intent or hard resource constraint.
8. **Show the envelope:** whenever possible, report the frontier of options:
   fastest, highest quality, longest context, lowest energy, and safest fit.

## Product Thesis

Broad runtimes try to work everywhere. Hawking can be unfairly good on Apple
Silicon by treating the Mac itself as part of the runtime:

```text
current Mac -> fit planner -> model/quant/context/KV policy ->
measured serve profile -> adaptive but overrideable runtime
```

The winning claim is not "Hawking limits resource use." The winning claim is
"Hawking gets the most out of this Mac and tells you exactly what it chose."

## Work Packages

### A1 - Hardware And Runtime Profiler

Detect the real machine profile:

- chip family and OS version,
- unified memory,
- available RAM and pressure state,
- Metal device limits,
- thermal and power state where available,
- disk/scratch availability,
- active Hawking jobs.

Done means `hawking doctor --json` exposes enough data for repeatable fit
decisions.

**STATUS: 🟡 PARTIAL (2026-06-22).** `detect_mac()` (`crates/hawking/src/main.rs`) reads the real machine via `sysctl`:
chip (`machdep.cpu.brand_string`), unified memory (`hw.memsize`), OS (`kern.osproductversion`). MEASURED on this box:
"Apple M3 Pro | 18.00 GiB unified | macOS 26.6". Wired into `hawking doctor` — its swap-risk was **hardcoded to an 18 GB
M3 Pro** (`m3_pro_18gb_swap_risk`); now it is machine-relative (`fit_zone(total_working, detected_total)` → low/watch/high)
and prints a `machine:` line. **`doctor --json` ✅ LANDED (2026-06-22):** emits a machine-readable object (machine{chip,
total_unified_bytes,os} + model{layers,kv_heads,head_dim,context,...} + kv_cache_estimate + swap_risk) for repeatable fit
decisions. **REMAINING:** live RAM pressure + thermal/power state (A4); Metal device limits; disk/scratch; active-job detection.

### A2 - Fit Planner

Build a planner over model, quant, context length, KV policy, batch, and serve
mode:

- predict resident memory and scratch,
- predict KV growth by context and concurrency,
- mark likely swap/OOM boundaries,
- estimate warm tps, TTFT, quality tier, and energy when baselines exist,
- show alternatives instead of only one recommendation.

Done means `hawking fit <model>` reports the usable envelope for the current
Mac before serving starts.

**STATUS: ✅ MVP LANDED (2026-06-22).** `hawking fit --weights <gguf> [--intent <I>] [--max-context <N>] [--concurrency <C>]`
(`fit_main`). CPU-only (metadata + sysctl; no weights, GPU, or network). Reports: machine, model (arch/layers/kv-heads/
head_dim/native-ctx), the **context × KV-policy (f16/f32) fit envelope** with FITS/TIGHT/SWAP/OOM zones vs the detected
unified memory, envelope ceilings (longest stable context, highest-quality f32 ceiling, safest comfortable context, native),
and an **intent-driven recommendation** (max-capability default; max-context/max-quality/max-speed/max-battery/safe-fit) with
explicit alternatives + override hints. Capability-first per the doctrine: it reports the MAX envelope and never caps; safety-
biased intents (`safe-fit`, `max-battery`) print an explicit **anti-throttle note** naming the stronger max-capability option.
SSM models (rwkv7/mamba2) get the moat row: **flat recurrent state, no per-token KV growth → context bounded by quality, not
RAM**. MEASURED: Qwen2.5-3B on M3-Pro-18GB → 4k–32k all FITS (f16+f32), native 32k @ f32 = strongest stable (KV 2.25 GiB);
RWKV-7 → ~887 MiB resident at ANY context. Gates: `fit_tests` (kv_cache_bytes scaling, fit_zone thresholds) GREEN; build +
clippy clean. **REMAINING:** measured tps/quality/energy per cell (today fit predicts memory only; A6); safetensors is
rejected (use `hawking press`); the picks are heuristic until A4 live pressure + A6 measurements land.

### A3 - Capability-First Auto Serve

Add `hawking serve --auto` as a thin policy over the fit planner:

- `--intent max-capability`,
- `--intent max-speed`,
- `--intent max-quality`,
- `--intent max-context`,
- `--intent max-battery`,
- `--intent safe-fit`.

Default auto behavior should be capability-first: choose the strongest stable
configuration for the model and declared workload. Safety-biased modes must be
named as safety-biased.

Done means auto mode explains the selected model, quant, KV policy, context
cap, concurrency, and stronger/safer alternatives.

**STATUS: ✅ LANDED (2026-06-22).** `hawking serve --auto --intent <max-capability|max-context|max-quality|max-speed|
max-battery|safe-fit>` (`auto_serve_pick` + serve wiring, `crates/hawking/src/main.rs`). On startup it announces the machine,
the chosen ctx / KV / profile / energy, the rationale, and stronger/safer alternatives; explicit flags override; safety-biased
intents print an EXPLICIT downgrade line, capability-first ones print "anti-throttle OK". It applies the KV policy (F16_KV) +
`--profile`; the serve KV-capacity cap is currently advisory (noted in output). Enforced by A8.

**STATUS: ✅ MVP LANDED (2026-06-22).** `hawking serve --auto [--intent <I>]` (`auto_serve_pick` + Serve dispatch wiring).
On `--auto` it detects the Mac, reads model facts, picks the strongest stable config for the intent, **announces** machine +
chosen (ctx / KV / profile / energy) + the anti-throttle verdict, and **applies** the safe levers — KV policy
(`resolved_f16_kv`), `--profile fast` (max-speed), efficient energy (max-battery) — ONLY where the user did not set them
explicitly (expert flags always win). MEASURED: Qwen-3B `max-capability` → ctx 32768 @ f32 ("anti-throttle OK: strongest
stable"); `safe-fit` → ctx 32768 @ f16 + "EXPLICIT DOWNGRADE: max-capability would serve … @ f32 KV"; RWKV-7 → flat SSM,
full native context. **REMAINING:** context cap is announced but advisory (serve KV capacity is wired elsewhere); model
auto-selection across a models dir (A7); intent → batch-policy coupling.

### A4 - Memory Pressure Engine

Handle unified-memory pressure without hiding capability:

- warn before the machine crosses known bad pressure zones,
- prefer admission control and queueing before quality downgrades,
- optionally reduce context, concurrency, or KV precision only with visible
  policy,
- persist a pressure event in logs and metrics,
- restore the original target when pressure clears.

Done means Hawking avoids swap collapse/OOM while preserving explicit user
control and reporting every intervention.

### A5 - Long-Context Routing

Use the Apple Fit planner to choose between transformer and SSM paths:

- route long-memory workloads to RWKV/SSM when measured quality is acceptable,
- keep transformer paths for tasks where quality requires them,
- expose route decisions in the quality card,
- benchmark 8k/32k/128k regimes separately.

Done means Hawking has a measured answer for "which local model should handle
this long context on this Mac?"

### A6 - Energy And Thermal Cards

Make sustained local AI legible:

- joules per token,
- battery drain estimate,
- sustained tps after warmup,
- thermal throttling notes,
- plugged-in versus battery profiles when measurable.

Done means launch SKUs can report speed, quality, footprint, and energy instead
of only peak tps.

### A7 - Mac-Native Model Experience

Tie the planner into the product surface:

- `hawking pull`,
- `hawking models list`,
- `hawkingd`,
- LaunchAgent integration,
- content-based model discovery,
- clear unsupported-file remediation.

Done means a Mac user can install, pull, fit, and serve without manually
tracking paths, formats, or memory math.

### A8 - Anti-Throttle Regression Gates

Prevent the planner from becoming a performance ceiling:

- compare auto-selected runs against best manual profiles,
- fail if auto loses material tps/quality/context without a stated constraint,
- record every downgrade reason in reports,
- keep expert override tests green.

Done means future "helpful" policy changes cannot silently make Hawking weaker.

**STATUS: ✅ LANDED (2026-06-22).** Enforced by unit tests on `auto_serve_pick`:
`serve_auto_tests::auto_serve_never_hidden_throttle` (across 8–64 GiB Macs: max-capability never carries a hidden downgrade;
safety-biased intents serve ≤ capability and ONLY with an explicit `safety_downgrade` reason; native f32 is served when it
fits) + `serve_auto_tests::ssm_is_never_throttled` + `fit_tests::auto_pick_is_capability_first_and_anti_throttle`. A future
policy change that silently lost context/precision vs max-capability fails `cargo test -p hawking --bins`.

**STATUS: 🟡 PARTIAL — pick-level invariant LANDED (2026-06-22).** The anti-throttle rule is enforced IN the chooser:
`auto_serve_pick` never returns a config below max-capability for a non-safety intent without setting `safety_downgrade`
(the explicit reason); a hard-RAM reduction is the capability ceiling, not a flagged downgrade. Unit-tested
(`fit_tests::auto_pick_is_capability_first_and_anti_throttle`): max-capability → no downgrade + native @ f32; safe-fit /
max-battery → `safety_downgrade.is_some()` + context ≤ native; tight-RAM → f16 forced by hard limit (no false downgrade
flag); SSM → flat, full native. **REMAINING:** the *measured* regression gate (run auto vs best-manual serve and fail on
unstated material tps/quality/context loss) needs A6 measurements + a serving harness; expert-override e2e tests; wiring the
gate into `tools/ci/`.

## Final Success Statement

Hawking's Apple Fit frontier succeeds when a user on any supported Apple Silicon
Mac can run:

```bash
hawking fit <model>
hawking serve --auto --intent max-capability <model>
```

and receive the strongest stable configuration for that Mac, with clear
alternatives, no hidden throttles, overrideable choices, measured speed/quality/
footprint/energy, and graceful behavior under real memory pressure.
