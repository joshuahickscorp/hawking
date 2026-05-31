# C1 — Energy / joules-per-token baseline (§8 L4.2)

**Date:** 2026-05-31  **Machine:** Apple M3 Pro (Mac15,7), macOS 26.5 (Darwin 25.x)
**Step:** overnight queue Lane C / C1 — the branded "runs cool / sips power" axis.
**Status:** **HALT-WITH-FINDING — no sudo-free SoC/GPU power source available.**
Tool built (`tools/bench/measure_joules.sh`); baseline NOT measured (needs `macmon`
or an attended `sudo` run). One-command attended recipe below.

---

## The metric

```
joules_per_token (J/tok) = avg_power_W * decode_wall_s / tokens_generated
```

Reported alongside `dec_tps`. `decode_wall_s` and `tokens` come straight from the
`[stats]` line `generate` already prints on stderr
(`... completion=N decode_ms=M dec_tps=T ...`); the only missing input is
`avg_power_W` sampled over the decode window. Report **package power (CPU+GPU+ANE)**
and **GPU power** separately when the source can split them.

## Why this halted (power-source audit, all sudo-free options probed)

| Source | Gives | Sudo-free? | Usable now? |
|---|---|---|---|
| `macmon` (IOReport-backed) | SoC/GPU/ANE rail power (W) | **yes** | **NO — not installed** (`brew install macmon`) |
| `powermetrics` | true SoC/GPU/ANE power (W) | **no — needs password** | NO (unattended = no sudo) |
| `IOReport.framework` directly | per-rail energy counters | yes (framework calls) | NO — not `dlopen`-able from ctypes here; not in the dyld cache under a loadable path on Darwin 25.x. Would need a compiled Obj-C/Rust binary linking the private framework (out of scope: measurement-only step) |
| `ioreg AppleSmartBattery → PowerTelemetryData.SystemPowerIn` | **wall/adapter input power** (W) | yes | Readable + it *does* vary (saw 9.51 W → 4.08 W idle), **but it is whole-machine AC-input power** (SoC + display + losses + battery charging), not SoC/GPU. Confounded by screen brightness and charge state → not a valid decode-energy number. Rejected as the baseline source. |
| `sysctl` | only `kern.pervasive_energy=1` flag, no counters | yes | NO usable counter |

**Conclusion:** the only sources that yield true SoC/GPU package power on this Mac
are `macmon` (sudo-free, **not installed**) and `powermetrics` (**needs sudo**). An
unattended agent can supply neither, so the J/tok baseline cannot be captured this
run. This is a tooling-availability halt, **not** a Type-1/Type-2 lever kill — the
axis is real and flyable; we're one `brew install` away from measuring it.

## The tool (built, committed-ready, NOT run for real here)

`tools/bench/measure_joules.sh` — auto-detects the power source in preference
order (`macmon` → `powermetrics`), runs a steady-state Qwen-3B decode under the
**locked fast-path** env (`DISMANTLE_QWEN_TCB=1 VOCAB_PRUNE=32000 Q4K_LMHEAD=1
FFN_DOWN_Q4K=1 Q4K_PREDEC=1`), samples power in a background loop during the decode
window, and prints `dec_tps`, avg package W, avg GPU W, decode energy (J), and
**J/token**. `--f16s` additionally runs the A6.5 `DISMANTLE_QWEN_PREDEC_F16SCALES=1`
lever and prints the J/tok + tps deltas (the "finish-sooner-and-idle" question:
does the +6–9% faster path also lower J/tok?).

Unattended-safe: if only `powermetrics` is present it refuses to invoke sudo
non-interactively (exits 4 with the recipe); if nothing is present it exits 3.
Verified both halt branches fire correctly; syntax clean.

## One-command attended recipes (pick one)

**Preferred — install the sudo-free reader, then no password is ever needed:**
```bash
brew install macmon
tools/bench/measure_joules.sh --tokens 256 --f16s
```

**Or — keep powermetrics, just pre-cache the sudo password once:**
```bash
sudo -v && tools/bench/measure_joules.sh --tokens 256 --f16s
```
(`sudo -v` caches the credential so the background sampler can spawn
`powermetrics` without a second prompt. The script auto-selects `powermetrics`
when `macmon` is absent.)

Either prints, for the locked baseline and the f16s-on lever:
```
dec_tps / tokens / decode_wall_s / avg pkg power (W) / avg GPU power (W)
decode energy (J) / >> J/token <<
... and the J/tok + tps % deltas between the two.
```

## Expectations to sanity-check the first real run against

- M3 Pro decode is **bandwidth-bound** (per A4–A10: GEMV at the memory-model
  optimum, ~56% peak BW on the dominant `_pair` kernel). Expect package power
  well under the M3 Pro's ~30–40 W sustained ceiling — likely the **~8–18 W**
  band during steady decode, GPU rail the dominant component.
- **f16s (A6.5)** cuts scale-byte traffic and is +6–9% dec_tps. If decode power
  is roughly flat (same kernels, just fewer bytes/token), J/tok should drop by
  **about the same % as the tps gain** (energy ≈ power × time; faster finish at
  near-constant W ⇒ lower J/tok). That would make f16s a *both-axes* win
  (faster **and** cooler) — worth stating explicitly in the bible's L4.2 entry.

## Files

- `tools/bench/measure_joules.sh` (new) — the reusable J/tok harness.
- `reports/c1_energy_baseline.md` (this file) — finding + formula + recipe.

No source/kernel changes (measurement-only step). Nothing committed.
