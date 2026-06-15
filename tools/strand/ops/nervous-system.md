# The nervous-system ops layer — organ map

_2026-06-11. The ops layer convergently evolved structures that look like brain organs.
This document names them, specifies the three formalized in the nervous-system wave
(habituation, salience, replay), and is honest at the bottom about which analogies are
load-bearing engineering and which are just good names. No mysticism: every mechanism
here is either EXACT (bit/byte-gated) or a logging/scheduling policy with explicit rules._

## 1. The organ map

| organ (analogy) | mechanism (the actual code) | status |
|---|---|---|
| **Afferent signaling** (sensory nerves) | `ops/conductor.sh` event wakes: `EVENT: …` + exit → the session wakes to interpret. Every decision derived from the filesystem each tick (stateless-resumable). | existing (v6), now salience-tagged (v7) |
| **Homeostasis** (autonomic regulation) | the governors: `ops/pod-governor.sh` (pod memory/process), conductor's POD_MEM guard, the MPS watermark-ratio launch recipe. Mechanical corrections without waking judgment. | existing |
| **Sleep / idle work** | conductor idle windows: the definitive speed sweep (#4), the ternary low-LR filler (#4b), and now replay (#5b). All gated on the same pgrep idleness discipline. | existing + extended (v7) |
| **Cortical consolidation** (long-term memory) | `will.md` — the inheritance document, updated at every milestone; auto-memory files as the ledger index. | existing |
| **Motor system** (deliberate action) | `ops/podctl.sh` — the single audited pod console (constructed patterns, tree kills). | existing |
| **Episodic record** | `research/results-ledger.jsonl` + `strand-eval ledger check` (the 15-digit tell as code). | existing |
| **HABITUATION** (sensory gating) | v7 conductor: event-fingerprint memory in `scratch/.conductor-habituation`. A repeated S2 fingerprint within 6 h logs-but-does-not-wake unless severity escalates. Rules in §2. | **new (this wave)** |
| **SALIENCE** (attention triage) | v7 conductor: every event tagged S1/S2/S3 (taxonomy in §3). | **new (this wave)** |
| **REPLAY / dreaming** (the immune system) | `ops/replay.sh`: idle-gated re-run of the cheap invariant suite — bit-rot detection between milestones. §4. | **new (this wave)** |

## 2. Habituation — the rules (conductor v7)

- **Memory:** `scratch/.conductor-habituation`, one line per fingerprint:
  `<fp> <epoch_last> <suppress_count> <max_sev> <class> <key…>`. Pruned to the window
  on every write. Persists across conductor exits — that is the point: the
  wake→relaunch cycle no longer resets the memory.
- **Fingerprint** = `md5("CLASS|normalized key")[0:12]`. Keys normalize digits → `N`
  where the key is log text (so timestamps/counters don't defeat the match); VERDICT
  keys on filename + content hash, so a genuinely new verdict is always a new
  fingerprint and always wakes.
- **Window:** 6 h (`HAB_WINDOW`). Repeat outside the window = fresh cycle (wakes).
- **Escalation (a repeat wakes anyway when):**
  1. the event's numeric **severity grew** past the recorded max (rc-line count,
     ledger ERROR count, pod json count, chain-failure count), or
  2. **safety valve:** the 10th suppression (`HAB_MAXSUP`) of one fingerprint in a
     window wakes once and resets the fingerprint — nothing is silenced forever, or
  3. the class is in the **always-wake list** (= S1, below) — those never enter
     habituation at all.
- Every suppression is logged (`HABITUATED <class> xN`) and counted into the daily
  digest. Tested: `bash ops/conductor.sh selftest` (5 rules, isolated temp files).

## 3. Salience classes — the explicit event taxonomy

| class | meaning | events |
|---|---|---|
| **S1** | wake ALWAYS, never habituated | `REPASS_EXHAUSTED`, `POD_UNREACHABLE`, `POD_MEM`, `REPLAY_FAIL` |
| **S2** | wake, habituatable (fingerprint + severity per §2) | `VERDICT` (key: json name+content md5), `FAILURE` (key: rc-line class; sev: rc count), `STALL_KILLED` (key: log name), `SPEED_SWEEP_DONE`, `POD_MILESTONE` (key+sev: json count), `POD_LADDER_EXITED`, `POD_CHAIN_FAILURE` (key: failure-line class; sev: line count), `LEDGER_TELL` (key: first ERROR class; sev: error count) |
| **S3** | log only; one daily digest line | `REPLAY_PASS`, every `HABITUATED_*` suppression; routine pod status polls were already log-only and stay that way |

The daily `DIGEST` line (once per calendar day in `scratch/conductor.log`) summarizes
24 h of S3 events plus the live habituation table.

## 4. Replay — the idle invariant sweep (`ops/replay.sh`)

Scheduling: the conductor launches it at most once per **quiet 6 h window**
(`scratch/.replay-last` stamped at launch; same pgrep idleness discipline as the other
idle jobs). The script self-gates again (idle check + pid lock), so running it by hand
is always safe. Verdict → `scratch/.replay-verdict`; the conductor maps
**PASS → S3** (digest line) and **FAIL → S1 wake** (bit-rot detected = maximum salience).

The suite (all EXACT pass/fail, measured 18 s on the idle M3):
1. `gate-kernels` — the 5,380-cell decode-path identity registry (byte-identity law).
2. `cargo test -p strand-quant --lib provenance` — the SPV3 hash KATs (must match >0 tests).
3. `attest-strand` on the shipped artifact (`scratch/artifacts/qwen05b-pv2-2bit.strand`)
   — real mmap consumer path, fixed==lean decode assert, **stored SPRV root ==
   independently recomputed model root**.
4. `strand-eval ledger check` — 0 ERROR lines.

Honesty rules: a SKIP (missing binary/artifact) is not an invariant failure, but a
replay where everything skipped **verifies nothing → verdict FAIL**. One JSONL record
per completed replay is appended to `research/results-ledger.jsonl` with
`harness_key=replay` and **no ppl field** — `ledger check` emits a benign
`record without ppl` WARN for each (replay records never enter canon comparisons or
the 15-digit grouping; the WARN is the known cost of not touching `ledger.py`).

## 5. Honesty section — which analogies are load-bearing

**Load-bearing (the analogy changed the engineering):**
- **Habituation** — the brain principle (repeated identical stimuli stop propagating;
  novel or intensifying ones break through) maps 1:1 onto the fingerprint + severity-
  escalation rules. Before v7 the conductor woke the session on every repeat of the
  same failure class; the structure of the fix is exactly sensory gating.
- **Salience** — triage-before-attention is the real design constraint: session wakes
  are expensive (a whole judgment context spins up), logs are cheap. S1/S2/S3 is an
  attention budget, mechanically enforced.
- **Replay as immune system** — the body re-checks self-integrity during quiet
  periods, not during exertion; replay re-runs the identity gates during idle windows
  and treats any failure as maximum salience. The scheduling (quiet windows only,
  cheap suite, loud on anomaly) follows from the analogy.

**Decorative (good names, no engineering content):**
- "Dreaming" for replay — replay re-executes deterministic checks; it does not
  recombine or consolidate anything. The brain's replay does; ours is a smoke test
  with a poetic name.
- "Cortical consolidation" for will.md — it's a documentation discipline. Nothing
  about memory biology constrains its format.
- "Afferent/motor" for conductor/podctl — accurate as a direction-of-signal metaphor,
  but they'd be designed identically without it.
- The daily digest as "memory consolidation during sleep" — it's a log-rotation
  summary line. Do not build on the metaphor.

The framing earns its place only through the gates: every new mechanism above is
either bit/byte-exact (replay's four checks), mechanically testable
(`conductor.sh selftest`), or a logging policy with explicit numeric rules
(6 h window, sev-growth, x10 valve). Where an analogy suggested a behavior we could
not gate, it stayed decorative and is listed as such.

## 6. Autotune — the cerebellum (idle motor tuning)

_2026-06-11._ The cerebellum tunes motor gains in the background without conscious
attention — that one sentence is the whole analogy; the rest is engineering. Organ-map
row: **AUTOTUNE (cerebellum)** | `ops/autotune.sh` + `tools/autotune/` →
`research/tuned-profile.toml` | new (this wave).

We hand-picked every performance constant once (rayon decode threads, interleave `S`,
`quantize-model --threads`, `KD_CHUNK=128`, eval `OMP_NUM_THREADS`, `EVAL_GPU_GB`).
The autotuner re-learns them per-machine while the box dreams:

- **Registry** (`tools/autotune/tunables.py`): each tunable declares its candidate
  values, the measurement command, the metric parser, and a **guard** that must pass
  for a sample to count — for decode tunables that is the gate bin's own bit-identity
  assertion (the bins refuse to print perf on any Q12 drift, will.md §5.2); for
  `encode_threads` it is additionally a **result-invariance fingerprint** (per-tensor
  rel-RMS, sorted) that must be identical across every thread count — a tunable that
  changes the output is a bug, and the sweep verdicts FAIL, not "tuned". Encode runs
  force `STRAND_NO_GPU=1` (CPU-only by contract).
- **Engine** (`tools/autotune/sweep.py`): plain grid / one-run-batch (the spaces are
  tiny), best-of-`--reps`, memoized shared runs, machine fingerprint
  (hw model + CPU + cores + mem + OS + rustc, sha256/16). Missing gate bins (the
  sibling speed wave owns them and may have them mid-rebuild) → per-tunable SKIP,
  never a failure; all-skip → "nothing runnable", no profile (the replay honesty rule:
  nothing measured must not read as a profile).
- **Scheduler** (`ops/autotune.sh`): replay.sh's exact discipline — same
  bracket-constructed pgrep idle gate (+ replay itself), pid lock, at most one sweep
  per quiet 6 h window (`scratch/.autotune-last`; `AUTOTUNE_FORCE=1` bypasses the
  window, never the idle gate), verdict → `scratch/.autotune-verdict`, one JSONL
  ledger record per completed sweep (`harness_key=autotune`, no ppl field — the same
  benign `ledger check` WARN as replay records).
- **THE CONTRACT:** the tuner **never changes a default in code**. Its only output is
  `research/tuned-profile.toml` (`meta.advisory = true` — every timing is advisory and
  machine-stamped). Consumers **opt in** by reading the profile and checking the
  fingerprint; the executable form of the pattern is `tools/autotune/apply.py`, which
  prints the recommended env/flags per known launcher (`RAYON_NUM_THREADS=…`,
  `--threads …`, `S=…` per deploy point) and falls back loudly to the hand-picked
  default on any non-TUNED tunable or fingerprint mismatch.

Defined-but-disabled tunables (the census stays complete even where the sweep can't
run): `kd_chunk` (needs an MPS training step — QAT runs ALONE on the 18 GB box, §7
freeze trap), `eval_omp_threads` (needs the torch eval stack, minutes/sample),
`eval_gpu_gb` (pod-side only — the local profile must never claim it).

First sweep (2026-06-11, M3 Pro fingerprint `873ad7d3d378c86b`, advisory): decode
rayon threads → 12 best (852 → 4322 Mw/s, monotone — the E-cores do pay their way);
interleave S=4 at both k=3/L=7 and k=2/L=12 (confirms the hand-picked default);
encode threads → 12 (19.4 s → 2.6 s on the 24-tensor probe), invariance fingerprint
identical across all 7 thread counts. Honesty note: this analogy is **load-bearing
only in the scheduling** (tune during idle, never during exertion — same clause as
replay); the coordinate-grid sweep itself owes nothing to neuroscience.
