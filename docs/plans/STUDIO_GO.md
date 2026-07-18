# STUDIO GO — the one-command entry point for the Hawking frontier program

> Paste target: Hawking is now on the Mac Studio. Establish the machine and efficiency receipts,
> review the safe ladder plan, then tell the coding agent "go". Heavy work remains gated and resumable.

The active v1/v2 model ordering and maximum-size handoff are recorded in
[`STUDIO_MODEL_LADDER.md`](STUDIO_MODEL_LADDER.md). The canonical next research system is
[`DOCTOR_V5.md`](DOCTOR_V5.md) with [`TRAINING_LADDER_V5.md`](TRAINING_LADDER_V5.md). This operating
guide controls current launch mechanics and prevents planned v5 identities from being mistaken for
scheduled or completed work. [`CONDENSATION_DOCTOR_V2.md`](CONDENSATION_DOCTOR_V2.md) remains the
historical, currently detached observation boundary; it is not the canonical scientific plan.

## STEP 1 — preflight (always run first)

```
cargo run -p hawking --bin hawking -- studio preflight
```

Checks Python deps, Rust toolchain, RAM/disk, the HIDE app Node/pnpm engine, that every
`tools/condense/*.py` compiles, that `cargo check --workspace` is clean, which model parents are staged,
that the frontier refresh ledger and
refreshed HF metadata ledger can be written, that the model-aware frontier launch gate is green against
that refresh artifact, that `reports/condense/studio_preflight_summary.json` is written with a canonical
SHA-256 signature over check results plus machine/network/power/thermal/developer-environment evidence,
and that the receipt harness verifies.
Exits 0 (green, safe to `go`) or 1 (red, prints exactly what to fix). Do not run `go` on a red preflight.
The product-facing command delegates to `tools/condense/preflight.py`.
Verify a saved summary with:

```
cargo run -p hawking --bin hawking -- studio verify-summary --path reports/condense/studio_preflight_summary.json
```

## STEP 1B — signed environment receipt

```
cargo run -p hawking --bin hawking -- studio environment-capture --out reports/condense/studio_environment.json
cargo run -p hawking --bin hawking -- studio environment-verify --path reports/condense/studio_environment.json
```

This is a no-download proof-environment capture. It records the target machine class, RAM/disk envelope,
CPU/model identifiers, Hugging Face DNS/route/API reachability, expected link budget, power source,
thermal status, and powermetrics availability. The receipt is independent of preflight so benchmark,
RAM-cliff, and claim bundles can point to a stable signed "this exact Studio was ready" artifact.

The delivered machine is an **M3 Ultra Mac Studio with 96 GB unified memory, 819 GB/s advertised
memory bandwidth, and a 1 TB SSD**. The operator budget is never the nominal SSD size: derive it from
current free space, leave **150 GB untouched**, and budget **64 GB scratch + 32 GB HF/Xet cache** inside
the remaining space. Regenerate the signed environment, preflight, ledger, launch-packet, and claim-wall
artifacts on this machine; receipts captured earlier on the M3 Pro or against an M1 Ultra profile are not
same-box evidence.

## STEP 1C — efficiency charter and E0 baseline (before the training ladder)

Read [`computational_efficiency_paradigms_2026_07_11.md`](computational_efficiency_paradigms_2026_07_11.md).
It governs this run alongside the proof discipline below. FLOPS and raw tok/s remain diagnostic counters,
not the objective. Every promoted result must preserve the declared capability gate and report, where the
platform exposes them:

- useful/correct completions per joule and joules per accepted token;
- useful capability per byte moved and per resident byte;
- useful capability per resident and active parameter, with routed/conditional parameters reported
  separately from total installed parameters;
- SLO goodput, TTFT, inter-token latency, p50/p95 wall time, and output length;
- model, KV, cache, SSD, and network bytes, plus rejected speculative work and recomputation;
- peak unified memory, memory-pressure state, swap delta, thermal state, and current free disk.

**E0 is first:** capture a no-training baseline for cold load, prefill, warm decode, long-context decode,
and the existing same-box baseline set before interpreting ladder wins. The first implementation may use
analytical tensor-byte accounting plus whole-system energy sampling, but must label estimates separately
from measured counters. Target `<2%` measurement overhead and phase-level accounting closure within 10%.
An apparent win that merely shifts work into SSD, CPU, rejected drafts, longer output, or quality loss is
not a win.

## STEP 2 — THE COMMAND

```
python3.12 tools/condense/studio_run.py go
```

Runs the guarded program in the foreground against the **96 GB** envelope. For the normal phone/remote
workflow, use the detached supervisor controls instead:

```
python3.12 tools/condense/studio_run.py start       # detached + caffeinated; resumes checkpoints
python3.12 tools/condense/studio_run.py --status    # resource + phase + move-safety JSON
python3.12 tools/condense/studio_run.py drain       # stop launches, checkpoint, fsync
python3.12 tools/condense/studio_run.py resume      # clear a completed drain and continue
python3.12 tools/condense/download_queue.py start  # detached 32B -> 72B -> 14B barrier -> 120B -> gated 284B -> gated 1.1T
python3.12 tools/condense/download_queue.py status # queue, pressure, thermal, disk, marker state
python3.12 tools/condense/processing_queue.py start  # detached verified-source -> admitted quant/Doctor work
python3.12 tools/condense/processing_queue.py status # promotion, coverage, pressure, and hold reasons
python3.12 tools/condense/frontier_stream_queue.py start   # detached 32B -> 72B -> 120B representative-shard research
python3.12 tools/condense/frontier_stream_queue.py status  # explicit oracle-only claim limits and 14B barrier
python3.12 tools/condense/terminal_frontier_queue.py start # read-only 120B -> 284B -> 1.1T -> remote-stream 1.6T schedule
python3.12 tools/condense/terminal_frontier_queue.py status # gates only; it never launches an unwired worker
python3.12 tools/condense/doctor_frontier_queue.py start    # detached v2 observer; pins plan, launches no experiment
python3.12 tools/condense/doctor_frontier_queue.py status   # Studio owner + lease + resource + launchability view
python3.12 tools/condense/doctor_v5.py validate reports/condense/doctor_v5_campaign.json
python3.12 tools/condense/quality_battery_v5.py validate reports/condense/quality_battery_v5.json
python3.12 tools/condense/training_ladder_v5.py validate reports/condense/training_ladder_v5.json
python3.12 tools/condense/doctor_v5_root.py validate reports/condense/doctor_v5_root.json
python3.12 tools/condense/doctor_v5_audit.py audit --output reports/condense/doctor_v5_audit.json
python3.12 tools/condense/doctor_v5_queue.py plan    # fixed 0.5B -> 120B bootstrap plan
python3.12 tools/condense/doctor_v5_queue.py start   # detached, caffeinated, checkpointed
python3.12 tools/condense/doctor_v5_queue.py ping    # read-only state/resources/ETA snapshot
python3.12 tools/condense/doctor_v5_queue.py drain   # checkpoint-safe stop request
python3.12 tools/condense/doctor_v5_queue.py resume  # resume after drain/reboot
python3.12 tools/condense/doctor_v5_queue.py inject pause
python3.12 tools/condense/doctor_v5_queue.py inject resume
```

### Doctor-v5 detached scale controller

The current Ultra execution ladder is `tools/condense/doctor_v5_ultra_queue.py`.
It declares an exact 320-cell matrix across eight source tiers
(`0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B, 120B`), ten physical ceilings
(`4, 3, 2, 1, 0.8, 0.55, 0.5, 0.33, 0.25, 0.1` bpw), and four branches
(`codec_control`, `doctor_static`, `doctor_conditional`, `doctor_full`). The 280
Qwen2.5 cells have reviewed executable adapters. The 40 GPT-OSS 120B cells remain
addressable but fail closed until a whole-file MXFP4/MoE execution adapter,
reassembly provenance, tokenizer/evaluator path, and lifecycle are reviewed; they
must never be counted as executed merely because the 120B source is installed.

The first sweep is one preliminary replicate per cell, not a dominance claim. The
Doctor branches are source-independent re-encoding/treatment candidates, the
conditional branch explicitly records that activation-conditioned dispatch is not
implemented, and 32B/72B quality evaluation is deferred. Physical accounting and
attestation remain valid evidence; absent quality observations remain null rather
than being inferred. A later front-to-back rerun may raise the replicate/seed and
Doctor ceiling after this scale map is reported.

Progression is detached, caffeinated, checkpointed, and automatic. Every terminal
cell is synchronized into the source-bound reporter before its packed payload can
be reclaimed. Garbage collection is pre-admission and limited to an exact hashed
`.strand` allowlist, so the next treatment/rate does not require duplicate packed
residency. Original model sources are retained. Reports are one consolidated
sub-120B bundle (280 cells) and one isolated 120B bundle (40 cells); each requires
an independently validated terminal checkpoint. ETA is empirical by branch and
becomes available after the first completed sample in all four branches.

The supervisor samples the complete child process group every five seconds and
stops at 78 GB RSS; pressure, swap, AC power, thermals, and disk admission are
rechecked every 30 seconds. Disk keeps a 150 GB reserve. Source, adapter, reporter,
and codec identities are re-hashed, so this campaign-owning worktree must remain
immutable after launch. Shipping cleanup must use a separate worktree and build
directory. The generated campaign, status, receipts, report ledger, and exact
resume command under `reports/condense/doctor_v5_ultra/` are runtime authority.

The older controller description below is retained as historical background; it
is not the Ultra matrix authority.

The v5 controller is a separate post-greenlight execution boundary; it does not mutate the immutable
Doctor-v5 planning campaign or relabel v1/v2 rows. Pass A reads and hashes source bytes without loading
model tensors, produces a restart journal and bootstrap census for the fixed
`0.5B, 1.5B, 7B, 14B, 32B, 72B, 120B` cohort, and publishes two report bundles: one aggregate sub-120B
bundle and one isolated 120B bundle. The official 120B `original/*` source is pinned by exact revision,
ordered paths, sizes, Git/LFS identities, and a 65,248,927,631-byte manifest. Its download may overlap
the earlier census only while the live gate preserves 150 GB immutable free space, 64 GB processing
scratch, 32 GB cache, and 2 GB control overhead. `start` launches or adopts that pinned transfer at
the beginning of Pass A rather than waiting for the serial 120B rung; a shared inherited singleton
lease prevents a replacement supervisor or a second detached procure command from duplicating it.

`ping` is on-demand and read-only; no Codex/chat observer is required. Injections are append-only,
hash-chained operational controls consumed at wave boundaries. They are not scientific authorization.
The Python process survives app/terminal detachment; a reboot stops it, after which `resume` revalidates
the source and reuses stat-identical completed shard-journal rows before a mandatory final full-source
revalidation. A pause stops queue advancement while an already detached download may continue; a drain
checkpoints and stops both the queue and its queue-owned transfer. `resume` records a durable resume
transition so an earlier unconsumed drain row cannot immediately drain the replacement. The state
explicitly does not claim launchd autostart or bit-exact legacy optimizer continuation.

Pass B remains `waiting-treatment-adapters` until a concrete source-hashed adapter, exact-resume replay,
role-separated receipts, and a per-ceiling whole-artifact/resident-memory budget validate. A source census
is not a quality result. The operator has authorized a later all-at-once cleanup after both the sub-120B
aggregate **quality** bundle and the isolated 120B **quality** bundle are durable and verified; until then,
automatic source deletion remains false. This prevents a compact inventory report from destroying the
parents required for the actual front-to-back Doctor-v5 run.

`start` detaches a low-overhead admission waiter; it does not hold the heavy lease while pressure,
swap, disk, or another CPU/RAM-heavy process keeps the gate red. Direct `go` and `resume` fail closed.
The waiter and the running GO have separate PID identities, both are drain/unplug-visible, and current
code is frozen before evidence promotion. A managed queue/procure/HF tree may overlap compute only when
its exact schema, command, path, ancestry, heartbeat, and RSS probes pass continuously. The downloader
yields checkpoint-safely if `65 + max(10, measured-download-GiB) + 2 > 78 GB`.

The Doctor-v2 queue is intentionally narrower than the other supervisors. It checkpoints its pinned
historical campaign identity and the next advisory choice, but `invoke_workers` is permanently false
in that scaffold. V5 does not silently replace that durable hash or migrate the active owner. Its
campaign, quality battery, ladder, package root, and audit are execution-free planning artifacts:
32,768 v5 candidates and 1,280 v5 lanes remain explicitly unwired and unlaunchable until a later
green light, exact source parameter manifests, role-separated trusted receipts, reviewed adapters,
source/data bindings, and resource admission exist.

The unattended download chain is bounded and fail-closed. It reconciles/verifies the staged 32B source,
downloads and verifies the 72B BF16 parent, waits for both 14B quantization/Doctor lanes, and then fetches
OpenAI's `gpt-oss-120b` official `original/*` MXFP4 checkpoint. It next exposes DeepSeek-V4-Flash
(284B/13B) as the lower-cost architecture bring-up, then the terminal Kimi-K2.6 install (1.1T repository,
1T declared model / 32B active, 595.2 GB native compressed-tensors). Each remains in a low-overhead
`planned-blocked` wait until its own explicit architecture-ready receipt and the live disk gate both pass.
All downloads remain isolated under
`scratch/staging/*.partial`; download completion is not memory admission. The queue refuses a new
transfer unless pressure is normal, swap is below 2 GB, AC and thermals are green, and free disk covers
the remaining source plus the 150 GB hard floor and a 32 GB transient-cache reserve. Processing has a
separate 64 GB scratch-admission gate and is never implied by download completion. The queue shares the Studio
drain request and terminates its active child checkpoint-safely on pressure, thermal trouble, or drain.
The queue never deletes sources. Storage reclamation remains blocked until a durable artifact is bound to
its verified load/quality receipt; applicable license/review gates remain independent.

The run is resumable: only phases and models with durable pass manifests skip; a phase left `running`
after power loss reruns from its underlying completed-config checkpoints. Dry-preview first with:

```
python3.12 tools/condense/studio_run.py --go-plan
```

The safe training order is **0.5B -> 1.5B -> 7B**, then **14B alone** after the live memory oracle is
green. GO reads 14B directly from its verified staging path under a marker hash + mutation-sensitive
model fingerprint. The detached processing queue later publishes the canonical link and accepts only
the same current studio/sub-bit receipts; if work is absent it takes the shared heavy lease and runs it.
A pre-120B barrier preserves temporary disk until both 14B lanes validate. The **32B full Doctor/PPL
rung is durably deferred** at the 85 GB estimate versus the 78 GB budget; its current safe execution is
the path/marker/shard-SHA-bound representative-shard frontier VTQ queue. Those rows are reconstruction
oracles, not deployable floors. Do not let a nominal `solo` flag override the live gate.

## LIVE QUANTIZATION INTERPRETATION AND HONEST CAPACITY CEILINGS

The ladder is already the requested quantization experiment, not a download-only exercise. Its studio
lane measures uniform 4/3/2/1-bit AWQ, mixed precision, residual coding, and Doctor recovery from the
3/2/1-bit bases. Its **true sub-bit** lane is now a vector-trellis (VTQ) reconstruction-oracle campaign:
one `k`-bit symbol reconstructs `d` weights, so the symbol payload is `k/d` bits per weight before
scales, LUTs, padding, framing, unquantized tensors, and any Doctor adapter are charged. Mandatory rows
include `d={2,3,4,8}`, frozen-versus-learned codebooks, iso-payload controls (`k1/d2` versus `k2/d4`,
and `k1/d4` versus `k2/d8`), a side-information/block-length sweep, and rank-8 Doctor recovery.
The canonical VTQ identity is the **raw `awq_alpha=0.0`** parent transform with column RHT; AWQ sigma
scaling is disabled for this campaign because unbilled/out-of-band sigma state would change the recipe.
Model completion now fails closed unless every required row has either a successful measured record or a
durable bounded-negative record. Operational completion advances the queue; scientific promotion still
requires every quality, packing, runtime, residency, and efficiency gate below.

Three things that used to be called "sub-bit" are not equivalent:

- `subbit.py measure` / **SUBBIT-0-THEORY** is an order-0 symbol-entropy lower bound plus logical
  side-information accounting. Its `k=1` probe now uses a true binary sign alphabet (the old
  ternary-as-one-bit calculation is invalidated). It does not implement bulk-symbol rANS or include all
  framing/pages/codebooks/unquantized weights, and is stamped `product_gate=false`, `deployable=false`.
  It may reject an impossible direction; it cannot promote one.
- `subbit.py ladder` is footprint arithmetic only. A `0.50` or `0.33` row means "this many bits would
  fit if a complete artifact existed," not that such an artifact has been built, restored, or served.
- VTQ `d2/d3/d4/d8` rows are real full reconstruction/evaluation experiments with exact oracle
  accounting, but are still `artifact_class=reconstruction_oracle`, `deployable=false`. Packed `.tq`
  v2 does not yet serialize the required per-tensor vector LUT records, and Hawking's native/GPU path is
  scalar-only. The oracle therefore rehydrates reconstructed weights for quality measurement; it proves
  neither compressed residency nor speed.

The first persisted single-tensor control (Qwen2.5-0.5B `q_proj`, 802,816 weights, block 256,
learned-codebook L5) recorded encoder-stream bpw of `0.8321/0.6681/0.5821/0.4571`; charging the
complete per-tensor SDSC-v2 Q12 vector-LUT record for every vector tensor—the 52-byte descriptor/hash envelope plus every i32
entry—raises the exact logical oracle values to `0.8352/0.6724/0.5878/0.4679` for
`d2/d3/d4/d8`, with `79.94%/86.62%/89.46%/94.08%` relative-RMS reconstruction error. Those are useful
negative/scale signals, not model-quality results: the mandatory full-model multiwindow PPL and
capability tripwire decide whether Doctor restored useful behavior. The gap between symbol payload and
effective bpw is exactly why payload-only labels may never be used as capacity claims.

The storage geometry is nevertheless worth testing. A projection-only, no-outlier physical-v2 estimate
for 7B is about `1.188/0.938/0.813` bpw at block 256 for `d2/d4/d8`; amortizing logical side information
at block 8192 projects `0.704/0.454/0.329` bpw. Thus `0.50` is arithmetically plausible and `~0.33` is
barely plausible at 7B-scale geometry, but blocks above 256 are currently ragged CPU/oracle research and
do not match the fixed GPU subscale layout. The smaller 0.5B control has an estimated `~0.341` padding
floor at the nominal one-third point. These projections prioritize the block sweep; they do not supersede
actual file-bpw, reconstruction-quality, or runtime receipts.

Interpret current results in three different envelopes:

| Envelope | Honest current ceiling | Why |
|---|---:|---|
| Largest current quality-measured rung | 7B | 7B has reached 4.846 effective bpw at +1.32% PPL; the full ladder and Doctor recovery are still in progress. |
| Safely admitted raw + quantize + Doctor | 14B | Estimated 65 GB solo peak inside the 78 GB interactive process budget. |
| Raw BF16 weight-only arithmetic | 39B | `78 GB / 2 bytes`; excludes KV, activations, conversion buffers, the OS, and the GPT/Codex tenant. It is not an execution promise. |
| Present native Hawking `.tq` implementation | roughly 16--17B all-linear equivalent, still unproven end-to-end | The loader retains a decoded i32 Q12 copy while also allocating compressed/GPU state, so artifact bpw is not resident bpw. |
| Compressed-artifact mathematics only | 139B at 4.5 bpw; 187B at 3.34; 267B at 2.34; 466B at 1.34; 624B at 1.0 | These are storage equations, not runnable-model claims, until the retained Q12 copy is removed and direct compressed serve is receipt-proven. |

The current evidence says scale helps at approximately four bits: 0.5B and 1.5B miss the +2% PPL gate,
while 7B passes it. Uniform one- and two-bit PTQ catastrophically collapse the small dense controls;
three-bit retains substantially more signal but still misses the gate. Mixed precision is better than
uniform two-bit, but overhead matters: a nominal four-bit candidate is about 4.8 effective bpw and a
nominal one-bit candidate is about 1.64 effective bpw. None of this demonstrates restoration yet; that
claim waits for completed Doctor + capability-tripwire evidence.

Doctor is observed, not assumed. Use `python3.12 tools/condense/doctor_status.py --pretty` for an
on-demand snapshot; do not keep a chat/agent attached merely to watch it. A future VTQ Doctor row starts
at **rank 8** and charges the exact serialized adapter bytes against the baker's exact quantized-weight
count. Rank 16 is allowed only when rank 8 improves held-out quality enough to justify its added bpw;
rank 64 is not the default. Earlier rank-64 rows that used the old denominator materially underbilled
small-model adapters and are invalid density evidence until rerun. The Doctor must retain the exact
pre-update zero-correction adapter as its fallback, record held-out progress, distinguish optimizer updates from microsteps, fsync checkpoints and
heartbeats, and fail closed on missing/stopped final evidence. A run whose best held-out point is the
parent or whose recovery misses the gate is a **complete negative experiment**, not a pass and not a
reason to wedge the detached queue.

Current Doctor restart safety is at the **configuration boundary**, not the optimizer step: an
interrupted Doctor config is rerun, while the atomic latest/best adapter remains useful evidence and a
fallback. Exact optimizer tensors, RNG/sampler state, and microstep position are not restored today, so
the ladder must not describe a reboot as bit-exact continuation or silently promote a partial run.

### Promotion ladder: reconstruction is not deployment

A candidate can advance only one evidence class at a time:

1. **Reconstruction oracle:** exact effective bpw, reconstruction error, multiwindow PPL, and a
   same-parent capability tripwire. This is where current VTQ rows stop.
2. **Packed round trip:** serialize every required learned LUT and all side information, reload without
   an out-of-band Python object, reproduce the oracle reconstruction, and measure actual file bpw.
3. **Runtime parity:** the production CPU path and then Metal path match the packed reference across
   deterministic tokens/logits; missing vector coverage is an error, never scalar/GGUF fallback.
4. **Resident execution:** no decoded i32/f16 parent copy survives; peak unified memory and bytes moved
   agree with the packed representation plus declared scratch/KV state.
5. **Capability-efficiency promotion:** multiwindow PPL and the f16-relative task tripwire pass, then
   same-box cold/warm latency, accepted-token energy, bytes moved, and pressure receipts beat the best
   matched baseline. A smaller file alone is not a win.

The task tripwire is deliberately relative to a captured f16 result for the same parent. A candidate may
lose at most one of 22 items overall and at most one item in any task family; missing or malformed
tripwire evidence blocks promotion. Negative rows remain in the ledger so the scaling curve is not
survivorship-biased.

Downloaded is not runnable. The verified 32B parent is a legitimate future streamed rung but its current
full processing estimate is 85 GB. The 72B BF16 parent cannot reside in 96 GB. The 120B source is native
MXFP4 gpt-oss rather than a raw BF16 Qwen parent, and needs a gpt-oss tensor/architecture adapter. The
next implementation priority is therefore streamed shard quantization + streamed/quantized evaluation,
followed by a Doctor design whose optimizer/teacher state is also blockwise; bypassing the memory gate is
not an experiment.

### Durable-artifact and source-release rule

Evaluation candidates are currently temporary: `audit_ladder.py` deletes each candidate after measuring
it. A floor receipt therefore records research data but does not, by itself, preserve a deployable model.
No parent source may be automatically erased until the selected winner has been materialized as a durable
artifact, hashed, reloaded through the production path, capability-checked, and bound by hash to its
source and receipt. Only then may the guarded release command remove that allow-listed model source.
"Wipe models to make room" never means delete unprocessed parents, adapters, receipts, or arbitrary user
data.

The largest **future install scheduled by total parameters that still has a plausible internal-SSD
lifecycle** is Kimi-K2.6: the repository reports roughly 1.1T parameters (1T in the model table), 32B
active, and a measured Hugging Face dry-run size of 595.2 GB for its native INT4/compressed-tensors
checkpoint. Source plus the 0.75-bpw research artifact is about 698 GB, leaving the 1 TB Studio lifecycle
possible only after guarded release of prior verified sources. It stays `planned-blocked` until Hawking
has a Kimi/MLA/MoE/compressed-tensors adapter and the live free-space requirement (remaining source plus
150 GB hard reserve and 32 GB cache reserve) passes. DeepSeek-V4-Pro is larger at 1.6T, but its roughly
894 GB source alone cannot coexist with those reserves on this disk. DeepSeek-V4-Flash remains the
smaller 284B/13B-active architecture bring-up immediately before Kimi in the queue.

## HARDENED RESEARCH MATRIX — FLOPS ARE A COUNTER, NOT THE OBJECTIVE

Let `C` be capability that passes the frozen quality/task gate, not raw generated-token count. Every
promoted row reports `C/joule`, `C/byte moved`, `C/resident byte`, `C/active parameter`, and `C/second`
at the declared SLO, alongside FLOPS only as a diagnostic. Rejected drafts, rehydration, cache misses,
SSD/network traffic, retries, and longer outputs are charged. The expected reductions below are
hypotheses until a same-box receipt measures them.

| Horizon | Proposal | Complexity and movement hypothesis | Latency/energy expectation | Difficulty and hardware compatibility | Required gate |
|---|---|---|---|---|---|
| **Immediate implementation** | VTQ `k/d` reconstruction matrix: `d2/d3/d4/d8`, frozen/learned, iso-payload, block sweep | Offline encode is `O(N * codec-state-cost)`; reconstruction is `O(N)`. Current oracle rehydrates weights, so **expected serve-bandwidth reduction is 0%**. | No runtime speed/energy claim; it cheaply maps information loss versus exact effective bpw and scale. | Medium. Runs as CPU encode plus ordinary GPU/MPS evaluation on existing GPUs and Apple Silicon; shard rows can distribute independently. Specialized vector hardware is not required for the oracle. | Exact accounting + full-model multiwindow PPL + f16-relative tripwire; `deployable=false`. |
| **Immediate implementation** | Rank-8 Doctor first, rank-16 conditional | Per adapted linear, the low-rank correction is `O(r(m+n))` storage/movement and two GEMVs, versus incorrectly materializing an `O(mn)` dense delta. Exact adapter bytes are added to bpw. | Expected to cut Doctor wall time and transient memory; quality recovery, not raw speed, is the first decision. | Medium; PyTorch CPU/MPS and conventional GPUs work now. Future hardware can fuse low-rank correction with vector decode. | Pre-update zero-correction fallback, held-out progress, checkpoint hash, exact adapter bpw, PPL + tripwire. |
| **Immediate implementation** | Detached download/processing supervisors and negative-result completion | Control work is `O(1)` per sample; model I/O remains checkpointed by shard/config. It prevents duplicate downloads and abandoned partial work. | No model-speed claim; expected wall-time/J savings come from avoiding OOM, swap, recomputation, and attached monitoring. | Low; macOS-native and hardware-agnostic. Existing GPU jobs stay detached and mutually admitted by the heavy-work lease. | Normal pressure, bounded swap, green thermal/AC, disk reserves, atomic ledger/checkpoints, explicit drain state. |
| **Immediate implementation** | Spec readiness checker only | `O(receipt size + artifact hash)`; no draft/target compute is launched. | Zero speculative speedup claimed; removes false-green experiments whose verifier executes a different model. | Low; all hardware compatible. It deliberately blocks both Apple and GPU launch paths. | Durable `.tq`, TQ batched parity, cost-aware oracle, and a real runner; all are currently unmet. |
| **Medium-term research** | Integrate the SDSC-v2 learned-LUT records into packed vector `.tq` and direct CPU decode | Packed weight traffic becomes `N*b_eff/8` bytes versus `2N` for f16: ideal weight-byte reduction is `1-b_eff/16` (about **94.8%--97.1%** for the observed `0.835--0.467` logical oracle bpw), before KV/metadata. | Lower latency/J only if vector decode costs less than avoided memory traffic; actual file bpw and cache behavior decide. | High. The O(header + LUT-section)-memory SDSC-v2 reference record, per-tensor/source hashes, and exact CPU Q12 round trip now exist; `quantize-model` archive integration and all serving paths remain blocked. Existing GPUs need unpack/decode kernels; Apple CPU is the first reference, then Metal. | Quantizer supplies exact LUTs for every vector tensor, framing is serialized, packed round-trip parity passes, actual file bpw is measured, and no out-of-band state remains. |
| **Medium-term research** | Direct-resident vector Metal path; remove retained decoded Q12/i32 copy | Decode stays `O(N)` but compressed bytes flow directly to the fused GEMV. Peak model residency should track `N*b_eff/8 + declared metadata`, not a parent copy. | This is the first plausible bandwidth/latency/J win on M3 Ultra; no percentage is claimed before kernel and whole-system measurement. | Very high. Not compatible with today's scalar Hawking path; Apple Silicon/Metal is the target. CUDA/other GPUs need separate kernels; future hardware should fuse lookup, accumulation, and state update. | CPU reference parity -> Metal parity -> all-linear ownership -> no fallback -> resident-memory closure -> same-box baseline win. |
| **Medium-term research** | Streamed shard quantization/evaluation and blockwise Doctor for 32B/72B/120B | Peak working memory becomes `O(shard + optimizer block + adapter)` instead of `O(model)`. Source bytes should be read once per pass; repeated shard scans are charged. | Enables otherwise impossible experiments; it is not inherently faster and may trade wall time for pressure safety. | High. CPU/GPU/Apple compatible at the orchestration layer; distributed nodes can own independent shards, but cross-shard reductions and recovery state need hashes/barriers. | Bounded working set, resumable shard receipts, no swap escalation, reconstruction/eval equivalence to the small in-core control. |
| **Medium-term research** | TQ-native batched verifier and cost-aware speculative oracle | Verification cost is measured for `B=1..8`, not assumed to be one forward. Useful work is exact committed target tokens; draft, verify, sync, and rejected tokens are charged. | Promotion requires speedup lower bound `>=1.10` in code, prose, and tool-JSON separately, plus lower J/accepted-token and non-regressing p95. | Very high. Existing GPUs and Apple Metal are possible only after the TQ batched path matches single-token TQ exactly. Distributed speculation additionally pays communication per branch. | Hash-bound TQ parity, zero skips, measured verifier curve, exact-match oracle, <=78 GiB dual residency, zero swap. |
| **Long-term paradigm shift** | Persistent inference transaction: compressed target, draft, KV/state, scheduler, and commit in one execution graph | Move only state for the longest exact committed prefix; asymptotic arithmetic may remain similar, but communication/state movement scales with committed work rather than attempted branches. | Target is higher `C/J`, `C/byte`, and SLO goodput through fewer launches, copies, and rejected-state writes—not higher peak FLOPS. | Frontier research. Best fit is Apple/future unified-memory hardware with persistent kernels; portable GPU runtimes need explicit residency and preemption support. | Exact distribution preservation, transactional rollback, measured branch waste, energy/byte accounting, no protected-SLO regression. |
| **Long-term paradigm shift** | Locality-routed MoE/sub-bit expert store with retrieval-first persistent state | Active compute/traffic scale with selected experts/state, `O(P_active)` rather than installed `O(P_total)`, plus routing/index communication. | Capability per resident byte/active parameter can rise even when raw tok/s does not; tail latency and miss energy are first-class. | Frontier research. Compatible conceptually with GPUs, Apple unified memory, SSD tiers, and specialized near-memory decoders; distributed serving must price expert-transfer bytes. | Expert sensitivity, router quality, cache-hit curve, miss/p95 bound, source/artifact parity, and full capability gate. |

Quantization and speculation remain separate axes: a smaller draft can help only if its saved bytes exceed
the acceptance loss, and a restored target can be speculated only after TQ batched parity. Distributed
execution may parallelize encoding, shard evaluation, and MoE experts, but communication bytes and
synchronization latency are charged to the same capability-efficiency receipt. Future architectures may
change the kernel, memory tier, or scheduler; they do not waive the packed-parity and capability gates.

## WHAT `go` DOES (E0 plus ten guarded phases — see `docs/plans/quintessential_engine_2026_06_29.md` for the full design)

- **E0 CAPABILITY-EFFICIENCY BASELINE** — establish same-box quality, energy, byte, memory-pressure,
  and wall-clock receipts before the ladder. This is the accounting control for every later phase.

- **P0 CODEC TRIAGE + STAGE/ADVISE** — one-time: `codec_parallelism.py --catalog` scores every
  candidate codec/kernel design for decode PARALLELISM (not just density) before any Rust build
  time is spent — the direct lesson from the QTIP-on-Metal dead end (serial decode ate the
  bandwidth win). Per-model (inline in P1/P4): `auto_bits.py` + `size_frontier.py` +
  `doctor.py registry --select` + `arch_coverage.py` recommend the bit format, the serve regime
  (RESIDENT/MOE-PAGED/DENSE-OOC), the auto-composed recovery chain, and which Doctor levers are
  architecture-compatible (dense/SSM/MoE — Mamba2 and RWKV-7 both get their real flat-state math,
  not an approximation) — all before any bake. For MoE frontier models, `expert.py cache`
  simulates hot-expert cache hit-rate/blended-tok-s across cache sizes so the eventual OOC pager's
  cache size is chosen from a measured sweep, not a guess.
- **P1 CONDENSE** — the bit-floor-vs-scale curve across the safely runnable
  {0.5B,1.5B,7B,14B} parents; 32B receives an exact current budget-defer receipt and routes to the
  representative-shard research queue. Runnable parents use the Doctor registry, multiwindow ppl +
  capability tripwire, one floor receipt per model, then the curve fit (H1 descent vs H0 flat).
  L0-L3 run first. **L4 block-QAT is gated** until
  the train-free stack leaves a measured quality gap on 7B+ and its checkpoint/memory oracle is green.
  **L5 GPTQ-Hessian is gated** until L4 or the best train-free baseline is reproducible and L5's predicted
  recoverable gap clears its own cost ceiling. Do not automatically execute L4/L5 just because registered.
  -> `reports/cron/bit_floor_curve.jsonl`, `receipts/official/*-floor.json`.
- **P2 SUBBIT** — the true vector-trellis reconstruction-oracle lane: mandatory frozen/learned
  `k1/d2`, `k1/d3`, `k1/d4`, `k1/d8`, iso-payload controls, staged block-length amortization, and
  rank-8-first Doctor with exact adapter billing. `subbit.py measure` is SUBBIT-0-THEORY only and is
  explicitly non-gating; `subbit.py ladder` is fit arithmetic only. Every VTQ row remains
  `deployable=false` until learned-LUT packing, round-trip, native runtime parity, direct residency,
  multiwindow/tripwire quality, and efficiency gates pass. MoE adds `expert.py sensitivity` without
  weakening any dense/vector gate. -> `reports/cron/bit_floor_subbit.jsonl`.
- **P3 SPEC** — remains **DEAD/BLOCKED by default**. Existing EAGLE (`tau≈0.877`) and n-gram
  (`tau≈1.43`) results miss Hawking's `tau>=2.5` resurrection threshold. No capture-retrain or governor
  run is allowed until the verifier is TQ-native and exact against the same durable `.tq` used by
  single-token greedy, B=1..8 costs are measured with zero skipped cases, a hash-bound cost oracle clears
  every workload gate, and a checkpointed runner exists. `spec_revive.py` is a fail-closed readiness
  checker, not a launcher; more RAM does not cure a verifier that silently reads parent GGUF weights.
  If reopened, distribution-preserving exact-match, accepted-token energy/bytes, and p95 gates all apply.
- **P4 FRONTIER** — the 100B+ research prize (235B-A22B / 405B / 671B / DeepSeek-V4-Flash /
  DeepSeek-V4-Pro / GLM-5.2 / Kimi-K2.6 / Kimi-K2.7-Code / Kimi-K2-Instruct; exact HF ids in `BASELINES.md` and
  `tools/condense/studio_manifest.py`),
  serve-oriented since they don't fit the doctor
  budget. Runs on streamed shards (entropy floor + per-expert sensitivity + serve-fit record + the
  auto-composed recovery chain). The native-serve quality + RAM-cliff are the serve build.
- **P5 EVAL + LONG-CONTEXT** — `eval_suite.py` (capability + NIAH) + `ctx_extend.py` (YaRN) +
  `kv.py frontier` (int2/trellis KV, SSD-paging, SSM) + `kv.py hybrid` (STKV: exact recall + unbounded reach).
- **P6 BASELINE** — `bench_baselines.py`: the wedge gate vs llama.cpp IQ1_S/IQ2 + MLX-4bit at matched
  effective bpw. WIN iff it beats IQ2 on 7B+; else reframe to portfolio.
- **P7 CLIFF** — `ramcliff_bench.py --all`: RAM-cliff tok/s + energy J/tok — the headline + the
  energy moat. A CLIFF-WIN requires native serve + >10x tok/s + lower J/tok.
- **P8 CODEC** — `codec_bakeoff.py`: STRAND vs QTIP/QuIP#/AQLM at matched bpw (CUDA-locked rivals
  are offline-encode-only; STRAND is the lone Metal-native trellis serve).
- **P9 SYNTH + SCORECARD** — fit both lane curves + the 70B/405B extrapolation, then `scorecard.py`:
  the populated competitive matrix. **Refuses any WIN cell without an R3+ receipt.**
  -> `reports/condense/SCORECARD.md`. **The deliverable.**

## PHASE E PILOTS — the research program beside the ladder

These pilots come from the computational-efficiency agenda and remain default-off. E0 accounting precedes
the ladder; the other pilots use ladder evidence and cannot displace a live proof gate without a measured
upper bound.

1. **E1 useful-token/locality scheduler:** price prefill, decode, cache hits, and protected p95; GO only
   at `>=1.5x` SLO goodput on a mixed trace without a protected-decode regression over 5%.
2. **E2 content-addressed state and copy-on-write:** exact hits must preserve byte-identical output; GO at
   `>=20%` TTFT reduction on a trace with 30% reuse and metadata below 10% of state bytes.
3. **E3 typed state/KV:** all-position addressability and lossless backing are mandatory; GO at `>=50%`
   state-byte reduction and `>=10%` long-context latency/energy reduction at the full quality gate.
4. **E4 predictive-innovation oracle:** count entropy-model, metadata, index, and decoder costs; GO only if
   realized held-out bytes beat the best static quantized representation by at least 20%.
5. **E5 retrieval/speculation oracle:** this is research evidence, not permission to revive P3; GO only
   after the existing `tau>=2.5` proposal threshold, TQ single-versus-batched exact parity, and measured
   B=1..8 verifier-cost prerequisites, then require lower accepted-token energy/bytes, latency, and
   non-regressing p95 across every intended workload.

## LOCKED CONTEXT — do NOT reopen

- Hardware: this M3 Ultra Studio, 96 GB unified, 819 GB/s advertised memory bandwidth, 1 TB SSD.
  Metal/MPS only, NO CUDA, no cloud, no 512 GB box. The GPT/Codex app remains an interactive tenant,
  so the training process does not own all unified memory. Run one heavy job at a time; the active memory
  monitor must stop launches under pressure and drain/checkpoint Hawking before swap threatens the app.
  Wall-clock is cheap — optimize for maximum proof, not nominal utilization. Use the highest-fidelity public source:
  bf16 where available, explicit compressed-source receipts where not.
- Respect the measured dead-ends: old low-rank resurrection attempts plateaued, so rank-8/16 Doctor is
  now a tightly billed diagnostic rather than an assumed cure; escalate to full-rank/codec-native work
  only after its held-out value clears the cost gate. Use NO uniform-STE through the trellis (codec-aware
  only); AWQ x residual is a non-win; calibration is domain-matched, not merely diverse; small models are
  controls and 7B+ decides scale claims. `subbit.py admm` already re-confirmed NanoQuant is a low-rank
  resurrection (KILLs on real qwen-05b) — do not iterate on it.

## PROOF DISCIPLINE (the program enforces this; do not relax it)

- EFFECTIVE bpw only (baker AGGREGATE incl. RHT + outlier + side-info), never nominal.
- VTQ oracle bpw must also charge learned LUTs, padding, framing/accounting scope, unquantized tensors,
  and exact serialized Doctor bytes. Symbol payload `k/d` and SUBBIT-0 entropy are never artifact bpw.
- Quality = output-space ppl vs the f16 parent with MULTIWINDOW>=4 + the multi_eval capability
  tripwire. A floor claim is void if ppl passes but a capability collapses.
- `reconstruction_oracle`, `packed_artifact`, and `native_resident_runtime` are distinct evidence classes.
  No row may inherit a stronger class from a predecessor. A bounded negative is operationally complete,
  remains visible in the ledger, and cannot be selected as a winner.
- Production headline numbers are CPU-bf16. No public WIN below repro level R3.
- FAKE-WIN BAN: a rung counts ONLY if the compressed payload stays in RAM and decode is folded into
  the GEMV. Any recipe whose served tensor is rehydrated to f16 counts ZERO. Spec-decode counts ONLY
  under the exact-match (bit-lossless) gate.

## THE TWO GATES THAT DECIDE THE MOONSHOT (both currently UNMEASURED, not refuted)

1. Does doctor recovery work inside the pressure-gated 96 GB envelope? (every +dr died on the 18 GB box
   by swap/timeout, not recipe; 14B runs alone and 32B stays gated until measured or streamed.)
2. Is MoE expert sensitivity non-uniform? (dense was uniform ~3% spread = dead; MoE is a different regime.)

If both pass: build toward the resident prize that this machine can actually support. **235B-A22B @
1.34 bpw = 39 GB, 405B @ 1.34 bpw = 68 GB, and DeepSeek-V4-Flash @ 1.34 bpw = 48 GB** are resident
targets with interactive headroom subject to measured KV and runtime peaks. **DeepSeek-V3 671B @ 1.0 bpw
= 84 GB is a pressure-sensitive edge/paging target, not a default resident claim.** GLM-5.2 @ 1.0 bpw
= 94 GB, Kimi @ 0.75 bpw = 94-103 GB, and DeepSeek-V4-Pro @ 0.50 bpw = 100 GB require paging,
streaming, a smaller verified artifact, or a non-interactive capacity mode. If recovery fails: density-only, usable floor
~3.3-3.8 bpw. If expert sensitivity is uniform: fall back to 405B @ 1.34 = 68 GB dense.
`0.50` and `0.33` dense symbol payloads are legitimate VTQ research points, not fit or serve claims.
Current block-256 probes show severe reconstruction loss, and side information raises effective bpw;
only a packed, restored, capability-clean, directly resident artifact can turn either point into a real
model claim. MoE amortization is a separate promising hypothesis, not an exemption from those gates.

## THE SERVE-BUILD CRITICAL PATH (the one gate on real wins, in order)

See `docs/plans/quintessential_engine_2026_06_29.md` §"Serve-build critical path" for the full spec.
RE-DERIVED FOR 96 GB: 235B-A22B, 405B, and V4-Flash have plausible resident rungs; 671B is an edge/paging
case, and GLM/Kimi/V4-Pro overflow the interactive resident budget. The OOC expert pager therefore remains
on the critical path for those larger-capability lanes, but cannot be marketed as a resident-speed win. The path:
(1) residual two-part GPU decode parity, (2) all-tensor `.tq` loader, (3) per-expert `.tq` writer +
resident heterogeneous MoE serve, (4) frontier native quality + RAM-cliff RESIDENT (flips P4/P7
GATED->MEASURED) for the three resident candidates, and (5) measured expert paging for 671B+.
Speculation is not on this critical path; its dead-lane resurrection gates remain separate. Until (1)-(4)
land, the size/quality/tps numbers stay honestly GATED.

Proof-mode native serve must fail closed. For Qwen-family `.tq` receipts, run with:

```
HAWKING_QWEN_TQ=1 \
HAWKING_QWEN_TQ_STRICT=1 \
HAWKING_QWEN_TQ_REQUIRE_ALL_LINEAR=1 \
HAWKING_QWEN_TQ_REQUIRE_GPU=1
```

Those levers make a missing sidecar, partial all-linear coverage, or silent CPU fallback an error instead
of a backward-compatible no-op. A RAM-cliff or tok/s receipt is not admissible without this proof-mode
coverage line plus the served-forward parity command.

The laptop-safe runtime contract is now a signed artifact:
`reports/condense/studio_runtime_contract.local.json`. Build and verify it before launch-packet. It
hashes the product runtime source of truth for profiles, workload defaults, energy modes, and the strict
native `.tq` proof-mode receipt requirements; it does not claim that a model has served natively.

The scorecard reads native serve receipts from `reports/condense/<LABEL>_serve.json`. A passing receipt
must state `status=pass`, `native_tq=true`, `rehydrate_f16=false`, `tq_strict=true`, `all_linear=true`,
`gpu_bitslice=true`, `served_forward_pass=true`, and a positive `tok_s`.

## STAGING (download on the Studio; `go` skips what is not present)

14B/32B/72B/MoE/100B+ parents/checkpoints are owner-gated downloads on a 1 TB SSD. Exact HF ids + sizes are in
`BASELINES.md`. `go` runs whatever is staged and skips the rest. Begin with the already staged
0.5B/1.5B/7B parents; 14B and 32B are verified in isolated staging. GO admits 14B directly from the
exact verified path and schedules it alone after the early wave; the processing queue subsequently
publishes/validates the canonical 14B view. The 32B source is download-complete but full-model processing
remains blocked until its pressure-gated or streamed plan is green. The detached chain handles 72B and
the architecture-gated 120B/284B/1.1T successors only through the current-free lifecycle. The 7B substrate +
its calibration/recovery data are the primary training control.
`procure.py --all-frontier
--link-mbs 300 --efficiency 0.7` estimates ~9.8 h download-only for the full nine-model frontier
manifest; perfect 300 MB/s sustain is ~6.9 h. `procure.py --cycle-frontier --link-mbs 300
--efficiency 0.7` is the operational view: download one source, bake/receipt the `.tq`, then release
that source before the next checkpoint. Every plan is derived from **current free space minus a 150 GB
safety reserve**, with **64 GB scratch and 32 GB HF/Xet cache** charged explicitly. Never plan against the
nominal 1 TB capacity. On the current disk, V4-Flash is the first feasible frontier source; 235B-A22B is
too tight as a whole-source download once reserves are charged, and all larger sources need verified
shard-streaming, external storage, or a later capacity change.

## CHECKPOINT, DRAIN, MOVE, AND REPLUG PROTOCOL

The machine may be moved between network locations. A terminal disconnect is harmless; removing power is
not. Use these boundaries:

1. **Before a long step:** record the model, phase, exact command, input hashes, current free disk, memory
   pressure, and expected checkpoint path in the Studio lifecycle ledger. Downloads use the same local
   directory and project-local HF/Xet cache so completed shards survive a restart. Processing writes a
   durable per-model/per-config checkpoint before advancing.
2. **Before unplugging:** request a drain. Stop launching new work, allow the active download shard or
   training save interval to finish, SIGTERM only the Hawking child if needed so `doctor.py` writes its
   atomic latest adapter, flush the phase/download ledger, and verify the newest artifact or HF cache.
   A PID file or an empty queue is not sufficient; the operator must see an explicit `SAFE TO UNPLUG`
   state with no active writer. The implemented command is
   `python3.12 tools/condense/studio_run.py drain`; confirm
   `python3.12 tools/condense/studio_run.py --status`, `download_queue.py status`, and
   `processing_queue.py status` agree that no child/writer is active and the Studio status reports
   `safe_to_unplug=true`. It also refuses whole-machine safety while an unrelated CPU/RAM-heavy process
   is still active. An interrupted Hugging Face transfer may resume completed cache chunks after reboot;
   that resumability does not make it safe to remove power during an active write.
3. **Move:** shut the Studio down normally. Do not unplug while a `.partial`, safetensors write, receipt
   signature, cache verification, or source-release step is active.
4. **After replugging:** confirm AC power, network route, thermal state, current free disk, memory pressure,
   and swap; rerun preflight/environment verification; then run `studio_run.py resume`. Completed
   shards/configurations/phases skip, and an incomplete current unit restarts from its last durable point.
5. **Never release a source checkpoint** until the `.tq` artifact inventory and required receipt verify.

The detached supervisors, rather than an attached Codex conversation, sample macOS memory pressure,
swap delta, process-group RSS, free disk, power, and thermal state. Yellow pressure pauses new launches;
red pressure or sustained swap growth requests a graceful Hawking checkpoint/drain. They must never kill
GPT/Codex to make a benchmark pass. Phone/remote observation should be a short status read, not a
resident polling agent that consumes the same memory/CPU budget being measured.

Operator loop for the giant frontier:

```
FREE_GB=$(df -g "$PWD" | awk 'NR==2 {print $4}')
STORAGE_BUDGET_GB=$((FREE_GB - 150))
test "$STORAGE_BUDGET_GB" -gt 0
cargo run -p hawking --bin hawking -- studio snapshot
cargo run -p hawking --bin hawking -- studio worktree-plan --out reports/condense/worktree_split_plan.local.json
cargo run -p hawking --bin hawking -- studio worktree-plan --verify reports/condense/worktree_split_plan.local.json
cargo run -p hawking --bin hawking -- studio density-receipt-build --out reports/condense/studio_density_receipt.local.json
cargo run -p hawking --bin hawking -- studio density-receipt-verify --path reports/condense/studio_density_receipt.local.json
cargo run -p hawking --bin hawking -- studio runtime-contract-build --out reports/condense/studio_runtime_contract.local.json
cargo run -p hawking --bin hawking -- studio runtime-contract-verify --path reports/condense/studio_runtime_contract.local.json
cargo run -p hawking --bin hawking -- studio status --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32
cargo run -p hawking --bin hawking -- studio storage-plan --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32 --link-mbs 300 --efficiency 0.7
cargo run -p hawking --bin hawking -- studio lifecycle --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32
cargo run -p hawking --bin hawking -- studio gate --phase procure --require-refresh reports/condense/frontier_refresh.preflight.json --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32
cargo run -p hawking --bin hawking -- studio license-plan
cargo run -p hawking --bin hawking -- studio license-decisions draft --out reports/condense/frontier_license_decisions.draft.json
cargo run -p hawking --bin hawking -- studio license-decisions verify --path reports/condense/frontier_license_decisions.draft.json
cargo run -p hawking --bin hawking -- studio review-plan --refresh reports/condense/frontier_refresh.preflight.json --out reports/condense/frontier_review_plan.local.json
cargo run -p hawking --bin hawking -- studio review-decisions draft --refresh reports/condense/frontier_refresh.preflight.json --out reports/condense/frontier_refresh_review_decisions.draft.json
cargo run -p hawking --bin hawking -- studio review-decisions verify --path reports/condense/frontier_refresh_review_decisions.draft.json
cargo run -p hawking --bin hawking -- studio proof-pack --force
cargo run -p hawking --bin hawking -- studio launch-packet-build --out reports/condense/studio_wave0_launch_packet.json --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32
cargo run -p hawking --bin hawking -- studio launch-packet-verify --path reports/condense/studio_wave0_launch_packet.json
cargo run -p hawking --bin hawking -- studio audit-grade-build --out reports/condense/studio_audit_grade.local.json
cargo run -p hawking --bin hawking -- studio audit-grade-verify --path reports/condense/studio_audit_grade.local.json
cargo run -p hawking --bin hawking -- studio run-next --require-refresh reports/condense/frontier_refresh.preflight.json --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32
python3.12 tools/condense/frontier_ops.py ledger --refresh-hf --out reports/condense/frontier_ledger.launch.json
python3.12 tools/condense/frontier_ops.py refresh
cargo run -p hawking --bin hawking -- studio review-plan --refresh reports/condense/frontier_refresh.json --out reports/condense/frontier_review_plan.launch.json
cargo run -p hawking --bin hawking -- studio review-decisions draft --refresh reports/condense/frontier_refresh.json --out reports/condense/frontier_refresh_review_decisions.launch.json
cargo run -p hawking --bin hawking -- studio review-decisions verify --path reports/condense/frontier_refresh_review_decisions.launch.json
# for every REVIEW candidate printed by refresh:
cargo run -p hawking --bin hawking -- studio review-candidate <hf_id> --decision watch --by <name> --note <why>
# or, after editing/re-drafting a final signed workbook with --by/--note/--final:
cargo run -p hawking --bin hawking -- studio review-decisions apply --path reports/condense/frontier_refresh_review_decisions.launch.json --confirm
cargo run -p hawking --bin hawking -- studio license-plan
cargo run -p hawking --bin hawking -- studio record-license <label> --status accepted --by <name> --license <id> --terms-url <url> --allowed-use research --redistribution none --source-policy local-only-delete-after-bake --note <decision>
# or, after filling a final signed license workbook per model:
cargo run -p hawking --bin hawking -- studio license-decisions sign --path reports/condense/frontier_license_decisions.draft.json --out reports/condense/frontier_license_decisions.final.json
cargo run -p hawking --bin hawking -- studio license-decisions verify --path reports/condense/frontier_license_decisions.final.json
cargo run -p hawking --bin hawking -- studio license-decisions apply --path reports/condense/frontier_license_decisions.final.json --confirm
cargo run -p hawking --bin hawking -- studio storage-plan --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32 --link-mbs 300 --efficiency 0.7
cargo run -p hawking --bin hawking -- studio lifecycle --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32 --link-mbs 300 --efficiency 0.7
cargo run -p hawking --bin hawking -- studio proof-pack --force --out reports/condense/frontier_proof_pack.local.json
cargo run -p hawking --bin hawking -- studio source-provenance-plan
cargo run -p hawking --bin hawking -- studio coverage-plan
cargo run -p hawking --bin hawking -- studio source-provenance-receipt draft <label> --sign-draft
# after procurement fills final HF revision, source format, file manifest, and download/verify receipt:
cargo run -p hawking --bin hawking -- studio source-provenance-receipt sign <label>
cargo run -p hawking --bin hawking -- studio source-provenance-receipt verify <label>
cargo run -p hawking --bin hawking -- studio parity-receipt draft <label> --sign-draft
# after architecture parity runs fill final rows, exact commands, hashes, trace hashes,
# adapter/tensor-map receipts, tokenizer/context contracts, unsupported exits, and verified features:
cargo run -p hawking --bin hawking -- studio parity-receipt sign <label>
cargo run -p hawking --bin hawking -- studio parity-receipt verify <label>
cargo run -p hawking --bin hawking -- studio coverage-receipt draft <label> --kind both --sign-draft
# after baseline/eval runs fill final rows, exact commands, trace artifacts,
# machine/environment proof, same-box group, and frozen suite/score-set hashes:
cargo run -p hawking --bin hawking -- studio coverage-receipt sign <label> --kind both
cargo run -p hawking --bin hawking -- studio coverage-receipt verify <label> --kind both
cargo run -p hawking --bin hawking -- studio receipt-plan
# before native serve/RAM-cliff runs:
cargo run -p hawking --bin hawking -- studio receipt-record draft <label> --kind both --sign-draft
# after native serve emits a strict JSON report:
cargo run -p hawking --bin hawking -- studio serve-capture <label> --artifact <artifact.tq> --bench-json <serve_report.json> --command '<exact hawking serve bench command>' --load-receipt <load_trace.json> --served-forward-receipt <served_forward_trace.json> --parity-receipt <serve_parity_trace.json> --force
# after native serve/RAM-cliff rows are final, measured, and traced:
cargo run -p hawking --bin hawking -- studio receipt-record sign <label> --kind both
cargo run -p hawking --bin hawking -- studio receipt-record verify <label> --kind both
cargo run -p hawking --bin hawking -- studio experiment-plan
# before expensive-mode experiments:
cargo run -p hawking --bin hawking -- studio experiment-receipt draft <label> --sign-draft
# after seeds/ablations/rungs/repeats/nulls/rebake rows are final, same-run, and trace-hashed:
cargo run -p hawking --bin hawking -- studio experiment-receipt sign <label>
cargo run -p hawking --bin hawking -- studio experiment-receipt verify <label>
cargo run -p hawking --bin hawking -- studio claim-bundle-build <label>
cargo run -p hawking --bin hawking -- studio claim-bundle-verify reports/condense/<LABEL>_claim_bundle.json
cargo run -p hawking --bin hawking -- studio run-next --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32 --link-mbs 300 --efficiency 0.7
python3.12 tools/condense/frontier_ops.py launch-gate --phase procure --require-refresh reports/condense/frontier_refresh.json --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32
python3.12 tools/condense/procure.py --cycle-frontier --link-mbs 300 --efficiency 0.7
python3.12 tools/condense/procure.py --cache-status
python3.12 tools/condense/procure.py <label> --retries 2 --min-observed-mbs 80 --verify --progress-interval-s 60 --stall-timeout-s 900
cargo run -p hawking --bin hawking -- studio status --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32   # refresh FREE_GB after each wave
# after bake + receipt + verify:
python3.12 tools/condense/frontier_ops.py record-event <label> --stage bake --status pass --duration-s <seconds> --artifact <path>
python3.12 tools/condense/frontier_ops.py artifact-inventory <label>
python3.12 tools/condense/frontier_ops.py release-source <label> --dry-run
```

`release-source` is dry-run by default and refuses to delete source checkpoints unless a `.tq` artifact
exists and either a frontier record or official receipt exists. Use `--yes` only after the dry-run evidence
is correct. `procure.py` pins `HF_HOME`, `HF_HUB_CACHE`, and `HF_XET_CACHE` under `scratch/` by default,
records cache deltas in every download receipt, and exposes `procure.py --cache-prune` as a dry-run cache
maintenance hook. The maximal download command retries resumably with fewer workers if the command fails
or under-runs the observed-throughput floor, records live progress samples, terminates a long no-progress
stall, attaches route/HF/DNS/network diagnostics to bad attempts, then runs `hf cache verify` against the
local dir before the attempt is considered green.
`license-plan` prints the accepted-terms command for every frontier model;
`reviewed` is not enough for procurement. The product-facing `hawking studio record-license` command
must record license id, terms URL, allowed use, redistribution policy, source-retention policy, signer,
and note for accepted terms. `license-decisions draft|sign|verify|apply` provides the same accepted-terms
gate as a signed batch workbook; apply still requires complete final rows and `--confirm`.
`frontier_ops.py lifecycle` is the
operator DAG: if it says `needs-license-review`, do not download; if it says `ready-bake`, bake; if it
says `ready-release-source`, run the release dry-run.
`run-next` is dry-run by default and refuses human-proof gates or placeholder commands; it requires explicit
flags before downloads or heavy compute, and downloads also require a `--require-refresh` artifact. `record-event`
makes bake/serve/eval/archive durations part of the ledger. `artifact-inventory` hashes the durable `.tq`
output and source release refuses to proceed without a matching inventory. Refresh candidates tagged `REVIEW`
must receive a `hawking studio review-candidate` decision before the launch gate can go green;
`review-plan` writes the durable candidate-decision queue from the refresh ledger, and
`review-decisions draft|verify|apply` provides a signed batch workbook for the same human decisions.
The Rust CLI now exposes the same proof state through `hawking studio`: `snapshot` reads signed local
artifacts and prints the current red/green wall; `worktree-plan` groups the dirty tree by subsystem for
review splitting and verifies the signed split receipt; `density-receipt-build` and
`density-receipt-verify` sign the largest-file, LOC, disk, and local artifact-mass snapshot without
deleting evidence or weakening gates; `runtime-contract-build` and
`runtime-contract-verify` seal the native `.tq` proof-mode/runtime policy before launch-packet; `status`, `storage-plan`, `lifecycle`, `gate`,
`license-plan`, `record-license`, `license-decisions`, `review-plan`, `review-candidate`, `review-decisions`, `source-provenance-plan`,
`source-provenance-receipt`, `parity-receipt`, `coverage-plan`, `coverage-receipt`, `receipt-plan`, `experiment-plan`,
`claim-bundle-build`, `claim-bundle-verify`, `proof-pack`, `launch-packet-build`, `launch-packet-verify`, and
`audit-grade-build`, `audit-grade-verify`, `receipt-record`, `experiment-receipt`, `serve-capture` delegate to the guarded
operator tool; and `run-next` prints the next command without executing it. This is the product-facing
shell for Studio wave 0; heavy work still requires explicit proof gates and allow flags.
`proof-pack` is the one-command non-compute wall for the frontier manifest. It writes a signed manifest,
signed but blocked draft envelopes for source provenance, parity, baseline/eval coverage, native
serve/RAM-cliff, and experiment depth for each frontier label, then builds
`<LABEL>_claim_bundle.local.json` files that hash those drafts and remain claim-inadmissible. It preserves
final receipts unless `--force-final` is explicitly passed, so it can be rerun before Studio measurements
without erasing completed proof. Use
`hawking studio proof-pack --force` as the product-facing entry point; `frontier_ops.py proof-pack` is the
lower-level equivalent. Verifying those `.local` bundles should stay red for admissibility while reporting
`signature_ok=true`; only final `<LABEL>_claim_bundle.json` bundles may satisfy public claims.
`launch-packet-build` signs the Studio wave-0 packet by hashing/summarizing the preflight summary,
environment receipt, signed worktree split plan, signed runtime contract, refresh ledger,
license/review workbooks, storage plan, lifecycle dry-run, procurement gate, proof pack, and run-next
dry-run. A valid packet can still be red; `launch-packet-verify` proves the packet did not drift while
preserving `procurement_permitted=false` until every launch gate is green.
`audit-grade-build` signs the audit-grade receipt by parsing the Studio deep-audit facet table, hashing
the external harsher audit, and summarizing the launch packet, proof pack, worktree split, runtime
contract, density receipt, claim gate, procurement gate, and scorecard artifact. `audit-grade-verify` proves that receipt
did not drift. A valid audit-grade receipt can still report `target_reached=false`; on the laptop it
should also report `runtime_contract_ok=true` and `frontier_claims_walled=true` while the public-claim
gates remain red.
`source-provenance plan` prints the required source checkpoint provenance path for every frontier model.
Before a public claim, each `<LABEL>_source_provenance.json` must be final and signed with exact HF
revision, source kind, source format, procurement command, download/cache verification receipt, and
file-manifest evidence. Compressed/FP4/FP8 sources must record `source_is_prequantized=true` and a
source-format receipt; bf16 parents must record `source_is_prequantized=false` and `source_format=bf16`.
`hawking studio source-provenance-receipt draft` writes signed but blocked source-provenance envelopes for a label.
After verified procurement fills final HF revision, source kind/format, exact procurement command,
download/cache verification receipt, and file-manifest evidence, `hawking studio source-provenance-receipt sign`
refuses to sign anything draft, placeholder-filled, source-format-inconsistent, or missing verification
evidence. `hawking studio source-provenance-receipt verify` rejects unsigned, tampered, draft, or
placeholder source provenance.
`coverage-plan` prints the required baseline and eval receipt paths for every frontier model. Before
any public claim, each `<LABEL>_baselines.json` must contain same-box measured rows or explicit N/A rows
with reasons for llama.cpp Q4_K_M, llama.cpp IQ2_S, llama.cpp mmap OOC, MLX 4-bit, Unsloth Dyn 2.0,
and EXL3/PonyExl3. Each `<LABEL>_eval.json` must cover ppl multiwindow, capability QA, math, coding,
tool-use, long-context recall, RAM-cliff, and native-serve domains with pass or reasoned N/A rows.
`hawking studio parity-receipt draft` writes signed but blocked architecture parity envelopes for a label.
After the real reference-backend and Hawking/native runs are complete, `hawking studio parity-receipt sign`
refuses to sign unless the record is final, measured, threshold-clean, trace-hashed, exact-commanded,
adapter/tensor-map backed, tokenizer/context contracted, and confirms every family-specific required
native feature from `frontier_parity.py`.
`hawking studio parity-receipt verify` rejects unsigned, tampered, draft, placeholder, loose-logit,
trace-free, contract-free, or feature-incomplete parity records.
`hawking studio coverage-receipt draft` writes signed but blocked baseline/eval envelopes for a label.
After the real same-box runs are complete, `hawking studio coverage-receipt sign` refuses to sign unless
the record is final, covers every required row, carries machine fingerprint/environment receipt,
same-box group, frozen suite hash, frozen score-set hash, exact non-placeholder commands, and a
receipt/artifact/log trace for every measured/pass row. Best-effort baseline rows cannot unlock claim
bundles. `hawking studio coverage-receipt verify` rejects unsigned or tampered records.
`receipt-plan` prints the stricter serve/RAM-cliff receipt contract. `<LABEL>_serve.json` must use
schema `hawking.frontier_serve.v1`, identify the artifact hash, record exact commands and commit, pass
native `.tq` proof mode, prove no f16 rehydrate, prove all-linear/GPU ownership, set `parity_pass=true`,
report positive tok/s, include a load trace, and prove positive peak/resident/unified-memory fields with
`resident_memory_ok=true`. `<LABEL>_ramcliff.json` must use schema `hawking.frontier_ramcliff.v1`,
be `source=measured`, not modeled, identify the artifact hash, show native `.tq` serving, Q4_K overflow,
>10x cliff, lower resident J/tok, exact commands, commit, and Studio machine class.
`hawking studio receipt-record draft` writes signed but blocked native serve/RAM-cliff envelopes for a label. After the
real native serve and RAM-cliff runs are complete, `hawking studio receipt-record sign` refuses to sign unless the record
is final, measured, strict-native, trace-backed, and non-placeholder. Serve records need a served-forward
or parity trace; RAM-cliff records need powermetrics/energy and baseline traces. `hawking studio receipt-record verify`
rejects unsigned, tampered, draft, placeholder, synthetic/modelled, or trace-free native receipts.
`serve-capture` is the Studio bridge for the native serve half. Feed it an existing `.tq` artifact, the
JSON report emitted by Hawking's native serve bench, the exact command, a load trace, and served-forward
plus parity trace receipts. It hashes the artifact and bench JSON, refuses f16 rehydrate/fallback reports,
requires strict/all-linear/GPU ownership, positive tok/s, resident memory proof, served-forward pass, and
parity pass, then writes the signed `<LABEL>_serve.json`. Use `hawking studio serve-capture` as the product-facing entry point;
`frontier_ops.py serve-capture` is the lower-level equivalent.
`experiment-plan` prints the expensive-mode matrix contract. `<LABEL>_experiment_matrix.json` must cover
at least 3 floor seeds, required calibration ablations, at least 4 bpw rungs, MoE expert allocation
or reasoned dense/N/A, 3 cold and 3 warm RAM-cliff runs, baseline variants, at least 2 archived null
certifications, and a rebake/hash verification row.
`hawking studio experiment-receipt draft` writes signed but blocked expensive-mode matrix envelopes for a label. After
the real experiment rows are complete, `hawking studio experiment-receipt sign` refuses to sign unless the matrix is
final, real/measured, covers every depth requirement, has exact non-placeholder commands, binds to one
Studio run with machine fingerprint, environment receipt, artifact inventory receipt/hash,
source-provenance receipt, and experiment-plan hash, and carries a row receipt/artifact/log/report trace
plus trace SHA-256 for every pass/measured/certified row. `hawking studio experiment-receipt verify`
rejects unsigned, tampered, draft, placeholder, trace-free, hash-free, or depth-incomplete matrices.
`hawking studio claim-bundle-build` signs the final public-claim evidence by SHA-256 after signed source
provenance, signed parity, signed baseline/eval, signed native serve/RAM-cliff, and signed experiment
matrix files exist. `hawking studio claim-bundle-verify` rejects stale, missing, or
claim-inadmissible bundles, reports whether the bundle signature itself is valid, and
`launch-gate --phase claim` treats missing signed bundles as a hard
failure.

Before any quality, tok/s, or RAM-cliff claim, run:

```
cargo run -p hawking --bin hawking -- studio coverage-plan
cargo run -p hawking --bin hawking -- studio source-provenance-receipt verify <label>
cargo run -p hawking --bin hawking -- studio coverage-receipt verify <label> --kind both
cargo run -p hawking --bin hawking -- studio parity-receipt verify <label>
cargo run -p hawking --bin hawking -- studio receipt-plan
cargo run -p hawking --bin hawking -- studio receipt-record verify <label> --kind both
cargo run -p hawking --bin hawking -- studio experiment-plan
cargo run -p hawking --bin hawking -- studio experiment-receipt verify <label>
python3.12 tools/condense/frontier_parity.py status
HAWKING_QWEN_TQ=1 HAWKING_QWEN_TQ_STRICT=1 HAWKING_QWEN_TQ_REQUIRE_ALL_LINEAR=1 HAWKING_QWEN_TQ_REQUIRE_GPU=1 \
  cargo test -p hawking-core --features tq --test qwen_tq_serve_parity -- --ignored --nocapture
cargo run -p hawking --bin hawking -- studio claim-bundle-build <label>
cargo run -p hawking --bin hawking -- studio claim-bundle-verify reports/condense/<LABEL>_claim_bundle.json
python3.12 tools/condense/frontier_ops.py launch-gate --phase claim --require-refresh reports/condense/frontier_refresh.json
```

The claim gate stays red until each frontier model has a passing signed `<LABEL>_parity.json` receipt plus
passing signed baseline/eval coverage receipts, strict signed native-serve/RAM-cliff receipts, and a complete
signed expensive-mode experiment matrix verified by the signed receipt runner, then a signed
`<LABEL>_claim_bundle.json` that verifies every evidence file by hash. A modeled or synthetic RAM-cliff
record is useful as a probe, but it is not claim-admissible.
