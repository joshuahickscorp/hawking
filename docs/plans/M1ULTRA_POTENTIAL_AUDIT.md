# M1 ULTRA POTENTIAL AUDIT: every evaluated category graded on the delivered box's ceiling

Audit date: 2026-07-03. Companion to `reports/condense/SCORECARD.md` (the receipt-gated competitive
matrix), `docs/plans/quintessential_engine_2026_06_29.md` (the serve-build critical path), and
`docs/plans/hawking_handoff_2026_06_28.md` (the codebase map). This document answers one question: if the
program moves onto the DELIVERED box (Mac Studio, M1 Ultra, 128 GB, 8 TB) and every pre-registered bet is
pursued to its end with NO wall-clock limit, what is the maximum each evaluated category can reach, and what
wall stops it going higher even there.

Maximal by mandate, honest by construction. Every ceiling below 10 names its wall. Every ceiling claimed
names the bet that must convert. Every bet inherits the proof discipline: EFFECTIVE bpw only, output-space
ppl plus the multi_eval capability tripwire, judge low-bit on 7B+ never 0.5B, no public WIN below repro
level R3, a tie or near-tie is a NULL, a served tensor rehydrated to f16 counts ZERO. Produced by a nine
subsystem code-grounded audit, each grade independently adversarially verified (the verifier cut four
scores and raised one; those adjustments are carried below). House style: no em or en dashes.

## 0. The machine, honestly (M1 Ultra, 128 GB, 8 TB)

The whole program was planned for a box that never arrived. Every doc (`GO.md`, `docs/plans/STUDIO_GO.md`,
`hawking_handoff_2026_06_28.md`, the "locked context") targets a Mac Studio M2 Max, 96 GB, 2 TB, ~400 GB/s.
The DELIVERED box is a Mac Studio M1 Ultra: 20-core CPU (16 performance + 4 efficiency), 48 to 64-core GPU,
32-core Neural Engine, 128 GB unified memory, ~800 GB/s memory bandwidth, 8 TB SSD.

Against the three reference points:

- Proven dev box (every measured number to date): M3 Pro, 18 GB, ~150 GB/s. Decode headline ~31 tok/s
  clean-room (`reports/a4_clean_walltime.json`, median 31.03, trials [30.63, 31.03, 31.21]; caveat: that
  receipt's model_id field says deepseek-v2-lite-q4 while its kernel profile and the README attribute the
  number to Qwen2.5-3B-Q4_K_M, so the model label is ambiguous even though the tok/s is real; the shipped
  default was ~26.6, labelled the contaminated anchor in `bible_archive.md`). The 7B doctor swap-died here
  (FAIL-001).
- Planned but never delivered: M2 Max, 96 GB, ~400 GB/s, 2 TB. Every device constant in the codebase
  (`ladder.py` RAM_GB=96, `size_frontier.py` DEVICES) is still computed for THIS box.
- Delivered, grade against THIS: M1 Ultra, 128 GB, ~800 GB/s, 8 TB.

The delivered box beats the PLANNED box on every axis that matters for a bandwidth-bound inference engine:
about 2x the memory bandwidth (800 vs 400), 1.33x the unified memory (128 vs 96), 4x the disk (8 vs 2 TB),
and more GPU cores (48-64 vs 30-38). It beats the PROVEN M3 Pro by about 5.3x bandwidth, 7x memory, and
44x disk. The one regression: M1-generation single-thread CPU is roughly 25 to 35 percent slower per core
than M2/M3. Twenty cores (vs twelve) partly offset it, and 800 GB/s helps even the CPU legs that are
bandwidth-bound (bf16 GEMV). The regression bites in exactly one place: serial, compute-bound host work.

What this splits into, and it governs every grade below:

- The box REMOVES the three resource walls the 18 GB laptop proved: the doctor swap-death (FAIL-001 was a
  measured hardware floor, not a recipe failure; the teacher-first KD peak is ~43 GB and every studio_run
  doctor peak, 7B=40 / 14B=65 / 32B=85 GB, now fits resident with headroom), the disk pressure (8 TB
  retires the 2 TB stage limit and holds every frontier parent plus its `.tq` at once), and single-model
  residency (128 GB holds a 70B `.tq` resident, and holds 235B-A22B at ~39 GB and 405B-dense at ~68 GB
  resident with NO expert pager).
- The box AMPLIFIES the one axis it can: decode is bandwidth-bound, so the ~800 GB/s roughly doubles the
  RAM-cliff tok/s ceiling the 400 GB/s plan assumed, and roughly quintuples the absolute tok/s the M3 Pro
  measured.
- The box does NOT lift the structural walls: the relative decode gap to llama.cpp is bandwidth-invariant
  (same Q4_K bytes, same bus), the Omega(N) associative-recall theorem still caps a fixed SSM state, the
  dense PTQ information floor (~4.85 eff-bpw on 7B) is entropy not RAM, and no box makes a small model
  smarter.

## 1. Scoring rule, and the unbounded-wall-clock axiom

Each category gets: the adversarially-verified proven-today score, the M1 Ultra ceiling, the bet that must
convert, and the wall above the ceiling. Theoretical means exactly that: the program's own history says
most bets null, and a null at this scale is TERMINAL knowledge (there is no bigger box behind this one
except rented CUDA, which doctrine forbids), so every null here is a program-routing decision, which is
itself value.

THE UNBOUNDED-WALL-CLOCK AXIOM (the operator's standing directive, first-class). There is no time limit.
The box is plugged in 24/7 and wall-clock is FREE. The mandate is to convert as much as physically
possible, EXPAND not reduce, and the slower M1 CPU making every doctor pass, bake, and floor-search take
LONGER is expected and accepted, never a reason to cut scope. This axiom has a sharp analytic consequence
that reshapes every ceiling: most of what the deflationary audit calls a "wall" is not a wall at all but an
UNBUILT BUILD or an UNRUN EXPERIMENT. Those are TIME-walls, and unbounded wall-clock is precisely the
solvent for them. Only a few walls are STRUCTURAL (a theorem, a bandwidth physics floor, or model
intelligence at a fixed size) and survive unlimited compute. So the scoreboard below carries two ceilings:
the BOUNDED ceiling (the adversarial audit's honest figure, which implicitly priced wall-clock and
build-effort as scarce) and the MAXIMAL ceiling (every time-wall converted, only the structural walls
left). The goal aims at the MAXIMAL ceiling and reports honestly against it.

## 2. The scoreboard (adversarially verified)

Proven is the verified current-today score (the adversarial verifier's figure where it cut or raised the
audit; the change is noted). Bounded is the audit's honest ceiling. Maximal is the ceiling once unbounded
wall-clock converts every time-wall. The wall column is the residual after the maximal push; each is tagged
STRUCTURAL (survives unlimited compute) or TIME (a build or a run, attackable and therefore a target here).

| Category | Proven | Bounded | Maximal | The bet that converts it | The wall that remains |
|---|--:|--:|--:|---|---|
| Decode throughput (single-stream, iso-quant gap) | 6.5 | 7.5 | 9.0 | Build the mul_mv-class GEMV behind a real trace oracle; match llama.cpp at iso-quant on 800 GB/s | STRUCTURAL: relative gap is bandwidth-invariant, so parity is the ceiling on iso-quant single-stream, not dominance. Dominance lives on the native sub-4-bit `.tq` axis llama.cpp cannot run. |
| Kernel maturity (45 gemm_q4_k perms, autotune) | 7.0 | 7.5 | 8.5 | Re-autotune on 48-64 cores; retire the megakernel skeleton and the 45-perm micro-opt debt; land the one new kernel technique | STRUCTURAL: the 45-perm family's micro-opt axis is closed (Type-1 dead). Above 8.5 needs a genuinely new algorithm, and the trace oracle to know where to aim it. |
| Prefill + host dispatch | 5.5 | 6.5 | 7.5 | Wire paged-KV and B greater-than 8 continuous batching so aggregate throughput hides the serial encodes | STRUCTURAL: prefill and ~255 host-encodes/token are serial CPU work, and M1 single-thread is ~1.3x worse. Bandwidth does not scale serial dispatch. |
| Effective-bpw floor + bit-floor-vs-scale LAW | 3.5 | 6.0 | 8.5 | Run the FULL 27-model floor ladder resident; the doctor pushes recovered configs under +2% below 4.85 bpw; fit the descent curve on many points not two | STRUCTURAL: dense PTQ floor at ~4.85 bpw on 7B is an entropy wall, not RAM. Recovery moves it; it cannot be wished below the information content. |
| Sub-1-bit SUBBIT lane | 2.5 | 4.0 | 7.0 | Stage a real 100B+ MoE parent; prove per-expert amortized sub-1.34-bpw serve crosses near-1:1 | STRUCTURAL: dense sub-1-bit is fantasy (ADMM killed on real 0.5B); only MoE amortization dips under, and it is bounded by the active-expert entropy. |
| STRAND codec quality vs QTIP/QuIP#/AQLM | 3.0 | 5.5 | 8.0 | Deepen the trellis (larger L, vec_dim greater-than 1 AQLM-class codebooks) and win a REAL model-tensor bakeoff at matched eff-bpw | STRUCTURAL on the crown, SOFT on the moat: rivals are CUDA-locked so STRAND is the lone Metal-native trellis serve (a 10 on availability), but on raw quality the synthetic probe has it BEHIND QTIP (0.8036 vs 0.8285); closing that is codec research with no guaranteed win. |
| Quality recovery / the Doctor (gradient moonshot) | 5.0 (raised from 4) | 8.0 | 9.0 | Run the full L4-L6 stack (block-QAT, GPTQ-Hessian, deep-KD) to convergence, all seeds, AdamW, resident at 7B/14B/32B; emit R3+ receipts | STRUCTURAL: low-rank LoRA is capped by high-rank quant error (a proven NO-GO receipt), and recovery can at best reach the f16 parent, never exceed it. Full-rank/codec-native recovery is UNCAPPED up to that parent. |
| Doctrine: CPU-bf16 headline vs bf16-on-MPS | 3.0 | 7.0 (cut from 8) | 7.5 | Pass the bf16-vs-f32 MPS sanity check; move the doctor's LAB device onto 48-64 GPU cores | STRUCTURAL (policy): BASELINES mandates a cpu_bf16_confirmed run for every headline number, so the slow M1 CPU stays on the confirmation critical path. MPS moves the lab device, not the headline. |
| Native `.tq` serve (build + throughput) | 6.0 (cut from 6.5) | 8.5 | 9.5 | Bake a full all-linear `.tq` (not the 20 MB ffn_down toy), run `qwen_tq_arena_decode_tps` resident, clear greater-than 10 tok/s, confirm the fused-arena path executes | STRUCTURAL: bitslice GEMV tops out at 60-74 percent of peak (Viterbi ALU wall); the arena emits 2-3 dispatches/site and disables Q4_K fusion. That efficiency ceiling caps 10. |
| RAM-cliff throughput demo (the money demo) | 0 | 9.0 | 9.5 | Bake a 70B all-linear TQ2, serve via HAWKING_QWEN_TQ resident, measure tok/s while a CONTROL box's Q4_K swaps | TIME then STRUCTURAL: unrun is zero by rule (a TIME-wall, fully attackable). Residual STRUCTURAL: the cliff is a FIT win; served tok/s is still bounded by 800 GB/s / active-bytes and the 60-74 percent bitslice efficiency. |
| Flat-cost long-context serve (SSM wired) | 6.0 | 8.0 | 9.0 | RWKV-7 and Mamba2 serve tok/s + multi-slot batching at 1.5B/7B with a decode receipt; one Mamba2 e2e serve smoke | STRUCTURAL: flat-cost is a memory property, and served-but-low-quality long context is capped by the recall wall below. Throughput itself is near-uncapped. |
| Honest recall (NIAH / Omega(N) theorem) | 1.0 | 6.0 | 8.0 | Build the RWKV-X-class sparse-attention hybrid; route exact recall through `hawking-index` RAG; run real NIAH at 32k-128k at graded needle hardness | STRUCTURAL: the Omega(N) bits theorem hard-caps a fixed O(1) state past ~4K hard needles. The hybrid and RAG raise the PRODUCT ceiling; pure-SSM perfect recall stays dead. |
| KV-frontier levers (int4-KV, YaRN, STKV) | 2.0 | 6.0 | 8.0 | Wire per-channel int4-KV into the hot path (the kernels are BUILT), PASS a real-model ppl + long-ctx NIAH gate; minimal STKV tier0; a real Rust YaRN kernel | STRUCTURAL: KV compression is a memory-constant lever, not quality, and per-layer error compounds over 36 layers. The ceiling is "measured, gated, shipped," not "free unbounded context." |
| EAGLE-3 trained draft head | 1.5 | 3.5 | 7.0 | Retrain a real 3-layer tap head WITH real capture-layer plumbing + batched-verify-with-logits (the CHANGELOG projects 1.5-2.2x); clear an offline oracle tau greater-than 2.5 as a resurrection gate | STRUCTURAL: spec is a low-batch memory-bound win that degrades negative at high request rate (H4), and a perfect head still caps at the model's structural draft-accept rate. Free n-gram (1.43) is today's floor to beat. |
| Verify path + lossless-verify exact-match gate | 0 | 4.0 | 8.0 | Make the batched-verify GEMM bit-identical to the greedy GEMV (match reduction order / tie handling); cheap partial routes b==1 to greedy | TIME then STRUCTURAL: the gate was RUN and FAILED 6/20, a real kernel bug (attackable). Residual STRUCTURAL: a perfect fix makes spec LOSSLESS, not FAST; the throughput still needs the accept rate the head must earn. |
| SpecGovernor + drafter economics | 3.0 | 5.0 | 6.5 | Gate on real batch occupancy (H4); land a token-native RWKV-7 drafter for Qwen (no text bridge) and measure accept rate | STRUCTURAL: a governor can only STOP a bad drafter, never make a good one; the ceiling is the best real drafter's accept rate at low batch. |
| Honesty / receipt discipline | 8.5 | 9.5 | 9.5 | Run the program so at least one R3+ win-eligible receipt is emitted and validated; reach R4 (third-party Mac) | STRUCTURAL: R5 (the format cited externally) needs the outside world, not the box. Everything up to R4 is purchasable. |
| CI gap (honesty machinery that never runs) | 3.0 | 6.5 | 9.0 | Add a CI step running the `tests/` integration targets; PIN a small GGUF fixture so `llama_cpp_oracle` asserts instead of self-skipping; un-ignore w4a8 | TIME: a process fix plus a committed fixture, fully attackable. The residual is only the cost of maintaining the gate. |
| The GO conveyor (`studio_run.py` P0-P9) | 5.0 (cut from 6) | 8.0 | 9.0 | FIRST fix the `:279/:286` unpack bug that ValueError-crashes P6/P8; then run `go()` end-to-end so P1 emits floor points, the law fits, and the scorecard flips at least one cell to WIN | TIME then STRUCTURAL: the crash is two lines (attackable). Residual STRUCTURAL: P3/P4/P7 are gated on the Rust native-serve build the driver only RECORDS; the conveyor is only as done as that build. |
| Architecture coverage (the model families) | 2.5 | 6.5 | 8.5 | Stage each family GGUF on 8 TB, RUN the skip-guarded parity gates, add token-parity for llama/gemma/phi, IMPLEMENT the qwen3moe stub and the DeepSeek-V3/GLM sigmoid-group-topk + yarn/mscale router | TIME then STRUCTURAL: the missing routing/rope is real code, attackable with unbounded time. Residual: correctness of a hand-written router against a reference is verification labor, not a ceiling. |
| Size frontier / expert-paging serve | 1.0 | 4.5 | 7.5 | Serve 235B-A22B (~39 GB) and 405B-dense (~68 GB) RESIDENT (both under 128 GB, no pager); measure warm tok/s. Implement the V3/GLM router first | STRUCTURAL: the deep frontier (671B/744B/1T/3T) is SSD-bound at ~6 GB/s; dense out-of-core is under 0.1 tok/s by the tool's own KILL rule. The madvise pager is Type-1 dead in the free-RAM regime, so resident-only is the honest path. |
| HIDE M1: pass state not text (handoff) | 6.5 | 8.5 | 9.0 | Bridge the `hide-personalize::kv_handoff` seam (today a FakeCopier) into the live engine `copy_kv_prefix_to_slot` / RWKV `fork` across the HTTP boundary; run a real planner-coder-reviewer handoff | STRUCTURAL: gated on H3 (unmeasured "can RWKV-7 code") and the H7 audit tap. The RWKV state primitives (to_bytes/from_bytes/fork) are REAL; the wall is the unconnected halves and the model-capability question. |
| HIDE M2: free local fleets | 5.0 | 8.0 | 8.5 | Run continuous batch greater-than-or-equal 5 across real worktrees; measure that the Nth agent costs only its divergent tokens | STRUCTURAL: the H4 spec regression and the laptop-class thermal envelope bound true concurrency; the fabric is proven, the economics is a predictor with no measured receipt. |
| HIDE M3: own the `.tq` format | 5.5 | 8.5 | 9.0 | `hawking serve` with HAWKING_QWEN_TQ against a condensed `.tq` end-to-end; record the first real streamed token + ppl/coherence | STRUCTURAL: the product value (multi-tier SKUs, in-grid QA-LoRA re-bake) is downstream of a first-class `.tq` dispatch, which today is an FFN-per-tensor hybrid. |
| HIDE M4: grammar-guaranteed tool calls | 3.0 | 7.0 | 7.5 | Wire `json_constrain::mask_logits` into the live decode loop; MEASURE first-try-valid on a real schema set; land AST grammars from `hawking-index` | STRUCTURAL (truthfulness): "guaranteed" is the plan's own admitted overclaim; the honest ceiling is "first-try-valid ~95 percent + repair," never a guarantee. The grammar compiler itself is real. |
| HIDE ship-readiness (Tauri, executor, tests) | 7.0 | 8.5 | 9.0 | Boot the signed 18 MB DMG, have RuntimeSupervisor spawn a live `.tq`-serving hawking, drive one full SubmitTurn-to-AcceptDiff producing real tokens | STRUCTURAL: release CI + auto-update host is blocked on external resources; desktop-on-WebKit perf on large repos is untested. |
| The thesis gate (can the local model code) | 0 | 7.0 | 7.5 | Run `hawking-eval` against the `.tq`-served local model on M1 Ultra (trivially feasible on Stage-A f16 with 128 GB); record honest pass@1 + Wilson CI | STRUCTURAL: a small local model at coding is realistically a CONDITIONAL (route weak recall into RAG), not a frontier GO. The box makes the gate RUNNABLE; it cannot make the model smarter. Serving a bigger resident model is the only lift. |

## 3. North-star axes and the honest overall

Six headline axes answer "is hawking working." Each is the min-anchored roll-up of its scoreboard rows (a
subsystem is only as proven as its weakest load-bearing claim, per the zero rule).

| North-star axis | Proven | Bounded ceiling | Maximal ceiling (unbounded clock) |
|---|--:|--:|--:|
| A. Serve engine (decode tok/s, kernels, native `.tq`) | 6.0 | 8.0 | 9.0 |
| B. Compression science (bpw floor, subbit, STRAND) | 3.0 | 5.5 | 8.0 |
| C. The Doctor (does quality recovery heal at scale) | 4.0 | 7.0 | 9.0 |
| D. Long-context + frontier scale (SSM, KV, MoE, size) | 1.5 | 6.0 | 8.5 |
| E. Speculative decode (head, lossless verify, governor) | 0.5 | 4.0 | 7.0 |
| F. Honesty + program conveyor (receipts, CI, GO, HIDE ship) | 4.5 | 7.5 | 9.0 |

Honest overall (mean of the six):

- Current proven: **3.25 / 10**. Outside the serve engine and the receipt discipline, almost every headline
  capability is GATED or UNPROVEN, and two axes (D, E) are near-zero because their load-bearing experiments
  have literally never run or ran RED. This is the correct, deflationary pre-box reading.
- Bounded ceiling (audit's honest figure, wall-clock priced scarce): **6.33 / 10**.
- Maximal ceiling (unbounded wall-clock, every time-wall converted): **~8.4 / 10**.

The gap from 3.25 to ~8.4 is the size of the prize on this box, and its shape is specific: the box does not
raise the floor by itself, it makes the never-run experiments RUNNABLE and gives unbounded time to build
the unbuilt code. The residual 1.6 to 10 is the sum of the structural walls: bandwidth-invariant iso-quant
parity, the dense PTQ entropy floor, the Omega(N) recall theorem, the spec low-batch regime, model
intelligence at a fixed size, and R5 needing the outside world. Those are the honest, permanent walls, and
naming them IS the value even when a lever nulls.

## 4. The unbounded-wall-clock reframe: time-walls versus structural walls

The single most important consequence of the operator's no-time-limit directive: the deflationary audit
priced build-effort and wall-clock as scarce, so it labelled as "walls" many things that are only unbuilt.
Sorted honestly, the walls above split into two kinds, and the maximization attacks the first kind
exhaustively:

TIME-walls (a build or a run, converted by spending unbounded wall-clock, EXPANDED not skipped):

- Every "UNRUN = 0 by rule" row: the RAM-cliff demo, the thesis gate, real NIAH, the full floor ladder.
  These are zero only because nobody ran them; the box runs them.
- Every "unbuilt build" row: the fused-arena `.tq` throughput measurement, the all-linear `.tq` bake, the
  per-channel int4-KV hot-path wiring (the kernels exist), the DeepSeek-V3/GLM router, the qwen3moe stub,
  the RWKV-X sparse-attention hybrid, the batched-verify bit-identity fix, the real EAGLE-3 capture
  plumbing, the CI fixture, the `studio_run.py` crash fix, the mul_mv-class GEMV.
- Every "measured but not to convergence" row: the doctor's L4-L6 gradient arm (died on swap, never on
  recipe; on 128 GB it runs, and unbounded clock runs it to convergence at every rung and every seed).

STRUCTURAL walls (survive unlimited compute on this box, and cap the maximal ceiling below 10):

- The iso-quant single-stream decode gap is bandwidth-invariant: the same Q4_K bytes cross the same bus, so
  the best achievable is parity with llama.cpp, not dominance. Dominance is only on the native sub-4-bit
  `.tq` axis a GGUF engine cannot run at all.
- The dense PTQ information floor (~4.85 eff-bpw on 7B) is entropy. Recovery moves it; nothing wishes it
  below the model's information content. Sub-1-bit is real ONLY via MoE per-expert amortization.
- The Omega(N) associative-recall theorem hard-caps a fixed O(1) SSM state past ~4K hard needles. The
  honest product answer is the RWKV-X hybrid plus RAG, not "perfect infinite memory."
- Speculative decode is a low-batch memory-bound win that degrades negative at high request rate. A perfect
  head still caps at the model's structural draft-accept rate.
- Model intelligence at a fixed parameter count. The box serves a BIGGER model (70B resident, 235B/405B
  resident); it does not make a 7B smarter.
- R5 (the codec cited externally, third parties adopting the format) requires the world, not the box.

The maximization strategy that follows is simple and matches the operator's "expand not reduce": treat
every TIME-wall as a target and convert them all, however long each takes on the slower M1 CPU, and for
every STRUCTURAL wall PROVE it (a certified null with a mechanistic reason) or CIRCUMVENT it with a build
(the hybrid for recall, the bigger resident model for capability, the native `.tq` axis for the decode
gap). Whatever number remains after that is the true, permanent ceiling of the box, and it is proven.

## 5. Part 2: the box as a NEW instrument, not the 96 GB plan's executor

The M2-Max-96GB plan priced a box that never shipped. The M1 Ultra licenses whole experiment CLASSES that
plan could not even schedule. Grouped by what the new envelope unlocks:

Class 1, the doctor finally runs at scale (128 GB). Every +dr gradient config died on 18 GB by swap
(peaks 6926 / 9003 / 11780 MB over ceiling) or timeout, never on recipe (FAIL-001). The teacher-first KD
peak is ~43 GB and every studio_run doctor peak (7B=40, 14B=65, 32B=85 GB) fits resident. This is the
single highest-value class: it converts the C-axis gradient moonshot from impossible to runnable and
answers STUDIO_GO's open question verbatim. Verifier caveat: the guards were ALREADY relaxed to 8h/12 GB
and still emitted zero receipts, so residency is a strong hypothesis, not a certainty; the new binding
constraint is wall-clock, which the operator has removed, and it is a MIX (bakes are bandwidth-favorable at
800 GB/s, L4-L6 backward passes are compute-bound where the M1 CPU is slowest).

Class 2, 70B-class resident serve and the sub-70GB frontier without a pager (128 GB + 800 GB/s). A 70B
`.tq` fits resident (~17.5 GB at 2-bit) with KV headroom. 235B-A22B (~39 GB @1.34) and 405B-dense (~68 GB
@1.34) both sit RESIDENT under 128 GB, needing NO expert pager. This is decisive because the OOC pager is
unbuilt and its madvise primitives are Type-1 dead in the free-RAM regime (`MADV_DONTNEED` cannot evict
with free RAM). The resident path sidesteps the dead pager and obsoletes the "405B is cloud/rented"
doctrine for exactly these SKUs.

Class 3, the full 235B-744B frontier lands on disk (8 TB). Four times the planned disk is what lets
multi-hundred-GB frontier parents exist on the box at all; the 671B/744B/1T/3T `.tq` physically fit where
they did not before. But this class stays SSD-bandwidth-bound at ~6 GB/s AND blocked on a correct V3/GLM
router that does not exist. The disk moves the KILL line from "does not fit" to "fits but SSD-bound + wrong
logits" (both attackable, the second by building the router, the first only by accepting sub-interactive
tok/s, which unbounded wall-clock makes a legitimate measured result).

Class 4, MPS/Metal doctor vs the CPU-bf16 doctrine (48-64 GPU cores). The CPU-bf16-headline doctrine rests
on an fp16-specific MPS GQA bug and an fp16 overflow, both fixed by bf16. With the fit constraint gone and
48-64 GPU cores available, bf16-on-MPS is the untested unlock that dodges the M1 CPU regression entirely.
Ceiling capped at 7 because BASELINES still mandates a CPU-bf16 confirmation run for every headline number;
MPS moves the LAB device, not the headline device.

Class 5, the never-run gates become executable at full scale. The thesis gate (can the local model code),
the RAM-cliff demo, and real NIAH at 32k-128k were all gated behind memory the 18 GB box lacked
(`ctx_extend.py` caps CTX_MAX_REAL at 8192 "for the small box"). All three now run. None CONVERTS on
hardware alone; the box makes them measurable, and the mandate is to measure them all.

Class 6, multi-job concurrency + week-to-days wall-clock. 128 GB holds a batch greater-than-or-equal 5
fleet plus many forked RWKV states (6-16 MB each) without eviction. The full condense ladder that took a
babysat, swap-fragile week on 18 GB runs unattended and resumable, credibly week-to-days, dominated by
bandwidth-favorable bakes but taxed on the compute-bound QAT/KD legs. Under the unbounded-wall-clock axiom
this class expands into a persistent gated queue: the entire model set, every recovery lever composed and
ledgered, 30-seed protocols as the DEFAULT gate, the exhaustive codec bakeoff, with the adversarial
`frontier_verifier` wired in as a stage so honesty survives the long unattended run.

## 6. Doctrine re-derivations the switch forces (consolidated, with UNSAFE flags)

Rules that existed ONLY because of 18 GB or the assumed M2-Max-96GB. Each is re-derived or retired. UNSAFE
flags mark any reading that would re-litigate a killed lever (see `docs/dead_levers.md`).

Memory-envelope (SAFE, pure sizing constants):
1. `ladder.py` RAM_GB=96 / WEIGHT_BUDGET=84 / CONDENSE_RESIDENT_MAX_B=34 are M2-Max constants. Re-derive
   for 128 GB: naive-condense cap ~34B to ~48B, serve budget ~84 to ~112 GB, moving 405B@1.34 and 671B@1.0
   out of TIGHT/EDGE into resident.
2. `size_frontier.py` DEVICES has NO m1ultra row (only m2max 84/2TB and m3ultra-512). Add ~110 GB / 8 TB /
   ~6.5 GB/s; every published frontier number is currently computed for the WRONG box.
3. The FRONTIER serve-fit table (`fits = artifact <= 84.0`) is stale: on 128 GB, 671B@1.0 fits with
   headroom and 405B@1.34 is COMFY not TIGHT. Re-derive the OVERFLOW labels. CAVEAT: fitting in RAM is
   necessary but not sufficient for a serve WIN; serve QUALITY at sub-Q4 bpw stays gated on the codec
   quality oracle.
4. `size_frontier.py` ram_gbps: re-denominate the "% of peak" kills against 800, not 150. This STRENGTHENS
   the compute-bound kills (Q3_K, QTIP) and is correct-direction SAFE.

Swap/timeout guards (SAFE, with a verifier correction):
5. The doctor swap ceilings and timeout that killed the recovery column were an 18 GB artifact; 128 GB fits
   the 7B/14B/32B doctor resident. CORRECTION: `run_7b_frontier.sh` ALREADY runs 8h/12GB/18GB, not the
   6000MB/120m the older docs cite; guards already relaxed and still zero receipts, so "residency finishes
   the doctor" is a hypothesis to test, not a certainty to assume.
6. `BAKE_CHUNKS=8` bounded the baker's F32 recon on 18 GB; on 128 GB the full 7B/14B/32B recon fits, so
   raise or retire it for the mid ladder.
7. SGD-not-AdamW (`doctor_strand.py`) was a pure 18 GB memory concession; AdamW's two states fit at 7B/32B
   on 128 GB, so re-derive the optimizer on quality/convergence grounds. No dead-lever kills AdamW.

Device-doctrine (SAFE with a hard policy caveat):
8. "CPU-bf16 is headline, MPS is lab-only" rests on an fp16-specific justification; re-derive by running
   the bf16-vs-f32 MPS sanity check, then move the doctor LAB device to the GPU. HARD CAVEAT: BASELINES
   mandates cpu_bf16_confirmed for every headline number; this moves the lab device, not the headline.
9. The CPU-bf16 leg leans on 20 cores + 800 GB/s to offset the M1 single-thread loss. PARTIALLY UNSAFE AS
   WORDED: the "week-to-days, bandwidth-dominated favorable" re-timing is FALSE for L4 block-QAT / L5
   GPTQ-Hessian / L6 deep-KD, which are compute-bound backward passes where the M1 CPU bites hardest and
   the path is unmeasured. Re-time as a MIX, not a clean win.

Recovery-column (SAFE, the verifier's key catch):
10. The empty recovery column is a RAM/swap death, NOT a recorded dead lever. `dead_levers.md` has no
    recovery/QAT/KD/L4-L6 Type-1 kill, so 128 GB legitimately REOPENS it. Do not treat it as a settled
    kill. Contrast: KV-eviction, FFN-block-sparsity, Q3_K byte-cut are measurement Type-1 kills and STAY
    dead.

Serve-path + doc-staleness (SAFE):
11. `native_tq_serving_impl.md` "decoder is test-only" is STALE: both the env-gated live path and the
    fused-arena hot path are BUILT (and `tq_gpu.rs:363` has a docstring that says "stub returns Err" over a
    body that returns Ok, a lie to fix). Retire the build-plan framing to "DONE, remaining = tps
    measurement."
12. README "Next: wire RWKV into serve" is STALE: RWKV-7 serves e2e. Re-point the RWKV effort at
    batched/concurrent SSM throughput at a non-toy size, plus a Mamba2 serve smoke (its referenced log is a
    dead reference).
13. "int4-KV is a memory-fit lever gated on fitting in RAM": on 128 GB f16-KV fits comfortably, so the
    lever is now QUALITY-gated, not memory-gated. UNSAFE IF READ AS A GREEN-LIGHT: `dead_levers.md` #15
    HELD int4-KV pending a real-model ppl/NIAH ship-gate that has NEVER run; the re-derivation is safe ONLY
    if it keeps that gate as a hard precondition.

Framing (SAFE):
14. The single-stream decode headline is a small-box artifact; on 128 GB the binding question flips to
    aggregate/continuous-batch throughput (B much greater than 8). Re-headline as aggregate-tok/s-at-max-
    batch. Use the conservative 48 tok/s B=8 figure where the docs disagree.
15. R7 "free fleets brown out the laptop, cap default N" re-derives upward on 128 GB/800 GB/s; re-tune the
    FleetGovernor headroom. SAFE only because the aggregate path is GPU continuous batching; UNSAFE if it
    implied CPU-decode streams add fleet throughput (Type-1 dead at 0.06 dec_tps).
16. The "~1 tok/s per-row RHT wall" (`RhtMode::Rows`) is latent, not active (GPU-refused, CPU-only, no
    baked artifact uses it). Re-measuring on 800 GB/s is SAFE precisely because it stays CPU-only and does
    not resurrect the per-row STRAND-decode Type-1 kill.

Expert-pager (the verifier's second key catch):
17. "Build the OOC pager" as originally worded PARTLY RE-LITIGATES a Type-1 kill: `expert_cache`'s
    `evict_cold` (MADV_DONTNEED) and `mark_warm` (MADV_WILLNEED) are exactly the primitives measured dead
    in the free-RAM regime. On a 128 GB box with a 39-68 GB model there is ALWAYS free RAM, so the madvise
    pager is a no-op where it would be invoked. Re-frame the frontier bet to the RESIDENT serve (models
    under 128 GB), which needs no paging. The pager is only for the deep SSD-bound frontier, where it is
    SSD-bandwidth-bound regardless.

## 7. The spine / priority order

Grounded in `studio_run.py` P0-P9 and the two moonshot gates. Ordered by leverage: what unblocks the most
downstream axes first. The standing rule from the sister program transfers: never refine an owned number
while an unbuilt instrument blocks a WIN cell.

SPINE-0 (blocking prerequisite, before any science): fix the `studio_run.py:279/:286` unpack bug that
ValueError-crashes P6 BASELINE and P8 CODEC. The conveyor has never run end-to-end; this two-line fix is
how it can. That the bug survived at all is proof the GO program has never completed a single pass.

SPINE-1 (highest-leverage converting bet): the thesis gate. Run `hawking-eval` against the `.tq`-served
local model on the M1 Ultra and record an honest pass@1 + Wilson CI. It is 0 today (NULL by rule) and it
GATES every HIDE moat per the plan's own B2. Trivially feasible on Stage-A f16 with 128 GB. Nothing in the
HIDE program is worth converting until this returns a real number; even a CONDITIONAL verdict routes the
rest.

SPINE-2 (moonshot gate 1, does gradient recovery heal 7B-32B): P1 CONDENSE + the doctor. Run L4-L6 recovery
resident on 128 GB so it emits floor points at 7B (ideally +14B/+32B), lets `scaling_law.py` fit the
descent instead of INSUFFICIENT, and clears the +2% gate. Converts the entire C-axis and the bpw-floor LAW
(B-axis). Rides SPINE-0's fixed conveyor. The train-free half (residual res3+2 already -4.34% at 7B) is
DE-RISKED; the open question is purely the gradient arm. Under the unbounded-clock axiom, run the FULL
27-model ladder and the whole L0-L6 registry composed and ledgered, not the cheapest sufficient recipe.

SPINE-3 (moonshot gate 2, is MoE per-expert sensitivity non-uniform): P4 FRONTIER, resident. Stage
235B-A22B (~39 GB, resident) and measure whether expert sensitivity is non-uniform enough to justify
per-expert bit allocation. The honest on-box frontier win, needing NO pager (Class 2). Gated on a correct
router (the V3/GLM routing math that does not exist), so router implementation is its prerequisite. Rides
SPINE-2's floor points.

SPINE-4 (rides the artifacts): native `.tq` serve tok/s + the RAM-cliff demo. Once a full all-linear `.tq`
is baked (not the 20 MB toy), run `qwen_tq_arena_decode_tps` on a resident 70B and the RAM-cliff demo vs a
CONTROL box. Converts the `.tq`-serve (6.0) and RAM-cliff (0) rows. Requires the serve-quality Rust build
`studio_run` only records.

SPINE-5 (independent, cheap, high-honesty-value): close the CI gap + lossless-verify. Wire `tests/` into CI
with a pinned GGUF fixture (converts the 3.0 CI row); make the batched-verify GEMM bit-identical (converts
the RED 0 verify gate). Neither needs the moonshots; both run in parallel.

LAST (attackable but architecturally deepest, so late even under unbounded clock): the deep frontier
(671B/744B/1T/3T serve, SSD-bound + needs the router), the EAGLE-3 head to a real accept-rate win (retrain
with capture plumbing, prove the offline oracle first), the RWKV-X sparse-attention recall hybrid, and the
codec-quality push to beat QTIP. Each is real work with a real potential null; none jumps the spine.

## 8. Wave 0: the boring de-risking, once, before science

Do these on the new box before spending any moonshot wall-clock. All cheap, none science, each removes an
excuse or answers an open doctrine question by MEASUREMENT.

1. Transfer + build green. Repo + models on the 8 TB SSD; `cargo build --workspace` and `cargo test
   --workspace` (INCLUDING the `tests/` integration targets CI skips) green on M1 Ultra.
2. Fix the conveyor crash (SPINE-0). Patch `studio_run.py:279/:286`; run `studio_run.py --go-plan` (dry)
   end-to-end to confirm all nine phases wire.
3. Regenerate the stale device constants. Run `hawking autotune` on M1 Ultra to replace the m3pro18 profile
   (v3_dual / target_tps=60 are stale on 48-64 cores). Add the m1ultra row to `size_frontier.py` DEVICES
   and re-run `--ceiling` so every frontier number is computed for the delivered box.
4. Stage bf16 parents. Only 0.5/1.5/7B bf16 are staged; the 32B on disk is Q4_K GGUF not bf16. Stage bf16
   7B + 14B (+ 32B if feasible) so the doctor ladder has parents to recover.
5. Day-0 microbench A, MPS-vs-CPU-bf16 doctor throughput. Run the bf16-vs-f32 MPS sanity check on a small
   model, then time one doctor step on CPU-bf16 vs bf16-on-MPS. Decides the device doctrine (Class 4 /
   re-derivation 8). Keep a CPU-bf16 confirmation run for the headline regardless.
6. Day-0 microbench B, resident `.tq` serve tok/s. Bake one full-model `.tq` (not the 20 MB ffn_down toy),
   serve resident via HAWKING_QWEN_TQ, record the first real streamed token + a coherence check, and
   confirm the M=1 predec GEMV still hits at least 50 percent of 800 GB/s. Validates the single largest
   projected unlock (the ~150-165 tok/s single-stream figure is a PROJECTION until this runs).
7. Pin the oracle fixture. Commit a small GGUF so `llama_cpp_oracle` asserts external agreement instead of
   self-skipping to a vacuous green. This is the one thing that turns the CI honesty machinery from theater
   into a real gate.
8. Create `M1ULTRA_RUN_REPORT.md` (the scoreboard scaffold) and treat it as the single accumulating
   artifact so a dead session loses at most one wave.

Once Wave 0 is green and microbenches A/B have answered the device-doctrine and resident-serve questions,
the spine runs in order: thesis gate, then the two moonshot gates, then the serve/cliff receipts that ride
their artifacts, with SPINE-5 in parallel.

## 9. The two gates that decide the moonshot, and the single sentence

Both are still UNMEASURED (the reason to go), and the box makes both runnable:
1. Does gradient recovery (the Doctor L0-L6 stack) heal low-bit quality to at most +2% ppl at sub-4-bit on
   7B-32B? 18 GB could never run it (FAIL-001). If yes: the bit-floor descends with scale, DeepSeek-V3
   671B @ 1.0 bpw = 84 GB serves entirely from 128 GB of RAM where llama.cpp Q4_K (377 GB) cannot even
   load. If a certified tie on a model that provably needs recovery: the Doctor is bounded and the honest
   density-only floor is ~3.3-3.8 bpw.
2. Is MoE per-expert sensitivity non-uniform enough to keep the sub-1-bit lane alive on a REAL bake? If
   yes: the sub-1-bit MoE amortized serve is real (671B @ 1.0 is the prize). If uniform: fall back to
   405B @ 1.34 = 68 GB dense, resident.

The single sentence: on the delivered M1 Ultra with unbounded wall-clock, hawking's proven 3.25/10 has a
maximal honest ceiling of ~8.4/10, the gap is almost entirely never-run experiments and unbuilt code that
unlimited clock converts one by one, and the residual to 10 is a small set of named structural walls
(bandwidth-invariant iso-quant parity, the dense PTQ entropy floor, the Omega(N) recall theorem, the spec
low-batch regime, model intelligence at fixed size, and R5) that the program's job is to PROVE or
CIRCUMVENT rather than to launder.
