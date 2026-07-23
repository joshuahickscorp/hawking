# MODEL LADDER IGNITION PRECHECK

Generated 2026-07-19. Session role: detached ignition / resume of the existing Frontier.

## Verdict

RESUME, not adopt. No live controller exists anywhere. G0 through G3 are sealed. The first
genuinely uncompleted gate is G4, and G4's required real forward is UNBUILT. That build is the
honest execution work for this session. The giants (685B, 1T) are source-absent and gated. CUDA is
budget-absent. Apple is the active lane.

## Live truth (verified, not from state files)

- HEAD == origin/main == `4fbca8bc`; tree clean except two untracked out-of-scope IDE docs.
- Hardware: Apple M3 Ultra, 96 GB unified, 28 cores (20P/8E). 79 GB free, 563 GB disk free. This is
  the 96 GB box, not the M1 Ultra 128 GB some docs assume.
- No authoritative controller alive. Every lease PID is dead (96622 g3, 84690 second-light, 41651 g2,
  27566 g0). Four stale leases cleared this session. One heavy lease is FREE.
- The two `com.hawking.doctorv5*` launchd jobs last exited 75 and 2 (dead), not running.

## Source and environment

- `models/gpt-oss-120b` present and complete: 65.2 GB, 7 shards, experts FP4 (mxfp4) + UE8 scales,
  everything else BF16. Real tokenizer + Harmony chat template present.
- The checkpoint is ORIGINAL OpenAI format (`block.N.mlp.mlp1_weight.blocks`), NOT HF naming, so
  `transformers.from_pretrained` cannot load it directly. The repo's byte-range `ProvenanceReader` is
  the loader; its bounded per-expert selftest is GREEN on the real source.
- Controller interpreter is the 3.12 framework Python: torch 2.6.0 (MPS true), transformers 5.6.2
  (gpt_oss present, used only as the parity math reference). The shell default python3 is 3.14 with
  no torch and must not be used.

## Gate ladder

| Gate | State |
|---|---|
| G0 reproduction | SEALED (PASS) |
| G1 larger expert | SEALED (PROXY) |
| G2 complete layer | SEALED (COMPLETE) |
| G3 cross-layer transfer | SEALED (COMPLETE; g3_000..g3_023, transfer_sha256 present) |
| G4 short end-to-end real forward | PENDING and UNBUILT (this session's target) |
| G5 complete 120B artifact | PENDING |

Resume point: G4.

## Science truth

Negative and converging at sub-bit. No Event Horizon, no capability pass anywhere.

- Second-Light baseline: 0.770 whole-artifact BPW, functional divergence 0.688, capability_pass false.
- G3 reality: a real complete layer costs 2.1 to 3.25 BPW (NOT sub-bit); weighted-combine divergence
  0.55 to 0.86; capability_parity false; mlp1 does not transfer to late layers, mlp2 does.
- Every forward to date is an APPROXIMATION on synthetic Harmony-ish token ids, measuring only
  RELATIVE orig-vs-packed divergence. The repo's own `_swiglu` is non-parity (split-half + plain
  SiLU), correct only because the wrong activation cancels in a relative comparison. No real logits,
  perplexity, or coherent generation has ever been produced in this campaign.
- Generation-M mechanics (RUN_ALL_COMPLETE) are speed wins that apply only once a quality-positive
  representation exists; none does.

## Why G4 is the honest ignition

G4 is the first gate that demands the capability tier: real logits, logit KL, top-k overlap,
next-token agreement, PPL/NLL, deterministic generation. That requires a real HF-validated forward,
which the repo explicitly lists as unbuilt ("the capabilities probe must stay False until they land").

Naive transformers is a structural wall (mxfp4 to bf16 dequant of 120B is ~234 GB, OOM on 96 GB, and
the checkpoint is OpenAI-format anyway). The honest route is to extend the PROVEN bounded-streaming
per-expert loader to a full 36-layer real forward with the CORRECT gpt-oss activation (interleaved
gate/up, clamp 7.0, alpha 1.702, `(up+1)*glu`), final norm, unembed, and the real tokenizer. This is
bounded-memory (stream one block plus a few experts at a time, ~65 GB read per forward, no offload).
The parity reference is transformers `modeling_gpt_oss.py`; the empirical validity gate is coherent
next-token prediction on real prompts.

## Giants and CUDA

- 685B (DeepSeek-V3.2) and 1T (Kimi-K2.6): source ABSENT, prep-only, gated behind 120B milestones.
  No giant hands off this session; this is verified, not an excuse. 563 GB disk cannot hold a runnable
  giant regardless.
- CUDA: BLOCKED, no sealed cloud budget. Budget is not inferred from key presence.

## Rollback

`git reset --hard 4fbca8bc` (HEAD == origin/main; nothing committed). No source is mutated; all model
reads are byte-range read-only. Gate rollbacks: G3 `e2609f94`, Mechanics `f5521233`.
