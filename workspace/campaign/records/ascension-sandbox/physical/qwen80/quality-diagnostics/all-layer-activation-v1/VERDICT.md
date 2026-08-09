# Q80 Gravity coherence lane — verdict

**Date:** 2026-08-09  
**Goal:** coherent artifact under 1.5 BPW before any TPS increase.  
**Verdict:** `ALL_LAYER_ACTIVATION_CAPTURE_NOT_YET_POSSIBLE`

Negative result is the valid deliverable. Fitting on L0 / component captures is refused.

---

## 1. Capture design and provenance

| artifact | path |
|---|---|
| readiness (sealed) | `CAPTURE_READINESS.json` |
| design (full) | `CAPTURE_DESIGN.json` |
| status | `STATUS.md` |
| broad input (32 probes / 3929 tokens) | `requests/QWEN80_BROAD_ACTIVATION_ALL_LAYER_ROUTE_CAPTURE_INPUT_*.json` |
| prepare provenance | `PREPARE_PROVENANCE.json` |
| null placeholder | `null-first/NULL_MISSING_CAPTURE.json` |

**Tokenizer reuse:** Q80 `tokenizer.json` sha256 equals Q30's
`19564a48c4f71a2a1b937cce34c737a1e662b171c5f5d7edf641a15cd896f07d` — token IDs
from the sealed Q30 broad corpus are reused with that equality checked at prepare.

**Baseline artifact (admitted, not coherent):**

| field | value |
|---|---|
| complete_physical_bpw | **1.1331148544404688** |
| ceiling | 1.5 |
| tensors | 74,391 |
| status | `CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED` |
| manifest seal | `14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b` |

---

## 2. Per-layer nulls (FIRST — none available)

| layer | mean null | status |
|---:|---:|---|
| 0..47 | **not measured** | capture missing |

Instrument refusal (not a null trap):

```
verdict: CAPTURE_MISSING_NULL_NOT_MEASURABLE
mean_null_all_scored: null
n_rows: 0
```

Source: `null-first/NULL_MISSING_CAPTURE.json`

Campaign lesson preserved: a 3-prompt null of 0.94 was the wrong instrument;
a 32-probe / 3929-token capture dropped Q30 null to ~0.41. Q80 has not yet
produced the instrument at any layer.

---

## 3. Family × BPW × weight-cos × output-cos × null × surplus

**Not priced.** Inventing a table without real routed activations would reverse
the campaign turnaround. When the all-layer capture lands:

```bash
python3 -m lab.operators.q80_activation_null_first_report \
  --capture-run <RUN_DIR> --label q80-all-layer \
  --out-json .../null-first/NULL_ALL_LAYER.json

# only after nulls are on disk:
python3 -m lab.operators.ascension_qwen80_activation_weighted_svd_repack \
  --capture-run <RUN_DIR>
```

Selection policy (wired, not yet applied to real organs):

| rule | value |
|---|---|
| primary metric | **surplus_over_null** |
| secondary | weight_cosine (distribution-local guard only) |
| family | `activation_weighted_svd_low_rank_q` |
| component BPW ceiling | 1.5 |
| complete_physical_bpw ceiling | 1.5 |
| require_all_layer_capture | **True** (default) |
| L0-only / partial | **refused** |

Unit check (synthetic low-rank plant, Q80 expert shape 512×2048): surplus-first
selection under 1.5 BPW still holds (same codec path as Q30).

---

## 4. Candidate BPW and coverage

| field | value |
|---|---|
| candidate | **not produced** |
| achieved BPW | n/a |
| layers covered | 0 / 48 |
| repacked tensors | 0 / 74,391 |
| coverage percent | 0% |
| cannot_be_coherent | **true** |

Preflight:

```
status: CAPTURE_MISSING
cannot_be_coherent: true
```

---

## 5. Exact missing pieces (do not rediscover)

1. **`gqa_full_layer_same_runtime_encode`** — ready 0/12. Layers
   3,7,11,15,19,23,27,31,35,39,43,47 refuse at CPU preflight before lease.
   Authority: `complete-runtime/QWEN80_MULTI_LAYER_GQA_ENCODE_GAP_20260809T210000Z.json`
   (`EARNED_NEGATIVE_GQA_FULL_LAYER_SAME_RUNTIME_ENCODE_ABSENT`).
   Exact missing: same-runtime full-layer GQA encode with caller-owned
   `gqa_key_cache` / `gqa_value_cache` slots + rollback buffers (schedule seal
   `54084ddf…`).

2. **`broad_activation_capture_binary` body** — multi-token sequential forward
   that writes full route membership + stratified router-input f32 hiddens for
   all 48 layers. Scaffold refuses pre-lease:
   `crates/hawking-core/examples/ascension_qwen80_broad_activation_all_layer_route_capture.rs`
   (Q30 reference: `ascension_qwen30_broad_activation_all_layer_route_capture.rs`).

3. **`multi_token_sequential_state_for_capture`** — earned multi-layer path is
   single-token component parity (L0..L2), not broad activation matrices.

**Not missing:** admitted binary baseline under 1.5 BPW; surplus-first codec;
null-first + repack operators; 32-probe tokenized input; coverage gate.

---

## 6. Cheapest honest path ranking

| rank | path | ready for coherence? |
|---:|---|---|
| 1 | Metal all-48 after GQA encode + multi-token capture writer | **no** (primary) |
| 2 | Metal L0..L2 multi-token only | instrument only; **not** coherence |
| 3 | CPU packed hybrid all-48 | no (GQA multi-token chain absent) |
| 4 | Streamed BF16 source teacher | no (path absent) |
| ✗ | Reuse L0/L1/multi-layer component captures | **dishonest** — refused |

---

## 7. Owner commands

### A. Capture refusal scaffold (CPU, no lease)

```bash
cd /Users/scammermike/.claude-grok/worktrees/q80-gravity-coherence-20260809-170134
cargo build -p hawking-core --example ascension_qwen80_broad_activation_all_layer_route_capture
./target/debug/examples/ascension_qwen80_broad_activation_all_layer_route_capture \
  --output-dir "$(pwd)/workspace/campaign/records/ascension-sandbox/physical/qwen80/quality-diagnostics/all-layer-activation-v1/runs/refusal_$(date -u +%Y%m%dT%H%M%SZ)" \
  --input-json "$(pwd)/workspace/campaign/records/ascension-sandbox/physical/qwen80/quality-diagnostics/all-layer-activation-v1/requests/QWEN80_BROAD_ACTIVATION_ALL_LAYER_ROUTE_CAPTURE_INPUT_84acbf66b6b8da35.json" \
  --schedule-authority /Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-runtime/QWEN80_48_LAYER_EXECUTION_SCHEDULE_AUTHORITY_20260809T192559Z.json \
  --gqa-gap /Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-runtime/QWEN80_MULTI_LAYER_GQA_ENCODE_GAP_20260809T210000Z.json
# expected exit 3 + capture-refusal.json
```

### B. When GQA encode is ready — physical capture (YOU run; Metal / gate)

After the capture body is wired on top of multi-layer encode at `layer_count=48`,
serialize against other Metal residents and run the capture binary with the same
input JSON. Then:

```bash
RUN=.../runs/<capture_id>
python3 -m lab.operators.q80_activation_null_first_report \
  --capture-run "$RUN" --label q80-all-layer-broad \
  --out-json .../null-first/NULL_ALL_LAYER.json

python3 -m lab.operators.ascension_qwen80_activation_weighted_svd_repack \
  --capture-run "$RUN" \
  --root .../quality-candidates/activation-weighted-svd-v1
```

### C. Admit and serve on a NEW port (only after candidate seals)

Do **not** repoint an existing server. After the repack seals a candidate under
`quality-candidates/activation-weighted-svd-v1/`:

```bash
# 1) Admit the new candidate (isolated quality admission; do not overwrite baseline)
#    Follow the Q30 quality-repack admission pattern against:
#      .../quality-candidates/activation-weighted-svd-v1/QWEN80_ACTIVATION_WEIGHTED_SVD_V1_COMPLETE_BINARY_GRAVITY_CANDIDATE.json
#    Expected: separate admission receipt + current pointer under the candidate root.

# 2) Serve on a NEW port (example 18481 — pick a free port; never 18430/18480 if occupied)
cargo run -p hawking-core --example ascension_qwen80_native_http_server -- \
  --manifest <CANDIDATE_MANIFEST_ABS> \
  --expected-manifest-seal-sha256 <CANDIDATE_SEAL> \
  --port 18481

# 3) Generate text on that port before any coherence claim.
#    Coherence is false until served text exists. TPS remains deferred.
```

If `ascension_qwen80_native_http_server` is not yet complete for multi-layer
decode past GQA, serving itself remains blocked by the same GQA encode gap —
state that plainly; do not claim coherence from an unserved pack.

---

## Claim boundary

- No server started, no exclusive GPU lease, no TPS benchmark.
- No family table fabricated.
- No L0-only pack.
- Campaign knowledge applied, not rediscovered.
- Coherence remains unearned until all-layer capture → null-first → surplus-first
  pack under 1.5 BPW → admit → **served generated text**.
