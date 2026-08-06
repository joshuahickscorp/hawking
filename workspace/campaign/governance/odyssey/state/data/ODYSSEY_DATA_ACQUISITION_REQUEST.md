# Odyssey data acquisition request

**Status:** awaiting human / controller decision  
**Lane:** `odyssey-data` — inventory and machinery only; **no download performed**  
**Binding constraint for T1–T7:** authorized training corpora are missing. Memory is not the blocker for the smallest streaming-adapter step (~2.1 GiB).

This document is a request for a licensing and provenance decision. Nothing listed here has been fetched.

---

## Decision required before any fetch

For **each** corpus below, answer:

1. **May we obtain it?** (yes / no / later)
2. **Under what licence**, and is that licence compatible with Odyssey’s use (training a student on a Math-Preserve substrate for internal research / possible release)?
3. **Who records provenance** (URL, snapshot date, revision/commit, contact)?
4. **Where may bytes land** on this machine (must **not** be `~/.cache/huggingface` owned by another campaign; prefer a dedicated Odyssey content-addressed store under a controller-chosen path)?

Until those answers exist, the membership machinery will keep reporting `DECLARED_NOT_PRESENT`.

---

## Declared corpora (from `ODYSSEY_DATA_MANIFEST.json`)

| Corpus id | Stage purpose | Approx. size to plan for | Candidate open / licensed sources | Licence question |
|-----------|---------------|--------------------------|-------------------------------------|------------------|
| **math-core** | T1 primary: capability-conditioned continued training under `math-v1` | **10–50 GB** raw text/JSONL depending on filter; start with a **1–5 GB** curated slice if streaming | OpenWebMath; Proof-Pile-2 (subset); MathPile; NuminaMath-CoT (check terms); OpenR1-Math (check terms); internally self-authored problem sets | Is the chosen source **permissive enough to train** and to **redistribute derived weights** if Odyssey ever ships? Many “research only” math sets forbid commercial derivatives. |
| **support-language** | T1 support: technical language, coding, tools (support halo preservation) | **5–20 GB**; start **0.5–2 GB** | The Stack (deduped, licence-filtered); Stack-Edu; OpenCodeInstruct; synthetic tool-call traces self-authored; technical docs with clear licences (e.g. RFCs) | Per-file licence in code corpora is mandatory. Reject copyleft that would infect weights if that is a product requirement. |
| **long-horizon** | T3 trajectory stabilization | **Not just text size** — need **parent trajectories**. Plan **1e3–1e5** rollouts; storage **10–200 GB** with logits/states, or **1–20 GB** if tokens-only | Self-generated from the released flagship parent after M18 (manifest gate); optional public long-context / agent traces only if licence and eval-disjointness hold | Manifest: *“T3 traces require the flagship source, which is released after M18.”* Is generation on the sealed Math-Preserve / flagship parent authorized here? Logging full trajectories may engage model ToS. |
| **sovereignty-corpus** | T4 permitted / boundary / paraphrase pairs | **Small–medium**: **10k–100k** pairs; **10–500 MB** | **Self-authored** (manifest `license_required: self-authored`); optional public safety suites only as **eval**, never train, unless explicitly dual-licensed | Must remain **self-authored** for training if product doctrine requires clean provenance. Confirm no scrape of proprietary refusal data. |

Manifest invariants that acquisition must preserve:

- every corpus is **content-addressed** before a single training step reads it  
- **no corpus overlaps any hidden evaluation set** (barrier is mechanical; see `ODYSSEY_CONTAMINATION_BARRIER.json`)  
- **licence is recorded per corpus**, not per collection run  

---

## Evaluation material (do **not** acquire as training data)

| Asset | Role | Notes |
|-------|------|-------|
| `odyssey/evaluation/support_halo_corpus_v0.jsonl` | **eval only** | Sealed; `corpus_sha256 = b3ebda04…54ec67`. Already present. Never train. |
| T0 hidden memberships under `odyssey/evaluation/hidden/` | **eval only** | Hash-committed held-out set. Never train. |
| Public selection set under `odyssey/t0/public_eval/` | **selection / eval** | Visible to training path as a list, but **must not** enter the train loader. |
| `tools/eval/thesis_*_corpus_v0.jsonl` | **eval fixtures** | Not Odyssey train sets. |

Any acquisition pipeline must run the contamination barrier against support-halo + hidden memberships **before** writing `admitted.jsonl`.

---

## Teacher traces (related, not a substitute for math-core)

| What exists today | Count / scope | Serves | Does **not** serve |
|-------------------|---------------|--------|--------------------|
| `GLM52_TEACHER_EVIDENCE_LEDGER.jsonl` (path in teacher manifest) | **122** lines; **118** `TEACHER_CAPTURED`; **~49** capsule ids; layers **0–77**; **20** synthetic windows; layer-scoped organ dumps | Partial **T2**-style representation / distillation evidence | **T1** text/math training; **T3** trajectory stabilization |

### What T1 needs from “teacher”

- Primarily **math-core + support-language** corpora (above), not more layer capsules.  
- Optional: more teacher capsules only if a distillation objective is chosen later.

### What T3 needs

- **Full parent trajectories** over long-horizon tasks: prompts, token sequences (or top-k logits), horizon labels, domain tags.  
- Order of magnitude: **thousands to hundreds of thousands** of trajectories, eval-disjoint.  
- Current gap: **0** trajectory traces of that kind; **122** layer-local ledger lines are a different object type.

### Acquisition question for traces

- Authorize **generation** of trajectories from the flagship / Math-Preserve parent on this machine after M18, under what logging policy and storage budget?  
- Or obtain an external trajectory corpus? If so, licence + eval-disjointness + parent alignment must be explicit.

---

## Suggested minimum to unblock stages

| Stage | Minimum data decision | Approx. disk once landed |
|-------|----------------------|---------------------------|
| **T1** | Approve **math-core** slice + **support-language** slice; run ingest + barrier | 1–10 GB to start |
| **T2** | Existing layer capsules may help; decide if additional capture is needed | already on disk (capsules); optional more |
| **T3** | Approve trajectory **generation or acquisition** plan (post-M18) | 1–200 GB depending on fidelity |
| **T4** | Authorize **self-authored** sovereignty pair writing process | &lt; 1 GB |
| **T5–T7** | No new train corpus required beyond above; need **eval** runs and hidden replication — do not unseal or train on hidden/support-halo | eval only |

---

## Explicit non-actions (this lane)

- No `curl` / `wget` / `huggingface-cli download` / dataset `git clone`  
- No read or write of `~/.cache/huggingface`  
- No training, no `odyssey/launch/` changes  
- No modification of sealed support-halo rules, corpus, or seal  

## Machinery already in-repo for the day a decision lands

- `python3 -m tools.odyssey.cli inventory`  
- `python3 -m tools.odyssey.cli membership-check`  
- `python3 -m tools.odyssey.cli ingest-fixture` (proof on labelled fixture only)  
- Ingest path: normalize → exact/near dedup → content-address → contamination barrier → `MEMBERSHIP.json` + `admitted.jsonl`  

Point the same ingest entry at an authorized raw JSONL **after** licence approval; do not invent math problems to fill the gap.
