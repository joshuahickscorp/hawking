# Decision: Qwen Ascension Metadata Preflight (30B / 80B)

**Status:** admission / preflight scaffold only  
**Date:** 2026-08-06  
**Code:** `lab/operators/qwen_ascension_preflight.py`  
**Tests:** `lab/tests/test_qwen_ascension_preflight.py`  
**Bible:** HAWKING_ASCENSION_BIBLE Â§7âÂ§9 (source admission before stream; dual-Qwen bootstrap)

---

## Decision

Install a **metadata-only, fail-closed preflight** for the prospective Qwen
**30B** (executor / `QWEN3_MOE`) and **80B** (reviewer / `QWEN3_NEXT`) lanes.

This scaffold **validates caller-supplied manifest and evidence mappings**.
It does **not**:

- contact the network
- invoke Hugging Face / Xet
- load a model
- launch a download
- create cache files
- invent exact public model IDs, revisions, or config digests

**Default decision is `BLOCKED`.** `download_permitted` remains `false` until
every required gate is green in the supplied evidence.

---

## Required gates (all must be green)

1. `source_admission`
2. `pinned_identity`
3. `runtime_loader_forward_support`
4. `resource_supervisor_green`
5. `actual_artifact_receipt`
6. `profiler_parity_capability`
7. `controller_approval`

Additionally:

- Manifest must resolve uniquely to the **30B** or **80B** lane record.
- Identity fields (source, 40-char revision, config digest, license, architecture)
  must be **supplied**, not invented by this module.
- Proposed `family_key` must appear in `runtime_capability.supported_families`.

A claim of `download_permitted=true` is **rejected** until those conditions hold.

---

## What this is not

| Claim | Allowed? |
|-------|----------|
| Permission to fetch model bodies | **No** |
| Live acquisition / stream authorization | **No** |
| Runtime loader or forward proof | **No** (only checks supplied receipts) |
| Replacement for credential-broker preflight | **No** (narrower metadata gate) |
| Replacement for parity / TG harnesses | **No** |

Even an `ADMITTED_METADATA_ONLY` result with `download_permitted=true` is a
**local readiness signal over supplied documents**. It is still not an
automatic fetch, not a Gravity load, and not controller permission to occupy
disk with weight bodies.

---

## Lane distinction

| Scale | Lane id | Family key | Role |
|-------|---------|------------|------|
| 30B | `qwen_30b` | `QWEN3_MOE` | executor |
| 80B | `qwen_80b` | `QWEN3_NEXT` | reviewer |

Records are kept distinct. Exact Hub identity is bound only when a later
admission lane supplies and pins it.

---

## Follow-ons (out of scope here)

- Bind real official source admission receipts when Proto-Frankenstein offload
  frees the machine and controller authorizes acquisition.
- Wire this decision into the credential-broker lifecycle without weakening
  secret isolation or the 15 GiB floor.
- Fill parity / TG / profiler bodies only after a lawful stream + verify.
