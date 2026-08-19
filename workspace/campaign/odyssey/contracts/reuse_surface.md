# ODYSSEY-I REUSE-SURFACE RECON (read-only)

Scout the Hawking repo at `/Users/scammermike/Downloads/hawking` and map the EXACT
reusable APIs the Odyssey-I control plane will build on. Do NOT edit anything.
Return a precise engineering report (this report IS the deliverable).

For each item give: exact function names + signatures, `file:line`, CLI usage,
and JSON schemas where relevant. Quote code. No hand-waving.

1. MEMORY / WORKER GATE — `tools/worker_gate.py` (and any `tools/machine_state.py`).
   The function(s) that decide whether it is safe to admit a new local model worker
   (wired mem, compressor, swap, reserve, Metal headroom). Names + signatures +
   return-dict shape. Grep the callers: how does existing code gate a spawn?

2. DISK RECLAIM — `tools/reclaim_safe.sh`. What it does, how invoked, what it
   protects, exit codes, any `--disk-floor` flag behavior.

3. DETACHED CONTROLLER — `tools/ascent_controller.py` (FULL). The loop; the JSON
   state file path + schema; how it ranks targets; how it forms a grok contract;
   how it launches a lane; how it verifies (Tier-1); how it updates the ledger.
   VERDICT: is it generic enough to EXTEND into an Odyssey PATIENT queue, or is it
   hard-wired to the old Genesis gates (G001..)? Quote the state schema.

4. NOS GATE PIPELINE — `tools/nos_pipeline.py` (FULL). The gate stages
   (SPAWN/TIMING/DOCTOR/PROVENANCE/PROMOTE), `qualify_and_promote` signature, what
   each gate needs as input, how to call it standalone.

5. DOCTOR — `tools/gravity_doctor_gate.py`, `gravity_doctor_capability.py`,
   `gravity_doctor_dimensions.py`, `doctor_seal.py`. How do you run a "fast doctor"
   on a model? What inputs (logits? a live runtime? prompts? a reference model?)?
   Can it grade an EXTERNAL model run (MLX / transformers), or ONLY a hawking NX
   artifact? Seal JSON schema. Dimensions + controls that exist.

6. RUNTIME / ARCH SUPPORT — `crates/hawking-core/src/model/*`. Which model
   architectures does the Rust runtime actually support today (enumerate them)?
   Any Gemma / Falcon-H1 / Mamba-SSM / generic-MoE support, or only Qwen-family +
   DeepSeek-V4? Where is the arch dispatch / model-load entrypoint?

7. BASELINE TPS — how is a stock model's baseline TPS / token_ns measured here? Is
   MLX or transformers the external baseline runtime? Exact script path. How was
   the existing Qwen3-30B-A3B "~29.3 TPS" recon produced (search receipts/tools)?

8. EXISTING ODYSSEY CONTROL — distinguish the OLD training-data odyssey
   (`tools/odyssey/`: ingest/dedup/contamination) from any MODEL-PATIENT odyssey.
   Is there ANY existing patient-queue / patient-packet / transfer-matrix /
   gravity-rulebase machinery already? List what exists under
   `workspace/campaign/odyssey`, `receipts/`, `tools/`.

OUTPUT: one section per item with `file:line` refs and code-quoted signatures, then
a final **REUSE VERDICT** table: `component | reuse as-is | extend | build-new | note`.
