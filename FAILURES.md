# FAILURES.md — structured failures (first-class artifacts)

> Spec source: `docs/plans/studio_maximization_2026_06_27.md` §20.5 / §20.11.
> Failures are deliverables, not embarrassment (§19.3 #4). Each entry carries a **real** receipt
> (`receipts/failures/<id>.json`, gate=fail) and an exact reproduce command. A failure that fires
> a §20.1 "kills the wedge" condition forces the full portfolio reframe; everything else just
> re-orders the work.
>
> **Severity legend:** `warn` (within a band) · `fail` (misses the gate) · `wedge-threat` (fires a
> §20.1 kill condition).

---

## FAIL-001 — 7B LoRA/blockwise recovery swap-death on 18 GB

- **model / family / size:** qwen / qwen2 / 7B (`scratch/qwen-7b`, bf16 safetensors — the real doctor input)
- **recipe / config:** `3-AWQ+dr-r16-v3`, `4-AWQ+dr-r16-v3`, blockwise doctor v3 (commit `42a0e9e`)
- **what was expected:** the v3 doctor configs complete a 7B recovery pass and emit a condensed artifact + receipt.
- **what happened (measured):** every v3 config swap-dies on the 18 GB box. Real kills from
  `reports/cron/7b_frontier.jsonl`:
  - `doctor killed: swap 6926MB > 6000MB ceiling (bits=3)`
  - `doctor killed: swap 9003MB > 6000MB ceiling (bits=3)`
  - `doctor killed: swap 11780MB > 6000MB ceiling (bits=4)`
  - `doctor killed: swap 6014MB > 6000MB ceiling (bits=1)`
  - `doctor timeout: 120m > 120m limit (bits=1/2/3)`
  - `doctor failed bits=2/3: ... resource_tracker: 1 leaked semaphore object`
  - run log shows peak `swap=25305MB`; conductor `branch=waiting-for-v3`, `candidate_count=0`.
  No artifact was ever emitted.
- **receipt:** `receipts/failures/FAIL-001.json` (a real receipt, `gate=fail`, `claim_type=negative`)
- **reproduce:** `./run_7b_frontier.sh` (drives `frontier_conductor.py --outbase reports/cron/7b_frontier`);
  the v3 doctor dies at the swap ceiling / 120-min timeout on any ≤18 GB machine.
- **category / tags:** `[memory-prediction | recovery-loss]` — a measured hardware floor, not a quality loss.
- **severity:** `fail` (hardware floor) — **NOT** a `wedge-threat`. This is the exact 18 GB dead-end the
  Mac Studio M2 Max (96 GB) is bought to clear; it does **not** fire a §20.1 kill condition.
- **roadmap effect:** confirms §3 / §21 STUDIO-ONLY boundary — the clean 7B floor point and the full-rank
  / blockwise QAT are Studio work. The fix is RAM, not recipe; the branch stays parked at `waiting-for-v3`.
- **pivot trigger?:** no — predicted by the plan; the Studio resolves it.

---

_New failures append below this line, one per `## FAIL-NNN` block, each with its receipt._
