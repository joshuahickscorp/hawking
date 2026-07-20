# VULTURE HANDOFF PRECHECK

2026-07-20. Harvest 120B, close it without demanding perfection, full-disk Qwen3-235B, next parent.

## Live truth

- 120B Doctor campaign RUNNING (do not touch): pid 42390, pgid 42390, exclusive lease
  `com.hawking.doctor_campaign`, row 15/28 (`reason_syllogism__D4_pq_doctor`), 14 sealed, `final=false`,
  stable since the byte-budget fix (3 prior memory-pressure kills healed). Candidate set verified: D0
  (6) + diagnosis mlp1/mlp2 (4) + D2 (6) + D4 (6) + D6 (6) = 28; D1 = the sealed G4 untreated control;
  D3/D5 = non-admission receipt (dominated by G1/G3).
- Git: `campaign/adaptive-transfer-ladder` @ `514212c9`, in sync with origin.
- Overnight supervisor LIVE in `WAIT_120B_FINAL` (launchd, 60s tick, survives reboot, auto-resumes a
  crashed campaign). Doctor supervisor retired. Telegram carries per-row card + ETA + fault.
- Resources: RAM avail 31 GB, disk 514 GiB free (575 after 120B release), swap 1.4 GB.

## Binding amendments applied to the supervisor this session

- **No refinement:** on `final=true` it verifies, seals pass OR honest boundary, and MOVES. The
  `NARROW_RATE_REFINEMENT` state is removed; `SEAL -> VULTURE_HARVEST -> EVALUATE_SOURCE_RELEASE`.
- **No further 120B compute:** the current campaign is the final permitted 120B compute.
- **Result does not block Qwen:** pass or boundary both proceed; the result sets Qwen's priors only.
- **Vulture harvest (Lane A):** a new `VULTURE_HARVEST` state runs `vulture_harvest.py` and seals the 8
  harvest artifacts (transfer priors, failure/Doctor/resource atlases, runtime lessons, rehydration)
  BEFORE the body is deleted (release gate 7 enforces harvest-sealed).
- **Qwen FULL_DISK_RESIDENT (Lane B):** the transfer now downloads ALL 118 official shards (not the
  priority subset) into one local root and keeps them for the whole Qwen campaign. Execution stays
  bounded (mmap, range reads, one-expert window, PressureAwareCache, 12 GB RAM floor, swap guard).

## Storage math

- Qwen 437.9 GiB fits full-disk after the 120B release: 575 - 438 = ~137 GiB free (>= 100 GiB preferred,
  >= 40 GiB hard reserve). One local copy, resumable, per-shard verified, no HF-cache duplicate.
- 235B never fits in 96 GB RAM; full-disk means all shards on DISK, compute streams per-expert.

## Giants (disk, precision-driven not param-count)

397B Qwen3.5 bf16 751 GiB (shard-serial), 685B DeepSeek fp8 642 GiB (shard-serial), 1T Kimi int4 554
GiB (barely full-disk, decide live). Draft `KIMI_1T_FULLDISK_DECISION_DRAFT.json`.

## Rollback

`git reset --hard 514212c9`; `kill 42390` stops the campaign; no source mutated; Qwen metadata only.
Deletion of the 7 gpt-oss shards happens only after 15/15 release gates incl. Vulture-harvest-sealed.
