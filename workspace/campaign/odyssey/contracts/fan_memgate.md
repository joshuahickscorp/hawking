# DELEGATION — MEMORY GATE + MULTI-MODEL PARALLEL (NEW module tools/odyssey_memgate.py ONLY)
Build `tools/odyssey_memgate.py` — a standalone deterministic memory-admission module so the
Odyssey loop can run MULTIPLE model experiments concurrently (user directive), bounded by swap.
Repo /Users/scammermike/Downloads/hawking; py /Library/Frameworks/Python.framework/Versions/3.12/bin/python3. Branch odyssey-i.
Do NOT edit ctl.py or the runner (parallel lanes own them); this is a self-contained importable module.

## API
- `observe()` -> dict {free_ram_gib, wired_gib, compressor_gib, swap_used_gib, swap_total_gib, cpu_load}. Use `vm_stat`, `sysctl vm.swapusage`, `memory_pressure` (parse macOS output; stdlib subprocess).
- `SWAP_MAX_GIB = 30` (module constant; also read override from workspace/campaign/odyssey/ODYSSEY_POLICY.json `detachment.memory` if present).
- `admit(est_gib, in_flight_gib=0, clean_room=False) -> {decision: GO|REFUSE, note, projected_swap_gib}`.
  Rule: REFUSE if `clean_room` and any other model worker is in flight (protected-timing needs exclusivity — clean_room ONLY for protected TPS timing, NOT for ordinary SPECIMEN experiments).
  Otherwise GO while projected memory keeps `swap_used + max(0, (wired+in_flight+est) - (physical_ram - reserve))` <= SWAP_MAX_GIB AND free_ram stays > a small reserve. Admit MULTIPLE concurrent model lanes as long as projected swap <= 30 GiB. est_gib defaults sensibly for a 4-bit MoE (~16) if unknown.
- `capacity(est_gib_each) -> int` : how many concurrent model lanes of that size fit under the swap bound now.
- physical RAM = 96 GiB (M3 Ultra); reserve ~= 12 GiB for OS/Metal/build headroom.

## Self-check (`python3 tools/odyssey_memgate.py --self-check`, exit 0)
Assert: with injected snapshots, admit GOes for several 16 GiB lanes under low swap; REFUSEs a clean_room lane when another is in-flight; REFUSEs once projected swap would exceed 30; capacity() monotone in est size. No network, no model.

## SCOPE
WRITE tools/odyssey_memgate.py
READ workspace/campaign/odyssey/ODYSSEY_POLICY.json
VERIFY tools/odyssey_memgate.py by running `python3 tools/odyssey_memgate.py --self-check` — exit 0.
