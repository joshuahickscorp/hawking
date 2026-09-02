# aud05 — FPGA software simulator and U50DD reconciliation

This is a discovery audit. Nothing was implemented. Evidence tier for this document is `STATIC_VERIFICATION`. This lane never wrote `MEASURED`. There is no U50DD on this host.

HEAD: `04193ccbc8ef9fdd2dfd595d65f656760829dddc`. Capability graph in-tree was generated from `7d642800` (not this HEAD).

## Answer

The FPGA software simulator is `tools/future/hwir.py` (6279 lines). It simulates a **host-CPU qGEMV numerical contract** plus **declared cost/cycle envelopes**. It does not simulate a U50DD, HBM2, PCIe, a bitstream, seconds, or tok/s.

What it **does** simulate, with the symbol that does it:

| Layer | Symbol | Evidence |
|---|---|---|
| Bit-exact qGEMV | `simulate_qgemv_functional` → `fpga_engines.qgemv` | `FUNCTIONAL_SIM` |
| Modelled critical path | `approximate_cycles` | `CYCLE_APPROX` (refuses seconds) |
| HBM bytes / host link bytes | `model_hbm_traffic`, `model_host_device_transfer` | `COST_MODEL` |
| LUT/DSP/BRAM/URAM fit | `estimate_qgemv_resources` / `fit_kernel_to_device` | `STATIC` |
| Row-split residency | `partition_qgemv` | `COST_MODEL` |
| Carrier downgrade | `constrain_device_profile` + `chestnut_current_firmware` | `THIRD_PARTY_REPORTED` overlay |
| Source emission | `HlsStyleEmitter` / `RustHdlEmitter` | PREHARDWARE text, not synthesis |

What it **does not** simulate: a board, XRT, Fmax, HBM2 physics, ASM2464PD training, DeltaNet/FFN/MTP as engines (`simulate_functional` returns `ok=False` unless `organ==qgemv`), place-and-route, DriverKit, Linux/XRT, or any `HARDWARE_MEASURED` quantity (`emit_evidence` refuses that tier).

Canonical functional kernel is **M=2, K=4**. A larger planning kernel (M=1024, K=4096) exists as COST_MODEL only.

Tests that actually call these symbols exist and **31 of them passed** in an isolated `/tmp` replica this lane (`0.19s`). That is software verification, not a U50 result.

Call-site law: `simulate_qgemv_functional` is **not** called from any non-test file other than `hwir.py` itself. CLI `--qgemv` makes it `CALLABLE`. That is not resident-runtime `INTEGRATED`. `HwirGraph` **is** constructed from `backend_contract.py`, `fusion_bridge.py`, `p6_projection.py`, and `propagate.py` — that is the IR, not the numeric simulator.

## Parallel stacks (do not collapse)

There is not one simulator. There are four, with three different interconnect numbers.

1. **`tools/future/hwir.py`** — canonical PREHARDWARE stack. Chestnut-aware *if* a `CarrierEnvelope` is passed. Default device is still `synthetic_u50_class` (DSP **9024**, PCIe UNPINNED), **not** U50DD (DSP **5952**).
2. **`hcli/agentos/fpga_preboard.py`** — organ maps + `TransportLinkSimulator(bandwidth_gbps=64.0)` (~8 GB/s) + `simulate_partition` with invented `apple_compute_ns` / `fpga_compute_ns`. CLI: `hcli agentos fpga-preboard`. Device genome is `TARGET_UNSELECTED`, `hbm_channels=0`. Schema `hcli.fpga.hwir.v1` is an organ list, not `HwirGraph`. `SimFPGAProvider` (roadmap PB-03) is **ABSENT**. `test_fpga_preboard.py` does not call the simulator symbols.
3. **`tools/future/fpga_fidelity.py`** — ANALYTICAL / FUNCTIONAL_SIM / CYCLE_MODEL on a **local `StructuralGraph`**. Comment still says swap it for `hwir.py` when F003 lands. HWIR has landed. Only test callers. HW_EMULATION and REAL_HARDWARE raise.
4. **`FpgaHwirBackend.execute`** (`tools/accelerator/backend_contract.py`), called from `tools/accelerator/repatriation_audit.py:76` — numpy **elementwise**, not `qgemv`. Bridge knob `DECLARED_BRIDGE_BW_GB_S = 12.0` GB/s.

Interconnect numbers in tree, none of them a Hawking measurement:

- Chestnut current firmware: **1.68e9 B/s**, PCIe Gen3 x2, `THIRD_PARTY_REPORTED`
- Preboard link sim: **64 Gbps ≈ 8 GB/s**
- Fusion bridge: **12.0 GB/s**
- “316 GB/s HBM / 188x cliff”: a **comment** in `hwir.py`. `DeviceProfile` has no HBM GB/s field.

## U50DD as it actually exists in source

`hwir.u50_family_profile("u50dd")` is selectable and tested. SKU `A-U50DD-P00G-ES3-G` (engineering sample). LUT 872000, DSP 5952, BRAM 1344, URAM 640, 8 GiB, 32 HBM channels, passive 75 W, PCIe Gen3 x16. Provenance: DS965 v1.2-era ES3 column + UG1371. Later DS965 v1.8 dropped that column.

Chestnut is encoded: `chestnut_current_firmware()` pins Gen3 x2, 1.68 GB/s completer ceiling (same at x4), **no airflow**. `chestnut_hawking_optimized()` is UNPINNED and **inherits** 1.68 GB/s until a Hawking measurement — it does not fail-open to a full slot. `constrain_device_profile(u50dd, chestnut)` drops the host-device beat to 2. `residency_ratio()` is the J.7 metric.

`tools/accelerator/hardware_doctor.py:782` **calls** `u50_family_profile("u50dd")` and the Chestnut constructors. That is a real non-test call of the U50DD profile.

What is *not* U50DD-bound: `HWIR_V1.json` preboard device (`synthetic-u50-class`), organ maps (`unselected-fpga-device`), fusion `DeviceBudget` (LUT `1<<16`, 1 HBM channel), link sim 64 Gbps, bridge 12 GB/s.

Inventory this lane: `ioreg` and `system_profiler SPThunderboltDataType` contain no Xilinx/Alveo/U50/ASM2464/Chestnut identity. `U50_PRESENT=false`. Inventory probe, not a performance measurement.

## Capability table

Vocabulary is the contract set. Imports are not callers. Tests are not board measurements.

| Capability | Status | Call of the symbol / blocker |
|---|---|---|
| HWIR | `INTEGRATED` | `HwirGraph(` at `backend_contract.py:1094`, `fusion_bridge.py:1444`, `p6_projection.py:857`, `propagate.py:695`. Second schema `hcli.fpga.hwir.v1` is a stub. |
| FPGA compiler | `SCAFFOLDED` | Lowering emitters TESTED as PREHARDWARE source. No vivado/vitis/v++ call site. Harness `CONTRACT_ONLY`. |
| software simulator | `TESTED` | `hwir.py:3464` calls `qgemv`. CLI `--qgemv`. No non-test caller outside `hwir.py`. |
| cycle model | `TESTED` | Two models (`approximate_cycles`, `fpga_fidelity.CycleModelProvider`). Both refuse seconds. No production caller outside tests/self-build. |
| HBM Doctor | `TESTED` | `solve()` callers are tests only. Receipt headline: `selected_ids=[]`, every Flash organ UNDECIDABLE. |
| hardware profiles | `TESTED` | `hardware_doctor.py:782` calls `u50_family_profile("u50dd")`. Catalog of `U50_DEVICE_PROFILE` points at the wrong file (`device_profiles.py`). |
| module cache | `SCAFFOLDED` | Organ-map `SCHEMA_ONLY`. No bitstream load. |
| placement | `INTEGRATED` | `kind_admits` from fusion_bridge / backend_contract / physical_graph_compiler. Not floorplanning. `kind_admits` returns True for every pair. |
| partitioning | `TESTED` | `partition_qgemv` (COST_MODEL row-split) tested. `simulate_partition` is a 64 Gbps scenario table. Qwen27 `within_ffn_split` asks ~225 GiB resident vs 8 GiB. |
| qGEMV | `TESTED` | Engine-school golden + HWIR wrap. Ladder **L1**. `U50_FIRST_NATIVE_ENGINE` stays `BLOCKED_HARDWARE`. |
| FFN organ split | `SCAFFOLDED` | Scenario literal + 65/35 prior recorded and **not applied**. `simulate_functional` does not run FFN. |
| persistent DeltaNet state | `SCAFFOLDED` | Organ + `from_organ_map` state owner + numpy `recurrent_state_transition`. Not wired into `simulate_functional`. |
| partial reductions | `SCAFFOLDED` | Frame kind + `c2h = rows*4` + scenario bytes. No fabric reduction circuit. |
| transport overlap | `SCAFFOLDED` | Cycle approx is H2C then max(compute,HBM) then C2H — **no** overlap with the host link. |
| MTP verification | `CONCEPT_ONLY` | Scenario table + name map. No verifier function called. |
| representation-native arithmetic | `TESTED` | Host goldens (qgemv/codebook/NF4). Not Appendix K.1 FPGA tournament. |
| DriverKit / direct-Mac | `CONCEPT_ONLY` | `DESIGN_NOT_IMPLEMENTED`. No dext / IOUserPCI path in HEAD. |
| optional Linux fixture | `ABSENT` | Appendix J.3 prose only. No module. |
| HMF/HGVAS integration | `BLOCKED_HARDWARE` | `HMF_PRESENT=false`. Selftest migrate to `FPGA_HBM_0` records UNKNOWN. |
| physical graph integration | `INTEGRATED` | `three_domain_plan` + `physical_graph_compiler` subprocess. FPGA domain `physical=False`. Mixed executable `BLOCKED_HARDWARE`. |

## Which gates the simulator may move

From a naive “FPGA is ABSENT” reading, the simulator **already** moved:

- `FPGA_HWIR`: not ABSENT. Capability graph `WIRED` (IR construction) is the right graph status. This audit adds `TESTED` for the qGEMV software simulator in the audit vocabulary. Not `ACCEPTED`. Not tok/s.
- `FPGA_PREBOARD_SCHEMAS` / `FPGA_LINK_SIM` / `FPGA_PARTITION_SIM`: not ABSENT. `SCAFFOLDED`/`CALLABLE`. **Do not** promote link/partition to U50 evidence while the producer is 64 Gbps + invented nanoseconds + `TARGET_UNSELECTED`.

The simulator **must not** move any of:

`U50_PURCHASE_ACCEPTANCE`, `U50_SAFE_COOLING`, `U50_DEVICE_PROFILE` (measured J.4), `U50_DMA_HBM`, `U50_FIRST_NATIVE_ENGINE`, `U50_MIXED_APPLE_FPGA_GRAPH`, `U50_34_TO_40` … `U50_80_TO_90`, `HMF_DEVICE_VISIBLE_TRUST`, `FUSION_FIRST_HETEROGENEOUS_EXECUTABLE`.

A numpy golden is not a native engine. Modelled cycles are not 40 tok/s. Appendix K.5 already forbids publishing 80–90 TPS because a bandwidth roof is larger.

## Generic FPGA branches

**Strike as a deliverable (or alias, do not build a fourth simulator):** PB-03 `SimFPGAProvider`; 64 Gbps as planning truth; `TARGET_UNSELECTED` organ maps after U50DD profiles exist.

**Demote to future-board generalisation:** `synthetic_u50_class` (keep, never default); U50C (all UNPINNED); `fpga_fidelity.StructuralGraph`; fusion invented `DeviceBudget` and 12.0 GB/s bridge; `hcli.fpga.hwir.v1` as a stub not the IR; two `hardware_doctor.py` files (sidecar ranker vs host+absent-U50DD census — catalog `U50_PURCHASE_ACCEPTANCE` points at the sidecar).

**After U50 tuition:** U50 / U50LV variants; custom Hawking FPGA board (roadmap Phase XVI); DFX / module cache / overlay; DriverKit dext; Linux/XRT as canonical runtime (J.3 is optional qualification, then return to Mac); K.1 arithmetic tournament; engine ladder L5–L8; MTP on FPGA; persistent DeltaNet circuit; TPS gates.

## Proposal (not implemented)

Evidence ladder, in order: **VENDOR → DERIVED → SIMULATED → MEASURED → REPRODUCED → PROTECTED VERIFIED**.

Hard rule: no simulator result becomes a U50 performance claim.

1. Default planning tuple: `(device=u50dd, carrier=chestnut-current-fw)`. Today’s default is the generic class envelope.
2. One interconnect story. Nothing in-tree may plan against a beat more optimistic than Chestnut current-fw until Hawking measures. Label it `THIRD_PARTY_REPORTED`, not `MEASURED`. Stop emitting `bandwidth_gbps` from the preboard link sim.
3. Organ maps stamp U50DD STATIC vendor fields with `physical_board_present=false`.
4. Canonical numeric simulator remains `hwir.run_qgemv_preboard`. Do not add `SimFPGAProvider`. Wire or freeze `fpga_fidelity`.
5. Either source HBM GB/s from vendor literature or mark it `UNPINNED`. Do not leave the 188x cliff as a comment.
6. `residency_ratio` is the admission test (J.7). A transport bottleneck triggers residency redesign, not a faster-cable purchase.
7. Gate map: `FPGA_HWIR` may stay WIRED/TESTED; link/partition stay SCAFFOLDED as U50 evidence until rebound; all `U50_*` stay `BLOCKED_HARDWARE`. Point `U50_DEVICE_PROFILE` at `hwir.py`.
8. When the board exists, execute Appendix J in order. Direct-Mac is canonical. J.3 Linux is optional qualification. J.2: the card is passive and Chestnut supplies no airflow — do not power without forced air.
9. Do not start a Linux-first or generic-FPGA campaign. Five eras, three Odysseys, FPGA inside Accelerator/Fusion.

## Surprises (loud)

1. Four simulators, three interconnect numbers. A single `CarrierEnvelope` used by hwir, preboard, and fusion_bridge, with a test that fails if any beat exceeds Chestnut current-fw, would settle this.
2. `HWIR_V1.json` defaults to `synthetic-u50-class` after U50DD profiles exist. Changing the default and mutation-testing it would settle this.
3. Organ maps still `TARGET_UNSELECTED` / `hbm_channels=0`, so `from_organ_map` cannot bind an HBM channel from the map.
4. `fpga_fidelity.py` still waits for HWIR to land.
5. 316 GB/s is not a `sourced_field`.
6. Production `FpgaHwirBackend.execute` is elementwise numpy, not qGEMV.
7. `test_fpga_preboard.py` does not test the simulator it sits next to.
8. Canonical FUNCTIONAL_SIM is 2×4.
9. HBM Doctor selected nothing. The receipt name sounds finished; the headline is UNDECIDABLE.
10. Qwen27 FFN-split scenario residents ~225 GiB on an 8 GiB card.
11. U50DD numbers come from a DS965 column later dropped; SKU is an engineering sample. J.1 seller/local verification is the settlement.
12. Capability graph commit ≠ this HEAD.
13. Catalog `U50_DEVICE_PROFILE` → `device_profiles.py`, which has no U50DD table.
14. `fusion_bridge.py:1444` `HwirGraph(` sits inside `lower_fpga_domain_to_hwir`, whose only callers are tests. Load-bearing non-test IR construction is `backend_contract` (via `repatriation_audit.py:76` `fpga.execute` / `.lower`). Flagged rather than smoothed: a call inside an uncalled function is easy to count as wired.

## Tests this lane ran

Isolated replica `/tmp/aud05-pytest` (HEAD blobs; did **not** write `receipts/future/` in the worktree).

```
31 passed in 0.19s
```

Smoke: `simulate_qgemv_functional` → `y=[10.0, 1.5]`, `HARDWARE_MEASURED` refused, U50DD DSP=5952, Chestnut-constrained host-device beat=2.

Full `tools/future` pytest was not run in the sparse worktree because those files are not materialized and `build()` would rewrite sealed receipts this lane is not allowed to touch.

## Constitution

Held: five eras, three Odysseys, no Era VI, FPGA inside Hawking Accelerator/Fusion, Theia is one generalist bounty model. North star unchanged.
