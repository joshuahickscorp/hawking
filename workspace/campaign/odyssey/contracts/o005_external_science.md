# DELEGATION B — O005 CANONICAL EXTERNAL SCIENCE (gate profile: MLX/Metal + network)

Patient O005 = `Qwen/Qwen3-30B-A3B` (CANONICAL, Apache-2.0), now on disk at
`~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
Repo: `/Users/scammermike/Downloads/hawking`. This produces the external-runtime
baseline + route map + fast-Doctor for the O005 packet (obligations A1/A2/A3).

The native Hawking Rust runtime does NOT support Qwen3-MoE yet (`load_engine`
raises Unimplemented for qwen3moe — see `workspace/campaign/odyssey/evidence/reuse_surface.md`).
So the baseline is an EXTERNAL SPECIMEN via mlx_lm, exactly like the prior recon.

## Read first
- `workspace/campaign/odyssey/a3b_recon.py` — the EXISTING mlx recon (route-tap + battery + tps). GENERALIZE it; do not start from scratch. It targeted the abliterated checkpoint at `workspace/campaign/records/runs/qwen3-30b-a3b/bf16`.
- `receipts/ascent-2026-08-18/A3B_RECON.json` — prior result on the ABLITERATED checkpoint (entropy 6.09/7.0, 0 cold experts, top16=18% mass, most-popular 1.42%, battery 10/12, refusals 0/2, tps_specimen 29.3). You will GROUND canonical vs this and report the delta.
- `workspace/campaign/odyssey/evidence/arch_archaeology_O000_O001_O005_O010.md` (O005 section: softmax→top8→renorm, no shared, 48/48 MoE, enable_thinking template, transformers>=4.51).
- `tools/worker_gate.py` — `observe()/gate(obs)`; MUST call before loading the model; abort on REFUSE.

## SETUP
- `pip install` mlx + mlx_lm into `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` (Apple-Silicon wheels; network is on under gate). mlx_lm is NOT currently installed — verify import after install.

## BUILD `tools/odyssey_patient_runner.py` (generalize a3b_recon.py)
Args: `--oxx O005 --weights <snapshot_dir> --runtime mlx --route-tokens 512 --out receipts/odyssey-i/O005_EXTERNAL.json`.
- Memory gate: call worker_gate before load; if REFUSE, do NOT load bf16 — instead `mlx_lm.convert -q` to 4-bit (~16 GB), LABEL the run `quant=4bit-mlx` and note fidelity caveat. Prefer bf16 if admitted (`quant=bf16`).
- Baseline: `tps_specimen` = tokens/wall for one `generate(max_tokens=64)` after a warmup; also record TTFT. LABEL SPECIMEN + contamination-aware (§14: this is NOT BASE_TRUE_TPS; a session may be open).
- Route map (MoE): tap each MoE layer's router over `--route-tokens` real tokens; emit per-layer + aggregate: expert frequency, route entropy bits (avg + max=log2(128)=7), most-popular-expert share, never-routed count, top-16 mass %, adjacent-token route overlap and P(E_t|E_t-1) transition stability, cross-layer co-occurrence (§57). Hot/cold verdict.
- Fast-Doctor: run the battery from a3b_recon (correctness pairs) + refusal controls; score X/12 and refusals; the controls MUST be able to fire (a real control must be watched to fail — §60 doctor-control gate). Produce a `doctor_seal.seal`-compatible seal via `tools/doctor_seal.py`.
- Config assertions (from archaeology NEXT): assert router path is softmax→top-8→renormalize, no shared expert, 48/48 layers MoE; assert `enable_thinking=False` changes only the template (first tokens differ, weights identical). Record pass/fail.
- Fill `workspace/campaign/odyssey/patients/O005/ODYSSEY_PATIENT_O005.json` execution + routing + doctor fields from the receipt (conform to `patient_packet_schema.json`).

## GROUND vs prior
In the receipt, add a `canonical_vs_abliterated` block: canonical route entropy / cold-expert count / battery vs A3B_RECON.json; state whether the abliterated route classification (uniform routing, MoE-universal sparse path, no cold experts) HOLDS on canonical. Label each MEASURED/DERIVED.

## Constraints
- SPECIMEN labels everywhere; never present mlx TPS as a Hawking native number.
- Do not delete the canonical weights.
- Keep the model loaded once; do baseline+route+doctor in one process (one load, memory-safe).

## ACCEPTANCE
- `receipts/odyssey-i/O005_EXTERNAL.json` exists with: quant, tps_specimen, ttft, route{entropy_avg,entropy_max,cold_experts,top16_mass_pct,most_popular_share,transition_stability}, doctor{battery,refusals,seal_ref}, config_assertions{router_ok,moe_layers,thinking_template_ok}, canonical_vs_abliterated.
- The O005 packet is updated and still schema-valid.

## VERIFY (runnable)
```
python3 -c "import mlx_lm" && \
python3 tools/odyssey_patient_runner.py --oxx O005 --weights ~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 --runtime mlx --route-tokens 512 --out receipts/odyssey-i/O005_EXTERNAL.json && \
python3 -c "import json;d=json.load(open('receipts/odyssey-i/O005_EXTERNAL.json'));assert d['route']['entropy_avg']>0 and 0<=d['route']['entropy_max']<=7.001 and d['doctor']['battery'];print('O005 external ok:',d['tps_specimen'],'tps',d['route']['entropy_avg'],'bits')"
```
Report the receipt, the delta vs abliterated, and any REFUSE/quant fallback that fired.

## SCOPE
WRITE tools/odyssey_patient_runner.py
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/patients/O005/
READ workspace/campaign/odyssey/a3b_recon.py, tools/worker_gate.py, tools/doctor_seal.py, receipts/ascent-2026-08-18/A3B_RECON.json
VERIFY tools/odyssey_patient_runner.py by running it on O005 then testing the receipts/odyssey-i/O005_EXTERNAL.json fields — must pass, exit 0.
