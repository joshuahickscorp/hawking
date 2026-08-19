# DELEGATION — PREDECLARED PATIENT MANIFEST (NEW ODYSSEY_MANIFEST.json, data-only)
Predeclare every Odyssey-I patient so the autonomous orchestrator never asks Claude "what source /
what target" (steer S004 §64/§65). Repo /Users/scammermike/Downloads/hawking. Branch odyssey-i.
Data file only; touch no code.

## BUILD `workspace/campaign/odyssey/ODYSSEY_MANIFEST.json`
Array of O000..O013 entries (use h_odyssey.md §35-48 + evidence/arch_archaeology for facts). Each:
{oxx, class(tiny-dense/hybrid/mm-dense/mm-moe/moe/large-dense/large-moe/streamed), canonical_source(repo),
gated(bool+reason), est_source_gib, est_4bit_gib, doctor_bar, stored_bpw_pressure, active_bpw_pressure,
tps_pressure_rel, search_class(cheap-lab/standard/deep/streamed), kernel_effort(low/moderate),
info_budget(low/med/high), arch_objective(stored-density/active-bytes/state-residency/modality/tokens-per-traversal/residency-io),
reference_sibling(or null), reopen_if, notes}. Sources: O000 google/gemma-3-1b-it(gated), O001 tiiuae/Falcon-H1-7B-Instruct,
O002 google/gemma-3-4b-it(gated), O003 moonshotai/Kimi-VL-A3B-Instruct, O004 mistralai/Mistral-Small-3.1-24B-Instruct-2503,
O005 Qwen/Qwen3-30B-A3B, O006 Qwen/Qwen3-VL-30B-A3B-Instruct(reference O005), O007 moonshotai/Kimi-Linear-48B-A3B-Instruct,
O008 ai21labs/AI21-Jamba-Mini-1.5, O009 Qwen/Qwen2.5-72B-Instruct, O010 zai-org/GLM-4.5-Air, O011 DSV4F(reconstruct from receipts),
O012 zai-org/GLM-4.5, O013 moonshotai/Kimi-K3(streamed, native-QAT). Mark UNKNOWN not guessed for sizes you cannot derive.

## Self-check: a tiny validator note in the file is not needed; ACCEPTANCE = valid JSON, 14 entries, each with canonical_source + arch_objective + reopen_if.
## SCOPE
WRITE workspace/campaign/odyssey/ODYSSEY_MANIFEST.json
READ h_odyssey.md, workspace/campaign/odyssey/evidence/arch_archaeology_O000_O001_O005_O010.md, workspace/campaign/odyssey/ODYSSEY.md
VERIFY the manifest by running `python3 -c "import json;d=json.load(open('workspace/campaign/odyssey/ODYSSEY_MANIFEST.json'));assert len(d)>=14 and all('canonical_source' in e and 'arch_objective' in e for e in d);print('manifest ok',len(d))"` — exit 0.
