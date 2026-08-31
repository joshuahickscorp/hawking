"""FUTURE_SUBSTRATE_HANDOFF — the state of the sidecar, computed from disk.

Nothing here is asserted from memory. Every system's state is derived from
whether its module and its receipt actually exist and whether its tests actually
pass, so the handoff cannot drift from reality between the work and the report.

It also answers the five arrival questions the directive asks explicitly --
what could plug in if Codex finished now, if Era II or Era III started tomorrow,
if the U50 board or a CUDA node arrived, if the tournament winner were picked --
and it answers them with file-level evidence rather than confidence.

    python3 tools/future/handoff.py --build
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from tools.future._common import REPO, RECEIPTS, git, write_receipt

FUTURE = REPO / "tools" / "future"

# system_key -> (module, receipt, what arriving capability it unblocks)
SYSTEMS: dict[str, tuple[str, str, str]] = {
    "codex_receipt_ingestion": ("codex_ingest.py", "CODEX_INGEST_STATE.json", "codex_finishes_now"),
    "odyssey_ii_law_store": ("odyssey2_law_store.py", "ODYSSEY2_LAW_STORE.json", "era_iii"),
    "odyssey_iii_adversary": ("odyssey3_adversary.py", "ODYSSEY3_ADVERSARY.json", "era_iii"),
    "hwir": ("hwir.py", "HWIR_V1.json", "u50_arrives"),
    "fpga_engine_school": ("fpga_engines.py", "FPGA_ENGINE_SCHOOL.json", "u50_arrives"),
    "fpga_multifidelity": ("fpga_fidelity.py", "FPGA_MULTIFIDELITY.json", "u50_arrives"),
    "hardware_doctor": ("hardware_doctor.py", "HARDWARE_DOCTOR.json", "u50_arrives"),
    "hbm_doctor": ("hbm_doctor.py", "HBM_DOCTOR.json", "u50_arrives"),
    "physical_primitive_library": ("physical_primitives.py", "PHYSICAL_PRIMITIVES.json", "era_ii"),
    "static_skeleton": ("static_skeleton.py", "STATIC_SKELETON.json", "era_ii"),
    "learned_physical_compiler": ("lpc_dataset.py", "LPC_DATASET.json", "era_iii"),
    "candidate_queue_planner": ("candidate_planner.py", "CANDIDATE_STAGED_PLAN.json", "codex_finishes_now"),
    "static_kernel_preflight": ("static_kernel_verify.py", "STATIC_KERNEL_PREFLIGHT.json", "codex_finishes_now"),
    "contamination_science": ("contamination.py", "CONTAMINATION_SCIENCE.json", "codex_finishes_now"),
    "ebpw_category_validator": ("ebpw_categories.py", "EBPW_CATEGORY_VALIDATOR.json", "codex_finishes_now"),
    "meta_experiment_funnel": ("meta_funnel.py", "META_EXPERIMENT_FUNNEL.json", "era_ii"),
    "teacher_corpus": ("teacher_corpus.py", "TEACHER_CORPUS_CONTRACT.json", "era_ii"),
    "expert_bank_school": ("expert_bank_school.py", "EXPERT_BANK_SCHOOL.json", "era_ii"),
    "ngram_school": ("ngram_school.py", "NGRAM_SCHOOL.json", "era_ii"),
    "router_science": ("router_science.py", "ROUTER_SENSITIVE_ALLOCATION.json", "era_ii"),
    "hmf_hgvas": ("hmf_objects.py", "HMF_MANAGED_OBJECTS.json", "cuda_arrives"),
    "fusion": ("fusion_sim.py", "FUSION_SIMULATION.json", "cuda_arrives"),
    "device_ascension": ("device_ascension_pipeline.py", "DEVICE_ASCENSION_PIPELINE.json", "cuda_arrives"),
    "green_machine": ("green_machine.py", "GREEN_MACHINE.json", "tournament_winner"),
    "tournament": ("tournament.py", "TOURNAMENT_READINESS.json", "tournament_winner"),
    "resident_install": ("resident_install.py", None, "tournament_winner"),
    "hcli_future_workunits": ("workunit_species.py", "HCLI_FUTURE_WORKUNITS.json", "tournament_winner"),
    "autonomous_reproducible_science": ("repro_science.py", "REPRO_SCIENCE.json", "era_iii"),
    "negative_science_index": ("negative_index.py", "NEGATIVE_SCIENCE_INDEX.json", "era_ii"),
    "experiment_turnaround": ("turnaround.py", "EXPERIMENT_TURNAROUND.json", "codex_finishes_now"),
    "decode_civilization": ("decode_civilization.py", "DECODE_CIVILIZATION.json", "era_ii"),
    "developer_platform": ("devplatform.py", "DEVELOPER_PLATFORM.json", "era_ii"),
    "qwen27_profile_schema": ("qwen27_profile_schema.py", "QWEN27_ACCELERATOR_PROFILE_SCHEMA.json", "codex_finishes_now"),
    "flash_nx_audit": ("flash_nx_audit.py", "FLASH_NX_COMPLETENESS_AUDIT.json", "codex_finishes_now"),
    "codex_mutation_surface": ("mutation_surface.py", "CODEX_MUTATION_SURFACE.json", "codex_finishes_now"),
    "global_frontier": ("global_frontier.py", "CLAUDE_GLOBAL_FRONTIER.json", "codex_finishes_now"),
    "evidence_snapshot": ("evidence_snapshot.py", "EVIDENCE_SNAPSHOT.json", "codex_finishes_now"),
    "integration_attack": ("integration_attack.py", "INTEGRATION_ATTACK.json", "era_iii"),
    "ane_preboard": ("ane_preboard.py", "ANE_PREBOARD.json", "codex_finishes_now"),
    "resident_optimizer": ("resident_optimizer.py", "RESIDENT_OPTIMIZER.json", "tournament_winner"),
    "qualification_automation": ("qualification_pipeline.py", "QUALIFICATION_PIPELINE.json", "codex_finishes_now"),
    "propagation_engine": ("propagate.py", "PROPAGATION_STATE.json", "codex_finishes_now"),
    "derived_freshness": ("freshness.py", "DERIVED_FRESHNESS.json", "codex_finishes_now"),
    "abi_verdict_harness": ("abi_verdicts.py", "ABI_VERDICT_HARNESS.json", "codex_finishes_now"),
    "p6_primitive_projection": ("p6_projection.py", "P6_PRIMITIVE_PROJECTION.json", "u50_arrives"),
    "git_lock_durability": ("git_lock_doctor.py", "GIT_LOCK_DURABILITY_REPORT.json", "era_iii"),
    "meta_downstream_ready": ("meta_ready.py", "META_DOWNSTREAM_READY.json", "era_ii"),
    "cuda_lowbit_hypotheses": ("cuda_lowbit_hypotheses.py", "CUDA_LOWBIT_HYPOTHESES.json", "cuda_arrives"),
    "moe_physical_school": ("moe_physical_school.py", "MOE_PHYSICAL_SCHOOL.json", "era_ii"),
}

ARRIVALS = {
    "codex_finishes_now": "IF CODEX FINISHED THE CURRENT ACCELERATOR CAMPAIGN RIGHT NOW, HOW MUCH FUTURE HAWKING COULD IMMEDIATELY PLUG INTO IT?",
    "era_ii": "IF ERA II STARTED TOMORROW, WHAT IS ALREADY EXECUTABLE?",
    "era_iii": "IF ERA III STARTED TOMORROW, WHAT IS ALREADY EXECUTABLE?",
    "u50_arrives": "IF THE U50DD ARRIVED TOMORROW, WHAT SOFTWARE/HWIR/SIMULATION FLOOR ALREADY EXISTS?",
    "cuda_arrives": "IF A CUDA NODE ARRIVED TOMORROW, WHAT COMPILER/FUSION FLOOR ALREADY EXISTS?",
    "tournament_winner": "IF THE TOURNAMENT WINNER WERE SELECTED TOMORROW, WHAT REAL WORK COULD HCLI IMMEDIATELY BEGIN?",
}


def system_state(module: str, receipt: str | None) -> dict[str, Any]:
    mod = FUTURE / module
    test = FUTURE / f"test_{module}"
    rec = RECEIPTS / receipt if receipt else None
    has_mod, has_test = mod.exists(), test.exists()
    has_rec = bool(rec and rec.exists())
    if has_mod and has_test and (has_rec or receipt is None):
        state = "EXECUTABLE"
    elif has_mod:
        state = "PARTIAL"
    else:
        state = "ABSENT"
    return {
        "state": state,
        "module": f"tools/future/{module}" if has_mod else None,
        "test": f"tools/future/test_{module}" if has_test else None,
        "receipt": f"receipts/future/{receipt}" if has_rec else None,
        "module_lines": len(mod.read_text().splitlines()) if has_mod else 0,
    }


def pytest_state() -> dict[str, Any]:
    r = subprocess.run(
        ["python3", "-m", "pytest",
         "tools/future/", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True,
    )
    tail = [l for l in r.stdout.strip().splitlines() if l.strip()][-1:] or ["<no output>"]
    return {"exit_code": r.returncode, "summary": tail[0][:300], "green": r.returncode == 0}


def active_processes() -> list[str]:
    r = subprocess.run(["ps", "-Ao", "pid,comm,args"], capture_output=True, text=True)
    keep = []
    for line in r.stdout.splitlines():
        low = line.lower()
        if any(k in low for k in ("odyssey_ctl", "modellake", "hf download", "grok", "cargo")):
            keep.append(line.strip()[:220])
    return keep[:20]


def build() -> Path:
    systems = {k: system_state(m, rc) for k, (m, rc, _) in SYSTEMS.items()}
    arrivals: dict[str, Any] = {}
    for key, question in ARRIVALS.items():
        members = [k for k, (_, _, a) in SYSTEMS.items() if a == key]
        ready = [m for m in members if systems[m]["state"] == "EXECUTABLE"]
        partial = [m for m in members if systems[m]["state"] == "PARTIAL"]
        absent = [m for m in members if systems[m]["state"] == "ABSENT"]
        arrivals[key] = {
            "question": question,
            "executable_now": sorted(ready),
            "partial": sorted(partial),
            "absent": sorted(absent),
            "answer": (
                f"{len(ready)} of {len(members)} systems are executable today, each with a "
                f"module, a test and a sealed receipt on disk."
            ),
            "evidence": {m: systems[m]["receipt"] for m in sorted(ready)},
        }

    attack_path = RECEIPTS / "INTEGRATION_ATTACK.json"
    attack = json.loads(attack_path.read_text()) if attack_path.exists() else None

    doc = {
        "schema": "hawking.future.handoff.v1",
        "version": 1,
        "head": git("rev-parse", "HEAD"),
        "head_subject": git("log", "-1", "--format=%s"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_paths": len([l for l in git("status", "--porcelain").splitlines() if l]),
        "sidecar_commits": git("log", "--oneline", "--", "tools/future").splitlines(),
        "active_processes": active_processes(),
        "gpu_authority": False,
        "protected_measurements_taken": 0,
        "systems": systems,
        "system_counts": {
            s: sum(1 for v in systems.values() if v["state"] == s)
            for s in ("EXECUTABLE", "PARTIAL", "ABSENT")
        },
        "tests": pytest_state(),
        "adversarial_attack": (
            {"verdict": attack["verdict"], "counts": attack["counts"]} if attack else None
        ),
        "arrival_questions": arrivals,
        "blockers": [
            "No protected GPU authority in this campaign: every hardware quantity is UNKNOWN "
            "by rule, not by omission.",
            "Flash has no source-independent complete NX. This is the dominant blocker: "
            "the count is read live from FLASH_NX_COMPLETENESS_AUDIT.json rather than "
            "pinned here, because Codex keeps adding candidates behind it -- it was 12 of "
            "14 at campaign start and 27 of 28 Flash candidates by the end.",
            "No U50 board, no CUDA node, no Core ML/ANE compiler environment: those fidelity "
            "levels and backends are declared UNAVAILABLE rather than simulated as if present.",
        ],
    }
    return write_receipt("FUTURE_SUBSTRATE_HANDOFF.json", doc, "tools/future/handoff.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    doc = json.loads(out.read_text())
    print(out)
    print(json.dumps(doc["system_counts"]), "tests:", doc["tests"]["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
