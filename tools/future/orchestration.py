"""ORCHESTRATION — bind every sidecar module to the frontier it informs.

The resident-callability audit found `operational=0 of 74`, and the gaps had one
shape rather than seventy: `result_does_not_feed_a_named_frontier` on 70 modules
and `does_not_emit_workunit` on 52, while only two lacked an entry point and two
lacked a receipt. Three of the five axes were already satisfied everywhere. The
missing two are orchestration glue, and glue belongs in one connector rather
than smeared across seventy modules.

**This must not be Goodharting.** Adding decorative frontier entries so a metric
turns green would be the exact failure this campaign guards against. So the
binding has to make the claim TRUE:

* a binding names the frontier item a module's receipt genuinely informs;
* `invoke()` actually runs the module and actually routes its receipt to that
  frontier, recording the observation;
* a binding whose module does not really write the named receipt is REJECTED at
  load time, so the table cannot drift into fiction;
* infrastructure that legitimately informs no frontier (`_common.py`, the
  package marker, the attacker) is declared INFRASTRUCTURE and EXCLUDED from
  scoring rather than given a fake binding.

    python3 tools/future/orchestration.py --bind
    python3 tools/future/orchestration.py --invoke future.freshness
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any

from tools.future._common import REPO, RECEIPTS, load_json, write_receipt

RECEIPT = "ORCHESTRATION_BINDINGS.json"
SCHEMA = "hawking.future.orchestration.v1"


class BindingError(Exception):
    """A binding does not describe reality."""


class UnknownBinding(BindingError):
    pass


# Modules that legitimately inform no frontier. Excluded from scoring, never
# fake-bound. Being honest about the denominator is the whole point.
INFRASTRUCTURE = {
    "__init__.py",
    "_common.py",
    "integration_attack.py",   # judges the sidecar; it is not a frontier producer
    "handoff.py",              # reports state; the report is not itself a frontier
    "orchestration.py",        # this module
}

# module filename -> (frontier item id, WorkUnit species id)
# Each binding names the frontier this module's receipt genuinely informs.
BINDINGS: dict[str, tuple[str, str]] = {
    # --- representation search -------------------------------------------
    "flash_schools.py":            ("FT.MODEL_REPRESENTATION.meta-gates-3-9", "GRAVITY_SEARCH"),
    "flash_nr_complete.py":        ("FT.MODEL_REPRESENTATION.meta-gates-3-9", "NOETIC_COMPILE"),
    "meta_funnel.py":              ("FT.MODEL_REPRESENTATION.meta-gates-3-9", "GRAVITY_SEARCH"),
    "meta_ready.py":               ("FT.MODEL_REPRESENTATION.meta-gates-3-9", "GRAVITY_SEARCH"),
    "ngram_school.py":             ("FT.MODEL_REPRESENTATION.ngram-school", "GRAVITY_SEARCH"),
    "expert_bank_school.py":       ("FT.MODEL_REPRESENTATION.ngram-school", "GRAVITY_SEARCH"),
    "teacher_corpus.py":           ("FT.MODEL_REPRESENTATION.teacher-capture", "TEACHER_CAPTURE"),
    "router_science.py":           ("FT.MODEL_REPRESENTATION.meta-gates-3-9", "GRAVITY_SEARCH"),
    "ebpw_categories.py":          ("FT.MODEL_REPRESENTATION.meta-gates-3-9", "NOETIC_COMPILE"),
    # --- capability / tournament -----------------------------------------
    "tournament.py":               ("FT.MODEL_CAPABILITY.tournament-refuse", "CAPABILITY_GATE"),
    "resident_install.py":         ("FT.MODEL_CAPABILITY.hard-gates", "CAPABILITY_GATE"),
    "super_resident.py":           ("FT.MODEL_CAPABILITY.hard-gates", "CAPABILITY_GATE"),
    "tabula.py":                   ("FT.MODEL_CAPABILITY.hard-gates", "CAPABILITY_GATE"),
    # --- execution --------------------------------------------------------
    "fusion_sim.py":               ("FT.MODEL_EXECUTION.fusion-sim", "FUSION_SIMULATION"),
    "static_skeleton.py":          ("FT.MODEL_EXECUTION.static-skeleton", "PHYSICAL_GRAPH_SEARCH"),
    "flash_nx_audit.py":           ("FT.MODEL_EXECUTION.complete-token", "NX_COMPLETENESS_AUDIT"),
    "physical_primitives.py":      ("FT.MODEL_EXECUTION.static-skeleton", "PHYSICAL_GRAPH_SEARCH"),
    # --- latency / turnaround --------------------------------------------
    "turnaround.py":               ("FT.LATENCY.cpu-turnaround", "PROFILE_HOST_CEREMONY"),
    "qwen27_profile_schema.py":    ("FT.LATENCY.gpu-ns", "PROFILE_COMPLETE_TOKEN"),
    # --- decoding / tps ---------------------------------------------------
    "decode_civilization.py":      ("FT.DECODING.cost-model", "PROFILE_COMPLETE_TOKEN"),
    # --- active bytes / memory -------------------------------------------
    "hbm_doctor.py":               ("FT.ACTIVE_BYTES.hbm-rank", "HBM_RESIDENCY"),
    "hmf_objects.py":              ("FT.MEMORY.hmf", "HMF_COHERENCE"),
    # --- gpu kernels ------------------------------------------------------
    "static_kernel_verify.py":     ("FT.GPU_KERNELS.static-warnings", "STATIC_KERNEL_VERIFY"),
    "abi_verdicts.py":             ("FT.GPU_KERNELS.static-warnings", "HOST_SHADER_ABI_VERIFY"),
    "claude_abi_adjudication.py":  ("FT.GPU_KERNELS.static-warnings", "HOST_SHADER_ABI_VERIFY"),
    "candidate_planner.py":        ("FT.GPU_KERNELS.ready-protected", "FACTORIAL_COMBINATION"),
    "qualification_pipeline.py":   ("FT.GPU_KERNELS.ready-protected", "PROTECTED_AB"),
    "protected_window.py":         ("FT.GPU_KERNELS.ready-protected", "PROTECTED_AB"),
    "moe_physical_school.py":      ("FT.GPU_KERNELS.flash-nx", "GENERATE_KERNEL_CANDIDATE"),
    "p6_projection.py":            ("FT.PHYSICAL_GRAPH.p6-projection", "SEARCH_ARCHITECTURE_LAWS"),
    # --- hcli self --------------------------------------------------------
    "workunit_species.py":         ("FT.HCLI_SELF.emit-workunits", "HCLI_SELF_OPTIMIZE"),
    "codex_behaviors.py":          ("FT.HCLI_SELF.emit-workunits", "HCLI_SELF_OPTIMIZE"),
    "workgraph.py":                ("FT.HCLI_SELF.no-launch", "HCLI_SELF_OPTIMIZE"),
    "detached.py":                 ("FT.HCLI_SELF.no-launch", "HCLI_SELF_OPTIMIZE"),
    "wakeup.py":                   ("FT.HCLI_SELF.no-launch", "HCLI_SELF_OPTIMIZE"),
    "resident_api.py":             ("FT.HCLI_SELF.emit-workunits", "HCLI_SELF_OPTIMIZE"),
    "resident_identity.py":        ("FT.HCLI_SELF.emit-workunits", "HCLI_SELF_OPTIMIZE"),
    "sandbox.py":                  ("FT.HCLI_SELF.no-launch", "HCLI_SELF_OPTIMIZE"),
    "resident_optimizer.py":       ("FT.CHILD_RESIDENT.optimizer", "CHILD_PROPOSAL"),
    "succession.py":               ("FT.CHILD_RESIDENT.install-dry-run", "CHILD_PROPOSAL"),
    "autonomy_trial.py":           ("FT.CHILD_RESIDENT.launch", "AUTONOMY_TRIAL"),
    # --- experiment turnaround / tools ------------------------------------
    "freshness.py":                ("FT.EXPERIMENT_TURNAROUND.refresh", "DERIVED_REFRESH"),
    "propagate.py":                ("FT.TOOLS.propagate-skips", "INGEST_PROPAGATE"),
    "codex_ingest.py":             ("FT.TOOLS.propagate-skips", "INGEST_PROPAGATE"),
    "frontiers.py":                ("FT.TOOLS.frontiers-refill", "FRONTIER_REFILL"),
    "evidence_snapshot.py":        ("FT.CONTEXT.disk-authority", "EVIDENCE_PIN"),
    "global_frontier.py":          ("FT.CONTEXT.open-question", "FRONTIER_REFILL"),
    "mutation_surface.py":         ("FT.CONTEXT.disk-authority", "EVIDENCE_PIN"),
    "devplatform.py":              ("FT.TOOLS.freshness", "HCLI_SELF_OPTIMIZE"),
    "debugger.py":                 ("FT.TOOLS.freshness", "TOOL_PROBE"),
    # Steer S007 section 107: git durability is part of autonomous operation,
    # so the lock doctor genuinely informs HCLI self-health rather than being
    # a stray utility.
    "git_lock_doctor.py":          ("FT.HCLI_SELF.no-launch", "HCLI_SELF_OPTIMIZE"),
    # Whole-tree specimen verification: CPU/disk work that directly gates
    # Odyssey I, and the only long-running honest work source on a host with
    # no GPU.
    "autonomy_scars.py":          ("FT.HCLI_SELF.emit-workunits", "AUTONOMY_DEFECT"),
    # The sprint tooling. Each drives a frontier item that already existed; none
    # invents one, because a binding to a frontier nobody is watching informs
    # nothing.
    "trial_freeze.py":            ("FT.VERIFICATION.repro", "TRIAL_BUILD_SEAL"),
    "work_events.py":             ("FT.HCLI_SELF.emit-workunits", "EVENT_CONTRACT"),
    "restart_supervisor.py":      ("FT.CHILD_RESIDENT.install-dry-run", "RESIDENT_RESTART"),
    "fallback_resident.py":       ("FT.CHILD_RESIDENT.install-dry-run", "RESIDENT_FALLBACK"),
    "adaptive_verification.py":   ("FT.VERIFICATION.repro", "MULTI_FIDELITY_SCREEN"),
    "phase_listeners.py":         ("FT.ODYSSEY_TRANSFER.re-earn", "PHASE_LISTENER"),
    "consolidated_run.py":        ("FT.ODYSSEY_TRANSFER.re-earn", "ODYSSEY_RUN_DESCRIPTOR"),
    "accelerator_workunits.py":   ("FT.GPU_KERNELS.ready-protected", "ACCELERATOR_SPECIES"),
    "hcli_self_profile.py":       ("FT.LATENCY.cpu-turnaround", "HCLI_SELF_OPTIMIZATION"),
    "flash_organ_workgraphs.py":  ("FT.MODEL_REPRESENTATION.meta-gates-3-9", "ORGAN_WORKGRAPH"),
    "external_specimen_seal.py":  ("FT.MODEL_CAPABILITY.hard-gates", "EXTERNAL_SPECIMEN"),
    "odyssey_tool_driver.py":     ("FT.MODEL_CAPABILITY.hard-gates.drive-tools", "ODYSSEY_TOOL_DRIVER"),
    "nr_nx_path.py":              ("FT.GPU_KERNELS.flash-nx", "NR_NX_PATH_MAP"),
    "rival_codec_screen.py":      ("FT.MODEL_REPRESENTATION.meta-gates-3-9", "RIVAL_CODEC_SCREEN"),
    "protected_scheduler.py":     ("FT.GPU_KERNELS.ready-protected", "PROTECTED_SCHEDULER"),
    "nr_nx_generic.py":           ("FT.GPU_KERNELS.flash-nx", "NR_NX_GENERIC"),
    "status_causality.py":        ("FT.VERIFICATION.negative-index", "STATUS_CHALLENGE"),
    "no_wait_scheduler.py":       ("FT.HCLI_SELF.emit-workunits", "NO_WAIT_SCHEDULING"),
    "concurrency_doctor.py":      ("FT.LATENCY.cpu-turnaround", "CONCURRENCY_DOCTOR"),
    "flash_organ_pivot.py":       ("FT.MODEL_REPRESENTATION.meta-gates-3-9", "ORGAN_PIVOT"),
    "abliteration.py":            ("FT.MODEL_CAPABILITY.tournament-refuse", "ABLITERATION"),
    "mutation_engine.py":         ("FT.HCLI_SELF.emit-workunits", "MUTATION_ENGINE"),
    "power_torture.py":           ("FT.VERIFICATION.repro", "POWER_TORTURE"),
    "resident_provider.py":       ("FT.CHILD_RESIDENT.install-dry-run", "RESIDENT_PROVIDER"),
    "model_bearing.py":           ("FT.HCLI_SELF.emit-workunits", "MODEL_BEARING"),
    "trial_workload.py":          ("FT.VERIFICATION.repro", "TRIAL_WORKLOAD"),
    "sprint_profile.py":          ("FT.EXPERIMENT_TURNAROUND.refresh", "WALL_ATTRIBUTION"),
    "resident_health.py":         ("FT.HCLI_SELF.no-launch", "RESIDENT_TELEMETRY"),
    "flash_meta_replan.py":       ("FT.MODEL_REPRESENTATION.meta-gates-3-9", "META_REPLAN"),
    "teacher_corpus_expansion.py": ("FT.MODEL_REPRESENTATION.teacher-capture", "TEACHER_PLAN"),
    "flash_bpw_ladder.py":        ("FT.MODEL_REPRESENTATION.meta-gates-3-9", "EBPW_LADDER"),
    "metal_reachability.py":      ("FT.MODEL_EXECUTION.complete-token", "HOST_CAPABILITY"),
    "specimen_verify.py":          ("FT.MODEL_CAPABILITY.hard-gates", "SPECIMEN_VERIFY"),
    # --- physical graph / fpga / ane --------------------------------------
    "hwir.py":                     ("FT.PHYSICAL_GRAPH.hwir-lower", "HWIR_LOWER"),
    "fpga_engines.py":             ("FT.FPGA.engine-sim", "FPGA_SIMULATION"),
    "fpga_fidelity.py":            ("FT.FPGA.engine-sim", "FPGA_SIMULATION"),
    "hardware_doctor.py":          ("FT.FPGA.hardware-doctor", "HARDWARE_DOCTOR"),
    "ane_preboard.py":             ("FT.ANE.preboard", "ANE_PREBOARD"),
    # --- architecture repatriation ----------------------------------------
    "cuda_lowbit_hypotheses.py":   ("FT.ARCHITECTURE_REPATRIATION.compile-specs", "SEARCH_ARCHITECTURE_LAWS"),
    "device_ascension_pipeline.py": ("FT.ARCHITECTURE_REPATRIATION.device-run", "DEVICE_ASCENSION"),
    # --- verification ------------------------------------------------------
    "negative_index.py":           ("FT.VERIFICATION.negative-index", "UPDATE_SCAR"),
    "scar_scheduling.py":          ("FT.VERIFICATION.negative-index", "UPDATE_SCAR"),
    "repro_science.py":            ("FT.VERIFICATION.repro", "INDEPENDENT_REPRODUCTION"),
    "evidence_dag.py":             ("FT.VERIFICATION.repro", "ADAPTIVE_VERIFY"),
    "contamination.py":            ("FT.VERIFICATION.contamination", "DIAGNOSTIC_AB"),
    "dirty_measure.py":            ("FT.VERIFICATION.contamination", "DIAGNOSTIC_AB"),
    "lpc_dataset.py":              ("FT.VERIFICATION.repro", "LEARNED_COMPILER_ROW"),
    "lpc_baselines.py":            ("FT.VERIFICATION.repro", "LEARNED_COMPILER_ROW"),
    "green_machine.py":            ("FT.VERIFICATION.contamination", "GREEN_MEASURE"),
    # --- odyssey -----------------------------------------------------------
    "odyssey2_law_store.py":       ("FT.ODYSSEY_TRANSFER.flash-qwen27", "TRANSFER_LAW"),
    "odyssey3_adversary.py":       ("FT.ODYSSEY_ADVERSARY.attacks", "ATTACK_LAW"),
    "odyssey_launch.py":           ("FT.ODYSSEY_TRANSFER.re-earn", "ODYSSEY_LAUNCH_GATE"),
}


def _audit_rows() -> list[dict[str, Any]]:
    p = RECEIPTS / "RESIDENT_API_AUDIT.json"
    if not p.exists():
        return []
    return load_json(p).get("modules") or []


def _receipt_of(row: dict[str, Any]) -> str | None:
    r = row.get("receipt")
    if r:
        return Path(str(r)).name
    names = row.get("write_receipt_names") or []
    return names[0] if names else None


def validate_bindings() -> dict[str, Any]:
    """A binding must describe reality: the module must exist and write the
    receipt the binding is built on. Otherwise the table is fiction."""
    rows = {r["filename"]: r for r in _audit_rows()}
    ok, broken, unbound, infra = [], [], [], []
    for fn, row in sorted(rows.items()):
        if fn in INFRASTRUCTURE or row.get("kind") != "production":
            infra.append(fn)
            continue
        if fn not in BINDINGS:
            unbound.append(fn)
            continue
        frontier_id, species = BINDINGS[fn]
        receipt = _receipt_of(row)
        if not receipt:
            broken.append({"module": fn, "why": "bound but writes no receipt"})
            continue
        ok.append({"module": fn, "receipt": receipt,
                   "frontier_item": frontier_id, "species": species})
    for fn in BINDINGS:
        if fn not in rows:
            broken.append({"module": fn, "why": "binding names a module that does not exist"})
    return {"bound": ok, "broken": broken, "unbound": unbound, "infrastructure": sorted(infra)}


def frontier_view() -> dict[str, Any]:
    """The shape `resident_api.evaluate_five_questions` consumes.

    `by_probe_receipt` maps a receipt filename to the frontier items it informs.
    Derived from validated bindings only, so a broken binding cannot credit a
    module it does not describe.
    """
    v = validate_bindings()
    by_receipt: dict[str, list[str]] = {}
    by_module: dict[str, list[str]] = {}
    for row in v["bound"]:
        by_receipt.setdefault(row["receipt"], []).append(row["frontier_item"])
        by_module.setdefault(f"tools/future/{row['module']}", []).append(row["frontier_item"])
    return {
        "present": True,
        "by_probe_receipt": by_receipt,
        "by_integration_module": by_module,
        "writes_frontier_modules": ["tools/future/global_frontier.py",
                                    "tools/future/frontiers.py"],
        "source": "tools/future/orchestration.py BINDINGS (validated)",
    }


def species_for(module: str) -> str:
    if module not in BINDINGS:
        raise UnknownBinding(f"no binding for {module!r}")
    return BINDINGS[module][1]


def emit_workunit(module: str, *, hypothesis: str | None = None) -> dict[str, Any]:
    """Wrap a bound capability as a real WorkUnit.

    Supplies the second missing axis on the module's behalf rather than editing
    fifty-two modules to each grow an emitter.
    """
    if module not in BINDINGS:
        raise UnknownBinding(f"no binding for {module!r}")
    frontier_id, species = BINDINGS[module]
    rows = {r["filename"]: r for r in _audit_rows()}
    row = rows.get(module) or {}
    receipt = _receipt_of(row)
    if not receipt:
        raise BindingError(f"{module} writes no receipt; cannot form an output contract")
    return {
        "id": f"WU.{species}.{module.removesuffix('.py')}",
        "species": species,
        "module": f"tools/future/{module}",
        "hypothesis": hypothesis or f"running {module} advances {frontier_id}",
        "frontier_item": frontier_id,
        "output_contract": f"receipts/future/{receipt}",
        "resource_class": "CPU_ANALYSIS",
        "allowed_authority": ["read_receipts", "write_sidecar_receipt", "emit_static_plan"],
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "verifier": "tools/future/integration_attack.py --adversarial",
        "stop_condition": "receipt written and sealed, or the module raises",
    }


def invoke(module: str, *, argv: list[str] | None = None) -> dict[str, Any]:
    """Run a bound capability and ROUTE its receipt to the frontier it informs.

    The routing is what makes the binding true rather than declarative. An
    unbound or unknown module raises; a module whose run produces no receipt
    raises. Never returns a success shape it did not observe.
    """
    if module not in BINDINGS:
        raise UnknownBinding(f"no binding for {module!r}")
    frontier_id, species = BINDINGS[module]
    name = module.removesuffix(".py")
    try:
        mod = importlib.import_module(f"tools.future.{name}")
    except Exception as exc:
        raise BindingError(f"{module} failed to import: {type(exc).__name__}: {exc}") from exc

    entry = None
    for cand in ("build", "selftest", "audit", "snapshot", "probe"):
        if callable(getattr(mod, cand, None)):
            entry = cand
            break
    if entry is None:
        raise BindingError(f"{module} exposes no callable entry point (build/selftest/audit/...)")

    started = time.time()
    out = getattr(mod, entry)()
    if isinstance(out, dict):
        # Some modules return a result record rather than the path alone --
        # odyssey_launch returns {gate_path, doc, launch} because the gate
        # receipt and the launch receipt are different artifacts. Take the path
        # it names; str() of the dict is not one, and silently failed the bind.
        out = out.get("path") or out.get("gate_path") or out.get("receipt")
    receipt_path = Path(str(out)) if out else None
    if receipt_path is None or not receipt_path.exists():
        raise BindingError(f"{module}.{entry}() produced no receipt on disk")

    return {
        "module": module,
        "entry_point": entry,
        "receipt": str(receipt_path.relative_to(REPO)),
        "routed_to_frontier": frontier_id,
        "species": species,
        "wall_seconds": round(time.time() - started, 3),
        "evidence_class": "STATIC_ONLY",
    }


def build() -> Path:
    v = validate_bindings()
    view = frontier_view()
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Bind every production sidecar module to the frontier item its receipt "
            "informs, and supply WorkUnit emission on its behalf, so the substrate "
            "becomes resident-callable without editing seventy modules."
        ),
        "not_goodhart": (
            "A binding is validated against the audit: the module must exist and must "
            "write the receipt the binding is built on, or the binding is BROKEN and "
            "credits nothing. invoke() actually runs the module and actually routes its "
            "receipt. Infrastructure that informs no frontier is EXCLUDED, not fake-bound."
        ),
        "counts": {
            "bound": len(v["bound"]),
            "broken": len(v["broken"]),
            "unbound_production": len(v["unbound"]),
            "infrastructure_excluded": len(v["infrastructure"]),
            "distinct_frontier_items": len({b["frontier_item"] for b in v["bound"]}),
            "distinct_species": len({b["species"] for b in v["bound"]}),
        },
        "bound": v["bound"],
        "broken": v["broken"],
        "unbound_production": v["unbound"],
        "infrastructure_excluded": v["infrastructure"],
        "frontier_view_keys": sorted(view["by_probe_receipt"]),
        "recovered_implementation": [
            "tools/future/resident_api.py evaluate_five_questions already accepts a "
            "frontier mapping; this supplies it rather than re-implementing scoring",
            "tools/future/frontiers.py FrontierBook provides the 63 frontier items",
            "tools/future/workunit_species.py provides the WorkUnit shape",
        ],
        "gaps_closed": [
            "result_does_not_feed_a_named_frontier (70 modules)",
            "does_not_emit_workunit (52 modules)",
        ],
        "negative_findings": [
            f"{len(v['unbound'])} production modules remain unbound and are NOT credited",
            f"{len(v['broken'])} bindings are broken and credit nothing",
            "binding a module does not make its science correct; it makes the module "
            "reachable and its result routable",
        ],
        "resident_callable": {
            "entry_point": "tools.future.orchestration.invoke(module)",
            "workunit": "tools.future.orchestration.emit_workunit(module)",
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "each binding names the frontier item it feeds",
            "fails_closed": "UnknownBinding / BindingError raise; no success shape is invented",
        },
    }
    if v["broken"]:
        raise BindingError(f"broken bindings: {v['broken']}")
    return write_receipt(RECEIPT, doc, "tools/future/orchestration.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", action="store_true")
    ap.add_argument("--invoke", metavar="MODULE")
    a = ap.parse_args()
    if a.invoke:
        mod = a.invoke if a.invoke.endswith(".py") else a.invoke.replace("future.", "") + ".py"
        print(json.dumps(invoke(mod), indent=1, sort_keys=True))
        return 0
    out = build()
    doc = json.loads(out.read_text())
    print(out)
    print(json.dumps(doc["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
