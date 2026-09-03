"""Acceptance runner for Odyssey I/II/III, HMF device-visible trust, and Fusion.

Calls each gate's own implementing symbol (an import is not a call). Writes
``receipts/acceptance/<GATE>.json`` recording the real run. Never writes
ModelLake, ``hcli/``, ``tools/roadmap``, ``tools/audit``, ``tools/theia``, or
``receipts/future``. Never mutates live Odyssey state (``mutate=False``).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from tools.roadmap import lineage

REPO = Path(__file__).resolve().parents[3]
RECEIPTS_DIR = REPO / "receipts" / "acceptance"
# The canonical roadmap is external and can vanish; tools.roadmap.lineage
# falls back to the digest-verified in-repo copy rather than a placeholder.
ROADMAP = lineage.roadmap_path()
PRIMARY = Path(os.environ.get("HAWKING_PRIMARY", "/Users/scammermike/Downloads/hawking"))
LAKE = Path("/Volumes/corpdrive/hawking-modellake")
SPECIMENS = LAKE / "specimens"
PRIMARY_STATE = PRIMARY / "workspace/campaign/odyssey/ODYSSEY_STATE.json"
PRIMARY_RULEBASE = PRIMARY / "workspace/campaign/odyssey/GRAVITY_RULEBASE.json"
PRIMARY_NEGATIVE = PRIMARY / "workspace/campaign/odyssey/NEGATIVE_SCIENCE.json"
PRIMARY_TRANSFER = PRIMARY / "workspace/campaign/odyssey/TRANSFER_MATRIX.json"

SCHEMA = "hawking.acceptance.gate.v1"
MANIFEST_SCHEMA = "hawking.acceptance.manifest.v1"

# Odyssey II OUTPUT (H-ROADMAP §6): measurable reduction in experiments/time/cost.
# The only cold-vs-transfer measurement on disk records evaluations_avoided = -8
# (COLD 2 evals, TRANSFER 10). A reduction must be strictly positive.
TRANSFER_REDUCTION_FIELD = "delta.evaluations_avoided"
TRANSFER_REDUCTION_OP = ">"
TRANSFER_REDUCTION_THRESHOLD = 0

GATES: tuple[str, ...] = (
    "ODYSSEY_I_DISCOVERY",
    "ODYSSEY_II_TRANSFER",
    "ODYSSEY_III_ADVERSARIAL_META_SCIENCE",
    "HMF_DEVICE_VISIBLE_TRUST",
    "FUSION_FIRST_HETEROGENEOUS_EXECUTABLE",
)

# Two architecture-diverse sealed specimens, headers only (no weight load).
CENSUS_SPECIMENS: tuple[tuple[str, str], ...] = (
    (
        "qwen3_0.6b",
        "Qwen--Qwen3-0.6B@c1899de289a0",
    ),
    (
        "falcon_h1_7b",
        "tiiuae--Falcon-H1-7B-Instruct@41e72f27effb",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def seal(doc: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    doc["seal_sha256"] = hashlib.sha256(blob).hexdigest()
    return doc


def roadmap_quote(start: int, end: int) -> str:
    if not ROADMAP.is_file():
        return f"(H-ROADMAP.md missing; catalog span {start}-{end})"
    lines = ROADMAP.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[start - 1 : end])


def git_show_json(rel: str) -> Any:
    blob = subprocess.check_output(["git", "-C", str(REPO), "show", f"HEAD:{rel}"])
    return json.loads(blob)


def git_ls_count(prefix: str) -> int:
    blob = subprocess.check_output(
        ["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", "HEAD", prefix]
    )
    return len([ln for ln in blob.decode().splitlines() if ln.strip()])


def ensure_hcli_importable() -> dict[str, Any]:
    """This sparse cone does not materialise ``hcli/``. Import from the primary
    checkout as a fallback so ``load_qualification_queue`` can be called, not
    reimplemented. Never writes there."""
    try:
        import hcli  # noqa: F401

        return {"ok": True, "source": "sys.path", "path": getattr(hcli, "__file__", None)}
    except ImportError:
        pass
    hcli_dir = PRIMARY / "hcli"
    if hcli_dir.is_dir() and str(PRIMARY) not in sys.path:
        sys.path.append(str(PRIMARY))
    try:
        import hcli  # noqa: F401

        return {"ok": True, "source": "primary_checkout", "path": str(hcli_dir)}
    except ImportError as exc:
        return {"ok": False, "source": None, "error": repr(exc)}


def accelerator_sys_path() -> None:
    acc = str(REPO / "tools" / "accelerator")
    if acc not in sys.path:
        sys.path.insert(0, acc)


def no_network_hf_info(_repo: str) -> tuple[bool, None, str]:
    return False, None, "acceptance-run: no network"


def reduction_holds(evaluations_avoided: float | int | None) -> bool:
    """Odyssey II bar: transfer must reduce experiment count (strictly > 0)."""
    if evaluations_avoided is None:
        return False
    try:
        measured = float(evaluations_avoided)
    except (TypeError, ValueError):
        return False
    return measured > float(TRANSFER_REDUCTION_THRESHOLD)


def write_receipt(gate: str, doc: dict[str, Any]) -> Path:
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RECEIPTS_DIR / f"{gate}.json"
    sealed = seal(doc)
    path.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _base(
    gate: str,
    *,
    start: int,
    end: int,
    invoked: dict[str, Any],
    command: str,
    evidence_tier: str,
    verdict: str,
    run: dict[str, Any],
    blocker: dict[str, Any] | None,
    numeric_comparisons: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "gate": gate,
        "verdict": verdict,
        "evidence_tier": evidence_tier,
        "criterion": {
            "file": str(ROADMAP),
            "start_line": start,
            "end_line": end,
            "quote": roadmap_quote(start, end),
        },
        "invoked_symbol": invoked,
        "command": command,
        "run": run,
        "blocker": blocker,
        "numeric_comparisons": numeric_comparisons or [],
        "criterion_altered": False,
        "generated_at": utc_now(),
        "generated_by": "tools.acceptance.odyssey.run",
        "repo": str(REPO),
    }
    if extra:
        doc.update(extra)
    return doc


# ---------------------------------------------------------------------------
# ODYSSEY_I_DISCOVERY
# ---------------------------------------------------------------------------


def _lake_inventory() -> dict[str, Any]:
    if not SPECIMENS.is_dir():
        return {
            "mounted": False,
            "n_specimens": 0,
            "model_types": {},
            "blocker": "MODELLAKE_MOUNTED",
        }
    types: dict[str, int] = {}
    n = 0
    n_config = 0
    for d in sorted(SPECIMENS.iterdir()):
        if not d.is_dir():
            continue
        n += 1
        cfg = d / "config.json"
        if not cfg.is_file():
            types["NO_CONFIG"] = types.get("NO_CONFIG", 0) + 1
            continue
        try:
            j = json.loads(cfg.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            types["UNREADABLE"] = types.get("UNREADABLE", 0) + 1
            continue
        n_config += 1
        mt = j.get("model_type") or j.get("architectures") or "UNKNOWN"
        if isinstance(mt, list):
            mt = ",".join(str(x) for x in mt)
        key = str(mt)
        types[key] = types.get(key, 0) + 1
    return {
        "mounted": True,
        "n_specimens": n,
        "n_with_config": n_config,
        "n_model_types": len(types),
        "model_types": dict(sorted(types.items())),
    }


def _census_pair() -> dict[str, Any]:
    from tools.odyssey_census import census

    out: dict[str, Any] = {}
    for label, slug in CENSUS_SPECIMENS:
        path = SPECIMENS / slug
        t0 = time.time()
        r = census(str(path))
        out[label] = {
            "path": str(path),
            "exists": path.is_dir(),
            "arch": r.get("arch"),
            "model_type": r.get("model_type"),
            "total_params": r.get("total_params"),
            "total_bytes": r.get("total_bytes"),
            "is_moe": r.get("is_moe"),
            "stored_bpw": r.get("stored_bpw"),
            "organs_params": r.get("organs_params"),
            "wall_s": round(time.time() - t0, 3),
        }
    return out


def run_odyssey_i() -> dict[str, Any]:
    """Call pick_acquire_candidate; independently census two sealed specimens."""
    invoked = {
        "module": "tools.odyssey_ctl",
        "symbol": "pick_acquire_candidate",
        "kind": "call",
        "file": "tools/odyssey_ctl.py",
    }
    command = (
        "python3 -c \"from tools.odyssey_ctl import pick_acquire_candidate; "
        "pick_acquire_candidate(state, mutate=False)\""
    )
    t0 = time.time()
    lake = _lake_inventory()
    school: dict[str, Any] = {
        "gravity_rulebase": None,
        "negative_science": None,
        "transfer_matrix": None,
        "odyssey_i_receipts_in_git": git_ls_count("receipts/odyssey-i"),
    }
    if PRIMARY_RULEBASE.is_file():
        rb = json.loads(PRIMARY_RULEBASE.read_text(encoding="utf-8"))
        school["gravity_rulebase"] = {
            "path": str(PRIMARY_RULEBASE),
            "n_rules": len(rb.get("rules") or []),
            "rule_ids": [r.get("id") for r in (rb.get("rules") or [])],
            "read_only": True,
        }
    if PRIMARY_NEGATIVE.is_file():
        ns = json.loads(PRIMARY_NEGATIVE.read_text(encoding="utf-8"))
        school["negative_science"] = {
            "path": str(PRIMARY_NEGATIVE),
            "n_entries": len(ns.get("entries") or []),
            "read_only": True,
        }
    if PRIMARY_TRANSFER.is_file():
        tm = json.loads(PRIMARY_TRANSFER.read_text(encoding="utf-8"))
        school["transfer_matrix"] = {
            "path": str(PRIMARY_TRANSFER),
            "n_rows": len(tm.get("rows") or []),
            "read_only": True,
        }

    picker: dict[str, Any]
    if not PRIMARY_STATE.is_file():
        picker = {
            "ok": False,
            "error": f"live Odyssey state not readable at {PRIMARY_STATE}",
        }
    else:
        import tools.odyssey_ctl as oc

        raw = json.loads(PRIMARY_STATE.read_text(encoding="utf-8"))
        st = copy.deepcopy(raw)
        cand, meta = oc.pick_acquire_candidate(
            st, mutate=False, hf_info_fn=no_network_hf_info
        )
        picker = {
            "ok": True,
            "state_file_read": str(PRIMARY_STATE),
            "mutate": False,
            "n_patients": len(st.get("patients") or []),
            "candidate_oxx": None if cand is None else cand.get("oxx"),
            "candidate_source": None
            if cand is None
            else (cand.get("canonical_source") or cand.get("source")),
            "candidate_state": None if cand is None else cand.get("state"),
            "n_skipped": len(meta.get("skipped") or []),
            "skipped_head": (meta.get("skipped") or [])[:12],
            "mutated": bool(meta.get("mutated")),
        }

    census = None
    census_error = None
    if lake.get("mounted"):
        try:
            census = _census_pair()
        except Exception as exc:  # noqa: BLE001 — recorded, not swallowed as success
            census_error = repr(exc)

    model_types_censused = sorted(
        {
            row.get("model_type")
            for row in (census or {}).values()
            if isinstance(row, dict) and row.get("model_type")
        }
    )
    picker_called = bool(picker.get("ok"))
    diverse = len(model_types_censused) >= 2
    lake_school = int(lake.get("n_specimens") or 0) >= 2 and int(
        lake.get("n_model_types") or 0
    ) >= 2
    # Promotion gate: a fresh process independently demonstrates the school
    # (architecture-diverse sealed specimens + census of model facts + the
    # wired picker actually selecting). Model-count alone is not the bar;
    # two different architectures independently censused is.
    accepted = bool(picker_called and diverse and lake_school and census_error is None)

    blocker = None
    verdict = "ACCEPTED"
    if not lake.get("mounted"):
        verdict = "BLOCKED"
        blocker = {
            "id": "MODELLAKE_MOUNTED",
            "missing_input": str(SPECIMENS),
            "wake_condition": "ModelLake specimens directory mounted at "
            f"{SPECIMENS} with architecture-diverse sealed specimens",
        }
    elif not accepted:
        verdict = "BLOCKED"
        blocker = {
            "id": "ODYSSEY_I_SCHOOL_NOT_DEMONSTRATED",
            "missing_input": {
                "picker_called": picker_called,
                "model_types_censused": model_types_censused,
                "census_error": census_error,
                "lake": {k: lake.get(k) for k in ("mounted", "n_specimens", "n_model_types")},
            },
            "wake_condition": (
                "fresh process censuses >=2 architectures from sealed ModelLake "
                "specimens AND pick_acquire_candidate returns on live state"
            ),
        }

    run = {
        "wall_s": round(time.time() - t0, 3),
        "pick_acquire_candidate": picker,
        "lake": lake,
        "census": census,
        "census_error": census_error,
        "model_types_censused": model_types_censused,
        "school_artifacts_read_only": school,
    }
    doc = _base(
        "ODYSSEY_I_DISCOVERY",
        start=531,
        end=553,
        invoked=invoked,
        command=command,
        evidence_tier="STATIC",
        verdict=verdict,
        run=run,
        blocker=blocker,
        extra={
            "section_6_output": (
                "first-principles model/device laws, scars, genomes, ModelLake school"
            ),
            "note": (
                "Census reads safetensors headers only (no weight materialisation). "
                "Live Odyssey state was deep-copied and pick_acquire_candidate ran "
                "with mutate=False; the live daemon's state file was not written."
            ),
        },
    )
    path = write_receipt("ODYSSEY_I_DISCOVERY", doc)
    return {"gate": "ODYSSEY_I_DISCOVERY", "verdict": verdict, "path": str(path), "doc": doc}


# ---------------------------------------------------------------------------
# ODYSSEY_II_TRANSFER
# ---------------------------------------------------------------------------


def _load_transfer_proven() -> dict[str, Any]:
    d = git_show_json("receipts/headless/ODYSSEY_TRANSFER_PROVEN.json")
    delta = d.get("delta") or {}
    avoided = delta.get("evaluations_avoided")
    return {
        "source": "git HEAD receipts/headless/ODYSSEY_TRANSFER_PROVEN.json",
        "generated_at": d.get("generated_at"),
        "generated_by": d.get("generated_by"),
        "specimen": d.get("specimen"),
        "evaluations_avoided": avoided,
        "cold_evaluations": delta.get("cold_evaluations"),
        "transfer_evaluations": delta.get("transfer_evaluations"),
        "same_landing": delta.get("same_landing"),
        "honest_note": d.get("honest_note"),
        "pass_field_in_receipt": d.get("pass"),
        "reduction_holds": reduction_holds(avoided),
    }


def run_odyssey_ii() -> dict[str, Any]:
    """Call load_qualification_queue; compare transfer savings against > 0."""
    invoked = {
        "module": "tools.future.qualification_pipeline",
        "symbol": "load_qualification_queue",
        "kind": "call",
        "file": "tools/future/qualification_pipeline.py",
    }
    command = (
        "python3 -c \"from tools.future.qualification_pipeline import "
        "load_qualification_queue; load_qualification_queue(path)\""
    )
    t0 = time.time()
    hcli = ensure_hcli_importable()
    queue_run: dict[str, Any]
    if not hcli.get("ok"):
        queue_run = {
            "ok": False,
            "error": "hcli not importable in this sparse cone and primary fallback failed",
            "hcli": hcli,
        }
    else:
        from tools.future.qualification_pipeline import load_qualification_queue

        # Committed queue, not the primary checkout's possibly-dirty copy.
        rel = "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"
        blob = subprocess.check_output(["git", "-C", str(REPO), "show", f"HEAD:{rel}"])
        tmp = RECEIPTS_DIR / "_queue_snapshot.json"
        RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(blob)
        try:
            q = load_qualification_queue(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        cands = q.get("candidates") or []
        queue_run = {
            "ok": True,
            "hcli": hcli,
            "loaded_from": q.get("_loaded_from"),
            "schema": q.get("schema"),
            "n_candidates": len(cands),
            "candidate_ids_head": [c.get("candidate_id") for c in cands[:8]],
            "statuses_head": [c.get("status") for c in cands[:8]],
        }

    proven = _load_transfer_proven()
    measured = proven.get("evaluations_avoided")
    comparison = {
        "field": TRANSFER_REDUCTION_FIELD,
        "measured": measured,
        "op": TRANSFER_REDUCTION_OP,
        "threshold": TRANSFER_REDUCTION_THRESHOLD,
        "passed": reduction_holds(measured),
        "source": proven.get("source"),
    }
    # The wired call succeeding is not the bar. The bar is a measured reduction.
    accepted = bool(queue_run.get("ok")) and bool(comparison["passed"])
    verdict = "ACCEPTED" if accepted else "BLOCKED"
    blocker = None
    if not accepted:
        blocker = {
            "id": "TRANSFER_REDUCTION_NOT_POSITIVE",
            "missing_input": (
                f"{TRANSFER_REDUCTION_FIELD} measured {measured!r}; "
                f"required {TRANSFER_REDUCTION_OP} {TRANSFER_REDUCTION_THRESHOLD}. "
                "COLD arm used 2 evaluations, TRANSFER arm used 10 "
                "(evaluations_avoided=-8). Odyssey II sidecar records "
                "actual_saved_experiments=null / UNMEASURED."
            ),
            "wake_condition": (
                "a cold-vs-transfer measurement on an unseen specimen with "
                "evaluations_avoided > 0 (or an equivalent measured reduction in "
                "time/cost), not a similarity score and not UNMEASURED"
            ),
        }
    run = {
        "wall_s": round(time.time() - t0, 3),
        "load_qualification_queue": queue_run,
        "transfer_proven": proven,
    }
    doc = _base(
        "ODYSSEY_II_TRANSFER",
        start=554,
        end=573,
        invoked=invoked,
        command=command,
        evidence_tier="STATIC",
        verdict=verdict,
        run=run,
        blocker=blocker,
        numeric_comparisons=[comparison],
        extra={
            "section_6_output": (
                "measurable reduction in experiments/time/cost on unseen models and devices"
            ),
            "note": (
                "load_qualification_queue was called on the git-HEAD queue snapshot. "
                "That call proves the planner is reachable. The acceptance bar is the "
                "numeric reduction, compared here against the cold-vs-transfer receipt "
                "rather than asserted. pass=true on that receipt is not this gate: its "
                "own honest_note reports the COLD arm winning the pre-registered target."
            ),
        },
    )
    path = write_receipt("ODYSSEY_II_TRANSFER", doc)
    return {"gate": "ODYSSEY_II_TRANSFER", "verdict": verdict, "path": str(path), "doc": doc}


# ---------------------------------------------------------------------------
# ODYSSEY_III_ADVERSARIAL_META_SCIENCE
# ---------------------------------------------------------------------------


def run_odyssey_iii() -> dict[str, Any]:
    """Call scars(); run the adversary closed loop; refuse synthetic as accepted."""
    invoked = {
        "module": "tools.future.autonomy_scars",
        "symbol": "scars",
        "kind": "call",
        "file": "tools/future/autonomy_scars.py",
    }
    command = (
        "python3 -c \"from tools.future.autonomy_scars import scars; scars()\" "
        "and odyssey3_adversary.selftest() (no --build, no receipt write)"
    )
    t0 = time.time()
    from tools.future.autonomy_scars import missing_regression_tests, scars
    from tools.future import odyssey3_adversary as o3

    rows = scars()
    missing = missing_regression_tests()
    st = o3.selftest()
    cpu = st.get("cpu_observation") or {}
    synthetic = bool((st.get("synthetic_result") or {}).get("synthetic")) or bool(
        cpu.get("synthetic")
    )
    physical_arm = cpu.get("physical_arm")
    # A synthetic REFUTED verdict moving a fixture law is the loop working, not
    # an independent refutation of a Hawking campaign law. Traps firing on a
    # synthetic vector are FUNCTIONAL_SIM of measurement-trap machinery.
    independent = (not synthetic) and physical_arm not in (None, "not_run")
    accepted = bool(rows) and independent and not missing
    verdict = "ACCEPTED" if accepted else "BLOCKED"
    blocker = None
    if not accepted:
        blocker = {
            "id": "NO_INDEPENDENT_LAW_REFUTATION",
            "missing_input": (
                "odyssey3_adversary.selftest applies a synthetic REFUTED verdict "
                f"(synthetic={synthetic}); physical_arm={physical_arm!r}; "
                f"evidence_class={st.get('evidence_class')!r}. scars() returned "
                f"{len(rows)} orchestrator scars with {len(missing)} missing "
                "regression tests — a registry, not a fresh law attack."
            ),
            "wake_condition": (
                "a non-synthetic result from a physical or specimen-backed arm "
                "that moves a named campaign law's scope DOWN, with evidence_class "
                "other than STATIC_ONLY/synthetic"
            ),
        }
    run = {
        "wall_s": round(time.time() - t0, 3),
        "scars": {
            "n_scars": len(rows),
            "ids": [r.get("id") for r in rows],
            "n_missing_regression_tests": len(missing),
            "missing": missing,
            "sample": [
                {k: r.get(k) for k in ("id", "law", "claim_refuted")} for r in rows[:4]
            ],
        },
        "adversary_selftest": {
            "law_id": st.get("law_id"),
            "selected_attack_id": st.get("selected_attack_id"),
            "selected_family": st.get("selected_family"),
            "scope_before": st.get("scope_before"),
            "scope_after": st.get("scope_after"),
            "moved_down": st.get("moved_down"),
            "synthetic_result": st.get("synthetic_result"),
            "evidence_class": st.get("evidence_class"),
            "cpu_observation": {
                "verdict": cpu.get("verdict"),
                "traps_fired": cpu.get("traps_fired"),
                "synthetic": cpu.get("synthetic"),
                "physical_arm": cpu.get("physical_arm"),
                "evidence_class": cpu.get("evidence_class"),
                "reason": cpu.get("reason"),
                "controls": cpu.get("controls"),
            },
        },
    }
    doc = _base(
        "ODYSSEY_III_ADVERSARIAL_META_SCIENCE",
        start=663,
        end=683,
        invoked=invoked,
        command=command,
        evidence_tier="FUNCTIONAL_SIM",
        verdict=verdict,
        run=run,
        blocker=blocker,
        extra={
            "section_6_output": (
                "broken laws, corrected priors, stronger verifiers, causal robustness"
            ),
            "note": (
                "CPU measurement traps (scale_invariance, skip_counted_as_pass) fired "
                "in this process on a synthetic vector explicitly labelled "
                "'not a model organ'. That demonstrates trap machinery, not an "
                "independent law falsification. selftest() does not write receipts."
            ),
        },
    )
    path = write_receipt("ODYSSEY_III_ADVERSARIAL_META_SCIENCE", doc)
    return {
        "gate": "ODYSSEY_III_ADVERSARIAL_META_SCIENCE",
        "verdict": verdict,
        "path": str(path),
        "doc": doc,
    }


# ---------------------------------------------------------------------------
# HMF_DEVICE_VISIBLE_TRUST
# ---------------------------------------------------------------------------


def run_hmf() -> dict[str, Any]:
    """Probe HMF_PRESENT; exercise the overlay; refuse UMA as device-visible HMF."""
    invoked = {
        "module": "tools.future.hmf_objects",
        "symbol": "is_coherent",
        "kind": "call",
        "file": "tools/future/hmf_objects.py",
        "also_called": [
            "tools.roadmap.hardware.probe('HMF_PRESENT')",
            "hmf_objects.establish_clean",
            "hmf_objects.cross_kernel_boundary",
            "hmf_objects.device_digest",
        ],
    }
    command = (
        "python3 -c \"from tools.roadmap.hardware import probe; probe('HMF_PRESENT')\" "
        "and hmf_objects is_coherent/cross_kernel_boundary on an in-memory UMA fixture"
    )
    t0 = time.time()
    from tools.roadmap.hardware import probe

    hw = probe("HMF_PRESENT")
    from tools.future import hmf_objects as ho

    obj = ho.ManagedObject(identity=ho.HgvasRef("acc6.trust.0"))
    ho.establish_clean(
        obj,
        location=ho.Location(ho.MemoryTier.UMA, "APPLE_DOMAIN_0"),
        visibility={"APPLE_DOMAIN_0"},
        evidence="acceptance in-memory fixture on Apple UMA; not an HMF device",
        digest="ab" * 16,
        payload=b"payload-bytes",
    )
    coh = ho.is_coherent(obj, on_device="APPLE_DOMAIN_0")
    moved = ho.cross_kernel_boundary(obj, synchronized=False)
    coh2 = ho.is_coherent(moved)
    dig = ho.device_digest(obj, provider=None)
    bool_refused = False
    try:
        bool(coh)
    except ho.CoherenceBooleanError:
        bool_refused = True

    overlay = {
        "clean_coherence": getattr(coh, "value", str(coh)),
        "after_unsynced_kernel_boundary": getattr(coh2, "value", str(coh2)),
        "device_digest_without_provider": str(dig),
        "boolean_collapse_refused": bool_refused,
        "evidence_tier": "FUNCTIONAL_SIM",
        "note": (
            "Apple UMA overlay. PRESENT on UMA is not HMF device-visible trust. "
            "Roadmap 16.2: Hawking targets coherence ladder 2-5; never falsely claim 6."
        ),
    }
    present = bool(hw.get("present"))
    # Device-visible trust cannot be accepted without the device. The overlay
    # working on UMA is explicitly not the bar.
    accepted = present and overlay["after_unsynced_kernel_boundary"] == "UNKNOWN"
    verdict = "ACCEPTED" if accepted else "BLOCKED"
    blocker = None
    if not present:
        blocker = {
            "id": "HMF_PRESENT",
            "missing_input": (
                "HMF/HGVAS/CXL memory appliance enumerated on this host. "
                f"probe evidence: {hw.get('evidence')}"
            ),
            "wake_condition": "HMF_PRESENT",
            "wake_condition_detail": hw,
        }
    run = {
        "wall_s": round(time.time() - t0, 3),
        "hardware_probe": hw,
        "overlay": overlay,
    }
    doc = _base(
        "HMF_DEVICE_VISIBLE_TRUST",
        start=2801,
        end=2866,
        invoked=invoked,
        command=command,
        evidence_tier="STATIC",
        verdict=verdict,
        run=run,
        blocker=blocker,
        extra={
            "trust_law": (
                "PRESENT != VALID != TRUSTED. UNKNOWN != LOST. "
                "READBACK CORRECTNESS DOES NOT PROVE DEVICE-COMPUTE VISIBILITY."
            ),
        },
    )
    path = write_receipt("HMF_DEVICE_VISIBLE_TRUST", doc)
    return {
        "gate": "HMF_DEVICE_VISIBLE_TRUST",
        "verdict": verdict,
        "path": str(path),
        "doc": doc,
    }


# ---------------------------------------------------------------------------
# FUSION_FIRST_HETEROGENEOUS_EXECUTABLE
# ---------------------------------------------------------------------------


def run_fusion() -> dict[str, Any]:
    """A heterogeneous executable needs two real domains. Probe and refuse knobs."""
    invoked = {
        "module": "tools.accelerator.fusion_planner",
        "symbol": "topology_apple_alone",
        "kind": "call",
        "file": "tools/accelerator/fusion_planner.py",
        "also_called": [
            "tools.future.fusion_sim.hawking_topology",
            "tools.future.fusion_sim.simulate_default",
            "tools.roadmap.hardware.probe_all",
        ],
    }
    command = (
        "python3 -c \"from fusion_planner import topology_apple_alone; "
        "topology_apple_alone()\" and fusion_sim.hawking_topology()/simulate_default()"
    )
    t0 = time.time()
    from tools.roadmap.hardware import probe_all

    probes = probe_all()
    accelerator_sys_path()
    import fusion_planner as fp  # type: ignore  # noqa: E402

    topo = fp.topology_apple_alone()
    apple_alone = {
        "domains": list(topo.domains),
        "n_domains": len(topo.domains),
        "physical": {k: bool(v) for k, v in topo.domains.items()},
    }
    from tools.future.fusion_sim import hawking_topology, simulate_default

    ftopo = hawking_topology()
    nodes = {
        n.id: {"present": bool(n.present), "missing": n.missing_dependency}
        for n in ftopo.nodes
    }
    sim = simulate_default()
    present_nodes = [k for k, v in nodes.items() if v.get("present")]
    second_domain = (
        bool(probes.get("U50_PRESENT", {}).get("present"))
        or bool(probes.get("EGPU_PRESENT", {}).get("present"))
        or bool(probes.get("DGX_PRESENT", {}).get("present"))
    )
    hmf_present = bool(probes.get("HMF_PRESENT", {}).get("present"))
    timing_decidable = bool(sim.get("timing_decidable"))
    # An executable on two real domains, not a COST_MODEL over knobs.
    accepted = (
        hmf_present
        and second_domain
        and len(present_nodes) >= 2
        and timing_decidable
        and sim.get("speedup") is not None
    )
    verdict = "ACCEPTED" if accepted else "BLOCKED"
    blocker = None
    if not accepted:
        wakes = []
        if not hmf_present:
            wakes.append("HMF_PRESENT")
        if not second_domain:
            wakes.extend(["U50_PRESENT", "EGPU_PRESENT", "DGX_PRESENT"])
        blocker = {
            "id": "NO_SECOND_PHYSICAL_DOMAIN",
            "missing_input": (
                "HMF_DEVICE_VISIBLE_TRUST is blocked on HMF_PRESENT; this host has "
                f"present fusion nodes {present_nodes} (need >=2). "
                f"U50={probes['U50_PRESENT']['present']}, "
                f"EGPU={probes['EGPU_PRESENT']['present']}, "
                f"DGX={probes['DGX_PRESENT']['present']}. "
                f"simulate_default timing_decidable={sim.get('timing_decidable')} "
                f"speedup={sim.get('speedup')!r}. fusion_planner apple-alone has "
                f"{apple_alone['n_domains']} domain. A COST_MODEL/SIMULATED hop "
                "is not a heterogeneous executable."
            ),
            "wake_condition": "HMF_PRESENT and one of U50_PRESENT | EGPU_PRESENT | DGX_PRESENT",
            "wake_conditions": wakes,
            "dependency": "HMF_DEVICE_VISIBLE_TRUST",
        }
    run = {
        "wall_s": round(time.time() - t0, 3),
        "hardware_probes": {
            k: {"present": v.get("present"), "evidence": v.get("evidence")}
            for k, v in probes.items()
        },
        "fusion_planner_apple_alone": apple_alone,
        "fusion_sim_nodes": nodes,
        "fusion_sim_present_nodes": present_nodes,
        "simulate_default": {
            "timing_decidable": sim.get("timing_decidable"),
            "speedup": sim.get("speedup"),
        },
    }
    doc = _base(
        "FUSION_FIRST_HETEROGENEOUS_EXECUTABLE",
        start=2867,
        end=2971,
        invoked=invoked,
        command=command,
        evidence_tier="STATIC",
        verdict=verdict,
        run=run,
        blocker=blocker,
        extra={
            "fusion_law": (
                "The target is not to make alien memory domains literally become "
                "Apple UMA. A first heterogeneous executable requires a second "
                "qualified domain (FPGA/eGPU/DGX) plus HMF/HGVAS object trust."
            ),
        },
    )
    path = write_receipt("FUSION_FIRST_HETEROGENEOUS_EXECUTABLE", doc)
    return {
        "gate": "FUSION_FIRST_HETEROGENEOUS_EXECUTABLE",
        "verdict": verdict,
        "path": str(path),
        "doc": doc,
    }


RUNNERS: dict[str, Callable[[], dict[str, Any]]] = {
    "ODYSSEY_I_DISCOVERY": run_odyssey_i,
    "ODYSSEY_II_TRANSFER": run_odyssey_ii,
    "ODYSSEY_III_ADVERSARIAL_META_SCIENCE": run_odyssey_iii,
    "HMF_DEVICE_VISIBLE_TRUST": run_hmf,
    "FUSION_FIRST_HETEROGENEOUS_EXECUTABLE": run_fusion,
}


def run_gate(gate: str) -> dict[str, Any]:
    fn = RUNNERS.get(gate)
    if fn is None:
        raise KeyError(f"unknown gate {gate!r}; known={list(RUNNERS)}")
    return fn()


def run_all(gates: tuple[str, ...] | None = None) -> dict[str, Any]:
    selected = tuple(gates or GATES)
    results = [run_gate(g) for g in selected]
    accepted = [r for r in results if r["verdict"] == "ACCEPTED"]
    blocked = [r for r in results if r["verdict"] == "BLOCKED"]
    manifest = seal(
        {
            "schema": MANIFEST_SCHEMA,
            "generated_at": utc_now(),
            "generated_by": "tools.acceptance.odyssey.run",
            "gates": [
                {"gate": r["gate"], "verdict": r["verdict"], "path": r["path"]}
                for r in results
            ],
            "n_gates": len(results),
            "n_accepted": len(accepted),
            "n_blocked": len(blocked),
            "accepted_ids": [r["gate"] for r in accepted],
            "blocked_ids": [r["gate"] for r in blocked],
            "criterion_altered": False,
            "note": (
                "A truthful BLOCKED with the exact missing input is success for "
                "this lane. No acceptance criterion was altered."
            ),
        }
    )
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    man_path = RECEIPTS_DIR / "ODYSSEY_HMF_FUSION_MANIFEST.json"
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "results": results,
        "n_accepted": len(accepted),
        "n_blocked": len(blocked),
        "manifest": str(man_path),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", choices=GATES, action="append")
    args = ap.parse_args(argv)
    summary = run_all(tuple(args.gate) if args.gate else None)
    print(json.dumps(
        {
            "n_accepted": summary["n_accepted"],
            "n_blocked": summary["n_blocked"],
            "manifest": summary["manifest"],
            "gates": [
                {"gate": r["gate"], "verdict": r["verdict"], "path": r["path"]}
                for r in summary["results"]
            ],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
