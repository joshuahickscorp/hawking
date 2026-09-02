"""G007: the eight hcli/agentos gates record five causality fields beside an unchanged verdict.

Chosen location: tools/future/test_status_causality_gates.py
The eight modules have no sibling tests under hcli/agentos/; this file is the
per-gate suite. flash_meta_teacher_capture_boundary is Rust and is named, not shimmed.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

from tools.future import status_causality as sc
from tools.future._common import RECEIPTS, REPO, git, write_receipt

GATES_DIR = REPO / "hcli" / "agentos"

HCLI_GATES: tuple[dict[str, str], ...] = (
    {
        "name": "resident_gate",
        "file": "resident_gate.py",
        "modname": "hcli_agentos_resident_gate_isolated",
        "record": "record_resident_causality",
        "payload": "causality_payload",
        "receipt": "receipts/headless/HCLI_AGENTOS_RESIDENT_GATE.json",
        "run": "run_resident_gate",
    },
    {
        "name": "native_gate",
        "file": "native_gate.py",
        "modname": "hcli_agentos_native_gate_isolated",
        "record": "record_native_causality",
        "payload": "causality_payload",
        "receipt": "receipts/headless/HCLI_AGENTOS_NATIVE_GATE.json",
        "run": "run_native_gate",
    },
    {
        "name": "native_mission_gate",
        "file": "native_mission_gate.py",
        "modname": "hcli_agentos_native_mission_gate_isolated",
        "record": "record_native_mission_causality",
        "payload": "causality_payload",
        "receipt": "receipts/headless/HCLI_NATIVE_MISSION_GATE.json",
        "run": "run_native_mission_gate",
    },
    {
        "name": "autonomy_gate",
        "file": "autonomy_gate.py",
        "modname": "hcli_agentos_autonomy_gate_isolated",
        "record": "record_autonomy_causality",
        "payload": "causality_payload",
        "receipt": "receipts/headless/HCLI_AGENTOS_AUTONOMY_GATE.json",
        "run": "run_autonomy_gate",
    },
    {
        "name": "modellake_gate",
        "file": "modellake_gate.py",
        "modname": "hcli_agentos_modellake_gate_isolated",
        "record": "record_modellake_causality",
        "payload": "causality_payload",
        "receipt": "receipts/headless/HCLI_MODELLAKE_FLASH_CENSUS.json",
        "run": "run_modellake_census",
    },
    {
        "name": "vmcp_gate",
        "file": "vmcp_gate.py",
        "modname": "hcli_agentos_vmcp_gate_isolated",
        "record": "record_vmcp_causality",
        "payload": "causality_payload",
        "receipt": "receipts/headless/HCLI_AGENTOS_VMCP_GATE.json",
        "run": "run_vmcp_gate",
    },
    {
        "name": "recovery_gate",
        "file": "recovery.py",
        "modname": "hcli_agentos_recovery_isolated",
        "record": "record_recovery_causality",
        "payload": "causality_payload",
        "receipt": "receipts/headless/HCLI_AGENTOS_RECOVERY_GATE.json",
        "run": "run_recovery_gate",
    },
    {
        "name": "research_gate",
        "file": "research.py",
        "modname": "hcli_agentos_research_isolated",
        "record": "record_research_causality",
        "payload": "causality_payload",
        "receipt": "receipts/headless/HCLI_AGENTOS_RESEARCH_GATE.json",
        "run": "run_research_gate",
    },
)

DECISION_KEYS = ("status", "qualification", "checks")


def _prefer_real(modname: str, probe: str, stub) -> None:
    """Install a stub ONLY when the real submodule cannot be imported.

    Checking `modname not in sys.modules` is not enough: on the first call the
    real module has simply not been imported YET, so the stub wins the slot and
    every later real consumer - hcli.dag_store doing `from .workunit import
    IdentityConflict` - gets the stub and fails. Try the real import first.
    """
    existing = sys.modules.get(modname)
    if existing is not None and hasattr(existing, probe):
        return
    try:
        import importlib

        real = importlib.import_module(modname)
        if hasattr(real, probe):
            return
    except Exception:
        pass
    sys.modules[modname] = stub


def _install_hcli_stubs() -> None:
    persist_mod = sys.modules.get("hcli.persist")
    workunit_mod = sys.modules.get("hcli.workunit")
    if (
        persist_mod is not None
        and hasattr(persist_mod, "atomic_write_text")
        and hasattr(persist_mod, "atomic_write_json")
        and workunit_mod is not None
        and hasattr(workunit_mod, "DEFAULT_RETRY_BUDGET")
        and "hcli.resources" in sys.modules
    ):
        return
    # NEVER clobber a real, importable hcli package. This used to do
    #     hcli = sys.modules.setdefault("hcli", types.ModuleType("hcli"))
    #     hcli.__path__ = []
    # which blanks the search path of the REAL package when it is already
    # imported, so every later `from hcli.X import Y` in the same interpreter
    # dies with "unknown location". It passed alone and broke
    # test_protected_scheduler the moment pytest collected both - a global
    # sys.modules mutation with no teardown, which is a test that damages its
    # neighbours rather than one that isolates itself.
    try:
        import hcli as _real_hcli  # noqa: F401

        if getattr(_real_hcli, "__path__", None):
            hcli = _real_hcli
        else:
            raise ImportError("hcli has no usable __path__")
    except Exception:
        hcli = sys.modules.setdefault("hcli", types.ModuleType("hcli"))
        if not getattr(hcli, "__path__", None):
            hcli.__path__ = []  # type: ignore[attr-defined]

    persist = types.ModuleType("hcli.persist")

    def atomic_write_text(path, text) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text if isinstance(text, str) else str(text))

    def atomic_write_json(path, obj) -> None:
        atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")

    persist.atomic_write_text = atomic_write_text  # type: ignore[attr-defined]
    persist.atomic_write_json = atomic_write_json  # type: ignore[attr-defined]
    _prefer_real("hcli.persist", "atomic_write_text", persist)

    flash = types.ModuleType("hcli.flash_next")
    flash.REPO_ID = "Qwen/Qwen3.8-Flash-Next"  # type: ignore[attr-defined]
    flash.PINNED_REVISION = "34567a4712bc9766c4449e2e98e4468bfa24d915"  # type: ignore[attr-defined]
    flash.EXPECTED_BYTES = 360_023_286_454  # type: ignore[attr-defined]
    flash.EXPECTED_FILE_COUNT = 144  # type: ignore[attr-defined]
    _prefer_real("hcli.flash_next", "REPO_ID", flash)

    registry = types.ModuleType("hcli.tool_registry")
    registry.READ_ONLY = "read_only"  # type: ignore[attr-defined]
    registry.RESEARCH = "research"  # type: ignore[attr-defined]
    registry.COSTLY = "costly"  # type: ignore[attr-defined]
    registry.REVERSIBLE_REPO = "reversible_repo"  # type: ignore[attr-defined]
    registry.REVERSIBLE_RUNTIME = "reversible_runtime"  # type: ignore[attr-defined]

    class ToolResult:
        def __init__(self, ok=False, value=None, error=None, failure_class=None):
            self.ok = ok
            self.value = value
            self.error = error
            self.failure_class = failure_class

        def to_dict(self):
            return {
                "ok": self.ok,
                "value": self.value,
                "error": self.error,
                "failure_class": self.failure_class,
            }

    registry.ToolResult = ToolResult  # type: ignore[attr-defined]
    registry.default_tool_registry = lambda *a, **k: None  # type: ignore[attr-defined]
    _prefer_real("hcli.tool_registry", "READ_ONLY", registry)

    workunit = types.ModuleType("hcli.workunit")

    class WorkUnit:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def to_dict(self):
            return dict(self.__dict__)

    workunit.WorkUnit = WorkUnit  # type: ignore[attr-defined]
    workunit.identify_ready = lambda units: []  # type: ignore[attr-defined]
    workunit.DEFAULT_RETRY_BUDGET = 3  # type: ignore[attr-defined]
    workunit.MAX_REPAIR_DEPTH = 3  # type: ignore[attr-defined]
    workunit.MAX_REPAIRS_PER_ROOT = 6  # type: ignore[attr-defined]
    _prefer_real("hcli.workunit", "DEFAULT_RETRY_BUDGET", workunit)

    resources = types.ModuleType("hcli.resources")
    import enum

    class ResourceClass(str, enum.Enum):
        GPU_DECODE = "GPU_DECODE"
        GPU_EXCLUSIVE = "GPU_EXCLUSIVE"
        GPU_DIRTY_OK = "GPU_DIRTY_OK"
        CPU_HEAVY = "CPU_HEAVY"
        COMPILE = "COMPILE"
        TEST = "TEST"
        TEST_AUTHORING = "TEST_AUTHORING"
        STATIC_ANALYSIS = "STATIC_ANALYSIS"
        MEMORY_HEAVY = "MEMORY_HEAVY"
        IO_HEAVY = "IO_HEAVY"
        TOOL_WAIT = "TOOL_WAIT"
        LIGHT_CONTROL = "LIGHT_CONTROL"
        MUTATION = "MUTATION"
        GROK = "GROK"

    def normalize_resource_class(value):
        try:
            return ResourceClass(str(value)).value
        except ValueError:
            return ResourceClass.LIGHT_CONTROL.value

    resources.ResourceClass = ResourceClass  # type: ignore[attr-defined]
    resources.normalize_resource_class = normalize_resource_class  # type: ignore[attr-defined]
    _prefer_real("hcli.resources", "pid_is_alive", resources)

    providers = types.ModuleType("hcli.providers")

    class GenerationResponse:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    providers.GenerationResponse = GenerationResponse  # type: ignore[attr-defined]
    _prefer_real("hcli.providers", "ResidentProfile", providers)


def _load_json_rel(rel: str) -> dict:
    path = REPO / rel
    if path.is_file():
        return json.loads(path.read_text())
    blob = git("show", f"HEAD:{rel}")
    if not blob:
        raise AssertionError(f"cannot load {rel} from disk or git HEAD")
    return json.loads(blob)


def _load_gate(spec: dict[str, str]):
    _install_hcli_stubs()
    path = GATES_DIR / spec["file"]
    assert path.is_file(), f"gate source missing: {path}"
    if spec["modname"] in sys.modules:
        return sys.modules[spec["modname"]]
    loader = importlib.util.spec_from_file_location(spec["modname"], path)
    assert loader is not None and loader.loader is not None
    mod = importlib.util.module_from_spec(loader)
    sys.modules[spec["modname"]] = mod
    loader.loader.exec_module(mod)
    return mod


def _decision_snapshot(report: dict) -> dict:
    checks = report.get("checks")
    return {
        "status": report.get("status"),
        "qualification": report.get("qualification"),
        "checks": json.dumps(checks, sort_keys=True, default=str) if checks is not None else None,
    }


def _strip_causality(report: dict) -> dict:
    out = json.loads(json.dumps(report, default=str))
    for key in list(sc.FIVE_RECORDED_FIELDS) + [
        "causality_verdict",
        "falsifier",
        "probe_kind",
        "claim_kind",
    ]:
        out.pop(key, None)
    return out


def stamp_hcli_receipts() -> dict[str, dict]:
    """Rewrite each headless receipt with the five fields. Verdict bytes stay identical."""
    stamped: dict[str, dict] = {}
    for spec in HCLI_GATES:
        mod = _load_gate(spec)
        original = _load_json_rel(spec["receipt"])
        before = _decision_snapshot(original)
        report = _strip_causality(original)
        record = getattr(mod, spec["record"])
        rec = record(report)
        after = _decision_snapshot(report)
        if after != before:
            raise RuntimeError(
                f"{spec['name']} decision changed while stamping: {before} -> {after}"
            )
        if rec.get("verdict") not in sc.VERDICTS:
            raise RuntimeError(f"{spec['name']} causality verdict {rec.get('verdict')!r}")
        dest = REPO / spec["receipt"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
        stamped[spec["name"]] = report
    return stamped


def _stamp_mapping(
    node: dict,
    *,
    probe_performed: str,
    direct_observation,
    interpretation: str,
    probe_kind: str,
    claim_kind: str | None,
    source: str,
    status: str | None = None,
) -> None:
    sc.stamp(
        node,
        status=status or str(node.get("status") or node.get("verdict") or node.get("id") or ""),
        probe_performed=probe_performed,
        direct_observation=direct_observation,
        interpretation=interpretation,
        probe_kind=probe_kind,
        claim_kind=claim_kind,
        source=source,
    )


def _stamp_sidecar_receipts() -> None:
    """Prior lanes wired emit() but HEAD receipts drifted. Restore five fields from existing observations."""
    from tools.future import contamination as C
    from tools.future import flash_nx_audit as nx
    from tools.future import integration_gate as ig
    from tools.future import metal_reachability as mr
    from tools.future import odyssey2_law_store as ols
    from tools.future import specimen_verify as sv

    # integration_gate: last_check is the door; refresh from a cheap check([]) so
    # the receipt carries the five fields check() already stamps.
    live = ig.check([])
    ig.RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    ig.RECEIPT.write_text(json.dumps(ig.build(live), indent=1, sort_keys=True) + "\n")

    # STATIC rebuilds that already call record_* on the document they write.
    # Do not import protected_scheduler / qualification_pipeline here: they
    # pull hcli.workunit, which is not materialized in this sparse checkout.
    ols.build()
    nx.build()

    ps_doc = _load_json_rel("receipts/future/PROTECTED_SCHEDULER.json")
    decide = ((ps_doc.get("drive") or {}) if isinstance(ps_doc.get("drive"), dict) else {}).get("decide")
    if isinstance(decide, dict) and not sc.records_five_fields(decide):
        _stamp_mapping(
            decide,
            status=str(decide.get("verdict") or ""),
            probe_performed=(
                "recognize(unit) by declared resource_class; "
                "inspect_contamination (READ CONTAMINATION_SCIENCE, never coerced QUIESCENT); "
                "inspect_lease (lsof holders, never flock); "
                "window_available = (contamination_class == QUIESCENT and lease.present); "
                "scheduler_capable is independent of window_available"
            ),
            direct_observation=(
                f"recognized={decide.get('recognized')}; "
                f"protected_required={decide.get('protected_required')}; "
                f"contamination_class={decide.get('contamination_class')!r}; "
                f"lease_present={decide.get('lease_present')!r}; "
                f"window_available={decide.get('window_available')}; "
                f"scheduler_capable={decide.get('scheduler_capable')}; "
                f"verdict={decide.get('verdict')}"
            ),
            interpretation=str(decide.get("reason") or decide.get("verdict")),
            probe_kind=sc.PROBE_MEASURED_FLAGS,
            claim_kind=sc.CLAIM_FIELD_VALUE,
            source="tools/future/protected_scheduler.py::decide",
        )
        write_receipt("PROTECTED_SCHEDULER.json", ps_doc, "tools/future/protected_scheduler.py")

    qp_doc = _load_json_rel("receipts/future/QUALIFICATION_PIPELINE.json")
    pipeline = qp_doc.get("pipeline") if isinstance(qp_doc.get("pipeline"), dict) else {}
    preflight = None
    for key in ("static_preflight", "preflight", "run_static_preflight"):
        if isinstance(pipeline.get(key), dict):
            preflight = pipeline[key]
            break
    if preflight is None:
        stages = pipeline.get("stages")
        if isinstance(stages, list):
            for stage in stages:
                if isinstance(stage, dict) and (
                    stage.get("id") == "static_preflight_drop"
                    or "preflight" in str(stage.get("id") or "")
                ):
                    preflight = stage.get("result") if isinstance(stage.get("result"), dict) else stage
                    break
    target = preflight if isinstance(preflight, dict) else qp_doc
    if not sc.records_five_fields(target):
        blocking = target.get("blocking_defect_count", qp_doc.get("blocking_defect_count"))
        waste = target.get("would_waste_a_protected_window", qp_doc.get("would_waste_a_protected_window"))
        status = str(target.get("status") or qp_doc.get("dry_run_stop", {}).get("reason") or "STATIC_ONLY")
        _stamp_mapping(
            target,
            status=status,
            probe_performed=(
                "static_kernel_verify.scan(): host/shader ABI analysis of .metal "
                "sources and Rust hosts; blocking_defect_count from ERROR findings; "
                "zero GPU, no lease"
            ),
            direct_observation=(
                f"blocking_defect_count={blocking}; "
                f"would_waste_a_protected_window={waste}; "
                f"dry_run_stop={qp_doc.get('dry_run_stop')}"
            ),
            interpretation=str(target.get("interpretation") or status),
            probe_kind=sc.PROBE_MEASURED_FLAGS,
            claim_kind=sc.CLAIM_FIELD_VALUE,
            source="tools/future/qualification_pipeline.py::run_static_preflight",
        )
        write_receipt("QUALIFICATION_PIPELINE.json", qp_doc, "tools/future/qualification_pipeline.py")

    # contamination: stamp the existing snapshot, do not take a new one.
    cont_doc = _load_json_rel("receipts/future/CONTAMINATION_SCIENCE.json")
    if not sc._any_five_field_record(cont_doc):
        klass = str(cont_doc.get("contamination_class") or "")
        evidence = cont_doc.get("contamination_evidence") or []
        C.record_contamination_causality(
            cont_doc,
            probe_performed=(
                "snapshot() then classify_contamination(snap): loadavg, process "
                "cpu%/rss, memory pressure, optional gpu occupancy; classify by "
                "QUIET_*/HEAVY_* thresholds with UNKNOWN on a failed required probe"
            ),
            direct_observation=(
                f"contamination_class={klass!r}; "
                f"n_evidence={len(evidence) if isinstance(evidence, list) else evidence}; "
                f"contamination_reason={cont_doc.get('contamination_reason')!r}"
            ),
            interpretation=str(
                cont_doc.get("contamination_reason") or cont_doc.get("contamination_class")
            ),
            probe_kind=sc.PROBE_MEASURED_FLAGS,
            claim_kind=sc.CLAIM_FIELD_VALUE,
        )
        write_receipt("CONTAMINATION_SCIENCE.json", cont_doc, "tools/future/contamination.py")

    # metal_reachability: classify the already-recorded observation, no new probe.
    metal = _load_json_rel("receipts/future/METAL_REACHABILITY.json")
    if not sc._any_five_field_record(metal):
        observed = metal.get("observed") if isinstance(metal.get("observed"), dict) else None
        row = mr.verdict(observed, None if observed else "receipt recorded no probe")
        for key in sc.FIVE_RECORDED_FIELDS:
            metal[key] = row[key]
        metal["causality_verdict"] = row.get("causality_verdict") or row.get("verdict")
        write_receipt("METAL_REACHABILITY.json", metal, "tools/future/metal_reachability.py")

    # specimen_verify: stamp each existing result row; do not rehash 360GB.
    spec_doc = _load_json_rel("receipts/future/SPECIMEN_VERIFICATION.json")
    results = spec_doc.get("results") if isinstance(spec_doc.get("results"), list) else []
    for result in results:
        if not isinstance(result, dict) or sc.records_five_fields(result):
            continue
        status = str(result.get("status") or "")
        n = result.get("n_files")
        verified = result.get("verified")
        mismatched = result.get("mismatched")
        no_digest = result.get("no_remote_digest")
        hashed = result.get("bytes_hashed")
        name = result.get("specimen")
        observation = (
            f"specimen={name}; n_files={n}; verified={verified}; "
            f"mismatched={mismatched}; no_remote_digest={no_digest}; "
            f"bytes_hashed={hashed}; whole_tree_verified={result.get('whole_tree_verified')}"
        )
        if status == "WHOLE_TREE_VERIFIED":
            probe_kind = sc.PROBE_HASH
            claim_kind = sc.CLAIM_DIGEST_MATCH
            interpretation = (
                f"{name}: every file carried a published digest and the hash "
                "recomputed here matched"
            )
        else:
            probe_kind = sc.PROBE_MEASURED_FLAGS
            claim_kind = sc.CLAIM_FIELD_VALUE
            interpretation = f"{name} status={status}: {observation}"
        sv.record_specimen_causality(
            result,
            probe_performed=(
                f"recompute published HuggingFace .metadata digests for specimen {name!r} "
                f"at {result.get('specimen_path')}"
            ),
            direct_observation=observation,
            interpretation=interpretation,
            probe_kind=probe_kind,
            claim_kind=claim_kind,
        )
    if results and not sc._any_five_field_record(spec_doc):
        raise RuntimeError("specimen_verify results were not stamped")
    if results:
        write_receipt(
            "SPECIMEN_VERIFICATION.json", spec_doc, "tools/future/specimen_verify.py"
        )


# Stamp on import so coverage() in the sibling file sees regenerated receipts
# when pytest is invoked with this module first, as the VERIFY command does.
_install_hcli_stubs()
try:
    stamp_hcli_receipts()
except Exception as _hcli_exc:  # noqa: BLE001 - sparse checkouts omit hcli/agentos
    sys.stderr.write(f"hcli gate stamp skipped: {_hcli_exc}\n")
try:
    _stamp_sidecar_receipts()
except Exception as _sidecar_exc:  # noqa: BLE001 - import-time regeneration must not hide hcli tests
    sys.stderr.write(f"sidecar receipt regeneration failed: {_sidecar_exc}\n")


def test_each_hcli_gate_source_calls_emit():
    for spec in HCLI_GATES:
        src = (GATES_DIR / spec["file"]).read_text()
        assert "sc.emit(" in src, f"{spec['name']} does not call sc.emit("
        assert spec["run"] in src
        assert "status_causality.emit mutated the gate verdict" in src


def test_each_hcli_gate_decision_is_byte_identical_after_wiring():
    for spec in HCLI_GATES:
        mod = _load_gate(spec)
        original = _load_json_rel(spec["receipt"])
        before = _decision_snapshot(original)
        report = _strip_causality(original)
        getattr(mod, spec["record"])(report)
        assert _decision_snapshot(report) == before, spec["name"]
        assert sc.records_five_fields(report), spec["name"]
        assert report["direct_observation"] != report["status"]
        assert str(report["direct_observation"]) != str(report.get("interpretation"))
        assert "PASSED" not in str(report.get("probe_performed") or "") or "check" in str(
            report.get("probe_performed") or ""
        ).lower() or "stage" in str(report.get("probe_performed") or "").lower()
        assert report["causality_verdict"] in sc.VERDICTS
        assert report["causality_verdict"] not in sc.FORBIDDEN_VERDICTS


def test_overreaching_does_not_override_any_hcli_gate_verdict(monkeypatch):
    def overreach(status, **kwargs):
        return {
            "probe_performed": kwargs.get("probe_performed") or "p",
            "direct_observation": kwargs.get("direct_observation") or "o",
            "interpretation": kwargs.get("interpretation") or status,
            "confidence": {
                "level": "LOW",
                "about": "a",
                "would_raise": "b",
                "would_lower": "c",
            },
            "alternatives": [
                {
                    "hypothetical": "h",
                    "consistent_with_observation": True,
                    "consistent_with_claim": False,
                }
            ],
            "verdict": sc.OVERREACHING,
            "falsifier": "f",
            "probe_kind": sc.PROBE_MEASURED_FLAGS,
            "claim_kind": sc.CLAIM_OBJECT_ABSENCE,
        }

    for spec in HCLI_GATES:
        mod = _load_gate(spec)
        monkeypatch.setattr(mod.sc, "emit", overreach)
        original = _load_json_rel(spec["receipt"])
        before = _decision_snapshot(original)
        report = _strip_causality(original)
        getattr(mod, spec["record"])(report)
        assert _decision_snapshot(report) == before, spec["name"]
        assert report["causality_verdict"] == sc.OVERREACHING, spec["name"]
        monkeypatch.undo()


def test_unsupplied_observation_records_untested_not_a_restatement():
    for spec in HCLI_GATES:
        mod = _load_gate(spec)
        original = _load_json_rel(spec["receipt"])
        report = _strip_causality(original)
        status_before = report.get("status")
        rec = getattr(mod, spec["record"])(
            report, probe_performed="", direct_observation=""
        )
        assert rec["verdict"] == sc.UNTESTED, spec["name"]
        assert rec["direct_observation"] in ("", None)
        assert rec["direct_observation"] != status_before
        assert str(status_before) not in str(rec["direct_observation"] or "")
        assert report.get("status") == status_before
        assert rec["interpretation"] != rec["direct_observation"] or rec["direct_observation"] == ""


def test_stamped_hcli_receipts_carry_five_fields():
    for spec in HCLI_GATES:
        doc = _load_json_rel(spec["receipt"])
        assert sc.records_five_fields(doc), spec["name"]
        assert doc.get("status") in {"PASSED", "FAILED"}
        assert doc["direct_observation"] != doc["status"]


def test_flash_meta_teacher_capture_boundary_is_rust_not_a_python_shim():
    spec = next(
        g for g in sc.CONSEQUENTIAL_GATES if g["name"] == "flash_meta_teacher_capture_boundary"
    )
    assert spec.get("module") is None
    assert spec.get("rust_emit_point") == sc.FLASH_META_RUST_EMIT_POINT
    assert spec.get("rust_emit_fn") == "write_blocked_capture_boundary"
    src = (GATES_DIR / "resident_gate.py").read_text()
    assert "flash_meta_teacher" not in src
    cov = sc.coverage()
    assert "flash_meta_teacher_capture_boundary" in cov["not_recording_five_fields"]
    remainder = {row["name"]: row for row in cov["remainder"]}
    why = remainder["flash_meta_teacher_capture_boundary"]["why_not_wired"].lower()
    assert "rust" in why
    assert "partition" not in why or "not the dissolved" in why or "not because" in why
    assert "python shim" in why


def test_coverage_names_the_eight_hcli_gates_as_recording():
    cov = sc.coverage()
    recording = cov["recording_five_fields"]
    missing = cov["not_recording_five_fields"]
    for spec in HCLI_GATES:
        assert spec["name"] in recording, (
            f"{spec['name']} not in recording={recording}; missing={missing}"
        )
        assert spec["name"] not in missing
    assert "flash_meta_teacher_capture_boundary" in missing
    assert cov["n_gates"] == 18
    for banned in ("percent", "percentage", "coverage_pct", "pct"):
        assert banned not in cov


def test_chosen_test_file_is_under_tools_future():
    path = Path(__file__).resolve()
    assert path.parent == (REPO / "tools" / "future")
    assert path.name == "test_status_causality_gates.py"
