"""Doctor diagnosis engine (H-ROADMAP §9, gene I-B).

FOREIGN MODEL → fingerprint → organ graph → negative science → technique
applicability (in 9.1 order) → three zeros (9.2) → diagnosis with evidence
and uncertainty (9.3).

This lane diagnoses receipts and sealed-specimen METADATA. It does not load
weight bytes, does not train, does not download, and does not run a GPU
campaign. A Doctor that always says healthy is refused: overall verdict is
never HEALTHY.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.doctor.access import AccessLog, WeightBytesForbidden, is_weight_file
from tools.doctor.anatomy import (
    FLASH_SLUG,
    organs_from_fingerprint,
    load_specimen_metadata,
    resolve_specimen,
)
from tools.doctor.order import TECHNIQUE_ORDER, Technique, walk_order
from tools.doctor.zeros import (
    FAIL,
    PASS,
    THREE_ZEROS,
    UNKNOWN,
    BROKEN_ORGAN,
    ROUTED_EXPERT_ORGAN,
    TIED_EMBED_ORGAN,
    check_three_zeros,
    ordinary_quantization,
    organs_from_doc,
    whole_artifact_organ,
    zeros_as_dict,
)


SCHEMA = "hawking.future.doctor_depth.v1"
RECEIPT = "DOCTOR_DEPTH.json"
RECORDED_BY = "tools/doctor/engine.py"
EVIDENCE_TIER = "STATIC"
FORBIDDEN_OVERALL = "HEALTHY"

QWEN80_RECEIPT = "receipts/QWEN80_BIT_BUDGET_LEDGER.json"
FLASH_EBPW_RECEIPT = "receipts/headless/FLASH_EBPW_BUDGET.json"
NNS_RECEIPT = "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json"
QWEN06_SLUG = "Qwen--Qwen3-0.6B@c1899de289a0"


def _repo() -> Path:
    return _REPO


def retrieve_negative(
    *,
    model: str | None,
    organ: str | None,
    family: str,
    scars: Sequence[Any] | None = None,
    nns: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Call tools.future.negative_index.refuse_if_dead. scars=[] skips ingest."""
    cited: list[dict[str, Any]] = []
    if isinstance(nns, Mapping):
        for entry in nns.get("entries") or []:
            if not isinstance(entry, Mapping):
                continue
            blob = json.dumps(entry, default=str).lower()
            if family.lower() in blob or (organ or "").lower() in blob:
                cited.append(
                    {
                        "id": entry.get("id"),
                        "claim_refuted": entry.get("claim_refuted"),
                        "kind": entry.get("kind"),
                        "source": NNS_RECEIPT,
                    }
                )
    refused = None
    try:
        from tools.future.negative_index import refuse_if_dead

        refused = refuse_if_dead(
            {
                "model": model,
                "organ": organ,
                "hypothesis_family": family,
            },
            scars=list(scars) if scars is not None else None,
        )
    except Exception as exc:  # index may be sparse-missing; never invent a scar
        refused = {"error": f"{type(exc).__name__}: {exc}", "called": True}
    return {
        "refuse_if_dead": refused,
        "nns_hits": cited[:6],
        "family": family,
        "called_refuse_if_dead": True,
    }


def _ask_technique(
    tech: Technique,
    organ: Mapping[str, Any],
    zeros: Mapping[str, Any],
    *,
    model: str | None,
    nns: Mapping[str, Any] | None,
    scars: Sequence[Any] | None,
    negative_science: bool,
) -> dict[str, Any]:
    """Ask one 9.1 stage. Always asked, in order. Never skipped."""
    cls = str(organ.get("organ_class") or organ.get("name") or "")
    zmap = zeros
    zero_verdicts = {
        name: (zmap[name].verdict if hasattr(zmap[name], "verdict") else zmap[name]["verdict"])
        for name in THREE_ZEROS
        if name in zmap
    }
    applicability = "UNKNOWN"
    reason = "no architectural tell for this stage"
    closed_by_scar = None

    if tech.index == 0:  # ELIMINATE
        if organ.get("tie_word_embeddings") is True and (
            "embed" in cls or "lm_head" in cls
        ):
            applicability = "PLAUSIBLE"
            reason = "tied embeddings: one of embed/lm_head can be eliminated as independent storage"
        elif cls in {"vision_backbone", "vision"}:
            applicability = "PLAUSIBLE"
            reason = "vision can be eliminated on a text-only decode"
        else:
            applicability = "UNLIKELY"
            reason = "architecture still names this organ"
    elif tech.index == 2:  # SHARE
        cosine = organ.get("cross_expert_cosine")
        if isinstance(cosine, (int, float)) and float(cosine) < 0.05:
            applicability = "CLOSED_BY_EVIDENCE"
            reason = (
                f"cross_expert_cosine={float(cosine):.6f} is near zero; "
                "SHARE precondition (correlated experts) fails"
            )
        elif zero_verdicts.get("ZERO_INDEPENDENT_INFORMATION") == PASS:
            applicability = "PLAUSIBLE"
            reason = "ZERO_INDEPENDENT_INFORMATION already PASSed; sharing/derivation exists"
        else:
            applicability = "UNLIKELY"
            reason = "no sharing evidence on this organ"
    elif tech.index == 3:  # FACTORIZE
        rank1 = organ.get("rank_1_energy")
        if isinstance(rank1, (int, float)) and float(rank1) < 0.5:
            applicability = "CLOSED_BY_EVIDENCE"
            reason = f"rank_1_energy={float(rank1):.4f} does not collapse; FACTORIZE precondition fails"
        else:
            applicability = "UNKNOWN"
            reason = "singular spectrum ABSENT on metadata"
    elif tech.index == 4:  # GENERATE
        if organ.get("generated_from"):
            applicability = "PLAUSIBLE"
            reason = f"generated_from={organ.get('generated_from')}"
        else:
            applicability = "UNLIKELY"
            reason = "no generator declared"
    elif tech.index == 5:  # ROUTE
        if "routed" in cls or (
            isinstance(organ.get("n_experts"), int)
            and isinstance(organ.get("experts_per_tok"), int)
            and organ["n_experts"] > organ["experts_per_tok"]
        ):
            applicability = "PLAUSIBLE"
            reason = "routing is already the architecture"
        else:
            applicability = "UNLIKELY"
            reason = "organ is not a routed expert bank"
    elif tech.index == 6:  # SENSITIVITY
        if organ.get("capability_sensitivity"):
            applicability = "PLAUSIBLE"
            reason = f"receipt sensitivity={organ.get('capability_sensitivity')}"
        else:
            applicability = "UNKNOWN"
            reason = "Hessian / sensitivity ABSENT; not invented"
    elif tech.index == 7:  # HEAL
        applicability = "UNLIKELY"
        reason = "healers are scar-gated; a Qwen/GLM low-rank heal is not reopened from metadata"
    elif tech.index == 8:  # QUANTIZE
        if ordinary_quantization(zeros):
            applicability = "ORDINARY_QUANTIZATION"
            reason = "all three zeros FAIL; a low-BPW result here is ordinary quantization (9.2)"
        else:
            applicability = "PLAUSIBLE"
            reason = "zeros did not all fail; quantize is still a later-stage representation, not the first ask"
    elif tech.index == 9:  # NATIVE
        applicability = "UNKNOWN"
        reason = "native_kernel_status not on this organ; PLAN_ONLY receipts are not kernels"
    elif tech.index == 10:  # STATE
        if "deltanet" in cls or "state" in cls or "kv" in cls:
            applicability = "PLAUSIBLE"
            reason = "recurrent/KV organ"
        else:
            applicability = "NOT_APPLICABLE"
            reason = "not a state organ"
    elif tech.index == 11:  # REMOVE COMPUTE
        if zero_verdicts.get("ZERO_EXECUTION") == PASS:
            applicability = "PLAUSIBLE"
            reason = "ZERO_EXECUTION PASSed; skip/route already exists"
        else:
            applicability = "UNKNOWN"
            reason = "activation sparsity / MoD not declared"
    elif tech.index == 12:  # REDUCE DECODE
        if cls in {"mtp", "ngram_engine", "ngram"}:
            applicability = "PLAUSIBLE"
            reason = "decode-reduction organ present in architecture"
        else:
            applicability = "NOT_APPLICABLE"
            reason = "not a decode-reduction organ"
    elif tech.index == 13:  # DEVICE
        applicability = "STATIC_ONLY"
        reason = (
            "Apple M3 Ultra is present; FPGA/U50, DGX, eGPU are ABSENT. "
            "Nothing about absent hardware is a measurement."
        )
    elif tech.index == 14:  # VERIFY
        applicability = "UNKNOWN"
        reason = "no capability contract executed in this lane"
    elif tech.index == 1:  # REPARAMETERIZE
        applicability = "UNKNOWN"
        reason = "outlier / incoherence features ABSENT without parent tensors"

    scar_block = None
    if negative_science and tech.family in {"DOC-SHARE", "DOC-HEALING", "DOC-COORDINATES"}:
        scar_block = retrieve_negative(
            model=model,
            organ=cls,
            family=tech.family,
            scars=scars,
            nns=nns,
        )
        hit = scar_block.get("refuse_if_dead")
        if isinstance(hit, Mapping) and hit.get("refused"):
            applicability = "CLOSED_BY_SCAR"
            closed_by_scar = hit
            reason = "negative_index.refuse_if_dead cited a scar"

    return {
        "index": tech.index,
        "name": tech.name,
        "family": tech.family,
        "asked": True,
        "applicability": applicability,
        "reason": reason,
        "zeros_consulted": list(tech.zeros),
        "cheapest_experiment": tech.cheapest_experiment,
        "negative_science": scar_block,
        "closed_by_scar": closed_by_scar,
        "evidence_tier": EVIDENCE_TIER,
        "uncertainty": "applicability is architectural / receipt-cited, not a new measurement",
    }


def _overall(organ_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passed: list[str] = []
    failed: list[str] = []
    unknown: list[str] = []
    ordinary_organs: list[str] = []
    zero_organs: list[str] = []
    for row in organ_rows:
        name = str(row.get("name") or row.get("organ_class") or "?")
        z = row.get("three_zeros") or {}
        if row.get("ordinary_quantization"):
            ordinary_organs.append(name)
        any_pass = False
        for zname in THREE_ZEROS:
            cell = z.get(zname) or {}
            v = cell.get("verdict") if isinstance(cell, Mapping) else None
            key = f"{name}:{zname}"
            if v == PASS:
                passed.append(key)
                any_pass = True
            elif v == FAIL:
                failed.append(key)
            elif v == UNKNOWN:
                unknown.append(key)
        if any_pass:
            zero_organs.append(name)
    if zero_organs and ordinary_organs:
        verdict = "MIXED"
    elif zero_organs:
        verdict = "ZERO_AVAILABLE"
    elif ordinary_organs and not unknown:
        verdict = "ORDINARY_QUANTIZATION"
    elif unknown and not passed and not failed:
        verdict = "INSUFFICIENT_EVIDENCE"
    elif ordinary_organs:
        verdict = "MIXED"
    else:
        verdict = "INSUFFICIENT_EVIDENCE"
    if verdict == FORBIDDEN_OVERALL:
        raise RuntimeError("Doctor refused to emit HEALTHY")
    return {
        "verdict": verdict,
        "never_healthy": True,
        "forbidden_overall": FORBIDDEN_OVERALL,
        "zeros_passed": passed,
        "zeros_failed": failed,
        "zeros_unknown": unknown,
        "organs_with_a_zero": zero_organs,
        "organs_ordinary_quantization": ordinary_organs,
        "uncertainty": (
            "verdict is the zeros search on metadata/receipts, not a capability "
            "certificate and not a hardware measurement"
        ),
        "evidence_tier": EVIDENCE_TIER,
    }


def _diagnose_organs(
    organs: Sequence[Mapping[str, Any]],
    *,
    model: str | None,
    nns: Mapping[str, Any] | None,
    scars: Sequence[Any] | None,
    negative_science: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for organ in organs:
        zeros = check_three_zeros(organ)
        asks = [
            _ask_technique(
                tech,
                organ,
                zeros,
                model=model,
                nns=nns,
                scars=scars,
                negative_science=negative_science and tech.family in {
                    "DOC-SHARE",
                    "DOC-HEALING",
                    "DOC-COORDINATES",
                },
            )
            for tech in walk_order()
        ]
        names = [a["name"] for a in asks]
        if names != [t.name for t in TECHNIQUE_ORDER]:
            raise RuntimeError("technique order was not asked as 9.1 states")
        next_exp = None
        for a in asks:
            if a["applicability"] in {"PLAUSIBLE", "UNKNOWN"} and a["index"] < 8:
                next_exp = {
                    "technique": a["name"],
                    "experiment": a["cheapest_experiment"],
                    "why": "cheapest discriminating ask still open before QUANTIZE",
                    "evidence_tier": EVIDENCE_TIER,
                }
                break
        if next_exp is None:
            next_exp = {
                "technique": "QUANTIZE",
                "experiment": TECHNIQUE_ORDER[8].cheapest_experiment,
                "why": "earlier stages closed or inapplicable",
                "evidence_tier": EVIDENCE_TIER,
            }
        rows.append(
            {
                "name": organ.get("name") or organ.get("organ_class"),
                "organ_class": organ.get("organ_class"),
                "three_zeros": zeros_as_dict(zeros),
                "ordinary_quantization": ordinary_quantization(zeros),
                "technique_order": asks,
                "next_experiment": next_exp,
                "evidence_tier": EVIDENCE_TIER,
                "uncertainty": "zeros and applicability are STATIC architecture/receipt evidence",
                "flags": {
                    "n_experts": organ.get("n_experts"),
                    "experts_per_tok": organ.get("experts_per_tok"),
                    "tie_word_embeddings": organ.get("tie_word_embeddings"),
                    "cross_expert_cosine": organ.get("cross_expert_cosine"),
                    "rank_1_energy": organ.get("rank_1_energy"),
                    "capability_sensitivity": organ.get("capability_sensitivity"),
                },
            }
        )
    return rows


def _load_nns(repo: Path, access: AccessLog | None) -> dict[str, Any] | None:
    path = repo / NNS_RECEIPT
    if not path.is_file():
        return None
    try:
        if access is not None:
            doc = access.open_json(path)
        else:
            doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def diagnose_receipt(
    path: Path,
    *,
    access: AccessLog | None = None,
    negative_science: bool = False,
    scars: Sequence[Any] | None = None,
) -> dict[str, Any]:
    access = access or AccessLog()
    doc = access.open_json(path)
    if not isinstance(doc, Mapping):
        raise ValueError(f"receipt is not an object: {path}")
    organs = organs_from_doc(doc)
    if not organs:
        organs = [whole_artifact_organ(doc)]
    repo = _repo()
    nns = _load_nns(repo, access) if negative_science else None
    model = None
    if isinstance(doc.get("model"), str):
        model = doc["model"]
    elif isinstance(doc.get("identity"), str):
        model = doc["identity"]
    rows = _diagnose_organs(
        organs,
        model=model,
        nns=nns,
        scars=scars if scars is not None else ([] if not negative_science else None),
        negative_science=negative_science,
    )
    ebpw = None
    try:
        from tools.future.ebpw_categories import validate as ebpw_validate

        try:
            src = str(path.relative_to(repo))
        except ValueError:
            src = str(path)
        ebpw = ebpw_validate(doc, source_path=src)
    except Exception as exc:
        ebpw = {"error": f"{type(exc).__name__}: {exc}", "called": True}
    rel = str(path)
    try:
        rel = str(path.relative_to(repo))
    except ValueError:
        pass
    return {
        "kind": "RECEIPT",
        "name": rel,
        "schema": doc.get("schema"),
        "status": doc.get("status"),
        "organs": rows,
        "overall": _overall(rows),
        "ebpw_validate": {
            "verdict": (ebpw or {}).get("verdict"),
            "can_promote": (ebpw or {}).get("can_promote"),
            "called": True,
        },
        "io": access.report(),
        "evidence_tier": EVIDENCE_TIER,
        "weights_opened": False,
    }


def diagnose_specimen(
    root: Path,
    *,
    access: AccessLog | None = None,
    negative_science: bool = False,
    scars: Sequence[Any] | None = None,
    repo: Path | None = None,
) -> dict[str, Any]:
    access = access or AccessLog()
    repo = repo or _repo()
    loaded = load_specimen_metadata(root, access, repo=repo)
    fp = loaded["fingerprint"]
    organs = organs_from_fingerprint(fp)
    nns = _load_nns(repo, None) if negative_science else None
    model = fp.get("model_type") or fp.get("architecture_family")
    rows = _diagnose_organs(
        organs,
        model=str(model) if model else None,
        nns=nns,
        scars=scars if scars is not None else ([] if not negative_science else None),
        negative_science=negative_science,
    )
    # Proof: no shard in the specimen tree was opened.
    for opened in access.files_opened:
        if is_weight_file(opened):
            raise WeightBytesForbidden(f"specimen diagnosis opened a weight file: {opened}")
    return {
        "kind": "SPECIMEN_METADATA",
        "name": root.name,
        "path": str(root),
        "fingerprint": {k: v for k, v in fp.items() if k != "config_keys"} | {
            "config_key_count": len(fp.get("config_keys") or []),
        },
        "n_weight_names_in_index": len(loaded.get("weight_names") or []),
        "organs": rows,
        "overall": _overall(rows),
        "io": access.report(),
        "evidence_tier": EVIDENCE_TIER,
        "weights_opened": False,
        "weights_opened_claimed_by_fingerprint": fp.get("weights_opened"),
    }


def diagnose(
    target: str | Path,
    *,
    access: AccessLog | None = None,
    negative_science: bool = False,
    scars: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Diagnose a receipt path, a specimen directory, or a lake slug."""
    access = access or AccessLog()
    path = Path(target)
    repo = _repo()
    if not path.is_absolute():
        cand = repo / path
        if cand.is_file() or cand.is_dir():
            path = cand
    if path.is_file() and path.suffix == ".json":
        if is_weight_file(path):
            raise WeightBytesForbidden(path)
        return diagnose_receipt(
            path, access=access, negative_science=negative_science, scars=scars
        )
    spec = resolve_specimen(path if path.is_dir() else target)
    if spec is None:
        spec = resolve_specimen(target)
    if spec is not None:
        return diagnose_specimen(
            spec,
            access=access,
            negative_science=negative_science,
            scars=scars,
            repo=repo,
        )
    raise FileNotFoundError(f"Doctor cannot diagnose {target!r}: not a receipt or specimen metadata dir")


def zeros_controls() -> dict[str, Any]:
    """Each zero FAILING on a bad input and PASSing on a good one."""
    broken = check_three_zeros(BROKEN_ORGAN)
    tied = check_three_zeros(TIED_EMBED_ORGAN)
    routed = check_three_zeros(ROUTED_EXPERT_ORGAN)
    return {
        "broken": {
            "organ": BROKEN_ORGAN["name"],
            "three_zeros": zeros_as_dict(broken),
            "ordinary_quantization": ordinary_quantization(broken),
            "expect": "all three FAIL",
        },
        "tied_embed": {
            "organ": TIED_EMBED_ORGAN["name"],
            "three_zeros": zeros_as_dict(tied),
            "ordinary_quantization": ordinary_quantization(tied),
            "expect": "ZERO_STORAGE PASS, ZERO_INDEPENDENT_INFORMATION PASS",
        },
        "routed_expert": {
            "organ": ROUTED_EXPERT_ORGAN["name"],
            "three_zeros": zeros_as_dict(routed),
            "ordinary_quantization": ordinary_quantization(routed),
            "expect": "ZERO_EXECUTION PASS",
        },
        "all_three_fail_on_broken": ordinary_quantization(broken) is True,
        "storage_pass_on_tied": tied["ZERO_STORAGE"].verdict == PASS,
        "info_pass_on_tied": tied["ZERO_INDEPENDENT_INFORMATION"].verdict == PASS,
        "execution_pass_on_routed": routed["ZERO_EXECUTION"].verdict == PASS,
        "storage_fail_on_broken": broken["ZERO_STORAGE"].verdict == FAIL,
        "info_fail_on_broken": broken["ZERO_INDEPENDENT_INFORMATION"].verdict == FAIL,
        "execution_fail_on_broken": broken["ZERO_EXECUTION"].verdict == FAIL,
    }


def _found_prior() -> list[dict[str, str]]:
    return [
        {
            "path": "tools/headless/doctor_diagnosis.py",
            "what": "N047 organ diagnosis on parent tensors. Not called: it streams BF16 weights.",
        },
        {
            "path": "tools/headless/doctor_transfer.py",
            "what": "CANONICAL_ORDER + THREE_ZEROS as documentation. Extended here as fail-able checks.",
        },
        {
            "path": "tools/odyssey/doctor_tournament.py",
            "what": "9.1-like order with safetensors probes. probes() is not called.",
        },
        {
            "path": "tools/future/specimen_events.py",
            "what": "fingerprint_from_config (config+index, weights_opened=False). Algorithm reused; module not imported (hcli.persist absent in this sparse cone).",
        },
        {
            "path": "tools/future/negative_index.py",
            "what": "refuse_if_dead is a real call site from retrieve_negative.",
        },
        {
            "path": "tools/future/ebpw_categories.py",
            "what": "validate() calls doctor_zeros_for_doc → check_three_zeros.",
        },
        {
            "path": "tools/future/flash_schools.py",
            "what": "THREE_ZEROS as school questions. This lane makes them FAIL-able checks.",
        },
        {
            "path": "tools/doctor_seal.py",
            "what": "G124 sealer. Not replaced.",
        },
        {
            "path": "receipts/headless/FLASH_DOCTOR_EXPERT_BANK_SCREEN.json",
            "what": "Prior Doctor bank screen; cited as STATIC cosine/rank evidence for Flash experts.",
        },
        {
            "path": "hcli/doctor/__init__.py",
            "what": "Ownership marker (OWNED_PATHS). Read via git show; not edited.",
        },
    ]


def build(repo: Path | None = None) -> Path:
    """Diagnose real artifacts, watch the zeros fail/pass, write the receipt."""
    from tools.future._common import write_receipt

    repo = repo or _repo()
    controls = zeros_controls()
    if not controls["all_three_fail_on_broken"]:
        raise AssertionError("broken organ must fail all three zeros")
    if not (
        controls["storage_pass_on_tied"]
        and controls["info_pass_on_tied"]
        and controls["execution_pass_on_routed"]
    ):
        raise AssertionError("good fixtures must PASS the zeros they are built to pass")

    artifacts: list[dict[str, Any]] = []
    q80 = repo / QWEN80_RECEIPT
    if q80.is_file():
        artifacts.append(
            diagnose_receipt(q80, negative_science=False, scars=[])
        )
    flash_ebpw = repo / FLASH_EBPW_RECEIPT
    if flash_ebpw.is_file():
        artifacts.append(
            diagnose_receipt(flash_ebpw, negative_science=False, scars=[])
        )
    flash_spec = resolve_specimen(FLASH_SLUG)
    if flash_spec is not None:
        artifacts.append(
            diagnose_specimen(flash_spec, negative_science=False, scars=[], repo=repo)
        )
    q06 = resolve_specimen(QWEN06_SLUG)
    if q06 is not None:
        artifacts.append(
            diagnose_specimen(q06, negative_science=False, scars=[], repo=repo)
        )

    if len(artifacts) < 2:
        raise RuntimeError(
            f"Doctor must diagnose at least 2 real artifacts; got {len(artifacts)}"
        )

    weight_bytes = sum(int((a.get("io") or {}).get("weight_bytes_loaded") or 0) for a in artifacts)
    if weight_bytes != 0:
        raise RuntimeError("weight bytes were loaded")
    for a in artifacts:
        opened = (a.get("io") or {}).get("metadata_files_opened") or []
        for p in opened:
            if is_weight_file(p):
                raise RuntimeError(f"weight file in access log: {p}")

    # Real call of refuse_if_dead (scars=[] so ingest is not required).
    share_call = retrieve_negative(
        model="qwen3.8-27b",
        organ="routed_experts",
        family="shared_basis",
        scars=[],
        nns=_load_nns(repo, None),
    )

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Deepen I-B Doctor so it diagnoses real receipts and sealed specimen "
            "metadata, walking the 9.1 technique order and treating the 9.2 three "
            "zeros as checks that can FAIL."
        ),
        "roadmap": {
            "section": "9",
            "title": "DOCTOR — MODEL PHYSICIAN AND EXPERIMENT PLANNER",
            "technique_order": "9.1",
            "three_zeros": "9.2",
            "diagnosis_evidence": "9.3",
            "gene_card": "I-B — Doctor",
            "source": "/Users/scammermike/Downloads/H-ROADMAP.md",
        },
        "evidence_tier": EVIDENCE_TIER,
        "claim_boundary": (
            "STATIC metadata/receipt diagnosis. No weight bytes, no GPU, no "
            "training, no download. Hardware present: Apple M3 Ultra. ABSENT: "
            "FPGA/U50, DGX, eGPU — anything about those is a model, not a measurement."
        ),
        "technique_order": [t.as_dict() for t in TECHNIQUE_ORDER],
        "three_zeros": list(THREE_ZEROS),
        "three_zeros_controls": controls,
        "artifacts": artifacts,
        "n_artifacts": len(artifacts),
        "artifact_names": [a.get("name") for a in artifacts],
        "weight_bytes_loaded": 0,
        "weights_opened": False,
        "negative_science_call": {
            "called_refuse_if_dead": share_call.get("called_refuse_if_dead"),
            "family": share_call.get("family"),
            "nns_hit_ids": [h.get("id") for h in share_call.get("nns_hits") or []],
            "note": "refuse_if_dead invoked with scars=[] so ingest is not required; NNS receipt is cited",
        },
        "recovered_implementation": _found_prior(),
        "gaps_closed": [
            "tools/doctor/ exists as the I-B diagnosis engine on receipts + specimen metadata.",
            "9.1 technique order is walked in order (asked=true on every stage).",
            "9.2 three zeros are checks that FAIL on a broken organ and PASS on good fixtures.",
            "ordinary_quantization is true only when all three zeros FAIL.",
            "ebpw_categories.validate calls check_three_zeros (a call of the gate's own symbol).",
            "AccessLog refuses weight shards; receipt records weight_bytes_loaded=0.",
        ],
        "negative_findings": [
            "tools/headless/doctor_diagnosis.py still requires parent tensors; this lane does not call it.",
            "tools/odyssey/doctor_tournament.py.probes() opens safetensors; not called.",
            "Activation/logit/hidden-state probes and Hessian are ABSENT (named, not invented).",
            "FPGA/U50, DGX, eGPU are ABSENT hardware; DEVICE COMPILE stays STATIC_ONLY.",
        ],
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()
