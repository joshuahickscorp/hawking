#!/usr/bin/env python3
"""Assemble HAWKING_ACTION_HANDOFF.json from disk so a fresh session can resume.

Every required field is MEASURED from a receipt, ledger, git, or status file,
or ABSENT with an explicit reason. Numbers are copied from the receipts that
own them. This module does not emit a campaign figure as a measurement.

    python3 tools/headless/action_handoff.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECEIPTS = REPO / "receipts" / "headless"
RECEIPT = RECEIPTS / "HAWKING_ACTION_HANDOFF.json"
SCHEMA = "hawking.headless.action_handoff.v1"

ULTRAGOAL_ROOT = Path.home() / ".claude" / "ultragoal"
ULTRAGOAL_ACTIVE = ULTRAGOAL_ROOT / "active"
PREDECESSOR_LEDGER = ULTRAGOAL_ROOT / "hawking-noetic-onebit" / "GOAL.md"


def _resolve_active_ledger() -> Path:
    """The ACTIVE goal, resolved from the armed slots on disk.

    This was pinned to hawking-noetic-onebit, which is the SEALED PREDECESSOR. A handoff
    that names a finished campaign as current is worse than no handoff: a fresh session
    reads it and resumes the wrong mission. One slot per session lives in
    ~/.claude/ultragoal/active/<session>.json; the newest armed one that still has an
    unVERIFIED obligation is the live campaign.
    """
    best = None
    if ULTRAGOAL_ACTIVE.is_dir():
        for slot in sorted(ULTRAGOAL_ACTIVE.glob("*.json")):
            try:
                doc = json.loads(slot.read_text())
            except Exception:
                continue
            led = Path(str(doc.get("ledger", "")))
            if not led.is_file():
                continue
            text = led.read_text()
            if "- [ ]" not in text:          # every obligation VERIFIED: not the live one
                continue
            mtime = slot.stat().st_mtime
            if best is None or mtime > best[0]:
                best = (mtime, led)
    if best:
        return best[1]
    return PREDECESSOR_LEDGER


ACTIVE_LEDGER = _resolve_active_ledger()
GROK_TASKS = Path.home() / ".claude-grok" / "tasks"

GENESIS_REL = "receipts/ascent-2026-08-18/Genesis.m3ultra.nx"
PARENT_REL = "receipts/headless/NOETIC_PARENT_A.json"
CONTROL_REL = "receipts/headless/CONVENTIONAL_CONTROL_SET.json"
GPU_LEDGER_REL = "receipts/headless/GPU_LEDGER.json"
BENCH_REL = "receipts/headless/PRODUCTION_BENCH.json"
ROOF_REL = "receipts/headless/BANDWIDTH_ROOF.json"
NEGSCI_REL = "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json"
MACHINE_REL = "receipts/headless/MACHINE_GENOME.json"
QWEN_EQ_REL = "receipts/headless/QWEN_MAX_EQUILIBRIUM.json"
GROK_EQ_REL = "receipts/headless/GROK_MAX_EQUILIBRIUM.json"
GIT_LEDGER_REL = "receipts/headless/GIT_STORAGE_LEDGER.json"
POLICY_REL = "docs/ultragoals/ARTIFACT_STORAGE_POLICY.md"
GPU_LEDGER_PY_REL = "tools/headless/gpu_ledger.py"

MEASURED = "MEASURED"
ABSENT = "ABSENT"

REQUIRED_KEYS = (
    "git",
    "ledgers",
    "leading_noetic_executable",
    "kernels_runtime_machine",
    "gpu_roof",
    "current_bottleneck",
    "concurrency_equilibrium",
    "artifact_store",
    "git_storage_policy",
    "active_grok_tasks",
    "next_workunits",
    "negative_science",
    # directive §100 names twelve things a fresh session must recover. The four below were
    # not in the v1 handoff, so a resuming session could not tell which specimen was under
    # study, what the libraries held, or what was queued next.
    "current_specimen",
    "odyssey_queue",
    "doctor_state",
    "libraries",
)

# The id prefix is per-campaign: the predecessor used N###, this one uses G###. Pinning
# the letter made the parser silently return ZERO obligations against the live ledger, and
# a handoff reporting 0/0 looks exactly like a finished campaign.
OBLIGATION_HEAD_RE = re.compile(
    r"^- \[([ xX])\] ([A-Z]\d+)\s+[—–-]\s+([A-Z0-9_/-]+)",
    re.M,
)
STATUS_RE = re.compile(r"\|\s*status:\s*([A-Z_]+)")
PEAK_GB_S_RE = re.compile(
    r"(?m)^PEAK_GB_S\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*(?:#.*)?$"
)
CITE_RE = re.compile(r"`([^`]+)`")


# ---------------------------------------------------------------------------
# Quantity constructors. ABSENT is exactly {kind, reason} as the test names.
# ---------------------------------------------------------------------------

def measured(value: Any, source: Sequence[str] | str) -> Dict[str, Any]:
    if isinstance(source, str):
        src: List[str] = [source]
    else:
        src = [str(s) for s in source]
    return {"kind": MEASURED, "source": src, "value": value}


def absent(reason: str) -> Dict[str, Any]:
    return {"kind": ABSENT, "reason": reason}


def canonical_dumps(doc: Any) -> str:
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# Disk / git readers. Never invent a number a receipt already owns.
# ---------------------------------------------------------------------------

def git_run(args: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )


def git_out(args: Sequence[str]) -> Optional[str]:
    r = git_run(args)
    if r.returncode != 0:
        return None
    return (r.stdout or "").rstrip("\n")


def git_exists(rel: str) -> bool:
    r = git_run(["cat-file", "-e", f"HEAD:{rel}"])
    return r.returncode == 0


def git_show_text(rel: str) -> Optional[str]:
    r = git_run(["show", f"HEAD:{rel}"])
    if r.returncode != 0:
        return None
    return r.stdout


def primary_checkout() -> Optional[Path]:
    cd = git_out(["rev-parse", "--git-common-dir"])
    if not cd:
        return None
    p = Path(cd)
    if not p.is_absolute():
        p = (REPO / p).resolve()
    else:
        p = p.resolve()
    if p.name == ".git":
        return p.parent
    return p.parent


def resolve_on_disk(rel: str) -> Optional[Path]:
    """Return an existing file for `rel` without writing or sparse-checkout."""
    seen = []
    here = REPO / rel
    seen.append(here)
    if here.is_file():
        return here
    primary = primary_checkout()
    if primary is not None:
        cand = primary / rel
        if cand not in seen and cand.is_file():
            return cand
    return None


def load_json_path(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_rel_json(rel: str) -> Tuple[Optional[Any], Optional[str]]:
    """Load JSON from worktree disk, else HEAD, else primary checkout."""
    here = REPO / rel
    if here.is_file():
        doc = load_json_path(here)
        if doc is not None:
            return doc, rel
        return None, None
    text = git_show_text(rel)
    if text is not None:
        try:
            return json.loads(text), f"HEAD:{rel}"
        except json.JSONDecodeError:
            return None, None
    disk = resolve_on_disk(rel)
    if disk is not None:
        doc = load_json_path(disk)
        if doc is not None:
            return doc, str(disk)
    return None, None


def load_rel_text(rel: str) -> Tuple[Optional[str], Optional[str]]:
    here = REPO / rel
    if here.is_file():
        try:
            return here.read_text(), rel
        except OSError:
            return None, None
    text = git_show_text(rel)
    if text is not None:
        return text, f"HEAD:{rel}"
    disk = resolve_on_disk(rel)
    if disk is not None:
        try:
            return disk.read_text(), str(disk)
        except OSError:
            return None, None
    return None, None


def dig(obj: Any, *path: Any, default: Any = None) -> Any:
    cur = obj
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def parse_spec_peak_gb_s() -> Tuple[Optional[float], Optional[str]]:
    """Read PEAK_GB_S from gpu_ledger.py source. Genesis does not own 819."""
    text, origin = load_rel_text(GPU_LEDGER_PY_REL)
    if text is None or origin is None:
        return None, None
    m = PEAK_GB_S_RE.search(text)
    if not m:
        return None, None
    return float(m.group(1)), origin


# ---------------------------------------------------------------------------
# Field builders
# ---------------------------------------------------------------------------

def field_git() -> Dict[str, Any]:
    head = git_out(["rev-parse", "HEAD"])
    branch = git_out(["rev-parse", "--abbrev-ref", "HEAD"])
    if not head:
        return absent("git rev-parse HEAD failed; this worktree is not a git repo")
    remotes = git_out(["remote"])
    remote_names = [ln for ln in (remotes or "").splitlines() if ln.strip()]
    remote_urls = {}
    for name in remote_names:
        url = git_out(["remote", "get-url", name])
        if url:
            remote_urls[name] = url
    upstream = git_out(["rev-parse", "--abbrev-ref", "@{u}"])
    ahead = None
    behind = None
    state = "no_upstream"
    if upstream:
        counts = git_out(["rev-list", "--left-right", "--count", "HEAD...@{u}"])
        if counts and "\t" in counts:
            left, right = counts.split("\t", 1)
            try:
                ahead = int(left)
                behind = int(right)
            except ValueError:
                ahead = None
                behind = None
        if ahead == 0 and behind == 0:
            state = "in_sync"
        elif ahead is not None and behind is not None:
            state = "diverged" if ahead and behind else (
                "ahead" if ahead else "behind"
            )
        else:
            state = "upstream_configured"
    return measured(
        {
            "head": head,
            "branch": branch,
            "remote_sync": {
                "remotes": remote_urls,
                "upstream": upstream,
                "ahead": ahead,
                "behind": behind,
                "state": state,
            },
        },
        source="git rev-parse / git remote / git rev-list",
    )


def _parse_obligations(text: str) -> List[Dict[str, Any]]:
    matches = list(OBLIGATION_HEAD_RE.finditer(text))
    rows: List[Dict[str, Any]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[m.start() : end]
        sm = STATUS_RE.search(block)
        status = sm.group(1) if sm else None
        box = m.group(1)
        checked = box.lower() == "x"
        if status is None:
            status = "VERIFIED" if checked else "PENDING"
        first = block.splitlines()[0].strip() if block.splitlines() else ""
        rows.append(
            {
                "id": m.group(2),
                "name": m.group(3),
                "checkbox": box,
                "checked": checked,
                "status": status,
                "headline": first,
            }
        )
    return rows


def _completed_ledgers(goal_text: str) -> List[Dict[str, Any]]:
    found: Dict[str, Dict[str, Any]] = {}

    def add(path: Path, why: str) -> None:
        key = str(path)
        rec = found.get(key)
        if rec is None:
            found[key] = {
                "path": key,
                "exists": path.is_file(),
                "cited_from": [why],
            }
        else:
            if why not in rec["cited_from"]:
                rec["cited_from"].append(why)

    if ULTRAGOAL_ROOT.is_dir():
        for p in sorted(ULTRAGOAL_ROOT.glob("*/COMPLETED.md")):
            add(p, f"{ULTRAGOAL_ROOT}/*/COMPLETED.md")
        parent = ULTRAGOAL_ROOT / "hawking-headless-v3" / "DISCOVERY_PARENT_32of32.md"
        if parent.is_file():
            add(parent, "hawking-headless-v3/DISCOVERY_PARENT_32of32.md")

    for raw in CITE_RE.findall(goal_text):
        if "ultragoal" not in raw:
            continue
        if "hawking-noetic-onebit" in raw:
            continue
        if not any(tok in raw for tok in ("GOAL.md", "COMPLETED.md", "DISCOVERY_")):
            continue
        add(Path(raw).expanduser(), "cited in active GOAL.md")

    return [found[k] for k in sorted(found)]


def field_ledgers() -> Dict[str, Any]:
    if not ACTIVE_LEDGER.is_file():
        return absent(
            f"active ledger not on disk: {ACTIVE_LEDGER} "
            f"(resolved from {ULTRAGOAL_ACTIVE}; falls back to {PREDECESSOR_LEDGER})"
        )
    text = ACTIVE_LEDGER.read_text()
    rows = _parse_obligations(text)
    by_status: Dict[str, List[str]] = {}
    for r in rows:
        by_status.setdefault(r["status"], []).append(r["id"])
    verified = [r["id"] for r in rows if r["checked"] or r["status"] == "VERIFIED"]
    pending = [r["id"] for r in rows if (not r["checked"]) or r["status"] == "PENDING"]
    # Keep unique, preserve parse order.
    def uniq(ids: List[str]) -> List[str]:
        seen = set()
        out = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    verified = uniq(verified)
    pending = uniq(pending)
    return measured(
        {
            "active_ledger_path": str(ACTIVE_LEDGER),
            "ultragoal": ACTIVE_LEDGER.parent.name,
            "resolved_from": str(ULTRAGOAL_ACTIVE),
            "predecessor_ledger": str(PREDECESSOR_LEDGER),
            "counts": {
                "n_total": len(rows),
                "n_checked": sum(1 for r in rows if r["checked"]),
                "n_unchecked": sum(1 for r in rows if not r["checked"]),
                "by_status": {k: len(v) for k, v in sorted(by_status.items())},
            },
            "ids_by_status": {k: v for k, v in sorted(by_status.items())},
            "ids_checked": [r["id"] for r in rows if r["checked"]],
            "ids_pending": [r["id"] for r in rows if not r["checked"]],
            "completed_ledgers": _completed_ledgers(text),
        },
        source=str(ACTIVE_LEDGER),
    )


def field_leading_noetic() -> Dict[str, Any]:
    parent, parent_origin = load_rel_json(PARENT_REL)
    if parent is None or parent_origin is None:
        return absent(
            f"{PARENT_REL} not on disk and not at HEAD; cannot cite the sealed leader"
        )
    closure = dig(parent, "executable_closure") or {}
    rep = dig(parent, "RepresentationGenome") or {}
    rt = dig(parent, "RuntimeGenome") or {}
    disp = parent.get("dispatch_count") or rt.get("dispatches_per_token") or {}
    artifact = parent.get("artifact") or {}
    sealed = artifact.get("path")
    sources = [parent_origin]

    mlx: Any
    ctrl, ctrl_origin = load_rel_json(CONTROL_REL)
    if ctrl is None or ctrl_origin is None:
        mlx = absent(f"{CONTROL_REL} not on disk and not at HEAD")
    else:
        sources.append(ctrl_origin)
        archived = dig(ctrl, "archived", "headline_vs_mlx", "value", "mlx_4bit_tps")
        historical = dig(
            ctrl, "comparison", "historical_headline", "value", "mlx_4bit_tps"
        )
        live_batch1 = dig(
            ctrl, "live", "metrics", "concurrency", "value", "batch_1_generation_tps_median"
        )
        live_decode = dig(
            ctrl, "comparison", "live_mlx_over_archived_llama", "live_mlx_decode_tps"
        )
        mlx_value: Dict[str, Any] = {
            "archived_headline_tps_field": (
                "archived.headline_vs_mlx.value.mlx_4bit_tps"
            ),
            "live_batch_1_field": (
                "live.metrics.concurrency.value.batch_1_generation_tps_median"
            ),
            "live_decode_field": (
                "comparison.live_mlx_over_archived_llama.live_mlx_decode_tps"
            ),
        }
        if archived is None:
            mlx_value["archived_headline_tps"] = absent(
                "archived.headline_vs_mlx.value.mlx_4bit_tps missing in "
                + CONTROL_REL
            )
        else:
            mlx_value["archived_headline_tps"] = archived
        if historical is None:
            mlx_value["historical_headline_tps"] = absent(
                "comparison.historical_headline.value.mlx_4bit_tps missing"
            )
        else:
            mlx_value["historical_headline_tps"] = historical
        if live_batch1 is None:
            mlx_value["live_batch_1_generation_tps_median"] = absent(
                "live.metrics.concurrency.value.batch_1_generation_tps_median missing"
            )
        else:
            mlx_value["live_batch_1_generation_tps_median"] = live_batch1
        if live_decode is None:
            mlx_value["live_decode_tps"] = absent(
                "comparison.live_mlx_over_archived_llama.live_mlx_decode_tps missing"
            )
        else:
            mlx_value["live_decode_tps"] = live_decode
        mlx = mlx_value

    sealed_present = bool(sealed) and Path(str(sealed)).expanduser().is_dir()
    return measured(
        {
            "closure_sha": closure.get("closure_sha256"),
            "full_executable_sha": closure.get("full_executable_sha256"),
            "complete_ebpw": rep.get("complete_ebpw"),
            "dispatches": disp,
            "sealed_path": sealed,
            "sealed_path_present": sealed_present,
            "immutable": parent.get("immutable"),
            "mix_id": artifact.get("mix_id"),
            "n_files": artifact.get("n_files") or closure.get("n_files"),
            "decode_tok_s": rt.get("decode_tok_s"),
            "conventional_mlx_control": mlx,
        },
        source=sources,
    )


def _exact_metal_hashes(kern: Dict[str, Any]) -> Any:
    metal = kern.get("metal_source_hashes") or {}
    if isinstance(metal, dict) and "exact_metal_source_hashes" in metal:
        return metal.get("exact_metal_source_hashes")
    if isinstance(metal, dict):
        return {
            k: v
            for k, v in metal.items()
            if isinstance(v, dict) and "sha256" in v
        }
    return metal


def field_kernels_runtime_machine() -> Dict[str, Any]:
    parent, parent_origin = load_rel_json(PARENT_REL)
    genesis_text, genesis_origin = load_rel_text(GENESIS_REL)
    peak, peak_origin = parse_spec_peak_gb_s()

    if parent is None and genesis_text is None and peak is None:
        return absent(
            f"{PARENT_REL}, {GENESIS_REL}, and {GPU_LEDGER_PY_REL} PEAK_GB_S "
            "are all unreadable from disk/HEAD/primary checkout"
        )

    sources: List[str] = []
    value: Dict[str, Any] = {}

    if genesis_text is None or genesis_origin is None:
        value["machine_from_genesis"] = absent(
            f"{GENESIS_REL} is not in this worktree, not at HEAD, and not "
            "readable from the primary checkout (git-common-dir parent)"
        )
    else:
        sources.append(genesis_origin)
        try:
            genesis = json.loads(genesis_text)
        except json.JSONDecodeError:
            value["machine_from_genesis"] = absent(
                f"{genesis_origin} is not valid JSON"
            )
        else:
            mg = genesis.get("compiled_for_machine_genome") or {}
            value["machine_from_genesis"] = {
                "chipset": mg.get("chipset"),
                "gpu_cores": mg.get("gpu_cores"),
                "unified_memory_bytes": mg.get("unified_memory_bytes"),
                "metal_family": mg.get("metal_family"),
                "measured_roof_gb_s": mg.get("measured_roof_gb_s"),
                "roof_provenance": mg.get("roof_provenance"),
                "genome_digest": mg.get("genome_digest"),
                "in_this_worktree": (REPO / GENESIS_REL).is_file(),
                "in_head": git_exists(GENESIS_REL),
            }

    if peak is None or peak_origin is None:
        value["spec_peak_gb_s"] = absent(
            f"PEAK_GB_S assignment not found in {GPU_LEDGER_PY_REL}; "
            f"Genesis.m3ultra.nx does not record a spec-peak field"
        )
    else:
        sources.append(f"{peak_origin}:PEAK_GB_S")
        value["spec_peak_gb_s"] = peak
        value["spec_peak_gb_s_provenance"] = (
            f"{peak_origin} assignment PEAK_GB_S — not a field of {GENESIS_REL}"
        )

    if parent is None or parent_origin is None:
        value["kernels_from_parent_a"] = absent(f"{PARENT_REL} unreadable")
        value["runtime_from_parent_a"] = absent(f"{PARENT_REL} unreadable")
        value["machine_from_parent_a"] = absent(f"{PARENT_REL} unreadable")
    else:
        sources.append(parent_origin)
        kern = parent.get("KernelGenome") or {}
        rt = parent.get("RuntimeGenome") or {}
        mg_parent = parent.get("MachineGenome") or {}
        binary = rt.get("binary")
        value["kernels_from_parent_a"] = {
            "production_kernel": kern.get("production_kernel"),
            "fused_kernels": kern.get("fused_kernels"),
            "family": kern.get("family"),
            "runtime_div_diagnostic": kern.get("runtime_div_diagnostic"),
            "n_bound": kern.get("n_bound"),
            "n_declared_in_tree": kern.get("n_declared_in_tree"),
            "exact_metal_source_hashes": _exact_metal_hashes(kern),
            "all_shader_sources_concat_sha256": kern.get(
                "all_shader_sources_concat_sha256"
            ),
            "compiler_settings": kern.get("compiler_settings"),
        }
        value["runtime_from_parent_a"] = {
            "binary": binary,
            "binary_present": bool(binary) and Path(str(binary)).is_file(),
            "example": rt.get("example"),
            "profile": rt.get("profile"),
            "fusion_enable": rt.get("fusion_enable"),
            "dispatches_per_token": rt.get("dispatches_per_token"),
            "decode_tok_s": rt.get("decode_tok_s"),
            "incumbent_tok_s": rt.get("incumbent_tok_s"),
        }
        value["machine_from_parent_a"] = mg_parent

    genesis_digest = dig(value, "machine_from_genesis", "genome_digest")
    parent_digest = dig(value, "machine_from_parent_a", "genome_digest")
    if genesis_digest and parent_digest:
        value["genome_digest_match"] = genesis_digest == parent_digest

    return measured(value, source=sources)


def field_gpu_roof() -> Dict[str, Any]:
    here = REPO / ROOF_REL
    if here.is_file():
        doc = load_json_path(here)
        if doc is None:
            return absent(f"{ROOF_REL} exists but is not valid JSON")
        return measured(
            {
                "schema": doc.get("schema"),
                "receipt": ROOF_REL,
                "body": doc,
            },
            source=ROOF_REL,
        )
    text = git_show_text(ROOF_REL)
    if text is not None:
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            return absent(f"HEAD:{ROOF_REL} is not valid JSON")
        return measured(
            {
                "schema": doc.get("schema"),
                "receipt": f"HEAD:{ROOF_REL}",
                "body": doc,
            },
            source=f"HEAD:{ROOF_REL}",
        )
    return absent(
        f"{ROOF_REL} is not on disk in this worktree and is not at HEAD. "
        "N017 (BANDWIDTH_ROOF_IS_UNVERIFIED) is the obligation that writes it. "
        "GPU_LEDGER.json and Genesis.m3ultra.nx record inherited scoring roofs; "
        "those are not a measured physical DRAM roof and are not substituted here."
    )


def field_current_bottleneck() -> Dict[str, Any]:
    ledger, origin = load_rel_json(GPU_LEDGER_REL)
    if ledger is None or origin is None:
        return absent(
            f"{GPU_LEDGER_REL} not on disk and not at HEAD; bottleneck unreadable"
        )
    q80 = ledger.get("q80_anchor") or {}
    warm = ledger.get("warm") or {}
    fields = ledger.get("fields") or {}
    return measured(
        {
            "verdict": q80.get("verdict"),
            "reading": q80.get("reading"),
            "q4_incumbent": q80.get("q4_incumbent"),
            "q80_anchor": q80.get("anchor"),
            "gpu_as_fraction_of_wall": warm.get("gpu_as_fraction_of_wall"),
            "warm_tps": warm.get("tps"),
            "dispatches": warm.get("dispatches"),
            "command_buffers": warm.get("command_buffers"),
            "GPU_IDLE_GAPS_NS": fields.get("GPU_IDLE_GAPS_NS"),
            "DRAM_READ_BYTES": fields.get("DRAM_READ_BYTES"),
            "ACTIVE_BYTES_PER_TOKEN": ledger.get("ACTIVE_BYTES_PER_TOKEN"),
            "DRAM_BYTES_PER_TOKEN": ledger.get("DRAM_BYTES_PER_TOKEN"),
        },
        source=origin,
    )


def field_concurrency_equilibrium() -> Dict[str, Any]:
    sources: List[str] = []
    value: Dict[str, Any] = {}

    bench, bench_origin = load_rel_json(BENCH_REL)
    if bench is None or bench_origin is None:
        value["production_bench"] = absent(
            f"{BENCH_REL} not on disk and not at HEAD"
        )
    else:
        sources.append(bench_origin)
        value["production_bench"] = {
            "winner": bench.get("winner"),
            "ranking_quantity": bench.get("ranking_quantity"),
            "scaling_vs_c1_aggregate_tps": bench.get(
                "scaling_vs_c1_aggregate_tps"
            ),
            "c8": {
                "ran": dig(bench, "c8", "ran"),
                "reason": dig(bench, "c8", "reason"),
                "prior_ceiling": dig(bench, "c8", "prior_ceiling"),
                "measured_c2_vs_c1": dig(bench, "c8", "measured_c2_vs_c1"),
                "measured_c4_vs_c1": dig(bench, "c8", "measured_c4_vs_c1"),
            },
            "prior_ceiling_not_rederived": bench.get(
                "prior_ceiling_not_rederived"
            ),
        }

    mg, mg_origin = load_rel_json(MACHINE_REL)
    if mg is None or mg_origin is None:
        value["machine_genome"] = absent(f"{MACHINE_REL} not on disk and not at HEAD")
    else:
        sources.append(mg_origin)
        value["machine_genome"] = {
            "RESIDENT_RUNTIME_LIMIT": mg.get("RESIDENT_RUNTIME_LIMIT"),
            "resident_limit_reason": mg.get("resident_limit_reason"),
            "ACTIVE_DECODE_LIMIT": mg.get("ACTIVE_DECODE_LIMIT"),
            "active_decode_reason": mg.get("active_decode_reason"),
            "single_decoder_tps": mg.get("single_decoder_tps"),
            "best_aggregate_tps": mg.get("best_aggregate_tps"),
            "aggregate_scaling_vs_1": mg.get("aggregate_scaling_vs_1"),
        }

    qwen, qwen_origin = load_rel_json(QWEN_EQ_REL)
    if qwen is None or qwen_origin is None:
        value["qwen_max_equilibrium"] = absent(
            f"{QWEN_EQ_REL} not on disk and not at HEAD"
        )
    else:
        sources.append(qwen_origin)
        value["qwen_max_equilibrium"] = {
            "finding": qwen.get("finding"),
            "contradicts_the_machine_genome": qwen.get(
                "contradicts_the_machine_genome"
            ),
        }

    grok, grok_origin = load_rel_json(GROK_EQ_REL)
    if grok is None or grok_origin is None:
        value["grok_lane_equilibrium"] = absent(
            f"{GROK_EQ_REL} not on disk and not at HEAD"
        )
    else:
        sources.append(grok_origin)
        value["grok_lane_equilibrium"] = {
            "useful_equilibrium": grok.get("useful_equilibrium"),
            "equilibrium_reason": grok.get("equilibrium_reason"),
            "rule": grok.get("rule"),
            "note": (
                "Grok-lane accepted/hour equilibrium, not decode-slot equilibrium"
            ),
        }

    if not sources:
        return absent(
            f"{BENCH_REL}, {MACHINE_REL}, {QWEN_EQ_REL}, and {GROK_EQ_REL} "
            "are all unreadable"
        )
    value["receipts_disagree"] = True
    value["receipts_disagree_note"] = (
        "MACHINE_GENOME, QWEN_MAX_EQUILIBRIUM, and PRODUCTION_BENCH do not "
        "report the same decode-slot ceiling. A fresh session must not collapse "
        "them; each figure stays attached to its receipt."
    )
    return measured(value, source=sources)


def field_artifact_store() -> Dict[str, Any]:
    policy_here = (REPO / POLICY_REL).is_file() or git_exists(POLICY_REL)
    ledger_here = (REPO / GIT_LEDGER_REL).is_file() or git_exists(GIT_LEDGER_REL)
    if policy_here or ledger_here:
        # N019 has landed something; surface what is actually on disk.
        sources = []
        value: Dict[str, Any] = {}
        if (REPO / GIT_LEDGER_REL).is_file() or git_exists(GIT_LEDGER_REL):
            doc, origin = load_rel_json(GIT_LEDGER_REL)
            if doc is not None and origin is not None:
                sources.append(origin)
                value["git_storage_ledger"] = {
                    "schema": doc.get("schema") if isinstance(doc, dict) else None,
                    "receipt": origin,
                }
        policy_text, policy_origin = load_rel_text(POLICY_REL)
        if policy_text is not None and policy_origin is not None:
            sources.append(policy_origin)
            value["artifact_storage_policy"] = {
                "path": policy_origin,
                "bytes": len(policy_text.encode("utf-8")),
            }
        if sources:
            return measured(value, source=sources)
    return absent(
        "N019 (STORAGE_ARCHITECTURE) has not written "
        f"{GIT_LEDGER_REL} or {POLICY_REL} on disk or at HEAD, so the "
        "content-addressed artifact-store location is not yet named. "
        "receipts/headless/ARTIFACT_LEDGER.json is a prior census, not the "
        "N019 store, and is not substituted."
    )


def field_git_storage_policy() -> Dict[str, Any]:
    gi = REPO / ".gitignore"
    gi_text: Optional[str] = None
    gi_origin: Optional[str] = None
    if gi.is_file():
        try:
            gi_text = gi.read_text()
            gi_origin = ".gitignore"
        except OSError:
            gi_text = None
    if gi_text is None:
        gi_text = git_show_text(".gitignore")
        if gi_text is not None:
            gi_origin = "HEAD:.gitignore"
    if gi_text is None or gi_origin is None:
        return absent(".gitignore is not on disk and not at HEAD")

    sha = git_out(["log", "-1", "--format=%H", "--", ".gitignore"])
    subject = git_out(["log", "-1", "--format=%s", "--", ".gitignore"])
    section = "Repo storage optimization (S018 followup)"
    has_section = section in gi_text

    n019_ledger = (REPO / GIT_LEDGER_REL).is_file() or git_exists(GIT_LEDGER_REL)
    n019_policy = (REPO / POLICY_REL).is_file() or git_exists(POLICY_REL)

    n019_ledger_field: Any
    if n019_ledger:
        n019_ledger_field = GIT_LEDGER_REL
    else:
        n019_ledger_field = absent(
            f"{GIT_LEDGER_REL} not on disk and not at HEAD; N019 PENDING"
        )
    n019_policy_field: Any
    if n019_policy:
        n019_policy_field = POLICY_REL
    else:
        n019_policy_field = absent(
            f"{POLICY_REL} not on disk and not at HEAD; N019 PENDING"
        )

    return measured(
        {
            "enacted_pointer": gi_origin,
            "enacted_section_present": has_section,
            "enacted_section": section if has_section else None,
            "last_commit_touching_gitignore": sha,
            "last_subject_touching_gitignore": subject,
            "n019_git_storage_ledger": n019_ledger_field,
            "n019_artifact_storage_policy": n019_policy_field,
        },
        source=gi_origin,
    )


def field_active_grok_tasks() -> Dict[str, Any]:
    if not GROK_TASKS.is_dir():
        return absent(f"Grok tasks directory not on disk: {GROK_TASKS}")
    running: List[Dict[str, Any]] = []
    for status_path in sorted(GROK_TASKS.glob("*/status")):
        try:
            st = status_path.read_text().strip()
        except OSError:
            continue
        if st != "running":
            continue
        task_dir = status_path.parent
        rec: Dict[str, Any] = {
            "id": task_dir.name,
            "status": st,
            "status_path": str(status_path),
        }
        meta_path = task_dir / "metadata.json"
        if meta_path.is_file():
            meta = load_json_path(meta_path)
            if isinstance(meta, dict):
                for k in (
                    "task_id",
                    "mode",
                    "profile",
                    "model",
                    "workdir",
                    "branch",
                    "base_commit",
                    "started_at",
                ):
                    if k in meta:
                        rec[k] = meta[k]
        running.append(rec)
    return measured(
        {
            "tasks_root": str(GROK_TASKS),
            "n_running": len(running),
            "running": running,
        },
        source=str(GROK_TASKS / "*/status"),
    )


def field_next_workunits() -> Dict[str, Any]:
    if not ACTIVE_LEDGER.is_file():
        return absent(f"active ledger not on disk: {ACTIVE_LEDGER}")
    text = ACTIVE_LEDGER.read_text()
    rows = _parse_obligations(text)
    pending = [r for r in rows if not r["checked"]]
    return measured(
        pending,
        source=str(ACTIVE_LEDGER),
    )


def field_negative_science() -> Dict[str, Any]:
    doc, origin = load_rel_json(NEGSCI_REL)
    if doc is None or origin is None:
        return absent(f"{NEGSCI_REL} not on disk and not at HEAD")
    return measured(
        {
            "path": origin,
            "schema": doc.get("schema"),
            "counts": doc.get("counts"),
            "wrote_to": doc.get("wrote_to"),
            "wrote_where": doc.get("wrote_where"),
        },
        source=origin,
    )




def field_current_specimen() -> Dict[str, Any]:
    sel, sel_src = load_rel_json("receipts/headless/MODEL_2_SELECTION.json")
    lake = Path("/Volumes/corpdrive/hawking-modellake/specimens")
    # Finder drops .DS_Store into any directory it displays; a dotfile is not a specimen.
    resident = sorted(p.name for p in lake.iterdir()
                      if p.is_dir() and not p.name.startswith(".")) if lake.is_dir() else []
    if sel is None and not resident:
        return absent("no MODEL_2_SELECTION.json and no resident specimens on the model lake")
    return measured(
        {
            "principal_specimen": dig(sel, "recommendation", default=None),
            "resident_in_model_lake": resident,
            "lake_root": str(lake.parent),
            "stale_on_disk_flags_found": dig(sel, "stale_flags", default=[]),
        },
        source=[x for x in (sel_src, str(lake)) if x],
    )


def field_odyssey_queue() -> Dict[str, Any]:
    q, q_src = load_rel_json("receipts/headless/ODYSSEY_QUEUE_RECOVERED.json")
    if q is None:
        return absent("receipts/headless/ODYSSEY_QUEUE_RECOVERED.json not on disk")
    rows = q.get("queue") or []
    return measured(
        {
            "n_patients": len(rows),
            "patients": [{"oxx": r.get("oxx"), "model": r.get("model"),
                          "class": r.get("class"),
                          "canonical_source": r.get("canonical_source"),
                          "canonical_revision": r.get("canonical_revision")}
                         for r in rows],
            "on_disk_flags_are_stale": True,
            "measure_presence_with": "tools/odyssey/model2_select.py (stats the real paths)",
        },
        source=q_src,
    )


def field_doctor_state() -> Dict[str, Any]:
    lib, lib_src = load_rel_json("receipts/headless/DOCTOR_TECHNIQUE_LIBRARY.json")
    tr, tr_src = load_rel_json("receipts/headless/DOCTOR_TRANSFER.json")
    if lib is None:
        return absent("receipts/headless/DOCTOR_TECHNIQUE_LIBRARY.json not on disk")
    techs = lib.get("techniques") or []
    return measured(
        {
            "n_techniques": len(techs),
            "all_KEEP": all(t.get("decision") == "KEEP" for t in techs),
            "pruning_law": "a single model's failure never prunes a technique",
            "last_prescription_run": None if tr is None else {
                "specimen": dig(tr, "specimen", default=None),
                "n_organs": tr.get("n_organs"),
                "experiments_prescribed": tr.get("distinct_experiments_prescribed"),
                "search_space_reduction":
                    dig(tr, "prescription_quality", "search_space_reduction", "value"),
            },
        },
        source=[x for x in (lib_src, tr_src) if x],
    )


def field_libraries() -> Dict[str, Any]:
    out, srcs = {}, []
    for name, rel in (
        ("organ_frontier_matrix", "receipts/headless/ORGAN_FRONTIER_MATRIX.json"),
        ("representation_library", "receipts/headless/REPRESENTATION_LIBRARY.json"),
        ("kernel_library", "receipts/headless/KERNEL_LIBRARY.json"),
        ("superoperator_library", "receipts/headless/SUPEROPERATOR_LIBRARY.json"),
        ("transfer_report", "receipts/headless/QWEN_TRANSFER_REPORT.json"),
        ("cross_model_laws", "receipts/headless/CROSS_MODEL_LAWS.json"),
    ):
        doc, src = load_rel_json(rel)
        if doc is None:
            out[name] = {"present": False}
            continue
        srcs.append(src)
        out[name] = {
            "present": True, "schema": doc.get("schema"),
            "counts": {k: doc[k] for k in
                       ("n_models", "n_measured", "n_families", "n_kernels", "n_complete",
                        "n_operators", "n_entries", "n_laws", "counts")
                       if k in doc},
        }
    if not srcs:
        return absent("no canonical library receipts on disk")
    return measured(out, source=srcs)


BUILDERS = {
    "git": field_git,
    "ledgers": field_ledgers,
    "leading_noetic_executable": field_leading_noetic,
    "kernels_runtime_machine": field_kernels_runtime_machine,
    "gpu_roof": field_gpu_roof,
    "current_bottleneck": field_current_bottleneck,
    "concurrency_equilibrium": field_concurrency_equilibrium,
    "artifact_store": field_artifact_store,
    "git_storage_policy": field_git_storage_policy,
    "active_grok_tasks": field_active_grok_tasks,
    "next_workunits": field_next_workunits,
    "negative_science": field_negative_science,
    "current_specimen": field_current_specimen,
    "odyssey_queue": field_odyssey_queue,
    "doctor_state": field_doctor_state,
    "libraries": field_libraries,
}


def build() -> Dict[str, Any]:
    doc: Dict[str, Any] = {"schema": SCHEMA}
    for key in REQUIRED_KEYS:
        doc[key] = BUILDERS[key]()
    return doc


def assemble_and_write(path: Path = RECEIPT) -> Dict[str, Any]:
    doc = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_dumps(doc))
    return doc


def main() -> int:
    doc = assemble_and_write()
    sys.stdout.write(canonical_dumps(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
