#!/usr/bin/env python3
"""Run a bounded, disposable ModelLake-organization bake-off.

This is a qualification harness, not an Odyssey writer.  Every model gets a
fresh temporary Git workspace containing a tiny synthetic lake fixture.  The
model must use the safe Hawking read/write/status/diff surface, and the
central grader scores the observed tool trace plus the returned proposal.  No
model output or temporary file is persisted after a lane finishes.

The paid ceiling is deliberately split into ten independent lane ceilings so
the aggregate remains below Hawking's configured one-dollar mission ceiling.
The caller should still provide the current Hawking Keychain resolver
variables; OPENROUTER_API_KEY remains the provider's higher-priority optional
override through the existing resolver path.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hawking.remote_cognition import (  # noqa: E402
    OpenRouterProvider,
    RemoteCostPolicy,
    _keychain_resolver_from_environment,
)
from hawking.tool_registry import (  # noqa: E402
    READ_ONLY,
    REVERSIBLE_REPO,
    default_tool_registry,
)


# This is the user's ten-model list.  The separately-mentioned MiMo wildcard
# is held out so this run is exactly ten information-collection lanes.
FRONTIER_MODELS: Tuple[str, ...] = (
    "z-ai/glm-5.3",
    "moonshotai/kimi-k3",
    "z-ai/glm-5.3-flash",
    "deepseek/deepseek-v4.1-flash",
    "qwen/qwen3.8-flash",
    "qwen/qwen3.8-2.4t-a95b",
    "qwen/qwen3.8-27b",
    "minimax/minimax-m3",
    "inclusionai/ling-3.0-flash",
    "nvidia/nemotron-3-ultra-550b-a55b",
)

REQUIRED_TOOLS: Tuple[str, ...] = (
    "tools.catalog",
    "fs.list",
    "fs.search",
    "fs.read",
    "filesystem.write",
    "git.status",
    "git.diff",
)
SAFE_PERMISSIONS = frozenset({READ_ONLY, REVERSIBLE_REPO})
MAX_TOOL_CALLS = len(REQUIRED_TOOLS)
MAX_TOKENS = 320
LANE_COST_CEILING_USD = 0.09
AGGREGATE_COST_CEILING_USD = LANE_COST_CEILING_USD * len(FRONTIER_MODELS)
QUALITY_SIGNALS: Tuple[str, ...] = (
    "canonical",
    "sealed",
    "incoming",
    "dedup",
    "duplicate",
    "revision",
    "provenance",
    "owner",
    "manifest",
    "index",
    "lineage",
    "receipt",
    "idempotent",
    "quarantine",
    "promotion",
    "validation",
    "integrity",
)


def _slug(model_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", model_id).strip("-").lower()
    return value[:80]


def _write_fixture(root: Path, model_id: str, slug: str) -> str:
    """Seed a deliberately small organization problem and stage its baseline."""
    scratch = f"scratch/{slug}.proposal.md"
    files: Dict[str, str] = {
        "README.md": (
            "# Disposable ModelLake qualification\n\n"
            "This repository is synthetic and disposable. The organization debt is intentional:\n"
            "the same logical candidate appears in incoming and catalog records, owners are not\n"
            "uniform, and the sealed disposition is the only immutable identity source.\n"
        ),
        "model_lake/README.md": (
            "# ModelLake intake\n\n"
            "Flow under review: incoming -> canonical index -> sealed disposition -> Odyssey board.\n"
            "Incoming data may be incomplete. A candidate must retain model id, revision, owner,\n"
            "provenance, integrity status, and an explicit disposition before promotion.\n"
        ),
        "model_lake/incoming/catalog.json": json.dumps(
            {
                "source": "synthetic-openrouter-observation",
                "candidates": [
                    {
                        "model_id": "z-ai/glm-5.3-flash",
                        "revision": "incoming-r1",
                        "owner": None,
                        "status": "incoming",
                    },
                    {
                        "model_id": "deepseek/deepseek-v4.1-flash",
                        "revision": "incoming-r1",
                        "owner": "unregistered",
                        "status": "incoming",
                    },
                    {
                        "model_id": "z-ai/glm-5.3-flash",
                        "revision": "incoming-r1",
                        "owner": "unregistered",
                        "status": "incoming",
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        "model_lake/index/canonical.json": json.dumps(
            {
                "model_id": "z-ai/glm-5.3-flash",
                "revision": "canonical-r0",
                "source_refs": ["model_lake/incoming/catalog.json#0"],
                "disposition": "PENDING",
            },
            indent=2,
        )
        + "\n",
        "receipts/sealed-disposition.json": json.dumps(
            {
                "model_id": "z-ai/glm-5.3-flash",
                "revision": "sealed-r7",
                "disposition": "RESEARCH SPECIMEN",
                "owner": "sealed-authority",
                "provenance": "synthetic-sealed-receipt",
                "integrity": "unknown-until-verified",
                "immutable": True,
            },
            indent=2,
        )
        + "\n",
        "workspace/campaign/odyssey/candidate-board.json": json.dumps(
            {
                "schema": "synthetic.candidate-board.v1",
                "candidates": [
                    {
                        "model_id": "z-ai/glm-5.3-flash",
                        "active_owner": "unregistered",
                        "tier0": "PENDING",
                    },
                    {
                        "model_id": "deepseek/deepseek-v4.1-flash",
                        "active_owner": "unregistered",
                        "tier0": "PENDING",
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        scratch: "# Proposal placeholder\nThis tracked file is intentionally overwritten by the lane.\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    # Stage the baseline without creating a commit.  This gives the
    # read-only git.diff tool a precise before/after comparison while avoiding
    # any mutation of the user's configured Git identity or main repository.
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "-A"], cwd=str(root), check=True, capture_output=True, text=True)
    return scratch


def _parse_json_text(value: Any) -> Dict[str, Any]:
    if not isinstance(value, str):
        return {}
    candidate = value.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        parsed = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _trace_value(row: Mapping[str, Any]) -> Mapping[str, Any]:
    result = row.get("result")
    if not isinstance(result, Mapping):
        return {}
    value = result.get("value")
    return value if isinstance(value, Mapping) else {}


def _path_is_inside(root: Path, raw: Any) -> bool:
    if not isinstance(raw, str) or not raw.strip():
        return True
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        candidate.resolve(strict=False).relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _grade(
    *,
    root: Path,
    model_id: str,
    slug: str,
    scratch: str,
    result: Optional[Mapping[str, Any]],
    error_code: Optional[str],
    wall_s: float,
) -> Dict[str, Any]:
    result_map = result if isinstance(result, Mapping) else {}
    trace = result_map.get("tool_trace")
    trace_rows = [row for row in trace if isinstance(row, Mapping)] if isinstance(trace, list) else []
    successful = {
        str(row.get("tool"))
        for row in trace_rows
        if row.get("dispatched") is True and row.get("ok") is True
    }
    attempted = {str(row.get("tool")) for row in trace_rows if row.get("tool")}
    missing = [name for name in REQUIRED_TOOLS if name not in successful]
    forbidden_attempts = sorted(attempted.difference(REQUIRED_TOOLS))
    path_violations: List[str] = []
    for row in trace_rows:
        arguments = row.get("arguments")
        if not isinstance(arguments, Mapping):
            continue
        for key in ("path", "root"):
            if key in arguments and not _path_is_inside(root, arguments.get(key)):
                path_violations.append(f"{row.get('tool')}:{key}")

    scratch_path = root / scratch
    scratch_changed = False
    if scratch_path.is_file():
        scratch_changed = scratch_path.read_text(encoding="utf-8", errors="replace") != (
            "# Proposal placeholder\nThis tracked file is intentionally overwritten by the lane.\n"
        )
    diff_has_scratch = False
    status_observed = False
    diff_observed = False
    for row in trace_rows:
        tool = str(row.get("tool") or "")
        value = _trace_value(row)
        if tool == "git.status" and row.get("ok") is True:
            status_observed = True
        if tool == "git.diff" and row.get("ok") is True:
            diff_observed = True
            diff_has_scratch = scratch in str(value.get("stdout") or "")

    content = str(result_map.get("content") or "")
    structured = _parse_json_text(content)
    quality_text = json.dumps(
        {
            "content": structured or content,
            "observations": result_map.get("observations"),
            "operations": result_map.get("operations"),
            "tests": result_map.get("tests"),
            "questions": result_map.get("questions"),
        },
        sort_keys=True,
        ensure_ascii=False,
    ).lower()
    quality_hits = [signal for signal in QUALITY_SIGNALS if signal in quality_text]

    coverage_points = round(
        50.0 * (len(REQUIRED_TOOLS) - len(missing)) / len(REQUIRED_TOOLS),
        2,
    )
    grounded_points = 0.0
    if scratch_changed:
        grounded_points += 10.0
    if status_observed:
        grounded_points += 4.0
    if diff_observed and diff_has_scratch:
        grounded_points += 6.0
    quality_points = min(20.0, float(len(quality_hits) * 1.25))
    if structured:
        quality_points = min(20.0, quality_points + 2.0)
    if isinstance(structured.get("operations"), list) and structured.get("operations"):
        quality_points = min(20.0, quality_points + 1.0)
    safety_points = 10.0 if not forbidden_attempts and not path_violations else 0.0
    score = round(coverage_points + grounded_points + quality_points + safety_points, 2)

    usage = result_map.get("usage")
    cost = None
    if isinstance(usage, Mapping):
        try:
            cost = float(usage.get("cost_usd")) if usage.get("cost_usd") is not None else None
        except (TypeError, ValueError):
            cost = None
    return {
        "model": model_id,
        "status": "completed" if result_map and not error_code else "failed",
        "error_code": error_code,
        "score": score,
        "tool_calls": len(trace_rows),
        "successful_tools": sorted(successful),
        "missing_tools": missing,
        "forbidden_attempts": forbidden_attempts,
        "path_violations": path_violations,
        "quality_signals": quality_hits,
        "scratch_changed": scratch_changed,
        "git_status_observed": status_observed,
        "git_diff_observed": diff_observed,
        "git_diff_grounded": diff_has_scratch,
        "cost_usd": cost,
        "wall_s": round(wall_s, 3),
        "output_discarded": True,
    }


def _run_lane(model_id: str, resolver: Any) -> Dict[str, Any]:
    started = time.monotonic()
    slug = _slug(model_id)
    try:
        with tempfile.TemporaryDirectory(prefix=f"hawking-bakeoff-{slug}-") as temporary:
            root = Path(temporary).resolve()
            scratch = _write_fixture(root, model_id, slug)
            registry = default_tool_registry(
                root,
                repo_root=root,
                permissions=SAFE_PERMISSIONS,
            )
            prompt = f"""You are the {model_id} lane in a disposable Hawking ModelLake organization bake-off.

Work only inside the temporary repository shown by the tool contract. Do not use network, model
downloads, shells, native runtimes, Flash/Pulsar, HCLI, AgentOS, HIDE, Grok, or any tool not
explicitly offered. The result is thrown away after grading; do not claim a production mutation.

Make a concise, concrete proposal for organizing the synthetic ModelLake flow so duplicate
incoming records cannot silently replace a sealed identity and Odyssey ownership remains explicit.
The proposal should distinguish incoming, canonical, sealed, and board state, preserve revision and
provenance, and say how validation/recovery would work.

You MUST actually call each of these seven tools, one call at a time, in this order:
1. tools.catalog with focus "safe ModelLake organization read write git tools".
2. fs.list on model_lake with recursive=true.
3. fs.search for "revision" from root "." with glob "**/*.json".
4. fs.read of model_lake/README.md.
5. filesystem.write to exactly {scratch}, with a short proposal and overwrite=true.
6. git.status on path ".".
7. git.diff on path ".".

After the tool sequence, return one JSON object with exactly these top-level fields:
observations (array), operations (array of proposed organization actions), tests (array),
questions (array), status (string), content (string). Keep it under 250 words and make content
refer to evidence actually seen in the fixture and the diff."""
            workunit = SimpleNamespace(
                id=f"frontier-bakeoff-{slug}",
                role="frontier_model_lake_bakeoff",
                description="Disposable ModelLake organization proposal qualification.",
            )
            provider = OpenRouterProvider(
                model_id,
                workspace=root,
                cost_policy=RemoteCostPolicy(
                    max_cost_usd=LANE_COST_CEILING_USD,
                    mission_limit_usd=LANE_COST_CEILING_USD,
                    monthly_limit_usd=9.99,
                ),
                key_resolver=resolver,
                timeout=90.0,
                max_attempts=2,
                backoff_s=0.5,
                max_backoff_s=2.0,
                remote_slot=f"frontier-bakeoff:{slug}",
            )
            result = provider.execute_workunit(
                workunit,
                {
                    "prompt": prompt,
                    "tool_registry": registry,
                    "hawking_tools": True,
                    "tool_write_authority": True,
                    "mutation_lock_held": True,
                    "allowed_tool_names": list(REQUIRED_TOOLS),
                    "max_tool_calls": MAX_TOOL_CALLS,
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.1,
                    "timeout": 90.0,
                    "estimated_cost_usd": 0.01,
                    "remote_cost_authorization": {"max_usd": LANE_COST_CEILING_USD},
                },
            )
            return _grade(
                root=root,
                model_id=model_id,
                slug=slug,
                scratch=scratch,
                result=result,
                error_code=None,
                wall_s=time.monotonic() - started,
            )
    except Exception as exc:  # each lane is independent; the bake-off continues
        code = getattr(exc, "code", None) or type(exc).__name__
        return {
            "model": model_id,
            "status": "failed",
            "error_code": str(code),
            "score": 0.0,
            "tool_calls": 0,
            "successful_tools": [],
            "missing_tools": list(REQUIRED_TOOLS),
            "forbidden_attempts": [],
            "path_violations": [],
            "quality_signals": [],
            "scratch_changed": False,
            "git_status_observed": False,
            "git_diff_observed": False,
            "git_diff_grounded": False,
            "cost_usd": None,
            "wall_s": round(time.monotonic() - started, 3),
            "output_discarded": True,
        }


def _catalog_row(models: Iterable[Mapping[str, Any]], model_id: str) -> Optional[Dict[str, Any]]:
    for row in models:
        if isinstance(row, Mapping) and str(row.get("id") or "") == model_id:
            return dict(row)
    return None


def _price(row: Mapping[str, Any], key: str) -> Optional[float]:
    pricing = row.get("pricing")
    if not isinstance(pricing, Mapping):
        return None
    try:
        value = pricing.get(key)
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-usd", type=float, default=9.99)
    args = parser.parse_args(argv)
    budget = max(0.0, float(args.budget_usd))
    if budget < AGGREGATE_COST_CEILING_USD:
        print(json.dumps({
            "status": "refused",
            "reason": "budget below aggregate Hawking lane ceiling",
            "budget_usd": budget,
            "aggregate_lane_ceiling_usd": AGGREGATE_COST_CEILING_USD,
        }, sort_keys=True))
        return 2

    resolver = _keychain_resolver_from_environment()
    key_present = False
    if callable(resolver):
        try:
            key_present = bool(resolver())
        except Exception:
            key_present = False
    # Deliberately print only presence, never the credential or an auth header.
    print(f"OPENROUTER_KEY_PRESENT={'PASS' if key_present else 'FAIL'}", flush=True)
    if not key_present:
        return 2

    catalog_provider = OpenRouterProvider(
        "__catalog__",
        key_resolver=resolver,
        timeout=90.0,
        max_attempts=2,
        backoff_s=0.5,
        max_backoff_s=2.0,
        catalog_ttl_s=0.0,
    )
    try:
        live_catalog = catalog_provider.list_models(force=True, timeout=90.0)
    except Exception as exc:
        print(json.dumps({
            "status": "catalog_failed",
            "error_code": str(getattr(exc, "code", None) or type(exc).__name__),
            "output_discarded": True,
        }, sort_keys=True))
        return 1

    rows: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    live_pricing: Dict[str, Dict[str, Optional[float]]] = {}
    for model_id in FRONTIER_MODELS:
        row = _catalog_row(live_catalog, model_id)
        if row is None:
            missing.append(model_id)
            continue
        rows[model_id] = row
        live_pricing[model_id] = {
            "prompt": _price(row, "prompt"),
            "completion": _price(row, "completion"),
        }
    if missing:
        print(json.dumps({
            "status": "catalog_incomplete",
            "catalog_count": len(live_catalog),
            "missing_models": missing,
            "output_discarded": True,
        }, sort_keys=True))
        return 1

    print(json.dumps({
        "status": "launching_parallel",
        "catalog_count": len(live_catalog),
        "model_count": len(FRONTIER_MODELS),
        "models": list(FRONTIER_MODELS),
        "parallel_workers": len(FRONTIER_MODELS),
        "budget_usd": budget,
        "aggregate_lane_ceiling_usd": AGGREGATE_COST_CEILING_USD,
        "live_pricing_per_token": live_pricing,
        "held_out_wildcard": "xiaomi/mimo-v2.5",
    }, sort_keys=True), flush=True)

    results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(FRONTIER_MODELS),
        thread_name_prefix="hawking-frontier-bakeoff",
    ) as pool:
        futures = [pool.submit(_run_lane, model_id, resolver) for model_id in FRONTIER_MODELS]
        for future in futures:
            results.append(future.result())

    ranked = sorted(results, key=lambda row: (-float(row.get("score") or 0.0), str(row.get("model") or "")))
    total_cost = sum(float(row["cost_usd"]) for row in results if row.get("cost_usd") is not None)
    print(json.dumps({
        "status": "complete",
        "ranking": ranked,
        "total_observed_cost_usd": round(total_cost, 8),
        "aggregate_lane_ceiling_usd": AGGREGATE_COST_CEILING_USD,
        "budget_usd": budget,
        "all_outputs_discarded": all(row.get("output_discarded") is True for row in results),
        "authoritative_tree_changed": False,
        "sealed_receipts_changed": False,
        "safe_tool_surface": list(REQUIRED_TOOLS),
        "unsafe_or_external_tools_exposed": False,
    }, sort_keys=True), flush=True)
    return 0 if all(row.get("status") == "completed" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
