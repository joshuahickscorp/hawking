#!/usr/bin/env python3
"""Derive HCLI's effective per-request context budget from live evidence.

This is the reproduction for the regression

    request (23532 tokens) exceeds the available context size (11008 tokens)

recorded at ``.hcli/receipts/887c6271-d82d-4d30-847e-1e02497ca8e6.json``.

It is a read-only diagnostic. It never spawns a runtime and never mutates
repository state. It walks the same resolution order the live code walks and
prints the arithmetic at every hop, so the failing number can be regenerated
from disk instead of remembered.

    python3 tools/headless/context_budget_probe.py                # human
    python3 tools/headless/context_budget_probe.py --json         # receipt

Exit status is 0 when the probe completed, 2 when the derived per-request
budget cannot hold the root mission demand (the condition that produced the
regression).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

# llama.cpp allocates the KV cache per sequence and rounds the per-sequence
# context up to this granularity. Verified against llama-server build 9430
# (d48a56eff) by exhaustion: of every plausible slot count, only n_parallel=3
# maps a configured 32768 onto the observed 11008.
LLAMA_KV_PAD = 256

# The observed failure, kept as data so the probe can assert against it.
OBSERVED_FAILURE = {
    "receipt": ".hcli/receipts/887c6271-d82d-4d30-847e-1e02497ca8e6.json",
    "n_prompt_tokens": 23532,
    "n_ctx": 11008,
}


def per_seq_context(total_ctx: int, n_parallel: int, pad: int = LLAMA_KV_PAD) -> int:
    """llama.cpp's per-sequence context for a given --ctx-size / --parallel."""
    if n_parallel < 1:
        raise ValueError("n_parallel must be >= 1")
    return int(math.ceil(total_ctx / n_parallel / pad) * pad)


def solve_parallel(total_ctx: int, observed_per_seq: int, limit: int = 64) -> Optional[int]:
    """Recover the slot count that maps total_ctx onto observed_per_seq."""
    hits = [n for n in range(1, limit + 1) if per_seq_context(total_ctx, n) == observed_per_seq]
    if len(hits) == 1:
        return hits[0]
    return None


def resolve_runtime_count() -> Dict[str, Any]:
    """Ask the live code, not a copy of it."""
    try:
        from hcli.cli import resolve_resident_runtime_limit  # type: ignore
    except Exception as exc:  # pragma: no cover - import failure is the finding
        return {"count": None, "source": f"import failed: {exc!r}"}
    count, source = resolve_resident_runtime_limit(str(REPO_ROOT))
    return {"count": int(count), "source": source}


def resolve_configured_ctx(n_parallel: int = 1) -> Dict[str, Any]:
    """The context size HCLI would hand llama-server today.

    Once the canonical authority exists, ask it. The source-scrape below is the
    pre-authority path and is kept only so this probe still reproduces the
    original regression on a tree that predates the fix.
    """
    try:
        from hcli import context_budget  # type: ignore

        b = context_budget.resolve(n_parallel=n_parallel)
        return {
            "ctx_size": b.total_ctx,
            "source": f"context_budget.resolve -> {b.source}",
            "per_request_ctx": b.per_request_ctx,
            "model_ceiling": b.model_ceiling,
            "authority": True,
        }
    except Exception:
        pass
    env = os.environ.get("HCLI_CTX_SIZE")
    if env:
        return {"ctx_size": int(env), "source": "env:HCLI_CTX_SIZE"}
    # Mirror of runtime.RuntimePool.__init__. Read the literal out of the live
    # source rather than restating it, so this probe cannot drift from it.
    runtime_py = REPO_ROOT / "hcli" / "runtime.py"
    for lineno, line in enumerate(runtime_py.read_text(encoding="utf-8").splitlines(), 1):
        if "HCLI_CTX_SIZE" in line and "environ" in line:
            digits = "".join(c for c in line.split('"')[-2] if c.isdigit())
            if digits:
                return {
                    "ctx_size": int(digits),
                    "source": f"hcli/runtime.py:{lineno} hardcoded default",
                }
    return {"ctx_size": None, "source": "not found in runtime.py"}


def resolve_topology() -> Dict[str, Any]:
    """Ask machine.resolve_decode_topology, which takes the repo root."""
    try:
        from hcli.machine import resolve_decode_topology  # type: ignore
    except Exception as exc:
        return {"topology": None, "source": f"import failed: {exc!r}"}
    topology, source = resolve_decode_topology(REPO_ROOT)
    return {"topology": topology, "source": source}


def probe_live_server(port: int = 8080, timeout: float = 3.0) -> Dict[str, Any]:
    """Read the real capability of an already-running llama-server."""
    url = f"http://127.0.0.1:{port}/props"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"reachable": False, "url": url, "error": repr(exc)}
    gen = data.get("default_generation_settings") or {}
    return {
        "reachable": True,
        "url": url,
        "model_path": data.get("model_path"),
        "total_slots": data.get("total_slots"),
        "per_slot_n_ctx": gen.get("n_ctx"),
    }


def build_report(root_prompt_tokens: int, generation_reserve: int) -> Dict[str, Any]:
    rc = resolve_runtime_count()
    topo = resolve_topology()
    _np = rc["count"] if topo.get("topology") == "slot" else 1
    cfg = resolve_configured_ctx(n_parallel=_np or 1)
    live = probe_live_server()

    total_ctx = cfg["ctx_size"]
    n_parallel = rc["count"] if topo.get("topology") == "slot" else 1
    usable = cfg.get("per_request_ctx")
    if usable is None:
        usable = (
            per_seq_context(total_ctx, n_parallel)
            if total_ctx and n_parallel
            else None
        )

    recovered = (
        solve_parallel(total_ctx, OBSERVED_FAILURE["n_ctx"])
        if total_ctx
        else None
    )

    demand = root_prompt_tokens + generation_reserve
    fits = usable is not None and demand <= usable

    return {
        "probe": "context_budget_probe",
        "repo": str(REPO_ROOT),
        "hops": {
            "1_runtime_count": rc,
            "2_configured_ctx_size": cfg,
            "3_decode_topology": topo,
            "4_llama_parallel": n_parallel,
            "5_per_request_usable_ctx": usable,
        },
        "arithmetic": {
            "rule": "per_seq = ceil(ctx_size / n_parallel / %d) * %d" % (LLAMA_KV_PAD, LLAMA_KV_PAD),
            "ctx_size": total_ctx,
            "n_parallel": n_parallel,
            "per_seq": usable,
        },
        "observed_failure": dict(
            OBSERVED_FAILURE,
            recovered_n_parallel=recovered,
            reproduced=usable == OBSERVED_FAILURE["n_ctx"],
        ),
        "live_runtime": live,
        "authority_in_use": bool(cfg.get("authority")),
        "capability_gap": {
            # What the hardware/model can actually do vs what HCLI assumes.
            "live_server_per_slot_n_ctx": live.get("per_slot_n_ctx"),
            "hcli_assumed_total_ctx": total_ctx,
            "hcli_reads_live_server": False,
            "note": (
                "No attach/discovery path exists: grep for '/props' in "
                "hcli/{runtime,backends}.py returns nothing, so "
                "HCLI always spawns its own server at its own default and "
                "never learns the real ceiling of one already running."
            ),
        },
        "preflight": {
            "root_prompt_tokens": root_prompt_tokens,
            "generation_reserve": generation_reserve,
            "demand": demand,
            "usable": usable,
            "fits": fits,
        },
    }


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit the receipt as JSON")
    ap.add_argument(
        "--root-prompt-tokens",
        type=int,
        default=OBSERVED_FAILURE["n_prompt_tokens"],
        help="root mission prompt size to preflight (default: the observed failure)",
    )
    ap.add_argument(
        "--generation-reserve",
        type=int,
        default=0,
        help="tokens reserved for generation/tools/safety",
    )
    args = ap.parse_args(argv)

    rep = build_report(args.root_prompt_tokens, args.generation_reserve)

    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        h = rep["hops"]
        print("HCLI context budget probe")
        print(f"  repo                  {rep['repo']}")
        print(f"  runtime_count         {h['1_runtime_count']['count']}  <- {h['1_runtime_count']['source']}")
        print(f"  configured ctx_size   {h['2_configured_ctx_size']['ctx_size']}  <- {h['2_configured_ctx_size']['source']}")
        print(f"  decode topology       {h['3_decode_topology']['topology']}  <- {h['3_decode_topology']['source']}")
        print(f"  llama --parallel      {h['4_llama_parallel']}")
        print(f"  per-request usable    {h['5_per_request_usable_ctx']}   ({rep['arithmetic']['rule']})")
        obs = rep["observed_failure"]
        print(f"  reproduces {obs['n_ctx']}?  {obs['reproduced']}  (recovered n_parallel={obs['recovered_n_parallel']})")
        lr = rep["live_runtime"]
        if lr.get("reachable"):
            print(f"  live server           per-slot n_ctx={lr['per_slot_n_ctx']} slots={lr['total_slots']}")
            print("  HCLI reads it?        NO (no /props discovery path)")
        pf = rep["preflight"]
        print(f"  preflight             demand={pf['demand']} usable={pf['usable']} fits={pf['fits']}")

    return 0 if rep["preflight"]["fits"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
