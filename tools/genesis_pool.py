#!/usr/bin/env python3
"""Drive the Genesis Qwen3.8 child pool, or measure how many children fit.

    tools/genesis_pool.py spawn PROMPT BUDGET [--timing] [--seq-len N]
    tools/genesis_pool.py poll  CHILD_ID
    tools/genesis_pool.py kill  CHILD_ID
    tools/genesis_pool.py measure [--counts 1,2,4,8] [--seq-lens 128,2048,8192]
    tools/genesis_pool.py e2e [--n 4] [--max-new-tokens 8] [--seq-len 128]
    tools/genesis_pool.py stub-child [same flags as the native binary]

TEXT tasks omit --timing and do not take the GPU lane lock. TIMING tasks
pass --timing so gpu_lane_lock.sh serializes them. Workload A (gravity
recipe) is TEXT. Workload B (kernel floor) is TIMING; the pool's value
there is starting candidate N+1's setup (a TEXT/no-lock spawn such as a
compile) while candidate N holds the lock.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lab.genesis_pool import (  # noqa: E402
    BINARY_NAME,
    MEASURED_SAFE_N,
    AdmissionRefused,
    ChildBudget,
    GenesisPool,
    PoolConfig,
    discover_artifact,
    discover_binary,
    discover_tokenizer,
    kv_bytes_estimate,
    live_ready,
    pgrep_binary_pids,
    process_liveness,
    recommended_safe_n,
    sample_children,
    sample_machine,
    stub_child_main,
)


def _pool_from_args(args: argparse.Namespace) -> GenesisPool:
    live = live_ready()
    binary = Path(args.binary) if getattr(args, "binary", None) else None
    artifact = Path(args.artifact_root) if getattr(args, "artifact_root", None) else None
    tokenizer = Path(args.tokenizer) if getattr(args, "tokenizer", None) else None
    if binary is None and live:
        binary = live[0]
    if artifact is None and live:
        artifact = live[1]
    if tokenizer is None and live:
        tokenizer = live[2]
    if binary is None:
        binary = Path(__file__)
    output = Path(args.output_root) if getattr(args, "output_root", None) else (
        REPO / "workspace/ops/local/genesis-pool"
    )
    safe_n = int(getattr(args, "safe_n", None) or recommended_safe_n(
        int(getattr(args, "seq_len", 128) or 128)
    ))
    cfg = PoolConfig(
        binary=binary,
        artifact_root=artifact,
        tokenizer=tokenizer,
        output_root=output,
        safe_n=safe_n,
        max_seq_len=int(getattr(args, "seq_len", 128) or 128),
        min_free_bytes=int(getattr(args, "min_free_bytes", 0) or 0),
    )
    return GenesisPool(cfg)


def cmd_spawn(args: argparse.Namespace) -> int:
    pool = _pool_from_args(args)
    budget = ChildBudget(
        max_new_tokens=int(args.budget),
        max_seq_len=int(args.seq_len),
    )
    try:
        child_id = pool.spawn(
            args.prompt,
            budget,
            hold_gpu_lock=bool(args.timing),
        )
    except AdmissionRefused as exc:
        print(json.dumps({"state": "refused", "reason": str(exc), "safe_n": exc.safe_n, "alive": exc.alive}))
        return 3
    rec = pool._children[child_id]
    print(json.dumps({"child_id": child_id, "pid": rec.pid, "output_dir": rec.output_dir}))
    return 0


def cmd_poll(args: argparse.Namespace) -> int:
    pool = _pool_from_args(args)
    st = pool.poll(args.child_id)
    print(json.dumps(st.to_dict()))
    return 0 if st.state != "failed" else 1


def cmd_kill(args: argparse.Namespace) -> int:
    pool = _pool_from_args(args)
    st = pool.kill(args.child_id)
    print(json.dumps(st.to_dict()))
    return 0


def _refuse_if_stray_children() -> list[int]:
    pids = pgrep_binary_pids()
    return pids


def cmd_measure(args: argparse.Namespace) -> int:
    """Spawn 1/2/4/8 on the real binary and record footprint + wall.

    Will not launch a count whose estimated residency would force swap.
    N=8 is expected to be refused from the N=4 saturation evidence.
    """
    live = live_ready()
    if live is None:
        print("measure: binary/artifact/tokenizer not found", file=sys.stderr)
        return 2
    binary, artifact, tokenizer = live
    stray = _refuse_if_stray_children()
    if stray and not args.ignore_strays:
        print(
            json.dumps({"state": "blocked", "reason": "stray_children", "pids": stray}),
            file=sys.stderr,
        )
        return 4
    counts = [int(x) for x in args.counts.split(",") if x.strip()]
    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    tokens = int(args.max_new_tokens)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    baseline_machine = sample_machine()
    points: list[dict] = []
    for seq in seq_lens:
        for n in counts:
            point = _measure_one(
                n=n,
                seq_len=seq,
                tokens=tokens,
                binary=binary,
                artifact=artifact,
                tokenizer=tokenizer,
                output_root=Path(args.output_root),
                prompt=args.prompt,
                allow_oversub=bool(args.force),
            )
            points.append(point)
            print(json.dumps({"event": "point", **{k: point[k] for k in ("n", "seq_len", "status")}}), flush=True)
            # settle so the next point does not inherit Metal residency
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and pgrep_binary_pids():
                time.sleep(0.5)
            time.sleep(1.0)
    body = {
        "schema": "hawking.ascent.genesis_children_capacity.v1",
        "date": time.strftime("%Y-%m-%d"),
        "label": "MEASURED",
        "binary": str(binary),
        "artifact": str(artifact),
        "tokenizer": str(tokenizer),
        "baseline_machine": baseline_machine,
        "kv_bytes_per_position_estimate": kv_bytes_estimate(1),
        "safe_n_measured": MEASURED_SAFE_N,
        "points": points,
    }
    out.write_text(json.dumps(body, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


def _measure_one(
    *,
    n: int,
    seq_len: int,
    tokens: int,
    binary: Path,
    artifact: Path,
    tokenizer: Path,
    output_root: Path,
    prompt: str,
    allow_oversub: bool,
) -> dict:
    machine0 = sample_machine()
    # Hard refuse: N=4 already swapped at seq-len 2048. Anything above the
    # measured safe N is a lie factory unless the caller forces it.
    if n > MEASURED_SAFE_N and not allow_oversub:
        return {
            "n": n,
            "seq_len": seq_len,
            "max_new_tokens": tokens,
            "status": "refused_would_swap",
            "saturated_resource": "unified_memory_metal_private_ioaccelerator",
            "reason": (
                f"N={n} exceeds measured safe_n={MEASURED_SAFE_N}. "
                "N=4 at seq-len 2048 already used 751 MB swap with 0.81 GB free "
                "and 14.73 GB private Metal buffers per child."
            ),
            "machine_before": machine0,
        }
    cfg = PoolConfig(
        binary=binary,
        artifact_root=artifact,
        tokenizer=tokenizer,
        output_root=output_root / f"n{n}_seq{seq_len}_{int(time.time())}",
        safe_n=n,
        max_seq_len=seq_len,
        min_free_bytes=0,
    )
    pool = GenesisPool(cfg)
    ids: list[str] = []
    t0 = time.monotonic_ns()
    try:
        for i in range(n):
            ids.append(
                pool.spawn(
                    f"{prompt} [{i}]",
                    ChildBudget(max_new_tokens=tokens, max_seq_len=seq_len),
                    hold_gpu_lock=False,
                )
            )
        # Sample once children exist; keep sampling until footprint appears or done.
        sample = None
        results = []
        pending = set(ids)
        peak_sample = None
        while pending:
            pids = [pool._children[c].pid for c in ids if process_liveness(pool._children[c].pid) == "running"]
            if pids:
                sample = sample_children(pids)
                if peak_sample is None or sample["sum_phys_footprint_bytes"] >= peak_sample["sum_phys_footprint_bytes"]:
                    peak_sample = sample
            for cid in list(pending):
                st = pool.poll(cid)
                if st.state != "running":
                    pending.remove(cid)
                    results.append(st.to_dict())
            if pending:
                time.sleep(0.5)
        wall_ns = time.monotonic_ns() - t0
        completed = [r for r in results if r["state"] == "done"]
        per_wall = [int(r["wall_ns"]) for r in completed if r.get("wall_ns") is not None]
        return {
            "n": n,
            "seq_len": seq_len,
            "max_new_tokens": tokens,
            "status": "ok" if len(completed) == n else "partial",
            "aggregate_wall_ns": wall_ns,
            "per_child_wall_ns": per_wall,
            "completions": len(completed),
            "results": results,
            "peak_sample": peak_sample,
            "machine_before": machine0,
            "machine_after": sample_machine(),
            "throughput_completions_per_s": (len(completed) / (wall_ns / 1e9)) if wall_ns else 0.0,
        }
    except Exception as exc:
        pool.shutdown(kill=True)
        return {
            "n": n,
            "seq_len": seq_len,
            "status": "error",
            "reason": str(exc),
            "machine_before": machine0,
        }
    finally:
        pool.shutdown(kill=True)


def cmd_e2e(args: argparse.Namespace) -> int:
    live = live_ready()
    if live is None:
        print("e2e: binary/artifact/tokenizer not found", file=sys.stderr)
        return 2
    binary, artifact, tokenizer = live
    stray = _refuse_if_stray_children()
    if stray and not args.ignore_strays:
        print(json.dumps({"state": "blocked", "reason": "stray_children", "pids": stray}))
        return 4
    n = int(args.n)
    if n > MEASURED_SAFE_N and not args.force:
        print(
            json.dumps(
                {
                    "state": "refused",
                    "reason": f"n={n} > measured safe_n={MEASURED_SAFE_N}",
                    "safe_n": MEASURED_SAFE_N,
                }
            )
        )
        return 3
    seq = int(args.seq_len)
    tokens = int(args.max_new_tokens)
    out_root = Path(args.output_root)
    cfg = PoolConfig(
        binary=binary,
        artifact_root=artifact,
        tokenizer=tokenizer,
        output_root=out_root,
        safe_n=max(n, MEASURED_SAFE_N),
        max_seq_len=seq,
    )
    pool = GenesisPool(cfg)
    prompts = [f"{args.prompt} #{i}" for i in range(n)]
    t0 = time.monotonic_ns()
    ids = [
        pool.spawn(p, ChildBudget(max_new_tokens=tokens, max_seq_len=seq), hold_gpu_lock=False)
        for p in prompts
    ]
    results = [pool.wait(cid) for cid in ids]
    wall_ns = time.monotonic_ns() - t0
    body = {
        "schema": "hawking.ascent.genesis_pool_e2e.v1",
        "n": n,
        "seq_len": seq,
        "max_new_tokens": tokens,
        "aggregate_wall_ns": wall_ns,
        "per_child": [r.to_dict() for r in results],
        "completions": sum(1 for r in results if r.state == "done"),
        "throughput_completions_per_s": n / (wall_ns / 1e9) if wall_ns else 0.0,
        "output_root": str(out_root),
        "hold_gpu_lock": False,
        "workload": "TEXT",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(body, indent=2) + "\n")
    print(json.dumps({"wrote": args.out, "completions": body["completions"], "wall_ns": wall_ns}))
    return 0 if body["completions"] == n else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--binary")
        sp.add_argument("--artifact-root")
        sp.add_argument("--tokenizer")
        sp.add_argument("--output-root", default=str(REPO / "workspace/ops/local/genesis-pool"))
        sp.add_argument("--safe-n", type=int, default=MEASURED_SAFE_N)
        sp.add_argument("--seq-len", type=int, default=128)
        sp.add_argument("--min-free-bytes", type=int, default=0)

    sp = sub.add_parser("spawn")
    add_common(sp)
    sp.add_argument("prompt")
    sp.add_argument("budget", type=int)
    sp.add_argument("--timing", action="store_true", help="hold gpu_lane_lock.sh (TIMING, not TEXT)")
    sp.set_defaults(func=cmd_spawn)

    pp = sub.add_parser("poll")
    add_common(pp)
    pp.add_argument("child_id")
    pp.set_defaults(func=cmd_poll)

    kp = sub.add_parser("kill")
    add_common(kp)
    kp.add_argument("child_id")
    kp.set_defaults(func=cmd_kill)

    mp = sub.add_parser("measure")
    add_common(mp)
    mp.add_argument("--counts", default="1,2,4,8")
    mp.add_argument("--seq-lens", default="128,2048")
    mp.add_argument("--max-new-tokens", type=int, default=8)
    mp.add_argument("--prompt", default="Name one way to cut bytes moved per token.")
    mp.add_argument("--out", default=str(REPO / "receipts/ascent-2026-08-16/GENESIS_CHILDREN_CAPACITY.json"))
    mp.add_argument("--ignore-strays", action="store_true")
    mp.add_argument("--force", action="store_true")
    mp.set_defaults(func=cmd_measure)

    ep = sub.add_parser("e2e")
    add_common(ep)
    ep.add_argument("--n", type=int, default=4)
    ep.add_argument("--max-new-tokens", type=int, default=8)
    ep.add_argument("--prompt", default="Say one word.")
    ep.add_argument("--out", default=str(REPO / "receipts/ascent-2026-08-16/GENESIS_POOL_E2E.json"))
    ep.add_argument("--ignore-strays", action="store_true")
    ep.add_argument("--force", action="store_true")
    ep.set_defaults(func=cmd_e2e)

    st = sub.add_parser("stub-child")
    st.set_defaults(func=lambda a: stub_child_main(sys.argv[sys.argv.index("stub-child") + 1 :]))
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "stub-child":
        return stub_child_main(argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
