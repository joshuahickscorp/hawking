#!/usr/bin/env python3
"""External Genesis lifecycle commands.

The long-running resident and ascent daemon do not receive promotion authority.
They can submit/dispatch a candidate request, but this executable independently
qualifies it, performs the durable worker handoff, and is the only local entry
point that can ask the protected lineage gate to move authority.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lab.lineage.lifecycle import (
    CandidateInbox,
    DEFAULT_CANDIDATE_ROOT,
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_LINEAGE_PATH,
    DEFAULT_WORKER_REGISTRY,
    DEFAULT_WORKTREE_ROOT,
    LifecycleError,
    PromotionController,
    WorkerRegistry,
    benchmark_pair_command,
    process_candidate_inbox_once,
)
from lab.lineage.state import LineageState


def _add_runtime_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--state", type=Path, default=DEFAULT_LINEAGE_PATH)
    parser.add_argument("--workers", type=Path, default=DEFAULT_WORKER_REGISTRY)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATE_ROOT)


def _controller(args: argparse.Namespace) -> PromotionController:
    return PromotionController(
        repo=args.repo,
        state_path=args.state,
        worker_registry=WorkerRegistry(args.workers),
        checkpoint_root=args.checkpoints,
        candidate_root=args.candidates,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="genesis_lifecycle")
    sub = parser.add_subparsers(dest="command", required=True)
    pair = sub.add_parser("benchmark-pair")
    pair.add_argument("--request", type=Path, required=True)
    bootstrap = sub.add_parser("bootstrap-workers")
    _add_runtime_paths(bootstrap)
    worktrees = sub.add_parser("provision-worktrees")
    _add_runtime_paths(worktrees)
    worktrees.add_argument("--worktree-root", type=Path, default=DEFAULT_WORKTREE_ROOT)
    promote = sub.add_parser("promote")
    promote.add_argument("--request", type=Path, required=True)
    _add_runtime_paths(promote)
    submit = sub.add_parser("submit")
    submit.add_argument("--request", type=Path, required=True)
    submit.add_argument("--repo", type=Path, default=REPO)
    submit.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    process = sub.add_parser("process-inbox")
    _add_runtime_paths(process)
    status = sub.add_parser("status")
    _add_runtime_paths(status)
    args = parser.parse_args(argv)
    try:
        if args.command == "benchmark-pair":
            return benchmark_pair_command(args.request)
        if args.command == "bootstrap-workers":
            workers = _controller(args).bootstrap_workers()
            print(json.dumps({"ok": True, "workers": workers}, indent=2, sort_keys=True))
            return 0
        if args.command == "provision-worktrees":
            controller = _controller(args)
            controller.bootstrap_workers()
            workers = WorkerRegistry(args.workers).provision_worktrees(
                repo=args.repo,
                worktree_root=args.worktree_root,
            )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "worktree_root": str(args.worktree_root.resolve()),
                        "workers": workers,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "promote":
            result = _controller(args).promote(args.request)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "submit":
            stored = CandidateInbox(args.candidates).submit(args.request, repo=args.repo)
            print(json.dumps({"ok": True, "submitted": str(stored)}, sort_keys=True))
            return 0
        if args.command == "process-inbox":
            result = process_candidate_inbox_once(
                repo=args.repo,
                candidate_root=args.candidates,
                state_path=args.state,
                worker_registry_path=args.workers,
                checkpoint_root=args.checkpoints,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1 if result.get("outcome") == "FAILED" else 0
        if args.command == "status":
            state_raw = json.loads(args.state.read_text())
            lineage = LineageState.from_dict(state_raw)
            workers, _sha = WorkerRegistry(args.workers).load()
            print(
                json.dumps(
                    {
                        "current": None if lineage.current is None else lineage.current.to_dict(),
                        "candidate": None if lineage.candidate is None else lineage.candidate.to_dict(),
                        "last_known_good": (
                            None
                            if lineage.last_known_good is None
                            else lineage.last_known_good.to_dict()
                        ),
                        "worker_count": len(workers),
                        "workers": workers,
                        "candidate_inbox": CandidateInbox(args.candidates).status(),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
    except LifecycleError as exc:
        print(f"genesis lifecycle refused: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
