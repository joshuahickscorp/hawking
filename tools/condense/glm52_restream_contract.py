#!/usr/bin/env python3.12
"""Generate the sealed <=90-GB GLM-5.2 range schedule and matching policy."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.operators.glm52_common import atomic_json  # noqa: E402
from lab.operators.glm52_restream_contract import build_contract  # noqa: E402
from lab.layout import evidence_dir  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(evidence_dir("glm52") / "GLM52_OFFICIAL_MANIFEST.json"))
    parser.add_argument("--graph", default=str(evidence_dir("glm52") / "GLM52_SHARD_DEPENDENCY_GRAPH.json"))
    parser.add_argument("--schedule", default=str(evidence_dir("glm52") / "GLM52_STREAMING_SCHEDULE_90GB.json"))
    parser.add_argument("--policy", default=str(evidence_dir("glm52") / "GLM52_RESOURCE_RESERVE_POLICY_90GB.json"))
    args = parser.parse_args(argv)
    schedule, policy = build_contract(manifest_path=args.manifest, graph_path=args.graph)
    atomic_json(Path(args.schedule), schedule)
    atomic_json(Path(args.policy), policy)
    print(f"schedule={args.schedule} seal={schedule['seal_sha256']}")
    print(f"policy={args.policy} seal={policy['seal_sha256']}")
    print(f"peak_incremental_bytes={schedule['incremental_accounting_contract']['peak_incremental_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
