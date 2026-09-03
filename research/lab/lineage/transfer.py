"""Checksummed research-state transfer. Science must survive every death.

The parent hands the successor the task graph, memories, genomes, negative
science, active hypotheses, Grok findings, open experiments, and
NEXT_BOTTLENECK. The far side verifies the checksum before authority moves.
"""
from __future__ import annotations

from typing import Any, Mapping

from lab.lineage.canon import digest, require_mapping, require_nonempty_str, utc_now
from lab.lineage.identity import GenesisInstance, as_instance
from lab.receipts import seal, verify

SCHEMA = "hawking.lineage.state_transfer.v1"

TRANSFER_PAYLOAD_KEYS: tuple[str, ...] = (
    "task_graph",
    "memories",
    "genomes",
    "negative_science",
    "active_hypotheses",
    "grok_findings",
    "open_experiments",
    "NEXT_BOTTLENECK",
)


class TransferError(ValueError):
    """State transfer refused."""


class TransferChecksumError(TransferError):
    """Far-side checksum did not match the payload."""


def empty_payload() -> dict[str, Any]:
    return {
        "task_graph": [],
        "memories": [],
        "genomes": {},
        "negative_science": [],
        "active_hypotheses": [],
        "grok_findings": [],
        "open_experiments": [],
        "NEXT_BOTTLENECK": "",
    }


def normalize_payload(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    data = dict(raw or {})
    missing = [key for key in TRANSFER_PAYLOAD_KEYS if key not in data]
    if missing:
        raise TransferError(f"transfer payload missing required keys: {missing}")
    extra = sorted(set(data) - set(TRANSFER_PAYLOAD_KEYS))
    if extra:
        raise TransferError(f"transfer payload has unknown keys: {extra}")
    next_bn = data["NEXT_BOTTLENECK"]
    if not isinstance(next_bn, str) or not next_bn.strip():
        raise TransferError("NEXT_BOTTLENECK must be a non-empty string (science must name the next bottleneck)")
    payload = {key: data[key] for key in TRANSFER_PAYLOAD_KEYS}
    payload["NEXT_BOTTLENECK"] = next_bn.strip()
    return payload


def payload_checksum(payload: Mapping[str, Any]) -> str:
    return digest(normalize_payload(payload))


def pack_state(
    parent: GenesisInstance | Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    to: GenesisInstance | Mapping[str, Any],
) -> dict[str, Any]:
    """Parent-side pack. Checksum is over the payload only."""
    parent_i = as_instance(parent, "parent")
    child_i = as_instance(to, "to")
    normalized = normalize_payload(payload)
    checksum = digest(normalized)
    document = {
        "schema": SCHEMA,
        "from_instance": parent_i.instance_id,
        "to_instance": child_i.instance_id,
        "packed_at": utc_now(),
        "payload": normalized,
        "checksum_sha256": checksum,
    }
    return seal(document)


def accept_transfer(package: Mapping[str, Any]) -> dict[str, Any]:
    """Far-side verify. Authority must not move if this raises."""
    data = require_mapping(package, "transfer_package")
    if data.get("schema") != SCHEMA:
        raise TransferError(f"unexpected transfer schema {data.get('schema')!r}")
    verify(data, label="state_transfer")
    payload = normalize_payload(require_mapping(data.get("payload"), "payload"))
    expected = digest(payload)
    recorded = data.get("checksum_sha256")
    if recorded != expected:
        raise TransferChecksumError(
            f"transfer checksum mismatch: recorded={recorded!r} expected={expected}"
        )
    require_nonempty_str(data.get("from_instance"), "from_instance")
    require_nonempty_str(data.get("to_instance"), "to_instance")
    return {
        "from_instance": data["from_instance"],
        "to_instance": data["to_instance"],
        "payload": payload,
        "checksum_sha256": expected,
        "verified": True,
    }


def parent_research_payload(
    *,
    next_bottleneck: str,
    task_graph: Any = None,
    memories: Any = None,
    genomes: Any = None,
    negative_science: Any = None,
    active_hypotheses: Any = None,
    grok_findings: Any = None,
    open_experiments: Any = None,
) -> dict[str, Any]:
    payload = empty_payload()
    payload["NEXT_BOTTLENECK"] = next_bottleneck
    if task_graph is not None:
        payload["task_graph"] = task_graph
    if memories is not None:
        payload["memories"] = memories
    if genomes is not None:
        payload["genomes"] = genomes
    if negative_science is not None:
        payload["negative_science"] = negative_science
    if active_hypotheses is not None:
        payload["active_hypotheses"] = active_hypotheses
    if grok_findings is not None:
        payload["grok_findings"] = grok_findings
    if open_experiments is not None:
        payload["open_experiments"] = open_experiments
    return normalize_payload(payload)
