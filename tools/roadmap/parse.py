"""Parse H-ROADMAP.md into a status-free IR skeleton.

Statuses are never written here. The auditor fills them from the repo.
Prose is referenced by source span, never copied.
"""
from __future__ import annotations

import json

import hashlib
import os
import re
from pathlib import Path
from typing import Any

from tools.roadmap.gitfs import REPO

DEFAULT_ROADMAP = Path("/Users/scammermike/Downloads/H-ROADMAP.md")

# Must match civilization/build_state.py CANONICAL_PROGRAMS.
GENE_IDS: tuple[str, ...] = (
    "I-A_AGENTOS_HCLI",
    "I-B_DOCTOR",
    "I-C_GRAVITY_NOETIC",
    "I-D_ACCELERATOR",
    "I-E_ODYSSEY_I",
    "II-A_ODYSSEY_II",
    "II-B_NOETIC_COMPILER_V1",
    "II-C_PHYSICAL_GRAPH_COMPILER",
    "II-D_STATE_TOKENIZER_DECODING",
    "II-E_GREEN_MACHINE",
    "III-A_ODYSSEY_III",
    "III-B_LEARNED_PHYSICAL_COMPILER",
    "III-C_RESIDENT_OPTIMIZER",
    "III-D_BEYOND_DENSE_REPRESENTATION",
    "III-E_AUTONOMOUS_REPRODUCIBLE_SCIENCE",
    "IV-A_FUSION",
    "IV-B_HMF_HGVAS",
    "IV-C_DGX_SPARK",
    "IV-D_EGPU",
    "IV-E_FUSION_BRIDGE_TOPOLOGY_ASCENSION",
    "V-A_PRODUCT_SOVEREIGNTY",
    "V-B_DEVELOPER_PLATFORM",
    "V-C_CONTINUOUS_VERIFIED_IMPROVEMENT",
    "V-D_DOMINANCE_SCOREBOARD",
    "V-E_PERPETUAL_HAWKING",
)

_PREFIX_TO_ID = {gid.split("_", 1)[0]: gid for gid in GENE_IDS}

_GATE_LINE = re.compile(
    r"^- `([A-Z][A-Z0-9_]+)=NOT_RUN\|PASS\|FAIL\|BLOCKED\|INCONCLUSIVE`\s*$"
)
_GENE_CARD = re.compile(r"^## ((?:I|II|III|IV|V)-[A-E]) — .+ — GENE CARD\s*$")
_ERA_HEADING = re.compile(r"^## ((?:I|II|III|IV|V)-[A-E])\.\s")
_SUBGENE = re.compile(r"^- (.+)$")


def roadmap_path() -> Path:
    override = os.environ.get("H_ROADMAP")
    if override:
        return Path(override)
    return DEFAULT_ROADMAP


def span(path: str, start: int, end: int, *, note: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "file": path,
        "start_line": start,
        "end_line": end,
    }
    if note:
        out["note"] = note
    return out


def parse_roadmap(path: Path | None = None) -> dict[str, Any]:
    road = path or roadmap_path()
    if not road.is_file():
        raise FileNotFoundError(f"canonical roadmap not readable: {road}")
    raw = road.read_bytes()
    text = raw.decode("utf-8")
    lines = text.splitlines()
    road_s = str(road)

    gates = _parse_gates(lines, road_s)
    genes = _parse_genes(lines, road_s)
    era_spans = _parse_era_spans(lines, road_s)
    for gene in genes.values():
        prefix = gene["id"].split("_", 1)[0]
        if prefix in era_spans:
            gene["era_span"] = era_spans[prefix]

    return {
        "roadmap_path": road_s,
        "roadmap_hash": hashlib.sha256(raw).hexdigest(),
        "roadmap_line_count": len(lines),
        "gates": gates,
        "genes": genes,
        "era_spans": era_spans,
    }


def _parse_gates(lines: list[str], road_s: str) -> dict[str, dict[str, Any]]:
    appendix_o = None
    for i, line in enumerate(lines, start=1):
        if line.startswith("# APPENDIX O"):
            appendix_o = i
            break
    if appendix_o is None:
        raise ValueError("APPENDIX O not found in roadmap")
    gates: dict[str, dict[str, Any]] = {}
    for i, line in enumerate(lines, start=1):
        if i < appendix_o:
            continue
        if i > appendix_o and line.startswith("# APPENDIX"):
            break
        m = _GATE_LINE.match(line)
        if not m:
            continue
        name = m.group(1)
        gates[name] = {
            "id": name,
            "kind": "gate",
            "name": name,
            "source_span": span(road_s, i, i, note="APPENDIX O ledger row"),
            "ledger_line": i,
        }
    if len(gates) < 71:
        raise ValueError(f"APPENDIX O yielded {len(gates)} gates, expected 71")
    gates.update(_supplementary_gates(len(gates)))
    return gates


# The roster used to be owned SOLELY by APPENDIX O of the roadmap. That made the
# recompiled roadmap unable to represent any capability the superseded document
# never listed -- and the superseded document is preserved lineage that must not
# be edited. A capability the machine has but the roster cannot express is
# invisible to every audit, which is the failure this supplement removes.
#
# Supplementary rows are marked so nothing mistakes them for historical ledger
# entries, and they are ADDITIVE: APPENDIX O still wins on any id it defines.
SUPPLEMENT = "civilization/GATE_ROSTER_SUPPLEMENT.json"


def _supplementary_gates(appendix_count: int) -> dict[str, dict[str, Any]]:
    path = REPO / SUPPLEMENT
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{SUPPLEMENT} is present but unreadable: {exc}") from exc
    # A real line in a real file, exactly as an APPENDIX O row gets. A gate whose
    # span cannot be opened is a gate nobody can check the declaration of, so the
    # invariant that every gate has a locatable source stays intact.
    lines = path.read_text().splitlines()
    out: dict[str, dict[str, Any]] = {}
    for name, row in sorted((doc.get("gates") or {}).items()):
        line_no = next(
            (i for i, ln in enumerate(lines, start=1) if f'"{name}"' in ln), 1
        )
        out[name] = {
            "id": name,
            "kind": "gate",
            "name": name,
            "source_span": {
                "file": SUPPLEMENT,
                "start_line": line_no,
                "end_line": line_no,
                "note": "roster supplement (NOT an APPENDIX O ledger row)",
            },
            "ledger_line": line_no,
            "roster_source": "supplement",
            "declared_because": row.get("because") or "",
        }
    return out


def _parse_genes(lines: list[str], road_s: str) -> dict[str, dict[str, Any]]:
    cards: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines, start=1):
        m = _GENE_CARD.match(line)
        if m:
            prefix = m.group(1)
            gid = _PREFIX_TO_ID.get(prefix)
            if gid is None:
                raise ValueError(f"gene card prefix {prefix!r} has no canonical program id")
            cards.append((i, prefix, gid))
    if len(cards) != 25:
        raise ValueError(f"APPENDIX A yielded {len(cards)} gene cards, expected 25")

    genes: dict[str, dict[str, Any]] = {}
    for idx, (start, prefix, gid) in enumerate(cards):
        end = cards[idx + 1][0] - 2 if idx + 1 < len(cards) else _next_appendix(lines, start) - 1
        genes[gid] = {
            "id": gid,
            "kind": "gene",
            "name": gid,
            "era": _era_of(prefix),
            "prefix": prefix,
            "source_span": span(road_s, start, end, note="APPENDIX A gene card"),
            "subgenes": _parse_subgenes(lines, start, end),
        }
    return genes


def _era_of(prefix: str) -> str:
    roman = prefix.rsplit("-", 1)[0]
    return roman


def _parse_subgenes(lines: list[str], start: int, end: int) -> list[str]:
    names: list[str] = []
    in_sub = False
    for i in range(start, min(end + 1, len(lines) + 1)):
        line = lines[i - 1]
        if line.startswith("### SUBGENES"):
            in_sub = True
            continue
        if in_sub:
            if line.startswith("### ") or line.startswith("## "):
                break
            m = _SUBGENE.match(line)
            if m:
                names.append(m.group(1).strip())
    return names


def _parse_era_spans(lines: list[str], road_s: str) -> dict[str, dict[str, Any]]:
    heads: list[tuple[int, str]] = []
    for i, line in enumerate(lines, start=1):
        m = _ERA_HEADING.match(line)
        if m:
            heads.append((i, m.group(1)))
    spans: dict[str, dict[str, Any]] = {}
    for idx, (start, prefix) in enumerate(heads):
        if idx + 1 < len(heads):
            end = heads[idx + 1][0] - 1
        else:
            end = _next_top_heading(lines, start) - 1
        spans[prefix] = span(road_s, start, end, note="§5 era membership")
    return spans


def _next_appendix(lines: list[str], after: int) -> int:
    for i in range(after + 1, len(lines) + 1):
        if lines[i - 1].startswith("# APPENDIX"):
            return i
    return len(lines) + 1


def _next_top_heading(lines: list[str], after: int) -> int:
    for i in range(after + 1, len(lines) + 1):
        if lines[i - 1].startswith("# "):
            return i
    return len(lines) + 1


def load_existing_state() -> dict[str, Any] | None:
    path = REPO / "civilization" / "ROADMAP_STATE.json"
    if not path.is_file():
        return None
    import json

    return json.loads(path.read_text())
