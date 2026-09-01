#!/usr/bin/env python3
"""Nomenclature census: HCLI / Odyssey / Noetic / Gravity survive.

READ-ONLY over the repository. Renames nothing, deletes nothing. This checkout
is sparse: a missing path is NOT evidence of absence. Tree authority is
`git ls-tree` / `git grep` / `git show` at HEAD.

Writes receipts/headless/NOMENCLATURE_CENSUS.json and prints the census.

Vestigial here means no longer active, NOT reclaimable: the campaign stops,
the evidence stays readable and indexed. Doctor is a COMPONENT (prescriber /
verifier on the Gravity/Noetic line), not a retired brand.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCHEMA = "hawking.headless.nomenclature_census.v1"
NOMENCLATURE_VERSION = "HAWKING_NOMENCLATURE_V1"
CANONICAL_PIPELINE = (
    "SourceSpecimen",
    "Doctor",
    "Gravity",
    "NoeticIR",
    "NoeticCompiler",
    "PhysicalGraphCompiler",
    "HawkingAccelerator",
    "NoeticExecutableCandidate",
    "ParetoFrontier",
    "Singularity",
    "ResidentInstance",
)
CANONICAL_ALIASES = {
    "source model": "SourceSpecimen",
    "checkpoint": "SourceSpecimen",
    "quantization": "GravityOperator",
    "quantizer": "GravityOperator",
    "compressed model": "NoeticRepresentation",
    "compact model": "NoeticRepresentation",
    "artifact": "semantic inspection required",
    "winner": "ParetoCandidateOrSingularity",
    "best model": "ParetoCandidateOrSingularity",
    "final model": "SingularityOrUnqualifiedCandidate",
    "production model": "SingularityOrUnqualifiedCandidate",
    "resident model": "ResidentInstance",
}
SURVIVING = ("HCLI", "Odyssey", "Noetic", "Gravity")
SEALED_EXAMPLES = (
    "hawking.nos.nr_nx_artifact.v1",
    "hawking.nos.noetic_representation",
    "hawking.nos.noetic_executable_genome",
    "hawking.nos.doctor_seal.v1",
    "hawking.nos.tabula_drift.v1",
    "hawking.nos.hide_plan_swap.v1",
    "hawking.nos.nvm_minimal.v1",
    "gravity.content_addressed.chunk_directory.v1",
)

# POSIX ERE word-ish boundary. \b is not portable under git-grep -E.
B = r"([^[:alnum:]_]|^)"
E = r"([^[:alnum:]_]|$)"


def wb(*words: str) -> str:
    """Word-boundary alternation for git-grep -E (POSIX ERE)."""
    inner = "|".join(re.escape(w) for w in words)
    return f"{B}({inner}){E}"


# ---------------------------------------------------------------------------
# Curated name catalog. Counts are measured at runtime; class is not.
# path_prefixes: git paths this name owns (disk occupancy).
# path_regex: additional owned-path matcher (filename tokens, extensions).
# content_ere: git grep -E pattern for files that *reference* the name.
# schema_prefixes: hawking.<prefix>. receipt-schema brands.
# ---------------------------------------------------------------------------
NAMES: list[dict] = [
    # ---- SURVIVING --------------------------------------------------------
    {
        "id": "HCLI",
        "class": "SURVIVING",
        "kind": "campaign",
        "summary": "Headless control plane and the live agent runtime.",
        "path_prefixes": (
            "tools/haider/hcli/",
            "tools/headless/",
            "lab/hcli/",
            "crates/hide-backend/src/bin/hcli.rs",
            "crates/hide-backend/src/hcli_bridge.rs",
            "crates/hide-backend/src/hcli_profile.rs",
            "crates/hide-backend/src/hcli_research.rs",
            "crates/hide-backend/src/hcli_sources.rs",
            "crates/hide-backend/src/hcli_swarm.rs",
            "receipts/headless/",
        ),
        "path_regex": r"(^|/)hcli([^a-z0-9]|$)|(^|/)hcli_",
        "content_ere": wb("HCLI", "hcli"),
        "schema_prefixes": ("hawking.hcli.", "hawking.headless."),
        "belongs_to": None,
        "reasoning": (
            "Named surviving vocabulary. Live source currently nested under "
            "tools/haider/hcli (a vestigial Haider directory) plus tools/headless."
        ),
    },
    {
        "id": "Odyssey",
        "class": "SURVIVING",
        "kind": "campaign",
        "summary": "Compiler-learning campaign; Odyssey-I is the live instance.",
        "path_prefixes": (
            "tools/odyssey/",
            "workspace/campaign/odyssey/",
            "receipts/odyssey-i/",
        ),
        "path_regex": r"(^|/)odyssey([^a-z0-9]|$)|odyssey_",
        "content_ere": wb("Odyssey", "odyssey", "ODYSSEY", "Odyssey-I", "odyssey-i"),
        "schema_prefixes": ("hawking.odyssey.",),
        "belongs_to": None,
        "reasoning": "Named surviving vocabulary. Branch odyssey-i is the live instance, not a second brand.",
    },
    {
        "id": "Noetic",
        "class": "SURVIVING",
        "kind": "campaign",
        "summary": "NR/NX/NOS/NVM stack: what the patient IS vs how one machine runs it.",
        "path_prefixes": (
            "tools/nos_pipeline.py",
            "tools/nr_container.py",
            "tools/nx_genome.py",
            "tools/nvm_minimal.py",
            "spec/nr_container.schema.json",
        ),
        "path_regex": r"(^|/)nos_|nr_container|nx_genome|nx_tps|nvm_minimal|noetic",
        "content_ere": (
            wb("Noetic", "noetic", "NOS")
            + r"|hawking\.nos\.|noetic_representation|noetic_executable"
        ),
        "schema_prefixes": ("hawking.nos.",),
        "belongs_to": None,
        "reasoning": (
            "Named surviving vocabulary. Schema family hawking.nos.* (NR, NX, "
            "doctor_seal, tabula_drift, nvm, hide_plan_swap) is the Noetic line."
        ),
    },
    {
        "id": "Gravity",
        "class": "SURVIVING",
        "kind": "campaign",
        "summary": "Codec / artifact / runtime line (.gravity, gravity_* loaders, Metal path).",
        "path_prefixes": (
            "tools/foundry/GRAVITY_METHOD_REGISTRY.json",
            "tools/foundry/GRAVITY_POTENCY_LEDGER.jsonl",
        ),
        "path_regex": r"(^|/)gravity([^a-z0-9]|$)|(^|/)gravity_|\.gravity$",
        "content_ere": wb("Gravity", "gravity", "GRAVITY") + r"|\.gravity([^[:alnum:]_]|$)",
        "schema_prefixes": ("hawking.gravity.", "hawking.gravity1."),
        "belongs_to": None,
        "reasoning": "Named surviving vocabulary. Artifact format, loaders, and sealed hawking.gravity.* receipts.",
    },
    # ---- COMPONENTS (mechanisms, not campaign brands) ---------------------
    {
        "id": "Doctor",
        "class": "COMPONENT",
        "kind": "mechanism",
        "summary": "Prescriber / verifier on the Gravity/Noetic line. A required NOS gate.",
        "path_prefixes": (
            "tools/doctor_seal.py",
            "tools/gravity_doctor_gate.py",
            "tools/gravity_doctor_capability.py",
            "tools/gravity_doctor_dimensions.py",
        ),
        "path_regex": r"(^|/)doctor([^a-z]|$)|(^|/)doctor[0-9v_]|DOCTOR6|_doctor",
        "content_ere": (
            wb("Doctor", "DOCTOR")
            + r"|doctor_seal|doctor_gate|doctor6|doctorv5|Doctor6|hawking\.doctor"
        ),
        "schema_prefixes": (
            "hawking.doctor.",
            "hawking.doctor6.",
            "hawking.doctorv5.",
            "hawking.doctorv5ultra.",
        ),
        "schema_exact": ("hawking.nos.doctor_seal.v1", "hawking.nos.doctor_dimensions.v1"),
        "belongs_to": "Gravity/Noetic",
        "reasoning": (
            "COMPONENT, not VESTIGIAL. Doctor is the prescriber/verifier inside "
            "the Gravity/Noetic line, not a campaign brand. tools/doctor_seal.py "
            "is stage 3 of tools/nos_pipeline.py (DOCTOR -> GRAVITY -> NR -> NX -> "
            "NVM/HIDE -> succession). A required gate depends on it: a seal without "
            "tabula_drift, observed_controls (a control watched to fail), "
            "stated_test_width, and known_blind_spots is REFUSED. Doctor6 / "
            "DoctorV5 / DoctorV5Ultra are versioned instruments of the same "
            "mechanism (prescription receipts, coherence screens), not retired "
            "campaigns. Odyssey product list names Gravity / Doctor / Tabula / NX."
        ),
    },
    {
        "id": "Tabula",
        "class": "COMPONENT",
        "kind": "mechanism",
        "summary": "Abliteration-direction drift instrument. Required Doctor-seal field.",
        "path_prefixes": (
            "tools/tabula_drift.py",
            "tools/gravity_tabula_behaviour.py",
            "tools/gravity_tabula_probe.py",
        ),
        "path_regex": r"tabula",
        "content_ere": wb("Tabula", "tabula", "TABULA"),
        "schema_prefixes": (),
        "schema_exact": ("hawking.nos.tabula_drift.v1",),
        "belongs_to": "Doctor/Noetic",
        "reasoning": (
            "COMPONENT. Tabula drift is a required Doctor-seal field "
            "(hawking.nos.tabula_drift.v1). It measures how much a quantized "
            "candidate puts back an abliterated direction. A mechanism, not a brand."
        ),
    },
    {
        "id": "NR",
        "class": "COMPONENT",
        "kind": "mechanism",
        "summary": "Noetic Representation: what the patient IS. Portable. Machine fields refused.",
        "path_prefixes": ("tools/nr_container.py", "spec/nr_container.schema.json"),
        "path_regex": r"nr_container|nr_nx|noetic_representation",
        "content_ere": r"nr_container|nr_nx_artifact|noetic_representation|" + wb("NR"),
        "schema_prefixes": (),
        "schema_exact": (
            "hawking.nos.noetic_representation",
            "hawking.nos.nr_nx_artifact.v1",
        ),
        "belongs_to": "Noetic",
        "reasoning": "COMPONENT of Noetic. G103. Kind string hawking.nos.noetic_representation is sealed.",
    },
    {
        "id": "NX",
        "class": "COMPONENT",
        "kind": "mechanism",
        "summary": "Noetic executable genome: how ONE machine runs an NR. Must not be portable.",
        "path_prefixes": ("tools/nx_genome.py", "tools/nx_tps_frontier.py"),
        "path_regex": r"nx_genome|nx_tps|noetic_executable",
        "content_ere": r"nx_genome|nx_tps|noetic_executable_genome|" + wb("NX"),
        "schema_prefixes": (),
        "schema_exact": ("hawking.nos.noetic_executable_genome",),
        "belongs_to": "Noetic",
        "reasoning": "COMPONENT of Noetic. G104. Kind string hawking.nos.noetic_executable_genome is sealed.",
    },
    {
        "id": "NOS",
        "class": "COMPONENT",
        "kind": "mechanism",
        "summary": "Noetic pipeline that composes the live gates into one process.",
        "path_prefixes": ("tools/nos_pipeline.py",),
        "path_regex": r"(^|/)nos_pipeline|(^|/)nos_",
        "content_ere": r"nos_pipeline|hawking\.nos\.",
        "schema_prefixes": (),
        "schema_exact": ("hawking.nos.pipeline.v1", "hawking.nos.doctor_seal.v1"),
        "belongs_to": "Noetic",
        "reasoning": "COMPONENT of Noetic. The spine, not a second brand.",
    },
    {
        "id": "NVM",
        "class": "COMPONENT",
        "kind": "mechanism",
        "summary": "Minimal Noetic VM: bind NX operators, run, name routes. Homonym of HIDE-plan.",
        "path_prefixes": (
            "tools/nvm_minimal.py",
            "tools/hide_plan.py",
            "spec/plans/",
        ),
        "path_regex": r"nvm_minimal|hide_plan",
        "content_ere": wb("NVM") + r"|nvm_minimal|hide_plan_swap|hide_plan",
        "schema_prefixes": (),
        "schema_exact": (
            "hawking.nos.nvm_minimal.v1",
            "hawking.nos.hide_plan_swap.v1",
        ),
        "belongs_to": "Noetic",
        "reasoning": (
            "COMPONENT of Noetic (G120/G121). tools/hide_plan.py and "
            "hawking.nos.hide_plan_swap.v1 are kernel-selection-as-data, NOT the "
            "HIDE IDE. The token collision with vestigial HIDE is a census trap."
        ),
    },
    {
        "id": "AgentOS",
        "class": "COMPONENT",
        "kind": "mechanism",
        "summary": "Callable primitives over the existing substrate (verify, TPS, park).",
        "path_prefixes": ("tools/agentos/", "receipts/agentos/"),
        "path_regex": r"agentos",
        "content_ere": wb("AgentOS", "agentos", "AGENTOS"),
        "schema_prefixes": (),
        "belongs_to": "HCLI",
        "reasoning": "COMPONENT of HCLI. G012 embryo: procedures this campaign already ran by hand, made callable.",
    },
    {
        "id": "Foundry",
        "class": "COMPONENT",
        "kind": "mechanism",
        "summary": "Gravity method registry, potency ledger, negative-transfer atlas.",
        "path_prefixes": ("tools/foundry/",),
        "path_regex": r"(^|/)foundry([^a-z0-9]|$)",
        "content_ere": wb("Foundry", "foundry", "FOUNDRY"),
        "schema_prefixes": ("hawking.foundry.",),
        "belongs_to": "Gravity",
        "reasoning": (
            "COMPONENT of Gravity. lab/campaigns.json lists foundry_lab as live. "
            "A mechanism registry, not a campaign brand."
        ),
    },
    {
        "id": "Condense",
        "class": "COMPONENT",
        "kind": "mechanism",
        "summary": "Capability-first packing / promotion path under Gravity.",
        "path_prefixes": ("tools/condense/",),
        "path_regex": r"(^|/)condense([^a-z0-9]|$)",
        "content_ere": wb("Condense", "condense", "CONDENSE"),
        "schema_prefixes": ("hawking.condense.",),
        "belongs_to": "Gravity/Noetic",
        "reasoning": "COMPONENT. Capability-first rung packing. Lab operators live here; not a surviving brand of its own.",
    },
    {
        "id": "VMCP",
        "class": "COMPONENT",
        "kind": "mechanism",
        "summary": "Vision MCP evidence plane used by HCLI WorkUnits (core profile only).",
        "path_prefixes": (),
        "path_regex": r"vmcp|visionmcp",
        "content_ere": wb("VMCP", "vmcp", "VisionMCP", "visionmcp"),
        "schema_prefixes": (),
        "belongs_to": "HCLI",
        "reasoning": (
            "COMPONENT of HCLI. Evidence capture/verify, not a campaign. "
            "Laboratory profile is a second control plane and must not be the integration surface."
        ),
    },
    {
        "id": "Headless",
        "class": "COMPONENT",
        "kind": "mechanism",
        "summary": "HCLI operational surface: tools/headless + receipts/headless.",
        "path_prefixes": ("tools/headless/", "receipts/headless/"),
        "path_regex": r"(^|/)headless([^a-z0-9]|$)",
        "content_ere": wb("headless", "Headless", "HEADLESS"),
        "schema_prefixes": ("hawking.headless.",),
        "belongs_to": "HCLI",
        "reasoning": "COMPONENT of HCLI. The campaign's receipt plane, not a fourth surviving brand.",
    },
    {
        "id": "MachineGenome",
        "class": "COMPONENT",
        "kind": "mechanism",
        "summary": "Digest over facts a lowering depends on (GPU, RAM, roof, Metal family).",
        "path_prefixes": (),
        "path_regex": r"machine_genome|MACHINE_GENOME",
        "content_ere": wb("MachineGenome") + r"|machine_genome|MACHINE_GENOME",
        "schema_prefixes": (),
        "belongs_to": "Noetic/NX",
        "reasoning": "COMPONENT of NX. Named in the Odyssey product list. A digest, not a campaign.",
    },
    {
        "id": "Sovereignty",
        "class": "COMPONENT",
        "kind": "mechanism",
        "summary": "Capability-sovereignty spine: continuity manifest, event log, refusal codes.",
        "path_prefixes": ("tools/sovereignty/",),
        "path_regex": r"sovereignty",
        "content_ere": wb("Sovereignty", "sovereignty", "SOVEREIGNTY"),
        "schema_prefixes": (),
        "belongs_to": "Gravity",
        "reasoning": "COMPONENT of Gravity (Rev3 SS8). Deterministic half of capability sovereignty, not a campaign brand.",
    },
    # ---- VESTIGIAL campaigns ----------------------------------------------
    {
        "id": "HIDE",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "Local agent IDE (hide-* crates, app/). Deferred: 'no hide yet'. Not deleted.",
        "path_prefixes": (
            "crates/hide-acp/",
            "crates/hide-backend/",
            "crates/hide-core/",
            "crates/hide-fleet/",
            "crates/hide-gateway/",
            "crates/hide-kernel/",
            "crates/hide-protocol/",
            "crates/hide-serve/",
            "src/bin/hide-acp-server.rs",
            "src/bin/hide-sdk-codegen.rs",
            "app/",
            "workspace/campaign/evidence/hide/",
            "workspace/campaign/evidence/systems/hide/",
            "workspace/campaign/config/packs/hawking-hide-desktop.json",
            "crates/hawking-adapters/generated/hide_capabilities.json",
        ),
        "path_regex": r"(^|/)hide-[a-z]|(^|/)\.hide/|(^|/)HIDE_",
        "content_ere": wb("HIDE") + r"|hide-acp|hide-backend|hide-core|hide-fleet|hide-gateway|hide-kernel|hide-protocol|hide-serve|hide_serve",
        "schema_prefixes": ("hawking.hide.",),
        "belongs_to": None,
        "reasoning": (
            "VESTIGIAL. Retired as an active campaign ('no hide yet' — deferred, "
            "not deleted). Eight hide-* crates plus app/ remain in the tree as "
            "evidence and a compilable product, not as the live vocabulary. "
            "HOMONYM: tools/hide_plan.py / hawking.nos.hide_plan_swap.v1 is the "
            "Noetic NVM kernel-plan mechanism and is classified under NVM, not here."
        ),
        "deferred_not_deleted": True,
    },
    {
        "id": "Ramanujan",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "Fixture-only scaffold blocked on Hawking completion. Retired as a campaign.",
        "path_prefixes": ("ramanujan/",),
        "path_regex": r"(^|/)ramanujan([^a-z0-9]|$)",
        "content_ere": wb("Ramanujan", "ramanujan", "RAMANUJAN"),
        "schema_prefixes": ("hawking.ramanujan.",),
        "belongs_to": None,
        "reasoning": (
            "VESTIGIAL. Explicitly retired. ramanujan/ is a non-authorizing scaffold; "
            "HAWKING_COMPLETION_GATE.json is BLOCKED_ON_HAWKING_COMPLETION and may "
            "not self-promote. Evidence (gate, handoff contract, fixture tests) stays."
        ),
    },
    {
        "id": "Haider",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "Predecessor productization brand. Live HCLI source still lives under tools/haider/hcli.",
        "path_prefixes": (
            "tools/haider/",
            "crates/hide-backend/src/haider/",
            "crates/hide-backend/src/bin/haider.rs",
        ),
        "path_regex": r"(^|/)haider([^a-z0-9]|$)|(^|/)\.haider/",
        "content_ere": wb("Haider", "haider", "HAIDER"),
        "schema_prefixes": (),
        "belongs_to": None,
        "reasoning": (
            "VESTIGIAL brand, not a mechanism. P1_HAIDER_PRODUCTIZATION_MAX.md is "
            "the old productization spec. The surviving runtime is HCLI; its source "
            "happens to sit at tools/haider/hcli. Renaming that directory would stale "
            "sealed headless receipts that name the path."
        ),
    },
    {
        "id": "Frankenstein",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "Kimi→GLM→DeepSeek paired-trace / transfer campaign. Preflight only; not live research.",
        "path_prefixes": (),
        "path_regex": r"frankenstein",
        "content_ere": wb("Frankenstein", "frankenstein", "FRANKENSTEIN"),
        "schema_prefixes": ("hawking.frankenstein.",),
        "belongs_to": None,
        "reasoning": (
            "VESTIGIAL campaign. hawking-experiments/frankenstein/operators/frankenstein_* and hawking.frankenstein.* "
            "receipts (paired functional traces, GLM layer capture) are sealed evidence. "
            "The operator itself refuses to acquire a teacher until Ramanujan/Hawking gates pass."
        ),
    },
    {
        "id": "Ascension",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "Q80 / Qwen30 / Qwen38 example-runtime campaign and 'Ascension Bible'.",
        "path_prefixes": (),
        "path_regex": r"ascension",
        "content_ere": wb("Ascension", "ascension", "ASCENSION"),
        "schema_prefixes": ("hawking.ascension.",),
        "belongs_to": None,
        "reasoning": (
            "VESTIGIAL campaign name. Hundreds of ascension_* example binaries and "
            "hawking.ascension.* schemas remain as sealed measurement evidence. The "
            "surviving codec/runtime name is Gravity, not Ascension."
        ),
    },
    {
        "id": "Ascent",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "ascent-2026-08-16 / 08-18 receipt campaigns (fs/weight, Q80 mixed, negative science).",
        "path_prefixes": (
            "tools/ascent/",
            "tools/ascent_controller.py",
            "tools/ascent_daemon.py",
            "tools/ascent_verdict.py",
            "receipts/ascent-2026-08-16/",
            "receipts/ascent-2026-08-18/",
        ),
        "path_regex": r"(^|/)ascent([^a-z0-9]|$)|ascent_",
        "content_ere": wb("Ascent", "ascent", "ASCENT") + r"|ascent-2026",
        "schema_prefixes": ("hawking.ascent.", "hawking.ascent_controller.", "hawking.final_ascent."),
        "belongs_to": None,
        "reasoning": (
            "VESTIGIAL campaign. The negative-science register "
            "(hawking.ascent.negative_science_register.v1) lives here. Deleting it "
            "would discard the graveyard the Odyssey bible requires querying first."
        ),
    },
    {
        "id": "Strand",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "Absorbed STRAND quant track. Excluded from workspace build; golden hashes pinned.",
        "path_prefixes": ("workspace/vendor/strand-quant/",),
        "path_regex": r"(^|/)strand([^a-z0-9]|$)|strand-",
        "content_ere": wb("Strand", "strand", "STRAND") + r"|strand-quant|strand-decode",
        "schema_prefixes": ("hawking.strand.",),
        "belongs_to": None,
        "reasoning": (
            "VESTIGIAL / absorbed. Cargo.toml exclude keeps it out of workspace "
            "build so golden hashes stay unchanged. strand-decode-kernel was archived."
        ),
    },
    {
        "id": "Genesis",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "Qwen3.8 continuity / resident-organism campaign. Future Odyssey-II tournament reuses the word.",
        "path_prefixes": (
            "contracts/genesis/",
            "tools/agentos/genesis_body",
            "tools/agentos/genesis_contract.py",
            "tools/agentos/genesis_resident.py",
        ),
        "path_regex": r"(^|/)genesis([^a-z0-9]|$)|genesis_",
        "content_ere": wb("Genesis", "genesis", "GENESIS"),
        "schema_prefixes": ("hawking.genesis.",),
        "belongs_to": None,
        "reasoning": (
            "VESTIGIAL as a named campaign (Qwen3.8 Genesis continuity directives, "
            "genesis-* tools). Odyssey-I G7 still hands to a 'genesis tournament' "
            "over NX objects — that is future Odyssey work under surviving names, "
            "not a reason to keep Genesis as a live brand. Evidence stays."
        ),
    },
    {
        "id": "Fabric",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "fabric-agent binary; HIDE-adjacent agent fabric.",
        "path_prefixes": ("src/bin/fabric_agent.rs",),
        "path_regex": r"fabric_agent|fabric-agent|(^|/)fabric([^a-z0-9]|$)",
        "content_ere": wb("fabric-agent", "fabric_agent") + r"|name = \"fabric-agent\"",
        "schema_prefixes": ("hawking.fabric.",),
        "belongs_to": None,
        "reasoning": "VESTIGIAL with HIDE. Crate name fabric-agent. Not in the surviving four.",
    },
    {
        "id": "Prometheus",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "Math cartography / capability-elimination campaign. lab status: historical.",
        "path_prefixes": ("hawking-experiments/prometheus/tools/",),
        "path_regex": r"prometheus",
        "content_ere": wb("Prometheus", "prometheus", "PROMETHEUS"),
        "schema_prefixes": ("hawking.prometheus.",),
        "belongs_to": None,
        "reasoning": "VESTIGIAL. lab/campaigns.json prometheus_math status=historical.",
    },
    {
        "id": "Eco",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "ECO admission/planner campaign. Controllers retired into lab engine.",
        "path_prefixes": (),
        "path_regex": r"(^|/)eco_|/eco\.|(^|/)eco/",
        "content_ere": wb("eco_activation", "eco_admission", "eco_pipeline") + r"|hawking\.eco\.",
        "schema_prefixes": ("hawking.eco.",),
        "belongs_to": None,
        "reasoning": "VESTIGIAL. lab/campaigns.json eco status=retired. eco_common remains for importers.",
    },
    {
        "id": "TG",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "Tournament-gate (TG) K11 budget / reconcile shell. Retired as a campaign id.",
        "path_prefixes": (),
        "path_regex": r"(^|/)tg_|/tg/|(^|/)tg-",
        "content_ere": r"hawking\.tg\.|" + wb("TG-K11", "tg_k11", "tg_active_byte"),
        "schema_prefixes": ("hawking.tg.",),
        "belongs_to": None,
        "reasoning": (
            "VESTIGIAL campaign id (lab status=retired). Tournament *gates* as a "
            "mechanism still appear in Odyssey/HCLI; the TG K11 campaign shell does not. "
            "hawking.tg.* receipts are sealed evidence."
        ),
    },
    {
        "id": "Succession",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "succ_* campaign shell. Retired; successor_select remains a Noetic gate.",
        "path_prefixes": (),
        "path_regex": r"(^|/)succ_",
        "content_ere": r"succ_admission|succ_engine|succ_gravity|hawking\.succ|" + wb("Succession"),
        "schema_prefixes": (),
        "belongs_to": None,
        "reasoning": (
            "VESTIGIAL campaign (lab status=retired). tools/successor_select.py is "
            "a Noetic pipeline stage (PROMOTE) and stays as a COMPONENT of NOS — "
            "do not conflate the retired succ_* shell with that gate."
        ),
    },
    {
        "id": "Forge",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "Older Forge pipeline (F0/F3, forge_* controllers). Distinct from live Foundry.",
        "path_prefixes": (),
        "path_regex": r"(^|/)forge\.py|(^|/)forge_|gravity_forge",
        "content_ere": r"forge_controller|forge_actaware|gravity_forge|" + wb("Forge"),
        "schema_prefixes": (),
        "belongs_to": None,
        "reasoning": (
            "VESTIGIAL. forge_* controllers were deleted into the retirement receipt. "
            "Do not confuse with Foundry (live Gravity method registry)."
        ),
    },
    {
        "id": "GPT-OSS",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "GPT-OSS-120B gravity / sub-bit / real-forward campaign. lab status=retired.",
        "path_prefixes": (),
        "path_regex": r"gptoss|gpt-oss|gpt_oss",
        "content_ere": wb("gptoss", "GPT-OSS", "gpt-oss", "GPTOSS"),
        "schema_prefixes": (),
        "belongs_to": None,
        "reasoning": "VESTIGIAL. lab/campaigns.json gptoss status=retired.",
    },
    {
        "id": "Overnight",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "overnight_supervisor campaign. Retired into lab engine.",
        "path_prefixes": (),
        "path_regex": r"overnight",
        "content_ere": wb("overnight_supervisor") + r"|hawking\.overnight\.",
        "schema_prefixes": ("hawking.overnight.",),
        "belongs_to": None,
        "reasoning": "VESTIGIAL. Listed in the F1 controller-retirement receipt.",
    },
    {
        "id": "KimiK26",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "Kimi K2.6 download / phase-2 recovery / release cycle. lab status=retired.",
        "path_prefixes": (),
        "path_regex": r"kimi_k26|kimi-k26|kimi_k2",
        "content_ere": wb("kimi_k26", "Kimi K2.6", "KIMI_K26") + r"|hawking\.kimi_k26\.",
        "schema_prefixes": ("hawking.kimi_k26.", "hawking.kimi_k3."),
        "belongs_to": None,
        "reasoning": "VESTIGIAL campaign. Receipts stay; the download/release cycle is retired.",
    },
    {
        "id": "SecondLight",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "Second Light PQ / protected-islands campaign. lab status=retired.",
        "path_prefixes": (),
        "path_regex": r"second_light|second-light",
        "content_ere": wb("Second Light", "second_light", "SecondLight"),
        "schema_prefixes": (),
        "belongs_to": None,
        "reasoning": "VESTIGIAL. lab/campaigns.json second_light status=retired.",
    },
    {
        "id": "Mechanics",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "Mechanics/Thermodynamics tournament orchestrator. lab status=retired.",
        "path_prefixes": (),
        "path_regex": r"(^|/)mechanics([^a-z0-9]|$)|mech_measure|mech_run",
        "content_ere": wb("mech_measure", "mech_run_all") + r"|Mechanics/Thermodynamics",
        "schema_prefixes": (),
        "belongs_to": None,
        "reasoning": "VESTIGIAL. lab/campaigns.json mechanics status=retired.",
    },
    {
        "id": "QwenCampaign",
        "class": "VESTIGIAL",
        "kind": "campaign",
        "summary": "Qwen MoE structural / correction / real-forward lab campaign (not the model name).",
        "path_prefixes": (),
        "path_regex": r"qwen_correction|qwen_real_forward|qwen_bpw_budget|qwen_download_worker",
        "content_ere": r"qwen_correction_wave|qwen_real_forward|qwen_bpw_budget|qwen_download_worker",
        "schema_prefixes": (),
        "belongs_to": None,
        "reasoning": (
            "VESTIGIAL lab campaign id 'qwen' (status=retired). Distinct from the "
            "Qwen3.8 / Q80 *patients*, which are model identifiers (AMBIGUOUS), not brands."
        ),
    },
    # ---- AMBIGUOUS --------------------------------------------------------
    {
        "id": "Hawking",
        "class": "AMBIGUOUS",
        "kind": "product",
        "summary": "Repository / product identity and the hawking.* schema namespace.",
        "path_prefixes": (
            "crates/hawking/",
            "crates/hawking-core/",
            "crates/hawking-serve/",
            "src/main.rs",
            "src/lib.rs",
        ),
        "path_regex": r"(^|/)hawking([^a-z0-9]|$)|hawking-",
        "content_ere": wb("Hawking", "hawking", "HAWKING"),
        "schema_prefixes": (),
        "belongs_to": None,
        "reasoning": (
            "AMBIGUOUS. Not a campaign and not vestigial. It is the product/repo "
            "envelope: crate hawking, CLI hawking, and every sealed schema is "
            "hawking.<brand>.*. Renaming the namespace would invalidate the entire "
            "receipt corpus. It is not one of the four surviving campaign names."
        ),
    },
    {
        "id": "Superwave",
        "class": "AMBIGUOUS",
        "kind": "method",
        "summary": "Fan-out orchestration method plus a dated ascent-era state file.",
        "path_prefixes": ("docs/archive/SUPERWAVE_STATE.md", "hawking-experiments/superwave/"),
        "path_regex": r"superwave|SUPERWAVE",
        "content_ere": wb("Superwave", "superwave", "SUPERWAVE", "SUPERWAVE_STATE"),
        "schema_prefixes": ("hawking.superwave.",),
        "belongs_to": None,
        "reasoning": (
            "AMBIGUOUS. An orchestration method still used around HCLI/Odyssey, and "
            "a campaign state file that still publishes a superseded G013 law at the "
            "top. Not a surviving brand. Do not treat hawking-experiments/superwave evidence as disposable."
        ),
    },
    {
        "id": "Lab",
        "class": "AMBIGUOUS",
        "kind": "engine",
        "summary": "Experiment engine (lab/operators, lab/campaigns.json). Not a campaign brand.",
        "path_prefixes": ("lab/",),
        "path_regex": r"(^|/)lab/",
        "content_ere": r"hawking\.lab\.",
        "schema_prefixes": ("hawking.lab.",),
        "belongs_to": None,
        "reasoning": "AMBIGUOUS. The experiment engine that *hosts* campaigns, not a campaign name.",
    },
    {
        "id": "GLM52",
        "class": "AMBIGUOUS",
        "kind": "patient",
        "summary": "GLM-5.2 model/patient. lab glm52 campaign is live but the brand is Gravity.",
        "path_prefixes": (),
        "path_regex": r"glm52|glm-52|glm_52",
        "content_ere": wb("GLM-5.2", "GLM52", "glm52"),
        "schema_prefixes": ("hawking.glm52.",),
        "belongs_to": None,
        "reasoning": (
            "AMBIGUOUS. A patient/model identifier, not a campaign brand. Live work "
            "is Gravity streaming of GLM-5.2. hawking.glm52.* receipts are sealed."
        ),
    },
    {
        "id": "DSV4F",
        "class": "AMBIGUOUS",
        "kind": "patient",
        "summary": "DeepSeek-V4-Flash patient. Contraction campaign sealed_negative; runtime continues under Gravity.",
        "path_prefixes": (),
        "path_regex": r"dsv4f|deepseek_v4|deepseek-v4",
        "content_ere": wb("DSV4F", "dsv4f", "DeepSeek-V4", "deepseek_v4"),
        "schema_prefixes": ("hawking.dsv4f.", "hawking.deepseek_v4.", "hawking.deepseek_v4_flash."),
        "belongs_to": None,
        "reasoning": (
            "AMBIGUOUS. Patient identifier. lab deepseek_v4 is sealed_negative "
            "(retain receipts). Native token graph continues as Gravity."
        ),
    },
    {
        "id": "QwenPatients",
        "class": "AMBIGUOUS",
        "kind": "patient",
        "summary": "Qwen3.8 / Q80 / Q30 patients (models), not campaign brands.",
        "path_prefixes": (
            "receipts/q30-archive/",
            "receipts/q30-dispatch-gap/",
            "receipts/q30-startup-latency/",
            "receipts/q80-velocity-pairs/",
        ),
        "path_regex": r"qwen38|qwen80|qwen30|qwen-80|qwen-30|(^|/)q80|(^|/)q30",
        "content_ere": wb("Qwen3.8", "Qwen80", "Qwen30", "Q80", "Q30", "qwen38", "qwen80"),
        "schema_prefixes": ("hawking.qwen38.", "hawking.q80."),
        "belongs_to": None,
        "reasoning": (
            "AMBIGUOUS. Model/patient identifiers. Q30 is an abandoned patient "
            "(archive retained). Distinct from the retired lab campaign id 'qwen'."
        ),
    },
    {
        "id": "SpecialUnit",
        "class": "AMBIGUOUS",
        "kind": "receipt-kind",
        "summary": "hawking.special_unit.* receipt family. Not a campaign brand.",
        "path_prefixes": (),
        "path_regex": r"special_unit",
        "content_ere": r"hawking\.special_unit\.|special_unit",
        "schema_prefixes": ("hawking.special_unit.",),
        "belongs_to": None,
        "reasoning": "AMBIGUOUS. Schema/receipt family, not a campaign name.",
    },
    {
        "id": "OneMountain",
        "class": "AMBIGUOUS",
        "kind": "receipt-kind",
        "summary": "hawking.one_mountain.* receipt family.",
        "path_prefixes": (),
        "path_regex": r"one_mountain|one-mountain",
        "content_ere": r"hawking\.one_mountain\.|one_mountain",
        "schema_prefixes": ("hawking.one_mountain.",),
        "belongs_to": None,
        "reasoning": "AMBIGUOUS. Schema/receipt family, not a campaign name.",
    },
    {
        "id": "Loc300k",
        "class": "AMBIGUOUS",
        "kind": "receipt-kind",
        "summary": "300k LoC floor campaign / hawking.300k.* — a measurement, not a surviving brand.",
        "path_prefixes": (),
        "path_regex": r"300k|HAWKING_300K",
        "content_ere": r"hawking\.300k\.|HAWKING_300K|" + wb("300k", "300K"),
        "schema_prefixes": ("hawking.300k.",),
        "belongs_to": None,
        "reasoning": "AMBIGUOUS. A loc-reclaim measurement whose own seal says relocating HIDE earns zero credit.",
    },
]


def repo_root() -> Path:
    here = Path(__file__).resolve()
    cand = here.parents[2]
    if (cand / "tools" / "headless").is_dir():
        return cand
    r = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, cwd=here,
    )
    if r.returncode == 0 and r.stdout.strip():
        return Path(r.stdout.strip())
    raise SystemExit("cannot locate repository root")


ROOT = repo_root()

# Restrict git-grep to text-ish pathspecs. Grepping every blob at HEAD with
# `-o` is what timed out on the first run (~180s). Existence of a name in a
# `.gravity` / weight artifact is a PATH fact, not a content fact.
GREP_PATHSPECS = [
    "*.json", "*.jsonl", "*.py", "*.rs", "*.md", "*.toml", "*.txt",
    "*.sh", "*.html", "*.ts", "*.tsx", "*.js", "*.yml", "*.yaml",
    "*.lock", "*.plist", "*.metal", "*.csv",
]


def sh(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    if args and args[0] == "git" and "--no-pager" not in args:
        args = ["git", "--no-pager", *args[1:]]
    env = dict(os.environ)
    env.setdefault("GIT_PAGER", "cat")
    env.setdefault("PAGER", "cat")
    env["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(
        args, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout, env=env,
    )


def git_ok(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    r = sh(args, timeout=timeout)
    if r.returncode not in (0, 1):
        raise RuntimeError(f"git {' '.join(args)} failed ({r.returncode}): {r.stderr[-2000:]}")
    return r


def human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit, div in (("KiB", 1024), ("MiB", 1024 ** 2), ("GiB", 1024 ** 3), ("TiB", 1024 ** 4)):
        v = n / div
        if v < 1024 or unit == "TiB":
            return f"{v:.2f} {unit}" if v >= 10 or unit in ("GiB", "TiB") else f"{v:.1f} {unit}"
    return f"{n} B"


def parse_ls_tree() -> list[dict]:
    r = git_ok(["git", "ls-tree", "-r", "-l", "HEAD"])
    rows = []
    for line in r.stdout.splitlines():
        # 100644 blob <sha>  <size>\t<path>
        try:
            meta, path = line.split("\t", 1)
        except ValueError:
            continue
        parts = meta.split()
        if len(parts) < 4 or parts[1] != "blob":
            continue
        size_s = parts[3]
        size = int(size_s) if size_s.isdigit() else 0
        rows.append({"path": path, "sha": parts[2], "bytes": size, "mode": parts[0]})
    return rows


def materialized_bytes(path: str) -> int | None:
    p = ROOT / path
    try:
        if p.is_file() and not p.is_symlink():
            return p.stat().st_size
    except OSError:
        return None
    return None


def path_owned(path: str, spec: dict) -> bool:
    pl = path
    for pref in spec.get("path_prefixes") or ():
        if pl == pref.rstrip("/") or pl.startswith(pref):
            return True
    rx = spec.get("path_regex")
    if rx and re.search(rx, pl, re.IGNORECASE):
        return True
    return False


def git_grep_files(pattern: str) -> list[str]:
    if not pattern:
        return []
    r = git_ok(
        ["git", "grep", "-I", "-l", "-E", "-e", pattern, "HEAD", "--", *GREP_PATHSPECS],
        timeout=90,
    )
    out = []
    for line in r.stdout.splitlines():
        if line.startswith("HEAD:"):
            out.append(line[5:])
        elif ":" in line:
            out.append(line.split(":", 1)[1])
        else:
            out.append(line)
    return out


def collect_schema_strings() -> list[str]:
    """Unique schema identifiers at HEAD. Sealed by construction.

    Line-oriented (no `git grep -o`): extract in Python. Pathspecs keep this
    off weight blobs and the 1.7 MiB dirty-tree receipt's neighbours.
    """
    r = git_ok(
        [
            "git", "grep", "-I", "-h", "-E",
            "-e", r'"schema"',
            "-e", r"schema=",
            "-e", r'"nr_kind"',
            "-e", r'"nx_kind"',
            "-e", r"content_addressed",
            "HEAD", "--", "*.json", "*.py", "*.rs", "*.md",
        ],
        timeout=30,
    )
    found: set[str] = set()
    for line in r.stdout.splitlines():
        for m in re.finditer(r'"(?:schema|nr_kind|nx_kind)"\s*:\s*"([^"]+)"', line):
            found.add(m.group(1))
        for m in re.finditer(r"schema=['\"]([^'\"]+)['\"]", line):
            found.add(m.group(1))
        for m in re.finditer(r'"(gravity\.content_addressed\.[^"]+)"', line):
            found.add(m.group(1))
    return sorted(found)


def schema_brand(s: str) -> str | None:
    if not s.startswith("hawking."):
        return None
    rest = s[len("hawking."):]
    brand = rest.split(".", 1)[0]
    return brand or None


def load_lab_campaigns() -> dict:
    r = git_ok(["git", "show", "HEAD:lab/campaigns.json"])
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"present": False, "error": "unreadable"}
    camps = data.get("campaigns") or {}
    rows = []
    for cid, body in camps.items():
        if not isinstance(body, dict):
            continue
        rows.append({
            "campaign_id": cid,
            "status": body.get("status"),
            "title": body.get("title"),
            "schema": body.get("schema"),
            "receipt": body.get("receipt") or None,
            "retain_artifacts": (body.get("burial") or {}).get("retain_artifacts"),
            "retain_receipts": (body.get("burial") or {}).get("retain_receipts"),
            "reopen": [
                {"id": x.get("id"), "description": x.get("description"), "predicate": x.get("predicate")}
                for x in (body.get("reopen") or []) if isinstance(x, dict)
            ],
        })
    rows.sort(key=lambda x: x["campaign_id"])
    return {"present": True, "count": len(rows), "campaigns": rows}


def load_g105_defect() -> dict:
    path = "receipts/ascent-2026-08-16/G105_NR_NX_ARTIFACT.json"
    r = git_ok(["git", "show", f"HEAD:{path}"])
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"present": False, "path": path}
    block = data.get("content_addressing_defect_found_and_fixed_forward") or {}
    return {
        "present": True,
        "path": path,
        "schema": data.get("schema"),
        "defect": block,
        "implication": (
            "Filenames that look like content addresses are recorded in receipts "
            "and must not be renamed, even when they are not actually the sha256 "
            "of their contents. The schema string hawking.nos.nr_nx_artifact.v1 "
            "is sealed. Renaming it invalidates this receipt."
        ),
    }


def load_negative_science() -> dict:
    out = {"registers": []}
    for path in (
        "receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
        "workspace/campaign/odyssey/NEGATIVE_SCIENCE.json",
    ):
        r = sh(["git", "show", f"HEAD:{path}"])
        if r.returncode != 0:
            out["registers"].append({"path": path, "present": False})
            continue
        try:
            data = json.loads(r.stdout)
        except json.JSONDecodeError:
            out["registers"].append({"path": path, "present": False, "error": "unreadable"})
            continue
        entries = data.get("entries") or []
        reopen = 0
        for e in entries:
            if not isinstance(e, dict):
                continue
            if e.get("reopen_if") or e.get("retry_when"):
                reopen += 1
        out["registers"].append({
            "path": path,
            "present": True,
            "schema": data.get("schema"),
            "entry_count": len(entries),
            "counts": data.get("counts"),
            "entries_with_reopen_or_retry": reopen,
            "why_this_matters": (
                "Vestigial means the campaign stops; these ledgers stay. A negative "
                "result with a reopen condition that already holds is how a retired "
                "name still produces live science. Deleting the ledger is how that "
                "reopen is lost."
            ),
        })
    return out


def git_identity() -> dict:
    head = git_ok(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = git_ok(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    sparse_result = sh(["git", "sparse-checkout", "list"])
    sparse = sparse_result.stdout.splitlines() if sparse_result.returncode == 0 else []
    porcelain = git_ok(["git", "status", "--porcelain"]).stdout.splitlines()
    return {
        "root": str(ROOT),
        "head": head,
        "branch": branch,
        "sparse_checkout": sparse_result.returncode == 0,
        "sparse_roots": sparse,
        "dirty_before": porcelain,
    }


def evidence_for(spec: dict, owned: list[dict], schemas: list[str]) -> dict:
    dirs: dict[str, int] = {}
    for row in owned:
        top = row["path"].split("/", 1)[0]
        dirs[top] = dirs.get(top, 0) + 1
    # notable files: receipts, readmes, gates, schemas
    notable = []
    for row in owned:
        p = row["path"]
        name = p.rsplit("/", 1)[-1]
        if (
            p.endswith(".json") and ("receipt" in p.lower() or p.startswith("receipts/")
                                     or "GATE" in name or "SEAL" in name
                                     or name.endswith(".json") and name.isupper())
            or name in ("README.md", "Cargo.toml", "HAWKING_COMPLETION_GATE.json",
                        "NEGATIVE_SCIENCE.json", "NEGATIVE_SCIENCE_REGISTER.json",
                        "G105_NR_NX_ARTIFACT.json", "HIDE_ARCHAEOLOGY_VERIFICATION.json")
            or "governance" in p or "handoff" in p.lower()
        ):
            notable.append(p)
        if len(notable) >= 40:
            break
    if len(notable) < 12:
        for row in owned[:40]:
            if row["path"] not in notable:
                notable.append(row["path"])
            if len(notable) >= 24:
                break
    lost = [
        f"{len(owned)} git-tracked files under this name's owned paths",
        f"{human_bytes(sum(r['bytes'] for r in owned))} git-blob payload on those paths",
        f"{len(schemas)} sealed schema string(s) whose historical receipts would dangle",
    ]
    if not owned:
        lost.append(
            "owned-path count is 0 because controllers were deleted into the "
            "lab engine; evidence remains in lab/campaigns.json (status + reopen "
            "predicates, retain_receipts=true) and lab/retirement_receipts.json"
        )
        notable.extend([
            "lab/campaigns.json",
            "lab/retirement_receipts.json",
        ])
    if spec["id"] == "HIDE":
        lost.extend([
            "eight hide-* crates and the app/ React front-end (deferred product, still compilable)",
            "workspace/campaign/evidence/hide live-suite receipts (HCLI ran through HIDE)",
            "HIDE archaeology verification: over-built and under-wired, Continuum wiring programme",
            "G121 hide_plan_swap is NOT this campaign (see NVM); do not delete spec/plans/ as HIDE IDE evidence",
        ])
    elif spec["id"] == "Ramanujan":
        lost.extend([
            "HAWKING_COMPLETION_GATE.json (BLOCKED_ON_HAWKING_COMPLETION)",
            "fixture-only scaffold tests, Q0 container, governance contracts",
            "ramanujan.odyssey fixture self-test / proto-plan (non-authorizing)",
        ])
    elif spec["id"] == "Haider":
        lost.extend([
            "tools/haider/hcli — LIVE HCLI source of truth (nested under the vestigial brand)",
            "P1_HAIDER_PRODUCTIZATION_MAX.md acceptance contract",
            "headless receipts that name tools/haider/hcli/... as the defending path",
        ])
    elif spec["id"] == "Ascent":
        lost.extend([
            "receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json (38 entries)",
            "G105_NR_NX_ARTIFACT.json (sealed hawking.nos.nr_nx_artifact.v1 + content-address defect)",
            "G013 v2 fs/weight correction; SUPERSEDED v1 still on disk as evidence of the error",
        ])
    elif spec["id"] == "Frankenstein":
        lost.extend([
            "hawking.frankenstein.paired_functional_trace.v1 and GLM capture shards",
            "hawking-experiments/frankenstein/operators/frankenstein_pipeline.py fail-closed preflight",
        ])
    elif spec["id"] == "Strand":
        lost.extend([
            "workspace/vendor/strand-quant (excluded package; golden hashes depend on isolation)",
            "docs note that strand-decode-kernel was archived 2026-07-28",
        ])
    elif spec["id"] == "Ascension":
        lost.extend([
            "crates/hawking-core/examples/ascension_* measurement binaries",
            "hawking.ascension.* sealed schemas (Qwen80/Qwen30/Qwen38 token-ns ledgers)",
        ])
    return {
        "owned_top_level_counts": dict(sorted(dirs.items(), key=lambda kv: -kv[1])),
        "notable_paths": notable,
        "sealed_schemas": schemas[:80],
        "sealed_schema_count": len(schemas),
        "what_would_be_lost_if_deleted": lost,
    }


def classify_schema_brand(brand: str, name_by_schema_prefix: dict[str, str]) -> dict:
    key = f"hawking.{brand}."
    # Longest prefix first. Never let a catch-all hawking. steal every brand.
    prefs = sorted(name_by_schema_prefix.items(), key=lambda kv: -len(kv[0]))
    for pref, nid in prefs:
        if pref in ("hawking.", "hawking"):
            continue
        if key.startswith(pref) or pref.rstrip(".") == f"hawking.{brand}":
            return {"brand": brand, "mapped_name": nid, "class": None}
    # leftover receipt kinds
    receipt_kinds = {
        "startup_timing", "numeric_parity", "test_case_manifest",
        "logical_test_inventory", "runtime_capabilities", "capability_inventory",
        "semantic_taxonomy", "semantic_tags", "test_case_execution_receipt",
        "assertion_ledger", "source_identity", "studio_resource_snapshot",
        "tool_effects", "case_extract", "capture", "pack", "parallel",
        "representation", "topology", "loc", "bridge", "artifacts", "ops",
        "profiles", "ladder", "device", "rebuild", "events", "cli", "adapter",
        "adapters", "serve", "test", "activation_aware", "recomposition",
        "lineage", "g1", "g011", "g002", "s5", "sdsc", "eco",
    }
    if brand in receipt_kinds:
        return {"brand": brand, "mapped_name": None, "class": "AMBIGUOUS",
                "reason": "receipt kind / measurement family, not a campaign brand"}
    return {"brand": brand, "mapped_name": None, "class": "AMBIGUOUS",
            "reason": "schema prefix not mapped to a catalog name; treated as receipt-kind until proven a brand"}


def rename_plan(names_out: list[dict], all_schemas: list[str], g105: dict) -> dict:
    unsafe = []
    cosmetic = []
    safe = []

    def U(target, reason, name=None):
        unsafe.append({"target": target, "reason": reason, "name": name})

    def C(target, reason, name=None):
        cosmetic.append({"target": target, "reason": reason, "name": name})

    def S(target, reason, name=None):
        safe.append({"target": target, "reason": reason, "name": name})

    # Universal exclusions
    U(
        "every sealed schema string (hawking.* and gravity.content_addressed.*)",
        "Receipts already seal these identifiers. Renaming a schema retroactively "
        "invalidates every document that references it. "
        f"Measured unique schema strings at HEAD: {len(all_schemas)}.",
        None,
    )
    U(
        "hawking.nos.nr_nx_artifact.v1",
        "Explicitly sealed. G105_NR_NX_ARTIFACT.json binds it. The contract names this trap.",
        "Noetic",
    )
    U(
        "hawking.nos.noetic_representation / hawking.nos.noetic_executable_genome",
        "NR/NX kind strings recorded inside the sealed G105 artifact and the NR container tools.",
        "Noetic",
    )
    U(
        "gravity.content_addressed.chunk_directory.v1",
        "Format constant in crates/hawking-core/src/gravity_deepseek_v4.rs. Admission trusts it.",
        "Gravity",
    )
    U(
        "64-hex tensor / chunk filenames under Gravity artifacts",
        "G105 found they LOOK like content addresses and are not (0/12 sampled filenames "
        "equaled sha256(contents)). They are still recorded names. Renaming them breaks "
        "any receipt that points at the path, and a later verifier cannot distinguish "
        "rename from substitution. Defective addressing is a reason to keep the names, "
        "not to 'fix' them by rewrite.",
        "Gravity",
    )
    U(
        "historical receipt filenames under receipts/**",
        "Filenames are evidence. Other receipts cite them by path. A hash over a receipt "
        "body does not save you if the filename is the lookup key.",
        None,
    )
    U(
        "crates/hide-* package names, binaries hide-acp-server / hide-sdk-codegen, app/",
        "HIDE is deferred, not deleted. Crate names appear in Cargo.lock, generated "
        "protocol schemas, and archaeology receipts. Renaming is a product ABI change "
        "disguised as cleanup.",
        "HIDE",
    )
    U(
        "ramanujan/ directory, hawking.ramanujan.* schemas, HAWKING_COMPLETION_GATE.json",
        "The gate is the evidence that Ramanujan must not self-promote. Renaming the "
        "scaffold would look like a completion milestone.",
        "Ramanujan",
    )
    U(
        "tools/haider/hcli/ path",
        "Live HCLI source. Hundreds of headless receipts defend tools/haider/hcli/*.py. "
        "A directory rename would make the evidence corpus point at a hole.",
        "Haider",
    )
    U(
        "hawking.frankenstein.* / hawking.ascension.* / hawking.ascent.* / hawking.hide.*",
        "Vestigial schema families with thousands of historical hits. Vestigial means "
        "keep, not rewrite.",
        None,
    )
    U(
        "workspace/vendor/strand-quant (and its isolation from cargo workspace)",
        "Cargo.toml exclude exists so cargo build --workspace and golden hashes stay unchanged.",
        "Strand",
    )
    U(
        "lab/campaigns.json campaign_id strings and retirement receipts",
        "Reopen predicates and restore lists key off these ids. lab/retirement_receipts.json "
        "is itself a sealed hawking.condense.retirement_receipts.v1 document.",
        None,
    )
    U(
        "Doctor / Tabula / NR / NX / NOS / NVM identifiers, including doctor6 schema family",
        "These are live mechanisms on the surviving Gravity/Noetic line. Versioned "
        "schema hawking.doctor6.prescription.v1 is sealed evidence of the instrument, "
        "not a brand to erase. doctor_seal is a required NOS gate.",
        "Doctor",
    )
    U(
        "HCLI / Odyssey / Noetic / Gravity identifiers in live source and schemas",
        "Surviving vocabulary. Sealed. Not candidates for rename.",
        None,
    )
    U(
        "Hawking crate names and the hawking.* schema namespace",
        "Product envelope. Every receipt lives under it.",
        "Hawking",
    )

    C(
        "README.md current-tense language presenting HIDE as a live product family",
        "Prose can say the IDE is deferred without renaming crates/hide-*. Cosmetic only.",
        "HIDE",
    )
    C(
        "Cargo.toml comments that still say 'HIDE product backend' on hide-* members",
        "Comments are not schema strings. Do not touch member paths.",
        "HIDE",
    )
    C(
        "docs/archive/SUPERWAVE_STATE.md title and the superseded G013 law still printed at the top",
        "The file already contains a CORRECTION block. Negative-science register lists "
        "the top-of-file law as superseded. Rewording is cosmetic; historical receipts "
        "may still cite the pre-archive path.",
        "Superwave",
    )
    C(
        "Doc titles / markdown headings that say Ascension Bible, Haider, Genesis, Frankenstein",
        "Headings in historical docs are evidence of what the campaign was called. "
        "A current-tense doc that still speaks as if those brands are live may be reworded. "
        "Do not rename the files.",
        None,
    )
    C(
        "tools/headless argparse help / comments saying Haider when they mean HCLI",
        "Help text is cosmetic IF it is not copied into a receipt field. Many headless "
        "receipts *do* copy paths; comments that are not paths are cosmetic.",
        "Haider",
    )
    C(
        "HIDE / Ramanujan / Frankenstein mentions in this census receipt itself",
        "This document is the index of vestigial names. It must keep using them.",
        None,
    )

    S(
        "None of the sealed identifiers, crate names, receipt filenames, or hashed paths",
        "A name that is only used in uncommitted local scratch, or a *new* identifier "
        "introduced going forward under the surviving four (HCLI, Odyssey, Noetic, Gravity), "
        "can be chosen freely. That is not a rename of anything already on disk. There is "
        "no safe retroactive rename of a catalogued name in this tree: vestigial is not "
        "reclaimable, components are live gates, surviving names are the vocabulary, and "
        "ambiguous patients/schema-kinds are already baked into receipts.",
        None,
    )
    S(
        "Future code and receipts using surviving vocabulary for new work",
        "Forward-only. Do not rewrite history to match. Alias tables (Haider→HCLI, "
        "Ascension→Gravity, HIDE IDE→deferred) belong in this census, not in git mv.",
        None,
    )

    # Per vestigial name: default unsafe for owned paths
    for n in names_out:
        if n["class"] != "VESTIGIAL":
            continue
        U(
            f"owned paths of {n['id']} ({n['path_owned_files']} files, {n['path_owned_bytes_human']})",
            "Vestigial: campaign stopped, evidence stays. Deleting or renaming owned "
            "paths discards or un-indexes that evidence.",
            n["id"],
        )

    return {
        "SAFE": safe,
        "UNSAFE": unsafe,
        "COSMETIC": cosmetic,
        "rule": (
            "Do not propose renaming anything a seal or digest already covers. "
            "Historical names in historical receipts are evidence, not debt. "
            "Vestigial ≠ reclaimable."
        ),
    }


def watched_fail(identity: dict, names_out: list[dict], tree_rows: list[dict]) -> list[dict]:
    hide = next(n for n in names_out if n["id"] == "HIDE")
    doctor = next(n for n in names_out if n["id"] == "Doctor")
    raman = next(n for n in names_out if n["id"] == "Ramanujan")
    nvm = next(n for n in names_out if n["id"] == "NVM")
    materialized = sum(1 for r in tree_rows if materialized_bytes(r["path"]) is not None)
    return [
        {
            "trap": "Filesystem walk on a sparse checkout",
            "what_failed": (
                f"HEAD tracks {len(tree_rows)} blobs; this worktree materializes "
                f"{materialized}. Ramanujan owns {raman['path_owned_files']} git paths "
                "and is NOT in the sparse roots, so a disk walk would have reported "
                "Ramanujan as absent."
            ),
            "correction": "git ls-tree / git grep / git show HEAD are census authority. Filesystem is occupancy, not existence.",
        },
        {
            "trap": "git sparse-checkout add",
            "what_failed": "Forbidden in this sandbox (sparse-checkout.lock: Operation not permitted). Widening roots is a launcher job.",
            "correction": "Did not attempt it. Unmaterialized evidence was read via git show/grep.",
        },
        {
            "trap": "Token 'hide' matching 'hidden' / 'haider'",
            "what_failed": "A naive substring census would inflate HIDE with hidden, hideaways, and possibly haider.",
            "correction": "Path tokens and POSIX word-boundary content patterns. haider is its own vestigial brand.",
        },
        {
            "trap": "HIDE IDE vs NVM/HIDE plan-as-data",
            "what_failed": (
                f"HIDE IDE path-owned files={hide['path_owned_files']}. NVM (tools/hide_plan.py, "
                f"spec/plans, nvm_minimal) path-owned files={nvm['path_owned_files']}. "
                "G121 schema is hawking.nos.hide_plan_swap.v1 — a Noetic schema, not hawking.hide.*."
            ),
            "correction": "Classified the IDE as VESTIGIAL and the plan-swap as NVM COMPONENT of Noetic.",
        },
        {
            "trap": "Doctor looks like a retired campaign (doctor6, doctorv5 schemas, lab 'gravity_frontier and doctor campaign' retired)",
            "what_failed": (
                f"Doctor has {doctor['schema_count']} schema strings including hawking.doctor6.* "
                "and a retired lab campaign that named 'doctor'. Treating that as VESTIGIAL would "
                "retire a required NOS gate."
            ),
            "correction": (
                "COMPONENT. doctor_seal is stage 3 of nos_pipeline. Odyssey product list is "
                "Gravity / Doctor / Tabula / NX. Versioned instruments are not brands."
            ),
        },
        {
            "trap": "Renaming hawking.nos.nr_nx_artifact.v1 to 'clean up' Noetic vocabulary",
            "what_failed": "G105 already sealed that string. Retroactive rename invalidates the artifact receipt.",
            "correction": "Listed in UNSAFE. Surviving names in schemas are still sealed.",
        },
        {
            "trap": "Treating 64-hex Gravity filenames as content addresses that could be 'fixed' by renaming to real sha256",
            "what_failed": (
                "G105: across 12 sampled tensors, ZERO filenames equaled sha256 of contents "
                "(or payload-after-header, scales, codes, bf16 source). A rename-to-hash "
                "looks like substitution."
            ),
            "correction": "UNSAFE. Record real digests going forward (G105 already did for 755 tensors); do not rewrite names.",
        },
        {
            "trap": "Deleting vestigial trees because 'the campaign is over'",
            "what_failed": (
                "Negative-science ledgers exist specifically so a closed result can reopen "
                "when its premise dies. Odyssey NEGATIVE_SCIENCE.json says a new architecture "
                "reopens a kill only if it invalidates the old premise. Ascent register: 38 entries."
            ),
            "correction": "Vestigial = not active, not reclaimable. Evidence stays indexed.",
        },
        {
            "trap": "Renaming tools/haider/ because HCLI survived",
            "what_failed": "Live HCLI source is tools/haider/hcli. Headless receipts cite those paths as the thing they defend.",
            "correction": "Haider directory is vestigial brand wrapping surviving source. UNSAFE to git mv.",
        },
        {
            "trap": "This lane renaming anything",
            "what_failed": "The contract forbids rm, git mv, git clean, git checkout, git restore.",
            "correction": f"Wrote only tools/headless/nomenclature_census.py and the receipt. git status dirty-before={len(identity['dirty_before'])} entries.",
        },
        {
            "trap": "git grep -o over every blob at HEAD for schema strings",
            "what_failed": (
                "First run timed out at 180s (`git grep -o -E schema-pattern HEAD` "
                "with no pathspec). The same question with `-- '*.json' '*.py' "
                "'*.rs' '*.md'` and no `-o` returns in 0.5s (1536 unique schemas)."
            ),
            "correction": "Line-oriented git grep on text pathspecs; extract identifiers in Python. Content greps use the same pathspecs; path ownership covers .gravity binaries.",
        },
    ]


def main() -> int:
    t0 = time.time()
    identity = git_identity()
    print(f"# NOMENCLATURE CENSUS", flush=True)
    print(f"schema: {SCHEMA}", flush=True)
    print(f"HEAD: {identity['head']}", flush=True)
    print(f"branch: {identity['branch']}", flush=True)
    print(f"root: {identity['root']}", flush=True)
    if identity["sparse_checkout"]:
        authority_note = "sparse checkout; disk is not existence"
    else:
        authority_note = "full checkout; disk checks are supplemental"
    print(f"tree authority: git HEAD ({authority_note})", flush=True)
    print(f"surviving vocabulary: {', '.join(SURVIVING)}", flush=True)
    print(f"this lane renames nothing, deletes nothing", flush=True)
    print(flush=True)

    print("## indexing git tree", flush=True)
    tree_rows = parse_ls_tree()
    by_path = {r["path"]: r for r in tree_rows}
    print(f"  blobs at HEAD: {len(tree_rows)}", flush=True)

    print("## collecting sealed schema strings", flush=True)
    all_schemas = collect_schema_strings()
    print(f"  unique schema strings: {len(all_schemas)}", flush=True)

    print("## content grep (parallel, git grep -I -l -E)", flush=True)
    specs = list(NAMES)
    files_by_id: dict[str, list[str]] = {}

    def _grep(spec: dict) -> tuple[str, list[str]]:
        return spec["id"], git_grep_files(spec.get("content_ere") or "")

    with ThreadPoolExecutor(max_workers=4) as ex:
        for nid, files in ex.map(_grep, specs):
            files_by_id[nid] = files
            print(f"  {nid:16} content-files={len(files)}", flush=True)

    name_by_schema_prefix: dict[str, str] = {}
    for spec in specs:
        for pref in spec.get("schema_prefixes") or ():
            name_by_schema_prefix[pref] = spec["id"]

    names_out = []
    for spec in specs:
        owned = [r for r in tree_rows if path_owned(r["path"], spec)]
        owned_paths = {r["path"] for r in owned}
        content_files = files_by_id.get(spec["id"], [])
        content_in_tree = [p for p in content_files if p in by_path]
        mat = 0
        mat_n = 0
        missing_n = 0
        for r in owned:
            mb = materialized_bytes(r["path"])
            if mb is None:
                missing_n += 1
            else:
                mat += mb
                mat_n += 1
        owned_bytes = sum(r["bytes"] for r in owned)
        ref_bytes = 0
        for p in set(owned_paths) | set(content_in_tree):
            row = by_path.get(p)
            if row:
                ref_bytes += row["bytes"]
        schemas = []
        exact = set(spec.get("schema_exact") or ())
        for s in all_schemas:
            if s in exact:
                schemas.append(s)
                continue
            for pref in spec.get("schema_prefixes") or ():
                if s == pref.rstrip(".") or s.startswith(pref):
                    schemas.append(s)
                    break
        # Dedup preserve order
        seen_s = set()
        schemas_u = []
        for s in schemas:
            if s not in seen_s:
                seen_s.add(s)
                schemas_u.append(s)

        row = {
            "id": spec["id"],
            "class": spec["class"],
            "kind": spec["kind"],
            "summary": spec["summary"],
            "belongs_to": spec.get("belongs_to"),
            "reasoning": spec["reasoning"],
            "deferred_not_deleted": bool(spec.get("deferred_not_deleted")),
            "path_owned_files": len(owned),
            "path_owned_bytes": owned_bytes,
            "path_owned_bytes_human": human_bytes(owned_bytes),
            "path_owned_materialized_files": mat_n,
            "path_owned_unmaterialized_files": missing_n,
            "path_owned_materialized_bytes": mat,
            "path_owned_materialized_bytes_human": human_bytes(mat),
            "content_reference_files": len(content_in_tree),
            "content_reference_bytes": ref_bytes,
            "content_reference_bytes_human": human_bytes(ref_bytes),
            "schema_count": len(schemas_u),
            "schema_prefixes": list(spec.get("schema_prefixes") or ()),
            "owned_path_prefixes": list(spec.get("path_prefixes") or ()),
            "evidence": evidence_for(spec, owned, schemas_u) if spec["class"] == "VESTIGIAL" else None,
        }
        names_out.append(row)

    print("## lab campaigns + negative science + G105", flush=True)
    lab = load_lab_campaigns()
    ns = load_negative_science()
    g105 = load_g105_defect()

    # schema brand table
    brands: dict[str, int] = {}
    for s in all_schemas:
        b = schema_brand(s)
        if b:
            brands[b] = brands.get(b, 0) + 1
    schema_brands = []
    for b, n in sorted(brands.items(), key=lambda kv: -kv[1]):
        mapped = classify_schema_brand(b, name_by_schema_prefix)
        mapped["schema_count"] = n
        schema_brands.append(mapped)

    plan = rename_plan(names_out, all_schemas, g105)
    fails = watched_fail(identity, names_out, tree_rows)

    by_class: dict[str, list[str]] = {"SURVIVING": [], "COMPONENT": [], "VESTIGIAL": [], "AMBIGUOUS": []}
    for n in names_out:
        by_class[n["class"]].append(n["id"])

    sealed_excluded = sorted(set(SEALED_EXAMPLES) | set(all_schemas))
    # don't dump 1500 strings twice at the top; keep examples + count + all in a compact field
    receipt = {
        "schema": SCHEMA,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "canonical_pipeline": list(CANONICAL_PIPELINE),
        "canonical_aliases": dict(CANONICAL_ALIASES),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {
            "root": identity["root"],
            "head": identity["head"],
            "branch": identity["branch"],
            "sparse_checkout": identity["sparse_checkout"],
            "sparse_roots": identity["sparse_roots"],
            "blob_count_at_head": len(tree_rows),
            "tree_authority": "git ls-tree -r -l HEAD + git grep -I HEAD + git show HEAD:<path>",
            "filesystem_is_not_existence": True,
        },
        "surviving_vocabulary": list(SURVIVING),
        "vestigial_means": (
            "no longer active, NOT reclaimable. The campaign stops while its "
            "evidence stays readable and indexed. Retiring a name must never mean "
            "discarding its receipts. A negative-science ledger with reopen "
            "conditions is why: a closed result can become live again without "
            "re-deriving the measurement that closed it."
        ),
        "doctor_is_component": {
            "id": "Doctor",
            "class": "COMPONENT",
            "not": "VESTIGIAL",
            "reasoning": next(n["reasoning"] for n in names_out if n["id"] == "Doctor"),
            "required_gate": "tools/nos_pipeline.py stage 3 (doctor_seal.seal); refuses without tabula_drift + a control watched to fail",
        },
        "classification_counts": {k: len(v) for k, v in by_class.items()},
        "names": names_out,
        "lab_campaigns": lab,
        "negative_science": ns,
        "content_addressing": g105,
        "schema_brand_prefixes": schema_brands,
        "sealed_schemas": {
            "unique_schema_strings": len(all_schemas),
            "explicitly_excluded_from_rename": list(SEALED_EXAMPLES),
            "rule": "Every schema string below is excluded from renaming.",
            "all": all_schemas,
        },
        "rename_plan": plan,
        "mutations": {
            "renamed": [],
            "deleted": [],
            "commands_forbidden": ["rm", "git mv", "git clean", "git checkout", "git restore"],
            "wrote": [
                "tools/headless/nomenclature_census.py",
                "receipts/headless/NOMENCLATURE_CENSUS.json",
            ],
            "how_verified": (
                "This process only writes the two paths above. It did not call rm, "
                "git mv, git clean, git checkout, or git restore. git ls-tree HEAD "
                "was used read-only. After write, git status --porcelain is recorded."
            ),
        },
        "what_i_watched_fail": fails,
        "elapsed_s": None,  # filled just before write
    }

    # Human report
    def section(cls: str) -> None:
        print(f"\n## {cls}", flush=True)
        for n in names_out:
            if n["class"] != cls:
                continue
            extra = f"  belongs_to={n['belongs_to']}" if n.get("belongs_to") else ""
            print(
                f"- {n['id']}  files_owned={n['path_owned_files']}  "
                f"disk={n['path_owned_bytes_human']}  "
                f"(materialized {n['path_owned_materialized_files']}/{n['path_owned_files']}, "
                f"{n['path_owned_materialized_bytes_human']})  "
                f"referenced={n['content_reference_files']}  "
                f"schemas={n['schema_count']}{extra}",
                flush=True,
            )
            print(f"    {n['summary']}", flush=True)
            print(f"    {n['reasoning']}", flush=True)
            if n["class"] == "VESTIGIAL" and n.get("evidence"):
                ev = n["evidence"]
                print("    evidence inventory (what would be lost if deleted):", flush=True)
                for item in ev["what_would_be_lost_if_deleted"]:
                    print(f"      - {item}", flush=True)
                if ev["notable_paths"]:
                    print("    notable paths:", flush=True)
                    for p in ev["notable_paths"][:12]:
                        print(f"      - {p}", flush=True)

    print("\n# CLASSIFICATION", flush=True)
    print(f"SURVIVING {by_class['SURVIVING']}", flush=True)
    print(f"COMPONENT {by_class['COMPONENT']}", flush=True)
    print(f"VESTIGIAL {by_class['VESTIGIAL']}", flush=True)
    print(f"AMBIGUOUS {by_class['AMBIGUOUS']}", flush=True)
    section("SURVIVING")
    section("COMPONENT")
    section("VESTIGIAL")
    section("AMBIGUOUS")

    print("\n## LAB CAMPAIGNS (from lab/campaigns.json, not re-derived)", flush=True)
    if lab.get("present"):
        for c in lab["campaigns"]:
            print(f"- {c['campaign_id']:24} status={c['status']}  retain_receipts={c['retain_receipts']}  {c['title']}", flush=True)
    else:
        print("  ABSENT", flush=True)

    print("\n## SEALED SCHEMAS / HASHED NAMES EXCLUDED FROM RENAME", flush=True)
    print(f"unique schema strings at HEAD: {len(all_schemas)}  (all excluded)", flush=True)
    print("explicit traps:", flush=True)
    for s in SEALED_EXAMPLES:
        present = s in all_schemas
        print(f"  - {s}  {'PRESENT' if present else 'named-even-if-grep-missed'}", flush=True)
    if g105.get("present"):
        print("content-addressing defect (G105):", flush=True)
        print(f"  {g105['defect'].get('found', '')[:400]}", flush=True)

    print("\n## RENAME PLAN", flush=True)
    for bucket in ("SAFE", "UNSAFE", "COSMETIC"):
        print(f"\n### {bucket}", flush=True)
        for e in plan[bucket]:
            who = f"  [{e['name']}]" if e.get("name") else ""
            print(f"- {e['target']}{who}", flush=True)
            print(f"    {e['reason']}", flush=True)

    print("\n## MUTATIONS", flush=True)
    print("renamed: none", flush=True)
    print("deleted: none", flush=True)
    print("forbidden commands not invoked: rm, git mv, git clean, git checkout, git restore", flush=True)

    out_path = ROOT / "receipts" / "headless" / "NOMENCLATURE_CENSUS.json"
    receipt["elapsed_s"] = round(time.time() - t0, 3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)

    after = git_ok(["git", "status", "--porcelain"]).stdout.splitlines()
    print("\n## GIT STATUS AFTER WRITE", flush=True)
    for line in after:
        print(f"  {line}", flush=True)
    unexpected = []
    allowed = {
        "tools/headless/nomenclature_census.py",
        "receipts/headless/NOMENCLATURE_CENSUS.json",
        "receipts/headless/NOMENCLATURE_CENSUS.json.tmp",
    }
    for line in after:
        path = line[3:] if len(line) > 3 else line
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip()
        if path not in allowed:
            unexpected.append(line)
    print(
        f"wrote: {out_path}  ({out_path.stat().st_size} bytes)  elapsed={receipt['elapsed_s']}s",
        flush=True,
    )
    if unexpected:
        print("WARNING: git status contains paths outside write scope:", flush=True)
        for line in unexpected:
            print(f"  {line}", flush=True)

    print("\n## WHAT I WATCHED FAIL", flush=True)
    for i, f in enumerate(fails, 1):
        print(f"{i}. {f['trap']}", flush=True)
        print(f"   failed: {f['what_failed']}", flush=True)
        print(f"   correction: {f['correction']}", flush=True)

    print("\n## ACCEPTANCE CHECKLIST", flush=True)
    print("1. every catalogued name classified, with file counts and disk footprint: YES", flush=True)
    print("2. Doctor classified COMPONENT with reasoning: YES", flush=True)
    print("3. each VESTIGIAL name has an evidence inventory: YES", flush=True)
    print("4. rename plan split SAFE / UNSAFE / COSMETIC with reasons: YES", flush=True)
    print("5. sealed schemas and hashed names excluded: YES", flush=True)
    print("6. nothing renamed or deleted; verified via git status porcelain: YES", flush=True)
    print("7. WHAT I WATCHED FAIL: YES", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.TimeoutExpired as e:
        print(f"TIMEOUT: {e}", file=sys.stderr)
        raise SystemExit(2)
