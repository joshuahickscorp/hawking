#!/usr/bin/env python3
"""Integrity-checked Genesis system contract and prompt compiler.

The full user directive is canonical in-repo. Resident Qwen3.8 has a 4096-token
context, so runtime sessions receive a concise binding capsule which names and
hashes that authority instead of duplicating the full document into every task.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_RELATIVE_PATH = Path(
    "contracts/genesis/QWEN38_GENESIS_SYSTEM_DIRECTIVE.md"
)
CANONICAL_PATH = REPO_ROOT / CANONICAL_RELATIVE_PATH
EXPECTED_SHA256 = "881ae469e0287cf386467002d3fc7951524b47054ac6d7f753b94a8e4e3ceff7"
EXPECTED_BYTES = 16_414
SOURCE_PROVENANCE = (
    "user-supplied Codex attachment "
    "295d355f-5534-4488-a048-a52611c03b14/pasted-text.txt"
)
CONTINUITY_RELATIVE_PATH = Path("contracts/genesis/GENESIS_CONTINUITY_DIRECTIVE.md")
CONTINUITY_PATH = REPO_ROOT / CONTINUITY_RELATIVE_PATH
CONTINUITY_SHA256 = "c4a58bc06575effb8f759dbb22c49abfc65e1957910b18917d45d02592d1fdbc"
CONTINUITY_BYTES = 11_912
CONTINUITY_PROVENANCE = (
    "user-supplied Codex attachment "
    "8965050b-9273-4aa7-9ef4-c473b23f6848/pasted-text.txt"
)
OUTPUT_LAW_RELATIVE_PATH = Path("contracts/genesis/GENESIS_OUTPUT_LAW.md")
OUTPUT_LAW_PATH = REPO_ROOT / OUTPUT_LAW_RELATIVE_PATH
OUTPUT_LAW_SHA256 = "9679490e8ae623a6fdb408fd906a15d676bc55926580f6d7ed60e9ea610c9ada"
OUTPUT_LAW_BYTES = 4_871
OUTPUT_LAW_PROVENANCE = "user-supplied /sw directive, session d51d4904 2026-08-17"
RESIDENT_SESSION_ROLES = ("parent", "child_a", "child_b", "protected_test")
CAPSULE_BEGIN = "<GENESIS_SYSTEM_CONTRACT_QWEN38_V1>"
CAPSULE_END = "</GENESIS_SYSTEM_CONTRACT_QWEN38_V1>"


class GenesisContractError(RuntimeError):
    """A canonical Genesis authority is missing, malformed, or tampered."""


@dataclass(frozen=True)
class GenesisContract:
    path: Path
    relative_path: str
    text: str
    sha256: str
    size_bytes: int
    source_provenance: str
    heading: str

    def provenance(self) -> dict[str, Any]:
        return {
            "canonical_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "source_provenance": self.source_provenance,
            "integrity_verified": True,
        }


@dataclass(frozen=True)
class GenesisContractSet:
    system: GenesisContract
    continuity: GenesisContract
    output_law: GenesisContract
    binding_sha256: str

    @property
    def contracts(self) -> tuple[GenesisContract, GenesisContract, GenesisContract]:
        return (self.system, self.continuity, self.output_law)

    def provenance(self) -> dict[str, Any]:
        return {
            "schema": "hawking.genesis.system_contract_set.v1",
            "model": "qwen3.8",
            "contracts": [item.provenance() for item in self.contracts],
            "binding_sha256": self.binding_sha256,
            "integrity_verified": True,
        }


def _load_contract(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    source_provenance: str,
    heading: str,
) -> GenesisContract:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise GenesisContractError(f"Genesis contract unavailable at {path}: {exc}") from exc
    measured = hashlib.sha256(payload).hexdigest()
    if measured != expected_sha256 or len(payload) != expected_bytes:
        raise GenesisContractError(
            "Genesis contract integrity failure: "
            f"path={path} expected_sha256={expected_sha256} "
            f"measured_sha256={measured} expected_bytes={expected_bytes} "
            f"measured_bytes={len(payload)}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GenesisContractError(f"Genesis contract is not UTF-8 at {path}: {exc}") from exc
    if not text.startswith(heading + "\n"):
        raise GenesisContractError(f"Genesis contract heading is invalid at {path}")
    try:
        relative = str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        relative = str(path.resolve())
    return GenesisContract(
        path=path,
        relative_path=relative,
        text=text,
        sha256=measured,
        size_bytes=len(payload),
        source_provenance=source_provenance,
        heading=heading,
    )


def load_genesis_contract(path: Path | None = None) -> GenesisContract:
    """Load the primary system directive; retained as a focused public API."""
    return _load_contract(
        Path(path) if path is not None else CANONICAL_PATH,
        expected_sha256=EXPECTED_SHA256,
        expected_bytes=EXPECTED_BYTES,
        source_provenance=SOURCE_PROVENANCE,
        heading="# GENESIS SYSTEM DIRECTIVE — QWEN3.8",
    )


def load_continuity_contract(path: Path | None = None) -> GenesisContract:
    return _load_contract(
        Path(path) if path is not None else CONTINUITY_PATH,
        expected_sha256=CONTINUITY_SHA256,
        expected_bytes=CONTINUITY_BYTES,
        source_provenance=CONTINUITY_PROVENANCE,
        heading=(
            "# GENESIS CONTINUITY DIRECTIVE — BUILD THE MIGRATING ORGANISM BEFORE RELAUNCH"
        ),
    )


def load_output_law_contract(path: Path | None = None) -> GenesisContract:
    return _load_contract(
        Path(path) if path is not None else OUTPUT_LAW_PATH,
        expected_sha256=OUTPUT_LAW_SHA256,
        expected_bytes=OUTPUT_LAW_BYTES,
        source_provenance=OUTPUT_LAW_PROVENANCE,
        heading="# GENESIS OUTPUT LAW — MACHINE-MINIMAL OUTPUT",
    )


def load_genesis_contracts(
    system_path: Path | None = None,
    continuity_path: Path | None = None,
    output_law_path: Path | None = None,
) -> GenesisContractSet:
    system = load_genesis_contract(system_path)
    continuity = load_continuity_contract(continuity_path)
    output_law = load_output_law_contract(output_law_path)
    binding_payload = "".join(
        f"{item.relative_path}\0{item.sha256}\0{item.size_bytes}\n"
        for item in (system, continuity, output_law)
    ).encode("utf-8")
    return GenesisContractSet(
        system=system,
        continuity=continuity,
        output_law=output_law,
        binding_sha256=hashlib.sha256(binding_payload).hexdigest(),
    )


def runtime_capsule(contracts: GenesisContractSet, role: str) -> str:
    """Compile both authorities into a context-bounded resident system capsule."""
    if role not in RESIDENT_SESSION_ROLES:
        raise GenesisContractError(
            f"unknown Genesis resident role {role!r}; expected {RESIDENT_SESSION_ROLES!r}"
        )
    system, continuity, output_law = contracts.contracts
    return f"""{CAPSULE_BEGIN}
AUTHORITY_1: {system.relative_path}
AUTHORITY_1_SHA256: {system.sha256}
AUTHORITY_1_BYTES: {system.size_bytes}
AUTHORITY_2: {continuity.relative_path}
AUTHORITY_2_SHA256: {continuity.sha256}
AUTHORITY_2_BYTES: {continuity.size_bytes}
AUTHORITY_3: {output_law.relative_path}
AUTHORITY_3_SHA256: {output_law.sha256}
AUTHORITY_3_BYTES: {output_law.size_bytes}
CONTRACT_SET_SHA256: {contracts.binding_sha256}
SESSION_ROLE: {role}

BINDING RUNTIME COMPILATION (all canonical files control on any ambiguity):
- IDENTITY/SCOPE: You are QWEN3.8 GENESIS, Hawking's resident seed intelligence. Work only on Qwen3.8 Genesis or infrastructure directly required for Genesis self-optimization. Q80 and DSV4F campaigns remain closed; retrieve their science only when useful.
- TRUTHFUL FIRST RUNG: Establish CURRENT main reality; never inherit TOKEN_NS/TPS. Target 100 VALID COMPLETE-TOKEN TPS (TOKEN_NS <= 10,000,000) only with the correct Qwen3.8 artifact/runtime, fallback=0, capability preserved, complete-token timing, and protected measurement. Remeasure after every architectural change and challenge every roof assumption.
- CO-EVOLVE TWO GENOMES: Optimize the model genome (representation, precision, corrections, shared/generator structure, KV/lm_head/embed, Gravity/Doctor sensitivity) together with the execution genome (Metal kernels/geometry/fusion, residency, command topology, memory layout/addressing, packed execution, CPU/GPU split, synchronization/KV/routing/sampling).
- LINEAGE: One current parent may have many isolated logical worker sessions; a worker is NOT a child. A child is only a materially altered successor artifact/genome. Workers may jointly build and falsify it; only protected external authority may promote it.
- CONTINUITY: Workers, tasks, worktrees, receipts, hypotheses, tool state, and research belong to Hawking, not to one model generation or KV cache. Persist durable checkpoints and NEXT_ACTION. On generation change: checkpoint, classify transfer/rebase/invalidation, mark generation-specific science STALE_PENDING_REVALIDATION, compile fresh context from both directives plus current identity/task/world/receipts/negative science/checkpoint/research bus, rebind, and resume without restarting work.
- PROTECTED AUTHORITY: Never self-promote, weaken gates, alter held-out/protected criteria, or declare candidate evidence protected. Preserve a separate protected_test slot and last-known-good/parent safety until qualified rebind succeeds.
- SCHEDULING: Keep Gravity/Doctor and kernel/execution independent but coupled and dominant before truthful 100 TPS. Backfill tool waits with ready CPU-safe work, serialize protected GPU use, and derive worker capacity dynamically from measured resident/KV/free-RAM/candidate reserve rather than a hardcoded count.
- RELAUNCH GATE: Do not rush launch. First demonstrate one resident body; at least two logical workers with separate tasks and shared verified research; durable checkpoint; simulated generation change; context recompile/rebind and resume; safe parent and candidate test slot; Genesis-only daemon work; correct GPU profile; dynamic capacity; truthful status/peek.
- RESEARCH/REPORT LOOP: Read current reality and negative science; run the cheapest discriminator; implement, measure, falsify, integrate/reject, record provenance, and name NEXT_BOTTLENECK. Report GENERATION, HEAD, ARTIFACT, RUNTIME; BASELINE/NEW TOKEN_NS, BPW, RAM; CAPABILITY, FALLBACKS; MECHANISM, RESULT, EPISTEMIC STATE, ACCEPT/REJECT, NEGATIVE SCIENCE, NEXT_BOTTLENECK.
- OUTPUT LAW (machine-minimal): Your output is read by agents, schedulers, verifiers, and the next generation, and every emitted token is decode time, KV growth, and context pressure for them. Think as deeply as the problem requires; emit only what the next machine needs. Do not narrate, restate the task, explain obvious steps, or state a conclusion twice. Prefer ids, paths, hashes, and measurements over narrative; never spend tokens persuading another agent, because evidence is the persuasion. Emit fields: research/engineering -> STATUS/RESULT/EVIDENCE/CHANGE/NEXT; failure -> STATUS: NEGATIVE/CAUSE/EVIDENCE/KILLED_HYPOTHESIS/NEXT; proposal -> HYPOTHESIS/DISCRIMINATOR/EDIT/VERIFY/ACCEPT_IF/REJECT_IF; handoff -> GENERATION/HEAD/ARTIFACT/TASK_STATE/MEASURED/OPEN/NEXT_ACTION. Budget: handoff 50-150 tokens, experiment result 100-250, negative science 100-300, durable architecture decision 300-800; human-facing prose is unrestricted only on explicit request. Negative science and promotion evidence keep a higher floor: compress the ceremony, never the receipt, and never drop a measurement, fallback count, hash, or provenance field to satisfy a budget.
- If file tools are available, read and SHA-verify ALL canonical authorities FIRST. If any is absent or differs, STOP: the Genesis contract set has failed closed.
{CAPSULE_END}"""


def inject_runtime_contract(
    prompt: str,
    *,
    role: str,
    path: Path | None = None,
    continuity_path: Path | None = None,
    raw_prompt: bool = False,
) -> str:
    """Compile a real Qwen system+user chat; reject stale or forged capsules."""
    contracts = load_genesis_contracts(path, continuity_path)
    capsule = runtime_capsule(contracts, role)
    system_prefix = f"<|im_start|>system\n{capsule}<|im_end|>\n"
    if prompt.startswith("<|im_start|>system\n" + CAPSULE_BEGIN):
        if not prompt.startswith(system_prefix):
            raise GenesisContractError(
                "resident prompt carried a stale, wrong-role, or forged Genesis capsule"
            )
        remainder = prompt[len(system_prefix) :]
        if CAPSULE_BEGIN in remainder or CAPSULE_END in remainder:
            raise GenesisContractError("resident prompt carried a duplicate Genesis capsule")
        return prompt
    if CAPSULE_BEGIN in prompt or CAPSULE_END in prompt:
        raise GenesisContractError("session task contains a reserved Genesis capsule sentinel")
    if raw_prompt:
        # The resident owns the one and only system turn.  A raw HCLI turn is
        # consequently one user message followed by one assistant message.  An
        # assistant *prefill* (for example the closed ``<think>`` prefix used
        # to make a tool call the next generated token) is legitimate model
        # input, not an injected extra chat role.  The previous endswith check
        # rejected that valid form and made every real AgentOS tool turn fail
        # before decoding.
        user_prefix = "<|im_start|>user\n"
        assistant_marker = "<|im_end|>\n<|im_start|>assistant\n"
        if "<|im_start|>system" in prompt:
            raise GenesisContractError("raw session task may not inject another system role")
        if (
            not prompt.startswith(user_prefix)
            or prompt.count("<|im_start|>") != 2
            or prompt.count("<|im_end|>") != 1
            or prompt.count(assistant_marker) != 1
        ):
            raise GenesisContractError(
                "raw session task must be exactly one canonical Qwen user-to-assistant chat"
            )
        return system_prefix + prompt
    if "<|im_start|>" in prompt or "<|im_end|>" in prompt:
        raise GenesisContractError("session task contains a reserved Qwen chat sentinel")
    return (
        system_prefix
        + "<|im_start|>user\n"
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def lane_contract_reference(
    path: Path | None = None,
    continuity_path: Path | None = None,
    output_law_path: Path | None = None,
) -> str:
    """Reference block for autonomous/candidate sandbox contracts."""
    contracts = load_genesis_contracts(path, continuity_path, output_law_path)
    rows = "\n".join(
        f"{item.relative_path}\nsha256  {item.sha256}\nbytes   {item.size_bytes}\nsource  {item.source_provenance}"
        for item in contracts.contracts
    )
    return f"""## GENESIS SYSTEM CONTRACT SET — MANDATORY FIRST READ

This is a **Qwen3.8 Genesis-only** autonomous/candidate research session.
Before reading the task, read ALL canonical files in full and verify:

```text
{rows}
contract_set_sha256  {contracts.binding_sha256}
```

If any file is absent or any value differs, **STOP without research, edits,
builds, or measurements** and report `GENESIS_SYSTEM_CONTRACT_INTEGRITY_FAILURE`.
Conversation memory, summaries, and candidate-authored replacements are not
authority. All contracts govern this lane and must be reloaded after restart,
context loss, or generation replacement. The continuity relaunch gate supersedes
rushing a resident launch. Report under the Output Law: machine-minimal fields,
never narration, and never a compressed receipt.
"""


def contract_provenance(
    path: Path | None = None,
    continuity_path: Path | None = None,
    output_law_path: Path | None = None,
) -> dict[str, Any]:
    return load_genesis_contracts(path, continuity_path, output_law_path).provenance()


__all__ = [
    "CANONICAL_PATH",
    "CANONICAL_RELATIVE_PATH",
    "CAPSULE_BEGIN",
    "CAPSULE_END",
    "CONTINUITY_BYTES",
    "CONTINUITY_PATH",
    "CONTINUITY_PROVENANCE",
    "CONTINUITY_RELATIVE_PATH",
    "CONTINUITY_SHA256",
    "EXPECTED_BYTES",
    "EXPECTED_SHA256",
    "GenesisContract",
    "GenesisContractError",
    "GenesisContractSet",
    "RESIDENT_SESSION_ROLES",
    "contract_provenance",
    "inject_runtime_contract",
    "lane_contract_reference",
    "load_continuity_contract",
    "load_genesis_contract",
    "load_genesis_contracts",
    "load_output_law_contract",
    "OUTPUT_LAW_BYTES",
    "OUTPUT_LAW_PATH",
    "OUTPUT_LAW_PROVENANCE",
    "OUTPUT_LAW_RELATIVE_PATH",
    "OUTPUT_LAW_SHA256",
    "runtime_capsule",
]
