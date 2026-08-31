"""Generic resident install contract for whichever NX wins the tournament.

The tournament outcome plugs in without a rewrite: every slot is generic over
the winner's identity document. This module never installs, launches, or
measures; it records the contract a later Codex-owned install must satisfy.

    python3 tools/future/resident_install.py --build
    python3 tools/future/resident_install.py --dry-run

Not a fork of hcli/agentos/resident_gate.py, native_gate.py, recovery.py,
protected_accelerator_benchmark.py, tools/agentos/genesis_resident.py, or
tools/hcli_resident/serve_sealed.py. Those remain the live lifecycle. This
sidecar only names the slots an NX winner must fill to sit in that lifecycle.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))


import argparse
import hashlib
import json
from typing import Any, Mapping

from tools.future._common import write_receipt

RECEIPT = "RESIDENT_INSTALL_CONTRACT.json"

# Order is the install sequence. Do not reorder without a schema bump.
PHASES: tuple[str, ...] = (
    "nx_identity",
    "executable_identity",
    "tokenizer_session",
    "memory_requirements",
    "launch_args",
    "readiness_probe",
    "shutdown",
    "unload",
    "protected_benchmark_evacuation",
    "restart",
    "crash_recovery",
    "fallback_policy",
    "capability_receipt",
    "performance_receipt",
)

# Structural required keys per phase. Values stay UNKNOWN/null until a winner
# identity actually fills them. No hardware measurement lives here.
PHASE_REQUIRED: dict[str, tuple[str, ...]] = {
    "nx_identity": ("contender_id", "nx_kind", "identity_path", "identity_digest"),
    "executable_identity": ("binary_path", "artifact_root"),
    "tokenizer_session": ("tokenizer_path", "prompt_contract"),
    "memory_requirements": ("resident_bytes", "source"),
    "launch_args": ("argv",),
    "readiness_probe": ("probe", "timeout_s"),
    "shutdown": ("signal", "grace_s"),
    "unload": ("drop_weights", "release_device"),
    "protected_benchmark_evacuation": (
        "stop_before_closing_quiescence",
        "lock_name",
        "owner",
    ),
    "restart": ("max_restarts", "reset_session"),
    "crash_recovery": ("policy", "module"),
    "fallback_policy": ("on_identity_mismatch", "on_unsealed"),
    "capability_receipt": ("path", "schema"),
    "performance_receipt": ("path", "required_bench_state"),
}

# Lifecycle owners already on disk. Consumed, not reimplemented.
EXISTING_LIFECYCLE: dict[str, str] = {
    "resident_gate": "hcli/agentos/resident_gate.py",
    "native_gate": "hcli/agentos/native_gate.py",
    "native_mission_gate": "hcli/agentos/native_mission_gate.py",
    "recovery_gate": "hcli/agentos/recovery.py",
    "protected_accelerator_benchmark": "hcli/agentos/protected_accelerator_benchmark.py",
    "genesis_resident": "tools/agentos/genesis_resident.py",
    "serve_sealed": "tools/hcli_resident/serve_sealed.py",
    "incumbent_identity": "hcli/hawking-native.sealed-3.14.json",
    "resident_seal": "receipts/headless/HCLI_RESIDENT_SEAL.json",
}


def _slot(phase: str, **fields: Any) -> dict[str, Any]:
    required = PHASE_REQUIRED[phase]
    body = {k: fields.get(k) for k in required}
    body.update({k: v for k, v in fields.items() if k not in body})
    body["phase"] = phase
    body["bound"] = all(body.get(k) not in (None, "", [], {}) for k in required)
    return body


def empty_contract() -> dict[str, Any]:
    """Unbound contract. Winner identity has not been plugged in."""
    slots = {phase: _slot(phase) for phase in PHASES}
    # Policy constants that do not depend on the winner. Identity/binary/tokenizer
    # stay unbound until bind_winner.
    slots["readiness_probe"] = _slot(
        "readiness_probe",
        probe="connector.start + identity.resident_health.pid",
        timeout_s=180.0,
        module=EXISTING_LIFECYCLE["resident_gate"],
    )
    slots["shutdown"] = _slot(
        "shutdown",
        signal="connector.stop",
        grace_s=5.0,
    )
    slots["unload"] = _slot(
        "unload",
        drop_weights=True,
        release_device=True,
        note="Unload must drop weights and release the device before any other resident binds.",
    )
    slots["protected_benchmark_evacuation"] = _slot(
        "protected_benchmark_evacuation",
        stop_before_closing_quiescence=True,
        lock_name="protected-accelerator-bench.lock",
        owner=EXISTING_LIFECYCLE["protected_accelerator_benchmark"],
        sidecar_must_not_take_lease=True,
    )
    slots["restart"] = _slot(
        "restart",
        max_restarts=0,
        reset_session=True,
        no_silent_restart=True,
    )
    slots["crash_recovery"] = _slot(
        "crash_recovery",
        policy="fail-closed fixture proof in recovery_gate; production recovery is Codex-owned",
        module=EXISTING_LIFECYCLE["recovery_gate"],
    )
    slots["fallback_policy"] = _slot(
        "fallback_policy",
        on_identity_mismatch="REFUSE naming the field",
        on_unsealed="refuse unless allow-unsealed; stamp sealed=false",
    )
    slots["performance_receipt"] = _slot(
        "performance_receipt",
        path=None,
        required_bench_state="PROTECTED_ABSOLUTE",
        sidecar_cannot_emit=True,
        diagnostic_relative_never_promotes=True,
    )
    return {
        "schema": "hawking.future.resident_install.v1",
        "winner_id": None,
        "generic": True,
        "phases": PHASES,
        "slots": slots,
        "policy": {
            "generic_over_winner": True,
            "hardcoded_winner_forbidden": True,
            "hardware_numbers_forbidden": True,
            "protected_evacuation": (
                "The resident is stopped before the closing quiescence sample. "
                "Codex owns the protected GPU lease; this sidecar never takes it."
            ),
            "fallback": (
                "Identity mismatch is a REFUSAL naming the field, not a warning. "
                "Unsealed serve is stamped sealed=false rather than silently diverging."
            ),
        },
        "existing_lifecycle": dict(EXISTING_LIFECYCLE),
    }


def _get(identity: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = identity
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def bind_winner(
    winner_id: str,
    identity: Mapping[str, Any],
    *,
    identity_path: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill the generic contract from any winner identity document.

    Accepts a Flash-style NX genome or a Qwen-style resident profile. Missing
    fields stay null; this function does not invent hashes, paths, or hardware
    numbers. `extra` may supply Codex-owned paths (capability/performance
    receipts) without rewriting the contract.
    """
    extra = dict(extra or {})
    physical = identity.get("physical_program") if isinstance(identity.get("physical_program"), Mapping) else {}
    generation = identity.get("generation") if isinstance(identity.get("generation"), Mapping) else {}
    prompt = identity.get("prompt_contract") if isinstance(identity.get("prompt_contract"), Mapping) else {}
    fusion = identity.get("fusion_env") if isinstance(identity.get("fusion_env"), Mapping) else {}
    limits = identity.get("limits") if isinstance(identity.get("limits"), Mapping) else {}

    executor = physical.get("executor")
    binary = (
        identity.get("resident_binary")
        or identity.get("binary")
        or (executor[0].get("path") if isinstance(executor, list) and executor and isinstance(executor[0], Mapping) else None)
    )
    artifact = (
        identity.get("artifact_root")
        or _get(identity, "lowers_nr", "path")
        or extra.get("artifact_root")
    )
    tokenizer = identity.get("tokenizer") or extra.get("tokenizer_path")
    nx_kind = identity.get("nx_kind") or identity.get("protocol") or identity.get("runtime")
    digest = identity.get("seal_sha256") or extra.get("identity_digest")
    if not digest:
        digest = hashlib.sha256(
            json.dumps(dict(identity), sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()

    resident_bytes = extra.get("resident_bytes")
    if resident_bytes is None:
        # Artifact accounting from a named seal, never a GPU measurement.
        resident_bytes = extra.get("quoted_artifact_bytes")

    env = dict(fusion)
    env.update(extra.get("env") or {})

    argv = extra.get("argv")
    if argv is None and binary:
        argv = [str(binary)]

    contract = empty_contract()
    contract["winner_id"] = str(winner_id)
    contract["slots"] = {
        "nx_identity": _slot(
            "nx_identity",
            contender_id=str(winner_id),
            nx_kind=nx_kind,
            identity_path=identity_path,
            identity_digest=digest,
            resident_identity=identity.get("resident_identity") or identity.get("model_id"),
            status=identity.get("status"),
        ),
        "executable_identity": _slot(
            "executable_identity",
            binary_path=binary,
            artifact_root=artifact,
            binary_sha256=extra.get("binary_sha256"),
            compiled_for_machine_genome=identity.get("compiled_for_machine_genome"),
        ),
        "tokenizer_session": _slot(
            "tokenizer_session",
            tokenizer_path=tokenizer,
            prompt_contract=prompt or extra.get("prompt_contract"),
            max_seq_len=identity.get("max_seq_len") or limits.get("context") or limits.get("context_length"),
            generation_temperature=generation.get("temperature"),
        ),
        "memory_requirements": _slot(
            "memory_requirements",
            resident_bytes=resident_bytes,
            source=extra.get("memory_source") or "identity-or-seal-accounting; not a GPU measurement",
            unified_memory_bytes=_get(identity, "compiled_for_machine_genome", "unified_memory_bytes"),
        ),
        "launch_args": _slot(
            "launch_args",
            argv=argv,
            env=env,
            mode=identity.get("mode"),
            require_fusion_env=identity.get("require_fusion_env"),
        ),
        "readiness_probe": _slot(
            "readiness_probe",
            probe=extra.get("probe") or "connector.start + identity.resident_health.pid",
            timeout_s=extra.get("timeout_s") or 180.0,
            module=EXISTING_LIFECYCLE["resident_gate"],
        ),
        "shutdown": _slot(
            "shutdown",
            signal=extra.get("shutdown_signal") or "connector.stop",
            grace_s=extra.get("grace_s") or 5.0,
        ),
        "unload": _slot(
            "unload",
            drop_weights=True,
            release_device=True,
            note="Unload must drop weights and release the device before any other resident binds.",
        ),
        "protected_benchmark_evacuation": _slot(
            "protected_benchmark_evacuation",
            stop_before_closing_quiescence=True,
            lock_name="protected-accelerator-bench.lock",
            owner=EXISTING_LIFECYCLE["protected_accelerator_benchmark"],
            sidecar_must_not_take_lease=True,
        ),
        "restart": _slot(
            "restart",
            max_restarts=extra.get("max_restarts") or 0,
            reset_session=True,
            no_silent_restart=True,
        ),
        "crash_recovery": _slot(
            "crash_recovery",
            policy="fail-closed fixture proof in recovery_gate; production recovery is Codex-owned",
            module=EXISTING_LIFECYCLE["recovery_gate"],
        ),
        "fallback_policy": _slot(
            "fallback_policy",
            on_identity_mismatch="REFUSE naming the field",
            on_unsealed="refuse unless allow-unsealed; stamp sealed=false",
            declared_fallbacks=identity.get("fallbacks"),
        ),
        "capability_receipt": _slot(
            "capability_receipt",
            path=extra.get("capability_receipt_path"),
            schema=extra.get("capability_schema") or "hawking.headless.capability_suite.v1",
        ),
        "performance_receipt": _slot(
            "performance_receipt",
            path=extra.get("performance_receipt_path"),
            required_bench_state="PROTECTED_ABSOLUTE",
            sidecar_cannot_emit=True,
            diagnostic_relative_never_promotes=True,
        ),
    }
    return contract


def validate_contract(contract: Mapping[str, Any]) -> list[str]:
    """Return problems. Empty list means every required slot is filled.

    Structural completeness is not NX completeness and is not a hardware claim.
    """
    problems: list[str] = []
    if not contract.get("winner_id"):
        problems.append("winner_id missing")
    slots = contract.get("slots")
    if not isinstance(slots, Mapping):
        return problems + ["slots missing"]
    for phase in PHASES:
        slot = slots.get(phase)
        if not isinstance(slot, Mapping):
            problems.append(f"{phase}: missing")
            continue
        for key in PHASE_REQUIRED[phase]:
            val = slot.get(key)
            if val in (None, "", [], {}):
                problems.append(f"{phase}.{key}: unbound")
    return problems


def bound(contract: Mapping[str, Any]) -> bool:
    return not validate_contract(contract)


def build() -> Any:
    doc = {
        "schema": "hawking.future.resident_install.v1",
        "version": 1,
        "purpose": (
            "Generic NX resident install contract. The tournament winner binds "
            "into these slots; the contract is not rewritten per contender."
        ),
        "phases": list(PHASES),
        "phase_required": {k: list(v) for k, v in PHASE_REQUIRED.items()},
        "unbound_template": empty_contract(),
        "existing_lifecycle": dict(EXISTING_LIFECYCLE),
        "negative_findings": [
            "hcli/agentos/resident.py does not exist; resident_gate.py is the live gate",
            "this module does not launch, install, or unload a resident",
            "performance_receipt.required_bench_state is PROTECTED_ABSOLUTE; sidecar cannot emit it",
        ],
        "gaps_closed": [
            "generic phase list covering identity through crash recovery and receipts",
            "bind_winner accepts Flash NX genomes and Qwen resident profiles without a rewrite",
            "validate_contract fail-closed on unbound required keys",
        ],
        "recovered_implementation": [
            {"path": p, "role": role} for role, p in EXISTING_LIFECYCLE.items()
        ],
    }
    return write_receipt(RECEIPT, doc, "tools/future/resident_install.py")


def selftest() -> Any:
    unbound = empty_contract()
    assert validate_contract(unbound), "unbound contract must fail validation"
    bound_c = bind_winner(
        "EXAMPLE_NOT_A_WINNER",
        {
            "nx_kind": "hawking.nos.example",
            "seal_sha256": "0" * 64,
            "resident_binary": "/nonexistent/bin",
            "artifact_root": "/nonexistent/art",
            "tokenizer": "/nonexistent/tokenizer.json",
            "prompt_contract": {"renderer": "example"},
            "fusion_env": {"EXAMPLE": "1"},
        },
        identity_path="example.json",
        extra={
            "quoted_artifact_bytes": 1,
            "memory_source": "synthetic selftest",
            "capability_receipt_path": "receipts/headless/CAPABILITY_example.json",
            "performance_receipt_path": "receipts/headless/PERF_example.json",
        },
    )
    problems = validate_contract(bound_c)
    assert not problems, problems
    assert bound_c["winner_id"] == "EXAMPLE_NOT_A_WINNER"
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        c = empty_contract()
        print(json.dumps({"winner_id": c["winner_id"], "phases": list(PHASES),
                          "unbound_problems": validate_contract(c)}, indent=2, sort_keys=True))
        print(selftest())
        return 0
    print(selftest() if a.build else build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
