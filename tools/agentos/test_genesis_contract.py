"""Persistent Qwen3.8 Genesis contract set, resident injection, and lane binding.

Three canonical authorities govern every Genesis session: the system directive,
the continuity directive, and the output law. They are one integrity-checked set
- a session that receives two of three is not governed, so the tests here assert
the set, never a single file.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
AGENTOS = REPO / "tools" / "agentos"
if str(AGENTOS) not in sys.path:
    sys.path.insert(0, str(AGENTOS))

import genesis_contract as contract  # noqa: E402
import genesis_resident as resident  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


daemon = _load(REPO / "tools" / "ascent_daemon.py", "ascent_daemon_contract_test")

ALL_AUTHORITIES = (
    (
        contract.CANONICAL_RELATIVE_PATH,
        contract.EXPECTED_SHA256,
        contract.EXPECTED_BYTES,
        "# GENESIS SYSTEM DIRECTIVE — QWEN3.8\n",
    ),
    (
        contract.CONTINUITY_RELATIVE_PATH,
        contract.CONTINUITY_SHA256,
        contract.CONTINUITY_BYTES,
        "# GENESIS CONTINUITY DIRECTIVE — BUILD THE MIGRATING ORGANISM BEFORE RELAUNCH\n",
    ),
    (
        contract.OUTPUT_LAW_RELATIVE_PATH,
        contract.OUTPUT_LAW_SHA256,
        contract.OUTPUT_LAW_BYTES,
        "# GENESIS OUTPUT LAW — MACHINE-MINIMAL OUTPUT\n",
    ),
)


def test_every_canonical_authority_is_verbatim_hash_bound_and_provenanced() -> None:
    loaded = contract.load_genesis_contracts()
    assert len(loaded.contracts) == 3
    for item, (rel, sha, size, heading) in zip(loaded.contracts, ALL_AUTHORITIES):
        assert item.path == REPO / rel
        assert item.sha256 == sha
        assert item.size_bytes == size
        assert item.text.startswith(heading)
        assert item.provenance() == {
            "canonical_path": str(rel),
            "sha256": sha,
            "size_bytes": size,
            "source_provenance": item.source_provenance,
            "integrity_verified": True,
        }

    assert loaded.system.text.endswith("# THEN DO IT AGAIN.\n")
    assert "# 30. IMMEDIATE ORDER" in loaded.system.text
    assert "# 20. END STATE" in loaded.continuity.text
    assert "# NEVER COMPRESS THE RECEIPT.\n" in loaded.output_law.text

    provenance = loaded.provenance()
    assert provenance["schema"] == "hawking.genesis.system_contract_set.v1"
    assert provenance["model"] == "qwen3.8"
    assert provenance["binding_sha256"] == loaded.binding_sha256
    assert provenance["integrity_verified"] is True
    assert [row["canonical_path"] for row in provenance["contracts"]] == [
        str(rel) for rel, _, _, _ in ALL_AUTHORITIES
    ]


def test_binding_sha_changes_when_any_single_authority_changes(tmp_path: Path) -> None:
    """The set hash must be a real function of all three, not of the first one."""
    baseline = contract.load_genesis_contracts().binding_sha256
    for loader_kwarg, source in (
        ("system_path", contract.CANONICAL_PATH),
        ("continuity_path", contract.CONTINUITY_PATH),
        ("output_law_path", contract.OUTPUT_LAW_PATH),
    ):
        swapped = tmp_path / f"{loader_kwarg}.md"
        swapped.write_bytes(source.read_bytes())
        # Same bytes at a different path: content is identical, identity is not.
        assert (
            contract.load_genesis_contracts(**{loader_kwarg: swapped}).binding_sha256
            != baseline
        )


@pytest.mark.parametrize(
    "loader",
    [
        contract.load_genesis_contract,
        contract.load_continuity_contract,
        contract.load_output_law_contract,
    ],
)
def test_each_loader_fails_closed_on_missing_or_tampered_bytes(
    loader,
    tmp_path: Path,
) -> None:
    with pytest.raises(contract.GenesisContractError, match="unavailable"):
        loader(tmp_path / "missing.md")

    tampered = tmp_path / "directive.md"
    tampered.write_bytes(loader().path.read_bytes() + b"tamper\n")
    with pytest.raises(contract.GenesisContractError, match="integrity failure"):
        loader(tampered)


def test_a_missing_output_law_fails_the_whole_set_closed(tmp_path: Path) -> None:
    """Two of three authorities is an ungoverned session, not a degraded one."""
    with pytest.raises(contract.GenesisContractError, match="unavailable"):
        contract.load_genesis_contracts(output_law_path=tmp_path / "gone.md")
    with pytest.raises(contract.GenesisContractError, match="unavailable"):
        contract.contract_provenance(output_law_path=tmp_path / "gone.md")
    with pytest.raises(contract.GenesisContractError, match="unavailable"):
        contract.lane_contract_reference(output_law_path=tmp_path / "gone.md")


def test_runtime_capsule_is_bounded_and_carries_all_binding_rules() -> None:
    loaded = contract.load_genesis_contracts()
    capsule = contract.runtime_capsule(loaded, "parent")
    # The resident body has a 4096-token context; the capsule must leave room for
    # the actual task. ~4 bytes/token puts this ceiling near a third of context.
    assert len(capsule.encode("utf-8")) < 7_000
    for required in (
        str(contract.CANONICAL_RELATIVE_PATH),
        str(contract.CONTINUITY_RELATIVE_PATH),
        str(contract.OUTPUT_LAW_RELATIVE_PATH),
        contract.EXPECTED_SHA256,
        contract.CONTINUITY_SHA256,
        contract.OUTPUT_LAW_SHA256,
        loaded.binding_sha256,
        "SESSION_ROLE: parent",
        "QWEN3.8 GENESIS",
        "100 VALID COMPLETE-TOKEN TPS",
        "CO-EVOLVE TWO GENOMES",
        "worker is NOT a child",
        "Never self-promote",
        "NEXT_BOTTLENECK",
        "OUTPUT LAW",
    ):
        assert required in capsule


def test_capsule_output_law_carries_the_shapes_budgets_and_receipt_floor() -> None:
    """A named law with no emission shape changes no behavior."""
    capsule = contract.runtime_capsule(contract.load_genesis_contracts(), "parent")
    for shape in (
        "STATUS/RESULT/EVIDENCE/CHANGE/NEXT",
        "KILLED_HYPOTHESIS",
        "HYPOTHESIS/DISCRIMINATOR/EDIT/VERIFY/ACCEPT_IF/REJECT_IF",
        "GENERATION/HEAD/ARTIFACT/TASK_STATE/MEASURED/OPEN/NEXT_ACTION",
    ):
        assert shape in capsule
    assert "50-150 tokens" in capsule
    # The law must never be readable as "shorten the evidence".
    assert "never the receipt" in capsule
    assert "evidence is the persuasion" in capsule.lower()


def test_runtime_capsule_rejects_an_unknown_session_role() -> None:
    with pytest.raises(contract.GenesisContractError, match="unknown Genesis resident role"):
        contract.runtime_capsule(contract.load_genesis_contracts(), "root")


def _conforming_body_reply(request: dict, text: str) -> dict:
    """What a body actually running this contract set must send back.

    The resident client refuses any reply that does not echo the binding, the
    mode, and a zero fallback count - so a fake that omits them models a body
    the client is right to reject, not a healthy one.
    """
    return {
        "ok": True,
        "text": text,
        "fallbacks": 0,
        "genesis_system_contract": request.get("genesis_system_contract"),
        "genesis_contract_mode": request.get("genesis_contract_mode"),
    }


def test_normal_research_prompt_is_injected_for_every_resident_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict] = []

    monkeypatch.setattr(resident, "body_is_up", lambda _path: True)

    def fake_rpc(_path: Path, request: dict, _timeout: float | None = None, **_kw):
        requests.append(request)
        return _conforming_body_reply(request, "proposal")

    monkeypatch.setattr(resident, "rpc", fake_rpc)
    binding = contract.load_genesis_contracts().binding_sha256
    for role in resident.SESSION_ROLES:
        response = resident.propose("measure the next bottleneck", session=role)
        assert response is not None
        request = requests[-1]
        assert request["session"] == role
        assert request["genesis_contract_mode"] == "runtime_capsule_injected"
        assert request["prompt"].startswith(
            "<|im_start|>system\n" + contract.CAPSULE_BEGIN
        )
        assert f"SESSION_ROLE: {role}" in request["prompt"]
        assert request["prompt"].endswith(
            "<|im_start|>user\nmeasure the next bottleneck"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        assert request["genesis_system_contract"]["binding_sha256"] == binding
        assert response["genesis_system_contract"]["integrity_verified"] is True


@pytest.mark.parametrize(
    ("prompt", "raw"),
    [
        ("Say hi.", False),
        (
            "<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n",
            True,
        ),
    ],
)
def test_explicit_protected_capability_path_preserves_prompt_bytes_and_binding(
    prompt: str,
    raw: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}
    monkeypatch.setattr(resident, "body_is_up", lambda _path: True)

    def fake_rpc(_path: Path, request: dict, _timeout: float | None = None, **_kw):
        seen.update(request)
        return _conforming_body_reply(request, "protected-result")

    monkeypatch.setattr(resident, "rpc", fake_rpc)
    response = resident.propose(
        prompt,
        raw=raw,
        session="protected_test",
        protected_capability=True,
    )
    assert response is not None
    assert seen["prompt"].encode("utf-8") == prompt.encode("utf-8")
    assert seen["raw"] is raw
    assert contract.CAPSULE_BEGIN not in seen["prompt"]
    assert seen["genesis_contract_mode"] == "protected_capability_prompt_preserved"
    assert (
        seen["genesis_system_contract"]["binding_sha256"]
        == contract.load_genesis_contracts().binding_sha256
    )
    assert response["genesis_contract_mode"] == "protected_capability_prompt_preserved"


def test_prompt_preservation_cannot_silently_bypass_ordinary_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="requires session='protected_test'"):
        resident.propose("ordinary task", session="parent", protected_capability=True)

    seen: dict = {}
    monkeypatch.setattr(resident, "body_is_up", lambda _path: True)

    def fake_rpc(_path: Path, request: dict, _timeout: float | None = None, **_kw):
        seen.update(request)
        return _conforming_body_reply(request, "research-result")

    monkeypatch.setattr(resident, "rpc", fake_rpc)
    resident.propose("ordinary task", session="protected_test")
    assert seen["prompt"].startswith("<|im_start|>system\n" + contract.CAPSULE_BEGIN)
    assert seen["genesis_contract_mode"] == "runtime_capsule_injected"


def test_a_body_running_a_different_contract_set_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The echo check is the only thing standing between us and an ungoverned body."""
    monkeypatch.setattr(resident, "body_is_up", lambda _path: True)

    def stale_binding(_path: Path, request: dict, _timeout: float | None = None, **_kw):
        reply = _conforming_body_reply(request, "proposal")
        reply["genesis_system_contract"] = {"binding_sha256": "0" * 64}
        return reply

    monkeypatch.setattr(resident, "rpc", stale_binding)
    assert resident.propose("measure the next bottleneck") is None

    def silent_fallback(_path: Path, request: dict, _timeout: float | None = None, **_kw):
        reply = _conforming_body_reply(request, "proposal")
        reply["fallbacks"] = 1
        return reply

    monkeypatch.setattr(resident, "rpc", silent_fallback)
    assert resident.propose("measure the next bottleneck") is None


def test_resident_health_reports_verified_contract_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resident,
        "rpc",
        lambda *_args, **_kwargs: {
            "ok": True,
            "pid": 123,
            "body_resident": True,
            "genesis_system_contract": contract.contract_provenance(),
        },
    )
    monkeypatch.setattr(resident, "process_alive", lambda _pid: True)
    info = resident.health(Path("/tmp/not-used.sock"))
    assert info is not None
    binding = info["genesis_system_contract"]
    assert binding["binding_sha256"] == contract.load_genesis_contracts().binding_sha256
    assert binding["integrity_verified"] is True
    assert len(binding["contracts"]) == 3


def test_cold_research_fallback_also_receives_runtime_capsule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "greedy"
    binary.write_text("stub\n")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}\n")
    monkeypatch.setenv("GENESIS_BIN", str(binary))
    monkeypatch.setenv("GENESIS_ARTIFACT", str(artifact))
    monkeypatch.setenv("GENESIS_TOKENIZER", str(tokenizer))
    seen: dict = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="GENERATED_TEXT_VERBATIM: proposal\nFALLBACKS: 0\n",
            stderr="",
        )

    monkeypatch.setattr(daemon.subprocess, "run", fake_run)
    assert daemon._try_shell_propose("research task") == "proposal"
    prompt = seen["argv"][seen["argv"].index("--prompt") + 1]
    assert prompt.startswith("<|im_start|>system\n" + contract.CAPSULE_BEGIN)
    assert "SESSION_ROLE: parent" in prompt
    assert contract.OUTPUT_LAW_SHA256 in prompt
    assert prompt.endswith(
        "<|im_start|>user\nresearch task<|im_end|>\n<|im_start|>assistant\n"
    )


def test_ascent_generated_lane_requires_full_contract_set_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(daemon, "LANES", tmp_path / "lanes")
    state = {"targets": []}
    harvested = [
        {
            "lane": "qwen38-contract-test",
            "status": "SHIPPED",
            "next_bottleneck": "metal.example 123 ns/token",
        }
    ]
    assert daemon.generate_targets(
        state,
        harvested,
        proposer=lambda _bottleneck: "test a named geometry mechanism",
    ) == 1
    target = state["targets"][0]
    text = Path(target["contract"]).read_text()
    assert text.startswith("## GENESIS SYSTEM CONTRACT SET — MANDATORY FIRST READ\n")
    for rel, sha, _, _ in ALL_AUTHORITIES:
        assert str(rel) in text
        assert sha in text
    assert "read ALL canonical files in full" in text
    assert "GENESIS_SYSTEM_CONTRACT_INTEGRITY_FAILURE" in text
    assert "never a compressed receipt" in text
    assert (
        target["genesis_system_contract"]["binding_sha256"]
        == contract.load_genesis_contracts().binding_sha256
    )


def test_ascent_lane_generation_fails_closed_on_tampered_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = tmp_path / "directive.md"
    bad.write_text("candidate replacement\n")
    monkeypatch.setattr(daemon, "GENESIS_SYSTEM_CONTRACT", bad)
    monkeypatch.setattr(daemon, "LANES", tmp_path / "lanes")
    with pytest.raises(RuntimeError, match="integrity failure"):
        daemon.generate_targets(
            {"targets": []},
            [
                {
                    "lane": "qwen38-bad-contract",
                    "status": "SHIPPED",
                    "next_bottleneck": "metal.example 123 ns/token",
                }
            ],
            proposer=lambda _bottleneck: "must not run",
        )
    assert not (tmp_path / "lanes").exists()
