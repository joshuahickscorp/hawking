#!/usr/bin/env python3.12
"""Cheap adversarial gates for elastic phase admission and host sprinting."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


CONDENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONDENSE))
import doctor_v5_aggressive_admission_policy as aggressive
import doctor_v5_elastic_phase_scheduler as elastic
import doctor_v5_host_sprint_plan as host
import doctor_v5_single_device_benchmark as benchmark


class ElasticHostSprintTests(unittest.TestCase):
    def setUp(self) -> None:
        # Production remains fail-closed on a closed runtime + measured thread
        # proof.  This explicit unittest-only mock exercises the inert receipt
        # chain without creating a runtime qualification bypass.
        self.real_closed_target_runtime_errors = (
            elastic._closed_target_runtime_errors
        )
        self.closed_runtime_fixture_patch = mock.patch.object(
            elastic, "_closed_target_runtime_errors", return_value=[]
        )
        self.closed_runtime_fixture_patch.start()
        aggressive.STAGE_ROOT.mkdir(parents=True, exist_ok=True)
        elastic.STAGE_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=elastic.STAGE_ROOT)
        self.directory = Path(self.temporary.name)
        self.plan = self._campaign_plan()
        self.queue_state = self._queue_state(self.plan)
        profile, binary = self._thread_profile(self.directory)
        rows = []
        for cell in self.plan["cells"]:
            rows.extend(self._process_samples(cell["cell_id"]))
        self.overlay = aggressive.build_overlay(
            self.plan, self.queue_state, rows,
            baseline_snapshot={"pressure_level": 1, "swap_used_mb": 0.0},
            thread_profile_path=profile, thread_binary_path=binary,
        )
        self.host_probe = self._host_probe()
        self.host_plan = host.build_plan(self.host_probe)
        self.overlay_path = self.directory / "aggressive-overlay.json"
        self.overlay_path.write_text(
            json.dumps(self.overlay, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.host_plan_path = self.directory / "host-plan.json"
        self.host_plan_path.write_text(
            json.dumps(self.host_plan, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.fixture_target = self.directory / "fixture_target.py"
        self.python_executable = elastic.local_observer.observe_process_invocation(
            os.getpid()
        )["executable"]["path"]
        self.fixture_target.write_text(
            "import argparse,hashlib,json,os,pathlib,time\n"
            "c=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()\n"
            "h=lambda v:hashlib.sha256(c(v)).hexdigest()\n"
            "p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--phase',required=True);p.add_argument('--cell',required=True)\n"
            "p.add_argument('--elastic-launch-request',required=True);p.add_argument('--elastic-claim-handshake',required=True);p.add_argument('--elastic-claim-ack',required=True)\n"
            "p.add_argument('--elastic-selected-threads',required=True);p.add_argument('--elastic-reservation-bytes',required=True);p.add_argument('--elastic-resource-spec-sha256',required=True)\n"
            "p.add_argument('--elastic-tier',required=True);p.add_argument('--elastic-rate',required=True);p.add_argument('--elastic-thread-selection-sha256',required=True);p.add_argument('--elastic-lend-decision-sha256',required=True);a=p.parse_args()\n"
            "q=json.loads(pathlib.Path(a.elastic_launch_request).read_bytes());cl=q['resource_claim'];cs=h(cl);tok=lambda v:'none' if v is None else str(v)\n"
            "vals=[a.elastic_selected_threads,a.elastic_reservation_bytes,a.elastic_resource_spec_sha256,a.elastic_tier,a.elastic_rate,a.elastic_thread_selection_sha256,a.elastic_lend_decision_sha256]\n"
            "exp=[tok(cl[k]) for k in ['selected_threads','reservation_bytes','resource_spec_sha256','tier','rate','thread_selection_sha256','lend_decision_sha256']]\n"
            "env={'DOCTOR_V5_ELASTIC_REQUEST_PATH':a.elastic_launch_request,'DOCTOR_V5_ELASTIC_RESOURCE_CLAIM_SHA256':cs,'DOCTOR_V5_ELASTIC_SELECTED_THREADS':exp[0],'DOCTOR_V5_ELASTIC_RESERVATION_BYTES':exp[1],'DOCTOR_V5_ELASTIC_RESOURCE_SPEC_SHA256':exp[2],'DOCTOR_V5_ELASTIC_TIER':exp[3],'DOCTOR_V5_ELASTIC_RATE':exp[4],'DOCTOR_V5_ELASTIC_THREAD_SELECTION_SHA256':exp[5],'DOCTOR_V5_ELASTIC_LEND_DECISION_SHA256':exp[6],**{k:exp[0] for k in ['RAYON_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']}}\n"
            "assert vals==exp and all(os.environ.get(k)==v for k,v in env.items()) and cs==q['resource_claim_sha256']\n"
            "tc={k:env[k] for k in ['RAYON_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']};applied={'selected_threads':cl['selected_threads'],'reservation_bytes':cl['reservation_bytes'],'thread_environment':tc,'resource_environment_sha256':h(q['target']['resource_environment']),'claim_argv_tail_sha256':h(q['target']['argv'][-20:]),'enforcement':'pre-process-authoritative-thread-env+source-bound-adapter+tree-rss-guard'}\n"
            "hs={'schema':'hawking.doctor_v5_target_resource_claim_handshake.v1','version':'2026-07-14.1','request_sha256':q['request_sha256'],'resource_claim_sha256':cs,'launch_nonce':q['launch_nonce'],'phase':q['phase'],'cell_id':q['cell_id'],'pid':os.getpid(),'parent_pid':os.getppid(),'target_executable_sha256':q['target']['executable']['sha256'],'target_argv_sha256':h(q['target']['argv']),'applied_resource_controls':applied,'created_wall_epoch':time.time(),'created_monotonic_ns':time.monotonic_ns(),'heavy_work_started':False,'awaiting_launcher_ack':True};hs['handshake_sha256']=h(hs)\n"
            "tmp=a.elastic_claim_handshake+'.tmp.'+str(os.getpid());fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.write(fd,json.dumps(hs,sort_keys=True).encode()+b'\\n');os.fsync(fd);os.close(fd);os.replace(tmp,a.elastic_claim_handshake)\n"
            "end=time.monotonic()+5;ap=pathlib.Path(a.elastic_claim_ack)\n"
            "while not ap.is_file() and time.monotonic()<end:time.sleep(.01)\n"
            "ack=json.loads(ap.read_bytes());ah=ack.pop('ack_sha256');assert ah==h(ack) and ack['request_sha256']==q['request_sha256'] and ack['resource_claim_sha256']==cs and ack['target_claim_handshake_sha256']==hs['handshake_sha256'] and ack['target_pid']==os.getpid() and ack['heavy_work_authorized'] is True\n"
            "payload={'fixture':'ok'}\n"
            "raw=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()\n"
            "value={'schema':'hawking.doctor_v5_fixture_phase_output.v1','version':'2026-07-14.1',"
            "'phase':a.phase,'cell_id':a.cell,'exact_output':True,'parity_verified':True,"
            "'zero_skips':True,'skipped_count':0,'payload':payload,'payload_sha256':hashlib.sha256(raw).hexdigest()}\n"
            "open(a.output,'w',encoding='utf-8').write(json.dumps(value,sort_keys=True)+'\\n')\n",
            encoding="utf-8"
        )
        self.base_invocation_entries = [
            self._build_inert_entry(
                phase=phase, cell_id=f"base-{phase}",
                state_path=self.directory / f"base-{phase}-state.json",
                contract_path=self.directory / "base-contract.json",
                generation=0,
            )[0]
            for phase in elastic.INVOCATION_PHASES
        ]
        invocation_manifest = elastic.build_invocation_manifest(
            self.base_invocation_entries
        )
        self.invocation_manifest_path = self.directory / "phase-invocations.json"
        self.invocation_manifest_path.write_text(
            json.dumps(invocation_manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.contract = elastic.build_contract(
            self.overlay, aggressive_overlay_path=self.overlay_path,
            host_probe=self.host_probe,
            host_plan_reference=elastic._file_reference(self.host_plan_path),
            invocation_manifest_path=self.invocation_manifest_path,
        )
        self.assertEqual("qualified", self.contract["status"],
                         self.contract["blockers"])
        self.assertEqual([], elastic.validate_contract(self.contract))
        initial_swap = aggressive.initial_swap_state(
            {"pressure_level": 1, "swap_used_mb": 0.0}, now_epoch=0.0
        )
        self.swap_state, raw_swap_decision = aggressive.advance_swap_state(
            initial_swap, {"pressure_level": 1, "swap_used_mb": 0.0},
            now_epoch=50.0, sealed_baseline_swap_mb=0.0,
        )
        self.swap_green = elastic.bind_aggressive_swap_decision(
            self.swap_state, raw_swap_decision
        )

    def tearDown(self) -> None:
        self.closed_runtime_fixture_patch.stop()
        self.temporary.cleanup()

    def test_signed_probe_binds_reviewed_asymmetric_topology(self) -> None:
        resource = self.contract["resource_policy"]
        self.assertEqual((28, 20, 8), (
            resource["physical_cores"], resource["performance_cores"],
            resource["efficiency_cores"],
        ))
        self.assertEqual(self.host_probe["probe_sha256"],
                         resource["host_probe_sha256"])
        forged = copy.deepcopy(self.host_probe)
        forged["topology"]["performance_cores"] = 28
        forged["topology"]["topology_sha256"] = host._hash_value(
            host._without(forged["topology"], "topology_sha256")
        )
        forged["probe_sha256"] = host._hash_value(
            host._without(forged, "probe_sha256")
        )
        blocked = elastic.build_contract(
            self.overlay, aggressive_overlay_path=self.overlay_path,
            host_probe=forged,
            host_plan_reference=elastic._file_reference(self.host_plan_path),
        )
        self.assertEqual("blocked", blocked["status"])
        self.assertTrue(any("topology" in row for row in blocked["blockers"]))

    def test_prepare_and_encode_are_mutually_exclusive_both_directions(self) -> None:
        state = elastic.new_state(self.contract)
        prepare = self._reservation("prepare", "prep", 8, 8_000_000_000)
        state = elastic.transition(
            state, self.contract, action="prepare_start", cell_id="prep",
            proof={**self._start_proof(
                "prepare", "prep", threads=8, ram=8_000_000_000
            ), "resource_reservation": prepare}, current_wall_epoch=70.0,
        )
        with self.assertRaisesRegex(elastic.ElasticError, "mutually exclusive"):
            self._start_encoder(state, "primary", "32B")
        state = elastic.transition(
            state, self.contract, action="prepare_complete", cell_id="prep",
            proof=self._completion_proof(
                state["prepare_owner"], "prepare", preempted=False
            ), current_wall_epoch=110.0,
        )
        state = self._start_encoder(state, "primary", "32B")
        with self.assertRaisesRegex(elastic.ElasticError, "mutually exclusive"):
            elastic.transition(
                state, self.contract, action="prepare_start", cell_id="prep2",
                proof=self._start_proof(
                    "prepare", "prep2", threads=8, ram=8_000_000_000
                ), current_wall_epoch=100.0,
            )

    def test_prepare_finalizer_overlap_needs_fresh_measured_aggregate(self) -> None:
        state = elastic.new_state(self.contract)
        prepare = self._reservation("prepare", "prep", 8, 8_000_000_000)
        state = elastic.transition(
            state, self.contract, action="prepare_start", cell_id="prep",
            proof={**self._start_proof(
                "prepare", "prep", threads=8, ram=8_000_000_000
            ), "resource_reservation": prepare}, current_wall_epoch=70.0,
        )
        finalizer = self._reservation("finalizer", "finish", 8, 7_000_000_000)
        finalizer_identity = self._identity("finish", "finalizer")
        with self.assertRaisesRegex(elastic.ElasticError, "overlap envelope"):
            elastic.transition(
                state, self.contract, action="finalizer_start", cell_id="finish",
                proof={**self._start_proof(
                    "finalizer", "finish", threads=8, ram=7_000_000_000,
                    process_identity=finalizer_identity,
                ), "resource_reservation": finalizer}, current_wall_epoch=100.0,
            )
        samples = self._overlap_samples(
            state["prepare_owner"], state["state_generation"]
        )
        envelope = self._overlap_envelope(
            state["prepare_owner"], finalizer, finalizer_identity, samples,
            state_generation=state["state_generation"],
        )
        admitted = elastic.transition(
            state, self.contract, action="finalizer_start", cell_id="finish",
            proof={**self._start_proof(
                "finalizer", "finish", threads=8, ram=7_000_000_000,
                process_identity=finalizer_identity,
            ), "resource_reservation": finalizer, "overlap_envelope": envelope},
            current_wall_epoch=100.0,
        )
        self.assertEqual(
            admitted["prepare_owner"]["overlap_envelope_sha256"],
            admitted["serial_finalizer_owner"]["overlap_envelope_sha256"],
        )
        red = copy.deepcopy(samples)
        red[-1]["pressure_level"] = 2
        red[-1]["sample_sha256"] = elastic._hash_value(
            elastic._without(red[-1], "sample_sha256")
        )
        red_envelope = self._overlap_envelope(
            state["prepare_owner"], finalizer, finalizer_identity, red,
            state_generation=state["state_generation"],
        )
        with self.assertRaisesRegex(elastic.ElasticError, "unauthenticated or red"):
            elastic.transition(
                state, self.contract, action="finalizer_start", cell_id="finish",
                proof={**self._start_proof(
                    "finalizer", "finish", threads=8, ram=7_000_000_000,
                    process_identity=finalizer_identity,
                ), "resource_reservation": finalizer,
                    "overlap_envelope": red_envelope}, current_wall_epoch=100.0,
            )

    def test_lending_authenticates_identity_probe_pressure_thermal_and_time(self) -> None:
        state = self._start_encoder(
            elastic.new_state(self.contract), "primary", "32B"
        )
        samples = self._idle_samples(state["encoder_owner"])
        allowed = self._lend(state, samples)
        self.assertTrue(allowed["allow"], allowed["blockers"])
        attacks = []
        for field, value in (
                ("primary_process_identity_sha256", "e" * 64),
                ("host_probe_sha256", "f" * 64),
                ("pressure_level", 2), ("thermal_state", "serious"),
                ("sampled_epoch", 101.0)):
            rows = copy.deepcopy(samples)
            rows[-1][field] = value
            rows[-1]["sample_sha256"] = elastic._hash_value(
                elastic._without(rows[-1], "sample_sha256")
            )
            attacks.append(rows)
        for rows in attacks:
            decision = self._lend(state, rows)
            self.assertFalse(decision["allow"], decision)
        forged_swap = copy.deepcopy(self.swap_green)
        forged_swap["decision"]["allow_launch"] = True
        forged_swap["decision"]["mode"] = "green"
        # Re-hashing the outer document cannot make an output detached from
        # the exact controller state semantically valid.
        forged_swap["decision"]["launch_limit"] = 999
        forged_swap["binding_sha256"] = elastic._hash_value(
            elastic._without(forged_swap, "binding_sha256")
        )
        blocked = elastic.lend_decision(
            state, self.contract, candidate_cell_id="companion",
            tier="14B", rate="q3", reservation_bytes=8_000_000_000,
            idle_samples=samples, aggressive_swap_state=self.swap_state,
            aggressive_swap_decision=forged_swap, now_epoch=100.0,
        )
        self.assertFalse(blocked["allow"])
        self.assertTrue(any("swap decision" in row for row in blocked["blockers"]))

    def test_blocked_contract_cannot_transition_or_lend(self) -> None:
        blocked_contract = copy.deepcopy(self.contract)
        blocked_contract["status"] = "blocked"
        blocked_contract["blockers"] = ["deliberate test blocker"]
        blocked_contract["contract_sha256"] = elastic._hash_value(
            elastic._without(blocked_contract, "contract_sha256")
        )
        state = elastic.new_state(blocked_contract)
        with self.assertRaisesRegex(elastic.ElasticError, "cannot transition"):
            elastic.transition(
                state, blocked_contract, action="prepare_start", cell_id="prep",
                proof={"resource_reservation": {}},
            )
        decision = elastic.lend_decision(
            state, blocked_contract, candidate_cell_id="companion",
            tier="14B", rate="q3", reservation_bytes=1,
            idle_samples=[], aggressive_swap_state=self.swap_state,
            aggressive_swap_decision=self.swap_green, now_epoch=100.0,
        )
        self.assertFalse(decision["allow"])
        self.assertIn("blocked/default-off elastic contract cannot lend resources",
                      decision["blockers"])

    def test_qualified_contract_revalidates_bound_files(self) -> None:
        self.assertEqual([], elastic.validate_contract(self.contract))
        binary_path = elastic.ROOT / self.contract[
            "source_bindings"]["thread_binary"]["path"]
        binary_path.write_bytes(b"drifted-after-contract")
        self.assertTrue(any("source binding" in row or "revalidation" in row
                            for row in elastic.validate_contract(self.contract)))

    def test_stale_green_swap_state_cannot_lend(self) -> None:
        state = self._start_encoder(
            elastic.new_state(self.contract), "primary", "32B"
        )
        initial = aggressive.initial_swap_state(
            {"pressure_level": 1, "swap_used_mb": 0.0}, now_epoch=0.0
        )
        stale_state, raw = aggressive.advance_swap_state(
            initial, {"pressure_level": 1, "swap_used_mb": 0.0},
            now_epoch=1.0, sealed_baseline_swap_mb=0.0,
        )
        binding = elastic.bind_aggressive_swap_decision(stale_state, raw)
        decision = elastic.lend_decision(
            state, self.contract, candidate_cell_id="companion",
            tier="14B", rate="q3", reservation_bytes=8_000_000_000,
            idle_samples=self._idle_samples(state["encoder_owner"]),
            aggressive_swap_state=stale_state,
            aggressive_swap_decision=binding, now_epoch=100.0,
        )
        self.assertFalse(decision["allow"])
        self.assertIn("aggressive swap controller sample is invalid or stale",
                      decision["blockers"])

    def test_twenty_thread_primary_is_exclusive_and_return_preempts(self) -> None:
        exclusive = self._start_encoder(
            elastic.new_state(self.contract), "exclusive", "72B"
        )
        blocked = self._lend(exclusive, self._idle_samples(exclusive["encoder_owner"]))
        self.assertFalse(blocked["allow"])
        self.assertTrue(any("20 performance cores" in row
                            for row in blocked["blockers"]))

        state = self._start_encoder(
            elastic.new_state(self.contract), "primary", "32B"
        )
        decision = self._lend(state, self._idle_samples(state["encoder_owner"]))
        state = elastic.transition(
            state, self.contract, action="companion_launch", cell_id="companion",
            proof={"lend_decision": decision,
                   "process_identity": self._identity("companion", "companion"),
                   "aggressive_swap_state": self.swap_state,
                   "aggressive_swap_decision": self.swap_green},
            current_wall_epoch=101.0,
        )
        encoder_completion = self._completion_proof(
            state["encoder_owner"], "encoder", preempted=False
        )
        state = elastic.transition(
            state, self.contract, action="encoder_return", cell_id="primary",
            proof=encoder_completion, current_wall_epoch=110.0,
        )
        self.assertTrue(state["companion_launch_closed"])
        self.assertTrue(state["companion_owner"]["preempt_required"])
        with self.assertRaisesRegex(elastic.ElasticError, "companion release"):
            elastic.transition(
                state, self.contract, action="finalizer_start", cell_id="finish",
                proof=self._start_proof(
                    "finalizer", "finish", threads=4, ram=4_000_000_000
                ), current_wall_epoch=110.0,
            )
        checkpoint = self._completion_proof(
            state["companion_owner"], "companion_checkpoint", preempted=True
        )
        state = elastic.transition(
            state, self.contract, action="companion_checkpointed",
            cell_id="companion", proof=checkpoint, current_wall_epoch=110.0,
        )
        state = elastic.transition(
            state, self.contract, action="finalizer_start", cell_id="finish",
            proof=self._start_proof(
                "finalizer", "finish", threads=4, ram=4_000_000_000
            ), current_wall_epoch=110.0,
        )
        self.assertEqual("finish", state["serial_finalizer_owner"]["cell_id"])

    def test_hash_chain_recovery_and_rollback_are_fail_closed(self) -> None:
        state = elastic.new_state(self.contract)
        state = elastic.transition(
            state, self.contract, action="prepare_start", cell_id="prep",
            proof=self._start_proof(
                "prepare", "prep", threads=4, ram=4_000_000_000
            ), current_wall_epoch=70.0,
        )
        recovery = elastic.crash_recovery_receipt(
            state, self.contract,
            observed_processes=[state["prepare_owner"]["process_identity"]],
            current_wall_epoch=100.0,
        )
        self.assertEqual(recovery["receipt_sha256"], elastic._hash_value(
            elastic._without(recovery, "receipt_sha256")
        ))
        tampered = copy.deepcopy(state)
        tampered["events"][0]["payload"]["cell_id"] = "other"
        tampered["state_sha256"] = elastic._hash_value(
            elastic._without(tampered, "state_sha256")
        )
        self.assertTrue(any("event chain" in row
                            for row in elastic.validate_state(tampered, self.contract)))
        state = elastic.transition(
            state, self.contract, action="prepare_complete", cell_id="prep",
            proof=self._completion_proof(
                state["prepare_owner"], "prepare", preempted=False
            ), current_wall_epoch=110.0,
        )
        receipt = elastic.rollback_receipt(
            state, self.contract, reason="test",
            restored_artifacts=[{"path": "x", "sha256": "9" * 64}],
        )
        self.assertFalse(receipt["runtime_defaults_changed"])
        self.assertFalse(receipt["completed_evidence_mutated"])
        elastic.STAGE_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=elastic.STAGE_ROOT) as raw:
            evidence_dir = Path(raw)
            recovery_ref = elastic.persist_evidence(
                evidence_dir / "recovery.json", recovery
            )
            rollback_ref = elastic.persist_evidence(
                evidence_dir / "rollback.json", receipt
            )
            state_ref = elastic.persist_state(
                evidence_dir / "state.json", state, self.contract
            )
            self.assertTrue(all(elastic._valid_sha(row["sha256"])
                                for row in (recovery_ref, rollback_ref, state_ref)))
            forged = copy.deepcopy(receipt)
            forged["completed_evidence_mutated"] = True
            forged["receipt_sha256"] = elastic._hash_value(
                elastic._without(forged, "receipt_sha256")
            )
            with self.assertRaises(elastic.ElasticError):
                elastic.persist_evidence(evidence_dir / "forged.json", forged)

    def test_overlap_uses_caller_time_spacing_and_exact_owner_identity(self) -> None:
        state = elastic.transition(
            elastic.new_state(self.contract), self.contract,
            action="prepare_start", cell_id="prep",
            proof=self._start_proof(
                "prepare", "prep", threads=8, ram=8_000_000_000
            ), current_wall_epoch=70.0,
        )
        finalizer = self._reservation("finalizer", "finish", 8, 7_000_000_000)
        identity = self._identity("finish", "finalizer")
        samples = self._overlap_samples(
            state["prepare_owner"], state["state_generation"]
        )

        too_close = copy.deepcopy(samples)
        too_close[1]["sampled_epoch"] = 82.0
        too_close[1]["sample_sha256"] = elastic._hash_value(
            elastic._without(too_close[1], "sample_sha256")
        )
        envelope = self._overlap_envelope(
            state["prepare_owner"], finalizer, identity, too_close,
            state_generation=state["state_generation"],
        )
        with self.assertRaisesRegex(elastic.ElasticError, "invalid or stale"):
            elastic.transition(
                state, self.contract, action="finalizer_start", cell_id="finish",
                proof={**self._start_proof(
                    "finalizer", "finish", threads=8, ram=7_000_000_000,
                    process_identity=identity,
                ), "resource_reservation": finalizer, "overlap_envelope": envelope},
                current_wall_epoch=100.0,
            )

        identity_tamper = copy.deepcopy(samples)
        identity_tamper[-1]["current_owner_process_identity_sha256"] = "f" * 64
        identity_tamper[-1]["sample_sha256"] = elastic._hash_value(
            elastic._without(identity_tamper[-1], "sample_sha256")
        )
        envelope = self._overlap_envelope(
            state["prepare_owner"], finalizer, identity, identity_tamper,
            state_generation=state["state_generation"],
        )
        with self.assertRaisesRegex(elastic.ElasticError, "unauthenticated"):
            elastic.transition(
                state, self.contract, action="finalizer_start", cell_id="finish",
                proof={**self._start_proof(
                    "finalizer", "finish", threads=8, ram=7_000_000_000,
                    process_identity=identity,
                ), "resource_reservation": finalizer, "overlap_envelope": envelope},
                current_wall_epoch=100.0,
            )

        initial = aggressive.initial_swap_state(
            {"pressure_level": 1, "swap_used_mb": 0.0}, now_epoch=0.0
        )
        fresh_state, raw = aggressive.advance_swap_state(
            initial, {"pressure_level": 1, "swap_used_mb": 0.0},
            now_epoch=150.0, sealed_baseline_swap_mb=0.0,
        )
        fresh_binding = elastic.bind_aggressive_swap_decision(fresh_state, raw)
        envelope = self._overlap_envelope(
            state["prepare_owner"], finalizer, identity, samples,
            state_generation=state["state_generation"],
        )
        replay_proof = {
            **self._start_proof(
                "finalizer", "finish", threads=8, ram=7_000_000_000,
                process_identity=identity,
            ), "resource_reservation": finalizer, "overlap_envelope": envelope,
            "aggressive_swap_state": fresh_state,
            "aggressive_swap_decision": fresh_binding,
        }
        with self.assertRaisesRegex(elastic.ElasticError, "invalid or stale"):
            elastic.transition(
                state, self.contract, action="finalizer_start", cell_id="finish",
                proof=replay_proof, current_wall_epoch=200.0,
            )

    def test_phase_start_requires_fresh_exact_swap_artifact(self) -> None:
        initial = aggressive.initial_swap_state(
            {"pressure_level": 1, "swap_used_mb": 0.0}, now_epoch=0.0
        )
        stale_state, raw = aggressive.advance_swap_state(
            initial, {"pressure_level": 1, "swap_used_mb": 0.0},
            now_epoch=1.0, sealed_baseline_swap_mb=0.0,
        )
        proof = self._start_proof(
            "prepare", "prep", threads=4, ram=4_000_000_000
        )
        proof["aggressive_swap_state"] = stale_state
        proof["aggressive_swap_decision"] = elastic.bind_aggressive_swap_decision(
            stale_state, raw
        )
        with self.assertRaisesRegex(elastic.ElasticError, "swap admission"):
            elastic.transition(
                elastic.new_state(self.contract), self.contract,
                action="prepare_start", cell_id="prep", proof=proof,
                current_wall_epoch=100.0,
            )

    def test_live_exact_process_cannot_complete_or_checkpoint(self) -> None:
        state = self._start_encoder(
            elastic.new_state(self.contract), "primary", "32B"
        )
        decision = self._lend(state, self._idle_samples(state["encoder_owner"]))
        state = elastic.transition(
            state, self.contract, action="companion_launch", cell_id="companion",
            proof={"lend_decision": decision,
                   "process_identity": self._identity("companion", "companion"),
                   "aggressive_swap_state": self.swap_state,
                   "aggressive_swap_decision": self.swap_green},
            current_wall_epoch=101.0,
        )
        owner = state["companion_owner"]
        live_proof = self._completion_proof(
            owner, "companion_checkpoint", preempted=True,
            active=[owner["process_identity"]],
        )
        with self.assertRaisesRegex(elastic.ElasticError, "exact process"):
            elastic.transition(
                state, self.contract, action="companion_checkpointed",
                cell_id="companion", proof=live_proof,
                current_wall_epoch=110.0,
            )
        forged = copy.deepcopy(live_proof)
        forged["exit_observation"]["exact_identity_running"] = False
        forged["exit_observation"]["exit_verified"] = True
        forged["exit_observation"]["observation_sha256"] = elastic._hash_value(
            elastic._without(forged["exit_observation"], "observation_sha256")
        )
        forged["output_receipt"] = elastic.build_phase_output_receipt(
            owner, self.contract, phase="companion_checkpoint",
            exit_observation=forged["exit_observation"],
            output={"sha256": "a" * 64, "bytes": 1},
            receipt={"sha256": "b" * 64, "bytes": 1}, checkpoint=True,
        )
        with self.assertRaisesRegex(elastic.ElasticError, "exact process"):
            elastic.transition(
                state, self.contract, action="companion_checkpointed",
                cell_id="companion", proof=forged, current_wall_epoch=110.0,
            )

    def test_completion_receipt_cannot_replay_across_process_identity(self) -> None:
        first = elastic.transition(
            elastic.new_state(self.contract), self.contract,
            action="prepare_start", cell_id="prep",
            proof=self._start_proof(
                "prepare", "prep", threads=4, ram=4_000_000_000
            ), current_wall_epoch=70.0,
        )
        old_completion = self._completion_proof(
            first["prepare_owner"], "prepare", preempted=False
        )
        replacement_identity = elastic.build_process_identity(
            pid=9_999, start_identity="boot:test:replacement",
            command_sha256="e" * 64,
        )
        replacement = elastic.transition(
            elastic.new_state(self.contract), self.contract,
            action="prepare_start", cell_id="prep",
            proof=self._start_proof(
                "prepare", "prep", threads=4, ram=4_000_000_000,
                process_identity=replacement_identity,
            ), current_wall_epoch=70.0,
        )
        with self.assertRaisesRegex(elastic.ElasticError, "exact process"):
            elastic.transition(
                replacement, self.contract, action="prepare_complete",
                cell_id="prep", proof=old_completion,
                current_wall_epoch=110.0,
            )

    def test_locked_cas_allows_only_one_scheduler_from_same_generation(self) -> None:
        observation = elastic.local_observer.observe_process_invocation(os.getpid())
        semantic_artifacts = {}
        argv = observation["argv"]
        if len(argv) >= 3 and argv[1] == "-m":
            module = argv[2]
            module_spec = importlib.util.find_spec(module)
            self.assertIsNotNone(module_spec)
            self.assertIsNotNone(module_spec.origin)
            module_closure = self.directory / "structural-module-closure.json"
            module_closure.write_text(json.dumps(
                elastic.build_python_module_closure(
                    module, [Path(module_spec.origin), Path(__file__)]
                ), sort_keys=True
            ) + "\n", encoding="utf-8")
            semantic_artifacts[f"module.{module}"] = module_closure
        structural = elastic.build_invocation_entry_from_observation(
            phase="prepare", observation=observation,
            semantic_artifact_paths=semantic_artifacts,
        )
        self.assertEqual("structural-only",
                         structural["production_execution_protocol"])
        manifest = elastic.build_invocation_manifest([structural])
        self.assertEqual("blocked", manifest["status"])
        self.assertTrue(any("inert-commit-v1-qualified" in row
                            for row in manifest["blockers"]))

    def test_production_cas_rejects_fake_pid_and_caller_empty_exit_list(self) -> None:
        elastic.STAGE_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=elastic.STAGE_ROOT) as raw:
            state_path = Path(raw) / "state.json"
            swap_path = Path(raw) / "swap-state.json"
            initial = elastic.new_state(self.contract)
            elastic.persist_state(state_path, initial, self.contract)
            swap_path.write_text(
                json.dumps(self.swap_state, sort_keys=True) + "\n", encoding="utf-8"
            )
            fake_proof = self._start_proof(
                "prepare", "fake", threads=4, ram=4_000_000_000,
                process_identity=elastic.build_process_identity(
                    pid=999_999, start_identity="ps-lstart:never",
                    command_sha256="f" * 64,
                ),
            )
            with mock.patch.object(
                    elastic.local_observer, "HEAVY_COMMAND_PATTERNS", ()), \
                    self.assertRaisesRegex(
                        elastic.ElasticError,
                        "cannot authenticate proposed|lacks exact inert launcher"
                    ):
                elastic.compare_and_swap_transition(
                    state_path, self.contract,
                    expected_state_sha256=initial["state_sha256"],
                    expected_state_generation=0, action="prepare_start",
                    cell_id="fake", proof=fake_proof,
                    current_wall_epoch=1.0,
                    aggressive_swap_state_path=swap_path,
                )
            self.assertEqual(initial, elastic._read_json(state_path))

            # The pure reducer may construct fixtures with synthetic identities,
            # but production completion refuses one that lacks a trusted CAS
            # acquisition receipt even when the caller supplies an empty list.
            synthetic = elastic.transition(
                initial, self.contract, action="prepare_start", cell_id="fake",
                proof=fake_proof, current_wall_epoch=70.0,
            )
            state_path.unlink()
            elastic.persist_state(state_path, synthetic, self.contract)
            owner = synthetic["prepare_owner"]
            output = Path(raw) / "output.bin"
            output.write_bytes(b"complete")
            completion_doc = elastic.build_worker_completion_document(
                owner, self.contract, phase="prepare",
                output_reference=elastic._file_reference(output),
            )
            completion_path = Path(raw) / "completion.json"
            completion_path.write_text(
                json.dumps(completion_doc, sort_keys=True) + "\n", encoding="utf-8"
            )
            caller_empty = self._completion_proof(
                owner, "prepare", preempted=False, active=[]
            )
            with self.assertRaisesRegex(
                    elastic.ElasticError,
                    "lacks one frozen inert entry|lacks a trusted lock-scoped"):
                elastic.compare_and_swap_transition(
                    state_path, self.contract,
                    expected_state_sha256=synthetic["state_sha256"],
                    expected_state_generation=synthetic["state_generation"],
                    action="prepare_complete", cell_id="fake", proof=caller_empty,
                    current_wall_epoch=2.0,
                )
            self.assertIsNotNone(elastic._read_json(state_path)["prepare_owner"])

    def test_inert_cas_launch_and_request_bound_semantic_completion(self) -> None:
        elastic.STAGE_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=elastic.STAGE_ROOT) as raw:
            state_path = Path(raw) / "state.json"
            contract_path = Path(raw) / "contract.json"
            swap_path = Path(raw) / "swap-state.json"
            output_path = self.directory / "real-0-output.json"
            entry, request, _ = self._build_inert_entry(
                phase="prepare", cell_id="real", state_path=state_path,
                contract_path=contract_path, generation=0,
                target_args=["--output", str(output_path), "--phase", "prepare",
                             "--cell", "real"],
            )
            self.contract = self._contract_with_extra_invocations([entry])
            contract_path.write_text(
                json.dumps(self.contract, sort_keys=True) + "\n", encoding="utf-8"
            )
            initial = elastic.new_state(self.contract)
            elastic.persist_state(state_path, initial, self.contract)
            swap_path.write_text(
                json.dumps(self.swap_state, sort_keys=True) + "\n", encoding="utf-8"
            )
            command = [row["literal"] for row in entry["argv_template"]]
            self.assertEqual([], elastic._invocation_entry_errors(entry))
            process = subprocess.Popen(command, cwd=entry["cwd"], env=dict(os.environ))
            deadline = time.time() + 5.0
            while not Path(request["paths"]["handshake"]).is_file() \
                    and time.time() < deadline:
                time.sleep(0.01)
            self.assertTrue(Path(request["paths"]["handshake"]).is_file())
            identity = elastic.trusted_process_identity(process.pid)
            observed_invocation = elastic.local_observer.observe_process_invocation(
                process.pid
            )
            self.assertEqual(command, observed_invocation["argv"])
            self.assertEqual(entry["executable"], observed_invocation["executable"])
            self.assertEqual(entry["cwd"], observed_invocation["cwd"])
            self.assertEqual(set(entry["environment_allowlist"]), set(
                observed_invocation["environment_value_sha256s"]
            ))
            resources = elastic.local_observer._direct_resources()
            resources.update({
                "pressure_level": 1, "swap_used_mb": 0.0,
                "thermal_green": True, "ac_power": True, "probe_valid": True,
            })
            resources["resource_sha256"] = elastic.local_observer._hash_value(
                elastic.local_observer._without(resources, "resource_sha256")
            )
            try:
                with mock.patch.object(
                        elastic.local_observer, "HEAVY_COMMAND_PATTERNS", ()), \
                        mock.patch.object(
                            elastic.local_observer, "_direct_resources",
                            return_value=resources,
                        ):
                    running, start_receipt = elastic.compare_and_swap_transition(
                        state_path, self.contract,
                        expected_state_sha256=initial["state_sha256"],
                        expected_state_generation=0, action="prepare_start",
                        cell_id="real", proof=self._start_proof(
                            "prepare", "real", threads=4, ram=4_000_000_000,
                            process_identity=identity,
                        ), current_wall_epoch=1.0,
                        aggressive_swap_state_path=swap_path,
                    )
                self.assertEqual(request["request_sha256"],
                                 start_receipt["launch_request_sha256"])
                self.assertTrue(elastic._valid_sha(
                    start_receipt["inert_handshake_sha256"]
                ))
                with self.assertRaisesRegex(elastic.ElasticError, "CAS conflict"):
                    elastic.compare_and_swap_transition(
                        state_path, self.contract,
                        expected_state_sha256=initial["state_sha256"],
                        expected_state_generation=0, action="prepare_start",
                        cell_id="real", proof=self._start_proof(
                            "prepare", "real", threads=4,
                            ram=4_000_000_000, process_identity=identity,
                        ), current_wall_epoch=1.0,
                        aggressive_swap_state_path=swap_path,
                    )
                self.assertEqual(0, process.wait(timeout=5))
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            with self.assertRaisesRegex(
                    elastic.ElasticError, "caller-selected completion paths"):
                elastic.compare_and_swap_transition(
                    state_path, self.contract,
                    expected_state_sha256=running["state_sha256"],
                    expected_state_generation=running["state_generation"],
                    action="prepare_complete", cell_id="real",
                    proof={"output_path": str(output_path)},
                    current_wall_epoch=2.0,
                )
            completed, receipt = elastic.compare_and_swap_transition(
                state_path, self.contract,
                expected_state_sha256=running["state_sha256"],
                expected_state_generation=running["state_generation"],
                action="prepare_complete", cell_id="real", proof={},
                current_wall_epoch=2.0,
            )
            self.assertIsNone(completed["prepare_owner"])
            self.assertEqual("trusted-local-observer-under-state-lock",
                             receipt["production_authority"])
            self.assertTrue(Path(request["paths"]["semantic_receipt"]).is_file())

    def test_inert_start_rejects_missing_stale_and_forged_handshakes(self) -> None:
        for mode in ("missing", "stale", "forged"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                    dir=elastic.STAGE_ROOT) as raw:
                state_path, swap_path, initial, _, request, process, identity = \
                    self._spawn_inert_prepare_fixture(raw, cell_id=f"hs-{mode}")
                handshake_path = Path(request["paths"]["handshake"])
                if mode == "missing":
                    handshake_path.unlink()
                    expected = "elastic input|handshake|cannot open"
                else:
                    handshake = json.loads(handshake_path.read_bytes())
                    if mode == "stale":
                        handshake["created_wall_epoch"] = 0.0
                        handshake["created_monotonic_ns"] = 1
                        expected = "stale or future-dated"
                    else:
                        handshake["launch_nonce"] = "f" * 64
                        expected = "identity/request binding differs"
                    handshake["handshake_sha256"] = elastic._hash_value(
                        elastic._without(handshake, "handshake_sha256")
                    )
                    handshake_path.write_text(
                        json.dumps(handshake, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                try:
                    with mock.patch.object(
                            elastic.local_observer,
                            "HEAVY_COMMAND_PATTERNS", ()), \
                            self.assertRaisesRegex(elastic.ElasticError, expected):
                        elastic.compare_and_swap_transition(
                            state_path, self.contract,
                            expected_state_sha256=initial["state_sha256"],
                            expected_state_generation=0,
                            action="prepare_start", cell_id=f"hs-{mode}",
                            proof=self._start_proof(
                                "prepare", f"hs-{mode}", threads=4,
                                ram=4_000_000_000, process_identity=identity,
                            ), current_wall_epoch=1.0,
                            aggressive_swap_state_path=swap_path,
                        )
                    self.assertEqual(initial, elastic._read_json(state_path))
                finally:
                    process.terminate()
                    process.wait(timeout=5)

    def test_inert_start_binds_resource_claim_and_requires_fresh_outputs(self) -> None:
        for mode in ("resource-claim", "stale-output"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                    dir=elastic.STAGE_ROOT) as raw:
                state_path, swap_path, initial, _, request, process, identity = \
                    self._spawn_inert_prepare_fixture(raw, cell_id=f"start-{mode}")
                if mode == "stale-output":
                    Path(request["paths"]["output"]).write_bytes(b"stale")
                    expected = "one-shot artifact already exists"
                    threads = 4
                else:
                    expected = "resource claim differs"
                    threads = 8
                try:
                    with mock.patch.object(
                            elastic.local_observer,
                            "HEAVY_COMMAND_PATTERNS", ()), \
                            self.assertRaisesRegex(elastic.ElasticError, expected):
                        elastic.compare_and_swap_transition(
                            state_path, self.contract,
                            expected_state_sha256=initial["state_sha256"],
                            expected_state_generation=0,
                            action="prepare_start", cell_id=f"start-{mode}",
                            proof=self._start_proof(
                                "prepare", f"start-{mode}", threads=threads,
                                ram=4_000_000_000, process_identity=identity,
                            ), current_wall_epoch=1.0,
                            aggressive_swap_state_path=swap_path,
                        )
                    self.assertEqual(initial, elastic._read_json(state_path))
                finally:
                    process.terminate()
                    process.wait(timeout=5)

    def test_resource_claim_mapping_rejects_underclaim_and_base_env_conflict(self) -> None:
        with self.assertRaisesRegex(
                elastic.ElasticError, "conflicting unbound thread control"):
            self._build_inert_entry(
                phase="prepare", cell_id="underclaim-argv",
                state_path=self.directory / "underclaim-state.json",
                contract_path=self.directory / "underclaim-contract.json",
                generation=0, target_args=["--threads", "20"],
            )
        conflicting = dict(os.environ)
        conflicting["OMP_NUM_THREADS"] = "20"
        with self.assertRaisesRegex(
                elastic.ElasticError, "base environment conflicts"):
            self._build_inert_entry(
                phase="prepare", cell_id="underclaim-env",
                state_path=self.directory / "underclaim-env-state.json",
                contract_path=self.directory / "underclaim-env-contract.json",
                generation=0, target_environment=conflicting,
            )

    def test_python_target_closure_is_honestly_structural_and_rejects_omission(self) -> None:
        dependency = self.directory / "closure_dependency.py"
        dependency.write_text("VALUE = 1\n", encoding="utf-8")
        target = self.directory / "closure_target.py"
        target.write_text(
            "import closure_dependency\nprint(closure_dependency.VALUE)\n",
            encoding="utf-8",
        )
        closure = elastic.build_target_dependency_closure(
            target, interpreter=Path(self.python_executable)
        )
        self.assertFalse(closure["inventory_complete"])
        self.assertFalse(closure["qualification_authority"])
        self.assertFalse(closure["system_site_packages_excluded"])
        self.assertFalse(closure["native_dependency_closure_proven"])
        unresolved = self.real_closed_target_runtime_errors({
            "target": {"executable": {"path": self.python_executable}},
            "resource_claim": {"reservation_bytes": None},
        })
        self.assertTrue(any("closed Python" in row for row in unresolved))
        self.assertTrue(any("RAM ceiling" in row for row in unresolved))
        self.assertTrue(any("thread-count" in row for row in unresolved))
        paths = {row["path"] for row in closure["workspace_sources"]}
        self.assertIn(elastic._workspace_artifact_reference(dependency)["path"], paths)
        omitted = copy.deepcopy(closure)
        omitted["workspace_sources"] = [
            row for row in omitted["workspace_sources"]
            if row["path"] != elastic._workspace_artifact_reference(dependency)["path"]
        ]
        omitted["closure_sha256"] = elastic._hash_value(
            elastic._without(omitted, "closure_sha256")
        )
        closure_path = self.directory / "omitted-closure.json"
        closure_path.write_text(
            json.dumps(omitted, sort_keys=True) + "\n", encoding="utf-8"
        )
        command = {
            "argv": [self.python_executable, "-I", "-S", str(target)]
        }
        self.assertTrue(elastic._target_dependency_closure_errors(
            elastic._workspace_artifact_reference(closure_path), command=command
        ))
        dynamic = self.directory / "dynamic-target.py"
        dynamic.write_text("exec('x = 1')\n", encoding="utf-8")
        with self.assertRaisesRegex(elastic.ElasticError, "dynamic code/import"):
            elastic.build_target_dependency_closure(
                dynamic, interpreter=Path(self.python_executable)
            )

    def test_inert_target_drift_is_blocked_before_and_after_cas(self) -> None:
        for timing in ("before", "after"):
            with self.subTest(timing=timing), tempfile.TemporaryDirectory(
                    dir=elastic.STAGE_ROOT) as raw:
                target = self.directory / f"drift-{timing}-target.py"
                original = self.fixture_target.read_bytes()
                target.write_bytes(original)
                state_path, swap_path, initial, _, request, process, identity = \
                    self._spawn_inert_prepare_fixture(
                        raw, cell_id=f"drift-{timing}", target_script=target,
                    )
                try:
                    if timing == "before":
                        target.write_bytes(original + b"# pre-CAS drift\n")
                        with mock.patch.object(
                                elastic.local_observer,
                                "HEAVY_COMMAND_PATTERNS", ()), \
                                self.assertRaisesRegex(
                                    elastic.ElasticError,
                                    "invalid elastic state|invocation|drifted",
                                ):
                            elastic.compare_and_swap_transition(
                                state_path, self.contract,
                                expected_state_sha256=initial["state_sha256"],
                                expected_state_generation=0,
                                action="prepare_start", cell_id="drift-before",
                                proof=self._start_proof(
                                    "prepare", "drift-before", threads=4,
                                    ram=4_000_000_000,
                                    process_identity=identity,
                                ), current_wall_epoch=1.0,
                                aggressive_swap_state_path=swap_path,
                            )
                        self.assertEqual(initial, elastic._read_json(state_path))
                    else:
                        original_atomic = elastic._atomic_json

                        def mutate_after_commit(path: Path, value: dict) -> None:
                            original_atomic(path, value)
                            if Path(path) == Path(request["paths"]["commit"]):
                                target.write_bytes(original + b"# post-CAS drift\n")

                        resources = self._green_direct_resources()
                        with mock.patch.object(
                                elastic.local_observer,
                                "HEAVY_COMMAND_PATTERNS", ()), \
                                mock.patch.object(
                                    elastic.local_observer, "_direct_resources",
                                    return_value=resources,
                                ), mock.patch.object(
                                    elastic, "_atomic_json",
                                    side_effect=mutate_after_commit,
                                ):
                            updated, _ = elastic.compare_and_swap_transition(
                                state_path, self.contract,
                                expected_state_sha256=initial["state_sha256"],
                                expected_state_generation=0,
                                action="prepare_start", cell_id="drift-after",
                                proof=self._start_proof(
                                    "prepare", "drift-after", threads=4,
                                    ram=4_000_000_000,
                                    process_identity=identity,
                                ), current_wall_epoch=1.0,
                                aggressive_swap_state_path=swap_path,
                            )
                        self.assertIsNotNone(updated["prepare_owner"])
                        self.assertEqual(78, process.wait(timeout=5))
                        self.assertFalse(Path(
                            request["paths"]["worker_receipt"]
                        ).exists())
                finally:
                    target.write_bytes(original)
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=5)

    def test_completion_rejects_false_nonzero_and_malformed_validators(self) -> None:
        false_source = (
            "import argparse,hashlib,json,pathlib\n"
            "p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--output');a=p.parse_args()\n"
            "q=json.loads(pathlib.Path(a.request).read_bytes());raw=pathlib.Path(a.output).read_bytes();"
            "hs=json.loads(pathlib.Path(q['paths']['target_claim_handshake']).read_bytes());ack=json.loads(pathlib.Path(q['paths']['target_claim_ack']).read_bytes());guard=json.loads(pathlib.Path(q['paths']['target_resource_guard']).read_bytes());"
            "rp=pathlib.Path(a.output).resolve();root=pathlib.Path.cwd().resolve();"
            "display=str(rp.relative_to(root))\n"
            "r={'schema':'hawking.doctor_v5_phase_semantic_validator_receipt.v1','version':'2026-07-14.1',"
            "'validator_profile':'fixture-false','phase':q['phase'],'cell_id':q['cell_id'],"
            "'request_sha256':q['request_sha256'],'output':{'path':display,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)},"
            "'resource_claim_sha256':q['resource_claim_sha256'],'target_claim_handshake_sha256':hs['handshake_sha256'],'target_claim_ack_sha256':ack['ack_sha256'],"
            "'target_process_identity_sha256':ack['target_process_identity']['process_identity_sha256'],'target_resource_guard_sha256':guard['guard_sha256'],"
            "'exact_output':True,'parity_verified':False,'zero_skips':True,'skipped_count':0,'semantic_checks':['fixture']}\n"
            "r['receipt_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest();print(json.dumps(r))\n"
        )
        drift_source = false_source.replace(
            "'parity_verified':False", "'parity_verified':True"
        ).replace(
            ";print(json.dumps(r))",
            ";pathlib.Path(a.output).write_bytes(raw+b' ');print(json.dumps(r))",
        )
        variants = {
            "false": (false_source, "exact/parity/zero-skip"),
            "nonzero": ("raise SystemExit(65)\n", "nonzero"),
            "malformed": ("print('{')\n", "stdout is malformed"),
            "output-drift": (drift_source, "output changed during semantic"),
        }
        for mode, (source, expected) in variants.items():
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                    dir=elastic.STAGE_ROOT) as raw:
                validator = self.directory / f"validator-{mode}.py"
                validator.write_text(source, encoding="utf-8")
                state_path, swap_path, initial, _, _, process, identity = \
                    self._spawn_inert_prepare_fixture(
                        raw, cell_id=f"validator-{mode}",
                        validator_script=validator,
                    )
                try:
                    running, _ = self._admit_inert_prepare(
                        state_path, swap_path, initial,
                        cell_id=f"validator-{mode}", identity=identity,
                    )
                    self.assertEqual(0, process.wait(timeout=5))
                    with self.assertRaisesRegex(elastic.ElasticError, expected):
                        elastic.compare_and_swap_transition(
                            state_path, self.contract,
                            expected_state_sha256=running["state_sha256"],
                            expected_state_generation=running["state_generation"],
                            action="prepare_complete",
                            cell_id=f"validator-{mode}", proof={},
                            current_wall_epoch=2.0,
                        )
                    self.assertIsNotNone(
                        elastic._read_json(state_path)["prepare_owner"]
                    )
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=5)

    def test_completion_rejects_missing_release_worker_and_output_drift(self) -> None:
        variants = {
            "missing-release": "cannot open|elastic input",
            "worker-forgery": "worker completion",
            "output-drift": "worker completion",
        }
        for mode, expected in variants.items():
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                    dir=elastic.STAGE_ROOT) as raw:
                state_path, swap_path, initial, _, request, process, identity = \
                    self._spawn_inert_prepare_fixture(
                        raw, cell_id=f"completion-{mode}"
                    )
                try:
                    running, _ = self._admit_inert_prepare(
                        state_path, swap_path, initial,
                        cell_id=f"completion-{mode}", identity=identity,
                    )
                    self.assertEqual(0, process.wait(timeout=5))
                    if mode == "missing-release":
                        Path(request["paths"]["release"]).unlink()
                    elif mode == "worker-forgery":
                        worker_path = Path(request["paths"]["worker_receipt"])
                        worker = json.loads(worker_path.read_bytes())
                        worker["target_returncode"] = 1
                        worker["worker_completion_sha256"] = elastic._hash_value(
                            elastic._without(worker, "worker_completion_sha256")
                        )
                        worker_path.write_text(
                            json.dumps(worker, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                    else:
                        with Path(request["paths"]["output"]).open("ab") as handle:
                            handle.write(b" ")
                    with self.assertRaisesRegex(elastic.ElasticError, expected):
                        elastic.compare_and_swap_transition(
                            state_path, self.contract,
                            expected_state_sha256=running["state_sha256"],
                            expected_state_generation=running["state_generation"],
                            action="prepare_complete",
                            cell_id=f"completion-{mode}", proof={},
                            current_wall_epoch=2.0,
                        )
                    self.assertIsNotNone(
                        elastic._read_json(state_path)["prepare_owner"]
                    )
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=5)

    def test_completion_rejects_missing_forged_stale_claim_and_guard_receipts(self) -> None:
        for mode in ("missing-ack", "forged-claim", "stale-claim", "forged-guard"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                    dir=elastic.STAGE_ROOT) as raw:
                state_path, swap_path, initial, _, request, process, identity = \
                    self._spawn_inert_prepare_fixture(
                        raw, cell_id=f"claim-completion-{mode}"
                    )
                try:
                    running, _ = self._admit_inert_prepare(
                        state_path, swap_path, initial,
                        cell_id=f"claim-completion-{mode}", identity=identity,
                    )
                    self.assertEqual(0, process.wait(timeout=5))
                    if mode == "missing-ack":
                        Path(request["paths"]["target_claim_ack"]).unlink()
                    elif mode in {"forged-claim", "stale-claim"}:
                        path = Path(request["paths"]["target_claim_handshake"])
                        value = json.loads(path.read_bytes())
                        if mode == "forged-claim":
                            value["resource_claim_sha256"] = "f" * 64
                        else:
                            value["created_wall_epoch"] = 0.0
                            value["created_monotonic_ns"] = 1
                        value["handshake_sha256"] = elastic._hash_value(
                            elastic._without(value, "handshake_sha256")
                        )
                        path.write_text(
                            json.dumps(value, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                    else:
                        path = Path(request["paths"]["target_resource_guard"])
                        value = json.loads(path.read_bytes())
                        value["max_tree_rss_bytes"] = 5_000_000_000
                        value["guard_sha256"] = elastic._hash_value(
                            elastic._without(value, "guard_sha256")
                        )
                        path.write_text(
                            json.dumps(value, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                    with self.assertRaisesRegex(
                            elastic.ElasticError,
                            "resource-claim|resource guard|cannot open|elastic input"):
                        elastic.compare_and_swap_transition(
                            state_path, self.contract,
                            expected_state_sha256=running["state_sha256"],
                            expected_state_generation=running["state_generation"],
                            action="prepare_complete",
                            cell_id=f"claim-completion-{mode}", proof={},
                            current_wall_epoch=2.0,
                        )
                    self.assertIsNotNone(
                        elastic._read_json(state_path)["prepare_owner"]
                    )
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=5)

    def test_missing_prework_claim_handshake_kills_group_and_writes_failure(self) -> None:
        target = self.directory / "no-claim-handshake-target.py"
        target.write_text("raise SystemExit(0)\n", encoding="utf-8")
        with tempfile.TemporaryDirectory(dir=elastic.STAGE_ROOT) as raw:
            state_path, swap_path, initial, _, request, process, identity = \
                self._spawn_inert_prepare_fixture(
                    raw, cell_id="missing-prework-claim", target_script=target,
                )
            running, _ = self._admit_inert_prepare(
                state_path, swap_path, initial,
                cell_id="missing-prework-claim", identity=identity,
            )
            self.assertIsNotNone(running["prepare_owner"])
            self.assertEqual(78, process.wait(timeout=5))
            failure_path = Path(request["paths"]["target_claim_failure"])
            self.assertTrue(failure_path.is_file())
            failure = json.loads(failure_path.read_bytes())
            self.assertEqual(
                "target-exited-before-claim-handshake", failure["failure_code"]
            )
            self.assertFalse(failure["heavy_work_authorized"])
            self.assertFalse(Path(request["paths"]["target_claim_ack"]).exists())
            self.assertFalse(Path(request["paths"]["worker_receipt"]).exists())
            target_pid = failure["target_pid"]
            ps = subprocess.run(
                ["/bin/ps", "-p", str(target_pid), "-o", "pid="],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual("", ps.stdout.strip())

    def test_direct_red_pressure_overrides_fabricated_green_swap_proof(self) -> None:
        elastic.STAGE_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=elastic.STAGE_ROOT) as raw:
            state_path, swap_path, initial, _, _, process, identity = \
                self._spawn_inert_prepare_fixture(raw, cell_id="red")
            red = elastic.local_observer._direct_resources()
            red["pressure_level"] = 4
            red["resource_sha256"] = elastic.local_observer._hash_value(
                elastic.local_observer._without(red, "resource_sha256")
            )
            forged_green = self._start_proof(
                "prepare", "red", threads=4, ram=4_000_000_000,
                process_identity=identity,
            )
            try:
                with mock.patch.object(
                        elastic.local_observer, "HEAVY_COMMAND_PATTERNS", ()), \
                        mock.patch.object(
                            elastic.local_observer, "_direct_resources", return_value=red
                        ), self.assertRaisesRegex(
                            elastic.ElasticError, "resources are not green"
                        ):
                    elastic.compare_and_swap_transition(
                        state_path, self.contract,
                        expected_state_sha256=initial["state_sha256"],
                        expected_state_generation=0, action="prepare_start",
                        cell_id="red", proof=forged_green,
                        current_wall_epoch=1.0,
                        aggressive_swap_state_path=swap_path,
                    )
            finally:
                process.terminate()
                process.wait(timeout=5)

    def test_persisted_swap_prior_overrides_caller_green_document(self) -> None:
        elastic.STAGE_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=elastic.STAGE_ROOT) as raw:
            state_path, swap_path, initial, _, _, process, identity = \
                self._spawn_inert_prepare_fixture(
                    raw, cell_id="persisted-prior",
                    swap_document='{"mode":"green"}\n',
                )
            resources = elastic.local_observer._direct_resources()
            resources.update({
                "pressure_level": 1, "swap_used_mb": 0.0,
                "thermal_green": True, "ac_power": True, "probe_valid": True,
            })
            resources["resource_sha256"] = elastic.local_observer._hash_value(
                elastic.local_observer._without(resources, "resource_sha256")
            )
            try:
                with mock.patch.object(
                        elastic.local_observer, "HEAVY_COMMAND_PATTERNS", ()), \
                        mock.patch.object(
                            elastic.local_observer, "_direct_resources",
                            return_value=resources,
                        ), self.assertRaisesRegex(
                            elastic.ElasticError,
                            "trusted swap transition prior is invalid"
                        ):
                    elastic.compare_and_swap_transition(
                        state_path, self.contract,
                        expected_state_sha256=initial["state_sha256"],
                        expected_state_generation=0, action="prepare_start",
                        cell_id="persisted-prior", proof=self._start_proof(
                            "prepare", "persisted-prior", threads=4,
                            ram=4_000_000_000, process_identity=identity,
                        ), current_wall_epoch=1.0,
                        aggressive_swap_state_path=swap_path,
                    )
            finally:
                process.terminate()
                process.wait(timeout=5)

    def test_production_cas_refuses_external_campaign_heavy_owner(self) -> None:
        elastic.STAGE_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=elastic.STAGE_ROOT) as raw:
            state_path, swap_path, initial, _, _, process, identity = \
                self._spawn_inert_prepare_fixture(raw, cell_id="blocked")
            original_observe = elastic.local_observer.observe_under_lock

            def injected_observer(*args, **kwargs):
                return self._inject_external_heavy(
                    original_observe(*args, **kwargs)
                )
            try:
                with mock.patch.object(
                        elastic.local_observer, "HEAVY_COMMAND_PATTERNS", ()), \
                        mock.patch.object(
                            elastic.local_observer, "observe_under_lock",
                            side_effect=injected_observer,
                        ), self.assertRaisesRegex(
                            elastic.ElasticError,
                            "external campaign-wide heavy owner",
                        ):
                    elastic.compare_and_swap_transition(
                            state_path, self.contract,
                            expected_state_sha256=initial["state_sha256"],
                            expected_state_generation=0, action="prepare_start",
                            cell_id="blocked", proof=self._start_proof(
                                "prepare", "blocked", threads=4,
                                ram=4_000_000_000, process_identity=identity,
                            ), current_wall_epoch=1.0,
                            aggressive_swap_state_path=swap_path,
                        )
            finally:
                process.terminate()
                process.wait(timeout=5)

    def test_rejected_outside_stage_path_creates_no_parent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            missing = Path(raw) / "must-not-be-created"
            with self.assertRaisesRegex(elastic.ElasticError, "below elastic_v1"):
                elastic._safe_stage_output(missing / "state.json")
            self.assertFalse(missing.exists())

    def test_malformed_public_validators_fail_closed(self) -> None:
        self.assertTrue(elastic.validate_state({}, None))
        malformed_state = elastic.new_state(self.contract)
        malformed_state["status"] = []
        malformed_state["state_sha256"] = elastic._hash_value(
            elastic._without(malformed_state, "state_sha256")
        )
        self.assertTrue(elastic.validate_state(malformed_state, self.contract))
        with self.assertRaises(elastic.ElasticError):
            elastic.persist_evidence(
                elastic.STAGE_ROOT / "never-written.json", {"schema": []}
            )
        malformed = copy.deepcopy(self.swap_state)
        malformed["previous_swap_mb"] = "not-a-number"
        malformed["state_sha256"] = elastic._hash_value(
            elastic._without(malformed, "state_sha256")
        )
        with self.assertRaises(elastic.ElasticError):
            elastic.bind_aggressive_swap_decision(
                malformed, self.swap_green["decision"]
            )

    def test_vllm_heavy_owner_patterns_are_narrow_and_cover_metal_server(self) -> None:
        def matches(command: str) -> bool:
            return any(
                pattern.search(command.lower())
                for pattern in elastic.local_observer.HEAVY_COMMAND_PATTERNS
            )

        self.assertTrue(matches(
            "/Users/scammermike/Downloads/vllm-metal-cx/.venv-vllm-metal/"
            "bin/python -"
        ))
        self.assertTrue(matches("vllm serve org/model --port 8000"))
        self.assertTrue(matches(
            "python3 -m vllm.entrypoints.openai.api_server --model fixture"
        ))
        self.assertFalse(matches("/usr/bin/python3 -"))
        self.assertFalse(matches("python3 mlx_small_fixture.py"))

    def test_leased_wrapper_descendant_snapshot_is_exact_ppid_closed(self) -> None:
        def row(pid: int, ppid: int) -> dict:
            return {
                "pid": pid, "ppid": ppid,
                "start_identity": f"ps-lstart:{pid}",
                "command_sha256": f"{pid:064x}",
                "process_generation_sha256": f"{pid + 100:064x}",
            }

        descendants = elastic.local_observer._descendant_rows(
            100, [row(100, 1), row(101, 100), row(102, 101), row(200, 1)]
        )
        self.assertEqual([101, 102], [child["pid"] for child in descendants])
        self.assertEqual([100, 101], [child["ppid"] for child in descendants])
        self.assertTrue(all(
            child["descendant_sha256"]
            == elastic.local_observer._hash_value(
                elastic.local_observer._without(child, "descendant_sha256")
            ) for child in descendants
        ))

    def test_trusted_host_gate_detects_heavy_owner_without_caller_snapshot(self) -> None:
        elastic.STAGE_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=elastic.STAGE_ROOT) as raw:
            state_path = Path(raw) / "state.json"
            swap_path = Path(raw) / "swap-state.json"
            elastic.persist_state(
                state_path, elastic.new_state(self.contract), self.contract
            )
            swap_path.write_text(
                json.dumps(self.swap_state, sort_keys=True) + "\n", encoding="utf-8"
            )
            original_observe = elastic.local_observer.observe_with_state_lock

            def injected_observer(*args, **kwargs):
                return self._inject_external_heavy(
                    original_observe(*args, **kwargs)
                )

            with mock.patch.object(
                    elastic.local_observer, "HEAVY_COMMAND_PATTERNS", ()), \
                    mock.patch.object(
                        elastic.local_observer, "observe_with_state_lock",
                        side_effect=injected_observer,
                    ):
                gate = host.evaluate_gate_from_local_observer(
                    self.host_plan, elastic_state_path=state_path,
                    aggressive_swap_state_path=swap_path,
                )
            self.assertFalse(gate["ok"])
            self.assertFalse(gate["production_authorized"])
            self.assertGreaterEqual(gate["heavy_owner_count"], 1)
            self.assertIn("trusted local observer found active heavy owners",
                          gate["blockers"])

    def test_invocation_manifest_source_drift_escape_and_symlink_fail_closed(self) -> None:
        self.assertEqual([], elastic.validate_contract(self.contract))
        outside_manifest = elastic.build_invocation_manifest(
            self.base_invocation_entries
        )
        with tempfile.TemporaryDirectory() as outside:
            outside_path = Path(outside) / "manifest.json"
            outside_path.write_text(
                json.dumps(outside_manifest, sort_keys=True) + "\n", encoding="utf-8"
            )
            blocked = elastic.build_contract(
                self.overlay, aggressive_overlay_path=self.overlay_path,
                host_probe=self.host_probe,
                host_plan_reference=elastic._file_reference(self.host_plan_path),
                invocation_manifest_path=outside_path,
            )
            self.assertEqual("blocked", blocked["status"])
            self.assertTrue(any("inside the workspace" in row
                                for row in blocked["blockers"]))

        link = self.directory / "manifest-link.json"
        link.symlink_to(self.invocation_manifest_path)
        blocked = elastic.build_contract(
            self.overlay, aggressive_overlay_path=self.overlay_path,
            host_probe=self.host_probe,
            host_plan_reference=elastic._file_reference(self.host_plan_path),
            invocation_manifest_path=link,
        )
        self.assertEqual("blocked", blocked["status"])
        self.assertTrue(any("symlink" in row for row in blocked["blockers"]))

        self.invocation_manifest_path.write_text(
            self.invocation_manifest_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
        self.assertTrue(any("source binding" in row or "manifest" in row
                            for row in elastic.validate_contract(self.contract)))

    def test_phase_invocation_allowlist_rejects_sleep_argv_cwd_env_and_accepts_fixture(self) -> None:
        fixture = self.directory / "phase_waiter.py"
        fixture_source = "import time\ntime.sleep(30)\n"
        fixture.write_text(fixture_source, encoding="utf-8")
        request = self.directory / "phase-request.json"
        request_source = '{"cell_id":"fixture","mode":"inert-test"}\n'
        request.write_text(request_source, encoding="utf-8")
        good_env = dict(os.environ)
        good_env["HAWKING_INVOCATION_MODE"] = "good"
        command = [
            sys.executable, str(fixture), "--cell", "fixture",
            "--request", str(request),
        ]
        sample = subprocess.Popen(command, cwd=elastic.ROOT, env=good_env)
        try:
            sample_identity = elastic.trusted_process_identity(sample.pid)
            sample_invocation = elastic.local_observer.observe_process_invocation(
                sample.pid
            )
            cell_position = sample_invocation["argv"].index("fixture")
            entry = elastic.build_invocation_entry_from_observation(
                phase="prepare", observation=sample_invocation,
                substitution_positions={cell_position: "cell_id"},
                allowed_substitutions={"cell_id": ["fixture"]},
            )
            self.contract = self._contract_with_extra_invocations([entry])
        finally:
            sample.terminate()
            sample.wait(timeout=5)

        resources = elastic.local_observer._direct_resources()
        resources.update({
            "pressure_level": 1, "swap_used_mb": 0.0,
            "thermal_green": True, "ac_power": True, "probe_valid": True,
        })
        resources["resource_sha256"] = elastic.local_observer._hash_value(
            elastic.local_observer._without(resources, "resource_sha256")
        )

        def attempt(argv: list[str], *, cwd: Path, env: dict[str, str],
                    expect_ok: bool) -> dict | None:
            process = subprocess.Popen(argv, cwd=cwd, env=env)
            try:
                identity = elastic.trusted_process_identity(process.pid)
                with tempfile.TemporaryDirectory(dir=elastic.STAGE_ROOT) as raw:
                    state_path = Path(raw) / "state.json"
                    swap_path = Path(raw) / "swap-state.json"
                    initial = elastic.new_state(self.contract)
                    elastic.persist_state(state_path, initial, self.contract)
                    swap_path.write_text(
                        json.dumps(self.swap_state, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    proof = self._start_proof(
                        "prepare", "fixture", threads=4, ram=4_000_000_000,
                        process_identity=identity,
                    )
                    # Caller declarations are deliberately wrong and ignored.
                    proof.update({"argv": ["forged"], "cwd": "/tmp",
                                  "environment": {"HAWKING_INVOCATION_MODE": "good"}})
                    with mock.patch.object(
                            elastic.local_observer, "HEAVY_COMMAND_PATTERNS", ()), \
                            mock.patch.object(
                                elastic.local_observer, "_direct_resources",
                                return_value=resources,
                            ):
                        if expect_ok:
                            _, receipt = elastic.compare_and_swap_transition(
                                state_path, self.contract,
                                expected_state_sha256=initial["state_sha256"],
                                expected_state_generation=0,
                                action="prepare_start", cell_id="fixture",
                                proof=proof, current_wall_epoch=1.0,
                                aggressive_swap_state_path=swap_path,
                            )
                            return receipt
                        with self.assertRaisesRegex(
                                elastic.ElasticError,
                                "does not match exactly one frozen phase entry"):
                            elastic.compare_and_swap_transition(
                                state_path, self.contract,
                                expected_state_sha256=initial["state_sha256"],
                                expected_state_generation=0,
                                action="prepare_start", cell_id="fixture",
                                proof=proof, current_wall_epoch=1.0,
                                aggressive_swap_state_path=swap_path,
                            )
                        return None
            finally:
                process.terminate()
                process.wait(timeout=5)

        attempt(["/bin/sleep", "30"], cwd=elastic.ROOT, env=good_env,
                expect_ok=False)
        attempt([
                    sys.executable, str(fixture), "--cell", "wrong",
                    "--request", str(request),
                ],
                cwd=elastic.ROOT, env=good_env, expect_ok=False)
        attempt(command, cwd=self.directory, env=good_env, expect_ok=False)
        bad_env = dict(good_env)
        bad_env["HAWKING_INVOCATION_MODE"] = "bad"
        attempt(command, cwd=elastic.ROOT, env=bad_env, expect_ok=False)
        pythonpath_env = dict(good_env)
        pythonpath_env["PYTHONPATH"] = "/tmp/unreviewed-code-injection"
        attempt(command, cwd=elastic.ROOT, env=pythonpath_env, expect_ok=False)
        dyld_env = dict(good_env)
        dyld_env["DYLD_LIBRARY_PATH"] = "/tmp/unreviewed-library-injection"
        attempt(command, cwd=elastic.ROOT, env=dyld_env, expect_ok=False)
        self.assertEqual("structural-only", entry["production_execution_protocol"])
        attempt(command, cwd=elastic.ROOT, env=good_env, expect_ok=False)
        request.write_text('{"cell_id":"fixture","mode":"mutated"}\n',
                           encoding="utf-8")
        self.assertTrue(any("invocation" in error
                            for error in elastic.validate_contract(self.contract)))
        request.write_text(request_source, encoding="utf-8")
        self.assertEqual([], elastic.validate_contract(self.contract))
        fixture.write_text("import time\ntime.sleep(29)\n", encoding="utf-8")
        self.assertTrue(any("invocation" in error
                            for error in elastic.validate_contract(self.contract)))
        fixture.write_text(fixture_source, encoding="utf-8")
        self.assertEqual([], elastic.validate_contract(self.contract))

    def test_invocation_manifest_rejects_unknown_fields_nan_and_path_tool_hijack(self) -> None:
        manifest = elastic.build_invocation_manifest(self.base_invocation_entries)
        unknown = copy.deepcopy(manifest)
        unknown["unreviewed"] = True
        unknown["manifest_sha256"] = elastic._hash_value(
            elastic._without(unknown, "manifest_sha256")
        )
        self.assertIn("phase invocation manifest keys are invalid",
                      elastic.validate_invocation_manifest(unknown))
        nan_manifest = copy.deepcopy(manifest)
        nan_manifest["unreviewed"] = float("nan")
        self.assertTrue(elastic.validate_invocation_manifest(nan_manifest))

        expected_tools = elastic.local_observer.authority_tool_references()
        fake_path = self.directory / "fake-bin"
        fake_path.mkdir()
        (fake_path / "ps").write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {"PATH": str(fake_path)}), \
                mock.patch.object(shutil, "which", side_effect=AssertionError(
                    "PATH tool resolution must not be consulted"
                )):
            identity = elastic.local_observer.observe_process_identity(os.getpid())
        self.assertEqual(os.getpid(), identity["pid"])
        self.assertEqual(expected_tools, self.contract["source_bindings"][
            "local_observer_authority_tools"
        ])

    def test_recovery_matches_process_identity_not_cell_label(self) -> None:
        state = elastic.transition(
            elastic.new_state(self.contract), self.contract,
            action="prepare_start", cell_id="prep",
            proof=self._start_proof(
                "prepare", "prep", threads=4, ram=4_000_000_000
            ), current_wall_epoch=70.0,
        )
        wrong = self._identity("prep", "different-command")
        receipt = elastic.crash_recovery_receipt(
            state, self.contract, observed_processes=[wrong],
            current_wall_epoch=100.0,
        )
        self.assertFalse(receipt["actions"][0]["observed_exact_process_identity"])
        receipt = elastic.crash_recovery_receipt(
            state, self.contract,
            observed_processes=[state["prepare_owner"]["process_identity"]],
            current_wall_epoch=100.0,
        )
        self.assertTrue(receipt["actions"][0]["observed_exact_process_identity"])

    def test_host_plan_is_default_off_and_spotlight_is_not_executable(self) -> None:
        self.assertEqual([], host.validate_plan(self.host_plan))
        spotlight = next(row for row in self.host_plan["proposals"]
                         if row["id"] == "workspace-spotlight-exclusion")
        self.assertFalse(spotlight["supported"])
        self.assertNotIsInstance(spotlight["proposal"], list)
        self.assertTrue(all(row["automatic"] is False
                            for row in self.host_plan["proposals"]))
        forged = copy.deepcopy(self.host_plan)
        forged["proposals"][0]["automatic"] = True
        forged["plan_sha256"] = host._hash_value(
            host._without(forged, "plan_sha256")
        )
        self.assertTrue(host.validate_plan(forged))
        red_probe = copy.deepcopy(self.host_probe)
        red_probe["thermal_green"] = False
        red_probe["probe_sha256"] = host._hash_value(
            host._without(red_probe, "probe_sha256")
        )
        probe_epoch = host._sampled_epoch(red_probe["sampled_at"])
        initial = aggressive.initial_swap_state(
            {"pressure_level": 1, "swap_used_mb": 0.0},
            now_epoch=probe_epoch - 2.0,
        )
        swap_state, raw = aggressive.advance_swap_state(
            initial, {"pressure_level": 1, "swap_used_mb": 0.0},
            now_epoch=probe_epoch - 1.0, sealed_baseline_swap_mb=0.0,
        )
        binding = host.bind_aggressive_swap_decision(swap_state, raw)
        owner_snapshot = host.build_owner_lease_snapshot(
            self.host_plan, red_probe, [], sampled_at=red_probe["sampled_at"]
        )
        gate = host.evaluate_gate(
            self.host_plan, red_probe, swap_state, binding, owner_snapshot,
            now_epoch=probe_epoch, sealed_baseline_swap_mb=0.0,
            expected_owner_leases=(),
        )
        self.assertFalse(gate["ok"])
        self.assertFalse(gate["fan_control_touched"])
        self.assertFalse(gate["os_services_mutated"])

    def test_capability_rows_match_full_stack_benchmark_contract(self) -> None:
        components = ["elastic-phase-admission", "host-sprint-isolation"]
        baseline = self._benchmark_run("baseline", [], 10.0)
        candidate = self._benchmark_run("candidate", components, 8.0)
        receipt = benchmark.build_receipt(
            scope=benchmark.SYNTHETIC_SCOPE, components=components,
            baseline_runs=[baseline], candidate_runs=[candidate],
            environment={"machine_identity_sha256": "a" * 64,
                         "same_machine_both_arms": True,
                         "randomized_interleaved_order": True},
            full_stack_end_to_end=False,
        )
        self.assertEqual([], benchmark.validate_receipt(
            receipt, require_production=False
        ))
        with self.assertRaises(benchmark.SprintBenchmarkError):
            benchmark.build_projection(
                production_eta={}, receipt=receipt, production_authority={}
            )

    def _identity(self, cell_id: str, role: str) -> dict:
        pid = 1_000 + sum(ord(char) for char in f"{cell_id}:{role}")
        return elastic.build_process_identity(
            pid=pid, start_identity=f"boot:test:{cell_id}:{role}",
            command_sha256=hashlib.sha256(
                f"cmd:{cell_id}:{role}".encode()
            ).hexdigest(),
        )

    def _build_inert_entry(self, *, phase: str, cell_id: str,
                           state_path: Path, contract_path: Path,
                           generation: int,
                           target_script: Path | None = None,
                           target_args: list[str] | None = None,
                           target_environment: dict[str, str] | None = None,
                           validator_script: Path | None = None) \
            -> tuple[dict, dict, Path]:
        target_script = target_script or self.fixture_target
        validator_script = validator_script or (
            CONDENSE / "doctor_v5_fixture_phase_validator.py"
        )
        request_path = self.directory / f"{cell_id}-{generation}-request.json"
        output_path = self.directory / f"{cell_id}-{generation}-output.json"
        semantic_path = self.directory / f"{cell_id}-{generation}-semantic.json"
        worker_path = self.directory / f"{cell_id}-{generation}-worker.json"
        handshake_path = self.directory / f"{cell_id}-{generation}-handshake.json"
        release_path = self.directory / f"{cell_id}-{generation}-release.json"
        claim_handshake_path = (
            self.directory / f"{cell_id}-{generation}-claim-handshake.json"
        )
        claim_ack_path = self.directory / f"{cell_id}-{generation}-claim-ack.json"
        claim_failure_path = (
            self.directory / f"{cell_id}-{generation}-claim-failure.json"
        )
        resource_guard_path = (
            self.directory / f"{cell_id}-{generation}-resource-guard.json"
        )
        target_env = dict(os.environ) if target_environment is None \
            else dict(target_environment)
        dependency_closure_path = (
            self.directory / f"{cell_id}-{generation}-target-closure.json"
        )
        dependency_closure_path.write_text(
            json.dumps(elastic.build_target_dependency_closure(
                target_script, interpreter=Path(self.python_executable)
            ), sort_keys=True) + "\n", encoding="utf-8",
        )
        claims = {
            "prepare": {
                "selected_threads": 4, "reservation_bytes": 4_000_000_000,
                "resource_spec_sha256": "c" * 64, "tier": None,
                "rate": None, "thread_selection_sha256": None,
                "lend_decision_sha256": None,
            },
            "finalizer": {
                "selected_threads": 4, "reservation_bytes": 4_000_000_000,
                "resource_spec_sha256": "c" * 64, "tier": None,
                "rate": None, "thread_selection_sha256": None,
                "lend_decision_sha256": None,
            },
            "encoder": {
                "selected_threads": 4, "reservation_bytes": None,
                "resource_spec_sha256": None, "tier": "14B", "rate": "q3",
                "thread_selection_sha256": "d" * 64,
                "lend_decision_sha256": None,
            },
            "companion": {
                "selected_threads": 4, "reservation_bytes": 4_000_000_000,
                "resource_spec_sha256": None, "tier": "14B", "rate": "q3",
                "thread_selection_sha256": "d" * 64,
                "lend_decision_sha256": "e" * 64,
            },
        }
        target_argv = [self.python_executable, str(target_script), *(target_args or [])]
        validator_argv = [
            self.python_executable, str(validator_script),
            "--request", str(request_path), "--output", str(output_path),
        ]
        request = elastic.build_inert_launch_request(
            phase=phase, cell_id=cell_id, request_path=request_path,
            contract_path=contract_path, state_path=state_path,
            expected_state_generation=generation,
            target_argv=target_argv, target_cwd=elastic.ROOT,
            target_environment=target_env,
            validator_argv=validator_argv, validator_cwd=elastic.ROOT,
            validator_environment={}, handshake_path=handshake_path,
            release_path=release_path,
            target_claim_handshake_path=claim_handshake_path,
            target_claim_ack_path=claim_ack_path,
            target_claim_failure_path=claim_failure_path,
            target_resource_guard_path=resource_guard_path,
            output_path=output_path,
            semantic_receipt_path=semantic_path,
            worker_receipt_path=worker_path, resource_claim=claims[phase],
            target_semantic_artifact_paths={
                "target.source": target_script,
                "target.dependency_closure": dependency_closure_path,
            },
            validator_semantic_artifact_paths={
                "validator.source": validator_script
            },
        )
        request_path.write_text(
            json.dumps(request, sort_keys=True) + "\n", encoding="utf-8"
        )
        executable = elastic.local_observer.stable_artifact_reference(
            Path(self.python_executable).resolve(strict=True)
        )
        argv = [
            executable["path"], str(elastic.INERT_LAUNCHER),
            "--request", str(request_path),
        ]
        entry = elastic.build_invocation_entry(
            phase=phase, executable=executable,
            argv_template=[{"literal": row} for row in argv],
            allowed_substitutions={}, cwd=str(elastic.ROOT.resolve(strict=True)),
            environment_allowlist={
                key: [hashlib.sha256(value.encode("utf-8")).hexdigest()]
                for key, value in sorted(target_env.items())
            },
            argv_artifacts=[
                {"position": 1, "argv_value": str(elastic.INERT_LAUNCHER),
                 "role": "script",
                 "reference": elastic._workspace_artifact_reference(
                     elastic.INERT_LAUNCHER
                 )},
                {"position": 3, "argv_value": str(request_path),
                 "role": "request",
                 "reference": elastic._workspace_artifact_reference(request_path)},
            ],
        )
        self.assertEqual(
            "inert-commit-v1-qualified",
            entry["production_execution_protocol"],
        )
        return entry, request, request_path

    def _spawn_inert_prepare_fixture(self, raw: str | Path, *, cell_id: str,
                                     swap_document: dict | str | None = None,
                                     target_script: Path | None = None,
                                     validator_script: Path | None = None) \
            -> tuple[Path, Path, dict, dict, dict, subprocess.Popen, dict]:
        directory = Path(raw)
        state_path = directory / "state.json"
        contract_path = directory / "contract.json"
        swap_path = directory / "swap-state.json"
        output_path = self.directory / f"{cell_id}-0-output.json"
        entry, request, _ = self._build_inert_entry(
            phase="prepare", cell_id=cell_id, state_path=state_path,
            contract_path=contract_path, generation=0,
            target_script=target_script,
            validator_script=validator_script,
            target_args=["--output", str(output_path), "--phase", "prepare",
                         "--cell", cell_id],
        )
        self.contract = self._contract_with_extra_invocations([entry])
        contract_path.write_text(
            json.dumps(self.contract, sort_keys=True) + "\n", encoding="utf-8"
        )
        initial = elastic.new_state(self.contract)
        elastic.persist_state(state_path, initial, self.contract)
        if isinstance(swap_document, str):
            swap_path.write_text(swap_document, encoding="utf-8")
        else:
            swap_path.write_text(
                json.dumps(
                    self.swap_state if swap_document is None else swap_document,
                    sort_keys=True,
                ) + "\n", encoding="utf-8",
            )
        command = [row["literal"] for row in entry["argv_template"]]
        process = subprocess.Popen(command, cwd=entry["cwd"], env=dict(os.environ))
        deadline = time.time() + 5.0
        while not Path(request["paths"]["handshake"]).is_file() \
                and time.time() < deadline:
            time.sleep(0.01)
        self.assertTrue(Path(request["paths"]["handshake"]).is_file())
        identity = elastic.trusted_process_identity(process.pid)
        return (state_path, swap_path, initial, entry, request, process, identity)

    @staticmethod
    def _green_direct_resources() -> dict:
        resources = elastic.local_observer._direct_resources()
        resources.update({
            "pressure_level": 1, "swap_used_mb": 0.0,
            "thermal_green": True, "ac_power": True, "probe_valid": True,
        })
        resources["resource_sha256"] = elastic.local_observer._hash_value(
            elastic.local_observer._without(resources, "resource_sha256")
        )
        return resources

    @staticmethod
    def _inject_external_heavy(receipt: dict) -> dict:
        value = copy.deepcopy(receipt)
        row = {
            "pid": 987_654, "ppid": 1,
            "start_identity": "ps-lstart:fixture",
            "command_sha256": "a" * 64,
            "matched_patterns": ["fixture-external-heavy"],
        }
        row["process_generation_sha256"] = elastic.local_observer._hash_value({
            key: row[key] for key in (
                "pid", "start_identity", "command_sha256"
            )
        })
        value["heavy_owners"] = [row]
        value["heavy_owner_count"] = 1
        value["observer_receipt_sha256"] = elastic.local_observer._hash_value(
            elastic.local_observer._without(
                value, "observer_receipt_sha256"
            )
        )
        return value

    def _admit_inert_prepare(self, state_path: Path, swap_path: Path,
                             initial: dict, *, cell_id: str,
                             identity: dict) -> tuple[dict, dict]:
        resources = self._green_direct_resources()
        with mock.patch.object(
                elastic.local_observer, "HEAVY_COMMAND_PATTERNS", ()), \
                mock.patch.object(
                    elastic.local_observer, "_direct_resources",
                    return_value=resources,
                ):
            return elastic.compare_and_swap_transition(
                state_path, self.contract,
                expected_state_sha256=initial["state_sha256"],
                expected_state_generation=initial["state_generation"],
                action="prepare_start", cell_id=cell_id,
                proof=self._start_proof(
                    "prepare", cell_id, threads=4, ram=4_000_000_000,
                    process_identity=identity,
                ), current_wall_epoch=1.0,
                aggressive_swap_state_path=swap_path,
            )

    def _contract_with_extra_invocations(
            self, entries: list[dict]) -> dict:
        manifest = elastic.build_invocation_manifest(
            [*self.base_invocation_entries, *entries]
        )
        digest = manifest["manifest_sha256"][:12]
        path = self.directory / f"phase-invocations-{digest}.json"
        path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        contract = elastic.build_contract(
            self.overlay, aggressive_overlay_path=self.overlay_path,
            host_probe=self.host_probe,
            host_plan_reference=elastic._file_reference(self.host_plan_path),
            invocation_manifest_path=path,
        )
        self.assertEqual("qualified", contract["status"], contract["blockers"])
        self.assertEqual([], elastic.validate_contract(contract))
        return contract

    def _start_proof(self, phase: str, cell_id: str, *, threads: int,
                     ram: int, process_identity: dict | None = None) -> dict:
        return {
            "resource_reservation": self._reservation(
                phase, cell_id, threads, ram
            ),
            "process_identity": process_identity or self._identity(cell_id, phase),
            "aggressive_swap_state": self.swap_state,
            "aggressive_swap_decision": self.swap_green,
        }

    def _completion_proof(self, owner: dict, phase: str, *,
                          preempted: bool,
                          active: list[dict] | None = None,
                          observed_epoch: float = 109.0) -> dict:
        observation = elastic.build_exit_observation(
            owner, self.contract, observed_epoch=observed_epoch,
            active_process_identities=active or [],
            preemption_verified=preempted,
        )
        receipt = elastic.build_phase_output_receipt(
            owner, self.contract, phase=phase, exit_observation=observation,
            output={"sha256": "a" * 64, "bytes": 1},
            receipt={"sha256": "b" * 64, "bytes": 1},
            checkpoint=phase == "companion_checkpoint",
        )
        return {"exit_observation": observation, "output_receipt": receipt}

    def _start_encoder(self, state: dict, cell_id: str, tier: str,
                       *, current_wall_epoch: float = 70.0) -> dict:
        key = json.dumps([tier, "q3"], separators=(",", ":"))
        selection = self.contract["thread_selections"][key]
        return elastic.transition(
            state, self.contract, action="encoder_start", cell_id=cell_id,
            proof={"tier": tier, "rate": "q3",
                   "selected_threads": selection["selected_threads"],
                   "thread_selection_sha256": selection["selection_sha256"],
                   "process_identity": self._identity(cell_id, "encoder"),
                   "aggressive_swap_state": self.swap_state,
                   "aggressive_swap_decision": self.swap_green},
            current_wall_epoch=current_wall_epoch,
        )

    def _lend(self, state: dict, samples: list[dict]) -> dict:
        return elastic.lend_decision(
            state, self.contract, candidate_cell_id="companion",
            tier="14B", rate="q3", reservation_bytes=8_000_000_000,
            idle_samples=samples, aggressive_swap_state=self.swap_state,
            aggressive_swap_decision=self.swap_green, now_epoch=100.0,
        )

    def _idle_samples(self, owner: dict) -> list[dict]:
        rows = []
        for sampled in (80.0, 88.0, 96.0):
            row = {
                "encoder_generation": owner["generation"],
                "primary_cell_id": owner["cell_id"],
                "primary_process_identity_sha256": owner[
                    "process_identity"]["process_identity_sha256"],
                "primary_lease_sha256": owner["lease"]["lease_sha256"],
                "state_generation": owner["lease"]["state_generation_at_acquire"],
                "host_probe_sha256": self.host_probe["probe_sha256"],
                "sampled_epoch": sampled, "total_active_cpu_cores": 16.0,
                "total_active_tree_rss_bytes": 30_000_000_000,
                "pressure_level": 1, "thermal_state": "nominal",
            }
            row["sample_sha256"] = elastic._hash_value(row)
            rows.append(row)
        return rows

    def _overlap_samples(self, owner: dict, state_generation: int) -> list[dict]:
        rows = []
        for sampled in (80.0, 88.0, 96.0):
            row = {
                "sampled_epoch": sampled, "total_active_cpu_cores": 8.0,
                "total_active_tree_rss_bytes": 15_000_000_000,
                "pressure_level": 1, "thermal_state": "nominal",
                "host_probe_sha256": self.host_probe["probe_sha256"],
                "state_generation": state_generation,
                "current_owner_cell_id": owner["cell_id"],
                "current_owner_role": owner["lease"]["role"],
                "current_owner_process_identity_sha256": owner[
                    "process_identity"]["process_identity_sha256"],
                "current_owner_lease_sha256": owner["lease"]["lease_sha256"],
            }
            row["sample_sha256"] = elastic._hash_value(row)
            rows.append(row)
        return rows

    def _overlap_envelope(self, current_owner: dict, new_reservation: dict,
                          new_process_identity: dict, samples: list[dict], *,
                          state_generation: int) -> dict:
        owners = sorted([
            {"cell_id": current_owner["cell_id"],
             "role": current_owner["lease"]["role"],
             "reservation_sha256": current_owner[
                 "resource_reservation"]["reservation_sha256"],
             "process_identity_sha256": current_owner[
                 "process_identity"]["process_identity_sha256"],
             "lease_sha256": current_owner["lease"]["lease_sha256"]},
            {"cell_id": new_reservation["cell_id"],
             "role": new_reservation["phase"],
             "reservation_sha256": new_reservation["reservation_sha256"],
             "process_identity_sha256": new_process_identity[
                 "process_identity_sha256"], "lease_sha256": None},
        ], key=lambda row: (row["cell_id"], row["role"]))
        envelope = {
            "schema": elastic.OVERLAP_SCHEMA, "version": elastic.VERSION,
            "contract_sha256": self.contract["contract_sha256"],
            "host_probe_sha256": self.host_probe["probe_sha256"],
            "state_generation": state_generation,
            "owners": owners, "samples": samples,
        }
        envelope["envelope_sha256"] = elastic._hash_value(envelope)
        return envelope

    def _reservation(self, phase: str, cell_id: str, threads: int,
                     ram: int) -> dict:
        row = {"contract_sha256": self.contract["contract_sha256"],
               "phase": phase, "cell_id": cell_id,
               "selected_threads": threads, "reservation_bytes": ram,
               "resource_spec_sha256": "c" * 64}
        row["reservation_sha256"] = elastic._hash_value(row)
        return row

    @staticmethod
    def _host_probe() -> dict:
        receipts = {
            field: {"argv": ["/usr/sbin/sysctl", "-n", key],
                    "returncode": 0, "output": str(host.REVIEWED_TOPOLOGY[field]),
                    "timed_out": False}
            for field, key in host.TOPOLOGY_SYSCTLS.items()
        }
        topology = {**host.REVIEWED_TOPOLOGY,
                    "reviewed_expected": host.REVIEWED_TOPOLOGY,
                    "verified_for_doctor_v5": True,
                    "source": "read-only-sysctl", "sysctl_receipts": receipts}
        topology["topology_sha256"] = host._hash_value(topology)
        probe = {
            "schema": host.PROBE_SCHEMA, "version": host.VERSION,
            "sampled_at": "2026-07-14T00:00:00+00:00", "topology": topology,
            "resource_snapshot": {"power_source": "AC Power",
                                  "pressure_level": 1,
                                  "swap_used_mb": 0.0},
            "thermal_probe": {"returncode": 0}, "thermal_green": True,
            "tools": {}, "spotlight_status_read_only": {"available": False},
            "backup_status_read_only": {"available": False},
            "fan_control_read_or_write_attempted": False,
            "os_service_mutation_attempted": False,
        }
        probe["probe_sha256"] = host._hash_value(probe)
        return probe

    @staticmethod
    def _cell(cell_id: str, tier: float, priority: int) -> dict:
        return {"cell_id": cell_id, "model_family": "qwen2.5-dense",
                "model_label": f"{tier:g}B",
                "nominal_params_b": tier, "branch": "codec_control",
                "priority": priority, "rate_id": "q3",
                "admission": {"whole_parent_residency_assumed": False},
                "parameter_manifest": {"source_weight_bytes": 10_000_000_000,
                                       "largest_source_shard_bytes": 1_000_000_000}}

    def _campaign_plan(self) -> dict:
        plan = {"schema": "hawking.doctor_v5_ultra_campaign_plan.v1",
                "cells": [self._cell("large", 32, 0),
                          self._cell("companion", 14, 1),
                          self._cell("exclusive", 72, 2)]}
        plan["plan_sha256"] = aggressive._hash_value(plan)
        return plan

    @staticmethod
    def _queue_state(plan: dict) -> dict:
        cells = {cell["cell_id"]: {"status": "pending",
                                   "request_sha256": f"{index:064x}"}
                 for index, cell in enumerate(plan["cells"], start=1)}
        state = {"schema": "hawking.doctor_v5_ultra_queue_state.v1",
                 "plan_sha256": plan["plan_sha256"], "status": "running",
                 "active_children": {}, "cells": cells}
        state["state_sha256"] = aggressive._hash_value(state)
        return state

    def _thread_profile(self, directory: Path) -> tuple[Path, Path]:
        contract = aggressive._load_thread_contract()
        binary = directory / "quantizer"
        binary.write_bytes(b"elastic-test-binary")
        binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
        winners = {
            "32B": {8: 10.0, 12: 8.0, 16: 5.0, 20: 6.0},
            "14B": {8: 4.0, 12: 5.0, 16: 6.0, 20: 7.0},
            "72B": {8: 10.0, 12: 8.0, 16: 6.0, 20: 4.0},
        }
        receipts = []
        for tier, timings in winners.items():
            for threads, wall in timings.items():
                receipt = {"schema": contract.RECEIPT_SCHEMA, "status": "pass",
                           "scope": "production", "synthetic": False,
                           "tier": tier, "rate": "q3", "threads": threads,
                           "binary_sha256": binary_sha, "source_sha256": "b" * 64,
                           "canonical_output_sha256": "c" * 64,
                           "output_sha256": "c" * 64, "exact_output": True,
                           "wall_seconds": wall, "peak_rss_bytes": 10_000 + threads,
                           "scratch_budget_bytes": 268_435_456,
                           "mode": "block_parallel"}
                path = directory / f"{tier}-{threads}.json"
                path.write_text(json.dumps(receipt, sort_keys=True) + "\n",
                                encoding="utf-8")
                receipts.append(path)
        profile = contract.build_profile(
            receipts, expected_binary_sha256=binary_sha, rss_limit_bytes=1_000_000
        )
        profile_path = directory / "thread-profile.json"
        profile_path.write_text(json.dumps(profile, sort_keys=True) + "\n",
                                encoding="utf-8")
        return profile_path, binary

    def _process_samples(self, cell_id: str) -> list[dict]:
        start = dt.datetime(2026, 7, 14, tzinfo=dt.timezone.utc)
        pgid = 10_000 + 10 * list(self.queue_state["cells"]).index(cell_id)
        rows = []
        for index in range(aggressive.MIN_AUTHENTICATED_SAMPLES):
            row = {"cell_id": cell_id, "plan_sha256": self.plan["plan_sha256"],
                   "request_sha256": self.queue_state["cells"][cell_id][
                       "request_sha256"],
                   "process_budget_bytes": aggressive.PROCESS_BUDGET_BYTES,
                   "sampled_at": (start + dt.timedelta(seconds=index * 8)).isoformat(),
                   "root_pid": pgid, "pgid": pgid, "process_count": 2,
                   "processes": [
                       {"pid": pgid, "ppid": 1, "pgid": pgid,
                        "rss_bytes": 1_000_000_000, "state": "S"},
                       {"pid": pgid + 1, "ppid": pgid, "pgid": pgid,
                        "rss_bytes": 9_000_000_000, "state": "R"},
                   ], "tree_rss_bytes": 10_000_000_000,
                   "max_tree_rss_bytes": 10_000_000_000,
                   "at_or_over_budget": False}
            rows.append(row)
        return rows

    @staticmethod
    def _benchmark_run(role: str, components: list[str], wall: float) -> dict:
        identity = {"sha256": "a" * 64, "bytes": 1}
        output = {"sha256": "b" * 64, "bytes": 1}
        receipt = {"sha256": "c" * 64, "bytes": 1}
        return {"role": role, "repeat_index": 0, "status": "complete",
                "exit_code": 0, "skipped": False,
                "source_files_deleted": False, "runtime_defaults_changed": False,
                "exercised_components": sorted(components), "program": identity,
                "benchmark_runner": identity,
                "input_bundle": identity, "output_bundle": output,
                "receipt_bundle": receipt, "invocation_sha256": "d" * 64,
                "semantic_contract_sha256": "e" * 64,
                "wall_seconds": wall, "cpu_seconds": wall * 10,
                "peak_rss_bytes": 1_000_000, "scratch_peak_bytes": 0,
                "disk_free_start_bytes": 1_000_000_000,
                "disk_free_end_bytes": 1_000_000_000, "gpu_seconds": 0.0,
                "swap_start_mb": 0.0, "swap_end_mb": 0.0,
                "memory_pressure_start": "normal", "memory_pressure_end": "normal",
                "thermal_start": "nominal", "thermal_end": "nominal"}


if __name__ == "__main__":
    unittest.main()
