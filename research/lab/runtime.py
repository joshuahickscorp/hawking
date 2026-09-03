from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
from lab.checkpoint import CheckpointStore
from lab.engine_support import ResourceGovernor, ResourceLimits
from lab.layout import REPORTS_ROOT, resolve_workspace_path
from lab.lease import LeaseError, SingletonLease
from lab.engine_support import Scheduler
from lab.engine_support import IllegalTransition, Phase, StateMachine
from lab.rules import GovernanceLedger, apply_governance
from lab.spec import CampaignPhase, ExperimentSpec, load_spec
from lab.science_registry import OperatorRegistry, load_default_registry
from lab.receipts import GateEvidence, Receipt, ReceiptAuthority, SealIntegrityError
Handler = Callable[['ExperimentRuntime', Mapping[str, Any]], dict[str, Any]]


def _existing_workspace_path(path: str | Path) -> Path:
    """Bridge a historic root-relative record to the live workspace if present."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    resolved = resolve_workspace_path(candidate)
    return resolved if resolved.exists() else candidate

@dataclass
class RunResult:
    campaign_id: str
    phase: str
    status: str
    completed_steps: list[str]
    receipt_path: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'campaign_id': self.campaign_id,
            'phase': self.phase,
            'status': self.status,
            'completed_steps': list(self.completed_steps),
            'receipt_path': self.receipt_path,
            'detail': dict(self.detail),
        }

def _handle_record(runtime: 'ExperimentRuntime', params: Mapping[str, Any]) -> dict[str, Any]:
    modules = params.get('modules') or params.get('module')
    if modules is None:
        return {'recorded': True, **dict(params)}
    names = [modules] if isinstance(modules, str) else [str(m) for m in modules]
    missing = [n for n in names if runtime.operators.get(n) is None]
    if missing and (not params.get('allow_missing')):
        raise RuntimeError(f'operators not in registry: {missing}')
    return {
        'recorded': True,
        'operators_present': [n for n in names if n not in missing],
        'operators_missing': missing,
        **{k: v for k, v in dict(params).items() if k not in {'modules', 'module'}},
    }

def _handle_precheck_fences(runtime: 'ExperimentRuntime', params: Mapping[str, Any]) -> dict[str, Any]:
    closed: list[str] = []
    overrides = params.get('allow') or {}
    for fence in runtime.spec.authorization_fences:
        if overrides.get(fence):
            raise RuntimeError(f'authorization fence forced open: {fence}')
        closed.append(fence)
    return {'fences_closed': closed}

def _handle_seal_receipt(runtime: 'ExperimentRuntime', params: Mapping[str, Any]) -> dict[str, Any]:
    step_id = runtime.active_step_id or 'seal'
    receipt = Receipt(
        campaign_id=f'{runtime.spec.campaign_id}:{step_id}',
        verdict=str(params.get('verdict') or 'SEALED'),
        status='sealed',
        phase=runtime.machine.phase.value,
        inputs={'campaign_id': runtime.spec.campaign_id},
        method={'handler': 'seal.receipt', 'step_id': step_id},
        measurement=dict(params.get('measurement') or {}),
        summary={k: v for k, v in dict(params).items() if k != 'measurement'},
        artifacts=tuple(str(p) for p in params.get('artifacts') or ()),
    )
    path = runtime.receipts.write(receipt)
    sealed = runtime.receipts.read(f'{runtime.spec.campaign_id}:{step_id}')
    return {'sealed': True, 'receipt_path': str(path), 'receipt_sha256': sealed['seal_sha256']}

def _handle_verify_gates(runtime: 'ExperimentRuntime', params: Mapping[str, Any]) -> dict[str, Any]:
    required = tuple(runtime.spec.verification_gates)
    supplied = params.get('gate_evidence') or params.get('evidence')
    # Historical records preserve their declared state, but are never a source
    # of a fresh live promotion.  Every live result must point at sealed input.
    if supplied is None:
        legacy = dict(params.get('gate_results') or {})
        if runtime.spec.status == 'released_historical_non_invocable' and legacy:
            if set(legacy) != set(required):
                raise RuntimeError('historical gate declaration must name exactly the required gates')
            runtime.gate_results.update({gate: bool(legacy[gate]) for gate in required})
            return {'gates': dict(runtime.gate_results), 'authority': 'HISTORICAL_DECLARATION_ONLY'}
        if required:
            raise RuntimeError('verify.gates requires sealed gate_evidence for every required gate')
        return {'gates': dict(runtime.gate_results)}
    if not isinstance(supplied, Mapping):
        raise RuntimeError('gate_evidence must be an object keyed by gate id')
    unknown = set(supplied) - set(required)
    missing = set(required) - set(supplied)
    if unknown or missing:
        raise RuntimeError(f'gate evidence must exactly match required gates; missing={sorted(missing)}, unknown={sorted(unknown)}')
    for gate in required:
        candidate = supplied[gate]
        if isinstance(candidate, Mapping):
            # A sealed inline record is persisted before it is read back; this
            # prevents an ephemeral object from becoming promotion authority.
            gate_path = runtime.receipts.write_gate_evidence(candidate)
        elif isinstance(candidate, str):
            gate_path = _existing_workspace_path(candidate)
            if not gate_path.is_absolute():
                gate_path = runtime.work_dir / gate_path
        else:
            raise RuntimeError(f'gate evidence for {gate!r} must be a sealed object or path')
        evidence = runtime.receipts.read_gate_evidence(gate_path, expected_gate=gate)
        runtime.gate_evidence[gate] = evidence
        runtime.gate_results[gate] = evidence.result == 'PASS'
        if evidence.result != 'PASS':
            raise RuntimeError(f'gate {gate!r} is sealed as {evidence.result}; promotion is refused')
    return {
        'gates': dict(runtime.gate_results),
        'gate_evidence': {gate: str(runtime.receipts.gate_path_for(gate)) for gate in required},
    }

def _handle_promote(runtime: 'ExperimentRuntime', params: Mapping[str, Any]) -> dict[str, Any]:
    return apply_governance(
        runtime.spec,
        ledger=runtime.ledger,
        verdict=str(params.get('verdict') or 'PASS'),
        gate_results=runtime.gate_results,
        gate_evidence=runtime.gate_evidence,
        author=str(params.get('author') or ''),
        admitter=str(params.get('admitter') or 'engine'),
        measurement_kind=str(params.get('measurement_kind') or 'real'),
        action='promote',
    )

def _handle_bury(runtime: 'ExperimentRuntime', params: Mapping[str, Any]) -> dict[str, Any]:
    arts = [_existing_workspace_path(p) for p in params.get('artifacts') or []]
    recs = [_existing_workspace_path(p) for p in params.get('receipts') or []]
    for p in arts + recs:
        if not p.exists():
            raise RuntimeError(f'burial retention path is missing: {p}')
    for p in recs:
        try:
            # Receipt.from_dict verifies the common seal, so this accepts both
            # normal experiment receipts and GateEvidence envelopes while
            # rejecting an arbitrary JSON file that merely looks like one.
            Receipt.from_dict(json.loads(p.read_text(encoding='utf-8')))
        except (OSError, ValueError, SealIntegrityError) as exc:
            raise RuntimeError(f'burial receipt is not sealed: {p}: {exc}') from exc
    return apply_governance(
        runtime.spec,
        ledger=runtime.ledger,
        verdict=str(params.get('verdict') or 'BURIED'),
        gate_results=runtime.gate_results,
        action='bury',
        artifacts=arts,
        receipts=recs,
    )
BUILTIN_HANDLERS: dict[str, Handler] = {
    'record': _handle_record,
    'precheck.fences': _handle_precheck_fences,
    'precheck.contract': _handle_record,
    'measure.parity': _handle_record,
    'report.summary': _handle_record,
    'seal.receipt': _handle_seal_receipt,
    'verify.gates': _handle_verify_gates,
    'promote': _handle_promote,
    'bury': _handle_bury,
}

class ExperimentRuntime:

    def __init__(
        self,
        spec: ExperimentSpec,
        *,
        work_dir: Path,
        handlers: Mapping[str, Handler] | None=None,
        operators: OperatorRegistry | None=None,
        controller_epoch: str='1',
        acquire_lease: bool=True,
    ) -> None:
        self.spec = spec
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.operators = operators or load_default_registry()
        self.handlers: dict[str, Handler] = {**BUILTIN_HANDLERS, **dict(handlers or {})}
        for key, handler in self.operators.handlers.items():
            self.handlers.setdefault(key, handler)
        if not handlers:
            from lab.science_registry import build_operator_handlers
            for key, handler in build_operator_handlers().items():
                self.handlers.setdefault(key, handler)
        self.controller_epoch = controller_epoch
        self.acquire_lease = acquire_lease
        limits = ResourceLimits.from_mapping(spec.metadata.get('resource_limits'))
        self.governor = ResourceGovernor(limits, root=self.work_dir)
        self.checkpoints = CheckpointStore(
            self.work_dir,
            campaign_id=spec.campaign_id,
            checkpoint_name=spec.checkpoint_name or 'checkpoint.json',
        )
        self.receipts = ReceiptAuthority(self.work_dir / 'receipts')
        self.ledger = GovernanceLedger(self.work_dir / 'governance.jsonl')
        self.lease = SingletonLease(
            self.work_dir / (spec.lease_name or f'{spec.campaign_id}.lease'),
            campaign_id=spec.campaign_id,
            controller_epoch=controller_epoch,
            owner=f'lab:{spec.campaign_id}',
        )
        self.machine = StateMachine.for_spec(spec)
        self.scheduler = Scheduler(spec)
        self.gate_results: dict[str, bool] = {}
        self.gate_evidence: dict[str, GateEvidence] = {}
        self.active_step_id: str | None = None
        self._lease_held = False

    def _sync_scheduler(self) -> None:
        self.scheduler = Scheduler(self.spec, completed=set(self.machine.completed_steps))

    def open(self) -> None:
        if self.acquire_lease:
            self.lease.acquire()
            self._lease_held = True
        resumed = self.checkpoints.resume_state()
        if resumed.get('phase') and resumed.get('phase') != 'idle':
            self.machine = StateMachine.from_snapshot(resumed)
            if self.machine.phase not in {Phase.COMPLETE, Phase.IDLE}:
                if self.machine.phase is not Phase.RESUME:
                    try:
                        self.machine.transition(
                            Phase.RESUME,
                            claim_id=(
                                f'resume:{self.machine.phase.value}:'
                                f'{len(self.machine.claims)}'
                            ),
                            detail={'from_checkpoint': True},
                        )
                    except IllegalTransition:
                        pass
        self._sync_scheduler()
        self.checkpoints.record('open', self.machine.snapshot())

    def close(self) -> None:
        try:
            self.checkpoints.save(self.machine.snapshot())
        finally:
            if self._lease_held:
                self.lease.release()
                self._lease_held = False

    def __enter__(self) -> 'ExperimentRuntime':
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def run_handler(self, step_id: str) -> dict[str, Any]:
        step = next((s for s in self.spec.steps if s.id == step_id), None)
        if step is None:
            raise KeyError(f'unknown step {step_id!r}')
        if self.machine.is_step_done(step_id) and step.idempotent:
            return {'skipped': True, 'reason': 'already_done'}
        name = step.handler or 'record'
        handler = self.handlers.get(name)
        if handler is None:
            if step.optional:
                self.scheduler.mark_skipped(step_id, f'no handler {name}')
                self.machine.mark_step(step_id)
                return {'skipped': True, 'reason': f'no handler {name}'}
            raise RuntimeError(f'no handler registered for {name!r} (step {step_id})')
        if self.acquire_lease:
            self.lease.assert_held()
        light = {
            'record',
            'precheck.fences',
            'precheck.contract',
            'measure.parity',
            'report.summary',
            'seal.receipt',
            'verify.gates',
            'promote',
            'bury',
        }
        if name not in light:
            self.governor.require()
        self.active_step_id = step_id
        try:
            result = handler(self, step.params)
        finally:
            self.active_step_id = None
        self.machine.mark_step(step_id)
        self.scheduler.mark_done(step_id, detail=result)
        self.checkpoints.record('step_done', {'step_id': step_id, 'handler': name, 'result': result})
        self.checkpoints.save(self.machine.snapshot())
        return result

    def advance_phase(self, target: str | Phase | CampaignPhase, *, claim_id: str) -> None:
        if self.acquire_lease:
            self.lease.assert_held()
        self.machine.transition(target, claim_id=claim_id)
        self.checkpoints.record('phase_transition', {'to': self.machine.phase.value, 'claim_id': claim_id})
        self.checkpoints.save(self.machine.snapshot())

    def run(self, *, phases: list[str] | None=None, stop_on_fail: bool=True) -> RunResult:
        wanted = set(phases or self.spec.phases)
        if self.machine.phase in {Phase.IDLE, Phase.RESUME}:
            first = next((p for p in self.spec.phases if p in wanted), None)
            if first is not None:
                try:
                    self.advance_phase(first, claim_id=f'enter:{first}')
                except IllegalTransition:
                    pass
        errors: list[str] = []
        for phase in self.spec.phases:
            if phase not in wanted:
                continue
            if self.machine.phase.value != phase:
                try:
                    self.advance_phase(phase, claim_id=f'phase:{phase}:{len(self.machine.claims)}')
                except IllegalTransition:
                    if self.machine.phase is Phase.COMPLETE:
                        break
            for step in self.spec.steps_for(phase):
                if self.machine.is_step_done(step.id) and step.idempotent:
                    continue
                try:
                    self.run_handler(step.id)
                except Exception as exc:
                    errors.append(f'{step.id}: {exc}')
                    self.scheduler.mark_failed(step.id, str(exc))
                    try:
                        self.advance_phase(Phase.FAULT, claim_id=f'fault:{step.id}:{len(self.machine.claims)}')
                    except IllegalTransition:
                        self.machine.fault_reason = str(exc)
                    if stop_on_fail:
                        return RunResult(
                            campaign_id=self.spec.campaign_id,
                            phase=self.machine.phase.value,
                            status='FAULT',
                            completed_steps=list(self.machine.completed_steps),
                            detail={'errors': errors},
                        )
        all_done = all((self.machine.is_step_done(s.id) or s.optional for s in self.spec.steps))
        receipt_path = None
        if all_done and (not errors):
            if self.machine.phase is not Phase.COMPLETE:
                try:
                    if self.machine.phase is not Phase.REPORT:
                        if 'report' in self.spec.phases and self.machine.can(Phase.REPORT):
                            self.advance_phase(Phase.REPORT, claim_id=f'report:{len(self.machine.claims)}')
                    if self.machine.can(Phase.COMPLETE):
                        self.advance_phase(Phase.COMPLETE, claim_id=f'complete:{len(self.machine.claims)}')
                except IllegalTransition:
                    pass
            receipt = Receipt(
                campaign_id=self.spec.campaign_id,
                verdict='PASS' if self.spec.status != 'sealed_negative' else 'SEALED_NEGATIVE',
                status=self.spec.status,
                phase=self.machine.phase.value,
                method={'family': self.spec.family, 'title': self.spec.title, 'reproduction': self.spec.reproduction},
                measurement={'completed_steps': list(self.machine.completed_steps), 'gates': dict(self.gate_results)},
                inputs={
                    'preregistration': dict(self.spec.preregistration),
                    'source_admission': dict(self.spec.source_admission),
                },
                summary={
                    'completed_steps': list(self.machine.completed_steps),
                    'title': self.spec.title,
                    'family': self.spec.family,
                },
                reproduction=self.spec.reproduction,
                artifacts=tuple((a for a in (self.spec.receipt, self.spec.fixture) if a)),
            )
            path = self.receipts.write(receipt)
            receipt_path = str(path)
            self.checkpoints.record('receipt_written', {'path': receipt_path})
            self.checkpoints.save(self.machine.snapshot())
            self.ledger.append('receipt', {'campaign_id': self.spec.campaign_id, 'path': receipt_path})
        status = 'PASS' if not errors and all_done else 'FAULT' if errors else 'PARTIAL'
        return RunResult(
            campaign_id=self.spec.campaign_id,
            phase=self.machine.phase.value,
            status=status,
            completed_steps=list(self.machine.completed_steps),
            receipt_path=receipt_path,
            detail={'errors': errors} if errors else {},
        )

    def status(self) -> dict[str, Any]:
        return {
            'campaign_id': self.spec.campaign_id,
            'title': self.spec.title,
            'family': self.spec.family,
            'spec_status': self.spec.status,
            'phase': self.machine.phase.value,
            'completed_steps': list(self.machine.completed_steps),
            'fault_reason': self.machine.fault_reason,
            'scheduler': self.scheduler.snapshot(),
            'lease_held': self.lease.held,
            'reproduction': self.spec.reproduction,
            'receipt': self.spec.receipt,
            'reopen': [r.to_dict() for r in self.spec.reopen],
            'gates': dict(self.gate_results),
            'gate_evidence': {gate: evidence.to_dict() for gate, evidence in self.gate_evidence.items()},
        }

def run_experiment(
    spec: ExperimentSpec | Mapping[str, Any] | str | Path,
    *,
    work_dir: Path,
    handlers: Mapping[str, Handler] | None=None,
    phases: list[str] | None=None,
    acquire_lease: bool=True,
) -> RunResult:
    if not isinstance(spec, ExperimentSpec):
        spec = load_spec(spec)
    with ExperimentRuntime(spec, work_dir=work_dir, handlers=handlers, acquire_lease=acquire_lease) as runtime:
        return runtime.run(phases=phases)
run_campaign = run_experiment
CampaignRuntime = ExperimentRuntime

def main(argv: list[str] | None=None) -> int:
    import argparse
    from lab.science_registry import build_operator_handlers
    parser = argparse.ArgumentParser(description='lab — experiment engine + science operators')
    parser.add_argument('spec', type=Path, nargs='?', default=None)
    parser.add_argument('--work-dir', type=Path, default=None)
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--no-lease', action='store_true')
    parser.add_argument('--classify', action='store_true')
    parser.add_argument('--list-ops', action='store_true', help='List operator handler keys')
    parser.add_argument('--dry-run', action='store_true', help='Dry-run campaign (no lease)')
    parser.add_argument(
        '--read-historical',
        type=Path,
        default=None,
        help='Normalize a historical receipt/ledger path and print JSON',
    )
    parser.add_argument('op_args', nargs='*', default=[], help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    # `python3.12 -m lab op <handler>` / `list-ops`
    if args.spec is not None and str(args.spec) == 'list-ops':
        args.list_ops = True
        args.spec = None
    if args.spec is not None and str(args.spec) == 'op':
        if not args.op_args:
            parser.error('lab op requires a handler key')
        key = args.op_args[0]
        registry = load_default_registry(handlers=build_operator_handlers())
        handler = registry.resolve_handler(key)
        if handler is None:
            # fail closed
            print(json.dumps({'ok': False, 'error': f'missing handler: {key}'}))
            return 2
        result = handler(params={})
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result.get('ok', True) is not False else 1
    if args.list_ops:
        registry = load_default_registry(handlers=build_operator_handlers())
        keys = sorted({k for r in registry.records for k in r.handler_keys} | set(registry.handlers))
        print(json.dumps({'handlers': keys, 'count': len(keys)}, indent=2))
        return 0
    if args.read_historical is not None:
        from lab.receipts import read_any_receipt, read_jsonl_ledger
        path = args.read_historical
        if path.suffix == '.jsonl':
            payload = list(read_jsonl_ledger(path))
        else:
            payload = read_any_receipt(path)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0
    if args.classify:
        registry = load_default_registry(handlers=build_operator_handlers())
        print(
            json.dumps(
                {
                    'summary': registry.summary(),
                    'modules': registry.classification(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.spec is None:
        parser.error('spec path is required unless --classify / --list-ops / op / --read-historical')
    spec = load_spec(args.spec)
    work_dir = args.work_dir or REPORTS_ROOT / 'lab' / spec.campaign_id
    handlers = build_operator_handlers()
    with ExperimentRuntime(
        spec,
        work_dir=work_dir,
        handlers=handlers,
        acquire_lease=not args.no_lease and not args.dry_run,
    ) as runtime:
        if args.dry_run:
            payload = {'status': 'DRY_RUN', 'campaign_id': spec.campaign_id, 'steps': [s.id for s in spec.steps]}
        else:
            payload = runtime.status() if args.status else runtime.run().to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if isinstance(payload, dict) and payload.get('status') == 'FAULT':
        return 1
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
