from __future__ import annotations
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
from lab.semantic_taxonomy import (
    CONDENSE_OPERATION,
    SemanticTaxonomyError,
    normalize_semantic_tags,
)
SCHEMA = 'hawking.lab.experiment_spec.v1'
LAB_DIR = Path(__file__).resolve().parent
CATALOG_PATH = LAB_DIR / 'campaigns.json'
SPECS_DIR = LAB_DIR

class SpecError(ValueError):
    pass

class CampaignPhase(str, Enum):
    PRECHECK = 'precheck'
    MEASURE = 'measure'
    ALLOCATE = 'allocate'
    PACK = 'pack'
    VERIFY = 'verify'
    SEAL = 'seal'
    PROMOTE = 'promote'
    BURY = 'bury'
    MONITOR = 'monitor'
    RESUME = 'resume'
    REPORT = 'report'

class _ResourceClass(str, Enum):
    LIGHT_READONLY = 'light-readonly'
    LIGHT = 'light'
    HEAVY = 'heavy'
    GPU_LAB = 'gpu-lab'
    NETWORK = 'network'
    DETACHED = 'detached'
DEFAULT_PHASE_ORDER: tuple[str, ...] = (
    CampaignPhase.PRECHECK.value,
    CampaignPhase.MEASURE.value,
    CampaignPhase.ALLOCATE.value,
    CampaignPhase.PACK.value,
    CampaignPhase.VERIFY.value,
    CampaignPhase.SEAL.value,
    CampaignPhase.PROMOTE.value,
    CampaignPhase.REPORT.value,
)

@dataclass(frozen=True)
class StepSpec:
    id: str
    phase: str
    description: str = ''
    handler: str = ''
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    idempotent: bool = True
    optional: bool = False
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'phase': self.phase,
            'description': self.description,
            'handler': self.handler,
            'inputs': list(self.inputs),
            'outputs': list(self.outputs),
            'idempotent': self.idempotent,
            'optional': self.optional,
            'params': dict(self.params),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> 'StepSpec':
        if not isinstance(raw, Mapping):
            raise SpecError('step must be an object')
        sid = raw.get('id')
        phase = raw.get('phase')
        if not isinstance(sid, str) or not sid:
            raise SpecError('step.id must be a non-empty string')
        if not isinstance(phase, str) or not phase:
            raise SpecError(f'step {sid!r}: phase must be non-empty')
        try:
            CampaignPhase(phase)
        except ValueError as exc:
            raise SpecError(
                f'step {sid!r}: unknown phase {phase!r}; '
                f'allowed={[p.value for p in CampaignPhase]}'
            ) from exc
        params = raw.get('params') or {}
        if not isinstance(params, Mapping):
            raise SpecError(f'step {sid!r}: params must be an object')
        return cls(
            id=sid,
            phase=phase,
            description=str(raw.get('description') or ''),
            handler=str(raw.get('handler') or ''),
            inputs=tuple((str(x) for x in raw.get('inputs') or ())),
            outputs=tuple((str(x) for x in raw.get('outputs') or ())),
            idempotent=bool(raw.get('idempotent', True)),
            optional=bool(raw.get('optional', False)),
            params=dict(params),
        )

@dataclass(frozen=True)
class ReopenCondition:
    id: str
    description: str
    predicate: str = ''
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'description': self.description,
            'predicate': self.predicate,
            'evidence': list(self.evidence),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> 'ReopenCondition':
        if not isinstance(raw, Mapping):
            raise SpecError('reopen_condition must be an object')
        rid = raw.get('id')
        if not isinstance(rid, str) or not rid:
            raise SpecError('reopen_condition.id must be a non-empty string')
        return cls(
            id=rid,
            description=str(raw.get('description') or ''),
            predicate=str(raw.get('predicate') or ''),
            evidence=tuple((str(x) for x in raw.get('evidence') or ())),
        )

@dataclass(frozen=True)
class PromotionRule:
    require_verdict: str = 'PASS'
    require_gates: tuple[str, ...] = ()
    author_may_admit: bool = False
    allow_synthetic_as_real: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            'require_verdict': self.require_verdict,
            'require_gates': list(self.require_gates),
            'author_may_admit': self.author_may_admit,
            'allow_synthetic_as_real': self.allow_synthetic_as_real,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> 'PromotionRule':
        raw = raw or {}
        if not isinstance(raw, Mapping):
            raise SpecError('promotion_rule must be an object')
        return cls(
            require_verdict=str(raw.get('require_verdict') or 'PASS'),
            require_gates=tuple((str(x) for x in raw.get('require_gates') or ())),
            author_may_admit=bool(raw.get('author_may_admit', False)),
            allow_synthetic_as_real=bool(raw.get('allow_synthetic_as_real', False)),
        )

@dataclass(frozen=True)
class BurialRule:
    retain_artifacts: bool = True
    retain_receipts: bool = True
    status_value: str = 'buried'

    def to_dict(self) -> dict[str, Any]:
        return {
            'retain_artifacts': self.retain_artifacts,
            'retain_receipts': self.retain_receipts,
            'status_value': self.status_value,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> 'BurialRule':
        raw = raw or {}
        if not isinstance(raw, Mapping):
            raise SpecError('burial_rule must be an object')
        return cls(
            retain_artifacts=bool(raw.get('retain_artifacts', True)),
            retain_receipts=bool(raw.get('retain_receipts', True)),
            status_value=str(raw.get('status_value') or 'buried'),
        )

@dataclass(frozen=True)
class ExperimentSpec:
    schema: str
    campaign_id: str
    title: str
    family: str
    status: str
    resource_class: str
    phases: tuple[str, ...]
    steps: tuple[StepSpec, ...]
    receipt: str = ''
    fixture: str = ''
    reproduction: str = ''
    reopen: tuple[ReopenCondition, ...] = ()
    lease_name: str = ''
    checkpoint_name: str = ''
    authorization_fences: tuple[str, ...] = ()
    preregistration: Mapping[str, Any] = field(default_factory=dict)
    source_admission: Mapping[str, Any] = field(default_factory=dict)
    measurement_plan: Mapping[str, Any] = field(default_factory=dict)
    allocation_policy: Mapping[str, Any] = field(default_factory=dict)
    pack_program: Mapping[str, Any] = field(default_factory=dict)
    verification_gates: tuple[str, ...] = ()
    promotion: PromotionRule = field(default_factory=PromotionRule)
    burial: BurialRule = field(default_factory=BurialRule)
    resume_policy: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ''
    metadata: Mapping[str, Any] = field(default_factory=dict)
    semantic_tags: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'schema': self.schema,
            'campaign_id': self.campaign_id,
            'title': self.title,
            'family': self.family,
            'status': self.status,
            'resource_class': self.resource_class,
            'phases': list(self.phases),
            'steps': [s.to_dict() for s in self.steps],
            'receipt': self.receipt,
            'fixture': self.fixture,
            'reproduction': self.reproduction,
            'reopen': [r.to_dict() for r in self.reopen],
            'lease_name': self.lease_name,
            'checkpoint_name': self.checkpoint_name,
            'authorization_fences': list(self.authorization_fences),
            'preregistration': dict(self.preregistration),
            'source_admission': dict(self.source_admission),
            'measurement_plan': dict(self.measurement_plan),
            'allocation_policy': dict(self.allocation_policy),
            'pack_program': dict(self.pack_program),
            'verification_gates': list(self.verification_gates),
            'promotion': self.promotion.to_dict(),
            'burial': self.burial.to_dict(),
            'resume_policy': dict(self.resume_policy),
            'notes': self.notes,
            'metadata': dict(self.metadata),
            'semantic_tags': dict(self.semantic_tags),
        }

    def steps_for(self, phase: str | CampaignPhase) -> list[StepSpec]:
        name = phase.value if isinstance(phase, CampaignPhase) else phase
        return [s for s in self.steps if s.phase == name]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> 'ExperimentSpec':
        return validate_spec(raw)
# 'released_historical_non_invocable' was written into lab/campaigns.json by
# 8b0c5405 and never added here, so three campaign specs have failed to load
# since 2026-07-29. The test that names them could not collect -- the same
# commit family broke its imports -- so nothing reported it.
_ALLOWED_STATUS = frozenset({'live', 'retired', 'sealed_negative', 'fixture_only', 'historical', 'buried', 'released_historical_non_invocable'})
_ACCEPTED_SCHEMAS = frozenset({SCHEMA, 'hawking.condense.experiment_spec.v1', 'hawking.lab.experiment.v1'})

def validate_spec(raw: Mapping[str, Any]) -> ExperimentSpec:
    if not isinstance(raw, Mapping):
        raise SpecError('spec root must be an object')
    schema = raw.get('schema')
    if not isinstance(schema, str) or schema not in _ACCEPTED_SCHEMAS:
        raise SpecError(f'unsupported schema {schema!r}; expected one of {sorted(_ACCEPTED_SCHEMAS)}')
    declared_identity = raw.get('semantic_identity')
    if declared_identity is not None and declared_identity != 'gravity':
        raise SpecError(
            "semantic_identity must be 'gravity'; Condense is an operation tag"
        )
    campaign_id = raw.get('campaign_id') or raw.get('id')
    if not isinstance(campaign_id, str) or not campaign_id:
        raise SpecError('campaign_id must be a non-empty string')
    title = str(raw.get('title') or campaign_id)
    family = str(raw.get('family') or 'unknown')
    status = str(raw.get('status') or 'historical')
    if status not in _ALLOWED_STATUS:
        raise SpecError(f'status {status!r} not in {sorted(_ALLOWED_STATUS)}')
    resource_class = str(raw.get('resource_class') or _ResourceClass.LIGHT.value)
    try:
        _ResourceClass(resource_class)
    except ValueError as exc:
        raise SpecError(f'unknown resource_class {resource_class!r}') from exc
    phases_raw = raw.get('phases')
    if phases_raw is None and raw.get('stages'):
        phases_raw = list(DEFAULT_PHASE_ORDER[:4]) + [CampaignPhase.REPORT.value]
    if not isinstance(phases_raw, Sequence) or isinstance(phases_raw, (str, bytes)):
        raise SpecError('phases must be a list of phase names')
    phases: list[str] = []
    for item in phases_raw:
        if not isinstance(item, str):
            raise SpecError('each phase must be a string')
        try:
            CampaignPhase(item)
        except ValueError as exc:
            raise SpecError(f'unknown phase {item!r}') from exc
        phases.append(item)
    if not phases:
        raise SpecError('phases must be non-empty')
    steps_raw = raw.get('steps')
    if steps_raw is None and raw.get('stages'):
        steps_raw = []
        for st in raw['stages']:
            if not isinstance(st, Mapping):
                raise SpecError('stage must be an object')
            steps_raw.append({
                'id': st.get('id'),
                'phase': CampaignPhase.MEASURE.value,
                'handler': 'record',
                'description': str(st.get('id') or ''),
                'params': {
                    'argv': st.get('argv') or [],
                    'shell': st.get('shell'),
                },
                'optional': bool(st.get('on_fail') == 'continue'),
            })
    steps_raw = steps_raw or []
    if not isinstance(steps_raw, Sequence) or isinstance(steps_raw, (str, bytes)):
        raise SpecError('steps must be a list')
    steps = tuple((StepSpec.from_dict(s) for s in steps_raw))
    seen: set[str] = set()
    for step in steps:
        if step.id in seen:
            raise SpecError(f'duplicate step id {step.id!r}')
        seen.add(step.id)
        if step.phase not in phases and step.phase != CampaignPhase.RESUME.value:
            raise SpecError(f'step {step.id!r} phase {step.phase!r} not listed in phases')
    reopen_raw = raw.get('reopen') or []
    if not isinstance(reopen_raw, Sequence) or isinstance(reopen_raw, (str, bytes)):
        raise SpecError('reopen must be a list')
    reopen = tuple((ReopenCondition.from_dict(r) for r in reopen_raw))
    fences = tuple((str(x) for x in raw.get('authorization_fences') or ()))
    gates = tuple((str(x) for x in raw.get('verification_gates') or ()))
    metadata = raw.get('metadata') or raw.get('meta') or {}
    if not isinstance(metadata, Mapping):
        raise SpecError('metadata must be an object')
    # These are plan-declared references only.  The semantic normalizer never
    # resolves them, so adding the tag cannot turn a plan into an artifact
    # availability claim.
    declared_artifact_references = tuple(
        str(value)
        for value in (raw.get('receipt'), raw.get('fixture'))
        if value
    )
    try:
        semantic_tags = normalize_semantic_tags(
            raw.get('semantic_tags'),
            operation=CONDENSE_OPERATION,
            artifact_kind='experiment_spec',
            raw_schema=schema,
            artifact_references=declared_artifact_references,
        )
    except SemanticTaxonomyError as exc:
        raise SpecError(str(exc)) from exc

    def _obj(key: str) -> dict[str, Any]:
        val = raw.get(key) or {}
        if not isinstance(val, Mapping):
            raise SpecError(f'{key} must be an object')
        return dict(val)
    return ExperimentSpec(
        schema=SCHEMA,
        campaign_id=campaign_id,
        title=title,
        family=family,
        status=status,
        resource_class=resource_class,
        phases=tuple(phases),
        steps=steps,
        receipt=str(raw.get('receipt') or ''),
        fixture=str(raw.get('fixture') or ''),
        reproduction=str(raw.get('reproduction') or ''),
        reopen=reopen,
        lease_name=str(raw.get('lease_name') or f'{campaign_id}.lease'),
        checkpoint_name=str(raw.get('checkpoint_name') or f'{campaign_id}.checkpoint.json'),
        authorization_fences=fences,
        preregistration=_obj('preregistration'),
        source_admission=_obj('source_admission'),
        measurement_plan=_obj('measurement_plan'),
        allocation_policy=_obj('allocation_policy'),
        pack_program=_obj('pack_program'),
        verification_gates=gates,
        promotion=PromotionRule.from_dict(raw.get('promotion')),
        burial=BurialRule.from_dict(raw.get('burial')),
        resume_policy=_obj('resume_policy'),
        notes=str(raw.get('notes') or ''),
        metadata=dict(metadata),
        semantic_tags=semantic_tags,
    )

def _load_catalog() -> dict[str, Any]:
    try:
        raw = json.loads(CATALOG_PATH.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecError(f'cannot read catalog {CATALOG_PATH}: {exc}') from exc
    if not isinstance(raw, dict):
        raise SpecError(f'catalog root must be an object: {CATALOG_PATH}')
    if 'campaigns' in raw and isinstance(raw['campaigns'], dict):
        return raw['campaigns']
    return raw

def load_spec(raw: Mapping[str, Any] | str | Path) -> ExperimentSpec:
    if isinstance(raw, (str, Path)):
        return load_spec_path(Path(raw))
    return validate_spec(raw)

def load_spec_path(path: Path) -> ExperimentSpec:
    path = Path(path)
    catalog = _load_catalog()
    stem = path.stem
    if stem in catalog:
        data = catalog[stem]
        if not isinstance(data, Mapping):
            raise SpecError(f'catalog entry {stem!r} must be an object')
        return validate_spec(data)
    if path.name in catalog:
        return validate_spec(catalog[path.name])
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecError(f'cannot read spec {path}: {exc}') from exc
    if not isinstance(data, Mapping):
        raise SpecError(f'spec root must be an object: {path}')
    return validate_spec(data)

def list_specs(directory: Path | None=None) -> list[Path]:
    if directory is not None and Path(directory) != SPECS_DIR:
        directory = Path(directory)
        if directory.is_dir():
            return sorted(directory.glob('*.json'))
        return []
    return [SPECS_DIR / f'{name}.json' for name in sorted(_load_catalog())]

load_all_specs = lambda directory=None: [load_spec_path(p) for p in list_specs(directory)]
