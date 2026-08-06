"""GLM52 contract+evidence offline (S6 F2 C4 densified)."""
from __future__ import annotations
import sys
from pathlib import Path as _Path_repo
_REPO = _Path_repo(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import pathlib
import sys
from pathlib import Path
import pytest
HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import json
import hashlib
from collections import Counter
from lab.operators import glm52_contract as contract
from lab.operators.glm52_common import Glm52Error, canonical, resolve_artifact, seal, sha256_file, verify_sealed
from lab.layout import evidence_dir
import base64
import subprocess
from lab.operators import glm52_evidence_auth as evidence
from tools.condense.tests._glm52_fakes import FakeKeychain
CONDENSE = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = CONDENSE.parents[1]

def _read(name: str) -> dict:
    # Historical receipts retain their logical evidence paths, while live
    # evidence lives in the compact workspace taxonomy.
    candidates = (
        evidence_dir('glm52') / name,
        evidence_dir('gravity') / name,
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f'authoritative evidence not found for {name}: {candidates}')
    value = json.loads(path.read_text(encoding='utf-8'))
    return verify_sealed(value, label=name)


def _frozen_evidence_sha256(relative_path: str, expected_sha256: str) -> str:
    """Hash a frozen audit leaf without regrowing its retired report tree.

    The pre-GLM audit intentionally retains a historical score snapshot while
    current evidence is consolidated under ``workspace/campaign/evidence/``.
    Prefer the exact checkout path, then a current relocated artifact, and finally the exact
    historical Git object at that path.  A missing or altered leaf never
    passes: every branch returns the same SHA-256 comparison target.
    """
    path = REPO_ROOT / relative_path
    if path.is_file():
        observed = sha256_file(path)
        if observed == expected_sha256:
            return observed
    try:
        relocated = resolve_artifact(Path(relative_path).name)
    except Glm52Error:
        relocated = None
    if relocated is not None:
        observed = sha256_file(relocated)
        if observed == expected_sha256:
            return observed

    commits = subprocess.run(
        ['/usr/bin/git', 'log', '--all', '--format=%H', '--', relative_path],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for commit in commits:
        payload = subprocess.run(
            ['/usr/bin/git', 'show', f'{commit}:{relative_path}'],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
        )
        if payload.returncode == 0:
            observed = hashlib.sha256(payload.stdout).hexdigest()
            if observed == expected_sha256:
                return observed
    raise FileNotFoundError(
        f'frozen evidence leaf is absent or hash-divergent in checkout and local Git history: {relative_path}'
    )

@pytest.fixture(scope='module')
def artifacts() -> dict[str, dict]:
    names = ('GLM52_OFFICIAL_MANIFEST.json', 'GLM52_ARCHITECTURE_CONTRACT.json', 'GLM52_LOGICAL_WEIGHT_LEDGER.json', 'GLM52_SOURCE_FORMAT_LEDGER.json',
        'GLM52_SHARD_DEPENDENCY_GRAPH.json',
         'GLM52_STREAMING_SCHEDULE.json', 'GLM52_SOURCE_ADMISSION.json')
    return {name: _read(name) for name in names}

def test_seal_rejects_tampering() -> None:
    value = seal({'schema': 'test', 'status': 'PASS', 'count': 1})
    verify_sealed(value)
    value['count'] = 2
    with pytest.raises(Glm52Error, match='seal mismatch'):
        verify_sealed(value)

def test_classifier_is_fail_closed_and_models_mtp_separately() -> None:
    config = {'num_hidden_layers': 78, 'num_nextn_predict_layers': 1, 'first_k_dense_replace': 3, 'n_routed_experts': 256,
        'indexer_types': ['full' if layer in {0, 1, 2, *range(6, 78, 4)} else 'shared' for layer in range(78)]}
    assert len(config['indexer_types']) == 78
    assert contract.classify_tensor('model.layers.78.self_attn.indexer.wk.weight', config).category == 'indexer'
    with pytest.raises(Glm52Error, match='stored indexer tensor appears on shared layer'):
        contract.classify_tensor('model.layers.3.self_attn.indexer.wk.weight', config)
    with pytest.raises(Glm52Error, match='unrecognized'):
        contract.classify_tensor('model.layers.3.future_component.weight', config)

def test_exact_official_totals_and_dtype_caveat(artifacts: dict[str, dict]) -> None:
    manifest = artifacts['GLM52_OFFICIAL_MANIFEST.json']
    logical = artifacts['GLM52_LOGICAL_WEIGHT_LEDGER.json']
    source = artifacts['GLM52_SOURCE_FORMAT_LEDGER.json']
    assert manifest['revision'] == contract.REVISION
    assert manifest['file_count'] == 295
    assert manifest['weight_shards'] == 282
    assert manifest['source_logical_bytes'] == 1506693036946
    assert logical['tensor_count'] == 59585
    assert logical['logical_weight_denominator'] == 753329940480
    assert logical['source_dtype_summary']['BF16']['logical_weights'] == 753329921024
    assert logical['source_dtype_summary']['F32']['logical_weights'] == 19456
    assert source['tensor_payload_bytes'] == 1506659919872
    assert source['safetensors_framing_bytes'] == 7467536

def test_architecture_distinguishes_config_and_checkpoint_mtp(artifacts: dict[str, dict]) -> None:
    architecture = artifacts['GLM52_ARCHITECTURE_CONTRACT.json']
    dsa = architecture['dsa_indexshare']
    assert architecture['geometry']['main_hidden_layers'] == 78
    assert architecture['geometry']['physically_stored_layers_including_mtp'] == 79
    assert dsa['main_full_indexer_layers'] == [0, 1, 2, *range(6, 78, 4)]
    assert len(dsa['main_shared_indexer_layers']) == 57
    assert dsa['mtp_checkpoint_indexer_type'] == 'full'
    assert dsa['stored_indexer_layers'] == [0, 1, 2, *range(6, 78, 4), 78]

def test_rate_budgets_bill_every_logical_weight(artifacts: dict[str, dict]) -> None:
    logical = artifacts['GLM52_LOGICAL_WEIGHT_LEDGER.json']
    weights = logical['logical_weight_denominator']
    for row in logical['rate_budgets'].values():
        expected = weights * row['numerator'] // (8 * row['denominator'])
        assert row['maximum_complete_physical_bytes'] == expected
    major = logical['major_accounting_views']
    assert major['main_text_model_logical_weights']['logical_weights'] == 743377019904
    assert major['mtp_logical_weights']['logical_weights'] == 9952920576
    assert major['main_text_routed_expert_weights']['logical_weights'] == 724775731200
    assert major['mtp_routed_expert_weights']['logical_weights'] == 9663676416

def test_dependency_graph_partitions_every_tensor_once(artifacts: dict[str, dict]) -> None:
    graph = artifacts['GLM52_SHARD_DEPENDENCY_GRAPH.json']
    organs = graph['organs']
    assert len(organs) == 81
    names = [name for organ in organs for name in organ['tensor_names']]
    counts = Counter(names)
    assert len(names) == 59585
    assert len(counts) == 59585
    assert max(counts.values()) == 1

def test_schedule_is_one_fetch_and_carry_closed(artifacts: dict[str, dict]) -> None:
    schedule = artifacts['GLM52_STREAMING_SCHEDULE.json']
    windows = schedule['windows']
    fetched = Counter((shard for window in windows for shard in window['new_fetch_shards']))
    assert len(fetched) == 282
    assert set(fetched.values()) == {1}
    assert schedule['planned_refetches'] == 0
    assert windows[0]['carry_in_shards'] == []
    assert windows[-1]['carry_out_shards'] == []
    for left, right in zip(windows, windows[1:]):
        assert left['carry_out_shards'] == right['carry_in_shards']
    assert max((len(window['source_shards']) for window in windows)) == schedule['maximum_resident_shards_in_one_window']

def test_admission_records_header_only_and_one_copy(artifacts: dict[str, dict]) -> None:
    admission = artifacts['GLM52_SOURCE_ADMISSION.json']
    manifest = artifacts['GLM52_OFFICIAL_MANIFEST.json']
    assert admission['main_matches_pinned_revision_at_admission'] is True
    assert admission['xet']['header_range_bytes_read'] == 7467536
    assert admission['xet']['body_bytes_read'] == 0
    assert admission['local_runtime']['current_hf_xet_stack_gate'] == 'PASS'
    assert admission['local_runtime']['isolated_environment_gate'] == 'PASS'
    assert admission['local_runtime']['complete_requirements_lock_gate'] == 'PASS'
    assert admission['local_runtime']['requirements_lock']['sha256'] == sha256_file(REPO_ROOT / 'tools/condense/requirements-glm52.txt')
    assert admission['local_runtime']['packages'] == contract.package_versions()
    assert admission['toolchain_binding'] == {
        'generator': {
            'path': 'lab/operators/glm52_contract.py',
            'sha256': sha256_file(REPO_ROOT / 'lab/operators/glm52_contract.py'),
        },
        'shared_common': {
            'path': 'lab/operators/glm52_common.py',
            'sha256': sha256_file(REPO_ROOT / 'lab/operators/glm52_common.py'),
        },
        'requirements_lock': {
            'path': 'tools/condense/requirements-glm52.txt',
            'sha256': sha256_file(REPO_ROOT / 'tools/condense/requirements-glm52.txt'),
        },
    }
    assert manifest['one_copy']['weight_body_copies'] == 0

def test_pre_audit_is_frozen_and_evidence_bound() -> None:
    audit = _read('GRAVITY_COMPLETENESS_AUDIT_GLM52_PRE.json')
    assert audit['snapshot']['later_glm_artifacts_excluded_from_scores'] is True
    assert {model: row['total'] for model, row in audit['scores'].items()} == {'GLM52_PRE': 16, 'GPT_OSS_120B': 81, 'KIMI_K26': 69, 'QWEN3_235B': 84}
    assert audit['scores']['GLM52_PRE']['maximum'] == 105
    for rows in audit['evidence'].values():
        for row in rows:
            assert _frozen_evidence_sha256(row['path'], row['sha256']) == row['sha256']

def test_external_matrix_separates_nominal_from_canonical_rates() -> None:
    matrix = _read('GRAVITY_EXTERNAL_BASELINE_MATRIX.json')
    assert len(matrix['methods']) == 13
    qmoe = next((row for row in matrix['methods'] if row['method'] == 'QMoE'))
    assert qmoe['closest_giant_moe_prior'] is True
    assert qmoe['rate']['canonical_artifact_bpw'] == 0.807
    stbllm = next((row for row in matrix['methods'] if row['method'] == 'STBLLM'))
    assert stbllm['rate']['nominal_or_method_bpw'][0] < 1
    assert stbllm['rate']['decoded_tensor_payload_bpw']['2_of_4_kernel_floor_before_metadata'] > 1
    assert 'First sub-1-bit PTQ.' in matrix['claim_policy']['unsafe']
    required = {'source_or_teacher_precision', 'architecture_class', 'largest_evaluated_scale', 'compression_regime', 'weight_activation_scope',
        'physical_accounting_level'}
    for row in matrix['methods']:
        assert set(row['structured_comparison']) == required
        precision = row['structured_comparison']['source_or_teacher_precision']
        assert 'BF16' in precision or precision.startswith('none:') or precision.startswith('NOT_REPORTED_BY_PRIMARY_SOURCE;')
        for source in row['sources']:
            if source['kind'] == 'paper':
                assert len(source['content_identity']['sha256']) == 64
            else:
                assert len(source['commit']) == 40
    assert matrix['methods']
CAMPAIGN = 'glm52-evidence-auth-test'
REVISION = 'b4734de4facf877f85769a911abafc5283eab3d9'
KEY = bytes(range(32))

def test_configure_is_idempotent_secret_free_and_loads_bound_auth() -> None:
    keychain = FakeKeychain()
    first = evidence.configure_hmac_key(keychain, random_bytes=lambda size: KEY)
    second = evidence.configure_hmac_key(keychain, random_bytes=lambda _size: (_ for _ in ()).throw(AssertionError()))
    assert first['status'] == 'CONFIGURED'
    assert second['status'] == 'ALREADY_CONFIGURED'
    assert first['key_identity_digest'] == second['key_identity_digest']
    assert len(keychain.set_calls) == 1
    assert KEY not in repr(first).encode()
    assert evidence.credential_status(keychain)['ready'] is True
    auth = evidence.load_evidence_auth(keychain, campaign_id=CAMPAIGN, source_revision=REVISION)
    assert auth.campaign_id == CAMPAIGN
    assert auth.source_revision == REVISION
    assert 'redacted' in repr(auth)

@pytest.mark.parametrize('encoded', ('invalid!!!', '', base64.urlsafe_b64encode(b'x').decode()))
def test_malformed_keychain_values_fail_closed(encoded: str) -> None:
    keychain = FakeKeychain({evidence.EVIDENCE_HMAC_SERVICE: encoded})
    assert evidence.credential_status(keychain) == {'configured': False, 'ready': False}
    with pytest.raises(evidence.EvidenceSecurityError, match='invalid'):
        evidence.load_evidence_auth(keychain, campaign_id=CAMPAIGN, source_revision=REVISION)

def test_configure_requires_verified_keychain_persistence() -> None:
    keychain = FakeKeychain(discard_writes=True)
    with pytest.raises(evidence.EvidenceSecurityError, match='post-write'):
        evidence.configure_hmac_key(keychain, random_bytes=lambda size: KEY)

def test_macos_keychain_write_keeps_secret_out_of_subprocesses() -> None:
    subprocess_calls = []
    native_calls = []

    def runner(arguments, **kwargs):
        subprocess_calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, 0, stdout='', stderr='')

    def writer(service, account, value):
        native_calls.append((service, account, value))
    provider = evidence.MacOSKeychain(runner=runner, native_reader=lambda _service, _account: None, native_writer=writer)
    encoded = base64.urlsafe_b64encode(KEY).decode('ascii')
    provider.set(evidence.EVIDENCE_HMAC_SERVICE, encoded)
    assert subprocess_calls == []
    assert native_calls == [(evidence.EVIDENCE_HMAC_SERVICE, evidence.KEYCHAIN_ACCOUNT, encoded)]

def test_macos_keychain_read_keeps_secret_out_of_subprocesses() -> None:
    subprocess_calls = []
    native_calls = []
    encoded = base64.urlsafe_b64encode(KEY).decode('ascii')

    def runner(arguments, **kwargs):
        subprocess_calls.append((arguments, kwargs))
        raise AssertionError('credential read must not spawn a process')

    def reader(service, account):
        native_calls.append((service, account))
        return encoded
    provider = evidence.MacOSKeychain(runner=runner, native_reader=reader, native_writer=lambda *_args: None)
    assert provider.get(evidence.EVIDENCE_HMAC_SERVICE) == encoded
    assert subprocess_calls == []
    assert native_calls == [(evidence.EVIDENCE_HMAC_SERVICE, evidence.KEYCHAIN_ACCOUNT)]

def test_wrong_service_is_rejected_before_keychain_access() -> None:
    provider = evidence.MacOSKeychain(runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()))
    with pytest.raises(evidence.EvidenceSecurityError, match='unrecognized'):
        provider.get('com.example.wrong')
