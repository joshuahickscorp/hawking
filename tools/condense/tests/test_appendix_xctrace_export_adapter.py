from __future__ import annotations

import copy
import json
import os
import pathlib
import sys
import time
import xml.etree.ElementTree as ET

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
TESTS = CONDENSE / "tests"
sys.path.insert(0, str(CONDENSE))
sys.path.insert(0, str(TESTS))

import appendix_physical_counter_normalizer as normalizer  # noqa: E402
import appendix_xctrace_export_adapter as adapter  # noqa: E402
from test_appendix_physical_counter_collector import _bundle  # noqa: E402


def _phase_bundle(kind: str = "device") -> dict:
    bundle = _bundle(kind)
    phase = bundle["raw_probe"]["phase_markers"]
    interval_hashes: dict[str, str] = {}
    intervals_by_old_hash: dict[str, dict] = {}
    for interval in phase["intervals"]:
        old_hash = interval.pop("interval_sha256")
        identity = {
            "schema": "hawking.physical_phase_interval_identity.v1",
            "run_nonce": interval["run_nonce"],
            "sequence": interval["sequence"],
            "phase": interval["phase"],
            "role": interval["role"],
            "batch": interval["batch"],
            "iteration": interval["iteration"],
        }
        interval_id = adapter.canonical_sha256(identity)
        interval["interval_id"] = interval_id
        interval["signpost_id"] = adapter._expected_signpost_id(interval_id)
        interval["interval_sha256"] = adapter.canonical_sha256(interval)
        interval_hashes[old_hash] = interval["interval_sha256"]
        intervals_by_old_hash[old_hash] = interval
    marker_hashes: dict[str, str] = {}
    for pair in phase["pairs"]:
        old_marker = pair.pop("phase_marker_sha256")
        old_baseline = pair["baseline_interval_sha256"]
        pair["baseline_interval_sha256"] = interval_hashes[old_baseline]
        pair["baseline_interval_id"] = intervals_by_old_hash[old_baseline]["interval_id"]
        role_hash_field = (
            "candidate_interval_sha256"
            if "candidate_interval_sha256" in pair
            else "verifier_interval_sha256"
        )
        role_id_field = role_hash_field.replace("_sha256", "_id")
        old_role = pair[role_hash_field]
        pair[role_hash_field] = interval_hashes[old_role]
        pair[role_id_field] = intervals_by_old_hash[old_role]["interval_id"]
        pair["phase_marker_sha256"] = adapter.canonical_sha256(pair)
        marker_hashes[old_marker] = pair["phase_marker_sha256"]
    phase.pop("phase_markers_sha256")
    phase["phase_markers_sha256"] = adapter.canonical_sha256(phase)
    raw = bundle["raw_probe"]
    if kind == "device":
        raw["benchmark"]["trial_phase_marker_sha256"] = [
            marker_hashes[value]
            for value in raw["benchmark"]["trial_phase_marker_sha256"]
        ]
    else:
        raw["measurement_protocol"]["phase_markers_sha256"] = phase["phase_markers_sha256"]
        for batch in raw["measurement_protocol"]["batches"]:
            for repeat in batch["repeats"]:
                repeat["phase_marker_sha256"] = marker_hashes[
                    repeat["phase_marker_sha256"]
                ]
    bundle.pop("raw_bundle_sha256")
    bundle["raw_bundle_sha256"] = adapter.canonical_sha256(bundle)
    return bundle


def _base_label(interval: dict) -> str:
    batch = "none" if interval["batch"] is None else str(interval["batch"])
    return (
        "hawking.physical.v1"
        f"|interval_id={interval['interval_id']}"
        f"|run_nonce={interval['run_nonce']}"
        f"|phase={interval['phase']}|role={interval['role']}"
        f"|batch={batch}|iteration={interval['iteration']}"
    )


def _table_rows(bundle: dict) -> dict[str, list[dict]]:
    tables = {table: [] for table in adapter.REQUIRED_TABLES}
    intervals = bundle["raw_probe"]["phase_markers"]["intervals"]
    for ordinal, interval in enumerate(intervals):
        base = _base_label(interval)
        for event_type, timestamp in (
            ("begin", interval["continuous_started_ns"] + 1),
            ("end", interval["continuous_ended_ns"] - 1),
        ):
            tables["signposts"].append({
                "signpost_event_id": f"sp-{ordinal}-{event_type}",
                "signpost_name": "HawkingPhysicalPhase",
                "signpost_payload": base,
                "signpost_type": event_type,
                "signpost_id": interval["signpost_id"],
                "signpost_timestamp_continuous_ns": timestamp,
                "process_id": 1234,
            })
        command_id, encoder_id = f"cb-{ordinal}", f"enc-{ordinal}"
        tables["command_buffers"].append({
            "command_buffer_id": command_id,
            "command_buffer_label": f"{base}|kind=command_buffer|command_index=0",
            "process_id": 1234,
            "metal_registry_id": "metal-test-1",
        })
        tables["encoders"].append({
            "encoder_id": encoder_id,
            "command_buffer_id": command_id,
            "encoder_label": (
                f"{base}|kind=compute_encoder|command_index=0"
                f"|encoder_index=0|kernel=test_kernel"
            ),
            "process_id": 1234,
            "metal_registry_id": "metal-test-1",
        })
        for event, gpu, physical, occupancy in ((0, 2, 100, 25.0), (1, 3, 200, 75.0)):
            tables["counters"].append({
                "source_event_id": f"counter-{ordinal}-{event}",
                "command_buffer_id": command_id,
                "encoder_id": encoder_id,
                "process_id": 1234,
                "metal_registry_id": "metal-test-1",
                "gpu_time_ns": gpu,
                "physical_bytes": physical,
                "occupancy_percent": occupancy,
                "skipped": False,
            })
    return tables


def _xml_table(table: str, rows: list[dict]) -> bytes:
    root = ET.Element("trace-query-result")
    schema = ET.SubElement(root, "schema")
    for index, (field, (unit, _kind)) in enumerate(adapter.TABLE_COLUMNS[table].items()):
        column = ET.SubElement(schema, "column", {"id": f"schema-{index}"})
        ET.SubElement(column, "mnemonic").text = field
        ET.SubElement(column, "unit").text = unit
    data = ET.SubElement(root, "rows")
    for values in rows:
        row = ET.SubElement(data, "row")
        for field in adapter.TABLE_COLUMNS[table]:
            cell = ET.SubElement(row, "cell")
            value = values[field]
            ET.SubElement(cell, "value").text = (
                "true" if value is True else "false" if value is False else str(value)
            )
    return ET.tostring(root)


def _toc(bundle: dict) -> bytes:
    intervals = bundle["raw_probe"]["phase_markers"]["intervals"]
    values = {
        "capture_started_at_unix_ns": min(i["wall_started_unix_ns"] for i in intervals) - 10,
        "capture_ended_at_unix_ns": max(i["wall_ended_unix_ns"] for i in intervals) + 10,
        "capture_started_at_continuous_ns": min(i["continuous_started_ns"] for i in intervals) - 10,
        "capture_ended_at_continuous_ns": max(i["continuous_ended_ns"] for i in intervals) + 10,
    }
    root = ET.Element("trace-toc")
    ET.SubElement(root, "template", {"name": "Metal System Trace"})
    for field, (unit, _kind) in adapter.TRACE_BOUND_COLUMNS.items():
        ET.SubElement(root, "bound", {"column": field, "unit": unit}).text = str(values[field])
    return ET.tostring(root)


def _profile(toc_bytes: bytes, table_bytes: dict[str, bytes], *, production: bool = False) -> dict:
    toc_doc = ET.fromstring(toc_bytes)
    exports = {}
    for table in adapter.REQUIRED_TABLES:
        document = ET.fromstring(table_bytes[table])
        columns = {}
        for index, (field, (unit, kind)) in enumerate(adapter.TABLE_COLUMNS[table].items()):
            columns[field] = {
                "column_id": field,
                "schema_index": index,
                "row_child_index": index,
                "value_xpath": "./value",
                "value_source": "text",
                "raw_unit": unit,
                "unit": unit,
                "value_type": kind,
                "scale_numerator": 1,
                "scale_denominator": 1,
            }
        exports[table] = {
            "document_format": "xml",
            "xpath": f"/trace-toc/run/data/table[@schema='{table}']",
            "row_xpath": ".//rows/row",
            "selector_mode": "positional-xctrace-xml-v1",
            "schema_columns_xpath": ".//schema/column",
            "schema_mnemonic_xpath": "./mnemonic",
            "schema_unit_xpath": "./unit",
            "fingerprint_value_attributes": ["id", "ref"],
            "schema_fingerprint_sha256": adapter.schema_fingerprint(
                document, "xml", ["id", "ref"],
            ),
            "columns": columns,
        }
    bounds = {
        field: {
            "column_id": field,
            "xpath": f".//bound[@column='{field}']",
            "value_source": "text",
            "unit": unit,
            "value_type": kind,
        }
        for field, (unit, kind) in adapter.TRACE_BOUND_COLUMNS.items()
    }
    issued = time.time_ns() - 1_000_000
    profile = {
        "schema": adapter.PROFILE_SCHEMA,
        "profile_id": "synthetic-positional-multi-table",
        "production_approved": production,
        "synthetic_fixture": not production,
        "operator_review": {
            "schema": adapter.REVIEW_SCHEMA,
            "profile_payload_sha256": "0" * 64,
            "issued_at_unix_ns": issued,
            "expires_at_unix_ns": issued + 3_600_000_000_000,
            "signer_identity": adapter.authority_root.SIGNER_IDENTITY,
            "signature_namespace": adapter.PROFILE_SSHSIG_NAMESPACE,
            "allowed_signers": {
                "path": "/synthetic/allowed_signers", "sha256": "3" * 64,
                "size_bytes": 1,
            },
            "detached_signature": {
                "path": "/synthetic/profile.sshsig", "sha256": "4" * 64,
                "size_bytes": 1,
            },
            "envelope_sha256": "0" * 64,
        },
        "xctrace": {
            "binary": {
                "path": str(adapter.FULL_XCODE_XCTRACE), "sha256": "1" * 64,
                "size_bytes": 1,
            },
            "version_argv": [str(adapter.FULL_XCODE_XCTRACE), "version"],
            "version_string": "synthetic xctrace",
            "version_stdout_sha256": "2" * 64,
            "template_name": "Metal System Trace",
            "environment": adapter.PINNED_EXPORT_ENV,
        },
        "toc": {
            "document_format": "xml",
            "schema_fingerprint_sha256": adapter.schema_fingerprint(
                toc_doc, "xml", ["name", "column", "id", "ref"],
            ),
            "fingerprint_value_attributes": ["name", "column", "id", "ref"],
            "template_xpath": ".//template",
            "template_value_source": "attribute:name",
            "capture_bounds": bounds,
        },
        "tables": {"exports": exports, "aggregation": adapter.AGGREGATION_RULE},
    }
    profile["operator_review"]["profile_payload_sha256"] = adapter.canonical_sha256(
        adapter._reviewed_payload(profile)
    )
    profile["operator_review"] = adapter._stamp(
        profile["operator_review"], "envelope_sha256",
    )
    return adapter._stamp(profile, "profile_sha256")


def _fixture(tmp_path: pathlib.Path, *, production: bool = False):
    bundle = _phase_bundle()
    tables = _table_rows(bundle)
    toc_bytes = _toc(bundle)
    table_bytes = {table: _xml_table(table, tables[table]) for table in adapter.REQUIRED_TABLES}
    profile = _profile(toc_bytes, table_bytes, production=production)
    paths = {
        "bundle": tmp_path / "bundle.json",
        "profile": tmp_path / "profile.json",
        "toc": tmp_path / "toc.xml",
    }
    paths["bundle"].write_text(json.dumps(bundle), encoding="utf-8")
    paths["profile"].write_text(json.dumps(profile), encoding="utf-8")
    paths["toc"].write_bytes(toc_bytes)
    export_paths = {}
    for table, raw in table_bytes.items():
        export_paths[table] = tmp_path / f"{table}.xml"
        export_paths[table].write_bytes(raw)
    trace = tmp_path / "metal.trace"
    trace.mkdir()
    (trace / "raw.data").write_bytes(b"synthetic trace; zero physical credit")
    if production:
        (trace / "raw.data").chmod(0o444)
        trace.chmod(0o555)
    return bundle, tables, profile, paths, trace, export_paths


def _build(tmp_path: pathlib.Path):
    bundle, tables, profile, paths, trace, export_paths = _fixture(tmp_path)
    capture, receipt = adapter.build_capture(
        kind="device", raw_bundle_path=paths["bundle"], profile_path=paths["profile"],
        trace_path=trace, toc_path=paths["toc"], export_paths=export_paths,
        probe_pid=1234, run_nonce=bundle["execution_authority"]["run_nonce"],
        probe_argv_sha256=bundle["execution_authority"]["argv_sha256"],
        metal_registry_id="metal-test-1", production=False,
    )
    return bundle, tables, profile, paths, trace, export_paths, capture, receipt


def test_multi_table_n_event_capture_is_exact_and_synthetic_zero_credit(tmp_path: pathlib.Path) -> None:
    bundle, _tables, _profile_value, _paths, _trace, _exports, capture, receipt = _build(tmp_path)
    assert normalizer._capture_errors(
        capture, schema=normalizer.METAL_CAPTURE_SCHEMA, backend=normalizer.METAL_BACKEND,
        probe_pid=1234, run_nonce=bundle["execution_authority"]["run_nonce"],
        probe_argv_sha256=bundle["execution_authority"]["argv_sha256"],
        metal_registry_id="metal-test-1",
    ) == []
    assert capture["records"][0]["gpu_time_ns"] == 5
    assert capture["records"][0]["physical_bytes"] == 300
    assert capture["records"][0]["occupancy_percent"] == 55.0
    assert capture["records"][0]["bandwidth_bytes_per_second"] == 60_000_000_000.0
    assert receipt["synthetic_fixture"] is True
    assert receipt["physical_evidence_eligible"] is False
    assert receipt["source_row_counts"] == receipt["consumed_row_counts"]


@pytest.mark.parametrize("mutation", ["missing", "mismatch", "reversed", "reused"])
def test_signpost_pair_adversaries_fail_closed(tmp_path: pathlib.Path, mutation: str) -> None:
    bundle = _phase_bundle()
    tables = _table_rows(bundle)
    signposts = tables["signposts"]
    if mutation == "missing":
        signposts.pop(1)
    elif mutation == "mismatch":
        signposts[1]["signpost_id"] += 1
    elif mutation == "reversed":
        signposts[0]["signpost_timestamp_continuous_ns"], signposts[1]["signpost_timestamp_continuous_ns"] = (
            signposts[1]["signpost_timestamp_continuous_ns"],
            signposts[0]["signpost_timestamp_continuous_ns"],
        )
    else:
        signposts[1]["signpost_event_id"] = signposts[0]["signpost_event_id"]
    toc_bytes = _toc(bundle)
    table_bytes = {table: _xml_table(table, tables[table]) for table in adapter.REQUIRED_TABLES}
    profile = _profile(toc_bytes, table_bytes)
    bundle_path, profile_path, toc_path = tmp_path / "b.json", tmp_path / "p.json", tmp_path / "t.xml"
    bundle_path.write_text(json.dumps(bundle)); profile_path.write_text(json.dumps(profile)); toc_path.write_bytes(toc_bytes)
    exports = {}
    for table, raw in table_bytes.items():
        exports[table] = tmp_path / f"{table}.xml"; exports[table].write_bytes(raw)
    trace = tmp_path / "x.trace"; trace.mkdir(); (trace / "raw").write_bytes(b"raw")
    with pytest.raises(adapter.XctraceAdapterError, match="signpost"):
        adapter.build_capture(
            kind="device", raw_bundle_path=bundle_path, profile_path=profile_path,
            trace_path=trace, toc_path=toc_path, export_paths=exports, probe_pid=1234,
            run_nonce=bundle["execution_authority"]["run_nonce"],
            probe_argv_sha256=bundle["execution_authority"]["argv_sha256"],
            metal_registry_id="metal-test-1", production=False,
        )


def test_unconsumed_and_cross_linked_rows_fail_closed(tmp_path: pathlib.Path) -> None:
    bundle = _phase_bundle(); tables = _table_rows(bundle)
    extra = copy.deepcopy(tables["command_buffers"][0]); extra["command_buffer_id"] = "extra"
    tables["command_buffers"].append(extra)
    with pytest.raises(adapter.XctraceAdapterError, match="command-buffer rows"):
        adapter._join_tables(
            tables, adapter._bundle_interval_population(
                bundle, adapter._bundle_targets(
                    bundle, kind="device", run_nonce=bundle["execution_authority"]["run_nonce"],
                    probe_argv_sha256=bundle["execution_authority"]["argv_sha256"],
                ),
            ),
            probe_pid=1234, metal_registry_id="metal-test-1",
            run_nonce=bundle["execution_authority"]["run_nonce"],
        )
    tables = _table_rows(bundle)
    tables["counters"][0]["command_buffer_id"] = "wrong"
    with pytest.raises(adapter.XctraceAdapterError, match="encoder and command"):
        adapter._join_tables(
            tables, adapter._bundle_interval_population(
                bundle, adapter._bundle_targets(
                    bundle, kind="device", run_nonce=bundle["execution_authority"]["run_nonce"],
                    probe_argv_sha256=bundle["execution_authority"]["argv_sha256"],
                ),
            ), probe_pid=1234, metal_registry_id="metal-test-1",
            run_nonce=bundle["execution_authority"]["run_nonce"],
        )


def test_positional_schema_mnemonic_unit_and_order_are_bound(tmp_path: pathlib.Path) -> None:
    _bundle_value, _tables, profile, _paths, _trace, exports = _fixture(tmp_path)
    document = ET.parse(exports["counters"]).getroot()
    document.find(".//schema/column/mnemonic").text = "forged"
    with pytest.raises(adapter.XctraceAdapterError, match="mnemonic"):
        adapter._parse_rows(document, profile["tables"]["exports"]["counters"])


def test_signed_review_covers_expiry_and_caps_validity(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bundle_value, _tables, profile, _paths, _trace, _exports = _fixture(tmp_path, production=True)
    first = adapter.appendix_contract.canonical_bytes(
        adapter._review_signed_payload(profile["operator_review"])
    )
    forged = copy.deepcopy(profile)
    forged["operator_review"]["expires_at_unix_ns"] += 1
    second = adapter.appendix_contract.canonical_bytes(
        adapter._review_signed_payload(forged["operator_review"])
    )
    assert first != second
    monkeypatch.setattr(adapter.authority_root, "load_default_registry", lambda: {})
    monkeypatch.setattr(
        adapter.authority_root, "allowed_signers_identity",
        lambda _registry: profile["operator_review"]["allowed_signers"],
    )
    forged["operator_review"] = adapter._stamp(
        forged["operator_review"], "envelope_sha256",
    )
    forged = adapter._stamp(forged, "profile_sha256")
    with pytest.raises(adapter.XctraceAdapterError, match="SSHSIG"):
        adapter.validate_profile(
            forged, production=True,
            signature_verifier=lambda _review, payload: (payload == first, "payload differs"),
        )
    overlong = copy.deepcopy(profile)
    overlong["operator_review"]["expires_at_unix_ns"] = (
        overlong["operator_review"]["issued_at_unix_ns"]
        + adapter.MAX_PROFILE_VALIDITY_NS + 1
    )
    overlong["operator_review"] = adapter._stamp(overlong["operator_review"], "envelope_sha256")
    overlong = adapter._stamp(overlong, "profile_sha256")
    with pytest.raises(adapter.XctraceAdapterError, match="seven days"):
        adapter.validate_profile(
            overlong, production=True,
            signature_verifier=lambda _review, _payload: (True, "valid signature"),
        )


def test_signpost_u64_collision_and_reserved_ids_fail_closed() -> None:
    assert adapter._expected_signpost_id("0" * 64) == 1
    assert adapter._expected_signpost_id("f" * 64) == 1
    first = {"interval_id": "1" * 16 + "0" * 48, "interval": {"signpost_id": int("1" * 16, 16)}}
    second = {"interval_id": "1" * 16 + "2" * 48, "interval": {"signpost_id": int("1" * 16, 16)}}
    with pytest.raises(adapter.XctraceAdapterError, match="collides"):
        adapter._join_tables(
            {table: [] for table in adapter.REQUIRED_TABLES}, [first, second],
            probe_pid=1234, metal_registry_id="metal-test-1", run_nonce="1" * 64,
        )


def test_production_trace_tree_must_be_frozen(tmp_path: pathlib.Path) -> None:
    trace = tmp_path / "mutable.trace"
    trace.mkdir()
    (trace / "raw").write_bytes(b"raw")
    with pytest.raises(adapter.XctraceAdapterError, match="writable"):
        adapter.trace_tree_identity(trace, require_immutable=True)
    (trace / "raw").chmod(0o444)
    trace.chmod(0o555)
    assert adapter.trace_tree_identity(trace, require_immutable=True)["tree_sha256"]


def _production_receipt_fixture(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    bundle, _tables, profile, paths, trace, exports = _fixture(tmp_path, production=True)
    monkeypatch.setattr(adapter.authority_root, "load_default_registry", lambda: {})
    monkeypatch.setattr(
        adapter.authority_root, "allowed_signers_identity",
        lambda _registry: profile["operator_review"]["allowed_signers"],
    )
    directory_stat = tmp_path.stat()
    output_directory = {
        "held": True, "path": str(tmp_path),
        "device": directory_stat.st_dev, "inode": directory_stat.st_ino,
    }
    operational_fd = 91
    operational = {
        "toc": f"/dev/fd/{operational_fd}/{paths['toc'].name}",
        **{table: f"/dev/fd/{operational_fd}/{path.name}" for table, path in exports.items()},
    }
    binary = profile["xctrace"]["binary"]["path"]
    toc_argv = [binary, "export", "--input", str(trace), "--toc", "--output", operational["toc"]]
    export_argvs = {
        table: [
            binary, "export", "--input", str(trace), "--xpath",
            profile["tables"]["exports"][table]["xpath"], "--output", operational[table],
        ] for table in adapter.REQUIRED_TABLES
    }
    runtime = {
        "binary": profile["xctrace"]["binary"],
        "version_string": profile["xctrace"]["version_string"],
        "version_stdout_sha256": profile["xctrace"]["version_stdout_sha256"],
        "template_name": profile["xctrace"]["template_name"],
        "toc_argv": toc_argv, "toc_argv_sha256": adapter.canonical_sha256(toc_argv),
        "export_argvs": export_argvs,
        "export_argv_sha256s": {table: adapter.canonical_sha256(argv) for table, argv in export_argvs.items()},
        "shell": False, "environment": adapter.PINNED_EXPORT_ENV,
        "environment_sha256": adapter.canonical_sha256(adapter.PINNED_EXPORT_ENV),
        "output_directory": output_directory,
        "operational_output_dir_fd": operational_fd,
        "operational_output_paths": operational,
        "published_output_paths": {
            "toc": str(paths["toc"]), **{table: str(path) for table, path in exports.items()},
        },
    }
    lease = {"inherited": True, "device": 9, "inode": 10}
    capture, receipt = adapter.build_capture(
        kind="device", raw_bundle_path=paths["bundle"], profile_path=paths["profile"],
        trace_path=trace, toc_path=paths["toc"], export_paths=exports,
        probe_pid=1234, run_nonce=bundle["execution_authority"]["run_nonce"],
        probe_argv_sha256=bundle["execution_authority"]["argv_sha256"],
        metal_registry_id="metal-test-1", production=True, lease=lease,
        xctrace_runtime=runtime, signature_verifier=lambda _review, _payload: (True, "ok"),
    )
    return bundle, paths, trace, exports, output_directory, lease, capture, receipt


@pytest.mark.parametrize("forgery", ["metric", "source_id"])
def test_receipt_verifier_recomputes_and_rejects_forgery(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, forgery: str,
) -> None:
    bundle, paths, trace, exports, output_directory, lease, capture, receipt = _production_receipt_fixture(
        tmp_path, monkeypatch,
    )
    forged = copy.deepcopy(capture)
    if forgery == "metric":
        forged["records"][0]["gpu_time_ns"] += 1
    else:
        forged["records"][0]["source_sample_id"] = "f" * 64
    forged = adapter._stamp(forged, "capture_sha256")
    forged_receipt = copy.deepcopy(receipt)
    forged_receipt["capture_sha256"] = forged["capture_sha256"]
    forged_receipt = adapter._stamp(forged_receipt, "receipt_sha256")
    capture_path, receipt_path = tmp_path / "capture.json", tmp_path / "receipt.json"
    capture_path.write_text(json.dumps(forged)); receipt_path.write_text(json.dumps(forged_receipt))
    with pytest.raises(adapter.XctraceAdapterError, match="recompute"):
        adapter.validate_receipt(
            receipt_path=receipt_path, capture_path=capture_path, kind="device",
            raw_bundle_path=paths["bundle"], profile_path=paths["profile"],
            trace_path=trace, toc_path=paths["toc"], export_paths=exports,
            probe_pid=1234, run_nonce=bundle["execution_authority"]["run_nonce"],
            probe_argv_sha256=bundle["execution_authority"]["argv_sha256"],
            metal_registry_id="metal-test-1", expected_lease=lease,
            expected_output_directory=output_directory,
            signature_verifier=lambda _review, _payload: (True, "ok"),
        )


def test_receipt_validation_survives_dirfd_close_and_reopen(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, paths, trace, exports, output_directory, lease, capture, receipt = _production_receipt_fixture(
        tmp_path, monkeypatch,
    )
    capture_path, receipt_path = tmp_path / "capture.json", tmp_path / "receipt.json"
    capture_path.write_text(json.dumps(capture)); receipt_path.write_text(json.dumps(receipt))
    first_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)); os.close(first_fd)
    other = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)); os.close(other)
    result = adapter.validate_receipt(
        receipt_path=receipt_path, capture_path=capture_path, kind="device",
        raw_bundle_path=paths["bundle"], profile_path=paths["profile"],
        trace_path=trace, toc_path=paths["toc"], export_paths=exports,
        probe_pid=1234, run_nonce=bundle["execution_authority"]["run_nonce"],
        probe_argv_sha256=bundle["execution_authority"]["argv_sha256"],
        metal_registry_id="metal-test-1", expected_lease=lease,
        expected_output_directory=output_directory,
        signature_verifier=lambda _review, _payload: (True, "ok"),
    )
    assert result["all_provenance_files_reopened"] is True
