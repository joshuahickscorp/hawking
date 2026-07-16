//! Exact-process energy sampling for the two Appendix release probes.
//!
//! Library and SDK provenance is produced by the separately reviewed Python
//! authority.  This module only accepts that hash-bound object; it never
//! invents provenance from the probe process.  Counter snapshots themselves
//! are taken in-process immediately around the exact operation-only
//! `PhaseRecorder` interval.

use anyhow::{bail, Context, Result};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::ffi::c_void;
use std::fs::{self, File};
use std::io::Read;
use std::mem;
use std::os::raw::c_int;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

const PROBE_COUNTERS_SCHEMA: &str = "hawking.probe_process_energy_counters.v1";
const SNAPSHOT_SCHEMA: &str = "hawking.libproc_rusage_v6_snapshot.v1";
const PHASE_RECORD_SCHEMA: &str = "hawking.libproc_process_joule_phase_record.v1";
const PROVENANCE_SCHEMA: &str = "hawking.libproc_rusage_v6_library_provenance.v1";
const BACKEND_ID: &str = "darwin-libproc-proc_pid_rusage-v6-ri_energy_nj-v1";
const BOUNDARY_PROTOCOL: &str = "probe-self-sampled-bracketing-phase-interval-v2";
const LIBPROC_INSTALL_NAME: &str = "/usr/lib/libproc.dylib";
const COLLECTOR_CONTRACT_SHA256: &str =
    "0c832e80b1336a2ceb88cb401590981d01478f3cad2b75aa15c7381e44984dba";
const STRUCT_LAYOUT_SHA256: &str =
    "f6f0425fd8a4b3e054077ca68ea4856dd77c7771597bfd065aa26b3c36d66494";
const RUSAGE_INFO_V6: c_int = 6;
const MAX_PROVENANCE_BYTES: u64 = 1024 * 1024;

const RUSAGE_V6_U64_FIELDS: [&str; 50] = [
    "ri_user_time",
    "ri_system_time",
    "ri_pkg_idle_wkups",
    "ri_interrupt_wkups",
    "ri_pageins",
    "ri_wired_size",
    "ri_resident_size",
    "ri_phys_footprint",
    "ri_proc_start_abstime",
    "ri_proc_exit_abstime",
    "ri_child_user_time",
    "ri_child_system_time",
    "ri_child_pkg_idle_wkups",
    "ri_child_interrupt_wkups",
    "ri_child_pageins",
    "ri_child_elapsed_abstime",
    "ri_diskio_bytesread",
    "ri_diskio_byteswritten",
    "ri_cpu_time_qos_default",
    "ri_cpu_time_qos_maintenance",
    "ri_cpu_time_qos_background",
    "ri_cpu_time_qos_utility",
    "ri_cpu_time_qos_legacy",
    "ri_cpu_time_qos_user_initiated",
    "ri_cpu_time_qos_user_interactive",
    "ri_billed_system_time",
    "ri_serviced_system_time",
    "ri_logical_writes",
    "ri_lifetime_max_phys_footprint",
    "ri_instructions",
    "ri_cycles",
    "ri_billed_energy",
    "ri_serviced_energy",
    "ri_interval_max_phys_footprint",
    "ri_runnable_time",
    "ri_flags",
    "ri_user_ptime",
    "ri_system_ptime",
    "ri_pinstructions",
    "ri_pcycles",
    "ri_energy_nj",
    "ri_penergy_nj",
    "ri_secure_time_in_system",
    "ri_secure_ptime_in_system",
    "ri_neural_footprint",
    "ri_lifetime_max_neural_footprint",
    "ri_interval_max_neural_footprint",
    "ri_conclave_footprint",
    "ri_page_wait_time_mach",
    "ri_page_cache_hits",
];

const MONOTONE_FIELD_INDEXES: [usize; 8] = [0, 1, 29, 30, 38, 39, 40, 41];
const RI_PROC_START_ABSTIME_INDEX: usize = 8;
const RI_ENERGY_NJ_INDEX: usize = 40;

#[repr(C)]
struct RUsageInfoV6 {
    ri_uuid: [u8; 16],
    counters: [u64; RUSAGE_V6_U64_FIELDS.len()],
    ri_reserved: [u64; 6],
}

#[repr(C)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}

#[link(name = "proc", kind = "dylib")]
extern "C" {
    fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *mut c_void) -> c_int;
}

extern "C" {
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> c_int;
}

/// One locally generated, self-hashed `proc_pid_rusage` snapshot.
pub struct Snapshot {
    value: Value,
    pid: u32,
    process_uuid: String,
    process_start_abstime: u64,
    monotone: [u64; MONOTONE_FIELD_INDEXES.len()],
    energy_nj: u64,
    read_started_at_unix_ns: u64,
    read_ended_at_unix_ns: u64,
    read_started_at_continuous_ns: u64,
    read_ended_at_continuous_ns: u64,
    snapshot_sha256: String,
}

/// Pinned sampler plus authority-produced provenance for one probe PID.
pub struct ProcessJouleCollector {
    pid: u32,
    library_provenance: Value,
}

fn wall_unix_ns() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("wall clock precedes UNIX epoch")?
        .as_nanos()
        .try_into()
        .context("wall clock nanoseconds overflow u64")
}

fn continuous_ns() -> Result<u64> {
    let mut info = MachTimebaseInfo { numer: 0, denom: 0 };
    if unsafe { mach_timebase_info(&mut info) } != 0 || info.denom == 0 {
        bail!("mach_timebase_info failed while sampling process energy");
    }
    let ticks = unsafe { mach_absolute_time() } as u128;
    (ticks * info.numer as u128 / info.denom as u128)
        .try_into()
        .context("mach continuous nanoseconds overflow u64")
}

fn ordered(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut output = Map::new();
            for key in keys {
                output.insert(key.clone(), ordered(&map[key]));
            }
            Value::Object(output)
        }
        Value::Array(rows) => Value::Array(rows.iter().map(ordered).collect()),
        scalar => scalar.clone(),
    }
}

fn canonical_sha256(value: &Value) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&ordered(value))?)
    ))
}

fn stamp_value(mut value: Value, field: &str) -> Result<Value> {
    let digest = canonical_sha256(&value)?;
    value
        .as_object_mut()
        .context("only JSON objects can be stamped")?
        .insert(field.to_owned(), Value::String(digest));
    Ok(value)
}

fn digest_field(value: &Value, field: &str) -> Result<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .with_context(|| format!("stamped process-energy object lacks {field}"))
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn exact_fields(map: &Map<String, Value>, expected: &[&str]) -> bool {
    map.len() == expected.len() && expected.iter().all(|field| map.contains_key(*field))
}

fn validate_file_identity(value: Option<&Value>, label: &str) -> Result<()> {
    let map = value
        .and_then(Value::as_object)
        .with_context(|| format!("{label} provenance identity must be an object"))?;
    if !exact_fields(map, &["path", "sha256", "size_bytes"]) {
        bail!("{label} provenance identity fields are incomplete or unexpected");
    }
    let path = map["path"]
        .as_str()
        .filter(|path| path.starts_with('/'))
        .with_context(|| format!("{label} provenance path must be absolute"))?;
    if path.is_empty() {
        bail!("{label} provenance path is empty");
    }
    let digest = map["sha256"]
        .as_str()
        .with_context(|| format!("{label} provenance hash is not a string"))?;
    if !is_lower_hex(digest, 64) {
        bail!("{label} provenance hash is not lowercase SHA-256");
    }
    if map["size_bytes"]
        .as_u64()
        .filter(|size| *size > 0)
        .is_none()
    {
        bail!("{label} provenance size is not positive");
    }
    Ok(())
}

fn expected_layout_sha256() -> Result<String> {
    let mut offsets = Map::new();
    offsets.insert("ri_uuid".to_owned(), json!(0));
    for (index, field) in RUSAGE_V6_U64_FIELDS.iter().enumerate() {
        offsets.insert(
            (*field).to_owned(),
            json!(16 + index * mem::size_of::<u64>()),
        );
    }
    offsets.insert(
        "ri_reserved".to_owned(),
        json!(16 + RUSAGE_V6_U64_FIELDS.len() * mem::size_of::<u64>()),
    );
    canonical_sha256(&Value::Object(offsets))
}

fn validate_local_abi() -> Result<()> {
    let expected_size = 16 + (RUSAGE_V6_U64_FIELDS.len() + 6) * mem::size_of::<u64>();
    if mem::size_of::<RUsageInfoV6>() != expected_size {
        bail!("local rusage_info_v6 size differs from the pinned ABI");
    }
    if expected_layout_sha256()? != STRUCT_LAYOUT_SHA256 {
        bail!("local rusage_info_v6 field offsets differ from the pinned ABI");
    }
    Ok(())
}

fn read_stable_json(path: &Path) -> Result<Value> {
    if fs::symlink_metadata(path)
        .with_context(|| format!("stat process-joule provenance {}", path.display()))?
        .file_type()
        .is_symlink()
    {
        bail!("process-joule provenance must not be a symlink");
    }
    let mut file = File::open(path)
        .with_context(|| format!("open process-joule provenance {}", path.display()))?;
    let before = file.metadata()?;
    if !before.is_file() || before.nlink() != 1 || before.len() == 0 {
        bail!("process-joule provenance must be a non-empty single-link regular file");
    }
    if before.len() > MAX_PROVENANCE_BYTES {
        bail!("process-joule provenance exceeds the bounded input size");
    }
    let mut bytes = Vec::with_capacity(before.len() as usize);
    file.read_to_end(&mut bytes)?;
    let after = file.metadata()?;
    let identity = |row: &fs::Metadata| {
        (
            row.dev(),
            row.ino(),
            row.len(),
            row.mtime(),
            row.mtime_nsec(),
            row.ctime(),
            row.ctime_nsec(),
            row.nlink(),
        )
    };
    if identity(&before) != identity(&after) || bytes.len() as u64 != after.len() {
        bail!("process-joule provenance changed while it was read");
    }
    serde_json::from_slice(&bytes).context("process-joule provenance is not valid JSON")
}

fn validate_provenance(value: &Value) -> Result<()> {
    let map = value
        .as_object()
        .context("process-joule provenance must be an object")?;
    let expected = [
        "schema",
        "library_install_name",
        "proc_libversion_major",
        "proc_libversion_minor",
        "dyld_shared_cache_uuid",
        "os_build",
        "machine",
        "resource_header",
        "libproc_header",
        "collector_contract_sha256",
        "struct_layout_sha256",
        "provenance_sha256",
    ];
    if !exact_fields(map, &expected) {
        bail!("process-joule provenance fields are incomplete or unexpected");
    }
    if map["schema"].as_str() != Some(PROVENANCE_SCHEMA)
        || map["library_install_name"].as_str() != Some(LIBPROC_INSTALL_NAME)
        || map["collector_contract_sha256"].as_str() != Some(COLLECTOR_CONTRACT_SHA256)
        || map["struct_layout_sha256"].as_str() != Some(STRUCT_LAYOUT_SHA256)
    {
        bail!("process-joule provenance differs from the pinned collector ABI");
    }
    if map["proc_libversion_major"].as_u64().is_none()
        || map["proc_libversion_minor"].as_u64().is_none()
    {
        bail!("process-joule provenance libproc version is invalid");
    }
    let cache_uuid = map["dyld_shared_cache_uuid"]
        .as_str()
        .context("process-joule shared-cache UUID is not a string")?;
    if !is_lower_hex(cache_uuid, 32) {
        bail!("process-joule shared-cache UUID is invalid");
    }
    for field in ["os_build", "machine"] {
        if map[field]
            .as_str()
            .filter(|value| !value.is_empty())
            .is_none()
        {
            bail!("process-joule provenance {field} is empty");
        }
    }
    validate_file_identity(map.get("resource_header"), "resource header")?;
    validate_file_identity(map.get("libproc_header"), "libproc header")?;
    let claimed = map["provenance_sha256"]
        .as_str()
        .context("process-joule provenance hash is not a string")?;
    if !is_lower_hex(claimed, 64) {
        bail!("process-joule provenance hash is invalid");
    }
    let mut unstamped = value.clone();
    unstamped
        .as_object_mut()
        .expect("validated provenance object")
        .remove("provenance_sha256");
    if canonical_sha256(&unstamped)? != claimed {
        bail!("process-joule provenance self-hash mismatch");
    }
    Ok(())
}

impl ProcessJouleCollector {
    pub fn from_provenance_file(path: &Path) -> Result<Self> {
        validate_local_abi()?;
        let library_provenance = read_stable_json(path)?;
        validate_provenance(&library_provenance)?;
        Ok(Self {
            pid: std::process::id(),
            library_provenance,
        })
    }

    pub fn snapshot(&self) -> Result<Snapshot> {
        let pid: c_int = self
            .pid
            .try_into()
            .context("probe PID does not fit Darwin pid_t")?;
        // This zero initialization is the documented input state for the C
        // function, and every byte subsequently emitted is in a named field.
        let mut usage: RUsageInfoV6 = unsafe { mem::zeroed() };
        let read_started_at_unix_ns = wall_unix_ns()?;
        let read_started_at_continuous_ns = continuous_ns()?;
        let result = unsafe {
            proc_pid_rusage(
                pid,
                RUSAGE_INFO_V6,
                (&mut usage as *mut RUsageInfoV6).cast::<c_void>(),
            )
        };
        let read_ended_at_continuous_ns = continuous_ns()?;
        let read_ended_at_unix_ns = wall_unix_ns()?;
        if result != 0 {
            bail!(
                "proc_pid_rusage failed for probe PID {}: {}",
                self.pid,
                std::io::Error::last_os_error()
            );
        }

        let process_uuid = usage
            .ri_uuid
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let mut counters = Map::new();
        for (field, value) in RUSAGE_V6_U64_FIELDS.iter().zip(usage.counters) {
            counters.insert((*field).to_owned(), json!(value));
        }
        let value = stamp_value(
            json!({
                "schema": SNAPSHOT_SCHEMA,
                "backend_id": BACKEND_ID,
                "pid": self.pid,
                "ri_uuid": process_uuid,
                "read_started_at_unix_ns": read_started_at_unix_ns,
                "read_ended_at_unix_ns": read_ended_at_unix_ns,
                "read_started_at_continuous_ns": read_started_at_continuous_ns,
                "read_ended_at_continuous_ns": read_ended_at_continuous_ns,
                "counters": counters,
            }),
            "snapshot_sha256",
        )?;
        let snapshot_sha256 = digest_field(&value, "snapshot_sha256")?;
        Ok(Snapshot {
            value,
            pid: self.pid,
            process_uuid,
            process_start_abstime: usage.counters[RI_PROC_START_ABSTIME_INDEX],
            monotone: MONOTONE_FIELD_INDEXES.map(|index| usage.counters[index]),
            energy_nj: usage.counters[RI_ENERGY_NJ_INDEX],
            read_started_at_unix_ns,
            read_ended_at_unix_ns,
            read_started_at_continuous_ns,
            read_ended_at_continuous_ns,
            snapshot_sha256,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn phase_record(
        &self,
        before: &Snapshot,
        after: &Snapshot,
        phase_marker_sha256: &str,
        interval_sha256: &str,
        interval_started_at_unix_ns: u64,
        interval_ended_at_unix_ns: u64,
        interval_started_at_continuous_ns: u64,
        interval_ended_at_continuous_ns: u64,
    ) -> Result<Value> {
        if before.pid != self.pid
            || after.pid != self.pid
            || before.process_uuid != after.process_uuid
            || before.process_start_abstime != after.process_start_abstime
        {
            bail!("PID identity changed between direct process-energy snapshots");
        }
        for (position, index) in MONOTONE_FIELD_INDEXES.iter().enumerate() {
            if after.monotone[position] < before.monotone[position] {
                bail!(
                    "direct process counter wrapped or decreased: {}",
                    RUSAGE_V6_U64_FIELDS[*index]
                );
            }
        }
        let energy_nj_delta = after
            .energy_nj
            .checked_sub(before.energy_nj)
            .context("direct process energy counter wrapped")?;
        if energy_nj_delta == 0 {
            bail!("direct process energy delta is not positive");
        }
        if !(before.read_started_at_unix_ns <= before.read_ended_at_unix_ns
            && before.read_ended_at_unix_ns <= interval_started_at_unix_ns
            && interval_started_at_unix_ns < interval_ended_at_unix_ns
            && interval_ended_at_unix_ns <= after.read_started_at_unix_ns
            && after.read_started_at_unix_ns <= after.read_ended_at_unix_ns
            && before.read_started_at_continuous_ns <= before.read_ended_at_continuous_ns
            && before.read_ended_at_continuous_ns <= interval_started_at_continuous_ns
            && interval_started_at_continuous_ns < interval_ended_at_continuous_ns
            && interval_ended_at_continuous_ns <= after.read_started_at_continuous_ns
            && after.read_started_at_continuous_ns <= after.read_ended_at_continuous_ns)
        {
            bail!("self-sampled counter reads do not bracket the exact phase interval");
        }
        if !is_lower_hex(phase_marker_sha256, 64) || !is_lower_hex(interval_sha256, 64) {
            bail!("phase/interval hash is invalid");
        }
        let source_sample_id = canonical_sha256(&json!({
            "before": before.snapshot_sha256,
            "after": after.snapshot_sha256,
            "phase": phase_marker_sha256,
        }))?;
        stamp_value(
            json!({
                "schema": PHASE_RECORD_SCHEMA,
                "backend_id": BACKEND_ID,
                "boundary_protocol": BOUNDARY_PROTOCOL,
                "source_sample_id": source_sample_id,
                "phase_marker_sha256": phase_marker_sha256,
                "interval_sha256": interval_sha256,
                "process_id": self.pid,
                "process_uuid": before.process_uuid,
                "process_start_abstime": before.process_start_abstime,
                "interval_started_at_unix_ns": interval_started_at_unix_ns,
                "interval_ended_at_unix_ns": interval_ended_at_unix_ns,
                "interval_started_at_continuous_ns": interval_started_at_continuous_ns,
                "interval_ended_at_continuous_ns": interval_ended_at_continuous_ns,
                "self_sampled_by_release_probe": true,
                "before": before.value,
                "after": after.value,
                "energy_nj_delta": energy_nj_delta,
                "energy_j": energy_nj_delta as f64 / 1_000_000_000.0,
                "quantity": "energy",
                "unit": "joule",
                "measurement_scope": "exact-probe-process",
                "attribution": "direct-counter",
                "estimated": false,
                "apportioned": false,
            }),
            "phase_record_sha256",
        )
    }

    pub fn counter_block(self, records: Vec<Value>) -> Result<Value> {
        if records.is_empty() {
            bail!("probe process-energy block has no interval records");
        }
        let mut markers = HashSet::new();
        let mut intervals = HashSet::new();
        let mut sources = HashSet::new();
        for (ordinal, record) in records.iter().enumerate() {
            let marker = digest_field(record, "phase_marker_sha256")?;
            let interval = digest_field(record, "interval_sha256")?;
            let source = digest_field(record, "source_sample_id")?;
            if !markers.insert(marker) || !intervals.insert(interval) || !sources.insert(source) {
                bail!("process-energy record {ordinal} reuses phase/interval/source identity");
            }
        }
        stamp_value(
            json!({
                "schema": PROBE_COUNTERS_SCHEMA,
                "backend_id": BACKEND_ID,
                "collector_contract_sha256": COLLECTOR_CONTRACT_SHA256,
                "probe_pid": self.pid,
                "library_provenance": self.library_provenance,
                "records": records,
            }),
            "counters_sha256",
        )
    }
}
