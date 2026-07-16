#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use strand_quant::metal_rht_probe::{MetalRhtProbe, RhtAxis, METAL_RHT_SOURCE};
use strand_quant::rht::{rht_forward_cols, rht_forward_rows, RhtConfig};
use strand_quant::sha256::sha256;

struct Args {
    receipt: PathBuf,
    staging_root: PathBuf,
    admission_receipt: Option<PathBuf>,
    lease_fd: Option<i32>,
    dispatch: bool,
}

fn parse_args() -> Args {
    let mut receipt = None;
    let mut staging_root = None;
    let mut admission_receipt = None;
    let mut lease_fd = None;
    let mut dispatch = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--receipt" => receipt = Some(args.next().expect("--receipt needs PATH").into()),
            "--staging-root" => {
                staging_root = Some(args.next().expect("--staging-root needs PATH").into())
            }
            "--admission-receipt" => {
                admission_receipt =
                    Some(args.next().expect("--admission-receipt needs PATH").into())
            }
            "--lease-fd" => {
                lease_fd = Some(
                    args.next()
                        .expect("--lease-fd needs FD")
                        .parse()
                        .expect("--lease-fd must be an integer"),
                )
            }
            "--dispatch" => dispatch = true,
            other => panic!("unknown argument {other}"),
        }
    }
    Args {
        receipt: receipt.expect("--receipt is required"),
        staging_root: staging_root.expect("--staging-root is required"),
        admission_receipt,
        lease_fd,
        dispatch,
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").unwrap();
    }
    out
}

fn digest(bytes: &[u8]) -> String {
    hex(&sha256(bytes))
}

fn file_digest(path: &Path) -> String {
    digest(&fs::read(path).expect("read file for SHA-256"))
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn confined_path(root: &Path, path: &Path, must_exist: bool) -> Result<PathBuf, String> {
    if fs::symlink_metadata(root)
        .map_err(|error| format!("inspect staging root: {error}"))?
        .file_type()
        .is_symlink()
    {
        return Err("staging root must not be a symlink".into());
    }
    let root =
        fs::canonicalize(root).map_err(|error| format!("canonicalize staging root: {error}"))?;
    let resolved = if must_exist {
        fs::canonicalize(path).map_err(|error| format!("canonicalize staged input: {error}"))?
    } else {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        fs::canonicalize(parent)
            .map_err(|error| format!("canonicalize receipt parent: {error}"))?
            .join(path.file_name().ok_or("receipt path has no file name")?)
    };
    if !resolved.starts_with(&root) {
        return Err("probe artifact escapes the staging root".into());
    }
    if !must_exist && fs::symlink_metadata(&resolved).is_ok() {
        return Err("refusing to replace an existing probe receipt".into());
    }
    Ok(resolved)
}

#[derive(Clone, Copy)]
struct TopLevelField<'a> {
    raw_value: &'a str,
    pair_start: usize,
    pair_end: usize,
}

fn skip_json_space(bytes: &[u8], mut index: usize) -> usize {
    while bytes
        .get(index)
        .is_some_and(|byte| matches!(byte, b' ' | b'\n' | b'\r' | b'\t'))
    {
        index += 1;
    }
    index
}

fn scan_json_string(bytes: &[u8], start: usize) -> Result<usize, String> {
    if bytes.get(start) != Some(&b'"') {
        return Err("expected a JSON string".into());
    }
    let mut index = start + 1;
    while let Some(byte) = bytes.get(index) {
        match *byte {
            b'"' => return Ok(index + 1),
            b'\\' => {
                index += 1;
                let escape = *bytes.get(index).ok_or("truncated JSON string escape")?;
                if escape == b'u' {
                    for offset in 1..=4 {
                        if !bytes.get(index + offset).is_some_and(u8::is_ascii_hexdigit) {
                            return Err("invalid JSON unicode escape".into());
                        }
                    }
                    index += 4;
                } else if !matches!(
                    escape,
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't'
                ) {
                    return Err("invalid JSON string escape".into());
                }
            }
            0x00..=0x1f => return Err("unescaped JSON control byte".into()),
            _ => {}
        }
        index += 1;
    }
    Err("unterminated JSON string".into())
}

fn scan_json_value(bytes: &[u8], start: usize) -> Result<usize, String> {
    match bytes.get(start) {
        Some(b'"') => scan_json_string(bytes, start),
        Some(b'{') | Some(b'[') => {
            let mut stack = vec![if bytes[start] == b'{' { b'}' } else { b']' }];
            let mut index = start + 1;
            while let Some(byte) = bytes.get(index) {
                match *byte {
                    b'"' => index = scan_json_string(bytes, index)?,
                    b'{' => {
                        stack.push(b'}');
                        index += 1;
                    }
                    b'[' => {
                        stack.push(b']');
                        index += 1;
                    }
                    b'}' | b']' => {
                        if stack.pop() != Some(*byte) {
                            return Err("mismatched JSON container".into());
                        }
                        index += 1;
                        if stack.is_empty() {
                            return Ok(index);
                        }
                    }
                    _ => index += 1,
                }
            }
            Err("unterminated JSON container".into())
        }
        Some(_) => {
            let mut index = start;
            while bytes
                .get(index)
                .is_some_and(|byte| !matches!(byte, b',' | b'}'))
            {
                index += 1;
            }
            let end = (start..index)
                .rev()
                .find(|candidate| !bytes[*candidate].is_ascii_whitespace())
                .map(|candidate| candidate + 1)
                .ok_or("empty JSON value")?;
            Ok(end)
        }
        None => Err("missing JSON value".into()),
    }
}

fn top_level_field<'a>(
    document: &'a str,
    requested: &str,
) -> Result<Option<TopLevelField<'a>>, String> {
    let document = document.strip_suffix('\n').unwrap_or(document);
    let bytes = document.as_bytes();
    if bytes.first() != Some(&b'{') || bytes.last() != Some(&b'}') {
        return Err("admission document root must be a JSON object".into());
    }
    let mut found = None;
    let mut index = skip_json_space(bytes, 1);
    loop {
        if bytes.get(index) == Some(&b'}') {
            if skip_json_space(bytes, index + 1) != bytes.len() {
                return Err("trailing data after admission object".into());
            }
            return Ok(found);
        }
        let pair_start = index;
        let key_end = scan_json_string(bytes, index)?;
        let raw_key = document
            .get(index + 1..key_end - 1)
            .ok_or("admission key is not valid UTF-8")?;
        if raw_key.contains('\\') {
            return Err("escaped top-level admission keys are inadmissible".into());
        }
        index = skip_json_space(bytes, key_end);
        if bytes.get(index) != Some(&b':') {
            return Err("admission key has no value separator".into());
        }
        index = skip_json_space(bytes, index + 1);
        let value_start = index;
        let value_end = scan_json_value(bytes, value_start)?;
        if raw_key == requested {
            if found.is_some() {
                return Err(format!("duplicate top-level admission field: {requested}"));
            }
            found = Some(TopLevelField {
                raw_value: document
                    .get(value_start..value_end)
                    .ok_or("admission value is not valid UTF-8")?,
                pair_start,
                pair_end: value_end,
            });
        }
        index = skip_json_space(bytes, value_end);
        match bytes.get(index) {
            Some(b',') => index = skip_json_space(bytes, index + 1),
            Some(b'}') => {}
            _ => return Err("admission object has an invalid field separator".into()),
        }
    }
}

fn top_level_string(document: &str, field: &str) -> Result<Option<String>, String> {
    let Some(value) = top_level_field(document, field)? else {
        return Ok(None);
    };
    if !value.raw_value.starts_with('"')
        || !value.raw_value.ends_with('"')
        || value.raw_value[1..value.raw_value.len() - 1].contains('\\')
    {
        return Err(format!(
            "top-level admission field {field} is not a plain string"
        ));
    }
    Ok(Some(
        value.raw_value[1..value.raw_value.len() - 1].to_string(),
    ))
}

fn top_level_u128(document: &str, field: &str) -> Result<Option<u128>, String> {
    let Some(value) = top_level_field(document, field)? else {
        return Ok(None);
    };
    if value.raw_value.is_empty() || !value.raw_value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!(
            "top-level admission field {field} is not an unsigned integer"
        ));
    }
    value
        .raw_value
        .parse()
        .map(Some)
        .map_err(|_| format!("top-level admission field {field} overflows u128"))
}

fn top_level_bool(document: &str, field: &str) -> Result<Option<bool>, String> {
    let Some(value) = top_level_field(document, field)? else {
        return Ok(None);
    };
    match value.raw_value {
        "true" => Ok(Some(true)),
        "false" => Ok(Some(false)),
        _ => Err(format!("top-level admission field {field} is not boolean")),
    }
}

fn verify_document_self_hash(document: &str) -> Result<(), String> {
    let document = document.strip_suffix('\n').unwrap_or(document);
    let field = top_level_field(document, "document_sha256")?
        .ok_or("admission document self-hash field is absent")?;
    let value = top_level_string(document, "document_sha256")?
        .ok_or("admission document self-hash field is absent")?;
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("admission document self-hash is not lowercase SHA-256".into());
    }
    let bytes = document.as_bytes();
    let (remove_start, remove_end) =
        if field.pair_start > 0 && bytes.get(field.pair_start - 1) == Some(&b',') {
            (field.pair_start - 1, field.pair_end)
        } else if bytes.get(field.pair_end) == Some(&b',') {
            (field.pair_start, field.pair_end + 1)
        } else {
            (field.pair_start, field.pair_end)
        };
    let mut body = String::with_capacity(document.len() - (remove_end - remove_start));
    body.push_str(&document[..remove_start]);
    body.push_str(&document[remove_end..]);
    if digest(body.as_bytes()) != value {
        return Err("admission document self-hash differs".into());
    }
    Ok(())
}

fn verify_bound_source(
    document: &str,
    path_field: &str,
    sha_field: &str,
    expected: &Path,
) -> Result<(), String> {
    let recorded = top_level_string(document, path_field)?
        .ok_or_else(|| format!("admission {path_field} is missing"))?;
    let expected = fs::canonicalize(expected)
        .map_err(|error| format!("resolve expected admission source: {error}"))?;
    let expected_digest = file_digest(&expected);
    if fs::canonicalize(&recorded)
        .map_err(|error| format!("resolve recorded admission source: {error}"))?
        != expected
        || top_level_string(document, sha_field)?.as_deref() != Some(expected_digest.as_str())
    {
        return Err(format!(
            "admission {path_field}/{sha_field} binding differs"
        ));
    }
    Ok(())
}

fn verify_lease_owner(
    fd: i32,
    path: &str,
    expected_device: u128,
    expected_inode: u128,
) -> Result<(), String> {
    const SCRIPT: &str = r#"import fcntl, os, stat, sys
fd = int(sys.argv[1]); path = sys.argv[2]; expected_device = int(sys.argv[3]); expected_inode = int(sys.argv[4])
try:
    owned = os.fstat(fd); named = os.stat(path)
except OSError:
    raise SystemExit(5)
if not stat.S_ISREG(owned.st_mode) or (owned.st_dev, owned.st_ino) != (named.st_dev, named.st_ino) or (owned.st_dev, owned.st_ino) != (expected_device, expected_inode):
    raise SystemExit(2)
fresh = os.open(path, os.O_RDWR)
try:
    fcntl.flock(fresh, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    pass
else:
    fcntl.flock(fresh, fcntl.LOCK_UN)
    raise SystemExit(3)
finally:
    os.close(fresh)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit(4)
"#;
    let status = Command::new("/usr/bin/python3")
        .args([
            "-c",
            SCRIPT,
            &fd.to_string(),
            path,
            &expected_device.to_string(),
            &expected_inode.to_string(),
        ])
        .status()
        .map_err(|error| format!("run lease ownership verifier: {error}"))?;
    if !status.success() {
        return Err(format!(
            "inherited heavy lease is not exclusively owned ({status})"
        ));
    }
    Ok(())
}

fn verify_dispatch_admission(args: &Args) -> Result<String, String> {
    let admission = args
        .admission_receipt
        .as_deref()
        .ok_or("--dispatch requires --admission-receipt")?;
    let fd = args.lease_fd.ok_or("--dispatch requires --lease-fd")?;
    if fd < 0 {
        return Err("--lease-fd must be nonnegative".into());
    }
    let admission = confined_path(&args.staging_root, admission, true)?;
    let modified = fs::metadata(&admission)
        .and_then(|metadata| metadata.modified())
        .map_err(|error| format!("inspect admission receipt time: {error}"))?;
    if SystemTime::now()
        .duration_since(modified)
        .map_err(|_| "admission receipt modification time is in the future")?
        > std::time::Duration::from_secs(60)
    {
        return Err("resource admission receipt file is stale".into());
    }
    let document = fs::read_to_string(&admission)
        .map_err(|error| format!("read resource admission receipt: {error}"))?;
    verify_document_self_hash(&document)?;
    if top_level_string(&document, "schema")?.as_deref()
        != Some("hawking.strand.native-probe-admission.v1")
        || top_level_string(&document, "generator_schema")?.as_deref()
            != Some("hawking.strand.native-probe-launcher.v1")
        || top_level_string(&document, "status")?.as_deref() != Some("admitted")
        || top_level_u128(&document, "active_heavy_owner_count")? != Some(0)
        || top_level_u128(&document, "lease_fd")? != Some(fd as u128)
        || top_level_bool(&document, "owners_rechecked_under_lease")? != Some(true)
        || top_level_bool(&document, "resource_health_ok")? != Some(true)
    {
        return Err("resource admission receipt is not owner-free and lease-bound".into());
    }
    let digest = file_digest(&admission);
    if std::env::var("HAWKING_NATIVE_PROBE_ADMITTED").as_deref() != Ok("1")
        || std::env::var("HAWKING_NATIVE_PROBE_ADMISSION_SHA256").as_deref() != Ok(digest.as_str())
    {
        return Err("admission receipt environment authority differs".into());
    }
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .map_err(|error| format!("resolve repository root: {error}"))?;
    verify_bound_source(
        &document,
        "generator_path",
        "generator_sha256",
        &repository.join("vendor/strand-quant/tools/native_probe.py"),
    )?;
    verify_bound_source(
        &document,
        "owner_pattern_source_path",
        "owner_pattern_source_sha256",
        &repository.join("tools/condense/doctor_v5_single_device_sprint_audit.py"),
    )?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "clock before epoch")?
        .as_nanos();
    let generated =
        top_level_u128(&document, "generated_unix_ns")?.ok_or("admission time missing")?;
    let expires =
        top_level_u128(&document, "expires_unix_ns")?.ok_or("admission expiry missing")?;
    if generated > now || now.saturating_sub(generated) > 60_000_000_000 || expires < now {
        return Err("resource admission receipt is stale or expired".into());
    }
    let lease_path = top_level_string(&document, "lease_path")?.ok_or("lease path missing")?;
    let lease_device = top_level_u128(&document, "lease_device")?.ok_or("lease device missing")?;
    let lease_inode = top_level_u128(&document, "lease_inode")?.ok_or("lease inode missing")?;
    let expected_lease = repository.join("reports/cron/studio_heavy.lock");
    if fs::canonicalize(&lease_path)
        .map_err(|error| format!("resolve heavy lease path: {error}"))?
        != fs::canonicalize(&expected_lease)
            .map_err(|error| format!("resolve expected heavy lease: {error}"))?
    {
        return Err("admission receipt names a non-Studio heavy lease".into());
    }
    let staging_in_receipt =
        top_level_string(&document, "staging_root")?.ok_or("staging root missing")?;
    if fs::canonicalize(&staging_in_receipt)
        .map_err(|error| format!("resolve admitted staging root: {error}"))?
        != fs::canonicalize(&args.staging_root)
            .map_err(|error| format!("resolve requested staging root: {error}"))?
    {
        return Err("admission receipt is for a different staging root".into());
    }
    let admitted_probe =
        top_level_string(&document, "probe_sha256")?.ok_or("probe hash missing")?;
    let executable =
        std::env::current_exe().map_err(|error| format!("current executable: {error}"))?;
    if admitted_probe != file_digest(&executable) {
        return Err("admission receipt is for a different probe binary".into());
    }
    verify_lease_owner(fd, &lease_path, lease_device, lease_inode)?;
    Ok(digest)
}

fn fixture() -> Vec<f32> {
    let mut state = 0xA117_5EED_CAFE_BABEu64;
    (0..1024)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = ((state >> 32) as u32) as f32 / u32::MAX as f32;
            (unit * 2.0 - 1.0) * (1.0 + (index % 19) as f32 / 16.0)
        })
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

fn exact_bits(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn main() {
    let args = parse_args();
    let receipt_path = confined_path(&args.staging_root, &args.receipt, false)
        .unwrap_or_else(|error| panic!("probe receipt confinement: {error}"));
    let admission_sha256 = if args.dispatch {
        Some(
            verify_dispatch_admission(&args)
                .unwrap_or_else(|error| panic!("physical dispatch admission: {error}")),
        )
    } else {
        None
    };
    let executable = std::env::current_exe().expect("current executable");
    let input = fixture();
    let seed = 0xBEEF_F00D_0000_2222;
    let cfg = RhtConfig::from_seed(seed);
    let cpu_rows = rht_forward_rows(&input, &cfg, 256);
    let cpu_cols = rht_forward_cols(&input, &cfg, 256);
    let input_sha = digest(&f32_bytes(&input));
    let cpu_rows_sha = digest(&f32_bytes(&cpu_rows));
    let cpu_cols_sha = digest(&f32_bytes(&cpu_cols));
    let generated_unix_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_nanos();

    let (status, device, rows_exact, cols_exact, dispatch_wall_ns) = if args.dispatch {
        let started = Instant::now();
        let probe = MetalRhtProbe::new().unwrap_or_else(|error| panic!("Metal RHT probe: {error}"));
        let rows = probe
            .forward(&input, seed, 256, RhtAxis::Rows)
            .unwrap_or_else(|error| panic!("Metal row RHT: {error}"));
        let cols = probe
            .forward(&input, seed, 256, RhtAxis::Cols)
            .unwrap_or_else(|error| panic!("Metal col RHT: {error}"));
        let rows_exact = exact_bits(&cpu_rows, &rows);
        let cols_exact = exact_bits(&cpu_cols, &cols);
        (
            if rows_exact && cols_exact {
                "pass"
            } else {
                "fail"
            },
            probe.device_name(),
            Some(rows_exact),
            Some(cols_exact),
            Some(started.elapsed().as_nanos()),
        )
    } else {
        ("compiled_not_dispatched", String::new(), None, None, None)
    };

    let option_bool = |value: Option<bool>| match value {
        Some(true) => "true",
        Some(false) => "false",
        None => "null",
    };
    let option_u128 = |value: Option<u128>| {
        value
            .map(|value| value.to_string())
            .unwrap_or_else(|| "null".into())
    };
    let receipt = format!(
        "{{\n  \"schema\": \"hawking.strand.metal-rht-parity.v1\",\n  \"status\": \"{status}\",\n  \"scope\": \"synthetic_only\",\n  \"generated_unix_ns\": {generated_unix_ns},\n  \"dispatch_requested\": {},\n  \"admission_receipt_sha256\": {},\n  \"program\": \"{}\",\n  \"program_sha256\": \"{}\",\n  \"shader_source_sha256\": \"{}\",\n  \"device\": \"{}\",\n  \"invocation_identity\": \"fixture=f32:4x256;seed=0xBEEFF00D00002222;axes=rows,cols;strict=f32_bits\",\n  \"input_sha256\": \"{input_sha}\",\n  \"cpu_rows_sha256\": \"{cpu_rows_sha}\",\n  \"cpu_cols_sha256\": \"{cpu_cols_sha}\",\n  \"rows_exact_bits\": {},\n  \"cols_exact_bits\": {},\n  \"measurements\": {{\"end_to_end_wall_ns\": {}, \"read_decode_ns\": 0, \"rht_preprocess_ns\": {}, \"encode_ns\": null, \"finalize_write_ns\": null, \"cpu_time_ns\": null, \"gpu_time_ns\": null, \"peak_rss_bytes\": null, \"swap_delta_bytes\": null, \"scratch_peak_bytes\": 1024, \"disk_read_bytes\": 0, \"disk_write_bytes\": null, \"thermal_start\": null, \"thermal_end\": null, \"phase_instrumentation_complete\": false}},\n  \"input_bundle_sha256\": \"{input_sha}\",\n  \"output_bundle_sha256\": null,\n  \"scientific_receipt_bundle_sha256\": null,\n  \"component_speedup_is_eta_evidence\": false,\n  \"runtime_activation\": false,\n  \"production_promotion_allowed\": false\n}}\n",
        args.dispatch,
        admission_sha256
            .as_ref()
            .map(|value| format!("\"{value}\""))
            .unwrap_or_else(|| "null".into()),
        json_escape(&executable.display().to_string()),
        file_digest(&executable),
        digest(METAL_RHT_SOURCE.as_bytes()),
        json_escape(&device),
        option_bool(rows_exact),
        option_bool(cols_exact),
        option_u128(dispatch_wall_ns),
        option_u128(dispatch_wall_ns),
    );
    strand_quant::native_io::write_preallocated(&receipt_path, receipt.as_bytes())
        .expect("atomically write Metal RHT receipt");
    print!("{receipt}");
    if status == "fail" {
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_fields_parse_exactly() {
        let document = r#"{"schema":"admitted","active_heavy_owner_count":0,"expires_unix_ns":42,"resource_health_ok":true}"#;
        assert_eq!(
            top_level_string(document, "schema").unwrap().as_deref(),
            Some("admitted")
        );
        assert_eq!(
            top_level_u128(document, "active_heavy_owner_count").unwrap(),
            Some(0)
        );
        assert_eq!(
            top_level_u128(document, "expires_unix_ns").unwrap(),
            Some(42)
        );
        assert_eq!(
            top_level_bool(document, "resource_health_ok").unwrap(),
            Some(true)
        );
        assert_eq!(top_level_u128(document, "missing").unwrap(), None);
    }

    #[test]
    fn actual_launcher_shape_uses_only_top_level_authority() {
        let body = r#"{"active_heavy_owner_count":0,"resource_health":{"snapshot":{"schema":"hawking.studio_resource_snapshot.v1"}},"schema":"hawking.strand.native-probe-admission.v1","status":"admitted"}"#;
        let sealed = format!(
            "{{\"active_heavy_owner_count\":0,\"document_sha256\":\"{}\",\"resource_health\":{{\"snapshot\":{{\"schema\":\"hawking.studio_resource_snapshot.v1\"}}}},\"schema\":\"hawking.strand.native-probe-admission.v1\",\"status\":\"admitted\"}}\n",
            digest(body.as_bytes())
        );
        assert_eq!(
            top_level_string(&sealed, "schema").unwrap().as_deref(),
            Some("hawking.strand.native-probe-admission.v1")
        );
        assert!(verify_document_self_hash(&sealed).is_ok());
        let duplicate = sealed.replace(
            "\"status\":\"admitted\"",
            "\"schema\":\"forged\",\"status\":\"admitted\"",
        );
        assert!(top_level_string(&duplicate, "schema").is_err());
    }

    #[test]
    fn admission_document_self_hash_is_mandatory_and_exact() {
        let body = r#"{"active_heavy_owner_count":0,"schema":"test"}"#;
        let sealed = format!(
            "{{\"active_heavy_owner_count\":0,\"document_sha256\":\"{}\",\"schema\":\"test\"}}\n",
            digest(body.as_bytes())
        );
        assert!(verify_document_self_hash(&sealed).is_ok());
        assert!(verify_document_self_hash(
            &sealed.replace("\"schema\":\"test\"", "\"schema\":\"tampered\"")
        )
        .is_err());
        assert!(verify_document_self_hash(body).is_err());
    }

    #[test]
    fn receipt_path_cannot_escape_staging_root() {
        let root =
            std::env::temp_dir().join(format!("strand-metal-probe-stage-{}", std::process::id()));
        let outside = std::env::temp_dir().join(format!(
            "strand-metal-probe-outside-{}.json",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();
        let inside = root.join("receipt.json");
        assert_eq!(
            confined_path(&root, &inside, false).unwrap(),
            fs::canonicalize(&root).unwrap().join("receipt.json")
        );
        assert!(confined_path(&root, &outside, false).is_err());
        fs::remove_dir(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn unheld_descriptor_cannot_self_admit_during_verification() {
        use std::fs::OpenOptions;
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::MetadataExt;

        let path = std::env::temp_dir().join(format!(
            "strand-metal-probe-unheld-{}.lock",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        let metadata = file.metadata().unwrap();
        assert!(verify_lease_owner(
            file.as_raw_fd(),
            path.to_str().unwrap(),
            metadata.dev() as u128,
            metadata.ino() as u128,
        )
        .is_err());
        drop(file);
        fs::remove_file(path).unwrap();
    }
}
