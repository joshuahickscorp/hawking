//! Q80 broad all-layer activation route+hidden capture — readiness gate.
//!
//! Mirrors the Q30 all-layer capture design
//! (`ascension_qwen30_broad_activation_all_layer_route_capture`) but **refuses
//! to execute** until the measured blockers are cleared:
//!
//! 1. GQA full-layer same-runtime encode ready for layers 3,7,...,47
//!    (authority: `QWEN80_MULTI_LAYER_GQA_ENCODE_GAP_*`, ready_count=0/12).
//! 2. Multi-token sequential state path for broad prompts (existing multi-layer
//!    same-runtime is single-token component parity only).
//!
//! This binary is intentionally a sealed refusal / design anchor so the lane
//! does not invent activations or fit on L0-only. When the blockers clear,
//! extend this example with the Q30 stratified-subsample writer on top of the
//! multi-layer same-runtime encode at layer_count=48.
//!
//! Diagnostic only. No coherence, HCLI, TPS, or capability claim. Does not
//! start the production server / watcher / HCLI adapter. Does not take a
//! Metal lease (refusal is pre-lease).

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

const RESULT_SCHEMA: &str =
    "hawking.ascension.qwen80_broad_activation_all_layer_route_capture_result.v1";
const REFUSAL_SCHEMA: &str =
    "hawking.ascension.qwen80_broad_activation_all_layer_route_capture_refusal.v1";
const INPUT_SCHEMA: &str =
    "hawking.ascension.qwen80_broad_activation_all_layer_route_capture_input.v1";
const CAPTURE_PROTOCOL_REVISION: &str =
    "q80-all-layer-route-hidden-capture-stratified-subsample-v1-REFUSAL-UNTIL-GQA";
const QWEN80_LAYERS: usize = 48;
const QWEN80_HIDDEN: usize = 2048;
const GQA_LAYERS: [usize; 12] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47];
const DELTANET_PREFIX_READY: usize = 3;

struct Arguments {
    input_json: Option<PathBuf>,
    output_dir: PathBuf,
    schedule_authority: Option<PathBuf>,
    gqa_gap: Option<PathBuf>,
    /// Force-run even when GQA is not ready — only allowed with --diagnostic-prefix-only
    /// and still does not execute Metal (prints design only).
    diagnostic_prefix_only: bool,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_broad_activation_all_layer_route_capture \
        --output-dir ABSOLUTE_PATH \
        [--input-json ABSOLUTE_PATH] \
        [--schedule-authority ABSOLUTE_PATH] \
        [--gqa-gap ABSOLUTE_PATH] \
        [--diagnostic-prefix-only]"
}

fn absolute(path: PathBuf, flag: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{flag} must be an absolute path; {}", usage()));
    }
    Ok(path)
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut input_json = None;
    let mut output_dir = None;
    let mut schedule_authority = None;
    let mut gqa_gap = None;
    let mut diagnostic_prefix_only = false;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--input-json" => {
                let v = args
                    .next()
                    .ok_or_else(|| format!("missing value for --input-json; {}", usage()))?;
                input_json = Some(absolute(PathBuf::from(v), "--input-json")?);
            }
            "--output-dir" => {
                let v = args
                    .next()
                    .ok_or_else(|| format!("missing value for --output-dir; {}", usage()))?;
                output_dir = Some(absolute(PathBuf::from(v), "--output-dir")?);
            }
            "--schedule-authority" => {
                let v = args
                    .next()
                    .ok_or_else(|| format!("missing value for --schedule-authority; {}", usage()))?;
                schedule_authority = Some(absolute(PathBuf::from(v), "--schedule-authority")?);
            }
            "--gqa-gap" => {
                let v = args
                    .next()
                    .ok_or_else(|| format!("missing value for --gqa-gap; {}", usage()))?;
                gqa_gap = Some(absolute(PathBuf::from(v), "--gqa-gap")?);
            }
            "--diagnostic-prefix-only" => diagnostic_prefix_only = true,
            "--help" | "-h" => {
                println!("{}", usage());
                process::exit(0);
            }
            other => return Err(format!("unknown flag {other}; {}", usage())),
        }
    }
    let output_dir = output_dir.ok_or_else(|| format!("missing --output-dir; {}", usage()))?;
    Ok(Arguments {
        input_json,
        output_dir,
        schedule_authority,
        gqa_gap,
        diagnostic_prefix_only,
    })
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("json {}: {e}", path.display()))
}

fn gqa_ready_count(schedule: Option<&Value>) -> Option<usize> {
    schedule
        .and_then(|s| s.get("aggregate"))
        .and_then(|a| a.get("same_runtime_gqa_encode_ready_layer_count"))
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(value).map_err(|e| format!("serialize: {e}"))?;
    fs::write(path, format!("{text}\n")).map_err(|e| format!("write {}: {e}", path.display()))
}

fn main() {
    let args = match parse_arguments() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("ascension_qwen80_broad_activation_all_layer_route_capture: {e}");
            process::exit(2);
        }
    };

    let schedule = match &args.schedule_authority {
        Some(p) => match read_json(p) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("{e}");
                process::exit(2);
            }
        },
        None => None,
    };
    let gqa_gap = match &args.gqa_gap {
        Some(p) => match read_json(p) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("{e}");
                process::exit(2);
            }
        },
        None => None,
    };

    let gqa_ready = gqa_ready_count(schedule.as_ref()).unwrap_or(0);
    let all_layer_ready = gqa_ready == GQA_LAYERS.len();

    let input_binding = match &args.input_json {
        Some(p) => {
            if !p.is_file() {
                eprintln!("input-json missing: {}", p.display());
                process::exit(2);
            }
            let doc = match read_json(p) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{e}");
                    process::exit(2);
                }
            };
            let schema = doc.get("schema").and_then(|s| s.as_str()).unwrap_or("");
            if schema != INPUT_SCHEMA && !schema.contains("broad_activation") {
                eprintln!(
                    "unexpected input schema {schema:?}; expected {INPUT_SCHEMA} or broad_activation family"
                );
                process::exit(2);
            }
            json!({
                "path": p.display().to_string(),
                "sha256": sha256_file(p).unwrap_or_default(),
                "schema": schema,
                "probe_count": doc.get("corpus_summary")
                    .and_then(|c| c.get("probe_count"))
                    .cloned()
                    .unwrap_or(Value::Null),
                "total_tokens": doc.get("corpus_summary")
                    .and_then(|c| c.get("total_tokens"))
                    .cloned()
                    .unwrap_or(Value::Null),
            })
        }
        None => Value::Null,
    };

    if !all_layer_ready {
        let exact_missing = gqa_gap
            .as_ref()
            .and_then(|g| g.get("exact_missing_input"))
            .and_then(|s| s.as_str())
            .unwrap_or(
                "same-runtime full-layer encode for the GQA mixer, with caller-owned \
                 gqa_key_cache/gqa_value_cache state slots and their rollback buffers",
            );

        let refusal = json!({
            "schema": REFUSAL_SCHEMA,
            "status": "REFUSED_ALL_LAYER_ACTIVATION_CAPTURE_BLOCKED",
            "result_schema_when_ready": RESULT_SCHEMA,
            "capture_protocol_revision": CAPTURE_PROTOCOL_REVISION,
            "recorded_at": chrono_like_now(),
            "model": {
                "key": "qwen80",
                "layers": QWEN80_LAYERS,
                "hidden": QWEN80_HIDDEN,
            },
            "gqa_encode_ready_layer_count": gqa_ready,
            "gqa_layers": GQA_LAYERS.as_slice(),
            "deltanet_prefix_ready_layer_count": DELTANET_PREFIX_READY,
            "exact_missing": [
                {
                    "id": "gqa_full_layer_same_runtime_encode",
                    "description": exact_missing,
                    "ready_count": gqa_ready,
                    "required_count": GQA_LAYERS.len(),
                },
                {
                    "id": "multi_token_sequential_state_for_capture",
                    "description": "existing multi-layer same-runtime path is single-token component parity; broad capture needs sequential state across prompt tokens",
                }
            ],
            "input_binding": input_binding,
            "bounded_storage_when_ready": {
                "full_route_membership_all_tokens_all_layers": true,
                "raw_hidden_strategy": "stratified_token_subsample",
                "default_max_hidden_tokens_per_layer": 1024,
                "hidden_bytes_at_default": 48 * 1024 * QWEN80_HIDDEN * 4,
            },
            "claim_boundary": {
                "metal_lease_not_taken": true,
                "server_not_started": true,
                "no_activations_invented": true,
                "no_family_fit": true,
                "no_coherence_claim": true,
                "fitting_on_layer0_only_refused": true,
                "diagnostic_prefix_only_flag": args.diagnostic_prefix_only,
            },
            "verdict": "ALL_LAYER_ACTIVATION_CAPTURE_NOT_YET_POSSIBLE",
            "verdict_detail": format!(
                "GQA full-layer same-runtime encode ready_count={gqa_ready}/12. \
                 Device multi-layer earned prefix is L0..L2 only. Refusing all-layer \
                 activation capture rather than packing a partial-layer candidate."
            ),
            "next": {
                "when_gqa_ready": "extend this binary with multi-token multi-layer encode at layer_count=48 and stratified hidden writer (Q30 pattern)",
                "null_first_operator": "lab/operators/q80_activation_null_first_report.py",
                "repack_operator": "lab/operators/ascension_qwen80_activation_weighted_svd_repack.py",
            }
        });

        let out = args.output_dir.join("capture-refusal.json");
        if let Err(e) = write_json(&out, &refusal) {
            eprintln!("{e}");
            process::exit(2);
        }
        // Also write a capture-result.json that null/repack tools will refuse honestly.
        let marker = json!({
            "schema": RESULT_SCHEMA,
            "status": "REFUSED_NOT_CAPTURED",
            "refusal_path": out.display().to_string(),
            "all_layer_activation_capture": false,
            "capture_summary": {
                "all_layer_activation_capture": false,
                "refused": true,
            },
            "probes": [],
            "claim_boundary": {
                "refused_before_execution": true,
                "no_activations_written": true,
            }
        });
        let marker_path = args.output_dir.join("capture-result.json");
        if let Err(e) = write_json(&marker_path, &marker) {
            eprintln!("{e}");
            process::exit(2);
        }

        eprintln!(
            "REFUSED all-layer Q80 activation capture: GQA encode ready {}/12. Wrote {}",
            gqa_ready,
            out.display()
        );
        process::exit(3);
    }

    // When GQA is ready this branch will be replaced with real capture logic.
    eprintln!(
        "GQA encode appears ready ({gqa_ready}/12) but multi-token capture body is not yet wired in this binary. Refuse to invent a partial implementation."
    );
    process::exit(4);
}

/// Minimal UTC timestamp without pulling chrono into the example if unavailable.
fn chrono_like_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix:{secs}")
}
