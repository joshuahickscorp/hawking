//! First real DSV4F activation-X capture against the sealed 43-layer source.
//!
//! Drives the existing streamed BOS forward (`execute_one_layer_with_x`) and
//! the existing first-N writer. Does **not** invent a new retention policy
//! or a new token graph. Honesty: each token is an independent position-0
//! (seqlen-1, window-only sparse attention) embed through all 43 layers —
//! not a multi-token sequence, and not the full ratio-4/128 compressed
//! indexer graph.
//!
//! Layer-major: for each layer, stream the weights once, run a token tile as
//! GEMMs (dense organs + per-expert grouped routed GEMMs), retain, flush,
//! free, then checkpoint HC + layer metadata so the run can resume.
//!
//! `--no-batch` keeps the original per-token GEMV loop for parity. `--resume`
//! still accepts a checkpoint written by that older driver.
//!
//! ```text
//! cargo run --profile release-fast -p hawking-core --example dsv4f_activation_x_source_capture -- \
//!   --output-dir /abs/path/out --tokens 64 --metal
//! ```

use hawking_core::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, PINNED_REPOSITORY, PINNED_REVISION,
};
use hawking_core::gravity_deepseek_v4_layer0_prefix::{HC_FLAT_WIDTH, HIDDEN_SIZE};
use hawking_core::gravity_deepseek_v4_layer_source_anchors::{
    verify_deepseek_v4_layer_source_anchors, DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT,
};
use hawking_core::gravity_deepseek_v4_runtime_spine::DSV4F_VOCAB_SIZE;
use hawking_core::gravity_deepseek_v4_streamed_forward::{
    default_token_tile, discover_sealed_dsv4f_artifact, execute_layer_tile, execute_one_layer_with_x,
    load_token_embed_hc, peak_rss_bytes, prepare_sealed_admission_root, OperatorProfile,
    ResidentLedger, StreamedLayerCapture, DECLARED_PEAK_RSS_BOUND_BYTES,
    DECLARED_WEIGHT_RESIDENT_BOUND_BYTES, STREAMED_EXECUTION_PATH, STREAMED_EXECUTION_PATH_METAL,
};
use hawking_core::gravity_deepseek_v4_streamed_native::StreamedNativeSession;
use hawking_core::model::dsv4f_activation_capture::{
    append_retained_layer_captures, build_token_index, emit_capture_result, empty_captures,
    format_capture_progress, n_fit_distribution, release_layer_retained_hiddens,
    resident_retained_hidden_bytes, write_json_new, write_layer_retained_rows, CaptureGeometry,
    CaptureSet, FlushBook, HiddenWrite, LayerActivationBatch, LayerTokenCapture, TokenFlushRecord,
    DEFAULT_MAX_HIDDEN_TOKENS_PER_EXPERT, DEFAULT_ROW_THRESHOLD, RESULT_SCHEMA,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

const CKPT_SCHEMA: &str = "hawking.dsv4f.activation_x_source_capture_ckpt.v1";
const CKPT_NAME: &str = "source_capture.ckpt.json";
const PROBE_ID: &str = "vocab_bos_v1";

struct Arguments {
    artifact: PathBuf,
    output_dir: PathBuf,
    tokens: usize,
    max_layer: usize,
    max_per_expert: usize,
    row_threshold: usize,
    use_metal: bool,
    resume: bool,
    batch: bool,
    token_tile: usize,
}

fn usage() -> &'static str {
    "usage: dsv4f_activation_x_source_capture --output-dir ABSOLUTE_PATH \\\n\
     \x20  [--artifact PATH] [--tokens N] [--max-layer N] \\\n\
     \x20  [--max-hidden-tokens-per-expert N] [--row-threshold N] \\\n\
     \x20  [--token-tile N] [--batch] [--no-batch] \\\n\
     \x20  [--metal] [--cpu] [--resume]"
}

fn fail(detail: impl AsRef<str>) -> ! {
    eprintln!(
        "dsv4f activation-x source capture refused: {}",
        detail.as_ref()
    );
    process::exit(2);
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("{flag} must be an unsigned decimal integer; {}", usage()))
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut artifact = None;
    let mut output_dir = None;
    let mut tokens = 64usize;
    let mut max_layer = DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT - 1;
    let mut max_per_expert = DEFAULT_MAX_HIDDEN_TOKENS_PER_EXPERT;
    let mut row_threshold = DEFAULT_ROW_THRESHOLD;
    let mut use_metal = true;
    let mut resume = false;
    let mut batch = true;
    let mut token_tile = 0usize;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--artifact" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for --artifact; {}", usage()))?;
                artifact = Some(PathBuf::from(value));
            }
            "--output-dir" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for --output-dir; {}", usage()))?;
                if output_dir.replace(PathBuf::from(value)).is_some() {
                    return Err("--output-dir supplied more than once".into());
                }
            }
            "--tokens" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for --tokens; {}", usage()))?;
                tokens = parse_usize(&value, "--tokens")?;
            }
            "--max-layer" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for --max-layer; {}", usage()))?;
                max_layer = parse_usize(&value, "--max-layer")?;
            }
            "--max-hidden-tokens-per-expert" => {
                let value = args.next().ok_or_else(|| {
                    format!(
                        "missing value for --max-hidden-tokens-per-expert; {}",
                        usage()
                    )
                })?;
                max_per_expert = parse_usize(&value, "--max-hidden-tokens-per-expert")?;
            }
            "--row-threshold" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for --row-threshold; {}", usage()))?;
                row_threshold = parse_usize(&value, "--row-threshold")?;
            }
            "--token-tile" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for --token-tile; {}", usage()))?;
                token_tile = parse_usize(&value, "--token-tile")?;
            }
            "--batch" => batch = true,
            "--no-batch" => batch = false,
            "--metal" => use_metal = true,
            "--cpu" => use_metal = false,
            "--resume" => resume = true,
            "--help" | "-h" => {
                println!("{}", usage());
                process::exit(0);
            }
            other => return Err(format!("unknown flag {other:?}; {}", usage())),
        }
    }
    let output_dir = output_dir.ok_or_else(|| format!("missing --output-dir; {}", usage()))?;
    if !output_dir.is_absolute() {
        return Err("--output-dir must be an absolute path".into());
    }
    if tokens == 0 {
        return Err("--tokens must be > 0".into());
    }
    if max_layer >= DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT {
        return Err("--max-layer must be in 0..42".into());
    }
    if max_per_expert == 0 {
        return Err("--max-hidden-tokens-per-expert must be > 0".into());
    }
    let artifact = match artifact {
        Some(path) => path,
        None => discover_sealed_dsv4f_artifact()
            .ok_or_else(|| "no --artifact and no sealed full-43-layer-stream.gravity found".to_string())?,
    };
    Ok(Arguments {
        artifact,
        output_dir,
        tokens,
        max_layer,
        max_per_expert,
        row_threshold,
        use_metal,
        resume,
        batch,
        token_tile,
    })
}

fn token_ids_for_budget(n: usize) -> Vec<u32> {
    // Independent position-0 embeds. Token 0 is the sealed BOS id; the rest
    // are consecutive vocab ids so the corpus is deterministic and replayable.
    (0..n)
        .map(|i| {
            let id = i as u64;
            if id >= DSV4F_VOCAB_SIZE as u64 {
                fail(format!("token budget {n} exceeds vocab {DSV4F_VOCAB_SIZE}"));
            }
            id as u32
        })
        .collect()
}

fn sha256_u16(bits: &[u16]) -> String {
    let mut digest = Sha256::new();
    for &value in bits {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn write_hc(path: &Path, hc: &[u16]) -> Result<(), String> {
    if hc.len() != HC_FLAT_WIDTH {
        return Err(format!("HC width {} != {HC_FLAT_WIDTH}", hc.len()));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("hc dir: {e}"))?;
    }
    let tmp = path.with_extension("u16le.tmp");
    {
        let mut file = File::create(&tmp).map_err(|e| format!("hc create: {e}"))?;
        for &value in hc {
            file.write_all(&value.to_le_bytes())
                .map_err(|e| format!("hc write: {e}"))?;
        }
        file.flush().map_err(|e| format!("hc flush: {e}"))?;
    }
    fs::rename(&tmp, path).map_err(|e| format!("hc rename: {e}"))?;
    Ok(())
}

fn read_hc(path: &Path) -> Result<Vec<u16>, String> {
    let mut raw = Vec::new();
    File::open(path)
        .map_err(|e| format!("hc open {}: {e}", path.display()))?
        .read_to_end(&mut raw)
        .map_err(|e| format!("hc read: {e}"))?;
    if raw.len() != HC_FLAT_WIDTH * 2 {
        return Err(format!(
            "hc {} has {} bytes, expected {}",
            path.display(),
            raw.len(),
            HC_FLAT_WIDTH * 2
        ));
    }
    Ok(raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn layer_meta_path(output_dir: &Path, layer: usize) -> PathBuf {
    output_dir.join("layer_meta").join(format!("L{layer:02}.json"))
}

fn hc_path(output_dir: &Path, pos: usize) -> PathBuf {
    output_dir.join("hc_state").join(format!("{pos:06}.u16le"))
}

fn persist_layer_meta(
    output_dir: &Path,
    layer: usize,
    probes: &[(String, Vec<u32>)],
    captures: &[Vec<Vec<LayerTokenCapture>>],
    book: &FlushBook,
) -> Result<(), String> {
    let mut tokens = Vec::new();
    for (pi, (_, token_ids)) in probes.iter().enumerate() {
        for pos in 0..token_ids.len() {
            let cap = captures
                .get(pi)
                .and_then(|p| p.get(pos))
                .and_then(|t| t.iter().find(|c| c.layer == layer))
                .ok_or_else(|| format!("missing capture for layer {layer} token {pos}"))?;
            let rec = book.rows.get(&(pi, pos, layer));
            tokens.push(json!({
                "probe": pi,
                "pos": pos,
                "selected_expert_ids": cap.selected_expert_ids,
                "normalized_route_weights": cap.normalized_route_weights,
                "hidden_retained": cap.hidden_retained,
                "router_input": rec.and_then(|r| r.router_input.as_ref()).map(|w| w.to_json()),
            }));
        }
    }
    let path = layer_meta_path(output_dir, layer);
    if path.exists() {
        return Err(format!("refusing to overwrite {}", path.display()));
    }
    write_json_new(
        &path,
        &json!({
            "layer": layer,
            "tokens": tokens,
        }),
    )
    .map_err(|e| e.to_string())
}

fn rebuild_flushed_layers(
    output_dir: &Path,
    probes: &[(String, Vec<u32>)],
    completed_layers: usize,
) -> Result<(Vec<Vec<Vec<LayerTokenCapture>>>, FlushBook), String> {
    let mut captures = empty_captures(probes);
    let mut book = FlushBook::default();
    for layer in 0..completed_layers {
        let text = fs::read_to_string(layer_meta_path(output_dir, layer))
            .map_err(|e| format!("read layer_meta L{layer}: {e}"))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("parse layer_meta L{layer}: {e}"))?;
        let rows = v["tokens"]
            .as_array()
            .ok_or_else(|| format!("layer_meta L{layer} missing tokens"))?;
        for row in rows {
            let pi = row["probe"].as_u64().ok_or("probe")? as usize;
            let pos = row["pos"].as_u64().ok_or("pos")? as usize;
            let ids: Vec<u32> = row["selected_expert_ids"]
                .as_array()
                .ok_or("ids")?
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            let weights: Vec<f32> = row["normalized_route_weights"]
                .as_array()
                .ok_or("weights")?
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            let hidden_retained = row["hidden_retained"].as_bool().unwrap_or(false);
            captures[pi][pos].push(LayerTokenCapture {
                layer,
                selected_expert_ids: ids,
                normalized_route_weights: weights,
                router_input_hidden: Vec::new(),
                hidden_retained,
                extra_x: Default::default(),
            });
            if hidden_retained {
                let meta = row
                    .get("router_input")
                    .filter(|v| !v.is_null())
                    .ok_or_else(|| format!("L{layer} token {pos} retained but no router_input meta"))?;
                let written = HiddenWrite {
                    relative_path: meta["relative_path"]
                        .as_str()
                        .ok_or("relative_path")?
                        .to_owned(),
                    sha256: meta["sha256"].as_str().ok_or("sha256")?.to_owned(),
                    bytes: meta["bytes"].as_u64().ok_or("bytes")? as usize,
                    elements: meta["elements"].as_u64().ok_or("elements")? as usize,
                };
                book.rows.insert(
                    (pi, pos, layer),
                    TokenFlushRecord {
                        router_input: Some(written),
                        extra: BTreeMap::new(),
                        routed_swiglu: Vec::new(),
                    },
                );
            }
        }
    }
    Ok((captures, book))
}

fn write_ckpt(
    path: &Path,
    next_layer: usize,
    tokens: &[u32],
    use_metal: bool,
    max_per_expert: usize,
    peak_rss: u64,
    peak_weight: u64,
    hidden_rows_retained_total: usize,
    hidden_bytes_written: usize,
    hidden_rows_per_layer: &[usize],
    wall_ms: u128,
    metal_dispatches: usize,
    fallbacks: usize,
    hc_sha256: &[String],
) -> Result<(), String> {
    let value = json!({
        "schema": CKPT_SCHEMA,
        "next_layer": next_layer,
        "tokens": tokens,
        "probe_id": PROBE_ID,
        "use_metal": use_metal,
        "max_per_expert": max_per_expert,
        "peak_rss_bytes": peak_rss,
        "peak_weight_bytes": peak_weight,
        "hidden_rows_retained_total": hidden_rows_retained_total,
        "hidden_bytes_written": hidden_bytes_written,
        "hidden_rows_per_layer": hidden_rows_per_layer,
        "wall_ms": wall_ms,
        "metal_dispatches": metal_dispatches,
        "fallbacks": fallbacks,
        "hc_sha256": hc_sha256,
    });
    let tmp = path.with_extension("json.tmp");
    let encoded = serde_json::to_vec_pretty(&value).map_err(|e| e.to_string())?;
    {
        let mut file = File::create(&tmp).map_err(|e| e.to_string())?;
        file.write_all(&encoded).map_err(|e| e.to_string())?;
        file.write_all(b"\n").map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
    }
    fs::rename(&tmp, path).map_err(|e| e.to_string())?;
    Ok(())
}

fn patch_honesty(
    result_path: &Path,
    execution_path: &str,
    use_metal: bool,
    metal_dispatches: usize,
    fallbacks: usize,
    tokens: usize,
    layers: usize,
    peak_rss: u64,
    peak_weight: u64,
    wall_ms: u128,
    artifact: &str,
    manifest_seal: &str,
    batched: bool,
    token_tile: usize,
) -> Result<(), String> {
    let text = fs::read_to_string(result_path).map_err(|e| e.to_string())?;
    let mut doc: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    if let Some(obj) = doc["runtime_binding"].as_object_mut() {
        obj.insert(
            "weight_backend".into(),
            json!("sealed_43_layer_stream_activations"),
        );
        obj.insert("metal_not_used".into(), json!(!use_metal));
        obj.insert("execution_path".into(), json!(execution_path));
        obj.insert("metal_dispatches".into(), json!(metal_dispatches));
        obj.insert("fallbacks".into(), json!(fallbacks));
        obj.insert("bos_window_only".into(), json!(true));
        obj.insert(
            "independent_position0_token_ids_not_a_sequence".into(),
            json!(true),
        );
        obj.insert("token_tile_batched".into(), json!(batched));
        obj.insert("token_tile".into(), json!(token_tile));
    }
    if let Some(obj) = doc["claim_boundary"].as_object_mut() {
        obj.insert(
            "full_43_layer_source_forward_not_executed_by_this_writer".into(),
            json!(false),
        );
        obj.insert(
            "source_forward_executed".into(),
            json!(true),
        );
        obj.insert("execution_path".into(), json!(execution_path));
        obj.insert("bos_window_only".into(), json!(true));
        obj.insert(
            "independent_position0_token_ids_not_a_sequence".into(),
            json!(true),
        );
        obj.insert(
            "full_compressed_indexer_graph".into(),
            json!(false),
        );
        obj.insert(
            "does_not_claim_COMPLETE_PHYSICAL_BPW_coherence_hcli_or_tps".into(),
            json!(true),
        );
    }
    doc["source_run"] = json!({
        "artifact": artifact,
        "manifest_seal_sha256": manifest_seal,
        "tokens": tokens,
        "layers": layers,
        "peak_rss_bytes": peak_rss,
        "peak_weight_resident_bytes": peak_weight,
        "wall_ms": wall_ms,
        "execution_path": execution_path,
        "metal_dispatches": metal_dispatches,
        "fallbacks": fallbacks,
        "token_corpus": "consecutive_vocab_ids_starting_at_bos_0",
        "position": 0,
        "seqlen": 1,
        "attention": "window_only_sparse_bos",
        "token_tile_batched": batched,
        "token_tile": token_tile,
    });
    fs::remove_file(result_path).map_err(|e| e.to_string())?;
    write_json_new(result_path, &doc).map_err(|e| e.to_string())?;
    Ok(())
}

fn main() {
    let args = parse_arguments().unwrap_or_else(|e| fail(e));
    let wall = Instant::now();
    let geometry = CaptureGeometry {
        layers: args.max_layer + 1,
        ..CaptureGeometry::sealed()
    };
    let set = CaptureSet::doctor6_only();
    let token_ids = token_ids_for_budget(args.tokens);
    let token_tile = if args.token_tile == 0 {
        default_token_tile(token_ids.len())
    } else {
        args.token_tile
    };
    if token_tile == 0 {
        fail("--token-tile must be > 0");
    }
    let probes = vec![(PROBE_ID.to_string(), token_ids.clone())];
    let token_index = build_token_index(&probes);
    let ckpt_path = args.output_dir.join(CKPT_NAME);

    if args.output_dir.exists() && !args.resume {
        fail(format!(
            "output dir {} exists; pass --resume to continue a checkpointed run",
            args.output_dir.display()
        ));
    }
    if args.resume && !ckpt_path.is_file() {
        fail(format!("--resume set but {} is missing", ckpt_path.display()));
    }
    fs::create_dir_all(&args.output_dir).unwrap_or_else(|e| fail(e.to_string()));

    eprintln!(
        "{}",
        format_capture_progress(
            probes.len(),
            token_ids.len(),
            &geometry,
            args.max_per_expert,
        )
    );
    eprintln!(
        "dsv4f source capture: artifact={} tokens={} layers=0..{} metal={} resume={} batch={} token_tile={}",
        args.artifact.display(),
        args.tokens,
        args.max_layer,
        args.use_metal,
        args.resume,
        args.batch,
        token_tile
    );

    let admission =
        prepare_sealed_admission_root(&args.artifact).unwrap_or_else(|e| fail(e.to_string()));
    let reader =
        DeepSeekV4FullStreamReader::admit(&admission.path).unwrap_or_else(|e| fail(e.to_string()));
    let anchors =
        verify_deepseek_v4_layer_source_anchors(&reader).unwrap_or_else(|e| fail(e.to_string()));
    if anchors.identity().repository != PINNED_REPOSITORY
        || anchors.identity().revision != PINNED_REVISION
    {
        fail("admitted reader source identity is not the pinned DSV4F revision");
    }

    let mut ledger = ResidentLedger::new(DECLARED_WEIGHT_RESIDENT_BOUND_BYTES);
    let mut profile = OperatorProfile::default();
    let mut metal = if args.use_metal {
        Some(StreamedNativeSession::new().unwrap_or_else(|e| fail(e.to_string())))
    } else {
        None
    };

    let mut next_layer = 0usize;
    let mut hidden_rows_retained_total = 0usize;
    let mut hidden_bytes_written = 0usize;
    let mut hidden_rows_per_layer = vec![0usize; geometry.layers];
    let mut peak_rss = peak_rss_bytes();
    let mut prior_wall_ms = 0u128;

    let (mut captures, mut book, mut hc_states) = if args.resume {
        let ckpt: Value = serde_json::from_str(
            &fs::read_to_string(&ckpt_path).unwrap_or_else(|e| fail(e.to_string())),
        )
        .unwrap_or_else(|e| fail(e.to_string()));
        if ckpt["schema"] != CKPT_SCHEMA {
            fail("checkpoint schema mismatch");
        }
        let ckpt_tokens: Vec<u32> = ckpt["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        if ckpt_tokens != token_ids {
            fail("checkpoint token list does not match --tokens");
        }
        next_layer = ckpt["next_layer"].as_u64().unwrap() as usize;
        hidden_rows_retained_total = ckpt["hidden_rows_retained_total"].as_u64().unwrap() as usize;
        hidden_bytes_written = ckpt["hidden_bytes_written"].as_u64().unwrap() as usize;
        prior_wall_ms = ckpt["wall_ms"].as_u64().unwrap_or(0) as u128;
        if let Some(rows) = ckpt["hidden_rows_per_layer"].as_array() {
            for (i, v) in rows.iter().enumerate() {
                if i < hidden_rows_per_layer.len() {
                    hidden_rows_per_layer[i] = v.as_u64().unwrap_or(0) as usize;
                }
            }
        }
        peak_rss = peak_rss.max(ckpt["peak_rss_bytes"].as_u64().unwrap_or(0));
        let (captures, book) = rebuild_flushed_layers(&args.output_dir, &probes, next_layer)
            .unwrap_or_else(|e| fail(e));
        let mut hc_states = Vec::with_capacity(token_ids.len());
        let sha_list = ckpt["hc_sha256"].as_array().cloned().unwrap_or_default();
        for pos in 0..token_ids.len() {
            let hc = read_hc(&hc_path(&args.output_dir, pos)).unwrap_or_else(|e| fail(e));
            if let Some(expected) = sha_list.get(pos).and_then(|v| v.as_str()) {
                let got = sha256_u16(&hc);
                if got != expected {
                    fail(format!("HC sha256 mismatch at token {pos}: {got} != {expected}"));
                }
            }
            hc_states.push(hc);
        }
        eprintln!("resumed at layer {next_layer} / {}", geometry.layers);
        (captures, book, hc_states)
    } else {
        let mut hc_states = Vec::with_capacity(token_ids.len());
        for (pos, &tid) in token_ids.iter().enumerate() {
            let hc = load_token_embed_hc(&reader, &mut ledger, tid as u64)
                .unwrap_or_else(|e| fail(e.to_string()));
            write_hc(&hc_path(&args.output_dir, pos), &hc).unwrap_or_else(|e| fail(e));
            hc_states.push(hc);
        }
        peak_rss = peak_rss.max(peak_rss_bytes());
        (empty_captures(&probes), FlushBook::default(), hc_states)
    };

    while next_layer < geometry.layers {
        if layer_meta_path(&args.output_dir, next_layer).is_file() {
            eprintln!(
                "dsv4f source capture: layer {next_layer} already flushed; adopting layer_meta"
            );
            let rebuilt = rebuild_flushed_layers(&args.output_dir, &probes, next_layer + 1)
                .unwrap_or_else(|e| fail(e));
            captures = rebuilt.0;
            book = rebuilt.1;
            next_layer += 1;
            continue;
        }
        let layer_anchor = anchors
            .layer(next_layer)
            .unwrap_or_else(|e| fail(e.to_string()))
            .clone();
        let layer_wall = Instant::now();
        let mut routes: Vec<(Vec<u32>, Vec<f32>)> = Vec::with_capacity(token_ids.len());
        let mut packed = vec![0.0f32; token_ids.len() * HIDDEN_SIZE];
        if args.batch {
            let mut t0 = 0usize;
            while t0 < token_ids.len() {
                let t1 = (t0 + token_tile).min(token_ids.len());
                let tile_ids: Vec<u64> = token_ids[t0..t1].iter().map(|&id| id as u64).collect();
                let tile_hc: Vec<Vec<u16>> = hc_states[t0..t1].to_vec();
                let captured = execute_layer_tile(
                    &reader,
                    &layer_anchor,
                    &tile_hc,
                    &tile_ids,
                    &mut ledger,
                    &mut profile,
                    metal.as_mut(),
                )
                .unwrap_or_else(|e| fail(format!("L{next_layer} tile {t0}..{t1}: {e}")));
                if captured.len() != t1 - t0 {
                    fail(format!(
                        "L{next_layer} tile {t0}..{t1} returned {} rows",
                        captured.len()
                    ));
                }
                for (i, cap) in captured.into_iter().enumerate() {
                    let t = t0 + i;
                    let tid = token_ids[t];
                    if cap.h_post_ffn_norm.len() != HIDDEN_SIZE {
                        fail(format!(
                            "L{next_layer} token {tid}: h_post_ffn_norm width {}",
                            cap.h_post_ffn_norm.len()
                        ));
                    }
                    packed[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE]
                        .copy_from_slice(&cap.h_post_ffn_norm);
                    routes.push((cap.selected_expert_ids, cap.normalized_route_weights));
                    hc_states[t] = cap.next_hc_bf16;
                    write_hc(&hc_path(&args.output_dir, t), &hc_states[t])
                        .unwrap_or_else(|e| fail(e));
                }
                peak_rss = peak_rss.max(peak_rss_bytes());
                eprintln!(
                    "  L{next_layer} tile {}/{} tokens {}..{} rss={} live_w={}",
                    (t0 / token_tile) + 1,
                    token_ids.len().div_ceil(token_tile),
                    t0,
                    t1,
                    peak_rss,
                    ledger.live_bytes()
                );
                if peak_rss > DECLARED_PEAK_RSS_BOUND_BYTES {
                    fail(format!(
                        "measured peak RSS {peak_rss} exceeded bound {DECLARED_PEAK_RSS_BOUND_BYTES}"
                    ));
                }
                t0 = t1;
            }
        } else {
            for (t, &tid) in token_ids.iter().enumerate() {
                let captured: StreamedLayerCapture = execute_one_layer_with_x(
                    &reader,
                    &layer_anchor,
                    &hc_states[t],
                    tid as u64,
                    &mut ledger,
                    &mut profile,
                    metal.as_mut(),
                )
                .unwrap_or_else(|e| fail(format!("L{next_layer} token {tid}: {e}")));
                if captured.h_post_ffn_norm.len() != HIDDEN_SIZE {
                    fail(format!(
                        "L{next_layer} token {tid}: h_post_ffn_norm width {}",
                        captured.h_post_ffn_norm.len()
                    ));
                }
                packed[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE]
                    .copy_from_slice(&captured.h_post_ffn_norm);
                routes.push((
                    captured.selected_expert_ids,
                    captured.normalized_route_weights,
                ));
                hc_states[t] = captured.next_hc_bf16;
                write_hc(&hc_path(&args.output_dir, t), &hc_states[t]).unwrap_or_else(|e| fail(e));
                peak_rss = peak_rss.max(peak_rss_bytes());
                if t == 0 || (t + 1) % 8 == 0 || t + 1 == token_ids.len() {
                    eprintln!(
                        "  L{next_layer} token {}/{} id={} rss={} live_w={}",
                        t + 1,
                        token_ids.len(),
                        tid,
                        peak_rss,
                        ledger.live_bytes()
                    );
                }
                if peak_rss > DECLARED_PEAK_RSS_BOUND_BYTES {
                    fail(format!(
                        "measured peak RSS {peak_rss} exceeded bound {DECLARED_PEAK_RSS_BOUND_BYTES}"
                    ));
                }
            }
        }
        let batch = LayerActivationBatch {
            h_post_ffn_norm: packed,
            ..LayerActivationBatch::default()
        };
        append_retained_layer_captures(
            &mut captures,
            &token_index,
            &mut routes,
            &batch,
            next_layer,
            &geometry,
            &set,
            args.max_per_expert,
        )
        .unwrap_or_else(|e| fail(e.to_string()));
        let resident = resident_retained_hidden_bytes(&captures);
        let stats = write_layer_retained_rows(
            &args.output_dir,
            &probes,
            next_layer,
            &captures,
            &geometry,
            &mut book,
        )
        .unwrap_or_else(|e| fail(e.to_string()));
        persist_layer_meta(&args.output_dir, next_layer, &probes, &captures, &book)
            .unwrap_or_else(|e| fail(e));
        hidden_rows_retained_total += stats.hidden_rows;
        hidden_bytes_written += stats.hidden_bytes;
        hidden_rows_per_layer[next_layer] = stats.hidden_rows;
        release_layer_retained_hiddens(&mut captures, next_layer);
        if resident_retained_hidden_bytes(&captures) != 0 {
            fail(format!(
                "layer {next_layer} hiddens must be freed before the next layer loads"
            ));
        }
        if ledger.live_bytes() != 0 {
            fail(format!(
                "layer {next_layer} leaked resident weight bytes: {}",
                ledger.live_bytes()
            ));
        }
        next_layer += 1;
        let hc_sha: Vec<String> = hc_states.iter().map(|hc| sha256_u16(hc)).collect();
        let metal_dispatches = metal.as_ref().map(|s| s.metal_dispatches()).unwrap_or(0);
        let fallbacks = metal.as_ref().map(|s| s.fallbacks()).unwrap_or(0);
        write_ckpt(
            &ckpt_path,
            next_layer,
            &token_ids,
            args.use_metal,
            args.max_per_expert,
            peak_rss,
            ledger.peak_bytes(),
            hidden_rows_retained_total,
            hidden_bytes_written,
            &hidden_rows_per_layer,
            prior_wall_ms + wall.elapsed().as_millis(),
            metal_dispatches,
            fallbacks,
            &hc_sha,
        )
        .unwrap_or_else(|e| fail(e));
        eprintln!(
            "dsv4f source capture layer {} done in {:.1}s; next={} retained_rows={} resident_before_free={} peak_rss={} peak_w={}",
            next_layer - 1,
            layer_wall.elapsed().as_secs_f64(),
            next_layer,
            stats.hidden_rows,
            resident,
            peak_rss,
            ledger.peak_bytes()
        );
    }

    let n_fit = n_fit_distribution(&captures, &geometry, args.row_threshold);
    // A resumed run that already emitted a result (for example after a
    // shorter --max-layer) must replace that file. Layer rows stay
    // create-new; only the summary is rewritten.
    let existing_result = args.output_dir.join("capture-result.json");
    if existing_result.is_file() {
        fs::remove_file(&existing_result).unwrap_or_else(|e| fail(e.to_string()));
    }
    let result_path = emit_capture_result(
        &args.output_dir,
        &probes,
        &captures,
        &book,
        &geometry,
        &set,
        args.max_per_expert,
        hidden_rows_retained_total,
        &hidden_rows_per_layer,
        hidden_bytes_written,
        &n_fit,
    )
    .unwrap_or_else(|e| fail(e.to_string()));

    let metal_dispatches = metal.as_ref().map(|s| s.metal_dispatches()).unwrap_or(0);
    let fallbacks = metal.as_ref().map(|s| s.fallbacks()).unwrap_or(0);
    let execution_path = if metal_dispatches > 0 {
        STREAMED_EXECUTION_PATH_METAL
    } else {
        STREAMED_EXECUTION_PATH
    };
    let wall_ms = prior_wall_ms + wall.elapsed().as_millis();
    peak_rss = peak_rss.max(peak_rss_bytes());
    patch_honesty(
        &result_path,
        execution_path,
        args.use_metal,
        metal_dispatches,
        fallbacks,
        token_ids.len(),
        geometry.layers,
        peak_rss,
        ledger.peak_bytes(),
        wall_ms,
        &admission.source_path.display().to_string(),
        reader.manifest_seal_sha256(),
        args.batch,
        token_tile,
    )
    .unwrap_or_else(|e| fail(e));

    let summary = json!({
        "status": "ok",
        "schema": RESULT_SCHEMA,
        "output_dir": args.output_dir,
        "result_path": result_path,
        "execution_path": execution_path,
        "tokens": token_ids.len(),
        "layers": geometry.layers,
        "hidden_rows_retained_total": hidden_rows_retained_total,
        "hidden_bytes_written": hidden_bytes_written,
        "peak_rss_bytes": peak_rss,
        "peak_weight_resident_bytes": ledger.peak_bytes(),
        "metal_dispatches": metal_dispatches,
        "fallbacks": fallbacks,
        "wall_ms": wall_ms,
        "n_fit_distribution": n_fit,
        "token_tile_batched": args.batch,
        "token_tile": token_tile,
        "honesty": {
            "synthetic_activations": false,
            "sealed_artifact_opened_read_only": true,
            "full_43_layer_source_forward_executed": true,
            "bos_window_only": true,
            "independent_position0_token_ids_not_a_sequence": true,
            "full_compressed_indexer_graph": false,
            "capture_set": "doctor6_only",
        },
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&summary).expect("summary serializes")
    );
    eprintln!(
        "dsv4f source capture complete: tokens={} layers={} path={} wall_s={:.1} peak_rss={} n_fit_mean={} zeros={}",
        token_ids.len(),
        geometry.layers,
        execution_path,
        wall_ms as f64 / 1000.0,
        peak_rss,
        n_fit["mean"],
        n_fit["count_zero"],
    );
}
