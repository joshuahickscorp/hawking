//! Complete Flash stateful-session attempt.
//!
//! This is the smallest end-to-end session-shaped executor built from the
//! already-qualified linear and full-attention organs.  It deliberately keeps
//! cross-species seams explicit (host vectors) while preserving recurrence in
//! every linear segment and KV state across every full-attention token list.
//! A receipt is emitted for either accepted-candidate success or the first
//! physical boundary; no promotion or performance claim is implied.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("Flash stateful session requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
#[path = "flash_full_attention_layer3.rs"]
mod full;
#[cfg(target_os = "macos")]
#[path = "flash_noetic_complete_layer0.rs"]
mod linear;
#[cfg(target_os = "macos")]
#[path = "flash_source_bf16_terminal.rs"]
mod terminal;

#[cfg(target_os = "macos")]
mod macos {
    use super::{full, linear, terminal};
    use hawking_core::metal::{MetalContext, TokenCommandBuffer};
    use hawking_core::model::qwen80_source_bf16_layer_major::SourceBf16Index;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeMap, BTreeSet};
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    const DEFAULT_ROOT: &str =
        "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc";
    const REPO_ID: &str = "Qwen/Qwen3.8-Flash-Next";
    const PINNED_REVISION: &str = "34567a4712bc9766c4449e2e98e4468bfa24d915";
    const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
    const HIDDEN: usize = 2560;
    const STREAMS: usize = 4;
    const HC: usize = HIDDEN * STREAMS;
    const FULL_LAYERS: [usize; 12] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47];
    const PROMPT_IDS: [usize; 5] = [5423, 799, 4581, 3817, 13];
    const DEFAULT_CANDIDATE: usize = 17;

    fn sha256(bytes: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(bytes);
        format!("{:x}", h.finalize())
    }

    fn arg_value(args: &[String], flag: &str) -> Option<String> {
        args.windows(2)
            .find(|pair| pair[0] == flag)
            .map(|pair| pair[1].clone())
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
    }

    fn parse_token_ids(argv: &[String]) -> Result<Vec<usize>, Box<dyn Error>> {
        if let Some(raw) = arg_value(argv, "--token-ids") {
            let ids = raw
                .split(',')
                .map(|part| part.trim().parse::<usize>())
                .collect::<Result<Vec<_>, _>>()?;
            if ids.is_empty() {
                return Err("--token-ids must not be empty".into());
            }
            return Ok(ids);
        }
        let mut ids = PROMPT_IDS.to_vec();
        ids.push(
            arg_value(argv, "--candidate")
                .map(|v| v.parse())
                .transpose()?
                .unwrap_or(DEFAULT_CANDIDATE),
        );
        Ok(ids)
    }

    /// Separate prompt from caller-supplied reference continuation tokens.
    ///
    /// Every continuation token is checked against the preceding terminal
    /// argmax.  The default preserves the historical one-candidate interface;
    /// a multi-token reference must name the boundary explicitly so no prompt
    /// token is silently counted as generated output.
    fn parse_prompt_len(argv: &[String], token_count: usize) -> Result<usize, Box<dyn Error>> {
        let prompt_len = arg_value(argv, "--prompt-length")
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(token_count.saturating_sub(1));
        if prompt_len == 0 || prompt_len >= token_count {
            return Err(format!(
                "--prompt-length must be in 1..{} for {token_count} token ids",
                token_count.saturating_sub(1)
            )
            .into());
        }
        Ok(prompt_len)
    }

    /// Exact route teacher for one bounded dense session.  This deliberately
    /// records every `(layer, token-slot)` top-k row as well as the per-layer
    /// union.  The compact candidate may use the union for storage, but must
    /// reproduce each individual teacher row before its terminal checks count.
    struct RouteTeacher {
        routes: BTreeMap<(usize, usize), Vec<u32>>,
        final_state_sha256: BTreeMap<(usize, usize), String>,
        unions: BTreeMap<usize, Vec<u32>>,
    }

    fn collect_route_rows(
        doc: &Value,
        root: &Path,
        rows: &mut Vec<Value>,
    ) -> Result<(), Box<dyn Error>> {
        for segment in doc
            .get("segments")
            .and_then(Value::as_array)
            .ok_or("route teacher omitted segments")?
        {
            if let Some(segment_rows) = segment.get("steps").and_then(Value::as_array) {
                rows.extend(segment_rows.iter().cloned());
                continue;
            }
            let receipt = segment
                .get("receipt")
                .and_then(Value::as_str)
                .ok_or("route teacher nonlinear segment omitted receipt")?;
            let layer = segment
                .get("layer")
                .and_then(Value::as_u64)
                .ok_or("route teacher nonlinear segment omitted layer")?;
            if layer >= 48 {
                return Err("route teacher nonlinear segment has invalid layer".into());
            }
            let path = {
                let candidate = PathBuf::from(receipt);
                if candidate.is_absolute() { candidate } else { root.join(candidate) }
            };
            let nested: Value = serde_json::from_slice(&fs::read(path)?)?;
            let nested_rows = nested
                .get("steps")
                .and_then(Value::as_array)
                .ok_or("route teacher attention receipt omitted steps")?;
            for nested_row in nested_rows {
                let mut stamped = nested_row.clone();
                let object = stamped
                    .as_object_mut()
                    .ok_or("route teacher attention row is not an object")?;
                if let Some(recorded_layer) = object.get("layer") {
                    if recorded_layer.as_u64() != Some(layer) {
                        return Err("route teacher attention row disagrees with segment layer".into());
                    }
                }
                object.insert("layer".to_string(), Value::from(layer));
                rows.push(stamped);
            }
        }
        Ok(())
    }

    fn parse_route_teacher(
        argv: &[String],
        token_ids: &[usize],
    ) -> Result<Option<RouteTeacher>, Box<dyn Error>> {
        let Some(raw_path) = arg_value(argv, "--route-teacher") else {
            return Ok(None);
        };
        let path = PathBuf::from(raw_path).canonicalize()?;
        let doc: Value = serde_json::from_slice(&fs::read(&path)?)?;
        let accepted_teacher_status = matches!(
            doc.get("status").and_then(Value::as_str),
            Some("PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE")
                | Some("PASSED_STATEFUL_COMPLETE_TOKEN_SESSION")
        );
        if !accepted_teacher_status || doc.get("model").and_then(Value::as_str) != Some(REPO_ID)
            || doc.get("pinned_revision").and_then(Value::as_str) != Some(PINNED_REVISION)
        {
            return Err("route teacher is not a pinned accepted Flash stateful control".into());
        }
        let teacher_tokens = doc
            .get("token_ids")
            .and_then(Value::as_array)
            .ok_or("route teacher omitted token IDs")?
            .iter()
            .map(|value| value.as_u64().map(|id| id as usize).ok_or("non-integer teacher token"))
            .collect::<Result<Vec<_>, _>>()?;
        if teacher_tokens != token_ids {
            return Err("route teacher token sequence differs from compact candidate".into());
        }
        let mut raw_rows = Vec::new();
        collect_route_rows(&doc, &repo_root(), &mut raw_rows)?;
        let mut routes = BTreeMap::new();
        let mut final_state_sha256 = BTreeMap::new();
        let mut unions: BTreeMap<usize, BTreeSet<u32>> = BTreeMap::new();
        for row in raw_rows {
            let layer = row.get("layer").and_then(Value::as_u64).ok_or("route teacher row missing layer")? as usize;
            let step = row.get("step").and_then(Value::as_u64).ok_or("route teacher row missing step")? as usize;
            let ids = row
                .get("route_ids")
                .and_then(Value::as_array)
                .ok_or("route teacher row omitted top-k IDs")?
                .iter()
                .map(|value| value.as_u64().map(|id| id as u32).ok_or("non-integer route ID"))
                .collect::<Result<Vec<_>, _>>()?;
            let state_hash = row
                .get("final_state_sha256")
                .and_then(Value::as_str)
                .filter(|value| value.len() == 64)
                .ok_or("route teacher row omitted final-state hash")?
                .to_owned();
            if layer >= 48 || step >= token_ids.len() || ids.len() != 10
                || ids.iter().any(|&id| id >= 512) || ids.iter().collect::<BTreeSet<_>>().len() != 10
                || routes.insert((layer, step), ids.clone()).is_some()
                || final_state_sha256.insert((layer, step), state_hash).is_some()
            {
                return Err("route teacher has invalid or duplicate top-k coverage".into());
            }
            unions.entry(layer).or_default().extend(ids);
        }
        if routes.len() != 48 * token_ids.len() || unions.len() != 48 {
            return Err("route teacher does not cover every Flash layer/token slot".into());
        }
        Ok(Some(RouteTeacher {
            routes,
            final_state_sha256,
            unions: unions.into_iter().map(|(layer, ids)| (layer, ids.into_iter().collect())).collect(),
        }))
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn bf16(bytes: &[u8], i: usize) -> f32 {
        f32::from_bits((u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]) as u32) << 16)
    }

    fn embedding_row(
        index: &SourceBf16Index,
        token_id: usize,
        vocab: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        if token_id >= vocab {
            return Err(format!("token {token_id} outside vocab {vocab}").into());
        }
        let bytes = index.read_raw_range(EMBEDDING, token_id * HIDDEN * 2, HIDDEN * 2)?;
        Ok((0..HIDDEN).map(|i| bf16(&bytes, i)).collect())
    }

    fn repeated_streams(row: &[f32]) -> Vec<f32> {
        (0..STREAMS).flat_map(|_| row.iter().copied()).collect()
    }

    fn vocab(root: &Path) -> Result<usize, Box<dyn Error>> {
        let config: Value = serde_json::from_slice(&fs::read(root.join("config.json"))?)?;
        config
            .get("text_config")
            .and_then(|v| v.get("vocab_size"))
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .ok_or_else(|| "text_config.vocab_size missing".into())
    }

    fn write_state(path: &Path, state: &[f32]) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, f32_bytes(state))?;
        Ok(())
    }

    fn current_rss_bytes() -> Option<u64> {
        let pid = std::process::id().to_string();
        let output = Command::new("ps")
            .args(["-o", "rss=", "-p", &pid])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        std::str::from_utf8(&output.stdout)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()
            .map(|kib| kib.saturating_mul(1024))
    }

    fn run_linear_segment(
        index: &SourceBf16Index,
        context: &MetalContext,
        layer_start: usize,
        layer_end: usize,
        token_ids: &[usize],
        vocab: usize,
        input_states: Option<&[Vec<f32>]>,
        route_teacher: Option<&RouteTeacher>,
        device_only_compact_banks: bool,
    ) -> Result<(Vec<Vec<f32>>, Value, Vec<linear::StatefulLinearLayer>), Box<dyn Error>> {
        if let Some(states) = input_states {
            if states.len() != token_ids.len() || states.iter().any(|state| state.len() != HC) {
                return Err("linear segment input states do not cover every exact token state".into());
            }
        }
        let first_row = embedding_row(&index, token_ids[0], vocab)?;
        let first_base = repeated_streams(&first_row);
        let mut layers = Vec::new();
        let load_started = Instant::now();
        let bytes_before_layers = index.bytes_read_total();
        for layer in layer_start..=layer_end {
            let session = if let Some(teacher) = route_teacher {
                let union = teacher
                    .unions
                    .get(&layer)
                    .ok_or("route teacher omitted linear-layer union")?;
                if device_only_compact_banks {
                    linear::StatefulLinearLayer::new_compact_union_device_only(
                        index,
                        context,
                        layer,
                        union,
                        &first_base,
                    )?
                } else {
                    linear::StatefulLinearLayer::new_compact_union(
                        index,
                        context,
                        layer,
                        union,
                        &first_base,
                    )?
                }
            } else {
                if device_only_compact_banks {
                    return Err("--device-only-compact-banks requires --route-teacher".into());
                }
                linear::StatefulLinearLayer::new_dense(index, context, layer, &first_base)?
            };
            layers.push(session);
        }
        let persistent_state_bytes = layers
            .len()
            .saturating_mul(linear::StatefulLinearLayer::persistent_state_bytes());
        let (source_load_ns, device_prepare_ns, graph_prepare_ns) = layers
            .iter()
            .map(linear::StatefulLinearLayer::prepare_timing_ns)
            .fold((0u64, 0u64, 0u64), |acc, value| {
                (
                    acc.0.saturating_add(value.0),
                    acc.1.saturating_add(value.1),
                    acc.2.saturating_add(value.2),
                )
            });
        let layer_prepare_ns = load_started.elapsed().as_nanos() as u64;
        let layer_source_bytes = index.bytes_read_total().saturating_sub(bytes_before_layers);
        let started = Instant::now();
        let mut outputs = Vec::with_capacity(token_ids.len());
        let mut rows = Vec::with_capacity(token_ids.len() * layers.len());
        for (step, &token_id) in token_ids.iter().enumerate() {
            let host_base = if let Some(states) = input_states {
                states[step].clone()
            } else {
                let row = embedding_row(&index, token_id, vocab)?;
                repeated_streams(&row)
            };
            let mut prior_device = None;
            let mut final_state = None;
            for (offset, layer) in layers.iter_mut().enumerate() {
                let (device_output, gpu_ns, wall_ns, dispatches, observed) = if offset == 0 {
                    layer.step(context, Some(&host_base), None, step == 0)?
                } else {
                    let input = prior_device
                        .as_ref()
                        .ok_or("missing device inter-layer state")?;
                    layer.step(context, None, Some(input), step == 0)?
                };
                let layer_index = layer_start + offset;
                if observed.iter().any(|v| !v.is_finite()) {
                    return Err(format!(
                        "non-finite linear state at layer {layer_index} step {step}"
                    )
                    .into());
                }
                // The dense control owns the complete immutable expert bank.
                // Record only the router's already-produced native top-k IDs
                // so a later compact candidate can bind an exact, bounded
                // route union rather than guess or reuse one token's route.
                let route_ids = layer.route_ids();
                if let Some(teacher) = route_teacher {
                    let expected = teacher
                        .routes
                        .get(&(layer_index, step))
                        .ok_or("route teacher omitted linear-layer token row")?;
                    if route_ids != *expected {
                        return Err(format!(
                            "route drift at linear layer {layer_index} token slot {step}: expected={expected:?} observed={route_ids:?}"
                        )
                        .into());
                    }
                }
                rows.push(json!({
                    "step": step,
                    "token_id": token_id,
                    "layer": layer_index,
                    "dispatches": dispatches,
                    "gpu_ns": gpu_ns,
                    "wall_ns": wall_ns,
                    "device_inter_layer_handoff": offset > 0,
                    "inter_species_input": input_states.is_some() && offset == 0,
                    "finite": true,
                    "route_ids": route_ids,
                    "final_state_sha256": sha256(&f32_bytes(&observed)),
                }));
                prior_device = Some(device_output);
                final_state = Some(observed);
            }
            outputs.push(final_state.ok_or("linear segment produced no output")?);
        }
        let execution_wall_ns = started.elapsed().as_nanos() as u64;
        let execution_gpu_ns = rows
            .iter()
            .map(|r| r.get("gpu_ns").and_then(Value::as_u64).unwrap_or(0))
            .sum::<u64>();
        let execution_dispatches = rows
            .iter()
            .map(|r| r.get("dispatches").and_then(Value::as_u64).unwrap_or(0))
            .sum::<u64>();
        let host_and_unattributed_ns = rows
            .iter()
            .map(|r| {
                r.get("wall_ns")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
                    .saturating_sub(r.get("gpu_ns").and_then(Value::as_u64).unwrap_or(0))
            })
            .sum::<u64>();
        Ok((
            outputs,
            json!({
                "layers": [layer_start, layer_end],
                "species": "linear_attention",
                "stateful_recurrence": true,
                "expert_bank_mode": if route_teacher.is_some() { "route_union_compact_teacher_bound" } else { "dense" },
                "immutable_weight_ownership": if device_only_compact_banks { "device_only_after_source_upload" } else { "host_and_device" },
                "host_weights_retained": layers.iter().all(linear::StatefulLinearLayer::host_weights_retained),
                "persistent_state": {
                    "layers": layers.len(),
                    "per_layer_bytes": linear::StatefulLinearLayer::persistent_state_bytes(),
                    "total_bytes": persistent_state_bytes,
                    "growth_bytes_per_token": 0,
                    "reset_only_before_first_session_token": true,
                },
                "device_inter_layer_handoffs": rows.iter().filter(|r| r.get("device_inter_layer_handoff") == Some(&Value::Bool(true))).count(),
                "steps": rows,
                "wall_ns": execution_wall_ns,
                "source_payload_bytes_read": layer_source_bytes,
                "source_payload_bytes_read_cumulative": index.bytes_read_total(),
                "timing": {
                    "layer_source_and_device_prepare_ns": layer_prepare_ns,
                    "source_load_ns": source_load_ns,
                    "device_prepare_ns": device_prepare_ns,
                    "graph_prepare_ns": graph_prepare_ns,
                    "layer_source_bytes_read": layer_source_bytes,
                    "execution_wall_ns": execution_wall_ns,
                    "execution_gpu_ns": execution_gpu_ns,
                    "execution_dispatches": execution_dispatches,
                    "host_and_unattributed_ns": host_and_unattributed_ns
                },
            }),
            layers,
        ))
    }

    fn run_full_layer(
        index: &SourceBf16Index,
        context: &MetalContext,
        layer: usize,
        token_ids: &[usize],
        input_states: &[Vec<f32>],
        out_dir: &Path,
        route_teacher: Option<&RouteTeacher>,
    ) -> Result<(Vec<Vec<f32>>, Value), Box<dyn Error>> {
        let receipt = out_dir.join(format!("layer-{layer}-attention.json"));
        let layer_call_started = Instant::now();
        let route_union = route_teacher
            .map(|teacher| teacher.unions.get(&layer).ok_or("route teacher omitted full-attention union"))
            .transpose()?;
        let mut bank = full::StatefulFullAttentionLayer::new_device_only(
            index,
            context,
            layer,
            token_ids.len(),
            route_union.map(Vec::as_slice),
        )?;
        let mut states = Vec::with_capacity(token_ids.len());
        let mut rows = Vec::with_capacity(token_ids.len());
        for (step, (&token_id, input)) in token_ids.iter().zip(input_states).enumerate() {
            let (_device_state, state, row) = bank.step(context, Some(input), None, step, token_id)?;
            if let Some(teacher) = route_teacher {
                let expected = teacher.routes.get(&(layer, step)).ok_or("route teacher omitted full-attention token row")?;
                let observed = row.get("route_ids").and_then(Value::as_array).ok_or("resident full-attention row omitted top-k IDs")?
                    .iter().map(|value| value.as_u64().map(|id| id as u32).ok_or("non-integer route ID"))
                    .collect::<Result<Vec<_>, _>>()?;
                if observed != *expected {
                    return Err(format!("route drift at resident full-attention layer {layer} token slot {step}: expected={expected:?} observed={observed:?}").into());
                }
            }
            states.push(state);
            rows.push(row);
        }
        if states.len() != token_ids.len() || states.iter().any(|s| s.len() != HC || s.iter().any(|v| !v.is_finite())) {
            return Err(format!("resident full-attention layer {layer} returned invalid state count or non-finite values").into());
        }
        let (source_load_ns, device_prepare_ns, graph_prepare_ns) = bank.prepare_timing_ns();
        let execution_gpu_ns = rows.iter().filter_map(|row| row.get("gpu_ns").and_then(Value::as_u64)).sum::<u64>();
        let execution_dispatches = rows.iter().filter_map(|row| row.get("dispatches").and_then(Value::as_u64)).sum::<u64>();
        let execution_wall_ns = rows.iter().filter_map(|row| row.get("wall_ns").and_then(Value::as_u64)).sum::<u64>();
        let attention_doc = json!({
            "schema": "hawking.flash.stateful_full_attention_resident_bank.v1",
            "status": "PASSED_STATEFUL_KV_ORGAN",
            "layer": layer,
            "steps": rows,
            "execution": {
                "process_boundary": "one native process",
                "context_reused": true,
                "weights_reused": true,
                "immutable_weight_ownership": "device_only_after_source_upload",
                "host_source_weights_retained_during_token_loop": bank.host_weights_retained(),
                "kv_cache_reused": true,
                "kv_cache_slots": token_ids.len(),
                "state_memory": {"persistent_kv_bytes": bank.persistent_state_bytes(), "growth_bytes_per_token": full::StatefulFullAttentionLayer::persistent_state_growth_bytes_per_token()},
                "expert_bank_mode": if route_teacher.is_some() { "route_union_compact" } else { "dense" },
                "source_payload_bytes_read": bank.source_payload_bytes(),
                "device_weight_bytes": bank.device_weight_bytes(),
                "timing": {"source_load_ns": source_load_ns, "device_prepare_ns": device_prepare_ns, "graph_prepare_ns": graph_prepare_ns, "execution_wall_ns": execution_wall_ns, "execution_gpu_ns": execution_gpu_ns, "execution_dispatches": execution_dispatches},
                "claim_boundary": "One full-attention bank owns its immutable Metal weights and KV state across the named token slots. This is an organ-level resident primitive, not complete token-major whole-model residency or TPS."
            }
        });
        fs::write(&receipt, serde_json::to_vec_pretty(&attention_doc)?)?;
        let attention_execution = attention_doc.get("execution").cloned().unwrap();
        Ok((states, json!({
            "layer": layer,
            "species": "full_attention",
            "stateful_kv": true,
            "expert_bank_mode": attention_execution.get("expert_bank_mode").cloned().unwrap_or(Value::Null),
            "immutable_weight_ownership": attention_execution.get("immutable_weight_ownership").cloned().unwrap_or(Value::Null),
            "host_source_weights_retained_during_token_loop": attention_execution.get("host_source_weights_retained_during_token_loop").cloned().unwrap_or(Value::Null),
            "state_memory": attention_execution.get("state_memory").cloned().unwrap_or_else(|| json!({"status": "UNAVAILABLE"})),
            "source_payload_bytes_read": attention_execution.get("source_payload_bytes_read").cloned().unwrap_or(Value::Null),
            "device_weight_bytes": attention_execution.get("device_weight_bytes").cloned().unwrap_or(Value::Null),
            "timing": {"full_layer_call_wall_ns": layer_call_started.elapsed().as_nanos() as u64, "receipt_read_ns": 0, "source_load_ns": attention_execution.pointer("/timing/source_load_ns").cloned().unwrap_or(Value::Null), "oracle_ns": Value::Null, "device_prepare_ns": attention_execution.pointer("/timing/device_prepare_ns").cloned().unwrap_or(Value::Null)},
            "receipt": receipt,
        })))
    }

    /// The first complete Flash body with a real token-major schedule.  Every
    /// bank is constructed before token zero, then survives while all token
    /// slots advance through all 48 layers.  The route teacher remains a
    /// bounded correctness oracle; it never supplies activations or routing
    /// decisions to the native banks.
    enum ResidentTokenMajorBank {
        Linear {
            layer: usize,
            bank: linear::StatefulLinearLayer,
        },
        FullAttention {
            layer: usize,
            bank: full::StatefulFullAttentionLayer,
        },
    }

    fn run_token_major_resident_session(
        index: &SourceBf16Index,
        context: &MetalContext,
        token_ids: &[usize],
        vocab: usize,
        teacher: Option<&RouteTeacher>,
        host_attention_seam_diagnostic: bool,
        host_linear_seam_diagnostic: bool,
        resident_bank_limit: usize,
        enforce_state_hashes: bool,
    ) -> Result<(Vec<Vec<f32>>, Vec<Value>, Vec<ResidentTokenMajorBank>), Box<dyn Error>> {
        let first_row = embedding_row(index, token_ids[0], vocab)?;
        let first_base = repeated_streams(&first_row);
        let construction_started = Instant::now();
        let mut banks = Vec::with_capacity(48);
        for layer in 0..resident_bank_limit {
            let routes = teacher
                .map(|bound| bound.unions.get(&layer).ok_or("route teacher omitted token-major route union"))
                .transpose()?;
            if FULL_LAYERS.contains(&layer) {
                banks.push(ResidentTokenMajorBank::FullAttention {
                    layer,
                    bank: full::StatefulFullAttentionLayer::new_device_only(
                        index,
                        context,
                        layer,
                        token_ids.len(),
                        routes.map(Vec::as_slice),
                    )?,
                });
            } else {
                banks.push(ResidentTokenMajorBank::Linear {
                    layer,
                    bank: if let Some(routes) = routes {
                        linear::StatefulLinearLayer::new_compact_union_device_only(
                            index, context, layer, routes, &first_base,
                        )?
                    } else {
                        linear::StatefulLinearLayer::new_dense(index, context, layer, &first_base)?
                    },
                });
            }
        }
        if banks.len() != resident_bank_limit {
            return Err("token-major resident construction did not retain the requested layer count".into());
        }
        let construction_ns = construction_started.elapsed().as_nanos() as u64;
        let mut rows_by_layer = (0..48).map(|_| Vec::with_capacity(token_ids.len())).collect::<Vec<_>>();
        let mut final_states = Vec::with_capacity(token_ids.len());
        let execution_started = Instant::now();
        for (step, &token_id) in token_ids.iter().enumerate() {
            let embedding = embedding_row(index, token_id, vocab)?;
            let host_base = repeated_streams(&embedding);
            let mut prior_device = None;
            let mut prior_host: Option<Vec<f32>> = None;
            let mut final_state = None;
            for bank in banks.iter_mut() {
                let (layer, device_output, observed, mut row) = match bank {
                    ResidentTokenMajorBank::Linear { layer, bank } => {
                        let (output, gpu_ns, wall_ns, dispatches, observed) = if prior_device.is_none() {
                            bank.step(context, Some(&host_base), None, step == 0)?
                        } else if host_linear_seam_diagnostic {
                            let input = prior_host
                                .as_deref()
                                .ok_or("token-major linear layer lacked diagnostic host input")?;
                            bank.step(context, Some(input), None, step == 0)?
                        } else {
                            bank.step(
                                context,
                                None,
                                prior_device.as_ref(),
                                step == 0,
                            )?
                        };
                        let route_ids = bank.route_ids();
                        (
                            *layer,
                            output,
                            observed,
                            json!({
                                "layer": *layer,
                                "step": step,
                                "token_id": token_id,
                                "dispatches": dispatches,
                                "gpu_ns": gpu_ns,
                                "wall_ns": wall_ns,
                                "device_inter_layer_handoff": *layer != 0,
                                "finite": true,
                                "route_ids": route_ids,
                            }),
                        )
                    }
                    ResidentTokenMajorBank::FullAttention { layer, bank } => {
                        let (output, observed, row) = if host_attention_seam_diagnostic {
                            let input = prior_host
                                .as_deref()
                                .ok_or("token-major full-attention layer lacked diagnostic host input")?;
                            bank.step(context, Some(input), None, step, token_id)?
                        } else {
                            let input = prior_device
                                .as_ref()
                                .ok_or("token-major full-attention layer lacked device input")?;
                            bank.step(context, None, Some(input), step, token_id)?
                        };
                        (*layer, output, observed, row)
                    }
                };
                if observed.len() != HC || observed.iter().any(|value| !value.is_finite()) {
                    return Err(format!("token-major resident layer {layer} produced invalid state at token slot {step}").into());
                }
                let route_ids = row
                    .get("route_ids")
                    .and_then(Value::as_array)
                    .ok_or("token-major resident row omitted route IDs")?
                    .iter()
                    .map(|value| value.as_u64().map(|id| id as u32).ok_or("non-integer token-major route ID"))
                    .collect::<Result<Vec<_>, _>>()?;
                let observed_state_sha256 = sha256(&f32_bytes(&observed));
                if let Some(teacher) = teacher {
                    let expected_state_sha256 = teacher
                        .final_state_sha256
                        .get(&(layer, step))
                        .ok_or("route teacher omitted token-major state hash")?;
                    if enforce_state_hashes && observed_state_sha256 != *expected_state_sha256 {
                        return Err(format!(
                            "state drift at token-major resident layer {layer} token slot {step}: expected={expected_state_sha256} observed={observed_state_sha256}"
                        )
                        .into());
                    }
                    let expected = teacher
                        .routes
                        .get(&(layer, step))
                        .ok_or("route teacher omitted token-major row")?;
                    if route_ids != *expected {
                        return Err(format!(
                            "route drift at token-major resident layer {layer} token slot {step}: expected={expected:?} observed={route_ids:?}"
                        )
                        .into());
                    }
                }
                let row_object = row
                    .as_object_mut()
                    .ok_or("token-major resident row was not an object")?;
                row_object.insert("layer".to_string(), Value::from(layer as u64));
                row_object.insert("device_inter_layer_handoff".to_string(), Value::Bool(layer != 0));
                row_object.insert("final_state_sha256".to_string(), Value::String(observed_state_sha256));
                rows_by_layer[layer].push(row);
                prior_device = Some(device_output);
                prior_host = Some(observed.clone());
                final_state = Some(observed);
            }
            final_states.push(final_state.ok_or("token-major resident token produced no final state")?);
        }
        let execution_wall_ns = execution_started.elapsed().as_nanos() as u64;
        if resident_bank_limit != 48 {
            return Err(format!(
                "token-major cardinality diagnostic passed all exact state and route checks through layer {} across {} token slots; no complete-model claim",
                resident_bank_limit - 1,
                token_ids.len(),
            )
            .into());
        }
        let mut segments = Vec::with_capacity(48);
        for bank in &banks {
            match bank {
                ResidentTokenMajorBank::Linear { layer, bank } => {
                    let rows = &rows_by_layer[*layer];
                    let (source_load_ns, device_prepare_ns, graph_prepare_ns) = bank.prepare_timing_ns();
                    let device_weight_bytes = bank
                        .source_payload_bytes()
                        .saturating_add((512 * std::mem::size_of::<u32>()) as u64);
                    segments.push(json!({
                        "layers": [*layer, *layer],
                        "species": "linear_attention",
                        "stateful_recurrence": true,
                        "expert_bank_mode": "route_union_compact_teacher_bound",
                        "immutable_weight_ownership": "device_only_after_source_upload",
                        "host_weights_retained": bank.host_weights_retained(),
                        "state_memory": {
                            "layers": 1,
                            "per_layer_bytes": linear::StatefulLinearLayer::persistent_state_bytes(),
                            "total_bytes": linear::StatefulLinearLayer::persistent_state_bytes(),
                            "growth_bytes_per_token": 0,
                            "reset_only_before_first_session_token": true,
                        },
                        "device_inter_layer_handoffs": rows.iter().filter(|row| row.get("device_inter_layer_handoff") == Some(&Value::Bool(true))).count(),
                        "steps": rows,
                        "source_payload_bytes_read": bank.source_payload_bytes(),
                        "device_weight_bytes": device_weight_bytes,
                        "timing": {
                            "construction_total_ns": construction_ns,
                            "source_load_ns": source_load_ns,
                            "device_prepare_ns": device_prepare_ns,
                            "graph_prepare_ns": graph_prepare_ns,
                        },
                    }));
                }
                ResidentTokenMajorBank::FullAttention { layer, bank } => {
                    let rows = &rows_by_layer[*layer];
                    let (source_load_ns, device_prepare_ns, graph_prepare_ns) = bank.prepare_timing_ns();
                    segments.push(json!({
                        "layer": *layer,
                        "species": "full_attention",
                        "stateful_kv": true,
                        "expert_bank_mode": "route_union_compact_teacher_bound",
                        "immutable_weight_ownership": "device_only_after_source_upload",
                        "host_source_weights_retained_during_token_loop": bank.host_weights_retained(),
                        "state_memory": {
                            "persistent_kv_bytes": bank.persistent_state_bytes(),
                            "growth_bytes_per_token": full::StatefulFullAttentionLayer::persistent_state_growth_bytes_per_token(),
                        },
                        "device_inter_layer_handoffs": rows.iter().filter(|row| row.get("device_inter_layer_handoff") == Some(&Value::Bool(true))).count(),
                        "steps": rows,
                        "source_payload_bytes_read": bank.source_payload_bytes(),
                        "device_weight_bytes": bank.device_weight_bytes(),
                        "timing": {
                            "construction_total_ns": construction_ns,
                            "source_load_ns": source_load_ns,
                            "device_prepare_ns": device_prepare_ns,
                            "graph_prepare_ns": graph_prepare_ns,
                        },
                    }));
                }
            }
        }
        if final_states.len() != token_ids.len() {
            return Err("token-major resident final state/token count mismatch".into());
        }
        let _ = execution_wall_ns;
        Ok((final_states, segments, banks))
    }

    /// Replays a fixed, already accepted prompt/continuation sequence through
    /// the retained body.  This deliberately runs after the exact diagnostic
    /// control: no state snapshots, route readbacks, terminal execution, or
    /// source tensor reads occur in its timed continuation window.
    fn run_token_major_clean_replay(
        banks: &mut [ResidentTokenMajorBank],
        context: &MetalContext,
        bases: &[Vec<f32>],
        token_ids: &[usize],
        prompt_len: usize,
        reps: usize,
        async_linear_submit: bool,
        linear_region_submit: bool,
    ) -> Result<Value, Box<dyn Error>> {
        if banks.len() != 48 || bases.len() != token_ids.len() || prompt_len == 0 || prompt_len >= token_ids.len() || reps == 0 {
            return Err("invalid clean token-major replay configuration".into());
        }
        let mut token_wall_ns = Vec::with_capacity(reps * (token_ids.len() - prompt_len));
        let mut token_gpu_samples_ns = Vec::with_capacity(reps * (token_ids.len() - prompt_len));
        let mut token_dispatches = Vec::with_capacity(reps * (token_ids.len() - prompt_len));
        let mut token_command_buffers = Vec::with_capacity(reps * (token_ids.len() - prompt_len));
        let replay_started = Instant::now();
        for _ in 0..reps {
            for bank in banks.iter_mut() {
                if let ResidentTokenMajorBank::FullAttention { bank, .. } = bank {
                    bank.reset_state();
                }
            }
            for (step, (&token_id, host_base)) in token_ids.iter().zip(bases).enumerate() {
                let timed_decode = step >= prompt_len;
                let token_started = Instant::now();
                let mut token_gpu_ns = 0_u64;
                let mut dispatches = 0_usize;
                let mut command_buffers = 0_usize;
                let mut prior_device = None;
                let mut bank_index = 0_usize;
                while bank_index < banks.len() {
                    if linear_region_submit
                        && matches!(banks.get(bank_index), Some(ResidentTokenMajorBank::Linear { .. }))
                    {
                        let mut region = TokenCommandBuffer::new(context);
                        while let Some(ResidentTokenMajorBank::Linear { bank, .. }) = banks.get_mut(bank_index) {
                            let layer_dispatches = if prior_device.is_none() {
                                bank.encode_step_into(context, &mut region, Some(host_base), None, step == 0)?
                            } else {
                                bank.encode_step_into(context, &mut region, None, prior_device.as_ref(), step == 0)?
                            };
                            dispatches = dispatches.saturating_add(layer_dispatches);
                            prior_device = Some(bank.final_state_buffer());
                            bank_index += 1;
                        }
                        region.commit_no_wait()?;
                        command_buffers = command_buffers.saturating_add(1);
                        continue;
                    }
                    let bank = banks
                        .get_mut(bank_index)
                        .ok_or("token-major scheduler bank index drifted")?;
                    let (output, gpu_ns, _wall_ns, layer_dispatches) = match bank {
                        ResidentTokenMajorBank::Linear { bank, .. } if async_linear_submit && prior_device.is_none() => {
                            let (output, dispatches) = bank.step_submit_fast(context, Some(host_base), None, step == 0)?;
                            (output, 0, 0, dispatches)
                        }
                        ResidentTokenMajorBank::Linear { bank, .. } if async_linear_submit => {
                            let (output, dispatches) = bank.step_submit_fast(context, None, prior_device.as_ref(), step == 0)?;
                            (output, 0, 0, dispatches)
                        }
                        ResidentTokenMajorBank::Linear { bank, .. } if prior_device.is_none() =>
                            bank.step_fast(context, Some(host_base), None, step == 0)?,
                        ResidentTokenMajorBank::Linear { bank, .. } =>
                            bank.step_fast(context, None, prior_device.as_ref(), step == 0)?,
                        ResidentTokenMajorBank::FullAttention { bank, .. } =>
                            bank.step_fast(context, None, prior_device.as_ref(), step, token_id)?,
                    };
                    token_gpu_ns = token_gpu_ns.saturating_add(gpu_ns);
                    dispatches = dispatches.saturating_add(layer_dispatches);
                    prior_device = Some(output);
                    command_buffers = command_buffers.saturating_add(1);
                    bank_index += 1;
                }
                if timed_decode {
                    token_wall_ns.push(token_started.elapsed().as_nanos() as u64);
                    token_gpu_samples_ns.push(token_gpu_ns);
                    token_dispatches.push(dispatches);
                    token_command_buffers.push(command_buffers);
                }
            }
        }
        let mut sorted_wall = token_wall_ns.clone();
        sorted_wall.sort_unstable();
        let median_wall_ns = sorted_wall[sorted_wall.len() / 2];
        let mean_wall_ns = token_wall_ns.iter().sum::<u64>() / token_wall_ns.len() as u64;
        let mean_gpu_ns = token_gpu_samples_ns.iter().sum::<u64>() / token_gpu_samples_ns.len() as u64;
        let mean_dispatches = token_dispatches.iter().sum::<usize>() / token_dispatches.len();
        let mean_command_buffers = token_command_buffers.iter().sum::<usize>() / token_command_buffers.len();
        Ok(json!({
            "status": "MEASURED_CLEAN_BOUNDED_REPLAY",
            "repetitions": reps,
            "prompt_tokens_replayed_untimed": prompt_len * reps,
            "timed_known_continuation_tokens": token_wall_ns.len(),
            "state_reset": "full-attention KV buffers explicitly zeroed before every replay; linear recurrent state reset at token zero",
            "timed_window": "retained device-only banks; no source tensor reads, host activation snapshots, route readback/comparison, or terminal execution",
            "scheduler": if linear_region_submit { "each contiguous linear region is encoded into one queue-ordered command buffer; the following full-attention command buffer drains it" } else if async_linear_submit { "linear banks commit without per-layer CPU wait; succeeding full-attention command buffers drain the ordered Metal queue" } else { "every resident bank commits and waits before the next bank" },
            "async_linear_submit": async_linear_submit || linear_region_submit,
            "linear_region_submit": linear_region_submit,
            "median_token_wall_ns": median_wall_ns,
            "mean_token_wall_ns": mean_wall_ns,
            "mean_token_gpu_ns": mean_gpu_ns,
            "gpu_timing_attribution": if async_linear_submit || linear_region_submit { "partial: only synchronously drained full-attention command buffers contribute per-layer GPU timings; token wall is the authoritative scheduler metric" } else { "sum of per-layer completed command-buffer GPU intervals" },
            "mean_dispatches_per_token": mean_dispatches,
            "mean_command_buffers_per_token": mean_command_buffers,
            "bounded_replay_decode_tps": 1_000_000_000_f64 / mean_wall_ns as f64,
            "qualified_canonical_decode_tps": Value::Null,
            "claim_boundary": "This is a bounded known-sequence replay over a body that previously passed exact state/route/token checks. It is not capability-qualified sustained decode TPS, future-route coverage, EBPW, or promotion evidence.",
            "replay_wall_ns_including_untimed_prompt": replay_started.elapsed().as_nanos() as u64,
        }))
    }

    fn terminal_token(
        executor: &terminal::TerminalExecutor,
        state: &[f32],
        path: &Path,
    ) -> Result<(usize, Value), Box<dyn Error>> {
        write_state(path, state)?;
        let receipt = path.with_extension("terminal.json");
        let (token, _doc) = executor
            .run_state(&path.to_path_buf(), Some(receipt.clone()))
            .map_err(|e| -> Box<dyn Error> { e.into() })?;
        Ok((
            token,
            json!({"token_id": token, "receipt": receipt, "state": path}),
        ))
    }

    fn main_impl() -> Result<(), Box<dyn Error>> {
        let argv: Vec<String> = env::args().collect();
        let root = PathBuf::from(arg_value(&argv, "--root").unwrap_or_else(|| {
            env::var("HCLI_FLASH_NEXT_ROOT").unwrap_or_else(|_| DEFAULT_ROOT.to_owned())
        }))
        .canonicalize()?;
        let out = PathBuf::from(arg_value(&argv, "--out").unwrap_or_else(|| {
            repo_root()
                .join("receipts/headless/FLASH_STATEFUL_COMPLETE_TOKEN_SESSION.json")
                .display()
                .to_string()
        }));
        // Session-local artifacts must be keyed by the receipt name.  A
        // route-trace control and an accepted-decode receipt can therefore be
        // compared or rolled back without one run rewriting the other's KV
        // and terminal evidence.
        let out_stem = out
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("FLASH_STATEFUL_COMPLETE_TOKEN_SESSION");
        let out_dir = out
            .parent()
            .unwrap_or_else(|| Path::new("receipts/headless"))
            .join(format!("{out_stem}.session_artifacts"));
        fs::create_dir_all(&out_dir)?;
        let token_ids = parse_token_ids(&argv)?;
        if token_ids.len() < 2 {
            return Err("complete session requires prompt plus candidate token".into());
        }
        let prompt_len = parse_prompt_len(&argv, token_ids.len())?;
        let route_teacher = parse_route_teacher(&argv, &token_ids)?;
        let retain_linear_banks = argv.iter().any(|arg| arg == "--retain-linear-banks");
        let device_only_compact_banks = argv.iter().any(|arg| arg == "--device-only-compact-banks");
        let token_major_resident_banks = argv.iter().any(|arg| arg == "--token-major-resident-banks");
        let token_major_host_attention_seam = argv.iter().any(|arg| arg == "--token-major-host-attention-seam");
        let token_major_host_linear_seam = argv.iter().any(|arg| arg == "--token-major-host-linear-seam");
        let token_major_allow_state_drift_candidate = argv
            .iter()
            .any(|arg| arg == "--token-major-allow-state-drift-candidate");
        let clean_replay_reps = arg_value(&argv, "--token-major-clean-replay-reps")
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(0);
        let resident_bank_limit = arg_value(&argv, "--token-major-construction-limit")
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(48);
        if device_only_compact_banks && route_teacher.is_none() {
            return Err("--device-only-compact-banks requires --route-teacher".into());
        }
        if token_major_resident_banks && route_teacher.is_some()
            && (!retain_linear_banks || !device_only_compact_banks)
        {
            return Err("teacher-bound --token-major-resident-banks requires --retain-linear-banks --device-only-compact-banks".into());
        }
        if token_major_host_attention_seam && !token_major_resident_banks {
            return Err("--token-major-host-attention-seam requires --token-major-resident-banks".into());
        }
        if token_major_host_linear_seam && !token_major_resident_banks {
            return Err("--token-major-host-linear-seam requires --token-major-resident-banks".into());
        }
        if clean_replay_reps > 0 && !token_major_resident_banks {
            return Err("--token-major-clean-replay-reps requires --token-major-resident-banks".into());
        }
        if token_major_allow_state_drift_candidate
            && (!token_major_resident_banks || route_teacher.is_none())
        {
            return Err("--token-major-allow-state-drift-candidate requires --token-major-resident-banks and --route-teacher".into());
        }
        if resident_bank_limit == 0 || resident_bank_limit > 48 {
            return Err("--token-major-construction-limit must be in 1..=48".into());
        }
        if resident_bank_limit != 48 && !token_major_resident_banks {
            return Err("--token-major-construction-limit requires --token-major-resident-banks".into());
        }
        let vocab = vocab(&root)?;
        // Keep the immutable source index and Metal context alive for the
        // entire native session. Linear segments still release their layer
        // weights at species seams, but no longer reopen the source or create
        // a fresh Metal device/queue for every segment.
        let linear_index = SourceBf16Index::open(&root)?;
        let linear_context = MetalContext::new_with_trace(true)?;
        let started = Instant::now();
        let mut retained_linear_banks: Vec<Vec<linear::StatefulLinearLayer>> = Vec::new();
        let mut token_major_banks: Option<Vec<ResidentTokenMajorBank>> = None;
        let (final_states, segments) = if token_major_resident_banks {
            let (states, receipts, banks) = run_token_major_resident_session(
                &linear_index,
                &linear_context,
                &token_ids,
                vocab,
                route_teacher.as_ref(),
                token_major_host_attention_seam,
                token_major_host_linear_seam,
                resident_bank_limit,
                !token_major_allow_state_drift_candidate,
            )?;
            token_major_banks = Some(banks);
            (states, receipts)
        } else {
            let mut current_states: Option<Vec<Vec<f32>>> = None;
            let mut segments = Vec::new();
            let mut cursor = 0usize;
            for &full_layer in FULL_LAYERS.iter() {
                if cursor < full_layer {
                    let end = full_layer - 1;
                    let (states, receipt, banks) = run_linear_segment(
                        &linear_index,
                        &linear_context,
                        cursor,
                        end,
                        &token_ids,
                        vocab,
                        current_states.as_deref(),
                        route_teacher.as_ref(),
                        device_only_compact_banks,
                    )?;
                    current_states = Some(states);
                    segments.push(receipt);
                    if retain_linear_banks {
                        retained_linear_banks.push(banks);
                    }
                }
                let input = current_states.take().ok_or(format!(
                    "missing input state before full-attention layer {full_layer}"
                ))?;
                let (states, receipt) = run_full_layer(
                    &linear_index,
                    &linear_context,
                    full_layer,
                    &token_ids,
                    &input,
                    &out_dir,
                    route_teacher.as_ref(),
                )?;
                current_states = Some(states);
                segments.push(receipt);
                cursor = full_layer + 1;
            }
            if cursor < 48 {
                let (states, receipt, banks) = run_linear_segment(
                    &linear_index,
                    &linear_context,
                    cursor,
                    47,
                    &token_ids,
                    vocab,
                    current_states.as_deref(),
                    route_teacher.as_ref(),
                    device_only_compact_banks,
                )?;
                current_states = Some(states);
                segments.push(receipt);
                if retain_linear_banks {
                    retained_linear_banks.push(banks);
                }
            }
            (
                current_states.ok_or("complete session produced no final states")?,
                segments,
            )
        };
        if final_states.len() != token_ids.len() {
            return Err("complete session final state/token count mismatch".into());
        }
        let terminal_dir = out_dir.join("terminal");
        fs::create_dir_all(&terminal_dir)?;
        let terminal_executor = terminal::TerminalExecutor::new(root.clone())?;
        let mut reference_checks = Vec::new();
        for (generation_index, &expected_token) in token_ids[prompt_len..].iter().enumerate() {
            let state_index = prompt_len - 1 + generation_index;
            let (predicted_token, terminal) = terminal_token(
                &terminal_executor,
                &final_states[state_index],
                &terminal_dir.join(format!("reference-{generation_index}-input.state.f32")),
            )?;
            reference_checks.push(json!({
                "generation_index": generation_index,
                "input_state_index": state_index,
                "expected_token_id": expected_token,
                "predicted_token_id": predicted_token,
                "accepted": predicted_token == expected_token,
                "terminal": terminal,
            }));
        }
        let accepted_generation_tokens = reference_checks
            .iter()
            .take_while(|check| check.get("accepted") == Some(&Value::Bool(true)))
            .count();
        let all_reference_tokens_accepted = accepted_generation_tokens == reference_checks.len();
        let final_state_index = final_states.len() - 1;
        let (next_after_reference, final_terminal) = terminal_token(
            &terminal_executor,
            &final_states[final_state_index],
            &terminal_dir.join("final-reference.state.f32"),
        )?;
        let candidate = token_ids[prompt_len];
        let predicted_candidate = reference_checks[0]
            .get("predicted_token_id")
            .and_then(Value::as_u64)
            .ok_or("first reference check omitted predicted token")?
            as usize;
        let first_reference_terminal = reference_checks[0]
            .get("terminal")
            .cloned()
            .ok_or("first reference check omitted terminal receipt")?;
        let multi_token_reference = reference_checks.len() >= 2;
        let async_linear_submit = matches!(
            env::var("HAWKING_FLASH_ASYNC_LINEAR_SUBMIT").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        );
        let linear_region_submit = matches!(
            env::var("HAWKING_FLASH_LINEAR_REGION_SUBMIT").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        );
        let clean_replay = if clean_replay_reps > 0 {
            if !all_reference_tokens_accepted || !multi_token_reference {
                return Err("clean token-major replay requires an accepted multi-token reference control".into());
            }
            let input_prepare_started = Instant::now();
            let source_before = linear_index.bytes_read_total();
            let replay_bases = token_ids
                .iter()
                .map(|&token_id| embedding_row(&linear_index, token_id, vocab).map(|row| repeated_streams(&row)))
                .collect::<Result<Vec<_>, _>>()?;
            let input_prepare_ns = input_prepare_started.elapsed().as_nanos() as u64;
            let input_source_bytes = linear_index.bytes_read_total().saturating_sub(source_before);
            let banks = token_major_banks
                .as_mut()
                .ok_or("clean token-major replay lost its resident banks")?;
            let mut replay = run_token_major_clean_replay(
                banks,
                &linear_context,
                &replay_bases,
                &token_ids,
                prompt_len,
                clean_replay_reps,
                async_linear_submit,
                linear_region_submit,
            )?;
            replay["input_prepare_ns_outside_timed_window"] = Value::from(input_prepare_ns);
            replay["input_embedding_source_bytes_outside_timed_window"] = Value::from(input_source_bytes);
            Some(replay)
        } else {
            None
        };
        let status = if token_major_allow_state_drift_candidate && all_reference_tokens_accepted && multi_token_reference {
            "CANDIDATE_ROUTE_EXACT_TERMINAL_ACCEPTED_STATE_HASH_WITHHELD"
        } else if all_reference_tokens_accepted && multi_token_reference {
            "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE"
        } else if all_reference_tokens_accepted {
            "PASSED_STATEFUL_COMPLETE_TOKEN_SESSION"
        } else {
            "PASSED_COMPLETE_FORWARD_REFERENCE_REJECTED"
        };
        let linear_persistent_bytes = segments
            .iter()
            .filter_map(|segment| {
                segment
                    .pointer("/persistent_state/total_bytes")
                    .and_then(Value::as_u64)
            })
            .sum::<u64>();
        let attention_persistent_bytes = segments
            .iter()
            .filter_map(|segment| {
                segment
                    .pointer("/state_memory/persistent_kv_bytes")
                    .and_then(Value::as_u64)
            })
            .sum::<u64>();
        let attention_growth_bytes_per_token = segments
            .iter()
            .filter_map(|segment| {
                segment
                    .pointer("/state_memory/growth_bytes_per_token")
                    .and_then(Value::as_u64)
            })
            .sum::<u64>();
        let source_payload_bytes_read = segments
            .iter()
            .filter_map(|segment| {
                segment
                    .get("source_payload_bytes_read")
                    .and_then(Value::as_u64)
            })
            .sum::<u64>();
        let total_persistent_bytes =
            linear_persistent_bytes.saturating_add(attention_persistent_bytes);
        // Keep the token-major owner alive through terminal verification.  Its
        // count is deliberately captured from the actual owning vector, not a
        // requested flag, so a receipt cannot claim a body that has dropped.
        let token_major_bank_count = token_major_banks.as_ref().map(Vec::len).unwrap_or(0);
        let retained_linear_layers = if token_major_resident_banks {
            segments
                .iter()
                .filter(|segment| segment.get("species") == Some(&Value::String("linear_attention".to_owned())))
                .count()
        } else {
            retained_linear_banks.iter().map(Vec::len).sum::<usize>()
        };
        let retained_linear_source_bytes = if retain_linear_banks && !device_only_compact_banks {
            segments
                .iter()
                .filter(|segment| segment.get("species") == Some(&Value::String("linear_attention".to_owned())))
                .filter_map(|segment| segment.get("source_payload_bytes_read").and_then(Value::as_u64))
                .sum::<u64>()
        } else {
            0
        };
        let retained_linear_device_bytes = if retain_linear_banks {
            segments
                .iter()
                .filter(|segment| segment.get("species") == Some(&Value::String("linear_attention".to_owned())))
                .filter_map(|segment| {
                    segment
                        .get("device_weight_bytes")
                        .or_else(|| segment.get("source_payload_bytes_read"))
                        .and_then(Value::as_u64)
                })
                .sum::<u64>()
        } else {
            0
        };
        let resident_full_attention_layers = segments
            .iter()
            .filter(|segment| segment.get("species") == Some(&Value::String("full_attention".to_owned())))
            .filter(|segment| {
                segment.get("immutable_weight_ownership")
                    == Some(&Value::String("device_only_after_source_upload".to_owned()))
                    && segment.get("host_source_weights_retained_during_token_loop")
                        == Some(&Value::Bool(false))
            })
            .count();
        let resident_full_attention_device_bytes = segments
            .iter()
            .filter(|segment| segment.get("species") == Some(&Value::String("full_attention".to_owned())))
            .filter_map(|segment| segment.get("device_weight_bytes").and_then(Value::as_u64))
            .sum::<u64>();
        let resident_rss_bytes = if retain_linear_banks || token_major_resident_banks {
            current_rss_bytes()
        } else {
            None
        };
        let first_rejection = reference_checks
            .iter()
            .find(|check| check.get("accepted") != Some(&Value::Bool(true)))
            .cloned();
        let mut receipt = json!({
            "schema": "hawking.flash.stateful_complete_token_session.v1",
            "status": status,
            "model": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "root": root,
            "token_ids": token_ids,
            "prompt_token_ids": &token_ids[..prompt_len],
            "reference_generated_token_ids": &token_ids[prompt_len..],
            "candidate_token_id": candidate,
            "vocab_size": vocab,
            "segments": segments,
            "terminal": {
                "prompt_final": first_reference_terminal,
                "reference_checks": reference_checks,
                "predicted_candidate": predicted_candidate,
                "next_after_reference": next_after_reference,
                "final_reference": final_terminal,
                "candidate_accepted": all_reference_tokens_accepted,
            },
            "execution": {
                "process_boundary": "one native process",
                "terminal_executor": {
                    "lifetime": "one session",
                    "source_index_reused": true,
                    "readout_weights_reused": true,
                    "lm_head_reused": true,
                    "source_payload_bytes_read": terminal_executor.source_bytes_read(),
                    "device": terminal_executor.device_name(),
                },
                "linear_recurrence_segments": true,
                "full_attention_kv_segments": true,
                "cross_species_activation_handoff": if token_major_resident_banks && token_major_host_attention_seam { "host diagnostic vectors at full-attention seams; device buffers elsewhere" } else if token_major_resident_banks { "device buffers; host snapshots are diagnostic-only and never feed a later layer" } else { "host diagnostic vectors" },
                "reference_contract": "each caller-supplied continuation token is independently compared with the terminal argmax of its preceding exact 48-layer state",
                "per_layer_state_hash_contract": if token_major_allow_state_drift_candidate { "withheld for candidate reduction-order experiment; exact routes and source-terminal token checks remain enforced" } else { "exact teacher state hash enforced at every layer/token slot" },
                "source_reset_or_reprefill": false,
                "state_reset_scope": "linear species reset only before session token zero; full-attention KV buffers allocated once per layer and indexed by the advancing token position",
                "state_memory": {
                    "linear_persistent_bytes": linear_persistent_bytes,
                    "full_attention_persistent_bytes": attention_persistent_bytes,
                    "total_persistent_bytes": total_persistent_bytes,
                    "growth_bytes_per_additional_token": attention_growth_bytes_per_token,
                    "session_token_slots": token_ids.len(),
                },
                "linear_compact_bank_residency": {
                    "requested": retain_linear_banks,
                    "status": if token_major_resident_banks { "ALL_LINEAR_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE" } else if retain_linear_banks && device_only_compact_banks { "PARTIAL_PROCESS_LIFETIME_DEVICE_ONLY_LINEAR_BANKS_RETAINED" } else if retain_linear_banks { "PARTIAL_PROCESS_LIFETIME_LINEAR_BANKS_RETAINED" } else { "NOT_REQUESTED" },
                    "retained_linear_layers": retained_linear_layers,
                    "retained_linear_source_weight_bytes": retained_linear_source_bytes,
                    "retained_linear_device_weight_bytes": retained_linear_device_bytes,
                    "immutable_weight_ownership": if device_only_compact_banks { "device_only_after_source_upload" } else { "host_and_device" },
                    "observed_process_rss_bytes": resident_rss_bytes,
                    "claim_boundary": if token_major_resident_banks { "All exact teacher-bound compact linear banks remain device-only through the accepted token sequence as part of the 48-layer token-major owner. This is a correctness control with per-layer diagnostics, not a clean warmed TPS measurement." } else if device_only_compact_banks { "This retains only device-side exact linear compact banks through this process lifetime; the source-side linear weights are released after upload. Full-attention banks are separately measured per layer; token-major whole-model resident decode remains unimplemented, so this is not warmed whole-model decode or TPS." } else { "This retains only the exact linear compact banks through this process lifetime. Full-attention banks are separately measured per layer; token-major whole-model resident decode remains unimplemented, so this is not warmed whole-model decode or TPS." },
                },
                "full_attention_compact_bank_residency": {
                    "status": if token_major_resident_banks && token_major_host_attention_seam && resident_full_attention_layers == FULL_LAYERS.len() { "ALL_FULL_ATTENTION_BANKS_RETAINED__HOST_SEAM_DIAGNOSTIC" } else if token_major_resident_banks && resident_full_attention_layers == FULL_LAYERS.len() { "ALL_FULL_ATTENTION_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE" } else if resident_full_attention_layers == FULL_LAYERS.len() { "PER_LAYER_DEVICE_ONLY_FULL_ATTENTION_BANKS_REUSED_ACROSS_SESSION_TOKENS" } else { "INCOMPLETE" },
                    "resident_full_attention_layers": resident_full_attention_layers,
                    "expected_full_attention_layers": FULL_LAYERS.len(),
                    "device_weight_bytes": resident_full_attention_device_bytes,
                    "host_source_weights_retained_during_token_loop": false,
                    "process_lifetime_all_banks_retained": token_major_resident_banks,
                    "claim_boundary": if token_major_resident_banks { "All exact teacher-bound compact full-attention banks retain device-only weights and persistent KV state while the token-major 48-layer body executes. This control still includes per-layer snapshots and route checks, so it is not a clean warmed TPS measurement." } else { "Each full-attention bank is reused across the session token slots and releases source tensors after upload, but is released after its layer-major pass. A token-major whole-model resident owner is still required before warm complete-model decode or TPS can be claimed." }
                },
                "token_major_resident_banks": {
                    "requested": token_major_resident_banks,
                    "status": if token_major_resident_banks && token_major_host_attention_seam && token_major_bank_count == 48 { "ALL_48_BANKS_RETAINED__HOST_SEAM_DIAGNOSTIC" } else if token_major_resident_banks && token_major_bank_count == 48 { "ALL_48_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE" } else if token_major_resident_banks { "INCOMPLETE" } else { "NOT_REQUESTED" },
                    "resident_layers": token_major_bank_count,
                    "expected_layers": 48,
                    "cross_layer_activation_handoff": if token_major_resident_banks && token_major_host_attention_seam { "host diagnostic vector at full-attention seams" } else if token_major_resident_banks { "device buffers" } else { "not_applicable" },
                    "source_reset_or_reprefill": false,
                    "claim_boundary": if token_major_resident_banks { "Every immutable compact bank is owned by one token-major native body until after the terminal checks. The retained teacher route union is bounded to the recorded sequence, and per-layer diagnostics remain enabled; no warm TPS, future-route coverage, EBPW, capability, or resident-promotion claim follows." } else { "Not requested." },
                },
                "source_payload_bytes_read": source_payload_bytes_read,
                "source_payload_bytes_per_session_token": source_payload_bytes_read
                    .checked_div(token_ids.len() as u64),
                "source_independent": false,
                "expert_bank_mode": if route_teacher.is_some() { "route_union_compact_teacher_bound" } else { "dense" },
                "route_teacher": if route_teacher.is_some() { "validated exact dense session" } else { "none" },
                "fallback_count": 0,
                "elapsed_wall_ns": started.elapsed().as_nanos() as u64,
            },
            "accepted_generation_tokens": accepted_generation_tokens,
            "accepted_tps": Value::Null,
            "clean_replay": clean_replay,
            "physical_configuration": {
                "source_bf16_vec4": hawking_core::env_on("HAWKING_FLASH_BF16_VEC4"),
                "source_bf16_geo": hawking_core::env_on("HAWKING_FLASH_BF16_GEO"),
                "source_bf16_geo_dual": hawking_core::env_on("HAWKING_FLASH_BF16_GEO_DUAL"),
                "hyperconnection_split": hawking_core::env_on("HAWKING_FLASH_HC_SPLIT"),
                "hyperconnection_fused_up_silu": hawking_core::env_on("HAWKING_FLASH_HC_FUSE_UP_SILU"),
                "hyperconnection_router_fused": hawking_core::env_on("HAWKING_FLASH_HC_ROUTER_FUSED"),
                "router_topk_fused": hawking_core::env_on("HAWKING_FLASH_ROUTER_TOPK_FUSED"),
                "compact_moe_vec4": hawking_core::env_on("HAWKING_FLASH_MOE_VEC4"),
                "compact_moe_geo": hawking_core::env_on("HAWKING_FLASH_MOE_GEO") && !hawking_core::env_on("HAWKING_FLASH_MOE_VEC4"),
                "compact_moe_gateup_geo": hawking_core::env_on("HAWKING_FLASH_MOE_GATEUP_GEO"),
                "deltanet_value_parallel_exact": hawking_core::env_on("HAWKING_FLASH_DELTANET_VI_EXACT"),
                "deltanet_value_parallel_simd_candidate": hawking_core::env_on("HAWKING_FLASH_DELTANET_VI_SIMD"),
                "queue_ordered_linear_submit": hawking_core::env_on("HAWKING_FLASH_ASYNC_LINEAR_SUBMIT"),
                "linear_region_submit": hawking_core::env_on("HAWKING_FLASH_LINEAR_REGION_SUBMIT"),
                "claim_boundary": "This records selected opt-in physical kernels for reproducibility. A true flag is not a qualification or promotion claim; the receipt contract remains authoritative."
            },
            "complete_system_ebpw": Value::Null,
            "promotion_allowed": false,
            "first_physical_failure_boundary": if all_reference_tokens_accepted { Value::Null } else { json!({"stage": "reference_token_acceptance", "first_rejection": first_rejection, "reason": "complete forward retained state through the supplied sequence, but a caller-supplied reference continuation token did not match the preceding terminal argmax"}) },
            "claim_boundary": if token_major_allow_state_drift_candidate && all_reference_tokens_accepted && multi_token_reference && route_teacher.is_some() {
                "This is a non-bitwise physical candidate: routes and consecutive source-terminal tokens passed, but the exact per-layer state-hash contract was intentionally withheld because its reduction order differs. It cannot replace the exact resident body or support capability, EBPW, qualified TPS, or promotion claims without a separately earned tolerance/capability contract."
            } else if token_major_resident_banks && token_major_host_attention_seam && all_reference_tokens_accepted && multi_token_reference && route_teacher.is_some() {
                "A token-major diagnostic retained all compact banks but deliberately restored the prior host activation vector at each full-attention seam. It distinguishes seam parity from bank/state ordering only; it is not device-only token-major residency or TPS."
            } else if token_major_resident_banks && all_reference_tokens_accepted && multi_token_reference && route_teacher.is_some() {
                "A token-major complete 48-layer Flash body retained all exact teacher-bound compact immutable banks device-only across the recorded accepted sequence. Every inter-layer activation handoff used a device buffer; linear recurrent state and full-attention KV state survived token boundaries; every route and continuation terminal was independently checked. This is bounded to the recorded teacher sequence and contains diagnostic snapshots/route checks, so it does not prove future-route coverage, clean warmed TPS, EBPW, capability, source-independent NX, or resident promotion."
            } else if all_reference_tokens_accepted && multi_token_reference && route_teacher.is_some() {
                "A complete 48-layer stateful Flash forward used only an exact dense-teacher route union for each routed expert bank, retained linear recurrent and full-attention KV state across multiple caller-supplied reference continuation tokens in one native process, and checked every observed route plus each token against its preceding terminal argmax. This is bounded to the recorded teacher sequence; cross-species seams still use host diagnostic vectors, and it does not prove future-route coverage, source-independent NX, TPS, EBPW, capability, or HCLI residency."
            } else if all_reference_tokens_accepted && multi_token_reference {
                "A complete 48-layer stateful Flash forward retained linear recurrent and full-attention KV state across multiple caller-supplied reference continuation tokens in one native process, checking each token against its preceding terminal argmax. Cross-species seams still use host diagnostic vectors; this does not prove source-independent NX, TPS, EBPW, capability, or HCLI residency."
            } else if all_reference_tokens_accepted {
                "A complete 48-layer stateful Flash forward accepted one tokenizer-bound reference candidate token in one native process. Cross-species seams still use host diagnostic vectors; this does not prove repeated decode, source-independent NX, TPS, EBPW, capability, or HCLI residency."
            } else {
                "A complete 48-layer stateful Flash forward retained state through the supplied reference sequence, but a reference token was rejected by terminal argmax. This is a precise continuation diagnosis, not a TPS, EBPW, capability, or residency claim."
            },
            "next": if token_major_resident_banks && all_reference_tokens_accepted && multi_token_reference { "Use this exact resident body to construct a separate clean repeated timing path that removes per-layer host snapshots, route comparisons, and terminal calls from the timed loop while retaining source, state, dispatch, and bytes/token census." } else if all_reference_tokens_accepted && multi_token_reference && route_teacher.is_some() { "Keep the teacher-bound compact banks resident, separate source/index/upload/pipeline/wait/state timing, then measure warmed repeated decode without reloading the banks." } else if all_reference_tokens_accepted && multi_token_reference { "Run clean repeated timing, then census state/dispatch/bytes-per-token before selecting the highest-value physical or representation optimization." } else if all_reference_tokens_accepted { "Supply one more terminal-derived reference token with --prompt-length fixed at the original prompt boundary, then rerun the same persistent session." } else { "Replace the rejected reference token with the recorded preceding terminal argmax and rerun without changing the prompt boundary or source seal." },
            "bench": {"state": "UNKNOWN", "recorded_at": format!("unix-ms:{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis()), "recorded_by": "flash_stateful_complete_token_session", "machine": "Apple Metal", "rule": "S032 §3 -- complete-session timing is not a promotion benchmark"},
        });
        receipt["seal_sha256"] = Value::String(sha256(&serde_json::to_vec(&receipt)?));
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&out, serde_json::to_vec_pretty(&receipt)?)?;
        println!("{}", serde_json::to_string_pretty(&receipt)?);
        Ok(())
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        main_impl()
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}
