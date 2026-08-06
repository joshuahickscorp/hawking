//! Bounded CPU source-algorithm receipt for the DeepSeek-V4 layer-0 position-1
//! ratio-zero causal KV transition.  It is not an upstream-runtime parity,
//! Metal result, full model forward, endpoint, or TPS claim.

use hawking_core::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, FULL_STREAM_SCHEMA, FULL_STREAM_STATUS,
};
use hawking_core::gravity_deepseek_v4_layer0_attention::{HEAD_DIM, NUM_HEADS, ROPE_HEAD_DIM};
use hawking_core::gravity_deepseek_v4_layer0_attention::{WO_A_ROWS, WO_B_ROWS};
use hawking_core::gravity_deepseek_v4_layer0_continuation::{
    layer0_position1_complete_attention_cpu_oracle, layer0_position1_continuation_cpu_oracle,
    verify_layer0_position1_continuation_anchors, Layer0Position1CompleteAttentionCpuOracleResult,
    Layer0Position1ContinuationCpuOracleResult, POSITION1, POSITION1_KV_ROWS, POSITION1_TOKEN_ID,
    POSITION1_TOKEN_STRING,
};
use hawking_core::gravity_deepseek_v4_layer0_prefix::{HC_MULT, HIDDEN_SIZE, PREFIX_TOKEN_ID};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "hawking.gravity.deepseek_v4.layer0_position1_continuation_cpu_oracle.v1";
const STATUS: &str = "PASS_SOURCE_DERIVED_CPU_LAYER0_POSITION1_CAUSAL_KV_NOT_RUNTIME";
const COMPLETE_SCHEMA: &str =
    "hawking.gravity.deepseek_v4.layer0_position1_complete_attention_cpu_oracle.v1";
const COMPLETE_STATUS: &str =
    "PASS_SOURCE_DERIVED_CPU_LAYER0_POSITION1_COMPLETE_ATTENTION_NOT_RUNTIME";

type ExampleResult<T> = Result<T, Box<dyn Error>>;

struct Args {
    artifact: PathBuf,
    out: PathBuf,
    complete_attention: bool,
}

fn main() -> ExampleResult<()> {
    let args = parse_args()?;
    let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
    verify_layer0_position1_continuation_anchors(&reader)?;
    if args.complete_attention {
        let result = layer0_position1_complete_attention_cpu_oracle(&reader)?;
        check_complete(&result)?;
        let causal = &result.causal;
        let unsigned = json!({
            "schema":COMPLETE_SCHEMA,
            "status":COMPLETE_STATUS,
            "artifact":{"path":reader.artifact_root().display().to_string(),"full_stream_schema":FULL_STREAM_SCHEMA,"full_stream_status":FULL_STREAM_STATUS,"manifest_seal_sha256":reader.manifest_seal_sha256(),"manifest_file_sha256":reader.manifest_file_sha256(),"restart_receipt_seal_sha256":reader.restart_seal_sha256(),"source_parent_retained":false},
            "source":{"repository":reader.source_identity().repository,"revision":reader.source_identity().revision,"assets":{"tokenizer.json":reader.source_metadata_asset_sha256("tokenizer.json")?,"inference/model.py":reader.source_metadata_asset_sha256("inference/model.py")?,"inference/kernel.py":reader.source_metadata_asset_sha256("inference/kernel.py")?,"inference/convert.py":reader.source_metadata_asset_sha256("inference/convert.py")?,"inference/config.json":reader.source_metadata_asset_sha256("inference/config.json")?}},
            "tokenizer_trace":{"token_ids":[PREFIX_TOKEN_ID,POSITION1_TOKEN_ID],"position1_bpe":POSITION1_TOKEN_STRING,"position":POSITION1,"no_unbounded_prompt_trace_retained":true},
            "source_semantics":{"ratio":0,"window_size":128,"source_window_topk_indices_at_position1":[0,1],"compressed_or_index_branch_executed":false,"position1_rope_applied":true,"position1_inverse_rope_applied_to_attention_output":true,"sparse_value_numerator_weights_bf16_stored_before_value_gemm":true,"tail":"inverse_rope -> convert.py BF16 WO-A/einsum -> source FP8 WO-B -> mHC attention post"},
            "bounded_source_reads":{"two_embedding_rows":true,"all_touched_chunk_sha256_verified_before_cpu_use":true,"parent_safetensors_materialized":false},
            "causal_checkpoints":checkpoints(causal)?,
            "tail_checkpoints":complete_checkpoints(&result)?,
            "work_accounting":{"batch":1,"sequence_tokens":1,"position":POSITION1,"cache_writes":2,"causal_kv_reads":2,"qk_dot_products":(NUM_HEADS*HEAD_DIM*POSITION1_KV_ROWS)as u64,"wo_a_rows":WO_A_ROWS,"wo_b_rows":WO_B_ROWS,"attention_hc_post_bf16_elements":HC_MULT*HIDDEN_SIZE,"gpu_dispatches":0,"command_buffers":0,"cpu_visible_waits":0},
            "claim_boundary":"CPU-only source translation through the complete position-1 ratio-zero layer-0 attention tail. It does not establish upstream PyTorch/TileLang parity, a Metal path, the layer FFN/router/MoE, a causal runtime, generation, HCLI, or BASE_TRUE_TPS."
        });
        let receipt = seal(unsigned)?;
        write_new(&args.out, &receipt)?;
        println!(
            "{}",
            serde_json::to_string(
                &json!({"status":COMPLETE_STATUS,"receipt":args.out,"seal_sha256":receipt["seal_sha256"]})
            )?
        );
        return Ok(());
    }
    let result = layer0_position1_continuation_cpu_oracle(&reader)?;
    check(&result)?;
    let unsigned = json!({
        "schema":SCHEMA,
        "status":STATUS,
        "artifact":{"path":reader.artifact_root().display().to_string(),"full_stream_schema":FULL_STREAM_SCHEMA,"full_stream_status":FULL_STREAM_STATUS,"manifest_seal_sha256":reader.manifest_seal_sha256(),"manifest_file_sha256":reader.manifest_file_sha256(),"restart_receipt_seal_sha256":reader.restart_seal_sha256(),"source_parent_retained":false},
        "source":{"repository":reader.source_identity().repository,"revision":reader.source_identity().revision,"assets":{"tokenizer.json":reader.source_metadata_asset_sha256("tokenizer.json")?,"inference/model.py":reader.source_metadata_asset_sha256("inference/model.py")?,"inference/kernel.py":reader.source_metadata_asset_sha256("inference/kernel.py")?,"inference/config.json":reader.source_metadata_asset_sha256("inference/config.json")?}},
        "tokenizer_trace":{"token_ids":[PREFIX_TOKEN_ID,POSITION1_TOKEN_ID],"position1_bpe":POSITION1_TOKEN_STRING,"position":POSITION1,"no_unbounded_prompt_trace_retained":true},
        "source_semantics":{"ratio":0,"window_size":128,"source_window_topk_indices_at_position1":[0,1],"compressed_or_index_branch_executed":false,"position1_rope_applied":true,"position1_inverse_rope_applied_to_attention_output":true,"mHC_temporal_cache":false,"temporal_state":"only layer-0 KV cache is cross-position state in this bounded attention proof","sparse_value_numerator_weights_bf16_stored_before_value_gemm":true},
        "bounded_source_reads":{"two_embedding_rows":true,"all_touched_chunk_sha256_verified_before_cpu_use":true,"parent_safetensors_materialized":false},
        "checkpoints":checkpoints(&result)?,
        "work_accounting":{"batch":1,"sequence_tokens":1,"position":POSITION1,"cache_writes":2,"causal_kv_reads":2,"qk_dot_products":(NUM_HEADS*HEAD_DIM*POSITION1_KV_ROWS)as u64,"gpu_dispatches":0,"command_buffers":0,"cpu_visible_waits":0},
        "claim_boundary":"CPU-only source translation for the two-row position-1 ratio-zero causal KV path. It does not establish upstream PyTorch/TileLang parity, a Metal path, the WO/mHC tail at position 1, full-layer execution, a causal runtime, generation, HCLI, or BASE_TRUE_TPS."
    });
    let receipt = seal(unsigned)?;
    write_new(&args.out, &receipt)?;
    println!(
        "{}",
        serde_json::to_string(
            &json!({"status":STATUS,"receipt":args.out,"seal_sha256":receipt["seal_sha256"]})
        )?
    );
    Ok(())
}

fn check_complete(result: &Layer0Position1CompleteAttentionCpuOracleResult) -> ExampleResult<()> {
    check(&result.causal)?;
    if result.wo_a_bf16_bits.len() != WO_A_ROWS
        || result.wo_b.output.fp32.len() != WO_B_ROWS
        || result.wo_b.output.bf16_bits.len() != WO_B_ROWS
        || result.hc_attention_post_bf16_bits.len() != HC_MULT * HIDDEN_SIZE
    {
        return Err("position-1 complete attention oracle returned invalid tail geometry".into());
    }
    Ok(())
}

fn check(result: &Layer0Position1ContinuationCpuOracleResult) -> ExampleResult<()> {
    if result.token0_id != PREFIX_TOKEN_ID
        || result.token1_id != POSITION1_TOKEN_ID
        || result.token1_prefix.embed_bf16_bits.len() != HIDDEN_SIZE
        || result.q_position1_rope_bf16_bits.len() != NUM_HEADS * HEAD_DIM
        || result.kv_position1_rope_bf16_bits.len() != HEAD_DIM
        || result.kv_cache_two_rows_bf16_bits.len() != POSITION1_KV_ROWS * HEAD_DIM
        || result.sparse_attention_scores_f32.len() != NUM_HEADS * POSITION1_KV_ROWS
        || result.sparse_attention_sink_denominators_f32.len() != NUM_HEADS
        || result.sparse_attention_bf16_bits.len() != NUM_HEADS * HEAD_DIM
        || result.sparse_attention_derotated_bf16_bits.len() != NUM_HEADS * HEAD_DIM
        || result.rope_table.cos_f32.len() != ROPE_HEAD_DIM / 2
        || result.rope_table.sin_f32.len() != ROPE_HEAD_DIM / 2
        || result
            .sparse_attention_scores_f32
            .iter()
            .chain(&result.sparse_attention_sink_denominators_f32)
            .any(|value| !value.is_finite())
    {
        return Err("position-1 continuation oracle returned invalid geometry/value".into());
    }
    Ok(())
}

fn checkpoints(result: &Layer0Position1ContinuationCpuOracleResult) -> ExampleResult<Value> {
    Ok(json!({
        "token1_prefix_embed_bf16_sha256":sha256(&u16bytes(&result.token1_prefix.embed_bf16_bits)),
        "q_position1_rope_bf16_sha256":sha256(&u16bytes(&result.q_position1_rope_bf16_bits)),
        "position0_kv_bf16_sha256":sha256(&u16bytes(&result.position0_kv_rope_bf16_bits)),
        "position1_kv_bf16_sha256":sha256(&u16bytes(&result.kv_position1_rope_bf16_bits)),
        "two_row_kv_cache_bf16_sha256":sha256(&u16bytes(&result.kv_cache_two_rows_bf16_bits)),
        "sparse_scores_f32_sha256":sha256(&f32bytes(&result.sparse_attention_scores_f32)),
        "sparse_denominators_f32_sha256":sha256(&f32bytes(&result.sparse_attention_sink_denominators_f32)),
        "sparse_output_bf16_sha256":sha256(&u16bytes(&result.sparse_attention_bf16_bits)),
        "sparse_derotated_bf16_sha256":sha256(&u16bytes(&result.sparse_attention_derotated_bf16_bits)),
        "q_position1_changed_by_rope":result.q_position1_rope_bf16_bits != result.q_head_norm_bf16_bits,
        "kv_position1_changed_by_rope":result.kv_position1_rope_bf16_bits != result.kv_inplace_qat.output_bf16_bits,
        "inverse_rope_changed_attention_output":result.sparse_attention_derotated_bf16_bits != result.sparse_attention_bf16_bits,
    }))
}

fn complete_checkpoints(
    result: &Layer0Position1CompleteAttentionCpuOracleResult,
) -> ExampleResult<Value> {
    Ok(json!({
        "wo_a_bf16_sha256":sha256(&u16bytes(&result.wo_a_bf16_bits)),
        "wo_b_fp32_sha256":sha256(&f32bytes(&result.wo_b.output.fp32)),
        "wo_b_bf16_sha256":sha256(&u16bytes(&result.wo_b.output.bf16_bits)),
        "attention_hc_post_bf16_sha256":sha256(&u16bytes(&result.hc_attention_post_bf16_bits)),
    }))
}

fn parse_args() -> ExampleResult<Args> {
    let mut artifact = None;
    let mut out = None;
    let mut complete_attention = false;
    let mut arguments = std::env::args_os().skip(1);
    while let Some(flag) = arguments.next() {
        match flag.to_string_lossy().as_ref() {
            "--artifact" => artifact = arguments.next().map(PathBuf::from),
            "--out" => out = arguments.next().map(PathBuf::from),
            "--complete-attention" => complete_attention = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    Ok(Args {
        artifact: artifact.ok_or("--artifact required")?,
        out: out.ok_or("--out required")?,
        complete_attention,
    })
}

fn sha256(bytes: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(bytes);
    format!("{:x}", hash.finalize())
}

fn u16bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn seal(mut value: Value) -> ExampleResult<Value> {
    let hash = sha256(&canonical(&value));
    value
        .as_object_mut()
        .ok_or("receipt must be an object")?
        .insert("seal_sha256".into(), Value::String(hash));
    Ok(value)
}

fn canonical(value: &Value) -> Vec<u8> {
    match value {
        Value::Null => b"null".to_vec(),
        Value::Bool(value) => value.to_string().into_bytes(),
        Value::Number(value) => value.to_string().into_bytes(),
        Value::String(value) => serde_json::to_vec(value).expect("string JSON"),
        Value::Array(values) => {
            let mut output = vec![b'['];
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                output.extend(canonical(value));
            }
            output.push(b']');
            output
        }
        Value::Object(values) => {
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort();
            let mut output = vec![b'{'];
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                output.extend(serde_json::to_vec(key).expect("key JSON"));
                output.push(b':');
                output.extend(canonical(&values[key]));
            }
            output.push(b'}');
            output
        }
    }
}

fn write_new(path: &Path, value: &Value) -> ExampleResult<()> {
    if path.exists() {
        return Err(format!("refusing overwrite {}", path.display()).into());
    }
    let parent = path.parent().ok_or("out parent missing")?;
    fs::create_dir_all(parent)?;
    let temp = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|x| x.to_str())
            .ok_or("out name")?,
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    drop(file);
    fs::hard_link(&temp, path)?;
    fs::remove_file(temp)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}
