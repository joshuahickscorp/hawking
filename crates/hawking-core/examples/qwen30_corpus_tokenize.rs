//! Offline Qwen30 corpus tokenizer bridge.
//!
//! Reads a JSON array of `{"probe_id","domain","text"}` rows, applies the same
//! one-user native chat framing used by the sealed broad-activation capture
//! (`<|im_start|>user\\n{text}<|im_end|>\\n<|im_start|>assistant\\n`), encodes
//! with the runtime's own `hawking_core::tokenizer::Tokenizer`, and emits
//! receipt fields that must match the Python prepare script:
//!
//! - `token_ids_u32le_sha256`: sha256 of ids packed little-endian u32
//! - `user_text_sha256`: sha256 of the raw UTF-8 user text bytes
//!
//! No model execution. Absolute paths only.

use hawking_core::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

/// `<|im_start|>` in the Qwen3-Coder / Qwen2.5 chat vocabulary.
const IM_START_TOKEN_ID: u32 = 151644;

struct Arguments {
    tokenizer_json: PathBuf,
    in_json: PathBuf,
    out_json: PathBuf,
}

#[derive(Debug, Deserialize)]
struct InputRow {
    probe_id: String,
    domain: String,
    text: String,
}

#[derive(Debug, Serialize)]
struct OutputRow {
    probe_id: String,
    domain: String,
    token_ids: Vec<u32>,
    token_count: usize,
    token_ids_u32le_sha256: String,
    user_text_sha256: String,
}

fn usage() -> &'static str {
    "usage: qwen30_corpus_tokenize \
        --tokenizer-json ABSOLUTE_PATH \
        --in-json ABSOLUTE_PATH \
        --out-json ABSOLUTE_PATH"
}

fn required_absolute_path(value: Option<PathBuf>, flag: &str) -> Result<PathBuf, String> {
    let value = value.ok_or_else(|| format!("missing {flag}; {}", usage()))?;
    if !value.is_absolute() {
        return Err(format!("{flag} must be an absolute path; {}", usage()));
    }
    Ok(value)
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut tokenizer_json = None;
    let mut in_json = None;
    let mut out_json = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag:?}; {}", usage()))?;
        match flag.as_str() {
            "--tokenizer-json" => {
                if tokenizer_json.replace(PathBuf::from(value)).is_some() {
                    return Err(format!(
                        "--tokenizer-json was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--in-json" => {
                if in_json.replace(PathBuf::from(value)).is_some() {
                    return Err(format!(
                        "--in-json was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--out-json" => {
                if out_json.replace(PathBuf::from(value)).is_some() {
                    return Err(format!(
                        "--out-json was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            other => {
                return Err(format!("unknown flag {other:?}; {}", usage()));
            }
        }
    }
    Ok(Arguments {
        tokenizer_json: required_absolute_path(tokenizer_json, "--tokenizer-json")?,
        in_json: required_absolute_path(in_json, "--in-json")?,
        out_json: required_absolute_path(out_json, "--out-json")?,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Match Python `token_ids_u32le_sha256`: sha256 of little-endian u32 packing.
fn token_ids_u32le_sha256(ids: &[u32]) -> String {
    let mut packed = Vec::with_capacity(ids.len() * 4);
    for &id in ids {
        packed.extend_from_slice(&id.to_le_bytes());
    }
    sha256_hex(&packed)
}

/// Match Python `one_user_native_prompt` framing.
fn render_one_user_native_prompt(user_text: &str) -> String {
    format!("<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n")
}

fn encode_one_user_native(tokenizer: &Tokenizer, user_text: &str) -> Result<Vec<u32>, String> {
    let rendered = render_one_user_native_prompt(user_text);
    // Special tokens are already present as literal text in the rendered
    // string.  `add_special_tokens=true` matches the sealed prepare receipts
    // (`add_special_tokens: true`) and the complete runtime encode path.
    // `Tokenizer::from_file` sets add_bos=false, so this does not double-BOS.
    let ids = tokenizer
        .encode(&rendered, true)
        .map_err(|e| format!("encode failed: {e}"))?;
    if ids.is_empty() {
        return Err("empty tokenization".into());
    }
    if ids[0] != IM_START_TOKEN_ID {
        return Err(format!(
            "expected chat start token {IM_START_TOKEN_ID} (<|im_start|>), got {}",
            ids[0]
        ));
    }
    Ok(ids)
}

fn load_input_rows(path: &Path) -> Result<Vec<InputRow>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("failed to read --in-json {}: {e}", path.display()))?;
    let value: Value = serde_json::from_str(&raw)
        .map_err(|e| format!("--in-json is not valid JSON ({}): {e}", path.display()))?;
    let arr = value
        .as_array()
        .ok_or_else(|| format!("--in-json must be a JSON array ({})", path.display()))?;
    let mut rows = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        let probe_id = item
            .get("probe_id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("row {i}: missing non-empty probe_id"))?
            .to_string();
        let domain = item
            .get("domain")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("row {i} ({probe_id}): missing non-empty domain"))?
            .to_string();
        let text = item
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("row {i} ({probe_id}): missing text string"))?
            .to_string();
        rows.push(InputRow {
            probe_id,
            domain,
            text,
        });
    }
    Ok(rows)
}

fn main() {
    let arguments = match parse_arguments() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            process::exit(2);
        }
    };

    if !arguments.tokenizer_json.is_file() {
        eprintln!(
            "tokenizer missing: {}",
            arguments.tokenizer_json.display()
        );
        process::exit(2);
    }
    if !arguments.in_json.is_file() {
        eprintln!("in-json missing: {}", arguments.in_json.display());
        process::exit(2);
    }

    let tokenizer = match Tokenizer::from_file(&arguments.tokenizer_json) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "failed to load tokenizer {}: {e}",
                arguments.tokenizer_json.display()
            );
            process::exit(2);
        }
    };

    let input_rows = match load_input_rows(&arguments.in_json) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            process::exit(2);
        }
    };
    if input_rows.is_empty() {
        eprintln!("--in-json contains no rows");
        process::exit(2);
    }

    let mut output: Vec<OutputRow> = Vec::with_capacity(input_rows.len());
    let mut total_tokens: usize = 0;
    for row in input_rows {
        let ids = match encode_one_user_native(&tokenizer, &row.text) {
            Ok(ids) => ids,
            Err(e) => {
                eprintln!("probe {}: {e}", row.probe_id);
                process::exit(2);
            }
        };
        let token_count = ids.len();
        total_tokens += token_count;
        output.push(OutputRow {
            probe_id: row.probe_id,
            domain: row.domain,
            token_ids_u32le_sha256: token_ids_u32le_sha256(&ids),
            user_text_sha256: sha256_hex(row.text.as_bytes()),
            token_ids: ids,
            token_count,
        });
    }

    if let Some(parent) = arguments.out_json.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = fs::create_dir_all(parent) {
                eprintln!(
                    "failed to create parent dir for {}: {e}",
                    arguments.out_json.display()
                );
                process::exit(2);
            }
        }
    }

    let body = match serde_json::to_string_pretty(&output) {
        Ok(s) => s + "\n",
        Err(e) => {
            eprintln!("failed to serialize output: {e}");
            process::exit(2);
        }
    };
    if let Err(e) = fs::write(&arguments.out_json, body.as_bytes()) {
        eprintln!(
            "failed to write --out-json {}: {e}",
            arguments.out_json.display()
        );
        process::exit(2);
    }

    println!(
        "{}",
        serde_json::json!({
            "out_json": arguments.out_json.display().to_string(),
            "probe_count": output.len(),
            "total_tokens": total_tokens,
        })
    );
}
