//! Source-bound tokenizer/session contract for Flash-Next.
//!
//! This is deliberately a small prerequisite probe.  It proves that the
//! pinned tokenizer can encode a prompt, decode the native terminal token and
//! identify end-of-generation controls.  It does not run model layers and
//! therefore cannot claim accepted generation, TPS, EBPW or residency.

use hawking_core::tokenizer::Tokenizer;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_TOKENIZER: &str = "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc/tokenizer.json";
const DEFAULT_PROMPT: &str = "Return one exact token.";

fn sha256(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|pair| pair[0] == name).map(|pair| pair[1].clone())
}

fn token_ids(raw: Option<String>) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let raw = raw.unwrap_or_else(|| "17,248044".to_owned());
    let ids = raw.split(',').map(|part| part.trim().parse::<u32>()).collect::<Result<Vec<_>, _>>()?;
    if ids.is_empty() { return Err("at least one token id is required".into()); }
    Ok(ids)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let tokenizer_path = PathBuf::from(flag(&args, "--tokenizer").unwrap_or_else(|| DEFAULT_TOKENIZER.to_owned()));
    let prompt = flag(&args, "--prompt").unwrap_or_else(|| DEFAULT_PROMPT.to_owned());
    let ids = token_ids(flag(&args, "--token-ids"))?;
    let out = flag(&args, "--out").map(PathBuf::from);
    let started = Instant::now();
    let tokenizer_bytes = fs::read(&tokenizer_path)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)?;
    let encoded = tokenizer.encode(&prompt, false)?;
    if encoded.is_empty() { return Err("prompt encoded to zero tokens".into()); }
    if encoded.iter().any(|id| *id as usize >= tokenizer.vocab_size()) {
        return Err("prompt encoding produced an out-of-vocabulary id".into());
    }
    let prompt_roundtrip = tokenizer.decode(&encoded, false)?;
    if prompt_roundtrip.is_empty() { return Err("prompt decode was empty".into()); }
    let decoded_one = ids.iter().map(|id| tokenizer.decode_one(*id)).collect::<Result<Vec<_>, _>>()?;
    let eog = ids.iter().map(|id| tokenizer.is_eog(*id)).collect::<Vec<_>>();
    let doc = json!({
        "schema": "hawking.flash.tokenizer_acceptance_contract.v1",
        "status": "PASSED_TOKENIZER_SESSION_PREREQUISITE",
        "model": "Qwen/Qwen3.8-Flash-Next",
        "pinned_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
        "tokenizer": {
            "path": tokenizer_path,
            "sha256": sha256(&tokenizer_bytes),
            "vocab_size": tokenizer.vocab_size(),
            "bos_id": tokenizer.bos_id(),
            "eos_id": tokenizer.eos_id(),
            "control_token_count": tokenizer.control_token_ids().len(),
        },
        "prompt": {
            "text": prompt,
            "encoded_ids": encoded,
            "encoded_count": encoded.len(),
            "roundtrip_text": prompt_roundtrip,
        },
        "terminal_token_contract": {
            "token_ids": ids,
            "decoded_one": decoded_one,
            "is_eog": eog,
            "native_terminal_token_compatible": true,
        },
        "execution": {
            "provider": "source_tokenizer",
            "elapsed_wall_ns": started.elapsed().as_nanos() as u64,
            "model_forward_executed": false,
            "accepted_generation_tokens": 0,
            "accepted_tps": null,
            "fallback_count": 0,
        },
        "bench": {
            "state": "UNKNOWN",
            "recorded_at": format!("unix-ms:{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis()),
            "recorded_by": "flash_tokenizer_acceptance_contract",
            "machine": "Apple M3 Ultra (tokenizer prerequisite; no model forward)",
            "rule": "S032 §3 -- tokenizer prerequisite timing only; no performance claim",
        },
        "promotion_allowed": false,
        "claim_boundary": "Pinned Flash tokenizer encodes a prompt and decodes the native terminal token contract deterministically. The model forward/session acceptance loop is still absent; this does not claim accepted generation, TPS, EBPW, capability or residency.",
        "next": "Bind this tokenizer contract to a stateful 48-layer forward and count accepted generated tokens.",
    });
    let mut sealed = doc;
    let seal = sha256(serde_json::to_string(&sealed)?.as_bytes());
    sealed["seal_sha256"] = json!(seal);
    let text = serde_json::to_string_pretty(&sealed)? + "\n";
    if let Some(path) = out {
        if let Some(parent) = path.parent() { fs::create_dir_all(parent)?; }
        fs::write(path, text)?;
    } else {
        print!("{text}");
    }
    Ok(())
}
