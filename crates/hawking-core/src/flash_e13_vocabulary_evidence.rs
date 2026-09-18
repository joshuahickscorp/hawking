//! Canonical CPU-only evidence producer for Flash-Next E13.
//!
//! This owner tokenizes one predeclared UTF-8 corpus with Hawking's exact
//! tokenizer, classifies independently decoded lexical tokens, and emits the
//! frequency/script/stratified-null inputs consumed by the Python admission
//! gate.  It never opens a model config, index, weight shard, tensor payload,
//! Engine, provider, GPU, or Metal device.

use crate::tokenizer::Tokenizer;
use crate::{Error, Result};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path};

pub const DECLARATION_SCHEMA: &str = "hawking.gravity.flash_e13_evidence_declaration.v1";
pub const PRODUCER_SCHEMA: &str = "hawking.gravity.flash_e13_evidence_producer.v1";
pub const LEXICAL_UNIVERSE_SCHEMA: &str = "hawking.gravity.flash_e13_lexical_universe.v1";
pub const PLAN_SCHEMA: &str = "hcli.agentos.flash_vocabulary_prefilter_plan.v1";
pub const FREQUENCY_EVIDENCE_SCHEMA: &str = "hcli.agentos.flash_vocabulary_frequency_evidence.v1";
pub const SCRIPT_EVIDENCE_SCHEMA: &str = "hcli.agentos.flash_vocabulary_script_evidence.v1";
pub const SCRIPT_CLASSIFIER_ALGORITHM: &str = "tokenizer_decoded_text_primary_script.v1";
pub const NULL_ALGORITHM: &str = "within_stratum_split_preserving_token_label_permutation.v2";
pub const FREQUENCY_PROVENANCE: &str = "tokenized_corpus_frequency.v1";
pub const SCRIPT_PROVENANCE: &str = "tokenizer_script_classification.v1";
pub const MAX_CORPUS_BYTES: u64 = 1 << 30;

static LATIN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\p{Script=Latin}").expect("valid Latin script property"));
static HAN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\p{Script=Han}").expect("valid Han script property"));

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct E13EvidenceDeclaration {
    pub schema: String,
    pub tokenizer_sha256: String,
    pub corpus_sha256: String,
    pub eligible_model_vocab_count: usize,
    pub row_budget: usize,
    pub fit_rows_per_stratum: usize,
    pub heldout_rows_per_stratum: usize,
    pub rare_max_count: u64,
    pub common_min_count: u64,
    pub short_max_utf8_bytes: usize,
    pub long_min_utf8_bytes: usize,
    pub selection_seed: u64,
    pub shuffled_null_seed: u64,
    pub source: Value,
    pub row_universe: Value,
}

#[derive(Debug, Clone)]
pub struct E13EvidenceBundle {
    pub frequency_evidence: Value,
    pub script_evidence: Value,
    pub plan: Value,
    pub producer_receipt: Value,
}

impl E13EvidenceBundle {
    pub fn frequency_bytes(&self) -> Result<Vec<u8>> {
        json_bytes(&self.frequency_evidence)
    }

    pub fn script_bytes(&self) -> Result<Vec<u8>> {
        json_bytes(&self.script_evidence)
    }

    pub fn plan_bytes(&self) -> Result<Vec<u8>> {
        json_bytes(&self.plan)
    }

    pub fn producer_receipt_bytes(&self) -> Result<Vec<u8>> {
        json_bytes(&self.producer_receipt)
    }
}

#[derive(Debug, Clone)]
struct Candidate {
    token_id: u32,
    count: u64,
    script: &'static str,
    decoded_utf8_byte_length: usize,
    decoded_token_surface_sha256: String,
    tokenizer_raw_surface_sha256: String,
    tokenizer_raw_surface_utf8_byte_length: usize,
}

#[derive(Debug, Clone, Copy)]
enum FrequencyBand {
    Rare,
    Common,
}

impl FrequencyBand {
    fn name(self) -> &'static str {
        match self {
            Self::Rare => "rare",
            Self::Common => "common",
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ByteBand {
    Short,
    Long,
}

impl ByteBand {
    fn name(self) -> &'static str {
        match self {
            Self::Short => "short",
            Self::Long => "long",
        }
    }
}

fn gravity(message: impl Into<String>) -> Error {
    Error::Gravity(format!("Flash E13 evidence: {}", message.into()))
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_bytes(value: &Value) -> Result<Vec<u8>> {
    serde_json::to_vec(value).map_err(|error| gravity(format!("JSON serialization: {error}")))
}

fn require_sha256(value: &str, field: &str) -> Result<String> {
    let normalized = value.to_ascii_lowercase();
    if normalized.len() != 64 || !normalized.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(gravity(format!("{field} must be a 64-character SHA-256")));
    }
    Ok(normalized)
}

fn bounded_regular_file(path: &Path, label: &str, maximum: u64) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        gravity(format!(
            "cannot inspect {label} {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(gravity(format!(
            "{label} must be a direct regular file, not a symlink: {}",
            path.display()
        )));
    }
    if metadata.len() > maximum {
        return Err(gravity(format!(
            "{label} is {} bytes, above the {}-byte bound",
            metadata.len(),
            maximum
        )));
    }
    let bytes = fs::read(path)
        .map_err(|error| gravity(format!("cannot read {label} {}: {error}", path.display())))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(gravity(format!("{label} changed while it was read")));
    }
    Ok(bytes)
}

fn relative_artifact_path(path: &str, field: &str) -> Result<()> {
    let path = Path::new(path);
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(gravity(format!(
            "{field} must be a non-empty relative path"
        )));
    }
    if path
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(gravity(format!(
            "{field} may contain only normal relative path components"
        )));
    }
    Ok(())
}

fn validate_declaration(declaration: &E13EvidenceDeclaration) -> Result<()> {
    if declaration.schema != DECLARATION_SCHEMA {
        return Err(gravity(format!(
            "declaration schema must be {DECLARATION_SCHEMA}"
        )));
    }
    require_sha256(&declaration.tokenizer_sha256, "tokenizer_sha256")?;
    require_sha256(&declaration.corpus_sha256, "corpus_sha256")?;
    if !declaration.source.is_object() || !declaration.row_universe.is_object() {
        return Err(gravity("source and row_universe must be JSON objects"));
    }
    if declaration.eligible_model_vocab_count == 0 {
        return Err(gravity("eligible_model_vocab_count must be positive"));
    }
    if declaration.fit_rows_per_stratum == 0 || declaration.heldout_rows_per_stratum == 0 {
        return Err(gravity(
            "every stratum needs positive fit and heldout counts",
        ));
    }
    let expected_budget = 8usize
        .checked_mul(
            declaration
                .fit_rows_per_stratum
                .checked_add(declaration.heldout_rows_per_stratum)
                .ok_or_else(|| gravity("stratum row count overflow"))?,
        )
        .ok_or_else(|| gravity("row budget overflow"))?;
    if declaration.row_budget != expected_budget {
        return Err(gravity(format!(
            "row_budget {} must equal eight strata times ({} fit + {} heldout) = {expected_budget}",
            declaration.row_budget,
            declaration.fit_rows_per_stratum,
            declaration.heldout_rows_per_stratum
        )));
    }
    if declaration.rare_max_count == 0 || declaration.common_min_count <= declaration.rare_max_count
    {
        return Err(gravity(
            "frequency bands must be disjoint: 1..rare_max_count, common_min_count..corpus_total",
        ));
    }
    if declaration.short_max_utf8_bytes == 0
        || declaration.long_min_utf8_bytes <= declaration.short_max_utf8_bytes
    {
        return Err(gravity(
            "byte bands must be disjoint: 1..short_max_utf8_bytes, long_min_utf8_bytes..observed_max",
        ));
    }
    Ok(())
}

fn primary_script(decoded: &str) -> Option<&'static str> {
    let latin = LATIN.find_iter(decoded).count();
    let han = HAN.find_iter(decoded).count();
    match latin.cmp(&han) {
        Ordering::Greater if latin > 0 => Some("Latin"),
        Ordering::Less if han > 0 => Some("Han"),
        _ => None,
    }
}

fn selected_rank(seed: u64, stratum: &str, token_id: u32) -> [u8; 32] {
    let input = format!("{PRODUCER_SCHEMA}|select|{seed}|{stratum}|{token_id}");
    Sha256::digest(input.as_bytes()).into()
}

fn in_frequency_band(
    count: u64,
    band: FrequencyBand,
    declaration: &E13EvidenceDeclaration,
) -> bool {
    match band {
        FrequencyBand::Rare => count <= declaration.rare_max_count,
        FrequencyBand::Common => count >= declaration.common_min_count,
    }
}

fn in_byte_band(length: usize, band: ByteBand, declaration: &E13EvidenceDeclaration) -> bool {
    match band {
        ByteBand::Short => length <= declaration.short_max_utf8_bytes,
        ByteBand::Long => length >= declaration.long_min_utf8_bytes,
    }
}

fn producer_implementation_sha256() -> String {
    sha256(include_bytes!("flash_e13_vocabulary_evidence.rs"))
}

fn bound_tokenizer(
    declaration: &E13EvidenceDeclaration,
    tokenizer_path: &Path,
) -> Result<(Tokenizer, String)> {
    let tokenizer_bytes = bounded_regular_file(tokenizer_path, "tokenizer", 64 * 1024 * 1024)?;
    let observed_tokenizer_sha256 = sha256(&tokenizer_bytes);
    if observed_tokenizer_sha256
        != require_sha256(&declaration.tokenizer_sha256, "tokenizer_sha256")?
    {
        return Err(gravity("tokenizer SHA-256 does not match the declaration"));
    }
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    if declaration.eligible_model_vocab_count > tokenizer.vocab_size() {
        return Err(gravity(format!(
            "eligible_model_vocab_count {} exceeds tokenizer size {}",
            declaration.eligible_model_vocab_count,
            tokenizer.vocab_size()
        )));
    }
    Ok((tokenizer, observed_tokenizer_sha256))
}

/// Count the exact tokenizer-decoded script/byte capacity of an E13 design.
///
/// This deliberately ignores corpus frequency.  It is the cheap structural
/// discriminator that answers whether each script/byte cell could ever hold
/// the declared fit+heldout row count before another corpus is acquired or
/// tokenized.  It opens no corpus, config, index, weight, tensor, or device.
pub fn inspect_e13_lexical_universe(
    declaration: &E13EvidenceDeclaration,
    tokenizer_path: &Path,
) -> Result<Value> {
    validate_declaration(declaration)?;
    let (tokenizer, observed_tokenizer_sha256) = bound_tokenizer(declaration, tokenizer_path)?;
    let per_cell = declaration.fit_rows_per_stratum + declaration.heldout_rows_per_stratum;
    let mut counts = BTreeMap::<String, usize>::new();
    let mut excluded_unclassifiable = 0usize;
    let mut excluded_byte_gap = 0usize;
    for token_id in 0..declaration.eligible_model_vocab_count as u32 {
        let decoded = tokenizer.decode_one(token_id)?;
        let Some(script) = primary_script(&decoded) else {
            excluded_unclassifiable += 1;
            continue;
        };
        let length = decoded.len();
        let byte_band = if length <= declaration.short_max_utf8_bytes {
            "short"
        } else if length >= declaration.long_min_utf8_bytes {
            "long"
        } else {
            excluded_byte_gap += 1;
            continue;
        };
        *counts
            .entry(format!("{}-{byte_band}", script.to_ascii_lowercase()))
            .or_default() += 1;
    }
    let cells: Vec<Value> = ["latin-short", "latin-long", "han-short", "han-long"]
        .into_iter()
        .map(|identifier| {
            let eligible_rows = counts.get(identifier).copied().unwrap_or(0);
            json!({
                "id": identifier,
                "eligible_rows": eligible_rows,
                "required_rows": per_cell,
                "structurally_possible": eligible_rows >= per_cell,
            })
        })
        .collect();
    let structurally_possible = cells
        .iter()
        .all(|cell| cell["structurally_possible"].as_bool() == Some(true));
    Ok(json!({
        "schema": LEXICAL_UNIVERSE_SCHEMA,
        "status": if structurally_possible {
            "STRUCTURALLY_CAPABLE__CORPUS_FREQUENCY_UNPROVEN"
        } else {
            "STRUCTURALLY_UNDERPOWERED__DO_NOT_ACQUIRE_ANOTHER_CORPUS_FOR_THIS_DESIGN"
        },
        "producer": {
            "schema": PRODUCER_SCHEMA,
            "implementation_sha256": producer_implementation_sha256(),
            "runtime": "hawking-core Rust tokenizer; CPU only",
        },
        "tokenizer_sha256": observed_tokenizer_sha256,
        "eligible_model_vocab_count": declaration.eligible_model_vocab_count,
        "row_budget": declaration.row_budget,
        "required_rows_per_script_byte_cell": per_cell,
        "byte_bands": {
            "short_max_utf8_bytes": declaration.short_max_utf8_bytes,
            "long_min_utf8_bytes": declaration.long_min_utf8_bytes,
        },
        "cells": cells,
        "excluded_unclassifiable_rows": excluded_unclassifiable,
        "excluded_byte_gap_rows": excluded_byte_gap,
        "all_script_byte_cells_structurally_possible": structurally_possible,
        "boundaries": {
            "corpus_opened": false,
            "model_loaded": false,
            "tensor_payload_bytes_read": 0,
            "row_extraction_performed": false,
            "gpu_session_started": false,
        },
        "claim_boundary": "Tokenizer-decoded lexical capacity only; corpus frequency, E13 evidence admission, row extraction, representation quality, complete NR, EBPW, TPS, capability, and promotion are unmeasured.",
    }))
}

/// Produce the exact E13 control artifacts from a predeclared corpus contract.
///
/// `frequency_artifact_path` and `script_artifact_path` are the relative paths
/// the Python preflight will resolve beneath `receipts/headless`; this function
/// does not write them and therefore cannot clobber existing evidence.
pub fn produce_e13_evidence(
    declaration: &E13EvidenceDeclaration,
    tokenizer_path: &Path,
    corpus_path: &Path,
    frequency_artifact_path: &str,
    script_artifact_path: &str,
) -> Result<E13EvidenceBundle> {
    validate_declaration(declaration)?;
    relative_artifact_path(frequency_artifact_path, "frequency_artifact_path")?;
    relative_artifact_path(script_artifact_path, "script_artifact_path")?;

    let (tokenizer, observed_tokenizer_sha256) = bound_tokenizer(declaration, tokenizer_path)?;
    let corpus_bytes = bounded_regular_file(corpus_path, "corpus", MAX_CORPUS_BYTES)?;
    let observed_corpus_sha256 = sha256(&corpus_bytes);
    if observed_corpus_sha256 != require_sha256(&declaration.corpus_sha256, "corpus_sha256")? {
        return Err(gravity("corpus SHA-256 does not match the declaration"));
    }
    let corpus = std::str::from_utf8(&corpus_bytes)
        .map_err(|error| gravity(format!("corpus must be exact UTF-8: {error}")))?;

    let token_ids = tokenizer.encode(corpus, false)?;
    if token_ids.is_empty() {
        return Err(gravity("corpus tokenization produced no tokens"));
    }
    let mut counts = vec![0u64; tokenizer.vocab_size()];
    for token_id in &token_ids {
        let slot = counts
            .get_mut(*token_id as usize)
            .ok_or_else(|| gravity(format!("tokenizer emitted out-of-range id {token_id}")))?;
        *slot = slot
            .checked_add(1)
            .ok_or_else(|| gravity("token frequency overflow"))?;
    }

    let mut candidates = Vec::new();
    let mut excluded_zero_frequency = 0usize;
    let mut excluded_unclassifiable = 0usize;
    for token_id in 0..declaration.eligible_model_vocab_count as u32 {
        let count = counts[token_id as usize];
        if count == 0 {
            excluded_zero_frequency += 1;
            continue;
        }
        let decoded = tokenizer.decode_one(token_id)?;
        let raw = tokenizer.raw_token_surface(token_id).ok_or_else(|| {
            gravity(format!(
                "tokenizer has no raw surface for eligible id {token_id}"
            ))
        })?;
        let Some(script) = primary_script(&decoded) else {
            excluded_unclassifiable += 1;
            continue;
        };
        let decoded_bytes = decoded.as_bytes();
        if decoded_bytes.is_empty() {
            excluded_unclassifiable += 1;
            continue;
        }
        candidates.push(Candidate {
            token_id,
            count,
            script,
            decoded_utf8_byte_length: decoded_bytes.len(),
            decoded_token_surface_sha256: sha256(decoded_bytes),
            tokenizer_raw_surface_sha256: sha256(raw.as_bytes()),
            tokenizer_raw_surface_utf8_byte_length: raw.len(),
        });
    }
    let maximum_decoded_bytes = candidates
        .iter()
        .map(|candidate| candidate.decoded_utf8_byte_length)
        .max()
        .ok_or_else(|| gravity("no frequency-bearing Latin/Han lexical token was decodable"))?;

    let mut selected = BTreeMap::<u32, Candidate>::new();
    let mut strata = Vec::new();
    let per_stratum = declaration.fit_rows_per_stratum + declaration.heldout_rows_per_stratum;
    for frequency_band in [FrequencyBand::Rare, FrequencyBand::Common] {
        for script in ["Latin", "Han"] {
            for byte_band in [ByteBand::Short, ByteBand::Long] {
                let identifier = format!(
                    "{}-{}-{}",
                    frequency_band.name(),
                    script.to_ascii_lowercase(),
                    byte_band.name()
                );
                let mut eligible: Vec<&Candidate> = candidates
                    .iter()
                    .filter(|candidate| {
                        candidate.script == script
                            && in_frequency_band(candidate.count, frequency_band, declaration)
                            && in_byte_band(
                                candidate.decoded_utf8_byte_length,
                                byte_band,
                                declaration,
                            )
                    })
                    .collect();
                eligible.sort_by_key(|candidate| {
                    (
                        selected_rank(declaration.selection_seed, &identifier, candidate.token_id),
                        candidate.token_id,
                    )
                });
                if eligible.len() < per_stratum {
                    return Err(gravity(format!(
                        "stratum {identifier} has {} eligible tokens but needs {per_stratum}; change only a predeclared corpus/threshold contract",
                        eligible.len()
                    )));
                }
                let chosen = &eligible[..per_stratum];
                let fit: Vec<u32> = chosen[..declaration.fit_rows_per_stratum]
                    .iter()
                    .map(|candidate| candidate.token_id)
                    .collect();
                let heldout: Vec<u32> = chosen[declaration.fit_rows_per_stratum..]
                    .iter()
                    .map(|candidate| candidate.token_id)
                    .collect();
                for candidate in chosen {
                    if selected
                        .insert(candidate.token_id, (*candidate).clone())
                        .is_some()
                    {
                        return Err(gravity(format!(
                            "token {} was selected into more than one stratum",
                            candidate.token_id
                        )));
                    }
                }
                let frequency_range = match frequency_band {
                    FrequencyBand::Rare => json!({
                        "min_inclusive": 1,
                        "max_inclusive": declaration.rare_max_count,
                    }),
                    FrequencyBand::Common => json!({
                        "min_inclusive": declaration.common_min_count,
                        "max_inclusive": token_ids.len() as u64,
                    }),
                };
                let byte_length_range = match byte_band {
                    ByteBand::Short => json!({
                        "min_inclusive": 1,
                        "max_inclusive": declaration.short_max_utf8_bytes,
                    }),
                    ByteBand::Long => json!({
                        "min_inclusive": declaration.long_min_utf8_bytes,
                        "max_inclusive": maximum_decoded_bytes,
                    }),
                };
                strata.push(json!({
                    "id": identifier,
                    "frequency_band": frequency_band.name(),
                    "frequency_range": frequency_range,
                    "script": script,
                    "byte_length_band": byte_band.name(),
                    "byte_length_range": byte_length_range,
                    "fit_token_ids": fit,
                    "heldout_token_ids": heldout,
                }));
            }
        }
    }
    if selected.len() != declaration.row_budget {
        return Err(gravity(format!(
            "selected {} unique rows, expected {}",
            selected.len(),
            declaration.row_budget
        )));
    }

    let producer = json!({
        "schema": PRODUCER_SCHEMA,
        "implementation_sha256": producer_implementation_sha256(),
        "runtime": "hawking-core Rust tokenizer; CPU only",
        "model_loaded": false,
        "tensor_payload_bytes_read": 0,
        "gpu_session_started": false,
    });
    let frequency_rows: Vec<Value> = selected
        .values()
        .map(|candidate| json!({"token_id": candidate.token_id, "count": candidate.count}))
        .collect();
    let script_rows: Vec<Value> = selected
        .values()
        .map(|candidate| {
            json!({
                "token_id": candidate.token_id,
                "script": candidate.script,
                "tokenizer_raw_surface_sha256": candidate.tokenizer_raw_surface_sha256,
                "tokenizer_raw_surface_utf8_byte_length": candidate.tokenizer_raw_surface_utf8_byte_length,
                "decoded_token_surface_sha256": candidate.decoded_token_surface_sha256,
                "decoded_utf8_byte_length": candidate.decoded_utf8_byte_length,
            })
        })
        .collect();
    let frequency_evidence = json!({
        "schema": FREQUENCY_EVIDENCE_SCHEMA,
        "provenance": FREQUENCY_PROVENANCE,
        "tokenizer_sha256": observed_tokenizer_sha256,
        "corpus_sha256": observed_corpus_sha256,
        "corpus_token_count": token_ids.len(),
        "producer": producer,
        "rows": frequency_rows,
    });
    let script_evidence = json!({
        "schema": SCRIPT_EVIDENCE_SCHEMA,
        "provenance": SCRIPT_PROVENANCE,
        "tokenizer_sha256": declaration.tokenizer_sha256.to_ascii_lowercase(),
        "classifier": {
            "algorithm": SCRIPT_CLASSIFIER_ALGORITHM,
            "implementation_sha256": producer_implementation_sha256(),
        },
        "producer": producer,
        "rows": script_rows,
    });
    let frequency_bytes = json_bytes(&frequency_evidence)?;
    let script_bytes = json_bytes(&script_evidence)?;
    let plan = json!({
        "schema": PLAN_SCHEMA,
        "row_budget": declaration.row_budget,
        "source": declaration.source,
        "row_universe": declaration.row_universe,
        "frequency_evidence": {
            "artifact_path": frequency_artifact_path,
            "artifact_sha256": sha256(&frequency_bytes),
        },
        "script_evidence": {
            "artifact_path": script_artifact_path,
            "artifact_sha256": sha256(&script_bytes),
        },
        "strata": strata,
        "shuffled_token_label_null": {
            "algorithm": NULL_ALGORITHM,
            "seed": declaration.shuffled_null_seed,
        },
    });
    let producer_receipt = json!({
        "schema": PRODUCER_SCHEMA,
        "status": "EVIDENCE_PRODUCED_NOT_EXPERIMENT_EXECUTED",
        "declaration": declaration,
        "tokenizer": {
            "path": tokenizer_path,
            "sha256": observed_tokenizer_sha256,
            "vocab_size": tokenizer.vocab_size(),
            "eligible_model_vocab_count": declaration.eligible_model_vocab_count,
        },
        "corpus": {
            "path": corpus_path,
            "sha256": observed_corpus_sha256,
            "bytes": corpus_bytes.len(),
            "token_count": token_ids.len(),
        },
        "selection": {
            "row_budget": declaration.row_budget,
            "selected_token_ids_sha256": sha256(&json_bytes(&json!(selected.keys().collect::<Vec<_>>()))?),
            "excluded_zero_frequency": excluded_zero_frequency,
            "excluded_unclassifiable_primary_script": excluded_unclassifiable,
            "frequency_artifact_sha256": sha256(&frequency_bytes),
            "script_artifact_sha256": sha256(&script_bytes),
            "plan_sha256": sha256(&json_bytes(&plan)?),
        },
        "boundaries": {
            "model_loaded": false,
            "tensor_payload_bytes_read": 0,
            "row_extraction_performed": false,
            "experiment_executed": false,
            "capability_claim": false,
            "density_claim": false,
            "tps_claim": false,
        },
        "producer": producer,
    });

    Ok(E13EvidenceBundle {
        frequency_evidence,
        script_evidence,
        plan,
        producer_receipt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    fn write_test_tokenizer(path: &Path, tokens: &[String]) {
        let vocab: HashMap<String, u32> = tokens
            .iter()
            .cloned()
            .enumerate()
            .map(|(id, token)| (token, id as u32))
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer.save(path, false).unwrap();
    }

    #[test]
    fn produces_disjoint_eight_cell_plan_from_exact_tokenizer_and_corpus() {
        let temp = tempfile::tempdir().unwrap();
        let tokenizer_path = temp.path().join("tokenizer.json");
        let corpus_path = temp.path().join("corpus.txt");
        let mut tokens = vec!["[UNK]".to_owned()];
        let mut rare = Vec::new();
        let mut common = Vec::new();
        let cells = [
            ["a", "b", "c", "d", "e", "f", "g", "h"],
            [
                "alpha", "bravo", "cider", "delta", "eagle", "fable", "giant", "honey",
            ],
            ["汉", "字", "语", "文", "话", "言", "学", "习"],
            [
                "汉语", "文字", "学习", "话语", "漢字", "语言", "中文", "漢語",
            ],
        ];
        for cell in cells {
            for (index, token) in cell.into_iter().enumerate() {
                tokens.push(token.to_owned());
                if index < 4 {
                    rare.push(token.to_owned());
                } else {
                    common.push(token.to_owned());
                }
            }
        }
        write_test_tokenizer(&tokenizer_path, &tokens);
        let mut corpus_tokens = rare.clone();
        for _ in 0..3 {
            corpus_tokens.extend(common.clone());
        }
        fs::write(&corpus_path, corpus_tokens.join(" ")).unwrap();
        let declaration = E13EvidenceDeclaration {
            schema: DECLARATION_SCHEMA.into(),
            tokenizer_sha256: sha256(&fs::read(&tokenizer_path).unwrap()),
            corpus_sha256: sha256(&fs::read(&corpus_path).unwrap()),
            eligible_model_vocab_count: tokens.len(),
            row_budget: 32,
            fit_rows_per_stratum: 2,
            heldout_rows_per_stratum: 2,
            rare_max_count: 1,
            common_min_count: 3,
            short_max_utf8_bytes: 3,
            long_min_utf8_bytes: 4,
            selection_seed: 11,
            shuffled_null_seed: 17,
            source: json!({"tokenizer_sha256": "bound-by-preflight"}),
            row_universe: json!({"schema": "test"}),
        };
        let bundle = produce_e13_evidence(
            &declaration,
            &tokenizer_path,
            &corpus_path,
            "e13/frequency-evidence.json",
            "e13/script-evidence.json",
        )
        .unwrap();
        assert_eq!(bundle.plan["strata"].as_array().unwrap().len(), 8);
        assert_eq!(
            bundle.frequency_evidence["rows"].as_array().unwrap().len(),
            32
        );
        assert_eq!(bundle.script_evidence["rows"].as_array().unwrap().len(), 32);
        assert_eq!(
            bundle.producer_receipt["status"],
            "EVIDENCE_PRODUCED_NOT_EXPERIMENT_EXECUTED"
        );
        assert_eq!(
            bundle.producer_receipt["boundaries"]["tensor_payload_bytes_read"],
            0
        );
    }

    #[test]
    fn lexical_universe_census_is_corpus_free_and_reports_structural_capacity() {
        let temp = tempfile::tempdir().unwrap();
        let tokenizer_path = temp.path().join("tokenizer.json");
        let tokens = [
            "[UNK]", "a", "b", "c", "d", "e", "f", "g", "h", "alpha", "bravo", "cider", "delta",
            "eagle", "fable", "giant", "honey", "汉", "字", "语", "文", "话", "言", "学", "习",
            "汉语", "文字", "学习", "话语", "漢字", "语言", "中文", "漢語",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
        write_test_tokenizer(&tokenizer_path, &tokens);
        let declaration = E13EvidenceDeclaration {
            schema: DECLARATION_SCHEMA.into(),
            tokenizer_sha256: sha256(&fs::read(&tokenizer_path).unwrap()),
            corpus_sha256: "0".repeat(64),
            eligible_model_vocab_count: tokens.len(),
            row_budget: 32,
            fit_rows_per_stratum: 2,
            heldout_rows_per_stratum: 2,
            rare_max_count: 1,
            common_min_count: 2,
            short_max_utf8_bytes: 3,
            long_min_utf8_bytes: 4,
            selection_seed: 11,
            shuffled_null_seed: 17,
            source: json!({"test": true}),
            row_universe: json!({"test": true}),
        };
        let census = inspect_e13_lexical_universe(&declaration, &tokenizer_path).unwrap();
        assert_eq!(
            census["status"],
            "STRUCTURALLY_CAPABLE__CORPUS_FREQUENCY_UNPROVEN"
        );
        assert_eq!(census["boundaries"]["corpus_opened"], false);
        assert_eq!(census["boundaries"]["tensor_payload_bytes_read"], 0);
        let cells = census["cells"].as_array().unwrap();
        assert_eq!(cells.len(), 4);
        assert_eq!(cells[0]["eligible_rows"], 8);
        assert_eq!(cells[1]["eligible_rows"], 9);
        assert_eq!(cells[2]["eligible_rows"], 8);
        assert_eq!(cells[3]["eligible_rows"], 8);
    }

    #[test]
    fn refuses_changed_corpus_before_tokenization() {
        let temp = tempfile::tempdir().unwrap();
        let tokenizer_path = temp.path().join("tokenizer.json");
        let corpus_path = temp.path().join("corpus.txt");
        write_test_tokenizer(&tokenizer_path, &["[UNK]".into(), "a".into()]);
        fs::write(&corpus_path, "a").unwrap();
        let declaration = E13EvidenceDeclaration {
            schema: DECLARATION_SCHEMA.into(),
            tokenizer_sha256: sha256(&fs::read(&tokenizer_path).unwrap()),
            corpus_sha256: "0".repeat(64),
            eligible_model_vocab_count: 2,
            row_budget: 16,
            fit_rows_per_stratum: 1,
            heldout_rows_per_stratum: 1,
            rare_max_count: 1,
            common_min_count: 2,
            short_max_utf8_bytes: 1,
            long_min_utf8_bytes: 2,
            selection_seed: 1,
            shuffled_null_seed: 1,
            source: json!({}),
            row_universe: json!({}),
        };
        let error = produce_e13_evidence(
            &declaration,
            &tokenizer_path,
            &corpus_path,
            "e13/frequency.json",
            "e13/script.json",
        )
        .unwrap_err();
        assert!(error.to_string().contains("corpus SHA-256"));
    }
}
