//! Per-layer quant tier map: pick the bit-width each layer's MoE expert
//! weights are materialized at. path-to-50 lever 2.
//!
//! ## Motivation
//!
//! V2-Lite ships with uniform-per-tensor quant (Q4_K gate+up, Q5_0 / Q6_K
//! down). The corpus calibration in
//! `artifacts/calibration/analysis/per_layer_residual_stats.json` shows the
//! per-layer residual abs_max varies by ~500× across layers — layers 0–3
//! tolerate Q4 fine, layers 4–24 need Q8 headroom, layers 25–26 settle at
//! Q6. This module loads a JSON file describing that tier map and answers
//! "which dtype should layer N's expert weights be materialized at?".
//!
//! ## File schema
//!
//! ```json
//! {
//!   "schema_version": 1,
//!   "model_arch": "deepseek2",
//!   "model_id": "deepseek-v2-lite-chat",
//!   "n_layers": 27,
//!   "comment": "free-form derivation notes",
//!   "layers": [
//!     { "layer": 0,  "gate_up": "q4_K", "down": "q4_K" },
//!     { "layer": 1,  "gate_up": "q4_K", "down": "q4_K" },
//!     ...
//!     { "layer": 26, "gate_up": "q6_K", "down": "q6_K" }
//!   ]
//! }
//! ```
//!
//! Entries are addressed by `layer`; layers not present in the map fall
//! through to GGUF native dtype. `gate_up` and `down` may be independently
//! absent / present per layer (mirrors the V2-Lite GGUF split where gate+up
//! and down already live at different dtypes).
//!
//! Allowed dtype strings: `"q4_K"`, `"q6_K"`, `"q8_0"`. Adding more later
//! is a one-line case-arm in [`parse_dtype`].
//!
//! ## Out of scope
//!
//! - Attention / norm / embed / lm_head precision (lever 2 is MoE-only)
//! - Activation quant
//! - Profile-time / autotune-driven map selection

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::gguf::GgmlType;
use crate::{Error, Result};

/// Which group of expert weights an entry refers to. The V2-Lite MoE
/// path has two distinct GEMV families per layer:
///
/// - `GateUp` covers the fused `ffn_gate_exps.weight` +
///   `ffn_up_exps.weight` pair (and the shared variants
///   `ffn_gate_shexp` / `ffn_up_shexp`).
/// - `Down` covers `ffn_down_exps.weight` and `ffn_down_shexp.weight`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupKind {
    GateUp,
    Down,
}

#[derive(Debug, Clone)]
pub struct TierMap {
    schema_version: u32,
    model_arch: String,
    n_layers: usize,
    /// Quant override per (layer, group). Missing entries fall through to
    /// native GGUF dtype at dispatch time.
    overrides: HashMap<(usize, GroupKind), GgmlType>,
}

#[derive(Debug, Deserialize)]
struct TierFile {
    schema_version: u32,
    model_arch: String,
    #[allow(dead_code)]
    #[serde(default)]
    model_id: String,
    n_layers: usize,
    #[allow(dead_code)]
    #[serde(default)]
    comment: String,
    layers: Vec<TierEntry>,
}

#[derive(Debug, Deserialize)]
struct TierEntry {
    layer: usize,
    #[serde(default)]
    gate_up: Option<String>,
    #[serde(default)]
    down: Option<String>,
}

const SCHEMA_VERSION: u32 = 1;

fn parse_dtype(s: &str) -> Result<GgmlType> {
    match s {
        "q4_K" | "Q4_K" | "q4k" | "Q4K" => Ok(GgmlType::Q4_K),
        "q6_K" | "Q6_K" | "q6k" | "Q6K" => Ok(GgmlType::Q6_K),
        "q8_0" | "Q8_0" | "q8" | "Q8" => Ok(GgmlType::Q8_0),
        other => Err(Error::Model(format!(
            "quant_tier_map: unsupported dtype string {other:?} \
             (allowed: q4_K, q6_K, q8_0)"
        ))),
    }
}

impl TierMap {
    /// Load and validate a tier-map JSON file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;
        let raw: TierFile = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Model(format!("quant_tier_map parse: {e}")))?;
        Self::from_parts(raw)
    }

    fn from_parts(raw: TierFile) -> Result<Self> {
        if raw.schema_version != SCHEMA_VERSION {
            return Err(Error::Model(format!(
                "quant_tier_map: unsupported schema_version {} (expected {})",
                raw.schema_version, SCHEMA_VERSION
            )));
        }
        if raw.n_layers == 0 {
            return Err(Error::Model("quant_tier_map: n_layers must be > 0".into()));
        }
        let mut overrides = HashMap::new();
        let mut seen = std::collections::HashSet::new();
        for e in &raw.layers {
            if e.layer >= raw.n_layers {
                return Err(Error::Model(format!(
                    "quant_tier_map: entry layer={} >= n_layers={}",
                    e.layer, raw.n_layers
                )));
            }
            if !seen.insert(e.layer) {
                return Err(Error::Model(format!(
                    "quant_tier_map: duplicate entry for layer {}",
                    e.layer
                )));
            }
            if let Some(s) = e.gate_up.as_deref() {
                overrides.insert((e.layer, GroupKind::GateUp), parse_dtype(s)?);
            }
            if let Some(s) = e.down.as_deref() {
                overrides.insert((e.layer, GroupKind::Down), parse_dtype(s)?);
            }
        }
        Ok(Self {
            schema_version: raw.schema_version,
            model_arch: raw.model_arch,
            n_layers: raw.n_layers,
            overrides,
        })
    }

    /// Validate the map matches a live model. Mismatched arch or layer
    /// count is a fail-fast error — a mis-applied map would silently
    /// re-quantize the wrong tensors.
    pub fn validate(&self, arch: &str, n_layers: usize) -> Result<()> {
        if self.model_arch != arch {
            return Err(Error::Model(format!(
                "quant_tier_map: model_arch={} but engine reports {}",
                self.model_arch, arch
            )));
        }
        if self.n_layers != n_layers {
            return Err(Error::Model(format!(
                "quant_tier_map: n_layers={} but model has {}",
                self.n_layers, n_layers
            )));
        }
        Ok(())
    }

    /// Lookup the tier override for `(layer_id, group)`. Returns `None`
    /// when the map has no entry — caller should fall through to the
    /// GGUF native dtype.
    #[inline]
    pub fn tier_for(&self, layer_id: usize, group: GroupKind) -> Option<GgmlType> {
        self.overrides.get(&(layer_id, group)).copied()
    }

    /// True when at least one layer × group is overridden. False maps
    /// (e.g. an empty `layers` array) are not an error; the engine
    /// simply behaves as if the path were `None`.
    pub fn any_overrides(&self) -> bool {
        !self.overrides.is_empty()
    }

    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> Result<TierMap> {
        let raw: TierFile =
            serde_json::from_str(json).map_err(|e| Error::Model(format!("test parse: {e}")))?;
        TierMap::from_parts(raw)
    }

    #[test]
    fn loads_simple_map() {
        let m = parse(
            r#"{
                "schema_version": 1,
                "model_arch": "deepseek2",
                "n_layers": 3,
                "layers": [
                    { "layer": 0, "gate_up": "q4_K", "down": "q4_K" },
                    { "layer": 2, "gate_up": "q8_0" }
                ]
            }"#,
        )
        .unwrap();
        assert_eq!(m.tier_for(0, GroupKind::GateUp), Some(GgmlType::Q4_K));
        assert_eq!(m.tier_for(0, GroupKind::Down), Some(GgmlType::Q4_K));
        assert_eq!(m.tier_for(1, GroupKind::GateUp), None);
        assert_eq!(m.tier_for(2, GroupKind::GateUp), Some(GgmlType::Q8_0));
        assert_eq!(m.tier_for(2, GroupKind::Down), None);
        assert!(m.any_overrides());
    }

    #[test]
    fn rejects_unknown_dtype() {
        let r = parse(
            r#"{ "schema_version": 1, "model_arch": "deepseek2", "n_layers": 2,
                 "layers": [{ "layer": 0, "gate_up": "q3_K" }] }"#,
        );
        assert!(r.is_err());
    }

    #[test]
    fn rejects_out_of_range_layer() {
        let r = parse(
            r#"{ "schema_version": 1, "model_arch": "deepseek2", "n_layers": 2,
                 "layers": [{ "layer": 5, "gate_up": "q4_K" }] }"#,
        );
        assert!(r.is_err());
    }

    #[test]
    fn rejects_duplicate_layer() {
        let r = parse(
            r#"{ "schema_version": 1, "model_arch": "deepseek2", "n_layers": 2,
                 "layers": [
                    { "layer": 0, "gate_up": "q4_K" },
                    { "layer": 0, "down": "q8_0" }
                 ] }"#,
        );
        assert!(r.is_err());
    }

    #[test]
    fn rejects_wrong_schema_version() {
        let r = parse(
            r#"{ "schema_version": 2, "model_arch": "deepseek2", "n_layers": 1, "layers": [] }"#,
        );
        assert!(r.is_err());
    }

    #[test]
    fn empty_layers_is_legal() {
        let m = parse(
            r#"{ "schema_version": 1, "model_arch": "deepseek2", "n_layers": 1, "layers": [] }"#,
        )
        .unwrap();
        assert!(!m.any_overrides());
    }

    #[test]
    fn validate_arch_and_layer_count() {
        let m = parse(
            r#"{ "schema_version": 1, "model_arch": "deepseek2", "n_layers": 27, "layers": [] }"#,
        )
        .unwrap();
        assert!(m.validate("deepseek2", 27).is_ok());
        assert!(m.validate("llama", 27).is_err());
        assert!(m.validate("deepseek2", 26).is_err());
    }
}
