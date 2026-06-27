use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingRecord {
    pub chunk_id: String,
    pub model_id: String,
    pub dim: usize,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridRetrievalWeights {
    pub lexical: f32,
    pub symbol: f32,
    pub semantic: f32,
    pub graph: f32,
}

impl Default for HybridRetrievalWeights {
    fn default() -> Self {
        Self {
            lexical: 1.0,
            symbol: 1.5,
            semantic: 0.5,
            graph: 0.75,
        }
    }
}

pub fn reciprocal_rank_fusion(ranks: &[usize], k: f32) -> f32 {
    ranks.iter().map(|rank| 1.0 / (k + *rank as f32)).sum()
}
