use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalFeedback {
    pub query: String,
    pub chosen_span_id: String,
    pub rejected_span_ids: Vec<String>,
    pub task_success: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearnedRetrievalWeights {
    pub lexical: f32,
    pub symbol: f32,
    pub semantic: f32,
    pub graph: f32,
    pub recency: f32,
}

impl Default for LearnedRetrievalWeights {
    fn default() -> Self {
        Self {
            lexical: 1.0,
            symbol: 1.5,
            semantic: 0.5,
            graph: 0.75,
            recency: 0.25,
        }
    }
}
