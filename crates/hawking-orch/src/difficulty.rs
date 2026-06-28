use hide_core::runtime::InferenceRequest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DifficultyEstimate {
    pub score: f32,
    pub reason: String,
    pub signals: Vec<DifficultySignal>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DifficultySignal {
    pub name: String,
    pub value: f32,
}

#[derive(Default)]
pub struct DifficultyEstimator;

impl DifficultyEstimator {
    pub fn estimate(&self, request: &InferenceRequest) -> DifficultyEstimate {
        let prompt_len = request.prompt.chars().count() as f32;
        let mut score = (prompt_len / 12_000.0).min(0.45);
        let mut signals = vec![DifficultySignal {
            name: "prompt_length".to_string(),
            value: score,
        }];
        for marker in [
            "refactor",
            "multi-file",
            "security",
            "architecture",
            "failing tests",
        ] {
            if request.prompt.to_lowercase().contains(marker) {
                score += 0.12;
                signals.push(DifficultySignal {
                    name: marker.to_string(),
                    value: 0.12,
                });
            }
        }
        score = score.min(1.0);
        DifficultyEstimate {
            score,
            reason: if score > 0.65 {
                "high difficulty; route to hero role".to_string()
            } else {
                "low/medium difficulty; cheap role can try first".to_string()
            },
            signals,
        }
    }
}
