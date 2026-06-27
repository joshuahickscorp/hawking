use crate::kg::Claim;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaimVerification {
    pub claim_id: String,
    pub status: ClaimStatus,
    pub supporting_sources: usize,
    pub refuting_sources: usize,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimStatus {
    Supported,
    Refuted,
    Contradicted,
    SingleSource,
    Unverified,
}

pub struct AdversarialVerifier;

impl AdversarialVerifier {
    pub fn verify(claim: &Claim, peer_claims: &[Claim]) -> ClaimVerification {
        let mut supporting_sources = 0;
        let mut refuting_sources = 0;
        for peer in peer_claims {
            if peer.id == claim.id {
                continue;
            }
            let a = claim.text.to_lowercase();
            let b = peer.text.to_lowercase();
            if lexical_overlap(&a, &b) > 0.4 {
                supporting_sources += 1;
            }
            if (a.contains("increase") && b.contains("decrease"))
                || (a.contains("faster") && b.contains("slower"))
            {
                refuting_sources += 1;
            }
        }
        let status = match (supporting_sources, refuting_sources) {
            (_, r) if r > 0 && supporting_sources > 0 => ClaimStatus::Contradicted,
            (_, r) if r > 0 => ClaimStatus::Refuted,
            (s, _) if s > 0 => ClaimStatus::Supported,
            _ => ClaimStatus::SingleSource,
        };
        ClaimVerification {
            claim_id: claim.id.clone(),
            status,
            supporting_sources,
            refuting_sources,
            notes: Vec::new(),
        }
    }
}

fn lexical_overlap(a: &str, b: &str) -> f32 {
    let words: Vec<_> = a.split_whitespace().filter(|w| w.len() > 3).collect();
    if words.is_empty() {
        return 0.0;
    }
    let hits = words.iter().filter(|w| b.contains(**w)).count();
    hits as f32 / words.len() as f32
}
