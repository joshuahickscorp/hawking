//! **Speculation collapse.** Speculation is optional and never part of the default scientific condensation
//! profile unless parent-bound parity is proven. All providers implement ONE contract —
//! propose/verify/accept/metrics — with separated concerns (prompt lookup, suffix automaton, EAGLE/head
//! drafting, policy, tuning, benchmarking) and no duplicated routers/governors/metrics/state machines.
//!
//! The reference provider is prompt-lookup drafting: propose the continuation seen after the last matching
//! n-gram in the prompt. Verification against the target's argmax decides acceptance — the target model
//! (the Seed's runtime) remains the only authority on correctness.

use super::provider::{Context, Provider, ProviderOutput, ResourceUsage};
use crate::pack::CapabilityKind;
use crate::Result;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct SpecMetrics {
    pub proposed: usize,
    pub accepted: usize,
    pub acceptance_rate: f64,
}

/// The one speculation contract. Providers are pure: they propose draft tokens and report metrics; the
/// target (the Seed's runtime) verifies and is the sole authority on the accepted sequence.
pub trait Speculator {
    fn name(&self) -> &str;

    /// propose: draft up to `k` continuation tokens from the context.
    fn propose(&self, context: &[u32], k: usize) -> Vec<u32>;

    /// verify: given the target's next-token oracle, return how many leading drafts are accepted.
    fn verify(&self, drafts: &[u32], target_next: &dyn Fn(&[u32]) -> u32, context: &[u32]) -> usize;

    /// accept: the accepted prefix of the drafts.
    fn accept<'a>(&self, drafts: &'a [u32], accepted: usize) -> &'a [u32] {
        &drafts[..accepted]
    }
}

/// Reference: prompt-lookup n-gram drafting.
pub struct PromptLookup {
    pub ngram: usize,
}

impl PromptLookup {
    pub fn new(ngram: usize) -> Self {
        PromptLookup { ngram: ngram.max(1) }
    }
}

impl Speculator for PromptLookup {
    fn name(&self) -> &str {
        "prompt_lookup"
    }

    fn propose(&self, context: &[u32], k: usize) -> Vec<u32> {
        if context.len() <= self.ngram {
            return Vec::new();
        }
        let tail = &context[context.len() - self.ngram..];
        // find the last earlier occurrence of `tail` and draft what followed it.
        for start in (0..context.len() - self.ngram).rev() {
            if &context[start..start + self.ngram] == tail {
                let after = start + self.ngram;
                let end = (after + k).min(context.len());
                if after < end {
                    return context[after..end].to_vec();
                }
            }
        }
        Vec::new()
    }

    fn verify(&self, drafts: &[u32], target_next: &dyn Fn(&[u32]) -> u32, context: &[u32]) -> usize {
        let mut seq = context.to_vec();
        let mut accepted = 0;
        for &d in drafts {
            let want = target_next(&seq);
            if want == d {
                accepted += 1;
                seq.push(d);
            } else {
                break;
            }
        }
        accepted
    }
}

/// A `Provider` over the speculation contract: draft against a deterministic target oracle and report
/// acceptance metrics. Speculation is `optional` — never activated in the default profile without parity.
pub struct SpeculationProviderImpl<S: Speculator> {
    pub spec: S,
    capability: String,
}

impl<S: Speculator> SpeculationProviderImpl<S> {
    pub fn new(spec: S) -> Self {
        let capability = format!("speculation.{}", spec.name());
        SpeculationProviderImpl { spec, capability }
    }
}

impl<S: Speculator> Provider for SpeculationProviderImpl<S> {
    fn capability(&self) -> &str {
        &self.capability
    }
    fn kind(&self) -> CapabilityKind {
        CapabilityKind::SpeculationProvider
    }
    fn run(&self, _ctx: &Context, input: serde_json::Value) -> Result<ProviderOutput> {
        // A repeating context makes prompt-lookup productive; the target oracle continues the pattern.
        let period = input["period"].as_u64().unwrap_or(4) as u32;
        let n = input["len"].as_u64().unwrap_or(24) as usize;
        let context: Vec<u32> = (0..n as u32).map(|i| i % period).collect();
        let target_next = |seq: &[u32]| -> u32 { (seq.len() as u32) % period };
        let drafts = self.spec.propose(&context, 4);
        let accepted = self.spec.verify(&drafts, &target_next, &context);
        let proposed = drafts.len();
        let metrics = SpecMetrics {
            proposed,
            accepted,
            acceptance_rate: if proposed == 0 { 0.0 } else { accepted as f64 / proposed as f64 },
        };
        let result = serde_json::json!({ "drafts": drafts, "accepted_prefix": self.spec.accept(&drafts, accepted) });
        Ok(ProviderOutput::sealed(result, serde_json::to_value(&metrics)?, ResourceUsage::default()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_lookup_proposes_and_target_verifies() {
        let spec = PromptLookup::new(2);
        let context: Vec<u32> = (0..16).map(|i| i % 4).collect(); // 0,1,2,3,0,1,2,3,...
        let drafts = spec.propose(&context, 4);
        assert!(!drafts.is_empty(), "prompt-lookup should find a repeat");
        let target_next = |seq: &[u32]| -> u32 { (seq.len() as u32) % 4 };
        let accepted = spec.verify(&drafts, &target_next, &context);
        assert!(accepted >= 1, "at least the first draft should match the periodic target");
    }
}
