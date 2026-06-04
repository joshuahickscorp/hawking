//! Decode-step driver for the continuous-batching control plane.
//!
//! This module deliberately does not pretend that the current single-KV
//! engines can safely mix unrelated requests. The caller must only mark slots
//! as `Decoding` after the engine has the matching per-slot KV context ready.
//! The next GPU-resident batch kernel plugs in behind `Engine::forward_tokens_batched`.

use crate::batch::{scheduler::Scheduler, DecodedToken};
use anyhow::Result;
use dismantle_core::{Engine, GenerateRequest};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeOutput {
    pub slot_id: u32,
    pub token: u32,
    pub text: String,
    pub finished: bool,
}

pub struct BatchDriver {
    pub scheduler: Scheduler,
}

impl BatchDriver {
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            scheduler: Scheduler::new(max_batch_size),
        }
    }

    pub fn admit(&mut self, engine: &dyn Engine, req: GenerateRequest) -> Result<Option<u32>> {
        let prompt_ids = engine.encode_prompt_for_batch(&req.prompt)?;
        Ok(self.scheduler.admit(req, prompt_ids))
    }

    pub fn decode_ready_once(
        &mut self,
        engine: &mut dyn Engine,
        max_batch: usize,
    ) -> Result<Vec<DecodeOutput>> {
        let batch = self.scheduler.decode_batch(max_batch);
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        let tokens: Vec<u32> = batch.iter().map(|step| step.token).collect();
        let positions: Vec<usize> = batch.iter().map(|step| step.position).collect();
        // STABLE KV region per slot = the slot id (0..max_batch_size, reused on
        // release), NOT the compacted batch index — so a slot keeps its KV as the
        // ready set churns. The multi-seq path keys KV by this region.
        let regions: Vec<usize> = batch.iter().map(|step| step.slot_id as usize).collect();
        // Continuous-batching DECODE: the batch holds N INDEPENDENT slots at
        // divergent positions, so route through the multi-seq seam (QwenDense
        // overrides it with the GPU weight-amortizing path) — NOT
        // forward_tokens_batched, which is one-sequence prefill/verify.
        let mut logits = engine.forward_multiseq_batched(&tokens, &positions, &regions)?;
        let eos_id = engine.eos_id_for_batch();
        let decoded = self
            .scheduler
            .apply_decode_logits(&batch, &mut logits, eos_id)?;

        decoded
            .into_iter()
            .map(|token| decode_output(engine, token))
            .collect()
    }
}

fn decode_output(engine: &dyn Engine, token: DecodedToken) -> Result<DecodeOutput> {
    Ok(DecodeOutput {
        slot_id: token.slot_id,
        token: token.token,
        text: engine.decode_token_for_batch(token.token)?,
        finished: token.finished,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dismantle_core::{
        EngineConfig, GenStats, GenerateRequest, SamplingParams, StreamEvent,
    };
    use std::path::Path;

    struct FakeEngine {
        calls: Vec<(Vec<u32>, Vec<usize>)>,
    }

    impl FakeEngine {
        fn new() -> Self {
            Self { calls: Vec::new() }
        }
    }

    impl Engine for FakeEngine {
        fn load(_weights: &Path, _config: EngineConfig) -> dismantle_core::Result<Self>
        where
            Self: Sized,
        {
            Ok(Self::new())
        }

        fn generate(
            &mut self,
            _req: GenerateRequest,
            _sink: &mut dyn FnMut(StreamEvent),
        ) -> dismantle_core::Result<GenStats> {
            Ok(GenStats {
                completion_tokens: 0,
                ..Default::default()
            })
        }

        fn model_id(&self) -> &str {
            "fake"
        }

        fn encode_prompt_for_batch(&self, prompt: &str) -> dismantle_core::Result<Vec<u32>> {
            Ok(prompt.bytes().map(u32::from).collect())
        }

        fn decode_token_for_batch(&self, token: u32) -> dismantle_core::Result<String> {
            Ok(format!("<{token}>"))
        }

        fn eos_id_for_batch(&self) -> Option<u32> {
            Some(2)
        }

        fn forward_tokens_for_test(
            &mut self,
            tokens: &[u32],
            positions: &[usize],
        ) -> dismantle_core::Result<Vec<Vec<f32>>> {
            self.forward_tokens_batched(tokens, positions)
        }

        fn forward_tokens_batched(
            &mut self,
            tokens: &[u32],
            positions: &[usize],
        ) -> dismantle_core::Result<Vec<Vec<f32>>> {
            self.calls.push((tokens.to_vec(), positions.to_vec()));
            Ok(tokens
                .iter()
                .map(|token| match *token {
                    10 => vec![0.0, 4.0, 1.0],
                    20 => vec![0.0, 1.0, 5.0],
                    _ => vec![3.0, 0.0, 0.0],
                })
                .collect())
        }
    }

    fn req(max_new_tokens: usize) -> GenerateRequest {
        GenerateRequest {
            prompt: "x".into(),
            max_new_tokens,
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            stop: Vec::new(),
            abort: None,
            max_stall_ms: 0,
        }
    }

    #[test]
    fn decode_ready_once_batches_tokens_and_applies_outputs() {
        let mut driver = BatchDriver::new(4);
        for (id, token) in [(0, 10), (1, 20)] {
            let slot_id = driver.scheduler.admit(req(4), vec![token]).expect("admit");
            assert_eq!(slot_id, id);
            assert!(driver.scheduler.mark_prefill_complete(slot_id));
        }
        let mut engine = FakeEngine::new();

        let out = driver
            .decode_ready_once(&mut engine, 4)
            .expect("decode once");

        assert_eq!(engine.calls, vec![(vec![10, 20], vec![1, 1])]);
        assert_eq!(
            out,
            vec![
                DecodeOutput {
                    slot_id: 0,
                    token: 1,
                    text: "<1>".into(),
                    finished: false,
                },
                DecodeOutput {
                    slot_id: 1,
                    token: 2,
                    text: "<2>".into(),
                    finished: true,
                },
            ]
        );
        assert_eq!(driver.scheduler.slots[0].last_token, Some(1));
        assert_eq!(driver.scheduler.slots[1].state, crate::batch::SlotState::Finishing);
    }

    #[test]
    fn decode_ready_once_no_ready_slots_is_noop() {
        let mut driver = BatchDriver::new(2);
        let mut engine = FakeEngine::new();
        let out = driver
            .decode_ready_once(&mut engine, 2)
            .expect("decode once");
        assert!(out.is_empty());
        assert!(engine.calls.is_empty());
    }

    #[test]
    fn admit_tokenizes_prompt_through_engine() {
        let mut driver = BatchDriver::new(1);
        let engine = FakeEngine::new();
        let slot_id = driver
            .admit(&engine, req(3))
            .expect("admit result")
            .expect("slot id");

        assert_eq!(slot_id, 0);
        assert_eq!(driver.scheduler.slots[0].prompt_ids, vec![b'x' as u32]);
        assert_eq!(driver.scheduler.slots[0].state, crate::batch::SlotState::Prefilling);
    }
}
