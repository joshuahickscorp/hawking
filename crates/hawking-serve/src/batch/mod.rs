//! Continuous batching: prefill/decode interleaving so concurrent
//! requests share MoE kernel launches. The MoE-specific win at batch ≥ 4.
//!
//! Phase 4 fills in the slot manager. The data structures that the
//! HTTP layer reads from are already shaped here so the seams don't
//! move when the implementation lands.

pub mod driver;
pub mod scheduler;

use hawking_core::{sample::Sampler, GenerateRequest};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    Idle,
    Prefilling,
    Decoding,
    Finishing,
}

#[derive(Debug)]
pub struct Slot {
    pub id: u32,
    pub state: SlotState,
    pub req: Option<GenerateRequest>,
    pub sampler: Option<Sampler>,
    pub prompt_ids: Vec<u32>,
    pub generated_ids: Vec<u32>,
    pub last_token: Option<u32>,
    pub position: usize,
    pub max_new_tokens: usize,
    /// Track 5.2: if >0, the first `prefix_skip` tokens of prompt_ids were already
    /// KV-copied from another slot. The prefill path should call
    /// prefill_slot_from_pos(slot_id, prompt_ids, prefix_skip) instead of
    /// prefill_slot(slot_id, prompt_ids) to skip re-computing those positions.
    pub prefix_skip: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeStep {
    pub slot_id: u32,
    pub token: u32,
    pub position: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodedToken {
    pub slot_id: u32,
    pub token: u32,
    pub finished: bool,
}

impl Slot {
    pub fn idle(id: u32) -> Self {
        Self {
            id,
            state: SlotState::Idle,
            req: None,
            sampler: None,
            prompt_ids: Vec::new(),
            generated_ids: Vec::new(),
            last_token: None,
            position: 0,
            max_new_tokens: 0,
            prefix_skip: 0,
        }
    }

    pub fn assign(&mut self, req: GenerateRequest, prompt_ids: Vec<u32>) {
        let seed = req
            .sampling
            .seed
            .unwrap_or(0xD15A_0000_0000_0000u64 ^ self.id as u64);
        self.sampler = Some(Sampler::new(seed));
        self.max_new_tokens = req.max_new_tokens;
        self.last_token = prompt_ids.last().copied();
        self.position = prompt_ids.len();
        self.prompt_ids = prompt_ids;
        self.generated_ids.clear();
        self.req = Some(req);
        self.state = SlotState::Prefilling;
    }

    pub fn mark_decoding(&mut self) {
        if self.state != SlotState::Idle {
            self.state = SlotState::Decoding;
        }
    }

    pub fn is_ready_to_decode(&self) -> bool {
        self.state == SlotState::Decoding
            && self.last_token.is_some()
            && self.generated_ids.len() < self.max_new_tokens
    }

    pub fn decode_step(&self) -> Option<DecodeStep> {
        Some(DecodeStep {
            slot_id: self.id,
            token: self.last_token?,
            position: self.position,
        })
        .filter(|_| self.is_ready_to_decode())
    }

    pub fn sample_next(&mut self, logits: &mut [f32]) -> Option<u32> {
        let req = self.req.as_ref()?;
        let sampler = self.sampler.as_mut()?;
        let token = sampler.sample(logits, &req.sampling);
        // Record the emitted token so the repetition penalty has history. The
        // single-stream generate() path does this; the batch path did not, so
        // the penalty was dead in `serve` and short prompts fell into `<|>`
        // repetition loops.
        sampler.record(token);
        Some(token)
    }

    pub fn record_token(&mut self, token: u32) {
        self.generated_ids.push(token);
        if let Some(sampler) = self.sampler.as_mut() {
            sampler.record(token);
        }
        self.last_token = Some(token);
        self.position += 1;
        if self.generated_ids.len() >= self.max_new_tokens {
            self.state = SlotState::Finishing;
        }
    }

    pub fn finish(&mut self) {
        if self.state != SlotState::Idle {
            self.state = SlotState::Finishing;
        }
    }

    pub fn apply_decoded_token(&mut self, token: u32, eos_id: Option<u32>) -> DecodedToken {
        self.record_token(token);
        if Some(token) == eos_id {
            self.finish();
        }
        DecodedToken {
            slot_id: self.id,
            token,
            finished: self.state == SlotState::Finishing,
        }
    }

    /// Seed the slot with the FIRST generated token, derived from the prefill's
    /// last-position prediction (returned by `Engine::prefill_slot`). Records it
    /// as the first output but does NOT advance `position`: prefill built KV for
    /// `0..prompt_len-1`, so this token is fed at `position` (= prompt_len) on the
    /// next decode step to produce the SECOND token. Re-feeding the last PROMPT
    /// token here instead (the old behaviour) produced a spurious leading word on
    /// every response because the decode kernels diverge from the batch prefill.
    pub fn seed_first_token(&mut self, token: u32, eos_id: Option<u32>) -> DecodedToken {
        self.generated_ids.push(token);
        if let Some(sampler) = self.sampler.as_mut() {
            sampler.record(token);
        }
        self.last_token = Some(token);
        if Some(token) == eos_id || self.generated_ids.len() >= self.max_new_tokens {
            self.state = SlotState::Finishing;
        }
        DecodedToken {
            slot_id: self.id,
            token,
            finished: self.state == SlotState::Finishing,
        }
    }

    pub fn release(&mut self) {
        let id = self.id;
        *self = Self::idle(id);
    }
}
