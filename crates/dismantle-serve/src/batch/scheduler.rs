//! Slot manager. Each slot owns one in-flight request's KV cache and
//! decode position. The scheduler picks slots that are ready for
//! prefill/decode and packs them into one model forward pass per
//! step, so each MoE kernel launch amortizes across all active slots.
//!
//! v0.1.0 Phase 4: real implementation. Until then, single-request
//! mode is the only path; the HTTP layer takes the engine mutex
//! directly per request.

use crate::batch::{DecodeStep, DecodedToken, Slot, SlotState};
use anyhow::{anyhow, Result};
use dismantle_core::GenerateRequest;

pub struct Scheduler {
    pub slots: Vec<Slot>,
    pub max_batch_size: usize,
}

impl Scheduler {
    pub fn new(max_batch_size: usize) -> Self {
        let slots = (0..max_batch_size as u32)
            .map(Slot::idle)
            .collect();
        Self {
            slots,
            max_batch_size,
        }
    }

    pub fn idle_slot(&mut self) -> Option<&mut Slot> {
        self.slots.iter_mut().find(|s| s.state == SlotState::Idle)
    }

    pub fn admit(&mut self, req: GenerateRequest, prompt_ids: Vec<u32>) -> Option<u32> {
        let slot = self.idle_slot()?;
        let id = slot.id;
        slot.assign(req, prompt_ids);
        Some(id)
    }

    pub fn active_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.state != SlotState::Idle)
            .count()
    }

    pub fn slot_mut(&mut self, id: u32) -> Option<&mut Slot> {
        self.slots.iter_mut().find(|s| s.id == id)
    }

    pub fn release_slot(&mut self, id: u32) -> bool {
        let Some(slot) = self.slot_mut(id) else {
            return false;
        };
        slot.release();
        true
    }

    pub fn ready_decode_indices(&self, max: usize) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.is_ready_to_decode())
            .take(max.min(self.max_batch_size))
            .map(|(idx, _)| idx)
            .collect()
    }

    pub fn ready_decode_slots(&self, max: usize) -> Vec<u32> {
        self.ready_decode_indices(max)
            .into_iter()
            .map(|idx| self.slots[idx].id)
            .collect()
    }

    pub fn prefill_indices(&self, max: usize) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.state == SlotState::Prefilling)
            .take(max.min(self.max_batch_size))
            .map(|(idx, _)| idx)
            .collect()
    }

    pub fn prefill_slots(&self, max: usize) -> Vec<u32> {
        self.prefill_indices(max)
            .into_iter()
            .map(|idx| self.slots[idx].id)
            .collect()
    }

    pub fn mark_prefill_complete(&mut self, id: u32) -> bool {
        let Some(slot) = self.slot_mut(id) else {
            return false;
        };
        if slot.state != SlotState::Prefilling {
            return false;
        }
        slot.mark_decoding();
        true
    }

    pub fn decode_batch(&self, max: usize) -> Vec<DecodeStep> {
        self.ready_decode_indices(max)
            .into_iter()
            .filter_map(|idx| self.slots[idx].decode_step())
            .collect()
    }

    pub fn apply_decode_logits(
        &mut self,
        batch: &[DecodeStep],
        logits: &mut [Vec<f32>],
        eos_id: Option<u32>,
    ) -> Result<Vec<DecodedToken>> {
        if batch.len() != logits.len() {
            return Err(anyhow!(
                "decode result shape mismatch: batch={} logits={}",
                batch.len(),
                logits.len()
            ));
        }

        let mut out = Vec::with_capacity(batch.len());
        for (step, logits) in batch.iter().zip(logits.iter_mut()) {
            let slot = self
                .slot_mut(step.slot_id)
                .ok_or_else(|| anyhow!("decode result for unknown slot {}", step.slot_id))?;
            if slot.decode_step() != Some(*step) {
                return Err(anyhow!(
                    "stale decode result for slot {}: expected {:?}, got {:?}",
                    step.slot_id,
                    slot.decode_step(),
                    step
                ));
            }
            let token = slot
                .sample_next(logits)
                .ok_or_else(|| anyhow!("slot {} cannot sample decode result", step.slot_id))?;
            out.push(slot.apply_decoded_token(token, eos_id));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dismantle_core::{GenerateRequest, SamplingParams};

    fn req(max_new_tokens: usize) -> GenerateRequest {
        GenerateRequest {
            prompt: "hello".into(),
            max_new_tokens,
            sampling: SamplingParams::default(),
            stop: Vec::new(),
            abort: None,
            max_stall_ms: 0,
        }
    }

    #[test]
    fn scheduler_starts_with_idle_slots() {
        let scheduler = Scheduler::new(3);
        assert_eq!(scheduler.active_count(), 0);
        assert_eq!(scheduler.slots.len(), 3);
        assert!(scheduler.slots.iter().all(|slot| slot.state == SlotState::Idle));
    }

    #[test]
    fn slot_assignment_tracks_decode_cursor() {
        let mut scheduler = Scheduler::new(1);
        let slot_id = scheduler.admit(req(4), vec![10, 11]).expect("admit");
        let slot = scheduler.slot_mut(slot_id).expect("slot");

        assert_eq!(slot.state, SlotState::Prefilling);
        assert_eq!(slot.last_token, Some(11));
        assert_eq!(slot.position, 2);
        assert!(!slot.is_ready_to_decode());

        assert!(scheduler.mark_prefill_complete(slot_id));
        let slot = scheduler.slot_mut(slot_id).expect("slot");
        assert!(slot.is_ready_to_decode());
        slot.record_token(12);
        assert_eq!(slot.generated_ids, vec![12]);
        assert_eq!(slot.last_token, Some(12));
        assert_eq!(slot.position, 3);
    }

    #[test]
    fn ready_decode_slots_respects_limit() {
        let mut scheduler = Scheduler::new(4);
        for id in 0..3 {
            let slot = scheduler.slot_mut(id).expect("slot");
            slot.assign(req(8), vec![id + 1]);
            slot.mark_decoding();
        }

        assert_eq!(scheduler.ready_decode_indices(2), vec![0, 1]);
        assert_eq!(scheduler.ready_decode_slots(8), vec![0, 1, 2]);
        assert_eq!(
            scheduler.decode_batch(2),
            vec![
                DecodeStep {
                    slot_id: 0,
                    token: 1,
                    position: 1,
                },
                DecodeStep {
                    slot_id: 1,
                    token: 2,
                    position: 1,
                },
            ]
        );
    }

    #[test]
    fn release_slot_resets_state() {
        let mut scheduler = Scheduler::new(1);
        let slot_id = scheduler.admit(req(1), vec![7]).expect("admit");
        assert!(scheduler.mark_prefill_complete(slot_id));
        scheduler.slots[0].record_token(8);

        assert_eq!(scheduler.slots[0].state, SlotState::Finishing);
        assert!(scheduler.release_slot(0));
        assert_eq!(scheduler.active_count(), 0);
        assert_eq!(scheduler.slots[0].state, SlotState::Idle);
        assert!(scheduler.slots[0].req.is_none());
        assert!(scheduler.slots[0].prompt_ids.is_empty());
    }

    #[test]
    fn apply_decode_logits_samples_and_advances_slots() {
        let mut scheduler = Scheduler::new(2);
        for id in 0..2 {
            let mut r = req(2);
            r.sampling.temperature = 0.0;
            let slot_id = scheduler.admit(r, vec![10 + id]).expect("admit");
            assert_eq!(slot_id, id);
            assert!(scheduler.mark_prefill_complete(slot_id));
        }

        let batch = scheduler.decode_batch(2);
        let mut logits = vec![
            vec![0.0, 3.0, 1.0],
            vec![0.0, 1.0, 5.0],
        ];
        let decoded = scheduler
            .apply_decode_logits(&batch, &mut logits, Some(2))
            .expect("apply logits");

        assert_eq!(
            decoded,
            vec![
                DecodedToken {
                    slot_id: 0,
                    token: 1,
                    finished: false,
                },
                DecodedToken {
                    slot_id: 1,
                    token: 2,
                    finished: true,
                },
            ]
        );
        assert_eq!(scheduler.slots[0].last_token, Some(1));
        assert_eq!(scheduler.slots[0].position, 2);
        assert_eq!(scheduler.slots[1].state, SlotState::Finishing);
    }

    #[test]
    fn admission_and_prefill_slots_track_lifecycle() {
        let mut scheduler = Scheduler::new(2);
        let first = scheduler.admit(req(4), vec![1]).expect("first slot");
        let second = scheduler.admit(req(4), vec![2]).expect("second slot");
        assert_eq!((first, second), (0, 1));
        assert!(scheduler.admit(req(4), vec![3]).is_none());

        assert_eq!(scheduler.prefill_slots(8), vec![0, 1]);
        assert!(scheduler.mark_prefill_complete(first));
        assert_eq!(scheduler.prefill_slots(8), vec![1]);
        assert_eq!(scheduler.ready_decode_slots(8), vec![0]);
        assert!(!scheduler.mark_prefill_complete(first));
    }
}
