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

/// Track 5.4 — batch admission policy.
///
/// Controls how `ready_decode_indices` orders and selects slots.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum BatchPolicy {
    /// Admit any ready slots up to max_batch (current behavior).
    #[default]
    Default,
    /// Prefer greedy (temperature=0) slots over sampling slots.
    ///
    /// Sorting greedy slots first maximises the probability that
    /// `decode_ready_once`'s `all_greedy` check succeeds, routing the step
    /// to the efficient token-only lane (B×4 byte readback, no logits).
    GreedyFirst,
    /// Fill batch with slots that share a common prefix (for amortised prefill).
    ///
    /// When multiple slots have matching prompt prefixes, grouping them lets a
    /// single prefill pass cover the shared prefix once, then branch.
    PrefixGrouped,
}

/// Track 5.1 — prefix reuse detection.
///
/// For each active slot, store a 64-bit hash of its prompt token sequence.
/// On admit, check for any active slot whose prefix matches the new request's
/// prefix at length L. When a match is found the caller can skip prefill for
/// the matching prefix and plant the existing KV into the new slot directly.
///
/// This is the data-plane scaffold; the actual KV-copy path lives in the engine.
/// The scheduler exposes `find_prefix_match` to the serve layer.
#[derive(Debug, Default)]
pub struct PrefixIndex {
    /// Map: slot_id → (hash, prefix_len). Updated on every admit.
    entries: Vec<(u32, u64, usize)>, // (slot_id, prefix_hash, len)
}

/// Hash a token sequence with FNV-1a. Fast, no dep.
fn hash_tokens(tokens: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &t in tokens {
        let bytes = t.to_le_bytes();
        for b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

impl PrefixIndex {
    pub fn upsert(&mut self, slot_id: u32, prompt_ids: &[u32]) {
        let h = hash_tokens(prompt_ids);
        if let Some(e) = self.entries.iter_mut().find(|e| e.0 == slot_id) {
            e.1 = h;
            e.2 = prompt_ids.len();
        } else {
            self.entries.push((slot_id, h, prompt_ids.len()));
        }
    }

    pub fn remove(&mut self, slot_id: u32) {
        self.entries.retain(|e| e.0 != slot_id);
    }

    /// Find the longest prefix match for `tokens` among active entries,
    /// excluding `exclude_slot`. Used after a new slot has been admitted
    /// (and therefore already inserted into the index) to find a **different**
    /// slot whose cached KV can be copied.
    ///
    /// Only considers prefixes of length ≥ `min_len`.
    pub fn find_prefix_match_excluding(
        &self,
        tokens: &[u32],
        min_len: usize,
        exclude_slot: u32,
    ) -> Option<(u32, usize)> {
        let mut best: Option<(u32, usize)> = None;
        for &(slot_id, stored_hash, stored_len) in &self.entries {
            if slot_id == exclude_slot {
                continue;
            }
            if stored_len < min_len {
                continue;
            }
            let overlap = stored_len.min(tokens.len());
            if overlap < min_len {
                continue;
            }
            let request_prefix_hash = hash_tokens(&tokens[..overlap]);
            if stored_hash == request_prefix_hash {
                if best.map(|(_, bl)| overlap > bl).unwrap_or(true) {
                    best = Some((slot_id, overlap));
                }
            }
        }
        best
    }

    /// Find the longest prefix match for `tokens` among active entries.
    /// Returns `(slot_id, shared_len)` of the best match, or `None`.
    /// Only considers prefixes of length ≥ `min_len`.
    pub fn find_prefix_match(&self, tokens: &[u32], min_len: usize) -> Option<(u32, usize)> {
        let mut best: Option<(u32, usize)> = None;
        for &(slot_id, stored_hash, stored_len) in &self.entries {
            if stored_len < min_len {
                continue;
            }
            let overlap = stored_len.min(tokens.len());
            if overlap < min_len {
                continue;
            }
            let request_prefix_hash = hash_tokens(&tokens[..overlap]);
            if stored_hash == request_prefix_hash {
                if best.map(|(_, bl)| overlap > bl).unwrap_or(true) {
                    best = Some((slot_id, overlap));
                }
            }
        }
        best
    }
}

/// Bucket index for prompt-length batching (bucket edges: 0-16, 17-64,
/// 65-256, 257-1024, 1025+). Adjacent slots in the same bucket have
/// prompt lengths within ~4× of each other.
#[inline]
fn prompt_length_bucket(len: usize) -> usize {
    match len {
        0..=16     => 0,
        17..=64    => 1,
        65..=256   => 2,
        257..=1024 => 3,
        _          => 4,
    }
}

/// Length of the longest common prefix of two token slices.
#[inline]
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Track 5.4 — prefix-affinity prefill cohort selection (PURE).
///
/// Given the full slot table, pick the set of `Prefilling` slots that share the
/// LONGEST common token prefix, returning their slot ids (capped at `max_batch`).
/// Batching same-prefix prompts lets one prefill pass cover the shared prefix
/// once (KV computed once, then branched per slot) instead of N times.
///
/// Determinism: candidates are processed in ascending slot-id order; the winning
/// group maximizes (shared_prefix_len, group_size) with the smallest anchor
/// slot-id as the final tie-break — a pure deterministic function of the table.
///
/// Latency-safety: when NO group of size >= 2 with shared_len >= `min_shared`
/// exists, fall back to admitting Prefilling slots in slot-id order (the same
/// set the Default/bucketed paths would admit) so a unique request is never
/// starved waiting for a co-prefix partner.
pub fn group_by_prefix(slots: &[Slot], max_batch: usize, min_shared: usize) -> Vec<u32> {
    // Collect Prefilling candidates as (slot_id, prompt_ids), ascending by id.
    let mut cands: Vec<(u32, &[u32])> = slots
        .iter()
        .filter(|s| s.state == SlotState::Prefilling)
        .map(|s| (s.id, s.prompt_ids.as_slice()))
        .collect();
    cands.sort_by_key(|&(id, _)| id);
    if cands.is_empty() || max_batch == 0 {
        return Vec::new();
    }

    let mut best: Option<(usize, usize, u32, Vec<u32>)> = None; // (shared, size, anchor_id, ids)
    for ai in 0..cands.len() {
        let (anchor_id, anchor_ids) = cands[ai];
        // (cpl_with_anchor, slot_id) for all other candidates.
        let mut partners: Vec<(usize, u32)> = cands
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != ai)
            .map(|(_, &(id, ids))| (common_prefix_len(anchor_ids, ids), id))
            .collect();
        // Descending by cpl; tie-break ascending slot-id for determinism.
        partners.sort_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)));
        let cap_partners = max_batch.saturating_sub(1).min(partners.len());
        for k in 1..=cap_partners {
            let shared_len = partners[k - 1].0; // k-th largest cpl (1-indexed)
            if shared_len < min_shared {
                break; // further k only lowers shared_len (sorted desc)
            }
            let size = k + 1;
            let mut group: Vec<u32> = Vec::with_capacity(size);
            group.push(anchor_id);
            for &(_, pid) in &partners[..k] {
                group.push(pid);
            }
            group.sort_unstable();
            let better = match &best {
                None => true,
                Some((bs, bz, ba, _)) => {
                    (shared_len, size).cmp(&(*bs, *bz)) == std::cmp::Ordering::Greater
                        || ((shared_len, size) == (*bs, *bz) && anchor_id < *ba)
                }
            };
            if better {
                best = Some((shared_len, size, anchor_id, group));
            }
        }
    }

    match best {
        Some((_, _, _, ids)) => ids.into_iter().take(max_batch).collect(),
        None => {
            // Latency-safety: no qualifying group -> FIFO admit by slot id.
            cands.into_iter().take(max_batch).map(|(id, _)| id).collect()
        }
    }
}

pub struct Scheduler {
    pub slots: Vec<Slot>,
    pub max_batch_size: usize,
    /// Track 5.1: prefix hash index for KV reuse detection.
    pub prefix_index: PrefixIndex,
    /// Track 5.4: batch admission policy (default = FIFO).
    pub policy: BatchPolicy,
}

impl Scheduler {
    pub fn new(max_batch_size: usize) -> Self {
        let slots = (0..max_batch_size as u32)
            .map(Slot::idle)
            .collect();
        Self {
            slots,
            max_batch_size,
            prefix_index: PrefixIndex::default(),
            policy: BatchPolicy::Default,
        }
    }

    pub fn idle_slot(&mut self) -> Option<&mut Slot> {
        self.slots.iter_mut().find(|s| s.state == SlotState::Idle)
    }

    pub fn admit(&mut self, req: GenerateRequest, prompt_ids: Vec<u32>) -> Option<u32> {
        let id = self.slots.iter().find(|s| s.state == SlotState::Idle)?.id;
        self.prefix_index.upsert(id, &prompt_ids);
        let slot = self.slots.iter_mut().find(|s| s.id == id)?;
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
        self.prefix_index.remove(id);
        true
    }

    pub fn ready_decode_indices(&self, max: usize) -> Vec<usize> {
        let cap = max.min(self.max_batch_size);
        match self.policy {
            BatchPolicy::Default | BatchPolicy::PrefixGrouped => {
                // Default: FIFO order by slot index.
                self.slots
                    .iter()
                    .enumerate()
                    .filter(|(_, slot)| slot.is_ready_to_decode())
                    .take(cap)
                    .map(|(idx, _)| idx)
                    .collect()
            }
            BatchPolicy::GreedyFirst => {
                // Sort ready slots: greedy (temp=0, no rep-penalty) first,
                // then sampling slots. Within each group, preserve slot-index order.
                let mut greedy: Vec<usize> = Vec::new();
                let mut sampled: Vec<usize> = Vec::new();
                for (idx, slot) in self.slots.iter().enumerate() {
                    if !slot.is_ready_to_decode() {
                        continue;
                    }
                    let is_greedy = slot
                        .req
                        .as_ref()
                        .map(|r| {
                            r.sampling.temperature <= 0.0
                                && r.sampling.repetition_penalty <= 1.0
                        })
                        .unwrap_or(false);
                    if is_greedy {
                        greedy.push(idx);
                    } else {
                        sampled.push(idx);
                    }
                }
                greedy.extend(sampled);
                greedy.truncate(cap);
                greedy
            }
        }
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

    /// Bucketed variant: pick at most `max` Prefilling slots from the single
    /// prompt-length bucket with the most queued work. Slots in a bucket have
    /// similar prompt lengths, so the parallel-prefill position loop exits at
    /// the right depth rather than being dragged by a long outlier.
    ///
    /// Tie-break: prefer the larger bucket index (longer prompts get
    /// batched together since they have the highest prefill cost).
    /// Degenerates to `prefill_slots` when all slots are in the same bucket.
    ///
    /// Bucket edges: [0,16] [17,64] [65,256] [257,1024] [1025+]
    pub fn prefill_slots_bucketed(&self, max: usize) -> Vec<u32> {
        let candidates: Vec<(usize, usize, u32)> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.state == SlotState::Prefilling)
            .map(|(idx, s)| (prompt_length_bucket(s.prompt_ids.len()), idx, s.id))
            .collect();
        if candidates.is_empty() {
            return Vec::new();
        }
        let mut bucket_counts = [0usize; 5];
        for &(b, _, _) in &candidates {
            bucket_counts[b] += 1;
        }
        // Compare by (count, bucket_index) so ties resolve toward larger bucket
        // (longer prompts). max_by is used instead of max_by_key so the
        // comparator can inspect both count and index simultaneously.
        let best_bucket = bucket_counts
            .iter()
            .enumerate()
            .max_by(|&(b1, &c1), &(b2, &c2)| c1.cmp(&c2).then(b1.cmp(&b2)))
            .map(|(b, _)| b)
            .unwrap_or(0);
        candidates
            .into_iter()
            .filter(|&(b, _, _)| b == best_bucket)
            .take(max.min(self.max_batch_size))
            .map(|(_, _, id)| id)
            .collect()
    }

    /// Track 5.4 — prefix-affinity prefill selector.
    ///
    /// When `policy == PrefixGrouped`, return the same-prefix cohort from
    /// `group_by_prefix`; otherwise fall back to the length-bucketed selector.
    /// `min_shared = 8` matches the serve layer's `find_prefix_match_excluding`
    /// threshold (8 tokens) so the chosen cohort is also a KV-copy candidate.
    pub fn prefill_slots_prefix_grouped(&self, max: usize) -> Vec<u32> {
        let cap = max.min(self.max_batch_size);
        match self.policy {
            BatchPolicy::PrefixGrouped => group_by_prefix(&self.slots, cap, 8),
            _ => self.prefill_slots_bucketed(cap),
        }
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

    /// Greedy token-only variant: token ids arrive pre-sampled (from GPU argmax),
    /// no logits involved. Slot validity checks mirror apply_decode_logits.
    pub fn apply_decode_tokens(
        &mut self,
        batch: &[DecodeStep],
        token_ids: Vec<u32>,
        eos_id: Option<u32>,
    ) -> Result<Vec<DecodedToken>> {
        if batch.len() != token_ids.len() {
            return Err(anyhow!(
                "decode tokens shape mismatch: batch={} tokens={}",
                batch.len(),
                token_ids.len()
            ));
        }
        let mut out = Vec::with_capacity(batch.len());
        for (step, token) in batch.iter().zip(token_ids.into_iter()) {
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

    #[test]
    fn bucketed_prefill_selects_homogeneous_bucket() {
        // 4 short slots (bucket 0) + 1 long slot (bucket 3).
        // Bucketed selector must pick the 4-slot bucket.
        let mut scheduler = Scheduler::new(8);
        for _ in 0..4 {
            scheduler.admit(req(4), (0..8u32).collect()).expect("admit short");
        }
        scheduler.admit(req(4), (0..512u32).collect()).expect("admit long");

        let chosen = scheduler.prefill_slots_bucketed(8);
        assert_eq!(chosen.len(), 4, "should pick all 4 short-prompt slots");
        assert!(!chosen.contains(&4), "long slot must not be in the chosen batch");
    }

    #[test]
    fn bucketed_prefill_tie_break_favours_longer_bucket() {
        // 2 short (bucket 0) vs 2 long (bucket 3) — tie; long wins.
        let mut scheduler = Scheduler::new(8);
        for _ in 0..2 {
            scheduler.admit(req(4), (0..8u32).collect()).expect("admit short");
        }
        for _ in 0..2 {
            scheduler.admit(req(4), (0..512u32).collect()).expect("admit long");
        }
        let chosen = scheduler.prefill_slots_bucketed(8);
        assert_eq!(chosen.len(), 2);
        assert!(chosen.iter().all(|&id| id >= 2), "tie should choose long bucket");
    }

    #[test]
    fn bucketed_prefill_homogeneous_queue_matches_plain() {
        let mut scheduler = Scheduler::new(4);
        for _ in 0..4 {
            scheduler.admit(req(4), (0..32u32).collect()).expect("admit");
        }
        assert_eq!(
            scheduler.prefill_slots_bucketed(4),
            scheduler.prefill_slots(4),
        );
    }

    fn prefilling(scheduler: &mut Scheduler, prompts: &[(u32, Vec<u32>)]) {
        for (id, ids) in prompts {
            let slot = scheduler.slot_mut(*id).expect("slot");
            slot.assign(req(8), ids.clone()); // assign -> SlotState::Prefilling
        }
    }

    #[test]
    fn group_by_prefix_cobatches_shared_prefix() {
        // slots 0,1,2 share a 10-token prefix; slot 3 is unrelated.
        let shared: Vec<u32> = (100..110).collect();
        let mut a = shared.clone(); a.push(1);
        let mut b = shared.clone(); b.push(2);
        let mut c = shared.clone(); c.push(3);
        let d: Vec<u32> = (900..912).collect();
        let mut scheduler = Scheduler::new(4);
        prefilling(&mut scheduler, &[(0, a), (1, b), (2, c), (3, d)]);

        let chosen = group_by_prefix(&scheduler.slots, 4, 8);
        // The 3 shared-prefix slots must co-batch; the unrelated one excluded.
        assert!(chosen.contains(&0) && chosen.contains(&1) && chosen.contains(&2),
            "shared-prefix trio must co-batch, got {chosen:?}");
        assert!(!chosen.contains(&3), "unrelated slot must not join, got {chosen:?}");
    }

    #[test]
    fn group_by_prefix_lone_unique_request_still_admits() {
        // Single Prefilling slot with a unique prompt: must admit promptly
        // (latency-safety), not return empty waiting for a co-prefix partner.
        let mut scheduler = Scheduler::new(4);
        prefilling(&mut scheduler, &[(2, (500..520).collect())]);
        let chosen = group_by_prefix(&scheduler.slots, 4, 8);
        assert_eq!(chosen, vec![2], "lone request must admit, got {chosen:?}");
    }

    #[test]
    fn group_by_prefix_no_shared_prefix_falls_back_fifo() {
        // Two slots, disjoint prompts (shared prefix 0 < min_shared) -> FIFO set,
        // not starvation. Both admit, in slot-id order.
        let mut scheduler = Scheduler::new(4);
        prefilling(&mut scheduler, &[(0, vec![1,2,3,4,5,6,7,8,9]), (1, vec![90,91,92,93,94,95,96,97,98])]);
        let chosen = group_by_prefix(&scheduler.slots, 4, 8);
        assert_eq!(chosen, vec![0, 1], "disjoint prompts should FIFO-admit both, got {chosen:?}");
    }

    #[test]
    fn group_by_prefix_deterministic_tie_break_prefers_longer_then_lower_anchor() {
        // Group X = slots {0,1} share 12 tokens. Group Y = slots {2,3} share 8.
        // Longer shared prefix (X) must win regardless of slot order.
        let px: Vec<u32> = (0..12).collect();
        let mut x0 = px.clone(); x0.push(70);
        let mut x1 = px.clone(); x1.push(71);
        let py: Vec<u32> = (200..208).collect();
        let mut y0 = py.clone(); y0.push(72);
        let mut y1 = py.clone(); y1.push(73);
        let mut scheduler = Scheduler::new(4);
        prefilling(&mut scheduler, &[(0, x0), (1, x1), (2, y0), (3, y1)]);
        let chosen = group_by_prefix(&scheduler.slots, 2, 8);
        assert_eq!(chosen, vec![0, 1], "longer-shared-prefix group must win, got {chosen:?}");
        // Determinism: identical inputs -> identical output.
        assert_eq!(chosen, group_by_prefix(&scheduler.slots, 2, 8));
    }

    #[test]
    fn prefill_slots_prefix_grouped_falls_back_when_policy_off() {
        // With BatchPolicy::Default the prefix selector must equal the bucketed one.
        let mut scheduler = Scheduler::new(4);
        for _ in 0..4 { scheduler.admit(req(4), (0..32u32).collect()).expect("admit"); }
        assert_eq!(
            scheduler.prefill_slots_prefix_grouped(4),
            scheduler.prefill_slots_bucketed(4),
        );
    }
}
