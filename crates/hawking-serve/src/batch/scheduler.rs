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
use hawking_core::GenerateRequest;

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
    /// One entry per live slot.  Keep rolling hashes for a cheap candidate
    /// lookup *and* the token IDs for collision-free prefix admission.
    ///
    /// This is deliberately a small in-memory index (at most `max_batch_size`
    /// requests), not a second KV store.  It follows the block/prefix cache
    /// admission principle used by vLLM: hash to find a candidate, then verify
    /// the immutable token identity before reusing device state.
    entries: Vec<PrefixEntry>,
}

#[derive(Debug, Clone)]
struct PrefixEntry {
    slot_id: u32,
    tokens: Vec<u32>,
    /// FNV-1a hash after each token. `rolling_hashes[n - 1]` is the hash of
    /// `tokens[..n]`, making arbitrary common-prefix probes O(1) after the
    /// request-local rolling hash is built.
    rolling_hashes: Vec<u64>,
}

/// FNV-1a after each token in a sequence. Fast, deterministic, no dependency.
/// Hashes are candidate filters only; every cache admission below compares IDs.
fn rolling_token_hashes(tokens: &[u32]) -> Vec<u64> {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut out = Vec::with_capacity(tokens.len());
    for &t in tokens {
        let bytes = t.to_le_bytes();
        for b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        out.push(h);
    }
    out
}

impl PrefixIndex {
    pub fn upsert(&mut self, slot_id: u32, prompt_ids: &[u32]) {
        let entry = PrefixEntry {
            slot_id,
            tokens: prompt_ids.to_vec(),
            rolling_hashes: rolling_token_hashes(prompt_ids),
        };
        if let Some(e) = self.entries.iter_mut().find(|e| e.slot_id == slot_id) {
            *e = entry;
        } else {
            self.entries.push(entry);
        }
    }

    pub fn remove(&mut self, slot_id: u32) {
        self.entries.retain(|e| e.slot_id != slot_id);
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
        let requested_hashes = rolling_token_hashes(tokens);
        for entry in &self.entries {
            if entry.slot_id == exclude_slot {
                continue;
            }
            if entry.tokens.len() < min_len {
                continue;
            }
            let overlap = entry.tokens.len().min(tokens.len());
            if overlap < min_len {
                continue;
            }
            if entry.rolling_hashes[overlap - 1] == requested_hashes[overlap - 1]
                && entry.tokens[..overlap] == tokens[..overlap]
                && best.map(|(_, bl)| overlap > bl).unwrap_or(true)
            {
                best = Some((entry.slot_id, overlap));
            }
        }
        best
    }

    /// Find the longest prefix match for `tokens` among active entries.
    /// Returns `(slot_id, shared_len)` of the best match, or `None`.
    /// Only considers prefixes of length ≥ `min_len`.
    pub fn find_prefix_match(&self, tokens: &[u32], min_len: usize) -> Option<(u32, usize)> {
        let mut best: Option<(u32, usize)> = None;
        let requested_hashes = rolling_token_hashes(tokens);
        for entry in &self.entries {
            if entry.tokens.len() < min_len {
                continue;
            }
            let overlap = entry.tokens.len().min(tokens.len());
            if overlap < min_len {
                continue;
            }
            if entry.rolling_hashes[overlap - 1] == requested_hashes[overlap - 1]
                && entry.tokens[..overlap] == tokens[..overlap]
                && best.map(|(_, bl)| overlap > bl).unwrap_or(true)
            {
                best = Some((entry.slot_id, overlap));
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
        0..=16 => 0,
        17..=64 => 1,
        65..=256 => 2,
        257..=1024 => 3,
        _ => 4,
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
            cands
                .into_iter()
                .take(max_batch)
                .map(|(id, _)| id)
                .collect()
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
    /// First slot eligible for the next bounded prefill turn.  This is not a
    /// second scheduler: it only rotates the already policy-ranked cohort
    /// after one capped chunk has made progress, preventing a long first slot
    /// from consuming every prefill turn ahead of its peers.
    prefill_next_slot: u32,
}

/// One bounded contiguous range selected for the next prefill submission.
/// Coordinates are relative to the request's original prompt, so successive
/// chunks keep using the same stable slot-resident KV region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillChunk {
    pub slot_id: u32,
    pub start: usize,
    pub end: usize,
    pub complete: bool,
}

impl Scheduler {
    pub fn new(max_batch_size: usize) -> Self {
        let slots = (0..max_batch_size as u32).map(Slot::idle).collect();
        Self {
            slots,
            max_batch_size,
            prefix_index: PrefixIndex::default(),
            policy: BatchPolicy::Default,
            prefill_next_slot: 0,
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

    /// Longest exact prefix held by a slot whose KV is already materialized.
    /// `PrefixIndex` intentionally includes newly admitted `Prefilling` slots
    /// for grouping, but those slots have no valid device KV yet and must not
    /// be used as copy sources.
    pub fn find_decoding_prefix_match_excluding(
        &self,
        tokens: &[u32],
        min_len: usize,
        exclude_slot: u32,
    ) -> Option<(u32, usize)> {
        self.slots
            .iter()
            .filter(|slot| slot.id != exclude_slot && slot.state == SlotState::Decoding)
            .filter_map(|slot| {
                let shared = common_prefix_len(&slot.prompt_ids, tokens);
                (shared >= min_len).then_some((slot.id, shared))
            })
            .max_by_key(|(_, shared)| *shared)
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
                            r.sampling.temperature <= 0.0 && r.sampling.repetition_penalty <= 1.0
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

    /// Apply an exact token-work cap after the normal policy has ranked a
    /// prefill cohort. This is the continuous-batching admission control that
    /// prevents one batch of long prompts from turning into an unbounded TTFT
    /// and p99 stall. `prefix_skip` is already resident KV, so only the
    /// remaining tokens are charged.
    ///
    /// Progress is deliberate: the first ranked request is always admitted,
    /// even when it alone exceeds `max_prefill_tokens`; otherwise a long
    /// request could wait forever behind a cap it can never fit. A zero budget
    /// therefore means "one request at a time", not "make no progress".
    pub fn prefill_slots_token_budgeted(&self, max: usize, max_prefill_tokens: usize) -> Vec<u32> {
        let ranked = self.prefill_slots_prefix_grouped(max);
        let mut selected = Vec::with_capacity(ranked.len());
        let mut charged = 0usize;
        for id in ranked {
            let Some(slot) = self.slots.iter().find(|slot| slot.id == id) else {
                continue;
            };
            let remaining = slot.prompt_ids.len().saturating_sub(slot.prefix_skip);
            if selected.is_empty() || charged.saturating_add(remaining) <= max_prefill_tokens {
                charged = charged.saturating_add(remaining);
                selected.push(id);
            }
        }
        selected
    }

    /// Plan bounded contiguous prompt work after the ordinary prefix/length
    /// policy ranks a cohort. This imports chunked prefill's latency boundary
    /// without changing the engine or duplicating a scheduler: a long request
    /// advances under the cap and retains its exact per-slot KV state.
    ///
    /// A zero budget means one-token progress rather than a deadlocked
    /// request. The returned work is not committed until the caller reports a
    /// successful engine submission through `commit_prefill_chunk`.
    pub fn prefill_chunks_token_budgeted(
        &mut self,
        max: usize,
        max_prefill_tokens: usize,
    ) -> Vec<PrefillChunk> {
        let mut ranked = self.prefill_slots_prefix_grouped(max);
        if let Some(offset) = ranked.iter().position(|id| *id >= self.prefill_next_slot) {
            ranked.rotate_left(offset);
        }
        let mut selected = Vec::with_capacity(ranked.len());
        let mut budget = max_prefill_tokens.max(1);
        for id in ranked {
            if budget == 0 {
                break;
            }
            let Some(slot) = self.slots.iter().find(|slot| slot.id == id) else {
                continue;
            };
            let len = slot.prompt_ids.len();
            let start = slot.prefill_cursor.max(slot.prefix_skip).min(len);
            let outstanding = len.saturating_sub(start);
            if outstanding == 0 {
                continue;
            }
            let end = start + outstanding.min(budget);
            budget -= end - start;
            selected.push(PrefillChunk {
                slot_id: id,
                start,
                end,
                complete: end == len,
            });
        }
        if let Some(last) = selected.last() {
            self.prefill_next_slot = (last.slot_id + 1) % self.max_batch_size.max(1) as u32;
        }
        selected
    }

    /// Advance only a successfully executed, contiguous prefill range. This
    /// cannot transition a slot to decoding; the final chunk is still subject
    /// to the existing complete-prompt and first-token path.
    pub fn commit_prefill_chunk(&mut self, id: u32, end: usize) -> bool {
        let Some(slot) = self.slot_mut(id) else {
            return false;
        };
        if slot.state != SlotState::Prefilling {
            return false;
        }
        let start = slot
            .prefill_cursor
            .max(slot.prefix_skip)
            .min(slot.prompt_ids.len());
        if end < start || end > slot.prompt_ids.len() {
            return false;
        }
        slot.prefill_cursor = end;
        // Imported-prefix accounting is consumed by the first successful
        // chunk. Later chunks continue from the materialized cursor itself.
        slot.prefix_skip = 0;
        true
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

    /// Seed a slot's first generated token after prefill (see
    /// `Slot::seed_first_token`). Returns the `DecodedToken` so the caller can
    /// stream the text and release the slot if it is already an EOS.
    pub fn seed_first_token(
        &mut self,
        id: u32,
        token: u32,
        eos_id: Option<u32>,
    ) -> Option<DecodedToken> {
        self.slot_mut(id).map(|s| s.seed_first_token(token, eos_id))
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
        for (step, token) in batch.iter().zip(token_ids) {
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
    use hawking_core::{GenerateRequest, SamplingParams};
    fn req(max_new_tokens: usize) -> GenerateRequest {
        GenerateRequest {
            prompt: "hello".into(),
            max_new_tokens,
            sampling: SamplingParams::default(),
            stop: Vec::new(),
            abort: None,
            max_stall_ms: 0,
            json_mode: false,
        }
    }
    #[test]
    fn scheduler_starts_with_idle_slots() {
        let scheduler = Scheduler::new(3);
        assert_eq!(scheduler.active_count(), 0);
        assert_eq!(scheduler.slots.len(), 3);
        assert!(scheduler
            .slots
            .iter()
            .all(|slot| slot.state == SlotState::Idle));
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
        let mut logits = vec![vec![0.0, 3.0, 1.0], vec![0.0, 1.0, 5.0]];
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
    fn copy_source_must_have_materialized_kv() {
        let mut scheduler = Scheduler::new(2);
        let source = scheduler
            .admit(req(4), (10..22).collect())
            .expect("source slot");
        let target = scheduler
            .admit(req(4), (10..20).collect())
            .expect("target slot");
        assert!(scheduler
            .find_decoding_prefix_match_excluding(&(10..20).collect::<Vec<_>>(), 8, target)
            .is_none());
        assert!(scheduler.mark_prefill_complete(source));
        assert_eq!(
            scheduler.find_decoding_prefix_match_excluding(
                &(10..20).collect::<Vec<_>>(),
                8,
                target,
            ),
            Some((source, 10))
        );
    }
    #[test]
    fn bucketed_prefill_selects_homogeneous_bucket() {
        let mut scheduler = Scheduler::new(8);
        for _ in 0..4 {
            scheduler
                .admit(req(4), (0..8u32).collect())
                .expect("admit short");
        }
        scheduler
            .admit(req(4), (0..512u32).collect())
            .expect("admit long");
        let chosen = scheduler.prefill_slots_bucketed(8);
        assert_eq!(chosen.len(), 4, "should pick all 4 short-prompt slots");
        assert!(
            !chosen.contains(&4),
            "long slot must not be in the chosen batch"
        );
    }
    #[test]
    fn bucketed_prefill_tie_break_favours_longer_bucket() {
        let mut scheduler = Scheduler::new(8);
        for _ in 0..2 {
            scheduler
                .admit(req(4), (0..8u32).collect())
                .expect("admit short");
        }
        for _ in 0..2 {
            scheduler
                .admit(req(4), (0..512u32).collect())
                .expect("admit long");
        }
        let chosen = scheduler.prefill_slots_bucketed(8);
        assert_eq!(chosen.len(), 2);
        assert!(
            chosen.iter().all(|&id| id >= 2),
            "tie should choose long bucket"
        );
    }
    #[test]
    fn bucketed_prefill_homogeneous_queue_matches_plain() {
        let mut scheduler = Scheduler::new(4);
        for _ in 0..4 {
            scheduler
                .admit(req(4), (0..32u32).collect())
                .expect("admit");
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
        let shared: Vec<u32> = (100..110).collect();
        let mut a = shared.clone();
        a.push(1);
        let mut b = shared.clone();
        b.push(2);
        let mut c = shared.clone();
        c.push(3);
        let d: Vec<u32> = (900..912).collect();
        let mut scheduler = Scheduler::new(4);
        prefilling(&mut scheduler, &[(0, a), (1, b), (2, c), (3, d)]);
        let chosen = group_by_prefix(&scheduler.slots, 4, 8);
        assert!(chosen.contains(&0) && chosen.contains(&1) && chosen.contains(&2));
        assert!(
            !chosen.contains(&3),
            "unrelated slot must not join, got {chosen:?}"
        );
    }
    #[test]
    fn group_by_prefix_lone_unique_request_still_admits() {
        let mut scheduler = Scheduler::new(4);
        prefilling(&mut scheduler, &[(2, (500..520).collect())]);
        let chosen = group_by_prefix(&scheduler.slots, 4, 8);
        assert_eq!(chosen, vec![2], "lone request must admit, got {chosen:?}");
    }
    #[test]
    fn group_by_prefix_no_shared_prefix_falls_back_fifo() {
        let mut scheduler = Scheduler::new(4);
        prefilling(
            &mut scheduler,
            &[
                (0, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]),
                (1, vec![90, 91, 92, 93, 94, 95, 96, 97, 98]),
            ],
        );
        let chosen = group_by_prefix(&scheduler.slots, 4, 8);
        assert_eq!(
            chosen,
            vec![0, 1],
            "disjoint prompts should FIFO-admit both, got {chosen:?}"
        );
    }
    #[test]
    fn group_by_prefix_deterministic_tie_break_prefers_longer_then_lower_anchor() {
        let px: Vec<u32> = (0..12).collect();
        let mut x0 = px.clone();
        x0.push(70);
        let mut x1 = px.clone();
        x1.push(71);
        let py: Vec<u32> = (200..208).collect();
        let mut y0 = py.clone();
        y0.push(72);
        let mut y1 = py.clone();
        y1.push(73);
        let mut scheduler = Scheduler::new(4);
        prefilling(&mut scheduler, &[(0, x0), (1, x1), (2, y0), (3, y1)]);
        let chosen = group_by_prefix(&scheduler.slots, 2, 8);
        assert_eq!(
            chosen,
            vec![0, 1],
            "longer-shared-prefix group must win, got {chosen:?}"
        );
        assert_eq!(chosen, group_by_prefix(&scheduler.slots, 2, 8));
    }
    #[test]
    fn prefill_slots_prefix_grouped_falls_back_when_policy_off() {
        let mut scheduler = Scheduler::new(4);
        for _ in 0..4 {
            scheduler
                .admit(req(4), (0..32u32).collect())
                .expect("admit");
        }
        assert_eq!(
            scheduler.prefill_slots_prefix_grouped(4),
            scheduler.prefill_slots_bucketed(4),
        );
    }

    #[test]
    fn token_budgeted_prefill_caps_cohort_but_never_starves_a_long_first_request() {
        let mut scheduler = Scheduler::new(3);
        for id in 0..3 {
            let slot = scheduler
                .slots
                .iter_mut()
                .find(|slot| slot.id == id)
                .expect("slot");
            slot.assign(req(4), vec![id + 1; 8]);
        }
        assert_eq!(scheduler.prefill_slots_token_budgeted(3, 16), vec![0, 1]);
        assert_eq!(scheduler.prefill_slots_token_budgeted(3, 0), vec![0]);

        // Already-copied KV does not consume the remaining prefill budget.
        scheduler.slot_mut(0).expect("slot 0").prefix_skip = 8;
        assert_eq!(scheduler.prefill_slots_token_budgeted(3, 8), vec![0, 1]);
    }

    #[test]
    fn chunked_prefill_advances_only_exact_committed_ranges() {
        let mut scheduler = Scheduler::new(2);
        scheduler.admit(req(4), (0..20).collect()).expect("first");
        scheduler
            .admit(req(4), (100..110).collect())
            .expect("second");
        assert_eq!(
            scheduler.prefill_chunks_token_budgeted(2, 8),
            vec![PrefillChunk {
                slot_id: 0,
                start: 0,
                end: 8,
                complete: false
            }]
        );
        assert!(scheduler.commit_prefill_chunk(0, 8));
        assert!(!scheduler.commit_prefill_chunk(0, 7));
        assert_eq!(
            scheduler.prefill_chunks_token_budgeted(2, 8),
            vec![PrefillChunk {
                slot_id: 0,
                start: 8,
                end: 16,
                complete: false
            }]
        );
        assert!(scheduler.commit_prefill_chunk(0, 16));
        assert_eq!(
            scheduler.prefill_chunks_token_budgeted(2, 8),
            vec![PrefillChunk {
                slot_id: 0,
                start: 16,
                end: 20,
                complete: true
            }]
        );
    }

    #[test]
    fn chunked_prefill_charges_only_tail_after_imported_prefix_kv() {
        let mut scheduler = Scheduler::new(1);
        scheduler
            .admit(req(4), (0..12).collect())
            .expect("admission");
        scheduler.slot_mut(0).expect("slot").prefix_skip = 8;
        assert_eq!(
            scheduler.prefill_chunks_token_budgeted(1, 3),
            vec![PrefillChunk {
                slot_id: 0,
                start: 8,
                end: 11,
                complete: false
            }]
        );
        assert!(scheduler.commit_prefill_chunk(0, 11));
        assert_eq!(
            scheduler.prefill_chunks_token_budgeted(1, 3),
            vec![PrefillChunk {
                slot_id: 0,
                start: 11,
                end: 12,
                complete: true
            }]
        );
    }

    #[test]
    fn chunked_prefill_round_robin_prevents_long_first_slot_starvation() {
        let mut scheduler = Scheduler::new(2);
        scheduler.admit(req(4), (0..20).collect()).expect("first");
        scheduler
            .admit(req(4), (100..120).collect())
            .expect("second");

        assert_eq!(
            scheduler.prefill_chunks_token_budgeted(2, 8),
            vec![PrefillChunk {
                slot_id: 0,
                start: 0,
                end: 8,
                complete: false,
            }]
        );
        assert!(scheduler.commit_prefill_chunk(0, 8));
        assert_eq!(
            scheduler.prefill_chunks_token_budgeted(2, 8),
            vec![PrefillChunk {
                slot_id: 1,
                start: 0,
                end: 8,
                complete: false,
            }]
        );
        assert!(scheduler.commit_prefill_chunk(1, 8));
        assert_eq!(
            scheduler.prefill_chunks_token_budgeted(2, 8),
            vec![PrefillChunk {
                slot_id: 0,
                start: 8,
                end: 16,
                complete: false,
            }]
        );
    }

    #[test]
    fn prefix_index_reuses_a_shorter_prefix_of_a_live_longer_prompt() {
        let mut index = PrefixIndex::default();
        index.upsert(7, &[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
        assert_eq!(
            index.find_prefix_match(&[10, 11, 12, 13, 14, 15, 16, 17], 8),
            Some((7, 8)),
            "a request must be able to reuse a live request's shorter exact prefix"
        );
    }

    #[test]
    fn prefix_index_requires_exact_token_identity_after_hash_lookup() {
        let mut index = PrefixIndex::default();
        index.upsert(1, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            index.find_prefix_match(&[1, 2, 3, 4, 5, 6, 7, 99, 9], 8),
            None,
            "a divergent eighth token must never reuse KV merely because a hash lookup found a candidate"
        );
        index.upsert(2, &[1, 2, 3, 4, 5, 6, 7, 8, 10]);
        assert_eq!(
            index.find_prefix_match_excluding(&[1, 2, 3, 4, 5, 6, 7, 8], 8, 2),
            Some((1, 8)),
            "excluding the new slot must still find another exact live source"
        );
    }
}
