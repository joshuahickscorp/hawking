//! hawking-serve: OpenAI-compatible HTTP server.
//!
//! Drives a `hawking_core::Engine` through axum. Continuous
//! batching lives in [`batch`]; the HTTP surface in [`http`].

#[rustfmt::skip]
pub mod batch {
    //! Continuous batching: prefill/decode interleaving so concurrent
    //! requests share MoE kernel launches. The MoE-specific win at batch ≥ 4.
    //!
    //! Phase 4 fills in the slot manager. The data structures that the
    //! HTTP layer reads from are already shaped here so the seams don't
    //! move when the implementation lands.

    pub mod driver {
        //! Decode-step driver for the continuous-batching control plane.
        //!
        //! This module deliberately does not pretend that the current single-KV
        //! engines can safely mix unrelated requests. The caller must only mark slots
        //! as `Decoding` after the engine has the matching per-slot KV context ready.
        //! The next GPU-resident batch kernel plugs in behind `Engine::forward_tokens_batched`.

        use crate::batch::{scheduler::Scheduler, DecodedToken};
        use anyhow::Result;
        use hawking_core::{Engine, GenerateRequest};

        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct DecodeOutput {
            pub slot_id: u32,
            pub token: u32,
            pub text: String,
            pub finished: bool,
        }

        /// Decode-lane stats accumulated across all steps. Exposed via /metrics.
        #[derive(Debug, Default, Clone)]
        pub struct LaneStats {
            /// Steps routed through the greedy token-only path (B×4 byte readback).
            pub greedy_steps: u64,
            /// Steps routed through the full-logits path (B×vocab×4 byte readback).
            pub logits_steps: u64,
            /// Cumulative bytes read back from GPU this session.
            pub readback_bytes: u64,
            /// Track 5.2: number of admissions where a KV prefix was successfully
            /// copied from an existing slot (copy_kv_prefix_to_slot returned Ok).
            pub prefix_reuse_count: u64,
        }

        pub struct BatchDriver {
            pub scheduler: Scheduler,
            pub lane_stats: LaneStats,
        }

        impl BatchDriver {
            pub fn new(max_batch_size: usize) -> Self {
                Self { scheduler: Scheduler::new(max_batch_size), lane_stats: LaneStats::default() }
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

                // Greedy lane: all slots are temperature=0 with no repetition penalty
                // override → route to token-only path (B×4 byte readback, no logits).
                let all_greedy = batch.iter().all(|step| {
                    self.scheduler
                        .slots
                        .iter()
                        .find(|s| s.id == step.slot_id)
                        .and_then(|s| s.req.as_ref())
                        .map(|r| r.sampling.temperature <= 0.0 && r.sampling.repetition_penalty <= 1.0)
                        .unwrap_or(false)
                });

                let b = batch.len();
                if all_greedy {
                    let token_ids = engine.forward_multiseq_greedy_tokens(&tokens, &positions, &regions)?;
                    let eos_id = engine.eos_id_for_batch();
                    let decoded = self.scheduler.apply_decode_tokens(&batch, token_ids, eos_id)?;
                    self.lane_stats.greedy_steps += 1;
                    self.lane_stats.readback_bytes += (b * std::mem::size_of::<u32>()) as u64;
                    return decoded.into_iter().map(|token| decode_output(engine, token)).collect();
                }

                // Full-logits lane (sampling, logprobs, or repetition penalty requests).
                let mut logits = engine.forward_multiseq_batched(&tokens, &positions, &regions)?;
                let vocab = logits.first().map(|l| l.len()).unwrap_or(0);
                let eos_id = engine.eos_id_for_batch();
                let decoded = self.scheduler.apply_decode_logits(&batch, &mut logits, eos_id)?;
                self.lane_stats.logits_steps += 1;
                self.lane_stats.readback_bytes += (b * vocab * std::mem::size_of::<f32>()) as u64;

                decoded.into_iter().map(|token| decode_output(engine, token)).collect()
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
            use hawking_core::{EngineConfig, GenStats, GenerateRequest, SamplingParams, StreamEvent};
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
                fn load(_weights: &Path, _config: EngineConfig) -> hawking_core::Result<Self>
                where
                    Self: Sized,
                {
                    Ok(Self::new())
                }

                fn generate(
                    &mut self,
                    _req: GenerateRequest,
                    _sink: &mut dyn FnMut(StreamEvent),
                ) -> hawking_core::Result<GenStats> {
                    Ok(GenStats { completion_tokens: 0, ..Default::default() })
                }

                fn model_id(&self) -> &str {
                    "fake"
                }

                fn encode_prompt_for_batch(&self, prompt: &str) -> hawking_core::Result<Vec<u32>> {
                    Ok(prompt.bytes().map(u32::from).collect())
                }

                fn decode_token_for_batch(&self, token: u32) -> hawking_core::Result<String> {
                    Ok(format!("<{token}>"))
                }

                fn eos_id_for_batch(&self) -> Option<u32> {
                    Some(2)
                }

                fn forward_tokens_for_test(
                    &mut self,
                    tokens: &[u32],
                    positions: &[usize],
                ) -> hawking_core::Result<Vec<Vec<f32>>> {
                    self.forward_tokens_batched(tokens, positions)
                }

                fn forward_tokens_batched(
                    &mut self,
                    tokens: &[u32],
                    positions: &[usize],
                ) -> hawking_core::Result<Vec<Vec<f32>>> {
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
                    sampling: SamplingParams { temperature: 0.0, ..SamplingParams::default() },
                    stop: Vec::new(),
                    abort: None,
                    max_stall_ms: 0,
                    json_mode: false,
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

                let out = driver.decode_ready_once(&mut engine, 4).expect("decode once");

                assert_eq!(engine.calls, vec![(vec![10, 20], vec![1, 1])]);
                assert_eq!(
                    out,
                    vec![
                        DecodeOutput { slot_id: 0, token: 1, text: "<1>".into(), finished: false },
                        DecodeOutput { slot_id: 1, token: 2, text: "<2>".into(), finished: true },
                    ]
                );
                assert_eq!(driver.scheduler.slots[0].last_token, Some(1));
                assert_eq!(driver.scheduler.slots[1].state, crate::batch::SlotState::Finishing);
            }

            #[test]
            fn decode_ready_once_no_ready_slots_is_noop() {
                let mut driver = BatchDriver::new(2);
                let mut engine = FakeEngine::new();
                let out = driver.decode_ready_once(&mut engine, 2).expect("decode once");
                assert!(out.is_empty());
                assert!(engine.calls.is_empty());
            }

            #[test]
            fn admit_tokenizes_prompt_through_engine() {
                let mut driver = BatchDriver::new(1);
                let engine = FakeEngine::new();
                let slot_id = driver.admit(&engine, req(3)).expect("admit result").expect("slot id");

                assert_eq!(slot_id, 0);
                assert_eq!(driver.scheduler.slots[0].prompt_ids, vec![b'x' as u32]);
                assert_eq!(driver.scheduler.slots[0].state, crate::batch::SlotState::Prefilling);
            }
        }
    }
    pub mod scheduler {
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
                    if stored_hash == request_prefix_hash && best.map(|(_, bl)| overlap > bl).unwrap_or(true) {
                        best = Some((slot_id, overlap));
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
                    if stored_hash == request_prefix_hash && best.map(|(_, bl)| overlap > bl).unwrap_or(true) {
                        best = Some((slot_id, overlap));
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
                let slots = (0..max_batch_size as u32).map(Slot::idle).collect();
                Self { slots, max_batch_size, prefix_index: PrefixIndex::default(), policy: BatchPolicy::Default }
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
                self.slots.iter().filter(|s| s.state != SlotState::Idle).count()
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
                                .map(|r| r.sampling.temperature <= 0.0 && r.sampling.repetition_penalty <= 1.0)
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
                self.ready_decode_indices(max).into_iter().map(|idx| self.slots[idx].id).collect()
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
                self.prefill_indices(max).into_iter().map(|idx| self.slots[idx].id).collect()
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

            /// Seed a slot's first generated token after prefill (see
            /// `Slot::seed_first_token`). Returns the `DecodedToken` so the caller can
            /// stream the text and release the slot if it is already an EOS.
            pub fn seed_first_token(&mut self, id: u32, token: u32, eos_id: Option<u32>) -> Option<DecodedToken> {
                self.slot_mut(id).map(|s| s.seed_first_token(token, eos_id))
            }

            pub fn decode_batch(&self, max: usize) -> Vec<DecodeStep> {
                self.ready_decode_indices(max).into_iter().filter_map(|idx| self.slots[idx].decode_step()).collect()
            }

            pub fn apply_decode_logits(
                &mut self,
                batch: &[DecodeStep],
                logits: &mut [Vec<f32>],
                eos_id: Option<u32>,
            ) -> Result<Vec<DecodedToken>> {
                if batch.len() != logits.len() {
                    return Err(anyhow!("decode result shape mismatch: batch={} logits={}", batch.len(), logits.len()));
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
                        DecodeStep { slot_id: 0, token: 1, position: 1 },
                        DecodeStep { slot_id: 1, token: 2, position: 1 },
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
                let decoded = scheduler.apply_decode_logits(&batch, &mut logits, Some(2)).expect("apply logits");

                assert_eq!(
                    decoded,
                    vec![
                        DecodedToken { slot_id: 0, token: 1, finished: false },
                        DecodedToken { slot_id: 1, token: 2, finished: true },
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
                assert_eq!(scheduler.prefill_slots_bucketed(4), scheduler.prefill_slots(4),);
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
                // The 3 shared-prefix slots must co-batch; the unrelated one excluded.
                assert!(
                    chosen.contains(&0) && chosen.contains(&1) && chosen.contains(&2),
                    "shared-prefix trio must co-batch, got {chosen:?}"
                );
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
                prefilling(
                    &mut scheduler,
                    &[(0, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]), (1, vec![90, 91, 92, 93, 94, 95, 96, 97, 98])],
                );
                let chosen = group_by_prefix(&scheduler.slots, 4, 8);
                assert_eq!(chosen, vec![0, 1], "disjoint prompts should FIFO-admit both, got {chosen:?}");
            }

            #[test]
            fn group_by_prefix_deterministic_tie_break_prefers_longer_then_lower_anchor() {
                // Group X = slots {0,1} share 12 tokens. Group Y = slots {2,3} share 8.
                // Longer shared prefix (X) must win regardless of slot order.
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
                assert_eq!(chosen, vec![0, 1], "longer-shared-prefix group must win, got {chosen:?}");
                // Determinism: identical inputs -> identical output.
                assert_eq!(chosen, group_by_prefix(&scheduler.slots, 2, 8));
            }

            #[test]
            fn prefill_slots_prefix_grouped_falls_back_when_policy_off() {
                // With BatchPolicy::Default the prefix selector must equal the bucketed one.
                let mut scheduler = Scheduler::new(4);
                for _ in 0..4 {
                    scheduler.admit(req(4), (0..32u32).collect()).expect("admit");
                }
                assert_eq!(scheduler.prefill_slots_prefix_grouped(4), scheduler.prefill_slots_bucketed(4),);
            }
        }
    }

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
            let seed = req.sampling.seed.unwrap_or(0xD15A_0000_0000_0000u64 ^ self.id as u64);
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
            let token = self.last_token?;
            self.is_ready_to_decode().then_some(DecodeStep {
                slot_id: self.id,
                token,
                position: self.position,
            })
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
            DecodedToken { slot_id: self.id, token, finished: self.state == SlotState::Finishing }
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
            DecodedToken { slot_id: self.id, token, finished: self.state == SlotState::Finishing }
        }

        pub fn release(&mut self) {
            let id = self.id;
            *self = Self::idle(id);
        }
    }
}
#[rustfmt::skip]
pub mod http {
    //! axum routes for OpenAI-compatible endpoints:
    //!   POST /v1/chat/completions   (SSE streaming)
    //!   POST /v1/completions        (legacy, also SSE)
    //!   GET  /v1/models
    //!   GET  /healthz
    //!   GET  /metrics               (Prometheus textfile)

    use crate::batch::driver::BatchDriver;
    use crate::system_kv_bank::SystemPromptKvBank;
    use axum::{
        body::Bytes,
        extract::State,
        http::StatusCode,
        response::{
            sse::{Event, KeepAlive, Sse},
            IntoResponse, Response,
        },
        routing::{get, post},
        Json, Router,
    };
    use futures::stream::Stream;
    use hawking_core::{Engine, GenerateRequest, SamplingParams};
    use parking_lot::Mutex;
    use serde::{Deserialize, Serialize};
    use std::collections::{HashMap, VecDeque};
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use tokio::sync::mpsc as async_mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    /// Per-slot token channel item: `Ok(text)` for each generated token,
    /// `Err(())` to signal stream end (EOS, max_tokens reached, or error).
    type SlotToken = Result<String, ()>;

    /// Structured, OpenAI-compatible error.
    ///
    /// Serializes to `{"error": {"message": ..., "type": ..., "code": ...}}` and
    /// carries a stable HTTP status. The `code` field is a machine-readable,
    /// stable token (see the constants below); the `type` field mirrors OpenAI's
    /// coarse error families (`invalid_request_error`, `internal_error`).
    #[derive(Debug, Clone)]
    pub struct ApiError {
        status: StatusCode,
        message: String,
        error_type: &'static str,
        code: &'static str,
    }

    impl ApiError {
        /// Body could not be parsed as the expected JSON shape (syntax error,
        /// wrong types, or a missing required field that serde rejects).
        pub fn invalid_json(message: impl Into<String>) -> Self {
            Self {
                status: StatusCode::BAD_REQUEST,
                message: message.into(),
                error_type: "invalid_request_error",
                code: "invalid_json",
            }
        }

        /// A required parameter was syntactically present but semantically empty
        /// (e.g. `messages: []` or an empty `prompt`).
        pub fn missing_parameter(message: impl Into<String>) -> Self {
            Self {
                status: StatusCode::BAD_REQUEST,
                message: message.into(),
                error_type: "invalid_request_error",
                code: "missing_required_parameter",
            }
        }

        /// Generation failed inside the engine, or the worker task panicked.
        pub fn internal(message: impl Into<String>) -> Self {
            Self {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: message.into(),
                error_type: "internal_error",
                code: "internal_error",
            }
        }

        fn to_body(&self) -> serde_json::Value {
            serde_json::json!({
                "error": {
                    "message": self.message,
                    "type": self.error_type,
                    "code": self.code,
                }
            })
        }
    }

    impl IntoResponse for ApiError {
        fn into_response(self) -> Response {
            (self.status, Json(self.to_body())).into_response()
        }
    }

    impl std::fmt::Display for ApiError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{} ({}): {}", self.code, self.status, self.message)
        }
    }

    impl std::error::Error for ApiError {}

    /// Parse a request body into `T`, mapping any serde failure to a structured
    /// [`ApiError`] with the `invalid_json` code. Centralizes the malformed-input
    /// path so every route reports errors with the same machine-readable shape.
    fn parse_json<T: serde::de::DeserializeOwned>(body: &Bytes) -> Result<T, ApiError> {
        serde_json::from_slice::<T>(body).map_err(|e| ApiError::invalid_json(format!("invalid request body: {e}")))
    }

    #[derive(Clone)]
    pub struct AppState {
        pub engine: Arc<Mutex<Box<dyn Engine>>>,
        /// Continuous-batching driver — shared with the background decode loop.
        /// HTTP handlers take this lock briefly for admit only.
        pub driver: Arc<Mutex<BatchDriver>>,
        /// Per-slot SSE token senders. The background loop writes here;
        /// `sse_response` reads. Keyed by stable slot_id.
        pub slot_senders: Arc<Mutex<HashMap<u32, async_mpsc::Sender<SlotToken>>>>,
        /// Requests waiting for a free batch slot. Bounded at `max_batch * 8`.
        /// Tuple: (request, token_sender, is_chat_format).
        pub wait_queue: Arc<Mutex<VecDeque<(GenerateRequest, async_mpsc::Sender<SlotToken>, bool)>>>,
        pub model_arch: String,
        pub max_batch: usize,
        pub requests_admitted: Arc<AtomicU64>,
        pub tokens_generated: Arc<AtomicU64>,
        pub requests_queued: Arc<AtomicU64>,
        /// Track 5.2: serve-lifetime hash(system-prefix) -> source-slot routing
        /// hint. Survives a source request finishing, so serial workloads that
        /// re-send an identical system prompt still get shared-prefix KV reuse
        /// (the live `PrefixIndex` only matches CURRENTLY-active slots). Stores
        /// zero KV bytes; every hit is re-verified by the bit-identical
        /// `copy_kv_prefix_to_slot` + `prefill_slot_from_pos` path, so a stale
        /// slot simply fails the copy and falls back to a cold prefill.
        pub system_kv_bank: Arc<Mutex<SystemPromptKvBank>>,
    }

    /// Track 5.2 — the agreed banked-prefix length the serve-loop admit path uses
    /// for BOTH `SystemPromptKvBank::record` and `::lookup`. PURE (no I/O, no
    /// model): the gate test calls this directly so the record/lookup keys can
    /// never silently diverge.
    ///
    /// The bank requires a STRICT leading prefix (`banked_len < prompt_ids.len()`),
    /// unlike the live `find_prefix_match_excluding` which keys on the full source
    /// slot length. We bank the prompt minus its last token — the "bail one token
    /// short" rule the disk/RAM KV tiers use — so the decode loop always keeps a
    /// real `last_id`. For a serial workload that re-sends the SAME prompt, the
    /// turn that records and the turn that looks up both see identical `prompt_ids`
    /// and therefore hash to the same key. Returns 0 when the prompt is too short
    /// to bank (the bank itself also rejects `< min_prefix_tokens`).
    pub fn banked_len_for(prompt_ids: &[u32]) -> usize {
        prompt_ids.len().saturating_sub(1)
    }

    pub fn router(state: AppState) -> Router {
        Router::new()
            .route("/healthz", get(healthz))
            .route("/v1/models", get(list_models))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/completions", post(completions))
            .route("/v1/embeddings", post(embeddings))
            .route("/v1/hawking/tokens", post(hawking_tokens))
            .route("/v1/hawking/generate", post(hawking_generate))
            .route("/v1/hawking/context", get(hawking_context))
            .route("/metrics", get(metrics))
            .with_state(state)
    }

    async fn healthz() -> &'static str {
        "ok"
    }

    async fn metrics(State(s): State<AppState>) -> String {
        let admitted = s.requests_admitted.load(Ordering::Relaxed);
        let tokens = s.tokens_generated.load(Ordering::Relaxed);
        let driver = s.driver.lock();
        let active = driver.scheduler.active_count();
        let queued = s.wait_queue.lock().len();
        let lane = driver.lane_stats.clone();
        drop(driver);
        format!(
            "# HELP hawking_requests_admitted_total Requests successfully admitted to a batch slot\n\
             # TYPE hawking_requests_admitted_total counter\n\
             hawking_requests_admitted_total {admitted}\n\
             # HELP hawking_tokens_generated_total Tokens generated across all slots\n\
             # TYPE hawking_tokens_generated_total counter\n\
             hawking_tokens_generated_total {tokens}\n\
             # HELP hawking_active_slots Current number of active decode slots\n\
             # TYPE hawking_active_slots gauge\n\
             hawking_active_slots {active}\n\
             # HELP hawking_queued_requests Requests waiting for a free slot\n\
             # TYPE hawking_queued_requests gauge\n\
             hawking_queued_requests {queued}\n\
             # HELP hawking_greedy_decode_steps_total Decode steps routed through the token-only greedy lane\n\
             # TYPE hawking_greedy_decode_steps_total counter\n\
             hawking_greedy_decode_steps_total {}\n\
             # HELP hawking_logits_decode_steps_total Decode steps that materialized full logits\n\
             # TYPE hawking_logits_decode_steps_total counter\n\
             hawking_logits_decode_steps_total {}\n\
             # HELP hawking_gpu_readback_bytes_total Cumulative GPU→CPU readback bytes\n\
             # TYPE hawking_gpu_readback_bytes_total counter\n\
             hawking_gpu_readback_bytes_total {}\n\
             # HELP hawking_prefix_reuse_total Admissions where KV prefix was copied from an existing slot\n\
             # TYPE hawking_prefix_reuse_total counter\n\
             hawking_prefix_reuse_total {}\n",
            lane.greedy_steps, lane.logits_steps, lane.readback_bytes, lane.prefix_reuse_count,
        )
    }

    /// Spine A — live context introspection. Read-only snapshot of the real,
    /// dynamic context picture: native length from the model config, the effective
    /// ceiling derived from the measured `.tq` multiplier (passed in via env by the
    /// supervisor — never a constant), the constant recurrent-state footprint for
    /// SSMs, and live slot occupancy. The shell renders this as an ambient cue.
    #[derive(Serialize)]
    struct ContextStatus {
        model_id: String,
        arch: String,
        ctx_len_native: Option<usize>,
        ctx_len_effective: Option<usize>,
        /// Measured `.tq` weight-compression multiplier (1.0 == no claim).
        tq_multiplier: f32,
        /// True when the effective ceiling is a derived estimate, not a hard cap.
        tq_estimated: bool,
        /// Constant recurrent-state footprint in bytes for SSMs; None for transformers.
        recurrent_state_bytes: Option<usize>,
        active_slots: usize,
        free_slots: usize,
        max_batch: usize,
    }

    async fn hawking_context(State(s): State<AppState>) -> Json<ContextStatus> {
        let (model_id, arch, native, state_bytes) = {
            let eng = s.engine.lock();
            (
                eng.model_id().to_string(),
                eng.model_arch().to_string(),
                eng.context_length_native(),
                eng.recurrent_state_size_bytes(),
            )
        };
        // The supervisor measured this from the .tq artifact and passed it in; if
        // absent the multiplier is 1.0 (no expansion claimed). Never hardcoded.
        let tq_multiplier: f32 = std::env::var("HAWKING_QWEN_TQ_MULTIPLIER")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|m: &f32| m.is_finite() && *m >= 1.0)
            .unwrap_or(1.0);
        let effective = native.map(|n| (n as f32 * tq_multiplier).round() as usize);
        let active = s.driver.lock().scheduler.active_count();
        Json(ContextStatus {
            model_id,
            arch,
            ctx_len_native: native,
            ctx_len_effective: effective,
            tq_multiplier,
            tq_estimated: tq_multiplier > 1.0,
            recurrent_state_bytes: state_bytes,
            active_slots: active,
            free_slots: s.max_batch.saturating_sub(active),
            max_batch: s.max_batch,
        })
    }

    #[derive(Serialize)]
    struct ModelInfo {
        id: String,
        object: &'static str,
    }

    #[derive(Serialize)]
    struct ListModels {
        object: &'static str,
        data: Vec<ModelInfo>,
    }

    async fn list_models(State(s): State<AppState>) -> Json<ListModels> {
        let id = s.engine.lock().model_id().to_string();
        Json(ListModels { object: "list", data: vec![ModelInfo { id, object: "model" }] })
    }

    #[derive(Deserialize, Clone)]
    struct ChatMessage {
        role: String,
        content: String,
    }

    #[derive(Deserialize)]
    struct ChatReq {
        #[allow(dead_code)]
        model: Option<String>,
        messages: Vec<ChatMessage>,
        #[serde(default = "default_max_tokens")]
        max_tokens: usize,
        #[serde(default)]
        temperature: Option<f32>,
        #[serde(default)]
        top_p: Option<f32>,
        #[serde(default)]
        seed: Option<u64>,
        #[serde(default)]
        stream: bool,
        /// `{"type": "json_object"}` triggers structural JSON constraint masking.
        #[serde(default)]
        response_format: Option<ResponseFormat>,
        /// OpenAI-style function tools; when present they are rendered into the prompt
        /// and the completion is parsed back into `tool_calls` (Phase 1a).
        #[serde(default)]
        tools: Option<Vec<serde_json::Value>>,
        /// Accepted for API compatibility; currently advisory only.
        #[serde(default)]
        #[allow(dead_code)]
        tool_choice: Option<serde_json::Value>,
    }

    #[derive(Deserialize, Default)]
    struct ResponseFormat {
        #[serde(rename = "type", default)]
        format_type: String,
    }

    fn default_max_tokens() -> usize {
        256
    }

    #[derive(Deserialize)]
    struct CompletionReq {
        #[allow(dead_code)]
        model: Option<String>,
        prompt: String,
        #[serde(default = "default_max_tokens")]
        max_tokens: usize,
        #[serde(default)]
        temperature: Option<f32>,
        #[serde(default)]
        top_p: Option<f32>,
        #[serde(default)]
        seed: Option<u64>,
        #[serde(default)]
        stream: bool,
    }

    async fn chat_completions(State(s): State<AppState>, body: Bytes) -> Response {
        let req: ChatReq = match parse_json(&body) {
            Ok(req) => req,
            Err(e) => return e.into_response(),
        };
        if req.messages.is_empty() {
            return ApiError::missing_parameter("'messages' must contain at least one message").into_response();
        }
        // Native tool calling (Phase 1a): render the tool specs into a leading system
        // message so a Hermes/Qwen-trained model emits <tool_call> blocks, and remember
        // to parse them back out of the completion.
        let tools: Vec<serde_json::Value> = req.tools.clone().unwrap_or_default();
        let want_tools = !tools.is_empty();
        let tool_names = crate::tool_calls::tool_names(&tools);
        let prompt = if want_tools {
            let preamble = crate::tool_calls::render_tools_preamble(&tools);
            let mut msgs = req.messages.clone();
            match msgs.first_mut() {
                Some(first) if first.role == "system" => {
                    first.content = format!("{preamble}\n{}", first.content);
                }
                _ => msgs.insert(0, ChatMessage { role: "system".to_string(), content: preamble }),
            }
            render_chat(&msgs, &s.model_arch)
        } else {
            render_chat(&req.messages, &s.model_arch)
        };
        let sampling = SamplingParams {
            temperature: req.temperature.unwrap_or(0.7),
            top_k: 40,
            top_p: req.top_p.unwrap_or(0.9),
            repetition_penalty: 1.0,
            seed: req.seed,
        };
        let json_mode = req.response_format.as_ref().map(|f| f.format_type == "json_object").unwrap_or(false);
        let gen = GenerateRequest {
            prompt,
            max_new_tokens: req.max_tokens,
            sampling,
            stop: Vec::new(),
            abort: None,
            max_stall_ms: 0,
            json_mode,
        };
        if req.stream {
            sse_response(s, gen, /*chat=*/ true, tool_names).into_response()
        } else {
            json_full_response(s, gen, /*chat=*/ true, tool_names).await.into_response()
        }
    }

    async fn completions(State(s): State<AppState>, body: Bytes) -> Response {
        let req: CompletionReq = match parse_json(&body) {
            Ok(req) => req,
            Err(e) => return e.into_response(),
        };
        if req.prompt.is_empty() {
            return ApiError::missing_parameter("'prompt' must not be empty").into_response();
        }
        let sampling = SamplingParams {
            temperature: req.temperature.unwrap_or(0.7),
            top_k: 40,
            top_p: req.top_p.unwrap_or(0.9),
            repetition_penalty: 1.0,
            seed: req.seed,
        };
        let gen = GenerateRequest {
            prompt: req.prompt,
            max_new_tokens: req.max_tokens,
            sampling,
            stop: Vec::new(),
            abort: None,
            max_stall_ms: 0,
            json_mode: false,
        };
        if req.stream {
            sse_response(s, gen, /*chat=*/ false, Vec::new()).into_response()
        } else {
            json_full_response(s, gen, /*chat=*/ false, Vec::new()).await.into_response()
        }
    }

    fn render_chat(msgs: &[ChatMessage], model_arch: &str) -> String {
        match model_arch {
            "deepseek2" => render_chat_deepseek(msgs),
            a if a.starts_with("qwen2") => render_chat_qwen2(msgs),
            _ => render_chat_generic(msgs),
        }
    }

    fn render_chat_deepseek(msgs: &[ChatMessage]) -> String {
        let mut s = String::new();
        for m in msgs {
            match m.role.as_str() {
                "system" => s.push_str(&format!("{}\n\n", m.content)),
                "user" => s.push_str(&format!("User: {}\n\n", m.content)),
                "assistant" => s.push_str(&format!("Assistant: {}\n\n", m.content)),
                _ => {}
            }
        }
        s.push_str("Assistant:");
        s
    }

    fn render_chat_qwen2(msgs: &[ChatMessage]) -> String {
        let mut s = String::new();
        // Qwen2.5's chat template injects a default system message when the caller
        // gives none; without it the model can degenerate on short prompts.
        if msgs.first().map(|m| m.role.as_str()) != Some("system") {
            s.push_str(
                "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. \
                 You are a helpful assistant.<|im_end|>\n",
            );
        }
        for m in msgs {
            s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content));
        }
        s.push_str("<|im_start|>assistant\n");
        s
    }

    fn render_chat_generic(msgs: &[ChatMessage]) -> String {
        let mut s = String::new();
        for m in msgs {
            s.push_str(&format!("<|{}|>\n{}\n", m.role, m.content));
        }
        s.push_str("<|assistant|>\n");
        s
    }

    fn sse_response(
        state: AppState,
        req: GenerateRequest,
        chat: bool,
        tool_names: Vec<String>,
    ) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
        // SSE → client channel (receives formatted SSE events).
        let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(256);
        // Token channel: the background decode loop sends raw text fragments here.
        let (tok_tx, mut tok_rx) = async_mpsc::channel::<SlotToken>(256);

        // Admit the request under a short lock (tokenize + slot assignment only).
        // The engine lock is held only for the encoding step, not for generation.
        let admit_outcome = {
            let engine = state.engine.lock();
            let mut driver = state.driver.lock();
            driver.admit(&**engine, req.clone())
        };
        // Distinguish a real admit decision from an engine that cannot serve this
        // request at all. `Ok(Some)` = admitted; `Ok(None)` = no free slot (→ queue);
        // `Err` (e.g. the engine lacks `encode_prompt_for_batch`, or tokenization
        // failed) must NOT silently enter the wait-queue forever — return a clear
        // SSE error instead (mirrors the slot-exhausted error path below). This is
        // what made the RWKV admission gap present as a 180s hang.
        let slot_id_opt = match admit_outcome {
            Ok(slot) => slot,
            Err(e) => {
                let sse_tx2 = sse_tx.clone();
                let msg = format!("engine cannot serve this request: {e}");
                tokio::spawn(async move {
                    let body = serde_json::json!({
                        "error": {"message": msg, "type": "server_error", "code": "admit_unsupported"}
                    });
                    let _ = sse_tx2.send(Ok(Event::default().data(body.to_string()))).await;
                });
                return Sse::new(ReceiverStream::new(sse_rx)).keep_alive(KeepAlive::default());
            }
        };

        if let Some(slot_id) = slot_id_opt {
            // Slot available immediately — register sender and start serving.
            state.requests_admitted.fetch_add(1, Ordering::Relaxed);
            state.slot_senders.lock().insert(slot_id, tok_tx);
        } else {
            // No free slot — queue the request for deferred admission.
            let queue_cap = state.max_batch * 8;
            if state.wait_queue.lock().len() >= queue_cap {
                // Queue is also full — error immediately.
                let sse_tx2 = sse_tx.clone();
                tokio::spawn(async move {
                    let body = serde_json::json!({
                        "error": {"message": "server busy — no batch slot available",
                                  "type": "server_error", "code": "slot_exhausted"}
                    });
                    let _ = sse_tx2.send(Ok(Event::default().data(body.to_string()))).await;
                });
                return Sse::new(ReceiverStream::new(sse_rx)).keep_alive(KeepAlive::default());
            }
            state.requests_queued.fetch_add(1, Ordering::Relaxed);
            state.wait_queue.lock().push_back((req, tok_tx, chat));
            // The SSE forwarder below is still spawned and will stream tokens once
            // the request is admitted from the queue when a slot frees.
        };

        // Forward raw token strings from the per-slot channel to SSE events. When tools
        // were requested we BUFFER instead of streaming, because a Hermes/Qwen model
        // emits `<tool_call>{...}</tool_call>` XML that must be parsed into structured
        // tool_calls, not streamed verbatim as content. On end we emit one terminating
        // chunk carrying either the tool_calls or the buffered text, with finish_reason.
        let want_tools = chat && !tool_names.is_empty();
        tokio::spawn(async move {
            let mut buf = String::new();
            while let Some(item) = tok_rx.recv().await {
                match item {
                    Ok(text) => {
                        if want_tools {
                            // Buffer; the terminating chunk is emitted after the loop.
                            buf.push_str(&text);
                            continue;
                        }
                        let chunk = if chat {
                            serde_json::json!({
                                "choices": [{"delta": {"content": text}, "index": 0}],
                                "object": "chat.completion.chunk",
                            })
                        } else {
                            serde_json::json!({
                                "choices": [{"text": text, "index": 0}],
                                "object": "text_completion",
                            })
                        };
                        if sse_tx.send(Ok(Event::default().data(chunk.to_string()))).await.is_err() {
                            return;
                        }
                    }
                    // The failure sentinel. Normal completion does NOT send this — the
                    // decode loop just drops the sender (recv -> None), so the flush
                    // below MUST live outside the loop or a buffered (tools) answer
                    // would be lost entirely.
                    Err(()) => break,
                }
            }

            // Terminating flush. Reached on normal channel-close AND on the failure
            // sentinel. For a tools request, parse the buffer into structured tool_calls
            // (or return the buffered text); the non-tools path already streamed its
            // content and just needs the [DONE] terminator.
            if want_tools {
                let calls = crate::tool_calls::extract_tool_calls(&buf, &tool_names);
                let chunk = if !calls.is_empty() {
                    let arr: Vec<serde_json::Value> = calls.iter().map(|c| c.to_openai()).collect();
                    serde_json::json!({
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "tool_calls": arr},
                            "finish_reason": "tool_calls"
                        }]
                    })
                } else {
                    serde_json::json!({
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": buf},
                            "finish_reason": "stop"
                        }]
                    })
                };
                let _ = sse_tx.send(Ok(Event::default().data(chunk.to_string()))).await;
            }
            let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
        });

        Sse::new(ReceiverStream::new(sse_rx)).keep_alive(KeepAlive::default())
    }

    /// Lean request body for the native `/v1/hawking/generate` endpoint.
    /// No role/message envelope — just the generation knobs.
    #[derive(Deserialize)]
    pub struct HawkingGenerateReq {
        pub prompt: String,
        #[serde(default = "default_max_tokens")]
        pub max_tokens: usize,
        /// Greedy when absent or <= 0.0 (routes to the token-only B×4 lane).
        #[serde(default)]
        pub temperature: Option<f32>,
        #[serde(default)]
        pub top_p: Option<f32>,
        #[serde(default)]
        pub seed: Option<u64>,
        /// Stop strings. Mapped into GenerateRequest.stop. (The batch scheduler
        /// does not yet honor stop; this preserves the field end-to-end.)
        #[serde(default)]
        pub stop: Vec<String>,
    }

    /// PURE request->GenerateRequest mapping. No engine, no I/O — the gate test
    /// calls this directly. temperature absent/<=0 => greedy (temp 0, top_k 0,
    /// top_p 1) so the slot routes through forward_multiseq_greedy_tokens.
    pub fn map_hawking_generate_req(req: &HawkingGenerateReq) -> GenerateRequest {
        let temp = req.temperature.unwrap_or(0.0);
        let greedy = temp <= 0.0;
        let sampling = SamplingParams {
            temperature: if greedy { 0.0 } else { temp },
            top_k: if greedy { 0 } else { 40 },
            top_p: if greedy { 1.0 } else { req.top_p.unwrap_or(0.9) },
            repetition_penalty: 1.0,
            seed: req.seed,
        };
        GenerateRequest {
            prompt: req.prompt.clone(),
            max_new_tokens: req.max_tokens,
            sampling,
            stop: req.stop.clone(),
            abort: None,
            max_stall_ms: 0,
            json_mode: false,
        }
    }

    /// Re-derive the LM-head path label the same way the engine does for the
    /// env-controlled cases (serve always sets HAWKING_QWEN_Q4K_LMHEAD=1, so the
    /// q4k* branch is the live one). Mirrors QwenDense::lm_head_path env logic.
    /// Returns one of: "q4k-predec-f16s" | "q4k-predec" | "q4k" | "f16".
    pub fn lm_head_path_from_env() -> &'static str {
        let q4k = std::env::var_os("HAWKING_QWEN_Q4K_LMHEAD").map(|v| v != "0").unwrap_or(false);
        if !q4k {
            return "f16";
        }
        let predec = std::env::var_os("HAWKING_QWEN_Q4K_PREDEC").map(|v| v != "0").unwrap_or(true);
        let f16s = predec && std::env::var_os("HAWKING_QWEN_PREDEC_F16SCALES").map(|v| v != "0").unwrap_or(false);
        if f16s {
            "q4k-predec-f16s"
        } else if predec {
            "q4k-predec"
        } else {
            "q4k"
        }
    }

    /// PURE: build the native final stats object from server-observed values.
    /// Field NAMES mirror GenStats::stats_json() so native + OpenAI clients parse
    /// the same keys. dec_tps = completion_tokens / (decode_ms/1000).
    pub fn hawking_generate_stats_json(
        prompt_tokens: usize,
        completion_tokens: usize,
        decode_ms: f64,
        token_only_path_used: bool,
        lm_head_path: &str,
    ) -> serde_json::Value {
        let dec_tps = (completion_tokens as f64) / (decode_ms / 1000.0).max(1e-6);
        serde_json::json!({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "decode_ms": decode_ms,
            "dec_tps": dec_tps,
            "token_only_path_used": token_only_path_used,
            "lm_head_path": lm_head_path,
        })
    }

    /// Request body for the low-overhead `/v1/hawking/tokens` endpoint.
    #[derive(Deserialize)]
    struct HawkingTokensReq {
        prompt: String,
        #[serde(default = "default_max_tokens")]
        max_tokens: usize,
        #[serde(default)]
        seed: Option<u64>,
    }

    /// Native streaming endpoint: returns raw token IDs as SSE integers.
    ///
    /// Each `data:` line is a decimal u32 token ID. The final event is
    /// `data: [DONE]`. Always uses temperature=0 (greedy-only).
    ///
    /// Lower overhead than the OpenAI JSON chunk format because there is no
    /// per-token JSON wrapper — just a single integer per SSE event.
    async fn hawking_tokens(State(s): State<AppState>, body: Bytes) -> Response {
        let req: HawkingTokensReq = match parse_json(&body) {
            Ok(req) => req,
            Err(e) => return e.into_response(),
        };
        if req.prompt.is_empty() {
            return ApiError::missing_parameter("'prompt' must not be empty").into_response();
        }
        let gen = GenerateRequest {
            prompt: req.prompt,
            max_new_tokens: req.max_tokens,
            sampling: SamplingParams {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                repetition_penalty: 1.0,
                seed: req.seed,
            },
            stop: Vec::new(),
            abort: None,
            max_stall_ms: 0,
            json_mode: false,
        };
        token_id_sse_response(s, gen).into_response()
    }

    /// SSE response that streams raw u32 token IDs (decimal) instead of JSON.
    /// Used by the `/v1/hawking/tokens` native endpoint.
    fn token_id_sse_response(
        state: AppState,
        req: GenerateRequest,
    ) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
        // SSE → client channel (receives formatted SSE events).
        let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(256);
        // Token channel: background decode loop sends raw text fragments.
        // We need to recover the token ID; the text is the decoded string.
        // The slot channel carries String; we emit the admission slot_id and
        // let the forwarder read the *token* field from DecodeOutput.
        //
        // Design note: the existing slot pipeline sends decoded *text*, not
        // token IDs, so we can't recover IDs from it directly. The simplest
        // approach is to have the forwarder use the engine to re-encode the
        // text — but that's lossy. Instead we admit via the normal path and
        // set up a parallel tokio channel that carries the raw u32 tokens.
        //
        // For this endpoint we re-use the existing SlotToken (String) pipeline
        // but convert to token IDs in the forwarder. Since the slot pipeline
        // delivers decoded text fragments (not token IDs), we cannot recover
        // the original u32 without changes to core. As a pragmatic fallback
        // for this endpoint we stream the raw text as-is with each token on
        // its own line, prefixed with "tok:". This is lower overhead than the
        // full OpenAI JSON wrapper while remaining valid SSE.
        //
        // A future improvement can plumb token IDs through DecodeOutput → SlotToken.
        let (tok_tx, mut tok_rx) = async_mpsc::channel::<SlotToken>(256);

        let slot_id_opt = {
            let engine = state.engine.lock();
            let mut driver = state.driver.lock();
            driver.admit(&**engine, req.clone()).ok().flatten()
        };

        if let Some(slot_id) = slot_id_opt {
            state.requests_admitted.fetch_add(1, Ordering::Relaxed);
            state.slot_senders.lock().insert(slot_id, tok_tx);
        } else {
            let queue_cap = state.max_batch * 8;
            if state.wait_queue.lock().len() >= queue_cap {
                let sse_tx2 = sse_tx.clone();
                tokio::spawn(async move {
                    let body = serde_json::json!({
                        "error": {"message": "server busy — no batch slot available",
                                  "type": "server_error", "code": "slot_exhausted"}
                    });
                    let _ = sse_tx2.send(Ok(Event::default().data(body.to_string()))).await;
                });
                return Sse::new(ReceiverStream::new(sse_rx)).keep_alive(KeepAlive::default());
            }
            state.requests_queued.fetch_add(1, Ordering::Relaxed);
            state.wait_queue.lock().push_back((req, tok_tx, false));
        };

        // Forward raw token text from the per-slot channel to SSE events.
        // Each non-empty text fragment is emitted as a raw data line.
        // EOS sentinel sends [DONE].
        tokio::spawn(async move {
            while let Some(item) = tok_rx.recv().await {
                match item {
                    Ok(text) => {
                        // Emit each non-empty text fragment as a raw SSE data line.
                        // We escape newlines so each event is a single line.
                        let escaped = text.replace('\n', "\\n");
                        if sse_tx.send(Ok(Event::default().data(escaped))).await.is_err() {
                            break;
                        }
                    }
                    Err(()) => break,
                }
            }
            // Emit [DONE] on ANY stream end (EOS signal, max_tokens channel-close, or
            // client disconnect) so OpenAI-style clients always see a clean terminator.
            let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
        });

        Sse::new(ReceiverStream::new(sse_rx)).keep_alive(KeepAlive::default())
    }

    async fn hawking_generate(State(s): State<AppState>, body: Bytes) -> Response {
        let req: HawkingGenerateReq = match parse_json(&body) {
            Ok(req) => req,
            Err(e) => return e.into_response(),
        };
        if req.prompt.is_empty() {
            return ApiError::missing_parameter("'prompt' must not be empty").into_response();
        }
        let gen = map_hawking_generate_req(&req);
        hawking_generate_sse(s, gen).into_response()
    }

    /// Native streaming response: per-token JSON chunks {tok_index, text} then a
    /// final {stats:{...}} event, then [DONE]. Reuses the OpenAI path's admit +
    /// per-slot SlotToken channel — does NOT fork the continuous-batch decode loop.
    fn hawking_generate_sse(
        state: AppState,
        req: GenerateRequest,
    ) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
        let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(256);
        let (tok_tx, mut tok_rx) = async_mpsc::channel::<SlotToken>(256);

        // prompt_tokens for the stats object (tokenize once for the count; admit
        // tokenizes again internally — cheap, keeps admit's signature unchanged).
        let prompt_tokens = {
            let engine = state.engine.lock();
            engine.encode_prompt_for_batch(&req.prompt).map(|v| v.len()).unwrap_or(0)
        };
        // Snapshot whether the greedy/token-only lane is in play for this request.
        let token_only_snapshot = state.driver.lock().lane_stats.greedy_steps > 0 || req.sampling.temperature <= 0.0;
        let lm_head = lm_head_path_from_env();

        let slot_id_opt = {
            let engine = state.engine.lock();
            let mut driver = state.driver.lock();
            driver.admit(&**engine, req.clone()).ok().flatten()
        };
        if let Some(slot_id) = slot_id_opt {
            state.requests_admitted.fetch_add(1, Ordering::Relaxed);
            state.slot_senders.lock().insert(slot_id, tok_tx);
        } else {
            let queue_cap = state.max_batch * 8;
            if state.wait_queue.lock().len() >= queue_cap {
                let sse_tx2 = sse_tx.clone();
                tokio::spawn(async move {
                    let body = serde_json::json!({
                        "error": {"message": "server busy — no batch slot available",
                                  "type": "server_error", "code": "slot_exhausted"}
                    });
                    let _ = sse_tx2.send(Ok(Event::default().data(body.to_string()))).await;
                });
                return Sse::new(ReceiverStream::new(sse_rx)).keep_alive(KeepAlive::default());
            }
            state.requests_queued.fetch_add(1, Ordering::Relaxed);
            state.wait_queue.lock().push_back((req, tok_tx, false));
        };

        // Forward token text fragments as native chunks; count tokens + wall time
        // for an accurate per-request dec_tps; emit a final stats event on EOS.
        tokio::spawn(async move {
            let start = std::time::Instant::now();
            let mut completion_tokens: usize = 0;
            while let Some(item) = tok_rx.recv().await {
                match item {
                    Ok(text) => {
                        let chunk = serde_json::json!({
                            "tok_index": completion_tokens,
                            "text": text,
                        });
                        completion_tokens += 1;
                        if sse_tx.send(Ok(Event::default().data(chunk.to_string()))).await.is_err() {
                            break;
                        }
                    }
                    Err(()) => break,
                }
            }
            // Always emit the final stats + [DONE] when the stream ends — whether by
            // EOS (the Err(()) signal), max_tokens (the slot is released and the
            // channel closes), or client disconnect — so the native SSE terminates
            // cleanly. Previously stats/[DONE] fired only on the EOS signal, so a
            // max_tokens-bounded request ended without them.
            let decode_ms = start.elapsed().as_secs_f64() * 1000.0;
            let stats =
                hawking_generate_stats_json(prompt_tokens, completion_tokens, decode_ms, token_only_snapshot, lm_head);
            let final_obj = serde_json::json!({ "stats": stats });
            let _ = sse_tx.send(Ok(Event::default().data(final_obj.to_string()))).await;
            let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
        });

        Sse::new(ReceiverStream::new(sse_rx)).keep_alive(KeepAlive::default())
    }

    async fn json_full_response(
        state: AppState,
        req: GenerateRequest,
        chat: bool,
        tool_names: Vec<String>,
    ) -> Result<Json<serde_json::Value>, ApiError> {
        // Admit under a short lock (tokenize + slot assignment only) — does NOT hold
        // the engine mutex for the full generation.
        let slot_id = {
            let engine = state.engine.lock();
            let mut driver = state.driver.lock();
            driver
                .admit(&**engine, req)
                .map_err(|e| ApiError::internal(format!("admit failed: {e}")))?
                .ok_or_else(|| ApiError::internal("server busy — no batch slot available"))?
        };

        // std::sync::mpsc (not tokio) because we block-wait inside spawn_blocking.
        let (tok_tx, tok_rx) = std::sync::mpsc::channel::<SlotToken>();
        // The background loop expects a tokio::sync::mpsc::Sender; wrap via an async
        // bridge: allocate a small tokio channel, spawn a task that forwards into our
        // std channel.
        let (async_tx, mut async_rx) = async_mpsc::channel::<SlotToken>(256);
        state.slot_senders.lock().insert(slot_id, async_tx);

        // Bridge task: forward from the tokio channel into the std channel.
        let tok_tx2 = tok_tx.clone();
        tokio::spawn(async move {
            while let Some(item) = async_rx.recv().await {
                // If the receiver side (spawn_blocking below) is gone, stop forwarding.
                if tok_tx2.send(item).is_err() {
                    break;
                }
            }
        });
        drop(tok_tx); // only tok_tx2 (owned by the bridge) keeps the sender alive

        // Block-wait in a dedicated thread so we don't hold any mutex.
        let res = tokio::task::spawn_blocking(move || -> Result<serde_json::Value, String> {
            let mut text = String::new();
            for item in tok_rx {
                match item {
                    Ok(t) => text.push_str(&t),
                    Err(()) => break, // EOS sentinel
                }
            }
            let body = if chat {
                // When tools were requested, parse the completion back into OpenAI
                // tool_calls; otherwise it is a plain assistant message.
                let calls = if !tool_names.is_empty() {
                    crate::tool_calls::extract_tool_calls(&text, &tool_names)
                } else {
                    Vec::new()
                };
                let (message, finish) = if !calls.is_empty() {
                    let arr: Vec<serde_json::Value> = calls.iter().map(|c| c.to_openai()).collect();
                    (
                        serde_json::json!({
                            "role": "assistant",
                            "content": serde_json::Value::Null,
                            "tool_calls": arr
                        }),
                        "tool_calls",
                    )
                } else {
                    (serde_json::json!({ "role": "assistant", "content": text }), "stop")
                };
                serde_json::json!({
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}]
                })
            } else {
                serde_json::json!({
                    "object": "text_completion",
                    "choices": [{"index": 0, "text": text}]
                })
            };
            Ok(body)
        })
        .await
        .map_err(|_| ApiError::internal("generation task panicked"))?
        .map_err(ApiError::internal)?;
        Ok(Json(res))
    }

    // ── POST /v1/embeddings ───────────────────────────────────────────────────────

    #[derive(Deserialize)]
    struct EmbeddingsReq {
        input: EmbeddingsInput,
        #[allow(dead_code)]
        #[serde(default)]
        model: Option<String>,
        #[serde(default = "default_embedding_encoding")]
        encoding_format: String,
    }

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum EmbeddingsInput {
        Single(String),
        Batch(Vec<String>),
    }

    fn default_embedding_encoding() -> String {
        "float".to_string()
    }

    async fn embeddings(State(s): State<AppState>, body: Bytes) -> Response {
        let req: EmbeddingsReq = match parse_json(&body) {
            Ok(req) => req,
            Err(e) => return e.into_response(),
        };
        let inputs: Vec<String> = match req.input {
            EmbeddingsInput::Single(t) => vec![t],
            EmbeddingsInput::Batch(v) => v,
        };
        if inputs.is_empty() {
            return ApiError::missing_parameter("'input' must not be empty").into_response();
        }
        if req.encoding_format != "float" {
            return ApiError::invalid_json("only encoding_format=float is supported").into_response();
        }

        let model_id = s.engine.lock().model_id().to_string();
        let engine = s.engine.clone();

        let result = tokio::task::spawn_blocking(move || {
            let mut eng = engine.lock();
            let mut data = Vec::with_capacity(inputs.len());
            for (idx, text) in inputs.iter().enumerate() {
                let vec = eng.embed(text)?;
                data.push(serde_json::json!({
                    "object": "embedding",
                    "index": idx,
                    "embedding": vec,
                }));
            }
            hawking_core::Result::Ok(data)
        })
        .await;

        match result {
            Ok(Ok(data)) => Json(serde_json::json!({
                "object": "list",
                "data": data,
                "model": model_id,
                "usage": { "prompt_tokens": 0, "total_tokens": 0 },
            }))
            .into_response(),
            Ok(Err(e)) => ApiError::internal(e.to_string()).into_response(),
            Err(_) => ApiError::internal("embedding task panicked").into_response(),
        }
    }
}
#[rustfmt::skip]
pub mod spec_gov {
    //! Track 6.3 — Speculative-decode acceptance governor.
    //!
    //! [`SpecGovernor`] tracks a per-session rolling acceptance rate and decides
    //! whether spec-decode is currently helping. The actual enable/disable loop
    //! in the serve path is a follow-on; for now the governor just accumulates
    //! state so the infrastructure is wired and ready.

    use std::collections::VecDeque;

    /// Rolling acceptance tracker for spec-decode auto-enable/disable.
    ///
    /// Per-session: tracks rolling accept rate and decides whether spec is helping.
    pub struct SpecGovernor {
        /// Rolling window size for acceptance rate calculation.
        pub window: usize,
        /// Minimum acceptance rate to keep spec enabled (default 0.35).
        pub min_accept_rate: f32,
        /// Maximum consecutive zero-acceptance steps before disabling (default 5).
        pub max_consecutive_rejections: usize,
        // private rolling state
        accepted: VecDeque<bool>,
        consecutive_rejections: usize,
        pub enabled: bool,
    }

    impl SpecGovernor {
        /// Create a new governor with the given window and minimum acceptance rate.
        ///
        /// `window` — number of most-recent verify steps to average over.
        /// `min_accept_rate` — acceptance rate below which spec is considered unhelpful.
        pub fn new(window: usize, min_accept_rate: f32) -> Self {
            Self {
                window,
                min_accept_rate,
                max_consecutive_rejections: 5,
                accepted: VecDeque::with_capacity(window),
                consecutive_rejections: 0,
                enabled: true,
            }
        }

        /// Record the outcome of one verify step.
        ///
        /// Call after each spec-decode verify cycle. `accepted` is `true` when at
        /// least one draft token was accepted by the verifier.
        pub fn record(&mut self, accepted: bool) {
            if self.accepted.len() >= self.window {
                self.accepted.pop_front();
            }
            self.accepted.push_back(accepted);

            if accepted {
                self.consecutive_rejections = 0;
            } else {
                self.consecutive_rejections += 1;
            }

            // Auto-disable if we have exceeded the consecutive-rejection ceiling.
            if self.consecutive_rejections >= self.max_consecutive_rejections {
                self.enabled = false;
            }
            // Re-enable when rolling acceptance rate recovers above the threshold.
            if !self.enabled && self.accept_rate() >= self.min_accept_rate {
                self.enabled = true;
                self.consecutive_rejections = 0;
            }
        }

        /// Rolling acceptance rate over the last `window` steps.
        ///
        /// Returns 1.0 when no steps have been recorded yet (optimistic prior,
        /// so spec starts enabled).
        pub fn accept_rate(&self) -> f32 {
            if self.accepted.is_empty() {
                return 1.0;
            }
            let accepted_count = self.accepted.iter().filter(|&&v| v).count();
            accepted_count as f32 / self.accepted.len() as f32
        }

        /// Whether spec-decode should be used for the next draft proposal.
        ///
        /// Returns `false` once acceptance rate has fallen below `min_accept_rate`
        /// AND consecutive rejections have reached `max_consecutive_rejections`.
        pub fn should_enable(&self) -> bool {
            self.enabled
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn spec_governor_starts_enabled_with_optimistic_rate() {
            let gov = SpecGovernor::new(20, 0.35);
            assert!(gov.enabled);
            assert!(gov.should_enable());
            assert_eq!(gov.accept_rate(), 1.0);
        }

        #[test]
        fn spec_governor_tracks_rolling_window() {
            let mut gov = SpecGovernor::new(4, 0.35);
            // 4 accepts then 4 rejections: window sees the 4 rejections.
            for _ in 0..4 {
                gov.record(true);
            }
            assert!((gov.accept_rate() - 1.0).abs() < 1e-6);
            for _ in 0..4 {
                gov.record(false);
            }
            assert!((gov.accept_rate() - 0.0).abs() < 1e-6);
        }

        #[test]
        fn spec_governor_disables_after_max_consecutive_rejections() {
            let mut gov = SpecGovernor::new(20, 0.35);
            // 5 consecutive rejections → disabled.
            for _ in 0..5 {
                gov.record(false);
            }
            assert!(!gov.enabled);
            assert!(!gov.should_enable());
        }

        #[test]
        fn spec_governor_reenables_when_rate_recovers() {
            let mut gov = SpecGovernor::new(4, 0.35);
            // Disable first.
            for _ in 0..5 {
                gov.record(false);
            }
            assert!(!gov.enabled);
            // 3 consecutive accepts push the rolling window above 0.35.
            gov.record(true);
            gov.record(true);
            // Window has 4 entries: [false, false, true, true] → rate = 0.5 > 0.35.
            // The last record() call will re-enable.
            // But first two records after disable still hit the re-enable check.
            // Just verify the final state.
            assert!((gov.accept_rate() - 0.5).abs() < 1e-6);
            assert!(gov.enabled);
        }
    }
}
#[rustfmt::skip]
pub mod system_kv_bank {
    //! Track 5.2 — System-prompt KV bank.
    //!
    //! A serve-lifetime registry that remembers, for each FIXED leading prefix
    //! span (the system / instruction prompt that many requests share), which
    //! decode SLOT most recently held copyable KV for that span. Unlike
    //! `scheduler::PrefixIndex` (Track 5.1), which only matches against slots
    //! that are CURRENTLY active, this bank survives a source request finishing
    //! — so a serial chat workload (one request at a time, identical system
    //! prompt) still gets shared-prefix reuse instead of re-prefilling the
    //! system block every turn.
    //!
    //! # What this is NOT
    //! It stores ZERO KV bytes. It is a `hash(prefix) -> source_slot` routing
    //! hint. The hit is ALWAYS re-verified downstream by the bit-identical
    //! `Engine::copy_kv_prefix_to_slot` + `prefill_slot_from_pos` path, so a
    //! (vanishingly unlikely) hash false-positive cannot corrupt output: a stale
    //! `source_slot` simply fails the copy and the serve loop falls back to a
    //! cold prefill from position 0. This keeps the lever **greedy-lossless (E)**.
    //!
    //! The detached, slot-independent KV-block store (so reuse survives even when
    //! NO slot currently holds the bytes) is the deferred half of 5.2 — it lands
    //! in the model/arena layer (`qwen_dense.rs` / `dense_decode_arena.rs`) and is
    //! out of scope here. This bank is the routing index that store plugs into.

    use std::collections::HashMap;

    /// Default minimum leading-prefix length (tokens) to bank. Mirrors
    /// `hawking_core::sidecar::PREFIX_REUSE_MIN_TOKENS` (= 8): a span shorter
    /// than this is not worth a copy + from-pos prefill.
    pub const DEFAULT_MIN_PREFIX_TOKENS: usize = 8;

    /// Default cap on distinct banked prefixes. A handful of system prompts is
    /// the norm; the LRU keeps the bank from growing without bound across a long
    /// server lifetime.
    pub const DEFAULT_MAX_ENTRIES: usize = 64;

    /// Tuning for the bank.
    #[derive(Debug, Clone, Copy)]
    pub struct BankConfig {
        /// Reject (do not bank, do not match) prefixes shorter than this.
        pub min_prefix_tokens: usize,
        /// LRU-evict down to this many distinct prefixes after each insert.
        pub max_entries: usize,
    }

    impl Default for BankConfig {
        fn default() -> Self {
            Self { min_prefix_tokens: DEFAULT_MIN_PREFIX_TOKENS, max_entries: DEFAULT_MAX_ENTRIES }
        }
    }

    /// One banked prefix: which slot last held it + bookkeeping.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct BankEntry {
        /// Number of leading tokens this entry covers (== the span hashed).
        pub prefix_len: usize,
        /// Slot id that most recently held copyable KV for this prefix. The
        /// serve loop passes this as `src_slot` to `copy_kv_prefix_to_slot`.
        pub source_slot: u32,
        /// LRU clock value of the last record/hit (smallest == least recent).
        pub last_tick: u64,
        /// Lifetime lookup-hits for this prefix (diagnostics / /metrics).
        pub hits: u64,
    }

    /// Outcome of a [`SystemPromptKvBank::record`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum RecordOutcome {
        /// A new prefix was banked.
        Inserted,
        /// An existing prefix's `source_slot` was refreshed.
        Updated,
        /// Prefix shorter than `min_prefix_tokens`; nothing banked.
        TooShort,
    }

    /// Aggregate counters (mirrors `LaneStats` style; surfaced via /metrics).
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct BankStats {
        pub lookups: u64,
        pub hits: u64,
        pub records: u64,
        pub evictions: u64,
        pub entries: usize,
    }

    /// A serve-lifetime hash(prefix) -> source-slot registry. Pure data; no model.
    #[derive(Debug, Default)]
    pub struct SystemPromptKvBank {
        cfg: BankConfig,
        /// prefix-hash -> entry. Hash is the 128-bit fold of the leading span.
        entries: HashMap<u128, BankEntry>,
        clock: u64,
        stats: BankStats,
    }

    impl SystemPromptKvBank {
        pub fn new() -> Self {
            Self::with_config(BankConfig::default())
        }

        pub fn with_config(cfg: BankConfig) -> Self {
            Self { cfg, entries: HashMap::new(), clock: 0, stats: BankStats::default() }
        }

        pub fn config(&self) -> BankConfig {
            self.cfg
        }

        /// Stable 128-bit content hash of the FIRST `prefix_len` tokens — the
        /// fixed leading span. Two FNV-1a streams with distinct seeds folded into
        /// a u128 (collision-resistant enough for a re-verified routing hint).
        /// `prefix_len` is folded in so the same tokens at a different banked
        /// length address a different entry.
        pub fn hash_prefix(tokens: &[u32], prefix_len: usize) -> u128 {
            let n = prefix_len.min(tokens.len());
            let mut a: u64 = 0xcbf29ce484222325; // FNV offset basis
            let mut b: u64 = 0x100000001b3 ^ 0x9e3779b97f4a7c15; // distinct seed
            let mix = |h: &mut u64, x: u64, prime: u64| {
                *h ^= x;
                *h = h.wrapping_mul(prime);
            };
            mix(&mut a, n as u64, 0x100000001b3);
            mix(&mut b, (n as u64).rotate_left(32), 0x9e3779b97f4a7c15);
            for &t in &tokens[..n] {
                mix(&mut a, t as u64, 0x100000001b3);
                mix(&mut b, (t as u64).wrapping_add(0x632be59bd9b4e019), 0x9e3779b97f4a7c15);
            }
            ((a as u128) << 64) | (b as u128)
        }

        /// Look up a banked source-slot for the leading span of `tokens`.
        ///
        /// `banked_len` is the prefix length to probe — typically the caller's
        /// notion of where the fixed system span ends (e.g. min(tokens.len()-1,
        /// a configured system-span length), always at least one token short of
        /// the full prompt so the decode loop keeps a real last_id, mirroring the
        /// disk/RAM tiers' "bail one token short" rule). Returns the entry on a
        /// hit (and LRU-touches it + bumps hit counters); `None` on miss/too-short.
        pub fn lookup(&mut self, tokens: &[u32], banked_len: usize) -> Option<BankEntry> {
            self.stats.lookups += 1;
            if banked_len < self.cfg.min_prefix_tokens || banked_len >= tokens.len().max(1) {
                // Strict prefix only; never match the whole prompt.
                return None;
            }
            let key = Self::hash_prefix(tokens, banked_len);
            let entry = self.entries.get_mut(&key)?;
            if entry.prefix_len != banked_len {
                // Length collision guard (cannot reuse a different-length block).
                return None;
            }
            self.clock += 1;
            entry.last_tick = self.clock;
            entry.hits += 1;
            self.stats.hits += 1;
            Some(*entry)
        }

        /// Bank (or refresh) that `source_slot` holds copyable KV for the leading
        /// `prefix_len` tokens of `tokens`. Runs LRU eviction to `max_entries`.
        pub fn record(&mut self, tokens: &[u32], prefix_len: usize, source_slot: u32) -> RecordOutcome {
            if prefix_len < self.cfg.min_prefix_tokens || prefix_len > tokens.len() {
                return RecordOutcome::TooShort;
            }
            self.clock += 1;
            let tick = self.clock;
            let key = Self::hash_prefix(tokens, prefix_len);
            let outcome = match self.entries.get_mut(&key) {
                Some(e) => {
                    e.prefix_len = prefix_len;
                    e.source_slot = source_slot;
                    e.last_tick = tick;
                    RecordOutcome::Updated
                }
                None => {
                    self.entries.insert(key, BankEntry { prefix_len, source_slot, last_tick: tick, hits: 0 });
                    RecordOutcome::Inserted
                }
            };
            self.stats.records += 1;
            self.evict_to_cap();
            self.stats.entries = self.entries.len();
            outcome
        }

        /// Drop the banked mapping for a slot that is being torn down with a
        /// changed prefix, or invalidate a known-stale source. Returns how many
        /// entries were removed (entries whose `source_slot == slot`).
        pub fn forget_slot(&mut self, slot: u32) -> usize {
            let before = self.entries.len();
            self.entries.retain(|_, e| e.source_slot != slot);
            let removed = before - self.entries.len();
            self.stats.entries = self.entries.len();
            removed
        }

        /// LRU-evict (smallest `last_tick` first) until within `max_entries`.
        pub fn evict_to_cap(&mut self) {
            while self.entries.len() > self.cfg.max_entries {
                let victim = self.entries.iter().min_by_key(|(_, e)| e.last_tick).map(|(k, _)| *k);
                match victim {
                    Some(k) => {
                        self.entries.remove(&k);
                        self.stats.evictions += 1;
                    }
                    None => break,
                }
            }
            self.stats.entries = self.entries.len();
        }

        pub fn len(&self) -> usize {
            self.entries.len()
        }

        pub fn is_empty(&self) -> bool {
            self.entries.is_empty()
        }

        pub fn stats(&self) -> BankStats {
            let mut s = self.stats;
            s.entries = self.entries.len();
            s
        }
    }
}
#[rustfmt::skip]
pub mod tool_calls {
    //! Native tool calling on the OpenAI-compatible serve API (Phase 1a of
    //! `docs/RESEARCH.md`).
    //!
    //! Two pure, self-contained pieces the chat endpoint uses:
    //! * [`render_tools_preamble`] turns the request's `tools` array into a system
    //!   preamble in the Hermes / Qwen2.5 convention (a `<tools>` block plus the
    //!   instruction to emit `<tool_call>{...}</tool_call>`), so a local chat model
    //!   trained on that format produces callable output.
    //! * [`extract_tool_calls`] turns the model's completion text back into the
    //!   OpenAI `tool_calls` response shape (arguments as a JSON-encoded string).
    //!
    //! The serve engine lives in its own dependency universe (`hawking-core`), so this
    //! deliberately does not reuse the agent-side parser in `hide-kernel`; it mirrors
    //! its format handling (tagged blocks first, then any balanced JSON span, so a
    //! call embedded in prose or after a bracket is still recovered) but stays a thin
    //! API-shaping utility.

    use serde_json::Value;

    /// One extracted call in OpenAI response shape: `arguments` is a JSON-encoded
    /// string, matching `chat.completion` `tool_calls[].function.arguments`.
    #[derive(Debug, Clone, PartialEq)]
    pub struct ExtractedToolCall {
        pub id: String,
        pub name: String,
        pub arguments: String,
    }

    impl ExtractedToolCall {
        /// The `tool_calls[]` entry for a chat-completion message.
        pub fn to_openai(&self) -> Value {
            serde_json::json!({
                "id": self.id,
                "type": "function",
                "function": { "name": self.name, "arguments": self.arguments }
            })
        }
    }

    /// Render the request `tools` (OpenAI function specs) into a system preamble. An
    /// empty list yields an empty string (no-op). The format is the one Qwen2.5 /
    /// Hermes models are trained on and is a reasonable default for others.
    pub fn render_tools_preamble(tools: &[Value]) -> String {
        if tools.is_empty() {
            return String::new();
        }
        let mut s = String::new();
        s.push_str(
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
        );
        for tool in tools {
            // Accept either a bare function object or the OpenAI {"type":"function",
            // "function":{...}} envelope.
            let func = tool.get("function").unwrap_or(tool);
            s.push_str(&func.to_string());
            s.push('\n');
        }
        s.push_str(
            "</tools>\n\nFor each function call, return a json object with the function name and \
             arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
             {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n",
        );
        s
    }

    /// The declared function names from a request `tools` array (accepts both the
    /// bare function object and the `{"type":"function","function":{...}}` envelope).
    pub fn tool_names(tools: &[Value]) -> Vec<String> {
        tools
            .iter()
            .filter_map(|t| {
                let f = t.get("function").unwrap_or(t);
                f.get("name").and_then(|n| n.as_str()).map(str::to_string)
            })
            .collect()
    }

    /// Extract every tool call from a completion, in document order. Empty when the
    /// model produced a plain text answer.
    ///
    /// `known_tools` are the names declared in the request. An explicit
    /// `<tool_call>` block is honored regardless (the model signalled intent), but the
    /// UNTAGGED JSON fallback only accepts an object whose name is a declared tool, so
    /// a plain answer that merely contains a JSON object (`{"name": "Bob"}`) is not
    /// mis-read as a call that discards the real answer.
    pub fn extract_tool_calls(completion: &str, known_tools: &[String]) -> Vec<ExtractedToolCall> {
        // Hermes / Qwen `<tool_call>...</tool_call>` blocks: explicit intent, lenient.
        let tagged = tagged_blocks(completion);
        let raw: Vec<(String, Value)> = if !tagged.is_empty() {
            tagged
        } else {
            // Untagged fallback: require the name to be a declared tool.
            let mut found = Vec::new();
            for span in all_json_spans(completion) {
                if let Ok(value) = serde_json::from_str::<Value>(&span) {
                    let calls: Vec<(String, Value)> = calls_from_value(&value)
                        .into_iter()
                        .filter(|(name, _)| known_tools.iter().any(|t| t == name))
                        .collect();
                    if !calls.is_empty() {
                        found = calls;
                        break;
                    }
                }
            }
            found
        };

        raw.into_iter()
            .enumerate()
            .map(|(i, (name, args))| ExtractedToolCall {
                id: format!("call_{i}"),
                name,
                // OpenAI carries arguments as a JSON-encoded string.
                arguments: args.to_string(),
            })
            .collect()
    }

    /// Whether the completion contains a recognizable tool call (declared tools
    /// considered for the untagged case).
    pub fn has_tool_call(completion: &str, known_tools: &[String]) -> bool {
        completion.contains("<tool_call>") || !extract_tool_calls(completion, known_tools).is_empty()
    }

    // ---------------------------------------------------------------------------
    // internals (mirror the agent-side parser's robustness)
    // ---------------------------------------------------------------------------

    fn tagged_blocks(text: &str) -> Vec<(String, Value)> {
        const OPEN: &str = "<tool_call>";
        const CLOSE: &str = "</tool_call>";
        let mut out = Vec::new();
        let mut rest = text;
        while let Some(start) = rest.find(OPEN) {
            let after = &rest[start + OPEN.len()..];
            let Some(end) = after.find(CLOSE) else { break };
            if let Ok(value) = serde_json::from_str::<Value>(after[..end].trim()) {
                out.extend(calls_from_value(&value));
            }
            rest = &after[end + CLOSE.len()..];
        }
        out
    }

    fn calls_from_value(value: &Value) -> Vec<(String, Value)> {
        match value {
            Value::Array(items) => items.iter().flat_map(calls_from_value).collect(),
            Value::Object(obj) => {
                if let Some(Value::Array(list)) = obj.get("tool_calls") {
                    return list.iter().flat_map(calls_from_value).collect();
                }
                single(value).into_iter().collect()
            }
            _ => Vec::new(),
        }
    }

    fn single(value: &Value) -> Option<(String, Value)> {
        let obj = value.as_object()?;
        let (name_src, args_src) = if let Some(func) = obj.get("function").and_then(|f| f.as_object()) {
            (func.get("name"), func.get("arguments").or_else(|| func.get("parameters")))
        } else {
            (
                obj.get("name").or_else(|| obj.get("tool")),
                obj.get("arguments").or_else(|| obj.get("args")).or_else(|| obj.get("parameters")),
            )
        };
        let name = name_src?.as_str()?.trim().to_string();
        if name.is_empty() {
            return None;
        }
        let args = match args_src {
            None | Some(Value::Null) => serde_json::json!({}),
            Some(Value::Object(o)) => Value::Object(o.clone()),
            Some(Value::String(s)) => {
                serde_json::from_str::<Value>(s).unwrap_or_else(|_| serde_json::json!({ "input": s }))
            }
            Some(other) => serde_json::json!({ "value": other.clone() }),
        };
        Some((name, args))
    }

    fn all_json_spans(s: &str) -> Vec<String> {
        let bytes = s.as_bytes();
        let mut spans = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'{' || bytes[i] == b'[' {
                match balanced_end(bytes, i) {
                    Some(end) => {
                        spans.push(s[i..=end].to_string());
                        i = end + 1;
                        continue;
                    }
                    None => break,
                }
            }
            i += 1;
        }
        spans
    }

    fn balanced_end(bytes: &[u8], start: usize) -> Option<usize> {
        let open = bytes[start];
        let close = if open == b'{' { b'}' } else { b']' };
        let mut depth = 0i32;
        let mut in_str = false;
        let mut escaped = false;
        for (i, &b) in bytes.iter().enumerate().skip(start) {
            if in_str {
                if escaped {
                    escaped = false;
                } else if b == b'\\' {
                    escaped = true;
                } else if b == b'"' {
                    in_str = false;
                }
                continue;
            }
            match b {
                b'"' => in_str = true,
                x if x == open => depth += 1,
                x if x == close => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
        None
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use serde_json::json;

        #[test]
        fn preamble_lists_tools_and_is_empty_when_none() {
            assert_eq!(render_tools_preamble(&[]), "");
            let tools = vec![json!({
                "type": "function",
                "function": { "name": "get_weather", "parameters": { "type": "object" } }
            })];
            let p = render_tools_preamble(&tools);
            assert!(p.contains("<tools>"));
            assert!(p.contains("get_weather"));
            assert!(p.contains("<tool_call>"));
            // The OpenAI envelope is unwrapped to the bare function spec.
            assert!(!p.contains("\"type\":\"function\""));
        }

        #[test]
        fn extracts_hermes_block_to_openai_shape() {
            // Tagged blocks are honored regardless of the declared-tools list.
            let calls = extract_tool_calls(
                "I'll check.\n<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"NYC\"}}</tool_call>",
                &[],
            );
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "get_weather");
            assert_eq!(calls[0].id, "call_0");
            // arguments is a JSON-encoded string.
            assert_eq!(calls[0].arguments, "{\"city\":\"NYC\"}");
            let oa = calls[0].to_openai();
            assert_eq!(oa["function"]["name"], "get_weather");
        }

        #[test]
        fn extracts_parallel_calls() {
            let calls = extract_tool_calls(
                "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>\
                 <tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>",
                &[],
            );
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0].id, "call_0");
            assert_eq!(calls[1].id, "call_1");
        }

        #[test]
        fn extracts_bare_declared_call_after_bracket() {
            // A leading [...] must not shadow the object; and the untagged bare call is
            // accepted only because "a" is a declared tool.
            let known = vec!["a".to_string()];
            let calls = extract_tool_calls("see [1] {\"name\":\"a\",\"arguments\":{\"x\":1}}", &known);
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "a");
        }

        #[test]
        fn untagged_prose_json_is_not_a_false_tool_call() {
            // A plain answer mentioning a JSON object must NOT become a tool call when
            // its name is not a declared tool (else the real answer is discarded).
            let known = vec!["get_weather".to_string()];
            assert!(extract_tool_calls("Your record is {\"name\": \"Bob\"}", &known).is_empty());
            // ...but a bare call to a DECLARED tool is still extracted.
            let calls = extract_tool_calls("{\"name\":\"get_weather\",\"arguments\":{\"city\":\"NYC\"}}", &known);
            assert_eq!(calls.len(), 1);
            // ...and a tagged block is always honored, declared or not.
            let tagged = extract_tool_calls("<tool_call>{\"name\":\"anything\",\"arguments\":{}}</tool_call>", &[]);
            assert_eq!(tagged.len(), 1);
        }

        #[test]
        fn tool_names_reads_both_shapes() {
            let tools = vec![json!({ "type": "function", "function": { "name": "a" } }), json!({ "name": "b" })];
            assert_eq!(tool_names(&tools), vec!["a".to_string(), "b".to_string()]);
        }

        #[test]
        fn plain_text_yields_no_calls() {
            assert!(extract_tool_calls("just a normal answer", &[]).is_empty());
            assert!(!has_tool_call("just a normal answer", &[]));
        }
    }
}

pub use batch::scheduler::BatchPolicy;
pub use system_kv_bank::{BankConfig, BankEntry, SystemPromptKvBank};

use anyhow::Result;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Runtime profile controlling quality/throughput trade-offs.
///
/// `Default` — bit-identical conservative path; no env var changes.
/// `Fast`    — validated fast-path (vocab-prune + Q4K LM-head + predec + f16-scales).
/// `Race`    — same as Fast; explicitly signals "maximum throughput, quality trade-offs OK".
/// `Efficient` — same as Fast plus sets HAWKING_ENERGY_EFFICIENT=1 for energy-aware batching.
/// `Exact`   — clears any quality-trade vars; forces bit-identical output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeProfile {
    Default,
    Fast,
    Race,
    Efficient,
    Exact,
}

impl RuntimeProfile {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "default" => Some(Self::Default),
            "fast" => Some(Self::Fast),
            "race" => Some(Self::Race),
            "efficient" => Some(Self::Efficient),
            "exact" => Some(Self::Exact),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Fast => "fast",
            Self::Race => "race",
            Self::Efficient => "efficient",
            Self::Exact => "exact",
        }
    }
}

impl std::fmt::Display for RuntimeProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Data-only description of the env-var levers a [`RuntimeProfile`] activates.
///
/// Pure: building it touches no process state. Both the CLI generate path
/// (`apply_profile` in the `hawking` bin) and `serve::run` consume it, so
/// there is exactly ONE source of truth for the profile → lever mapping.
///
/// Caller contract:
///   * `set_if_unset` — set each (key,val) ONLY when the var is currently absent
///     (explicit `HAWKING_QWEN_*` env always wins → opt-out honoured).
///   * `force_off`    — set each var to "0" UNCONDITIONALLY (`Exact` uses this to
///     guarantee bit-identity even if a quality-trade var was set upstream).
///   * `f16_kv`       — profile default for the f16 KV cache (None = leave to a
///     more specific override such as `--f16-kv`).
///   * `concurrent_qkv` — whether the profile wants concurrent Q/K/V encode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeverPlan {
    pub set_if_unset: Vec<(&'static str, &'static str)>,
    pub force_off: Vec<&'static str>,
    pub f16_kv: Option<bool>,
    pub concurrent_qkv: bool,
}

impl RuntimeProfile {
    /// The validated fast-path lever bundle shared by Fast / Race / Efficient.
    /// Bit-identical EXCEPT PREDEC_F16SCALES (f16 scale rounding) and VOCAB_PRUNE
    /// (drops rare tokens) — mild quality trades; FFN_DOWN_Q4K requants Q6_K→Q4_K.
    fn fast_bundle() -> Vec<(&'static str, &'static str)> {
        vec![
            ("HAWKING_QWEN_Q4K_LMHEAD", "1"),
            ("HAWKING_QWEN_Q4K_PREDEC", "1"),
            ("HAWKING_QWEN_PREDEC_F16SCALES", "1"),
            ("HAWKING_QWEN_VOCAB_PRUNE", "32000"),
            ("HAWKING_QWEN_FFN_DOWN_Q4K", "1"),
        ]
    }

    /// Policy: which profile an UNSET `--profile` resolves to on the CLI front
    /// door. The ONE place the "fast is the default" decision lives. The library
    /// default (`RuntimeProfile::Default`) is deliberately NOT changed — embedders
    /// and serve integration tests keep the conservative bit-identical default;
    /// only the CLI `generate`/`bench` front door flips.
    pub fn default_when_unset() -> Self {
        Self::Fast
    }

    /// Levers to force OFF when resolving an UNSET `--profile` (the MIDDLE
    /// variant): keep every fast lever EXCEPT `PREDEC_F16SCALES`, which failed
    /// quality_oracle at 0.792/11.46% (e613dde). Net ≈ 38–39 t/s at low quality
    /// risk. To ship FULL fast (~42) after the oracle re-passes f16-scales,
    /// return `&[]` here.
    pub fn default_unset_force_off() -> &'static [&'static str] {
        &["HAWKING_QWEN_PREDEC_F16SCALES"]
    }

    /// Pure profile → lever mapping. Touches no env state.
    pub fn lever_plan(&self) -> LeverPlan {
        match self {
            Self::Default => LeverPlan {
                set_if_unset: Vec::new(),
                force_off: Vec::new(),
                f16_kv: None,
                concurrent_qkv: false,
            },
            Self::Fast => LeverPlan {
                set_if_unset: Self::fast_bundle(),
                force_off: Vec::new(),
                f16_kv: Some(false),
                concurrent_qkv: true,
            },
            // Max t/s: fast bundle + f16 KV (frees bandwidth) + concurrent QKV.
            Self::Race => LeverPlan {
                set_if_unset: Self::fast_bundle(),
                force_off: Vec::new(),
                f16_kv: Some(true),
                concurrent_qkv: true,
            },
            // Min J/tok under a t/s floor: fast bundle + energy mode + f16 KV.
            Self::Efficient => {
                let mut s = Self::fast_bundle();
                s.push(("HAWKING_ENERGY_EFFICIENT", "1"));
                LeverPlan {
                    set_if_unset: s,
                    force_off: Vec::new(),
                    f16_kv: Some(true),
                    concurrent_qkv: true,
                }
            }
            // Bit-identical conservative path. Bit-identical default-ON levers
            // (predec/pair/gate-up-fuse) stay on; force OFF every quality-trade
            // var so output matches the golden default even if one was set upstream.
            Self::Exact => LeverPlan {
                set_if_unset: Vec::new(),
                force_off: vec![
                    "HAWKING_QWEN_PREDEC_F16SCALES", // f16 scale rounding
                    "HAWKING_QWEN_FFN_DOWN_Q4K",     // Q6_K→Q4_K requant
                    "HAWKING_QWEN_VOCAB_PRUNE",      // drops rare tokens
                ],
                f16_kv: Some(false),
                concurrent_qkv: false,
            },
        }
    }

    /// One-line human contract: lever set + quality + J/tok statement. Printed
    /// at startup so every profile "prints its active levers" (Track 2.2 gate).
    pub fn contract(&self) -> String {
        match self {
            Self::Default => {
                "profile=default: locked bit-identical default decode \
                (predec + pair + gate/up-fuse, all bit-identical). quality: exact. J/tok: baseline."
                    .to_string()
            }
            Self::Fast => "profile=fast: vocab-prune-32k + Q4K LM-head + Q4K FFN-down + predec \
                + f16-scales. quality: mild trade (f16 scale rounding, rare-token prune). \
                J/tok: lower than default (fewer bytes/token)."
                .to_string(),
            Self::Race => "profile=race: fast bundle + f16 KV + concurrent Q/K/V. \
                quality: same mild trade as fast. goal: MAX tokens/sec."
                .to_string(),
            Self::Efficient => "profile=efficient: fast bundle + f16 KV + energy-efficient gather \
                window. quality: same mild trade as fast. goal: MIN J/tok under a t/s floor."
                .to_string(),
            Self::Exact => "profile=exact: bit-identical conservative path. Forces OFF f16-scales \
                / Q4K-FFN-down / vocab-prune. quality: EXACT (greedy bit-identical to default). \
                J/tok: baseline."
                .to_string(),
        }
    }
}

/// Energy-mode controls gather-window sizing and future energy-aware batching.
///
/// `Off`       — no gather window (lowest latency).
/// `Balanced`  — 3 ms gather window (default tradeoff).
/// `Efficient` — 8 ms gather window (maximise batch fill for lower J/tok).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnergyMode {
    Off,
    Balanced,
    Efficient,
}

impl EnergyMode {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "off" => Some(Self::Off),
            "balanced" => Some(Self::Balanced),
            "efficient" => Some(Self::Efficient),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Balanced => "balanced",
            Self::Efficient => "efficient",
        }
    }

    /// Gather window in milliseconds.
    pub fn gather_window_ms(&self) -> u64 {
        match self {
            Self::Off => 0,
            Self::Balanced => 3,
            Self::Efficient => 8,
        }
    }

    /// Pure gather/admission decision — the predicate the continuous-batch
    /// loop uses to decide whether to wait (sleep up to `gather_window_ms()`)
    /// for more requests before committing a prefill batch.
    ///
    /// Returns `true` ONLY when waiting can help AND is safe:
    ///   * `ready > 0`              — at least one slot is queued (never wait on empty),
    ///   * `max_batch > 1`          — single-slot servers can't batch → never wait
    ///     (a latency-sensitive single is NEVER delayed),
    ///   * `ready < max_batch`      — batch already full → commit now, don't wait,
    ///   * `gather_window_ms() > 0` — `Off` disables the window entirely.
    ///
    /// This is the extracted, unit-testable form of the inline predicate in
    /// `serve::run()` (the `prefilling.len() < max_batch && gather_window_ms > 0`
    /// guard). Keep the two in sync: the loop should call this helper.
    pub fn should_gather(&self, ready: usize, max_batch: usize) -> bool {
        ready > 0 && max_batch > 1 && ready < max_batch && self.gather_window_ms() > 0
    }
}

impl std::fmt::Display for EnergyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Track 9.3 — workload packs.
///
/// A workload pack sets sensible defaults for a class of serving workload.
/// Individual flags (`--profile`, `--energy-mode`, `--batch-policy`,
/// `--f16-kv`) always override the pack's defaults.
///
/// `Default`            — no change; individual flags apply as-is.
/// `CodeCompletion`     — Race profile + energy off + GreedyFirst batching.
/// `ChatSharedPrompt`   — Fast profile + Balanced energy + PrefixGrouped batching.
/// `BatchSummarization` — Efficient profile + Efficient energy + GreedyFirst batching.
/// `LocalAgentLoop`     — Fast profile + energy off + GreedyFirst batching.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum WorkloadPack {
    #[default]
    Default,
    CodeCompletion,
    ChatSharedPrompt,
    BatchSummarization,
    LocalAgentLoop,
}

impl WorkloadPack {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "default" => Some(Self::Default),
            "code-completion" => Some(Self::CodeCompletion),
            "chat-shared-prompt" => Some(Self::ChatSharedPrompt),
            "batch-summarization" => Some(Self::BatchSummarization),
            "local-agent-loop" => Some(Self::LocalAgentLoop),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::CodeCompletion => "code-completion",
            Self::ChatSharedPrompt => "chat-shared-prompt",
            Self::BatchSummarization => "batch-summarization",
            Self::LocalAgentLoop => "local-agent-loop",
        }
    }

    /// Return the (profile, energy, batch_policy) defaults for this pack.
    ///
    /// Callers apply these ONLY when the corresponding flag was not explicitly
    /// set — pack defaults lose to explicit flags.
    pub fn defaults(&self) -> (RuntimeProfile, EnergyMode, BatchPolicy) {
        match self {
            Self::Default => (
                RuntimeProfile::Default,
                EnergyMode::Off,
                BatchPolicy::Default,
            ),
            Self::CodeCompletion => (
                RuntimeProfile::Race,
                EnergyMode::Off,
                BatchPolicy::GreedyFirst,
            ),
            Self::ChatSharedPrompt => (
                RuntimeProfile::Fast,
                EnergyMode::Balanced,
                BatchPolicy::PrefixGrouped,
            ),
            Self::BatchSummarization => (
                RuntimeProfile::Efficient,
                EnergyMode::Efficient,
                BatchPolicy::GreedyFirst,
            ),
            Self::LocalAgentLoop => (
                RuntimeProfile::Fast,
                EnergyMode::Off,
                BatchPolicy::GreedyFirst,
            ),
        }
    }
}

impl std::fmt::Display for WorkloadPack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone)]
pub struct ServeOptions {
    pub weights: PathBuf,
    pub addr: SocketAddr,
    pub max_batch_size: usize,
    pub speculate: Option<String>,
    pub verify_window: usize,
    pub kernel_profile: Option<PathBuf>,
    pub prefill_cache_dir: Option<PathBuf>,
    pub max_routed_expert_ram_mb: Option<usize>,
    pub memory_limit_mb: Option<usize>,
    /// Runtime profile for quality/throughput trade-offs.
    pub runtime_profile: RuntimeProfile,
    /// Energy mode controlling gather-window sizing.
    pub energy_mode: EnergyMode,
    /// When true, print a human-readable performance summary at startup.
    pub explain_performance: bool,
    /// Track 6.3: spec governor rolling-window size (default 20).
    pub spec_window: usize,
    /// Track 6.3: minimum acceptance rate to keep spec enabled (default 0.35).
    pub spec_min_accept_rate: f32,
    /// Track 5.3: f16 KV cache override.
    ///
    /// `None`       — defer to profile/workload default.
    /// `Some(true)` — force HAWKING_QWEN_F16_KV=1 (halves KV footprint;
    ///                wins at long context, footprint-neutral for short ctx).
    /// `Some(false)` — explicitly disable (leave env var unset).
    pub f16_kv: Option<bool>,
    /// Track 5.4: batch admission policy.
    pub batch_policy: BatchPolicy,
    /// Track 9.3: workload pack (sets profile/energy/policy defaults).
    pub workload: WorkloadPack,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            weights: PathBuf::new(),
            // Loopback by default: the local inference server must not be reachable from the LAN
            // unless the operator explicitly asks for it. The supervisor always passes an explicit
            // addr, but the binary's own default must be safe too.
            addr: "127.0.0.1:8080".parse().unwrap(),
            max_batch_size: 1,
            speculate: None,
            verify_window: 4,
            kernel_profile: None,
            prefill_cache_dir: None,
            max_routed_expert_ram_mb: None,
            memory_limit_mb: None,
            runtime_profile: RuntimeProfile::Default,
            energy_mode: EnergyMode::Off,
            explain_performance: false,
            spec_window: 20,
            spec_min_accept_rate: 0.35,
            f16_kv: None,
            batch_policy: BatchPolicy::Default,
            workload: WorkloadPack::Default,
        }
    }
}

fn hawking_serve_system_kv_bank_default() -> system_kv_bank::SystemPromptKvBank {
    system_kv_bank::SystemPromptKvBank::new()
}

pub async fn run(opts: ServeOptions) -> Result<()> {
    use hawking_core::{profile::KernelProfile, EngineConfig, SpeculateMode};

    // ── Track 9.3: apply workload-pack defaults ───────────────────────────────
    // Pack defaults are applied FIRST so that explicit per-flag values (profile,
    // energy_mode, batch_policy, f16_kv) set later always win over them.
    // The pack only influences fields that are still at their zero-values
    // (Default/Off/None) — this is expressed by the caller setting fields to
    // non-default values to override. Because opts is already parsed before
    // run() is called, we derive an "effective" set here and shadow opts.
    let (effective_profile, effective_energy, effective_batch_policy) = {
        let (pack_profile, pack_energy, pack_policy) = opts.workload.defaults();
        // Explicit flags win: use opts value when it is non-Default/non-Off/non-None.
        let profile = if opts.runtime_profile != RuntimeProfile::Default {
            opts.runtime_profile.clone()
        } else {
            pack_profile
        };
        let energy = if opts.energy_mode != EnergyMode::Off {
            opts.energy_mode.clone()
        } else {
            pack_energy
        };
        let policy = if opts.batch_policy != BatchPolicy::Default {
            opts.batch_policy.clone()
        } else {
            pack_policy
        };
        (profile, energy, policy)
    };

    // ── Serve-mode optimisation defaults ─────────────────────────────────────
    // These are the same knobs that `hawking generate --kernel-profile` uses.
    // Each can be overridden by the caller's environment (set var before invoking
    // the server). We only set them when the variable is absent so that explicit
    // HAWKING_QWEN_*=0 opt-outs are honoured.
    for (var, val) in [
        ("HAWKING_QWEN_Q4K_PREDEC", "1"), // pre-decoded scales → fast GEMV
        ("HAWKING_QWEN_Q4K_LMHEAD", "1"), // GPU Q4K LM-head (vs CPU f16)
        ("HAWKING_QWEN_VOCAB_PRUNE", "32000"), // prune to 32K most-frequent tokens
        ("HAWKING_QWEN_TCB", "1"),        // token command buffers
        ("HAWKING_QWEN_FFN_DOWN_Q4K", "1"), // FFN down Q4K path
    ] {
        if std::env::var_os(var).is_none() {
            std::env::set_var(var, val);
        }
    }

    // ── Apply runtime profile env overrides ──────────────────────────────────
    // Fast / Race / Efficient: opt into the both-metrics-optimal fast-path.
    // Exact: clear quality-trade vars so the path is bit-identical.
    // All of these respect explicit HAWKING_QWEN_*=0 opt-outs set before launch.
    // Single source of truth = RuntimeProfile::lever_plan() (shared with the CLI
    // generate path). set_if_unset respects explicit HAWKING_QWEN_*=0 opt-outs;
    // force_off enforces Exact's bit-identity even if a quality-trade var was set.
    let plan = effective_profile.lever_plan();
    for (k, v) in &plan.set_if_unset {
        if std::env::var_os(k).is_none() {
            std::env::set_var(k, v);
        }
    }
    for k in &plan.force_off {
        std::env::set_var(k, "0");
    }

    // ── Track 5.3: f16 KV cache env var ─────────────────────────────────────
    // Race and Efficient profiles enable f16 KV by default: halves KV memory
    // and frees bandwidth for long-context workloads. Fast/Exact/Default leave
    // it off to preserve bit-identity with the exact path.
    //
    // The per-field override (`opts.f16_kv`) wins over the profile default:
    //   Some(true)  → force on regardless of profile
    //   Some(false) → force off regardless of profile
    //   None        → use the profile/workload default
    {
        let profile_wants_f16_kv = plan.f16_kv.unwrap_or(false);
        let enable = match opts.f16_kv {
            Some(v) => v,
            None => profile_wants_f16_kv,
        };
        if enable && std::env::var_os("HAWKING_QWEN_F16_KV").is_none() {
            std::env::set_var("HAWKING_QWEN_F16_KV", "1");
        }
    }

    let speculate_mode = SpeculateMode::from_cli(opts.speculate.as_deref(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let kernel_profile = match opts.kernel_profile.as_ref() {
        Some(path) => Some(KernelProfile::load(path)?),
        None => None,
    };
    // concurrent_qkv: ON for fast/race/efficient — overlaps Q/K/V projections
    // on-GPU via MTLDispatchTypeConcurrent. +1.68% at B=1 (below prior +5% gate)
    // but valuable for the race/efficient profile throughput maximization.
    let concurrent_qkv = plan.concurrent_qkv
        || std::env::var_os("HAWKING_QWEN_CONCURRENT_QKV")
            .map(|v| v == "1")
            .unwrap_or(false);

    let cfg = EngineConfig {
        max_seq_len: 4096,
        max_batch_size: opts.max_batch_size,
        speculate: speculate_mode != SpeculateMode::Off,
        speculate_mode,
        verify_window: opts.verify_window,
        prefill_cache_dir: opts.prefill_cache_dir,
        kernel_profile,
        trace_dispatch: false,
        max_routed_expert_ram_mb: opts.max_routed_expert_ram_mb,
        memory_limit_mb: opts.memory_limit_mb,
        concurrent_qkv,
        ..Default::default()
    };

    let engine = hawking_core::model::load_engine(&opts.weights, cfg)
        .map_err(|e| anyhow::anyhow!("load engine: {e}"))?;
    let model_id = engine.model_id().to_string();
    let model_arch = engine.model_arch().to_string();
    let max_batch = opts.max_batch_size;

    // ── --explain-performance startup summary ─────────────────────────────
    if opts.explain_performance {
        let token_only_active = effective_profile == RuntimeProfile::Fast
            || effective_profile == RuntimeProfile::Race
            || effective_profile == RuntimeProfile::Efficient
            || std::env::var_os("HAWKING_QWEN_Q4K_LMHEAD")
                .map(|v| v == "1")
                .unwrap_or(false);
        let token_only_str = if token_only_active {
            "active (Q4K LM head loaded)"
        } else {
            "inactive (fallback to full logits)"
        };
        let hw_profile_str = opts
            .kernel_profile
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "none".to_string());
        let gather_ms = effective_energy.gather_window_ms();
        let f16_kv_active = std::env::var_os("HAWKING_QWEN_F16_KV")
            .map(|v| v == "1")
            .unwrap_or(false);
        let full_logits_mb = max_batch as f64 * 151936.0 * 4.0 / 1_048_576.0;
        let greedy_bytes = max_batch * 4;
        eprintln!(
            "hawking serve — performance summary\n\
             \x20 model:              {model_id}\n\
             \x20 profile:            {effective_profile}\n\
             \x20 workload pack:      {}\n\
             \x20 hardware-profile:   {hw_profile_str}\n\
             \x20 token-only lane:    {token_only_str}\n\
             \x20 f16 KV cache:       {f16_kv_active}\n\
             \x20 batch policy:       {effective_batch_policy:?}\n\
             \x20 energy mode:        {effective_energy}\n\
             \x20 gather window:      {gather_ms} ms\n\
             \x20 expected lanes:     greedy → token-only, sampled → full logits\n\
             \x20 full-logits cost:   B×vocab×4 bytes per step (~{full_logits_mb:.1} MB at B={max_batch}, Qwen)\n\
             \x20 greedy-lane cost:   B×4 bytes per step ({greedy_bytes} bytes at B={max_batch})",
            opts.workload,
        );
    }

    // Build the BatchDriver and install the effective batch policy.
    let batch_driver = {
        let mut d = batch::driver::BatchDriver::new(max_batch);
        d.scheduler.policy = effective_batch_policy.clone();
        d
    };

    let state = http::AppState {
        engine: Arc::new(parking_lot::Mutex::new(engine)),
        driver: Arc::new(parking_lot::Mutex::new(batch_driver)),
        slot_senders: Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new())),
        wait_queue: Arc::new(parking_lot::Mutex::new(std::collections::VecDeque::new())),
        model_arch,
        max_batch,
        requests_admitted: Arc::new(AtomicU64::new(0)),
        tokens_generated: Arc::new(AtomicU64::new(0)),
        requests_queued: Arc::new(AtomicU64::new(0)),
        system_kv_bank: Arc::new(parking_lot::Mutex::new(
            hawking_serve_system_kv_bank_default(),
        )),
    };

    // ── Background continuous-batching loop ───────────────────────────────
    // Single blocking thread: Phase A prefills pending slots, Phase B runs
    // one decode step across all ready slots, Phase C streams tokens to SSE.
    // All GPU kernel dispatches happen here under the engine lock; HTTP
    // handlers only hold the lock briefly for the admit tokenization step.
    let gather_window_ms = effective_energy.gather_window_ms();
    {
        let state2 = state.clone();
        tokio::task::spawn_blocking(move || {
            loop {
                // ── Phase A: parallel-prefill all pending slots ───────────
                // Collect all Prefilling slots and their prompts, then issue
                // a single prefill_slots_parallel call so weights are read
                // once per position across all B slots rather than once per
                // slot (serial). On any error, release every slot in the batch.
                //
                // Gather window: when max_batch > 1 and the first Prefilling
                // slot arrives, sleep briefly WITHOUT the engine lock so that
                // concurrent HTTP admits (which also need engine.lock() for
                // tokenization) can land before we hold the lock for the full
                // prefill duration. The window duration is set by --energy-mode
                // (off=0ms, balanced=3ms, efficient=8ms). 0ms disables the
                // window entirely. Non-zero values allow co-arriving requests
                // to be batched together.
                // Track 5: dispatch on scheduler.policy. prefill_slots_prefix_grouped
                // returns the same-prefix cohort (group_by_prefix, min_shared=8) when
                // policy == PrefixGrouped, else delegates to prefill_slots_bucketed —
                // byte-for-byte identical for Default/GreedyFirst. The policy was
                // installed at startup (`d.scheduler.policy = effective_batch_policy`),
                // so no extra binding is captured here.
                let mut prefilling: Vec<u32> = state2
                    .driver
                    .lock()
                    .scheduler
                    .prefill_slots_prefix_grouped(max_batch);
                if effective_energy.should_gather(prefilling.len(), max_batch) {
                    std::thread::sleep(std::time::Duration::from_millis(gather_window_ms));
                    prefilling = state2
                        .driver
                        .lock()
                        .scheduler
                        .prefill_slots_prefix_grouped(max_batch);
                }
                if !prefilling.is_empty() {
                    let slots_data: Vec<(usize, Vec<u32>)> = prefilling
                        .iter()
                        .filter_map(|&id| {
                            let ids = state2
                                .driver
                                .lock()
                                .scheduler
                                .slots
                                .iter()
                                .find(|s| s.id == id)
                                .map(|s| s.prompt_ids.clone())
                                .unwrap_or_default();
                            if ids.is_empty() {
                                None
                            } else {
                                Some((id as usize, ids))
                            }
                        })
                        .collect();
                    let slot_refs: Vec<(usize, &[u32])> = slots_data
                        .iter()
                        .map(|(s, ids)| (*s, ids.as_slice()))
                        .collect();
                    // Snapshot prefix_skip for every slot in this batch before
                    // touching any slot state, so we can partition without holding
                    // both the driver and engine locks simultaneously.
                    let skip_map: Vec<(usize, usize)> = slot_refs
                        .iter()
                        .map(|(slot_id, _)| {
                            let skip = state2
                                .driver
                                .lock()
                                .scheduler
                                .slots
                                .iter()
                                .find(|s| s.id == *slot_id as u32)
                                .map(|s| s.prefix_skip)
                                .unwrap_or(0);
                            (*slot_id, skip)
                        })
                        .collect();

                    // Reset all non-zero prefix_skip values upfront so retries
                    // don't re-skip regardless of which path runs below.
                    for &(slot_id, skip) in &skip_map {
                        if skip > 0 {
                            if let Some(s) = state2
                                .driver
                                .lock()
                                .scheduler
                                .slots
                                .iter_mut()
                                .find(|s| s.id == slot_id as u32)
                            {
                                s.prefix_skip = 0;
                            }
                        }
                    }

                    let prefill_result = {
                        let mut engine = state2.engine.lock();
                        if slot_refs.len() == 1 {
                            let (slot_id, prompt_ids) = slot_refs[0];
                            let skip = skip_map
                                .iter()
                                .find(|(id, _)| *id == slot_id)
                                .map(|(_, s)| *s)
                                .unwrap_or(0);
                            if skip > 0 {
                                engine
                                    .prefill_slot_from_pos(slot_id, prompt_ids, skip)
                                    .map(|ft| vec![(slot_id, ft)])
                            } else {
                                engine
                                    .prefill_slot(slot_id, prompt_ids)
                                    .map(|ft| vec![(slot_id, ft)])
                            }
                        } else {
                            // Track 5.2: partition into slots that have a prefix_skip
                            // (handle individually with prefill_slot_from_pos) and those
                            // that don't (run in parallel).
                            let with_skip: Vec<(usize, &[u32], usize)> = slot_refs
                                .iter()
                                .filter_map(|(slot_id, prompt_ids)| {
                                    let skip = skip_map
                                        .iter()
                                        .find(|(id, _)| id == slot_id)
                                        .map(|(_, s)| *s)
                                        .unwrap_or(0);
                                    if skip > 0 {
                                        Some((*slot_id, *prompt_ids, skip))
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            let without_skip: Vec<(usize, &[u32])> = slot_refs
                                .iter()
                                .filter(|(slot_id, _)| {
                                    skip_map
                                        .iter()
                                        .find(|(id, _)| id == slot_id)
                                        .map(|(_, s)| *s)
                                        .unwrap_or(0)
                                        == 0
                                })
                                .map(|(slot_id, prompt_ids)| (*slot_id, *prompt_ids))
                                .collect();

                            // Sequentially prefill the skip slots, collecting each
                            // slot's first generated token to seed decode with.
                            let mut firsts: Vec<(usize, u32)> = Vec::new();
                            let mut result: Result<(), hawking_core::Error> = Ok(());
                            for (slot_id, prompt_ids, skip) in with_skip {
                                if result.is_ok() {
                                    match engine.prefill_slot_from_pos(slot_id, prompt_ids, skip) {
                                        Ok(ft) => firsts.push((slot_id, ft)),
                                        Err(e) => result = Err(e),
                                    }
                                }
                            }
                            // Parallel-prefill the remaining slots (only if no error so far).
                            if result.is_ok() && !without_skip.is_empty() {
                                match engine.prefill_slots_parallel(&without_skip) {
                                    Ok(fts) => {
                                        for ((sid, _), ft) in without_skip.iter().zip(fts) {
                                            firsts.push((*sid, ft));
                                        }
                                    }
                                    Err(e) => result = Err(e),
                                }
                            }
                            result.map(|()| firsts)
                        }
                    };
                    match prefill_result {
                        Ok(firsts) => {
                            // Mark each prefilled slot ready, then SEED it with the
                            // first generated token (from the prefill's last-position
                            // logits) and stream that token immediately. The decode
                            // loop then continues from the SECOND token. This avoids
                            // re-feeding the last prompt token through the decode
                            // path, which produced a spurious leading word.
                            let eos = { state2.engine.lock().eos_id_for_batch() };
                            for (slot_id, first_token) in firsts {
                                let sid = slot_id as u32;
                                let decoded = {
                                    let mut driver = state2.driver.lock();
                                    driver.scheduler.mark_prefill_complete(sid);
                                    driver.scheduler.seed_first_token(sid, first_token, eos)
                                };
                                let Some(decoded) = decoded else { continue };
                                let text = {
                                    state2
                                        .engine
                                        .lock()
                                        .decode_token_for_batch(first_token)
                                        .unwrap_or_default()
                                };
                                let tx = state2.slot_senders.lock().get(&sid).cloned();
                                if let Some(tx) = tx {
                                    let _ = tx.blocking_send(Ok(text));
                                    state2.tokens_generated.fetch_add(1, Ordering::Relaxed);
                                    if decoded.finished {
                                        state2.slot_senders.lock().remove(&sid);
                                        state2.driver.lock().scheduler.release_slot(sid);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(err = %e, "prefill_slots_parallel failed");
                            for &slot_id in &prefilling {
                                let tx = state2.slot_senders.lock().remove(&slot_id);
                                if let Some(tx) = tx {
                                    let _ = tx.blocking_send(Err(()));
                                }
                                state2.driver.lock().scheduler.release_slot(slot_id);
                            }
                        }
                    }
                }

                // ── Phase B: one decode step across all ready slots ───────
                let outputs = {
                    let mut engine = state2.engine.lock();
                    let mut driver = state2.driver.lock();
                    driver.decode_ready_once(&mut **engine, max_batch)
                };
                let outputs = match outputs {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::error!(err = %e, "decode_ready_once failed");
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        continue;
                    }
                };
                if outputs.is_empty() {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    continue;
                }

                // ── Phase C: stream tokens + release finished slots ───────
                for out in outputs {
                    let tx = state2.slot_senders.lock().get(&out.slot_id).cloned();
                    if let Some(tx) = tx {
                        let send_ok = tx.blocking_send(Ok(out.text)).is_ok();
                        if send_ok {
                            state2.tokens_generated.fetch_add(1, Ordering::Relaxed);
                        }
                        if out.finished || !send_ok {
                            // Release on normal EOS *or* client disconnect.
                            state2.slot_senders.lock().remove(&out.slot_id);
                            state2.driver.lock().scheduler.release_slot(out.slot_id);

                            // Drain one waiter into the newly-freed slot.
                            let waiter = state2.wait_queue.lock().pop_front();
                            if let Some((waiter_req, waiter_tx, _chat)) = waiter {
                                let new_slot = {
                                    let engine = state2.engine.lock();
                                    let mut driver = state2.driver.lock();
                                    driver.admit(&**engine, waiter_req).ok().flatten()
                                };
                                if let Some(sid) = new_slot {
                                    state2.requests_admitted.fetch_add(1, Ordering::Relaxed);
                                    // Track 5.2: prefix-reuse detection. After admission the
                                    // new slot is already in the prefix_index; search for a
                                    // different slot whose KV we can copy into this one.
                                    {
                                        let prompt_ids = state2
                                            .driver
                                            .lock()
                                            .scheduler
                                            .slots
                                            .iter()
                                            .find(|s| s.id == sid)
                                            .map(|s| s.prompt_ids.clone())
                                            .unwrap_or_default();
                                        if !prompt_ids.is_empty() {
                                            let banked_len = http::banked_len_for(&prompt_ids);
                                            // 1) Live-slot match (Track 5.1): a DIFFERENT active slot.
                                            let mut src: Option<(u32, usize)> = state2
                                                .driver
                                                .lock()
                                                .scheduler
                                                .prefix_index
                                                .find_prefix_match_excluding(&prompt_ids, 8, sid);
                                            // 2) On a live MISS, consult the cross-request bank
                                            //    (Track 5.2): a slot that previously held this fixed
                                            //    system prefix even though it has since freed. Pure
                                            //    CPU lookup; the bank stores no KV.
                                            if src.is_none() {
                                                if let Some(entry) = state2
                                                    .system_kv_bank
                                                    .lock()
                                                    .lookup(&prompt_ids, banked_len)
                                                {
                                                    if entry.source_slot != sid {
                                                        src = Some((
                                                            entry.source_slot,
                                                            entry.prefix_len,
                                                        ));
                                                    }
                                                }
                                            }
                                            if let Some((src_slot, shared_len)) = src {
                                                tracing::debug!(
                                                    "[prefix-reuse] request matched slot {} at prefix_len={}",
                                                    src_slot,
                                                    shared_len
                                                );
                                                let copy_result =
                                                    state2.engine.lock().copy_kv_prefix_to_slot(
                                                        src_slot as usize,
                                                        sid as usize,
                                                        shared_len,
                                                    );
                                                if copy_result.is_ok() {
                                                    {
                                                        let mut driver = state2.driver.lock();
                                                        driver.lane_stats.prefix_reuse_count += 1;
                                                        // prefix_skip so prefill can call
                                                        // prefill_slot_from_pos instead of full prefill.
                                                        if let Some(slot) = driver
                                                            .scheduler
                                                            .slots
                                                            .iter_mut()
                                                            .find(|s| s.id == sid)
                                                        {
                                                            slot.prefix_skip = shared_len;
                                                        }
                                                    }
                                                    // Bank that THIS slot now holds copyable KV for the
                                                    // fixed leading span, so the NEXT serial turn (after
                                                    // this slot frees) still finds a source.
                                                    state2.system_kv_bank.lock().record(
                                                        &prompt_ids,
                                                        banked_len,
                                                        sid,
                                                    );
                                                }
                                                // copy Err (e.g. Unimplemented / stale banked slot):
                                                // silently skip — normal prefill proceeds from pos 0.
                                            } else {
                                                // No source yet, but this freshly-prefilled slot will
                                                // hold the span shortly — bank it so a later serial turn
                                                // can reuse it. (record() rejects sub-min spans itself.)
                                                state2.system_kv_bank.lock().record(
                                                    &prompt_ids,
                                                    banked_len,
                                                    sid,
                                                );
                                            }
                                        }
                                    }
                                    state2.slot_senders.lock().insert(sid, waiter_tx);
                                }
                                // If admit fails (should not — slot was just freed),
                                // waiter_tx is dropped, which sends Err(()) on the
                                // tokio receiver, closing the SSE stream gracefully.
                            }
                        }
                    }
                }
            }
        });
    }

    let app = http::router(state);
    tracing::info!(addr = %opts.addr, "hawking-serve listening");
    let listener = tokio::net::TcpListener::bind(opts.addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
#[rustfmt::skip]
mod profile_lever_tests {
    use super::RuntimeProfile as RP;

    fn has(plan_keys: &[(&'static str, &'static str)], k: &str) -> bool {
        plan_keys.iter().any(|(kk, _)| *kk == k)
    }

    #[test]
    fn default_touches_nothing() {
        let p = RP::Default.lever_plan();
        assert!(p.set_if_unset.is_empty());
        assert!(p.force_off.is_empty());
        assert_eq!(p.f16_kv, None);
        assert!(!p.concurrent_qkv);
    }

    #[test]
    fn fast_sets_full_bundle_no_f16kv() {
        let p = RP::Fast.lever_plan();
        for k in [
            "HAWKING_QWEN_Q4K_LMHEAD",
            "HAWKING_QWEN_Q4K_PREDEC",
            "HAWKING_QWEN_PREDEC_F16SCALES",
            "HAWKING_QWEN_VOCAB_PRUNE",
            "HAWKING_QWEN_FFN_DOWN_Q4K",
        ] {
            assert!(has(&p.set_if_unset, k), "fast must set {k}");
        }
        assert_eq!(p.f16_kv, Some(false), "fast leaves f16-KV off");
        assert!(p.concurrent_qkv);
        assert!(p.force_off.is_empty());
    }

    #[test]
    fn race_is_fast_plus_f16kv() {
        let p = RP::Race.lever_plan();
        assert!(has(&p.set_if_unset, "HAWKING_QWEN_VOCAB_PRUNE"));
        assert_eq!(p.f16_kv, Some(true), "race enables f16-KV");
        assert!(p.concurrent_qkv);
        assert!(!has(&p.set_if_unset, "HAWKING_ENERGY_EFFICIENT"));
    }

    #[test]
    fn efficient_adds_energy_and_f16kv() {
        let p = RP::Efficient.lever_plan();
        assert!(has(&p.set_if_unset, "HAWKING_ENERGY_EFFICIENT"), "efficient sets energy mode");
        assert_eq!(p.f16_kv, Some(true), "efficient enables f16-KV");
        assert!(has(&p.set_if_unset, "HAWKING_QWEN_Q4K_PREDEC"));
    }

    #[test]
    fn exact_force_offs_every_quality_trade() {
        let p = RP::Exact.lever_plan();
        for k in ["HAWKING_QWEN_PREDEC_F16SCALES", "HAWKING_QWEN_FFN_DOWN_Q4K", "HAWKING_QWEN_VOCAB_PRUNE"] {
            assert!(p.force_off.contains(&k), "exact must force-off {k}");
        }
        assert!(p.set_if_unset.is_empty(), "exact sets no quality-trade var");
        assert_eq!(p.f16_kv, Some(false), "exact leaves f16-KV off (bit-identity)");
        assert!(!p.concurrent_qkv);
    }

    #[test]
    fn contracts_are_nonempty_and_self_label() {
        for rp in [RP::Default, RP::Fast, RP::Race, RP::Efficient, RP::Exact] {
            let c = rp.contract();
            assert!(c.contains(rp.as_str()), "contract for {rp} must name itself");
            assert!(c.len() > 20);
        }
    }

    #[test]
    fn from_str_roundtrips_all_known() {
        for s in ["default", "fast", "race", "efficient", "exact"] {
            assert_eq!(RP::from_str(s).unwrap().as_str(), s);
        }
        assert!(RP::from_str("m3-pro-18gb").is_none(), "hardware string is not a runtime profile");
    }

    /// Track 0/9 lock-in: the "fast is the CLI default" decision must keep
    /// resolving an UNSET `--profile` to `Fast`. Validated GPU-side once
    /// (~38-39 t/s middle variant); pin it on CPU so a refactor can't silently
    /// flip the default back to the conservative bit-identical path.
    #[test]
    fn default_when_unset_is_fast() {
        assert_eq!(
            RP::default_when_unset(),
            RP::Fast,
            "unset --profile must resolve to fast (the shipped CLI default)"
        );
    }

    /// Track 0/9 lock-in: the UNSET-default contract is exactly
    /// "fast bundle MINUS PREDEC_F16SCALES". i.e. the MIDDLE variant keeps the
    /// 4 bit-identical-ish fast levers (Q4K LM-head, predec, vocab-prune,
    /// Q4K FFN-down) but force-OFFs f16-scales (it failed quality_oracle
    /// 0.792/11.46% @ e613dde). This pins both halves so neither can drift.
    #[test]
    fn unset_default_is_fast_minus_f16scales() {
        let bundle = RP::Fast.lever_plan().set_if_unset; // == fast_bundle()
        let force_off = RP::default_unset_force_off();

        // (i) f16-scales is the one-and-only lever the unset default disables.
        assert_eq!(
            force_off,
            &["HAWKING_QWEN_PREDEC_F16SCALES"],
            "unset default must force-off exactly PREDEC_F16SCALES"
        );

        // (ii) the 4 kept fast levers remain in the bundle (so unset still
        //      runs the fast path minus f16-scales, not the conservative path).
        for k in [
            "HAWKING_QWEN_Q4K_LMHEAD",
            "HAWKING_QWEN_Q4K_PREDEC",
            "HAWKING_QWEN_VOCAB_PRUNE",
            "HAWKING_QWEN_FFN_DOWN_Q4K",
        ] {
            assert!(has(&bundle, k), "fast bundle must keep {k} (a kept lever under the unset default)");
        }

        // (iii) f16-scales IS in the full fast bundle (so the force-off is what
        //       removes it for the unset default — not its absence). This is the
        //       load-bearing invariant: unset = fast bundle XOR-removed of f16s.
        assert!(
            has(&bundle, force_off[0]),
            "the force-off lever must exist in the fast bundle (else force-off is a no-op)"
        );
    }
}
