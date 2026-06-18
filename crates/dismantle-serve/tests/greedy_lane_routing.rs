//! Confirmation test for the greedy token-only serving lane (continuous-batch
//! decode). Locks in three invariants that must not silently regress:
//!
//!   (a) LANE CLASSIFICATION: a batch where every slot is greedy (temp=0, no
//!       repetition penalty) routes through `forward_multiseq_greedy_tokens`
//!       (token-only), while any sampling / rep-penalty slot routes through the
//!       full-logits path. Verified via the `LaneStats.greedy_steps` /
//!       `logits_steps` counters the driver bumps per step.
//!
//!   (b) READBACK DROPS TO B×4: the greedy lane charges `B × size_of::<u32>()`
//!       (= B×4) readback bytes per step; the full-logits lane charges
//!       `B × vocab × size_of::<f32>()`. This is the bandwidth win the
//!       token-only lane exists for, and it is accounted in
//!       `LaneStats.readback_bytes`.
//!
//!   (c) TOKENS BIT-IDENTICAL: for the SAME logits, the token ids produced via
//!       the greedy lane equal those produced via the full-logits lane. (Both
//!       resolve to argmax; the fixtures use unique maxima so the two argmax
//!       implementations — engine `max_by` vs sampler strict `>` — agree.)
//!
//! Hermetic, model-free: a stub `Engine` returns fixed logits per token, mirror-
//! ing `dismantle-serve/src/batch/driver.rs`'s own `FakeEngine`. The default
//! `Engine::forward_multiseq_greedy_tokens` delegates to
//! `forward_multiseq_batched`, which delegates to `forward_tokens_for_test`,
//! which the stub implements — so the same stub drives both lanes. Gates:
//!
//!   cargo test -p dismantle-serve --test greedy_lane_routing

use dismantle_core::{
    Engine, EngineConfig, GenStats, GenerateRequest, Result as CoreResult, SamplingParams,
    StreamEvent,
};
use dismantle_serve::batch::driver::BatchDriver;
use std::path::Path;

/// Stub engine: token 10 → logits with unique argmax at index 1; token 20 →
/// unique argmax at index 2; anything else → unique argmax at index 0. Records
/// how many forward calls happened so we can assert exactly one batched call
/// per decode step. Vocab is 3 (len of each logit vector).
struct StubEngine {
    forward_calls: usize,
}

impl StubEngine {
    fn new() -> Self {
        Self { forward_calls: 0 }
    }
    fn logits_for(token: u32) -> Vec<f32> {
        match token {
            10 => vec![0.0, 4.0, 1.0], // argmax = 1
            20 => vec![0.0, 1.0, 5.0], // argmax = 2
            _ => vec![3.0, 0.0, 0.0],  // argmax = 0
        }
    }
}

impl Engine for StubEngine {
    fn load(_weights: &Path, _config: EngineConfig) -> CoreResult<Self>
    where
        Self: Sized,
    {
        Ok(Self::new())
    }

    fn generate(
        &mut self,
        _req: GenerateRequest,
        _sink: &mut dyn FnMut(StreamEvent),
    ) -> CoreResult<GenStats> {
        Ok(GenStats {
            completion_tokens: 0,
            ..Default::default()
        })
    }

    fn model_id(&self) -> &str {
        "stub-greedy"
    }

    fn encode_prompt_for_batch(&self, prompt: &str) -> CoreResult<Vec<u32>> {
        Ok(prompt.bytes().map(u32::from).collect())
    }

    fn decode_token_for_batch(&self, token: u32) -> CoreResult<String> {
        Ok(format!("<{token}>"))
    }

    fn eos_id_for_batch(&self) -> Option<u32> {
        // Keep EOS out of the way of our fixture argmax tokens {0,1,2} by
        // picking an id none of them produce, so slots don't finish early.
        Some(9999)
    }

    fn forward_tokens_for_test(
        &mut self,
        tokens: &[u32],
        _positions: &[usize],
    ) -> CoreResult<Vec<Vec<f32>>> {
        self.forward_calls += 1;
        Ok(tokens.iter().map(|&t| Self::logits_for(t)).collect())
    }
}

const VOCAB: usize = 3; // len of each stub logit vector

/// Greedy request: temperature 0, no repetition penalty → greedy lane.
fn greedy_req(max_new_tokens: usize) -> GenerateRequest {
    GenerateRequest {
        prompt: "x".into(),
        max_new_tokens,
        sampling: SamplingParams {
            temperature: 0.0,
            repetition_penalty: 1.0,
            ..SamplingParams::default()
        },
        stop: Vec::new(),
        abort: None,
        max_stall_ms: 0,
        json_mode: false,
    }
}

/// Logits-lane request: temperature 0 (so the sampler still picks argmax, giving
/// us a deterministic, bit-identical comparison) but repetition_penalty > 1.0 so
/// the lane predicate (`temp<=0 && rep<=1`) is FALSE → full-logits path. On the
/// first decode step the rep-penalty history is empty, so logits are unperturbed
/// and the sampler's temp=0 branch returns plain argmax.
fn logits_lane_req(max_new_tokens: usize) -> GenerateRequest {
    GenerateRequest {
        prompt: "x".into(),
        max_new_tokens,
        sampling: SamplingParams {
            temperature: 0.0,
            repetition_penalty: 1.5,
            ..SamplingParams::default()
        },
        stop: Vec::new(),
        abort: None,
        max_stall_ms: 0,
        json_mode: false,
    }
}

/// Admit two slots seeded with last_token 10 and 20, mark them ready to decode.
fn seed_two_slots(driver: &mut BatchDriver, req_fn: impl Fn(usize) -> GenerateRequest) {
    for (id, token) in [(0u32, 10u32), (1u32, 20u32)] {
        let slot_id = driver
            .scheduler
            .admit(req_fn(4), vec![token])
            .expect("admit");
        assert_eq!(slot_id, id);
        assert!(driver.scheduler.mark_prefill_complete(slot_id));
    }
}

/// (a)+(b): an all-greedy batch routes through the token-only lane and charges
/// B×4 readback bytes.
#[test]
fn all_greedy_batch_routes_token_only_and_charges_b_times_4() {
    let mut driver = BatchDriver::new(4);
    seed_two_slots(&mut driver, greedy_req);
    let mut engine = StubEngine::new();

    let out = driver
        .decode_ready_once(&mut engine, 4)
        .expect("decode once");

    // Tokens are the per-slot argmax: slot0 (from token 10) → 1, slot1 (20) → 2.
    let toks: Vec<u32> = out.iter().map(|o| o.token).collect();
    assert_eq!(toks, vec![1, 2], "greedy lane argmax tokens");

    // Lane classification: greedy path taken, logits path not.
    assert_eq!(driver.lane_stats.greedy_steps, 1, "one greedy step");
    assert_eq!(driver.lane_stats.logits_steps, 0, "no logits step");

    // Readback dropped to B×4 (B=2 slots → 8 bytes), NOT B×vocab×4.
    let b = 2u64;
    assert_eq!(
        driver.lane_stats.readback_bytes,
        b * std::mem::size_of::<u32>() as u64,
        "greedy lane charges exactly B×4 readback bytes"
    );
    // Exactly one batched forward call serviced the whole step.
    assert_eq!(engine.forward_calls, 1);
}

/// (a)+(b): a batch containing a rep-penalty (non-greedy) slot routes through
/// the full-logits lane and charges B×vocab×4 readback bytes.
#[test]
fn rep_penalty_batch_routes_full_logits_and_charges_b_times_vocab_times_4() {
    let mut driver = BatchDriver::new(4);
    seed_two_slots(&mut driver, logits_lane_req);
    let mut engine = StubEngine::new();

    let out = driver
        .decode_ready_once(&mut engine, 4)
        .expect("decode once");

    let toks: Vec<u32> = out.iter().map(|o| o.token).collect();
    assert_eq!(
        toks,
        vec![1, 2],
        "logits lane argmax tokens (temp=0 → argmax)"
    );

    assert_eq!(driver.lane_stats.greedy_steps, 0, "no greedy step");
    assert_eq!(driver.lane_stats.logits_steps, 1, "one logits step");

    let b = 2u64;
    assert_eq!(
        driver.lane_stats.readback_bytes,
        b * VOCAB as u64 * std::mem::size_of::<f32>() as u64,
        "full-logits lane charges B×vocab×4 readback bytes"
    );
    assert_eq!(engine.forward_calls, 1);
}

/// A single non-greedy slot is enough to force the whole batch onto the
/// full-logits lane (the `all_greedy` predicate is an AND across slots).
#[test]
fn one_sampling_slot_forces_full_logits_for_the_batch() {
    let mut driver = BatchDriver::new(4);
    // slot0 greedy, slot1 rep-penalty → batch is NOT all-greedy.
    let g = driver
        .scheduler
        .admit(greedy_req(4), vec![10])
        .expect("admit g");
    let s = driver
        .scheduler
        .admit(logits_lane_req(4), vec![20])
        .expect("admit s");
    assert!(driver.scheduler.mark_prefill_complete(g));
    assert!(driver.scheduler.mark_prefill_complete(s));
    let mut engine = StubEngine::new();

    let _ = driver
        .decode_ready_once(&mut engine, 4)
        .expect("decode once");

    assert_eq!(driver.lane_stats.greedy_steps, 0);
    assert_eq!(
        driver.lane_stats.logits_steps, 1,
        "mixed batch → full logits"
    );
    let b = 2u64;
    assert_eq!(
        driver.lane_stats.readback_bytes,
        b * VOCAB as u64 * std::mem::size_of::<f32>() as u64,
    );
}

/// (c) TOKENS BIT-IDENTICAL across lanes: same logits → same token ids whether
/// routed greedy (token-only argmax) or full-logits (sampler temp=0 argmax).
/// Drives two independent drivers with identical slot seeds, differing only in
/// the lane predicate.
#[test]
fn greedy_and_logits_lanes_produce_identical_tokens() {
    let mut greedy_driver = BatchDriver::new(4);
    seed_two_slots(&mut greedy_driver, greedy_req);
    let mut e1 = StubEngine::new();
    let greedy_out = greedy_driver
        .decode_ready_once(&mut e1, 4)
        .expect("greedy decode");

    let mut logits_driver = BatchDriver::new(4);
    seed_two_slots(&mut logits_driver, logits_lane_req);
    let mut e2 = StubEngine::new();
    let logits_out = logits_driver
        .decode_ready_once(&mut e2, 4)
        .expect("logits decode");

    // Confirm they actually took different lanes...
    assert_eq!(greedy_driver.lane_stats.greedy_steps, 1);
    assert_eq!(logits_driver.lane_stats.logits_steps, 1);

    // ...yet produced bit-identical token ids per slot.
    let g: Vec<(u32, u32)> = greedy_out.iter().map(|o| (o.slot_id, o.token)).collect();
    let l: Vec<(u32, u32)> = logits_out.iter().map(|o| (o.slot_id, o.token)).collect();
    assert_eq!(
        g, l,
        "greedy lane and full-logits lane must yield same tokens"
    );
}
