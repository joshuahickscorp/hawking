use hawking_core::{
    Engine, EngineConfig, GenStats, GenerateRequest, Result as CoreResult, SamplingParams,
    StreamEvent,
};
use hawking_serve::batch::driver::BatchDriver;
use std::path::Path;
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
#[test]
fn all_greedy_batch_routes_token_only_and_charges_b_times_4() {
    let mut driver = BatchDriver::new(4);
    seed_two_slots(&mut driver, greedy_req);
    let mut engine = StubEngine::new();
    let out = driver
        .decode_ready_once(&mut engine, 4)
        .expect("decode once");
    let toks: Vec<u32> = out.iter().map(|o| o.token).collect();
    assert_eq!(toks, vec![1, 2], "greedy lane argmax tokens");
    assert_eq!(driver.lane_stats.greedy_steps, 1, "one greedy step");
    assert_eq!(driver.lane_stats.logits_steps, 0, "no logits step");
    let b = 2u64;
    assert_eq!(
        driver.lane_stats.readback_bytes,
        b * std::mem::size_of::<u32>() as u64
    );
    assert_eq!(engine.forward_calls, 1);
}
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
        b * VOCAB as u64 * std::mem::size_of::<f32>() as u64
    );
    assert_eq!(engine.forward_calls, 1);
}
#[test]
fn one_sampling_slot_forces_full_logits_for_the_batch() {
    let mut driver = BatchDriver::new(4);
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
        b * VOCAB as u64 * std::mem::size_of::<f32>() as u64
    );
}
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
    assert_eq!(greedy_driver.lane_stats.greedy_steps, 1);
    assert_eq!(logits_driver.lane_stats.logits_steps, 1);
    let g: Vec<(u32, u32)> = greedy_out.iter().map(|o| (o.slot_id, o.token)).collect();
    let l: Vec<(u32, u32)> = logits_out.iter().map(|o| (o.slot_id, o.token)).collect();
    assert_eq!(
        g, l,
        "greedy lane and full-logits lane must yield same tokens"
    );
}
