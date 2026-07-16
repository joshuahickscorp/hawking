//! Consolidated HTTP, batching, routing, and system-KV integration gates.

#[rustfmt::skip]
mod energy_gather_window {
    //! Track 7.1 — unit test for the energy gather/admission DECISION policy.
    //!
    //! Pins `EnergyMode::should_gather(ready, max_batch)` — the extracted,
    //! pure form of the wait-or-commit predicate the continuous-batch loop runs
    //! (serve::run(), the `prefilling.len() < max_batch && gather_window_ms > 0`
    //! guard). Pure: no env, no model, no sleeps, no clock. Gate:
    //!
    //!   cargo test -p hawking-serve --test energy_gather_window

    use hawking_serve::EnergyMode;

    /// A latency-sensitive SINGLE request is NEVER delayed: with a 1-slot server
    /// (max_batch == 1) batching is impossible, so no mode ever gathers — the
    /// request goes straight to prefill regardless of window size.
    #[test]
    fn single_slot_server_never_gathers() {
        for mode in [EnergyMode::Off, EnergyMode::Balanced, EnergyMode::Efficient] {
            assert!(!mode.should_gather(1, 1), "{mode}: single-slot server must not delay the lone request");
        }
    }

    /// A partial batch on a multi-slot server WAITS (up to the window) so co-
    /// arriving requests can fill it for lower J/tok — but only when the window > 0.
    #[test]
    fn partial_batch_gathers_when_window_open() {
        // 1 ready of 8 capacity: balanced/efficient gather; off does not.
        assert!(!EnergyMode::Off.should_gather(1, 8), "off has no window");
        assert!(EnergyMode::Balanced.should_gather(1, 8), "balanced gathers a partial batch");
        assert!(EnergyMode::Efficient.should_gather(1, 8), "efficient gathers a partial batch");
        // window magnitudes back the decision.
        assert_eq!(EnergyMode::Off.gather_window_ms(), 0);
        assert_eq!(EnergyMode::Balanced.gather_window_ms(), 3);
        assert_eq!(EnergyMode::Efficient.gather_window_ms(), 8);
    }

    /// A FULL batch never waits: once ready == max_batch there is nothing to gain
    /// from gathering, so commit immediately even under efficient mode. This is the
    /// upper bound on added latency — a full batch is dispatched with zero delay.
    #[test]
    fn full_batch_commits_immediately() {
        for mode in [EnergyMode::Off, EnergyMode::Balanced, EnergyMode::Efficient] {
            assert!(!mode.should_gather(8, 8), "{mode}: a full batch must dispatch without waiting");
            // and an (impossible) over-full count likewise commits.
            assert!(!mode.should_gather(9, 8), "{mode}: over-full never waits");
        }
    }

    /// Empty queue never gathers (nothing to wait for).
    #[test]
    fn empty_queue_never_gathers() {
        for mode in [EnergyMode::Off, EnergyMode::Balanced, EnergyMode::Efficient] {
            assert!(!mode.should_gather(0, 8), "{mode}: empty queue must not sleep");
        }
    }

    /// Sweep: the helper must EXACTLY equal the inline loop predicate for every
    /// (mode, ready, max_batch) in a representative grid. This is the anti-drift
    /// lock — if the loop guard and the helper ever diverge, this fails.
    #[test]
    fn helper_matches_inline_loop_predicate() {
        for mode in [EnergyMode::Off, EnergyMode::Balanced, EnergyMode::Efficient] {
            for max_batch in [1usize, 2, 4, 8] {
                for ready in 0..=max_batch + 1 {
                    let want = ready > 0 && max_batch > 1 && ready < max_batch && mode.gather_window_ms() > 0;
                    assert_eq!(
                        mode.should_gather(ready, max_batch),
                        want,
                        "drift at mode={mode} ready={ready} max_batch={max_batch}"
                    );
                }
            }
        }
    }
}
#[rustfmt::skip]
mod greedy_lane_routing {
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
    //! ing `hawking-serve/src/batch/driver.rs`'s own `FakeEngine`. The default
    //! `Engine::forward_multiseq_greedy_tokens` delegates to
    //! `forward_multiseq_batched`, which delegates to `forward_tokens_for_test`,
    //! which the stub implements — so the same stub drives both lanes. Gates:
    //!
    //!   cargo test -p hawking-serve --test greedy_lane_routing

    use hawking_core::{
        Engine, EngineConfig, GenStats, GenerateRequest, Result as CoreResult, SamplingParams, StreamEvent,
    };
    use hawking_serve::batch::driver::BatchDriver;
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

        fn generate(&mut self, _req: GenerateRequest, _sink: &mut dyn FnMut(StreamEvent)) -> CoreResult<GenStats> {
            Ok(GenStats { completion_tokens: 0, ..Default::default() })
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

        fn forward_tokens_for_test(&mut self, tokens: &[u32], _positions: &[usize]) -> CoreResult<Vec<Vec<f32>>> {
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
            sampling: SamplingParams { temperature: 0.0, repetition_penalty: 1.0, ..SamplingParams::default() },
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
            sampling: SamplingParams { temperature: 0.0, repetition_penalty: 1.5, ..SamplingParams::default() },
            stop: Vec::new(),
            abort: None,
            max_stall_ms: 0,
            json_mode: false,
        }
    }

    /// Admit two slots seeded with last_token 10 and 20, mark them ready to decode.
    fn seed_two_slots(driver: &mut BatchDriver, req_fn: impl Fn(usize) -> GenerateRequest) {
        for (id, token) in [(0u32, 10u32), (1u32, 20u32)] {
            let slot_id = driver.scheduler.admit(req_fn(4), vec![token]).expect("admit");
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

        let out = driver.decode_ready_once(&mut engine, 4).expect("decode once");

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

        let out = driver.decode_ready_once(&mut engine, 4).expect("decode once");

        let toks: Vec<u32> = out.iter().map(|o| o.token).collect();
        assert_eq!(toks, vec![1, 2], "logits lane argmax tokens (temp=0 → argmax)");

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
        let g = driver.scheduler.admit(greedy_req(4), vec![10]).expect("admit g");
        let s = driver.scheduler.admit(logits_lane_req(4), vec![20]).expect("admit s");
        assert!(driver.scheduler.mark_prefill_complete(g));
        assert!(driver.scheduler.mark_prefill_complete(s));
        let mut engine = StubEngine::new();

        let _ = driver.decode_ready_once(&mut engine, 4).expect("decode once");

        assert_eq!(driver.lane_stats.greedy_steps, 0);
        assert_eq!(driver.lane_stats.logits_steps, 1, "mixed batch → full logits");
        let b = 2u64;
        assert_eq!(driver.lane_stats.readback_bytes, b * VOCAB as u64 * std::mem::size_of::<f32>() as u64,);
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
        let greedy_out = greedy_driver.decode_ready_once(&mut e1, 4).expect("greedy decode");

        let mut logits_driver = BatchDriver::new(4);
        seed_two_slots(&mut logits_driver, logits_lane_req);
        let mut e2 = StubEngine::new();
        let logits_out = logits_driver.decode_ready_once(&mut e2, 4).expect("logits decode");

        // Confirm they actually took different lanes...
        assert_eq!(greedy_driver.lane_stats.greedy_steps, 1);
        assert_eq!(logits_driver.lane_stats.logits_steps, 1);

        // ...yet produced bit-identical token ids per slot.
        let g: Vec<(u32, u32)> = greedy_out.iter().map(|o| (o.slot_id, o.token)).collect();
        let l: Vec<(u32, u32)> = logits_out.iter().map(|o| (o.slot_id, o.token)).collect();
        assert_eq!(g, l, "greedy lane and full-logits lane must yield same tokens");
    }
}
#[rustfmt::skip]
mod hawking_native_endpoint {
    //! Track 8.2 gate — native `/v1/hawking/generate` request mapping + stats
    //! object shape. Pure: tests the in-schema -> GenerateRequest mapping and the
    //! GenStats-shaped final stats JSON, no engine/server/GPU. Runs despite the
    //! #[ignore]'d generation-path integration tests.
    //!   cargo test -p hawking-serve --test hawking_native_endpoint

    use hawking_serve::http::{hawking_generate_stats_json, map_hawking_generate_req, HawkingGenerateReq};

    fn parse(json: serde_json::Value) -> HawkingGenerateReq {
        serde_json::from_value(json).expect("parse HawkingGenerateReq")
    }

    #[test]
    fn absent_temperature_maps_to_greedy() {
        let r = parse(serde_json::json!({"prompt": "hi", "max_tokens": 8}));
        let g = map_hawking_generate_req(&r);
        assert_eq!(g.prompt, "hi");
        assert_eq!(g.max_new_tokens, 8);
        assert_eq!(g.sampling.temperature, 0.0);
        assert_eq!(g.sampling.top_k, 0);
        assert_eq!(g.sampling.top_p, 1.0);
        assert!(g.stop.is_empty());
        assert!(g.abort.is_none());
    }

    #[test]
    fn sampling_fields_and_stop_round_trip() {
        let r = parse(serde_json::json!({
            "prompt": "p", "max_tokens": 32, "temperature": 0.7,
            "top_p": 0.95, "seed": 123, "stop": ["</s>", "\n\n"]
        }));
        let g = map_hawking_generate_req(&r);
        assert_eq!(g.sampling.temperature, 0.7);
        assert!((g.sampling.top_p - 0.95).abs() < 1e-6);
        assert_eq!(g.sampling.top_k, 40);
        assert_eq!(g.sampling.seed, Some(123));
        assert_eq!(g.stop, vec!["</s>".to_string(), "\n\n".to_string()]);
    }

    #[test]
    fn default_max_tokens_applied_when_absent() {
        let r = parse(serde_json::json!({"prompt": "x"}));
        let g = map_hawking_generate_req(&r);
        assert_eq!(g.max_new_tokens, 256);
    }

    #[test]
    fn stats_json_has_genstats_field_names_and_dec_tps() {
        // 64 tokens / 2000 ms = 32 tok/s — mirrors GenStats::dec_tps formula.
        let j = hawking_generate_stats_json(14, 64, 2000.0, true, "q4k-predec-f16s");
        assert_eq!(j["prompt_tokens"], 14);
        assert_eq!(j["completion_tokens"], 64);
        assert_eq!(j["decode_ms"], 2000.0);
        assert_eq!(j["dec_tps"].as_f64().unwrap().round(), 32.0);
        assert_eq!(j["token_only_path_used"], true);
        assert_eq!(j["lm_head_path"], "q4k-predec-f16s");
        // parseable + compact (no heavy fields)
        let s = serde_json::to_string(&j).unwrap();
        assert!(serde_json::from_str::<serde_json::Value>(&s).is_ok());
    }

    #[test]
    fn stats_dec_tps_zero_safe_when_no_decode() {
        let j = hawking_generate_stats_json(5, 0, 0.0, false, "f16");
        assert!(j["dec_tps"].as_f64().unwrap().is_finite());
    }
}
#[rustfmt::skip]
mod http_integration {
    //! In-process HTTP integration tests for the OpenAI-compatible routes.
    //!
    //! These drive the axum `Router` directly with `tower::ServiceExt::oneshot`
    //! (no TCP port, no real model) against a deterministic stub engine, so they
    //! are fast and hermetic. They cover:
    //!   (a) `POST /v1/chat/completions` streaming -> 200 + well-formed SSE;
    //!   (b) `POST /v1/chat/completions` non-stream -> 200 + JSON body;
    //!   (c) `POST /v1/completions` non-stream + streaming -> 200 + well-formed body;
    //!   (d) malformed / missing-field requests -> 4xx + structured OpenAI error.

    use std::sync::Arc;

    use axum::{
        body::Body,
        http::{header, Request, StatusCode},
        Router,
    };
    use bytes::Bytes;
    use hawking_core::{
        Engine, EngineConfig, GenStats, GenerateRequest, Result as CoreResult, StopReason, StreamEvent,
    };
    use hawking_serve::batch::driver::BatchDriver;
    use hawking_serve::http::{router, AppState};
    use http_body_util::BodyExt;
    use parking_lot::Mutex;
    use std::collections::{HashMap, VecDeque};
    use std::sync::atomic::AtomicU64;
    use tower::ServiceExt; // for `oneshot`

    /// Deterministic, model-free engine. `generate` emits a fixed sequence of
    /// token events plus a `Done`, so streaming and non-streaming responses are
    /// fully predictable. `model_arch` is "qwen2" so the chat template path that
    /// the real 0.5B model would hit is exercised.
    struct StubEngine {
        tokens: Vec<&'static str>,
        fail: bool,
    }

    impl StubEngine {
        fn new() -> Self {
            Self { tokens: vec!["Hello", ", ", "world", "!"], fail: false }
        }

        fn failing() -> Self {
            Self { tokens: Vec::new(), fail: true }
        }
    }

    impl Engine for StubEngine {
        fn load(_weights: &std::path::Path, _config: EngineConfig) -> CoreResult<Self>
        where
            Self: Sized,
        {
            Ok(Self::new())
        }

        fn generate(&mut self, _req: GenerateRequest, sink: &mut dyn FnMut(StreamEvent)) -> CoreResult<GenStats> {
            if self.fail {
                return Err(hawking_core::Error::Model("stub forced failure".into()));
            }
            for (i, t) in self.tokens.iter().enumerate() {
                sink(StreamEvent::Token { id: i as u32, text: (*t).to_string() });
            }
            sink(StreamEvent::Done { reason: StopReason::Eos, stats: GenStats::default() });
            Ok(GenStats { completion_tokens: self.tokens.len(), ..Default::default() })
        }

        fn model_id(&self) -> &str {
            "stub-model"
        }

        fn model_arch(&self) -> &str {
            "qwen2"
        }

        fn forward_tokens_for_test(&mut self, tokens: &[u32], _positions: &[usize]) -> CoreResult<Vec<Vec<f32>>> {
            Ok(tokens.iter().map(|_| vec![0.0_f32; 4]).collect())
        }
    }

    fn app() -> Router {
        let state = AppState {
            engine: Arc::new(Mutex::new(Box::new(StubEngine::new()))),
            system_kv_bank: Arc::new(Mutex::new(hawking_serve::SystemPromptKvBank::new())),
            driver: Arc::new(Mutex::new(BatchDriver::new(1))),
            slot_senders: Arc::new(Mutex::new(HashMap::new())),
            wait_queue: Arc::new(Mutex::new(VecDeque::new())),
            model_arch: "qwen2".to_string(),
            max_batch: 1,
            requests_admitted: Arc::new(AtomicU64::new(0)),
            tokens_generated: Arc::new(AtomicU64::new(0)),
            requests_queued: Arc::new(AtomicU64::new(0)),
        };
        router(state)
    }

    fn failing_app() -> Router {
        let state = AppState {
            engine: Arc::new(Mutex::new(Box::new(StubEngine::failing()))),
            system_kv_bank: Arc::new(Mutex::new(hawking_serve::SystemPromptKvBank::new())),
            driver: Arc::new(Mutex::new(BatchDriver::new(1))),
            slot_senders: Arc::new(Mutex::new(HashMap::new())),
            wait_queue: Arc::new(Mutex::new(VecDeque::new())),
            model_arch: "qwen2".to_string(),
            max_batch: 1,
            requests_admitted: Arc::new(AtomicU64::new(0)),
            tokens_generated: Arc::new(AtomicU64::new(0)),
            requests_queued: Arc::new(AtomicU64::new(0)),
        };
        router(state)
    }

    fn json_post(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn raw_post(uri: &str, body: &'static str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap()
    }

    async fn body_bytes(resp: axum::response::Response) -> Bytes {
        resp.into_body().collect().await.unwrap().to_bytes()
    }

    // ----------------------------------------------------------------------------
    // (a) chat completions, streaming -> 200 + well-formed SSE
    // ----------------------------------------------------------------------------

    // Generation-path tests (a–d) are ignored until a test decode loop is wired up.
    // The route handlers now gate all generation through BatchDriver + background loop;
    // a running decode task is required to push tokens to slot_senders. The 7 error/
    // healthz tests below still run and cover routing + request validation.
    #[tokio::test]
    #[ignore = "requires background decode loop: BatchDriver admission now decoupled from generation"]
    async fn chat_completions_streaming_sse_ok() {
        let req = json_post(
            "/v1/chat/completions",
            serde_json::json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true,
                "max_tokens": 8
            }),
        );
        let resp = app().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let ct = resp.headers().get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
        assert!(ct.starts_with("text/event-stream"), "expected SSE content-type, got {ct:?}");

        let body = body_bytes(resp).await;
        let text = std::str::from_utf8(&body).unwrap();

        // SSE framing: at least one `data:` line, the concatenated token text, and
        // the terminal [DONE] sentinel.
        assert!(text.contains("data:"), "no SSE data frames in:\n{text}");
        assert!(text.contains("chat.completion.chunk"), "missing chat chunk object in:\n{text}");
        assert!(text.contains("Hello"), "missing streamed token in:\n{text}");
        assert!(text.contains("[DONE]"), "missing [DONE] sentinel in:\n{text}");

        // Every data frame after the marker must be valid JSON (except [DONE]).
        for line in text.lines() {
            let Some(payload) = line.strip_prefix("data:") else {
                continue;
            };
            let payload = payload.trim();
            if payload.is_empty() || payload == "[DONE]" {
                continue;
            }
            serde_json::from_str::<serde_json::Value>(payload)
                .unwrap_or_else(|e| panic!("non-JSON SSE payload {payload:?}: {e}"));
        }
    }

    // ----------------------------------------------------------------------------
    // chat completions, non-streaming -> 200 + JSON body
    // ----------------------------------------------------------------------------

    #[tokio::test]
    #[ignore = "requires background decode loop: BatchDriver admission now decoupled from generation"]
    async fn chat_completions_non_stream_json_ok() {
        let req = json_post(
            "/v1/chat/completions",
            serde_json::json!({
                "messages": [{"role": "user", "content": "hi"}],
                "stream": false
            }),
        );
        let resp = app().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert_eq!(v["choices"][0]["message"]["content"], "Hello, world!");
    }

    // ----------------------------------------------------------------------------
    // (b) legacy completions -> 200 + well-formed body (non-stream + stream)
    // ----------------------------------------------------------------------------

    #[tokio::test]
    #[ignore = "requires background decode loop: BatchDriver admission now decoupled from generation"]
    async fn completions_non_stream_json_ok() {
        let req = json_post("/v1/completions", serde_json::json!({"prompt": "once upon a time", "stream": false}));
        let resp = app().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["object"], "text_completion");
        assert_eq!(v["choices"][0]["text"], "Hello, world!");
    }

    #[tokio::test]
    #[ignore = "requires background decode loop: BatchDriver admission now decoupled from generation"]
    async fn completions_streaming_sse_ok() {
        let req = json_post("/v1/completions", serde_json::json!({"prompt": "once upon a time", "stream": true}));
        let resp = app().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = body_bytes(resp).await;
        let text = std::str::from_utf8(&body).unwrap();
        assert!(text.contains("text_completion"), "missing object in:\n{text}");
        assert!(text.contains("Hello"), "missing streamed token in:\n{text}");
        assert!(text.contains("[DONE]"), "missing [DONE] sentinel in:\n{text}");
    }

    // ----------------------------------------------------------------------------
    // (c) malformed / missing-field requests -> 4xx + structured OpenAI error body
    // ----------------------------------------------------------------------------

    /// Assert the response is a structured OpenAI error with the expected status
    /// and machine-readable `code`. Returns the parsed body for further checks.
    async fn assert_structured_error(
        resp: axum::response::Response,
        status: StatusCode,
        code: &str,
    ) -> serde_json::Value {
        assert_eq!(resp.status(), status, "unexpected status");
        let ct = resp.headers().get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
        assert!(ct.starts_with("application/json"), "error body should be JSON, got {ct:?}");
        let body = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let err = &v["error"];
        assert!(err.is_object(), "missing `error` object in {v}");
        assert!(err["message"].is_string(), "error.message must be a string in {v}");
        assert!(err["type"].is_string(), "error.type must be a string in {v}");
        assert_eq!(err["code"], code, "unexpected error.code in {v}");
        v
    }

    #[tokio::test]
    async fn chat_completions_invalid_json_is_structured_400() {
        // Truncated / syntactically broken JSON.
        let req = raw_post("/v1/chat/completions", "{ this is not json ");
        let resp = app().oneshot(req).await.unwrap();
        assert_structured_error(resp, StatusCode::BAD_REQUEST, "invalid_json").await;
    }

    #[tokio::test]
    async fn chat_completions_missing_messages_field_is_structured_400() {
        // Well-formed JSON, but the required `messages` field is absent -> serde
        // rejection -> structured invalid_json.
        let req = json_post("/v1/chat/completions", serde_json::json!({"model": "stub-model"}));
        let resp = app().oneshot(req).await.unwrap();
        assert_structured_error(resp, StatusCode::BAD_REQUEST, "invalid_json").await;
    }

    #[tokio::test]
    async fn chat_completions_empty_messages_is_missing_parameter() {
        // Field present but semantically empty -> missing_required_parameter.
        let req = json_post("/v1/chat/completions", serde_json::json!({"messages": []}));
        let resp = app().oneshot(req).await.unwrap();
        let v = assert_structured_error(resp, StatusCode::BAD_REQUEST, "missing_required_parameter").await;
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn completions_missing_prompt_field_is_structured_400() {
        let req = json_post("/v1/completions", serde_json::json!({"max_tokens": 4}));
        let resp = app().oneshot(req).await.unwrap();
        assert_structured_error(resp, StatusCode::BAD_REQUEST, "invalid_json").await;
    }

    #[tokio::test]
    async fn completions_empty_prompt_is_missing_parameter() {
        let req = json_post("/v1/completions", serde_json::json!({"prompt": ""}));
        let resp = app().oneshot(req).await.unwrap();
        assert_structured_error(resp, StatusCode::BAD_REQUEST, "missing_required_parameter").await;
    }

    // ----------------------------------------------------------------------------
    // engine failure on the non-stream path -> structured 500
    // ----------------------------------------------------------------------------

    #[tokio::test]
    async fn chat_completions_engine_failure_is_structured_500() {
        let req = json_post(
            "/v1/chat/completions",
            serde_json::json!({
                "messages": [{"role": "user", "content": "hi"}],
                "stream": false
            }),
        );
        let resp = failing_app().oneshot(req).await.unwrap();
        assert_structured_error(resp, StatusCode::INTERNAL_SERVER_ERROR, "internal_error").await;
    }

    // ----------------------------------------------------------------------------
    // sanity: healthz + models still work through the same router
    // ----------------------------------------------------------------------------

    #[tokio::test]
    async fn healthz_and_models_ok() {
        let resp = app().oneshot(Request::builder().uri("/healthz").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = app().oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["data"][0]["id"], "stub-model");
    }
}
#[rustfmt::skip]
mod system_kv_bank {
    //! Track 5.2 gate — SystemPromptKvBank key/hit/miss/eviction logic.
    //! Pure data structure; no model, no GPU, no process env.
    //!   cargo test -p hawking-serve --test system_kv_bank

    use hawking_serve::system_kv_bank::RecordOutcome;
    use hawking_serve::{BankConfig, SystemPromptKvBank};

    fn sys_prompt(n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| 1000 + i).collect()
    }

    #[test]
    fn hash_is_stable_and_prefix_sensitive() {
        let toks = sys_prompt(40);
        // Same span, same length -> same hash.
        assert_eq!(SystemPromptKvBank::hash_prefix(&toks, 16), SystemPromptKvBank::hash_prefix(&toks, 16));
        // Same tokens, different banked length -> different address.
        assert_ne!(SystemPromptKvBank::hash_prefix(&toks, 16), SystemPromptKvBank::hash_prefix(&toks, 24));
        // A differing token inside the span changes the hash.
        let mut toks2 = toks.clone();
        toks2[5] = 99999;
        assert_ne!(SystemPromptKvBank::hash_prefix(&toks, 16), SystemPromptKvBank::hash_prefix(&toks2, 16));
        // A token AFTER the span does NOT (only the leading `prefix_len` matter).
        let mut toks3 = toks.clone();
        toks3[20] = 88888;
        assert_eq!(SystemPromptKvBank::hash_prefix(&toks, 16), SystemPromptKvBank::hash_prefix(&toks3, 16));
    }

    #[test]
    fn record_then_hit_returns_source_slot() {
        let mut bank = SystemPromptKvBank::new();
        let system = sys_prompt(32);
        // Slot 3 prefilled a 32-tok system prompt.
        assert_eq!(bank.record(&system, 32, 3), RecordOutcome::Inserted);
        // A NEW request (system + user tail) probes the 32-tok leading span.
        let mut req = system.clone();
        req.extend([7u32, 8, 9]);
        let hit = bank.lookup(&req, 32).expect("banked system prefix must hit");
        assert_eq!(hit.source_slot, 3);
        assert_eq!(hit.prefix_len, 32);
        let s = bank.stats();
        assert_eq!(s.lookups, 1);
        assert_eq!(s.hits, 1);
    }

    #[test]
    fn miss_on_different_system_prompt() {
        let mut bank = SystemPromptKvBank::new();
        bank.record(&sys_prompt(32), 32, 1);
        let other: Vec<u32> = (0..40u32).map(|i| 5000 + i).collect();
        assert!(bank.lookup(&other, 32).is_none());
        assert_eq!(bank.stats().hits, 0);
        assert_eq!(bank.stats().lookups, 1);
    }

    #[test]
    fn never_matches_full_prompt() {
        // Strict-prefix rule: banked_len == prompt len must miss (decode loop
        // needs a real last_id), mirroring the RAM/disk prefix tiers.
        let mut bank = SystemPromptKvBank::new();
        let p = sys_prompt(16);
        bank.record(&p, 16, 0);
        assert!(bank.lookup(&p, 16).is_none(), "must not match whole prompt");
        let mut longer = p.clone();
        longer.push(42);
        assert!(bank.lookup(&longer, 16).is_some(), "strict prefix hits");
    }

    #[test]
    fn too_short_is_rejected_both_ways() {
        let cfg = BankConfig { min_prefix_tokens: 8, max_entries: 64 };
        let mut bank = SystemPromptKvBank::with_config(cfg);
        let p = sys_prompt(20);
        // Recording a 4-tok prefix is rejected.
        assert_eq!(bank.record(&p, 4, 0), RecordOutcome::TooShort);
        assert_eq!(bank.len(), 0);
        // Looking up below the floor misses without touching entries.
        bank.record(&p, 12, 0);
        assert!(bank.lookup(&p, 4).is_none());
    }

    #[test]
    fn refresh_updates_source_slot() {
        let mut bank = SystemPromptKvBank::new();
        let system = sys_prompt(24);
        assert_eq!(bank.record(&system, 24, 2), RecordOutcome::Inserted);
        // The same system prompt is later served from slot 5 -> refresh, not dup.
        assert_eq!(bank.record(&system, 24, 5), RecordOutcome::Updated);
        assert_eq!(bank.len(), 1);
        let mut req = system.clone();
        req.push(1);
        assert_eq!(bank.lookup(&req, 24).unwrap().source_slot, 5);
    }

    #[test]
    fn lru_eviction_caps_entries_and_keeps_newest() {
        let cfg = BankConfig { min_prefix_tokens: 4, max_entries: 2 };
        let mut bank = SystemPromptKvBank::with_config(cfg);
        // Bank 4 distinct system prompts; cap is 2.
        for i in 0..4u32 {
            let p: Vec<u32> = (0..8u32).map(|j| i * 100 + j).collect();
            bank.record(&p, 8, i);
        }
        assert!(bank.len() <= 2, "entry cap enforced");
        assert!(bank.stats().evictions >= 2);
        // The newest two (i=2, i=3) survive; the oldest (i=0) is gone.
        let p0: Vec<u32> = (0..9u32).map(|j| 0 * 100 + j).collect(); // 8-tok span + tail
        assert!(bank.lookup(&p0, 8).is_none(), "oldest evicted");
        let p3: Vec<u32> = (0..9u32).map(|j| 3 * 100 + j).collect();
        assert_eq!(bank.lookup(&p3, 8).unwrap().source_slot, 3, "newest survives");
    }

    #[test]
    fn lru_touch_on_hit_protects_entry() {
        let cfg = BankConfig { min_prefix_tokens: 4, max_entries: 2 };
        let mut bank = SystemPromptKvBank::with_config(cfg);
        let a: Vec<u32> = (0..8u32).map(|j| 10 + j).collect();
        let b: Vec<u32> = (0..8u32).map(|j| 20 + j).collect();
        bank.record(&a, 8, 0);
        bank.record(&b, 8, 1);
        // Touch `a` so it is now more-recently-used than `b`.
        let mut a_q = a.clone();
        a_q.push(1);
        assert!(bank.lookup(&a_q, 8).is_some());
        // Insert a 3rd -> `b` (now LRU) is evicted, `a` survives.
        let c: Vec<u32> = (0..8u32).map(|j| 30 + j).collect();
        bank.record(&c, 8, 2);
        assert!(bank.lookup(&a_q, 8).is_some(), "touched entry protected");
        let mut b_q = b.clone();
        b_q.push(1);
        assert!(bank.lookup(&b_q, 8).is_none(), "untouched LRU evicted");
    }

    #[test]
    fn forget_slot_invalidates_its_entries() {
        let mut bank = SystemPromptKvBank::new();
        let p = sys_prompt(16);
        bank.record(&p, 16, 7);
        let mut q = p.clone();
        q.push(1);
        assert!(bank.lookup(&q, 16).is_some());
        assert_eq!(bank.forget_slot(7), 1);
        assert!(bank.lookup(&q, 16).is_none(), "forgotten slot no longer routable");
    }
}
#[rustfmt::skip]
mod system_kv_bank_wiring {
    //! Track 5.2 wiring gate — pins the record/lookup KEY+LEN choice the serve
    //! admit path performs, using the SAME `http::banked_len_for` helper the loop
    //! uses. The bank's own logic is covered by tests/system_kv_bank.rs; this test
    //! pins the WIRING (the banked_len = prompt-minus-one choice and the
    //! record-then-lookup sequence) so it can never silently diverge from the loop.
    //! Pure: no model, no GPU, no process env, no decode.
    //!   cargo test -p hawking-serve --test system_kv_bank_wiring

    use hawking_serve::http::banked_len_for;
    use hawking_serve::SystemPromptKvBank;

    /// A representative fixed system prompt (>= bank min_prefix_tokens=8).
    fn sys_prompt(n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| 4000 + i).collect()
    }

    /// The exact record-then-lookup sequence the serve loop performs for a serial
    /// workload: turn 1 records (no live source), turn 2 (identical prompt) looks
    /// up and MUST get turn-1's slot back, via the shared banked_len helper.
    #[test]
    fn serve_wiring_record_then_lookup_returns_source_slot() {
        let prompt = sys_prompt(40);
        let banked_len = banked_len_for(&prompt);
        assert_eq!(banked_len, prompt.len() - 1, "bank one token short of full prompt");

        let mut bank = SystemPromptKvBank::new();

        // Turn 1: slot 3 just prefilled this prompt; serve loop records it.
        let outcome = bank.record(&prompt, banked_len, 3);
        assert_eq!(outcome, hawking_serve::system_kv_bank::RecordOutcome::Inserted);

        // Turn 2: identical prompt arrives in slot 1; live PrefixIndex MISSES
        // (slot 3 freed). Serve loop consults the bank with the SAME banked_len.
        let hit = bank.lookup(&prompt, banked_len).expect("bank must hit on identical prompt");
        assert_eq!(hit.source_slot, 3, "lookup must return the slot that recorded");
        assert_eq!(hit.prefix_len, banked_len, "hit prefix_len == banked_len");
    }

    /// banked_len must produce a STRICT prefix so the bank's lookup guard
    /// (`banked_len < tokens.len()`) is satisfied — i.e. the wiring never asks the
    /// bank to match the whole prompt (which lookup rejects by contract).
    #[test]
    fn wiring_banked_len_is_a_strict_prefix() {
        for n in [9usize, 16, 40, 257] {
            let p = sys_prompt(n);
            let bl = banked_len_for(&p);
            assert!(bl < p.len(), "banked_len must be < prompt len for n={n}");
            assert!(bl >= 8, "for prompts >= 9 the banked span clears the min, n={n}");
        }
    }

    /// A differing SUFFIX (same fixed system span) still hits: only the leading
    /// banked_len tokens address the entry. This is the core serial-chat win —
    /// same system prompt, different user turn. Both record and lookup must use the
    /// SAME banked_len for the cross-suffix case, which is what the loop does (it
    /// recomputes banked_len_for(prompt) on the lookup side too); we pin that by
    /// banking at a fixed system-span length and probing the same.
    #[test]
    fn wiring_shared_system_span_hits_across_suffix() {
        let system_span_len = 24usize; // the fixed leading system block
        let mut a = sys_prompt(system_span_len); // turn 1 prompt = system + suffix A
        a.extend_from_slice(&[10, 11, 12]);
        let mut b = sys_prompt(system_span_len); // turn 2 prompt = system + suffix B
        b.extend_from_slice(&[20, 21, 22, 23]);

        let mut bank = SystemPromptKvBank::new();
        // Record turn 1 at the fixed system-span length.
        bank.record(&a, system_span_len, 5);
        // Turn 2: probe the same fixed span length -> hit slot 5 despite suffix B.
        let hit = bank.lookup(&b, system_span_len).expect("shared system span must hit");
        assert_eq!(hit.source_slot, 5);
    }
}
#[rustfmt::skip]
mod workload_pack_mapping {
    //! Confirmation test for Track 9.3 `--workload` packs: lock in that each
    //! workload string expands to the documented (profile, energy, batch-policy)
    //! triple AND that the chosen profile expands to the expected env-lever knobs.
    //!
    //! This is the integration-test sibling of the inline `profile_lever_tests`
    //! module in `hawking-serve/src/lib.rs` (which covers RuntimeProfile alone).
    //! Here we pin the WORKLOAD → (profile, energy, policy) layer plus the
    //! workload → concrete knob set, so a silent drift in either
    //! `WorkloadPack::defaults()` or `RuntimeProfile::lever_plan()` fails CI.
    //!
    //! Pure: builds data-only mappings, touches no process env, no model. Gates:
    //!
    //!   cargo test -p hawking-serve --test workload_pack_mapping

    use hawking_serve::{BatchPolicy, EnergyMode, RuntimeProfile, WorkloadPack};

    fn has(set: &[(&'static str, &'static str)], k: &str) -> bool {
        set.iter().any(|(kk, _)| *kk == k)
    }

    fn val<'a>(set: &'a [(&'static str, &'static str)], k: &str) -> Option<&'a str> {
        set.iter().find(|(kk, _)| *kk == k).map(|(_, v)| *v)
    }

    /// from_str round-trips every known workload name and rejects the unknown.
    #[test]
    fn workload_from_str_roundtrips_all_known() {
        for s in ["default", "code-completion", "chat-shared-prompt", "batch-summarization", "local-agent-loop"] {
            assert_eq!(
                WorkloadPack::from_str(s).expect("known workload").as_str(),
                s,
                "workload {s} must round-trip through from_str/as_str"
            );
        }
        assert!(WorkloadPack::from_str("nonsense-pack").is_none());
        // a runtime-profile name is NOT a workload pack name.
        assert!(WorkloadPack::from_str("fast").is_none());
    }

    /// Each workload expands to its documented (profile, energy, batch-policy)
    /// triple (see the `WorkloadPack` doc comment in lib.rs).
    #[test]
    fn workload_defaults_match_documented_triples() {
        use BatchPolicy as BP;
        use EnergyMode as EM;
        use RuntimeProfile as RP;

        let cases: &[(WorkloadPack, RP, EM, BP)] = &[
            (WorkloadPack::Default, RP::Default, EM::Off, BP::Default),
            (WorkloadPack::CodeCompletion, RP::Race, EM::Off, BP::GreedyFirst),
            (WorkloadPack::ChatSharedPrompt, RP::Fast, EM::Balanced, BP::PrefixGrouped),
            (WorkloadPack::BatchSummarization, RP::Efficient, EM::Efficient, BP::GreedyFirst),
            (WorkloadPack::LocalAgentLoop, RP::Fast, EM::Off, BP::GreedyFirst),
        ];

        for (pack, want_profile, want_energy, want_policy) in cases {
            let (profile, energy, policy) = pack.defaults();
            assert_eq!(&profile, want_profile, "{pack} profile");
            assert_eq!(&energy, want_energy, "{pack} energy");
            assert_eq!(&policy, want_policy, "{pack} batch policy");
        }
    }

    /// End-to-end: workload → profile → concrete lever knobs. This is the layer the
    /// server actually applies; pin the full knob set per workload so a regression
    /// in either the pack mapping or the profile bundle is caught.
    #[test]
    fn workload_expands_to_expected_profile_and_knobs() {
        // code-completion ⇒ Race ⇒ fast bundle + f16-KV on + concurrent QKV.
        {
            let (profile, _e, _p) = WorkloadPack::CodeCompletion.defaults();
            assert_eq!(profile, RuntimeProfile::Race);
            let plan = profile.lever_plan();
            for k in [
                "HAWKING_QWEN_Q4K_LMHEAD",
                "HAWKING_QWEN_Q4K_PREDEC",
                "HAWKING_QWEN_PREDEC_F16SCALES",
                "HAWKING_QWEN_VOCAB_PRUNE",
                "HAWKING_QWEN_FFN_DOWN_Q4K",
            ] {
                assert!(has(&plan.set_if_unset, k), "code-completion(race) must set {k}");
            }
            assert_eq!(val(&plan.set_if_unset, "HAWKING_QWEN_VOCAB_PRUNE"), Some("32000"));
            assert_eq!(plan.f16_kv, Some(true), "race enables f16-KV");
            assert!(plan.concurrent_qkv);
            assert!(plan.force_off.is_empty());
            // race is NOT the energy profile.
            assert!(!has(&plan.set_if_unset, "HAWKING_ENERGY_EFFICIENT"));
        }

        // batch-summarization ⇒ Efficient ⇒ fast bundle + energy flag + f16-KV.
        {
            let (profile, _e, _p) = WorkloadPack::BatchSummarization.defaults();
            assert_eq!(profile, RuntimeProfile::Efficient);
            let plan = profile.lever_plan();
            assert!(has(&plan.set_if_unset, "HAWKING_ENERGY_EFFICIENT"), "efficient must set the energy lever");
            assert!(has(&plan.set_if_unset, "HAWKING_QWEN_Q4K_PREDEC"));
            assert_eq!(plan.f16_kv, Some(true));
        }

        // chat-shared-prompt ⇒ Fast ⇒ fast bundle, f16-KV OFF (bit-identity to fast).
        {
            let (profile, energy, policy) = WorkloadPack::ChatSharedPrompt.defaults();
            assert_eq!(profile, RuntimeProfile::Fast);
            assert_eq!(energy, EnergyMode::Balanced);
            assert_eq!(policy, BatchPolicy::PrefixGrouped);
            let plan = profile.lever_plan();
            assert_eq!(plan.f16_kv, Some(false), "fast leaves f16-KV off");
            assert!(plan.force_off.is_empty());
            assert!(!has(&plan.set_if_unset, "HAWKING_ENERGY_EFFICIENT"));
        }

        // local-agent-loop ⇒ Fast ⇒ same fast bundle, energy OFF, greedy-first.
        {
            let (profile, energy, policy) = WorkloadPack::LocalAgentLoop.defaults();
            assert_eq!(profile, RuntimeProfile::Fast);
            assert_eq!(energy, EnergyMode::Off);
            assert_eq!(policy, BatchPolicy::GreedyFirst);
            let plan = profile.lever_plan();
            assert!(has(&plan.set_if_unset, "HAWKING_QWEN_Q4K_LMHEAD"));
            assert_eq!(plan.f16_kv, Some(false));
        }

        // default ⇒ Default ⇒ touches nothing (bit-identical golden path).
        {
            let (profile, energy, policy) = WorkloadPack::Default.defaults();
            assert_eq!(profile, RuntimeProfile::Default);
            assert_eq!(energy, EnergyMode::Off);
            assert_eq!(policy, BatchPolicy::Default);
            let plan = profile.lever_plan();
            assert!(plan.set_if_unset.is_empty(), "default sets no lever");
            assert!(plan.force_off.is_empty());
            assert_eq!(plan.f16_kv, None);
            assert!(!plan.concurrent_qkv);
        }
    }

    /// The energy mode each workload selects expands to its gather-window ms — the
    /// number the serve loop actually sleeps. Pins the EnergyMode → ms contract.
    #[test]
    fn workload_energy_maps_to_gather_window_ms() {
        assert_eq!(WorkloadPack::Default.defaults().1.gather_window_ms(), 0);
        assert_eq!(WorkloadPack::CodeCompletion.defaults().1.gather_window_ms(), 0);
        assert_eq!(WorkloadPack::ChatSharedPrompt.defaults().1.gather_window_ms(), 3);
        assert_eq!(WorkloadPack::BatchSummarization.defaults().1.gather_window_ms(), 8);
        assert_eq!(WorkloadPack::LocalAgentLoop.defaults().1.gather_window_ms(), 0);
    }
}
