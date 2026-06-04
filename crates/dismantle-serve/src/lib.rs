//! dismantle-serve: OpenAI-compatible HTTP server.
//!
//! Drives a `dismantle_core::Engine` through axum. Continuous
//! batching lives in [`batch`]; the HTTP surface in [`http`].

pub mod batch;
pub mod http;

use anyhow::Result;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

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
}

pub async fn run(opts: ServeOptions) -> Result<()> {
    use dismantle_core::{profile::KernelProfile, EngineConfig, SpeculateMode};

    let speculate_mode = SpeculateMode::from_cli(opts.speculate.as_deref(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let kernel_profile = match opts.kernel_profile.as_ref() {
        Some(path) => Some(KernelProfile::load(path)?),
        None => None,
    };
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
        ..Default::default()
    };

    let engine = dismantle_core::model::load_engine(&opts.weights, cfg)
        .map_err(|e| anyhow::anyhow!("load engine: {e}"))?;
    let model_arch = engine.model_arch().to_string();
    let max_batch = opts.max_batch_size;

    let state = http::AppState {
        engine: Arc::new(parking_lot::Mutex::new(engine)),
        driver: Arc::new(parking_lot::Mutex::new(batch::driver::BatchDriver::new(max_batch))),
        slot_senders: Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new())),
        wait_queue: Arc::new(parking_lot::Mutex::new(std::collections::VecDeque::new())),
        model_arch,
        max_batch,
        requests_admitted: Arc::new(AtomicU64::new(0)),
        tokens_generated: Arc::new(AtomicU64::new(0)),
        requests_queued: Arc::new(AtomicU64::new(0)),
    };

    // ── Background continuous-batching loop ───────────────────────────────
    // Single blocking thread: Phase A prefills pending slots, Phase B runs
    // one decode step across all ready slots, Phase C streams tokens to SSE.
    // All GPU kernel dispatches happen here under the engine lock; HTTP
    // handlers only hold the lock briefly for the admit tokenization step.
    {
        let state2 = state.clone();
        tokio::task::spawn_blocking(move || {
            loop {
                // ── Phase A: prefill all slots in Prefilling state ────────
                let prefilling: Vec<u32> = state2.driver.lock().scheduler.prefill_slots(max_batch);
                for slot_id in prefilling {
                    let prompt_ids = state2.driver.lock().scheduler.slots
                        .iter()
                        .find(|s| s.id == slot_id)
                        .map(|s| s.prompt_ids.clone())
                        .unwrap_or_default();
                    if prompt_ids.is_empty() { continue; }
                    match state2.engine.lock().prefill_slot(slot_id as usize, &prompt_ids) {
                        Ok(_last_tok) => {
                            state2.driver.lock().scheduler.mark_prefill_complete(slot_id);
                        }
                        Err(e) => {
                            tracing::warn!(slot = slot_id, err = %e, "prefill_slot failed");
                            let tx = state2.slot_senders.lock().remove(&slot_id);
                            if let Some(tx) = tx { let _ = tx.blocking_send(Err(())); }
                            state2.driver.lock().scheduler.release_slot(slot_id);
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
    tracing::info!(addr = %opts.addr, "dismantle-serve listening");
    let listener = tokio::net::TcpListener::bind(opts.addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
