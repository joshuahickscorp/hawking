//! Batched teacher-capture driver for the on-device RWKV-7 post-train.
//!
//! The single-stream `hawking generate --prompts-file` loop decodes ONE
//! prompt at a time (`max_batch_size = 1`), so each weight read serves a single
//! sequence. For the RWKV-7 teacher corpus we run *thousands* of independent
//! prompts through the same model — exactly the workload the continuous-batch
//! / multiseq path was built for. This driver issues `prefill_slots_parallel`
//! then `forward_multiseq_greedy_tokens` over groups of up to B=8 sequences, so
//! each Q4_K weight read is amortised across the whole group.
//!
//! **Quality is identical to single-stream greedy capture.** The multiseq
//! greedy lane uses the same per-prompt `forward_token_greedy_tcb` prefill and
//! the same Q4_K-LM-head batched argmax the single-stream path uses; the only
//! difference is that B sequences share each weight read. There is no sampling,
//! no temperature, no draft — it is the `--profile exact`, temperature-0 teacher
//! the runbook already specifies. See `qwen_dense.rs::prefill_slot` (which
//! documents the bit-for-bit match to the CLI) and
//! `forward_tokens_multiseq_greedy`.
//!
//! Output is **sharded JSONL**: each completed batch-group is flushed to its own
//! `<out>.shard-NNNN.jsonl` immediately, so a streaming SFT trainer can begin on
//! finished shards while capture continues (pipeline overlap — see the runbook
//! §"Pipeline capture → train"). Each line is
//! `{"idx", "prompt", "completion", "stop"}`.
//!
//! Only greedy (temperature == 0) capture is batched. A non-greedy request
//! falls back to the caller's per-prompt path with a clear message — the teacher
//! corpus is greedy by construction so this is not a limitation in practice.

use anyhow::{anyhow, Result};
use hawking_core::Engine;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Hard cap on concurrent multiseq slots — must match `MAX_MULTISEQ_SLOTS` in
/// `qwen_dense.rs`. The arena is allocated for exactly this many slot-strided
/// KV regions.
pub const MAX_CAPTURE_BATCH: usize = 8;

/// One captured (prompt, completion) record. Serialised as a JSONL line.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CaptureRecord {
    /// 0-based index of this prompt in the input prompts file (stable across
    /// shards so the trainer can re-join with the source corpus).
    pub idx: usize,
    pub prompt: String,
    pub completion: String,
    /// Why decode stopped for this prompt: "eos" | "max_tokens".
    pub stop: &'static str,
}

/// Per-slot decode state inside one batch-group.
struct SlotRun {
    /// Index into the original prompts list (for stable record ids).
    global_idx: usize,
    prompt: String,
    /// Stable multiseq slot id (0..B). Keys the slot-strided KV region.
    slot_id: usize,
    /// Token ids generated so far (decoded to text at the end of the group).
    out_tokens: Vec<u32>,
    /// Absolute decode position of the NEXT token to feed (prompt_len, then +1
    /// per accepted token).
    next_pos: usize,
    /// Set once this slot hits EOS or the token budget.
    done: bool,
    stop: &'static str,
}

/// Config for a batched-capture run.
pub struct CaptureConfig {
    pub batch: usize,
    pub max_new_tokens: usize,
    pub out_path: PathBuf,
    /// Max prompt length in tokens; longer prompts are skipped with a warning
    /// (they would overflow the per-slot KV region, MAX_MULTISEQ_CTX = 4096).
    pub max_prompt_tokens: usize,
}

/// Run batched greedy capture over `prompts`, writing sharded JSONL.
///
/// Returns the total number of records written. Each group of up to
/// `cfg.batch` prompts is decoded together and flushed to its own shard file
/// before the next group starts — so a streaming trainer can consume completed
/// shards while later groups are still being captured.
pub fn run_batched_capture(
    engine: &mut dyn Engine,
    prompts: &[String],
    cfg: &CaptureConfig,
    abort: &Arc<AtomicBool>,
) -> Result<usize> {
    let batch = cfg.batch.clamp(1, MAX_CAPTURE_BATCH);
    let eos = engine.eos_id_for_batch();

    // Sanity: confirm the engine actually implements the batched seam. If
    // prefill_slot is unimplemented (non-Qwen / non-macOS), fail loud rather
    // than silently degrading — the caller picks the serial path instead.
    if engine.encode_prompt_for_batch("probe").is_err() {
        return Err(anyhow!(
            "engine does not implement the batched-capture seam \
             (encode_prompt_for_batch). Use the serial --prompts-file path \
             (drop --batched-capture) for this model/platform."
        ));
    }

    // Pre-tokenize every prompt once (CPU) so we can skip over-length prompts
    // and report token counts. encode_prompt_for_batch is the SAME tokenizer
    // the single-stream generate() uses.
    let mut tokenized: Vec<(usize, &String, Vec<u32>)> = Vec::with_capacity(prompts.len());
    for (i, p) in prompts.iter().enumerate() {
        let ids = engine.encode_prompt_for_batch(p)?;
        if ids.is_empty() {
            eprintln!("[capture] skip prompt {i}: empty after tokenization");
            continue;
        }
        if ids.len() > cfg.max_prompt_tokens {
            eprintln!(
                "[capture] skip prompt {i}: {} tokens > max_prompt_tokens {}",
                ids.len(),
                cfg.max_prompt_tokens
            );
            continue;
        }
        tokenized.push((i, p, ids));
    }
    if tokenized.is_empty() {
        return Err(anyhow!(
            "no usable prompts after tokenization/length filter"
        ));
    }

    let total = tokenized.len();
    let n_groups = total.div_ceil(batch);
    eprintln!(
        "[capture] batched greedy capture: {total} prompts, B={batch}, \
         {n_groups} groups, max_new_tokens={}, out={}",
        cfg.max_new_tokens,
        cfg.out_path.display()
    );

    let mut written = 0usize;
    for (group_idx, chunk) in tokenized.chunks(batch).enumerate() {
        if abort.load(Ordering::SeqCst) {
            eprintln!("[capture] aborted before group {group_idx}");
            break;
        }
        let recs = capture_one_group(engine, chunk, cfg.max_new_tokens, eos, abort)?;
        let shard = shard_path(&cfg.out_path, group_idx);
        write_shard(&shard, &recs)?;
        written += recs.len();
        eprintln!(
            "[capture] group {}/{} done -> {} ({} records, {} total)",
            group_idx + 1,
            n_groups,
            shard.display(),
            recs.len(),
            written
        );
    }
    Ok(written)
}

/// Decode one group of ≤B prompts together via the multiseq greedy lane.
///
/// Each prompt gets a stable slot id `0..b`. We prefill all `b` slots in one
/// `prefill_slots_parallel` pass (weights read once per position across slots),
/// seed each slot with its first generated token, then step the *active*
/// (not-yet-finished) slots together until all are done. Finished slots drop
/// out of the active set so the kernel only ever sees live sequences.
fn capture_one_group(
    engine: &mut dyn Engine,
    chunk: &[(usize, &String, Vec<u32>)],
    max_new_tokens: usize,
    eos: Option<u32>,
    abort: &Arc<AtomicBool>,
) -> Result<Vec<CaptureRecord>> {
    let b = chunk.len();

    // Parallel prefill: slot id == position in this group. Returns the FIRST
    // generated token per slot (argmax of each prompt's last position).
    let slot_refs: Vec<(usize, &[u32])> = chunk
        .iter()
        .enumerate()
        .map(|(slot_id, (_, _, ids))| (slot_id, ids.as_slice()))
        .collect();
    let first_tokens = engine
        .prefill_slots_parallel(&slot_refs)
        .map_err(|e| anyhow!("prefill_slots_parallel (group of {b}): {e}"))?;

    let mut runs: Vec<SlotRun> = chunk
        .iter()
        .enumerate()
        .map(|(slot_id, (gidx, p, ids))| {
            let first = first_tokens[slot_id];
            let is_eos = eos == Some(first);
            // The first token already counts toward the budget. If it is EOS
            // the completion is empty and the slot is immediately done.
            let (out_tokens, done, stop) = if is_eos {
                (Vec::new(), true, "eos")
            } else if max_new_tokens <= 1 {
                (vec![first], true, "max_tokens")
            } else {
                (vec![first], false, "max_tokens")
            };
            SlotRun {
                global_idx: *gidx,
                prompt: (*p).clone(),
                slot_id,
                out_tokens,
                next_pos: ids.len(), // first decode token is fed at pos = prompt_len
                done,
                stop,
            }
        })
        .collect();

    // Decode loop: step all still-active slots in lock-step. The token seeded
    // above is at logical position `next_pos`; feeding it produces the token at
    // `next_pos + 1`, so we advance positions per accepted token.
    loop {
        if abort.load(Ordering::SeqCst) {
            for r in runs.iter_mut().filter(|r| !r.done) {
                r.done = true;
                r.stop = "aborted";
            }
            break;
        }

        // Gather the active slots (not finished, still under budget).
        let active: Vec<usize> = runs
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.done)
            .map(|(i, _)| i)
            .collect();
        if active.is_empty() {
            break;
        }

        // Feed each active slot its LAST generated token at its current
        // position. tokens/positions/regions are parallel arrays; region ==
        // the slot's stable KV id.
        let tokens: Vec<u32> = active
            .iter()
            .map(|&i| *runs[i].out_tokens.last().unwrap())
            .collect();
        let positions: Vec<usize> = active.iter().map(|&i| runs[i].next_pos).collect();
        let regions: Vec<usize> = active.iter().map(|&i| runs[i].slot_id).collect();

        let next = engine
            .forward_multiseq_greedy_tokens(&tokens, &positions, &regions)
            .map_err(|e| anyhow!("forward_multiseq_greedy_tokens (b={}): {e}", active.len()))?;

        for (k, &slot_i) in active.iter().enumerate() {
            let tok = next[k];
            let r = &mut runs[slot_i];
            r.next_pos += 1;
            if eos == Some(tok) {
                r.done = true;
                r.stop = "eos";
                continue;
            }
            r.out_tokens.push(tok);
            if r.out_tokens.len() >= max_new_tokens {
                r.done = true;
                r.stop = "max_tokens";
            }
        }
    }

    // Decode each slot's token stream to text (CPU). decode_token_for_batch is
    // the same per-token detokenizer the streaming path uses.
    let mut recs = Vec::with_capacity(b);
    for r in &runs {
        let mut completion = String::new();
        for &t in &r.out_tokens {
            completion.push_str(&engine.decode_token_for_batch(t).unwrap_or_default());
        }
        recs.push(CaptureRecord {
            idx: r.global_idx,
            prompt: r.prompt.clone(),
            completion,
            stop: r.stop,
        });
    }
    // Restore input order within the group (already in order, but be explicit).
    recs.sort_by_key(|r| r.idx);
    Ok(recs)
}

/// `<dir>/<stem>.shard-NNNN.<ext>` for group `g`. Keeps the user's extension.
fn shard_path(out: &Path, g: usize) -> PathBuf {
    let parent = out.parent().filter(|p| !p.as_os_str().is_empty());
    let stem = out
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "capture".into());
    let ext = out
        .extension()
        .map(|e| e.to_string_lossy().into_owned())
        .unwrap_or_else(|| "jsonl".into());
    let name = format!("{stem}.shard-{g:04}.{ext}");
    match parent {
        Some(p) => p.join(name),
        None => PathBuf::from(name),
    }
}

fn write_shard(path: &Path, recs: &[CaptureRecord]) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow!("create shard dir {}: {e}", parent.display()))?;
    }
    let mut f =
        std::fs::File::create(path).map_err(|e| anyhow!("create shard {}: {e}", path.display()))?;
    for r in recs {
        let line = serde_json::to_string(r).map_err(|e| anyhow!("serialize record: {e}"))?;
        f.write_all(line.as_bytes())?;
        f.write_all(b"\n")?;
    }
    f.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hawking_core::{
        EngineConfig, GenStats, GenerateRequest, Result as CoreResult, StreamEvent,
    };

    /// A deterministic fake engine that implements only the batched seam.
    /// "Tokenization" = bytes; decode = a fixed token stream per slot that ends
    /// in EOS, so we can assert grouping + stop handling without a GPU.
    struct FakeEngine {
        eos: u32,
        /// next-token map keyed by (current token) -> next token; EOS terminates.
        chain: std::collections::HashMap<u32, u32>,
    }

    impl FakeEngine {
        fn new() -> Self {
            // 100 -> 101 -> 102 -> EOS(0); 200 -> 201 -> EOS(0)
            let mut chain = std::collections::HashMap::new();
            chain.insert(100, 101);
            chain.insert(101, 102);
            chain.insert(102, 0);
            chain.insert(200, 201);
            chain.insert(201, 0);
            Self { eos: 0, chain }
        }
    }

    impl Engine for FakeEngine {
        fn load(_w: &std::path::Path, _c: EngineConfig) -> CoreResult<Self> {
            Ok(Self::new())
        }
        fn generate(
            &mut self,
            _r: GenerateRequest,
            _s: &mut dyn FnMut(StreamEvent),
        ) -> CoreResult<GenStats> {
            Ok(GenStats::default())
        }
        fn model_id(&self) -> &str {
            "fake"
        }
        fn encode_prompt_for_batch(&self, p: &str) -> CoreResult<Vec<u32>> {
            // first prompt byte 'a' -> seed 100, 'b' -> seed 200, else single tok
            Ok(p.bytes().map(u32::from).collect())
        }
        fn decode_token_for_batch(&self, t: u32) -> CoreResult<String> {
            Ok(format!("[{t}]"))
        }
        fn eos_id_for_batch(&self) -> Option<u32> {
            Some(self.eos)
        }
        fn forward_tokens_for_test(
            &mut self,
            tokens: &[u32],
            _pos: &[usize],
        ) -> CoreResult<Vec<Vec<f32>>> {
            // Not used by the batched path, but required by the trait.
            Ok(tokens.iter().map(|_| vec![0.0]).collect())
        }
        // The batched seam the capture driver actually calls:
        fn prefill_slots_parallel(&mut self, slots: &[(usize, &[u32])]) -> CoreResult<Vec<u32>> {
            // First generated token = 100 if prompt starts with 'a', else 200.
            Ok(slots
                .iter()
                .map(|(_, ids)| {
                    if ids.first() == Some(&(b'a' as u32)) {
                        100
                    } else {
                        200
                    }
                })
                .collect())
        }
        fn forward_multiseq_greedy_tokens(
            &mut self,
            tokens: &[u32],
            _positions: &[usize],
            _regions: &[usize],
        ) -> CoreResult<Vec<u32>> {
            Ok(tokens
                .iter()
                .map(|t| *self.chain.get(t).unwrap_or(&0))
                .collect())
        }
    }

    fn tmp(name: &str) -> PathBuf {
        let mut d = std::env::temp_dir();
        d.push(format!(
            "hawking_capture_test_{}_{}",
            std::process::id(),
            name
        ));
        d
    }

    #[test]
    fn shard_path_keeps_stem_and_ext() {
        let p = shard_path(Path::new("/tmp/x/teacher.jsonl"), 3);
        assert_eq!(p, PathBuf::from("/tmp/x/teacher.shard-0003.jsonl"));
        let p2 = shard_path(Path::new("out.jsonl"), 0);
        assert_eq!(p2, PathBuf::from("out.shard-0000.jsonl"));
    }

    #[test]
    fn batched_capture_groups_and_stops_on_eos() {
        let mut eng = FakeEngine::new();
        let prompts: Vec<String> = vec!["a-first".into(), "b-second".into(), "a-third".into()];
        let out = tmp("groups.jsonl");
        let cfg = CaptureConfig {
            batch: 2,
            max_new_tokens: 16,
            out_path: out.clone(),
            max_prompt_tokens: 4096,
        };
        let abort = Arc::new(AtomicBool::new(false));
        let n = run_batched_capture(&mut eng, &prompts, &cfg, &abort).expect("capture");
        assert_eq!(n, 3, "all three prompts captured");

        // Group 0 = prompts 0,1 ; group 1 = prompt 2. Read both shards.
        let s0 = std::fs::read_to_string(shard_path(&out, 0)).unwrap();
        let s1 = std::fs::read_to_string(shard_path(&out, 1)).unwrap();
        let lines0: Vec<_> = s0.lines().collect();
        let lines1: Vec<_> = s1.lines().collect();
        assert_eq!(lines0.len(), 2);
        assert_eq!(lines1.len(), 1);

        // 'a' chain: 100,101,102 then EOS -> completion "[100][101][102]", stop eos
        let r0: serde_json::Value = serde_json::from_str(lines0[0]).unwrap();
        assert_eq!(r0["idx"], 0);
        assert_eq!(r0["completion"], "[100][101][102]");
        assert_eq!(r0["stop"], "eos");
        // 'b' chain: 200,201 then EOS
        let r1: serde_json::Value = serde_json::from_str(lines0[1]).unwrap();
        assert_eq!(r1["idx"], 1);
        assert_eq!(r1["completion"], "[200][201]");
        assert_eq!(r1["stop"], "eos");

        let _ = std::fs::remove_file(shard_path(&out, 0));
        let _ = std::fs::remove_file(shard_path(&out, 1));
    }

    #[test]
    fn max_new_tokens_caps_completion() {
        let mut eng = FakeEngine::new();
        let prompts = vec!["a-x".to_string()];
        let out = tmp("cap.jsonl");
        let cfg = CaptureConfig {
            batch: 4,
            max_new_tokens: 2, // first token + one more, then cut
            out_path: out.clone(),
            max_prompt_tokens: 4096,
        };
        let abort = Arc::new(AtomicBool::new(false));
        run_batched_capture(&mut eng, &prompts, &cfg, &abort).expect("capture");
        let s = std::fs::read_to_string(shard_path(&out, 0)).unwrap();
        let r: serde_json::Value = serde_json::from_str(s.lines().next().unwrap()).unwrap();
        // seed=100 (counts as 1), then 101 -> budget 2 reached, stop max_tokens.
        assert_eq!(r["completion"], "[100][101]");
        assert_eq!(r["stop"], "max_tokens");
        let _ = std::fs::remove_file(shard_path(&out, 0));
    }
}
