mod common;
use common::weights_path_deepseek as weights_path;
fn load_engine() -> Option<Box<dyn hawking_core::Engine>> {
    let p = weights_path();
    if !p.exists() {
        eprintln!(
            "v1_1_phase4C_batched_forward_parity: no weights at {:?}, skipping",
            p
        );
        return None;
    }
    let cfg = hawking_core::EngineConfig::default();
    match hawking_core::model::load_engine(&p, cfg) {
        Ok(e) => Some(e),
        Err(err) => {
            eprintln!("v1_1_phase4C_batched_forward_parity: load failed: {err}, skipping");
            None
        }
    }
}
fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
fn check_argmax_parity(
    engine: &mut Box<dyn hawking_core::Engine>,
    tokens: &[u32],
    positions: &[usize],
    label: &str,
) {
    let seq_logits = engine
        .forward_tokens_for_test(tokens, positions)
        .unwrap_or_else(|e| panic!("{label} sequential: {e}"));
    engine.reset_kv_for_test();
    let batch_logits = engine
        .forward_tokens_batched_for_test(tokens, positions)
        .unwrap_or_else(|e| panic!("{label} batched: {e}"));
    assert_eq!(
        seq_logits.len(),
        batch_logits.len(),
        "{label} result count mismatch"
    );
    for m in 0..tokens.len() {
        let seq_top = argmax(&seq_logits[m]);
        let bat_top = argmax(&batch_logits[m]);
        assert_eq!(
            seq_top, bat_top,
            "{label} position {m}: batched argmax={bat_top} != sequential argmax={seq_top}"
        );
    }
}
#[test]
fn batched_argmax_matches_sequential_k4_k8() {
    let Some(mut engine) = load_engine() else {
        return;
    };
    check_argmax_parity(&mut engine, &[1u32, 315, 1012, 297], &[0, 1, 2, 3], "K=4");
    engine.reset_kv_for_test();
    check_argmax_parity(
        &mut engine,
        &[1u32, 315, 1012, 297, 338, 263, 1243, 310],
        &[0, 1, 2, 3, 4, 5, 6, 7],
        "K=8",
    );
}
