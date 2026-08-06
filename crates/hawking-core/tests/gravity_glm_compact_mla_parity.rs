use hawking_core::numeric_parity::{format_score_line, score_pair, Bounds, PairedScore, SCHEMA};
const TOKENS: usize = 11;
const HEADS: usize = 3;
const LATENT: usize = 17;
const NOPE: usize = 13;
const ROPE: usize = 4;
const VALUE: usize = 9;
const SELECTED_K: usize = 7;
#[derive(Debug)]
struct Fixture {
    latents: Vec<f32>,
    rope_keys: Vec<f32>,
    query_nope: Vec<f32>,
    query_rope: Vec<f32>,
    key_weight: Vec<f32>,
    value_weight: Vec<f32>,
    index_scores: Vec<f32>,
}
#[derive(Debug)]
struct F32Attention {
    /// `[head][selected-position-in-ascending-order]`.
    logits: Vec<f32>,
    /// `[head][value]`.
    context: Vec<f32>,
}
#[derive(Debug)]
struct F64Attention {
    logits: Vec<f64>,
    context: Vec<f64>,
}
fn deterministic_f32(mut state: u32, len: usize, scale: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let unit = ((state >> 8) as f32) * (1.0 / 8_388_608.0) - 1.0;
        out.push(unit * scale);
    }
    out
}
fn fixture() -> Fixture {
    Fixture {
        latents: deterministic_f32(0x1020_3040, TOKENS * LATENT, 0.8),
        rope_keys: deterministic_f32(0x5566_7788, TOKENS * ROPE, 0.6),
        query_nope: deterministic_f32(0x90ab_cdef, HEADS * NOPE, 0.7),
        query_rope: deterministic_f32(0x3141_5926, HEADS * ROPE, 0.5),
        key_weight: deterministic_f32(0x2718_2818, HEADS * NOPE * LATENT, 0.35),
        value_weight: deterministic_f32(0xdead_beef, HEADS * VALUE * LATENT, 0.4),
        index_scores: vec![
            0.31, -0.20, 0.88, 0.31, 0.05, 0.74, 0.99, -0.50, 0.44, 0.66, 0.12,
        ],
    }
}
fn stable_topk_f32(values: &[f32], k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    indices.truncate(k.min(indices.len()));
    indices
}
fn stable_topk_f64(values: &[f64], k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    indices.truncate(k.min(indices.len()));
    indices
}
fn selected_ascending(selected_ranked: &[usize]) -> Vec<usize> {
    let mut ascending = selected_ranked.to_vec();
    ascending.sort_unstable();
    ascending
}
fn softmax_f32(logits: &[f32], accumulation_order: &[usize]) -> Vec<f32> {
    let mut best = f32::NEG_INFINITY;
    for &slot in accumulation_order {
        best = best.max(logits[slot]);
    }
    let exp: Vec<f32> = logits.iter().map(|&score| (score - best).exp()).collect();
    let mut total = 0.0f32;
    for &slot in accumulation_order {
        total += exp[slot];
    }
    exp.into_iter().map(|value| value / total).collect()
}
fn softmax_f64(logits: &[f64], accumulation_order: &[usize]) -> Vec<f64> {
    let mut best = f64::NEG_INFINITY;
    for &slot in accumulation_order {
        best = best.max(logits[slot]);
    }
    let exp: Vec<f64> = logits.iter().map(|&score| (score - best).exp()).collect();
    let mut total = 0.0f64;
    for &slot in accumulation_order {
        total += exp[slot];
    }
    exp.into_iter().map(|value| value / total).collect()
}
fn expanded_f32(f: &Fixture, selected_ranked: &[usize]) -> F32Attention {
    let selected = selected_ascending(selected_ranked);
    let canonical_slots: Vec<usize> = (0..selected.len()).collect();
    let scale = ((NOPE + ROPE) as f32).powf(-0.5);
    let mut logits = vec![0.0f32; HEADS * selected.len()];
    let mut context = vec![0.0f32; HEADS * VALUE];
    for head in 0..HEADS {
        let mut expanded_keys = vec![0.0f32; selected.len() * NOPE];
        let mut expanded_values = vec![0.0f32; selected.len() * VALUE];
        for (slot, &token) in selected.iter().enumerate() {
            for key_dim in 0..NOPE {
                let mut acc = 0.0f32;
                for latent_dim in 0..LATENT {
                    let w = f.key_weight[(head * NOPE + key_dim) * LATENT + latent_dim];
                    acc += w * f.latents[token * LATENT + latent_dim];
                }
                expanded_keys[slot * NOPE + key_dim] = acc;
            }
            for value_dim in 0..VALUE {
                let mut acc = 0.0f32;
                for latent_dim in 0..LATENT {
                    let w = f.value_weight[(head * VALUE + value_dim) * LATENT + latent_dim];
                    acc += w * f.latents[token * LATENT + latent_dim];
                }
                expanded_values[slot * VALUE + value_dim] = acc;
            }
        }
        for (slot, &token) in selected.iter().enumerate() {
            let mut dot = 0.0f32;
            for key_dim in 0..NOPE {
                dot += f.query_nope[head * NOPE + key_dim] * expanded_keys[slot * NOPE + key_dim];
            }
            for rope_dim in 0..ROPE {
                dot += f.query_rope[head * ROPE + rope_dim] * f.rope_keys[token * ROPE + rope_dim];
            }
            logits[head * selected.len() + slot] = dot * scale;
        }
        let head_logits = &logits[head * selected.len()..(head + 1) * selected.len()];
        let probabilities = softmax_f32(head_logits, &canonical_slots);
        for slot in canonical_slots.iter().copied() {
            let probability = probabilities[slot];
            for value_dim in 0..VALUE {
                context[head * VALUE + value_dim] +=
                    probability * expanded_values[slot * VALUE + value_dim];
            }
        }
    }
    F32Attention { logits, context }
}
fn compact_absorbed_f32(
    f: &Fixture,
    selected_ranked: &[usize],
    accumulate_in_rank_order: bool,
) -> F32Attention {
    let selected = selected_ascending(selected_ranked);
    let accumulation_positions = if accumulate_in_rank_order {
        selected_ranked.to_vec()
    } else {
        selected.clone()
    };
    let accumulation_slots: Vec<usize> = accumulation_positions
        .iter()
        .map(|position| {
            selected
                .binary_search(position)
                .expect("ranked selection must exist in ascending selection")
        })
        .collect();
    let scale = ((NOPE + ROPE) as f32).powf(-0.5);
    let mut logits = vec![0.0f32; HEADS * selected.len()];
    let mut context = vec![0.0f32; HEADS * VALUE];
    for head in 0..HEADS {
        let mut query_latent = vec![0.0f32; LATENT];
        for latent_dim in 0..LATENT {
            let mut acc = 0.0f32;
            for key_dim in 0..NOPE {
                let w = f.key_weight[(head * NOPE + key_dim) * LATENT + latent_dim];
                acc += w * f.query_nope[head * NOPE + key_dim];
            }
            query_latent[latent_dim] = acc;
        }
        for (slot, &token) in selected.iter().enumerate() {
            let mut dot = 0.0f32;
            for latent_dim in 0..LATENT {
                dot += query_latent[latent_dim] * f.latents[token * LATENT + latent_dim];
            }
            for rope_dim in 0..ROPE {
                dot += f.query_rope[head * ROPE + rope_dim] * f.rope_keys[token * ROPE + rope_dim];
            }
            logits[head * selected.len() + slot] = dot * scale;
        }
        let head_logits = &logits[head * selected.len()..(head + 1) * selected.len()];
        let probabilities = softmax_f32(head_logits, &accumulation_slots);
        let mut weighted_latent = vec![0.0f32; LATENT];
        for slot in accumulation_slots.iter().copied() {
            let token = selected[slot];
            let probability = probabilities[slot];
            for latent_dim in 0..LATENT {
                weighted_latent[latent_dim] += probability * f.latents[token * LATENT + latent_dim];
            }
        }
        for value_dim in 0..VALUE {
            let mut acc = 0.0f32;
            for latent_dim in 0..LATENT {
                let w = f.value_weight[(head * VALUE + value_dim) * LATENT + latent_dim];
                acc += w * weighted_latent[latent_dim];
            }
            context[head * VALUE + value_dim] = acc;
        }
    }
    F32Attention { logits, context }
}
fn expanded_f64_authority(f: &Fixture, selected_ranked: &[usize]) -> F64Attention {
    let selected = selected_ascending(selected_ranked);
    let canonical_slots: Vec<usize> = (0..selected.len()).collect();
    let scale = ((NOPE + ROPE) as f64).powf(-0.5);
    let mut logits = vec![0.0f64; HEADS * selected.len()];
    let mut context = vec![0.0f64; HEADS * VALUE];
    for head in 0..HEADS {
        let mut expanded_values = vec![0.0f64; selected.len() * VALUE];
        for (slot, &token) in selected.iter().enumerate() {
            let mut dot = 0.0f64;
            for key_dim in 0..NOPE {
                let mut key = 0.0f64;
                for latent_dim in 0..LATENT {
                    let w = f.key_weight[(head * NOPE + key_dim) * LATENT + latent_dim] as f64;
                    key += w * f.latents[token * LATENT + latent_dim] as f64;
                }
                dot += f.query_nope[head * NOPE + key_dim] as f64 * key;
            }
            for rope_dim in 0..ROPE {
                dot += f.query_rope[head * ROPE + rope_dim] as f64
                    * f.rope_keys[token * ROPE + rope_dim] as f64;
            }
            logits[head * selected.len() + slot] = dot * scale;
            for value_dim in 0..VALUE {
                let mut value = 0.0f64;
                for latent_dim in 0..LATENT {
                    let w = f.value_weight[(head * VALUE + value_dim) * LATENT + latent_dim] as f64;
                    value += w * f.latents[token * LATENT + latent_dim] as f64;
                }
                expanded_values[slot * VALUE + value_dim] = value;
            }
        }
        let head_logits = &logits[head * selected.len()..(head + 1) * selected.len()];
        let probabilities = softmax_f64(head_logits, &canonical_slots);
        for slot in canonical_slots.iter().copied() {
            for value_dim in 0..VALUE {
                context[head * VALUE + value_dim] +=
                    probabilities[slot] * expanded_values[slot * VALUE + value_dim];
            }
        }
    }
    F64Attention { logits, context }
}
fn compact_absorbed_f64(f: &Fixture, selected_ranked: &[usize]) -> F64Attention {
    let selected = selected_ascending(selected_ranked);
    let canonical_slots: Vec<usize> = (0..selected.len()).collect();
    let scale = ((NOPE + ROPE) as f64).powf(-0.5);
    let mut logits = vec![0.0f64; HEADS * selected.len()];
    let mut context = vec![0.0f64; HEADS * VALUE];
    for head in 0..HEADS {
        let mut query_latent = vec![0.0f64; LATENT];
        for latent_dim in 0..LATENT {
            for key_dim in 0..NOPE {
                let w = f.key_weight[(head * NOPE + key_dim) * LATENT + latent_dim] as f64;
                query_latent[latent_dim] += w * f.query_nope[head * NOPE + key_dim] as f64;
            }
        }
        for (slot, &token) in selected.iter().enumerate() {
            let mut dot = 0.0f64;
            for latent_dim in 0..LATENT {
                dot += query_latent[latent_dim] * f.latents[token * LATENT + latent_dim] as f64;
            }
            for rope_dim in 0..ROPE {
                dot += f.query_rope[head * ROPE + rope_dim] as f64
                    * f.rope_keys[token * ROPE + rope_dim] as f64;
            }
            logits[head * selected.len() + slot] = dot * scale;
        }
        let head_logits = &logits[head * selected.len()..(head + 1) * selected.len()];
        let probabilities = softmax_f64(head_logits, &canonical_slots);
        let mut weighted_latent = vec![0.0f64; LATENT];
        for slot in canonical_slots.iter().copied() {
            let token = selected[slot];
            for latent_dim in 0..LATENT {
                weighted_latent[latent_dim] +=
                    probabilities[slot] * f.latents[token * LATENT + latent_dim] as f64;
            }
        }
        for value_dim in 0..VALUE {
            for latent_dim in 0..LATENT {
                let w = f.value_weight[(head * VALUE + value_dim) * LATENT + latent_dim] as f64;
                context[head * VALUE + value_dim] += w * weighted_latent[latent_dim];
            }
        }
    }
    F64Attention { logits, context }
}
fn max_abs_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}
fn score_attention(
    expanded: &F32Attention,
    compact: &F32Attention,
    authority: &F64Attention,
) -> (Vec<PairedScore>, PairedScore) {
    let mut score_bounds = Bounds::logits();
    score_bounds.top_k = 3;
    let mut score_pairs = Vec::with_capacity(HEADS);
    for head in 0..HEADS {
        let range = head * SELECTED_K..(head + 1) * SELECTED_K;
        score_pairs.push(score_pair(
            &expanded.logits[range.clone()],
            &compact.logits[range.clone()],
            &authority.logits[range],
            &score_bounds,
        ));
    }
    let mut context_bounds = Bounds::continuous_only();
    context_bounds.top_k = 5;
    let context_pair = score_pair(
        &expanded.context,
        &compact.context,
        &authority.context,
        &context_bounds,
    );
    (score_pairs, context_pair)
}
#[test]
fn compact_absorbed_mla_requires_ranked_selected_accumulation_under_v2_1() {
    let f = fixture();
    let index_authority: Vec<f64> = f.index_scores.iter().map(|&value| value as f64).collect();
    let selected_authority = stable_topk_f64(&index_authority, SELECTED_K);
    let selected_expanded = stable_topk_f32(&f.index_scores, SELECTED_K);
    let selected_compact = stable_topk_f32(&f.index_scores, SELECTED_K);
    assert_eq!(selected_authority, vec![6, 2, 5, 9, 8, 0, 3]);
    assert_eq!(selected_expanded, selected_authority);
    assert_eq!(selected_compact, selected_authority);
    assert_eq!(
        selected_ascending(&selected_authority),
        vec![0, 2, 3, 5, 6, 8, 9]
    );
    let mut index_bounds = Bounds::logits();
    index_bounds.top_k = SELECTED_K;
    let index_pair = score_pair(
        &f.index_scores,
        &f.index_scores,
        &index_authority,
        &index_bounds,
    );
    assert!(index_pair.pass, "index decision gate: {index_pair:?}");
    let authority_expanded = expanded_f64_authority(&f, &selected_authority);
    let authority_compact = compact_absorbed_f64(&f, &selected_authority);
    let f64_logits_abs = max_abs_f64(&authority_expanded.logits, &authority_compact.logits);
    let f64_context_abs = max_abs_f64(&authority_expanded.context, &authority_compact.context);
    assert!(
        f64_logits_abs <= 1e-15,
        "absorbed K is not algebraically equivalent in f64: {f64_logits_abs:.3e}"
    );
    assert!(
        f64_context_abs <= 1e-15,
        "absorbed V is not algebraically equivalent in f64: {f64_context_abs:.3e}"
    );
    let expanded = expanded_f32(&f, &selected_authority);
    let compact_source_order = compact_absorbed_f32(&f, &selected_authority, false);
    let compact_rank_order = compact_absorbed_f32(&f, &selected_authority, true);
    let (score_pairs, context_pair) =
        score_attention(&expanded, &compact_source_order, &authority_expanded);
    let (_, rank_order_context_pair) =
        score_attention(&expanded, &compact_rank_order, &authority_expanded);
    for (head, pair) in score_pairs.iter().enumerate() {}
    assert!(
        score_pairs.iter().all(|pair| pair.pass),
        "attention-score V2.1 gate failed: {score_pairs:#?}"
    );
    assert!(
        !context_pair.pass
            && context_pair
                .device
                .failures
                .iter()
                .any(|failure| failure.contains("meaningful_rel")),
        "ascending-position compact accumulation is the expected negative witness: \
         {context_pair:#?}"
    );
    assert!(
        rank_order_context_pair.pass,
        "DSA-rank compact accumulation must pass V2.1: {rank_order_context_pair:#?}"
    );
    assert!(
        score_pairs
            .iter()
            .all(|pair| pair.host.discrete.top_k_exact_match
                && pair.device.discrete.top_k_exact_match),
        "attention top-k decisions must be exact"
    );
    assert!(
        rank_order_context_pair.device.discrete.greedy_match
            && rank_order_context_pair.device.discrete.top_k_exact_match,
        "rank-order context decisions must be exact"
    );
    assert_ne!(compact_source_order.context, expanded.context);
    assert_ne!(compact_rank_order.context, compact_source_order.context);
}
