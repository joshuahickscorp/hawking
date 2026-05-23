/// One work item: one expert + the list of tokens routed to it, with
/// per-token gate weights.
#[derive(Debug, Clone)]
pub struct ExpertWorkItem {
    pub expert_id: u32,
    pub tokens: Vec<u32>,
    pub weights: Vec<f32>,
}

/// Group per-token (expert, weight) routes into per-expert buckets so
/// each expert only runs once per forward step.
pub fn build_work_queue(
    routes_per_token: &[Vec<(usize, f32)>],
    n_experts: usize,
) -> Vec<ExpertWorkItem> {
    let mut out: Vec<ExpertWorkItem> = (0..n_experts)
        .map(|e| ExpertWorkItem {
            expert_id: e as u32,
            tokens: Vec::new(),
            weights: Vec::new(),
        })
        .collect();
    for (token_idx, routes) in routes_per_token.iter().enumerate() {
        for &(eid, w) in routes {
            out[eid].tokens.push(token_idx as u32);
            out[eid].weights.push(w);
        }
    }
    // Drop empties so the GPU work-queue is dense.
    out.into_iter().filter(|w| !w.tokens.is_empty()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_work_queue_buckets_correctly() {
        let routes = vec![vec![(0, 0.5), (2, 0.5)], vec![(2, 1.0)], vec![(0, 1.0)]];
        let q = build_work_queue(&routes, 4);
        assert_eq!(q.len(), 2);
        let e0 = q.iter().find(|w| w.expert_id == 0).unwrap();
        let e2 = q.iter().find(|w| w.expert_id == 2).unwrap();
        assert_eq!(e0.tokens, vec![0, 2]);
        assert_eq!(e2.tokens, vec![0, 1]);
    }
}
