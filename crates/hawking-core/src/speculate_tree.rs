//! Event Horizon Phase 6 — token-tree builder + CPU linear-fallback verifier.
//! Metal single-pass kernel = documented TODO; supports_tree_verify()=false.

use crate::speculate::proposal::{Budget, Proposal};
use crate::speculate::verifier::{ExactTarget, Verifier, VerifyOutcome};
use crate::Result;

/// A single node in a token tree.
#[derive(Debug, Clone)]
pub struct TreeNode {
    pub token: u32,
    pub parent: usize, // usize::MAX for root
    pub depth: usize,  // root=0
}

/// Builds a token tree incrementally. The tree stores parent links so that
/// ancestor-mask computation and path extraction are O(depth) per node.
#[derive(Debug, Clone)]
pub struct TokenTreeBuilder {
    nodes: Vec<TreeNode>,
    children: Vec<Vec<usize>>,
}

impl TokenTreeBuilder {
    /// Create a new tree with a single root node at depth 0.
    pub fn new(root_token: u32) -> Self {
        let root = TreeNode { token: root_token, parent: usize::MAX, depth: 0 };
        Self { nodes: vec![root], children: vec![Vec::new()] }
    }

    /// Attach a child token under `parent_idx`. Returns the new node's index.
    ///
    /// # Panics
    /// Panics if `parent_idx >= self.len()`.
    pub fn add_child(&mut self, parent_idx: usize, token: u32) -> usize {
        let depth = self.nodes[parent_idx].depth + 1;
        let new_idx = self.nodes.len();
        self.nodes.push(TreeNode { token, parent: parent_idx, depth });
        self.children.push(Vec::new());
        self.children[parent_idx].push(new_idx);
        new_idx
    }

    /// Total number of nodes in the tree (including the root).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True iff the tree contains no nodes. In practice this is never true after
    /// `new()`, but satisfies the `is_empty` lint convention.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Depth of a node (root == 0).
    pub fn depth_of(&self, idx: usize) -> usize {
        self.nodes[idx].depth
    }

    /// Indices of all leaf nodes (nodes with no children).
    pub fn leaves(&self) -> Vec<usize> {
        (0..self.nodes.len()).filter(|&i| self.children[i].is_empty()).collect()
    }

    /// Token sequence along the path from the root to `idx` (inclusive).
    /// The root token is at position 0; `nodes[idx].token` is the last element.
    pub fn path_from_root(&self, idx: usize) -> Vec<u32> {
        let mut path = Vec::new();
        let mut cur = idx;
        loop {
            path.push(self.nodes[cur].token);
            let p = self.nodes[cur].parent;
            if p == usize::MAX {
                break;
            }
            cur = p;
        }
        path.reverse();
        path
    }

    /// 64-bit ancestor mask for node `idx`.
    /// Bit `j` is set iff node `j` lies on the root-to-`idx` path (inclusive of
    /// both endpoints). Nodes with index >= 64 do not set a bit (caller must keep
    /// trees shallow or use the Metal tree-verify path which handles wider masks).
    pub fn ancestor_mask_at(&self, idx: usize) -> u64 {
        let mut mask: u64 = 0;
        let mut cur = idx;
        loop {
            if cur < 64 {
                mask |= 1u64 << cur;
            }
            let p = self.nodes[cur].parent;
            if p == usize::MAX {
                break;
            }
            cur = p;
        }
        mask
    }

    /// Ancestor masks for every node, in node-index order.
    pub fn ancestor_masks(&self) -> Vec<u64> {
        (0..self.nodes.len()).map(|i| self.ancestor_mask_at(i)).collect()
    }

    /// Per-node position ids, defined as each node's depth. The root is at
    /// `bonus_pos` in the sequence; callers must add the base offset externally.
    pub fn position_ids(&self) -> Vec<usize> {
        self.nodes.iter().map(|n| n.depth).collect()
    }

    /// Serialize the tree into a `Proposal::TokenTree` for the verifier dispatch.
    pub fn to_proposal(&self) -> Proposal {
        Proposal::TokenTree { nodes: self.nodes.iter().map(|n| n.token).collect(), ancestor_mask: self.ancestor_masks(), position_ids: self.position_ids() }
    }
}

/// CPU fallback: verify each root→leaf path via `Verifier::verify_line` and
/// return the `VerifyOutcome` with the highest acceptance score.
///
/// The root token (`nodes[0]`) IS the bonus token passed to `verify_line`; the
/// draft for a leaf is the path tokens starting at index 1.
///
/// # Metal TODO
/// Replace with a single GPU dispatch using `ancestor_mask` once
/// `ExactTarget::supports_tree_verify()` returns `true`.
pub fn verify_tree_cpu<T: ExactTarget>(tree: &TokenTreeBuilder, verifier: &Verifier, target: &mut T, bonus: u32, bonus_pos: usize) -> Result<VerifyOutcome> {
    let leaves = tree.leaves();
    if leaves.is_empty() {
        return verifier.verify_line(target, bonus, bonus_pos, &[]);
    }
    let mut best: Option<VerifyOutcome> = None;
    let mut best_score = 0usize;
    for leaf_idx in leaves {
        let full_path = tree.path_from_root(leaf_idx);
        // path[0] is the root token, which equals `bonus` — skip it for draft.
        let draft = &full_path[1..];
        let outcome = verifier.verify_line(target, bonus, bonus_pos, draft)?;
        let score = outcome.accepted.len() + if outcome.correction.is_some() { 1 } else { 0 };
        if score > best_score || best.is_none() {
            best_score = score;
            best = Some(outcome);
        }
    }
    Ok(best.unwrap())
}

/// Build a degenerate single-branch tree from a linear token sequence.
/// `tokens[0]` becomes the root; subsequent tokens are chained as
/// single children. Returns an empty tree (root only with `tokens[0]`)
/// when `tokens` is empty — callers should prefer passing a non-empty
/// slice (at least the bonus token as root).
///
/// # Panics
/// Panics if `tokens` is empty (there is no meaningful root token).
pub fn build_line_as_tree(tokens: &[u32]) -> TokenTreeBuilder {
    assert!(!tokens.is_empty(), "build_line_as_tree: tokens must not be empty");
    let mut tree = TokenTreeBuilder::new(tokens[0]);
    let mut parent = 0usize;
    for &tok in &tokens[1..] {
        parent = tree.add_child(parent, tok);
    }
    tree
}

/// Compute per-branch draft depth given a total budget and a desired branching
/// width. Always returns at least 1.
pub fn tree_draft_k(budget: Budget, width: usize) -> usize {
    (budget.k / width.max(1)).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Result;

    // ------------------------------------------------------------------
    // Minimal mock target for tests — no Metal, no model.
    // `preds` is the canned argmax sequence returned from
    // forward_tokens_verify (first n elements) or forward_token_greedy
    // (element 0).
    // ------------------------------------------------------------------
    struct MockTarget {
        preds: Vec<u32>,
    }

    impl ExactTarget for MockTarget {
        fn forward_tokens_verify(&mut self, tokens: &[u32], _positions: &[usize]) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
            let n = tokens.len();
            Ok((self.preds[..n].to_vec(), vec![Vec::new(); n]))
        }

        fn forward_token_greedy(&mut self, _token: u32, _pos: usize) -> Result<u32> {
            Ok(self.preds[0])
        }
    }

    // ------------------------------------------------------------------
    // Test 1: two-branch tree builds correct ancestor masks.
    //
    // Tree layout (node index in parens):
    //   root(0, tok=100)
    //     left(1, tok=10)
    //       left2(2, tok=11)
    //     right(3, tok=20)
    //
    // Expected ancestor masks:
    //   masks[0] = 0b0001  (only itself)
    //   masks[1] = 0b0011  (root + self)
    //   masks[2] = 0b0111  (root + left + self)
    //   masks[3] = 0b1001  (root + self)
    // ------------------------------------------------------------------
    #[test]
    fn two_branch_tree_builds_correct_ancestor_masks() {
        let mut tree = TokenTreeBuilder::new(100);
        let left = tree.add_child(0, 10);
        let _left2 = tree.add_child(left, 11);
        let _right = tree.add_child(0, 20);

        let masks = tree.ancestor_masks();
        assert_eq!(masks.len(), 4);
        assert_eq!(masks[0], 0b0001, "root mask");
        assert_eq!(masks[1], 0b0011, "left mask");
        assert_eq!(masks[2], 0b0111, "left2 mask");
        assert_eq!(masks[3], 0b1001, "right mask");
    }

    // ------------------------------------------------------------------
    // Test 2: position_ids equal the node depth.
    // ------------------------------------------------------------------
    #[test]
    fn position_ids_equal_depth() {
        let mut tree = TokenTreeBuilder::new(1);
        let a = tree.add_child(0, 2);
        let b = tree.add_child(a, 3);
        let _c = tree.add_child(0, 4); // second branch

        let pos = tree.position_ids();
        assert_eq!(pos[0], 0, "root depth");
        assert_eq!(pos[a], 1);
        assert_eq!(pos[b], 2);
        // The second branch child of root is also depth 1.
        assert_eq!(pos[3], 1);
    }

    // ------------------------------------------------------------------
    // Test 3: leaves_are_correct.
    //
    //   root(0) → a(1) → b(2)
    //           → c(3)
    //
    // Leaves = {2, 3} (nodes with no children).
    // ------------------------------------------------------------------
    #[test]
    fn leaves_are_correct() {
        let mut tree = TokenTreeBuilder::new(0);
        let a = tree.add_child(0, 1);
        let _b = tree.add_child(a, 2);
        let _c = tree.add_child(0, 3);

        let mut leaves = tree.leaves();
        leaves.sort();
        assert_eq!(leaves, vec![2, 3]);
    }

    // ------------------------------------------------------------------
    // Test 4: cpu_fallback_accepts_matching_branch.
    //
    // Tree: root(99) → left(10) → left2(20)
    //                → right(50)
    //
    // MockTarget preds = [10, 99]: forward_tokens_verify returns the first
    // n preds, so for a 2-token verify call it returns [10, 99].
    //
    // Left-branch path (root=99, draft=[10,20]):
    //   vtoks = [99, 10] at positions [pos, pos+1]
    //   preds = [10, 99] → draft[0]=10 matches preds[0]=10  (accepted)
    //                     → draft[1]=20 ≠ preds[1]=99        (rejected, correction=99)
    //   score = 1 accepted + 1 correction = 2
    //
    // Right-branch path (root=99, draft=[50]):
    //   vtoks = [99] at [pos]  → preds = [10]
    //   draft[0]=50 ≠ preds[0]=10  (rejected immediately, correction=10)
    //   score = 0 accepted + 1 correction = 1
    //
    // Best = left branch (score 2): accepted=[10], correction=Some(99).
    // ------------------------------------------------------------------
    #[test]
    fn cpu_fallback_accepts_matching_branch() {
        let mut tree = TokenTreeBuilder::new(99);
        let left = tree.add_child(0, 10);
        let _left2 = tree.add_child(left, 20);
        let _right = tree.add_child(0, 50);

        let mut target = MockTarget { preds: vec![10, 99] };
        let verifier = Verifier::default();
        let outcome = verify_tree_cpu(&tree, &verifier, &mut target, 99, 5).unwrap();

        assert_eq!(outcome.accepted, vec![10], "left branch token accepted");
        assert_eq!(outcome.correction, Some(99), "correction from second preds slot");
    }

    // ------------------------------------------------------------------
    // Test 5: path_from_root_is_correct.
    //
    // root(tok=1) → a(tok=2) → b(tok=3)
    // path from b = [1, 2, 3]
    // ------------------------------------------------------------------
    #[test]
    fn path_from_root_is_correct() {
        let mut tree = TokenTreeBuilder::new(1);
        let a = tree.add_child(0, 2);
        let b = tree.add_child(a, 3);

        assert_eq!(tree.path_from_root(0), vec![1]);
        assert_eq!(tree.path_from_root(a), vec![1, 2]);
        assert_eq!(tree.path_from_root(b), vec![1, 2, 3]);
    }

    // ------------------------------------------------------------------
    // Test 6: to_proposal produces the TokenTree variant with correct fields.
    // ------------------------------------------------------------------
    #[test]
    fn to_proposal_produces_token_tree_variant() {
        let mut tree = TokenTreeBuilder::new(7);
        let a = tree.add_child(0, 8);
        let _b = tree.add_child(a, 9);

        let proposal = tree.to_proposal();
        match proposal {
            Proposal::TokenTree { nodes, ancestor_mask, position_ids } => {
                assert_eq!(nodes, vec![7, 8, 9]);
                assert_eq!(position_ids, vec![0, 1, 2]);
                // masks[0]=0b001, masks[1]=0b011, masks[2]=0b111
                assert_eq!(ancestor_mask[0], 0b001);
                assert_eq!(ancestor_mask[1], 0b011);
                assert_eq!(ancestor_mask[2], 0b111);
            }
            _ => panic!("expected Proposal::TokenTree"),
        }
    }

    // ------------------------------------------------------------------
    // Extra: build_line_as_tree creates a correct single-branch tree.
    // ------------------------------------------------------------------
    #[test]
    fn build_line_as_tree_is_single_branch() {
        let tree = build_line_as_tree(&[1, 2, 3, 4]);
        assert_eq!(tree.len(), 4);
        // Only the last node is a leaf.
        assert_eq!(tree.leaves(), vec![3]);
        assert_eq!(tree.path_from_root(3), vec![1, 2, 3, 4]);
        // All position ids should equal depth == index (linear chain).
        assert_eq!(tree.position_ids(), vec![0, 1, 2, 3]);
    }

    // ------------------------------------------------------------------
    // Extra: tree_draft_k respects minimum-1 rule and division.
    // ------------------------------------------------------------------
    #[test]
    fn tree_draft_k_basic() {
        let budget = Budget::line(6);
        assert_eq!(tree_draft_k(budget, 2), 3);
        assert_eq!(tree_draft_k(budget, 7), 1); // floor(6/7)=0 → clamp to 1
        assert_eq!(tree_draft_k(budget, 0), 6); // width.max(1)=1
    }
}
