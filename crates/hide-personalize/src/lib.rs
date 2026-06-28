//! Personalization and self-improvement backend (HIDE bible chapter 11).
//!
//! Real backend logic for the ch.11 bleeding-edge capabilities, staged after the
//! shell. What is REAL here (vs. a clean seam):
//!
//!   * **records** — the §11.1.1 capture record with blake3 `[u8;32]` hashes,
//!     microsecond timestamps, and constructors for all four outcomes.
//!   * **store / curate** — scrub-on-write secret redaction (real
//!     [`hide_security::Redactor`]), the `dataset/vNNN` layout, and the full
//!     §11.1.2 curation pipeline (p95×3 latency outliers + recency weighting).
//!   * **eval** — oracles that actually execute (a `Command` oracle spawns a
//!     real process; `Regex`/`GoldenDiff` evaluate real output) + an
//!     [`eval::EvalMiner`] that mines functions-without-tests from the code index.
//!   * **rlef** — reward **derived** from execution outcomes (not supplied),
//!     GRPO group-relative advantage, and a daemon + PPL-gate seam.
//!   * **retrieval** — the [`retrieval::MetaRouter`] trait + a real ε-greedy /
//!     online-SGD router over the code index.
//!   * **kv_handoff** — the §11.5 `KvShareGroup` protocol with a clean seam to
//!     the in-tree `copy_kv_prefix_to_slot`.
//!
//! Seams (post-shell): the actual LoRA gradient step (Hawking Condense), the PPL
//! forward pass, and the runtime KV block copy are trait seams, not faked.

pub mod curate;
pub mod eval;
pub mod kv_handoff;
pub mod prompts;
pub mod records;
pub mod retrieval;
pub mod rlef;
pub mod store;
pub mod world;

pub use records::{Hash32, Outcome, PersonalizationRecord, TaskClass};
pub use store::{
    scrub_record, DynPersonalizationStore, InMemoryPersonalizationStore,
    JsonlPersonalizationStore, PersonalLayout, PersonalizationStore,
};

pub use curate::{curate, write_dataset, CuratedDataset, CurationPolicy};
pub use eval::{
    run_eval, run_suite, AdapterGateReport, CandidateStatus, EvalCase, EvalMiner, EvalMinerConfig,
    EvalOracle, EvalResult, EvalTaskCandidate,
};
pub use kv_handoff::{
    copy_for_group, AgentId, BroadcastReport, GenerateRequest, KvHandle, KvKey, KvPrefixCopier,
    KvShareGroup,
};
pub use retrieval::{
    route_and_search, EpsilonGreedyRouter, LearnedRetrievalWeights, MetaRouter, QueryType,
    RetrievalOutcomeRecord, RetrievalStrategy,
};
pub use rlef::{
    assemble_dataset, assemble_group, ppl_gate, reward_for, Attempt, ExecutionOutcome,
    FeedbackSignal, GateOutcome, PplEvaluator, RewardConfig, RlefConfig, RlefDaemon, TaskGroup,
    TrainingTuple,
};
