//! HIDE context and memory substrate (bible ch.04).
//!
//! This is the shell-side compiler described in `docs/hide-bible/04-*`: it
//! ranks sources, packs a token budget with a real reservation-aware knapsack,
//! and emits a replayable manifest. It also owns the hierarchical memory store
//! (SQLite/FTS5 + cosine vectors), the per-task context profiles, and the KV
//! reuse-banking seam to `hawking-serve`.

pub mod budget;
pub mod capability;
pub mod compiler;
pub mod embed;
pub mod fidelity;
pub mod kv;
pub mod manifest;
pub mod memory;
pub mod memory_classes;
pub mod memory_os;
pub mod personal_tools;
pub mod skill_foundry;
pub mod privacy;
pub mod profiles;
pub mod recall;
pub mod rot;
pub mod sources;

pub use budget::{estimate_tokens, RegionBudget, Reservations, TokenBudget, TokenCounter};
pub use capability::{
    CompactionMode, ContextCapability, CurvePoint, DeclaredNumber, NumberSource, RetrievalMode,
};
pub use compiler::{
    CompileInput, CompiledContext, ContextCandidate, ContextCompiler, ContextSource, RealizedSpan,
};
pub use embed::{cosine, EmbeddingClient, HashingEmbeddingClient, HttpEmbeddingClient};
pub use kv::{
    CheckpointId, CheckpointMeta, EvictionChoice as KvEvictionChoice, HttpKvStore, KvCheckpoint,
    KvHandle, KvStore, KvStoreClient, KvStoreStats, KvTier, PrefixHandle, PrefixKey,
    RestoredSession, SlotId, StubKvStore, WorkingSetBudget,
};
pub use manifest::{
    CompactionEvent, ContextManifest, ContextMeter, ContextSourceKind, ContextSpan, DropReason,
    DroppedContextSpan, ManifestLive, PinState, SpanSignals, WatermarkLevel,
};
pub use memory::{
    InMemoryMemoryStore, MemoryKind, MemoryQuery, MemoryRecord, MemoryStore, RankedMemory,
    ScoredMemory, SqliteMemoryStore,
};
pub use memory_classes::{
    ClassBudgets, ClassCompileRetrieval, ClassMemoryDraft, ClassMemoryRecord, ClassProvenance,
    ClassRetrievalSlice, ClassedMemorySystem, DynClassedMemory, EpisodicWriteCap, InspectFilter,
    MemoryClass, MemoryExport, PersonalScope, ProceduralWriteCap, ProjectWriteCap, ScopePromotion,
    TurnWriteCap, UserWriteCap, VerifierWriteCap, WriteAuthority,
};
pub use memory_os::{
    ConsolidateResult, InMemoryMemoryOs, MemoryExplain, MemoryItem, MemoryItemDraft,
    MemoryItemPatch, MemoryOs, MemoryOsError, MemoryOsQuery, MemoryTier, VerificationState,
};
pub use personal_tools::{
    execute_with_receipt, execute_without_receipt, LivePersonalTool,
    PermissionDecision as ToolPermissionDecision, PersonalTool, PersonalToolAbi,
    PersonalToolRegistry, ToolEffectClass, ToolExecuteResult, ToolPermissionGate, ToolPermissions,
    ToolProposal, ToolReceipt, ToolStatus,
};
pub use privacy::{
    ConnectorCapableHandle, EncryptedVaultHandle, EphemeralEndReport, NetworkCapableHandle,
    PrivacyBoundaryError, PrivacyMode, PrivacyPolicy, PrivacySession,
};
pub use profiles::{
    ContextProfile, EvictionChoice, KvPrecision, OrderingPolicy, PositionPolicy, SourceWeights,
    WorkingSetMode,
};
pub use rot::{detect_context_rot, ContextRotReport, RotSeverity, RotSignal, RotThresholds};
pub use skill_foundry::{
    example_skill_spec, AdmissionStage, ProtectedControllerCap, SandboxProposeCap, SkillCompatibility,
    SkillEnvironment, SkillFailureMode, SkillFoundry, SkillFoundryError, SkillIoField, SkillProvenance,
    SkillRecord, SkillSpec, SkillStatus, SkillStep, SkillTest, SkillVersion, StageReceipt,
};
