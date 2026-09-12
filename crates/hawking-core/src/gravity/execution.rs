//! Provider-neutral semantic graph and legal execution-region validation.
//!
//! Gravity may change physical boundaries, but it may not infer semantics from
//! model or organ names.  Callers supply exact operations and effects.  This
//! module validates a proposed region and its backend schedule without claiming
//! that a device executed it or that its numerical/capability contract passed.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use thiserror::Error;

pub const SCHEMA: &str = "hawking.gravity.execution_region.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum StageAnnotation {
    Load,
    RepresentationDecode,
    AutoregressiveDecode,
    Route,
    Project,
    Accumulate,
    StateUpdate,
    Sample,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Backend {
    Cpu,
    Metal,
    Ane,
    Fpga,
    ExternalAccelerator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Ownership {
    Exclusive,
    Shared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum StateLifetime {
    Region,
    Invocation,
    Session,
    Resident,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Visibility {
    Operation,
    Region,
    Device,
    Host,
}

impl Access {
    fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NumericalPolicy {
    Exact,
    Tolerated { contract_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct StateEffect {
    pub resource: String,
    pub access: Access,
    pub alias_group: Option<String>,
    pub ownership: Ownership,
    pub lifetime: StateLifetime,
    pub loop_carried: bool,
    pub visibility: Visibility,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Operation {
    pub id: String,
    pub semantic_kind: String,
    pub stages: BTreeSet<StageAnnotation>,
    pub inputs: BTreeSet<String>,
    pub outputs: BTreeSet<String>,
    pub control_dependencies: BTreeSet<String>,
    pub state_effects: Vec<StateEffect>,
    pub randomness_state: Option<String>,
    pub numerical_policy: NumericalPolicy,
    pub supported_backends: BTreeSet<Backend>,
    pub scratch_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticGraph {
    pub graph_id: String,
    pub operations: Vec<Operation>,
    pub external_inputs: BTreeSet<String>,
    pub graph_outputs: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RegionBoundary {
    pub inputs: BTreeSet<String>,
    pub outputs: BTreeSet<String>,
    pub outside_consumers: BTreeSet<String>,
    pub control_inputs: BTreeSet<String>,
    pub control_outputs: BTreeSet<String>,
    pub outside_control_consumers: BTreeSet<String>,
    pub state_reads: BTreeSet<String>,
    pub state_writes: BTreeSet<String>,
    pub alias_groups: BTreeSet<String>,
    pub state_effects: BTreeSet<StateEffect>,
    pub randomness_states: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionCandidate {
    pub region_id: String,
    pub operation_ids: BTreeSet<String>,
    pub boundary: RegionBoundary,
    pub backend: Backend,
    pub numerical_policy: NumericalPolicy,
    pub available_scratch_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendProgram {
    pub program_id: String,
    pub backend: Backend,
    pub operation_ids: BTreeSet<String>,
    /// Earlier backend programs whose writes must be visible before this starts.
    pub waits_for: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedSchedule {
    pub region_id: String,
    pub programs: Vec<BackendProgram>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidatedRegion {
    pub schema: String,
    pub graph_id: String,
    pub semantic_graph_sha256: String,
    pub region: RegionCandidate,
    pub schedule: SelectedSchedule,
    pub required_scratch_bytes: u64,
    pub qualification: String,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ValidationError {
    #[error("duplicate operation id {0:?}")]
    DuplicateOperation(String),
    #[error("value {value:?} has multiple producers ({first:?}, {second:?})")]
    DuplicateProducer {
        value: String,
        first: String,
        second: String,
    },
    #[error("operation {operation:?} input {value:?} has no producer or external declaration")]
    MissingProducer { operation: String, value: String },
    #[error("operation {operation:?} has unknown control dependency {dependency:?}")]
    UnknownControlDependency {
        operation: String,
        dependency: String,
    },
    #[error("graph output {0:?} has no producer or external declaration")]
    UnknownGraphOutput(String),
    #[error("semantic graph contains a dependency cycle")]
    DependencyCycle,
    #[error("region is empty")]
    EmptyRegion,
    #[error("region references unknown operation {0:?}")]
    UnknownRegionOperation(String),
    #[error("region is non-convex: selected operation {from:?} reaches {to:?} through an outside consumer")]
    NonConvexRegion { from: String, to: String },
    #[error("region boundary field {field} differs: declared={declared:?}, actual={actual:?}")]
    BoundaryMismatch {
        field: &'static str,
        declared: BTreeSet<String>,
        actual: BTreeSet<String>,
    },
    #[error("region boundary state effects differ: declared={declared:?}, actual={actual:?}")]
    StateEffectBoundaryMismatch {
        declared: BTreeSet<StateEffect>,
        actual: BTreeSet<StateEffect>,
    },
    #[error("operation {operation:?} does not support backend {backend:?}")]
    UnsupportedBackend { operation: String, backend: Backend },
    #[error("operation {operation:?} numerical policy differs from the region policy")]
    NumericalPolicyMismatch { operation: String },
    #[error("operation {operation:?} has an invalid state effect for {resource:?}: {reason}")]
    InvalidStateEffect {
        operation: String,
        resource: String,
        reason: &'static str,
    },
    #[error(
        "operation {operation:?} randomness state {resource:?} lacks a read-write state effect"
    )]
    RandomnessStateUnaccounted { operation: String, resource: String },
    #[error("required scratch bytes overflow")]
    ScratchOverflow,
    #[error("region needs {required} scratch bytes but only {available} are available")]
    ScratchLimit { required: u64, available: u64 },
    #[error("schedule region id {schedule:?} does not match candidate {candidate:?}")]
    ScheduleRegionMismatch { schedule: String, candidate: String },
    #[error("backend program {0:?} has no operations")]
    EmptyProgram(String),
    #[error("backend program {program:?} uses {actual:?}, region selected {expected:?}")]
    ProgramBackendMismatch {
        program: String,
        actual: Backend,
        expected: Backend,
    },
    #[error("duplicate backend program id {0:?}")]
    DuplicateProgram(String),
    #[error("backend program {program:?} waits for unknown or non-earlier program {dependency:?}")]
    InvalidProgramWait { program: String, dependency: String },
    #[error("operation {operation:?} runs before dependency {dependency:?}")]
    ProgramDependencyOrder {
        operation: String,
        dependency: String,
    },
    #[error("program {program:?} must wait for {dependency:?} across an operation dependency")]
    MissingProgramSynchronization { program: String, dependency: String },
    #[error("operation {operation:?} is covered more than once by the selected schedule")]
    DuplicateScheduleCoverage { operation: String },
    #[error("schedule coverage differs: selected={selected:?}, covered={covered:?}")]
    ScheduleCoverageMismatch {
        selected: BTreeSet<String>,
        covered: BTreeSet<String>,
    },
}

impl ValidationError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::DuplicateOperation(_) => "DUPLICATE_OPERATION",
            Self::DuplicateProducer { .. } => "DUPLICATE_PRODUCER",
            Self::MissingProducer { .. } => "MISSING_PRODUCER",
            Self::UnknownControlDependency { .. } => "UNKNOWN_CONTROL_DEPENDENCY",
            Self::UnknownGraphOutput(_) => "UNKNOWN_GRAPH_OUTPUT",
            Self::DependencyCycle => "DEPENDENCY_CYCLE",
            Self::EmptyRegion => "EMPTY_REGION",
            Self::UnknownRegionOperation(_) => "UNKNOWN_REGION_OPERATION",
            Self::NonConvexRegion { .. } => "NON_CONVEX_REGION",
            Self::BoundaryMismatch { .. } => "BOUNDARY_MISMATCH",
            Self::StateEffectBoundaryMismatch { .. } => "STATE_EFFECT_BOUNDARY_MISMATCH",
            Self::UnsupportedBackend { .. } => "UNSUPPORTED_BACKEND",
            Self::NumericalPolicyMismatch { .. } => "NUMERICAL_POLICY_MISMATCH",
            Self::InvalidStateEffect { .. } => "INVALID_STATE_EFFECT",
            Self::RandomnessStateUnaccounted { .. } => "RANDOMNESS_STATE_UNACCOUNTED",
            Self::ScratchOverflow => "SCRATCH_OVERFLOW",
            Self::ScratchLimit { .. } => "SCRATCH_LIMIT",
            Self::ScheduleRegionMismatch { .. } => "SCHEDULE_REGION_MISMATCH",
            Self::EmptyProgram(_) => "EMPTY_PROGRAM",
            Self::ProgramBackendMismatch { .. } => "PROGRAM_BACKEND_MISMATCH",
            Self::DuplicateProgram(_) => "DUPLICATE_PROGRAM",
            Self::InvalidProgramWait { .. } => "INVALID_PROGRAM_WAIT",
            Self::ProgramDependencyOrder { .. } => "PROGRAM_DEPENDENCY_ORDER",
            Self::MissingProgramSynchronization { .. } => "MISSING_PROGRAM_SYNCHRONIZATION",
            Self::DuplicateScheduleCoverage { .. } => "DUPLICATE_SCHEDULE_COVERAGE",
            Self::ScheduleCoverageMismatch { .. } => "SCHEDULE_COVERAGE_MISMATCH",
        }
    }
}

struct IndexedGraph<'a> {
    operations: BTreeMap<&'a str, &'a Operation>,
    producer: BTreeMap<&'a str, &'a str>,
    consumers: BTreeMap<&'a str, BTreeSet<&'a str>>,
    successors: BTreeMap<&'a str, BTreeSet<&'a str>>,
    control_consumers: BTreeMap<&'a str, BTreeSet<&'a str>>,
}

fn index_graph(graph: &SemanticGraph) -> Result<IndexedGraph<'_>, ValidationError> {
    let mut operations = BTreeMap::new();
    let mut producer = BTreeMap::new();
    for operation in &graph.operations {
        if operations
            .insert(operation.id.as_str(), operation)
            .is_some()
        {
            return Err(ValidationError::DuplicateOperation(operation.id.clone()));
        }
        for output in &operation.outputs {
            if let Some(first) = producer.insert(output.as_str(), operation.id.as_str()) {
                return Err(ValidationError::DuplicateProducer {
                    value: output.clone(),
                    first: first.to_owned(),
                    second: operation.id.clone(),
                });
            }
        }
    }

    let mut consumers: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut successors: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut control_consumers: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for operation in &graph.operations {
        for input in &operation.inputs {
            consumers
                .entry(input.as_str())
                .or_default()
                .insert(operation.id.as_str());
            if let Some(owner) = producer.get(input.as_str()) {
                successors
                    .entry(owner)
                    .or_default()
                    .insert(operation.id.as_str());
            } else if !graph.external_inputs.contains(input) {
                return Err(ValidationError::MissingProducer {
                    operation: operation.id.clone(),
                    value: input.clone(),
                });
            }
        }
        for dependency in &operation.control_dependencies {
            if !operations.contains_key(dependency.as_str()) {
                return Err(ValidationError::UnknownControlDependency {
                    operation: operation.id.clone(),
                    dependency: dependency.clone(),
                });
            }
            successors
                .entry(dependency.as_str())
                .or_default()
                .insert(operation.id.as_str());
            control_consumers
                .entry(dependency.as_str())
                .or_default()
                .insert(operation.id.as_str());
        }
    }
    for output in &graph.graph_outputs {
        if !producer.contains_key(output.as_str()) && !graph.external_inputs.contains(output) {
            return Err(ValidationError::UnknownGraphOutput(output.clone()));
        }
    }

    let mut indegree = operations
        .keys()
        .map(|id| (*id, 0usize))
        .collect::<BTreeMap<_, _>>();
    for targets in successors.values() {
        for target in targets {
            *indegree.get_mut(target).expect("successor was indexed") += 1;
        }
    }
    let mut ready = indegree
        .iter()
        .filter_map(|(id, degree)| (*degree == 0).then_some(*id))
        .collect::<VecDeque<_>>();
    let mut visited = 0usize;
    while let Some(id) = ready.pop_front() {
        visited += 1;
        if let Some(targets) = successors.get(id) {
            for target in targets {
                let degree = indegree.get_mut(target).expect("successor was indexed");
                *degree -= 1;
                if *degree == 0 {
                    ready.push_back(target);
                }
            }
        }
    }
    if visited != operations.len() {
        return Err(ValidationError::DependencyCycle);
    }

    Ok(IndexedGraph {
        operations,
        producer,
        consumers,
        successors,
        control_consumers,
    })
}

fn reject_non_convex_region(
    selected: &BTreeSet<String>,
    indexed: &IndexedGraph<'_>,
) -> Result<(), ValidationError> {
    for start in selected {
        let mut queue = VecDeque::from([(start.as_str(), false)]);
        let mut seen = BTreeSet::new();
        while let Some((current, left_region)) = queue.pop_front() {
            if !seen.insert((current, left_region)) {
                continue;
            }
            for successor in indexed.successors.get(current).into_iter().flatten() {
                let successor_selected = selected.contains(*successor);
                if left_region && successor_selected {
                    return Err(ValidationError::NonConvexRegion {
                        from: start.clone(),
                        to: (*successor).to_owned(),
                    });
                }
                queue.push_back((successor, left_region || !successor_selected));
            }
        }
    }
    Ok(())
}

fn actual_boundary(
    graph: &SemanticGraph,
    selected: &BTreeSet<String>,
    indexed: &IndexedGraph<'_>,
) -> RegionBoundary {
    let mut boundary = RegionBoundary::default();
    for id in selected {
        let operation = indexed.operations[id.as_str()];
        for input in &operation.inputs {
            if indexed
                .producer
                .get(input.as_str())
                .map_or(true, |owner| !selected.contains(*owner))
            {
                boundary.inputs.insert(input.clone());
            }
        }
        for output in &operation.outputs {
            let outside = indexed
                .consumers
                .get(output.as_str())
                .into_iter()
                .flatten()
                .filter(|consumer| !selected.contains(**consumer))
                .copied()
                .collect::<BTreeSet<_>>();
            if graph.graph_outputs.contains(output) || !outside.is_empty() {
                boundary.outputs.insert(output.clone());
            }
            for consumer in outside {
                boundary
                    .outside_consumers
                    .insert(format!("{consumer}:{output}"));
            }
        }
        for dependency in &operation.control_dependencies {
            if !selected.contains(dependency) {
                boundary.control_inputs.insert(dependency.clone());
            }
        }
        for consumer in indexed
            .control_consumers
            .get(id.as_str())
            .into_iter()
            .flatten()
            .filter(|consumer| !selected.contains(**consumer))
        {
            boundary.control_outputs.insert(id.clone());
            boundary
                .outside_control_consumers
                .insert(format!("{consumer}:{id}"));
        }
        for effect in &operation.state_effects {
            boundary.state_effects.insert(effect.clone());
            if effect.access.reads() {
                boundary.state_reads.insert(effect.resource.clone());
            }
            if effect.access.writes() {
                boundary.state_writes.insert(effect.resource.clone());
            }
            if let Some(alias) = &effect.alias_group {
                boundary.alias_groups.insert(alias.clone());
            }
        }
        if let Some(state) = &operation.randomness_state {
            boundary.randomness_states.insert(state.clone());
        }
    }
    boundary
}

fn compare_boundary(
    field: &'static str,
    declared: &BTreeSet<String>,
    actual: &BTreeSet<String>,
) -> Result<(), ValidationError> {
    if declared == actual {
        Ok(())
    } else {
        Err(ValidationError::BoundaryMismatch {
            field,
            declared: declared.clone(),
            actual: actual.clone(),
        })
    }
}

pub fn validate_region(
    graph: &SemanticGraph,
    region: RegionCandidate,
    schedule: SelectedSchedule,
) -> Result<ValidatedRegion, ValidationError> {
    let indexed = index_graph(graph)?;
    if region.operation_ids.is_empty() {
        return Err(ValidationError::EmptyRegion);
    }
    for id in &region.operation_ids {
        if !indexed.operations.contains_key(id.as_str()) {
            return Err(ValidationError::UnknownRegionOperation(id.clone()));
        }
    }
    reject_non_convex_region(&region.operation_ids, &indexed)?;

    let actual = actual_boundary(graph, &region.operation_ids, &indexed);
    for (field, declared, observed) in [
        ("inputs", &region.boundary.inputs, &actual.inputs),
        ("outputs", &region.boundary.outputs, &actual.outputs),
        (
            "outside_consumers",
            &region.boundary.outside_consumers,
            &actual.outside_consumers,
        ),
        (
            "control_inputs",
            &region.boundary.control_inputs,
            &actual.control_inputs,
        ),
        (
            "control_outputs",
            &region.boundary.control_outputs,
            &actual.control_outputs,
        ),
        (
            "outside_control_consumers",
            &region.boundary.outside_control_consumers,
            &actual.outside_control_consumers,
        ),
        (
            "state_reads",
            &region.boundary.state_reads,
            &actual.state_reads,
        ),
        (
            "state_writes",
            &region.boundary.state_writes,
            &actual.state_writes,
        ),
        (
            "alias_groups",
            &region.boundary.alias_groups,
            &actual.alias_groups,
        ),
        (
            "randomness_states",
            &region.boundary.randomness_states,
            &actual.randomness_states,
        ),
    ] {
        compare_boundary(field, declared, observed)?;
    }
    if region.boundary.state_effects != actual.state_effects {
        return Err(ValidationError::StateEffectBoundaryMismatch {
            declared: region.boundary.state_effects.clone(),
            actual: actual.state_effects,
        });
    }

    let mut required_scratch_bytes = 0u64;
    for id in &region.operation_ids {
        let operation = indexed.operations[id.as_str()];
        if !operation.supported_backends.contains(&region.backend) {
            return Err(ValidationError::UnsupportedBackend {
                operation: id.clone(),
                backend: region.backend,
            });
        }
        if operation.numerical_policy != region.numerical_policy {
            return Err(ValidationError::NumericalPolicyMismatch {
                operation: id.clone(),
            });
        }
        for effect in &operation.state_effects {
            if effect.loop_carried
                && (!matches!(effect.access, Access::ReadWrite)
                    || !matches!(
                        effect.lifetime,
                        StateLifetime::Session | StateLifetime::Resident
                    )
                    || !matches!(effect.visibility, Visibility::Device | Visibility::Host))
            {
                return Err(ValidationError::InvalidStateEffect {
                    operation: id.clone(),
                    resource: effect.resource.clone(),
                    reason: "loop-carried state must be read-write, session/resident-lived, and device/host-visible",
                });
            }
            if effect.ownership == Ownership::Shared && effect.access.writes() {
                return Err(ValidationError::InvalidStateEffect {
                    operation: id.clone(),
                    resource: effect.resource.clone(),
                    reason: "shared writable state needs an explicit synchronization owner",
                });
            }
        }
        if let Some(randomness) = &operation.randomness_state {
            if !operation.state_effects.iter().any(|effect| {
                effect.resource == *randomness && matches!(effect.access, Access::ReadWrite)
            }) {
                return Err(ValidationError::RandomnessStateUnaccounted {
                    operation: id.clone(),
                    resource: randomness.clone(),
                });
            }
        }
        required_scratch_bytes = required_scratch_bytes
            .checked_add(operation.scratch_bytes)
            .ok_or(ValidationError::ScratchOverflow)?;
    }
    if required_scratch_bytes > region.available_scratch_bytes {
        return Err(ValidationError::ScratchLimit {
            required: required_scratch_bytes,
            available: region.available_scratch_bytes,
        });
    }

    if schedule.region_id != region.region_id {
        return Err(ValidationError::ScheduleRegionMismatch {
            schedule: schedule.region_id,
            candidate: region.region_id,
        });
    }
    let mut covered = BTreeSet::new();
    let mut program_indices = BTreeMap::new();
    let mut operation_programs = BTreeMap::new();
    for (index, program) in schedule.programs.iter().enumerate() {
        if program_indices
            .insert(program.program_id.as_str(), index)
            .is_some()
        {
            return Err(ValidationError::DuplicateProgram(
                program.program_id.clone(),
            ));
        }
        if program.operation_ids.is_empty() {
            return Err(ValidationError::EmptyProgram(program.program_id.clone()));
        }
        if program.backend != region.backend {
            return Err(ValidationError::ProgramBackendMismatch {
                program: program.program_id.clone(),
                actual: program.backend,
                expected: region.backend,
            });
        }
        for operation in &program.operation_ids {
            if !covered.insert(operation.clone()) {
                return Err(ValidationError::DuplicateScheduleCoverage {
                    operation: operation.clone(),
                });
            }
            operation_programs.insert(operation.as_str(), index);
        }
    }
    if covered != region.operation_ids {
        return Err(ValidationError::ScheduleCoverageMismatch {
            selected: region.operation_ids.clone(),
            covered,
        });
    }
    for (index, program) in schedule.programs.iter().enumerate() {
        for dependency in &program.waits_for {
            if program_indices
                .get(dependency.as_str())
                .map_or(true, |dependency_index| *dependency_index >= index)
            {
                return Err(ValidationError::InvalidProgramWait {
                    program: program.program_id.clone(),
                    dependency: dependency.clone(),
                });
            }
        }
        for operation_id in &program.operation_ids {
            let operation = indexed.operations[operation_id.as_str()];
            let mut dependencies = operation.control_dependencies.clone();
            for input in &operation.inputs {
                if let Some(producer) = indexed.producer.get(input.as_str()) {
                    if region.operation_ids.contains(*producer) {
                        dependencies.insert((*producer).to_owned());
                    }
                }
            }
            for dependency in dependencies {
                let dependency_index = operation_programs[dependency.as_str()];
                if dependency_index > index {
                    return Err(ValidationError::ProgramDependencyOrder {
                        operation: operation_id.clone(),
                        dependency,
                    });
                }
                if dependency_index < index {
                    let dependency_program = &schedule.programs[dependency_index].program_id;
                    if !program.waits_for.contains(dependency_program) {
                        return Err(ValidationError::MissingProgramSynchronization {
                            program: program.program_id.clone(),
                            dependency: dependency_program.clone(),
                        });
                    }
                }
            }
        }
    }

    let graph_bytes = serde_json::to_vec(graph).expect("SemanticGraph serialization is infallible");
    let semantic_graph_sha256 = format!("{:x}", Sha256::digest(graph_bytes));

    Ok(ValidatedRegion {
        schema: SCHEMA.to_owned(),
        graph_id: graph.graph_id.clone(),
        semantic_graph_sha256,
        region,
        schedule,
        required_scratch_bytes,
        qualification: "STRUCTURALLY_VALIDATED_PLAN_ONLY".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn op(id: &str, input: &str, output: &str) -> Operation {
        Operation {
            id: id.to_owned(),
            semantic_kind: id.to_owned(),
            stages: BTreeSet::from([StageAnnotation::Project]),
            inputs: set(&[input]),
            outputs: set(&[output]),
            control_dependencies: BTreeSet::new(),
            state_effects: Vec::new(),
            randomness_state: None,
            numerical_policy: NumericalPolicy::Exact,
            supported_backends: BTreeSet::from([Backend::Cpu]),
            scratch_bytes: 4,
        }
    }

    fn graph() -> SemanticGraph {
        SemanticGraph {
            graph_id: "test".to_owned(),
            operations: vec![op("a", "input", "a.out"), op("b", "a.out", "b.out")],
            external_inputs: set(&["input"]),
            graph_outputs: set(&["b.out"]),
        }
    }

    fn candidate() -> RegionCandidate {
        RegionCandidate {
            region_id: "ab".to_owned(),
            operation_ids: set(&["a", "b"]),
            boundary: RegionBoundary {
                inputs: set(&["input"]),
                outputs: set(&["b.out"]),
                ..RegionBoundary::default()
            },
            backend: Backend::Cpu,
            numerical_policy: NumericalPolicy::Exact,
            available_scratch_bytes: 8,
        }
    }

    fn schedule() -> SelectedSchedule {
        SelectedSchedule {
            region_id: "ab".to_owned(),
            programs: vec![
                BackendProgram {
                    program_id: "first".to_owned(),
                    backend: Backend::Cpu,
                    operation_ids: set(&["a"]),
                    waits_for: BTreeSet::new(),
                },
                BackendProgram {
                    program_id: "second".to_owned(),
                    backend: Backend::Cpu,
                    operation_ids: set(&["b"]),
                    waits_for: set(&["first"]),
                },
            ],
        }
    }

    #[test]
    fn one_region_may_lower_to_multiple_programs() {
        let validated = validate_region(&graph(), candidate(), schedule()).unwrap();
        assert_eq!(validated.required_scratch_bytes, 8);
        assert_eq!(validated.schedule.programs.len(), 2);
        assert_eq!(validated.qualification, "STRUCTURALLY_VALIDATED_PLAN_ONLY");
    }

    #[test]
    fn missing_boundary_effect_is_refused() {
        let mut graph = graph();
        graph.operations[1].state_effects.push(StateEffect {
            resource: "kv.layer0".to_owned(),
            access: Access::Write,
            alias_group: Some("kv".to_owned()),
            ownership: Ownership::Exclusive,
            lifetime: StateLifetime::Session,
            loop_carried: true,
            visibility: Visibility::Device,
        });
        let error = validate_region(&graph, candidate(), schedule()).unwrap_err();
        assert_eq!(error.code(), "BOUNDARY_MISMATCH");
        assert!(matches!(
            error,
            ValidationError::BoundaryMismatch {
                field: "state_writes",
                ..
            }
        ));
    }

    #[test]
    fn non_convex_selection_is_refused() {
        let graph = SemanticGraph {
            graph_id: "non-convex".to_owned(),
            operations: vec![
                op("a", "input", "a.out"),
                op("outside", "a.out", "outside.out"),
                op("c", "outside.out", "c.out"),
            ],
            external_inputs: set(&["input"]),
            graph_outputs: set(&["c.out"]),
        };
        let region = RegionCandidate {
            region_id: "ac".to_owned(),
            operation_ids: set(&["a", "c"]),
            boundary: RegionBoundary::default(),
            backend: Backend::Cpu,
            numerical_policy: NumericalPolicy::Exact,
            available_scratch_bytes: 8,
        };
        let error = validate_region(
            &graph,
            region,
            SelectedSchedule {
                region_id: "ac".to_owned(),
                programs: vec![],
            },
        )
        .unwrap_err();
        assert_eq!(error.code(), "NON_CONVEX_REGION");
    }

    #[test]
    fn schedule_must_cover_each_operation_once() {
        let mut schedule = schedule();
        schedule.programs[1].operation_ids = set(&["a", "b"]);
        let error = validate_region(&graph(), candidate(), schedule).unwrap_err();
        assert_eq!(error.code(), "DUPLICATE_SCHEDULE_COVERAGE");
    }

    #[test]
    fn graph_rejects_missing_producer_before_region_reasoning() {
        let mut graph = graph();
        graph.operations[0].inputs = set(&["invented"]);
        let error = validate_region(&graph, candidate(), schedule()).unwrap_err();
        assert_eq!(error.code(), "MISSING_PRODUCER");
    }

    #[test]
    fn representation_and_token_decode_are_distinct_annotations() {
        assert_ne!(
            StageAnnotation::RepresentationDecode,
            StageAnnotation::AutoregressiveDecode
        );
    }

    #[test]
    fn outgoing_control_dependency_must_be_declared() {
        let mut graph = graph();
        graph.operations[1].inputs = set(&["input"]);
        graph.operations[1].control_dependencies = set(&["a"]);
        let region = RegionCandidate {
            region_id: "a-only".to_owned(),
            operation_ids: set(&["a"]),
            boundary: RegionBoundary {
                inputs: set(&["input"]),
                ..RegionBoundary::default()
            },
            backend: Backend::Cpu,
            numerical_policy: NumericalPolicy::Exact,
            available_scratch_bytes: 4,
        };
        let error = validate_region(
            &graph,
            region,
            SelectedSchedule {
                region_id: "a-only".to_owned(),
                programs: vec![BackendProgram {
                    program_id: "a".to_owned(),
                    backend: Backend::Cpu,
                    operation_ids: set(&["a"]),
                    waits_for: BTreeSet::new(),
                }],
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ValidationError::BoundaryMismatch {
                field: "control_outputs",
                ..
            }
        ));
    }

    #[test]
    fn reversed_program_order_is_refused() {
        let mut schedule = schedule();
        schedule.programs.reverse();
        let error = validate_region(&graph(), candidate(), schedule).unwrap_err();
        assert_eq!(error.code(), "INVALID_PROGRAM_WAIT");
    }

    #[test]
    fn unknown_graph_output_is_refused() {
        let mut graph = graph();
        graph.graph_outputs = set(&["never.produced"]);
        let error = validate_region(&graph, candidate(), schedule()).unwrap_err();
        assert_eq!(error.code(), "UNKNOWN_GRAPH_OUTPUT");
    }

    #[test]
    fn unsupported_backend_and_scratch_limit_are_refused() {
        let mut unsupported = candidate();
        unsupported.backend = Backend::Metal;
        assert_eq!(
            validate_region(&graph(), unsupported, schedule())
                .unwrap_err()
                .code(),
            "UNSUPPORTED_BACKEND"
        );
        let mut too_small = candidate();
        too_small.available_scratch_bytes = 7;
        assert_eq!(
            validate_region(&graph(), too_small, schedule())
                .unwrap_err()
                .code(),
            "SCRATCH_LIMIT"
        );
    }
}
