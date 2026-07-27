//! The deterministic tree-walking interpreter.
//!
//! # Why an in-crate interpreter
//!
//! The runtime is a small tree-walking evaluator over the [`Expr`] AST rather
//! than an embedded scripting engine (Lua, JS, wasm, ...). That choice is
//! deliberate: a heavyweight engine brings its own I/O surface, its own clock,
//! its own allocator, and its own nondeterminism, all of which would have to be
//! fenced off again. A purpose-built evaluator has *no* capability we did not
//! give it. There is no `import`, no host-function table beyond the closed
//! [`HandleName`] set, no ambient clock, and no source of entropy except a
//! seeded rng. Determinism and "no ambient authority" fall out of the design
//! instead of being bolted on.
//!
//! Everything here is model-free. The runtime evaluates data-shaped programs
//! over host-provided read handles; it never runs a model.

use crate::ast::{BinOp, Expr, JoinKind, Lambda, Operator, Order, Program, SchemaField, SchemaSpec};
use crate::error::{Result, RuntimeError};
use crate::handles::{HandleGrants, HandleName, HostHandles};
use crate::limits::{Limits, Meter, Usage};
use crate::proposal::{WriteKind, WriteProposal};
use crate::value::{Citation, Value};

/// A spilled artifact produced by `spill_to_artifact`. Held out of the returned
/// value so a large intermediate does not blow the output budget.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Artifact {
    pub id: String,
    pub name: String,
    pub byte_len: u64,
    pub digest: String,
    pub content: Value,
}

/// The result of running a program.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RunOutput {
    /// The program's return value.
    pub value: Value,
    /// Mutations the program prepared. NONE were executed by the runtime.
    pub proposals: Vec<WriteProposal>,
    /// Artifacts the program spilled.
    pub artifacts: Vec<Artifact>,
    /// Deterministic resource usage.
    pub usage: Usage,
}

/// A single variable binding in a parent-linked scope chain. Lambda and `let`
/// each introduce exactly one, so the chain is cheap and needs no cloning.
struct Scope<'a> {
    name: &'a str,
    value: &'a Value,
    parent: Option<&'a Scope<'a>>,
}

impl<'a> Scope<'a> {
    fn lookup(&self, name: &str) -> Option<&Value> {
        if self.name == name {
            Some(self.value)
        } else {
            self.parent.and_then(|p| p.lookup(name))
        }
    }
}

fn lookup<'a>(scope: Option<&'a Scope<'a>>, name: &str) -> Option<&'a Value> {
    scope.and_then(|s| s.lookup(name))
}

/// A minimal deterministic rng (SplitMix64). Seeded from the program; no entropy
/// source is consulted, so `sample` is reproducible.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next() % bound as u64) as usize
        }
    }
}

/// Interpreter state for one run.
struct Exec<'h> {
    host: &'h dyn HostHandles,
    grants: &'h HandleGrants,
    meter: Meter,
    rng: SplitMix64,
    proposals: Vec<WriteProposal>,
    artifacts: Vec<Artifact>,
    proposal_seq: u64,
    artifact_seq: u64,
    /// The current attempt index, threaded to handles so a retry can present a
    /// deterministic "next attempt" to a flaky fixture.
    attempt: u32,
}

impl<'h> Exec<'h> {
    fn new(
        host: &'h dyn HostHandles,
        grants: &'h HandleGrants,
        limits: Limits,
        seed: u64,
        clock_start_ms: u64,
    ) -> Self {
        Exec {
            host,
            grants,
            meter: Meter::new(limits, clock_start_ms),
            // Mix the seed so a zero seed is not a degenerate state.
            rng: SplitMix64(seed ^ 0xA5A5_A5A5_5A5A_5A5A),
            proposals: Vec::new(),
            artifacts: Vec::new(),
            proposal_seq: 0,
            artifact_seq: 0,
            attempt: 0,
        }
    }

    fn eval(&mut self, expr: &Expr, scope: Option<&Scope>, depth: u32) -> Result<Value> {
        self.meter.tick_instruction()?;
        self.meter.check_recursion(depth)?;

        match expr {
            Expr::Lit { value } => {
                self.meter.observe_value(value.estimated_bytes())?;
                Ok(value.clone())
            }
            Expr::Var { name } => lookup(scope, name)
                .cloned()
                .ok_or_else(|| RuntimeError::UnboundVariable(name.clone())),
            Expr::Field { base, path } => {
                let b = self.eval(base, scope, depth + 1)?;
                Ok(b.get_path(path).cloned().unwrap_or(Value::Null))
            }
            Expr::List { items } => {
                let mut out = Vec::with_capacity(items.len());
                for it in items {
                    out.push(self.eval(it, scope, depth + 1)?);
                }
                let v = Value::List(out);
                self.meter.observe_value(v.estimated_bytes())?;
                Ok(v)
            }
            Expr::MapLit { entries } => {
                let mut m = std::collections::BTreeMap::new();
                for (k, ve) in entries {
                    let val = self.eval(ve, scope, depth + 1)?;
                    m.insert(k.clone(), val);
                }
                let v = Value::Map(m);
                self.meter.observe_value(v.estimated_bytes())?;
                Ok(v)
            }
            Expr::Handle { name, args } => {
                let a = self.eval(args, scope, depth + 1)?;
                self.call_handle(*name, &a)
            }
            Expr::Let { name, value, body } => {
                let bound = self.eval(value, scope, depth + 1)?;
                let inner = Scope {
                    name,
                    value: &bound,
                    parent: scope,
                };
                self.eval(body, Some(&inner), depth + 1)
            }
            Expr::If {
                cond,
                then,
                otherwise,
            } => {
                let c = self.eval(cond, scope, depth + 1)?;
                if c.is_truthy() {
                    self.eval(then, scope, depth + 1)
                } else {
                    self.eval(otherwise, scope, depth + 1)
                }
            }
            Expr::BinOp { op, lhs, rhs } => self.eval_binop(*op, lhs, rhs, scope, depth),
            Expr::Not { value } => {
                let v = self.eval(value, scope, depth + 1)?;
                Ok(Value::Bool(!v.is_truthy()))
            }
            Expr::ProposeWrite { spec } => {
                let s = self.eval(spec, scope, depth + 1)?;
                self.propose_write(&s)
            }
            Expr::Op { operator } => self.eval_op(operator, scope, depth),
        }
    }

    fn eval_binop(
        &mut self,
        op: BinOp,
        lhs: &Expr,
        rhs: &Expr,
        scope: Option<&Scope>,
        depth: u32,
    ) -> Result<Value> {
        // Short-circuit the boolean connectives.
        match op {
            BinOp::And => {
                let l = self.eval(lhs, scope, depth + 1)?;
                if !l.is_truthy() {
                    return Ok(Value::Bool(false));
                }
                let r = self.eval(rhs, scope, depth + 1)?;
                return Ok(Value::Bool(r.is_truthy()));
            }
            BinOp::Or => {
                let l = self.eval(lhs, scope, depth + 1)?;
                if l.is_truthy() {
                    return Ok(Value::Bool(true));
                }
                let r = self.eval(rhs, scope, depth + 1)?;
                return Ok(Value::Bool(r.is_truthy()));
            }
            _ => {}
        }

        let l = self.eval(lhs, scope, depth + 1)?;
        let r = self.eval(rhs, scope, depth + 1)?;
        let out = match op {
            BinOp::Eq => Value::Bool(l == r),
            BinOp::Ne => Value::Bool(l != r),
            BinOp::Lt => Value::Bool(l.total_cmp(&r).is_lt()),
            BinOp::Le => Value::Bool(l.total_cmp(&r).is_le()),
            BinOp::Gt => Value::Bool(l.total_cmp(&r).is_gt()),
            BinOp::Ge => Value::Bool(l.total_cmp(&r).is_ge()),
            BinOp::Add => arith(&l, &r, |a, b| a + b, |a, b| a.wrapping_add(b))?,
            BinOp::Sub => arith(&l, &r, |a, b| a - b, |a, b| a.wrapping_sub(b))?,
            BinOp::Mul => arith(&l, &r, |a, b| a * b, |a, b| a.wrapping_mul(b))?,
            BinOp::Contains => Value::Bool(contains(&l, &r)),
            BinOp::And | BinOp::Or => unreachable!("handled above"),
        };
        Ok(out)
    }

    fn call_handle(&mut self, name: HandleName, args: &Value) -> Result<Value> {
        if !self.grants.is_granted(name) {
            return Err(RuntimeError::HandleNotGranted(name.as_str().to_string()));
        }
        self.meter.charge_tool_call()?;
        let out = self.host.call(name, args, self.attempt)?;
        self.meter.observe_value(out.estimated_bytes())?;
        Ok(out)
    }

    fn propose_write(&mut self, spec: &Value) -> Result<Value> {
        let m = spec
            .as_map()
            .ok_or_else(|| RuntimeError::Type("propose_write expects a map".into()))?;
        let kind_str = m
            .get("kind")
            .and_then(Value::as_str)
            .ok_or_else(|| RuntimeError::Type("propose_write: missing 'kind' string".into()))?;
        let kind = WriteKind::from_str(kind_str)
            .ok_or_else(|| RuntimeError::Type(format!("propose_write: unknown kind {kind_str:?}")))?;
        let summary = m
            .get("summary")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let payload = m.get("payload").cloned().unwrap_or(Value::Null);
        let citations = m.get("citations").map(Citation::list_from).unwrap_or_default();

        let id = format!("wp-{}", self.proposal_seq);
        self.proposal_seq += 1;
        self.proposals.push(WriteProposal {
            id: id.clone(),
            kind,
            summary: summary.clone(),
            payload,
            citations,
        });

        // Return a reference the program can embed in its result. The mutation
        // itself is NOT executed here or anywhere in this crate.
        Ok(crate::value::map_of([
            ("@write_proposal", Value::Str(id)),
            ("kind", Value::Str(kind.as_str().to_string())),
            ("summary", Value::Str(summary)),
        ]))
    }

    // -- operators ---------------------------------------------------------

    fn eval_list(&mut self, e: &Expr, scope: Option<&Scope>, depth: u32) -> Result<Vec<Value>> {
        let v = self.eval(e, scope, depth + 1)?;
        match v {
            Value::List(items) => Ok(items),
            other => Err(RuntimeError::Type(format!(
                "expected a list, got {}",
                type_name(&other)
            ))),
        }
    }

    fn apply(
        &mut self,
        lambda: &Lambda,
        arg: &Value,
        scope: Option<&Scope>,
        depth: u32,
    ) -> Result<Value> {
        let inner = Scope {
            name: &lambda.param,
            value: arg,
            parent: scope,
        };
        self.eval(&lambda.body, Some(&inner), depth + 1)
    }

    fn eval_op(&mut self, op: &Operator, scope: Option<&Scope>, depth: u32) -> Result<Value> {
        match op {
            Operator::ParallelMap {
                input,
                func,
                concurrency,
            } => {
                let width = concurrency.unwrap_or(1);
                self.meter.check_concurrency(width)?;
                self.map_over(input, func, scope, depth)
            }
            Operator::BoundedMap { input, func, bound } => {
                self.meter.check_concurrency(*bound)?;
                self.map_over(input, func, scope, depth)
            }
            Operator::Filter { input, pred } => {
                let items = self.eval_list(input, scope, depth)?;
                let mut out = Vec::new();
                for it in &items {
                    if self.apply(pred, it, scope, depth)?.is_truthy() {
                        out.push(it.clone());
                    }
                }
                self.finish_list(out)
            }
            Operator::Rank {
                input,
                key,
                order,
                limit,
            } => {
                let items = self.eval_list(input, scope, depth)?;
                let mut keyed: Vec<(Value, Value)> = Vec::with_capacity(items.len());
                for it in &items {
                    let k = self.apply(key, it, scope, depth)?;
                    keyed.push((k, it.clone()));
                }
                keyed.sort_by(|a, b| {
                    let ord = a.0.total_cmp(&b.0);
                    match order {
                        Order::Asc => ord,
                        Order::Desc => ord.reverse(),
                    }
                });
                let mut out: Vec<Value> = keyed.into_iter().map(|(_, v)| v).collect();
                if let Some(n) = limit {
                    out.truncate(*n);
                }
                self.finish_list(out)
            }
            Operator::Group { input, key } => {
                let items = self.eval_list(input, scope, depth)?;
                // Sorted by canonical key so group order is deterministic.
                let mut groups: std::collections::BTreeMap<String, (Value, Vec<Value>)> =
                    std::collections::BTreeMap::new();
                for it in &items {
                    let k = self.apply(key, it, scope, depth)?;
                    let ck = k.canonical_key();
                    groups
                        .entry(ck)
                        .or_insert_with(|| (k.clone(), Vec::new()))
                        .1
                        .push(it.clone());
                }
                let out: Vec<Value> = groups
                    .into_iter()
                    .map(|(_, (k, items))| {
                        crate::value::map_of([("key", k), ("items", Value::List(items))])
                    })
                    .collect();
                self.finish_list(out)
            }
            Operator::Join {
                left,
                right,
                left_key,
                right_key,
                kind,
            } => self.eval_join(left, right, left_key, right_key, *kind, scope, depth),
            Operator::Reduce {
                input,
                init,
                acc,
                item,
                body,
            } => {
                let items = self.eval_list(input, scope, depth)?;
                let mut acc_val = self.eval(init, scope, depth + 1)?;
                for it in &items {
                    let new = {
                        let s_acc = Scope {
                            name: acc,
                            value: &acc_val,
                            parent: scope,
                        };
                        let s_item = Scope {
                            name: item,
                            value: it,
                            parent: Some(&s_acc),
                        };
                        self.eval(body, Some(&s_item), depth + 1)?
                    };
                    acc_val = new;
                }
                Ok(acc_val)
            }
            Operator::Paginate {
                input,
                page_size,
                page,
            } => {
                let items = self.eval_list(input, scope, depth)?;
                let start = page.saturating_mul(*page_size);
                let out: Vec<Value> = items
                    .into_iter()
                    .skip(start)
                    .take(*page_size)
                    .collect();
                self.finish_list(out)
            }
            Operator::RetryWithPolicy { body, policy } => {
                let attempts = policy.max_attempts.max(1);
                let saved = self.attempt;
                let mut last: Option<RuntimeError> = None;
                for attempt in 0..attempts {
                    if attempt > 0 {
                        self.meter
                            .advance_clock(policy.backoff_ms.saturating_mul(attempt as u64))?;
                    }
                    self.attempt = attempt;
                    match self.eval(body, scope, depth + 1) {
                        Ok(v) => {
                            self.attempt = saved;
                            return Ok(v);
                        }
                        Err(RuntimeError::Handle(h))
                            if h.retryable && attempt + 1 < attempts =>
                        {
                            last = Some(RuntimeError::Handle(h));
                        }
                        Err(e) => {
                            self.attempt = saved;
                            return Err(e);
                        }
                    }
                }
                self.attempt = saved;
                Err(last.unwrap_or_else(|| {
                    RuntimeError::Type("retry_with_policy: no attempts ran".into())
                }))
            }
            Operator::SchemaValidate { input, schema } => {
                let v = self.eval(input, scope, depth + 1)?;
                validate(&v, schema, "$").map_err(RuntimeError::Schema)?;
                Ok(v)
            }
            Operator::Dedup { input, key } => {
                let items = self.eval_list(input, scope, depth)?;
                let mut seen: std::collections::BTreeSet<String> =
                    std::collections::BTreeSet::new();
                let mut out = Vec::new();
                for it in &items {
                    let k = match key {
                        Some(l) => self.apply(l, it, scope, depth)?.canonical_key(),
                        None => it.canonical_key(),
                    };
                    if seen.insert(k) {
                        out.push(it.clone());
                    }
                }
                self.finish_list(out)
            }
            Operator::Sample { input, k } => {
                let items = self.eval_list(input, scope, depth)?;
                let out = self.sample(items, *k);
                self.finish_list(out)
            }
            Operator::SpillToArtifact { input, name } => {
                let v = self.eval(input, scope, depth + 1)?;
                self.spill(v, name)
            }
            Operator::CitationPreservation { input, func } => {
                let items = self.eval_list(input, scope, depth)?;
                let mut out = Vec::with_capacity(items.len());
                for src in &items {
                    let mapped = self.apply(func, src, scope, depth)?;
                    let preserved = mapped.with_merged_citations(&src.citations());
                    out.push(preserved);
                }
                self.finish_list(out)
            }
        }
    }

    fn map_over(
        &mut self,
        input: &Expr,
        func: &Lambda,
        scope: Option<&Scope>,
        depth: u32,
    ) -> Result<Value> {
        let items = self.eval_list(input, scope, depth)?;
        let mut out = Vec::with_capacity(items.len());
        for it in &items {
            out.push(self.apply(func, it, scope, depth)?);
        }
        self.finish_list(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_join(
        &mut self,
        left: &Expr,
        right: &Expr,
        left_key: &Lambda,
        right_key: &Lambda,
        kind: JoinKind,
        scope: Option<&Scope>,
        depth: u32,
    ) -> Result<Value> {
        let lefts = self.eval_list(left, scope, depth)?;
        let rights = self.eval_list(right, scope, depth)?;

        // Index the right side by key, preserving input order within a key.
        let mut index: std::collections::BTreeMap<String, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (i, r) in rights.iter().enumerate() {
            let rk = self.apply(right_key, r, scope, depth)?.canonical_key();
            index.entry(rk).or_default().push(i);
        }

        let mut out = Vec::new();
        for l in &lefts {
            let lk = self.apply(left_key, l, scope, depth)?.canonical_key();
            match index.get(&lk) {
                Some(matches) => {
                    for &ri in matches {
                        out.push(join_record(l, &rights[ri]));
                    }
                }
                None => {
                    if matches!(kind, JoinKind::Left) {
                        out.push(join_record(l, &Value::Null));
                    }
                }
            }
        }
        self.finish_list(out)
    }

    fn sample(&mut self, items: Vec<Value>, k: usize) -> Vec<Value> {
        if k >= items.len() {
            return items;
        }
        // Partial Fisher-Yates: select k indices, then return them in original
        // input order for a stable, deterministic subset.
        let mut idx: Vec<usize> = (0..items.len()).collect();
        for i in 0..k {
            let j = i + self.rng.below(items.len() - i);
            idx.swap(i, j);
        }
        let mut chosen: Vec<usize> = idx[..k].to_vec();
        chosen.sort_unstable();
        chosen.into_iter().map(|i| items[i].clone()).collect()
    }

    fn spill(&mut self, v: Value, name: &str) -> Result<Value> {
        let bytes = serde_json::to_vec(&v).map_err(|e| RuntimeError::Type(e.to_string()))?;
        let byte_len = bytes.len() as u64;
        self.meter.check_artifact(byte_len)?;
        let digest = blake3::hash(&bytes).to_hex().to_string();
        let id = format!("artifact-{}", self.artifact_seq);
        self.artifact_seq += 1;
        self.artifacts.push(Artifact {
            id: id.clone(),
            name: name.to_string(),
            byte_len,
            digest: digest.clone(),
            content: v,
        });
        Ok(crate::value::map_of([
            ("@artifact", Value::Str(id)),
            ("name", Value::Str(name.to_string())),
            ("byte_len", Value::Int(byte_len as i64)),
            ("digest", Value::Str(digest)),
        ]))
    }

    fn finish_list(&mut self, out: Vec<Value>) -> Result<Value> {
        let v = Value::List(out);
        self.meter.observe_value(v.estimated_bytes())?;
        Ok(v)
    }
}

fn join_record(l: &Value, r: &Value) -> Value {
    let base = crate::value::map_of([("left", l.clone()), ("right", r.clone())]);
    // Carry both sides' provenance onto the combined record.
    let mut cites = l.citations();
    cites.extend(r.citations());
    base.with_merged_citations(&cites)
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Int(_) => "int",
        Value::Float(_) => "float",
        Value::Str(_) => "string",
        Value::List(_) => "list",
        Value::Map(_) => "map",
    }
}

fn arith(
    l: &Value,
    r: &Value,
    ff: impl Fn(f64, f64) -> f64,
    fi: impl Fn(i64, i64) -> i64,
) -> Result<Value> {
    match (l, r) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(fi(*a, *b))),
        (a, b) => match (a.as_f64(), b.as_f64()) {
            (Some(x), Some(y)) => Ok(Value::Float(ff(x, y))),
            _ => Err(RuntimeError::Type(format!(
                "arithmetic on non-numbers: {} and {}",
                type_name(l),
                type_name(r)
            ))),
        },
    }
}

fn contains(haystack: &Value, needle: &Value) -> bool {
    match haystack {
        Value::Str(s) => needle.as_str().map(|n| s.contains(n)).unwrap_or(false),
        Value::List(items) => items.iter().any(|it| it == needle),
        Value::Map(m) => needle
            .as_str()
            .map(|k| m.contains_key(k))
            .unwrap_or(false),
        _ => false,
    }
}

/// Validate a value against a schema. Returns `Ok(())` or a path-qualified
/// message. Deterministic and total.
fn validate(v: &Value, schema: &SchemaSpec, path: &str) -> std::result::Result<(), String> {
    let mismatch = |want: &str| Err(format!("{path}: expected {want}, got {}", type_name(v)));
    match schema {
        SchemaSpec::Any => Ok(()),
        SchemaSpec::Null => matches!(v, Value::Null).then_some(()).ok_or(()).or(mismatch("null")),
        SchemaSpec::Bool => matches!(v, Value::Bool(_))
            .then_some(())
            .ok_or(())
            .or(mismatch("bool")),
        SchemaSpec::Int => matches!(v, Value::Int(_))
            .then_some(())
            .ok_or(())
            .or(mismatch("int")),
        SchemaSpec::Float => matches!(v, Value::Float(_))
            .then_some(())
            .ok_or(())
            .or(mismatch("float")),
        SchemaSpec::Number => matches!(v, Value::Int(_) | Value::Float(_))
            .then_some(())
            .ok_or(())
            .or(mismatch("number")),
        SchemaSpec::Str => matches!(v, Value::Str(_))
            .then_some(())
            .ok_or(())
            .or(mismatch("string")),
        SchemaSpec::List { items } => {
            let Value::List(vs) = v else {
                return mismatch("list");
            };
            for (i, item) in vs.iter().enumerate() {
                validate(item, items, &format!("{path}[{i}]"))?;
            }
            Ok(())
        }
        SchemaSpec::Map { fields } => {
            let Value::Map(m) = v else {
                return mismatch("map");
            };
            for SchemaField {
                name,
                schema,
                required,
            } in fields
            {
                match m.get(name) {
                    Some(fv) => validate(fv, schema, &format!("{path}.{name}"))?,
                    None if *required => {
                        return Err(format!("{path}.{name}: required field missing"));
                    }
                    None => {}
                }
            }
            Ok(())
        }
    }
}

/// Run a program to completion under the given host, grants, and limits.
///
/// The runtime never touches the world except through the granted read
/// [`HandleName`]s, never executes a [`WriteProposal`], and is fully
/// deterministic: the same program, host, grants, and limits produce a
/// byte-identical [`RunOutput`] every time.
pub fn run(
    program: &Program,
    host: &dyn HostHandles,
    grants: &HandleGrants,
    limits: Limits,
) -> Result<RunOutput> {
    let mut exec = Exec::new(host, grants, limits, program.seed, program.clock_start_ms);
    let value = exec.eval(&program.root, None, 0)?;

    // The returned value must fit the output budget.
    let out_bytes = serde_json::to_vec(&value)
        .map(|b| b.len() as u64)
        .unwrap_or(0);
    exec.meter.check_output(out_bytes)?;

    Ok(RunOutput {
        value,
        proposals: exec.proposals,
        artifacts: exec.artifacts,
        usage: exec.meter.usage(),
    })
}

/// The reserved record field name that carries provenance, re-exported for
/// callers building fixtures.
pub use crate::value::CITATIONS_KEY as CITATIONS_FIELD;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;
    use crate::handles::{DenyAllHost, FnHost};
    use crate::value::map_of;

    fn run_pure(root: Expr, limits: Limits) -> Result<RunOutput> {
        let prog = Program::new(root);
        run(&prog, &DenyAllHost, &HandleGrants::none(), limits)
    }

    #[test]
    fn literals_and_arithmetic() {
        let e = Expr::bin(BinOp::Add, Expr::lit(2i64), Expr::lit(3i64));
        let out = run_pure(e, Limits::unbounded()).unwrap();
        assert_eq!(out.value, Value::Int(5));
    }

    #[test]
    fn filter_rank_over_list() {
        let rows = Expr::lit(Value::List(vec![
            map_of([("n", Value::Int(3))]),
            map_of([("n", Value::Int(1))]),
            map_of([("n", Value::Int(2))]),
        ]));
        let prog = Expr::op(Operator::Rank {
            input: Box::new(Expr::op(Operator::Filter {
                input: Box::new(rows),
                pred: Lambda::new(
                    "r",
                    Expr::bin(
                        BinOp::Ge,
                        Expr::field(Expr::var("r"), ["n"]),
                        Expr::lit(2i64),
                    ),
                ),
            })),
            key: Lambda::new("r", Expr::field(Expr::var("r"), ["n"])),
            order: Order::Desc,
            limit: None,
        });
        let out = run_pure(prog, Limits::unbounded()).unwrap();
        let list = out.value.as_list().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].get_path(&["n".into()]), Some(&Value::Int(3)));
        assert_eq!(list[1].get_path(&["n".into()]), Some(&Value::Int(2)));
    }

    #[test]
    fn ungranted_handle_is_denied() {
        let e = Expr::handle(HandleName::FileRead, Expr::lit("x"));
        let err = run_pure(e, Limits::unbounded()).unwrap_err();
        assert_eq!(err, RuntimeError::HandleNotGranted("file.read".into()));
    }

    #[test]
    fn granted_handle_is_called() {
        let host = FnHost::new(|h, _a, _t| {
            assert_eq!(h, HandleName::FileRead);
            Ok(Value::Str("hello".into()))
        });
        let prog = Program::new(Expr::handle(HandleName::FileRead, Expr::lit("p")));
        let out = run(
            &prog,
            &host,
            &HandleGrants::of([HandleName::FileRead]),
            Limits::unbounded(),
        )
        .unwrap();
        assert_eq!(out.value, Value::Str("hello".into()));
        assert_eq!(out.usage.tool_calls, 1);
    }

    #[test]
    fn reduce_sums() {
        let rows = Expr::lit(Value::List(vec![
            Value::Int(1),
            Value::Int(2),
            Value::Int(4),
        ]));
        let prog = Expr::op(Operator::Reduce {
            input: Box::new(rows),
            init: Box::new(Expr::lit(0i64)),
            acc: "a".into(),
            item: "x".into(),
            body: Box::new(Expr::bin(BinOp::Add, Expr::var("a"), Expr::var("x"))),
        });
        let out = run_pure(prog, Limits::unbounded()).unwrap();
        assert_eq!(out.value, Value::Int(7));
    }

    #[test]
    fn dedup_and_paginate() {
        let rows = Expr::lit(Value::List(vec![
            Value::Int(1),
            Value::Int(1),
            Value::Int(2),
            Value::Int(3),
            Value::Int(3),
        ]));
        let dedup = Expr::op(Operator::Dedup {
            input: Box::new(rows),
            key: None,
        });
        let page = Expr::op(Operator::Paginate {
            input: Box::new(dedup),
            page_size: 2,
            page: 1,
        });
        let out = run_pure(page, Limits::unbounded()).unwrap();
        assert_eq!(out.value, Value::List(vec![Value::Int(3)]));
    }

    #[test]
    fn schema_validate_rejects_bad_shape() {
        let bad = Expr::lit(map_of([("n", Value::Str("x".into()))]));
        let prog = Expr::op(Operator::SchemaValidate {
            input: Box::new(bad),
            schema: SchemaSpec::Map {
                fields: vec![SchemaField {
                    name: "n".into(),
                    schema: SchemaSpec::Int,
                    required: true,
                }],
            },
        });
        let err = run_pure(prog, Limits::unbounded()).unwrap_err();
        assert!(matches!(err, RuntimeError::Schema(_)));
    }

    #[test]
    fn sample_is_seed_deterministic() {
        let rows: Vec<Value> = (0..20).map(Value::Int).collect();
        let mk = Expr::op(Operator::Sample {
            input: Box::new(Expr::lit(Value::List(rows))),
            k: 5,
        });
        let prog = Program::new(mk).with_seed(42);
        let a = run(&prog, &DenyAllHost, &HandleGrants::none(), Limits::unbounded()).unwrap();
        let b = run(&prog, &DenyAllHost, &HandleGrants::none(), Limits::unbounded()).unwrap();
        assert_eq!(a.value, b.value);
        assert_eq!(a.value.as_list().unwrap().len(), 5);
    }

    #[test]
    fn retry_recovers_from_flaky_handle() {
        // Fails on attempts 0 and 1, succeeds on attempt 2.
        let host = FnHost::new(|_h, _a, attempt| {
            if attempt < 2 {
                Err(crate::error::HandleError::retryable("git.log", "transient"))
            } else {
                Ok(Value::Str("ok".into()))
            }
        });
        let prog = Program::new(Expr::op(Operator::RetryWithPolicy {
            body: Box::new(Expr::handle(HandleName::GitLog, Expr::lit(Value::Null))),
            policy: RetryPolicy::new(3, 1),
        }));
        let out = run(
            &prog,
            &host,
            &HandleGrants::of([HandleName::GitLog]),
            Limits::unbounded(),
        )
        .unwrap();
        assert_eq!(out.value, Value::Str("ok".into()));
        assert_eq!(out.usage.tool_calls, 3);
    }
}
