//! The program AST.
//!
//! A program is a tree of [`Expr`] nodes evaluated by the interpreter in
//! `interp`. The tree is `serde`-serializable, so a program is *data*: it can be
//! authored as JSON, stored, hashed, and replayed. There are no loops or
//! unbounded recursion primitives - iteration happens only through the bounded
//! collection [`Operator`]s, which is what keeps evaluation total and metered.
//!
//! The AST has no node that touches the world. The only outward edge is
//! [`Expr::Handle`], which names a read-only [`HandleName`] and is gated by
//! grants at run time. Mutation is expressed with [`Expr::ProposeWrite`], which
//! builds a proposal and returns - it never executes.

use serde::{Deserialize, Serialize};

use crate::handles::HandleName;
use crate::value::Value;

/// A single-argument function used by collection operators. When applied, `param`
/// is bound to the current element and `body` is evaluated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lambda {
    pub param: String,
    pub body: Box<Expr>,
}

impl Lambda {
    pub fn new(param: impl Into<String>, body: Expr) -> Self {
        Lambda {
            param: param.into(),
            body: Box::new(body),
        }
    }
}

/// A comparison / boolean / arithmetic operator used inside [`Expr::BinOp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Add,
    Sub,
    Mul,
    /// String / list containment: `lhs contains rhs`.
    Contains,
}

/// Sort direction for `rank`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Order {
    Asc,
    Desc,
}

/// Join flavor for `join`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinKind {
    /// Only rows with a match on both sides.
    Inner,
    /// Every left row; unmatched left rows get a null `right`.
    Left,
}

/// Retry policy for `retry_with_policy`. Backoff is virtual (it advances the
/// runtime clock and counts against the wall-time budget), never a real sleep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    /// Milliseconds of virtual backoff added before attempt N (linear in N).
    pub backoff_ms: u64,
}

impl RetryPolicy {
    pub fn new(max_attempts: u32, backoff_ms: u64) -> Self {
        Self {
            max_attempts,
            backoff_ms,
        }
    }
}

/// A minimal, deterministic schema used by `schema_validate`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum SchemaSpec {
    Any,
    Null,
    Bool,
    Int,
    Float,
    /// Any number (int or float).
    Number,
    Str,
    /// A homogeneous list.
    List { items: Box<SchemaSpec> },
    /// An object with typed, optionally-required fields.
    Map { fields: Vec<SchemaField> },
}

/// One field in a [`SchemaSpec::Map`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaField {
    pub name: String,
    pub schema: SchemaSpec,
    #[serde(default)]
    pub required: bool,
}

/// A built-in collection / control operator. Each is deterministic and bounded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "op")]
pub enum Operator {
    /// Map `func` over a list. `concurrency` is the requested logical width,
    /// checked against the concurrency limit; results are always produced in
    /// input order so output is deterministic regardless of width.
    ParallelMap {
        input: Box<Expr>,
        func: Lambda,
        #[serde(default)]
        concurrency: Option<u32>,
    },
    /// Map `func` over a list with an explicit in-flight `bound`.
    BoundedMap {
        input: Box<Expr>,
        func: Lambda,
        bound: u32,
    },
    /// Keep the elements for which `pred` is truthy.
    Filter { input: Box<Expr>, pred: Lambda },
    /// Sort by `key`, optionally keep the top `limit`.
    Rank {
        input: Box<Expr>,
        key: Lambda,
        order: Order,
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Group elements by `key`; yields a sorted list of `{key, items}` records.
    Group { input: Box<Expr>, key: Lambda },
    /// Join two lists on matching keys; yields `{left, right}` records.
    Join {
        left: Box<Expr>,
        right: Box<Expr>,
        left_key: Lambda,
        right_key: Lambda,
        kind: JoinKind,
    },
    /// Fold a list into a single value. `acc` and `item` are the accumulator and
    /// element variable names bound while evaluating `body`.
    Reduce {
        input: Box<Expr>,
        init: Box<Expr>,
        acc: String,
        item: String,
        body: Box<Expr>,
    },
    /// Take one page of a list.
    Paginate {
        input: Box<Expr>,
        page_size: usize,
        page: usize,
    },
    /// Evaluate `body`, retrying on a retryable handle error per `policy`.
    RetryWithPolicy {
        body: Box<Expr>,
        policy: RetryPolicy,
    },
    /// Validate a value against a schema; returns the value or a schema error.
    SchemaValidate {
        input: Box<Expr>,
        schema: SchemaSpec,
    },
    /// Remove duplicates. With `key`, dedup by the key; without, by whole value.
    /// First occurrence wins; order is preserved.
    Dedup {
        input: Box<Expr>,
        #[serde(default)]
        key: Option<Lambda>,
    },
    /// Deterministically sample up to `k` elements using the seeded rng.
    Sample { input: Box<Expr>, k: usize },
    /// Spill a large value to an artifact and return a reference. Enforces the
    /// per-artifact byte budget.
    SpillToArtifact {
        input: Box<Expr>,
        name: String,
    },
    /// Map `func` over records while preserving each source record's citations
    /// onto the corresponding output record (merged, deduplicated).
    CitationPreservation {
        input: Box<Expr>,
        func: Lambda,
    },
}

/// An expression node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "expr")]
pub enum Expr {
    /// A literal value.
    Lit { value: Value },
    /// A variable reference (a lambda / let binding).
    Var { name: String },
    /// Follow a key path into a value.
    Field { base: Box<Expr>, path: Vec<String> },
    /// Build a list from element expressions.
    List { items: Vec<Expr> },
    /// Build a map from key / value-expression pairs.
    MapLit { entries: Vec<(String, Expr)> },
    /// Call a granted read handle with an argument value.
    Handle { name: HandleName, args: Box<Expr> },
    /// Bind `name` to `value` inside `body`.
    Let {
        name: String,
        value: Box<Expr>,
        body: Box<Expr>,
    },
    /// Choose a branch by a truthy condition.
    If {
        cond: Box<Expr>,
        then: Box<Expr>,
        otherwise: Box<Expr>,
    },
    /// A binary operation.
    BinOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// Logical negation of a truthy value.
    Not { value: Box<Expr> },
    /// Prepare (do not execute) a write. The argument is a map describing the
    /// proposal; see `interp` for the accepted shape.
    ProposeWrite { spec: Box<Expr> },
    /// Apply a built-in operator.
    Op { operator: Box<Operator> },
}

// -- ergonomic constructors ---------------------------------------------------

impl Expr {
    pub fn lit(value: impl Into<Value>) -> Expr {
        Expr::Lit {
            value: value.into(),
        }
    }

    pub fn var(name: impl Into<String>) -> Expr {
        Expr::Var { name: name.into() }
    }

    pub fn field(base: Expr, path: impl IntoIterator<Item = &'static str>) -> Expr {
        Expr::Field {
            base: Box::new(base),
            path: path.into_iter().map(str::to_string).collect(),
        }
    }

    pub fn list(items: impl IntoIterator<Item = Expr>) -> Expr {
        Expr::List {
            items: items.into_iter().collect(),
        }
    }

    pub fn map_lit(entries: impl IntoIterator<Item = (&'static str, Expr)>) -> Expr {
        Expr::MapLit {
            entries: entries
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        }
    }

    pub fn handle(name: HandleName, args: Expr) -> Expr {
        Expr::Handle {
            name,
            args: Box::new(args),
        }
    }

    pub fn let_(name: impl Into<String>, value: Expr, body: Expr) -> Expr {
        Expr::Let {
            name: name.into(),
            value: Box::new(value),
            body: Box::new(body),
        }
    }

    pub fn if_(cond: Expr, then: Expr, otherwise: Expr) -> Expr {
        Expr::If {
            cond: Box::new(cond),
            then: Box::new(then),
            otherwise: Box::new(otherwise),
        }
    }

    pub fn bin(op: BinOp, lhs: Expr, rhs: Expr) -> Expr {
        Expr::BinOp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn not(value: Expr) -> Expr {
        Expr::Not {
            value: Box::new(value),
        }
    }

    pub fn propose_write(spec: Expr) -> Expr {
        Expr::ProposeWrite {
            spec: Box::new(spec),
        }
    }

    pub fn op(operator: Operator) -> Expr {
        Expr::Op {
            operator: Box::new(operator),
        }
    }
}

/// A complete program: the root expression plus the deterministic seed used by
/// `sample` and the clock start. Programs are pure data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub root: Expr,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub clock_start_ms: u64,
}

impl Program {
    pub fn new(root: Expr) -> Self {
        Self {
            root,
            seed: 0,
            clock_start_ms: 0,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}
