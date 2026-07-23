//! hide-program-runtime: the sandboxed programmatic tool runtime (Bible Book V,
//! sec 18-19).
//!
//! A HIDE agent does not only call one tool at a time - it can run a small
//! *program* that fans out over read handles, filters, ranks, joins, and
//! reduces the results into a structured answer, all inside a sandbox. This
//! crate is that sandbox and its evaluator.
//!
//! # The sandbox in one paragraph
//!
//! A [`Program`] is a tree of [`Expr`] nodes (pure data - it serializes to
//! JSON). It is evaluated by a deterministic in-crate interpreter that has NO
//! ambient authority. The only way a program reaches the outside world is by
//! calling a handle from the closed, read-only [`HandleName`] set, and only if
//! the host granted it: `search.text`, `search.symbol`, `index.references`,
//! `file.read`, `git.diff`, `git.log`, `diagnostic.list`, `test.result.read`,
//! `artifact.read`, `mcp.readonly`. There is no filesystem-write,
//! subprocess, network-egress, environment, or credential handle to name, so a
//! program cannot express one. The clock is virtual and the rng is seeded, so a
//! run is reproducible byte for byte.
//!
//! # Write separation
//!
//! A program may *prepare* a mutation but never commit one. Building a write
//! (an edit, a shell command, a network call, an external mutation) produces a
//! typed [`WriteProposal`] that is collected and returned in [`RunOutput`]; the
//! runtime executes none of them. The proposal travels the normal action plane
//! where real approval + execution live, outside this crate.
//!
//! # Built-in operators
//!
//! `parallel_map`, `bounded_map`, `filter`, `rank`, `group`, `join`, `reduce`,
//! `pagination`, `retry_with_policy`, `schema_validate`, `dedup`, `sample`,
//! `spill_to_artifact`, and `citation_preservation` are provided as [`Operator`]
//! nodes. Iteration exists only through these bounded operators, which keeps
//! evaluation total and every dimension metered.
//!
//! # Enforced limits
//!
//! [`Limits`] bounds instruction count, virtual wall time, peak memory, output
//! bytes, tool calls, map concurrency, per-artifact bytes, and recursion depth.
//! Exhausting any one raises a typed [`RuntimeError::LimitExceeded`] carrying the
//! [`LimitKind`].
//!
//! # Model-free
//!
//! This crate is entirely model-free (RIP doctrine). It evaluates data-shaped
//! programs over host-supplied read handles and proves itself with deterministic
//! tests over fixtures. It never runs a model, opens a socket, spawns a process,
//! or touches the filesystem. Binding a real model to author these programs is a
//! job for a model-bearing layer and is out of scope here; see
//! `DEFERRED_MODEL_REQUIRED` below.
//!
//! DEFERRED_MODEL_REQUIRED: nothing in this crate synthesizes a program from a
//! natural-language goal. Program authoring by a model is deferred; the runtime
//! only *executes* programs it is handed, deterministically.
//!
//! # Example program
//!
//! Fan out a search, project + rank the hits while preserving their citations,
//! and prepare (but do not execute) a follow-up edit.
//!
//! ```
//! use hide_program_runtime::{
//!     run, BinOp, Expr, HandleGrants, HandleName, Lambda, Limits, Operator, Order,
//!     Program, Value, FnHost, map_of, Citation,
//! };
//!
//! // A host that answers `search.text` with two cited rows. Read-only.
//! let host = FnHost::new(|handle, _args, _attempt| {
//!     assert_eq!(handle, HandleName::SearchText);
//!     let row = |path: &str, score: i64| {
//!         map_of([("path", Value::from(path)), ("score", Value::from(score))])
//!             .with_merged_citations(&[Citation::new("search.text").with_locator(path)])
//!     };
//!     Ok(Value::List(vec![row("a.rs", 2), row("b.rs", 9)]))
//! });
//!
//! // Program: search -> keep the path field (preserving citations) -> rank by
//! // score desc -> also stage an edit proposal, and return both.
//! let hits = Expr::handle(HandleName::SearchText, Expr::lit("needle"));
//! let projected = Expr::op(Operator::CitationPreservation {
//!     input: Box::new(hits),
//!     func: Lambda::new(
//!         "h",
//!         Expr::map_lit([("path", Expr::field(Expr::var("h"), ["path"]))]),
//!     ),
//! });
//! let ranked = Expr::op(Operator::Rank {
//!     input: Box::new(Expr::handle(HandleName::SearchText, Expr::lit("needle"))),
//!     key: Lambda::new("h", Expr::field(Expr::var("h"), ["score"])),
//!     order: Order::Desc,
//!     limit: Some(1),
//! });
//! let proposal = Expr::propose_write(Expr::map_lit([
//!     ("kind", Expr::lit("edit")),
//!     ("summary", Expr::lit("rename symbol in top hit")),
//! ]));
//! let root = Expr::map_lit([
//!     ("projected", projected),
//!     ("top", ranked),
//!     ("staged", proposal),
//! ]);
//!
//! let program = Program::new(root);
//! let out = run(&program, &host, &HandleGrants::of([HandleName::SearchText]), Limits::strict())
//!     .expect("program runs");
//!
//! // Citations survived the projection.
//! let projected = out.value.get_path(&["projected".into()]).unwrap();
//! let first = &projected.as_list().unwrap()[0];
//! assert_eq!(first.citations().len(), 1);
//!
//! // The edit was prepared, not executed: one proposal, and it went nowhere.
//! assert_eq!(out.proposals.len(), 1);
//! assert_eq!(out.proposals[0].summary, "rename symbol in top hit");
//! ```

pub mod ast;
pub mod error;
pub mod handles;
pub mod interp;
pub mod limits;
pub mod proposal;
pub mod value;

pub use ast::{
    BinOp, Expr, JoinKind, Lambda, Operator, Order, Program, RetryPolicy, SchemaField, SchemaSpec,
};
pub use error::{HandleError, LimitKind, Result, RuntimeError};
pub use handles::{DenyAllHost, FnHost, HandleGrants, HandleName, HostHandles};
pub use interp::{run, Artifact, RunOutput};
pub use limits::{Limits, Usage};
pub use proposal::{WriteKind, WriteProposal};
pub use value::{map_of, Citation, Value, CITATIONS_KEY};
