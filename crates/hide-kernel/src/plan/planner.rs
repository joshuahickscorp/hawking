use crate::plan::schema::Plan;
use futures::future::BoxFuture;
use hide_core::Result;

pub trait Planner: Send + Sync {
    fn synthesize<'a>(&'a self, objective: &'a str) -> BoxFuture<'a, Result<Plan>>;
}

#[derive(Default)]
pub struct StubPlanner;

impl Planner for StubPlanner {
    fn synthesize<'a>(&'a self, objective: &'a str) -> BoxFuture<'a, Result<Plan>> {
        Box::pin(async move { Ok(Plan::single_step("Stub plan", objective)) })
    }
}
