use crate::verify::oracle::{Oracle, Verdict, VerdictStatus, VerificationInput};
use futures::future::BoxFuture;
use hide_core::Result;

pub struct StubDeterministicOracle {
    pub name: String,
}

impl Oracle for StubDeterministicOracle {
    fn name(&self) -> &str {
        &self.name
    }

    fn verify<'a>(&'a self, _input: &'a VerificationInput) -> BoxFuture<'a, Result<Verdict>> {
        Box::pin(async move {
            Ok(Verdict {
                status: VerdictStatus::Pass,
                score: 1.0,
                oracle: self.name.clone(),
                detail: "stub deterministic oracle".to_string(),
            })
        })
    }
}
