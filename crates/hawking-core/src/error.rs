use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("gguf: {0}")]
    Gguf(String),

    #[error("metal: {0}")]
    Metal(String),

    #[error("model: {0}")]
    Model(String),

    #[error("kernel: {0}")]
    Kernel(String),

    #[error("not yet implemented: {0}")]
    Unimplemented(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<hawking_speculate::Error> for Error {
    fn from(e: hawking_speculate::Error) -> Self {
        match e {
            hawking_speculate::Error::Io(io) => Error::Io(io),
            hawking_speculate::Error::Model(s) => Error::Model(s),
            hawking_speculate::Error::Unimplemented(s) => Error::Unimplemented(s),
        }
    }
}
